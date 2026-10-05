/* rad_block_hc.h -- the GATED RESIDUAL, which is what Qwen4-Exp has instead of a pre-norm and a
 * residual add.
 *
 * ============================== THE RESIDUAL STREAM IS `hc` TIMES AS WIDE ==============================
 *
 * hidden_size 2560 at hc_count 4 is a 10240-wide stream. It is entered as FOUR IDENTICAL COPIES of
 * the embedding, carried through all 48 layers, and collapsed once at the end by a connection with
 * no write half. Every block -- attention or delta net, then the feed-forward, so 96 of them --
 * reads a 2560 mix of the four streams and writes its output back into all four, and both halves
 * are data-dependent:
 *
 *     read    N      = grouped_rmsnorm(h)    one rsqrt per 2560 stream, one 10240 gain, `1 + w`
 *             g      = sigmoid(mix_up @ silu(mix_down @ N / hc))
 *             x      = mean over the four streams of g * N            -> the block's input
 *             inj[s] = 2 * sigmoid(inject_w[s] . N / hc)              -> four scalars in (0, 2)
 *     write   h[s]  += inj[s] * y                                     for each stream s
 *
 * docs/OPS.md's `hc_read` / `hc_write` rows carry the same statement and name the three choices a
 * second implementation has to match (the mix is a MEAN; each gate argument has its own `/ hc`;
 * the write adds into the UNNORMED stream).
 *
 * ============================== WHY IT IS A BLOCK AND NOT TWO CALLS ==============================
 *
 * Because the two halves share a weight set and a buffer set, and because the pairing is the
 * invariant: a read whose `inj` nothing consumes is a bug, and a write whose `inj` came from a
 * different read is a worse one. One struct declares both, holds the four weights once, and gives
 * the plugin a `read()` and a `write()` that cannot be mismatched.
 *
 * ============================== WHAT IT COSTS, AND WHERE THAT GOES ==============================
 *
 * The arithmetic is nothing: 2 x 10240 x 320 plus 4 x 10240 is 6.6M MACs a token against the 5.9M
 * of one expert pair. The LAUNCHES are the cost. A kernel costs a dispatch whatever it computes,
 * and there are 96 connections a step, so every extra op in a connection is 96 more dispatches --
 * a real share of a decode budget measured in tens of milliseconds.
 *
 * That is why the read emits the fp8 form of `x` itself rather than leaving a `quant_act_fp8`
 * behind it, and it is why the write is a candidate to be FOLDED INTO THE PRODUCING BLOCK'S
 * EPILOGUE later (this tree's own rule: fold the quantiser into the producer, and quantise the
 * bf16 that was written). Neither the read's five ops nor the write's one are on the table as
 * separate launches.
 *
 * ============================== WHAT IS NOT SHARDED ==============================
 *
 * Nothing here. The residual stream is replicated on every rank (spec §9) and so is every weight
 * in this file: 6.7 MB a connection at bf16, 1.3 GB over the model, which is the price of not
 * putting a collective inside the connection. A sharded low-rank gate would need an all-reduce
 * between `mix_down` and `mix_up` -- 96 more collectives a step, on 1280 bytes each, and a decode
 * all-reduce is mostly fixed cost whatever the message.
 */
#ifndef RAD_BLOCK_HC_H
#define RAD_BLOCK_HC_H

#include "rad_fp8.h"

namespace rad {
namespace arch {

/* THE WIDE STREAM'S FIRST VALUE: hc identical copies of the embedding.
 *
 * One launch for the whole step, and it is its own op rather than part of the embedding lookup
 * because that op's weight is [vocab, n_embd] and its output width is n_embd -- a wide output
 * would be a different weight shape. The copies are identical here and diverge at the first
 * connection, since every block writes into all hc streams with a DIFFERENT per-branch gain. */
struct HcEnter {
    Geom    g{};
    int64_t hc = 0;
    rad_buf x = 0, h = 0;
    rad_op  op = 0;

    int declare(RadBuilder* b, const Geom& geom, int64_t hc_, rad_buf x_, rad_buf h_) {
        g = geom; hc = hc_; x = x_; h = h_;
        if (hc <= 1 || !x || !h) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "hc_enter",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                              RAD_INT("hc", hc), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {x}, {h});
        return op ? RAD_OK : RAD_E_INVAL;
    }

    void step(RadCtx* c_, int64_t T) const {
        RAD_ISSUE_N(c_, op, T, brows(x, T), brows(h, T));
    }
};

struct HyperConn {
    struct Config {
        int64_t hc      = 0;    /* streams; the reference refuses hc <= 1 and so does this */
        int64_t lowrank = 0;    /* the read gate's waist */
        /* Whether this connection has a WRITE half. The final mixer does not: it collapses the
         * wide stream into the one hidden state the lm_head reads and nothing follows it. */
        int     inject  = 1;
        /* Whether the read also emits the block input's fp8 codes. The trunk connections all do;
         * the mixer does not, because its consumer is the lm_head, which reads a bf16 input. */
        int     quant   = 1;
        /* Whether the WRITE also carries the all-reduce of the block's output, which the block in
         * front then skips on every step takes_ar() answers yes to. The block's partial sum and the
         * write's input are the same buffer, so the reduction moves into the write's kernel and
         * the pair's second launch disappears. */
        int     reduce  = 0;
        /* Whether the read also emits the HADAMARD-ROTATED fp8 codes of the block input, for a
         * consumer whose weight carries the rotation (MoeFP8's rotated expert plane). They ride in
         * the plain codes' finisher, so this needs `quant`, and the consumer then declares no
         * quantiser of its own -- one dependent launch fewer a connection. */
        int     rotate  = 0;
        /* Whether those codes are INT8 (the input's q8/s8 pair) rather than E4M3 (q/s): a model
         * whose trunk linears are int8 (LinearFP8::i8) reads them there, so the read writing them
         * is one quantiser launch fewer a consumer. The rotated twin stays E4M3 either way -- it
         * is the routed experts' operand. */
        int     codes_i8 = 0;
    };

    struct Wire {
        rad_buf h   = 0;    /* [max_tok, hc * n_embd]  THE WIDE STREAM, in place */
        ActFP8  x{};        /* [max_tok, n_embd]       the block input; q/s unused if !quant */
        rad_buf inj = 0;    /* [max_tok, hc]           the per-branch write gains */
        rad_buf rq  = 0;    /* [max_tok, n_embd]       the rotated codes, when `rotate` */
        rad_buf rs  = 0;    /* [max_tok, n_embd/128]   and their scales */
    };

    struct Src {
        const char* norm      = nullptr;   /* hc_norm.weight,                [hc*n_embd] */
        const char* mix_down  = nullptr;   /* input_mix_weight_down.weight,  [lowrank, hc*n_embd] */
        const char* mix_up    = nullptr;   /* input_mix_weight_up.weight,    [hc*n_embd, lowrank] */
        const char* inject_w  = nullptr;   /* block_inject_weight.weight,    [hc, hc*n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};

    rad_weight w_norm = 0, w_down = 0, w_up = 0, w_inj = 0;
    /* Whether the two mixing matrices are E4M3 rows with a scale a group of each row rather than
     * bf16 -- OPS.md's `mix` -- which is their encoding's to say, read at declare. */
    bool       mix_e4m3 = false;
    rad_op     op_read = 0, op_write = 0;
    /* The write with the all-reduce folded in; 0 when not asked for or when no kernel serves it.
     * `op_ar_write6` is the same write on the ROTATED 6-BIT wire, declared only where a step can
     * reach that wire -- an install that accepts a lossy payload -- and issued on exactly the steps
     * ar_wire_is_exact answers no to: at a prefill chunk that is every all-reduce the trunk makes,
     * and the payload is 2.6x smaller than the exact one. */
    rad_op     op_ar_write = 0;
    rad_op     op_ar_write6 = 0;
    /* AND WITH THE MoE GATHER FOLDED IN FRONT OF THAT: the block behind this write leaves its
     * routed experts' rows unsummed, and the reduction's message is their sum. Declared by
     * declare_gather() after the block, whose buffers it reads; `gather_rows` is the bound it serves
     * and 0 when it was not declared or no kernel serves it (see gather_taken). */
    struct GatherSrc {
        rad_buf ye = 0, ew = 0, sorted = 0, sh = 0, sg = 0;   /* sh and sg: the shared arm, or 0 */
        int64_t top_k = 0;
        int64_t rows = 0;                                      /* one pass of the block, at most */
    };
    GatherSrc  gsrc{};
    rad_op     op_ar_gather_write = 0, op_ar_gather_write6 = 0;
    int64_t    gather_rows = 0, gather_rows6 = 0;

    int64_t hn() const { return c.hc * g.n_embd; }
    /* THE REDUCING WRITE THIS STEP ISSUES, on the wire its message goes out on; 0 where that wire
     * has no fused form, and the block's own collective then runs in front of the plain write. At
     * four ranks there is no rotated form, so a step ar_exact_write_takes declines gets 0. */
    rad_op ar_write(int64_t T) const {
        return ar_exact_write_takes(g, T) ? op_ar_write : op_ar_write6;
    }
    /* WHETHER THIS STEP'S WRITE REDUCES THE BLOCK'S OUTPUT. The block in front asks the same
     * question of the same T -- ar_taken with ar_take() -- before skipping its own collective. */
    bool takes_ar(int64_t T) const { return ar_write(T) != 0; }
    /* What that block is told: which wires this write takes. The rotated form is declared only
     * beside the exact one, so the two together take every message. */
    int ar_take() const { return op_ar_write6 ? kArOutBoth : kArOutExact; }
    bool takes_gather(int64_t T) const {
        return takes_ar(T) && gather_taken(g, T, gather_rows, gather_rows6);
    }

    /* `name` is the declared prefix -- "blk.7.attn_hc" -- and `src` the checkpoint side. */
    int declare(RadBuilder* b, Names& nm, const Geom& geom, const char* name,
                const Config& cfg, const Wire& wire, const Src& src);

    int  declare_gather(RadBuilder* b, const GatherSrc& src);

    void read (RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const;
    void write(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const;
};

inline int HyperConn::declare(RadBuilder* b, Names& nm, const Geom& geom, const char* name,
                              const Config& cfg, const Wire& wire, const Src& src) {
    g = geom; c = cfg; w = wire;
    if (c.hc <= 1 || c.lowrank <= 0 || g.n_embd <= 0) {
        fprintf(stderr, "radiance: %s: hc = %lld, lowrank = %lld -- a gated residual needs hc > 1 "
                        "and a positive waist\n",
                name, (long long)c.hc, (long long)c.lowrank);
        return RAD_E_INVAL;
    }
    if (!src.norm || !src.mix_down || !src.mix_up) return RAD_E_INVAL;
    /* An inject WEIGHT with no `inj` buffer, or the reverse, is a connection whose two halves
     * disagree about whether there is a write. Refused here rather than discovered as an
     * all-rejected draft three stages later. */
    if (c.inject && (!src.inject_w || !w.inj)) {
        fprintf(stderr, "radiance: %s: inject is on but %s is missing\n", name,
                src.inject_w ? "the inj buffer" : "block_inject_weight");
        return RAD_E_INVAL;
    }
    if (!c.inject && (src.inject_w || w.inj)) {
        fprintf(stderr, "radiance: %s: inject is off, so neither block_inject_weight nor an inj "
                        "buffer may be passed\n", name);
        return RAD_E_INVAL;
    }
    if (c.quant && (c.codes_i8 ? (!w.x.q8 || !w.x.s8) : (!w.x.q || !w.x.s))) return RAD_E_INVAL;
    if (c.rotate && (!c.quant || !w.rq || !w.rs)) {
        fprintf(stderr, "radiance: %s: the rotated codes need the plain ones (quant) and the "
                        "rq/rs pair on the wire\n", name);
        return RAD_E_INVAL;
    }
    if (!w.h || !w.x.x) return RAD_E_INVAL;

    const int64_t HN = hn();

    /* The name maps first: they are how the model is asked what each weight is. */
    const char* nn = nm.f("%s_norm.weight", name);
    const char* dn = nm.f("%s_down.weight", name);
    const char* un = nm.f("%s_up.weight", name);
    const char* in = nm.f("%s_inject.weight", name);
    RAD_ARCH_TRY(map_copy(b, nn, src.norm));
    RAD_ARCH_TRY(map_copy(b, dn, src.mix_down));
    RAD_ARCH_TRY(map_copy(b, un, src.mix_up));
    if (c.inject) RAD_ARCH_TRY(map_copy(b, in, src.inject_w));

    /* THE MIXING MATRICES ARE WHATEVER THEIR ENCODING IS: bf16, or E4M3 rows with a scale a
     * group of each row -- lowrank 320 is 2.5 scale blocks, so a 128x128 tile cannot cover them.
     * Each is declared over every plane it has, at the codes' dtype, and the read's kernel takes
     * codes and scales together. The two are one format or the read has no kernel. */
    const uint32_t mdt = weight_codes_dtype(b, dn, RAD_BF16);
    if (weight_codes_dtype(b, un, RAD_BF16) != mdt || (mdt != RAD_BF16 && mdt != RAD_F8E4M3)) {
        fprintf(stderr, "radiance: %s: the mixing matrices are %s and %s; a connection reads both "
                        "as bf16 or both as E4M3 rows\n", name,
                rad_dtype_name(mdt), rad_dtype_name(weight_codes_dtype(b, un, RAD_BF16)));
        return RAD_E_DTYPE;
    }
    mix_e4m3 = mdt == RAD_F8E4M3;

    /* F32 like every other norm gain in this tree: 10240 numbers read once a connection, and the
     * norm's arithmetic is f32 regardless, so narrowing them buys nothing. */
    w_norm = decl_w(b, nn, RAD_F32, {HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_down = decl_w(b, dn, mdt, {c.lowrank, HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                    grp_model());
    w_up   = decl_w(b, un, mdt, {HN, c.lowrank}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                    grp_model());
    if (c.inject)
        w_inj = decl_w(b, in, RAD_BF16, {c.hc, HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                       grp_model());
    if (!w_norm || !w_down || !w_up || (c.inject && !w_inj)) return RAD_E_INVAL;

    /* `inject` and `group` are declared even when off, with the value 0, because a kernel row
     * bands on them: a fused read that writes four gates is a different kernel from one that does
     * not, and the resolver needs to be able to say so. An ABSENT key fails a constraint, so
     * leaving them out would make both rows unselectable rather than one. */
    /* RAD_WEIGHTS expands to TWO arguments -- the array and its length -- so it cannot sit in a
     * ternary, and the four-or-three choice is made by building the array. The absent inject
     * weight is a SHORTER list and not a zero handle: rad_plugin.h's absent-operand rule is about
     * the launch, but a declared weight of 0 is a declaration failure here. */
    const rad_weight wts[4] = { w_norm, w_down, w_up, w_inj };
    const int n_wts = c.inject ? 4 : 3;

    /* The write set is positional and the optional OUTPUTS are the tail, so an absent one shortens
     * the list exactly as an absent weight does. The read's outputs are x, inj?, q?, scale? and
     * both switches drop a suffix, never a middle -- which is why the schema puts inj before the
     * quantiser's pair rather than after. */
    rad_buf wr[6] = { w.x.x, 0, 0, 0, 0, 0 };
    int n_wr = 1;
    if (c.inject) wr[n_wr++] = w.inj;
    if (c.quant)  { wr[n_wr++] = c.codes_i8 ? w.x.q8 : w.x.q; wr[n_wr++] = c.codes_i8 ? w.x.s8 : w.x.s; }
    if (c.rotate) { wr[n_wr++] = w.rq;  wr[n_wr++] = w.rs; }

    op_read = RAD_OP(b, "hc_read",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                             RAD_INT("hc", c.hc), RAD_INT("lowrank", c.lowrank),
                             RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                             RAD_F64("wadd", g.wadd), RAD_INT("inject", c.inject ? 1 : 0),
                             RAD_INT("group", c.quant ? RAD_FP8_BLOCK : 0),
                             RAD_INT("rotate", c.rotate ? 1 : 0),
                             RAD_STR("mix", mix_e4m3 ? "e4m3" : "bf16")),
                  wts, n_wts);
    if (!op_read) return RAD_E_INVAL;
    {
        const rad_buf rd[1] = { w.h };
        rad_op_reads (b, op_read, rd, 1);
        rad_op_writes(b, op_read, wr, n_wr);
    }

    if (c.inject) {
        op_write = rw(b, RAD_OP(b, "hc_write",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_INT("hc", c.hc), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.x.x, w.inj, w.h}, {w.h});
        if (!op_write) return RAD_E_INVAL;
        /* A handle that did not resolve is 0 and not an error: a backend without the fused kernel
         * keeps the pair, and takes_ar() then answers no on every step. At four ranks only the
         * exact form exists, and it takes a step only where ar_wide_write_exact says the
         * standalone collective would have been the exact one-shot. */
        if (c.reduce && (g.world == 2 || g.world == 4))
            op_ar_write = rw(b, RAD_OP(b, "ar_hc_write",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                           RAD_INT("hc", c.hc), RAD_INT("world_size", g.world),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS),
                             {w.x.x, w.inj, w.h}, {w.x.x, w.h});
        if (op_ar_write && !g.wire_exact && g.world == 2)
            op_ar_write6 = rw(b, RAD_OP(b, "ar_hc_write",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                            RAD_INT("hc", c.hc), RAD_INT("world_size", g.world),
                                            RAD_STR("dtype", g.dtype), RAD_INT("wire", 6)),
                                 RAD_NOWEIGHTS),
                              {w.x.x, w.inj, w.h}, {w.x.x, w.h});
    }
    return RAD_OK;
}

inline int HyperConn::declare_gather(RadBuilder* b, const GatherSrc& src) {
    if (!op_ar_write || src.top_k <= 0 || !src.ye || !src.ew || !src.sorted) return RAD_OK;
    /* Two ranks, and four on the exact wire: there the fused write exists in its exact form only,
     * so the gather does too -- op_ar_write6 is 0 at four and the rotated form below is never
     * declared. Any other width keeps its gather and both bounds stay 0. */
    if (g.world != 2 && g.world != 4) return RAD_OK;
    const bool shared = src.sh && src.sg;
    gsrc = src;
    /* EVERY ROW COUNT ONE PASS CARRIES, prefill chunks included: the kernel reads its slots from
     * an LDS table at a decode step and from an inverse table at a chunk, where the gather's reads
     * then run beside the reduction's push instead of in a launch of their own. */
    int64_t rows = src.rows;
    if (rows > g.max_tok) rows = g.max_tok;
    if (rows < 1) return RAD_OK;
    const rad_buf rd[8] = { src.ye, src.ew, src.sorted, w.x.x, w.inj, w.h, src.sh, src.sg };
    const rad_buf wr[2] = { w.x.x, w.h };
    /* One op a wire, as the write itself: the rotated form beside the rotated write only. A handle
     * that did not resolve is 0 and not an error: the block keeps its gather and this write keeps
     * the reduction alone, and that wire's bound stays 0 so both say so. */
    op_ar_gather_write = RAD_OP(b, "ar_gather_hc_write",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                           RAD_INT("hc", c.hc), RAD_INT("top_k", src.top_k),
                                           RAD_INT("world_size", g.world),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_STR("act", shared ? "sigmoid" : "none")),
                                RAD_NOWEIGHTS);
    if (op_ar_gather_write) {
        RAD_ARCH_TRY(rad_op_reads (b, op_ar_gather_write, rd, shared ? 8 : 6));
        RAD_ARCH_TRY(rad_op_writes(b, op_ar_gather_write, wr, 2));
        gather_rows = rows;
    }
    if (op_ar_write6) {
        op_ar_gather_write6 = RAD_OP(b, "ar_gather_hc_write",
                                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                                RAD_INT("hc", c.hc), RAD_INT("top_k", src.top_k),
                                                RAD_INT("world_size", g.world),
                                                RAD_STR("dtype", g.dtype),
                                                RAD_STR("act", shared ? "sigmoid" : "none"),
                                                RAD_INT("wire", 6)),
                                     RAD_NOWEIGHTS);
        if (op_ar_gather_write6) {
            RAD_ARCH_TRY(rad_op_reads (b, op_ar_gather_write6, rd, shared ? 8 : 6));
            RAD_ARCH_TRY(rad_op_writes(b, op_ar_gather_write6, wr, 2));
            gather_rows6 = rows;
        }
    }
    return RAD_OK;
}

inline void HyperConn::read(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    (void)T;
    const int64_t HN = hn();
    RAD_ISSUE_N(c_, op_read, rows,
                brow_slice(w.h, r0, rows, HN), RAD_W(w_norm), RAD_W(w_down), RAD_W(w_up),
                c.inject ? RAD_W(w_inj) : RAD_NONE,
                brow_slice(w.x.x, r0, rows, g.n_embd),
                c.inject ? brow_slice(w.inj, r0, rows, c.hc) : RAD_NONE,
                c.quant  ? brow_slice(c.codes_i8 ? w.x.q8 : w.x.q, r0, rows, g.n_embd) : RAD_NONE,
                c.quant  ? brow_slice(c.codes_i8 ? w.x.s8 : w.x.s, r0, rows,
                                      fp8_blocks(g.n_embd)) : RAD_NONE,
                c.rotate ? brow_slice(w.rq, r0, rows, g.n_embd) : RAD_NONE,
                c.rotate ? brow_slice(w.rs, r0, rows, fp8_blocks(g.n_embd)) : RAD_NONE);
}

/* `y` is the block's output, and it is `w.x.x` -- the same buffer the read wrote the block's INPUT
 * into. Every block in this tree leaves its delta in the activation its norm produced, so the
 * write reads it there; the input is dead by then. */
inline void HyperConn::write(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    if (!op_write) return;
    if (takes_gather(T)) {
        const int64_t k = gsrc.top_k;
        const bool shared = gsrc.sh && gsrc.sg;
        const rad_op op = ar_exact_write_takes(g, T) ? op_ar_gather_write : op_ar_gather_write6;
        RAD_ISSUE_N(c_, op, rows,
                    brows(gsrc.ye, rows * k), brow_slice(gsrc.ew, r0, rows, k),
                    brows(gsrc.sorted, rows * k),
                    shared ? brow_slice(gsrc.sh, r0, rows, g.n_embd) : RAD_NONE,
                    shared ? brow_slice(gsrc.sg, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.x.x, r0, rows, g.n_embd),
                    brow_slice(w.inj, r0, rows, c.hc),
                    brow_slice(w.h, r0, rows, hn()));
        return;
    }
    RAD_ISSUE_N(c_, takes_ar(T) ? ar_write(T) : op_write, rows,
                brow_slice(w.x.x, r0, rows, g.n_embd),
                brow_slice(w.inj, r0, rows, c.hc),
                brow_slice(w.h, r0, rows, hn()));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_HC_H */
