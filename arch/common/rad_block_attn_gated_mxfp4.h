/* rad_block_attn_gated_mxfp4.h -- the gated GQA block with OCP-MX projections.
 *
 * THE ARCHITECTURE IS rad_block_attn_gated.h's AND IS DOCUMENTED THERE, and this file is the
 * OCP-MX twin of rad_block_attn_gated_fp8.h, which documents the block-scaled-FP8 form of it: the
 * interleaved query/gate projection and why it is declared rank 3, the partial mrope and what is
 * not expressible about it, why k and v are not fused, why the fused qk_norm_rope_gate probe is
 * withdrawn, and why the gate is a sigmoid. None of that changes with the weight format and none
 * of it is repeated here.
 *
 * What differs from the fp8 twin:
 *
 *   - q|gate, k, v and o are LinearMX (rad_mx.h): an OCP-MX codebook -- fp4 e2m1, fp8 e4m3, or
 *     fp6 e2m3 -- with an E8M0 exponent per group of 32 along K, through `gemm_nt_q` at the
 *     codebook's dtype (mxfp4a8 / mxfp8a8 / mxfp6a8). For fp4 the exponent plane is two views of
 *     one logical weight (the block scale and the per-row reference) and the LinearMX is three
 *     operands of one GEMM; fp8 and fp6 codes self-scale, declare no reference, and are two.
 *   - two `quant_act_fp8_row` producers a layer, not four: the three input projections all read the
 *     post-norm hidden state, so one NormQuantMX feeds all three, and the output projection's gate
 *     stage gets its own because it reads the gated attention output. THE GATE STAGE IS THREE OPS
 *     HERE, NOT ONE: `sigmoid` -> `mul` -> `quant_act_fp8_row`, the unfused pair the bf16 arm runs
 *     plus the quantiser shared with every other MX producer. The fp8 twin's fused `gate_quant_fp8`
 *     cannot serve this block: it is declared at `M = T*n_head` and its per-128 scale flattening
 *     argument does not carry to the MX per-row quantiser (see the gate stage below). A per-row
 *     `gate_quant_fp8_row` form exists and would serve at `M = T`, but this arm does not wire it.
 *     The activation scale is ONE f32 PER ROW, `[M, 1]`.
 *   - everything between the GEMMs -- the per-head norms, rope, kv_store, attn_paged, the sigmoid
 *     and the gate multiply -- is UNCHANGED and still bf16.
 *   - THE KV CACHE'S WIDTH IS `g.kv_dtype` AND NOT `g.dtype`, exactly as the fp8 twin states:
 *     `dtype` follows the WEIGHTS; `kv_dtype` follows the CACHE, which is a deployment choice.
 *
 * The q|gate projection is [2*n_head*head_dim, n_embd] = [12288, 5120] at 27B, and its E8M0 plane
 * is [12288, 160] -- rows are the weight's ROWS, because `block[0] == 1`. Row-sharding by head
 * keeps whole plane rows together trivially, and 2*head_dim is 512, a multiple of the kernel's
 * 16-row N tile, so a rank's slice of the head range has N in 16s. LinearMX checks that rather
 * than trusting it.
 *
 * ONE ADDITIONAL DELIBERATE SIMPLIFICATION beyond the two the task names, marked at its seam below:
 * the fp8 twin's `op_attn_gq` (the sparse path's `attn_paged_gate_quant`, attention and the gate
 * quantiser as one merge) has NO per-row MX form registered, so it is not declared and the sparse
 * path takes the separate three-op gate arm (`sigmoid` -> `mul` -> `quant_act_fp8_row`) instead.
 * This is correctness-preserving -- the same arithmetic, two extra launches -- and is the only
 * reason the non-sparse gate arm runs on the sparse path. See the seam at its declaration.
 *
 * THE UNFUSED q|gate, k and v ARE THE DEFAULT AND STAY SO. The fp8 twin's single-`attn_qkv` GEMM
 * (upstream PR #30) is ported here as an ENV-GATED variant, RADIANCE_MX_ATTN_FUSE_QKV, default OFF
 * (see `fuse_qkv`): one MX GEMM over the row-concatenated weight, fed to the SAME consumers by three
 * bf16 row copies, so the tree and every existing container are unchanged with the flag unset. The
 * per-rank GQA sharding of k/v (which differs from q's) is preserved by the fused weight's
 * `row_parts`, exactly as GdnMX's ssm_inz does it -- see the declaration.
 */
#ifndef RAD_BLOCK_ATTN_GATED_MXFP4_H
#define RAD_BLOCK_ATTN_GATED_MXFP4_H

#include "rad_mx.h"

#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

/* THE QKV-FUSION A/B SWITCH. An OCP-MX checkpoint's q|gate, k and v concatenate along their rows
 * into one `attn_qkv` weight over the same normed activation (see the block field `fuse_qkv`). It
 * is a RUNTIME bool so the coordinator can A/B without a rebuild:
 *
 *     RADIANCE_MX_ATTN_FUSE_QKV=1 rad-convert ...   # build the fused container
 *     RADIANCE_MX_ATTN_FUSE_QKV=1 radiance ... serve # serve it
 *     radiance --model <unfused container> serve     # default: unchanged tree
 *
 * THE DEFAULT IS OFF, so a tree and every existing container are byte-for-byte what they were: with
 * the flag unset, declare() takes the unfused path and nothing here is allocated or issued. The
 * env is read once and reaches every read identically; it must NOT be a compile-time constant in
 * one place and an env read in another, or the declare-side container and the serve-side graph
 * would disagree about whether q|gate,k,v is one weight or three. Conversion and service MUST use
 * the same setting: the fused container holds `blk.*.attn_qkv` and no `attn_qg/k/v`, so the
 * opposite setting at serve time finds none of the separate weights and refuses by name. */
inline constexpr bool kMxAttnFuseQkvDefault = false;
inline bool mx_attn_fuse_qkv_from_env() {
    const char* v = std::getenv("RADIANCE_MX_ATTN_FUSE_QKV");
    if (!v || !v[0]) return kMxAttnFuseQkvDefault;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}

struct AttnGatedMX {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        ActMX   h{};        /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        /* [rope_table_rows, rot] f32, the model's, shared by every layer and filled once a step.
         * Zero means the plugin has not been taught to build one, and then the fused prologue is
         * not asked for at all. */
        rad_buf cos_sin = 0;
        ActMX   attn{};     /* [max_tok, q_dim]   attention output, then gated in place */
        /* [max_tok, q_dim] the de-interleaved sigmoid gate (`sigmoid` then `mul`,
         * rad_block_attn_gated.h's pair). BOTH arms of this block use it: the bf16 arm takes it
         * from the plugin, and the MX arm's gate stage is unfused too (the gate stage below says why
         * the fp8 twin's `M = T*n_head` fused form does not carry to the MX per-row quantiser). A
         * plugin that wired only the fused arm leaves it 0 and declare() allocates it. Its
         * [T, q_dim] extent is the same bytes as the [T*n_head, head_dim] the sigmoid reads. */
        rad_buf gate = 0;
        /* [max_tok, qg_dim + 2*kv_dim] -- the ONE projection when the block fuses q|gate, k and v
         * (RADIANCE_MX_ATTN_FUSE_QKV, `fuse_qkv`); 0 otherwise. The plugin leaves it 0 and
         * declare() allocates it, exactly as it does `gate`. */
        rad_buf qkv  = 0;
#ifdef QWEN35_PAROQUANT
        /* PATH A: the rotated activations the paroquant producers build. qg/kp/vp read `pq_h`
         * (the attn norm output, n_embd wide, reused one at a time); o reads `pq_attn` (the gated
         * attention output, q_dim wide). Declared once by the plugin, shared across layers. */
        ActMX   pq_h{};     /* [max_tok, n_embd] */
        ActMX   pq_h2{};    /* [max_tok, n_embd]  the second consumer of the SAME normed row */
        ActMX   pq_h3{};    /* [max_tok, n_embd]  the third */
        ActMX   pq_attn{};  /* [max_tok, q_dim]  */
#endif
    };

    struct Src {
        const char* norm   = nullptr;
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* THE KV PAGE, IN TOKENS. 0 means RAD_KV_BLOCK, which is 16 and is what every geometry in
     * this tree has wanted. It is a field rather than that constant because Qwen4-Exp's QSA
     * selects 4-TOKEN BLOCKS, and a page of 4 makes the selected set a paged block table instead
     * of a new attention kernel -- see docs/QSA.md. The attention kernel's own rows must carry a
     * matching C_EQ("block_size", ...) or nothing resolves, which is the check that keeps this
     * from being a number anyone can set. */
    int64_t     kv_block = 0;
    /* THE QSA SELECTION, when this layer has an indexer. Set, and on a step where one query a
     * sequence makes the selection per-SEQUENCE, the attention is issued with these in place of
     * the sequence's own block table and context length -- which is the whole of what makes it
     * sparse. Both are ordinary operands (the block table has always been one), so this is the
     * SAME op and the SAME kernel reading a different list.
     *
     * `sel` is `topk + 1` wide and the kernel takes its block-table pitch from the operand, so
     * nothing else has to be told. See docs/QSA.md; arch/common/rad_qsa.h produces them. */
    rad_buf     qsa_sel  = 0;
    rad_buf     qsa_sequ = 0;
    /* THE CONTEXT UP TO WHICH THE DENSE PATH IS THE SAME ANSWER, `budget + ratio - 1`. Below it
     * every complete block is selected and the two paths agree to the bit, and the dense one is
     * cheaper at prefill -- it reads each key once a query TILE where the sparse one reads a
     * table a query. Above it the dense path is a different model. Zero means "always sparse",
     * which is what a caller that has not been told the bound should get. */
    int64_t     qsa_exact_to = 0;
    /* `sel`'s row pitch, `topk + 1`. The extra slot is the partial page and is not spare --
     * r4d_qsa_select_bf16.hip says why. */
    int64_t     qsa_topk = 0;
    /* HOW MANY QUERY ROWS ONE SPARSE ATTENTION LAUNCH CARRIES, and it is a scratch budget rather
     * than a preference. The decode kernel's split-KV partials are sized AT DECLARE from the
     * `max_seqs` and `q_len` parameters, so the only way to issue more rows than `max_seqs` is to
     * declare a larger one -- which grows the scratch arena by `rows * q_len_band * n_head *
     * (head_dim + 2) * 4`, about 32 MiB at 256 rows on this model, ONCE for the whole program
     * because the arena is the maximum over ops.
     *
     * It matters because the alternative is `max_seqs` itself. At `--max-num-seqs 4` a 2048-token
     * prefill chunk would be 512 launches a layer, and the dispatch cost of that dominates what
     * the sparse path saves. 256 makes it 8.
     *
     * Raising it further is pure scratch: 512 rows is 63 MiB, 2048 is 253. 256 is the point at
     * which the launch cost has stopped mattering. */
    static constexpr int64_t kQsaRows = 256;
    int64_t qsa_rows() const { return g.max_seqs > kQsaRows ? g.max_seqs : kQsaRows; }
    /* THE GATED FORM TAKES THE WHOLE STEP IN ONE LAUNCH, because its band is q_len 1 and so its
     * partials are only `rows * n_head * (2 * head_dim + 8)` bytes -- 12.8 MiB at a 2048-row step
     * on this model, where the ungated op's q_len band makes the same rows cost the figures
     * above. Fewer launches still pay here: one 2048-row launch takes about 2.16 ms a layer
     * against about 3.0 ms for eight 256-row ones, each of which is 256 one-wave workgroups -- two
     * waves a SIMD -- with its own tail. */
    int64_t qsa_gq_rows() const { return g.max_tok > qsa_rows() ? g.max_tok : qsa_rows(); }
    /* THE ROTATION'S PAIRING, a question the checkpoint cannot answer by inspection and that no
     * per-op check can reach, because both readings -- halves (NeoX, what HF's rotate_half does
     * over the partial rotary width) and adjacent pairs (GPT-J) -- are self-consistent and libref
     * shares whichever one the graph picks. Either answer is a DIFFERENT MODEL. "mrope" is what the
     * model does, and over one position component a token it is NeoX (rad_block_attn_gated.h).
     *
     * THE q|gate ORDER IS FIXED TOO. q is the FIRST half of each head's pair and the gate the
     * second, which is how vLLM chunks q_gate, and every consumer of the projection has to agree
     * on it: the unfused q norm reads column 0 of each head, the gate quantiser column head_dim,
     * and the fused prologue splits the pair the same way inside its kernel. A swap would be
     * arithmetically silent -- q_norm renormalises either half and the sigmoid accepts either --
     * so an order that only some of them followed would be fluent and wrong. */
    const char* mode  = "mrope";

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    LinearMX   qg{}, kp{}, vp{}, o{};
    /* Q|GATE, K AND V AS ONE GEMM, behind RADIANCE_MX_ATTN_FUSE_QKV (`fuse_qkv`, default off).
     * An OCP-MX weight has its scale group along K and `block[0] == 1`, so the three projections
     * concatenate along their rows into one [qg_dim + 2*kv_dim, n_embd] weight over the SAME normed
     * activation -- one launch where k and v were two of their own at ~2.6 MB each, which at decode
     * are almost all fixed cost. This is the same mechanism GdnMX's `ssm_inz` already uses: several
     * checkpoint sources joined along dim 0 with `row_parts` giving THIS RANK's rows of each, so
     * the GQA split of k and v (which is NOT q's split) is preserved. See the declaration.
     *
     * The consumers are FED UNCHANGED: three bf16 row copies (`op_cqg` -> w.qg, `op_ck` -> w.k,
     * `op_cv` -> w.v, each a column slice of the fused row) reproduce exactly what the three
     * separate projections wrote, so every downstream op -- the fused prologue, the norms, the
     * gate, the cache store -- reads its own plane as it always did. The unfused path is the
     * default and is byte-identical with the flag off. */
    bool       fuse_qkv = false;
    LinearMX   qkv{};
    rad_op     op_cqg = 0, op_ck = 0, op_cv = 0;
    NormQuantMX nq_h{};
    bool       paro = false;
#ifdef QWEN35_PAROQUANT
    /* PATH A: q_proj, k_proj, v_proj and o_proj each carry their own rotation. The block norm emits
     * q's, k's and v's rotated activations in ONE launch (fpn_qkv); `o` is fed its own producer
     * over the gated attention output. The old per-linear producers remain for the external-norm
     * case (`ext_in`), where there is no norm launch to fuse into. */
    ParoFusedNorm fpn_qkv{};
    ParoProducer  pp_qg{}, pp_kp{}, pp_vp{}, pp_o{};
#endif

    rad_op     op_fused = 0;                                 /* qk_norm_rope_gate, if it resolves */
    rad_op     op_fused_kv = 0;                              /* ...with kv_store folded on, likewise */
    int64_t    fused_rows = 0;                               /* ...and the widest T it is issued at */
    rad_op     op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op     op_attn = 0, op_kv_store = 0, op_gq = 0;
    /* SEAM: THE SPARSE PATH'S FUSED GATE IS NOT DECLARED. rad_block_attn_gated_fp8.h declares
     * `op_attn_gq` as `attn_paged_gate_quant` here -- the split-KV merge carrying the gate quantiser
     * -- for the sparse path. There is no per-row MX schema for it, so this op is left 0 and
     * `step()`'s `gated` branch never runs: the sparse path takes the separate three-op gate arm
     * (`op_sig` -> `op_mul` -> `op_gq`), the same arithmetic with two extra launches. To finish:
     * register/point `attn_paged_gate_quant` at the MX per-row epilogue, declare it here with
     * `w.attn.q/s` and a `[M, 1]` scale, and the branch below lights up unchanged. */
    rad_op     op_attn_gq = 0;
    rad_op     op_sig = 0, op_mul = 0;                       /* the bf16 gate, Geom::w_bf16 */
    rad_op     op_ar = 0, op_add = 0;
    /* ...AND WHETHER THE COLLECTIVE AT THE END OF THIS BLOCK IS THE NEXT BLOCK'S FIRST PASS.
     * `ar_in` says this block's own norm absorbs its PREDECESSOR's all-reduce; `ar_out` says the
     * block after it absorbs THIS one's, so this one must not issue it. They are the residual
     * fold's two flags again, one boundary later, and they are separate for the same reason: a
     * block is declared without knowing what follows it. On EITHER wire -- the fused kernel
     * carries both payloads -- and both sides ask ar_fused_ok(g, T), so the two cannot disagree
     * about whether the collective has already been issued. */
    bool ar_out = false;
    /* ...and WHICH WIRES that consumer takes (ArOut, rad_mx.h): the norm's by default, and a gated
     * residual's write says its own (HyperConn::ar_take). Set before declare; both sides then ask
     * ar_taken(g, T, ar_out, ar_out_take). */
    int  ar_out_take = kArOutNorm;

    /* THE GPTQ CALIBRATION TAP (rad_arch.h:101), the same one MlpMX declares and conditioned the
     * same way (fp4's `wref`). The block's four MX linears share TWO input activations: qg, k and v
     * all read the normed hidden state `w.h.x`, and o reads the gated attention output `w.attn.x`.
     * So there are two accumulators and the normed input is written under each of qg, k and v's
     * names -- rad-convert quantises each declared weight separately and each looks for its own
     * `gram.r<rank>.<stem>.bin`.
     *
     * IT TAPS bf16, not the codes: the MX per-row scale cannot be expressed to the fp8
     * `gram_accum` row, whose `group` is fixed at 128 (see MlpMX's field comment). */
    bool        calib = false;
    rad_buf     b_gram_h = 0, b_gram_o = 0;
    rad_op      op_gram_h = 0, op_gram_o = 0;
    const char* gram_h_stem[3] = {nullptr, nullptr, nullptr};
    const char* gram_o_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;

    /* `fold` says the PREVIOUS block left its delta in `w.h.x` for this norm to add; `add_out`
     * says the NEXT one will not, so this block issues the trailing add itself. See rad_mx.h.
     *
     * `ar_in` is forwarded to the norm as its `fuse_ar` (kMxArFold true; see rad_mx.h) and
     * `ar_out_` is what the NEXT block's norm absorbs. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnGatedMX::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;
#ifdef QWEN35_PAROQUANT
    /* The paroquant arch's trunk is rotated, so q/k/v/o each take a producer. The fused q|gate,k,v
     * projection is refused: its members have one rotation each, which is not one GEMM's. */
    paro = true;
#else
    paro = false;
#endif

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* THE INPUT MAY BE PREPARED BY THE ARCHITECTURE, and `src.norm == nullptr` is how it says so.
     *
     * A pre-norm model's block owns its own norm: one op that adds the previous block's delta into
     * the residual stream, normalises it and quantises it (rad_mx.h's NormQuantMX). A GATED
     * RESIDUAL model's does not -- arch/common/rad_block_hc.h's `hc_read` produces the block input
     * from a stream four times as wide, with a data-dependent mix the block has no way to compute,
     * and there is nothing left for a norm here to do. So this block takes `wire.h` as ALREADY
     * normed and quantised and declares no norm weight at all.
     *
     * `ar_in` is refused in that mode rather than ignored: the fused all-reduce lives INSIDE the
     * norm, so with no norm there is nowhere to fold it and a caller that asked would silently get
     * an unreduced input. It emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
        if (g.w_bf16 && !w.gate) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm, one quantise, three consumers -- and one op, see rad_mx.h's NormQuantMX.
     *
     * THE AR FOLD IS `ar_in` (kMxArFold true; see rad_mx.h). The norm absorbs the predecessor's
     * all-reduce where `ar_in` names one and the fused op is declared (world 2); the predecessor's
     * standalone collective is skipped on the same predicate. */
    if (!ext_in && !paro)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));
    /* THE THREE PROJECTIONS, OR ONE. The default is three, exactly as before. With
     * RADIANCE_MX_ATTN_FUSE_QKV set they are ONE OCP-MX GEMM over the concatenated weight, which
     * is possible because the MX scale group runs along K and `block[0] == 1`: q|gate (2*q_dim
     * rows), k and v (kv_dim each) stack along dim 0 over the SAME normed activation, and the E8M0
     * plane's rows are the weight's rows, so it stacks with them.
     *
     * THE ROW PARTS ARE THE WHOLE POINT, and the GQA sharding is why. Rank r must own its slice of
     * EACH of q|gate, k and v, and k/v are NOT q's slice: at world <= n_head_kv_all the even shard
     * of the stack would give rank 0 all of q|gate, all of k and part of v, a well-formed wrong
     * model. `row_parts` is THIS RANK's rows per part -- {qg_dim, kv_dim, kv_dim}, per-rank units --
     * and core/format/share.cpp:101-120 takes [index*take, (index+1)*take) of each part, so the
     * q/k/v head split is preserved part by part. The parts must sum to the entry's rows over the
     * world (VR/world) or the load is refused by name, which is the check that a KV-replicated world
     * (world > n_head_kv_all, where kv_dim is a replicated head count) fails safely here rather than
     * silently. This mirrors GdnMX's ssm_inz: several ckpt sources joined along dim 0 with
     * row_parts. */
    fuse_qkv = mx_attn_fuse_qkv_from_env();
#ifdef QWEN35_PAROQUANT
    if (paro) fuse_qkv = false;
#endif
    if (fuse_qkv) {
        if (g.world > g.n_head_kv_all) {
            fprintf(stderr, "radiance: RADIANCE_MX_ATTN_FUSE_QKV with world_size %d over "
                            "n_head_kv %lld: the fused q|gate,k,v weight has no replicated-KV row "
                            "span; serve it unfused\n", g.world, (long long)g.n_head_kv_all);
            return RAD_E_INVAL;
        }
        if (!w.qkv) {
            w.qkv = decl_b(b, nm.f("blk.%d.attn_qkv_out", l), g.act_dtype,
                           { g.max_tok, qg_dim + 2 * kv_dim });
            if (!w.qkv) return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), qg_dim + 2 * kv_dim, g.n_embd,
                                 RAD_SHARD_ROW, grp_layer(l), { src.qg, src.k, src.v }, w.h, w.qkv,
                                 0, { qg_dim, kv_dim, kv_dim }));
        /* THE FEED FOR THE UNCHANGED CONSUMERS: a bf16 row copy out of each column slice of the
         * fused row, byte-for-byte what the separate projection wrote. `cast` is the bf16->bf16
         * strided row copy and reads both row pitches off the operands, so the source slice (pitch
         * qg_dim + 2*kv_dim) and the destination (pitch qg_dim / kv_dim) are both expressible. */
        auto cp = [&](int64_t n, rad_buf dst) {
            return rw(b, RAD_OP(b, "cast",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                                     RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                          RAD_NOWEIGHTS),
                      { w.qkv }, { dst });
        };
        op_cqg = cp(qg_dim, w.qg);
        op_ck  = cp(kv_dim, w.k);
        op_cv  = cp(kv_dim, w.v);
        if (!op_cqg || !op_ck || !op_cv) return RAD_E_INVAL;
    } else {
#ifdef QWEN35_PAROQUANT
        if (paro) {
            const char* qn = nm.f("blk.%d.attn_qg", l);
            const char* kn = nm.f("blk.%d.attn_k", l);
            const char* vn = nm.f("blk.%d.attn_v", l);
            RadEncoding qe{}, ke{}, ve{};
            RAD_ARCH_TRY(qg.probe(b, nm, g, qn, { src.qg }));
            RAD_ARCH_TRY(kp.probe(b, nm, g, kn, { src.k }));
            RAD_ARCH_TRY(vp.probe(b, nm, g, vn, { src.v }));
            if (!weight_enc(b, nm.f("%s.weight", qn), &qe) ||
                !weight_enc(b, nm.f("%s.weight", kn), &ke) ||
                !weight_enc(b, nm.f("%s.weight", vn), &ve))
                return RAD_E_DTYPE;
            if (!ext_in) {
                /* ONE norm launch emits q's, k's and v's rotated activations (fpn_qkv); the three
                 * GEMMs then read their own buffers. No per-linear producer launch remains. */
                std::vector<ParoFusedNorm::Part> qkv(3);
                qkv[0].name = qn; qkv[0].owner = nm.f("%s.weight", qn); qkv[0].enc = &qe; qkv[0].out = w.pq_h;
                qkv[1].name = kn; qkv[1].owner = nm.f("%s.weight", kn); qkv[1].enc = &ke; qkv[1].out = w.pq_h2;
                qkv[2].name = vn; qkv[2].owner = nm.f("%s.weight", vn); qkv[2].enc = &ve; qkv[2].out = w.pq_h3;
                RAD_ARCH_TRY(fpn_qkv.declare(b, nm, g, l, fold ? w.h.x : w.x, fold ? w.x : 0, w_norm, w.h.x,
                                             std::move(qkv), g.n_embd));
                RAD_ARCH_TRY(qg.declare(b, nm, g, qn, qg_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.qg }, fpn_qkv.parts[0].out, w.qg));
                RAD_ARCH_TRY(kp.declare(b, nm, g, kn, kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.k }, fpn_qkv.parts[1].out, w.k));
                RAD_ARCH_TRY(vp.declare(b, nm, g, vn, kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.v }, fpn_qkv.parts[2].out, w.v));
            } else {
                RAD_ARCH_TRY(pp_qg.declare(b, nm, g, l, qn, nm.f("%s.weight", qn), w.h.x, w.pq_h, g.n_embd, &qe));
                RAD_ARCH_TRY(pp_kp.declare(b, nm, g, l, kn, nm.f("%s.weight", kn), w.h.x, w.pq_h, g.n_embd, &ke));
                RAD_ARCH_TRY(pp_vp.declare(b, nm, g, l, vn, nm.f("%s.weight", vn), w.h.x, w.pq_h, g.n_embd, &ve));
                RAD_ARCH_TRY(qg.declare(b, nm, g, qn, qg_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.qg }, pp_qg.a, w.qg));
                RAD_ARCH_TRY(kp.declare(b, nm, g, kn, kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.k }, pp_kp.a, w.k));
                RAD_ARCH_TRY(vp.declare(b, nm, g, vn, kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { src.v }, pp_vp.a, w.v));
            }
        } else
#endif
        {
        RAD_ARCH_TRY(qg.declare(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd,
                                RAD_SHARD_ROW, grp_layer(l), { src.qg }, w.h, w.qg));
        RAD_ARCH_TRY(kp.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                                RAD_SHARD_ROW, grp_layer(l), { src.k }, w.h, w.k));
        RAD_ARCH_TRY(vp.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                                RAD_SHARD_ROW, grp_layer(l), { src.v }, w.h, w.v));
        }
    }

    /* THE FUSED PROLOGUE, AND IT IS A PROBE: the four ops below are declared either way and are
     * what runs when nothing serves this one. rad_block_attn_gated.h does not make the probe,
     * because the schema takes a PRECOMPUTED cos/sin table rather than `theta` and that block has
     * none; `rope_table` builds one here, once a step for every layer, and `w.cos_sin` is where
     * the model puts it. A block handed no table cannot ask for the fusion, which is what keeps a
     * plugin that does not build the table on the unfused path.
     *
     * NEOX ONLY. The fused kernel rotates NeoX pairs and has no `mode`, so the probe is made only
     * where mrope reduces to neox -- one position component a token, which is every text batch and
     * is argued in rad_block_attn_gated.h. A vision span would need the sections first.
     *
     * NO `gate_out`. The gate stage below reads the gate at its column offset inside the
     * interleaved projection, so a contiguous copy would be q_dim of bf16 a layer written for
     * nothing. The operand is optional and this passes none.
     *
     * IT IS BIT-IDENTICAL TO THE FOUR, checked in r4d_selftest (`qk_norm_rope_gate ... matches the
     * unfused pair`) rather than argued: the kernel rounds its norm output to bf16 before rotating
     * because that is what a norm that stores a tensor and a rope that reads it back do. It
     * reproduces `r4d_rmsnorm_bf16`'s sum-of-squares fold element by element rather than folding
     * eight elements a lane, and r4d_selftest asserts byte equality. A merely RELATIVE bound is
     * not enough here: a fold order that agrees to a ULP and not to the last bit moves the
     * engine's transcript within a few dozen steps. */
    /* M-RoPE qualifies too: the table is filled at one component a token, which is exact on every
     * pass whose tokens' components agree, and the step takes the fused path only then. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    fused_rows = qk_fuse_rows(g);
    if (w.cos_sin && rope_neox && fused_rows > 0)
        op_fused = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, fused_rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", g.n_head),
                                    RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_q_norm, w_k_norm)),
                     {w.qg, w.k, w.cos_sin}, {w.q, w.k});

    op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_q_norm)),
                   {w.qg}, {w.q});
    op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_k_norm)),
                   {w.k}, {w.k});
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    /* Declared before kv_store because block_size is RAD_DERIVED and kv_store reads it back --
     * rad_block_attn_gated.h explains why the declared order differs from the issue order. */
    op_attn = RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                /* The sparse path issues one row a QUERY, so the bound the split-KV
                                 * scratch is sized from is the widest such launch and not the
                                 * sequence count -- see kQsaRows. */
                                RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_rows()
                                                                          : g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
    /* THE QSA SELECTION IS A READ OF THIS OP AND HAS TO BE DECLARED AS ONE. The block table has
     * always been a raw batch pointer, so `attn_paged`'s read set was just the query -- and
     * `sel`/`seqused` are BUFFERS, whose lifetimes the Builder plans from exactly these lists. Left
     * out, their last use is `qsa_select` and the arena is free to lay another buffer over them;
     * anything written between the selection and the attention then arrives as a page id, and a
     * garbage page id is an address far outside the cache -- an aperture violation at the first
     * step whose plan happens to overlap them, which need not be a step that ran before. Spelled
     * with rad_op_reads rather than rw() because the list is conditional -- a layer with no
     * indexer has no such read.
     *
     * THE DECLARATION ORDER IS THE BUFFER PLAN, and this is one of the shapes that rule takes. */
    if (op_attn) {
        const rad_buf rd[3] = { w.q, qsa_sel, qsa_sequ };
        rad_op_reads(b, op_attn, rd, (qsa_sel && qsa_sequ) ? 3 : 1);
        rad_op_writes(b, op_attn, &w.attn.x, 1);
    }

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* THE PROLOGUE AND THE CACHE STORE AS ONE LAUNCH, wherever the fused prologue is taken: a k
     * head's block already holds its token's rotated k, one element a lane, so it writes that and
     * the head's v into the cache too, byte-for-byte what kv_store would (the fold's row says why).
     * Declared here and not beside op_fused because the cache's block size is read back from the
     * attention op above. It resolves for an E4M3 and a bf16 cache alike. */
    if (op_fused)
        op_fused_kv = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});

    /* THE OUTPUT GATE, THE MULTIPLY AND THE QUANTISER. The bf16 arm runs the unfused pair
     * (`sigmoid` then `mul`, rad_block_attn_gated.h) and stops there. THE MX ARM RUNS THE SAME
     * PAIR AND THEN A WHOLE-ROW `quant_act_fp8_row`.
     *
     * The fp8 twin declares one fused `gate_quant_fp8` over `M = T*n_head`, `n = head_dim`, and
     * argues its scale plane [T*n_head, head_dim/128] is byte-for-byte the [T, q_dim/128] the out
     * projection reads. THAT ARGUMENT DOES NOT CARRY TO MX. It holds only because group == 128
     * tiles each row: n_head * head_dim values partition into n_head * head_dim/128 groups, which
     * flatten to the same contiguous bytes as q_dim/128 groups over T rows -- the n_head factor is
     * cancelled by the column factor. The MX per-row group has NO column factor: one group a
     * (token, head) row is ONE scale a (token, head), so `M = T*n_head` writes T*n_head scales into
     * a plane the out projection reads as T. That is an overrun of the [max_tok, 1] scale plane and
     * the wrong values, not a relabelling. A per-row `gate_quant_fp8_row` form does exist in the
     * registry and would serve at `M = T`, `n = q_dim` over the already-de-interleaved gate, but
     * this arm does not wire it: the gate is de-interleaved by `op_sig` regardless, and the fold is
     * left to a later edit. rad_block_attn_gated_fp8.h is the sibling whose group made the fused
     * `M = T*n_head` form expressible; MX cannot, so it de-interleaves instead.
     *
     * The sequence is therefore: `op_sig` reads the gate's fixed stride inside the [q|gate]
     * projection into `w.gate` ([T*n_head, head_dim], the same bytes as [T, q_dim]); `op_mul`
     * multiplies the flattened gate into the attention output in place; `op_gq` is
     * `quant_act_fp8_row` over the whole [T, q_dim] row. THE RESULTING SCALE IS [T, 1], EXACTLY the
     * extent `o.step` reads -- LinearMX::step issues `brow_slice(a.s, r0, rows, 1)`. That equality
     * is the invariant this arm exists to keep; a future edit that reintroduces an `M = T*n_head`
     * quantiser breaks it.
     *
     * The bf16 product is still written, because that is what the unfused pair left in `w.attn.x`
     * and the codes are taken over it -- see r4d_fused_quant_fp8.hip on why that is a requirement
     * (the `*_row` forms round the product to bf16 before quantising, so the values the quantiser
     * sees must be the same bf16 ones). PREFILL AND DECODE BOTH TAKE IT: unlike the gdn fold this
     * one has no reduction across the row, so there is no shape it cannot serve. */
    /* THE GATE PLANE. The bf16 arm always took `w.gate` from the plugin. The MX arm did not need
     * one, because its fused gate wrote the codes directly -- so a plugin that only wired the fused
     * arm leaves it 0 and the three-op arm below would write through a null buffer. Declare it here
     * at [max_tok, q_dim] bf16, the same bytes the sigmoid's [T*n_head, head_dim] view addresses. */
    if (!g.w_bf16 && !w.gate) {
        w.gate = decl_b(b, nm.f("blk.%d.attn_gate", l), g.act_dtype, {g.max_tok, q_dim});
        if (!w.gate) return RAD_E_INVAL;
    }
    if (g.w_bf16) {
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {w.gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, w.gate}, {w.attn.x});
        if (!op_sig || !op_mul) return RAD_E_INVAL;
    } else {
        /* The bf16 pair verbatim, on the MX path's same buffers. */
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {w.gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, w.gate}, {w.attn.x});
        /* ONE PER-TOKEN SCALE OVER THE WHOLE ROW. `M` is T and `n` is q_dim -- NOT T*n_head and
         * head_dim -- so `w.attn.s` is written at its declared [T, 1] extent. This mirrors
         * QuantMX::declare's quant_act_fp8_row exactly, with the width widened from one linear's
         * input to the whole attention output. */
        /* PATH A DOES NOT NEED THIS: `pp_o` rotates `w.attn.x` directly, so the E4M3 codes
         * `op_gq` writes are never read -- declaring it would be a wasted launch per layer. */
        if (!paro) {
            op_gq = rw(b, RAD_OP(b, "quant_act_fp8_row",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                      RAD_STR("dtype", g.dtype)),
                           RAD_NOWEIGHTS),
                       {w.attn.x}, {w.attn.q, w.attn.s});
        }
        if (!op_sig || !op_mul || (!paro && !op_gq)) return RAD_E_INVAL;
    }

    /* ---- SEAM: THE SPARSE PATH'S FUSED GATE (the fp8 twin's `op_attn_gq`). rad_block_attn_gated_fp8.h
     * declares `attn_paged_gate_quant` here, the split-KV merge carrying the gate quantiser's pass,
     * and step() issues it on the sparse path instead of the separate gate arm. There is no per-row
     * MX schema for that op, so it is not declared: `op_attn_gq` stays 0, step()'s `gated` test is
     * false, and the sparse path runs the three-op arm (`op_sig` -> `op_mul` -> `op_gq`, the same
     * arithmetic, two extra launches). The seam comment on the field says what to wire. */

    const char* on = nm.f("blk.%d.attn_output", l);
#ifdef QWEN35_PAROQUANT
    if (paro) {
        RadEncoding oe{};
        RAD_ARCH_TRY(o.probe(b, nm, g, on, { src.o }));
        if (!weight_enc(b, nm.f("%s.weight", on), &oe)) return RAD_E_DTYPE;
        RAD_ARCH_TRY(pp_o.declare(b, nm, g, l, on, nm.f("%s.weight", on), w.attn.x, w.pq_attn, q_dim,
                                  &oe, RAD_SHARD_COL));
        RAD_ARCH_TRY(o.declare(b, nm, g, on, g.n_embd, q_dim, RAD_SHARD_COL, grp_layer(l),
                               { src.o }, pp_o.a, w.h.x));
    } else
#endif
    RAD_ARCH_TRY(o.declare(b, nm, g, on, g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x));

    /* THE CALIBRATION TAP, declared after every linear so its buffers cannot alias one the block
     * still reads (PERSIST, and declaration order IS the buffer plan). */
    calib = calib_dir() != nullptr && (fuse_qkv ? qkv.wref != 0 : qg.wref != 0)
            && calib_layer_in(l);
    if (calib) {
        /* A rank holding a slice of the o projection's K sees a slice of that Hessian. Calibration
         * runs at one rank, as MlpMX's tap says. */
        if (g.world > 1) {
            fprintf(stderr, "radiance: calibration runs at one rank; this rank holds a slice of "
                            "the o projection's width, and that Hessian would be of the slice\n");
            return RAD_E_INVAL;
        }
        /* ONE STEM WHEN FUSED. The tap writes the same normed-input Hessian under every declared
         * weight's name, because rad-convert quantises each declared weight separately and each
         * looks for its own gram. Fused there is ONE weight (`attn_qkv`), so there is one stem; the
         * run loop below skips the absent ones. */
        if (fuse_qkv) {
            gram_h_stem[0] = nm.f("blk.%d.attn_qkv", l);
        } else {
            gram_h_stem[0] = nm.f("blk.%d.attn_qg", l);
            gram_h_stem[1] = nm.f("blk.%d.attn_k",  l);
            gram_h_stem[2] = nm.f("blk.%d.attn_v",  l);
        }
        gram_o_stem    = nm.f("blk.%d.attn_output", l);
        b_gram_h = decl_b(b, nm.f("blk.%d.gram_h", l), RAD_F32, {g.n_embd, g.n_embd},
                          RAD_BUF_PERSIST);
        b_gram_o = decl_b(b, nm.f("blk.%d.gram_o", l), RAD_F32, {q_dim, q_dim}, RAD_BUF_PERSIST);
        op_gram_h = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.h.x, b_gram_h}, {b_gram_h});
        op_gram_o = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                      RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.attn.x, b_gram_o}, {b_gram_o});
        if (!b_gram_h || !b_gram_o || !op_gram_h || !op_gram_o) return RAD_E_INVAL;
    }

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    /* Only where the NEXT block's norm will not do it; see rad_mx.h's NormQuantMX. */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

inline void AttnGatedMX::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    if (!ext_in && !paro) nq_h.step(c, T);
    /* One OCP-MX GEMM over [q|gate | k | v] and three bf16 row copies out of its column slices, or
     * the three separate projections. The copies reproduce exactly the planes the consumers below
     * read, so NOTHING after this point knows which path ran. */
    if (fuse_qkv) {
        const int64_t qgd = (int64_t)g.n_head * 2 * g.head_dim;
        const int64_t kvd = (int64_t)g.n_head_kv * g.head_dim;
        qkv.step(c, w.h, w.qkv, T);
        RAD_ISSUE_N(c, op_cqg, T, bcol(w.qkv, 0, qgd, T), brows(w.qg, T));
        RAD_ISSUE_N(c, op_ck,  T, bcol(w.qkv, qgd, kvd, T), brows(w.k, T));
        RAD_ISSUE_N(c, op_cv,  T, bcol(w.qkv, qgd + kvd, kvd, T), brows(w.v, T));
    } else {
#ifdef QWEN35_PAROQUANT
        if (paro) {
            if (!ext_in) {
                fpn_qkv.step(c, T);
                qg.step(c, fpn_qkv.parts[0].out, w.qg, T);
                kp.step(c, fpn_qkv.parts[1].out, w.k,  T);
                vp.step(c, fpn_qkv.parts[2].out, w.v,  T);
            } else {
                pp_qg.step(c, T); qg.step(c, pp_qg.a, w.qg, T);
                pp_kp.step(c, T); kp.step(c, pp_kp.a, w.k,  T);
                pp_vp.step(c, T); vp.step(c, pp_vp.a, w.v,  T);
            }
        } else
#endif
        {
        qg.step(c, w.h, w.qg, T);
        kp.step(c, w.h, w.k,  T);
        vp.step(c, w.h, w.v,  T);
        }
    }

    /* Four launches or one. The fused kernel takes the [q|gate] projection WHOLE and splits it
     * itself -- so `w.qg` goes in as it stands and not as the column slice the unfused norm needs
     * -- and it rotates k in place, which is what the unfused pair does too. */
    /* THE FUSED PATH READS ONE COMPONENT A TOKEN, so a pass carrying media tokens -- whose
     * components differ -- takes the unfused pair with the whole [3, T] instead. */
    const bool fuse_ok = T <= fused_rows && !rope_mixed(batch);
    const bool fused_kv = op_fused_kv && fuse_ok;
    if (fused_kv) {
        RAD_ISSUE_N(c, op_fused_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else if (op_fused && fuse_ok) {
        RAD_ISSUE_N(c, op_fused, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    }

    if (!fused_kv)
        RAD_ISSUE(c, op_kv_store,
                  brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    /* ONE ROW A QUERY IS WHAT MAKES THE SPARSE PATH WORK AT PREFILL, and it needs no new kernel
     * for the same reason the decode path needed none: `attn_paged` indexes its block table and
     * its length by the ROW, and nothing says a row has to be a sequence. A chunk of T queries
     * issued as T rows of q_len 1, each with its own selection and its own `seqused`, is exactly
     * per-query sparse attention -- and at decode, where T is the sequence count, it is the same
     * issue it has always been.
     *
     * DENSE BELOW THE BOUND, because there the two are the same answer and dense is cheaper: it
     * reads each key once a query TILE where this reads a table a query. `max_ctx_len` is the
     * host-side bound the batch already carries, so the test costs nothing.
     *
     * CHUNKED AT max_seqs, and that is not tidiness. The decode kernel's split-KV scratch is
     * sized AT DECLARE from the `max_seqs` and `q_len` parameters -- max_seqs * q_len_max * heads
     * rows -- and handing it T rows past that is refused as RAD_E_SCRATCH at issue. max_seqs rows
     * of q_len 1 is strictly inside what q_len_max >= 1 already bought, so the bound holds
     * whatever the kernel's band turns out to be. */
    const int64_t reach  = (int64_t)batch->max_ctx_len + batch->max_q_len;
    const bool    sparse = qsa_sel && qsa_sequ &&
                           (T == batch->n_seq || qsa_exact_to <= 0 || reach > qsa_exact_to);
    const bool gated = sparse && op_attn_gq;
    if (gated) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            RAD_ISSUE_N(c, op_attn_gq, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw),
                        bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                        brow_slice(w.attn.q, off, rows, qw),
                        /* The per-row MX scale is [M, 1]; the fp8 twin's was [M, q_dim / 128]. */
                        brow_slice(w.attn.s, off, rows, 1));
        }
    } else if (sparse) {
        const int64_t chunk = qsa_rows();
        const int64_t qw = g.q_dim();
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            RAD_ISSUE_N(c, op_attn, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw));
        }
    } else {
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, layer),
                    praw2(table, RAD_I32, batch->n_seq,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    /* cu_seqlens. The batch may be ragged -- one prefill chunk beside other
                     * sequences one decode row each -- and without it the kernel has to
                     * assume a single q_len for the whole step, which it cannot. */
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn.x, T));
    }

    /* THE GATE STAGE, in two passes on the bf16 arm and three on the MX arm (the bf16 out
     * projection reads the bf16 product, so it has no quantiser). `!gated` keeps the fp8 twin's
     * fused sparse seam honest: when `op_attn_gq` is eventually wired it already gated and
     * quantised the row in its merge, and re-issuing this arm would apply the sigmoid twice. */
    if (op_sig && !gated) {
        /* The gate is the strided second half of each head's [q|gate] pair. */
        RAD_ISSUE_N(c, op_sig, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.gate, T));
        RAD_ISSUE(c, op_mul, brows(w.attn.x, T), brows(w.gate, T), brows(w.attn.x, T));
        /* ONE ROW A TOKEN, NOT A (TOKEN, HEAD): `M = T`, `n = q_dim`, so `w.attn.s` is written at
         * its [T, 1] extent -- the same one `o.step` reads. */
        if (op_gq)
            RAD_ISSUE_N(c, op_gq, T,
                        brows(w.attn.x, T), brows(w.attn.q, T), brows(w.attn.s, T));
    }

    /* THE CALIBRATION TAP, after the gate stage has finalised `w.attn.x` and BEFORE the o
     * projection, which overwrites `w.h.x` (the normed input one of the taps reads). */
    if (calib && !calib_done) {
        if (calib_rows == 0) {
            rad_memset_async(rad_buf_ptr(c, b_gram_h), 0, g.n_embd * g.n_embd * 4, rad_stream(c));
            rad_memset_async(rad_buf_ptr(c, b_gram_o), 0, g.q_dim() * g.q_dim() * 4, rad_stream(c));
        }
        RAD_ISSUE_N(c, op_gram_h, T, brows(w.h.x, T), RAD_NONE, brows(b_gram_h, g.n_embd));
        RAD_ISSUE_N(c, op_gram_o, T, brows(w.attn.x, T), RAD_NONE, brows(b_gram_o, g.q_dim()));
        calib_rows += T;
        if (calib_rows >= calib_target()) {
            bool ok = true;
            for (int i = 0; i < 3; ++i)
                if (gram_h_stem[i])
                    ok = calib_write(c, b_gram_h, g.n_embd, calib_rows, gram_h_stem[i], 0) && ok;
            ok = calib_write(c, b_gram_o, g.q_dim(), calib_rows, gram_o_stem, 0) && ok;
            calib_done = true;
            if (layer == 0)
                std::fprintf(stderr, "radiance: calibration wrote layer Hessians after %lld "
                                     "tokens to %s%s\n", calib_rows, calib_dir(),
                             ok ? "" : " (WITH ERRORS)");
        }
    }
#ifdef QWEN35_PAROQUANT
    if (paro) { pp_o.step(c, T); o.step(c, pp_o.a, w.h.x, T); }
    else
#endif
    o.step(c, w.attn, w.h.x, T);
    if (op_ar && !(kMxArFold && ar_taken(g, T, ar_out, ar_out_take)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_MXFP4_H */
