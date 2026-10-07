/* rad_block_mlp_mxfp4.h -- the SwiGLU feed-forward block with OCP-MX linears.
 *
 * THE ARCHITECTURE IS rad_block_mlp.h's AND IS DOCUMENTED THERE, and this file is the OCP-MX twin
 * of rad_block_mlp_fp8.h, which documents the block-scaled-FP8 form of it. Like that file it is
 * rad_block_mlp.h with its projections made LinearMX and its producer made NormQuantMX, and it
 * repeats none of the architecture prose: a second copy is a second thing to keep true.
 *
 * What differs from the fp8 twin, and only this:
 *
 *   - the two projections are LinearMX (rad_mx.h): an OCP-MX codebook -- fp4 e2m1, fp8 e4m3, or
 *     fp6 e2m3 -- with an E8M0 exponent per group of 32 along K, read through `gemm_nt_q` at the
 *     codebook's dtype (mxfp4a8 / mxfp8a8 / mxfp6a8). The exponent plane is declared as a weight
 *     of its own -- twice for fp4, the [N, K/32] block scale and the [N, K/32] per-row reference,
 *     because fp4's epilogue folds against that reference; fp8 and fp6 codes self-scale, so their
 *     weight is TWO operands of the GEMM and fp4's is THREE.
 *   - the activation scale is ONE f32 PER ROW (`rmsnorm_quant_fp8_row`, `gated_quant_fp8_row`),
 *     not one per (row, 128 columns). Between ops the activations are still bf16; only the two
 *     GEMM operands are narrow. So `Geom::dtype` stays "bf16" for this block and every op but the
 *     two linears resolves exactly as it does in the bf16 plugin.
 *   - the norm, the activation and the collective are UNCHANGED.
 *
 * THE GATE/UP FUSION SURVIVES THE FORMAT, and here it is simpler than in fp8. The E8M0 plane's own
 * extents are [N, K/32] -- its dim 0 is the weight's ROWS, because `block[0] == 1` -- so
 * concatenating gate and up along dim 0 concatenates the plane rows one for one: gate and up are
 * [17408,5120] each with [17408,160] planes, and the fused weight is [34816,5120] against
 * [34816,160]. There is no 128-row seam to keep aligned the way the fp8 twin's [136,40]+[136,40]
 * has; the row_parts carry the {n_ff, n_ff} split.
 *
 * THE GATED FOLD IS WIRED AT DECODE, FOR FP4 ONLY: `op_gated` is `gemm_nt_q_gated_mx` (rad_mx.h's
 * per-row twin of the fp8 fold), declared over the three MX views of gate_up -- which exist only
 * for fp4, since fp8 and fp6 codes self-scale and declare no reference -- and taken for M <= 64.
 * The pair (`LinearMX::step` then `GateQuantMX::step`) still runs for prefill (M > 64), for the
 * fp8/fp6 codebooks at every M, and where the row does not resolve, and is byte-identical to the
 * fold.
 */
#ifndef RAD_BLOCK_MLP_MXFP4_H
#define RAD_BLOCK_MLP_MXFP4_H

#include "rad_mx.h"

namespace rad {
namespace arch {

struct MlpMX {
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActMX   h{};           /* [max_tok, n_embd]  normed input, then the down-projection output */
        rad_buf gate_up = 0;   /* [max_tok, 2 * n_ff] */
        ActMX   ffn{};         /* [max_tok, n_ff] */
#ifdef QWEN35_PAROQUANT
        /* THE PATH-A ROTATED ACTIVATIONS. A paroquant weight is served rotated, so gate/up/down
         * each read an activation this block's ParoProducers build -- gate/up share the n_embd-wide
         * pair (produced then consumed one at a time), down its own n_ff-wide one. Declared by the
         * plugin so they are shared across layers, not allocated per weight. */
        ActMX   pq_h{};        /* [max_tok, n_embd] */
        ActMX   pq_h2{};       /* [max_tok, n_embd]  the second consumer of the SAME normed row */
        ActMX   pq_ff{};       /* [max_tok, n_ff]   */
        /* gate and up write their own contiguous [max_tok, n_ff] buffers -- a gemm_nt_q output
         * must be contiguous -- and a `cast` places each into its half of the [gate|up] the silu
         * producer reads. */
        rad_buf gate2 = 0, up2 = 0;
#endif
    };

    /* The checkpoint side. `gate`, `up` and `down` name the `.weight` tensors; an MX checkpoint's
     * exponent plane and its per-row reference sit beside each and the checkpoint reader pairs them,
     * so a caller cannot spell one of the set and forget the others. */
    struct Src {
        const char* norm  = nullptr;
        const char* gate  = nullptr;
        const char* up    = nullptr;
        const char* down  = nullptr;
    };

    Geom      g{};
    Wire      w{};
    int       layer = 0;

    rad_weight w_norm = 0;
    LinearMX   gate_up{}, down{};
    NormQuantMX nq_h{};
    GateQuantMX gq_ffn{};
    bool         paro = false;
#ifdef QWEN35_PAROQUANT
    /* PATH A: gate and up are two GEMMs because their rotations differ (the checkpoint's `pairs`
     * are shared a layer but `theta` is a weight's own). The norm emits BOTH rotations in its own
     * launch (ParoFusedNorm) -- the per-linear producer launch it replaces was ~100% of path A's
     * decode gap -- and `down` is fed the fused silu-product producer (ParoFusedGated). */
    LinearMX       gate_g{}, up_g{};
    ParoFusedNorm  fpn_gu{};
    ParoFusedGated fpg_down{};
#endif
    rad_op     op_ar = 0, op_add = 0;
    /* THE gate_up PROJECTION WITH THE GATE AND THE QUANTISER IN ITS EPILOGUE. `gq_ffn`'s only input
     * is `gate_up`'s only output, so the fold removes the launch and the second pass over the
     * [M, 2*n_ff] bf16 tensor the GEMM's transposing store had just scattered. It is
     * `gemm_nt_q_gated_mx` (libr4d/r4d_rows.cpp) over the three MX views `gate_up.w, gate_up.ws,
     * gate_up.wref` -- its schema carries `b_ref` at operand 7, which the gated layout hook remaps
     * to 6 (r4d_layout.cpp's mx_gated_operand). DECLARED BESIDE THE PAIR AND NOT INSTEAD OF IT: the
     * fold is the narrow tile's epilogue and serves M <= 64, so a prefill chunk takes the pair. The
     * two are byte-identical, so this is two spellings of one arithmetic. */
    rad_op     op_gated = 0;
    int64_t    gated_rows = 0;
    /* ...AND WHETHER THE COLLECTIVE AT THE END OF THIS BLOCK IS THE NEXT BLOCK'S FIRST PASS.
     * `ar_in` says this block's own norm absorbs its PREDECESSOR's all-reduce; `ar_out` says the
     * block after it absorbs THIS one's, so this one must not issue it. They are the residual
     * fold's two flags again, one boundary later, and they are separate for the same reason: a
     * block is declared without knowing what follows it. On EITHER wire -- the fused kernel
     * carries both payloads -- and both sides ask ar_fused_ok(g, T), so the two cannot disagree
     * about whether the collective has already been issued. */
    bool ar_out = false;

    /* THE GPTQ CALIBRATION TAP (rad_arch.h:101). When RADIANCE_CALIB_DIR is set and this block's
     * weights are OCP-MX fp4, it declares one f32 [K,K] PERSIST accumulator per MX linear input
     * and a `gram_accum` into it, runs until RADIANCE_CALIB_TOKENS tokens have passed, then writes
     * each to its weight's stem and switches itself off.
     *
     * IT TAPS THE bf16 ACTIVATION AND NOT THE CODES, and here that is forced rather than chosen.
     * The MX activation's scale is ONE f32 A ROW (ActMX), and the fp8 `gram_accum` row fixes its
     * `group` at 128 and its scale plane at [M, K/128] -- a per-row scale cannot be expressed to
     * it. So the bf16 row is the one this can use: the activation the quantiser rounds to before
     * taking codes, and therefore the GEMM's operand up to that rounding.
     *
     * CONDITIONED ON THE CODEBOOK, as the rest of rad_mx.h is: fp4 declares a per-row reference
     * (`wref`) and fp8/fp6 do not, so `wref` is the fp4 test and mx8/mx6 declare no tap at all.
     *
     * NAMED AFTER THE DECLARED WEIGHT, minus `.weight`: rad-convert quantises the declared logical
     * name and libquant strips the suffix to look for `gram.r<rank>.<stem>.bin` (lq_hess.cpp). One
     * accumulator covers every weight that reads the same activation. */
    bool        calib = false;
    rad_buf     b_gram_gu = 0, b_gram_dn = 0;
    rad_op      op_gram_gu = 0, op_gram_dn = 0;
    const char* gram_gu_stem = nullptr;
    const char* gram_dn_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;


    /* `fold` says this block has a PREDECESSOR whose delta is still in `w.h.x`, so its norm does
     * the residual add that block would otherwise have issued. False for the first block of a
     * chain, whose residual stream comes straight from the embedding. See rad_mx.h.
     *
     * `ar_in` is forwarded to the norm as its `fuse_ar`: this block's norm absorbs the predecessor's
     * all-reduce where `ar_in` names one and the fused op is declared (kMxArFold true, world 2).
     * `ar_out` then says whether the NEXT block's norm absorbs THIS one's. See rad_mx.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* One row slice of the whole block; `rows <= 0` means the rest of the step. */
    void slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
};

inline int MlpMX::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                          const Src& src, bool fold, bool add_out,
                          int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_;
#ifdef QWEN35_PAROQUANT
    /* The paroquant arch serves only rotated trunk weights, so every MLP linear here takes the
     * path-A branch below. A weight that is not `givens` is refused by ParoProducer, which is the
     * correct answer for an arch whose container is paroquant. */
    paro = true;
#else
    paro = false;
#endif

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* THE NORM, THE QUANTISER AND THE RESIDUAL ADD ARE ONE OP, and rad_mx.h's NormQuantMX says
     * why: both were a launch and a second pass over a row this kernel already had in registers.
     * Folded, the norm reads the PREVIOUS block's delta out of the shared `h` buffer and adds it
     * into the stream; unfolded it reads the stream directly and the block ends with its own add.
     *
     * The AR fold into this norm is `ar_in` (kMxArFold true; see rad_mx.h). The unfused norm it
     * declares is taken wherever the fused op is not reachable; the predecessor's standalone
     * collective is issued on the same predicate, so the two cannot disagree. */
    if (!paro)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd, fold ? w.x : 0,
                                  0, ar_in));
    const char* down_base = nm.f("blk.%d.ffn_down", l);
    if (!paro) {
    RAD_ARCH_TRY(gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                                 2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                 { src.gate, src.up }, w.h, w.gate_up, 0, { g.n_ff, g.n_ff }));

    /* THE bf16 SILU PRODUCT IS WRITTEN ONLY FOR A READER: a down projection that reads the bf16
     * plane (LinearMX::reads_x). An MX down_proj takes the codes and their per-row scale alone. */
    RAD_ARCH_TRY(down.probe(b, nm, g, down_base, { src.down }));
    RAD_ARCH_TRY(gq_ffn.declare(b, g, w.gate_up, w.ffn, g.n_ff, "silu", 0, down.reads_x));

    /* THE FUSED GATED PROJECTION (the fp8 twin's `op_gated`), in its OCP-MX form. It runs the MX
     * GEMM and folds the SILU gate and the per-row fp8 quantiser into its epilogue, over the same
     * three MX views of `gate_up` and into the same activation the pair writes, and it serves the
     * narrow decode band only (M <= 64). Not RAD_ARCH_TRY: a plugin without the row is the
     * ordinary case and the pair below still runs; a prefill chunk (M > 64) takes the pair, and the
     * two are byte-identical. */
    {
        const int64_t hi = g.max_tok < 64 ? g.max_tok : 64;
        /* `gate_up.wref` IS the fp4 gate: the gated fold is fp4's epilogue, and an fp8/fp6 weight
         * declares no reference view, so those codebooks take the pair at every M. */
        if (hi > 0 && !gate_up.bf16 && gate_up.wref) {
            op_gated = rw(b, RAD_OP(b, "gemm_nt_q_gated_mx",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", gate_up.N),
                                        RAD_INT("K", gate_up.K),
                                        RAD_STR("dtype", "mxfp4a8"), RAD_STR("act", "silu")),
                             RAD_WEIGHTS(gate_up.w, gate_up.ws, gate_up.wref)),
                         { w.h.q, w.h.s },
                         gq_ffn.want_x ? std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s, w.ffn.x }
                                       : std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s });
            if (op_gated) gated_rows = hi;
        }
    }

    RAD_ARCH_TRY(down.declare(b, nm, g, down_base,
                              g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l),
                              { src.down }, w.ffn, w.h.x));
    }
#ifdef QWEN35_PAROQUANT
    else {
        /* PATH A, FUSED. The block norm emits BOTH gate's and up's rotated activations in one launch
         * (fpn_gu); the two GEMMs then read their own buffers. `down` is fed the fused silu-product
         * producer (fpg_down), so no separate producer launch exists anywhere in this block. */
        const char* gn  = nm.f("blk.%d.ffn_gate", l);
        const char* un  = nm.f("blk.%d.ffn_up", l);
        const char* gwn = nm.f("blk.%d.ffn_gate.weight", l);
        const char* uwn = nm.f("blk.%d.ffn_up.weight", l);
        const char* dwn = nm.f("blk.%d.ffn_down.weight", l);
        RadEncoding ge{}, ue{}, de{};
        RAD_ARCH_TRY(gate_g.probe(b, nm, g, gn, { src.gate }));
        RAD_ARCH_TRY(up_g.probe(b, nm, g, un, { src.up }));
        RAD_ARCH_TRY(down.probe(b, nm, g, down_base, { src.down }));
        if (!weight_enc(b, gwn, &ge) || !weight_enc(b, uwn, &ue) || !weight_enc(b, dwn, &de))
            return RAD_E_DTYPE;
        std::vector<ParoFusedNorm::Part> gup(2);
        gup[0].name = gn; gup[0].owner = gwn; gup[0].enc = &ge; gup[0].out = w.pq_h;
        gup[1].name = un; gup[1].owner = uwn; gup[1].enc = &ue; gup[1].out = w.pq_h2;
        RAD_ARCH_TRY(fpn_gu.declare(b, nm, g, l, fold ? w.h.x : w.x, fold ? w.x : 0, w_norm, w.h.x,
                                    std::move(gup), g.n_embd));
        RAD_ARCH_TRY(gate_g.declare(b, nm, g, gn, g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                    { src.gate }, fpn_gu.parts[0].out, w.gate2));
        RAD_ARCH_TRY(up_g.declare(b, nm, g, un, g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                  { src.up }, fpn_gu.parts[1].out, w.up2));
        /* NO MERGED [gate|up] AND NO CAST: the two GEMMs' outputs ARE the gated producer's two
         * inputs, which removes the two `cast` launches the merged form needed. */
        std::vector<ParoFusedGated::Part> dpart(1);
        dpart[0].name = down_base; dpart[0].owner = dwn; dpart[0].enc = &de; dpart[0].out = w.pq_ff;
        RAD_ARCH_TRY(fpg_down.declare(b, nm, g, l, w.gate2, w.up2, std::move(dpart), g.n_ff, "silu",
                                      RAD_SHARD_COL));
        RAD_ARCH_TRY(down.declare(b, nm, g, down_base, g.n_embd, g.n_ff, RAD_SHARD_COL,
                                  grp_layer(l), { src.down }, fpg_down.parts[0].out, w.h.x));
    }
#endif

    /* THE CALIBRATION TAP, declared after every linear so its buffers cannot alias one the block
     * still reads (they are PERSIST, so the planner would not alias them anyway; declaration order
     * IS the buffer plan and a reader should not have to check the kind to know that). */
    calib = calib_dir() != nullptr && gate_up.wref != 0 && calib_layer_in(l);
    if (calib) {
        /* A rank holding a slice of the down projection's width sees a slice of its K, and a
         * Hessian of a slice is not the Hessian the packer needs. Calibration runs at one rank. */
        if (g.world > 1) {
            fprintf(stderr, "radiance: calibration runs at one rank; this rank holds a slice of "
                            "the down projection's width, and that Hessian would be of the slice\n");
            return RAD_E_INVAL;
        }
        gram_gu_stem = nm.f("blk.%d.ffn_gate_up", l);
        gram_dn_stem = nm.f("blk.%d.ffn_down", l);
        b_gram_gu = decl_b(b, nm.f("blk.%d.gram_gu", l), RAD_F32, {g.n_embd, g.n_embd},
                           RAD_BUF_PERSIST);
        b_gram_dn = decl_b(b, nm.f("blk.%d.gram_dn", l), RAD_F32, {g.n_ff, g.n_ff},
                           RAD_BUF_PERSIST);
        /* `group` is a required parameter the bf16 row does not read (its only divisor is n % 4),
         * so it carries the MX grid width and nothing depends on the value. */
        op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.h.x, b_gram_gu}, {b_gram_gu});
        op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                      RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.ffn.x, b_gram_dn}, {b_gram_dn});
        if (!b_gram_gu || !b_gram_dn || !op_gram_gu || !op_gram_dn) return RAD_E_INVAL;
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

/* ONE ROW SLICE OF THE BLOCK. `rows` defaults to the whole step, and at r0 = 0 with rows = T every
 * operand is byte-for-byte the one the unsliced form built -- `brow_slice` at offset 0 is `brows`.
 *
 * Every op here is per row: the norm and the quantiser group along K within a row, the GEMMs give
 * each output row its own accumulation, and the fused all-reduce closes every reduction over one
 * row (r4d_ar.cpp says so where it caps the block count). So a slice computes exactly the rows it
 * names and nothing about the arithmetic depends on how the chunk was cut -- PROVIDED the cut
 * obeys the band rule in core/runtime/ctx.h. */
inline void MlpMX::slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
#ifdef QWEN35_PAROQUANT
    if (paro) {
        /* ONE norm launch emits gate's and up's rotated activations; one gated launch emits down's.
         * No per-linear producer launch remains -- the whole of path A's decode fix. */
        fpn_gu.step(c, T, r0, rows);
        gate_g.step(c, fpn_gu.parts[0].out, w.gate2, T, r0, rows);
        up_g.step(c, fpn_gu.parts[1].out, w.up2, T, r0, rows);
        fpg_down.step(c, T, r0, rows);
        down.step(c, fpg_down.parts[0].out, w.h.x, T, r0, rows);
    } else
#endif
    {
    nq_h.step(c, T, r0, rows);
    if (op_gated && rows <= gated_rows) {
        RAD_ISSUE_N(c, op_gated, rows,
                    brow_slice(w.h.q, r0, rows, g.n_embd),
                    brow_slice(w.h.s, r0, rows, 1), RAD_W(gate_up.w), RAD_W(gate_up.ws),
                    brow_slice(w.ffn.q, r0, rows, g.n_ff),
                    brow_slice(w.ffn.s, r0, rows, 1),
                    gq_ffn.want_x ? brow_slice(w.ffn.x, r0, rows, g.n_ff) : RAD_NONE,
                    RAD_W(gate_up.wref));
    } else {
        gate_up.step(c, w.h, w.gate_up, T, r0, rows);
        gq_ffn.step(c, w.gate_up, T, r0, rows);
    }
    /* THE CALIBRATION TAP, issued between the two GEMMs' inputs being written and the pass ending.
     * It reads exactly the operands the GEMMs above read and writes nothing they touch -- the
     * property that lets a calibration run be compared against a serving one. BEFORE the down
     * projection, which overwrites `w.h.x`. */
    if (calib && !calib_done) {
        if (calib_rows == 0) {
            /* The accumulators are arena memory and arrive holding whatever was there. */
            rad_memset_async(rad_buf_ptr(c, b_gram_gu), 0, g.n_embd * g.n_embd * 4, rad_stream(c));
            rad_memset_async(rad_buf_ptr(c, b_gram_dn), 0, g.n_ff * g.n_ff * 4, rad_stream(c));
        }
        RAD_ISSUE_N(c, op_gram_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE,
                    brows(b_gram_gu, g.n_embd));
        RAD_ISSUE_N(c, op_gram_dn, rows, brow_slice(w.ffn.x, r0, rows, g.n_ff), RAD_NONE,
                    brows(b_gram_dn, g.n_ff));
        calib_rows += rows;
        if (calib_rows >= calib_target()) {
            const bool a = calib_write(c, b_gram_gu, g.n_embd, calib_rows, gram_gu_stem, 0);
            const bool e = calib_write(c, b_gram_dn, g.n_ff, calib_rows, gram_dn_stem, 0);
            calib_done = true;
            if (layer == 0)
                std::fprintf(stderr, "radiance: calibration wrote layer Hessians after %lld "
                                     "tokens to %s%s\n", calib_rows, calib_dir(),
                             (a && e) ? "" : " (WITH ERRORS)");
        }
    }
    down.step(c, w.ffn, w.h.x, T, r0, rows);
    }
    if (op_ar && !(kMxArFold && ar_out && ar_fused_ok(g, rows)))
        RAD_ISSUE_N(c, op_ar, rows * g.n_embd, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    if (op_add)
        RAD_ISSUE_N(c, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

inline void MlpMX::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    slice(c, T, 0, T);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MLP_MXFP4_H */
