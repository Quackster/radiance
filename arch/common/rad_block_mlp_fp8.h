/* rad_block_mlp_fp8.h -- the SwiGLU feed-forward block with block-scaled FP8 linears.
 *
 *   rmsnorm -> quant_act_fp8 -> gate_up gemm_nt_q -> silu_mul -> quant_act_fp8 -> down gemm_nt_q
 *   -> [all_reduce] -> residual add
 *
 * THE ARCHITECTURE IS rad_block_mlp.h's AND IS DOCUMENTED THERE -- the gate/up fusion, why the
 * all-reduce is emitted by the plugin, what the residual add is for. This file is that block with
 * two ops added and two changed, and it repeats none of it: a second copy of the prose is a second
 * thing to keep true.
 *
 * What differs, and only this:
 *
 *   - the two projections are LinearFP8 (rad_fp8.h): an E4M3 weight, a bf16 [N/128][K/128] scale
 *     plane declared as a weight of its own, and `gemm_nt_q` at dtype fp8a8 instead of `gemm_nt`.
 *   - each of them is preceded by a `quant_act_fp8` over its input, because an fp8a8 GEMM takes an
 *     E4M3 activation and the checkpoint's `activation_scheme` is `dynamic` -- there is no
 *     activation scale on disk and one is computed every step.
 *   - the norm, the activation and the collective are UNCHANGED. Activations between ops are still
 *     bf16; only the two GEMM operands are narrow. So `Geom::dtype` stays "bf16" for this block and
 *     every op in it but the two linears resolves exactly as it does in the bf16 plugin.
 *
 * The gate/up fusion survives the format, which is not obvious and is the reason it is kept: the
 * two sources are [17408,5120] each and their scale planes [136,40] each, and concatenating both
 * along dim 0 gives [34816,5120] against [272,40] -- which is the grid, because 34816/128 is 272.
 * A fusion whose parts were not whole multiples of 128 rows would silently misalign every scale
 * after the seam.
 */
#ifndef RAD_BLOCK_MLP_FP8_H
#define RAD_BLOCK_MLP_FP8_H

#include "rad_fp8.h"

namespace rad {
namespace arch {

struct MlpFP8 {
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActFP8  h{};           /* [max_tok, n_embd]  normed input, then the down-projection output */
        rad_buf gate_up = 0;   /* [max_tok, 2 * n_ff] */
        ActFP8  ffn{};         /* [max_tok, n_ff] */
    };

    /* The checkpoint side. `gate`, `up` and `down` name the `.weight` tensors; a block-fp8
     * checkpoint's scale planes sit beside them and the checkpoint reader pairs them, so a caller
     * cannot spell one of the pair and forget the other. */
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
    LinearFP8  gate_up{}, down{};
    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_ffn{};
    rad_op     op_ar = 0, op_add = 0;
    /* ...AND THE gate_up PROJECTION WITH THE GATE AND THE QUANTISER IN ITS EPILOGUE.
     *
     * `gq_ffn`'s only input is `gate_up`'s only output. Unfused it is one of the larger remaining
     * items in a decode step, and roughly half of that is the launch and half the second pass over
     * a [M, 2*n_ff] bf16 tensor the GEMM's transposing store had just scattered. Folded, that
     * tensor is never written at all.
     *
     * DECLARED BESIDE THE PAIR AND NOT INSTEAD OF IT, exactly as LinearFP8's matvec is: the fold
     * is the narrow tile's epilogue and serves M <= 64, so a prefill chunk takes the pair. The two
     * are byte-identical -- r4d_selftest holds them to it, and ref_test holds the reference to
     * the same claim -- so this is two spellings of one arithmetic and not two answers. */
    rad_op     op_gated = 0;
    int64_t    gated_rows = 0;
    /* op_gated is `gemm_nt_q_gated_mx` (an MXFP4 gate_up, RADIANCE_MXFP4_GATED_FOLD) rather than the fp8
     * `gemm_nt_q_gated`: its operand list carries the weight's reference exponent plane. */
    bool       gated_mx = false;
    /* ...AND WHETHER THE COLLECTIVE AT THE END OF THIS BLOCK IS THE NEXT BLOCK'S FIRST PASS.
     * `ar_in` says this block's own norm absorbs its PREDECESSOR's all-reduce; `ar_out` says the
     * block after it absorbs THIS one's, so this one must not issue it. They are the residual
     * fold's two flags again, one boundary later, and they are separate for the same reason: a
     * block is declared without knowing what follows it. On EITHER wire -- the fused kernel
     * carries both payloads -- and both sides ask ar_fused_ok(g, T), so the two cannot disagree
     * about whether the collective has already been issued. */
    bool ar_out = false;


    /* `fold` says this block has a PREDECESSOR whose delta is still in `w.h.x`, so its norm does
     * the residual add that block would otherwise have issued. False for the first block of a
     * chain, whose residual stream comes straight from the embedding. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* One row slice of the whole block; `rows <= 0` means the rest of the step. */
    void slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const;
};

inline int MlpFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                           const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_;

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* THE NORM, THE QUANTISER AND THE RESIDUAL ADD ARE ONE OP, and rad_fp8.h's NormQuantFP8 says
     * why: both were a launch and a second pass over a row this kernel already had in registers.
     * Folded, the norm reads the PREVIOUS block's delta out of the shared `h` buffer and adds it
     * into the stream; unfolded it reads the stream directly and the block ends with its own add. */
    RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd, fold ? w.x : 0,
                              0, ar_in));
    RAD_ARCH_TRY(gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                                 2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                 { src.gate, src.up }, w.h, w.gate_up, 0, { g.n_ff, g.n_ff }));

    /* THE bf16 SILU PRODUCT IS WRITTEN ONLY FOR A READER: a down projection that reads the bf16
     * plane (LinearFP8::reads_x). A block-fp8 down_proj takes the codes and their scale alone. */
    const char* down_base = nm.f("blk.%d.ffn_down", l);
    RAD_ARCH_TRY(down.probe(b, nm, g, down_base, { src.down }));
    RAD_ARCH_TRY(gq_ffn.declare(b, g, w.gate_up, w.ffn, g.n_ff, "silu", 0, down.reads_x));

    /* The fold, over the same weights and into the same activation the pair writes. Not
     * RAD_ARCH_TRY: a plugin without the row is the ordinary case and the pair still runs. It
     * multiplies fp8 codes by a block-fp8 gate_up, so a plain gate_up has no fold -- and neither
     * has an MXFP4 one: libr4d's gated fold is an fp8a8 tile, and the pair serves it. */
    {
        const int64_t hi = g.max_tok < 64 ? g.max_tok : 64;
        if (hi > 0 && !gate_up.bf16 && !gate_up.mx) {
            op_gated = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", gate_up.N),
                                        RAD_INT("K", gate_up.K),
                                        RAD_INT("group", RAD_FP8_BLOCK),
                                        RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                             RAD_WEIGHTS(gate_up.w, gate_up.ws)),
                         { w.h.q, w.h.s },
                         gq_ffn.want_x ? std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s, w.ffn.x }
                                       : std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s });
            if (op_gated) gated_rows = hi;
        } else if (hi > 0 && !gate_up.bf16 && gate_up.mx && mxfp4_gated_fold_on(gate_up.N)) {
            /* THE MXFP4 FOLD: the same epilogue behind libr4d's decode GEMM; opt-in. Its operand list is
             * the fp8 fold's with the weight's reference exponent plane (`b_ref`) after `scale`. */
            op_gated = rw(b, RAD_OP(b, "gemm_nt_q_gated_mx",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", gate_up.N),
                                        RAD_INT("K", gate_up.K),
                                        RAD_INT("group", RAD_MX_GROUP),
                                        RAD_STR("dtype", "mxfp4a8"), RAD_STR("act", "silu")),
                             RAD_WEIGHTS(gate_up.w, gate_up.ws, gate_up.wref)),
                         { w.h.q, w.h.s },
                         gq_ffn.want_x ? std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s, w.ffn.x }
                                       : std::initializer_list<rad_buf>{ w.ffn.q, w.ffn.s });
            if (op_gated) { gated_rows = hi; gated_mx = true; }
        }
    }
    RAD_ARCH_TRY(down.declare(b, nm, g, down_base,
                              g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l),
                              { src.down }, w.ffn, w.h.x));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    /* Only where the NEXT block's norm will not do it; see rad_fp8.h's NormQuantFP8. */
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
inline void MlpFP8::slice(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    nq_h.step(c, T, r0, rows);
    if (op_gated && gated_mx && rows <= gated_rows) {
        RAD_ISSUE_N(c, op_gated, rows, brow_slice(w.h.q, r0, rows, g.n_embd),
                    brow_slice(w.h.s, r0, rows, g.n_embd / RAD_FP8_BLOCK), RAD_W(gate_up.w),
                    RAD_W(gate_up.ws), brow_slice(w.ffn.q, r0, rows, g.n_ff),
                    brow_slice(w.ffn.s, r0, rows, g.n_ff / RAD_FP8_BLOCK), RAD_W(gate_up.wref),
                    gq_ffn.want_x ? brow_slice(w.ffn.x, r0, rows, g.n_ff) : RAD_NONE);
    } else if (op_gated && rows <= gated_rows) {
        RAD_ISSUE_N(c, op_gated, rows, brow_slice(w.h.q, r0, rows, g.n_embd),
                    brow_slice(w.h.s, r0, rows, g.n_embd / RAD_FP8_BLOCK), RAD_W(gate_up.w),
                    RAD_W(gate_up.ws), brow_slice(w.ffn.q, r0, rows, g.n_ff),
                    brow_slice(w.ffn.s, r0, rows, g.n_ff / RAD_FP8_BLOCK),
                    gq_ffn.want_x ? brow_slice(w.ffn.x, r0, rows, g.n_ff) : RAD_NONE);
    } else {
        gate_up.step(c, w.h, w.gate_up, T, r0, rows);
        gq_ffn.step(c, w.gate_up, T, r0, rows);
    }
    down.step(c, w.ffn, w.h.x, T, r0, rows);
    if (op_ar && !(ar_out && ar_fused_ok(g, rows)))
        RAD_ISSUE_N(c, op_ar, rows * g.n_embd, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    if (op_add)
        RAD_ISSUE_N(c, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

inline void MlpFP8::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    slice(c, T, 0, T);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MLP_FP8_H */
