#ifndef RAD_MX_H
#define RAD_MX_H

/* rad_mx.h -- the OCP-MX (mxfp4a8/mxfp8a8/mxfp6a8) trunk's composed pieces, and the twin of rad_fp8.h.
 *
 * The two are deliberately the same file twice over, so read rad_fp8.h first. What differs is the
 * WEIGHT FORMAT and one property of the activation, and both come from OCP-MX rather than from a
 * choice this tree made:
 *
 *   - THE WEIGHT IS AN OCP-MX CODEBOOK: fp4 e2m1, fp8 e4m3, or fp6 e2m3 codes with an E8M0
 *     exponent per group of 32 along K (spec.md:610, `fp4_e2m1*e8m0[1x32]`). libr4d's layout hook
 *     is the authority on how that is stored (libr4d/r4d_layout.cpp:591-666, mx_enc_layout): the
 *     codes go to WMMA fragment order and the exponents to a [K/32][N] plane. `b_ref` -- each
 *     row's largest exponent, a reduction over the plane rather than a choice -- EXISTS ONLY FOR
 *     FP4, whose epilogue folds against it; fp8 and fp6 codes self-scale and declare no
 *     reference. The arch DECLARES the views; it does not permute.
 *   - THE ACTIVATION SCALE IS ONE f32 PER ROW: `As[m]`, not one per (row, 128 columns). That is why
 *     every op below names a `*_row` op -- quant_act_fp8_row, rmsnorm_quant_fp8_row,
 *     gated_quant_fp8_row -- which is quant_act_fp8 with the group dropped (libr4d/r4d_rows.cpp:
 *     966-975, docs/OPS.md:303-313). There is no `group` parameter because the group IS the whole
 *     row by the op's meaning, and the constraint system has no param-eq-param predicate, so
 *     `group == n` is a new op rather than a parameter on the old one.
 *
 * EVERYTHING ELSE -- the shapes of the activations, the fusion of the residual add, the all-reduce
 * fold, the bf16 fallback -- is rad_fp8.h's, restated here only where a name has to change. The
 * all-reduce helpers (ar_wire_is_exact, ar_fused_ok, kArWire6Chunk, ArFold, ArOut, ar_taken) are
 * reused by inclusion rather than copied: rad_fp8.h has an include guard, and a second copy of the
 * wire rule is a second answer to one question, which is what spec §2.3 exists to prevent.
 */

#include "rad_arch.h"
#include "rad_fp8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

/* The scale block's width along K. A property of the FORMAT (`fp4_e2m1*e8m0[1x32]`), compiled
 * into libr4d as R4D_MXFP4_GROUP and carried by its constraint rows as C_DIV("K", 32), so a weight
 * quantised at any other width resolves to no kernel and says so by name. fc, along K the fp4
 * PACKING is 16 -- two e2m1 codes a byte -- but the exponent plane needs whole 32-wide groups, so
 * the group is the binding number. Named because four files have to agree about it. RAD_MX_GROUP
 * itself is declared once, in rad_fp8.h (included above); this file only uses it. */
/* The arch cannot include libr4d's r4d_args.h, so bind the value the same way the MX kernels do:
 * a compile error if the format width ever moves. Keep this in step with R4D_MXFP4_GROUP. */
static_assert(RAD_MX_GROUP == 32, "OCP-MX group width changed; update R4D_MXFP4_GROUP and the kernels");

/* Columns of the exponent plane for a K-wide weight, and rows for an N-row one. The E8M0 plane is
 * block[1 x 32], so its own extents are (N rows) x (ceil(K/32) columns) -- NOT the [K/32][N] the
 * kernel reads. The canonical orientation is what `decl_view` must be given: core/build/
 * rad_builder.cpp:193-240 holds a single-plane view to its plane's canonical extents and refuses
 * a transposed declaration with the plane named. rad-convert/relayout transposes afterwards. */
inline int64_t mx_blocks(int64_t n) { return (n + RAD_MX_GROUP - 1) / RAD_MX_GROUP; }

/* An activation and its fp8 twin. This IS ActFP8's activation side, with two differences that both
 * come from the MX GEMM's epilogue:
 *
 *   - `s` IS [max_tok, 1], ONE SCALE PER ROW. ActFP8's `s` is [max_tok, ceil(n/128)]; every MX
 *     kernel carries a single f32 per token and has no per-128 group on the activation side
 *     (libr4d/r4d_gemm_mxfp4a8_nt_m64.hip reads `As[m]`, and its fp8/fp6 siblings read it the same).
 *   - THERE IS NO INT8 PAIR. MX is an OCP-MX codebook against fp8 activations; LinearMX has no i8 branch,
 *     so q8/s8 and q8_fed have no meaning here and are not declared. Leaving them off is not a
 *     simplification: an int8 pair nothing reads is an arena plane the planner still charges. */
struct ActMX {
    rad_buf x = 0;   /* [max_tok, n]  bf16 */
    rad_buf q = 0;   /* [max_tok, n]  e4m3 */
    rad_buf s = 0;   /* [max_tok, 1]  f32  -- ONE per row */

    /* Declare all three at once; `x` is passed in because the block already had it. `rows` is not
     * optional for a plane that is not one row per token, for ActFP8::declare_qs's reason: the
     * default is right for an activation and silently wrong for a routed expert plane, and the
     * overrun would surface at a later reader. */
    int declare_qs(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t n,
                   int64_t rows = 0) {
        /* A bf16 model's linears read `x` and nothing else (Geom::w_bf16), so there are no codes
         * to declare. */
        if (g.w_bf16) { q = s = 0; return RAD_OK; }
        const int64_t m = rows > 0 ? rows : g.max_tok;
        q = decl_b(b, nm.f("%s.q", name), RAD_F8E4M3, { m, n });
        s = decl_b(b, nm.f("%s.s", name), RAD_F32,    { m, 1 });
        return (q && s) ? RAD_OK : RAD_E_INVAL;
    }
};

/* ================================================================== the activation quantiser */
struct QuantMX {
    ActMX   a{};
    int64_t n = 0;
    /* THE MATVEC BAND IS DECLINED, SO THERE IS NO min_rows BUT ONE. rad_fp8.h's QuantFP8 declines
     * to run below fp8_matvec_rows because the fp8a16 matvec reads the bf16 activation directly and
     * would leave the codes unread. There is no bf16 matvec band here -- an MX kernel always
     * reads `a.q`/`a.s` -- so the quantiser is read at every T and min_rows is 1. The field is kept
     * so step() has the same shape as its twin and a future band can narrow it without moving the
     * call. */
    int64_t min_rows = 1;
    rad_op  op = 0;

    int  declare(RadBuilder* b, const Geom& g, const ActMX& act, int64_t width);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int QuantMX::declare(RadBuilder* b, const Geom& g, const ActMX& act, int64_t width) {
    a = act;
    n = width;
    /* Nothing to quantise for a bf16 model: no op, and step() issues nothing. */
    if (g.w_bf16) { op = 0; return a.x ? RAD_OK : RAD_E_INVAL; }
    if (!a.x || !a.q || !a.s || n <= 0) return RAD_E_INVAL;
    /* `dtype` is the INPUT's, bf16: the output width is the op name's, and libr4d's row constrains
     * it to bf16 for that reason while ref reads each operand's own dtype. Pretty importantly, no
     * `group`: see the file header. */
    op = rw(b, RAD_OP(b, "quant_act_fp8_row",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                          RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
           { a.x }, { a.q, a.s });
    return RAD_OK;
}

inline void QuantMX::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    if (rows <= 0) rows = T - r0;
    if (rows < min_rows) return;
    /* Pitch 1: `s` has one column. In ActFP8 this is n / RAD_FP8_BLOCK. */
    RAD_ISSUE(c, op, brow_slice(a.x, r0, rows, n), brow_slice(a.q, r0, rows, n),
              brow_slice(a.s, r0, rows, 1));
}

/* ================================== the quantiser folded into the op that PRODUCES the row
 *
 * The same two fusions rad_fp8.h's NormQuantFP8/GateQuantFP8 make, at the same band, with the
 * per-row scale and the `*_row` op names.
 *
 * THE AR FOLD IS OFF. The two-rank byte-identity test passes at the decode shape
 * (`ar_qr_two_rank`, `--case ar_rmsnorm_quant_fp8_row`) AND at a prefill chunk (M=2048, added
 * below), so the KERNEL is correct. But when the fold is enabled the served engine faults at
 * prefill -- `ar_rmsnorm_quant_fp8_row` is issued at the trunk's M=2048 norm and the launch
 * faults (HSA_STATUS_ERROR_MEMORY_FAULT, device queue stuck in ar_rmsnorm_quant_fp8_row_kernel),
 * which the engine reports as a selection refusal and aborts the step, failing every prefill
 * >256 tokens. That is an engine-integration/scratch-lifecycle bug the standalone harness does
 * not reproduce, so the fold stays OFF and the MX blocks forward `kArNone`: the standalone
 * collectives are numerically correct and this is the safe path. Re-enable only after an e2e
 * prefill run passes with the fold on.
 *
 * This constant also gates the producer-side collective skip in the MX blocks: with it true, a
 * block whose successor's norm absorbs its all-reduce does not issue the standalone collective, and
 * the plugin does not issue the embedding reduction where layer 0's norm takes it. Where the fold
 * is not reachable -- T outside the fused band, a lossy wire this install forbids, a width the
 * wire's chunk does not tile -- `pick` falls back to the pair and the standalone collective is
 * issued, exactly as rad_fp8.h's predicates decide.
 *
 * THE A/B SWITCH. `kMxArFold` is the single source of truth for the fold, read both here (which
 * fused ops are declared and picked) and by the MX blocks / plugin (whether the producer skips its
 * standalone collective). It is a RUNTIME bool so the coordinator can A/B without a rebuild:
 *
 *     RADIANCE_MX_AR_FOLD=1 radiance --model ... serve      # fold on
 *     radiance --model ... serve                            # fold off (default)
 *
 * It must not become `constexpr`: the MX blocks test it in a plain `if`, and a runtime value keeps
 * the declare side, the `pick` side and the producer-skip side reading the SAME answer -- a
 * compile-time constant in one place and an env read in another would let them disagree and either
 * double-reduce or drop a collective. The default stays FALSE (kMxArFoldDefault) until the
 * served-engine prefill fault is fixed; see libr4d/r4d_ar_entry.hip's fused-row shim. */
inline constexpr bool kMxArFoldDefault = false;
inline bool mx_ar_fold_from_env() {
    const char* v = std::getenv("RADIANCE_MX_AR_FOLD");
    if (!v || !v[0]) return kMxArFoldDefault;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y';
}
/* The name the MX blocks and the plugin read. An inline `const bool` rather than a `constexpr` so a
 * runtime value reaches every translation unit (the blocks are headers) identically; every read is
 * at declare/step time, well after its dynamic initialization, and the env is read exactly once. */
inline const bool kMxArFold = mx_ar_fold_from_env();


struct NormQuantMX {
    ActMX      a{};
    rad_weight w = 0;
    rad_buf    src = 0;
    rad_buf    res = 0;
    rad_op     op = 0;
    /* ...AND THE SAME OP WITH THE PRECEDING ALL-REDUCE FOLDED INTO IT. Declared only where the
     * predecessor has an all-reduce this norm is the sole consumer of, which is exactly where
     * `residual` is non-null. BOTH are declared and the wire is chosen at step time, because the
     * wire depends on T and each fused kernel serves one; see rad_fp8.h:391-402. */
    rad_op     op_arn = 0;
    /* ...and the ROTATED 6-BIT wire's form, declared separately because the wire is a
     * declaration-time parameter of the op; see rad_fp8.h:398-403. */
    rad_op     op_arn6 = 0;
    Geom       gm{};
    /* The row width, kept so a ROW SLICE can turn a row offset into an element offset. */
    int64_t    n = 0;

    int  declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x, const ActMX& act,
                 int64_t n, rad_buf residual = 0, int64_t max_rows = 0, int fuse_ar = kArNone);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
    rad_op pick(int64_t T) const {
        if (op_arn && ar_wire_is_exact(gm, T)) return op_arn;
        if (op_arn6 && ar_fused_ok(gm, T)) return op_arn6;
        return op;
    }
};

inline int NormQuantMX::declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x,
                                const ActMX& act, int64_t n, rad_buf residual,
                                int64_t max_rows, int fuse_ar) {
    a = act;
    w = gain;
    src = x;
    res = residual;
    gm = g;
    this->n = n;                    /* the parameter shadows the member; the member is the pitch */
    /* A bf16 model of this family takes every block input from its gated residual (hc_read) and
     * declares no block norm; refused by name rather than declared as a quantiser nothing reads,
     * exactly as NormQuantFP8 does (rad_fp8.h:431-438). */
    if (g.w_bf16) {
        fprintf(stderr, "radiance: a fused norm and quantiser was declared on a model whose "
                        "linears are bf16; its blocks take a prepared input instead\n");
        return RAD_E_UNSUPPORTED;
    }
    if (!src || !a.x || !a.q || !a.s || !w || n <= 0) return RAD_E_INVAL;
    /* The per-row form: no `group`, and `scale` is [M, 1] rather than [M, ceil(n/128)]. */
    op = RAD_OP(b, "rmsnorm_quant_fp8_row",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_F64("eps", g.eps),
                           RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                RAD_WEIGHTS(w));
    if (!op) return RAD_E_INVAL;
    /* Not rw()'s initializer_lists: the sets are conditional and a braced list in a ternary has no
     * lifetime extension. `residual` is read AND written, so it is in both (rad_fp8.h:447-454). */
    rad_buf rd[2] = { src, res };
    rad_buf wr[4] = { a.q, a.s, a.x, res };
    rad_op_reads (b, op, rd, res ? 2 : 1);
    rad_op_writes(b, op, wr, res ? 4 : 3);

    /* THE FUSED FORM, ON WHERE THE PREDECESSOR HAS A COLLECTIVE (see kMxArFold). a caller may
     * pass any fuse_ar; g.world == 2 gates it, and the fused op is declared only there. The wire
     * predicates are rad_fp8.h's. */
    if (kMxArFold && fuse_ar && g.world == 2) {
        op_arn = RAD_OP(b, "ar_rmsnorm_quant_fp8_row",
                        RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                   RAD_INT("n", n), RAD_F64("eps", g.eps),
                                   RAD_INT("world_size", g.world),
                                   RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                        RAD_WEIGHTS(w));
        if (op_arn) {
            rad_buf wr_ar[4] = { a.q, a.s, a.x, res ? res : src };
            rad_op_reads (b, op_arn, rd, res ? 2 : 1);
            rad_op_writes(b, op_arn, wr_ar, 4);
            /* The compressed wire only where it can be reached and served: a width its 512-element
             * chunk does not tile would be a fatal step-time shape refusal, not a choice not taken
             * (rad_fp8.h:480-483). */
            if (fuse_ar != kArExact && !g.wire_exact && (n % kArWire6Chunk) == 0) {
                op_arn6 = RAD_OP(b, "ar_rmsnorm_quant_fp8_row",
                                 RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                            RAD_INT("n", n), RAD_F64("eps", g.eps),
                                            RAD_INT("world_size", g.world),
                                            RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                            RAD_INT("wire", 6)),
                                 RAD_WEIGHTS(w));
                if (op_arn6) {
                    rad_op_reads (b, op_arn6, rd, res ? 2 : 1);
                    rad_op_writes(b, op_arn6, wr_ar, 4);
                }
            }
        }
    }
    return RAD_OK;
}

/* `r0`/`rows` narrow it to a ROW RANGE; at offset 0 `brow_slice` builds exactly the whole-step
 * operand, so the unsliced path is unchanged by construction (rad_fp8.h:501-508). The `s` pitch is
 * 1, where NormQuantFP8's is n / RAD_FP8_BLOCK. */
inline void NormQuantMX::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE_N(c, pick(rows), rows, brow_slice(src, r0, rows, n),
                res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(w),
                brow_slice(a.q, r0, rows, n),
                brow_slice(a.s, r0, rows, 1),
                brow_slice(a.x, r0, rows, n));
}

/* `act` is the vocabulary's name for the gate: "silu" is down_proj's input, "sigmoid" a gated
 * attention o_proj's. `want_x` mirrors GateQuantFP8's exactly -- a down_proj that reads codes does
 * not need the bf16 product -- and so does the bf16 `silu_mul` fallback. There is no `rot` here:
 * the rotated 4-bit expert grid is rad_fp8.h's, not MX's. */
struct GateQuantMX {
    ActMX   a{};
    rad_op  op = 0;
    bool    want_x = true;
    int64_t n = 0;
    bool    bf16 = false;

    int  declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActMX& act, int64_t n,
                 const char* act_name = "silu", int64_t max_rows = 0, bool bf16_out = true);
    void step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int GateQuantMX::declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActMX& act,
                                int64_t n, const char* act_name, int64_t max_rows,
                                bool bf16_out) {
    a = act;
    want_x = bf16_out;
    this->n = n;
    /* A bf16 model's down projection reads the product itself, so the producer is `silu_mul` and
     * writes the bf16 plane alone (rad_fp8.h:552-568). */
    bf16 = g.w_bf16;
    if (bf16) {
        if (!gate_up || !a.x || n <= 0 || std::strcmp(act_name, "silu") != 0) {
            fprintf(stderr, "radiance: a bf16 gate is silu over [gate|up] into a bf16 plane; "
                            "asked for '%s'\n", act_name);
            return RAD_E_INVAL;
        }
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    if (!gate_up || !a.x || !a.q || !a.s || n <= 0) return RAD_E_INVAL;
    /* One op with the bf16 product in the write list or without it, chosen at declare so the plan
     * says what actually happens (rad_fp8.h:523-529). */
    op = want_x
        ? rw(b, RAD_OP(b, "gated_quant_fp8_row",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_STR("dtype", g.dtype), RAD_STR("act", act_name)),
                RAD_NOWEIGHTS),
            { gate_up }, { a.q, a.s, a.x })
        : rw(b, RAD_OP(b, "gated_quant_fp8_row",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_STR("dtype", g.dtype), RAD_STR("act", act_name)),
                RAD_NOWEIGHTS),
            { gate_up }, { a.q, a.s });
    return op ? RAD_OK : RAD_E_INVAL;
}

inline void GateQuantMX::step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0,
                              int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n), brow_slice(a.x, r0, rows, n));
        return;
    }
    /* `gate_up` is 2n wide because it is [gate|up]; `s` is one column. */
    RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                brow_slice(a.q, r0, rows, n),
                brow_slice(a.s, r0, rows, 1),
                want_x ? brow_slice(a.x, r0, rows, n) : RAD_NONE);
}

/* ================================================================== one MX linear */
struct LinearMX {
    /* THE WEIGHT IS TWO OR THREE VIEWS OF ONE LOGICAL WEIGHT, and they are operands of one
     * gemm_nt_q (oGemmNtQ positions b, b_scale, b_ref; libr4d/r4d_rows.cpp:689-707):
     *
     *   w    the codebook's codes,          [N, K]            selector "codes"
     *        (RAD_FP4E2M1 / RAD_F8E4M3 / RAD_FP6E2M3, the codebook mx_dtype_str names)
     *   ws   the E8M0 block exponent plane, [N, K/32]        selector "scale", RAD_E8M0
     *   wref each row's reference byte,     [N, K/32] decl.   selector "scale", RAD_E8M0
     *        -- DECLARED FOR FP4 ONLY: fp8 and fp6 codes self-scale, and b_ref is absent
     *
     * `ws` and `wref` select the SAME canonical plane -- mx_enc_layout:605-608 derives both from
     * "scale" -- and are declared identically; what differs is the OPERAND the op takes them as,
     * which is what makes the layout hook store one as [K/32][N] (Q_BSCALE) and the other as the
     * per-row [N] reference (Q_BREF). That is why the shapes below are [N, K/32] and NOT the
     * [K/32][N] the kernel reads: core/build/rad_builder.cpp:193-240 holds a single-plane view to
     * the plane's canonical extents, and a transposed declaration is refused with the plane named.
     * The transposition is the relayout's, not the arch's. */
    rad_weight w = 0, ws = 0, wref = 0;
    rad_op     op = 0;
    int64_t    N = 0, K = 0;
    /* A PLAIN bf16 WEIGHT read by `gemm_nt` over the activation's bf16 plane, in a model whose
     * linears are bf16 (Geom::w_bf16) or that keeps one projection exact. */
    bool       bf16 = false;
    /* WHETHER THIS LINEAR READS THE INPUT'S bf16 PLANE, known before its producer is declared --
     * MediaTeX's need, rad_fp8.h:634-639. An MX linear reads the codes, so where an fp8 linear
     * sets this for int8 it does not; only a bf16 weight here reads `x`. */
    bool       reads_x = false;
    bool       mapped  = false;   /* probe() made the name map; declare() does not make it again */

    int  probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
               std::initializer_list<const char*> src);
    /* `base` is the declared name without a suffix; the logical weight is `<base>.weight`, its
     * scale view `<base>.scale`, and the reference byte view `<base>.wref`. `src` is the checkpoint
     * side, one name or several concatenated along dim 0. `span_lo`/`span_hi` is this rank's uneven
     * slice of the split dimension in model units (rows of each row part for RAD_SHARD_ROW, columns
     * for RAD_SHARD_COL), whole 32-column groups. */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 std::initializer_list<const char*> src,
                 const ActMX& a, rad_buf y, int64_t max_rows = 0,
                 std::initializer_list<int64_t> row_parts = {},
                 int64_t span_lo = 0, int64_t span_hi = 0);
    void step(RadCtx* c, const ActMX& a, rad_buf y, int64_t T, int64_t r0 = 0,
              int64_t rows = 0) const;
    /* The same GEMM writing an N-wide column slice of a wider fused buffer (paroquant gate/up and
     * gdn qkv/z): `pitch` is the fused row's width, `col0` the slice's first column. */
    void step_col(RadCtx* c, const ActMX& a, int64_t T, int64_t r0, int64_t rows, rad_buf fused,
                  int64_t pitch, int64_t col0) const;
};

/* The MX weight LinearMX serves: codes of one MX codebook -- fp4 e2m1, fp8 e4m3, fp6 e2m3 -- at
 * block[1 x 1], one E8M0 plane at block[1 x 32], no transform -- which is exactly what libr4d's
 * mx_enc_layout family accepts (r4d_layout.cpp:602). The plane order is the encoding's --
 * `rad_enc_affine(<code>, RAD_E8M0, 1, 32)` -- so plane 0 is the codes and plane 1 is the exponent
 * plane, which is what the field checks below read. The answer is the KERNEL's dtype string: the
 * codebook also picks the views, because fp4's epilogue folds against a per-row reference (three
 * views, `wref`) while fp8 and fp6 codes self-scale (two views, no b_ref operand). */
inline const char* mx_dtype_str(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine")) return nullptr;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (!sc || sc->dtype != RAD_E8M0 || sc->block[0] != 1 || sc->block[1] != RAD_MX_GROUP)
        return nullptr;
    if (e.plane[0].block[1] != 1) return nullptr;
    /* A TRANSFORM IS ADMITTED ONLY WHERE A PRODUCER EXISTS TO UNDO IT. A weight stored rotated --
     * the ParoQuant path-A container, `givens` over fp6 codes with the pairs/angle/t_scale tables as
     * trailing planes -- is served ONLY by the paroquant arch, which defines QWEN35_PAROQUANT and
     * computes the activation the GEMM must read. Every other arch refuses a transform exactly as
     * before, so teaching the layout hook to lay out a rotated weight cannot leak into them. */
#ifdef QWEN35_PAROQUANT
    const bool rot_ok = e.transform[0] == 0
                            ? e.n_planes == 2
                            : (!std::strcmp(e.transform, "givens") && e.n_planes > 2 &&
                               e.plane[0].dtype == RAD_FP6E2M3);
#else
    const bool rot_ok = e.transform[0] == 0 && e.n_planes == 2;
#endif
    if (!rot_ok) return nullptr;
    switch (e.plane[0].dtype) {
        case RAD_FP4E2M1: return "mxfp4a8";
        case RAD_F8E4M3:  return "mxfp8a8";
        case RAD_FP6E2M3: return "mxfp6a8";
        default:          return nullptr;
    }
}

/* The predicate for callers outside this file; declare's own branch reads mx_dtype_str directly,
 * because that string is the operand it needs, not just the yes. */
inline bool linear_is_mx(const RadEncoding& e) { return mx_dtype_str(e) != nullptr; }

inline int LinearMX::probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                           std::initializer_list<const char*> src) {
    if (src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    mapped = true;
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;
    /* An MX weight reads the codes, so it does NOT set reads_x -- unlike fp8's int8 case. */
    bf16    = g.w_bf16 || plain;
    reads_x = g.w_bf16 || plain;
    return RAD_OK;
}

inline int LinearMX::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                             int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                             std::initializer_list<const char*> src,
                             const ActMX& a, rad_buf y, int64_t max_rows,
                             std::initializer_list<int64_t> row_parts,
                             int64_t span_lo, int64_t span_hi) {
    N = n_out;
    K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* THE NAME MAP FIRST: it is how the checkpoint or the recipe is asked what the weight is, and
     * that answer picks the branch below. */
    const char* wn = nm.f("%s.weight", base);
    if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;

    /* ---- A bf16 WEIGHT, `gemm_nt` over `a.x`, exactly as LinearFP8 does it: row parts and an
     * uneven slice apply to it as they would to the codes, with no block to respect. */
    bf16 = g.w_bf16 || plain;
    if (bf16) {
        if (known && !plain) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16 -- a "
                            "bf16 container is served bf16 throughout\n", wn, en);
            return RAD_E_DTYPE;
        }
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is plain bf16 and its input has no bf16 plane in this "
                            "model; quantise it with its neighbours\n", wn);
            return RAD_E_DTYPE;
        }
        int64_t parts[RAD_MAX_ROW_PARTS] = {0};
        int     np = 0;
        for (int64_t p : row_parts) {
            if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
            parts[np++] = p;
        }
        w = decl_w_n(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, parts, np);
        if (!w) return RAD_E_INVAL;
        if (span_hi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        op = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w)),
               { a.x }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- AN MX WEIGHT. When nothing is known (a builder with no model behind it) this is still
     * the branch, and the codebook it declares is fp4's -- an MX linear is what this file is, and
     * a declaration against nothing is how a test or a synthetic tool exercises it. A known
     * encoding that is neither plain bf16 nor an MX codebook is refused BY NAME -- a block-fp8
     * container (group 128, not 32) served by the MX plugin would otherwise declare a quantised
     * GEMM over planes its layout hooks cannot lay out and produce fluent wrong text. */
    const char* mxd = known ? mx_dtype_str(enc) : "mxfp4a8";
    if (!mxd) {
        char en[256];
        rad_enc_format(&enc, en, sizeof en);
        fprintf(stderr, "radiance: '%s' is %s, and this model's linears are OCP-MX (an affine "
                        "fp4_e2m1, fp8_e4m3, or fp6_e2m3 code times e8m0[1x32] weight); quantise "
                        "it as MX or use the fp8 architecture\n", wn, en);
        return RAD_E_DTYPE;
    }
    /* THE PLUGIN'S QUANT id PROMISES THE CODEBOOK. The op below follows the container's own
     * encoding, so a container that claims this plugin's quant id over a different MX codebook
     * would be SERVED -- consistently wrong, and mislabelled by every message the plugin prints.
     * A plugin that names its quant id (the qwen35 TUs define QWEN35_QUANT) therefore refuses the
     * mismatch by name; a common-header user with no quant id declares what the container holds. */
#if defined(QWEN35_PAROQUANT)
    /* The paroquant arch claims a rotated fp6 weight under its own quant id; the only codebook it
     * serves is e2m3 (mx_dtype_str refuses a transform on anything else). */
    if (known && std::strcmp(mxd, "mxfp6a8") != 0) {
        fprintf(stderr, "radiance: '%s' carries %s codes, which the paroquant arch (quant id \"%s\") "
                        "does not serve; a paroquant weight is e2m3 under an e8m0[1x32] plane\n",
                wn, mxd, QWEN35_QUANT);
        return RAD_E_DTYPE;
    }
#elif defined(QWEN35_QUANT)
    if (known && std::strcmp(mxd, QWEN35_QUANT "a8") != 0) {
        fprintf(stderr, "radiance: '%s' carries %s codes, which this plugin's quant id \"%s\" "
                        "does not name; convert it under the quant id its codebook matches\n",
                wn, mxd, QWEN35_QUANT);
        return RAD_E_DTYPE;
    }
#endif
    /* The codebook picks the code view's dtype and whether a reference view exists: fp4 folds
     * `b_ref` into its epilogue, fp8 and fp6 codes self-scale. */
    const int  code_dt = (std::strcmp(mxd, "mxfp8a8") == 0)   ? RAD_F8E4M3
                         : (std::strcmp(mxd, "mxfp6a8") == 0) ? RAD_FP6E2M3
                                                              : RAD_FP4E2M1;
    const bool folds   = (code_dt == RAD_FP4E2M1);

    /* THE GEOMETRY, checked once per weight with the weight named rather than as a kernel miss.
     * K % 32 is the exponent group's width, and it also makes the packings whole -- fp4's 2 codes
     * a byte and fp6's 4 codes per 3 bytes, both of which a 32-group divides; N % 16 is the
     * kernel's output tile (cGemmNtMxfp4A8 and its fp8/fp6 siblings carry C_DIV("N", 16)).
     * These are PER-RANK extents, so this is also the check that a tensor-parallel width does not
     * cut a group. */
    if (K % RAD_MX_GROUP || N % 16) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank, which is not whole 32-wide "
                        "exponent groups with N in 16s. An MX weight cannot be split inside a "
                        "group; use a bf16 linear here or a world_size that divides cleanly.\n",
                base, (long long)N, (long long)K);
        return RAD_E_INVAL;
    }

    /* ROW PARTS AND THE EXPONENT PLANE, and here the canonical orientation is what makes the MX
     * case expressible at all. The E8M0 plane's OWN extents are [N, K/32] -- N is dim 0, not the
     * [K/32][N] the kernel reads -- and its block along N is 1, so a part of p rows of the weight
     * is a part of p rows of the plane: sparts == wparts, with no /32. The [K/32][N] layout in the
     * kernel is a transpose the relayout applies AFTER slicing, so a stacked [gate|up] whose parts
     * are 1024/3072 rows is 1024/3072 plane rows, and the N-parts land correctly. (Had the plane
     * really been declared [K/32][N], row_parts -- which core/format/share.cpp applies to dim 0 --
     * could not express N-parts and the stacked case would have to be refused by name. It does not
     * have to be.) */
    int64_t wparts[RAD_MAX_ROW_PARTS] = {0}, sparts[RAD_MAX_ROW_PARTS] = {0};
    int     np = 0;
    for (int64_t p : row_parts) {
        if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
        wparts[np] = p;
        sparts[np] = p;                 /* the plane's rows ARE the weight's rows; block[0] == 1 */
        ++np;
    }

    const char* sn = nm.f("%s.scale", base);
    const char* rn = nm.f("%s.wref",  base);
    /* THE VIEWS OF ONE LOGICAL WEIGHT. wref selects the same plane as ws; the operand it is taken
     * as (Q_BREF) is what makes its stored form the per-row reference, so both are declared at the
     * plane's canonical extents and neither is transposed here. fp8 and fp6 codes self-scale: no
     * reference view exists for them, and their layout hooks refuse a b_ref operand. */
    w  = decl_view(b, wn, nullptr, "codes", code_dt, { N, K },
                   RAD_ACCESS_PER_TOKEN, shard, grp, wparts, np);
    ws = decl_view(b, sn, wn, "scale", RAD_E8M0, { N, mx_blocks(K) },
                   RAD_ACCESS_PER_TOKEN, shard, grp, sparts, np);
    if (folds)
        wref = decl_view(b, rn, wn, "scale", RAD_E8M0, { N, mx_blocks(K) },
                          RAD_ACCESS_PER_TOKEN, shard, grp, sparts, np);
    if (!w || !ws || (folds && !wref)) return RAD_E_INVAL;
    /* AN UNEVEN SLICE, and the exponent plane takes the same slice. A ROW slice is rows of the
     * plane as it is; a COL slice is columns, and 32 weight columns are one plane column, so the
     * slice is in whole groups. Both ws and wref follow the same rule as the weight. */
    if (span_hi > 0) {
        const bool col = shard == RAD_SHARD_COL;
        if (col && (span_lo % RAD_MX_GROUP || span_hi % RAD_MX_GROUP)) {
            fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                            "%d-wide exponent group.\n", base, (long long)span_lo,
                    (long long)span_hi, RAD_MX_GROUP);
            return RAD_E_INVAL;
        }
        const int64_t slo = col ? span_lo / RAD_MX_GROUP : span_lo;
        const int64_t shi = col ? span_hi / RAD_MX_GROUP : span_hi;
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        RAD_ARCH_TRY(rad_weight_shard_span(b, ws, slo, shi));
        if (folds) RAD_ARCH_TRY(rad_weight_shard_span(b, wref, slo, shi));
    }

    const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
    /* `dtype` is the codebook's kernel string (mxd, picked by the codes' dtype above; libr4d's
     * rows are the authority on the names), and the weight list follows the same choice: fp4
     * carries three views, fp8 and fp6 two -- `b_ref` is RAD_OPTIONAL in oGemmNtQ, so an absent
     * reference is a declared absence, not a shape the kernel has to discover. */
    /* `group` IS REQUIRED BY THE SCHEMA AND ALSO VALIDATED BY THE LAYOUT HOOK, so it is not a
     * formality: pGemmNtQ declares it RAD_REQUIRED (libr4d/r4d_rows.cpp:676-682,
     * libref/ref_registry.cpp:236) and mx_enc_layout rejects any value but 32
     * (libr4d/r4d_layout.cpp:599-601, wgeom reads the op's group at :105). It is passed as the
     * format's 32. */
    rad_weight wts[3] = { w, ws, wref };   /* wref is 0 where folds is false and never counted */
    op = rw(b, rad_decl_op(b, "gemm_nt_q",
               RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                          RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", mxd)),
               wts, folds ? 3 : 2),
           { a.q, a.s }, { y });
    return op ? RAD_OK : RAD_E_INVAL;
}

/* THE OPERAND LIST IS SEVEN LONG AND PASSED IN FULL, positionally: a, a_scale, b, b_scale, y,
 * a_sum, b_ref. `a_sum` is RAD_NONE -- it belongs to the asymmetric integer grids and an MX grid
 * has none -- and `b_ref` is the reference view, present only where one was declared: the fp4
 * epilogue reads it, the fp8 and fp6 epilogues scale by the group exponents alone and take
 * RAD_NONE there. The activation scale's pitch is 1, not K / 32: the scale is one per row, and
 * the exponent plane is the WEIGHT's, indexed by the GEMM itself. */
inline void LinearMX::step(RadCtx* c, const ActMX& a, rad_buf y, int64_t T, int64_t r0,
                           int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_W(w),
                    brow_slice(y, r0, rows, N));
        return;
    }
    RAD_ISSUE_N(c, op, rows,
                brow_slice(a.q, r0, rows, K),
                brow_slice(a.s, r0, rows, 1),
                RAD_W(w), RAD_W(ws),
                brow_slice(y, r0, rows, N),
                RAD_NONE,
                wref ? RAD_W(wref) : RAD_NONE);
}

/* ONE NATIVE LINEAR'S GEMM INTO A COLUMN OF A FUSED BUFFER. gate and up are ONE stacked weight for
 * an MX container -- one rotation -- but a ParoQuant container's gate and up carry different
 * rotations, so the paroquant arch declares them as two GEMMs writing the two halves of the same
 * [M, 2N] buffer that GateQuantMX already reads. `col0` is the half's first column and `2N` its
 * row pitch, so `bcol_at` addresses exactly the N-wide slice the GEMM writes. Every other caller
 * uses `step`, which always writes at column 0. */
inline void LinearMX::step_col(RadCtx* c, const ActMX& a, int64_t T, int64_t r0, int64_t rows,
                               rad_buf fused, int64_t pitch, int64_t col0) const {
    if (rows <= 0) rows = T - r0;
    const RadOperand y = bcol_at(fused, r0, pitch, col0, N, rows);
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_W(w), y);
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(a.q, r0, rows, K), brow_slice(a.s, r0, rows, 1),
                RAD_W(w), RAD_W(ws), y, RAD_NONE, wref ? RAD_W(wref) : RAD_NONE);
}

/* ================================ the ParoQuant path-A producer ================================
 *
 * THE ACTIVATION HALF OF A STILL-ROTATED WEIGHT, and the one way the paroquant arch differs from its
 * MX siblings at the op level. A ParoQuant weight is stored rotated -- `w = R(v) . D`, `givens` over
 * e2m3 codes with the pairs/angle/t_scale tables as trailing planes -- so the GEMM that reads the
 * codes `v` must be fed `a = (x / t_scale) . M_dec^T`, not the plain normed activation. This
 * declares that producer: the three tables as weight views of the SAME logical weight the consuming
 * LinearMX reads (so they are cached with it, not fed per call), and `paroquant_quant_fp8_row`
 * (libr4d/r4d_paroquant_quant_fp8_row.hip) writing the rotated E4M3 codes and their per-row scale.
 *
 * ONE PER LINEAR, because the rotation is per weight: gate_proj and up_proj of the same layer share
 * `pairs` but not `theta`/`channel_scales`, so a single shared activation cannot feed both. The
 * producer's output width is the GEMM's K (the linear's input), and under tensor parallelism that K
 * is FULL -- these projections shard on N -- so the tables, indexed by K, are never cut and need no
 * shard. The `enc` is the weight's reading; `stages` comes from `t.pairs`'s own extent, so the arch
 * carries no hard-coded rotation depth. */
struct ParoProducer {
    ActMX      a{};                        /* a.x is the bf16 input; a.q/a.s the producer's output */
    rad_weight pairs = 0, angle = 0, tscale = 0;
    rad_op     op = 0;
    int64_t    n = 0, stages = 0;

    /* `input` is the bf16 buffer the producer rotates (the block's norm output for q/k/v/gate/up,
     * the silu product for down); `out.q`/`out.s` are the Wire's rotated-activation buffers,
     * declared ONCE by the plugin so they are shared across layers. */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, int l, const char* base,
                 const char* owner, rad_buf input, const ActMX& out, int64_t width,
                 const RadEncoding* enc, int shard = RAD_SHARD_NONE);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int ParoProducer::declare(RadBuilder* b, Names& nm, const Geom& g, int l, const char* base,
                                 const char* owner, rad_buf input, const ActMX& out, int64_t width,
                                 const RadEncoding* enc, int shard) {
    if (g.w_bf16 || !input || !out.q || !out.s || width <= 0) return RAD_E_INVAL;
    a = out;
    a.x = input;
    n = width;
    /* A ROW-PARALLEL WEIGHT CUTS K, and the rotation rides that cut: `width` is this rank's slice,
     * the table views are declared at the same shard so the loader column-slices them, and
     * `k_base` — this rank's first column of the FULL K the pair values index — rebases them in the
     * kernel. An even split puts the base at rank*width. The groups are 128 wide and the shards are
     * whole groups, so a rotation never straddles a rank. */
    const int k_base = shard == RAD_SHARD_COL ? (int)(g.rank * width) : 0;
    const RadEncPlane* pp = enc ? rad_enc_plane(enc, "t.pairs") : nullptr;
    if (!pp || pp->dtype != RAD_I32) return RAD_E_DTYPE;
    stages = pp->extent[0];
    if (stages <= 0 || (width % 128)) return RAD_E_SHAPE;
    pairs = decl_view(b, nm.f("%s.pairs", base), owner, "t.pairs", RAD_I32, { stages, width },
                      RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
    angle = decl_view(b, nm.f("%s.angle", base), owner, "t.angle", RAD_F32,
                      { stages, width / 2 }, RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
    tscale = decl_view(b, nm.f("%s.tscale", base), owner, "t.scale", RAD_F32, { 1, width },
                       RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
    if (!pairs || !angle || !tscale) return RAD_E_INVAL;
    op = rw(b, RAD_OP(b, "paroquant_quant_fp8_row",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width),
                          RAD_INT("stages", stages), RAD_STR("dtype", g.dtype),
                          RAD_INT("k_base", k_base)),
               RAD_WEIGHTS(pairs, angle, tscale)),
           { a.x }, { a.q, a.s });
    return op ? RAD_OK : RAD_E_INVAL;
}

inline void ParoProducer::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    /* DIAGNOSTIC (RADIANCE_PQ_SKIP_PRODUCER): issue nothing, so the GEMM reads a stale activation
     * and the step is numerically wrong -- this exists only to measure how much of the decode cost
     * is the producer launches, never to serve. */
    if (getenv("RADIANCE_PQ_SKIP_PRODUCER")) return;
    if (rows <= 0) rows = T - r0;
    if (rows <= 0) return;
    /* The operand list is the schema's, positionally: x, pairs, angle, t_scale, q, scale. The three
     * tables are the weight operands, passed here as the GEMM passes its own. */
    RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, n), RAD_W(pairs), RAD_W(angle),
                RAD_W(tscale), brow_slice(a.q, r0, rows, n), brow_slice(a.s, r0, rows, 1));
}

/* ============================= the FUSED path-A producers (rad_mx.h's producers minus a launch)
 *
 * The standalone ParoProducer is a launch ON TOP of the norm (or the gated product) that produced its
 * input, and that extra launch per consumer is ~100% of path A's decode gap to Plain-MXFP6. These
 * two declare the fused row op -- `rmsnorm_quant_fp8_row`/`gated_quant_fp8_row` with the Givens
 * rotation and the per-row quantiser folded into the SAME launch, once per consumer partition
 * (libr4d/r4d_paroquant_fused_row.hip). The consumers are the same, the arithmetic is byte-identical
 * (gated by a device selftest), and one block norm/gate launch now feeds all of them.
 *
 * ONE PER DISTINCT INPUT. gate_proj and up_proj of a layer share `pairs` but not `theta`, so a single
 * shared activation cannot feed both -- but both are read from the SAME normed row, so the norm
 * emits BOTH rotations in its own launch. q/k/v are the same. This is the whole of the path-A fix. */

struct ParoFusedNorm {
    struct Part {
        const char*        name  = nullptr;   /* the consumer weight's stem, for the table views */
        const char*        owner = nullptr;   /* the checkpoint owner of those views */
        const RadEncoding* enc   = nullptr;
        ActMX              out{};             /* the rotated-activation buffers this partition writes */
        rad_weight         pairs = 0, angle = 0, tscale = 0;
    };
    std::vector<Part> parts;
    rad_weight wnorm = 0;
    rad_buf    src = 0, res = 0, y = 0;
    rad_op     op = 0;
    int64_t    n = 0, stages = 0;

    /* `x`/`res` are the block's residual stream and the folded-add source; `wnorm` the gain; `y` the
     * OPTIONAL bf16 normed row a consumer may still read (the delta net's a/b gate); `parts` the
     * consumers, in the operand order their tables and outputs are passed. */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, int l, rad_buf x, rad_buf res_,
                 rad_weight wnorm_, rad_buf y_, std::vector<Part> parts_, int64_t width);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int ParoFusedNorm::declare(RadBuilder* b, Names& nm, const Geom& g, int l, rad_buf x,
                                  rad_buf res_, rad_weight wnorm_, rad_buf y_,
                                  std::vector<Part> parts_, int64_t width) {
    if (g.w_bf16 || !x || !wnorm_ || width <= 0 || parts_.empty() || parts_.size() > 3)
        return RAD_E_INVAL;
    src = x; res = res_; wnorm = wnorm_; y = y_; n = width; parts = std::move(parts_);
    for (size_t p = 0; p < parts.size(); ++p) {
        Part& pt = parts[p];
        if (!pt.enc || !pt.out.q || !pt.out.s || !pt.name || !pt.owner) return RAD_E_INVAL;
        const RadEncPlane* pp = rad_enc_plane(pt.enc, "t.pairs");
        if (!pp || pp->dtype != RAD_I32) return RAD_E_DTYPE;
        if (p == 0) stages = pp->extent[0];
        if (stages <= 0 || (width % 128)) return RAD_E_SHAPE;
        pt.pairs = decl_view(b, nm.f("%s.pairs", pt.name), pt.owner, "t.pairs", RAD_I32,
                             { stages, width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        pt.angle = decl_view(b, nm.f("%s.angle", pt.name), pt.owner, "t.angle", RAD_F32,
                             { stages, width / 2 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        pt.tscale = decl_view(b, nm.f("%s.tscale", pt.name), pt.owner, "t.scale", RAD_F32,
                              { 1, width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        if (!pt.pairs || !pt.angle || !pt.tscale) return RAD_E_INVAL;
    }
    const int P = (int)parts.size();
    std::vector<rad_weight> w;
    w.push_back(wnorm);
    for (Part& pt : parts) { w.push_back(pt.pairs); w.push_back(pt.angle); w.push_back(pt.tscale); }
    const RadParam ps[] = {
        RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width), RAD_INT("stages", stages),
        RAD_STR("dtype", g.dtype), RAD_INT("P", P), RAD_INT("k_base", 0),
        RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
        RAD_INT("no_rot", getenv("RADIANCE_PQ_NO_ROT") ? 1 : 0),
    };
    const char* opname = P == 2 ? "paroquant_rmsnorm_quant_fp8_row2"
                                : "paroquant_rmsnorm_quant_fp8_row3";
    op = rad_decl_op(b, opname, ps, (int)(sizeof ps / sizeof ps[0]), w.data(), (int)w.size());
    if (!op) return RAD_E_INVAL;
    const rad_buf rd[2] = { src, res };
    rad_op_reads(b, op, rd, res ? 2 : 1);
    std::vector<rad_buf> wr;
    for (Part& pt : parts) { wr.push_back(pt.out.q); wr.push_back(pt.out.s); }
    if (y) wr.push_back(y);
    rad_op_writes(b, op, wr.data(), (int)wr.size());
    return RAD_OK;
}

inline void ParoFusedNorm::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    /* DIAGNOSTIC: same as ParoProducer::step -- measure the producers' total cost. */
    if (getenv("RADIANCE_PQ_SKIP_PRODUCER")) return;
    if (rows <= 0) rows = T - r0;
    if (rows <= 0) return;
    const int P = (int)parts.size();
    if (P == 2)
        RAD_ISSUE_N(c, op, rows, brow_slice(src, r0, rows, n),
                    res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(wnorm),
                    RAD_W(parts[0].pairs), RAD_W(parts[0].angle), RAD_W(parts[0].tscale),
                    RAD_W(parts[1].pairs), RAD_W(parts[1].angle), RAD_W(parts[1].tscale),
                    brow_slice(parts[0].out.q, r0, rows, n), brow_slice(parts[0].out.s, r0, rows, 1),
                    brow_slice(parts[1].out.q, r0, rows, n), brow_slice(parts[1].out.s, r0, rows, 1),
                    y ? brow_slice(y, r0, rows, n) : RAD_NONE);
    else if (P == 3)
        RAD_ISSUE_N(c, op, rows, brow_slice(src, r0, rows, n),
                    res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(wnorm),
                    RAD_W(parts[0].pairs), RAD_W(parts[0].angle), RAD_W(parts[0].tscale),
                    RAD_W(parts[1].pairs), RAD_W(parts[1].angle), RAD_W(parts[1].tscale),
                    RAD_W(parts[2].pairs), RAD_W(parts[2].angle), RAD_W(parts[2].tscale),
                    brow_slice(parts[0].out.q, r0, rows, n), brow_slice(parts[0].out.s, r0, rows, 1),
                    brow_slice(parts[1].out.q, r0, rows, n), brow_slice(parts[1].out.s, r0, rows, 1),
                    brow_slice(parts[2].out.q, r0, rows, n), brow_slice(parts[2].out.s, r0, rows, 1),
                    y ? brow_slice(y, r0, rows, n) : RAD_NONE);
}

/* The gated twin: silu/sigmoid product -> P rotated activations. Used for the MLP down projection,
 * whose input is the [gate|up] product. P is 1 in this tree. */
struct ParoFusedGated {
    struct Part {
        const char*        name  = nullptr;
        const char*        owner = nullptr;
        const RadEncoding* enc   = nullptr;
        ActMX              out{};
        rad_weight         pairs = 0, angle = 0, tscale = 0;
    };
    std::vector<Part> parts;
    rad_buf    gu = 0, gu2 = 0;            /* gu = gate (or merged [gate|up]); gu2 = up alone */
    rad_op     op = 0;
    int64_t    n = 0, stages = 0;
    const char* act = "silu";

    int  declare(RadBuilder* b, Names& nm, const Geom& g, int l, rad_buf gu_, rad_buf gu2_,
                 std::vector<Part> parts_, int64_t width, const char* act_name = "silu",
                 int shard = RAD_SHARD_NONE);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int ParoFusedGated::declare(RadBuilder* b, Names& nm, const Geom& g, int l, rad_buf gu_,
                                   rad_buf gu2_, std::vector<Part> parts_, int64_t width,
                                   const char* act_name, int shard) {
    if (g.w_bf16 || !gu_ || width <= 0 || parts_.empty() || parts_.size() > 3) return RAD_E_INVAL;
    gu = gu_; gu2 = gu2_; n = width; parts = std::move(parts_); act = act_name;
    /* A ROW-PARALLEL consumer (the down projection) cuts K; its activation rides that cut and the
     * tables are column-sliced and rebased exactly as ParoProducer's. */
    const int k_base = shard == RAD_SHARD_COL ? (int)(g.rank * width) : 0;
    for (size_t p = 0; p < parts.size(); ++p) {
        Part& pt = parts[p];
        if (!pt.enc || !pt.out.q || !pt.out.s || !pt.name || !pt.owner) return RAD_E_INVAL;
        const RadEncPlane* pp = rad_enc_plane(pt.enc, "t.pairs");
        if (!pp || pp->dtype != RAD_I32) return RAD_E_DTYPE;
        if (p == 0) stages = pp->extent[0];
        if (stages <= 0 || (width % 128)) return RAD_E_SHAPE;
        pt.pairs = decl_view(b, nm.f("%s.pairs", pt.name), pt.owner, "t.pairs", RAD_I32,
                             { stages, width }, RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
        pt.angle = decl_view(b, nm.f("%s.angle", pt.name), pt.owner, "t.angle", RAD_F32,
                             { stages, width / 2 }, RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
        pt.tscale = decl_view(b, nm.f("%s.tscale", pt.name), pt.owner, "t.scale", RAD_F32,
                              { 1, width }, RAD_ACCESS_PER_TOKEN, shard, grp_layer(l));
        if (!pt.pairs || !pt.angle || !pt.tscale) return RAD_E_INVAL;
    }
    const int P = (int)parts.size();
    std::vector<rad_weight> w;
    for (Part& pt : parts) { w.push_back(pt.pairs); w.push_back(pt.angle); w.push_back(pt.tscale); }
    const RadParam ps[] = {
        RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width), RAD_INT("stages", stages),
        RAD_STR("dtype", g.dtype), RAD_INT("P", P), RAD_INT("k_base", k_base), RAD_STR("act", act),
    };
    op = rad_decl_op(b, "paroquant_gated_quant_fp8_row1", ps, (int)(sizeof ps / sizeof ps[0]),
                     w.data(), (int)w.size());
    if (!op) return RAD_E_INVAL;
    const rad_buf rd[2] = { gu, gu2 };
    rad_op_reads(b, op, rd, gu2 ? 2 : 1);
    std::vector<rad_buf> wr;
    for (Part& pt : parts) { wr.push_back(pt.out.q); wr.push_back(pt.out.s); }
    rad_op_writes(b, op, wr.data(), (int)wr.size());
    return RAD_OK;
}

inline void ParoFusedGated::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    /* DIAGNOSTIC: same as ParoProducer::step -- measure the producers' total cost. */
    if (getenv("RADIANCE_PQ_SKIP_PRODUCER")) return;
    if (rows <= 0) rows = T - r0;
    if (rows <= 0) return;
    const int P = (int)parts.size();
    const int64_t gwd = gu2 ? n : 2 * n;
    if (P == 1)
        RAD_ISSUE_N(c, op, rows, brow_slice(gu, r0, rows, gwd),
                    RAD_W(parts[0].pairs), RAD_W(parts[0].angle), RAD_W(parts[0].tscale),
                    brow_slice(parts[0].out.q, r0, rows, n), brow_slice(parts[0].out.s, r0, rows, 1),
                    gu2 ? brow_slice(gu2, r0, rows, n) : RAD_NONE);
}

}  /* namespace arch */
}  /* namespace rad */

#endif  /* RAD_MX_H */
