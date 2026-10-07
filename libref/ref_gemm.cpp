/* ref_gemm.cpp -- the GEMM family and the two vocabulary edges.
 *
 *     nt: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * B is stored TRANSPOSED because that is the layout every weight in a transformer already has, so
 * both operands are k-contiguous and the inner loop is a dot product over k.
 *
 * This is the one family where the per-element dtype switch would BE the cost -- one element is
 * read per FLOP -- so the inner loop is templated on the operand dtypes and selected once per
 * launch through a function pointer. Everything else in this plugin pays the switch. The cost of
 * not doing it is not marginal: a single prefill chunk is on the order of 10^11 element loads even
 * for a small model, and a switch on each of them turns minutes of work into hours
 * (docs/IMPLEMENTATION.md).
 *
 * Parallelism is over N, not M. A decode step is M=1 and a verify step is M<=64, so an M-major
 * split would leave all but one core idle on exactly the shape the engine spends its life in; N is
 * thousands in every GEMM a transformer issues. Each thread owns whole output elements, so the
 * summation order does not depend on the thread count.
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* `Mat`, `as_mat`, `row_stride`, `Scales` and `make_scales` live in ref_gemm.h, because the
 * mixture-of-experts GEMM reads its operands and scale streams exactly the same way. */

/* ------------------------------------------------------------------ the unquantised core */
template <uint32_t DA, uint32_t DB>
static void gemm_plain(const Mat& A, const Mat& B, const RadTensor* Y, const RadTensor* bias,
                       int64_t M, int64_t N, int64_t K, float alpha) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const void* bp = B.t->data;
        const int64_t b0 = n * B.rs;
        for (int64_t m = 0; m < M; ++m) {
            const int64_t a0 = m * A.rs;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += Cvt<DA>::ld(A.t->data, a0 + k * A.cs) * Cvt<DB>::ld(bp, b0 + k * B.cs);
            if (bias) acc += ldt(bias, n * laststride(bias));
            stt(Y, m * ys_r + n * ys_c, acc * alpha);
        }
    }
}

/* The same loop with the switch left in. Anything the table below does not name lands here and is
 * correct, just slow -- which is the deal this whole plugin makes. */
static void gemm_plain_generic(const Mat& A, const Mat& B, const RadTensor* Y,
                               const RadTensor* bias, int64_t M, int64_t N, int64_t K,
                               float alpha) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const uint32_t da = A.t->dtype, db = B.t->dtype;
    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const int64_t b0 = n * B.rs;
        for (int64_t m = 0; m < M; ++m) {
            const int64_t a0 = m * A.rs;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k)
                acc += ld_dt(da, A.t->data, a0 + k * A.cs) * ld_dt(db, B.t->data, b0 + k * B.cs);
            if (bias) acc += ldt(bias, n * laststride(bias));
            stt(Y, m * ys_r + n * ys_c, acc * alpha);
        }
    }
}

typedef void (*PlainFn)(const Mat&, const Mat&, const RadTensor*, const RadTensor*,
                        int64_t, int64_t, int64_t, float);

#define REF_PLAIN_ROW(DA)                                            \
    case DA:                                                         \
        switch (db) {                                                \
            case RAD_F32:  return &gemm_plain<DA, RAD_F32>;          \
            case RAD_F16:  return &gemm_plain<DA, RAD_F16>;          \
            case RAD_BF16: return &gemm_plain<DA, RAD_BF16>;         \
            case RAD_I8:   return &gemm_plain<DA, RAD_I8>;           \
            default:       return nullptr;                           \
        }

static PlainFn pick_plain(uint32_t da, uint32_t db) {
    switch (da) {
        REF_PLAIN_ROW(RAD_F32)
        REF_PLAIN_ROW(RAD_F16)
        REF_PLAIN_ROW(RAD_BF16)
        REF_PLAIN_ROW(RAD_I8)
        default: return nullptr;
    }
}
#undef REF_PLAIN_ROW

/* ------------------------------------------------------------------ the quantised core
 * `quant_dot` in ref_gemm.h is the DEFINITION of one quantised product -- how the group sub-sum is
 * formed from the raw codes and in what order the two scales fold onto it. This template is that
 * function unrolled over a whole plane with the loads dispatched at COMPILE time, which is the only
 * difference: `Cvt<DT>::ld` and `ld_dt(DT, ...)` return the same float for every dtype.
 *
 * SO THE TWO CAN DRIFT, and the thing that holds them together is a test rather than a comment:
 * ref_test runs gemm_nt_q_gated -- which goes through `quant_dot` -- against gemm_nt_q followed by
 * gated_quant_fp8, which comes through here, and requires the two byte-identical. */
template <uint32_t DA, uint32_t DB>
static void gemm_quant(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                       const RadTensor* Y, const RadTensor* bias,
                       int64_t M, int64_t N, int64_t K, float alpha) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const int64_t g = SB.group > 0 ? SB.group : K;
    const int64_t ngrp = (K + g - 1) / g;
    const bool coarse_a = !SA.p || SA.group >= g;
    /* Must match `quant_dot` bit for bit: a folded mxfp4 weight folds 2^-dsh into each element as an
     * E4M3 magnitude and applies 2^(Wref-127) once per row, so the row reference is hoisted and the
     * codes folded per group. DB is a compile-time parameter, so the branch folds away for every
     * other weight dtype. */
    constexpr bool kMxfp4 = (DB == RAD_FP4E2M1);
    const bool folded = kMxfp4 && SB.ref != nullptr;

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n) {
        const int64_t b0 = n * B.rs;
        const int wref_b = folded ? SB.wref_at(n) : 0;
        const float rowfac = folded ? SB.row_factor(wref_b) : 1.0f;
        for (int64_t m = 0; m < M; ++m) {
            const int64_t a0 = m * A.rs;
            float acc = 0.0f;
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
                const float sb = SB.p ? (folded ? rowfac : SB.grp(n, gi, wref_b)) : 1.0f;
                const int dsh = folded ? SB.group_dsh(n, gi, wref_b) : 0;
                float sub = 0.0f;
                if (coarse_a) {
                    for (int64_t k = k0; k < k1; ++k) {
                        float bv = Cvt<DB>::ld(B.t->data, b0 + k * B.cs);
                        if (folded) bv = SB.fold_mag(bv, dsh);
                        sub += Cvt<DA>::ld(A.t->data, a0 + k * A.cs) * bv;
                    }
                    acc += sub * (SA.p ? SA.at(m, k0) : 1.0f) * sb;
                } else {
                    for (int64_t k = k0; k < k1; ++k) {
                        float bv = Cvt<DB>::ld(B.t->data, b0 + k * B.cs);
                        if (folded) bv = SB.fold_mag(bv, dsh);
                        sub += Cvt<DA>::ld(A.t->data, a0 + k * A.cs) * SA.at(m, k) * bv;
                    }
                    acc += sub * sb;
                }
            }
            if (bias) acc += ldt(bias, n * laststride(bias));
            stt(Y, m * ys_r + n * ys_c, acc * alpha);
        }
    }
}

static void gemm_quant_generic(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                               const RadTensor* Y, const RadTensor* bias,
                               int64_t M, int64_t N, int64_t K, float alpha) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    #pragma omp parallel for schedule(static)
    for (int64_t n = 0; n < N; ++n)
        for (int64_t m = 0; m < M; ++m) {
            float acc = quant_dot(A, B, SA, SB, m, n, K);
            if (bias) acc += ldt(bias, n * laststride(bias));
            stt(Y, m * ys_r + n * ys_c, acc * alpha);
        }
}

typedef void (*QuantFn)(const Mat&, const Mat&, const Scales&, const Scales&, const RadTensor*,
                        const RadTensor*, int64_t, int64_t, int64_t, float);

#define REF_QUANT_ROW(DA)                                            \
    case DA:                                                         \
        switch (db) {                                                \
            case RAD_I4:    return &gemm_quant<DA, RAD_I4>;          \
            case RAD_I2:    return &gemm_quant<DA, RAD_I2>;          \
            case RAD_FP4E2M1: return &gemm_quant<DA, RAD_FP4E2M1>;       \
            case RAD_FP6E2M3: return &gemm_quant<DA, RAD_FP6E2M3>;       \
            case RAD_I8:    return &gemm_quant<DA, RAD_I8>;          \
            case RAD_F8E4M3:return &gemm_quant<DA, RAD_F8E4M3>;      \
            case RAD_BF16:  return &gemm_quant<DA, RAD_BF16>;        \
            case RAD_F32:   return &gemm_quant<DA, RAD_F32>;         \
            default:        return nullptr;                          \
        }

static QuantFn pick_quant(uint32_t da, uint32_t db) {
    switch (da) {
        REF_QUANT_ROW(RAD_I8)
        REF_QUANT_ROW(RAD_F8E4M3)
        REF_QUANT_ROW(RAD_BF16)
        REF_QUANT_ROW(RAD_F32)
        default: return nullptr;
    }
}
#undef REF_QUANT_ROW

/* gemm_nt_bias's EPILOGUE, applied to what the GEMM stored: an activation, then a residual. Each
 * step reads the stored value and stores its result, so every one of them is rounded to the
 * output's dtype -- which is the unfused chain exactly: a linear layer's output, an activation of
 * it, a residual sum, three tensors. `act` 0 none, 1 exact (erf) GELU, 2 tanh GELU. */
static float gelu_erf(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }
static float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x * x * x)));
}

static void gemm_epilogue(const RadTensor* Y, const RadTensor* res, int act, int64_t M, int64_t N) {
    const int64_t ys_r = row_stride(Y, N), ys_c = laststride(Y);
    const int64_t rs_r = res ? row_stride(res, N) : 0, rs_c = res ? laststride(res) : 0;
    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t o = m * ys_r + n * ys_c;
            if (act == 1) stt(Y, o, gelu_erf(ldt(Y, o)));
            else if (act == 2) stt(Y, o, gelu_tanh(ldt(Y, o)));
            if (res) stt(Y, o, ldt(res, m * rs_r + n * rs_c) + ldt(Y, o));
        }
}

/* Shared entry for gemm_nt / gemm_nt_bias / logits_gemm. `nk_from` names which parameters carry N
 * and K, because logits_gemm spells them n_vocab and n_embd. */
static int gemm_nt_impl(const RadArgs* a, bool bias_operand, const char* nkey, const char* kkey) {
    /* gemm_nt_bias is a, b, bias, res?, y; the other two are a, b, y. */
    const int nt = bias_operand ? 5 : 3;
    if (a->n_t < nt || !a->t[0].data || !a->t[1].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[nt - 1];
    const RadTensor* bias = bias_operand ? t_in(a, 2) : nullptr;
    const RadTensor* res  = bias_operand ? t_in(a, 3) : nullptr;
    static const char* kActs[] = { "none", "gelu", "gelu_tanh" };
    const int act = bias_operand ? p_enum(a, "act", kActs, 3, 0) : 0;

    const int64_t K = p_int(a, kkey, B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, nkey, B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    if (res && numel(res) < M * N) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, M, K), mb = as_mat(B, N, K);
    PlainFn fn = pick_plain(A->dtype, B->dtype);
    if (fn) fn(ma, mb, Y, bias, M, N, K, 1.0f);
    else    gemm_plain_generic(ma, mb, Y, bias, M, N, K, 1.0f);
    if (act || res) gemm_epilogue(Y, res, act, M, N);
    return RAD_OK;
}

static int gemm_ntq_impl(const RadArgs* a, bool bias_operand) {
    const int nt = bias_operand ? 6 : 5;
    if (a->n_t < nt || !a->t[0].data || !a->t[2].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *Y = &a->t[nt - 1];
    const RadTensor* as_ = t_in(a, 1);
    const RadTensor* bs_ = t_in(a, 3);
    const RadTensor* bias = bias_operand ? t_in(a, 4) : nullptr;

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    /* An OCP-MX weight -- e2m1, e2m3 or an e4m3 plane scaled by an e8m0 grid (mxfp4a8, mxfp6a8,
     * mxfp8a8) -- reads its b_scale as exponent bytes. `make_scales` turns that on ONLY when the
     * scale stream is itself u8, so block-scaled fp8 is untouched: the checkpoint's weight_scale_inv
     * is a bf16 plane and lands on the float read whatever this flag says. The no-fold rule is
     * this: e2m3 and e4m3 codes are self-scaling, no `b_ref` is ever attached to them (the shape
     * hook declines to describe one, and `scales_ref` needs the e8m0 grid the e4m3/bf16 pair never
     * forms), and the folded path stays keyed to `db == RAD_FP4E2M1` in quant_dot and gemm_quant.
     * DENSE INT5 (w5a8) rides this flag too, but its b_scale is f16 rather than u8, so `e8m0` stays
     * false and the scale is read as a float; the flag is harmless there and keeps the "a block
     * weight beside an integer/float code" reading in one place. */
    const bool mx = B->dtype == RAD_FP4E2M1 || B->dtype == RAD_FP6E2M3 || B->dtype == RAD_F8E4M3 ||
                    B->dtype == RAD_I5;
    Scales SA = make_scales(as_, M, K, false);
    Scales SB = make_scales(bs_, N, K, mx);
    /* The OCP-MX fold's reference exponent, at operand 6 -- `gemm_nt_q`'s `a_sum`/`b_ref` trailing
     * pair; the bias form has no such slot and no MX kernel. Attaching it makes the weight's block
     * scale the device's saturated 2^(Wref-127-dsh); absent, `make_scales` reads the e8m0 plane
     * directly and this call is exactly what it was before the fold existed. */
    if (!bias_operand) scales_ref(SB, t_in(a, 6), N);
    /* `group` is in the schema and is what the SELECTOR matches on; the loop uses the group the
     * scale stream actually implies. They agree in every well-formed call, and where they do not
     * the stream is the one that describes the bytes. */
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SB.p && SB.group != gp) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, M, K), mb = as_mat(B, N, K);
    QuantFn fn = pick_quant(A->dtype, B->dtype);
    if (fn) fn(ma, mb, SA, SB, Y, bias, M, N, K, 1.0f);
    else    gemm_quant_generic(ma, mb, SA, SB, Y, bias, M, N, K, 1.0f);
    return RAD_OK;
}

}  /* namespace */

int ref_gemm_nt(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, false, "N", "K");
}
int ref_gemm_nt_bias(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, true, "N", "K");
}
int ref_gemm_nt_q(const RadArgs* a, RadStream) {
    return gemm_ntq_impl(a, false);
}
int ref_gemm_nt_q_bias(const RadArgs* a, RadStream) {
    return gemm_ntq_impl(a, true);
}

/* ------------------------------------------------------------------ logits_gemm */
/* The same arithmetic as gemm_nt, and a separate op anyway: lm_head is its own access class
 * (RAD_ACCESS_VOCAB) and the sampler wants to fuse against it, so the two have to be selectable
 * apart even when one kernel serves both (docs/OPS.md). N and K are spelled n_vocab and n_embd. */
int ref_logits_gemm(const RadArgs* a, RadStream) {
    return gemm_nt_impl(a, false, "n_vocab", "n_embd");
}

/* ------------------------------------------------------------------ embed_lookup */
/* A gather, not a GEMM, which is half of why it is its own op. A NEGATIVE token id writes a zero
 * row -- that is the padding convention a batch with a short sequence needs.
 *
 * `vocab_offset` (optional, default 0) makes the table THIS RANK'S SLICE of a vocab-parallel
 * embedding: it covers global ids [vocab_offset, vocab_offset + n_vocab), ids stay global, and one
 * outside the slice writes a zero row so that summing the ranks reconstructs x. See the schema.
 *
 * WITHOUT AN OFFSET AN ID AT OR ABOVE n_vocab STILL FAILS THE LAUNCH, because there it is a caller
 * bug rather than another rank's token and a plausible embedding is the worst answer. With one,
 * that check is not available to this kernel -- it cannot know the global vocabulary -- so it is
 * the caller's, and the sampler and the tokeniser are both already bounded by it. */
int ref_embed_lookup(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *tok = &a->t[0], *wte = &a->t[1], *x = &a->t[2];
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / n_embd;
    if (M <= 0 || numel(tok) < M || numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;

    /* Validated before the loop, because an OpenMP body cannot return.
     *
     * THE GUARD IS ABOUT HOLDING THE WHOLE VOCABULARY, NOT ABOUT THE OFFSET BEING ZERO. An id at
     * or above n_vocab is a caller's mistake only where this rank holds every row; under a
     * vocab-parallel split it is another rank's token, and the loop below writes it as a zero row
     * exactly as it writes the padding id, which is what makes summing the ranks reconstruct x.
     * Rank 0 of such a split has offset 0 -- so a guard keyed on the offset alone would refuse
     * the engine's own operands on every step whose token belongs to the other half, where the
     * device kernel correctly takes the zero-row path at any offset. */
    if (voff == 0 && a->world_size <= 1) {
        for (int64_t m = 0; m < M; ++m) {
            int64_t id = ld_int(tok->dtype, tok->data, offlin(tok, m));
            if (id >= n_vocab) return RAD_E_INVAL;
        }
    }

    const int64_t ws = wte->rank >= 2 ? wte->stride[wte->rank - 2] : n_embd;
    const int64_t wc = wte->rank >= 2 ? wte->stride[wte->rank - 1] : 1;
    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ld_int(tok->dtype, tok->data, offlin(tok, m)) - voff;
        const int64_t xb = rowoff(x, m, n_embd), xs = laststride(x);
        const bool have_row = id >= 0 && id < n_vocab;
        for (int64_t i = 0; i < n_embd; ++i)
            stt(x, xb + i * xs, have_row ? ldt(wte, id * ws + i * wc) : 0.0f);
    }
    return RAD_OK;
}

/* A gather from a QUANTISED table: one scale for the whole thing.
 *
 * WHY IT IS A SEPARATE OP AND NOT A `dtype` ON embed_lookup. The operand list differs -- there is a
 * scale -- and operands are positional, so an op whose count depends on a string means different
 * things to different kernels. That is the same argument `moe_gemm_q` and `gemm_nt_q` make against
 * their unquantised twins, and adding the scale to embed_lookup as an OPTIONAL operand would have
 * moved `x` from slot 2 to slot 3 for every caller in the tree.
 *
 * ONE SCALE FOR THE WHOLE TABLE, and that is the checkpoint's choice rather than a simplification.
 * A hundreds-of-millions-of-rows n-gram embedding ships as E4M3 with a single bf16 `weight_scale`,
 * and the reason that works is structural: E4M3 carries its own exponent, so a row spans very
 * little dynamic range in units of the scale and the reconstruction is insensitive to the scale
 * over octaves. A per-row scale buys a couple of percent there, which is why the format has none.
 * (INT8 in the same byte WOULD gain substantially, because it spends all eight bits on resolution;
 * that is a different format and not this one.)
 */
int ref_embed_lookup_q(const RadArgs* a, RadStream) {
    /* `scale` (operand 2) is optional; the other three must be there. */
    if (a->n_t < 4 || !rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 3)) return RAD_E_INVAL;
    const RadTensor *tok = &a->t[0], *wte = &a->t[1], *sc = &a->t[2], *x = &a->t[3];
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / n_embd;
    if (M <= 0 || numel(tok) < M || numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;
    /* AN ABSENT SCALE IS ONE: a table stored as it ships -- a plain bf16 n-gram embedding read in
     * place from a container -- takes the same op and the same host path as the quantised one,
     * which is what keeps a table too large for any device pool on the mapped tier either way. */
    const float scale = sc->data && numel(sc) >= 1 ? ldt(sc, 0) : 1.0f;

    /* Validated before the loop, because an OpenMP body cannot return. Same rule as the dense
     * gather above: a negative id is padding and writes a zero row, an id past the table is a
     * launch error rather than a read of whatever follows it. */
    if (voff == 0) {
        for (int64_t m = 0; m < M; ++m)
            if (ld_int(tok->dtype, tok->data, offlin(tok, m)) >= n_vocab) return RAD_E_INVAL;
    }

    const int64_t ws = wte->rank >= 2 ? wte->stride[wte->rank - 2] : n_embd;
    const int64_t wc = wte->rank >= 2 ? wte->stride[wte->rank - 1] : 1;
    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ld_int(tok->dtype, tok->data, offlin(tok, m)) - voff;
        const int64_t xb = rowoff(x, m, n_embd), xs = laststride(x);
        const bool have_row = id >= 0 && id < n_vocab;
        for (int64_t i = 0; i < n_embd; ++i)
            stt(x, xb + i * xs, have_row ? ldt(wte, id * ws + i * wc) * scale : 0.0f);
    }
    return RAD_OK;
}
