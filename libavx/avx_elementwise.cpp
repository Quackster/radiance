/* avx_elementwise.cpp -- the elementwise vocabulary: add, mul, the activations, the gated product,
 * the row gather and scatter, cast, softmax and the per-row scale.
 *
 * ============================== THESE ARE MEMORY-BOUND, NOT COMPUTE-BOUND ==============================
 *
 * Every op in this file touches each element a constant number of times and does a handful of
 * flops with it, so the ceiling is the load/store units and then the cache hierarchy. That decides
 * the shape of the code twice over:
 *
 *   THE DTYPE SWITCH IS HOISTED TO THE ROW. libref calls ld_dt() per element, which is a switch on
 *   every load and the reason it is slow -- it is not the scalar arithmetic, it is that no
 *   vectoriser can cross that branch. `in_row`/`out_row` (avx_support.cpp) convert a whole row
 *   once and hand back f32. For an f32 operand they hand back the caller's own pointer and copy
 *   nothing.
 *
 *   THE FUSED FORMS ROUND EXACTLY WHERE libref ROUNDS. This is the part that is not a performance
 *   decision and cannot be traded: `softmax` writes the un-normalised exponentials to `y`, reads
 *   them BACK through the output dtype, and multiplies by 1/sum -- so at bf16 the answer carries
 *   two roundings and not one. `rmsnorm_add` does the same with `residual_out`. A faster
 *   implementation that kept f32 all the way through would be half an ulp BETTER and would
 *   disagree with the oracle and with every device kernel this is checked beside. The round trip
 *   is done in a hot scratch buffer rather than through the output operand, which is the same
 *   arithmetic at cache-resident cost.
 *
 * ============================== THE BROADCAST RULES ARE TWO DIFFERENT OPS ==============================
 *
 * `mul` broadcasts a single ROW down the rows -- a bias's shape. `scale_rows` broadcasts a single
 * COLUMN across them -- a per-token gate's. They are separate ops precisely so that neither has to
 * guess, and the guess would be ambiguous exactly when rows == n. libref's header makes the same
 * point; it is repeated here because the temptation to merge them is real and the failure is a
 * plausible number rather than an error.
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

/* Round a row of f32 through an operand's storage dtype and back, which is what a fused op that
 * writes an intermediate and reads it again computes. A no-op for f32, which is why the f32
 * tolerance for these ops is as tight as it is. */
static inline void roundtrip(uint32_t dt, float* v, int64_t n, void* tmp) {
    if (dt == RAD_F32) return;
    row_from_f32(dt, v, tmp, n);
    row_to_f32(dt, tmp, v, n);
}

/* ------------------------------------------------------------------ add / mul */
/* y = a op b over the output's extent. `b` may carry exactly one row and is then broadcast down
 * the rows; nothing else broadcasts, because silent broadcasting of an arbitrary shape is how a
 * shape bug becomes a plausible number. */
template <bool MUL>
static int elementwise2(const RadArgs* a) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n) return RAD_E_SHAPE;
    const bool bcast = numel(B) == n && rows > 1;
    if (!bcast && numel(B) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sb = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* x = in_row(A, rowoff(A, r, n), n, sa);
        const float* y = in_row(B, bcast ? 0 : rowoff(B, r, n), n, sb);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf u = vf_loadu(x + i), v = vf_loadu(y + i);
            vf_storeu(sy + i, MUL ? vf_mul(u, v) : vf_add(u, v));
        }
        for (; i < n; ++i) sy[i] = MUL ? x[i] * y[i] : x[i] + y[i];
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

AVX_KERNEL(add) { return elementwise2<false>(a); }
AVX_KERNEL(mul) { return elementwise2<true>(a); }

/* y = x * w, `w` a weight of one element or one row (libref's mul_weight). */
AVX_KERNEL(mul_weight) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *W = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const bool scalar = numel(W) == 1;
    if (!scalar && numel(W) != n) return RAD_E_SHAPE;
    /* The gain, widened once: it is the same row for every row of x. */
    float* wrow = (float*)scratch_raw((size_t)n * sizeof(float) + 64);
    if (scalar) { const float v = ldt(W, 0); for (int64_t i = 0; i < n; ++i) wrow[i] = v; }
    else for (int64_t i = 0; i < n; ++i) wrow[i] = ldt(W, i * laststride(W));

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* x = in_row(X, rowoff(X, r, n), n, sa);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_loadu(x + i), vf_loadu(wrow + i)));
        for (; i < n; ++i) sy[i] = x[i] * wrow[i];
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ act_mul */
/* y = act(a) * b over two operands, libref's contract: the activation is rounded to y's dtype
 * before the product, because the unfused pair stores it.
 *
 * silu and gelu_tanh are VECTORISED on vf_exp. An encoder's gated FFN puts this op on every token
 * of every layer, and the scalar libm form was 9% of an EmbeddingGemma 2 step on a 9950X3D -- a
 * per-element tanhf behind 2048 columns. gelu_tanh keeps the reference's own structure,
 *
 *     0.5 g (1 + tanh z),   z = sqrt(2/pi) (g + 0.044715 g^3),   tanh z = (1 - e^(-2z)) / (1 + e^(-2z))
 *
 * and not the algebraically equal g / (1 + e^(-2z)): the two differ by an ulp or two of f32, and
 * after the bf16 rounding the rearranged form moved an EmbeddingGemma 2 embedding measurably
 * further from transformers' (cosine 0.999919 -> 0.999894 on one input; torch computes this same
 * 0.5 g (1 + tanh) in f32). vf_exp saturates rather than overflowing, so the large-|g| limits come
 * out as g and 0. erf has no vector form in this plugin, so gelu stays scalar. */
static inline float act_of(int act, float g) {
    if (act == 0) return g / (1.0f + std::exp(-g));
    if (act == 1) return 0.5f * g * (1.0f + std::erf(g * 0.70710678118654752f));
    if (act == 3) return 1.0f / (1.0f + std::exp(-g));
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * g * (1.0f + std::tanh(k0 * (g + k1 * g * g * g)));
}

/* sy[0..n) = act(g[0..n)) for silu (0) and gelu_tanh (2); the tail is the scalar form. */
static inline void act_row(int act, const float* g, float* sy, int64_t n) {
    int64_t i = 0;
    const vf one = vf_set1(1.0f);
    if (act == 0) {
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(g + i);
            vf_storeu(sy + i, vf_div(x, vf_add(one, vf_exp(vf_sub(vf_zero(), x)))));
        }
    } else if (act == 2) {
        const vf m2k0 = vf_set1(-2.0f * 0.7978845608028654f), k1 = vf_set1(0.044715f);
        const vf half = vf_set1(0.5f);
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(g + i);
            const vf inner = vf_fma(vf_mul(k1, x), vf_mul(x, x), x);     /* g + k1 g^3 */
            const vf e = vf_exp(vf_mul(m2k0, inner));                     /* e^(-2z) */
            const vf t = vf_div(vf_sub(one, e), vf_add(one, e));           /* tanh z */
            vf_storeu(sy + i, vf_mul(vf_mul(half, x), vf_add(one, t)));
        }
    }
    for (; i < n; ++i) sy[i] = act_of(act, g[i]);
}

AVX_KERNEL(act_mul) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n || numel(B) < rows * n) return RAD_E_SHAPE;
    static const char* kActs[] = { "silu", "gelu", "gelu_tanh", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 4, -1);
    if (act < 0) return RAD_E_INVAL;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sa = scratch_f32((size_t)n);
        float* sb = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        const float* g = in_row(A, rowoff(A, r, n), n, sa);
        const float* u = in_row(B, rowoff(B, r, n), n, sb);
        act_row(act, g, sy, n);
        /* sa is free once the activation is taken (g is either sa or the tensor itself); sb may
         * hold u, which the product still reads. */
        roundtrip(Y->dtype, sy, n, sa);
        for (int64_t i = 0; i < n; ++i) sy[i] *= u[i];
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ relu / clamp */
/* y = max(x, 0), and y = min(max(x, lo), hi) with lo and hi WEIGHTS of one element each (a
 * clipped linear's range). Row-wise at each operand's own pitch, like the libref forms. */
AVX_KERNEL(relu) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(sy + i, vf_max(vf_loadu(v + i), vf_zero()));
        for (; i < n; ++i) sy[i] = v[i] > 0.0f ? v[i] : 0.0f;
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

AVX_KERNEL(clamp) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *L = &a->t[1], *H = &a->t[2], *Y = &a->t[3];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0 || numel(L) < 1 || numel(H) < 1) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(X) < rows * n) return RAD_E_SHAPE;
    const float lo = ldt(L, 0), hi = ldt(H, 0);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(X, rowoff(X, r, n), n, sx);
        for (int64_t i = 0; i < n; ++i) {
            float t = v[i] < lo ? lo : v[i];
            sy[i] = t > hi ? hi : t;
        }
        out_row(Y, rowoff(Y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv2d */
/* libref's segment-wise convolution over (time, width), channels innermost. The weight is turned
 * once a launch from PyTorch's [cout, cin, kh, kw] into [kh][kw][cin][cout], so each input value is
 * broadcast against a contiguous run of output channels: the inner loop is a vector FMA over cout
 * rather than a dot product over cin. Same sum (in a different order), one rounding. */
AVX_KERNEL(conv2d) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[4];
    const RadTensor* V = a->t[3].data ? &a->t[3] : nullptr;
    const int64_t W = p_int(a, "width", 0), cin = p_int(a, "cin", 0), cout = p_int(a, "cout", 0);
    const int64_t kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    const int64_t st = p_int(a, "stride", 1), pad = p_int(a, "pad", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1), vst = p_int(a, "valid_stride", 1);
    if (W <= 0 || cin <= 0 || cout <= 0 || kh <= 0 || kw <= 0 || st <= 0 || pad < 0 || cdiv <= 0 ||
        vst <= 0)
        return RAD_E_SHAPE;
    const int64_t Wo = (W + 2 * pad - kw) / st + 1;
    if (Wo <= 0 || numel(Wt) < cout * cin * kh * kw) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t rin = W * cin, rout = Wo * cout;
    const int64_t Min = rows_of(X, rin), Mout = rows_of(Y, rout);

    float* wt = (float*)scratch_raw((size_t)(kh * kw * cin * cout) * sizeof(float) + 64);
    const int64_t ws = laststride(Wt);
    for (int64_t co = 0; co < cout; ++co)
        for (int64_t ci = 0; ci < cin; ++ci)
            for (int64_t i = 0; i < kh; ++i)
                for (int64_t j = 0; j < kw; ++j)
                    wt[((i * kw + j) * cin + ci) * cout + co] =
                        ldt(Wt, (((co * cin + ci) * kh + i) * kw + j) * ws);

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || (c0 / cdiv) % st || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, T = c1 / cdiv - t0;
        if (T <= 0) continue;
        const int64_t To = (T + 2 * pad - kh) / st + 1, o0 = t0 / st;
        if (To <= 0) continue;
        if (t0 + T > Min || o0 + To > Mout) return RAD_E_SHAPE;
        if (V && numel(V) < (t0 + T - 1) * vst + 1) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF(To * rout * cin * kh * kw >= AVX_PAR_MIN)
        for (int64_t to = 0; to < To; ++to) {
            float* acc = scratch_f32((size_t)rout);
            float* sx = scratch_f32_b((size_t)rin);
            for (int64_t e = 0; e < rout; ++e) acc[e] = 0.0f;
            for (int64_t i = 0; i < kh; ++i) {
                const int64_t ti = to * st - pad + i;
                if (ti < 0 || ti >= T) continue;
                const int64_t tg = t0 + ti;
                if (V && ldi(V, offlin(V, tg * vst)) == 0) continue;
                const float* xr = in_row(X, rowoff(X, tg, rin), rin, sx);
                for (int64_t wo = 0; wo < Wo; ++wo) {
                    float* ar = acc + wo * cout;
                    for (int64_t j = 0; j < kw; ++j) {
                        const int64_t wi = wo * st - pad + j;
                        if (wi < 0 || wi >= W) continue;
                        const float* xv = xr + wi * cin;
                        const float* wb = wt + (i * kw + j) * cin * cout;
                        for (int64_t ci = 0; ci < cin; ++ci) {
                            const vf b = vf_set1(xv[ci]);
                            const float* wr = wb + ci * cout;
                            int64_t c = 0;
                            for (; c + VF_N <= cout; c += VF_N)
                                vf_storeu(ar + c, vf_fma(b, vf_loadu(wr + c), vf_loadu(ar + c)));
                            for (; c < cout; ++c) ar[c] += xv[ci] * wr[c];
                        }
                    }
                }
            }
            out_row(Y, rowoff(Y, o0 + to, rout), rout, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv1d_dw */
/* libref's causal depthwise convolution over time, the weight turned to [k][C] once a launch so
 * a row is a vector FMA per tap. */
AVX_KERNEL(conv1d_dw) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Wt = &a->t[1], *cu = &a->t[2], *Y = &a->t[3];
    const int64_t C = p_int(a, "n", 0), K = p_int(a, "k", 0), cdiv = p_int(a, "cu_div", 1);
    if (C <= 0 || K <= 0 || cdiv <= 0 || numel(Wt) < C * K) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t Mx = rows_of(X, C), My = rows_of(Y, C);

    float* wt = (float*)scratch_raw((size_t)(K * C) * sizeof(float) + 64);
    for (int64_t c = 0; c < C; ++c)
        for (int64_t j = 0; j < K; ++j) wt[j * C + c] = ldt(Wt, (c * K + j) * laststride(Wt));

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, t1 = c1 / cdiv;
        if (t1 > Mx || t1 > My) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF((t1 - t0) * C * K >= AVX_PAR_MIN)
        for (int64_t t = t0; t < t1; ++t) {
            float* acc = scratch_f32((size_t)C);
            float* sx = scratch_f32_b((size_t)C);
            for (int64_t c = 0; c < C; ++c) acc[c] = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                const int64_t tt = t - (K - 1) + j;
                if (tt < t0) continue;
                const float* xr = in_row(X, rowoff(X, tt, C), C, sx);
                const float* wr = wt + j * C;
                int64_t c = 0;
                for (; c + VF_N <= C; c += VF_N)
                    vf_storeu(acc + c, vf_fma(vf_loadu(xr + c), vf_loadu(wr + c), vf_loadu(acc + c)));
                for (; c < C; ++c) acc[c] += xr[c] * wr[c];
            }
            out_row(Y, rowoff(Y, t, C), C, acc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rel_sinusoid */
/* libref's table, to the operation: the same f32 steps and the same libm calls. */
AVX_KERNEL(rel_sinusoid) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* Y = &a->t[0];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || n % 2) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const int64_t h = n / 2;
    const double mt = (double)p_f32(a, "max_timescale", 10000.0f);
    const float inc = (float)(std::log(mt / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t d = 0; d < rows; ++d) {
        float* sy = scratch_f32((size_t)n);
        for (int64_t i = 0; i < h; ++i) {
            const float inv = std::exp((float)i * -inc);
            const float ang = (float)d * inv;
            sy[i] = std::sin(ang);
            sy[h + i] = std::cos(ang);
        }
        out_row(Y, rowoff(Y, d, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ seg_mean */
/* One row a cu_seqlens segment: the mean of its rows, summed in double and rounded once, which is
 * libref's arithmetic -- a pooled embedding should not depend on which host kernel pooled it. */
AVX_KERNEL(seg_mean) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *cu = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t S = numel(cu) - 1;
    if (n <= 0 || S <= 0 || numel(Y) < S * n) return RAD_E_SHAPE;
    const int64_t total = ldi(cu, offlin(cu, S));
    if (total < 0 || numel(X) < total * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(S >= 2)
    for (int64_t s = 0; s < S; ++s) {
        const int64_t b0 = ldi(cu, offlin(cu, s)), b1 = ldi(cu, offlin(cu, s + 1));
        double* acc = (double*)scratch_raw((size_t)n * sizeof(double) + 64);
        float* sx = scratch_f32_b((size_t)n);
        float* sy = scratch_f32_c((size_t)n);
        for (int64_t i = 0; i < n; ++i) acc[i] = 0.0;
        for (int64_t r = b0; r < b1; ++r) {
            const float* x = in_row(X, rowoff(X, r, n), n, sx);
            for (int64_t i = 0; i < n; ++i) acc[i] += x[i];
        }
        const double inv = b1 > b0 ? 1.0 / (double)(b1 - b0) : 0.0;
        for (int64_t i = 0; i < n; ++i) sy[i] = b1 > b0 ? (float)(acc[i] * inv) : 0.0f;
        out_row(Y, rowoff(Y, s, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ pool_rows */
/* libref's: the mean of each k consecutive rows summed in f32 and rounded to y's dtype, then the
 * multiplier and a second rounding -- the unfused pair. Parallel over output rows. */
AVX_KERNEL(pool_rows) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *X = &a->t[0], *Y = &a->t[1];
    const int64_t n = p_int(a, "n", 0), k = p_int(a, "k", 0);
    const float mult = p_f32(a, "mult", 1.0f);
    if (n <= 0 || k <= 0) return RAD_E_SHAPE;
    const int64_t rows_in = rows_of(X, n);
    if (rows_in <= 0 || rows_in % k) return RAD_E_SHAPE;
    const int64_t rows = rows_in / k;
    if (numel(Y) < rows * n) return RAD_E_SHAPE;
    const float inv = 1.0f / (float)k;

    AVX_PARALLEL_FOR_IF(rows * n * k >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* acc = scratch_f32((size_t)n);
        float* sx = scratch_f32_b((size_t)n);
        for (int64_t i = 0; i < n; ++i) acc[i] = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            const float* v = in_row(X, rowoff(X, r * k + j, n), n, sx);
            for (int64_t i = 0; i < n; ++i) acc[i] += v[i];
        }
        for (int64_t i = 0; i < n; ++i) acc[i] *= inv;
        roundtrip(Y->dtype, acc, n, sx);
        for (int64_t i = 0; i < n; ++i) acc[i] *= mult;
        out_row(Y, rowoff(Y, r, n), n, acc);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scale_rows */
/* y[m, i] = x[m, i] * act(s[m]), then + add[m, i] if the optional residual is passed.
 *
 * The activation is applied to the SCALAR and not to the product, which is what a gate means --
 * Qwen3.5-MoE's shared expert is scaled by sigmoid(shared_expert_gate(x)).
 *
 * THE RESIDUAL ADD IS A ROUND TRIP AND THAT IS THE CONTRACT. libref stores the product into `y`,
 * READS IT BACK, and stores the sum -- so at bf16 the product is narrowed before the add sees it.
 * That is not an artefact of writing it as two statements: it is what a fused kernel does, because
 * the product has to be stored anyway. Reproduced here in a hot buffer rather than through the
 * operand, which is the same arithmetic without the second pass over memory. */
AVX_KERNEL(scale_rows) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *sc = &a->t[1], *y = &a->t[3];
    const RadTensor* ad = rad_arg_in(a, 2);
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n || numel(sc) < rows) return RAD_E_SHAPE;
    if (ad && numel(ad) < rows * n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;

    const int64_t ss = laststride(sc);
    const size_t tmpb = (size_t)n * sizeof(float) + 64;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const float v0 = ldt(sc, r * ss);
        /* The SCALAR sigmoid, through the ABI's arithmetic and not the vector polynomial: one
         * value per row, so there is nothing to vectorise, and going through the same expf the
         * oracle does removes a source of disagreement for free. */
        const float v = sig ? 1.0f / (1.0f + std::exp(-v0)) : v0;
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* xr = in_row(x, rowoff(x, r, n), n, sx);
        const vf vv = vf_set1(v);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(sy + i, vf_mul(vf_loadu(xr + i), vv));
        for (; i < n; ++i) sy[i] = xr[i] * v;
        if (ad) {
            void* tp = scratch_raw(tmpb);
            roundtrip(y->dtype, sy, n, tp);
            float* sa = scratch_f32_c((size_t)n);
            const float* av = in_row(ad, rowoff(ad, r, n), n, sa);
            i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_add(vf_loadu(sy + i), vf_loadu(av + i)));
            for (; i < n; ++i) sy[i] += av[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ silu_mul */
/* y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The gate is the FIRST half, which is the
 * order rad-convert fuses gate and up into one weight in, so the two conventions cannot drift.
 *
 * THE bf16 OUTPUT TAKES A CHEAPER SILU, and that is a precision argument rather than a
 * punt: vf_silu's degree-5 minimax exp plus a full division is ~0.6 ulp, while a bf16 store
 * rounds at 3e-3 -- the output cannot carry what the exact path computes. The fast form is
 * vf_silu_fast (avx_vec.h -- shared with the conv kernels since, same rule: keyed on the
 * OUTPUT dtype, twentyfold inside the bf16 bucket), and f32 outputs keep the exact path.
 * Local to ops whose product is consumed once -- shared vf_sigmoid feeds gate chains whose
 * errors accumulate. */

AVX_KERNEL(silu_mul) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        /* One in_row over the WHOLE 2n row rather than two over its halves: the two halves are
         * adjacent in memory and one conversion of 2n beats two of n by the call overhead and by
         * keeping the hardware prefetcher on a single stream. */
        float* s = scratch_f32((size_t)(2 * n));
        float* sy = scratch_f32_b((size_t)n);
        const float* g = in_row(gu, rowoff(gu, r, 2 * n), 2 * n, s);
        const float* u = g + n;
#if AVX_LEVEL >= 2
        /* The fast silu is a precision decision keyed on the OUTPUT dtype (see above): a bf16
         * store cannot carry 6e-5, an f32 one is held at 1e-6. */
        if (y->dtype == RAD_BF16) {
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_mul(vf_silu_fast(vf_loadu(g + i)), vf_loadu(u + i)));
            for (; i < n; ++i) {
                const float gi = g[i];
                sy[i] = (gi / (1.0f + std::exp(-gi))) * u[i];
            }
            out_row(y, rowoff(y, r, n), n, sy);
            continue;
        }
#endif
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sy + i, vf_mul(vf_silu(vf_loadu(g + i)), vf_loadu(u + i)));
        for (; i < n; ++i) {
            const float gi = g[i];
            sy[i] = (gi / (1.0f + std::exp(-gi))) * u[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu / silu / sigmoid
 *
 * These run over the whole tensor rather than row by row -- libref's `unary` does the same -- so
 * the blocking is over a fixed tile that stays in L1 rather than over a row whose length the op
 * does not know.
 *
 * GELU IS THE ONE OP HERE WITH A SCALAR INNER LOOP AND IT IS DELIBERATE. libref computes
 * 0.5*x*(1 + erf(x/sqrt2)) with libm's erff, and the kbench tolerance for a `*`/f32 op is 1e-5 --
 * which a minimax erf polynomial can meet, and which is not the point: the polynomial would put a
 * second definition of gelu in the tree whose agreement with the first is a property of the
 * coefficients. The dtype switch is still hoisted, so this is far from libref's speed even
 * scalar; if gelu ever appears on a hot path in a model this project serves, the polynomial is
 * the obvious next step and --bench gelu is where it would be justified. Qwen and Gemma are silu
 * throughout, so today it does not. */
enum { UN_GELU = 0, UN_SILU = 1, UN_SIGMOID = 2 };

template <int WHICH>
static int unary(const RadArgs* a) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    /* The strided case falls back to element-at-a-time, as libref does: `offlin` has to be
     * evaluated per element there and no amount of vectorisation survives it. */
    if (!dense_row(x) || !dense_row(y) || !rad_tensor_is_contiguous(x) ||
        !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i) {
            const float v = ldt(x, offlin(x, i));
            float o;
            if (WHICH == UN_GELU)         o = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));
            else if (WHICH == UN_SILU)    o = v / (1.0f + std::exp(-v));
            else                          o = 1.0f / (1.0f + std::exp(-v));
            stt(y, offlin(y, i), o);
        }
        return RAD_OK;
    }

    const int64_t TILE = 4096;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* s  = scratch_f32((size_t)TILE);
        float* so = scratch_f32_b((size_t)TILE);
        const float* v = in_row(x, off, m, s);
        int64_t i = 0;
        if (WHICH != UN_GELU) {
            for (; i + VF_N <= m; i += VF_N) {
                vf u = vf_loadu(v + i);
                vf_storeu(so + i, WHICH == UN_SILU ? vf_silu(u) : vf_sigmoid(u));
            }
        }
        for (; i < m; ++i) {
            const float u = v[i];
            if (WHICH == UN_GELU)      so[i] = 0.5f * u * (1.0f + std::erf(u * 0.70710678118654752f));
            else if (WHICH == UN_SILU) so[i] = u / (1.0f + std::exp(-u));
            else                       so[i] = 1.0f / (1.0f + std::exp(-u));
        }
        out_row(y, off, m, so);
    }
    return RAD_OK;
}

AVX_KERNEL(gelu)    { return unary<UN_GELU>(a); }
AVX_KERNEL(silu)    { return unary<UN_SILU>(a); }
AVX_KERNEL(sigmoid) { return unary<UN_SIGMOID>(a); }

/* ------------------------------------------------------------------ fill */
/* y = value over [M, n] at y's own row pitch, the value rounded once by the store. Nothing to
 * vectorise that the store does not already do: it is a write of one number. */
AVX_KERNEL(fill) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    if (M <= 0) return RAD_E_SHAPE;
    const float v = p_f32(a, "value", 0.0f);
    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < M; ++r) {
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows */
/* y[i, :] = x[idx[i], :]. A negative index writes a ZERO ROW -- the padding rule embed_lookup
 * follows, and what a step with fewer sampled rows than the buffer's extent leaves behind. An
 * index at or past x's row count is RAD_E_INVAL and is checked BEFORE the loop, because an OpenMP
 * body cannot return. */
AVX_KERNEL(gather_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    const int64_t src_rows = rows_of(x, n);
    if (M <= 0 || src_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    for (int64_t m = 0; m < M; ++m)
        if (ldi(idx, offlin(idx, m)) >= src_rows) return RAD_E_INVAL;

    /* A ROW COPY, NOT A CONVERSION, when the dtypes match -- which they always do for a gather.
     * memcpy on a whole row beats a widen/narrow round trip by the width of the dtype, and a
     * gather that changed the bits of what it moved would be a cast wearing a gather's name. */
    const bool same = x->dtype == y->dtype && dense_row(x) && dense_row(y);
    const size_t rowb = same ? (size_t)((rad_dtype_bits(x->dtype) * n) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r  = ldi(idx, offlin(idx, m));
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        if (r < 0) {
            if (same) std::memset((void*)byte_at(y->data, y->dtype, yb), 0, rowb);
            else for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, 0.0f);
            continue;
        }
        const int64_t xb = rowoff(x, r, n);
        if (same) {
            std::memcpy((void*)byte_at(y->data, y->dtype, yb),
                        byte_at(x->data, x->dtype, xb), rowb);
        } else {
            float* s = scratch_f32((size_t)n);
            out_row(y, yb, n, in_row(x, xb, n, s));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ scatter_rows */
/* x[idx[i], :] = v[i, :], the mirror of gather_rows. `x` is INOUT and the write is PARTIAL: a row
 * no index names keeps what it held, and a NEGATIVE index writes nothing, as a negative one reads
 * a zero row on the gather side. Indices are distinct by contract (RAD_OPD_F_IDX_UNIQUE), which is
 * what lets the rows go to the threads in any order. */
AVX_KERNEL(scatter_rows) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", v->rank >= 1 ? v->shape[v->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(v, n);
    const int64_t dst_rows = rows_of(x, n);
    if (M <= 0 || dst_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    for (int64_t m = 0; m < M; ++m)
        if (ldi(idx, offlin(idx, m)) >= dst_rows) return RAD_E_INVAL;

    /* A row copy when the dtypes match, for gather_rows' reason. */
    const bool same = v->dtype == x->dtype && dense_row(v) && dense_row(x);
    const size_t rowb = same ? (size_t)((rad_dtype_bits(x->dtype) * n) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ldi(idx, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), xb = rowoff(x, r, n);
        if (same) {
            std::memcpy((void*)byte_at(x->data, x->dtype, xb), byte_at(v->data, v->dtype, vb),
                        rowb);
        } else {
            float* s = scratch_f32((size_t)n);
            out_row(x, xb, n, in_row(v, vb, n, s));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ cast */
/* The tensors carry their own dtypes and those are what the bytes actually are; `from`/`to` are in
 * the schema because the SELECTOR needs them, and a disagreement between the parameter and the
 * tensor is a caller bug this cannot diagnose better than the caller can. libref says the same.
 *
 * Integer pairs go through the INTEGER path, so a token id or a block index above 2^24 survives a
 * cast that f32 would round. That is not a tolerance question -- it is the difference between a
 * token id and a different token id. */
AVX_KERNEL(cast) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    const bool int_pair = (x->dtype == RAD_I32 || x->dtype == RAD_I64 || x->dtype == RAD_U32 ||
                           x->dtype == RAD_I16) &&
                          (y->dtype == RAD_I32 || y->dtype == RAD_I64 || y->dtype == RAD_U32 ||
                           y->dtype == RAD_I16);
    if (int_pair || !rad_tensor_is_contiguous(x) || !rad_tensor_is_contiguous(y)) {
        AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
        for (int64_t i = 0; i < nel; ++i) {
            if (int_pair) st_int_dt(y->dtype, y->data, offlin(y, i),
                                    ld_int_dt(x->dtype, x->data, offlin(x, i)));
            else          stt(y, offlin(y, i), ldt(x, offlin(x, i)));
        }
        return RAD_OK;
    }

    const int64_t TILE = 8192;
    const int64_t ntile = (nel + TILE - 1) / TILE;
    AVX_PARALLEL_FOR_IF(nel >= AVX_PAR_MIN)
    for (int64_t t = 0; t < ntile; ++t) {
        const int64_t off = t * TILE;
        const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
        float* s = scratch_f32((size_t)TILE);
        out_row(y, off, m, in_row(x, off, m, s));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ softmax */
/* Max-subtracted, f32, over the last n elements of a row. The max subtraction is not an
 * optimisation: without it a 60-nat logit range overflows, and every kernel this is checked beside
 * does it, so doing it differently would be a difference larger than rounding.
 *
 * THE ROUND TRIP IS THE CONTRACT. libref writes the exponentials to `y`, reads them back, and
 * multiplies by 1/sum -- so a bf16 output carries two roundings. This does the same arithmetic in
 * a hot scratch buffer instead of through the output operand: identical result, one pass over
 * memory instead of three. */
AVX_KERNEL(softmax) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n) return RAD_E_SHAPE;
    const size_t tmpb = (size_t)n * 8 + 64;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* s  = scratch_f32((size_t)n);
        float* e  = scratch_f32_b((size_t)n);
        void*  tp = scratch_raw(tmpb);
        const float* v = in_row(x, rowoff(x, r, n), n, s);
        const float mx = row_max(v, n);

        const vf vmx = vf_set1(mx);
        vf acc = vf_zero();
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf t = vf_exp(vf_sub(vf_loadu(v + i), vmx));
            vf_storeu(e + i, t);
            acc = vf_add(acc, t);
        }
        float sum = vf_hsum(acc);
        for (; i < n; ++i) { e[i] = std::exp(v[i] - mx); sum += e[i]; }

        roundtrip(y->dtype, e, n, tp);
        const vf inv = vf_set1(1.0f / sum);
        i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(e + i, vf_mul(vf_loadu(e + i), inv));
        for (; i < n; ++i) e[i] *= (1.0f / sum);
        out_row(y, rowoff(y, r, n), n, e);
    }
    return RAD_OK;
}
