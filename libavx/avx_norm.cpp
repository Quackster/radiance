/* avx_norm.cpp -- rmsnorm, rmsnorm_add and layernorm.
 *
 * ============================== ONE PASS OVER THE ROW, TWICE ==============================
 *
 * Every norm here is a REDUCTION followed by a SCALE, which means two passes over the row and no
 * way around it: the scale needs a number the whole row contributed to. What can be avoided is
 * paying for the dtype conversion twice, and that is the shape of every kernel below -- convert
 * the row to f32 ONCE into scratch, reduce out of scratch, scale out of scratch. libref converts
 * per element and therefore converts each element three times in rmsnorm_add.
 *
 * The reduction itself is `row_sumsq` (avx_support.cpp), which is a four-accumulator vector tree.
 * That does NOT sum in libref's ascending-index order, and it is the whole reason the bf16
 * tolerance for a norm is 4e-3 rather than zero. Stated here because it is the one thing about
 * these kernels that a reader should not have to derive: the arithmetic is the same, the ORDER is
 * not, and the order is what a tolerance is for.
 *
 * ============================== THE TWO ROUNDING CONTRACTS ==============================
 *
 * `wadd` is GemmaRMSNorm's `1 +`, and it is added to the weight in f32 AFTER the weight has been
 * widened -- never to the stored bf16 gain. Qwen3.5, Qwen3-Next and Gemma all store a zero-centred
 * gain, and forming `1 + w` in bf16 first loses 0.3% per channel. libr4d's fused norms say the
 * same thing; this is the third implementation of that sentence and they agree.
 *
 * `rmsnorm_add` takes the variance from the UNROUNDED f32 sum and the multiply from the sum AFTER
 * it has been narrowed into residual_out. That asymmetry is not an accident of libref's loop: it
 * is what vLLM's fused_add_rms_norm does, because the residual has to be stored anyway and
 * re-reading it is free, and every fused kernel in this project will do it. An implementation that
 * kept f32 throughout would be half an ulp better and would fail against every kernel it is meant
 * to certify.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>

using namespace avx;

/* The gain row, widened once per LAUNCH rather than once per row.
 *
 * WHY THIS IS WORTH ITS OWN BUFFER: `w` is the same n elements for every one of `rows` rows, so
 * converting it inside the row loop converts it `rows` times -- at a 4096-wide bf16 gain and a
 * 2048-token prefill chunk that is 8 M pointless conversions. It cannot go in the thread_local
 * scratch, because every thread needs it and they would each build their own; it is built once
 * into a caller-provided buffer before the parallel region and read by all of them.
 *
 * `wadd` is folded in HERE, which is also where it is cheapest: once per element of the gain,
 * rather than once per element of the output. */
static void widen_gain(const RadTensor* w, int64_t n, float wadd, float* out) {
    const int64_t ws = laststride(w);
    if (ws == 1) {
        row_to_f32(w->dtype, w->data, out, n);
    } else {
        for (int64_t i = 0; i < n; ++i) out[i] = ldt(w, i * ws);
    }
    if (wadd != 0.0f) {
        const vf va = vf_set1(wadd);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(out + i, vf_add(vf_loadu(out + i), va));
        for (; i < n; ++i) out[i] += wadd;
    }
}

/* ------------------------------------------------------------------ rmsnorm */
/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd) */
AVX_KERNEL(rmsnorm) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[2];
    /* An absent gain is one (libref's reading), and `wadd` belongs to a gain. */
    const RadTensor* w = a->t[1].data ? &a->t[1] : nullptr;
    const int64_t n = p_int(a, "n", w && w->rank ? w->shape[w->rank - 1] : 0);
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const int64_t round_before_weight = p_int(a, "round_before_weight", 0);
    if (round_before_weight != 0 && round_before_weight != 1) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    if (!w && wadd != 0.0f) return RAD_E_INVAL;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || (w && numel(w) < n)) return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    if (w) widen_gain(w, n, wadd, gain);
    else   for (int64_t i = 0; i < n; ++i) gain[i] = 1.0f;
    /* `mult` folded into the gain: the row is scaled before its one rounding. */
    const float mult = p_f32(a, "mult", 1.0f);
    if (mult != 1.0f)
        for (int64_t i = 0; i < n; ++i) gain[i] *= mult;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, sx);
        const float ss = row_sumsq(v, n);
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vs = vf_set1(s);
        int64_t i = 0;
        if (round_before_weight && x->dtype != RAD_F32) {
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_mul(vf_loadu(v + i), vs));
            for (; i < n; ++i) sy[i] = v[i] * s;
            /* LFM2 narrows the normalized row before applying its gain. The input has been
             * consumed now, so sx can hold the packed row without another scratch allocation. */
            row_from_f32(x->dtype, sy, sx, n);
            row_to_f32(x->dtype, sx, sy, n);
            i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_mul(vf_loadu(sy + i), vf_loadu(gain + i)));
            for (; i < n; ++i) sy[i] *= gain[i];
        } else {
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(sy + i, vf_mul(vf_mul(vf_loadu(v + i), vs), vf_loadu(gain + i)));
            for (; i < n; ++i) sy[i] = v[i] * s * gain[i];
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_add */
/* residual_out = x + residual;  y = rmsnorm(residual_out) * (w + wadd), with the variance from the
 * unrounded sum and the multiply from the narrowed one. */
AVX_KERNEL(rmsnorm_add) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *res = &a->t[1], *w = &a->t[2], *y = &a->t[3], *ro = &a->t[4];
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps  = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(ro) < rows * n ||
        numel(res) < rows * n || numel(w) < n) return RAD_E_SHAPE;

    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    widen_gain(w, n, wadd, gain);
    const uint32_t ro_dt = ro->dtype;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sr = scratch_f32_b((size_t)n);
        float* sm = scratch_f32_c((size_t)n);
        const float* xv = in_row(x,   rowoff(x, r, n),   n, sx);
        const float* rv = in_row(res, rowoff(res, r, n), n, sr);

        vf acc0 = vf_zero(), acc1 = vf_zero();
        int64_t i = 0;
        for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
            vf s0 = vf_add(vf_loadu(xv + i), vf_loadu(rv + i));
            vf s1 = vf_add(vf_loadu(xv + i + VF_N), vf_loadu(rv + i + VF_N));
            vf_storeu(sm + i, s0);
            vf_storeu(sm + i + VF_N, s1);
            acc0 = vf_fma(s0, s0, acc0);
            acc1 = vf_fma(s1, s1, acc1);
        }
        float ss = vf_hsum(vf_add(acc0, acc1));
        for (; i < n; ++i) { float v = xv[i] + rv[i]; sm[i] = v; ss += v * v; }

        /* The residual is written from the UNROUNDED sum... */
        out_row(ro, rowoff(ro, r, n), n, sm);
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);

        /* ...and read back NARROWED for the multiply. Done in registers rather than by re-reading
         * the operand: `row_from_f32` then `row_to_f32` over a hot buffer is the same arithmetic
         * and does not depend on the store having retired. */
        /* `sr` is reused as the narrowing buffer and that is not a shortcut, it is the ONLY slot
         * available: `gain` lives in this thread's scratch_raw arena and calling scratch_raw again
         * inside the parallel region would grow that arena, free the old block, and leave every
         * other thread reading a dangling `gain`. `sr` is n floats -- 4n bytes -- which holds any
         * dtype this operand can be, and `rv` (which may or may not be `sr`, depending on whether
         * `residual` was already f32) is dead the moment the sum is formed. */
        if (ro_dt != RAD_F32) {
            row_from_f32(ro_dt, sm, sr, n);
            row_to_f32(ro_dt, sr, sm, n);
        }

        const vf vs = vf_set1(s);
        i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(sm + i, vf_mul(vf_mul(vf_loadu(sm + i), vs), vf_loadu(gain + i)));
        for (; i < n; ++i) sm[i] = sm[i] * s * gain[i];
        out_row(y, rowoff(y, r, n), n, sm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ layernorm */
/* y = (x - mean) * rsqrt(var + eps) * w + b, BIASED variance (divide by n), which is what every
 * transformer implementation means by layer norm. `b` is optional -- a ViT that dropped its bias
 * passes a null operand rather than a zero tensor. */
AVX_KERNEL(layernorm) {
    if (a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *y = &a->t[3];
    const RadTensor* b = rad_arg_in(a, 2);
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(w) < n) return RAD_E_SHAPE;
    if (b && numel(b) < n) return RAD_E_SHAPE;

    /* Both the gain and the bias are launch-invariant, so both are widened once, into one
     * allocation so there is a single scratch_raw call live at a time (avx_common.h: a second call
     * may reallocate and invalidate the first pointer). */
    float* gb = (float*)scratch_raw((size_t)(2 * n) * sizeof(float));
    float* gain = gb;
    float* bias = gb + n;
    widen_gain(w, n, 0.0f, gain);
    if (b) {
        const int64_t bs = laststride(b);
        if (bs == 1) row_to_f32(b->dtype, b->data, bias, n);
        else for (int64_t i = 0; i < n; ++i) bias[i] = ldt(b, i * bs);
    }
    const bool has_b = b != nullptr;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* sx = scratch_f32((size_t)n);
        float* sy = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, sx);

        /* TWO PASSES, NOT THE ONE-PASS (sum, sumsq) FORM. The single-pass variance
         * E[x^2] - E[x]^2 is one reduction instead of two and is catastrophically worse when the
         * mean is large relative to the spread -- which is exactly a post-attention residual. What
         * it costs here is one extra read of a row that is in L1 by then; what it would buy is a
         * cancellation the tolerance would hide until it did not. libref takes the mean first too,
         * so this also keeps the two agreeing for the right reason. */
        const float mean = row_sum(v, n) / (float)n;
        const vf vm = vf_set1(mean);
        vf acc = vf_zero();
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf d = vf_sub(vf_loadu(v + i), vm);
            acc = vf_fma(d, d, acc);
        }
        float ss = vf_hsum(acc);
        for (; i < n; ++i) { float d = v[i] - mean; ss += d * d; }

        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        const vf vs = vf_set1(s);
        i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            vf t = vf_mul(vf_mul(vf_sub(vf_loadu(v + i), vm), vs), vf_loadu(gain + i));
            if (has_b) t = vf_add(t, vf_loadu(bias + i));
            vf_storeu(sy + i, t);
        }
        for (; i < n; ++i) {
            float t = (v[i] - mean) * s * gain[i];
            if (has_b) t += bias[i];
            sy[i] = t;
        }
        out_row(y, rowoff(y, r, n), n, sy);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed */
/* A learned side x side position table resampled bilinearly to each row's grid (corners aligned,
 * edge taps clamped) and added in -- libref's grid_embed, operation for operation: the four taps
 * summed in f32 in the same order, rounded to x's dtype, then added and rounded again. One row a
 * vision-tower patch, off every step path, so it is written for clarity over lanes. */
AVX_KERNEL(grid_embed) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *tab = &a->t[1], *crd = &a->t[2];
    const int64_t n = p_int(a, "n", tab->rank ? tab->shape[tab->rank - 1] : 0);
    const int64_t side = p_int(a, "side", 0);
    const char* mode = p_str(a, "mode", "bilinear");
    const bool antialias = std::strcmp(mode, "bilinear_antialias") == 0;
    if (!antialias && std::strcmp(mode, "bilinear") != 0) return RAD_E_INVAL;
    if (n <= 0 || side <= 0 || numel(tab) < side * side * n) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < rows)
        return RAD_E_SHAPE;
    const int64_t cs = crd->stride[0], ms = crd->stride[1];
    const int64_t ts = laststride(tab);

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        int64_t g[4];
        for (int c = 0; c < 4; ++c) g[c] = ldi(crd, c * cs + r * ms);
        if (antialias) {
            // Separable half-pixel triangle filtering. Widen the footprint when reducing, then
            // normalize only surviving taps at an edge. The table dtype determines rounding of
            // raw/normalized weights and the horizontal intermediate, as in GPU interpolate.
            auto narrow = [&](float v) {
                if (tab->dtype == RAD_BF16) return rad_bf16_to_f32(rad_f32_to_bf16(v));
                if (tab->dtype == RAD_F16) return rad_f16_to_f32(rad_f32_to_f16(v));
                return v;
            };
            float* wh = scratch_f32((size_t)(2 * side));
            float* ww = wh + side;
            auto weights_axis = [&](int64_t at, int64_t size, float* w) {
                size = size > 0 ? size : 1;
                at = at < 0 ? 0 : (at >= size ? size - 1 : at);
                const float scale = (float)side / (float)size;
                const float support = scale > 1.0f ? scale : 1.0f;
                const float inv = 1.0f / support;
                const float center = scale * ((float)at + 0.5f);
                const int64_t first = std::max<int64_t>(0, (int64_t)(center - support + 0.5f));
                const int64_t end = std::min<int64_t>(side, (int64_t)(center + support + 0.5f));
                float total = 0.0f;
                for (int64_t j = 0; j < side; ++j) w[j] = 0.0f;
                for (int64_t j = first; j < end; ++j) {
                    const float distance = std::fabs(((float)(j - first) +
                        ((float)first - center) + 0.5f) * inv);
                    const float value = distance < 1.0f ? 1.0f - distance : 0.0f;
                    w[j] = narrow(value);
                    total += value;
                }
                for (int64_t j = first; j < end; ++j) w[j] = narrow(w[j] / total);
            };
            const int64_t grid[4] = { g[0], g[1], g[2], g[3] };
            weights_axis(grid[0], grid[2], wh);
            weights_axis(grid[1], grid[3], ww);
            const int64_t xb = rowoff(x, r, n), xs = laststride(x);
            for (int64_t i = 0; i < n; ++i) {
                float value = 0.0f;
                for (int64_t y = 0; y < side; ++y) {
                    if (wh[y] == 0.0f) continue;
                    float horizontal = 0.0f;
                    for (int64_t col = 0; col < side; ++col) {
                        if (ww[col] == 0.0f) continue;
                        horizontal += ldt(tab, rowoff(tab, y * side + col, n) + i * ts) * ww[col];
                    }
                    value += narrow(horizontal) * wh[y];
                }
                float pe = narrow(value);
                if (x->dtype == RAD_BF16) pe = rad_bf16_to_f32(rad_f32_to_bf16(pe));
                if (x->dtype == RAD_F16) pe = rad_f16_to_f32(rad_f32_to_f16(pe));
                stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
            }
            continue;
        }
        int64_t th[2], tw[2];
        float wh[2], ww[2];
        auto axis = [&](int64_t idx, int64_t size, int64_t* t, float* w) {
            const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
            const float fl = std::floor(src);
            for (int k = 0; k < 2; ++k) {
                const int64_t tap = (int64_t)fl + k;
                t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
                const float d = std::fabs(src - fl - (float)k);
                w[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
            }
        };
        axis(g[0], g[2], th, wh);
        axis(g[1], g[3], tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            float pe = acc;
            if (x->dtype == RAD_BF16) pe = rad_bf16_to_f32(rad_f32_to_bf16(pe));
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}
