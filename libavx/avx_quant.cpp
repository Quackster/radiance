/* avx_quant.cpp -- the quantisers, the rotation, and the Gram accumulator.
 *
 * ============================== THE WALSH-HADAMARD IS BIT-IDENTICAL, ON PURPOSE ==============================
 *
 * The rotation is a sequence of butterflies at ascending strides 1, 2, 4, ... g/2, pairing i with
 * i + h where (i & h) == 0. The butterfly matrices for different strides COMMUTE -- H is a tensor
 * power of the 2x2 Hadamard -- so any order of `h` gives the same transform in exact arithmetic,
 * and it is tempting to reorder them for the vectoriser.
 *
 * That temptation is refused here and the order is libref's exactly, ASCENDING, because commuting
 * in exact arithmetic is not the same as commuting in f32: the intermediate values differ and the
 * roundings land in different places. Keeping the order means every butterfly is the same add and
 * the same subtract on the same two floats as the reference performs, so the rotated block is
 * BIT-IDENTICAL and not merely close -- which is what lets the whole rotated-int8 family be held
 * at the quantiser tolerance (1e-6) instead of a loosened one. What is vectorised is WHICH LANES a
 * butterfly runs on, not the arithmetic:
 *
 *   h >= VF_N   the pair (i, i+h) is two whole vectors. One load, one add, one sub, two stores.
 *   h <  VF_N   the pair is inside one vector, so the partner is a SWIZZLE of it -- swap adjacent
 *               lanes, pairs of lanes, 128-bit lanes, 256-bit halves -- and the two results are
 *               selected by a lane mask. Four such steps at AVX-512, three at AVX2.
 *
 * Doing the small strides scalar was the obvious first implementation and it is the wrong one: at
 * g = 128 and VF_N = 16 that is four scalar passes over 128 elements against three vector passes,
 * so the part that was left scalar would be most of the cost.
 *
 * ============================== THE ROTATION IS PERFORMED ONCE, NOT TWICE ==============================
 *
 * libref rotates each block twice -- once to find the absolute maximum and once to write the codes
 * -- because it has a 4096-float stack frame and no row-sized scratch. That is a factor of two on
 * every rotated op and it is pure recomputation: the rotated row is deterministic. Here the whole
 * row is rotated once into scratch, the maximum is taken over it, and the codes are written from
 * it. Identical arithmetic, half the butterflies.
 *
 * ============================== THE TWO `group` READINGS, AND THE TWO sqrt(g) CONVENTIONS ==============================
 *
 * docs/OPS.md is explicit that `group` means two different things in this table and that a kernel
 * reading it the other way is WRONG rather than slow:
 *
 *   quant_act_i8, quant_act_fp8, dequant   `group` is the number of contiguous elements sharing
 *   and the fp8 fusions                    one scale. `scale` is [M, ceil(n/group)].
 *   every had_* INT8 form                  `group` is the ROTATION WIDTH and the scale is per ROW,
 *                                          carrying the 1/sqrt(group).
 *   the had_* FP8 forms                    `group` is BOTH, and there is NO normalisation at all:
 *                                          <Hw, Ha> = g <w, a> and the whole factor g is folded
 *                                          into the stored weight scale at convert, where dividing
 *                                          a bf16 by a power of two is exact. Splitting it as the
 *                                          int8 forms do would round amax/7/sqrt(g) into bf16 and
 *                                          land ~0.2% of multiplicative error on every group.
 *
 * Three readings of one parameter name is not a pleasant fact, and it is the vocabulary's rather
 * than this plugin's. It is restated here because getting it wrong produces a plausible number.
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

#include "avx_had.h"

/* ================================================================== int8 store
 * `q` is almost always RAD_I8, in which case the row store is the vectorised cvtps/packs path
 * (avx_vec.h) whose rounding is MXCSR round-to-nearest-even and whose saturation is the pack
 * instruction's -- exactly libref's nearbyintf-and-clamp, bit for bit. Anything else falls back to
 * the generic row store, which is correct and slower. */
static inline void store_codes(const RadTensor* q, int64_t base, int64_t n, const float* v) {
    out_row(q, base, n, v);
}

/* Scale a row by a scalar, in place. */
static inline void scale_row(float* v, int64_t n, float s) {
    const vf vs = vf_set1(s);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) vf_storeu(v + i, vf_mul(vf_loadu(v + i), vs));
    for (; i < n; ++i) v[i] *= s;
}

/* ------------------------------------------------------------------ quant_act_i8 */
/* q = round(x * 127 / amax), scale = amax / 127, per `group` contiguous elements. An all-zero
 * group gets scale 0 and codes 0, which dequantises back to zero -- the alternative, a tiny
 * epsilon scale, turns a zero row into denormal noise in the GEMM that reads it. */
AVX_KERNEL(quant_act_i8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    /* DENSE-i8 FAST PATH: the per-group scale feeds the int8 encoder directly, with no f32
     * scratch round trip between them. */
    if (q->dtype == RAD_I8 && dense_row(q)) {
        AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
        for (int64_t r = 0; r < rows; ++r) {
            float* buf = scratch_f32((size_t)n);
            const float* v = in_row(x, rowoff(x, r, n), n, buf);
            int8_t* qb = (int8_t*)byte_at(q->data, q->dtype, rowoff(q, r, n));
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
                const float amax = row_amax(v + k0, k1 - k0);
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                cvt_scaled_f32_to_i8(v + k0, inv, qb + k0, k1 - k0);
                stt(s, rowoff(s, r, ngrp) + gi * laststride(s), amax / 127.0f);
            }
        }
        return RAD_OK;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        float* w = scratch_f32_b((size_t)n);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float amax = row_amax(v + k0, k1 - k0);
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            const vf vi = vf_set1(inv);
            int64_t i = k0;
            for (; i + VF_N <= k1; i += VF_N) vf_storeu(w + i, vf_mul(vf_loadu(v + i), vi));
            for (; i < k1; ++i) w[i] = v[i] * inv;
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), amax / 127.0f);
        }
        store_codes(q, rowoff(q, r, n), n, w);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ had_quant_act_i8 */
/* Rotation at width `group`, then symmetric int8 with ONE scale per ROW carrying the
 * 1/sqrt(group). The absolute maximum is over the WHOLE rotated row, because the scale is per row. */
AVX_KERNEL(had_quant_act_i8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    /* Same dense-i8 fusion as quant_act_i8: the rotation stays, but its output is scaled +
     * encoded straight to bytes. The amax here is whole-row (one scale per row), so the fusion
     * is a single call rather than per group. */
    if (q->dtype == RAD_I8 && dense_row(q)) {
        AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
        for (int64_t r = 0; r < rows; ++r) {
            float* buf = scratch_f32((size_t)n);
            const float* v = in_row(x, rowoff(x, r, n), n, buf);
            if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
            fwht_row(buf, n, g);
            const float amax = row_amax(buf, n);
            cvt_scaled_f32_to_i8(buf, amax > 0.0f ? 127.0f / amax : 0.0f,
                                 (int8_t*)byte_at(q->data, q->dtype, rowoff(q, r, n)), n);
            stt(s, r * laststride(s), amax / 127.0f * norm);
        }
        return RAD_OK;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        /* in_row may hand back the operand's own pointer for a dense f32 input, and the rotation
         * is IN PLACE -- so the row is copied into scratch whenever that happened. Rotating the
         * caller's activation would be a kernel that destroys its own input. */
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
        fwht_row(buf, n, g);
        const float amax = row_amax(buf, n);
        scale_row(buf, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        store_codes(q, rowoff(q, r, n), n, buf);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_had_quant_i8 */
/* The fused rmsnorm + hadamard + quant_act_i8 a w4a8 layer wants where a bf16 layer wants a plain
 * rmsnorm.
 *
 * `residual` PRESENT IS THE fused_add_rms_norm FORM: the add happens first, the sum is written
 * back into `residual` (that is what makes it INOUT and what the next layer's shortcut reads), and
 * the norm is taken over the sum. The variance comes from the f32 sum and the rotation reads the
 * sum back AFTER it has been narrowed to the residual tensor's own dtype -- which is what the pair
 * of kernels this replaces does, and matching it is what makes the fused form substitutable for
 * that pair.
 *
 * `out_bf16` is the post-norm, PRE-ROTATION tensor for a layer that also has a bf16 consumer: the
 * rotation is an artefact of the int8 path, and a bf16 consumer wants the normalised activation
 * rather than a Walsh-Hadamard of it. */
AVX_KERNEL(rmsnorm_had_quant_i8) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* ob  = rad_arg_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const int64_t g = p_int(a, "group", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows || numel(w) < n) return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    /* The gain, widened once per launch with `wadd` already folded in -- see avx_norm.cpp on why
     * this cannot live in the per-thread scratch. */
    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, n);
        else for (int64_t i = 0; i < n; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < n; ++i) gain[i] += wadd;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* src = scratch_f32((size_t)n);
        float* buf = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);

        if (res) {
            /* x + residual, stored back, then READ BACK NARROWED -- the round trip is the
             * contract. Done through a hot buffer rather than by re-reading the operand: same
             * arithmetic, and it does not depend on the store having retired. */
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            const float* rv = in_row(res, rowoff(res, r, n), n, tmp);
            for (int64_t i = 0; i < n; ++i) buf[i] = xv[i] + rv[i];
            out_row(res, rowoff(res, r, n), n, buf);
            if (res->dtype != RAD_F32) {
                row_from_f32(res->dtype, buf, tmp, n);
                row_to_f32(res->dtype, tmp, buf, n);
            }
            std::memcpy(src, buf, (size_t)n * sizeof(float));
        } else {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            if (xv != src) std::memcpy(src, xv, (size_t)n * sizeof(float));
        }

        const float sc = 1.0f / std::sqrt(row_sumsq(src, n) / (float)n + eps);
        const vf vsc = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(buf + i, vf_mul(vf_mul(vf_loadu(src + i), vsc), vf_loadu(gain + i)));
        for (; i < n; ++i) buf[i] = src[i] * sc * gain[i];

        if (ob) out_row(ob, rowoff(ob, r, n), n, buf);   /* post-norm, PRE-rotation */

        fwht_row(buf, n, g);
        const float amax = row_amax(buf, n);
        scale_row(buf, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        store_codes(q, rowoff(q, r, n), n, buf);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gated_had_quant_i8 */
/* An elementwise gated product, then the rotation and the quantiser. `act` picks between the two
 * rotated sites a w4a8 model has: silu(a)*b is down_proj's input, sigmoid(a)*b is o_proj's on a
 * gated-attention layer. `mode` is libr4d's integer spelling of the same choice.
 *
 * `b` ABSENT IS THE PACKED FORM: `a` is [M, 2n], gate first half, up second -- which is what a
 * fused gate_up projection produces. Two operands with the second optional is the only shape that
 * expresses both readings. */
AVX_KERNEL(gated_had_quant_i8) {
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor *ga = &a->t[0], *q = &a->t[2], *s = &a->t[3];
    const RadTensor* up = rad_arg_in(a, 1);
    if (!ga->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(q) / n;
    if (rows <= 0 || numel(s) < rows) return RAD_E_SHAPE;
    if (numel(ga) < rows * (up ? n : 2 * n)) return RAD_E_SHAPE;
    if (up && numel(up) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);
    const int64_t gw = up ? n : 2 * n;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* gbuf = scratch_f32((size_t)gw);
        float* ubuf = scratch_f32_b((size_t)n);
        float* out  = scratch_f32_c((size_t)n);
        const float* gv = in_row(ga, rowoff(ga, r, gw), gw, gbuf);
        const float* uv = up ? in_row(up, rowoff(up, r, n), n, ubuf) : gv + n;

        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) {
            const vf gt = vf_loadu(gv + i);
            const vf act_v = (act == 1) ? vf_sigmoid(gt) : vf_silu(gt);
            vf_storeu(out + i, vf_mul(act_v, vf_loadu(uv + i)));
        }
        for (; i < n; ++i) {
            const float gt = gv[i];
            const float sg = 1.0f / (1.0f + std::exp(-gt));
            out[i] = (act == 1 ? sg : gt * sg) * uv[i];
        }

        fwht_row(out, n, g);
        const float amax = row_amax(out, n);
        scale_row(out, n, amax > 0.0f ? 127.0f / amax : 0.0f);
        store_codes(q, rowoff(q, r, n), n, out);
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dequant */
/* q * scale, with the SCALE STREAM'S OWN EXTENT deciding how many groups there are. The inverse of
 * quant_act_i8 and NOT of the rotated forms: undoing a rotation is a second Hadamard, and an op
 * that silently did one would make `dequant(had_quant(x)) == x` look true when the two have
 * different meanings. */
AVX_KERNEL(dequant) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    const int64_t ngrp = numel(s) / rows > 0 ? numel(s) / rows : 1;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0) g = (n + ngrp - 1) / ngrp;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(q, rowoff(q, r, n), n, buf);
        const int64_t sb = rowoff(s, r, ngrp), ss = laststride(s);
        /* GROUP BY GROUP, with the scale hoisted out of the element loop. libref reads the scale
         * per element through a divide; the group edge is known here, so the divide happens once
         * per group and the body is a vector multiply. Same product, same order. */
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g;
            if (k0 >= n) break;
            const int64_t k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = ldt(s, sb + gi * ss);
            const vf vs = vf_set1(sc);
            int64_t i = k0;
            for (; i + VF_N <= k1; i += VF_N) vf_storeu(out + i, vf_mul(vf_loadu(v + i), vs));
            for (; i < k1; ++i) out[i] = v[i] * sc;
        }
        /* A row longer than the groups describe takes the LAST group's scale, which is libref's
         * clamp (`gi < ngrp ? gi : ngrp - 1`) and is what a ragged final group means. */
        if (ngrp * g < n) {
            const float sc = ldt(s, sb + (ngrp - 1) * ss);
            for (int64_t i = ngrp * g; i < n; ++i) out[i] = v[i] * sc;
        }
        out_row(y, rowoff(y, r, n), n, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_fp8 */
/* The three choices this shares with libr4d's kernel, none of them derivable:
 *
 *   448        E4M3's largest finite value, and the scale is `amax / 448` -- a DEQUANT multiplier,
 *              the same sense as the checkpoint's `weight_scale_inv`, so the GEMM's fold multiplies
 *              the two scales together and nothing has to be inverted.
 *   scale 1    for an all-zero group, where amax/448 is zero and its reciprocal infinite. NOT the
 *              zero scale quant_act_i8 uses: that kernel's consumer reads a per-row int8 scale and
 *              a zero row is a zero row, while here the GEMM multiplies the two block scales and a
 *              zero would erase the weight's.
 *   the clamp  is not defensive. amax * (1/scale) is 448 in exact arithmetic and can land an ulp
 *              above it once the reciprocal is rounded, which encodes as a NaN rather than as the
 *              largest finite. */
AVX_KERNEL(quant_act_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            float sc;
            fp8_group(v, out, k0, k1, &sc);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
        out_row(q, rowoff(q, r, n), n, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- quant_act_fp8_row */
/* The per-row form, for the OCP-MX (mxfp4a8) GEMMs, which read one f32 activation scale per
 * token and no per-128 group. It reuses quant_act_fp8's body rather than restating it: that body
 * reads an absent `group` as 0 and clamps it to n, which is exactly one scale per row. */
AVX_KERNEL(quant_act_fp8_row) {
    return AVX_FN(avx_quant_act_fp8)(a, stream);
}

/* ------------------------------------------------------------- rmsnorm_quant_fp8 */
/* RMS norm, an optional residual add, and quant_act_fp8, in one pass.
 *
 * `residual` PRESENT IS THE fused_add_rms_norm FORM and the add is STORED BEFORE THE VARIANCE READS
 * IT: the sum goes back into `residual`, narrowed to that tensor's dtype, and the norm -- variance
 * included -- is taken over the narrowed sum. That is the `add` + `rmsnorm` pair, and not
 * rmsnorm_add, which takes its variance from the unrounded sum. The normalised row is then narrowed
 * to bf16 and the codes taken over it (avx_had.h). */
AVX_KERNEL(rmsnorm_quant_fp8) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = rad_arg_in(a, 1);
    const RadTensor* ob  = rad_arg_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp || numel(w) < n)
        return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    /* The gain with `wadd` folded in, widened once on the calling thread -- see avx_norm.cpp on
     * why it cannot live in the per-thread scratch. */
    float* gain = (float*)scratch_raw((size_t)n * sizeof(float));
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, n);
        else for (int64_t i = 0; i < n; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < n; ++i) gain[i] += wadd;
    }

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* src = scratch_f32((size_t)n);
        float* buf = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);

        if (res) {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            const float* rv = in_row(res, rowoff(res, r, n), n, tmp);
            for (int64_t i = 0; i < n; ++i) buf[i] = xv[i] + rv[i];
            out_row(res, rowoff(res, r, n), n, buf);
            if (res->dtype != RAD_F32) {
                row_from_f32(res->dtype, buf, tmp, n);
                row_to_f32(res->dtype, tmp, buf, n);
            }
            std::memcpy(src, buf, (size_t)n * sizeof(float));
        } else {
            const float* xv = in_row(x, rowoff(x, r, n), n, src);
            if (xv != src) std::memcpy(src, xv, (size_t)n * sizeof(float));
        }

        const float sc = 1.0f / std::sqrt(row_sumsq(src, n) / (float)n + eps);
        const vf vsc = vf_set1(sc);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N)
            vf_storeu(buf + i, vf_mul(vf_mul(vf_loadu(src + i), vsc), vf_loadu(gain + i)));
        for (; i < n; ++i) buf[i] = src[i] * sc * gain[i];
        narrow_bf16(buf, n, tmp);
        fp8_emit(buf, src, n, g, q, s, ob, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gated_quant_fp8 */
/* act(gate) * up over a PACKED [M, 2n] gate_up plane, gate half first, then quant_act_fp8 over the
 * bf16-narrowed product. `act` is silu or sigmoid, `mode` libr4d's integer spelling of the same
 * choice, silu when neither is given. */
AVX_KERNEL(gated_quant_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* ob = rad_arg_in(a, 3);
    const int64_t n = p_int(a, "n", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(gu) / (2 * n);
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* gbuf = scratch_f32((size_t)(2 * n));
        float* out  = scratch_f32_b((size_t)n);
        float* tmp  = scratch_f32_c((size_t)n);
        const float* gv = in_row(gu, rowoff(gu, r, 2 * n), 2 * n, gbuf);
        gate_bf16(gv, gv + n, out, n, act, tmp);
        fp8_emit(out, tmp, n, g, q, s, ob, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- the per-row (OCP-MX) forms */
/* Both bodies above read an absent `group` as 0 and clamp it to n, which is exactly one scale per
 * row -- so they ARE the per-row implementations and these wrappers only give the registry the
 * names whose `_shape` twins declare `scale` at [M, 1]. Same argument as quant_act_fp8_row above. */
AVX_KERNEL(rmsnorm_quant_fp8_row) {
    return AVX_FN(avx_rmsnorm_quant_fp8)(a, stream);
}
AVX_KERNEL(gated_quant_fp8_row) {
    return AVX_FN(avx_gated_quant_fp8)(a, stream);
}

/* ------------------------------------------------------------- had_quant_act_fp8 */
/* The same quantiser with a `group`-wide Hadamard in front of the absmax, which is the only place
 * a rotation can go: the whole benefit is that the maximum the grid is scaled by gets smaller.
 *
 * THE sqrt(g) IS NOT UNDONE HERE, and that is a deliberate difference from had_quant_act_i8 above,
 * which folds 1/sqrt(g) into its own scale. H is symmetric, so <Hw, Ha> = g <w, a>, and the
 * residual g has to be divided out exactly once between the two sides. The int8 op splits it; this
 * one gives the WHOLE factor to the weight, where dividing a bf16 scale by a power of two is an
 * exponent shift and therefore exact. */
AVX_KERNEL(had_quant_act_fp8) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    /* n % g != 0 is refused rather than rounded: a rotation block that straddled the end of the
     * row would mix real values with padding, and no weight block was rotated that way. */
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = n / g;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        float* out = scratch_f32_b((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        if (v != buf) std::memcpy(buf, v, (size_t)n * sizeof(float));
        fwht_row(buf, n, g);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            float sc;
            fp8_group(buf, out, gi * g, gi * g + g, &sc);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
        out_row(q, rowoff(q, r, n), n, out);
    }
    return RAD_OK;
}

/* ---------------------------------------------------------------- the Gram accumulator
 *
 * h[k1][k2] += sum over rows of x[m][k1] * x[m][k2], where x is the DEQUANTISED E4M3 input, or a
 * bf16 input as it stands when no scale is passed -- the second moment GPTQ's error feedback
 * needs, over exactly the operand the expert GEMM contracts.
 *
 * `h` IS READ AS WELL AS WRITTEN and that is the op: the caller zeroes it once and drives a whole
 * calibration corpus through the graph, so each launch adds its own block. An implementation that
 * stored instead of added would agree on one call and disagree on a corpus.
 *
 * THE ACCUMULATION IS f64 HERE AND f32 ON THE DEVICE, and that is libref's choice reproduced
 * rather than an accident: this is the one op in the vocabulary whose result is consumed OFFLINE,
 * by the packer, through a file -- so being tighter than the kernel is the right trade, and the
 * kbench tolerance covers the gap. It also means the inner reduction is NOT the f32 vector path
 * the rest of this file uses; there is nothing to vectorise about an f64 accumulation of an f32
 * product that would not change the answer.
 *
 * What IS vectorised is the dequantisation: rows are widened and scaled once into a scratch plane
 * rather than inside the k2 loop, which is O(n^2 * rows) with a per-element dtype dispatch in it
 * otherwise. */
AVX_KERNEL(gram_accum) {
    /* `a_scale` is optional: absent, `a` is read as it stands, which is the bf16 form. */
    if (a->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = rad_arg_in(a, 0), *s = rad_arg_in(a, 1), *h = rad_arg_in(a, 2);
    if (!x || !h) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = s ? p_int(a, "group", 0) : n;
    if (n <= 0 || g <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = (n + g - 1) / g;
    if (rows <= 0 || (s && numel(s) < rows * ngrp)) return RAD_E_SHAPE;
    if (h->rank != 2 || h->shape[0] != n || h->shape[1] != n) return RAD_E_SHAPE;

    float* xs = (float*)scratch_raw((size_t)rows * (size_t)n * sizeof(float) + 64);
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, rowoff(x, r, n), n, buf);
        float* dst = xs + r * n;
        const int64_t sb = s ? rowoff(s, r, ngrp) : 0, sst = s ? laststride(s) : 0;
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            const float sc = s ? ldt(s, sb + gi * sst) : 1.0f;
            const vf vs = vf_set1(sc);
            int64_t k = k0;
            for (; k + VF_N <= k1; k += VF_N) vf_storeu(dst + k, vf_mul(vf_loadu(v + k), vs));
            for (; k < k1; ++k) dst[k] = v[k] * sc;
        }
    }

    const int64_t hst = laststride(h);
    AVX_PARALLEL_FOR_IF(n * n >= 1024)
    for (int64_t k1 = 0; k1 < n; ++k1) {
        const int64_t hb = rowoff(h, k1, n);
        for (int64_t k2 = 0; k2 < n; ++k2) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r)
                acc += (double)xs[r * n + k1] * (double)xs[r * n + k2];
            const int64_t off = hb + k2 * hst;
            stt(h, off, (float)((double)ldt(h, off) + acc));
        }
    }
    return RAD_OK;
}
