/* ref_quant.cpp -- activation quantisation, the Walsh-Hadamard rotation, and the fused forms.
 *
 * Activations in this engine are ROTATED INT8, not fp8: several times better error at the same
 * width, and the rotation is what makes a single scale per row viable at all (docs/OPS.md).
 *
 * ============================== WHAT `group` MEANS ==============================
 *
 * docs/OPS.md gives all five ops one parameter called `group` and it does not mean the same thing
 * in all five. Ref fixes the reading, because a fallback that guessed differently from the fast
 * kernel would produce plausible wrong numbers:
 *
 *   quant_act_i8   `group` is the number of contiguous elements sharing one f32 scale, so `scale`
 *                  is [M, ceil(n/group)]. group <= 0 or group >= n is one scale per ROW, which is
 *                  what every int8 GEMM in libr4d consumes.
 *   dequant        the same reading, inverted: the scale stream's extent says how many groups
 *                  there are and `group` says how wide each is.
 *   had_*          `group` is the WALSH-HADAMARD ROTATION WIDTH (libr4d calls it `had`) and the
 *                  int8 scale is per ROW, carrying the 1/sqrt(group) normalisation. That is the
 *                  reading docs/OPS.md's own prose forces -- "the rotation (a per-group
 *                  Walsh-Hadamard) is what makes a single scale per row viable" -- and it is what
 *                  r4d_had_quant_act_i8 does. A rotation block may not cross a tensor-parallel
 *                  shard boundary, so the width has to divide the PER-RANK n.
 *
 * ============================== THE ROTATION ==============================
 *
 * Unnormalised butterflies at ascending strides 1, 2, 4, ... group/2, pairing i with i+h where
 * (i & h) == 0 -- the same order and the same pairing as r4d_fwht.h, so the two agree bit for bit
 * on the same input. The 1/sqrt(group) is folded into the scale rather than applied to the values,
 * again because that is what the fast kernel does: dividing every element by a constant cannot
 * change which int8 code it rounds to, and folding it costs nothing.
 */
#include "ref_common.h"
#include "ref_gemm.h"   /* Mat and Scales: gemm_nt_q_gated reads a quantised operand the same way */
#include "ref_ops.h"

using namespace ref;

/* Symmetric int8 everywhere below: q = round(x * 127 / amax), scale = amax / 127, with the store
 * rounding to nearest and saturating (ref_common.h). An all-zero row gets scale 0 and codes 0,
 * which dequantises back to zero -- the alternative, a tiny epsilon scale, turns a zero row into
 * denormal noise in the GEMM that reads it. */

/* ------------------------------------------------------------------ quant_act_i8 */
int ref_quant_act_i8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    /* THE ASUM FORM IS A DIFFERENT CONTRACT, AND THIS DECLINES IT BY NAME. `asum` present selects
     * libr4d's asymmetric form, of which the extra output is only the visible half: that kernel
     * writes ONE SCALE PER ROW over the whole of `n` -- `group` is the asum grouping width there,
     * not the scale granularity -- and it stores `q` in fragment order rather than row-major. Run
     * beside it, the symmetric quantiser below compares neither plane against what it means.
     *
     * The name matters more than the declining. Without it the per-group scale plane this code
     * writes is simply wider than the per-row one the caller allocated, the size check refuses,
     * and a sweep reports the ENGINE'S operands as the fault -- the production 2-bit draft head's
     * quantiser among them. */
    if (a->n_t > 3 && a->t[3].data) return RAD_E_UNSUPPORTED;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            float amax = 0.0f;
            for (int64_t i = k0; i < k1; ++i) {
                float m = std::fabs(ldt(x, xb + i * xs));
                if (m > amax) amax = m;
            }
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            for (int64_t i = k0; i < k1; ++i)
                st_dt(q->dtype, q->data, qb + i * qs, ldt(x, xb + i * xs) * inv);
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), amax / 127.0f);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ had_quant_act_i8 */
int ref_had_quant_act_i8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float buf[REF_HAD_MAX];
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        /* amax is over the WHOLE rotated row, because the scale is per row. */
        float amax = 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i) buf[i] = ldt(x, xb + (b + i) * xs);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i) { float m = std::fabs(buf[i]); if (m > amax) amax = m; }
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i) buf[i] = ldt(x, xb + (b + i) * xs);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i)
                st_dt(q->dtype, q->data, qb + (b + i) * qs, buf[i] * inv);
        }
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_had_quant_i8 */
/* The fused form of rmsnorm + hadamard + quant_act_i8, which is what a w4a8 layer wants where a
 * bf16 layer wants a plain rmsnorm (spec §2.3). A plugin asks for this first and emits the three
 * unfused ops when it does not resolve; ref resolves it, so the fallback path is one launch.
 *
 * `residual` PRESENT IS THE fused_add_rms_norm FORM: the add happens first, the sum is written
 * back into `residual` (that is what makes it INOUT and what the next layer's shortcut reads), and
 * the norm is taken over the sum. The variance comes from the f32 sum and the rotation reads the
 * sum back AFTER it has been narrowed to the residual tensor's own dtype, which is what the pair
 * of kernels this replaces does -- matching it is what makes the fused form substitutable for
 * that pair.
 *
 * `out_bf16` is the post-norm, PRE-ROTATION tensor for a layer that also has a bf16 consumer of
 * it. Pre-rotation because the rotation is an artefact of the int8 path; a bf16 consumer wants the
 * normalised activation, not a Walsh-Hadamard of it. */
int ref_rmsnorm_had_quant_i8(const RadArgs* a, RadStream) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = t_in(a, 1);
    const RadTensor* ob  = t_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const int64_t g = p_int(a, "group", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows || numel(w) < n) return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob && numel(ob) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float buf[REF_HAD_MAX];
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q), ws = laststride(w);

        /* The residual add, written back before anything reads it, so what follows reads the
         * narrowed sum rather than the f32 one. */
        const int64_t rb = res ? rowoff(res, r, n) : 0;
        const int64_t rst = res ? laststride(res) : 1;
        if (res)
            for (int64_t i = 0; i < n; ++i)
                stt(res, rb + i * rst, ldt(x, xb + i * xs) + ldt(res, rb + i * rst));

        auto src = [&](int64_t i) {
            return res ? ldt(res, rb + i * rst) : ldt(x, xb + i * xs);
        };

        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { const float v = src(i); ss += v * v; }
        const float sc = 1.0f / std::sqrt(ss / (float)n + eps);

        if (ob) {
            const int64_t obb = rowoff(ob, r, n), obs = laststride(ob);
            for (int64_t i = 0; i < n; ++i)
                stt(ob, obb + i * obs, src(i) * sc * (ldt(w, i * ws) + wadd));
        }

        float amax = 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i)
                buf[i] = src(b + i) * sc * (ldt(w, (b + i) * ws) + wadd);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i) { float m = std::fabs(buf[i]); if (m > amax) amax = m; }
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i)
                buf[i] = src(b + i) * sc * (ldt(w, (b + i) * ws) + wadd);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i)
                st_dt(q->dtype, q->data, qb + (b + i) * qs, buf[i] * inv);
        }
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gated_had_quant_i8 */
/* An elementwise gated product, then the rotation and the quantiser. Two of the four rotated sites
 * in a w4a8 model have this shape and differ by one multiply, so `act` picks between them:
 * silu(a)*b is down_proj's input, sigmoid(a)*b is o_proj's on a gated-attention layer
 * (Qwen3_5Attention computes attn_output * sigmoid(gate)). It defaults to silu, which is the form
 * docs/OPS.md's `silu_mul` names. `mode` is libr4d's integer spelling of the same choice and is
 * accepted where `act` is absent, so a caller written against either name works.
 *
 * `b` ABSENT IS THE PACKED FORM: `a` is [M, 2n], the gate is its first half and the up its second.
 * That is vLLM's layout and the one a fused gate_up projection produces, so both readings have to
 * exist; two operands with the second optional is the only shape that expresses both. */
int ref_gated_had_quant_i8(const RadArgs* a, RadStream) {
    if (a->n_t < 4) return RAD_E_INVAL;
    const RadTensor *ga = &a->t[0], *q = &a->t[2], *s = &a->t[3];
    const RadTensor* up = t_in(a, 1);
    if (!ga->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, (int)p_int(a, "mode", 0));
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (act != 0 && act != 1) return RAD_E_INVAL;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(q) / n;
    if (rows <= 0 || numel(s) < rows) return RAD_E_SHAPE;
    if (numel(ga) < rows * (up ? n : 2 * n)) return RAD_E_SHAPE;
    if (up && numel(up) < rows * n) return RAD_E_SHAPE;
    const float norm = 1.0f / std::sqrt((float)g);

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float buf[REF_HAD_MAX];
        const int64_t gb = rowoff(ga, r, up ? n : 2 * n), qb = rowoff(q, r, n);
        const int64_t gs = laststride(ga), qs = laststride(q);
        const int64_t ub = up ? rowoff(up, r, n) : 0;
        const int64_t us = up ? laststride(up) : 0;
        auto gated = [&](int64_t i) {
            const float gate = ldt(ga, gb + i * gs);
            const float x    = up ? ldt(up, ub + i * us) : ldt(ga, gb + (n + i) * gs);
            return (act == 1 ? act_sigmoid(gate) : act_silu(gate)) * x;
        };
        float amax = 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i) buf[i] = gated(b + i);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i) { float m = std::fabs(buf[i]); if (m > amax) amax = m; }
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (int64_t b = 0; b < n; b += g) {
            for (int64_t i = 0; i < g; ++i) buf[i] = gated(b + i);
            fwht(buf, g);
            for (int64_t i = 0; i < g; ++i)
                st_dt(q->dtype, q->data, qb + (b + i) * qs, buf[i] * inv);
        }
        stt(s, r * laststride(s), amax / 127.0f * norm);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ dequant */
/* q * scale, with the scale stream's own extent deciding how many groups there are. The inverse of
 * quant_act_i8 and NOT of the rotated forms: undoing a rotation is a second Hadamard, and an op
 * that silently did one would make `dequant(had_quant(x)) == x` look true when the two have
 * different meanings. A caller that wants the rotation undone asks for it. */
int ref_dequant(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *s = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    const int64_t ngrp = numel(s) / rows > 0 ? numel(s) / rows : 1;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0) g = (n + ngrp - 1) / ngrp;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t qb = rowoff(q, r, n), yb = rowoff(y, r, n), sb = rowoff(s, r, ngrp);
        const int64_t qs = laststride(q), ys = laststride(y), ss = laststride(s);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t gi = g > 0 ? i / g : 0;
            const float sc = ldt(s, sb + (gi < ngrp ? gi : ngrp - 1) * ss);
            stt(y, yb + i * ys, ldt(q, qb + i * qs) * sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ quant_act_fp8
 *
 * The oracle for the block-scaled FP8 family. Without a reference for `quant_act_fp8` the one
 * activation format the production checkpoints ship in is the one format rad-kbench cannot
 * measure (spec §17).
 *
 * The arithmetic is r4d_quant_act_fp8.hip's, restated rather than approximated, because the two
 * have to agree on three choices that are all defensible and none of which are derivable:
 *
 *   448        E4M3's largest finite value, and the scale is `amax / 448` -- a DEQUANT multiplier,
 *              the same sense as the checkpoint's `weight_scale_inv`, so the GEMM's fold
 *              multiplies the two scales together and nothing has to be inverted.
 *   scale 1    for an all-zero group, where `amax / 448` is zero and its reciprocal infinite. NOT
 *              the zero scale ref_quant_act_i8 uses: that kernel's consumers read a per-row int8
 *              scale and a zero row is a zero row, while here the GEMM multiplies the two block
 *              scales and a zero would erase the weight's.
 *   the clamp  is not defensive. `amax * (1/scale)` is 448 in exact arithmetic and can land an ulp
 *              above it once the reciprocal is rounded, which encodes as a NaN rather than as the
 *              largest finite.
 *
 * `M` and `n` are the ROW COUNT and the ROW WIDTH, and rows are dim 0 -- so a rank-3 activation
 * ([tokens, heads, width], which is what a GDN block hands its out-projection) is one row per
 * token, not one per head. That is the reading the consuming GEMM forces: `n` is its K.
 */
int ref_quant_act_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;

    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
            float amax = 0.0f;
            for (int64_t i = k0; i < k1; ++i) {
                const float m = std::fabs(ldt(x, xb + i * xs));
                if (m > amax) amax = m;
            }
            const float sc  = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            for (int64_t i = k0; i < k1; ++i) {
                float v = ldt(x, xb + i * xs) * inv;
                if (v > 448.0f) v = 448.0f;
                if (v < -448.0f) v = -448.0f;
                st_dt(q->dtype, q->data, qb + i * qs, v);
            }
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- had_quant_act_fp8 */
/* The same quantiser with a `group`-wide Hadamard in front of the absmax, which is the only place
 * a rotation can go: the whole benefit is that the maximum the grid is scaled by gets smaller.
 *
 * THE sqrt(g) IS NOT UNDONE HERE, and that is a DELIBERATE difference from ref_had_quant_act_i8
 * above, which folds 1/sqrt(g) into its own scale. H is symmetric, so
 * <Hw, Ha> = g <w, a>, and the residual g has to be divided out exactly once between the two
 * sides. The int8 op splits it; this one gives the WHOLE factor to the weight, where dividing a
 * bf16 scale by a power of two is an exponent shift and therefore exact. Splitting it would make
 * the stored weight scale bf16(amax/7/sqrt(g)) instead of bf16(amax/7)/g -- and the codes were
 * chosen against amax/7, so that rounding lands as a ~0.2% multiplicative error on every group,
 * which is a fifth of the int4 grid's own error bought for nothing.
 *
 * So this op's output is NOT interchangeable with quant_act_fp8's, and its consumer is not free
 * either: only a weight plane stored as w4a8h carries the matching rotation and the matching /g. */
int ref_had_quant_act_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    /* n % g != 0 is refused rather than rounded: a rotation block that straddled the end of the
     * row would mix real values with padding, and no weight block was rotated that way. */
    if (n <= 0 || !pow2(g) || n % g != 0) return RAD_E_SHAPE;
    if (g > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = n / g;
    if (rows <= 0 || numel(q) < rows * n) return RAD_E_SHAPE;
    if (numel(s) < rows * ngrp) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        float buf[REF_HAD_MAX];
        const int64_t xb = rowoff(x, r, n), qb = rowoff(q, r, n);
        const int64_t xs = laststride(x), qs = laststride(q);
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * g;
            for (int64_t i = 0; i < g; ++i) buf[i] = ldt(x, xb + (k0 + i) * xs);
            fwht(buf, g);
            float amax = 0.0f;
            for (int64_t i = 0; i < g; ++i) {
                const float m = std::fabs(buf[i]);
                if (m > amax) amax = m;
            }
            const float sc  = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            for (int64_t i = 0; i < g; ++i) {
                float v = buf[i] * inv;
                if (v > 448.0f) v = 448.0f;
                if (v < -448.0f) v = -448.0f;
                st_dt(q->dtype, q->data, qb + (k0 + i) * qs, v);
            }
            stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
        }
    }
    return RAD_OK;
}

/* ---------------------------------------------------------------- the Gram accumulator
 *
 * h[k1][k2] += sum over rows of x[m][k1] * x[m][k2], where x is the DEQUANTISED E4M3 input, or
 * a bf16 input as it stands when no scale is passed -- the second moment GPTQ's error feedback
 * needs, over exactly the operand the expert GEMM contracts. libr4d's kernel is the device half;
 * this is the oracle for it.
 *
 * THE ACCUMULATOR IS READ AS WELL AS WRITTEN, which is the one thing a reader coming from every
 * other op in this file has to hold: `h` is INOUT, the caller zeroes it once, and each call adds
 * its own block. An oracle that stored instead of added would agree with the kernel on a single
 * call and disagree on a corpus.
 *
 * Rows are dequantised once into a scratch plane rather than inside the k2 loop: the triple loop
 * is O(n^2 * rows) and a per-element dtype dispatch inside it is the difference between an oracle
 * that finishes and one that does not. The accumulation is f64 here and f32 on the device on
 * purpose -- an oracle should be tighter than the thing it checks, not bit-identical to it. */
int ref_gram_accum(const RadArgs* a, RadStream) {
    /* `a_scale` is optional: absent, `a` is read as it stands, which is the bf16 form. */
    if (a->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = t_in(a, 0), *s = t_in(a, 1), *h = t_in(a, 2);
    if (!x || !h) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t g = p_int(a, "group", 0);
    if (n <= 0 || (s && g <= 0)) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    const int64_t ngrp = s ? (n + g - 1) / g : 0;
    if (rows <= 0 || (s && numel(s) < rows * ngrp)) return RAD_E_SHAPE;
    if (h->rank != 2 || h->shape[0] != n || h->shape[1] != n) return RAD_E_SHAPE;

    std::vector<float> xs((size_t)rows * (size_t)n);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), xst = laststride(x);
        const int64_t sb = s ? rowoff(s, r, ngrp) : 0, sst = s ? laststride(s) : 0;
        for (int64_t k = 0; k < n; ++k)
            xs[(size_t)r * (size_t)n + (size_t)k] =
                ldt(x, xb + k * xst) * (s ? ldt(s, sb + (k / g) * sst) : 1.0f);
    }

    const int64_t hst = laststride(h);
    #pragma omp parallel for schedule(static)
    for (int64_t k1 = 0; k1 < n; ++k1) {
        const int64_t hb = rowoff(h, k1, n);
        for (int64_t k2 = 0; k2 < n; ++k2) {
            double acc = 0.0;
            for (int64_t r = 0; r < rows; ++r)
                acc += (double)xs[(size_t)r * (size_t)n + (size_t)k1] *
                       (double)xs[(size_t)r * (size_t)n + (size_t)k2];
            const int64_t off = hb + k2 * hst;
            stt(h, off, (float)((double)ldt(h, off) + acc));
        }
    }
    return RAD_OK;
}

/* ======================================================================== the fp8 fusions
 *
 * THREE OPS THAT EXIST SO THAT A HOST RUN IS POSSIBLE AT ALL. libr4d fuses the fp8 activation
 * quantiser into whichever kernel already has the row in registers, and declares the fused form as
 * its own op. Without a host kernel for those names, `RADIANCE_HOST_DOMAIN` -- the only thing that
 * can answer "is the GRAPH right", because the per-op oracle reads the same graph both sides --
 * cannot start on an fp8 model ("site 'host' is unavailable for this band").
 *
 * These are the unfused arithmetic written out in one pass, which is what a reference is for. They
 * are NOT required to be bit-identical to libr4d's kernels the way those kernels are required to be
 * bit-identical to the pair they replace: the oracle compares within a tolerance, and a reference
 * that chased the device's rounding would stop being a statement of what the op MEANS.
 *
 * WHAT THEY DO FOLLOW is the bf16 round-trip, because that is meaning and not rounding. The unfused
 * pair writes a bf16 intermediate and the quantiser reads those bytes back, so the codes are taken
 * over a bf16-narrowed value; a reference that quantised the f32 would be a different function, and
 * `out_bf16` -- which the caller may ask for precisely so the fused and unfused forms leave the same
 * buffers behind -- would not hold the value the codes were taken over.
 */

/* One group-scaled fp8 row, from values already in `v`. The scale layout is quant_act_fp8's and
 * the reason for each constant is written there. */
static inline void fp8_row(const float* v, int64_t n, int64_t g, const RadTensor* q,
                           const RadTensor* s, int64_t r) {
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t qb = rowoff(q, r, n), qs = laststride(q);
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
        float amax = 0.0f;
        for (int64_t i = k0; i < k1; ++i) { const float m = std::fabs(v[i]); if (m > amax) amax = m; }
        const float sc  = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
        const float inv = 1.0f / sc;
        for (int64_t i = k0; i < k1; ++i) {
            float c = v[i] * inv;
            if (c >  448.0f) c =  448.0f;
            if (c < -448.0f) c = -448.0f;
            st_dt(q->dtype, q->data, qb + i * qs, c);
        }
        stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
    }
}

/* ------------------------------------------------------------- rmsnorm_quant_fp8 */
/* `x`, `residual`:inout?, `w`w -> `q`, `scale`, `out_bf16`?
 *
 * residual present is the fused_add_rms_norm form, and the add is STORED BEFORE THE VARIANCE READS
 * IT -- the sum the norm is taken over is the bf16-narrowed one, not the f32 one. libr4d's kernel
 * makes the same choice and says at length why: the pair it replaces writes a bf16 residual and
 * reads those bytes back, and half a ULP per channel is the difference between the same text and
 * very nearly the same text. `wadd` is added to every gain in f32 for the reason ref_rmsnorm gives.
 */
int ref_rmsnorm_quant_fp8(const RadArgs* a, RadStream) {
    if (a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[2], *q = &a->t[3], *s = &a->t[4];
    const RadTensor* res = t_in(a, 1);
    const RadTensor* ob  = t_in(a, 5);
    if (!x->data || !w->data || !q->data || !s->data) return RAD_E_INVAL;
    const int64_t n    = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float   eps  = p_f32(a, "eps", 1e-6f);
    const float   wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(q) < rows * n || numel(s) < rows * ngrp || numel(w) < n)
        return RAD_E_SHAPE;
    if (res && numel(res) < rows * n) return RAD_E_SHAPE;
    if (ob  && numel(ob)  < rows * n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> v((size_t)n);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x), ws = laststride(w);
        const int64_t rb = res ? rowoff(res, r, n) : 0;
        const int64_t rst = res ? laststride(res) : 1;
        if (res)
            for (int64_t i = 0; i < n; ++i)
                stt(res, rb + i * rst, ldt(x, xb + i * xs) + ldt(res, rb + i * rst));
        auto src = [&](int64_t i) { return res ? ldt(res, rb + i * rst) : ldt(x, xb + i * xs); };

        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { const float t = src(i); ss += t * t; }
        const float rs = 1.0f / std::sqrt(ss / (float)n + eps);

        /* NARROWED TO BF16 AND THEN QUANTISED, and written out where asked. The codes belong to
         * the value `out_bf16` holds; taking them over the f32 would make the two disagree. */
        const int64_t obb = ob ? rowoff(ob, r, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const uint16_t h = f32_to_bf16(src(i) * rs * (ldt(w, i * ws) + wadd));
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gated_quant_fp8 */
/* `gate_up` -> `q`, `scale`, `out_bf16`?
 *
 * The packed form only: `gate_up` is [M, 2n], gate first. `act` picks silu(gate)*up (down_proj's
 * input) from sigmoid(gate)*up (a gated attention o_proj's), the same two ref_gated_had_quant_i8
 * takes and under the same two names. */
int ref_gated_quant_fp8(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *q = &a->t[1], *s = &a->t[2];
    const RadTensor* ob = t_in(a, 3);
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

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> v((size_t)n);
        const int64_t gb = rowoff(gu, r, 2 * n), gs = laststride(gu);
        const int64_t obb = ob ? rowoff(ob, r, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const float gt = ldt(gu, gb + i * gs);
            const float up = ldt(gu, gb + (n + i) * gs);
            const float sg = 1.0f / (1.0f + std::exp(-gt));
            const uint16_t h = f32_to_bf16((act == 0 ? gt * sg : sg) * up);
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, r);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- gemm_nt_q_gated */
/* `a`, `a_scale`, `b`w, `b_scale`w -> `q`, `scale`, `out_bf16`?
 *
 * gemm_nt_q over a [N, K] gate_up plane -- N is the WEIGHT's rows, twice the fused width, gate half
 * first -- with gated_quant_fp8's pass in the epilogue, so the [M, N] bf16 tensor between them is
 * never written. `act` is REQUIRED here and optional on the quantiser, for the reason docs/OPS.md
 * gives: an implementation is free to compile one gate, and an optional parameter would let it
 * resolve for another.
 *
 * BYTE-IDENTICAL TO THE PAIR IT REPLACES, which is a requirement and not a hope -- the libr4d row
 * that serves this op makes the same claim, and a fold that is a slightly different function from
 * gemm_nt_q followed by gated_quant_fp8 is a different model. Two things earn it here: the product
 * comes from `quant_dot`, which is also what gemm_nt_q's plane comes from and is the one place the
 * grouping and the scale order are written down; and both halves are narrowed to bf16 BEFORE the
 * gate, because the unfused form writes a bf16 [M, N] tensor and gated_quant_fp8 reads those bytes
 * back. Taking either half off the f32 accumulator is a different function. ref_test holds both.
 *
 * A NOTE FOR ANYONE COMPARING THIS AGAINST A DEVICE KERNEL ON THE ENGINE'S OWN OPERANDS. It will
 * disagree, completely, and the kernel is not wrong: libr4d stores an fp8a8 weight in WMMA FRAGMENT
 * ORDER (r4d.fp8.w.frag64), the same bytes in a different arrangement, and libref reads [N, K]
 * row-major. Two readings of one permuted plane are uncorrelated tensors, which shows up as a
 * relative error around 1.35 on every operand from element zero. The per-op oracle inverts the
 * layout through the row's own RadUnrelayoutFn, back to the encoding's canonical planes, before it
 * compares, and says so when a row publishes no inverse; a comparison that skips that step is
 * measuring the permutation.
 *
 * The product is formed a row at a time rather than all at once. A reference may be slow and may
 * not be surprising about memory: [M, N] at a prefill M is hundreds of megabytes for a value the
 * op exists to avoid materialising. */
int ref_gemm_nt_q_gated(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *q = &a->t[4], *s = &a->t[5];
    const RadTensor* as_ = t_in(a, 1);
    const RadTensor* bs_ = t_in(a, 3);
    const RadTensor* ob  = t_in(a, 6);
    if (!A->data || !B->data || !q->data || !s->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0 || (N & 1)) return RAD_E_SHAPE;
    const int64_t n = N / 2;
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, -1);
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t M = numel(q) / n;
    if (M <= 0 || numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (numel(s) < M * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < M * n) return RAD_E_SHAPE;

    const bool mx = B->dtype == RAD_FP4E2M1;
    const Scales SA = make_scales(as_, M, K, false);
    const Scales SB = make_scales(bs_, N, K, mx);
    const Mat ma = as_mat(A, M, K), mb = as_mat(B, N, K);

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> row((size_t)N), v((size_t)n);
        for (int64_t j = 0; j < N; ++j) row[(size_t)j] = quant_dot(ma, mb, SA, SB, m, j, K);
        const int64_t obb = ob ? rowoff(ob, m, n) : 0;
        const int64_t obs = ob ? laststride(ob) : 1;
        /* Both halves narrowed to bf16 before the gate; the header says why. */
        for (int64_t i = 0; i < n; ++i) {
            const float gt = bf16_to_f32(f32_to_bf16(row[(size_t)i]));
            const float up = bf16_to_f32(f32_to_bf16(row[(size_t)(n + i)]));
            const float sg = 1.0f / (1.0f + std::exp(-gt));
            const uint16_t h = f32_to_bf16((act == 0 ? gt * sg : sg) * up);
            if (ob) stt(ob, obb + i * obs, bf16_to_f32(h));
            v[(size_t)i] = bf16_to_f32(h);
        }
        fp8_row(v.data(), n, g, q, s, m);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------- dflash_select */
/* `cand`, `unary`, `hp`?, `anchor`, `pred`w, `succ`w -> `tokens`
 *
 * DFlash2's greedy walk over the candidate lattice. A block-diffusion drafter proposes every
 * position of a block in ONE backbone pass, so it cannot condition position l+1 on what it chose at
 * l; this puts that dependence back.
 *
 *   score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * hp[l][r] * succ[cand[l, c]][r]
 *   pid(0) = anchor[s * anchor_stride]        the token the target just committed
 *   pid(l) = cand[l - 1, chosen(l - 1)]       the walk's own previous pick
 *
 * `hp` ABSENT IS A PLANE OF ONES, which drops the trilinear form to a plain low-rank bigram and is
 * DSpark's Markov head rather than a degenerate case.
 *
 * THE WALK IS SERIAL IN l, which is why it is an op and not a GEMM and an argmax: scoring the full
 * [steps, K, K] edge tensor computes K times the edges that are ever read, because the predecessor
 * is not known until the previous step has been decided. The parallelism here is over sequences. */
/* ONE BODY FOR BOTH SPELLINGS. `psc` and `ssc` are the per-row (or one-for-all) f32 scales of an
 * E4M3 codebook and are null for the plain one, where every factor below is 1. A scale is applied
 * where the kernel applies it: the predecessor row's into `ax` once a step, the candidate row's
 * onto the finished dot -- so the two spellings agree on a codebook that is the same numbers either way. */
static int dflash_select_body(const RadArgs* a, const RadTensor* cand, const RadTensor* un,
                              const RadTensor* hp, const RadTensor* anc, const RadTensor* pred,
                              const RadTensor* psc, const RadTensor* succ, const RadTensor* ssc,
                              const RadTensor* out) {
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    /* `cand` is [M * steps, K] and it is the call; `M` is the band. See ref_common.h. */
    const int64_t M     = steps > 0 && K > 0
                        ? band_rows(numel(cand) / (steps * K), p_int(a, "M", 0)) : 0;
    const int64_t R     = p_int(a, "rank", pred->rank >= 2 ? pred->shape[pred->rank - 1] : 0);
    const int64_t nv    = p_int(a, "n_vocab", pred->rank >= 2 ? pred->shape[pred->rank - 2] : 0);
    const int64_t astr  = p_int(a, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_SHAPE;
    if (numel(un) < M * steps * K) return RAD_E_SHAPE;
    if (numel(anc) < (M - 1) * astr + 1 || numel(out) < M * steps) return RAD_E_SHAPE;
    if (numel(pred) < nv * R || numel(succ) < nv * R) return RAD_E_SHAPE;
    if (hp && numel(hp) < M * steps * R) return RAD_E_SHAPE;
    /* A scale plane is one element for the whole codebook or one a row, and nothing in between. */
    auto scale_ok = [&](const RadTensor* t) {
        return !t || (t->data && (numel(t) == 1 || numel(t) >= nv));
    };
    if (!scale_ok(psc) || !scale_ok(ssc)) return RAD_E_SHAPE;
    auto row_scale = [&](const RadTensor* t, int64_t id) -> float {
        if (!t) return 1.0f;
        return ldt(t, numel(t) == 1 ? 0 : id * laststride(t));
    };

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);
    const int64_t ps = laststride(pred), sxs = laststride(succ);
    const int64_t hs = hp ? laststride(hp) : 1;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> ax((size_t)R);
        int64_t pid = ld_int(anc->dtype, anc->data, offlin(anc, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t r     = m * steps + l;
            const int64_t cb    = rowoff(cand, r, K), ub = rowoff(un, r, K);
            const int64_t hb    = hp ? rowoff(hp, r, R) : 0;
            /* The half of the form that does not depend on the candidate, computed once and read
             * K times -- which is the shape the kernel is built around too. A predecessor outside
             * the vocabulary contributes nothing rather than reading past the codebook: at step 0
             * it is whatever the target committed, and a padding row's is not a token. */
            const bool ok = pid >= 0 && pid < nv;
            const float pscl = ok ? row_scale(psc, pid) : 0.0f;
            for (int64_t i = 0; i < R; ++i)
                ax[(size_t)i] = (ok ? ldt(pred, rowoff(pred, pid, R) + i * ps) * pscl : 0.0f) *
                                (hp ? ldt(hp, hb + i * hs) : 1.0f);

            int64_t best_c = 0;
            float   best_s = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                if (tok >= 0 && tok < nv) {
                    const int64_t sb = rowoff(succ, tok, R);
                    float dot = 0.0f;
                    for (int64_t i = 0; i < R; ++i) dot += ax[(size_t)i] * ldt(succ, sb + i * sxs);
                    sc += dot * row_scale(ssc, tok);
                }
                if (sc > best_s) { best_s = sc; best_c = c; }
            }
            pid = ld_int(cand->dtype, cand->data, cb + best_c * cs);
            st_int(out->dtype, out->data, rowoff(out, m, steps) + l * os, pid);
        }
    }
    return RAD_OK;
}

int ref_dflash_select(const RadArgs* a, RadStream) {
    if (a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *succ = &a->t[5], *out = &a->t[6];
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;
    return dflash_select_body(a, cand, un, t_in(a, 2), anc, pred, nullptr, succ, nullptr, out);
}

/* `cand`, `unary`, `hp`?, `anchor`, `pred`w, `pred_scale`w, `succ`w, `succ_scale`w -> `tokens`
 *
 * dflash_select over codebooks held as E4M3 codes: codebook[v][r] = codes[v][r] * scale[v], one f32
 * scale a row (or one for the whole table). A SEPARATE OP and not a dtype on the plain one, for the
 * reason embed_lookup_q is: the operand list differs and operands are positional. The two
 * codebooks are 2 x 121 MiB at bf16 on the shipped checkpoint; this halves them, and the KV cache
 * takes the difference. */
int ref_dflash_select_q(const RadArgs* a, RadStream) {
    if (a->n_t < 9) return RAD_E_INVAL;
    const RadTensor *cand = &a->t[0], *un = &a->t[1], *anc = &a->t[3];
    const RadTensor *pred = &a->t[4], *psc = &a->t[5], *succ = &a->t[6], *ssc = &a->t[7];
    const RadTensor* out = &a->t[8];
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data ||
        !psc->data || !ssc->data)
        return RAD_E_INVAL;
    return dflash_select_body(a, cand, un, t_in(a, 2), anc, pred, psc, succ, ssc, out);
}
