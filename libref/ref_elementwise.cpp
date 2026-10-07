/* ref_elementwise.cpp -- the norms and the elementwise ops.
 *
 * Nothing here is interesting except where the fused forms round, and those places are the whole
 * reason the ops exist: `rmsnorm_add` is one pass in every real implementation, so the reference
 * has to say exactly which of its two intermediates is rounded and which is not, or a fused kernel
 * checked against it fails on a tolerance nobody can explain.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>

using namespace ref;

/* Rows of a tensor read as [rows, n]. `n` is a declared parameter and the row count is whatever
 * the operand actually carries, which is the rule this plugin follows everywhere (ref_ops.h). */
static inline int64_t rows_of(const RadTensor* t, int64_t n) {
    return n > 0 ? numel(t) / n : 0;
}

/* ------------------------------------------------------------------ rmsnorm */
/* y = x * rsqrt(mean(x^2) + eps) * (w + w_add)
 *
 * The mean is over the LAST n elements of a row and is accumulated in f32 in ascending index
 * order. `w_add` is GemmaRMSNorm's `1 +`: Qwen3.5, Qwen3-Next and Gemma store a zero-centred gain
 * and use 1 + w, and forming that in the weight's dtype first loses 0.3% per channel (libr4d's
 * r4d_rmsnorm_had_quant_i8 says the same). It defaults to 0, so a plain RMS norm is the plain
 * call. */
int ref_rmsnorm(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const int64_t round_before_weight = p_int(a, "round_before_weight", 0);
    if (round_before_weight != 0 && round_before_weight != 1) return RAD_E_INVAL;
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(w) < n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y), ws = laststride(w);
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { float v = ldt(x, xb + i * xs); ss += v * v; }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            float value = ldt(x, xb + i * xs) * s;
            if (round_before_weight) {
                alignas(8) unsigned char narrowed[8] = {};
                st_dt(x->dtype, narrowed, 0, value);
                value = ld_dt(x->dtype, narrowed, 0);
            }
            stt(y, yb + i * ys, value * (ldt(w, i * ws) + wadd));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rmsnorm_add */
/* residual_out = x + residual;  y = rmsnorm(residual_out) * (w + w_add)
 *
 * THE ROUNDING IS THE CONTRACT. The sum is formed in f32 and the variance is taken from the
 * UNROUNDED sum, but the multiply that produces y reads the sum back from residual_out AFTER it
 * has been narrowed to that operand's dtype. That is what vLLM's fused_add_rms_norm does and what
 * every fused kernel in this project will do, because the residual has to be stored anyway and
 * re-reading it is free. A reference that kept f32 all the way through would be half a ULP better
 * and would disagree with every kernel it certifies.
 */
int ref_rmsnorm_add(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *res = &a->t[1], *w = &a->t[2], *y = &a->t[3], *ro = &a->t[4];
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(ro) < rows * n ||
        numel(res) < rows * n || numel(w) < n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), rb = rowoff(res, r, n);
        const int64_t yb = rowoff(y, r, n), ob = rowoff(ro, r, n);
        const int64_t xs = laststride(x), rs = laststride(res);
        const int64_t ys = laststride(y), os = laststride(ro), ws = laststride(w);
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float v = ldt(x, xb + i * xs) + ldt(res, rb + i * rs);
            ss += v * v;
            stt(ro, ob + i * os, v);
        }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys, ldt(ro, ob + i * os) * s * (ldt(w, i * ws) + wadd));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ layernorm */
/* y = (x - mean) * rsqrt(var + eps) * w + b, with the BIASED variance (divide by n, not n-1),
 * which is what every transformer implementation means by layer norm. `b` is optional: a ViT that
 * dropped its bias passes a null operand rather than a zero tensor. */
int ref_layernorm(const RadArgs* a, RadStream) {
    if (a->n_t < 4 || !a->t[0].data || !a->t[1].data || !a->t[3].data) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *y = &a->t[3];
    const RadTensor* b = t_in(a, 2);
    const int64_t n = p_int(a, "n", w->rank ? w->shape[w->rank - 1] : 0);
    const float eps = p_f32(a, "eps", 1e-5f);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(x, n);
    if (rows <= 0 || numel(y) < rows * n || numel(w) < n) return RAD_E_SHAPE;
    if (b && numel(b) < n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y), ws = laststride(w);
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) sum += ldt(x, xb + i * xs);
        const float mean = sum / (float)n;
        float ss = 0.0f;
        for (int64_t i = 0; i < n; ++i) { float d = ldt(x, xb + i * xs) - mean; ss += d * d; }
        const float s = 1.0f / std::sqrt(ss / (float)n + eps);
        for (int64_t i = 0; i < n; ++i) {
            float v = (ldt(x, xb + i * xs) - mean) * s * ldt(w, i * ws);
            if (b) v += ldt(b, i * laststride(b));
            stt(y, yb + i * ys, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ grid_embed */
/* A learned side x side table of position rows, resampled to a grid of its own size per row of x
 * and added in: the vision tower's position embedding, which is trained at one grid and applied to
 * images of any shape.
 *
 * THE SAMPLING IS BILINEAR WITH THE CORNERS ALIGNED: grid row r of a grid of height g sits at table
 * coordinate r * (side - 1) / max(g - 1, 1), in f32, and its two taps are the floor and the floor
 * plus one, clamped to the table, weighted by distance. The four products are summed in f32 in the
 * order (top-left, top-right, bottom-left, bottom-right) and rounded to x's dtype -- the position
 * embedding as a tensor of its own -- and only then added to x, which is rounded again. */
static float round_to(uint32_t dt, float v) {
    return dt == RAD_BF16 ? rad_bf16_to_f32(rad_f32_to_bf16(v)) : v;
}

int ref_grid_embed(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
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

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gr = ld_int(crd->dtype, crd->data, 0 * cs + r * ms);
        const int64_t gc = ld_int(crd->dtype, crd->data, 1 * cs + r * ms);
        const int64_t gh = ld_int(crd->dtype, crd->data, 2 * cs + r * ms);
        const int64_t gw = ld_int(crd->dtype, crd->data, 3 * cs + r * ms);
        if (antialias) {
            // Separable half-pixel triangle filtering. Widen the footprint when reducing, then
            // normalize only surviving taps at an edge. The table dtype determines rounding of
            // raw/normalized weights and the horizontal intermediate, as in GPU interpolate.
            auto narrow = [&](float v) {
                if (tab->dtype == RAD_BF16) return rad_bf16_to_f32(rad_f32_to_bf16(v));
                if (tab->dtype == RAD_F16) return rad_f16_to_f32(rad_f32_to_f16(v));
                return v;
            };
            std::vector<float> weights((size_t)(2 * side));
            float* wh = weights.data();
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
            const int64_t grid[4] = { gr, gc, gh, gw };
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
                int64_t tap = (int64_t)fl + k;
                t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
                const float d = std::fabs(src - fl - (float)k);
                w[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
            }
        };
        axis(gr, gh, th, wh);
        axis(gc, gw, tw, ww);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int ki = 0; ki < 2; ++ki)
                for (int kj = 0; kj < 2; ++kj)
                    acc += ldt(tab, rowoff(tab, th[ki] * side + tw[kj], n) + i * ts) *
                           (wh[ki] * ww[kj]);
            const float pe = round_to(x->dtype, acc);
            stt(x, xb + i * xs, ldt(x, xb + i * xs) + pe);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ add / mul */
/* Elementwise over the output's extent. `b` may carry one row instead of all of them, so a bias
 * add is this op rather than a second one: an operand with exactly `n` elements against an output
 * of rows*n broadcasts along rows, and nothing else broadcasts at all. Silent broadcasting of an
 * arbitrary shape is how a shape bug becomes a plausible number. */
static int elementwise2(const RadArgs* a, bool mul) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[2];
    const int64_t n = p_int(a, "n", numel(Y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(Y, n);
    if (rows <= 0 || numel(A) < rows * n) return RAD_E_SHAPE;
    const bool bcast = numel(B) == n && rows > 1;
    if (!bcast && numel(B) < rows * n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t ab = rowoff(A, r, n), yb = rowoff(Y, r, n);
        const int64_t bb = bcast ? 0 : rowoff(B, r, n);
        const int64_t as = laststride(A), bs = laststride(B), ys = laststride(Y);
        for (int64_t i = 0; i < n; ++i) {
            float u = ldt(A, ab + i * as), v = ldt(B, bb + i * bs);
            stt(Y, yb + i * ys, mul ? u * v : u + v);
        }
    }
    return RAD_OK;
}

int ref_add(const RadArgs* a, RadStream) { return elementwise2(a, false); }
int ref_mul(const RadArgs* a, RadStream) { return elementwise2(a, true); }

/* ------------------------------------------------------------------ scale_rows */
/* y[m, i] = x[m, i] * act(s[m]) -- one scalar per ROW, broadcast across the row.
 *
 * IT IS NOT `mul`, AND THE DIFFERENCE IS WHICH AXIS BROADCASTS. `mul` lets `b` carry one ROW and
 * broadcasts it down the rows, which is a bias's shape; this carries one COLUMN and broadcasts it
 * across them, which is a per-token gate's. Giving `mul` both rules would make them ambiguous
 * exactly when rows == n, and a silent choice between two broadcasts is how a shape bug becomes a
 * plausible number -- which is the rule the comment above elementwise2 already states.
 *
 * The activation is applied to the SCALAR and not to the product, which is what a gate means:
 * Qwen3.5-MoE's shared expert is scaled by `sigmoid(shared_expert_gate(x))`. */
/* `add` PRESENT IS THE FUSED GATE-AND-ADD FORM: y = narrow(x * act(s)) + add, and the NARROW is
 * the whole of why this is expressible as one op rather than an approximation of two. The pair it
 * stands in for writes the product into a bf16 buffer and then adds that buffer to `add`, so the
 * product is rounded BEFORE the sum sees it; an f32 product carried into the add would differ in
 * the last bit of every channel. Written here as the two stores it replaces, through `y` itself,
 * because the oracle's job is to be obviously the same rather than to be quick.
 *
 * `add` is read before `y` is written so the two may alias without the sum reading its own output. */
int ref_scale_rows(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *sc = &a->t[1], *y = &a->t[3];
    const RadTensor *ad = t_in(a, 2);
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n || numel(sc) < rows) return RAD_E_SHAPE;
    if (ad && numel(ad) < rows * n) return RAD_E_SHAPE;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;

    const int64_t ss = laststride(sc);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float v0 = ldt(sc, r * ss);
        const float v  = sig ? act_sigmoid(v0) : v0;
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y);
        const int64_t ab = ad ? rowoff(ad, r, n) : 0;
        const int64_t as = ad ? laststride(ad) : 1;
        for (int64_t i = 0; i < n; ++i) {
            const float av = ad ? ldt(ad, ab + i * as) : 0.0f;
            stt(y, yb + i * ys, ldt(x, xb + i * xs) * v);
            if (ad) stt(y, yb + i * ys, ldt(y, yb + i * ys) + av);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ silu_mul */
/* y[.., i] = silu(gate_up[.., i]) * gate_up[.., n + i]. The gate is the FIRST half, which is the
 * order rad-convert fuses gate and up into one weight in, so the two conventions cannot drift. */
int ref_silu_mul(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *gu = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t gb = rowoff(gu, r, 2 * n), yb = rowoff(y, r, n);
        const int64_t gs = laststride(gu), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i)
            stt(y, yb + i * ys, act_silu(ldt(gu, gb + i * gs)) * ldt(gu, gb + (n + i) * gs));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gelu / silu */
static int unary(const RadArgs* a, int which) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nel; ++i) {
        float v = ldt(x, offlin(x, i));
        stt(y, offlin(y, i), which == 0 ? act_gelu(v) : which == 1 ? act_silu(v) : act_sigmoid(v));
    }
    return RAD_OK;
}

int ref_gelu(const RadArgs* a, RadStream) { return unary(a, 0); }
int ref_silu(const RadArgs* a, RadStream) { return unary(a, 1); }

/* ------------------------------------------------------------------ sigmoid */
/* Its own op rather than a `which` on gelu/silu, because a gated shared expert issues it against
 * the router's output and the selector has to be able to name it (docs/OPS.md, "New ops"). Without
 * it that expert runs ungated, which is a different model and not a slower one. */
int ref_sigmoid(const RadArgs* a, RadStream) { return unary(a, 2); }

/* ------------------------------------------------------------------ fill */
/* y = value over [M, n], at y's own row pitch. The value is rounded once, by the store. */
int ref_fill(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0) return RAD_E_SHAPE;
    const float v = p_f32(a, "value", 0.0f);

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), ys = laststride(y);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, v);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ gather_rows */
/* y[i, :] = x[idx[i], :]. The op that turns "correct" into "affordable": without it a plugin runs
 * the lm_head GEMM over every token of a prefill chunk when the sampler wants one row per
 * sequence -- n_tok/n_seq times the work, and at max_tok 8192 against a 248K vocab the logits
 * buffer alone is 8 GiB (docs/OPS.md).
 *
 * `M` is a RANGE parameter, so the number of rows actually gathered is the extent of `y` and not
 * the declared worst case; `idx` is read for exactly that many entries. A negative index writes a
 * zero row, matching embed_lookup's padding rule -- it is what a step with fewer sampled rows
 * than the buffer's extent leaves behind, and zeros are the only answer that cannot be mistaken
 * for a real logit row. An index at or past x's row count is RAD_E_INVAL: that is a caller bug,
 * and reading it would be reading another sequence's hidden state. */
int ref_gather_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *y = &a->t[2];
    const int64_t n = p_int(a, "n", y->rank >= 1 ? y->shape[y->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(y, n);
    const int64_t src_rows = rows_of(x, n);
    if (M <= 0 || src_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    /* Validated before the loop, because an OpenMP body cannot return. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r >= src_rows) return RAD_E_INVAL;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r  = ld_int(idx->dtype, idx->data, offlin(idx, m));
        const int64_t yb = rowoff(y, m, n), ys = laststride(y);
        if (r < 0) { for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, 0.0f); continue; }
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, ldt(x, xb + i * xs));
    }
    return RAD_OK;
}


/* ------------------------------------------------------------------ scatter_rows */
/* The mirror of gather_rows, and it exists for the same reason that one does: without it the only
 * way to write M rows to M arbitrary destinations is M transfers, and at a few hundred bytes a row
 * that is dispatch cost with the data as a rounding error.
 *
 * `x` IS INOUT AND THE WRITE IS PARTIAL. Rows no index names keep whatever they held -- which is
 * the whole point when the destination is a pool other sequences are also using. An op that
 * returned a fresh `x` would have to copy everything it was not asked to change.
 *
 * DUPLICATE INDICES ARE NOT LEGAL, and the schema says so with RAD_OPD_F_IDX_UNIQUE rather than
 * leaving it to prose. Two rows landing on one destination has no defined winner on a parallel
 * machine, so an op that permitted it would be one whose reference implementation and whose real
 * kernel are allowed to disagree -- which is exactly what the oracle exists to rule out. */
int ref_scatter_rows(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *v = &a->t[0], *idx = &a->t[1], *x = &a->t[2];
    const int64_t n = p_int(a, "n", v->rank >= 1 ? v->shape[v->rank - 1] : 0);
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t M = rows_of(v, n);
    const int64_t dst_rows = rows_of(x, n);
    if (M <= 0 || dst_rows <= 0 || numel(idx) < M) return RAD_E_SHAPE;

    /* Validated before the loop, because an OpenMP body cannot return. A negative index is a
     * skip, mirroring the zero row gather_rows writes for one: both mean "this slot has no
     * counterpart", and on this side the honest response is to leave the destination alone. */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r >= dst_rows) return RAD_E_INVAL;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t r = ld_int(idx->dtype, idx->data, offlin(idx, m));
        if (r < 0) continue;
        const int64_t vb = rowoff(v, m, n), vs = laststride(v);
        const int64_t xb = rowoff(x, r, n), xs = laststride(x);
        for (int64_t i = 0; i < n; ++i) stt(x, xb + i * xs, ldt(v, vb + i * vs));
    }
    return RAD_OK;
}
/* ------------------------------------------------------------------ cast */
/* `from` and `to` are in the schema because the SELECTOR needs them -- a kernel that converts bf16
 * to int8 is a different kernel from one that converts f32 to f16, and the constraint has to be
 * expressible. Ref does not read them: the tensors carry their own dtypes and those are what the
 * bytes actually are. A disagreement between the parameter and the tensor is a caller bug that
 * ref cannot diagnose better than the caller can. */
int ref_cast(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t nel = numel(y);
    if (nel <= 0 || numel(x) < nel) return RAD_E_SHAPE;

    /* Integer targets and sources go through the integer path, so a token id or a block index
     * survives a cast that f32 would round. */
    const bool int_pair = (x->dtype == RAD_I32 || x->dtype == RAD_I64 || x->dtype == RAD_U32 ||
                           x->dtype == RAD_I16) &&
                          (y->dtype == RAD_I32 || y->dtype == RAD_I64 || y->dtype == RAD_U32 ||
                           y->dtype == RAD_I16);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nel; ++i) {
        if (int_pair) st_int(y->dtype, y->data, offlin(y, i), ld_int(x->dtype, x->data, offlin(x, i)));
        else          stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ softmax */
/* Max-subtracted, f32, over the last n elements of a row. The max subtraction is not an
 * optimisation: without it a 60-nat logit range overflows, and every kernel this is an oracle for
 * does it, so the reference must too or the two differ by more than rounding. */
int ref_softmax(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    const int64_t n = p_int(a, "n", numel(y));
    if (n <= 0) return RAD_E_SHAPE;
    const int64_t rows = rows_of(y, n);
    if (rows <= 0 || numel(x) < rows * n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), yb = rowoff(y, r, n);
        const int64_t xs = laststride(x), ys = laststride(y);
        float mx = -INFINITY;
        for (int64_t i = 0; i < n; ++i) { float v = ldt(x, xb + i * xs); if (v > mx) mx = v; }
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            float e = ref_exp(ldt(x, xb + i * xs) - mx);
            sum += e;
            stt(y, yb + i * ys, e);
        }
        const float inv = 1.0f / sum;
        for (int64_t i = 0; i < n; ++i) stt(y, yb + i * ys, ldt(y, yb + i * ys) * inv);
    }
    return RAD_OK;
}
