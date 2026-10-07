/* avx_misc.cpp -- the PLE pair, causal convolution, hashed n-gram ids, and two collectives.
 *
 * Grouped because none of them is a vector kernel and the reason differs in each case, which is
 * worth saying rather than leaving as an absence:
 *
 *   ple_gate IS vectorised, in the two places that carry the work -- the per-stream norms and the
 *   gated broadcast of `v`. What is not is the dot between the two normed streams, which is
 *   accumulated in f64 because libref accumulates it in f64 and this is the one op where that
 *   choice is visible: the dot feeds a SIGNED SQUARE ROOT, which has unbounded relative sensitivity
 *   near zero, so an f32 reduction of a 10240-wide product would move the gate by more than the
 *   tolerance on exactly the rows where the gate is near its hinge.
 *
 *   ple_conv is a depthwise convolution of width 4 over a channel, which is 4 multiplies per
 *   output element with the taps `dilation` apart. The channel loop is the parallel axis and the
 *   inner one is four terms; there is no vector work in it that the outer parallelism does not
 *   already have. What it gets instead is the rolling-window rule written out exactly.
 *
 *   ngram_ids is INTEGER HASHING with a data-dependent early cut, and its arithmetic is defined by
 *   torch's -- a wrapping 64-bit multiply and a FLOORING modulo, neither of which is C's. There is
 *   nothing here to vectorise and a great deal to get exactly right.
 *
 *   all_reduce / all_gather serve WORLD SIZE 1 and refuse anything wider by name. See the note on
 *   the collectives below; it is a deliberate limit, not an omission.
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

static inline float narrow1(uint32_t dt, float v) {
    uint32_t buf[2] = { 0, 0 };
    rad_store_f32(buf, dt, 0, v);
    return rad_load_f32(buf, dt, 0);
}

/* One group's reciprocal RMS over `n` contiguous f32. f64 accumulation, matching libref's
 * `group_rsqrt`: the rsqrt is per n-wide stream while the gain below it spans all hc*n, and that
 * asymmetry is the whole of what `group_size` means on Qwen4ExpTextRMSNorm. */
static float group_rsqrt_f64(const float* v, int64_t n, float eps) {
    double acc = 0;
    for (int64_t d = 0; d < n; ++d) acc += (double)v[d] * (double)v[d];
    return (float)(1.0 / std::sqrt(acc / (double)n + (double)eps));
}

/* ------------------------------------------------------------------ ple_gate */
/*   k        [M][hc*n]   key_proj(embedding), UN-normed
 *   q        [M][hc*n]   the incoming hyper-connection stream, UN-normed
 *   v        [M][n]      value_proj(embedding) -- ONE row, gated differently per stream
 *   w_key / w_query / w_conv   [hc*n]  gains
 *   gv       [M][hc*n]   out, UN-normed: what the residual add reads
 *   gvn      [M][hc*n]   out, normed:    what the convolution reads
 *
 * THE THIRD NORM IS TAKEN OVER WHAT THIS OP ITSELF WROTE, and it is READ BACK rather than carried
 * in registers -- which is why the op owns all three norms instead of the graph owning the last
 * one. Splitting it would make two definitions of the same rounding.
 *
 * THE SIGNED SQUARE ROOT IS IN THE REFERENCE'S ORDER: the magnitude is clamped to `gate_eps`, the
 * root is taken of the clamped magnitude, and the sign is reapplied afterwards. Clamping after the
 * root, or taking the root of the signed value, are both different functions near zero and the
 * gate spends its life near zero. */
AVX_KERNEL(ple_gate) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor* k = &a->t[0];
    const RadTensor* q = &a->t[1];
    const RadTensor* v = &a->t[2];
    const RadTensor* wk = &a->t[3];
    const RadTensor* wq = &a->t[4];
    const RadTensor* wc = &a->t[5];
    const RadTensor* gv = &a->t[6];
    const RadTensor* gvn = &a->t[7];

    const int64_t n = p_int(a, "n", 0);
    /* `v` is [M, n] and it is the call; `M` is the band. */
    const int64_t M = n > 0 ? band_rows(numel(v) / n, p_int(a, "M", 0)) : 0;
    const int64_t hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    /* The floor the gate's magnitude is clamped to before the square root. */
    const float geps = p_f32(a, "gate_eps", 1e-6f);
    if (M <= 0 || n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    if (numel(k) < M * hn || numel(q) < M * hn || numel(v) < M * n) return RAD_E_SHAPE;
    if (numel(wk) < hn || numel(wq) < hn || numel(wc) < hn) return RAD_E_SHAPE;
    if (numel(gv) < M * hn || numel(gvn) < M * hn) return RAD_E_SHAPE;

    /* The three gains, widened once with `wadd` folded in. One allocation -- the arena grows, so a
     * second scratch_raw would free this one under the worker threads. */
    float* gains = (float*)scratch_raw((size_t)(3 * hn) * sizeof(float) + 64);
    float* gk = gains;
    float* gq = gains + hn;
    float* gc = gains + 2 * hn;
    {
        const int64_t sk = laststride(wk), sq = laststride(wq), sc = laststride(wc);
        for (int64_t i = 0; i < hn; ++i) {
            gk[i] = ldt(wk, i * sk) + wadd;
            gq[i] = ldt(wq, i * sq) + wadd;
            gc[i] = ldt(wc, i * sc) + wadd;
        }
    }

    const int64_t gls = laststride(gv), gnls = laststride(gvn);

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t kb = rowoff(k, m, hn), qb = rowoff(q, m, hn), vb = rowoff(v, m, n);
        const int64_t gb = rowoff(gv, m, hn), nb = rowoff(gvn, m, hn);
        float* kv = scratch_f32((size_t)hn);
        float* qv = scratch_f32_b((size_t)hn);
        float* out = scratch_f32_c((size_t)(hn + n) + 16);
        float* vv = out + hn + 8;
        const float* kk = in_row(k, kb, hn, kv);
        const float* qq = in_row(q, qb, hn, qv);
        const float* v0 = in_row(v, vb, n, vv);
        if (v0 != vv) std::memcpy(vv, v0, (size_t)n * sizeof(float));

        for (int64_t c = 0; c < hc; ++c) {
            const float rk = group_rsqrt_f64(kk + c * n, n, eps);
            const float rq = group_rsqrt_f64(qq + c * n, n, eps);
            /* f64, and see the file header: the dot feeds a signed square root whose relative
             * sensitivity is unbounded near zero. */
            double dot = 0;
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                const float a1 = narrow1(k->dtype, kk[j] * rk * gk[j]);
                const float b1 = narrow1(q->dtype, qq[j] * rq * gq[j]);
                dot += (double)a1 * (double)b1;
            }
            float s = (float)(dot / std::sqrt((double)n));
            const float mag = std::sqrt(std::fabs(s) < geps ? geps : std::fabs(s));
            s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);
            const float g = 1.0f / (1.0f + std::exp(-s));

            const vf vg = vf_set1(g);
            int64_t d = 0;
            for (; d + VF_N <= n; d += VF_N)
                vf_storeu(out + c * n + d, vf_mul(vg, vf_loadu(vv + d)));
            for (; d < n; ++d) out[c * n + d] = g * vv[d];
        }
        if (gls == 1) out_row(gv, gb, hn, out);
        else for (int64_t j = 0; j < hn; ++j) stt(gv, gb + j * gls, out[j]);

        /* The third norm, over what was just written -- read BACK through gv's dtype, not carried
         * in registers. The round trip is done here rather than by re-reading the operand: same
         * arithmetic, and it does not depend on the store having retired. */
        if (gv->dtype != RAD_F32) {
            row_from_f32(gv->dtype, out, kv, hn);
            row_to_f32(gv->dtype, kv, out, hn);
        }
        for (int64_t c = 0; c < hc; ++c) {
            const float r = group_rsqrt_f64(out + c * n, n, eps);
            for (int64_t d = 0; d < n; ++d) {
                const int64_t j = c * n + d;
                qv[j] = narrow1(gvn->dtype, out[j] * r * gc[j]);
            }
        }
        if (gnls == 1) out_row(gvn, nb, hn, qv);
        else for (int64_t j = 0; j < hn; ++j) stt(gvn, nb + j * gnls, qv[j]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ causal_conv1d */
/* LFM2's short depthwise convolution has no activation or residual. Like PLE, its useful CPU
 * parallelism is over channels: each worker widens the taps and the old history once, then
 * walks that channel's tokens. Every sequence is validated before any worker changes state. */
AVX_KERNEL(causal_conv1d) {
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* w = rad_arg_in(a, 1);
    const RadTensor* state = rad_arg_in(a, 2);
    const RadTensor* cu = rad_arg_in(a, 3);
    const RadTensor* indices = rad_arg_in(a, 4);
    const RadTensor* initialized = rad_arg_in(a, 5);
    const RadTensor* accepted = rad_arg_in(a, 6);
    const RadTensor* y = rad_arg_in(a, 7);
    if (!x || !w || !state || !cu || !y) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), width = p_int(a, "width", 0);
    const int64_t rows = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    if (n <= 0 || width <= 0 || rows <= 0 || state->rank != 3) return RAD_E_SHAPE;
    const int64_t history = width - 1;
    constexpr int64_t kHistoryMax = 64;
    if (history > kHistoryMax) return RAD_E_UNSUPPORTED;
    int64_t packed_stride = 1;
    for (int i = 2; i >= 0; --i) {
        if (state->shape[i] != 1 && state->stride[i] != packed_stride) return RAD_E_SHAPE;
        packed_stride *= state->shape[i];
    }
    const int64_t depth = state->shape[2], count = numel(cu) - 1;
    if (count <= 0 || state->shape[1] != n || depth < history ||
        numel(w) < n * width || numel(y) < rows * n ||
        (indices && numel(indices) < count) ||
        (initialized && numel(initialized) < count) ||
        (accepted && numel(accepted) < count)) return RAD_E_SHAPE;
    auto integer = [](const RadTensor* t, int64_t i) { return ldi(t, offlin(t, i)); };
    if (integer(cu, 0) < 0 || integer(cu, count) > rows) return RAD_E_SHAPE;

    for (int64_t s = 0; s < count; ++s) {
        const int64_t begin = integer(cu, s), end = integer(cu, s + 1);
        const int64_t slot = indices ? integer(indices, s) : s;
        const int64_t offset = accepted ? integer(accepted, s) - 1 : 0;
        if (begin < 0 || end < begin || end > rows || slot < -1 || slot >= state->shape[0] ||
            offset < 0) return RAD_E_INVAL;
        if (offset > depth - history ||
            (accepted && history && end > begin && end - begin > depth - history + 1))
            return RAD_E_SHAPE;
        if (slot >= 0 && end > begin) {
            for (int64_t earlier = 0; earlier < s; ++earlier) {
                const int64_t other_slot = indices ? integer(indices, earlier) : earlier;
                if (other_slot == slot && integer(cu, earlier + 1) > integer(cu, earlier))
                    return RAD_E_INVAL;
            }
        }
    }

    const int64_t xr = rowoff(x, 1, n), xc = laststride(x);
    const int64_t yr = rowoff(y, 1, n), yc = laststride(y);
    const int64_t wr = rowoff(w, 1, width), wc = laststride(w);
    AVX_PARALLEL_FOR_IF(count * n >= 64)
    for (int64_t worker = 0; worker < count * n; ++worker) {
        const int64_t s = worker / n, channel = worker % n;
        const int64_t begin = integer(cu, s), length = integer(cu, s + 1) - begin;
        const int64_t slot = indices ? integer(indices, s) : s;
        if (slot < 0 || length == 0) continue;
        const int64_t offset = accepted ? integer(accepted, s) - 1 : 0;
        const bool warm = !initialized || integer(initialized, s) != 0;
        const int64_t base = (slot * n + channel) * depth;
        float prior[kHistoryMax], taps[kHistoryMax + 1];
        for (int64_t j = 0; j < history; ++j)
            prior[j] = warm ? ldt(state, base + offset + j) : 0.0f;
        for (int64_t j = 0; j < width; ++j)
            taps[j] = ldt(w, channel * wr + j * wc);
        for (int64_t token = 0; token < length; ++token) {
            float value = 0.0f;
            for (int64_t tap = 0; tap < width; ++tap) {
                const int64_t source = token - history + tap;
                const float input = source < 0 ? prior[source + history]
                    : ldt(x, (begin + source) * xr + channel * xc);
                value += input * taps[tap];
            }
            stt(y, (begin + token) * yr + channel * yc, value);
        }
        if (!history) continue;
        if (accepted) {
            /* A verify stores one shifted history followed by all candidates. The next step
             * picks the accepted prefix by reading at num_accepted - 1. */
            for (int64_t j = 0; j + 1 < history; ++j) stt(state, base + j, prior[j + 1]);
            for (int64_t token = 0; token < length; ++token)
                stt(state, base + history - 1 + token,
                    ldt(x, (begin + token) * xr + channel * xc));
        } else {
            /* A prefill commits every input, including a chunk shorter than its history. */
            for (int64_t j = 0; j < history; ++j) {
                const int64_t source = length - history + j;
                stt(state, base + j, source < 0 ? prior[source + history]
                    : ldt(x, (begin + source) * xr + channel * xc));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ple_conv */
/* PLE's dilated depthwise convolution, silu AFTER it, and the residual add folded in.
 *
 * TWO RULES THAT ARE NOT DERIVABLE AND ARE BOTH SILENT WHEN WRONG:
 *
 *   A ZERO `has_init` MEANS THE SLOT HOLDS NO HISTORY -- the window in front of the first token is
 *   ZEROS, not whatever the slot last held. Without it a fresh sequence landing on a recycled slot
 *   convolves the previous sequence's tail into its first `(width-1)*dilation` tokens: fluent
 *   output with the wrong prefix.
 *
 *   `num_accepted` SHIFTS THE READ AND ITS PRESENCE PICKS THE WINDOW LEFT BEHIND -- libref's
 *   ref_ple.cpp states the contract in full. Present, the step is a decode step whose drafts a
 *   verify may reject: the history read is shifted down by one and every input of the step follows
 *   it, so a next step that kept k tokens reads at k - 1 and finds the history ending at the k-th.
 *   Absent, every token commits and the last (width-1)*dilation inputs land at offset zero. The
 *   write happens in a SECOND pass after every read is done, because a sequence longer than the
 *   window would otherwise overwrite history it is still reading. */
AVX_KERNEL(ple_conv) {
    if (a->n_t < 9) return RAD_E_INVAL;
    const RadTensor* x = rad_arg_in(a, 0);
    const RadTensor* w = rad_arg_in(a, 1);
    const RadTensor* cs = rad_arg_in(a, 2);
    const RadTensor* cu = rad_arg_in(a, 3);
    const RadTensor* resid = rad_arg_in(a, 4);
    const RadTensor* sidx = rad_arg_in(a, 5);
    const RadTensor* hini = rad_arg_in(a, 6);
    const RadTensor* nacc = rad_arg_in(a, 7);
    const RadTensor* y = rad_arg_in(a, 8);
    if (!x || !w || !cs || !cu || !y) return RAD_E_INVAL;

    const int64_t n = p_int(a, "n", 0);
    /* `x` is [M, n] and it is the call; `M` is the band. */
    const int64_t M = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    const int64_t wid = p_int(a, "width", 0);
    const int64_t dil = p_int(a, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_SHAPE;
    const int64_t hist_len = (wid - 1) * dil;
    /* The read window lives in a stack frame. Nine at this model's shape; the bound is stated so a
     * wider one is refused by name rather than overflowing. Same number as libref's. */
    enum { kHistMax = 64 };
    if (hist_len > kHistMax) return RAD_E_UNSUPPORTED;

    const int64_t N = numel(cu) - 1;
    if (N <= 0) return RAD_E_SHAPE;
    if (numel(x) < M * n || numel(y) < M * n || numel(w) < n * wid) return RAD_E_SHAPE;
    if (resid && numel(resid) < M * n) return RAD_E_SHAPE;

    const int64_t state_len = cs->rank >= 3 ? cs->shape[cs->rank - 1] : hist_len;
    if (cs->data && state_len < hist_len) return RAD_E_SHAPE;

    /* Checked for every sequence BEFORE any of them is written: a state too shallow for the offset,
     * or for the window a decode step leaves behind, is a caller error, and an OpenMP body cannot
     * return. */
    for (int64_t s = 0; s < N; ++s) {
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        if (off < 0) return RAD_E_INVAL;
        if (off + hist_len > state_len) return RAD_E_SHAPE;
        const int64_t slen = ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s));
        if (nacc && cs->data && hist_len > 0 && slen > 0 && hist_len - 1 + slen > state_len)
            return RAD_E_SHAPE;
    }

    const int64_t xrow = rowoff(x, 1, n), xcol = laststride(x);
    const int64_t yrow = rowoff(y, 1, n), ycol = laststride(y);
    const int64_t rrow = resid ? rowoff(resid, 1, n) : 0;
    const int64_t rcol = resid ? laststride(resid) : 0;
    const int64_t wrow = rowoff(w, 1, wid), wcol = laststride(w);
    /* [n_slots, n, state_len], tight -- the layout gdn's conv state already uses. */
    const int64_t cs_slot = n * state_len, cs_ch = state_len;

    AVX_PARALLEL_FOR_IF(N * n >= 64)
    for (int64_t sc = 0; sc < N * n; ++sc) {
        const int64_t s = sc / n, c = sc % n;
        const int64_t bos = ldi(cu, offlin(cu, s));
        const int64_t slen = ldi(cu, offlin(cu, s + 1)) - bos;
        if (slen <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
        const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;

        float hist[kHistMax];
        for (int64_t i = 0; i < hist_len; ++i)
            hist[i] = (warm && cs->data) ? ldt(cs, slot * cs_slot + c * cs_ch + off + i) : 0.0f;

        /* The taps, read once per channel rather than per output element -- libref reads them
         * through `ldt` inside the t loop, which is `slen * width` dtype switches for `width`
         * distinct values. In scratch rather than a stack array so `width` carries no bound of its
         * own: the only limit this op has is the read window's, stated above. */
        float* tap = scratch_f32((size_t)wid);
        for (int64_t j = 0; j < wid; ++j) tap[j] = ldt(w, c * wrow + j * wcol);

        for (int64_t t = 0; t < slen; ++t) {
            float acc = 0.0f;
            for (int64_t j = 0; j < wid; ++j) {
                const int64_t src = t - hist_len + j * dil;
                const float vv = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                          : hist[src + hist_len];
                acc += vv * tap[j];
            }
            const float r = resid ? ldt(resid, (bos + t) * rrow + c * rcol) : 0.0f;
            stt(y, (bos + t) * yrow + c * ycol, r + acc / (1.0f + std::exp(-acc)));
        }
    }

    if (cs->data && hist_len > 0) {
        AVX_PARALLEL_FOR_IF(N * n >= 64)
        for (int64_t sc = 0; sc < N * n; ++sc) {
            const int64_t s = sc / n, c = sc % n;
            const int64_t bos = ldi(cu, offlin(cu, s));
            const int64_t slen = ldi(cu, offlin(cu, s + 1)) - bos;
            if (slen <= 0) continue;
            const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
            if (slot < 0) continue;
            const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
            const int64_t off = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
            const int64_t base = slot * cs_slot + c * cs_ch;

            float win[kHistMax];
            for (int64_t i = 0; i < hist_len; ++i) win[i] = warm ? ldt(cs, base + off + i) : 0.0f;

            if (nacc) {
                for (int64_t i = 0; i + 1 < hist_len; ++i) stt(cs, base + i, win[i + 1]);
                for (int64_t t = 0; t < slen; ++t)
                    stt(cs, base + hist_len - 1 + t, ldt(x, (bos + t) * xrow + c * xcol));
            } else {
                float keep[kHistMax];
                for (int64_t i = 0; i < hist_len; ++i) {
                    const int64_t src = slen - hist_len + i;
                    keep[i] = src >= 0 ? ldt(x, (bos + src) * xrow + c * xcol)
                                       : win[src + hist_len];
                }
                for (int64_t i = 0; i < hist_len; ++i) stt(cs, base + i, keep[i]);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ ngram_ids
 *
 * ids[m][h] = (hash of the n-gram ending at m) % vocab[h] + offset[h].
 *
 * THE ARITHMETIC IS TORCH'S, NOT C's, and both differences bite:
 *
 *   A SIGNED OVERFLOW IS UB IN C++ AND WRAPPING IN TORCH, so the multiply goes through uint64 and
 *   comes back -- two's complement, which is what an int64 tensor does on every machine this runs
 *   on.
 *
 *   `torch.remainder` FLOORS WHERE C TRUNCATES, so the modulo is written out rather than left to
 *   `%`.
 *
 * On a real checkpoint neither can fire -- the multiplier build bounds every multiplier by
 * (2^63-1)//vocab_size, so a real token's product has its sign bit clear and truncating and
 * flooring agree on a non-negative dividend. They are here so that a DRAWN case, whose indices no
 * vocabulary bounds, gets torch's answer rather than a platform's.
 *
 * THE EOS RULE IS COLLAPSED: once a shift falls off the front of the sequence, every longer shift
 * is EOS too, so the cut is carried rather than re-tested. A COLD slot's window is EOS and not
 * zero, because zero is a real token. */
static inline int64_t wrap_mul(int64_t x, int64_t y) {
    return (int64_t)((uint64_t)x * (uint64_t)y);
}
static inline int64_t floor_mod(int64_t x, int64_t y) {   /* y > 0 */
    const int64_t r = x % y;
    return r < 0 ? r + y : r;
}

AVX_KERNEL(ngram_ids) {
    if (a->n_t < 10) return RAD_E_INVAL;
    const RadTensor* tok = rad_arg_in(a, 0);
    const RadTensor* st = rad_arg_in(a, 1);
    const RadTensor* cu = rad_arg_in(a, 2);
    const RadTensor* mult = rad_arg_in(a, 3);
    const RadTensor* vsz = rad_arg_in(a, 4);
    const RadTensor* off_ = rad_arg_in(a, 5);
    const RadTensor* sidx = rad_arg_in(a, 6);
    const RadTensor* hini = rad_arg_in(a, 7);
    const RadTensor* nacc = rad_arg_in(a, 8);
    const RadTensor* ids = rad_arg_in(a, 9);
    /* THE PROMPT THAT FOLLOWS, optional: `tok` holds its row count more tokens after the step's,
     * continuing the LAST sequence, and their ids go here. The window is the step's own. */
    const RadTensor* ahead = a->n_t > 10 ? rad_arg_in(a, 10) : nullptr;
    if (!tok || !st || !cu || !mult || !vsz || !off_ || !ids) return RAD_E_INVAL;

    const int64_t heads = p_int(a, "heads", 0);
    if (heads <= 0) return RAD_E_SHAPE;
    /* The token vector is the call and the prompt that follows it; `M` is the band. */
    const int64_t A = ahead ? numel(ahead) / heads : 0;
    const int64_t M = band_rows(numel(tok) - A, p_int(a, "M", 0));
    const int64_t ngram = p_int(a, "ngram", 0);
    const int64_t eos = p_int(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return RAD_E_SHAPE;
    /* Heads are `ngram - 1` blocks of `heads_per_ngram`, block b using the (b+2)-gram. */
    if (heads % (ngram - 1)) return RAD_E_SHAPE;
    const int64_t per = heads / (ngram - 1);
    const int64_t ctx = ngram - 1;
    if (ngram > 8) return RAD_E_UNSUPPORTED;      /* the shift window lives in a stack frame */

    if (numel(mult) < ngram || numel(vsz) < heads || numel(off_) < heads) return RAD_E_SHAPE;
    if (numel(tok) < M + A || numel(ids) < M * heads) return RAD_E_SHAPE;

    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t W = st->rank >= 2 ? st->shape[st->rank - 1] : ctx;
    if (W < ctx) return RAD_E_SHAPE;
    /* The read window, and the window a decode step leaves behind: `ctx - 1` ids of history plus
     * every id of the step. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;
        if (o < 0) return RAD_E_INVAL;
        if (o + ctx > W) return RAD_E_SHAPE;
        const int64_t T = ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s));
        if (nacc && T > 0 && ctx - 1 + T > W) return RAD_E_SHAPE;
    }

    const int64_t ids_ld = rowoff(ids, 1, heads);
    const int64_t ahd_ld = ahead ? rowoff(ahead, 1, heads) : 0;

    /* The per-head tables, read once. libref reads `vocab_sizes` and `offsets` through ld_int
     * inside the innermost loop, which is `M * heads` dtype switches for `heads` distinct values. */
    int64_t* tab = (int64_t*)scratch_raw((size_t)(2 * heads + ngram) * sizeof(int64_t) + 64);
    int64_t* vtab = tab;
    int64_t* otab = tab + heads;
    int64_t* mtab = tab + 2 * heads;
    for (int64_t h = 0; h < heads; ++h) {
        vtab[h] = ldi(vsz, h);
        otab[h] = ldi(off_, h);
        if (vtab[h] <= 0 || otab[h] < 0) return RAD_E_SHAPE;
    }
    for (int64_t k = 0; k < ngram; ++k) mtab[k] = ldi(mult, k);

    /* Serial over sequences, as libref is: each writes its own state slot, the work per sequence is
     * a token loop with a handful of integer ops, and a parallel region over a handful of
     * sequences would cost more than it saves. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t bos = ldi(cu, offlin(cu, s));
        const int64_t eot = ldi(cu, offlin(cu, s + 1));
        if (bos < 0 || eot < bos || eot > M) return RAD_E_SHAPE;
        const int64_t T = eot - bos;
        if (T <= 0) continue;
        const int64_t slot = sidx ? ldi(sidx, offlin(sidx, s)) : s;
        if (slot < 0) continue;
        const bool warm = !hini || ldi(hini, offlin(hini, s)) != 0;
        const int64_t o = nacc ? ldi(nacc, offlin(nacc, s)) - 1 : 0;

        int64_t prev[8];
        for (int64_t i = 0; i < ctx; ++i)
            prev[i] = warm ? ldi(st, rowoff(st, slot, W) + o + i) : eos;

        auto hist = [&](int64_t p) -> int64_t {
            return p < ctx ? prev[p] : ldi(tok, offlin(tok, bos + p - ctx));
        };

        const int64_t reach = T + (ahead && s + 1 == n_seq ? A : 0);
        for (int64_t t = 0; t < reach; ++t) {
            const int64_t p = ctx + t;
            int64_t sh[8];
            sh[0] = hist(p);
            bool cut = false;
            for (int64_t k = 1; k < ngram; ++k) {
                if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
                sh[k] = hist(p - k);
            }

            /* One mix per n-gram order, EXTENDED IN PLACE: the (b+2)-gram's mix is the
             * (b+1)-gram's XOR'd with one more term, which saves recomputing the shared prefix. */
            int64_t mixed = 0;
            int64_t blk = -1;
            for (int64_t k = 0; k < ngram; ++k) {
                const int64_t term = wrap_mul(sh[k], mtab[k]);
                mixed = k == 0 ? term : (mixed ^ term);
                if (k == 0) continue;           /* a 1-gram addresses no head */
                ++blk;                          /* k = 1 is the 2-gram block, k = 2 the 3-gram */
                for (int64_t j = 0; j < per; ++j) {
                    const int64_t h = blk * per + j;
                    if (t < T)
                        st_int_dt(ids->dtype, ids->data, (bos + t) * ids_ld + h,
                                  floor_mod(mixed, vtab[h]) + otab[h]);
                    else
                        st_int_dt(ahead->dtype, ahead->data, (t - T) * ahd_ld + h,
                                  floor_mod(mixed, vtab[h]) + otab[h]);
                }
            }
        }

        /* The window this step leaves behind, written after every read of it is done, in the
         * layout ple_conv's state follows: shifted by one with every id of the step behind it when
         * `num_accepted` is present, the last `ctx` ids at offset zero when it is not. */
        const int64_t base = rowoff(st, slot, W);
        if (nacc) {
            for (int64_t i = 0; i + 1 < ctx; ++i)
                st_int_dt(st->dtype, st->data, base + i, prev[i + 1]);
            for (int64_t t = 0; t < T; ++t)
                st_int_dt(st->dtype, st->data, base + ctx - 1 + t, ldi(tok, offlin(tok, bos + t)));
        } else {
            for (int64_t i = 0; i < ctx; ++i) {
                const int64_t src = T - ctx + i;
                const int64_t vv = src >= 0 ? ldi(tok, offlin(tok, bos + src)) : prev[src + ctx];
                st_int_dt(st->dtype, st->data, base + i, vv);
            }
        }
    }
    return RAD_OK;
}

/* ================================================================== the collectives
 *
 * WORLD SIZE 1 ONLY, AND THAT IS A DELIBERATE LIMIT RATHER THAN AN OMISSION.
 *
 * libref implements the real thing, and it can because of how this engine is shaped: one process,
 * one thread per rank, one address space, so a peer's buffer is a POINTER and the whole collective
 * is a rendezvous, a read of every rank's pointer, and a second rendezvous before anyone
 * overwrites theirs. That makes the host backend a fake multi-device backend, which is genuinely
 * useful -- the sharded declare, the collective placement and the distributed top-k can all be
 * exercised on a box with no second card.
 *
 * It is also a piece of SHARED MUTABLE STATE with a barrier, a generation counter and a timeout,
 * and there is exactly one of it per process. Two kernel plugins each carrying their own would be
 * two rendezvous groups for one world, and which one a rank joined would depend on which plugin
 * won selection for that op -- a deadlock that appears only when the hierarchy changes. A CPU
 * kernel library has no business owning the wire; libref already owns it and sits below this
 * plugin in the hierarchy, so a tensor-parallel host deployment resolves the collectives there and
 * everything else here.
 *
 * So: at world 1 both are the degenerate case and are served exactly. Above 1 this refuses by name
 * with RAD_E_UNSUPPORTED, which makes the selector fall through to libref's row rather than
 * producing a wrong sum.
 *
 * `exact` IS HONOURED IN ONE DIRECTION ONLY, as libref does: exact=1 gets the exact sum, and
 * exact=0 -- a caller that has declared it will accept a lossy quantised wire -- ALSO gets the
 * exact sum, because there is no wire here to quantise. Serving a lossy request exactly is safe;
 * the constraint exists so a caller who wanted exactness cannot be handed a lossy result by
 * accident, not the reverse. */
AVX_KERNEL(all_reduce) {
    if (!rad_args_have(a, 1)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* y = rad_arg_in(a, 1);
    /* The message is the operand; `numel` is the band. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    if (nel <= 0) return RAD_E_SHAPE;
    if (y && numel(y) < nel) return RAD_E_SHAPE;

    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (world > 1) return RAD_E_UNSUPPORTED;

    /* In place: the answer is already there. With a separate destination it still has to be COPIED
     * there, because a caller that passed one reads it and not x. */
    if (!y) return RAD_OK;
    if (rad_tensor_is_contiguous(x) && rad_tensor_is_contiguous(y)) {
        const int64_t TILE = 8192;
        for (int64_t off = 0; off < nel; off += TILE) {
            const int64_t m = (nel - off) < TILE ? (nel - off) : TILE;
            float* s = scratch_f32((size_t)TILE);
            out_row(y, off, m, in_row(x, off, m, s));
        }
        return RAD_OK;
    }
    for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    return RAD_OK;
}

AVX_KERNEL(all_gather) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    /* The contribution is the operand; `numel` is the band. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (nel <= 0 || numel(y) < nel * world) return RAD_E_SHAPE;
    if (world > 1) return RAD_E_UNSUPPORTED;

    /* THE `row` PLACEMENT DOES NOT MATTER AT WORLD 1 and is still checked, because a caller that
     * passes an illegal one at world 1 would get a silent pass here and a refusal at world 2 --
     * and the whole value of running a tensor-parallel graph at world 1 is that it is the same
     * graph. */
    const int64_t row = p_int(a, "row", 0);
    if (row < 0 || (row > 0 && nel % row)) return RAD_E_SHAPE;

    if (rad_tensor_is_contiguous(x) && rad_tensor_is_contiguous(y) && x->dtype == y->dtype) {
        std::memcpy(y->data, x->data, (size_t)((nel * rad_dtype_bits(x->dtype) + 7) / 8));
        return RAD_OK;
    }
    for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
    return RAD_OK;
}
