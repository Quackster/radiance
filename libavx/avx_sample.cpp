/* avx_sample.cpp -- the sampler ops.
 *
 * ============================== THESE ARE NOT WHERE THE TIME GOES, EXCEPT WHERE THEY ARE ==============================
 *
 * Eleven of these twelve ops work over a CANDIDATE SET of a thousand entries at most, and their
 * cost is control flow rather than arithmetic. Vectorising them would be work spent where no time
 * is; what they get here is the one thing that does matter -- the candidate values widened once
 * into f32 so the per-element dtype switch is gone -- and otherwise the reference's structure,
 * which is legible and provably right.
 *
 * The exceptions are the three that touch the WHOLE VOCABULARY, and at a 248K vocabulary that is a
 * megabyte a row:
 *
 *   sample_temp     a row multiply. Pure vector work, and the trivial one.
 *   sample_argmax   a maximum with the LOWEST index on a tie. Vectorised through the same
 *                   `topk_next_f32` the routers use, so the greedy path and a top-1 path cannot
 *                   disagree about a tie -- which they must not, because a greedy request is
 *                   staged at top_k = 1 when lm_head is vocab-sharded.
 *   sample_mask     a bitmask over the vocabulary. This one has an AVX-512 form that is not
 *                   available below it and is worth the `#if`: the mask word IS a `__mmask16`,
 *                   so applying 16 tokens of grammar is one masked store of -inf and no shifts at
 *                   all. Below AVX-512 it is a shift and a test per token.
 *
 * ============================== EVERY SCALAR IS PER ROW ==============================
 *
 * `rep`, `freq`, `pres`, `k`, `p`, the seed and the position are FIELDS OF A ROW (abi/rad_sample.h),
 * not parameters. A parameter is geometry -- frozen at declare, matched on by the selector -- so
 * two requests in one batch with different top_p would be two resolutions of one op and a
 * deployment would have exactly one top_p, which is not a sampler.
 *
 * ============================== THE CANDIDATE SET ==============================
 *
 * `sample_topk` produces a FIXED-WIDTH set and the narrowing stages work IN PLACE. The tensors do
 * not shrink, so a dropped candidate needs a spelling: index -1, value -infinity. Kept candidates
 * stay where they are, in descending order.
 *
 * `cand_val` carries LOGITS, not probabilities, and each narrowing stage takes its own softmax over
 * the set -- which is what llama.cpp does and what keeps the stages composable in any order.
 *
 * A GREEDY ROW (RAD_SP_GREEDY) HAS NO CANDIDATES: its token is sample_argmax's, top-k and the
 * merge write it an empty set, the narrowing stages leave it alone and the pick does not touch its
 * token -- libref's contract, for the reason given there.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include "rad_sample.h"

#include <cfloat>

using namespace avx;

/* The params operand, checked once. A sampler op with no rows to act on is not an error -- it is a
 * step in which nothing asked for that stage.
 *
 * The row array is a BYTE BLOB viewed through a 32-bit dtype: the sampler declares it
 * [rows, sizeof(RadSampleParams)/4] u32, which is the only way the builder can express "this many
 * bytes a row". The dtype is REQUIRED to be a 32-bit one rather than run through
 * rad_dtype_bytes -- "any width as long as the arithmetic works out" is the wrong contract, since
 * the blob is addressed as 32-bit words and a 16-bit view would pass a byte count and then misread
 * every row. */
struct Rows { const RadSampleParams* p; int64_t n; };

static inline int rows_of_params(const RadTensor* t, int64_t M, Rows* out) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return RAD_E_DTYPE;
    if (numel(t) * 4 < M * (int64_t)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    out->p = (const RadSampleParams*)t->data;
    out->n = M;
    return RAD_OK;
}

#define AVX_TRY(expr) do { const int _s = (expr); if (_s < 0) return _s; } while (0)

/* The history window both history stages read, libref's reading (ref_sample.cpp): `hist_off` and
 * `hist_len` address the row's own row of the history plane, clamped to it, and `last_n` keeps
 * the TAIL of that span. last_n 0 turns the stage off for the row, a negative one keeps the whole
 * span -- llama.cpp's meaning for repeat_last_n and dry_penalty_last_n. */
static inline bool hist_window(const RadSampleParams& sp, int32_t last_n, int64_t hmax,
                               int64_t* ho_out, int64_t* hl_out) {
    if (last_n == 0) return false;
    int64_t hl = sp.hist_len;
    int64_t ho = sp.hist_off < 0 ? 0 : sp.hist_off;
    if (ho > hmax) ho = hmax;
    if (hl > hmax - ho) hl = hmax - ho;
    if (last_n > 0 && hl > last_n) {
        ho += hl - last_n;
        hl = last_n;
    }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

/* ------------------------------------------------------------------ sample_penalties */
/* llama.cpp's ordering, in one pass over the row's history:
 *
 *     repetition   logit = logit > 0 ? logit / rep : logit * rep       (once per DISTINCT token)
 *     frequency    logit -= count * freq
 *     presence     logit -= pres                                       (once per distinct token)
 *
 * The repetition penalty being multiplicative and SIGN-DEPENDENT is the part everyone gets wrong:
 * dividing a negative logit would make it larger, so it is multiplied instead.
 *
 * The history is pulled into a small int array once. libref reads it back through ld_int inside
 * the O(hl^2) first-occurrence test, which is a dtype switch per comparison on a loop that runs
 * hl^2 times; the array makes the inner test an integer compare. The ORDER and the answer are
 * unchanged -- this is the same quadratic scan, over the same values.
 *
 * History ids are GLOBAL and the logits row is this rank's shard, so a token's column is its id
 * less `vocab_off`, and a token outside the shard is another rank's to penalise. */
AVX_KERNEL(sample_penalties) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_PENALTY)) continue;
        const float rep = sp.rep_penalty, freq = sp.freq_penalty, pres = sp.pres_penalty;
        if (rep == 1.0f && freq == 0.0f && pres == 0.0f) continue;

        int64_t ho = 0, hl = 0;
        if (!hist_window(sp, sp.penalty_last_n, hmax, &ho, &hl)) continue;

        const int64_t hb = rowoff(hist, m, hmax) + ho * laststride(hist);
        const int64_t hs = laststride(hist);
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        int32_t* h = (int32_t*)scratch_f32((size_t)hl);
        for (int64_t i = 0; i < hl; ++i) h[i] = (int32_t)ldi(hist, hb + i * hs);

        for (int64_t i = 0; i < hl; ++i) {
            const int32_t tok = h[i];
            const int64_t col = (int64_t)tok - voff;
            if (tok < 0 || col < 0 || col >= n_vocab) continue;
            /* First occurrence only: the penalties are per distinct token, and the count is what
             * carries the multiplicity. */
            bool first = true;
            int64_t count = 0;
            for (int64_t j = 0; j < hl; ++j) {
                if (h[j] != tok) continue;
                if (j < i) { first = false; break; }
                ++count;
            }
            if (!first) continue;
            float v = ldt(lg, lb + col * ls);
            v = v > 0.0f ? v / rep : v * rep;
            v -= (float)count * freq;
            v -= pres;
            stt(lg, lb + col * ls, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_dry */
/* DRY -- "don't repeat yourself". libref's definition (ref_sample.cpp), which is llama.cpp's:
 * each token that would EXTEND a suffix of the window that has occurred before is penalised ONCE,
 * by multiplier * base^(L - allowed_length) for the LONGEST repeat L it would extend, repeats
 * capped at the row's `dry_rep_limit`. A position the optional `breakers` plane marks holds a
 * whole sequence breaker, which is never penalised. The penalty is formed by repeated
 * multiplication stopping short of overflow, the way every implementation of this op forms it,
 * so they agree to the bit.
 *
 * The window is pulled into an int array once, as the penalties do, and the per-position repeat
 * lengths go into a second one; the longest per token is then found by scanning those lengths for
 * each qualifying position, which needs no map and keeps the quadratic scan's memory access
 * inside two small arrays. */
static inline float dry_penalty(float multiplier, float base, int64_t e) {
    float pen = multiplier;
    if (base == 1.0f) return pen;
    const float lim = FLT_MAX / base;
    for (int64_t i = 0; i < e; ++i) {
        if (pen > lim) break;
        pen *= base;
    }
    return pen;
}

AVX_KERNEL(sample_dry) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const RadTensor* brk = rad_arg_in(a, 3);
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    if (brk && numel(brk) < M * hmax) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_DRY) || !(sp.dry_multiplier > 0.0f)) continue;
        const int64_t allowed = sp.dry_allowed_length > 0 ? sp.dry_allowed_length : 2;

        int64_t ho = 0, hl = 0;
        if (!hist_window(sp, sp.dry_penalty_last_n, hmax, &ho, &hl)) continue;
        if (hl <= allowed) continue;
        const int64_t cap = sp.dry_rep_limit > 0 && sp.dry_rep_limit < hl ? sp.dry_rep_limit : hl;
        if (cap < allowed) continue;

        const int64_t hb = rowoff(hist, m, hmax) + ho * laststride(hist);
        const int64_t hs = laststride(hist);
        const int64_t bb = brk ? rowoff(brk, m, hmax) + ho * laststride(brk) : 0;
        const int64_t bs = brk ? laststride(brk) : 0;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        int32_t* h   = (int32_t*)scratch_f32((size_t)hl);
        int32_t* len = (int32_t*)scratch_f32_b((size_t)hl);
        for (int64_t i = 0; i < hl; ++i) h[i] = (int32_t)ldi(hist, hb + i * hs);

        /* len[j]: how long a suffix of the window also ends at position j, capped; 0 where the
         * continuation after j is exempt or outside this rank's shard, since such a j is never
         * penalised and never the longest for a token that is. */
        for (int64_t j = 0; j + 1 < hl; ++j) {
            const int64_t lim = j + 1 < cap ? j + 1 : cap;
            int64_t n = 0;
            while (n < lim && h[j - n] == h[hl - 1 - n]) ++n;
            const int64_t col = (int64_t)h[j + 1] - voff;
            const bool skip = (brk && ldi(brk, bb + (j + 1) * bs) != 0) || h[j + 1] < 0 ||
                              col < 0 || col >= n_vocab;
            len[j] = skip ? 0 : (int32_t)n;
        }

        const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;
        for (int64_t j = 0; j + 1 < hl; ++j) {
            if (len[j] < allowed) continue;
            /* The representative of this continuation token: its longest repeat, the earliest
             * position on a tie. Every other position of the same token defers to it, so the
             * token is penalised once. */
            const int32_t next = h[j + 1];
            bool rep = true;
            for (int64_t j2 = 0; j2 + 1 < hl && rep; ++j2) {
                if (j2 == j || h[j2 + 1] != next) continue;
                if (len[j2] > len[j] || (len[j2] == len[j] && j2 < j)) rep = false;
            }
            if (!rep) continue;
            const int64_t off = lb + ((int64_t)next - voff) * ls;
            stt(lg, off, ldt(lg, off) - dry_penalty(sp.dry_multiplier, base, len[j] - allowed));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_temp */
/* temp <= 0 leaves the row untouched: greedy decoding is `sample_argmax`, and a temperature of
 * zero silently meaning something else is how a sampler chain becomes unreadable. */
AVX_KERNEL(sample_temp) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const float t = R.p[m].temp;
        if (!(t > 0.0f) || t == 1.0f) continue;
        const float inv = 1.0f / t;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        if (ls != 1) {
            for (int64_t i = 0; i < n_vocab; ++i) stt(lg, lb + i * ls, ldt(lg, lb + i * ls) * inv);
            continue;
        }
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = in_row(lg, lb, n_vocab, buf);
        float* out = (v == buf) ? buf : scratch_f32_b((size_t)n_vocab);
        const vf vi = vf_set1(inv);
        int64_t i = 0;
        for (; i + VF_N <= n_vocab; i += VF_N) vf_storeu(out + i, vf_mul(vf_loadu(v + i), vi));
        for (; i < n_vocab; ++i) out[i] = v[i] * inv;
        out_row(lg, lb, n_vocab, out);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_topk */
AVX_KERNEL(sample_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    /* The candidate WIDTH is the buffer's, not the request's: it is the deployment bound the arena
     * was sized for, and every later stage reads it. */
    const int64_t n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : numel(ci) / M;
    if (n_cand <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;

    /* THE VOCAB-PARALLEL PAIR PLANE, optional. cand_idx stays LOCAL -- it indexes this rank's
     * logits row and every stage after top-k reads it that way -- and `pairs` carries the same
     * candidates with `vocab_off` added, as (u32 GLOBAL id, f32 logit), which is what the
     * all-gather moves and what sample_merge_topk reads. */
    const RadTensor* pp = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (pp && numel(pp) < M * 2 * n_cand) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        /* A row's own top_k, clamped to the buffer. 0 or negative means "the whole candidate
         * width", which is what a request that set no top_k asks for. A greedy row takes none,
         * and its row of logits is not even read. */
        int64_t k = R.p[m].top_k;
        if (k <= 0 || k > n_cand) k = n_cand;
        if (R.p[m].flags & RAD_SP_GREEDY) k = 0;

        const int64_t lb = rowoff(lg, m, n_vocab);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int64_t pb = pp ? rowoff(pp, m, 2 * n_cand) : 0, ps = pp ? laststride(pp) : 0;
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = k > 0 ? in_row(lg, lb, n_vocab, buf) : buf;
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            const bool ok = r < k && topk_next_f32(v, n_vocab, &pv, &pi);
            st_int_dt(ci->dtype, ci->data, ib + r * is, ok ? pi : -1);
            stt(cv, vb + r * vs, ok ? pv : -INFINITY);
            if (!pp) continue;
            /* 0xffffffff is the hole, which the merge tests as a negative signed id -- the same
             * convention cand_idx carries as -1. */
            uint32_t bits;
            const float val = ok ? pv : -INFINITY;
            std::memcpy(&bits, &val, 4);
            st_int_dt(pp->dtype, pp->data, pb + (2 * r) * ps,
                      ok ? (int64_t)(pi + voff) : (int64_t)0xffffffffu);
            st_int_dt(pp->dtype, pp->data, pb + (2 * r + 1) * ps, (int64_t)bits);
        }
    }
    return RAD_OK;
}

/* ================================================================== the narrowing stages
 *
 * All four widen the candidate values into f32 ONCE and then work on that array, which is the only
 * change from libref's structure: it reads `cand_val` back through `ldt` inside loops that are
 * O(n_cand) (top-p, min-p, XTC) and O(n_cand^2) (typical), so the dtype switch runs up to a
 * million times on a thousand-wide set for one row.
 *
 * The typical stage additionally precomputes the probability and the surprisal distance per
 * candidate, because libref recomputes `exp` and `log` inside its threshold search -- three libm
 * calls per (candidate, candidate) pair. The search itself stays O(n_cand^2) and stays exactly the
 * scan libref performs, because the answer is a threshold chosen among the distances that actually
 * occur and any faster search would have to sort, which changes which of two equal distances is
 * found first. */
enum Narrow { NR_TOPP, NR_MINP, NR_TYPICAL, NR_XTC };

static int narrow(const RadArgs* a, Narrow which) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (sp.flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float* val = scratch_f32((size_t)n_cand);
        float* prob = scratch_f32_b((size_t)n_cand);
        unsigned char* alive = (unsigned char*)scratch_f32_c((size_t)n_cand / 4 + 4);
        for (int64_t i = 0; i < n_cand; ++i) {
            alive[i] = ldi(ci, ib + i * is) >= 0 ? 1 : 0;
            val[i] = alive[i] ? ldt(cv, vb + i * vs) : -INFINITY;
        }

        auto drop = [&](int64_t i) {
            st_int_dt(ci->dtype, ci->data, ib + i * is, -1);
            stt(cv, vb + i * vs, -INFINITY);
        };

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) if (alive[i] && val[i] > mx) mx = val[i];
        if (mx == -INFINITY) continue;

        if (which == NR_MINP) {
            const float p = sp.min_p;
            if (!(p > 0.0f)) continue;
            /* Keep everything within a factor p of the most likely candidate. Comparing
             * exp(v - max) against p is the same test as prob >= p * max_prob without forming the
             * normaliser -- one fewer pass and one fewer rounding. */
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (val[i] == mx) continue;                  /* the max always survives */
                if (std::exp(val[i] - mx) < p) drop(i);
            }
            continue;
        }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) if (alive[i]) sum += std::exp(val[i] - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int64_t i = 0; i < n_cand; ++i)
            prob[i] = alive[i] ? std::exp(val[i] - mx) * inv : 0.0f;

        if (which == NR_TOPP) {
            const float p = sp.top_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* The candidates arrive in DESCENDING order from sample_topk and this op does not
             * re-sort them. Keep the shortest prefix whose cumulative probability reaches p, and
             * always keep at least one -- a p small enough to select nothing still has to produce
             * a token. */
            float cum = 0.0f;
            bool done = false;
            int64_t kept = 0;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (done) { drop(i); continue; }
                cum += prob[i];
                ++kept;
                if (cum >= p && kept >= 1) done = true;
            }
            continue;
        }

        if (which == NR_TYPICAL) {
            const float p = sp.typical_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Locally typical sampling: keep the candidates whose surprisal -log(prob) is closest
             * to the distribution's entropy, until their mass reaches p. The candidate order is by
             * probability and this order is by |surprisal - H|, so the two disagree and the
             * selection is a threshold on the distance rather than a prefix. Ties are kept, which
             * can overshoot p slightly -- the alternative is an order that depends on the
             * candidate index. */
            float H = 0.0f;
            for (int64_t i = 0; i < n_cand; ++i)
                if (alive[i] && prob[i] > 0.0f) H -= prob[i] * std::log(prob[i]);
            float* dist = val;    /* `val` is dead from here: the distances reuse its storage. */
            for (int64_t i = 0; i < n_cand; ++i)
                dist[i] = (alive[i] && prob[i] > 0.0f) ? std::fabs(-std::log(prob[i]) - H) : 0.0f;

            float cut = 0.0f;
            bool have_cut = false;
            for (int64_t c = 0; c < n_cand; ++c) {
                if (!alive[c] || !(prob[c] > 0.0f)) continue;
                const float dc = dist[c];
                float mass = 0.0f;
                for (int64_t i = 0; i < n_cand; ++i) {
                    if (!alive[i] || !(prob[i] > 0.0f)) continue;
                    if (dist[i] <= dc) mass += prob[i];
                }
                if (mass >= p && (!have_cut || dc < cut)) { cut = dc; have_cut = true; }
            }
            if (!have_cut) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (!(prob[i] > 0.0f) || dist[i] > cut) drop(i);
            }
            continue;
        }

        /* XTC -- "exclude top choices". With probability `xtc_probability`, DROP every candidate
         * above `xtc_threshold` except the LEAST likely of them, which is the opposite of every
         * other stage here and is the point: it removes the obvious continuations and leaves the
         * interesting one. The draw is per row and comes from the row's own stream, so it
         * reproduces. */
        {
            if (!(sp.xtc_probability > 0.0f) || !(sp.xtc_threshold > 0.0f)) continue;
            if (rad_sample_uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= sp.xtc_probability)
                continue;
            int64_t last_above = -1, n_above = 0;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i]) continue;
                if (prob[i] >= sp.xtc_threshold) { last_above = i; ++n_above; }
            }
            /* Nothing to do unless at least two candidates clear the threshold: dropping the only
             * one would leave the row without its own choice. */
            if (n_above < 2) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!alive[i] || i == last_above) continue;
                if (prob[i] >= sp.xtc_threshold) drop(i);
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(sample_topp)    { return narrow(a, NR_TOPP); }
AVX_KERNEL(sample_minp)    { return narrow(a, NR_MINP); }
AVX_KERNEL(sample_typical) { return narrow(a, NR_TYPICAL); }
AVX_KERNEL(sample_xtc)     { return narrow(a, NR_XTC); }

/* ------------------------------------------------------------------ sample_mask */
/* Where a GBNF grammar lands. The state machine stays on host and what crosses is a token bitmask
 * per sequence -- 19 KiB at a 151K vocab, cheap enough to upload every step.
 *
 * A SET BIT MEANS ALLOWED. That polarity has to be stated: a grammar produces a set of admissible
 * tokens, and the mask is that set rather than its complement. Word w bit b covers token 32*w + b,
 * least significant bit first.
 *
 * THE AVX-512 FORM IS DIFFERENT IN KIND, not just in width, and that is why it gets an `#if`. A
 * `__mmask16` IS sixteen bits of the grammar word, so applying sixteen tokens is one masked store
 * of -inf with the mask inverted -- no shift, no test, no branch per token. Below AVX-512 the mask
 * has to be materialised as a vector of 0/-1 and blended, which costs more than the shift-and-test
 * it replaces at these widths, so those levels keep the scalar loop and say so. */
AVX_KERNEL(sample_mask) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *bm = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    const int64_t words = (n_vocab + 31) / 32;
    const int64_t mask_rows = numel(bm) / (words > 0 ? words : 1);
    if (mask_rows <= 0) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_MASK)) continue;
        const int64_t row = sp.mask_row;
        if (row < 0 || row >= mask_rows) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t bb = rowoff(bm, row, words), bs = laststride(bm);

#if AVX_LEVEL == 3
        if (ls == 1 && lg->dtype == RAD_F32 && bs == 1 &&
            (bm->dtype == RAD_U32 || bm->dtype == RAD_I32)) {
            float* p = (float*)lg->data + lb;
            const uint32_t* w = (const uint32_t*)bm->data + bb;
            const __m512 neg = _mm512_set1_ps(-INFINITY);
            int64_t i = 0;
            for (; i + 32 <= n_vocab; i += 32) {
                const uint32_t bits = w[i / 32];
                /* The mask is the COMPLEMENT: a set grammar bit is allowed and must be left
                 * alone, so -inf is stored where the bit is clear. */
                _mm512_mask_storeu_ps(p + i, (__mmask16)(~(uint16_t)(bits & 0xffffu)), neg);
                _mm512_mask_storeu_ps(p + i + 16, (__mmask16)(~(uint16_t)(bits >> 16)), neg);
            }
            for (; i < n_vocab; ++i)
                if (!((w[i / 32] >> (i % 32)) & 1u)) p[i] = -INFINITY;
            continue;
        }
#endif
        for (int64_t i = 0; i < n_vocab; ++i) {
            const uint32_t word = (uint32_t)ldi(bm, bb + (i / 32) * bs);
            if (!((word >> (i % 32)) & 1u)) stt(lg, lb + i * ls, -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_pick */
/* The inverse CDF, walked in candidate order, accumulating in f32. The fallback is the LAST
 * CANDIDATE ABOVE -INFINITY: `u` is strictly below 1 and the probabilities sum to 1 in exact
 * arithmetic, but they do not in f32, and a draw that falls through the last comparison must still
 * produce a token the distribution allows -- not one of the grammar-masked candidates that top-k
 * leaves at -infinity at the tail of the set. A greedy row is skipped: its token is the argmax's.
 *
 * THE DRAW IS SPECIFIED rather than left to a library (abi/rad_sample.h): splitmix64 over
 * (seed * golden ratio + pos), 24 bits as a float in [0, 1). The same stream has to come out of a
 * host reference, a device kernel and a rerun. */
AVX_KERNEL(sample_pick) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2], *tok = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand || numel(tok) < M) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        if (R.p[m].flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ldi(ci, ib + i * is) < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) { st_int_dt(tok->dtype, tok->data, offlin(tok, m), -1); continue; }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ldi(ci, ib + i * is) < 0) continue;
            sum += std::exp(ldt(cv, vb + i * vs) - mx);
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const float u = rad_sample_uniform01(R.p[m].seed ^ RAD_SAMPLE_KEY_PICK, R.p[m].pos);

        float cum = 0.0f;
        int64_t chosen = -1, last = -1;
        for (int64_t i = 0; i < n_cand; ++i) {
            const int64_t id = ldi(ci, ib + i * is);
            if (id < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > -INFINITY) last = id;
            cum += std::exp(v - mx) * inv;
            if (u < cum) { chosen = id; break; }
        }
        st_int_dt(tok->dtype, tok->data, offlin(tok, m), chosen >= 0 ? chosen : last);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_argmax */
/* Greedy, ties to the LOWER token id -- the same tie rule the top-k selection uses, so a greedy
 * step and a top-1 step cannot disagree. That is not decoration: when lm_head is vocab-sharded,
 * `sample_argmax` is unusable (its answer is a LOCAL column) and a greedy request is staged at
 * top_k = 1 through the chain instead, so the two paths must select the same token.
 *
 * Run through the same `topk_next_f32` the routers use, for exactly that reason. */
AVX_KERNEL(sample_argmax) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *tok = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0 || numel(tok) < M) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_vocab >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)n_vocab);
        const float* v = in_row(lg, rowoff(lg, m, n_vocab), n_vocab, buf);
        float pv = INFINITY;
        int64_t pi = -1;
        const bool ok = topk_next_f32(v, n_vocab, &pv, &pi);
        st_int_dt(tok->dtype, tok->data, offlin(tok, m), ok ? pi : -1);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_merge_topk */
/* THE VOCAB-PARALLEL MERGE. lm_head is vocab-sharded, so each rank runs top-k over ITS OWN shard
 * and the candidate sets are all-gathered -- k * world_size pairs, 2 KiB at k=64 and world_size=4,
 * against the ~77 MiB a full-logits gather would move.
 *
 * `gathered` is [M, world_size * n_cand] pairs of (token id, logit) as raw u32 words. The token id
 * is already GLOBAL, because each rank adds its own vocab offset before the gather -- a merge that
 * had to know the offsets would need the shard layout, which is the caller's business. THE LOGIT
 * IS READ BY BIT PATTERN: the buffer's dtype can only be one of the two halves' types, it is u32,
 * and a float read through it would turn 0x40800000 into 1.08e9 instead of 4.0.
 *
 * THE ROW'S OWN top_k IS RE-APPLIED, and that is why this takes `params`. Every rank ran top-k over
 * its own shard before the gather, so the merged set holds up to world_size * k candidates where
 * the request asked for k, and no stage after this one narrows again. At k = 1 that is not a
 * slightly wider support, it is a different ANSWER: a greedy row is staged at top_k = 1 exactly so
 * this chain returns the argmax, and with world_size candidates alive `sample_pick` draws among
 * them against the request's seed. */
AVX_KERNEL(sample_merge_topk) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *gt = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    AVX_TRY(rows_of_params(pr, M, &R));
    if (gt->dtype != RAD_U32 && gt->dtype != RAD_I32) return RAD_E_DTYPE;
    const int64_t per_row = numel(gt) / M;
    const int64_t n_in = per_row / 2;
    if (n_in <= 0) return RAD_E_SHAPE;
    auto bits_to_f32 = [](int64_t w) {
        const uint32_t u = (uint32_t)w;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    };

    AVX_PARALLEL_FOR_IF(M >= 2)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t gb = rowoff(gt, m, per_row), gs = laststride(gt);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int32_t tk = R.p[m].top_k;
        const int64_t keep = (R.p[m].flags & RAD_SP_GREEDY) ? 0
                           : (tk > 0 && (int64_t)tk < n_cand) ? (int64_t)tk : n_cand;

        /* The pairs are unpacked ONCE into two arrays. libref reads both halves back through
         * ld_int inside the O(n_cand * n_in) selection, which is two dtype switches per comparison.
         * The selection itself stays a selection sort over n_in entries, ties to the lower token
         * id -- the tie rule every other selection in this vocabulary uses. */
        float* gv = scratch_f32((size_t)n_in);
        int32_t* gi = (int32_t*)scratch_f32_b((size_t)n_in);
        for (int64_t j = 0; j < n_in; ++j) {
            gi[j] = (int32_t)(uint32_t)ldi(gt, gb + 2 * j * gs);
            gv[j] = bits_to_f32(ldi(gt, gb + (2 * j + 1) * gs));
        }

        float taken_v = INFINITY;
        int64_t taken_i = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            if (r >= keep) {
                st_int_dt(ci->dtype, ci->data, ib + r * is, -1);
                stt(cv, vb + r * vs, -INFINITY);
                continue;
            }
            float best = -INFINITY;
            int64_t best_id = -1;
            for (int64_t j = 0; j < n_in; ++j) {
                const int64_t id = gi[j];
                const float v = gv[j];
                if (id < 0) continue;
                const bool after = v < taken_v || (v == taken_v && id > taken_i);
                if (!after) continue;
                if (best_id < 0 || v > best || (v == best && id < best_id)) {
                    best = v;
                    best_id = id;
                }
            }
            st_int_dt(ci->dtype, ci->data, ib + r * is, best_id);
            stt(cv, vb + r * vs, best_id >= 0 ? best : -INFINITY);
            if (best_id < 0) continue;
            taken_v = best;
            taken_i = best_id;
        }
    }
    return RAD_OK;
}

/* ================================================================== dflash_select
 *
 * Not a sampler, and here because it is the other op that picks TOKENS: DFlash2's greedy walk over
 * a candidate lattice. A block-diffusion drafter proposes every position of a block in one pass, so
 * it cannot condition position l+1 on its pick at l; this puts that dependence back.
 *
 *   score[l, c] = unary[l, c] + sum_r pred[pid(l)][r] * hp[l][r] * succ[cand[l, c]][r]
 *   pid(0) = anchor[s * anchor_stride],   pid(l) = cand[l - 1, chosen(l - 1)]
 *
 * `hp` ABSENT IS A PLANE OF ONES -- a plain low-rank bigram. The walk is SERIAL IN l, since the
 * predecessor is not known until the previous step is decided, so the parallel axis is the
 * sequence and the vector work is the rank-R dot a candidate costs. A predecessor or a candidate
 * outside the vocabulary contributes nothing rather than reading past a codebook; the highest
 * score wins and a tie goes to the lower candidate, libref's rule. */
static inline float dot_r(const float* a, const float* b, int64_t n) {
    vf acc = vf_zero();
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N) acc = vf_fma(vf_loadu(a + i), vf_loadu(b + i), acc);
    float s = vf_hsum(acc);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* ONE BODY FOR BOTH SPELLINGS: `psc` / `ssc` are the f32 scales of an E4M3 codebook (one a row, or one
 * for the table) and null for the plain one. The predecessor row's scale goes into `ax` once a step
 * and the candidate row's onto the finished dot, which is libref's and the kernel's own order. */
static int dflash_select_walk(const RadArgs* a, const RadTensor* cand, const RadTensor* un,
                              const RadTensor* hp, const RadTensor* anc, const RadTensor* pred,
                              const RadTensor* psc, const RadTensor* succ, const RadTensor* ssc,
                              const RadTensor* out) {
    if (!cand->data || !un->data || !anc->data || !pred->data || !succ->data || !out->data)
        return RAD_E_INVAL;

    const int64_t steps = p_int(a, "steps", 0);
    const int64_t K     = p_int(a, "top_k", 0);
    /* `cand` is [M * steps, K] and it is the call; `M` is the band (avx_common.h, band_rows). */
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
    auto scale_ok = [&](const RadTensor* t) {
        return !t || (t->data && (numel(t) == 1 || numel(t) >= nv));
    };
    if (!scale_ok(psc) || !scale_ok(ssc)) return RAD_E_SHAPE;
    auto row_scale = [&](const RadTensor* t, int64_t id) -> float {
        if (!t) return 1.0f;
        return ldt(t, numel(t) == 1 ? 0 : id * laststride(t));
    };

    const int64_t cs = laststride(cand), us = laststride(un), os = laststride(out);

    AVX_PARALLEL_FOR_IF(M * steps * K * R >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* ax = scratch_f32((size_t)R);
        float* rv = scratch_f32_b((size_t)R);
        float* hv = scratch_f32_c((size_t)R);
        int64_t pid = ldi(anc, offlin(anc, m * astr));
        for (int64_t l = 0; l < steps; ++l) {
            const int64_t r  = m * steps + l;
            const int64_t cb = rowoff(cand, r, K), ub = rowoff(un, r, K);
            /* The half of the form that does not depend on the candidate, once a step. */
            const bool ok = pid >= 0 && pid < nv;
            const float* pv = ok ? in_row(pred, rowoff(pred, pid, R), R, rv) : nullptr;
            const float* hw = hp ? in_row(hp, rowoff(hp, r, R), R, hv) : nullptr;
            const float pscl = ok ? row_scale(psc, pid) : 0.0f;
            for (int64_t i = 0; i < R; ++i)
                ax[i] = (ok ? pv[i] * pscl : 0.0f) * (hp ? hw[i] : 1.0f);

            int64_t best_c = 0;
            float   best_s = -INFINITY;
            for (int64_t c = 0; c < K; ++c) {
                const int64_t tok = ld_int_dt(cand->dtype, cand->data, cb + c * cs);
                float sc = ldt(un, ub + c * us);
                if (tok >= 0 && tok < nv)
                    sc += dot_r(ax, in_row(succ, rowoff(succ, tok, R), R, rv), R) *
                          row_scale(ssc, tok);
                if (sc > best_s) { best_s = sc; best_c = c; }
            }
            pid = ld_int_dt(cand->dtype, cand->data, cb + best_c * cs);
            st_int_dt(out->dtype, out->data, rowoff(out, m, steps) + l * os, pid);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(dflash_select) {
    if (a->n_t < 7) return RAD_E_INVAL;
    return dflash_select_walk(a, &a->t[0], &a->t[1], rad_arg_in(a, 2), &a->t[3], &a->t[4], nullptr,
                              &a->t[5], nullptr, &a->t[6]);
}

/* `cand`, `unary`, `hp`?, `anchor`, `pred`w, `pred_scale`w, `succ`w, `succ_scale`w -> `tokens`:
 * the same walk over codebooks held as E4M3 codes with an f32 scale a row. */
AVX_KERNEL(dflash_select_q) {
    if (a->n_t < 9) return RAD_E_INVAL;
    if (!a->t[5].data || !a->t[7].data) return RAD_E_INVAL;
    return dflash_select_walk(a, &a->t[0], &a->t[1], rad_arg_in(a, 2), &a->t[3], &a->t[4], &a->t[5],
                              &a->t[6], &a->t[7], &a->t[8]);
}
