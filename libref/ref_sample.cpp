/* ref_sample.cpp -- the sampler ops.
 *
 * Every sampler in this engine is a KERNEL and not a host lift: at a 151K vocab and batch 256 the
 * logits are ~77 MiB a step, and moving that across the link to run a host chain costs both the
 * bandwidth and a synchronisation, on the step path, at exactly the concurrency the engine exists
 * to serve (spec §13). llama.cpp's chain is kept as a host reference in core/sample/, and THIS is
 * the second implementation of the same arithmetic -- the two are deliberately independent, so
 * that comparing them says something.
 *
 * ============================== EVERY SCALAR IS PER ROW ==============================
 *
 * `rep`, `freq`, `pres`, `k` and `p` cannot be op PARAMETERS. A parameter is geometry: the selector
 * matches constraints against it and freezes it at declare (spec §2.2), so two requests in one
 * batch with different top_p would be two different resolutions of one op. A deployment would have
 * exactly one top_p, which is not a sampler.
 *
 * So they arrive as DATA, in a `params` operand of RadSampleParams rows -- one per sampled
 * position, uploaded once a step (abi/rad_sample.h). That is the schema core/sample/sampler.h
 * issues against, and the one every op below reads its scalars from.
 *
 * `seed` and `pos` are fields of the row for the same reason, rather than two more operands of
 * `sample_pick`.
 *
 * ============================== THE CANDIDATE SET ==============================
 *
 * `sample_topk` produces a fixed-width candidate set and the narrowing stages work IN PLACE. The
 * tensors do not shrink, so a dropped candidate has to be expressible, and the convention is:
 * index -1, value -infinity. Kept candidates stay where they are, in descending order, so a later
 * stage can stop at the first -1 or skip the holes -- both work, and `sample_pick` skips.
 *
 * `cand_val` carries LOGITS, not probabilities. Each narrowing stage takes its own softmax over
 * the candidate set, which is what llama.cpp does and what keeps the stages composable in any
 * order.
 *
 * A ROW'S OWN top_k MAY BE NARROWER THAN THE BUFFER. The candidate width is a deployment bound
 * chosen at declare; a request asking for top_k 40 out of a 1024-wide buffer gets 40 candidates
 * and 984 holes, which every later stage already knows how to skip.
 *
 * A GREEDY ROW (RAD_SP_GREEDY) HAS NO CANDIDATES. Its token is sample_argmax's, and in a batch
 * that mixes greedy and sampled rows every stage runs over every row -- so top-k and the merge
 * write it an empty set, the narrowing stages leave it alone, and the pick does not touch its
 * token. A pick that ignored the flag would replace the argmax with a draw.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include "rad_sample.h"

#include <algorithm>
#include <cfloat>
#include <unordered_map>

using namespace ref;

namespace {

/* The params operand, checked once. A sampler op with no rows to act on is not an error -- it is
 * a step in which nothing asked for that stage. */
struct Rows {
    const RadSampleParams* p;
    int64_t                n;
};

static inline int rows_of(const RadTensor* t, int64_t M, Rows* out) {
    if (!t || !t->data) return RAD_E_INVAL;
    /* The row array is a BYTE BLOB viewed through some 32-bit dtype -- the sampler declares it
     * [rows, sizeof(RadSampleParams)/4] u32, which is the only way the builder can express "this
     * many bytes a row". So the extent is checked in words, and the dtype is required to be a
     * 32-bit one rather than run through rad_dtype_bytes -- not because that is out of reach (it
     * is static inline in rad_types.h, and this file could call it), but because "any width, as
     * long as the arithmetic works out" is the wrong contract here: the blob is addressed as
     * 32-bit words and a 16-bit view of it would pass a byte count and then misread every row. */
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return RAD_E_DTYPE;
    if (numel(t) * 4 < M * (int64_t)sizeof(RadSampleParams)) return RAD_E_SHAPE;
    out->p = (const RadSampleParams*)t->data;
    out->n = M;
    return RAD_OK;
}

}  /* namespace */

/* ------------------------------------------------------------------ the history window */
/* The window both history stages read, as the row states it: `hist_off` and `hist_len` address the
 * row's OWN row of the history plane, clamped to it, and `last_n` then keeps the TAIL of that span
 * -- the last n tokens, not the first, because a window applied at the wrong end penalises what
 * the request has already forgotten. last_n 0 turns the stage off for the row (false), a negative
 * one keeps the whole span: llama.cpp's meaning for repeat_last_n and dry_penalty_last_n. */
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
 *     presence     logit -= pres                                        (once per distinct token)
 *
 * The repetition penalty being multiplicative and sign-dependent is the part everyone gets wrong:
 * dividing a negative logit would make it LARGER, so it is multiplied instead.
 *
 * The history holds GLOBAL token ids and the logits row is this rank's shard of the vocabulary,
 * so a token's column is its id less `vocab_off`, and a token outside the shard is another rank's
 * to penalise. */
int ref_sample_penalties(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    #pragma omp parallel for schedule(static)
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
        for (int64_t i = 0; i < hl; ++i) {
            const int64_t tok = ld_int(hist->dtype, hist->data, hb + i * hs);
            const int64_t col = tok - voff;
            if (tok < 0 || col < 0 || col >= n_vocab) continue;
            /* First occurrence only: the penalties are per distinct token, and the count is what
             * carries the multiplicity. O(hist^2) and the history is tens of tokens. */
            bool first = true;
            int64_t count = 0;
            for (int64_t j = 0; j < hl; ++j) {
                const int64_t t2 = ld_int(hist->dtype, hist->data, hb + j * hs);
                if (t2 != tok) continue;
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
/* DRY -- "don't repeat yourself". A penalty on the token that would EXTEND a suffix of the history
 * that has occurred before, which is what stops a model looping on a phrase rather than on a
 * token. llama.cpp's rule (llama_sampler_dry_apply):
 *
 *     for every earlier position j, n_j is how long a suffix of the window also ends at j, capped
 *     at the breaker limit; the token after j would extend that repeat. Each such token is
 *     penalised ONCE, by multiplier * base^(L - allowed_length), where L is the LONGEST n_j among
 *     the positions it follows.
 *
 * Only lengths at or above `allowed_length` count, so ordinary repetition of short n-grams is
 * untouched, and a token that is itself a whole sequence breaker is never penalised -- it is what
 * ends the repetition. Both breaker facts come from the host, which is the side with the
 * vocabulary: the row's `dry_rep_limit` (0 for none) and the optional `breakers` plane, one byte
 * per history position, nonzero where that position's token is a single-token breaker.
 *
 * The penalty is multiplier * base^e by repeated multiplication, stopping short of overflow, so
 * every implementation of this op computes the same float; llama.cpp clamps the exponent to the
 * same end. History ids are GLOBAL and a column is an id less `vocab_off`, as for the penalties.
 *
 * Written the obvious quadratic way over the window: this is the oracle, not the fast path. */
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

int ref_sample_dry(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *hist = &a->t[2];
    const RadTensor* brk = t_in(a, 3);
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));
    const int64_t hmax = numel(hist) / M;
    if (hmax <= 0) return RAD_E_SHAPE;
    /* The breaker plane is the history's twin, position for position. */
    if (brk && numel(brk) < M * hmax) return RAD_E_SHAPE;
    const int64_t voff = p_int(a, "vocab_off", 0);

    #pragma omp parallel for schedule(static)
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
        auto tok_at = [&](int64_t i) {
            return ld_int(hist->dtype, hist->data, hb + i * hs);
        };
        auto exempt = [&](int64_t i) {
            return brk && ld_int(brk->dtype, brk->data, bb + i * bs) != 0;
        };

        /* The longest repeat each continuation token would extend. */
        std::unordered_map<int64_t, int64_t> longest;
        for (int64_t j = 0; j + 1 < hl; ++j) {
            const int64_t lim = j + 1 < cap ? j + 1 : cap;
            int64_t n = 0;
            while (n < lim && tok_at(j - n) == tok_at(hl - 1 - n)) ++n;
            if (n < allowed || exempt(j + 1)) continue;
            const int64_t next = tok_at(j + 1);
            const int64_t col  = next - voff;
            if (next < 0 || col < 0 || col >= n_vocab) continue;
            int64_t& L = longest[next];
            if (n > L) L = n;
        }

        const float base = sp.dry_base > 0.0f ? sp.dry_base : 1.75f;
        for (const auto& e : longest) {
            const int64_t off = lb + (e.first - voff) * ls;
            stt(lg, off, ldt(lg, off) - dry_penalty(sp.dry_multiplier, base, e.second - allowed));
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_temp */
/* temp <= 0 leaves the row untouched: greedy decoding is `sample_argmax`, and a temperature of
 * zero silently meaning something else is how a sampler chain becomes unreadable. */
int ref_sample_temp(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const float t = R.p[m].temp;
        if (!(t > 0.0f) || t == 1.0f) continue;
        const float inv = 1.0f / t;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        for (int64_t i = 0; i < n_vocab; ++i) stt(lg, lb + i * ls, ldt(lg, lb + i * ls) * inv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_topk */
int ref_sample_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));
    /* The candidate WIDTH is the buffer's, not the request's: it is the deployment bound the
     * arena was sized for, and every later stage reads it. */
    const int64_t n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : numel(ci) / M;
    if (n_cand <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;

    /* THE VOCAB-PARALLEL PAIR PLANE, optional. cand_idx stays LOCAL -- it indexes this rank's
     * logits row and every stage after top-k reads it that way -- and `pairs` carries the same
     * candidates with `vocab_off` added, as (u32 GLOBAL id, f32 logit), which is what the
     * all-gather moves and what sample_merge_topk reads. Absent, and this is the single-rank
     * top-k unchanged. */
    const RadTensor* pp = a->n_t > 4 && a->t[4].data ? &a->t[4] : nullptr;
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (pp && numel(pp) < M * 2 * n_cand) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        /* A row's own top_k, clamped to the buffer. 0 or negative means "the whole candidate
         * width", which is what a request that set no top_k asks for. A greedy row takes none. */
        int64_t k = R.p[m].top_k;
        if (k <= 0 || k > n_cand) k = n_cand;
        if (R.p[m].flags & RAD_SP_GREEDY) k = 0;

        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);
        const int64_t pb = pp ? rowoff(pp, m, 2 * n_cand) : 0, ps = pp ? laststride(pp) : 0;
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            const bool ok = r < k && topk_next(lg, lb, ls, n_vocab, &pv, &pi);
            st_int(ci->dtype, ci->data, ib + r * is, ok ? pi : -1);
            stt(cv, vb + r * vs, ok ? pv : -INFINITY);
            if (!pp) continue;
            /* 0xffffffff is the hole, which the merge tests as a negative signed id -- the same
             * convention cand_idx carries as -1. */
            uint32_t bits;
            const float val = ok ? pv : -INFINITY;
            std::memcpy(&bits, &val, 4);
            st_int(pp->dtype, pp->data, pb + (2 * r) * ps,
                   ok ? (int64_t)(pi + voff) : (int64_t)0xffffffffu);
            st_int(pp->dtype, pp->data, pb + (2 * r + 1) * ps, (int64_t)bits);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the narrowing stages */
namespace {

enum Narrow { NR_TOPP, NR_MINP, NR_TYPICAL, NR_XTC };

static int narrow(const RadArgs* a, Narrow which) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (sp.flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        auto drop = [&](int64_t i) {
            st_int(ci->dtype, ci->data, ib + i * is, -1);
            stt(cv, vb + i * vs, -INFINITY);
        };
        auto live = [&](int64_t i) {
            return ld_int(ci->dtype, ci->data, ib + i * is) >= 0;
        };

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i)
            if (live(i)) { const float v = ldt(cv, vb + i * vs); if (v > mx) mx = v; }
        if (mx == -INFINITY) continue;

        if (which == NR_MINP) {
            const float p = sp.min_p;
            if (!(p > 0.0f)) continue;
            /* Keep everything within a factor p of the most likely candidate. Comparing
             * exp(v - max) against p is the same test as prob >= p * max_prob without forming the
             * normaliser, which is one fewer pass and one fewer rounding. */
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                const float v = ldt(cv, vb + i * vs);
                if (v == mx) continue;                       /* the max always survives */
                if (ref_exp(v - mx) < p) drop(i);
            }
            continue;
        }

        /* The remaining three all need the normalised probabilities. */
        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i)
            if (live(i)) sum += ref_exp(ldt(cv, vb + i * vs) - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

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
                if (!live(i)) continue;
                if (done) { drop(i); continue; }
                cum += ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                ++kept;
                if (cum >= p && kept >= 1) done = true;
            }
            continue;
        }

        if (which == NR_TYPICAL) {
            const float p = sp.typical_p;
            if (!(p < 1.0f) || !(p > 0.0f)) continue;
            /* Locally typical sampling: keep the candidates whose surprisal -log(prob) is closest
             * to the distribution's entropy, until their mass reaches p. The candidate order is
             * by probability and this order is by |surprisal - H|, so the two disagree and the
             * selection is a threshold on the distance rather than a prefix.
             *
             * Two passes and no sort: find the distance at which the mass reaches p, then keep
             * everything at or below it. Ties are kept, which can overshoot p slightly -- the
             * alternative is an order that depends on the candidate index. */
            float H = 0.0f;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                const float q = ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                if (q > 0.0f) H -= q * ref_log(q);
            }
            /* The smallest distance threshold whose mass reaches p, found by scanning the
             * distances that actually occur. n_cand is a thousand at most. */
            float cut = 0.0f;
            bool have_cut = false;
            for (int64_t c = 0; c < n_cand; ++c) {
                if (!live(c)) continue;
                const float qc = ref_exp(ldt(cv, vb + c * vs) - mx) * inv;
                if (!(qc > 0.0f)) continue;
                const float dc = ref_abs(-ref_log(qc) - H);
                float mass = 0.0f;
                for (int64_t i = 0; i < n_cand; ++i) {
                    if (!live(i)) continue;
                    const float q = ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                    if (!(q > 0.0f)) continue;
                    if (ref_abs(-ref_log(q) - H) <= dc) mass += q;
                }
                if (mass >= p && (!have_cut || dc < cut)) { cut = dc; have_cut = true; }
            }
            if (!have_cut) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                const float q = ref_exp(ldt(cv, vb + i * vs) - mx) * inv;
                if (!(q > 0.0f) || ref_abs(-ref_log(q) - H) > cut) drop(i);
            }
            continue;
        }

        /* XTC -- "exclude top choices". With probability `xtc_probability`, DROP every candidate
         * above `xtc_threshold` except the least likely of them, which is the opposite of every
         * other stage here and is the point: it removes the obvious continuations and leaves the
         * interesting one. The draw is per row and comes from the row's own stream, so it
         * reproduces. */
        {
            if (!(sp.xtc_probability > 0.0f) || !(sp.xtc_threshold > 0.0f)) continue;
            if (rad_sample_uniform01(sp.seed ^ RAD_SAMPLE_KEY_XTC, sp.pos) >= sp.xtc_probability)
                continue;
            int64_t last_above = -1, n_above = 0;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i)) continue;
                if (ref_exp(ldt(cv, vb + i * vs) - mx) * inv >= sp.xtc_threshold) {
                    last_above = i;
                    ++n_above;
                }
            }
            /* Nothing to do unless at least two candidates clear the threshold: dropping the only
             * one would leave the row without its own choice. */
            if (n_above < 2) continue;
            for (int64_t i = 0; i < n_cand; ++i) {
                if (!live(i) || i == last_above) continue;
                if (ref_exp(ldt(cv, vb + i * vs) - mx) * inv >= sp.xtc_threshold) drop(i);
            }
        }
    }
    return RAD_OK;
}

}  /* namespace */

int ref_sample_topp(const RadArgs* a, RadStream)    { return narrow(a, NR_TOPP); }
int ref_sample_minp(const RadArgs* a, RadStream)    { return narrow(a, NR_MINP); }
int ref_sample_typical(const RadArgs* a, RadStream) { return narrow(a, NR_TYPICAL); }
int ref_sample_xtc(const RadArgs* a, RadStream)     { return narrow(a, NR_XTC); }

/* ------------------------------------------------------------------ sample_mask */
/* Where a GBNF grammar lands. The state machine stays on host and what crosses is a token bitmask
 * per sequence -- 19 KiB at a 151K vocab, cheap enough to upload every step (spec §13).
 *
 * A SET BIT MEANS ALLOWED. That polarity has to be stated: a grammar produces a set of admissible
 * tokens, and the mask is that set rather than its complement. Word w bit b covers token
 * 32*w + b, least significant bit first.
 *
 * The row says WHICH mask row it uses, because a constrained sequence and an unconstrained one
 * share the plane and `mask_row` is -1 for the second. */
int ref_sample_mask(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *bm = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));
    const int64_t words = (n_vocab + 31) / 32;
    const int64_t mask_rows = numel(bm) / (words > 0 ? words : 1);
    if (mask_rows <= 0) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const RadSampleParams& sp = R.p[m];
        if (!(sp.flags & RAD_SP_MASK)) continue;
        const int64_t row = sp.mask_row;
        if (row < 0 || row >= mask_rows) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t bb = rowoff(bm, row, words), bs = laststride(bm);
        for (int64_t i = 0; i < n_vocab; ++i) {
            const uint32_t word = (uint32_t)ld_int(bm->dtype, bm->data, bb + (i / 32) * bs);
            if (!((word >> (i % 32)) & 1u)) stt(lg, lb + i * ls, -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_pick */
int ref_sample_pick(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ci = &a->t[0], *cv = &a->t[1], *pr = &a->t[2], *tok = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand || numel(tok) < M) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        /* A greedy row's token is the argmax's, already written. */
        if (R.p[m].flags & RAD_SP_GREEDY) continue;
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        float mx = -INFINITY;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ld_int(ci->dtype, ci->data, ib + i * is) < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > mx) mx = v;
        }
        if (mx == -INFINITY) { st_int(tok->dtype, tok->data, offlin(tok, m), -1); continue; }

        float sum = 0.0f;
        for (int64_t i = 0; i < n_cand; ++i) {
            if (ld_int(ci->dtype, ci->data, ib + i * is) < 0) continue;
            sum += ref_exp(ldt(cv, vb + i * vs) - mx);
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        const float u = rad_sample_uniform01(R.p[m].seed ^ RAD_SAMPLE_KEY_PICK, R.p[m].pos);

        /* The inverse CDF, walked in candidate order, accumulating in f32. The fallback is the
         * last candidate whose logit is ABOVE -infinity: `u` is strictly below 1 and the
         * probabilities sum to 1 in exact arithmetic, but they do not in f32, and a draw that
         * falls through the last comparison must still produce a token -- one the distribution
         * allows. top-k fills its width from a masked row, so the grammar's forbidden tokens sit
         * at the tail of the set at -infinity, and falling through to one of them emits a token
         * the grammar ruled out. Such a candidate adds nothing to `cum`, so the comparison never
         * chooses one either. */
        float cum = 0.0f;
        int64_t chosen = -1, last = -1;
        for (int64_t i = 0; i < n_cand; ++i) {
            const int64_t id = ld_int(ci->dtype, ci->data, ib + i * is);
            if (id < 0) continue;
            const float v = ldt(cv, vb + i * vs);
            if (v > -INFINITY) last = id;
            cum += ref_exp(v - mx) * inv;
            if (u < cum) { chosen = id; break; }
        }
        st_int(tok->dtype, tok->data, offlin(tok, m), chosen >= 0 ? chosen : last);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_argmax */
/* Greedy, ties to the LOWER token id -- the same tie rule the top-k selection uses, so a greedy
 * step and a top-1 step cannot disagree. */
int ref_sample_argmax(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *tok = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    if (n_vocab <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    if (M <= 0 || numel(tok) < M) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        float best = -INFINITY;
        int64_t bi = -1;
        for (int64_t i = 0; i < n_vocab; ++i) {
            const float v = ldt(lg, lb + i * ls);
            if (bi < 0 || v > best) { best = v; bi = i; }
        }
        st_int(tok->dtype, tok->data, offlin(tok, m), bi);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_logprobs */
/* THE RAW DISTRIBUTION'S SUMMARY, for a row that asks for logprobs (RAD_SP_LOGPROBS): the row's
 * maximum, the sum of exp(l - max) over it, and its k largest as (id, logit) pairs, descending,
 * ties to the lower id, ids as floats (exact below 2^24). Read off the logits before any sampling
 * stage has touched them, which is the distribution OpenAI's and llama-server's logprobs report.
 * A row without the flag is not written. Under a split vocabulary the ids are global
 * (+ `vocab_off`) and the host merges the ranks' summaries; a slot past the row's width is id -1,
 * logit -infinity. The sum is taken in double here: this is the oracle. */
int ref_sample_logprobs(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *pr = &a->t[1], *st = &a->t[2];
    const int64_t n_vocab = p_int(a, "n_vocab", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t k = p_int(a, "k", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    if (n_vocab <= 0 || k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_vocab;
    const int64_t w = 2 + 2 * k;
    if (M <= 0 || numel(st) < M * w) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        if (!(R.p[m].flags & RAD_SP_LOGPROBS)) continue;
        const int64_t lb = rowoff(lg, m, n_vocab), ls = laststride(lg);
        const int64_t sb = rowoff(st, m, w), ss = laststride(st);
        float mx = -INFINITY;
        for (int64_t i = 0; i < n_vocab; ++i) mx = std::max(mx, ldt(lg, lb + i * ls));
        double sum = 0.0;
        for (int64_t i = 0; i < n_vocab; ++i) sum += std::exp((double)ldt(lg, lb + i * ls) - mx);
        std::vector<std::pair<float, int64_t>> top;
        top.reserve((size_t)k + 1);
        auto above = [](const std::pair<float, int64_t>& x, const std::pair<float, int64_t>& y) {
            return x.first > y.first || (x.first == y.first && x.second < y.second);
        };
        for (int64_t i = 0; i < n_vocab; ++i) {
            const std::pair<float, int64_t> e{ ldt(lg, lb + i * ls), i };
            if ((int64_t)top.size() == k && !above(e, top.back())) continue;
            top.insert(std::upper_bound(top.begin(), top.end(), e, above), e);
            if ((int64_t)top.size() > k) top.pop_back();
        }
        stt(st, sb, mx);
        stt(st, sb + ss, (float)sum);
        for (int64_t j = 0; j < k; ++j) {
            const bool have_j = j < (int64_t)top.size();
            stt(st, sb + (2 + 2 * j) * ss, have_j ? (float)(top[(size_t)j].second + voff) : -1.0f);
            stt(st, sb + (3 + 2 * j) * ss, have_j ? top[(size_t)j].first : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ sample_merge_topk */
/* THE VOCAB-PARALLEL MERGE. lm_head is vocab-sharded, so each rank runs top-k over ITS OWN shard
 * and the candidate sets are all-gathered -- k * world_size pairs, 2 KiB at k=64 and world_size=4,
 * against the ~77 MiB a full-logits gather would move (spec §9, §13).
 *
 * `gathered` is [M, world_size * n_cand] pairs of (token id, logit) as raw u32 words: the token id
 * is already GLOBAL, because each rank adds its own vocab offset before the gather -- a merge that
 * had to know the offsets would need the shard layout, which is the caller's business and not a
 * kernel's. The output is the same descending candidate set the single-rank path produces, so
 * every stage after it is identical. */
int ref_sample_merge_topk(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *gt = &a->t[0], *pr = &a->t[1], *ci = &a->t[2], *cv = &a->t[3];
    const int64_t n_cand = p_int(a, "n_cand", ci->rank >= 2 ? ci->shape[ci->rank - 1] : 0);
    if (n_cand <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(ci) / n_cand;
    if (M <= 0 || numel(cv) < M * n_cand) return RAD_E_SHAPE;
    Rows R;
    RAD_REF_TRY(rows_of(pr, M, &R));
    /* Each gathered entry is a (u32 id, f32 logit) PAIR IN ONE BUFFER, so the two halves have
     * different types and the buffer's dtype can only be one of them. It is u32, and the logit is
     * read by BIT PATTERN: `ldt` on a u32 buffer would convert 0x40800000 to 1.08e9 rather than
     * to 4.0. A word-typed buffer is the only way the builder can express a packed pair. */
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

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t gb = rowoff(gt, m, per_row), gs = laststride(gt);
        const int64_t ib = rowoff(ci, m, n_cand), is = laststride(ci);
        const int64_t vb = rowoff(cv, m, n_cand), vs = laststride(cv);

        /* Selection sort over the gathered pairs, n_cand times. n_in is world_size * n_cand and
         * n_cand is a thousand at most, so this is the same O(n*k) the single-rank top-k already
         * pays and it needs no scratch. Ties to the lower token id, which is the tie rule every
         * other selection here uses. */
        /* THE ROW'S OWN top_k, RE-APPLIED. Every rank ran top-k over its own shard before the
         * gather, so the merged set holds up to world_size * k candidates where the request asked
         * for k, and no stage after this one narrows again. That is a different sampling support,
         * and at k = 1 it is a different ANSWER: a greedy row is staged at top_k = 1 exactly so
         * this chain returns the argmax, and with world_size candidates alive sample_pick draws
         * among them against the request's seed. 0 or negative means the whole width. */
        const int32_t tk = R.p[m].top_k;
        const int64_t keep = (R.p[m].flags & RAD_SP_GREEDY) ? 0
                           : (tk > 0 && (int64_t)tk < n_cand) ? (int64_t)tk : n_cand;

        float taken_v = INFINITY;
        int64_t taken_i = -1;
        for (int64_t r = 0; r < n_cand; ++r) {
            if (r >= keep) {
                st_int(ci->dtype, ci->data, ib + r * is, -1);
                stt(cv, vb + r * vs, -INFINITY);
                continue;
            }
            float best = -INFINITY;
            int64_t best_id = -1;
            for (int64_t j = 0; j < n_in; ++j) {
                const int64_t id = (int64_t)(uint32_t)ld_int(gt->dtype, gt->data, gb + 2 * j * gs);
                const float   v  = bits_to_f32(ld_int(gt->dtype, gt->data, gb + (2 * j + 1) * gs));
                if (id < 0) continue;
                const bool after = v < taken_v || (v == taken_v && id > taken_i);
                if (!after) continue;
                if (best_id < 0 || v > best || (v == best && id < best_id)) { best = v; best_id = id; }
            }
            st_int(ci->dtype, ci->data, ib + r * is, best_id);
            stt(cv, vb + r * vs, best_id >= 0 ? best : -INFINITY);
            if (best_id < 0) continue;
            taken_v = best;
            taken_i = best_id;
        }
    }
    return RAD_OK;
}
