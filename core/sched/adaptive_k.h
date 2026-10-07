/* adaptive_k.h -- per-sequence verify length for a block drafter (opt-in: RADIANCE_ADAPTIVE_K=1).
 *
 * WHAT IT CHANGES. Only how many of the drafter's tokens a step VERIFIES. The block drafter has
 * already produced all `depth` tokens (one pass, depth is free); set_draft() / step() take a
 * prefix. A shorter prefix is a shorter verify row group for that sequence: fewer rows through
 * the trunk (M falls, and with it the decode step -- ~34 ms at M=8 against ~58 ms at M=64), and
 * fewer rows in the drafter's context pass, which mirrors the trunk step. The query pass is
 * untouched (a fixed block of `block` rows a sequence). Every emitted token is still the target's
 * own greedy/sampled choice, so the output is the same stream with fewer rows spent on positions
 * that were going to be rejected.
 *
 * THE MODEL. Sequence i accepts each further draft position with probability q_i, independently
 * (geometric). Verifying k_i drafts yields in expectation
 *         E_i(k) = 1 + sum_{j=1..k} q_i^j          tokens,
 * for 1 + k rows. The step time is T(M) with M = sum_i (1 + k_i): a + b*M by default, or a
 * piecewise-linear table measured on the card. The batch maximises
 *         sum_i E_i(k_i) / T(M)
 * over k_i drawn from a SMALL SET (default 1,3,5,7), by the greedy marginal-gain rule: start every
 * sequence at the smallest length, repeatedly take the upgrade with the best expected-tokens per
 * extra row (concave per sequence, so this is the optimum for every fixed M), and stop where
 * tokens/T(M) peaks. Deterministic: ties go to the lower index.
 *
 * q_i IS ESTIMATED from the sequence's own history with censoring: a verify of k drafts that took
 * `a` of them evaluated a + (a < k) positions, and q = hits / evaluated, over exponentially
 * decayed sums, shrunk toward the pooled estimate with `prior` pseudo-positions so a new sequence
 * starts at the batch average.
 *
 * WHY QUANTISED. A recorded pass is keyed by n_tok, max_q_len and n_spec (Ctx::pass_key), not by
 * which sequence has which length. Free per-sequence lengths would produce a new n_tok nearly
 * every step and thrash the 24 recorded tapes, which costs a live issue per miss. The length set
 * bounds max_q_len/n_spec, and `quant` restricts the total row count M to multiples of `quant`
 * (plus the unshrunk total), so there are few distinct shapes: for 8 sequences and quant 8 at
 * most the 7 shapes M = 16, 24, .., 64.
 *
 * SINGLE STREAM. Below `min_rows` total rows (default 32 = 4 sequences at depth 7) nothing is
 * shrunk: the step is weight-bound there and a draft row is nearly free.
 *
 * Pure host code: no allocation after configure(), no clocks, no globals except the env read.
 */
#pragma once
#include "draft.h"      /* RAD_SCHED_MAX_SPEC */

#include <cstdint>
#include <vector>

namespace rad {

class AdaptiveK {
public:
    enum { kMaxLens = 8, kMaxPts = 16 };

    struct Params {
        bool    enabled = false;
        int     n_lens  = 4;
        int     lens[kMaxLens] = { 1, 3, 5, 7 };   /* ascending, each >= 1; the depth is always added */
        double  alpha   = 0.25;      /* weight of the newest verify in a sequence's estimate */
        double  prior   = 2.0;       /* pseudo-positions pulling q_i toward the pooled estimate */
        double  cost_a  = 30.57;     /* ms: T(M) = a + b*M  (34 ms at M=8, 58 ms at M=64) */
        double  cost_b  = 0.4286;    /* ms per verify row */
        int     n_pts   = 0;         /* optional measured table (M, ms), ascending M; overrides a/b */
        double  pt_m[kMaxPts]  = {};
        double  pt_ms[kMaxPts] = {};
        int     min_rows = 32;       /* total rows below which nothing is shrunk */
        int     quant    = 8;        /* total rows restricted to multiples of this; 0 = free */

        /* RADIANCE_ADAPTIVE_K (0/1), _LENS "7,5,3,1", _ALPHA, _PRIOR, _COST_A, _COST_B,
         * _COST_TABLE "8:34,16:40,64:58", _MIN_ROWS, _QUANT. A malformed value keeps the default
         * and warns. */
        static Params from_env();
    };

    /* One sequence's history: decayed hits and decayed evaluated positions. */
    struct Stat { double hits = 0.0, seen = 0.0; };

    struct Cand {
        double q;     /* acceptance estimate, 0..1 */
        int    n;     /* drafts available (>= 1) */
    };

    int  configure(const Params& p, int depth, int max_cands);
    bool enabled() const { return p_.enabled && depth_ > 0; }
    const Params& params() const { return p_; }
    int  depth() const { return depth_; }

    /* A verify of `drafted` tokens took `accepted`. Updates the sequence and the pooled estimate. */
    void   observe(Stat& s, int drafted, int accepted);
    double q_of(const Stat& s) const;
    double q_pool() const { return pool_seen_ > 0.0 ? pool_hits_ / pool_seen_ : 0.75; }

    double cost_ms(double rows) const;

    /* The lengths for `n` candidates in k_out (1..n_i each). `base_rows` is every other row of the
     * step (sequences that take no draft). Returns the total verify rows of the choice. When the
     * unshrunk total is below min_rows, or the policy is off, k_out[i] = n_i. */
    int  choose(const Cand* c, int n, int base_rows, int* k_out);

private:
    Params p_;
    int    depth_ = 0;
    double pool_hits_ = 0.0, pool_seen_ = 0.0;
    std::vector<int>    lvl_;      /* current level index per candidate */
    std::vector<int>    path_;     /* upgrade order (candidate indices) */
    std::vector<int>    lens_;     /* the effective set: p_.lens < depth, then depth */
};

}  /* namespace rad */
