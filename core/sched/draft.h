/* draft.h -- the speculative draft-depth controller.
 *
 * IT IS STATIC. The depth is `--num-speculative-tokens` and nothing moves it.
 *
 * No dynamic policy sits here, and the reason is structural rather than a tuning failure. The
 * dynamic policy this engine is otherwise modelled on -- a confidence-product gate against a
 * decayed per-position acceptance estimate, plus a batch-size ceiling schedule -- gates on the
 * DRAFTER's own top-1 confidences, per slot, on device, after the drafter has run. The scheduler
 * decides the depth BEFORE the drafter runs and cannot see them; the only substitute available
 * here is measured acceptance, which is the right quantity in expectation but arrives one step
 * late and pooled across sequences. Against a draft head whose cost is nearly flat in depth (one
 * extra verify row on a step that is weight-bound, not compute-bound) that estimate has nothing to
 * buy: the best depth is a property of the model pair, it is known before the serve starts, and it
 * is a flag. A controller that can only pick a worse depth than the constant contributes only the
 * risk that it picks one.
 *
 * What survives is the accounting, because the acceptance rate is a reported metric (spec §14)
 * and because a verify that accepts nothing has to be visible.
 *
 * UNLESS RADIANCE_DRAFT_ADAPT=1, and then the premise above is what does not hold. On a model
 * whose weights stream -- a MoE that reads experts per verify row, a head that runs one pass a
 * round -- a step is NOT flat in depth: each draft is one more round of the head and one more
 * verify row, and on QFN at tp4 a step measured 12.0 ms at depth 3 and 15.0 ms at depth 5. And
 * the best depth is not a property of the model pair but of the TEXT: per-position acceptance
 * holds at ~0.9 to position five on code and JSON, sits near 0.6 on prose and falls to 0.2-0.4
 * past the third position on a summary. No constant is right for all of them, and the one that is
 * right for one costs the others either rounds nobody accepts or tokens nobody drafted.
 *
 * So the switch adds a policy, per sequence and on the host: a smoothed estimate of each
 * position's conditional acceptance, fed by the same verify outcomes observe() counts (DraftAdapt),
 * and before every draft the window d that maximises E(d) / T(d) -- the tokens a step of that
 * window is expected to emit, 1 + sum_k prod_{j<=k} a_j, over what such a step costs. It moves
 * only when the gain is worth a re-recorded pass (Params::hyst, Params::hold). `max_depth` stays
 * the ceiling, and everything sized for it stays sized for it.
 *
 * IT CHANGES HOW MANY DRAFTS A STEP VERIFIES AND NOTHING ELSE. Acceptance is the rule in
 * core/sample/accept.h, so what a step emits is the target's own token at every position whatever
 * window it verified. At temperature 0 that is the same text at every window, to exactly the
 * degree a speculative run matches a plain one at all: exact given the same logits, and a verify
 * row's logits are computed at an M that now moves with the window (accept.h says what that does
 * and does not change).
 */
#pragma once
#include "rad_core.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows precisely because a verify step falls in that
 * band (spec §10), so 64 is the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* ONE SEQUENCE'S HALF OF THE ADAPTIVE POLICY: what the drafter has done on this sequence's own
 * text, and the window that buys. It lives in the scheduler's per-request state and the controller
 * does the arithmetic, so neither has to look the other up.
 *
 * `depth` 0 is "not started": the first ask fills the estimate from the prior and picks a window.
 * It belongs to the REQUEST, not to the cache it holds -- a preemption keeps it, because the text
 * it describes is still the text the request is writing -- and a recycled slot starts it over. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the window the sequence gets now; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int max_depth = 0;        /* the deployment's window; 0 disables speculation entirely */

        /* THE ADAPTIVE POLICY (see the top of this file), off unless RADIANCE_DRAFT_ADAPT=1 --
         * from_env() is where the switch and the knobs below are read, once per process. With it
         * off every field below is ignored and the depth is max_depth, as it always was. */
        bool  adapt     = false;
        /* T(d), what a step verifying d drafts costs, by d; any unit, only ratios are read. An
         * entry above zero is a measured point, the rest are interpolated between the points and
         * extrapolated past them along the nearest two. Fewer than two points is no table, and
         * then T(d) = 1 + 0.06 d. Index 0 is unused. */
        float cost[RAD_SCHED_MAX_SPEC + 1] = {};
        float hyst      = 0.02f;  /* a move must buy this much more E/T than the window it leaves */
        int   hold      = 8;      /* and come at least this many verifies after the last move */
        float half_life = 16.0f;  /* of each position's estimate, in verifies that evaluate it */
        float prior     = 0.8f;   /* for a position too few verifies have evaluated to say */
        int   min_depth = 1;      /* the policy never drafts fewer */
        /* THE LOAD CAP. The cost table is a step's cost at ONE sequence, and a deeper window costs
         * more under load than it says -- every row a step adds is expert bandwidth all its
         * neighbours wait on. So with more than `busy` sequences running, every window is capped
         * at `busy_depth`; the estimates go on learning underneath and the windows come back when
         * the load does. 0 is no cap. */
        int   busy       = 0;
        int   busy_depth = 3;
    };

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* RADIANCE_DRAFT_ADAPT and its companions (RADIANCE_DRAFT_COST, _HYST, _HOLD, _HALFLIFE,
     * _PRIOR, _MIN, _BUSY, _BUSY_DEPTH), parsed ONCE per process and copied into the adaptive fields of `p`;
     * max_depth is left alone. Once, so that every step of a run is decided by the same rule --
     * a pass recorded under one setting and played under another is a pass the audit fails. */
    static void from_env(Params* p);

    /* The draft depth for the next step. 0 means "do not speculate". Constant by construction:
     * under the adaptive policy it is the CEILING, and depth_for() is the window. */
    int  next_depth() const { return p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }
    /* The window this sequence may draft into: its policy depth, starting the estimate on the
     * first ask; max_depth when the policy is off. */
    int  depth_for(DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Moves its
     * estimate, and its depth when that clears the hysteresis. Returns the depth after. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* The load cap for a cycle with `n_running` sequences running: busy_depth past the threshold,
     * 0 for none. The scheduler asks once a cycle and holds the answer (Scheduler::draft_cycle). */
    int  busy_cap(int n_running) const {
        return (adapt_ && p_.busy > 0 && n_running > p_.busy) ? p_.busy_depth : 0;
    }
    /* What the policy maximises, E(d) / T(d), for `s`'s estimate as it stands. For tests and logs. */
    double gain(const DraftAdapt& s, int d) const;
    /* The cost table as configured, T(1..max_depth). */
    double cost(int d) const {
        return (d >= 1 && d <= RAD_SCHED_MAX_SPEC) ? cost_[d] : 0.0;
    }
    /* Verifies by the window they carried, counted only under the policy: the histogram of the
     * depths it actually chose, which the mean gauge on its own cannot show. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* One verify's outcome: `drafted` tokens were proposed and a prefix of `accepted` of them was
     * taken. Feeds the reported metrics and nothing else. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* PER-POSITION ACCEPTANCE, and it is a diagnostic rather than a policy input.
     *
     * The pooled ratio above cannot tell a drafter that decays from one that falls off a cliff,
     * and the difference is the difference between "this is what the head is worth" and "round
     * three is broken". A drafter whose accepted-tokens-per-step stops rising past some depth has
     * a hazard that goes to zero there; pooled over the whole window that reads as a uniformly
     * mediocre drafter, which is a different diagnosis and a different fix.
     *
     * `reached` is "every earlier position was accepted, so this one was evaluated"; `hits` is
     * "and it was accepted too". Their ratio is the conditional hazard. A position the draft
     * never reached is charged nothing, which is why a rejection at position 0 does not libel
     * every position behind it. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has ever reached; how much of the two arrays is meaningful. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  cost_[RAD_SCHED_MAX_SPEC + 1] = {};   /* T(d), resolved for 1..max_depth */
    double  alpha_ = 0.0;                         /* the smoothing step from half_life */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};

    /* The window in [min_depth, max_depth] that maximises E/T for estimate `a`, the smaller on a
     * tie; its E/T in *best, and the E/T of window `at` in *at_gain. */
    int     best_depth(const float* a, int at, double* best, double* at_gain) const;
};

}  /* namespace rad */
