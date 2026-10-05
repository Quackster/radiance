/* draft.cpp -- the static draft-depth policy, and the adaptive one RADIANCE_DRAFT_ADAPT turns on.
 * See draft.h for why the first is the default and what the second changes.
 */
#include "draft.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rad {

namespace {

/* THE COST MODEL WHEN NO TABLE IS GIVEN: every draft adds six percent to a step -- one round of
 * the head (about 0.4 ms profiled on QFN at tp4) and one more verify row, against a step of about
 * 12 ms. A guess at a shape, not a measurement of any deployment: one that has measured its own
 * steps by depth gives them as RADIANCE_DRAFT_COST. */
constexpr double kCostSlope = 0.06;

/* THE PRIOR WEIGHS AS MUCH AS THIS MANY VERIFIES. The (n+1)-th real one moves an estimate by
 * 1 / (n + k + 1) -- a running mean over the prior and the evidence -- until that step falls below
 * the half-life's, and from then on it is the moving average it is meant to be. Without it a
 * sequence would spend its first ~16 verifies at whatever window the prior chose, which is most of
 * a short answer. */
constexpr double kPriorWeight = 4.0;

/* A POSITION THE DEPLOYMENT HAS EVALUATED FEWER TIMES THAN THIS starts a new sequence from the
 * fixed prior rather than from the global count: a rate over a handful of verifies is noise, and a
 * position nobody has drafted to is no rate at all. */
constexpr int64_t kGlobalMinReach = 64;

/* RADIANCE_DRAFT_COST="1:10.6,2:11.3,3:12.0": depth:cost pairs, comma separated. Returns how many
 * it read, or -1 when the string is not that. */
int parse_cost(const char* s, float* cost) {
    int n = 0;
    const char* p = s;
    for (;;) {
        while (*p == ' ') ++p;
        if (!*p) return n;
        char* end = nullptr;
        const long d = std::strtol(p, &end, 10);
        if (end == p || *end != ':') return -1;
        p = end + 1;
        const double v = std::strtod(p, &end);
        if (end == p) return -1;
        if (d < 1 || d > RAD_SCHED_MAX_SPEC || !std::isfinite(v) || !(v > 0.0)) return -1;
        cost[d] = (float)v;
        ++n;
        p = end;
        while (*p == ' ') ++p;
        if (*p == ',') ++p;
        else if (*p) return -1;
    }
}

/* One knob, or the default with a warning when it does not parse. A knob that is silently not
 * read is the one that gets reported as "made no difference". */
void env_real(const char* name, double lo, double hi, float* out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (end == v || *end || !(x >= lo && x <= hi)) {
        RAD_WARN("%s=%s is not a number in [%g, %g]; keeping %g", name, v, lo, hi, (double)*out);
        return;
    }
    *out = (float)x;
}

void env_int(const char* name, long lo, long hi, int* out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end || x < lo || x > hi) {
        RAD_WARN("%s=%s is not an integer in [%ld, %ld]; keeping %d", name, v, lo, hi, *out);
        return;
    }
    *out = (int)x;
}

}  /* namespace */

void DraftController::from_env(Params* p) {
    static const Params env = [] {
        Params e;
        const char* on = std::getenv("RADIANCE_DRAFT_ADAPT");
        e.adapt = on && *on && *on != '0';
        if (const char* c = std::getenv("RADIANCE_DRAFT_COST"); c && *c) {
            const int n = parse_cost(c, e.cost);
            if (n < 2) {
                RAD_WARN("RADIANCE_DRAFT_COST=\"%s\" %s; the adaptive draft depth prices a step at "
                         "1 + %.2f per draft instead", c,
                         n < 0 ? "is not a list of depth:cost pairs"
                               : "has fewer than two depth:cost pairs", kCostSlope);
                std::memset(e.cost, 0, sizeof e.cost);
            }
        }
        env_real("RADIANCE_DRAFT_ADAPT_HYST", 0.0, 10.0, &e.hyst);
        env_int ("RADIANCE_DRAFT_ADAPT_HOLD", 0, 1000000, &e.hold);
        env_real("RADIANCE_DRAFT_ADAPT_HALFLIFE", 0.5, 1e6, &e.half_life);
        env_real("RADIANCE_DRAFT_ADAPT_PRIOR", 0.0, 1.0, &e.prior);
        env_int ("RADIANCE_DRAFT_ADAPT_MIN", 1, RAD_SCHED_MAX_SPEC, &e.min_depth);
        env_int ("RADIANCE_DRAFT_ADAPT_BUSY", 0, 1000000, &e.busy);
        env_int ("RADIANCE_DRAFT_ADAPT_BUSY_DEPTH", 1, RAD_SCHED_MAX_SPEC, &e.busy_depth);
        return e;
    }();
    p->adapt     = env.adapt;
    std::memcpy(p->cost, env.cost, sizeof p->cost);
    p->hyst      = env.hyst;
    p->hold      = env.hold;
    p->half_life = env.half_life;
    p->prior     = env.prior;
    p->min_depth = env.min_depth;
    p->busy       = env.busy;
    p->busy_depth = env.busy_depth;
}

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    if (p.adapt && (!(p.hyst >= 0.0f) || p.hold < 0 || !(p.half_life > 0.0f) ||
                    !(p.prior >= 0.0f && p.prior <= 1.0f) || p.min_depth < 1 || p.busy < 0 ||
                    p.busy_depth < 1)) {
        RAD_ERR("adaptive draft depth: hysteresis %g, hold %d, half-life %g, prior %g, minimum %d, "
                "busy %d, busy depth %d -- they want >= 0, >= 0, > 0, 0..1, >= 1, >= 0 and >= 1",
                (double)p.hyst, p.hold, (double)p.half_life, (double)p.prior, p.min_depth, p.busy,
                p.busy_depth);
        return RAD_E_INVAL;
    }
    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;

    adapt_ = p.adapt && p.max_depth > 0;
    alpha_ = 0.0;
    for (int d = 0; d <= RAD_SCHED_MAX_SPEC; ++d) { cost_[d] = 0.0; by_depth_[d] = 0; }
    /* SAID, because a switch that does nothing reads exactly like a switch that made no
     * difference. */
    if (p.adapt && !adapt_)
        RAD_WARN("RADIANCE_DRAFT_ADAPT is set and this deployment does not speculate "
                 "(--num-speculative-tokens 0): there is no window to adapt");
    if (!adapt_) return RAD_OK;
    if (p_.min_depth > p_.max_depth) p_.min_depth = p_.max_depth;
    alpha_ = 1.0 - std::exp2(-1.0 / (double)p_.half_life);

    /* THE COST TABLE, resolved for every window the policy may choose. A point that was measured
     * is taken as it is; between two points the line between them; past the ends the line through
     * the nearest two -- a table for depths 1..6 still prices depth 8 the way its own last two
     * steps say. A table that extrapolates to a cost at or below zero is a table with the wrong
     * sign in it, and the default is the better guess. */
    int    pd[RAD_SCHED_MAX_SPEC];
    double pv[RAD_SCHED_MAX_SPEC];
    int    np = 0;
    for (int d = 1; d <= RAD_SCHED_MAX_SPEC; ++d)
        if (std::isfinite(p.cost[d]) && p.cost[d] > 0.0f) {
            pd[np] = d;
            pv[np] = (double)p.cost[d];
            ++np;
        }
    bool table = np >= 2;
    for (int d = 1; table && d <= p_.max_depth; ++d) {
        int i = 0;
        while (i + 2 < np && pd[i + 1] < d) ++i;
        const double t = (double)(d - pd[i]) / (double)(pd[i + 1] - pd[i]);
        cost_[d] = pv[i] + t * (pv[i + 1] - pv[i]);
        if (!(cost_[d] > 0.0)) {
            RAD_WARN("adaptive draft depth: the cost table extrapolates to %g at depth %d; pricing "
                     "a step at 1 + %.2f per draft instead", cost_[d], d, kCostSlope);
            table = false;
        }
    }
    if (!table)
        for (int d = 1; d <= p_.max_depth; ++d) cost_[d] = 1.0 + kCostSlope * (double)d;

    std::string t;
    for (int d = 1; d <= p_.max_depth; ++d) t += fmt("%s%d:%.3g", d > 1 ? " " : "", d, cost_[d]);
    RAD_INFO("draft depth: ADAPTIVE per sequence (RADIANCE_DRAFT_ADAPT), %d..%d -- the window "
             "maximising E(d)/T(d) for T = [%s]%s; moves at +%.1f%% and at most every %d verifies; "
             "acceptance half-life %.1f verifies, prior %.2f",
             p_.min_depth, p_.max_depth, t.c_str(), table ? "" : " (the default model)",
             100.0 * (double)p_.hyst, p_.hold, (double)p_.half_life, (double)p_.prior);
    if (p_.busy > 0)
        RAD_INFO("draft depth: above %d running sequences every window is capped at %d "
                 "(RADIANCE_DRAFT_ADAPT_BUSY)", p_.busy, p_.busy_depth);
    return RAD_OK;
}

void DraftController::observe(int drafted, int accepted) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;

    /* Position j was EVALUATED only if every position before it was accepted, so this verify
     * produces exactly: 0..accepted-1 evaluated and accepted, and -- when the prefix was not the
     * whole draft -- position `accepted` evaluated and rejected. Nothing past that was reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
    if (adapt_) ++by_depth_[drafted];
}

/* ------------------------------------------------------------------ the adaptive policy */
int DraftController::best_depth(const float* a, int at, double* best, double* at_gain) const {
    double e = 1.0, run = 1.0, bg = -1.0, ag = 0.0;
    int arg = p_.min_depth;
    for (int d = 1; d <= p_.max_depth; ++d) {
        run *= (double)a[d - 1];
        e += run;
        const double g = e / cost_[d];
        if (d == at) ag = g;
        if (d >= p_.min_depth && g > bg) { bg = g; arg = d; }
    }
    if (best) *best = bg;
    if (at_gain) *at_gain = ag;
    return arg;
}

double DraftController::gain(const DraftAdapt& s, int d) const {
    if (!adapt_ || d < 1 || d > p_.max_depth) return 0.0;
    double e = 1.0, run = 1.0;
    for (int k = 1; k <= d; ++k) { run *= (double)s.a[k - 1]; e += run; }
    return e / cost_[d];
}

int DraftController::depth_for(DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth > 0) return s.depth;
    /* NOT STARTED. Each position from what the deployment has measured there, where it has
     * measured enough; otherwise from the fixed prior, which is optimistic on purpose: a window
     * that starts deep evaluates every position and learns all of them, and one that starts
     * shallow learns nothing about the positions it never drafts to. */
    for (int j = 0; j < p_.max_depth; ++j) {
        s.a[j] = reach_[j] >= kGlobalMinReach ? (float)((double)hit_[j] / (double)reach_[j])
                                              : p_.prior;
        s.n[j] = 0;
    }
    s.depth = best_depth(s.a, 0, nullptr, nullptr);
    s.held  = 0;
    return s.depth;
}

int DraftController::adapt_observe(DraftAdapt& s, int drafted, int accepted) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) (void)depth_for(s);
    if (drafted <= 0) return s.depth;
    const int D = p_.max_depth;
    if (drafted > D) drafted = D;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    /* THE POSITIONS THIS VERIFY EVALUATED, by observe()'s rule: every one before the first
     * rejection, taken, and the rejected one, refused. A position behind the rejection was drafted
     * and never looked at, and says nothing either way. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        const double w = std::fmax(alpha_, 1.0 / ((double)s.n[j] + kPriorWeight + 1.0));
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* AND THE POSITIONS THE WINDOW DID NOT REACH DRIFT TOWARD THE DEEPEST ONE IT DID. Nothing is
     * learned about them while the window is shorter -- and a sequence held at three by prose that
     * turns into code would otherwise keep pricing positions four and up at whatever they read
     * before, and never draft deep enough to find out. The guess is that acceptance decays with
     * position rather than falling off a cliff, which is what QFN's head measured, so the window's
     * edge stands in for what lies past it. Where that is wrong it costs a move out, the verifies
     * that refute it, and a move back. */
    const double edge = (double)s.a[drafted - 1];
    for (int j = drafted; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));

    /* THE MOVE, rate-limited twice. A window changes the shape of the verify pass and of the
     * head's history pass, and a shape is a new recording before it plays (runtime/ctx.h); a
     * window that follows every flicker of the estimate pays that for nothing. So it moves only
     * when the window it would take is worth `hyst` more than the one it has, and not again within
     * `hold` verifies of the last move. */
    if (s.held < INT32_MAX) ++s.held;
    if (s.held < p_.hold) return s.depth;
    double best = 0.0, cur = 0.0;
    const int d = best_depth(s.a, s.depth, &best, &cur);
    if (d != s.depth && best > cur * (1.0 + (double)p_.hyst)) {
        s.depth = d;
        s.held  = 0;
    }
    return s.depth;
}

}  /* namespace rad */
