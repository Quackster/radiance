/* lookup_draft.h -- whole-context prompt lookup that replaces a BLOCK drafter's proposal for a step
 * (opt-in: RADIANCE_LOOKUP_DRAFT=1). The host half of vllm-sly-radiance's radiance_lookup_draft.py.
 *
 * WHY. A block drafter with a sliding window (DFlash2: 2048) cannot see a source further back than
 * that, so a far copy / edit / re-emitted block is drafted at ~3.3 tokens a step instead of ~7. The
 * existing n-gram fallback (Scheduler::draft_ngram) only fills a proposal the drafter left EMPTY and
 * looks at 2..4 tokens; this one looks at the whole context, wants long matches, and takes over a
 * step the drafter DID propose for -- but only when it is very likely to win.
 *
 * THE MATCH. The longest earlier occurrence (up to `max_match` tokens, most recent on a tie) of the
 * sequence's current suffix whose CONTINUATION starts more than `min_dist` tokens back (the drafter's
 * window: a source inside it is one the drafter sees, and it is better at it). The 7 tokens that
 * followed it are the proposal.
 *
 * THE POLICY (ENTER 8, STAY 3, HOT 3). Enter lookup mode on a match of >= ENTER tokens; stay in it
 * while the last lookup step accepted >= HOT draft tokens and a match of >= STAY exists; otherwise
 * the drafter's proposal stands untouched. "Any match >= 3" loses 1-17 % on free generation.
 *
 * LOSSLESS. The verify is unchanged: a draft token is accepted iff it is the target's own pick.
 *
 * THE INDEX (LookupIndex), per sequence, incremental. A hash chain over the `ng` tokens (ng =
 * min(ENTER, STAY, 4)) ending at every position p, newest first, holding ONLY positions that are
 * already eligible, i.e. p <= length - min_dist: positions enter the index min_dist steps after they
 * were written, so the window rule needs no skipping and a chain walk visits only legal sources.
 * Cost per step: <= 1 + accepted inserts, one chain walk of <= `chain` candidates, each a backward
 * compare of <= max_match tokens that stops at the first mismatch; a walk stops early at a
 * max_match match. A sequence shorter than min_dist + ng has no index at all. The chain cap is the
 * only place the answer can differ from an exhaustive scan (pathologically repetitive text).
 *
 * Pure host code. No clocks, no globals except the env read.
 */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

/* Tokens of a sequence as the scheduler holds them: prompt followed by output. */
struct TokView {
    const int32_t* a = nullptr; int64_t na = 0;
    const int32_t* b = nullptr; int64_t nb = 0;
    int64_t size() const { return na + nb; }
    int32_t operator[](int64_t i) const { return i < na ? a[i] : b[i - na]; }
};

struct LookupParams {
    bool enabled   = false;
    int  enter     = 8;      /* match length that starts lookup mode */
    int  stay      = 3;      /* match length that keeps it going after a hot step */
    int  hot       = 3;      /* accepted draft tokens of the last lookup step that keep it hot */
    int  max_match = 24;     /* match length cap */
    int  min_dist  = -1;     /* <0: the drafter's window (else 2048); >=0: that many tokens; 0 = any */
    int  chain     = 1024;   /* candidates one search may visit */

    int  min_match() const { return enter < stay ? enter : stay; }
    bool valid() const { return enter >= 1 && stay >= 1 && hot >= 1 && enter >= stay && enter <= max_match &&
                                max_match <= 64 && chain >= 1; }
    /* RADIANCE_LOOKUP_DRAFT (0/1), _ENTER, _STAY, _HOT, _MAX_MATCH, _MIN_DIST (auto|N), _CHAIN. */
    static LookupParams from_env();
};

/* The entry rule of the policy, as a pure function: does a match of `m` tokens take the step? */
inline bool lookup_use(const LookupParams& p, int m, bool hot) {
    return m > 0 && m >= (hot ? p.stay : p.enter);
}

/* THE POLICY'S PER-SEQUENCE STATE. `used`: the proposal standing for the next step is the lookup's.
 * `hot`: the last lookup step that was verified accepted >= HOT draft tokens, which lowers the match
 * length that keeps lookup mode from ENTER to STAY for exactly the next decision. */
struct LookupMode {
    bool used = false;
    bool hot  = false;
    void reset() { used = false; hot = false; }
    /* The decision for a new proposal given the best match length (0 = none). Consumes `hot`. */
    bool decide(const LookupParams& p, int m) {
        used = lookup_use(p, m, hot);
        hot = false;
        return used;
    }
    /* The step that carried the standing proposal was verified and `accepted` draft tokens were taken. */
    void verified(const LookupParams& p, int accepted) {
        hot = used && accepted >= p.hot;
        used = false;
    }
};

/* Reference: exhaustive scan. (m, p) of the longest suffix of t (>= minl, capped at maxl) that ends
 * at t[p-1] for some 1 <= p <= n - max(min_dist, 1); ties -> the largest p. (0, 0) when none. */
void lookup_ref_best(const TokView& t, int minl, int maxl, int64_t min_dist, int* m, int64_t* p);

class LookupIndex {
public:
    void reset();
    bool empty() const { return indexed_ == 0; }
    int64_t indexed() const { return indexed_; }
    int64_t memory_bytes() const { return (int64_t)(head_.capacity() + prev_.capacity()) * 4; }

    /* Brings the index up to t (inserts the positions that have become eligible) and returns the
     * best match: m tokens (0 = none, or below the min match) with the continuation starting at *p. */
    int search(const TokView& t, const LookupParams& pr, int64_t min_dist, int64_t* p);

    /* Candidates the last search visited (a cost probe for the tests and the bench). */
    int last_visited() const { return visited_; }

private:
    enum { kHashBits = 17 };
    void insert(const TokView& t, int64_t p);
    uint32_t hash(const TokView& t, int64_t end) const;   /* of the ng_ tokens ending at end-1 */

    int ng_ = 0;
    int64_t indexed_ = 0;                 /* positions <= indexed_ are in the index */
    std::vector<int32_t> head_;           /* bucket -> newest position, 0 = none */
    std::vector<int32_t> prev_;           /* position -> next older position in its bucket */
    int visited_ = 0;
};

}  /* namespace rad */
