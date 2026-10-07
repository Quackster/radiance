/* lookup_draft.cpp -- see lookup_draft.h. */
#include "lookup_draft.h"

#include <cstdlib>
#include <cstring>

namespace rad {

namespace {
bool env_int(const char* name, int* out, int lo, int hi) {
    const char* e = std::getenv(name);
    if (!e || !*e) return false;
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || *end || v < lo || v > hi) {
        RAD_WARN("%s=%s is not an integer in [%d, %d]: keeping the default", name, e, lo, hi);
        return false;
    }
    *out = (int)v;
    return true;
}
}  /* namespace */

LookupParams LookupParams::from_env() {
    LookupParams p;
    const char* on = std::getenv("RADIANCE_LOOKUP_DRAFT");
    p.enabled = on && *on && std::strcmp(on, "0") != 0;
    env_int("RADIANCE_LOOKUP_ENTER", &p.enter, 1, 64);
    env_int("RADIANCE_LOOKUP_STAY", &p.stay, 1, 64);
    env_int("RADIANCE_LOOKUP_HOT", &p.hot, 1, 64);
    env_int("RADIANCE_LOOKUP_MAX_MATCH", &p.max_match, 1, 64);
    env_int("RADIANCE_LOOKUP_CHAIN", &p.chain, 1, 1 << 20);
    if (const char* e = std::getenv("RADIANCE_LOOKUP_MIN_DIST")) {
        if (*e && std::strcmp(e, "auto") != 0) env_int("RADIANCE_LOOKUP_MIN_DIST", &p.min_dist, 0, 1 << 24);
    }
    if (p.enabled && !p.valid()) {
        RAD_WARN("lookup draft: enter %d stay %d hot %d max_match %d out of range (need 1 <= stay <= enter <= "
                 "max_match <= 64, hot >= 1): OFF", p.enter, p.stay, p.hot, p.max_match);
        p.enabled = false;
    }
    return p;
}

void lookup_ref_best(const TokView& t, int minl, int maxl, int64_t min_dist, int* m_out, int64_t* p_out) {
    const int64_t n = t.size();
    int bm = 0;
    int64_t bp = 0;
    if (min_dist < 1) min_dist = 1;
    for (int64_t p = 1; p <= n - min_dist; ++p) {
        int m = 0;
        while (m < maxl && p - 1 - m >= 0 && t[p - 1 - m] == t[n - 1 - m]) ++m;
        if (m >= minl && (m > bm || (m == bm && p > bp))) { bm = m; bp = p; }
    }
    *m_out = bm;
    *p_out = bp;
}

void LookupIndex::reset() {
    ng_ = 0;
    indexed_ = 0;
    visited_ = 0;
    /* keep the capacity: a slot is reused by the next request */
    if (!head_.empty()) std::memset(head_.data(), 0, head_.size() * 4);
    prev_.clear();
}

uint32_t LookupIndex::hash(const TokView& t, int64_t end) const {
    uint32_t h = 2166136261u;
    for (int i = ng_; i >= 1; --i) {
        h ^= (uint32_t)t[end - i];
        h *= 16777619u;
        h ^= h >> 15;
    }
    h *= 2654435761u;
    return h >> (32 - kHashBits);
}

void LookupIndex::insert(const TokView& t, int64_t p) {
    /* p is the position the continuation would start at: the n-gram is t[p-ng .. p-1] */
    const uint32_t h = hash(t, p);
    if ((int64_t)prev_.size() <= p) prev_.resize((size_t)p + 1, 0);
    prev_[(size_t)p] = head_[h];
    head_[h] = (int32_t)p;
}

int LookupIndex::search(const TokView& t, const LookupParams& pr, int64_t min_dist, int64_t* p_out) {
    visited_ = 0;
    *p_out = 0;
    const int64_t n = t.size();
    const int minl = pr.min_match();
    const int ng = minl < 4 ? minl : 4;
    if (min_dist < 1) min_dist = 1;
    const int64_t limit = n - min_dist;        /* the largest eligible p */
    if (limit < ng || n < minl) return 0;
    if (ng_ != ng || head_.empty()) {
        if (head_.empty()) head_.assign((size_t)1 << kHashBits, 0);
        else std::memset(head_.data(), 0, head_.size() * 4);
        prev_.clear();
        indexed_ = 0;
        ng_ = ng;
    }
    if (indexed_ > n) {                       /* the sequence is not the one this index was built over */
        std::memset(head_.data(), 0, head_.size() * 4);
        prev_.clear();
        indexed_ = 0;
    }
    if (prev_.capacity() < (size_t)limit + 1) prev_.reserve((size_t)limit + 1 + 4096);
    /* the first position that can end an ng-gram is ng */
    for (int64_t p = (indexed_ < ng - 1 ? ng : indexed_ + 1); p <= limit; ++p) insert(t, p);
    if (indexed_ < limit) indexed_ = limit;

    const uint32_t h = hash(t, n);
    int best = 0;
    int64_t bp = 0;
    int vis = 0;
    for (int32_t c = head_[h]; c > 0 && vis < pr.chain; c = prev_[(size_t)c], ++vis) {
        int m = 0;
        while (m < pr.max_match && c - 1 - m >= 0 && t[c - 1 - m] == t[n - 1 - m]) ++m;
        if (m > best) {            /* newest first, so a tie keeps the larger p */
            best = m;
            bp = c;
            if (m >= pr.max_match) break;
        }
    }
    visited_ = vis;
    if (best < minl) return 0;
    *p_out = bp;
    return best;
}

}  /* namespace rad */
