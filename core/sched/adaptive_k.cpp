/* adaptive_k.cpp -- see adaptive_k.h. */
#include "adaptive_k.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace rad {

namespace {

bool env_double(const char* name, double* out, double lo, double hi) {
    const char* e = std::getenv(name);
    if (!e || !*e) return false;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (end == e || *end || !(v >= lo && v <= hi)) {
        RAD_WARN("%s=%s is not a number in [%g, %g]: keeping the default", name, e, lo, hi);
        return false;
    }
    *out = v;
    return true;
}

}  /* namespace */

AdaptiveK::Params AdaptiveK::Params::from_env() {
    Params p;
    const char* on = std::getenv("RADIANCE_ADAPTIVE_K");
    p.enabled = on && *on && std::strcmp(on, "0") != 0;

    if (const char* e = std::getenv("RADIANCE_ADAPTIVE_K_LENS")) {
        int v[kMaxLens], n = 0;
        bool bad = !*e;
        const char* s = e;
        while (!bad && *s) {
            char* end = nullptr;
            const long x = std::strtol(s, &end, 10);
            if (end == s || x < 1 || x > RAD_SCHED_MAX_SPEC || n >= kMaxLens) { bad = true; break; }
            v[n++] = (int)x;
            s = end;
            if (*s == ',') ++s;
            else if (*s) bad = true;
        }
        if (bad || n == 0) {
            RAD_WARN("RADIANCE_ADAPTIVE_K_LENS=%s is not a list of 1..%d lengths: keeping the default",
                     e, RAD_SCHED_MAX_SPEC);
        } else {
            std::sort(v, v + n);
            n = (int)(std::unique(v, v + n) - v);
            p.n_lens = n;
            for (int i = 0; i < n; ++i) p.lens[i] = v[i];
        }
    }
    env_double("RADIANCE_ADAPTIVE_K_ALPHA", &p.alpha, 0.01, 1.0);
    env_double("RADIANCE_ADAPTIVE_K_PRIOR", &p.prior, 0.0, 1000.0);
    env_double("RADIANCE_ADAPTIVE_K_COST_A", &p.cost_a, 0.0, 1e6);
    env_double("RADIANCE_ADAPTIVE_K_COST_B", &p.cost_b, 0.0, 1e6);
    double t;
    if (env_double("RADIANCE_ADAPTIVE_K_MIN_ROWS", &t, 0, 1 << 20)) p.min_rows = (int)t;
    if (env_double("RADIANCE_ADAPTIVE_K_QUANT", &t, 0, 1 << 20)) p.quant = (int)t;

    if (const char* e = std::getenv("RADIANCE_ADAPTIVE_K_COST_TABLE")) {
        double m[kMaxPts], ms[kMaxPts];
        int n = 0;
        bool bad = !*e;
        const char* s = e;
        while (!bad && *s) {
            char* end = nullptr;
            const double a = std::strtod(s, &end);
            if (end == s || *end != ':' || n >= kMaxPts) { bad = true; break; }
            s = end + 1;
            const double b = std::strtod(s, &end);
            if (end == s || a <= 0 || b <= 0 || (n > 0 && a <= m[n - 1])) { bad = true; break; }
            m[n] = a; ms[n] = b; ++n;
            s = end;
            if (*s == ',') ++s;
            else if (*s) bad = true;
        }
        if (bad || n < 2) {
            RAD_WARN("RADIANCE_ADAPTIVE_K_COST_TABLE=%s is not 'M:ms,M:ms,..' (ascending M, >= 2 points): "
                     "using a + b*M", e);
        } else {
            p.n_pts = n;
            for (int i = 0; i < n; ++i) { p.pt_m[i] = m[i]; p.pt_ms[i] = ms[i]; }
        }
    }
    return p;
}

int AdaptiveK::configure(const Params& p, int depth, int max_cands) {
    if (depth < 0 || depth > RAD_SCHED_MAX_SPEC || max_cands < 0) return RAD_E_INVAL;
    p_ = p;
    depth_ = depth;
    pool_hits_ = pool_seen_ = 0.0;
    lens_.clear();
    for (int i = 0; i < p_.n_lens && i < kMaxLens; ++i)
        if (p_.lens[i] >= 1 && p_.lens[i] < depth && (lens_.empty() || p_.lens[i] > lens_.back()))
            lens_.push_back(p_.lens[i]);
    if (depth > 0) lens_.push_back(depth);
    lvl_.assign((size_t)max_cands, 0);
    path_.clear();
    path_.reserve((size_t)max_cands * (size_t)kMaxLens);
    return RAD_OK;
}

void AdaptiveK::observe(Stat& s, int drafted, int accepted) {
    if (drafted <= 0) return;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    const double seen = (double)accepted + (accepted < drafted ? 1.0 : 0.0);
    const double g = 1.0 - p_.alpha;
    s.hits = g * s.hits + (double)accepted;
    s.seen = g * s.seen + seen;
    /* The pool decays slowly: it is only the prior of sequences with little history. */
    const double gp = 0.99;
    pool_hits_ = gp * pool_hits_ + (double)accepted;
    pool_seen_ = gp * pool_seen_ + seen;
}

double AdaptiveK::q_of(const Stat& s) const {
    const double qp = q_pool();
    const double d = s.seen + p_.prior;
    if (d <= 0.0) return qp;
    double q = (s.hits + p_.prior * qp) / d;
    return q < 0.0 ? 0.0 : (q > 1.0 ? 1.0 : q);
}

double AdaptiveK::cost_ms(double rows) const {
    if (p_.n_pts >= 2) {
        const int n = p_.n_pts;
        int i = 1;
        while (i < n - 1 && rows > p_.pt_m[i]) ++i;
        const double x0 = p_.pt_m[i - 1], x1 = p_.pt_m[i], y0 = p_.pt_ms[i - 1], y1 = p_.pt_ms[i];
        double t = y0 + (y1 - y0) * (rows - x0) / (x1 - x0);   /* extrapolates on the end segments */
        return t > 1e-3 ? t : 1e-3;
    }
    const double t = p_.cost_a + p_.cost_b * rows;
    return t > 1e-3 ? t : 1e-3;
}

int AdaptiveK::choose(const Cand* c, int n, int base_rows, int* k_out) {
    int full_rows = base_rows;
    for (int i = 0; i < n; ++i) {
        k_out[i] = c[i].n;
        full_rows += 1 + c[i].n;
    }
    if (!enabled() || n <= 0 || n > (int)lvl_.size() || lens_.size() < 2 || full_rows < p_.min_rows)
        return full_rows;

    /* Level ladder of candidate i: the set's lengths below n_i, then n_i itself. */
    auto len_at = [&](int i, int l) -> int {
        const int m = (int)lens_.size();
        int cnt = 0;                       /* lengths strictly below n_i */
        while (cnt < m && lens_[(size_t)cnt] < c[i].n) ++cnt;
        return l < cnt ? lens_[(size_t)l] : c[i].n;   /* l == cnt is the top (n_i) */
    };
    auto top_level = [&](int i) -> int {
        const int m = (int)lens_.size();
        int cnt = 0;
        while (cnt < m && lens_[(size_t)cnt] < c[i].n) ++cnt;
        return cnt;
    };
    /* sum_{j=lo+1..hi} q^j */
    auto chunk = [](double q, int lo, int hi) {
        double v = std::pow(q, (double)lo), s = 0.0;
        for (int j = lo + 1; j <= hi; ++j) { v *= q; s += v; }
        return s;
    };

    int rows = base_rows;
    double toks = 0.0;
    for (int i = 0; i < n; ++i) {
        lvl_[(size_t)i] = 0;
        const int k = len_at(i, 0);
        rows += 1 + k;
        toks += 1.0 + chunk(c[i].q, 0, k);
    }
    path_.clear();
    int    best_j = -1;                    /* -1: nothing admissible yet */
    double best_f = 0.0;
    auto admissible = [&](int r, bool full) {
        return full || p_.quant <= 1 || (r % p_.quant) == 0;
    };
    {
        const double f = toks / cost_ms((double)rows);
        if (admissible(rows, rows == full_rows)) { best_j = 0; best_f = f; }
    }
    for (;;) {
        int bi = -1;
        double br = -1.0;
        for (int i = 0; i < n; ++i) {
            const int l = lvl_[(size_t)i];
            if (l >= top_level(i)) continue;
            const int lo = len_at(i, l), hi = len_at(i, l + 1);
            const double r = chunk(c[i].q, lo, hi) / (double)(hi - lo);
            if (r > br) { br = r; bi = i; }          /* strict: ties keep the lower index */
        }
        if (bi < 0) break;
        const int l = lvl_[(size_t)bi];
        const int lo = len_at(bi, l), hi = len_at(bi, l + 1);
        rows += hi - lo;
        toks += chunk(c[bi].q, lo, hi);
        lvl_[(size_t)bi] = l + 1;
        path_.push_back(bi);
        const double f = toks / cost_ms((double)rows);
        if (admissible(rows, rows == full_rows) && (best_j < 0 || f >= best_f)) {
            best_j = (int)path_.size();
            best_f = f;
        }
    }
    if (best_j < 0) best_j = (int)path_.size();      /* cannot happen: the full state is admissible */

    for (int i = 0; i < n; ++i) lvl_[(size_t)i] = 0;
    for (int j = 0; j < best_j; ++j) ++lvl_[(size_t)path_[(size_t)j]];
    int total = base_rows;
    for (int i = 0; i < n; ++i) {
        k_out[i] = len_at(i, lvl_[(size_t)i]);
        total += 1 + k_out[i];
    }
    return total;
}

}  /* namespace rad */
