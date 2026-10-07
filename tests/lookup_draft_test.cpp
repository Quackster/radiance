/* lookup_draft_test.cpp -- sched/lookup_draft.h: the suffix matcher against an exhaustive reference, the
 * window and tie rules, the ENTER / STAY / HOT state machine, and what a search costs on the host at 262k. */
#include "rad_test.h"
#include "sched/lookup_draft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace rad;

namespace {

LookupParams params(int enter, int stay, int hot, int maxm, int min_dist, int chain = 1 << 20) {
    LookupParams p;
    p.enabled = true; p.enter = enter; p.stay = stay; p.hot = hot; p.max_match = maxm;
    p.min_dist = min_dist; p.chain = chain;
    return p;
}

/* a view with `split` tokens as the "prompt" and the rest as the "output", as the scheduler holds them */
TokView view(const std::vector<int32_t>& v, int64_t n, int64_t split) {
    TokView t;
    if (split > n) split = n;
    t.a = v.data(); t.na = split;
    t.b = v.data() + split; t.nb = n - split;
    return t;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  /* namespace */

TEST(lookup_matcher_equals_the_exhaustive_scan_on_random_text) {
    struct Cfg { int alpha, enter, stay, maxm, min_dist; };
    const Cfg cfgs[] = { {2, 8, 3, 24, 1}, {3, 8, 3, 24, 7}, {5, 8, 3, 24, 40}, {5, 4, 2, 6, 5},
                         {50, 8, 3, 24, 40}, {4, 3, 3, 24, 16}, {2, 8, 8, 24, 0}, {7, 12, 1, 24, 100} };
    std::mt19937 rng(12345);
    for (const Cfg& c : cfgs) {
        for (int trial = 0; trial < 6; ++trial) {
            const int N = 700;
            std::vector<int32_t> seq((size_t)N);
            /* random text with planted repeats, so long matches exist */
            for (int i = 0; i < N; ++i) {
                if (i > 60 && rng() % 100 < 3) {
                    const int len = 5 + (int)(rng() % 40), src = (int)(rng() % (uint32_t)(i - 20));
                    for (int j = 0; j < len && i < N; ++j, ++i) seq[(size_t)i] = seq[(size_t)(src + j)];
                    --i;
                } else {
                    seq[(size_t)i] = 100 + (int32_t)(rng() % (uint32_t)c.alpha);
                }
            }
            const LookupParams p = params(c.enter, c.stay, 3, c.maxm, c.min_dist);
            const int64_t split = (int64_t)(rng() % 300);
            LookupIndex ix;
            int bad = 0;
            for (int64_t n = 1; n <= N; ++n) {
                /* sometimes skip a search: the index must catch up */
                if (n % 5 == 0 && rng() % 3 == 0) continue;
                const TokView t = view(seq, n, split);
                int rm; int64_t rp;
                lookup_ref_best(t, p.min_match(), p.max_match, c.min_dist, &rm, &rp);
                int64_t gp = 0;
                const int gm = ix.search(t, p, c.min_dist, &gp);
                if (gm != rm || (rm > 0 && gp != rp)) {
                    if (++bad <= 3)
                        std::fprintf(stderr, "    alpha %d n %lld: index (%d, %lld) ref (%d, %lld)\n", c.alpha,
                                     (long long)n, gm, (long long)gp, rm, (long long)rp);
                }
            }
            CHECK_EQ(bad, 0);
        }
    }
}

TEST(lookup_matcher_takes_only_sources_further_back_than_the_window) {
    /* A 10-token pattern at 0..9 and again at 100..109; the sequence ends with the pattern's first 9
     * tokens at 150..158 (n = 159). Continuation of the match at p=9 starts 150 back, of the one at
     * p=109 start 50 back. */
    std::vector<int32_t> s((size_t)159, 0);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 7000 + (int32_t)(i * 13 % 211);   /* background */
    for (int j = 0; j < 9; ++j) { s[(size_t)j] = 1 + j; s[(size_t)(100 + j)] = 1 + j; s[(size_t)(150 + j)] = 1 + j; }
    s[9] = 50; s[109] = 60;
    const TokView t = view(s, 159, 20);
    int64_t p = 0;
    {   /* the window is 40: the nearer copy (continuation 50 back) is legal and more recent */
        LookupIndex ix;
        CHECK_EQ(ix.search(t, params(8, 3, 3, 24, 40), 40, &p), 9);
        CHECK_EQ((int)p, 109);
    }
    {   /* the window is 100: the nearer copy is the drafter's, the far one is taken */
        LookupIndex ix;
        CHECK_EQ(ix.search(t, params(8, 3, 3, 24, 100), 100, &p), 9);
        CHECK_EQ((int)p, 9);
    }
    {   /* the window is 200: nothing is far enough */
        LookupIndex ix;
        CHECK_EQ(ix.search(t, params(8, 3, 3, 24, 200), 200, &p), 0);
    }
    {   /* the boundary: a continuation exactly min_dist back is legal, one token closer is not */
        LookupIndex ix;
        CHECK_EQ(ix.search(t, params(8, 3, 3, 24, 150), 150, &p), 9);
        CHECK_EQ((int)p, 9);
        LookupIndex iy;
        CHECK_EQ(iy.search(t, params(8, 3, 3, 24, 151), 151, &p), 0);
    }
}

TEST(lookup_matcher_prefers_the_longer_match_then_the_most_recent) {
    /* three earlier copies of an 8-token suffix: at 20 (extended to 11 by the tokens before it), at 60 and
     * at 100 (both 8 long); the sequence ends at 160. */
    std::vector<int32_t> s((size_t)160, 0);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 9000 + (int32_t)(i * 31 % 307);
    const int32_t suf[8] = { 11, 12, 13, 14, 15, 16, 17, 18 };
    for (int j = 0; j < 8; ++j) { s[(size_t)(152 + j)] = suf[j]; s[(size_t)(12 + j)] = suf[j]; s[(size_t)(52 + j)] = suf[j];
                                  s[(size_t)(92 + j)] = suf[j]; }
    /* the match at 12..19 also agrees on the three tokens before the suffix: make them equal */
    for (int j = 1; j <= 3; ++j) { s[(size_t)(152 - j)] = 500 + j; s[(size_t)(12 - j)] = 500 + j; }
    const TokView t = view(s, 160, 0);
    int64_t p = 0;
    LookupIndex ix;
    CHECK_EQ(ix.search(t, params(8, 3, 3, 24, 30), 30, &p), 11);   /* longest wins although older */
    CHECK_EQ((int)p, 20);
    /* without the longer one the most recent of the two equal ones wins */
    s[(size_t)(12 - 1)] = 1;
    LookupIndex iy;
    CHECK_EQ(iy.search(view(s, 160, 0), params(8, 3, 3, 24, 30), 30, &p), 8);
    CHECK_EQ((int)p, 100);
    /* max_match caps the length; a capped match found first is the most recent one */
    LookupIndex iz;
    CHECK_EQ(iz.search(view(s, 160, 0), params(8, 3, 3, 8, 30), 30, &p), 8);
    CHECK_EQ((int)p, 100);
}

TEST(lookup_policy_enters_at_enter_and_stays_at_stay_only_after_a_hot_step) {
    const LookupParams p = params(8, 3, 3, 24, 2048);
    CHECK(!lookup_use(p, 0, false));
    CHECK(!lookup_use(p, 7, false));
    CHECK(lookup_use(p, 8, false));
    CHECK(!lookup_use(p, 2, true));
    CHECK(lookup_use(p, 3, true));
    CHECK(lookup_use(p, 24, true));

    LookupMode m;
    CHECK(!m.decide(p, 5));            /* a short match does not enter */
    CHECK(m.decide(p, 8));             /* a long one does */
    m.verified(p, 7);                  /* accepted 7 >= HOT: hot */
    CHECK(m.hot);
    CHECK(m.decide(p, 3));             /* ... so STAY is enough */
    m.verified(p, 2);                  /* accepted 2 < HOT: cold again */
    CHECK(!m.hot);
    CHECK(!m.decide(p, 5));            /* back to ENTER */
    CHECK(m.decide(p, 9));
    m.verified(p, 3);                  /* exactly HOT counts */
    CHECK(m.hot);
    CHECK(!m.decide(p, 2));            /* hot but no match >= STAY: drafter's proposal, mode left */
    CHECK(!m.used);
    m.verified(p, 0);                  /* a step that carried the drafter's proposal never makes it hot */
    CHECK(!m.hot);
    CHECK(!m.decide(p, 4));
    /* hot is consumed by one decision: a second one without a verify in between is cold */
    CHECK(m.decide(p, 8));
    m.verified(p, 6);
    CHECK(m.decide(p, 4));
    CHECK(!m.decide(p, 4));
}

TEST(lookup_params_from_env_validate) {
    LookupParams p = params(8, 3, 3, 24, -1);
    CHECK(p.valid());
    p.enter = 2; p.stay = 3;           /* ENTER < STAY would never enter on what STAY keeps */
    CHECK(!p.valid());
    p = params(25, 3, 3, 24, -1);
    CHECK(!p.valid());                 /* ENTER above the match cap can never be met */
    p = params(8, 3, 0, 24, -1);
    CHECK(!p.valid());
}

TEST(lookup_matcher_cost_on_the_host_at_262k) {
    /* natural-ish text: a Zipf-like token distribution (frequent short n-grams, which stress the chains),
     * a far copy planted every so often. The drafter's window is 2048. */
    const int64_t N = 262144;
    const int64_t MD = 2048;
    std::mt19937 rng(7);
    std::vector<int32_t> seq((size_t)N);
    auto zipf = [&]() {
        const double u = (double)(rng() % 1000000) / 1e6;
        return 1000 + (int32_t)(std::pow(u, 3.0) * 6000.0);
    };
    for (int64_t i = 0; i < N; ++i) {
        if (i > 10000 && rng() % 2000 == 0) {
            const int len = 30 + (int)(rng() % 200);
            const int64_t src = (int64_t)(rng() % (uint32_t)(i - 5000));
            for (int j = 0; j < len && i < N; ++j, ++i) seq[(size_t)i] = seq[(size_t)(src + j)];
            --i;
        } else {
            seq[(size_t)i] = zipf();
        }
    }
    const LookupParams p = params(8, 3, 3, 24, MD, 1024);

    /* (1) the first search over a 262k history: builds the whole index once */
    {
        LookupIndex ix;
        int64_t pos = 0;
        const double t0 = now_us();
        (void)ix.search(view(seq, N - 4, 200000), p, MD, &pos);
        const double dt = now_us() - t0;
        std::fprintf(stderr, "    first search over %lld tokens (index build): %.1f ms, index %.1f MB\n",
                     (long long)(N - 4), dt / 1e3, (double)ix.memory_bytes() / 1048576.0);
        CHECK(dt < 100e3);
    }
    /* (2) the steady state: the sequence grows by 1..8 tokens a step (a verify step commits 1 + accepted),
     * one search per step, from 254k to 262k. Three texts: the Zipf one above, a phrase-bank text (repeated
     * phrases of 3..12 tokens out of 2000: many frequent n-grams and long matches, like code or prose with
     * quotations) and uniform random tokens (nothing ever matches). */
    std::vector<int32_t> bank_text((size_t)N), uniform((size_t)N);
    {
        std::vector<std::vector<int32_t>> bank(2000);
        for (auto& ph : bank) {
            ph.resize(3 + rng() % 10);
            for (auto& t : ph) t = 1000 + (int32_t)(rng() % 20000);
        }
        for (int64_t i = 0; i < N;) {
            const auto& ph = bank[(size_t)(std::pow((double)(rng() % 100000) / 1e5, 2.0) * 1999.0)];
            for (size_t j = 0; j < ph.size() && i < N; ++j, ++i) bank_text[(size_t)i] = ph[j];
            if (rng() % 4 == 0 && i < N) bank_text[(size_t)i++] = 1000 + (int32_t)(rng() % 20000);
        }
        for (auto& t : uniform) t = 1000 + (int32_t)(rng() % 150000);
    }
    struct Text { const char* name; const std::vector<int32_t>* v; };
    for (const Text& tx : { Text{"zipf", &seq}, Text{"phrase bank", &bank_text}, Text{"uniform", &uniform} }) {
        LookupIndex ix;
        int64_t pos = 0;
        int64_t n = N - 8192;
        (void)ix.search(view(*tx.v, n, 4000), p, MD, &pos);
        std::vector<double> us;
        double vis = 0;
        int hits = 0;
        while (n + 8 <= N) {
            n += 1 + (int64_t)(rng() % 8);
            const double t0 = now_us();
            const int m = ix.search(view(*tx.v, n, 4000), p, MD, &pos);
            us.push_back(now_us() - t0);
            vis += ix.last_visited();
            hits += m > 0;
        }
        std::sort(us.begin(), us.end());
        double mean = 0;
        for (double x : us) mean += x;
        mean /= (double)us.size();
        std::fprintf(stderr, "    steady search at ~254k..262k, %-11s: mean %.2f us, p50 %.2f, p99 %.2f, max %.2f us "
                             "(%zu steps, %.1f candidates a search, %d matches)\n", tx.name, mean, us[us.size() / 2],
                     us[us.size() * 99 / 100], us.back(), us.size(), vis / (double)us.size(), hits);
        CHECK(mean < 200.0);
        CHECK(us.back() < 20000.0);
    }
    /* (3) worst cases for a chain: one token repeated, and a period-2 text, each at 262k */
    for (int kind = 0; kind < 2; ++kind) {
        std::vector<int32_t> w((size_t)N);
        for (int64_t i = 0; i < N; ++i) w[(size_t)i] = kind == 0 ? 5 : 5 + (int32_t)(i & 1);
        LookupIndex ix;
        int64_t pos = 0;
        int64_t n = N - 4096;
        (void)ix.search(view(w, n, 100), p, MD, &pos);
        double worst = 0, sum = 0;
        int cnt = 0;
        while (n + 8 <= N) {
            n += 4;
            const double t0 = now_us();
            (void)ix.search(view(w, n, 100), p, MD, &pos);
            const double dt = now_us() - t0;
            worst = std::max(worst, dt); sum += dt; ++cnt;
        }
        std::fprintf(stderr, "    degenerate text (%s): mean %.2f us, max %.2f us\n",
                     kind == 0 ? "one token" : "period 2", sum / cnt, worst);
        CHECK(sum / cnt < 200.0);
    }
    /* (4) a sequence that is no longer than the window costs nothing: the index is never built */
    {
        LookupIndex ix;
        int64_t pos = 0;
        const double t0 = now_us();
        for (int64_t n = 100; n < MD; n += 3) (void)ix.search(view(seq, n, 50), p, MD, &pos);
        std::fprintf(stderr, "    %d searches under the window: %.1f us total, indexed %lld\n", (int)((MD - 100) / 3),
                     now_us() - t0, (long long)ix.indexed());
        CHECK(ix.empty());
    }
}

RAD_TEST_MAIN()
