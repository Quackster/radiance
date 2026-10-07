/* r4d_bands_test.cpp -- libr4d's gemm_nt_q rows against core/plugin/bands.cpp's rules, on the host.
 *
 * The real row table is read out of libr4d.so with dlopen/dlsym, exactly as rad-schemas does: no
 * device is opened and nothing is launched. The band table is rebuilt here from the same rule
 * bands.cpp states (boundaries = LE / GE-1 / EQ-1,EQ of EVERY row of the op, each band resolved at
 * its upper bound) and rows are matched with the core's own constraint_holds, highest priority
 * first, ties to the earlier row.
 *
 * WHY: constraints of one row are AND-ed, so a band edge placed on a real row narrows that row. An
 * `M <= 16` on the decode row would make it serve M <= 16 only and push M 17..64 onto the tiled
 * kernel (a 4-5x slower GEMM). The edges now sit on gemm_nt_q_bands, a row that matches no tensor.
 * This test holds both halves of that: the band list, and that our decode row is what every band up
 * to 64 resolves to and our tiled row what every band above it resolves to.
 *
 * THE RUNTIME SWITCH. The native decode/tiled rows (gemm_mxfp4a8_decode / gemm_mxfp4a8_tiled) are
 * the "tuned bands": they carry the ks/bk/nt and tn/dec tune axes. RADIANCE_MX_TUNE=1 raises them at
 * row registration, so one binary serves both and the same test is run twice for an A/B:
 *
 *     ./r4d_bands_test                                  # default: our rows win
 *     RADIANCE_MX_TUNE=1 ./r4d_bands_test               # tuned: the ported PR rows win
 *
 * The .so reads the environment when the row table is first built, which is the dlopen this test
 * performs, so the expected row names follow the environment. Unset/0 is byte-for-byte today's
 * behaviour and is what ctest runs.
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

/* The tuned rows this switch selects, and the default rows they stand against. `hyb` is the
 * M-banded decode hybrid's low bound: when PR decode is selected the PR decode row is narrowed to
 * M >= hyb+1 (RADIANCE_MX_HYBRID_M, default 16), so bands at or below `hyb` resolve to our own
 * gemm_mxfp4a8_nt_m64 and every band above still resolves to the PR decode row. */
struct Names {
    const char* decode;
    const char* tiled;
    bool tuned;
    long long hyb;
};

Names expected_names() {
    const char* t = std::getenv("RADIANCE_MX_TUNE");
    const char* e = std::getenv("RADIANCE_MX_GEMM");
    const char* h = std::getenv("RADIANCE_MX_HYBRID_M");
    const bool tune = t && t[0] && std::strcmp(t, "0") != 0;
    const bool pr_dec = tune || (e && e[0] && (std::strstr(e, "pr_decode") || std::strstr(e, "ablk")));
    const bool pr_til = tune || (e && e[0] && (std::strstr(e, "pr_tiled")  || std::strstr(e, "ablk")));
    long long hyb = pr_dec ? 16 : 0;
    if (h && h[0]) hyb = std::strtoll(h, nullptr, 10);
    if (!pr_dec || hyb <= 0 || hyb >= 128) hyb = 0;
    /* Upstream's native decode row is the default for the whole M <= 128 band (its MXDEC priority
     * outranks the skinny row), so both switches leave the same row names -- they only move the
     * priority. The M-banded hybrid, when pr_dec is set, narrows the decode row to M >= hyb+1 so
     * the low bands fall back to our gemm_mxfp4a8_nt_m64. */
    return Names{ "gemm_mxfp4a8_decode", "gemm_mxfp4a8_tiled", pr_dec || pr_til, hyb };
}

struct Rows {
    void* h = nullptr;
    std::vector<const RadKernelInfo*> gemm;     /* op gemm_nt_q, declaration order */
};

bool open_rows(Rows* out) {
    std::string path;
    if (const char* p = std::getenv("RAD_R4D_SO")) path = p;
    else {
        const char* h = std::getenv("RADIANCE_HOME");
        path = std::string(h ? h : "radiance_home") + "/kernels/libr4d.so";
    }
    if (access(path.c_str(), R_OK) != 0) {
        fprintf(stderr, "  SKIP no libr4d.so at %s\n", path.c_str());
        return false;
    }
    out->h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!out->h) { fprintf(stderr, "  SKIP dlopen(%s): %s\n", path.c_str(), dlerror()); return false; }
    auto cnt = (int (*)(void))dlsym(out->h, "rad_kernel_count");
    auto at  = (const RadKernelInfo* (*)(int))dlsym(out->h, "rad_kernel_at");
    if (!cnt || !at) { fprintf(stderr, "  SKIP %s is not a kernel plugin\n", path.c_str()); return false; }
    for (int i = 0, n = cnt(); i < n; ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::string(k->op) == "gemm_nt_q" && k->domain == RAD_DOMAIN_DEVICE)
            out->gemm.push_back(k);
    }
    return true;
}

const RadKernelInfo* by_name(const Rows& r, const char* name) {
    for (const RadKernelInfo* k : r.gemm) if (std::string(k->name) == name) return k;
    return nullptr;
}

/* bands.cpp's boundary rule, over [lo, hi]. */
std::vector<long long> edges(const Rows& r, long long lo, long long hi) {
    std::vector<long long> b;
    auto add = [&](long long v) { if (v >= lo && v < hi) b.push_back(v); };
    for (const RadKernelInfo* k : r.gemm)
        for (int i = 0; i < k->n_constraints; ++i) {
            const RadConstraint& c = k->constraints[i];
            if (!c.key || std::string(c.key) != "M") continue;
            if (c.op == RAD_C_LE) add(c.ival);
            else if (c.op == RAD_C_GE) add(c.ival - 1);
            else if (c.op == RAD_C_EQ) { add(c.ival - 1); add(c.ival); }
        }
    std::sort(b.begin(), b.end());
    b.erase(std::unique(b.begin(), b.end()), b.end());
    b.push_back(hi);
    return b;
}

/* select.cpp's rule: every constraint holds, highest priority wins, a tie goes to the earlier row. */
const RadKernelInfo* resolve(const Rows& r, const Geometry& g) {
    const RadKernelInfo* best = nullptr;
    for (const RadKernelInfo* k : r.gemm) {
        bool ok = true;
        for (int i = 0; i < k->n_constraints && ok; ++i) ok = constraint_holds(k->constraints[i], g);
        if (ok && (!best || k->priority > best->priority)) best = k;
    }
    return best;
}

Geometry geom(const char* dtype, long long M) {
    Geometry g;
    g.set_i("M", M); g.set_i("N", 4096); g.set_i("K", 4096); g.set_s("dtype", dtype);
    return g;
}

}  /* namespace */

TEST(gemm_nt_q_band_list) {
    Rows r;
    if (!open_rows(&r)) return;
    /* The union of every gemm_nt_q row's LE / GE-1 / EQ-1,EQ on M, plus the gemm_nt_q_bands edges:
     * the ordinary rows already say 8 (fp8a16 m16), 16 (bf16 m16) and 64 (every decode row), and the
     * bands row adds 32, 48, 128, 256, 512, 1024 and 2048. */
    const std::vector<long long> want = { 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 8192 };
    CHECK(edges(r, 1, 8192) == want);
}

TEST(mxfp4a8_rows_serve_their_bands) {
    Rows r;
    if (!open_rows(&r)) return;
    const Names n = expected_names();
    printf("mode: %s (decode=%s tiled=%s)\n", n.tuned ? "RADIANCE_MX_TUNE" : "default",
           n.decode, n.tiled);
    long long prev = 0;
    for (long long hi : edges(r, 1, 8192)) {
        const RadKernelInfo* k = resolve(r, geom("mxfp4a8", hi));
        const std::string name = k ? k->name : "(none)";
        /* one line per band, for tools/scripts that want to show it (a script can grep "^band ") */
        printf("band M in (%lld, %lld] -> %s\n", prev, hi, name.c_str());
        prev = hi;
        /* The M-banded hybrid: at or below `hyb` the PR decode row is excluded and ours wins. */
        const char* dec_hi = (n.hyb > 0 && hi <= n.hyb) ? "gemm_mxfp4a8_nt_m64" : n.decode;
        if (hi <= 128) CHECK_EQ(name, dec_hi);
        else          CHECK_EQ(name, n.tiled);
    }
    /* and every single M in between, not only the band tops */
    for (long long m = 1; m <= 128; ++m) {
        const char* dec_m = (n.hyb > 0 && m <= n.hyb) ? "gemm_mxfp4a8_nt_m64" : n.decode;
        CHECK_EQ(std::string(resolve(r, geom("mxfp4a8", m))->name), dec_m);
    }
}

/* A row that exists only to carry band edges must be unselectable for every dtype any tensor has,
 * and must never outrank anything even if it were reached. */
TEST(band_edge_row_is_inert) {
    Rows r;
    if (!open_rows(&r)) return;
    const RadKernelInfo* bands = by_name(r, "gemm_nt_q_bands");
    REQUIRE(bands != nullptr);
    CHECK(bands->launch != nullptr);
    CHECK_EQ(bands->priority, 0);
    for (const RadKernelInfo* k : r.gemm) CHECK(k == bands || k->priority > bands->priority);
    for (const char* dt : { "mxfp4a8", "mxfp8a8", "mxfp6a8", "w4a16", "w8a16", "w4a8",
                            "w4a8_asym", "w8a8", "w5a8", "w2a8", "bf16", "fp8a16", "fp8a8",
                            "i8a8", "f16", "int8", "" })
        for (long long m : { 1LL, 16LL, 48LL, 64LL, 2048LL, 8192LL }) {
            const RadKernelInfo* k = resolve(r, geom(dt, m));
            CHECK(!k || std::string(k->name) != "gemm_nt_q_bands");
        }
}

/* The wide decode form (dec=1) reads a row-major A. Where the producer can write A fragment-tiled
 * (afrag = T <= the band's upper bound M) the tiled row's tune_valid must refuse it, so a dec=1 tune
 * line cannot meet the atiled kernel's layout. Our gemm_nt_q schema does not declare `afrag` (the
 * fragment-tiled A producers are not ported), but tune_valid reads it straight off the geometry, so
 * the guard is exercised here even though no op in this tree supplies it. Both tiled rows carry the
 * same test; the tuned row is checked when RADIANCE_MX_TUNE is on. */
TEST(tiled_row_refuses_wide_form_with_fragment_plane) {
    Rows r;
    if (!open_rows(&r)) return;
    const char* name = expected_names().tiled;
    const RadKernelInfo* tiled = by_name(r, name);
    REQUIRE(tiled != nullptr);
    REQUIRE(tiled->tune_valid != nullptr);
    auto legal = [&](long long M, long long afrag, long long dec) {
        Geometry g = geom("mxfp4a8", M);
        if (afrag) g.set_i("afrag", afrag);
        RadParam c{};
        c.key = "dec"; c.kind = RAD_P_INT; c.ival = dec;
        return tiled->tune_valid(g.params(), g.n_params(), &c, 1) == 1;
    };
    CHECK(legal(128, 0, 1));        /* no fragment plane: the wide form is fine */
    CHECK(legal(128, 513, 1));      /* T above the band: no M of it is tiled */
    CHECK(!legal(128, 100, 1));     /* band reaches T */
    CHECK(!legal(128, 128, 1));     /* T == upper bound */
    CHECK(legal(64, 200, 1));       /* band is below T */
    CHECK(!legal(256, 0, 1));       /* wide form stops at 128 (the native decode's band) */
    CHECK(legal(128, 100, 0));      /* dec=0 is the tiled kernel, whatever T */
    CHECK(legal(2048, 513, 0));
}

RAD_TEST_MAIN()
