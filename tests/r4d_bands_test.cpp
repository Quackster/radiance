/* r4d_bands_test.cpp -- libr4d's gemm_nt_q rows against core/plugin/bands.cpp's rules, on the host.
 *
 * The real row table is read out of libr4d.so with dlopen/dlsym, exactly as rad-schemas does: no
 * device is opened and nothing is launched. The band table is rebuilt here from the same rule
 * bands.cpp states (boundaries = LE / GE-1 / EQ-1,EQ of EVERY row of the op, each band resolved at
 * its upper bound) and rows are matched with the core's own constraint_holds, highest priority
 * first, ties to the earlier row.
 *
 * WHY: constraints of one row are AND-ed, so a band edge placed on a real row narrows that row. An
 * `M <= 16` on gemm_mxfp4a8_decode made it serve M <= 16 only, and M 17..64 fell to the tiled
 * kernel (GEMM 3.5x slower, c8 385 -> 177 t/s). The edges now sit on gemm_nt_q_bands, a row that
 * matches no tensor. This test holds both halves of that: the band list, and that the decode row is
 * what every band up to 64 resolves to.
 */
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

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
    const std::vector<long long> want = { 8, 16, 32, 48, 64, 128, 256, 512, 1024, 2048, 8192 };
    CHECK(edges(r, 1, 8192) == want);
}

TEST(mxfp4a8_decode_row_serves_every_band_up_to_64) {
    Rows r;
    if (!open_rows(&r)) return;
    long long prev = 0;
    for (long long hi : edges(r, 1, 8192)) {
        const RadKernelInfo* k = resolve(r, geom("mxfp4a8", hi));
        const std::string name = k ? k->name : "(none)";
        /* one line per band, for tools/scripts that want to show it (a script can grep "^band ") */
        printf("band M in (%lld, %lld] -> %s\n", prev, hi, name.c_str());
        prev = hi;
        if (hi <= 64) CHECK_EQ(name, "gemm_mxfp4a8_decode");
        else          CHECK_EQ(name, "gemm_mxfp4a8_tiled");
    }
    /* and every single M in between, not only the band tops */
    for (long long m = 1; m <= 64; ++m) {
        const RadKernelInfo* k = resolve(r, geom("mxfp4a8", m));
        CHECK(k && std::string(k->name) == "gemm_mxfp4a8_decode");
    }
}

/* A row that exists only to carry band edges must be unselectable for every dtype any tensor has,
 * and must never outrank anything even if it were reached. */
TEST(band_edge_row_is_inert) {
    Rows r;
    if (!open_rows(&r)) return;
    const RadKernelInfo* bands = nullptr;
    for (const RadKernelInfo* k : r.gemm)
        if (std::string(k->name) == "gemm_nt_q_bands") bands = k;
    REQUIRE(bands != nullptr);
    CHECK(bands->launch != nullptr);
    CHECK_EQ(bands->priority, 0);
    for (const RadKernelInfo* k : r.gemm) CHECK(k == bands || k->priority > bands->priority);
    for (const char* dt : { "mxfp4a8", "w4a16", "w8a16", "w4a8", "w4a8_asym", "w8a8", "bf16", "fp8",
                            "f16", "w2a8", "int8", "" })
        for (long long m : { 1LL, 16LL, 48LL, 64LL, 2048LL, 8192LL}) {
            const RadKernelInfo* k = resolve(r, geom(dt, m));
            CHECK(!k || std::string(k->name) != "gemm_nt_q_bands");
        }
}

/* The wide decode form (dec=1) reads a row-major A. Where the producer can write A fragment-tiled
 * (afrag = T <= the band's upper bound M) the tiled row's tune_valid must refuse it, so a dec=1 tune
 * line cannot meet the atiled kernel's layout. */
TEST(tiled_row_refuses_wide_form_with_fragment_plane) {
    Rows r;
    if (!open_rows(&r)) return;
    const RadKernelInfo* tiled = nullptr;
    for (const RadKernelInfo* k : r.gemm)
        if (std::string(k->name) == "gemm_mxfp4a8_tiled") tiled = k;
    REQUIRE(tiled != nullptr);
    REQUIRE(tiled->tune_valid != nullptr);
    auto legal = [&](long long M, long long afrag, long long dec) {
        Geometry g = geom("mxfp4a8", M);
        if (afrag) g.set_i("afrag", afrag);
        RadParam c{};
        c.key = "dec"; c.kind = RAD_P_INT; c.ival = dec;
        return tiled->tune_valid(g.params(), g.n_params(), &c, 1) == 1;
    };
    CHECK(legal(256, 0, 1));        /* no fragment plane: the wide form is fine */
    CHECK(legal(256, 513, 1));      /* T above the band: no M of it is tiled */
    CHECK(!legal(256, 200, 1));     /* band reaches T */
    CHECK(!legal(256, 256, 1));     /* T == upper bound */
    CHECK(legal(128, 200, 1));      /* band (64,128] is below T */
    CHECK(!legal(512, 0, 1));       /* wide form stops at 256 as before */
    CHECK(legal(256, 200, 0));      /* dec=0 is the tiled kernel, whatever T */
    CHECK(legal(2048, 513, 0));
}

RAD_TEST_MAIN()
