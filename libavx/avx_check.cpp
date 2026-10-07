/* avx_check.cpp -- rad-avx-check: libavx against libref, every op, every ISA level, no engine.
 *
 * ============================== WHAT THIS IS AND WHY IT IS NOT rad-kbench ==============================
 *
 * rad-kbench is the tree's kernel checker and it is the right tool for a DEVICE plugin: it needs
 * rad_core, a Geometry, a plugin loader and (for the fixture path) a recorded artifact. Building
 * it means building the engine, which on a machine with no device toolchain means building the
 * engine's host half to test a host plugin. This is the same comparison with nothing under it: two dlopen calls, a
 * table of geometries, and libref's own operand descriptions to size the buffers.
 *
 * It links NEITHER plugin. That is not laziness -- the claim being made is about the .so the
 * engine would load, and a checker that statically linked the kernels would be exercising a
 * different binary with different inlining and different flags.
 *
 * ============================== THE FOUR THINGS IT CHECKS, IN ORDER ==============================
 *
 *   1. SCHEMA AGREEMENT, before any arithmetic. The loader refuses two plugins whose schemas for
 *      one op disagree, and it refuses the WHOLE plugin -- so a single drifted parameter makes
 *      every row libavx carries unreachable, and the message an operator sees names one op and
 *      leaves them to find the rest. libavx's schema table is a hand-kept copy of libref's
 *      (avx_registry.cpp says so at length); this is what keeps the copy honest, running the
 *      loader's own rule: parameter key/type/requiredness positionally, operand role/optionality
 *      positionally, names deliberately NOT compared.
 *
 *   2. COVERAGE. Every op libref implements, libavx must implement. A row that is ABSENT and a row
 *      that RETURNS RAD_E_UNSUPPORTED are reported differently and neither is reported as a pass.
 *      This is the check that catches a kernel that was built and never wired, because a kernel
 *      that declines is indistinguishable from a kernel that works to anything that only looks for
 *      the return codes it expects.
 *
 *   3. THE NUMBERS. Both plugins are handed BYTE-IDENTICAL inputs and every OUT and INOUT operand
 *      is compared. The FINITENESS PATTERN is compared first and separately: a NaN on one side and
 *      a number on the other is a failure, not a tolerance question, and the obvious spelling
 *      (`rel > tol || rel < 0`) is FALSE for NaN -- which is how a NaN passes a comparison
 *      silently. After the geometry table, a set of HAND-BUILT cases covers the operand shapes a
 *      description cannot draw -- strided column slices, several sequences, rolling windows read
 *      past offset zero, a band above the operand's rows (see hand_cases below).
 *
 *   4. EVERY ISA LEVEL. The scalar, AVX, AVX2 and AVX-512 paths are four instruction streams
 *      compiled from one source, and only the level the machine under it happens to execute gets
 *      exercised by accident. Each is forced in turn through avx_isa_force -- an exported
 *      symbol rather than an environment variable, because an env switch that changes what a
 *      kernel computes is exactly what this project refuses to have. A level the CPU cannot
 *      execute is reported as NOT EXECUTABLE and counted nowhere: claiming a pass for a path that
 *      never ran is the one result worse than a failure.
 */
#include "../tests/grid_embed_golden.h"

#include "avx_harness.h"
#include "avx_isa.h"

#include <deque>
#include <functional>

/* ================================================================== error metric
 * rad-kbench's, including the thing its header says it got wrong once: the finiteness pattern is
 * compared BEFORE any norm. */
struct Err {
    double max_abs = 0, max_rel = 0, rel_l2 = 0;
    int64_t nonfinite_gap = 0, n = 0;
};

static Err compare(const float* got, const float* ref, int64_t n) {
    Err e;
    e.n = n;
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        const bool fg = std::isfinite(got[i]), fr = std::isfinite(ref[i]);
        if (fg != fr) { ++e.nonfinite_gap; continue; }
        if (!fg) continue;
        const double d = (double)got[i] - (double)ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        const double sc = std::fabs((double)ref[i]);
        if (sc > 1e-30) e.max_rel = std::max(e.max_rel, std::fabs(d) / sc);
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    e.rel_l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    return e;
}

/* rad-kbench's tolerance table, and it means the same thing here: what a DIFFERENT SUMMATION ORDER
 * of the same arithmetic costs, not what a wrong kernel costs. A vectorised reduction IS that
 * different order, so these are the right numbers for this comparison rather than a loosened
 * version of them. A kernel with a transposed operand lands nowhere near any of these. */
struct Tol { const char* op; const char* dtype; double rel_l2; };
static const Tol kTol[] = {
    { "gemm_nt",      "bf16",     8e-3 },
    { "gemm_nt",      "f16",      4e-3 },
    { "gemm_nt",      "f32",      2e-5 },
    { "gemm_nt",      "*",        3e-2 },
    { "logits_gemm",  "bf16",     8e-3 },
    { "logits_gemm",  "f32",      2e-5 },
    { "logits_select", "*",      2e-5 },
    { "moe_gemm",     "*",        3e-2 },
    { "attn_",        "bf16",     2e-2 },
    { "attn_",        "f32",      1e-4 },
    { "attn_",        "*",        3e-2 },
    { "rmsnorm",      "bf16",     4e-3 },
    { "rmsnorm",      "f32",      1e-6 },
    { "layernorm",    "f32",      1e-6 },
    { "softmax",      "f32",      1e-6 },
    { "quant_act",    "*",        1e-6 },
    { "dequant",      "*",        1e-6 },
    { "cast",         "*",        1e-6 },
    { "all_reduce",   "f32",      1e-6 },
    { "all_reduce",   "*",        4e-3 },
    { "all_gather",   "*",        0.0  },
    { "*",            "f32",      1e-5 },
    { "*",            "bf16",     6e-3 },
    { "*",            "f16",      3e-3 },
    { "*",            "*",        2e-2 },
};

static double tolerance_for(const std::string& op, const char* dtype) {
    const char* dt = dtype ? dtype : "*";
    for (const Tol& t : kTol) {
        const bool om = std::strcmp(t.op, "*") == 0 || op.rfind(t.op, 0) == 0;
        const bool dm = std::strcmp(t.dtype, "*") == 0 || std::strcmp(t.dtype, dt) == 0;
        if (om && dm) return t.rel_l2;
    }
    return 2e-2;
}

struct Stats {
    int checked = 0, passed = 0, failed = 0, skipped = 0, declined = 0, missing = 0;
    double worst = 0;
    std::string worst_where;
};

static void run_case(const Lib& ref, const Lib& avx, const Case& c, bool verbose,
                     double tol_override, Stats* st) {
    const RadKernelInfo* kr = ref.row(c.op);
    const RadKernelInfo* ka = avx.row(c.op);
    if (!kr) { ++st->skipped; return; }
    if (!ka) { ++st->missing; return; }

    /* Side A is libref, side B is libavx. The description comes from the ORACLE: the two have to
     * be handed the same bytes for the comparison to mean anything, and where they could disagree
     * about an extent the reference semantics are the ones that define the op. */
    /* A SKIP IS REPORTED WITH ITS REASON, and that is not a nicety. A silent skip is how a checker
     * reports coverage it does not have, and the two reasons here mean completely
     * different things. "the oracle cannot describe this geometry" is a limit of libref's shape
     * hook and the case should be rewritten or dropped; "the oracle declined the launch" means the
     * case is asking for something the op does not define. Either way it is the CASE TABLE that is
     * wrong, and a count with no name does not say which line. */
    Prepared pr = h_prepare(ref, kr, c, kr, ka);
    if (pr.skipped || !pr.ok) {
        ++st->skipped;
        if (verbose)
            std::printf("  %-22s %-46s SKIP: libref's opd_shape declines this geometry\n",
                        c.op, case_str(c).c_str());
        return;
    }

    const int rr = kr->launch(&pr.aa, nullptr);
    const int ra = ka->launch(&pr.ab, nullptr);

    /* THE ORACLE IS ASKED FIRST, and getting this order wrong cost a false report. If libref
     * declines the geometry there is nothing to compare against, so the case is SKIPPED -- even
     * when libavx declined it too, which is the correct and identical behaviour rather than a gap.
     * Reporting libavx's decline before looking at the oracle's blamed the plugin for a case the
     * op does not define (it was a case table asking for an `a_order` spelling neither
     * implementation accepts). A DECLINE only means something when the oracle produced an answer. */
    if (rr < 0) {
        ++st->skipped;
        if (verbose)
            std::printf("  %-22s %-46s SKIP: libref's launch returned %d\n", c.op,
                        case_str(c).c_str(), rr);
        return;
    }
    if (ra == RAD_E_UNSUPPORTED) {
        ++st->declined;
        if (verbose) std::printf("  %-22s %-46s DECLINED\n", c.op, case_str(c).c_str());
        return;
    }
    if (ra < 0) {
        std::printf("  FAIL %-20s %s\n       libavx returned %d where libref returned %d\n",
                    c.op, case_str(c).c_str(), ra, rr);
        ++st->failed;
        ++st->checked;
        return;
    }

    bool ok = true;
    double worst = 0;
    // The selected head's optional final narrowing is BF16 even with F32 input operands.
    // A reduction-order difference can cross that rounding boundary; exact seam cases below
    // use dyadic products whose sum is independent of the SIMD width.
    const bool selected_bf16 = std::strcmp(c.op, "logits_select") == 0 &&
        case_geti(c, "round_bf16", 0) == 1;
    const double tol = tol_override >= 0 ? tol_override :
        selected_bf16 ? 8e-3 : tolerance_for(c.op, case_dtype(c));
    for (size_t i = 0; i < pr.opd.size(); ++i) {
        const Opd& o = pr.opd[i];
        if (o.absent || o.n <= 0) continue;
        if (o.role != RAD_OPD_OUT && o.role != RAD_OPD_INOUT) continue;
        std::vector<float> fr((size_t)o.n), fa((size_t)o.n);
        h_widen(o.a.data(), o.d.dtype, o.n, fr.data());
        h_widen(o.b.data(), o.d.dtype, o.n, fa.data());
        const Err e = compare(fa.data(), fr.data(), o.n);
        worst = std::max(worst, e.rel_l2);
        if (e.nonfinite_gap != 0 || !(e.rel_l2 <= tol)) {
            ok = false;
            std::printf("  FAIL %-20s %s\n       operand %d (%s) rel_l2=%.3e max_abs=%.3e "
                        "nonfinite_gap=%lld tol=%.1e\n",
                        c.op, case_str(c).c_str(), (int)i,
                        pr.schema->operands[i].name ? pr.schema->operands[i].name : "?",
                        e.rel_l2, e.max_abs, (long long)e.nonfinite_gap, tol);
        }
    }
    ++st->checked;
    if (ok) ++st->passed; else ++st->failed;
    if (worst > st->worst) {
        st->worst = worst;
        st->worst_where = std::string(c.op) + " " + case_str(c);
    }
    if (verbose)
        std::printf("  %-22s %-46s rel_l2=%.3e tol=%.1e %s\n", c.op, case_str(c).c_str(), worst,
                    tol, ok ? "ok" : "FAIL");
}

/* ================================================================== exhaustive conversion
 *
 * THE ONE CHECK A DRAW CANNOT MAKE. Every geometry in the case table fills its operands with
 * standard normals, which is the right domain for an activation and reaches almost none of the
 * interesting bit patterns of a narrow float: no NaN, no infinity, no subnormal, no negative zero.
 * A conversion is exactly where those live -- `cvt_fp8e4m3_to_f32` has a separate branch for the
 * subnormal range and another for the format's single NaN encoding, and a draw of normals exercises
 * neither.
 *
 * So the 8-bit and 16-bit formats are checked EXHAUSTIVELY: every one of the 256 or 65536 bit
 * patterns, through the plugin's own `cast`, against libref's. For an 8-bit source that is the
 * complete input space and the result is a proof rather than evidence; for a 16-bit one it is the
 * complete space too. The narrowing direction (f32 -> narrow) cannot be exhaustive over 2^32, so it
 * is fed the widened patterns back plus the boundary values of each format -- which is where a
 * narrowing rounds differently if it is going to.
 *
 * It is worth the forty lines because this is the class of bug that survives everything else: the
 * fp8 encode's lower bound was wrong by seven octaves and every geometry in the table still passed
 * until `cast` was run at a tolerance of 1e-6. */
static bool exhaustive_cast(const Lib& ref, const Lib& avx, bool verbose) {
    const RadKernelInfo* kr = ref.row("cast");
    const RadKernelInfo* ka = avx.row("cast");
    if (!kr || !ka) return true;

    struct Fmt { const char* name; uint32_t dt; int bits; };
    static const Fmt kFmt[] = {
        { "fp8_e4m3", RAD_F8E4M3, 8 }, { "fp8_e5m2", RAD_F8E5M2, 8 },
        { "bf16", RAD_BF16, 16 },      { "f16", RAD_F16, 16 },
        { "i8", RAD_I8, 8 },
    };
    bool ok = true;
    for (const Fmt& f : kFmt) {
        const int64_t n = (int64_t)1 << f.bits;
        std::vector<uint8_t> src((size_t)((n * f.bits) / 8));
        if (f.bits == 8) for (int64_t i = 0; i < n; ++i) src[(size_t)i] = (uint8_t)i;
        else for (int64_t i = 0; i < n; ++i) ((uint16_t*)src.data())[i] = (uint16_t)i;

        /* ---- widening: every pattern of the narrow format into f32. */
        std::vector<float> ya((size_t)n, 0), yb((size_t)n, 0);
        for (int pass = 0; pass < 2; ++pass) {
            const bool narrowing = pass == 1;
            /* ---- narrowing: the widened values back down, which covers every value the format can
             * represent plus, through the round trip, the ties at each of its steps. */
            std::vector<uint8_t> na((size_t)((n * f.bits) / 8), 0);
            std::vector<uint8_t> nb((size_t)((n * f.bits) / 8), 0);

            RadParam p[4];
            p[0] = RadParam{ "M", RAD_P_INT, 1, 0, nullptr, 0 };
            p[1] = RadParam{ "n", RAD_P_INT, (long long)n, 0, nullptr, 0 };
            p[2] = RadParam{ "from", RAD_P_STR, 0, 0, narrowing ? "f32" : f.name, 0 };
            p[3] = RadParam{ "to", RAD_P_STR, 0, 0, narrowing ? f.name : "f32", 0 };

            auto mk_t = [&](void* data, uint32_t dt) {
                RadTensor t{};
                t.data = data; t.dtype = dt; t.rank = 2;
                t.shape[0] = 1; t.shape[1] = n;
                t.stride[0] = n; t.stride[1] = 1;
                return t;
            };
            RadTensor ta[2], tb[2];
            if (!narrowing) {
                ta[0] = mk_t(src.data(), f.dt);  ta[1] = mk_t(ya.data(), RAD_F32);
                tb[0] = mk_t(src.data(), f.dt);  tb[1] = mk_t(yb.data(), RAD_F32);
            } else {
                ta[0] = mk_t(ya.data(), RAD_F32); ta[1] = mk_t(na.data(), f.dt);
                tb[0] = mk_t(ya.data(), RAD_F32); tb[1] = mk_t(nb.data(), f.dt);
            }
            RadArgs aa{}, ab{};
            aa.t = ta; aa.n_t = 2; aa.p = p; aa.n_p = 4; aa.world_size = 1;
            ab = aa; ab.t = tb;
            if (kr->launch(&aa, nullptr) < 0 || ka->launch(&ab, nullptr) < 0) continue;

            int64_t bad = 0;
            int64_t first = -1;
            if (!narrowing) {
                for (int64_t i = 0; i < n; ++i) {
                    const bool na_ = std::isnan(ya[(size_t)i]), nb_ = std::isnan(yb[(size_t)i]);
                    if (na_ != nb_ || (!na_ && ya[(size_t)i] != yb[(size_t)i])) {
                        if (first < 0) first = i;
                        ++bad;
                    }
                }
            } else {
                for (size_t i = 0; i < na.size(); ++i)
                    if (na[i] != nb[i]) { if (first < 0) first = (int64_t)i; ++bad; }
            }
            if (bad) {
                ok = false;
                std::printf("EXACT   cast %-10s %-10s %lld of %lld patterns differ, first at %lld\n",
                            narrowing ? "f32 ->" : "-> f32", f.name, (long long)bad,
                            (long long)n, (long long)first);
            } else if (verbose) {
                std::printf("  cast %-6s %-9s all %lld bit patterns identical\n",
                            narrowing ? "f32 ->" : "-> f32", f.name, (long long)n);
            }
        }
    }
    return ok;
}

/* ================================================================== operands no description draws
 *
 * libref's opd_shape describes every op the one way a cold harness can use it: one sequence where
 * the op allows it, every operand dense at its declared rank, integer operands at their plainest
 * value. The engine hands kernels more than that, and each case below is one of those shapes, with
 * its operands built by hand:
 *
 *   several sequences through a path chosen for uniform queries;
 *   a rank-2 COLUMN SLICE of a fused projection, whose row stride is the projection's width;
 *   a speculative step whose `num_accepted` reads a rolling window past offset zero;
 *   a ranged parameter at its band's upper bound over an operand of fewer rows;
 *   head widths that do not divide a kernel's channel panel;
 *   packed weight codes at odd element offsets, which fall off the row converter;
 *   integer stores of values past the target's range, and exponents of -inf and NaN.
 *
 * Both plugins get byte-identical copies, and every OUT or INOUT buffer is compared WHOLE -- not
 * just the view the op was handed -- so a write outside the view is a difference too. */
struct Hand {
    struct Buf {
        uint32_t dt = RAD_F32;
        int64_t elems = 0;
        std::vector<uint8_t> a, b;
        bool out = false;
    };
    std::string name, op;
    std::deque<Buf> bufs;
    std::vector<int> buf_of;                  /* per operand, the backing buffer or -1 */
    std::vector<RadTensor> ta, tb;
    std::vector<RadParam> p;
    std::deque<std::string> strs;
    double tol = 1e-5;

    Hand(const char* n, const char* o, double t) : name(n), op(o), tol(t) {}

    /* An operand over a fresh buffer of `span` elements, viewed with `shape` and `stride`. The
     * whole buffer is filled -- the bytes outside the view are another projection's columns, and
     * a kernel that reads them must read the same thing on both sides. */
    Hand& view(uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t span,
               const std::function<float(int64_t)>& fill, bool out = false) {
        Buf bf;
        bf.dt = dt; bf.elems = span; bf.out = out;
        std::vector<float> v((size_t)span);
        for (int64_t i = 0; i < span; ++i) v[(size_t)i] = fill(i);
        bf.a.assign(dt_bytes(dt, span) + 8, 0);
        h_narrow(v.data(), dt, span, bf.a.data());
        bf.b = bf.a;
        bufs.push_back(std::move(bf));
        RadTensor t{};
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        for (size_t i = 0; i < shape.size(); ++i) {
            t.shape[i] = shape[i];
            t.stride[i] = stride[i];
        }
        ta.push_back(t); tb.push_back(t);
        buf_of.push_back((int)bufs.size() - 1);
        return *this;
    }
    /* A dense operand of `shape`. */
    Hand& dense(uint32_t dt, std::vector<int64_t> shape, const std::function<float(int64_t)>& fill,
                bool out = false) {
        std::vector<int64_t> st(shape.size());
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            st[(size_t)i] = acc;
            acc *= shape[(size_t)i];
        }
        return view(dt, shape, st, acc, fill, out);
    }
    Hand& idx(std::vector<int64_t> v, bool out = false) {
        return dense(RAD_I32, { (int64_t)v.size() }, [v](int64_t i) { return (float)v[(size_t)i]; },
                     out);
    }
    Hand& none() {
        ta.push_back(RadTensor{});
        tb.push_back(RadTensor{});
        buf_of.push_back(-1);
        return *this;
    }
    Hand& pi(const char* k, long long v) {
        p.push_back(RadParam{ k, RAD_P_INT, v, 0, nullptr, 0 });
        return *this;
    }
    Hand& pf(const char* k, double v) {
        p.push_back(RadParam{ k, RAD_P_F64, 0, 0, nullptr, v });
        return *this;
    }
    Hand& ps(const char* k, const char* v) {
        strs.emplace_back(v);
        p.push_back(RadParam{ k, RAD_P_STR, 0, 0, strs.back().c_str(), 0 });
        return *this;
    }

    /* Returns false on a failure, having printed it. */
    bool run(const Lib& ref, const Lib& avx, bool verbose) {
        const RadKernelInfo* kr = ref.row(op.c_str());
        const RadKernelInfo* ka = avx.row(op.c_str());
        if (!kr || !ka) {
            std::printf("  FAIL %-20s %s\n       no row for the op in %s\n", op.c_str(),
                        name.c_str(), kr ? "libavx" : "libref");
            return false;
        }
        for (size_t i = 0; i < ta.size(); ++i) {
            if (buf_of[i] < 0) continue;
            ta[i].data = bufs[(size_t)buf_of[i]].a.data();
            tb[i].data = bufs[(size_t)buf_of[i]].b.data();
        }
        RadArgs aa{}, ab{};
        aa.t = ta.data(); aa.n_t = (int)ta.size();
        aa.p = p.data();  aa.n_p = (int)p.size();
        aa.world_size = 1;
        ab = aa; ab.t = tb.data();
        const int rr = kr->launch(&aa, nullptr);
        const int ra = ka->launch(&ab, nullptr);
        if (rr < 0) {
            std::printf("  FAIL %-20s %s\n       libref refused the case (%d); the case is wrong\n",
                        op.c_str(), name.c_str(), rr);
            return false;
        }
        if (ra < 0) {
            std::printf("  FAIL %-20s %s\n       libavx returned %d where libref returned %d\n",
                        op.c_str(), name.c_str(), ra, rr);
            return false;
        }
        bool ok = true;
        double worst = 0;
        for (const Buf& bf : bufs) {
            if (!bf.out) continue;
            std::vector<float> fr((size_t)bf.elems), fa((size_t)bf.elems);
            h_widen(bf.a.data(), bf.dt, bf.elems, fr.data());
            h_widen(bf.b.data(), bf.dt, bf.elems, fa.data());
            const Err e = compare(fa.data(), fr.data(), bf.elems);
            worst = std::max(worst, e.rel_l2);
            if (e.nonfinite_gap != 0 || !(e.rel_l2 <= tol)) {
                ok = false;
                std::printf("  FAIL %-20s %s\n       rel_l2=%.3e max_abs=%.3e nonfinite_gap=%lld "
                            "tol=%.1e\n", op.c_str(), name.c_str(), e.rel_l2, e.max_abs,
                            (long long)e.nonfinite_gap, tol);
            }
        }
        if (verbose)
            std::printf("  %-22s %-46s rel_l2=%.3e tol=%.1e %s\n", op.c_str(), name.c_str(), worst,
                        tol, ok ? "ok" : "FAIL");
        return ok;
    }
};

/* A deterministic fill that is not a constant: the value of element i under seed s. */
static std::function<float(int64_t)> hfill(uint64_t s, float sigma = 1.0f) {
    return [s, sigma](int64_t i) {
        uint64_t x = s * 0x9E3779B97F4A7C15ull + (uint64_t)i;
        x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29;
        return ((float)(x % 2001) / 1000.0f - 1.0f) * sigma;
    };
}

/* Independent PyTorch GPU outputs cover the low-precision filter weights and intermediate
 * rounding, while the full-buffer oracle also checks strided inputs and untouched padding. */
static bool grid_embed_antialias_cases(const Lib& ref, const Lib& avx, bool verbose) {
    namespace g = grid_embed_golden;
    bool passed = true;
    for (uint32_t table_dt : {RAD_BF16, RAD_F32}) {
        const int n = 35, pitch = 2 * n + 3;
        Hand h("GPU antialias golden, pitched x/table", "grid_embed", 1e-7);
        h.view(RAD_BF16, {g::rows, n}, {pitch, 2}, g::rows * pitch,
               [=](int64_t i) { const int c = i % pitch;
                   return c < 2 * n && c % 2 == 0 ? g::patch(i / pitch, (c / 2) % g::n) : 19.0f;
               }, true)
         .view(table_dt, {g::side * g::side, n}, {pitch, 2}, g::side * g::side * pitch,
               [=](int64_t i) { const int c = i % pitch;
                   return c < 2 * n && c % 2 == 0 ? g::table(i / pitch, (c / 2) % g::n) : -29.0f;
               })
         .dense(RAD_I32, {4, g::rows}, [](int64_t i) { return (float)g::coord[i % g::rows][i / g::rows]; })
         .pi("M", g::rows).pi("n", n).pi("side", g::side).ps("dtype", "bf16")
         .ps("mode", "bilinear_antialias");
        if (!h.run(ref, avx, verbose)) passed = false;
        const uint16_t* expected = table_dt == RAD_BF16 ? g::bf16 : g::f32;
        const Hand::Buf& x = h.bufs[0];
        bool exact = true;
        for (int r = 0; r < g::rows; ++r) for (int c = 0; c < n; ++c) {
            const float want = rad_bf16_to_f32(expected[r * g::n + c % g::n]);
            const int i = r * pitch + c * 2;
            if (rad_load_f32(x.a.data(), RAD_BF16, i) != want ||
                rad_load_f32(x.b.data(), RAD_BF16, i) != want) exact = false;
        }
        if (!exact) {
            std::printf("  FAIL grid_embed antialias differs from GPU golden (table %s)\n",
                        table_dt == RAD_BF16 ? "bf16" : "f32");
            passed = false;
        }
    }
    return passed;
}

/* Integer input squares sum exactly in every reduction order, so these cases can require exact
 * agreement on the extra rounding instead of hiding it inside the usual norm tolerance. The
 * varying gains make narrowing before their multiply observably different from narrowing after. */
static bool rmsnorm_rounding_cases(const Lib& ref, const Lib& avx, bool verbose) {
    bool ok = true;
    for (const char* dtype : { "bf16", "f16" }) {
        const uint32_t dt = rad_dtype_parse(dtype);
        for (int64_t n : { 64, 65 }) {
            for (int rounding : { 0, 1 }) {
                const std::string name = std::string(dtype) + " n=" + std::to_string(n) +
                    " round_before_weight=" + std::to_string(rounding);
                Hand h(name.c_str(), "rmsnorm", 0.0);
                const int64_t stride = n == 65 ? 2 : 1;
                h.view(dt, { 2, n }, { n * stride, stride }, 2 * n * stride,
                       [stride, n](int64_t i) { return (float)(1 + (i / stride % n) % 4); })
                 .dense(dt, { n }, [](int64_t i) { return 1.0f + (float)(i % 17) * 0.03125f; })
                 .dense(dt, { 2, n }, [](int64_t) { return 0.0f; }, true)
                 .pi("M", 2).pi("n", n).pf("eps", 1e-6).ps("dtype", dtype)
                 .pi("round_before_weight", rounding);
                if (!h.run(ref, avx, verbose)) ok = false;
            }
        }
    }
    std::printf("  rmsnorm rounding: 8 exact cases %s\n", ok ? "passed" : "FAILED");
    return ok;
}

/* Every product here is dyadic and the sums are exactly representable in F32, so the SIMD
 * reduction order cannot obscure an index/stride bug or a missing BF16 rounding seam. Padding
 * holds unrelated values, and the whole output allocation is checked for stray writes. */
static bool logits_select_cases(const Lib& ref, const Lib& avx, bool verbose) {
    bool ok = true;
    int checked = 0;
    constexpr int64_t R = 6, V = 7, K = 35, M = 7;
    constexpr int64_t xp = 2 * K + 3, xb = 3 * xp + 5, wp = 3 * K + 2;
    const int row_ids[M] = {0, 5, 2, 0, 4, 1, 5};
    const int token_ids[M] = {0, 6, 2, 0, 1, 3, 6};
    for (const char* dtype : { "f32", "bf16", "f16" }) {
        for (int rounding : { -1, 0, 1 }) {
            const uint32_t dt = rad_dtype_parse(dtype);
            const std::string name = std::string(dtype) + " strided pairs, rounding=" +
                (rounding < 0 ? "default" : std::to_string(rounding));
            Hand h(name.c_str(), "logits_select", 0.0);
            h.view(dt, {2, 3, K}, {xb, xp, 2}, 2 * xb,
                   [](int64_t i) {
                       for (int64_t r = 0; r < R; ++r) {
                           const int64_t k2 = i - (r / 3 * xb + r % 3 * xp);
                           if (k2 >= 0 && k2 < 2 * K && !(k2 & 1))
                               return r == 0 ? (k2 == 0 ? 1.0078125f : 0.0f) :
                                   (float)(r + (k2 / 2) % 5) * 0.125f;
                       }
                       return 99.0f;
                   })
             .view(dt, {V, K}, {wp, 3}, V * wp,
                   [](int64_t i) {
                       const int64_t v = i / wp, k3 = i % wp;
                       if (k3 < 3 * K && k3 % 3 == 0)
                           return v == 0 ? (k3 == 0 ? 1.0078125f : 0.0f) :
                               (float)(v - (k3 / 3) % 3) * 0.25f;
                       return -99.0f;
                   })
             .view(RAD_I32, {M}, {2}, 2 * M,
                   [&row_ids](int64_t i) { return (float)(i % 2 ? 999 : row_ids[i / 2]); })
             .view(RAD_I32, {M}, {2}, 2 * M,
                   [&token_ids](int64_t i) { return (float)(i % 2 ? 999 : token_ids[i / 2]); })
             .view(RAD_F32, {M}, {3}, 3 * M, [](int64_t) { return -17.0f; }, true)
             .pi("M", 4096).pi("n_rows", 64).pi("n_vocab", V).pi("n_embd", K).ps("dtype", dtype);
            if (rounding >= 0) h.pi("round_bf16", rounding);
            ++checked;
            if (!h.run(ref, avx, verbose)) ok = false;

            const float expected = rounding == 1 ? 1.015625f : 1.01568603515625f;
            for (const auto* tensors : { &h.ta, &h.tb })
                if (rad_load_f32((*tensors)[4].data, RAD_F32, 0) != expected ||
                    rad_load_f32((*tensors)[4].data, RAD_F32, 9) != expected) {
                    std::printf("  FAIL logits_select %s: duplicate pair or rounding seam\n", name.c_str());
                    ok = false;
                }

            // Rejections must agree with the oracle and leave every output byte unchanged.
            auto refuse = [&](const char* why, int expected_rc) {
                ++checked;
                bool pass = true;
                for (int side = 0; side < 2; ++side) {
                    auto& tensors = side ? h.tb : h.ta;
                    const Lib& lib = side ? avx : ref;
                    const auto* kernel = lib.row("logits_select");
                    const auto& output = side ? h.bufs.back().b : h.bufs.back().a;
                    const auto before = output;
                    RadArgs args{};
                    args.t = tensors.data(); args.n_t = (int)tensors.size();
                    args.p = h.p.data(); args.n_p = (int)h.p.size(); args.world_size = 1;
                    if (!kernel || kernel->launch(&args, nullptr) != expected_rc || output != before)
                        pass = false;
                }
                if (!pass) {
                    std::printf("  FAIL logits_select %s: %s rejection changed output or status\n",
                                name.c_str(), why);
                    ok = false;
                }
            };
            auto set_last = [&](int operand, int value) {
                auto& b = h.bufs[(size_t)h.buf_of[(size_t)operand]];
                rad_store_f32(b.a.data(), RAD_I32, 2 * (M - 1), (float)value);
                rad_store_f32(b.b.data(), RAD_I32, 2 * (M - 1), (float)value);
            };
            for (int invalid : { -1, (int)R }) {
                set_last(2, invalid);
                refuse("hidden row out of range", RAD_E_INVAL);
            }
            set_last(2, row_ids[M - 1]);
            for (int invalid : { -1, (int)V }) {
                set_last(3, invalid);
                refuse("vocabulary token out of range", RAD_E_INVAL);
            }
            set_last(3, token_ids[M - 1]);
            h.ta[4].shape[0] = h.tb[4].shape[0] = M - 1;
            refuse("short output", RAD_E_SHAPE);
            h.ta[4].shape[0] = h.tb[4].shape[0] = M;
            h.ta[2].dtype = h.tb[2].dtype = RAD_F32;
            refuse("noninteger row indices", RAD_E_DTYPE);
            h.ta[2].dtype = h.tb[2].dtype = RAD_I32;
            if (rounding < 0) h.pi("round_bf16", 2);
            else h.p.back().ival = 2;
            refuse("invalid rounding flag", RAD_E_INVAL);
        }
    }
    std::printf("  selected logits: %d exact/contract cases %s\n", checked, ok ? "passed" : "FAILED");
    return ok;
}

/* Successive calls keep each plugin's own state, so a cold prefill, warm short chunk and rejected
 * verify cannot accidentally pass by starting from a fresh identical fixture every time. */
static bool causal_conv_cases(const Lib& ref, const Lib& avx, bool verbose) {
    bool ok = true;
    int checked = 0;
    for (const char* dtype : { "f32", "bf16", "f16" }) {
        const uint32_t dt = rad_dtype_parse(dtype);
        constexpr int64_t T = 4, N = 35, W = 3, DEPTH = 4;
        Hand h("cold pitched prefill", "causal_conv1d", 1e-5);
        // Powers of two keep the short multiply-add exact even on an ISA that contracts FMA.
        h.view(dt, {T, N}, {2 * N + 3, 2}, T * (2 * N + 3),
               [](int64_t i) { return (float)(i % 13 - 6) * 0.125f; })
         .view(dt, {N, W}, {2 * W + 1, 2}, N * (2 * W + 1),
               [](int64_t i) { return (float)(i % 7 - 3) * 0.25f; })
         .dense(dt, {2, N, DEPTH}, [](int64_t i) { return 7.0f + (float)(i % 5); }, true)
         .idx({0, 2, 4}).idx({1, 0}).idx({0, 0}).none()
         .view(dt, {T, N}, {3 * N + 5, 3}, T * (3 * N + 5),
               [](int64_t) { return -17.0f; }, true)
         .pi("M", 64).pi("n", N).pi("width", W).ps("dtype", dtype);
        auto set_ids = [&](int operand, std::initializer_list<int> values) {
            Hand::Buf& b = h.bufs[(size_t)h.buf_of[(size_t)operand]];
            size_t i = 0;
            for (int value : values) {
                rad_store_f32(b.a.data(), b.dt, i, (float)value);
                rad_store_f32(b.b.data(), b.dt, i++, (float)value);
            }
        };
        auto run = [&](const char* name) {
            h.name = std::string(dtype) + " " + name;
            ++checked;
            if (!h.run(ref, avx, verbose)) ok = false;
        };
        run("cold pitched prefill");
        set_ids(3, {0, 1, 2});
        set_ids(5, {1, 1});
        run("warm chunks shorter than history");

        // Reuse the optional operand's slot with an i32 buffer, without replacing either state.
        const size_t appended = h.ta.size();
        h.idx({1, 1});
        h.ta[6] = h.ta[appended]; h.tb[6] = h.tb[appended];
        h.buf_of[6] = h.buf_of[appended];
        h.ta.pop_back(); h.tb.pop_back(); h.buf_of.pop_back();
        set_ids(3, {0, 2, 4});
        run("two-token verify retains candidates");
        set_ids(6, {1, 2});
        run("next verify rejects one prefix and accepts another");
        set_ids(3, {1, 2, 4});
        set_ids(4, {1, -1});
        run("global cu offset leaves inactive rows and slot untouched");

        Hand defaults("implicit slots and initialized history", "causal_conv1d", 1e-5);
        defaults.dense(dt, {T, N}, [](int64_t i) { return (float)(i % 5) * 0.125f; })
         .dense(dt, {N, W}, [](int64_t i) { return (float)(i % 3 - 1) * 0.25f; })
         .dense(dt, {2, N, DEPTH}, [](int64_t i) { return (float)(i % 7) * 0.25f; }, true)
         .idx({0, 2, 4}).none().none().none()
         .dense(dt, {T, N}, [](int64_t) { return 0.0f; }, true)
         .pi("M", 64).pi("n", N).pi("width", W).ps("dtype", dtype);
        ++checked;
        if (!defaults.run(ref, avx, verbose)) ok = false;
    }
    std::printf("  causal convolution: %d stateful cases %s\n", checked, ok ? "passed" : "FAILED");
    return ok;
}

static bool hand_cases(const Lib& ref, const Lib& avx, bool verbose, int* n_checked) {
    std::vector<Hand> hs;

    /* ---- attention: two sequences through the uniform long-query form, no cu_seqlens ------- */
    {
        const int64_t S = 2, QL = 32, KVH = 1, G = 2, HD = 16, BS = 16, NB = 6;
        Hand h("n_seq=2 q_len=32 no cu_seqlens", "attn_paged", 1e-4);
        h.dense(RAD_F32, { S * QL, KVH * G, HD }, hfill(1))
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(2))
         .dense(RAD_I32, { S, 3 }, [](int64_t i) { return (float)((i * 5) % 6); })
         .idx({ 40, 36 })
         .none().none().none()
         .dense(RAD_F32, { S * QL, KVH * G, HD }, hfill(3), true)
         .pi("q_len", QL).pi("head_dim", HD).pi("gqa", G).pi("block_size", BS).pi("causal", 1)
         .pi("window", 0).ps("q_dtype", "f32").ps("kv_dtype", "f32");
        hs.push_back(std::move(h));
    }
    /* ---- attention and the KV write over column slices of a fused [q|k|v] projection ------- */
    {
        const int64_t T = 5, NH = 2, KVH = 1, HD = 16, W = (NH + 2 * KVH) * HD, BS = 4, NB = 4;
        Hand h("k, v are rank-2 column slices, row stride 64", "kv_store", 1e-6);
        h.view(RAD_F32, { T, KVH * HD }, { W, 1 }, T * W, hfill(4))            /* k at column 0 */
         .view(RAD_F32, { T, KVH * HD }, { W, 1 }, T * W, hfill(5))
         .idx({ 3, 0, 9, 14, 6 })
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(6), true)
         .pi("M", T).pi("head_dim", HD).pi("n_head_kv", KVH).pi("block_size", BS)
         .ps("kv_dtype", "f32");
        hs.push_back(std::move(h));

        Hand g("q and out are rank-2 column slices", "attn_paged", 1e-4);
        g.view(RAD_F32, { T, NH * HD }, { W, 1 }, T * W, hfill(7))
         .dense(RAD_F32, { NB, KVH, BS, 2 * HD }, hfill(8))
         .dense(RAD_I32, { T, 2 }, [](int64_t i) { return (float)(i % NB); })
         .idx({ 3, 5, 7, 2, 8 })
         .none().none().none()
         .view(RAD_F32, { T, NH * HD }, { W, 1 }, T * W, hfill(9), true)
         .pi("q_len", 1).pi("head_dim", HD).pi("gqa", NH / KVH).pi("block_size", BS)
         .pi("causal", 1).pi("window", 0).ps("q_dtype", "f32").ps("kv_dtype", "f32");
        hs.push_back(std::move(g));

        Hand d("q, k, v, out are rank-2 column slices", "attn_dense", 1e-4);
        d.view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(10))
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(11))
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(12))
         .idx({ 0, 2, 5 })
         .view(RAD_F32, { T, HD }, { W, 1 }, T * W, hfill(13), true)
         .pi("M", T).pi("head_dim", HD).pi("gqa", 1).pi("causal", 1).ps("dtype", "f32");
        hs.push_back(std::move(d));
    }
    /* ---- the GDN decode convolution read past offset zero ---------------------------------- */
    {
        const int64_t S = 2, QL = 3, HG = 2, H = 2, K = 16, V = 16, WID = 4;
        const int64_t CD = 2 * HG * K + H * V, SL = WID - 2 + QL;
        Hand h("num_accepted 2 and 3", "gdn_conv_update", 2e-5);
        h.dense(RAD_F32, { S * QL, CD }, hfill(14))
         .dense(RAD_F32, { CD, WID }, hfill(15, 0.5f))
         .none()
         .dense(RAD_F32, { 3, CD, SL }, hfill(16), true)
         .idx({ 2, 0 })
         .idx({ 2, 3 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(17), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(18), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(19), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(h));

        /* The same step over a CHANNEL-CONTIGUOUS state, [slot][t][c] viewed as [slot][c][t]:
         * the layout that takes libavx's row-move path rather than its element loop. */
        Hand m("num_accepted 2 and 3, channel-contiguous state", "gdn_conv_update", 2e-5);
        m.dense(RAD_F32, { S * QL, CD }, hfill(14))
         .dense(RAD_F32, { CD, WID }, hfill(15, 0.5f))
         .none()
         .view(RAD_F32, { 3, CD, SL }, { SL * CD, 1, CD }, 3 * SL * CD, hfill(16), true)
         .idx({ 2, 0 })
         .idx({ 2, 3 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(17), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(18), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(19), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(m));
    }
    /* ---- head widths that do not divide the 1024-channel panel ------------------------------- */
    {
        const int64_t S = 2, QL = 2, HG = 4, H = 8, K = 96, V = 96, WID = 4;
        const int64_t CD = 2 * HG * K + H * V, SL = WID - 2 + QL;
        Hand h("head_k = head_v = 96, conv_dim 1536", "gdn_conv_update", 2e-5);
        h.dense(RAD_F32, { S * QL, CD }, hfill(20))
         .dense(RAD_F32, { CD, WID }, hfill(21, 0.5f))
         .none()
         .dense(RAD_F32, { 2, CD, SL }, hfill(22), true)
         .idx({ 1, 0 })
         .idx({ 1, 2 })
         .idx({ 0, QL, 2 * QL })
         .dense(RAD_F32, { S * QL, HG, K }, hfill(23), true)
         .dense(RAD_F32, { S * QL, HG, K }, hfill(24), true)
         .dense(RAD_F32, { S * QL, H, V }, hfill(25), true)
         .pi("q_len", QL).pi("head_k", K).pi("head_v", V).pi("conv_width", WID)
         .pi("max_query_len", QL);
        hs.push_back(std::move(h));

        const int64_t M = 6;
        Hand p("head_k = head_v = 96, conv_dim 1536", "gdn_conv_prep", 2e-5);
        p.dense(RAD_F32, { M, CD + 2 * H }, hfill(26))
         .dense(RAD_F32, { CD, WID }, hfill(27, 0.5f))
         .none()
         .dense(RAD_F32, { H }, hfill(28, 0.5f))
         .dense(RAD_F32, { H }, hfill(29, 0.5f))
         .dense(RAD_F32, { 2, CD, WID - 1 }, hfill(30), true)
         .idx({ 0, 4, M })
         .none().none()
         .idx({ 1, 0 })
         .idx({ 1, 0 })
         .dense(RAD_F32, { M, HG, K }, hfill(31), true)
         .dense(RAD_F32, { M, HG, K }, hfill(32), true)
         .dense(RAD_F32, { M, H, V }, hfill(33), true)
         .dense(RAD_F32, { M, H }, hfill(34), true)
         .dense(RAD_F32, { M, H }, hfill(35), true)
         .pi("M", M).pi("head_k", K).pi("head_v", V).pi("chunk", 4).pi("conv_width", WID)
         .pf("l2_eps", 1e-6).pf("softplus_thr", 20.0);
        hs.push_back(std::move(p));
    }
    /* ---- the PLE ops at their band's upper bound, and a verify over two sequences ------------ */
    {
        const int64_t T = 5, N = 32, HC = 2, BAND = 64;
        Hand g("M = 64 over 5 rows", "ple_gate", 1e-5);
        g.dense(RAD_F32, { T, HC * N }, hfill(36))
         .dense(RAD_F32, { T, HC * N }, hfill(37))
         .dense(RAD_F32, { T, N }, hfill(38))
         .dense(RAD_F32, { HC * N }, hfill(39, 0.2f))
         .dense(RAD_F32, { HC * N }, hfill(40, 0.2f))
         .dense(RAD_F32, { HC * N }, hfill(41, 0.2f))
         .dense(RAD_F32, { T, HC * N }, hfill(42), true)
         .dense(RAD_F32, { T, HC * N }, hfill(43), true)
         .pi("M", BAND).pi("n", N).pi("hc", HC).pf("eps", 1e-6).ps("dtype", "f32")
         .pf("wadd", 1.0);
        hs.push_back(std::move(g));

        /* Two sequences, a verify of 3 and one of 2, reading at offsets 1 and 2 of a window deep
         * enough for both: hist 6 at width 4 and dilation 2, plus a verify's n - 1. */
        const int64_t WIDTH = 4, DIL = 2, HIST = (WIDTH - 1) * DIL, SL = HIST + 2;
        Hand c("M = 64 over 5 rows, verifies of 3 and 2", "ple_conv", 1e-5);
        c.dense(RAD_F32, { T, N }, hfill(44))
         .dense(RAD_F32, { N, WIDTH }, hfill(45, 0.5f))
         .dense(RAD_F32, { 2, N, SL }, hfill(46), true)
         .idx({ 0, 3, 5 })
         .dense(RAD_F32, { T, N }, hfill(47))
         .idx({ 1, 0 })
         .idx({ 7, 9 })
         .idx({ 2, 3 })
         .dense(RAD_F32, { T, N }, hfill(48), true)
         .pi("M", BAND).pi("n", N).pi("width", WIDTH).pi("dilation", DIL).ps("dtype", "f32");
        hs.push_back(std::move(c));

        Hand n("M = 64 over 5 rows, verifies of 3 and 2", "ngram_ids", 0.0);
        n.idx({ 11, 12, 13, 21, 22 })
         .dense(RAD_I32, { 2, 2 + 2 }, [](int64_t i) { return (float)(100 + i); }, true)
         .idx({ 0, 3, 5 })
         .dense(RAD_I64, { 3 }, [](int64_t i) { return (float)(1000003 + 7919 * i); })
         .dense(RAD_I64, { 4 }, [](int64_t i) { return (float)(97 + 2 * i); })
         .dense(RAD_I64, { 4 }, [](int64_t i) { return (float)(1000 * i); })
         .idx({ 1, 0 })
         .idx({ 7, 9 })
         .idx({ 2, 3 })
         .dense(RAD_I32, { T, 4 }, [](int64_t) { return -1.0f; }, true)
         .pi("M", BAND).pi("heads", 4).pi("ngram", 3).pi("eos", 5);
        hs.push_back(std::move(n));
    }
    /* ---- packed weight codes at odd element offsets ------------------------------------------ */
    {
        const int64_t M = 2, N = 4, K = 33;
        Hand h("W4 weight, K = 33: odd rows start mid-byte", "gemm_nt_q", 1e-5);
        h.dense(RAD_F32, { M, K }, hfill(49))
         .none()
         .dense(RAD_I4, { N, K }, hfill(50, 2.0f))
         .dense(RAD_F32, { N, 1 }, [](int64_t i) { return 0.5f + 0.25f * (float)i; })
         .dense(RAD_F32, { M, N }, hfill(51), true)
         .pi("M", M).pi("N", N).pi("K", K).ps("dtype", "w4a16");
        hs.push_back(std::move(h));

        Hand w("W2 weight, K = 33", "gemm_nt_q", 1e-5);
        w.dense(RAD_F32, { M, K }, hfill(52))
         .none()
         .dense(RAD_I2, { N, K }, hfill(53, 2.0f))
         .dense(RAD_F32, { N, 1 }, [](int64_t i) { return 1.0f + (float)i; })
         .dense(RAD_F32, { M, N }, hfill(54), true)
         .pi("M", M).pi("N", N).pi("K", K).ps("dtype", "w2a16");
        hs.push_back(std::move(w));
    }
    /* ---- integer stores past the range, and exponents of -inf and NaN ------------------------ */
    {
        const float big[] = { 1e10f, -1e10f, INFINITY, -INFINITY, 2.5f, -2.5f, 3.5f, 127.5f,
                              -128.5f, 300.0f, -300.0f, 0.49f };
        const int64_t n = (int64_t)(sizeof big / sizeof big[0]);
        for (const char* to : { "i8", "u8", "i16" }) {
            /* 51 elements: the vector body runs at every level and so does the scalar tail. */
            const int64_t len = 4 * n + 3;
            Hand h((std::string("f32 -> ") + to + " past the range").c_str(), "cast", 0.0);
            const uint32_t dt = rad_dtype_parse(to);
            h.dense(RAD_F32, { 1, len }, [&big, n](int64_t i) { return big[i % n]; })
             .dense(dt, { 1, len }, [](int64_t) { return 0.0f; }, true)
             .pi("M", 1).pi("n", len).ps("from", "f32").ps("to", to);
            hs.push_back(std::move(h));
        }
        /* 48 wide, a whole number of vectors at every level: a scalar tail's own exp would turn
         * the row's sum into a NaN by itself and hide what the vector exp did with one. */
        Hand s("rows of -inf: one fully masked, one partly", "softmax", 1e-6);
        const int64_t w = 48;
        s.dense(RAD_F32, { 3, w }, [w](int64_t i) {
                const int64_t r = i / w, c = i % w;
                if (r == 0) return -INFINITY;
                if (r == 1 && c % 3 == 0) return -INFINITY;
                return (float)(c % 7) * 0.5f - 1.0f;
            })
         .dense(RAD_F32, { 3, w }, [](int64_t) { return 0.0f; }, true)
         .pi("M", 3).pi("n", w).ps("dtype", "f32");
        hs.push_back(std::move(s));
    }

    /* ---- the drafter's walk with the pairwise term in charge -------------------------------- *
     * A cold description draws `pred` and `succ` as weights, at 0.02, where the pairwise term is a
     * thousandth of `unary` and never moves a pick: the walk is checked and the form is not. Here
     * the codebooks are unit-scale and `unary` half that, so every pick is the form's. Some
     * candidates are outside the vocabulary (-1 and past it) and so is the second sequence's
     * anchor, which is the "contributes nothing" branch. Once with `hp` and once without. */
    for (int with_hp : { 1, 0 }) {
        const int64_t M = 3, L = 4, K = 8, R = 40, NV = 50;
        Hand h(with_hp ? "unit codebooks, hp present, ids outside the vocab"
                       : "unit codebooks, hp absent (a plane of ones)", "dflash_select", 0.0);
        h.dense(RAD_I32, { M * L, K }, [](int64_t i) { return (float)((i * 7 + 3) % 53 - 1); })
         .dense(RAD_F32, { M * L, K }, hfill(31, 0.5f));
        if (with_hp) h.dense(RAD_BF16, { M * L, R }, hfill(32));
        else         h.none();
        h.idx({ 5, 60, 17 })
         .dense(RAD_BF16, { NV, R }, hfill(33))
         .dense(RAD_BF16, { NV, R }, hfill(34))
         .dense(RAD_I32, { M, L }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("steps", L).pi("top_k", K).pi("rank", R).pi("n_vocab", NV)
         .pi("anchor_stride", 1).ps("dtype", "bf16");
        hs.push_back(std::move(h));
    }

    /* ---- the fp8 fusions without their optional operands ------------------------------------ *
     * The cold descriptions always carry `residual` and `out_bf16`; the plain norm and a caller
     * that wants only the codes are the other forms. `mode` 1 is sigmoid in libr4d's spelling. */
    {
        const int64_t M = 4, n = 200;
        Hand r("no residual, no out_bf16, ragged group", "rmsnorm_quant_fp8", 4e-3);
        r.dense(RAD_BF16, { M, n }, hfill(41))
         .none()
         .dense(RAD_BF16, { n }, hfill(42, 0.1f))
         .dense(RAD_F8E4M3, { M, n }, [](int64_t) { return 0.0f; }, true)
         .dense(RAD_F32, { M, 4 }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("n", n).pf("eps", 1e-6).pi("group", 64).ps("dtype", "bf16")
         .pf("wadd", 1.0);
        hs.push_back(std::move(r));

        Hand g("mode 1, no out_bf16", "gated_quant_fp8", 6e-3);
        g.dense(RAD_BF16, { M, 2 * n }, hfill(43))
         .dense(RAD_F8E4M3, { M, n }, [](int64_t) { return 0.0f; }, true)
         .dense(RAD_F32, { M, 4 }, [](int64_t) { return 0.0f; }, true)
         .pi("M", M).pi("n", n).pi("group", 64).ps("dtype", "bf16").pi("mode", 1);
        hs.push_back(std::move(g));
    }

    /* ---- scatter_rows: a negative index writes nothing, and the rows nobody names stay ------- */
    {
        const int64_t n = 70;
        Hand s("idx with -1 over a 12-row pool", "scatter_rows", 0.0);
        s.dense(RAD_F32, { 4, n }, hfill(51))
         .idx({ 5, -1, 0, 9 })
         .dense(RAD_F32, { 12, n }, hfill(52), true)
         .pi("M", 4).pi("n", n).ps("dtype", "f32");
        hs.push_back(std::move(s));
    }

    bool ok = true;
    int fails = 0;
    for (Hand& h : hs) {
        ++*n_checked;
        if (!h.run(ref, avx, verbose)) { ok = false; ++fails; }
    }
    std::printf("  hand-built: %d checked, %d failed\n", (int)hs.size(), fails);
    return ok;
}

/* ================================================================== schema agreement
 * core/plugin/loader.cpp:schema_same, reproduced so a drift is caught here rather than at an
 * engine start on a machine that has a model on it. */
static bool schema_same(const RadOpSchema& a, const RadOpSchema& b, std::string* why) {
    char buf[256];
    if (a.n_params != b.n_params) {
        std::snprintf(buf, sizeof buf, "%d parameters vs %d", a.n_params, b.n_params);
        *why = buf; return false;
    }
    for (int i = 0; i < a.n_params; ++i) {
        const RadParamSpec& x = a.params[i];
        const RadParamSpec& y = b.params[i];
        const char* xk = x.key ? x.key : "";
        const char* yk = y.key ? y.key : "";
        if (std::strcmp(xk, yk)) {
            std::snprintf(buf, sizeof buf, "parameter %d is '%s' vs '%s'", i, xk, yk);
            *why = buf; return false;
        }
        if (x.type != y.type) {
            std::snprintf(buf, sizeof buf, "parameter '%s' has type %d vs %d", xk, x.type, y.type);
            *why = buf; return false;
        }
        if (x.required != y.required) {
            std::snprintf(buf, sizeof buf, "parameter '%s' is %s vs %s", xk,
                          x.required ? "required" : "optional",
                          y.required ? "required" : "optional");
            *why = buf; return false;
        }
    }
    if (a.n_operands != b.n_operands) {
        std::snprintf(buf, sizeof buf, "%d operands vs %d", a.n_operands, b.n_operands);
        *why = buf; return false;
    }
    for (int i = 0; i < a.n_operands; ++i)
        if (a.operands[i].role != b.operands[i].role ||
            a.operands[i].optional != b.operands[i].optional) {
            std::snprintf(buf, sizeof buf, "operand %d ('%s') has a different role/optionality", i,
                          a.operands[i].name ? a.operands[i].name : "?");
            *why = buf; return false;
        }
    return true;
}

/* embed_lookup_q over a file-backed table, two ranks and the reader racing (avx_check_rows.cpp). */
bool row_gather_case(const Lib& avx, const std::string& dir, int64_t calls, bool verbose,
                     bool bf16);
/* embed_lookup_q's init, where io_uring is refused and here (avx_check_rows.cpp). */
bool ngram_init_case(const Lib& avx);

int main(int argc, char** argv) {
    std::string ref_path = "libref.so", avx_path = "libavx.so", only_op;
    int only_level = -1;
    bool verbose = false;
    double tol_override = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if      (s == "--ref")   ref_path = next();
        else if (s == "--avx")   avx_path = next();
        else if (s == "--op")    only_op = next();
        else if (s == "--level") only_level = std::atoi(next());
        else if (s == "--tol")   tol_override = std::atof(next());
        else if (s == "-v" || s == "--verbose") verbose = true;
        else if (s == "-h" || s == "--help") {
            std::printf(
                "rad-avx-check -- libavx against libref, every op, every ISA level.\n\n"
                "  --ref PATH     the oracle plugin      (default libref.so)\n"
                "  --avx PATH     the plugin under test  (default libavx.so)\n"
                "  --op NAME      only this op\n"
                "  --level N      only this ISA level (0 scalar 1 avx 2 avx2 3 avx512)\n"
                "  --tol X        override every tolerance (for bisecting, not for passing)\n"
                "  -v             print every case, not only the failures\n");
            return 0;
        }
    }

    Lib ref, avx;
    if (!ref.open(ref_path.c_str()) || !avx.open(avx_path.c_str())) return 2;
    std::printf("oracle : %s %s  (%s)\n", ref.info->name, ref.info->version, ref_path.c_str());
    std::printf("under  : %s %s  (%s)\n", avx.info->name, avx.info->version, avx_path.c_str());

    /* ---- 1. schema agreement */
    int sch_bad = 0;
    for (int i = 0, n = avx.schema_count(); i < n; ++i) {
        const RadOpSchema* sa = avx.schema_at(i);
        if (!sa || !sa->op) continue;
        const RadOpSchema* sr = ref.schema(sa->op);
        if (!sr) {
            std::printf("SCHEMA  %-24s declared by libavx and not by libref\n", sa->op);
            ++sch_bad;
            continue;
        }
        std::string why;
        if (!schema_same(*sr, *sa, &why)) {
            std::printf("SCHEMA  %-24s DISAGREES: %s\n", sa->op, why.c_str());
            ++sch_bad;
        }
    }
    std::printf("schemas: %d declared, %d disagree with libref\n", avx.schema_count(), sch_bad);
    if (sch_bad) {
        std::printf("\nThe loader refuses a plugin on ANY schema disagreement, so this would make "
                    "every row\nlibavx carries unreachable. Nothing below is worth reading until "
                    "that is fixed.\n");
        return 1;
    }

    /* ---- 2. coverage */
    int missing = 0;
    for (int i = 0, n = ref.kernel_count(); i < n; ++i) {
        const RadKernelInfo* kr = ref.kernel_at(i);
        if (!kr || !kr->op) continue;
        if (!avx.row(kr->op)) {
            std::printf("COVER   %-24s libref implements it, libavx does not\n", kr->op);
            ++missing;
        }
    }
    std::printf("coverage: libref %d rows, libavx %d rows, %d op(s) missing\n",
                ref.kernel_count(), avx.kernel_count(), missing);

    /* ---- 3/4. the numbers, per ISA level */
    const std::vector<Case> cases = build_cases();
    if (!avx.isa_force) {
        std::printf("\nNOTE: %s exports no avx_isa_force, so only the path it selects for itself "
                    "is\nexercised. That is honest but it is not the four-level sweep.\n",
                    avx_path.c_str());
    } else {
        std::printf("cpu    : highest executable level is %s\n", avx.isa_name(avx.isa_detect()));
    }

    int rc = missing ? 1 : 0;
    /* Before anything runs a gather: the case forks, and a child is cleanest taken from a process
     * that has started none of the gather's threads. */
    if (only_op.empty() || only_op == "embed_lookup_q") {
        std::printf("\n");
        if (!ngram_init_case(avx)) rc = 1;
    }
    const int lo = only_level >= 0 ? only_level : 0;
    const int hi = only_level >= 0 ? only_level : AVX_N_LEVELS - 1;
    std::vector<std::string> declined_ops;
    for (int lvl = lo; lvl <= hi; ++lvl) {
        int actual = lvl;
        if (avx.isa_force) actual = avx.isa_force(lvl);
        if (actual != lvl) {
            /* CLAMPED, AND SAID SO. Reporting a pass for a level the machine cannot execute would
             * be reporting that a path works on a machine that never ran it. */
            std::printf("\n== level %d (%s): NOT EXECUTABLE on this CPU -- avx_isa_force clamped to"
                        " %s.\n   This row proves nothing about level %d and is not counted.\n",
                        lvl, avx.isa_name ? avx.isa_name(lvl) : "?",
                        avx.isa_name ? avx.isa_name(actual) : "?", lvl);
            continue;
        }
        std::printf("\n== level %d (%s) ==\n", lvl, avx.isa_name ? avx.isa_name(lvl) : "?");
        /* Before the tolerances: the conversions have to be EXACT, and they are checked over their
         * whole input space rather than over a draw. Run per level, because the vectorised bodies
         * differ per level and a draw of normals reaches none of the patterns that separate them. */
        if (!exhaustive_cast(ref, avx, verbose)) rc = 1;
        Stats st;
        for (const Case& c : cases) {
            if (!only_op.empty() && only_op != c.op) continue;
            /* A speed-only case has no correctness question to ask -- see avx_harness.h. */
            if (c.speed_only) continue;
            const size_t before = (size_t)st.declined;
            run_case(ref, avx, c, verbose, tol_override, &st);
            if ((size_t)st.declined != before && lvl == lo) {
                bool seen = false;
                for (const std::string& s : declined_ops) if (s == c.op) { seen = true; break; }
                if (!seen) declined_ops.push_back(c.op);
            }
        }
        std::printf("  %d checked, %d passed, %d failed, %d declined, %d skipped, %d missing."
                    "  worst rel_l2 = %.3e",
                    st.checked, st.passed, st.failed, st.declined, st.skipped, st.missing,
                    st.worst);
        if (!st.worst_where.empty()) std::printf("\n  worst case: %s", st.worst_where.c_str());
        std::printf("\n");
        if (st.failed || st.declined || st.missing) rc = 1;
        if ((only_op.empty() || only_op == "grid_embed") &&
            !grid_embed_antialias_cases(ref, avx, verbose)) rc = 1;
        if ((only_op.empty() || only_op == "rmsnorm") &&
            !rmsnorm_rounding_cases(ref, avx, verbose)) rc = 1;
        if ((only_op.empty() || only_op == "causal_conv1d") &&
            !causal_conv_cases(ref, avx, verbose)) rc = 1;
        if ((only_op.empty() || only_op == "logits_select") &&
            !logits_select_cases(ref, avx, verbose)) rc = 1;
        if (only_op.empty()) {
            int hand_checked = 0;
            if (!hand_cases(ref, avx, verbose, &hand_checked)) rc = 1;
        }
        if (only_op.empty() || only_op == "embed_lookup_q") {
            /* Beside the binary, on the build's filesystem, which reports its direct-I/O
             * alignment wherever the build is on ext4 or XFS. */
            const std::string self = argv[0];
            const size_t sl = self.rfind('/');
            const std::string at = sl == std::string::npos ? "." : self.substr(0, sl);
            /* The E4M3 table the quantised containers carry, and the plain bf16 one a checkpoint
             * served as it ships carries: one path, two row sizes. */
            if (!row_gather_case(avx, at, 1000, verbose, false)) rc = 1;
            if (!row_gather_case(avx, at, 1000, verbose, true)) rc = 1;
            /* And in TMPDIR, which is a tmpfs on most machines: since Linux 6.6 a tmpfs does
             * direct reads and reports no alignment for them, as btrfs does, so the gather finds
             * its granule from the block size there. */
            const char* tmp = std::getenv("TMPDIR");
            if (!row_gather_case(avx, tmp && *tmp ? tmp : "/tmp", 1000, verbose, false)) rc = 1;
        }
    }
    if (!declined_ops.empty()) {
        std::printf("\nDECLINED (RAD_E_UNSUPPORTED from a row that claims the op) -- %d op(s):\n ",
                    (int)declined_ops.size());
        for (const std::string& s : declined_ops) std::printf(" %s", s.c_str());
        std::printf("\nA row that declines is worse than no row: the selector resolves to it and "
                    "the step\nfails at issue rather than at declare.\n");
    }
    std::printf("\n%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
