/* rad-kbench -- falsify every resolved kernel against libref, per op, on the real container.
 *
 * This is the most important tool in the project. A from-scratch engine whose kernels cannot be
 * individually falsified is a project that ends with fluent wrong output and no way to bisect it
 * (spec §17): every component tests clean, the whole emits degenerate text, and there is nowhere
 * to look.
 *
 * The properties that make a per-kernel check worth trusting:
 *
 *   THE ARTIFACT, NOT A SYNTHETIC.   Weight operands come from the .rad, in the layout the
 *   resolved kernel asked for and rad-convert produced. A test that quantises in-process shares
 *   the quantiser, the layout and the geometry with the code under test and therefore proves too
 *   little.
 *
 *   FORMAT LATCHED FROM THE ARTIFACT.  The geometry every op is checked at comes from the
 *   container's metadata through the same declare phase the engine runs, not from a literal here.
 *
 *   SKIP RATHER THAN FAIL when a kernel legitimately does not serve a case, and say which.
 *
 *   PER-(OP, DTYPE) TOLERANCES. One threshold cannot govern 5-bit codebook experts, fp8 dense and
 *   bf16 dense alike: it is either loose enough to pass the first or tight enough to test the
 *   last, never both.
 *
 *   NON-FINITE IS A FAILURE, CHECKED FIRST.  A threshold written `rel > tol || rel < 0` is FALSE
 *   for NaN, so a NaN error passes such a check silently. Here the finiteness pattern of both
 *   sides is compared before any norm is computed.
 *
 *   PER-CASE SEEDS.  Drawing every input from one global generator makes what a case sees depend
 *   on which cases ran before it, so two runs with different flags are not comparable. Here the
 *   seed is a hash of (op, band, domain, operand), so a case is reproducible independently of
 *   every other case and of the flags.
 */
#include "kfixture.h"
#include "opshapes.h"
#include "format/encoding.h"

#include "rad_device.h"
#include "sample/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <map>
#include <set>
#include <unistd.h>

using namespace rad;

namespace {

/* ================================================================== error metrics */
struct Err {
    double max_abs = 0;
    double max_rel = 0;
    double rel_l2  = 0;
    int64_t n_nonfinite_mismatch = 0;   /* elements where one side is finite and the other is not */
    int64_t n = 0;
};

Err compare(const float* got, const float* ref, int64_t n) {
    Err e;
    e.n = n;
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        const bool fg = std::isfinite(got[i]), fr = std::isfinite(ref[i]);
        if (fg != fr) { ++e.n_nonfinite_mismatch; continue; }
        if (!fg) continue;
        const double d = (double)got[i] - (double)ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        const double scale = std::fabs((double)ref[i]);
        if (scale > 1e-30) e.max_rel = std::max(e.max_rel, std::fabs(d) / scale);
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    e.rel_l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    return e;
}

/* ================================================================== tolerances */
/* Keyed by (op, dtype), most specific first, "*" matching anything. The numbers are what a
 * DIFFERENT SUMMATION ORDER of the same arithmetic costs, not what a wrong kernel costs -- a
 * kernel that tiles a K=5120 reduction differently from ref's straight loop lands here, and a
 * kernel with a transposed operand lands nowhere near it.
 *
 * A table and not one constant, because no single threshold is simultaneously tight enough to
 * catch a bf16 rmsnorm and loose enough to pass a w4a8 GEMM. */
struct Tol { const char* op; const char* dtype; double rel_l2; };
const Tol g_tol[] = {
    /* Reductions over K: the error grows with the reduction length and the tiling order. */
    { "gemm_nt",      "bf16",     8e-3 },
    { "gemm_nt",      "f16",      4e-3 },
    { "gemm_nt",      "f32",      2e-5 },
    { "gemm_nt",      "*",        3e-2 },   /* w4a8, w4a16, mxfp4a8: the quant dominates */
    { "logits_gemm",  "bf16",     8e-3 },
    { "logits_gemm",  "f32",      2e-5 },
    { "moe_gemm",     "*",        3e-2 },
    /* Attention: a softmax over a long context plus a second reduction. */
    { "attn_",        "bf16",     2e-2 },
    { "attn_",        "f32",      1e-4 },
    { "attn_",        "*",        3e-2 },
    /* Elementwise and norms: one pass, so anything but a rounding difference is a bug. */
    { "rmsnorm",      "bf16",     4e-3 },
    { "rmsnorm",      "f32",      1e-6 },
    { "layernorm",    "f32",      1e-6 },
    { "softmax",      "f32",      1e-6 },
    /* Quantisation is lossy BY CONSTRUCTION, and what is being checked is that two
     * implementations of the same quantiser agree -- so this is tight, not loose. */
    { "quant_act",    "*",        1e-6 },
    { "dequant",      "*",        1e-6 },
    { "cast",         "*",        1e-6 },
    /* Collectives: `exact` means exact. */
    { "all_reduce",   "f32",      1e-6 },
    { "all_reduce",   "*",        4e-3 },
    { "all_gather",   "*",        0.0  },   /* a copy; bit-exact or broken */
    { "*",            "f32",      1e-5 },
    { "*",            "bf16",     6e-3 },
    { "*",            "f16",      3e-3 },
    { "*",            "*",        2e-2 },
};

/* PER OPERAND, consulted before the table above, for an output whose error is not the op's
 * arithmetic but a function of it.
 *
 * The fp8 twins of a computed row: E4M3 keeps three mantissa bits, so a rounding difference d in
 * the bf16 value being quantised moves about d / 2^-3 of the codes by one step of about 2^-3 -- a
 * rel_l2 near sqrt(d / 8), which is 2.7e-2 at the 6e-3 the row itself is held to and 1e-2 at the
 * 1e-3 a gated residual read typically lands at. The quantiser is not what differs; its input is.
 * A twin quantised from the wrong values or with the wrong rotation lands near 1, and the twins'
 * f32 scales stay under the row's own tolerance. */
struct OpdTol { const char* op; const char* operand; double rel_l2; };
const OpdTol g_opd_tol[] = {
    { "hc_read", "q",  3e-2 },
    { "hc_read", "rq", 3e-2 },
};

double tolerance_for(const std::string& op, const char* dtype, const char* operand) {
    if (operand)
        for (const OpdTol& t : g_opd_tol)
            if (op == t.op && std::strcmp(operand, t.operand) == 0) return t.rel_l2;
    const char* dt = dtype ? dtype : "*";
    double best = -1;
    for (const Tol& t : g_tol) {
        const bool op_match = std::strcmp(t.op, "*") == 0 || op.rfind(t.op, 0) == 0;
        const bool dt_match = std::strcmp(t.dtype, "*") == 0 || std::strcmp(t.dtype, dt) == 0;
        if (op_match && dt_match) { best = t.rel_l2; break; }
    }
    return best < 0 ? 2e-2 : best;
}


/* ================================================================== the run */
struct Stats { int checked = 0, passed = 0, failed = 0, skipped = 0, capped = 0, diverged = 0,
                overrun = 0, guarded = 0;
                double worst = 0; };

struct Row {
    std::string op, kernel, plugin, geom, dtype, choice, verdict;
    /* The band this point came from -- "M in (64, 512]" -- so a tail failure at the bottom of a
     * band reads as one instead of as an arbitrary M. */
    std::string band;
    const char* domain = "";
    double      rel_l2 = -1;
    double      tol = 0;
    double      us = -1;        /* < 0 when not benchmarked */
    /* How far the fastest and slowest timed rep were apart, as a percent of the median. A number
     * taken across a clock ramp is otherwise indistinguishable from a steady one. */
    double      us_spread = -1;
    /* THE ORACLE'S OWN TIME FOR THE SAME CASE, and it costs nothing to take: the reference launch
     * already happens on every checked case, so a clock around it turns the correctness pass into
     * a two-library speed comparison for free. It is ONE call, not a median of reps -- a host
     * gemm at n_vocab=248320 takes minutes and nobody is tuning libref -- and the report says so
     * rather than presenting it as the same kind of number as the device column. */
    double      ref_us = -1;
    int64_t     bytes = 0;      /* operand bytes touched once, for a GB/s that names its own basis */
    bool        synthesised = false;
};

/* WHAT ONE LIBRARY IS, INDEPENDENTLY OF THIS CONTAINER.
 *
 * Correctness and speed are per-case and come out of the run. COMPLETENESS is not: it is a
 * property of the library's row tables, it is answerable without a model, and answering it from
 * the run would be wrong -- an op this container never declares would count as a gap in every
 * library at once. So it is read off the registry, and the vocabulary it is measured against is
 * every op ANY loaded plugin declares a schema for (§2.3), which is the only definition of "the
 * whole op set" that does not privilege one library's idea of it. */
struct LibRow {
    std::string name, version, file, build_target;
    int  order = 0;
    bool oracle = false;
    /* op -> the domains this library has a row for it in. A map and not a set because the
     * completeness cell names the domain: "host" and "device" are different answers to "does
     * this library implement `gemm_nt`", and collapsing them to a tick loses the one that
     * matters -- a host-only row is the fallback, not an implementation of the fast path. */
    std::map<std::string, std::set<int>> ops;
    std::set<int>         domains;
    int  n_rows = 0;
    int  n_tunable_rows = 0;            /* rows declaring at least one axis */
    int  n_fusion_rows  = 0;            /* rows declaring `replaces` */
    int  n_layout_rows  = 0;            /* rows that re-lay a weight out */
    /* EVERY ROW BY NAME, SO THE REPORT CAN SAY WHICH ONES NEVER RAN.
     *
     * `n_rows` counts what a library declares and `cases` counts what happened, and the two
     * cannot be compared: one case exercises one row, several cases exercise one row, and a row
     * no case reaches contributes to neither number. A library can be at 100% of the vocabulary,
     * pass everything it ran and still have a third of its rows never once executed -- a
     * specialised band nothing in the fixture lands in, a dtype no container in scope uses, a
     * tail variant behind a constraint no recorded geometry satisfies.
     *
     * That row is the one that breaks, because nothing has run it since it was written. Named
     * here rather than counted, since "which" is the actionable half. */
    std::map<std::string, std::string> rows;   /* kernel name -> the op it serves */
    std::set<std::string> ran;                 /* kernel names a case actually reached */
    /* Filled from the run. */
    int    cases = 0, passed = 0, failed = 0;
    double worst = -1;
    std::vector<double> us;
};

/* An operand description in the form a reader can compare at a glance: `i8[56,5120]`. Used only
 * to say what two libraries disagree about, so it prints the two things they can disagree about
 * and nothing else. */
std::string shape_str(const OpdShape& d) {
    std::string s = rad_dtype_name(d.dtype);
    s += "[";
    for (size_t i = 0; i < d.shape.size(); ++i) {
        if (i) s += ",";
        s += fmt("%lld", (long long)d.shape[i]);
    }
    return s + "]";
}

std::string md_escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '|') o += "\\|"; else o += c; }
    return o;
}

const char* domain_short(int d) {
    switch (d) {
        case RAD_DOMAIN_HOST:   return "host";
        case RAD_DOMAIN_DEVICE: return "device";
        default:                return "?";
    }
}

std::string domains_str(const std::set<int>& d) {
    std::string s;
    for (int x : d) { if (!s.empty()) s += "+"; s += domain_short(x); }
    return s.empty() ? "-" : s;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/* ONE STREAM FOR THE WHOLE RUN, not one a case. The backend never reuses a queue id -- an event's
 * waiters name their queues by id -- so a stream a case exhausts the process's queues after about
 * a thousand cases and every case after that skips. What a kernel keeps per stream, a finisher's
 * arrival counter, is keyed by it too and would otherwise be taken afresh each case. A case drains
 * the stream before it frees anything the stream could still touch. */
RadStream run_stream() {
    static RadStream s = nullptr;
    if (!s && rad_stream_create(&s, 0) < 0) s = nullptr;
    return s;
}

/* ================================================================== the timed window
 *
 * WARM THE CLOCKS, NOT JUST THE CACHES, AND SAY HOW MUCH THE NUMBER MOVED.
 *
 * These cards sit in a low-power state whenever nothing is running, and the ramp back up is
 * longer than a short timed window: 300 launches of a 100 us kernel is about 30 ms, which is
 * still inside it, and an UNCHANGED kernel at an unchanged shape can take close to twice as long
 * after an idle gap as it does after other work. A harness that starts cold does not measure a
 * slow kernel -- it measures a sleeping card, and the failure is invisible, because the number is
 * plausible and it moves with the shape.
 *
 * A correctness check runs on the host between cases here, which is exactly the idle gap that
 * lets the clock fall back, so this tool needs the warm-up more than libr4d's selftest or the
 * tuner do.
 *
 * AND THE SPREAD IS REPORTED BESIDE THE MEDIAN, the way the tuner reports it. A median of three
 * reps taken across a ramp is a plausible number with nothing to mark it as unreliable; the
 * distance between the fastest and the slowest is what makes that visible without pretending the
 * warm-up removed it.
 *
 * Returns the median microseconds per launch, or < 0 if any launch refused the shape.
 */
double time_launch(const RadKernelInfo* info, RadArgs* a, RadStream stream,
                   int iters, int reps, int warm_ms, double* spread_pct) {
    if (spread_pct) *spread_pct = -1;
    if (!info || !info->launch || iters <= 0 || reps <= 0) return -1;

    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto w0 = now();
    while (std::chrono::duration<double, std::milli>(now() - w0).count() < (double)warm_ms) {
        for (int i = 0; i < iters; ++i) if (info->launch(a, stream) < 0) return -1;
        if (rad_stream_sync(stream) < 0) return -1;
    }

    std::vector<double> us;
    us.reserve((size_t)reps);
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = now();
        for (int it = 0; it < iters; ++it) if (info->launch(a, stream) < 0) return -1;
        /* THE CLOCK STOPS AFTER THE DEVICE DOES. Every launch is asynchronous, so reading the
         * clock without this sync measures the host's submission cost: sub-microsecond and flat
         * across every shape, which ranks kernels out of noise. */
        if (rad_stream_sync(stream) < 0) return -1;
        us.push_back(std::chrono::duration<double, std::micro>(now() - t0).count() / (double)iters);
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    if (spread_pct) *spread_pct = med > 0 ? 100.0 * (us.back() - us.front()) / med : 0.0;
    return med;
}


/* THE CENSUS, TAKEN OFF THE REGISTRY AND NOT OFF THE RUN.
 *
 * What a library implements is a property of its row tables. Reading it out of the run instead
 * would answer a different and much weaker question -- "what did this container happen to ask
 * for" -- and would report an op no model declares as a gap in every library at once. It also
 * has to work for a library that serves nothing here at all, which is exactly the library
 * somebody is about to debug. */
std::vector<LibRow> census(const Registry& reg, const std::string& ref_name) {
    std::map<std::string, LibRow> by_name;
    for (const Plugin& p : reg.plugins()) {
        if (!(p.kind & RAD_PLUGIN_KERNEL)) continue;
        LibRow l;
        l.name = p.name; l.version = p.version; l.file = p.file;
        l.build_target = p.build_target;
        l.order = p.order;
        l.oracle = p.name == ref_name;
        by_name[p.name] = std::move(l);
    }
    for (const KernelRow& r : reg.rows()) {
        auto it = by_name.find(r.plugin);
        if (it == by_name.end() || !r.info || !r.info->op) continue;
        LibRow& l = it->second;
        ++l.n_rows;
        l.ops[r.info->op].insert(r.info->domain);
        if (r.info->name) l.rows[r.info->name] = r.info->op;
        l.domains.insert(r.info->domain);
        if (r.info->n_tunables > 0) ++l.n_tunable_rows;
        if (r.info->replaces)       ++l.n_fusion_rows;
        if (r.info->layout)         ++l.n_layout_rows;
    }
    /* Hierarchy order, which is the order that decides selection and therefore the order a reader
     * of this report is thinking in. The oracle sits last whatever its order, because it is not
     * competing. */
    std::vector<LibRow> out;
    for (auto& [n, l] : by_name) out.push_back(std::move(l));
    std::sort(out.begin(), out.end(), [](const LibRow& a, const LibRow& b) {
        if (a.oracle != b.oracle) return b.oracle;
        if (a.order != b.order)   return a.order < b.order;
        return a.name < b.name;
    });
    return out;
}

/* What the run did, folded onto what the registries declare. Shared by the report and by the
 * line every pass prints, so the two cannot disagree about a coverage number. */
void fold_run(std::vector<LibRow>& libs, const std::vector<Row>& rows) {
    std::map<std::string, LibRow*> by_name;
    for (LibRow& l : libs) by_name[l.name] = &l;
    for (const Row& r : rows) {
        auto it = by_name.find(r.plugin);
        if (it == by_name.end()) continue;
        LibRow& l = *it->second;
        ++l.cases;
        if (!r.kernel.empty()) l.ran.insert(r.kernel);
        if (r.verdict == "ok") ++l.passed; else ++l.failed;
        l.worst = std::max(l.worst, r.rel_l2);
        if (r.us >= 0) l.us.push_back(r.us);
    }
}

/* THE COVERAGE LINE EVERY PASS PRINTS, whether or not anybody asked for a report.
 *
 * "468 checked, 420 passed" says nothing about the rows no case reached, and a reader has no
 * reason to suspect there are any -- the numbers are large and the failures are few. One line
 * with the ratio in it is what makes the question visible at all; `--report` names the rows. */
void print_row_coverage(std::vector<LibRow> libs, const std::vector<Row>& rows) {
    fold_run(libs, rows);
    size_t declared = 0, ran = 0;
    for (const LibRow& l : libs) { declared += l.rows.size(); ran += l.ran.size(); }
    if (declared == 0) return;
    std::printf("%zu of %zu declared kernel row(s) ran", ran, declared);
    std::string cold;
    for (const LibRow& l : libs) {
        const size_t missed = l.rows.size() - l.ran.size();
        if (!missed) continue;
        cold += (cold.empty() ? "" : ", ") + l.name + " " + std::to_string(missed);
    }
    if (cold.empty()) std::printf(" -- every row every loaded library declares\n");
    else std::printf("; not reached: %s (--report names them)\n", cold.c_str());
}

/* ================================================================== the report
 *
 * ONE DOCUMENT, EVERY LIBRARY, THREE QUESTIONS.
 *
 * A kernel library is worth having only if it is correct, complete enough to serve the graph and
 * faster than what it replaces -- and those three are not separable in practice. A library that
 * wins every benchmark on the eight ops it implements and falls back to the reference for the
 * ninth is slower end to end than one that is uniformly mediocre, and a per-op speed table alone
 * says the opposite. So the three are sections of ONE report out of ONE run, and the summary at
 * the top carries all three for each library side by side.
 *
 * ONE PASS, BOTH MEASUREMENTS. A correctness check and a benchmark set up exactly the same thing
 * -- the declared graph, the resolved kernel, the real container's weights in the layout that
 * kernel asked for, operands sized and filled per the op's own description -- and then one of
 * them compares the answer and the other times it. Doing that twice in two tools means two
 * chances for the two setups to differ, and a timing taken at a different extent from the
 * correctness check is a timing of something else.
 *
 * WHAT THE TIMING IS AND IS NOT. It is `iters` back-to-back launches on ONE set of operands,
 * median of `reps`, with the stream synced inside the timed region -- submission alone measures
 * the queue, not the kernel. Because the operands are reused, a weight small enough to sit in the
 * last-level cache is READ FROM CACHE on every launch after the first, so the derived GB/s is an
 * upper bound and not a serving number -- it can be around twice what the same kernel sustains in
 * a real step. Rotating buffers between launches is what defeats that; this tool does not, and
 * the report says so at the top of the table rather than in a footnote. These numbers rank
 * kernels against each other at real model shapes; they do not size a roofline.
 */
int write_report(const std::string& path, const std::vector<Row>& rows,
                 const std::map<std::string, int>& skipped,
                 const std::map<std::string, int>& fuse_skipped,
                 const std::map<std::string, int>& diverged,
                 const std::map<std::string, int>& overruns,
                 const std::string& analysis_path,
                 /* Where these numbers came from, in one line: the container they were measured
                  * against, or the fixture and when it was recorded. The two modes answer
                  * different questions and a report that did not say which is which invites
                  * somebody to read a replay as a container check. */
                 const std::string& provenance,
                 int fuse_passed, int fuse_failed,
                 const Stats& st, const std::string& model, const std::string& ref_name,
                 const std::string& libraries, const std::string& arch_plugin,
                 const std::vector<std::string>& graph_ops,
                 std::vector<LibRow> libs, bool benched) {
    /* ---------------------------------------------------------------- fold the run into libs */
    fold_run(libs, rows);

    /* The union of every op any loaded plugin declares a schema for: the vocabulary completeness
     * is measured against. */
    std::set<std::string> vocab;
    for (const LibRow& l : libs) for (const auto& [o, _d] : l.ops) vocab.insert(o);
    std::set<std::string> needed(graph_ops.begin(), graph_ops.end());

    std::string s;
    s += "# radiance kernel libraries\n\n";
    s += "Generated by `rad-kbench`. Every kernel every loaded library supplies, run on the same "
         "bytes\nand " + (benched ? std::string("timed in the same pass")
                                  : std::string("not timed (`--bench` was not given)")) +
         ". Correctness, completeness and speed are three readings of one\nrun, because they are "
         "not separable: a library that is fastest on the ops it has and absent\non the rest is "
         "slower end to end than one that is uniformly mediocre.\n\n";
    s += "- " + provenance + "\n";
    if (!arch_plugin.empty()) s += "- architecture plugin: `" + arch_plugin + "`\n";
    s += "- libraries: `" + libraries + "`\n";
    s += "- reference: `" + ref_name + "` -- every correctness number is relative to it, so it "
         "has none of its own\n";
    s += fmt("- %d case(s) checked, %d passed, %d failed, %d skipped, %d capped; worst rel_l2 "
             "%.3e\n", st.checked, st.passed, st.failed, st.skipped, st.capped, st.worst);
    if (st.diverged)
        s += fmt("- %d case(s) where two libraries describe the same output differently -- not a\n"
                 "  correctness result, see **Divergences** below\n", st.diverged);
    /* Only where the fusion oracle actually ran. A replay does not run it -- it needs the
     * kernel's own library's unfused chain at the same geometry, which is the container pass's
     * job -- and "0 fusions differ" would read as a result rather than as an absence. */
    if (fuse_passed || fuse_failed)
        s += fmt("- %d fusion(s) byte-identical to the chain they declare, %d differ\n",
                 fuse_passed, fuse_failed);
    s += "\n";

    /* ================================================================ 1. the summary */
    s += "## Libraries\n\n";
    s += "`ops` is how many of the " + fmt("%zu", vocab.size()) +
         "-op vocabulary this library has a row for, and `graph` how many of\nthe " +
         fmt("%zu", needed.size()) + " ops THIS container's declared graph asks for. The two "
         "differ on purpose: a library can\nbe complete for one model and full of holes for the "
         "next.\n\n";
    s += "`rows` is how many kernel rows the library declares and `run` how many of them a case "
         "in this\npass actually reached. The gap is the real coverage number: a library can be "
         "complete for the\nvocabulary, pass everything it ran, and still have rows nothing has "
         "executed since they were\nwritten -- see **Rows nothing ran** below for which.\n\n";
    s += "| library | version | domain | rows | run | ops | graph | cases | pass | fail | "
         "worst rel_l2 |";
    if (benched) s += " median us |";
    s += "\n|---|---|---|---|---|---|---|---|---|---|---|";
    if (benched) s += "---|";
    s += "\n";
    for (const LibRow& l : libs) {
        int in_graph = 0;
        for (const std::string& o : needed) if (l.ops.count(o)) ++in_graph;
        s += "| `" + md_escape(l.name) + "`" + (l.oracle ? " (oracle)" : "") + " | " +
             md_escape(l.version) + " | " + domains_str(l.domains) + " | " +
             fmt("%d", l.n_rows) + " | " +
             fmt("%zu of %zu", l.ran.size(), l.rows.size()) + " | " +
             fmt("%zu of %zu", l.ops.size(), vocab.size()) + " | " +
             fmt("%d of %zu", in_graph, needed.size()) + " | " +
             fmt("%d", l.cases) + " | " + fmt("%d", l.passed) + " | " +
             (l.failed ? fmt("**%d**", l.failed) : std::string("0")) + " | " +
             /* The oracle has a number here in a replay and none in a container pass, and which
              * it is follows from whether anything ran rather than from what it is called. */
             (l.worst >= 0 ? fmt("%.2e", l.worst)
                           : (l.oracle ? std::string("- (it is the oracle)")
                                       : std::string("-"))) + " |";
        if (benched) s += (l.us.empty() ? std::string(" - |") : fmt(" %.2f |",
                                                                   median_of(l.us)));
        s += "\n";
    }
    s += "\nRows declaring an optional hook, which is what separates a library that merely runs "
         "from one\nthe core can tune, fuse and check against a container:\n\n";
    s += "| library | tunable rows | fusion claims | weight layouts |\n|---|---|---|---|\n";
    for (const LibRow& l : libs)
        s += "| `" + md_escape(l.name) + "` | " + fmt("%d", l.n_tunable_rows) + " | " +
             fmt("%d", l.n_fusion_rows) + " | " + fmt("%d", l.n_layout_rows) + " |\n";

    /* ================================================================ 1b. rows nothing ran
     *
     * THE COMPLETENESS TABLE BELOW ANSWERS A WEAKER QUESTION THAN IT LOOKS LIKE IT DOES. It is
     * per OP, and one op is several rows: a band for short M and another for long, a bf16 row
     * beside an fp8 one, a tail variant behind a constraint. A tick there means at least one of
     * them exists, and a pass means at least one of them ran -- neither says anything about the
     * others.
     *
     * A row nothing ran is the row that breaks. It compiled, it is registered, it is selected on
     * a geometry no fixture in the tree contains, and the first thing that lands on it is a
     * container in production. Naming them is most of what this tool can do about that: the fix
     * is a fixture recorded from a model that reaches them, and the list is what says which model
     * that would have to be.
     */
    s += "\n## Rows nothing ran\n\n";
    {
        size_t declared = 0, ran = 0;
        for (const LibRow& l : libs) { declared += l.rows.size(); ran += l.ran.size(); }
        s += fmt("%zu of %zu declared row(s) were reached by a case in this pass.\n\n", ran,
                 declared);
        bool any = false;
        for (const LibRow& l : libs) {
            std::vector<std::pair<std::string, std::string>> missed;
            for (const auto& [kernel, op] : l.rows)
                if (!l.ran.count(kernel)) missed.push_back({ kernel, op });
            if (missed.empty()) continue;
            any = true;
            s += "`" + md_escape(l.name) + "` -- " + fmt("%zu", missed.size()) +
                 " row(s) not reached:\n\n| kernel | op |\n|---|---|\n";
            for (const auto& [kernel, op] : missed)
                s += "| `" + md_escape(kernel) + "` | `" + md_escape(op) + "` |\n";
            s += "\n";
        }
        if (!any) s += "Every row every loaded library declares ran at least once.\n";
    }
    /* ================================================================ 2. completeness */
    s += "\n## Completeness\n\n";
    s += "One row per op in the vocabulary. A cell names the DOMAIN the library serves that op in, "
         "or `-`\nwhere it has no row at all. `graph` marks the ops this container's declared "
         "program asks for --\nan empty cell on a marked row is a hole a real model falls into, "
         "and one on an unmarked row is\nnot.\n\n";
    s += "| op | graph |";
    for (const LibRow& l : libs) s += " " + md_escape(l.name) + " |";
    s += "\n|---|---|";
    for (size_t i = 0; i < libs.size(); ++i) s += "---|";
    s += "\n";
    for (const std::string& o : vocab) {
        s += "| `" + md_escape(o) + "` | " + (needed.count(o) ? "yes" : " ") + " |";
        for (const LibRow& l : libs) {
            const auto it = l.ops.find(o);
            if (it == l.ops.end()) { s += " - |"; continue; }
            /* The domains this library has a row for THIS op in, not its domains overall. */
            s += " " + domains_str(it->second) + " |";
        }
        s += "\n";
    }

    /* ================================================================ 3. correctness */
    /* ONE ROW PER DISTINCT CASE, NOT PER LAYER.
     *
     * A 64-layer model resolves the same kernel at the same geometry 64 times, and printing all of
     * them makes a 468-row table in which nothing stands out -- which is the opposite of what a
     * report is for. Group by everything that identifies the CASE (op, kernel, plugin, domain,
     * geometry, tuned point) and carry three numbers across the group: how many times it ran, the
     * WORST error any of them showed, and the MEDIAN time.
     *
     * Worst rather than mean for the error, because one layer disagreeing is the finding and an
     * average of 64 hides it. Median rather than mean for the time, for the ordinary reason. */
    struct Agg {
        Row     first;
        int     n = 0;
        double  worst_rel = -1;
        bool    any_fail = false;
        bool    any_synth = false;
        std::vector<double> us, ref_us;
        /* The WORST spread any rep of any run in the group showed, for the same reason the error
         * column carries the worst: one unstable measurement is the finding, and a mean of
         * sixty-four hides it. */
        double  worst_spread = -1;
    };
    std::map<std::string, Agg> agg;
    std::vector<std::string> order;
    for (const Row& r : rows) {
        const std::string key = r.op + "\x1f" + r.kernel + "\x1f" + r.plugin + "\x1f" + r.domain +
                                "\x1f" + r.geom + "\x1f" + r.choice;
        auto it = agg.find(key);
        if (it == agg.end()) { it = agg.emplace(key, Agg{}).first; it->second.first = r;
                               order.push_back(key); }
        Agg& a = it->second;
        ++a.n;
        a.worst_rel = std::max(a.worst_rel, r.rel_l2);
        a.any_fail  = a.any_fail  || r.verdict != "ok";
        a.any_synth = a.any_synth || r.synthesised;
        if (r.us >= 0) a.us.push_back(r.us);
        if (r.ref_us >= 0) a.ref_us.push_back(r.ref_us);
        a.worst_spread = std::max(a.worst_spread, r.us_spread);
    }

    s += "\n## Correctness\n\n";
    s += fmt("%zu distinct case(s) over %zu check(s). `n` is how many times the case ran -- one "
             "per layer,\ntypically -- and `rel_l2` is the WORST of them, because one layer "
             "disagreeing is the finding.\n`(synth w)` means this op's weights are not in the "
             "container and were drawn instead, so the\ncase proves the arithmetic and not the "
             "artifact.\n\n", order.size(), rows.size());
    s += "| op | kernel | library | domain | geometry | tuned | n | verdict | rel_l2 | tol |\n";
    s += "|---|---|---|---|---|---|---|---|---|---|\n";
    for (const std::string& key : order) {
        Agg& a = agg[key];
        const Row& r = a.first;
        s += "| `" + md_escape(r.op) + "` | `" + md_escape(r.kernel) + "` | " +
             md_escape(r.plugin) + " | " + r.domain + " | `" + md_escape(r.geom) + "` | " +
             md_escape(r.choice.empty() ? "-" : r.choice) + " | " + fmt("%d", a.n) + " | " +
             (a.any_fail ? "**FAIL**" : "ok") + (a.any_synth ? " (synth w)" : "") + " | " +
             (a.worst_rel < 0 ? "-" : fmt("%.2e", a.worst_rel)) + " | " +
             fmt("%.0e", r.tol) + " |\n";
    }

    /* ================================================================ 4. speed */
    if (benched) {
        s += "\n## Speed\n\n";
        s += "> **These times rank kernels; they do not size a roofline.** Every launch in a "
             "timed\n> run reuses one set of operands, so a weight that fits in the last-level "
             "cache is\n> read from cache after the first launch, and a bandwidth-bound kernel "
             "can read\n> several times its true rate. Ranking two kernels at one shape is what "
             "these times are\n> for; a roofline is not.\n>\n> `operands` is the total the case ALLOCATED, not what the kernel read: a "
             "gather\n> touches M rows of a table, not the table. Dividing the two columns "
             "produces a\n> bandwidth figure this tool cannot stand behind, which is why it does "
             "not print one.\n>\n> The `" + ref_name + "` column is ONE untimed-warmup call, not a "
             "median of reps: it is there to\n> show the order of magnitude a reference "
             "implementation costs, which is the reason the\n> device kernels exist. It is not a "
             "measurement of a library anyone tunes.\n>\n> `spread` is how far the fastest and slowest timed rep were apart, as a percent of\n> the median. Each case is warmed with untimed launches first, because these cards sit in\n> a low-power state whenever nothing is running and the ramp back up is longer than a\n> short timed window -- but a warm-up is not a guarantee, and a wide spread is what marks\n> a number the card was still ramping through. Read a row with a large one as an order of\n> magnitude, not a measurement.\n\n";
        s += "| op | kernel | library | geometry | us | spread | " + md_escape(ref_name) +
             " us | x | operands |\n|---|---|---|---|---|---|---|---|---|\n";
        /* Sorted by what the case costs, descending: the top of this table is where a decode step
         * actually goes, and an alphabetical one buries it. */
        std::vector<std::string> by_cost;
        for (const std::string& k : order) if (!agg[k].us.empty()) by_cost.push_back(k);
        std::sort(by_cost.begin(), by_cost.end(), [&](const std::string& a, const std::string& b) {
            return median_of(agg[a].us) > median_of(agg[b].us);
        });
        for (const std::string& key : by_cost) {
            Agg& a = agg[key];
            const Row& r = a.first;
            const double us = median_of(a.us);
            const double ru = median_of(a.ref_us);
            s += "| `" + md_escape(r.op) + "` | `" + md_escape(r.kernel) + "` | " +
                 md_escape(r.plugin) + " | `" + md_escape(r.geom) + "` | " +
                 fmt("%.2f", us) + " | " +
                 (a.worst_spread < 0 ? std::string("-") : fmt("%.0f%%", a.worst_spread)) + " | " +
                 (ru < 0 ? std::string("-") : fmt("%.0f", ru)) + " | " +
                 (ru < 0 || us <= 0 ? std::string("-") : fmt("%.0fx", ru / us)) + " | " +
                 (r.bytes > 0 ? rad_humanb_w(r.bytes, 0) : std::string("-")) + " |\n";
        }
    }

    /* ================================================================ 4b. memory safety */
    s += "\n## Memory safety\n\n";
    s += "Every device allocation in this run -- every operand and every kernel's own scratch -- "
         "was\nmade with a poisoned 256-byte margin on both sides, the kernel was handed the "
         "middle, and the\nmargins were read back and verified after the checked launch. It "
         "catches an out-of-bounds\nWRITE, which is the one that corrupts a neighbouring tensor "
         "and surfaces as wrong output three\nops later in a kernel that is correct. An "
         "out-of-bounds READ it cannot see.\n\n";
    s += "AMD's amdgcn AddressSanitizer (`/opt/rocm/amdgcn/bitcode/asanrtl.bc`) is not an option "
         "on every\ncard: its shadow is page-fault based and needs `xnack+`, which RDNA does not "
         "implement --\n`rocminfo` reports `XNACK enabled: NO` there. The margins are what "
         "replaces it.\n\n";
    s += fmt("- %d case(s) ran inside guarded allocations\n", st.guarded);
    s += fmt("- %d wrote outside one\n", st.overrun);
    if (!overruns.empty()) {
        s += "\n| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : overruns)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    /* ================================================================ 5. the gaps */
    s += "\n## Not covered\n\n";
    s += "A skip is a real answer: the tool declined rather than invent an extent, a dtype or an\n"
         "oracle it does not have. Each line below is a gap in COVERAGE, not a failing kernel.\n\n";
    /* THE COLLECTIVES, NAMED RATHER THAN MISSING. `all_reduce`, `all_gather` and the fused
     * all-reduce forms appear in the completeness matrix and in no case, and a reader who does
     * not know why will assume the tool covered them. It cannot: those kernels establish one
     * instance across ranks through a rendezvous over real peer memory, so a single process has
     * nobody to reduce with and a kernel that waits for a peer that never arrives hangs rather
     * than failing. Checking them needs a second rank, which needs a second process, which is a
     * different tool. */
    s += "**Collective ops are not checked here and cannot be.** `all_reduce`, `all_gather` and "
         "the\nfused all-reduce forms establish one instance across ranks through a rendezvous "
         "over peer\nmemory: a single process has nobody to reduce with, and a kernel waiting for "
         "a peer that\nnever arrives hangs rather than failing. They appear in the completeness "
         "matrix because the\nrows exist; they carry no correctness or speed number because "
         "producing one needs a second\nrank, and that is a different harness.\n\n";
    if (!skipped.empty()) {
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : skipped)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    if (!diverged.empty()) {
        s += "\n## Divergences\n\n";
        s += "Two libraries implementing the same op name and DESCRIBING AN OUTPUT DIFFERENTLY. "
             "This is\nnot a correctness result and these cases are counted apart from the passes "
             "and the failures:\nthe comparison would be between two different tensors, and a "
             "kernel that writes exactly what\nits own hook declares would be reported as "
             "catastrophically wrong. A divergence is a question\nfor the two libraries' authors "
             "-- one of them is the op and the other needs a different name --\nand it is a real "
             "finding, just not the one the tolerance column answers.\n\n";
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : diverged)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }
    if (!fuse_skipped.empty()) {
        s += "\n## Fusions not checked\n\n";
        s += "A kernel that declares `replaces` is run against that chain out of its OWN library "
             "and the\noutputs compared byte for byte -- a stronger claim than a tolerance against "
             "the oracle. These\nare the ones where the chain could not be assembled, named rather "
             "than counted.\n\n";
        s += "| what | cases |\n|---|---|\n";
        for (const auto& [what, n] : fuse_skipped)
            s += "| " + md_escape(what) + " | " + fmt("%d", n) + " |\n";
    }

    /* ================================================================ 6. what the analysers say
     *
     * Spliced verbatim from a file somebody else produced. rad-kbench does not run clang-tidy and
     * does not build a sanitised tree -- it runs on a card, the analysers run in CI, and the
     * report is one document because a reader wants one document and not because one process
     * should do both. A missing file is a missing section and not an error: the measurements
     * above are complete without it. */
    if (!analysis_path.empty()) {
        FILE* af = std::fopen(analysis_path.c_str(), "r");
        if (af) {
            s += "\n";
            char buf[4096];
            size_t got;
            while ((got = std::fread(buf, 1, sizeof buf, af)) > 0) s.append(buf, got);
            std::fclose(af);
        } else {
            RAD_WARN("--analysis %s: %s (the section is omitted)", analysis_path.c_str(),
                     std::strerror(errno));
        }
    }

    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { RAD_ERR("--report %s: %s", path.c_str(), std::strerror(errno)); return RAD_E_IO; }
    std::fwrite(s.data(), 1, s.size(), f);
    std::fclose(f);
    return RAD_OK;
}
/* A GUARDED DEVICE ALLOCATION: the real base, the pointer the kernel is handed 256 bytes into it,
 * and whether THIS case owns the allocation or borrowed it from the weight cache. At namespace
 * scope because the cache outlives any one case. */
struct DevBuf {
    void*   base = nullptr;
    void*   p = nullptr;
    int64_t bytes = 0;
    bool    on = false;      /* guarded, so its margins are worth reading back */
    bool    owned = true;    /* false: the cache frees it, not the case that used it */
    /* The description the relayout produced. Cached with the bytes because rebuilding it would
     * mean running the relayout to find out. */
    RadTensor desc{};
};


/* ONE KERNEL, ONE RECORDED CASE.
 *
 * Allocates, uploads, launches, compares against what the oracle left behind and times it -- the
 * same five steps the container pass takes, minus the two it cannot take here (a weight out of a
 * .rad, and an oracle run beside the kernel). Device allocations are guarded on both sides, for
 * the reason the container pass guards them: an out-of-bounds write is the one fault that shows
 * up as a pass.
 */
int replay_one(const KfCase& c, const Geometry& g, int64_t M,
               const std::vector<std::vector<uint8_t>>& in,
               const std::vector<std::vector<std::vector<uint8_t>>>& planes, const KernelRow* row,
               const std::string& lib, int dom, const RadOpSchema* schema,
               double tol_override, bool bench, int bench_iters, int bench_reps,
               int bench_warm_ms, bool verbose,
               Stats* st, std::vector<Row>* rows, std::map<std::string, int>* skipped,
               /* The device-side weight cache, owned by the replay driver and shared across every
                * case: a 206 MB upload repeated once per sweep point is most of a run. */
               std::map<std::string, DevBuf>* dcache, int64_t* dcache_bytes,
               int64_t dcache_budget,
               std::map<std::string, int>* overruns) {
    /* A RECORDING IS A STATEMENT ABOUT A SCHEMA, AND SCHEMAS MOVE. Operands are positional, so a
     * case recorded before an op gained an operand hands the kernel a short list: the kernel reads
     * its last operand past the end, sees nothing there, and refuses -- which this tool would
     * otherwise report as "a kernel refused a geometry its own constraints accept", blaming the
     * one component that is behaving correctly.
     *
     * The count is the whole check. Whether each operand still MEANS what it did is not knowable
     * from a recording, but a fixture that disagrees about how many there are is stale by
     * construction, and the answer is to re-record rather than to read it. */
    const size_t n_opd = c.opd.size();
    if (schema && (int)n_opd != schema->n_operands) {
        (*skipped)[c.op + "  (fixture recorded " + std::to_string(n_opd) + " operand(s), the "
                   "schema declares " + std::to_string(schema->n_operands) +
                   " -- re-record this fixture)"]++;
        ++st->skipped;
        return RAD_E_UNSUPPORTED;
    }
    std::vector<std::vector<uint8_t>> buf(in);      /* the kernel's own copy; it may write in place */
    std::vector<RadTensor> t(n_opd);
    for (size_t k = 0; k < n_opd; ++k) {
        if (c.opd[k].absent) { t[k] = RadTensor{}; continue; }
        RadTensor x{};
        x.data  = buf[k].data();
        x.dtype = c.opd[k].dtype;
        x.rank  = (uint32_t)c.opd[k].shape.size();
        for (size_t d = 0; d < c.opd[k].shape.size() && d < RAD_MAX_RANK; ++d)
            x.shape[d] = c.opd[k].shape[d];
        rad_tensor_pack(&x);
        t[k] = x;
    }

    /* EACH LIBRARY LAYS THE WEIGHT OUT ITS OWN WAY, FROM THE ONE SET OF PLANES -- BUT ONLY ONCE.
     *
     * `in` holds what the oracle read: each weight's planes decoded to the operand's tensor.
     * `planes` holds the planes themselves, which is what a container holds and what a library's
     * layout hooks arrange into the bytes its kernel reads -- here, per kernel. That is what lets
     * ONE recording serve two libraries whose fragment orders disagree, and why the fixture must
     * not store a laid-out plane: it would be storing one library's private arrangement and
     * calling it the reference.
     *
     * DEFERRED, because a relayout over a whole lm_head is not cheap and the sweep asks for the
     * same weight ten times. So the device cache is consulted FIRST, and on a hit neither the
     * relayout nor the upload happens at all: the bytes are already on the card from the first
     * point of the sweep, with the description they were built with. */
    auto lay_out = [&](size_t k, RadTensor* desc) -> int {
        RadLayout lay{};
        std::vector<uint8_t> laid;
        const int lrc = kf_lay_weight(row->info, g.params(), g.n_params(), (int)k, c.opd[k],
                                      planes[k], &lay, &laid);
        const char* opd_name = schema->operands[k].name ? schema->operands[k].name : "?";
        if (lrc == RAD_E_UNSUPPORTED && c.opd[k].sel.size() > 1) {
            (*skipped)[c.op + "  (" + lib + "/" + row->info->name + " reads '" + opd_name +
                       "' as stored, and it is " + std::to_string(c.opd[k].sel.size()) +
                       " planes -- a declaration the engine refuses)"]++;
            return RAD_E_SHAPE;
        }
        if (lrc == RAD_E_UNSUPPORTED) { *desc = t[k]; return RAD_OK; }
        if (lrc < 0) {
            (*skipped)[c.op + "  (" + lib + "/" + row->info->name + " cannot lay out '" +
                       opd_name + "' from " + enc_name(c.opd[k].enc) + ": " +
                       rad_strerror(lrc) + ")"]++;
            return lrc;
        }
        buf[k].swap(laid);
        kf_describe_laid(c.opd[k], lay, buf[k].data(), desc);
        return RAD_OK;
    };

    constexpr int64_t RED_BYTES = 256;
    constexpr int     RED_FILL  = 0xA5;
    std::vector<DevBuf> dev(n_opd);
    DevBuf scr;
    RadStream stream = nullptr;
    std::vector<uint8_t> host_scratch;

    auto guarded = [&](DevBuf& d, int64_t bytes) -> bool {
        d.bytes = bytes;
        d.base  = rad_dev_alloc(bytes + 2 * RED_BYTES, RAD_MEM_DEVICE);
        if (!d.base) return false;
        d.p = (uint8_t*)d.base + RED_BYTES;
        d.on = true;
        return rad_memset_async(d.base, RED_FILL, RED_BYTES, stream) >= 0 &&
               rad_memset_async((uint8_t*)d.p + bytes, RED_FILL, RED_BYTES, stream) >= 0;
    };
    auto release = [&]() {
        if (stream) (void)rad_stream_sync(stream);
        for (auto& d : dev) if (d.base && d.owned) rad_dev_free(d.base, RAD_MEM_DEVICE);
        if (scr.base) rad_dev_free(scr.base, RAD_MEM_DEVICE);
    };

    if (dom == RAD_DOMAIN_DEVICE) {
        if (!(stream = run_stream())) { ++st->skipped; return RAD_E_DEVICE; }
        for (size_t k = 0; k < n_opd; ++k) {
            if (c.opd[k].absent || buf[k].empty()) continue;

            /* A WEIGHT IS LAID OUT, UPLOADED AND DESCRIBED ONCE PER (LIBRARY, SHAPE).
             *
             * The key is the library's name with the weight's own seed, because two libraries lay
             * the same logical weight out differently and what is cached is the LAID-OUT plane.
             * The description is cached with it: it is the one the relayout produced, and
             * rebuilding it would mean running the relayout to find out.
             *
             * The red zone survives the sharing and still means what it did. A weight operand is
             * read-only, so the only thing that can disturb its margins is a kernel writing past
             * one, which is exactly what is being looked for -- and it is checked after every
             * launch, not only after the one that filled the cache. */
            if (c.opd[k].role == RAD_OPD_WEIGHT) {
                /* THE KERNEL, NOT JUST THE LIBRARY. Two kernels in one library can want the same
                 * logical weight laid out differently -- a fragment-order fp8 GEMM and a row-major
                 * one for the same shapes -- so a key that stopped at the library name would hand
                 * the second kernel the first one's permutation. That is not a slow test, it is a
                 * wrong one, and it surfaces as a kernel refusing a dtype only in a FULL run: in
                 * isolation nothing has filled the cache with the other kernel's plane. */
                const std::string dk = lib + "\x1f" + row->info->name + "\x1f" +
                                       std::to_string(kf_weight_seed(c.opd[k], k)) + "\x1f" +
                                       std::to_string(in[k].size());
                auto it = dcache->find(dk);
                if (it != dcache->end()) {
                    dev[k] = it->second;
                    dev[k].owned = false;            /* the cache frees it, not this case */
                    t[k] = it->second.desc;
                    t[k].data = dev[k].p;
                    continue;
                }
                const int lrc = lay_out(k, &t[k]);
                if (lrc < 0) { ++st->skipped; release(); return lrc; }
                if (!guarded(dev[k], (int64_t)buf[k].size())) {
                    (*skipped)[c.op + "  (no device memory for this case)"]++;
                    ++st->skipped; release(); return RAD_E_NOMEM;
                }
                if (rad_memcpy_async(dev[k].p, buf[k].data(), dev[k].bytes, stream) < 0) {
                    ++st->skipped; release(); return RAD_E_DEVICE;
                }
                t[k].data = dev[k].p;
                if (*dcache_bytes + dev[k].bytes <= dcache_budget) {
                    *dcache_bytes += dev[k].bytes;
                    dev[k].owned = false;
                    dev[k].desc  = t[k];
                    dcache->emplace(dk, dev[k]);
                }
                continue;
            }

            if (!guarded(dev[k], (int64_t)buf[k].size())) {
                (*skipped)[c.op + "  (no device memory for this case)"]++;
                ++st->skipped; release(); return RAD_E_NOMEM;
            }
            if (rad_memcpy_async(dev[k].p, buf[k].data(), dev[k].bytes, stream) < 0) {
                ++st->skipped; release(); return RAD_E_DEVICE;
            }
            t[k].data = dev[k].p;
        }
        if (rad_stream_sync(stream) < 0) { ++st->skipped; release(); return RAD_E_DEVICE; }
    }
    if (dom != RAD_DOMAIN_DEVICE) {
        /* No card, no cache: lay the weights out in place. */
        for (size_t k = 0; k < n_opd; ++k) {
            if (c.opd[k].absent || c.opd[k].role != RAD_OPD_WEIGHT) continue;
            const int lrc = lay_out(k, &t[k]);
            if (lrc < 0) { ++st->skipped; release(); return lrc; }
        }
    }

    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size();
    a.p = g.params(); a.n_p = g.n_params();
    a.rank = 0; a.world_size = 1;
    if (row->info->scratch) {
        const int64_t need = row->info->scratch(&a);
        if (need > 0) {
            if (dom == RAD_DOMAIN_DEVICE) {
                if (!guarded(scr, need)) { ++st->skipped; release(); return RAD_E_NOMEM; }
                rad_memset_async(scr.p, 0, need, stream);
                a.scratch = scr.p;
            } else {
                host_scratch.assign((size_t)need, 0);
                a.scratch = host_scratch.data();
            }
            a.scratch_bytes = need;
        }
    }

    const int s = row->info->launch(&a, dom == RAD_DOMAIN_DEVICE ? stream : nullptr);
    const int sync = (s >= 0 && dom == RAD_DOMAIN_DEVICE) ? rad_stream_sync(stream) : RAD_OK;
    if (s < 0 || sync < 0) {
        /* A kernel refusing a geometry its own constraints accepted is a bug in the ROW, not a
         * runtime condition (spec §17): this tool asked exactly what the constraints said it
         * could serve. */
        std::printf("  %-22s %-44s FAIL  %s/%s refused a geometry its constraints accept: %s\n",
                    c.op.c_str(), g.str().c_str(), lib.c_str(), row->info->name,
                    rad_strerror(s < 0 ? s : sync));
        ++st->failed;
        release();
        return s < 0 ? s : sync;
    }

    /* The outputs back on the host BEFORE anything is timed: many ops are in place, and a timed
     * loop of twenty launches would leave `rope` rotated twenty-one times and report a large
     * error on a correct kernel. */
    if (dom == RAD_DOMAIN_DEVICE) {
        for (size_t k = 0; k < n_opd; ++k) {
            if (!dev[k].on) continue;
            const int role = schema->operands[k].role;
            if (role == RAD_OPD_OUT || role == RAD_OPD_INOUT)
                rad_memcpy_async(buf[k].data(), dev[k].p, dev[k].bytes, stream);
        }
        rad_stream_sync(stream);

        auto margin = [&](const DevBuf& d, const std::string& what) {
            if (!d.on) return;
            uint8_t lo[RED_BYTES], hi[RED_BYTES];
            if (rad_memcpy_async(lo, d.base, RED_BYTES, stream) < 0) return;
            if (rad_memcpy_async(hi, (const uint8_t*)d.p + d.bytes, RED_BYTES, stream) < 0) return;
            if (rad_stream_sync(stream) < 0) return;
            int64_t before = 0, after = 0;
            for (int64_t i = 0; i < RED_BYTES; ++i) {
                if (lo[i] != (uint8_t)RED_FILL) before = std::max(before, RED_BYTES - i);
                if (hi[i] != (uint8_t)RED_FILL) after  = std::max(after, i + 1);
            }
            if (!before && !after) return;
            std::string m = c.op + "  (" + lib + "/" + row->info->name + " wrote ";
            if (before) m += fmt("%lld byte(s) BEFORE the start", (long long)before);
            if (before && after) m += " and ";
            if (after)  m += fmt("%lld byte(s) PAST the end", (long long)after);
            m += " of " + what + ")";
            (*overruns)[m]++;
            ++st->overrun;
            std::printf("  %-22s %-44s OUT OF BOUNDS: %s\n", c.op.c_str(), g.str().c_str(),
                        m.c_str());
        };
        for (size_t k = 0; k < n_opd; ++k)
            if (dev[k].on)
                margin(dev[k], std::string("operand '") +
                               (schema->operands[k].name ? schema->operands[k].name : "?") + "'");
        margin(scr, "its own scratch");
        ++st->guarded;
    }

    /* ---------------------------------------------------------------- against the recording */
    /* EACH OPERAND AGAINST ITS OWN TOLERANCE, and the line names the worst: a failing operand
     * before a passing one, the larger error between two of the same verdict. */
    double worst_rel = -1, worst_abs = 0, worst_max_rel = 0, worst_tol = 0;
    bool worst_bad = false, all_ok = true;
    int64_t nonfinite_gap = 0;
    int n_out = 0;
    std::string per_opd;
    for (size_t k = 0; k < n_opd; ++k) {
        const KfOperand& d = c.opd[k];
        if (d.absent || !d.have_ref) continue;
        const int role = schema->operands[k].role;
        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
        const int64_t n = d.ref_n;
        std::vector<float> got;
        rad_widen(buf[k].data(), d.dtype, n, &got);
        if ((int64_t)got.size() < n) continue;

        /* THE SAMPLE GIVES THE ERROR AND THE NORM CATCHES WHAT THE SAMPLE MISSES.
         *
         * A kernel that writes a correct first tile and leaves the rest untouched agrees with the
         * reference at most sampled positions and has the wrong sum of squares, by a lot -- so the
         * two together are much harder to fool than either. The norm is reported as a relative
         * difference and folded into the same verdict, because "the values I looked at are right
         * and the total energy is half" is a failure however it is phrased. */
        double ss = 0;
        int64_t nf = 0;
        std::vector<float> mine;
        kf_reduce(got.data(), n, kf_operand_seed(c.seed, 0x5000 + (int)k), &ss, &nf, &mine);
        double num = 0, den = 0, mx = 0, mxr = 0;
        const size_t ns = std::min(mine.size(), d.sample.size());
        for (size_t i = 0; i < ns; ++i) {
            const bool fa = std::isfinite(mine[i]), fb = std::isfinite(d.sample[i]);
            if (fa != fb) { ++nonfinite_gap; continue; }
            if (!fa) continue;
            const double diff = (double)mine[i] - (double)d.sample[i];
            mx = std::max(mx, std::fabs(diff));
            const double sc = std::fabs((double)d.sample[i]);
            if (sc > 1e-30) mxr = std::max(mxr, std::fabs(diff) / sc);
            num += diff * diff;
            den += (double)d.sample[i] * (double)d.sample[i];
        }
        double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
        /* The norm disagreement, on the same scale as rel_l2 so one column carries both. */
        const double rn = d.ref_sumsq > 0
                              ? std::fabs(std::sqrt(ss) - std::sqrt(d.ref_sumsq)) /
                                    std::sqrt(d.ref_sumsq)
                              : (ss > 0 ? 1.0 : 0.0);
        rel = std::max(rel, rn);
        nonfinite_gap += std::llabs((long long)(nf - d.ref_nonfinite));
        const char* nm = schema->operands[k].name ? schema->operands[k].name : "?";
        per_opd += fmt("%s%s=%.3e", per_opd.empty() ? "" : " ", nm, rel);
        const double tk = tol_override >= 0 ? tol_override
                                            : tolerance_for(c.op, g.get_s("dtype"), nm);
        /* Written so NaN FAILS. The natural spelling of the opposite test, `rel > tol || rel < 0`,
         * is FALSE for NaN and lets a NaN error through as a pass. */
        const bool bad = !(rel >= 0 && rel <= tk);
        all_ok = all_ok && !bad;
        if (worst_rel < 0 || (bad && !worst_bad) || (bad == worst_bad && rel > worst_rel)) {
            worst_rel = rel; worst_abs = mx; worst_max_rel = mxr; worst_tol = tk; worst_bad = bad;
        }
        ++n_out;
    }
    if (n_out == 0) { ++st->skipped; release(); return RAD_OK; }

    const double tol = worst_tol;
    const bool ok = nonfinite_gap == 0 && all_ok;
    ++st->checked;
    st->worst = std::max(st->worst, worst_rel);
    if (ok) ++st->passed; else ++st->failed;

    /* ---------------------------------------------------------------- and the clock */
    double case_us = -1, case_spread = -1;
    if (bench && dom == RAD_DOMAIN_DEVICE && bench_iters > 0)
        case_us = time_launch(row->info, &a, stream, bench_iters, bench_reps, bench_warm_ms,
                              &case_spread);
    release();

    Row rw;
    rw.op      = c.op;
    rw.kernel  = row->info->name;
    rw.plugin  = lib;
    rw.geom    = g.str();
    rw.domain  = domain_name(dom);
    rw.verdict = ok ? "ok" : "**FAIL**";
    rw.rel_l2  = worst_rel;
    rw.tol     = tol;
    rw.us      = case_us;
    rw.us_spread = case_spread;
    rw.band    = c.band;
    for (const KfOperand& d : c.opd)
        if (!d.absent) rw.bytes += rad_dtype_bytes(d.dtype, rad_numel(d.shape));
    rows->push_back(std::move(rw));

    if (!ok || verbose)
        std::printf("  %-22s %-44s %-8s %-28s rel=%.3e max_abs=%.3e max_rel=%.3e tol=%.1e %s\n",
                    c.op.c_str(), g.str().c_str(), domain_name(dom), row->info->name,
                    worst_rel, worst_abs, worst_max_rel, tol, ok ? "ok" : "FAIL");
    if (!ok && n_out > 1) std::printf("      per operand: %s\n", per_opd.c_str());
    return RAD_OK;
}

/* ================================================================== replay
 *
 * EVERY LIBRARY, EVERY RECORDED CASE, NO MODEL AND NO ORACLE IN THE ROOM.
 *
 * This is the mode that answers the question the report is about, and it is structurally
 * different from the container pass above rather than a flag on it. There is no declare phase,
 * because the geometries are in the file. There is no selection, because selection returns ONE
 * kernel and the whole point here is to run every library that has a row for the case -- a
 * hierarchy hides the loser, and the loser is half of a comparison. And there is no oracle run,
 * because libref already ran, once, when the fixture was recorded.
 *
 * What that buys is comparability: a library's numbers stand beside another library's, because
 * both were handed the same bytes and measured against the same recorded answer, on the same
 * card, in the same pass.
 */
/* The library the others are checked against: the one --ref names, else the first kernel library
 * that declared itself a reference implementation (RadPluginReferenceFn) -- by that declaration,
 * never by name, so the library radiance ships is found exactly as one a third party wrote. */
std::string resolve_reference(const Registry& reg, const std::string& named) {
    if (!named.empty()) return named;
    for (const Plugin& p : reg.plugins())
        if (p.kind == RAD_PLUGIN_KERNEL && p.reference) return p.name;
    RAD_ERR("no kernel library on the search path declares itself a reference implementation, so "
            "there is nothing to check against; name one with --ref");
    return std::string();
}

int replay(const Kfixture& fx, const std::string& home,
           const std::vector<std::string>& hierarchy, const std::string& ref_arg,
           const std::vector<std::string>& op_filters, double tol_override, int max_cases,
           int64_t max_case_bytes,
           bool bench, int bench_iters, int bench_reps, int bench_warm_ms, bool verbose,
           const std::string& report_path, const std::string& fixture_path,
           const std::string& analysis_path) {
    /* The plugins, and NOT an architecture: a kernel library is checked without one, which is
     * also the check that it does not secretly need one. Arch id and quant are empty for the same
     * reason -- there is no container to take them from. */
    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, "", "", &plugins) < 0) return 1;
    Registry& reg = rad_tools_registry();
    const std::string ref_name = resolve_reference(reg, ref_arg);
    if (ref_name.empty()) return 1;

    std::printf("rad-kbench %s\n", fixture_path.c_str());
    std::printf("  fixture      %zu case(s), recorded %s from %s\n", fx.cases.size(),
                fx.recorded.empty() ? "?" : fx.recorded.c_str(),
                fx.source.empty() ? "?" : fx.source.c_str());
    std::printf("  reference    %s, %d sampled position(s) an operand plus the full norm\n",
                fx.oracle.c_str(), KF_SAMPLES);
    std::printf("  libraries    %s\n\n", plugins.libraries().c_str());

    /* THE HIGHEST-PRIORITY ROW ONE LIBRARY HAS FOR THIS CASE. Not `Registry::select`, which
     * applies the hierarchy and would answer with whichever library won -- here every library is
     * asked separately and a library with no row is a COMPLETENESS result, not an absence. */
    auto row_in = [&](const std::string& plugin, const std::string& op, const Geometry& g,
                      int domain) -> const KernelRow* {
        const KernelRow* best = nullptr;
        for (const KernelRow& r : reg.rows()) {
            if (r.plugin != plugin || !r.info || op != r.info->op) continue;
            if (r.info->domain != domain || !r.info->launch) continue;
            bool ok = true;
            for (int c = 0; c < r.info->n_constraints && ok; ++c)
                ok = constraint_holds(r.info->constraints[c], g);
            if (!ok) continue;
            if (!best || r.info->priority > best->info->priority) best = &r;
        }
        return best;
    };

    Stats st;
    std::map<std::string, int> skipped_ops, overruns, diverged, fuse_skipped;
    std::map<std::string, int> group_seen;
    std::vector<Row> report_rows;
    /* Drawn weights, keyed by kf_weight_seed, and the same weights on the CARD per library --
     * each lays them out differently. Both are caches and not requirements: past the budget they
     * stop growing and the misses simply draw or upload again, which costs time and never
     * correctness.
     *
     * Both budgets follow --max-bytes rather than standing at a fixed size of their own, so ONE
     * knob bounds what a pass holds. A per-case cap beside caches that keep growing bounds
     * nothing: the run still arrives at the same total, one case at a time. The device cache is
     * freed at the end of the run rather than per case -- see the comment at the upload. */
    std::map<uint64_t, std::pair<std::vector<uint8_t>, std::vector<std::vector<uint8_t>>>> wcache;
    int64_t wcache_bytes = 0;
    const int64_t wcache_budget = max_case_bytes / 2;
    std::map<std::string, DevBuf> dcache;
    int64_t dcache_bytes = 0;
    const int64_t dcache_budget = max_case_bytes;

    /* WHICH LIBRARIES ARE UNDER TEST, AND WHY THE REFERENCE IS ONE OF THEM HERE.
     *
     * The container pass excludes the oracle, and must: it runs libref beside the kernel on the
     * same inputs, so libref would be scored against itself and report a perfect zero.
     *
     * A REPLAY HAS NO ORACLE IN THE ROOM. The answers come from a recording checked into the
     * tree, made by a different build, which is an independent artifact -- so libref against it
     * is a real comparison, and the only regression test the oracle has. It is not a small thing
     * to leave out: every correctness number this project reports
     * about any kernel is defined relative to libref, and a drift in libref's arithmetic moves
     * the reference every other measurement is quoted against.
     *
     * It also means a replay with no device library loaded still checks something, rather than
     * refusing with nothing to do -- which is the position a build machine with no card is in,
     * and it is the machine where a regression in a host implementation is most likely to land. */
    std::vector<std::string> libs;
    for (const Plugin& p : reg.plugins())
        if (p.kind & RAD_PLUGIN_KERNEL) libs.push_back(p.name);
    if (libs.empty()) {
        RAD_ERR("no kernel library loaded from %s/kernels", home.c_str());
        return 1;
    }

    for (const KfCase& c : fx.cases) {
        if (!op_filters.empty()) {
            bool hit = false;
            for (const std::string& f : op_filters) if (c.op.rfind(f, 0) == 0) { hit = true; break; }
            if (!hit) continue;
        }
        const RadOpSchema* schema = reg.schema(c.op);
        if (!schema) { skipped_ops[c.op + "  (no plugin declares a schema for it)"]++;
                       ++st.skipped; continue; }
        /* A CASE THAT DOES NOT FIT IS REFUSED BY NAME RATHER THAN BY THE OOM KILLER.
         *
         * The replay needs this bound even more than the container pass does, because the replay
         * is the pass that runs in ctest on an arbitrary machine. One case holds every operand as
         * host floats, again as the kernel's dtype, and again per library on the device; a 27B
         * lm_head weight alone is over a gigabyte, so a small machine dies at exit 137 with two
         * lines of log and no indication which op it died on. --max-bytes moves the line. The same
         * three-copies-of-every-operand estimate as the container pass, so the two agree about
         * what a case costs. */
        {
            int64_t want = 0;
            for (const KfOperand& d : c.opd)
                if (!d.absent) want += 3 * rad_dtype_bytes(d.dtype, rad_numel(d.shape));
            if (want > max_case_bytes) {
                skipped_ops[c.op + "  (one case needs " + humanb(want) + ", over the " +
                            humanb(max_case_bytes) + " budget -- raise --max-bytes to check it)"]++;
                ++st.skipped;
                continue;
            }
        }


        Geometry g;
        for (const KfParam& p : c.params) {
            if (p.kind == RAD_P_INT)      g.set_i(p.key, p.i);
            else if (p.kind == RAD_P_F64) g.set_f(p.key, p.d);
            else if (p.kind == RAD_P_STR) g.set_s(p.key, p.s);
        }
        /* The ranged key's value, read off the recording rather than guessed at from the
         * geometry: the key it lives under is the op's own, and a reader trying "M" then "q_len"
         * would guess wrong for exactly the ops whose ranged key is unusual. */
        const int64_t M = c.m;

        /* THE INPUTS, ONCE, SHARED BY EVERY LIBRARY. This is the whole reason a replay can
         * compare two libraries at all: both are handed the same bytes, drawn by the same code
         * from the seed the file carries. */
        std::vector<std::vector<uint8_t>> in(c.opd.size());
        std::vector<std::vector<std::vector<uint8_t>>> planes(c.opd.size());
        const int64_t vocab_rows = kf_vocab_rows(c.opd);
        std::string undecodable;
        for (size_t k = 0; k < c.opd.size(); ++k) {
            /* WEIGHTS ARE DRAWN ONCE AND KEPT. The M sweep asks for the same (N, K) weight at
             * every query length in a band, and kf_weight_seed makes those asks identical by
             * construction -- so the second one is a map lookup instead of a Box-Muller over 178
             * million elements. This is the difference between a replay that takes minutes and
             * one somebody runs in ctest.
             *
             * Bounded, because "keep every weight this fixture mentions" is several gigabytes:
             * once past the budget the cache stops growing and the misses simply draw, which
             * costs time and never correctness. */
            const KfOperand& d = c.opd[k];
            if (d.role == RAD_OPD_WEIGHT && !d.absent) {
                /* THE DRAW IS SEEDED BY THE SHAPE, THE CACHE IS KEYED BY WHAT WAS DRAWN. Two
                 * weights of one shape and dtype are the same draw -- that is the point of the
                 * seed -- but not the same planes when their encodings differ: a plain bf16
                 * embedding and a two-plane fp8 lm_head of [vocab, n_embd] share a seed, and
                 * handing the second the first's one plane sends its relayout past the end of a
                 * buffer that is not there. So the key adds the encoding and the selection. */
                const uint64_t wk = kf_weight_seed(d, k);
                uint64_t ck = wk;
                if (d.has_enc) {
                    const uint8_t* e = (const uint8_t*)&d.enc;
                    for (size_t i = 0; i < sizeof d.enc; ++i) ck = (ck ^ e[i]) * 0x100000001b3ull;
                    for (int s : d.sel) ck = (ck ^ (uint64_t)(uint32_t)s) * 0x100000001b3ull;
                    uint64_t mix = ck ^ 0x5bd1e995ull;
                    ck = rad_splitmix(mix);
                }
                auto it = wcache.find(ck);
                if (it != wcache.end()) {
                    in[k] = it->second.first;
                    planes[k] = it->second.second;
                    continue;
                }
                if (d.has_enc) {
                    kf_draw_planes(d, wk, &planes[k]);
                    std::string why;
                    if (kf_planes_logical(d, planes[k], &in[k], &why) != RAD_OK) {
                        undecodable = enc_name(d.enc) + ": " + why;
                        break;
                    }
                } else {
                    kf_draw(c.opd, k, M, vocab_rows, 0, &in[k]);
                }
                int64_t held = (int64_t)in[k].size();
                for (const auto& pl : planes[k]) held += (int64_t)pl.size();
                if (wcache_bytes + held <= wcache_budget) {
                    wcache_bytes += held;
                    wcache.emplace(ck, std::make_pair(in[k], planes[k]));
                }
                continue;
            }
            kf_draw(c.opd, k, M, vocab_rows, kf_operand_seed(c.seed, (int)k), &in[k]);
        }
        if (!undecodable.empty()) {
            skipped_ops[c.op + "  (a weight's planes do not make the operand the reference reads, "
                        "" + undecodable + ")"]++;
            ++st.skipped;
            continue;
        }

        for (const std::string& lib : libs) {
            for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
                const KernelRow* row = row_in(lib, c.op, g, dom);
                if (!row) continue;
                if (max_cases > 0) {
                    const std::string gk = c.op + "\x1f" + row->info->name + "\x1f" + lib +
                                           "\x1f" + std::to_string(dom) + "\x1f" + g.str();
                    if (++group_seen[gk] > max_cases) { ++st.capped; continue; }
                }
                /* THE DIVERGENCE THE RECORDING NAMES, NAMED AGAIN. A case is recorded once, from
                 * the oracle's description, and replayed against every library's row -- including
                 * the one the recording found describing an output differently and set apart.
                 * Replayed without this check that row reads as a kernel that is catastrophically
                 * wrong (quant_act_i8: libr4d writes one scale a row into a plane described with
                 * one a group). The recorded operands are the oracle's description, so the
                 * comparison is the one the recording made. */
                if (row->info->opd_shape) {
                    std::vector<OpdShape> ksh;
                    std::string why;
                    if (rad_operand_shapes_hook(row->info, g, schema->n_operands, &ksh) >= 0 &&
                        ksh.size() == c.opd.size()) {
                        for (size_t k = 0; k < ksh.size() && why.empty(); ++k) {
                            const int role = schema->operands[k].role;
                            if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                            if (c.opd[k].absent || ksh[k].absent) continue;
                            if (c.opd[k].dtype == ksh[k].dtype && c.opd[k].shape == ksh[k].shape)
                                continue;
                            OpdShape rec;
                            rec.dtype = c.opd[k].dtype;
                            rec.shape = c.opd[k].shape;
                            const char* nm = schema->operands[k].name
                                                 ? schema->operands[k].name : "?";
                            why = std::string("'") + nm + "': the recording says " +
                                  shape_str(rec) + ", " + row->info->name + " says " +
                                  shape_str(ksh[k]);
                        }
                    }
                    if (!why.empty()) {
                        diverged[c.op + "  (" + lib + "/" + row->info->name + " describes an "
                                 "output differently -- " + why + ")"]++;
                        ++st.diverged;
                        continue;
                    }
                }
                const int rc = replay_one(c, g, M, in, planes, row, lib, dom, schema, tol_override,
                                          bench, bench_iters, bench_reps, bench_warm_ms, verbose,
                                          &st, &report_rows, &skipped_ops,
                                          &dcache, &dcache_bytes, dcache_budget, &overruns);
                (void)rc;
            }
        }
    }

    for (auto& [k2, d] : dcache) if (d.base) rad_dev_free(d.base, RAD_MEM_DEVICE);
    dcache.clear();

    for (const auto& [what, n] : skipped_ops) std::printf("  skipped %-60s x%d\n", what.c_str(), n);
    if (!overruns.empty()) {
        std::printf("\nOUT OF BOUNDS -- a kernel wrote outside what it was given\n");
        for (const auto& [what, n] : overruns) std::printf("  %-70s x%d\n", what.c_str(), n);
    }
    if (!diverged.empty()) {
        std::printf("\ndescribed differently by the two libraries (not a correctness result)\n");
        for (const auto& [what, n] : diverged) std::printf("  %-70s x%d\n", what.c_str(), n);
    }
    std::printf("\n%d checked, %d passed, %d failed, %d skipped, %d diverged, %d capped. "
                "worst rel_l2 = %.3e\n", st.checked, st.passed, st.failed, st.skipped,
                st.diverged, st.capped, st.worst);
    std::printf("memory: %d case(s) ran inside guarded allocations, %d wrote outside one\n",
                st.guarded, st.overrun);
    print_row_coverage(census(reg, ref_name), report_rows);
    if (!report_path.empty()) {
        std::vector<std::string> graph_ops;
        for (const KfCase& c : fx.cases) graph_ops.push_back(c.op);
        std::sort(graph_ops.begin(), graph_ops.end());
        graph_ops.erase(std::unique(graph_ops.begin(), graph_ops.end()), graph_ops.end());
        if (write_report(report_path, report_rows, skipped_ops, fuse_skipped, diverged, overruns,
                         analysis_path,
                         "fixture: `" + fixture_path + "`, " + fmt("%zu", fx.cases.size()) +
                             " case(s) recorded " + (fx.recorded.empty() ? "?" : fx.recorded) +
                             " from `" + (fx.source.empty() ? "?" : fx.source) + "`",
                         0, 0, st, fx.source, ref_name, plugins.libraries(), "",
                         graph_ops, census(reg, ref_name), bench) >= 0)
            std::printf("report written to %s (%zu row(s))\n", report_path.c_str(),
                        report_rows.size());
    }
    return st.failed ? 1 : 0;
}

void usage() {
    std::printf(
        "rad-kbench -- correctness, completeness and speed of every kernel library, in one report\n"
        "\n"
        "usage: rad-kbench [options]                     replay the recorded fixture (no model)\n"
        "       rad-kbench -m <model.rad> [options]      check against a real container\n"
        "       rad-kbench -m <model.rad> --record FILE  record a new fixture\n"
        "\n"
        "  With no -m it REPLAYS: every kernel every loaded library supplies, against the reference library's\n"
        "  recorded answers on seeded inputs, inside guarded allocations. Seconds, any machine.\n"
        "  With -m it checks the same kernels against a REAL container's weights in the layout\n"
        "  the resolved kernel asked for, which is a test of the artifact and of the layout\n"
        "  hooks and is the only way to get one.\n"
        "\n"
        "      --fixture FILE the recording to replay (default $RADIANCE_HOME/kernels.rkb)\n"
        "      --record FILE  with -m: write a fixture. Weights are DRAWN, not read from the\n"
        "                     container, because a reference nothing can reproduce is not one.\n"
        "      --analysis FILE  a markdown fragment from tools/kbench-analysis.sh, spliced into\n"
        "                     the report as its static-and-dynamic-analysis section\n"
        "  -m, --model FILE   the .rad. Weight operands are read FROM IT, in the layout the\n"
        "                     resolved kernel asked for -- that is the point of the tool.\n"
        "      --max-ctx N    the deepest context this run considers: DECLARE runs at it and the\n"
        "                     attention sweep builds to it. Default: the checkpoint's training\n"
        "                     context for declare, 16384 for the sweep. An architecture whose\n"
        "                     graph depends on the bound -- Qwen4-Exp sizes its QSA buffers by it\n"
        "                     -- is checked at the bound given here.\n"
        "                     Attention is swept over DEPTH as well as query length: the paged\n"
        "                     block table, the split over KV and the window all live there.\n"
        "      --home DIR     $RADIANCE_HOME\n"
        "      --kernels A:B  kernel plugin hierarchy, first wins\n"
        "      --op NAME      check only this op (repeatable prefix match)\n"
        "      --ref NAME     the oracle plugin's name (default: the library that declares itself a\n"
        "                     reference implementation)\n"
        "      --seed N       base seed; every case derives its own from it, so changing a\n"
        "                     filter does not change any other case's inputs\n"
        "      --tol X        override every tolerance with X (for bisecting, not for passing)\n"
        "      --max-tok N    the max_tok declare is run at (default 512; small on purpose)\n"
        "      --mm-max-patches N  the encoder pass declare is run at, so a media tower's\n"
        "                     kernels are in the graph: default 2048 (rounded to whole output\n"
        "                     rows) when the container carries an encoder, 0 to leave it out\n"
        "      --no-fusion    skip the fusion oracle: a kernel that declares the chain it\n"
        "                     replaces is otherwise run against that chain, out of its own\n"
        "                     plugin, and the outputs compared BYTE for byte\n"
        "      --max-cases N  runs per distinct (op, kernel, geometry, tuned point); 0 means every\n"
        "                     one (default 4). A 62-layer model presents the same kernel at the\n"
        "                     same shape sixty times and only the weights differ.\n"
        "      --max-bytes MIB  skip a case whose operands would need more than this, by name\n"
        "                     (default: half the memory free at start, at most 8192). One case\n"
        "                     holds every operand as host floats, again as the kernel's dtype and\n"
        "                     again per library on the device, so a small machine is otherwise\n"
        "                     killed by the OOM killer with no indication which op it died on.\n"
        "                     Also bounds the weight caches.\n"
        "      --bench        time each case that passes, in the same pass and on the same\n"
        "                     bytes. See the caveat the report prints about the GB/s column.\n"
        "      --bench-iters N  launches per timed rep (default 20)\n"
        "      --bench-reps N   timed reps, median reported (default 3)\n"
        "      --bench-warm-ms N  untimed launches before each timed window, in ms (default 500).\n"
        "                     These cards idle-suspend and the ramp is longer than a short timed\n"
        "                     loop, so a cold measurement is of the governor, not the kernel.\n"
        "      --report FILE  write the markdown report: a summary of every loaded library, the\n"
        "                     op-by-library completeness matrix, every case and its error, the\n"
        "                     speed table, and every skip named\n"
        "      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at, as it is on the\n"
        "                     engine: an architecture whose attention kernels serve only one width\n"
        "                     declares no attention at the other. Default bf16\n"
        "      --num-speculative-tokens N\n"
        "                     draft depth to declare at; the default is the container's own, which\n"
        "                     is what puts the DRAFTER's kernels in the checked graph at all\n"
        "  -v, --verbose      print every case, not only the failures\n"
        "  -h, --help\n");
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string model, home = rad_home(), ref_name;
    /* REPEATABLE, as the usage text states. Held as a single string the last --op would silently
     * win and the earlier ones be discarded, so a run filtered to three ops would quietly check
     * one. An op matches if it has ANY of these as a prefix; empty means every op. */
    std::vector<std::string> op_filters;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    uint64_t seed = 20260904;
    double   tol_override = -1;
    int64_t  max_tok = 512;
    /* THE ENCODER PASS. A program declared with no pass has no media tower in it, and none of the
     * tower's kernels would ever be checked. -1 is auto: a pass when the container carries an
     * encoder (radiance.encoder), smaller than a serve's -- the tower's ops are the same kernels
     * at every pass size, and the cases are capped per geometry anyway. */
    int64_t  mm_max_patches = -1;
    /* THE DEEPEST CONTEXT AN ATTENTION CASE IS BUILT AT. A paged KV cache at this depth is
     * allocated per case, so it is a memory budget as much as a coverage choice: at 16384 the
     * 27B's cache is 67 MB a case, at 262144 it would be a gigabyte. The sweep takes 2048 and
     * 16384 -- one past the window the block table rebases at, one well past it. */
    /* 0 = not given. THE SAME FLAG SETS BOTH: the deepest context this run considers is the one
     * declare runs at and the one the attention sweep builds to. An architecture whose graph
     * depends on the bound -- Qwen4-Exp refuses a context above the one its QSA is exactly dense
     * to -- cannot be checked at all if declare uses the training context unconditionally. */
    int64_t  max_ctx_flag = 0;
    int      kv_dtype = RAD_DT_INVALID;
    int64_t  sweep_max_ctx = 16384;
    /* --record writes the fixture; with no model and no --record the tool REPLAYS one. */
    std::string record_path, fixture_path;
    /* A markdown fragment tools/kbench-analysis.sh produced, spliced into the report verbatim.
     * Read rather than run: the analysis belongs on a machine that need not have a GPU, and a
     * report that needs both present to render is a report that renders nowhere. */
    std::string analysis_path;
    /* HOW MANY TIMES ONE (op, kernel, geometry, tuned point) IS RUN.
     *
     * A 62-layer model presents the SAME kernel at the SAME geometry sixty times over, and what
     * differs between them is only which weight came out of the container. That is worth checking
     * -- it is half of what this tool is for -- but it is not worth checking sixty times: the
     * kernel is identical and the report already collapses the group to one row carrying the
     * WORST of it. The cost is the oracle: ref computes a 512 x 17408 x 5120 GEMM on the host for
     * every one of those cases, so an uncapped pass over a large model does not terminate in any
     * useful time.
     *
     * So: full coverage of every distinct kernel and geometry, sampled over the container. 0 means
     * every case, which is what to use before trusting a release. */
    int      max_cases = 4;
    /* The most one CASE may allocate, counting every operand built for the kernel, for the oracle,
     * and a re-laid weight's third copy. Above every weight in the models this tree serves and
     * below the n-gram embedding table, which is 51.2 GB on its own.
     *
     * 0 means "ask the machine", and that is the default because this runs in ctest on an
     * arbitrary machine. The biggest case in the tree's own fixture needs 7.3 GiB by the
     * estimate above, and a fixed ceiling higher than what the machine has is not a bound at
     * all -- it is the OOM killer with extra steps, which is exit 137, two lines of log, and no
     * indication which op it died on. A machine with room gets the full 8 GiB ceiling. */
    int64_t  max_case_bytes = 0;
    /* ON BY DEFAULT, because a fusion's whole claim is that it is substitutable and a claim
     * nothing checks is a comment. --no-fusion is for bisecting, not for passing. */
    bool     fuse_check = true;
    bool     bench = false;
    std::string report_path;
    int      bench_iters = 20, bench_reps = 3;
    /* Untimed launches before the timed window. Half a second, which is what libr4d's own
     * selftest uses for these cards: the ramp out of the low-power state is longer than a
     * short timed loop, and a benchmark that starts cold measures the governor. Lower it when
     * iterating on one op with --op; --bench is opt-in, so nothing pays it by default. */
    int      bench_warm_ms = 500;
    int      n_spec = -1;   /* -1 = the container's own operating point */
    bool     verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* w) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "rad-kbench: %s needs a value\n", w);
                                 std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-m" || a == "--model") model = next("--model");
        else if (a == "--home")    home = next("--home");
        else if (a == "--kernels") hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--op")      op_filters.push_back(next("--op"));
        else if (a == "--ref")     ref_name = next("--ref");
        else if (a == "--seed")    seed = (uint64_t)std::strtoull(next("--seed"), nullptr, 10);
        else if (a == "--tol")     tol_override = std::strtod(next("--tol"), nullptr);
        else if (a == "--max-tok") max_tok = std::atoll(next("--max-tok"));
        else if (a == "--max-ctx") max_ctx_flag = std::atoll(next("--max-ctx"));
        else if (a == "--mm-max-patches") mm_max_patches = std::atoll(next("--mm-max-patches"));
        else if (a == "--kv-cache-dtype") {
            const std::string v = next("--kv-cache-dtype");
            if (v == "fp8")       kv_dtype = RAD_F8E4M3;
            else if (v != "bf16") { std::fprintf(stderr, "rad-kbench: --kv-cache-dtype takes bf16 "
                                                         "or fp8, not '%s'\n", v.c_str());
                                    return 2; }
        }
        else if (a == "--record")  record_path  = next("--record");
        else if (a == "--fixture") fixture_path = next("--fixture");
        else if (a == "--analysis") analysis_path = next("--analysis");
        else if (a == "--max-cases") max_cases = std::atoi(next("--max-cases"));
        else if (a == "--max-bytes")
            max_case_bytes = (int64_t)std::atoll(next("--max-bytes")) << 20;
        else if (a == "--no-fusion") fuse_check = false;
        else if (a == "--bench")   bench = true;
        else if (a == "--bench-iters") bench_iters = std::atoi(next("--bench-iters"));
        else if (a == "--bench-reps")  bench_reps  = std::atoi(next("--bench-reps"));
        else if (a == "--bench-warm-ms") bench_warm_ms = std::atoi(next("--bench-warm-ms"));
        else if (a == "--report")  report_path = next("--report");
        else if (a == "--num-speculative-tokens") n_spec = std::atoi(next("--num-speculative-tokens"));
        else if (a == "-v" || a == "--verbose") { verbose = true; log_set_level(Log::Debug); }
        else { std::fprintf(stderr, "rad-kbench: unrecognised argument '%s'. Try --help.\n",
                            a.c_str()); return 2; }
    }

    /* THE CASE BUDGET, RESOLVED FROM WHAT THE MACHINE HAS IF NOBODY SAID.
     *
     * Half of what is free right now, and never more than 8 GiB -- so a machine with room checks
     * every case the 8 GiB ceiling admits and a small one refuses the two or three cases it
     * cannot hold, by name, instead of being killed. Half
     * rather than all because this figure is a snapshot and the run outlives it, and because the
     * estimate the budget is compared against is of ONE case while the caches hold more.
     *
     * Reported when it bites, because a pass that silently checked less than the last one is the
     * failure this tool exists to make impossible elsewhere. */
    if (max_case_bytes <= 0) {
        const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
        const int64_t avail = pages > 0 && page > 0 ? (int64_t)pages * (int64_t)page : 0;
        int64_t want = avail > 0 ? avail / 2 : ((int64_t)8 << 30);
        if (want > ((int64_t)8 << 30)) want = (int64_t)8 << 30;
        if (want < ((int64_t)512 << 20)) want = (int64_t)512 << 20;
        max_case_bytes = want;
        if (want < ((int64_t)8 << 30))
            RAD_WARN("case budget is %s, taken from the %s this machine has free -- cases over it "
                     "are named and skipped. --max-bytes MIB overrides.",
                     humanb(want).c_str(), humanb(avail).c_str());
    }

    /* ================================================== WHICH OF THE TWO JOBS THIS IS
     *
     * NO MODEL MEANS REPLAY, and that is the ordinary case. A kernel library is checked against a
     * recorded reference in seconds on any machine, and the 27B container is needed only by
     * whoever records. `-m` runs the other job: the same cases against a REAL container's weights
     * in the layout its resolved kernel asked for, which is a test of the artifact and of the
     * layout hooks, and which nothing else can do. Both are worth having and they answer
     * different questions; conflating them makes a library benchmark need a 27B file. */
    if (model.empty()) {
        if (fixture_path.empty()) fixture_path = rad_home_file(home, "kernels.rkb");
        Kfixture fx;
        const int rc = fx.load(fixture_path);
        if (rc < 0) {
            RAD_ERR("no fixture at %s: %s", fixture_path.c_str(), rad_strerror(rc));
            RAD_ERR("record one with: rad-kbench -m <model.rad> --record %s",
                    fixture_path.c_str());
            return 1;
        }
        return replay(fx, home, hierarchy, ref_name, op_filters, tol_override, max_cases, max_case_bytes,
                      bench, bench_iters, bench_reps, bench_warm_ms, verbose, report_path, fixture_path,
                      analysis_path);
    }

    RadFile f;
    if (f.open(model.c_str()) < 0) return 1;

    /* ---------------------------------------------------------------- plugins and declare */
    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, f.str(f.header().arch_id),
                               f.str(f.header().quant), &plugins) < 0) return 1;
    ref_name = resolve_reference(rad_tools_registry(), ref_name);
    if (ref_name.empty()) return 1;


    RadModelMeta meta{};
    std::vector<std::string> kk, kv;
    std::vector<const char*> kp, vp;
    meta.arch_id = f.str(f.header().arch_id);
    meta.name    = f.str(f.header().model_name);
    meta.quant   = f.str(f.header().quant);
    auto mi = [&](const char* k, int64_t d) { int64_t v = d; f.get_i(k, &v); return v; };
    auto mf = [&](const char* k, double d) { double v = d; f.get_f(k, &v); return (float)v; };
    meta.n_layers = mi("n_layers", 0);  meta.n_embd = mi("n_embd", 0);
    meta.n_head   = mi("n_head", 0);    meta.n_head_kv = mi("n_head_kv", 0);
    meta.head_dim = mi("head_dim", 0);  meta.n_ff = mi("n_ff", 0);
    meta.n_vocab  = mi("n_vocab", 0);   meta.n_ctx_train = mi("n_ctx_train", 0);
    meta.n_expert = mi("n_expert", 0);  meta.n_expert_used = mi("n_expert_used", 0);
    meta.n_expert_shared = mi("n_expert_shared", 0);
    meta.rms_eps = mf("rms_eps", 1e-6); meta.rope_theta = mf("rope_theta", 10000.f);
    meta.rope_scale = 1.f;   /* the rotary's scaling is the free-form rope keys (rad_builder.h) */

    /* The gated-delta-net head counts. No op names them -- the kernels read them off the operands
     * -- so a recipe sizing a GDN case would otherwise have to invent them, and a GDN check at an
     * invented head ratio is a check of a model nobody is running. They are in the container
     * because the architecture plugin read them from it. Zero means this container is not a hybrid
     * and the gdn_* recipes decline, which is the same "skipped and counted" the tool does for
     * every op it cannot size. */
    /* THROUGH THE STRING TABLE, not get_i. rad-convert writes a checkpoint's config verbatim and
     * a JSON number arrives as a string, which is why the architecture plugin reads these with its
     * own parse -- get_i only matches a key stored as RAD_P_INT and answers 0 for these. */
    auto ms = [&](const char* k, int64_t d) {
        int64_t v = d;
        if (f.get_i(k, &v)) return v;
        const char* s = f.get_s(k, nullptr);
        if (!s || !*s) return d;
        char* end = nullptr;
        const long long p = std::strtoll(s, &end, 10);
        return (end && end != s) ? (int64_t)p : d;
    };
    ShapeCtx sctx;
    sctx.n_head_k = ms("linear_num_key_heads",   ms("qwen35.ssm.group_count", 0));
    sctx.n_head_v = ms("linear_num_value_heads", ms("qwen35.ssm.time_step_rank", 0));
    sctx.n_seq    = 1;
    RAD_DEBUG("gdn head counts from the container: n_head_k=%lld n_head_v=%lld",
              (long long)sctx.n_head_k, (long long)sctx.n_head_v);
    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const RadFileKV& e = f.meta_at(i);
        if (e.type != RAD_P_STR) continue;
        kk.push_back(f.str(e.key));
        kv.push_back(f.str(e.v.s));
    }
    for (auto& s : kk) kp.push_back(s.c_str());
    for (auto& s : kv) vp.push_back(s.c_str());
    meta.n_kv = (int)kk.size(); meta.kv_key = kp.data(); meta.kv_val = vp.data();

    /* THE DRAFTER HAS TO BE IN THE GRAPH OR ITS KERNELS ARE NEVER CHECKED. `max_spec` left at 0
     * declares a program with no speculative path in it at all, so a drafter's conv and selector,
     * the MTP head, the 2-bit draft GEMM, `row_topk`, `row_topk_merge` and `quant_act_i8` are
     * never falsified against the oracle -- on a deployment where speculation is on by default and
     * a third of the decode path is the drafter. rad-tune takes the same flag for the same reason:
     * the two tools have to declare the same program, or a shape tuned is not a shape checked.
     *
     * The default is what the PLUGIN says its drafter is worth (rad_arch_probe), the same answer
     * the engine reaches, so the graph checked is the graph served without anyone passing a flag. */
    RadBuildCtx ctx{};
    ctx.rank = 0; ctx.world_size = 1; ctx.max_tok = max_tok;
    RadArchProbe probe{};
    (void)rad_tools_probe(plugins, meta, &probe);
    ctx.max_spec = n_spec >= 0 ? n_spec : probe.draft_depth;
    if (mm_max_patches < 0) {
        const char* enc = rad_meta_gets(&meta, "radiance.encoder", "");
        mm_max_patches = (enc && *enc) ? 2048 : 0;
    }
    if (mm_max_patches > 0 && probe.enc_merge > 1)
        mm_max_patches -= mm_max_patches % probe.enc_merge;
    ctx.max_enc_patches = mm_max_patches > 0 ? mm_max_patches : 0;
    {
        const char* enc = rad_meta_gets(&meta, "radiance.encoder", "");
        if (mm_max_patches != 0 && enc && std::strstr(enc, "audio") && probe.audio_frames > 0)
            ctx.max_audio_frames = probe.audio_frames;
    }
    if (ctx.max_enc_patches > 0)
        RAD_INFO("  encoder pass %lld patches (--mm-max-patches)", (long long)ctx.max_enc_patches);
    ctx.max_seqs = 8;
    ctx.max_ctx = max_ctx_flag > 0 ? max_ctx_flag : meta.n_ctx_train;
    ctx.kv_dtype = kv_dtype;
    ctx.scope = "";
    if (max_ctx_flag > 0) sweep_max_ctx = max_ctx_flag;
    if (ctx.max_spec > 0)
        RAD_INFO("  draft depth  %d (the container's own; --num-speculative-tokens overrides)",
                 ctx.max_spec);

    /* AND THE SAMPLER CHAIN, FOR EXACTLY THE REASON ABOVE, ONE COMPONENT FURTHER ON.
     *
     * The twelve sampler stages are declared by the ENGINE, not by the architecture plugin, so a
     * tool that declares only the plugin's ops produces a program with no sampling in it -- and
     * none of `sample_penalties`, `sample_dry`, `sample_temp`, `sample_mask`, `sample_topk`,
     * `sample_argmax`, `sample_topp`, `sample_minp`, `sample_typical`, `sample_xtc`,
     * `sample_pick` and `sample_merge_topk` would ever be checked against the oracle. They
     * decide the text.
     *
     * libref carries a shape recipe for each of them, written with a fill law chosen so that the
     * narrowing stages are really exercised on some rows and identity on others. The engine's
     * sampler takes the token buffer as an argument (core/engine.cpp) so that rad-kbench can
     * drive the same chain against a buffer it owns, which is this call.
     *
     * NOT FATAL. A sampler that will not declare is a reason to check less, not a reason to check
     * nothing: the rest of the graph is still worth the pass, and a tool that refused a container
     * over its sampler would be a worse trade than the coverage is worth.
     *
     * A REPLAY carries sampler cases only when its fixture was recorded with the chain declared. */
    Sampler sampler;
    auto also = [&](RadBuilder* b, int64_t logits_width) -> int {
        SamplerConfig sc{};
        sc.n_vocab       = meta.n_vocab;
        sc.n_vocab_local = logits_width > 0 ? logits_width : meta.n_vocab;
        sc.max_seqs      = ctx.max_seqs;
        sc.max_spec      = ctx.max_spec;
        sc.rank          = ctx.rank;
        sc.world_size    = ctx.world_size;
        const int s = sampler.declare(b, sc);
        if (s < 0)
            RAD_WARN("the sampler chain did not declare (%s); its twelve stages are not in this "
                     "pass", rad_strerror(s));
        return RAD_OK;
    };

    Program prog;
    if (rad_tools_declare(plugins, meta, ctx, &prog, also, nullptr,
                          rad_container_sources(f)) < 0) return 1;

    Registry& reg = rad_tools_registry();

    /* The oracle. It is a KERNEL PLUGIN like any other and it sits last in the hierarchy, so it
     * is found by scanning rows rather than by selecting -- selection would hand back whichever
     * kernel is under test, which is the opposite of an oracle. */
    auto ref_for = [&](const std::string& op, const Geometry& g) -> const KernelRow* {
        const KernelRow* best = nullptr;
        for (const KernelRow& r : reg.rows()) {
            if (r.plugin != ref_name || !r.info || op != r.info->op) continue;
            if (r.info->domain != RAD_DOMAIN_HOST) continue;
            bool ok = true;
            for (int c = 0; c < r.info->n_constraints && ok; ++c)
                ok = constraint_holds(r.info->constraints[c], g);
            if (!ok) continue;
            if (!best || r.info->priority > best->info->priority) best = &r;
        }
        return best;
    };


    /* THE SAME PLUGIN'S UNFUSED ROWS, for checking a declared fusion against the chain it names.
     *
     * Deliberately NOT the oracle and deliberately NOT selection. A fusion row's claim is
     * "byte-identical to the two ops it replaces" -- and the ops it replaces are its OWN library's,
     * run on the same card in the same arithmetic. Comparing against libref would be a different
     * and much weaker question, answered to a tolerance -- and for most fused ops ref has no
     * kernel at all, so it would skip them rather than check them. Comparing against whatever
     * selection returns would hand back the fused kernel itself. */
    auto row_in = [&](const std::string& plugin, const std::string& op, const Geometry& g,
                      int domain) -> const KernelRow* {
        const KernelRow* best = nullptr;
        for (const KernelRow& r : reg.rows()) {
            if (r.plugin != plugin || !r.info || op != r.info->op) continue;
            if (r.info->domain != domain || !r.info->launch) continue;
            bool ok = true;
            for (int c = 0; c < r.info->n_constraints && ok; ++c)
                ok = constraint_holds(r.info->constraints[c], g);
            if (!ok) continue;
            if (!best || r.info->priority > best->info->priority) best = &r;
        }
        return best;
    };

    std::printf("rad-kbench %s\n", model.c_str());
    std::printf("  oracle       %s (host domain)\n", ref_name.c_str());
    std::printf("  libraries    %s\n", plugins.libraries().c_str());
    std::printf("  ops declared %zu, seed %llu\n\n", prog.ops.size() - 1,
                (unsigned long long)seed);

    /* THE FIXTURE BEING RECORDED. Empty and unused unless --record was given; the recording is a
     * side effect of the container pass rather than a mode of its own, because the two must see
     * exactly the same cases or the file describes a run nobody made. */
    const bool recording = !record_path.empty();
    Kfixture fixture;
    /* The container's file name, not its path: the fixture is checked in, and where one machine
     * keeps its models is nothing a reader of the report needs. */
    fixture.source = model.substr(model.find_last_of('/') + 1);
    fixture.oracle = ref_name;
    Stats st;
    std::map<std::string, int> skipped_ops;
    /* One counter per distinct (op, kernel, plugin, domain, geometry, tuned point) -- the same key the
     * report groups on, so "n = 4 of 60" in a row means exactly what the cap did. */
    std::map<std::string, int> group_seen;
    /* The fusion oracle keeps its own tally: it answers a different question from the ref
     * comparison (byte-identity against a chain, not closeness against a reference), so folding
     * the two into one pass/fail would make both numbers mean less than either does alone. */
    std::map<std::string, int> fuse_skipped;
    /* Cases where the kernel and the oracle describe the same OUTPUT differently. Its own tally,
     * beside the fusion one and for the same reason: it answers "do these two libraries mean the
     * same thing by this op name", which is neither a pass nor a fail of the arithmetic. */
    std::map<std::string, int> diverged;
    /* Kernels caught writing outside an operand or outside their own scratch. Counted apart from
     * everything else because a kernel can be numerically perfect and still corrupt its
     * neighbour, and because in this harness that corruption lands in an unused allocation --
     * which is exactly why it has to be looked for rather than waited for. */
    std::map<std::string, int> overruns;
    int fuse_passed = 0, fuse_failed = 0;
    std::vector<Row> report_rows;

    for (size_t oi = 1; oi < prog.ops.size(); ++oi) {
        const OpInfo& o = prog.ops[oi];
        if (!op_filters.empty()) {
            bool hit = false;
            for (const std::string& f : op_filters) if (o.op.rfind(f, 0) == 0) { hit = true; break; }
            if (!hit) continue;
        }
        if (!o.schema) continue;

        /* ================================================== THE POINTS THIS OP IS TESTED AT
         *
         * ONE POINT IS NOT A TEST OF A BAND. Taking only the band's upper bound checks a GEMM band
         * covering M in (64, 512] at 512 and nowhere else, so a kernel whose tail handling is
         * wrong for M = 65 -- the rows that do not fill its tile, which is where tail bugs live --
         * passes. The same shape appears one level up in a tile ladder: a width nothing ever asks
         * at is a width that silently runs the next tile up.
         *
         * So each band is swept: its lower edge, its upper edge and geometric interior points.
         * The lower edge is the one that matters most and is the one a band's upper bound can
         * never reach -- M = lo is the shortest call the kernel claims, and the selector will send
         * real traffic there.
         *
         * AND SEPARATELY, THE CONTEXT DEPTH. For attention the query length is the ranged key and
         * the CONTEXT is not: `ref_attn_paged_shape` builds one sequence whose context is exactly
         * its own query, so without a depth sweep a decode of 10 query rows is checked against 10
         * tokens of KV and `attn_paged` is never checked at any real depth. That is the dimension
         * the paged kernel's block table, its split-K over KV and its window masking all live on,
         * and where a bug costs whole percentage points of decode throughput -- a rebased block
         * table leaving stale slots past the window is the shape of it. Depth is swept for any op
         * whose operand description RESPONDS to it, which is asked rather than listed: a new
         * attention-shaped op is swept without anyone editing a table here. */
        std::vector<int64_t> ms;
        {
            const bool ranged = !o.ranged_key.empty();
            for (size_t b = 0; b < o.bands.size(); ++b) {
                int64_t hi = o.bands[b].hi;
                if (hi <= 0 || hi > (1 << 20)) hi = max_tok;
                if (!ranged) { ms.push_back(1); continue; }
                hi = std::min<int64_t>(hi, max_tok);
                int64_t lo = b ? o.bands[b - 1].hi + 1 : std::max<int64_t>(o.range_lo, 1);
                if (lo <= 0) lo = 1;
                lo = std::min(lo, hi);
                /* lo, hi and a geometric interior: a band spanning 65..512 is tested at 65, 128,
                 * 256 and 512 rather than at 512 alone. Geometric and not arithmetic because a
                 * GEMM's tile count and an attention's split count both move with the ratio. */
                ms.push_back(lo);
                for (int64_t m = lo * 2; m < hi; m *= 2) ms.push_back(m);
                if (hi != lo) ms.push_back(hi);
            }
            std::sort(ms.begin(), ms.end());
            ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
        }
        /* Which band each point belongs to -- the resolved kernel is the band's, and a point must
         * be checked against the kernel the selector would actually hand it. */
        auto band_of = [&](int64_t m) -> size_t {
            for (size_t b = 0; b < o.bands.size(); ++b) {
                int64_t hi = o.bands[b].hi;
                if (hi <= 0 || hi > (1 << 20)) hi = max_tok;
                if (m <= hi) return b;
            }
            return o.bands.empty() ? 0 : o.bands.size() - 1;
        };

        /* Does this op's operand description depend on the context depth? Probed against the
         * oracle, whose description defines the op. */
        std::vector<int64_t> ctxs{ 0 };   /* 0 means "whatever the recipe's own default is" */
        if (!o.bands.empty()) {
            const Resolved* r0 = nullptr;
            for (size_t b = 0; b < o.bands.size() && !r0; ++b)
                for (int d = 0; d < RAD_N_DOMAINS && !r0; ++d)
                    if (o.bands[b].dom[d] && o.bands[b].dom[d].row) r0 = &o.bands[b].dom[d];
            const KernelRow* orc = r0 ? ref_for(o.op, r0->geom) : nullptr;
            if (orc) {
                Geometry ga = r0->geom, gb = r0->geom;
                ga.set_i("ctx", 64); gb.set_i("ctx", 4096);
                std::vector<OpdShape> sa, sb;
                if (rad_operand_shapes_hook(orc->info, ga, o.schema->n_operands, &sa) >= 0 &&
                    rad_operand_shapes_hook(orc->info, gb, o.schema->n_operands, &sb) >= 0 &&
                    sa.size() == sb.size()) {
                    for (size_t k = 0; k < sa.size(); ++k)
                        if (sa[k].shape != sb[k].shape) {
                            for (int64_t c : { (int64_t)2048, (int64_t)16384 })
                                if (c <= sweep_max_ctx) ctxs.push_back(c);
                            break;
                        }
                }
            }
        }

        /* ONE LOOP, NOT A NESTED ONE. The sweep multiplies the iteration space rather
         * than nesting another level inside a body that is already four deep -- the point list
         * carries (M, depth) and the band each M belongs to is looked up, which is the same
         * answer the selector gave and not a second opinion about it. */
        struct CasePt { int64_t m; int64_t ctx; };
        std::vector<CasePt> pts;
        for (int64_t m : ms) for (int64_t c : ctxs) pts.push_back({ m, c });

        for (const CasePt& pt : pts) {
            const size_t b = band_of(pt.m);
            for (int dom = 0; dom < RAD_N_DOMAINS; ++dom) {
                const Resolved& r = o.bands[b].dom[dom];
                if (!r || !r.row || !r.row->info || !r.row->info->launch) continue;
                /* Checking ref against itself proves nothing and costs a lot at 151K vocab. */
                if (r.row->plugin == ref_name) continue;
                /* THE POINT, AND THE GEOMETRY THAT CARRIES IT.
                 *
                 * `r.geom` is what the SELECTOR resolved -- the band's upper bound with the tuned
                 * axes folded in -- and it is not what this case runs at. The sweep's value goes
                 * into a copy, and everything downstream (the shape hooks, the launch params, the
                 * report row, the cap key) reads the copy: a kernel handed M = 512 in its params
                 * while its operands were sized for M = 65 would read past every one of them, and
                 * it would be this tool's fault. The tuned axes ride along unchanged, because the
                 * point is inside the band those axes were chosen for. */
                const int64_t M = pt.m;
                Geometry g = r.geom;
                if (!o.ranged_key.empty()) g.set_i(o.ranged_key, M);

                const KernelRow* oracle = ref_for(o.op, g);
                /* NO ORACLE IS NOT THE END OF THE CASE, if the kernel declares a chain.
                 *
                 * ref's op list is the conventional vocabulary and an op outside it has no
                 * fallback and no reference -- a deliberate decision (spec §17). But a FUSED op
                 * is outside it precisely because it is a fusion, and the thing it stands in for
                 * IS in the vocabulary. So a kernel that declares `replaces` carries its own
                 * oracle with it, and a better one: the claim is byte-identity against its own
                 * plugin's chain, not closeness to a host reference. On a model with many fused
                 * ops this is the difference between a page of skips and a page of checks. */
                const bool fuse_only = !oracle && fuse_check && dom == RAD_DOMAIN_DEVICE &&
                                       r.row->info->replaces;
                if (!oracle && !fuse_only) {
                    /* ref's op list is the conventional vocabulary; an op outside it has no
                     * fallback AND no oracle, and that is a deliberate decision (spec §17). */
                    skipped_ops[o.op + "  (no " + ref_name + " kernel: no oracle exists for it)"]++;
                    ++st.skipped;
                    continue;
                }

                /* ENOUGH LAYERS TO PROVE THE KERNEL, NOT EVERY LAYER IN THE MODEL.
                 *
                 * Counted AFTER the skips above, deliberately: a skip is cheap and its inventory
                 * is diagnostic, so capping before them would hide why an op was never checked.
                 * What the cap limits is the expensive part -- the buffers, the two launches and
                 * the comparison -- for a (kernel, geometry, tuned point) that has already been run
                 * `max_cases` times over different weights. */
                if (max_cases > 0) {
                    const std::string gk = o.op + "\x1f" + r.row->info->name + "\x1f" +
                                           r.row->plugin + "\x1f" + std::to_string(dom) + "\x1f" +
                                           g.str() + "\x1f" + r.choice + "\x1f" +
                                           std::to_string(pt.ctx);
                    if (++group_seen[gk] > max_cases) { ++st.capped; continue; }
                }

                /* THE ORACLE DESCRIBES THE OPERANDS, and only if it cannot does anything else.
                 *
                 * The two implementations must be handed the same bytes or the comparison means
                 * nothing, and where they could disagree about an extent it is the REFERENCE
                 * semantics that define the op -- so ref's `opd_shape` is the authority here, not
                 * the kernel under test's. Asking the kernel under test would let a kernel whose
                 * idea of an operand has drifted describe the very case that would have caught it.
                 *
                 * Then the kernel's own hook, for an op ref has no row for at all (there is no
                 * oracle in that case either, so this only ever supplies a shape for a case that
                 * is about to be skipped for want of one -- it costs nothing and keeps the two
                 * fallbacks in one order everywhere). Then the table in opshapes.h, the fallback
                 * for a plugin without the hook, which can go once every plugin has one.
                 *
                 * A decline from all three is a SKIP, counted and named. A tool that invents an
                 * extent tests the kernel's bounds handling and calls the result correctness. */
                /* WHAT THE OPS DO NOT SAY, HANDED TO THE HOOK THE ONLY WAY IT CAN BE.
                 *
                 * The gated-delta-net family reads its head COUNTS off the operands, so no gdn op
                 * names them and no resolved geometry carries them -- which is why opshapes.h
                 * takes a ShapeCtx beside the geometry. RadShapeFn deliberately has no such
                 * argument: its whole point is that a kernel is asked in the same terms a launch
                 * is. So they go in as ordinary parameters, exactly as a declared parameter
                 * would, and a hook that does not care never looks.
                 *
                 * Copied rather than set on g: the resolved geometry is what the kernel was
                 * SELECTED on, and adding keys to it after the fact would make the graph dump and
                 * the tune cache describe a shape nothing resolved against. */
                Geometry hgeom = g;
                if (sctx.n_head_k > 0) hgeom.set_i("n_head_k", sctx.n_head_k);
                if (sctx.n_head_v > 0) hgeom.set_i("n_head_v", sctx.n_head_v);
                if (sctx.n_seq   > 0) hgeom.set_i("n_seq",   sctx.n_seq);
                /* THE DEPTH, HANDED TO THE HOOK THE SAME WAY THE GDN HEAD COUNTS ARE: as an
                 * ordinary parameter. No op declares `ctx` -- the paged kernels read the context
                 * off `seqused` and the block table, which is exactly right for the KERNEL and
                 * leaves the DESCRIPTION with nothing to size the cache from but the query
                 * length. Zero leaves the recipe on its own default, which is what an op whose
                 * description does not respond to depth always sees. */
                if (pt.ctx > 0) hgeom.set_i("ctx", pt.ctx);

                std::vector<OpdShape> sh;
                int shst = oracle ? rad_operand_shapes_hook(oracle->info, hgeom,
                                                            o.schema->n_operands, &sh)
                                  : RAD_E_UNSUPPORTED;
                if (shst < 0)
                    shst = rad_operand_shapes_hook(r.row->info, hgeom, o.schema->n_operands, &sh);
                if (shst < 0)
                    shst = rad_operand_shapes(o.op, g, o.schema->n_operands, M, &sh, &sctx);
                if (shst < 0) {
                    skipped_ops[o.op + "  (neither the oracle nor the kernel describes its "
                                       "operands, and opshapes.h has no recipe)"]++;
                    ++st.skipped;
                    continue;
                }
                /* AN OPTIONAL WEIGHT THE DECLARATION LEAVES UNBOUND IS ABSENT, as it is at every
                 * launch: a norm with no gain, a LayerNorm with no bias. The description names
                 * the operand because the op can take it; this call does not, so both sides get
                 * a null tensor. The weight list is positional over WEIGHT-role operands, so an
                 * unbound one still takes its place in it. */
                {
                    size_t wpos = 0;
                    for (size_t k = 0; k < sh.size() && k < (size_t)o.schema->n_operands; ++k) {
                        const RadOperandSpec& os = o.schema->operands[k];
                        if (os.role != RAD_OPD_WEIGHT) continue;
                        const rad_weight h = wpos < o.weights.size() ? o.weights[wpos] : 0;
                        ++wpos;
                        if (!h && os.optional) sh[k].absent = true;
                    }
                }

                /* TWO LIBRARIES CAN DISAGREE ABOUT WHAT AN OP IS, AND THAT IS NOT A FAILING
                 * KERNEL.
                 *
                 * The operands are sized from the ORACLE's description, deliberately -- ref's
                 * semantics define the op. But a kernel publishes its own description too, and
                 * where the two disagree about an operand the kernel WRITES, the comparison below
                 * is not a numerical question at all: it hands libr4d a [M, n/group] scale plane,
                 * libr4d writes the [M, 1] its own hook declares and its own GEMMs consume, and
                 * the untouched remainder comes back as rel_l2 0.99 against an oracle that filled
                 * it. That reads as "this kernel is catastrophically wrong" and it means "these
                 * two libraries implement different ops under one name".
                 *
                 * `quant_act_i8` is the live case, named in r4d_shapes.cpp as well: libr4d takes
                 * no `group` at all and writes one scale per row; ref reads `group` as the number
                 * of elements sharing a scale. Both are coherent, only libr4d's is what the w2a8
                 * GEMM reads, and neither is a bug -- but reported as a FAIL it is
                 * indistinguishable from one.
                 *
                 * So it is named as a DIVERGENCE, with both descriptions printed, and it is
                 * counted apart from both the passes and the failures. Inputs are not compared:
                 * the two sides read the same bytes whatever they do with them, and an input the
                 * kernel reads differently shows up as a wrong output, which this pass would
                 * catch. */
                if (oracle && r.row->info->opd_shape) {
                    std::vector<OpdShape> ksh;
                    if (rad_operand_shapes_hook(r.row->info, hgeom, o.schema->n_operands,
                                                &ksh) >= 0 && ksh.size() == sh.size()) {
                        std::string why;
                        for (size_t k = 0; k < sh.size() && why.empty(); ++k) {
                            const int role = o.schema->operands[k].role;
                            if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                            if (sh[k].absent || ksh[k].absent) continue;
                            if (sh[k].dtype == ksh[k].dtype && sh[k].shape == ksh[k].shape)
                                continue;
                            const char* nm = o.schema->operands[k].name
                                                 ? o.schema->operands[k].name : "?";
                            why = std::string("'") + nm + "': " + ref_name + " says " +
                                  shape_str(sh[k]) + ", " + r.row->info->name + " says " +
                                  shape_str(ksh[k]);
                        }
                        if (!why.empty()) {
                            diverged[o.op + "  (" + std::string(r.row->info->name) + " and " +
                                     ref_name + " describe the same output differently -- " + why +
                                     ")"]++;
                            ++st.diverged;
                            continue;
                        }
                    }
                }

                /* THE CASE AS THE FIXTURE WILL HOLD IT, built before anything is allocated.
                 *
                 * `sh` is the oracle's description in the tool's own struct; `kop` is the same
                 * thing in the ON-DISK struct, and from here down the tool reads `kop`. That is
                 * deliberate and not an extra copy for its own sake: everything the replay does
                 * -- size the buffers, draw the inputs, decide which operands are outputs -- it
                 * does from `kop`, so a recorded case that replays differently from the run that
                 * recorded it would have to be a bug in ONE struct rather than a disagreement
                 * between two.
                 *
                 * The seed is per case and the operand seeds hang off it, so a replay reproduces
                 * the bytes knowing only what the file says. A seed derived per operand from (op,
                 * band, domain, operand) would not survive into the file at all. */
                const uint64_t case_seed =
                    rad_case_seed(seed, o.op, (int)b, dom, 0) ^ ((uint64_t)M << 32) ^
                    (uint64_t)pt.ctx;
                std::vector<KfOperand> kop((size_t)o.schema->n_operands);
                for (size_t k = 0; k < sh.size() && k < kop.size(); ++k) {
                    KfOperand& d = kop[k];
                    d.role       = o.schema->operands[k].role;
                    d.dtype      = sh[k].dtype;
                    d.shape      = sh[k].shape;
                    d.absent     = sh[k].absent;
                    d.fill       = sh[k].fill;
                    d.is_index   = sh[k].is_index;
                    d.idx_unique = sh[k].idx_unique;
                    d.idx_cu     = sh[k].idx_cu;
                    d.idx_max    = sh[k].idx_max;
                    d.idx_const  = sh[k].idx_const;
                    d.fill_chunk = sh[k].fill_chunk;
                }

                /* Allocate two output sets and one input set. Inputs are shared so the two
                 * implementations see identical bytes, which is the only way the comparison
                 * means anything. */
                /* WHAT THIS CASE WOULD COST, BEFORE ANY OF IT IS ALLOCATED.
                 *
                 * Every operand is built twice -- once for the kernel and once for the oracle --
                 * and a re-laid weight is built a third time, because the stored plane and the
                 * inverted one are different bytes. At Qwen4-Exp's n-gram table that is a 51.2 GB
                 * operand and ~150 GB of case, and the failure mode without this test is an OOM
                 * KILL: exit 137, two lines of log, and no indication which op it died on.
                 *
                 * A refusal that names the op is worth more than a check it cannot run. --max-bytes
                 * moves the line for a machine that can afford more; the default is chosen to sit
                 * above every weight in the models this tree serves (a 27B lm_head is 1.2 GiB) and
                 * below the one that does not fit at all. */
                {
                    int64_t want = 0;
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent)
                            want += 3 * rad_dtype_bytes(sh[k].dtype, rad_numel(sh[k].shape));
                    if (want > max_case_bytes) {
                        skipped_ops[o.op + "  (one case needs " + humanb(want) +
                                    ", over the " + humanb(max_case_bytes) +
                                    " budget -- raise --max-bytes to check it)"]++;
                        ++st.skipped;
                        continue;
                    }
                }

                std::vector<std::vector<uint8_t>> bufs(sh.size()), ref_out(sh.size());
                std::vector<RadTensor> t_test(sh.size()), t_ref(sh.size());
                int wk = 0;
                bool from_container = false, unusable = false;

                for (size_t k = 0; k < sh.size() && !unusable; ++k) {
                    const int64_t n = rad_numel(sh[k].shape);
                    const int role = o.schema->operands[k].role;

                    /* An optional operand this case deliberately does not pass: a null RadTensor,
                     * which is what the ABI already means by absent. Nothing is allocated and the
                     * comparison loop below skips it -- passing a buffer the ORACLE cannot fill
                     * would report a gap in ref as a fault in the kernel. */
                    if (sh[k].absent) {
                        /* The declared weight list is positional over WEIGHT-role operands, so an
                         * absent weight still consumes its handle -- otherwise every weight after
                         * it reads the one before, which is the silent-garbage class this tool
                         * exists to catch. */
                        if (role == RAD_OPD_WEIGHT) ++wk;
                        t_test[k] = RadTensor{};
                        t_ref[k]  = RadTensor{};
                        continue;
                    }

                    bufs[k].assign((size_t)rad_dtype_bytes(sh[k].dtype, n), 0);
                    if (bufs[k].empty() && n > 0) { unusable = true; break; }

                    /* THE DESCRIPTION BOTH SIDES SHARE, defined at the top of the loop body
                     * because the re-laid weight path below needs it and then leaves early. */
                    auto mk = [&](std::vector<uint8_t>& store) {
                        RadTensor t{};
                        t.data  = store.data();
                        t.dtype = sh[k].dtype;
                        t.rank  = (uint32_t)sh[k].shape.size();
                        for (size_t d = 0; d < sh[k].shape.size() && d < RAD_MAX_RANK; ++d)
                            t.shape[d] = sh[k].shape[d];
                        rad_tensor_pack(&t);
                        return t;
                    };

                    if (role == RAD_OPD_WEIGHT) {
                        /* A WEIGHT IS ITS ENCODING'S PLANES, AND THE TWO SIDES READ THEM
                         * DIFFERENTLY.
                         *
                         * Everywhere else in this loop the two implementations share one input
                         * buffer, because that is the only way the comparison means anything. A
                         * weight is the exception and the ONLY one: the kernel reads what its own
                         * layout hooks make of the planes -- fragment order, nibbles reordered, a
                         * row's scales appended to it -- and the oracle reads the tensor the
                         * planes decode to. Handing either side the other's bytes turns every
                         * such case into a confident FAIL at rel_l2 ~1.4, the value two unrelated
                         * tensors give, on a kernel that is fine.
                         *
                         * DRAWN OR READ. A RECORDING has to be replayable on a machine with no
                         * checkpoint, so it always draws, and records the encoding beside the
                         * operand so a replay draws the same planes. A CONTAINER PASS reads --
                         * except for a weight the container does not hold, which happens whenever
                         * the declaration is wider than the artifact: a bring-up slice converted
                         * at four layers and then checked with the full forty-eight. That one is
                         * drawn the same way and the report says "synthesised".
                         *
                         * `t_test` is described the way `Ctx::refresh_weights` describes it --
                         * STORED dtype, DECLARED shape (kf_describe_laid). */
                        const rad_weight h = wk < (int)o.weights.size() ? o.weights[(size_t)wk++]
                                                                       : 0;
                        const WeightInfo* wi = h ? &prog.weights[h] : nullptr;
                        const char* opd_name = o.schema->operands[k].name
                                                   ? o.schema->operands[k].name : "?";
                        if (!wi) {
                            skipped_ops[o.op + "  (operand '" + opd_name + "' is a weight and "
                                        "the declaration bound none to it)"]++;
                            unusable = true;
                            break;
                        }
                        kf_operand_enc(*wi, &kop[k]);
                        const RadFileEntry* e = recording ? nullptr : f.find(wi->source);
                        std::vector<std::vector<uint8_t>> pl;
                        if (!e) {
                            kf_draw_planes(kop[k], kf_weight_seed(kop[k], k), &pl);
                        } else {
                            /* One rank of one: each selected plane is the whole plane. */
                            for (int j = 0; j < wi->n_sel; ++j) {
                                const RadFilePlane& fp = f.plane(*e, wi->sel[j]);
                                const int64_t want =
                                    wi->sel_rows[j] *
                                    rad_enc_row_bytes(wi->enc.plane[wi->sel[j]].dtype,
                                                      wi->sel_cols[j]);
                                if ((int64_t)fp.bytes != want) {
                                    skipped_ops[o.op + "  ('" + wi->source + "' plane '" +
                                                wi->enc.plane[wi->sel[j]].role + "' holds " +
                                                std::to_string(fp.bytes) + " bytes and the "
                                                "declaration reads " + std::to_string(want) +
                                                ")"]++;
                                    unusable = true;
                                    break;
                                }
                                const uint8_t* src = (const uint8_t*)f.data(fp);
                                pl.emplace_back(src, src + fp.bytes);
                            }
                            if (unusable) break;
                            from_container = true;
                        }
                        std::string why;
                        if (kf_planes_logical(kop[k], pl, &ref_out[k], &why) != RAD_OK) {
                            skipped_ops[o.op + "  ('" + wi->source + "' as " +
                                        enc_name(wi->enc) + " does not make the operand '" +
                                        opd_name + "' the reference reads: " + why + ")"]++;
                            unusable = true;
                            break;
                        }
                        t_ref[k] = mk(ref_out[k]);
                        RadLayout lay{};
                        std::vector<uint8_t> laid;
                        const int lrc = kf_lay_weight(r.row->info, g.params(), g.n_params(),
                                                      (int)k, kop[k], pl, &lay, &laid);
                        if (lrc == RAD_OK) {
                            bufs[k].swap(laid);
                            kf_describe_laid(kop[k], lay, bufs[k].data(), &t_test[k]);
                        } else if (lrc == RAD_E_UNSUPPORTED && pl.size() == 1) {
                            /* read as stored: the plane, widened where the operand is wider */
                            bufs[k] = ref_out[k];
                            t_test[k] = mk(bufs[k]);
                        } else {
                            skipped_ops[o.op + "  (" + std::string(r.row->info->name) +
                                        " cannot lay out operand '" + opd_name + "' from " +
                                        enc_name(wi->enc) + ": " +
                                        (lrc == RAD_E_UNSUPPORTED ? "it reads several planes "
                                                                    "as stored"
                                                                  : rad_strerror(lrc)) +
                                        ")"]++;
                            unusable = true;
                            break;
                        }
                        continue;
                    } else if (role == RAD_OPD_IN || role == RAD_OPD_INOUT) {
                        /* THROUGH THE FIXTURE'S OWN DRAW, and not a second copy of it here.
                         *
                         * The recorder and the reader must produce byte-identical inputs or every
                         * recorded reference is noise, so there is exactly one implementation of
                         * "what does this operand get filled with" and both call it. Spelling
                         * the draw out inline here would be a hundred lines that can drift from
                         * kfixture.cpp's copy without anything failing to build. */
                        kf_draw(kop, k, M, kf_vocab_rows(kop), kf_operand_seed(case_seed, (int)k),
                                &bufs[k]);
                    }

                    ref_out[k] = bufs[k];   /* same inputs, separate outputs */
                    t_test[k] = mk(bufs[k]);
                    t_ref[k]  = mk(ref_out[k]);
                }
                if (unusable) { ++st.skipped; continue; }

                std::vector<uint8_t> scratch_a, scratch_b;
                auto launch = [&](const KernelRow* row, std::vector<RadTensor>& tv,
                                  std::vector<uint8_t>& scratch) {
                    RadArgs a{};
                    a.t = tv.data(); a.n_t = (int)tv.size();
                    a.p = g.params(); a.n_p = g.n_params();
                    a.rank = 0; a.world_size = 1;
                    if (row->info->scratch) {
                        const int64_t need = row->info->scratch(&a);
                        if (need > 0) { scratch.assign((size_t)need, 0);
                                        a.scratch = scratch.data(); a.scratch_bytes = need; }
                    }
                    a.instance = nullptr;
                    return row->info->launch(&a, nullptr);
                };

                /* ---- THE DEVICE DOMAIN NEEDS DEVICE MEMORY.
                 *
                 * The tensors above point at std::vector storage, which is exactly right for the
                 * oracle -- libref is domain HOST -- and is a host pointer if it reaches a device
                 * kernel. That does not fail as a diagnostic: the kernel either faults or, more
                 * often, writes nothing, and the comparison then reads an untouched output buffer.
                 * Where the inputs are zeros too, zero-vs-zero reports `rel_l2 = 0.000e+00 ok` --
                 * a device kernel that was never run, reported as a pass.
                 *
                 * So: one allocation per operand, the same bytes uploaded, a real stream, and the
                 * outputs brought back into the host buffers the comparison already reads. The
                 * oracle keeps running on the host set, which is the point -- the two see identical
                 * input bytes and differ only in where they ran. */
                /* ================================================== THE RED ZONES
                 *
                 * A KERNEL THAT WRITES PAST ITS OPERAND IS THE ONE BUG THIS TOOL WOULD OTHERWISE
                 * REPORT AS A PASS.
                 *
                 * Everything else here compares what a kernel wrote INSIDE the tensor it was
                 * given. An off-by-one on the last tile, a workgroup that rounds M up to its tile
                 * height and stores the padding, a split-K partial written at the wrong stride --
                 * each of those lands outside the operand, and in this harness the neighbouring
                 * bytes belong to a separate allocation, so the comparison never sees it and the
                 * case passes. In the engine those same bytes are another tensor, and the symptom
                 * is wrong output somewhere else entirely, three ops later, in a kernel that is
                 * correct. A short trailing chunk whose padding rows are clamped rather than
                 * masked is one of those: it poisons every prompt longer than the chunk width.
                 *
                 * DEVICE ASAN IS NOT AVAILABLE ON RDNA and will not become available: AMD's
                 * amdgcn ASan needs xnack+ for its page-fault shadow, RDNA does not implement
                 * xnack, and `rocminfo` reports `XNACK enabled: NO` on these cards. The bitcode
                 * ships in /opt/rocm/amdgcn/bitcode/asanrtl.bc and cannot run there. So the check
                 * is built here instead, out of what any GPU can do: every device allocation gets a
                 * poisoned margin on BOTH sides, the kernel is handed the middle, and the margins
                 * are read back and verified afterwards. It catches the out-of-bounds WRITE, which
                 * is the one that corrupts a neighbour; an out-of-bounds read it cannot see.
                 *
                 * 256 bytes because that is wide enough to catch a tile's worth of overrun and is
                 * a multiple of every alignment a kernel here asks of an operand -- handing a
                 * kernel a pointer the allocator would not have produced would test the kernel's
                 * alignment handling and call the result memory safety.
                 */
                constexpr int64_t RED_BYTES = 256;
                constexpr int     RED_FILL  = 0xA5;
                std::vector<DevBuf> dev(sh.size());
                DevBuf dev_scratch;
                RadStream stream = nullptr;
                bool dev_ok = true;
                /* Every device allocation in this case goes through here, so there is one place
                 * that decides what is guarded and no way to add an unguarded one by accident. */
                auto dev_guarded_alloc = [&](DevBuf& d, int64_t bytes) -> bool {
                    d.bytes = bytes;
                    d.base  = rad_dev_alloc(bytes + 2 * RED_BYTES, RAD_MEM_DEVICE);
                    if (!d.base) return false;
                    d.p = (uint8_t*)d.base + RED_BYTES;
                    d.on = true;
                    return rad_memset_async(d.base, RED_FILL, RED_BYTES, stream) >= 0 &&
                           rad_memset_async((uint8_t*)d.p + bytes, RED_FILL, RED_BYTES,
                                            stream) >= 0;
                };
                if (dom == RAD_DOMAIN_DEVICE) {
                    if (!(stream = run_stream())) dev_ok = false;
                    for (size_t k = 0; k < sh.size() && dev_ok; ++k) {
                        if (sh[k].absent || bufs[k].empty()) continue;
                        if (!dev_guarded_alloc(dev[k], (int64_t)bufs[k].size())) {
                            dev_ok = false; break;
                        }
                        if (rad_memcpy_async(dev[k].p, bufs[k].data(), dev[k].bytes, stream) < 0)
                            dev_ok = false;
                        t_test[k].data = dev[k].p;
                    }
                    if (dev_ok && rad_stream_sync(stream) < 0) dev_ok = false;
                }

                /* AN INOUT OPERAND'S ORIGINAL BYTES, kept only when a fusion check will need them.
                 *
                 * The chain below re-runs this case out of the same plugin's unfused rows and
                 * compares the bytes. It can SHARE the fused run's input and weight buffers --
                 * that is the point, both sides must see the same bytes, and it saves uploading a
                 * 90 MB weight twice. An INOUT operand cannot be shared: the fused launch has
                 * already overwritten it by the time the chain runs, so the chain would start from
                 * the fused kernel's own answer and agree with it trivially. */
                std::vector<std::vector<uint8_t>> inout0;
                const bool want_fuse = fuse_check && dom == RAD_DOMAIN_DEVICE && dev_ok &&
                                       r.row->info->replaces;
                if (want_fuse) {
                    inout0.resize(sh.size());
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent && o.schema->operands[k].role == RAD_OPD_INOUT)
                            inout0[k] = bufs[k];
                }

                auto dev_release = [&]() {
                    if (stream) (void)rad_stream_sync(stream);
                    for (auto& d : dev) if (d.base && d.owned) rad_dev_free(d.base, RAD_MEM_DEVICE);
                    if (dev_scratch.base) rad_dev_free(dev_scratch.base, RAD_MEM_DEVICE);
                };

                if (!dev_ok) {
                    std::printf("  %-22s %-40s SKIP  no device memory for this case: %s\n",
                                o.op.c_str(), g.str().c_str(), rad_dev_last_error());
                    dev_release();
                    ++st.skipped;
                    continue;
                }

                /* Scratch too. A kernel that asks for scratch and is handed a host pointer reads
                 * and writes the host's address space from the card, which is the same fault as an
                 * operand and is easier to miss because nothing compares it. */
                auto launch_dev = [&](const KernelRow* row, std::vector<RadTensor>& tv) -> int {
                    RadArgs a{};
                    a.t = tv.data(); a.n_t = (int)tv.size();
                    a.p = g.params(); a.n_p = g.n_params();
                    a.rank = 0; a.world_size = 1;
                    if (row->info->scratch) {
                        const int64_t need = row->info->scratch(&a);
                        if (need > 0) {
                            /* Scratch is guarded too, and it is the likelier of the two to be
                             * overrun: the size comes from the kernel's OWN RadScratchFn, so a
                             * scratch overrun is a kernel disagreeing with itself about how much
                             * it asked for -- which nothing else in this tool would notice. */
                            if (!dev_guarded_alloc(dev_scratch, need)) return RAD_E_NOMEM;
                            if (rad_memset_async(dev_scratch.p, 0, need, stream) < 0)
                                return RAD_E_DEVICE;
                            a.scratch = dev_scratch.p;
                            a.scratch_bytes = need;
                        }
                    }
                    a.instance = nullptr;
                    const int s = row->info->launch(&a, stream);
                    if (s < 0) return s;
                    return rad_stream_sync(stream);
                };

                /* THE ORACLE'S OWN COST, TAKEN FOR FREE. This launch happens on every checked
                 * case whether anyone times it or not, so a clock around it turns the correctness
                 * pass into a two-library speed comparison at no cost at all. One call, no warmup,
                 * no median -- libref is the definition of the op and not a contender, and a
                 * column that pretended otherwise would invite tuning it. */
                const auto tref0 = std::chrono::steady_clock::now();
                const int s_ref  = oracle ? launch(oracle, t_ref, scratch_b) : RAD_OK;
                const double ref_us =
                    oracle && s_ref >= 0
                        ? std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - tref0).count()
                        : -1.0;
                const int s_test = dom == RAD_DOMAIN_DEVICE
                                       ? launch_dev(r.row, t_test)
                                       : launch(r.row, t_test, scratch_a);

                /* THE CHECKED RESULT COMES BACK BEFORE ANYTHING IS TIMED, and the order is not a
                 * detail -- getting it wrong makes the benchmark corrupt the correctness answer it
                 * shares a pass with.
                 *
                 * MANY OPS ARE IN PLACE. `rope` rotates its operand where it lies, so a timed loop
                 * of twenty launches leaves the buffer rotated twenty-one times, and comparing
                 * that against an oracle that rotated once reports a large error on a kernel that
                 * is correct -- `rope` and `gdn_conv_update` would FAIL with --bench on and pass
                 * with it off. An instrumentation that changes the measurement is worse than no
                 * instrumentation, because it is believed.
                 *
                 * So: copy the outputs the checked launch produced into the host buffers first,
                 * then time on the device tensors, which are still bound to device memory and are
                 * repointed at the host only afterwards. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0) {
                    for (size_t k = 0; k < sh.size(); ++k) {
                        if (!dev[k].p) continue;
                        const int role = o.schema->operands[k].role;
                        if (role == RAD_OPD_OUT || role == RAD_OPD_INOUT)
                            rad_memcpy_async(bufs[k].data(), dev[k].p, dev[k].bytes, stream);
                    }
                    rad_stream_sync(stream);
                }

                /* THE MARGINS, READ BACK BEFORE THE TIMED LOOP RUNS.
                 *
                 * Before, for the same reason the copy-back is before: twenty more launches would
                 * tell you that SOMETHING overran, and this tells you that the one launch the
                 * correctness verdict is about did. The margin is reported by the side it is on
                 * and by how far in it was touched, because "wrote 4 bytes past the end" and
                 * "wrote 208 bytes past the end" are different bugs. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0) {
                    auto margin_check = [&](const DevBuf& d, const std::string& what) {
                        if (!d.on) return;
                        uint8_t lo[RED_BYTES], hi[RED_BYTES];
                        if (rad_memcpy_async(lo, d.base, RED_BYTES, stream) < 0) return;
                        if (rad_memcpy_async(hi, (const uint8_t*)d.p + d.bytes, RED_BYTES,
                                             stream) < 0) return;
                        if (rad_stream_sync(stream) < 0) return;
                        int64_t before = 0, after = 0;
                        for (int64_t i = 0; i < RED_BYTES; ++i) {
                            /* How FAR out, not how many bytes: the distance from the operand is
                             * what names the stride or the tile that got it wrong. */
                            if (lo[i] != (uint8_t)RED_FILL) before = std::max(before,
                                                                              RED_BYTES - i);
                            if (hi[i] != (uint8_t)RED_FILL) after  = std::max(after, i + 1);
                        }
                        if (!before && !after) return;
                        std::string msg = o.op + "  (" + std::string(r.row->info->name) + " wrote ";
                        if (before) msg += fmt("%lld byte(s) BEFORE the start", (long long)before);
                        if (before && after) msg += " and ";
                        if (after)  msg += fmt("%lld byte(s) PAST the end", (long long)after);
                        msg += " of " + what + ")";
                        overruns[msg]++;
                        ++st.overrun;
                        std::printf("  %-22s %-40s OUT OF BOUNDS: %s\n", o.op.c_str(),
                                    g.str().c_str(), msg.c_str());
                    };
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (dev[k].on)
                            margin_check(dev[k], std::string("operand '") +
                                                 (o.schema->operands[k].name
                                                      ? o.schema->operands[k].name : "?") + "'");
                    margin_check(dev_scratch, "its own scratch");
                    ++st.guarded;
                }

                /* ================================================== THE FUSION ORACLE
                 *
                 * RUN THE CHAIN THIS KERNEL SAYS IT REPLACES, OUT OF THE SAME PLUGIN, AND COMPARE
                 * THE BYTES.
                 *
                 * Most fused ops have no libref kernel, so without this they are skipped BY NAME
                 * -- and those skips include every fp8 quantiser fold, which is most of what a
                 * decode step launches. There is nothing wrong with the skip; there is simply no
                 * oracle.
                 *
                 * A declared chain supplies one, and a BETTER one than ref would be. The rows
                 * claim byte-identity, not closeness -- byte-identical to the ops they replace is
                 * what makes a fusion substitutable rather than a second numerical path -- so the
                 * test is memcmp and not a tolerance. Checked here rather than by hand inside a
                 * plugin's own selftest, so every declared fusion is covered and the check runs
                 * wherever this tool runs.
                 *
                 * WHAT IS SHARED AND WHAT IS NOT. Inputs and weights are the fused run's own
                 * device buffers: both sides must see identical bytes and a 90 MB weight should
                 * not be uploaded twice. Outputs get fresh allocations, since comparing a buffer
                 * against itself proves nothing. INOUT operands get fresh allocations restored
                 * from the snapshot taken before the fused launch, for the same reason.
                 *
                 * A TEMPORARY is a buffer the chain passes between its own steps and the fusion
                 * never materialises -- gemm_nt_q_gated's [M, 2n] product is the case. It is sized
                 * and typed by the STEP's own opd_shape, which is the correctness check's hook
                 * doing a second job: a kernel that can describe its operands for a correctness
                 * check can describe them here, and one that cannot is declined rather than
                 * guessed at. */
                if (want_fuse && s_test >= 0) {
                    std::string why;
                    std::vector<DevBuf> fdev(sh.size());
                    std::vector<DevBuf> ftmp(RAD_MAX_FUSE_TMP);
                    std::vector<RadTensor> ftmp_t(RAD_MAX_FUSE_TMP);
                    std::vector<std::vector<uint8_t>> fout(sh.size());
                    DevBuf fscratch;

                    /* Fresh device memory for everything the chain writes. */
                    for (size_t k = 0; k < sh.size() && why.empty(); ++k) {
                        if (sh[k].absent || bufs[k].empty()) continue;
                        const int role = o.schema->operands[k].role;
                        if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                        fdev[k].bytes = (int64_t)bufs[k].size();
                        fdev[k].p = rad_dev_alloc(fdev[k].bytes, RAD_MEM_DEVICE);
                        if (!fdev[k].p) { why = "no device memory for the chain's outputs"; break; }
                        /* An INOUT starts from the bytes the FUSED run started from, not from what
                         * it left behind. An OUT starts zeroed, so a step that fails to write is
                         * visible as a difference rather than as inherited garbage. */
                        const void* src = role == RAD_OPD_INOUT && !inout0[k].empty()
                                              ? (const void*)inout0[k].data() : nullptr;
                        if (src) {
                            if (rad_memcpy_async(fdev[k].p, src, fdev[k].bytes, stream) < 0)
                                why = "upload failed";
                        } else if (rad_memset_async(fdev[k].p, 0, fdev[k].bytes, stream) < 0) {
                            why = "memset failed";
                        }
                    }
                    if (why.empty() && rad_stream_sync(stream) < 0) why = "stream sync failed";

                    /* Walk the chain. The hook reports its own length by declining the step past
                     * the end, so nothing here can disagree with it about how many there are. */
                    for (int step = 0; step < RAD_MAX_FUSE_OPD && why.empty(); ++step) {
                        RadFuseStep fs{};
                        /* THE REAL OPERAND SET, not RAD_FUSE_PRESENT_ALL. This tool allocated
                         * those buffers from a description that says which are absent, so it
                         * KNOWS, and a chain described for operands nobody allocated would read a
                         * buffer that does not exist. */
                        uint64_t present = 0;
                        for (size_t k = 0; k < sh.size() && k < 64; ++k)
                            if (!sh[k].absent) present |= (uint64_t)1 << k;
                        const int fst = r.row->info->replaces(g.params(), g.n_params(),
                                                              present, step, &fs);
                        if (fst == RAD_E_NOTFOUND) {
                            if (step == 0) why = "declares an empty chain";
                            break;
                        }
                        if (fst < 0) {
                            why = std::string("the chain declines this geometry: ") +
                                  rad_strerror(fst);
                            break;
                        }
                        if (!fs.op) { why = "a step with no op name"; break; }

                        Geometry sg;
                        for (int pi = 0; pi < fs.n_p && pi < RAD_MAX_FUSE_PARAMS; ++pi) {
                            const RadParam& sp = fs.p[pi];
                            if (!sp.key) continue;
                            if (sp.kind == RAD_P_INT)      sg.set_i(sp.key, sp.ival);
                            else if (sp.kind == RAD_P_STR) sg.set_s(sp.key, sp.sval ? sp.sval : "");
                            else if (sp.kind == RAD_P_F64) sg.set_f(sp.key, sp.dval);
                        }

                        const KernelRow* srow = row_in(r.row->plugin, fs.op, sg, RAD_DOMAIN_DEVICE);
                        if (!srow) {
                            why = std::string("no ") + r.row->plugin + " kernel for step '" +
                                  fs.op + "' at " + sg.str();
                            break;
                        }
                        const RadOpSchema* ss = reg.schema(fs.op);
                        if (!ss) { why = std::string("step '") + fs.op + "' has no schema"; break; }
                        /* THE OPERAND COUNT IS THE DRIFT CHECK, exactly as it is for RadShapeFn: a
                         * hook that has fallen behind the vocabulary is a bug to name, not a
                         * decline to absorb. */
                        if (fs.n_from != ss->n_operands) {
                            why = fmt("step '%s' wires %d operands, its schema declares %d",
                                      fs.op, fs.n_from, ss->n_operands);
                            break;
                        }

                        /* The step's own description, for any temporary it names. */
                        std::vector<OpdShape> ssh;
                        const bool have_ssh =
                            rad_operand_shapes_hook(srow->info, sg, ss->n_operands, &ssh) >= 0;

                        std::vector<RadTensor> st_t((size_t)ss->n_operands);
                        for (int oi = 0; oi < ss->n_operands && why.empty(); ++oi) {
                            const int16_t src = fs.from[oi];
                            if (src == RAD_FUSE_NONE) { st_t[(size_t)oi] = RadTensor{}; continue; }
                            if (RAD_FUSE_IS_TMP(src)) {
                                const int ti = RAD_FUSE_TMP_INDEX(src);
                                if (ti < 0 || ti >= RAD_MAX_FUSE_TMP) {
                                    why = "temporary index out of range"; break;
                                }
                                if (!ftmp[(size_t)ti].p) {
                                    if (!have_ssh || (size_t)oi >= ssh.size()) {
                                        why = fmt("step '%s' names temporary %d and does not "
                                                  "describe its operands", fs.op, ti);
                                        break;
                                    }
                                    const OpdShape& d = ssh[(size_t)oi];
                                    const int64_t nb =
                                        rad_dtype_bytes(d.dtype, rad_numel(d.shape));
                                    if (nb <= 0) { why = "a temporary of no size"; break; }
                                    ftmp[(size_t)ti].p = rad_dev_alloc(nb, RAD_MEM_DEVICE);
                                    if (!ftmp[(size_t)ti].p) { why = "no memory for a temporary";
                                                               break; }
                                    ftmp[(size_t)ti].bytes = nb;
                                    rad_memset_async(ftmp[(size_t)ti].p, 0, nb, stream);
                                    RadTensor t{};
                                    t.data = ftmp[(size_t)ti].p;
                                    t.dtype = d.dtype;
                                    t.rank = (uint32_t)d.shape.size();
                                    for (size_t dd = 0; dd < d.shape.size() && dd < RAD_MAX_RANK;
                                         ++dd)
                                        t.shape[dd] = d.shape[dd];
                                    rad_tensor_pack(&t);
                                    ftmp_t[(size_t)ti] = t;
                                }
                                st_t[(size_t)oi] = ftmp_t[(size_t)ti];
                                continue;
                            }
                            if (src < 0 || (size_t)src >= sh.size()) {
                                why = "an operand index outside the fused kernel's list"; break;
                            }
                            if (sh[(size_t)src].absent) { st_t[(size_t)oi] = RadTensor{}; continue; }
                            /* A SHARED WEIGHT MUST BE READ THE WAY THE FUSED KERNEL READS IT.
                             *
                             * Weight operands are handed to the chain as the fused run's own
                             * device buffers, holding whatever layout the container was written in
                             * -- fragment order, for every fp8 GEMM here. libr4d has row-major
                             * readers of the same op (`gemm_fp8a16_nt_m1` wants the checkpoint's
                             * own bytes), and a step that landed on one would read a permuted
                             * plane as rows and report a difference that is the harness's fault
                             * rather than the kernel's. Same class of false failure the layout
                             * skip exists to prevent, arriving by a different route -- and tested
                             * only where a weight is actually shared, since a step with no weight
                             * operands has no layout to agree about. */
                            if (o.schema->operands[(size_t)src].role == RAD_OPD_WEIGHT &&
                                srow->info->layout != r.row->info->layout) {
                                why = fmt("step '%s' resolved to %s, which reads '%s' in a "
                                          "different layout from the fused kernel", fs.op,
                                          srow->info->name,
                                          o.schema->operands[(size_t)src].name
                                              ? o.schema->operands[(size_t)src].name : "?");
                                break;
                            }
                            /* The fused kernel's own tensor, repointed at whichever device buffer
                             * this side of the comparison owns: its own for what the chain writes,
                             * the shared one for what both only read. */
                            RadTensor t = t_test[(size_t)src];
                            t.data = fdev[(size_t)src].p ? fdev[(size_t)src].p
                                                         : dev[(size_t)src].p;
                            if (!t.data) { why = "a step operand with no buffer"; break; }
                            st_t[(size_t)oi] = t;
                        }
                        if (!why.empty()) break;

                        RadArgs sa{};
                        sa.t = st_t.data(); sa.n_t = (int)st_t.size();
                        sa.p = sg.params(); sa.n_p = sg.n_params();
                        sa.rank = 0; sa.world_size = 1;
                        if (srow->info->scratch) {
                            const int64_t need = srow->info->scratch(&sa);
                            if (need > fscratch.bytes) {
                                if (fscratch.p) rad_dev_free(fscratch.p, RAD_MEM_DEVICE);
                                fscratch.p = rad_dev_alloc(need, RAD_MEM_DEVICE);
                                fscratch.bytes = fscratch.p ? need : 0;
                            }
                            if (need > 0 && !fscratch.p) { why = "no scratch for a step"; break; }
                            if (need > 0) rad_memset_async(fscratch.p, 0, need, stream);
                            sa.scratch = fscratch.p; sa.scratch_bytes = fscratch.bytes;
                        }
                        const int sst = srow->info->launch(&sa, stream);
                        if (sst < 0) {
                            why = fmt("step '%s' (%s) refused: %s", fs.op, srow->info->name,
                                      rad_strerror(sst));
                            break;
                        }
                        if (rad_stream_sync(stream) < 0) { why = "a step faulted"; break; }
                    }

                    /* Bring the chain's outputs back and compare them with the fused kernel's,
                     * which are already in bufs[] from the copy-back above. */
                    size_t n_cmp = 0, n_diff = 0;
                    std::string first_diff;
                    if (why.empty()) {
                        for (size_t k = 0; k < sh.size(); ++k) {
                            if (!fdev[k].p) continue;
                            fout[k].assign((size_t)fdev[k].bytes, 0);
                            rad_memcpy_async(fout[k].data(), fdev[k].p, fdev[k].bytes, stream);
                        }
                        if (rad_stream_sync(stream) < 0) why = "copy-back failed";
                    }
                    if (why.empty()) {
                        for (size_t k = 0; k < sh.size(); ++k) {
                            if (fout[k].empty() || fout[k].size() != bufs[k].size()) continue;
                            ++n_cmp;
                            if (std::memcmp(fout[k].data(), bufs[k].data(), fout[k].size()) != 0) {
                                ++n_diff;
                                if (first_diff.empty())
                                    first_diff = o.schema->operands[k].name
                                                     ? o.schema->operands[k].name : "?";
                            }
                        }
                        if (n_cmp == 0) why = "nothing comparable came back";
                    }

                    for (auto& d : fdev) if (d.p) rad_dev_free(d.p, RAD_MEM_DEVICE);
                    for (auto& d : ftmp) if (d.p) rad_dev_free(d.p, RAD_MEM_DEVICE);
                    if (fscratch.p) rad_dev_free(fscratch.p, RAD_MEM_DEVICE);

                    if (!why.empty()) {
                        fuse_skipped[o.op + "  (" + std::string(r.row->info->name) + ": " +
                                     why + ")"]++;
                    } else if (n_diff) {
                        ++fuse_failed;
                        std::printf("  %-22s %-40s FUSION DIFFERS from the chain it declares: "
                                    "%zu of %zu operand(s), first '%s'  [%s]\n",
                                    o.op.c_str(), g.str().c_str(), n_diff, n_cmp,
                                    first_diff.c_str(), r.row->info->name);
                    } else {
                        ++fuse_passed;
                    }
                }


                /* Scratch is whatever launch_dev already allocated, so the timed loop allocates
                 * nothing -- an allocation inside a timed region measures the allocator. The sync
                 * is INSIDE the timed region: without it the number is the queue's submission
                 * latency, which is sub-microsecond and the same for every shape in the table. */
                double case_us = -1, case_spread = -1;
                if (bench && dom == RAD_DOMAIN_DEVICE && s_test >= 0 && bench_iters > 0) {
                    RadArgs a{};
                    a.t = t_test.data(); a.n_t = (int)t_test.size();
                    a.p = g.params(); a.n_p = g.n_params();
                    a.rank = 0; a.world_size = 1;
                    a.scratch = dev_scratch.p; a.scratch_bytes = dev_scratch.bytes;
                    case_us = time_launch(r.row->info, &a, stream, bench_iters, bench_reps,
                                          bench_warm_ms, &case_spread);
                }

                /* And only NOW are the tensors repointed at the host buffers, so everything after
                 * this line is domain-independent. The bytes they name were captured above, before
                 * any timed launch could overwrite them. */
                if (dom == RAD_DOMAIN_DEVICE && s_test >= 0)
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (dev[k].p) t_test[k].data = bufs[k].data();
                dev_release();

                /* A FUSION-ONLY CASE STOPS HERE: there is no reference for it and the chain
                 * comparison above already gave its verdict. Counting it under `checked` as well
                 * would be counting one case twice under two different questions. */
                if (fuse_only) continue;

                if (s_ref < 0) {
                    std::printf("  %-22s %-40s ORACLE REFUSED: %s\n", o.op.c_str(),
                                g.str().c_str(), rad_strerror(s_ref));
                    ++st.skipped;
                    continue;
                }
                if (s_test < 0) {
                    /* A kernel returning negative at issue is a bug in SELECTION, not a runtime
                     * condition (spec §17). The selector said this kernel serves this geometry
                     * and it does not, so this is a failure and not a skip. */
                    std::printf("  %-22s %-40s FAIL  %s/%s refused the geometry it was selected "
                                "for: %s\n", o.op.c_str(), g.str().c_str(),
                                r.row->plugin.c_str(), r.row->info->name, rad_strerror(s_test));
                    ++st.failed;
                    continue;
                }

                /* Compare every OUT and INOUT operand, AND NAME THE WORST ONE. An op with five
                 * outputs reports one number, and which output carries it is most of the
                 * diagnosis: `gdn_conv_update` writing a correct q and k and a wrong conv_state is
                 * a different bug from one that writes all three wrong, and the summary line alone
                 * cannot tell them apart. */
                Err worst;
                /* THE NON-FINITE COUNT IS SUMMED, NOT TAKEN FROM THE WORST OPERAND. The worst one
                 * is picked by rel_l2 so the line names the output furthest off; a NaN on one side
                 * only is not in rel_l2 at all, so an output carrying one and a later output with
                 * a larger (but passing) error would otherwise drop it and pass the case. */
                int64_t nonfinite = 0;
                /* Each operand is judged against its own tolerance; `worst` is a failing operand
                 * before a passing one, the larger error between two of the same verdict. */
                double worst_tol = tol_override >= 0
                                       ? tol_override
                                       : tolerance_for(o.op, g.get_s("dtype"), nullptr);
                bool worst_bad = false, all_ok = true;

                std::string per_opd;
                int n_out = 0;
                for (size_t k = 0; k < sh.size(); ++k) {
                    const int role = o.schema->operands[k].role;
                    if (sh[k].absent) continue;
                    if (role != RAD_OPD_OUT && role != RAD_OPD_INOUT) continue;
                    const int64_t n = rad_numel(sh[k].shape);
                    std::vector<float> a, bref;
                    rad_widen(bufs[k].data(), sh[k].dtype, n, &a);
                    rad_widen(ref_out[k].data(), sh[k].dtype, n, &bref);
                    const Err e = compare(a.data(), bref.data(), n);
                    const char* nm = o.schema->operands[k].name ? o.schema->operands[k].name : "?";
                    per_opd += fmt("%s%s=%.3e", per_opd.empty() ? "" : " ", nm, e.rel_l2);
                    if (e.n_nonfinite_mismatch)
                        per_opd += fmt("(%lld non-finite)", (long long)e.n_nonfinite_mismatch);
                    nonfinite += e.n_nonfinite_mismatch;
                    const double tk = tol_override >= 0
                                          ? tol_override
                                          : tolerance_for(o.op, g.get_s("dtype"), nm);
                    /* Written so NaN FAILS. The natural spelling of the opposite test,
                     * `rel > tol || rel < 0`, is false for NaN and lets a NaN error pass; this
                     * form cannot. */
                    const bool bad = !(e.rel_l2 >= 0 && e.rel_l2 <= tk);
                    all_ok = all_ok && !bad;
                    /* Written so a NaN error, once it is the worst, stays the worst. */
                    if (!std::isnan(worst.rel_l2) &&
                        ((bad && !worst_bad) ||
                         (bad == worst_bad && !(e.rel_l2 <= worst.rel_l2)))) {
                        worst = e;
                        worst_tol = tk;
                        worst_bad = bad;
                    }
                    ++n_out;
                    /* AND, IF SOMEBODY IS RECORDING, THE ORACLE'S ANSWER GOES INTO THE FIXTURE.
                     *
                     * Reduced here, from the widened plane that is already in hand, rather than in
                     * a second pass: the sum of squares is over every element and a second walk of
                     * a 31 MB output to compute a number this loop already has in front of it is
                     * the sort of thing that makes a recorder too slow to run. */
                    if (recording) {
                        KfOperand& d = kop[k];
                        d.have_ref = true;
                        d.ref_n    = n;
                        kf_reduce(bref.data(), n, kf_operand_seed(case_seed, 0x5000 + (int)k),
                                  &d.ref_sumsq, &d.ref_nonfinite, &d.sample);
                    }
                }

                if (n_out == 0) { ++st.skipped; continue; }
                worst.n_nonfinite_mismatch = nonfinite;

                const double tol = worst_tol;
                const bool ok = worst.n_nonfinite_mismatch == 0 && all_ok;

                ++st.checked;
                st.worst = std::max(st.worst, worst.rel_l2);
                if (ok) ++st.passed; else ++st.failed;

                {
                    Row rw;
                    rw.op = o.op;
                    rw.kernel = r.row->info->name;
                    rw.plugin = r.row->plugin;
                    rw.geom = g.str();
                    rw.domain = domain_name(dom);
                    rw.choice = r.choice;
                    rw.verdict = ok ? "ok" : "**FAIL**";
                    rw.rel_l2 = worst.rel_l2;
                    rw.tol = tol;
                    rw.us = case_us;
                    rw.us_spread = case_spread;
                    rw.ref_us = ref_us;
                    rw.synthesised = !from_container;
                    /* The GB/s denominator, and it names its own basis: the bytes ONE launch
                     * touches once, summed over every operand it was actually given. Not a FLOP
                     * count -- this tool does not know a kernel's arithmetic and will not infer it
                     * from an op name -- and not a serving figure, for the cache reason the report
                     * header states. */
                    for (size_t k = 0; k < sh.size(); ++k)
                        if (!sh[k].absent) rw.bytes += (int64_t)bufs[k].size();
                    report_rows.push_back(std::move(rw));
                    if (recording) {
                        KfCase c;
                        c.op   = o.op;
                        c.seed = case_seed;
                        c.m    = M;
                        c.band = o.bands[b].span;
                        c.opd  = kop;
                        /* TYPED, not `Geometry::str()` parsed back. There is no parser and
                         * writing one means deciding on every load, forever, whether `1e+07` is a
                         * rope theta or a string. */
                        for (int pi2 = 0; pi2 < g.n_params(); ++pi2) {
                            const RadParam& sp = g.params()[pi2];
                            if (!sp.key) continue;
                            KfParam kp;
                            kp.key  = sp.key;
                            kp.kind = sp.kind;
                            if (sp.kind == RAD_P_INT)      kp.i = sp.ival;
                            else if (sp.kind == RAD_P_F64) kp.d = sp.dval;
                            else if (sp.kind == RAD_P_STR) kp.s = sp.sval ? sp.sval : "";
                            else continue;   /* a RANGE is not a point and cannot be replayed */
                            c.params.push_back(std::move(kp));
                        }
                        fixture.cases.push_back(std::move(c));
                    }
                }

                if (!ok || verbose) {
                    std::printf("  %-22s %-44s %-8s %-16s rel_l2=%.3e max_abs=%.3e max_rel=%.3e "
                                "tol=%.1e %s%s\n",
                                o.op.c_str(), g.str().c_str(), domain_name(dom),
                                r.row->info->name, worst.rel_l2, worst.max_abs, worst.max_rel,
                                tol, ok ? "ok" : "FAIL",
                                from_container ? "" : "  [weights synthesised: not in the "
                                                      "container]");
                    if (n_out > 1)
                        std::printf("      per operand: %s\n", per_opd.c_str());
                    if (worst.n_nonfinite_mismatch)
                        std::printf("      %lld element(s) are finite on one side and not the "
                                    "other. That is not a tolerance question.\n",
                                    (long long)worst.n_nonfinite_mismatch);
                }
            }
        }
    }

    if (!skipped_ops.empty()) {
        std::printf("\nskipped\n");
        for (const auto& [what, n] : skipped_ops)
            std::printf("  %-70s x%d\n", what.c_str(), n);
    }

    /* THE FUSION ORACLE'S OWN TALLY. Kept separate from the counts above on purpose: this pass
     * answers "is the fused kernel byte-identical to the chain it declares", and the other answers
     * "is the kernel within tolerance of the reference". Folding them would make one number stand
     * for two different claims, and the byte-identical one is the stronger of the two. */
    if (fuse_check && (fuse_passed || fuse_failed || !fuse_skipped.empty())) {
        std::printf("\nfusions: %d byte-identical to their declared chain, %d differ\n",
                    fuse_passed, fuse_failed);
        if (!fuse_skipped.empty()) {
            std::printf("  not checked\n");
            for (const auto& [what, n] : fuse_skipped)
                std::printf("    %-70s x%d\n", what.c_str(), n);
        }
    }

    /* UNCONDITIONAL ON EVERY OTHER TALLY. An out-of-bounds write is the most serious thing this
     * tool can find -- it corrupts a neighbour and surfaces later as wrong output from a kernel
     * that is correct -- so it is not reported as a detail of some other result. */
    if (!overruns.empty()) {
        std::printf("\nOUT OF BOUNDS -- a kernel wrote outside what it was given\n");
        for (const auto& [what, n] : overruns)
            std::printf("  %-70s x%d\n", what.c_str(), n);
    }

    if (!diverged.empty()) {
        std::printf("\ndescribed differently by the two libraries (not a correctness result)\n");
        for (const auto& [what, n] : diverged)
            std::printf("  %-70s x%d\n", what.c_str(), n);
    }

    std::printf("\n%d checked, %d passed, %d failed, %d skipped, %d diverged, %d capped. "
                "worst rel_l2 = %.3e\n", st.checked, st.passed, st.failed, st.skipped,
                st.diverged, st.capped, st.worst);
    std::printf("memory: %d case(s) ran inside guarded allocations, %d wrote outside one\n",
                st.guarded, st.overrun);
    print_row_coverage(census(reg, ref_name), report_rows);
    if (st.failed == 0 && st.checked > 0)
        std::printf("every resolved kernel agrees with %s on this container's bytes\n",
                    ref_name.c_str());
    if (st.checked == 0)
        std::printf("NOTHING WAS CHECKED. Either every op resolved to %s (so there is nothing to "
                    "falsify) or no op has a shape recipe.\n", ref_name.c_str());
    if (!report_path.empty()) {
        /* The ops THIS container's declared program asks for -- the other half of the
         * completeness question. `ops declared` counts band-resolved entries and repeats an op
         * once per layer, so it is the distinct set that is wanted here. */
        std::vector<std::string> graph_ops;
        {
            std::set<std::string> seen;
            for (size_t oi = 1; oi < prog.ops.size(); ++oi)
                if (prog.ops[oi].schema) seen.insert(prog.ops[oi].op);
            graph_ops.assign(seen.begin(), seen.end());
        }
        if (write_report(report_path, report_rows, skipped_ops, fuse_skipped, diverged, overruns,
                         analysis_path,
                         "container: `" + model + "` (weights read from it, in the layout each\n  resolved kernel asked for)",
                         fuse_passed, fuse_failed, st, model, ref_name,
                         plugins.libraries(), plugins.arch_plugin, graph_ops,
                         census(reg, ref_name), bench) >= 0)
            std::printf("report written to %s (%zu row(s))\n", report_path.c_str(),
                        report_rows.size());
    }
    if (recording) {
        /* The date, so a fixture that has fallen behind the kernels is visible in the report
         * rather than inferred from a git log nobody reads. */
        char day[32] = { 0 };
        const std::time_t now = std::time(nullptr);
        std::strftime(day, sizeof day, "%Y-%m-%d", std::gmtime(&now));
        fixture.recorded = day;
        const int rc = fixture.save(record_path);
        if (rc < 0) { RAD_ERR("--record %s: %s", record_path.c_str(), rad_strerror(rc)); return 1; }
        std::printf("recorded %zu case(s) to %s\n", fixture.cases.size(), record_path.c_str());
    }
    return st.failed ? 1 : 0;
}
