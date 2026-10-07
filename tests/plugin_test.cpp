/* plugin_test.cpp -- the plugin layer: loading, the hierarchy, op schemas, selection, bucket
 * tables, the selection table and per-instance init. spec.md §2 and §16.
 *
 * The plugins it loads are core/plugin/testplugin/, built as real .so files: a loader tested
 * against a mock is a loader that has never been dlopen'd. Their rows are chosen so that the
 * assertions here are the spec's own claims -- §2.2's worked example is asserted band for band, so
 * the paragraph is a regression test rather than a paragraph.
 *
 * A passing run PRINTS error and warning lines. They are the diagnostics under test: a refused
 * plugin and a missing host kernel are meant to say so out loud, and asserting on a message nobody
 * ever sees is how a message rots.
 */
#include "rad_test.h"

#include "plugin/registry.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <string>

using namespace rad;

/* ------------------------------------------------------------------ fixtures */
namespace {

/* $RADIANCE_HOME is what ctest sets; RAD_TESTPLUGIN_HOME overrides it for a hand-built run. */
std::string home() {
    if (const char* d = std::getenv("RAD_TESTPLUGIN_HOME")) return d;
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}
std::string kernels_dir() { return home() + "/testkernels"; }
std::string bad_dir()     { return home() + "/testkernels/bad"; }
std::string order_dir()   { return home() + "/testkernels/order"; }
std::string arch_dir()    { return home() + "/testarch"; }

/* The hierarchy every test but the hierarchy tests uses: the override first, the base behind it. */
int load(Registry& r, std::vector<std::string> hierarchy = { "radtest_over", "radtest_base" }) {
    const int st = r.load_dir(kernels_dir(), hierarchy);
    if (st == RAD_E_IO)
        fprintf(stderr, "    (no test plugins in %s -- build the radtest_* targets)\n",
                kernels_dir().c_str());
    return st;
}

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

std::string errs(const Registry& r) {
    std::string s;
    for (const auto& e : r.load_errors()) { s += e; s += '\n'; }
    return s;
}

Geometry gemm_geom(long long M, long long K = 4096, const char* dtype = "bf16") {
    Geometry g;
    g.set_i("M", M);
    g.set_i("N", 4096);
    g.set_i("K", K);
    g.set_s("dtype", dtype);
    return g;
}

/* A t_attn geometry every constraint of t_attn_paged holds for, WITHOUT the derived block_size --
 * which is how a caller actually asks, since block_size is what it is trying to find out. The
 * operator tests break exactly one key at a time, so the first unmet constraint is the one under
 * test. */
Geometry attn_geom() {
    Geometry g;
    g.set_i("q_len", 8);
    g.set_i("head_dim", 128);
    g.set_i("window", 0);
    g.set_i("causal", 1);
    g.set_i("n_head", 8);
    g.set_s("kv_dtype", "bf16");
    return g;
}

/* The same question with the derived key PINNED by the caller. */
Geometry attn_geom_pinned(long long block_size) {
    Geometry g = attn_geom();
    g.set_i("block_size", block_size);
    return g;
}

std::string picked(const KernelRow* k) { return k ? k->info->name : std::string("<none>"); }

/* The reason select() gives for a refusal, which is the miss column of §16's table. */
std::string refusal(Registry& r, const char* op, const Geometry& g, int domain = RAD_DOMAIN_DEVICE) {
    SelectionTrace t;
    const KernelRow* k = r.select(op, g, domain, &t);
    return k ? std::string("<resolved to ") + k->info->name + ">" : t.reason;
}

RadParam pint(const char* k, long long v)   { return RadParam{ k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pstr(const char* k, const char* v) { return RadParam{ k, RAD_P_STR, 0, 0, v, 0.0 }; }
RadParam pf64(const char* k, double v)      { return RadParam{ k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam prange(const char* k, long long lo, long long hi) {
    return RadParam{ k, RAD_P_RANGE, lo, hi, nullptr, 0.0 };
}

}  /* namespace */

/* ================================================================== loading and the hierarchy */

TEST(loading_reads_identity_and_calls_open) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(r.plugins().size(), (size_t)2);

    const Plugin* base = r.plugin("radtest_base");
    CHECK(base != nullptr);
    if (!base) return;
    CHECK_EQ((int)base->kind, RAD_PLUGIN_KERNEL);
    CHECK_EQ(base->version, "0.1");
    CHECK_EQ(base->n_kernels, 17);
    CHECK_EQ(base->n_schemas, 10);

    /* The optional open hook runs once, after load and before any selection. */
    auto open_calls = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_open_calls"));
    CHECK(open_calls != nullptr);
    if (open_calls) CHECK(open_calls() >= 1);
}

/* The hierarchy is CONFIGURED, not discovered: a named plugin takes its position and one present
 * on disk but absent from the list loads at the end, after everything named. */
TEST(hierarchy_order_is_configured) {
    Registry r;
    CHECK_OK(r.load_dir(kernels_dir(), { "radtest_base" }));
    const Plugin* base = r.plugin("radtest_base");
    const Plugin* over = r.plugin("radtest_over");
    CHECK(base && over);
    if (!base || !over) return;
    CHECK_EQ(base->order, 0);
    CHECK_EQ(over->order, 1);      /* unnamed, so it loads behind everything named */

    Registry r2;
    CHECK_OK(r2.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    CHECK_EQ(r2.plugin("radtest_over")->order, 0);
    CHECK_EQ(r2.plugin("radtest_base")->order, 1);
}

/* AND WITH NOTHING NAMED, THE REFERENCE LIBRARY IS STILL LAST. It declares a kernel for every op
 * in the vocabulary, so a plugin ranked behind it can never be selected: an unnamed kernel library
 * for this machine's card would load, win nothing, and leave the whole model resolving to the
 * naive implementation while the engine reported a full plugin set.
 *
 * The fixture is built so that alphabetical order alone would get this wrong -- `radtest_ord_a.so`
 * is the one declaring itself a reference (RadPluginReferenceFn) and is staged first -- which is
 * what makes the assertion mean something. Both plugins declare the same schemas, which is not a
 * conflict. */
TEST(the_reference_library_sorts_last_with_no_hierarchy) {
    Registry r;
    const int st = r.load_dir(order_dir(), {});
    if (st == RAD_E_IO) {
        fprintf(stderr, "    (no order fixture in %s -- build the radtest_ord_* targets)\n",
                order_dir().c_str());
        return;
    }
    CHECK_OK(st);
    const Plugin* ref   = r.plugin("radtest_reference");
    const Plugin* other = r.plugin("radtest_base");
    CHECK(ref && other);
    if (!ref || !other) return;
    CHECK(ref->reference);
    CHECK(!other->reference);
    CHECK(other->order < ref->order);

    /* Naming it is still an override: an operator comparing a library against the reference puts
     * the reference in front on purpose, and the pin must not defeat that. */
    Registry r2;
    CHECK_OK(r2.load_dir(order_dir(), { "radtest_reference" }));
    CHECK_EQ(r2.plugin("radtest_reference")->order, 0);
    CHECK(r2.plugin("radtest_base")->order > 0);
}

TEST(abi_mismatch_is_refused_by_name_with_both_versions) {
    Registry r;
    const int st = r.load_plugin(bad_dir() + "/radtest_badabi.so", 0);
    CHECK_EQ(st, RAD_E_ABI);
    CHECK_EQ(r.plugins().size(), (size_t)0);
    const std::string e = errs(r);
    CHECK(has(e, "radtest_badabi.so"));
    CHECK(has(e, fmt("ABI version %u", (unsigned)RAD_ABI_VERSION + 7u)));
    CHECK(has(e, fmt("this engine is version %u", (unsigned)RAD_ABI_VERSION)));
}

/* Two disagreeing schemas would make the same positional call mean different things depending on
 * which plugin won selection. Refused at load, with both plugins named. */
TEST(schema_conflict_is_refused_with_both_plugins_named) {
    Registry r;
    CHECK_OK(load(r));
    const size_t rows_before = r.rows().size();
    const Plugin* owner = r.schema_owner("t_gemm");
    CHECK(owner != nullptr);

    const int st = r.load_plugin(bad_dir() + "/radtest_conflict.so", 9);
    CHECK_EQ(st, RAD_E_DUPLICATE);
    CHECK(r.plugin("radtest_conflict") == nullptr);
    CHECK_EQ(r.rows().size(), rows_before);        /* nothing of a refused plugin is registered */

    const std::string e = errs(r);
    CHECK(has(e, "t_gemm"));
    CHECK(has(e, "radtest_conflict"));
    if (owner) CHECK(has(e, owner->name));

    /* Its priority-1000 row would have won every t_gemm question had it registered. */
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8), RAD_DOMAIN_DEVICE)), "t_gemm_m16");
}

/* An architecture plugin is loaded and IDENTIFIED here; the build component drives it. One plugin
 * implements one architecture id and no two may claim the same one, because the id is what selects
 * the plugin a model is built from. */
TEST(architecture_plugins_are_identified_and_deduplicated) {
    Registry r;
    const int st = r.load_dir(arch_dir(), {});
    CHECK_EQ(st, RAD_E_DUPLICATE);
    CHECK_EQ(r.plugins().size(), (size_t)1);

    const Plugin* a = r.plugin("radtest_arch");
    CHECK(a != nullptr);
    if (!a) return;
    CHECK_EQ((int)a->kind, RAD_PLUGIN_ARCH);
    CHECK_EQ(a->arch_id, "radtest.v1");
    CHECK(a->arch_declare != nullptr);
    CHECK(a->arch_step != nullptr);
    CHECK_EQ(a->n_kernels, 0);     /* it exports no kernel symbols, and that is not an error */

    const std::string e = errs(r);
    CHECK(has(e, "radtest.v1"));
    CHECK(has(e, "radtest_arch2"));
}

/* THE REPLY FORMAT IS AN OPTIONAL EXPORT. Absent, the loader leaves the slot null and the plugin
 * loads -- rad_arch_probe, which this plugin does not export either, shows the same rule -- and
 * the engine derives the format from the chat template. Present, it is resolved, and what it
 * returns is a description the reply parser can run. */
TEST(an_architecture_plugin_may_declare_its_reply_format) {
    Registry r;
    (void)r.load_dir(arch_dir(), {});
    const Plugin* a = r.plugin("radtest_arch");
    CHECK(a != nullptr);
    if (!a) return;
    CHECK(a->arch_probe == nullptr);
    REQUIRE(a->arch_chat_format != nullptr);
    const RadChatFormat* d = a->arch_chat_format(nullptr);
    REQUIRE(d != nullptr);
    CHECK_EQ(d->struct_size, (uint32_t)sizeof(RadChatFormat));

    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(d, &f, &why));
    CHECK_OK(chat_format_check(f, &why));
    CHECK_EQ(f.name, std::string("radtest-format"));
    CHECK_EQ(f.call_open, std::string("<call>"));
    CHECK_EQ(f.arg_value_prefix, std::string("="));
    CHECK(f.triggers == std::vector<std::string>{ "<call>" });

    /* And the parser reads a reply in it. */
    std::shared_ptr<const ReplyFormat> rf;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    const nlohmann::ordered_json tools = nlohmann::ordered_json::parse(
        R"([{"type":"function","function":{"name":"go","parameters":{"type":"object",)"
        R"("properties":{"n":{"type":"integer"},"s":{"type":"string"}}}}}])");
    ReplyOptions ro;
    ro.tools = &tools;
    std::shared_ptr<const ReplyParser> rp;
    CHECK_OK(ReplyParser::create(rf, ro, &rp, &why));
    const ReplyMessage m = reply_parse(*rp, "<reason>hm</reason>ok<call><fn go><arg n=3</arg>"
                                            "<arg s=<![CDATA[x<y]]></arg></fn></call>");
    CHECK_EQ(m.reasoning, std::string("hm"));
    CHECK_EQ(m.content, std::string("ok"));
    REQUIRE(m.calls.size() == 1);
    CHECK_EQ(m.calls[0].name, std::string("go"));
    CHECK_EQ(m.calls[0].arguments, std::string("{\"n\":3,\"s\":\"x<y\"}"));
}

TEST(every_row_s_op_has_a_schema) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_OK(r.check_complete());
    const auto ops = r.op_names();
    CHECK_EQ(ops.size(), (size_t)10);
    CHECK_EQ(ops[0], "t_attn");
}

/* ================================================================== op schemas */

TEST(schema_validation_accepts_the_declared_shape) {
    Registry r;
    CHECK_OK(load(r));
    const RadOpSchema* s = r.schema("t_gemm");
    CHECK(s != nullptr);
    if (!s) return;
    CHECK_EQ(s->n_params, 4);

    std::string err;
    RadParam ok[] = { pint("M", 8), pint("N", 4096), pint("K", 4096), pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_gemm", ok, 4, &err));

    /* A range is an integer that has not been collapsed yet, so it satisfies an integer slot. */
    RadParam ranged[] = { prange("M", 1, 8192), pint("N", 4096), pint("K", 4096),
                          pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_gemm", ranged, 4, &err));
}

TEST(schema_validation_names_the_near_miss) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam ok[] = { pint("M", 8), pint("N", 4096), pint("K", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemmm", ok, 4, &err), RAD_E_NOSCHEMA);
    CHECK(has(err, "did you mean 't_gemm'"));
    CHECK_EQ(r.near_miss("t_atn"), "t_attn");
    CHECK_EQ(r.near_miss("completely_different"), "");

    RadParam typo[] = { pint("M", 8), pint("N", 4096), pint("KK", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", typo, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "did you mean 'K'"));
}

TEST(schema_validation_rejects_a_bad_param_set) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam missing[] = { pint("M", 8), pint("N", 4096), pint("K", 4096) };
    CHECK_EQ(r.validate("t_gemm", missing, 3, &err), RAD_E_SCHEMA);
    CHECK(has(err, "dtype"));

    RadParam wrong_kind[] = { pstr("M", "eight"), pint("N", 4096), pint("K", 4096),
                              pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", wrong_kind, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "integer"));

    RadParam twice[] = { pint("M", 8), pint("M", 9), pint("N", 4096), pint("K", 4096),
                         pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", twice, 5, &err), RAD_E_SCHEMA);
    CHECK(has(err, "twice"));

    /* At most one ranged parameter per op: that is what keeps the bucket-table index a single
     * integer at issue. */
    RadParam two_ranges[] = { prange("M", 1, 64), pint("N", 4096), prange("K", 1, 64),
                              pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", two_ranges, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "at most one ranged parameter"));
}

/* A RAD_DERIVED parameter is not one the caller owes: demanding it would re-create the very
 * circularity it exists to break. Supplying it is allowed, and pins it. */
TEST(schema_validation_does_not_demand_a_derived_parameter) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam without[] = { pint("q_len", 8), pint("head_dim", 128), pint("window", 0),
                           pint("causal", 1), pint("n_head", 8), pstr("kv_dtype", "bf16") };
    CHECK_OK(r.validate("t_attn", without, 6, &err));

    RadParam with[] = { pint("q_len", 8), pint("head_dim", 128), pint("window", 0),
                        pint("block_size", 16), pint("causal", 1), pint("n_head", 8),
                        pstr("kv_dtype", "bf16") };
    CHECK_OK(r.validate("t_attn", with, 7, &err));
}

/* Constraints never match on a float, so the schema check is the only thing standing between
 * RAD_F64("eps", 1e-6) and an integer spelling of it that reaches the kernel as a zero. */
TEST(schema_validation_separates_a_float_from_an_integer) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam f64[] = { pint("M", 8), pint("n", 4096), pf64("eps", 1e-6), pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_norm", f64, 4, &err));

    RadParam as_int[] = { pint("M", 8), pint("n", 4096), pint("eps", 0), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_norm", as_int, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "declare it with RAD_F64()"));

    RadParam f64_for_int[] = { pf64("M", 8.0), pint("n", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_norm", f64_for_int, 3, &err), RAD_E_SCHEMA);
    CHECK(has(err, "is a float"));
}

/* ================================================================== selection */

TEST(selection_takes_the_highest_declared_priority) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE)), "t_gemm_m16");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(17), RAD_DOMAIN_DEVICE)), "t_gemm_m64");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(65), RAD_DOMAIN_DEVICE)), "t_gemm_any");

    /* K = 1000 is divisible by neither 256 nor 16, and w8a8 is in no row's set: both specialised
     * rows drop out and the unconstrained one serves it. */
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8, 1000), RAD_DOMAIN_DEVICE)), "t_gemm_any");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8, 4096, "w8a8"), RAD_DOMAIN_DEVICE)),
             "t_gemm_any");

    SelectionTrace t;
    r.select("t_gemm", gemm_geom(8, 1000), RAD_DOMAIN_DEVICE, &t);
    CHECK_EQ(t.candidates.size(), (size_t)3);      /* the host row is not a candidate */
    CHECK(t.reason.empty());
}

/* Specificity is declared, not derived, and a tie goes to the row declared first. */
TEST(selection_breaks_ties_on_declaration_order) {
    Registry r;
    CHECK_OK(load(r));
    Geometry g;
    g.set_i("M", 4);
    g.set_i("n", 1);
    CHECK_EQ(picked(r.select("t_tie", g, RAD_DOMAIN_DEVICE)), "t_tie_first");
}

/* THE HIERARCHY IS ABSOLUTE. radtest_over's t_norm row is unconstrained and priority 1;
 * radtest_base's is EQ n==4096 and priority 100. Whichever plugin is named first wins, and
 * priority never crosses the plugin boundary. */
TEST(hierarchy_beats_priority_and_specificity) {
    Geometry g;
    g.set_i("M", 1);
    g.set_i("n", 4096);
    g.set_s("dtype", "bf16");

    Registry over_first;
    CHECK_OK(over_first.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    CHECK_EQ(picked(over_first.select("t_norm", g, RAD_DOMAIN_DEVICE)), "t_over_norm");

    Registry base_first;
    CHECK_OK(base_first.load_dir(kernels_dir(), { "radtest_base", "radtest_over" }));
    CHECK_EQ(picked(base_first.select("t_norm", g, RAD_DOMAIN_DEVICE)), "t_norm_specialised");

    /* Domains resolve independently: radtest_over has no host row, so the host question falls
     * through to the plugin below it rather than being answered "none". */
    CHECK_EQ(picked(over_first.select("t_norm", g, RAD_DOMAIN_HOST)), "t_norm_host");
}

TEST(a_missing_domain_is_reported_not_substituted) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8), RAD_DOMAIN_HOST)), "t_gemm_host");

    SelectionTrace t;
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_HOST, &t);
    CHECK(k == nullptr);
    CHECK(t.other_domain_has_rows);
    CHECK(has(t.reason, "no host kernel"));
}

TEST(an_unknown_op_names_the_near_miss) {
    Registry r;
    CHECK_OK(load(r));
    const std::string why = refusal(r, "t_gem", gemm_geom(8));
    CHECK(has(why, "no loaded kernel plugin implements op 't_gem'"));
    CHECK(has(why, "did you mean 't_gemm'"));
}

/* Every constraint operator, and the sentence each one produces when it refuses. */
TEST(every_constraint_operator_refuses_with_its_own_reason) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE)), "t_attn_paged");

    Geometry g = attn_geom(); g.set_i("head_dim", 64);
    CHECK(has(refusal(r, "t_attn", g), "head_dim == 128, have 64"));

    g = attn_geom(); g.set_i("window", -1);
    CHECK(has(refusal(r, "t_attn", g), "window >= 0, have -1"));

    /* Pinned by the caller, and therefore matched normally rather than skipped. */
    CHECK(has(refusal(r, "t_attn", attn_geom_pinned(32)), "block_size == 16, have 32"));

    g = attn_geom(); g.set_i("causal", 2);
    CHECK(has(refusal(r, "t_attn", g), "causal in {0 1}, have 2"));

    g = attn_geom(); g.set_i("q_len", 17);
    CHECK(has(refusal(r, "t_attn", g), "q_len <= 16, have 17"));

    g = attn_geom(); g.set_i("n_head", 6);
    CHECK(has(refusal(r, "t_attn", g), "n_head % 4 == 0, have 6"));

    g = attn_geom(); g.set_s("kv_dtype", "f32");
    CHECK(has(refusal(r, "t_attn", g), "kv_dtype in {bf16 fp8_e4m3}, have f32"));

    /* A constraint whose key the geometry lacks does NOT hold -- libr4d's rule: a row that needs
     * a parameter the caller did not state is not a match. */
    Geometry partial;
    partial.set_i("q_len", 8);
    CHECK(has(refusal(r, "t_attn", partial), "did not name head_dim"));
}

/* RAD_DERIVED: the parameter the KERNEL supplies. Skipped during matching when the caller did not
 * name it -- otherwise spec §7.2 is circular, since the block manager reads the block size off the
 * resolved kernel and the kernel cannot resolve without it -- and written into the resolved
 * geometry once a winner exists. */
TEST(a_derived_parameter_is_skipped_then_supplied) {
    Registry r;
    CHECK_OK(load(r));

    SelectionTrace t;
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE, &t);
    CHECK_EQ(picked(k), "t_attn_paged");        /* would refuse "did not name block_size" without */
    CHECK_EQ(t.derived.size(), (size_t)1);
    if (t.derived.size() == 1) {
        CHECK_EQ(t.derived[0].key, "block_size");
        CHECK(t.derived[0].known);
        CHECK_EQ(t.derived[0].value, 16);
    }

    /* The resolved geometry carries it, so rad_kv_block_size() and RadArgs.p see one number. */
    Resolved res = r.resolve("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(res);
    long long bs = 0;
    CHECK(res.geom.get_i("block_size", &bs));
    CHECK_EQ(bs, 16);

    /* A caller that supplies it has pinned it: matched normally, so a mismatch is a refusal by
     * name rather than a silent substitution, and nothing is derived afterwards. */
    SelectionTrace pinned;
    CHECK_EQ(picked(r.select("t_attn", attn_geom_pinned(16), RAD_DOMAIN_DEVICE, &pinned)),
             "t_attn_paged");
    CHECK_EQ(pinned.derived.size(), (size_t)0);
    CHECK(r.select("t_attn", attn_geom_pinned(32), RAD_DOMAIN_DEVICE) == nullptr);
}

/* A winner with no RAD_C_EQ on the derived key -- ref's `block_size >= 1`, which admits every
 * value -- leaves it unset. "any" is not a number, and the caller has to decide rather than be
 * handed a zero that looks like an answer. */
TEST(a_derived_key_with_no_eq_stays_unset) {
    Registry r;
    CHECK_OK(load(r));

    SelectionTrace t;
    CHECK_EQ(picked(r.select("t_attn_ref", attn_geom(), RAD_DOMAIN_DEVICE, &t)), "t_attn_ref_any");
    CHECK_EQ(t.derived.size(), (size_t)1);
    if (t.derived.size() == 1) {
        CHECK_EQ(t.derived[0].key, "block_size");
        CHECK(!t.derived[0].known);
    }
    Resolved res = r.resolve("t_attn_ref", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(res);
    CHECK(!res.geom.has("block_size"));
}

TEST(bands_carry_the_derived_value_into_every_band) {
    Registry r;
    CHECK_OK(load(r));
    std::vector<Band> b = r.build_bands("t_attn", "q_len", 1, 64, attn_geom());
    /* q_len <= 16 is the only boundary the rows place on it, so (0,16] resolves and (16,64] does
     * not -- and the resolved band still carries the kernel's block size. */
    CHECK_EQ(b.size(), (size_t)2);
    if (b.size() != 2) return;
    CHECK_EQ(b[0].hi, 16);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_attn_paged");
    long long bs = 0;
    CHECK(b[0].dom[RAD_DOMAIN_DEVICE].geom.get_i("block_size", &bs));
    CHECK_EQ(bs, 16);
    CHECK(!b[1].dom[RAD_DOMAIN_DEVICE]);
}

/* §7.2: the block size comes from the resolved attention kernel's own constraints, not from a core
 * constant. This is the accessor the block manager reads it with. */
TEST(a_resolved_kernel_s_constraints_are_readable) {
    Registry r;
    CHECK_OK(load(r));
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(k != nullptr);
    if (!k) return;
    long long v = 0;
    CHECK(kernel_constraint(*k, "block_size", RAD_C_EQ, &v));
    CHECK_EQ(v, 16);
    CHECK(kernel_constraint(*k, "q_len", RAD_C_LE, &v));
    CHECK_EQ(v, 16);
    /* Absent is not a default: the kernel says nothing about window with EQ, and the caller has to
     * notice rather than being handed a zero. */
    CHECK(!kernel_constraint(*k, "window", RAD_C_EQ, &v));
    CHECK(!kernel_constraint(*k, "nonesuch", RAD_C_EQ, &v));
}

/* ================================================================== bucket tables */

/* spec.md §2.2's worked example, asserted band for band: three kernels constraining M<=16, M<=64
 * and nothing produce (0,16], (16,64], (64,max] and no policy was chosen by anyone. */
TEST(band_table_is_the_worked_example) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    std::vector<Band> b = r.build_bands("t_gemm", "M", 1, 8192, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;

    CHECK_EQ(b[0].hi, 16);
    CHECK_EQ(b[1].hi, 64);
    CHECK_EQ(b[2].hi, 8192);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m16");
    CHECK_EQ(picked(b[1].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m64");
    CHECK_EQ(picked(b[2].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_any");

    /* Both domains are resolved wherever both exist, so the core can change execution site at
     * runtime without re-resolving. */
    for (const Band& band : b) {
        CHECK_EQ(picked(band.dom[RAD_DOMAIN_HOST].row), "t_gemm_host");
        CHECK(band.miss[RAD_DOMAIN_DEVICE].empty());
    }

    /* Each band is frozen AT ITS UPPER BOUND, which is the worst case in the band. */
    long long m = 0;
    CHECK(b[1].dom[RAD_DOMAIN_DEVICE].geom.get_i("M", &m));
    CHECK_EQ(m, 64);
    CHECK_EQ(b[1].dom[RAD_DOMAIN_DEVICE].tune_source, "axis defaults");

}

/* A range with a hole in it is reported as that band, not as the whole op. The boundaries are the
 * constraint values themselves: LE 64 gives 64, GE 4096 ends the band below it at 4095. */
TEST(band_boundaries_come_from_the_constraints) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    std::vector<Band> b = r.build_bands("t_skinny", "M", 1, 8192, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;
    CHECK_EQ(b[0].hi, 64);
    CHECK_EQ(b[1].hi, 4095);
    CHECK_EQ(b[2].hi, 8192);

    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_skinny_m64");
    CHECK(!b[1].dom[RAD_DOMAIN_DEVICE]);
    CHECK(has(b[1].miss[RAD_DOMAIN_DEVICE], "M <= 64, have 4095"));
    CHECK(has(b[1].miss[RAD_DOMAIN_DEVICE], "M >= 4096, have 4095"));
    CHECK_EQ(picked(b[2].dom[RAD_DOMAIN_DEVICE].row), "t_skinny_big");

    /* No host kernel anywhere for this op: every band says which placement option that removed. */
    for (const Band& band : b) {
        CHECK(!band.dom[RAD_DOMAIN_HOST]);
        CHECK(has(band.miss[RAD_DOMAIN_HOST], "site 'host' is unavailable"));
    }
}

TEST(an_eq_constraint_cuts_both_edges) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("n", 1);

    std::vector<Band> b = r.build_bands("t_eq", "M", 1, 64, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;
    CHECK_EQ(b[0].hi, 31);
    CHECK_EQ(b[1].hi, 32);
    CHECK_EQ(b[2].hi, 64);
    CHECK(!b[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK_EQ(picked(b[1].dom[RAD_DOMAIN_DEVICE].row), "t_eq_m32");
    CHECK(!b[2].dom[RAD_DOMAIN_DEVICE]);
}

TEST(bands_clamp_to_the_declared_range) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    /* 16 is below the range and 64 is at or above its top: neither cuts anything. */
    std::vector<Band> narrow = r.build_bands("t_gemm", "M", 32, 48, base);
    CHECK_EQ(narrow.size(), (size_t)1);
    CHECK_EQ(narrow[0].hi, 48);
    CHECK_EQ(picked(narrow[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m64");

    std::vector<Band> tiny = r.build_bands("t_gemm", "M", 1, 16, base);
    CHECK_EQ(tiny.size(), (size_t)1);
    CHECK_EQ(tiny[0].hi, 16);
    CHECK_EQ(picked(tiny[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m16");
}

TEST(an_op_with_no_range_gets_one_band) {
    Registry r;
    CHECK_OK(load(r));
    std::vector<Band> b = r.build_bands("t_attn", "", 0, 0, attn_geom());
    CHECK_EQ(b.size(), (size_t)1);
    if (b.empty()) return;
    CHECK_EQ(b[0].hi, INT64_MAX);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_attn_paged");
    CHECK(!b[0].dom[RAD_DOMAIN_HOST]);
}

/* ================================================================== instances */

TEST(init_runs_once_per_resolved_instance) {
    Registry r;
    CHECK_OK(load(r));
    const Plugin* base = r.plugin("radtest_base");
    CHECK(base != nullptr);
    if (!base) return;
    auto n_init = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_init_calls"));
    auto n_fini = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_fini_calls"));
    auto inst_m = reinterpret_cast<long long (*)(void*)>(dlsym(base->handle, "radtest_instance_m"));
    CHECK(n_init && n_fini && inst_m);
    if (!n_init || !n_fini || !inst_m) return;
    const int init0 = n_init(), fini0 = n_fini();

    Geometry g;
    g.set_i("N", 4096);
    g.set_i("K", 4096);
    g.set_s("dtype", "bf16");
    std::vector<Band> b = r.build_bands("t_gemm", "M", 1, 8192, g);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;

    for (Band& band : b) CHECK_OK(r.init_instance(band.dom[RAD_DOMAIN_DEVICE], 0, 1));
    CHECK_EQ(n_init() - init0, 3);
    CHECK_EQ(r.n_instances(), (size_t)3);

    /* The instance saw the FROZEN geometry -- its own band's upper bound, not the range. */
    CHECK_EQ(inst_m(b[0].dom[RAD_DOMAIN_DEVICE].instance), 16);
    CHECK_EQ(inst_m(b[2].dom[RAD_DOMAIN_DEVICE].instance), 8192);

    /* The host rows have no init hook: that is RAD_OK and a null instance, not a call. */
    CHECK_OK(r.init_instance(b[0].dom[RAD_DOMAIN_HOST], 0, 1));
    CHECK(b[0].dom[RAD_DOMAIN_HOST].instance == nullptr);
    CHECK_EQ(n_init() - init0, 3);

    r.fini_instances();
    CHECK_EQ(n_fini() - fini0, 3);
    CHECK_EQ(r.n_instances(), (size_t)0);
}

TEST(a_negative_init_is_a_value_not_a_crash) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("n", 1);
    std::vector<Band> b = r.build_bands("t_eq", "M", 32, 32, base);
    CHECK_EQ(b.size(), (size_t)1);
    if (b.empty()) return;
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_eq_m32");
    CHECK_EQ(r.init_instance(b[0].dom[RAD_DOMAIN_DEVICE], 0, 1), RAD_E_NOMEM);
    CHECK_EQ(r.n_instances(), (size_t)0);
}

/* ================================================================== the selection table (§16) */

TEST(the_selection_table_folds_repeats_and_names_the_refusal) {
    Registry r;
    CHECK_OK(load(r));
    r.clear_selections();

    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE);
    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE);   /* the same question, folded */
    CHECK_EQ(r.n_selections(), (size_t)1);

    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_HOST);     /* a different domain is a different
                                                             * question */
    CHECK_EQ(r.n_selections(), (size_t)2);

    Geometry g = attn_geom();
    g.set_i("q_len", 17);
    r.select("t_attn", g, RAD_DOMAIN_DEVICE);
    CHECK_EQ(r.n_selections(), (size_t)3);

    r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK_EQ(r.n_selections(), (size_t)4);

    const std::string table = r.selection_table();
    CHECK(has(table, "4 distinct questions, 5 asked"));
    CHECK(has(table, "t_gemm_m16 (radtest_base, prio 30)"));
    CHECK(has(table, "M=16 N=4096 K=4096 dtype=bf16"));
    CHECK(has(table, "-- none: q_len <= 16, have 17"));
    /* A derived key was not part of the question, so it is printed as the kernel's answer and not
     * inside the geometry column. */
    CHECK(has(table, "derived block_size:=16"));
    CHECK(!has(table, "block_size=16 "));

    r.clear_selections();
    CHECK_EQ(r.n_selections(), (size_t)0);
}

/* A "device." CONSTRAINT IS READ FROM THE MACHINE, not from the query: the engine measures peer
 * access at launch and the selector holds a kernel that needs it to that. Unmeasured, the kernel
 * is not selected and the refusal says nothing measured it; measured absent, the next row in the
 * plugin takes the op, and an op with no such row is refused naming the fact. The query never
 * carries the key, so no geometry -- and no tuning key -- changes. */
TEST(a_device_fact_constraint_is_matched_against_the_machine) {
    Registry r;
    CHECK_OK(load(r));
    Geometry g;
    g.set_i("M", 4);
    g.set_i("n", 64);

    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_staged");
    CHECK(has(refusal(r, "t_peer_only", g), "device.p2p is not known"));

    r.set_device_fact(RAD_FACT_P2P, 1);
    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_direct");
    CHECK_EQ(picked(r.select("t_peer_only", g, RAD_DOMAIN_DEVICE)), "t_peer_only_direct");
    CHECK(!g.has(RAD_FACT_P2P));

    r.set_device_fact(RAD_FACT_P2P, 0);
    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_staged");
    const std::string why = refusal(r, "t_peer_only", g);
    CHECK(has(why, "device.p2p == 1, have 0"));
    CHECK(has(why, "measured at launch"));
}

/* THE SHIPPED libr4d PICKS THE ALL-REDUCE BY MESSAGE SIZE, and the selector is what says so. At 4
 * and 8 ranks the exact sum is a one-shot for a decode-sized message and a two-shot for a prefill
 * chunk; the two-shot rows start one element above the one-shot bound, so that bound is a band
 * edge of the blocks' ranged `numel`. The band at or under the edge has to stay resolvable for
 * every row that places a DIV on `numel` -- the two-rank wht6 wire (64) and the four-rank ti8 wire
 * (128) among them -- or a lossy two-rank serve would lose its decode band. Checked here against
 * the real plugin, because nothing else can run a four- or eight-rank resolution on a two-card
 * machine. Geometry is Qwen3.8-Flash-Next's: 2560 wide, a 2048-token chunk. */
TEST(libr4d_resolves_the_all_reduce_by_message_size) {
    const std::string so = home() + "/kernels/libr4d.so";
    if (access(so.c_str(), R_OK) != 0) {
        fprintf(stderr, "  SKIP %s: no %s (a build without HIP)\n", ::radtest::current(),
                so.c_str());
        return;
    }
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    r.set_device_fact(RAD_FACT_P2P, 1);

    const int64_t width = 2560, chunk = 2048;
    auto bands = [&](int world, int exact, int64_t max_tok) {
        Geometry g;
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_i("exact", exact);
        g.set_i("min_bytes", 128 * 1024);
        return r.build_bands("all_reduce", "numel", width, max_tok * width, g);
    };
    auto names = [](const std::vector<Band>& b) {
        std::string s;
        for (const Band& x : b)
            s += fmt("%lld:%s ", (long long)x.hi, picked(x.dom[RAD_DOMAIN_DEVICE].row).c_str());
        return s;
    };

    CHECK_EQ(names(bands(4, 1, chunk)),
             "524288:ar_oneshot_4rank_exact 5242880:ar_twoshot_4rank_exact ");
    CHECK_EQ(names(bands(8, 1, chunk)),
             "524288:ar_oneshot_8rank_exact 5242880:ar_twoshot_8rank_exact ");
    CHECK_EQ(names(bands(4, 0, chunk)),
             "524288:ar_twoshot_4rank_ti8 5242880:ar_twoshot_4rank_ti8 ");
    CHECK_EQ(names(bands(2, 0, chunk)),
             "524288:ar_oneshot_2rank_wht6 5242880:ar_oneshot_2rank_wht6 ");
    CHECK_EQ(names(bands(2, 1, chunk)),
             "524288:ar_oneshot_2rank_exact 5242880:ar_oneshot_2rank_exact ");

    /* Above the two-shot's cap the one-shot takes the message back. */
    CHECK_EQ(names(bands(4, 1, 16384)),
             "524288:ar_oneshot_4rank_exact 20971520:ar_twoshot_4rank_exact "
             "41943040:ar_oneshot_4rank_exact ");

    /* A step that never exceeds the bound has one band, and it is the one-shot's. */
    CHECK_EQ(names(bands(4, 1, 128)), "327680:ar_oneshot_4rank_exact ");

    /* Eight ranks have no lossy wire: every band is a miss that names the request. */
    const std::vector<Band> none = bands(8, 0, chunk);
    CHECK(!none.empty());
    for (const Band& b : none) {
        CHECK(!b.dom[RAD_DOMAIN_DEVICE]);
        CHECK(has(b.miss[RAD_DOMAIN_DEVICE], "exact"));
    }
}

/* THE FOUR-RANK FUSED WRITES SERVE THE EXACT WIRE ONLY, AND SAY SO IN SELECTION. A caller states
 * `wire` 0 for the exact form and gets the four-rank rows; a request for the rotated wire at four
 * ranks is a miss that names `wire`, so the caller keeps its unfused pair -- not a row that
 * matches and then refuses its instance, which would stop declare. At two ranks both wires keep
 * their row. Geometry is Qwen3.8-Flash-Next's: 2560 wide, four streams, top-k 10, declared over a
 * decode or verify step of 8 sequences at depth 3. */
TEST(libr4d_resolves_the_four_rank_fused_writes_on_the_exact_wire_only) {
    const std::string so = home() + "/kernels/libr4d.so";
    if (access(so.c_str(), R_OK) != 0) {
        fprintf(stderr, "  SKIP %s: no %s (a build without HIP)\n", ::radtest::current(),
                so.c_str());
        return;
    }
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    r.set_device_fact(RAD_FACT_P2P, 1);

    const int64_t rows = 8 * (1 + 3);
    auto bands = [&](const std::string& op, int world, int wire) {
        Geometry g;
        g.set_i("n", 2560);
        g.set_i("hc", 4);
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_i("wire", wire);
        if (op == "ar_gather_hc_write") {
            g.set_i("top_k", 10);
            g.set_s("act", "sigmoid");
        }
        return r.build_bands(op, "M", 1, rows, g);
    };
    auto names = [](const std::vector<Band>& b) {
        std::string s;
        for (const Band& x : b)
            s += fmt("%lld:%s ", (long long)x.hi, picked(x.dom[RAD_DOMAIN_DEVICE].row).c_str());
        return s;
    };

    CHECK_EQ(names(bands("ar_hc_write", 4, 0)), "32:ar_hc_write_4rank ");
    CHECK_EQ(names(bands("ar_gather_hc_write", 4, 0)), "32:ar_gather_hc_write_4rank ");
    for (int wire : { 0, 6 }) {
        CHECK_EQ(names(bands("ar_hc_write", 2, wire)), "32:ar_hc_write ");
        CHECK_EQ(names(bands("ar_gather_hc_write", 2, wire)), "32:ar_gather_hc_write ");
    }

    for (const char* op : { "ar_hc_write", "ar_gather_hc_write" }) {
        const std::vector<Band> none = bands(op, 4, 6);
        CHECK(!none.empty());
        for (const Band& b : none) {
            CHECK(!b.dom[RAD_DOMAIN_DEVICE]);
            CHECK(has(b.miss[RAD_DOMAIN_DEVICE], "wire"));
        }
    }
}

/* THE VISION TOWER'S ATTENTION RESOLVES AS rad_block_vit.h DECLARES IT: head_dim 72, no `window`.
 * `window` is optional in attn_dense's schema and a constraint on a key the geometry lacks does not
 * hold, so a row that constrains it is unreachable for a caller that leaves it out -- which is how
 * every Qwen image request on a card came to stop the engine with "no device kernel for this
 * band". rad-kbench cannot see that: its cases always name the window. */
TEST(libr4d_resolves_the_vision_attention_without_a_window) {
    const std::string so = home() + "/kernels/libr4d.so";
    if (access(so.c_str(), R_OK) != 0) {
        fprintf(stderr, "  SKIP %s: no %s (a build without HIP)\n", ::radtest::current(),
                so.c_str());
        return;
    }
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    Geometry g;
    g.set_i("head_dim", 72);
    g.set_i("gqa", 1);
    g.set_i("causal", 0);
    g.set_s("dtype", "bf16");
    g.set_i("max_seqlen", 16384);
    const std::vector<Band> b = r.build_bands("attn_dense", "M", 1, 16384, g);
    CHECK(!b.empty());
    for (const Band& x : b) CHECK_EQ(picked(x.dom[RAD_DOMAIN_DEVICE].row), "attn_vit_h72_bf16");
}

RAD_TEST_MAIN()
