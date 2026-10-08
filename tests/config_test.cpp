/* config_test.cpp -- the command line.
 *
 * WHAT THIS IS HOLDING DOWN. Every pool in this engine is budgeted explicitly (spec §6), and a
 * budget flag is the one kind of input where a typo and a decision look identical downstream:
 * nothing after this function can tell "the operator asked for 0.82" from "the operator asked for
 * 0.87 and it did not parse". The parser's whole job is to make that distinction at the door --
 * to refuse by name rather than to fall through to a default -- and a refusal that stops
 * refusing is silent everywhere else.
 *
 * Three groups of property here, and they fail differently:
 *
 *   TAKEN     the value reaches the field an operator thinks it reaches
 *   REFUSED   a value outside the legal set stops the server rather than being rounded into it
 *   RESOLVED  what is left unsaid comes from the environment or from a stated default
 *
 * No device, no model, no container: argc/argv in, a Config out.
 */
#include "rad_test.h"

#include "rad_core.h"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <cstring>
#include <vector>

using namespace rad;

namespace {

/* config_parse takes `char**`, so the argument strings have to be writable. Held by the caller
 * for the life of the call, which is what argv is. */
struct Args {
    std::vector<std::string> store;
    std::vector<char*>       argv;

    explicit Args(std::initializer_list<const char*> v) {
        store.emplace_back("radiance");           /* argv[0], which the parser skips */
        for (const char* s : v) store.emplace_back(s);
        for (std::string& s : store) argv.push_back(s.data());
    }
    int argc() const { return (int)argv.size(); }
};

/* Parse, with a model so that the required-argument check is not what every case trips over. */
int parse(std::initializer_list<const char*> v, Config* c) {
    std::vector<std::string> store{ "radiance", "--model", "/m/x.rad" };
    for (const char* s : v) store.emplace_back(s);
    std::vector<char*> argv;
    for (std::string& s : store) argv.push_back(s.data());
    return config_parse((int)argv.size(), argv.data(), c);
}

/* The same into a Config of its own, for a case whose answer depends on flags not carrying over
 * from an earlier parse. */
int parse_fresh(std::initializer_list<const char*> v) {
    Config c;
    return parse(v, &c);
}

/* The level `-v` moves and every other test in this binary reads. */
struct LogGuard {
    Log saved = log_level();
    ~LogGuard() { log_set_level(saved); }
};

}  /* namespace */

/* ================================================================== the shape of the call */

TEST(help_is_reported_separately_from_an_error) {
    /* 1 rather than a negative: the caller prints usage and exits ZERO. Folding it into the error
     * path would make `--help` a failed run in anything scripting the binary. */
    Config c;
    Args a{ "--help" };
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), 1);
    Args b{ "-h" };
    CHECK_EQ(config_parse(b.argc(), b.argv.data(), &c), 1);
    /* And it wins from anywhere on the line, including ahead of an otherwise fatal argument. */
    Args d{ "--tp", "0", "--help" };
    CHECK_EQ(config_parse(d.argc(), d.argv.data(), &c), 1);
}

/* AN UNKNOWN FLAG IS AN ERROR NAMING IT, NOT A WARNING. A typo in a budget flag is the difference
 * between the plan you asked for and one the planner invented. */
TEST(an_unknown_option_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--not-a-flag" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-num-seq", "8" }, &c), RAD_E_INVAL);   /* near-miss on a real flag */
    CHECK_EQ(parse({ "-x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "positional" }, &c), RAD_E_INVAL);
}

/* A RETIRED FLAG IS REFUSED, NOT ACCEPTED AND IGNORED. Each of these names a decision the engine
 * does not take from the operator: the KV cache is elastic on every run and derives its own
 * buffer, the activation arena is computed from the buffer plan and charged by name, and a
 * finished conversation is copied to host memory and on to disk as soon as it can be rather than
 * after an idle time. A command line carrying one expects behaviour the engine does not have, and
 * starting it anyway gives an operator a server that is not the one they asked for and no way to
 * tell. Refusing names the flag and stops; accepting and warning puts the
 * answer in a log nobody reads. */
TEST(a_retired_flag_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--kv-flex-mib", "512" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--vram-headroom-mib", "2000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-idle-to-host", "30" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-idle-to-disk", "1800" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-host-cache-mib", "4096" }, &c), RAD_E_INVAL);
}

/* A FLAG WHOSE ARGUMENT IS OFF THE END OF THE LINE. Reading past argv is the failure this
 * prevents, and the one shape that reaches it is the flag written last. */
TEST(a_flag_missing_its_argument_is_refused_rather_than_read_past) {
    Config c;
    CHECK_EQ(parse({ "--tp" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--port" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype" }, &c), RAD_E_INVAL);
    Args a{ "--model" };
    CHECK_EQ(config_parse(a.argc(), a.argv.data(), &c), RAD_E_INVAL);
}

/* THE KL MODE TAKES ONE DIRECTION AND WHAT THAT DIRECTION READS. Recording into a directory and
 * scoring against one in the same run would leave the engine to guess which the operator meant,
 * and guessing wrong makes a quantised model the yardstick. */
TEST(the_kl_mode_records_or_scores_and_says_which_input_it_lacks) {
    Config c;
    CHECK_EQ(parse({ "--kld-record", "/r", "--kld-corpus", "/c.jsonl" }, &c), RAD_OK);
    CHECK_EQ(c.kld_record, std::string("/r"));
    CHECK_EQ(c.kld_corpus, std::string("/c.jsonl"));
    CHECK(c.kld());

    Config d;
    CHECK_EQ(parse({ "--kld-ref", "/r", "--kld-out", "/o.json" }, &d), RAD_OK);
    CHECK_EQ(d.kld_ref, std::string("/r"));
    CHECK_EQ(d.kld_out, std::string("/o.json"));
    CHECK(d.kld());

    Config e;
    CHECK_EQ(parse({}, &e), RAD_OK);
    CHECK(!e.kld());

    Config f;
    CHECK_EQ(parse({ "--kld-record", "/r", "--kld-ref", "/s" }, &f), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kld-record", "/r" }, &f), RAD_E_INVAL);              /* no corpus */
    CHECK_EQ(parse({ "--kld-ref", "/r", "--kld-corpus", "/c" }, &f), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kld-out", "/o.json" }, &f), RAD_E_INVAL);            /* nothing to report */
    CHECK_EQ(parse({ "--kld-ref" }, &f), RAD_E_INVAL);
}

TEST(the_model_is_required_and_nothing_else_is) {
    Config c;
    Args none{};
    CHECK_EQ(config_parse(none.argc(), none.argv.data(), &c), RAD_E_INVAL);

    Config ok;
    CHECK_EQ(parse({}, &ok), RAD_OK);
    CHECK_EQ(ok.model, std::string("/m/x.rad"));
}

/* ================================================================== taken */

TEST(the_pool_size_flags_are_gone_and_say_what_replaced_them) {
    /* The card's split is the resolver's (core/mem/vram_budget.h); a launch line that sizes a
     * pool is refused by name, not served with a split other than the one it asked for. */
    Config c;
    CHECK_EQ(parse({ "--vram-weights-mib", "20480" }, &c), RAD_E_INVAL);
    Config k;
    CHECK_EQ(parse({ "--vram-kv-mib", "6144" }, &k), RAD_E_INVAL);
}

TEST(the_sizes_and_counts_reach_the_fields_they_name) {
    Config c;
    CHECK_EQ(parse({ "--tp", "2",
                     "--max-num-batched-tokens", "512",
                     "--max-num-seqs", "8",
                     "--max-model-len", "200000",
                     "--host-pool-mib", "12288",
                     "--prefix-cache-host-mib", "4096",
                     "--prefix-cache-disk-mib", "65536",
                     "--checkpoint-interval", "2048",
                     "--checkpoint-slots", "8",
                     "--tp-wire-min-kb", "64",
                     "--port", "8100" }, &c), RAD_OK);
    CHECK_EQ(c.tp, 2);
    CHECK_EQ(c.max_tok, 512ll);
    CHECK_EQ(c.max_seqs, 8ll);
    CHECK_EQ(c.max_ctx, 200000ll);
    CHECK_EQ(c.host_pool_mib, 12288ll);
    CHECK_EQ(c.prefix_cache_host_mib, 4096ll);
    CHECK_EQ(c.prefix_cache_disk_mib, 65536ll);
    CHECK_EQ(c.checkpoint_interval, 2048ll);
    CHECK_EQ(c.checkpoint_slots, 8ll);
    CHECK_EQ(c.tp_wire_min_kb, 64ll);
    CHECK_EQ(c.port, 8100);

    /* A CONTEXT BOUND THAT DOES NOT FIT IN AN int. --max-model-len is the flag most likely to
     * carry a number past two billion, and atoi would wrap it. */
    Config big;
    CHECK_EQ(parse({ "--max-model-len", "4294967296" }, &big), RAD_OK);
    CHECK_EQ(big.max_ctx, 4294967296ll);
}

TEST(the_paths_and_names_reach_their_fields) {
    Config c;
    CHECK_EQ(parse({ "--radiance-home", "/rh",
                     "--weights-disk-tier",
                     "--prefix-cache-dir", "/kvdir",
                     "--placement", "expert_tiered",
                     "--host", "0.0.0.0",
                     "--generation-config", "/g.json",
                     "--reasoning-effort", "medium",
                     "--served-model-name", "qwen-coder",
                     "--override-chat-template", "/t.jinja" }, &c), RAD_OK);
    CHECK_EQ(c.radiance_home, std::string("/rh"));
    CHECK(c.weights_disk_tier);
    CHECK_EQ(c.prefix_cache_dir, std::string("/kvdir"));
    CHECK_EQ(c.placement, std::string("expert_tiered"));
    CHECK_EQ(c.host, std::string("0.0.0.0"));
    CHECK_EQ(c.generation_config, std::string("/g.json"));
    CHECK_EQ(c.reasoning_effort, std::string("medium"));
    CHECK_EQ(c.served_model_name, std::string("qwen-coder"));
    CHECK_EQ(c.override_chat_template, std::string("/t.jinja"));
}

/* The reasoning budget is a count or nothing: unset is -1, which no request default can mean. */
TEST(thinking_token_budget_takes_a_count_and_defaults_to_none) {
    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.thinking_token_budget, (int64_t)-1);
    Config d;
    CHECK_EQ(parse({ "--thinking-token-budget", "2048" }, &d), RAD_OK);
    CHECK_EQ(d.thinking_token_budget, (int64_t)2048);
    Config e;
    CHECK_EQ(parse({ "--thinking-token-budget", "0" }, &e), RAD_OK);
    CHECK_EQ(e.thinking_token_budget, (int64_t)0);
    Config f;
    CHECK(parse({ "--thinking-token-budget", "-1" }, &f) < 0);
    Config g;
    CHECK(parse({ "--thinking-token-budget", "lots" }, &g) < 0);
}

TEST(the_boolean_flags_are_off_until_they_are_given) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(!d.deterministic);
    CHECK(d.prefix_cache);                 /* on by default; the flag only turns it off */
    CHECK(!d.debug_graph);
    CHECK(!d.debug_selection);
    CHECK(!d.debug_placement);
    CHECK(!d.accept_reference_kernels);
    CHECK(!d.profile_ops);
    CHECK(!d.live_view);

    Config c;
    CHECK_EQ(parse({ "--deterministic", "--no-prefix-cache", "--debug-placement",
                     "--debug-accept-reference-kernels", "--profile-ops", "--live" }, &c), RAD_OK);
    CHECK(c.deterministic);
    CHECK(!c.prefix_cache);
    CHECK(c.debug_placement);
    CHECK(c.accept_reference_kernels);
    CHECK(c.profile_ops);
    CHECK(c.live_view);
}

/* THE GRAPH DUMP IMPLIES THE SELECTION TABLE, because a graph without the kernel that resolved
 * each node is a picture of a program nobody can tell is the one that will run. */
TEST(debug_graph_turns_the_selection_table_on_too) {
    Config c;
    CHECK_EQ(parse({ "--debug-graph" }, &c), RAD_OK);
    CHECK(c.debug_graph);
    CHECK(c.debug_selection);

    Config s;
    CHECK_EQ(parse({ "--debug-selection" }, &s), RAD_OK);
    CHECK(!s.debug_graph);                 /* and not the other way round */
    CHECK(s.debug_selection);
}

TEST(the_verbosity_flags_move_the_log_level) {
    LogGuard guard;
    Config c;
    log_set_level(Log::Info);
    CHECK_EQ(parse({ "-v" }, &c), RAD_OK);
    CHECK((int)log_level() == (int)Log::Debug);
    CHECK_EQ(parse({ "-vv" }, &c), RAD_OK);
    CHECK((int)log_level() == (int)Log::Trace);
}

/* ================================================================== refused */

TEST(tp_below_one_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--tp", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp", "notanumber" }, &c), RAD_E_INVAL);   /* atoi gives 0 */
    CHECK_EQ(parse({ "--tp", "1" }, &c), RAD_OK);
}

TEST(the_wire_mode_is_one_of_two_names) {
    Config c;
    CHECK_EQ(parse({ "--tp-wire", "exact" }, &c), RAD_OK);
    CHECK_EQ(c.tp_wire_exact, 1);
    CHECK_EQ(parse({ "--tp-wire", "wht6" }, &c), RAD_OK);
    CHECK_EQ(c.tp_wire_exact, 0);
    /* A LOSSY WIRE IS A DEPLOYMENT CHOICE AND NOTHING DOWNSTREAM CAN DETECT IT: the sum a rank
     * receives is not the sum it would have received. So a name this build does not know has to
     * stop the server rather than fall through to whichever mode happens to be the default. */
    CHECK_EQ(parse({ "--tp-wire", "wht8" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp-wire", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--tp-wire", "EXACT" }, &c), RAD_E_INVAL);
}

TEST(the_p2p_mode_is_one_of_three_names) {
    Config c;
    CHECK_EQ(c.p2p, RAD_P2P_AUTO);
    CHECK_EQ(parse({ "--p2p", "on" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_ON);
    CHECK_EQ(parse({ "--p2p", "off" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_OFF);
    CHECK_EQ(parse({ "--p2p", "auto" }, &c), RAD_OK);
    CHECK_EQ(c.p2p, RAD_P2P_AUTO);
    /* A name this build does not know stops the server: `on` exists so a deployment measured on
     * P2P cannot quietly serve on host memory, and a typo falling through to auto would do that. */
    CHECK_EQ(parse({ "--p2p", "yes" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--p2p", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--p2p", "OFF" }, &c), RAD_E_INVAL);
}

TEST(the_kv_cache_dtype_is_one_of_two_names) {
    Config c;
    CHECK_EQ(parse({ "--kv-cache-dtype", "bf16" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("bf16"));
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp8" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("fp8"));
    /* Two values and not three: an "auto" resolving to bf16 would be a second name for the
     * default, and a flag whose values are not all distinct cannot be read back out of a log. */
    CHECK_EQ(parse({ "--kv-cache-dtype", "auto" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--kv-cache-dtype", "fp16" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.kv_cache_dtype, std::string("bf16"));
}

/* THE TWO TABLE PLACEMENTS: the embedding takes ram (the default) or vram, the n-gram table auto
 * (the default), vram, ram or disk, and nothing else. A parse records whether the embedding's was
 * named and leaves the resolved values to Engine::plan: where each table may go is the engine's to
 * refuse, with the model in hand -- not the parser's. */
TEST(the_embedding_is_ram_or_vram_and_the_ngram_table_takes_four) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.embedding_placement == TablePlacement::Ram);
    CHECK(!d.embedding_placement_given);
    CHECK(d.ngram_placement == TablePlacement::Auto);
    CHECK_EQ(d.embedding_host_mib, 0ll);                     /* the resolver's */
    Config c;
    CHECK_EQ(parse({ "--embedding-placement", "ram", "--ngram-placement", "disk" }, &c), RAD_OK);
    CHECK(c.embedding_placement == TablePlacement::Ram);
    CHECK(c.embedding_placement_given);
    CHECK(c.ngram_placement == TablePlacement::Disk);
    Config v;
    CHECK_EQ(parse({ "--embedding-placement", "vram", "--ngram-placement", "auto" }, &v), RAD_OK);
    CHECK(v.embedding_placement == TablePlacement::Vram);
    CHECK(v.embedding_placement_given);
    CHECK(v.ngram_placement == TablePlacement::Auto);
    CHECK(v.embedding_resolved == TablePlacement::Vram);     /* untouched until plan */
    CHECK_EQ(parse({ "--ngram-placement", "ram" }, &v), RAD_OK);
    CHECK(v.ngram_placement == TablePlacement::Ram);
    for (const char* bad : { "auto", "disk" })
        CHECK_EQ(parse({ "--embedding-placement", bad }, &c), RAD_E_INVAL);
    for (const char* bad : { "RAM", "memory", "host", "" }) {
        CHECK_EQ(parse({ "--embedding-placement", bad }, &c), RAD_E_INVAL);
        CHECK_EQ(parse({ "--ngram-placement", bad }, &c), RAD_E_INVAL);
    }
    CHECK_EQ(parse({ "--ngram-placement" }, &c), RAD_E_INVAL);
}

/* --draft-window takes its two names and nothing else; the default is kDraftWindowDefault, and the
 * help text states it from there. Whether the window CAN adapt -- a serial drafter, a depth above
 * zero -- is the scheduler's to say with the program in hand, not the parser's. */
TEST(the_draft_window_is_fixed_or_adaptive) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.draft_window == kDraftWindowDefault);
    Config c;
    CHECK_EQ(parse({ "--draft-window", "adaptive" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Adaptive);
    CHECK_EQ(parse({ "--draft-window", "fixed" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Fixed);
    CHECK_EQ(parse({ "--draft-window", "adaptive", "--num-speculative-tokens", "6" }, &c), RAD_OK);
    CHECK(c.draft_window == DraftWindow::Adaptive);
    CHECK_EQ(c.n_spec, 6);
    for (const char* bad : { "Adaptive", "auto", "dynamic", "1", "" })
        CHECK_EQ(parse({ "--draft-window", bad }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-window" }, &c), RAD_E_INVAL);
    CHECK_EQ(std::string(draft_window_name(DraftWindow::Adaptive)), std::string("adaptive"));
    CHECK_EQ(std::string(draft_window_name(DraftWindow::Fixed)), std::string("fixed"));
}

/* A POLICY NAME THIS BUILD DOES NOT KNOW WOULD FALL THROUGH TO THE DEFAULT, and a measurement
 * comparing the default against itself is the failure that costs the most to notice. */
TEST(the_checkpoint_policy_is_one_of_the_three_that_exist) {
    Config c;
    for (const char* p : { "geometric-backoff", "tip-only", "keep-all" }) {
        CHECK_EQ(parse({ "--checkpoint-policy", p }, &c), RAD_OK);
        CHECK_EQ(c.checkpoint_policy, std::string(p));
    }
    CHECK_EQ(parse({ "--checkpoint-policy", "lru" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--checkpoint-policy", "geometric_backoff" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.checkpoint_policy, std::string("geometric-backoff"));
}

/* THE SAME RANGES THE REQUEST PARSER ENFORCES, refused here rather than at the first request: a
 * default outside them is a 400 on every request that omits the field, which is a server that
 * starts, reports healthy and answers nothing. */
TEST(the_sampler_defaults_are_range_checked_at_startup) {
    Config c;
    CHECK_EQ(parse({ "--temp", "0.7", "--top-k", "20", "--top-p", "0.95",
                     "--min-p", "0.05" }, &c), RAD_OK);
    CHECK_NEAR(c.sample_temp, 0.7f, 1e-6);
    CHECK_EQ(c.sample_top_k, 20);
    CHECK_NEAR(c.sample_top_p, 0.95f, 1e-6);
    CHECK_NEAR(c.sample_min_p, 0.05f, 1e-6);

    /* The legal endpoints. Temperature 0 is greedy and top_k 0 is off: both are real settings. */
    CHECK_EQ(parse({ "--temp", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--temp", "2" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--top-k", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--top-p", "1" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--min-p", "0" }, &c), RAD_OK);
    CHECK_EQ(parse({ "--min-p", "1" }, &c), RAD_OK);

    CHECK_EQ(parse({ "--temp", "2.1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "0" }, &c), RAD_E_INVAL);   /* a nucleus of nothing */
    CHECK_EQ(parse({ "--top-p", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--min-p", "1.5" }, &c), RAD_E_INVAL);

    /* A negative value is refused rather than read as "unset", which is what a negative field
     * means below the parser -- the container's default would be served in its place. */
    CHECK_EQ(parse({ "--temp", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-k", "-5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--min-p", "-0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "nan" }, &c), RAD_E_INVAL);

    /* UNSET IS NEGATIVE AND NOT ZERO, which is what lets the container's own generation config be
     * resolved underneath: zero is a real temperature and a real top_k, so a default of zero
     * would be indistinguishable from an operator asking for greedy. */
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.sample_temp < 0.0f);
    CHECK(d.sample_top_k < 0);
    CHECK(d.sample_top_p < 0.0f);
    CHECK(d.sample_min_p < 0.0f);
}

/* A NUMBER IS THE WHOLE ARGUMENT. atof read "high" as 0, which for --temp is greedy decoding. */
TEST(a_sampler_default_that_is_not_a_number_is_refused) {
    Config c;
    CHECK_EQ(parse({ "--temp", "high" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "0.7x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--temp", "" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-k", "20.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--top-p", "inf" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--presence-penalty", "nan" }, &c), RAD_E_INVAL);
}

/* ================================================================== request defaults and bounds
 *
 * NONE OF THEM MOVES ANYTHING UNTIL IT IS GIVEN. Each field's unset value is the one the server
 * used before the flag existed (or a marker the engine reads as "leave it"), so a command line
 * that names none of them serves exactly what it served before. */
TEST(the_request_defaults_and_bounds_are_unset_until_given) {
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(!d.sample_typical_p && !d.sample_rep_penalty && !d.sample_pres_penalty &&
          !d.sample_freq_penalty && !d.sample_dry_multiplier && !d.sample_dry_base &&
          !d.sample_xtc_probability && !d.sample_xtc_threshold);
    CHECK(!d.sample_penalty_last_n && !d.sample_dry_allowed_length &&
          !d.sample_dry_penalty_last_n && !d.sample_dry_seq_breakers);
    CHECK_EQ(d.default_max_tokens, -1ll);
    CHECK_EQ(d.max_tokens_cap, 0ll);
    CHECK_EQ(d.max_n, 0ll);
    CHECK_EQ(d.max_queued_requests, 0ll);
    CHECK_EQ(d.max_stop_strings, 64ll);
    CHECK_EQ(d.max_stop_bytes, 4096ll);
    CHECK(d.chat_template_kwargs.empty());
    CHECK_EQ(d.reasoning_format, std::string("auto"));
    /* The transport's own values (server/http.h TransportOptions). */
    CHECK_EQ(d.http_threads, 0);
    CHECK_EQ(d.http_read_timeout_s, 30);
    CHECK_EQ(d.http_write_timeout_s, 600);
    CHECK_EQ(d.http_keep_alive_timeout_s, 5);
    CHECK_EQ(d.http_max_body_mib, 512ll);
    CHECK_EQ(d.retry_after_s, 1);
    CHECK(d.cors);
    CHECK(!d.media_flags_set());
}

TEST(the_request_defaults_reach_their_fields) {
    Config c;
    CHECK_EQ(parse({ "--typical-p", "0.9", "--presence-penalty", "1.5",
                     "--frequency-penalty", "-0.5", "--repetition-penalty", "1.05",
                     "--repeat-last-n", "-1", "--dry-multiplier", "0.8", "--dry-base", "1.75",
                     "--dry-allowed-length", "3", "--dry-penalty-last-n", "256",
                     "--dry-sequence-breakers", R"(["\n", ":"])",
                     "--xtc-probability", "0.5", "--xtc-threshold", "0.1",
                     "--default-max-tokens", "4096", "--max-tokens-cap", "8192",
                     "--max-n", "4", "--max-queued-requests", "64",
                     "--max-stop-strings", "8", "--max-stop-bytes", "256",
                     "--reasoning-format", "none" }, &c), RAD_OK);
    CHECK_NEAR(*c.sample_typical_p, 0.9f, 1e-6);
    CHECK_NEAR(*c.sample_pres_penalty, 1.5f, 1e-6);
    CHECK_NEAR(*c.sample_freq_penalty, -0.5f, 1e-6);
    CHECK_NEAR(*c.sample_rep_penalty, 1.05f, 1e-6);
    CHECK_EQ(*c.sample_penalty_last_n, -1);
    CHECK_NEAR(*c.sample_dry_multiplier, 0.8f, 1e-6);
    CHECK_NEAR(*c.sample_dry_base, 1.75f, 1e-6);
    CHECK_EQ(*c.sample_dry_allowed_length, 3);
    CHECK_EQ(*c.sample_dry_penalty_last_n, 256);
    CHECK(*c.sample_dry_seq_breakers == std::vector<std::string>({ "\n", ":" }));
    CHECK_NEAR(*c.sample_xtc_probability, 0.5f, 1e-6);
    CHECK_NEAR(*c.sample_xtc_threshold, 0.1f, 1e-6);
    CHECK_EQ(c.default_max_tokens, 4096ll);
    CHECK_EQ(c.max_tokens_cap, 8192ll);
    CHECK_EQ(c.max_n, 4ll);
    CHECK_EQ(c.max_queued_requests, 64ll);
    CHECK_EQ(c.max_stop_strings, 8ll);
    CHECK_EQ(c.max_stop_bytes, 256ll);
    CHECK_EQ(c.reasoning_format, std::string("none"));

    /* `auto` is 0, which the server resolves per request to what the context leaves. */
    Config a;
    CHECK_EQ(parse({ "--default-max-tokens", "auto" }, &a), RAD_OK);
    CHECK_EQ(a.default_max_tokens, 0ll);

    /* An empty breaker list is a stated value -- DRY with no breakers -- and stays one. */
    Config e;
    CHECK_EQ(parse({ "--dry-sequence-breakers", "[]" }, &e), RAD_OK);
    CHECK(e.sample_dry_seq_breakers && e.sample_dry_seq_breakers->empty());

    /* A negative penalty is a value a request can send, so the flag can state it. */
    Config n;
    CHECK_EQ(parse({ "--presence-penalty", "-2" }, &n), RAD_OK);
    CHECK_NEAR(*n.sample_pres_penalty, -2.0f, 1e-6);
}

/* THE RANGES A REQUEST IS HELD TO, at startup: a default outside them would be a 400 on every
 * request that leaves the field to it. */
TEST(the_request_defaults_are_range_checked_like_a_request) {
    Config c;
    CHECK_EQ(parse({ "--typical-p", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--typical-p", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--presence-penalty", "2.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--frequency-penalty", "-3" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--repetition-penalty", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--repeat-last-n", "-2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-base", "0.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-penalty-last-n", "-2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--xtc-probability", "1.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--xtc-threshold", "-0.1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-allowed-length", "3.5" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-sequence-breakers", R"("\n")" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--dry-sequence-breakers", "[1]" }, &c), RAD_E_INVAL);
    /* The sampler takes at most 16 (sample/host_ref.h); every DRY request would be refused. */
    CHECK_EQ(parse({ "--dry-sequence-breakers",
                     R"(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"])" },
                   &c), RAD_E_INVAL);

    CHECK_EQ(parse({ "--default-max-tokens", "0" }, &c), RAD_E_INVAL);   /* auto is the word */
    CHECK_EQ(parse({ "--default-max-tokens", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--default-max-tokens", "AUTO" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--default-max-tokens", "4294967296" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-tokens-cap", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-n", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-queued-requests", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-stop-strings", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-stop-bytes", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--reasoning-format", "deepseek" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--reasoning-format", "" }, &c), RAD_E_INVAL);
}

/* BOUNDS THAT CONTRADICT EACH OTHER. A default above the cap is a default no request is served
 * at; an `n` above the queue is a request that is 429 however long its caller waits. */
TEST(request_bounds_that_contradict_each_other_are_refused) {
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "4096", "--max-tokens-cap", "1024" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "1024", "--max-tokens-cap", "1024" }), RAD_OK);
    CHECK_EQ(parse_fresh({ "--default-max-tokens", "auto", "--max-tokens-cap", "1024" }), RAD_OK);

    CHECK_EQ(parse_fresh({ "--max-n", "65", "--max-queued-requests", "64" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--max-n", "64", "--max-queued-requests", "64" }), RAD_OK);
    /* The queue's default is 8 x --max-num-seqs, and the check reads it. */
    CHECK_EQ(parse_fresh({ "--max-num-seqs", "2", "--max-n", "17" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--max-num-seqs", "2", "--max-n", "16" }), RAD_OK);
    /* And a stated queue smaller than the default n (8, or --max-num-seqs). */
    CHECK_EQ(parse_fresh({ "--max-queued-requests", "4" }), RAD_E_INVAL);
}

TEST(the_template_variables_merge_key_by_key_and_later_wins) {
    Config c;
    CHECK_EQ(parse({ "--chat-template-kwargs", R"({"enable_thinking": false, "custom": "a"})",
                     "--chat-template-kwargs", R"({"custom": "b", "n": 3})" }, &c), RAD_OK);
    CHECK_EQ(c.chat_template_kwargs.size(), 3u);
    CHECK_EQ(c.chat_template_kwargs["enable_thinking"], std::string("false"));
    CHECK_EQ(c.chat_template_kwargs["custom"], std::string("\"b\""));   /* JSON text, quoted */
    CHECK_EQ(c.chat_template_kwargs["n"], std::string("3"));

    /* @FILE reads the object from a file. */
    char path[] = "/tmp/rad_kwargs_XXXXXX";
    const int fd = mkstemp(path);
    REQUIRE(fd >= 0);
    const std::string body = R"({"date_string": "01 Jan 2026"})";
    CHECK_EQ((size_t)write(fd, body.data(), body.size()), body.size());
    close(fd);
    Config f;
    const std::string at = std::string("@") + path;
    CHECK_EQ(parse({ "--chat-template-kwargs", at.c_str() }, &f), RAD_OK);
    CHECK_EQ(f.chat_template_kwargs["date_string"], std::string("\"01 Jan 2026\""));
    unlink(path);
    CHECK_EQ(parse({ "--chat-template-kwargs", at.c_str() }, &f), RAD_E_INVAL);   /* gone */
}

TEST(a_template_variable_the_server_owns_or_cannot_read_is_refused) {
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", "[1]" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", "{nope" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"("enable_thinking")" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"({"enable_thinking": "no"})" }), RAD_E_INVAL);
    /* The renderer writes these over the context; one here would replace every conversation. */
    for (const char* k : { "messages", "tools", "bos_token", "eos_token", "add_generation_prompt" }) {
        const std::string j = std::string("{\"") + k + "\": 1}";
        CHECK_EQ(parse_fresh({ "--chat-template-kwargs", j.c_str() }), RAD_E_INVAL);
    }
    /* Two defaults for thinking, in two flags, with no order between them. */
    CHECK_EQ(parse_fresh({ "--reasoning-effort", "low",
                           "--chat-template-kwargs", R"({"reasoning_effort": "high"})" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--chat-template-kwargs", R"({"enable_thinking": true})",
                           "--reasoning-effort", "none" }), RAD_E_INVAL);
    CHECK_EQ(parse_fresh({ "--reasoning-effort", "low",
                           "--chat-template-kwargs", R"({"custom": 1})" }), RAD_OK);
}

TEST(the_transport_flags_reach_their_fields_and_hold_their_floors) {
    Config c;
    CHECK_EQ(parse({ "--http-threads", "16", "--read-timeout", "10", "--write-timeout", "120",
                     "--keep-alive-timeout", "30", "--max-body-mib", "64", "--retry-after", "5",
                     "--no-cors" }, &c), RAD_OK);
    CHECK_EQ(c.http_threads, 16);
    CHECK_EQ(c.http_read_timeout_s, 10);
    CHECK_EQ(c.http_write_timeout_s, 120);
    CHECK_EQ(c.http_keep_alive_timeout_s, 30);
    CHECK_EQ(c.http_max_body_mib, 64ll);
    CHECK_EQ(c.retry_after_s, 5);
    CHECK(!c.cors);

    /* The transport keeps four workers whatever it is told, so fewer is refused, not ignored. */
    CHECK_EQ(parse({ "--http-threads", "2" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--read-timeout", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--write-timeout", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--keep-alive-timeout", "x" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-body-mib", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--retry-after", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--retry-after", "0" }, &c), RAD_OK);
}

TEST(the_media_flags_reach_their_fields_and_refuse_an_empty_band) {
    Config c;
    CHECK_EQ(parse({ "--image-min-pixels", "3136", "--image-max-pixels", "1003520",
                     "--video-min-pixels", "4096", "--video-max-pixels", "8388608",
                     "--video-fps", "1.5", "--video-min-frames", "2", "--video-max-frames", "64",
                     "--video-max-frame-tokens", "0", "--max-source-pixels", "16777216" }, &c),
             RAD_OK);
    CHECK(c.media_flags_set());
    CHECK_EQ(*c.image_min_pixels, 3136ll);
    CHECK_EQ(*c.image_max_pixels, 1003520ll);
    CHECK_EQ(*c.video_min_pixels, 4096ll);
    CHECK_EQ(*c.video_max_pixels, 8388608ll);
    CHECK_NEAR(*c.video_fps, 1.5, 1e-9);
    CHECK_EQ(*c.video_min_frames, 2);
    CHECK_EQ(*c.video_max_frames, 64);
    CHECK_EQ(*c.video_max_frame_tokens, 0);      /* 0 is "no per-frame cap" */
    CHECK_EQ(*c.max_source_pixels, 16777216ll);

    CHECK_EQ(parse({ "--image-min-pixels", "2000", "--image-max-pixels", "1000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-pixels", "2000", "--video-max-pixels", "1000" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-frames", "8", "--video-max-frames", "4" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--image-max-pixels", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-fps", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-fps", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-min-frames", "0" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--video-max-frame-tokens", "-1" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--max-source-pixels", "0" }, &c), RAD_E_INVAL);
}

/* REFUSED, NOT QUIETLY IGNORED. Accepting it and then serving the container's own drafter -- or
 * none -- hands the caller a server that is not the one they asked for, with nothing in the
 * output to say so. A caller coming from vLLM will reach for this flag. */
TEST(draft_model_is_refused_by_name_rather_than_ignored) {
    Config c;
    CHECK_EQ(parse({ "--draft-model", "/d.rad" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--draft-model" }, &c), RAD_E_INVAL);
}

/* ================================================================== resolved */

TEST(auto_speculation_is_negative_one_and_not_zero) {
    /* The depth is a property of the DRAFTER, which the architecture plugin states once the
     * container is open -- so "auto" has to survive as a distinct value here. Zero is off. */
    Config c;
    CHECK_EQ(parse({ "--num-speculative-tokens", "auto" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, -1);
    CHECK_EQ(parse({ "--num-speculative-tokens", "0" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, 0);
    CHECK_EQ(parse({ "--num-speculative-tokens", "3" }, &c), RAD_OK);
    CHECK_EQ(c.n_spec, 3);
    /* Only the word says auto: a negative depth would read as it downstream. */
    CHECK_EQ(parse({ "--num-speculative-tokens", "-3" }, &c), RAD_E_INVAL);
    CHECK_EQ(parse({ "--num-speculative-tokens", "-1" }, &c), RAD_E_INVAL);

    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.n_spec, -1);                /* auto is also the default */
}

/* IT TAKES THE SHORT NAME TOO: "-mib" is the unit, and this is the flag an operator reaches for.
 * Both spellings parse to the same field, so a line written either way means one thing. */
TEST(gpu_headroom_parses_under_both_spellings) {
    Config a, b, d;
    CHECK_EQ(parse({ "--gpu-headroom-mib", "256" }, &a), RAD_OK);
    CHECK_EQ(parse({ "--gpu-headroom", "256" }, &b), RAD_OK);
    CHECK_EQ(a.gpu_headroom_mib, 256ll);
    CHECK_EQ(b.gpu_headroom_mib, 256ll);
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK_EQ(d.gpu_headroom_mib, 96ll);
}

TEST(the_kernel_hierarchy_is_comma_separated_and_ordered) {
    Config c;
    CHECK_EQ(parse({ "--kernels", "libr4d.so,libref.so" }, &c), RAD_OK);
    CHECK_EQ(c.kernel_hierarchy.size(), 2u);
    CHECK_EQ(c.kernel_hierarchy[0], std::string("libr4d.so"));   /* first wins, so order is data */
    CHECK_EQ(c.kernel_hierarchy[1], std::string("libref.so"));

    /* Empty segments are dropped rather than kept as empty names: a trailing comma is a typing
     * artefact, and a plugin path of "" matches nothing and would warn on every start. */
    Config t;
    CHECK_EQ(parse({ "--kernels", ",a,,b," }, &t), RAD_OK);
    CHECK_EQ(t.kernel_hierarchy.size(), 2u);
    CHECK_EQ(t.kernel_hierarchy[0], std::string("a"));
    CHECK_EQ(t.kernel_hierarchy[1], std::string("b"));

    Config one;
    CHECK_EQ(parse({ "--kernels", "solo" }, &one), RAD_OK);
    CHECK_EQ(one.kernel_hierarchy.size(), 1u);
}

/* AND THERE IS NO BUILT-IN DEFAULT HIERARCHY, deliberately: naming one card's kernel library in
 * the core would, on every other machine, rank the library that IS installed behind the reference
 * implementation. An empty hierarchy is not "no order" -- the loader supplies one. */
TEST(an_unnamed_hierarchy_stays_empty_for_the_loader_to_order) {
    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.kernel_hierarchy.size(), 0u);
}

/* THE ENGINE HONOURS THE SAME VARIABLE THE TOOLS DO. Otherwise an operator who validated a
 * third-party plugin with rad-kbench or rad-tune has no way to serve with it short of editing
 * every launcher's command line. */
TEST(the_environment_supplies_what_the_line_left_unsaid) {
    const char* old_home = getenv("RADIANCE_HOME");
    const char* old_kern = getenv("RADIANCE_KERNELS");
    const std::string save_home = old_home ? old_home : "";
    const std::string save_kern = old_kern ? old_kern : "";

    setenv("RADIANCE_HOME", "/env/home", 1);
    setenv("RADIANCE_KERNELS", "env_a,env_b", 1);

    Config c;
    CHECK_EQ(parse({}, &c), RAD_OK);
    CHECK_EQ(c.radiance_home, std::string("/env/home"));
    CHECK_EQ(c.kernel_hierarchy.size(), 2u);
    CHECK_EQ(c.kernel_hierarchy[0], std::string("env_a"));

    /* The command line wins over the environment, in both cases. */
    Config f;
    CHECK_EQ(parse({ "--radiance-home", "/flag/home", "--kernels", "flag_only" }, &f), RAD_OK);
    CHECK_EQ(f.radiance_home, std::string("/flag/home"));
    CHECK_EQ(f.kernel_hierarchy.size(), 1u);
    CHECK_EQ(f.kernel_hierarchy[0], std::string("flag_only"));

    /* An empty variable says nothing rather than saying "the empty hierarchy". */
    setenv("RADIANCE_KERNELS", "", 1);
    Config e;
    CHECK_EQ(parse({}, &e), RAD_OK);
    CHECK_EQ(e.kernel_hierarchy.size(), 0u);

    unsetenv("RADIANCE_HOME");
    Config h;
    CHECK_EQ(parse({}, &h), RAD_OK);
    CHECK(!h.radiance_home.empty());               /* the compiled-in default, whatever it is */

    if (old_home) setenv("RADIANCE_HOME", save_home.c_str(), 1); else unsetenv("RADIANCE_HOME");
    if (old_kern) setenv("RADIANCE_KERNELS", save_kern.c_str(), 1); else unsetenv("RADIANCE_KERNELS");
}

/* LATER WINS, for every flag that takes a value. Scripts build these lines by appending, and a
 * parser that kept the first occurrence would make an override a no-op. */
TEST(the_download_dir_is_taken_and_left_empty_until_given) {
    /* Empty means "the XDG cache", which hf_model.cpp resolves only when --model is a repo id:
     * a server started on a local file never reads HOME for it. */
    Config c;
    CHECK_EQ(parse({}, &c), 0);
    CHECK(c.download_dir.empty());
    Config d;
    CHECK_EQ(parse({ "--download-dir", "/big/models" }, &d), 0);
    CHECK_EQ(d.download_dir, std::string("/big/models"));
}

TEST(a_repeated_flag_takes_the_last_one) {
    Config c;
    CHECK_EQ(parse({ "--port", "1", "--port", "8100" }, &c), RAD_OK);
    CHECK_EQ(c.port, 8100);
    CHECK_EQ(parse({ "--kv-cache-dtype", "bf16", "--kv-cache-dtype", "fp8" }, &c), RAD_OK);
    CHECK_EQ(c.kv_cache_dtype, std::string("fp8"));
    /* And a bad value refuses even when a good one preceded it. */
    CHECK_EQ(parse({ "--tp-wire", "exact", "--tp-wire", "nope" }, &c), RAD_E_INVAL);
}

TEST(usage_names_every_flag_it_documents) {
    /* config_usage writes to stderr; the property worth holding is that it mentions the flags an
     * operator is most likely to be looking for, so a flag added without a help line is caught. */
    FILE* f = tmpfile();
    CHECK(f != nullptr);
    if (!f) return;
    const int saved = dup(2);
    CHECK(saved >= 0);
    fflush(stderr);
    dup2(fileno(f), 2);
    config_usage("radiance");
    fflush(stderr);
    dup2(saved, 2);
    close(saved);

    std::string text;
    rewind(f);
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    fclose(f);

    CHECK(text.find("--model") != std::string::npos);
    CHECK(text.find("--host-pool-mib") != std::string::npos);
    CHECK(text.find("--kv-cache-dtype") != std::string::npos);
    CHECK(text.find("--num-speculative-tokens") != std::string::npos);
    CHECK(text.find("--draft-window") != std::string::npos);
    CHECK(text.find(std::string("Default ") + draft_window_name(kDraftWindowDefault)) !=
          std::string::npos);
    CHECK(text.find("--checkpoint-policy") != std::string::npos);
    CHECK(text.find("--prefix-cache-host-mib") != std::string::npos);
    CHECK(text.find("--tp-wire") != std::string::npos);
    CHECK(text.find("--embedding-placement") != std::string::npos);
    CHECK(text.find("--ngram-placement") != std::string::npos);
    CHECK(text.find("--kld-record") != std::string::npos);
    CHECK(text.find("--kld-ref") != std::string::npos);
    for (const char* f : { "--default-max-tokens", "--max-tokens-cap", "--presence-penalty",
                           "--dry-sequence-breakers", "--chat-template-kwargs",
                           "--reasoning-format", "--max-n", "--max-queued-requests",
                           "--max-stop-strings", "--http-threads", "--max-body-mib", "--no-cors",
                           "--image-max-pixels", "--video-fps", "--max-source-pixels",
                           "--download-dir" })
        CHECK(text.find(f) != std::string::npos);
    CHECK(text.size() > 1000);
}

/* --rope-scaling is checked at the command line: a typo is a refusal, not a model served with its
 * trained rotary at a length it cannot reach. */
TEST(rope_scaling_takes_yarn_with_a_factor_above_one_or_none) {
    for (const char* ok : { "yarn:4", "yarn:2.5", "yarn:4:262144", "none" }) {
        Config c;
        CHECK_EQ(parse({ "--rope-scaling", ok }, &c), RAD_OK);
        CHECK_EQ(c.rope_scaling, std::string(ok));
    }
    for (const char* bad : { "yarn:1", "yarn:0.5", "yarn:4x", "yarn:", "yarn:4:", "yarn:4:0",
                             "yarn:4:12k", "linear:2", "yarn" }) {
        Config c;
        CHECK(parse({ "--rope-scaling", bad }, &c) < 0);
    }
    Config d;
    CHECK_EQ(parse({}, &d), RAD_OK);
    CHECK(d.rope_scaling.empty());
}

RAD_TEST_MAIN()
