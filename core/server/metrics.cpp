#include "metrics.h"
#include "iface.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace rad {
namespace server {

/* ------------------------------------------------------------------ primitives */

Histogram::Histogram(std::vector<double> bounds) : bounds_(std::move(bounds)) {
    cnt_.reset(new std::atomic<uint64_t>[bounds_.size()]);
    for (size_t i = 0; i < bounds_.size(); ++i) cnt_[i].store(0);
}

void Histogram::observe(double v) {
    if (!std::isfinite(v)) return;    /* a NaN latency is a bug upstream, not a data point */
    /* Cumulative: every bucket whose bound is >= v counts this observation. Linear because the
     * bucket lists are twenty-odd entries and a binary search plus a fixup loop is not faster. */
    for (size_t i = 0; i < bounds_.size(); ++i)
        if (v <= bounds_[i]) cnt_[i].fetch_add(1, std::memory_order_relaxed);
    n_.fetch_add(1, std::memory_order_relaxed);
    double cur = sum_.load(std::memory_order_relaxed);
    while (!sum_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {}
}

std::vector<double> buckets_ttft() {
    return {0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75,
            1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0};
}

std::vector<double> buckets_tpot() {
    return {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
            0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0};
}

std::vector<double> buckets_latency() {
    return {0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0,
            30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0};
}

/* vLLM's build_1_2_5_buckets: mantissas 1, 2, 5 at every decade, stopping at the first value past
 * max_value. A 32K-context model therefore gets 1..20000 and no more. */
std::vector<double> buckets_1_2_5(int64_t max_value) {
    std::vector<double> out;
    if (max_value < 1) max_value = 1;
    const int mant[3] = {1, 2, 5};
    for (int exp = 0; exp < 12; ++exp) {
        for (int m = 0; m < 3; ++m) {
            double v = (double)mant[m] * std::pow(10.0, exp);
            if (v > (double)max_value) return out;
            out.push_back(v);
        }
    }
    return out;
}

/* ------------------------------------------------------------------ registry */

static std::string esc_label(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '\\' || c == '"') { o.push_back('\\'); o.push_back(c); }
        else if (c == '\n') o += "\\n";
        else o.push_back(c);
    }
    return o;
}

Metrics::Metrics(std::string model_name, int64_t max_model_len)
    : model_(std::move(model_name)),
      h_ttft_(buckets_ttft()),
      h_tpot_(buckets_tpot()),
      h_e2e_(buckets_latency()),
      h_decode_(buckets_latency()),
      h_prompt_tokens_(buckets_1_2_5(max_model_len > 0 ? max_model_len : 32768)),
      h_generation_tokens_(buckets_1_2_5(max_model_len > 0 ? max_model_len : 32768)),
      h_params_n_({1, 2, 5, 10, 20}),
      h_params_max_tokens_(buckets_1_2_5(max_model_len > 0 ? max_model_len : 32768)) {
    labels_ = "{model_name=\"" + esc_label(model_) + "\",engine=\"0\"}";
}

static const char* reason_str(SuccessReason r) {
    switch (r) {
        case SuccessReason::Stop:      return "stop";
        case SuccessReason::Length:    return "length";
        case SuccessReason::ToolCalls: return "tool_calls";
        case SuccessReason::Abort:     return "abort";
        case SuccessReason::Error:     return "error";
        default:                       return "stop";
    }
}

void Metrics::observe_request(int64_t n_prompt, int64_t n_generated,
                              double ttft_s, double tpot_s, double e2e_s, double decode_s,
                              int n_choices, int64_t max_tokens, SuccessReason r) {
    c_prompt_tokens_.inc((double)n_prompt);
    c_generation_tokens_.inc((double)n_generated);
    c_success_[(size_t)r].inc();

    h_prompt_tokens_.observe((double)n_prompt);
    h_generation_tokens_.observe((double)n_generated);
    h_params_n_.observe((double)n_choices);
    if (max_tokens > 0) h_params_max_tokens_.observe((double)max_tokens);

    if (ttft_s >= 0.0) h_ttft_.observe(ttft_s);
    /* TPOT is undefined for a one-token generation -- there is no inter-token interval to
     * measure. Recording a zero there would drag every quantile down, so it is simply not an
     * observation. */
    if (tpot_s >= 0.0 && n_generated > 1) h_tpot_.observe(tpot_s);
    if (e2e_s >= 0.0) h_e2e_.observe(e2e_s);
    if (decode_s >= 0.0 && n_generated > 1) h_decode_.observe(decode_s);
}

void Metrics::observe_rejected()   { c_rejected_.inc(); }
void Metrics::observe_disconnect() { c_disconnect_.inc(); }
void Metrics::observe_overflow()   { c_overflow_.inc(); }

void Metrics::observe_http(int status) {
    if (status >= 500) c_http_5xx_.inc();
    else if (status >= 400) c_http_4xx_.inc();
    else c_http_2xx_.inc();
}

void Metrics::sample(const SchedMetrics& m) {
    std::lock_guard<std::mutex> lk(sample_mu_);
    g_running_.set((double)m.running);
    g_waiting_.set((double)m.waiting);
    g_preempted_.set((double)m.preempted);
    g_kv_util_.set(m.kv_util);
    g_kv_max_tokens_.set((double)m.kv_tokens_max);
    g_kv_max_bytes_.set((double)m.kv_bytes_carved);
    g_prefix_hit_.set(m.prefix_hit_rate);
    g_prefill_tps_.set(m.prefill_tps);
    g_decode_tps_.set(m.decode_tps);
    g_draft_accept_.set(m.draft_acceptance);
    const int seen = std::max(0, std::min(m.draft_depth_seen, (int)MAX_DRAFT_POS));
    for (int j = 0; j < seen; ++j) {
        const int64_t r = m.draft_pos_reached[j];
        g_draft_pos_reached_[j].set((double)r);
        g_draft_pos_[j].set(r ? (double)m.draft_pos_accepted[j] / (double)r : 0.0);
    }
    /* Published after the gauges it bounds, so a concurrent render never lists a position whose
     * gauge has not been written yet. */
    draft_pos_seen_.store(seen, std::memory_order_release);
    if (m.draft_adaptive) {
        g_draft_depth_adapt_.set(m.draft_depth_adaptive);
        int top = 0;
        for (int d = 1; d <= (int)MAX_DRAFT_POS; ++d) {
            g_draft_vdepth_[d].set((double)m.draft_verify_depth[d]);
            if (m.draft_verify_depth[d] > 0) top = d;
        }
        /* After the gauges, for the reason above. */
        draft_vdepth_top_.store(top, std::memory_order_release);
        draft_adapt_on_.store(true, std::memory_order_release);
    }
    c_preemptions_.inc(std::max(0.0, (double)m.preemptions - c_preemptions_.get()));
    c_steps_.inc(std::max(0.0, (double)m.steps - c_steps_.get()));
    c_steps_prefill_.inc(std::max(0.0, (double)m.steps_prefill - c_steps_prefill_.get()));
    c_steps_mixed_.inc(std::max(0.0, (double)m.steps_mixed - c_steps_mixed_.get()));
    c_steps_decode_.inc(std::max(0.0, (double)m.steps_decode - c_steps_decode_.get()));
    c_steps_ahead_.inc(std::max(0.0, (double)m.steps_ahead - c_steps_ahead_.get()));
    c_rows_unspec_.inc(std::max(0.0,
        (double)m.decode_rows_unspeculated - c_rows_unspec_.get()));
    c_step_secs_.inc(std::max(0.0, (double)m.step_ns * 1e-9 - c_step_secs_.get()));
    c_step_secs_decode_.inc(std::max(0.0,
        (double)m.step_ns_decode * 1e-9 - c_step_secs_decode_.get()));
    c_prefill_tokens_.inc(std::max(0.0, (double)m.prefill_tokens - c_prefill_tokens_.get()));
    c_decode_tokens_.inc(std::max(0.0, (double)m.decode_tokens - c_decode_tokens_.get()));
    c_draft_tokens_.inc(std::max(0.0, (double)m.draft_tokens - c_draft_tokens_.get()));
    c_draft_accepted_.inc(std::max(0.0, (double)m.draft_accepted - c_draft_accepted_.get()));
    c_draft_refills_.inc(std::max(0.0, (double)m.draft_refills - c_draft_refills_.get()));
    c_draft_refill_tokens_.inc(
        std::max(0.0, (double)m.draft_refill_tokens - c_draft_refill_tokens_.get()));
    c_attn_cached_.inc(std::max(0.0, (double)m.attn_cached_tokens - c_attn_cached_.get()));
    c_linear_cached_.inc(std::max(0.0, (double)m.linear_cached_tokens - c_linear_cached_.get()));
    c_checkpoints_.inc(std::max(0.0, (double)m.checkpoints_written - c_checkpoints_.get()));
    /* The cache's own cumulative counts, so the delta is taken the same way the others are.
     * A prefix cache that is evicting under steady load is one that cannot hold the working
     * set, and the next turn of every session it drops re-prefills from scratch. */
    c_prefix_evict_.inc(std::max(0.0, (double)m.prefix_evictions - c_prefix_evict_.get()));
    c_prefix_ckpt_evict_.inc(std::max(0.0,
        (double)m.prefix_ckpt_evictions - c_prefix_ckpt_evict_.get()));
    g_linear_hit_.set(m.linear_hit_rate);
}

/* ------------------------------------------------------------------ exposition */

namespace {

/* Prometheus wants a bare decimal; %g would emit "1e+06" for a bucket bound, which parses, and
 * "inf" for +Inf, which does not. 17 significant digits round-trips a double exactly. */
std::string num(double v) {
    if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
    if (v == (double)(long long)v && std::fabs(v) < 1e15) {
        char b[32];
        snprintf(b, sizeof b, "%lld", (long long)v);
        return b;
    }
    char b[64];
    snprintf(b, sizeof b, "%.17g", v);
    return b;
}

struct Writer {
    std::string out;

    void help(const char* name, const char* doc, const char* type) {
        out += "# HELP "; out += name; out += ' '; out += doc; out += '\n';
        out += "# TYPE "; out += name; out += ' '; out += type; out += '\n';
    }
    void gauge(const char* name, const char* doc, const std::string& labels, double v) {
        help(name, doc, "gauge");
        out += name; out += labels; out += ' '; out += num(v); out += '\n';
    }
    void counter(const char* name, const char* doc, const std::string& labels, double v) {
        /* The _total suffix is part of the exposed family name for a counter, which is what the
         * Python client does and therefore what the dashboards match on. */
        help(name, doc, "counter");
        out += name; out += labels; out += ' '; out += num(v); out += '\n';
    }
    void hist(const char* name, const char* doc, const std::string& labels, const Histogram& h) {
        help(name, doc, "histogram");
        for (size_t i = 0; i < h.bounds().size(); ++i) {
            out += name; out += "_bucket";
            out += labels.substr(0, labels.size() - 1);   /* drop the closing brace */
            out += ",le=\""; out += num(h.bounds()[i]); out += "\"} ";
            out += num((double)h.bucket(i)); out += '\n';
        }
        out += name; out += "_bucket";
        out += labels.substr(0, labels.size() - 1);
        out += ",le=\"+Inf\"} ";
        out += num((double)h.count()); out += '\n';
        out += name; out += "_sum"; out += labels; out += ' '; out += num(h.sum()); out += '\n';
        out += name; out += "_count"; out += labels; out += ' ';
        out += num((double)h.count()); out += '\n';
    }
};

}  /* namespace */

std::string Metrics::render() const {
    Writer w;
    w.out.reserve(8192);
    const std::string& L = labels_;

    /* ---------------- vLLM-compatible: gauges ---------------- */
    w.gauge("vllm:num_requests_running",
            "Number of requests currently running on GPU.", L, g_running_.get());
    w.gauge("vllm:num_requests_waiting",
            "Number of requests waiting to be processed.", L, g_waiting_.get());
    w.gauge("vllm:gpu_cache_usage_perc",
            "GPU KV-cache usage. 1 means 100 percent usage.", L, g_kv_util_.get());
    w.gauge("vllm:kv_cache_usage_perc",
            "KV-cache usage. 1 means 100 percent usage.", L, g_kv_util_.get());
    w.gauge("radiance:kv_cache_max_tokens",
            "Tokens of context the KV cache holds with every block it lends the experts taken "
            "back.", L, g_kv_max_tokens_.get());
    w.gauge("radiance:kv_cache_max_bytes",
            "VRAM the KV cache holds with every block it lends the experts taken back, one card.",
            L, g_kv_max_bytes_.get());
    w.gauge("vllm:gpu_prefix_cache_hit_rate",
            "GPU prefix cache block hit rate.", L, g_prefix_hit_.get());

    /* ---------------- vLLM-compatible: counters ---------------- */
    w.counter("vllm:prompt_tokens_total",
              "Number of prefill tokens processed.", L, c_prompt_tokens_.get());
    w.counter("vllm:generation_tokens_total",
              "Number of generation tokens processed.", L, c_generation_tokens_.get());
    w.counter("vllm:num_preemptions_total",
              "Cumulative number of preemptions from the engine.", L, c_preemptions_.get());

    w.help("vllm:request_success_total",
           "Count of successfully processed requests.", "counter");
    for (size_t i = 0; i < (size_t)SuccessReason::N_; ++i) {
        w.out += "vllm:request_success_total";
        w.out += L.substr(0, L.size() - 1);
        w.out += ",finished_reason=\"";
        w.out += reason_str((SuccessReason)i);
        w.out += "\"} ";
        w.out += num(c_success_[i].get());
        w.out += '\n';
    }

    /* ---------------- vLLM-compatible: histograms ---------------- */
    w.hist("vllm:time_to_first_token_seconds",
           "Histogram of time to first token in seconds.", L, h_ttft_);
    w.hist("vllm:time_per_output_token_seconds",
           "Histogram of time per output token in seconds.", L, h_tpot_);
    /* The name current vLLM uses for the same measurement. Same data, both spellings, so a
     * dashboard written against either version works unchanged. */
    w.hist("vllm:inter_token_latency_seconds",
           "Histogram of inter-token latency in seconds.", L, h_tpot_);
    w.hist("vllm:e2e_request_latency_seconds",
           "Histogram of end to end request latency in seconds.", L, h_e2e_);
    w.hist("vllm:request_decode_time_seconds",
           "Histogram of time spent in the decode phase, in seconds.", L, h_decode_);
    w.hist("vllm:request_prompt_tokens",
           "Number of prefill tokens processed.", L, h_prompt_tokens_);
    w.hist("vllm:request_generation_tokens",
           "Number of generation tokens processed.", L, h_generation_tokens_);
    w.hist("vllm:request_params_n",
           "Histogram of the n request parameter.", L, h_params_n_);
    w.hist("vllm:request_params_max_tokens",
           "Histogram of the max_tokens request parameter.", L, h_params_max_tokens_);

    /* ---------------- radiance-specific ----------------
     * Under our own prefix, never under vllm:. A dashboard that finds an unfamiliar vllm: name is
     * a dashboard that has been lied to about which engine it is looking at. */
    w.gauge("radiance:prefill_tokens_per_second",
            "Prefill throughput over the scheduler's recent window.", L, g_prefill_tps_.get());
    w.gauge("radiance:decode_tokens_per_second",
            "Decode throughput over the scheduler's recent window.", L, g_decode_tps_.get());
    w.counter("radiance:engine_steps_total",
              "Scheduler steps executed. One forward pass of the model, plus whatever draft "
              "rounds followed it; with speculation a step emits more than one token.",
              L, c_steps_.get());
    /* STEPS BY SHAPE, and how much of the decode ran unspeculated.
     * `radiance:decode_rows_unspeculated_total` counts rows that took no draft for ANY reason --
     * depth 0, a request that may not speculate, a drafter with nothing to propose. Its rate
     * against engine_steps_total says how much of the decode speculation is not reaching. */
    w.counter("radiance:engine_steps_prefill_total",
              "Scheduler steps carrying only prefill chunks.", L, c_steps_prefill_.get());
    w.counter("radiance:engine_steps_mixed_total",
              "Scheduler steps carrying both a prefill chunk and a decode row.",
              L, c_steps_mixed_.get());
    w.counter("radiance:engine_steps_decode_total",
              "Scheduler steps carrying only decode rows.", L, c_steps_decode_.get());
    w.counter("radiance:engine_steps_ahead_total",
              "Steps whose forward pass was issued behind the step before, ahead of its commit.",
              L, c_steps_ahead_.get());
    w.counter("radiance:decode_rows_unspeculated_total",
              "Decode rows that took no draft, for any reason.",
              L, c_rows_unspec_.get());
    /* THE STEP TIME, AS A COUNTER AND NOT A RATE. Divide by engine_steps_decode_total for the
     * milliseconds a decode step costs. Wall clock over a step count answers that only for a
     * window that was saturated and carried no prefill, and says nothing when it was not. */
    w.counter("radiance:engine_step_seconds_total",
              "Seconds spent inside a scheduler step, all shapes.", L, c_step_secs_.get());
    w.counter("radiance:engine_step_seconds_decode_total",
              "Seconds spent inside a scheduler step carrying only decode rows.",
              L, c_step_secs_decode_.get());

    /* THE EXACT TOTALS, beside the rate gauges above rather than instead of them. A gauge is what
     * a dashboard wants; a counter is what a comparison wants, because two runs that settled into
     * different mixes of prefill and decode have incomparable rates and perfectly comparable
     * totals. These are the only counters that answer it: vllm:generation_tokens_total is charged
     * when a request FINISHES, and is therefore identically zero over a window in which nothing
     * finishes. */
    w.counter("radiance:prefill_tokens_total",
              "Prompt tokens the engine has computed, exact.", L, c_prefill_tokens_.get());
    w.counter("radiance:decode_tokens_total",
              "Tokens the engine has generated, exact, counted when the step commits rather "
              "than when the request finishes.", L, c_decode_tokens_.get());
    w.counter("radiance:draft_tokens_total",
              "Speculative tokens proposed.", L, c_draft_tokens_.get());
    w.counter("radiance:draft_accepted_total",
              "Speculative tokens accepted; the pair the acceptance ratio is computed from.",
              L, c_draft_accepted_.get());
    w.counter("radiance:draft_refills_total",
              "Prefix hits ended a drafter window early so a windowed block drafter has context.",
              L, c_draft_refills_.get());
    w.counter("radiance:draft_refill_tokens_total",
              "Cached prompt tokens those hits gave back to be recomputed.",
              L, c_draft_refill_tokens_.get());

    /* PREFIX CACHING, DECOUPLED (spec §7.3). vllm:gpu_prefix_cache_hit_rate is the ATTENTION one,
     * because that is what vLLM's single number has always meant. The linear side hits at a
     * different granularity -- whole checkpoints, not blocks -- and averaging them would hide the
     * effect the decoupling exists to expose. Without these a deployment cannot tell whether
     * linear checkpointing is working at all. */
    w.gauge("radiance:linear_prefix_cache_hit_rate",
            "Fraction of prompt tokens served from a linear-state checkpoint (spec 7.3).",
            L, g_linear_hit_.get());
    w.counter("radiance:attn_cached_tokens_total",
              "Prompt tokens served from the paged attention prefix cache.", L, c_attn_cached_.get());
    w.counter("radiance:linear_cached_tokens_total",
              "Prompt tokens served from a linear-state checkpoint.", L, c_linear_cached_.get());
    w.counter("radiance:linear_checkpoints_written_total",
              "Linear-state checkpoints written.", L, c_checkpoints_.get());
    /* THE LEADING INDICATOR FOR A PREFIX-CACHE COLLAPSE. `gpu_prefix_cache_hit_rate` falls
     * only after a reusable prefix is already gone; this rises as it is dropped. Above ~75%
     * pool utilisation the cache starts evicting prefixes that the next turn wants, and an
     * agent's turn-2 latency goes from seconds to minutes. */
    w.counter("radiance:prefix_cache_evictions_total",
              "Prefix-cache entries evicted (attention blocks).", L, c_prefix_evict_.get());
    w.counter("radiance:prefix_cache_ckpt_evictions_total",
              "Prefix-cache linear-state checkpoints evicted.", L, c_prefix_ckpt_evict_.get());
    w.gauge("radiance:draft_acceptance_ratio",
            "Fraction of speculative draft tokens accepted (spec 10).", L, g_draft_accept_.get());
    /* CONDITIONAL, per draft position: of the verifies that reached position j -- every earlier
     * one accepted -- the fraction that took j as well. The pooled ratio above cannot tell a
     * drafter that decays from one whose later rounds are broken; these can, and the companion
     * `_reached_total` says how much evidence each position carries. */
    const int seen = draft_pos_seen_.load(std::memory_order_acquire);
    if (seen > 0) {
        const std::string base = L.substr(0, L.size() - 1);
        w.help("radiance:draft_position_acceptance_ratio",
               "Conditional acceptance at each draft position, given every earlier one was taken.",
               "gauge");
        for (int j = 0; j < seen; ++j)
            w.out += "radiance:draft_position_acceptance_ratio" + base + ",position=\"" +
                     std::to_string(j) + "\"} " + num(g_draft_pos_[j].get()) + '\n';
        w.help("radiance:draft_position_reached_total",
               "Verifies that evaluated each draft position.", "counter");
        for (int j = 0; j < seen; ++j)
            w.out += "radiance:draft_position_reached_total" + base + ",position=\"" +
                     std::to_string(j) + "\"} " + num(g_draft_pos_reached_[j].get()) + '\n';
    }
    /* THE ADAPTIVE DRAFT WINDOW, only while RADIANCE_DRAFT_ADAPT is on: where the policy stands --
     * at one sequence, that sequence's window -- and where it has been, as verifies by the
     * number of drafts they carried. */
    if (draft_adapt_on_.load(std::memory_order_acquire)) {
        w.gauge("radiance:draft_depth_adaptive",
                "Mean draft window the adaptive policy gives the running sequences.",
                L, g_draft_depth_adapt_.get());
        const int top = draft_vdepth_top_.load(std::memory_order_acquire);
        if (top > 0) {
            const std::string base = L.substr(0, L.size() - 1);
            w.help("radiance:draft_verify_depth_total",
                   "Verifies by the number of drafts they carried, under the adaptive window.",
                   "counter");
            for (int d = 1; d <= top; ++d)
                w.out += "radiance:draft_verify_depth_total" + base + ",depth=\"" +
                         std::to_string(d) + "\"} " + num(g_draft_vdepth_[d].get()) + '\n';
        }
    }
    w.gauge("radiance:num_requests_preempted",
            "Requests currently preempted and awaiting recompute.", L, g_preempted_.get());
    w.gauge("radiance:http_queue_depth",
            "Requests admitted by the HTTP layer and not yet finished.", L, queue_depth_.get());
    w.counter("radiance:http_admission_rejected_total",
              "Requests refused with 429 because the admission queue was full.",
              L, c_rejected_.get());
    w.counter("radiance:client_disconnect_total",
              "Streams cancelled because the client went away.", L, c_disconnect_.get());
    w.counter("radiance:stream_overflow_total",
              "Requests failed because the HTTP reader could not drain the sink.",
              L, c_overflow_.get());

    w.help("radiance:http_responses_total", "HTTP responses by status class.", "counter");
    w.out += "radiance:http_responses_total" + L.substr(0, L.size() - 1) + ",class=\"2xx\"} "
           + num(c_http_2xx_.get()) + '\n';
    w.out += "radiance:http_responses_total" + L.substr(0, L.size() - 1) + ",class=\"4xx\"} "
           + num(c_http_4xx_.get()) + '\n';
    w.out += "radiance:http_responses_total" + L.substr(0, L.size() - 1) + ",class=\"5xx\"} "
           + num(c_http_5xx_.get()) + '\n';

    return w.out;
}

}  /* namespace server */
}  /* namespace rad */
