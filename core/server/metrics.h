/* core/server/metrics.h -- Prometheus /metrics, with vLLM's metric names.
 *
 * The names are not ours to choose. There is a published Grafana dashboard for vLLM and a great
 * deal of alerting built on `vllm:time_to_first_token_seconds_bucket` and friends, and the point
 * of matching them is that swapping vLLM for radiance behind an existing deployment changes
 * nothing on the observability side. So: the vLLM names, the vLLM bucket boundaries, and the
 * vLLM label set (`model_name`, `engine`), copied exactly. Where vLLM renamed something we emit
 * both spellings -- `vllm:gpu_cache_usage_perc` is what every dashboard in the wild queries and
 * `vllm:kv_cache_usage_perc` is what current vLLM emits, and two gauges cost nothing.
 *
 * Anything vLLM does not have goes under `radiance:`. Inventing a new `vllm:` name would be worse
 * than useless: it would look to a dashboard like a metric it knows and is not.
 *
 * What is deliberately NOT emitted is as important. `vllm:iteration_tokens_total` and
 * `vllm:request_queue_time_seconds` are properties of the scheduler's step loop, not of the HTTP
 * boundary, and the server cannot observe them. Emitting a plausible-looking approximation of a
 * latency breakdown is how a dashboard ends up lying, so they are absent until the scheduler
 * reports them, and their absence is documented rather than papered over.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct SchedMetrics;

/* ------------------------------------------------------------------ primitives */

class Counter {
public:
    void inc(double v = 1.0) {
        double cur = v_.load(std::memory_order_relaxed);
        while (!v_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {}
    }
    double get() const { return v_.load(std::memory_order_relaxed); }
private:
    std::atomic<double> v_{0.0};
};

class Gauge {
public:
    void set(double v) { v_.store(v, std::memory_order_relaxed); }
    double get() const { return v_.load(std::memory_order_relaxed); }
private:
    std::atomic<double> v_{0.0};
};

/* Cumulative buckets, the Prometheus convention: bucket i counts observations <= bounds[i], and
 * the implicit +Inf bucket is `count`. Atomics rather than a lock because /metrics is scraped
 * from one thread while every HTTP thread observes into it. */
class Histogram {
public:
    explicit Histogram(std::vector<double> bounds);
    void observe(double v);

    const std::vector<double>& bounds() const { return bounds_; }
    uint64_t bucket(size_t i) const { return cnt_[i].load(std::memory_order_relaxed); }
    uint64_t count() const { return n_.load(std::memory_order_relaxed); }
    double   sum() const { return sum_.load(std::memory_order_relaxed); }

private:
    std::vector<double> bounds_;
    std::unique_ptr<std::atomic<uint64_t>[]> cnt_;
    std::atomic<uint64_t> n_{0};
    std::atomic<double>   sum_{0.0};
};

/* vLLM's bucket layouts, verbatim. Changing one of these silently invalidates every histogram
 * quantile a dashboard computes across a version boundary, so they are constants, not options. */
std::vector<double> buckets_ttft();          /* vllm:time_to_first_token_seconds */
std::vector<double> buckets_tpot();          /* vllm:time_per_output_token_seconds */
std::vector<double> buckets_latency();       /* every vllm:request_*_seconds */
std::vector<double> buckets_1_2_5(int64_t max_value);   /* token-count histograms */

/* ------------------------------------------------------------------ the registry */

/* The finish reasons `vllm:request_success_total` is broken down by. Fixed, because a label whose
 * cardinality depends on user input is a Prometheus outage waiting to happen. */
enum class SuccessReason { Stop = 0, Length, ToolCalls, Abort, Error, N_ };

class Metrics {
public:
    Metrics(std::string model_name, int64_t max_model_len);

    /* --- observed at the HTTP boundary, per finished request --- */
    void observe_request(int64_t n_prompt, int64_t n_generated,
                         double ttft_s, double tpot_s, double e2e_s, double decode_s,
                         int n_choices, int64_t max_tokens, SuccessReason r);
    void observe_rejected();        /* 429: the admission queue was full */
    void observe_disconnect();      /* the client went away mid-stream */
    void observe_overflow();        /* the sink ring filled: the HTTP side could not keep up */
    void observe_http(int status);

    void set_queue_depth(int64_t d) { queue_depth_.set((double)d); }

    /* --- pulled from the scheduler at scrape time --- */
    void sample(const SchedMetrics& m);

    /* Prometheus text exposition format, version 0.0.4. */
    std::string render() const;


private:
    std::string model_;
    std::string labels_;        /* {model_name="...",engine="0"} */

    Gauge   g_running_, g_waiting_, g_kv_util_, g_prefix_hit_;
    Gauge   g_kv_max_tokens_, g_kv_max_bytes_;
    Gauge   g_prefill_tps_, g_decode_tps_, g_draft_accept_, g_preempted_;
    Gauge   queue_depth_;
    /* The per-position draft hazard, kept as plain gauges because that is what it is: a
     * conditional acceptance rate at each draft position, which is the only number that separates
     * a drafter that decays from one whose later rounds are broken. See core/sched/draft.h. */
    enum { MAX_DRAFT_POS = 64 };
    Gauge   g_draft_pos_[MAX_DRAFT_POS];
    Gauge   g_draft_pos_reached_[MAX_DRAFT_POS];
    Gauge   g_draft_pos_accepted_[MAX_DRAFT_POS];
    std::atomic<int> draft_pos_seen_{0};
    /* THE ADAPTIVE DRAFT WINDOW (--draft-window adaptive): the mean window of the running sequences,
     * and the verifies by the window they carried, index = drafts. Rendered only while the policy
     * is on, so a deployment without it exposes exactly what it always did. */
    Gauge   g_draft_depth_adapt_;
    Gauge   g_draft_vdepth_[MAX_DRAFT_POS + 1];
    std::atomic<bool> draft_adapt_on_{false};
    std::atomic<int>  draft_vdepth_top_{0};   /* the deepest window with a verify counted */

    /* sample() turns the scheduler's cumulative totals into counter increments by reading each
     * counter and adding the difference. Two scrapes doing that at once both read the old value
     * and both add the whole difference, so they are serialised. */
    std::mutex sample_mu_;

    Counter c_prompt_tokens_, c_generation_tokens_, c_preemptions_;
    /* ENGINE STEPS. Tokens a second cannot answer "how long is a step", and with speculation the
     * two stop being the same question: a step emits 1 + however many drafts survived. Exposed as
     * a counter so two scrapes and a clock give the step time directly. */
    Counter c_steps_;
    /* STEPS BY SHAPE, and the decode rows a mixed step cost speculation. Exposed as counters so
     * two scrapes give the rate over a window rather than a number dominated by startup. */
    Counter c_steps_prefill_, c_steps_mixed_, c_steps_decode_, c_rows_unspec_, c_steps_ahead_;
    /* SECONDS SPENT INSIDE A STEP, and the decode half of it. A step time is not wall clock over
     * a step count -- SchedMetrics::step_ns says why -- and without these every scraper has to
     * assume a saturated window and no prefill in it. */
    Counter c_step_secs_, c_step_secs_decode_;
    /* THE EXACT TOTALS BEHIND THE RATE GAUGES. A gauge answers "how fast right now"; a counter
     * answers "how much between these two scrapes", and only the second survives a comparison
     * between two runs that settled into different mixes. */
    Counter c_prefill_tokens_, c_decode_tokens_, c_draft_tokens_, c_draft_accepted_;
    Counter c_draft_refills_, c_draft_refill_tokens_;
    Counter c_lookup_taken_, c_lookup_steps_, c_lookup_drafted_, c_lookup_accepted_;
    Counter c_attn_cached_, c_linear_cached_, c_checkpoints_;
    Counter c_prefix_evict_, c_prefix_ckpt_evict_;
    Gauge   g_linear_hit_;
    Counter c_success_[(size_t)SuccessReason::N_];
    Counter c_rejected_, c_disconnect_, c_overflow_;
    Counter c_http_2xx_, c_http_4xx_, c_http_5xx_;

    Histogram h_ttft_, h_tpot_, h_e2e_, h_decode_;
    Histogram h_prompt_tokens_, h_generation_tokens_, h_params_n_, h_params_max_tokens_;
};

}  /* namespace server */
}  /* namespace rad */
