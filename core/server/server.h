/* core/server/server.h -- the OpenAI-compatible server (spec §14).
 *
 * Threading, in the spec's terms (§1): HTTP threads feed a request queue, one scheduler thread
 * produces one step. This object owns the HTTP thread pool and the admission queue; it does not
 * own the step loop and never calls into it. Output comes back per request through a Sink the
 * scheduler pushes into (sink.h), which `Request::sink` points at.
 *
 * Backpressure is admission control, not buffering. A full queue is a 429 with Retry-After. The
 * alternative -- accepting everything and letting the waiting list grow -- turns a load spike
 * into an out-of-memory kill twenty minutes later, at which point every in-flight request dies
 * rather than the marginal ones being refused up front.
 *
 * Cancellation is not an edge case. An agent client that abandons a long generation is the normal
 * case, so a dropped connection is detected on the streaming path (a failed write, or a closed
 * socket seen while waiting) and turned into IScheduler::cancel, which frees the request's KV
 * blocks immediately rather than at completion.
 */
#pragma once
#include <functional>
#include "http.h"
#include "iface.h"
#include "liveview.h"
#include "metrics.h"
#include "oai.h"
#include "sink.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct Deps {
    IScheduler*       sched   = nullptr;   /* required */
    ITokenizer*       tok     = nullptr;   /* required */
    IChatTemplate*    chat    = nullptr;   /* absent -> /v1/chat/completions is 501 */
    IGrammarCompiler* grammar = nullptr;   /* absent -> response_format is 501 */
    IEmbedder*        embed   = nullptr;   /* absent -> /v1/embeddings is 501 */
    IMultimodal*      mm      = nullptr;   /* absent -> image and audio parts are 501 */

    /* WHAT ONLY THE ENGINE CAN SEE: per-card VRAM, the expert plane. The server holds interfaces
     * and no device handle, so this dependency is a function rather than an object -- there is
     * nothing to name it after. Absent, the dashboard shows the scheduler's half and omits the
     * cards rather than drawing zeroes.
     *
     * The SAME function the live view is built with (liveview.h), so the terminal view and the
     * web page cannot drift into disagreeing about what the engine is doing. */
    LiveView::Source  live;
};

struct ServerOptions {
    std::string model_id = "radiance";
    /* AN EMBEDDING MODEL (rad_declare_pooled) GENERATES NOTHING: /v1/chat/completions and
     * /v1/completions answer 501 naming /v1/embeddings, rather than queueing a request the engine
     * would finish after one step with no text. */
    bool        embedding_model = false;
    /* What --model named, reported as the model entry's `root` the way vLLM reports the path it
     * loaded: a served name chosen for clients says nothing about which file is behind it. Empty
     * leaves the field out. */
    std::string model_path;
    std::string host = "0.0.0.0";
    int         port = 8000;
    int         n_threads = 0;

    /* How many requests may be admitted and unfinished at once. Beyond this the answer is 429.
     * Defaults to 8x max_seqs: enough that the scheduler always has a full batch to choose from
     * and a queue to prefill out of, and small enough that the memory held by waiting requests is
     * bounded by a number the operator picked. */
    int64_t     queue_depth = 0;
    int64_t     max_seqs = 256;
    int         retry_after_s = 1;

    int64_t     max_ctx = 0;
    int64_t     max_batch_tokens = 0;       /* 0: automatic shared-prefill grouping disabled */
    int64_t     default_max_tokens = 0;     /* 0: what the context leaves (OaiLimits) */
    int64_t     max_tokens_cap = 0;         /* 0: none */
    int         max_n = 8;
    int64_t     max_stops = 64;
    int64_t     max_stop_bytes = 4096;

    bool        supports_logprobs = false;
    /* Model metadata, not inferred from its name. Only a declared d1 profile currently has a
     * System One prompt/readout adapter. Empty means the model has no decision capability. */
    std::string decision_type;
    /* The deployment default for `reasoning_effort`, empty for the template's own. See
     * OaiLimits::default_reasoning_effort for why a server-side default is needed at all. */
    std::string default_reasoning_effort;
    /* The deployment default for `thinking_token_budget`, -1 for none. See
     * OaiLimits::default_thinking_budget. */
    int64_t     default_thinking_budget = -1;
    /* What a request that sends no sampler fields is served with. See
     * OaiLimits::default_sampling; only the sampler knobs are read from it. */
    SamplingParams default_sampling;
    bool        default_dry_breakers_set = false;
    /* The deployment's chat template variables and reply format. See OaiLimits. */
    std::map<std::string, std::string> default_template_kwargs;
    /* --reasoning-effort-map. See OaiLimits::reasoning_effort_map. */
    std::map<std::string, std::string> reasoning_effort_map;
    std::string reasoning_format = "auto";
    bool        allow_image = false;
    bool        allow_video = false;
    bool        allow_audio = false;

    /* WHAT /server_info AND /version REPORT ABOUT THE ENGINE BEHIND THIS SERVER.
     *
     * `engine_info` is a JSON OBJECT as text, spliced in verbatim under "engine". The server
     * cannot derive any of it -- it deliberately knows nothing about the device layer, the
     * container or the placement plan -- and a struct with a field per fact would be a struct
     * this component has to grow every time the engine learns something new about itself. */
    std::string version = RAD_VERSION;
    std::string engine_info;
    /* The declared compute graph, as JSON, for the Model view. A PROVIDER, called per request,
     * rather than a string rendered once after declare.
     *
     * The declaration itself does not change while the server runs, but the dump also carries what
     * the run phase has DONE with it -- `OpInfo::issued` -- and a snapshot taken before the first
     * step reports every op in the model as never issued, permanently, which makes /graph useless
     * for finding a dead declaration.
     *
     * Regenerating reads the Program from the HTTP thread while the step path writes that one
     * field. It is a monotonic false -> true bool, so the worst case is a graph fetched mid-step
     * showing an op as unissued one step before it flips. Registry::select is asked with
     * record = false and mutates nothing else. */
    std::function<std::string()> graph_json;

    std::string api_key;
    bool        cors = true;

    /* The transport's timeouts and body bound. See TransportOptions. */
    int         read_timeout_s = 30;
    int         write_timeout_s = 600;
    int         keep_alive_timeout_s = 5;
    size_t      max_body_bytes = 512ull << 20;
};

class Server {
public:
    Server(const Deps& d, const ServerOptions& o);
    /* Lifetime rule, and it is load-bearing: stop the scheduler before destroying the Server.
     * `Request::sink` is a raw pointer into storage this object owns, and a request the scheduler
     * has not finished is a request still writing into it. The destructor waits a bounded time
     * for anything it cancelled and then deliberately leaks the rest rather than free memory a
     * running step may touch. */
    ~Server();

    /* Every endpoint, registered. Tests drive this directly -- same handlers, no socket. */
    const Router& router() const { return router_; }

    int  run();            /* binds, serves, blocks. RAD_OK or negative. */

    /* THE DEPLOYMENT'S TEMPLATE DEFAULTS, RENDERED ONCE BEFORE A CALLER DEPENDS ON THEM. A value
     * the chat template refuses is a 400 on every request that leaves it to the default, so a
     * one-message chat is parsed exactly as a request would be: first with no template defaults,
     * then with the --chat-template-kwargs ones, then with the --reasoning-effort one as well.
     * A render that fails at the first step says nothing about the defaults and is not reported.
     * `kwargs_why` / `effort_why` are left empty, or hold the template's refusal at that step. */
    void check_template_defaults(std::string* kwargs_why, std::string* effort_why) const;
    /* THE DEPLOYMENT'S thinking_token_budget, asked for by a one-message chat the way a request
     * would ask for it. Empty when this model's reasoning block can hold one; otherwise the 400 a
     * request naming the budget would get, which says why. */
    std::string check_thinking_budget() const;
    /* THE EFFORTS THE CHAT TEMPLATE TAKES, found by parsing a one-message chat with each word this
     * server knows of (low, medium, high, xhigh, max) the way a request would send it, with no
     * deployment defaults. A template that takes a word nobody uses takes any word, and the answer
     * is empty. Kept for the refusal of a word outside the set and for /server_info; called once,
     * before run(). */
    std::vector<std::string> probe_reasoning_efforts();
    /* --reasoning-effort-map's targets, each rendered as a request's effort would be. Empty when
     * the template takes every one; otherwise the first refusal, naming the target. */
    std::string check_reasoning_effort_map() const;
    void stop();
    int  port() const { return transport_.port(); }   /* 0 until run() is accepting */

    int64_t  in_flight() const { return in_flight_.load(std::memory_order_relaxed); }

private:
    struct Generation;
    struct StreamState;

    static UsageOut usage_of(const OaiRequest& r, const std::vector<Generation>& gens);
    static TimingsOut timings_of(const Sink* const* sinks, size_t n);
    static TimingsOut timings_of(const std::vector<Generation>& gens);

    void register_routes();

    HttpResponse handle_chat(const HttpRequest& req);
    HttpResponse handle_completion(const HttpRequest& req);
    HttpResponse handle_embeddings(const HttpRequest& req);
    HttpResponse handle_systemone(const HttpRequest& req);
    HttpResponse handle_models(const HttpRequest& req);
    HttpResponse handle_metrics(const HttpRequest& req);

    /* ------------------------------------------------------------------ admin.cpp
     * vLLM's non-generation surface. Separate because none of it touches the scheduler's step
     * path, the admission queue or a Sink: these are questions about the server, answered on the
     * HTTP thread and finished before they return. */
    HttpResponse handle_health(const HttpRequest& req);
    HttpResponse handle_ping(const HttpRequest& req);
    HttpResponse handle_version(const HttpRequest& req);
    HttpResponse handle_props(const HttpRequest& req);
    HttpResponse handle_apply_template(const HttpRequest& req);
    HttpResponse handle_input_tokens(const HttpRequest& req);
    HttpResponse handle_messages(const HttpRequest& req);
    HttpResponse handle_count_tokens(const HttpRequest& req);
    HttpResponse handle_responses(const HttpRequest& req);
    HttpResponse handle_responses_tokens(const HttpRequest& req);
    /* A request another API's shape was translated into: admitted and served like a chat one,
     * answered in `r.shape`. */
    HttpResponse serve_translated(const HttpRequest& req, OaiRequest& r);
    HttpResponse handle_native_completion(const HttpRequest& req);
    HttpResponse handle_infill(const HttpRequest& req);
    json native_result(const OaiRequest& r, const Generation& g, const std::string& content);
    HttpResponse handle_load(const HttpRequest& req);
    HttpResponse handle_tokenize(const HttpRequest& req);
    HttpResponse handle_detokenize(const HttpRequest& req);
    HttpResponse handle_tokenizer_info(const HttpRequest& req);
    HttpResponse handle_server_info(const HttpRequest& req);
    /* What this deployment can do -- chat, tools, reasoning, grammar, media, logprobs, the
     * reasoning efforts it takes -- as /server_info's `capabilities` and the model entry both
     * report it, built in one place so the two cannot disagree. */
    json capabilities() const;
    HttpResponse handle_sessions(const HttpRequest& req);
    HttpResponse handle_reset_prefix_cache(const HttpRequest& req);

    /* ------------------------------------------------------------------ dashboard.cpp */
    HttpResponse handle_dashboard(const HttpRequest& req);
    HttpResponse handle_stats(const HttpRequest& req);
    HttpResponse handle_graph(const HttpRequest& req);

    HttpResponse error_response(const ApiError& e, const std::string& peer,
                                ApiShape shape = ApiShape::OpenAI);
    /* THE REQUEST LOG. One line when a request is admitted and one from record() when it ends,
     * at the info level, so `docker logs` answers "who asked what, how long, how did it end"
     * without the engine's own levels. The debug level adds the last user message's opening. */
    void log_admitted(const OaiRequest& r) const;

    /* Admission. Returns false when the queue is full; the caller answers 429. */
    bool  admit(int64_t n);
    void  release(int64_t n);
    void  sweep_abandoned();
    /* Wait, bounded, for the scheduler to finish everything it was told to cancel. */
    void  drain_graveyard(int timeout_ms);

    /* Turn a parsed request into `n_choices` submitted engine requests. */
    int   submit_all(const OaiRequest& r, std::vector<Generation>* gens, ApiError* err);
    void  cancel_all(std::vector<Generation>& gens, bool count_disconnect);
    void  retire(std::vector<Generation>& gens);

    /* Drain sinks into decoded text. Returns true if anything new arrived. */
    bool  pump(std::vector<Generation>& gens, const OaiRequest& r);
    bool  all_done(const std::vector<Generation>& gens) const;

    HttpResponse blocking_completion(const HttpRequest& req, OaiRequest& r);
    HttpResponse streaming_completion(const HttpRequest& req, OaiRequest& r);

    void  record(const OaiRequest& r, const std::vector<Generation>& gens, Finish overall);

    Deps           deps_;
    ServerOptions  opt_;
    OaiLimits      lim_;
    OaiDeps        odeps_;
    Router         router_;
    Metrics        metrics_;
    HttpTransport  transport_;

    std::atomic<int64_t> in_flight_{0};
    int64_t              queue_depth_ = 0;
    std::atomic<bool>    stopping_{false};
    int64_t              started_at_ = 0;

    /* Sinks whose HTTP side is gone but whose engine side may still be mid-step. Held until the
     * scheduler reports the request terminal, because Request::sink is a raw pointer and freeing
     * it under a running step is a use-after-free in the one place it is unbisectable. */
    std::mutex                          grave_m_;
    std::vector<std::shared_ptr<Sink>>  grave_;
};

}  /* namespace server */
}  /* namespace rad */
