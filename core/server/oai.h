/* core/server/oai.h -- the OpenAI schema, in and out.
 *
 * Two rules shape everything here.
 *
 * Errors are values. Nothing in this file throws; every entry point returns RAD_OK or a negative
 * status and fills an ApiError that already knows its HTTP status, its `param`, and a message
 * that names what was wrong. The HTTP layer turns that into a status code and a body. An
 * exception crossing the transport is a connection closed with no body, which is the least
 * diagnosable thing a client can be handed.
 *
 * A parameter we do not support is a 400 that names it. Not a silent ignore. A client that sends
 * `logit_bias` and gets a 200 has been told its bias was applied, and the resulting bug report
 * ("the model ignores my bias") is unanswerable from the server side. The cost of strictness is
 * that a client sending a field we have not heard of gets a 400 rather than a shrug; the
 * allowlist below is where the genuinely inert OpenAI metadata fields (`user`, `store`, ...)
 * are named, so that cost falls only on fields that actually mean something.
 */
#pragma once
#include "iface.h"
#include "rad_core.h"
#include "sink.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* ================================================================== errors */

struct ApiError {
    int         status = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param;      /* "" when the error is not about one field */
    std::string code;       /* "" when there is nothing more specific to say */

    std::string dump() const;    /* the OpenAI error envelope, as a JSON string */
};

ApiError err_bad(const std::string& param, const std::string& why);
ApiError err_unsupported(const std::string& param, const std::string& why);
ApiError err_not_implemented(const std::string& what);

/* SERIALISE OUT OF LINE. nlohmann's dump() inlines a shared_ptr release that GCC 16 speculatively
 * devirtualises onto std::mutex and then reports as an out-of-bounds destroy -- a false positive
 * in a vendored header, which appears or not depending on how deeply the CALLING function
 * inlined. That makes "is this file warning-clean" a property of how short its handlers are,
 * which is not something anybody can act on. One definition, in one translation unit. */
std::string json_dump(const json& j);
ApiError err_overloaded(const std::string& why);

/* ================================================================== a parsed request */

enum class Endpoint { Chat, Completion, Embedding };

struct PromptInput {
    std::vector<int32_t> tokens;
    std::string          text;    /* the rendered prompt; needed for `echo` and for the logs */
    /* The media parts, processed and placed; null for a text prompt (Request::media). */
    std::shared_ptr<mm::PromptMedia> media;
};

struct OaiRequest {
    Endpoint    endpoint = Endpoint::Chat;
    std::string model;
    std::string id;               /* "chatcmpl-..." / "cmpl-..." */
    int64_t     created = 0;      /* unix seconds */

    bool stream = false;
    bool include_usage = false;   /* stream_options.include_usage */

    int             n = 1;
    SamplingParams  sp;
    int32_t         max_tokens = 0;
    std::vector<std::string> stop;
    /* The reasoning block's closer when the rendered prompt leaves one open, else "": the stop
     * strings above then apply only after it (IDetokenizer::defer_stops_until). */
    std::string stops_after;
    /* Generate past end-of-text, to max_tokens. See Request::ignore_eos. */
    bool        ignore_eos = false;
    int             priority = 0;

    std::vector<PromptInput> prompts;   /* /v1/completions accepts a batch; chat has exactly one */
    bool echo = false;

    int  n_logprobs = 0;          /* 0 = off */
    bool want_logprobs = false;

    bool tools_present = false;
    /* FOR THE REQUEST LOG, counted while the parse walks the body anyway: the messages and tools
     * a chat request carried, and the last user message's opening (one line, bounded) that the
     * debug level prints so a log line can be matched to what was typed. Completion requests
     * leave them at zero and empty. */
    int         n_messages = 0;
    int         n_tools = 0;
    std::string last_user;
    std::string peer;             /* the handler sets it from the transport; "" in-process */
    bool parse_tools = false;
    /* The reply parser this request's render produced. Per request and not per server: it is the
     * model's format bound to the tools THIS call sent, so two concurrent requests against the
     * same model can legitimately need different parsers. */
    std::shared_ptr<IReplyParser> reply_parser;

    /* The control tokens this request's reply must be decoded AS TEXT, so `tool_parser` can see
     * the structure it is looking for. See ITokenizer::detokenizer. */
    std::vector<std::string> preserved_tokens;

    /* /v1/embeddings */
    bool base64_embeddings = false;

    int  n_choices() const { return (int)prompts.size() * n; }
};

/* What the server needs to know about itself to validate a request. */
struct OaiLimits {
    std::string model_id = "radiance";
    int64_t     max_ctx = 0;              /* 0 = unknown, no length check */
    /* What a request that names no max_tokens may generate. 0 is "whatever the context leaves",
     * resolved against the prompt in clamp_max_tokens, which needs max_ctx to resolve against.
     *
     * 0 BY DEFAULT, as vLLM's is: many clients send no max_tokens, and a reasoning model spends
     * the first thousands of tokens of an answer thinking. A fixed default of 512 ended those
     * answers inside the reasoning, with finish_reason "length" that a chat client does not show,
     * so they read as a model that stops at random. */
    int64_t     default_max_tokens = 0;
    /* The most any request may generate; a larger max_tokens is clamped, not refused. 0 = none. */
    int64_t     max_tokens_cap = 0;
    int         max_n = 8;
    /* How many stop strings a request may send, and how long each may be. */
    int64_t     max_stops = 64;
    int64_t     max_stop_bytes = 4096;
    bool        allow_image = false;
    bool        allow_video = false;
    bool        allow_audio = false;
    bool        supports_logprobs = false;

    /* THE DEPLOYMENT'S DEFAULT REASONING EFFORT, empty for "whatever the template's own default
     * is". It exists because the template's default is not always a servable one: Qwen3.8's is
     * `xhigh`, which on a short task reliably spends the whole token budget inside <think> and
     * returns a turn with EMPTY content and finish_reason "length" -- an agent turn that carries
     * no answer and no tool call. The per-request field is the right control and most OpenAI
     * clients never send it, so without this an operator serving such a client cannot reach the
     * dial at all. Seeded only when the request says nothing, so a caller always wins. */
    std::string default_reasoning_effort;

    /* THE DEPLOYMENT'S DEFAULT `thinking_token_budget`, -1 for none. For the same reason the
     * effort default exists: an agent framework sizes max_tokens per turn and sends no budget, and
     * a reasoning block that spends the whole of max_tokens returns a turn with no answer in it.
     * Applied only to a request that names no budget, and only where the reply's reasoning block
     * can be counted. */
    int64_t     default_thinking_budget = -1;

    /* WHAT A REQUEST THAT SENDS NO SAMPLER FIELDS IS SERVED WITH, field by field.
     *
     * The struct's own defaults are an untruncated sampler -- temperature 1, top_k off, top_p 1
     * -- which is the right thing to fall back to and the wrong thing to serve a checkpoint
     * with: a model tuned against a truncated tail, sampled without one, will occasionally draw
     * a token from far down its distribution. End-of-turn is one of those tokens, and drawing it
     * early ends an agent's turn with prose and no tool call, which reads as the server dropping
     * the request. The deployment resolves this from what the checkpoint itself states and hands
     * the result over here.
     *
     * Only the sampler knobs are read: the grammar fields are per request and are filled in
     * after this is copied. A request that names a field always wins, exactly as with
     * default_reasoning_effort. */
    SamplingParams default_sampling;
    /* SamplingParams cannot tell an empty breaker list that was stated from one that was not, and
     * the two mean different things: unstated is llama-server's four, stated-empty is none. */
    bool           default_dry_breakers_set = false;

    /* THE DEPLOYMENT'S CHAT TEMPLATE VARIABLES, key -> JSON text, applied under the request's own
     * chat_template_kwargs so a key the request names is the request's. The thinking keys follow
     * parse_thinking's rule: a request that says anything about thinking replaces them whole. */
    std::map<std::string, std::string> default_template_kwargs;
    /* "auto" returns reasoning as reasoning_content; "none" leaves it inline in content. */
    std::string reasoning_format = "auto";
};

/* WHAT A REQUEST SAYS ABOUT THINKING, in any of its three spellings: `effort` when it names a
 * reasoning effort (top-level or in chat_template_kwargs, non-empty), `thinking` when it says
 * anything at all -- an effort, or enable_thinking in either place. */
struct RequestThinking {
    bool thinking = false;
    bool effort = false;
};
RequestThinking request_thinking(const json& b);

/* The deployment's template variables and reply format, onto `opt`, before the request's own
 * chat_template_kwargs are written over them. A request that says anything about thinking keeps
 * the deployment's enable_thinking out; one that names an effort keeps its reasoning_effort out.
 * The chat endpoint and /tokenize both render through this, so a prompt counted by one is the
 * prompt served by the other. */
void apply_template_defaults(const json& b, const OaiLimits& lim, ChatRenderOptions* opt);

/* True when `v` is a legal value of the sampler field a request spells `field`; otherwise `why`
 * says what the legal set is, in the words a 400 carries. The same table checks a request and the
 * command-line flag that sets the field's default, so a default the command line accepts is one
 * every request omitting the field can be served with. A field with no bounds takes any value. */
bool sampler_value_ok(const char* field, double v, std::string* why);

struct OaiDeps {
    ITokenizer*       tok = nullptr;
    IChatTemplate*    chat = nullptr;
    IGrammarCompiler* grammar = nullptr;
    IMultimodal*      mm = nullptr;
};

/* Parse. `body` is the raw request body; a malformed body is a 400 that says where. */
int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err);
int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);
/* Embeddings: fills `out->prompts` with one entry per input. */
int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);

/* ================================================================== responses */

struct LogprobEntry {
    int32_t     token = 0;
    std::string text;
    float       logprob = 0.0f;
};

struct ChoiceOut {
    int           index = 0;
    std::string   text;          /* raw generated text */
    ParsedMessage msg;           /* chat: content / reasoning / tool_calls after parsing */
    Finish        finish = Finish::None;
    std::vector<LogprobEntry> logprobs;
};

struct UsageOut {
    int64_t prompt = 0;
    int64_t completion = 0;
    int64_t cached = 0;
    /* Of `completion`, the tokens inside the reply's reasoning block, its markers included: OpenAI's
     * completion_tokens_details.reasoning_tokens. -1 when the block cannot be counted, and the
     * breakdown is then left out. */
    int64_t reasoning = -1;
};

/* WHERE ONE REQUEST'S TIME WENT, in llama.cpp's `timings` shape, which proxies and dashboards
 * built against llama-server already read (llama-swap's activity view is one). /metrics holds the
 * same quantities summed over every request, and two overlapping requests cannot be told apart
 * there; this is the per-request half.
 *
 *   prompt_n      prompt tokens computed, the prefix cache's excluded; cache_n is those
 *   prompt_ms     from the first step the scheduler gave the request to its first token -- the
 *                 prefill, without the time it waited for admission
 *   predicted_n   tokens generated: usage.completion_tokens
 *   predicted_ms  from the first token to the last
 *   draft_n, draft_n_accepted
 *                 speculative tokens proposed for this request and accepted; present when any
 *                 were proposed, as llama.cpp does
 *
 * The rates divide the tokens by the time that produced them: prompt_n by prompt_ms, and the
 * predicted_n - 1 tokens after the first by predicted_ms, since the first token is the prompt
 * phase's product -- the same figure as /metrics' time per output token. A rate whose count or
 * time is zero is null. Over several choices the counts are summed and each time runs from the
 * earliest choice's start to the latest choice's end. */
struct TimingsOut {
    bool    valid = false;       /* some choice produced a token; nothing is reported otherwise */
    int64_t prompt_n = 0, cache_n = 0;
    double  prompt_ms = -1.0;    /* < 0: the engine did not report the admission */
    int64_t predicted_n = 0;
    int64_t decode_n = 0;        /* of predicted_n, the tokens after each choice's first */
    double  predicted_ms = 0.0;
    int64_t draft_n = 0, draft_n_accepted = 0;
};

/* `timings` is added at the top level when it is valid. */
std::string chat_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t = TimingsOut());
std::string text_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t = TimingsOut());
std::string embeddings_body(const OaiRequest& r, const std::vector<std::vector<float>>& vecs,
                            const UsageOut& u);
std::string models_body(const std::string& model_id, int64_t created);

/* The usage chunk of each endpoint's stream: an SSE event whose choices array is empty and whose
 * only payload is `usage`. Emitted when the request asked for stream_options.include_usage, ONCE
 * per stream and after every choice's final chunk -- `n` choices share one prompt and one total,
 * so a chunk per choice would count the prompt n times and interleave with the choices. */
std::string text_usage_chunk(const OaiRequest& r, const UsageOut& u,
                             const TimingsOut& t = TimingsOut());
std::string chat_usage_chunk(const OaiRequest& r, const UsageOut& u,
                             const TimingsOut& t = TimingsOut());
/* The `timings` object, for a stream's final chunks. */
json timings_json(const TimingsOut& t);

/* Whether a parsed reply's calls are all calls a client can run: none left open, each named,
 * each one's arguments one JSON value (json_partial.h). A reply is answered `tool_calls` by this
 * alone, streamed or not. */
bool calls_complete(const ParsedMessage& m);

/* ================================================================== streaming
 *
 * One of these per choice. It feeds the generated text to the choice's reply reader as it grows
 * and forwards each increment the reader completes as an SSE chunk. The reader never revises
 * what it has delivered, so nothing here diffs, holds back or reconciles: a delta is the next
 * piece of the answer, and the chunks a client concatenates are the answer the buffered path
 * returns.
 */
class ChatDeltaStream {
public:
    ChatDeltaStream(const OaiRequest& r, int index, const IReplyParser* parser);

    /* The `role: assistant` opener OpenAI clients expect before any content. */
    std::string first_chunk();

    /* Feed the text generated since the last push; returns zero or more SSE `data:` lines.
     * `is_final` says the generation is over, which resolves whatever the reader was holding. */
    std::string push(std::string_view appended, bool is_final);

    /* The terminating chunk carrying finish_reason, and this choice's `timings` when they are
     * valid -- on the final chunk whether or not the request asked for a usage chunk, as
     * llama.cpp sends them. The usage chunk is the stream's, not the choice's: chat_usage_chunk. */
    std::string final_chunk(Finish f, const TimingsOut& t = TimingsOut());

    /* What the whole generation parsed to -- the non-streaming shape. Complete once push has
     * been called with is_final. */
    const ParsedMessage& parsed() const { return last_; }
    Finish adjusted_finish(Finish f) const;

private:
    std::string chunk(json delta, const char* finish_reason, const TimingsOut* t = nullptr);
    std::string emit(const std::vector<ReplyDelta>& ev);

    /* A delta's chunk is written, not built: the frame around the delta object is the same in
     * every one, so it is serialised once, and the delta between is written with
     * json_quote_append. The bytes are the ones chunk() writes for the same delta. */
    void delta_open(std::string& out);
    void delta_close(std::string& out) const;
    static void quote(std::string& out, std::string_view s);
    std::string head_;     /* `data: {...,"choices":[{"index":N,"delta":`, built on first use */

    std::unique_ptr<IReplyReader> reader_;
    std::string id_, model_;
    int64_t     created_;
    int         index_;
    bool        finished_ = false;

    ParsedMessage            last_;
    std::vector<ReplyDelta>  ev_;
};

/* The SSE terminator. Written once at the end of every stream, including a failed one, because a
 * client blocked on the reader has no other way to learn the stream is over. */
extern const char* const SSE_DONE;

std::string sse_event(const std::string& json_text);
std::string sse_error(const ApiError& e);

/* ================================================================== small shared helpers */

std::string base64_encode(const void* data, size_t len);
bool        base64_decode(const std::string& in, std::vector<uint8_t>* out);
/* A token id as a request spells it: an integer in [0, INT32_MAX]. False for anything else, and
 * the caller refuses the request -- narrowing instead would make 2^32 + 5 an alias of token 5. */
bool        token_id(const json& v, int32_t* out);

/* Parse a request body into a tree of at most json_value_budget(max_ctx) values, nested at most
 * kJsonMaxDepth deep. A syntax error comes back discarded, as json::parse's would; a body with
 * more values than the budget sets `*over`, and one nested deeper than the bound sets `*deep`,
 * and either comes back holding only what fitted, which the caller refuses. See the definition
 * for why the tree is bounded rather than the body alone. */
json        parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep);
int64_t     json_value_budget(int64_t max_ctx);
constexpr int kJsonMaxDepth = 1024;

std::string random_id(const char* prefix);
int64_t     unix_now();

}  /* namespace server */
}  /* namespace rad */
