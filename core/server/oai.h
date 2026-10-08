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
 * A parameter that would change the answer and is not served is a 400 that names it, not a
 * silent ignore: a client that sends a `logit_bias` and gets a 200 has been told its bias was
 * applied, and the bug report that follows ("the model ignores my bias") is unanswerable from the
 * server side. Such a field is accepted at its "off" value, which is what llama.cpp-built clients
 * send on every request. A field this server has never heard of is ignored, as llama.cpp and vLLM
 * ignore it (oai.cpp, key vetting).
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
    /* The values `param` takes, when the error is a value outside a known set: `supported` in the
     * envelope, so a client can retry with one rather than parse the sentence. */
    std::vector<std::string> supported;
    /* A prompt the context cannot hold: its length and the context's, under llama-server's names,
     * so a client that compacts its history on overflow reads the numbers rather than the
     * sentence. -1 leaves them out. */
    int64_t     n_prompt_tokens = -1;
    int64_t     n_ctx = -1;

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

/* WHICH API A CHAT REQUEST CAME IN THROUGH, and so which shape its answer goes out in. Every shape
 * is parsed into the same chat request (Anthropic's is translated first, anthropic_to_chat) and
 * served by the same path; only the bytes on the wire differ. */
enum class ApiShape { OpenAI, Anthropic, Responses, LlamaNative };

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

    ApiShape shape = ApiShape::OpenAI;
    bool stream = false;
    bool include_usage = false;   /* stream_options.include_usage */
    /* llama.cpp's streaming extras: `timings` on every chunk, not only the last; and
     * `prompt_progress` chunks while the prompt is read. */
    bool timings_per_token = false;
    bool return_progress = false;
    /* The conversation as the client names it (OpenRouter's `session_id`), "" when it names none:
     * a label on /sessions, nothing else. */
    std::string session_label;
    /* The build, as OpenAI's `system_fingerprint` and llama.cpp's own: what produced the reply. */
    std::string system_fingerprint;

    int             n = 1;
    SamplingParams  sp;
    int32_t         max_tokens = 0;
    std::vector<std::string> stop;
    /* Of `stop`, how many the client sent; the rest are the chat format's own end-of-turn
     * markers. A reply cut on one of the first is Anthropic's `stop_sequence`, on the rest its
     * `end_turn`. */
    size_t          n_client_stops = 0;
    /* The assistant text a continued conversation ends in (assistant prefill), "" otherwise. The
     * OpenAI shape answers the whole message, as llama-server does; Anthropic's answers what
     * follows it, as Anthropic's API does. */
    std::string     continued;
    /* The Responses API's request fields its response object echoes (instructions, tools,
     * tool_choice, text, reasoning, ...), as a JSON object; "" for the other shapes. */
    std::string     echo_json;
    /* llama-server's native /completion: the generated token ids beside the text. */
    bool            return_tokens = false;
    /* The reasoning block's closer when the reply starts inside the block, else "": the stop
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
    /* OpenAI's deprecated functions API: the request sent `functions`, so the reply carries its
     * call as `function_call` (one at most) and ends in finish_reason "function_call". */
    bool legacy_functions = false;
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
    /* OpenAI's `dimensions`: the vector's first `dimensions` components, renormalised -- a
     * Matryoshka-trained model's shorter embedding. 0 is the model's own width. */
    int64_t dimensions = 0;
    /* /v1/embeddings: a soft-token budget for the request's images, 0 the server's. */
    int32_t image_tokens = 0;

    int  n_choices() const { return (int)prompts.size() * n; }
};

/* What the server needs to know about itself to validate a request. */
struct OaiLimits {
    std::string model_id = "radiance";
    std::string version;                  /* the build, for `system_fingerprint` */
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

    /* THE WORDS THE CHAT TEMPLATE TAKES AS `reasoning_effort`, found at startup by rendering each
     * word this server knows of (Server::probe_reasoning_efforts). Empty when the template takes
     * any word -- it does not read the variable, or reads it without refusing one -- and when
     * nothing was probed. A request whose effort the template refuses is answered with this set
     * rather than with the template's traceback (template_refusal). */
    std::vector<std::string> reasoning_efforts;

    /* --reasoning-effort-map: an effort a request sends, replaced by the deployment's word before
     * the template sees it. For a client that always sends a word of OpenAI's ladder ("high")
     * to a model whose template has its own ("xhigh"). */
    std::map<std::string, std::string> reasoning_effort_map;

    /* A STARTUP PROBE, NOT A REQUEST: what the template refuses is the answer being asked for, so
     * its refusal is not logged (ChatRenderOptions::quiet). */
    bool        probe = false;

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

/* --reasoning-effort-map, on the effort chat_template_kwargs put in the render options: a mapped
 * word replaces it, and a word mapped to "none" or "minimal" turns thinking off instead. The chat
 * endpoint maps the top-level `reasoning_effort` itself, before it reads "none" and "minimal";
 * this is the kwargs spelling, for it and for /tokenize. */
void map_reasoning_effort(const OaiLimits& lim, ChatRenderOptions* opt);

/* THE 400 FOR A TEMPLATE THAT REFUSED A RENDER. An effort outside the template's known set
 * (OaiLimits::reasoning_efforts) is named as `reasoning_effort` with the set in `supported`, the
 * way OpenAI answers an unsupported value; anything else is `messages` with the template's own
 * sentence. */
ApiError template_refusal(const ChatRenderOptions& opt, const OaiLimits& lim, int rs,
                          const std::string& why);

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
    IEmbedder*        embed = nullptr;
};

/* Parse. `body` is the raw request body; a malformed body is a 400 that says where. */
int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err);
/* `native` is llama-server's /completion: its `return_tokens`, and a prompt array of strings and
 * token ids mixed as ONE prompt, as llama-server reads it. */
int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err, bool native = false);
/* Embeddings: fills `out->prompts` with one entry per input. */
int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);

/* ================================================================== responses */

/* One emitted token's logprobs as the wire carries them. `bytes` is the token's own bytes and
 * `text` is their longest well-formed UTF-8 prefix -- a byte-level token can hold half a
 * character, and a JSON string cannot -- llama-server's spelling. `end` is where the token's
 * decoded text ends in the choice's text, which is what decides whether it went out before a stop
 * string cut the reply. */
struct LogprobAlt {
    int32_t     token = 0;
    std::string text, bytes;
    float       logprob = 0.0f;
};
struct LogprobEntry {
    int32_t     token = 0;
    std::string text, bytes;
    float       logprob = 0.0f;
    std::vector<LogprobAlt> top;
    size_t      end = 0;
};

/* The entries as a JSON array's elements, comma-joined, in OpenAI's chat shape with llama-server's
 * `id` beside each token -- which is also llama-server's native `completion_probabilities`. */
std::string logprobs_content(const LogprobEntry* e, size_t n);
/* OpenAI's legacy /v1/completions `logprobs` object for the entries, `offset` the text offset of
 * the first, with llama-server's `content` array beside the legacy arrays. */
std::string logprobs_legacy(const LogprobEntry* e, size_t n, size_t offset);

struct ChoiceOut {
    int           index = 0;
    std::string   text;          /* raw generated text */
    std::string   stop_word;     /* the stop string that cut it, "" when none or unknown */
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
/* ONE MODEL ENTRY: what /v1/models lists and /v1/models/:id returns on its own. */
struct ModelCard {
    std::string id;
    int64_t     created = 0;
    /* The context the server was started with (--max-model-len, else the container's trained
     * length), the field vLLM's model card carries. Clients size their own context from it -- an
     * agent that compresses its history at the window's edge, a UI that shows how much is left --
     * and without it they guess from the model's name, which is wrong whenever the deployment sets
     * its own length. 0 leaves the field out rather than reporting a length the server does not
     * know. */
    int64_t     max_model_len = 0;
    /* What the server loaded (ServerOptions::model_path), vLLM's `root`. Empty leaves it out. */
    std::string root;
    /* What the model can be asked for -- tools, reasoning, images -- as /server_info reports it.
     * There is no field for it in OpenAI's model object, so without this a client learns it by
     * sending a request and reading the refusal; the entry it already fetches can say so up
     * front. A JSON object as text, spliced in as it is; empty leaves it out. */
    std::string capabilities;
};
/* The list, for GET /v1/models. */
std::string models_body(const ModelCard& m);
/* The entry itself, for GET /v1/models/:id: OpenAI's retrieve answers the model object, not a
 * list of one, and its SDK reads `id` off the top level. */
std::string model_body(const ModelCard& m);

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
 *
 * THE WIRE SHAPE IS THE SUBCLASS'S. Reading the reply -- the reader, the message it adds up to,
 * the finish it implies -- is the same for every API this server speaks, so it lives here once;
 * what one increment looks like on the wire is OpenAI's chunk (ChatDeltaStream) or Anthropic's
 * event (AnthropicStream).
 */
class ReplyStream {
public:
    ReplyStream(const OaiRequest& r, int index, const IReplyParser* parser);
    virtual ~ReplyStream() = default;

    /* What opens the stream: OpenAI's `role: assistant` chunk, Anthropic's message_start. */
    virtual std::string first_chunk() = 0;

    /* Feed the text generated since the last push; returns the bytes for what it completed.
     * `is_final` says the generation is over, which resolves whatever the reader was holding. */
    std::string push(std::string_view appended, bool is_final);

    /* What closes the choice: the finish reason, and this choice's `timings` when they are valid
     * (llama.cpp sends them on the final chunk). `u` is the request's usage, for the shapes that
     * close a message with it. */
    virtual std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) = 0;
    /* What closes a choice the engine failed, in place of final_chunk: the shape's own failure,
     * so a client reading to the terminal event never takes a truncated reply as complete.
     * OpenAI's chunks have none, so theirs is the final chunk with finish_reason "error" and the
     * error object after it. */
    virtual std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u);

    /* What the whole generation parsed to -- the non-streaming shape. Complete once push has
     * been called with is_final. */
    const ParsedMessage& parsed() const { return last_; }
    Finish adjusted_finish(Finish f) const;

    /* llama.cpp's `timings_per_token`, for the shapes that carry timings. */
    virtual void set_timings(const TimingsOut&) {}
    /* Logprobs entries (logprobs_content's elements) for the next chunk this stream writes, for
     * the shapes that carry them; the others drop them. */
    virtual void add_logprobs(const std::string&) {}

protected:
    /* The bytes for the increments one push completed. Each emitter folds every increment into
     * `last_` through absorb() first. */
    virtual std::string emit(const std::vector<ReplyDelta>& ev) = 0;
    bool absorb(const ReplyDelta& d);
    static void quote(std::string& out, std::string_view s);

    std::unique_ptr<IReplyReader> reader_;
    std::string id_, model_, fingerprint_;
    int64_t     created_;
    int         index_;
    bool        finished_ = false;

    ParsedMessage            last_;
    std::vector<ReplyDelta>  ev_;
};

/* ANTHROPIC'S MESSAGES API, translated into the chat request it means (`*chat`, a /v1/chat/completions
 * body) so that every refusal and default is the chat endpoint's. `*label` is the conversation's
 * label if the request carries one (metadata.user_id). Claude Code's per-request billing stamp in
 * the system prompt is held constant, as llama-server holds it, or no turn of a conversation could
 * reuse the one before it. */
int anthropic_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* label, ApiError* err);
/* The non-streaming answer, and an error, in Anthropic's shapes. */
std::string anthropic_message_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u);
std::string anthropic_error_body(const ApiError& e);
/* OPENAI'S RESPONSES API, translated into the chat request it means, as Anthropic's is. `*echo` is
 * the request's fields the response object repeats back. previous_response_id needs responses
 * stored server-side and is refused; tool types other than function have no chat equivalent and
 * are left out, as llama-server leaves them out. */
int responses_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* echo, ApiError* err);
std::string responses_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u);

/* Anthropic's stop_reason for a finish and the stop string that caused it. */
const char* anthropic_stop_reason(const OaiRequest& r, Finish f, const std::string& stop_word);

/* OpenAI's chat.completion.chunk. */
class ChatDeltaStream final : public ReplyStream {
public:
    ChatDeltaStream(const OaiRequest& r, int index, const IReplyParser* parser)
        : ReplyStream(r, index, parser), legacy_(r.legacy_functions) {}
    std::string first_chunk() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;

    /* llama.cpp's `timings_per_token`: the timings the next pushes' chunks carry, beside
     * `choices` as the final chunk carries them. Invalid timings carry nothing. */
    void set_timings(const TimingsOut& t) override;
    /* OpenAI's `logprobs.content` on the next chunk out: entries wait for one, because a push may
     * complete no delta (a tool call's opening, a held-back byte) and the tokens still count. */
    void add_logprobs(const std::string& elems) override;

private:
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    std::string chunk(json delta, const char* finish_reason, const TimingsOut* t = nullptr);
    std::string lp_;       /* pending logprobs entries, comma-joined */

    /* A delta's chunk is written, not built: the frame around the delta object is the same in
     * every one, so it is serialised once, and the delta between is written with
     * json_quote_append. The bytes are the ones chunk() writes for the same delta. */
    void delta_open(std::string& out);
    void delta_close(std::string& out);
    std::string head_;     /* `data: {...,"choices":[{"index":N,"delta":`, built on first use */
    std::string timings_;  /* `"timings":{...}` for every delta chunk, or "" */
    bool        legacy_;   /* OaiRequest::legacy_functions */
};

/* ANTHROPIC'S EVENTS: message_start, then a content block a kind -- thinking, text, each tool_use
 * -- opened, filled by deltas and stopped, then message_delta with the stop reason and the usage,
 * and message_stop. The prefill a continued conversation ends in is not repeated (`continued`). */
class AnthropicStream final : public ReplyStream {
public:
    AnthropicStream(const OaiRequest& r, int index, const IReplyParser* parser);
    std::string first_chunk() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;
    /* Anthropic's `error` event, which ends the stream with no message_stop. */
    std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) override;
    /* The stop string that cut the reply, for stop_reason and stop_sequence. */
    void set_stop_word(const std::string& w) { stop_word_ = w; }

private:
    enum class Block { None, Thinking, Text, Tool };
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    void open(std::string& out, Block b, const ToolCall* tc);
    void close(std::string& out);
    static void event(std::string& out, const char* type, const json& data);

    const OaiRequest* r_;
    int64_t     n_input_ = 0;
    std::string skip_;           /* the prefill still to be passed over */
    std::string stop_word_;
    Block       block_ = Block::None;
    int         n_blocks_ = 0;   /* blocks opened so far: the next one's index */
};

/* THE RESPONSES API'S EVENTS: response.created and .in_progress; an output item a kind --
 * reasoning, the message, each function call -- added, its content part added, filled by deltas,
 * and done; then response.completed (or .incomplete at the token limit) with the whole response.
 * Every event carries its output_index, content_index and sequence_number, which the OpenAI SDK's
 * stream accumulator places deltas by. */
class ResponsesStream final : public ReplyStream {
public:
    ResponsesStream(const OaiRequest& r, int index, const IReplyParser* parser);
    std::string first_chunk() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;
    /* response.failed, with the items finished so far and the error. */
    std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) override;

private:
    enum class Item { None, Reasoning, Message, Call };
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    void open(std::string& out, Item k, size_t call);
    void close(std::string& out);
    void event(std::string& out, const char* type, json data);
    json response(const char* status) const;

    const OaiRequest* r_;
    Item        item_ = Item::None;
    std::string item_id_, text_;  /* the open item's id and its text so far */
    size_t      call_ = 0;            /* the open call's index into last_.tool_calls */
    std::vector<std::string> items_;  /* every closed item, serialised, in output order */
    int64_t     seq_ = 0;
    UsageOut    usage_;
    const char* incomplete_ = nullptr;
};

/* The SSE terminator. Written once at the end of every stream, including a failed one, because a
 * client blocked on the reader has no other way to learn the stream is over. */
extern const char* const SSE_DONE;

std::string sse_event(const std::string& json_text);
std::string sse_error(const ApiError& e);

/* ================================================================== small shared helpers */

std::string base64_encode(const void* data, size_t len);
bool        base64_decode(const std::string& in, std::vector<uint8_t>* out);
/* Shared by chat image parts and System One images. Decode data URLs only; never fetch a URL. */
int decode_media_url(const std::string& url, const std::string& field,
                     std::vector<uint8_t>* out, ApiError* err);
/* A token id as a request spells it: an integer in [0, INT32_MAX]. False for anything else, and
 * the caller refuses the request -- narrowing instead would make 2^32 + 5 an alias of token 5. */
bool        token_id(const json& v, int32_t* out);

/* Parse a request body into a tree of at most json_value_budget(max_ctx) values, nested at most
 * kJsonMaxDepth deep. A syntax error comes back discarded, as json::parse's would; a body with
 * more values than the budget sets `*over`, and one nested deeper than the bound sets `*deep`,
 * and either comes back holding only what fitted, which the caller refuses. See the definition
 * for why the tree is bounded rather than the body alone. */
json        parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep);
/* The body of a generation endpoint: parse_json_bounded, and a 400 saying which bound it broke or
 * that it is not one JSON object. */
int         parse_request_object(const std::string& body, const OaiLimits& lim, json* b,
                                 ApiError* err);
int64_t     json_value_budget(int64_t max_ctx);
constexpr int kJsonMaxDepth = 1024;

std::string random_id(const char* prefix);
int64_t     unix_now();

}  /* namespace server */
}  /* namespace rad */
