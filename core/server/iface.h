/* core/server/iface.h -- everything the server needs from components it does not own.
 *
 * The server is written against pure-virtual interfaces rather than against core/sched and
 * core/text directly, for two reasons worth stating. The first is that the HTTP layer then carries
 * no header dependency on those components and can be compiled without them. The second is that a
 * test can drive every endpoint against a fake scheduler in-process, which is the only way to
 * check cancellation and the streaming wire format without binding a port and loading a model.
 *
 * The cost is one virtual call per operation. That is paid on the admission path and on the HTTP
 * thread, never inside rad_arch_step and never per token on the step path, so it is free where it
 * would matter. The concrete types (rad::Scheduler, rad::Tokenizer, ...) are bound to these by
 * the adapters in adapters.h: one line each at the wiring site.
 */
#pragma once
#include "rad_core.h"

#include <nlohmann/json_fwd.hpp>

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad {
namespace server {

/* ordered_json, not json, and the reason is not cosmetic: a tool's parameter schema is rendered
 * into the prompt in the order the caller wrote it, and property order in a JSON schema changes
 * what the model emits. An unordered map would silently re-sort every caller's tool definitions. */
using json = nlohmann::ordered_json;

/* ================================================================== the scheduler
 *
 * ASSUMED of core/sched. `submit` takes the request by value and returns the id it assigned;
 * `cancel` is idempotent and must free the request's KV blocks immediately rather than at the
 * next completion (spec §14), because an agent client abandoning a long generation is the normal
 * case, not an edge case.
 */
struct SchedMetrics {
    int64_t waiting = 0;
    int64_t running = 0;
    int64_t preempted = 0;
    double  kv_util = 0.0;            /* 0..1 of the paged KV pool */
    /* The paged pools in TOKENS. A percentage answers "how close to preemption"; tokens answer
     * "how much context fits", and an operator sizing a deployment is asking the second. */
    int64_t kv_tokens_used = 0;
    int64_t kv_tokens_total = 0;
    /* AND HOW MUCH OF `used` NO RUNNING REQUEST IS READING, which is the difference between a
     * cache doing its job and a leak. A finished turn's blocks stay held so the next turn can
     * reuse them, so on a server carrying conversations this is most of the pool -- and with one
     * number for the whole of it there is no way to tell that from a pool that never releases.
     * It is the cache's UNSHARED holding: a block a running request is also reading belongs to
     * that request in this split, because evicting it would free nothing. */
    int64_t kv_tokens_cached = 0;
    /* AND THE PAGED AND STATEFUL POOLS TOGETHER, IN BYTES: how much of the card the KV cache is
     * holding, which is the question an operator watching VRAM is actually asking and the one
     * neither figure above can answer. Tokens cannot be summed across groups -- two paged groups
     * share one token axis and a recurrent group has none -- so the totals above are one group's,
     * the binding one. Bytes are disjoint allocations and add up, so these are every group.
     *
     * USED IS WHAT IS HELD, not what a live request is reading. A finished turn's blocks stay held
     * by the prefix cache so the next turn can reuse them, and they are occupying VRAM for exactly
     * as long; they stop counting when they reach the free list, which is where an eviction, a
     * cancellation and an idle-tier demotion all put them. */
    int64_t kv_bytes_used = 0;
    int64_t kv_bytes_total = 0;
    /* What the pool was carved to address, which an elastic pool may not be holding. */
    int64_t kv_bytes_carved = 0;
    /* THE MOST CONTEXT THE CACHE CAN HOLD: every block of its carve, the ones lent to the expert
     * slab taken back. `kv_bytes_carved` is the same figure in bytes. The paged groups share one
     * token axis, so it is the smallest of theirs; a windowed group is left out, because its carve
     * bounds how far back a sequence keeps context rather than how much fits. */
    int64_t kv_tokens_max = 0;

    double  prefix_hit_rate = 0.0;    /* 0..1 */
    /* Cumulative prefix-cache evictions. `prefix_hit_rate` falls only AFTER a prefix that
     * would have been reused is gone; this rises as it happens, which is the difference
     * between an alert and a post-mortem. */
    int64_t prefix_evictions = 0;
    int64_t prefix_ckpt_evictions = 0;
    /* PREFIX CACHING, DECOUPLED (spec §7.3). `prefix_hit_rate` above is the ATTENTION one, because
     * that is what vLLM's single number has always meant. The linear side hits at a different
     * granularity -- whole checkpoints, not blocks -- and averaging the two would hide exactly the
     * effect the decoupling exists to expose. So both are carried, and the token counts beside
     * them, and the checkpoints actually written. Without all of it a deployment cannot tell
     * whether linear checkpointing is working at all. */
    double  linear_hit_rate = 0.0;
    int64_t attn_cached_tokens = 0;
    int64_t linear_cached_tokens = 0;
    int64_t checkpoints_written = 0;
    double  prefill_tps = 0.0;
    double  decode_tps = 0.0;
    /* AND THE EXACT COUNTS BEHIND THEM. The two rates above are EWMAs, which is the right thing
     * for a dashboard and the wrong thing for anything that has to compare two runs: a gauge
     * describes whatever mix the engine happened to be in when it was read. These cumulative
     * totals are the comparable form, and the only one: `vllm:generation_tokens_total` is charged
     * when a request FINISHES and is therefore identically zero over any steady-state window. */
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    double  draft_acceptance = 0.0;   /* 0..1, spec §10 */
    /* The exact pair the ratio is computed from, for the same reason. */
    int64_t draft_tokens = 0;
    int64_t draft_accepted = 0;
    /* Prefix hits ended early for a windowed block drafter, and the tokens they gave back. */
    int64_t draft_refills = 0;
    int64_t draft_refill_tokens = 0;
    /* THE PER-POSITION DRAFT HAZARD. `draft_acceptance` above is pooled over the whole window and
     * cannot distinguish a drafter that decays with depth from one whose later rounds are broken:
     * both read as one small number. These carry the conditional rate at each position -- of the
     * verifies that reached j because every earlier position was accepted, the fraction that took
     * j as well -- and the reach counts that say how much evidence each one has. */
    static const int MAX_DRAFT_POS = 64;
    int     draft_depth_seen = 0;
    int64_t draft_pos_reached [MAX_DRAFT_POS] = {};
    int64_t draft_pos_accepted[MAX_DRAFT_POS] = {};
    /* THE ADAPTIVE DRAFT WINDOW (RADIANCE_DRAFT_ADAPT), all zero while it is off: the mean window
     * of the running sequences, and the verifies by the window they carried (index = drafts). */
    bool    draft_adaptive = false;
    double  draft_depth_adaptive = 0.0;
    int64_t draft_verify_depth[MAX_DRAFT_POS + 1] = {};
    int64_t preemptions = 0;
    int64_t total_requests = 0;
    /* ENGINE STEPS. One forward pass of the model plus whatever draft rounds followed it, which
     * with speculation is more than one token -- so tokens a second cannot answer "how long is a
     * step" and the two questions have to be asked separately. */
    int64_t steps = 0;
    /* STEPS BY SHAPE. A step carrying a prefill chunk cannot roll back a rejected draft, so
     * nothing in it speculates, decode rows included. These say how often the load produces that
     * shape and how many decode rows paid for it. */
    int64_t steps_prefill = 0;
    int64_t steps_mixed = 0;
    int64_t steps_decode = 0;
    int64_t steps_ahead = 0;          /* issued behind the step before, ahead of its commit */
    int64_t decode_rows_unspeculated = 0;
    /* WHAT THE STEPS COST: cumulative nanoseconds spent inside a step, and the decode half of it
     * separately. A step time is not wall clock over a step count -- the engine idles between an
     * agent's turns and a prefill chunk costs an order of magnitude more than a decode step, and
     * a gauge that divides charges both to the decode step. `step_ns_decode / steps_decode` is
     * what a decode step cost, whatever the load was doing around it. */
    int64_t step_ns = 0;
    int64_t step_ns_decode = 0;
};


/* WHAT ONE REQUEST IS DOING RIGHT NOW. The aggregate metrics answer "is the engine busy"; they
 * cannot answer "which of these eight is stuck", and a request sitting in Waiting because its
 * prompt will never fit is invisible in every one of them. */

/* ONE STORED CONVERSATION. Not a request: a request is in flight and is gone when it answers,
 * while this is what the engine is still HOLDING on a session's behalf between turns -- which is
 * the thing that decides whether the next turn costs a copy or a re-prefill.
 *
 * THE SIZES OVERLAP AND THE TOTAL IS NOT A SUM. Sessions that share a system prompt share the
 * blocks for it, which is the entire point of a prefix cache; a view that hid that would make a
 * server look like it was storing several times what it holds. Anything rendering these says so.
 *
 * A SESSION IS ROUTINELY SPLIT ACROSS TIERS, because a maintenance pass is bounded and moves a
 * long conversation over several of them. So "where is it" is three counts and not a label. */
struct SessionStat {
    uint64_t    id = 0;
    std::string model;            /* the container it was built against */
    int64_t     n_tokens = 0;
    int64_t     bytes = 0;
    double      idle_s = 0;       /* since its last turn */
    double      age_s = 0;        /* since it was first seen */
    int64_t     on_device = 0, on_host = 0, on_disk = 0;   /* blocks */
    /* LINEAR STATE, on a hybrid model, and counted apart from blocks because the two are not
     * interchangeable. A session showing every block on disk and no snapshot anywhere will come
     * back and re-run every recurrent layer over its whole transcript -- which the hit rate will
     * report as a hit. Zero everywhere on a model that has no recurrent state. */
    int64_t     snapshots = 0;
    int64_t     snap_host = 0, snap_disk = 0;    /* in the store; only on disk */
    int64_t     snap_bytes = 0;
    /* The prefix a return skips: the deepest snapshot's position (IdleTiers::SessionInfo). */
    int64_t     resume_tokens = 0;
    bool        moving = false;   /* a transfer is outstanding */
};

/* WHAT THE STORED CONVERSATIONS OCCUPY OFF THE CARD, blocks and snapshots each counted once --
 * the one total the per-session rows cannot be summed into. Caps are 0 for a tier that is off. */
struct SessionOccupancy {
    int64_t host_bytes = 0, disk_bytes = 0;
    int64_t host_cap_bytes = 0, disk_cap_bytes = 0;
};

struct RequestStat {
    uint64_t    id = 0;
    std::string state;             /* waiting | prefill | decode | preempted | done */
    int64_t     prompt_tokens = 0;
    int64_t     computed_tokens = 0;
    int64_t     cached_tokens = 0;
    int64_t     output_tokens = 0;
    int64_t     max_tokens = 0;
    int64_t     ctx_tokens = 0;
    int64_t     kv_blocks = 0;    /* summed over the paged groups: they are disjoint allocations */
    /* The context it is holding, on the token axis the paged groups share -- NOT their sum. See
     * the note at the definition: the two figures are in different units on purpose. */
    int64_t     kv_tokens = 0;
    int32_t     n_draft = 0;
    int32_t     slot = -1;
    double      age_s = 0;
    double      ttft_s = -1;       /* negative until the first token */
    double      decode_tps = 0;
};

class IScheduler {
public:
    virtual ~IScheduler() = default;
    virtual uint64_t     submit(Request&& r) = 0;
    /* THE SEQUENCES OF ONE API REQUEST, all admitted before a step can see any of them: a prompt
     * array or `n` > 1 submitted one call at a time can have a step begin between two calls, so
     * the batch they decode in -- and its tie-level rounding -- would depend on thread timing.
     * ids[i] is what submit() would have returned for rs[i]. This default is that loop, for a
     * scheduler whose steps cannot interleave with submission. */
    virtual void submit_all(std::vector<Request>& rs, std::vector<uint64_t>* ids) {
        ids->clear();
        for (Request& r : rs) ids->push_back(submit(std::move(r)));
    }
    virtual void         cancel(uint64_t id) = 0;
    virtual SchedMetrics metrics() const = 0;

    /* "I am finished with this request's sink; you may drop the Request." core/sched keeps a
     * terminal request alive precisely because the HTTP thread still holds its sink, so somebody
     * has to say when that stops being true, and the HTTP thread is the only one who knows.
     * Called at most once per submitted id, after the response has been written. Default no-op,
     * for a scheduler that frees at completion. */
    virtual void reap(uint64_t /*id*/) {}

    /* CAN THIS REQUEST'S SAMPLING PARAMETERS BE SERVED AT ALL?
     *
     * A sampler op that does not resolve is deliberately not a declare failure -- a deployment
     * that never sends `typical_p` does not need a typical kernel (core/sample/sampler.h). It
     * becomes a failure when a request asks for it, and Sampler::check_request exists to say so
     * by name. This is what the step loop relies on when it treats an unresolved stage as
     * unreachable: without the check here, such a request is admitted, runs, and is cancelled
     * mid-step with the reason in the log and a dead stream at the client.
     *
     * Asked once per submission, on the HTTP thread, before the first Request is built. Returns
     * RAD_OK, or a negative status with `why` set. The default admits everything so a test's fake
     * scheduler needs no change; the real answer comes from SchedulerBridge. */
    virtual int admit(const SamplingParams& /*sp*/, std::string* /*why*/) const { return RAD_OK; }

    /* DROP EVERY PREFIX-CACHE ENTRY, returning how many were dropped -- or a negative status when
     * this build has no prefix cache, which is a 501 rather than a lie about having cleared one.
     *
     * vLLM has this endpoint and it exists for the same reason: the second run of a benchmark is
     * not measuring what the first one measured, and an operator who cannot say "start cold"
     * cannot compare two numbers. Releasing the cache's references is safe against a running step
     * -- a block a live sequence holds stays allocated -- but it still takes the scheduler's lock,
     * because the alternative is an HTTP thread walking the map a step is inserting into. */
    virtual int64_t reset_prefix_cache() { return RAD_E_UNSUPPORTED; }

    /* The live request table. Fills `out` and returns how many, oldest first; the default answers
     * "this scheduler does not keep one", which is an empty table and not a wrong one. */
    virtual int requests(std::vector<RequestStat>* /*out*/) const { return 0; }

    /* The stored conversations: what the engine is still holding between turns, and where. The
     * default answers "this build has no idle tiers", which is an empty table and not a wrong
     * one -- with the tiers off, everything the cache holds is in VRAM by definition. */
    virtual int sessions(std::vector<SessionStat>* /*out*/, SessionOccupancy* /*occ*/) const {
        return 0;
    }

    /* WHAT THE TIERS HAVE ACTUALLY DONE. Without it the session table looks the same whether the
     * tiers are working, switched off, or trying and failing every pass -- three very different
     * situations that all render as "everything is in VRAM". */
    struct TierCounters {
        /* `to_host` and `to_disk` count COPIES; `dropped` is VRAM given up after one and
         * `host_freed` host slots given up after the disk copy. A conversation that returns before
         * anything wants its VRAM is copied and never dropped. `host_drops` is entries lost with
         * no disk tier to hold them. */
        int64_t to_host = 0, dropped = 0, to_disk = 0, promoted = 0;
        int64_t host_freed = 0, host_drops = 0, failures = 0;
        /* Entries and snapshots read off disk into host memory ahead of the restore that wanted
         * them; `promoted` counts the restore itself. */
        int64_t fetched = 0, ck_fetched = 0;
        int64_t host_bytes = 0, disk_bytes = 0;
        /* THE LINEAR HALF, MOVED. Separate counters because a hybrid server can move every block
         * it has and no snapshots at all, and no block counter can tell that apart from tiering
         * working. `ck_drops` is snapshots whose only copy was lost. */
        int64_t ck_to_disk = 0, ck_drops = 0;
        int64_t ck_bytes = 0;      /* one snapshot; 0 on a model with no recurrent state */
        int64_t ck_host_slots = 0, ck_disk_slots = 0;   /* the store's; the disk store's */
        int64_t restores = 0, restore_ns = 0, flushes = 0, flush_ns = 0;   /* IdleTiers::Stats */
        /* WHETHER ANYTHING CAN MOVE AT ALL, which no counter above can answer: a tier that is on
         * and has had nothing to do reads the same as one that was never given a size, and the
         * two call for opposite responses from whoever is looking. */
        bool host_on = false, disk_on = false;
    };
    virtual TierCounters tier_counters() const { return TierCounters{}; }
};

/* ================================================================== text frontend
 *
 * ASSUMED of core/text. Detokenisation is incremental and streaming-safe (spec §12): byte-level
 * BPE emits partial UTF-8 sequences, so `push` returns "" until the bytes resolve into a
 * character, and `flush` drains whatever is left at the end of a generation.
 */
class IDetokenizer {
public:
    virtual ~IDetokenizer() = default;
    virtual std::string push(int32_t token) = 0;
    virtual std::string flush() = 0;
    /* True once the decoder itself swallowed a stop string. core/text's Detokenizer can be built
     * with the request's stop strings and then never emits one; without this the server would
     * search text that can no longer contain what it is searching for, and the generation would
     * run to max_tokens. Default false, for a decoder that leaves stop handling to us. */
    virtual bool stopped() const { return false; }
    /* Match the stop strings only after `marker` has been generated (core/text Detokenizer::
     * defer_stops_until): the reply starts inside its reasoning block. */
    virtual void defer_stops_until(const std::string& /*marker*/) {}
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    /* `parse_special` decides whether a control token spelled out in the text is that token or
     * its characters, and there is no default because a default is how the wrong one gets picked
     * silently. Text the server is sent as a whole prompt is parsed: a rendered chat template
     * carries the control tokens that frame its turns, and a raw /v1/completions prompt is the
     * client's own rendering of the same, as vLLM and llama.cpp read it. Text that must reach the
     * model as characters is encoded with false. */
    virtual std::vector<int32_t> encode(const std::string& text, bool add_special,
                                        bool parse_special) const = 0;
    virtual std::string          decode(const std::vector<int32_t>& toks) const = 0;
    virtual int32_t              eos() const = 0;
    virtual int32_t              bos() const = 0;

    /* WHAT THIS VOCABULARY IS, for /tokenize and /get_tokenizer_info. Off every hot path: asked
     * once per request to those two endpoints and never during generation.
     *
     * A struct rather than eight accessors because the set will grow, and a client that reads
     * this to decide how to build a prompt would otherwise be reading it across eight virtual
     * calls that can disagree about which vocabulary they describe. The default is "I do not
     * know", which is what a fake tokenizer in a test honestly is. */
    struct VocabInfo {
        int32_t     n_tokens = 0;
        int32_t     bos = -1, eos = -1, eot = -1, unk = -1, pad = -1;
        bool        add_bos = false, add_eos = false;
        std::string kind;            /* "bpe", "spm", "wpm", ... -- how it splits */
    };
    virtual VocabInfo vocab_info() const { return VocabInfo{}; }

    /* The raw piece an id stands for, NOT its decoded text: /tokenize's `return_token_strs` shows
     * what the tokeniser actually chose, and decoding each id one at a time would render a
     * byte-fallback token as a replacement character and lose exactly the detail the caller
     * turned the flag on to see. "" when the id is out of range. */
    virtual std::string piece(int32_t /*id*/) const { return std::string(); }

    /* One incremental decoder per in-flight generation. It is made by the tokenizer rather than
     * default-constructed by the caller, because it needs the vocabulary to decode anything.
     * `stops` is passed down because core/text's Detokenizer can suppress a stop string that
     * straddles a token boundary, which is strictly better than the server finding it after the
     * fact -- and IDetokenizer::stopped() is how it tells us it did.
     *
     * `preserved` is the control tokens this generation must decode as TEXT. A tool-calling
     * format can be built out of control tokens -- MiniCPM5 writes `<function`, `<param` and
     * their closers that way -- and a decoder that suppresses them the way control tokens are
     * normally suppressed hands the reply parser the argument values with the structure gone.
     * The chat render says which ones; empty everywhere else. */
    virtual std::unique_ptr<IDetokenizer> detokenizer(
        const std::vector<std::string>& stops,
        const std::vector<std::string>& preserved = {}) const = 0;
};

class IReplyParser;

/* What a chat template needs told, beyond the messages themselves.
 *
 * `template_kwargs` VALUES ARE JSON TEXT, not display text: `"false"`, `"\"high\""`, `"3"`. That
 * is the shape the request carries them in and the shape the template context wants them in, and
 * converting in between would mean this layer deciding that the string "false" is a boolean --
 * which is exactly the guess that makes `enable_thinking: "false"` silently enable thinking. */
struct ChatRenderOptions {
    bool        add_generation_prompt = true;
    bool        enable_thinking = true;
    bool        parallel_tool_calls = true;
    std::string tool_choice = "auto";        /* auto | none | required */
    std::string grammar;                     /* the caller's GBNF, or "" */
    std::string json_schema;                 /* a response_format schema, as JSON text, or "" */
    std::string reasoning_format = "auto";
    std::map<std::string, std::string> template_kwargs;
};

/* What it gives back. A STRING IS NOT ENOUGH, and that is the whole reason this struct exists: a
 * tool-calling template emits a prompt AND the grammar that makes the model's call parseable AND
 * the triggers that keep that grammar off the prose AND the stop strings the format ends on. A
 * seam that returns only the prompt drops the other three silently, which is a chat endpoint that
 * accepts `tools` and then produces calls no client can read. */
struct ChatRender {
    std::string prompt;

    std::string grammar;                     /* the template's, merged with the caller's */
    bool        grammar_lazy = false;
    std::vector<SamplingParams::GrammarTrigger> grammar_triggers;
    std::vector<std::string> additional_stops;

    /* The assistant prefix already inside `prompt`. The grammar describes the whole turn, so it
     * has to be advanced over this before it constrains the first generated token. */
    std::string generation_prompt;

    /* The reply parser this render implies: the model's reply format bound to THESE tools and
     * THIS generation prompt, so it cannot be a server-wide singleton. It reads calls only when
     * the request sent tools. Null when nothing needs parsing. */
    std::shared_ptr<IReplyParser> parser;

    /* The tokens that parser must be able to SEE. See ITokenizer::detokenizer. */
    std::vector<std::string> preserved_tokens;

    /* THE REASONING BLOCK THIS RENDER LEAVES THE REPLY FACING, as the parser reads it: the
     * markers with their optional whitespace trimmed, and whether the reply starts inside the
     * block. Empty `think_close` is a model with no reasoning block. The server turns these into
     * token ids (ReasoningSpan) to count the block and to hold a budget on it. */
    std::string think_open, think_close;
    bool        think_starts_open = false;
};

class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    /* Render. Returns RAD_OK, or a negative status with `why` naming what the template refused --
     * which becomes the 400's message, so it has to read like something a caller can act on. */
    virtual int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
                      ChatRender* out, std::string* why) const = 0;

    virtual bool supports_tools() const = 0;
    virtual bool supports_reasoning() const = 0;

    /* WHY THIS MODEL'S TOOL CALLS CANNOT BE READ, or "" when they can. A template may declare tool
     * support in a format the reply parser does not describe; such a model still chats, and a
     * request that sends tools is refused with this sentence rather than answered with calls
     * nobody can read back. */
    virtual std::string tool_refusal() const { return std::string(); }
};

/* A reply taken apart: what core/text's reply parser gives back once the reply is complete. */
struct ToolCall {
    std::string id;          /* "" from the parser; the server assigns one */
    std::string name;
    std::string arguments;   /* JSON text; unterminated when the reply stopped inside the call */
};

struct ParsedMessage {
    std::string           content;
    std::string           reasoning;    /* "" unless the template supports reasoning */
    std::vector<ToolCall> tool_calls;
    /* The reply ended inside its last call: that call never closed, and is no call to run. */
    bool                  open_call = false;
};

/* One increment of a reply that is being read as it is generated. What has been delivered is
 * never revised: the concatenated increments of each kind ARE the finished values, which is what
 * lets the streaming endpoint forward each one as it comes. */
struct ReplyDelta {
    enum Kind : uint8_t {
        Reasoning,   /* text: a reasoning delta */
        Content,     /* text: a content delta */
        CallBegin,   /* text: the function name; `call` is the new call's index */
        CallArgs,    /* text: an arguments delta for call `call` */
        CallEnd,     /* call `call` is closed */
    };
    Kind        kind = Content;
    uint32_t    call = 0;
    std::string text;
};

/* One choice's reply, read once, in order, as it is generated. feed() appends generated text
 * and finish() says there is no more; each appends the increments it completes. The cost is
 * linear in the reply however it is cut up -- the text is never read from the top again. */
class IReplyReader {
public:
    virtual ~IReplyReader() = default;
    virtual void feed(std::string_view text, std::vector<ReplyDelta>* out) = 0;
    virtual void finish(std::vector<ReplyDelta>* out) = 0;
    /* The whole reply, once finish() has run. */
    virtual ParsedMessage message() const = 0;
    /* Once finish() has run: the content with an open last call (ParsedMessage::open_call) read
     * as the text it was written as -- the content before the call, then the call's own bytes
     * from `reply`, the whole text fed. The content alone when no call is open. */
    virtual std::string content_with_open_call(std::string_view reply) const = 0;
};

/* A request's reply format. Each choice opens its own reader, and a request that sent no tools
 * gets a parser that reads no calls: call syntax in its reply is text to that request. */
class IReplyParser {
public:
    virtual ~IReplyParser() = default;
    virtual std::unique_ptr<IReplyReader> open() const = 0;
};

/* JSON-schema -> GBNF (spec §13). Errors are values: RAD_OK, or negative with `err` naming the
 * part of the schema that could not be compiled. */
class IGrammarCompiler {
public:
    virtual ~IGrammarCompiler() = default;
    virtual int from_json_schema(const json& schema, std::string& out, std::string& err) const = 0;

    /* Compile this GBNF, here, on the admitting thread, and hand back the program for the request
     * to carry (SamplingParams::grammar_program). Compiling is where a grammar is found not to
     * run -- left recursion, a construct the mask engine cannot serve, a bound it outgrows -- and
     * where its mask plan is built, which is host work in the tens to hundreds of milliseconds.
     * The alternative is doing both on the scheduler thread, where the only thing left to do about
     * a failure is kill the request three tokens into its answer, and every other request's step
     * waits for the build. A grammar that cannot be compiled is a 400 that names why. */
    virtual int compile(const std::string& gbnf, std::shared_ptr<const GbnfProgram>* out,
                        std::string& err) const = 0;
};

/* ================================================================== optional backends */

/* /v1/embeddings. Absent, the endpoint is 501 naming that this model was not loaded with an
 * embedding head -- not an empty 200. */
class IEmbedder {
public:
    virtual ~IEmbedder() = default;
    virtual int     embed(const std::vector<int32_t>& tokens, std::vector<float>& out) = 0;
    virtual int64_t dim() const = 0;
};

/* Multimodal preprocessing (spec §11). The server pulls the bytes out of a content part; the
 * processor behind this decodes them, resizes and cuts them for the encoder, and says which
 * tokens take the part's place in the prompt -- placeholder runs, the markers around them and,
 * for a video, the timestamps between its frames. */
enum class MediaKind { Image, Video, Audio };

/* The placeholder a chat template renders where a media part sat. Both the template and the
 * server have to agree on the spelling or the prompt silently loses its images, so it is declared
 * once, here, rather than twice. llama.cpp's mtmd uses the same string, which is not an accident:
 * the templates are the same templates. */
#define RAD_MEDIA_MARKER "<__media__>"

class IMultimodal {
public:
    virtual ~IMultimodal() = default;
    virtual bool accepts(MediaKind k) const = 0;
    /* Decode and process one part. `bytes` is the decoded file, not a URL: the server never
     * fetches a remote URL on a request's behalf. On success `*out` carries the patches and the
     * token run that replaces the marker; on failure `why` says what is wrong with the media. */
    virtual int prepare(MediaKind k, const std::vector<uint8_t>& bytes,
                        std::shared_ptr<mm::Item>* out, std::string* why) = 0;
};


}  /* namespace server */
}  /* namespace rad */
