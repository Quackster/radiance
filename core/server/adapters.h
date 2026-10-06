/* core/server/adapters.h -- bind the real components onto the server's interfaces.
 *
 * iface.h deliberately knows nothing about core/sched or core/text, so that the HTTP layer can be
 * compiled and tested against fakes with no engine present. This file is where the two sides
 * actually meet, and it is included by exactly one translation unit: whoever wires the engine
 * (core/engine.cpp). Nothing in core/server/ includes it, so the server's own build stays
 * independent of those components' headers.
 *
 *   rad::server::SchedulerBridge  sched(&engine_scheduler);
 *   rad::server::TokenizerBridge  tok(vocab);
 *   rad::server::Deps d; d.sched = &sched; d.tok = &tok; ...
 *
 * Every impedance mismatch between what the server wants and what those components provide is
 * resolved here, in one place, with the reason written down -- rather than being smeared across
 * either component as a special case for the other.
 */
#pragma once
#include "iface.h"
#include "mem/kvtier.h"

#if !defined(RAD_SERVER_NO_ENGINE_BRIDGE) && \
    __has_include("sched/scheduler.h") && __has_include("text/tokenizer.h")
#define RAD_SERVER_HAVE_ENGINE_BRIDGE 1

#include "sample/gbnf.h"
#include "sample/grammar.h"
#include "sample/sampler.h"
#include "sched/scheduler.h"
#include "rad_builder.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "text/tokenizer.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <memory>

namespace rad {
namespace server {

/* ================================================================== scheduler */

class SchedulerBridge final : public IScheduler {
public:
    /* `sampler` is the rank-0 sampler, and it is optional only so that a caller with no engine
     * behind it still compiles. Every rank declares the same chain against the same hierarchy, so
     * whether a stage resolved is not a per-rank fact and asking one of them is asking all. */
    /* `tiers` is rank 0's IdleTiers, and it is given whether or not a tier can move anything:
     * the object holds the session table either way, and a server with no --prefix-cache-host-mib is
     * one whose every stored session is in VRAM rather than one with no sessions. Null only when
     * there is no engine behind this at all. */
    explicit SchedulerBridge(Scheduler* s, const Sampler* sampler = nullptr,
                             const IdleTiers* tiers = nullptr)
        : s_(s), sampler_(sampler), tiers_(tiers) {}

    int admit(const SamplingParams& sp, std::string* why) const override {
        if (!sampler_) return RAD_OK;
        return sampler_->check_request(sp, why);
    }

    /* Request ids are the server's to allocate: Scheduler::add refuses a duplicate id and does
     * not mint one, which is the right split -- the id has to exist before submission so that a
     * connection dropped during submission still has something to cancel. */
    uint64_t submit(Request&& r) override {
        auto p = std::make_unique<Request>(std::move(r));
        p->id = next_.fetch_add(1, std::memory_order_relaxed);
        const uint64_t id = p->id;
        int st = s_->add(std::move(p));
        if (st < 0) {
            /* Failing here means the request never entered the queue, so nothing will ever finish
             * its sink. Tell the sink itself, or the HTTP thread waits forever. */
            RAD_WARN("server: scheduler refused request %llu: %s",
                     (unsigned long long)id, rad_strerror(st));
            return 0;
        }
        return id;
    }

    void submit_all(std::vector<Request>& rs, std::vector<uint64_t>* ids) override {
        const size_t n = rs.size();
        std::vector<std::unique_ptr<Request>> ps(n);
        std::vector<int> rc(n, RAD_OK);
        ids->assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            ps[i] = std::make_unique<Request>(std::move(rs[i]));
            ps[i]->id = next_.fetch_add(1, std::memory_order_relaxed);
            (*ids)[i] = ps[i]->id;
        }
        s_->add_all(ps.data(), n, rc.data());
        for (size_t i = 0; i < n; ++i) {
            if (rc[i] >= 0) continue;
            RAD_WARN("server: scheduler refused request %llu: %s",
                     (unsigned long long)(*ids)[i], rad_strerror(rc[i]));
            (*ids)[i] = 0;
        }
    }

    void cancel(uint64_t id) override { s_->cancel(id); }
    void reap(uint64_t id) override { s_->reap(id); }

    int64_t reset_prefix_cache() override { return s_->reset_prefix_cache(); }

    /* THE STORED CONVERSATIONS. Idle time is computed here rather than carried, because the tiers
     * keep a monotonic timestamp and a browser wants "how long ago" -- converting at the edge
     * means one clock is authoritative and the other never has to be trusted. */
    TierCounters tier_counters() const override {
        TierCounters c;
        if (!tiers_) return c;
        const IdleTiers::Stats& s = tiers_->stats();
        c.to_host = s.to_host; c.dropped = s.dropped; c.to_disk = s.to_disk;
        c.promoted = s.promoted;
        c.host_freed = s.host_freed; c.host_drops = s.host_drops; c.failures = s.failures;
        c.fetched = s.fetched; c.ck_fetched = s.ck_fetched;
        c.host_bytes = s.host_bytes; c.disk_bytes = s.disk_bytes;
        c.ck_to_disk = s.ck_to_disk; c.ck_drops = s.ck_drops;
        c.restores = s.restores; c.restore_ns = s.restore_ns;
        c.flushes = s.flushes; c.flush_ns = s.flush_ns;
        c.ck_bytes = tiers_->ckpt_bytes();
        c.ck_host_slots = tiers_->ckpt_slots();
        c.ck_disk_slots = tiers_->ckpt_disk_slots();
        c.host_on = tiers_->host_on();
        c.disk_on = tiers_->disk_on();
        return c;
    }

    int sessions(std::vector<SessionStat>* out, SessionOccupancy* occ) const override {
        out->clear();
        if (!tiers_) return 0;
        IdleTiers::Occupancy held;
        const std::vector<IdleTiers::SessionInfo> v = tiers_->sessions(occ ? &held : nullptr);
        if (occ) {
            occ->host_bytes = held.host_bytes;         occ->disk_bytes = held.disk_bytes;
            occ->host_cap_bytes = held.host_cap_bytes; occ->disk_cap_bytes = held.disk_cap_bytes;
        }
        const uint64_t now = rad_mono_ns();
        out->reserve(v.size());
        for (const IdleTiers::SessionInfo& s : v) {
            SessionStat o;
            o.id        = s.id;
            o.model     = s.model;
            o.n_tokens  = s.n_tokens;
            o.bytes     = s.bytes;
            o.idle_s    = now > s.last_used_ns  ? (double)(now - s.last_used_ns)  / 1e9 : 0.0;
            o.age_s     = now > s.first_seen_ns ? (double)(now - s.first_seen_ns) / 1e9 : 0.0;
            o.on_device = s.on_device;
            o.on_host   = s.on_host;
            o.on_disk   = s.on_disk;
            o.moving    = s.moving;
            o.snapshots   = s.snapshots;
            o.snap_host   = s.snap_host;
            o.snap_disk   = s.snap_disk;
            o.snap_bytes  = s.snap_bytes;
            o.resume_tokens = s.resume_tokens;
            out->push_back(std::move(o));
        }
        return (int)out->size();
    }

    /* THE STATE NAME IS DERIVED, not stored: ReqState has no `Prefill`, because to the scheduler
     * prefill is not a state -- it is a running request whose prompt is not yet all in the KV.
     * That distinction is the single most useful thing this table shows, so it is made here
     * rather than left for every reader to reconstruct from two integers. */
    int requests(std::vector<RequestStat>* out) const override {
        std::vector<Scheduler::ReqStat> raw(256);
        const int n = s_->observed_requests(raw.data(), (int)raw.size());
        out->clear();
        out->reserve((size_t)n);
        for (int i = 0; i < n; ++i) {
            const Scheduler::ReqStat& r = raw[(size_t)i];
            RequestStat o;
            o.id = r.id;
            switch ((ReqState)r.state) {
                case ReqState::Waiting:   o.state = "waiting";   break;
                case ReqState::Preempted: o.state = "preempted"; break;
                case ReqState::Running:
                    o.state = r.computed_tokens < r.prompt_tokens ? "prefill" : "decode";
                    break;
                default:                  o.state = "done";      break;
            }
            o.prompt_tokens   = r.prompt_tokens;
            o.computed_tokens = r.computed_tokens;
            o.cached_tokens   = r.cached_tokens;
            o.output_tokens   = r.output_tokens;
            o.max_tokens      = r.max_tokens;
            o.ctx_tokens      = r.ctx_tokens;
            o.kv_blocks       = r.kv_blocks;
            o.kv_tokens       = r.kv_tokens;
            o.n_draft         = r.n_draft;
            o.slot            = r.slot;
            o.age_s           = r.age_s;
            o.ttft_s          = r.ttft_s;
            o.decode_tps      = r.decode_tps;
            out->push_back(std::move(o));
        }
        return n;
    }

    /* Both read the copy the engine thread publishes (Scheduler::publish), never the
     * scheduler's live state: every caller is an observer. */
    SchedMetrics metrics() const override {
        const rad::SchedMetrics m = s_->observed_metrics();
        SchedMetrics o;
        o.waiting = m.queue_depth;
        o.running = m.running;
        o.preempted = m.preempted_waiting;

        /* One KV utilisation number, and it is the MAXIMUM over groups rather than the mean. A
         * hybrid model has a full-attention group and a linear-state group with different
         * geometries (spec §7.2), and exhausting either one preempts -- so the number a dashboard
         * should be watching is the closest one to the wall, not the average of it with the one
         * that has room.
         *
         * AND THE TOKEN CAPACITY IS NOT A SUM, for the same reason, which is easier to get wrong
         * because the units look additive. PAGED GROUPS SHARE ONE TOKEN AXIS: on Qwen4-Exp
         * `kv_attn` and `kv_qsa_bkey` are both 129216 blocks of 4 tokens, and one token of context
         * takes a slot in BOTH of them (26.00 KiB + 3.25 KiB a block, 7.31 KiB a token together).
         * Adding them would report 1,033,728 tokens of capacity for a pool that holds 516,864 --
         * off by exactly the number of paged groups, an error that is invisible on a one-group
         * model.
         *
         * So the figure is the BINDING group's: the one nearest the wall, which is the one that
         * decides when the next sequence is preempted, and the same group `kv_util` reports. */
        o.kv_util = 0.0;
        int bind = -1;
        for (int i = 0; i < m.n_kv_groups; ++i) {
            if (m.kv[i].block_tokens <= 0) continue;     /* a linear state has no token axis */
            if (bind < 0 || m.kv[i].utilisation > m.kv[bind].utilisation) bind = i;
        }
        for (int i = 0; i < m.n_kv_groups; ++i)
            o.kv_util = std::max(o.kv_util, (double)m.kv[i].utilisation);
        if (bind >= 0) {
            const int64_t bt = m.kv[bind].block_tokens;
            o.kv_tokens_total = m.kv[bind].blocks_total * bt;
            o.kv_tokens_used  = (m.kv[bind].blocks_total - m.kv[bind].blocks_free) * bt;
        }

        /* AND THE WHOLE POOL IN BYTES, WHICH IS THE ONE FIGURE THAT IS A SUM. The paragraph above
         * is about why tokens are not: the groups share a token axis. Bytes do not overlap -- each
         * group was carved its own extent of the card -- so every group is added here, the
         * recurrent ones included, and the answer is how much VRAM the KV cache is holding. */
        for (int i = 0; i < m.n_kv_groups; ++i) {
            o.kv_bytes_total += m.kv[i].bytes_total;
            o.kv_bytes_used  += m.kv[i].bytes_used;
            o.kv_bytes_carved += m.kv[i].bytes_carved;
            if (m.kv[i].block_tokens <= 0 || m.kv[i].window) continue;
            const int64_t t = m.kv[i].blocks_total * m.kv[i].block_tokens;
            if (o.kv_tokens_max == 0 || t < o.kv_tokens_max) o.kv_tokens_max = t;
        }

        /* vLLM's gpu_prefix_cache_hit_rate is a single number and attention caching is what it
         * has always meant. The linear-checkpoint hit rate is a different measurement at a
         * different granularity (spec §7.3) and averaging the two would hide exactly the effect
         * the decoupling exists to expose, so it is not folded in here. */
        o.prefix_hit_rate = m.attn_hit_rate;
        o.prefix_evictions = m.prefix_evictions;
        /* WHAT ONLY THE CACHE IS HOLDING -- see SchedMetrics::prefix_held_tokens. Clamped to what
         * the pool reports held: they are counted at different moments in a step and a cached
         * figure above the used one would draw a negative live share. */
        o.kv_tokens_cached = std::min(m.prefix_held_tokens, o.kv_tokens_used);
        o.prefix_ckpt_evictions = m.prefix_ckpt_evictions;

        o.prefill_tps = m.prefill_tps;
        o.decode_tps = m.decode_tps;
        o.draft_acceptance = m.draft_acceptance;
        o.prefill_tokens = m.prefill_tokens;   o.decode_tokens = m.decode_tokens;
        o.draft_tokens = m.draft_tokens;       o.draft_accepted = m.draft_accepted;
        o.draft_refill_tokens = m.draft_refill_tokens; o.draft_refills = m.draft_refills;
        o.lookup_taken = m.lookup_taken;       o.lookup_steps = m.lookup_steps;
        o.lookup_drafted = m.lookup_drafted;   o.lookup_accepted = m.lookup_accepted;
        o.linear_hit_rate = m.linear_hit_rate;
        o.attn_cached_tokens = m.attn_cached_tokens;
        o.linear_cached_tokens = m.linear_cached_tokens;
        o.checkpoints_written = m.checkpoints_written;
        o.draft_depth_seen = m.draft_depth_seen;
        if (o.draft_depth_seen > SchedMetrics::MAX_DRAFT_POS)
            o.draft_depth_seen = SchedMetrics::MAX_DRAFT_POS;
        for (int j = 0; j < o.draft_depth_seen; ++j) {
            o.draft_pos_reached [j] = m.draft_pos_reached [j];
            o.draft_pos_accepted[j] = m.draft_pos_accepted[j];
        }
        o.preemptions = m.preemptions;
        o.total_requests = m.admitted;
        o.steps = m.steps;
        o.steps_prefill = m.steps_prefill;  o.steps_mixed = m.steps_mixed;
        o.steps_decode = m.steps_decode;
        o.steps_ahead = m.steps_ahead;
        o.decode_rows_unspeculated = m.decode_rows_unspeculated;
        o.step_ns = m.step_ns;  o.step_ns_decode = m.step_ns_decode;
        return o;
    }

private:
    /* DECLARED IN CONSTRUCTION ORDER. Members initialise in declaration order whatever the
     * initialiser list says, so a list that disagrees with it draws a -Wreorder warning and
     * becomes a use-before-initialisation the first time one of these is derived from another. */
    Scheduler*            s_;
    const Sampler*        sampler_ = nullptr;
    const IdleTiers*      tiers_ = nullptr;
    /* Ids start at 1: zero is the "no request" value the sink and the metrics use. */
    std::atomic<uint64_t> next_{1};
};

/* ================================================================== text */

class DetokenizerBridge final : public IDetokenizer {
public:
    DetokenizerBridge(std::shared_ptr<const Vocab> v, std::vector<std::string> stops,
                      const std::vector<std::string>& preserved)
        : d_(std::move(v), std::move(stops), /*render_special=*/false, preserved) {}
    std::string push(int32_t t) override { return d_.push(t); }
    std::string flush() override { return d_.flush(); }
    bool stopped() const override { return d_.stopped(); }
    void defer_stops_until(const std::string& m) override { d_.defer_stops_until(m); }
private:
    Detokenizer d_;
};

class TokenizerBridge final : public ITokenizer {
public:
    explicit TokenizerBridge(std::shared_ptr<const Vocab> v) : vocab_(std::move(v)) {
        tok_.init(vocab_);
    }

    std::vector<int32_t> encode(const std::string& text, bool add_special,
                                bool parse_special) const override {
        std::vector<int32_t> out;
        int st = tok_.encode(text, out, add_special, parse_special);
        if (st < 0) {
            /* The caller checks for an empty result and answers 400. Returning a partial
             * tokenisation would be worse than returning none: it is a prompt that is silently
             * not the prompt that was sent. */
            RAD_WARN("server: tokenise failed: %s", rad_strerror(st));
            out.clear();
        }
        return out;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        /* Sanitised for the same reason Detokenizer::push is: a token array naming byte-fallback
         * ids need not decode to well-formed UTF-8, and this text is echoed back inside a JSON
         * string, where a malformed byte costs the client the whole response. */
        return utf8_sanitize(tok_.decode(ids, /*render_special=*/false));
    }
    int32_t eos() const override { return vocab_->eos(); }
    int32_t bos() const override { return vocab_->bos(); }

    VocabInfo vocab_info() const override {
        VocabInfo i;
        i.n_tokens = vocab_->n_tokens();
        i.bos = vocab_->bos(); i.eos = vocab_->eos(); i.eot = vocab_->eot();
        i.unk = vocab_->unk(); i.pad = vocab_->pad();
        i.add_bos = vocab_->add_bos(); i.add_eos = vocab_->add_eos();
        switch (vocab_->kind()) {
            case RAD_TOK_BPE:       i.kind = "bpe";       break;
            case RAD_TOK_UNIGRAM:   i.kind = "unigram";   break;
            case RAD_TOK_WORDPIECE: i.kind = "wordpiece"; break;
            case RAD_TOK_RWKV:      i.kind = "rwkv";      break;
            default:                i.kind = "unknown";   break;
        }
        return i;
    }

    /* Vocab::text, sanitised: a byte-level BPE piece is a mapped codepoint sequence that is valid
     * UTF-8, but a byte-fallback vocabulary's piece need not be, and this goes into a JSON string
     * where one malformed byte costs the client the whole response. */
    std::string piece(int32_t id) const override {
        if (id < 0 || id >= vocab_->n_tokens()) return std::string();
        return utf8_sanitize(vocab_->text(id));
    }

    std::unique_ptr<IDetokenizer> detokenizer(
        const std::vector<std::string>& stops,
        const std::vector<std::string>& preserved = {}) const override {
        return std::unique_ptr<IDetokenizer>(new DetokenizerBridge(vocab_, stops, preserved));
    }

private:
    std::shared_ptr<const Vocab> vocab_;
    Tokenizer                    tok_;
};

/* ================================================================== chat
 *
 * THE ONE PLACE THE TWO CHAT INTERFACES MEET. core/text's ChatTemplate renders the prompt and the
 * generation prompt; the model's reply format (chatfmt.h) says how the reply is spelled, and
 * everything that depends on the spelling -- the grammar, its trigger words, the tokens the
 * decoder must render, and the parser that reads the reply back -- is derived from that one
 * description here. Two decisions worth stating:
 *
 *   * `preserved_tokens` IS CARRIED, and the reason is not the one it has upstream. Upstream it
 *     stops a sampler that bans special tokens from banning the ones a tool-call format is made
 *     of; this sampler has no ban stage at all (core/sample/sampler.h), so there is nothing to
 *     preserve them FROM. It is carried because the DECODER suppresses control tokens, and a
 *     format can be built out of them -- MiniCPM5 writes `<function`, `<param` and their closers
 *     as control tokens, so a reply decoded the ordinary way reaches the parser as ` name="get
 *     _weather"> name="city">Berlin`, every tag gone, and comes back to the client as prose.
 *     These, and only these, are decoded as text. If a ban stage is ever added it wants this
 *     same field, for the original reason.
 *
 *   * THE PARSER IS BUILT PER RENDER and handed over behind IReplyParser. A server-wide parser
 *     cannot be correct: which arguments are JSON depends on the schemas THIS request sent, and
 *     whether the reply starts inside a reasoning block depends on THIS generation prompt.
 */
class ReplyReaderBridge final : public IReplyReader {
public:
    ReplyReaderBridge(std::unique_ptr<ReplyStream> s, const std::string* format_name)
        : s_(std::move(s)), name_(format_name),
          keep_(log_level() >= Log::Trace) {}

    void feed(std::string_view text, std::vector<ReplyDelta>* out) override {
        if (keep_) raw_.append(text);
        s_->feed(text, out ? &ev_ : nullptr);
        move_out(out);
    }

    void finish(std::vector<ReplyDelta>* out) override {
        s_->finish(out ? &ev_ : nullptr);
        move_out(out);
        /* A reply that left the grammar still came back whole -- a call to a function that is
         * not a tool, text inside a call, a call after a closing sentence -- but it is worth a
         * line: it is either a model writing its format loosely or a format described wrongly,
         * and nothing in the response says which. */
        if (!s_->strict() && !warned_) {
            warned_ = true;
            RAD_WARN("server: a reply left the '%s' reply format and was read by its recovery "
                     "rules; at trace level the reply is logged", name_->c_str());
            if (keep_) RAD_TRACE("server: the reply was <<<%s>>>", raw_.c_str());
        }
    }

    ParsedMessage message() const override {
        const ReplyMessage& r = s_->message();
        ParsedMessage m;
        m.content   = r.content;
        m.reasoning = r.reasoning;
        m.tool_calls.reserve(r.calls.size());
        for (const ReplyCall& c : r.calls) m.tool_calls.push_back({ std::string(), c.name, c.arguments });
        m.open_call = !r.calls.empty() && !r.calls.back().closed;
        return m;
    }

    std::string content_with_open_call(std::string_view reply) const override {
        return s_->content_with_open_call(reply);
    }

private:
    void move_out(std::vector<ReplyDelta>* out) {
        if (!out) return;
        for (ReplyEvent& e : ev_) {
            ReplyDelta d;
            switch (e.kind) {
            case ReplyEvent::Reasoning: d.kind = ReplyDelta::Reasoning; break;
            case ReplyEvent::Content:   d.kind = ReplyDelta::Content;   break;
            case ReplyEvent::CallBegin: d.kind = ReplyDelta::CallBegin; break;
            case ReplyEvent::CallArgs:  d.kind = ReplyDelta::CallArgs;  break;
            case ReplyEvent::CallEnd:   d.kind = ReplyDelta::CallEnd;   break;
            }
            d.call = e.call;
            d.text = std::move(e.text);
            out->push_back(std::move(d));
        }
        ev_.clear();
    }

    std::unique_ptr<ReplyStream> s_;
    const std::string*           name_;
    std::vector<ReplyEvent>      ev_;
    bool                         keep_ = false, warned_ = false;
    std::string                  raw_;   /* the reply, kept only when trace logging is on */
};

class ReplyParserBridge final : public IReplyParser {
public:
    explicit ReplyParserBridge(std::shared_ptr<const ReplyParser> p) : p_(std::move(p)) {}
    std::unique_ptr<IReplyReader> open() const override {
        return std::unique_ptr<IReplyReader>(new ReplyReaderBridge(p_->open(), &p_->format().name));
    }
private:
    std::shared_ptr<const ReplyParser> p_;
};

class ChatTemplateBridge final : public IChatTemplate {
public:
    /* Returns RAD_OK, or the status ChatTemplate::load_from_vocab gave. A server whose template
     * did not compile must not be wired: the endpoint answering 501 is a better outcome than one
     * answering 200 with ChatML-shaped nonsense.
     *
     * `declared` is the architecture plugin's reply format (rad_arch_chat_format), or null to
     * derive it from the template. A declaration the parser cannot run is refused here, at
     * startup, rather than at the first request that needs it. */
    int init(const Vocab& v, const RadChatFormat* declared = nullptr,
             const std::string& plugin = std::string()) {
        /* The spellings, resolved once. See ReplyOptions::terminators. */
        terminators_.clear();
        for (int32_t id : v.eog()) {
            const std::string s = v.text(id);
            if (!s.empty()) terminators_.push_back(s);
        }
        const int st = t_.load_from_vocab(v);
        if (st < 0) return st;

        ChatFormat f;
        std::string why;
        declared_ = declared != nullptr;
        if (declared) {
            if (chat_format_from_abi(declared, &f, &why) < 0 || chat_format_check(f, &why) < 0) {
                RAD_ERR("chat: plugin '%s' declares a reply format the parser cannot run: %s",
                        plugin.c_str(), why.c_str());
                return RAD_E_INVAL;
            }
            if (f.name.empty()) f.name = "plugin:" + plugin;
        } else {
            const std::string bos = v.bos() >= 0 ? v.text(v.bos()) : std::string();
            const std::string eos = v.eos() >= 0 ? v.text(v.eos()) : std::string();
            std::string refused;
            if (chat_format_from_template(v.chat_template(), bos, eos, &f, &refused, &why) < 0) {
                RAD_WARN("chat: this model's reply format cannot be described (%s). Replies are "
                         "returned as content with no reasoning split, and requests with tools "
                         "are refused.", why.c_str());
                f = ChatFormat{};
                f.name = "content";
                tool_refusal_ = why;
            } else if (!refused.empty()) {
                tool_refusal_ = refused;
            } else if (t_.caps().tools && !f.has_tools()) {
                tool_refusal_ = "the template analysis found no tool-call format in it";
            }
        }
        if (ReplyFormat::compile(f, &fmt_, &why) < 0) {
            RAD_ERR("chat: the reply format '%s' cannot be run: %s", f.name.c_str(), why.c_str());
            return RAD_E_INVAL;
        }
        RAD_INFO("chat: reply format %s", chat_format_describe(f).c_str());
        if (t_.caps().tools && !tool_refusal_.empty())
            RAD_WARN("chat: tool calls are refused for this model: %s", tool_refusal_.c_str());
        return RAD_OK;
    }

    bool ready() const { return t_.ready() && fmt_ != nullptr; }
    bool supports_tools() const override { return t_.caps().tools; }
    bool supports_reasoning() const override { return t_.caps().preserve_reasoning; }
    std::string tool_refusal() const override { return tool_refusal_; }
    const std::string& source() const { return t_.source(); }

    int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
              ChatRender* out, std::string* why) const override {
        ChatOptions o;
        o.add_generation_prompt = opt.add_generation_prompt;
        o.enable_thinking       = opt.enable_thinking;
        o.parallel_tool_calls   = opt.parallel_tool_calls;
        o.tool_choice           = opt.tool_choice;
        o.reasoning_format      = opt.reasoning_format;
        o.grammar               = opt.grammar;
        o.json_schema           = opt.json_schema;
        o.template_kwargs       = opt.template_kwargs;

        ChatPrompt p;
        const int st = t_.apply(messages, tools, o, &p);
        if (st < 0) {
            /* THE TEMPLATE'S OWN SENTENCE WHERE THERE IS ONE. A jinja `raise_exception` names the
             * field and the values it accepts; the generic line below names neither, and a caller
             * reading a 400 has no other way to find out. */
            if (why)
                *why = !p.error.empty()
                           ? p.error
                           : st == RAD_E_UNSUPPORTED
                                 ? "the model's chat template refused this request"
                                 : "the messages could not be rendered by this model's chat "
                                   "template";
            return st;
        }

        const ChatFormat& f = fmt_->format();
        /* Calls are parsed only when the request offered tools to call. With none, or with
         * `tool_choice: "none"` (which arrives here with the tools already cleared), call syntax
         * in the reply is prose to this request. */
        const bool calls = opt.tool_choice != "none" && tools.is_array() && !tools.empty();
        const bool doc = !opt.json_schema.empty();

        out->prompt            = std::move(p.prompt);
        out->additional_stops  = std::move(p.additional_stops);
        out->generation_prompt = p.generation_prompt;
        out->preserved_tokens  = declared_ ? chat_format_markers(f) : p.preserved_tokens;
        out->grammar_triggers.clear();

        if (calls || doc) {
            /* The grammar, and the words that trigger it, come from the reply format: the same
             * description the parser reads the reply back with. */
            const nlohmann::ordered_json schema =
                doc ? nlohmann::ordered_json::parse(opt.json_schema, nullptr, false)
                    : nlohmann::ordered_json();
            ChatGrammarRequest gq;
            gq.tools               = calls ? &tools : nullptr;
            gq.json_schema         = doc && schema.is_object() ? &schema : nullptr;
            gq.generation_prompt   = p.generation_prompt;
            gq.parallel_tool_calls = opt.parallel_tool_calls;
            gq.extract_reasoning   = opt.reasoning_format != "none";
            ChatGrammar g;
            std::string gerr;
            if (chat_format_grammar(f, gq, &g, &gerr) < 0) {
                if (why) *why = "the grammar for this request could not be built: " + gerr;
                return RAD_E_INVAL;
            }
            out->grammar      = std::move(g.text);
            out->grammar_lazy = g.lazy;
            for (const std::string& w : g.trigger_words)
                out->grammar_triggers.push_back({ kTriggerWord, w });
        } else {
            out->grammar      = std::move(p.grammar);
            out->grammar_lazy = p.grammar_lazy;
            for (const auto& t : p.grammar_triggers)
                out->grammar_triggers.push_back({ t.type, t.value });
        }

        ReplyOptions ro;
        ro.generation_prompt   = p.generation_prompt;
        ro.extract_reasoning   = opt.reasoning_format != "none";
        ro.tools               = calls ? &tools : nullptr;
        ro.parallel_tool_calls = opt.parallel_tool_calls;
        ro.response_format     = doc;
        ro.terminators         = terminators_;
        std::shared_ptr<const ReplyParser> rp;
        std::string perr;
        const int ps = ReplyParser::create(fmt_, ro, &rp, &perr);
        if (ps < 0) {
            if (why) *why = perr;
            return ps;
        }
        const ReasoningMarkers rm = rp->reasoning();
        out->think_open        = rm.open;
        out->think_close       = rm.close;
        out->think_starts_open = rm.starts_open;
        out->parser = std::make_shared<ReplyParserBridge>(std::move(rp));
        return RAD_OK;
    }

private:
    /* A lazy grammar's trigger is a literal word; SamplingParams::GrammarTrigger numbers the
     * kinds the way llama.cpp's common_grammar_trigger_type does. */
    static constexpr int kTriggerWord = 1;

    ChatTemplate t_;
    std::vector<std::string> terminators_;
    std::shared_ptr<const ReplyFormat> fmt_;
    bool declared_ = false;
    std::string tool_refusal_;
};

/* ================================================================== grammar
 *
 * Compiles at ADMISSION, not at first token, so a schema that cannot be compiled is a 400 naming
 * it rather than a request that starts and then dies on the step path -- and so the scheduler
 * thread never compiles one.
 */
class GrammarBridge final : public IGrammarCompiler {
public:
    explicit GrammarBridge(const VocabView* v) : v_(v) {}

    int from_json_schema(const json& schema, std::string& out, std::string& err) const override {
        return grammar_from_json_schema(schema.dump(), "root", &out, &err);
    }

    /* The whole compile -- parse, checks and mask plan -- on the admission thread. The program
     * goes back to ride on the request; the sampler finds it by its text when the request starts,
     * for as long as the request holds it. */
    int compile(const std::string& gbnf, std::shared_ptr<const GbnfProgram>* out,
                std::string& err) const override {
        if (gbnf.empty()) return RAD_OK;
        if (!v_) { err = "no vocabulary attached"; return RAD_E_STATE; }
        return gbnf_compile(gbnf, "root", v_, out, &err);
    }

private:
    const VocabView* v_;
};

/* MEDIA, through the processor the container describes (core/mm). Present only when the program
 * declared an encoder: `accepts` answers from what that encoder said it serves, so a request for
 * video against an image-only tower is refused by the server with a reason rather than processed
 * into patches nothing can encode. */
class MultimodalBridge final : public IMultimodal {
public:
    MultimodalBridge(const mm::Processor* p, uint32_t modalities) : p_(p), mod_(modalities) {}

    bool accepts(MediaKind k) const override {
        if (!p_ || !mm::decoder_available()) return false;
        if (k == MediaKind::Image) return (mod_ & RAD_MM_IMAGE) != 0;
        if (k == MediaKind::Video) return (mod_ & RAD_MM_VIDEO) != 0;
        return false;
    }

    int prepare(MediaKind k, const std::vector<uint8_t>& bytes, std::shared_ptr<mm::Item>* out,
                std::string* why) override {
        if (!accepts(k)) { if (why) *why = "this kind of media is not served"; return RAD_E_UNSUPPORTED; }
        return k == MediaKind::Image ? p_->image(bytes.data(), bytes.size(), out, why)
                                     : p_->video(bytes.data(), bytes.size(), out, why);
    }

private:
    const mm::Processor* p_;
    uint32_t             mod_;
};

}  /* namespace server */
}  /* namespace rad */

#endif  /* the engine components are present */
