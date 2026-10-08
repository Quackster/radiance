/* tests/server_test.cpp -- the HTTP layer, driven in process.
 *
 * No port is bound, except in the transport section at the end. Router::dispatch is the same call
 * the transport makes, so every handler here runs exactly the code a real request runs; what is
 * faked is the engine behind it, which is the only way to make cancellation and the streaming wire
 * format deterministic. Binding a loopback socket would test cpp-httplib, which is already tested,
 * and would make the suite depend on a free port and on timing. The one exception is how the pool
 * is CONFIGURED, which is ours and which only a socket can observe; it binds port 0.
 */
#include "mm/processor.h"
#include "rad_test.h"

#include "server/adapters.h"
#include "server/http.h"
#include "server/json_partial.h"
#include "server/liveview.h"
#include "server/metrics.h"
#include "server/oai.h"
#include "server/server.h"
#include "server/sink.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace rad;
using namespace rad::server;
using json = nlohmann::ordered_json;

/* A substring test, so a CHECK reads as the claim it is making rather than as npos arithmetic. */
static bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/* ================================================================== fakes */

/* One token per byte. Trivially round-trips, which makes a stop string a substring test and an
 * expected completion a literal in the test rather than a vocabulary lookup. */
struct FakeDetok : IDetokenizer {
    std::string push(int32_t t) override {
        if (t <= 0) return "";
        return std::string(1, (char)t);
    }
    std::string flush() override { return ""; }
};

struct FakeTokenizer : ITokenizer {
    mutable bool last_parse_special = false;
    /* A vocabulary with llama's three fill-in-the-middle markers as special tokens, for /infill. */
    bool fim = false;
    static const char* fim_name(int32_t id) {
        return id == 1001 ? "<PRE>" : id == 1002 ? "<SUF>" : id == 1003 ? "<MID>" : nullptr;
    }
    std::vector<int32_t> encode(const std::string& s, bool, bool parse_special) const override {
        last_parse_special = parse_special;
        if (fim && parse_special)
            for (int32_t id = 1001; id <= 1003; ++id)
                if (s == fim_name(id)) return { id };
        std::vector<int32_t> v;
        v.reserve(s.size());
        for (unsigned char c : s) v.push_back((int32_t)c);
        return v;
    }
    std::string decode(const std::vector<int32_t>& t) const override {
        std::string s;
        for (int32_t x : t) if (x > 0) s += token_bytes(x);
        return s;
    }
    std::string token_bytes(int32_t id) const override {
        if (fim && fim_name(id)) return fim_name(id);
        return std::string(1, (char)id);
    }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return 1; }
    /* Stop strings are deliberately NOT given to the fake decoder, so the server's own scan is
     * what these tests exercise. The real Detokenizer can take them and report stopped(). */
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                              const std::vector<std::string>&) const override {
        return std::unique_ptr<IDetokenizer>(new FakeDetok());
    }
};

struct FakeChatTemplate : IChatTemplate {
    bool tools_ok = true;
    /* What the last render was asked for, so a test can assert that a request's options actually
     * reached the template rather than being dropped between the two. */
    mutable ChatRenderOptions last_opt;
    mutable int last_n_tools = 0;

    int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
              ChatRender* out, std::string* why) const override {
        last_opt = opt;
        last_n_tools = (int)tools.size();
        /* A template that raise_exception's on one variable, as Qwen3.8's does on an effort it
         * does not know. */
        if (!refuse_kwarg.empty() && opt.template_kwargs.count(refuse_kwarg)) {
            if (why) *why = "the template does not take " + refuse_kwarg;
            return RAD_E_UNSUPPORTED;
        }
        /* And one with a set of its own, as Qwen3.8's xhigh|medium|low. */
        if (!efforts.empty() && opt.template_kwargs.count("reasoning_effort")) {
            const std::string& v = opt.template_kwargs.at("reasoning_effort");
            if (std::find(efforts.begin(), efforts.end(), v) == efforts.end()) {
                if (why) *why = "Jinja Exception: Unexpected reasoning effort " + v;
                return RAD_E_UNSUPPORTED;
            }
        }
        std::string s;
        if (!tools.empty()) s += "[tools:" + std::to_string(tools.size()) + "]";
        if (!opt.enable_thinking) s += "[nothink]";
        for (const auto& kv : opt.template_kwargs) s += "[" + kv.first + "=" + kv.second + "]";
        for (const auto& m : messages) {
            s += "<" + m.value("role", std::string("?")) + ">";
            const auto& c = m.at("content");
            if (c.is_string()) s += c.get<std::string>();
            else for (const auto& p : c) if (p.value("type", std::string()) == "text")
                s += p.value("text", std::string());
        }
        if (opt.add_generation_prompt) s += "<assistant>";
        out->prompt = s;
        /* A tool-calling template's grammar is lazy, and the tests care that the laziness
         * survives the trip: a non-lazy one would constrain the whole reply to be a call. */
        if (!tools.empty()) {
            out->grammar = "root ::= \"TOOL:\" [^\\n]*";
            out->grammar_lazy = true;
            out->grammar_triggers.push_back({ /*type=word*/ 1, "TOOL:" });
        } else {
            out->grammar = opt.grammar;
        }
        /* The parser is built per render and reads calls only for a request that sent tools --
         * the contract the real bridge keeps (IReplyParser). */
        if (real_parser) out->parser = real_parser;
        else if (with_parser) out->parser = std::make_shared<FakeReplyParser>(!tools.empty());
        out->think_open = think_open;
        out->think_close = think_close;
        out->think_starts_open = think_starts_open;
        return RAD_OK;
    }
    bool supports_tools() const override { return tools_ok; }
    bool supports_reasoning() const override { return false; }
    std::string tool_refusal() const override { return refusal; }

    bool        with_parser = false;
    /* A real reply parser to hand out instead of the fake one. */
    std::shared_ptr<IReplyParser> real_parser;
    std::string refusal;
    std::string refuse_kwarg;
    std::vector<std::string> efforts;   /* JSON text, as the template context carries them */
    /* The reasoning block a render reports. FakeTokenizer makes a one-byte marker one token. */
    std::string think_open, think_close;
    bool        think_starts_open = false;

    /* Reads "TOOL:<name>:<json...>" -- everything after the second colon is the arguments,
     * however incomplete, and it arrives a piece at a time. That is exactly the shape the
     * streaming path has to cope with. With calls off every byte is content. The call is open
     * when the reply ends before its arguments are whole, and its text is then the reply. */
    struct FakeReplyReader : IReplyReader {
        explicit FakeReplyReader(bool calls) : calls_(calls) {}
        void feed(std::string_view t, std::vector<ReplyDelta>* out) override {
            text_.append(t);
            step(out, false);
        }
        void finish(std::vector<ReplyDelta>* out) override {
            step(out, true);
            msg_.open_call = !msg_.tool_calls.empty() && !json_complete(msg_.tool_calls[0].arguments);
        }
        ParsedMessage message() const override { return msg_; }
        std::string content_with_open_call(std::string_view reply) const override {
            return msg_.open_call ? std::string(reply) : msg_.content;
        }
    private:
        void emit(std::vector<ReplyDelta>* out, ReplyDelta::Kind k, std::string s) {
            if (out) out->push_back({ k, 0, std::move(s) });
        }
        void step(std::vector<ReplyDelta>* out, bool final) {
            static const std::string tag = "TOOL:";
            if (mode_ == 0) {
                /* Undecided while what has arrived could still be the start of the tag. */
                if (calls_ && !final && text_.size() < tag.size() &&
                    tag.compare(0, text_.size(), text_) == 0)
                    return;
                mode_ = calls_ && text_.rfind(tag, 0) == 0 ? 2 : 1;
            }
            if (mode_ == 1) {
                if (text_.size() > sent_) emit(out, ReplyDelta::Content, text_.substr(sent_));
                msg_.content = text_;
                sent_ = text_.size();
                return;
            }
            if (!named_) {
                const size_t c2 = text_.find(':', tag.size());
                if (c2 == std::string::npos) return;
                named_ = true;
                msg_.tool_calls.push_back({ "", text_.substr(tag.size(), c2 - tag.size()), "" });
                emit(out, ReplyDelta::CallBegin, msg_.tool_calls[0].name);
                sent_ = c2 + 1;
            }
            if (text_.size() > sent_) emit(out, ReplyDelta::CallArgs, text_.substr(sent_));
            msg_.tool_calls[0].arguments += text_.substr(sent_);
            sent_ = text_.size();
        }
        bool          calls_;
        int           mode_ = 0;
        bool          named_ = false;
        std::string   text_;
        size_t        sent_ = 0;
        ParsedMessage msg_;
    };

    struct FakeReplyParser : IReplyParser {
        explicit FakeReplyParser(bool calls) : calls_(calls) {}
        std::unique_ptr<IReplyReader> open() const override {
            return std::unique_ptr<IReplyReader>(new FakeReplyReader(calls_));
        }
        bool calls_;
    };
};

struct FakeGrammar : IGrammarCompiler {
    int from_json_schema(const json&, std::string& out, std::string&) const override {
        out = "root ::= object";
        return RAD_OK;
    }
    /* "bad" is the one grammar this fake refuses, so a test can drive the admission-time
     * rejection without a real GBNF parser behind it. */
    int compile(const std::string& g, std::shared_ptr<const GbnfProgram>*,
                std::string& err) const override {
        if (g.find("bad") != std::string::npos) { err = "left recursion"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

/* Takes any media and says it fills one position, so a request's media parts get as far as the
 * server's own validation of them. */
struct FakeMultimodal : IMultimodal {
    bool accepts(MediaKind) const override { return true; }
    int prepare(MediaKind, const std::vector<uint8_t>&, const MediaOpts&,
                std::shared_ptr<mm::Item>* out, std::string*) override {
        auto it = std::make_shared<mm::Item>();
        it->expand = { 0 };
        *out = std::move(it);
        return RAD_OK;
    }
};

/* The scheduler. Each submitted request gets a thread that writes `reply` into the sink one byte
 * at a time. In `hang` mode it stops after three bytes and waits to be cancelled, which is what
 * the cancellation test needs: a generation that will never finish on its own. */
struct FakeScheduler : IScheduler {
    std::string reply = "hello";
    bool hang = false;
    /* Nonzero: after the reply, the request is failed with this status and reason, as the engine
     * fails every request it holds when a step fails. */
    int fail_status = 0;
    const char* fail_why = nullptr;
    /* Nonzero: every token after the first is a verify step that proposed this many drafts and
     * accepted one, as a speculating engine reports them. */
    int32_t drafts = 0;
    server::SchedMetrics m;

    std::mutex mu;
    std::vector<uint64_t> submitted;
    SamplingParams last_sp;              /* what the last submitted request was to sample with */
    std::vector<std::string> last_stop;  /* ...and the stop strings it was handed */
    std::string last_label;              /* ...its session label */
    int32_t last_max_tokens = 0;         /* ...and its max_tokens */
    std::vector<uint64_t> cancelled;
    std::vector<std::thread> workers;
    std::atomic<uint64_t> next_id{1};

    ~FakeScheduler() override { join_all(); }

    void join_all() {
        for (auto& t : workers) if (t.joinable()) t.join();
        workers.clear();
    }

    uint64_t submit(Request&& r) override {
        uint64_t id = next_id.fetch_add(1);
        {
            std::lock_guard<std::mutex> lk(mu);
            submitted.push_back(id);
            last_sp = r.sp;
            last_stop = r.stop;
            last_label = r.session_label;
            last_max_tokens = r.max_tokens;
        }
        void* sink = r.sink;
        int32_t cap = r.max_tokens;
        std::string text = reply;
        bool h = hang;
        const int fst = fail_status;
        const char* fwhy = fail_why;
        const int32_t nd = drafts;
        const int np = r.sp.n_probs;
        /* Under the lock: over a real socket, submits arrive from several HTTP workers at once. */
        std::lock_guard<std::mutex> lk(mu);
        workers.emplace_back([sink, cap, text, h, fst, fwhy, nd, np] {
            sink_admitted(sink);
            sink_prompt_stats(sink, 8, 4);
            int32_t n = 0;
            for (char c : text) {
                if (n >= cap) break;
                if (sink_cancelled(sink)) { sink_finish(sink, Finish::Cancelled); return; }
                if (n > 0 && nd > 0) sink_drafts(sink, nd, 1);
                /* Logprobs when asked: the n-th token at -(n+1)/4, and alternatives 'x', 'y', ...
                 * at -1, -2, ... */
                TokenLogprobs lp;
                lp.token = (int32_t)(unsigned char)c;
                lp.logprob = -0.25f * (float)(n + 1);
                lp.n_top = np > 0 ? np : 0;
                for (int j = 0; j < lp.n_top; ++j) {
                    lp.top_id[j] = 'x' + j;
                    lp.top_lp[j] = -1.0f - (float)j;
                }
                if (!sink_push(sink, (int32_t)(unsigned char)c, np >= 0 ? &lp : nullptr)) return;
                ++n;
                if (h && n >= 3) break;
            }
            if (h) {
                for (int i = 0; i < 20000; ++i) {
                    if (sink_cancelled(sink)) { sink_finish(sink, Finish::Cancelled); return; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                sink_finish(sink, Finish::Length);
                return;
            }
            if (fst) { sink_fail(sink, fst, fwhy); return; }
            sink_finish(sink, n >= cap ? Finish::Length : Finish::Stop);
        });
        return id;
    }

    void cancel(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu);
        cancelled.push_back(id);
    }

    server::SchedMetrics metrics() const override { return m; }

    bool saw_cancel() {
        std::lock_guard<std::mutex> lk(mu);
        return !cancelled.empty();
    }
};

/* A server wired to the fakes. */
struct Fixture {
    FakeScheduler    sched;
    FakeTokenizer    tok;
    FakeChatTemplate tmpl;
    FakeGrammar      gram;
    std::unique_ptr<Server> srv;

    explicit Fixture(bool with_tools = true, int64_t max_ctx = 0, const char* model_path = "",
                     bool logprobs = false) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        d.grammar = &gram;
        /* The parser travels with the render, so "no tool parsing" is a template that returns
         * no parser rather than a null server-wide dependency. */
        tmpl.with_parser = with_tools;
        ServerOptions o;
        o.model_id = "test-model";
        o.default_max_tokens = 32;
        o.max_seqs = 4;
        o.max_ctx = max_ctx;
        o.model_path = model_path;
        o.supports_logprobs = logprobs;
        srv.reset(new Server(d, o));
    }

    /* Members are destroyed in reverse declaration order, which would tear down the Server while
     * the fake's worker threads are still writing into sinks it owns. The engine has the same
     * obligation: stop the scheduler, then destroy the server (server.h). Doing it explicitly
     * here is the test honouring the rule rather than getting away with breaking it. */
    ~Fixture() {
        sched.join_all();
        srv.reset();
    }

    OaiDeps odeps() {
        OaiDeps d;
        d.tok = &tok;
        d.chat = &tmpl;
        d.grammar = &gram;
        return d;
    }
    OaiLimits limits() {
        OaiLimits l;
        l.model_id = "test-model";
        l.default_max_tokens = 32;
        return l;
    }

    HttpResponse post(const char* path, const std::string& body) {
        return srv->router().call("POST", path, body);
    }
    HttpResponse get(const char* path) { return srv->router().call("GET", path); }
};

/* Drain a streaming response the way the transport does. */
static std::string drain(HttpResponse& res, int max_chunks = 100000) {
    std::string all;
    for (int i = 0; i < max_chunks; ++i) {
        std::string c;
        bool more = res.next(c);
        all += c;
        if (!more) break;
    }
    if (res.on_close) res.on_close(true);
    return all;
}

/* ================================================================== parsing */

TEST(oai_parses_a_full_chat_request_into_sampling_params) {
    Fixture f;
    const std::string body = R"({
      "model":"test-model",
      "messages":[{"role":"system","content":"be brief"},
                  {"role":"user","content":[{"type":"text","text":"hi"}]}],
      "temperature":0.7,"top_p":0.9,"top_k":40,"min_p":0.05,"typical_p":0.95,
      "presence_penalty":0.5,"frequency_penalty":-0.5,
      "repetition_penalty":1.1,"repeat_last_n":128,
      "seed":12345,"stop":["\n\n","END"],"max_tokens":64,"n":2,"stream":false,
      "ignore_eos":true,
      "xtc_probability":0.2,"xtc_threshold":0.15,
      "dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":3,
      "dry_penalty_last_n":256,"dry_sequence_breakers":["\n"],
      "response_format":{"type":"json_schema",
                         "json_schema":{"name":"x","schema":{"type":"object"}}},
      "tools":[{"type":"function","function":{"name":"get_weather",
                "parameters":{"type":"object"}}}],
      "tool_choice":"auto",
      "user":"u-1"
    })";

    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(body, f.odeps(), f.limits(), &r, &e));

    CHECK_NEAR(r.sp.temp, 0.7f, 1e-6);
    CHECK_NEAR(r.sp.top_p, 0.9f, 1e-6);
    CHECK_EQ(r.sp.top_k, 40);
    CHECK_NEAR(r.sp.min_p, 0.05f, 1e-6);
    CHECK_NEAR(r.sp.typical_p, 0.95f, 1e-6);
    CHECK_NEAR(r.sp.pres_penalty, 0.5f, 1e-6);
    CHECK_NEAR(r.sp.freq_penalty, -0.5f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.1f, 1e-6);
    CHECK_EQ(r.sp.penalty_last_n, 128);
    CHECK_NEAR(r.sp.xtc_probability, 0.2f, 1e-6);
    CHECK_NEAR(r.sp.xtc_threshold, 0.15f, 1e-6);
    CHECK_NEAR(r.sp.dry_multiplier, 0.8f, 1e-6);
    CHECK_EQ(r.sp.dry_allowed_length, 3);
    CHECK_EQ(r.sp.dry_penalty_last_n, 256);
    CHECK_EQ((int)r.sp.dry_seq_breakers.size(), 1);
    CHECK_EQ((long long)r.sp.seed, 12345LL);
    CHECK_EQ(r.max_tokens, 64);
    CHECK_EQ(r.n, 2);
    CHECK_EQ((int)r.stop.size(), 2);
    CHECK(r.ignore_eos);
    CHECK(!r.sp.grammar.empty());          /* response_format compiled to a grammar */
    CHECK(r.tools_present);
    CHECK(r.parse_tools);
    CHECK_EQ((int)r.prompts.size(), 1);
    CHECK_EQ(r.n_choices(), 2);
    CHECK(r.id.rfind("chatcmpl-", 0) == 0);
    /* The template ran and the prompt tokenised. */
    CHECK(r.prompts[0].text.find("be brief") != std::string::npos);
    CHECK(!r.prompts[0].tokens.empty());
}

TEST(oai_parses_a_minimal_chat_request_with_defaults) {
    Fixture f;
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})",
                                f.odeps(), f.limits(), &r, &e));
    CHECK_NEAR(r.sp.temp, 1.0f, 1e-6);
    CHECK_NEAR(r.sp.top_p, 1.0f, 1e-6);
    CHECK_EQ(r.sp.top_k, 0);
    CHECK_NEAR(r.sp.min_p, 0.0f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.0f, 1e-6);
    CHECK_EQ(r.n, 1);
    CHECK_EQ(r.max_tokens, 32);           /* the server default, not zero */
    CHECK_EQ((int)r.stop.size(), 0);
    CHECK(r.sp.grammar.empty());
    CHECK(!r.stream);
    CHECK(!r.tools_present);
    CHECK_EQ(r.model, std::string("test-model"));
    /* llama-server's DRY breakers when none are named; a list that is sent replaces them. */
    CHECK(r.sp.dry_seq_breakers == std::vector<std::string>({ "\n", ":", "\"", "*" }));
    OaiRequest none;
    CHECK_OK(parse_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"dry_sequence_breakers":[]})",
        f.odeps(), f.limits(), &none, &e));
    CHECK(none.sp.dry_seq_breakers.empty());
}

/* THE DEPLOYMENT'S SAMPLER REACHES A REQUEST THAT SENDS NONE, AND LOSES TO ONE THAT DOES.
 *
 * The built-in defaults are an untruncated sampler, which serves a checkpoint tuned against a
 * truncated tail with its whole vocabulary live -- and the visible symptom of that is a reply
 * that ends early rather than one that reads badly, because end-of-turn is among the tokens the
 * tail holds. OaiLimits::default_sampling is how a deployment states the sampler the checkpoint
 * asked for; a caller that names a field still wins, field by field. */
TEST(oai_seeds_sampling_from_the_deployment_default) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_sampling.temp  = 0.7f;
    lim.default_sampling.top_k = 20;
    lim.default_sampling.top_p = 0.95f;
    lim.default_sampling.min_p = 0.01f;

    /* Nothing sent: every field is the deployment's. */
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}]})",
                                f.odeps(), lim, &r, &e));
    CHECK_NEAR(r.sp.temp, 0.7f, 1e-6);
    CHECK_EQ(r.sp.top_k, 20);
    CHECK_NEAR(r.sp.top_p, 0.95f, 1e-6);
    CHECK_NEAR(r.sp.min_p, 0.01f, 1e-6);

    /* One field sent: that one is the caller's and the rest are still the deployment's. A
     * default that could not be overridden would be a server the caller cannot steer. */
    OaiRequest r2;
    CHECK_OK(parse_chat_request(
        R"({"messages":[{"role":"user","content":"hi"}],"temperature":0,"top_k":0})",
        f.odeps(), lim, &r2, &e));
    CHECK_NEAR(r2.sp.temp, 0.0f, 1e-6);
    CHECK_EQ(r2.sp.top_k, 0);              /* 0 is a value, not silence */
    CHECK_NEAR(r2.sp.top_p, 0.95f, 1e-6);
    CHECK(r2.sp.greedy());

    /* The same on /v1/completions: one model served two ways must sample the same. */
    OaiRequest r3;
    CHECK_OK(parse_completion_request(R"({"prompt":"ab"})", f.odeps(), lim, &r3, &e));
    CHECK_NEAR(r3.sp.temp, 0.7f, 1e-6);
    CHECK_EQ(r3.sp.top_k, 20);
    CHECK_NEAR(r3.sp.top_p, 0.95f, 1e-6);
}

TEST(oai_parses_a_completion_prompt_in_every_shape) {
    Fixture f;
    OaiLimits lim = f.limits();
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"ab"})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 1);
        CHECK_EQ((int)r.prompts[0].tokens.size(), 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":["a","bc"]})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 2);
        CHECK_EQ(r.n_choices(), 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[104,105]})", f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 1);
        CHECK_EQ(r.prompts[0].text, std::string("hi"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[[104],[105,106]]})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.prompts.size(), 2);
    }
}

/* A max_tokens THAT DOES NOT FIT IS CLAMPED, NOT REFUSED, and this is the test that keeps it that
 * way. The field is an upper bound and the context is a lower one; refusing the request produces
 * no tokens where clamping produces every token that could ever have been produced. What stays a
 * refusal is a prompt with no room left to generate into, because that one is not servable at any
 * max_tokens. */
TEST(a_max_tokens_past_the_context_is_clamped_to_what_is_left) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_ctx = 64;

    /* Chat. The assertion is against the RENDERED length rather than a constant: the prompt a
     * template produces is not the content that went into it, and a test that hard-codes the
     * difference breaks on the next change to the fake template rather than on a real one. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_chat_request(
            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":9999})",
            f.odeps(), lim, &r, &e));
        CHECK_EQ((int64_t)r.prompts[0].tokens.size() + r.max_tokens, lim.max_ctx);
    }

    /* A request that already fits is left exactly as it asked. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"ab","max_tokens":3})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.max_tokens, 3);
    }

    /* A batch clamps against its LONGEST prompt, not its last: one max_tokens covers them all. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":["a","abcde"],"max_tokens":9999})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ((int)r.max_tokens, 64 - 5);
    }

    /* And a prompt that does not fit on its own is still a 400 that says so. */
    {
        OaiRequest r; ApiError e;
        std::string body = R"({"prompt":")" + std::string(70, 'x') + R"("})";
        CHECK_EQ(parse_completion_request(body, f.odeps(), lim, &r, &e), RAD_E_INVAL);
        CHECK_EQ(e.code, std::string("context_length_exceeded"));
    }
}

/* A max_tokens PAST 32 BITS IS REFUSED, NOT NARROWED. Narrowed, 2^31 is negative and 2^32 is zero,
 * and the scheduler reads either as "no limit" -- a request that asked for a bound got none. The
 * limits here declare no context, so nothing downstream could clamp it back either. */
TEST(a_max_tokens_past_32_bits_is_refused_rather_than_wrapped) {
    Fixture f;
    OaiLimits lim = f.limits();

    for (const char* v : {"2147483648", "4294967296", "4294967297", "18446744073709551615"}) {
        OaiRequest r; ApiError e;
        std::string body = std::string(R"({"prompt":"a","max_tokens":)") + v + "}";
        CHECK(parse_completion_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("max_tokens"));
    }
    /* Both spellings: the chat endpoint's newer one names itself. */
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],
                                     "max_completion_tokens":2147483648})",
                                 f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("max_completion_tokens"));
    }
    /* The largest value that fits is still a value. */
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":"a","max_tokens":2147483647})",
                                          f.odeps(), lim, &r, &e));
        CHECK_EQ(r.max_tokens, (int32_t)2147483647);
    }
    /* And over HTTP it is the 400, not an admitted request. */
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"a","max_tokens":4294967296})").status, 400);
    CHECK(f.sched.submitted.empty());
}

/* A TOKEN ID IS AN INT32 OR IT IS REFUSED. Narrowed, 2^32 + 5 is token 5 -- a prompt that is
 * silently not the one that was sent. */
TEST(a_token_id_outside_int32_is_refused_rather_than_aliased) {
    Fixture f;
    OaiLimits lim = f.limits();
    for (const char* body : {R"({"prompt":[104,4294967301]})", R"({"prompt":[-1]})",
                             R"({"prompt":[[104],[2147483648]]})",
                             R"({"prompt":[18446744073709551615]})"}) {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("prompt"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":[4294967301]})", f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("input"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(R"({"prompt":[104,2147483647]})", f.odeps(), lim, &r,
                                          &e));
        REQUIRE_EQ(r.prompts.size(), (size_t)1);
        REQUIRE_EQ(r.prompts[0].tokens.size(), (size_t)2);
        CHECK_EQ(r.prompts[0].tokens[1], (int32_t)2147483647);
    }
}

/* A FIELD OF THE WRONG TYPE IS THE CALLER'S MISTAKE, so it is a 400 that names it -- never an
 * exception out of the JSON library, which the transport can only report as a 500. */
TEST(a_wrongly_typed_field_is_a_400_not_an_exception) {
    Fixture f;
    FakeMultimodal mm;
    OaiDeps d = f.odeps();
    d.mm = &mm;
    OaiLimits lim = f.limits();
    lim.allow_image = true;
    lim.allow_audio = true;

    const char* bodies[] = {
        R"({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":5,"function":{"name":"f"}}]})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":7}}]})",
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":5}}]}]})",
        R"({"messages":[{"role":"user","content":[{"type":"input_audio",
            "input_audio":{"data":"AAAA","format":5}}]}]})",
        R"({"messages":[{"role":"user","content":[{"type":"input_audio",
            "input_audio":{"data":5,"format":"wav"}}]}]})",
    };
    for (const char* body : bodies) {
        OaiRequest r; ApiError e;
        int rc = RAD_OK;
        bool threw = false;
        try { rc = parse_chat_request(body, d, lim, &r, &e); } catch (...) { threw = true; }
        CHECK(!threw);
        CHECK(rc < 0);
        CHECK_EQ(e.status, 400);
    }
}

/* EVERY STOP STRING IS SCANNED FOR IN EVERY STRETCH OF OUTPUT, so the list is bounded in count and
 * in length, and a list past either bound is refused rather than paid for on every token. */
TEST(a_stop_list_past_its_bounds_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    auto body_with = [](size_t n, size_t len) {
        json b;
        b["prompt"] = "a";
        b["stop"] = json::array();
        for (size_t i = 0; i < n; ++i) b["stop"].push_back(std::string(len, (char)('a' + i % 26)));
        return b.dump();
    };
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(body_with(64, 4096), f.odeps(), lim, &r, &e));
        CHECK_EQ(r.stop.size(), (size_t)64);
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body_with(65, 1), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(body_with(1, 4097), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
    {
        OaiRequest r; ApiError e;
        const std::string one = R"({"prompt":"a","stop":")" + std::string(4097, 'z') + R"("})";
        CHECK(parse_completion_request(one, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.param, std::string("stop"));
    }
}

/* THE PARSED TREE IS BOUNDED, NOT ONLY THE BODY: `[1,1,...]` is two bytes a value on the wire and
 * several times that once parsed. One value past the budget is refused, on the generation
 * endpoints and on /detokenize, whose body is exactly such an array. */
TEST(a_body_holding_more_values_than_the_budget_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    const int64_t budget = json_value_budget(lim.max_ctx);
    CHECK(json_value_budget(1000) > budget);   /* it scales with the context */

    /* The object and the array are values too, so `inner` scalars make inner + 2 in all. */
    auto ids = [](const char* key, int64_t inner) {
        std::string s = std::string("{\"") + key + "\":[";
        s.reserve((size_t)inner * 2 + 32);
        for (int64_t i = 0; i < inner; ++i) { if (i) s += ','; s += '7'; }
        return s + "]}";
    };
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_completion_request(ids("prompt", budget - 2), f.odeps(), lim, &r, &e));
        REQUIRE_EQ(r.prompts.size(), (size_t)1);
        CHECK_EQ((int64_t)r.prompts[0].tokens.size(), budget - 2);
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request(ids("prompt", budget - 1), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.message, "JSON values"));
    }
    CHECK_EQ(f.post("/detokenize", ids("tokens", budget - 1)).status, 400);
    CHECK_EQ(f.post("/detokenize", ids("tokens", 3)).status, 200);
}

/* A serialiser recurses once a nesting level, so depth is bounded apart from the value count: a
 * few hundred kilobytes of brackets is far inside the value budget and far past a worker's stack.
 * The bound covers the body itself and the strings the chat template parses as JSON. */
TEST(a_body_or_template_json_nested_past_the_bound_is_refused) {
    Fixture f;
    OaiLimits lim = f.limits();
    auto nest = [](int depth) {
        return std::string((size_t)depth, '[') + "1" + std::string((size_t)depth, ']');
    };
    auto quoted = [](const std::string& s) { return "\"" + s + "\""; };
    const int deep = 200000, fine = kJsonMaxDepth - 8;
    {
        OaiRequest r; ApiError e;
        CHECK(parse_completion_request("{\"prompt\":\"x\",\"logit_bias\":" + nest(deep) + "}",
                                       f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.message, "levels deep"));
    }
    CHECK_EQ(f.post("/detokenize", "{\"tokens\":" + nest(deep) + "}").status, 400);

    auto call = [&](const std::string& args) {
        return R"({"messages":[{"role":"user","content":"hi"},
                   {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"f","arguments":)" + quoted(args) + R"(}}]},
                   {"role":"tool","tool_call_id":"c1","content":"ok"}]})";
    };
    auto result = [&](const std::string& content) {
        return R"({"messages":[{"role":"user","content":"hi"},
                   {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
                    "function":{"name":"f","arguments":"{}"}}]},
                   {"role":"tool","tool_call_id":"c1","content":)" + quoted(content) + "}]}";
    };
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(call(nest(deep)), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK(has(e.param, "arguments"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_chat_request(result(nest(deep)), f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
    }
    /* Brackets inside a string literal of the argument text are not nesting. */
    {
        OaiRequest r; ApiError e;
        const std::string in_str = "{\\\"s\\\":\\\"" + std::string(5000, '[') + "\\\"}";
        CHECK_OK(parse_chat_request(call(in_str), f.odeps(), lim, &r, &e));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_chat_request(call(nest(fine)), f.odeps(), lim, &r, &e));
        CHECK_OK(parse_chat_request(result(nest(fine)), f.odeps(), lim, &r, &e));
    }
}

/* ================================================================== the 400 path */

TEST(unsupported_parameters_are_400_and_name_themselves) {
    Fixture f;

    /* A parameter we know about and do not implement. */
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "logit_bias":{"5":10}})");
    CHECK_EQ(r.status, 400);
    json e = json::parse(r.body);
    CHECK_EQ(e["error"]["param"].get<std::string>(), std::string("logit_bias"));
    CHECK_EQ(e["error"]["code"].get<std::string>(), std::string("unsupported_parameter"));
    CHECK(e["error"]["message"].get<std::string>().find("logit_bias") != std::string::npos);

    /* A parameter nobody has heard of is a client's own extension, and is ignored as llama.cpp
     * and vLLM ignore it (#47: OpenRouter's session_id, refused, broke the client). */
    HttpResponse r2 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "wobble":1,"session_id":"abc"})");
    CHECK_EQ(r2.status, 200);

    /* What llama.cpp-built clients send on every request: the samplers this server does not run,
     * at their off values. Accepted there, and refused by name the moment one is switched on. */
    HttpResponse off = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "mirostat":0,"top_n_sigma":-1,"dynatemp_range":0,
                                 "logit_bias":{},"cache_prompt":true,"id_slot":-1,
                                 "backend_sampling":false,"speculative.n_max":16})");
    CHECK_EQ(off.status, 200);
    HttpResponse on = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],"mirostat":2})");
    CHECK_EQ(on.status, 400);
    CHECK_EQ(json::parse(on.body)["error"]["param"].get<std::string>(), std::string("mirostat"));

    /* Out-of-range values name the field and the range. */
    HttpResponse r3 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "temperature":9})");
    CHECK_EQ(r3.status, 400);
    CHECK(r3.body.find("temperature") != std::string::npos);

    /* logprobs without sampler support is refused rather than answered with nulls. */
    HttpResponse r5 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],
                                 "logprobs":true})");
    CHECK_EQ(r5.status, 400);
    CHECK(r5.body.find("logprobs") != std::string::npos);

    /* A body that is not an object at all. */
    HttpResponse r6 = f.post("/v1/chat/completions", "[]");
    CHECK_EQ(r6.status, 400);
}

TEST(unknown_routes_and_methods_say_which) {
    Fixture f;
    HttpResponse r = f.get("/v1/nope");
    CHECK_EQ(r.status, 404);
    CHECK(r.body.find("/v1/nope") != std::string::npos);
    HttpResponse r2 = f.get("/v1/chat/completions");
    CHECK_EQ(r2.status, 405);
}

/* ================================================================== non-streaming shape */

TEST(non_streaming_chat_response_has_the_openai_shape) {
    Fixture f;
    f.sched.reply = "hello";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(!r.streaming());

    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("chat.completion"));
    CHECK(j["id"].get<std::string>().rfind("chatcmpl-", 0) == 0);
    CHECK_EQ(j["model"].get<std::string>(), std::string("test-model"));
    CHECK(j["created"].is_number_integer());
    CHECK_EQ((int)j["choices"].size(), 1);

    const auto& c = j["choices"][0];
    CHECK_EQ(c["index"].get<int>(), 0);
    CHECK_EQ(c["message"]["role"].get<std::string>(), std::string("assistant"));
    CHECK_EQ(c["message"]["content"].get<std::string>(), std::string("hello"));
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(c["logprobs"].is_null());

    CHECK_EQ(j["usage"]["completion_tokens"].get<int>(), 5);
    CHECK(j["usage"]["prompt_tokens"].get<int>() > 0);
    CHECK_EQ(j["usage"]["total_tokens"].get<int>(),
             j["usage"]["prompt_tokens"].get<int>() + 5);
    CHECK(j["usage"]["prompt_tokens_details"]["cached_tokens"].is_number());
}

TEST(non_streaming_completion_response_has_the_openai_shape) {
    Fixture f;
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","max_tokens":8})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("text_completion"));
    CHECK_EQ(j["choices"][0]["text"].get<std::string>(), std::string("abc"));
    CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(j["choices"][0]["logprobs"].is_null());
}

TEST(n_greater_than_one_fans_out_into_one_choice_each) {
    Fixture f;
    f.sched.reply = "xy";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"n":3})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ((int)j["choices"].size(), 3);
    CHECK_EQ(j["choices"][2]["index"].get<int>(), 2);
    CHECK_EQ((int)f.sched.submitted.size(), 3);
}

TEST(a_stop_string_is_cut_out_of_the_answer) {
    Fixture f;
    f.sched.reply = "abENDcd";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(), std::string("ab"));
    CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
    /* Cutting the answer frees the blocks immediately rather than at completion (spec §14). */
    CHECK(f.sched.saw_cancel());
}

TEST(a_complete_tool_call_is_reported_as_tool_calls) {
    Fixture f;
    f.sched.reply = "TOOL:get_weather:{\"city\":\"berlin\"}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    const auto& c = j["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("tool_calls"));
    CHECK_EQ((int)c["message"]["tool_calls"].size(), 1);
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["name"].get<std::string>(),
             std::string("get_weather"));
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["arguments"].get<std::string>(),
             std::string("{\"city\":\"berlin\"}"));
    CHECK(c["message"]["content"].is_null());
}

/* The three spellings of "do not think" all have to reach the template, and the kwarg has to
 * reach BOTH places: the lift overwrites the context key `enable_thinking` with its own field
 * (chat-auto-parser-helpers.cpp), so a kwarg that is not also lifted into the field is discarded
 * and the caller gets a thinking model back having asked for the opposite. */
TEST(the_ways_of_turning_thinking_off_all_reach_the_template) {
    for (const char* body : {
             R"({"messages":[{"role":"user","content":"hi"}],
                 "chat_template_kwargs":{"enable_thinking":false}})",
             R"({"messages":[{"role":"user","content":"hi"}],"enable_thinking":false})",
             R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"none"})" }) {
        Fixture f;
        CHECK_EQ(f.post("/v1/chat/completions", body).status, 200);
        CHECK(!f.tmpl.last_opt.enable_thinking);
    }
    /* ...and the default is left alone. */
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);
    CHECK(f.tmpl.last_opt.enable_thinking);
}

/* ================================================================== deployment defaults
 *
 * Every default the command line can move, seen from a request: it reaches a request that leaves
 * the field out, and loses to one that names it. */

static OaiRequest chat_with(Fixture& f, const OaiLimits& lim, const std::string& extra) {
    OaiRequest r;
    ApiError e;
    const std::string body =
        R"({"messages":[{"role":"user","content":"hi"}])" + (extra.empty() ? "" : "," + extra) + "}";
    CHECK_OK(parse_chat_request(body, f.odeps(), lim, &r, &e));
    return r;
}

TEST(the_rest_of_the_sampler_is_seeded_from_the_deployment_too) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_sampling.pres_penalty = 1.5f;
    lim.default_sampling.rep_penalty = 1.05f;
    lim.default_sampling.penalty_last_n = 256;
    lim.default_sampling.typical_p = 0.9f;
    lim.default_sampling.dry_multiplier = 0.8f;
    lim.default_sampling.xtc_probability = 0.5f;

    OaiRequest r = chat_with(f, lim, "");
    CHECK_NEAR(r.sp.pres_penalty, 1.5f, 1e-6);
    CHECK_NEAR(r.sp.rep_penalty, 1.05f, 1e-6);
    CHECK_EQ(r.sp.penalty_last_n, 256);
    CHECK_NEAR(r.sp.typical_p, 0.9f, 1e-6);
    CHECK_NEAR(r.sp.dry_multiplier, 0.8f, 1e-6);
    CHECK_NEAR(r.sp.xtc_probability, 0.5f, 1e-6);

    OaiRequest r2 = chat_with(f, lim, R"("presence_penalty":0,"repeat_last_n":64)");
    CHECK_NEAR(r2.sp.pres_penalty, 0.0f, 1e-6);
    CHECK_EQ(r2.sp.penalty_last_n, 64);
    CHECK_NEAR(r2.sp.rep_penalty, 1.05f, 1e-6);

    /* A STATED EMPTY BREAKER LIST STAYS EMPTY, which SamplingParams alone cannot say: unstated is
     * llama-server's four. A request's own list still wins. */
    lim.default_sampling.dry_seq_breakers.clear();
    lim.default_dry_breakers_set = true;
    CHECK(chat_with(f, lim, "").sp.dry_seq_breakers.empty());
    CHECK(chat_with(f, lim, R"("dry_sequence_breakers":["x"])").sp.dry_seq_breakers ==
          std::vector<std::string>({ "x" }));
    lim.default_sampling.dry_seq_breakers = { "\n" };
    CHECK(chat_with(f, lim, "").sp.dry_seq_breakers == std::vector<std::string>({ "\n" }));
}

/* A sampler field's range is ONE table for the request and the flag; this holds the request half
 * of it to the exact sentences it has always carried. */
TEST(the_shared_sampler_ranges_refuse_in_the_words_a_request_always_got) {
    struct Case { const char* body; const char* param; const char* msg; };
    const Case cases[] = {
        { R"("temperature":2.5)",        "temperature",        "must be in [0, 2], got 2.5" },
        { R"("top_p":0)",                "top_p",              "must be in (0, 1]" },
        { R"("top_k":-1)",               "top_k",              "must be >= 0 (0 disables it)" },
        { R"("min_p":1.5)",              "min_p",              "must be in [0, 1], got 1.5" },
        { R"("typical_p":0)",            "typical_p",          "must be in (0, 1]" },
        { R"("presence_penalty":3)",     "presence_penalty",   "must be in [-2, 2], got 3" },
        { R"("frequency_penalty":-3)",   "frequency_penalty",  "must be in [-2, 2], got -3" },
        { R"("repetition_penalty":0)",   "repetition_penalty", "must be > 0" },
        { R"("repeat_last_n":-2)",       "repeat_last_n",      "must be >= -1" },
        { R"("xtc_probability":2)",      "xtc_probability",    "must be in [0, 1], got 2" },
        { R"("xtc_threshold":-1)",       "xtc_threshold",      "must be in [0, 1], got -1" },
        { R"("dry_base":0.5)",           "dry_base",           "must be >= 1" },
        { R"("dry_penalty_last_n":-2)",  "dry_penalty_last_n", "must be >= -1" },
    };
    Fixture f;
    for (const Case& c : cases) {
        OaiRequest r;
        ApiError e;
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":"hi"}],)") + c.body + "}";
        CHECK_EQ(parse_chat_request(body, f.odeps(), f.limits(), &r, &e), RAD_E_INVAL);
        CHECK_EQ(e.param, std::string(c.param));
        CHECK_EQ(e.message, std::string(c.param) + ": " + c.msg);
    }
    /* The other half: the same function, as config.cpp calls it. */
    std::string why;
    CHECK(sampler_value_ok("presence_penalty", -2.0, &why));
    CHECK(!sampler_value_ok("presence_penalty", std::nan(""), &why));
    CHECK(!sampler_value_ok("temperature", std::nan(""), &why));
    CHECK(sampler_value_ok("dry_multiplier", -5.0, &why));     /* unbounded, as a request is */
}

/* `--default-max-tokens auto` IS WHAT THE CONTEXT LEAVES, and `--max-tokens-cap` CLAMPS like the
 * context does: a request asking for more is served up to the cap, not refused. */
TEST(the_max_tokens_default_and_cap_bound_a_request_like_the_context) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_ctx = 64;
    lim.default_max_tokens = 0;

    OaiRequest r = chat_with(f, lim, "");
    CHECK_EQ((int64_t)r.prompts[0].tokens.size() + r.max_tokens, lim.max_ctx);
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":5)").max_tokens, 5);

    lim.max_tokens_cap = 10;
    CHECK_EQ(chat_with(f, lim, "").max_tokens, 10);                     /* auto, then the cap */
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":9999)").max_tokens, 10);
    CHECK_EQ(chat_with(f, lim, R"("max_tokens":5)").max_tokens, 5);

    /* AND `auto` IS THE DEFAULT: limits built with nothing said about max_tokens give a request
     * that names none what the context leaves. 512 ended a reasoning model's answers inside its
     * reasoning, with a finish_reason "length" a chat client does not show. */
    {
        OaiLimits d;
        d.model_id = lim.model_id;
        d.max_ctx = 64;
        OaiRequest a = chat_with(f, d, "");
        CHECK_EQ((int64_t)a.prompts[0].tokens.size() + a.max_tokens, d.max_ctx);
    }

    /* The cap holds where no context is declared, and on /v1/completions. */
    lim.max_ctx = 0;
    lim.default_max_tokens = 512;
    CHECK_EQ(chat_with(f, lim, "").max_tokens, 10);
    OaiRequest c;
    ApiError e;
    CHECK_OK(parse_completion_request(R"({"prompt":"ab","max_tokens":100})", f.odeps(), lim, &c, &e));
    CHECK_EQ(c.max_tokens, 10);

    /* Unset, nothing moves: the fixture's default with no cap and no context. */
    CHECK_EQ(chat_with(f, f.limits(), "").max_tokens, 32);
    CHECK_EQ(chat_with(f, f.limits(), R"("max_tokens":100000)").max_tokens, 100000);
}

TEST(the_stop_bounds_are_the_deployments) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.max_stops = 2;
    lim.max_stop_bytes = 3;
    OaiRequest r;
    ApiError e;
    CHECK_OK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":["a","bcd"]})",
                                f.odeps(), lim, &r, &e));
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":["a","b","c"]})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK(has(e.message, "at most 2 stop strings"));
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"stop":"abcd"})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK(has(e.message, "at most 3 bytes"));
}

/* THE DEPLOYMENT'S TEMPLATE VARIABLES go under the request's, key by key. */
TEST(the_deployments_template_variables_reach_the_template_under_the_requests) {
    Fixture f;
    OaiLimits lim = f.limits();
    lim.default_template_kwargs = { { "custom", "\"a\"" }, { "depth", "2" } };

    chat_with(f, lim, "");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("custom"), std::string("\"a\""));
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("depth"), std::string("2"));

    chat_with(f, lim, R"("chat_template_kwargs":{"custom":"b"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("custom"), std::string("\"b\""));
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("depth"), std::string("2"));

    lim.reasoning_format = "none";
    chat_with(f, lim, "");
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("none"));
    chat_with(f, f.limits(), "");
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("auto"));
    CHECK(f.tmpl.last_opt.template_kwargs.empty());

    /* /tokenize renders with the same variables, so the prompt it counts is the prompt chat
     * serves -- and with none set it renders exactly as before. */
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    const std::string body = R"({"messages":[{"role":"user","content":"hi"}]})";
    {
        ServerOptions o;
        o.default_template_kwargs = { { "enable_thinking", "false" }, { "custom", "\"a\"" } };
        Server srv(d, o);
        CHECK_EQ(srv.router().call("POST", "/tokenize", body).status, 200);
        CHECK(!tmpl.last_opt.enable_thinking);
        CHECK_EQ(tmpl.last_opt.template_kwargs.at("custom"), std::string("\"a\""));
        CHECK_EQ(srv.router().call("POST", "/tokenize",
                                   R"({"messages":[{"role":"user","content":"hi"}],
                                       "chat_template_kwargs":{"enable_thinking":true}})").status, 200);
        CHECK_EQ(tmpl.last_opt.template_kwargs.at("enable_thinking"), std::string("true"));
    }
    {
        ServerOptions o;
        Server srv(d, o);
        CHECK_EQ(srv.router().call("POST", "/tokenize", body).status, 200);
        CHECK(tmpl.last_opt.enable_thinking);
        CHECK(tmpl.last_opt.template_kwargs.empty());
    }
}

/* THINKING IS ONE SETTING WITH THREE SPELLINGS, and a deployment default in one spelling must not
 * outlive a request that used another. */
TEST(a_request_that_says_anything_about_thinking_replaces_the_deployments_default) {
    Fixture f;
    OaiLimits off = f.limits();
    off.default_template_kwargs = { { "enable_thinking", "false" } };

    chat_with(f, off, "");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("enable_thinking"), std::string("false"));

    for (const char* extra : { R"("enable_thinking":true)",
                               R"("chat_template_kwargs":{"enable_thinking":true})",
                               R"("reasoning_effort":"high")",
                               R"("chat_template_kwargs":{"reasoning_effort":"high"})" }) {
        chat_with(f, off, extra);
        CHECK(f.tmpl.last_opt.enable_thinking);
    }
    chat_with(f, off, R"("reasoning_effort":"high")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"high\""));
    CHECK(!f.tmpl.last_opt.template_kwargs.count("enable_thinking"));
    /* An empty effort is no effort, and leaves the default standing. */
    chat_with(f, off, R"("reasoning_effort":"")");
    CHECK(!f.tmpl.last_opt.enable_thinking);

    /* --reasoning-effort none is the same default in its other spelling. */
    OaiLimits none = f.limits();
    none.default_reasoning_effort = "none";
    chat_with(f, none, "");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    chat_with(f, none, R"("enable_thinking":true)");
    CHECK(f.tmpl.last_opt.enable_thinking);
    chat_with(f, none, R"("reasoning_effort":"low")");
    CHECK(f.tmpl.last_opt.enable_thinking);

    /* A LEVEL is an effort default: a request that only turns thinking on keeps it, as before,
     * and one that names its own effort replaces it. */
    OaiLimits low = f.limits();
    low.default_reasoning_effort = "low";
    chat_with(f, low, R"("enable_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));
    chat_with(f, low, R"("reasoning_effort":"xhigh")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, low, R"("chat_template_kwargs":{"reasoning_effort":"xhigh"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));

    OaiLimits kw_low = f.limits();
    kw_low.default_template_kwargs = { { "reasoning_effort", "\"low\"" } };
    chat_with(f, kw_low, R"("enable_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));
    chat_with(f, kw_low, R"("reasoning_effort":"xhigh")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
}

/* A TEMPLATE DEFAULT THE TEMPLATE REFUSES is found at startup, and attributed to the flag that
 * set it. A render that fails without any default blames nothing. */
TEST(a_template_default_the_template_refuses_is_found_before_a_caller_sends_one) {
    auto run = [](const std::map<std::string, std::string>& kw, const std::string& effort,
                  const std::string& refuse, std::string* kw_why, std::string* eff_why) {
        FakeScheduler sched;
        FakeTokenizer tok;
        FakeChatTemplate tmpl;
        tmpl.refuse_kwarg = refuse;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        ServerOptions o;
        o.default_template_kwargs = kw;
        o.default_reasoning_effort = effort;
        Server srv(d, o);
        srv.check_template_defaults(kw_why, eff_why);
    };
    std::string kw_why, eff_why;
    run({ { "bad", "1" } }, "", "bad", &kw_why, &eff_why);
    CHECK(has(kw_why, "does not take bad"));
    CHECK(eff_why.empty());

    run({ { "fine", "1" } }, "high", "reasoning_effort", &kw_why, &eff_why);
    CHECK(kw_why.empty());
    CHECK(has(eff_why, "does not take reasoning_effort"));

    run({ { "fine", "1" } }, "high", "", &kw_why, &eff_why);
    CHECK(kw_why.empty() && eff_why.empty());
    /* "none" is enable_thinking false, never a template variable, so nothing to refuse. */
    run({}, "none", "reasoning_effort", &kw_why, &eff_why);
    CHECK(kw_why.empty() && eff_why.empty());
}

/* AN EFFORT THE TEMPLATE REFUSES IS NAMED AS ONE. The template's own set is probed at startup, a
 * request outside it is answered as OpenAI answers an unsupported value -- `reasoning_effort`,
 * `unsupported_value`, the set in `supported` -- and /server_info publishes the set beforehand.
 * Codeberg #31: the 400 blamed `messages` and carried the set only inside a Jinja traceback. */
TEST(an_effort_the_template_refuses_is_answered_with_the_set_it_takes) {
    Fixture f;
    f.tmpl.efforts = { "\"xhigh\"", "\"medium\"", "\"low\"" };
    const std::vector<std::string> set = f.srv->probe_reasoning_efforts();
    REQUIRE_EQ(set.size(), 3u);
    CHECK_EQ(set[0], std::string("low"));
    CHECK_EQ(set[1], std::string("medium"));
    CHECK_EQ(set[2], std::string("xhigh"));

    OaiLimits lim = f.limits();
    lim.reasoning_efforts = set;
    for (const char* extra : { R"("reasoning_effort":"high")",
                               R"("chat_template_kwargs":{"reasoning_effort":"high"})" }) {
        OaiRequest r;
        ApiError e;
        const std::string body =
            std::string(R"({"messages":[{"role":"user","content":"hi"}],)") + extra + "}";
        CHECK(parse_chat_request(body, f.odeps(), lim, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
        CHECK_EQ(e.param, std::string("reasoning_effort"));
        CHECK_EQ(e.code, std::string("unsupported_value"));
        CHECK(!has(e.message, "Jinja"));
        CHECK(has(e.message, "'xhigh'"));
        const json env = json::parse(e.dump());
        REQUIRE(env["error"]["supported"].is_array());
        CHECK_EQ(env["error"]["supported"].size(), 5u);
        CHECK_EQ(env["error"]["supported"][0].get<std::string>(), std::string("none"));
        CHECK_EQ(env["error"]["supported"][4].get<std::string>(), std::string("xhigh"));
    }
    chat_with(f, lim, R"("reasoning_effort":"xhigh")");
    chat_with(f, lim, R"("reasoning_effort":"none")");

    const json j = json::parse(f.srv->router().call("GET", "/server_info").body);
    const json& caps = j["capabilities"]["reasoning_efforts"];
    REQUIRE(caps.is_array());
    CHECK_EQ(caps.size(), 5u);
    CHECK_EQ(caps[2].get<std::string>(), std::string("low"));
}

/* A TEMPLATE THAT TAKES ANY WORD has no set to publish or to refuse with, and a template that
 * refuses for another reason keeps blaming `messages` with its own sentence. */
TEST(a_template_that_takes_any_effort_publishes_no_set) {
    Fixture f;
    CHECK(f.srv->probe_reasoning_efforts().empty());
    const json j = json::parse(f.srv->router().call("GET", "/server_info").body);
    CHECK(j["capabilities"]["reasoning_efforts"].is_null());

    f.tmpl.refuse_kwarg = "custom";
    OaiLimits lim = f.limits();
    lim.reasoning_efforts = { "low" };
    OaiRequest r;
    ApiError e;
    CHECK(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],
                                 "reasoning_effort":"low","chat_template_kwargs":{"custom":1}})",
                             f.odeps(), lim, &r, &e) < 0);
    CHECK_EQ(e.param, std::string("messages"));
    CHECK(has(e.message, "does not take custom"));
}

/* --reasoning-effort-map: the deployment's word replaces the request's before the template sees
 * it, in either spelling; a word mapped to "none" turns thinking off; a target the template
 * refuses is found at startup. */
TEST(a_mapped_effort_reaches_the_template_as_the_deployments_word) {
    Fixture f;
    f.tmpl.efforts = { "\"xhigh\"", "\"medium\"", "\"low\"" };
    OaiLimits lim = f.limits();
    lim.reasoning_effort_map = { { "high", "xhigh" }, { "minimal", "low" }, { "max", "none" } };
    chat_with(f, lim, R"("reasoning_effort":"high")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, lim, R"("chat_template_kwargs":{"reasoning_effort":"high"})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"xhigh\""));
    chat_with(f, lim, R"("reasoning_effort":"medium")");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"medium\""));
    chat_with(f, lim, R"("reasoning_effort":"max")");
    CHECK(!f.tmpl.last_opt.enable_thinking);
    CHECK(!f.tmpl.last_opt.template_kwargs.count("reasoning_effort"));
    /* The map is read before "none" and "minimal" are, so it can name either. */
    chat_with(f, lim, R"("reasoning_effort":"minimal")");
    CHECK(f.tmpl.last_opt.enable_thinking);
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("reasoning_effort"), std::string("\"low\""));

    auto check = [&](const std::map<std::string, std::string>& m) {
        FakeScheduler sched;
        FakeTokenizer tok;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &f.tmpl;
        ServerOptions o;
        o.reasoning_effort_map = m;
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK_EQ(j["request_defaults"]["reasoning_effort_map"].size(), m.size());
        return srv.check_reasoning_effort_map();
    };
    CHECK(check({ { "high", "xhigh" }, { "max", "none" } }).empty());
    CHECK(has(check({ { "high", "ultra" } }), "high=ultra"));
}

TEST(server_info_reports_what_a_request_that_leaves_a_field_out_gets) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    {
        ServerOptions o;
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK(j["server"]["default_max_tokens"].is_null());        /* auto */
        CHECK(j["server"]["max_tokens_cap"].is_null());
        CHECK_EQ(j["server"]["max_stop_strings"].get<int64_t>(), 64);
        const json& r = j["request_defaults"];
        CHECK_NEAR(r["temperature"].get<double>(), 1.0, 1e-9);
        CHECK_NEAR(r["repetition_penalty"].get<double>(), 1.0, 1e-9);
        CHECK_EQ(r["dry_sequence_breakers"].size(), 4u);
        CHECK(r["reasoning_effort"].is_null());
        CHECK(r["chat_template_kwargs"].is_object() && r["chat_template_kwargs"].empty());
        CHECK_EQ(r["reasoning_format"].get<std::string>(), std::string("auto"));
    }
    {
        ServerOptions o;
        o.default_max_tokens = 0;
        o.max_tokens_cap = 4096;
        o.default_sampling.pres_penalty = 1.5f;
        o.default_template_kwargs = { { "enable_thinking", "false" }, { "tag", "\"x\"" } };
        Server srv(d, o);
        const json j = json::parse(srv.router().call("GET", "/server_info").body);
        CHECK(j["server"]["default_max_tokens"].is_null());          /* auto */
        CHECK_EQ(j["server"]["max_tokens_cap"].get<int64_t>(), 4096);
        const json& r = j["request_defaults"];
        CHECK_NEAR(r["presence_penalty"].get<double>(), 1.5, 1e-6);
        CHECK_EQ(r["chat_template_kwargs"]["enable_thinking"].get<bool>(), false);
        CHECK_EQ(r["chat_template_kwargs"]["tag"].get<std::string>(), std::string("x"));
    }
}

/* `preserve_thinking` AT THE TOP LEVEL IS THE TEMPLATE VARIABLE, as `enable_thinking` is. oh-my-pi
 * sends it there and in chat_template_kwargs on every request to a Qwen model, and refusing it as
 * an unknown field failed all of them. */
TEST(preserve_thinking_at_the_top_level_reaches_the_template) {
    Fixture f;
    OaiLimits lim = f.limits();
    chat_with(f, lim, R"("preserve_thinking":true)");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("true"));
    CHECK(f.tmpl.last_opt.enable_thinking);                /* a history knob, not a thinking one */

    /* oh-my-pi's exact pair, and the top-level spelling winning a disagreement. */
    chat_with(f, lim, R"("preserve_thinking":true,"chat_template_kwargs":{"preserve_thinking":true})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("true"));
    chat_with(f, lim, R"("preserve_thinking":false,"chat_template_kwargs":{"preserve_thinking":true})");
    CHECK_EQ(f.tmpl.last_opt.template_kwargs.at("preserve_thinking"), std::string("false"));

    chat_with(f, lim, "");
    CHECK(!f.tmpl.last_opt.template_kwargs.count("preserve_thinking"));

    OaiRequest r;
    ApiError e;
    CHECK_EQ(parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"preserve_thinking":"yes"})",
                                f.odeps(), lim, &r, &e), RAD_E_INVAL);
    CHECK_EQ(e.param, std::string("preserve_thinking"));
    /* A chat field, not a completion one: /v1/completions has no template to hand it to, and
     * ignores it as it ignores any field it does not read. */
    CHECK_EQ(parse_completion_request(R"({"prompt":"ab","preserve_thinking":true})", f.odeps(), lim,
                                      &r, &e), RAD_OK);
}

/* parallel_tool_calls reaches the template, because it is the one tool-choice knob the generated
 * grammar actually honours -- it narrows the repetition rule rather than being advice. */
TEST(parallel_tool_calls_false_reaches_the_template) {
    Fixture f;
    f.sched.reply = "TOOL:a:{}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"a"}}],
                                "parallel_tool_calls":false})");
    CHECK_EQ(r.status, 200);
    CHECK(!f.tmpl.last_opt.parallel_tool_calls);
    CHECK_EQ(f.tmpl.last_n_tools, 1);
}

/* A forced choice reaches the template as "required", with the function it names; the grammar
 * that holds the reply to it is built from that (chatparse_test). */
TEST(a_forced_tool_choice_reaches_the_template) {
    Fixture f;
    f.sched.reply = "TOOL:a:{}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                "tools":[{"type":"function","function":{"name":"a"}}],
                                "tool_choice":"required"})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.tmpl.last_opt.tool_choice, std::string("required"));
    CHECK(f.tmpl.last_opt.tool_name.empty());

    HttpResponse r2 = f.post("/v1/chat/completions",
                             R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":128,
                                 "tools":[{"type":"function","function":{"name":"a"}},
                                          {"type":"function","function":{"name":"b"}}],
                                 "tool_choice":{"type":"function","function":{"name":"b"}}})");
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(f.tmpl.last_opt.tool_choice, std::string("required"));
    CHECK_EQ(f.tmpl.last_opt.tool_name, std::string("b"));
    CHECK_EQ(f.tmpl.last_n_tools, 2);
}

/* What cannot be forced is a 400 naming tool_choice: a call with no tools, a function the request
 * does not offer, and a call that would have to share the whole reply with a document. */
TEST(a_forced_tool_choice_that_cannot_be_kept_is_refused) {
    Fixture f;
    auto refused = [&](const std::string& body) {
        HttpResponse r = f.post("/v1/chat/completions", body);
        return r.status == 400 && r.body.find("tool_choice") != std::string::npos;
    };
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"})"));
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],
                      "tools":[{"type":"function","function":{"name":"a"}}],
                      "tool_choice":{"type":"function","function":{"name":"zz"}}})"));
    CHECK(refused(R"({"messages":[{"role":"user","content":"hi"}],
                      "tools":[{"type":"function","function":{"name":"a"}}],
                      "tool_choice":"required","response_format":{"type":"json_object"}})"));
}

/* A grammar the compiler cannot run is a 400 at admission, not a request that dies mid-stream. */
TEST(a_grammar_that_cannot_be_run_is_refused_at_the_door) {
    Fixture f;
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "grammar":"root ::= bad"})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("grammar") != std::string::npos);
}

/* The same on /v1/completions, which has no template in between: the grammar is the client's own
 * text, and the first place it can be refused is here. */
TEST(a_completion_grammar_that_cannot_be_run_is_refused_at_the_door) {
    Fixture f;
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","grammar":"root ::= bad"})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("grammar") != std::string::npos);
}

/* A model whose tool calls the reply parser cannot describe still chats, and a request that sends
 * tools is refused at the door with the reason -- never answered with calls read by some other
 * parser, and never with prose where a call was meant. */
TEST(tools_for_a_model_whose_calls_cannot_be_read_are_refused_with_the_reason) {
    Fixture f;
    f.tmpl.refusal = "the template writes each call as a JSON object";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 400);
    CHECK(r.body.find("tools") != std::string::npos);
    CHECK(r.body.find("JSON object") != std::string::npos);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);
}

TEST(an_incomplete_tool_call_is_content_not_a_tool_call) {
    Fixture f;
    /* The model stopped mid-object. Reporting this as a tool call would have the client invoke
     * the tool with truncated arguments (json_partial.h). */
    f.sched.reply = "TOOL:get_weather:{\"city\":\"ber";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "tools":[{"type":"function","function":{"name":"get_weather"}}]})");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    const auto& c = j["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK(!c["message"].contains("tool_calls"));
    CHECK(c["message"]["content"].get<std::string>().rfind("TOOL:", 0) == 0);
}

/* ================================================================== streaming */

/* Split an SSE body into its `data:` payloads and check the framing while doing it. */
static std::vector<std::string> sse_payloads(const std::string& body, int* failures_at_line) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < body.size()) {
        size_t end = body.find("\n\n", i);
        if (end == std::string::npos) { *failures_at_line = __LINE__; break; }
        std::string ev = body.substr(i, end - i);
        if (ev.rfind("data: ", 0) != 0) { *failures_at_line = __LINE__; break; }
        out.push_back(ev.substr(6));
        i = end + 2;
    }
    return out;
}

TEST(streaming_chat_is_valid_sse_and_ends_with_done) {
    Fixture f;
    f.sched.reply = "hey";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "stream":true,
                                "stream_options":{"include_usage":true}})");
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());
    CHECK(r.content_type.rfind("text/event-stream", 0) == 0);

    std::string body = drain(r);
    /* Every event is "data: <payload>\n\n" and the last one is the sentinel. */
    CHECK(body.size() > sizeof(SSE_DONE));
    CHECK(body.compare(body.size() - std::strlen(SSE_DONE), std::strlen(SSE_DONE), SSE_DONE) == 0);

    int bad = 0;
    auto payloads = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);
    CHECK(payloads.size() >= 3);
    CHECK_EQ(payloads.back(), std::string("[DONE]"));

    std::string text;
    bool saw_role = false, saw_finish = false, saw_usage = false;
    for (size_t i = 0; i + 1 < payloads.size(); ++i) {
        json j = json::parse(payloads[i]);
        CHECK_EQ(j["object"].get<std::string>(), std::string("chat.completion.chunk"));
        CHECK(j["id"].get<std::string>().rfind("chatcmpl-", 0) == 0);
        if (j.contains("usage")) { saw_usage = true; CHECK_EQ((int)j["choices"].size(), 0); continue; }
        const auto& c = j["choices"][0];
        CHECK_EQ(c["index"].get<int>(), 0);
        if (c["delta"].contains("role")) saw_role = true;
        if (c["delta"].contains("content")) text += c["delta"]["content"].get<std::string>();
        if (!c["finish_reason"].is_null()) {
            saw_finish = true;
            CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
            CHECK_EQ((int)c["delta"].size(), 0);
        }
    }
    CHECK(saw_role);
    CHECK(saw_finish);
    CHECK(saw_usage);
    CHECK_EQ(text, std::string("hey"));
}

TEST(streaming_completions_use_the_text_completion_chunk_shape) {
    Fixture f;
    f.sched.reply = "pq";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","stream":true})");
    CHECK_EQ(r.status, 200);
    std::string body = drain(r);
    int bad = 0;
    auto p = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);
    CHECK_EQ(p.back(), std::string("[DONE]"));
    std::string text;
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        json j = json::parse(p[i]);
        CHECK_EQ(j["object"].get<std::string>(), std::string("text_completion"));
        text += j["choices"][0]["text"].get<std::string>();
    }
    CHECK_EQ(text, std::string("pq"));
}

/* A REQUEST THE ENGINE FAILED SAYS WHAT FAILED, in the 500 and in a stream's error event: the
 * reason and the status the engine stopped on. It read "the engine failed this request: ok" when
 * a step failed, because nothing carried the status to the request. */
TEST(a_request_the_engine_failed_says_what_failed) {
    Fixture f;
    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_UNSUPPORTED;
    f.sched.fail_why = "the engine stopped on an error and is not serving";
    const std::string want = "the engine failed this request: the engine stopped on an error and "
                             "is not serving (unsupported)";

    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 500);
    CHECK_EQ(json::parse(r.body)["error"]["message"].get<std::string>(), want);

    HttpResponse s = f.post("/v1/completions", R"({"prompt":"hi","stream":true})");
    CHECK_EQ(s.status, 200);
    int bad = 0, errors = 0;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        json j = json::parse(p);
        if (!j.contains("error")) continue;
        ++errors;
        CHECK_EQ(j["error"]["message"].get<std::string>(), want);
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(errors, 1);

    /* With no reason given, the status alone -- and never "ok". */
    f.sched.fail_why = nullptr;
    HttpResponse t = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(json::parse(t.body)["error"]["message"].get<std::string>(),
             std::string("the engine failed this request: unsupported"));
}

/* A REPLY THAT ENDS INSIDE ITS LAST CALL, read by the real parser in Qwen's format: that call is
 * the text the model wrote, after the content before it, and nothing else moves -- the reasoning
 * stays reasoning, a call that closed before it stays a call, and the finish reason is "stop",
 * streamed or not. The stream has sent the open call's deltas already and cannot take them back;
 * its finish reason is what says not to run it, and it is the buffered one. */
static std::shared_ptr<IReplyParser> qwen_reply_parser(const json& tools) {
    ChatFormat f;
    f.reasoning_open = "<think>\n";
    f.reasoning_close = "\n</think>\n\n";
    f.reasoning_breaks = { "<tool_call>\n" };
    f.call_open = "<tool_call>\n";
    f.call_close = "</tool_call>";
    f.name_prefix = "<function=";
    f.name_suffix = ">\n";
    f.call_end = "</function>\n";
    f.arg_prefix = "<parameter=";
    f.arg_name_suffix = ">\n";
    f.arg_value_suffix = "</parameter>\n";
    f.triggers = { "<tool_call>\n" };
    std::shared_ptr<const ReplyFormat> rf;
    std::string why;
    if (ReplyFormat::compile(f, &rf, &why) < 0) return nullptr;
    ReplyOptions ro;
    ro.generation_prompt = "<|im_start|>assistant\n<think>\n";
    ro.tools = &tools;
    std::shared_ptr<const ReplyParser> rp;
    if (ReplyParser::create(rf, ro, &rp, &why) < 0) return nullptr;
    return std::make_shared<ReplyParserBridge>(rp);
}

TEST(a_reply_ending_inside_a_call_keeps_its_reasoning_and_earlier_calls) {
    const json tools = json::parse(R"([{"type":"function","function":{"name":"bash",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},
                      "required":["command"]}}}])");
    const std::string call1 = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                              "</function>\n</tool_call>\n";
    const std::string open = "<tool_call>\n<function=bash>\n<parameter=command>\ncat /a";
    const std::string reply = "Look first.\n</think>\n\nChecking.\n" + call1 + open;
    const std::string body = R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1000,
        "tools":)" + tools.dump();

    Fixture f;
    f.tmpl.real_parser = qwen_reply_parser(tools);
    REQUIRE(f.tmpl.real_parser != nullptr);
    f.sched.reply = reply;
    HttpResponse r = f.post("/v1/chat/completions", body + "}");
    CHECK_EQ(r.status, 200);
    const json c = json::parse(r.body)["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
    CHECK_EQ(c["message"]["reasoning_content"].get<std::string>(), std::string("Look first.\n"));
    CHECK_EQ(c["message"]["content"].get<std::string>(), "Checking.\n" + open);
    REQUIRE(c["message"].contains("tool_calls") && c["message"]["tool_calls"].size() == 1);
    CHECK_EQ(c["message"]["tool_calls"][0]["function"]["arguments"].get<std::string>(),
             std::string("{\"command\":\"ls\"}"));

    /* Streamed: the same reasoning, the closed call, the same finish reason. */
    HttpResponse s = f.post("/v1/chat/completions", body + R"(,"stream":true})");
    CHECK_EQ(s.status, 200);
    int bad = 0;
    std::string reasoning, finish;
    std::vector<std::string> args;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        if (j["choices"].empty()) continue;
        const json& ch = j["choices"][0];
        if (!ch["finish_reason"].is_null()) finish = ch["finish_reason"].get<std::string>();
        const json& d = ch["delta"];
        if (d.contains("reasoning_content")) reasoning += d["reasoning_content"].get<std::string>();
        if (!d.contains("tool_calls")) continue;
        const size_t k = d["tool_calls"][0]["index"].get<size_t>();
        if (k >= args.size()) args.resize(k + 1);
        if (d["tool_calls"][0]["function"].contains("arguments"))
            args[k] += d["tool_calls"][0]["function"]["arguments"].get<std::string>();
    }
    CHECK_EQ(bad, 0);
    CHECK_EQ(finish, std::string("stop"));
    CHECK_EQ(reasoning, std::string("Look first.\n"));
    REQUIRE(!args.empty());
    CHECK_EQ(args[0], std::string("{\"command\":\"ls\"}"));
}

TEST(a_streamed_tool_call_is_named_before_its_arguments_flow) {
    Fixture f;
    f.sched.reply = "TOOL:lookup:{\"q\":\"x\"}";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                "tools":[{"type":"function","function":{"name":"lookup"}}]})");
    CHECK_EQ(r.status, 200);
    std::string body = drain(r);
    int bad = 0;
    auto p = sse_payloads(body, &bad);
    CHECK_EQ(bad, 0);

    bool named = false;
    std::string args;
    std::string finish;
    for (size_t i = 0; i + 1 < p.size(); ++i) {
        json j = json::parse(p[i]);
        if (j["choices"].empty()) continue;
        const auto& d = j["choices"][0]["delta"];
        if (!j["choices"][0]["finish_reason"].is_null())
            finish = j["choices"][0]["finish_reason"].get<std::string>();
        if (!d.contains("tool_calls")) continue;
        const auto& tc = d["tool_calls"][0];
        if (tc.contains("function") && tc["function"].contains("name")) {
            /* The name arrives with the entry, before any arguments -- an entry with no name is
             * an index the client can never resolve. */
            CHECK(!named);
            CHECK(args.empty());
            CHECK_EQ(tc["function"]["name"].get<std::string>(), std::string("lookup"));
            CHECK(tc.contains("id"));
            CHECK_EQ(tc["type"].get<std::string>(), std::string("function"));
            named = true;
        }
        if (tc.contains("function") && tc["function"].contains("arguments"))
            args += tc["function"]["arguments"].get<std::string>();
    }
    CHECK(named);
    CHECK_EQ(args, std::string("{\"q\":\"x\"}"));
    CHECK_EQ(finish, std::string("tool_calls"));
}

/* A FINISHED COMPLETION STREAMS ITS WHOLE TEXT. The hold-back rules wait for bytes that may still
 * arrive; at the end none will, so a text ending in `\` or a partial `\uXXXX` -- ordinary text in
 * a completion -- must still arrive whole. With and without a stop string, because a stop list
 * adds a hold-back of its own. */
TEST(a_streamed_completion_ends_with_its_whole_text) {
    for (const char* reply : {"ab\\", "x\\u12", "q\\u", "tail\\\\"}) {
        for (const char* body : {R"({"prompt":"hi","stream":true})",
                                 R"({"prompt":"hi","stream":true,"stop":["ZZZZ"]})"}) {
            Fixture f;
            f.sched.reply = reply;
            HttpResponse r = f.post("/v1/completions", body);
            CHECK_EQ(r.status, 200);
            int bad = 0;
            auto p = sse_payloads(drain(r), &bad);
            CHECK_EQ(bad, 0);
            std::string text;
            for (size_t i = 0; i + 1 < p.size(); ++i) {
                json j = json::parse(p[i]);
                if (!j["choices"].empty()) text += j["choices"][0]["text"].get<std::string>();
            }
            CHECK_EQ(text, std::string(reply));
        }
    }
}

/* A REQUEST THAT SENT NO TOOLS IS NEVER TOLD `tool_calls`, streamed or not: it has nothing to
 * dispatch a call to. The fake template hands back a parser for every render, the way a real one
 * does for the reasoning split, and for a request without tools that parser reads the call
 * syntax as text -- so the reply comes back as the text it is, identically on both paths. */
TEST(a_chat_stream_that_sent_no_tools_never_reports_a_tool_call) {
    const std::string reply = "TOOL:lookup:{\"q\":\"x\"}";
    {
        Fixture f;
        f.sched.reply = reply;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"stream":true})");
        CHECK_EQ(r.status, 200);
        int bad = 0;
        auto p = sse_payloads(drain(r), &bad);
        CHECK_EQ(bad, 0);
        std::string content, finish;
        bool saw_call = false;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            json j = json::parse(p[i]);
            if (j["choices"].empty()) continue;
            const auto& c = j["choices"][0];
            if (c["delta"].contains("tool_calls")) saw_call = true;
            if (c["delta"].contains("content")) content += c["delta"]["content"].get<std::string>();
            if (!c["finish_reason"].is_null()) finish = c["finish_reason"].get<std::string>();
        }
        CHECK(!saw_call);
        CHECK_EQ(finish, std::string("stop"));
        CHECK_EQ(content, reply);
    }
    {
        Fixture f;
        f.sched.reply = reply;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        json j = json::parse(r.body);
        const auto& c = j["choices"][0];
        CHECK(!c["message"].contains("tool_calls"));
        CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("stop"));
        CHECK_EQ(c["message"]["content"].get<std::string>(), reply);
    }
}

/* ONE USAGE CHUNK PER STREAM, after every choice's final chunk and right before [DONE]. `n`
 * choices share one prompt and one total; a chunk per choice counts the prompt n times and lands
 * between the choices. Both endpoints. */
TEST(n_streamed_choices_share_one_usage_chunk_after_every_final_chunk) {
    const char* bodies[] = {
        R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"n":3,
            "stream_options":{"include_usage":true}})",
        R"({"prompt":"hi","stream":true,"n":3,"stream_options":{"include_usage":true}})",
    };
    const char* paths[] = {"/v1/chat/completions", "/v1/completions"};
    for (int k = 0; k < 2; ++k) {
        Fixture f;
        f.sched.reply = "ok";
        HttpResponse r = f.post(paths[k], bodies[k]);
        CHECK_EQ(r.status, 200);
        int bad = 0;
        auto p = sse_payloads(drain(r), &bad);
        CHECK_EQ(bad, 0);
        REQUIRE(p.size() >= 2);
        CHECK_EQ(p.back(), std::string("[DONE]"));

        int usage = 0, finals = 0;
        size_t usage_at = 0, last_final_at = 0;
        for (size_t i = 0; i + 1 < p.size(); ++i) {
            json j = json::parse(p[i]);
            if (j.contains("usage")) {
                ++usage;
                usage_at = i;
                CHECK_EQ((int)j["choices"].size(), 0);
                CHECK_EQ(j["usage"]["completion_tokens"].get<int64_t>(), (int64_t)6);
                continue;
            }
            if (!j["choices"][0]["finish_reason"].is_null()) { ++finals; last_final_at = i; }
        }
        CHECK_EQ(usage, 1);
        CHECK_EQ(finals, 3);
        CHECK(usage_at > last_final_at);
        CHECK_EQ(usage_at, p.size() - 2);
    }
}

/* ================================================================== reasoning */

/* HOW MUCH OF completion_tokens WAS REASONING, on both paths: the reply below opens inside the
 * block (the render says so), and `|` -- one token to FakeTokenizer -- closes it after three body
 * tokens, so four of the seven tokens are reasoning, the close included. A model whose block
 * cannot be counted reports no breakdown rather than a zero. */
TEST(usage_reports_the_reasoning_tokens_inside_completion_tokens) {
    {
        Fixture f;
        f.tmpl.think_close = "|";
        f.tmpl.think_starts_open = true;
        f.sched.reply = "abc|xyz";
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        CHECK_EQ(j["usage"]["completion_tokens"].get<int>(), 7);
        CHECK_EQ(j["usage"]["completion_tokens_details"]["reasoning_tokens"].get<int>(), 4);

        HttpResponse st = f.post("/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                     "stream_options":{"include_usage":true}})");
        int bad = 0;
        auto p = sse_payloads(drain(st), &bad);
        CHECK_EQ(bad, 0);
        REQUIRE(p.size() >= 2);
        const json u = json::parse(p[p.size() - 2]);
        CHECK_EQ(u["usage"]["completion_tokens_details"]["reasoning_tokens"].get<int>(), 4);
    }
    {
        Fixture f;
        f.tmpl.think_close = "</t>";       /* four tokens: not a countable block */
        f.tmpl.think_starts_open = true;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        CHECK(!json::parse(r.body)["usage"].contains("completion_tokens_details"));
    }
}

/* A REPLY THAT STARTS INSIDE ITS REASONING BLOCK keeps its stop strings for the answer: one
 * written while thinking is text, and the first one after the closer cuts. The scheduler, which
 * cannot tell reasoning from answer, is handed none. A closer of several tokens arrives across
 * drains, which the incremental search for it has to bridge. */
TEST(stop_strings_wait_for_the_reasoning_closer) {
    for (const char* close : { "|", "</t>" }) {
        Fixture f;
        f.tmpl.think_close = close;
        f.tmpl.think_starts_open = true;
        f.sched.reply = std::string("aENDb") + close + "cENDd";
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(),
                 std::string("aENDb") + close + "c");
        CHECK_EQ(j["choices"][0]["finish_reason"].get<std::string>(), std::string("stop"));
        CHECK(f.sched.last_stop.empty());
    }
    /* A stop string only inside the reasoning never stops the reply. */
    Fixture f;
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    f.sched.reply = "aENDb|cd";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stop":["END"]})");
    CHECK_EQ(r.status, 200);
    const json j = json::parse(r.body);
    CHECK_EQ(j["choices"][0]["message"]["content"].get<std::string>(), std::string("aENDb|cd"));
    CHECK(!f.sched.saw_cancel());             /* ended by the reply itself, not cut */
}

/* THE BUDGET REACHES THE ENGINE in every spelling -- vLLM's, Qwen's and llama.cpp's two -- with
 * the block it is held on; one the block cannot hold and two spellings that disagree are 400s
 * naming the field, and a negative one is llama.cpp's "no budget". */
TEST(a_thinking_budget_reaches_the_engine_or_is_refused_by_name) {
    Fixture f;
    f.tmpl.think_open = "{";
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    for (const char* k : { "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
                           "reasoning_budget_tokens" }) {
        HttpResponse r = f.post("/v1/chat/completions",
                                std::string(R"({"messages":[{"role":"user","content":"hi"}],")") +
                                    k + R"(":7})");
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.thinking_budget, 7);
        CHECK_EQ(f.sched.last_sp.reasoning.close, (int32_t)'|');
        CHECK_EQ(f.sched.last_sp.reasoning.open, (int32_t)'{');
        CHECK(f.sched.last_sp.reasoning.starts_open);
    }

    struct Case { const char* extra; const char* param; };
    for (const Case& c : { Case{ R"("thinking_token_budget":3,"thinking_budget":4)",
                                 "thinking_budget" },
                           Case{ R"("thinking_budget":"3")", "thinking_budget" } }) {
        HttpResponse r = f.post("/v1/chat/completions",
                                std::string(R"({"messages":[{"role":"user","content":"hi"}],)") +
                                    c.extra + "}");
        CHECK_EQ(r.status, 400);
        CHECK_EQ(json::parse(r.body)["error"]["param"].get<std::string>(), std::string(c.param));
    }

    {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],
                                    "reasoning_budget_tokens":-1})");
        CHECK_EQ(r.status, 200);
        f.sched.join_all();
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.thinking_budget, -1);
    }

    Fixture multi;
    multi.tmpl.think_close = "</t>";
    multi.tmpl.think_starts_open = true;
    HttpResponse r = multi.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],
                                    "thinking_token_budget":5})");
    CHECK_EQ(r.status, 400);
    CHECK(json::parse(r.body)["error"]["message"].get<std::string>().find("4 tokens") !=
          std::string::npos);

    Fixture none;
    HttpResponse n = none.post("/v1/chat/completions",
                               R"({"messages":[{"role":"user","content":"hi"}],
                                   "thinking_token_budget":5})");
    CHECK_EQ(n.status, 400);
    CHECK(json::parse(n.body)["error"]["message"].get<std::string>().find("no reasoning block") !=
          std::string::npos);
}

/* THE DEPLOYMENT'S BUDGET applies to a request that names none, loses to one that does, and is
 * not applied where the block cannot be counted -- it could not be held there. The startup check
 * says why a model cannot take it at all. */
TEST(the_deployment_s_thinking_budget_is_a_default_a_request_can_override) {
    Fixture f;
    f.tmpl.think_close = "|";
    f.tmpl.think_starts_open = true;
    OaiLimits lim = f.limits();
    lim.default_thinking_budget = 64;
    CHECK_EQ(chat_with(f, lim, "").sp.thinking_budget, 64);
    CHECK_EQ(chat_with(f, lim, R"("thinking_token_budget":3)").sp.thinking_budget, 3);
    CHECK_EQ(chat_with(f, lim, R"("reasoning_budget_tokens":-1)").sp.thinking_budget, -1);
    CHECK_EQ(chat_with(f, f.limits(), "").sp.thinking_budget, -1);

    f.tmpl.think_close.clear();
    CHECK_EQ(chat_with(f, lim, "").sp.thinking_budget, -1);

    Fixture g;
    ServerOptions o;
    o.model_id = "test-model";
    o.default_max_tokens = 32;
    o.default_thinking_budget = 64;
    Deps d;
    d.sched = &g.sched;
    d.tok = &g.tok;
    d.chat = &g.tmpl;
    d.grammar = &g.gram;
    Server s(d, o);
    CHECK(s.check_thinking_budget().find("no reasoning block") != std::string::npos);
    g.tmpl.think_close = "|";
    g.tmpl.think_starts_open = true;
    CHECK(s.check_thinking_budget().empty());
}

/* ================================================================== timings */

/* WHERE THE REQUEST'S TIME WENT, in llama.cpp's shape, on every path a client can take: the
 * buffered body, each streamed choice's final chunk whether or not a usage chunk was asked for,
 * and the usage chunk when it was. The fake places 8 prompt tokens with 4 from the cache, and
 * every token after the first is a verify step that proposed 3 drafts and took 1. */
TEST(a_response_carries_its_requests_timings_and_draft_counts) {
    auto check = [](const json& t, int64_t predicted, bool drafts) {
        CHECK_EQ(t["prompt_n"].get<int64_t>(), (int64_t)4);
        CHECK_EQ(t["cache_n"].get<int64_t>(), (int64_t)4);
        CHECK_EQ(t["predicted_n"].get<int64_t>(), predicted);
        CHECK(t["prompt_ms"].is_number());
        CHECK(t["predicted_ms"].is_number());
        CHECK(t["prompt_ms"].get<double>() >= 0.0);
        for (const char* k : { "prompt_per_token_ms", "prompt_per_second",
                               "predicted_per_token_ms", "predicted_per_second" })
            CHECK(t[k].is_number() || t[k].is_null());
        CHECK_EQ(t.contains("draft_n"), drafts);
        if (drafts) {
            CHECK_EQ(t["draft_n"].get<int64_t>(), (predicted - 1) * 3);
            CHECK_EQ(t["draft_n_accepted"].get<int64_t>(), predicted - 1);
        }
    };
    {
        Fixture f;
        f.sched.drafts = 3;
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}]})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        REQUIRE(j.contains("timings"));
        check(j["timings"], 5, true);
        CHECK_EQ(j["timings"]["predicted_n"].get<int64_t>(),
                 j["usage"]["completion_tokens"].get<int64_t>());
    }
    {
        Fixture f;
        HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","max_tokens":8})");
        CHECK_EQ(r.status, 200);
        const json j = json::parse(r.body);
        REQUIRE(j.contains("timings"));
        check(j["timings"], 5, false);
    }
    const char* paths[] = { "/v1/chat/completions", "/v1/completions" };
    const char* bodies[] = { R"({"messages":[{"role":"user","content":"hi"}],"stream":true)",
                             R"({"prompt":"hi","stream":true)" };
    for (int k = 0; k < 2; ++k)
        for (bool usage : { false, true }) {
            Fixture f;
            f.sched.drafts = 3;
            HttpResponse r = f.post(paths[k], std::string(bodies[k]) +
                                    (usage ? R"(,"stream_options":{"include_usage":true}})"
                                           : "}"));
            CHECK_EQ(r.status, 200);
            int bad = 0;
            auto p = sse_payloads(drain(r), &bad);
            CHECK_EQ(bad, 0);
            int finals = 0, with_timings = 0;
            for (size_t i = 0; i + 1 < p.size(); ++i) {
                const json j = json::parse(p[i]);
                if (j.contains("usage")) {
                    REQUIRE(j.contains("timings"));
                    check(j["timings"], 5, true);
                    ++with_timings;
                    continue;
                }
                if (j["choices"][0]["finish_reason"].is_null()) {
                    CHECK(!j.contains("timings"));
                    continue;
                }
                ++finals;
                REQUIRE(j.contains("timings"));
                check(j["timings"], 5, true);
                ++with_timings;
            }
            CHECK_EQ(finals, 1);
            CHECK_EQ(with_timings, usage ? 2 : 1);
        }
}

/* The rates are the tokens over the time that produced them, the first token being the prompt
 * phase's; a rate whose count or time is zero is null, and so is a prompt time the engine never
 * reported. */
TEST(timings_rates_divide_the_tokens_by_the_time_that_produced_them) {
    TimingsOut t;
    t.valid = true;
    t.prompt_n = 100;
    t.prompt_ms = 50.0;
    t.predicted_n = 10;
    t.decode_n = 9;
    t.predicted_ms = 90.0;
    json j = timings_json(t);
    CHECK_NEAR(j["prompt_per_second"].get<double>(), 2000.0, 1e-9);
    CHECK_NEAR(j["prompt_per_token_ms"].get<double>(), 0.5, 1e-12);
    CHECK_NEAR(j["predicted_per_second"].get<double>(), 100.0, 1e-9);
    CHECK_NEAR(j["predicted_per_token_ms"].get<double>(), 10.0, 1e-12);
    CHECK(!j.contains("draft_n"));

    t.decode_n = 0;
    t.predicted_ms = 0.0;
    t.prompt_ms = -1.0;
    j = timings_json(t);
    CHECK(j["predicted_per_second"].is_null());
    CHECK(j["predicted_per_token_ms"].is_null());
    CHECK(j["prompt_ms"].is_null());
    CHECK(j["prompt_per_second"].is_null());
}

/* The sink keeps the first admission across a preemption and sums the drafts it is told of. */
TEST(a_sink_keeps_its_first_admission_and_sums_its_drafts) {
    Sink s;
    CHECK(s.t_admitted_ns() < 0);
    s.mark_admitted();
    const int64_t a = s.t_admitted_ns();
    CHECK(a >= 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    s.mark_admitted();
    CHECK_EQ(s.t_admitted_ns(), a);
    s.add_drafts(4, 2);
    s.add_drafts(3, 3);
    CHECK_EQ(s.n_drafted(), (int64_t)7);
    CHECK_EQ(s.n_draft_accepted(), (int64_t)5);
}

/* ================================================================== cancellation */

TEST(a_dropped_stream_cancels_at_the_scheduler) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;          /* never finishes on its own */

    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/chat/completions";
    q.body = R"({"messages":[{"role":"user","content":"hi"}],"stream":true})";
    std::atomic<bool> gone{false};
    q.closed = [&gone] { return gone.load(); };

    HttpResponse r = f.srv->router().dispatch(q);
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());

    /* Pull a few chunks so the generation is genuinely running. */
    for (int i = 0; i < 4; ++i) { std::string c; r.next(c); }
    CHECK(!f.sched.saw_cancel());
    CHECK_EQ(f.srv->in_flight(), 1LL);

    /* The peer goes away. The transport notices on its next pull and tears the provider down. */
    gone.store(true);
    std::string c;
    bool more = r.next(c);
    CHECK(!more);
    r.on_close(false);

    CHECK(f.sched.saw_cancel());
    CHECK_EQ(f.sched.cancelled[0], f.sched.submitted[0]);
    /* The admission slot is returned at the same moment, not when the generation would have
     * ended -- otherwise an abandoned stream keeps the queue full for its whole max_tokens. */
    CHECK_EQ(f.srv->in_flight(), 0LL);

    /* The cancel has to actually stop the worker, not merely be recorded. If it did not, this
     * join would hang for the fake's full twenty-second hold. */
    f.sched.join_all();
}

TEST(a_dropped_non_streaming_request_cancels_too) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;

    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/chat/completions";
    q.body = R"({"messages":[{"role":"user","content":"hi"}]})";
    std::atomic<bool> gone{false};
    q.closed = [&gone] { return gone.load(); };

    /* The handler blocks until the answer or the drop, so the drop has to come from elsewhere. */
    std::thread hangup([&gone] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        gone.store(true);
    });
    HttpResponse r = f.srv->router().dispatch(q);
    hangup.join();

    CHECK_EQ(r.status, 499);
    CHECK(f.sched.saw_cancel());
    CHECK_EQ(f.srv->in_flight(), 0LL);
}

TEST(a_full_admission_queue_is_429_with_retry_after) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;

    /* queue_depth defaults to 8 * max_seqs, and the fixture set max_seqs to 4. */
    std::vector<HttpResponse> live;
    std::atomic<bool> never{false};
    int rejected = 0;
    for (int i = 0; i < 40; ++i) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = R"({"messages":[{"role":"user","content":"hi"}],"stream":true})";
        q.closed = [&never] { return never.load(); };
        HttpResponse r = f.srv->router().dispatch(q);
        if (r.status == 429) {
            ++rejected;
            bool has_retry = false;
            for (auto& h : r.headers) if (h.first == "Retry-After") has_retry = true;
            CHECK(has_retry);
            CHECK(r.body.find("queue") != std::string::npos);
        } else {
            CHECK_EQ(r.status, 200);
            live.push_back(std::move(r));
        }
    }
    CHECK(rejected > 0);
    CHECK_EQ((int)live.size(), 32);

    /* Unwind: every admitted stream is torn down, which must return every admission slot. */
    for (auto& r : live) if (r.on_close) r.on_close(false);
    CHECK_EQ(f.srv->in_flight(), 0LL);
    f.sched.join_all();
}

/* ================================================================== /metrics */

/* A minimal Prometheus text-format parser: enough to prove the output is well formed rather than
 * merely non-empty. Every non-comment line must be `name{labels} value` with a numeric value, and
 * every family must have declared its type before its samples. */
struct PromDoc {
    std::map<std::string, std::string> type;      /* family -> gauge|counter|histogram */
    std::map<std::string, std::string> help;
    std::multimap<std::string, double> samples;   /* full sample name -> value */
    std::vector<std::string> errors;
};

static PromDoc parse_prom(const std::string& body) {
    PromDoc d;
    size_t i = 0;
    while (i < body.size()) {
        size_t nl = body.find('\n', i);
        if (nl == std::string::npos) nl = body.size();
        std::string line = body.substr(i, nl - i);
        i = nl + 1;
        if (line.empty()) continue;
        if (line[0] == '#') {
            size_t s1 = line.find(' ');
            size_t s2 = line.find(' ', s1 + 1);
            size_t s3 = line.find(' ', s2 + 1);
            std::string kind = line.substr(s1 + 1, s2 - s1 - 1);
            std::string name = line.substr(s2 + 1, s3 - s2 - 1);
            if (kind == "TYPE") d.type[name] = line.substr(s3 + 1);
            else if (kind == "HELP") d.help[name] = line.substr(s3 + 1);
            else d.errors.push_back("unknown comment: " + line);
            continue;
        }
        size_t sp = line.rfind(' ');
        if (sp == std::string::npos) { d.errors.push_back("no value: " + line); continue; }
        std::string name = line.substr(0, sp);
        std::string val = line.substr(sp + 1);
        try {
            d.samples.emplace(name, std::stod(val));
        } catch (...) {
            d.errors.push_back("bad value: " + line);
            continue;
        }
        size_t brace = name.find('{');
        std::string family = brace == std::string::npos ? name : name.substr(0, brace);
        if (brace != std::string::npos && name.back() != '}')
            d.errors.push_back("unclosed labels: " + line);
        /* Bucket, sum and count samples belong to the histogram family. */
        for (const char* suf : {"_bucket", "_sum", "_count"}) {
            size_t n = std::strlen(suf);
            if (family.size() > n && family.compare(family.size() - n, n, suf) == 0) {
                std::string base = family.substr(0, family.size() - n);
                if (d.type.count(base)) { family = base; break; }
            }
        }
        if (!d.type.count(family)) d.errors.push_back("sample before TYPE: " + family);
    }
    return d;
}

TEST(metrics_is_valid_prometheus_text_with_vllm_names) {
    Fixture f;
    f.sched.reply = "hello";
    f.sched.m.running = 3;
    f.sched.m.waiting = 7;
    f.sched.m.kv_util = 0.42;
    f.sched.m.prefix_hit_rate = 0.25;
    f.sched.m.decode_tps = 180.5;
    f.sched.m.draft_acceptance = 0.61;
    f.sched.m.preemptions = 2;

    /* One completed request so the histograms have an observation in them. */
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})").status, 200);

    HttpResponse r = f.get("/metrics");
    CHECK_EQ(r.status, 200);
    CHECK(r.content_type.find("version=0.0.4") != std::string::npos);

    PromDoc d = parse_prom(r.body);
    if (!d.errors.empty()) {
        for (auto& e : d.errors) fprintf(stderr, "    prom: %s\n", e.c_str());
    }
    CHECK_EQ((int)d.errors.size(), 0);

    /* The names existing dashboards query, with the types they expect. */
    CHECK_EQ(d.type["vllm:num_requests_running"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:num_requests_waiting"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:gpu_cache_usage_perc"], std::string("gauge"));
    CHECK_EQ(d.type["vllm:prompt_tokens_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:generation_tokens_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:num_preemptions_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:request_success_total"], std::string("counter"));
    CHECK_EQ(d.type["vllm:time_to_first_token_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:time_per_output_token_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:e2e_request_latency_seconds"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:request_prompt_tokens"], std::string("histogram"));
    CHECK_EQ(d.type["vllm:request_generation_tokens"], std::string("histogram"));

    /* Radiance's own metrics are under our own prefix, never invented under vllm:. */
    CHECK_EQ(d.type["radiance:draft_acceptance_ratio"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:decode_tokens_per_second"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:http_admission_rejected_total"], std::string("counter"));

    /* The scheduler's numbers actually reached the gauges. */
    auto find_one = [&](const std::string& prefix) -> double {
        for (auto& kv : d.samples)
            if (kv.first.rfind(prefix, 0) == 0) return kv.second;
        return -12345.0;
    };
    CHECK_NEAR(find_one("vllm:num_requests_running{"), 3.0, 1e-9);
    CHECK_NEAR(find_one("vllm:num_requests_waiting{"), 7.0, 1e-9);
    CHECK_NEAR(find_one("vllm:gpu_cache_usage_perc{"), 0.42, 1e-9);
    CHECK_NEAR(find_one("radiance:draft_acceptance_ratio{"), 0.61, 1e-9);
    CHECK_NEAR(find_one("vllm:generation_tokens_total{"), 5.0, 1e-9);

    /* Every sample carries the model_name label vLLM's dashboards group by. */
    for (auto& kv : d.samples)
        CHECK(kv.first.find("model_name=\"test-model\"") != std::string::npos);

    /* Histogram invariants: buckets are cumulative and non-decreasing, +Inf equals _count. */
    double last = -1.0, inf = -1.0, count = -1.0;
    for (auto& kv : d.samples) {
        if (kv.first.rfind("vllm:time_to_first_token_seconds_bucket", 0) == 0) {
            if (kv.first.find("le=\"+Inf\"") != std::string::npos) inf = kv.second;
            else { CHECK(kv.second >= last); last = kv.second; }
        }
        if (kv.first.rfind("vllm:time_to_first_token_seconds_count", 0) == 0) count = kv.second;
    }
    CHECK_NEAR(inf, count, 1e-9);
    CHECK_NEAR(count, 1.0, 1e-9);

    /* The bucket layout is vLLM's, not one we invented: the first and last bounds must match. */
    CHECK(r.body.find("vllm:time_to_first_token_seconds_bucket{model_name=\"test-model\","
                      "engine=\"0\",le=\"0.001\"}") != std::string::npos);
    CHECK(r.body.find("le=\"2560\"") != std::string::npos);
}

/* THE SCHEDULER'S TOTALS ARE CUMULATIVE, and a scrape turns them into counter increments by adding
 * the difference from what the counter already holds. Scrapes arriving together must still count
 * each total once: however many of them run, the counter ends at the total. */
TEST(concurrent_scrapes_count_each_total_once) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.steps = 1000;
    s.prefill_tokens = 5000;
    s.draft_depth_seen = 3;
    s.draft_pos_reached[0] = 10;
    s.draft_pos_accepted[0] = 5;

    std::vector<std::thread> scrapers;
    for (int t = 0; t < 8; ++t)
        scrapers.emplace_back([&m, &s] {
            for (int i = 0; i < 2000; ++i) { m.sample(s); if (i % 100 == 0) m.render(); }
        });
    for (auto& t : scrapers) t.join();

    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    auto value = [&d](const std::string& family) {
        for (const auto& kv : d.samples)
            if (kv.first.rfind(family + "{", 0) == 0) return kv.second;
        return -1.0;
    };
    CHECK_EQ(value("radiance:engine_steps_total"), 1000.0);
    CHECK_EQ(value("radiance:prefill_tokens_total"), 5000.0);
}

/* EACH DRAFT POSITION'S ACCEPTANCES ARE A COUNTER NEXT TO ITS REACH. At a fixed depth position
 * j+1's reach is j's acceptances, but an adaptive window can end at j with j taken, so a ratio
 * over a window needs both. Here two verifies end at position 1 with it accepted. */
TEST(each_draft_position_counts_its_acceptances) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    s.draft_depth_seen = 3;
    s.draft_pos_reached[0] = 10; s.draft_pos_accepted[0] = 7;
    s.draft_pos_reached[1] = 7;  s.draft_pos_accepted[1] = 5;
    s.draft_pos_reached[2] = 3;  s.draft_pos_accepted[2] = 2;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:draft_position_accepted_total"], std::string("counter"));
    CHECK_EQ(d.type["radiance:draft_position_reached_total"], std::string("counter"));
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"";
    const std::string A = "radiance:draft_position_accepted_total" + L + ",position=";
    CHECK_NEAR(value(A + "\"0\"}"), 7.0, 1e-9);
    CHECK_NEAR(value(A + "\"1\"}"), 5.0, 1e-9);
    CHECK_NEAR(value(A + "\"2\"}"), 2.0, 1e-9);
    CHECK_NEAR(value(A + "\"3\"}"), -1.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_position_acceptance_ratio" + L + ",position=\"1\"}"),
               5.0 / 7.0, 1e-9);
}

/* THE ADAPTIVE DRAFT WINDOW (--draft-window adaptive) IS EXPOSED ONLY WHILE IT IS ON, so a
 * deployment without it scrapes exactly what it always did -- and when it is on, the gauge and the
 * histogram are well-formed families with the labels every other sample carries. */
TEST(the_adaptive_draft_window_is_exposed_only_while_it_is_on) {
    Metrics m("m", 1024);
    server::SchedMetrics s;
    m.sample(s);
    std::string body = m.render();
    CHECK(body.find("radiance:draft_depth_adaptive") == std::string::npos);
    CHECK(body.find("radiance:draft_verify_depth_total") == std::string::npos);

    s.draft_adaptive = true;
    s.draft_depth_adaptive = 2.5;
    s.draft_verify_depth[2] = 7;
    s.draft_verify_depth[5] = 3;
    m.sample(s);
    PromDoc d = parse_prom(m.render());
    CHECK_EQ((int)d.errors.size(), 0);
    CHECK_EQ(d.type["radiance:draft_depth_adaptive"], std::string("gauge"));
    CHECK_EQ(d.type["radiance:draft_verify_depth_total"], std::string("counter"));
    auto value = [&d](const std::string& name) {
        auto it = d.samples.find(name);
        return it == d.samples.end() ? -1.0 : it->second;
    };
    const std::string L = "{model_name=\"m\",engine=\"0\"";
    CHECK_NEAR(value("radiance:draft_depth_adaptive" + L + "}"), 2.5, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"1\"}"), 0.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"2\"}"), 7.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"5\"}"), 3.0, 1e-9);
    CHECK_NEAR(value("radiance:draft_verify_depth_total" + L + ",depth=\"6\"}"), -1.0, 1e-9);
}

TEST(models_and_health_answer) {
    Fixture f;
    HttpResponse r = f.get("/v1/models");
    CHECK_EQ(r.status, 200);
    json j = json::parse(r.body);
    CHECK_EQ(j["object"].get<std::string>(), std::string("list"));
    CHECK_EQ(j["data"][0]["id"].get<std::string>(), std::string("test-model"));

    HttpResponse one = f.get("/v1/models/test-model");
    CHECK_EQ(one.status, 200);
    /* The entry itself, the way OpenAI's retrieve answers it. */
    CHECK_EQ(json::parse(one.body).value("id", std::string()), std::string("test-model"));
    CHECK_EQ(json::parse(one.body).value("object", std::string()), std::string("model"));
    CHECK_EQ(f.get("/v1/models/other").status, 404);

    HttpResponse h = f.get("/health");
    CHECK_EQ(h.status, 200);
    CHECK(json::parse(h.body)["status"].get<std::string>() == "ok");
}

TEST(the_model_entry_carries_the_context_length_the_server_was_started_with) {
    /* Both routes a client sizes its window from: the list, and the entry it fetches by id. */
    Fixture f(true, 262144, "/models/m.rad");
    CHECK_EQ(json::parse(f.get("/v1/models/test-model").body).value("root", std::string()),
             std::string("/models/m.rad"));
    HttpResponse list = f.get("/v1/models");
    CHECK_EQ(list.status, 200);
    CHECK_EQ(json::parse(list.body)["data"][0].value("max_model_len", (int64_t)0), (int64_t)262144);
    HttpResponse one = f.get("/v1/models/test-model");
    CHECK_EQ(one.status, 200);
    CHECK_EQ(json::parse(one.body).value("max_model_len", (int64_t)0), (int64_t)262144);
    Fixture unknown;
    CHECK(!json::parse(unknown.get("/v1/models").body)["data"][0].contains("max_model_len"));
}

TEST(the_model_entry_and_server_info_report_the_same_capabilities) {
    Fixture f;
    const json info = json::parse(f.get("/server_info").body)["capabilities"];
    const json list = json::parse(f.get("/v1/models").body)["data"][0]["capabilities"];
    const json one  = json::parse(f.get("/v1/models/test-model").body)["capabilities"];
    CHECK(info.is_object());
    CHECK(list == info);
    CHECK(one == info);
    /* Read per request, not frozen at startup: a template that takes no tools says so. */
    CHECK(info.value("tools", false));
    f.tmpl.tools_ok = false;
    CHECK(!json::parse(f.get("/v1/models").body)["data"][0]["capabilities"].value("tools", true));
}

TEST(embeddings_without_an_embedder_is_501_not_an_empty_200) {
    Fixture f;
    HttpResponse r = f.post("/v1/embeddings", R"({"input":"hi"})");
    CHECK_EQ(r.status, 501);
    CHECK(r.body.find("embedding") != std::string::npos);
}

/* ================================================================== partial JSON */

TEST(partial_json_knows_complete_from_truncated) {
    CHECK(json_complete("{}"));
    CHECK(json_complete("  {\"a\":[1,2,{\"b\":null}]}  "));
    CHECK(json_complete("\"hi\""));
    CHECK(json_complete("-1.5e3"));
    CHECK(!json_complete("{"));
    CHECK(!json_complete("{\"a\":"));
    CHECK(!json_complete("{\"a\":1"));
    CHECK(!json_complete("{\"a\":\"unterminated"));
    CHECK(!json_complete("{} trailing"));
    CHECK(!json_complete(""));
}

TEST(a_stable_prefix_never_cuts_a_code_point_or_an_escape) {
    /* Two-byte code point split across a boundary. */
    std::string s = "ab\xc3\xa9";
    CHECK_EQ((long long)json_stable_prefix(s), 4LL);
    CHECK_EQ((long long)json_stable_prefix(s.substr(0, 3)), 2LL);   /* lead byte only */

    /* A lone trailing backslash is held back; an escaped backslash is not. */
    CHECK_EQ((long long)json_stable_prefix("a\\"), 1LL);
    CHECK_EQ((long long)json_stable_prefix("a\\\\"), 3LL);

    /* A half-typed \u escape is held back until all four digits are there. */
    CHECK_EQ((long long)json_stable_prefix("x\\u00"), 1LL);
    CHECK_EQ((long long)json_stable_prefix("x\\u0041"), 7LL);
}

/* A string literal is written exactly as nlohmann writes it, and refused exactly where nlohmann
 * refuses: every byte value alone and after each lead byte, and random strings drawn from the
 * bytes that decide -- controls, the quote and backslash, DEL, continuation and lead bytes. */
TEST(a_quoted_string_is_what_nlohmann_writes) {
    auto agree = [](const std::string& s) {
        std::string got = "x";
        const bool ok = json_quote_append(got, s);
        std::string want;
        bool refused = false;
        try {
            want = "x" + json(s).dump();
        } catch (const nlohmann::json::type_error&) {
            refused = true;
        }
        CHECK_EQ(ok, !refused);
        if (ok) CHECK_EQ(got, want);
        else CHECK_EQ(got, std::string("x"));
        return ok == !refused && (!ok || got == want);
    };
    for (int a = 0; a < 256; ++a) {
        agree(std::string(1, (char)a));
        for (int b = 0; b < 256; ++b) agree(std::string{ (char)a, (char)b, 'z' });
    }
    const unsigned char pool[] = { 0x00, 0x01, 0x08, 0x09, 0x0a, 0x0c, 0x0d, 0x1f, 0x20, '"',
                                   '\\', '/', 'a', 0x7f, 0x80, 0x8f, 0x90, 0x9f, 0xa0, 0xbf,
                                   0xc0, 0xc1, 0xc2, 0xdf, 0xe0, 0xed, 0xee, 0xef, 0xf0, 0xf4,
                                   0xf5, 0xff };
    std::mt19937 rng(7);
    int bad = 0;
    for (int i = 0; i < 200000 && bad < 4; ++i) {
        std::string s(rng() % 12, '\0');
        for (char& c : s) c = (char)pool[rng() % sizeof pool];
        if (!agree(s)) ++bad;
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the live view */

TEST(the_live_view_prints_ordinary_lines_when_it_is_not_a_terminal) {
    LiveSnapshot s;
    s.step = 12;
    s.model = "m";
    s.running = 4;
    s.waiting = 1;
    s.kv_util = 0.5;
    s.decode_tps = 1234.0;
    s.draft_acceptance = 0.7;
    CardStat c;
    c.index = 0;
    c.name = "R9700";
    c.util = 0.99;
    c.vram_used = 20LL << 30;
    c.vram_total = 32LL << 30;
    s.cards.push_back(c);
    s.experts.n_layers = 2;
    s.experts.n_experts = 4;
    s.experts.tier = {0, 0, 1, 3, 0, 2, 3, 3};
    /* THE TALLY IS THE SNAPSHOT'S, NOT THE RENDERER'S. Counting it in the renderer counts the
     * CELLS of the plane, and a plane is a rectangle a model need not fill. The engine counts the
     * units it actually placed and both renderers read that, so a snapshot that carries a plane
     * and no counts renders a plane and no counts. */
    s.experts.n_units = 8;
    s.experts.units[0] = 3; s.experts.units[1] = 1;
    s.experts.units[2] = 1; s.experts.units[3] = 3;
    s.experts.bytes[0] = 3LL << 30; s.experts.total_bytes = 8LL << 30;
    s.experts.promotions = 40; s.experts.demotions = 37; s.experts.readmits = 12;
    s.link.present = true;
    s.link.world = 2;
    s.link.ar_bytes = 5000;
    s.link.h2d_bytes = 900;
    s.link.d2h_bytes = 800;
    /* The router asked for 200 experts and found 150 of them resident; the 50 it did not cost
     * 700 bytes of GEMM streaming across the link. */
    s.experts.routed = 200; s.experts.routed_resident = 150;
    s.link.stream_bytes = 700;

    /* A pipe is not a terminal, so the constructor picks the line renderer and the frame must
     * contain no escape sequence at all -- a log full of cursor-home codes is ungreppable and the
     * breakage is invisible on the terminal where it was written (spec §16). */
    LiveView v(nullptr, 1000, stderr);
    std::string line = v.render_line(s);
    CHECK(line.find('\033') == std::string::npos);
    CHECK(line.find("step=12") != std::string::npos);
    CHECK(line.find("run=4") != std::string::npos);
    /* Tokens a second whole, never with a prefix. */
    CHECK(line.find("decode_tps=1234") != std::string::npos);
    CHECK(line.find("draft_accept=0.70") != std::string::npos);
    CHECK(line.find("gpu0=") != std::string::npos);
    CHECK(line.find("experts vram=3") != std::string::npos);
    CHECK(line.find("ssd=3") != std::string::npos);
    /* Residency as a share, and the churn beside the move totals: a promotion count without the
     * re-admissions cannot distinguish an engine filling from one thrashing. */
    CHECK(line.find("resident=37.50%") != std::string::npos);
    CHECK(line.find("readmit=12") != std::string::npos);
    /* HIT RATE IS NOT RESIDENCY: 37.50% of the units are in VRAM and 75% of the routing found
     * one there, which is the placement working. Both numbers, because either alone misleads. */
    CHECK(line.find("hit=75.00%") != std::string::npos);
    /* THE LINK, cumulative, all three of the things that share it. */
    CHECK(line.find("link ar=5000") != std::string::npos);
    CHECK(line.find("h2d=900") != std::string::npos);
    CHECK(line.find("d2h=800") != std::string::npos);
    CHECK(line.find("stream=700") != std::string::npos);
    CHECK(line.back() == '\n');

    /* The terminal renderer is the one that may use escapes, and it colours by tier. */
    std::string screen = v.render_screen(s);
    CHECK(screen.find('\033') != std::string::npos);
    CHECK(screen.find("R9700") != std::string::npos);
    CHECK(screen.find("8 units") != std::string::npos);
    CHECK(screen.find("37.5%") != std::string::npos);
    /* THE WEB PAGE'S FIGURES: three significant figures, binary sizes with their unit said once
     * for a used/total pair, SI prefixes on a rate. */
    CHECK(screen.find("20.0 / 32.0 GiB") != std::string::npos);
    CHECK(screen.find("99.0%") != std::string::npos);
    CHECK(screen.find("8.00 GiB") != std::string::npos);
    CHECK(screen.find("3.00 GiB") != std::string::npos);
    CHECK(screen.find("decode 1234 tok/s") != std::string::npos);
    CHECK(screen.find("20.0G") == std::string::npos);
}

/* ================================================================== the sink */

TEST(the_sink_is_a_single_producer_ring_that_fails_loudly_on_overflow) {
    Sink s(16);
    CHECK(!s.done());
    for (int i = 0; i < 16; ++i) CHECK(s.push(65 + i));
    /* One more than capacity is an overflow, and overflow fails the request rather than dropping
     * a token in the middle of an answer. */
    CHECK(!s.push(99));
    CHECK(s.overflowed());
    CHECK(s.done());
    CHECK_EQ((int)s.finish_reason(), (int)Finish::Error);
    CHECK_EQ(s.status(), RAD_E_FULL);

    SinkToken buf[32];
    CHECK_EQ((long long)s.read(buf, 32), 16LL);
    CHECK_EQ(buf[0].token, 65);
    CHECK_EQ(buf[15].token, 80);

    /* The first terminal state wins: a cancel racing an EOS must not report twice. */
    Sink t(16);
    t.finish(Finish::Stop);
    t.finish(Finish::Cancelled);
    CHECK_EQ((int)t.finish_reason(), (int)Finish::Stop);
}


/* ================================================================== the admin surface */

TEST(version_load_and_ping_answer_the_shapes_clients_read) {
    Fixture f;

    HttpResponse v = f.get("/version");
    CHECK_EQ(v.status, 200);
    /* vLLM's key, and a client that checks compatibility reads exactly this one. */
    CHECK(has(v.body, "\"version\""));

    HttpResponse l = f.get("/load");
    CHECK_EQ(l.status, 200);
    CHECK(has(l.body, "\"server_load\":0"));

    /* Both methods, because different load balancers send different ones. */
    CHECK_EQ(f.get("/ping").status, 200);
    CHECK_EQ(f.post("/ping", "").status, 200);
    CHECK_EQ(f.get("/health").status, 200);
}

TEST(server_info_lists_the_router_rather_than_a_copy_of_it) {
    Fixture f;
    HttpResponse r = f.get("/server_info");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"model\":\"test-model\""));
    CHECK(has(r.body, "\"chat\":true"));
    /* Every registered route appears, so a route added without a test still shows up here. */
    for (const std::string& sig : f.srv->router().routes()) CHECK(has(r.body, sig));
}

TEST(props_answers_what_a_llama_server_client_reads) {
    Fixture f;
    HttpResponse r = f.get("/props");
    CHECK_EQ(r.status, 200);
    const json p = json::parse(r.body);
    CHECK_EQ(p.value("total_slots", (int64_t)0), (int64_t)4);
    /* No media pipeline attached: no image button. */
    CHECK(!p["modalities"].value("vision", true));
    /* The fixture knows no context length, so none is invented. */
    CHECK(!p["default_generation_settings"].contains("n_ctx"));
    /* The same defaults /server_info reports, under llama-server's names. */
    const json defs = json::parse(f.get("/server_info").body)["request_defaults"];
    const json params = p["default_generation_settings"]["params"];
    CHECK_NEAR(params.value("temperature", -1.0), defs.value("temperature", -2.0), 1e-9);
    CHECK_NEAR(params.value("repeat_penalty", -1.0), defs.value("repetition_penalty", -2.0), 1e-9);
    CHECK_EQ(params.value("n_predict", (int64_t)0), (int64_t)32);   /* the fixture's default */
    CHECK(has(p.value("build_info", std::string()), "radiance"));
}

TEST(props_reports_the_context_one_request_may_use) {
    /* The fixture with a known length: build one rather than widen the shared Fixture. */
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    FakeGrammar gram;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    d.grammar = &gram;
    ServerOptions o;
    o.model_id = "test-model";
    o.max_ctx = 262144;
    Server srv(d, o);
    const json p = json::parse(srv.router().call("GET", "/props").body);
    CHECK_EQ(p["default_generation_settings"].value("n_ctx", (int64_t)0), (int64_t)262144);
    sched.join_all();
}

TEST(server_info_drops_an_engine_blob_that_is_not_an_object) {
    /* A malformed blob must not corrupt the response: /server_info missing a section beats
     * /server_info being unparseable. */
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.model_id = "test-model";
    o.engine_info = "not json at all";
    Server srv(d, o);
    HttpResponse r = srv.router().call("GET", "/server_info");
    CHECK_EQ(r.status, 200);
    CHECK(!has(r.body, "\"engine\""));

    ServerOptions o2 = o;
    o2.engine_info = R"({"arch":"testarch"})";
    Server srv2(d, o2);
    HttpResponse r2 = srv2.router().call("GET", "/server_info");
    CHECK(has(r2.body, "\"arch\":\"testarch\""));
}

TEST(tokenize_takes_a_prompt_or_messages_and_never_both) {
    Fixture f;

    HttpResponse r = f.post("/tokenize", R"({"prompt":"AB"})");
    CHECK_EQ(r.status, 200);
    /* FakeTokenizer maps a byte to its code point. */
    CHECK(has(r.body, "\"count\":2"));
    CHECK(has(r.body, "[65,66]"));
    CHECK(has(r.body, "\"token_strs\":null"));

    HttpResponse s = f.post("/tokenize", R"({"prompt":"AB","return_token_strs":true})");
    CHECK(has(s.body, "\"token_strs\":["));

    /* The chat form renders the template first, and the fake's render is not the raw text -- so
     * a count equal to the message body would mean the template was skipped. */
    HttpResponse c = f.post("/tokenize", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(c.status, 200);
    CHECK(!has(c.body, "\"count\":2"));

    CHECK_EQ(f.post("/tokenize", R"({"prompt":"A","messages":[]})").status, 400);
    CHECK_EQ(f.post("/tokenize", R"({"tokens":[1]})").status, 400);
    CHECK_EQ(f.post("/tokenize", "[1,2]").status, 400);
}

TEST(a_raw_prompt_parses_control_tokens_on_every_endpoint_that_takes_one) {
    /* A client that renders its own chat template sends "<|im_start|>" meaning the token, as vLLM
     * and llama.cpp read it. /tokenize must count what /v1/completions will see. */
    Fixture f;
    f.tok.last_parse_special = false;
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"<|im_start|>user","max_tokens":4})").status, 200);
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":["a","b"],"max_tokens":4})").status, 200);
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    f.post("/tokenize", R"({"prompt":"<|im_start|>"})");
    CHECK(f.tok.last_parse_special);
    f.tok.last_parse_special = false;
    f.post("/tokenize", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK(f.tok.last_parse_special);
    for (const char* body : {R"({"input":"<|im_start|>"})", R"({"input":["a","b"]})"}) {
        OaiRequest r; ApiError e;
        f.tok.last_parse_special = false;
        CHECK_OK(parse_embeddings_request(body, f.odeps(), f.limits(), &r, &e));
        CHECK(f.tok.last_parse_special);
    }
}

/* An embedding model's request fields: prompt_name puts the model's named prompt in front of text
 * and content-part inputs and is refused on token ids and on a name the model does not state;
 * dimensions must be a positive integer; image_tokens one of the trained budgets; a messages input
 * renders through the model's template and excludes `input`. */
struct FakeEmbedder final : IEmbedder {
    IChatTemplate* tpl = nullptr;
    int64_t dim() const override { return 768; }
    int64_t max_input() const override { return 8192; }
    bool prompt(const std::string& name, std::string* text) const override {
        if (name != "query") return false;
        *text = "task: search result | query: ";
        return true;
    }
    IChatTemplate* message_template() const override { return tpl; }
};

TEST(embeddings_take_the_models_named_prompts_and_its_fields) {
    Fixture f;
    FakeEmbedder emb;
    emb.tpl = &f.tmpl;
    OaiDeps d = f.odeps();
    d.embed = &emb;
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":"capital?","prompt_name":"query"})", d,
                                          f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)1);
        CHECK_EQ(r.prompts[0].text, std::string("task: search result | query: capital?"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":["a","b"],"prompt_name":"query"})", d,
                                          f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)2);
        CHECK_EQ(r.prompts[1].text, std::string("task: search result | query: b"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":"x","prompt_name":"nope"})", d, f.limits(),
                                       &r, &e) < 0);
        CHECK_EQ(e.param, std::string("prompt_name"));
    }
    {
        OaiRequest r; ApiError e;
        CHECK(parse_embeddings_request(R"({"input":[1,2,3],"prompt_name":"query"})", d,
                                       f.limits(), &r, &e) < 0);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(R"({"input":"x","dimensions":256})", d, f.limits(), &r,
                                          &e));
        CHECK_EQ(r.dimensions, (int64_t)256);
        CHECK(parse_embeddings_request(R"({"input":"x","dimensions":0})", d, f.limits(), &r,
                                       &e) < 0);
        CHECK(parse_embeddings_request(R"({"input":"x","image_tokens":300})", d, f.limits(), &r,
                                       &e) < 0);
    }
    {
        OaiRequest r; ApiError e;
        CHECK_OK(parse_embeddings_request(
            R"({"messages":[{"role":"user","content":"hello"}]})", d, f.limits(), &r, &e));
        CHECK_EQ(r.prompts.size(), (size_t)1);
        CHECK(!r.prompts[0].tokens.empty());
        CHECK(parse_embeddings_request(
                  R"({"input":"x","messages":[{"role":"user","content":"hello"}]})", d,
                  f.limits(), &r, &e) < 0);
    }
}

TEST(detokenize_answers_under_the_key_vllm_uses) {
    Fixture f;
    HttpResponse r = f.post("/detokenize", R"({"tokens":[65,66,67]})");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"prompt\":\"ABC\""));

    CHECK_EQ(f.post("/detokenize", R"({"tokens":"ABC"})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[1,"x"]})").status, 400);
    CHECK_EQ(f.post("/detokenize", "{}").status, 400);
}

/* An id that is not an int32 is refused, not narrowed: 2^40 narrowed is token 0, and the answer
 * would be the text of a token nobody sent. */
TEST(detokenize_refuses_an_id_that_is_not_an_int32) {
    Fixture f;
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[1099511627776]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[65,2147483648]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[-1]})").status, 400);
    CHECK_EQ(f.post("/detokenize", R"({"tokens":[18446744073709551615]})").status, 400);
}

TEST(tokenizer_info_reports_the_vocabulary_and_what_the_template_can_do) {
    Fixture f;
    HttpResponse r = f.get("/get_tokenizer_info");
    CHECK_EQ(r.status, 200);
    CHECK(has(r.body, "\"chat_template\":true"));
    CHECK(has(r.body, "\"supports_tools\":true"));
}

TEST(reset_prefix_cache_is_501_when_there_is_no_cache_to_reset) {
    /* The default IScheduler has no prefix cache, and saying so is the point: a 200 here would
     * tell a benchmark it started cold when it did not. */
    Fixture f;
    HttpResponse r = f.post("/reset_prefix_cache", "");
    CHECK_EQ(r.status, 501);
}

/* ================================================================== the transport
 *
 * A real listener on 127.0.0.1, port 0. What is under test is the worker pool's configuration:
 * every connection holds a worker for as long as it is open, which no in-process dispatch can
 * show. */

/* A loopback connection that gives up on a read after `timeout_s`, or -1. */
static int dial(int port, int timeout_s) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    timeval tv{};
    tv.tv_sec = timeout_s;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, (const sockaddr*)&a, sizeof a) < 0) { ::close(fd); return -1; }
    return fd;
}

static bool send_all(int fd, const std::string& s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

/* The status line of the response on `fd`, or "" if none arrived before the read timeout. */
static std::string status_line(int fd) {
    std::string got;
    char buf[512];
    while (got.find("\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        got.append(buf, (size_t)n);
    }
    const size_t e = got.find("\r\n");
    return e == std::string::npos ? std::string() : got.substr(0, e);
}

/* A Server on a real socket, torn down in the order the engine uses: connections first, so every
 * worker they hold comes back, then the listener, then the scheduler's own threads. */
struct LiveServer {
    FakeScheduler           sched;
    FakeTokenizer           tok;
    std::unique_ptr<Server> srv;
    std::thread             serving;
    std::vector<int>        fds;

    /* Generations never finish on their own; each one holds its worker until its peer leaves. */
    LiveServer(int n_threads, int64_t max_seqs,
               const std::function<void(ServerOptions&)>& tweak = nullptr) {
        sched.reply = "abcdefgh";
        sched.hang = true;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        ServerOptions o;
        o.host = "127.0.0.1";
        o.port = 0;
        o.n_threads = n_threads;
        o.max_seqs = max_seqs;
        o.default_max_tokens = 32;
        if (tweak) tweak(o);
        srv.reset(new Server(d, o));
        serving = std::thread([this] { srv->run(); });
        for (int i = 0; i < 500 && srv->port() == 0; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ~LiveServer() {
        for (int fd : fds) ::close(fd);
        srv->stop();
        serving.join();
        sched.join_all();
        srv.reset();
    }
    int open(const std::string& send, int timeout_s = 5) {
        const int fd = dial(srv->port(), timeout_s);
        if (fd < 0) return -1;
        fds.push_back(fd);
        return send_all(fd, send) ? fd : -1;
    }
};

/* THE POOL OUTGROWS THE ADMISSION CEILING. A generation holds its connection's worker for as long
 * as it runs, so a pool no larger than the ceiling fills before admission refuses anyone: the 429
 * never comes, and a probe queues behind generations. Here the ceiling is 8 and the resident pool
 * 4. Every admitted request is running, forty connections sit on half-sent headers, and the ninth
 * request is still told 429 while /health still answers. */
TEST(a_probe_and_a_429_are_answered_while_every_admitted_request_holds_a_worker) {
    LiveServer s(/*n_threads=*/4, /*max_seqs=*/1);
    REQUIRE(s.srv->port() > 0);

    const std::string gen = "POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
                            "Content-Type: application/json\r\nContent-Length: 15\r\n\r\n"
                            "{\"prompt\":\"hi\"}";
    for (int i = 0; i < 8; ++i) CHECK(s.open(gen) >= 0);
    for (int i = 0; i < 500 && s.srv->in_flight() < 8; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK_EQ(s.srv->in_flight(), (int64_t)8);

    for (int i = 0; i < 40; ++i) CHECK(s.open("GET /health HTTP/1.1\r\nHost: t\r\n") >= 0);

    const int over = s.open(gen);
    REQUIRE(over >= 0);
    CHECK(has(status_line(over), " 429 "));

    const int probe = s.open("GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    REQUIRE(probe >= 0);
    CHECK(has(status_line(probe), " 200 "));
}

/* The response head on `fd`: everything up to the blank line, or what arrived before the read
 * timeout. */
static std::string response_head(int fd) {
    std::string got;
    char buf[512];
    while (got.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n <= 0) break;
        got.append(buf, (size_t)n);
    }
    return got.substr(0, got.find("\r\n\r\n"));
}

/* THE TRANSPORT SETTINGS REACH THE LISTENER. CORS headers are on unless turned off, and a body
 * past --max-body-mib is refused before it is read. */
TEST(the_transport_settings_reach_the_listener) {
    const std::string probe = "GET /health HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n";
    {
        LiveServer s(4, 1);
        REQUIRE(s.srv->port() > 0);
        const int fd = s.open(probe);
        REQUIRE(fd >= 0);
        CHECK(has(response_head(fd), "Access-Control-Allow-Origin: *"));
    }
    {
        LiveServer s(4, 1, [](ServerOptions& o) {
            o.cors = false;
            o.max_body_bytes = 64;
        });
        REQUIRE(s.srv->port() > 0);
        const int fd = s.open(probe);
        REQUIRE(fd >= 0);
        const std::string head = response_head(fd);
        CHECK(has(head, " 200 "));
        CHECK(!has(head, "Access-Control-Allow-Origin"));

        const std::string big = std::string(R"({"prompt":")") + std::string(100, 'x') + "\"}";
        const int b = s.open("POST /v1/completions HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
                             "Content-Type: application/json\r\nContent-Length: " +
                             std::to_string(big.size()) + "\r\n\r\n" + big);
        REQUIRE(b >= 0);
        CHECK(has(status_line(b), " 413 "));
    }
}

/* The Retry-After a 429 carries is the deployment's. */
TEST(a_429_carries_the_deployments_retry_after) {
    FakeScheduler sched;
    FakeTokenizer tok;
    sched.reply = "abcdefgh";
    sched.hang = true;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.max_seqs = 1;
    o.queue_depth = 1;
    o.retry_after_s = 7;
    o.default_max_tokens = 32;
    auto srv = std::make_unique<Server>(d, o);
    std::atomic<bool> never{false};
    std::vector<HttpResponse> live;
    std::string retry;
    for (int i = 0; i < 2; ++i) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/completions";
        q.body = R"({"prompt":"hi","stream":true})";
        q.closed = [&never] { return never.load(); };
        HttpResponse r = srv->router().dispatch(q);
        if (r.status == 429) {
            for (auto& h : r.headers) if (h.first == "Retry-After") retry = h.second;
        } else {
            live.push_back(std::move(r));
        }
    }
    CHECK_EQ(retry, std::string("7"));
    for (auto& r : live) if (r.on_close) r.on_close(false);
    sched.join_all();
    srv.reset();
}

/* ================================================================== llama-server parity */

/* llama-server's spellings reach the engine as the fields they name, and two spellings of one
 * field that disagree are refused. -1 is llama-server's "no limit": not the deployment's default
 * (the fixture's is 32) and not a refusal. */
TEST(llama_server_spellings_reach_the_engine) {
    Fixture f;
    auto post = [&](const std::string& extra) {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}])" + extra + "}");
        f.sched.join_all();
        return r;
    };
    CHECK_EQ(post(R"(,"repeat_penalty":1.25,"n_predict":5)").status, 200);
    {
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_NEAR(f.sched.last_sp.rep_penalty, 1.25, 1e-6);
        CHECK_EQ(f.sched.last_max_tokens, 5);
    }
    CHECK_EQ(post(R"(,"repeat_penalty":1.2,"repetition_penalty":1.3)").status, 400);
    CHECK_EQ(post(R"(,"max_tokens":4,"n_predict":5)").status, 400);
    CHECK_EQ(post("").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK_EQ(f.sched.last_max_tokens, 32); }
    CHECK_EQ(post(R"(,"max_tokens":-1)").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_max_tokens != 32); }
    CHECK_EQ(post(R"(,"max_tokens":0)").status, 400);
}

/* `samplers`: a stage left out is off, temperature after the cuts sets temp_last, and an order
 * the chain cannot run is refused by name -- as is a sampler it does not have. */
TEST(the_sampler_order_is_honoured_or_refused) {
    Fixture f;
    auto post = [&](const std::string& extra) {
        HttpResponse r = f.post("/v1/chat/completions",
                                R"({"messages":[{"role":"user","content":"hi"}],"temperature":1.5,)"
                                R"("top_p":0.5,"top_k":40)" + extra + "}");
        f.sched.join_all();
        return r;
    };
    CHECK_EQ(post(R"(,"samplers":"kpt")").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_sp.temp_last); }
    CHECK_EQ(post(R"(,"samplers":["top_k","temperature","top_p"])").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(!f.sched.last_sp.temp_last); }
    /* llama.cpp's default list, typ_p before top_p: typical is off, so its place is free. */
    CHECK_EQ(post(R"(,"samplers":["penalties","dry","top_n_sigma","top_k","typ_p","top_p","min_p","xtc","temperature"])").status, 200);
    { std::lock_guard<std::mutex> lk(f.sched.mu); CHECK(f.sched.last_sp.temp_last); }
    CHECK_EQ(post(R"(,"samplers":["top_p"])").status, 200);
    {
        std::lock_guard<std::mutex> lk(f.sched.mu);
        CHECK_EQ(f.sched.last_sp.top_k, 0);                       /* left out: off */
        CHECK_NEAR(f.sched.last_sp.temp, 1.0, 1e-6);
    }
    HttpResponse bad = post(R"(,"min_p":0.1,"samplers":["top_p","temperature","min_p"])");
    CHECK_EQ(bad.status, 400);
    CHECK_EQ(json::parse(bad.body)["error"]["param"].get<std::string>(), std::string("samplers"));
    CHECK_EQ(post(R"(,"samplers":["infill"])").status, 400);
}

/* OpenRouter's session_id travels to the engine as the conversation's label, and nothing else. */
TEST(the_session_id_labels_the_request) {
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"session_id":"conv-7"})").status,
             200);
    f.sched.join_all();
    std::lock_guard<std::mutex> lk(f.sched.mu);
    CHECK_EQ(f.sched.last_label, std::string("conv-7"));
}

/* ASSISTANT PREFILL: a conversation ending in an assistant message is continued unless the
 * request asks for a generation prompt; asking for both explicitly is refused. */
TEST(a_trailing_assistant_message_is_continued) {
    Fixture f;
    const std::string msgs =
        R"("messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"The"}])";
    CHECK_EQ(f.post("/v1/chat/completions", "{" + msgs + "}").status, 200);
    CHECK_EQ(f.tmpl.last_opt.continue_final, std::string("auto"));
    CHECK(!f.tmpl.last_opt.add_generation_prompt);
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"continue_final_message":"content"})").status, 200);
    CHECK_EQ(f.tmpl.last_opt.continue_final, std::string("content"));
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"add_generation_prompt":true})").status, 200);
    CHECK(f.tmpl.last_opt.continue_final.empty());
    CHECK(f.tmpl.last_opt.add_generation_prompt);
    CHECK_EQ(f.post("/v1/chat/completions",
                    "{" + msgs + R"(,"continue_final_message":true,"add_generation_prompt":true})")
                 .status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"continue_final_message":true})")
                 .status, 400);
    f.sched.join_all();
}

/* The deprecated functions API, llama-server's json_schema and reasoning_format reach the
 * template as the fields they spell. */
TEST(older_and_llama_server_chat_spellings_reach_the_template) {
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],
                        "functions":[{"name":"f","parameters":{"type":"object"}}]})").status, 200);
    CHECK_EQ(f.tmpl.last_n_tools, 1);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"functions":[],"tools":[]})")
                 .status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"reasoning_format":"none"})")
                 .status, 200);
    CHECK_EQ(f.tmpl.last_opt.reasoning_format, std::string("none"));
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],
                        "reasoning_format":"deepseek-legacy"})").status, 400);
    /* The fake template makes no grammar of a schema, so this one is refused after the render;
     * what is checked is that the schema reached it. */
    (void)f.post("/v1/chat/completions",
                 R"({"messages":[{"role":"user","content":"hi"}],"json_schema":{"type":"object"}})");
    CHECK_EQ(f.tmpl.last_opt.json_schema, std::string(R"({"type":"object"})"));
    f.sched.join_all();
}

/* The routes llama-server answers and this server now does too: without /v1, the token count,
 * the rendered prompt, and llama-server's /tokenize and /detokenize shapes. */
TEST(llama_server_routes_answer) {
    Fixture f;
    CHECK_EQ(f.post("/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})").status,
             200);
    CHECK_EQ(f.post("/completions", R"({"prompt":"ab"})").status, 200);
    CHECK_EQ(f.get("/models").status, 200);
    CHECK_EQ(f.get("/v1/health").status, 200);
    f.sched.join_all();

    HttpResponse at = f.post("/apply-template", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(at.status, 200);
    const std::string prompt = json::parse(at.body).value("prompt", std::string());
    CHECK(has(prompt, "<user>hi"));
    HttpResponse it = f.post("/v1/chat/completions/input_tokens",
                             R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(it.status, 200);
    CHECK_EQ(json::parse(it.body).value("input_tokens", (int64_t)-1), (int64_t)prompt.size());

    HttpResponse t = f.post("/tokenize", R"({"content":["ab",7]})");
    CHECK_EQ(t.status, 200);
    CHECK(json::parse(t.body)["tokens"] == json::array({ 97, 98, 7 }));
    HttpResponse tp = f.post("/tokenize", R"({"content":"a","with_pieces":true})");
    CHECK_EQ(json::parse(tp.body)["tokens"][0].value("piece", std::string()), std::string("a"));
    HttpResponse d = f.post("/detokenize", R"({"tokens":[104,105]})");
    CHECK_EQ(json::parse(d.body).value("content", std::string()), std::string("hi"));
    CHECK_EQ(json::parse(d.body).value("prompt", std::string()), std::string("hi"));

    const json p = json::parse(f.get("/props").body);
    CHECK_EQ(p.value("role", std::string()), std::string("model"));
    CHECK(p["default_generation_settings"]["params"]["samplers"].is_array());
    CHECK_EQ(p["default_generation_settings"]["params"].value("mirostat", -1), 0);
}

/* A prompt the context cannot hold names itself in both of the phrasings the agents that compact
 * on overflow look for, with llama-server's two numbers beside OpenAI's code. */
TEST(an_overflow_reads_as_one_to_the_clients_that_compact) {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    FakeGrammar gram;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    d.chat = &tmpl;
    d.grammar = &gram;
    ServerOptions o;
    o.model_id = "test-model";
    o.max_ctx = 8;
    Server srv(d, o);
    HttpResponse r = srv.router().call("POST", "/v1/chat/completions",
                                       R"({"messages":[{"role":"user","content":"a long prompt"}]})");
    CHECK_EQ(r.status, 400);
    const json e = json::parse(r.body)["error"];
    CHECK_EQ(e.value("code", std::string()), std::string("context_length_exceeded"));
    CHECK(has(e.value("message", std::string()), "maximum context length is 8 tokens"));
    CHECK(has(e.value("message", std::string()), "exceeds the available context size"));
    CHECK_EQ(e.value("n_ctx", (int64_t)0), (int64_t)8);
    CHECK(e.value("n_prompt_tokens", (int64_t)0) > 8);
    sched.join_all();
}

/* ================================================================== Anthropic's Messages API */

/* The translation is the chat request the Anthropic one means: the system blocks as one system
 * message with Claude Code's billing stamp held still, a tool result as a tool message ahead of
 * the user's own words, the tools and the choice, thinking, stop sequences and the label. */
TEST(an_anthropic_request_becomes_the_chat_request_it_means) {
    OaiLimits lim;
    std::string chat, label;
    ApiError e;
    CHECK_OK(anthropic_to_chat(R"({
        "model":"x","max_tokens":9,"stop_sequences":["END"],
        "system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=1; cch=a5145;"},
                  {"type":"text","text":"Be terse.","cache_control":{"type":"ephemeral"}}],
        "metadata":{"user_id":"user_1_session_9"},
        "thinking":{"type":"disabled"},
        "tools":[{"name":"get","description":"d","input_schema":{"type":"object"}}],
        "tool_choice":{"type":"any","disable_parallel_tool_use":true},
        "messages":[
          {"role":"user","content":"weather?"},
          {"role":"assistant","content":[{"type":"text","text":"Looking."},
                                         {"type":"tool_use","id":"t1","name":"get","input":{"c":"P"}}]},
          {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"sun"},
                                    {"type":"text","text":"thanks"}]}]})",
        lim, &chat, &label, &e));
    const json c = json::parse(chat);
    CHECK_EQ(c["messages"][0].value("role", ""), std::string("system"));
    CHECK(has(c["messages"][0].value("content", ""), "cch=fffff;"));
    CHECK(has(c["messages"][0].value("content", ""), "Be terse."));
    CHECK_EQ(c["messages"][2]["tool_calls"][0]["function"].value("arguments", ""),
             std::string(R"({"c":"P"})"));
    CHECK_EQ(c["messages"][3].value("role", ""), std::string("tool"));        /* result first */
    CHECK_EQ(c["messages"][3].value("tool_call_id", ""), std::string("t1"));
    CHECK_EQ(c["messages"][4].value("role", ""), std::string("user"));        /* then the words */
    CHECK_EQ(c.value("tool_choice", ""), std::string("required"));
    CHECK(!c.value("parallel_tool_calls", true));
    CHECK(!c.value("enable_thinking", true));
    CHECK_EQ(c["stop"][0].get<std::string>(), std::string("END"));
    CHECK_EQ(label, std::string("user_1_session_9"));

    /* What would change the answer and cannot be served is refused, by name. */
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":[{"type":"document"}]}]})",
                            lim, &chat, &label, &e) < 0);
    CHECK(anthropic_to_chat(R"({"messages":[{"role":"user","content":"x"}],
                                "tools":[{"type":"web_search_20250305","name":"web_search"}]})",
                            lim, &chat, &label, &e) < 0);
}

/* The answer and the errors in Anthropic's shapes, through the server. */
TEST(the_messages_api_answers_in_anthropics_shapes) {
    Fixture f;
    HttpResponse r = f.post("/v1/messages",
                            R"({"model":"x","max_tokens":16,"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    const json m = json::parse(r.body);
    CHECK_EQ(m.value("type", ""), std::string("message"));
    CHECK(has(m.value("id", ""), "msg_"));
    CHECK_EQ(m["content"][0].value("type", ""), std::string("text"));
    CHECK_EQ(m["content"][0].value("text", ""), std::string("hello"));
    CHECK_EQ(m.value("stop_reason", ""), std::string("end_turn"));
    CHECK(m["usage"].contains("input_tokens") && m["usage"].contains("output_tokens"));
    f.sched.join_all();

    HttpResponse bad = f.post("/v1/messages", R"({"model":"x","max_tokens":16})");
    CHECK_EQ(bad.status, 400);
    const json e = json::parse(bad.body);
    CHECK_EQ(e.value("type", ""), std::string("error"));
    CHECK_EQ(e["error"].value("type", ""), std::string("invalid_request_error"));

    HttpResponse n = f.post("/v1/messages/count_tokens",
                            R"({"model":"x","messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(n.status, 200);
    CHECK(json::parse(n.body).value("input_tokens", (int64_t)0) > 0);

    /* The stream: message_start, a text block opened, filled and stopped, then message_delta
     * and message_stop -- and no [DONE]. */
    HttpResponse s = f.post("/v1/messages", R"({"model":"x","max_tokens":16,"stream":true,
                                               "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(s.status, 200);
    const std::string body = drain(s);
    std::vector<std::string> types;
    for (size_t at = 0; (at = body.find("event: ", at)) != std::string::npos;) {
        const size_t nl = body.find('\n', at);
        types.push_back(body.substr(at + 7, nl - at - 7));
        at = nl;
    }
    CHECK(!types.empty() && types.front() == "message_start");
    CHECK(types.size() >= 2 && types.back() == "message_stop");
    CHECK(std::find(types.begin(), types.end(), "content_block_start") != types.end());
    CHECK(std::find(types.begin(), types.end(), "message_delta") != types.end());
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();
}

/* ================================================================== OpenAI's Responses API */

/* The translation: instructions as the system message, input items as the turns they are --
 * a reasoning item and a call joined into one assistant turn, the call's output as a tool message
 * -- function tools only, max_output_tokens, reasoning.effort, text.format, and every field the
 * Responses API does not define passed through. */
TEST(a_responses_request_becomes_the_chat_request_it_means) {
    OaiLimits lim;
    std::string chat, echo;
    ApiError e;
    CHECK_OK(responses_to_chat(R"({
        "model":"m","instructions":"Be terse.","max_output_tokens":7,
        "reasoning":{"effort":"low"},"top_k":5,
        "text":{"format":{"type":"json_schema","name":"a","schema":{"type":"object"}}},
        "tools":[{"type":"function","name":"get","parameters":{"type":"object"}},
                 {"type":"web_search"}],
        "input":[
          {"role":"user","content":"weather?"},
          {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"think"}]},
          {"type":"function_call","call_id":"c1","name":"get","arguments":"{}"},
          {"type":"function_call_output","call_id":"c1","output":"sun"}]})",
        lim, &chat, &echo, &e));
    const json c = json::parse(chat);
    CHECK_EQ(c["messages"][0].value("role", ""), std::string("system"));
    CHECK_EQ(c["messages"][1].value("role", ""), std::string("user"));
    CHECK_EQ(c["messages"][2].value("reasoning_content", ""), std::string("think"));
    CHECK_EQ(c["messages"][2]["tool_calls"][0].value("id", ""), std::string("c1"));
    CHECK_EQ(c["messages"][3].value("role", ""), std::string("tool"));
    CHECK_EQ(c["tools"].size(), (size_t)1);                       /* web_search left out */
    CHECK_EQ(c.value("max_tokens", 0), 7);
    CHECK_EQ(c.value("reasoning_effort", ""), std::string("low"));
    CHECK_EQ(c.value("top_k", 0), 5);                             /* passed through */
    CHECK_EQ(c["response_format"].value("type", ""), std::string("json_schema"));
    CHECK(!c.contains("input") && !c.contains("text") && !c.contains("instructions"));
    CHECK_EQ(json::parse(echo).value("instructions", ""), std::string("Be terse."));

    CHECK(responses_to_chat(R"({"input":"x","previous_response_id":"resp_1"})", lim, &chat,
                            &echo, &e) < 0);
}

TEST(the_responses_api_answers_with_a_response_object_and_its_events) {
    Fixture f;
    HttpResponse r = f.post("/v1/responses", R"({"model":"m","input":"hi"})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o.value("object", ""), std::string("response"));
    CHECK(has(o.value("id", ""), "resp_"));
    CHECK_EQ(o.value("status", ""), std::string("completed"));
    CHECK_EQ(o["output"][0].value("type", ""), std::string("message"));
    CHECK_EQ(o["output"][0]["content"][0].value("text", ""), std::string("hello"));
    CHECK(o["usage"].contains("total_tokens"));
    f.sched.join_all();

    HttpResponse n = f.post("/v1/responses/input_tokens", R"({"model":"m","input":"hi"})");
    CHECK_EQ(json::parse(n.body).value("object", ""), std::string("response.input_tokens"));

    HttpResponse s = f.post("/v1/responses", R"({"model":"m","input":"hi","stream":true})");
    const std::string body = drain(s);
    std::vector<std::string> types;
    int64_t last_seq = -1;
    bool ordered = true;
    for (size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const size_t nl = body.find('\n', at);
        const json ev = json::parse(body.substr(at + 6, nl - at - 6), nullptr, false);
        at = nl;
        if (!ev.is_object()) continue;
        types.push_back(ev.value("type", ""));
        const int64_t q = ev.value("sequence_number", (int64_t)-1);
        ordered = ordered && q == last_seq + 1;
        last_seq = q;
    }
    CHECK(ordered);
    CHECK(!types.empty() && types.front() == "response.created");
    CHECK(!types.empty() && types.back() == "response.completed");
    CHECK(std::find(types.begin(), types.end(), "response.output_text.delta") != types.end());
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();
}

/* ================================================================== llama-server's native shape */

/* /completion and /completions answer llama-server's native result: the text, how it ended, the
 * counts, the settings it ran with -- with the generated ids when asked -- and a prompt of strings
 * and ids mixed is one prompt. */
TEST(the_native_completion_answers_in_llama_servers_shape) {
    Fixture f;
    HttpResponse r = f.post("/completion", R"({"prompt":["a",98],"n_predict":10,"return_tokens":true})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o.value("content", ""), std::string("hello"));
    CHECK(o.value("stop", false));
    CHECK_EQ(o.value("stop_type", ""), std::string("eos"));
    CHECK_EQ(o.value("tokens_evaluated", (int64_t)0), (int64_t)2);      /* 'a' and 98: one prompt */
    CHECK_EQ(o["tokens"].size(), (size_t)5);
    CHECK(o.contains("generation_settings") && o["generation_settings"].contains("samplers"));
    f.sched.join_all();

    /* The stream: chunks with the ids that made them, a final result, no [DONE]. */
    HttpResponse s = f.post("/completions", R"({"prompt":"ab","stream":true})");
    const std::string body = drain(s);
    CHECK(has(body, "\"tokens\":["));
    CHECK(has(body, "\"stop\":true"));
    CHECK(!has(body, "[DONE]"));
    f.sched.join_all();

    /* return_tokens is the native endpoint's own; elsewhere it is guarded at its off value. */
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"ab","return_tokens":true})").status, 400);
}

/* ================================================================== logprobs */

/* A sink carries each token's alternatives beside it, for a request that enabled them. */
TEST(the_sink_carries_a_tokens_alternatives) {
    Sink s(16);
    s.enable_logprobs(3);
    TokenLogprobs lp;
    lp.logprob = -0.5f;
    lp.n_top = 2;
    lp.top_id[0] = 7; lp.top_lp[0] = -0.1f;
    lp.top_id[1] = 9; lp.top_lp[1] = -2.0f;
    CHECK(s.push(5, &lp));
    CHECK(s.push(6));
    SinkToken buf[4];
    SinkAlt alts[4 * 3];
    CHECK_EQ((long long)s.read(buf, 4, alts), 2LL);
    CHECK_EQ(buf[0].token, 5);
    CHECK_EQ(buf[0].logprob, -0.5f);
    CHECK_EQ(buf[0].n_alt, 2);
    CHECK_EQ(alts[0].id, 7);
    CHECK_EQ(alts[1].id, 9);
    CHECK_EQ(alts[1].logprob, -2.0f);
    CHECK_EQ(buf[1].n_alt, 0);
}

/* Chat: `logprobs` and `top_logprobs` reach the sampler as n_probs, and every token comes back in
 * OpenAI's shape with llama-server's id beside it. */
TEST(chat_logprobs_come_back_per_token_with_their_alternatives) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],
                                "logprobs":true,"top_logprobs":2})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(f.sched.last_sp.n_probs, 2);
    const json c = json::parse(r.body)["choices"][0]["logprobs"]["content"];
    CHECK_EQ(c.size(), (size_t)3);
    CHECK_EQ(c[1]["token"].get<std::string>(), std::string("b"));
    CHECK_EQ(c[1]["id"].get<int>(), (int)'b');
    CHECK_EQ(c[1]["logprob"].get<float>(), -0.5f);
    CHECK_EQ(c[1]["bytes"][0].get<int>(), (int)'b');
    CHECK_EQ(c[1]["top_logprobs"].size(), (size_t)2);
    CHECK_EQ(c[1]["top_logprobs"][1]["token"].get<std::string>(), std::string("y"));
    CHECK_EQ(c[1]["top_logprobs"][1]["logprob"].get<float>(), -2.0f);
    f.sched.join_all();

    /* `logprobs` alone is the token's own and no alternatives -- OpenAI's default of 0. */
    HttpResponse z = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true})");
    CHECK_EQ(f.sched.last_sp.n_probs, 0);
    const json zc = json::parse(z.body)["choices"][0]["logprobs"]["content"];
    CHECK_EQ(zc.size(), (size_t)3);
    CHECK(zc[0]["top_logprobs"].empty());
    f.sched.join_all();

    /* Streamed: each token's entry rides a chunk, and together they are the reply's. */
    HttpResponse s = f.post("/v1/chat/completions",
                            R"({"messages":[{"role":"user","content":"hi"}],"stream":true,
                                "logprobs":true,"top_logprobs":1})");
    const std::string body = drain(s);
    size_t n = 0;
    for (size_t at = body.find("data: {"); at != std::string::npos;
         at = body.find("data: {", at + 1)) {
        const json ev = json::parse(body.substr(at + 6, body.find("\n\n", at) - at - 6));
        const json& ch = ev["choices"][0];
        if (ch.contains("logprobs") && ch["logprobs"].is_object())
            n += ch["logprobs"]["content"].size();
    }
    CHECK_EQ(n, (size_t)3);
    f.sched.join_all();
}

TEST(logprobs_are_refused_out_of_range_and_with_echo) {
    Fixture f(true, 0, "", true);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"logprobs":true,
                        "top_logprobs":21})").status, 400);
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"top_logprobs":2})").status,
             400);
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"hi","logprobs":2,"echo":true})").status,
             400);
    CHECK_EQ(f.post("/completion", R"({"prompt":"hi","n_probs":2,"post_sampling_probs":true})")
                 .status, 400);
}

/* Legacy completions: OpenAI's parallel arrays, with llama-server's content array beside them. */
TEST(completion_logprobs_have_the_legacy_shape) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":2})");
    CHECK_EQ(r.status, 200);
    const json lp = json::parse(r.body)["choices"][0]["logprobs"];
    CHECK_EQ(lp["tokens"].size(), (size_t)3);
    CHECK_EQ(lp["tokens"][2].get<std::string>(), std::string("c"));
    CHECK_EQ(lp["token_logprobs"][2].get<float>(), -0.75f);
    CHECK_EQ(lp["top_logprobs"][0]["x"].get<float>(), -1.0f);
    CHECK_EQ(lp["text_offset"][2].get<int>(), 2);
    CHECK_EQ(lp["content"].size(), (size_t)3);
    f.sched.join_all();
}

/* A stop string's own tokens never went out, so their logprobs do not either. */
TEST(logprobs_end_where_the_stop_string_cut_the_reply) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "abXYc";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1,"stop":["XY"]})");
    const json o = json::parse(r.body)["choices"][0];
    CHECK_EQ(o["text"].get<std::string>(), std::string("ab"));
    CHECK_EQ(o["logprobs"]["tokens"].size(), (size_t)2);
    f.sched.join_all();
}

/* llama-server's native n_probs: completion_probabilities on the answer, and on each streamed
 * chunk instead when streaming. */
TEST(native_n_probs_return_completion_probabilities) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "ab";
    HttpResponse r = f.post("/completion", R"({"prompt":"hi","n_probs":3})");
    CHECK_EQ(r.status, 200);
    const json o = json::parse(r.body);
    CHECK_EQ(o["completion_probabilities"].size(), (size_t)2);
    CHECK_EQ(o["completion_probabilities"][0]["top_logprobs"].size(), (size_t)3);
    CHECK_EQ(o["completion_probabilities"][0]["id"].get<int>(), (int)'a');
    CHECK_EQ(o["generation_settings"]["n_probs"].get<int>(), 3);
    f.sched.join_all();

    HttpResponse s = f.post("/completion", R"({"prompt":"hi","n_probs":1,"stream":true})");
    const std::string body = drain(s);
    CHECK(has(body, "\"completion_probabilities\":[{"));
    f.sched.join_all();
}

/* A model with no fill-in-the-middle tokens -- the fixture's -- cannot infill, and says so. */
TEST(infill_without_fim_tokens_is_refused_by_name) {
    Fixture f;
    HttpResponse r = f.post("/infill", R"({"input_prefix":"def f():\n    ","input_suffix":"\n"})");
    CHECK_EQ(r.status, 501);
    CHECK(has(r.body, "fill-in-the-middle"));
}

/* With the markers, a fill is served; a field of the wrong type, or a body nested past the
 * bound every generation endpoint holds, is a 400. */
TEST(infill_serves_a_fill_and_refuses_a_malformed_body) {
    Fixture f;
    f.tok.fim = true;
    CHECK_EQ(f.post("/infill", R"({"input_prefix":"a","input_suffix":"b",
                                   "input_extra":[{"filename":"x.py","text":"y"}]})").status, 200);
    f.sched.join_all();
    for (const char* body : { R"({"input_prefix":"a","n_predict":"many"})",
                              R"({"input_prefix":"a","max_tokens":1.5})",
                              R"({"input_prefix":"a","input_extra":[{"filename":5}]})",
                              R"({"input_prefix":"a","input_extra":[{"text":["y"]}]})" })
        CHECK_EQ(f.post("/infill", body).status, 400);
    const std::string deep = R"({"input_prefix":"a","input_extra":)" + std::string(2000, '[') +
                             std::string(2000, ']') + "}";
    CHECK_EQ(f.post("/infill", deep).status, 400);
}

/* ================================================================== malformed and failed */

/* A FIELD OF THE WRONG TYPE is a 400 naming it on every API, never a 500: the translators read
 * the client's fields before the chat parser checks them, and the chat parser reads a few before
 * its own validation. */
TEST(a_field_of_the_wrong_type_is_a_400_on_every_api) {
    Fixture f;
    struct Case { const char* path; const char* body; const char* param; };
    for (const Case& c : {
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":1,"content":"hi"}]})",
                   "messages[].role" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":"user",
                   "content":[{"type":"text","text":5}]}]})", "messages[].content[].text" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"messages":[{"role":"user",
                   "content":[{"type":"image","source":"x"}]}]})", "messages[].content[].source" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"system":[{"type":"text","text":[]}],
                   "messages":[{"role":"user","content":"hi"}]})", "system[].text" },
             Case{ "/v1/messages", R"({"model":"x","max_tokens":9,"tool_choice":{"type":"auto",
                   "disable_parallel_tool_use":"yes"},"messages":[{"role":"user","content":"hi"}]})",
                   "disable_parallel_tool_use" },
             Case{ "/v1/chat/completions", R"({"messages":["x",{"role":"assistant","content":"hi"}]})",
                   "messages[]" },
             Case{ "/v1/chat/completions", R"({"messages":[{"role":"user",
                   "content":[{"type":"text","text":5}]}]})", "messages[].content[].text" },
             Case{ "/v1/responses", R"({"input":[{"type":5}]})", "input[].type" },
             Case{ "/v1/responses", R"({"input":[{"role":"user","content":[{"type":"input_text",
                   "text":5}]}]})", "input[].content[].text" },
             Case{ "/v1/responses", R"({"input":[{"type":"function_call","name":"f","arguments":{}}]})",
                   "input[].arguments" },
             Case{ "/v1/responses", R"({"input":"hi","tool_choice":{"type":"function","name":7}})",
                   "tool_choice.name" } }) {
        HttpResponse r = f.post(c.path, c.body);
        CHECK_EQ(r.status, 400);
        const json e = json::parse(r.body, nullptr, false);
        CHECK(e.is_object() && e.contains("error"));
        if (e.is_object() && e["error"].contains("param"))
            CHECK_EQ(e["error"]["param"].get<std::string>(), std::string(c.param));
        else
            CHECK(has(r.body, c.param));
    }
}

/* A STREAM THE ENGINE FAILED ends in its API's own failure, never in its success event: a
 * client that stops reading at the terminal event must not take the partial reply as whole. */
TEST(a_failed_stream_ends_in_its_apis_own_failure) {
    Fixture f;
    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_UNSUPPORTED;
    f.sched.fail_why = "boom";

    HttpResponse a = f.post("/v1/messages", R"({"model":"x","max_tokens":16,"stream":true,
                                               "messages":[{"role":"user","content":"hi"}]})");
    const std::string ab = drain(a);
    CHECK(has(ab, "event: error\ndata: {\"type\":\"error\""));
    CHECK(has(ab, "boom"));
    CHECK(!has(ab, "message_delta"));
    CHECK(!has(ab, "message_stop"));
    f.sched.join_all();

    HttpResponse r = f.post("/v1/responses", R"({"model":"m","input":"hi","stream":true})");
    const std::string rb = drain(r);
    CHECK(has(rb, "event: response.failed"));
    CHECK(has(rb, "\"status\":\"failed\""));
    CHECK(has(rb, "boom"));
    CHECK(!has(rb, "response.completed"));
    f.sched.join_all();

    HttpResponse c = f.post("/v1/chat/completions", R"({"stream":true,
                                                       "messages":[{"role":"user","content":"hi"}]})");
    const std::string cb = drain(c);
    CHECK(has(cb, "\"finish_reason\":\"error\""));
    CHECK(has(cb, "boom"));
    f.sched.join_all();
}

/* ================================================================== the functions API */

/* OpenAI's DEPRECATED functions API end to end: the reply's call is `function_call` with
 * finish_reason "function_call", streamed and not, one call at most; and the next turn -- the
 * assistant's function_call and the `function` message with its result -- is served. */
TEST(the_functions_api_answers_and_continues_in_its_own_shape) {
    Fixture f;
    f.sched.reply = R"(TOOL:get:{"c":"P"})";
    const std::string fns = R"("functions":[{"name":"get","parameters":{"type":"object"}}],)";
    HttpResponse r = f.post("/v1/chat/completions",
                            "{" + fns + R"("messages":[{"role":"user","content":"weather?"}]})");
    CHECK_EQ(r.status, 200);
    CHECK(!f.tmpl.last_opt.parallel_tool_calls);
    const json c = json::parse(r.body)["choices"][0];
    CHECK_EQ(c["finish_reason"].get<std::string>(), std::string("function_call"));
    CHECK_EQ(c["message"]["function_call"]["name"].get<std::string>(), std::string("get"));
    CHECK_EQ(c["message"]["function_call"]["arguments"].get<std::string>(),
             std::string(R"({"c":"P"})"));
    CHECK(!c["message"].contains("tool_calls"));
    f.sched.join_all();

    HttpResponse s = f.post("/v1/chat/completions",
                            "{" + fns + R"("stream":true,"messages":[{"role":"user","content":"w"}]})");
    const std::string body = drain(s);
    CHECK(has(body, R"("delta":{"function_call":{"name":"get","arguments":""}})"));
    CHECK(has(body, R"("finish_reason":"function_call")"));
    CHECK(!has(body, "tool_calls"));
    f.sched.join_all();

    f.sched.reply = "sunny";
    HttpResponse t = f.post("/v1/chat/completions", "{" + fns + R"("messages":[
        {"role":"user","content":"weather?"},
        {"role":"assistant","content":null,"function_call":{"name":"get","arguments":"{\"c\":\"P\"}"}},
        {"role":"function","name":"get","content":"sun"}]})");
    CHECK_EQ(t.status, 200);
    CHECK_EQ(json::parse(t.body)["choices"][0]["message"]["content"].get<std::string>(),
             std::string("sunny"));
    f.sched.join_all();
}

/* ================================================================== odds and ends */

/* What the Responses API cannot carry is refused rather than computed and dropped: several
 * choices, and logprobs in any spelling. The chat stream's own extras are dropped. */
TEST(the_responses_api_refuses_what_a_response_cannot_carry) {
    Fixture f(true, 0, "", true);
    for (const char* extra : { R"("n":2)", R"("logprobs":true)", R"("top_logprobs":3)",
                               R"("n_probs":2)", R"("include":["message.output_text.logprobs"])" })
        CHECK_EQ(f.post("/v1/responses", std::string(R"({"input":"hi",)") + extra + "}").status, 400);
    HttpResponse ok = f.post("/v1/responses", R"({"input":"hi","stream":true,"return_progress":true,
                                                  "stream_options":{"include_usage":true},
                                                  "include":["reasoning.encrypted_content"]})");
    CHECK_EQ(ok.status, 200);
    const std::string body = drain(ok);
    CHECK(!has(body, "chat.completion.chunk"));
    CHECK(has(body, "response.completed"));
    f.sched.join_all();
}

/* text_offset is where each token's text starts in the reply, streamed or not, even where a token
 * carries half a character and its own text is empty. */
TEST(text_offset_is_the_same_streamed_and_not_past_a_split_character) {
    Fixture f(true, 0, "", true);
    f.sched.reply = "a\xC3\xA9" "b";
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1})");
    CHECK_EQ(r.status, 200);
    const json want = json::array({ 0, 1, 2, 3 });
    CHECK(json::parse(r.body)["choices"][0]["logprobs"]["text_offset"] == want);
    f.sched.join_all();

    HttpResponse s = f.post("/v1/completions", R"({"prompt":"hi","logprobs":1,"stream":true})");
    json got = json::array();
    int bad = 0;
    for (const std::string& p : sse_payloads(drain(s), &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        const json& lp = j["choices"][0]["logprobs"];
        if (lp.is_object()) for (const json& o : lp["text_offset"]) got.push_back(o);
    }
    CHECK_EQ(bad, 0);
    CHECK(got == want);
    f.sched.join_all();
}

/* A piece that is not whole UTF-8 goes as its bytes, 0 to 255, as llama-server sends it. */
TEST(tokenize_pieces_carry_bytes_unsigned) {
    Fixture f;
    HttpResponse r = f.post("/tokenize", R"({"content":"\u00e9","with_pieces":true})");
    CHECK_EQ(r.status, 200);
    const json t = json::parse(r.body)["tokens"];
    CHECK_EQ(t.size(), (size_t)2);
    CHECK(t[0]["piece"] == json::array({ 0xC3 }));
    CHECK(t[1]["piece"] == json::array({ 0xA9 }));
}

/* A conversation label is cut to 128 bytes where a character starts, so it stays text. */
TEST(a_long_session_label_is_cut_on_a_character) {
    Fixture f;
    const std::string id = std::string(127, 'a') + "\xC3\xA9";
    CHECK_EQ(f.post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],
                                               "session_id":")" + id + "\"}").status, 200);
    f.sched.join_all();
    std::lock_guard<std::mutex> lk(f.sched.mu);
    CHECK_EQ(f.sched.last_label, std::string(127, 'a'));
}

/* Anthropic's reply names the model the request asked for, as Anthropic's does. */
TEST(the_messages_api_names_the_requested_model) {
    Fixture f;
    HttpResponse r = f.post("/v1/messages", R"({"model":"claude-x","max_tokens":4,
                                               "messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body).value("model", ""), std::string("claude-x"));
    f.sched.join_all();
}

RAD_TEST_MAIN()
