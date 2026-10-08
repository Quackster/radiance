/* core/server/admin.cpp -- vLLM's and llama-server's non-generation endpoints.
 *
 * /version /ping /load /tokenize /detokenize /get_tokenizer_info /server_info /reset_prefix_cache
 * /props.
 *
 * Separate from server.cpp because none of it touches the step path: no admission, no Sink, no
 * cancellation, nothing that outlives the handler. They are questions ABOUT the server, answered
 * on the HTTP thread and finished before they return, and mixing them into the file that owns the
 * streaming lifetime would make that file harder to read for no shared code at all.
 *
 * WHAT IS DELIBERATELY NOT HERE. vLLM's surface also carries /pooling, /classify, /score and
 * /rerank (a classifier or cross-encoder head, which this engine does not load), /sleep, /wake_up
 * and /is_sleeping (there is no sleep mode: weights are placed once, see spec §5), the LoRA
 * adapter endpoints (no LoRA), /invocations (a SageMaker envelope around the others) and the
 * Responses API's stored responses (retrieved or deleted by id: a conversation store, which is a
 * component and not an endpoint); llama-server's carries /slots (its own slot state). Each would
 * be a lie in the shape of a 200, so each is absent and answers 404 rather than pretending.
 */
#include "server.h"

#include "rad_internal.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

namespace rad {
namespace server {

namespace {

/* The body every "how loaded is this server" answer shares. /health and /ping are the same
 * question asked by a load balancer and by a human, and vLLM answers both, so both exist here and
 * neither is a redirect to the other -- a probe that follows a redirect is a probe that reports
 * the redirect's latency. */
/* `queue_cap` AND NOT `queue_depth`. The value is the CEILING -- `opt_.queue_depth` or
 * max_seqs * 8 -- and it never changes while the server runs. Spelling it "queue_depth" would
 * collide head-on with `radiance:http_queue_depth`, which is the OCCUPANCY ("admitted by the HTTP
 * layer and not yet finished"): one name for two opposite quantities on the same server, so a
 * dashboard graphing both plots a flat ceiling beside a number that moves and calls them the same
 * thing. `queue_depth` therefore means occupancy everywhere it appears, and the occupancy is
 * carried here too under its own names -- `in_flight` on /health and /server_info, `server_load`
 * on /load. */
json health_body(const std::string& model, bool stopping, int64_t in_flight, int64_t depth) {
    json o;
    o["status"]      = stopping ? "stopping" : "ok";
    o["model"]       = model;
    o["in_flight"]   = in_flight;
    o["queue_cap"]   = depth;
    return o;
}

/* nlohmann parses "3" and "[1,2]" happily, and both are the wrong shape for every endpoint here.
 * Checking is_object() at the door means each handler's field reads can be plain rather than each
 * one re-deciding what a non-object body means. Bounded like the generation endpoints' bodies
 * (parse_json_bounded): /detokenize takes a token array, which is the body that grows most. */
bool parse_object(const std::string& body, int64_t max_ctx, json* out, ApiError* err) {
    *out = json::object();
    if (body.empty()) return true;              /* {} -- every field takes its default */
    bool over = false, deep = false;
    json j = parse_json_bounded(body, max_ctx, &over, &deep);
    if (deep) {
        *err = err_bad("body", "the request body nests more than " +
                                   std::to_string(kJsonMaxDepth) + " levels deep");
        return false;
    }
    if (over) {
        *err = err_bad("body", "the request body holds more than " +
                                   std::to_string(json_value_budget(max_ctx)) + " JSON values");
        return false;
    }
    if (j.is_discarded()) {
        *err = err_bad("body", "the request body is not valid JSON");
        return false;
    }
    if (!j.is_object()) {
        *err = err_bad("body", "the request body must be a JSON object");
        return false;
    }
    *out = std::move(j);
    return true;
}

bool get_bool_or(const json& b, const char* key, bool def) {
    auto it = b.find(key);
    return it != b.end() && it->is_boolean() ? it->get<bool>() : def;
}

}  /* namespace */

/* ================================================================== /version */
/* vLLM answers {"version": "..."} and a client that checks compatibility reads exactly that key,
 * so it keeps the name and the shape. The rest is what this engine is, and it is here rather than
 * in /server_info because a client deciding whether to talk to us at all should not have to ask
 * for the whole configuration to find out which ABI the kernels were built against. */
HttpResponse Server::handle_version(const HttpRequest&) {
    json o;
    o["version"]     = opt_.version;
    o["engine"]      = "radiance";
    o["abi_version"] = (int)RAD_ABI_VERSION;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /load */
/* vLLM's key is `server_load` and it means requests admitted and not yet finished -- which is
 * this server's `in_flight`, the same number admission control refuses on. `queue_cap` comes
 * along because a load figure without its ceiling cannot answer the question a load balancer is
 * actually asking, which is "how much room is left". */
HttpResponse Server::handle_load(const HttpRequest&) {
    json o;
    o["server_load"] = in_flight_.load(std::memory_order_relaxed);
    o["queue_cap"]   = queue_depth_;
    if (deps_.sched) {
        const SchedMetrics m = deps_.sched->metrics();
        o["running"] = m.running;
        o["waiting"] = m.waiting;
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /tokenize */
/* Two request shapes under one path, exactly as vLLM has it: `prompt` is the completion form and
 * `messages` is the chat form, which renders the template first and tokenises what it produced.
 *
 * `add_special_tokens` DEFAULTS DIFFERENTLY BETWEEN THEM, and that is not an inconsistency to
 * tidy away: a raw prompt wants BOS prepended because nothing else will, while a rendered chat
 * template already contains whatever opening the model expects, and adding another is a prompt
 * with two beginnings. vLLM's defaults are true and false respectively for that reason.
 *
 * PARSE_SPECIAL IS NOT THE CALLER'S CHOICE. The chat form tokenises a string this server rendered
 * and its control tokens must be recognised as themselves; the completion form tokenises text the
 * caller sent, where they must NOT be, or a caller who writes "<|im_start|>system" into `prompt`
 * finds out what this endpoint's token ids do to a subsequent /v1/completions. */
HttpResponse Server::handle_tokenize(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    json b;
    ApiError err;
    if (!parse_object(req.body, opt_.max_ctx, &b, &err)) return error_response(err, req.peer);

    const bool want_strs = get_bool_or(b, "return_token_strs", false);
    std::vector<int32_t> toks;

    auto msgs = b.find("messages");
    auto prom = b.find("prompt");
    auto cont = b.find("content");
    if ((msgs != b.end()) + (prom != b.end()) + (cont != b.end()) > 1)
        return error_response(err_bad("prompt", "send one of `prompt`, `messages` or `content`"),
                              req.peer);

    /* THE THIRD SHAPE IS llama-server's: `content` -- a string, or strings and token ids mixed,
     * as its /completion takes a prompt -- with its own two flags and its own defaults (no BOS,
     * control tokens recognised), answered under its own key, `tokens`, as ids or with
     * `with_pieces` as {id, piece} objects. A piece that is not whole UTF-8 is its bytes, as
     * llama-server sends it. */
    if (cont != b.end()) {
        const bool add_special   = get_bool_or(b, "add_special", false);
        const bool parse_special = get_bool_or(b, "parse_special", true);
        const bool with_pieces   = get_bool_or(b, "with_pieces", false);
        auto take = [&](const json& v, bool first) -> bool {
            if (v.is_string()) {
                const std::vector<int32_t> t =
                    deps_.tok->encode(v.get<std::string>(), first && add_special, parse_special);
                toks.insert(toks.end(), t.begin(), t.end());
                return true;
            }
            int32_t id = 0;
            if (!v.is_number_integer() || !token_id(v, &id)) return false;
            toks.push_back(id);
            return true;
        };
        bool ok = true;
        if (cont->is_array()) {
            bool first = true;
            for (const json& v : *cont) { ok = ok && take(v, first); first = false; }
        } else {
            ok = cont->is_string() && take(*cont, true);
        }
        if (!ok)
            return error_response(err_bad("content", "must be a string, or an array of strings "
                                                     "and token ids"), req.peer);
        json arr = json::array();
        for (int32_t t : toks) {
            if (!with_pieces) { arr.push_back(t); continue; }
            const std::string piece = deps_.tok->token_bytes(t);
            json e;
            e["id"] = t;
            /* A piece that is not whole UTF-8 goes as its bytes, 0-255, as llama-server sends it. */
            if (utf8_valid_prefix(piece) == piece.size()) e["piece"] = piece;
            else e["piece"] = std::vector<int>((const unsigned char*)piece.data(),
                                               (const unsigned char*)piece.data() + piece.size());
            arr.push_back(std::move(e));
        }
        json o;
        o["tokens"]        = std::move(arr);
        o["count"]         = (int64_t)toks.size();
        o["max_model_len"] = opt_.max_ctx;
        metrics_.observe_http(200);
        return HttpResponse::json_body(200, json_dump(o));
    }

    if (msgs != b.end()) {
        if (!deps_.chat)
            return error_response(err_not_implemented(
                "tokenising `messages` requires a chat template; this model was not loaded with "
                "one. Send `prompt` with the text you want tokenised"), req.peer);
        if (!msgs->is_array() || msgs->empty())
            return error_response(err_bad("messages", "must be a non-empty array"), req.peer);

        ChatRenderOptions opt;
        opt.add_generation_prompt = get_bool_or(b, "add_generation_prompt", true);
        apply_template_defaults(b, lim_, &opt);
        auto kw = b.find("chat_template_kwargs");
        if (kw != b.end() && kw->is_object())
            for (auto it = kw->begin(); it != kw->end(); ++it)
                opt.template_kwargs[it.key()] = json_dump(it.value());
        map_reasoning_effort(lim_, &opt);

        json tools = json::array();
        auto t = b.find("tools");
        if (t != b.end() && t->is_array()) tools = *t;

        ChatRender render;
        std::string why;
        if (const int rs = deps_.chat->apply(*msgs, tools, opt, &render, &why); rs < 0)
            return error_response(template_refusal(opt, lim_, rs, why), req.peer);
        toks = deps_.tok->encode(render.prompt,
                                 get_bool_or(b, "add_special_tokens", false),
                                 /*parse_special=*/true);
    } else {
        if (prom == b.end() || !prom->is_string())
            return error_response(err_bad("prompt", "must be a string, or send `messages`"),
                                  req.peer);
        /* Encoded as /v1/completions encodes it, so the count is the count that endpoint sees. */
        toks = deps_.tok->encode(prom->get<std::string>(),
                                 get_bool_or(b, "add_special_tokens", true),
                                 /*parse_special=*/true);
    }

    json o;
    o["count"]         = (int64_t)toks.size();
    o["max_model_len"] = opt_.max_ctx;
    o["tokens"]        = toks;
    if (want_strs) {
        json strs = json::array();
        for (int32_t t : toks) strs.push_back(deps_.tok->piece(t));
        o["token_strs"] = std::move(strs);
    } else {
        o["token_strs"] = nullptr;
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /detokenize */
/* vLLM's response key is `prompt` -- not `text` -- because the endpoint's purpose is to show what
 * a token array would arrive as. The name is odd and it is the name clients read. */
HttpResponse Server::handle_detokenize(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    json b;
    ApiError err;
    if (!parse_object(req.body, opt_.max_ctx, &b, &err)) return error_response(err, req.peer);

    auto t = b.find("tokens");
    if (t == b.end() || !t->is_array())
        return error_response(err_bad("tokens", "must be an array of integers"), req.peer);

    std::vector<int32_t> toks;
    toks.reserve(t->size());
    for (const auto& v : *t) {
        if (!v.is_number_integer())
            return error_response(err_bad("tokens", "must be an array of integers"), req.peer);
        int32_t id = 0;
        if (!token_id(v, &id))
            return error_response(err_bad("tokens", "a token id must be in [0, 2147483647], "
                                                    "got " + v.dump()), req.peer);
        toks.push_back(id);
    }

    /* `prompt` is vLLM's key and `content` llama-server's, for the same text. */
    json o;
    o["prompt"] = deps_.tok->decode(toks);
    o["content"] = o["prompt"];
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /props */
/* llama-server's /props, for the clients built against it: a front-end that sizes its context
 * from default_generation_settings.n_ctx, shows an image button when modalities.vision is set,
 * and reads what a request that sends no sampler fields inherits. /server_info says all of it and
 * more, but nothing written for llama-server knows to ask there.
 *
 * Only what this server can fill honestly, under llama-server's names: no GGUF metadata, no
 * per-slot state. n_ctx is the context ONE request may use, which is what llama-server's per-slot
 * n_ctx means to the clients reading it; left out when the server does not know its length. */
HttpResponse Server::handle_props(const HttpRequest&) {
    const SamplingParams& sp = opt_.default_sampling;
    /* EVERY FIELD llama-server's `params` CARRIES, so a client that reads one does not find it
     * missing. The samplers this server does not run are reported at their off values -- which
     * is what it serves -- and `samplers` is its own order, temperature first. */
    const int64_t n_predict = opt_.default_max_tokens > 0 ? opt_.default_max_tokens : -1;
    json params;
    params["n_predict"]           = n_predict;
    params["max_tokens"]          = n_predict;
    params["seed"]                = -1;
    params["temperature"]         = sp.temp;
    params["dynatemp_range"]      = 0.0;
    params["dynatemp_exponent"]   = 1.0;
    params["top_k"]               = sp.top_k;
    params["top_p"]               = sp.top_p;
    params["min_p"]               = sp.min_p;
    params["top_n_sigma"]         = -1.0;
    params["typical_p"]           = sp.typical_p;
    params["typ_p"]               = sp.typical_p;
    params["xtc_probability"]     = sp.xtc_probability;
    params["xtc_threshold"]       = sp.xtc_threshold;
    params["repeat_last_n"]       = sp.penalty_last_n;
    params["repeat_penalty"]      = sp.rep_penalty;
    params["presence_penalty"]    = sp.pres_penalty;
    params["frequency_penalty"]   = sp.freq_penalty;
    params["dry_multiplier"]      = sp.dry_multiplier;
    params["dry_base"]            = sp.dry_base;
    params["dry_allowed_length"]  = sp.dry_allowed_length;
    params["dry_penalty_last_n"]  = sp.dry_penalty_last_n;
    params["dry_sequence_breakers"] = sp.dry_seq_breakers.empty() && !opt_.default_dry_breakers_set
                                    ? std::vector<std::string>{ "\n", ":", "\"", "*" }
                                    : sp.dry_seq_breakers;
    params["mirostat"]            = 0;
    params["mirostat_tau"]        = 5.0;
    params["mirostat_eta"]        = 0.1;
    params["stop"]                = json::array();
    params["n_keep"]              = 0;
    params["n_discard"]           = 0;
    params["ignore_eos"]          = false;
    params["stream"]              = true;
    params["logit_bias"]          = json::array();
    params["n_probs"]             = 0;
    params["min_keep"]            = 0;
    params["grammar"]             = "";
    params["grammar_lazy"]        = false;
    params["grammar_triggers"]    = json::array();
    params["preserved_tokens"]    = json::array();
    params["chat_format"]         = "";
    params["reasoning_format"]    = "auto";
    params["reasoning_in_content"] = false;
    params["generation_prompt"]   = "";
    params["samplers"] = { "penalties", "dry", "temperature", "top_k", "top_p", "min_p", "typ_p",
                           "xtc" };
    params["backend_sampling"]    = false;
    params["timings_per_token"]   = false;
    params["post_sampling_probs"] = false;
    params["lora"]                = json::array();

    json gen;
    if (opt_.max_ctx > 0) gen["n_ctx"] = opt_.max_ctx;
    gen["params"] = std::move(params);

    json modal;
    modal["vision"] = deps_.mm && deps_.mm->accepts(MediaKind::Image);
    modal["video"]  = deps_.mm && deps_.mm->accepts(MediaKind::Video);
    modal["audio"]  = deps_.mm && deps_.mm->accepts(MediaKind::Audio);

    json o;
    o["default_generation_settings"] = std::move(gen);
    o["total_slots"] = opt_.max_seqs;
    o["model_alias"] = opt_.model_id;
    o["model_path"]  = opt_.model_path;
    o["modalities"]  = std::move(modal);
    o["endpoint_slots"]   = false;
    o["endpoint_props"]   = false;
    o["endpoint_metrics"] = true;
    o["chat_template"]      = deps_.chat ? deps_.chat->source() : std::string();
    o["chat_template_caps"] = json::parse(deps_.chat ? deps_.chat->caps() : std::string("{}"),
                                          nullptr, false);
    std::string bos, eos;
    if (deps_.tok) {
        const ITokenizer::VocabInfo v = deps_.tok->vocab_info();
        if (v.bos >= 0) bos = deps_.tok->piece(v.bos);
        if (v.eos >= 0) eos = deps_.tok->piece(v.eos);
    }
    o["bos_token"]   = bos;
    o["eos_token"]   = eos;
    o["build_info"]  = "radiance " + opt_.version;
    o["role"]        = "model";
    o["is_sleeping"] = false;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /get_tokenizer_info */
/* vLLM returns the HuggingFace tokenizer config verbatim, which this engine does not have: a .rad
 * container carries a normalised vocabulary and not the JSON it was built from. So this answers
 * the questions that config is actually READ for -- how big is the vocabulary, which ids end a
 * generation, is a chat template available -- and says nothing it would have to invent. */
HttpResponse Server::handle_tokenizer_info(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    const ITokenizer::VocabInfo v = deps_.tok->vocab_info();
    json o;
    o["tokenizer_class"]      = v.kind;
    o["vocab_size"]           = v.n_tokens;
    o["model_max_length"]     = opt_.max_ctx;
    o["bos_token_id"]         = v.bos >= 0 ? json(v.bos) : json(nullptr);
    o["eos_token_id"]         = v.eos >= 0 ? json(v.eos) : json(nullptr);
    o["eot_token_id"]         = v.eot >= 0 ? json(v.eot) : json(nullptr);
    o["unk_token_id"]         = v.unk >= 0 ? json(v.unk) : json(nullptr);
    o["pad_token_id"]         = v.pad >= 0 ? json(v.pad) : json(nullptr);
    o["add_bos_token"]        = v.add_bos;
    o["add_eos_token"]        = v.add_eos;
    o["chat_template"]        = deps_.chat != nullptr;
    o["supports_tools"]       = deps_.chat && deps_.chat->supports_tools();
    o["supports_reasoning"]   = deps_.chat && deps_.chat->supports_reasoning();
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== capabilities */
json Server::capabilities() const {
    json caps;
    caps["chat"]      = deps_.chat != nullptr;
    caps["tools"]     = deps_.chat && deps_.chat->supports_tools();
    caps["reasoning"] = deps_.chat && deps_.chat->supports_reasoning();
    caps["grammar"]   = deps_.grammar != nullptr;
    caps["embedding"] = deps_.embed != nullptr;
    caps["image"]     = deps_.mm && deps_.mm->accepts(MediaKind::Image);
    caps["video"]     = deps_.mm && deps_.mm->accepts(MediaKind::Video);
    caps["audio"]     = deps_.mm && deps_.mm->accepts(MediaKind::Audio);
    caps["logprobs"]  = opt_.supports_logprobs;
    /* THE EFFORTS A CHAT REQUEST MAY SEND, as probed from the template at startup: null when it
     * takes any word. "none" and "minimal" are this server's, and turn thinking off. */
    if (lim_.reasoning_efforts.empty()) {
        caps["reasoning_efforts"] = nullptr;
    } else {
        std::vector<std::string> e = { "none", "minimal" };
        e.insert(e.end(), lim_.reasoning_efforts.begin(), lim_.reasoning_efforts.end());
        caps["reasoning_efforts"] = e;
    }
    return caps;
}

/* ================================================================== /server_info */
/* vLLM returns the repr() of its config object, which is a Python string a client can only print.
 * This returns the same information as JSON, in two halves: what the SERVER decided, which is
 * this component's own state, and what the ENGINE decided, which arrives pre-rendered because the
 * server knows nothing about devices, containers or placement (ServerOptions::engine_info).
 *
 * `endpoints` is the router's own table rather than a list kept beside it. A hand-written list is
 * a list that goes stale the first time somebody adds a route, and this endpoint exists precisely
 * so that a client does not have to guess what is served. */
HttpResponse Server::handle_server_info(const HttpRequest&) {
    json srv;
    srv["model"]              = opt_.model_id;
    srv["version"]            = opt_.version;
    srv["max_model_len"]      = opt_.max_ctx;
    srv["max_num_seqs"]       = opt_.max_seqs;
    srv["queue_cap"]          = queue_depth_;
    /* null is `--default-max-tokens auto`: what the context leaves after the prompt. */
    srv["default_max_tokens"] = opt_.default_max_tokens > 0 ? json(opt_.default_max_tokens)
                                                            : json(nullptr);
    srv["max_tokens_cap"]     = opt_.max_tokens_cap > 0 ? json(opt_.max_tokens_cap) : json(nullptr);
    srv["max_n"]              = opt_.max_n;
    srv["max_stop_strings"]   = opt_.max_stops;
    srv["max_stop_bytes"]     = opt_.max_stop_bytes;
    srv["uptime_s"]           = unix_now() - started_at_;
    srv["in_flight"]          = in_flight_.load(std::memory_order_relaxed);

    json caps = capabilities();

    /* WHAT A REQUEST THAT LEAVES A FIELD OUT GETS, under the request's own spelling of each
     * field, so a client can read off exactly what leaving one out means on this deployment. */
    const SamplingParams& sp = opt_.default_sampling;
    json defs;
    defs["temperature"]        = sp.temp;
    defs["top_k"]              = sp.top_k;
    defs["top_p"]              = sp.top_p;
    defs["min_p"]              = sp.min_p;
    defs["typical_p"]          = sp.typical_p;
    defs["presence_penalty"]   = sp.pres_penalty;
    defs["frequency_penalty"]  = sp.freq_penalty;
    defs["repetition_penalty"] = sp.rep_penalty;
    defs["repeat_last_n"]      = sp.penalty_last_n;
    defs["dry_multiplier"]     = sp.dry_multiplier;
    defs["dry_base"]           = sp.dry_base;
    defs["dry_allowed_length"] = sp.dry_allowed_length;
    defs["dry_penalty_last_n"] = sp.dry_penalty_last_n;
    defs["dry_sequence_breakers"] = opt_.default_dry_breakers_set
                                        ? json(sp.dry_seq_breakers)
                                        : json(std::vector<std::string>{ "\n", ":", "\"", "*" });
    defs["xtc_probability"]    = sp.xtc_probability;
    defs["xtc_threshold"]      = sp.xtc_threshold;
    defs["max_tokens"]         = srv["default_max_tokens"];
    defs["reasoning_effort"]   = opt_.default_reasoning_effort.empty()
                                     ? json(nullptr) : json(opt_.default_reasoning_effort);
    json emap = json::object();
    for (const auto& [from, to] : lim_.reasoning_effort_map) emap[from] = to;
    defs["reasoning_effort_map"] = std::move(emap);
    defs["thinking_token_budget"] = opt_.default_thinking_budget < 0
                                        ? json(nullptr) : json(opt_.default_thinking_budget);
    json kw = json::object();
    for (const auto& [k, v] : opt_.default_template_kwargs) {
        json parsed = json::parse(v, nullptr, false);
        kw[k] = parsed.is_discarded() ? json(v) : std::move(parsed);
    }
    defs["chat_template_kwargs"] = std::move(kw);
    defs["reasoning_format"]     = opt_.reasoning_format;

    json o;
    o["server"]           = std::move(srv);
    o["request_defaults"] = std::move(defs);
    o["capabilities"]     = std::move(caps);
    o["endpoints"]    = router_.routes();

    /* Spliced rather than re-parsed field by field: the engine owns what it says about itself and
     * this component is a pipe for it. A blob that is not an object is dropped with a warning
     * instead of corrupting the response -- an unparseable /server_info is worse than one that is
     * missing a section. */
    if (!opt_.engine_info.empty()) {
        json e = json::parse(opt_.engine_info, nullptr, false);
        if (e.is_discarded() || !e.is_object())
            RAD_WARN("server: engine_info is not a JSON object; /server_info omits it");
        else
            o["engine"] = std::move(e);
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /reset_prefix_cache */
/* 501 when there is no prefix cache to reset, and that is the point: vLLM answers 200 with an
 * empty body whether or not anything happened, so a benchmark that thinks it started cold has no
 * way to find out it did not. This says how many entries went. */
HttpResponse Server::handle_reset_prefix_cache(const HttpRequest& req) {
    if (!deps_.sched)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    const int64_t n = deps_.sched->reset_prefix_cache();
    if (n < 0)
        return error_response(err_not_implemented(
            "this engine has no prefix cache (started with --no-prefix-cache, or the model "
            "declares no cacheable KV group)"), req.peer);

    json o;
    o["entries_dropped"] = n;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /health, /ping */
/* NOT counted in the HTTP status metrics, and neither is /metrics. A load balancer probing every
 * second would otherwise dominate `request_success_total` and make the 2xx/5xx ratio a statement
 * about the probe rather than about the service. */
HttpResponse Server::handle_health(const HttpRequest& req) { return handle_ping(req); }

HttpResponse Server::handle_ping(const HttpRequest&) {
    const bool stopping = stopping_.load();
    return HttpResponse::json_body(
        stopping ? 503 : 200,
        json_dump(health_body(opt_.model_id, stopping,
                              in_flight_.load(std::memory_order_relaxed), queue_depth_)));
}

}  /* namespace server */
}  /* namespace rad */
