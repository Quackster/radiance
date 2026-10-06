#include "oai.h"
#include "json_partial.h"
#include "rad_internal.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <random>

namespace rad {
namespace server {

const char* const SSE_DONE = "data: [DONE]\n\n";

/* ================================================================== small helpers */

int64_t unix_now() { return (int64_t)std::time(nullptr); }

std::string json_dump(const json& j) { return j.dump(); }

std::string random_id(const char* prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    uint64_t a = rng(), b = rng();
    std::string s = prefix;
    for (int i = 0; i < 16; ++i) s.push_back(hex[(a >> (i * 4)) & 0xF]);
    for (int i = 0; i < 8; ++i)  s.push_back(hex[(b >> (i * 4)) & 0xF]);
    return s;
}

static const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (uint32_t)p[i + 1] << 8 | p[i + 2];
        out.push_back(kB64[(v >> 18) & 63]); out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(kB64[(v >> 6) & 63]);  out.push_back(kB64[v & 63]);
    }
    if (i < len) {
        uint32_t v = (uint32_t)p[i] << 16;
        bool two = (i + 1 < len);
        if (two) v |= (uint32_t)p[i + 1] << 8;
        out.push_back(kB64[(v >> 18) & 63]);
        out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(two ? kB64[(v >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

bool base64_decode(const std::string& in, std::vector<uint8_t>* out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    uint32_t acc = 0;
    int bits = 0;
    out->clear();
    out->reserve(in.size() * 3 / 4);
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back((uint8_t)((acc >> bits) & 0xFF));
        }
    }
    return true;
}

/* THE PARSED TREE IS BOUNDED, NOT ONLY THE BODY. A JSON value costs the tree a 16-byte node plus
 * its container's growth slack however few bytes it took on the wire, so `[1,1,1,...]` grows
 * about eightfold in parsing, and a body at the transport's size limit becomes gigabytes before
 * a single field has been read -- let alone a prompt checked against the context. Nothing a
 * request can legitimately carry needs many values per token of context: a token-id prompt is
 * one per token and must fit the context, and everything else (messages, tools, schemas, stop
 * strings) is small beside it. So past a budget scaled from the context the parse stops keeping
 * values, and the request is refused having cost a bounded tree. */
int64_t json_value_budget(int64_t max_ctx) {
    const int64_t kFloor = 1 << 20;        /* what a request needs with no context to scale by */
    const int64_t kPerToken = 16;          /* a batch of up to this many full-context prompts */
    return kFloor + kPerToken * std::max<int64_t>(max_ctx, 0);
}

/* AND ITS DEPTH IS BOUNDED, because the parser is the only thing here that walks a tree without
 * recursing. Serialising one (`dump`) and destroying a copy made field by field both recurse once
 * a level, and a request's schema is serialised on the way to its grammar -- so a body of a few
 * hundred kilobytes of brackets, well inside the value budget, would overflow the worker's stack.
 * Nothing a request carries nests anywhere near the bound: a schema deep enough to matter is
 * refused by the grammar compiler at half of it. */
json parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep) {
    const int64_t budget = json_value_budget(max_ctx);
    int64_t n = 0;
    *over = false;
    *deep = false;
    /* Every completed value is counted once: a scalar at `value`, a container at its end. Once
     * over or too deep, every event answers false, which tells the parser to keep nothing
     * further. */
    json::parser_callback_t keep = [&](int depth, json::parse_event_t ev, json&) {
        if (!*deep && depth > kJsonMaxDepth) *deep = true;
        if (!*over && (ev == json::parse_event_t::value ||
                       ev == json::parse_event_t::object_end ||
                       ev == json::parse_event_t::array_end))
            *over = ++n > budget;
        return !*over && !*deep;
    };
    return json::parse(body, keep, /*allow_exceptions=*/false);
}

bool token_id(const json& v, int32_t* out) {
    if (v.is_number_unsigned()) {
        if (v.get<uint64_t>() > (uint64_t)INT32_MAX) return false;
    } else if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > INT32_MAX) {
        return false;
    }
    *out = (int32_t)v.get<int64_t>();
    return true;
}

/* ================================================================== errors */

std::string ApiError::dump() const {
    json e;
    e["message"] = message;
    e["type"] = type;
    if (param.empty()) e["param"] = nullptr; else e["param"] = param;
    if (code.empty()) e["code"] = nullptr; else e["code"] = code;
    json o;
    o["error"] = e;
    return o.dump();
}

ApiError err_bad(const std::string& param, const std::string& why) {
    ApiError e;
    e.status = 400;
    e.param = param;
    e.message = param.empty() ? why : (param + ": " + why);
    return e;
}

ApiError err_unsupported(const std::string& param, const std::string& why) {
    ApiError e = err_bad(param, why);
    e.code = "unsupported_parameter";
    return e;
}

ApiError err_not_implemented(const std::string& what) {
    ApiError e;
    e.status = 501;
    e.type = "not_implemented_error";
    e.message = what;
    return e;
}

ApiError err_overloaded(const std::string& why) {
    ApiError e;
    e.status = 429;
    e.type = "rate_limit_error";
    e.code = "server_overloaded";
    e.message = why;
    return e;
}

/* ================================================================== key vetting
 *
 * The three lists below are the whole policy. `kAccepted*` is what we read; `kInert` is the
 * OpenAI metadata that genuinely does nothing anywhere and would break the official SDKs if it
 * 400'd; `kRejected` is the set we know about and do not implement, each with the reason, so the
 * error message tells a caller something they can act on rather than "unsupported".
 */
namespace {

struct Rejected { const char* key; const char* why; };

const char* kAcceptedCommon[] = {
    "model", "stream", "stream_options", "n", "seed", "stop", "max_tokens",
    "temperature", "top_p", "top_k", "min_p", "typical_p",
    "presence_penalty", "frequency_penalty", "repetition_penalty", "repeat_last_n",
    "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers",
    "grammar", "priority", "logprobs", "ignore_eos",
};

const char* kAcceptedChat[] = {
    "messages", "tools", "tool_choice", "response_format", "top_logprobs",
    "parallel_tool_calls", "add_generation_prompt", "max_completion_tokens",
    "chat_template_kwargs", "reasoning_effort", "enable_thinking", "preserve_thinking",
    "thinking_token_budget", "thinking_budget",
};

const char* kAcceptedCompletion[] = { "prompt", "echo" };

const char* kAcceptedEmbedding[] = { "model", "input", "encoding_format", "dimensions" };

/* Inert by definition in the OpenAI schema: routing and bookkeeping labels the server is free to
 * ignore. Named explicitly so that "we ignore it" is a decision in the source and not an
 * accident of a missing branch. */
const char* kInert[] = {
    "user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key",
};

const Rejected kRejected[] = {
    { "logit_bias",
      "SamplingParams has no per-token bias stage and the device sampler (spec 13) does not "
      "implement one" },
    { "functions",     "the deprecated functions API; send `tools` instead" },
    { "function_call", "the deprecated function_call API; send `tool_choice` instead" },
    { "best_of",       "sampling n candidates and returning the best is not implemented" },
    { "suffix",        "infilling is not implemented" },
    { "mirostat",      "mirostat is not one of the device samplers (spec 13)" },
    { "mirostat_tau",  "mirostat is not one of the device samplers (spec 13)" },
    { "mirostat_eta",  "mirostat is not one of the device samplers (spec 13)" },
    { "top_n_sigma",   "top-n-sigma is not one of the device samplers (spec 13)" },
    { "dynatemp_range",    "dynamic temperature is not one of the device samplers (spec 13)" },
    { "dynatemp_exponent", "dynamic temperature is not one of the device samplers (spec 13)" },
    { "lora",          "LoRA is explicitly out of scope (spec, Scope)" },
    { "modalities",    "output modalities other than text are not implemented" },
    { "audio",         "audio output is not implemented" },
    { "prediction",    "predicted outputs are not implemented" },
    { "continue_final_message", "assistant prefill is not implemented" },
    { "cache_prompt",  "prefix caching is a server-wide setting (spec 7.3), not per request" },
};

bool in_list(const char* const* list, size_t n, const std::string& k) {
    for (size_t i = 0; i < n; ++i) if (k == list[i]) return true;
    return false;
}
#define ARRN(a) (sizeof(a) / sizeof((a)[0]))

int vet_keys(const json& body, Endpoint ep, ApiError* err) {
    for (auto it = body.begin(); it != body.end(); ++it) {
        const std::string& k = it.key();
        if (in_list(kInert, ARRN(kInert), k)) continue;
        if (ep == Endpoint::Embedding) {
            if (in_list(kAcceptedEmbedding, ARRN(kAcceptedEmbedding), k)) continue;
        } else {
            if (in_list(kAcceptedCommon, ARRN(kAcceptedCommon), k)) continue;
            if (ep == Endpoint::Chat &&
                in_list(kAcceptedChat, ARRN(kAcceptedChat), k)) continue;
            if (ep == Endpoint::Completion &&
                in_list(kAcceptedCompletion, ARRN(kAcceptedCompletion), k)) continue;
        }
        for (const auto& r : kRejected) {
            if (k == r.key) { *err = err_unsupported(k, r.why); return RAD_E_UNSUPPORTED; }
        }
        *err = err_unsupported(
            k, "unrecognised parameter for this endpoint. A parameter the server does not "
               "implement is refused rather than ignored, so that a request that returns 200 "
               "means every field of it was honoured");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* Typed getters that refuse rather than coerce: OpenAI clients that send "0.7" as a string are
 * sending a bug, and silently accepting it hides it until the day the string is "warm". */
bool get_num(const json& b, const char* k, double* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_number()) { *err = err_bad(k, "expected a number"); return false; }
    *out = it->get<double>();
    return true;
}
/* EVERY INTEGER FIELD THIS FILE READS IS STORED IN 32 BITS, so a value that does not fit is
 * refused here rather than narrowed at the call site. Narrowing is not a clamp: 2^32 becomes 0
 * and 2^31 becomes negative, and for `max_tokens` both of those read downstream as "no limit".
 * The unsigned branch matters too -- nlohmann keeps a literal above INT64_MAX as unsigned, and
 * reading that as int64 wraps it negative. */
bool get_int(const json& b, const char* k, int64_t* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_number_integer()) { *err = err_bad(k, "expected an integer"); return false; }
    const bool fits = it->is_number_unsigned()
                          ? it->get<uint64_t>() <= (uint64_t)INT32_MAX
                          : it->get<int64_t>() >= INT32_MIN && it->get<int64_t>() <= INT32_MAX;
    if (!fits) {
        *err = err_bad(k, "must fit in a 32-bit signed integer, got " + it->dump());
        return false;
    }
    *out = it->get<int64_t>();
    return true;
}
bool get_bool(const json& b, const char* k, bool* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_boolean()) { *err = err_bad(k, "expected a boolean"); return false; }
    *out = it->get<bool>();
    return true;
}
bool get_str(const json& b, const char* k, std::string* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_string()) { *err = err_bad(k, "expected a string"); return false; }
    *out = it->get<std::string>();
    return true;
}


/* The body of every generation endpoint: one JSON object, parsed into a bounded tree. */
int parse_request_object(const std::string& body, const OaiLimits& lim, json* b, ApiError* err) {
    bool over = false, deep = false;
    *b = parse_json_bounded(body, lim.max_ctx, &over, &deep);
    if (deep) {
        *err = err_bad("", "the request body nests more than " + std::to_string(kJsonMaxDepth) +
                               " levels deep");
        return RAD_E_INVAL;
    }
    if (over) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "the request body holds more than %lld JSON values, which no request this "
                 "server can serve needs", (long long)json_value_budget(lim.max_ctx));
        *err = err_bad("", msg);
        return RAD_E_INVAL;
    }
    if (b->is_discarded() || !b->is_object()) {
        *err = err_bad("", "request body is not a JSON object");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

}  /* namespace */

/* ================================================================== sampling params */

namespace {

/* One row per bounded sampler field. `lo_open` excludes the lower bound; `hi` of infinity is no
 * upper bound. `why` is the refusal when the range alone does not say it well, and null for a
 * closed interval, whose refusal names the interval and the value. */
struct SamplerRange {
    const char* field;
    double      lo, hi;
    bool        lo_open;
    const char* why;
};

constexpr double kInf = std::numeric_limits<double>::infinity();

const SamplerRange kSamplerRanges[] = {
    { "temperature",        0.0,  2.0,  false, nullptr },
    { "top_p",              0.0,  1.0,  true,  "must be in (0, 1]" },
    { "top_k",              0.0,  kInf, false, "must be >= 0 (0 disables it)" },
    { "min_p",              0.0,  1.0,  false, nullptr },
    { "typical_p",          0.0,  1.0,  true,  "must be in (0, 1]" },
    { "presence_penalty",   -2.0, 2.0,  false, nullptr },
    { "frequency_penalty",  -2.0, 2.0,  false, nullptr },
    { "repetition_penalty", 0.0,  kInf, true,  "must be > 0" },
    { "repeat_last_n",      -1.0, kInf, false, "must be >= -1" },
    { "xtc_probability",    0.0,  1.0,  false, nullptr },
    { "xtc_threshold",      0.0,  1.0,  false, nullptr },
    { "dry_base",           1.0,  kInf, false, "must be >= 1" },
    { "dry_penalty_last_n", -1.0, kInf, false, "must be >= -1" },
};

}  /* namespace */

/* Written as "inside the range" and negated, so a NaN -- which a flag can parse and JSON cannot
 * carry -- fails every row rather than passing them all. */
bool sampler_value_ok(const char* field, double v, std::string* why) {
    for (const SamplerRange& r : kSamplerRanges) {
        if (std::strcmp(r.field, field)) continue;
        const bool in = (r.lo_open ? v > r.lo : v >= r.lo) && v <= r.hi;
        if (in) return true;
        if (why) {
            if (r.why) *why = r.why;
            else {
                char b[160];
                snprintf(b, sizeof b, "must be in [%g, %g], got %g", r.lo, r.hi, v);
                *why = b;
            }
        }
        return false;
    }
    return true;
}

static int parse_sampling(const json& b, const OaiLimits& lim, OaiRequest* r, ApiError* err) {
    double d;
    int64_t i64;
    bool bl;
    auto ok = [err](const char* k, double v) {
        std::string why;
        if (sampler_value_ok(k, v, &why)) return true;
        *err = err_bad(k, why);
        return false;
    };

    /* THE DEPLOYMENT'S DEFAULTS ARE THE SEED, and every field below reads its current value
     * before looking for one in the body -- so a field the request names overrides, and a field
     * it omits keeps what the checkpoint asked for. Done here rather than at each call site
     * because a sampler default that reached one endpoint and not the other would make the same
     * model answer differently through /v1/completions than through /v1/chat/completions. */
    r->sp = lim.default_sampling;

    d = r->sp.temp;
    if (!get_num(b, "temperature", &d, err)) return RAD_E_INVAL;
    if (!ok("temperature", d)) return RAD_E_INVAL;
    r->sp.temp = (float)d;

    d = r->sp.top_p;
    if (!get_num(b, "top_p", &d, err)) return RAD_E_INVAL;
    if (!ok("top_p", d)) return RAD_E_INVAL;
    r->sp.top_p = (float)d;

    i64 = r->sp.top_k;
    if (!get_int(b, "top_k", &i64, err)) return RAD_E_INVAL;
    if (!ok("top_k", (double)i64)) return RAD_E_INVAL;
    r->sp.top_k = (int)i64;

    d = r->sp.min_p;
    if (!get_num(b, "min_p", &d, err)) return RAD_E_INVAL;
    if (!ok("min_p", d)) return RAD_E_INVAL;
    r->sp.min_p = (float)d;

    d = r->sp.typical_p;
    if (!get_num(b, "typical_p", &d, err)) return RAD_E_INVAL;
    if (!ok("typical_p", d)) return RAD_E_INVAL;
    r->sp.typical_p = (float)d;

    d = r->sp.pres_penalty;
    if (!get_num(b, "presence_penalty", &d, err)) return RAD_E_INVAL;
    if (!ok("presence_penalty", d)) return RAD_E_INVAL;
    r->sp.pres_penalty = (float)d;

    d = r->sp.freq_penalty;
    if (!get_num(b, "frequency_penalty", &d, err)) return RAD_E_INVAL;
    if (!ok("frequency_penalty", d)) return RAD_E_INVAL;
    r->sp.freq_penalty = (float)d;

    d = r->sp.rep_penalty;
    if (!get_num(b, "repetition_penalty", &d, err)) return RAD_E_INVAL;
    if (!ok("repetition_penalty", d)) return RAD_E_INVAL;
    r->sp.rep_penalty = (float)d;

    i64 = r->sp.penalty_last_n;
    if (!get_int(b, "repeat_last_n", &i64, err)) return RAD_E_INVAL;
    if (!ok("repeat_last_n", (double)i64)) return RAD_E_INVAL;
    r->sp.penalty_last_n = (int)i64;

    d = r->sp.xtc_probability;
    if (!get_num(b, "xtc_probability", &d, err)) return RAD_E_INVAL;
    if (!ok("xtc_probability", d)) return RAD_E_INVAL;
    r->sp.xtc_probability = (float)d;

    d = r->sp.xtc_threshold;
    if (!get_num(b, "xtc_threshold", &d, err)) return RAD_E_INVAL;
    if (!ok("xtc_threshold", d)) return RAD_E_INVAL;
    r->sp.xtc_threshold = (float)d;

    d = r->sp.dry_multiplier;
    if (!get_num(b, "dry_multiplier", &d, err)) return RAD_E_INVAL;
    r->sp.dry_multiplier = (float)d;
    d = r->sp.dry_base;
    if (!get_num(b, "dry_base", &d, err)) return RAD_E_INVAL;
    if (!ok("dry_base", d)) return RAD_E_INVAL;
    r->sp.dry_base = (float)d;
    i64 = r->sp.dry_allowed_length;
    if (!get_int(b, "dry_allowed_length", &i64, err)) return RAD_E_INVAL;
    r->sp.dry_allowed_length = (int)i64;
    i64 = r->sp.dry_penalty_last_n;
    if (!get_int(b, "dry_penalty_last_n", &i64, err)) return RAD_E_INVAL;
    if (!ok("dry_penalty_last_n", (double)i64)) return RAD_E_INVAL;
    r->sp.dry_penalty_last_n = (int)i64;

    /* ABSENT MEANS llama.cpp's FOUR, not none. A client that turns DRY on and names no breakers
     * gets what llama-server gives it -- newline, colon, quote and asterisk end a repeat -- and a
     * list that is sent replaces them outright, an empty one included. The default costs nothing
     * with DRY off: the sampler resolves breakers only for a request that runs DRY. */
    auto dsb = b.find("dry_sequence_breakers");
    if (dsb == b.end() || dsb->is_null()) {
        if (r->sp.dry_seq_breakers.empty() && !lim.default_dry_breakers_set)
            r->sp.dry_seq_breakers = { "\n", ":", "\"", "*" };
    } else {
        r->sp.dry_seq_breakers.clear();
        if (!dsb->is_array()) {
            *err = err_bad("dry_sequence_breakers", "expected an array of strings");
            return RAD_E_INVAL;
        }
        for (const auto& s : *dsb) {
            if (!s.is_string()) {
                *err = err_bad("dry_sequence_breakers", "expected an array of strings");
                return RAD_E_INVAL;
            }
            r->sp.dry_seq_breakers.push_back(s.get<std::string>());
        }
    }

    /* An unset seed must not mean seed 0, or every request in a deployment that never sets one
     * would draw the identical sample. Absent means "pick one", which is what OpenAI's contract
     * says and what makes `seed` meaningful when it IS set. */
    auto sit = b.find("seed");
    if (sit != b.end() && !sit->is_null()) {
        if (!sit->is_number_integer()) { *err = err_bad("seed", "expected an integer"); return RAD_E_INVAL; }
        r->sp.seed = (uint64_t)sit->get<int64_t>();
    } else {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        r->sp.seed = rng();
    }

    /* n. Implemented by fan-out: n independent requests over the same prompt, which the prefix
     * cache makes nearly free after the first (spec 7.3). Their seeds are offset from the
     * request's, because n identical seeds would produce n identical completions. */
    i64 = 1;
    if (!get_int(b, "n", &i64, err)) return RAD_E_INVAL;
    if (i64 < 1 || i64 > lim.max_n) {
        char msg[128];
        snprintf(msg, sizeof msg, "must be in [1, %d]", lim.max_n);
        *err = err_bad("n", msg);
        return RAD_E_INVAL;
    }
    r->n = (int)i64;

    i64 = 0;
    if (!get_int(b, "priority", &i64, err)) return RAD_E_INVAL;
    r->priority = (int)i64;

    /* max_tokens, with max_completion_tokens as the newer OpenAI spelling of the same field. */
    int64_t mt = 0;
    if (!get_int(b, "max_tokens", &mt, err)) return RAD_E_INVAL;
    if (b.contains("max_completion_tokens") && !b.at("max_completion_tokens").is_null()) {
        if (!get_int(b, "max_completion_tokens", &mt, err)) return RAD_E_INVAL;
    }
    if (mt < 0) { *err = err_bad("max_tokens", "must be >= 1"); return RAD_E_INVAL; }
    r->max_tokens = (int32_t)(mt > 0 ? mt : lim.default_max_tokens);

    /* ignore_eos: generate to max_tokens whatever the model emits. See Request::ignore_eos. */
    if (!get_bool(b, "ignore_eos", &r->ignore_eos, err)) return RAD_E_INVAL;

    /* stop: a string or an array of strings.
     *
     * BOUNDED IN COUNT AND LENGTH, because every stop string is searched for in each stretch of
     * text a choice produces -- by the decoder and again by the server's own scan -- so what one
     * request costs per token is its number of stop strings times the longest of them. OpenAI
     * allows four. The bounds here are far past anything a client sends and exist so that no
     * request can buy that scan without limit. The deployment may move them (--max-stop-strings,
     * --max-stop-bytes). */
    const size_t max_stops = (size_t)lim.max_stops, max_stop_bytes = (size_t)lim.max_stop_bytes;
    auto st = b.find("stop");
    if (st != b.end() && !st->is_null()) {
        if (st->is_string()) r->stop.push_back(st->get<std::string>());
        else if (st->is_array()) {
            if (st->size() > max_stops) {
                *err = err_bad("stop", "at most " + std::to_string(max_stops) +
                                           " stop strings are accepted, got " +
                                           std::to_string(st->size()));
                return RAD_E_INVAL;
            }
            for (const auto& s : *st) {
                if (!s.is_string()) {
                    *err = err_bad("stop", "expected a string or an array of strings");
                    return RAD_E_INVAL;
                }
                r->stop.push_back(s.get<std::string>());
            }
        } else {
            *err = err_bad("stop", "expected a string or an array of strings");
            return RAD_E_INVAL;
        }
        for (const auto& s : r->stop) {
            if (s.size() > max_stop_bytes) {
                *err = err_bad("stop", "a stop string may be at most " +
                                           std::to_string(max_stop_bytes) + " bytes, got one of " +
                                           std::to_string(s.size()));
                return RAD_E_INVAL;
            }
        }
    }

    bl = false;
    if (!get_bool(b, "stream", &bl, err)) return RAD_E_INVAL;
    r->stream = bl;

    auto so = b.find("stream_options");
    if (so != b.end() && !so->is_null()) {
        if (!so->is_object()) { *err = err_bad("stream_options", "expected an object"); return RAD_E_INVAL; }
        bool iu = false;
        if (!get_bool(*so, "include_usage", &iu, err)) return RAD_E_INVAL;
        r->include_usage = iu;
    }

    return RAD_OK;
}

/* ================================================================== thinking
 *
 * Three spellings reach this server for one question -- does the model reason before it answers.
 * `chat_template_kwargs: {"enable_thinking": false}` is what vLLM's clients send and is the one
 * that matters here; `enable_thinking` at the top level is the shorthand several clients grew;
 * `reasoning_effort` is OpenAI's own, and "none"/"minimal" is the only value of it this server
 * can honour, because the rest ask the model to think HARDER and no chat template exposes a dial
 * for that.
 *
 * THE KWARG ALONE IS NOT ENOUGH, and this is the trap the whole function exists for. The vendored
 * llama.cpp chat layer copies chat_template_kwargs into the template context and THEN overwrites
 * the key `enable_thinking` with its own boolean field (chat-auto-parser-helpers.cpp). So a
 * caller who sends the kwarg and nothing else has it silently overwritten with the default true,
 * and sees the model think anyway -- a 200 that did not honour the request, which is the one
 * outcome this file's header forbids. Lifting the kwarg into the field is what makes it real.
 *
 * THE DEPLOYMENT'S DEFAULTS GO IN FIRST AND THE REQUEST OVERWRITES THEM, key by key -- except that
 * thinking is one setting with three spellings, and a default in one spelling must not survive a
 * request that used another. A request that says anything about thinking (either enable_thinking,
 * or a reasoning effort) drops the deployment's enable_thinking; one that names an effort drops
 * the deployment's effort too. Otherwise `--chat-template-kwargs '{"enable_thinking":false}'`
 * would turn off thinking for a caller who sent `reasoning_effort: "high"`, and
 * `--reasoning-effort none` would do the same to one who sent `enable_thinking: true`.
 */
RequestThinking request_thinking(const json& b) {
    auto kw = b.find("chat_template_kwargs");
    const bool obj = kw != b.end() && kw->is_object();
    auto top = b.find("enable_thinking");
    /* An empty reasoning_effort string reads as unset, so it names no effort either. */
    auto re = b.find("reasoning_effort");
    RequestThinking t;
    t.effort = (obj && kw->contains("reasoning_effort")) ||
               (re != b.end() && re->is_string() && !re->get<std::string>().empty());
    t.thinking = t.effort || (top != b.end() && !top->is_null()) ||
                 (obj && kw->contains("enable_thinking"));
    return t;
}

void apply_template_defaults(const json& b, const OaiLimits& lim, ChatRenderOptions* opt) {
    const RequestThinking says = request_thinking(b);
    for (const auto& [k, v] : lim.default_template_kwargs) {
        if (k == "enable_thinking") {
            if (says.thinking) continue;
            opt->enable_thinking = v == "true";
        }
        if (k == "reasoning_effort" && says.effort) continue;
        opt->template_kwargs[k] = v;
    }
    opt->reasoning_format = lim.reasoning_format;
}

static int parse_thinking(const json& b, const OaiLimits& lim, ChatRenderOptions* opt,
                          ApiError* err) {
    auto kw = b.find("chat_template_kwargs");
    const bool has_kw = kw != b.end() && !kw->is_null();
    if (has_kw && !kw->is_object()) {
        *err = err_bad("chat_template_kwargs", "expected an object");
        return RAD_E_INVAL;
    }
    const RequestThinking says = request_thinking(b);
    apply_template_defaults(b, lim, opt);

    if (has_kw) {
        for (auto it = kw->begin(); it != kw->end(); ++it) {
            /* dump(), not get<string>(): the template context wants JSON, so a string value has
             * to arrive still quoted. See ChatRenderOptions. */
            opt->template_kwargs[it.key()] = it.value().dump();
            if (it.key() == "enable_thinking") {
                if (!it.value().is_boolean()) {
                    *err = err_bad("chat_template_kwargs.enable_thinking", "expected a boolean");
                    return RAD_E_INVAL;
                }
                opt->enable_thinking = it.value().get<bool>();
            }
        }
    }

    bool think = opt->enable_thinking;
    if (!get_bool(b, "enable_thinking", &think, err)) return RAD_E_INVAL;
    opt->enable_thinking = think;

    /* `preserve_thinking` AT THE TOP LEVEL is the template variable of that name, the same
     * shorthand `enable_thinking` is: llama.cpp maps the field into the template, and clients
     * written against it (oh-my-pi, for every Qwen model) send it there as well as in
     * chat_template_kwargs. The Qwen3.6 and later templates read it to keep earlier assistant
     * turns' reasoning in the prompt; a template that does not read it ignores it, as it ignores
     * any variable it does not name. Like enable_thinking, the top-level spelling wins. */
    auto pt = b.find("preserve_thinking");
    if (pt != b.end() && !pt->is_null()) {
        if (!pt->is_boolean()) {
            *err = err_bad("preserve_thinking", "expected a boolean");
            return RAD_E_INVAL;
        }
        opt->template_kwargs["preserve_thinking"] = pt->dump();
    }

    /* REASONING EFFORT IS THE TEMPLATE'S VOCABULARY, NOT OURS, so there is no allow-list here.
     * One would be wrong in both directions: OpenAI's ladder is none/minimal/low/medium/high,
     * while a model's template may accept a different set -- Qwen3.8's takes xhigh/medium/low and
     * raise_exception's on anything else. An allow-list would admit `"high"` only for the template
     * to refuse it, and refuse `"xhigh"`, which that template would have taken.
     *
     * What this layer actually owns is the one mapping that is a SERVER semantic and not a
     * template one: "none" and "minimal" mean do not think, which is `enable_thinking = false`
     * and is honoured whatever the template's dial is called. Everything else is passed through
     * as template context, and a template that does not read it ignores it -- exactly as the
     * kwarg path already does. A template that DOES read it and refuses says so in its own words,
     * because ChatPrompt::error carries the sentence out (core/text/chat.h).
     *
     * An explicit chat_template_kwargs entry still wins: emplace does not overwrite. */
    std::string effort;
    if (!get_str(b, "reasoning_effort", &effort, err)) return RAD_E_INVAL;
    /* The deployment's default, where the request did not say. Last, so every explicit spelling
     * above it wins; emplace, so a chat_template_kwargs entry that already set the key stands.
     * "none" and "minimal" are an enable_thinking default, so any word about thinking drops them;
     * a level is an effort default, dropped by an effort. */
    if (effort.empty()) {
        const std::string& def = lim.default_reasoning_effort;
        const bool off = def == "none" || def == "minimal";
        if (off ? !says.thinking : !says.effort) effort = def;
    }
    if (!effort.empty()) {
        if (effort == "none" || effort == "minimal") opt->enable_thinking = false;
        else opt->template_kwargs.emplace("reasoning_effort", json(effort).dump());
    }
    return RAD_OK;
}

/* ================================================================== grammar / response_format */

/* `schema_json`, when non-null, means "hand me the schema instead of compiling it" -- which is
 * what the chat endpoint needs. A chat template BUILDS the grammar itself, from the schema and
 * from its own tool-call syntax together, and it takes the schema as a schema
 * (chat-auto-parser-generator.cpp keys `has_response_format` off `inputs.json_schema`). Handing
 * it a pre-compiled GBNF in the `grammar` slot instead means it sees no response format at all,
 * emits no grammar, and the compiled one is then overwritten by that empty result -- a
 * `response_format` that returns 200 and constrains nothing. /v1/completions has no template, so
 * it passes null and the schema is compiled here. */
static int parse_output_format(const json& b, const OaiDeps& d, OaiRequest* r, ApiError* err,
                               std::string* schema_json = nullptr) {
    std::string gram;
    if (!get_str(b, "grammar", &gram, err)) return RAD_E_INVAL;

    auto rf = b.find("response_format");
    const bool has_rf = rf != b.end() && !rf->is_null();
    if (!gram.empty() && has_rf) {
        *err = err_bad("response_format", "cannot be combined with an explicit `grammar`");
        return RAD_E_INVAL;
    }
    if (!gram.empty()) { r->sp.grammar = gram; return RAD_OK; }
    if (!has_rf) return RAD_OK;

    if (!rf->is_object()) { *err = err_bad("response_format", "expected an object"); return RAD_E_INVAL; }
    std::string type;
    if (!get_str(*rf, "type", &type, err)) return RAD_E_INVAL;
    if (type.empty() || type == "text") return RAD_OK;

    json schema;
    if (type == "json_object") {
        schema = json::object();
        schema["type"] = "object";
    } else if (type == "json_schema") {
        auto w = rf->find("json_schema");
        if (w == rf->end() || !w->is_object()) {
            *err = err_bad("response_format.json_schema", "expected an object");
            return RAD_E_INVAL;
        }
        auto s = w->find("schema");
        if (s == w->end() || !s->is_object()) {
            *err = err_bad("response_format.json_schema.schema", "expected a JSON Schema object");
            return RAD_E_INVAL;
        }
        schema = *s;
    } else {
        *err = err_bad("response_format.type",
                       "must be one of \"text\", \"json_object\", \"json_schema\", got \"" +
                       type + "\"");
        return RAD_E_INVAL;
    }

    if (schema_json) { *schema_json = schema.dump(); return RAD_OK; }

    if (!d.grammar) {
        *err = err_not_implemented(
            "response_format requires the JSON-schema to GBNF compiler (spec 13), which this "
            "build was wired without");
        return RAD_E_UNSUPPORTED;
    }
    std::string out, why;
    int s = d.grammar->from_json_schema(schema, out, why);
    if (s < 0) {
        *err = err_bad("response_format", "schema could not be compiled to a grammar: " + why);
        return s;
    }
    r->sp.grammar = out;
    return RAD_OK;
}

/* ================================================================== multimodal content parts */

/* The marker the chat template renders for a media part. Both sides have to agree on it; it is
 * declared here and in iface.h so that neither component owns it privately. */
static const char* const MEDIA_MARKER = RAD_MEDIA_MARKER;

/* Pull the bytes out of an OpenAI image_url. Only `data:` URLs are accepted: fetching an
 * arbitrary http(s) URL on a caller's behalf turns the inference server into an SSRF proxy with
 * network access to everything the deployment can reach, and no amount of allowlisting makes
 * that a good default. */
static int decode_media_url(const std::string& url, const std::string& field,
                            std::vector<uint8_t>* out, ApiError* err) {
    if (url.rfind("data:", 0) != 0) {
        *err = err_bad(field, "only data: URLs are accepted; the server does not fetch remote URLs "
                              "on a request's behalf");
        return RAD_E_INVAL;
    }
    size_t comma = url.find(',');
    if (comma == std::string::npos) {
        *err = err_bad(field, "malformed data: URL");
        return RAD_E_INVAL;
    }
    const std::string meta = url.substr(5, comma - 5);
    if (meta.find("base64") == std::string::npos) {
        *err = err_bad(field, "data: URL must be base64 encoded");
        return RAD_E_INVAL;
    }
    if (!base64_decode(url.substr(comma + 1), out)) {
        *err = err_bad(field, "invalid base64");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* THE DEEPEST NESTING IN A JSON TEXT, counted without parsing it: brackets outside string
 * literals, whether or not the text is valid JSON. */
static int json_text_depth(const std::string& s) {
    int d = 0, most = 0;
    bool in_str = false, esc = false;
    for (char c : s) {
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '[' || c == '{') most = std::max(most, ++d);
        else if ((c == ']' || c == '}') && d > 0) --d;
    }
    return most;
}

/* STRINGS THE CHAT TEMPLATE PARSES AS JSON are bounded like the body is (parse_json_bounded). An
 * assistant message's tool-call arguments and a tool message's content arrive as strings, one
 * level deep in the request, and the template layer parses them into trees and serialises those
 * again when it renders -- recursively, one call a level. Their nesting is checked here, before
 * that happens, so a string of brackets cannot take the worker's stack. */
static int check_template_json(const json& msg, const std::string& role, ApiError* err) {
    if (role == "assistant") {
        auto tcs = msg.find("tool_calls");
        if (tcs != msg.end() && tcs->is_array()) {
            for (const auto& tc : *tcs) {
                if (!tc.is_object()) continue;
                auto fn = tc.find("function");
                if (fn == tc.end() || !fn->is_object()) continue;
                auto args = fn->find("arguments");
                if (args != fn->end() && args->is_string() &&
                    json_text_depth(args->get_ref<const std::string&>()) > kJsonMaxDepth) {
                    *err = err_bad("messages[].tool_calls[].function.arguments",
                                   "nests more than " + std::to_string(kJsonMaxDepth) +
                                       " levels deep");
                    return RAD_E_INVAL;
                }
            }
        }
    }
    if (role == "tool") {
        auto c = msg.find("content");
        if (c != msg.end() && c->is_string() &&
            json_text_depth(c->get_ref<const std::string&>()) > kJsonMaxDepth) {
            *err = err_bad("messages[].content", "a tool result nests more than " +
                                                     std::to_string(kJsonMaxDepth) +
                                                     " levels deep");
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* Rewrite content parts in place: text stays, media becomes the marker and its bytes are handed
 * to the encoder. `media` collects (kind, bytes) in the order they appear, which is the order the
 * markers appear in the rendered prompt. */
static int lower_content_parts(json& messages, const OaiDeps& d, const OaiLimits& lim,
                               std::vector<std::pair<MediaKind, std::vector<uint8_t>>>* media,
                               ApiError* err) {
    if (!messages.is_array()) { *err = err_bad("messages", "expected an array"); return RAD_E_INVAL; }
    if (messages.empty()) { *err = err_bad("messages", "must not be empty"); return RAD_E_INVAL; }

    for (auto& msg : messages) {
        if (!msg.is_object()) { *err = err_bad("messages[]", "expected an object"); return RAD_E_INVAL; }
        std::string role;
        auto rit = msg.find("role");
        if (rit == msg.end() || !rit->is_string()) {
            *err = err_bad("messages[].role", "expected a string");
            return RAD_E_INVAL;
        }
        role = rit->get<std::string>();
        if (role != "system" && role != "developer" && role != "user" &&
            role != "assistant" && role != "tool") {
            *err = err_bad("messages[].role", "unknown role \"" + role + "\"");
            return RAD_E_INVAL;
        }
        RAD_TRY(check_template_json(msg, role, err));

        auto cit = msg.find("content");
        if (cit == msg.end() || cit->is_null()) {
            if (role == "assistant" && msg.contains("tool_calls")) continue;
            *err = err_bad("messages[].content",
                           "required for every message except an assistant message that carries "
                           "tool_calls");
            return RAD_E_INVAL;
        }
        if (cit->is_string()) continue;
        if (!cit->is_array()) {
            *err = err_bad("messages[].content", "expected a string or an array of content parts");
            return RAD_E_INVAL;
        }

        for (auto& part : *cit) {
            std::string type;
            auto tit = part.find("type");
            if (tit == part.end() || !tit->is_string()) {
                *err = err_bad("messages[].content[].type", "expected a string");
                return RAD_E_INVAL;
            }
            type = tit->get<std::string>();

            if (type == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    *err = err_bad("messages[].content[].text", "expected a string");
                    return RAD_E_INVAL;
                }
                continue;
            }
            /* IMAGE AND VIDEO PARTS, in the two spellings clients send: OpenAI's
             * {"type":"image_url","image_url":{"url":...}} (and the bare-string form), the
             * Responses API's "input_image", and the video equivalents vLLM's clients use --
             * {"type":"video_url","video_url":{"url":...}} and "input_video". */
            const bool is_image = type == "image_url" || type == "input_image";
            const bool is_video = type == "video_url" || type == "input_video";
            if (is_image || is_video) {
                const MediaKind kind = is_image ? MediaKind::Image : MediaKind::Video;
                const char* what = is_image ? "image" : "video";
                const bool allowed = is_image ? lim.allow_image : lim.allow_video;
                if (!allowed || !d.mm || !d.mm->accepts(kind)) {
                    *err = err_not_implemented(
                        std::string(what) + " input requires a vision encoder (spec 11); this "
                        "model was loaded without one");
                    return RAD_E_UNSUPPORTED;
                }
                const std::string key = is_image ? "image_url" : "video_url";
                const std::string field = "messages[].content[]." + key;
                std::string url;
                auto iu = part.find(key);
                if (iu == part.end() && type == "input_image") iu = part.find("image_url");
                if (iu == part.end() && type == "input_video") iu = part.find("video_url");
                if (iu != part.end() && iu->is_object() && iu->contains("url") &&
                    iu->at("url").is_string())
                    url = iu->at("url").get<std::string>();
                else if (iu != part.end() && iu->is_string())
                    url = iu->get<std::string>();
                else {
                    *err = err_bad(field, "expected {\"url\": ...}");
                    return RAD_E_INVAL;
                }
                std::vector<uint8_t> bytes;
                RAD_TRY(decode_media_url(url, field + ".url", &bytes, err));
                media->emplace_back(kind, std::move(bytes));
                /* A MEDIA MARKER, NOT TEXT. The chat layer flattens a message's parts into one
                 * string and joins TEXT parts with a newline; it joins a marker to its neighbours
                 * with nothing, which is what the model's own template does with an image part
                 * (<|vision_end|> runs straight into the text after it). Joined as text, every
                 * picture would cost one '\n' token the model's own prompt does not have. */
                part.erase(key);
                part["type"] = "media_marker";
                part["text"] = MEDIA_MARKER;
                continue;
            }
            if (type == "input_audio") {
                if (!lim.allow_audio || !d.mm || !d.mm->accepts(MediaKind::Audio)) {
                    *err = err_not_implemented(
                        "audio input requires an audio encoder (spec 11); this model was loaded "
                        "without one");
                    return RAD_E_UNSUPPORTED;
                }
                auto ia = part.find("input_audio");
                if (ia == part.end() || !ia->is_object() || !ia->contains("data") ||
                    !ia->at("data").is_string()) {
                    *err = err_bad("messages[].content[].input_audio",
                                   "expected {\"data\": ..., \"format\": ...}");
                    return RAD_E_INVAL;
                }
                auto fit = ia->find("format");
                const std::string fmt =
                    fit != ia->end() && fit->is_string() ? fit->get<std::string>() : std::string();
                if (fmt != "wav" && fmt != "mp3") {
                    *err = err_bad("messages[].content[].input_audio.format",
                                   "must be \"wav\" or \"mp3\"");
                    return RAD_E_INVAL;
                }
                std::vector<uint8_t> bytes;
                if (!base64_decode(ia->at("data").get<std::string>(), &bytes)) {
                    *err = err_bad("messages[].content[].input_audio.data", "invalid base64");
                    return RAD_E_INVAL;
                }
                media->emplace_back(MediaKind::Audio, std::move(bytes));
                part.erase("input_audio");
                part["type"] = "media_marker";
                part["text"] = MEDIA_MARKER;
                continue;
            }
            *err = err_bad("messages[].content[].type",
                           "unsupported content part type \"" + type + "\"");
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* Tokenise a rendered prompt that contains media markers, replacing each marker with the token
 * run its media part became: the processor's placeholders -- the model's own pad ids, which a
 * hashed n-gram embedding reads like any other -- with whatever markers and timestamps the model
 * wants around them. The encoder's rows replace the placeholders' embeddings later (spec §11);
 * here only the ids and their offsets are fixed, and with them the prompt's rotary layout.
 *
 * Each media part is processed HERE, on the request's own worker, because decoding a video and
 * resizing its frames is the most expensive host work a request can bring: it must not reach the
 * scheduler's thread, where every other request's step would wait on it. */
static int tokenize_with_media(const std::string& rendered, const OaiDeps& d,
                               std::vector<std::pair<MediaKind, std::vector<uint8_t>>>& media,
                               PromptInput* out, ApiError* err) {
    out->text = rendered;
    auto pm = std::make_shared<mm::PromptMedia>();
    const std::string marker = MEDIA_MARKER;
    size_t pos = 0, mi = 0;
    for (;;) {
        size_t at = rendered.find(marker, pos);
        std::string seg = rendered.substr(pos, at == std::string::npos ? std::string::npos : at - pos);
        if (!seg.empty()) {
            /* add_special is false for every segment, the first included: the template renders
             * whatever special tokens the model wants, and an extra BOS would shift every
             * position after it -- the same reason the text-only path gives. */
            auto ids = d.tok->encode(seg, false, /*parse_special=*/true);
            out->tokens.insert(out->tokens.end(), ids.begin(), ids.end());
        }
        if (at == std::string::npos) break;
        if (mi >= media.size()) {
            *err = err_bad("messages",
                           "the chat template rendered more media markers than the request had "
                           "media parts");
            return RAD_E_INVAL;
        }
        std::shared_ptr<mm::Item> item;
        std::string why;
        const int s = d.mm->prepare(media[mi].first, media[mi].second, &item, &why);
        if (s < 0 || !item) {
            *err = err_bad("messages[].content[]", "the media could not be used: " +
                                                       (why.empty() ? std::string(rad_strerror(s))
                                                                    : why));
            return s < 0 ? s : RAD_E_INVAL;
        }
        media[mi].second.clear();
        media[mi].second.shrink_to_fit();
        mm::Placed pl;
        pl.item = item;
        pl.tok = (int32_t)out->tokens.size();
        out->tokens.insert(out->tokens.end(), item->expand.begin(), item->expand.end());
        pm->items.push_back(std::move(pl));
        ++mi;
        pos = at + marker.size();
    }
    if (mi != media.size()) {
        *err = err_bad("messages",
                       "the chat template dropped a media part: the request carried more media "
                       "than the rendered prompt has markers for");
        return RAD_E_INVAL;
    }
    pm->layout();
    out->media = std::move(pm);
    return RAD_OK;
}

/* ================================================================== the context bound
 *
 * THE CONTEXT CLAMPS max_tokens RATHER THAN REFUSING THE REQUEST.
 *
 * `max_tokens` is an upper bound -- "stop by here at the latest" -- and the context is a second
 * upper bound that holds whether or not the request acknowledges it: a sequence cannot run past
 * it however many tokens were asked for. So refusing does not produce more tokens than clamping
 * does, it produces none, and the field that caused the refusal could not have been honoured at
 * any value the caller might have sent instead.
 *
 * That matters because a fixed `max_tokens` is what clients actually send. A caller whose default
 * is 128k gets a 400 on every prompt long enough to matter -- exactly the requests a long-context
 * server exists for -- and the only fix available to them is to compute the server's remaining
 * window themselves, which they cannot do without knowing the tokeniser.
 *
 * A PROMPT THAT DOES NOT FIT ON ITS OWN IS STILL AN ERROR. There is no window left to generate
 * into, nothing about the request is servable, and the caller has to shorten the prompt -- so that
 * case keeps its refusal and names how much has to go.
 *
 * This is the one deliberate exception to the rule the key vetting above enforces, that a 200
 * means every field of the request was honoured. It is an exception because those keys are
 * SETTINGS -- dropping one serves a different model than the caller asked for -- and this one is a
 * BOUND, whose purpose is to stop generation early and which the context was going to stop earlier
 * anyway. What the caller gets back still says so: the completion ends with finish_reason
 * "length" and `usage` reports the tokens it actually produced.
 */
/* THE DEPLOYMENT'S CAP IS A THIRD BOUND OF THE SAME KIND, clamped for the same reason: a request
 * asking for more than the operator allows is still a request worth serving up to the cap. And a
 * max_tokens of 0 is the `--default-max-tokens auto` default, which is "what the context leaves" --
 * resolved here, against this prompt, so that the scheduler always sees a finite bound. */
static int clamp_max_tokens(size_t n_prompt, const OaiLimits& lim, const char* field,
                            OaiRequest* out, ApiError* err) {
    if (lim.max_tokens_cap > 0 &&
        (out->max_tokens <= 0 || (int64_t)out->max_tokens > lim.max_tokens_cap))
        out->max_tokens = (int32_t)lim.max_tokens_cap;

    if (lim.max_ctx <= 0) return RAD_OK;          /* no bound declared, nothing to clamp against */

    const int64_t room = lim.max_ctx - (int64_t)n_prompt;
    if (room <= 0) {
        char msg[256];
        snprintf(msg, sizeof msg,
                 "%zu prompt tokens does not fit the %lld token context, which leaves nothing to "
                 "generate into; shorten the prompt by at least %lld tokens",
                 n_prompt, (long long)lim.max_ctx, (long long)(1 - room));
        *err = err_bad(field, msg);
        err->code = "context_length_exceeded";
        return RAD_E_INVAL;
    }

    if (out->max_tokens <= 0) {
        out->max_tokens = (int32_t)std::min<int64_t>(room, INT32_MAX);
    } else if ((int64_t)out->max_tokens > room) {
        /* Debug rather than Warn: on a long-context deployment whose clients send a constant
         * max_tokens this fires on every request that is working correctly, and a line printed
         * per request is how a server's log stops being read at all. */
        RAD_DEBUG("max_tokens %d does not fit beside %zu prompt tokens in a %lld token context; "
                  "serving %lld", (int)out->max_tokens, n_prompt, (long long)lim.max_ctx,
                  (long long)room);
        out->max_tokens = (int32_t)room;
    }
    return RAD_OK;
}

/* ================================================================== chat */

/* THE REASONING BLOCK BY TOKEN ID, and the budget on it.
 *
 * The render names the block's markers as text; the engine counts and closes it by token, so
 * the markers are encoded here, once per request. A marker that is not one token leaves the span
 * unknown: the reply's reasoning is still split out, but it is not counted and cannot be closed
 * by forcing a token. The opener matters only to a reply that must open its own block.
 *
 * `thinking_token_budget` is vLLM's name for the field and `thinking_budget` Qwen's API's; both
 * are read, as max_tokens and max_completion_tokens are. A budget the span cannot hold is a 400
 * naming why -- accepting it and generating unbounded reasoning is the outcome this file's
 * header forbids. The deployment's default applies only where it can be held: a reply whose
 * block cannot be counted gets none, as it would on a model with no reasoning block at all. */
static int parse_thinking_budget(const json& b, const OaiDeps& d, const OaiLimits& lim,
                                 const ChatRender& render, SamplingParams* sp, ApiError* err) {
    ReasoningSpan span;
    size_t n_close = 0, n_open = 0;
    if (!render.think_close.empty() && d.tok) {
        const std::vector<int32_t> c = d.tok->encode(render.think_close, false, true);
        std::vector<int32_t> o;
        if (!render.think_open.empty()) o = d.tok->encode(render.think_open, false, true);
        n_close = c.size();
        n_open  = o.size();
        if (c.size() == 1 && (render.think_starts_open || o.size() == 1)) {
            span.close       = c[0];
            span.open        = o.size() == 1 ? o[0] : -1;
            span.starts_open = render.think_starts_open;
        }
    }
    sp->reasoning = span;

    int64_t budget = -1;
    const char* named = nullptr;
    for (const char* k : { "thinking_token_budget", "thinking_budget" }) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        int64_t v = 0;
        if (!get_int(b, k, &v, err)) return RAD_E_INVAL;
        if (v < 0) { *err = err_bad(k, "must be >= 0"); return RAD_E_INVAL; }
        if (named && v != budget) {
            *err = err_bad(k, std::string("disagrees with ") + named + ", which is the same field");
            return RAD_E_INVAL;
        }
        budget = v;
        named = k;
    }
    if (!named) {
        if (span.known()) sp->thinking_budget = (int32_t)lim.default_thinking_budget;
        return RAD_OK;
    }
    if (!span.known()) {
        std::string why;
        if (render.think_close.empty())
            why = "this model's chat template has no reasoning block to hold a budget on";
        else if (n_close != 1)
            why = "this model's reasoning block ends in `" + render.think_close + "`, which is " +
                  std::to_string(n_close) + " tokens, and a budget is held by forcing that end "
                  "as one token";
        else
            why = "this reply has to open its own reasoning block, and the opener `" +
                  render.think_open + "` is " + std::to_string(n_open) + " tokens, so the block "
                  "cannot be found by token";
        *err = err_unsupported(named, why);
        return RAD_E_UNSUPPORTED;
    }
    sp->thinking_budget = (int32_t)budget;
    return RAD_OK;
}

int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Chat, err));

    out->endpoint = Endpoint::Chat;
    out->id = random_id("chatcmpl-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    RAD_TRY(parse_sampling(b, lim, out, err));
    std::string schema_json;
    RAD_TRY(parse_output_format(b, d, out, err, &schema_json));

    /* logprobs. The wire format is implemented; whether the sampler produces the numbers is a
     * property of the build, and if it does not we refuse rather than return nulls that look
     * like "the model had no opinion". */
    bool want = false;
    if (!get_bool(b, "logprobs", &want, err)) return RAD_E_INVAL;
    int64_t top = 0;
    if (!get_int(b, "top_logprobs", &top, err)) return RAD_E_INVAL;
    if (!want && top > 0) {
        *err = err_bad("top_logprobs", "requires logprobs to be true");
        return RAD_E_INVAL;
    }
    if (top < 0 || top > 20) { *err = err_bad("top_logprobs", "must be in [0, 20]"); return RAD_E_INVAL; }
    if (want && !lim.supports_logprobs) {
        *err = err_unsupported("logprobs",
                               "this build's sampler does not return per-token probabilities");
        return RAD_E_UNSUPPORTED;
    }
    out->want_logprobs = want;
    out->n_logprobs = (int)top;

    if (!d.chat) {
        *err = err_not_implemented(
            "/v1/chat/completions requires a chat template (spec 12); this model was loaded "
            "without one. Use /v1/completions");
        return RAD_E_UNSUPPORTED;
    }

    auto mit = b.find("messages");
    if (mit == b.end()) { *err = err_bad("messages", "is required"); return RAD_E_INVAL; }
    json messages = *mit;
    if (messages.is_array()) {
        out->n_messages = (int)messages.size();
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (!it->is_object() || !it->contains("role") || it->at("role") != "user") continue;
            const auto cit = it->find("content");
            if (cit == it->end()) break;
            /* At most 160 bytes are kept, so no more than that is copied out of a message that
             * may be a whole document, and the cut lands on a UTF-8 lead byte: a log line split
             * inside a character is not valid text. */
            constexpr size_t kKeep = 160;
            std::string t;
            auto take = [&t](const std::string& s) {
                if (t.size() <= kKeep) t.append(s, 0, kKeep + 1 - t.size());
            };
            if (cit->is_string()) take(cit->get_ref<const std::string&>());
            else if (cit->is_array())
                for (const auto& part : *cit)
                    if (part.is_object() && part.value("type", "") == "text")
                        take(part.value("text", ""));
            for (char& c : t) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            if (t.size() > kKeep) {
                size_t cut = kKeep - 3;
                while (cut > 0 && ((unsigned char)t[cut] & 0xC0) == 0x80) --cut;
                t.resize(cut);
                t += "...";
            }
            out->last_user = std::move(t);
            break;
        }
    }

    /* tools and tool_choice. */
    json tools = json::array();
    auto tit = b.find("tools");
    if (tit != b.end() && !tit->is_null()) {
        if (!tit->is_array()) { *err = err_bad("tools", "expected an array"); return RAD_E_INVAL; }
        for (const auto& t : *tit) {
            if (!t.is_object() || !t.contains("type") || t.at("type") != "function" ||
                !t.contains("function") || !t.at("function").is_object() ||
                !t.at("function").contains("name") ||
                !t.at("function").at("name").is_string()) {
                *err = err_bad("tools[]",
                               "expected {\"type\":\"function\",\"function\":{\"name\":...}}");
                return RAD_E_INVAL;
            }
        }
        tools = *tit;
        out->n_tools = (int)tools.size();
    }

    /* FORCING A CALL IS REFUSED, and not because the grammar is out of reach -- the template
     * returns one. It is that the grammar it returns for a forced choice does not force anything:
     * it admits leading prose (the template invites "optional reasoning BEFORE the function
     * call") and then an unbounded run of calls with no terminating state, so the reply is prose
     * followed by invented calls and `finish_reason: "length"`. Serving that would be worse than
     * refusing it: the caller asked for exactly one call and would get a request that never stops.
     * `parallel_tool_calls: false` IS honoured -- that one narrows the grammar's repetition rule,
     * which is a change the generator actually makes. */
    std::string tool_choice = "auto", named;
    auto tcit = b.find("tool_choice");
    if (tcit != b.end() && !tcit->is_null()) {
        if (tcit->is_string()) {
            tool_choice = tcit->get<std::string>();
        } else if (tcit->is_object()) {
            auto f = tcit->find("function");
            if (f == tcit->end() || !f->is_object() || !f->contains("name") ||
                !f->at("name").is_string()) {
                *err = err_bad("tool_choice",
                               "an object tool_choice must be {\"type\":\"function\","
                               "\"function\":{\"name\":...}}");
                return RAD_E_INVAL;
            }
            named = f->at("name").get<std::string>();
            tool_choice = "required";
        } else {
            *err = err_bad("tool_choice", "expected a string or an object");
            return RAD_E_INVAL;
        }
    }
    if (tool_choice == "none") {
        tools = json::array();
    } else if (tool_choice == "required") {
        *err = err_unsupported(
            "tool_choice",
            named.empty()
                ? "\"auto\" and \"none\" are implemented. \"required\" is not: the grammar this "
                  "model's template produces for it admits prose before the call and an unbounded "
                  "run of calls after it, so it does not force one call -- it produces a reply "
                  "that runs to max_tokens. Send \"auto\" and check the response for a call."
                : "a named function is not implemented, for the same reason \"required\" is not: "
                  "the template's forced-call grammar does not terminate. Send \"auto\" with only "
                  "that function in `tools`.");
        return RAD_E_UNSUPPORTED;
    } else if (tool_choice != "auto") {
        *err = err_bad("tool_choice", "must be \"none\", \"auto\", \"required\", or an object");
        return RAD_E_INVAL;
    }

    /* Honoured by the grammar rather than by a prompt instruction: the auto-parser emits a
     * repeatable call section only when this is true (chat-auto-parser-generator.cpp), so false
     * genuinely admits one call and not several. */
    bool parallel = true;
    if (!get_bool(b, "parallel_tool_calls", &parallel, err)) return RAD_E_INVAL;

    if (!tools.empty() && !d.chat->supports_tools()) {
        *err = err_unsupported("tools",
                               "this model's chat template does not declare tool support");
        return RAD_E_UNSUPPORTED;
    }
    if (!tools.empty()) {
        const std::string why = d.chat->tool_refusal();
        if (!why.empty()) {
            *err = err_unsupported("tools", "this model's tool calls cannot be read back: " + why);
            return RAD_E_UNSUPPORTED;
        }
    }
    out->tools_present = !tools.empty();

    ChatRenderOptions opt;
    opt.tool_choice = tool_choice;
    opt.parallel_tool_calls = parallel;
    if (!get_bool(b, "add_generation_prompt", &opt.add_generation_prompt, err)) return RAD_E_INVAL;

    /* The template's grammar and the caller's are ONE grammar, not two, and the template is the
     * one that knows how to combine them -- a tool-calling format has to leave room for the call
     * syntax around whatever the caller constrained. So both halves go IN and the merged result
     * comes back out: a raw GBNF as `grammar`, a `response_format` as the schema it started as. */
    opt.grammar     = out->sp.grammar;
    opt.json_schema = schema_json;

    RAD_TRY(parse_thinking(b, lim, &opt, err));

    std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
    RAD_TRY(lower_content_parts(messages, d, lim, &media, err));

    ChatRender render;
    std::string why;
    const int rs = d.chat->apply(messages, tools, opt, &render, &why);
    if (rs < 0) {
        *err = rs == RAD_E_UNSUPPORTED
                   ? err_unsupported("messages", why.empty() ? "the chat template refused this "
                                                               "request" : why)
                   : err_bad("messages", why.empty() ? "the chat template could not render this "
                                                       "request" : why);
        return rs;
    }
    if (render.prompt.empty()) {
        *err = err_bad("messages", "the chat template rendered an empty prompt");
        return RAD_E_INVAL;
    }

    /* Everything the render decided that the rest of the request has to carry. The grammar can
     * come back non-empty when none went in -- that is a tool-calling template constraining its
     * own call syntax -- and it arrives LAZY, so the reply is free prose until the model starts
     * writing a call. Dropping `grammar_lazy` here would constrain the whole reply to be a tool
     * call, which is a model that can no longer answer in words. */
    /* A REQUEST THAT ASKED TO BE CONSTRAINED AND CAME BACK UNCONSTRAINED IS A REFUSAL, not a 200.
     * `response_format` and `grammar` are promises about the shape of the answer; a template that
     * returns no grammar for one has not kept it, and returning the answer anyway would leave the
     * caller parsing free prose against a schema they were told was enforced. */
    if (render.grammar.empty() && !(opt.grammar.empty() && opt.json_schema.empty())) {
        *err = err_unsupported(opt.json_schema.empty() ? "grammar" : "response_format",
                               "this model's chat template produced no grammar for it, so the "
                               "constraint could not be enforced");
        return RAD_E_UNSUPPORTED;
    }

    out->sp.grammar          = render.grammar;
    out->sp.grammar_lazy     = render.grammar_lazy;
    out->sp.grammar_triggers = render.grammar_triggers;
    out->sp.grammar_prefix   = render.generation_prompt;
    out->reply_parser        = render.parser;
    out->preserved_tokens    = render.preserved_tokens;
    out->parse_tools         = out->tools_present && render.parser != nullptr;
    RAD_TRY(parse_thinking_budget(b, d, lim, render, &out->sp, err));

    /* The format's own end-of-turn markers join the caller's stop strings rather than living in a
     * second list: both end the generation and both report `finish_reason: "stop"` (sink.cpp), so
     * a separate list would be a distinction with no observable difference. */
    for (const auto& s : render.additional_stops)
        if (std::find(out->stop.begin(), out->stop.end(), s) == out->stop.end())
            out->stop.push_back(s);

    /* A reply that starts inside the reasoning block: its stop strings are the answer's, so they
     * apply only after the block's closer (Detokenizer::defer_stops_until). The chat format says
     * whether the rendered prompt leaves the block open and what closes it. */
    if (render.think_starts_open && !render.think_close.empty())
        out->stops_after = render.think_close;

    /* Checked HERE, where the answer is still a 400. The grammar being validated may be the
     * caller's, or one the template produced for its own tool-call syntax; either way the first
     * moment it can fail is when the automaton is built, and building it on the step path would
     * mean failing a request that is already streaming. */
    if (!out->sp.grammar.empty() && d.grammar) {
        std::string gerr;
        if (d.grammar->compile(out->sp.grammar, &out->sp.grammar_program, gerr) < 0) {
            *err = err_bad(out->tools_present ? "tools" : "grammar",
                           "the grammar this request implies cannot be run: " + gerr);
            return RAD_E_INVAL;
        }
    }

    const std::string& rendered = render.prompt;

    PromptInput p;
    if (media.empty()) {
        /* add_special is false: a chat template renders whatever special tokens the model wants,
         * including its BOS. Adding another is a silent off-by-one in every position. */
        p.tokens = d.tok->encode(rendered, false, /*parse_special=*/true);
        p.text = rendered;
    } else {
        RAD_TRY(tokenize_with_media(rendered, d, media, &p, err));
    }
    if (p.tokens.empty()) {
        *err = err_bad("messages", "the rendered prompt tokenised to nothing");
        return RAD_E_INVAL;
    }
    RAD_TRY(clamp_max_tokens(p.tokens.size(), lim, "messages", out, err));
    out->prompts.push_back(std::move(p));
    return RAD_OK;
}

/* ================================================================== completions */

int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Completion, err));

    out->endpoint = Endpoint::Completion;
    out->id = random_id("cmpl-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    RAD_TRY(parse_sampling(b, lim, out, err));
    RAD_TRY(parse_output_format(b, d, out, err));

    /* Checked HERE for the same reason as on the chat path: a grammar the machine refuses is a
     * 400 while the request can still be refused, rather than a compile on the scheduler thread
     * that cancels a request the client was told had been accepted. */
    if (!out->sp.grammar.empty() && d.grammar) {
        std::string gerr;
        if (d.grammar->compile(out->sp.grammar, &out->sp.grammar_program, gerr) < 0) {
            *err = err_bad(b.contains("grammar") ? "grammar" : "response_format",
                           "the grammar this request implies cannot be run: " + gerr);
            return RAD_E_INVAL;
        }
    }

    if (!get_bool(b, "echo", &out->echo, err)) return RAD_E_INVAL;

    /* /v1/completions `logprobs` is an integer count, not a boolean. */
    auto lit = b.find("logprobs");
    if (lit != b.end() && !lit->is_null()) {
        if (!lit->is_number_integer()) {
            *err = err_bad("logprobs", "expected an integer (the number of alternatives)");
            return RAD_E_INVAL;
        }
        int64_t nl = lit->get<int64_t>();
        if (nl < 0 || nl > 5) { *err = err_bad("logprobs", "must be in [0, 5]"); return RAD_E_INVAL; }
        if (nl > 0 && !lim.supports_logprobs) {
            *err = err_unsupported("logprobs",
                                   "this build's sampler does not return per-token probabilities");
            return RAD_E_UNSUPPORTED;
        }
        out->want_logprobs = nl > 0;
        out->n_logprobs = (int)nl;
    }

    auto pit = b.find("prompt");
    if (pit == b.end() || pit->is_null()) {
        *err = err_bad("prompt", "is required");
        return RAD_E_INVAL;
    }

    /* A raw prompt is the client's whole input: no template wraps it and no role is the server's
     * to protect, so its control tokens are parsed, as vLLM and llama.cpp parse them. A client
     * that renders its own chat template sends "<|im_start|>" meaning the token; read as text it
     * reaches the model as six pieces of punctuation. Token ids could name any control token
     * already. */
    auto push_text = [&](const std::string& s) {
        PromptInput p;
        p.text = s;
        p.tokens = d.tok->encode(s, true, /*parse_special=*/true);
        out->prompts.push_back(std::move(p));
    };
    auto push_ids = [&](const json& arr, ApiError* e) -> bool {
        PromptInput p;
        for (const auto& v : arr) {
            if (!v.is_number_integer()) {
                *e = err_bad("prompt", "token arrays must contain integers only");
                return false;
            }
            int32_t id = 0;
            if (!token_id(v, &id)) {
                *e = err_bad("prompt", "a token id must be in [0, 2147483647], got " + v.dump());
                return false;
            }
            p.tokens.push_back(id);
        }
        p.text = d.tok->decode(p.tokens);
        out->prompts.push_back(std::move(p));
        return true;
    };

    if (pit->is_string()) {
        push_text(pit->get<std::string>());
    } else if (pit->is_array()) {
        if (pit->empty()) { *err = err_bad("prompt", "must not be empty"); return RAD_E_INVAL; }
        if (pit->front().is_string()) {
            for (const auto& s : *pit) {
                if (!s.is_string()) {
                    *err = err_bad("prompt", "a prompt array must be all strings or all tokens");
                    return RAD_E_INVAL;
                }
                push_text(s.get<std::string>());
            }
        } else if (pit->front().is_number_integer()) {
            if (!push_ids(*pit, err)) return RAD_E_INVAL;
        } else if (pit->front().is_array()) {
            for (const auto& a : *pit) {
                if (!a.is_array()) {
                    *err = err_bad("prompt", "a batch of token arrays must be all arrays");
                    return RAD_E_INVAL;
                }
                if (!push_ids(a, err)) return RAD_E_INVAL;
            }
        } else {
            *err = err_bad("prompt", "expected a string, an array of strings, an array of token "
                                     "ids, or an array of token id arrays");
            return RAD_E_INVAL;
        }
    } else {
        *err = err_bad("prompt", "expected a string or an array");
        return RAD_E_INVAL;
    }

    /* THE LONGEST PROMPT SETS THE CLAMP. This endpoint takes a BATCH of prompts and `max_tokens`
     * is one field shared by all of them, so clamping against each in turn would leave the last
     * one's room standing as the answer for every prompt before it -- and a short prompt would be
     * cut off for a window it was never short of. */
    size_t longest = 0;
    for (const auto& p : out->prompts) {
        if (p.tokens.empty()) { *err = err_bad("prompt", "tokenised to nothing"); return RAD_E_INVAL; }
        if (p.tokens.size() > longest) longest = p.tokens.size();
    }
    RAD_TRY(clamp_max_tokens(longest, lim, "prompt", out, err));
    return RAD_OK;
}

/* ================================================================== embeddings */

int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Embedding, err));

    out->endpoint = Endpoint::Embedding;
    out->id = random_id("embd-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    std::string enc = "float";
    if (!get_str(b, "encoding_format", &enc, err)) return RAD_E_INVAL;
    if (enc != "float" && enc != "base64") {
        *err = err_bad("encoding_format", "must be \"float\" or \"base64\"");
        return RAD_E_INVAL;
    }
    out->base64_embeddings = (enc == "base64");

    if (b.contains("dimensions") && !b.at("dimensions").is_null()) {
        *err = err_unsupported("dimensions",
                               "Matryoshka truncation is not implemented; the model's embedding "
                               "dimension is fixed");
        return RAD_E_UNSUPPORTED;
    }

    auto it = b.find("input");
    if (it == b.end() || it->is_null()) { *err = err_bad("input", "is required"); return RAD_E_INVAL; }

    auto push_ids = [&](const json& arr, ApiError* e) -> bool {
        PromptInput p;
        for (const auto& v : arr) {
            if (!v.is_number_integer()) {
                *e = err_bad("input", "token arrays must contain integers only");
                return false;
            }
            int32_t id = 0;
            if (!token_id(v, &id)) {
                *e = err_bad("input", "a token id must be in [0, 2147483647], got " + v.dump());
                return false;
            }
            p.tokens.push_back(id);
        }
        out->prompts.push_back(std::move(p));
        return true;
    };

    /* Parsed like a /v1/completions prompt, and for the same reason: the input is the client's
     * own text with no template around it. */
    if (it->is_string()) {
        PromptInput p;
        p.text = it->get<std::string>();
        p.tokens = d.tok->encode(p.text, true, /*parse_special=*/true);
        out->prompts.push_back(std::move(p));
    } else if (it->is_array()) {
        if (it->empty()) { *err = err_bad("input", "must not be empty"); return RAD_E_INVAL; }
        if (it->front().is_string()) {
            for (const auto& s : *it) {
                if (!s.is_string()) { *err = err_bad("input", "expected all strings"); return RAD_E_INVAL; }
                PromptInput p;
                p.text = s.get<std::string>();
                p.tokens = d.tok->encode(p.text, true, /*parse_special=*/true);
                out->prompts.push_back(std::move(p));
            }
        } else if (it->front().is_number_integer()) {
            if (!push_ids(*it, err)) return RAD_E_INVAL;
        } else if (it->front().is_array()) {
            for (const auto& a : *it) {
                if (!a.is_array()) { *err = err_bad("input", "expected arrays"); return RAD_E_INVAL; }
                if (!push_ids(a, err)) return RAD_E_INVAL;
            }
        } else {
            *err = err_bad("input", "expected a string, an array of strings, or token ids");
            return RAD_E_INVAL;
        }
    } else {
        *err = err_bad("input", "expected a string or an array");
        return RAD_E_INVAL;
    }

    for (const auto& p : out->prompts) {
        if (p.tokens.empty()) { *err = err_bad("input", "tokenised to nothing"); return RAD_E_INVAL; }
        if (lim.max_ctx > 0 && (int64_t)p.tokens.size() > lim.max_ctx) {
            *err = err_bad("input", "exceeds the model context");
            err->code = "context_length_exceeded";
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* ================================================================== responses */

static json tool_calls_json(const std::vector<ToolCall>& tcs) {
    json a = json::array();
    for (const auto& t : tcs) {
        json f;
        f["name"] = t.name;
        f["arguments"] = t.arguments;
        json c;
        c["id"] = t.id;
        c["type"] = "function";
        c["function"] = f;
        a.push_back(std::move(c));
    }
    return a;
}

static json usage_json(const UsageOut& u) {
    json d;
    d["cached_tokens"] = u.cached;
    json o;
    o["prompt_tokens"] = u.prompt;
    o["completion_tokens"] = u.completion;
    o["total_tokens"] = u.prompt + u.completion;
    o["prompt_tokens_details"] = d;
    /* HOW MUCH OF completion_tokens THE CLIENT NEVER SEES AS CONTENT. Without it a reply cut at
     * max_tokens inside its reasoning reads as a model that answered nothing; with it the client
     * can tell the budget went to reasoning. */
    if (u.reasoning >= 0) {
        json c;
        c["reasoning_tokens"] = u.reasoning;
        o["completion_tokens_details"] = c;
    }
    return o;
}

json timings_json(const TimingsOut& t) {
    auto per = [](int64_t n, double ms) -> json {
        return n > 0 && ms > 0.0 ? json(ms / (double)n) : json(nullptr);
    };
    auto rate = [](int64_t n, double ms) -> json {
        return n > 0 && ms > 0.0 ? json((double)n * 1000.0 / ms) : json(nullptr);
    };
    const json pms = t.prompt_ms >= 0.0 ? json(t.prompt_ms) : json(nullptr);
    json o;
    o["cache_n"]                = t.cache_n;
    o["prompt_n"]               = t.prompt_n;
    o["prompt_ms"]              = pms;
    o["prompt_per_token_ms"]    = per(t.prompt_n, t.prompt_ms);
    o["prompt_per_second"]      = rate(t.prompt_n, t.prompt_ms);
    o["predicted_n"]            = t.predicted_n;
    o["predicted_ms"]           = t.predicted_ms;
    o["predicted_per_token_ms"] = per(t.decode_n, t.predicted_ms);
    o["predicted_per_second"]   = rate(t.decode_n, t.predicted_ms);
    if (t.draft_n > 0) {
        o["draft_n"]          = t.draft_n;
        o["draft_n_accepted"] = t.draft_n_accepted;
    }
    return o;
}

/* THE USAGE CHUNKS.
 *
 * `stream_options.include_usage` is parsed, validated and accepted on both endpoints, so both
 * streams have to emit it; accepting the field and then not honouring it is the outcome
 * core/server/oai.h's header forbids in its opening paragraph.
 *
 * One contract for both, OpenAI's: an empty choices array, sent after the last content chunk
 * and before [DONE]. Only the `object` differs, because it is a different endpoint's stream. */
static std::string usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t,
                               const char* object) {
    json o;
    o["id"] = r.id;
    o["object"] = object;
    o["created"] = r.created;
    o["model"] = r.model;
    o["choices"] = json::array();
    o["usage"] = usage_json(u);
    if (t.valid) o["timings"] = timings_json(t);
    return sse_event(o.dump());
}

std::string text_usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t) {
    return usage_chunk(r, u, t, "text_completion");
}

std::string chat_usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t) {
    return usage_chunk(r, u, t, "chat.completion.chunk");
}

static json chat_logprobs_json(const std::vector<LogprobEntry>& lp) {
    json content = json::array();
    for (const auto& e : lp) {
        json bytes = json::array();
        for (unsigned char c : e.text) bytes.push_back((int)c);
        json j;
        j["token"] = e.text;
        j["logprob"] = e.logprob;
        j["bytes"] = bytes;
        j["top_logprobs"] = json::array();
        content.push_back(std::move(j));
    }
    json o;
    o["content"] = content;
    return o;
}

std::string chat_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t) {
    json arr = json::array();
    for (const auto& c : choices) {
        json m;
        m["role"] = "assistant";
        if (c.msg.content.empty() && !c.msg.tool_calls.empty()) m["content"] = nullptr;
        else m["content"] = c.msg.content;
        if (!c.msg.reasoning.empty()) m["reasoning_content"] = c.msg.reasoning;
        if (!c.msg.tool_calls.empty()) m["tool_calls"] = tool_calls_json(c.msg.tool_calls);

        json ch;
        ch["index"] = c.index;
        ch["message"] = std::move(m);
        if (r.want_logprobs) ch["logprobs"] = chat_logprobs_json(c.logprobs);
        else ch["logprobs"] = nullptr;
        ch["finish_reason"] = finish_reason_oai(c.finish);
        arr.push_back(std::move(ch));
    }
    json o;
    o["id"] = r.id;
    o["object"] = "chat.completion";
    o["created"] = r.created;
    o["model"] = r.model;
    o["choices"] = std::move(arr);
    o["usage"] = usage_json(u);
    if (t.valid) o["timings"] = timings_json(t);
    return o.dump();
}

std::string text_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t) {
    json arr = json::array();
    for (const auto& c : choices) {
        json ch;
        ch["text"] = c.text;
        ch["index"] = c.index;
        if (r.want_logprobs) {
            json toks = json::array(), lps = json::array(), offs = json::array();
            int64_t off = 0;
            for (const auto& e : c.logprobs) {
                toks.push_back(e.text);
                lps.push_back(e.logprob);
                offs.push_back(off);
                off += (int64_t)e.text.size();
            }
            json lp;
            lp["tokens"] = toks;
            lp["token_logprobs"] = lps;
            lp["top_logprobs"] = json::array();
            lp["text_offset"] = offs;
            ch["logprobs"] = std::move(lp);
        } else {
            ch["logprobs"] = nullptr;
        }
        ch["finish_reason"] = finish_reason_oai(c.finish);
        arr.push_back(std::move(ch));
    }
    json o;
    o["id"] = r.id;
    o["object"] = "text_completion";
    o["created"] = r.created;
    o["model"] = r.model;
    o["choices"] = std::move(arr);
    o["usage"] = usage_json(u);
    if (t.valid) o["timings"] = timings_json(t);
    return o.dump();
}

std::string embeddings_body(const OaiRequest& r, const std::vector<std::vector<float>>& vecs,
                            const UsageOut& u) {
    json data = json::array();
    for (size_t i = 0; i < vecs.size(); ++i) {
        json e;
        e["object"] = "embedding";
        if (r.base64_embeddings)
            e["embedding"] = base64_encode(vecs[i].data(), vecs[i].size() * sizeof(float));
        else
            e["embedding"] = vecs[i];
        e["index"] = (int64_t)i;
        data.push_back(std::move(e));
    }
    json usage;
    usage["prompt_tokens"] = u.prompt;
    usage["total_tokens"] = u.prompt;
    json o;
    o["object"] = "list";
    o["data"] = std::move(data);
    o["model"] = r.model;
    o["usage"] = std::move(usage);
    return o.dump();
}

std::string models_body(const std::string& model_id, int64_t created) {
    json m;
    m["id"] = model_id;
    m["object"] = "model";
    m["created"] = created;
    m["owned_by"] = "radiance";
    json o;
    o["object"] = "list";
    o["data"] = json::array({m});
    return o.dump();
}

/* ================================================================== SSE */

std::string sse_event(const std::string& json_text) {
    return "data: " + json_text + "\n\n";
}

std::string sse_error(const ApiError& e) {
    /* A failure after the headers have gone out cannot change the status code, so the error has
     * to travel in the stream. Clients that ignore it still see the stream end; clients that read
     * it get the reason instead of a truncated answer. */
    return "data: " + e.dump() + "\n\n";
}

/* ================================================================== the streaming path */

ChatDeltaStream::ChatDeltaStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : reader_(parser ? parser->open() : nullptr), id_(r.id), model_(r.model),
      created_(r.created), index_(index) {}

std::string ChatDeltaStream::chunk(json delta, const char* finish_reason, const TimingsOut* t) {
    json ch;
    ch["index"] = index_;
    ch["delta"] = std::move(delta);
    if (finish_reason) ch["finish_reason"] = finish_reason;
    else ch["finish_reason"] = nullptr;
    json o;
    o["id"] = id_;
    o["object"] = "chat.completion.chunk";
    o["created"] = created_;
    o["model"] = model_;
    o["choices"] = json::array({ch});
    if (t && t->valid) o["timings"] = timings_json(*t);
    return sse_event(o.dump());
}

void ChatDeltaStream::delta_open(std::string& out) {
    /* Built from nlohmann's own serialisation of each field, on first use rather than in the
     * constructor, so a field nlohmann refuses throws where chunk() would have. */
    if (head_.empty())
        head_ = "data: {\"id\":" + json(id_).dump() +
                ",\"object\":\"chat.completion.chunk\",\"created\":" + json(created_).dump() +
                ",\"model\":" + json(model_).dump() + ",\"choices\":[{\"index\":" +
                json(index_).dump() + ",\"delta\":";
    out += head_;
}

void ChatDeltaStream::delta_close(std::string& out) const {
    out += ",\"finish_reason\":null}]}\n\n";
}

/* Text that is not well-formed UTF-8 goes to nlohmann, which refuses it with the exception the
 * tree in chunk() raises for it. */
void ChatDeltaStream::quote(std::string& out, std::string_view s) {
    if (!json_quote_append(out, s)) out += json(std::string(s)).dump();
}

std::string ChatDeltaStream::first_chunk() {
    json d;
    d["role"] = "assistant";
    d["content"] = "";
    return chunk(std::move(d), nullptr);
}

/* One SSE chunk per increment, in the order the reader completed them. A call is announced with
 * its id, name and empty arguments before any of its arguments flow -- an entry with no name is
 * an index the client can never resolve -- and the id is minted here, because the parser reads
 * the model's text and the model writes none. */
std::string ChatDeltaStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            if (d.text.empty()) continue;
            last_.reasoning += d.text;
            delta_open(out);
            out += "{\"reasoning_content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::Content:
            if (d.text.empty()) continue;
            last_.content += d.text;
            delta_open(out);
            out += "{\"content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::CallBegin: {
            ToolCall tc;
            tc.id   = random_id("call_");
            tc.name = d.text;
            delta_open(out);
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"id\":";
            quote(out, tc.id);
            out += ",\"type\":\"function\",\"function\":{\"name\":";
            quote(out, tc.name);
            out += ",\"arguments\":\"\"}}]}";
            last_.tool_calls.push_back(std::move(tc));
            break;
        }
        case ReplyDelta::CallArgs: {
            if (d.text.empty() || d.call >= last_.tool_calls.size()) continue;
            last_.tool_calls[d.call].arguments += d.text;
            delta_open(out);
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"function\":{\"arguments\":";
            quote(out, d.text);
            out += "}}]}";
            break;
        }
        case ReplyDelta::CallEnd:
            continue;
        }
        delta_close(out);
    }
    return out;
}

std::string ChatDeltaStream::push(std::string_view appended, bool is_final) {
    if (finished_) return std::string();
    std::string out;
    if (!reader_) {
        /* Nothing to parse: the text is the answer, as it arrives. */
        if (!appended.empty()) {
            last_.content.append(appended);
            delta_open(out);
            out += "{\"content\":";
            quote(out, appended);
            out += '}';
            delta_close(out);
        }
        finished_ = is_final;
        return out;
    }
    ev_.clear();
    if (!appended.empty()) reader_->feed(appended, &ev_);
    if (is_final) reader_->finish(&ev_);
    out = emit(ev_);
    if (is_final) {
        finished_ = true;
        /* The reader's message is the reply by definition; the deltas above add up to it, and
         * the ids are the ones this stream minted. */
        ParsedMessage m = reader_->message();
        for (size_t i = 0; i < m.tool_calls.size() && i < last_.tool_calls.size(); ++i)
            m.tool_calls[i].id = last_.tool_calls[i].id;
        last_ = std::move(m);
    }
    return out;
}

bool calls_complete(const ParsedMessage& m) {
    if (m.tool_calls.empty() || m.open_call) return false;
    for (const auto& tc : m.tool_calls)
        if (tc.name.empty() || !json_complete(tc.arguments)) return false;
    return true;
}

Finish ChatDeltaStream::adjusted_finish(Finish f) const {
    if (f != Finish::Stop && f != Finish::StopString) return f;
    /* Only claim tool_calls if every call closed and its arguments parse. A truncated arguments
     * object reported as a tool call is a tool invoked with the wrong arguments (json_partial.h).
     * The deltas of an open call have gone out already; the finish reason says not to run it. */
    return calls_complete(last_) ? Finish::ToolCalls : f;
}

std::string ChatDeltaStream::final_chunk(Finish f, const TimingsOut& t) {
    /* The generation is over, so whatever the reader was holding is resolved now. A caller that
     * already said so gets nothing twice. */
    std::string out = push(std::string_view(), true);
    out += chunk(json::object(), finish_reason_oai(adjusted_finish(f)), &t);
    return out;
}

}  /* namespace server */
}  /* namespace rad */
