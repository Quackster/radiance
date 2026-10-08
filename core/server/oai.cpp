#include "oai.h"
#include "json_partial.h"
#include "rad_internal.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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
    /* One table lookup a character: 0..63 a digit, -2 padding or whitespace (skipped wherever it
     * stands), -1 anything else. A media part is megabytes of this, and a branch ladder a
     * character decoded a 30 s clip at about 120 MB/s. */
    static const auto kVal = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) { t[(size_t)('A' + i)] = (int8_t)i; t[(size_t)('a' + i)] = (int8_t)(26 + i); }
        for (int i = 0; i < 10; ++i) t[(size_t)('0' + i)] = (int8_t)(52 + i);
        t[(size_t)'+'] = 62;
        t[(size_t)'/'] = 63;
        for (char c : { '=', '\n', '\r', ' ', '\t' }) t[(size_t)(uint8_t)c] = -2;
        return t;
    }();
    out->resize(in.size() * 3 / 4 + 3);
    uint8_t* o = out->data();
    const uint8_t* p = (const uint8_t*)in.data();
    const uint8_t* const end = p + in.size();
    uint32_t acc = 0;
    int bits = 0;
    while (p < end) {
        /* Four digits at once while they are four digits: the body of any real payload. */
        if (bits == 0 && end - p >= 4) {
            const int a = kVal[p[0]], b = kVal[p[1]], c = kVal[p[2]], d = kVal[p[3]];
            if ((a | b | c | d) >= 0) {
                const uint32_t v = ((uint32_t)a << 18) | ((uint32_t)b << 12) | ((uint32_t)c << 6) | (uint32_t)d;
                o[0] = (uint8_t)(v >> 16); o[1] = (uint8_t)(v >> 8); o[2] = (uint8_t)v;
                o += 3;
                p += 4;
                continue;
            }
        }
        const int v = kVal[*p++];
        if (v == -2) continue;
        if (v < 0) { out->clear(); return false; }   /* nothing of a refused payload is left */
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            *o++ = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    out->resize((size_t)(o - out->data()));
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
    if (!supported.empty()) e["supported"] = supported;
    if (n_prompt_tokens >= 0) e["n_prompt_tokens"] = n_prompt_tokens;
    if (n_ctx >= 0) e["n_ctx"] = n_ctx;
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
 * FOUR KINDS OF FIELD, and the lists below are the whole policy.
 *
 * `kAccepted*` is what this server reads and honours, in OpenAI's, vLLM's and llama.cpp's
 * spellings alike (`repetition_penalty` and `repeat_penalty`, `max_tokens` and `n_predict`).
 *
 * `kInert` changes nothing about the answer: routing and bookkeeping labels (`user`, `metadata`,
 * OpenRouter's `session_id`), and llama.cpp's knobs for its own speed and slot management
 * (`cache_prompt`, `id_slot`, `speculative.*`, ...) -- this server caches prefixes and speculates
 * on its own terms, and the output is the same either way. Accepted and ignored.
 *
 * `kGuarded` is what this server knows and does not implement, where honouring it WOULD change
 * the answer: mirostat, top-n-sigma, dynamic temperature, a logit bias. llama.cpp-built clients
 * send every one of them on every request at its "off" value, so each is accepted at that value
 * and refused by name at any other -- a 200 still means every field that could change the answer
 * was honoured, and a client that turned mirostat on learns that it was not.
 *
 * ANYTHING ELSE IS IGNORED, as llama.cpp and vLLM ignore it. A field this server has never heard
 * of is a client's own extension far more often than a request it has misunderstood, and refusing
 * it breaks the client for nothing (#47). The ignored names are logged at debug.
 */
namespace {

const char* kAcceptedCommon[] = {
    "model", "stream", "stream_options", "n", "seed", "stop", "max_tokens", "n_predict",
    "temperature", "top_p", "top_k", "min_p", "typical_p",
    "presence_penalty", "frequency_penalty", "repetition_penalty", "repeat_penalty",
    "repeat_last_n", "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "samplers",
    "grammar", "json_schema", "priority", "logprobs", "ignore_eos",
    "timings_per_token", "return_progress", "session_id", "n_probs",
};

const char* kAcceptedChat[] = {
    "messages", "tools", "tool_choice", "response_format", "top_logprobs",
    "parallel_tool_calls", "add_generation_prompt", "max_completion_tokens",
    "chat_template_kwargs", "reasoning_effort", "enable_thinking", "preserve_thinking",
    "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
    "reasoning_budget_tokens", "reasoning_format", "continue_final_message",
    "functions", "function_call",
};

const char* kAcceptedCompletion[] = { "prompt", "echo" };

const char* kAcceptedEmbedding[] = { "model", "input", "messages", "encoding_format",
                                      "dimensions", "image_tokens", "prompt_name" };

/* Named explicitly so that "we ignore it" is a decision in the source and not an accident of a
 * missing branch -- and so the debug line of ignored names lists only what nobody decided on. */
const char* kInert[] = {
    /* OpenAI's and OpenRouter's bookkeeping */
    "user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key",
    "prediction",
    /* llama.cpp's own speed, slot and transport knobs */
    "cache_prompt", "id_slot", "n_cache_reuse", "backend_sampling", "reasoning_control",
    "t_max_prompt_ms", "n_keep", "n_discard", "sse_ping_interval", "verbose", "response_fields",
    /* the parameters of a guarded sampler, which mean nothing while it is off */
    "mirostat_tau", "mirostat_eta", "dynatemp_exponent",
};
/* llama.cpp's speculative-decoding knobs travel as dotted keys ("speculative.n_max"). */
const char* kInertPrefixes[] = { "speculative." };

/* A field this server does not implement and the value at which it does nothing. */
struct Guarded {
    const char* key;
    bool (*neutral)(const json& v);
    const char* off;     /* how a client turns it off, for the refusal */
    const char* why;     /* why it is not served */
};

bool is_zero(const json& v)      { return v.is_null() || (v.is_number() && v.get<double>() == 0.0); }
bool non_positive(const json& v) { return v.is_null() || (v.is_number() && v.get<double>() <= 0.0); }
bool falsy(const json& v)        { return v.is_null() || (v.is_boolean() && !v.get<bool>()); }
bool empty_value(const json& v) {
    return v.is_null() || ((v.is_object() || v.is_array() || v.is_string()) && v.empty());
}
bool at_most_one(const json& v)  { return v.is_null() || (v.is_number() && v.get<double>() <= 1.0); }
bool text_only(const json& v) {
    if (v.is_null()) return true;
    if (!v.is_array()) return false;
    for (const json& m : v) if (!m.is_string() || m.get<std::string>() != "text") return false;
    return true;
}

const Guarded kGuarded[] = {
    { "logit_bias", empty_value, "an empty object",
      "the device sampler (spec 13) has no per-token bias stage" },
    { "mirostat", is_zero, "0", "mirostat is not one of the device samplers (spec 13)" },
    { "top_n_sigma", non_positive, "-1", "top-n-sigma is not one of the device samplers (spec 13)" },
    { "dynatemp_range", non_positive, "0",
      "dynamic temperature is not one of the device samplers (spec 13)" },
    { "min_keep", is_zero, "0", "the cut stages keep no minimum candidate count" },
    { "post_sampling_probs", falsy, "false",
      "logprobs are the raw distribution's; the distribution after the sampler's stages is not "
      "returned" },
    { "best_of", at_most_one, "1", "sampling several candidates and returning the best is not "
                                   "implemented" },
    { "suffix", empty_value, "an empty string", "infilling through /v1/completions is not "
                                                "implemented" },
    { "lora", empty_value, "an empty list", "LoRA is explicitly out of scope (spec, Scope)" },
    { "modalities", text_only, "[\"text\"]", "output modalities other than text are not "
                                             "implemented" },
    { "audio", empty_value, "null", "audio output is not implemented" },
    { "n_indent", is_zero, "0", "indentation-limited infill is not implemented" },
    { "t_max_predict_ms", non_positive, "0", "a generation time limit is not implemented" },
    { "return_tokens", falsy, "false", "token ids are not returned beside the text" },
    { "grammar_lazy", falsy, "false",
      "a caller's grammar applies from the first token; the lazy form is the chat format's own" },
    { "grammar_triggers", empty_value, "an empty list",
      "a caller's grammar applies from the first token; the lazy form is the chat format's own" },
    { "preserved_tokens", empty_value, "an empty list",
      "the chat format names the control tokens a reply keeps" },
};

bool in_list(const char* const* list, size_t n, const std::string& k) {
    for (size_t i = 0; i < n; ++i) if (k == list[i]) return true;
    return false;
}
#define ARRN(a) (sizeof(a) / sizeof((a)[0]))

int vet_keys(const json& body, Endpoint ep, ApiError* err, bool native = false) {
    std::string ignored;
    for (auto it = body.begin(); it != body.end(); ++it) {
        const std::string& k = it.key();
        if (in_list(kInert, ARRN(kInert), k)) continue;
        bool prefixed = false;
        for (const char* p : kInertPrefixes) prefixed = prefixed || k.rfind(p, 0) == 0;
        if (prefixed) continue;
        if (ep == Endpoint::Embedding) {
            if (in_list(kAcceptedEmbedding, ARRN(kAcceptedEmbedding), k)) continue;
        } else {
            if (in_list(kAcceptedCommon, ARRN(kAcceptedCommon), k)) continue;
            if (ep == Endpoint::Chat &&
                in_list(kAcceptedChat, ARRN(kAcceptedChat), k)) continue;
            if (ep == Endpoint::Completion &&
                in_list(kAcceptedCompletion, ARRN(kAcceptedCompletion), k)) continue;
        }
        /* llama-server's native completion returns the token ids, so the field is served there. */
        if (native && k == "return_tokens") continue;
        bool guarded = false;
        for (const Guarded& g : kGuarded) {
            if (k != g.key) continue;
            guarded = true;
            if (!g.neutral(it.value())) {
                *err = err_unsupported(k, std::string(g.why) + "; it is accepted only at " +
                                              g.off + ", where it does nothing");
                return RAD_E_UNSUPPORTED;
            }
        }
        if (!guarded) ignored += (ignored.empty() ? "" : ", ") + k;
    }
    if (!ignored.empty()) RAD_DEBUG("request fields ignored: %s", ignored.c_str());
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
/* A MEMBER OF A NESTED OBJECT -- a message, a content block, a tool -- read as a string: absent or
 * null leaves `*out` as it was, any other type is refused, named by `param`. json::value() throws
 * on a type it cannot convert, and a throw out of a handler is a 500 in the library's words. */
bool member_str(const json& o, const char* k, const char* param, std::string* out, ApiError* err) {
    auto it = o.find(k);
    if (it == o.end() || it->is_null()) return true;
    if (!it->is_string()) { *err = err_bad(param, "expected a string"); return false; }
    *out = it->get<std::string>();
    return true;
}
/* The same read where the type is checked later, by the validation the value goes through: here a
 * member that is not a string reads as `def`. */
std::string str_or(const json& o, const char* k, const std::string& def = std::string()) {
    auto it = o.find(k);
    return it != o.end() && it->is_string() ? it->get<std::string>() : def;
}

/* A conversation's label as /sessions shows it: at most 128 bytes, cut where a character starts,
 * because the label is serialised into JSON that refuses a string split inside one. */
static std::string session_label_of(const std::string& s) {
    constexpr size_t kMax = 128;
    if (s.size() <= kMax) return s;
    size_t cut = kMax;
    while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) --cut;
    return s.substr(0, cut);
}

/* ONE FIELD, SEVERAL SPELLINGS: OpenAI's, vLLM's and llama.cpp's name for the same thing. Any of
 * them may be sent; two that disagree are refused, because there is no order in which to read
 * them that the caller could have meant. */
bool get_num_any(const json& b, std::initializer_list<const char*> keys, double* out,
                 ApiError* err) {
    const char* named = nullptr;
    double v = *out;
    for (const char* k : keys) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        double x = 0;
        if (!get_num(b, k, &x, err)) return false;
        if (named && x != v) {
            *err = err_bad(k, std::string("disagrees with ") + named + ", which is the same field");
            return false;
        }
        v = x;
        named = k;
    }
    *out = v;
    return true;
}
bool get_int_any(const json& b, std::initializer_list<const char*> keys, int64_t* out,
                 bool* present, ApiError* err) {
    const char* named = nullptr;
    int64_t v = *out;
    for (const char* k : keys) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        int64_t x = 0;
        if (!get_int(b, k, &x, err)) return false;
        if (named && x != v) {
            *err = err_bad(k, std::string("disagrees with ") + named + ", which is the same field");
            return false;
        }
        v = x;
        named = k;
    }
    *out = v;
    if (present) *present = named != nullptr;
    return true;
}

}  /* namespace */

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

/* ================================================================== the sampler order
 *
 * llama.cpp's `samplers`: which stages run and in what order, as names ("top_k", "temperature",
 * with llama.cpp's aliases) or as one string of its letters ("kypmxt"). A stage the list leaves
 * out does not run, whatever its own field says -- llama.cpp's meaning, and the reason a client
 * sends the list.
 *
 * THE CHAIN HERE IS FIXED BUT FOR ONE THING (core/sample/sampler.h): penalties, DRY, top-k, then
 * top-p, min-p, typical and XTC, with temperature either before the cuts (vLLM's order, the
 * default) or after them (llama.cpp's). So a list is honoured when the stages it enables appear in
 * that relative order with temperature at either end of the cuts, and refused by name otherwise.
 * Only ENABLED stages are compared: a stage at its identity value is the same wherever it sits,
 * which is what lets llama.cpp's default list -- typ_p before top_p -- through whenever typical
 * is off. top-n-sigma is guarded at its off value (kGuarded), so it is free to appear anywhere. */
static int apply_sampler_order(const json& b, SamplingParams* sp, ApiError* err) {
    auto it = b.find("samplers");
    if (it == b.end() || it->is_null()) return RAD_OK;

    enum { PEN, DRY, TOPK, TOPP, MINP, TYP, XTC, TEMP, SIGMA, N_STAGE };
    static const char* const kName[N_STAGE] = { "penalties", "dry", "top_k", "top_p", "min_p",
                                                "typ_p", "xtc", "temperature", "top_n_sigma" };
    auto by_name = [](std::string n) -> int {
        std::replace(n.begin(), n.end(), '-', '_');
        if (n == "penalties") return PEN;
        if (n == "dry") return DRY;
        if (n == "top_k") return TOPK;
        if (n == "top_p" || n == "nucleus") return TOPP;
        if (n == "min_p") return MINP;
        if (n == "typ_p" || n == "typ" || n == "typical_p" || n == "typical") return TYP;
        if (n == "xtc") return XTC;
        if (n == "temperature" || n == "temp") return TEMP;
        if (n == "top_n_sigma") return SIGMA;
        return -1;
    };
    auto by_char = [](char c) -> int {
        switch (c) {
            case 'e': return PEN;  case 'd': return DRY;  case 'k': return TOPK;
            case 'p': return TOPP; case 'm': return MINP; case 'y': return TYP;
            case 'x': return XTC;  case 't': return TEMP; case 's': return SIGMA;
            default:  return -1;
        }
    };
    std::vector<int> list;
    if (it->is_string()) {
        for (char c : it->get<std::string>()) {
            const int st = by_char(c);
            if (st < 0) {
                *err = err_unsupported("samplers", std::string("'") + c + "' names no sampler this "
                                       "server runs (infill and adaptive_p are not served)");
                return RAD_E_UNSUPPORTED;
            }
            list.push_back(st);
        }
    } else if (it->is_array()) {
        for (const json& n : *it) {
            if (!n.is_string()) { *err = err_bad("samplers", "expected sampler names"); return RAD_E_INVAL; }
            const int st = by_name(n.get<std::string>());
            if (st < 0) {
                *err = err_unsupported("samplers", "\"" + n.get<std::string>() + "\" names no "
                                       "sampler this server runs (infill and adaptive_p are not "
                                       "served)");
                return RAD_E_UNSUPPORTED;
            }
            list.push_back(st);
        }
    } else {
        *err = err_bad("samplers", "expected an array of sampler names or a string of letters");
        return RAD_E_INVAL;
    }

    auto listed = [&](int st) { return std::find(list.begin(), list.end(), st) != list.end(); };
    /* What the list leaves out is off. Greedy stays greedy: temperature 0 is the argmax in either
     * engine whatever surrounds it. */
    if (!listed(PEN))  { sp->rep_penalty = 1.0f; sp->freq_penalty = 0.0f; sp->pres_penalty = 0.0f; }
    if (!listed(DRY))  sp->dry_multiplier = 0.0f;
    if (!listed(TOPK)) sp->top_k = 0;
    if (!listed(TOPP)) sp->top_p = 1.0f;
    if (!listed(MINP)) sp->min_p = 0.0f;
    if (!listed(TYP))  sp->typical_p = 1.0f;
    if (!listed(XTC))  sp->xtc_probability = 0.0f;
    if (!listed(TEMP) && sp->temp > 0.0f) sp->temp = 1.0f;

    bool on[N_STAGE] = {};
    on[PEN]  = sp->rep_penalty != 1.0f || sp->freq_penalty != 0.0f || sp->pres_penalty != 0.0f;
    on[DRY]  = sp->dry_multiplier > 0.0f;
    on[TOPK] = sp->top_k > 0;
    on[TOPP] = sp->top_p < 1.0f;
    on[MINP] = sp->min_p > 0.0f;
    on[TYP]  = sp->typical_p < 1.0f;
    on[XTC]  = sp->xtc_probability > 0.0f;
    on[TEMP] = sp->temp > 0.0f && sp->temp != 1.0f;
    if (sp->temp <= 0.0f) return RAD_OK;     /* greedy: no order to honour */

    /* The enabled stages, first occurrence each, in the list's order. */
    std::vector<int> seq;
    for (int st : list)
        if (on[st] && std::find(seq.begin(), seq.end(), st) == seq.end()) seq.push_back(st);
    auto pos = [&](int st) {
        return (int)(std::find(seq.begin(), seq.end(), st) - seq.begin());
    };
    auto refuse = [&](int a, int bst) {
        *err = err_unsupported("samplers", std::string("\"") + kName[a] + "\" after \"" +
                               kName[bst] + "\" is an order this server's sampler chain cannot run "
                               "(penalties, dry, top_k, top_p, min_p, typ_p, xtc, with temperature "
                               "first or last)");
        return RAD_E_UNSUPPORTED;
    };
    /* Everything but temperature in the chain's own relative order. */
    static const int kChain[] = { PEN, DRY, TOPK, TOPP, MINP, TYP, XTC };
    int last = -1, last_st = -1;
    for (int st : kChain) {
        if (!on[st]) continue;
        const int p = pos(st);
        if (p < last) return refuse(last_st, st);
        last = p;
        last_st = st;
    }
    if (!on[TEMP]) return RAD_OK;
    /* Temperature: after the penalties, and before every enabled cut or after every one. top-k
     * commutes with it -- a monotone rescale keeps the same k largest -- so it is not a cut here. */
    const int pt = pos(TEMP);
    for (int st : { PEN, DRY })
        if (on[st] && pos(st) > pt) return refuse(st, TEMP);
    bool before = false, after = false;
    int first_cut = -1;
    for (int st : { TOPP, MINP, TYP, XTC }) {
        if (!on[st]) continue;
        if (first_cut < 0) first_cut = st;
        (pos(st) > pt ? before : after) = true;
    }
    if (before && after) return refuse(TEMP, first_cut);
    sp->temp_last = after;
    return RAD_OK;
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
    if (!get_num_any(b, { "repetition_penalty", "repeat_penalty" }, &d, err)) return RAD_E_INVAL;
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

    /* max_tokens, with max_completion_tokens as OpenAI's newer spelling and n_predict as
     * llama.cpp's. A NEGATIVE VALUE IS llama.cpp's "no limit": whatever the context leaves, which
     * is what 0 resolves to in clamp_max_tokens -- and unlike an absent field it does not take the
     * deployment's --default-max-tokens. */
    int64_t mt = 0;
    bool mt_named = false;
    if (!get_int_any(b, { "max_tokens", "max_completion_tokens", "n_predict" }, &mt, &mt_named,
                     err))
        return RAD_E_INVAL;
    if (mt_named && mt == 0) { *err = err_bad("max_tokens", "must be >= 1, or -1 for no limit"); return RAD_E_INVAL; }
    r->max_tokens = (int32_t)(mt > 0 ? mt : mt < 0 ? 0 : lim.default_max_tokens);

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

    /* llama.cpp's streaming extras: the timings on every chunk rather than the last, and the
     * prompt's progress as chunks while it is being read. */
    if (!get_bool(b, "timings_per_token", &r->timings_per_token, err)) return RAD_E_INVAL;
    if (!get_bool(b, "return_progress", &r->return_progress, err)) return RAD_E_INVAL;

    /* OpenRouter's `session_id`: which conversation the request belongs to, as the client names
     * it. It labels the conversation on /sessions and the dashboard and changes nothing else. A
     * value that is not a string is not a label and is ignored, as the field is everywhere. */
    auto sid = b.find("session_id");
    if (sid != b.end() && sid->is_string()) r->session_label = session_label_of(sid->get<std::string>());

    RAD_TRY(apply_sampler_order(b, &r->sp, err));

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

void map_reasoning_effort(const OaiLimits& lim, ChatRenderOptions* opt) {
    if (lim.reasoning_effort_map.empty()) return;
    auto it = opt->template_kwargs.find("reasoning_effort");
    if (it == opt->template_kwargs.end()) return;
    const json v = json::parse(it->second, nullptr, false);
    if (!v.is_string()) return;
    auto m = lim.reasoning_effort_map.find(v.get<std::string>());
    if (m == lim.reasoning_effort_map.end()) return;
    if (m->second == "none" || m->second == "minimal") {
        opt->enable_thinking = false;
        opt->template_kwargs.erase(it);
    } else {
        it->second = json(m->second).dump();
    }
}

ApiError template_refusal(const ChatRenderOptions& opt, const OaiLimits& lim, int rs,
                          const std::string& why) {
    /* AN EFFORT OUTSIDE THE SET IS THE REASON, when the set is known and the word is not in it:
     * the render that refused carried no other variable the template was probed for. The
     * template's own sentence is a traceback through its source, which says nothing a client can
     * act on that the set does not. */
    auto it = opt.template_kwargs.find("reasoning_effort");
    if (it != opt.template_kwargs.end() && !lim.reasoning_efforts.empty()) {
        const json v = json::parse(it->second, nullptr, false);
        const std::vector<std::string>& set = lim.reasoning_efforts;
        if (v.is_string() && std::find(set.begin(), set.end(), v.get<std::string>()) == set.end()) {
            std::string list;
            for (size_t i = 0; i < set.size(); ++i)
                list += (i ? (i + 1 == set.size() ? " and " : ", ") : "") + ("'" + set[i] + "'");
            ApiError e = err_bad("reasoning_effort",
                                 "'" + v.get<std::string>() + "' is not a level this model's chat "
                                 "template takes. It takes " + list + "; 'none' and 'minimal' "
                                 "turn thinking off");
            e.code = "unsupported_value";
            e.supported = { "none", "minimal" };
            e.supported.insert(e.supported.end(), set.begin(), set.end());
            return e;
        }
    }
    return rs == RAD_E_UNSUPPORTED
               ? err_unsupported("messages", why.empty() ? "the chat template refused this request"
                                                         : why)
               : err_bad("messages", why.empty() ? "the chat template could not render this "
                                                   "request" : why);
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
    map_reasoning_effort(lim, opt);

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
    /* --reasoning-effort-map, before "none" and "minimal" are read, so a map can name either. */
    if (auto m = lim.reasoning_effort_map.find(effort);
        !effort.empty() && m != lim.reasoning_effort_map.end())
        effort = m->second;
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
    /* llama.cpp's `json_schema`: the schema itself at the top level, the shorthand for a
     * response_format of type json_schema. */
    auto js = b.find("json_schema");
    if (js != b.end() && !js->is_null()) {
        if (!gram.empty() || has_rf) {
            *err = err_bad("json_schema", "cannot be combined with `grammar` or `response_format`");
            return RAD_E_INVAL;
        }
        if (!js->is_object()) { *err = err_bad("json_schema", "expected a JSON Schema object"); return RAD_E_INVAL; }
        json schema = *js;
        if (schema_json) { *schema_json = schema.dump(); return RAD_OK; }
        if (!d.grammar) {
            *err = err_not_implemented("json_schema requires the JSON-schema to GBNF compiler "
                                       "(spec 13), which this build was wired without");
            return RAD_E_UNSUPPORTED;
        }
        std::string out, why;
        const int st = d.grammar->from_json_schema(schema, out, why);
        if (st < 0) { *err = err_bad("json_schema", "could not be compiled to a grammar: " + why); return st; }
        r->sp.grammar = out;
        return RAD_OK;
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
int decode_media_url(const std::string& url, const std::string& field,
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
/* ONE MEDIA CONTENT PART, read and decoded: 1 with `*kind` and `*bytes` set when `part` is an
 * image, video or audio part, 0 when it is some other type (the caller's to judge), negative with
 * `*err` set when it is a media part that cannot be served. `field` prefixes the error's field
 * name -- "messages[].content[]" for chat, "input[]" for embeddings.
 *
 * IMAGE AND VIDEO PARTS, in the two spellings clients send: OpenAI's
 * {"type":"image_url","image_url":{"url":...}} (and the bare-string form), the Responses API's
 * "input_image", and the video equivalents vLLM's clients use -- {"type":"video_url",
 * "video_url":{"url":...}} and "input_video". AUDIO is OpenAI's {"type":"input_audio",
 * "input_audio":{"data": base64, "format": "wav"|"mp3"}} -- and the other formats the decoder
 * admits by the same spelling -- or vLLM's {"type":"audio_url","audio_url":{"url": data URL}}. */
static int read_media_part(const json& part, const std::string& type, const std::string& field,
                           const OaiDeps& d, const OaiLimits& lim, MediaKind* kind,
                           std::vector<uint8_t>* bytes, ApiError* err) {
    const bool is_image = type == "image_url" || type == "input_image";
    const bool is_video = type == "video_url" || type == "input_video";
    if (is_image || is_video) {
        *kind = is_image ? MediaKind::Image : MediaKind::Video;
        const char* what = is_image ? "image" : "video";
        const bool allowed = is_image ? lim.allow_image : lim.allow_video;
        if (!allowed || !d.mm || !d.mm->accepts(*kind)) {
            *err = err_not_implemented(
                std::string(what) + " input requires a vision encoder (spec 11); this "
                "model was loaded without one");
            return RAD_E_UNSUPPORTED;
        }
        const std::string key = is_image ? "image_url" : "video_url";
        const std::string fld = field + "." + key;
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
            *err = err_bad(fld, "expected {\"url\": ...}");
            return RAD_E_INVAL;
        }
        RAD_TRY(decode_media_url(url, fld + ".url", bytes, err));
        return 1;
    }
    if (type == "input_audio" || type == "audio_url") {
        *kind = MediaKind::Audio;
        if (!lim.allow_audio || !d.mm || !d.mm->accepts(MediaKind::Audio)) {
            *err = err_not_implemented(
                "audio input requires an audio encoder (spec 11); this model was loaded "
                "without one");
            return RAD_E_UNSUPPORTED;
        }
        if (type == "audio_url") {
            const std::string fld = field + ".audio_url";
            auto au = part.find("audio_url");
            std::string url;
            if (au != part.end() && au->is_object() && au->contains("url") &&
                au->at("url").is_string())
                url = au->at("url").get<std::string>();
            else if (au != part.end() && au->is_string())
                url = au->get<std::string>();
            else {
                *err = err_bad(fld, "expected {\"url\": ...}");
                return RAD_E_INVAL;
            }
            RAD_TRY(decode_media_url(url, fld + ".url", bytes, err));
            return 1;
        }
        auto ia = part.find("input_audio");
        if (ia == part.end() || !ia->is_object() || !ia->contains("data") ||
            !ia->at("data").is_string()) {
            *err = err_bad(field + ".input_audio", "expected {\"data\": ..., \"format\": ...}");
            return RAD_E_INVAL;
        }
        auto fit = ia->find("format");
        const std::string fmt =
            fit != ia->end() && fit->is_string() ? fit->get<std::string>() : std::string();
        /* The decoder recognises the container by its bytes; the format is checked against
         * what it admits so a typo is named here rather than as an undecodable clip. */
        static const char* const kAudioFormats[] = { "wav", "mp3", "flac", "ogg", "opus",
                                                     "m4a", "aac", "webm", "aiff" };
        bool known = false;
        for (const char* f : kAudioFormats) known = known || fmt == f;
        if (!known) {
            *err = err_bad(field + ".input_audio.format",
                           "must be one of wav, mp3, flac, ogg, opus, m4a, aac, webm, aiff");
            return RAD_E_INVAL;
        }
        if (!base64_decode(ia->at("data").get<std::string>(), bytes)) {
            *err = err_bad(field + ".input_audio.data", "invalid base64");
            return RAD_E_INVAL;
        }
        return 1;
    }
    return 0;
}

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
            /* IMAGE, VIDEO AND AUDIO PARTS (read_media_part). A MEDIA MARKER, NOT TEXT, takes the
             * part's place: the chat layer flattens a message's parts into one string and joins
             * TEXT parts with a newline; it joins a marker to its neighbours with nothing, which
             * is what the model's own template does with an image part (<|vision_end|> runs
             * straight into the text after it). Joined as text, every picture would cost one
             * '\n' token the model's own prompt does not have. */
            {
                MediaKind kind;
                std::vector<uint8_t> bytes;
                const int mr = read_media_part(part, type, "messages[].content[]", d, lim, &kind,
                                               &bytes, err);
                if (mr < 0) return mr;
                if (mr > 0) {
                    media->emplace_back(kind, std::move(bytes));
                    part.erase(type == "input_audio" ? "input_audio"
                               : type == "audio_url" ? "audio_url"
                               : (type == "image_url" || type == "input_image") ? "image_url"
                                                                                : "video_url");
                    part["type"] = "media_marker";
                    part["text"] = MEDIA_MARKER;
                    continue;
                }
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
                               const MediaOpts& opts, PromptInput* out, ApiError* err) {
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
        const int s = d.mm->prepare(media[mi].first, media[mi].second, opts, &item, &why);
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
        /* WORDED FOR THE CLIENTS THAT READ IT. An agent compacts its history when a request
         * overflows, and it recognises the overflow by the error's text: OpenAI's "maximum
         * context length is N tokens" and llama-server's "exceeds the available context size"
         * are the two phrases those matchers look for, so the sentence carries both, beside
         * OpenAI's code and llama-server's two numbers. */
        char msg[384];
        snprintf(msg, sizeof msg,
                 "This model's maximum context length is %lld tokens: the request (%zu tokens) "
                 "exceeds the available context size (%lld tokens) and leaves nothing to generate "
                 "into; shorten the prompt by at least %lld tokens",
                 (long long)lim.max_ctx, n_prompt, (long long)lim.max_ctx, (long long)(1 - room));
        *err = err_bad(field, msg);
        err->message = msg;
        err->code = "context_length_exceeded";
        err->n_prompt_tokens = (int64_t)n_prompt;
        err->n_ctx = lim.max_ctx;
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
    for (const char* k : { "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
                           "reasoning_budget_tokens" }) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        int64_t v = 0;
        if (!get_int(b, k, &v, err)) return RAD_E_INVAL;
        /* llama.cpp's spellings take a negative budget for "none": said, it overrides the
         * deployment's default as a number does. */
        if (v < 0) v = -1;
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
    if (budget < 0) return RAD_OK;
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

/* LOGPROBS, in each endpoint's spelling, into one count.
 *
 *   chat         `logprobs` true, with `top_logprobs` alternatives (0 when absent, OpenAI's default)
 *   completions  `logprobs` the number of alternatives, OpenAI's legacy integer
 *   any          llama-server's `n_probs`, which wins over both
 *
 * A count of 0 still returns each token's own logprob on chat (OpenAI's meaning) and returns
 * nothing on completions (llama-server's: its `logprobs` 0 and `n_probs` 0 are both "off"). The
 * numbers are the RAW distribution's, as OpenAI and llama-server report them. */
static int parse_logprobs(const json& b, const OaiLimits& lim, Endpoint ep, OaiRequest* out,
                          ApiError* err) {
    bool want = false;
    int64_t top = 0;
    const char* field = "logprobs";
    if (ep == Endpoint::Chat) {
        if (!get_bool(b, "logprobs", &want, err)) return RAD_E_INVAL;
        if (!get_int(b, "top_logprobs", &top, err)) return RAD_E_INVAL;
        auto t = b.find("top_logprobs");
        const bool named = t != b.end() && !t->is_null();
        if (!want && named) {
            *err = err_bad("top_logprobs", "requires logprobs to be true");
            return RAD_E_INVAL;
        }
        if (named) field = "top_logprobs";
    } else {
        auto lit = b.find("logprobs");
        if (lit != b.end() && !lit->is_null()) {
            if (!lit->is_number_integer()) {
                *err = err_bad("logprobs", "expected an integer (the number of alternatives)");
                return RAD_E_INVAL;
            }
            top = lit->get<int64_t>();
            want = top > 0;
        }
    }
    auto np = b.find("n_probs");
    if (np != b.end() && !np->is_null()) {
        if (!np->is_number_integer()) {
            *err = err_bad("n_probs", "expected an integer (the number of alternatives)");
            return RAD_E_INVAL;
        }
        top = np->get<int64_t>();
        want = top > 0;
        field = "n_probs";
    }
    if (top < 0 || top > RAD_MAX_LOGPROBS) {
        *err = err_bad(field, "must be in [0, " + std::to_string(RAD_MAX_LOGPROBS) + "]");
        return RAD_E_INVAL;
    }
    if (want && !lim.supports_logprobs) {
        *err = err_unsupported(field, "the kernel libraries loaded have no sample_logprobs");
        return RAD_E_UNSUPPORTED;
    }
    out->want_logprobs = want;
    out->n_logprobs = (int)top;
    out->sp.n_probs = want ? (int)top : -1;
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
    if (!lim.version.empty()) out->system_fingerprint = "radiance-" + lim.version;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    RAD_TRY(parse_sampling(b, lim, out, err));
    std::string schema_json;
    RAD_TRY(parse_output_format(b, d, out, err, &schema_json));

    RAD_TRY(parse_logprobs(b, lim, Endpoint::Chat, out, err));

    if (!d.chat) {
        *err = err_not_implemented(
            "/v1/chat/completions requires a chat template (spec 12); this model was loaded "
            "without one. Use /v1/completions");
        return RAD_E_UNSUPPORTED;
    }

    auto mit = b.find("messages");
    if (mit == b.end()) { *err = err_bad("messages", "is required"); return RAD_E_INVAL; }
    json messages = *mit;
    /* The deprecated functions API's turns, as the current API spells them: an assistant
     * message's `function_call` is one tool call, and a `function` message is the result of the
     * call before it. The call ids pair them; the old API has none. */
    if (messages.is_array()) {
        std::string call_id;
        int n_calls = 0;
        for (json& m : messages) {
            const std::string role = str_or(m, "role");
            auto fc = m.find("function_call");
            if (role == "assistant" && fc != m.end() && fc->is_object() && !m.contains("tool_calls")) {
                call_id = "call_" + std::to_string(n_calls++);
                m["tool_calls"] = json::array(
                    { json{ { "id", call_id }, { "type", "function" }, { "function", *fc } } });
                m.erase("function_call");
            } else if (role == "function") {
                m["role"] = "tool";
                m["tool_call_id"] = call_id;
            }
        }
    }
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
                    if (str_or(part, "type") == "text") take(str_or(part, "text"));
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

    /* OpenAI's DEPRECATED functions API, which older clients still send: `functions` is `tools`
     * with each entry unwrapped, and `function_call` is `tool_choice` -- "none", "auto", or
     * {"name": ...}. Rewritten into the current fields, and refused beside them. The old API
     * answers one call at most, so the grammar admits one. */
    auto fns = b.find("functions");
    if (fns != b.end() && !fns->is_null()) {
        if (b.contains("tools") && !b.at("tools").is_null()) {
            *err = err_bad("functions", "send `tools` or the deprecated `functions`, not both");
            return RAD_E_INVAL;
        }
        if (!fns->is_array()) { *err = err_bad("functions", "expected an array"); return RAD_E_INVAL; }
        json t = json::array();
        for (const json& f : *fns) t.push_back(json{ { "type", "function" }, { "function", f } });
        b["tools"] = std::move(t);
        b["parallel_tool_calls"] = false;
        out->legacy_functions = true;
    }
    auto fcall = b.find("function_call");
    if (fcall != b.end() && !fcall->is_null()) {
        if (b.contains("tool_choice") && !b.at("tool_choice").is_null()) {
            *err = err_bad("function_call",
                           "send `tool_choice` or the deprecated `function_call`, not both");
            return RAD_E_INVAL;
        }
        if (fcall->is_string()) {
            b["tool_choice"] = *fcall;
        } else if (fcall->is_object() && fcall->contains("name") && fcall->at("name").is_string()) {
            b["tool_choice"] = json{ { "type", "function" },
                                     { "function", { { "name", fcall->at("name") } } } };
        } else {
            *err = err_bad("function_call", "expected \"none\", \"auto\" or {\"name\": ...}");
            return RAD_E_INVAL;
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

    /* A forced choice -- "required", or an object naming one function -- makes the call grammar
     * the whole reply instead of a lazy one: the reasoning block, then the calls (one when
     * `parallel_tool_calls` is false), then nothing, so the reply ends at the last call. A named
     * function narrows the calls to that one. */
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
        if (tools.empty()) {
            *err = err_bad("tool_choice", "a tool call is required, but `tools` is empty");
            return RAD_E_INVAL;
        }
        if (!named.empty() &&
            std::none_of(tools.begin(), tools.end(), [&](const json& t) {
                return t.at("function").at("name") == named;
            })) {
            *err = err_bad("tool_choice", "names \"" + named + "\", which is not in `tools`");
            return RAD_E_INVAL;
        }
        if (!schema_json.empty() || !out->sp.grammar.empty()) {
            *err = err_bad("tool_choice", "a forced tool call is the whole reply, so it cannot be "
                                          "combined with `response_format` or `grammar`");
            return RAD_E_INVAL;
        }
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
    opt.quiet = lim.probe;
    opt.tool_choice = tool_choice;
    opt.tool_name = named;
    opt.parallel_tool_calls = parallel;
    const bool agp_named = b.contains("add_generation_prompt") && !b.at("add_generation_prompt").is_null();
    if (!get_bool(b, "add_generation_prompt", &opt.add_generation_prompt, err)) return RAD_E_INVAL;

    /* llama.cpp's `reasoning_format`. "none" leaves the reasoning in `content` between the
     * template's own markers; "auto" and "deepseek" move it to `reasoning_content`, which is what
     * this server does anyway. "deepseek-legacy" -- both at once -- is not served. */
    std::string rfmt;
    if (!get_str(b, "reasoning_format", &rfmt, err)) return RAD_E_INVAL;
    if (rfmt == "deepseek-legacy") {
        *err = err_unsupported("reasoning_format", "\"deepseek-legacy\" keeps the reasoning in "
                               "`content` and `reasoning_content` both, which is not served; send "
                               "\"none\" or \"deepseek\"");
        return RAD_E_UNSUPPORTED;
    } else if (!rfmt.empty() && rfmt != "none" && rfmt != "auto" && rfmt != "deepseek") {
        *err = err_bad("reasoning_format", "must be \"none\", \"auto\" or \"deepseek\"");
        return RAD_E_INVAL;
    }

    /* ASSISTANT PREFILL. A conversation that ends in an assistant message is continued, not
     * answered: the reply picks up where that message stops. llama.cpp's default and the
     * Anthropic API's meaning; vLLM's `continue_final_message` names it explicitly, true or
     * llama.cpp's "content" / "reasoning_content". A request that names `add_generation_prompt`
     * true opens a new turn after it instead, as vLLM does; naming that and
     * `continue_final_message` both is a contradiction, refused as llama.cpp refuses it. */
    {
        std::string cont;
        auto cf = b.find("continue_final_message");
        if (cf != b.end() && !cf->is_null()) {
            if (cf->is_boolean()) cont = cf->get<bool>() ? "auto" : "";
            else if (cf->is_string() && (*cf == "content" || *cf == "reasoning_content"))
                cont = cf->get<std::string>();
            else {
                *err = err_bad("continue_final_message",
                               "expected a boolean, \"content\" or \"reasoning_content\"");
                return RAD_E_INVAL;
            }
            if (!cont.empty() && agp_named && opt.add_generation_prompt) {
                *err = err_bad("continue_final_message",
                               "cannot be combined with add_generation_prompt true");
                return RAD_E_INVAL;
            }
        }
        const json* last = messages.is_array() && !messages.empty() ? &messages.back() : nullptr;
        const bool ends_assistant = last && str_or(*last, "role") == "assistant";
        if (cont.empty() && cf == b.end() && ends_assistant && !agp_named) cont = "auto";
        if (!cont.empty()) {
            if (!ends_assistant) {
                *err = err_bad("continue_final_message",
                               "the last message is not an assistant message to continue");
                return RAD_E_INVAL;
            }
            if (last->contains("tool_calls") && !last->at("tool_calls").is_null() &&
                !last->at("tool_calls").empty()) {
                *err = err_bad("messages", "an assistant message that made tool calls cannot be "
                                           "continued");
                return RAD_E_INVAL;
            }
            if (messages.size() >= 2 && str_or(messages[messages.size() - 2], "role") == "assistant") {
                *err = err_bad("messages", "two assistant messages at the end cannot be "
                                           "continued");
                return RAD_E_INVAL;
            }
            opt.continue_final = cont;
            opt.add_generation_prompt = false;
            const json& c = last->contains("content") ? last->at("content") : json();
            if (c.is_string()) out->continued = c.get<std::string>();
            else if (c.is_array())
                for (const json& p : c)
                    if (str_or(p, "type") == "text") out->continued += str_or(p, "text");
        }
    }

    /* The template's grammar and the caller's are ONE grammar, not two, and the template is the
     * one that knows how to combine them -- a tool-calling format has to leave room for the call
     * syntax around whatever the caller constrained. So both halves go IN and the merged result
     * comes back out: a raw GBNF as `grammar`, a `response_format` as the schema it started as. */
    opt.grammar     = out->sp.grammar;
    opt.json_schema = schema_json;

    RAD_TRY(parse_thinking(b, lim, &opt, err));
    /* The request's reasoning_format over the deployment's (--reasoning-format), which the
     * template defaults above have just applied. */
    if (!rfmt.empty()) opt.reasoning_format = rfmt;

    std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
    RAD_TRY(lower_content_parts(messages, d, lim, &media, err));

    ChatRender render;
    std::string why;
    const int rs = d.chat->apply(messages, tools, opt, &render, &why);
    if (rs < 0) {
        *err = template_refusal(opt, lim, rs, why);
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
    out->n_client_stops = out->stop.size();
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
        RAD_TRY(tokenize_with_media(rendered, d, media, MediaOpts{}, &p, err));
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
                             OaiRequest* out, ApiError* err, bool native) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Completion, err, native));
    if (native && !get_bool(b, "return_tokens", &out->return_tokens, err)) return RAD_E_INVAL;

    out->endpoint = Endpoint::Completion;
    out->id = random_id("cmpl-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!lim.version.empty()) out->system_fingerprint = "radiance-" + lim.version;
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

    RAD_TRY(parse_logprobs(b, lim, Endpoint::Completion, out, err));
    /* `echo` with logprobs asks for the PROMPT's logprobs too, which are not computed: a prompt
     * position's distribution never leaves the card. */
    if (out->echo && out->want_logprobs) {
        *err = err_unsupported("echo", "logprobs for the prompt are not returned; send echo "
                                       "without logprobs, or logprobs without echo");
        return RAD_E_UNSUPPORTED;
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

    /* llama-server's mixed prompt: strings and token ids in one array are ONE prompt, the strings
     * tokenised in place, only the first of them with the BOS. */
    bool mixed = false;
    if (native && pit->is_array()) {
        bool s = false, n = false;
        for (const auto& v : *pit) { s = s || v.is_string(); n = n || v.is_number_integer(); }
        mixed = s && n;
    }
    if (mixed) {
        PromptInput p;
        bool first = true;
        for (const auto& v : *pit) {
            if (v.is_string()) {
                const std::vector<int32_t> t = d.tok->encode(v.get<std::string>(), first, true);
                p.tokens.insert(p.tokens.end(), t.begin(), t.end());
            } else {
                int32_t id = 0;
                if (!v.is_number_integer() || !token_id(v, &id)) {
                    *err = err_bad("prompt", "a mixed prompt holds strings and token ids only");
                    return RAD_E_INVAL;
                }
                p.tokens.push_back(id);
            }
            first = false;
        }
        p.text = d.tok->decode(p.tokens);
        out->prompts.push_back(std::move(p));
    } else if (pit->is_string()) {
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
        const json& dv = b.at("dimensions");
        if (!dv.is_number_integer() || dv.get<int64_t>() <= 0) {
            *err = err_bad("dimensions", "must be a positive integer");
            return RAD_E_INVAL;
        }
        /* Bounded by the model's width in the handler, which knows it. */
        out->dimensions = dv.get<int64_t>();
    }
    /* A soft-token budget for this request's images (Gemma 4's processor); the processor refuses
     * it on a model that sizes images otherwise. */
    if (b.contains("image_tokens") && !b.at("image_tokens").is_null()) {
        const json& iv = b.at("image_tokens");
        const int64_t n = iv.is_number_integer() ? iv.get<int64_t>() : 0;
        if (n != 70 && n != 140 && n != 280 && n != 560 && n != 1120) {
            *err = err_bad("image_tokens", "must be one of 70, 140, 280, 560, 1120");
            return RAD_E_INVAL;
        }
        out->image_tokens = (int32_t)n;
    }

    /* THE NAMED PROMPT: sentence-transformers' `prompt_name`, the model's own text in front of
     * each text or content-part input. A model that states a default prompt applies it to a
     * request that names none, as its reference library does. */
    std::string prefix;
    const bool prompt_named = b.contains("prompt_name") && !b.at("prompt_name").is_null();
    {
        std::string pname;
        const bool named = prompt_named;
        if (named) {
            if (!b.at("prompt_name").is_string()) {
                *err = err_bad("prompt_name", "expected a string");
                return RAD_E_INVAL;
            }
            pname = b.at("prompt_name").get<std::string>();
        } else if (d.embed) {
            pname = d.embed->default_prompt();
        }
        if (!pname.empty() && (!d.embed || !d.embed->prompt(pname, &prefix))) {
            *err = err_bad("prompt_name", "this model states no prompt named \"" + pname + "\"");
            return RAD_E_INVAL;
        }
    }

    /* THE MESSAGES FORM, vLLM's chat embedding: one input rendered through the model's own chat
     * template (EmbeddingGemma 2's concatenates the system text and then every message's text and
     * media placeholders), the tokenizer's special tokens around the whole as for an input. */
    if (b.contains("messages") && !b.at("messages").is_null()) {
        if (b.contains("input") && !b.at("input").is_null()) {
            *err = err_bad("messages", "a request takes `input` or `messages`, not both");
            return RAD_E_INVAL;
        }
        if (b.contains("prompt_name") && !b.at("prompt_name").is_null()) {
            *err = err_bad("prompt_name", "a `messages` input is rendered by the chat template "
                                          "and takes no prompt");
            return RAD_E_INVAL;
        }
        IChatTemplate* tpl = d.embed ? d.embed->message_template() : nullptr;
        if (!tpl) {
            *err = err_bad("messages", "this model has no chat template to render messages with");
            return RAD_E_INVAL;
        }
        json messages = b.at("messages");
        std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
        RAD_TRY(lower_content_parts(messages, d, lim, &media, err));
        ChatRenderOptions opt;
        opt.add_generation_prompt = false;
        ChatRender render;
        std::string why;
        if (tpl->apply(messages, json::array(), opt, &render, &why) < 0 || render.prompt.empty()) {
            *err = err_bad("messages", "the chat template did not render the messages" +
                                           (why.empty() ? std::string() : ": " + why));
            return RAD_E_INVAL;
        }
        PromptInput p;
        if (media.empty()) {
            p.text = render.prompt;
            p.tokens = d.tok->encode(render.prompt, true, /*parse_special=*/true);
        } else {
            MediaOpts mo;
            mo.image_tokens = out->image_tokens;
            RAD_TRY(tokenize_with_media(render.prompt, d, media, mo, &p, err));
            const ITokenizer::VocabInfo vi = d.tok->vocab_info();
            if (vi.add_bos && vi.bos >= 0) {
                p.tokens.insert(p.tokens.begin(), vi.bos);
                for (auto& pl : p.media->items) pl.tok += 1;
                p.media->layout();
            }
            if (vi.add_eos && vi.eos >= 0) p.tokens.push_back(vi.eos);
        }
        out->prompts.push_back(std::move(p));
        return RAD_OK;
    }

    auto it = b.find("input");
    if (it == b.end() || it->is_null()) { *err = err_bad("input", "is required"); return RAD_E_INVAL; }

    auto push_ids = [&](const json& arr, ApiError* e) -> bool {
        /* A prompt is text; token ids take none, and asking for one by name is refused rather
         * than dropped. A model's default prompt applies to text inputs only. */
        if (prompt_named && !prefix.empty()) {
            *e = err_bad("prompt_name", "a prompt is text and cannot go in front of token ids");
            return false;
        }
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

    /* ONE INPUT OF CONTENT PARTS -- text, images, video, audio, in order -- the same parts a chat
     * message carries, and one vector for the lot. The parts are joined with nothing between
     * them and each media part becomes its processor's token run, which is what an embedding
     * model's own template does (EmbeddingGemma 2's concatenates text and placeholders), and the
     * tokenizer's own special tokens go around the whole: what HuggingFace's processor feeds the
     * model. */
    auto push_parts = [&](const json& parts, ApiError* e) -> int {
        std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
        std::string rendered = prefix;
        for (const auto& part : parts) {
            if (!part.is_object()) {
                *e = err_bad("input[]", "a content part must be an object");
                return RAD_E_INVAL;
            }
            auto tit = part.find("type");
            if (tit == part.end() || !tit->is_string()) {
                *e = err_bad("input[].type", "expected a string");
                return RAD_E_INVAL;
            }
            const std::string type = tit->get<std::string>();
            if (type == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    *e = err_bad("input[].text", "expected a string");
                    return RAD_E_INVAL;
                }
                const std::string t = part.at("text").get<std::string>();
                /* The marker is how a media part's position survives into tokenisation, so text
                 * that spells it would be taken for one. */
                if (t.find(MEDIA_MARKER) != std::string::npos) {
                    *e = err_bad("input[].text", std::string("contains \"") + MEDIA_MARKER +
                                                     "\", which this server reserves for media");
                    return RAD_E_INVAL;
                }
                rendered += t;
                continue;
            }
            MediaKind kind;
            std::vector<uint8_t> bytes;
            const int mr = read_media_part(part, type, "input[]", d, lim, &kind, &bytes, e);
            if (mr < 0) return mr;
            if (mr == 0) {
                *e = err_bad("input[].type", "unsupported content part type \"" + type + "\"");
                return RAD_E_INVAL;
            }
            media.emplace_back(kind, std::move(bytes));
            rendered += MEDIA_MARKER;
        }
        PromptInput p;
        if (media.empty()) {
            p.text = rendered;
            p.tokens = d.tok->encode(rendered, true, /*parse_special=*/true);
        } else {
            MediaOpts mo;
            mo.image_tokens = out->image_tokens;
            RAD_TRY(tokenize_with_media(rendered, d, media, mo, &p, e));
            /* tokenize_with_media adds no special tokens (a chat template renders its own);
             * this input has no template, so the tokenizer's own rule applies around it. */
            const ITokenizer::VocabInfo vi = d.tok->vocab_info();
            if (vi.add_bos && vi.bos >= 0) {
                p.tokens.insert(p.tokens.begin(), vi.bos);
                for (auto& pl : p.media->items) pl.tok += 1;
                p.media->layout();
            }
            if (vi.add_eos && vi.eos >= 0) p.tokens.push_back(vi.eos);
        }
        out->prompts.push_back(std::move(p));
        return RAD_OK;
    };
    auto is_parts = [](const json& a) { return a.is_array() && !a.empty() && a.front().is_object(); };

    /* Parsed like a /v1/completions prompt, and for the same reason: the input is the client's
     * own text with no template around it. */
    if (it->is_string()) {
        PromptInput p;
        p.text = prefix + it->get<std::string>();
        p.tokens = d.tok->encode(p.text, true, /*parse_special=*/true);
        out->prompts.push_back(std::move(p));
    } else if (it->is_array()) {
        if (it->empty()) { *err = err_bad("input", "must not be empty"); return RAD_E_INVAL; }
        if (it->front().is_number_integer()) {
            if (!push_ids(*it, err)) return RAD_E_INVAL;
        } else if (is_parts(*it)) {
            RAD_TRY(push_parts(*it, err));
        } else {
            /* A BATCH: each element its own input -- a string, a token array or an array of
             * content parts. */
            for (const auto& el : *it) {
                if (el.is_string()) {
                    PromptInput p;
                    p.text = prefix + el.get<std::string>();
                    p.tokens = d.tok->encode(p.text, true, /*parse_special=*/true);
                    out->prompts.push_back(std::move(p));
                } else if (is_parts(el)) {
                    RAD_TRY(push_parts(el, err));
                } else if (el.is_array()) {
                    if (!push_ids(el, err)) return RAD_E_INVAL;
                } else {
                    *err = err_bad("input", "expected strings, token arrays or arrays of content "
                                            "parts");
                    return RAD_E_INVAL;
                }
            }
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
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
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

static json lp_bytes(const std::string& b) {
    json a = json::array();
    for (unsigned char c : b) a.push_back((int)c);
    return a;
}

static json lp_entry(const LogprobEntry& e) {
    json top = json::array();
    for (const LogprobAlt& a : e.top)
        top.push_back(json{ { "id", a.token }, { "token", a.text }, { "bytes", lp_bytes(a.bytes) },
                            { "logprob", a.logprob } });
    return json{ { "id", e.token }, { "token", e.text }, { "bytes", lp_bytes(e.bytes) },
                 { "logprob", e.logprob }, { "top_logprobs", std::move(top) } };
}

static json lp_content(const LogprobEntry* e, size_t n) {
    json a = json::array();
    for (size_t i = 0; i < n; ++i) a.push_back(lp_entry(e[i]));
    return a;
}

std::string logprobs_content(const LogprobEntry* e, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        if (i) s += ',';
        s += lp_entry(e[i]).dump();
    }
    return s;
}

/* OpenAI's legacy shape: parallel arrays, and the alternatives as one {text: logprob} object a
 * token. Two alternatives with the same text -- distinct ids that decode alike -- collapse to the
 * likelier, which is what that shape can say. */
static json lp_legacy(const LogprobEntry* e, size_t n, size_t offset) {
    json toks = json::array(), lps = json::array(), tops = json::array(), offs = json::array();
    for (size_t i = 0; i < n; ++i) {
        toks.push_back(e[i].text);
        lps.push_back(e[i].logprob);
        json t = json::object();
        for (const LogprobAlt& a : e[i].top)
            if (!t.contains(a.text)) t[a.text] = a.logprob;
        tops.push_back(std::move(t));
        /* Where the token's text starts in the choice's, which is where the one before it ended:
         * its own `text` can be shorter than what it added (half a character, held back). */
        offs.push_back(offset);
        offset = e[i].end;
    }
    return json{ { "tokens", std::move(toks) }, { "token_logprobs", std::move(lps) },
                 { "top_logprobs", std::move(tops) }, { "text_offset", std::move(offs) },
                 { "content", lp_content(e, n) } };
}

std::string logprobs_legacy(const LogprobEntry* e, size_t n, size_t offset) {
    return lp_legacy(e, n, offset).dump();
}

static json chat_logprobs_json(const std::vector<LogprobEntry>& lp) {
    json o;
    o["content"] = lp_content(lp.data(), lp.size());
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
        if (!c.msg.tool_calls.empty() && r.legacy_functions)
            m["function_call"] = json{ { "name", c.msg.tool_calls[0].name },
                                       { "arguments", c.msg.tool_calls[0].arguments } };
        else if (!c.msg.tool_calls.empty())
            m["tool_calls"] = tool_calls_json(c.msg.tool_calls);

        json ch;
        ch["index"] = c.index;
        ch["message"] = std::move(m);
        if (r.want_logprobs) ch["logprobs"] = chat_logprobs_json(c.logprobs);
        else ch["logprobs"] = nullptr;
        ch["finish_reason"] = r.legacy_functions && c.finish == Finish::ToolCalls
                                  ? "function_call" : finish_reason_oai(c.finish);
        arr.push_back(std::move(ch));
    }
    json o;
    o["id"] = r.id;
    o["object"] = "chat.completion";
    o["created"] = r.created;
    o["model"] = r.model;
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
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
            ch["logprobs"] = lp_legacy(c.logprobs.data(), c.logprobs.size(), 0);
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
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
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

static json model_card(const ModelCard& c) {
    json m;
    m["id"] = c.id;
    m["object"] = "model";
    m["created"] = c.created;
    m["owned_by"] = "radiance";
    if (c.max_model_len > 0) m["max_model_len"] = c.max_model_len;
    if (!c.root.empty()) m["root"] = c.root;
    if (!c.capabilities.empty()) {
        json caps = json::parse(c.capabilities, nullptr, false);
        if (!caps.is_discarded() && caps.is_object()) m["capabilities"] = std::move(caps);
    }
    return m;
}

std::string models_body(const ModelCard& c) {
    json o;
    o["object"] = "list";
    o["data"] = json::array({model_card(c)});
    return o.dump();
}

std::string model_body(const ModelCard& c) {
    return model_card(c).dump();
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

ReplyStream::ReplyStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : reader_(parser ? parser->open() : nullptr), id_(r.id), model_(r.model),
      fingerprint_(r.system_fingerprint), created_(r.created), index_(index) {}

/* What one increment adds to the message the stream adds up to, and the id a call is announced
 * with: minted here, because the parser reads the model's text and the model writes none. Returns
 * false for an increment that adds nothing and is not sent. */
bool ReplyStream::absorb(const ReplyDelta& d) {
    switch (d.kind) {
    case ReplyDelta::Reasoning:
        if (d.text.empty()) return false;
        last_.reasoning += d.text;
        return true;
    case ReplyDelta::Content:
        if (d.text.empty()) return false;
        last_.content += d.text;
        return true;
    case ReplyDelta::CallBegin: {
        ToolCall tc;
        tc.id   = random_id("call_");
        tc.name = d.text;
        last_.tool_calls.push_back(std::move(tc));
        return true;
    }
    case ReplyDelta::CallArgs:
        if (d.text.empty() || d.call >= last_.tool_calls.size()) return false;
        last_.tool_calls[d.call].arguments += d.text;
        return true;
    case ReplyDelta::CallEnd:
        return true;
    }
    return false;
}

std::string ReplyStream::push(std::string_view appended, bool is_final) {
    if (finished_) return std::string();
    std::string out;
    if (!reader_) {
        /* Nothing to parse: the text is the answer, as it arrives. */
        if (!appended.empty()) {
            ev_.clear();
            ReplyDelta d;
            d.kind = ReplyDelta::Content;
            d.text = std::string(appended);
            ev_.push_back(std::move(d));
            out = emit(ev_);
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

std::string ReplyStream::error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) {
    return final_chunk(Finish::Error, t, u) + sse_error(e);
}

bool calls_complete(const ParsedMessage& m) {
    if (m.tool_calls.empty() || m.open_call) return false;
    for (const auto& tc : m.tool_calls)
        if (tc.name.empty() || !json_complete(tc.arguments)) return false;
    return true;
}

Finish ReplyStream::adjusted_finish(Finish f) const {
    if (f != Finish::Stop && f != Finish::StopString) return f;
    /* Only claim tool_calls if every call closed and its arguments parse. A truncated arguments
     * object reported as a tool call is a tool invoked with the wrong arguments (json_partial.h).
     * The deltas of an open call have gone out already; the finish reason says not to run it. */
    return calls_complete(last_) ? Finish::ToolCalls : f;
}

/* ------------------------------------------------------------------ OpenAI's chunks */

std::string ChatDeltaStream::chunk(json delta, const char* finish_reason, const TimingsOut* t) {
    json ch;
    ch["index"] = index_;
    ch["delta"] = std::move(delta);
    if (!lp_.empty()) {
        ch["logprobs"] = json{ { "content", json::parse("[" + lp_ + "]") } };
        lp_.clear();
    }
    if (finish_reason) ch["finish_reason"] = finish_reason;
    else ch["finish_reason"] = nullptr;
    json o;
    o["id"] = id_;
    o["object"] = "chat.completion.chunk";
    o["created"] = created_;
    o["model"] = model_;
    if (!fingerprint_.empty()) o["system_fingerprint"] = fingerprint_;
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
                ",\"model\":" + json(model_).dump() +
                (fingerprint_.empty() ? std::string()
                                      : ",\"system_fingerprint\":" + json(fingerprint_).dump()) +
                ",\"choices\":[{\"index\":" +
                json(index_).dump() + ",\"delta\":";
    out += head_;
}

void ChatDeltaStream::add_logprobs(const std::string& elems) {
    if (elems.empty()) return;
    if (!lp_.empty()) lp_ += ',';
    lp_ += elems;
}

void ChatDeltaStream::delta_close(std::string& out) {
    if (!lp_.empty()) {
        out += ",\"logprobs\":{\"content\":[";
        out += lp_;
        out += "]}";
        lp_.clear();
    }
    if (timings_.empty()) { out += ",\"finish_reason\":null}]}\n\n"; return; }
    out += ",\"finish_reason\":null}],";
    out += timings_;
    out += "}\n\n";
}

void ChatDeltaStream::set_timings(const TimingsOut& t) {
    timings_.clear();
    if (t.valid) timings_ = "\"timings\":" + timings_json(t).dump();
}

/* Text that is not well-formed UTF-8 goes to nlohmann, which refuses it with the exception the
 * tree in chunk() raises for it. */
void ReplyStream::quote(std::string& out, std::string_view s) {
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
 * an index the client can never resolve. */
std::string ChatDeltaStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        if (!absorb(d) || d.kind == ReplyDelta::CallEnd) continue;
        /* The functions API has one call; a second one past the grammar's one is not sent. */
        if (legacy_ && (d.kind == ReplyDelta::CallBegin || d.kind == ReplyDelta::CallArgs) &&
            (d.kind == ReplyDelta::CallBegin ? last_.tool_calls.size() - 1 : d.call) > 0)
            continue;
        delta_open(out);
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            out += "{\"reasoning_content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::Content:
            out += "{\"content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::CallBegin: {
            const ToolCall& tc = last_.tool_calls.back();
            if (legacy_) {
                out += "{\"function_call\":{\"name\":";
                quote(out, tc.name);
                out += ",\"arguments\":\"\"}}";
                break;
            }
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"id\":";
            quote(out, tc.id);
            out += ",\"type\":\"function\",\"function\":{\"name\":";
            quote(out, tc.name);
            out += ",\"arguments\":\"\"}}]}";
            break;
        }
        case ReplyDelta::CallArgs:
            if (legacy_) {
                out += "{\"function_call\":{\"arguments\":";
                quote(out, d.text);
                out += "}}";
                break;
            }
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"function\":{\"arguments\":";
            quote(out, d.text);
            out += "}}]}";
            break;
        case ReplyDelta::CallEnd:
            break;
        }
        delta_close(out);
    }
    return out;
}

std::string ChatDeltaStream::final_chunk(Finish f, const TimingsOut& t, const UsageOut&) {
    /* The generation is over, so whatever the reader was holding is resolved now. A caller that
     * already said so gets nothing twice. */
    std::string out = push(std::string_view(), true);
    const Finish a = adjusted_finish(f);
    out += chunk(json::object(), legacy_ && a == Finish::ToolCalls ? "function_call"
                                                                   : finish_reason_oai(a), &t);
    return out;
}

/* ================================================================== Anthropic's Messages API */

namespace {

/* CLAUDE CODE STAMPS EVERY REQUEST'S SYSTEM PROMPT. Its first block opens
 * `x-anthropic-billing-header: cc_version=...; cc_entrypoint=cli; cch=a5145;` and the cch value
 * changes per request, so the very first tokens of the prompt differ every turn and the prefix
 * cache can never reuse a conversation. It is an integrity stamp for Anthropic's billing and
 * means nothing to a model; held constant, as llama-server holds it. */
void hold_billing_stamp(std::string& s) {
    static const char kHead[] = "x-anthropic-billing-header:";
    if (s.rfind(kHead, 0) != 0) return;
    const size_t at = s.find("cch=", sizeof kHead - 1);
    if (at == std::string::npos) return;
    const size_t end = s.find(';', at + 4);
    if (end == std::string::npos) return;
    for (size_t i = at + 4; i < end; ++i) s[i] = 'f';
}

/* An image block's source as the content part the chat path reads. A URL source stays a URL,
 * which the chat path refuses as it refuses every remote fetch (decode_media_url). `at` names the
 * block in a refusal. */
int anthropic_image(const json& block, const std::string& at, json* out, ApiError* err) {
    auto sit = block.find("source");
    if (sit == block.end() || !sit->is_object()) {
        *err = err_bad(at + ".source", "expected an object");
        return RAD_E_INVAL;
    }
    const json& src = *sit;
    std::string type, url;
    if (!member_str(src, "type", (at + ".source.type").c_str(), &type, err)) return RAD_E_INVAL;
    if (type == "base64") {
        std::string media = "image/jpeg", data;
        if (!member_str(src, "media_type", (at + ".source.media_type").c_str(), &media, err) ||
            !member_str(src, "data", (at + ".source.data").c_str(), &data, err))
            return RAD_E_INVAL;
        url = "data:" + media + ";base64," + data;
    } else if (!member_str(src, "url", (at + ".source.url").c_str(), &url, err)) {
        return RAD_E_INVAL;
    }
    *out = json{ { "type", "image_url" }, { "image_url", { { "url", url } } } };
    return RAD_OK;
}

}  /* namespace */

int anthropic_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* label, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    json o = json::object();
    json msgs = json::array();

    /* The system prompt: a string or text blocks, which become one system message. */
    auto sys = b.find("system");
    if (sys != b.end() && !sys->is_null()) {
        std::string text;
        if (sys->is_string()) {
            text = sys->get<std::string>();
            hold_billing_stamp(text);
        } else if (sys->is_array()) {
            for (const json& blk : *sys) {
                if (str_or(blk, "type") != "text") {
                    *err = err_bad("system", "expected a string or text blocks");
                    return RAD_E_INVAL;
                }
                std::string t;
                if (!member_str(blk, "text", "system[].text", &t, err)) return RAD_E_INVAL;
                hold_billing_stamp(t);
                if (!text.empty()) text += "\n";
                text += t;
            }
        } else {
            *err = err_bad("system", "expected a string or text blocks");
            return RAD_E_INVAL;
        }
        msgs.push_back(json{ { "role", "system" }, { "content", text } });
    }

    auto mit = b.find("messages");
    if (mit == b.end() || !mit->is_array()) {
        *err = err_bad("messages", "is required, as an array");
        return RAD_E_INVAL;
    }
    for (const json& m : *mit) {
        if (!m.is_object()) { *err = err_bad("messages[]", "expected an object"); return RAD_E_INVAL; }
        std::string role;
        if (!member_str(m, "role", "messages[].role", &role, err)) return RAD_E_INVAL;
        if (role != "user" && role != "assistant") {
            *err = err_bad("messages[].role", "must be \"user\" or \"assistant\", got \"" + role + "\"");
            return RAD_E_INVAL;
        }
        const json content = m.contains("content") ? m.at("content") : json();
        if (content.is_string()) {
            msgs.push_back(json{ { "role", role }, { "content", content } });
            continue;
        }
        if (!content.is_array()) {
            *err = err_bad("messages[].content", "expected a string or content blocks");
            return RAD_E_INVAL;
        }
        /* THE TOOL RESULTS GO FIRST. Anthropic carries them in the user turn that follows the
         * calls, beside whatever the user says next; the chat format wants each one as a tool
         * message directly after the assistant turn that made the call, and the user's own words
         * after them. */
        json parts = json::array(), calls = json::array(), results = json::array();
        std::string reasoning;
        for (const json& blk : content) {
            if (!blk.is_object()) {
                *err = err_bad("messages[].content[]", "expected an object");
                return RAD_E_INVAL;
            }
            std::string type;
            if (!member_str(blk, "type", "messages[].content[].type", &type, err)) return RAD_E_INVAL;
            if (type == "text") {
                std::string t;
                if (!member_str(blk, "text", "messages[].content[].text", &t, err)) return RAD_E_INVAL;
                parts.push_back(json{ { "type", "text" }, { "text", t } });
            } else if (type == "image") {
                json part;
                RAD_TRY(anthropic_image(blk, "messages[].content[]", &part, err));
                parts.push_back(std::move(part));
            } else if (type == "thinking") {
                std::string t;
                if (!member_str(blk, "thinking", "messages[].content[].thinking", &t, err))
                    return RAD_E_INVAL;
                reasoning += t;
            } else if (type == "redacted_thinking") {
                /* Encrypted for Anthropic's own models; nothing here can read it. */
            } else if (type == "tool_use") {
                std::string id, name;
                if (!member_str(blk, "id", "messages[].content[].id", &id, err) ||
                    !member_str(blk, "name", "messages[].content[].name", &name, err))
                    return RAD_E_INVAL;
                calls.push_back(json{ { "id", id },
                                      { "type", "function" },
                                      { "function", { { "name", name },
                                                      { "arguments", blk.value("input", json::object()).dump() } } } });
            } else if (type == "tool_result") {
                std::string use_id;
                if (!member_str(blk, "tool_use_id", "messages[].content[].tool_use_id", &use_id, err))
                    return RAD_E_INVAL;
                json rc = blk.contains("content") ? blk.at("content") : json("");
                json tc;
                if (rc.is_string()) {
                    tc = rc;
                } else if (rc.is_array()) {
                    json rp = json::array();
                    bool media = false;
                    std::string flat;
                    for (const json& c : rc) {
                        if (!c.is_object()) {
                            *err = err_bad("messages[].content[].content[]", "expected an object");
                            return RAD_E_INVAL;
                        }
                        std::string ct;
                        if (!member_str(c, "type", "messages[].content[].content[].type", &ct, err))
                            return RAD_E_INVAL;
                        if (ct == "text") {
                            std::string t;
                            if (!member_str(c, "text", "messages[].content[].content[].text", &t, err))
                                return RAD_E_INVAL;
                            flat += t;
                            rp.push_back(json{ { "type", "text" }, { "text", t } });
                        } else if (ct == "image") {
                            json part;
                            RAD_TRY(anthropic_image(c, "messages[].content[].content[]", &part, err));
                            media = true;
                            rp.push_back(std::move(part));
                        } else {
                            *err = err_unsupported("messages[].content[].content", "a tool result "
                                                   "block of type \"" + ct + "\" is not served");
                            return RAD_E_UNSUPPORTED;
                        }
                    }
                    tc = media ? rp : json(flat);
                } else {
                    tc = "";
                }
                results.push_back(json{ { "role", "tool" },
                                        { "tool_call_id", use_id },
                                        { "content", tc } });
            } else {
                /* A block that would change the answer and that this server cannot serve --
                 * documents, server tools' results -- is refused by name, not dropped. */
                *err = err_unsupported("messages[].content[].type",
                                       "a content block of type \"" + type + "\" is not served");
                return RAD_E_UNSUPPORTED;
            }
        }
        for (json& t : results) msgs.push_back(std::move(t));
        if (parts.empty() && calls.empty() && reasoning.empty()) continue;
        json nm = { { "role", role } };
        if (role == "assistant") {
            /* An assistant turn's text is one string, which is what every template renders. */
            std::string text;
            for (const json& p : parts)
                if (p.at("type") == "text") text += p.at("text").get<std::string>();
            nm["content"] = text;
        } else {
            nm["content"] = parts.empty() ? json("") : parts;
        }
        if (!calls.empty()) nm["tool_calls"] = calls;
        if (!reasoning.empty()) nm["reasoning_content"] = reasoning;
        msgs.push_back(std::move(nm));
    }
    o["messages"] = std::move(msgs);

    /* Client tools are functions. A server tool -- Anthropic's own web search or code
     * execution, named by `type` -- runs on Anthropic's side and has no counterpart here. */
    auto tit = b.find("tools");
    if (tit != b.end() && tit->is_array() && !tit->empty()) {
        json tools = json::array();
        for (const json& t : *tit) {
            std::string type = "custom", name;
            if (t.is_object() && (!member_str(t, "type", "tools[].type", &type, err) ||
                                  !member_str(t, "name", "tools[].name", &name, err)))
                return RAD_E_INVAL;
            if (!t.is_object() || (type != "custom" && !t.contains("input_schema"))) {
                *err = err_unsupported("tools", "the server tool \"" + type + "\" runs on "
                                       "Anthropic's side and is not served; send client tools");
                return RAD_E_UNSUPPORTED;
            }
            json fn = { { "name", name },
                        { "parameters", t.contains("input_schema") ? t.at("input_schema") : json::object() } };
            if (t.contains("description")) fn["description"] = t.at("description");
            tools.push_back(json{ { "type", "function" }, { "function", fn } });
        }
        o["tools"] = std::move(tools);
    }
    auto tc = b.find("tool_choice");
    if (tc != b.end() && tc->is_object()) {
        std::string type = "auto", name;
        if (!member_str(*tc, "type", "tool_choice.type", &type, err) ||
            !member_str(*tc, "name", "tool_choice.name", &name, err))
            return RAD_E_INVAL;
        if (type == "auto" || type == "none") o["tool_choice"] = type;
        else if (type == "any") o["tool_choice"] = "required";
        else if (type == "tool")
            o["tool_choice"] = json{ { "type", "function" }, { "function", { { "name", name } } } };
        else {
            *err = err_bad("tool_choice.type", "must be auto, any, tool or none");
            return RAD_E_INVAL;
        }
        bool one = false;
        if (!get_bool(*tc, "disable_parallel_tool_use", &one, err)) return RAD_E_INVAL;
        if (one) o["parallel_tool_calls"] = false;
    }

    /* `model` too, which the reply names back as Anthropic's does. */
    for (const char* k : { "model", "max_tokens", "temperature", "top_p", "top_k", "stream" })
        if (b.contains(k)) o[k] = b.at(k);
    if (b.contains("stop_sequences")) o["stop"] = b.at("stop_sequences");

    /* Extended thinking: on with a budget, off, or on with the template's own judgement. */
    auto th = b.find("thinking");
    if (th != b.end() && th->is_object()) {
        std::string type;
        if (!member_str(*th, "type", "thinking.type", &type, err)) return RAD_E_INVAL;
        if (type == "disabled") {
            o["enable_thinking"] = false;
        } else if (type == "enabled" || type == "adaptive") {
            o["enable_thinking"] = true;
            if (th->contains("budget_tokens")) o["thinking_token_budget"] = th->at("budget_tokens");
        }
    }

    /* The caller's own name for whoever is asking, which Claude Code fills with its session. */
    auto md = b.find("metadata");
    if (label && md != b.end() && md->is_object() && md->contains("user_id") &&
        md->at("user_id").is_string())
        *label = session_label_of(md->at("user_id").get<std::string>());

    *chat = o.dump();
    return RAD_OK;
}

const char* anthropic_stop_reason(const OaiRequest& r, Finish f, const std::string& stop_word) {
    switch (f) {
        case Finish::ToolCalls: return "tool_use";
        case Finish::Length:    return "max_tokens";
        case Finish::StopString: {
            /* The client's own stop sequence is stop_sequence; the chat format's end-of-turn marker
             * is the turn ending, which is end_turn. */
            const auto end = r.stop.begin() + (long)std::min(r.n_client_stops, r.stop.size());
            if (!stop_word.empty() && std::find(r.stop.begin(), end, stop_word) != end)
                return "stop_sequence";
            return "end_turn";
        }
        default: return "end_turn";
    }
}

static json anthropic_usage(const UsageOut& u) {
    return json{ { "input_tokens", u.prompt - u.cached },
                 { "output_tokens", u.completion },
                 { "cache_read_input_tokens", u.cached },
                 { "cache_creation_input_tokens", 0 } };
}

std::string anthropic_message_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u) {
    json blocks = json::array();
    if (!c.msg.reasoning.empty())
        blocks.push_back(json{ { "type", "thinking" }, { "thinking", c.msg.reasoning },
                               { "signature", "" } });
    std::string text = c.msg.content;
    if (!r.continued.empty() && text.rfind(r.continued, 0) == 0) text.erase(0, r.continued.size());
    if (!text.empty()) blocks.push_back(json{ { "type", "text" }, { "text", text } });
    for (const ToolCall& tc : c.msg.tool_calls) {
        json in = json::parse(tc.arguments, nullptr, false);
        if (in.is_discarded() || !in.is_object()) in = json::object();
        blocks.push_back(json{ { "type", "tool_use" }, { "id", tc.id }, { "name", tc.name },
                               { "input", in } });
    }
    const char* sr = anthropic_stop_reason(r, c.finish, c.stop_word);
    json o;
    o["id"] = r.id;
    o["type"] = "message";
    o["role"] = "assistant";
    o["model"] = r.model;
    o["content"] = std::move(blocks);
    o["stop_reason"] = sr;
    o["stop_sequence"] = std::strcmp(sr, "stop_sequence") == 0 ? json(c.stop_word) : json(nullptr);
    o["usage"] = anthropic_usage(u);
    return o.dump();
}

std::string anthropic_error_body(const ApiError& e) {
    const char* type = e.status == 401 ? "authentication_error"
                     : e.status == 403 ? "permission_error"
                     : e.status == 404 ? "not_found_error"
                     : e.status == 413 ? "request_too_large"
                     : e.status == 429 ? "rate_limit_error"
                     : e.status >= 500 ? "api_error"
                                       : "invalid_request_error";
    json o;
    o["type"] = "error";
    o["error"] = json{ { "type", type }, { "message", e.message } };
    return o.dump();
}

AnthropicStream::AnthropicStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : ReplyStream(r, index, parser), r_(&r), skip_(r.continued) {
    for (const PromptInput& p : r.prompts) n_input_ += (int64_t)p.tokens.size();
}

void AnthropicStream::event(std::string& out, const char* type, const json& data) {
    out += "event: ";
    out += type;
    out += "\ndata: ";
    out += data.dump();
    out += "\n\n";
}

std::string AnthropicStream::first_chunk() {
    json m;
    m["id"] = id_;
    m["type"] = "message";
    m["role"] = "assistant";
    m["model"] = model_;
    m["content"] = json::array();
    m["stop_reason"] = nullptr;
    m["stop_sequence"] = nullptr;
    m["usage"] = json{ { "input_tokens", n_input_ }, { "output_tokens", 0 } };
    std::string out;
    event(out, "message_start", json{ { "type", "message_start" }, { "message", m } });
    return out;
}

void AnthropicStream::open(std::string& out, Block b, const ToolCall* tc) {
    close(out);
    json cb;
    if (b == Block::Thinking)  cb = { { "type", "thinking" }, { "thinking", "" } };
    else if (b == Block::Text) cb = { { "type", "text" }, { "text", "" } };
    else cb = { { "type", "tool_use" }, { "id", tc->id }, { "name", tc->name },
                { "input", json::object() } };
    event(out, "content_block_start",
          json{ { "type", "content_block_start" }, { "index", n_blocks_ }, { "content_block", cb } });
    ++n_blocks_;
    block_ = b;
}

void AnthropicStream::close(std::string& out) {
    if (block_ == Block::None) return;
    /* A thinking block ends with its signature, which a client hands back with the block on the
     * next turn. Anthropic's is cryptographic and verified by Anthropic; a local model has none,
     * and the empty one is what llama-server sends. */
    if (block_ == Block::Thinking)
        event(out, "content_block_delta",
              json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                    { "delta", { { "type", "signature_delta" }, { "signature", "" } } } });
    event(out, "content_block_stop", json{ { "type", "content_block_stop" }, { "index", n_blocks_ - 1 } });
    block_ = Block::None;
}

std::string AnthropicStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        if (!absorb(d)) continue;
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            if (block_ != Block::Thinking) open(out, Block::Thinking, nullptr);
            event(out, "content_block_delta",
                  json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                        { "delta", { { "type", "thinking_delta" }, { "thinking", d.text } } } });
            break;
        case ReplyDelta::Content: {
            /* The prefill is the caller's own text; Anthropic's API answers what follows it. */
            std::string_view text = d.text;
            if (!skip_.empty()) {
                const size_t n = std::min(skip_.size(), text.size());
                if (skip_.compare(0, n, text.substr(0, n)) == 0) {
                    skip_.erase(0, n);
                    text.remove_prefix(n);
                } else {
                    skip_.clear();
                }
            }
            if (text.empty()) break;
            if (block_ != Block::Text) open(out, Block::Text, nullptr);
            event(out, "content_block_delta",
                  json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                        { "delta", { { "type", "text_delta" }, { "text", std::string(text) } } } });
            break;
        }
        case ReplyDelta::CallBegin:
            open(out, Block::Tool, &last_.tool_calls.back());
            break;
        case ReplyDelta::CallArgs:
            event(out, "content_block_delta",
                  json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                        { "delta", { { "type", "input_json_delta" }, { "partial_json", d.text } } } });
            break;
        case ReplyDelta::CallEnd:
            close(out);
            break;
        }
    }
    return out;
}

std::string AnthropicStream::final_chunk(Finish f, const TimingsOut&, const UsageOut& u) {
    std::string out = push(std::string_view(), true);
    close(out);
    const char* sr = anthropic_stop_reason(*r_, adjusted_finish(f), stop_word_);
    json delta = { { "stop_reason", sr },
                   { "stop_sequence", std::strcmp(sr, "stop_sequence") == 0 ? json(stop_word_)
                                                                             : json(nullptr) } };
    event(out, "message_delta",
          json{ { "type", "message_delta" }, { "delta", delta }, { "usage", anthropic_usage(u) } });
    event(out, "message_stop", json{ { "type", "message_stop" } });
    return out;
}

std::string AnthropicStream::error_chunk(const ApiError& e, const TimingsOut&, const UsageOut&) {
    return "event: error\ndata: " + anthropic_error_body(e) + "\n\n";
}

/* ================================================================== OpenAI's Responses API */

namespace {

/* An input content part as the chat path's: text, or an image as an image_url. */
int responses_part(const json& p, json* out, ApiError* err) {
    if (!p.is_object()) { *err = err_bad("input[].content[]", "expected an object"); return RAD_E_INVAL; }
    std::string type;
    if (!member_str(p, "type", "input[].content[].type", &type, err)) return RAD_E_INVAL;
    if (type == "input_text" || type == "output_text" || type == "text") {
        std::string text;
        if (!member_str(p, "text", "input[].content[].text", &text, err)) return RAD_E_INVAL;
        *out = json{ { "type", "text" }, { "text", text } };
        return RAD_OK;
    }
    if (type == "input_image") {
        const json u = p.contains("image_url") ? p.at("image_url") : json();
        std::string url;
        if (u.is_string()) url = u.get<std::string>();
        else if (u.is_object() && !member_str(u, "url", "input[].content[].image_url.url", &url, err))
            return RAD_E_INVAL;
        *out = json{ { "type", "image_url" }, { "image_url", { { "url", url } } } };
        return RAD_OK;
    }
    *err = err_unsupported("input[].content[].type",
                           "a content part of type \"" + type + "\" is not served");
    return RAD_E_UNSUPPORTED;
}

/* The id a response item carries for a call, from the call id this server minted ("call_..."). */
std::string fc_id(const std::string& call_id) {
    return "fc_" + (call_id.rfind("call_", 0) == 0 ? call_id.substr(5) : call_id);
}

json response_usage(const UsageOut& u) {
    return json{ { "input_tokens", u.prompt },
                 { "input_tokens_details", { { "cached_tokens", u.cached } } },
                 { "output_tokens", u.completion },
                 { "output_tokens_details", { { "reasoning_tokens", std::max<int64_t>(u.reasoning, 0) } } },
                 { "total_tokens", u.prompt + u.completion } };
}

json reasoning_item(const std::string& id, const std::string& text, const char* status) {
    return json{ { "id", id }, { "type", "reasoning" }, { "summary", json::array() },
                 { "content", json::array({ json{ { "type", "reasoning_text" }, { "text", text } } }) },
                 { "status", status } };
}
json text_part(const std::string& text) {
    return json{ { "type", "output_text" }, { "text", text }, { "annotations", json::array() },
                 { "logprobs", json::array() } };
}
json message_item(const std::string& id, const std::string& text, const char* status) {
    return json{ { "id", id }, { "type", "message" }, { "role", "assistant" }, { "status", status },
                 { "content", text.empty() && std::strcmp(status, "in_progress") == 0
                                  ? json::array() : json::array({ text_part(text) }) } };
}
json call_item(const ToolCall& tc, const char* status) {
    return json{ { "id", fc_id(tc.id) }, { "type", "function_call" }, { "call_id", tc.id },
                 { "name", tc.name }, { "arguments", tc.arguments }, { "status", status } };
}

/* The response object, around its output items, with the request fields it echoes. */
json response_object(const OaiRequest& r, const char* status, const json& output,
                     const UsageOut* u, const char* incomplete) {
    json o = r.echo_json.empty() ? json::object() : json::parse(r.echo_json, nullptr, false);
    if (!o.is_object()) o = json::object();
    o["id"] = r.id;
    o["object"] = "response";
    o["created_at"] = r.created;
    o["status"] = status;
    o["model"] = r.model;
    o["output"] = output;
    o["error"] = nullptr;
    o["incomplete_details"] = incomplete ? json{ { "reason", incomplete } } : json(nullptr);
    if (u) o["usage"] = response_usage(*u);
    else o["usage"] = nullptr;
    return o;
}

}  /* namespace */

int responses_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* echo, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    auto prev = b.find("previous_response_id");
    if (prev != b.end() && prev->is_string() && !prev->get<std::string>().empty()) {
        *err = err_unsupported("previous_response_id", "responses are not stored on this server; "
                               "send the conversation as `input`");
        return RAD_E_UNSUPPORTED;
    }
    /* WHAT A RESPONSE OBJECT CANNOT CARRY is refused, not computed and dropped: several choices
     * (a response is one), and logprobs, which the Responses API reports per output_text part --
     * a split of the reply's tokens this server does not make. */
    {
        int64_t n = 1;
        if (!get_int(b, "n", &n, err)) return RAD_E_INVAL;
        if (n != 1) {
            *err = err_unsupported("n", "a response is one reply; send one request a reply");
            return RAD_E_UNSUPPORTED;
        }
        bool lp = false;
        int64_t top = 0, np = -1;
        if (!get_bool(b, "logprobs", &lp, err) || !get_int(b, "top_logprobs", &top, err) ||
            !get_int(b, "n_probs", &np, err))
            return RAD_E_INVAL;
        const char* asked = lp ? "logprobs" : top > 0 ? "top_logprobs" : np > 0 ? "n_probs" : nullptr;
        auto inc = b.find("include");
        if (inc != b.end() && inc->is_array())
            for (const json& i : *inc)
                if (i == "message.output_text.logprobs") asked = "include";
        if (asked) {
            *err = err_unsupported(asked, "logprobs are not served on /v1/responses; "
                                          "/v1/chat/completions serves them");
            return RAD_E_UNSUPPORTED;
        }
    }
    /* EVERY FIELD THE RESPONSES API DOES NOT DEFINE PASSES THROUGH to the chat request, as
     * llama-server passes it: a client that adds chat_template_kwargs or top_k to a Responses
     * request means them. The Responses API's own fields are translated below, and the chat
     * stream's own extras -- the usage chunk and the prompt progress chunks, which a Responses
     * stream has no frame for -- are dropped: response.completed carries the usage. */
    json o = b;
    for (const char* k : { "input", "instructions", "tools", "tool_choice", "max_output_tokens",
                           "reasoning", "text", "previous_response_id", "store", "metadata",
                           "include", "truncation", "background", "conversation", "prompt",
                           "stream_options", "return_progress", "logprobs", "top_logprobs",
                           "n_probs" })
        o.erase(k);
    json msgs = json::array();
    if (b.contains("instructions") && b.at("instructions").is_string())
        msgs.push_back(json{ { "role", "system" }, { "content", b.at("instructions") } });

    auto in = b.find("input");
    if (in == b.end()) { *err = err_bad("input", "is required"); return RAD_E_INVAL; }
    auto last_assistant = [&]() -> json* {
        return !msgs.empty() && msgs.back().at("role") == "assistant" ? &msgs.back() : nullptr;
    };
    if (in->is_string()) {
        msgs.push_back(json{ { "role", "user" }, { "content", *in } });
    } else if (in->is_array()) {
        for (const json& item : *in) {
            if (!item.is_object()) { *err = err_bad("input[]", "expected an object"); return RAD_E_INVAL; }
            std::string type = item.contains("role") ? "message" : "";
            if (!member_str(item, "type", "input[].type", &type, err)) return RAD_E_INVAL;
            if (type == "message") {
                std::string role;
                if (!member_str(item, "role", "input[].role", &role, err)) return RAD_E_INVAL;
                const json content = item.contains("content") ? item.at("content") : json("");
                if (role == "assistant") {
                    /* A turn the model gave earlier: its text, joined to a reasoning item or a
                     * call of the same turn that came before it. */
                    std::string text;
                    if (content.is_string()) {
                        text = content.get<std::string>();
                    } else if (content.is_array()) {
                        for (const json& p : content) {
                            std::string t;
                            if (!member_str(p, "text", "input[].content[].text", &t, err))
                                return RAD_E_INVAL;
                            text += t;
                        }
                    }
                    if (json* prev = last_assistant()) {
                        (*prev)["content"] = prev->at("content").get<std::string>() + text;
                    } else {
                        msgs.push_back(json{ { "role", "assistant" }, { "content", text } });
                    }
                    continue;
                }
                if (role != "user" && role != "system" && role != "developer") {
                    *err = err_bad("input[].role", "must be user, system, developer or assistant");
                    return RAD_E_INVAL;
                }
                json parts = json::array();
                if (content.is_string()) {
                    parts = content;
                } else if (content.is_array()) {
                    for (const json& p : content) {
                        json part;
                        RAD_TRY(responses_part(p, &part, err));
                        parts.push_back(std::move(part));
                    }
                }
                msgs.push_back(json{ { "role", role }, { "content", parts } });
            } else if (type == "function_call") {
                std::string id, name, args = "{}";
                if (!member_str(item, "call_id", "input[].call_id", &id, err) ||
                    !member_str(item, "name", "input[].name", &name, err) ||
                    !member_str(item, "arguments", "input[].arguments", &args, err))
                    return RAD_E_INVAL;
                json call = { { "id", id }, { "type", "function" },
                              { "function", { { "name", name }, { "arguments", args } } } };
                json* prev = last_assistant();
                if (!prev) {
                    msgs.push_back(json{ { "role", "assistant" }, { "content", "" } });
                    prev = &msgs.back();
                }
                (*prev)["tool_calls"].push_back(std::move(call));
            } else if (type == "function_call_output") {
                const json out = item.contains("output") ? item.at("output") : json("");
                json content;
                if (out.is_string()) {
                    content = out;
                } else if (out.is_array()) {
                    content = json::array();
                    for (const json& p : out) {
                        json part;
                        RAD_TRY(responses_part(p, &part, err));
                        content.push_back(std::move(part));
                    }
                } else {
                    content = "";
                }
                std::string id;
                if (!member_str(item, "call_id", "input[].call_id", &id, err)) return RAD_E_INVAL;
                msgs.push_back(json{ { "role", "tool" }, { "tool_call_id", id }, { "content", content } });
            } else if (type == "reasoning") {
                /* The reasoning of a turn the model gave earlier, which a template that keeps
                 * reasoning across turns renders. Its text is the content, else the summary. */
                std::string text;
                for (const char* k : { "content", "summary" }) {
                    if (!text.empty() || !item.contains(k) || !item.at(k).is_array()) continue;
                    for (const json& p : item.at(k)) {
                        std::string t;
                        if (!member_str(p, "text", "input[].content[].text", &t, err)) return RAD_E_INVAL;
                        text += t;
                    }
                }
                msgs.push_back(json{ { "role", "assistant" }, { "content", "" },
                                     { "reasoning_content", text } });
            } else {
                *err = err_unsupported("input[].type",
                                       "an input item of type \"" + type + "\" is not served");
                return RAD_E_UNSUPPORTED;
            }
        }
    } else {
        *err = err_bad("input", "expected a string or an array of items");
        return RAD_E_INVAL;
    }
    o["messages"] = std::move(msgs);

    /* Function tools only. Built-in tools (web search, file search, computer use) and freeform
     * custom tools have no chat-completions counterpart; the model is offered what it can call. */
    json echo_tools = json::array();
    auto tit = b.find("tools");
    if (tit != b.end() && tit->is_array()) {
        json tools = json::array();
        for (const json& t : *tit) {
            if (str_or(t, "type") != "function") {
                RAD_DEBUG("responses: tool of type \"%s\" left out: no chat equivalent",
                          str_or(t, "type", "?").c_str());
                continue;
            }
            std::string name;
            if (!member_str(t, "name", "tools[].name", &name, err)) return RAD_E_INVAL;
            json fn = { { "name", name },
                        { "parameters", t.contains("parameters") ? t.at("parameters") : json::object() } };
            if (t.contains("description")) fn["description"] = t.at("description");
            tools.push_back(json{ { "type", "function" }, { "function", fn } });
            echo_tools.push_back(t);
        }
        if (!tools.empty()) o["tools"] = std::move(tools);
    }
    auto tc = b.find("tool_choice");
    if (tc != b.end() && !tc->is_null()) {
        std::string type = "auto", name;
        if (tc->is_object() && (!member_str(*tc, "type", "tool_choice.type", &type, err) ||
                                !member_str(*tc, "name", "tool_choice.name", &name, err)))
            return RAD_E_INVAL;
        if (!tc->is_object()) o["tool_choice"] = *tc;
        else if (type == "function")
            o["tool_choice"] = json{ { "type", "function" }, { "function", { { "name", name } } } };
        else o["tool_choice"] = type;
    }
    if (b.contains("max_output_tokens")) o["max_tokens"] = b.at("max_output_tokens");
    auto re = b.find("reasoning");
    if (re != b.end() && re->is_object() && re->contains("effort") && re->at("effort").is_string())
        o["reasoning_effort"] = re->at("effort");
    /* Structured output: text.format is response_format under another name. */
    auto tx = b.find("text");
    if (tx != b.end() && tx->is_object() && tx->contains("format") && tx->at("format").is_object()) {
        const json& f = tx->at("format");
        std::string type = "text", name = "schema";
        if (!member_str(f, "type", "text.format.type", &type, err) ||
            !member_str(f, "name", "text.format.name", &name, err))
            return RAD_E_INVAL;
        if (type == "json_schema")
            o["response_format"] = json{ { "type", "json_schema" },
                                          { "json_schema", { { "name", name },
                                                             { "schema", f.contains("schema") ? f.at("schema") : json::object() } } } };
        else if (type == "json_object")
            o["response_format"] = json{ { "type", "json_object" } };
    }
    *chat = o.dump();

    json e;
    e["instructions"] = b.contains("instructions") ? b.at("instructions") : json(nullptr);
    e["tools"] = echo_tools;
    e["tool_choice"] = b.contains("tool_choice") ? b.at("tool_choice") : json("auto");
    e["parallel_tool_calls"] = !b.contains("parallel_tool_calls") || b.at("parallel_tool_calls") != false;
    e["temperature"] = b.contains("temperature") ? b.at("temperature") : json(nullptr);
    e["top_p"] = b.contains("top_p") ? b.at("top_p") : json(nullptr);
    e["max_output_tokens"] = b.contains("max_output_tokens") ? b.at("max_output_tokens") : json(nullptr);
    e["text"] = b.contains("text") ? b.at("text") : json{ { "format", { { "type", "text" } } } };
    e["reasoning"] = b.contains("reasoning") ? b.at("reasoning")
                                             : json{ { "effort", nullptr }, { "summary", nullptr } };
    e["metadata"] = b.contains("metadata") ? b.at("metadata") : json::object();
    e["previous_response_id"] = nullptr;
    e["store"] = false;
    e["truncation"] = "disabled";
    if (echo) *echo = e.dump();
    return RAD_OK;
}

std::string responses_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u) {
    json out = json::array();
    if (!c.msg.reasoning.empty())
        out.push_back(reasoning_item(random_id("rs_"), c.msg.reasoning, "completed"));
    if (!c.msg.content.empty())
        out.push_back(message_item(random_id("msg_"), c.msg.content, "completed"));
    for (const ToolCall& tc : c.msg.tool_calls) out.push_back(call_item(tc, "completed"));
    const bool cut = c.finish == Finish::Length;
    return response_object(r, cut ? "incomplete" : "completed", out, &u,
                           cut ? "max_output_tokens" : nullptr).dump();
}

ResponsesStream::ResponsesStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : ReplyStream(r, index, parser), r_(&r) {}

void ResponsesStream::event(std::string& out, const char* type, json data) {
    data["type"] = type;
    data["sequence_number"] = seq_++;
    out += "event: ";
    out += type;
    out += "\ndata: ";
    out += data.dump();
    out += "\n\n";
}

json ResponsesStream::response(const char* status) const {
    const bool done = std::strcmp(status, "in_progress") != 0;
    json items = json::array();
    for (const std::string& it : items_) items.push_back(json::parse(it));
    return response_object(*r_, status, items, done ? &usage_ : nullptr, incomplete_);
}

std::string ResponsesStream::first_chunk() {
    std::string out;
    event(out, "response.created", json{ { "response", response("in_progress") } });
    event(out, "response.in_progress", json{ { "response", response("in_progress") } });
    return out;
}

void ResponsesStream::open(std::string& out, Item k, size_t call) {
    close(out);
    item_ = k;
    text_.clear();
    const int64_t at = (int64_t)items_.size();
    if (k == Item::Reasoning) {
        item_id_ = random_id("rs_");
        json it = reasoning_item(item_id_, "", "in_progress");
        it["content"] = json::array();
        event(out, "response.output_item.added", json{ { "output_index", at }, { "item", it } });
        event(out, "response.content_part.added",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", { { "type", "reasoning_text" }, { "text", "" } } } });
    } else if (k == Item::Message) {
        item_id_ = random_id("msg_");
        event(out, "response.output_item.added",
              json{ { "output_index", at }, { "item", message_item(item_id_, "", "in_progress") } });
        event(out, "response.content_part.added",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", text_part("") } });
    } else {
        call_ = call;
        const ToolCall& tc = last_.tool_calls[call];
        item_id_ = fc_id(tc.id);
        event(out, "response.output_item.added",
              json{ { "output_index", at }, { "item", call_item(tc, "in_progress") } });
    }
}

void ResponsesStream::close(std::string& out) {
    if (item_ == Item::None) return;
    const int64_t at = (int64_t)items_.size();
    json item;
    if (item_ == Item::Reasoning) {
        event(out, "response.reasoning_text.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "text", text_ } });
        event(out, "response.content_part.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", { { "type", "reasoning_text" }, { "text", text_ } } } });
        item = reasoning_item(item_id_, text_, "completed");
    } else if (item_ == Item::Message) {
        event(out, "response.output_text.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "text", text_ }, { "logprobs", json::array() } });
        event(out, "response.content_part.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", text_part(text_) } });
        item = message_item(item_id_, text_, "completed");
    } else {
        const ToolCall& tc = last_.tool_calls[call_];
        event(out, "response.function_call_arguments.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "arguments", tc.arguments } });
        item = call_item(tc, "completed");
    }
    event(out, "response.output_item.done", json{ { "output_index", at }, { "item", item } });
    items_.push_back(item.dump());
    item_ = Item::None;
}

std::string ResponsesStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        if (!absorb(d)) continue;
        const int64_t at = (int64_t)items_.size();
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            if (item_ != Item::Reasoning) open(out, Item::Reasoning, 0);
            text_ += d.text;
            event(out, "response.reasoning_text.delta",
                  json{ { "item_id", item_id_ }, { "output_index", (int64_t)items_.size() },
                        { "content_index", 0 }, { "delta", d.text } });
            break;
        case ReplyDelta::Content:
            if (item_ != Item::Message) open(out, Item::Message, 0);
            text_ += d.text;
            event(out, "response.output_text.delta",
                  json{ { "item_id", item_id_ }, { "output_index", (int64_t)items_.size() },
                        { "content_index", 0 }, { "delta", d.text }, { "logprobs", json::array() } });
            break;
        case ReplyDelta::CallBegin:
            open(out, Item::Call, last_.tool_calls.size() - 1);
            break;
        case ReplyDelta::CallArgs:
            event(out, "response.function_call_arguments.delta",
                  json{ { "item_id", item_id_ }, { "output_index", at }, { "delta", d.text } });
            break;
        case ReplyDelta::CallEnd:
            close(out);
            break;
        }
    }
    return out;
}

std::string ResponsesStream::final_chunk(Finish f, const TimingsOut&, const UsageOut& u) {
    std::string out = push(std::string_view(), true);
    close(out);
    usage_ = u;
    const bool cut = adjusted_finish(f) == Finish::Length;
    incomplete_ = cut ? "max_output_tokens" : nullptr;
    event(out, cut ? "response.incomplete" : "response.completed",
          json{ { "response", response(cut ? "incomplete" : "completed") } });
    return out;
}

std::string ResponsesStream::error_chunk(const ApiError& e, const TimingsOut&, const UsageOut& u) {
    std::string out;
    usage_ = u;
    json resp = response("failed");
    resp["error"] = json{ { "code", "server_error" }, { "message", e.message } };
    event(out, "response.failed", json{ { "response", std::move(resp) } });
    return out;
}

}  /* namespace server */
}  /* namespace rad */
