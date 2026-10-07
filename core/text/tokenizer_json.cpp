/* tokenizer_json.cpp -- reading a vocab in, and writing it into the .rad.
 *
 * Three sources and one sink:
 *   parse_tokenizer_json()  the HuggingFace file, interpreted rather than hashed. This is the
 *                           preferred path and the one spec §12 is about.
 *   vocab_from_gguf()       GGUF metadata, where the chain has already been lost and has to be
 *                           reconstructed from `tokenizer.ggml.pre`, a name llama.cpp derived by
 *                           hashing the original tokenizer.json.
 *   vocab_serialize()       into the .rad's vocab section, and back.
 */
#include "text/tokenizer.h"
#include "text/tokenizer_regex.h"
#include "text/gguf.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <new>
#include <sstream>

namespace rad {

using json = nlohmann::json;

namespace {

/* HuggingFace's SplitDelimiterBehavior, in the order our iarg encodes. */
int behavior_from_name(const std::string& s) {
    if (s == "Removed")            return 0;
    if (s == "Isolated")           return 1;
    if (s == "MergedWithPrevious") return 2;
    if (s == "MergedWithNext")     return 3;
    if (s == "Contiguous")         return 4;
    return 1;   /* Isolated is the default in every tokenizer.json that omits it */
}

int prepend_scheme_from_name(const std::string& s) {
    if (s == "never")  return 0;
    if (s == "first")  return 1;
    if (s == "always") return 2;
    return 2;
}

const json* get(const json& o, const char* k) {
    if (!o.is_object()) return nullptr;
    auto it = o.find(k);
    return it == o.end() || it->is_null() ? nullptr : &*it;
}

std::string str_or(const json& o, const char* k, const char* dflt) {
    const json* v = get(o, k);
    return v && v->is_string() ? v->get<std::string>() : std::string(dflt);
}

bool bool_or(const json& o, const char* k, bool dflt) {
    const json* v = get(o, k);
    return v && v->is_boolean() ? v->get<bool>() : dflt;
}

/* A regex a step carries is compiled HERE as well as when the vocab loads, so that rad-convert
 * refuses a pattern outside the compiler's dialect with the construct named, instead of writing a
 * container no engine will load. */
int check_regex(const std::string& pat, const char* what, const char* origin) {
    std::unique_ptr<TokRegex> re;
    std::string why;
    if (TokRegex::compile(pat, &re, &why) < 0) {
        RAD_ERR("%s: the %s pattern '%s' is outside the supported regex dialect: %s", origin, what,
                pat.c_str(), why.c_str());
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* HuggingFace's pattern field is either {"String": "..."} (a literal) or {"Regex": "..."}. */
bool read_pattern(const json& o, std::string* pat, bool* is_regex) {
    const json* p = get(o, "pattern");
    if (!p) return false;
    if (p->is_string()) { *pat = p->get<std::string>(); *is_regex = false; return true; }
    const json* s = get(*p, "String");
    if (s && s->is_string()) { *pat = s->get<std::string>(); *is_regex = false; return true; }
    const json* r = get(*p, "Regex");
    if (r && r->is_string()) { *pat = r->get<std::string>(); *is_regex = true;  return true; }
    return false;
}

/* A token id out of the file, which indexes the token table as soon as it is read: a negative
 * one, or one that is not an integer, writes outside it, and a huge one sizes it. `limit` is
 * one past the largest id accepted (see parse_tokenizer_json_buf). */
bool read_id(const json& v, uint64_t limit, int32_t* out) {
    if (!v.is_number_integer()) return false;
    if (!v.is_number_unsigned() && v.get<int64_t>() < 0) return false;
    const uint64_t id = v.is_number_unsigned() ? v.get<uint64_t>() : (uint64_t)v.get<int64_t>();
    if (id >= limit || id > (uint64_t)INT32_MAX) return false;
    *out = (int32_t)id;
    return true;
}

/* --- the three chains ---------------------------------------------------------------------- */

int read_normalizer(const json& n, std::vector<VocabStep>& out, const char* origin);
int read_pretok(const json& n, std::vector<VocabStep>& out, const char* origin);
int read_decoder(const json& n, std::vector<VocabStep>& out, const char* origin);

int read_normalizer(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "normalizers");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_normalizer(c, out, origin));
        return RAD_OK;
    }
    if (t == "NFC")  { s.kind = RAD_NORM_NFC;  out.push_back(s); return RAD_OK; }
    if (t == "NFKC") { s.kind = RAD_NORM_NFKC; out.push_back(s); return RAD_OK; }
    if (t == "NFD")  { s.kind = RAD_NORM_NFD;  out.push_back(s); return RAD_OK; }
    if (t == "NFKD") { s.kind = RAD_NORM_NFKD; out.push_back(s); return RAD_OK; }
    if (t == "Lowercase") { s.kind = RAD_NORM_LOWERCASE; out.push_back(s); return RAD_OK; }
    if (t == "Strip") {
        s.kind  = RAD_NORM_STRIP;
        s.flags = (bool_or(n, "strip_left", true) ? 1u : 0u) |
                  (bool_or(n, "strip_right", true) ? 2u : 0u);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Prepend") {
        s.kind = RAD_NORM_PREPEND;
        s.arg0 = str_or(n, "prepend", "");
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Replace") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Replace normalizer has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        s.kind  = RAD_NORM_REPLACE;
        s.flags = is_regex ? 1u : 0u;
        s.arg1  = str_or(n, "content", "");
        if (is_regex) RAD_TRY(check_regex(s.arg0, "Replace", origin));
        out.push_back(s);
        return RAD_OK;
    }
    /* Everything else is refused BY NAME rather than ignored. An ignored normaliser is a
     * tokeniser that runs, produces different boundaries, and says nothing -- exactly the
     * failure this component exists to remove (spec §12). */
    RAD_ERR("%s: normalizer '%s' has no RadVocabStep. Refusing rather than approximating it; "
            "add a RAD_NORM_* for it if a model needs it.", origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

int read_pretok(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "pretokenizers");
        if (!seq) seq = get(n, "pre_tokenizers");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_pretok(c, out, origin));
        return RAD_OK;
    }
    if (t == "ByteLevel") {
        s.kind  = RAD_PRE_BYTELEVEL;
        s.flags = (bool_or(n, "add_prefix_space", false) ? 1u : 0u) |
                  (bool_or(n, "use_regex", true) ? 2u : 0u);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Split") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Split pre-tokenizer has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        if (!is_regex) {
            /* A literal delimiter is a regex with every metacharacter escaped. Doing the escape
             * at convert time keeps the runtime with one kind of Split instead of two. */
            std::string q;
            for (char c : s.arg0) {
                if (std::strchr("\\^$.|?*+()[]{}", c)) q += '\\';
                q += c;
            }
            s.arg0.swap(q);
        }
        s.kind  = RAD_PRE_SPLIT;
        s.flags = bool_or(n, "invert", false) ? 1u : 0u;
        s.iarg  = behavior_from_name(str_or(n, "behavior", "Isolated"));
        RAD_TRY(check_regex(s.arg0, "Split", origin));
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Metaspace") {
        s.kind  = RAD_PRE_METASPACE;
        s.arg0  = str_or(n, "replacement", "\xe2\x96\x81");
        s.iarg  = prepend_scheme_from_name(str_or(n, "prepend_scheme", "always"));
        s.flags = bool_or(n, "split", true) ? 1u : 0u;
        /* The pre-0.14 spelling: add_prefix_space instead of prepend_scheme. */
        if (!get(n, "prepend_scheme")) s.iarg = bool_or(n, "add_prefix_space", true) ? 2 : 0;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Whitespace" || t == "WhitespaceSplit" || t == "BertPreTokenizer") {
        s.kind = RAD_PRE_WHITESPACE;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Punctuation") {
        s.kind = RAD_PRE_PUNCTUATION;
        s.iarg = behavior_from_name(str_or(n, "behavior", "Isolated"));
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Digits") {
        s.kind  = RAD_PRE_DIGITS;
        s.flags = bool_or(n, "individual_digits", false) ? 1u : 0u;
        out.push_back(s);
        return RAD_OK;
    }
    RAD_ERR("%s: pre_tokenizer '%s' has no RadVocabStep. Refusing rather than falling back to a "
            "default split -- a default split is llama.cpp's failure mode, not ours.",
            origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

int read_decoder(const json& n, std::vector<VocabStep>& out, const char* origin) {
    if (n.is_null()) return RAD_OK;
    const std::string t = str_or(n, "type", "");
    VocabStep s;
    if (t == "Sequence") {
        const json* seq = get(n, "decoders");
        if (!seq || !seq->is_array()) return RAD_OK;
        for (const json& c : *seq) RAD_TRY(read_decoder(c, out, origin));
        return RAD_OK;
    }
    if (t == "ByteLevel")    { s.kind = RAD_DEC_BYTELEVEL; out.push_back(s); return RAD_OK; }
    if (t == "Fuse")         { s.kind = RAD_DEC_FUSE; out.push_back(s); return RAD_OK; }
    if (t == "ByteFallback") { s.kind = RAD_DEC_BYTE_FALLBACK; out.push_back(s); return RAD_OK; }
    if (t == "Metaspace") {
        s.kind = RAD_DEC_METASPACE;
        s.arg0 = str_or(n, "replacement", "\xe2\x96\x81");
        s.iarg = prepend_scheme_from_name(str_or(n, "prepend_scheme", "always"));
        if (!get(n, "prepend_scheme")) s.iarg = bool_or(n, "add_prefix_space", true) ? 2 : 0;
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Replace") {
        bool is_regex = false;
        if (!read_pattern(n, &s.arg0, &is_regex)) {
            RAD_ERR("%s: Replace decoder has no readable pattern", origin);
            return RAD_E_FORMAT;
        }
        if (is_regex) {
            RAD_ERR("%s: a regex Replace decoder is not implemented; the decoder runs per token "
                    "on the streaming path and a regex there is a per-token allocation", origin);
            return RAD_E_UNSUPPORTED;
        }
        s.kind = RAD_DEC_REPLACE;
        s.arg1 = str_or(n, "content", "");
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "Strip") {
        s.kind = RAD_DEC_STRIP;
        s.arg0 = str_or(n, "content", " ");
        const json* a = get(n, "start");
        const json* b = get(n, "stop");
        const int64_t st = a && a->is_number() ? a->get<int64_t>() : 0;
        const int64_t sp = b && b->is_number() ? b->get<int64_t>() : 0;
        s.iarg = (sp << 32) | (st & 0xFFFFFFFF);
        out.push_back(s);
        return RAD_OK;
    }
    if (t == "WordPiece") {
        /* WordPiece's decoder is a Replace of the continuation prefix plus a leading strip.
         * Expressing it as the two steps it is keeps the runtime free of a per-model branch. */
        VocabStep r;
        r.kind = RAD_DEC_REPLACE;
        r.arg0 = str_or(n, "prefix", "##");
        r.arg1 = "";
        out.push_back(r);
        return RAD_OK;
    }
    if (t == "BPEDecoder") {
        VocabStep r;
        r.kind = RAD_DEC_REPLACE;
        r.arg0 = str_or(n, "suffix", "</w>");
        r.arg1 = " ";
        out.push_back(r);
        return RAD_OK;
    }
    RAD_ERR("%s: decoder '%s' has no RadVocabStep", origin, t.c_str());
    return RAD_E_UNSUPPORTED;
}

/* --- the model section --------------------------------------------------------------------- */

int read_model(const json& m, VocabBuild& out, const char* origin, uint64_t id_limit) {
    const std::string type = str_or(m, "type", "");
    const json* vocab = get(m, "vocab");
    if (!vocab) {
        RAD_ERR("%s: model has no vocab", origin);
        return RAD_E_FORMAT;
    }

    /* {text: id}, the BPE and WordPiece shape. The ids are dense, but nothing guarantees the
     * JSON order matches the id order, so each is placed by id -- which is why every one is
     * checked before the table is sized from them. */
    auto read_text_to_id = [&](const char* kind) -> int {
        if (!vocab->is_object()) {
            RAD_ERR("%s: %s vocab is not an object of {text: id}", origin, kind);
            return RAD_E_FORMAT;
        }
        int32_t max_id = -1;
        for (auto it = vocab->begin(); it != vocab->end(); ++it) {
            int32_t id = 0;
            if (!read_id(it.value(), id_limit, &id)) {
                RAD_ERR("%s: %s vocab entry '%s' has id %.64s, which is not a token id",
                        origin, kind, it.key().c_str(), it.value().dump().c_str());
                return RAD_E_FORMAT;
            }
            max_id = std::max(max_id, id);
        }
        out.tokens.assign((size_t)max_id + 1, std::string());
        for (auto it = vocab->begin(); it != vocab->end(); ++it) {
            int32_t id = 0;
            read_id(it.value(), id_limit, &id);
            out.tokens[(size_t)id] = it.key();
        }
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
        return RAD_OK;
    };

    if (type == "BPE") {
        out.kind = RAD_TOK_BPE;
        out.ignore_merges = bool_or(m, "ignore_merges", false);
        RAD_TRY(read_text_to_id("BPE"));
    } else if (type == "Unigram") {
        out.kind = RAD_TOK_UNIGRAM;
        if (!vocab->is_array()) {
            RAD_ERR("%s: Unigram vocab is not an array", origin);
            return RAD_E_FORMAT;
        }
        out.tokens.reserve(vocab->size());
        out.scores.reserve(vocab->size());
        for (const json& e : *vocab) {
            if (!e.is_array() || e.size() != 2 || !e[0].is_string() || !e[1].is_number()) {
                RAD_ERR("%s: Unigram vocab entry is not [text, score]", origin);
                return RAD_E_FORMAT;
            }
            out.tokens.push_back(e[0].get<std::string>());
            out.scores.push_back(e[1].get<float>());
        }
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
        /* The encoder emits this id for text no piece covers, so one that names no token would
         * put an id outside the vocabulary into the output. */
        const json* u = get(m, "unk_id");
        if (u && !read_id(*u, out.tokens.size(), &out.unk)) {
            RAD_ERR("%s: Unigram unk_id %.64s is not an id in its %zu-token vocab", origin,
                    u->dump().c_str(), out.tokens.size());
            return RAD_E_FORMAT;
        }
    } else if (type == "WordPiece") {
        out.kind = RAD_TOK_WORDPIECE;
        RAD_TRY(read_text_to_id("WordPiece"));
    } else {
        RAD_ERR("%s: model type '%s' is not implemented (BPE, Unigram and WordPiece are)",
                origin, type.c_str());
        return RAD_E_UNSUPPORTED;
    }

    /* Byte fallback is a property of the MODEL in tokenizer.json and a step for us, because it
     * changes what the encoder does on a miss and the encoder is where the miss happens. */
    if (bool_or(m, "byte_fallback", false)) {
        VocabStep s;
        s.kind = RAD_PRE_BYTE_FALLBACK;
        out.steps.push_back(s);
    }

    if (type != "BPE") return RAD_OK;

    /* Merges. Two spellings in the wild: ["a b", ...] and [["a","b"], ...]. The second exists
     * because a token can contain a space. */
    const json* merges = get(m, "merges");
    if (!merges || !merges->is_array()) return RAD_OK;

    std::unordered_map<std::string, uint32_t> id_of;
    id_of.reserve(out.tokens.size() * 2);
    for (size_t i = 0; i < out.tokens.size(); ++i) id_of.emplace(out.tokens[i], (uint32_t)i);

    size_t dropped = 0;
    out.merges.reserve(merges->size());
    for (const json& e : *merges) {
        std::string l, r;
        if (e.is_array() && e.size() == 2 && e[0].is_string() && e[1].is_string()) {
            l = e[0].get<std::string>();
            r = e[1].get<std::string>();
        } else if (e.is_string()) {
            const std::string s = e.get<std::string>();
            const size_t sp = s.find(' ');
            if (sp == std::string::npos) { ++dropped; continue; }
            l = s.substr(0, sp);
            r = s.substr(sp + 1);
        } else {
            ++dropped;
            continue;
        }
        auto li = id_of.find(l), ri = id_of.find(r);
        if (li == id_of.end() || ri == id_of.end()) { ++dropped; continue; }
        out.merges.emplace_back(li->second, ri->second);
    }
    if (dropped) {
        /* Every dropped merge is a merge the tokeniser will not make, so the boundary moves.
         * It is a warning rather than an error only because a handful of vocabs genuinely ship
         * merges over strings they never added as tokens. */
        RAD_WARN("%s: %zu of %zu merges reference text that is not in the vocab and were "
                 "dropped; token boundaries may differ from the reference tokeniser",
                 origin, dropped, merges->size());
    }
    return RAD_OK;
}

/* Added tokens override the model vocab and may extend past it. `special` decides whether the
 * pre-splitter treats them as control (isolated even when parse_special is off is NOT the rule --
 * see Tokenizer::encode) or as user-defined. */
int read_added(const json& arr, VocabBuild& out, const char* origin, uint64_t id_limit) {
    if (!arr.is_array()) return RAD_OK;
    for (const json& e : arr) {
        const json* idv = get(e, "id");
        const json* cv  = get(e, "content");
        if (!idv || !cv) continue;
        int32_t sid = 0;
        if (!read_id(*idv, id_limit, &sid) || !cv->is_string()) {
            RAD_ERR("%s: added token %.64s is not {id, content} with an id and a string",
                    origin, e.dump().c_str());
            return RAD_E_FORMAT;
        }
        const size_t id = (size_t)sid;
        if (id >= out.tokens.size()) {
            out.tokens.resize(id + 1);
            out.types.resize(id + 1, (uint8_t)RAD_TT_NORMAL);
            if (out.kind == RAD_TOK_UNIGRAM) out.scores.resize(id + 1, 0.0f);
        }
        out.tokens[id] = cv->get<std::string>();
        out.types[id]  = bool_or(e, "special", false) ? (uint8_t)RAD_TT_CONTROL
                                                      : (uint8_t)RAD_TT_USER_DEFINED;
    }
    return RAD_OK;
}

/* tokenizer_config.json is where the ids live. tokenizer.json holds the vocab and the chain but
 * not which token is BOS, so a convert that reads only tokenizer.json produces a vocab that
 * cannot start a sequence. Reading the sibling is opportunistic: absent, the ids stay -1 and
 * rad-convert can fill them from elsewhere. */
void read_config_json(const std::string& dir, VocabBuild& out) {
    const std::string path = dir + "tokenizer_config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_INFO("tokenizer: no %s; bos/eos ids and the chat template are unset", path.c_str());
        return;
    }
    json c;
    try { f >> c; }
    catch (const std::exception& e) {
        RAD_WARN("tokenizer: %s did not parse (%s); ids left unset", path.c_str(), e.what());
        return;
    }
    std::unordered_map<std::string, int32_t> id_of;
    for (size_t i = 0; i < out.tokens.size(); ++i) id_of.emplace(out.tokens[i], (int32_t)i);

    auto token_id = [&](const char* key) -> int32_t {
        const json* v = get(c, key);
        if (!v) return -1;
        std::string text;
        const json* t = v->is_string() ? v : get(*v, "content");
        if (!t || !t->is_string()) return -1;
        text = t->get<std::string>();
        auto it = id_of.find(text);
        return it == id_of.end() ? -1 : it->second;
    };
    out.bos = token_id("bos_token");
    out.eos = token_id("eos_token");
    out.pad = token_id("pad_token");
    out.unk = out.unk >= 0 ? out.unk : token_id("unk_token");
    out.sep = token_id("sep_token");
    out.eot = out.eos;
    out.add_bos = bool_or(c, "add_bos_token", false);
    out.add_eos = bool_or(c, "add_eos_token", false);

    const json* ct = get(c, "chat_template");
    if (ct && ct->is_string()) out.chat_template = ct->get<std::string>();
    else if (ct && ct->is_array() && !ct->empty()) {
        /* The multi-template form: a list of {name, template}. The one named "default" wins,
         * else the first, which is what every renderer does. */
        for (const json& e : *ct) {
            if (str_or(e, "name", "") == "default") { out.chat_template = str_or(e, "template", ""); break; }
        }
        if (out.chat_template.empty()) out.chat_template = str_or((*ct)[0], "template", "");
    }
}


/* THE TEMPLATE MAY NOT BE IN tokenizer_config.json AT ALL. Recent transformers versions write it
 * to a standalone chat_template.jinja and omit the key, so a checkpoint saved by one of those
 * converts with NO template at all unless this sibling is read -- and the only symptom is the
 * server answering /v1/chat/completions with 501, a long way from the cause.
 *
 * The file wins where it exists, which is what transformers itself does; in a checkpoint that
 * carries both, the config key is the superseded copy. */
void read_chat_template_file(const std::string& dir, VocabBuild& out) {
    const std::string path = dir + "chat_template.jinja";
    std::ifstream f(path);
    if (!f) return;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string t = ss.str();
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    if (t.empty()) return;
    RAD_INFO("tokenizer: chat template from %s (%zu bytes)%s", path.c_str(), t.size(),
             out.chat_template.empty() ? "" : ", overriding the tokenizer_config.json key");
    out.chat_template = std::move(t);
}

void read_config(const std::string& dir, VocabBuild& out) {
    read_config_json(dir, out);
    read_chat_template_file(dir, out);
}

}  /* namespace */

/* ================================================================== parse_tokenizer_json */

namespace {

int interpret_tokenizer_json(const json& j, uint64_t id_limit, const char* origin,
                             VocabBuild& out) {
    const json* model = get(j, "model");
    if (!model) {
        RAD_ERR("%s: no \"model\" section", origin);
        return RAD_E_FORMAT;
    }
    RAD_TRY(read_model(*model, out, origin, id_limit));

    /* Order matters and it is the file's order: normaliser, then pre-tokeniser, then decoder.
     * They land in one list because that is what RadVocabHeader holds, and the kind ranges keep
     * them separable. read_model may already have pushed a ByteFallback step, which belongs
     * with the pre-tokenisers; it is re-sorted below. */
    std::vector<VocabStep> pre_from_model;
    pre_from_model.swap(out.steps);

    std::vector<VocabStep> norm, pre, dec;
    if (const json* n = get(j, "normalizer"))     RAD_TRY(read_normalizer(*n, norm, origin));
    if (const json* p = get(j, "pre_tokenizer"))  RAD_TRY(read_pretok(*p, pre, origin));
    if (const json* d = get(j, "decoder"))        RAD_TRY(read_decoder(*d, dec, origin));
    for (VocabStep& s : pre_from_model) pre.push_back(std::move(s));

    out.steps.clear();
    out.steps.insert(out.steps.end(), norm.begin(), norm.end());
    out.steps.insert(out.steps.end(), pre.begin(),  pre.end());
    out.steps.insert(out.steps.end(), dec.begin(),  dec.end());

    if (const json* a = get(j, "added_tokens")) RAD_TRY(read_added(*a, out, origin, id_limit));

    if (pre.empty() && out.kind == RAD_TOK_BPE) {
        RAD_ERR("%s: a BPE model with no pre_tokenizer would tokenise the whole prompt as one "
                "word. Refusing.", origin);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

}  /* namespace */

int parse_tokenizer_json_buf(std::string_view text, const char* origin, VocabBuild& out) {
    json j;
    try {
        j = json::parse(text.begin(), text.end());
    } catch (const std::exception& e) {
        RAD_ERR("%s: not valid JSON: %s", origin, e.what());
        return RAD_E_FORMAT;
    }
    out = VocabBuild{};
    out.source = origin;

    /* NO ID PAST THE SIZE OF THE FILE. Ids are dense, so a vocabulary that names one beyond the
     * number of bytes listing it would leave more ids unused than the file has bytes; bounding
     * them by it is what bounds the token table they size to a multiple of the input, whatever
     * the ids claim. */
    const uint64_t id_limit = text.size();

    /* A value of the wrong JSON type throws out of nlohmann on the first get<>() that meets it.
     * The reader checks the shapes that decide what is written where; this turns any other one
     * into the refusal it is rather than into std::terminate in the converter. */
    try {
        return interpret_tokenizer_json(j, id_limit, origin, out);
    } catch (const json::exception& e) {
        RAD_ERR("%s: a value has the wrong type: %s", origin, e.what());
        return RAD_E_FORMAT;
    } catch (const std::bad_alloc&) {
        RAD_ERR("%s: out of memory reading the vocabulary", origin);
        return RAD_E_NOMEM;
    }
}

int parse_tokenizer_json(const char* path, VocabBuild& out) {
    if (!path) return RAD_E_INVAL;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        RAD_ERR("tokenizer: cannot open %s", path);
        return RAD_E_IO;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    RAD_TRY(parse_tokenizer_json_buf(text, path, out));

    /* The sibling config carries the ids and the chat template. */
    const std::string p(path);
    const size_t slash = p.find_last_of('/');
    read_config(slash == std::string::npos ? std::string() : p.substr(0, slash + 1), out);

    RAD_INFO("tokenizer: %s -> %zu tokens, %zu merges, %zu chain steps",
             path, out.tokens.size(), out.merges.size(), out.steps.size());
    for (const VocabStep& s : out.steps) RAD_INFO("tokenizer:   %s", step_describe(s).c_str());
    return RAD_OK;
}

/* ================================================================== vocab_from_gguf */

namespace {

/* `tokenizer.ggml.pre` -> the pre-tokeniser regex list.
 *
 * These are llama.cpp's ADAPTED patterns, copied verbatim from llm_tokenizer_bpe's constructor
 * (06938ac12). Copying rather than deriving is deliberate: the adaptation llama.cpp applied is
 * part of the pattern's identity for a GGUF -- gpt-4o's `((?=[\p{L}])([^a-z]))*` is not o200k's
 * `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*`, and a GGUF tokenises the way llama.cpp runs it. They
 * go through the same compiler as a tokenizer.json pattern.
 *
 * The one exception is kimi-k2. llama.cpp spells it `\p{Han}+` and uses that string only as the
 * key of a hand-written splitter approximating Kimi-K2's real pattern; as a regex it would split
 * Han runs and nothing else. The entry is the real pattern, from the model's tokenization_kimi.py,
 * which is what that splitter was written to stand in for.
 *
 * The list is not exhaustive and does not try to be. A GGUF whose `pre` is not here is REFUSED,
 * with the instruction to convert from tokenizer.json instead -- which is the path that does not
 * depend on this table at all. */
struct PreEntry { const char* name; const char* regex[4]; bool byte_encode; };

const PreEntry k_pre[] = {
    { "default", {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]" }, true },
    { "llama3", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "llama-bpe", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "llama-v3",  { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "falcon3",   { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "pixtral",   { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "lfm2",      { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "dbrx",      { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "smaug-bpe", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "glm4",      { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "chatglm-bpe", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },

    { "gpt-2", { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "phi-2", { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "mpt",   { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "olmo",  { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "jais",  { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "trillion", { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "granite-docling", { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },

    { "starcoder", { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "refact",    { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "command-r", { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "smollm",    { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "codeshell", { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "exaone",    { "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },
    { "minerva-7b",{ "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" }, true },

    { "qwen2",            { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "stablelm2",        { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "deepseek-r1-qwen", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "hunyuan",          { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "solar-open",       { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "grok-2",           { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },

    { "qwen35",           { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },

    { "gpt-4o",     { "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "llama4",     { "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "tekken",     { "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },

    { "seed-coder", { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "kimi-k2",    { "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" }, true },
    { "whitespace", { "\\S+" }, false },
    { "gemma4",     { "[^\\n]+|[\\n]+" }, false },
    { "sarvam-moe", { "[^\\n]+|[\\n]+" }, false },
};

}  /* namespace */

int vocab_from_gguf(const GgufFile& g, VocabBuild& out) {
    out = VocabBuild{};
    out.source = g.path();

    std::string model;
    if (!g.get_str("tokenizer.ggml.model", &model)) {
        RAD_ERR("gguf: %s has no tokenizer.ggml.model", g.path().c_str());
        return RAD_E_FORMAT;
    }
    if (!g.get_strs("tokenizer.ggml.tokens", &out.tokens) || out.tokens.empty()) {
        RAD_ERR("gguf: %s has no tokenizer.ggml.tokens", g.path().c_str());
        return RAD_E_FORMAT;
    }

    std::vector<int64_t> types;
    if (g.get_i64s("tokenizer.ggml.token_type", &types) && types.size() == out.tokens.size()) {
        out.types.resize(types.size());
        for (size_t i = 0; i < types.size(); ++i) out.types[i] = (uint8_t)types[i];
    } else {
        out.types.assign(out.tokens.size(), (uint8_t)RAD_TT_NORMAL);
    }
    g.get_f32s("tokenizer.ggml.scores", &out.scores);

    auto id = [&](const char* k, int32_t* dst) {
        int64_t v = 0;
        if (g.get_i64(k, &v)) *dst = (int32_t)v;
    };
    id("tokenizer.ggml.bos_token_id", &out.bos);
    id("tokenizer.ggml.eos_token_id", &out.eos);
    id("tokenizer.ggml.eot_token_id", &out.eot);
    id("tokenizer.ggml.padding_token_id", &out.pad);
    id("tokenizer.ggml.unknown_token_id", &out.unk);
    id("tokenizer.ggml.seperator_token_id", &out.sep);
    if (out.eot < 0) out.eot = out.eos;
    int64_t b = 0;
    if (g.get_i64("tokenizer.ggml.add_bos_token", &b)) out.add_bos = b != 0;
    if (g.get_i64("tokenizer.ggml.add_eos_token", &b)) out.add_eos = b != 0;
    g.get_str("tokenizer.chat_template", &out.chat_template);

    if (model == "gpt2") {
        out.kind = RAD_TOK_BPE;
        std::string pre;
        if (!g.get_str("tokenizer.ggml.pre", &pre)) pre = "default";

        const PreEntry* e = nullptr;
        for (const PreEntry& p : k_pre) if (pre == p.name) { e = &p; break; }
        if (!e) {
            /* This is the refusal spec §12 asks for. llama.cpp would fall back to `default` here
             * and produce plausible text with wrong boundaries; we would rather not run. */
            RAD_ERR("gguf: tokenizer.ggml.pre = '%s' is not in this build's table, and guessing "
                    "would give a silently different split. Convert from the model's "
                    "tokenizer.json instead, which needs no table.", pre.c_str());
            return RAD_E_UNSUPPORTED;
        }
        for (const char* r : e->regex) {
            if (!r) break;
            VocabStep s;
            s.kind = RAD_PRE_SPLIT;
            s.arg0 = r;
            s.iarg = 1;   /* Isolated */
            out.steps.push_back(s);
        }
        if (e->byte_encode) {
            VocabStep s;
            s.kind  = RAD_PRE_BYTELEVEL;
            s.flags = 0;   /* the split above already carries the regex */
            out.steps.push_back(s);
            VocabStep d;
            d.kind = RAD_DEC_BYTELEVEL;
            out.steps.push_back(d);
        } else {
            /* SPM-shaped BPE: the vocab is raw UTF-8 with U+2581 for space. */
            VocabStep s;
            s.kind = RAD_PRE_METASPACE;
            s.arg0 = "\xe2\x96\x81";
            s.iarg = 2;
            out.steps.push_back(s);
            VocabStep d;
            d.kind = RAD_DEC_METASPACE;
            d.arg0 = "\xe2\x96\x81";
            d.iarg = 2;
            out.steps.push_back(d);
        }

        std::vector<std::string> merges;
        g.get_strs("tokenizer.ggml.merges", &merges);
        std::unordered_map<std::string, uint32_t> id_of;
        id_of.reserve(out.tokens.size() * 2);
        for (size_t i = 0; i < out.tokens.size(); ++i) id_of.emplace(out.tokens[i], (uint32_t)i);

        size_t dropped = 0;
        out.merges.reserve(merges.size());
        for (const std::string& m : merges) {
            const size_t sp = m.find(' ');
            if (sp == std::string::npos) { ++dropped; continue; }
            auto l = id_of.find(m.substr(0, sp));
            auto r = id_of.find(m.substr(sp + 1));
            if (l == id_of.end() || r == id_of.end()) { ++dropped; continue; }
            out.merges.emplace_back(l->second, r->second);
        }
        if (dropped)
            RAD_WARN("gguf: %zu of %zu merges dropped (a half is not a token); token boundaries "
                     "may differ from the reference tokeniser", dropped, merges.size());
    } else if (model == "llama") {
        out.kind = RAD_TOK_BPE;
        /* SPM proper: whitespace escaped to U+2581, byte fallback for the rest. Expressed as
         * steps so the runtime has no per-model branch. */
        VocabStep m;
        m.kind = RAD_PRE_METASPACE;
        m.arg0 = "\xe2\x96\x81";
        m.iarg = 2;
        out.steps.push_back(m);
        VocabStep bf;
        bf.kind = RAD_PRE_BYTE_FALLBACK;
        out.steps.push_back(bf);
        VocabStep d;
        d.kind = RAD_DEC_METASPACE;
        d.arg0 = "\xe2\x96\x81";
        d.iarg = 2;
        out.steps.push_back(d);
        RAD_WARN("gguf: '%s' is a SentencePiece vocab and this build tokenises it as BPE over the "
                 "same pieces. That is right for the vocabs that ship as GGUF today and it is "
                 "not the SPM bigram algorithm; convert from tokenizer.json to be sure.",
                 model.c_str());
    } else if (model == "t5") {
        out.kind = RAD_TOK_UNIGRAM;
        VocabStep m;
        m.kind = RAD_PRE_METASPACE;
        m.arg0 = "\xe2\x96\x81";
        m.iarg = 2;
        out.steps.push_back(m);
        VocabStep d;
        d.kind = RAD_DEC_METASPACE;
        d.arg0 = "\xe2\x96\x81";
        d.iarg = 2;
        out.steps.push_back(d);
    } else if (model == "bert") {
        out.kind = RAD_TOK_WORDPIECE;
        VocabStep n;
        n.kind = RAD_NORM_LOWERCASE;
        out.steps.push_back(n);
        VocabStep w;
        w.kind = RAD_PRE_WHITESPACE;
        out.steps.push_back(w);
        VocabStep p;
        p.kind = RAD_PRE_PUNCTUATION;
        p.iarg = 1;
        out.steps.push_back(p);
        VocabStep d;
        d.kind = RAD_DEC_REPLACE;
        d.arg0 = "##";
        out.steps.push_back(d);
    } else {
        RAD_ERR("gguf: tokenizer.ggml.model = '%s' is not implemented", model.c_str());
        return RAD_E_UNSUPPORTED;
    }

    RAD_INFO("gguf: vocab %zu tokens, %zu merges, %zu chain steps",
             out.tokens.size(), out.merges.size(), out.steps.size());
    for (const VocabStep& s : out.steps) RAD_INFO("gguf:   %s", step_describe(s).c_str());
    return RAD_OK;
}

/* ================================================================== serialise */

namespace {
template <class T> void put(std::vector<uint8_t>& b, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void pad_to(std::vector<uint8_t>& b, size_t base, size_t align) {
    while ((base + b.size()) % align) b.push_back(0);
}
}  /* namespace */

int vocab_serialize(const VocabBuild& in, StringSink& strings, uint64_t off,
                    std::vector<uint8_t>& out) {
    if (in.tokens.empty()) return RAD_E_INVAL;
    if (in.tokens.size() > UINT32_MAX || in.merges.size() > UINT32_MAX ||
        in.steps.size() > UINT32_MAX) return RAD_E_INVAL;

    /* Every string goes into the file's one blob and only its offset is stored here, per
     * rad_format.h. That is what makes the section fixed-stride and mmap-walkable. */
    std::vector<rad_stroff> tok_off(in.tokens.size());
    for (size_t i = 0; i < in.tokens.size(); ++i) tok_off[i] = strings.intern(in.tokens[i]);

    std::vector<RadVocabStep> steps(in.steps.size());
    for (size_t i = 0; i < in.steps.size(); ++i) {
        steps[i].kind  = in.steps[i].kind;
        steps[i].flags = in.steps[i].flags;
        steps[i].arg0  = strings.intern(in.steps[i].arg0);
        steps[i].arg1  = strings.intern(in.steps[i].arg1);
        steps[i].iarg  = in.steps[i].iarg;
    }
    const rad_stroff tmpl = strings.intern(in.chat_template);

    out.clear();
    RadVocabHeader h{};
    h.kind     = in.kind;
    h.n_tokens = (uint32_t)in.tokens.size();
    h.n_merges = (uint32_t)in.merges.size();
    h.n_steps  = (uint32_t)in.steps.size();
    h.bos = in.bos; h.eos = in.eos; h.eot = in.eot;
    h.pad_id = in.pad; h.unk = in.unk; h.sep = in.sep;
    h.add_bos = in.add_bos ? 1u : 0u;
    h.add_eos = in.add_eos ? 1u : 0u;
    h.chat_template = tmpl;
    h.model_flags = in.ignore_merges ? RAD_VOCAB_F_IGNORE_MERGES : 0;
    out.resize(sizeof(RadVocabHeader), 0);

    /* Each array is aligned to its own element so a mapped reader can point a typed pointer at
     * it. RAD_ALIGN_SUB would be enough for the container's rules; 8 is what the types need. */
    auto place = [&](size_t align, const void* p, size_t bytes) -> uint64_t {
        pad_to(out, (size_t)off, align);
        const uint64_t at = off + out.size();
        const uint8_t* s = static_cast<const uint8_t*>(p);
        out.insert(out.end(), s, s + bytes);
        return at;
    };

    h.tok_text_off = place(8, tok_off.data(), tok_off.size() * sizeof(rad_stroff));
    if (!in.scores.empty())
        h.tok_score_off = place(4, in.scores.data(), in.scores.size() * sizeof(float));
    if (!in.types.empty())
        h.tok_type_off = place(1, in.types.data(), in.types.size());
    if (!in.merges.empty()) {
        std::vector<uint32_t> flat(in.merges.size() * 2);
        for (size_t i = 0; i < in.merges.size(); ++i) {
            flat[i * 2 + 0] = in.merges[i].first;
            flat[i * 2 + 1] = in.merges[i].second;
        }
        h.merge_off = place(4, flat.data(), flat.size() * sizeof(uint32_t));
    }
    if (!steps.empty())
        h.step_off = place(8, steps.data(), steps.size() * sizeof(RadVocabStep));

    std::memcpy(out.data(), &h, sizeof h);
    return RAD_OK;
}

int vocab_deserialize(const void* file_base, uint64_t file_bytes, const RadVocabHeader* h,
                      VocabBuild& out) {
    if (!file_base || !h) return RAD_E_INVAL;
    const uint8_t* base = static_cast<const uint8_t*>(file_base);
    const char* strs = nullptr;
    uint64_t str_bytes = 0;
    {
        /* The string blob's extent comes from the file header, which sits at offset 0. Reading
         * it here rather than taking it as an argument keeps the caller from having to keep the
         * two in sync. */
        if (file_bytes < sizeof(RadFileHeader)) return RAD_E_FORMAT;
        const RadFileHeader* fh = reinterpret_cast<const RadFileHeader*>(base);
        if (fh->str_off + fh->str_bytes > file_bytes) return RAD_E_FORMAT;
        strs = reinterpret_cast<const char*>(base + fh->str_off);
        str_bytes = fh->str_bytes;
    }
    auto S = [&](rad_stroff o) -> std::string {
        if (o >= str_bytes) return {};
        const char* p = strs + o;
        const size_t max = (size_t)(str_bytes - o);
        return std::string(p, strnlen(p, max));
    };
    auto in_file = [&](uint64_t o, uint64_t n) { return o <= file_bytes && n <= file_bytes - o; };

    out = VocabBuild{};
    out.kind = h->kind;
    out.bos = h->bos; out.eos = h->eos; out.eot = h->eot;
    out.pad = h->pad_id; out.unk = h->unk; out.sep = h->sep;
    out.add_bos = h->add_bos != 0;
    out.add_eos = h->add_eos != 0;
    out.ignore_merges = (h->model_flags & RAD_VOCAB_F_IGNORE_MERGES) != 0;
    out.chat_template = S(h->chat_template);

    if (!in_file(h->tok_text_off, (uint64_t)h->n_tokens * sizeof(rad_stroff)))
        return RAD_E_FORMAT;
    const rad_stroff* toff = reinterpret_cast<const rad_stroff*>(base + h->tok_text_off);
    out.tokens.resize(h->n_tokens);
    for (uint32_t i = 0; i < h->n_tokens; ++i) out.tokens[i] = S(toff[i]);

    if (h->tok_score_off) {
        if (!in_file(h->tok_score_off, (uint64_t)h->n_tokens * sizeof(float))) return RAD_E_FORMAT;
        const float* p = reinterpret_cast<const float*>(base + h->tok_score_off);
        out.scores.assign(p, p + h->n_tokens);
    }
    if (h->tok_type_off) {
        if (!in_file(h->tok_type_off, h->n_tokens)) return RAD_E_FORMAT;
        const uint8_t* p = base + h->tok_type_off;
        out.types.assign(p, p + h->n_tokens);
    }
    if (h->merge_off) {
        if (!in_file(h->merge_off, (uint64_t)h->n_merges * 2 * sizeof(uint32_t)))
            return RAD_E_FORMAT;
        const uint32_t* p = reinterpret_cast<const uint32_t*>(base + h->merge_off);
        out.merges.resize(h->n_merges);
        for (uint32_t i = 0; i < h->n_merges; ++i) out.merges[i] = { p[i * 2], p[i * 2 + 1] };
    }
    if (h->step_off) {
        if (!in_file(h->step_off, (uint64_t)h->n_steps * sizeof(RadVocabStep)))
            return RAD_E_FORMAT;
        const RadVocabStep* p = reinterpret_cast<const RadVocabStep*>(base + h->step_off);
        out.steps.resize(h->n_steps);
        for (uint32_t i = 0; i < h->n_steps; ++i) {
            out.steps[i].kind  = p[i].kind;
            out.steps[i].flags = p[i].flags;
            out.steps[i].arg0  = S(p[i].arg0);
            out.steps[i].arg1  = S(p[i].arg1);
            out.steps[i].iarg  = p[i].iarg;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
