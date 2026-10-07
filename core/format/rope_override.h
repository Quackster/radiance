/* --rope-scaling, over a container's free-form metadata.
 *
 * Static YaRN is read by the architecture (arch/common/rad_arch.h, yarn_from) out of the model's
 * own keys: `text_config.rope_parameters.`, `rope_parameters.`, `text_config.rope_scaling.`,
 * `rope_scaling.`, in that order, the FIRST of them that names a rotary type being the one that
 * counts; a DFlash2 drafter reads `draft.rope_parameters.` / `draft.rope_scaling.` on its own. A
 * container converted without YaRN therefore needs a reconvert to get it -- or these keys rewritten
 * when it is loaded, which is all this does. The architecture then resolves the result exactly as
 * it would a converted container, so there is one reader and one set of checks.
 *
 * The rewrite goes into the object the reader would take, not beside it. A Qwen3.8 container says
 * `text_config.rope_parameters.rope_type = default`; a YaRN written under `rope_parameters.` would
 * sit behind that and never be read.
 *
 * Pure functions over two parallel string vectors, so they are tested without an engine
 * (tests/yarn_test.cpp holds them to yarn_from). */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

struct RopeSpec {
    enum Kind { Keep, None, Yarn } kind = Keep;
    double  factor = 0.0;
    int64_t orig = 0;   /* 0: the container's own trained length */
};

/* "yarn:FACTOR[:ORIG]" with FACTOR > 1 and ORIG > 0, or "none"; empty is Keep. False for anything
 * else, so a typo is refused where it was typed. */
inline bool rope_spec_parse(const char* s, RopeSpec* out) {
    RopeSpec r;
    if (!s || !*s) { *out = r; return true; }
    if (!std::strcmp(s, "none")) { r.kind = RopeSpec::None; *out = r; return true; }
    if (std::strncmp(s, "yarn:", 5)) return false;
    char* e = nullptr;
    r.factor = std::strtod(s + 5, &e);
    if (e == s + 5 || !(r.factor > 1.0) || r.factor > 1e6) return false;
    if (*e == ':') {
        char* e2 = nullptr;
        const long long o = std::strtoll(e + 1, &e2, 10);
        if (e2 == e + 1 || *e2 || o <= 0) return false;
        r.orig = o;
    } else if (*e) {
        return false;
    }
    r.kind = RopeSpec::Yarn;
    *out = r;
    return true;
}

struct RopeKeys {
    std::vector<std::string>& k;
    std::vector<std::string>& v;

    int find(const std::string& key) const {
        for (size_t i = 0; i < k.size(); ++i)
            if (k[i] == key) return (int)i;
        return -1;
    }
    const std::string* get(const std::string& key) const {
        const int i = find(key);
        return i < 0 ? nullptr : &v[(size_t)i];
    }
    void set(const std::string& key, const std::string& val) {
        const int i = find(key);
        if (i >= 0) { v[(size_t)i] = val; return; }
        k.push_back(key);
        v.push_back(val);
    }
    void erase(const std::string& key) {
        const int i = find(key);
        if (i < 0) return;
        k.erase(k.begin() + i);
        v.erase(v.begin() + i);
    }
    bool any_prefix(const char* pre) const {
        const size_t n = std::strlen(pre);
        for (const auto& s : k)
            if (!s.compare(0, n, pre)) return true;
        return false;
    }
};

/* The reader's order -- yarn_from's, kept in step with it. */
inline std::vector<std::string> rope_homes(const char* group) {
    if (!*group)
        return { "text_config.rope_parameters.", "rope_parameters.", "text_config.rope_scaling.",
                 "rope_scaling." };
    return { std::string(group) + "rope_parameters.", std::string(group) + "rope_scaling." };
}

/* The object the reader takes and the key that names its type, or empty strings when no object
 * names one (the model's rotary is then the default). */
inline void rope_authoritative(const RopeKeys& kv, const char* group, std::string* home,
                               std::string* type_key) {
    home->clear();
    type_key->clear();
    for (const auto& h : rope_homes(group))
        for (const char* t : { "rope_type", "type" })
            if (kv.get(h + t)) { *home = h; *type_key = h + t; return; }
}

struct RopeResolved {
    std::string type;      /* "" when nothing names one */
    double      factor = 1.0;
    double      orig = 0.0;
};

inline RopeResolved rope_resolve(std::vector<std::string>& k, std::vector<std::string>& v,
                                 const char* group) {
    RopeKeys kv{ k, v };
    RopeResolved r;
    std::string home, tk;
    rope_authoritative(kv, group, &home, &tk);
    if (home.empty()) return r;
    r.type = *kv.get(tk);
    if (const std::string* f = kv.get(home + "factor")) r.factor = std::strtod(f->c_str(), nullptr);
    if (const std::string* o = kv.get(home + "original_max_position_embeddings"))
        r.orig = std::strtod(o->c_str(), nullptr);
    return r;
}

/* Apply SPEC to one rotary group ("" the target, "draft." a DFlash2 drafter). `trained` is the
 * group's own trained length, the default ORIG. Returns false, with *why set, when the group's
 * rotary is one this flag does not stack YaRN over (linear position interpolation, say). */
inline bool rope_override_group(std::vector<std::string>& k, std::vector<std::string>& v,
                                const RopeSpec& spec, const char* group, int64_t trained,
                                std::string* why) {
    if (spec.kind == RopeSpec::Keep) return true;
    RopeKeys kv{ k, v };
    std::string home, tk;
    rope_authoritative(kv, group, &home, &tk);
    const std::string type = home.empty() ? std::string() : *kv.get(tk);
    if (spec.kind == RopeSpec::None) {
        if (type == "yarn") kv.set(tk, "default");
        return true;
    }
    if (!type.empty() && type != "default" && type != "yarn") {
        *why = "the " + std::string(*group ? "drafter's" : "model's") + " rotary is '" + type +
               "' (" + tk + "); YaRN over it is not --rope-scaling's to decide";
        return false;
    }
    if (home.empty()) {
        home = *group ? std::string(group) + "rope_parameters."
             : kv.any_prefix("text_config.") ? "text_config.rope_parameters."
                                             : "rope_parameters.";
        tk = home + "rope_type";
    }
    /* A checkpoint's own YaRN keeps its trained length unless one is given; anything that was
     * tied to its old factor goes, so the reader derives it for the new one. */
    int64_t orig = spec.orig;
    if (!orig && type == "yarn")
        if (const std::string* o = kv.get(home + "original_max_position_embeddings"))
            orig = std::atoll(o->c_str());
    if (!orig) orig = trained;
    if (orig <= 0) {
        *why = "no trained length to scale from; give one: yarn:FACTOR:ORIG";
        return false;
    }
    char buf[64];
    kv.set(tk, "yarn");
    std::snprintf(buf, sizeof buf, "%.17g", spec.factor);
    kv.set(home + "factor", buf);
    std::snprintf(buf, sizeof buf, "%lld", (long long)orig);
    kv.set(home + "original_max_position_embeddings", buf);
    for (const char* stale : { "attention_factor", "mscale", "mscale_all_dim" })
        kv.erase(home + stale);
    return true;
}

}  /* namespace rad */
