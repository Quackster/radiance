/* checkpoint.cpp -- see checkpoint.h. */
#include "checkpoint.h"
#include "encoding.h"
#include "rad_plugin.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>

namespace rad {

/* ================================================================== metadata */

void MetaOwn::set(const std::string& k, const std::string& v) {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) { vals[i] = v; return; }
    kv(k, v);
}

bool MetaOwn::has(const std::string& k) const {
    return std::find(keys.begin(), keys.end(), k) != keys.end();
}

const char* MetaOwn::get(const std::string& k) const {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) return vals[i].c_str();
    return nullptr;
}

void MetaOwn::finish() {
    kp.clear();
    vp.clear();
    for (auto& s : keys) kp.push_back(s.c_str());
    for (auto& s : vals) vp.push_back(s.c_str());
    m.arch_id = arch.c_str();
    m.name    = name.c_str();
    m.quant   = quant.c_str();
    m.n_kv    = (int)keys.size();
    m.kv_key  = kp.data();
    m.kv_val  = vp.data();
}

/* ---------------------------------------------------------------- config.json
 *
 * A safetensors checkpoint carries its architecture and its dimensions in config.json, and both
 * have to reach the plugin: RadModelMeta's typed fields are what declare sizes everything from, so
 * an unparsed config means a graph built on zeros. Parsing it is the core's job and not the
 * architecture plugin's, because the plugin is handed RadModelMeta and never sees the file.
 *
 * Two shapes exist and both are handled. A text-only model puts its dimensions at the top level; a
 * multimodal one nests them under `text_config` and puts the tower under `vision_config`. The
 * nested case is not a special case here: EVERY key is flattened into the free-form list with a
 * dotted path, and the typed fields are then read from `text_config.X` falling back to `X`. So a
 * plugin that wants `vision_config.depth` finds it under exactly that name whatever the nesting.
 */
static void flatten_json(const nlohmann::json& j, const std::string& prefix, MetaOwn* out) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        const nlohmann::json& v = it.value();
        if (v.is_object())            flatten_json(v, key, out);
        else if (v.is_boolean())      out->kv(key, v.get<bool>() ? "1" : "0");
        else if (v.is_number_integer()) out->kv(key, fmt("%lld", (long long)v.get<int64_t>()));
        else if (v.is_number())       out->kv(key, fmt("%.17g", v.get<double>()));
        else if (v.is_string())       out->kv(key, v.get<std::string>());
        else if (v.is_array()) {
            /* An array of scalars becomes a space-separated string: mrope_section "11 11 10",
             * layer_types "linear_attention full_attention ...". A plugin splits it. Arrays of
             * objects are not flattened -- nothing in a config uses one for a dimension. */
            std::string s;
            bool ok = true;
            for (const auto& e : v) {
                if (!s.empty()) s += ' ';
                if (e.is_number_integer()) s += fmt("%lld", (long long)e.get<int64_t>());
                else if (e.is_number())    s += fmt("%.17g", e.get<double>());
                else if (e.is_string())    s += e.get<std::string>();
                else { ok = false; break; }
            }
            if (ok) out->kv(key, s);
        }
    }
}

/* A SECOND config.json, flattened under a prefix and nothing else.
 *
 * None of meta_from_config_json's typed extraction applies: a drafter's `num_hidden_layers` is 5
 * and the target's is 64, and writing the drafter's into RadModelMeta would size the trunk from
 * the draft head. So the whole of it goes into the free-form table with `draft.` in front, which
 * is where an architecture plugin reads a dimension the struct does not name anyway. The
 * `text_config` aliasing is deliberately not repeated -- a drafter is a text model by
 * construction, and aliasing here would collide with the target's bare keys. */
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A draft checkpoint carries its own geometry there -- the "
                "block size it was trained for, which trunk layers it reads, the selector's rank "
                "-- and none of it is derivable from the target.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- generation_config.json
 *
 * HOW THE CHECKPOINT SAYS IT WANTS TO BE SAMPLED, carried into the container so that serving it
 * does not depend on a file nobody copied. It is a separate file from config.json because it is a
 * separate thing: config.json is the architecture and is required, this is the sampler and is
 * advisory, and a checkpoint that omits it is read without complaint.
 *
 * It matters more than "advisory" suggests. A model tuned against a truncated tail and served
 * without one will occasionally draw a token from far down its distribution, and end-of-turn is
 * one such token -- so the visible symptom is not bad prose, it is a reply that stops early.
 *
 * Flattened under `generation.` rather than merged into the bare keys: `temperature` and `top_k`
 * are common enough words that a bare spelling would collide with an architecture that happens to
 * use one, and the prefix is what tells a reader which file a value came from. */
int meta_merge_generation_config(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/generation_config.json";
    std::ifstream f(path);
    if (!f) return RAD_OK;               /* absent is not an error; nothing is written */
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_WARN("%s: %s; no sampling defaults are carried", path.c_str(), e.what());
        return RAD_OK;
    }
    if (!j.is_object()) {
        RAD_WARN("%s: not a JSON object; no sampling defaults are carried", path.c_str());
        return RAD_OK;
    }
    flatten_json(j, "generation", out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- the preprocessor configs
 *
 * HOW AN IMAGE AND A VIDEO BECOME PATCHES, carried for the reason the sampling defaults are:
 * serving the model must not depend on a file nobody copied. The engine's processor (core/mm)
 * reads the pixel bands, the normalisation and the frame rate from here; the geometry the encoder
 * itself needs is already in config.json's vision_config. Flattened under their own prefixes --
 * `size.shortest_edge` means two different things in the two files. Absent is not an error: a
 * text model has neither. */
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out) {
    static const struct { const char* file; const char* prefix; } kFiles[] = {
        { "preprocessor_config.json", "preprocessor" },
        { "video_preprocessor_config.json", "video_preprocessor" },
    };
    for (const auto& f : kFiles) {
        const std::string path = dir + "/" + f.file;
        std::ifstream in(path);
        if (!in) continue;
        nlohmann::json j;
        try { in >> j; }
        catch (const std::exception& e) {
            RAD_WARN("%s: %s; it is not carried", path.c_str(), e.what());
            continue;
        }
        if (j.is_object()) flatten_json(j, f.prefix, out);
    }
    return RAD_OK;
}

int meta_from_config_json(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A safetensors checkpoint carries its architecture and its "
                "dimensions there; without it nothing can be sized.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_ERR("%s: %s", path.c_str(), e.what());
        return RAD_E_FORMAT;
    }

    flatten_json(j, "", out);

    /* Alias text_config.X to a bare X wherever no bare X already exists.
     *
     * A multimodal checkpoint nests the language model's dimensions under `text_config` and a
     * text-only one does not; that nesting is the ONLY difference between the two files. The
     * struct fields below already prefer the prefixed spelling, but the free-form key/value table
     * is handed to the plugin verbatim, so without this a plugin reading `partial_rotary_factor`
     * finds nothing on exactly the checkpoints that have a vision tower -- and the failure is at
     * declare, after the architecture has already been matched. One rule, applied once here, is
     * better than every plugin having to know what a vision tower does to its config.
     *
     * The bare key wins where both exist: a top-level value is the file's own answer, and the
     * prefixed one is the aliased fallback. */
    {
        const size_t n = out->keys.size();
        for (size_t i = 0; i < n; ++i) {
            const std::string& k = out->keys[i];
            if (k.rfind("text_config.", 0) != 0) continue;
            const std::string bare = k.substr(12);
            bool have = false;
            for (size_t j2 = 0; j2 < n; ++j2) if (out->keys[j2] == bare) { have = true; break; }
            if (!have) out->kv(bare, out->vals[i]);
        }
    }

    auto gs = [&](const char* k) -> const char* { return out->get(k); };
    /* text_config first, then the top level: the nesting is the only difference between a
     * multimodal config and a text one, and every field below appears in exactly one of the two. */
    auto gi = [&](const char* k, int64_t dflt) -> int64_t {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtoll(v, nullptr, 10) : dflt;
    };
    auto gf = [&](const char* k, double dflt) -> double {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtod(v, nullptr) : dflt;
    };

    /* HuggingFace's model_type is the architecture id, minus the underscores its naming convention
     * uses and GGUF's does not: `qwen3_5` and GGUF's `qwen35` are the same architecture, and a
     * plugin should not have to claim both spellings. */
    if (const char* mt = gs("model_type")) {
        out->arch.clear();
        for (const char* p = mt; *p; ++p) if (*p != '_') out->arch += *p;
    }
    if (const char* nm = gs("_name_or_path")) out->name = nm;

    RadModelMeta& m = out->m;
    m.n_layers    = gi("num_hidden_layers", 0);
    m.n_embd      = gi("hidden_size", 0);
    m.n_head      = gi("num_attention_heads", 0);
    m.n_head_kv   = gi("num_key_value_heads", m.n_head);
    /* Read head_dim; do NOT derive it. On qwen35 it is 256 while hidden_size/num_attention_heads
     * is 213, and a derived value makes every downstream shape silently wrong. */
    m.head_dim    = gi("head_dim", m.n_head ? m.n_embd / m.n_head : 0);
    /* ...AND A ROUTED CHECKPOINT HAS NO `intermediate_size` AT ALL. A routed config states
     * `moe_intermediate_size` (one expert's width) and `shared_expert_intermediate_size` and
     * nothing else, so reading only the dense spelling leaves n_ff at zero -- and geom_from refuses
     * a model with a zero feed-forward width by name, which reads as a broken container rather
     * than as an unread key. The per-expert width is what a GEMM sees, so it is what n_ff means;
     * the shared expert's is its own key and the plugin reads it from the free-form table. */
    m.n_ff        = gi("intermediate_size", gi("moe_intermediate_size", 0));
    m.n_vocab     = gi("vocab_size", 0);
    m.n_ctx_train = gi("max_position_embeddings", 0);
    m.n_expert        = gi("num_experts", gi("num_local_experts", 0));
    m.n_expert_used   = gi("num_experts_per_tok", 0);
    m.n_expert_shared = gi("shared_expert_intermediate_size", 0) ? 1 : 0;
    m.rms_eps     = (float)gf("rms_norm_eps", 1e-6);
    m.rope_theta  = (float)gf("rope_parameters.rope_theta", gf("rope_theta", 10000.0));
    /* The rotary's scaling stays in the free-form keys with its type beside it; the architecture
     * reads both (arch/common/rad_arch.h, yarn_from). A factor copied here without its type would
     * be a linear position divide for llama3, longrope or YaRN. */
    m.rope_scale  = 1.0f;

    /* The quantisation descriptor, which with the architecture selects the plugin (spec §2.4).
     * "fp8" plus its format, so qwen35_fp8 and qwen35_bf16 are told apart by the checkpoint
     * rather than by a flag. */
    if (const char* qm = gs("quantization_config.quant_method")) {
        out->quant = qm;
        if (const char* fm = gs("quantization_config.fmt")) { out->quant += "_"; out->quant += fm; }
        /* COMPRESSED-TENSORS NAMES ITS FORMAT, NOT ITS METHOD. `quant_method` is the container
         * library (llm-compressor's) and says nothing about the numbers; `format` does. An
         * OCP-MXFP4 release -- e2m1 codes, one E8M0 exponent a 32 -- is `mxfp4`, the descriptor
         * qwen35_mxfp4 claims; any other format keeps its name so it is refused as itself. */
        if (out->quant == "compressed-tensors") {
            const char* cf = gs("quantization_config.format");
            out->quant = cf && !std::strcmp(cf, "mxfp4-pack-quantized")
                             ? std::string("mxfp4")
                             : "compressed-tensors_" + std::string(cf ? cf : "?");
        }
    }

    out->finish();
    return RAD_OK;
}

int meta_from_gguf(GgufFile& g, MetaOwn* out) {
    if (!g.get_str("general.architecture", &out->arch)) {
        RAD_ERR("the GGUF has no general.architecture, so no architecture plugin can be chosen");
        return RAD_E_FORMAT;
    }
    g.get_str("general.name", &out->name);

    /* Dimensions come from metadata, so one plugin covers every size in a family. The keys are
     * GGUF's `<arch>.<field>` convention; anything the struct does not name is passed through in
     * the free-form KV list, which the core never interprets. */
    const std::string a = out->arch;
    auto gi = [&](const char* suffix, int64_t dflt) -> int64_t {
        int64_t v = 0;
        return g.get_i64(a + "." + suffix, &v) ? v : dflt;
    };
    auto gf = [&](const char* suffix, double dflt) -> double {
        double v = 0;
        return g.get_f64(a + "." + suffix, &v) ? v : dflt;
    };

    RadModelMeta& m = out->m;
    m.n_layers    = gi("block_count", 0);
    m.n_embd      = gi("embedding_length", 0);
    m.n_head      = gi("attention.head_count", 0);
    m.n_head_kv   = gi("attention.head_count_kv", m.n_head);
    m.head_dim    = gi("attention.key_length", m.n_head ? m.n_embd / m.n_head : 0);
    m.n_ff        = gi("feed_forward_length", 0);
    m.n_ctx_train = gi("context_length", 0);
    m.n_expert       = gi("expert_count", 0);
    m.n_expert_used  = gi("expert_used_count", 0);
    m.n_expert_shared= gi("expert_shared_count", 0);
    m.rms_eps     = (float)gf("attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_theta  = (float)gf("rope.freq_base", 10000.0);
    m.rope_scale  = 1.0f;

    /* THE ROTARY'S SCALING, written as the config.json keys a safetensors release carries, because
     * those are what the architecture reads (arch/common/rad_arch.h, yarn_from) -- one reader, so
     * a type is refused or honoured by name whichever file it came from. llama.cpp's YaRN
     * attn_factor, ext_factor and log multiplier modify the magnitude and ramp in ways config.json
     * has no spelling for, so a file that sets one to anything but its neutral value is refused
     * here rather than served as plain YaRN. */
    {
        /* llama.cpp's reading: the type defaults to linear, the factor is rope.scaling.factor or
         * the older rope.scale_linear, and 0 or 1 is no scaling. */
        std::string type = "linear";
        g.get_str(a + ".rope.scaling.type", &type);
        const double factor = gf("rope.scaling.factor", gf("rope.scale_linear", 0.0));
        const bool none = type == "none" || (type == "linear" && (factor == 0.0 || factor == 1.0));
        if (!none) {
            const struct { const char* key; double neutral; } kOff[] = {
                { "rope.scaling.attn_factor", 1.0 }, { "rope.scaling.yarn_attn_factor", 1.0 },
                { "rope.scaling.yarn_log_multiplier", 0.0 } };
            for (const auto& k : kOff)
                if (gf(k.key, k.neutral) != k.neutral) {
                    RAD_ERR("the GGUF sets %s.%s = %g; that modifier of the rotary's scaling is "
                            "not implemented", a.c_str(), k.key, gf(k.key, k.neutral));
                    return RAD_E_UNSUPPORTED;
                }
            const double ext = gf("rope.scaling.yarn_ext_factor", -1.0);
            if (ext >= 0.0 && ext != 1.0) {
                RAD_ERR("the GGUF sets %s.rope.scaling.yarn_ext_factor = %g; YaRN is implemented "
                        "at extrapolation factor 1", a.c_str(), ext);
                return RAD_E_UNSUPPORTED;
            }
            out->kv("rope_scaling.type", type);
            if (factor != 0.0) out->kv("rope_scaling.factor", fmt("%.17g", factor));
            const int64_t orig = gi("rope.scaling.original_context_length", 0);
            if (orig > 0) out->kv("rope_scaling.original_max_position_embeddings",
                                  fmt("%lld", (long long)orig));
            for (const char* beta : { "beta_fast", "beta_slow" }) {
                const double v = gf((std::string("rope.scaling.yarn_") + beta).c_str(), 0.0);
                if (v != 0.0) out->kv(std::string("rope_scaling.") + beta, fmt("%.17g", v));
            }
        }
    }

    std::vector<std::string> toks;
    if (g.get_strs("tokenizer.ggml.tokens", &toks)) m.n_vocab = (int64_t)toks.size();
    if (m.n_vocab == 0) m.n_vocab = gi("vocab_size", 0);

    /* Everything else, verbatim and stringified. A plugin reads what it needs; the core never
     * interprets these. Arrays are summarised rather than expanded -- a 151k-token array in a
     * key/value list is not metadata, it is the vocab, and the vocab has its own section. */
    for (const GgufKV& kv : g.kv()) {
        if (kv.key.rfind("tokenizer.", 0) == 0) continue;
        std::string v;
        if (kv.is_array) {
            v = fmt("[%zu values]", kv.type == GGUF_STR ? kv.s.size()
                                  : !kv.i.empty()       ? kv.i.size() : kv.f.size());
            if (kv.type != GGUF_STR && kv.i.size() <= 8 && !kv.i.empty()) {
                v.clear();
                for (size_t j = 0; j < kv.i.size(); ++j)
                    v += fmt("%s%lld", j ? "," : "", (long long)kv.i[j]);
            }
        } else if (!kv.s.empty()) v = kv.s.front();
        else if (!kv.f.empty() && kv.type == GGUF_F32) v = fmt("%g", kv.f.front());
        else if (!kv.f.empty() && kv.type == GGUF_F64) v = fmt("%g", kv.f.front());
        else if (!kv.i.empty()) v = fmt("%lld", (long long)kv.i.front());
        if (v.size() > 240) v.resize(240);
        out->kv(kv.key, v);
    }

    /* THE DESCRIPTOR A SAFETENSORS RELEASE WOULD CARRY. A float file -- all F32 (0), mostly F16
     * (1) or mostly BF16 (32), the vectors F32 either way -- is an unquantised checkpoint and
     * selects the plugin a bf16 release does. Any other file type names its quantisation, which no
     * plugin serves as it stands. */
    int64_t ftype = 0;
    const bool typed = g.get_i64("general.file_type", &ftype);
    out->quant = !typed || ftype == 0 || ftype == 1 || ftype == 32
                     ? std::string() : fmt("gguf_ftype_%lld", (long long)ftype);
    out->finish();
    return RAD_OK;
}

/* ================================================================== the checkpoint */

int Checkpoint::open_gguf(const std::string& path) {
    RAD_TRY(gguf_.open(path.c_str()));
    kind_ = Gguf;
    for (const GgufTensor& t : gguf_.tensors()) {
        SrcTensor s;
        s.name  = t.name;
        s.ggml  = t.ggml_type;
        s.rad_dtype = gguf_to_rad_dtype(t.ggml_type);
        /* GGUF is fastest-first; everything from here on is row-major. */
        for (int d = (int)t.n_dims - 1; d >= 0; --d) s.shape.push_back(t.ne[d]);
        s.data  = gguf_.tensor_data(t);
        s.bytes = (int64_t)t.bytes;
        if (!s.data) {
            RAD_ERR("%s: '%s' has no readable extent", path.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return meta_from_gguf(gguf_, &meta_);
}

int Checkpoint::add_shard(const std::string& p, const std::string& prefix) {
    shards_.push_back(std::make_unique<SafeTensorsFile>());
    RAD_TRY(shards_.back()->open(p));
    for (const auto& t : shards_.back()->tensors()) {
        if (t.dtype == RAD_DT_INVALID) {
            RAD_ERR("%s: '%s' has dtype %s, which has no RAD equivalent", p.c_str(),
                    t.name.c_str(), t.dtype_str.c_str());
            return RAD_E_DTYPE;
        }
        SrcTensor s;
        s.name      = prefix + t.name;
        s.rad_dtype = t.dtype;
        s.shape     = t.shape;         /* safetensors is already row-major */
        s.data      = t.data;
        s.bytes     = (int64_t)t.bytes;
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return RAD_OK;
}

namespace {

/* Every shard an index names, once each and in name order, or the one model.safetensors. */
int shard_list(const std::string& dir, const std::string& index, bool index_given,
               std::vector<std::string>* out) {
    std::unordered_map<std::string, std::string> map;
    const int s = st_load_index(index, &map);
    if (s == RAD_OK) {
        for (auto& [t, f] : map)
            if (std::find(out->begin(), out->end(), f) == out->end()) out->push_back(f);
        std::sort(out->begin(), out->end());
        for (auto& f : *out) f = dir + "/" + f;
        return RAD_OK;
    }
    if (s != RAD_E_NOTFOUND || index_given) return s;
    out->push_back(dir + "/model.safetensors");
    return RAD_OK;
}

}  /* namespace */

int Checkpoint::open_safetensors(const std::string& path) {
    kind_ = SafeT;
    /* One shard, an index, or the directory holding either: one namespace whichever it is. */
    std::vector<std::string> shards;
    const bool is_index = path.size() > 10 && path.compare(path.size() - 10, 10, "index.json") == 0;
    const bool is_file = path.size() > 12 &&
                         path.compare(path.size() - 12, 12, ".safetensors") == 0;
    if (is_file) {
        shards.push_back(path);
    } else {
        RAD_TRY(shard_list(dir_, is_index ? path : dir_ + "/model.safetensors.index.json",
                           is_index, &shards));
    }
    for (const auto& sh : shards) RAD_TRY(add_shard(sh));

    RAD_TRY(meta_from_config_json(dir_, &meta_));
    RAD_TRY(meta_merge_generation_config(dir_, &meta_));
    RAD_TRY(meta_merge_preprocessor_configs(dir_, &meta_));
    /* FP8's block, where the checkpoint states one. */
    if (const char* bs = meta_.get("quantization_config.weight_block_size")) {
        long long r = 0, c = 0;
        if (std::sscanf(bs, "%lld %lld", &r, &c) == 2 && r > 0 && c > 0) {
            fp8_br_ = r;
            fp8_bc_ = c;
        }
    }
    if (meta_.quant == "mxfp4") RAD_TRY(alias_mxfp4());
    detect_import();
    meta_.finish();
    return RAD_OK;
}

/* A COMPRESSED-TENSORS MXFP4 LINEAR IS TWO U8 TENSORS, and they are already the two planes of
 * `fp4_e2m1*e8m0[1x32]` byte for byte:
 *
 *   X.weight_packed   U8 [N][K/2]    two e2m1 codes a byte, element 2j in the LOW nibble
 *   X.weight_scale    U8 [N][K/32]   an E8M0 exponent a 32 of K
 *
 * llm-compressor packs `x[..., 0::2] | x[..., 1::2] << 4`, and a 4-bit plane here is low nibble
 * first (rad_load_bits), so the codes need no repacking at all. Each pair is ALIASED as the names
 * the rest of this file already reads -- `X.weight` at fp4_e2m1 [N][K] and its companion
 * `X.weight_scale_inv` at e8m0 -- over the same mapped bytes, which makes the weight trivial
 * exactly as a block-fp8 one is and lets rad-convert copy it. Nothing is aliased for a pair whose
 * shapes disagree; it is refused by name, since a scale grid read at the wrong width is a model
 * that loads and is wrong. */
int Checkpoint::alias_mxfp4() {
    static const char kP[] = ".weight_packed";
    const size_t lp = sizeof(kP) - 1;
    std::vector<SrcTensor> add;
    for (const SrcTensor& t : tensors_) {
        if (t.name.size() <= lp || t.name.compare(t.name.size() - lp, lp, kP) != 0) continue;
        const std::string base = t.name.substr(0, t.name.size() - lp);
        const SrcTensor* sc = find(base + ".weight_scale");
        if (t.rad_dtype != RAD_U8 || t.shape.size() != 2 || !sc || sc->rad_dtype != RAD_U8 ||
            sc->shape.size() != 2 || sc->shape[0] != t.shape[0] ||
            sc->shape[1] * 32 != t.shape[1] * 2) {
            RAD_ERR("'%s' is not an MXFP4 pair: want U8 [N][K/2] codes beside a U8 [N][K/32] "
                    "'%s.weight_scale'", t.name.c_str(), base.c_str());
            return RAD_E_FORMAT;
        }
        if (find(base + ".weight")) continue;
        SrcTensor w = t;
        w.name      = base + ".weight";
        w.rad_dtype = RAD_FP4E2M1;
        w.shape     = { t.shape[0], t.shape[1] * 2 };
        SrcTensor s = *sc;
        s.name      = base + ".weight_scale_inv";
        s.rad_dtype = RAD_E8M0;
        add.push_back(std::move(w));
        add.push_back(std::move(s));
    }
    for (SrcTensor& s : add) {
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    RAD_INFO("checkpoint: %zu compressed-tensors MXFP4 linear(s) read as fp4_e2m1*e8m0[1x32]",
             add.size() / 2);
    return RAD_OK;
}

/* WHICH PUBLISHED QUANTISATION THIS IS, from quantization_config alone: the tensors are checked
 * one by one when a weight is resolved (import_piece), so a file that says one thing and holds
 * another is refused at the weight that differs. */
void Checkpoint::detect_import() {
    auto qc = [&](const std::string& k) {
        const char* v = meta_.get("quantization_config." + k);
        return v ? std::string(v) : std::string();
    };
    auto any_under = [&](const std::string& prefix) {
        for (const std::string& k : meta_.keys)
            if (k.rfind("quantization_config." + prefix, 0) == 0) return true;
        return false;
    };
    const std::string qm = qc("quant_method");
    if (qm == "auto-round") {
        /* AutoRound's GPTQ-compatible export; its own `auto_round` packing is another layout. */
        const std::string pf = qc("packing_format");
        if (pf == "auto_round:auto_gptq") import_ = Import::Gptq;
        else import_why_ = "AutoRound's packing_format '" + pf + "' is not one this reads; its "
                           "auto_round:auto_gptq export is";
    } else if (qm == "gptq") {
        /* v1, the format auto_gptq writes: the zero point stored less one. gptq_v2 stores it as
         * it is, and reading it as v1 would shift every code by one. */
        const std::string cf = qc("checkpoint_format");
        if (cf.empty() || cf == "gptq") import_ = Import::Gptq;
        else import_why_ = "GPTQ's checkpoint_format '" + cf + "' is not one this reads; 'gptq' is";
    } else if (qm == "quark") {
        const std::string w = "global_quant_config.weight.";
        if (qc(w + "dtype") == "fp4" && qc(w + "qscheme") == "per_group" &&
            qc(w + "group_size") == "32" && qc(w + "scale_format") == "e8m0" &&
            !any_under("layer_quant_config.") && !any_under("layer_type_quant_config."))
            import_ = Import::QuarkMx;
        else
            import_why_ = "Quark's weight scheme here is not MXFP4 (fp4, per_group 32, e8m0 scales) "
                          "throughout, which is the one this reads";
    }
    /* THE TOOL'S DESCRIPTION IS KEPT, UNDER ANOTHER NAME. A plugin reads quantization_config as
     * the statement of the format it serves (qwen35_fp8 checks quant_method is "fp8"), and an
     * imported checkpoint's says how a tool made it instead: every weight states its own
     * encoding. Renamed, it still travels into the container as provenance. */
    if (import_ != Import::None)
        for (std::string& k : meta_.keys)
            if (k.rfind("quantization_config.", 0) == 0) k = "imported_" + k;
}

int Checkpoint::import_piece(const std::string& name, const SrcTensor** codes,
                             const SrcTensor** scale, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (import_ == Import::None) return RAD_E_NOTFOUND;
    if (auto hit = imported_.find(name); hit != imported_.end()) {
        *codes = hit->second.first;
        *scale = hit->second.second;
        return RAD_OK;
    }

    if (import_ == Import::QuarkMx) {
        /* QUARK'S MXFP4 IS THE ENCODING AS IT STANDS: two codes a byte, element 2j in the low
         * nibble, and one biased exponent a 32 of K -- `fp4_e2m1*e8m0[1x32]`. A view renames the
         * dtypes and the logical width; no byte moves. Its excluded layers are plain tensors. */
        const SrcTensor* w = find(name);
        const SrcTensor* e = w ? find(name + "_scale") : nullptr;
        if (!w || !e || w->rad_dtype != RAD_U8) return RAD_E_NOTFOUND;
        if (e->rad_dtype != RAD_U8 || w->shape.size() != 2 || e->shape.size() != 2 ||
            e->shape[0] != w->shape[0] || e->shape[1] * 32 != w->shape[1] * 2)
            return refuse(RAD_E_SHAPE, "'" + e->name + "' is not the [N, K/32] exponent plane of '" +
                                       w->name + "'");
        SrcTensor c = *w;
        c.rad_dtype = RAD_FP4E2M1;
        c.shape = { w->shape[0], w->shape[1] * 2 };
        SrcTensor x = *e;
        x.rad_dtype = RAD_E8M0;
        sliced_.push_back(std::move(c));
        *codes = &sliced_.back();
        sliced_.push_back(std::move(x));
        *scale = &sliced_.back();
        imported_[name] = { *codes, *scale };
        return RAD_OK;
    }

    /* GPTQ-PACKED: `X.weight` is shipped as X.qweight, X.scales, X.qzeros and, optionally,
     * X.g_idx. The width is read off the tensors rather than quantization_config, because a
     * release may mix widths -- int4 experts beside int8 projections -- and say so only per layer. */
    const std::string suffix = ".weight";
    if (name.size() <= suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return RAD_E_NOTFOUND;
    const std::string base = name.substr(0, name.size() - suffix.size());
    const SrcTensor* qw = find(base + ".qweight");
    if (!qw) return RAD_E_NOTFOUND;
    const SrcTensor* sc = find(base + ".scales");
    const SrcTensor* qz = find(base + ".qzeros");
    const SrcTensor* gi = find(base + ".g_idx");
    const std::string what = "'" + qw->name + "'";
    if (qw->rad_dtype != RAD_I32 || qw->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " is not an int32 [K*bits/32, N] matrix");
    if (!sc || sc->shape.size() != 2 ||
        (sc->rad_dtype != RAD_F16 && sc->rad_dtype != RAD_BF16 && sc->rad_dtype != RAD_F32))
        return refuse(RAD_E_FORMAT, what + " has no float [groups, N] '" + base + ".scales' beside it");
    if (!qz || qz->rad_dtype != RAD_I32 || qz->shape.size() != 2)
        return refuse(RAD_E_FORMAT, what + " has no int32 '" + base + ".qzeros' beside it");
    const int64_t N = qw->shape[1];
    const int64_t bits = N > 0 && (qz->shape[1] * 32) % N == 0 ? qz->shape[1] * 32 / N : 0;
    if (bits != 4 && bits != 8)
        return refuse(RAD_E_UNSUPPORTED, what + " packs " + std::to_string(bits) + "-bit codes "
                                         "(from its qzeros' width); 4 and 8 are read");
    const int64_t K = qw->shape[0] * 32 / bits;
    const int64_t G = sc->shape[0];
    if (sc->shape[1] != N || G <= 0 || K % G)
        return refuse(RAD_E_SHAPE, "'" + sc->name + "' is not a [groups, " + std::to_string(N) +
                                   "] scale tensor for " + std::to_string(K) + " columns");
    const int64_t group = K / G;
    /* THE SERVED SCALE COVERS 128 OF K, so a group of 128 is one scale each and a wider one --
     * per channel included -- repeats; a narrower group has no form the kernels read. */
    if (K % 128 || group % 128)
        return refuse(RAD_E_UNSUPPORTED, what + " has groups of " + std::to_string(group) +
                                         " columns; the imported form scales a 128 of K, so the "
                                         "group must be a multiple of 128");
    if (qz->shape[0] != G)
        return refuse(RAD_E_SHAPE, "'" + qz->name + "' has " + std::to_string(qz->shape[0]) +
                                   " rows for " + std::to_string(G) + " groups");
    /* SYMMETRIC, OR NOT READ: every zero point the v1 2^(bits-1) - 1, which is what makes a code
     * the signed value it offsets to. An asymmetric weight would need a zero plane. */
    const uint32_t zero = bits == 4 ? 0x77777777u : 0x7f7f7f7fu;
    for (int64_t i = 0; i < qz->numel(); ++i) {
        uint32_t v;
        std::memcpy(&v, (const uint8_t*)qz->data + i * 4, 4);
        if (v != zero)
            return refuse(RAD_E_UNSUPPORTED, "'" + qz->name + "' holds a zero point other than "
                                             "the symmetric one; an asymmetric weight is not read");
    }
    /* AND IN ORDER: act-order (desc_act) permutes the columns' groups, which a [1x128] scale
     * cannot follow. The identity, or none, is the only g_idx read. */
    if (gi) {
        if (gi->rad_dtype != RAD_I32 || gi->numel() != K)
            return refuse(RAD_E_FORMAT, "'" + gi->name + "' is not an int32 [K] group index");
        for (int64_t k = 0; k < K; ++k) {
            int32_t v;
            std::memcpy(&v, (const uint8_t*)gi->data + k * 4, 4);
            if (v != k / group)
                return refuse(RAD_E_UNSUPPORTED, "'" + gi->name + "' reorders the columns' groups "
                                                 "(act-order); only an in-order index is read");
        }
    }
    SrcTensor c = *qw;
    c.rad_dtype = bits == 4 ? RAD_I4 : RAD_I8;
    c.shape = { N, K };
    c.gptq_bits = (int32_t)bits;
    c.gptq_group = group;
    sliced_.push_back(std::move(c));
    *codes = &sliced_.back();
    *scale = sc;
    imported_[name] = { *codes, *scale };
    return RAD_OK;
}

int Checkpoint::open(const std::string& path) {
    path_ = path;
    const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    dir_ = path;
    const size_t slash = path.find_last_of('/');
    auto ends = [&](const char* s) {
        const size_t n = std::strlen(s);
        return path.size() > n && path.compare(path.size() - n, n, s) == 0;
    };
    const bool is_file = gguf || ends(".safetensors") || ends("index.json");
    if (is_file) dir_ = slash == std::string::npos ? "." : path.substr(0, slash);
    return gguf ? open_gguf(path) : open_safetensors(path);
}

int Checkpoint::add_draft(const std::string& dir, const std::string& prefix) {
    if (kind_ != SafeT) {
        RAD_ERR("a drafter is merged into a safetensors target's namespace; %s is not one",
                path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    RAD_TRY(meta_merge_config_json(dir, prefix.empty() || prefix.back() != '.'
                                            ? prefix
                                            : prefix.substr(0, prefix.size() - 1), &meta_));
    std::vector<std::string> shards;
    RAD_TRY(shard_list(dir, dir + "/model.safetensors.index.json", false, &shards));
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, prefix));
    meta_.finish();
    return RAD_OK;
}

const SrcTensor* Checkpoint::find(const std::string& n) const {
    auto it = index_.find(n);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

/* ONE EXPERT'S SLICE OF A STACKED TENSOR, AS A VIEW.
 *
 * Slicing dim 0 of a row-major tensor is an offset and a shorter shape -- nothing is copied, so
 * the 512 declared weights of a MoE layer read 512 disjoint windows of the same mapped bytes.
 *
 * IT IS ALSO EXACT FOR A BLOCK-QUANTISED SOURCE, and that is not luck: ggml packs blocks along the
 * ROW, and a dim-0 slice is a whole number of rows, so a sub-tensor is a whole number of blocks.
 * `bytes % shape[0] == 0` therefore holds for every dtype and the division is the real byte count
 * rather than an estimate of one. A slice along any other dimension would not have that property
 * and is not offered. */
const SrcTensor* Checkpoint::slice(const SrcTensor& st, int index, const char* declared) {
    if (st.shape.size() < 2) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which has rank %zu. Only a STACKED tensor "
                "-- rank 2 or more, with the stack on dim 0 -- has sub-tensors to take.",
                declared, index, st.name.c_str(), st.shape.size());
        return nullptr;
    }
    const int64_t n = st.shape[0];
    if (index < 0 || index >= n) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which holds %lld of them",
                declared, index, st.name.c_str(), (long long)n);
        return nullptr;
    }
    if (!st.data || st.bytes <= 0 || st.bytes % n != 0) {
        RAD_ERR("'%s': '%s' is %lld byte(s) over %lld sub-tensor(s), which does not divide -- the "
                "slice would straddle two of them",
                declared, st.name.c_str(), (long long)st.bytes, (long long)n);
        return nullptr;
    }
    SrcTensor v;
    /* The name carries the index because every diagnostic below this point prints it, and
     * "experts.gate_up_proj" said 512 times for 512 different weights cannot be read. */
    v.name      = st.name + "[" + std::to_string(index) + "]";
    v.ggml      = st.ggml;
    v.rad_dtype = st.rad_dtype;
    v.shape.assign(st.shape.begin() + 1, st.shape.end());
    v.bytes     = st.bytes / n;
    v.data      = (const uint8_t*)st.data + (size_t)index * (size_t)v.bytes;
    sliced_.push_back(std::move(v));
    return &sliced_.back();
}

/* ================================================================== logical weights */

namespace {

/* FP8's scale plane is a companion tensor: `X.weight` is scaled by `X.weight_scale_inv`. */
std::string scale_name(const std::string& codes) { return codes + "_scale_inv"; }

/* The paroquant base of a `.weight` tensor name -- the name its `.pairs`/`.theta`/`channel_scales`
 * companions hang off -- or "" for a name that is not one. A rotated linear is a `.weight` with a
 * `.pairs` beside it, so this is the whole test ckpt_resolve needs. */
std::string paro_base_of(const std::string& n) {
    return n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0
               ? n.substr(0, n.size() - 7) : std::string();
}

}  /* namespace */

std::string ckpt_tensor_of(const SrcTensor& t) {
    const size_t br = t.name.size() && t.name.back() == ']' ? t.name.rfind('[') : std::string::npos;
    return br == std::string::npos ? t.name : t.name.substr(0, br);
}

bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit) {
    bool any = false;
    for (const auto* list : { &w.pieces, &w.scales })
        for (const SrcTensor* t : *list) {
            if (!t) continue;
            const std::string n = ckpt_tensor_of(*t);
            if (names.count(n)) {
                any = true;
                if (hit) hit->insert(n);
            }
        }
    return any;
}

int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    *out = CkptWeight{};
    out->name = name;

    /* EVERY map for the name, in declaration order. A CONCAT map may be declared in several
     * calls -- the ABI's `src[8]` against a 128-shard tensor -- and they join end to end; every
     * other repeat was refused at declare, so more than one implies CONCAT. */
    std::vector<const RadNameMap*> maps;
    for (const auto& nm : prog.name_map)
        if (nm.declared && name == nm.declared) maps.push_back(&nm);
    if (maps.empty()) return refuse(RAD_E_NOTFOUND, "the name map has no entry for it");
    struct Src1 { const char* name; int index; };
    std::vector<Src1> srcs;
    for (const RadNameMap* pm : maps)
        for (int k = 0; k < pm->n_src; ++k) srcs.push_back({ pm->src[k], pm->src_index[k] });
    out->concat_dim = maps.front()->concat_dim;

    /* THE MODE DECIDES WHAT SEVERAL SOURCES MEAN.
     *
     *   RAD_MAP_CONCAT   all of them, joined -- gate and up into gate_up.
     *   RAD_MAP_COPY     ALTERNATIVES, first present wins: a tied lm_head is output.weight where
     *                    the checkpoint ships one and the embedding where it does not.
     *
     * Treating a COPY's alternatives as a concatenation is silently wrong in both directions: a
     * checkpoint holding both writes the head at twice its rows, and one holding only the
     * embedding fails at the missing first source instead of falling back to it. */
    const bool alternatives = maps.front()->mode == RAD_MAP_COPY && srcs.size() > 1;
    for (const Src1& s1 : srcs) {
        const SrcTensor* st = ck.find(s1.name);
        const SrcTensor* sc = st ? ck.find(scale_name(s1.name)) : nullptr;
        /* A WEIGHT THE CHECKPOINT SHIPS IN AN IMPORTED FORM is read through that form's views. */
        {
            const SrcTensor *ic = nullptr, *is = nullptr;
            std::string iw;
            const int ir = ck.import_piece(s1.name, &ic, &is, &iw);
            if (ir == RAD_OK) {
                if (s1.index >= 0)
                    return refuse(RAD_E_UNSUPPORTED, "it is a slice of '" + std::string(s1.name) +
                                                     "', which the checkpoint ships quantised; an "
                                                     "imported tensor is read whole");
                st = ic;
                sc = is;
            } else if (ir != RAD_E_NOTFOUND) {
                return refuse(ir, iw);
            }
        }
        /* A SUB-TENSOR INDEX IS TAKEN, never ignored: ignoring it gives every expert a copy of
         * the whole stack, in a model that loads and answers. Its scale plane is stacked the same
         * way and sliced at the same index. */
        if (st && s1.index >= 0) {
            st = ck.slice(*st, s1.index, name.c_str());
            if (!st) return refuse(RAD_E_SHAPE, "a sub-tensor it names is not there");
            if (sc && !(sc = ck.slice(*sc, s1.index, name.c_str())))
                return refuse(RAD_E_SHAPE, "its scale plane cannot be sliced at the same index");
        }
        if (!st) {
            if (alternatives) continue;
            if (optional) { out->pieces.clear(); break; }
            return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensor '") + s1.name +
                                          "', which is not in " + ck.path());
        }
        out->pieces.push_back(st);
        out->scales.push_back(sc);
        if (alternatives) break;
    }
    if (out->pieces.empty()) {
        if (optional) return refuse(RAD_E_NOTFOUND, "none of its tensors is in the checkpoint");
        std::string tried;
        for (size_t k = 0; k < srcs.size(); ++k) tried += (k ? " or " : "") + std::string(srcs[k].name);
        return refuse(RAD_E_NOTFOUND, "it needs one of " + tried + ", and " + ck.path() +
                                      " holds none of them");
    }

    /* A PAROQUANT LINEAR, when every piece's `.weight` has the rotation companions beside it. It
     * is read into the core's own terms here and presented at its LOGICAL [N, K] -- the packed
     * codes are 3K/4 bytes a row, not K -- so a recipe's quantiser sees the de-rotated f32 rows
     * and never the codes. It is not trivial: only a recipe rule can decide it. */
    if (!paro_base_of(out->pieces[0]->name).empty() &&
        ck.find(paro_base_of(out->pieces[0]->name) + ".pairs")) {
        if (out->concat_dim != 0)
            return refuse(RAD_E_SHAPE, "'" + name + "' joins paroquant linears along a column");
        out->paro.reserve(out->pieces.size());
        int64_t cols0 = -1;
        for (const SrcTensor* st : out->pieces) {
            const std::string base = paro_base_of(st->name);
            std::string w1;
            auto p = std::make_shared<ParoLinear>();
            const int rc = base.empty() ? RAD_E_FORMAT : ckpt_paro_read(ck, base, p.get(), &w1);
            if (rc != RAD_OK)
                return refuse(rc == RAD_E_NOTFOUND ? RAD_E_FORMAT : rc,
                              "'" + name + "': " + (w1.empty() ? "a piece is not a paroquant "
                                                                "linear" : w1));
            if (cols0 < 0) cols0 = p->cols;
            else if (p->cols != cols0)
                return refuse(RAD_E_SHAPE, "'" + name + "' joins paroquant linears of different "
                                           "column counts");
            out->paro.push_back(std::move(p));
        }
        out->rank = 2;
        out->shape[0] = 0;
        for (const auto& p : out->paro) out->shape[0] += p->rows;
        out->shape[1] = out->paro[0]->cols;
        rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);
        out->trivial = false;
        out->why = "its source is a paroquant_mxfp6 linear, which no recipe rule has quantised";
        return RAD_OK;
    }

    /* THE LOGICAL SHAPE: the pieces joined along concat_dim, every other extent agreeing. */
    const SrcTensor& p0 = *out->pieces[0];
    const int d = out->concat_dim;
    if (p0.shape.empty() || p0.shape.size() > RAD_MAX_RANK || d < 0 || d >= (int)p0.shape.size())
        return refuse(RAD_E_SHAPE, "'" + p0.name + "' cannot be joined along dim " +
                                   std::to_string(d));
    out->rank = (uint32_t)p0.shape.size();
    for (uint32_t i = 0; i < out->rank; ++i) out->shape[i] = p0.shape[i];
    for (size_t k = 1; k < out->pieces.size(); ++k) {
        const SrcTensor& pk = *out->pieces[k];
        bool fits = pk.shape.size() == p0.shape.size();
        for (uint32_t i = 0; fits && i < out->rank; ++i)
            if ((int)i != d && pk.shape[i] != p0.shape[i]) fits = false;
        if (!fits)
            return refuse(RAD_E_SHAPE, "'" + pk.name + "' and '" + p0.name +
                                       "' do not join along dim " + std::to_string(d));
        out->shape[d] += pk.shape[(size_t)d];
    }
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);

    /* THE ENCODING THE PIECES ARE, when it is trivial (spec §4.3). */
    const uint32_t dt = p0.rad_dtype;
    bool same = true, scaled = true, unscaled = true;
    for (size_t k = 0; k < out->pieces.size(); ++k) {
        if (out->pieces[k]->rad_dtype != dt) same = false;
        if (out->scales[k]) unscaled = false; else scaled = false;
    }
    if (dt == RAD_DT_INVALID) {
        out->why = std::string("'") + p0.name + "' is a GGUF " +
                   (gguf_dtype(p0.ggml) ? gguf_dtype(p0.ggml)->name : "block") + " tensor";
    } else if (!same) {
        out->why = "its pieces are stored in different dtypes";
    } else if (p0.gptq_bits) {
        /* GPTQ-PACKED, IMPORTED: one width and one group throughout, joined only along rows --
         * a column join would put two pieces' groups in one scale block. */
        out->imported = true;
        for (size_t k = 1; k < out->pieces.size() && out->imported; ++k)
            if (out->pieces[k]->gptq_bits != p0.gptq_bits ||
                out->pieces[k]->gptq_group != p0.gptq_group) {
                out->imported = false;
                out->why = "its GPTQ-packed pieces differ in width or group";
            }
        if (out->imported && d != 0 && out->pieces.size() > 1) {
            out->imported = false;
            out->why = "its GPTQ-packed pieces are joined along a column, across their groups";
        }
        if (out->imported) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_BF16, 1, 128);
            out->why = "it is GPTQ-packed int" + std::to_string(p0.gptq_bits) + ", which "
                       "rad-convert imports with its scales rounded to bf16 -- serve the "
                       "container it writes";
        }
    } else if (dt == RAD_FP4E2M1 && scaled) {
        /* QUARK'S MXFP4, byte for byte: the codes and exponents are the planes of
         * `fp4_e2m1*e8m0[1x32]`. Joined only along rows, which is how gate and up become one. */
        out->trivial = d == 0 || out->pieces.size() == 1;
        if (!out->trivial) out->why = "its MXFP4 pieces are joined along a column";
        if (out->trivial) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", RAD_FP4E2M1, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", RAD_E8M0, 1, 32);
        }
    } else if ((dt == RAD_F8E4M3 || dt == RAD_F8E5M2) && scaled) {
        /* BLOCK-SCALED FP8, byte for byte: the codes and the scale plane are the planes of
         * `fp8_e4m3*bf16[128x128]` (whatever the scale's own dtype). Joined only along rows, and
         * only where each piece but the last fills its last row block -- a scale covering the
         * tail of one piece and the head of the next belongs to neither. */
        const int64_t br = ck.fp8_block_rows(), bc = ck.fp8_block_cols();
        const uint32_t sdt = out->scales[0]->rad_dtype;
        out->trivial = true;
        if (d != 0 && out->pieces.size() > 1) {
            out->trivial = false;
            out->why = "its FP8 pieces are joined along a column, across their scale blocks";
        }
        for (size_t k = 0; k < out->pieces.size() && out->trivial; ++k) {
            const SrcTensor& pk = *out->pieces[k];
            const SrcTensor& sk = *out->scales[k];
            int64_t r = 0, c = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &r, &c);
            const int64_t want_r = (r + br - 1) / br, want_c = (c + bc - 1) / bc;
            int64_t sr = 0, scl = 0;
            rad_enc_view((uint32_t)sk.shape.size(), sk.shape.data(), &sr, &scl);
            if (sk.rad_dtype != sdt || sr != want_r || scl != want_c) {
                out->trivial = false;
                out->why = "'" + sk.name + "' is not the [" + std::to_string(want_r) + ", " +
                           std::to_string(want_c) + "] " + std::to_string(br) + "x" +
                           std::to_string(bc) + " scale plane of '" + pk.name + "'";
            } else if (k + 1 < out->pieces.size() && r % br) {
                out->trivial = false;
                out->why = "'" + pk.name + "' ends inside a scale block, so the next piece's "
                           "scales cannot follow it";
            }
        }
        if (out->trivial) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", sdt, br, bc);
        }
    } else if (dt == RAD_FP4E2M1 && scaled) {
        /* OCP-MXFP4, the pair alias_mxfp4() made: e2m1 codes and one E8M0 exponent per 32 of K
         * in each row. The block is one row tall, so pieces join along rows with no alignment
         * condition; a join along columns would interleave two exponent grids and is not one. */
        out->trivial = d == 0 || out->pieces.size() == 1;
        if (!out->trivial) out->why = "its MXFP4 pieces are joined along a column";
        for (size_t k = 0; k < out->pieces.size() && out->trivial; ++k) {
            const SrcTensor& pk = *out->pieces[k];
            const SrcTensor& sk = *out->scales[k];
            if (sk.rad_dtype != RAD_E8M0 || pk.shape.size() != 2 || sk.shape.size() != 2 ||
                sk.shape[0] != pk.shape[0] || pk.shape[1] % 32 || sk.shape[1] != pk.shape[1] / 32) {
                out->trivial = false;
                out->why = "'" + sk.name + "' is not the [rows][K/32] E8M0 plane of '" + pk.name + "'";
            }
        }
        if (out->trivial) out->enc = rad_enc_affine(RAD_FP4E2M1, RAD_E8M0, 1, 32);
    } else if (!unscaled) {
        out->why = "some of its pieces carry a scale plane and some do not";
    } else if (dt == RAD_F8E4M3 || dt == RAD_F8E5M2) {
        out->why = "it is FP8 with no scale plane beside it";
    } else {
        out->trivial = rad_dtype_bits(dt) >= 8;
        if (out->trivial) out->enc = rad_enc_plain(dt);
        else out->why = std::string("it is ") + rad_dtype_name(dt);
    }
    return RAD_OK;
}

namespace {

/* ---------------------------------------------------------------- GGUF block types
 *
 * The ggml block types a weight can be read from as f32. Their GEOMETRY is core/text's -- that is
 * what sizes a tensor -- and decoding one is content, so the short table is here, where a weight
 * is read. The K-quants are refused by name rather than half-implemented: a super-block scale read
 * wrong is a model that talks and is wrong. */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
       GGML_Q8_0 = 8, GGML_BF16 = 30 };

int ggml_rows_f32(uint32_t type, const uint8_t* p, int64_t n, float* dst) {
    auto half = [&](const uint8_t* q) { uint16_t h; std::memcpy(&h, q, 2); return rad_f16_to_f32(h); };
    switch (type) {
        case GGML_Q8_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 34) {
                const float d = half(p);
                for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float)(int8_t)p[2 + j];
            }
            return RAD_OK;
        case GGML_Q4_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 18) {
                const float d = half(p);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)((p[2 + j] & 0x0F) - 8);
                    dst[b * 32 + j + 16] = d * (float)((p[2 + j] >> 4)   - 8);
                }
            }
            return RAD_OK;
        case GGML_Q4_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 20) {
                const float d = half(p), m = half(p + 2);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)(p[4 + j] & 0x0F) + m;
                    dst[b * 32 + j + 16] = d * (float)(p[4 + j] >> 4)   + m;
                }
            }
            return RAD_OK;
        case GGML_Q5_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 22) {
                const float d = half(p);
                uint32_t qh; std::memcpy(&qh, p + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[6 + j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int hi = (int)((p[6 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst[b * 32 + j]      = d * (float)lo;
                    dst[b * 32 + j + 16] = d * (float)hi;
                }
            }
            return RAD_OK;
        case GGML_Q5_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 24) {
                const float d = half(p), m = half(p + 2);
                uint32_t qh; std::memcpy(&qh, p + 4, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[8 + j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int hi = (int)((p[8 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
                    dst[b * 32 + j]      = d * (float)lo + m;
                    dst[b * 32 + j + 16] = d * (float)hi + m;
                }
            }
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

/* Where flat element `i` of the joined tensor lives: which piece, and the element in it. A join
 * along dim d repeats, for each index of the dims before d, every piece's [d..] extent in turn. */
struct Joined {
    std::vector<int64_t> inner;   /* each piece's elements per outer index */
    int64_t total_inner = 0;
    int     n = 0;
    void init(const std::vector<const SrcTensor*>& pieces, int d) {
        n = (int)pieces.size();
        inner.assign((size_t)n, 1);
        total_inner = 0;
        for (int k = 0; k < n; ++k) {
            for (size_t i = (size_t)d; i < pieces[(size_t)k]->shape.size(); ++i)
                inner[(size_t)k] *= pieces[(size_t)k]->shape[i];
            total_inner += inner[(size_t)k];
        }
    }
    /* Visit [a, b) as runs that lie in one piece: f(piece, element in it, count, at). */
    template <class F> int walk(int64_t a, int64_t b, F f) const {
        while (a < b) {
            const int64_t o = a / total_inner;
            int64_t w = a % total_inner, k = 0;
            while (w >= inner[(size_t)k]) w -= inner[(size_t)k++];
            const int64_t run = std::min<int64_t>(b - a, inner[(size_t)k] - w);
            const int rc = f((int)k, o * inner[(size_t)k] + w, run);
            if (rc < 0) return rc;
            a += run;
        }
        return RAD_OK;
    }
};

/* `n` elements of a stored-dtype piece from element `e`, as f32. */
int piece_f32(const SrcTensor& t, int64_t e, int64_t n, float* out) {
    if (t.gptq_bits) {
        /* the signed code: element (row, k) of the logical [N, K] matrix is field k % (32/bits)
         * of word [k / (32/bits), row], less its zero point */
        const int64_t N = t.shape[0], K = t.shape[1];
        const int64_t per = 32 / t.gptq_bits;
        const uint32_t mask = (1u << t.gptq_bits) - 1u;
        const int zp = 1 << (t.gptq_bits - 1);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t row = (e + i) / K, k = (e + i) % K;
            uint32_t w;
            std::memcpy(&w, (const uint8_t*)t.data + ((k / per) * N + row) * 4, 4);
            out[i] = (float)((int)((w >> ((k % per) * t.gptq_bits)) & mask) - zp);
        }
        return RAD_OK;
    }
    if (t.rad_dtype == RAD_DT_INVALID) {
        /* a GGUF block type: whole rows only, and a row is whole blocks */
        const int64_t cols = t.shape.back();
        if (cols <= 0 || e % cols || n % cols) return RAD_E_SHAPE;
        const int64_t rows = t.numel() / cols, rb = t.bytes / (rows ? rows : 1);
        return ggml_rows_f32(t.ggml, (const uint8_t*)t.data + (e / cols) * rb, n, out);
    }
    const uint8_t* p = (const uint8_t*)t.data;
    switch (t.rad_dtype) {
        case RAD_F32:
            std::memcpy(out, (const float*)t.data + e, (size_t)n * 4);
            return RAD_OK;
        case RAD_BF16: {
            /* The top half of an f32, so a shift -- and a loop the compiler vectorises, which the
             * per-element dtype switch below is not. Most of a checkpoint is this. */
            const uint16_t* q = (const uint16_t*)t.data + e;
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t)q[i] << 16;
                std::memcpy(out + i, &u, 4);
            }
            return RAD_OK;
        }
        case RAD_F16: case RAD_F8E4M3: case RAD_F8E5M2:
        case RAD_FP4E2M1: case RAD_E8M0:
        case RAD_I8: case RAD_U8: case RAD_I16: case RAD_I32: case RAD_U32: case RAD_U16:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_load_f32(p, t.rad_dtype, e + i);
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

}  /* namespace */

int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (r0 < 0 || nr < 0 || r0 + nr > w.rows) return refuse(RAD_E_SHAPE, "rows out of range");
    /* PAROQUANT: the pieces are rotated linears joined along rows; each is de-rotated to
     * base-oriented f32 rows by the core's givens decoder. */
    if (!w.paro.empty()) {
        int64_t base = 0;
        for (const std::shared_ptr<ParoLinear>& p : w.paro) {
            const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + p->rows);
            if (hi > lo) {
                std::string w1;
                const int st = ckpt_paro_rows_f32(*p, lo - base, hi - lo,
                                                  out + (lo - r0) * w.cols, &w1);
                if (st != RAD_OK) return refuse(st, w1);
            }
            base += p->rows;
        }
        return RAD_OK;
    }
    Joined j;
    j.init(w.pieces, w.concat_dim);
    /* the runs come in order, so the cursor is the elements already written */
    int64_t at = 0;
    const int rc = j.walk(r0 * w.cols, (r0 + nr) * w.cols, [&](int k, int64_t e, int64_t n) {
        const int s = piece_f32(*w.pieces[(size_t)k], e, n, out + at);
        at += n;
        return s;
    });
    if (rc < 0)
        return refuse(rc, "'" + w.pieces[0]->name + "' is stored in a form that cannot be read "
                          "as f32 (" + (w.pieces[0]->rad_dtype == RAD_DT_INVALID
                                            ? std::string(gguf_dtype(w.pieces[0]->ggml)
                                                              ? gguf_dtype(w.pieces[0]->ggml)->name
                                                              : "?")
                                            : std::string(rad_dtype_name(w.pieces[0]->rad_dtype))) +
                          ")");
    /* A GPTQ-PACKED CODE TIMES ITS GROUP'S SCALE, the scale read as shipped: the scale tensor is
     * [groups, N], so row r's scales are a column of it. */
    if (w.imported) {
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            const int64_t pr = pk.shape[0];
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *=
                        rad_load_f32(sk.data, sk.rad_dtype, (c / pk.gptq_group) * pr + (r - base));
            base += pr;
        }
        return RAD_OK;
    }
    /* FP8 IS ITS CODE TIMES ITS BLOCK'S SCALE (and MXFP4 its code times its exponent): the pieces
     * are joined along rows, so logical row r is in the piece whose rows cover it, and its scales
     * are that piece's. */
    if (!w.scales.empty() && w.scales[0]) {
        const int64_t br = w.enc.n_planes > 1 ? w.enc.plane[1].block[0] : 128;
        const int64_t bc = w.enc.n_planes > 1 ? w.enc.plane[1].block[1] : 128;
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            int64_t pr = 0, pc = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &pr, &pc);
            const int64_t sc = (pc + bc - 1) / bc;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *=
                        rad_load_f32(sk.data, sk.rad_dtype, ((r - base) / br) * sc + c / bc);
            base += pr;
        }
    }
    return RAD_OK;
}

int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (!(w.trivial || w.imported) || k < 0 || k >= w.enc.n_planes)
        return refuse(RAD_E_STATE, "not a plane of a trivial or imported encoding");
    const std::vector<const SrcTensor*>& src = k == 0 ? w.pieces : w.scales;
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
    if (r0 < 0 || nr < 0 || r0 + nr > pr) return refuse(RAD_E_SHAPE, "plane rows out of range");
    const uint32_t dt = w.enc.plane[k].dtype;
    const int64_t bits = rad_dtype_bits(dt);
    const int64_t eb = bits / 8;
    uint8_t* o = (uint8_t*)dst;
    if (w.imported) {
        /* GPTQ-PACKED, RE-LAID: a row of codes is a column of the qweight, its words' bytes in
         * order -- two codes a byte for int4, low nibble first, as the encoding packs them -- each
         * offset to signed by flipping the top bit of every field; a row of scales is a column of
         * the scale tensor, one a 128 of K, rounded to bf16. */
        int64_t base = 0;
        for (size_t p = 0; p < w.pieces.size(); ++p) {
            const SrcTensor& c = *w.pieces[p];
            const SrcTensor& s = *w.scales[p];
            const int64_t N = c.shape[0], K = c.shape[1];
            const uint8_t flip = c.gptq_bits == 4 ? 0x88 : 0x80;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + N); ++r) {
                const int64_t n = r - base;
                if (k == 0) {
                    const uint8_t* q = (const uint8_t*)c.data;
                    for (int64_t wi = 0; wi < K * c.gptq_bits / 32; ++wi) {
                        uint32_t v;
                        std::memcpy(&v, q + (wi * N + n) * 4, 4);
                        for (int b = 0; b < 4; ++b) *o++ = (uint8_t)(v >> (8 * b)) ^ flip;
                    }
                } else {
                    for (int64_t g = 0; g < K / 128; ++g) {
                        const uint16_t h = rad_f32_to_bf16(
                            rad_load_f32(s.data, s.rad_dtype, ((g * 128) / c.gptq_group) * N + n));
                        std::memcpy(o, &h, 2);
                        o += 2;
                    }
                }
            }
            base += N;
        }
        return RAD_OK;
    }
    if (k == 0) {
        /* the codes: the pieces joined, bytes as they are. A sub-byte code (MXFP4's e2m1) is
         * copied in whole bytes, which every run here is: a run starts and ends on a row, and a
         * row of an even-K 4-bit plane is whole bytes. */
        Joined j;
        j.init(src, w.concat_dim);
        return j.walk(r0 * pc, (r0 + nr) * pc, [&](int p, int64_t e, int64_t n) {
            if ((e * bits) % 8 || (n * bits) % 8) return (int)RAD_E_SHAPE;
            std::memcpy(o, (const uint8_t*)src[(size_t)p]->data + e * bits / 8, (size_t)(n * bits / 8));
            o += n * bits / 8;
            return (int)RAD_OK;
        });
    }
    /* an FP8 scale plane: the pieces' scale rows one after another */
    int64_t base = 0;
    for (const SrcTensor* s : src) {
        int64_t sr = 0, scl = 0;
        rad_enc_view((uint32_t)s->shape.size(), s->shape.data(), &sr, &scl);
        const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + sr);
        if (hi > lo) {
            std::memcpy(o, (const uint8_t*)s->data + (lo - base) * scl * eb,
                        (size_t)((hi - lo) * scl * eb));
            o += (hi - lo) * scl * eb;
        }
        base += sr;
    }
    return RAD_OK;
}

/* ================================================================== paroquant_mxfp6 */

namespace {

/* The rotation group width a published ParoQuant checkpoint uses: `group_size`, and the width the
 * check performed below relies on -- `pairs` are a permutation of 0..127 inside each 128. */
constexpr int64_t kParoGroup = 128;

bool paro_rank2(const SrcTensor* t, int64_t* r, int64_t* c) {
    if (t->shape.size() != 2) return false;
    *r = t->shape[0];
    *c = t->shape[1];
    return true;
}

}  /* namespace */

int ckpt_paro_read(Checkpoint& ck, const std::string& base, ParoLinear* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    *out = ParoLinear{};
    out->name = base;
    const SrcTensor* w  = ck.find(base + ".weight");
    const SrcTensor* ws = ck.find(base + ".weight_scale");
    const SrcTensor* pa = ck.find(base + ".pairs");
    const SrcTensor* th = ck.find(base + ".theta");
    const SrcTensor* cs = ck.find(base + ".channel_scales");
    if (!w || !ws || !pa || !th || !cs) {
        std::string miss;
        auto need = [&](const char* s, const SrcTensor* t) {
            if (!t) miss += (miss.empty() ? "" : ", ") + std::string(s);
        };
        need(".weight", w); need(".weight_scale", ws); need(".pairs", pa);
        need(".theta", th); need(".channel_scales", cs);
        return refuse(RAD_E_NOTFOUND, "'" + base + "' is missing " + miss);
    }
    int64_t N = 0, wc = 0, sr = 0, sc = 0, S = 0, pc = 0, tr = 0, tc = 0;
    if (w->rad_dtype != RAD_U8 || !paro_rank2(w, &N, &wc))
        return refuse(RAD_E_FORMAT, "'" + w->name + "' is not a rank-2 u8 codes plane");
    if (ws->rad_dtype != RAD_U8 || !paro_rank2(ws, &sr, &sc) || sr != N || sc < 1)
        return refuse(RAD_E_FORMAT, "'" + ws->name + "' is not an [" + std::to_string(N) +
                                    ", K/32] u8 scale plane for '" + w->name + "'");
    const int64_t K = sc * 32;
    if (wc * 4 != K * 3)
        return refuse(RAD_E_FORMAT, "'" + w->name + "' holds " + std::to_string(wc) +
                                    " bytes a row and " + std::to_string(K) +
                                    " columns would pack into " + std::to_string(K * 3 / 4));
    if (pa->rad_dtype != RAD_I16 || !paro_rank2(pa, &S, &pc) || S < 1 || pc != K)
        return refuse(RAD_E_FORMAT, "'" + pa->name + "' is not an [S, " + std::to_string(K) +
                                    "] i16 pairs plane");
    if (th->rad_dtype != RAD_F16 || !paro_rank2(th, &tr, &tc) || tr != S || tc != K / 2)
        return refuse(RAD_E_FORMAT, "'" + th->name + "' is not an [" + std::to_string(S) + ", " +
                                    std::to_string(K / 2) + "] f16 angle plane");
    if (cs->rad_dtype != RAD_F16 || cs->numel() != K)
        return refuse(RAD_E_FORMAT, "'" + cs->name + "' is not a " + std::to_string(K) +
                                    "-entry f16 channel scale plane");
    if (K % kParoGroup || K % 2)
        return refuse(RAD_E_SHAPE, "'" + base + "' has " + std::to_string(K) +
                                   " columns, not a whole number of " +
                                   std::to_string(kParoGroup) + "-wide rotation groups");

    out->rows = N;
    out->cols = K;
    out->stages = S;
    out->group = kParoGroup;
    /* the codes and their MX scale are read as they are: the packed 6-bit bitstream is the ABI's
     * own (lowest bits first along the row), and an E8M0 plane is a byte a group. They point into
     * the checkpoint rather than being copied. */
    out->code = w->data;
    out->wscale = ws->data;
    std::string t;
    if (enc_givens_paro_tables(S, K, kParoGroup, (const int16_t*)pa->data, (const uint16_t*)th->data,
                               (const uint16_t*)cs->data, &out->pairs, &out->angle,
                               &out->tscale, &t) != RAD_OK)
        return refuse(RAD_E_FORMAT, "'" + base + "': " + t);

    rad_enc_clear(&out->enc);
    rad_enc_copy_str(out->enc.scheme, "affine");
    rad_enc_add_plane(&out->enc, "codes", RAD_FP6E2M3, 1, 1);
    rad_enc_add_plane(&out->enc, "scale", RAD_E8M0, 1, 32);
    rad_enc_add_table(&out->enc, "t.pairs", RAD_I32, S, K);
    rad_enc_add_table(&out->enc, "t.angle", RAD_F32, S, K / 2);
    rad_enc_add_table(&out->enc, "t.scale", RAD_F32, 1, K);
    rad_enc_copy_str(out->enc.transform, "givens");
    return RAD_OK;
}

int ckpt_paro_rows_f32(const ParoLinear& w, int64_t r0, int64_t nr, float* out,
                       std::string* why) {
    if (r0 < 0 || nr < 0 || r0 + nr > w.rows) {
        if (why) *why = "paroquant rows out of range";
        return RAD_E_SHAPE;
    }
    const void* p[RAD_ENC_MAX_PLANES] = {};
    for (int k = 0; k < w.enc.n_planes && k < RAD_ENC_MAX_PLANES; ++k) p[k] = w.data(k);
    EncodedView v;
    enc_view(w.enc, w.rows, w.cols, p, &v);
    return enc_decode_rows(v, r0, nr, out, why);
}

}  /* namespace rad */
