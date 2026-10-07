/* LFM2's text backbone: short gated convolutions interleaved with full GQA attention.
 * The layer schedule and norm gains come from the checkpoint. In particular, d1-3B has
 * an irregular schedule, plain RMSNorm (not 1+w), and a tied vocabulary projection.
 *
 * BF16 Hugging Face weights are served on one rank. LFM2-VL adds its SigLIP2 encoder when the
 * deployment enables media; image rows replace their placeholder embeddings before the stack.
 */
#include <arch/rad_arch.h>
#include "vision.h"

#include <sstream>

namespace lfm2_bf16 {
using namespace rad::arch;

struct Layer {
    bool attention = false;
    int index = 0;
    rad_weight norm = 0, in = 0, out = 0, conv = 0, qnorm = 0, knorm = 0, k = 0, v = 0;
    rad_op op_norm = 0, op_in = 0, op_out = 0, op_add = 0;
    rad_op op_pre_gate = 0, op_conv = 0, op_post_gate = 0;
    rad_op op_k = 0, op_v = 0, op_qnorm = 0, op_knorm = 0;
    rad_op op_rope_q = 0, op_rope_k = 0, op_attn = 0, op_store = 0;
    rad_weight ffn_norm = 0, gate_up = 0, down = 0;
    rad_op op_ffn_norm = 0, op_gate_up = 0, op_silu = 0, op_ffn_gate = 0, op_down = 0;
};

struct Model {
    Names nm{""};
    Geom g{};
    int64_t width = 0;
    rad_kvgroup kv_attn = 0, kv_conv = 0;
    rad_buf x = 0, h = 0, projected = 0, conv_in = 0, conv_out = 0;
    rad_buf q = 0, k = 0, v = 0, attn_out = 0, gate_up = 0, ffn = 0, hout = 0, logits = 0;
    rad_buf logits_bf16 = 0;
    rad_weight embedding = 0, final_norm = 0, output = 0;
    rad_op op_embed = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    rad_op op_logits_round = 0, op_logits_widen = 0, op_logits_select = 0;
    std::vector<Layer> layers;
    Siglip2Vision vision;
    MediaRows media_rows;
};
static Model models[MAX_RANKS];

static const char* text_key(const RadModelMeta* meta, const char* key, const char* fallback = nullptr) {
    const std::string nested = std::string("text_config.") + key;
    return rad_meta_gets(meta, nested.c_str(), rad_meta_gets(meta, key, fallback));
}
static int64_t text_int(const RadModelMeta* meta, const char* key, int64_t fallback) {
    const char* value = text_key(meta, key);
    return value ? std::strtoll(value, nullptr, 10) : fallback;
}

static int schedule(const RadModelMeta* meta, int64_t count, std::vector<Layer>& layers) {
    const char* value = text_key(meta, "layer_types");
    if (!value) {
        std::fprintf(stderr, "radiance: lfm2 requires the checkpoint's layer_types schedule\n");
        return RAD_E_FORMAT;
    }
    std::istringstream input(value);
    std::string kind;
    while (input >> kind) {
        if ((int64_t)layers.size() >= count || (kind != "conv" && kind != "full_attention")) {
            std::fprintf(stderr, "radiance: lfm2 has an invalid layer_types schedule\n");
            return RAD_E_FORMAT;
        }
        Layer l;
        l.attention = kind == "full_attention";
        l.index = (int)layers.size();
        layers.push_back(l);
    }
    if ((int64_t)layers.size() != count) {
        std::fprintf(stderr, "radiance: lfm2 layer_types names %zu layers, expected %lld\n",
                     layers.size(), (long long)count);
        return RAD_E_FORMAT;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        std::fprintf(stderr, "radiance: lfm2_bf16 currently serves text on one rank; "
                             "tensor parallelism is not declared\n");
        return RAD_E_UNSUPPORTED;
    }
    if (meta->n_expert || text_int(meta, "conv_bias", 0) ||
        !text_int(meta, "block_use_swiglu", 1) || !text_int(meta, "use_pos_enc", 1)) {
        std::fprintf(stderr, "radiance: lfm2_bf16 requires dense SwiGLU, rotary attention, "
                             "and bias-free short convolutions\n");
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_gets(meta, "general.architecture", nullptr)) {
        std::fprintf(stderr, "radiance: lfm2_bf16 currently maps Hugging Face tensors; "
                             "use the original safetensors checkpoint\n");
        return RAD_E_UNSUPPORTED;
    }
    Model& m = models[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RadModelMeta dims = *meta;
    /* Older LFM2 configs state the FF width before their rounding rule. */
    if (text_int(meta, "block_auto_adjust_ff_dim", 0)) {
        dims.n_ff = 2 * dims.n_ff / 3;
        const char* multiplier = text_key(meta, "block_ffn_dim_multiplier");
        if (multiplier) {
            const double f = std::strtod(multiplier, nullptr);
            const int64_t multiple = text_int(meta, "block_multiple_of", 256);
            if (!(f > 0) || multiple <= 0) return RAD_E_FORMAT;
            dims.n_ff = (int64_t)(dims.n_ff * f);
            dims.n_ff = multiple * ((dims.n_ff + multiple - 1) / multiple);
        }
    }
    if (const char* eps = text_key(meta, "norm_eps")) dims.rms_eps = (float)std::strtod(eps, nullptr);
    RAD_ARCH_TRY(geom_from(m.g, &dims, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    Names& nm = m.nm;
    RAD_ARCH_TRY(yarn_from(g, meta, "", g.head_dim));
    m.width = text_int(meta, "conv_L_cache", 0);
    if (m.width < 2 || m.width > 65 || g.head_dim % 2) return RAD_E_FORMAT;
    RAD_ARCH_TRY(schedule(meta, g.n_layer, m.layers));
    const bool vl = std::strcmp(rad_meta_gets(meta, "model_type", "lfm2"), "lfm2_vl") == 0;
    const char* prefix = vl ? "model.language_model" : "model";

    RadKVGroupDecl ad{};
    ad.kind = RAD_KV_FULL; ad.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
    ad.n_head_kv = g.n_head_kv; ad.head_dim = g.head_dim;
    m.kv_attn = rad_decl_kv_group(b, nm.f("kv_attn"), &ad);
    g.kv_dtype = rad_dtype_name(ad.dtype);
    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV; cd.dtype = RAD_BF16; cd.n_head_kv = 1;
    cd.head_dim = g.n_embd; cd.conv_width = m.width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_conv"), &cd);

    m.x = decl_b(b, nm.f("x"), RAD_BF16, {g.max_tok, g.n_embd});
    m.h = decl_b(b, nm.f("h"), RAD_BF16, {g.max_tok, g.n_embd});
    m.projected = decl_b(b, nm.f("conv_bcu"), RAD_BF16, {g.max_tok, 3 * g.n_embd});
    m.conv_in = decl_b(b, nm.f("conv_in"), RAD_BF16, {g.max_tok, g.n_embd});
    m.conv_out = decl_b(b, nm.f("conv_out"), RAD_BF16, {g.max_tok, g.n_embd});
    m.q = decl_b(b, nm.f("q"), RAD_BF16, {g.max_tok, g.q_dim()});
    m.k = decl_b(b, nm.f("k"), RAD_BF16, {g.max_tok, g.kv_dim()});
    m.v = decl_b(b, nm.f("v"), RAD_BF16, {g.max_tok, g.kv_dim()});
    m.attn_out = decl_b(b, nm.f("attn_out"), RAD_BF16, {g.max_tok, g.q_dim()});
    m.gate_up = decl_b(b, nm.f("gate_up"), RAD_BF16, {g.max_tok, 2 * g.n_ff});
    m.ffn = decl_b(b, nm.f("ffn"), RAD_BF16, {g.max_tok, g.n_ff});
    m.hout = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, g.n_embd});
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    m.logits_bf16 = decl_b(b, nm.f("logits_bf16"), RAD_BF16, {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", prefix)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.embedding_norm.weight", prefix)));
    m.embedding = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                         RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.final_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    if (text_int(meta, "tie_word_embeddings", 0)) {
        m.output = m.embedding;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
        m.output = decl_w(b, nm.f("output.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                          RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    }
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype)),
        RAD_WEIGHTS(m.embedding)), {}, {m.x});

    for (Layer& l : m.layers) {
        const int i = l.index;
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", i),
                             nm.ckpt("%s.layers.%d.operator_norm.weight", prefix, i)));
        l.norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
        l.op_norm = rw(b, RAD_OP(b, "rmsnorm",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                       RAD_F64("eps", g.eps), RAD_INT("round_before_weight", 1),
                       RAD_STR("dtype", g.dtype)),
            RAD_WEIGHTS(l.norm)), {m.x}, {m.h});
        if (l.attention) {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, i, m.kv_attn));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", prefix, i)));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", prefix, i)));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", prefix, i)));
            l.in = decl_w(b, nm.f("blk.%d.attn_q.weight", i), RAD_BF16,
                          {g.q_dim(), g.n_embd}, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_NONE, grp_layer(i));
            l.k = decl_w(b, nm.f("blk.%d.attn_k.weight", i), RAD_BF16,
                        {g.kv_dim(), g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.v = decl_w(b, nm.f("blk.%d.attn_v.weight", i), RAD_BF16,
                        {g.kv_dim(), g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.q_layernorm.weight", prefix, i)));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.k_layernorm.weight", prefix, i)));
            l.qnorm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.knorm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", i),
                                 nm.ckpt("%s.layers.%d.self_attn.out_proj.weight", prefix, i)));
            l.out = decl_w(b, nm.f("blk.%d.attn_output.weight", i), RAD_BF16,
                           {g.n_embd, g.q_dim()}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.op_in = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.q_dim()),
                           RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.in)), {m.h}, {m.q});
            l.op_k = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.kv_dim()),
                           RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.k)), {m.h}, {m.k});
            l.op_v = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.kv_dim()),
                           RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.v)), {m.h}, {m.v});
            l.op_qnorm = rw(b, RAD_OP(b, "rmsnorm",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head), RAD_INT("n", g.head_dim),
                           RAD_F64("eps", g.eps), RAD_INT("round_before_weight", 1),
                           RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.qnorm)), {m.q}, {m.q});
            l.op_knorm = rw(b, RAD_OP(b, "rmsnorm",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv), RAD_INT("n", g.head_dim),
                           RAD_F64("eps", g.eps), RAD_INT("round_before_weight", 1),
                           RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.knorm)), {m.k}, {m.k});
            l.op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0,
                                         g.head_dim, "neox"), {m.q}, {m.q});
            l.op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0,
                                         g.head_dim, "neox"), {m.k}, {m.k});
            l.op_attn = rw(b, RAD_OP(b, "attn_paged",
                RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                           RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1), RAD_INT("window", 0),
                           RAD_INT("n_head", g.n_head), RAD_INT("block_size", RAD_KV_BLOCK),
                           RAD_INT("max_ctx", g.max_ctx), RAD_INT("max_seqs", g.max_seqs),
                           RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                RAD_NOWEIGHTS), {m.q}, {m.attn_out});
            l.op_store = rw(b, RAD_OP(b, "kv_store",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                           RAD_INT("n_head_kv", g.n_head_kv),
                           RAD_INT("block_size", rad_kv_block_size(b, m.kv_attn)),
                           RAD_STR("kv_dtype", g.kv_dtype)), RAD_NOWEIGHTS), {m.k, m.v}, {});
            l.op_out = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                           RAD_INT("K", g.q_dim()), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.out)), {m.attn_out}, {m.h});
        } else {
            RAD_ARCH_TRY(rad_bind_layer_kv(b, i, m.kv_conv));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.shortconv_in.weight", i),
                                 nm.ckpt("%s.layers.%d.conv.in_proj.weight", prefix, i)));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.shortconv.weight", i),
                                 nm.ckpt("%s.layers.%d.conv.conv.weight", prefix, i)));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.shortconv_out.weight", i),
                                 nm.ckpt("%s.layers.%d.conv.out_proj.weight", prefix, i)));
            l.in = decl_w(b, nm.f("blk.%d.shortconv_in.weight", i), RAD_BF16,
                          {3 * g.n_embd, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.conv = decl_w(b, nm.f("blk.%d.shortconv.weight", i), RAD_BF16,
                            {g.n_embd, m.width}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.out = decl_w(b, nm.f("blk.%d.shortconv_out.weight", i), RAD_BF16,
                           {g.n_embd, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
            l.op_in = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 3 * g.n_embd),
                           RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.in)), {m.h}, {m.projected});
            l.op_pre_gate = rw(b, RAD_OP(b, "mul",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                           RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS), {m.projected}, {m.conv_in});
            l.op_conv = rw(b, RAD_OP(b, "causal_conv1d",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                           RAD_INT("width", m.width), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.conv)), {m.conv_in}, {m.conv_out});
            l.op_post_gate = rw(b, RAD_OP(b, "mul",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                           RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
                {m.projected, m.conv_out}, {m.conv_out});
            l.op_out = rw(b, RAD_OP(b, "gemm_nt",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                           RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                RAD_WEIGHTS(l.out)), {m.conv_out}, {m.h});
        }
        l.op_add = rw(b, RAD_OP(b, "add",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                       RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS), {m.x, m.h}, {m.x});
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", i),
                             nm.ckpt("%s.layers.%d.ffn_norm.weight", prefix, i)));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", i),
                               {nm.ckpt("%s.layers.%d.feed_forward.w1.weight", prefix, i),
                                nm.ckpt("%s.layers.%d.feed_forward.w3.weight", prefix, i)}, 0));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_down.weight", i),
                             nm.ckpt("%s.layers.%d.feed_forward.w2.weight", prefix, i)));
        l.ffn_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                           RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
        l.gate_up = decl_w(b, nm.f("blk.%d.ffn_gate_up.weight", i), RAD_BF16,
                          {2 * g.n_ff, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
        l.down = decl_w(b, nm.f("blk.%d.ffn_down.weight", i), RAD_BF16,
                       {g.n_embd, g.n_ff}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(i));
        l.op_ffn_norm = rw(b, RAD_OP(b, "rmsnorm",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                       RAD_F64("eps", g.eps), RAD_INT("round_before_weight", 1),
                       RAD_STR("dtype", g.dtype)), RAD_WEIGHTS(l.ffn_norm)), {m.x}, {m.h});
        l.op_gate_up = rw(b, RAD_OP(b, "gemm_nt",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * g.n_ff),
                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
            RAD_WEIGHTS(l.gate_up)), {m.h}, {m.gate_up});
        /* The checkpoint rounds SiLU to BF16 before multiplication by the up branch. */
        l.op_silu = rw(b, RAD_OP(b, "silu",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                       RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS), {m.gate_up}, {m.ffn});
        l.op_ffn_gate = rw(b, RAD_OP(b, "mul",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                       RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS), {m.ffn, m.gate_up}, {m.ffn});
        l.op_down = rw(b, RAD_OP(b, "gemm_nt",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                       RAD_INT("K", g.n_ff), RAD_STR("dtype", g.dtype)),
            RAD_WEIGHTS(l.down)), {m.ffn}, {m.h});
    }
    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                   RAD_F64("eps", g.eps), RAD_INT("round_before_weight", 1),
                   RAD_STR("dtype", g.dtype)),
        RAD_WEIGHTS(m.final_norm)), {m.hout}, {m.hout});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                   RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS), {m.x}, {m.hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                   RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
        RAD_WEIGHTS(m.output)), {m.hout}, {m.logits});
    /* The source model's BF16 vocabulary projection is widened only for log-softmax. */
    m.op_logits_round = rw(b, RAD_OP(b, "cast",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_vocab),
                   RAD_STR("from", "f32"), RAD_STR("to", "bf16")), RAD_NOWEIGHTS),
        {m.logits}, {m.logits_bf16});
    m.op_logits_widen = rw(b, RAD_OP(b, "cast",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_vocab),
                   RAD_STR("from", "bf16"), RAD_STR("to", "f32")), RAD_NOWEIGHTS),
        {m.logits_bf16}, {m.logits});
    m.op_logits_select = rw(b, RAD_OP(b, "logits_select",
        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows * RAD_MAX_SCORE_CANDIDATES),
                   RAD_INT("n_rows", g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                   RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype),
                   RAD_INT("round_bf16", 1)), RAD_WEIGHTS(m.output)), {m.hout}, {});
    RAD_ARCH_TRY(rad_declare_token_scores(b));
    rad_note(b, "LFM2 BF16 text: %lld layers, explicit schedule, short-conv width %lld, "
                "full-head NeoX rotary, RMS epsilon %g, %s vocabulary head",
             (long long)g.n_layer, (long long)m.width, (double)g.eps,
             m.output == m.embedding ? "tied" : "untied");
    if (vl) {
        RAD_ARCH_TRY(m.vision.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
        if (m.vision.on) RAD_ARCH_TRY(m.media_rows.declare(b, g.max_tok, g.n_embd, m.x));
    }
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = models[rad_rank(c)];
    const Geom& g = m.g;
    if (batch->enc) {
        m.vision.step(c, batch);
        return;
    }
    if (batch->draft_pass || batch->spec_parent || (batch->n_mm_rows > 0 && !m.vision.on)) {
        rad_step_fail(c, "lfm2_bf16 has no draft/tree pass, or image rows were supplied without an encoder");
        return;
    }
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.embedding), brows(m.x, T));
    m.media_rows.step(c, batch, m.x, T);
    for (const Layer& l : m.layers) {
        RAD_ISSUE(c, l.op_norm, brows(m.x, T), RAD_W(l.norm), brows(m.h, T));
        if (l.attention) {
            const RadKVGroupBatch* kv = kv_batch(batch, m.kv_attn);
            RAD_ISSUE(c, l.op_in, brows(m.h, T), RAD_W(l.in), brows(m.q, T));
            RAD_ISSUE(c, l.op_k, brows(m.h, T), RAD_W(l.k), brows(m.k, T));
            RAD_ISSUE(c, l.op_v, brows(m.h, T), RAD_W(l.v), brows(m.v, T));
            RAD_ISSUE_N(c, l.op_qnorm, T * g.n_head, brows(m.q, T), RAD_W(l.qnorm), brows(m.q, T));
            RAD_ISSUE_N(c, l.op_knorm, T * g.n_head_kv, brows(m.k, T), RAD_W(l.knorm), brows(m.k, T));
            RAD_ISSUE(c, l.op_rope_q, brows(m.q, T), rope_pos1(batch, T));
            RAD_ISSUE(c, l.op_rope_k, brows(m.k, T), rope_pos1(batch, T));
            RAD_ISSUE(c, l.op_store, brows(m.k, T), brows(m.v, T),
                      praw(kv ? kv->slot_mapping : nullptr, RAD_I32, T), kv_cache(m.kv_attn, l.index));
            RAD_ISSUE_N(c, l.op_attn, batch->max_q_len, brows(m.q, T), kv_cache(m.kv_attn, l.index),
                        praw2(kv ? kv->block_table : nullptr, RAD_I32, batch->n_seq,
                              kv ? kv->block_table_pitch : 0),
                        praw(kv ? kv->seqused : nullptr, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(m.attn_out, T));
            RAD_ISSUE(c, l.op_out, brows(m.attn_out, T), RAD_W(l.out), brows(m.h, T));
        } else {
            const RadKVGroupBatch* cv = kv_batch(batch, m.kv_conv);
            const int32_t* idx = cv ? cv->state_index : nullptr;
            RAD_ISSUE(c, l.op_in, brows(m.h, T), RAD_W(l.in), brows(m.projected, T));
            RAD_ISSUE(c, l.op_pre_gate, bcol(m.projected, 0, g.n_embd, T),
                      bcol(m.projected, 2 * g.n_embd, g.n_embd, T), brows(m.conv_in, T));
            int64_t D = 0, DT = 0;
            batch_split(batch, &D, &DT);
            if (D > 0)
                RAD_ISSUE_N(c, l.op_conv, DT, brows(m.conv_in, DT), RAD_W(l.conv),
                            kv_cache(m.kv_conv, l.index), praw(batch->cu_seqlens, RAD_I32, D + 1),
                            praw(idx, RAD_I32, D), praw(batch->ctx_lens, RAD_I32, D),
                            praw(batch->num_accepted, RAD_I32, D), brows(m.conv_out, DT));
            if (batch->n_seq > D)
                RAD_ISSUE(c, l.op_conv, brows(m.conv_in, T), RAD_W(l.conv), kv_cache(m.kv_conv, l.index),
                          praw(batch->cu_seqlens ? batch->cu_seqlens + D : nullptr,
                               RAD_I32, batch->n_seq - D + 1),
                          praw(idx ? idx + D : nullptr, RAD_I32, batch->n_seq - D),
                          praw(batch->ctx_lens ? batch->ctx_lens + D : nullptr,
                               RAD_I32, batch->n_seq - D), RAD_NONE, brows(m.conv_out, T));
            RAD_ISSUE(c, l.op_post_gate, bcol(m.projected, g.n_embd, g.n_embd, T),
                      brows(m.conv_out, T), brows(m.conv_out, T));
            RAD_ISSUE(c, l.op_out, brows(m.conv_out, T), RAD_W(l.out), brows(m.h, T));
        }
        RAD_ISSUE(c, l.op_add, brows(m.x, T), brows(m.h, T), brows(m.x, T));
        RAD_ISSUE(c, l.op_ffn_norm, brows(m.x, T), RAD_W(l.ffn_norm), brows(m.h, T));
        RAD_ISSUE(c, l.op_gate_up, brows(m.h, T), RAD_W(l.gate_up), brows(m.gate_up, T));
        RAD_ISSUE(c, l.op_silu, bcol(m.gate_up, 0, g.n_ff, T), brows(m.ffn, T));
        RAD_ISSUE(c, l.op_ffn_gate, brows(m.ffn, T), bcol(m.gate_up, g.n_ff, g.n_ff, T), brows(m.ffn, T));
        RAD_ISSUE(c, l.op_down, brows(m.ffn, T), RAD_W(l.down), brows(m.h, T));
        RAD_ISSUE(c, l.op_add, brows(m.x, T), brows(m.h, T), brows(m.x, T));
    }
    if (batch->n_out > 0) {
        const bool selected = batch->n_score_values > 0;
        const int64_t full = selected ? batch->n_full_logits : batch->n_out;
        if (batch->n_out > g.max_logit_rows || full < 0 || full > batch->n_out ||
            (selected && (!batch->score_rows || !batch->score_tokens || !batch->score_values ||
                          batch->n_score_values > g.max_logit_rows * RAD_MAX_SCORE_CANDIDATES))) {
            rad_step_fail(c, "invalid LFM2 selected-logit batch");
            return;
        }
        RAD_ISSUE_N(c, m.op_gather, batch->n_out, brows(m.x, T),
                    praw(batch->out_ids, RAD_I32, batch->n_out), brows(m.hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_final_norm, batch->n_out, brows(m.hout, batch->n_out),
                    RAD_W(m.final_norm), brows(m.hout, batch->n_out));
        if (full > 0) {
            RAD_ISSUE_N(c, m.op_logits, full, brows(m.hout, full), RAD_W(m.output), brows(m.logits, full));
            RAD_ISSUE_N(c, m.op_logits_round, full, brows(m.logits, full), brows(m.logits_bf16, full));
            RAD_ISSUE_N(c, m.op_logits_widen, full, brows(m.logits_bf16, full), brows(m.logits, full));
        }
        if (selected)
            RAD_ISSUE_N(c, m.op_logits_select, batch->n_score_values, brows(m.hout, batch->n_out),
                        RAD_W(m.output), praw(batch->score_rows, RAD_I32, batch->n_score_values),
                        praw(batch->score_tokens, RAD_I32, batch->n_score_values),
                        praw(batch->score_values, RAD_F32, batch->n_score_values));
    }
}
}  /* namespace lfm2_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(lfm2_bf16, "lfm2", "", "0.1.0",
                "LFM2 BF16: short gated convolution, full attention and optional SigLIP2 vision")
#endif
