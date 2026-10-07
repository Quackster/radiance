/* embgemma2 -- EmbeddingGemma 2, a bidirectional Gemma encoder that pools a whole input into one
 * vector. bf16 weights and bf16 activations.
 *
 * AN EMBEDDING MODEL, NOT A GENERATIVE ONE. It declares a pooled output (rad_declare_pooled) and
 * no logits, so the engine runs it with no sampler and no KV cache, carries every input whole in
 * one step and hands each sequence's pooled row back as its answer. Attention is BIDIRECTIONAL on
 * every layer, which is why nothing is paged: a token attends to the tokens after it, so there is
 * nothing a cache could carry from one step to the next.
 *
 * ============================== THE SHAPE ==============================
 *
 *   24 layers, hidden 512, ffn 2048, vocab 262144, embedding scale sqrt(512) rounded to bf16
 *   layer_types: five `sliding_attention` then one `full_attention`, four times over
 *     sliding: 4 heads / 2 kv heads / head_dim 256, rope theta 1e4, |i - j| <= 512
 *     full:    4 heads / 1 kv head  / head_dim 512, rope theta 1e6, the whole input
 *   attention scale 1.0 (the q/k norms carry the temperature); q, k AND v are RMS-normed per head,
 *   v with no gain
 *   Gemma's sandwich norms: a norm before AND after attention and the feed-forward, all plain
 *   RMSNorm gains (x_hat * w, not 1 + w)
 *   a gated gelu-tanh feed-forward
 *   PER-LAYER EMBEDDINGS, projection-only: the token embedding projected to 24 x 512, normed, and
 *   each layer's slice gating that layer's residual through a 512 -> 512 -> 512 bottleneck
 *   a learned scalar per layer, multiplying the residual stream at the layer's end
 *   final norm, then the mean over every token of the input, then 512 -> 768
 *
 * The reference is transformers' modeling_embedding_gemma2 (model_type embedding_gemma2, which
 * rad-convert spells `embeddinggemma2`). Its mean pooling and normalisation are
 * sentence-transformers' modules on top; the normalisation is the server's (it truncates to
 * `dimensions` first), so the pooled row here is the unnormalised mean.
 *
 * ============================== WHAT THIS PLUGIN DOES NOT YET DECLARE ==============================
 *
 * Nothing the checkpoint carries: the text model, the vision tower and the audio encoder are all
 * below.
 */
#include <arch/rad_arch.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace embgemma2 {

using namespace rad::arch;

/* What a layer's attention is: the two kinds differ in head width, KV head count, rotary base and
 * window, and nothing else. */
struct Kind {
    int64_t head_dim  = 0;
    int64_t n_head    = 0;
    int64_t n_head_kv = 0;
    double  theta     = 0.0;
    int64_t window    = 0;      /* attn_dense's `window`: |i - j| < window, 0 the whole input */
    int64_t qkv() const { return (n_head + 2 * n_head_kv) * head_dim; }
    int64_t q()   const { return n_head * head_dim; }
};

struct Layer {
    int     full = 0;
    rad_weight w_in_norm = 0, w_qkv = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0, w_post_attn = 0;
    rad_weight w_pre_ff = 0, w_gate_up = 0, w_down = 0, w_post_ff = 0;
    rad_weight w_ple_gate = 0, w_ple_proj = 0, w_post_ple = 0, w_scalar = 0;
    rad_op op_in_norm = 0, op_qkv = 0, op_qk = 0, op_attn = 0, op_o = 0, op_post_attn = 0;
    rad_op op_add_attn = 0, op_pre_ff = 0, op_gate_up = 0, op_act = 0, op_down = 0;
    rad_op op_post_ff = 0, op_add_ff = 0, op_ple_gate = 0, op_ple_act = 0, op_ple_proj = 0;
    rad_op op_post_ple = 0, op_add_ple = 0, op_scale = 0;
};

/* ============================== THE VISION TOWER (Gemma 4's) ==============================
 *
 *   a patch is 16 x 16 x 3, staged by the processor as 2 * (u / 255) - 1 in bf16, (y, x, channel)
 *   patch embedding: a 768 x 768 linear, plus a learned row for the patch's column and one for its
 *   row (a [2, 10240, 768] table), the two summed first
 *   16 layers: Gemma's sandwich norms, q/k RMS-normed with a gain and v without one, a 2-D rotary
 *   (theta 100) with x over the first half of each 64-wide head and y over the second, each half a
 *   rotary of its own; bidirectional attention within an image at scale 1; a gated gelu-tanh MLP
 *   pooling: the mean of each 3 x 3 window -- the processor lays a window's nine patches out
 *   consecutively -- times sqrt(768), then the multimodal embedder: an RMS norm with no gain and a
 *   768 -> 512 linear. Its rows take the place of the image's placeholder tokens BEFORE the
 *   per-layer inputs are projected, so the per-layer inputs of an image token are its image row's.
 *
 * One encoder pass carries RadBuildCtx::max_enc_patches patches over any number of segments (an
 * image, or one frame of a video). */
struct Vision {
    struct Layer {
        rad_weight in_norm = 0, qkv = 0, q_norm = 0, k_norm = 0, o = 0, post_attn = 0;
        rad_weight pre_ff = 0, gate_up = 0, down = 0, post_ff = 0;
        rad_op op_in = 0, op_qkv = 0, op_qk = 0, op_attn = 0, op_o = 0, op_pa = 0, op_add1 = 0;
        rad_op op_pf = 0, op_gu = 0, op_act = 0, op_down = 0, op_pff = 0, op_add2 = 0;
    };
    bool on = false;
    int64_t P = 0, R = 0, width = 0, heads = 0, hd = 0, ff = 0, patch_dim = 0, pool = 0;
    int64_t npos = 0, n_embd = 0;
    rad_weight w_in = 0, w_px = 0, w_py = 0, w_proj = 0;
    rad_op op_patch = 0, op_px = 0, op_py = 0, op_padd = 0, op_add = 0, op_pool = 0, op_norm = 0;
    rad_op op_out = 0;
    rad_buf b_x = 0, b_h = 0, b_o = 0, b_qkv = 0, b_att = 0, b_gu = 0, b_ff = 0, b_px = 0,
            b_py = 0, b_pool = 0, b_pn = 0, b_out = 0;
    std::vector<Layer> L;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "vision_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_vision");
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group) {
        on = false;
        if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;
        width = rad_meta_geti(m, "vision_config.hidden_size", 0);
        heads = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
        hd    = rad_meta_geti(m, "vision_config.head_dim", 0);
        ff    = rad_meta_geti(m, "vision_config.intermediate_size", 0);
        pool  = rad_meta_geti(m, "vision_config.pooling_kernel_size", 0);
        npos  = rad_meta_geti(m, "vision_config.position_embedding_size", 0);
        const int64_t depth = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
        const int64_t patch = rad_meta_geti(m, "vision_config.patch_size", 0);
        const int64_t kvh   = rad_meta_geti(m, "vision_config.num_key_value_heads", heads);
        const double  eps   = rad_meta_getf(m, "vision_config.rms_norm_eps", 1e-6);
        const double  theta = rad_meta_getf(m, "vision_config.rope_parameters.rope_theta", 0.0);
        const char*   rt    = rad_meta_gets(m, "vision_config.rope_parameters.rope_type", "");
        const char*   act   = rad_meta_gets(m, "vision_config.hidden_activation", "");
        if (width <= 0 || heads <= 0 || hd <= 0 || ff <= 0 || pool <= 0 || npos <= 0 ||
            depth <= 0 || patch <= 0 || !(theta > 0) || heads * hd != width || kvh != heads ||
            hd % 4) {
            fprintf(stderr, "radiance: embgemma2: the container's vision_config is incomplete or "
                            "not the shape this plugin serves (hidden %lld, heads %lld x %lld, kv "
                            "%lld, ff %lld, pooling %lld, layers %lld)\n", (long long)width,
                    (long long)heads, (long long)hd, (long long)kvh, (long long)ff,
                    (long long)pool, (long long)depth);
            return RAD_E_INVAL;
        }
        /* What this tower does not carry is refused by name: clipped linears (their bounds are
         * buffers the vision checkpoint states as infinite), feature standardisation, and any
         * rotary or activation other than the axial one and gelu-tanh. */
        if (rad_meta_geti(m, "vision_config.use_clipped_linears", 0) ||
            rad_meta_geti(m, "vision_config.standardize", 0) || std::strcmp(rt, "axial") ||
            std::strcmp(act, "gelu_pytorch_tanh")) {
            fprintf(stderr, "radiance: embgemma2: this vision tower uses clipped linears, "
                            "standardisation, a '%s' rotary or a '%s' activation, which this "
                            "plugin does not serve\n", rt, act);
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        patch_dim = 3 * patch * patch;
        const int64_t kk = pool * pool;
        P = ctx->max_enc_patches;
        if (P % kk) P -= P % kk;
        R = P / kk;
        const int A = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        const char* V = "vision_tower";

        /* ---- weights: the stem, the position tables and the embedder are model-level; each
         * tower layer is its own group, past every text layer. */
        RAD_ARCH_TRY(map_copy(b, nm.f("v.patch.weight"), nm.ckpt("%s.patch_embedder.input_proj.weight", V)));
        RAD_ARCH_TRY(map_slice(b, nm.f("v.pos_x.weight"),
                               nm.ckpt("%s.patch_embedder.position_embedding_table", V), 0));
        RAD_ARCH_TRY(map_slice(b, nm.f("v.pos_y.weight"),
                               nm.ckpt("%s.patch_embedder.position_embedding_table", V), 1));
        RAD_ARCH_TRY(map_copy(b, nm.f("v.embed.weight"), nm.ckpt("embed_vision.embedding_projection.weight")));
        w_in   = decl_w(b, nm.f("v.patch.weight"), RAD_BF16, {width, patch_dim}, A, N, grp_model());
        w_px   = decl_w(b, nm.f("v.pos_x.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_py   = decl_w(b, nm.f("v.pos_y.weight"), RAD_BF16, {npos, width}, A, N, grp_model());
        w_proj = decl_w(b, nm.f("v.embed.weight"), RAD_BF16, {n_embd, width}, A, N, grp_model());
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const long long li = (long long)i;
            const RadWeightGroup g = grp_layer(first_group + (int)i);
            auto src = [&](const char* s) { return nm.ckpt("%s.encoder.layers.%lld.%s", V, li, s); };
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.in_norm", li), src("input_layernorm.weight")));
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%lld.qkv", li),
                                    {src("self_attn.q_proj.linear.weight"),
                                     src("self_attn.k_proj.linear.weight"),
                                     src("self_attn.v_proj.linear.weight")}));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.q_norm", li), src("self_attn.q_norm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.k_norm", li), src("self_attn.k_norm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.o", li), src("self_attn.o_proj.linear.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.post_attn", li), src("post_attention_layernorm.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.pre_ff", li), src("pre_feedforward_layernorm.weight")));
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%lld.gate_up", li),
                                    {src("mlp.gate_proj.linear.weight"),
                                     src("mlp.up_proj.linear.weight")}));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.down", li), src("mlp.down_proj.linear.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("v.blk.%lld.post_ff", li), src("post_feedforward_layernorm.weight")));
            l.in_norm   = decl_w(b, nm.f("v.blk.%lld.in_norm", li),   RAD_BF16, {width}, A, N, g);
            l.qkv       = decl_w(b, nm.f("v.blk.%lld.qkv", li),       RAD_BF16, {3 * width, width}, A, N, g);
            l.q_norm    = decl_w(b, nm.f("v.blk.%lld.q_norm", li),    RAD_BF16, {hd}, A, N, g);
            l.k_norm    = decl_w(b, nm.f("v.blk.%lld.k_norm", li),    RAD_BF16, {hd}, A, N, g);
            l.o         = decl_w(b, nm.f("v.blk.%lld.o", li),         RAD_BF16, {width, width}, A, N, g);
            l.post_attn = decl_w(b, nm.f("v.blk.%lld.post_attn", li), RAD_BF16, {width}, A, N, g);
            l.pre_ff    = decl_w(b, nm.f("v.blk.%lld.pre_ff", li),    RAD_BF16, {width}, A, N, g);
            l.gate_up   = decl_w(b, nm.f("v.blk.%lld.gate_up", li),   RAD_BF16, {2 * ff, width}, A, N, g);
            l.down      = decl_w(b, nm.f("v.blk.%lld.down", li),      RAD_BF16, {width, ff}, A, N, g);
            l.post_ff   = decl_w(b, nm.f("v.blk.%lld.post_ff", li),   RAD_BF16, {width}, A, N, g);
        }

        /* ---- activations, sized by the pass. */
        b_x    = decl_b(b, nm.f("v.x"),    RAD_BF16, {P, width});
        b_h    = decl_b(b, nm.f("v.h"),    RAD_BF16, {P, width});
        b_o    = decl_b(b, nm.f("v.o"),    RAD_BF16, {P, width});
        b_qkv  = decl_b(b, nm.f("v.qkv"),  RAD_BF16, {P, 3 * width});
        b_att  = decl_b(b, nm.f("v.att"),  RAD_BF16, {P, width});
        b_gu   = decl_b(b, nm.f("v.gu"),   RAD_BF16, {P, 2 * ff});
        b_ff   = decl_b(b, nm.f("v.ff"),   RAD_BF16, {P, ff});
        b_px   = decl_b(b, nm.f("v.px"),   RAD_BF16, {P, width});
        b_py   = decl_b(b, nm.f("v.py"),   RAD_BF16, {P, width});
        b_pool = decl_b(b, nm.f("v.pool"), RAD_BF16, {R, width});
        b_pn   = decl_b(b, nm.f("v.pn"),   RAD_BF16, {R, width});
        b_out  = decl_b(b, nm.f("v.out"),  RAD_BF16, {R, n_embd});

        auto rms = [&](int64_t rows, rad_weight w) {
            if (!w)
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                                       RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                       RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS);
            return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                                   RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto gemm = [&](int64_t rows, int64_t Nn, int64_t K, rad_weight w) {
            return RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", Nn),
                                                   RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto add = [&](int64_t rows) {
            return RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", width),
                                               RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        auto lookup = [&](rad_weight w) {
            return RAD_OP(b, "embed_lookup",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n_embd", width),
                                     RAD_INT("n_vocab", npos), RAD_STR("dtype", "bf16"),
                                     RAD_INT("vocab_offset", 0), RAD_F64("mult", 1.0)),
                          RAD_WEIGHTS(w));
        };
        op_patch = rw(b, gemm(P, width, patch_dim, w_in), {}, {b_x});
        op_px    = rw(b, lookup(w_px), {}, {b_px});
        op_py    = rw(b, lookup(w_py), {}, {b_py});
        op_padd  = rw(b, add(P), {b_px, b_py}, {b_px});
        op_add   = rw(b, add(P), {b_x, b_px}, {b_x});
        char sec[48];
        std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(hd / 4), (long long)(hd / 4));
        for (Layer& l : L) {
            l.op_in   = rw(b, rms(P, l.in_norm), {b_x}, {b_h});
            l.op_qkv  = rw(b, gemm(P, 3 * width, width, l.qkv), {b_h}, {b_qkv});
            /* x is the column (plane 1 of the staged coordinates), y the row (plane 0). */
            l.op_qk   = rw(b, RAD_OP(b, "qk_norm_rope",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                                RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                                RAD_F64("theta", theta), RAD_F64("eps", eps),
                                                RAD_STR("mode", "blocks"), RAD_INT("v_norm", 1),
                                                RAD_STR("sections", sec), RAD_STR("comps", "1 0")),
                                     RAD_WEIGHTS(l.q_norm, l.k_norm)),
                           {b_qkv}, {b_qkv});
            l.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                                RAD_STR("dtype", "bf16"), RAD_F64("scale", 1.0),
                                                RAD_INT("max_seqlen", P), RAD_INT("window", 0)),
                                     RAD_NOWEIGHTS),
                           {b_qkv}, {b_att});
            l.op_o    = rw(b, gemm(P, width, width, l.o), {b_att}, {b_o});
            l.op_pa   = rw(b, rms(P, l.post_attn), {b_o}, {b_o});
            l.op_add1 = rw(b, add(P), {b_x, b_o}, {b_x});
            l.op_pf   = rw(b, rms(P, l.pre_ff), {b_x}, {b_h});
            l.op_gu   = rw(b, gemm(P, 2 * ff, width, l.gate_up), {b_h}, {b_gu});
            l.op_act  = rw(b, RAD_OP(b, "act_mul",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff),
                                                RAD_STR("act", "gelu_tanh"), RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           {b_gu}, {b_ff});
            l.op_down = rw(b, gemm(P, width, ff, l.down), {b_ff}, {b_o});
            l.op_pff  = rw(b, rms(P, l.post_ff), {b_o}, {b_o});
            l.op_add2 = rw(b, add(P), {b_x, b_o}, {b_x});
        }
        op_pool = rw(b, RAD_OP(b, "pool_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_INT("k", kk), RAD_F64("mult", std::sqrt((double)width)),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {b_x}, {b_pool});
        op_norm = rw(b, rms(R, 0), {b_pool}, {b_pn});
        op_out  = rw(b, gemm(R, n_embd, width, w_proj), {b_pn}, {b_out});

        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE | RAD_MM_VIDEO;
        e.patch_dim   = patch_dim;
        e.merge       = kk;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "vision tower: %lld layers of %lld (%lld heads x %lld), ff %lld, patch %lld, "
                    "pooled %lldx%lld; %lld patches a pass -> %lld rows of %lld",
                 (long long)depth, (long long)width, (long long)heads, (long long)hd, (long long)ff,
                 (long long)patch, (long long)pool, (long long)pool, (long long)P, (long long)R,
                 (long long)n_embd);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg, kk = pool * pool;
        if (!on || Pn <= 0 || Pn > P || Pn % kk || S <= 0) {
            rad_step_fail(c, "an encoder pass this program's vision tower cannot serve: no tower, "
                             "more patches than it was declared for, or not whole pooling windows");
            return;
        }
        const int64_t Rn = Pn / kk;
        const int32_t* crd = batch->enc_coord;
        const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
        const RadOperand rc  = praw2(crd, RAD_I32, 2, Pn);
        const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);

        RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_in), brows(b_x, Pn));
        RAD_ISSUE_N(c, op_px, Pn, praw(crd + Pn, RAD_I32, Pn), RAD_W(w_px), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_py, Pn, praw(crd, RAD_I32, Pn), RAD_W(w_py), brows(b_py, Pn));
        RAD_ISSUE_N(c, op_padd, Pn, brows(b_px, Pn), brows(b_py, Pn), brows(b_px, Pn));
        RAD_ISSUE_N(c, op_add, Pn, brows(b_x, Pn), brows(b_px, Pn), brows(b_x, Pn));
        for (const Layer& l : L) {
            RAD_ISSUE_N(c, l.op_in, Pn, brows(b_x, Pn), RAD_W(l.in_norm), brows(b_h, Pn));
            RAD_ISSUE_N(c, l.op_qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv), brows(b_qkv, Pn));
            RAD_ISSUE_N(c, l.op_qk, Pn, brows(b_qkv, Pn), rc, RAD_W(l.q_norm), RAD_W(l.k_norm));
            RAD_ISSUE_N(c, l.op_attn, Pn, bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                        bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
            RAD_ISSUE_N(c, l.op_o, Pn, brows(b_att, Pn), RAD_W(l.o), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_pa, Pn, brows(b_o, Pn), RAD_W(l.post_attn), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_add1, Pn, brows(b_x, Pn), brows(b_o, Pn), brows(b_x, Pn));
            RAD_ISSUE_N(c, l.op_pf, Pn, brows(b_x, Pn), RAD_W(l.pre_ff), brows(b_h, Pn));
            RAD_ISSUE_N(c, l.op_gu, Pn, brows(b_h, Pn), RAD_W(l.gate_up), brows(b_gu, Pn));
            RAD_ISSUE_N(c, l.op_act, Pn, bcol(b_gu, 0, ff, Pn), bcol(b_gu, ff, ff, Pn),
                        brows(b_ff, Pn));
            RAD_ISSUE_N(c, l.op_down, Pn, brows(b_ff, Pn), RAD_W(l.down), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_pff, Pn, brows(b_o, Pn), RAD_W(l.post_ff), brows(b_o, Pn));
            RAD_ISSUE_N(c, l.op_add2, Pn, brows(b_x, Pn), brows(b_o, Pn), brows(b_x, Pn));
        }
        RAD_ISSUE_N(c, op_pool, Pn, brows(b_x, Pn), brows(b_pool, Rn));
        RAD_ISSUE_N(c, op_norm, Rn, brows(b_pool, Rn), RAD_NONE, brows(b_pn, Rn));
        RAD_ISSUE_N(c, op_out, Rn, brows(b_pn, Rn), RAD_W(w_proj), brows(b_out, Rn));
    }
};

/* ============================== THE AUDIO ENCODER (Gemma 4's USM conformer) ==============================
 *
 *   a frame is 128 log-mel bins (the processor's), four frames an output row
 *   subsampling: two 3x3 stride-2 convolutions over (time, frequency) -- 1 -> 128 -> 32 channels,
 *   each input masked to the clip's real frames first, each followed by a channel LayerNorm (no
 *   bias) and a ReLU -- then 32 x 32 features a row through a 1024 linear
 *   12 conformer blocks: a feed-forward at half weight, local attention (11 frames back, a
 *   relative-position term, a softplus per-dimension query scale, a softcap of 50), a gated
 *   depthwise causal convolution, a second half-weight feed-forward, a norm; their linears are
 *   CLIPPED, input and output clamped to bounds the checkpoint stores
 *   1024 -> 1536 with a bias, then the multimodal embedder: an RMS norm with no gain, 1536 -> 512.
 *
 * The blocks' clamps at +-1e10 ("gradient clipping") are left out: they bound activations ten
 * orders of magnitude past anything bf16 holds near them, so they change nothing a pass computes.
 * One clip is one segment; a pass carries up to RadBuildCtx::max_audio_frames frames. */
struct Audio {
    /* A clipped linear: the weight and its four bounds. */
    struct Clip {
        rad_weight w = 0, in_lo = 0, in_hi = 0, out_lo = 0, out_hi = 0;
        rad_op op_in = 0, op_mm = 0, op_out = 0;
    };
    struct FF {
        rad_weight pre = 0, post = 0;
        Clip l1, l2;
        rad_op op_pre = 0, op_act = 0, op_post = 0, op_add = 0;
    };
    struct Layer {
        FF ff1, ff2;
        rad_weight n_pre_attn = 0, n_post_attn = 0, n_out = 0, w_rel = 0, pds = 0;
        Clip q, k, v, post;
        rad_weight lc_pre = 0, lc_dw = 0, lc_norm = 0;
        Clip lc_start, lc_end;
        rad_op op_npa = 0, op_rel = 0, op_attn = 0, op_npo = 0, op_add_attn = 0;
        rad_op op_lc_pre = 0, op_glu = 0, op_dw = 0, op_lc_norm = 0, op_lc_act = 0, op_lc_add = 0;
        rad_op op_out = 0;
    };
    bool on = false;
    int64_t P = 0, R = 0, H = 0, heads = 0, hd = 0, ffw = 0, n_embd = 0, out_dims = 0;
    int64_t c0 = 0, c1 = 0, nmel = 0, merge = 4, left = 0, chunk = 0, n_rel = 0;
    double eps = 1e-6, softcap = 0;
    rad_weight w_c0 = 0, w_c1 = 0, w_n0 = 0, w_n1 = 0, w_in = 0, w_out = 0, b_outw = 0, w_emb = 0;
    rad_op op_c0 = 0, op_n0 = 0, op_r0 = 0, op_c1 = 0, op_n1 = 0, op_r1 = 0, op_in = 0;
    rad_op op_relpos = 0, op_outp = 0, op_en = 0, op_emb = 0;
    rad_buf b_s0 = 0, b_s1 = 0, b_x = 0, b_h = 0, b_hc = 0, b_f = 0, b_q = 0, b_k = 0, b_v = 0;
    rad_buf b_att = 0, b_o = 0, b_g = 0, b_u = 0, b_relpos = 0, b_relk = 0, b_o2 = 0, b_on = 0;
    rad_buf b_out = 0;
    std::vector<Layer> L;

    static bool present(const RadModelMeta* m) {
        const char* t = rad_meta_gets(m, "audio_config.model_type", "");
        return t && !std::strcmp(t, "gemma4_audio");
    }
    /* The longest clip in frames: 30 s at the extractor's 10 ms hop, rounded up to whole rows. */
    static int64_t frames_for(const RadModelMeta* m) {
        const int64_t rate = rad_meta_geti(m, "preprocessor.sampling_rate", 16000);
        const int64_t fl = rad_meta_geti(m, "preprocessor.frame_length", 320);
        const int64_t hop = rad_meta_geti(m, "preprocessor.hop_length", 160);
        if (rate <= 0 || hop <= 0) return 0;
        const int64_t n = 30 * rate;
        const int64_t real = (n + fl / 2 - fl - 1) / hop + 1;
        return (real + 3) / 4 * 4;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group) {
        on = false;
        if (!present(m) || ctx->max_audio_frames <= 0) return RAD_OK;
        const char* A = "audio_config";
        auto gi = [&](const char* k, int64_t d) {
            char key[96];
            std::snprintf(key, sizeof key, "%s.%s", A, k);
            return rad_meta_geti(m, key, d);
        };
        auto gf = [&](const char* k, double d) {
            char key[96];
            std::snprintf(key, sizeof key, "%s.%s", A, k);
            return rad_meta_getf(m, key, d);
        };
        H        = gi("hidden_size", 0);
        heads    = gi("num_attention_heads", 0);
        out_dims = gi("output_proj_dims", 0);
        chunk    = gi("attention_chunk_size", 0);
        /* transformers masks a key unless 0 <= q - k < attention_context_left - 1, so the
         * furthest key back is context_left - 2 frames: 11 for this checkpoint, the query and
         * eleven before it. */
        left     = gi("attention_context_left", 0) - 2;
        softcap  = gf("attention_logit_cap", 0.0);
        eps      = gf("rms_norm_eps", 1e-6);
        const int64_t depth = gi("num_hidden_layers", 0);
        const int64_t kconv = gi("conv_kernel_size", 0);
        const int64_t right = gi("attention_context_right", 0);
        const double  resw  = gf("residual_weight", 0.0);
        const char*   act   = rad_meta_gets(m, "audio_config.hidden_act", "");
        const char*   chs   = rad_meta_gets(m, "audio_config.subsampling_conv_channels", "");
        nmel = rad_meta_geti(m, "preprocessor.feature_size", 0);
        if (std::sscanf(chs ? chs : "", "%lld %lld", (long long*)&c0, (long long*)&c1) != 2)
            c0 = c1 = 0;
        hd = heads > 0 ? H / heads : 0;
        ffw = 4 * H;
        n_rel = left + 1;
        if (H <= 0 || heads <= 0 || hd * heads != H || out_dims <= 0 || depth <= 0 || kconv <= 0 ||
            chunk <= 0 || left < 0 || right != 0 || !(softcap > 0) || resw != 0.5 ||
            std::strcmp(act ? act : "", "silu") || c0 <= 0 || c1 <= 0 || nmel != 128 ||
            (nmel / 4) * c1 != H) {
            fprintf(stderr, "radiance: embgemma2: the container's audio_config is not the "
                            "conformer this plugin serves (hidden %lld, heads %lld, output %lld, "
                            "layers %lld, conv %lld, left %lld right %lld, residual %g, act %s, "
                            "subsampling %lld/%lld over %lld bins)\n", (long long)H,
                    (long long)heads, (long long)out_dims, (long long)depth, (long long)kconv,
                    (long long)left, (long long)right, resw, act ? act : "", (long long)c0,
                    (long long)c1, (long long)nmel);
            return RAD_E_UNSUPPORTED;
        }
        n_embd = model_embd;
        P = ctx->max_audio_frames;
        if (P % merge) return RAD_E_SHAPE;
        R = P / merge;
        const int AC = RAD_ACCESS_PER_REQUEST, N = RAD_SHARD_NONE;
        const char* T = "audio_tower";

        /* ---- weights */
        auto w = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                     RadWeightGroup g) -> rad_weight {
            if (map_copy(b, dn, src) < 0) return 0;
            return decl_w(b, dn, RAD_BF16, shp, AC, N, g);
        };
        auto clip = [&](Clip& c, const char* dn, const char* src, int64_t out, int64_t in,
                        RadWeightGroup g) {
            char d1[160], s1[200];
            std::snprintf(d1, sizeof d1, "%s.w", dn); std::snprintf(s1, sizeof s1, "%s.linear.weight", src);
            c.w = w(nm.f("%s", d1), nm.ckpt("%s", s1), {out, in}, g);
            const char* bn[4] = { "input_min", "input_max", "output_min", "output_max" };
            rad_weight* dst[4] = { &c.in_lo, &c.in_hi, &c.out_lo, &c.out_hi };
            for (int i = 0; i < 4; ++i) {
                std::snprintf(d1, sizeof d1, "%s.%s", dn, bn[i]);
                std::snprintf(s1, sizeof s1, "%s.%s", src, bn[i]);
                *dst[i] = w(nm.f("%s", d1), nm.ckpt("%s", s1), {1}, g);
            }
        };
        w_c0 = w(nm.f("a.sscp.c0"), nm.ckpt("%s.subsample_conv_projection.layer0.conv.weight", T), {c0, 1, 3, 3}, grp_model());
        w_n0 = w(nm.f("a.sscp.n0"), nm.ckpt("%s.subsample_conv_projection.layer0.norm.weight", T), {c0}, grp_model());
        w_c1 = w(nm.f("a.sscp.c1"), nm.ckpt("%s.subsample_conv_projection.layer1.conv.weight", T), {c1, c0, 3, 3}, grp_model());
        w_n1 = w(nm.f("a.sscp.n1"), nm.ckpt("%s.subsample_conv_projection.layer1.norm.weight", T), {c1}, grp_model());
        w_in = w(nm.f("a.sscp.proj"), nm.ckpt("%s.subsample_conv_projection.input_proj_linear.weight", T), {H, H}, grp_model());
        w_out  = w(nm.f("a.out.w"), nm.ckpt("%s.output_proj.weight", T), {out_dims, H}, grp_model());
        b_outw = w(nm.f("a.out.b"), nm.ckpt("%s.output_proj.bias", T), {out_dims}, grp_model());
        w_emb  = w(nm.f("a.embed"), nm.ckpt("embed_audio.embedding_projection.weight"), {n_embd, out_dims}, grp_model());
        L.assign((size_t)depth, Layer{});
        for (int64_t i = 0; i < depth; ++i) {
            Layer& l = L[(size_t)i];
            const long long li = (long long)i;
            const RadWeightGroup g = grp_layer(first_group + (int)i);
            char d[96], s[160];
            auto D = [&](const char* x) { std::snprintf(d, sizeof d, "a.blk.%lld.%s", li, x); return nm.f("%s", d); };
            auto S = [&](const char* x) { std::snprintf(s, sizeof s, "%s.layers.%lld.%s", T, li, x); return nm.ckpt("%s", s); };
            for (int f = 0; f < 2; ++f) {
                FF& ff = f ? l.ff2 : l.ff1;
                const char* fn = f ? "feed_forward2" : "feed_forward1";
                char px[64]; std::snprintf(px, sizeof px, "%s.pre_layer_norm.weight", fn);
                ff.pre = w(D(px), S(px), {H}, g);
                std::snprintf(px, sizeof px, "%s.post_layer_norm.weight", fn);
                ff.post = w(D(px), S(px), {H}, g);
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.%s.l1", li, fn);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.%s.ffw_layer_1", T, li, fn);
                clip(ff.l1, dn, sn, ffw, H, g);
                std::snprintf(dn, sizeof dn, "a.blk.%lld.%s.l2", li, fn);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.%s.ffw_layer_2", T, li, fn);
                clip(ff.l2, dn, sn, H, ffw, g);
            }
            l.n_pre_attn  = w(D("norm_pre_attn"), S("norm_pre_attn.weight"), {H}, g);
            l.n_post_attn = w(D("norm_post_attn"), S("norm_post_attn.weight"), {H}, g);
            l.n_out       = w(D("norm_out"), S("norm_out.weight"), {H}, g);
            l.w_rel       = w(D("rel_k"), S("self_attn.relative_k_proj.weight"), {H, H}, g);
            l.pds         = w(D("per_dim_scale"), S("self_attn.per_dim_scale"), {hd}, g);
            const char* qn[4] = { "q_proj", "k_proj", "v_proj", "post" };
            Clip* qc[4] = { &l.q, &l.k, &l.v, &l.post };
            for (int j = 0; j < 4; ++j) {
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.attn.%s", li, qn[j]);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.self_attn.%s", T, li, qn[j]);
                clip(*qc[j], dn, sn, H, H, g);
            }
            l.lc_pre  = w(D("lconv.pre"), S("lconv1d.pre_layer_norm.weight"), {H}, g);
            l.lc_dw   = w(D("lconv.dw"), S("lconv1d.depthwise_conv1d.weight"), {H, 1, kconv}, g);
            l.lc_norm = w(D("lconv.norm"), S("lconv1d.conv_norm.weight"), {H}, g);
            {
                char dn[96], sn[160];
                std::snprintf(dn, sizeof dn, "a.blk.%lld.lconv.start", li);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.lconv1d.linear_start", T, li);
                clip(l.lc_start, dn, sn, 2 * H, H, g);
                std::snprintf(dn, sizeof dn, "a.blk.%lld.lconv.end", li);
                std::snprintf(sn, sizeof sn, "%s.layers.%lld.lconv1d.linear_end", T, li);
                clip(l.lc_end, dn, sn, H, H, g);
            }
        }

        /* ---- activations: the subsampler's planes at their own widths, the blocks' at R rows */
        b_s0     = decl_b(b, nm.f("a.s0"),     RAD_BF16, {P / 2, (nmel / 2) * c0});
        b_s1     = decl_b(b, nm.f("a.s1"),     RAD_BF16, {R, (nmel / 4) * c1});
        b_x      = decl_b(b, nm.f("a.x"),      RAD_BF16, {R, H});
        b_h      = decl_b(b, nm.f("a.h"),      RAD_BF16, {R, H});
        b_hc     = decl_b(b, nm.f("a.hc"),     RAD_BF16, {R, H});
        b_f      = decl_b(b, nm.f("a.f"),      RAD_BF16, {R, ffw});
        b_q      = decl_b(b, nm.f("a.q"),      RAD_BF16, {R, H});
        b_k      = decl_b(b, nm.f("a.k"),      RAD_BF16, {R, H});
        b_v      = decl_b(b, nm.f("a.v"),      RAD_BF16, {R, H});
        b_att    = decl_b(b, nm.f("a.att"),    RAD_BF16, {R, H});
        b_o      = decl_b(b, nm.f("a.o"),      RAD_BF16, {R, H});
        b_g      = decl_b(b, nm.f("a.g"),      RAD_BF16, {R, 2 * H});
        b_u      = decl_b(b, nm.f("a.u"),      RAD_BF16, {R, H});
        b_relpos = decl_b(b, nm.f("a.relpos"), RAD_BF16, {n_rel, H});
        b_relk   = decl_b(b, nm.f("a.relk"),   RAD_BF16, {n_rel, H});
        b_o2     = decl_b(b, nm.f("a.o2"),     RAD_BF16, {R, out_dims});
        b_on     = decl_b(b, nm.f("a.on"),     RAD_BF16, {R, out_dims});
        b_out    = decl_b(b, nm.f("a.out"),    RAD_BF16, {R, n_embd});

        auto gemm = [&](int64_t rows, int64_t Nn, int64_t K, rad_weight wt) {
            return RAD_OP(b, "gemm_nt", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", Nn),
                                                   RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(wt));
        };
        auto rms = [&](int64_t rows, int64_t n, rad_weight wt, double mult) {
            if (!wt)
                return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                       RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                       RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS);
            return RAD_OP(b, "rmsnorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                   RAD_F64("eps", eps), RAD_F64("wadd", 0.0),
                                                   RAD_STR("dtype", "bf16"), RAD_F64("mult", mult)),
                          RAD_WEIGHTS(wt));
        };
        auto clamp = [&](int64_t rows, int64_t n, rad_weight lo, rad_weight hi) {
            return RAD_OP(b, "clamp", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                 RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(lo, hi));
        };
        auto unary = [&](const char* op, int64_t rows, int64_t n) {
            return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                            RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
        };
        auto add = [&](int64_t rows, int64_t n) { return unary("add", rows, n); };
        auto lnorm = [&](int64_t rows, int64_t n, rad_weight g) {
            return RAD_OP(b, "layernorm", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                                     RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(g));
        };
        auto conv = [&](int64_t rows, int64_t width, int64_t ci, int64_t co, int64_t cu_div,
                        int64_t vstride, rad_weight wt) {
            return RAD_OP(b, "conv2d", RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("width", width),
                                                  RAD_INT("cin", ci), RAD_INT("cout", co),
                                                  RAD_INT("kh", 3), RAD_INT("kw", 3),
                                                  RAD_INT("stride", 2), RAD_INT("pad", 1),
                                                  RAD_INT("cu_div", cu_div),
                                                  RAD_INT("valid_stride", vstride),
                                                  RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(wt));
        };
        /* A clipped linear over rows of `in` into `out`: the input clamp writes a copy (the input
         * is shared by the projections it feeds, each with its own bounds), the output clamp is in
         * place. The copy is a whole [R, n_in] buffer: a GEMM's input must be contiguous, and a
         * column slice of a wider one is not. */
        auto clipped = [&](Clip& c, int64_t n_out, int64_t n_in, rad_buf in, rad_buf tmp,
                           rad_buf out) {
            c.op_in  = rw(b, clamp(R, n_in, c.in_lo, c.in_hi), {in}, {tmp});
            c.op_mm  = rw(b, gemm(R, n_out, n_in, c.w), {tmp}, {out});
            c.op_out = rw(b, clamp(R, n_out, c.out_lo, c.out_hi), {out}, {out});
        };

        op_c0 = rw(b, conv(P, nmel, 1, c0, 1, 1, w_c0), {}, {b_s0});
        op_n0 = rw(b, lnorm(P / 2 * (nmel / 2), c0, w_n0), {b_s0}, {b_s0});
        op_r0 = rw(b, unary("relu", P / 2 * (nmel / 2), c0), {b_s0}, {b_s0});
        op_c1 = rw(b, conv(P / 2, nmel / 2, c0, c1, 2, 2, w_c1), {b_s0}, {b_s1});
        op_n1 = rw(b, lnorm(R * (nmel / 4), c1, w_n1), {b_s1}, {b_s1});
        op_r1 = rw(b, unary("relu", R * (nmel / 4), c1), {b_s1}, {b_s1});
        op_in = rw(b, gemm(R, H, H, w_in), {b_s1}, {b_x});
        op_relpos = rw(b, RAD_OP(b, "rel_sinusoid",
                                 RAD_PARAMS(RAD_RANGE("M", 1, n_rel), RAD_INT("n", H),
                                            RAD_F64("max_timescale", 10000.0),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       {}, {b_relpos});
        for (Layer& l : L) {
            auto ffw_block = [&](FF& ff) {
                ff.op_pre = rw(b, rms(R, H, ff.pre, 1.0), {b_x}, {b_h});
                clipped(ff.l1, ffw, H, b_h, b_hc, b_f);
                ff.op_act = rw(b, unary("silu", R, ffw), {b_f}, {b_f});
                /* the activation's only reader is this linear, so its clamp is in place */
                clipped(ff.l2, H, ffw, b_f, b_f, b_o);
                ff.op_post = rw(b, rms(R, H, ff.post, 0.5), {b_o}, {b_o});
                ff.op_add  = rw(b, add(R, H), {b_x, b_o}, {b_x});
            };
            ffw_block(l.ff1);
            l.op_npa = rw(b, rms(R, H, l.n_pre_attn, 1.0), {b_x}, {b_h});
            clipped(l.q, H, H, b_h, b_hc, b_q);
            clipped(l.k, H, H, b_h, b_hc, b_k);
            clipped(l.v, H, H, b_h, b_hc, b_v);
            l.op_rel  = rw(b, gemm(n_rel, H, H, l.w_rel), {b_relpos}, {b_relk});
            l.op_attn = rw(b, RAD_OP(b, "attn_local_rel",
                                     RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                                RAD_INT("n_head", heads), RAD_INT("left", left),
                                                RAD_INT("right", 0), RAD_F64("softcap", softcap),
                                                RAD_INT("cu_div", merge), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(l.pds)),
                           {b_q, b_k, b_v, b_relk}, {b_att});
            clipped(l.post, H, H, b_att, b_hc, b_o);
            l.op_npo = rw(b, rms(R, H, l.n_post_attn, 1.0), {b_o}, {b_o});
            l.op_add_attn = rw(b, add(R, H), {b_x, b_o}, {b_x});
            l.op_lc_pre = rw(b, rms(R, H, l.lc_pre, 1.0), {b_x}, {b_h});
            clipped(l.lc_start, 2 * H, H, b_h, b_hc, b_g);
            /* GLU: the first half times the sigmoid of the second. */
            l.op_glu = rw(b, RAD_OP(b, "act_mul",
                                    RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                               RAD_STR("act", "sigmoid"), RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          {b_g}, {b_u});
            l.op_dw = rw(b, RAD_OP(b, "conv1d_dw",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", H),
                                              RAD_INT("k", kconv), RAD_INT("cu_div", merge),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(l.lc_dw)),
                         {b_u}, {b_o});
            l.op_lc_norm = rw(b, rms(R, H, l.lc_norm, 1.0), {b_o}, {b_o});
            l.op_lc_act  = rw(b, unary("silu", R, H), {b_o}, {b_o});
            clipped(l.lc_end, H, H, b_o, b_hc, b_u);
            l.op_lc_add = rw(b, add(R, H), {b_x, b_u}, {b_x});
            ffw_block(l.ff2);
            l.op_out = rw(b, rms(R, H, l.n_out, 1.0), {b_x}, {b_x});
        }
        op_outp = rw(b, RAD_OP(b, "gemm_nt_bias",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", out_dims),
                                          RAD_INT("K", H), RAD_STR("dtype", "bf16"),
                                          RAD_STR("act", "none")),
                               RAD_WEIGHTS(w_out, b_outw)),
                     {b_x}, {b_o2});
        op_en  = rw(b, rms(R, out_dims, 0, 1.0), {b_o2}, {b_on});
        op_emb = rw(b, gemm(R, n_embd, out_dims, w_emb), {b_on}, {b_out});

        RadEncoderDecl e{};
        e.name        = "audio";
        e.modalities  = RAD_MM_AUDIO;
        e.patch_dim   = nmel;
        e.merge       = merge;
        e.max_patches = P;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "audio encoder: %lld conformer blocks of %lld (%lld heads x %lld), "
                    "subsampling %lld/%lld channels, local attention %lld back; %lld frames a "
                    "pass -> %lld rows of %lld", (long long)depth, (long long)H, (long long)heads,
                 (long long)hd, (long long)c0, (long long)c1, (long long)left, (long long)P,
                 (long long)R, (long long)n_embd);
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
        if (!on || Pn <= 0 || Pn > P || Pn % merge || S <= 0) {
            rad_step_fail(c, "an audio pass this program's audio encoder cannot serve: none "
                             "declared, more frames than it was declared for, or not whole rows");
            return;
        }
        const int64_t Rn = Pn / merge;
        const int32_t* crd = batch->enc_coord;
        const RadOperand frames = praw2(batch->enc_pixels, RAD_BF16, Pn, nmel);
        const RadOperand valid = praw(crd + Pn, RAD_I32, Pn);
        const RadOperand cu = praw(batch->enc_cu, RAD_I32, S + 1);
        auto clipped = [&](const Clip& cl, rad_buf in, int64_t n_in, rad_buf tmp, rad_buf out,
                           int64_t n_out) {
            RAD_ISSUE_N(c, cl.op_in, Rn, bcol(in, 0, n_in, Rn), RAD_W(cl.in_lo), RAD_W(cl.in_hi),
                        bcol(tmp, 0, n_in, Rn));
            RAD_ISSUE_N(c, cl.op_mm, Rn, bcol(tmp, 0, n_in, Rn), RAD_W(cl.w), bcol(out, 0, n_out, Rn));
            RAD_ISSUE_N(c, cl.op_out, Rn, bcol(out, 0, n_out, Rn), RAD_W(cl.out_lo), RAD_W(cl.out_hi),
                        bcol(out, 0, n_out, Rn));
        };

        RAD_ISSUE_N(c, op_c0, Pn, frames, RAD_W(w_c0), cu, valid, brows(b_s0, Pn / 2));
        RAD_ISSUE_N(c, op_n0, Pn / 2 * (nmel / 2), brows(b_s0, Pn / 2), RAD_W(w_n0), RAD_NONE,
                    brows(b_s0, Pn / 2));
        RAD_ISSUE_N(c, op_r0, Pn / 2 * (nmel / 2), brows(b_s0, Pn / 2), brows(b_s0, Pn / 2));
        RAD_ISSUE_N(c, op_c1, Pn / 2, brows(b_s0, Pn / 2), RAD_W(w_c1), cu, valid, brows(b_s1, Rn));
        RAD_ISSUE_N(c, op_n1, Rn * (nmel / 4), brows(b_s1, Rn), RAD_W(w_n1), RAD_NONE,
                    brows(b_s1, Rn));
        RAD_ISSUE_N(c, op_r1, Rn * (nmel / 4), brows(b_s1, Rn), brows(b_s1, Rn));
        RAD_ISSUE_N(c, op_in, Rn, brows(b_s1, Rn), RAD_W(w_in), brows(b_x, Rn));
        RAD_ISSUE_N(c, op_relpos, n_rel, brows(b_relpos, n_rel));
        for (const Layer& l : L) {
            auto ffw_block = [&](const FF& ff) {
                RAD_ISSUE_N(c, ff.op_pre, Rn, brows(b_x, Rn), RAD_W(ff.pre), brows(b_h, Rn));
                clipped(ff.l1, b_h, H, b_hc, b_f, ffw);
                RAD_ISSUE_N(c, ff.op_act, Rn, brows(b_f, Rn), brows(b_f, Rn));
                clipped(ff.l2, b_f, ffw, b_f, b_o, H);
                RAD_ISSUE_N(c, ff.op_post, Rn, brows(b_o, Rn), RAD_W(ff.post), brows(b_o, Rn));
                RAD_ISSUE_N(c, ff.op_add, Rn, brows(b_x, Rn), brows(b_o, Rn), brows(b_x, Rn));
            };
            ffw_block(l.ff1);
            RAD_ISSUE_N(c, l.op_npa, Rn, brows(b_x, Rn), RAD_W(l.n_pre_attn), brows(b_h, Rn));
            clipped(l.q, b_h, H, b_hc, b_q, H);
            clipped(l.k, b_h, H, b_hc, b_k, H);
            clipped(l.v, b_h, H, b_hc, b_v, H);
            RAD_ISSUE_N(c, l.op_rel, n_rel, brows(b_relpos, n_rel), RAD_W(l.w_rel),
                        brows(b_relk, n_rel));
            RAD_ISSUE_N(c, l.op_attn, Rn, brows(b_q, Rn), brows(b_k, Rn), brows(b_v, Rn),
                        brows(b_relk, n_rel), RAD_W(l.pds), cu, brows(b_att, Rn));
            clipped(l.post, b_att, H, b_hc, b_o, H);
            RAD_ISSUE_N(c, l.op_npo, Rn, brows(b_o, Rn), RAD_W(l.n_post_attn), brows(b_o, Rn));
            RAD_ISSUE_N(c, l.op_add_attn, Rn, brows(b_x, Rn), brows(b_o, Rn), brows(b_x, Rn));
            RAD_ISSUE_N(c, l.op_lc_pre, Rn, brows(b_x, Rn), RAD_W(l.lc_pre), brows(b_h, Rn));
            clipped(l.lc_start, b_h, H, b_hc, b_g, 2 * H);
            RAD_ISSUE_N(c, l.op_glu, Rn, bcol(b_g, H, H, Rn), bcol(b_g, 0, H, Rn), brows(b_u, Rn));
            RAD_ISSUE_N(c, l.op_dw, Rn, brows(b_u, Rn), RAD_W(l.lc_dw), cu, brows(b_o, Rn));
            RAD_ISSUE_N(c, l.op_lc_norm, Rn, brows(b_o, Rn), RAD_W(l.lc_norm), brows(b_o, Rn));
            RAD_ISSUE_N(c, l.op_lc_act, Rn, brows(b_o, Rn), brows(b_o, Rn));
            clipped(l.lc_end, b_o, H, b_hc, b_u, H);
            RAD_ISSUE_N(c, l.op_lc_add, Rn, brows(b_x, Rn), brows(b_u, Rn), brows(b_x, Rn));
            ffw_block(l.ff2);
            RAD_ISSUE_N(c, l.op_out, Rn, brows(b_x, Rn), RAD_W(l.n_out), brows(b_x, Rn));
        }
        RAD_ISSUE_N(c, op_outp, Rn, brows(b_x, Rn), RAD_W(w_out), RAD_W(b_outw), RAD_NONE,
                    brows(b_o2, Rn));
        RAD_ISSUE_N(c, op_en, Rn, brows(b_o2, Rn), RAD_NONE, brows(b_on, Rn));
        RAD_ISSUE_N(c, op_emb, Rn, brows(b_on, Rn), RAD_W(w_emb), brows(b_out, Rn));
    }
};

struct Model {
    Names nm{""};
    Geom  g{};
    Kind  ks{}, kf{};           /* sliding, full */
    int64_t n_ple = 0;          /* hidden_size_per_layer_input */
    int64_t n_out = 0;          /* embedding_dim */

    rad_buf b_x = 0, b_h = 0, b_o = 0, b_qkv_s = 0, b_qkv_f = 0, b_att_s = 0, b_att_f = 0;
    rad_buf b_gu = 0, b_ff = 0, b_p = 0, b_ple = 0, b_mean = 0, b_proj = 0, b_pooled = 0;

    rad_weight w_tok = 0, w_ple_proj = 0, w_ple_norm = 0, w_norm = 0, w_out = 0;
    rad_op op_embed = 0, op_ple = 0, op_ple_norm = 0, op_norm = 0, op_mean = 0, op_out = 0;
    rad_op op_cast = 0;
    rad_op op_mm = 0;           /* the media rows over their placeholders' embeddings */

    std::vector<Layer> layers;
    Vision vis;
    Audio aud;
};

static Model g_model[MAX_RANKS];

/* A value as a bf16 tensor holds it: round to nearest even on the top 16 bits of the f32. */
static double bf16_round(double v) {
    float f = (float)v;
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&f, &u, 4);
    return (double)f;
}

/* The layer schedule, read and not derived: layer_types is the checkpoint's statement of which
 * layers attend to the whole input, and a pattern assumed from the model card would build a model
 * that runs and pools the wrong function if a sibling checkpoint moved one. */
static int read_schedule(const RadModelMeta* meta, int64_t n_layer, std::vector<int>* full) {
    const char* s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s || !*s) {
        fprintf(stderr, "radiance: embgemma2: the container has no layer_types; it says which "
                        "layers are sliding and which see the whole input, and is not guessed\n");
        return RAD_E_INVAL;
    }
    full->clear();
    for (const char* p = s; *p; ) {
        while (*p == ' ') ++p;
        if (!*p) break;
        const char* t = p;
        while (*p && *p != ' ') ++p;
        const size_t len = (size_t)(p - t);
        if (len == 14 && !std::strncmp(t, "full_attention", 14))         full->push_back(1);
        else if (len == 17 && !std::strncmp(t, "sliding_attention", 17)) full->push_back(0);
        else {
            fprintf(stderr, "radiance: embgemma2: layer_types names '%.*s'; this plugin knows "
                            "full_attention and sliding_attention\n", (int)len, t);
            return RAD_E_UNSUPPORTED;
        }
    }
    if ((int64_t)full->size() != n_layer) {
        fprintf(stderr, "radiance: embgemma2: layer_types names %zu layers and the model has "
                        "%lld\n", full->size(), (long long)n_layer);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    /* A 270M encoder is a fraction of one card; splitting it would cost an all-reduce a layer to
     * save nothing, so it is not offered. */
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: embgemma2 runs on one card (--tp 1); the text model is 270M "
                        "parameters\n");
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd;

    std::vector<int> full;
    RAD_ARCH_TRY(read_schedule(meta, g.n_layer, &full));

    /* The two attention kinds. The sliding one is the config's top-level geometry; the full one
     * overrides head width and KV heads per layer (per_layer_config), and every full layer of
     * this checkpoint states the same override -- checked below, layer by layer. */
    m.ks.head_dim  = g.head_dim;
    m.ks.n_head    = g.n_head;
    m.ks.n_head_kv = g.n_head_kv;
    m.ks.theta     = rad_meta_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    /* A RADIUS in the reference (|i - j| <= sliding_window), and attn_dense's window is the
     * strict bound, so one more. */
    const int64_t radius = meta_int2(meta, "", "sliding_window", 0);
    m.ks.window = radius + 1;
    m.kf.n_head = g.n_head;
    m.kf.theta  = rad_meta_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    m.kf.window = 0;
    for (int64_t l = 0; l < g.n_layer; ++l) {
        if (!full[(size_t)l]) continue;
        char k1[96], k2[96];
        std::snprintf(k1, sizeof k1, "per_layer_config.%02lld.head_dim", (long long)l);
        std::snprintf(k2, sizeof k2, "per_layer_config.%02lld.num_key_value_heads", (long long)l);
        const int64_t hd = meta_int2(meta, "", k1, 0), kv = meta_int2(meta, "", k2, 0);
        if (!m.kf.head_dim) { m.kf.head_dim = hd; m.kf.n_head_kv = kv; }
        if (hd <= 0 || kv <= 0 || hd != m.kf.head_dim || kv != m.kf.n_head_kv) {
            fprintf(stderr, "radiance: embgemma2: full-attention layer %lld states head_dim %lld "
                            "and %lld KV heads; this plugin serves one full geometry\n",
                    (long long)l, (long long)hd, (long long)kv);
            return RAD_E_INVAL;
        }
    }
    m.n_ple = meta_int2(meta, "", "hidden_size_per_layer_input", 0);
    m.n_out = meta_int2(meta, "", "embedding_dim", 0);
    if (!(m.ks.theta > 0) || !(m.kf.theta > 0) || radius <= 0 || m.kf.head_dim <= 0 ||
        m.n_ple <= 0 || m.n_out <= 0 || m.ks.n_head % m.ks.n_head_kv || m.kf.n_head % m.kf.n_head_kv) {
        fprintf(stderr, "radiance: embgemma2: the container does not describe the model (rope "
                        "thetas %g / %g, sliding_window %lld, full head_dim %lld, "
                        "hidden_size_per_layer_input %lld, embedding_dim %lld)\n",
                m.ks.theta, m.kf.theta, (long long)radius, (long long)m.kf.head_dim,
                (long long)m.n_ple, (long long)m.n_out);
        return RAD_E_INVAL;
    }
    const int64_t P = m.n_ple * g.n_layer;
    /* ONE INPUT IS AT MOST ONE STEP (Request::embed), so the longest attention segment is the
     * step's token budget. */
    const int64_t max_seg = T;

    /* ---- activations. ple holds every layer's per-layer input for the whole step, read a
     * column slice a layer. */
    m.b_x     = decl_b(b, nm.f("x"),       RAD_BF16, {T, E});
    m.b_h     = decl_b(b, nm.f("h"),       RAD_BF16, {T, E});
    m.b_o     = decl_b(b, nm.f("o"),       RAD_BF16, {T, E});
    m.b_qkv_s = decl_b(b, nm.f("qkv_s"),   RAD_BF16, {T, m.ks.qkv()});
    m.b_qkv_f = decl_b(b, nm.f("qkv_f"),   RAD_BF16, {T, m.kf.qkv()});
    m.b_att_s = decl_b(b, nm.f("att_s"),   RAD_BF16, {T, m.ks.q()});
    m.b_att_f = decl_b(b, nm.f("att_f"),   RAD_BF16, {T, m.kf.q()});
    m.b_gu    = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * g.n_ff});
    m.b_ff    = decl_b(b, nm.f("ffn"),     RAD_BF16, {T, g.n_ff});
    m.b_p     = decl_b(b, nm.f("ple_p"),   RAD_BF16, {T, m.n_ple});
    m.b_ple   = decl_b(b, nm.f("ple"),     RAD_BF16, {T, P});
    m.b_mean  = decl_b(b, nm.f("mean"),    RAD_BF16, {g.max_seqs, E});
    m.b_proj  = decl_b(b, nm.f("proj"),    RAD_BF16, {g.max_seqs, m.n_out});
    m.b_pooled= decl_b(b, nm.f("pooled"),  RAD_F32,  {g.max_seqs, m.n_out}, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(rad_declare_pooled(b, m.b_pooled));

    /* ---- the model-level weights. Every one at the checkpoint's own bf16, the norm gains
     * included: the reference widens a gain to f32 at use, which is exact, so storing it wider
     * buys nothing and costs reading it in place on the host backend. */
    const char* L = "language_model";
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("ple_proj.weight"),
                          nm.ckpt("%s.ple.per_layer_model_projection.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("ple_norm.weight"),
                          nm.ckpt("%s.ple.per_layer_projection_norm.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("embd_proj.weight"),
                          nm.ckpt("%s.embedding_projection.weight", L)));
    m.w_tok      = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E},
                          RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_ple_proj = decl_w(b, nm.f("ple_proj.weight"), RAD_BF16, {P, E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_ple_norm = decl_w(b, nm.f("ple_norm.weight"), RAD_BF16, {m.n_ple},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_norm     = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_out      = decl_w(b, nm.f("embd_proj.weight"), RAD_BF16, {m.n_out, E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    auto rmsnorm = [&](int64_t rows, int64_t n, double eps, rad_weight w) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                 RAD_F64("wadd", 0.0), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto elem = [&](const char* op, int64_t n, rad_weight w) {
        if (w)
            return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                            RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                        RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    auto act_mul = [&](int64_t n) {
        return RAD_OP(b, "act_mul", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                               RAD_STR("act", "gelu_tanh"),
                                               RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };

    /* The embedding, scaled by sqrt(hidden) AS THE REFERENCE ROUNDS IT: the scale is a bf16
     * buffer, so sqrt(512) = 22.6274 multiplies as 22.625. */
    const double escale = bf16_round(std::sqrt((double)E));
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", 0), RAD_F64("mult", escale)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* THE PER-LAYER INPUTS: the embedding projected to n_layer x n_ple and RMS-normed per layer.
     * The reference scales the projection by hidden^-0.5 before the norm; an RMS norm of s*x with
     * eps is exactly the norm of x with eps / s^2, which is what this declares, so the scale is
     * applied without the extra bf16 rounding of s*x the reference pays. */
    /* A media part's rows over its placeholders' embeddings, before anything reads them: the
     * per-layer inputs are projected from what the model embeds, and for an image token that is
     * its image row (declared only when a tower is). */
    if ((Vision::present(meta) && ctx->max_enc_patches > 0) ||
        (Audio::present(meta) && ctx->max_audio_frames > 0))
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});
    m.op_ple = rw(b, gemm(T, P, E, m.w_ple_proj), {m.b_x}, {m.b_ple});
    m.op_ple_norm = rw(b, rmsnorm(T * g.n_layer, m.n_ple, (double)g.eps * (double)E, m.w_ple_norm),
                       {m.b_ple}, {m.b_ple});

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.full = full[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const rad_buf qkv = y.full ? m.b_qkv_f : m.b_qkv_s;
        const rad_buf att = y.full ? m.b_att_f : m.b_att_s;
        const long long ll = (long long)l;
        const RadWeightGroup gl = grp_layer((int)l);
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%lld.%s", L, ll, s); };

        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.in_norm", ll),   src("input_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.qkv", ll),
                                {src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                                 src("self_attn.v_proj.weight")}));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.q_norm", ll),    src("self_attn.q_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.k_norm", ll),    src("self_attn.k_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.o", ll),         src("self_attn.o_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_attn", ll),
                              src("post_attention_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.pre_ff", ll),
                              src("pre_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.gate_up", ll),
                                {src("mlp.gate_proj.weight"), src("mlp.up_proj.weight")}));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.down", ll),      src("mlp.down_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_ff", ll),
                              src("post_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ple_gate", ll),
                              src("ple_block.per_layer_input_gate.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ple_proj", ll),
                              src("ple_block.per_layer_projection.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_ple", ll),
                              src("ple_block.post_per_layer_input_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.scalar", ll),    src("layer_scalar")));

        const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
        y.w_in_norm   = decl_w(b, nm.f("blk.%lld.in_norm", ll),   RAD_BF16, {E}, A, N, gl);
        y.w_qkv       = decl_w(b, nm.f("blk.%lld.qkv", ll),       RAD_BF16, {k.qkv(), E}, A, N, gl);
        y.w_q_norm    = decl_w(b, nm.f("blk.%lld.q_norm", ll),    RAD_BF16, {k.head_dim}, A, N, gl);
        y.w_k_norm    = decl_w(b, nm.f("blk.%lld.k_norm", ll),    RAD_BF16, {k.head_dim}, A, N, gl);
        y.w_o         = decl_w(b, nm.f("blk.%lld.o", ll),         RAD_BF16, {E, k.q()}, A, N, gl);
        y.w_post_attn = decl_w(b, nm.f("blk.%lld.post_attn", ll), RAD_BF16, {E}, A, N, gl);
        y.w_pre_ff    = decl_w(b, nm.f("blk.%lld.pre_ff", ll),    RAD_BF16, {E}, A, N, gl);
        y.w_gate_up   = decl_w(b, nm.f("blk.%lld.gate_up", ll),   RAD_BF16, {2 * g.n_ff, E}, A, N, gl);
        y.w_down      = decl_w(b, nm.f("blk.%lld.down", ll),      RAD_BF16, {E, g.n_ff}, A, N, gl);
        y.w_post_ff   = decl_w(b, nm.f("blk.%lld.post_ff", ll),   RAD_BF16, {E}, A, N, gl);
        y.w_ple_gate  = decl_w(b, nm.f("blk.%lld.ple_gate", ll),  RAD_BF16, {m.n_ple, E}, A, N, gl);
        y.w_ple_proj  = decl_w(b, nm.f("blk.%lld.ple_proj", ll),  RAD_BF16, {E, m.n_ple}, A, N, gl);
        y.w_post_ple  = decl_w(b, nm.f("blk.%lld.post_ple", ll),  RAD_BF16, {E}, A, N, gl);
        /* bf16 and one element: the reference multiplies the bf16 residual by it in bf16. */
        y.w_scalar    = decl_w(b, nm.f("blk.%lld.scalar", ll),    RAD_BF16, {1}, A, N, gl);

        y.op_in_norm = rw(b, rmsnorm(T, E, g.eps, y.w_in_norm), {m.b_x}, {m.b_h});
        y.op_qkv     = rw(b, gemm(T, k.qkv(), E, y.w_qkv), {m.b_h}, {qkv});
        y.op_qk      = rw(b, RAD_OP(b, "qk_norm_rope",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.head_dim),
                                               RAD_INT("n_head", k.n_head),
                                               RAD_INT("n_head_kv", k.n_head_kv),
                                               RAD_F64("theta", k.theta), RAD_F64("eps", g.eps),
                                               RAD_STR("mode", "neox"), RAD_INT("v_norm", 1)),
                                    RAD_WEIGHTS(y.w_q_norm, y.w_k_norm)),
                          {qkv}, {qkv});
        y.op_attn    = rw(b, RAD_OP(b, "attn_dense",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.head_dim),
                                               RAD_INT("gqa", k.n_head / k.n_head_kv),
                                               RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                               RAD_F64("scale", 1.0),
                                               RAD_INT("max_seqlen", max_seg),
                                               RAD_INT("window", k.window)),
                                    RAD_NOWEIGHTS),
                          {qkv}, {att});
        y.op_o        = rw(b, gemm(T, E, k.q(), y.w_o), {att}, {m.b_o});
        y.op_post_attn= rw(b, rmsnorm(T, E, g.eps, y.w_post_attn), {m.b_o}, {m.b_o});
        y.op_add_attn = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_pre_ff   = rw(b, rmsnorm(T, E, g.eps, y.w_pre_ff), {m.b_x}, {m.b_h});
        y.op_gate_up  = rw(b, gemm(T, 2 * g.n_ff, E, y.w_gate_up), {m.b_h}, {m.b_gu});
        y.op_act      = rw(b, act_mul(g.n_ff), {m.b_gu}, {m.b_ff});
        y.op_down     = rw(b, gemm(T, E, g.n_ff, y.w_down), {m.b_ff}, {m.b_o});
        y.op_post_ff  = rw(b, rmsnorm(T, E, g.eps, y.w_post_ff), {m.b_o}, {m.b_o});
        y.op_add_ff   = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_ple_gate = rw(b, gemm(T, m.n_ple, E, y.w_ple_gate), {m.b_x}, {m.b_p});
        y.op_ple_act  = rw(b, act_mul(m.n_ple), {m.b_p, m.b_ple}, {m.b_p});
        y.op_ple_proj = rw(b, gemm(T, E, m.n_ple, y.w_ple_proj), {m.b_p}, {m.b_o});
        y.op_post_ple = rw(b, rmsnorm(T, E, g.eps, y.w_post_ple), {m.b_o}, {m.b_o});
        y.op_add_ple  = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_scale    = rw(b, RAD_OP(b, "mul_weight",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                                RAD_STR("dtype", "bf16"), RAD_INT("scalar", 1)),
                                     RAD_WEIGHTS(y.w_scalar)),
                           {m.b_x}, {m.b_x});
    }

    m.op_norm = rw(b, rmsnorm(T, E, g.eps, m.w_norm), {m.b_x}, {m.b_h});
    m.op_mean = rw(b, RAD_OP(b, "seg_mean",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {m.b_h}, {m.b_mean});
    m.op_out  = rw(b, gemm(g.max_seqs, m.n_out, E, m.w_out), {m.b_mean}, {m.b_proj});
    m.op_cast = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", m.n_out),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS),
                   {m.b_proj}, {m.b_pooled});

    RAD_ARCH_TRY(m.vis.declare(b, meta, ctx, nm, E, (int)g.n_layer));
    RAD_ARCH_TRY(m.aud.declare(b, meta, ctx, nm, E,
                               (int)g.n_layer + (int)m.vis.L.size()));

    int n_full = 0;
    for (int f : full) n_full += f;
    rad_note(b, "embedding model: %lld layers (%d whole-input, %lld sliding |i-j|<=%lld), hidden "
                "%lld, ffn %lld; sliding %lldx%lld (kv %lld) theta %g, full %lldx%lld (kv %lld) "
                "theta %g; per-layer inputs %lld; mean-pooled to %lld",
             (long long)g.n_layer, n_full, (long long)(g.n_layer - n_full), (long long)radius,
             (long long)E, (long long)g.n_ff, (long long)m.ks.n_head, (long long)m.ks.head_dim,
             (long long)m.ks.n_head_kv, m.ks.theta, (long long)m.kf.n_head,
             (long long)m.kf.head_dim, (long long)m.kf.n_head_kv, m.kf.theta,
             (long long)m.n_ple, (long long)m.n_out);
    return RAD_OK;
}

/* Before declare: the vision tower's pooling window, so a pass size the core picks itself is a
 * whole number of them. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (Vision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.pooling_kernel_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    if (Audio::present(meta)) out->audio_frames = Audio::frames_for(meta);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, E = g.n_embd;
    if (batch->enc) {
        if (batch->enc == RAD_MM_AUDIO) m.aud.step(c, batch);
        else                            m.vis.step(c, batch);
        return;
    }
    if (T <= 0 || S <= 0) return;
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, E),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, T));
    RAD_ISSUE(c, m.op_ple, brows(m.b_x, T), RAD_W(m.w_ple_proj), brows(m.b_ple, T));
    /* [T, n_layer * n_ple] read as [T * n_layer, n_ple]: one norm row per (token, layer). */
    RAD_ISSUE_N(c, m.op_ple_norm, T * g.n_layer, brows(m.b_ple, T), RAD_W(m.w_ple_norm),
                brows(m.b_ple, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const Layer& y = m.layers[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const rad_buf qkv = y.full ? m.b_qkv_f : m.b_qkv_s;
        const rad_buf att = y.full ? m.b_att_f : m.b_att_s;
        const int64_t q = k.q(), kv = k.n_head_kv * k.head_dim;

        RAD_ISSUE(c, y.op_in_norm, brows(m.b_x, T), RAD_W(y.w_in_norm), brows(m.b_h, T));
        RAD_ISSUE(c, y.op_qkv, brows(m.b_h, T), RAD_W(y.w_qkv), brows(qkv, T));
        RAD_ISSUE(c, y.op_qk, brows(qkv, T), pos, RAD_W(y.w_q_norm), RAD_W(y.w_k_norm));
        RAD_ISSUE(c, y.op_attn, bcol(qkv, 0, q, T), bcol(qkv, q, kv, T), bcol(qkv, q + kv, kv, T),
                  cu, brows(att, T));
        RAD_ISSUE(c, y.op_o, brows(att, T), RAD_W(y.w_o), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_attn, brows(m.b_o, T), RAD_W(y.w_post_attn), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_attn, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_pre_ff, brows(m.b_x, T), RAD_W(y.w_pre_ff), brows(m.b_h, T));
        RAD_ISSUE(c, y.op_gate_up, brows(m.b_h, T), RAD_W(y.w_gate_up), brows(m.b_gu, T));
        RAD_ISSUE(c, y.op_act, bcol(m.b_gu, 0, g.n_ff, T), bcol(m.b_gu, g.n_ff, g.n_ff, T),
                  brows(m.b_ff, T));
        RAD_ISSUE(c, y.op_down, brows(m.b_ff, T), RAD_W(y.w_down), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_ff, brows(m.b_o, T), RAD_W(y.w_post_ff), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_ff, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_ple_gate, brows(m.b_x, T), RAD_W(y.w_ple_gate), brows(m.b_p, T));
        RAD_ISSUE(c, y.op_ple_act, brows(m.b_p, T), bcol(m.b_ple, l * m.n_ple, m.n_ple, T),
                  brows(m.b_p, T));
        RAD_ISSUE(c, y.op_ple_proj, brows(m.b_p, T), RAD_W(y.w_ple_proj), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_ple, brows(m.b_o, T), RAD_W(y.w_post_ple), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_ple, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_scale, brows(m.b_x, T), RAD_W(y.w_scalar), brows(m.b_x, T));
    }

    RAD_ISSUE(c, m.op_norm, brows(m.b_x, T), RAD_W(m.w_norm), brows(m.b_h, T));
    RAD_ISSUE(c, m.op_mean, brows(m.b_h, T), cu, brows(m.b_mean, S));
    RAD_ISSUE_N(c, m.op_out, S, brows(m.b_mean, S), RAD_W(m.w_out), brows(m.b_proj, S));
    RAD_ISSUE_N(c, m.op_cast, S, brows(m.b_proj, S), brows(m.b_pooled, S));
    (void)E;
}

}  /* namespace embgemma2 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(embgemma2, "embeddinggemma2", "", "0.1.0",
                "EmbeddingGemma 2: a bidirectional Gemma encoder with per-layer embeddings, "
                "mean-pooled to one vector an input; bf16")
RAD_ARCH_PROBE(embgemma2)
#endif
