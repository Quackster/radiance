/* LFM2-VL's SigLIP2 encoder and spatial projector. The processor supplies real patches only,
 * grouped by image/tile, with each factor x factor block contiguous in (dy, dx) order. Keeping
 * those rows in that order through noncausal attention makes pixel unshuffle a final view.
 * Coordinates still name each patch's original raster position for the learned position table.
 */
#pragma once
#include <arch/rad_block_vit.h>

namespace lfm2_bf16 {
using namespace rad::arch;

struct Siglip2Vision {
    struct Layer {
        rad_weight nw[2]{}, nb[2]{};
        rad_weight qkv = 0, qkv_bias = 0, out = 0, out_bias = 0;
        rad_weight up = 0, up_bias = 0, down = 0, down_bias = 0;
        rad_op norm[2]{}, project = 0, attend = 0, output = 0, expand = 0, contract = 0;
    };
    bool on = false;
    int64_t P = 0, width = 0, patch_dim = 0, merge = 0, out_width = 0, max_seg = 0;
    rad_weight patch = 0, patch_bias = 0, pos = 0, final_norm = 0, final_bias = 0;
    rad_weight proj_norm = 0, proj_norm_bias = 0, proj1 = 0, proj1_bias = 0, proj2 = 0, proj2_bias = 0;
    rad_buf x = 0, h = 0, qkv = 0, attn = 0, ff = 0, merged = 0, projected = 0, output = 0;
    rad_op embed = 0, position = 0, residual = 0, normalize = 0, project_norm = 0, project1 = 0, project2 = 0;
    std::vector<Layer> layers;

    int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Names& nm,
                int64_t n_embd, int first_group) {
        const int64_t depth = rad_meta_geti(meta, "vision_config.num_hidden_layers", 0);
        if (ctx->max_enc_patches <= 0 || depth <= 0) return RAD_OK;
        if (std::strcmp(rad_meta_gets(meta, "vision_config.model_type", ""), "siglip2_vision_model"))
            return RAD_E_UNSUPPORTED;
        width = rad_meta_geti(meta, "vision_config.hidden_size", 0);
        const int64_t heads = rad_meta_geti(meta, "vision_config.num_attention_heads", 0);
        const int64_t hidden = rad_meta_geti(meta, "vision_config.intermediate_size", 0);
        const int64_t edge = rad_meta_geti(meta, "vision_config.patch_size", 0);
        const int64_t chans = rad_meta_geti(meta, "vision_config.num_channels", 3);
        const int64_t positions = rad_meta_geti(meta, "vision_config.num_patches", 0);
        const int64_t side = (int64_t)std::llround(std::sqrt((double)positions));
        const int64_t factor = rad_meta_geti(meta, "downsample_factor", 0);
        const int64_t project_width = rad_meta_geti(meta, "projector_hidden_size", 0);
        const float eps = (float)rad_meta_getf(meta, "vision_config.layer_norm_eps", 1e-6);
        const bool norm_projector = rad_meta_geti(meta, "projector_use_layernorm", 0) != 0;
        const bool bias_projector = rad_meta_geti(meta, "projector_bias", 1) != 0;
        const char* hidden_act = rad_meta_gets(meta, "vision_config.hidden_act", "gelu_pytorch_tanh");
        const char* project_act = rad_meta_gets(meta, "projector_hidden_act", "gelu");
        auto activation = [](const char* act) -> const char* {
            return !std::strcmp(act, "gelu") ? "gelu" :
                   !std::strcmp(act, "gelu_pytorch_tanh") ? "gelu_tanh" : nullptr;
        };
        const char* act = activation(hidden_act);
        const char* pact = activation(project_act);
        max_seg = rad_meta_geti(meta, "max_num_patches", 1024);
        if (width <= 0 || heads <= 0 || width % heads || hidden <= 0 || edge <= 0 ||
            chans <= 0 || side <= 0 || side * side != positions || factor <= 0 || factor > 8 ||
            project_width <= 0 || n_embd <= 0 || !(eps > 0) || max_seg <= 0)
            return RAD_E_FORMAT;
        if (!act || !pact || !bias_projector) return RAD_E_UNSUPPORTED;
        patch_dim = chans * edge * edge;
        merge = factor * factor;
        out_width = n_embd;
        P = ctx->max_enc_patches;
        if (P % merge) return RAD_E_SHAPE;
        if (ctx->shape_probe) P = std::max<int64_t>(merge, std::min(P, ctx->max_tok) / merge * merge);
        const int64_t R = P / merge;

        auto weight = [&](const char* name, const char* source,
                          std::initializer_list<int64_t> shape, RadWeightGroup group) {
            if (map_copy(b, name, source) < 0) return (rad_weight)0;
            return decl_w(b, name, RAD_BF16, shape, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
        };
        const char* root = "model.vision_tower.vision_model";
        patch = weight(nm.f("v.patch.weight"), nm.ckpt("%s.embeddings.patch_embedding.weight", root),
                       {width, patch_dim}, grp_model());
        patch_bias = weight(nm.f("v.patch.bias"), nm.ckpt("%s.embeddings.patch_embedding.bias", root),
                            {width}, grp_model());
        pos = weight(nm.f("v.pos.weight"), nm.ckpt("%s.embeddings.position_embedding.weight", root),
                     {positions, width}, grp_model());
        final_norm = weight(nm.f("v.output_norm.weight"), nm.ckpt("%s.post_layernorm.weight", root),
                            {width}, grp_model());
        final_bias = weight(nm.f("v.output_norm.bias"), nm.ckpt("%s.post_layernorm.bias", root),
                            {width}, grp_model());
        layers.resize((size_t)depth);
        for (int i = 0; i < depth; ++i) {
            Layer& l = layers[(size_t)i];
            const auto group = grp_layer(first_group + i);
            for (int j = 0; j < 2; ++j) {
                l.nw[j] = weight(nm.f("v.blk.%d.norm%d.weight", i, j + 1),
                    nm.ckpt("%s.encoder.layers.%d.layer_norm%d.weight", root, i, j + 1), {width}, group);
                l.nb[j] = weight(nm.f("v.blk.%d.norm%d.bias", i, j + 1),
                    nm.ckpt("%s.encoder.layers.%d.layer_norm%d.bias", root, i, j + 1), {width}, group);
            }
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%d.qkv.weight", i),
                {nm.ckpt("%s.encoder.layers.%d.self_attn.q_proj.weight", root, i),
                 nm.ckpt("%s.encoder.layers.%d.self_attn.k_proj.weight", root, i),
                 nm.ckpt("%s.encoder.layers.%d.self_attn.v_proj.weight", root, i)}, 0));
            RAD_ARCH_TRY(map_concat(b, nm.f("v.blk.%d.qkv.bias", i),
                {nm.ckpt("%s.encoder.layers.%d.self_attn.q_proj.bias", root, i),
                 nm.ckpt("%s.encoder.layers.%d.self_attn.k_proj.bias", root, i),
                 nm.ckpt("%s.encoder.layers.%d.self_attn.v_proj.bias", root, i)}, 0));
            l.qkv = decl_w(b, nm.f("v.blk.%d.qkv.weight", i), RAD_BF16, {3 * width, width},
                          RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
            l.qkv_bias = decl_w(b, nm.f("v.blk.%d.qkv.bias", i), RAD_BF16, {3 * width},
                               RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
            l.out = weight(nm.f("v.blk.%d.out.weight", i),
                nm.ckpt("%s.encoder.layers.%d.self_attn.out_proj.weight", root, i), {width, width}, group);
            l.out_bias = weight(nm.f("v.blk.%d.out.bias", i),
                nm.ckpt("%s.encoder.layers.%d.self_attn.out_proj.bias", root, i), {width}, group);
            l.up = weight(nm.f("v.blk.%d.up.weight", i),
                nm.ckpt("%s.encoder.layers.%d.mlp.fc1.weight", root, i), {hidden, width}, group);
            l.up_bias = weight(nm.f("v.blk.%d.up.bias", i),
                nm.ckpt("%s.encoder.layers.%d.mlp.fc1.bias", root, i), {hidden}, group);
            l.down = weight(nm.f("v.blk.%d.down.weight", i),
                nm.ckpt("%s.encoder.layers.%d.mlp.fc2.weight", root, i), {width, hidden}, group);
            l.down_bias = weight(nm.f("v.blk.%d.down.bias", i),
                nm.ckpt("%s.encoder.layers.%d.mlp.fc2.bias", root, i), {width}, group);
        }
        if (norm_projector) {
            proj_norm = weight(nm.f("v.projector.norm.weight"),
                "model.multi_modal_projector.layer_norm.weight", {merge * width}, grp_model());
            proj_norm_bias = weight(nm.f("v.projector.norm.bias"),
                "model.multi_modal_projector.layer_norm.bias", {merge * width}, grp_model());
        }
        proj1 = weight(nm.f("v.projector.1.weight"), "model.multi_modal_projector.linear_1.weight",
                       {project_width, merge * width}, grp_model());
        proj2 = weight(nm.f("v.projector.2.weight"), "model.multi_modal_projector.linear_2.weight",
                       {n_embd, project_width}, grp_model());
        if (bias_projector) {
            proj1_bias = weight(nm.f("v.projector.1.bias"), "model.multi_modal_projector.linear_1.bias",
                                {project_width}, grp_model());
            proj2_bias = weight(nm.f("v.projector.2.bias"), "model.multi_modal_projector.linear_2.bias",
                                {n_embd}, grp_model());
        }
        if (!patch || !patch_bias || !pos || !final_norm || !final_bias || !proj1 || !proj2 ||
            !proj1_bias || !proj2_bias || (norm_projector && (!proj_norm || !proj_norm_bias)))
            return RAD_E_INVAL;
        for (const Layer& l : layers)
            if (!l.nw[0] || !l.nb[0] || !l.nw[1] || !l.nb[1] || !l.qkv || !l.qkv_bias ||
                !l.out || !l.out_bias || !l.up || !l.up_bias || !l.down || !l.down_bias)
                return RAD_E_INVAL;

        x = decl_b(b, nm.f("v.x"), RAD_BF16, {P, width});
        h = decl_b(b, nm.f("v.h"), RAD_BF16, {P, width});
        qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, {P, 3 * width});
        attn = decl_b(b, nm.f("v.attn"), RAD_BF16, {P, width});
        ff = decl_b(b, nm.f("v.ff"), RAD_BF16, {P, hidden});
        merged = decl_b(b, nm.f("v.merged"), RAD_BF16, {R, merge * width});
        projected = decl_b(b, nm.f("v.projected"), RAD_BF16, {R, project_width});
        output = decl_b(b, nm.f("v.output"), RAD_BF16, {R, n_embd});
        if (!x || !h || !qkv || !attn || !ff || !merged || !projected || !output) return RAD_E_INVAL;
        auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* nonlinearity,
                        rad_weight w, rad_weight bias) {
            return RAD_OP(b, "gemm_nt_bias",
                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                           RAD_STR("dtype", "bf16"), RAD_STR("act", nonlinearity)), RAD_WEIGHTS(w, bias));
        };
        auto norm = [&](int64_t rows, int64_t n, float epsilon, rad_weight w, rad_weight bias) {
            return RAD_OP(b, "layernorm",
                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", epsilon),
                           RAD_STR("dtype", "bf16")), RAD_WEIGHTS(w, bias));
        };
        embed = rw(b, gemm(P, width, patch_dim, "none", patch, patch_bias), {}, {x});
        position = rw(b, RAD_OP(b, "grid_embed",
            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_INT("side", side),
                       RAD_STR("dtype", "bf16"), RAD_STR("mode", "bilinear_antialias")),
            RAD_WEIGHTS(pos)), {x}, {x});
        residual = rw(b, RAD_OP(b, "add",
            RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_STR("dtype", "bf16")),
            RAD_NOWEIGHTS), {h, x}, {x});
        for (Layer& l : layers) {
            l.norm[0] = rw(b, norm(P, width, eps, l.nw[0], l.nb[0]), {x}, {h});
            l.project = rw(b, gemm(P, 3 * width, width, "none", l.qkv, l.qkv_bias), {h}, {qkv});
            l.attend = rw(b, RAD_OP(b, "attn_dense",
                RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", width / heads), RAD_INT("gqa", 1),
                           RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                           RAD_F64("scale", 1.0 / std::sqrt((double)(width / heads))),
                           RAD_INT("max_seqlen", max_seg)), RAD_NOWEIGHTS), {qkv}, {attn});
            l.output = rw(b, gemm(P, width, width, "none", l.out, l.out_bias), {attn}, {h});
            l.norm[1] = rw(b, norm(P, width, eps, l.nw[1], l.nb[1]), {x}, {h});
            l.expand = rw(b, gemm(P, hidden, width, act, l.up, l.up_bias), {h}, {ff});
            l.contract = rw(b, gemm(P, width, hidden, "none", l.down, l.down_bias), {ff}, {h});
        }
        normalize = rw(b, norm(P, width, eps, final_norm, final_bias), {x}, {merged});
        if (norm_projector)
            project_norm = rw(b, norm(R, merge * width, 1e-5f, proj_norm, proj_norm_bias), {merged}, {merged});
        project1 = rw(b, gemm(R, project_width, merge * width, pact, proj1, proj1_bias), {merged}, {projected});
        project2 = rw(b, gemm(R, n_embd, project_width, "none", proj2, proj2_bias), {projected}, {output});
        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name = "siglip2"; e.modalities = RAD_MM_IMAGE; e.patch_dim = patch_dim;
            e.merge = merge; e.max_patches = P; e.n_embd = n_embd; e.out = output;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        }
        on = true;
        rad_note(b, "SigLIP2 vision: %lld blocks, width %lld, heads %lld, patches %lld, spatial merge %lld",
                 (long long)depth, (long long)width, (long long)heads, (long long)P, (long long)merge);
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* b) const {
        const int64_t n = b->enc_n_patch, R = merge ? n / merge : 0;
        if (!on || n <= 0 || n > P || n % merge || b->enc_n_seg <= 0 ||
            b->enc_max_seg > max_seg || !b->enc_pixels || !b->enc_coord || !b->enc_cu) {
            rad_step_fail(c, "invalid SigLIP2 encoder pass or vision was not enabled at conversion");
            return;
        }
        const auto coord = praw2(b->enc_coord, RAD_I32, 4, n);
        const auto cu = praw(b->enc_cu, RAD_I32, b->enc_n_seg + 1);
        RAD_ISSUE_N(c, embed, n, praw2(b->enc_pixels, RAD_BF16, n, patch_dim),
                    RAD_W(patch), RAD_W(patch_bias), RAD_NONE, brows(x, n));
        RAD_ISSUE_N(c, position, n, brows(x, n), RAD_W(pos), coord);
        for (const Layer& l : layers) {
            RAD_ISSUE_N(c, l.norm[0], n, brows(x, n), RAD_W(l.nw[0]), RAD_W(l.nb[0]), brows(h, n));
            RAD_ISSUE_N(c, l.project, n, brows(h, n), RAD_W(l.qkv), RAD_W(l.qkv_bias), RAD_NONE, brows(qkv, n));
            RAD_ISSUE_N(c, l.attend, n, bcol(qkv, 0, width, n), bcol(qkv, width, width, n),
                        bcol(qkv, 2 * width, width, n), cu, brows(attn, n));
            RAD_ISSUE_N(c, l.output, n, brows(attn, n), RAD_W(l.out), RAD_W(l.out_bias), RAD_NONE, brows(h, n));
            RAD_ISSUE_N(c, residual, n, brows(h, n), brows(x, n), brows(x, n));
            RAD_ISSUE_N(c, l.norm[1], n, brows(x, n), RAD_W(l.nw[1]), RAD_W(l.nb[1]), brows(h, n));
            RAD_ISSUE_N(c, l.expand, n, brows(h, n), RAD_W(l.up), RAD_W(l.up_bias), RAD_NONE, brows(ff, n));
            RAD_ISSUE_N(c, l.contract, n, brows(ff, n), RAD_W(l.down), RAD_W(l.down_bias), RAD_NONE, brows(h, n));
            RAD_ISSUE_N(c, residual, n, brows(h, n), brows(x, n), brows(x, n));
        }
        RAD_ISSUE_N(c, normalize, n, brows(x, n), RAD_W(final_norm), RAD_W(final_bias), brows(merged, R));
        if (project_norm)
            RAD_ISSUE_N(c, project_norm, R, brows(merged, R), RAD_W(proj_norm), RAD_W(proj_norm_bias), brows(merged, R));
        RAD_ISSUE_N(c, project1, R, brows(merged, R), RAD_W(proj1), proj1_bias ? RAD_W(proj1_bias) : RAD_NONE,
                    RAD_NONE, brows(projected, R));
        RAD_ISSUE_N(c, project2, R, brows(projected, R), RAD_W(proj2), proj2_bias ? RAD_W(proj2_bias) : RAD_NONE,
                    RAD_NONE, brows(output, R));
    }
};
}  // namespace lfm2_bf16
