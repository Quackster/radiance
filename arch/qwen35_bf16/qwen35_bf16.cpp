/* qwen35_bf16 -- Qwen3.5 / 3.6 / 3.8 dense, bf16 weights and bf16 activations.
 *
 * Qwen3.5, Qwen3.6 and Qwen3.8 dense models all report the architecture id `qwen35`, and
 * `qwen35moe` is the same graph with a routed FFN. Dimensions come from model metadata, so this
 * one file serves every size in the family -- 0.8B through 27B. docs/ARCHITECTURES.md has the
 * geometry and the traps that stay plausible when they are wrong.
 *
 * One plugin implements one architecture id for one quantisation scheme (spec §2.4). A
 * `qwen35_w8a8` sibling is a different directory that composes the same blocks against
 * rmsnorm_had_quant_i8, gemm_nt_q and gdn_gated_norm_had_quant_i8 -- not a branch in here. That is
 * why the declare function below is a list of what runs, in order, with no branch on quant in it.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * The unquantised release -- Qwen3.8-27B in bf16, quantisation descriptor "" -- served at its own
 * precision: every projection a bf16 GEMM. The block-scaled fp8 release is qwen35_fp8's, which
 * declares the same graph over fp8 linears. The lm_head here is declared at whatever the model
 * holds it as, so a container whose recipe made it block fp8 reads it through the fp8 logits
 * kernel and a checkpoint loaded as it stands through the bf16 one.
 *
 * ============================== THE SHAPE, AT 27B ==============================
 *
 *   64 layers, full_attention_interval 4 -> layers 3,7,...,63 are FULL ATTENTION (16 of them),
 *                                           the other 48 are GATED DELTA NET
 *   hidden 5120   ffn 17408   vocab 248320   (lm_head NOT tied)
 *   24 heads / 4 kv heads / head_dim 256  -> GQA 6, and head_dim is READ, not derived:
 *                                            n_embd / n_head is 213 and every shape after it
 *                                            would be wrong.
 *   attn_output_gate -> q_proj is [n_embd, 2 * n_head * head_dim], query and gate interleaved
 *   partial_rotary_factor 0.25 -> MROPE over the first 64 of 256 head dims, theta 1e7
 *   linear: 16 key heads / 48 value heads of 128, conv width 4, fp32 recurrent state
 *
 * ============================== WHAT THIS PLUGIN DOES NOT DECLARE ==============================
 *
 * The MTP / NextN head (one extra decoder block plus fc, enorm, hnorm) is real scope and is NOT
 * here. It does not block text generation, and it has to be declared from tensor names read off
 * the checkpoint -- a name map written from memory is a model that loads half its weights and
 * says nothing. It sits BESIDE the trunk under an `mtp.` prefix of its own, so
 * `nextn_predict_layers` is not subtracted from the layer count (see declare()), and declaring
 * the MTP block as a trunk layer would run it in the main pass, which is a different model.
 *
 * The VISION TOWER (27 blocks, hidden 1152, 16 heads of 72, patch 16, out 5120) is declared
 * through arch/common/rad_block_vit.h, only when the container carries one and the deployment
 * asked for media.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated.h>
#include <arch/rad_block_gdn.h>
#include <arch/rad_block_mlp.h>
#include <arch/rad_block_vit.h>

#include <vector>

namespace qwen35_bf16 {

using namespace rad::arch;

/* One trunk layer. A hybrid model's layer is one of two kinds and the plugin holds both structs
 * rather than a union: they are small, and a variant would buy nothing except a cast at every use.
 * `full` is the only branch in step(), and it is a branch on the MODEL, not on a configuration --
 * the same layer is the same kind on every step and in every deployment. */
struct Layer {
    int           full = 0;
    AttnGatedBF16 attn;
    GdnBF16       gdn;
    MlpBF16       mlp;
};

struct Model {
    Names           nm{""};
    Geom            g{};
    GdnBF16::Config gcfg{};
    int64_t         n_rot  = 0;
    int64_t         n_main = 0;    /* trunk layers, excluding the MTP block */

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    /* full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_gate = 0, b_attn = 0;
    /* gated delta net */
    rad_buf b_in = 0, b_z = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0, b_go = 0;
    /* feed-forward */
    rad_buf b_gate_up = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows,
     * declared only when the container carries a tower and the deployment asked for media. */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    std::vector<Layer> layers;
};

/* Declare runs once per rank, in one process, on the rank's own thread (spec §1, §9). A single
 * static Model would have rank 1's declare overwrite rank 0's handles; this is indexed by rank and
 * read back in step() with rad_rank(). */
static Model g_model[MAX_RANKS];

/* Layer kind. llama.cpp is ground truth: `is_recr(i) = (i < n_main) && ((i + 1) % interval != 0)`,
 * so with interval 4 the FULL-ATTENTION layers are 3, 7, ... and everything else is linear. */
static inline bool is_full_attention(int64_t l, int64_t interval) {
    return interval > 0 && ((l + 1) % interval == 0);
}

static int check_layer_schedule(const RadModelMeta* meta, int64_t n_layer, int64_t interval) {
    const char* s = rad_meta_gets(meta, "qwen35.layer_types", nullptr);
    if (!s) s = rad_meta_gets(meta, "text_config.layer_types", nullptr);
    if (!s) s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s || !*s) return RAD_OK;

    int64_t l = 0;
    for (const char* p = s; *p; ) {
        while (*p == ' ') ++p;
        if (!*p) break;
        const char* tok = p;
        while (*p && *p != ' ') ++p;
        const size_t len = (size_t)(p - tok);

        const bool says_full = len == 14 && std::strncmp(tok, "full_attention", 14) == 0;
        const bool says_lin  = len == 16 && std::strncmp(tok, "linear_attention", 16) == 0;
        if (!says_full && !says_lin) {
            fprintf(stderr, "radiance: qwen35: layer_types[%lld] is '%.*s'; this plugin knows "
                            "'full_attention' and 'linear_attention' and will not guess at a "
                            "third kind of layer.\n", (long long)l, (int)len, tok);
            return RAD_E_UNSUPPORTED;
        }
        /* Past the trunk is the MTP block, which the array may or may not include; it is not a
         * trunk layer and its kind is the drafter's business (see declare_draft). */
        if (l >= n_layer) break;
        if (says_full != is_full_attention(l, interval)) {
            fprintf(stderr, "radiance: qwen35: the container says layer %lld is %s, and "
                            "full_attention_interval %lld makes it %s. The interval rule is "
                            "llama.cpp's and it is not a property of this family -- a model whose "
                            "layer_types differ from it would be built with the wrong layers "
                            "linear, and would load and generate fluent text from an architecture "
                            "that is not the one in the checkpoint. Read layer_types directly "
                            "before serving this model.\n",
                    (long long)l, says_full ? "full_attention" : "linear_attention",
                    (long long)interval,
                    is_full_attention(l, interval) ? "full_attention" : "linear_attention");
            return RAD_E_UNSUPPORTED;
        }
        ++l;
    }
    if (l && l < n_layer) {
        fprintf(stderr, "radiance: qwen35: layer_types names %lld layers and the model has %lld. "
                        "A short schedule is not a schedule.\n", (long long)l, (long long)n_layer);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    /* A routed model is a different graph, not a flag on this one. qwen35moe_bf16 serves it. */
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: qwen35_bf16 refuses a checkpoint with %lld experts; that is "
                        "qwen35moe\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- geometry that RadModelMeta does not name, read from the container's free-form header.
     * Both spellings are accepted: a .rad converted from the FP8 safetensors carries the
     * HuggingFace keys and one converted from a GGUF carries llama.cpp's. */
    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);
    const int64_t nextn    = meta_int2(meta, "qwen35.nextn_predict_layers",
                                       "mtp_num_hidden_layers", 0);
    if (interval <= 0 || nextn < 0 || nextn >= g.n_layer) {
        fprintf(stderr, "radiance: qwen35: full_attention_interval %lld / nextn %lld out of range "
                        "for %lld layers\n",
                (long long)interval, (long long)nextn, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(check_layer_schedule(meta, g.n_layer, interval));
    /* EVERY DECLARED LAYER IS A TRUNK LAYER. `mtp_num_hidden_layers` is NOT part of
     * `num_hidden_layers` in this family -- Qwen3.8-27B-FP8 ships layers-0..63 under
     * `model.language_model.layers.N` and its one MTP block separately, under an `mtp.` prefix of
     * its own, in mtp.safetensors. Subtracting it here would drop layer 63, which is a
     * full-attention layer, and the model would answer with the trunk one layer short: still a
     * distribution, still self-consistent op by op, and wrong. `nextn` is kept for the note below,
     * which is the only thing the count is good for until the MTP head is declared. */
    m.n_main = g.n_layer;

    /* Qwen3.5's norms are GemmaRMSNorm: `x_hat * (1 + w)`, not `x_hat * w`. vLLM says so by
     * importing it under the family's own name -- `from ...layernorm import GemmaRMSNorm as
     * Qwen3_5RMSNorm` -- and the checkpoint says so too: `input_layernorm.weight` sits around 0.07
     * to 0.29, which is a gain of 1.1 to 1.3 the right way round and a five-to-ten-times shrink
     * the wrong way. It applies to input_layernorm, post_attention_layernorm, the model's final
     * norm and the q/k head norms -- and NOT to the delta net's ssm_norm, which is a plain
     * RMSNormGated with gains around 0.88.
     *
     * The checkpoint answers two of the three families by itself: the layer norms (0.07..0.29)
     * and the q/k head norms (partly NEGATIVE) can only be `1 + w`. The model's FINAL norm sits at
     * 0.68..0.96, which reads as either, and takes `1 + w` because the reference implementation
     * does. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    /* The rotary width. partial_rotary_factor 0.25 over head_dim 256 is 64, and a container that
     * omits it is refused rather than defaulted to head_dim: a full rotation is a different model
     * and its symptom is fluent wrong text, which is exactly what spec §17 is about. */
    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        /* BOTH SPELLINGS, because the family uses both. Qwen3.8-27B-FP8 states
         * `partial_rotary_factor` at the text_config level AND inside `rope_parameters`;
         * Qwen3.5-0.8B states it ONLY inside `rope_parameters`, and rad-convert flattens nested
         * config with a dot. Reading only one spelling refuses a container of the same
         * architecture with "no usable rotary width", which is a true statement about the key and
         * a false one about the model. */
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        if (prf > 0.0) m.n_rot = (int64_t)(g.head_dim * prf);
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: qwen35: no usable rotary width -- give the container "
                        "qwen35.rope.dimension_count or partial_rotary_factor (head_dim %lld)\n",
                (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* The linear-attention geometry. llama.cpp maps the GGUF ssm keys onto these:
     *   ssm.state_size -> the key AND value head width, ssm.group_count -> key heads,
     *   ssm.time_step_rank -> value heads, ssm.inner_size -> value heads * value width. */
    GdnBF16::Config& gc = m.gcfg;
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count",
                                           "linear_num_key_heads", 0);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank",
                                           "linear_num_value_heads", 0);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 0);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 0);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 0);
    if (gc.head_v <= 0) {
        const int64_t inner = meta_int2(meta, "qwen35.ssm.inner_size", "linear_inner_size", 0);
        if (inner > 0 && n_head_v_all > 0) gc.head_v = inner / n_head_v_all;
    }
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 ||
        gc.conv_width <= 0) {
        fprintf(stderr, "radiance: qwen35: the container does not describe the linear layers "
                        "(key heads %lld, value heads %lld, key width %lld, value width %lld, "
                        "conv width %lld)\n",
                (long long)n_head_k_all, (long long)n_head_v_all, (long long)gc.head_k,
                (long long)gc.head_v, (long long)gc.conv_width);
        return RAD_E_INVAL;
    }
    /* The chunked scan's tile length. Not a model parameter: it is libr4d's
     * gdn_chunk_scan_k128_v128_c64_bf16, and it is also the interval a linear-state checkpoint is
     * snapshotted at (spec §7.3). Note that gdn_conv_prep takes it while gdn_chunk_scan derives it
     * from the kernel, and nothing checks that the two agree. */
    gc.chunk = 64;

    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0)
        return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
        return s;

    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = gc.v_dim();
    const int64_t conv_dim = gc.conv_dim();

    /* ---- THREE KV groups, which is why hybrid KV groups are first-class rather than an
     * extension (spec §7.2). Every layer is bound to the group it owns state in, below: a group
     * with no bound layers is a declare error, because how many state instances a sequence holds
     * is the layer count and assuming one would make the page size a guess.
     *
     * The block size is NOT declared for the full-attention group -- it comes off the resolved
     * attention kernel's constraints and is read back with rad_kv_block_size() inside the block. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* The recurrent state is fp32 because that is the only state width libr4d has a kernel for
     * (gdn_recurrent_update_k128_v128_bf16_fp32state) and what `mamba_ssm_dtype: float32` asks
     * for. It costs 48 x 128 x 128 x 4 = 3 MiB a layer and 144 MiB across the 48 linear layers,
     * PER SEQUENCE, which is the largest single number in this deployment's memory plan.
     * An fp16 state would halve that and cost a little step time, but declaring it here would be
     * declaring a geometry no kernel serves, so it waits for the kernel. */
    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    /* The conv window: conv_width-1 + n_spec entries of the whole convolved row, counted in HEADS
     * so the channel split never straddles a q/k/v boundary at any tensor-parallel width. */
    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);

    /* ---- activation buffers, sized at max_tok because that is the worst step. Every layer of a
     * kind shares these: layers that share a geometry share the buffer, and the buffer plan's
     * liveness analysis over the declared read and write sets is what makes one arena enough
     * (spec §3.1, §8). */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h       = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});

    /* The query-and-gate projection is declared RANK 3, [tokens, heads, 2*head_dim], because that
     * is what makes the interleaving addressable: the query is the first head_dim columns of the
     * last dimension and the gate is the second head_dim, so both are ordinary column slices whose
     * row stride is the declared width. Flattened to [tokens, 2*q_dim] the same bytes describe a
     * per-head view no operand can name. */
    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_gate    = decl_b(b, nm.f("attn_gate"),g.act_dtype, {g.max_tok, q_dim});
    m.b_attn    = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});

    m.b_in      = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, gc.in_dim()});
    m.b_z       = decl_b(b, nm.f("gdn_z"),    g.act_dtype, {g.max_tok, v_dim});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    /* The decay, beta and the chunk inverse are f32 and not bf16. The decay is a cumulative sum
     * whose intra-chunk span reaches hundreds of nats on real prefill inputs; a bf16 running sum
     * loses the low bits of exactly the quantity every decay factor is an exponent of. */
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    /* BF16, not f32: gdn_kkt_solve writes this plane as bf16 and gdn_chunk_scan reads it as bf16
     * (libr4d's own row text, and the casts in both .hip files). See qwen35_fp8.cpp. */
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    m.b_go      = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});

    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn     = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    /* The lm_head runs over the SAMPLED ROWS, not over every token of the chunk. `max_logit_rows`
     * is one row per sequence, or one per verified draft position per sequence -- at 248320 vocab
     * that is the difference between 63 MiB and 8 GiB, and between one GEMM per sequence and one
     * per token. `b_hout` is where gather_rows puts the hidden states it selects; it is narrow
     * (n_embd, not n_vocab) which is why gathering before the GEMM and not after is the whole
     * point. */
    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- vocabulary edges.
     * BOTH ENDS OF THE VOCABULARY ARE ROW-SHARDED. Replicating the embedding while sharding the
     * lm_head would put 248320 x 5120 in bf16 -- 2.43 GiB -- on every rank, for a table each rank
     * uses a slice of. `embed_lookup` takes a `vocab_offset`: a rank gathers the ids in its own
     * slice, writes a zero row for every id belonging to another rank, and the all_reduce below
     * sums the ranks into x. lm_head is sharded because the sampler runs a distributed top-k and full
     * logits are never gathered anywhere (spec §9, §13); the embedding is sharded because the
     * other end of the same vocabulary already was. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    /* This model does NOT tie its lm_head -- `tie_word_embeddings: false`, and the checkpoint
     * ships lm_head.weight beside the embedding. The second source is for the small members of the
     * family that do tie; first source present wins, and without it a family whose members differ
     * only in tying needs two plugins. The maps come first: they are how the model is asked what
     * each weight is. */
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* ROW-SHARDED with the vocabulary: g.n_vocab is THIS RANK'S share and rad_arch.h says what
     * the split is and what it buys. The two change together. Declared at its codes' dtype: bf16
     * as the checkpoint ships it, E4M3 where a recipe made it block fp8 -- and the logits kernel
     * that reads that encoding is the one selected. */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* THE ENCODER'S ROWS, over the lookup and before its reduction (MediaRows says why). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    /* EXACT, and not g.wire_exact: this sums DISJOINT rows -- every token comes from exactly one
     * rank and the others contribute zeros -- so a lossy wire would be loss on a value that is
     * already the answer, not a bandwidth trade against a partial product. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    /* ---- the layers. This loop is the model card. */
    AttnGatedBF16::Wire aw{};
    aw.x = m.b_x; aw.h = m.b_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.gate = m.b_gate; aw.attn = m.b_attn;

    GdnBF16::Wire gw{};
    gw.x = m.b_x; gw.h = m.b_h; gw.in = m.b_in; gw.z = m.b_z;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.b_go;

    MlpBF16::Wire mw{};
    mw.x = m.b_x; mw.h = m.b_h; mw.gate_up = m.b_gate_up; mw.ffn = m.b_ffn;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedBF16::Src as{};
            as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",     pre, (long long)l);
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight",    pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight",    pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight",    pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight",    pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight",    pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as));
        } else {
            GdnBF16::Src gs{};
            gs.norm     = nm.ckpt("%s.%lld.input_layernorm.weight",           pre, (long long)l);
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight",   pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",     pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",     pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",     pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",        pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",                pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",              pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",          pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",      pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs));
        }

        MlpBF16::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms));
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});

    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* ---- what a reader of --debug-graph should be told, in the plugin's own words. */
    rad_note(b, "rank %d/%d: %lld trunk layers, %lld full attention + %lld gated delta net, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main,
             (long long)(m.n_main / interval), (long long)(m.n_main - m.n_main / interval),
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "gdn: %lld key heads x %lld, %lld value heads x %lld, conv width %lld, chunk %lld; "
                "conv row %lld channels, projection row %lld",
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width, (long long)gc.chunk,
             (long long)conv_dim, (long long)gc.in_dim());
    rad_note(b, "ssm_a_log carries A_log AS A LOG: the kernels form -exp(A_log)*softplus(a+dt). "
                "A container built from a GGUF has ssm_a = -exp(A_log) already applied and must "
                "undo it, or every head's decay collapses");
    rad_note(b, "rope is mrope over %lld of %lld head dims, sections [11,11,10]; RadBatch carries "
                "one position component per token, so the three sections see the same position -- "
                "exact for text, wrong for an image span",
             (long long)m.n_rot, (long long)g.head_dim);
    if (!have_interval)
        rad_note(b, "the container did not state full_attention_interval; assuming %lld, which "
                    "makes layers 3,7,... full attention", (long long)interval);
    rad_note(b, "logits are computed for the %lld sampled rows of a step, not for all %lld "
                "tokens: RadBatch::out_ids names them and gather_rows selects the hidden states "
                "before the lm_head, which is the difference between 63 MiB and 8 GiB at this "
                "vocabulary", (long long)g.max_logit_rows, (long long)g.max_tok);
    if (nextn > 0)
        rad_note(b, "%lld MTP/NextN layer(s) sit BESIDE the %lld trunk layers -- under an `mtp.` "
                    "prefix of their own, not as layers %lld.. -- and are NOT declared: this "
                    "plugin serves the trunk only",
                 (long long)nextn, (long long)m.n_main, (long long)m.n_main);

    /* THE VISION TOWER, LAST, so its activations share the arena with every buffer above. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* AN ENCODER PASS RUNS THE VISION TOWER AND NOTHING ELSE (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    m.mrows.step(c, batch, m.b_x, batch->n_tok);
    /* The vocab shard's reduction; declared only when world > 1, so tp1 is unchanged. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    for (const Layer& l : m.layers) {
        if (l.full) l.attn.step(c, batch);
        else        l.gdn.step(c, batch);
        l.mlp.step(c, batch);
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));

    /* The lm_head runs on the rows the sampler asked for and on nothing else. A prefill chunk in
     * the middle of a long prompt sets n_out to zero: it finishes no sequence, so there is no
     * distribution to draw from and the widest GEMM in the model is not issued at all. */
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

}  /* namespace qwen35_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(qwen35_bf16, "qwen35", "", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), bf16 weights and activations, gated "
                "attention over paged KV every fourth layer and gated delta net between")
#endif
