/* qwen35_mxfp4 -- Qwen3.5 / 3.6 / 3.8 dense, OCP-MX FP4 weights and fp8 activations.
 *
 * The MX twin of qwen35_fp8, and the sibling of qwen35_bf16. Same architecture, same graph, same
 * metadata, same name maps, same MTP head, same DFlash2 drafter, same vision tower -- so the
 * architecture prose lives in qwen35_bf16.cpp and in the three MX block headers, and this file
 * says only what makes it a different plugin. One plugin implements one architecture id for one
 * quantisation scheme (spec §2.4), and this is the third scheme.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * A container converted with
 *
 *     rad-convert <Qwen3.8-27B bf16 checkpoint> --quant-id mxfp4 \
 *         --recipe data/recipes/q38-27b-mxfp4.recipe -o q38-27b-mxfp4.rad
 *
 * whose trunk linears are OCP MXFP4: fp4 E2M1 codes with an E8M0 shared exponent per group of 32
 * along K (`fp4_e2m1*e8m0[1x32]`), read by libr4d's `gemm_nt_q` rows at dtype mxfp4a8 as three
 * views of one logical weight -- the packed codes, the [K/32][N] exponent plane, and the per-row
 * u8 reference the layout derives. The activation is E4M3 with ONE f32 scale a row, not the
 * block-fp8 per-128 grid.
 *
 *   MXFP4 trunk      blk.*.attn_qg / attn_k / attn_v / attn_output, ssm_inz, ssm_out,
 *                    ffn_gate_up, ffn_down
 *   block FP8        output.weight (the lm_head), mtp.* / blk.64.* (the MTP head),
 *                    dflash.* (the DFlash2 drafter)
 *   BF16             every norm, A_log, dt_bias, conv1d, embed_tokens, and the delta net's
 *                    in_proj_a / in_proj_b (48 rows, not a whole group)
 *
 * THE LM_HEAD, THE MTP HEAD AND THE DFLASH2 DRAFTER STAY BLOCK FP8. The lm_head is the widest
 * GEMM in the model and the plan keeps it at half the bytes with the logits path libr4d already
 * ships; the MTP head and the DFlash2 drafter are block-fp8 from their own releases (the recipe
 * says so), read through the fp8 blocks unchanged.
 *
 * ============================== WHAT COMES FROM rad_fp8.h ==============================
 *
 * `rad_mx.h` is the OCP-MX trunk and includes `rad_fp8.h` -- the all-reduce wire predicates, the
 * fused-attention band, the rope table height, and every block-scaled-fp8 piece the head and the
 * drafter still need (ActFP8, LinearFP8, DenseFP8, NormQuantFP8/GateQuantFP8, and the
 * rad_block_mtp_fp8.h / rad_block_dflash2.h blocks). So this plugin includes BOTH rad_mx.h and
 * rad_fp8.h -- through rad_mx.h -- because the model is half MX and half fp8.
 *
 * THE TRUNK'S TWO LINEARS PER BLOCK ARE MX (LinearMX, rad_mx.h): fp4 codes, an E8M0 exponent
 * plane and a per-row reference, `gemm_nt_q` at dtype mxfp4a8, fed by `rmsnorm_quant_fp8_row` /
 * `gated_quant_fp8_row` / `quant_act_fp8_row`, each writing ONE f32 scale a row.
 *
 * ============================== THE FOLDS THIS PLUGIN TAKES ==============================
 *
 * The MX blocks take the same folds the fp8 twin does, at the same bands:
 *
 *   - the AR fold into the norm is ON (kMxArFold true in rad_mx.h). `ar_rmsnorm_quant_fp8_row`'s
 *     two-rank byte-identity test passes (`ar_qr_two_rank` in libr4d/r4d_selftest.cpp, selected as
 *     `--case ar_rmsnorm_quant_fp8_row`), so the blocks forward `ar_in` and the norm absorbs the
 *     predecessor's all-reduce. On tp2 the standalone `all_reduce` is issued only where the fused
 *     op is not reachable, on the same predicate the predecessor skips it.
 *   - the GATED PAIR is folded at decode: `MlpMX` declares `gemm_nt_q_gated_mx` over the gate_up
 *     weights for M <= 64, and runs the pair (LinearMX::step then GateQuantMX::step) for prefill.
 *     Byte-identical.
 *   - the only fold left on the table is the SPARSE attention gate: there is no per-row
 *     `attn_paged_gate_quant` schema, so that path runs `sigmoid` -> `mul` -> `quant_act_fp8_row`.
 *     Same arithmetic, two extra launches on a sparse prefill.
 *
 * None of this changes a number; each is one launch the twin removes and this plugin does not.
 */
#ifndef QWEN35_NS
#define QWEN35_NS qwen35_mxfp4
#endif
#ifndef QWEN35_ARCH_ID
#define QWEN35_ARCH_ID "qwen35"
#endif
#ifndef QWEN35_QUANT
#define QWEN35_QUANT "mxfp4"
#endif
#ifndef QWEN35_MX_CODE_NAME
#define QWEN35_MX_CODE_NAME "E2M1"
#endif
/* The graph below is codebook-agnostic -- rad_mx.h reads the codebook from the container's own
 * encoding -- so arch/qwen35_mxfp8 and arch/qwen35_mxfp6 include this file as a thin TU with four
 * #defines, the qwen35moe pattern. QWEN35_NS_STR lets a message name the plugin the TU actually
 * registered, not this file. */
#define QWEN35_STR_(x) #x
#define QWEN35_NS_STR  QWEN35_STR_(QWEN35_NS)

#include <cstdlib>
#include <cstddef>
#include <arch/rad_arch.h>
#include <arch/rad_mx.h>
#include <arch/rad_block_attn_gated_mxfp4.h>
#include <arch/rad_block_gdn_mxfp4.h>
#include <arch/rad_block_mlp_mxfp4.h>
#include <arch/rad_block_mtp_fp8.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_vit.h>

#include <cstring>
#include <vector>

namespace QWEN35_NS {

using namespace rad::arch;

/* The feed-forward this build declares, and the ONE name the rest of the file uses for it. The
 * MTP head stays the block-fp8 one: its DECODER-block linears are block fp8 in the checkpoint, and
 * the head is a separate small transformer rather than a trunk layer. */
using Ffn     = MlpMX;
using MtpHead = MtpFP8;

struct Layer {
    int          full = 0;
    AttnGatedMX  attn;
    GdnMX        gdn;
    Ffn          mlp;
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnMX::Config  gcfg{};
    int64_t        n_rot  = 0;
    int64_t        n_main = 0;

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActMX   a_h{};
    /* full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0;
    /* [rope_table_rows, n_rot] f32, the rope cos/sin every attention layer reads. One buffer and
     * one fill a step, against a rope launch a layer -- see rad_block_attn_gated_mxfp4.h. */
    rad_buf b_rope_cs = 0;
    ActMX   a_attn{};
    /* gated delta net */
    ActMX   a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0;
    /* feed-forward */
    rad_buf b_gate_up = 0;
    ActMX   a_ffn{};
#ifdef QWEN35_PAROQUANT
    /* PATH A: the rotated-activation buffers the paroquant blocks' producers write (shared across
     * layers, like every Wire activation). pq_h feeds any n_embd-wide input (attn q/k/v, gdn
     * qkv/z, mlp gate/up); pq_attn the attn o projection; pq_v the gdn out projection; pq_ff the
     * mlp down projection. attn/gdn/mlp of one layer run in turn, so pq_h is reused by them all. */
    ActMX   a_pq_h{}, a_pq_h2{}, a_pq_h3{}, a_pq_attn{}, a_pq_v{}, a_pq_ff{};
    /* separate contiguous GEMM outputs that a `cast` places into the fused buffers */
    rad_buf b_gate2 = 0, b_up2 = 0, b_qkv2 = 0, b_z2 = 0;
#endif

    /* THE DFLASH2 DRAFTER, declared only when the container carries one and the deployment
     * speculates. It is the second drafter and it is not an alternative spelling of the first: an
     * MTP head is one chained forward per draft token, this is one forward for the whole block.
     * Both can be in a container; only one is declared, and DFlash2 wins because it is the better
     * drafter on this target. It stays BLOCK FP8, so its own Wire carries the E4M3/f32 twins it
     * declares explicitly (rad_block_dflash2.h), not the trunk's MX planes. */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;

    /* the multi-token-prediction head, declared only when the deployment speculates */
    bool     have_mtp = false;
    MtpHead  mtp{};
    rad_buf  b_mtp_e = 0, b_mtp_c = 0, b_mtp_x = 0, b_mtp_h = 0;
    rad_buf  b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf  b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;
    /* THE MTP HEAD IS BLOCK FP8 WHERE THE TRUNK IS MX. Its inner attention and feed-forward are
     * AttnGatedFP8 / MlpFP8, so they need an ActFP8 -- E4M3 codes with a per-128 f32 scale -- that
     * the trunk no longer declares. Their bf16 planes ARE the trunk's (`a_h.x`, `a_attn.x`,
     * `a_ffn.x`), which are dead by the time a draft pass runs, so only the narrow pair is new;
     * see the MTP declaration for the wiring. */
    ActFP8   mtp_h{}, mtp_attn{}, mtp_ffn{};

    /* THE AUXILIARY HIDDEN-STATE TAPS, declared only when the container carries a drafter that
     * asks for them. A DFlash2 drafter does not read the trunk's FINAL hidden state -- it reads
     * the residual stream at several intermediate layers and projects the concatenation down, so
     * `fc` is [n_embd, n_taps * n_embd] and its input has to be one contiguous row. The trunk
     * keeps one residual buffer that every layer overwrites, so the rows have to be copied out as
     * they pass, into the column slice that layer owns. Nothing else in the trunk changes, and on
     * a container with no drafter none of this is declared. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    /* The embedding's single f32 scale when the container holds the table as E4M3 (a recipe's
     * `codes=fp8_e4m3 block=*x* scale=f32`), else 0 and the table is bf16. */
    rad_weight w_toks = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    rad_op op_rope_cs = 0;   /* rope_table, if the fused prologue is reachable */

    /* THE DFLASH2 VERIFY HEAD (RADIANCE_VERIFY_HEAD, default off). A SECOND target-logits path
     * that points the drafter's own 2-bit `w_dh`/`w_dhs` view of the shared lm_head at the TARGET's
     * hidden states, takes its coarse top-R, reranks those exactly against the trunk's fp8 head,
     * and writes a SPARSE b_logits for the existing sampler instead of the full fp8 logits GEMM.
     * The lever is bytes: the coarse pass reads 0.167 GiB/rank against the fp8 head's 1.18 GiB, and
     * at the M=64 tp2 geometry that is ~3.5x faster. It is exact over the sampler's support only
     * under the batch's verify_head gate, which the builder computes (abi/rad_runtime.h) -- so the
     * step takes the int2 path only when that is set, and declares none of this when the env is
     * unset, which keeps the default graph byte-for-byte the one it always was. */
    bool    verify_head = false;
    rad_buf b_vh_q = 0, b_vh_s = 0, b_vh_sum = 0;   /* quant_act_i8 of b_hout */
    rad_buf b_vh_log = 0;                            /* [max_logit_rows, head_rows] bf16 */
    rad_buf b_vh_cand = 0, b_vh_unary = 0;           /* [max_logit_rows, R] */
    rad_buf b_vh_ridx = 0, b_vh_rval = 0;            /* [max_logit_rows, R] */
    rad_buf b_vh_pair = 0, b_vh_gath = 0;            /* vocab-parallel pair, sharded only */
    rad_op  op_vh_q = 0, op_vh_log = 0, op_vh_topk = 0, op_vh_gath = 0, op_vh_merge = 0;
    rad_op  op_vh_rerank = 0, op_vh_scatter = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows,
     * declared only when the container carries a tower and the deployment asked for media. It
     * stays BF16 -- libr4d serves the tower in bf16 -- so it is the same declaration the fp8 and
     * bf16 plugins make. */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    std::vector<Layer> layers;
};

/* THE SSM STATE'S WIDTH, `RADIANCE_GDN_STATE=f16` -- the MX twin of qwen35_fp8's switch. f32 (the
 * default, and what this model's HF config asks for) or f16, which halves the largest item of the
 * KV plan: 144 -> 72 MiB a slot, two slots a sequence, and the checkpoint a slot with it -- about
 * 1.1 GiB at eight sequences, ~37k KV tokens, plus the same again if the checkpoint slots are kept.
 * libr4d reads the width off the state operand: the GDN decode rows
 * (`gdn_recurrent_update_k128_v128_bf16_fp32state`, `gdn_conv_recurrent_update_k128_v128_bf16`) and
 * the chunk scan (`gdn_chunk_scan_k128_v128_c64_bf16`) each take `state->dtype == RAD_F16`, so this
 * declaration is the whole switch on the MX path too. It changes the generated text (round-trip
 * error ~5e-4 relative a step), so it is opt-in and is to be judged on KL / GSM8K / acceptance
 * before it is anybody's default. Anything but f32/f16 warns and uses f32. */
static uint32_t gdn_state_dtype() {
    const char* e = std::getenv("RADIANCE_GDN_STATE");
    if (!e || !*e || !std::strcmp(e, "f32") || !std::strcmp(e, "fp32")) return RAD_F32;
    if (!std::strcmp(e, "f16") || !std::strcmp(e, "fp16")) {
        fprintf(stderr, "radiance: RADIANCE_GDN_STATE=f16 -- the delta-net state is held in f16 "
                        "(half the KV plan's largest item; the text differs from the f32 state's)\n");
        return RAD_F16;
    }
    fprintf(stderr, "radiance: RADIANCE_GDN_STATE=%s is neither f32 nor f16; using f32\n", e);
    return RAD_F32;
}

/* "5 19 33 47 61" -> the layers whose output a drafter wants. rad-convert flattens a config.json
 * array into a space-separated string (tools/rad_convert.cpp, flatten_json), so this is the shape
 * every list-valued key arrives in. Out-of-range entries are dropped rather than clamped. */
static int parse_tap_layers(const char* s, int64_t n_layer, int* out, int max) {
    if (!s) return 0;
    int n = 0;
    for (const char* p = s; *p && n < max;) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        p = e;
        if (v >= 0 && v < n_layer) out[n++] = (int)v;
    }
    return n;
}

static Model g_model[MAX_RANKS];

static inline bool is_full_attention(int64_t l, int64_t interval) {
    return interval > 0 && ((l + 1) % interval == 0);
}

/* THE CONTAINER'S OWN LAYER SCHEDULE, CHECKED AGAINST THE ONE WE COMPUTE. See qwen35_fp8.cpp for
 * the full argument: `layer_types` is authoritative where the container states it, and a
 * disagreement is fatal because the interval rule is llama.cpp's and not a property of the
 * family. */
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

/* THE CONTAINER'S OWN STATEMENT ABOUT ITS FORMAT. The SELECTION KEY already matched: the loader
 * chose this plugin because the container's quantisation descriptor is exactly QWEN35_QUANT. This
 * is the second guard, by name -- the same reason qwen35_fp8 refuses a non-fp8 container -- so a
 * hand-built or mis-registered container is refused rather than served with an MX grid read over
 * bytes that are bf16 or block fp8, which would load and produce fluent wrong text.
 *
 * A container that states a HuggingFace `quantization_config` at all must state an MX method.
 * A container converted with `--quant-id <quant>` from a bf16 checkpoint carries NO
 * quantization_config -- the recipe is a conversion-time instruction, not a checkpoint key -- and
 * that is the ordinary case, accepted here. */
static int check_format(const RadModelMeta* meta) {
    if (meta->quant && std::strcmp(meta->quant, QWEN35_QUANT) != 0) {
        fprintf(stderr, "radiance: %s refuses a container quantised by '%s'. This plugin "
                        "serves an OCP-MX " QWEN35_QUANT " trunk only; qwen35_bf16 serves an "
                        "unquantised one and qwen35_fp8 a block-scaled fp8 one.\n",
                QWEN35_NS_STR, meta->quant);
        return RAD_E_UNSUPPORTED;
    }
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    if (method && *method && std::strcmp(method, QWEN35_QUANT) != 0 &&
        std::strcmp(method, "mx") != 0) {
        fprintf(stderr, "radiance: %s refuses a container whose quantization_config "
                        "says '%s'. The trunk is OCP-MX " QWEN35_QUANT " (codes with an E8M0 "
                        "exponent a group of 32); a bf16 or block-fp8 checkpoint is a different "
                        "plugin.\n", QWEN35_NS_STR, method);
        return RAD_E_UNSUPPORTED;
    }
    /* THE WEIGHT GRID, where the container states one -- the MX twin of the fp8 plugin's
     * weight_block_size check. The OCP-MX weight is `codes*e8m0[1x32]`: one code row by a
     * 32-wide exponent group, and the group is compiled into libr4d, so a stated [*, *] that is
     * not [1, 32] reads the right bytes with the wrong gain. A container converted from bf16 with
     * `--quant-id <quant>` states no quantization_config at all and is the ordinary case, accepted
     * here; this arm fires only on a container that claims a grid. */
    const char* block = rad_meta_gets(meta, "quantization_config.weight_block_size", nullptr);
    if (block) {
        long long b0 = 0, b1 = 0;
        if (std::sscanf(block, "%lld %lld", &b0, &b1) != 2) { b0 = 0; b1 = 0; }
        if (b0 != 1 || b1 != RAD_MX_GROUP) {
            fprintf(stderr, "radiance: %s refuses weight_block_size [%lld, %lld]. The "
                            "OCP-MX grid is " QWEN35_QUANT "*e8m0[1x32] -- one row by a %d-wide "
                            "exponent group -- and the group width is compiled into libr4d, so "
                            "another width reads the right bytes with the wrong gain; [1, %d] is "
                            "the only grid this plugin serves.\n",
                    QWEN35_NS_STR, b0, b1, RAD_MX_GROUP, RAD_MX_GROUP);
            return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

/* THE DFLASH2 DRAFTER: its geometry, its KV group, its buffers and its block. Split out of
 * declare() because it is A SECOND MODEL and not a head. It is format-independent -- its six
 * linears are block fp8, declared inside rad_block_dflash2.h from the recipe -- so this is the
 * fp8 plugin's function unchanged. */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg = m.g;   /* n_embd, the vocabulary, the deployment bounds, the rank, the wire */

    dg.n_layer       = rad_meta_geti(meta, "draft.num_hidden_layers", 0);
    dg.head_dim      = rad_meta_geti(meta, "draft.head_dim", 0);
    dg.n_head_all    = rad_meta_geti(meta, "draft.num_attention_heads", 0);
    dg.n_head_kv_all = rad_meta_geti(meta, "draft.num_key_value_heads", 0);
    dg.n_ff_all      = rad_meta_geti(meta, "draft.intermediate_size", 0);
    dg.eps           = (float)rad_meta_getf(meta, "draft.rms_norm_eps", 0.0);
    dg.theta         = (float)rad_meta_getf(meta, "draft.rope_parameters.rope_theta",
                                            rad_meta_getf(meta, "draft.rope_theta", 0.0));
    dg.rope_scale    = 1.0f;
    /* PLAIN RMS NORM. The drafter uses the ordinary norm where the trunk uses GemmaRMSNorm, and
     * the checkpoint agrees rather than only vLLM -- rad_block_dflash2.h has the numbers. */
    dg.wadd = 0.0f;
    dg.wadd_qk = 0.0f;

    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: the container's `draft.` config does not describe a DFlash2 "
                        "drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, ffn %lld, "
                        "rms_eps %g, rope_theta %g\n",
                (long long)dg.n_layer, (long long)dg.n_head_all, (long long)dg.n_head_kv_all,
                (long long)dg.head_dim, (long long)dg.n_ff_all, (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    int s;
    if ((s = divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads", &dg.n_head))    < 0) return s;
    if ((s = divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads", &dg.n_head_kv)) < 0) return s;
    if ((s = divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",   &dg.n_ff))      < 0) return s;

    /* THE DRAFTER BORROWS THE TARGET'S EMBEDDING TABLE AND lm_head -- it ships neither -- so a
     * vocabulary that disagrees is not a drafter for this target at all. */
    const long long dvocab = rad_meta_geti(meta, "draft.vocab_size", dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: the DFlash2 drafter claims a %lld-token vocabulary and the "
                        "target has %lld. The drafter ships no embedding table and no lm_head; it "
                        "reads the target's, so the two have to be the same vocabulary.\n",
                dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }

    Dflash2::Config dc{};
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    /* THE CANDIDATE WIDTH IS AN INFERENCE-TIME KNOB. The selector is a trained rank-`sel_rank`
     * SCORER and `sel_topk` only truncates its ranked output -- it is the width of the top-R plane
     * `op_dtopk` emits and the `top_k` `op_sel` walks -- so admitting more of the head's candidates
     * touches no weight. The vLLM MXFP4 fork exposes the same lever as
     * RADIANCE_DFLASH_SELECTOR_TOPK and tunes the retrained drafter at 32
     * (radiance-vllm-mxfp4/aijuus/paroquant/drafter/RETRAIN.md); the checkpoint ships 16. There
     * `RADIANCE_DRAFT_RERANK` is a separate pool, but here the exact top-R IS the pool, so this one
     * value is both. 0/unset keeps the checkpoint value. */
    if (const char* e = std::getenv("RADIANCE_DFLASH_SELECTOR_TOPK")) {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v > 0) dc.sel_topk = v;
    }
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld). Without it the block of drafts has no path "
                        "through it.\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }
    /* THE `row_topk` SCHEMA NOW SERVES R <= 128, so a wider pool is refused by name here rather
     * than silently truncated to the ceiling at declaration time. The cap was 32 until the DFlash2
     * verify head needed R = 80; the selector's own pool and that head share the same row_topk. */
    if (dc.sel_topk > 128) {
        fprintf(stderr, "radiance: DFlash2 selector top_k %lld exceeds the 128 the row_topk schema "
                        "serves; RADIANCE_DFLASH_SELECTOR_TOPK tops out at 128.\n",
                (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }

    /* THE QUERY BLOCK MUST BE A POWER OF TWO, and this is where that is said. The convolution
     * masks a row's position within the block with `block_size - 1`, so any other value silently
     * convolves one sequence's block into the next. */
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block has to be a power of two. This serve "
                        "asks for %d draft tokens, which makes it %lld; the checkpoint was trained "
                        "at %lld (%lld draft tokens).\n",
                ctx->max_spec, (long long)dc.block, trained, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and this serve "
                    "asks for %lld. A block-diffusion drafter denoises a fixed number of mask "
                    "positions and was trained for one block length; it will run, and it will "
                    "accept less than the checkpoint can do",
                 trained, trained - 1, (long long)dc.block);

    /* Three scalars this drafter's family can carry that this implementation does not apply. */
    const double escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    const double omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (escale != 1.0 || omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: this DFlash2 checkpoint scales its input embedding by %g and "
                        "its draft logits by %g, and soft-caps them at %g. None of the three is "
                        "implemented, and a drafter that ignored them would just accept less.\n",
                escale, omul, scap);
        return RAD_E_UNSUPPORTED;
    }

    /* THE DRAFTER'S OWN KV CACHE, a group of its own: different head count, different head_dim,
     * five layers against sixty-four, and a WINDOW. Its width is the trunk's --kv-cache-dtype. */
    RadKVGroupDecl dd{};
    dd.kind      = dc.window > 0 ? RAD_KV_WINDOW : RAD_KV_FULL;
    dd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    dd.window    = dc.window;
    dd.n_head_kv = dg.n_head_kv;
    dd.head_dim  = dg.head_dim;
    m.kv_draft = rad_decl_kv_group(b, nm.f("kv_draft"), &dd);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, m.kv_draft));
    dg.kv_dtype = rad_dtype_name(dd.dtype);

    /* Activation buffers. The context pass is one row a COMMITTED TOKEN; everything after it is
     * one row a DRAFT POSITION, `block` of them a sequence. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.b_dfsrc;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    /* THE FP8 TWINS. The drafter's six linears are block-scaled fp8 over a bf16 checkpoint
     * (rad_fp8.h's DenseFP8), so every activation that feeds one carries an E4M3 copy and a plane
     * of f32 scales beside it. */
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3,
                        {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,
                        {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    /* GROUP-major, [n_embd/group][ceil(M/16)*16] -- quant_act_i8's asum plane. */
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, dc.sel_topk});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, dc.sel_topk});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, dc.sel_topk, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.sel_topk, 2});
    }
    /* NAMED, because the engine reads the draft out of it by name. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    m.df.w_toks = m.w_toks;
    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id, and every drafted position embeds it -- "
                        "there is nothing to run.\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    /* ---- THE DFLASH2 VERIFY HEAD, behind RADIANCE_VERIFY_HEAD (DEFAULT OFF). When unset this
     * block does not run and the graph is byte-for-byte the one it always was. R is FIXED at 80:
     * the builder's sampling gate is fixed at 80 (core/sched/batch.cpp k_verify_head_rerank), and a
     * plugin-side override would let a top_k pass a gate written for a different set. */
    if (const char* vh = std::getenv("RADIANCE_VERIFY_HEAD"); vh && std::strcmp(vh, "1") == 0) {
        m.verify_head = true;
        const int64_t VR = 80;
        const int64_t hr = Dflash2::head_rows(dg);
        const int64_t lr = dg.max_logit_rows;
        const int64_t vpad = ((lr + 15) / 16) * 16;
        const bool    vshard = Dflash2::head_sharded(dg);
        m.b_vh_q    = decl_b(b, nm.f("vh_q"),    RAD_I8,  {lr, dg.n_embd});
        m.b_vh_s    = decl_b(b, nm.f("vh_s"),    RAD_F32, {lr, 1});
        m.b_vh_sum  = decl_b(b, nm.f("vh_sum"),  RAD_F32, {dg.n_embd / 128, vpad});
        m.b_vh_log  = decl_b(b, nm.f("vh_log"),  dg.act_dtype, {lr, hr});
        m.b_vh_cand = decl_b(b, nm.f("vh_cand"), RAD_I32, {lr, VR});
        m.b_vh_unary= decl_b(b, nm.f("vh_unary"),RAD_F32, {lr, VR});
        m.b_vh_ridx = decl_b(b, nm.f("vh_ridx"), RAD_I32, {lr, VR});
        m.b_vh_rval = decl_b(b, nm.f("vh_rval"), RAD_F32, {lr, VR});
        if (vshard) {
            m.b_vh_pair = decl_b(b, nm.f("vh_pair"), RAD_F32, {lr, VR, 2});
            m.b_vh_gath = decl_b(b, nm.f("vh_gath"), RAD_F32, {lr, (int64_t)dg.world * VR, 2});
        }
        m.op_vh_q = rw(b, RAD_OP(b, "quant_act_i8",
                         RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("n", dg.n_embd),
                                    RAD_INT("group", 128), RAD_STR("dtype", dg.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_hout}, {m.b_vh_q, m.b_vh_s, m.b_vh_sum});
        m.op_vh_log = rw(b, RAD_OP(b, "gemm_nt_q",
                           RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("N", hr),
                                      RAD_INT("K", dg.n_embd), RAD_INT("group", 128),
                                      RAD_STR("dtype", "w2a8")),
                           RAD_WEIGHTS(m.df.w_dh, m.df.w_dhs)),
                       {m.b_vh_q, m.b_vh_s, m.b_vh_sum}, {m.b_vh_log});
        m.op_vh_topk = vshard
            ? rw(b, RAD_OP(b, "row_topk",
                      RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("N", hr), RAD_INT("R", VR),
                                 RAD_INT("vocab_off", dg.vocab_off), RAD_STR("dtype", dg.dtype)),
                      RAD_NOWEIGHTS),
                  {m.b_vh_log}, {m.b_vh_cand, m.b_vh_unary, m.b_vh_pair})
            : rw(b, RAD_OP(b, "row_topk",
                      RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("N", hr), RAD_INT("R", VR),
                                 RAD_STR("dtype", dg.dtype)),
                      RAD_NOWEIGHTS),
                  {m.b_vh_log}, {m.b_vh_cand, m.b_vh_unary});
        if (vshard) {
            m.op_vh_gath = rw(b, RAD_OP(b, "all_gather",
                               RAD_PARAMS(RAD_INT("world_size", dg.world),
                                          RAD_INT("numel", lr * VR * 2),
                                          RAD_STR("dtype", "f32"), RAD_INT("row", VR * 2)),
                               RAD_NOWEIGHTS),
                           {m.b_vh_pair}, {m.b_vh_gath});
            m.op_vh_merge = rw(b, RAD_OP(b, "row_topk_merge",
                                RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("R", VR),
                                           RAD_INT("world_size", dg.world),
                                           RAD_STR("dtype", dg.dtype)),
                                RAD_NOWEIGHTS),
                            {m.b_vh_gath}, {m.b_vh_cand, m.b_vh_unary});
        }
        m.op_vh_rerank = rw(b, RAD_OP(b, "logit_rerank",
                             RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("R", VR),
                                        RAD_INT("n_vocab", hr), RAD_INT("n_embd", dg.n_embd),
                                        RAD_INT("vocab_off", dg.vocab_off),
                                        RAD_STR("dtype", dg.dtype)),
                             RAD_WEIGHTS(m.w_lm_head)),
                         {m.b_hout, m.b_vh_cand},
                         vshard ? std::initializer_list<rad_buf>{m.b_vh_ridx, m.b_vh_rval, m.b_vh_pair}
                                : std::initializer_list<rad_buf>{m.b_vh_ridx, m.b_vh_rval});
        m.op_vh_scatter = rw(b, RAD_OP(b, "logits_scatter",
                              RAD_PARAMS(RAD_RANGE("M", 1, lr), RAD_INT("R", VR),
                                         RAD_INT("n_vocab", dg.n_vocab),
                                         RAD_INT("vocab_off", dg.vocab_off)),
                              RAD_NOWEIGHTS),
                          {m.b_vh_ridx, m.b_vh_rval}, {m.b_logits});
        rad_note(b, "the DFlash2 verify head is declared (RADIANCE_VERIFY_HEAD): the target's "
                    "logits run through the drafter's own 2-bit head over %lld rows/rank, the "
                    "top-%lld are reranked exactly against the trunk's fp8 lm_head, and a sparse "
                    "b_logits is written -- only for batches the builder marks safe",
                 (long long)hr, (long long)VR);
    }

    rad_note(b, "the DFlash2 drafter is declared: %lld layers of its own at absolute %d..%d, "
                "%lld heads / %lld kv x %lld, ffn %lld, %s attention with a %lld-token window, "
                "reading the trunk at %d tap(s). ONE forward proposes %lld tokens -- there are no "
                "rounds -- and the candidate selector walks a path through the block",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.steps());
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the ACTIVATION dtype and it is what every op but the MX linears matches on: the
     * norms, rope, attention, the delta-net kernels and the collectives all run on bf16 here, and
     * the E4M3 activations exist only between a quantiser and the GEMM that reads them. Saying
     * "fp8" or "mx" here would be saying it about the residual stream, which is false. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    /* DENSE ONLY. A routed checkpoint is qwen35moe, and there is no MX MoE block to serve it
     * (rad_mx.h has no MoeMX); refusing by name is the honest answer. */
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: %s refuses a checkpoint with %lld experts; that is "
                        "qwen35moe, and this plugin serves the dense graph\n",
                QWEN35_NS_STR, (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

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
     * `num_hidden_layers` in this family -- it sits separately under an `mtp.` prefix. */
    m.n_main = g.n_layer;

    /* Qwen3.5's norms are GemmaRMSNorm: `x_hat * (1 + w)`. Applies to input_layernorm,
     * post_attention_layernorm, the final norm and the q/k head norms -- NOT the delta net's
     * ssm_norm, which is a plain RMSNormGated. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        /* BOTH SPELLINGS, because the family uses both. */
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

    GdnMX::Config& gc = m.gcfg;
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

    /* ---- the three KV groups. The attention cache is BF16 unless --kv-cache-dtype fp8 makes it
     * E4M3; the SSM state is f32 unless RADIANCE_GDN_STATE=f16; the conv window is bf16. Identical
     * to the fp8 plugin: the cache width follows the DEPLOYMENT, not the weight format. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = gdn_state_dtype();   /* f32 unless RADIANCE_GDN_STATE=f16, see above */
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);

    /* ---- activation buffers. Every activation that FEEDS AN MX GEMM carries an fp8 twin: an
     * E4M3 plane of the same shape and ONE f32 scale a row (ActMX), where the fp8 twin's carried
     * one per 128 columns. The twins are transients, shared by liveness with whatever is dead. */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x     = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd));

    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x  = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    /* THE ROPE TABLE: persisted, filled once a step, read by every attention layer. */
    m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32, {rope_table_rows(g), m.n_rot},
                         RAD_BUF_PERSIST);
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim));

    /* `gdn_in` is conv_dim + v_dim: q|k|v from ssm_inz, then the output gate in its tail. */
    m.a_in.x    = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab      = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {g.max_tok, gc.ab_dim()});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    /* BF16, not f32: gdn_kkt_solve WRITES this plane as bf16 and gdn_chunk_scan reads it back the
     * same way. Declaring it f32 sizes it at twice the bytes and makes the two kernels disagree
     * about every element. */
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    m.a_go.x    = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim));

    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff));
#ifdef QWEN35_PAROQUANT
    RAD_ARCH_TRY(m.a_pq_h.declare_qs(b, nm, g, "pq_h", g.n_embd));
    RAD_ARCH_TRY(m.a_pq_h2.declare_qs(b, nm, g, "pq_h2", g.n_embd));
    RAD_ARCH_TRY(m.a_pq_h3.declare_qs(b, nm, g, "pq_h3", g.n_embd));
    RAD_ARCH_TRY(m.a_pq_attn.declare_qs(b, nm, g, "pq_attn", q_dim));
    RAD_ARCH_TRY(m.a_pq_v.declare_qs(b, nm, g, "pq_v", v_dim));
    RAD_ARCH_TRY(m.a_pq_ff.declare_qs(b, nm, g, "pq_ff", g.n_ff));
    m.b_gate2 = decl_b(b, nm.f("gate2"), g.act_dtype, {g.max_tok, g.n_ff});
    m.b_up2   = decl_b(b, nm.f("up2"),   g.act_dtype, {g.max_tok, g.n_ff});
    m.b_qkv2  = decl_b(b, nm.f("qkv2"),  g.act_dtype, {g.max_tok, gc.conv_dim()});
    m.b_z2    = decl_b(b, nm.f("z2"),    g.act_dtype, {g.max_tok, gc.v_dim()});
#endif

    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* The auxiliary taps. Gated on the DRAFTER being in the container AND on the deployment
     * intending to speculate. `cast` copies the residual stream after a named layer into the
     * column slice that tap owns. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_tap_layers(
                rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            }
        }
    }
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- vocabulary edges. The embedding is bf16 unless the container holds it E4M3, in which
     * case it is one f32 scale for the whole table (embed_lookup_q). The lm_head is block fp8 in
     * this recipe (`output.weight rtn codes=fp8_e4m3 block=128x128 scale=bf16`), declared at its
     * codes' dtype so the fp8 logits kernel that reads that encoding is selected. Both ends are
     * row-sharded with the vocabulary. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    /* THE EMBEDDING AT E4M3, when the container holds it so: one scale for the whole table, which
     * is what an embedding's narrow per-row range allows (docs/OPS.md, embed_lookup_q), half the
     * bytes of the bf16 table -- 1.18 GiB at 248k x 5120 -- and all of it handed to the KV cache.
     * A gather reads a handful of rows a step, so the format costs no time. Decided from what the
     * container says the table IS, so a bf16 embedding is declared and served exactly as before
     * and a recipe line is all it takes to opt in. */
    {
        RadEncoding e{};
        const bool q = weight_enc(b, nm.f("token_embd.weight"), &e) && e.n_planes == 2 &&
                       e.plane[0].dtype == RAD_F8E4M3;
        if (q) {
            m.w_tok  = decl_view(b, nm.f("token_embd.weight"), nullptr, "codes", RAD_F8E4M3,
                                 {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
            m.w_toks = decl_view(b, nm.f("token_embd.scale"), nm.f("token_embd.weight"), "scale",
                                 RAD_F32, {1, 1}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
            if (!m.w_tok || !m.w_toks) return RAD_E_INVAL;
        } else {
            m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                             RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
        }
    }
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = m.w_toks
        ? rw(b, RAD_OP(b, "embed_lookup_q",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                            RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                            RAD_INT("vocab_offset", g.vocab_off)),
                 RAD_WEIGHTS(m.w_tok, m.w_toks)),
             {}, {m.b_x})
        : rw(b, RAD_OP(b, "embed_lookup",
                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                            RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                            RAD_INT("vocab_offset", g.vocab_off)),
                 RAD_WEIGHTS(m.w_tok)),
             {}, {m.b_x});
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    /* EXACT, unconditionally: this sums DISJOINT rows, so a lossy wire would be pure loss. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    /* ---- the rope table's fill, declared before the layers that read it. */
    m.op_rope_cs = rw(b, RAD_OP(b, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)), RAD_INT("rot", m.n_rot),
                                    RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                    RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs});

    AttnGatedMX::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;
#ifdef QWEN35_PAROQUANT
    aw.pq_h = m.a_pq_h; aw.pq_h2 = m.a_pq_h2; aw.pq_h3 = m.a_pq_h3; aw.pq_attn = m.a_pq_attn;
#endif
    GdnMX::Wire gw{};
    gw.x = m.b_x; gw.h = m.a_h; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;
#ifdef QWEN35_PAROQUANT
    gw.pq_h = m.a_pq_h; gw.pq_h2 = m.a_pq_h2; gw.pq_v = m.a_pq_v; gw.qkv2 = m.b_qkv2; gw.z2 = m.b_z2;
#endif

    Ffn::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;
#ifdef QWEN35_PAROQUANT
    mw.pq_h = m.a_pq_h; mw.pq_h2 = m.a_pq_h2; mw.pq_ff = m.a_pq_ff;
    mw.gate2 = m.b_gate2; mw.up2 = m.b_up2;
#endif

    /* ---- WHERE THE RESIDUAL ADD HAPPENS. Every block ends `x += h` and every block begins
     * `norm(x)`; the add can move to the front of the NEXT block's norm. TWO THINGS READ `x`
     * BETWEEN BLOCKS and each keeps the add at the end of its block: the DRAFTER'S TAPS and the
     * FINAL NORM. The MTP head below is left unfolded for the same reason in reverse. */
    auto is_tap = [&](int64_t l) {
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l) return true;
        return false;
    };

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        const bool mix_fold = l > 0 && !is_tap(l - 1);
        const bool mlp_fold = true;
        const bool mix_add  = !mlp_fold;
        const bool mlp_add  = is_tap(l) || l + 1 >= m.n_main;
        /* The MX blocks fold the all-reduce into the norm exactly as the fp8 twin does
         * (kMxArFold true), so each flag means what it means there: `*_ar_in` is what this block's
         * norm absorbs, `*_ar_out` is what the next block's does, and a block issues its standalone
         * collective only where neither takes it. */
        const bool tp2      = g.world == 2;
        const int  mix_ar_in  = !tp2               ? kArNone
                              : mix_fold           ? kArAny
                              : (l == 0 && m.op_embed_ar) ? kArExact
                                                   : kArNone;
        const bool mix_ar_out = !mix_add && tp2;
        const bool mlp_ar_in  = mlp_fold && tp2;
        const bool mlp_ar_out = !mlp_add && tp2;

        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedMX::Src as{};
            as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",     pre, (long long)l);
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight",    pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight",    pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight",    pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight",    pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight",    pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          mix_fold, mix_add, mix_ar_in, mix_ar_out));
        } else {
            GdnMX::Src gs{};
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
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         mix_fold, mix_add, mix_ar_in, mix_ar_out));
        }

        Ffn::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, mlp_fold, mlp_add,
                                     mlp_ar_in, mlp_ar_out));
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.a_h.x});

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});

    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* ---- the multi-token-prediction head (rad_block_mtp_fp8.h).
     *
     * DECLARED ONLY WHEN THE DEPLOYMENT SPECULATES, and ONLY WHEN THERE IS NO DFLASH2 DRAFTER
     * (`n_taps` is the sign of one). THE HEAD IS BLOCK FP8: its inner attention and feed-forward
     * are the fp8 blocks, so their wires carry ActFP8 planes -- E4M3 with a per-128 f32 scale --
     * which the trunk no longer declares. Their bf16 planes ARE the trunk's (`a_h.x`, `a_attn.x`,
     * `a_ffn.x`), dead by the time a draft pass runs, so only the narrow pair is new. The plain
     * linears' intermediates (`attn_qg`, `attn_k`, the ffn planes) are format-agnostic and are
     * shared with the trunk exactly as the fp8 plugin shares them. */
    if (nextn > 0 && g.max_spec > 0 && m.n_taps == 0) {
        m.b_mtp_h = decl_b(b, nm.f("mtp_h"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_e = decl_b(b, nm.f("mtp_e"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_c = decl_b(b, nm.f("mtp_c"), g.act_dtype, {g.max_tok, 2 * g.n_embd});
        m.b_mtp_x = decl_b(b, nm.f("mtp_x"), g.act_dtype, {g.max_tok, g.n_embd});

        if (mtp_draft_head_q2(b, nm.f("mtp.draft_head.weight"), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,   {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32,  {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32,  {g.n_embd / 128, dmpad});
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype, {g.max_logit_rows, g.n_vocab_all});
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32,  {g.max_logit_rows, 1});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32,  {g.max_logit_rows, 1});
        }

        /* The head's FP8 planes, over the trunk's bf16 planes. */
        m.mtp_h.x = m.a_h.x;
        RAD_ARCH_TRY(m.mtp_h.declare_qs(b, nm, g, "mtp_h", g.n_embd));
        m.mtp_attn.x = m.a_attn.x;
        RAD_ARCH_TRY(m.mtp_attn.declare_qs(b, nm, g, "mtp_attn", q_dim));
        m.mtp_ffn.x = m.a_ffn.x;
        RAD_ARCH_TRY(m.mtp_ffn.declare_qs(b, nm, g, "mtp_ffn", g.n_ff));

        /* THE MTP WIRE IS ENUMERATED HERE, NOT COPIED, because the head is block-fp8 while the
         * trunk is MX: its inner AttnGatedFP8/MlpFP8 wires are a DIFFERENT TYPE from the trunk's
         * AttnGatedMX/MlpMX ones, so `tw_attn = aw` is not available the way it is in
         * qwen35_fp8.cpp. A field added to AttnGatedFP8::Wire or MlpFP8::Wire would then be
         * silently left zero. Pin the last field of each to the struct's end so an appended member
         * fails the build here instead. */
        static_assert(offsetof(AttnGatedFP8::Wire, gate) +
                          sizeof(((AttnGatedFP8::Wire*)nullptr)->gate) == sizeof(AttnGatedFP8::Wire),
                      "AttnGatedFP8::Wire gained a member after `gate`; the MTP wire built in "
                      "qwen35_mxfp4.cpp enumerates its fields and would leave the new one zero");
        static_assert(offsetof(MlpFP8::Wire, ffn) +
                          sizeof(((MlpFP8::Wire*)nullptr)->ffn) == sizeof(MlpFP8::Wire),
                      "MlpFP8::Wire gained a member after `ffn`; the MTP wire built in "
                      "qwen35_mxfp4.cpp enumerates its fields and would leave the new one zero");

        AttnGatedFP8::Wire tw_attn{};
        tw_attn.x = m.b_mtp_x; tw_attn.h = m.mtp_h;
        tw_attn.qg = m.b_qg; tw_attn.k = m.b_k; tw_attn.v = m.b_v; tw_attn.q = m.b_q;
        tw_attn.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;
        tw_attn.attn = m.mtp_attn;
        MlpFP8::Wire tw_mlp{};
        tw_mlp.x = m.b_mtp_x; tw_mlp.h = m.mtp_h;
        tw_mlp.gate_up = m.b_gate_up; tw_mlp.ffn = m.mtp_ffn;

        MtpHead::Wire tw{};
        tw.src = m.a_h.x;   tw.hout = m.b_mtp_h; tw.logits = m.b_logits;
        tw.e = m.b_mtp_e;   tw.c = m.b_mtp_c;    tw.x = m.b_mtp_x;
        tw.dq = m.b_mtp_dq; tw.ds = m.b_mtp_ds;  tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.attn = tw_attn;  tw.mlp = tw_mlp;

        MtpHead::Src ts{};
        ts.fc                 = nm.ckpt("mtp.fc.weight");
        ts.norm               = nm.ckpt("mtp.norm.weight");
        ts.pre_norm_hidden    = nm.ckpt("mtp.pre_fc_norm_hidden.weight");
        ts.pre_norm_embedding = nm.ckpt("mtp.pre_fc_norm_embedding.weight");
        ts.attn.norm   = nm.ckpt("mtp.layers.0.input_layernorm.weight");
        ts.attn.qg     = nm.ckpt("mtp.layers.0.self_attn.q_proj.weight");
        ts.attn.k      = nm.ckpt("mtp.layers.0.self_attn.k_proj.weight");
        ts.attn.v      = nm.ckpt("mtp.layers.0.self_attn.v_proj.weight");
        ts.attn.q_norm = nm.ckpt("mtp.layers.0.self_attn.q_norm.weight");
        ts.attn.k_norm = nm.ckpt("mtp.layers.0.self_attn.k_norm.weight");
        ts.attn.o      = nm.ckpt("mtp.layers.0.self_attn.o_proj.weight");
        ts.mlp.norm    = nm.ckpt("mtp.layers.0.post_attention_layernorm.weight");
        ts.mlp.gate    = nm.ckpt("mtp.layers.0.mlp.gate_proj.weight");
        ts.mlp.up      = nm.ckpt("mtp.layers.0.mlp.up_proj.weight");
        ts.mlp.down    = nm.ckpt("mtp.layers.0.mlp.down_proj.weight");
        ts.lm          = nm.ckpt("lm_head.weight");

        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)m.n_main, m.kv_full));
        m.mtp.media = m.media;
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)m.n_main, m.kv_full, m.n_rot, m.w_tok,
                                   m.w_lm_head, final_wadd, tw, ts));
        m.have_mtp = true;

        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = 1;
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
    }

    /* ---- the DFlash2 drafter, at absolute layers n_main.. of its own KV group. Gated on the same
     * two things the taps are: a drafter in the container, and a deployment that means to
     * speculate. `n_taps` is already both. */
    if (m.n_taps > 0) {
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)m.n_main));
        if (nextn > 0)
            rad_note(b, "the DFlash2 drafter DISPLACES the %lld-layer MTP head, which is not "
                        "declared and whose weights this container therefore does not carry. One "
                        "drafter a container: the two are 4.3 GB between them and a deployment "
                        "drafts with one of them. Convert without --draft-model for the MTP one",
                     (long long)nextn);
    }

    rad_note(b, "rank %d/%d: %lld trunk layers, %lld full attention + %lld gated delta net, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main,
             (long long)(m.n_main / interval), (long long)(m.n_main - m.n_main / interval),
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "trunk weights are OCP-MX " QWEN35_QUANT ": %s [N,K] codes with an E8M0 [N,%d] "
                "exponent plane%s, one stored weight taken by gemm_nt_q at dtype " QWEN35_QUANT
                "a8 -- no repacking and no dequantisation, so the trunk costs what the "
                "checkpoint does",
             QWEN35_MX_CODE_NAME, RAD_MX_GROUP,
             std::strcmp(QWEN35_QUANT, "mxfp4") == 0 ? " and a per-row reference the layout "
                                                     "derives" : "");
    rad_note(b, "trunk activations are quantised to E4M3 with ONE f32 scale per row every step "
                "(quant_act_fp8_row and the fused rmsnorm/gated/gate row forms): one quantiser per "
                "activation that feeds a GEMM, not per GEMM");
    rad_note(b, "the lm_head, the MTP head and the DFlash2 drafter are BLOCK FP8, not " QWEN35_QUANT
                ": the widest GEMM in the model and the two drafters' releases are block-scaled "
                "fp8, and both are served by the fp8 blocks and the fp8 logits path");
    rad_note(b, "NOT quantised: every norm, A_log, dt_bias, conv1d, the embedding, and the delta "
                "net's in_proj_a / in_proj_b -- 48 columns each, which is not a whole %d-row "
                "group, so they run as a separate bf16 gemm_nt", RAD_MX_GROUP);
    rad_note(b, "gdn: %lld key heads x %lld, %lld value heads x %lld, conv width %lld, chunk %lld; "
                "conv row %lld channels, a|b row %lld",
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width, (long long)gc.chunk,
             (long long)conv_dim, (long long)gc.ab_dim());
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
    if (m.have_mtp)
        rad_note(b, "the MTP head is declared: %lld NextN layer(s) exist under an `mtp.` prefix of "
                    "their own and the FIRST is served, as absolute layer %lld of the "
                    "full-attention KV group. It drafts; the trunk verifies",
                 (long long)nextn, (long long)m.n_main);
    else if (nextn > 0)
        rad_note(b, "%lld MTP/NextN layer(s) sit BESIDE the %lld trunk layers -- under an `mtp.` "
                    "prefix of their own, not as layers %lld.. -- and are NOT declared, because "
                    "this deployment asked for no speculative window (max_spec 0)",
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

    /* AN MTP PASS runs the head and NOTHING ELSE. The sign of `draft_pass` says which kind; the
     * trunk already ran this step and left its hidden states in `h`. */
    if (batch->draft_pass != 0) {
        /* THE DRAFTER'S OWN cos/sin ROWS FIRST, AND THE TRUNK DID NOT WRITE THEM. */
        if (m.op_rope_cs && batch->n_tok <= qk_fuse_rows(m.g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok,
                        rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));
        if (m.have_df) {
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                      m.df.context(c, batch);
            return;
        }
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container has neither an `mtp.` block nor a `draft.` one, or "
                             "declare ran at max_spec 0, which is the engine saying it would not "
                             "draft");
            return;
        }
        m.mtp.step(c, batch);
        return;
    }

    if (m.w_toks)
        RAD_ISSUE(c, m.op_embed,
                  praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok), RAD_W(m.w_toks),
                  brows(m.b_x, batch->n_tok), RAD_NONE);
    else
        RAD_ISSUE(c, m.op_embed,
                  praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
                  brows(m.b_x, batch->n_tok));
    m.mrows.step(c, batch, m.b_x, batch->n_tok);
    /* The vocab shard's reduction -- only where layer 0's norm will not do it. With kMxArFold true
     * layer 0's norm absorbs it (`kArExact`), so this goes out standalone only where the exact
     * fused wire is not reachable: the `!(kMxArFold && ...)` guard is the same one the blocks use
     * to decide whether to SKIP their own collective, and the two must agree or the message is
     * issued by neither. */
    if (m.op_embed_ar && !(kMxArFold && ar_wire_is_exact(m.g, batch->n_tok)))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    /* ONE FILL A STEP, HERE, because every attention layer below reads it. */
    if (m.op_rope_cs && batch->n_tok <= qk_fuse_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok,
                    rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));

    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        if (l.full) l.attn.step(c, batch);
        else        l.gdn.step(c, batch);
        dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_x, m.a_h.x);
        l.mlp.step(c, batch);
        dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_x, m.a_h.x);

        /* THE TAP, AFTER the layer, because what a drafter conditions on is this layer's OUTPUT. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.a_h.x, batch->n_tok));

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        /* THE VERIFY HEAD TAKES THE TARGET'S ROWS ONLY WHEN THE BATCH SAYS THEY ARE SAFE. The gate
         * is the builder's (batch->verify_head, abi/rad_runtime.h) -- greedy, or top_k <= R/4 with
         * min_p 0 and no grammar -- so a single unsafe request sends the whole step down the exact
         * fp8 GEMM below, which is what keeps the default and every gated-off step byte-identical. */
        if (m.verify_head && batch->verify_head) {
            const int64_t n = batch->n_out;
            RAD_ISSUE_N(c, m.op_vh_q, n,
                        brows(m.b_hout, n), brows(m.b_vh_q, n), brows(m.b_vh_s, n),
                        RAD_B(m.b_vh_sum));
            RAD_ISSUE_N(c, m.op_vh_log, n,
                        brows(m.b_vh_q, n), brows(m.b_vh_s, n), RAD_W(m.df.w_dh),
                        RAD_W(m.df.w_dhs), brows(m.b_vh_log, n), RAD_B(m.b_vh_sum), RAD_NONE);
            if (m.op_vh_merge) {
                RAD_ISSUE_N(c, m.op_vh_topk, n, brows(m.b_vh_log, n), brows(m.b_vh_cand, n),
                            brows(m.b_vh_unary, n), brows(m.b_vh_pair, n));
                RAD_ISSUE(c, m.op_vh_gath, brows(m.b_vh_pair, n), brows(m.b_vh_gath, n));
                RAD_ISSUE_N(c, m.op_vh_merge, n, brows(m.b_vh_gath, n), brows(m.b_vh_cand, n),
                            brows(m.b_vh_unary, n));
            } else {
                RAD_ISSUE_N(c, m.op_vh_topk, n, brows(m.b_vh_log, n), brows(m.b_vh_cand, n),
                            brows(m.b_vh_unary, n), RAD_NONE);
            }
            RAD_ISSUE_N(c, m.op_vh_rerank, n,
                        brows(m.b_hout, n), RAD_W(m.w_lm_head), brows(m.b_vh_cand, n),
                        brows(m.b_vh_ridx, n), brows(m.b_vh_rval, n),
                        m.op_vh_merge ? brows(m.b_vh_pair, n) : RAD_NONE);
            RAD_ISSUE_N(c, m.op_vh_scatter, n,
                        brows(m.b_vh_ridx, n), brows(m.b_vh_rval, n), brows(m.b_logits, n));
        } else {
            RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                        brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                        brows(m.b_logits, batch->n_out));
        }
    }
}


/* ================================================================== the probe
 *
 * WHAT THIS CONTAINER'S DRAFTER IS WORTH SPECULATING TO, answered before anything is declared,
 * because `max_spec` is an input to declaration. Identical to the fp8 plugin: an MTP head is
 * serial (3 is the top of the plateau) and DFlash2 is block-diffusion (its block length is what
 * the checkpoint was trained at). 0 when the container carries no drafter. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash2")) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 8);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;   /* trained at exactly this block; any other is a refusal */
        return RAD_OK;
    }
    if (meta_int2(meta, "qwen35.nextn_predict_layers", "mtp_num_hidden_layers", 0) > 0)
        out->draft_depth = 3;
    return RAD_OK;
}

}  /* namespace QWEN35_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* ONE MORE LEVEL OF EXPANSION, and it is load-bearing. RAD_ARCH_PLUGIN stringifies its NS argument
 * to name the plugin, and `#` applies to the parameter as WRITTEN -- so passing QWEN35_NS directly
 * would register a plugin called "QWEN35_NS", which the registry would then refuse the second copy
 * of by name. An argument that is not stringified in the intermediate macro IS macro-expanded
 * before substitution, so these two forward the expanded namespace. */
#define QWEN35_PROBE_X(NS)              RAD_ARCH_PROBE(NS)
#define QWEN35_PLUGIN_X(NS, ...)        RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
QWEN35_PROBE_X(QWEN35_NS)
/* The arch id stays the family's ("qwen35"); the quantisation is exactly QWEN35_QUANT so
 * `rad-convert --quant-id <quant>` selects it and no bf16 or fp8 container can. */
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, QWEN35_QUANT, "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), OCP-MX " QWEN35_QUANT
                " weights and fp8 activations, "
                "gated attention over paged KV every fourth layer and gated delta net between")
#endif
