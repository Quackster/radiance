/* rad_block_dflash2.h -- the DFlash2 block-diffusion drafter, bf16.
 *
 * ============================== WHAT IT IS ==============================
 *
 * A SECOND, SMALLER TRANSFORMER that proposes a whole block of tokens in ONE forward pass. It is
 * the other drafter this engine serves, and it is a different shape of thing from the MTP head
 * beside it (rad_block_mtp_fp8.h):
 *
 *     MTP        depth d costs d CHAINED forwards. Round i+1 cannot start until round i has
 *                produced a token, because the token is what it embeds.
 *     DFlash2    depth d costs ONE forward over d+1 positions. The block is denoised together --
 *                the anchor token plus d mask tokens go in, d+1 hidden states come out.
 *
 * That is why depth is nearly free here and expensive there, and it is why this file has no
 * `draft_pass`: there are no rounds. Published acceptance for this drafter is 4.10-5.46 against
 * MTP's 3.74-5.02 at 7 drafts.
 *
 * The price of proposing the positions together is that position l+1 was computed WITHOUT KNOWING
 * what position l would choose. The candidate selector is what puts that dependence back: each
 * position keeps the draft head's top K, a low-rank bilinear form scores the edge between
 * consecutive positions, and a greedy walk picks one path. `dflash_select` is that kernel.
 *
 * ============================== THE TWO PASSES ==============================
 *
 * THE CONTEXT PASS, one row a committed token. The drafter never runs its layers over the context
 * -- it reads the TARGET's hidden states instead, at five intermediate layers, and turns them
 * straight into its own K and V:
 *
 *     c  = fc(concat of the trunk's residual stream at layers 5, 19, 33, 47, 61)   [n_taps*n_embd]
 *     c  = hidden_norm(c)
 *     for each draft layer:  k|v = kv_proj(c);  k = k_norm(k) then rope;  kv_store(k, v)
 *
 * The trunk's side of that -- the taps, and the `dflash_src` buffer they land in -- is in
 * qwen35_fp8.cpp. This file reads the buffer and never asks where the rows came from.
 *
 * THE QUERY PASS, one row a draft position, `block` = 1 + n_spec of them a sequence:
 *
 *     x = embed(anchor, mask, mask, ...)
 *     for each layer:
 *         h  = attn_norm(x);   coef = attn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], attn_conv_base[0])          <- prepare
 *         h  = o_proj(attention(q_proj(h), kv_proj(h)))                    <- non-causal, windowed
 *         h  = dflash_conv(h, coef[taps*NG:], attn_conv_base[1]);  x += h  <- finish
 *         h  = ffn_norm(x);    coef = ffn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], ffn_conv_base[0])
 *         h  = down(silu_mul(gate_up(h)))
 *         h  = dflash_conv(h, coef[taps*NG:], ffn_conv_base[1]);   x += h
 *     x = norm(x)
 *
 * and then, over the MASK rows of each sequence's block -- offsets 1..block-1, NOT 0..steps-1.
 * The anchor at offset 0 is a token the target has already chosen, so its row predicts nothing and
 * is there to condition the rest; each mask row predicts the token AT ITS OWN POSITION:
 *
 *     cand, unary = top_k(draft_head(x))          the candidates and their logits
 *     hp          = hidden_projection(x)          the selector's query, rank 256
 *     tokens      = dflash_select(cand, unary, hp, anchor, predecessor, successor)
 *
 * ============================== WHAT IS SHARED WITH THE TARGET ==============================
 *
 * The embedding table and the lm_head, both. The checkpoint ships neither -- 81 tensors, no
 * `embed_tokens` and no `lm_head` -- so a drafter without a target is not a model. `mask_token_id`
 * is an ordinary id in the target's vocabulary (248070 here), which is why there is no separate
 * mask embedding to load.
 *
 * The head reads the 2-BIT copy of the lm_head, exactly as the MTP head does and for exactly the
 * reason rad_block_mtp_fp8.h gives at op_dlogits: a draft is a proposal, the target verifies every
 * one of them with its own untouched head, and speculative decoding is distribution-preserving --
 * so a coarser draft costs acceptance and cannot change a token the model emits. Here it also buys
 * something MTP does not need: the head stays a SINGLE-RANK computation, so the candidate ids are
 * global without an all-gather and the selector's codebooks can be indexed directly.
 *
 * ============================== THE NORMS ARE PLAIN ==============================
 *
 * `w`, not `1 + w`, unlike every norm in the trunk this drafter is bolted to. vLLM says so by
 * using the ordinary RMSNorm rather than the family's GemmaRMSNorm, and the checkpoint agrees:
 * hidden_norm sits at 0.11..1.24 (mean 0.82), input_layernorm at 0.05..1.70, k_norm at 0.32..2.53
 * and the final norm at 2.45..5.25. All positive, all centred near one -- which is a plain gain
 * and is not what a `1 + w` tensor looks like. So `Geom::wadd` and `wadd_qk` are ZERO here while
 * the trunk's are one, and the two geometries are separate structs for this reason among others.
 *
 * ============================== SHARDING ==============================
 *
 * The layers shard the way any transformer's do: q/kv and gate_up by row, o and down by column
 * with an all-reduce. Ten collectives a draft pass, each 5120 * block * 2 bytes -- 80 KiB at the
 * shipped block of 8 -- which is below the lossy wire's floor and is served exactly.
 *
 * THREE THINGS ARE REPLICATED AND EACH IS A DECISION:
 *
 *   fc              [n_embd, n_taps*n_embd], 262 MiB. It could be COLUMN-sharded, since its input
 *                   is the replicated residual stream and its output would then be a partial sum:
 *                   131 MiB a rank and half the read bytes on the context pass's widest GEMM. Not
 *                   done, because it puts an all-reduce on the PREFILL path to save a small
 *                   fraction of it, and the real fix for the context pass's cost is not to run it
 *                   over the whole prompt at all (see below).
 *   conv weights    the projection and the base kernel, 26 MiB a layer. The convolution is over
 *                   the whole replicated hidden row and every rank needs every coefficient; a row
 *                   shard would give each rank a fraction of the groups.
 *   the codebooks   [n_vocab, 256] each, 127 MiB each. They are gathered by GLOBAL token id and
 *                   the ids come out of a whole-vocabulary draft head, so a vocabulary shard would
 *                   let a rank score only the candidates that happened to land in its own slice.
 *                   The gather is 128 rows a pass, so the bytes READ are nothing; it is the
 *                   residency that costs.
 *
 * ============================== THE CONTEXT PASS IS THE EXPENSIVE HALF ==============================
 *
 * In bytes: `fc` is 262 MiB and the five KV projections read 3.4 GB over a 2048-token prefill
 * chunk, which is a real fraction of what the chunk costs.
 *
 * THE FIX IS NOT A FASTER GEMM. The drafter's attention has a 2048-token window, so a context
 * position more than 2048 before the query is never read: for a prompt of any real length, all but
 * the last 2048 rows of the context pass are computed and thrown away. The scheduler can simply
 * not run it over them. That is scheduler work and it is noted here rather than done here.
 */
#ifndef RAD_BLOCK_DFLASH2_H
#define RAD_BLOCK_DFLASH2_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

/* A row slice of a WEIGHT, the same thing brow_slice() is for a buffer. The conv's base kernel is
 * [2, taps, n_embd] and the two sides are the prepare tap set and the finish tap set; there is one
 * checkpoint tensor and the op takes one side at a time. */
inline RadOperand wrow_slice(rad_weight h, int64_t first, int64_t n, int64_t pitch) {
    RadOperand o = RAD_W(h);
    o.offset = first * pitch;
    o.rows   = n;
    return o;
}

struct Dflash2 {
    enum { MAX_LAYERS = 8 };

    /* Everything the drafter's own config.json says that its geometry does not. Read by the
     * plugin from `draft.*` metadata keys and handed in whole, so this file reads no environment
     * and no container. */
    struct Config {
        int64_t n_taps   = 0;     /* how many trunk layers `fc` concatenates */
        int64_t block    = 0;     /* query rows a sequence: the anchor plus n_spec mask tokens */
        int64_t taps     = 2;     /* the convolution's */
        int64_t group    = 16;
        int64_t sel_rank = 256;   /* the codebooks' width */
        int64_t sel_topk = 16;    /* candidates kept per position */
        int64_t window   = 0;     /* sliding attention width, 0 for none */
        int     causal   = 0;
        int64_t steps() const { return block - 1; }
        int64_t ng(int64_t n_embd) const { return group ? n_embd / group : 0; }
        /* The kernel projection emits [2, taps, NG] per token -- one coefficient set for `prepare`
         * and one for `finish` -- and `dflash_conv` reads a slice of it at the FULL row pitch. */
        int64_t coef_dim(int64_t n_embd) const { return 2 * taps * ng(n_embd); }
    };

    struct Wire {
        /* the context pass */
        rad_buf src  = 0;   /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        rad_buf c    = 0;   /* [max_tok, n_embd]           fc's output, then normed in place */
        /* K AND V ARE TWO BUFFERS, not one [max_tok, 2*kv_dim] with K in its left half. Both
         * halves are written by one projection and read by one kv_store, so fusing them is the
         * obvious thing -- and it is wrong, because of the PER-HEAD NORM in between. `rmsnorm`
         * takes one row pitch. K inside a fused row is `head_dim` apart within a token and
         * `2*kv_dim` apart between tokens, and no single pitch is both; answering with the
         * between-tokens stride normalises every eighth head against the wrong numbers. libr4d
         * refuses that operand (libr4d/r4d_model.cpp, rows_pitch), and the drafter gives it rows
         * that are what they claim to be. The trunk's attention block splits
         * k and v for the same reason -- see rad_block_attn_gated.h. */
        rad_buf ck   = 0;   /* [max_tok, kv_dim]           ONE layer's K, reused across layers */
        rad_buf cv   = 0;   /* [max_tok, kv_dim]           ONE layer's V, reused across layers */
        /* the query pass */
        rad_buf x    = 0;   /* [qrows, n_embd]  the drafter's residual stream */
        rad_buf h    = 0;   /* [qrows, n_embd]  normed input, o/down output, and the final norm */
        rad_buf h2   = 0;   /* [qrows, n_embd]  the convolution's output -- it may not alias x */
        rad_buf coef = 0;   /* [qrows, coef_dim] */
        rad_buf q    = 0;   /* [qrows, q_dim] */
        rad_buf k    = 0;   /* [qrows, kv_dim]  -- two buffers, and `ck` above says why */
        rad_buf v    = 0;   /* [qrows, kv_dim] */
        rad_buf attn = 0;   /* [qrows, q_dim] */
        rad_buf gate_up = 0;/* [qrows, 2 * n_ff] */
        rad_buf ffn  = 0;   /* [qrows, n_ff] */
        /* THE FP8 TWINS OF THE FOUR ACTIVATIONS A LINEAR READS. `quant_act_fp8` writes them and
         * the six `gemm_nt_q` linears read them; nothing else touches them, and the bf16 buffer
         * beside each stays the thing the rest of the graph reads and writes.
         *
         * ONE PAIR PER ACTIVATION, NOT PER LINEAR. q and k|v both read h2, so quantising it once
         * is the difference between two launches a layer and three -- the same argument rad_fp8.h
         * makes for keeping QuantFP8 a separate struct from LinearFP8. h2 is quantised twice a
         * layer all the same, because the convolution rewrites it between the attention half and
         * the feed-forward half. */
        rad_buf srcq = 0, srcs = 0;  /* [max_tok, n_taps*n_embd] e4m3 | [.., /128] f32 */
        rad_buf cq   = 0, cs   = 0;  /* [max_tok, n_embd] */
        rad_buf h2q  = 0, h2s  = 0;  /* [qrows,   n_embd] */
        rad_buf atq  = 0, ats  = 0;  /* [qrows,   q_dim] */
        rad_buf ffq  = 0, ffs  = 0;  /* [qrows,   n_ff] */
        /* the head and the selector, at one row per DRAFT POSITION (block - 1 a sequence) */
        rad_buf sel_h = 0;  /* [srows, n_embd]     the gathered final hidden states */
        rad_buf hp    = 0;  /* [srows, sel_rank]   hidden_projection of them */
        rad_buf dq    = 0;  /* [srows, n_embd]  i8   the 2-bit head's quantised activation */
        rad_buf ds    = 0;  /* [srows, 1]       f32  one scale a row */
        rad_buf dsum  = 0;  /* [n_embd/128, pad] f32 GROUP-major, quant_act_i8's asum plane */
        rad_buf dlog  = 0;  /* [srows, head_rows(g)] bf16  the coarse logits */
        rad_buf cand  = 0;  /* [srows, sel_topk] i32  candidate token ids, global */
        rad_buf unary = 0;  /* [srows, sel_topk] f32  and their logits */
        /* The vocab-parallel pair, declared only when head_sharded(g). Both are f32 (int32 id,
         * float value) pairs: this rank's top-R, and the all-gathered world_size * R of them. */
        rad_buf dpair = 0;  /* [srows, sel_topk, 2]         f32 */
        rad_buf dgath = 0;  /* [srows, world * sel_topk, 2] f32 */
        rad_buf tokens = 0; /* [max_seqs, steps] i32  THE DRAFT -- the engine reads this by name */
    };

    /* WHETHER THE 2-BIT DRAFT HEAD IS VOCAB-SHARDED. The test itself is rad_arch.h's
     * `vocab_head_sharded` -- shared, because the second block-diffusion drafter in this library
     * asks it too and both the op and the buffer under it have to agree. These two names stay so
     * that whoever declares Dflash2's buffers can ask the BLOCK rather than re-derive the rule.
     *
     * Sharding halves the widest read on the draft path -- 341 MiB of codes and scales at 248320
     * rows when it is replicated -- at the price of one all_gather of R pairs a row. At 248320
     * over two ranks the shard is 124160, which is 7760 whole 16-row blocks. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    /* Rows of the draft head, and therefore the width of `dlog`, on THIS rank. */
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    Geom    g{};      /* the DRAFTER's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, so the KV group's layer_slot lookup finds these five */
    int64_t qrows = 0, crows = 0, srows = 0;
    int64_t gated_rows = 0;    /* the band op_gug was declared over; 0 when there is no fold */
    rad_kvgroup kv = 0;

    /* model-scope weights.
     *
     * THE SIX LINEARS ARE FP8 AND THE CHECKPOINT IS BF16. rad_fp8.h's DenseFP8 declares each as
     * block fp8 and the recipe makes it so -- and the reason is bytes: the drafter is read WHOLE on
     * every decode step, which makes it one of the largest single items in the decode budget
     * after the trunk's own GEMMs.
     * Nothing it computes can change an emitted token, because every draft is verified against the
     * trunk exactly; the only thing quantising it can move is acceptance. */
    DenseFP8   w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_selp = 0, w_pred = 0, w_succ = 0;
    /* ...and their f32 scales, when the container holds the two codebooks as E4M3 codes (a scale a
     * row). Both or neither: the selector op is one or the other. */
    rad_weight w_preds = 0, w_succs = 0;
    rad_weight w_tok = 0;                    /* the TARGET's embedding table */
    rad_weight w_toks = 0;                   /* ...and its E4M3 scale, set before declare() */
    rad_weight w_dh = 0, w_dhs = 0;          /* the 2-bit copy of the TARGET's lm_head */
    bool       dh4 = false;                  /* ...or a 4-bit one, when the recipe made it u4 */
    bool       dh8 = false;                  /* ...or an int8 one (i8 codes, bf16 scale a 128) */

    /* per-layer weights */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    /* K AND V ARE TWO WEIGHTS, and the fp8 layout hook is why. One fused [2*kv_dim, n_embd]
     * tensor issued a half at a time through wrow_slice would cost a launch and no bytes -- and a
     * block-scaled weight cannot be used that way. The layout of an operand is a function of the
     * OP's N, and this op's N is kv_dim because it computes one half; the hook would be handed a
     * [2*kv_dim/128][K/128] plane for a [kv_dim, K] op and refuse it. Two weights make the op's N
     * and the planes' rows the same number again, shard independently at kv_dim, and issue the
     * same two launches. */
    DenseFP8   w_q[MAX_LAYERS]{},  w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};
    rad_weight w_acp[MAX_LAYERS] = {}, w_acb[MAX_LAYERS] = {};
    rad_weight w_fcp[MAX_LAYERS] = {}, w_fcb[MAX_LAYERS] = {};

    /* context-pass ops */
    rad_op op_cfc = 0, op_cnorm = 0, op_crope = 0, op_cstore = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    /* THE ACTIVATION QUANTISERS, one per activation that feeds a linear. Weightless and shape-only,
     * so one declaration each serves every layer -- the per-layer ops above are per-layer only
     * because a weight's prefetch interval comes from the op that declares it. */
    rad_op op_qsrc = 0, op_qc = 0;                  /* the context pass's two */
    rad_op op_qh2 = 0, op_qat = 0, op_qff = 0;      /* the query pass's three */

    /* query-pass ops. The weightless ones are declared once and issued per layer: an op's declared
     * read and write sets are what the buffer planner reads, and they are the same every layer. */
    rad_op op_embed = 0, op_embed_ar = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {};
    rad_op op_acoef[MAX_LAYERS] = {}, op_fcoef[MAX_LAYERS] = {};
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_gug[MAX_LAYERS] = {};   /* the gate_up|silu|quant fold; 0 when nothing serves it */
    rad_op op_aconv[MAX_LAYERS] = {}, op_fconv[MAX_LAYERS] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    rad_op op_act = 0, op_ar = 0, op_add = 0, op_final = 0;

    /* head and selector */
    rad_op op_gather = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0;
    rad_op op_dgather = 0, op_dmerge = 0;   /* the vocab-parallel pair; 0 when the head is whole */
    rad_op op_hp = 0, op_sel = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck_prefix,
                 const char* ck_lm, const Wire& wire);
    /* The context pass, one row a committed token. `batch` carries those rows and the draft KV
     * group's slot mapping; nothing about the query block is read. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass and the selector, one call, `block` rows a sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dflash2::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                            int base_layer, rad_kvgroup group, rad_weight tok_embd,
                            const char* ck, const char* ck_lm, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_lm || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: dflash2 has %lld layers; this block carries at most %d\n",
                (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0 || cfg.sel_rank <= 0 || cfg.sel_topk <= 0)
        return RAD_E_INVAL;

    const int64_t NG   = cfg.ng(g.n_embd);
    const int64_t CD   = cfg.coef_dim(g.n_embd);
    const int64_t qd   = q_dim(), kvd = kv_dim();

    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dflash.fc"), g.n_embd, cfg.n_taps * g.n_embd,
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(),
                              { nm.ckpt("%s.fc.weight", ck) }));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.norm.weight"),        nm.ckpt("%s.norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_proj.weight"),
                          nm.ckpt("%s.candidate_selector.hidden_projection.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_pred"),
                          nm.ckpt("%s.candidate_selector.predecessor_codebook", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_succ"),
                          nm.ckpt("%s.candidate_selector.successor_codebook", ck)));

    w_hnorm = decl_w(b, nm.f("dflash.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dflash.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_selp  = decl_w(b, nm.f("dflash.sel_proj.weight"), RAD_BF16, {cfg.sel_rank, g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* THE SELECTOR'S CODEBOOKS AT E4M3, when the container holds them so. Two [248320, 256] tables
     * of bf16 are 2 x 121 MiB resident; at E4M3 codes and an f32 scale a row they are half that,
     * and the difference goes to the KV cache (~3.9k tokens). The walk reads K candidate rows and
     * one predecessor row a step, so the format costs no time. What it can move is acceptance, and
     * only that: a draft is verified. Decided from what the container says each table IS (the
     * same rule the token embedding follows), so a container with bf16 codebooks is declared and
     * served exactly as before, and a recipe line is all it takes to opt in. A scale plane that is
     * one for the whole table is read too (a 1 x 1 view); the shipped recipe writes one a row. */
    {
        RadEncoding ep{}, es{};
        auto e4m3 = [](const RadEncoding& e) {
            return e.n_planes == 2 && e.plane[0].dtype == RAD_F8E4M3 &&
                   rad_enc_find(&e, "scale") == 1 && e.plane[1].dtype == RAD_F32;
        };
        const bool qp = weight_enc(b, nm.f("dflash.sel_pred"), &ep) && e4m3(ep);
        const bool qs = weight_enc(b, nm.f("dflash.sel_succ"), &es) && e4m3(es);
        if (qp != qs) {
            fprintf(stderr, "radiance: dflash2's selector codebooks are %s for pred and %s for "
                    "succ; the selector op takes both E4M3 or both bf16\n",
                    qp ? "E4M3" : "not E4M3", qs ? "E4M3" : "not E4M3");
            return RAD_E_UNSUPPORTED;
        }
        if (qp) {
            auto rows_of = [&](const RadEncoding& e) -> int64_t {
                return e.plane[1].block[0] == 1 ? g.n_vocab_all : 1;
            };
            w_pred  = decl_view(b, nm.f("dflash.sel_pred"), nullptr, "codes", RAD_F8E4M3,
                                {g.n_vocab_all, cfg.sel_rank}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                grp_model());
            w_preds = decl_view(b, nm.f("dflash.sel_pred.scale"), nm.f("dflash.sel_pred"), "scale",
                                RAD_F32, {rows_of(ep), 1}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                grp_model());
            w_succ  = decl_view(b, nm.f("dflash.sel_succ"), nullptr, "codes", RAD_F8E4M3,
                                {g.n_vocab_all, cfg.sel_rank}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                grp_model());
            w_succs = decl_view(b, nm.f("dflash.sel_succ.scale"), nm.f("dflash.sel_succ"), "scale",
                                RAD_F32, {rows_of(es), 1}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                                grp_model());
            if (!w_pred || !w_preds || !w_succ || !w_succs) return RAD_E_INVAL;
        } else {
            w_pred  = decl_w(b, nm.f("dflash.sel_pred"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                             RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
            w_succ  = decl_w(b, nm.f("dflash.sel_succ"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                             RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        }
    }

    /* THE 2-BIT DRAFT HEAD. The same declaration the MTP head makes, for the same reasons, argued
     * at length in rad_block_mtp_fp8.h: `dflash.draft_head.weight` is the target's lm_head under a
     * name of its own, which the recipe quantises to the 2-bit grid (u2 codes, an f16 scale and a
     * u8 zero a group of 128) -- one checkpoint tensor, two logical weights. The GEMM takes the
     * codes as one operand and the scale and zero planes, interleaved by its relayout, as the
     * other.
     *
     * IT IS SHARDED, AND THE MERGE IS WHAT MAKES THAT SAFE. The draft path never goes through the
     * sampler, so its argmax is a `row_topk` over one rank's plane -- and a sharded plane on its
     * own would let each rank propose only from its own half, while the selector's codebooks are
     * gathered by GLOBAL token id and would then see only the candidates that landed locally. The
     * merge runs BEFORE the selector: each rank's top-R comes back as global ids, an all_gather
     * moves R pairs a row, and `row_topk_merge` reduces them to the same global top-R an unsharded
     * head would have found -- exactly, ties included (docs/OPS.md). Both ranks then hold the same
     * candidate set and the codebooks are replicated, so everything downstream is unchanged.
     *
     * What it buys is the widest read on the draft path halved -- 341 MiB of codes and scales at
     * the whole vocabulary -- and that much device memory given back per rank. What it costs is
     * one all_gather of `world * R` pairs a row, which at R = 16 is 2 KiB: the link's latency and
     * essentially no bytes. */
    enum { DF_W2_GROUP = 128 };
    const bool dh_shard = head_sharded(g);
    const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const char* dhn = nm.f("dflash.draft_head.weight");
    RAD_ARCH_TRY(map_copy(b, dhn, ck_lm));
    w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {head_rows(g), g.n_embd},
                      RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    w_dhs = decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                      {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
    if (!w_dh || !w_dhs) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.attention_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_base", i),
                              nm.ckpt("%s.layers.%d.attention_conv.base_kernel", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_base", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.base_kernel", ck, i)));

        w_anorm[i] = decl_w(b, nm.f("dflash.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_fnorm[i] = decl_w(b, nm.f("dflash.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dflash.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        /* K, V AND Q ARE THREE WEIGHTS, which is not the fusion every other attention block in
         * this library makes. Q is separate because of the context pass: it needs k and v over the
         * trunk's projected hidden state and has no use for q at all, so a fused qkv would make it
         * compute and discard a q_dim-wide GEMM per layer per context token. K and V are separate
         * because a block-scaled weight's layout follows its op's N, and both passes take one of
         * them at a time -- see the note beside the declarations. */
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dflash.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dflash.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        w_qn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_kn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dflash.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_gate_up", i),
                                     2 * g.n_ff, g.n_embd,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(first_layer + i),
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
        w_acp[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_acb[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcp[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcb[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));

    }

    /* NON-CAUSAL, WINDOWED. The block is denoised as a whole, so every query row sees every other
     * row of its own block as well as the context -- `causal` 0 is the drafter's own config
     * (`is_causal: false`) and not a default. The window is the checkpoint's `sliding_window`, and
     * it is the reason the context pass has a bounded amount of real work in it.
     *
     * DECLARED FIRST, before either kv_store, even though it runs long after both. `block_size` is
     * RAD_DERIVED -- the KV block is a property of the resolved attention kernel and not a number
     * anyone gets to choose -- so it is omitted here and READ BACK with rad_kv_block_size() for the
     * two stores, which need it as a parameter. Ask before the attention op is declared and the
     * answer is zero. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {w.q}, {w.attn});

    /* ---------------------------------------------------------------- the context pass */
    /* EVERY LINEAR HERE IS `fp8a8`, and every activation that feeds one gets a `quant_act_fp8` in
     * front of it. The weights are E4M3 with a 128x128 scale plane (rad_fp8.h's DenseFP8 says
     * why), and there is no `fp8a16` band anywhere in this block: that kernel is M <= 1, and
     * neither pass here ever runs at one row -- a context pass is a committed chunk and a query
     * pass is `block` rows by construction.
     *
     * AND NOTHING HERE IS ROW-SLICED. The fp8a8 pair covers every M (narrow to 16, tiled above),
     * so a context pass over a 2048-token prefill chunk is one launch a layer rather than one per
     * 64 rows.
     *
     * The convolutions' coefficient projections and the selector's hidden projection are still
     * bf16, and they are not sliced either: libr4d's bf16 GEMM row is C_LE("M", 65536), and all
     * three of these carry K = n_embd, a multiple of 16, so the wide row serves them whole. */
    op_qsrc = rw(b, RAD_OP(b, "quant_act_fp8",
                    RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.src}, {w.srcq, w.srcs});
    op_cfc = rw(b, RAD_OP(b, "gemm_nt_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", g.n_embd),
                              RAD_INT("K", cfg.n_taps * g.n_embd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                   RAD_WEIGHTS(w_fc.w, w_fc.s)),
               {w.srcq, w.srcs}, {w.c});
    /* IN PLACE. Safe for any implementation that computes a row's scale before writing the row,
     * which none can avoid doing; the per-head q/k norms below run in place for the same reason. */
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    /* ONE QUANTISE FOR ALL FIVE LAYERS. `c` is the normed projection of the trunk's taps and every
     * layer's k|v reads that same buffer, so this is one launch a context pass and not five. */
    op_qc = rw(b, RAD_OP(b, "quant_act_fp8",
                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.c}, {w.cq, w.cs});
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* TWO OPS OF THE SAME SHAPE, one per weight. An op's declared weight list is what gives a
         * weight its prefetch interval and its residency class, so two weights want two ops even
         * where one would issue both. */
        op_ck[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_k[i].w, w_k[i].s)),
                     {w.cq, w.cs}, {w.ck});
        op_cv[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_v[i].w, w_v[i].s)),
                     {w.cq, w.cs}, {w.cv});
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    /* K-ONLY ROPE. `rope` rotates the first n_head + n_head_kv heads of its operand and leaves the
     * V heads that follow alone, so a K-only tensor is described by n_head = n_head_kv and
     * n_head_kv = 0: all the heads there are rotate, none follow them. Weightless, so one op
     * serves every layer. */
    op_crope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    op_embed = w_toks
        ? rw(b, RAD_OP(b, "embed_lookup_q",
                 RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                            RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                            RAD_INT("vocab_offset", g.vocab_off)),
                 RAD_WEIGHTS(w_tok, w_toks)),
             {}, {w.x})
        : rw(b, RAD_OP(b, "embed_lookup",
                 RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                            RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                            RAD_INT("vocab_offset", g.vocab_off)),
                 RAD_WEIGHTS(w_tok)),
             {}, {w.x});
    /* The trunk's embedding is vocab-sharded, so this gather is partial and takes the same exact
     * reduction. It is on the DRAFT path, which runs once a step ahead of the trunk: a second
     * small collective. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});
    /* The query pass's three quantisers, declared once and issued per layer -- weightless ops with
     * the same shape every layer, exactly like the ropes and the residual add below. h2 is
     * quantised TWICE a layer because the convolution rewrites it between the two halves; attn and
     * ffn once each. */
    op_qh2 = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.h2}, {w.h2q, w.h2s});
    op_qat = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.attn}, {w.atq, w.ats});
    op_qff = rw(b, RAD_OP(b, "quant_act_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.ffn}, {w.ffq, w.ffs});

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_anorm[i])),
                        {w.x}, {w.h});
        op_acoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acp[i])),
                        {w.h}, {w.coef});
        op_qp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", qd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_q[i].w, w_q[i].s)),
                     {w.h2q, w.h2s}, {w.q});
        op_kp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_k[i].w, w_k[i].s)),
                     {w.h2q, w.h2s}, {w.k});
        op_vp[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", kvd),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_v[i].w, w_v[i].s)),
                     {w.h2q, w.h2s}, {w.v});
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", g.n_embd),
                                    RAD_INT("K", qd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_o[i].w, w_o[i].s)),
                     {w.atq, w.ats}, {w.h});
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fnorm[i])),
                        {w.x}, {w.h});
        op_fcoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcp[i])),
                        {w.h}, {w.coef});
        op_gu[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", 2 * g.n_ff),
                                    RAD_INT("K", g.n_embd), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                     {w.h2q, w.h2s}, {w.gate_up});
        /* THE SAME FOLD THE TRUNK'S MLP HAS, over the same weights and into the same activation;
         * rad_block_mlp_fp8.h declares it beside the pair for the same reason. Unfused, a layer
         * runs gate_up -> silu_mul -> quant, which is two extra launches and two extra passes over
         * the [T, n_ff] product. The gated kernel is also the faster of the two on the SAME shape,
         * because the fold saves the bf16 round trip.
         *
         * NOT RAD_ARCH_TRY: a kernel set without the row is the ordinary case and the pair below
         * still runs. `gated_rows` is the band it was declared over and the issue site checks it,
         * because M > 64 is a refusal in the kernel and not a slower path. */
        {
            const int64_t hi = qrows < 64 ? qrows : 64;
            if (hi > 0) {
                op_gug[i] = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * g.n_ff),
                                              RAD_INT("K", g.n_embd),
                                              RAD_INT("group", RAD_FP8_BLOCK),
                                              RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                               {w.h2q, w.h2s}, {w.ffq, w.ffs});
                if (op_gug[i]) gated_rows = hi;
            }
        }
        op_dn[i] = rw(b, RAD_OP(b, "gemm_nt_q",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", g.n_embd),
                                    RAD_INT("K", g.n_ff), RAD_INT("group", RAD_FP8_BLOCK),
                                    RAD_STR("dtype", "fp8a8")),
                         RAD_WEIGHTS(w_dn[i].w, w_dn[i].s)),
                     {w.ffq, w.ffs}, {w.h});
    }

    /* The convolutions. `block_size` is the QUERY BLOCK -- the mask that kills the shifted tap at
     * the first row of each block is what keeps one sequence's block from convolving with the
     * previous sequence's, so it is the batch's per-sequence row count and not a tuning number.
     *
     * ONE OP PER (LAYER, CONVOLUTION) even though every one of them is the same shape, because the
     * base kernel is a WEIGHT and an op's declared weight list is what gives a weight its prefetch
     * interval and its residency class. A single shared declaration would leave ten weights that
     * no op admits to using. Each is issued twice -- `prepare` and `finish` -- against the two
     * halves of one [2, taps, n_embd] tensor. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_aconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acb[i])),
                        {w.h, w.coef}, {w.h2});
        op_fconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcb[i])),
                        {w.h, w.coef}, {w.h2});
    }
    op_qrope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.x, w.h2}, {w.x});
    op_final = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* ---------------------------------------------------------------- the head and the selector
     *
     * THE MASK ROWS, offsets 1..block-1 of each sequence's block. Offset 0 is the ANCHOR -- the
     * token the target has already chosen -- and its row predicts nothing: it is there so the mask
     * rows have something to attend to. Each mask row predicts the token at ITS OWN position, so
     * the head runs over `steps` rows a sequence and not `block`. Getting this off by one is the
     * kind of mistake that costs nothing visible: the drafts are simply wrong, acceptance
     * collapses, and the model's own output stays exactly correct.
     *
     * The gather is what makes those rows contiguous, and it is also what fixes the head's row
     * numbering at s * steps + l -- which is the numbering `dflash_select` reads. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.sel_h});
    op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_INT("group", DF_W2_GROUP), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.sel_h}, {w.dq, w.ds, w.dsum});
    op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                       RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                                  RAD_INT("K", g.n_embd), RAD_INT("group", DF_W2_GROUP),
                                  RAD_STR("dtype", "w2a8")),
                       RAD_WEIGHTS(w_dh, w_dhs)),
                   {w.dq, w.ds, w.dsum}, {w.dlog});
    /* Sharded, the top-R runs over THIS RANK'S plane and reports global ids, and it also writes
     * the pair plane the gather moves. `vocab_off` is omitted entirely when the head is
     * replicated, so that graph is the one it always was -- an optional parameter present with
     * the value 0 is not the same declaration as an absent one. */
    op_dtopk = dh_shard
        ? rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.sel_topk), RAD_INT("vocab_off", g.vocab_off),
                           RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary, w.dpair})
        : rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.sel_topk), RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary});
    if (dh_shard) {
        /* ROW-INTERLEAVED, so one draft position's pairs from every rank land side by side --
         * `row_topk_merge` reads one row at one row stride, the same reason the sampler's own
         * candidate gather carries `row`. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * cfg.sel_topk * 2),
                                      RAD_STR("dtype", "f32"),
                                      RAD_INT("row", cfg.sel_topk * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                          RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", cfg.sel_topk),
                                     RAD_INT("world_size", g.world),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.dgath}, {w.cand, w.unary});
    }
    op_hp = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", cfg.sel_rank),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_selp)),
              {w.sel_h}, {w.hp});
    /* `anchor_stride` is `block`, because the operand is the step's own token_ids: a draft pass
     * carries `block` ids a sequence and the anchor is the first of them. */
    op_sel = w_preds
        ? rw(b, RAD_OP(b, "dflash_select_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_pred, w_preds, w_succ, w_succs)),
               {w.cand, w.unary, w.hp}, {w.tokens})
        : rw(b, RAD_OP(b, "dflash_select",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_pred, w_succ)),
               {w.cand, w.unary, w.hp}, {w.tokens});

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dflash2::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;


    /* NOT SLICED. A context pass is one row a committed token, so it is as wide as the prefill
     * chunk; the fp8a8 pair covers every M, so each op here is one launch and not one per 64
     * rows. */
    RAD_ISSUE_N(c, op_qsrc, T, brows(w.src, T), brows(w.srcq, T), brows(w.srcs, T));
    RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                brows(w.c, T), RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv,
                    brows(w.ck, T), RAD_W(w_kn[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dflash2::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;


    const int64_t NG = cfg.ng(g.n_embd);
    const int64_t half = cfg.taps * NG;      /* prepare's coefficients, then finish's */

    if (w_toks)
        RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), RAD_W(w_toks),
                    brows(w.x, T), RAD_NONE);
    else
        RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        /* ---- attention, wrapped in the two halves of one convolution */
        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_acoef[i], T, brows(w.h, T), RAD_W(w_acp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, op_aconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_acb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));

        RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_q[i].w), RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, RAD_NONE);
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn, T));
        RAD_ISSUE_N(c, op_qat, T, brows(w.attn, T), brows(w.atq, T), brows(w.ats, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.atq, T), brows(w.ats, T),
                    RAD_W(w_o[i].w), RAD_W(w_o[i].s), brows(w.h, T), RAD_NONE, RAD_NONE);
        /* BEFORE the closing convolution, not after. The convolution is linear in its input and
         * its coefficients come from the replicated pre-attention norm, so either order is the
         * same arithmetic -- and reducing first keeps the residual stream's contract ("what the
         * ranks add to x has already been agreed") in one place. */
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_aconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                    wrow_slice(w_acb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h2, T), brows(w.x, T));

        /* ---- the feed-forward, wrapped in the other convolution */
        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_fcoef[i], T, brows(w.h, T), RAD_W(w_fcp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, op_fconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_fcb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.h2q, T), brows(w.h2s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.h2q, T), brows(w.h2s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE,
                        RAD_NONE);
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ffn, T), brows(w.ffq, T), brows(w.ffs, T));
        }
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffq, T), brows(w.ffs, T),
                    RAD_W(w_dn[i].w), RAD_W(w_dn[i].s), brows(w.h, T), RAD_NONE, RAD_NONE);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, op_fconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                    wrow_slice(w_fcb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h2, T), brows(w.x, T));
    }

    RAD_ISSUE_N(c, op_final, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    /* The rows that predict something: `steps` of every sequence's `block`. The scheduler names
     * them through out_ids, the same field the trunk's own logits gather reads. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h), praw(batch->out_ids, RAD_I32, S), brows(w.sel_h, S));

    /* `dsum` is quant_act_i8's GROUP-major plane, [n_embd/group][ceil(M/16)*16], so it is passed
     * whole: there is no row slice of it that means anything. */
    RAD_ISSUE_N(c, op_dquant, S, brows(w.sel_h, S), brows(w.dq, S), brows(w.ds, S), RAD_B(w.dsum));
    /* The trailing operand is mxfp4's optional reference and is absent here; an absent optional
     * operand is a null tensor in a full-length list, not a shorter list. */
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.dq, S), brows(w.ds, S), RAD_W(w_dh), RAD_W(w_dhs),
                brows(w.dlog, S), RAD_B(w.dsum), RAD_NONE);
    /* RAD_NONE is the `pairs` slot: operands are POSITIONAL, so an optional one that the graph
     * does not use is still passed, absent. Omitting it is a miscount and the engine refuses the
     * op rather than guessing which of the four was left out.
     *
     * SHARDED, the top-R over this rank's plane is not the answer -- it is this rank's CANDIDATES,
     * and `cand`/`unary` are overwritten by the merge two ops below. Writing them here anyway
     * costs nothing (they are the same R entries the pairs plane holds) and keeps one issue site
     * for both paths. */
    if (op_dmerge) {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_dmerge, S, brows(w.dgath, S), brows(w.cand, S), brows(w.unary, S));
    } else {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    RAD_NONE);
    }

    RAD_ISSUE_N(c, op_hp, S, brows(w.sel_h, S), RAD_W(w_selp), brows(w.hp, S));
    if (w_preds)
        RAD_ISSUE_N(c, op_sel, batch->n_seq, brows(w.cand, S), brows(w.unary, S), brows(w.hp, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred), RAD_W(w_preds),
                    RAD_W(w_succ), RAD_W(w_succs), brows(w.tokens, batch->n_seq));
    else
        RAD_ISSUE_N(c, op_sel, batch->n_seq, brows(w.cand, S), brows(w.unary, S), brows(w.hp, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred), RAD_W(w_succ),
                    brows(w.tokens, batch->n_seq));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DFLASH2_H */
