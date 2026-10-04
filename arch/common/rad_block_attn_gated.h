/* rad_block_attn_gated.h -- one GATED grouped-query attention block, bf16 operands, paged KV.
 *
 *   rmsnorm -> q|gate gemm, k gemm, v gemm -> qk-norm + rope + gate extraction
 *   -> kv_store -> paged attention -> attn * sigmoid(gate) -> o gemm -> [all_reduce] -> residual
 *
 * This is Qwen3.5's attention and not Qwen3's, and the difference is not a flag:
 *
 *   * `q_proj` emits the query AND a per-head output gate INTERLEAVED at a pitch of 2*head_dim.
 *     Head h's query is at h*2*head_dim and its gate at h*2*head_dim + head_dim, so neither is a
 *     contiguous half of the projection and both are strided views of one buffer. libr4d's
 *     `qk_norm_rope_gate` splits [q|gate] per head and takes the stride off the `qg` OPERAND's own
 *     row pitch, so the interleaved layout is the one to declare rather than to convert away at
 *     load. There is no `qg_pitch` PARAMETER to pass -- see pQkNormRopeGate in
 *     libr4d/r4d_rows.cpp -- the stride comes from the operand.
 *   * The gate multiplies the attention OUTPUT -- `out = attn * sigmoid(gate)` -- so it is
 *     extracted before attention and consumed after it. `output_gate_type: "swish"` in the config
 *     names the GDN block's gate, not this one; llama.cpp uses sigmoid here and it is ground truth
 *     (../llama.cpp/src/models/qwen35.cpp, build_layer_attn).
 *   * RoPE is PARTIAL -- `partial_rotary_factor` 0.25 rotates the first 64 of 256 head dims -- and
 *     it is MROPE. See the note on `mode` in declare(): with one position component per token,
 *     which is all RadBatch carries, mrope and neox are the same arithmetic, and that equality is
 *     what makes this declaration honest for text and wrong for an image span.
 *
 * K AND V ARE NOT FUSED INTO ONE PROJECTION, and that is a decision with a cost: two GEMM launches
 * a layer instead of one, and two movement units instead of one. It buys every consumer a
 * CONTIGUOUS operand. A fused [k|v] row makes the per-head k norm a rank-2 column slice whose last
 * dimension is kv_dim while the norm's width is head_dim, and a kernel that reads extents off its
 * operands -- libref does, by rule -- then walks it as if it were contiguous and normalises
 * across the k/v seam. The fused form is expressible the moment the slice carries the head
 * dimension in its own right; until then this is correct and slightly slow rather than fast and
 * silently wrong. libr4d's kernel takes a `k_pitch` for the fused form, so the w8a8 sibling can
 * make the other choice without changing this block's shape.
 */
#ifndef RAD_BLOCK_ATTN_GATED_H
#define RAD_BLOCK_ATTN_GATED_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct AttnGatedBF16 {
    /* The activation buffers this block reads and writes. Declared once by the plugin and shared
     * by every layer of this kind: layers that share a geometry share the buffer (spec §8), and
     * the liveness analysis over the declared read and write sets is what makes that safe. */
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        rad_buf gate  = 0;  /* [max_tok, q_dim]   de-interleaved, then sigmoid in place */
        rad_buf attn = 0;   /* [max_tok, q_dim]   attention output, then gated in place */
    };

    /* The checkpoint-side names for one layer. The plugin owns both sides of the map and is the
     * only component that knows both (spec §2.4), so the block takes the spelling rather than
     * inventing one -- this architecture arrives as `model.language_model.layers.N.*` from the
     * safetensors checkpoint and as `blk.N.*` from a GGUF. */
    struct Src {
        const char* norm   = nullptr;   /* input_layernorm.weight */
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer   = 0;
    rad_kvgroup kv      = 0;
    int64_t     n_rot   = 0;          /* partial rotary: the leading n_rot of head_dim */
    const char* mode    = "mrope";

    rad_weight w_norm = 0, w_qg = 0, w_k = 0, w_v = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0;

    rad_op op_norm = 0, op_qg = 0, op_kp = 0, op_vp = 0;
    rad_op op_fused = 0;                                     /* qk_norm_rope_gate, if it resolves */
    rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;   /* the fallback */
    rad_op op_attn = 0, op_kv_store = 0, op_sigmoid = 0, op_gate = 0;
    rad_op op_o = 0, op_ar = 0, op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnGatedBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                  rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                  const Src& src) {
    g = geom; w = wire; layer = l; kv = group; n_rot = rotary_dim;

    const int64_t q_dim   = g.q_dim();
    const int64_t kv_dim  = g.kv_dim();
    const int64_t qg_dim  = 2 * q_dim;          /* query and gate, one per head, interleaved */

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* KV-HEAD REPLICATION IS NOT PLUMBLED HERE. `geom_from` answers a world size above the KV head
     * count by giving every rank ONE kv head and a `kv_off` row span into a projection that stays
     * `n_head_kv_all * head_dim` rows in the container -- and this block declares its k and v with
     * the plain even row-shard, which at four ranks would hand a rank half of one KV head. The
     * shapes would declare, the numbers would be wrong, and the text would be fluent. Refuse by
     * name instead: rad_block_attn_gated_fp8.h is the block that passes the span, and a bf16 model
     * that needs TP4 plumbs it the same way. */
    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld; the KV-head replication in geom_from needs the "
                        "span passed to its k/v weights.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- weights. Shapes are [out, in]: gemm_nt is A[M,K] times B transposed [N,K], the layout
     * every transformer weight already has, so nothing is transposed at load.
     *
     * The q|gate projection row-shards by HEAD, and that works only because each head's query and
     * gate are adjacent: rows [h*2*head_dim, (h+1)*2*head_dim) are one head's pair, so a rank's
     * slice of the head range is a contiguous row range. A projection that put all queries before
     * all gates would not shard this way. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),   src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_qg.weight", l),     src.qg));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),      src.k));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),      src.v));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l), src.o));

    w_norm   = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),   RAD_F32,  {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qg     = decl_w(b, nm.f("blk.%d.attn_qg.weight", l),     RAD_BF16, {qg_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_k      = decl_w(b, nm.f("blk.%d.attn_k.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_v      = decl_w(b, nm.f("blk.%d.attn_v.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* One gain vector shared by every head, so it is replicated across ranks even though the heads
     * are not. */
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_o      = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    /* ---- ops, in the order they run, with the one exception noted at attn_paged. Each states the
     * buffers it reads and writes, because that is what the buffer planner computes liveness
     * from -- and a gated attention layer has six activation buffers where a plain one has three,
     * so an undeclared use here costs an arena, not a rounding error. */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    op_qg = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", qg_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_qg)),
               {w.h}, {w.qg});
    op_kp = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_k)),
               {w.h}, {w.k});
    op_vp = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_v)),
               {w.h}, {w.v});

    /* Fusion is a selection, not a compiler pass (spec §2.3): ask for the fused op, and emit the
     * sequence it stands for when nothing in the hierarchy serves it.
     *
     * THE PROBE IS NOT MADE, because this block cannot fill the fused op's schema. libr4d's
     * r4d_qk_norm_rope_gate (libr4d/r4d_rows.cpp, pQkNormRopeGate) names the rotary width `rot`,
     * takes `q_dtype`, and -- the part that matters -- takes a PRECOMPUTED COS/SIN TABLE as a
     * required input operand instead of `theta`. Nothing in arch/ or core/ builds such a table.
     * Declaring the op with parameters the schema does not have is a STRUCTURAL error and fails
     * declare outright; it is not a miss that falls through to the sequence below, because an
     * unknown parameter is a caller bug and the builder is right to say so.
     *
     * What making the probe needs, precisely: a PERSIST buffer of [max_ctx, rot] filled once from
     * theta, and a way for a plugin to fill a declared buffer at load -- which the ABI has no hook
     * for, since buffers are arena storage and weights come from the container. Both are real work
     * and neither belongs in a probe. Until then this model runs the unfused sequence, which is
     * correct everywhere and is what makes it run on an install with no libr4d at all.
     *
     * `mode` would be "mrope" because that is what the model does: mrope_section [11,11,10] over
     * an interleaved layout. TWO THINGS ABOUT THAT ARE NOT EXPRESSIBLE and both stay true of the
     * unfused path. The sections have no parameter in any op schema, and RadBatch carries ONE
     * position component per token, so all three sections would receive the same position. For
     * text that is not an approximation: every rotary dimension has its own frequency and the
     * section only decides which position component feeds it, so mrope over equal components is
     * bit-identical to neox. It stops being identical the moment an image span gives the
     * components different values, which is why the vision tower needs both a sections parameter
     * and a 3-row position array before it can be correct. */
    op_fused = RAD_NULL_HANDLE;
    {

        /* Per-head RMS norm is an ordinary rmsnorm with M = tokens x heads and n = head_dim. The q
         * one reads the STRIDED view of the interleaved projection and writes the contiguous q
         * buffer, so it de-interleaves and normalises in one pass: the copy llama.cpp spends a
         * ggml_cont on is this op's output operand. The k one runs in place, which is safe for any
         * implementation that computes a row's scale before writing the row -- and none can do
         * otherwise. */
        op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                      RAD_INT("n", g.head_dim),
                                      RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                           RAD_WEIGHTS(w_q_norm)),
                       {w.qg}, {w.q});
        op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                      RAD_INT("n", g.head_dim),
                                      RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                           RAD_WEIGHTS(w_k_norm)),
                       {w.k}, {w.k});
        /* Two rope calls rather than one, because q and k are two buffers here. `rope` describes a
         * fused [q|k|v] row through n_head and n_head_kv; a call that names only q heads is that
         * row with an empty k and v, which is what a separate q buffer is. */
        op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                       {w.q}, {w.q});
        op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                       {w.k}, {w.k});
    }

    /* attn_paged is declared BEFORE kv_store even though it runs after it. block_size is
     * RAD_DERIVED -- the KV block size is a property of the resolved attention kernel and not a
     * number anyone gets to choose (spec §7.2) -- so it is OMITTED here, the selector skips
     * constraints on it while matching, writes the winner's value into the resolved geometry, and
     * kv_store reads it back with rad_kv_block_size(). The two ops touch disjoint buffers, so the
     * declared order differing from the issue order costs the buffer planner nothing.
     *
     * The ranged parameter is q_len, not M: one query row at decode, up to n_spec+1 at a
     * speculative verify, thousands in a prefill chunk, and libr4d has a different kernel for each
     * band. `scale` is left off: 1/sqrt(head_dim) is the default and is what this model uses. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                /* n_head, max_seqs and max_ctx are what the resolved kernel's SCRATCH HOOK
                                 * needs at declare: libr4d sizes a split-KV partial buffer of
                                 * num_seqs * q_len * q_heads * splits rows, and gqa gives the head
                                 * ratio rather than the count. All three are bounds, which is the
                                 * right kind of number for an arena sized once for the worst step. */
                                RAD_INT("n_head", g.n_head), RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn});

    /* kv_store writes the group's pool, which is not a declared buffer -- so its write set is
     * empty and its read set is the two projections it stores. */
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The gate. Two ops rather than one fused `mul(attn, sigmoid(gate))`, because the vocabulary
     * has no gated multiply for a bf16 output -- `gated_had_quant_i8` is the same arithmetic with
     * an int8 epilogue and is what the w8a8 sibling will use instead of this pair.
     *
     * The sigmoid's INPUT differs between the two paths and its output does not: the fused kernel
     * has already written a contiguous gate, and the fallback reads the strided view of the
     * interleaved projection. Same op, same geometry, one operand spelled two ways -- which is why
     * there is no second declaration and no branch after this point. */
    op_sigmoid = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg, w.gate}, {w.gate});

    op_gate = rw(b, RAD_OP(b, "mul",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.attn, w.gate}, {w.attn});

    op_o = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                             RAD_INT("K", q_dim), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_o)),
              {w.attn}, {w.h});

    /* The output projection is column-sharded, so each rank holds a partial sum of the residual
     * update and the ranks must agree before the add. The core inserts nothing: this collective is
     * here because the plugin put it here, and --debug-graph shows it exactly here (spec §9).
     * `exact` is stated rather than defaulted -- a caller that does not say it will accept a
     * quantised wire gets the exact kernel or no kernel. `hops` is not stated: one-shot against
     * two-shot is the kernel's business. */
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});

    return RAD_OK;
}

inline void AttnGatedBF16::step(RadCtx* c, const RadBatch* batch) const {
    /* rad_issue records its own failure and the core aborts the step and fails the requests in it,
     * naming the kernel and the geometry (spec §17). There is nothing a plugin can usefully do
     * with the status, and ignoring it is what keeps the run phase a flat readable list. */
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    RAD_ISSUE(c, op_qg,   brows(w.h, T), RAD_W(w_qg),   brows(w.qg, T));
    RAD_ISSUE(c, op_kp,   brows(w.h, T), RAD_W(w_k),    brows(w.k, T));
    RAD_ISSUE(c, op_vp,   brows(w.h, T), RAD_W(w_v),    brows(w.v, T));

    if (op_fused) {
        RAD_ISSUE(c, op_fused,
                  brows(w.qg, T), brows(w.k, T), praw(batch->positions, RAD_I32, T),
                  RAD_W(w_q_norm), RAD_W(w_k_norm),
                  brows(w.q, T), brows(w.gate, T));
    } else {
        /* The q view of the interleaved projection: head_dim columns out of every 2*head_dim, the
         * declared width kept as the stride. */
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    }

    RAD_ISSUE(c, op_kv_store,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    /* The actual query length, not the batch's token count: it is what selects the attention band,
     * and a verify step and a prefill chunk can carry the same token count at very different
     * q_len. */
    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                /* cu_seqlens. The batch may be ragged -- one prefill chunk beside other
                 * sequences one decode row each -- and without it the kernel has to
                 * assume a single q_len for the whole step, which it cannot. */
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    /* out = attn * sigmoid(gate). The fused kernel wrote the gate contiguously; the fallback left
     * it interleaved and this is where it is read out. */
    if (op_fused) RAD_ISSUE_N(c, op_sigmoid, T * g.n_head, brows(w.gate, T), brows(w.gate, T));
    else          RAD_ISSUE_N(c, op_sigmoid, T * g.n_head,
                              bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.gate, T));
    RAD_ISSUE(c, op_gate, brows(w.attn, T), brows(w.gate, T), brows(w.attn, T));

    RAD_ISSUE(c, op_o, brows(w.attn, T), RAD_W(w_o), brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_H */
