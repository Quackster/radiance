/* rad_block_attn_fp8.h -- the PLAIN GQA attention block with block-scaled FP8 projections.
 *
 *   rmsnorm -> quant_act_fp8 -> fused q|k|v gemm_nt_q -> rope -> kv_store -> attn_paged
 *   -> quant_act_fp8 -> o gemm_nt_q -> [all_reduce] -> residual add
 *
 * IT IS rad_block_attn_gated_fp8.h WITHOUT THE TWO THINGS THAT MAKE THAT ONE GATED, and those two
 * are what a family answers differently rather than what a model tunes:
 *
 *   - NO PER-HEAD Q/K NORM. Qwen3.5 norms each head against its own gain before rotating; LLaMA
 *     and MiniCPM5 rotate the projection as it comes out of the GEMM. A block that normed anyway
 *     against an absent gain would be a different model, and one that normed against a gain of
 *     ones would be a wasted launch a layer.
 *   - NO OUTPUT GATE. There is no second half of the query projection and no sigmoid: the
 *     attention output goes straight into the out-projection.
 *
 * WHAT THAT BUYS is the fused q|k|v, which the gated block cannot have: its query projection is
 * [q|gate] interleaved per head and k and v are separate tensors, so there are three GEMMs a layer.
 * Here the three weights concatenate along dim 0 into one [q_dim + 2*kv_dim, n_embd] and `rope`
 * takes exactly that row -- q heads then k heads, v left alone, which is the operand shape the op
 * was written for (docs/OPS.md). One GEMM, one rope, one quantise in front of all of it.
 *
 * THE ROW PARTS ARE NOT DECORATION. A tensor-parallel rank owns a slice of EACH of the three
 * projections, not a contiguous slice of the stack; without `row_parts` rank 0 gets the whole query
 * and rank 1 the whole key and value, and both ranks then compute a perfectly well-formed wrong
 * thing. LinearFP8 also checks that each part is a whole number of 128-row scale blocks, which is
 * what makes a block-scaled fusion expressible at all.
 *
 * Everything else -- the fused norm/quantise/residual/all-reduce op at the front, the KV group's
 * own dtype, the fold flags -- is rad_fp8.h's and rad_block_attn_gated_fp8.h's and is documented
 * there.
 */
#ifndef RAD_BLOCK_ATTN_FP8_H
#define RAD_BLOCK_ATTN_FP8_H

#include "rad_fp8.h"

#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

struct AttnFP8 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]                   the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]                   normed input, then the o output */
        rad_buf qkv  = 0;   /* [max_tok, q_dim + 2*kv_dim]         the fused projection */
        rad_buf q    = 0;   /* [max_tok, q_dim]                    the query, CONTIGUOUS */
        ActFP8  attn{};     /* [max_tok, q_dim]                    attention output */
    };

    /* The checkpoint's tensor names. Whether they hold block fp8 already or a dense tensor the
     * recipe quantises is the container's business; the declaration is the same either way. */
    struct Src {
        const char* norm  = nullptr;
        const char* q     = nullptr;
        const char* k     = nullptr;
        const char* v     = nullptr;
        const char* o     = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* The rotation pairs HALVES (NeoX), which is what HF's rotate_half does and what every LLaMA
     * derivative ships. GPT-J's adjacent pairs would be arithmetically silent too -- no per-op
     * check tells the two apart -- so this is stated rather than inferred. */
    const char* mode  = "neox";

    rad_weight w_norm = 0;
    LinearFP8  qkv{}, o{};
    NormQuantFP8 nq_h{};
    QuantFP8     q_attn{};
    rad_op     op_rope = 0, op_qlift = 0, op_attn = 0, op_kv_store = 0;
    rad_op     op_ar = 0, op_add = 0;
    bool       ar_out = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                            int64_t rotary_dim, const Wire& wire, const Src& src,
                            bool fold, bool add_out, int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* NO KV-HEAD REPLICATION HERE. The fused [q|k|v] row-shards by head with the plain even shard,
     * so above the KV head count a rank would get a slice of one KV head and declare fine while
     * reading the wrong rows. rad_block_attn_gated.h refuses for the same reason; the block that
     * plumbs geom_from's `kv_off` is rad_block_attn_gated_fp8.h. */
    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm, one quantise, one consumer -- and one op, see rad_fp8.h's NormQuantFP8. */
    RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                              fold ? w.x : 0, 0, ar_in));
    RAD_ARCH_TRY(qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), q_dim + 2 * kv_dim, g.n_embd,
                             RAD_SHARD_ROW, grp_layer(l), { src.q, src.k, src.v }, w.h, w.qkv,
                             0, { q_dim, kv_dim, kv_dim }));

    op_rope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", mode), RAD_INT("rotary_dim", n_rot)),
                     RAD_NOWEIGHTS),
                 {w.qkv}, {w.qkv});

    /* THE QUERY IS LIFTED OUT OF THE FUSED ROW, and this op is here because libr4d's paged
     * attention indexes `q` as one dense [total_q, n_head, head_dim] block and has no pitch
     * argument for it (r4d_attn_args.h, fill_paged). A column slice of the fused projection has a
     * row pitch of q_dim + 2*kv_dim, so passing it is refused at issue -- correctly, and loudly.
     *
     * The gated block beside this one never needs it: its q comes out of a q|gate de-interleave
     * that writes a buffer of its own. A plain LLaMA attention has no de-interleave, so the choice
     * is one `cast` a layer (this) or splitting the projection into q and kv (two more launches a
     * layer, one GEMM and one rope). The copy is the cheaper of the two at decode, which is the
     * step that matters, and it is one line to delete if the kernels ever take a q pitch.
     *
     * `from` and `to` are the same dtype: `cast` with no conversion is this tree's row copy, and
     * the drafter tap uses it the same way. */
    op_qlift = rw(b, RAD_OP(b, "cast",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                 RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.qkv}, {w.q});

    /* Declared before kv_store because `block_size` is RAD_DERIVED and kv_store reads the resolved
     * kernel's value back; see rad_block_attn_gated.h on why the declared order differs from the
     * issue order. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn.x});

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.qkv}, {});

    /* The out-projection's input quantiser. The gated block folds this into its gate pass; there
     * is no gate here, so it is the plain `quant_act_fp8` over the attention output. */
    RAD_ARCH_TRY(q_attn.declare(b, g, w.attn, q_dim));
    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});
    return RAD_OK;
}

inline void AttnFP8::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    nq_h.step(c, T);
    qkv.step(c, w.h, w.qkv, T);
    /* IN PLACE OVER THE FUSED ROW, which is what makes one launch do both halves: `rope` rotates
     * `n_head` query heads then `n_head_kv` key heads and leaves the value heads alone. */
    RAD_ISSUE(c, op_rope, brows(w.qkv, T), praw(batch->positions, RAD_I32, T));

    /* k and v are COLUMN SLICES of the same row, at q_dim and q_dim + kv_dim. */
    RAD_ISSUE(c, op_kv_store,
              bcol(w.qkv, q_dim, kv_dim, T), bcol(w.qkv, q_dim + kv_dim, kv_dim, T),
              praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE(c, op_qlift, bcol(w.qkv, 0, q_dim, T), brows(w.q, T));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn.x, T));

    q_attn.step(c, T);
    o.step(c, w.attn, w.h.x, T);
    if (op_ar && !(ar_out && ar_fused_ok(g, T)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_FP8_H */
