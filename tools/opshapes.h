/* opshapes.h -- how big is operand i of this op, given its resolved geometry.
 *
 * THE ONE PLACE THE TOOLS KNOW SOMETHING ABOUT AN OP, AND THEY SHOULD NOT HAVE TO.
 *
 * An op's schema names its operands and their roles but not their extents, and the extents are
 * not derivable from the parameters in general -- the run phase supplies them at issue. So a tool
 * that wants to CALL an op cold, which rad-kbench and rad-tune both do, has to size its buffers
 * somehow, and the honest options are a table here or a hook on RadKernelInfo.
 *
 * The hook is the better answer: a `RadShapeFn` beside RadLayoutFn, asking the kernel to describe
 * operand i for a geometry (see rad_operand_shapes_hook below). libref implements every op in the
 * conventional vocabulary, so once ref supplies the hook for all of them this file can go away.
 * Until then this covers the vocabulary in docs/OPS.md whose shapes follow from its
 * parameters, and every op it does not cover is SKIPPED AND COUNTED by its caller rather than
 * guessed at -- a tool that invents an extent tests the kernel's bounds handling and calls the
 * result correctness.
 *
 * Shared by rad-kbench and rad-tune so the two cannot disagree about what shape a case is; a
 * tuning measurement taken at a different extent from the correctness check is a measurement of
 * something else.
 */
#pragma once
#include "iface.h"

#include <cstring>
#include <string>
#include <vector>

namespace rad {

struct OpdShape {
    uint32_t             dtype = RAD_F32;
    std::vector<int64_t> shape;
    bool                 is_index = false;   /* token ids / row indices, not normals */
    /* AN INDEX OPERAND'S RANGE, when it is not the previous operand's leading extent. A KV slot
     * indexes n_blocks * block_size positions and a block-table entry indexes n_blocks, and neither
     * is any operand's dim 0 -- drawing them from the wrong range gives a case that tests the
     * kernel's out-of-range handling and calls the result correctness. 0 keeps the default. */
    int64_t              idx_max = 0;
    /* A CONSTANT, for an index operand whose value is a length rather than a choice. `seqused` is
     * the one: a random context length shorter than the query makes a causal attention read a
     * prefix that does not exist, which both implementations would then agree about while neither
     * computed what the model asks for. -1 keeps the draw. */
    int64_t              idx_const = -1;
    /* DISTINCT indices, for a SCATTER. `slot_mapping` is the case: two tokens drawn onto the same
     * slot make the store a race, and the two implementations resolve it differently -- ref runs
     * the token loop under OpenMP and the kernel runs a workgroup a token -- so the comparison
     * measures the scheduler rather than the kernel. A read-side index (a gather row, a block
     * table entry) may repeat and does. */
    bool                 idx_unique = false;
    /* CUMULATIVE SEQUENCE LENGTHS: [0, ..., T]. Every GDN op takes `cu` and opens by reading it,
     * so it is not optional and it is not a draw -- it is the batch's shape. */
    bool                 idx_cu = false;
    /* THE DOMAIN OF A NON-INDEX OPERAND, where a normal draw is outside it. `beta` is a sigmoid
     * output in (0, 1) and `g` is an intra-chunk cumulative sum of log-decays, non-increasing and
     * never positive -- and the gated-delta-net scan depends on that: every decay factor it forms
     * is e^{g_i - g_j} with j <= i, which is <= 1 only because g is monotone. Feeding it normals
     * makes the ORACLE overflow to 2e38 while the kernel does something else, and the comparison
     * then measures the test. RAD_GDN_CHUNK carries the chunk width the cumsum resets at. */
    enum Fill { FillNormal = 0, FillSigmoid, FillGateCumsum, FillTriLower, FillUnitRow };
    int                  fill = FillNormal;
    int64_t              fill_chunk = 0;
    /* An OPTIONAL operand this case deliberately does not pass. The recipe still has to emit one
     * entry per schema operand -- the count check below is what catches a recipe that has drifted
     * from the schema -- so "absent" needs a spelling, and a null RadTensor is exactly what the
     * ABI already means by it (rad_abi.h). The caller allocates nothing and compares nothing.
     *
     * It earns its keep on `quant_act_i8`, whose third output selects libr4d's `_asum` form:
     * libref implements no asymmetric grid, so it does not produce those sums, and passing
     * the buffer anyway would compare a zeroed reference against a real one and report a failure
     * that is a gap in the ORACLE rather than a fault in the kernel. */
    bool                 absent = false;
};

/* What the OPS DO NOT SAY and a recipe still needs. The gated-delta-net family reads its head
 * counts off the operands, so no op names them and no geometry carries them -- and a check run at
 * an invented head ratio is a check of a model nobody is running. The caller reads them from the
 * container's metadata and passes them here; absent, the gdn_* recipes decline and their cases are
 * skipped and counted, which is this file's rule everywhere else. */
struct ShapeCtx {
    int64_t n_head_k = 0;   /* linear_num_key_heads */
    int64_t n_head_v = 0;   /* linear_num_value_heads */
    int64_t n_seq    = 1;   /* sequences in the checked batch */
};

/* ================================================================== the hook, which wins
 *
 * ASK THE KERNEL FIRST. `RadKernelInfo::opd_shape` is the hook this file's header describes: the
 * component that knows an operand's extent is the one that reads it, and a table in `tools/` can
 * only ever describe the ops whoever wrote the table had heard of. Everything below this function
 * is the fallback for a plugin that does not implement the hook.
 *
 * WHICH kernel to ask is a real choice and it is the caller's:
 *
 *   - A CORRECTNESS check wants the ORACLE's description. The two implementations have to be
 *     handed the same bytes for the comparison to mean anything, and where they disagree about an
 *     extent the reference semantics are the ones that define the op.
 *   - A BENCHMARK wants the KERNEL UNDER TEST's own. A device-only fusion has no oracle at all --
 *     which is why a large part of an accelerated library is reachable only this way -- and timing
 *     one still needs buffers.
 *
 * So this takes the RadKernelInfo to ask rather than deciding, and returns RAD_E_UNSUPPORTED if
 * that kernel has no hook or declines the geometry. A decline is a real answer and the caller
 * SKIPS AND COUNTS it; only a caller that invents an extent is wrong, because it then tests the
 * kernel's bounds handling and calls the result correctness. */
inline int rad_operand_shapes_hook(const RadKernelInfo* k, const Geometry& g, int n_opd,
                                   std::vector<OpdShape>* out) {
    if (!k || !k->opd_shape || n_opd <= 0) return RAD_E_UNSUPPORTED;
    out->assign((size_t)n_opd, OpdShape{});
    for (int i = 0; i < n_opd; ++i) {
        RadOpdDesc d{};
        d.idx_const = -1;   /* the ABI's "draw", so a hook that leaves it alone means the default */
        const int st = k->opd_shape(g.params(), g.n_params(), i, &d);
        if (st < 0) return st;
        /* A rank the ABI cannot express is a hook bug, not a geometry it declined, and it must not
         * become a zero-extent buffer that every comparison then passes vacuously. */
        if (d.rank == 0 || d.rank > RAD_MAX_RANK) {
            if (!(d.flags & RAD_OPD_F_ABSENT)) return RAD_E_SHAPE;
        }
        OpdShape& o = (*out)[(size_t)i];
        o.dtype = d.dtype;
        o.shape.assign(d.shape, d.shape + (d.rank <= RAD_MAX_RANK ? d.rank : 0));
        o.absent     = (d.flags & RAD_OPD_F_ABSENT)     != 0;
        o.idx_unique = (d.flags & RAD_OPD_F_IDX_UNIQUE) != 0;
        o.idx_cu     = (d.flags & RAD_OPD_F_IDX_CU)     != 0;
        o.is_index   = d.fill == RAD_FILL_INDEX;
        o.idx_max    = d.idx_max;
        o.idx_const  = d.idx_const;
        o.fill_chunk = d.fill_chunk;
        switch (d.fill) {
            case RAD_FILL_SIGMOID:     o.fill = OpdShape::FillSigmoid;    break;
            case RAD_FILL_GATE_CUMSUM: o.fill = OpdShape::FillGateCumsum; break;
            case RAD_FILL_TRI_LOWER:   o.fill = OpdShape::FillTriLower;   break;
            case RAD_FILL_UNIT_ROW:    o.fill = OpdShape::FillUnitRow;    break;
            default:                   o.fill = OpdShape::FillNormal;     break;
        }
        for (int64_t e : o.shape) if (e <= 0 && !o.absent) return RAD_E_SHAPE;
    }
    return RAD_OK;
}

/* Fills one entry per schema operand, in order. RAD_E_UNSUPPORTED when this op has no recipe or
 * the geometry does not carry the parameters the recipe needs. `M` is the value of the ranged
 * parameter for the band being sized. */
inline int rad_operand_shapes(const std::string& op, const Geometry& g, int n_opd, int64_t M,
                              std::vector<OpdShape>* out, const ShapeCtx* ctx = nullptr) {
    auto gi = [&](const char* k, int64_t dflt) {
        long long v = 0;
        return g.get_i(k, &v) ? (int64_t)v : dflt;
    };
    /* THE ACTIVATION DTYPE, and the fallback matters. Several ops carry no `dtype` parameter at
     * all -- `rope` and `qk_norm_rope` name a `mode` and a `theta` and nothing about the width,
     * because the tensor they rotate in place is whatever the caller declared. Falling back to f32
     * there manufactures an f32 activation, which a shim that reads a bf16 rotation refuses on
     * dtype -- a check reporting FAIL on a kernel the engine runs correctly. `q_dtype` and
     * `kv_dtype` are attention's spellings of the same thing, and bf16 is what every activation in
     * this project is when nothing says otherwise. */
    const char* dts = g.get_s("dtype");
    if (!dts) dts = g.get_s("q_dtype");
    if (!dts) dts = g.get_s("kv_dtype");
    const uint32_t dt = dts ? rad_dtype_parse(dts) : (uint32_t)RAD_BF16;
    const uint32_t D  = dt == RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;

    const int64_t n     = gi("n", 0);
    const int64_t N     = gi("N", 0);
    const int64_t K     = gi("K", 0);
    const int64_t group = gi("group", 128);

    out->clear();
    auto add = [&](uint32_t d, std::vector<int64_t> s, bool idx = false) {
        OpdShape o;
        o.dtype = d; o.shape = std::move(s); o.is_index = idx;
        out->push_back(std::move(o));
    };
    auto add_fill = [&](uint32_t d, std::vector<int64_t> s, int f, int64_t chunk = 0) {
        OpdShape o;
        o.dtype = d; o.shape = std::move(s); o.fill = f; o.fill_chunk = chunk;
        out->push_back(std::move(o));
    };
    auto add_idx = [&](std::vector<int64_t> s, int64_t max, int64_t konst = -1,
                       bool uniq = false) {
        OpdShape o;
        o.dtype = (uint32_t)RAD_I32; o.shape = std::move(s); o.is_index = true;
        o.idx_max = max; o.idx_const = konst; o.idx_unique = uniq;
        out->push_back(std::move(o));
    };
    /* One optional operand this case does not pass. See OpdShape::absent. */
    auto skip_opd = [&]() { OpdShape o; o.absent = true; out->push_back(std::move(o)); };

    const int64_t head_dim  = gi("head_dim", 0);
    const int64_t n_head    = gi("n_head", 0);
    const int64_t n_head_kv = gi("n_head_kv", n_head);

    if (op == "rmsnorm")           { add(D, {M, n}); add(D, {n}); add(D, {M, n}); }
    else if (op == "l2_norm" || op == "softplus") { add(D, {M, n}); add(D, {M, n}); }
    else if (op == "rope" || op == "rope_multi") {
        /* One inout operand plus positions. qkv is the fused row: n_head queries and n_head_kv
         * each of k and v, all head_dim wide. */
        if (head_dim <= 0 || n_head <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, (n_head + 2 * n_head_kv) * head_dim});
        add((uint32_t)RAD_I32, {M}, true);
    }
    else if (op == "rmsnorm_add")  { add(D, {M, n}); add(D, {M, n}); add(D, {n});
                                     add(D, {M, n}); add(D, {M, n}); }
    else if (op == "layernorm")    { add(D, {M, n}); add(D, {n}); add(D, {n}); add(D, {M, n}); }
    else if (op == "add" || op == "mul") { add(D, {M, n}); add(D, {M, n}); add(D, {M, n}); }
    else if (op == "silu_mul")     { add(D, {M, 2 * n}); add(D, {M, n}); }
    else if (op == "gelu" || op == "silu" || op == "sigmoid") { add(D, {M, n}); add(D, {M, n}); }
    else if (op == "softmax")      { add(D, {M, n}); add(D, {M, n}); }
    else if (op == "gather_rows") {
        /* x is a buffer of many rows and idx picks M of them. Sized at 4M rows so the gather is
         * not the identity -- the identity is the one case that passes while broken. */
        add(D, {4 * M, n}); add((uint32_t)RAD_I32, {M}, true); add(D, {M, n});
    }
    else if (op == "cast") {
        const uint32_t from = rad_dtype_parse(g.get_s("from") ? g.get_s("from") : "f32");
        const uint32_t to   = rad_dtype_parse(g.get_s("to")   ? g.get_s("to")   : "f32");
        add(from ? from : (uint32_t)RAD_F32, {M, n});
        add(to   ? to   : (uint32_t)RAD_F32, {M, n});
    }
    else if (op == "gemm_nt")      { add(D, {M, K}); add(D, {N, K}); add(D, {M, N}); }
    else if (op == "gemm_nt_bias") { add(D, {M, K}); add(D, {N, K}); add(D, {N}); add(D, {M, N}); }
    else if (op == "embed_lookup") {
        add((uint32_t)RAD_I32, {M}, true);
        add(D, {gi("n_vocab", 0), gi("n_embd", 0)});
        add(D, {M, gi("n_embd", 0)});
    }
    else if (op == "logits_gemm") {
        add(D, {M, gi("n_embd", 0)});
        add(D, {gi("n_vocab", 0), gi("n_embd", 0)});
        add((uint32_t)RAD_F32, {M, gi("n_vocab", 0)});
    }
    else if (op == "quant_act_i8" || op == "had_quant_act_i8") {
        add(D, {M, n}); add((uint32_t)RAD_I8, {M, n});
        add((uint32_t)RAD_F32, {M, group ? n / group : 1});
        /* quant_act_i8 alone carries the optional `asum` output, and it is not passed here --
         * see OpdShape::absent. had_quant_act_i8's schema stops at the scale. */
        if (op == "quant_act_i8") skip_opd();
    }
    else if (op == "quant_act_fp8") {
        /* The same three operands as quant_act_i8 at a different output width. `dtype` names the
         * INPUT, which is bf16; the output width is what the op name says. */
        if (n <= 0 || group <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, n});
        add((uint32_t)RAD_F8E4M3, {M, n});
        add((uint32_t)RAD_F32, {M, (n + group - 1) / group});
    }
    else if (op == "gemm_nt_q") {
        /* BLOCK-SCALED FP8 ONLY, and the restriction is the honest one rather than laziness.
         *
         * `gemm_nt_q` is one op over six weight formats and the SCALE GRID is what differs between
         * them -- per row, per row-group, per output-row-block, an e8m0 exponent byte, with or
         * without a zero point or a reference exponent. Every one of those is a property of the
         * kernel that reads it, not of the schema, so a recipe that guessed would be testing the
         * bounds handling of five kernels and calling the result correctness (see the header).
         *
         * The fp8 family is different: its grid is stated by the CHECKPOINT -- E4M3 [N][K] against
         * a bf16 [ceil(N/128)][ceil(K/128)] plane, with `group` carrying the 128 -- so there is
         * nothing left to guess, and it is the family the production model runs on. The rest are
         * skipped and counted.
         *
         * `a_sum` and `b_ref` belong to the asymmetric-integer and mxfp4 grids and are absent. */
        const char* q = g.get_s("dtype");
        /* OCP-MXFP4 is the other grid a checkpoint states: e2m1 codes with one E8M0 exponent per
         * 32 of K (`group` 32), read against the fp8 family's own activation pair -- E4M3 with an
         * f32 scale per (row, 128 of K), which is what qwen35_mxfp4 feeds it. The weight operands
         * arrive in libr4d's stored layout, and their BYTE COUNTS are what is sized here: the
         * codes are K/2 bytes a row in fragment order, the exponents [K/32][N] and the per-row
         * reference [N]. Drawn bytes are fine for timing -- the kernel's work does not depend on
         * which exponents it folds. */
        if (q && std::strcmp(q, "mxfp4a8") == 0) {
            if (N <= 0 || K <= 0 || group != 32) return RAD_E_UNSUPPORTED;
            add((uint32_t)RAD_F8E4M3, {M, K});
            add((uint32_t)RAD_F32,    {M, (K + 127) / 128});
            add((uint32_t)RAD_U8,     {N, K / 2});     /* b       -- e2m1 codes, fragment order */
            add((uint32_t)RAD_U8,     {K / 32, N});    /* b_scale -- E8M0, [K/32][N] */
            add((uint32_t)RAD_BF16,   {M, N});         /* y */
            skip_opd();                                /* a_sum */
            add((uint32_t)RAD_U8,     {N});            /* b_ref   -- the per-row reference */
        } else {
        const bool a8  = q && std::strcmp(q, "fp8a8")  == 0;
        const bool a16 = q && std::strcmp(q, "fp8a16") == 0;
        if ((!a8 && !a16) || N <= 0 || K <= 0 || group <= 0) return RAD_E_UNSUPPORTED;
        const int64_t kb = (K + group - 1) / group;
        const int64_t nb = (N + group - 1) / group;
        /* a, a_scale */
        if (a8) { add((uint32_t)RAD_F8E4M3, {M, K}); add((uint32_t)RAD_F32, {M, kb}); }
        else    { add((uint32_t)RAD_BF16,   {M, K}); skip_opd(); }
        add((uint32_t)RAD_F8E4M3, {N, K});    /* b       -- the checkpoint's own bytes */
        add((uint32_t)RAD_BF16,   {nb, kb});  /* b_scale -- weight_scale_inv, one per 128x128 tile */
        add((uint32_t)RAD_BF16,   {M, N});    /* y */
        skip_opd();                           /* a_sum */
        skip_opd();                           /* b_ref */
        }
    }
    else if (op == "dequant") {
        add((uint32_t)RAD_I8, {M, n}); add((uint32_t)RAD_F32, {M, group ? n / group : 1});
        add(D, {M, n});
    }
    else if (op == "all_reduce" || op == "all_gather") {
        /* Both take two operands. all_gather's second is [world_size * numel]; all_reduce's is the
         * optional SECOND DESTINATION -- absent means in place, the form docs/OPS.md draws and
         * every block issues. It is passed so the out-of-place form is exercised too. */
        const int64_t numel = gi("numel", n ? n : M);
        add(D, {numel});
        add(D, {op == "all_gather" ? numel * gi("world_size", 1) : numel});
    }
    /* ---- the paged pair. They share one layout and it is stated once for the whole project in
     * libref/ref_ops.h:
     *
     *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]   K first, V second in a slot
     *
     * Without a recipe here the attention kernel and the KV scatter -- the pair the whole cache
     * design rests on -- cannot be compared against the oracle on real bytes at all. The extents
     * they need beyond the parameters are a block COUNT and a context length, and both are this
     * tool's to choose rather than the model's: any cache big enough for the case will do, so long
     * as the indices land inside it. */
    else if (op == "kv_store") {
        const int64_t bs = gi("block_size", 0);
        const int64_t kvh = gi("n_head_kv", 0);
        if (bs <= 0 || kvh <= 0 || head_dim <= 0) return RAD_E_UNSUPPORTED;
        /* Two blocks of slack so the slot mapping is not a dense prefix: a scatter that only ever
         * writes block 0 agrees with anything about the block index. */
        const int64_t nb = (M + bs - 1) / bs + 2;
        /* Only the cache takes `kv_dtype`: k and v are bf16 whatever the cache stores, and D,
         * which falls back on kv_dtype for an op with no `dtype`, would make them fp8 too. */
        const char*    kvs = g.get_s("kv_dtype");
        const uint32_t kvd = kvs ? rad_dtype_parse(kvs) : (uint32_t)RAD_DT_INVALID;
        add((uint32_t)RAD_BF16, {M, kvh, head_dim});
        add((uint32_t)RAD_BF16, {M, kvh, head_dim});
        add_idx({M}, nb * bs, -1, /*uniq*/ true);
        add(kvd == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : kvd, {nb, kvh, bs, 2 * head_dim});
    }
    else if (op == "attn_paged") {
        /* `M` is q_len here -- the ranged parameter is the query length and not a token count,
         * because one query row at decode and thousands in a prefill chunk are different kernels
         * (docs/OPS.md). One sequence, whose context is exactly this query: a causal attention
         * over its own prefix, which is the case a prefill chunk actually runs. */
        const int64_t bs  = gi("block_size", 16);
        const int64_t gqa = gi("gqa", 1);
        const int64_t nh  = gi("n_head", gqa);
        if (bs <= 0 || gqa <= 0 || nh <= 0 || head_dim <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
        const int64_t kvh = nh / gqa;
        const int64_t n_seq = 1, ctx = M;
        const int64_t max_blocks = (ctx + bs - 1) / bs;
        add(D, {n_seq * M, nh, head_dim});
        add(D, {max_blocks, kvh, bs, 2 * head_dim});
        add_idx({n_seq, max_blocks}, max_blocks);
        /* NOT a draw: a context shorter than the query makes causal attention read a prefix that
         * does not exist, and both implementations would agree about the garbage. */
        add_idx({n_seq}, 0, ctx);
        skip_opd();                           /* k_descale -- this cache is not quantised */
        skip_opd();                           /* v_descale */
        add(D, {n_seq * M, nh, head_dim});
    }
    /* ---- gated delta net. The layouts are libref/ref_gdn.cpp's, stated there once:
     *
     *     q, k     [T, Hk, head_k]         v, o  [T, Hv, head_v]
     *     g, beta  [T, Hv]  fp32           A     [T, Hv, chunk]
     *     h0, ht   [N, Hv, head_v, head_k] fp32  cu    [N+1]
     *     conv w   [conv_dim, conv_width]        conv_state [n_slots, conv_dim, state_len]
     *
     * with conv_dim = 2 * Hk * head_k + Hv * head_v.
     *
     * THE HEAD COUNTS ARE NOT PARAMETERS of any of these ops -- the kernels read them off the
     * operands -- so they come from the container's metadata through `ctx`. Without them a recipe
     * would have to invent a head count, and a GDN check at the wrong head ratio is a check of a
     * model nobody is running. `ctx` absent means the caller could not supply them, and the op is
     * skipped rather than guessed at, which is this file's rule everywhere else. */
    else if (op.rfind("gdn_", 0) == 0 && op != "gdn_gated_rmsnorm") {
        if (!ctx || ctx->n_head_k <= 0 || ctx->n_head_v <= 0) return RAD_E_UNSUPPORTED;
        const int64_t Hk = ctx->n_head_k, Hv = ctx->n_head_v;
        const int64_t hk = gi("head_k", 0), hv = gi("head_v", hk);
        const int64_t chunk = gi("chunk", 64);
        const int64_t width = gi("conv_width", 4);
        if (hk <= 0 || hv <= 0 || chunk <= 0 || width <= 1) return RAD_E_UNSUPPORTED;
        const int64_t conv_dim = 2 * Hk * hk + Hv * hv;
        const int64_t S = ctx->n_seq > 0 ? ctx->n_seq : 1;

        if (op == "gdn_conv_prep") {
            /* T tokens over S sequences, one conv-state slot each. `state_len` is the manager's
             * conv_width - 1 + n_spec; with no speculation that is conv_width - 1, and the kernel
             * reads its window at the slot the last committed token left. */
            add(D, {M, conv_dim});                     /* x -- the fused projection row */
            add(D, {conv_dim, width});                 /* w */
            skip_opd();                                /* b -- this checkpoint has no conv bias */
            add((uint32_t)RAD_F32, {Hv});              /* A_log */
            add((uint32_t)RAD_F32, {Hv});              /* dt_bias */
            add(D, {S, conv_dim, width - 1});          /* conv_state, INOUT */
            add_idx({S + 1}, 0, -1);                   /* cu -- filled cumulative below */
            out->back().idx_cu = true;
            add(D, {M, Hv});                           /* a */
            add(D, {M, Hv});                           /* b_gate */
            add_idx({S}, S, -1, /*uniq*/ true);        /* cache_idx -- one slot a sequence */
            add_idx({S}, 0, 0);                        /* has_init -- a cold prefill, SAID rather
                                                        * than left absent: ref reads a null as WARM
                                                        * and libr4d as COLD, so the default is the
                                                        * one thing the two do not share. */
            add(D, {M, Hk, hk});                       /* q */
            add(D, {M, Hk, hk});                       /* k */
            add(D, {M, Hv, hv});                       /* v */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
        }
        else if (op == "gdn_kkt_solve") {
            add(D, {M, Hk, hk});                       /* k */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_idx({S + 1}, 0, -1);                   /* cu */
            out->back().idx_cu = true;
            add_fill(D, {M, Hv, chunk}, OpdShape::FillTriLower, chunk);              /* A */
        }
        else if (op == "gdn_chunk_scan") {
            add(D, {M, Hk, hk});                       /* q */
            add(D, {M, Hk, hk});                       /* k */
            add(D, {M, Hv, hv});                       /* v */
            add_fill(D, {M, Hv, chunk}, OpdShape::FillTriLower, chunk);              /* A */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
            add((uint32_t)RAD_F32, {S, Hv, hv, hk});   /* h0 */
            add_idx({S + 1}, 0, -1);                   /* cu */
            out->back().idx_cu = true;
            add(D, {M, Hv, hv});                       /* o */
            add((uint32_t)RAD_F32, {S, Hv, hv, hk});   /* ht */
            add_idx({S}, S, -1, /*uniq*/ true);        /* state_idx */
        }
        else if (op == "gdn_conv_update") {
            /* The decode pair. `M` is q_len here, one row per candidate token, and the window is
             * conv_width - 2 + q_len deep so a rejection is a change of read offset rather than a
             * recompute (ref_ops.h's rollback contract). num_accepted is 1-based and all ones is
             * the non-speculative decode. */
            const int64_t sl = width - 2 + M;
            add(D, {S * M, conv_dim});                 /* x */
            add(D, {conv_dim, width});                 /* w */
            skip_opd();                                /* b */
            add(D, {S, conv_dim, sl});                 /* conv_state, INOUT */
            add_idx({S, M}, S, -1);                    /* state_idx */
            add_idx({S}, 0, 1);                        /* num_accepted -- none rejected */
            add_idx({S + 1}, 0, -1);                   /* cu */
            out->back().idx_cu = true;
            add(D, {S * M, Hk, hk});                   /* q */
            add(D, {S * M, Hk, hk});                   /* k */
            add(D, {S * M, Hv, hv});                   /* v */
        }
        else if (op == "gdn_recurrent_update") {
            add(D, {S * M, Hk, hk});                   /* q */
            add(D, {S * M, Hk, hk});                   /* k */
            add(D, {S * M, Hv, hv});                   /* v */
            add(D, {S * M, Hv});                       /* a */
            add(D, {S * M, Hv});                       /* b */
            add((uint32_t)RAD_F32, {Hv});              /* A_log */
            add((uint32_t)RAD_F32, {Hv});              /* dt_bias */
            add((uint32_t)RAD_F32, {S * M, Hv, hv, hk});  /* state, one per candidate token */
            add_idx({S, M}, S * M, -1, /*uniq*/ true); /* state_idx */
            add_idx({S}, 0, 1);                        /* num_accepted */
            add_idx({S + 1}, 0, -1);                   /* cu */
            out->back().idx_cu = true;
            add(D, {S * M, Hv, hv});                   /* z */
            add((uint32_t)RAD_F32, {hv});              /* norm_w */
            add(D, {S * M, Hv, hv});                   /* o */
        }
        else return RAD_E_UNSUPPORTED;
    }
    else if (op == "gdn_gated_rmsnorm") {
        /* One row per (token, head): `M` is already tokens x heads and `channels` is head_v. */
        const int64_t ch = gi("channels", 0);
        if (ch <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, ch}); add(D, {M, ch}); add((uint32_t)RAD_F32, {ch}); add(D, {M, ch});
    }
    /* Deliberately absent, and this is the argument for RadShapeFn rather than an oversight:
     * ssm_conv and qk_norm_rope_gate take operands whose extents come from the KV group's state
     * geometry and from INTERLEAVED strided views -- attn_q is [n_embd, 2*head_dim*n_head] with
     * query and gate alternating per head, so it is not a contiguous split and a recipe here would
     * have to know that. The kernel already does. Without the kernel's hook these are skipped and
     * counted by the caller, which is the honest outcome: a tool that invents an extent tests the
     * kernel's bounds handling and calls it correctness. */
    else return RAD_E_UNSUPPORTED;

    if ((int)out->size() != n_opd) {
        /* The recipe and the schema disagree. That is a bug in one of them, and padding with a
         * guess would hide it behind a number that looks like a result. */
        RAD_DEBUG("%s: shape recipe produces %zu operands, the schema declares %d",
                  op.c_str(), out->size(), n_opd);
        return RAD_E_UNSUPPORTED;
    }
    for (const OpdShape& s : *out)
        for (int64_t d : s.shape)
            if (d <= 0) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}

inline int64_t rad_numel(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
}

/* Everything is materialised in f32 for comparison and for filling. A kernel writing bf16 is
 * compared after widening: two implementations disagreeing in the last bf16 bit is what a
 * tolerance is for, and comparing raw uint16 would make every such case a failure. */
void rad_widen (const void* src, uint32_t dtype, int64_t n, std::vector<float>* out);
void rad_narrow(const float* src, uint32_t dtype, int64_t n, void* dst);

/* splitmix64, keyed per case so one case's inputs do not depend on how many ran before it. */
inline uint64_t rad_splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint64_t rad_case_seed(uint64_t base, std::string_view op, int band, int domain, int operand);
void     rad_fill_normal(float* p, int64_t n, uint64_t seed, float sigma);
/* THE BLOCK WIDTH IS PART OF THE CONTRACT, not an implementation detail: a caller that streams a
 * large operand block by block must use the same width rad_fill_normal splits at, or the two
 * produce different values for the same seed. */
extern const int64_t RAD_FILL_BLOCK;
void     rad_fill_normal_block(float* p, int64_t n, uint64_t seed, int64_t block, float sigma);

}  /* namespace rad */
