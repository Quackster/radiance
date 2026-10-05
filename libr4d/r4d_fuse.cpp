/* r4d_fuse.cpp -- what chain of conventional ops does this kernel compute in one launch.
 *
 * `RadKernelInfo::replaces` (rad_abi.h) turns a claim that sixteen rows of this plugin make in
 * PROSE -- "byte-identical to the two ops it replaces" -- into something a tool can execute. The
 * value is an ORACLE: eleven of libr4d's fusions have no libref kernel at all, so rad-kbench
 * otherwise skips them BY NAME. A fused row that names its chain can be run against that chain
 * built out of THIS PLUGIN'S OWN unfused rows, and the answer should not be within a tolerance --
 * it should be the same bytes.
 *
 * THE AUTHORITY FOR EVERY CHAIN BELOW IS THE DEVICE KERNEL, not the row's prose and not this file.
 * A fusion is exact or it is not, and the difference is one `f2bf` in a kernel body: the fp8 folds
 * round their intermediate to bf16 and read it back precisely so that the fused and unfused forms
 * agree, and the int8 folds deliberately do NOT, because their reference is a different pair. So
 * each chain below names the libr4d source and the line that decides it, and a reader can check
 * one against the other. That is not decoration. `qk_norm_rope_gate` is the worked counter-example
 * -- it folds a sum of squares over one wave per head where `rmsnorm` folds it over
 * r4d_act_threads(M, n) threads, so the two differ by a ULP and it publishes NO HOOK -- and the ABI
 * says why: "a kernel that is merely close to the sequence it names should publish no hook at all",
 * because a false failure is worse than a skip and drowns the true ones.
 *
 * ============================================================ WHICH OPERANDS THE CALL PASSES
 *
 * `rmsnorm_quant_fp8` computes `add` then `rmsnorm` then the quantiser when it is given a residual,
 * and `rmsnorm` then the quantiser when it is not: THE CHAIN LENGTH DEPENDS ON AN OPERAND. That is
 * the reason the chain is a hook rather than a table, and the reason the hook has to be told which
 * operands the call carries rather than only `(p, n_p)`.
 *
 * Two other ways of deciding it look plausible and both are wrong:
 *
 *   DERIVE IT FROM A PARAMETER. There is no such parameter, and `wadd` in particular is not one.
 *       `wadd` is GemmaRMSNorm's `1 +` -- a property of the norm GAIN, not of the residual -- and
 *       arch/common/rad_fp8.h:369 passes it unconditionally in both forms. Keying on it would
 *       answer "three steps" for a Gemma-style model that passes no residual and "two steps" for a
 *       Llama-style one that does, which is a wrong chain rather than a missing one.
 *
 *   DECLARE BOTH WIRINGS WITH RAD_FUSE_NONE. That spelling marks an OPTIONAL operand of a STEP
 *       that this chain does not pass. It cannot express this difference, because the operands that
 *       vary are REQUIRED ones -- `add`'s `b` and `rmsnorm`'s `x` -- and because the CHAIN LENGTH
 *       itself changes, which no per-operand marker reaches.
 *
 * So the ABI carries it: `present` is a bit per operand of the fused op, and this file reads it
 * with RAD_FUSE_HAS(). A caller that genuinely does not know passes RAD_FUSE_PRESENT_ALL, which
 * means "answer for every operand your own description says exists" -- and that is a contract
 * rather than a coupling, because this plugin's `r4d_shape_rmsnorm_quant_fp8` publishes all six as
 * present and is the list a cold caller allocated from.
 *
 * Note how the question VANISHES wherever a fold's step carries the same optional operand the fold
 * does: `ar_rmsnorm_quant_fp8` below wires its residual and its out_bf16 straight through to a
 * `rmsnorm_quant_fp8` step where both are optional as well, so absence propagates and that chain
 * needs no knowledge of presence at all. That is worth preferring where a chain has the choice.
 *
 * ============================================================ WHAT IS DECLINED, AND WHY
 *
 * A decline is a real answer here, the same way it is in r4d_shapes.cpp: the caller counts the row
 * as uncovered, which is true, where a chain that is merely close produces a FAILURE that means
 * nothing. These rows carry a null `replaces` deliberately.
 *
 *   rmsnorm_had_quant_i8, gated_had_quant_i8, gdn_gated_norm_had_quant_i8
 *       NOT because the vocabulary has no `hadamard` op -- it does not, but that is not what stops
 *       these. The rotation is inside `had_quant_act_i8`, which IS an op, so the algebra
 *       `rmsnorm` + `had_quant_act_i8` is available and it still does not hold. The reason is the
 *       fusion's own trick: r4d_rmsnorm_had_quant_i8.hip runs the FWHT on `x * (w + wadd)` BEFORE
 *       the norm closes -- legitimate, since the transform is linear and `inv_rms` is a per-row
 *       scalar that the int8 codes cancel -- and folds `inv_rms` into `S[row]` instead. So the
 *       codes are taken over values that were never normalised and never rounded to bf16, while
 *       the unfused pair would rotate a stored bf16 norm output. That file states the gap itself
 *       ("normalising from a bf16-rounded sum produces thousands of +/-1 code errors and a 1e-3
 *       scale error against 5e-7") and calls storing bf16 correct and normalising from it not.
 *       r4d_gated_had_quant_i8.hip is the same shape and says so outright: "silu(x) = x *
 *       sigmoid(x) is computed in fp32. Doing it in bf16 would cost more than the quantiser that
 *       follows" -- which is the exact opposite of the fp8 sibling's bargain, and it is what makes
 *       `silu_mul` + `had_quant_act_i8` a different function. That row also reads a CLIP from the
 *       environment (R4D_CLIP_MLP_DOWN), so it is not even a pure function of the geometry.
 *
 *   had_quant_act_i8, quant_act_i8, quant_act_fp8
 *       NOT FUSIONS. They are primitive ops of the vocabulary with nothing to decompose into: the
 *       rotation has no op name of its own, and a chain of one step naming the op itself would be
 *       a tautology that inflates the coverage metric and checks nothing.
 *
 *   ar_ln_had_quant_i8
 *       The collective's half is expressible -- `all_reduce` then `rmsnorm_had_quant_i8`, both ops
 *       -- and the arithmetic is not. r4d_ar_ln_had_quant_i8.hip states its whole design as "the
 *       all-reduce OUTPUT never exists ... the peer's contribution is simply a second operand on
 *       the norm's load -- `t = mine + peer + residual`": one f32 expression, with no bf16 round
 *       between the reduce and the add. The pair it would replace has one, because the standalone
 *       all-reduce stores bf16. Its `ARN_W6_MATCH` makes only the COMPRESSED wire round, and only
 *       to match the compressed all-reduce. The fp8 sibling below took the opposite decision --
 *       `a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a;` -- and that one difference is why one of
 *       these two rows gets a hook and the other does not.
 *
 *   qk_norm_rope_gate
 *       Stated in docs/OPS.md and in the row itself: "It is not byte-identical to the four kernels
 *       it replaces", the summation partition differs, and selecting it is a rebaseline.
 *
 *   sample_chain
 *       It "stands beside the decomposed stages rather than replacing them" (r4d_rows.cpp). One
 *       workgroup walking an L2-resident row is not eight launches over a 248K vocabulary in any
 *       rounding sense, nothing in this project implements those eight on a device, and the fused
 *       row additionally produces `tok_excl` and `p_query`, which no chain of them writes.
 *
 *   dflash_conv, dflash_select, the five gdn_* rows
 *       "Neither op has an unfused equivalent in the vocabulary" (r4d_shapes.cpp), and the GDN
 *       preamble's constituents -- a causal conv against a state cache, an L2 norm, a per-chunk
 *       gate cumsum -- have no op names either.
 *
 * NO TUNED AXIS IS READ HERE, for the reason r4d_shapes.cpp gives about extents: no axis this
 * plugin declares changes WHAT a kernel computes. They name tile shapes, split counts, wave
 * counts and all-reduce launch geometry. An axis that changed the chain would be read from `p`,
 * where the core appends the chosen point alongside the geometry.
 *
 * Pure host code: no r4d.h, no HIP. It is in the CMake `SOURCES` list beside r4d_rows.cpp,
 * r4d_layout.cpp and r4d_shapes.cpp for the reason that split exists -- the tables a transcription
 * error hides in should compile and be readable on a box with no ROCm.
 */

#include "r4d_plugin.h"

#include <initializer_list>

namespace {

/* ==================================================================== reading the fused geometry
 *
 * NOT r4d_pgeti_or(), and the difference is the one r4d_shapes.cpp spells out: that reader takes
 * RAD_P_INT only, which is right for a LAUNCH -- by then the core has collapsed every range to the
 * band's value -- and wrong here, because this hook, like `opd_shape`, can be called with a
 * geometry whose ranges are still ranges. A RAD_P_RANGE reads as its HIGH bound wherever a scalar
 * is needed, which is the same reading the operand descriptions take and therefore the reading a
 * caller's buffers were sized under. */
int64_t pi(const RadParam* p, int n_p, const char* k, int64_t dflt = 0) {
    if (!p || !k) return dflt;
    for (int i = 0; i < n_p; ++i) {
        if (!p[i].key || std::strcmp(p[i].key, k) != 0) continue;
        if (p[i].kind == RAD_P_INT)   return (int64_t)p[i].ival;
        if (p[i].kind == RAD_P_RANGE) return (int64_t)p[i].ihi;
        return dflt;
    }
    return dflt;
}

/* ==================================================================== building one step
 *
 * A step is built by naming its op, then its parameters in the STEP op's own spelling, then its
 * operand wiring in the STEP op's schema order. Building it in that order rather than filling a
 * struct field by field is what makes the operand COUNT checkable by eye against the schema in
 * r4d_rows.cpp -- the ABI checks it too, and a hook that has drifted from its schema is a
 * diagnosable error rather than a silent misfire, but a list a reader can count is the half that
 * stops the drift happening.
 *
 * `ok` carries a required parameter the fused geometry did not have. It is checked once, at
 * `wire()`, for the reason r4d_shapes.cpp checks its whole operand list before handing out one
 * entry: a chain whose second step is missing an `eps` is a chain a caller would run and compare,
 * and a step resolved at the wrong geometry produces a difference that reads as the kernel's.
 */
struct Step {
    RadFuseStep* s;
    bool         ok = true;

    Step(RadFuseStep* out, const char* op) : s(out) {
        *s = RadFuseStep{};
        s->op = op;
    }

    /* Copy a parameter of the fused op VERBATIM, kind and all. Verbatim matters: `M` is the ranged
     * parameter of every op in these chains, and re-emitting a range as its collapsed high bound
     * would hand the step a different BAND from the one the fused kernel was selected for -- which
     * for a banded family like the fp8 GEMMs is a different kernel with a different summation
     * order. `eps` and `wadd` are RAD_P_F64 and would be destroyed by an integer reader. */
    Step& mirror(const RadParam* p, int n_p, const char* key) {
        const RadParam* f = rad_param_find(p, n_p, key);
        if (!f || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        s->p[s->n_p++] = *f;
        return *this;
    }

    /* The same, for a parameter that is OPTIONAL in both ops: absent stays absent rather than
     * becoming an explicit default. The two are not the same thing to a constraint -- libr4d's
     * rule is that a constraint whose key the geometry does not carry does not hold -- so
     * inventing `wadd = 0` where the caller supplied none could route a step to a different row. */
    Step& mirror_opt(const RadParam* p, int n_p, const char* key) {
        const RadParam* f = rad_param_find(p, n_p, key);
        if (!f) return *this;
        return mirror(p, n_p, key);
    }

    /* Built field by field rather than through rad_types.h's RAD_INT / RAD_STR: those are compound
     * literals, which are C and only a GNU extension in C++, and this file is ordinary host C++
     * that has to compile with or without ROCm present. */
    Step& pint(const char* key, long long v) {
        if (s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key  = key;
        q.kind = RAD_P_INT;
        q.ival = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    Step& pstr(const char* key, const char* v) {
        if (!v || s->n_p >= RAD_MAX_FUSE_PARAMS) { ok = false; return *this; }
        RadParam q{};
        q.key  = key;
        q.kind = RAD_P_STR;
        q.sval = v;
        s->p[s->n_p++] = q;
        return *this;
    }

    /* ONE ENTRY PER OPERAND OF THE STEP'S OP, in its schema's order. A non-negative value is an
     * operand index of the FUSED kernel, RAD_FUSE_TMP(t) a temporary this chain passes between its
     * own steps, RAD_FUSE_NONE an optional operand this step does not pass. */
    int wire(std::initializer_list<int> from) {
        if (!ok) return RAD_E_UNSUPPORTED;
        if (from.size() > RAD_MAX_FUSE_OPD) return RAD_E_INVAL;
        for (int v : from) s->from[s->n_from++] = (int16_t)v;
        return RAD_OK;
    }
};

/* The activation a gated fold applies, in the vocabulary's spelling. `act` is OPTIONAL on both
 * quantiser ops and the two shims default it DIFFERENTLY -- r4d_fp8.cpp reads `mode` 0 (silu) for
 * `gated_quant_fp8` and `mode` 1 (sigmoid) for `gate_quant_fp8`, because the first serves
 * down_proj's input and the second a gated attention's o_proj -- so the default is the caller's,
 * not this file's, and each hook passes its own. An `act` this plugin does not compile is
 * RAD_E_UNSUPPORTED at the launch too, so a chain for it would be a chain for a kernel that
 * refuses the call. */
bool act_is(const RadParam* p, int n_p, const char* dflt, const char* want) {
    const char* a = r4d_pgets(p, n_p, "act");
    return std::strcmp(a ? a : dflt, want) == 0;
}

/* Every fp8 fold in this file quantises on the CHECKPOINT'S scale grid and refuses anything else
 * by name -- the shims test `group` against r4d_quant_act_fp8_group() and so do the operand
 * descriptions. A geometry that declares a different one has no fused kernel, so it has no chain
 * either, and saying so here keeps the two hooks answering the same set. */
bool fp8_geometry_ok(const RadParam* p, int n_p, int64_t M, int64_t n) {
    return M > 0 && n > 0 && pi(p, n_p, "group", R4D_FP8_BLOCK) == R4D_FP8_BLOCK;
}

}  /* namespace */

/* ==========================================================================================
 * rmsnorm_quant_fp8 -- the RMS norm and the fp8 activation quantiser in one pass.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, rmsnorm_quant_fp8_kernel; shim: r4d_fp8.cpp)
 *
 * Operands: [0] x  [1] residual?  [2] w  [3] q  [4] scale  [5] out_bf16?
 *
 * THE CHAIN IS `add` + `rmsnorm` + `quant_act_fp8`, AND IT IS NOT `rmsnorm_add` + `quant_act_fp8`.
 * That distinction is the reason to read the kernel rather than the row's prose: the other choice
 * produces a chain that fails a byte comparison for no visible reason. The two candidates differ
 * in ONE place -- which value the sum of squares is taken over:
 *
 *   this kernel   `h = f2bf(bf2f(rr[i]) + bf2f(xr[i])); rr[i] = h; v = bf2f(h); ss += v * v;`
 *                 The sum is rounded to bf16, STORED, and read back before the variance. Its own
 *                 entry point says so in as many words: "`res += x` in f32, ROUNDED TO BF16 and
 *                 stored, and the norm is then taken of the stored value -- which is bit-for-bit
 *                 the `add` + `rmsnorm` pair this replaces".
 *
 *   rmsnorm_add   `float v = ldt(x, ..) + ldt(res, ..); ss += v * v; stt(ro, .., v);`
 *                 (libref/ref_elementwise.cpp). The variance comes off the UNROUNDED f32 sum
 *                 and only the multiply reads the stored value back. That file calls the choice
 *                 "THE ROUNDING IS THE CONTRACT" and is right to -- it is vLLM's
 *                 fused_add_rms_norm -- but it is a different function from this kernel's, by
 *                 half a ULP on every channel, and the whole point of the declaration is that a
 *                 caller may compare the bytes.
 *
 * So the residual add is spelled as the op that actually matches, and the payoff is larger than
 * correctness alone: `add`, `rmsnorm` and `quant_act_fp8` are all rows of THIS plugin
 * (add_bf16, rmsnorm_bf16, quant_act_fp8), so the chain can be checked libr4d-against-libr4d,
 * which is the byte-identical comparison the ABI describes rather than a tolerance against a host
 * oracle. `rmsnorm_add` has no libr4d row at all and would have forced the check cross-plugin.
 *
 * THE BYTE-IDENTITY RESTS ON THE WORKGROUP WIDTH AGREEING, which is worth knowing because it is
 * the thing most likely to break silently. Both kernels pick their thread count from
 * `r4d_act_threads(M, n)` and the width decides which columns a thread folds into the sum of
 * squares; r4d_model_bf16.hip's launcher says "r4d_rmsnorm_quant_fp8 calls the SAME function, and
 * it has to". `M` is mirrored verbatim below for exactly that reason.
 *
 * Step 0 writes its output INTO the fused residual operand, which is both its own `b` and the
 * fused kernel's INOUT residual. That is not an accident of the wiring: the fused kernel overwrites
 * the residual in place, so a chain that wrote the sum anywhere else would leave a different set of
 * buffers behind and the comparison would miss the difference. `add_bf16` is safe in place --
 * binary_kernel reads a[i] and b[i] and writes y[i], one element per thread, no neighbours.
 *
 * Step 1's `y` is fused operand 5 rather than a temporary, and the ABI's own note explains why:
 * the fold still WRITES its intermediate when asked to, precisely so that the fused and unfused
 * forms leave the same buffers behind. `norm_quant_row` stores exactly the bf16 it then quantises
 * (`h = f2bf(bf2f(nr[c]) * rs * (g + wadd)); if (yr) yr[c] = h; v[j] = bf2f(h);`), so operand 5 is
 * not an optimisation the chain can route around -- it IS the value the codes were taken over. See
 * the file header for why passing it is assumed rather than observed.
 */
extern "C" int r4d_fuse_rmsnorm_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                          RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;

    /* The two optional operands, read off the call rather than assumed. `residual` decides whether
     * there is an `add` at all -- and therefore how long the chain is -- and `out_bf16` decides
     * whether the bf16 the codes are taken over is a buffer the caller can see or one the chain
     * has to make for itself. */
    const bool has_res = RAD_FUSE_HAS(present, 1) != 0;
    const bool has_out = RAD_FUSE_HAS(present, 5) != 0;
    /* The value the quantiser reads: the fused kernel's own out_bf16 where it has one, so both
     * sides leave the same buffers behind, and a temporary where it does not. The kernel writes
     * that bf16 either way (`h = f2bf(...); if (yr) yr[c] = h; v[j] = bf2f(h);`) -- operand 5 is
     * not an optimisation the chain can route around, it IS the value the codes were taken over. */
    const int16_t mid = has_out ? (int16_t)5 : RAD_FUSE_TMP(0);
    /* Without a residual there is no `add`, so every step shifts down by one. */
    const int s = has_res ? step : step + 1;

    switch (s) {
    case 0: {
        /* residual = x + residual, rounded to bf16 -- the store the norm then reads back. */
        Step st(out, "add");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return st.wire({ 0, 1, 1 });                /* a, b, y */
    }
    case 1: {
        /* The norm of the STORED sum -- or of `x` itself when nothing was added to it. `wadd` is
         * mirrored only when the caller supplied it. */
        Step st(out, "rmsnorm");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "eps")
          .mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wadd");
        return st.wire({ has_res ? (int16_t)1 : (int16_t)0, 2, mid });   /* x, w, y */
    }
    case 2: {
        /* The codes, taken over the bf16 the norm wrote. */
        Step st(out, "quant_act_fp8");
        st.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return st.wire({ mid, 3, 4 });              /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gated_quant_fp8 -- an elementwise gated product and the fp8 quantiser in one pass.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, gated_quant_fp8_kernel)
 *
 * Operands: [0] gate_up  [1] q  [2] scale  [3] out_bf16?
 *
 * `act` silu IS expressible and `act` sigmoid is NOT, and the reason is not the one it looks like.
 * Both `sigmoid` and `mul` are ops of the vocabulary and both have libr4d rows (sigmoid_bf16,
 * mul_bf16), so the missing piece is not an activation -- it is that `gate_up` is ONE [M, 2n]
 * buffer whose halves are the gate and the up, and NOTHING IN THE OPERAND WIRING CAN NAME HALF OF
 * AN OPERAND. `silu_mul` is the one op in the vocabulary that takes the packed plane and does the
 * split itself, which is what makes the silu form reachable; there is no `sigmoid_mul` beside it.
 *
 * There is a second, independent reason, and it is the one that would still bite if such an op
 * were added carelessly. This kernel keeps the activation in f32 through the product and rounds
 * ONCE:
 *
 *     const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
 *     const uint16_t h = f2bf(a * bf2f(gr[n + c]));
 *
 * which is `silu_mul_bf16` element for element -- `f32_to_bf16((g / (1.0f + expf(-g))) * bf2f(gr[n
 * + i]))` in r4d_model_bf16.hip -- and is NOT `sigmoid` followed by `mul`, which would round the
 * activation to bf16 in between. Its sibling `gate_quant_fp8` below rounds there deliberately,
 * because ITS reference is two kernels; this one's is one. A `sigmoid_mul` op would have to make
 * the same choice this kernel did or the chain would be a different function again.
 *
 * So the sigmoid form gets RAD_E_UNSUPPORTED -- an honest decline, and the vocabulary gap it names
 * is a `sigmoid_mul` over a packed [M, 2n] plane that rounds once, not a missing `sigmoid`.
 * Nothing is lost: cGemmNtQGatedFp8 and the arch both reach the sigmoid gate through
 * `gate_quant_fp8`, whose operands are two buffers rather than one plane.
 */
extern "C" int r4d_fuse_gated_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                        RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;
    /* r4d_fp8.cpp's shim defaults this row's `mode` to 0, which is silu. */
    if (!act_is(p, n_p, "silu", "silu")) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* silu(gate) * up over the packed [M, 2n] plane, into the bf16 the codes are taken over. */
        Step s(out, "silu_mul");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ 0, 3 });                    /* gate_up, y */
    }
    case 1: {
        Step s(out, "quant_act_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return s.wire({ 3, 1, 2 });                 /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gate_quant_fp8 -- the same epilogue with the gate in its own operand.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, gate_quant_fp8_kernel)
 *
 * Operands: [0] gate  [1] x  [2] q  [3] scale  [4] out_bf16?
 *
 * THIS ONE IS EXPRESSIBLE AND ITS SIBLING ABOVE IS NOT, which is the opposite of what the operand
 * lists suggest, and the kernel says why in a comment written for exactly this question:
 *
 *     const uint16_t ab = f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
 *     const uint16_t h  = f2bf(bf2f(ab) * bf2f(xr[c]));
 *     // THE ACTIVATION IS ROUNDED BEFORE THE MULTIPLY, which is the one place this differs from
 *     // gated_quant_fp8 above and it is not an oversight. [...] This one's is `sigmoid` then
 *     // `mul`, TWO kernels with a bf16 buffer between them -- so the round trip is part of the
 *     // arithmetic being replaced, and skipping it changes the model's output. [...]
 *
 * That round trip is the chain. `sigmoid_bf16` computes `f32_to_bf16(1.0f / (1.0f +
 * expf(-bf2f(xr[i]))))` and `mul_bf16` computes `f32_to_bf16(bf2f(a) * bf2f(b))`, which are the
 * two lines above term for term, and the gate arrives as its own [M, n] operand so there is no
 * half-of-an-operand problem. Three launches an attention layer becoming one is what the fusion
 * bought; the three are named here.
 *
 * `act` silu IS DECLINED, and not because the arithmetic differs -- `silu` is an op of the
 * vocabulary and `f2bf(gv / (1.0f + expf(-gv)))` is what a silu step would have to compute. It is
 * declined because THIS PLUGIN HAS NO `silu` ROW: libr4d declares `sigmoid` (sigmoid_bf16) and
 * `silu_mul` (silu_mul_bf16) and no bare `silu`, so the only implementation a caller could reach
 * is libref's, whose `act_silu(x)` is `x * act_sigmoid(x)` -- two f32 roundings where this
 * kernel does one divide. A chain that resolved there would report a difference in the last bit of
 * every element and call it a fault in the kernel, and a false failure is worse than a skip. A
 * libr4d `silu` row spelled `x / (1 + expf(-x))` would close the gap; without one the decline
 * stands. Nothing in the tree issues this row with silu -- the gated attention's gate is a
 * sigmoid, which is why the shim's `mode` defaults to 1.
 *
 * TMP(0) is act(gate): an [M, n] bf16 plane the fusion genuinely never writes, so it is a
 * temporary rather than an operand. `out_bf16` is not -- the kernel stores the product there when
 * asked, and the codes are taken over those same bytes.
 */
extern "C" int r4d_fuse_gate_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                       RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (!fp8_geometry_ok(p, n_p, M, n)) return RAD_E_UNSUPPORTED;
    /* r4d_fp8.cpp's shim defaults this row's `mode` to 1, which is sigmoid. */
    if (!act_is(p, n_p, "sigmoid", "sigmoid")) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* The gate through the logistic, rounded to bf16 -- the buffer the pair had between them. */
        Step s(out, "sigmoid");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ 0, RAD_FUSE_TMP(0) });      /* x, y */
    }
    case 1: {
        Step s(out, "mul");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "dtype");
        return s.wire({ RAD_FUSE_TMP(0), 1, 4 });   /* a, b, y */
    }
    case 2: {
        Step s(out, "quant_act_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        return s.wire({ 4, 2, 3 });                 /* x, q, scale */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gemm_nt_q_gated -- the narrow fp8 GEMM with the SwiGLU and the quantiser in its epilogue.
 * (row: gemm_fp8a8_gated_nt_m16; shim: r4d_fp8.cpp)
 *
 * Operands: [0] a  [1] a_scale  [2] b  [3] b_scale  [4] q  [5] scale  [6] out_bf16?
 *
 * The row already states the claim -- "Byte-identical to gemm_nt_q followed by gated_quant_fp8 --
 * the accumulator is rounded to bf16 before the gate and the multiply, and the amax is an fmaxf
 * reduction, which is exact and order-free however it is regrouped across the waves" -- and this
 * is that sentence made executable. Both steps are libr4d rows, so the check is
 * libr4d-against-libr4d.
 *
 * TMP(0) IS THE ONE GENUINE TEMPORARY IN THIS FILE: the [M, N] bf16 plane the unfused pair passed
 * between the GEMM and the quantiser, and the plane whose non-existence is the entire point of the
 * fusion ("no [M, 2n] tensor is ever written"). Its extents come from step 0's own `opd_shape`,
 * which is the ABI's rule for a temporary, and `r4d_shape_gemm_nt_q` answers [M, N] bf16 for
 * operand 4 while `r4d_shape_gated_quant_fp8` answers [M, 2n] for operand 0 -- the same plane,
 * since `n` below is N/2.
 *
 * TWO PLACES WHERE A PARAMETER CANNOT SIMPLY BE COPIED, and both are the kind of thing that
 * resolves to a plausible wrong kernel rather than to an error:
 *
 *   `n`      is N/2 and not N. `N` on the fused op is the WEIGHT'S ROWS -- twice the fused width,
 *            gate then up -- which is exactly why this op has a schema of its own rather than a
 *            flag on `gemm_nt_q`. The quantiser's row is the OUTPUT width.
 *
 *   `dtype`  is "bf16" on the quantiser step and "fp8a8" on the GEMM step, from one key on the
 *            fused op. `dtype` names the operand COMBINATION for a GEMM and the width of the
 *            tensor an activation op READS, and the constraint rows disagree accordingly:
 *            cGemmNtQGatedFp8 has `C_IN("dtype", "fp8a8")` and cGatedQuantFp8 has
 *            `C_IN("dtype", "bf16")`. Mirroring the fused value into step 1 would match no row of
 *            this plugin, and the chain would be unresolvable rather than wrong -- but only
 *            because nothing else declares a bf16-named `gated_quant_fp8`, which is not a
 *            guarantee worth relying on.
 *
 * `a_sum` and `b_ref` are the two operands libr4d appended after `gemm_nt_q`'s `y`, and an fp8
 * grid has neither -- `a_sum` belongs to the asymmetric integer grids and `b_ref` to mxfp4, and
 * r4d_shape_gemm_nt_q says the same by describing both as absent. They are RAD_FUSE_NONE and not
 * simply omitted, because ONE ENTRY PER SCHEMA OPERAND is what makes the count a drift check.
 */
extern "C" int r4d_fuse_gemm_nt_q_gated(const RadParam* p, int n_p, uint64_t present, int step,
                                        RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const char* dt = r4d_pgets(p, n_p, "dtype");
    if (M <= 0 || N <= 0 || K <= 0 || N % 2) return RAD_E_UNSUPPORTED;
    if (pi(p, n_p, "group", R4D_FP8_BLOCK) != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    if (!dt || std::strcmp(dt, "fp8a8")) return RAD_E_UNSUPPORTED;
    /* REQUIRED on this op, unlike the quantiser's, because only the silu form is compiled -- an
     * optional parameter would let the row resolve for a gate it cannot serve. */
    if (!act_is(p, n_p, "silu", "silu")) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        /* The GEMM over the whole [N, K] gate_up plane, into the bf16 that never exists. */
        Step s(out, "gemm_nt_q");
        s.mirror(p, n_p, "M").mirror(p, n_p, "N").mirror(p, n_p, "K")
         .mirror(p, n_p, "group").mirror(p, n_p, "dtype");
        /* a, a_scale, b, b_scale, y, a_sum, b_ref */
        return s.wire({ 0, 1, 2, 3, RAD_FUSE_TMP(0), RAD_FUSE_NONE, RAD_FUSE_NONE });
    }
    case 1: {
        /* The gate, the multiply and the codes over the accumulator plane, at half its width. */
        Step s(out, "gated_quant_fp8");
        s.mirror(p, n_p, "M").pint("n", (long long)(N / 2)).mirror(p, n_p, "group")
         .pstr("dtype", "bf16").mirror(p, n_p, "act");
        return s.wire({ RAD_FUSE_TMP(0), 4, 5, 6 });   /* gate_up, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * gdn_conv_recurrent_update -- the decode convolution folded into the recurrent update's prologue.
 * (row: gdn_conv_recurrent_update_k128_v128_bf16; kernel: libr4d/
 * r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip, r4d_gdn_recurrent_update_kernel under CV)
 *
 * Operands: [0] x  [1] conv_w  [2] conv_b  [3] conv_state  [4] conv_idx  [5] a  [6] b  [7] A_log
 *           [8] dt_bias  [9] state  [10] state_idx  [11] num_accepted  [12] cu  [13] z
 *           [14] norm_w  [15] o  [16] o_q  [17] o_scale
 *
 * The pair's q, k and v are the three temporaries. Every parameter is the step's own under the
 * same name, so the chain is copies: the convolution's four, then the recurrent update's. */
extern "C" int r4d_fuse_gdn_conv_recurrent_update(const RadParam* p, int n_p, uint64_t present,
                                                  int step, RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    const int16_t cb = RAD_FUSE_HAS(present, 2) ? 2 : RAD_FUSE_NONE;
    const int16_t oq = RAD_FUSE_HAS(present, 16) ? 16 : RAD_FUSE_NONE;
    const int16_t os = RAD_FUSE_HAS(present, 17) ? 17 : RAD_FUSE_NONE;
    switch (step) {
    case 0: {
        Step s(out, "gdn_conv_update");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_k").mirror(p, n_p, "head_v")
         .mirror(p, n_p, "conv_width");
        /* x, w, b, conv_state, state_idx, num_accepted, cu, q, k, v */
        return s.wire({ 0, 1, cb, 3, 4, 11, 12, RAD_FUSE_TMP(0), RAD_FUSE_TMP(1),
                        RAD_FUSE_TMP(2) });
    }
    case 1: {
        Step s(out, "gdn_recurrent_update");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_k").mirror(p, n_p, "head_v")
         .mirror_opt(p, n_p, "n_head_v").mirror_opt(p, n_p, "scale").mirror_opt(p, n_p, "eps")
         .mirror_opt(p, n_p, "l2_eps").mirror_opt(p, n_p, "softplus_thr")
         .mirror_opt(p, n_p, "act").mirror_opt(p, n_p, "state_fp16")
         .mirror_opt(p, n_p, "state_form");
        /* q, k, v, a, b, A_log, dt_bias, state, state_idx, num_accepted, cu, z, norm_w, o, o_q,
         * o_scale */
        return s.wire({ RAD_FUSE_TMP(0), RAD_FUSE_TMP(1), RAD_FUSE_TMP(2), 5, 6, 7, 8, 9, 10, 11,
                        12, 13, 14, 15, oq, os });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * qk_norm_rope_gate_kv_store -- the gated-attention prologue with the cache store folded on.
 * (rows: qk_norm_rope_gate_kv_store_fp8 and _bf16; kernel: libr4d/r4d_qk_norm_rope_gate.hip,
 * r4d_qk_norm_rope_gate_kernel under KV)
 *
 * Operands: [0] q_gate?  [1] k  [2] cos_sin  [3] positions  [4] q_w?  [5] k_w  [6] q_out?
 *           [7] k_out  [8] gate_out?  [9] v  [10] slot_mapping  [11] kv_cache
 *
 * No temporaries: the store reads the prologue's own `k_out`, which the fold writes as well. */
extern "C" int r4d_fuse_qk_norm_rope_gate_kv_store(const RadParam* p, int n_p, uint64_t present,
                                                   int step, RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    auto opt = [&](int i) -> int16_t {
        return RAD_FUSE_HAS(present, i) ? (int16_t)i : (int16_t)RAD_FUSE_NONE;
    };
    switch (step) {
    case 0: {
        Step s(out, "qk_norm_rope_gate");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head")
         .mirror(p, n_p, "n_head_kv").mirror(p, n_p, "rot").mirror(p, n_p, "q_dtype")
         .mirror_opt(p, n_p, "eps").mirror_opt(p, n_p, "wadd").mirror_opt(p, n_p, "cs_f32")
         .mirror_opt(p, n_p, "pos_i64");
        return s.wire({ opt(0), 1, 2, 3, opt(4), 5, opt(6), 7, opt(8) });
    }
    case 1: {
        Step s(out, "kv_store");
        s.mirror(p, n_p, "M").mirror(p, n_p, "head_dim").mirror(p, n_p, "n_head_kv")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "kv_dtype")
         .mirror_opt(p, n_p, "k_scale").mirror_opt(p, n_p, "v_scale");
        /* k, v, slot_mapping, kv_cache */
        return s.wire({ 7, 9, 10, 11 });
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * ar_rmsnorm_quant_fp8 -- the two-rank all-reduce with the norm and the quantiser folded on.
 * (kernel: libr4d/r4d_fused_quant_fp8.hip, ar_rmsnorm_quant_fp8_kernel; shim: r4d_ar.cpp)
 *
 * Operands: [0] x (inout)  [1] residual?  [2] w  [3] q  [4] scale  [5] out_bf16?
 *
 * THE CHEAPEST HONEST CHAIN IN THE FILE, and the one that needs no assumption about optional
 * operands at all. Its second step is `rmsnorm_quant_fp8`, which is itself an op of the vocabulary
 * and carries the SAME two optional operands in the same positions -- so `residual` and `out_bf16`
 * are wired straight through and absence propagates through the wiring rather than having to be
 * observed. That is worth noticing as a design rule: a fold whose tail is itself a declared op
 * hands the discrimination to the step, and the ABI gap described in the file header disappears.
 *
 * BOTH WIRES, and the kernel earns the claim rather than approximating it. On the exact wire pass
 * one is `a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a;` -- the standalone one-shot's own f32 sum
 * stored as bf16, written back into `x` so that "a reader of `x` that is not this norm still finds
 * the same bytes". On the compressed wire the reduce loop stores the inverse-rotated sum into `x`
 * as bf16 "which is what the standalone compressed all-reduce wrote, so pass two below reads the
 * bytes it would have read". Everything after the handshake is `norm_quant_row`, the same device
 * function the standalone fold above runs. Contrast `ar_ln_had_quant_i8`, which takes the opposite
 * decision (`t = mine + peer + residual` with no round between) and therefore publishes no hook.
 *
 * THREE PARAMETERS THAT ARE NOT COPIES, and each of them decides which all-reduce row serves the
 * step:
 *
 *   `numel`      is M * n. The fused op declares the message by its [M, n] operand and
 *                `all_reduce` declares it as an element count -- and as THE LARGEST message the
 *                instance will carry, not the one a step passes, which is why the band's high
 *                bound is the right value and `pi` reads a range as one.
 *
 *   `exact`      is the fused `wire` inverted. It is REQUIRED on `all_reduce` and it is the
 *                predicate that separates cArExact from cArWht6; a caller that does not declare it
 *                gets no kernel at all.
 *
 *   `min_bytes`  is 0, and ONLY on the compressed wire. The wht6 shim serves a message below its
 *                floor with the EXACT kernel instead -- safe, because both kernels share the
 *                instance's scratch and flag space -- and the fused kernel has no such fallback: a
 *                `wire` of 6 is 6 at every size. Its comment names the spelling for this exact
 *                case: "A caller that needs the rotated wire unconditionally asks for a floor of
 *                0." Leaving the key out would let a small message go out exact on one side of the
 *                comparison and rotated on the other.
 *
 * `hops` is deliberately NOT emitted. A query that omits it falls through to the one-shot rows,
 * which is what this kernel is; naming it would be pinning a topology that does not exist at two
 * ranks. `y` is RAD_FUSE_NONE: the fused kernel reduces in place into `x`, which is what the
 * one-shot rows do and what the two-shot rows refuse by name.
 */
/* ==========================================================================================
 * attn_paged_gate_quant -- attn_paged, then gate_quant_fp8 over its output.
 * (row: attn_decode_gq_h256_gqa12_fp8kv; kernels: libr4d/r4d_attn_decode_h256_gqa6.hip, the
 * `_gq` split-KV merges)
 *
 * Operands: [0] q  [1] kv_cache  [2] block_table  [3] seqused  [4] k_descale?  [5] v_descale?
 *           [6] cu_seqlens?  [7] out  [8] gate  [9] q8  [10] q8_scale
 *
 * No temporaries: gate_quant_fp8 reads `out` and writes its bf16 product back over it, which is
 * the in-place pair the fold replaces. Its rows are attn_paged's (token, head) pairs, so its `M` is
 * the band's query rows times the head count -- a bound, which is all its shim reads it as -- and
 * its `n` is `head_dim`. */
extern "C" int r4d_fuse_attn_paged_gate_quant(const RadParam* p, int n_p, uint64_t present,
                                              int step, RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    auto opt = [&](int i) -> int16_t {
        return RAD_FUSE_HAS(present, i) ? (int16_t)i : (int16_t)RAD_FUSE_NONE;
    };
    const int64_t q_len = pi(p, n_p, "q_len"), nh = pi(p, n_p, "n_head");
    const int64_t hd = pi(p, n_p, "head_dim"), seqs = pi(p, n_p, "max_seqs", 1);
    if (q_len <= 0 || nh <= 0 || hd <= 0 || seqs <= 0) return RAD_E_UNSUPPORTED;
    switch (step) {
    case 0: {
        Step s(out, "attn_paged");
        s.mirror(p, n_p, "q_len").mirror(p, n_p, "head_dim").mirror(p, n_p, "gqa")
         .mirror(p, n_p, "block_size").mirror(p, n_p, "causal").mirror(p, n_p, "window")
         .mirror(p, n_p, "q_dtype").mirror(p, n_p, "kv_dtype").mirror_opt(p, n_p, "scale")
         .mirror_opt(p, n_p, "max_ctx").mirror_opt(p, n_p, "n_head")
         .mirror_opt(p, n_p, "max_seqs");
        return s.wire({ 0, 1, 2, 3, opt(4), opt(5), opt(6), 7 });
    }
    case 1: {
        Step s(out, "gate_quant_fp8");
        s.pint("M", (long long)(seqs * q_len * nh)).pint("n", (long long)hd)
         .mirror(p, n_p, "group").pstr("dtype", "bf16").mirror(p, n_p, "act");
        return s.wire({ 8, 7, 9, 10, 7 });   /* gate, x, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

/* ==========================================================================================
 * ar_gather_hc_write -- moe_gather, then ar_hc_write.
 * (kernel: libr4d/r4d_ar_oneshot_2rank_exact.hip, r4d_ar_gather_hc_write_kernel, and at four ranks
 * libr4d/r4d_ar_oneshot_Nrank_exact.hip, r4d_ar_gather_hc_write_wide_kernel; shim: r4d_ar_entry.hip)
 *
 * Operands: [0] y_expert  [1] expert_w  [2] sorted_tok  [3] shared?  [4] shared_gate?  [5] y (inout)
 *           [6] inj  [7] h (inout)
 *
 * Two declared ops and nothing in between: the gather writes `y`, which is the reduction's message
 * and is then reduced in place and written into the streams. The shared-arm pair is optional in
 * both ops at the same positions, so its absence propagates through the wiring. `wire` is the
 * reducing write's, and it goes to that step unchanged; so does `world_size`, which at four selects
 * ar_hc_write_4rank, the reduction and write the four-rank row runs.
 */
extern "C" int r4d_fuse_ar_gather_hc_write(const RadParam* p, int n_p, uint64_t present, int step,
                                           RadFuseStep* out) {
    (void)present;
    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), k = pi(p, n_p, "top_k");
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    switch (step) {
    case 0: {
        Step s(out, "moe_gather");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "top_k")
         .mirror(p, n_p, "dtype").mirror_opt(p, n_p, "act");
        return s.wire({ 0, 1, 2, 3, 4, 5 });   /* y_expert, expert_w, sorted_tok, shared, gate, y */
    }
    case 1: {
        Step s(out, "ar_hc_write");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "hc")
         .mirror(p, n_p, "world_size").mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wire");
        return s.wire({ 5, 6, 7 });            /* y, inj, h */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}

extern "C" int r4d_fuse_ar_rmsnorm_quant_fp8(const RadParam* p, int n_p, uint64_t present, int step,
                                             RadFuseStep* out) {

    if (!out) return RAD_E_INVAL;
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const int64_t world = pi(p, n_p, "world_size");
    const int64_t wire = pi(p, n_p, "wire", 0);
    if (!fp8_geometry_ok(p, n_p, M, n) || world <= 0) return RAD_E_UNSUPPORTED;
    /* The entry point serves 0 and 6 and refuses anything else; a chain for a wire that has no
     * kernel would name a payload neither side can produce. */
    if (wire != 0 && wire != 6) return RAD_E_UNSUPPORTED;

    switch (step) {
    case 0: {
        Step s(out, "all_reduce");
        s.mirror(p, n_p, "world_size").pint("numel", (long long)(M * n))
         .mirror(p, n_p, "dtype").pint("exact", wire == 0 ? 1 : 0);
        if (wire != 0) s.pint("min_bytes", 0);
        return s.wire({ 0, RAD_FUSE_NONE });        /* x (inout), y */
    }
    case 1: {
        Step s(out, "rmsnorm_quant_fp8");
        s.mirror(p, n_p, "M").mirror(p, n_p, "n").mirror(p, n_p, "eps")
         .mirror(p, n_p, "group").mirror(p, n_p, "dtype").mirror_opt(p, n_p, "wadd");
        return s.wire({ 0, 1, 2, 3, 4, 5 });        /* x, residual, w, q, scale, out_bf16 */
    }
    default:
        return RAD_E_NOTFOUND;
    }
}
