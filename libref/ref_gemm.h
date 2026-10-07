/* ref_gemm.h -- how libref reads a matrix operand and its scale stream.
 *
 * Shared by the dense GEMM and the mixture-of-experts GEMM, which need the same three answers:
 * where row `r` of an operand begins, what the scale of a (row, k) pair is, and what a block-
 * scaled grid looks like against the operand it scales. A grouped GEMM is a GEMM per row against
 * a different weight, so it needs exactly the machinery the dense one has and nothing else; a
 * second copy of `make_scales` in ref_moe.cpp would be a second place for the rblk rule to be
 * wrong, and the rblk rule is what makes this reference usable as an oracle for a block-scaled
 * fp8 checkpoint.
 *
 * Everything here is a pure reading of a RadTensor. No op, no dispatch, no allocation.
 */
#ifndef REF_GEMM_H
#define REF_GEMM_H

#include "ref_common.h"

namespace ref {

/* A tensor read as [rows, cols], which is what every GEMM here wants and is NOT the same as
 * "dim 0 and dim 1".
 *
 * `stride[rank - 2]` is the row stride only when the LAST dimension is already `cols`. An
 * activation that arrives rank 3 -- [tokens, heads, width], which is how the gated delta net's
 * out-projection input is declared, because the scan kernels require that rank -- is read as
 * [tokens, heads * width], and its row stride is the product of the trailing dimensions, not the
 * head stride. Taking the head stride instead lands row 0 correctly and starts every row after it
 * `width` elements in -- which reads as a correct implementation right up to the second row.
 *
 * THE OUTPUT SIDE IS THE WORSE HALF and has the same rule. A gated attention layer's q|gate
 * projection is declared [tokens, heads, 2 * head_dim] -- rank 3, because the de-interleaving that
 * follows needs the head axis -- and writing it at the HEAD stride does not merely write the wrong
 * bytes, it leaves most of the buffer NEVER WRITTEN, so rows after the first carry a tail of exact
 * zeros.
 *
 * This is libref, so it is the ORACLE: every tool that certifies a kernel against this arithmetic
 * inherits whatever it gets wrong, which is why the rule is stated here once. */
static inline int64_t row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    /* Fold trailing dimensions until they span `cols`; the next one out carries the row. */
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];      /* the shapes disagree; fall back to the row axis */
}

/* The element offset of row `r` of a [rows, cols] operand, and the stride along cols. Weights
 * arrive contiguous from the mapper; activations may be a slice of a wider arena buffer, so both
 * are read through their strides. */
struct Mat {
    const RadTensor* t;
    int64_t rs;      /* elements between rows */
    int64_t cs;      /* elements between columns */
};

static inline Mat as_mat(const RadTensor* t, int64_t rows, int64_t cols) {
    Mat m{ t, cols, 1 };
    (void)rows;
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

/* ------------------------------------------------------------------ scale streams
 * A quantised operand's scales are a separate SoA stream (spec §4.1), shaped [rows, groups] with
 * the group index fastest, or [rows] for one scale per row. The group SIZE is derived from the
 * stream's own extent rather than trusted from the parameter: a `group` that disagrees with the
 * scale tensor is the kind of mismatch that produces plausible wrong numbers, and deriving it
 * makes the disagreement structurally impossible.
 *
 * A BLOCK-SCALED grid is the same stream with fewer rows than the operand. DeepSeek-style fp8 --
 * which is what Qwen3.8-27B-FP8 and every other `weight_block_size: [128, 128]` checkpoint ships --
 * stores one scale per 128x128 TILE, so `weight_scale_inv` is [ceil(N/128)][ceil(K/128)] against an
 * [N][K] weight. `rblk` is how many operand rows share a scale row; it is 1 for every per-row grid,
 * which is every other format in this project, so the division below costs those nothing.
 *
 * Without it this reference reads a 17408-row weight's 136-row scale plane as if it had 17408 rows
 * and lands on `groups = 0`, which is not a diagnosable failure -- it is a plausible wrong number,
 * and it makes ref useless as the oracle for exactly the format the production models are in. */
struct Scales {
    const void* p;
    uint32_t dt;
    int64_t   rs;        /* elements between scale rows */
    int64_t   cs;        /* elements between groups */
    int64_t   group;     /* K elements per scale */
    int64_t   rblk;      /* OPERAND rows per scale row; 1 unless the grid is block-scaled */
    bool      e8m0;      /* the scale stream is OCP e8m0 exponent bytes, not floats */

    /* THE OCP-MX `b_ref` REFERENCE, and it is NOT an optimisation of the e8m0 read -- it is a
     * different value. The device never reads a block's exponent as the scale: it takes a per-row
     * reference Wref[n] = max_k E8M0[n][k] once, folds 2^-dsh (dsh = clamp(d, 0, 15)) into the E4M3
     * ELEMENT as E4M3_RNE(code * 2^-dsh), and multiplies the row by 2^(Wref[n]-127) in the epilogue.
     * The recovered value is therefore a per-CODE element, not a per-block scale: the E4M3 rounding
     * is code-dependent and the fold UNDERFLOWS TO ZERO for dsh around 13+, so a block far below its
     * row's reference contributes nothing. Reading the plane directly (the old behaviour) disagrees
     * with the device on exactly those blocks, for every value in them, and the disagreement is
     * quiet -- both answers are plausible floats. A `b_ref` present is what makes the fold
     * authoritative, so it is only attached when one was supplied; without it the e8m0 plane is read
     * directly and the pre-existing fp8/b_ref-less paths are bit-for-bit unchanged. See ref_quant.cpp
     * for the two call sites that attach it. */
    const uint8_t* ref;  /* per-output-row E8M0 reference bytes, or null */
    int64_t        ref_rs;   /* elements between reference rows */
    int64_t        ref_cs;   /* elements between reference blocks within a row */
    int64_t        ref_n;    /* blocks a row when `ref` is a full e8m0 plane, else 1 */

    /* The per-row reference exponent `Wref[n]`. It is usually already the derived per-row scalar a
     * kernel reads (a [N] or [N,1] E8M0 stream), but the container's canonical encoding plane is the
     * same [N][K/32] e8m0 plane `b_scale` came from, and an oracle that reconstructs operands from
     * the canonical planes may hand that over instead. Both are handled here and agree by
     * construction: the plane form takes the row's maximum, which is exactly how the device derives
     * its `Wref`. A `b_scale` row and a derived reference therefore produce the same answer as a
     * pre-derived one -- and the max is over the REFERENCE plane, not the b_scale one, so a test is
     * free to hand a reference that disagrees with the block plane (the saturated/under-reference
     * cases) and still have both sides read the same value. */
    inline int wref_at(int64_t row) const {
        int w = 0;
        for (int64_t b = 0; b < ref_n; ++b) {
            const int v = (int)ref[row * ref_rs + b * ref_cs];
            if (v > w) w = v;
        }
        return w;
    }

    /* dsh = clamp(Wref - E8M0, 0, 15): the block-exponent difference the device folds into the
     * element before the WMMA (r4d_gemm_mxfp4a8_nt_m64.hip's `dsh`). */
    inline int dsh_at(int64_t row, int64_t i, int wref) const {
        int d = wref - (int)((const uint8_t*)p)[i];
        return d < 0 ? 0 : (d > 15 ? 15 : d);
    }

    /* THE DEVICE'S FOLDED E4M3 MAGNITUDE of one raw code. The kernel does NOT multiply the code by
     * 2^-dsh in f32: it stores E4M3_RNE(code * 2^-dsh) -- round-to-nearest-even, UNDERFLOWING TO
     * ZERO below E4M3's smallest subnormal (2^-9) -- and multiplies the whole row by 2^(Wref-127)
     * once in its epilogue. So the folded element is `fold_mag(code, dsh) * row_factor(wref)`: a
     * per-CODE E4M3 rounding a single per-block scale cannot express, and the reason blocks far
     * below their row's reference contribute 0 rather than a small nonzero value. */
    inline float fold_mag(float code, int dsh) const {
        return fp8e4m3_to_f32(f32_to_fp8e4m3(code * std::ldexp(1.0f, -dsh)));
    }
    inline float row_factor(int wref) const { return std::ldexp(1.0f, wref - 127); }

    /* The block scale the fold implies when it is treated as a pure scale: 2^(Wref-dsh-127). It is
     * NOT the folded element -- it omits the per-code E4M3 rounding and the underflow to zero -- so
     * a folded quantised product uses `fold_mag` + `row_factor` (see `quant_dot`). Retained for the
     * b_ref-less direct read and for callers that only need the exponent. */
    inline float block(int64_t row, int64_t i, int wref) const {
        if (e8m0) {
            const uint8_t e = ((const uint8_t*)p)[i];
            if (ref) return e8m0_to_f32((uint8_t)(wref - dsh_at(row, i, wref)));
            return e8m0_to_f32(e);
        }
        return ld_dt(dt, p, i);
    }
    /* The dsh of a given group, computed once so a caller can hoist `wref_at` out of the group
     * loop (the reference row max is invariant across groups). */
    inline int group_dsh(int64_t row, int64_t g, int wref) const {
        const int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return dsh_at(row, i, wref);
    }
    inline float at(int64_t row, int64_t k) const {
        int64_t g = group > 0 ? k / group : 0;
        int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return block(row, i, ref ? wref_at(row) : 0);
    }
    inline float grp(int64_t row, int64_t g, int wref) const {
        int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return block(row, i, wref);
    }
    inline float grp(int64_t row, int64_t g) const {
        return grp(row, g, ref ? wref_at(row) : 0);
    }
};

/* THE BLOCK A GRID OF `blocks` PIECES IMPLIES OVER `ext` ELEMENTS, and it is not ext / blocks.
 * A grid's last block may be partial -- ceil(300 / 128) is 3 scales over 128, 128 and 44 -- so
 * the quotient (100 there) assigns elements 100..127 to the second scale. The block is the POWER
 * OF TWO whose ceiling division gives `blocks`, which is unique whenever there is more than one
 * block, and every scale grid in this project is one (32, 64, 128). Where no power of two fits,
 * the smallest block that yields `blocks` pieces is taken. Either way ceil(ext / block) is at
 * most `blocks`, so an index derived from it never leaves the grid. */
static inline int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

static inline Scales make_scales(const RadTensor* s, int64_t rows, int64_t K, bool mxfp4_weight) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p     = s->data;
    q.dt    = s->dtype;
    /* A rank-2 stream states its own grid, and that statement is preferred: it is the only source
     * that can distinguish "one scale per row over `groups` groups" from "one scale per row BLOCK".
     * FEWER scale rows than operand rows is a block grid, whatever the remainder: [ceil(N/128)]
     * rows against N = 129 is two blocks, the second of one row. As many or more is one scale row
     * per operand row, read through the stream's own row stride.
     *
     * A rank-1 or flattened stream has no grid to state. With a scale for every row it is per row
     * over nel / rows groups; with fewer it is one scale per BLOCK of rows. Reading it per row
     * there walks off the end of the plane at the first row past its length. */
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs   = s->stride[s->rank - 2];
        q.cs   = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) {
            groups = nel / rows;
            q.rs   = groups;
        } else {
            groups = 1;
            q.rblk = grid_block(rows, nel);
            q.rs   = 1;
        }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    /* An OCP-MX block scale is an e8m0 exponent byte. When the stream arrives as u8 beside a
     * block-scaled weight -- e2m1, e2m3 or the MXFP8 e4m3 plane; the flag is named for the first
     * member of that family -- there is nothing else it could be; a float stream beside the same
     * weight is read as a float, so a quantiser that chose to store the scale plainly still works,
     * and block-fp8's bf16 weight_scale_inv lands here as a float whatever the flag says. */
    q.e8m0  = mxfp4_weight && (s->dtype == RAD_U8);
    return q;
}

/* Attach an OCP-MX `b_ref` (a per-output-row E8M0 reference byte) to a weight's scale grid, which
 * turns `Scales::grp`/`at` from a direct e8m0 read into the device's folded element (see `fold_mag`
 * and `quant_dot`). It is a
 * separate step rather than a make_scales parameter because `b_ref` is a distinct OPERAND, not a
 * property of the b_scale stream, and the two are only meaningful together for an mxfp4 weight whose
 * b_scale is itself e8m0 (which is what `s.e8m0` records). Anything else -- a float b_scale, a b_ref
 * of the wrong width or dtype, rows to a different operand -- leaves the grid untouched, so the
 * direct read is the floor and this can never make a well-formed call worse. `rows` is the weight's
 * row count N, because the reference is indexed by output row. */
static inline void scales_ref(Scales& s, const RadTensor* r, int64_t rows) {
    if (!s.e8m0 || !s.p || !r || !r->data || r->dtype != RAD_U8) return;
    const int64_t nel = numel(r);
    if (rows <= 0 || nel < rows) return;
    s.ref    = (const uint8_t*)r->data;
    /* A rank-1 [N] stream has one byte a row; a rank-2 [N, 1] stream carries the row stride itself
     * (and is what the device's own wref layout states). A rank-2 [N, blocks] stream is the canonical
     * e8m0 plane, from which the reference is the row maximum -- see `wref_at`. */
    s.ref_rs = r->rank >= 2 ? r->stride[r->rank - 2] : 1;
    s.ref_cs = r->rank >= 2 ? r->stride[r->rank - 1] : 1;
    const int64_t inner = r->rank >= 2 ? r->shape[r->rank - 1] : 1;
    s.ref_n  = (inner > 1 && nel >= rows * inner) ? inner : 1;
    if (s.ref_rs <= 0) s.ref_rs = 1;
    if (s.ref_cs <= 0) s.ref_cs = 1;
}

/* ------------------------------------------------------------------ one quantised product
 *
 * ONE (m, n) dot of a quantised GEMM, and the ONE DEFINITION of how the scales fold into it.
 *
 * The group sub-sum is accumulated from the RAW CODES and scaled once at the end of the group,
 * which is what an int8 or fp8 WMMA kernel does: it accumulates in the fragment's type and applies
 * the block scales when the tile closes. Folding the scales into every product instead is the same
 * value in exact arithmetic and a different one in the last bits, and rad-kbench would blame the
 * kernel for the reference's choice.
 *
 * `sub * sa * sb` IS AN ORDER AND NOT AN EXPRESSION. Float multiplication is commutative and not
 * associative, so (sub*sa)*sb and (sub*sb)*sa differ by an ulp on some operands -- which is enough
 * to move one fp8 code, and a fused op being byte-identical to the pair it replaces is the whole
 * claim such a fold rests on. Every caller goes through here so that there is one order.
 *
 * The loads go through `ld_dt` rather than `Cvt<DT>` so that one function serves every dtype pair.
 * The two return the same float for every dtype, so a caller that dispatches on the type at
 * compile time for speed still computes these bits.
 */
static inline float quant_dot(const Mat& A, const Mat& B, const Scales& SA, const Scales& SB,
                              int64_t m, int64_t n, int64_t K, int wref_b_in = -1) {
    const int64_t g = SB.group > 0 ? SB.group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* No format in this project gives the activation a finer group than the weight; where one
     * appears the scale cannot be hoisted out of the group and the sub-sum is formed per element. */
    const bool coarse_a = !SA.p || SA.group >= g;
    const uint32_t da = A.t->dtype, db = B.t->dtype;
    const int64_t a0 = m * A.rs, b0 = n * B.rs;
    /* A folded mxfp4 weight (`b_ref` attached) is NOT a pure per-block scale: the device folds
     * 2^-dsh into each ELEMENT as an E4M3 magnitude and applies 2^(Wref-127) once per row. Hoist the
     * row reference out of the group loop and fold the codes per group. */
    const bool folded = SB.ref != nullptr && db == RAD_FP4E2M1;
    const int wref_b = folded ? (wref_b_in >= 0 ? wref_b_in : SB.wref_at(n)) : 0;
    const float rowfac = folded ? SB.row_factor(wref_b) : 1.0f;
    float acc = 0.0f;
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
        const float sb = SB.p ? (folded ? rowfac : SB.grp(n, gi, wref_b)) : 1.0f;
        const int dsh = folded ? SB.group_dsh(n, gi, wref_b) : 0;
        float sub = 0.0f;
        if (coarse_a) {
            for (int64_t k = k0; k < k1; ++k) {
                float bv = ld_dt(db, B.t->data, b0 + k * B.cs);
                if (folded) bv = SB.fold_mag(bv, dsh);
                sub += ld_dt(da, A.t->data, a0 + k * A.cs) * bv;
            }
            acc += sub * (SA.p ? SA.at(m, k0) : 1.0f) * sb;
        } else {
            for (int64_t k = k0; k < k1; ++k) {
                float bv = ld_dt(db, B.t->data, b0 + k * B.cs);
                if (folded) bv = SB.fold_mag(bv, dsh);
                sub += ld_dt(da, A.t->data, a0 + k * A.cs) * SA.at(m, k) * bv;
            }
            acc += sub * sb;
        }
    }
    return acc;
}

/* ------------------------------------------------------------------ weight tables
 *
 * A RAD_OPD_WTAB operand: `data` is a device (here, host) array of `shape[0]` pointers, one per
 * expert, and `stride[0]` is 0 to say so -- see RAD_WTAB in abi/rad_runtime.h. The STACKED form,
 * a plain rank-3 [n_expert, N, K] tensor with an ordinary leading stride, is accepted too and is
 * what a caller outside the engine hands over: rad-kbench draws one dense plane and the fixture
 * has nowhere to put a pointer array. Both are one expert per call to this, so nothing downstream
 * has to know which arrived.
 *
 * Returns entry `e` as a rank-(rank-1) tensor over the same dtype, or a null-data tensor when the
 * operand cannot be read that way. */
static inline RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank  = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d]  = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        void* const* tab = (void* const*)w->data;
        t.data = tab[e];
    } else {
        /* An element offset in a packed dtype can be mid-byte; the stacked form is only ever
         * whole planes, so the arithmetic is done in bytes off the leading stride. */
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = (int64_t)bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;          /* data stays null: unreadable */
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

}  /* namespace ref */

#endif /* REF_GEMM_H */
