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

/* THE FRAGMENT-TILED E4M3 ACTIVATION (`afrag` on rmsnorm_quant_fp8, gated_quant_fp8 and, for
 * mxfp4a8, gemm_nt_q): when the op's M >= afrag the codes are stored as 16 (row) x 16 (k) fragments
 * of 256 contiguous bytes, lane l = (k half) * 16 + row, byte j of a lane = k offset j:
 *     off(r, c) = ((r / 16) * (K / 16) + c / 16) * 256 + (((c / 8) % 2) * 16 + r % 16) * 8 + c % 8
 * The arithmetic is unchanged; only where a code lives is. libr4d's writers and its
 * gemm_mxfp4a8_atiled reader agree on this formula (r4d_fused_quant_fp8.hip, quant_group). */
static inline int64_t afrag_off(int64_t r, int64_t c, int64_t K) {
    return ((r >> 4) * (K >> 4) + (c >> 4)) * 256 + (((c >> 3) & 1) * 16 + (r & 15)) * 8 + (c & 7);
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

    inline float at(int64_t row, int64_t k) const {
        int64_t g = group > 0 ? k / group : 0;
        int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return e8m0 ? e8m0_to_f32(((const uint8_t*)p)[i]) : ld_dt(dt, p, i);
    }
    inline float grp(int64_t row, int64_t g) const {
        int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return e8m0 ? e8m0_to_f32(((const uint8_t*)p)[i]) : ld_dt(dt, p, i);
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
    /* mxfp4's block scale is an e8m0 exponent byte. When the stream arrives as u8 beside an mxfp4
     * weight there is nothing else it could be; a float stream beside the same weight is read as
     * a float, so a quantiser that chose to store the scale plainly still works. */
    q.e8m0  = mxfp4_weight && (s->dtype == RAD_U8);
    return q;
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
                              int64_t m, int64_t n, int64_t K) {
    const int64_t g = SB.group > 0 ? SB.group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* No format in this project gives the activation a finer group than the weight; where one
     * appears the scale cannot be hoisted out of the group and the sub-sum is formed per element. */
    const bool coarse_a = !SA.p || SA.group >= g;
    const uint32_t da = A.t->dtype, db = B.t->dtype;
    const int64_t a0 = m * A.rs, b0 = n * B.rs;
    float acc = 0.0f;
    for (int64_t gi = 0; gi < ngrp; ++gi) {
        const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
        const float sb = SB.p ? SB.grp(n, gi) : 1.0f;
        float sub = 0.0f;
        if (coarse_a) {
            for (int64_t k = k0; k < k1; ++k)
                sub += ld_dt(da, A.t->data, a0 + k * A.cs) * ld_dt(db, B.t->data, b0 + k * B.cs);
            acc += sub * (SA.p ? SA.at(m, k0) : 1.0f) * sb;
        } else {
            for (int64_t k = k0; k < k1; ++k)
                sub += ld_dt(da, A.t->data, a0 + k * A.cs) * SA.at(m, k) *
                       ld_dt(db, B.t->data, b0 + k * B.cs);
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
