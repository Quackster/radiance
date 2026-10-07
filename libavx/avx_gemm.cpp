/* avx_gemm.cpp -- the GEMM family and the two vocabulary edges.
 *
 *     nt: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * B is stored TRANSPOSED because that is the layout every weight in a transformer already has, so
 * both operands are k-contiguous and the inner loop is a DOT PRODUCT over k rather than an outer
 * product over n. That single fact decides the whole shape of this file, and it is worth being
 * explicit about what it costs, because the textbook GEMM does the opposite.
 *
 * ============================== WHY DOT PRODUCT AND NOT OUTER PRODUCT ==============================
 *
 * The classical high-performance GEMM (Goto's, and every BLAS since) packs B into panels laid out
 * so the inner loop broadcasts one A element against a VECTOR of n, accumulating C[m, n0:n0+w] in
 * registers with no horizontal reduction anywhere. It is the right shape when you may choose B's
 * layout.
 *
 * Here B's layout is given, it is k-contiguous, and a transformer weight is READ ONCE PER LAUNCH --
 * there is no second GEMM against the same weight to amortise a pack over. Packing B into n-major
 * panels would mean touching the largest operand twice to save horizontal reductions that only
 * happen once per output element. At decode (M = 1) the whole GEMM is a matvec: the weight is
 * streamed exactly once, the arithmetic intensity is 2 flops a byte, and nothing about the
 * instruction mix matters because the memory system is the wall. At prefill the register blocking
 * below recovers most of what a packed form would give.
 *
 * So: REGISTER-BLOCKED DOT PRODUCT. MR rows of A against NR rows of B, MR*NR vector accumulators
 * live across the whole k loop, MR + NR vector loads per k step, and the horizontal reduction paid
 * once per output element rather than once per k step.
 *
 * ============================== THE BLOCK SIZE IS THE REGISTER FILE, AND IT WAS SWEPT ==============================
 *
 * MEASURED, gemm_nt bf16 at M=512, N=2048, K=2048, one thread, on the part libavx/README.md names:
 *
 *   AVX-512, 32 ZMM         4x4  23.2 ms      4x6  22.1 ms  <-- chosen
 *                           6x4  24.2 ms      6x6  24.0 ms
 *                           8x4  23.8 ms      8x2  35.2 ms      12x2  34.1 ms
 *   AVX2/AVX, 16 YMM        3x3  34.4 ms      2x6  30.7 ms  <-- chosen
 *                           2x5  31.9 ms      2x4  34.8 ms      2x3  40.5 ms
 *                           3x4  58.4 ms      4x3  58.1 ms
 *
 * TWO THINGS IN THAT TABLE ARE WORTH MORE THAN THE CHOICE. The first is the cliff: 3x4 and 4x3 at
 * AVX2 are 58 ms against 31-34 for everything around them, because 12 accumulators plus 3 or 4 A
 * vectors plus one B vector is 16 or 17 live in a 16-register file -- the spill lands INSIDE the k
 * loop and costs 1.9x. That is why this is written out per level rather than derived from a
 * formula, and why the numbers sit next to the code.
 *
 * The second is that WIDER IN N BEAT WIDER IN M at both levels, at equal arithmetic intensity:
 * 2x6 and 3x3 both load 8 vectors for 12 and 9 FMAs respectively -- intensity 1.5 either way -- and
 * 2x6 is 11% faster. The B vector is loaded once and reused across the MR inner iterations, so a
 * small MR keeps fewer values live and schedules better. Nothing about the flop count predicts
 * this; it came out of the sweep.
 *
 * scalar stays 2x2, where "vector" is one float and the blocking only helps the load/store
 * scheduler.
 *
 * ============================== ONE MICROKERNEL, NOT TWENTY-EIGHT ==============================
 *
 * libref templates its inner loop on (A dtype, B dtype) and selects through a function pointer,
 * because for it the per-element dtype switch WOULD be the cost -- one element is read per FLOP.
 * That is 16 instantiations for the plain family and 28 for the quantised one.
 *
 * This plugin instead PACKS TO f32 and has ONE microkernel:
 *
 *     A is widened once into a contiguous [M, K] f32 buffer   -- M*K conversions
 *     each B row is widened once, per n-block, into a thread-local buffer  -- N*K conversions
 *
 * The conversion count is the same as the fused form's (every element of both operands is
 * converted exactly once either way); what it buys is that W4, W2, MXFP4, fp8 and i8 all arrive at
 * the same code, so the packed-nibble unpack is written once (avx_support.cpp) and the microkernel
 * never sees a dtype. 28 instantiations x 4 ISA levels of hand-written vector code is not a
 * maintainable object, and the ones nobody runs would be the ones that are wrong.
 *
 * What it costs is an L1 round trip: the packed rows are written and read back. Against an operand
 * that came from L2 or DRAM -- which a weight always did -- that is not measurable, and the
 * exception is stated where it happens (an f32 contiguous A is used in place, unpacked).
 *
 * THE MEMORY THIS COSTS IS BOUNDED BY THE M BLOCK below -- the pack is MB*K floats, not M*K -- and
 * it is not blocked over K. That is a deliberate omission: K-blocking would bound the pack further
 * but would force the C accumulator into memory, because the k reduction could then no longer live
 * in registers across the whole launch. C is M*N floats, larger than the pack at every shape this
 * plugin sees, and another pass over it.
 *
 * ============================== PARALLELISM IS OVER N ==============================
 *
 * A decode step is M = 1 and a verify step is M <= 64, so an M-major split leaves all but one core
 * idle on exactly the shape the engine spends its life in. N is thousands in every GEMM a
 * transformer issues. libref made the same choice for the same reason. Each thread owns whole
 * output elements, so the summation order does not depend on the thread count and two runs at
 * different loads are bit-identical.
 */
#include "avx_vec.h"
#include "avx_common.h"
#include "avx_rows.h"

#include <vector>

using namespace avx;

#include "avx_had.h"

/* ================================================================== register blocking */
#if   AVX_LEVEL == 3
constexpr int64_t MR = 4, NR = 6;
#elif AVX_LEVEL == 2 || AVX_LEVEL == 1
constexpr int64_t MR = 2, NR = 6;
#else
constexpr int64_t MR = 2, NR = 2;
#endif

/* MR x NR dot products over [k0, k1), accumulated into `out` (which is ADDED to, not overwritten,
 * so the grouped GEMM can call this per group).
 *
 * The tail is summed separately and added to the horizontal reduction rather than folded into a
 * masked vector: the two give different orders and the masked form is only free at AVX-512. One
 * order, four levels. */
template <int MB_, int NB_>
static inline void dot_block(const float* A, int64_t lda, const float* B, int64_t ldb,
                             int64_t k0, int64_t k1, float out[MB_][NB_]) {
    vf acc[MB_][NB_];
    for (int i = 0; i < MB_; ++i)
        for (int j = 0; j < NB_; ++j) acc[i][j] = vf_zero();

    int64_t k = k0;
    for (; k + VF_N <= k1; k += VF_N) {
        vf av[MB_];
        for (int i = 0; i < MB_; ++i) av[i] = vf_loadu(A + i * lda + k);
        for (int j = 0; j < NB_; ++j) {
            const vf bv = vf_loadu(B + j * ldb + k);
            for (int i = 0; i < MB_; ++i) acc[i][j] = vf_fma(av[i], bv, acc[i][j]);
        }
    }
    for (int i = 0; i < MB_; ++i)
        for (int j = 0; j < NB_; ++j) {
            float s = vf_hsum(acc[i][j]);
            for (int64_t t = k; t < k1; ++t) s += A[i * lda + t] * B[j * ldb + t];
            out[i][j] += s;
        }
}

/* One dot product, for the ragged edge of the M and N loops. Four accumulators, which is what
 * covers the FMA latency; the edge is at most MR-1 by NR-1 blocks of a launch, so its cost is
 * noise, but writing it as a scalar loop would make a narrow GEMM (N < NR) run scalar throughout
 * and a draft head's lm_head is exactly that shape. */
static inline float dot1(const float* a, const float* b, int64_t k0, int64_t k1) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t k = k0;
    for (; k + 4 * VF_N <= k1; k += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
        a1 = vf_fma(vf_loadu(a + k + VF_N), vf_loadu(b + k + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + k + 2 * VF_N), vf_loadu(b + k + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + k + 3 * VF_N), vf_loadu(b + k + 3 * VF_N), a3);
    }
    for (; k + VF_N <= k1; k += VF_N) a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; k < k1; ++k) s += a[k] * b[k];
    return s;
}

/* ================================================================== operand readings
 * libref/ref_gemm.h's, and they are copied for the same reason avx_common.h copies the addressing
 * helpers: they are the CONTRACT for which bytes an operand's (row, col) lives at, not a
 * convenience. `row_stride` (avx_common.h) in particular is the one that makes an oracle wrong
 * when it is wrong -- a rank-3 [tokens, heads, width] activation read as [tokens, heads*width] has
 * a row stride that is the product of the trailing dimensions and NOT stride[rank-2], and taking
 * the head stride instead lands row 0 correctly and starts every row after it `width` elements
 * in. */

struct Mat { const RadTensor* t; int64_t rs, cs; };

static inline Mat as_mat(const RadTensor* t, int64_t cols) {
    Mat m{ t, cols, 1 };
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

/* A quantised operand's scale stream. libref/ref_gemm.h is the definition, including the `rblk`
 * rule that makes a DeepSeek-style [ceil(N/128)][ceil(K/128)] block grid readable: without it a
 * 17408-row weight's 136-row scale plane is read as if it had 17408 rows and lands on groups = 0,
 * which is a plausible wrong number rather than a diagnosable failure. */
struct Scales {
    const void* p;
    uint32_t dt;
    int64_t rs, cs, group, rblk;
    bool e8m0;
    /* The OCP-MX `b_ref` reference, exactly libref/ref_gemm.h's: attached only for an mxfp4 weight
     * whose e8m0 plane has one, it makes a folded product reproduce the device's element fold --
     * E4M3_RNE(code * 2^-dsh) per element and 2^(Wref-127) once per row -- instead of reading
     * 2^(E8M0-127). Without it the direct read stands, so every fp8 and b_ref-less path is
     * unchanged. */
    const uint8_t* ref;
    int64_t ref_rs, ref_cs, ref_n;
    inline int wref_at(int64_t row) const {
        int w = 0;
        for (int64_t b = 0; b < ref_n; ++b) {
            const int v = (int)ref[row * ref_rs + b * ref_cs];
            if (v > w) w = v;
        }
        return w;
    }
    inline int dsh_at(int64_t row, int64_t i, int wref) const {
        int d = wref - (int)((const uint8_t*)p)[i];
        return d < 0 ? 0 : (d > 15 ? 15 : d);
    }
    /* The device's folded E4M3 magnitude of one code (see libref/ref_gemm.h::fold_mag): E4M3_RNE
     * of code * 2^-dsh, underflowing to zero, with the row factor applied once by the caller. */
    inline float fold_mag(float code, int dsh) const {
        return rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(code * std::ldexp(1.0f, -dsh)));
    }
    inline float row_factor(int wref) const { return std::ldexp(1.0f, wref - 127); }
    inline int group_dsh(int64_t row, int64_t g, int wref) const {
        const int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        return dsh_at(row, i, wref);
    }
    inline float grp(int64_t row, int64_t g, int wref) const {
        const int64_t i = (rblk > 1 ? row / rblk : row) * rs + g * cs;
        if (e8m0) {
            const uint8_t e = ((const uint8_t*)p)[i];
            if (ref) {
                const uint8_t we = (uint8_t)(wref - dsh_at(row, i, wref));
                return we == 0xff ? std::nanf("") : std::ldexp(1.0f, (int)we - 127);
            }
            return e == 0xff ? std::nanf("") : std::ldexp(1.0f, (int)e - 127);
        }
        return rad_load_f32(p, dt, i);
    }
    inline float grp(int64_t row, int64_t g) const {
        return grp(row, g, ref ? wref_at(row) : 0);
    }
};

static Scales make_scales(const RadTensor* s, int64_t rows, int64_t K, bool mxfp4_weight) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p = s->data;
    q.dt = s->dtype;
    /* libref's reading, block for block: a rank-2 stream with fewer rows than the operand is a
     * block grid whatever the remainder, and a flat stream shorter than the row count is one
     * scale per block of rows rather than a per-row stream to read past its end. */
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs = s->stride[s->rank - 2];
        q.cs = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) { groups = nel / rows; q.rs = groups; }
        else             { groups = 1; q.rblk = grid_block(rows, nel); q.rs = 1; }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    q.e8m0 = mxfp4_weight && (s->dtype == RAD_U8);
    return q;
}

/* Attach an OCP-MX `b_ref` (per-output-row E8M0 reference byte) to a weight's e8m0 grid, mirroring
 * libref/ref_gemm.h's `scales_ref` so the host pair stays clamp-aware together. Untouched unless the
 * weight is mxfp4, its b_scale is e8m0, and the reference is a u8 stream at least N wide. */
static void scales_ref(Scales& s, const RadTensor* r, int64_t rows) {
    if (!s.e8m0 || !s.p || !r || !r->data || r->dtype != RAD_U8) return;
    const int64_t nel = numel(r);
    if (rows <= 0 || nel < rows) return;
    s.ref    = (const uint8_t*)r->data;
    s.ref_rs = r->rank >= 2 ? r->stride[r->rank - 2] : 1;
    s.ref_cs = r->rank >= 2 ? r->stride[r->rank - 1] : 1;
    const int64_t inner = r->rank >= 2 ? r->shape[r->rank - 1] : 1;
    s.ref_n  = (inner > 1 && nel >= rows * inner) ? inner : 1;
    if (s.ref_rs <= 0) s.ref_rs = 1;
    if (s.ref_cs <= 0) s.ref_cs = 1;
}

/* ================================================================== packing
 * A row of an operand, widened into f32 and made contiguous. Returns the caller's own pointer when
 * the operand is ALREADY dense f32 -- the one case where the pack is skipped entirely, and the
 * reason a f32 GEMM does not pay for the generality. */
static const float* pack_rows(const Mat& m, int64_t r0, int64_t nrows, int64_t K, float* dst,
                              int64_t ldd) {
    const RadTensor* t = m.t;
    if (m.cs == 1 && t->dtype == RAD_F32 && m.rs == ldd)
        return (const float*)t->data + r0 * m.rs;
    for (int64_t r = 0; r < nrows; ++r) {
        const int64_t base = (r0 + r) * m.rs;
        if (m.cs == 1) {
            /* Byte-alignment mask: I4/e2m1 need an even element, I2 and e2m3 a multiple of four
             * (a 6-bit code is at a byte boundary exactly when 6i % 8 == 0). A misaligned start
             * goes to the scalar gather rather than reading the row bit-shifted. */
            const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                          : (t->dtype == RAD_I2 || t->dtype == RAD_FP6E2M3) ? 3
                          : (t->dtype == RAD_I5) ? 7 : 0;
            if (!sub || (base & sub) == 0) {
                row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), dst + r * ldd, K);
                continue;
            }
        }
        for (int64_t k = 0; k < K; ++k)
            dst[r * ldd + k] = ld_elem(t->data, t->dtype, base + k * m.cs);
    }
    return dst;
}

/* ================================================================== the driver
 *
 * `SB`/`SA` null means the unquantised family, which is the same loop with one group of width K
 * and both scales 1 -- so there is one driver and not two. The grouped form accumulates the
 * sub-sum from the RAW CODES and scales once at the end of the group, which is what an int8 WMMA
 * kernel does and what libref does; folding the scales into every product instead would be a
 * different number in the last bits and the checker would blame the kernel for the choice. */
static int gemm_drive(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                      const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                      int64_t K) {
    const int64_t ys_r = row_stride(Y, N);
    const int64_t ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const int64_t g = (SB && SB->p && SB->group > 0) ? SB->group : K;
    const int64_t ngrp = (K + g - 1) / g;
    /* An activation grid FINER than the weight's cannot have its scale hoisted out of the group.
     * No format in this project has one; the branch exists so that a format that did would be
     * refused rather than silently mis-scaled. */
    const bool coarse_a = !SA || !SA->p || SA->group >= g;
    if (!coarse_a) return RAD_E_UNSUPPORTED;

    /* ================================================== THE M BLOCK, AND A NULL RESULT
     *
     * The loop nest below is (M block) x (N block) x (K). With the N loop outermost and the whole
     * of A packed, every one of the N/NR n-blocks walks the ENTIRE packed A, so at M=512, K=2048
     * that is a 4 MiB panel re-read 341 times: 1.4 GiB of traffic to do 4.3 GFLOP. Blocking M so
     * the panel stays resident is the textbook fix, and the traffic argument predicts a large win.
     *
     * MEASURED, AND THE PREDICTION DOES NOT HOLD HERE. gemm_nt bf16 at M=512, N=2048, K=2048,
     * AVX-512, one thread, sweeping the panel target:
     *
     *       128 KiB  26.0 ms        4 MiB  25.4 ms
     *       256 KiB  23.4 ms       16 MiB  25.4 ms
     *         1 MiB  24.0 ms      256 MiB  25.0 ms   (= no blocking at this shape)
     *
     * Six percent, not the several-fold the traffic argument implies -- and at M=64 it is WORSE
     * (2.85 ms unblocked against 2.91 at 256 KiB), because there the block splits a batch that did
     * not need splitting and B gets packed twice.
     *
     * THE REASON IS THE PART. The measurement above comes from one with 128 MiB of L3, where the
     * 4 MiB A panel never leaves cache however many times it is walked -- so the traffic the
     * blocking removes is served at L3 bandwidth and not DRAM's. On a part with a 32 MiB L3
     * the same argument still holds; on one with 8 MiB the blocking is the several-fold win the
     * theory predicts. It is kept because it is never much worse and is much better on hardware
     * the measurement above cannot represent -- but it is recorded here as a SIX PERCENT effect on
     * a large-cache part, and not as the fix it looks like.
     *
     * THE BLOCK SIZE IS COMPUTED FROM K, NOT TUNED: the panel is MB x K floats against a fixed
     * byte target. There is nothing to search -- unlike a GPU tile it does not interact with an
     * occupancy limit -- which is why avx_registry.cpp declares no tunable axes. The target is
     * stated rather than probed from CPUID because the cache descriptors there are a fiction on
     * more parts than they are a fact, and the sweep above shows the curve is flat enough that
     * being wrong about it costs ten percent rather than an order. The paneling below was replaced
     * by a whole-A pack with n-blocks outer so B is folded once (see the fold pre-pass); the
     * measurement above is kept as the reason the panel shape was never a large factor. */

    const int64_t lda = K;
    /* Pack the WHOLE A once and loop n-blocks OUTER, so B is packed and (when folded) folded once
     * per n-block instead of once per M-block. The per-element E4M3 fold is O(N*K), so re-running it
     * inside the paneled A loop multiplied it by ceil(M/MB). */
    float* apack = (float*)scratch_raw((size_t)M * (size_t)K * sizeof(float) + 64);
    /* Packed on the calling thread, before the parallel region: every worker reads it, and nothing
     * inside the region touches this arena, so the pointer cannot move under them. */
    const float* Ap = pack_rows(ma, 0, M, K, apack, lda);
    const int64_t nblocks = (N + NR - 1) / NR;

    AVX_PARALLEL_FOR_IF((int64_t)M * N * K >= 65536)
    for (int64_t nb = 0; nb < nblocks; ++nb) {
        const int64_t n0 = nb * NR;
        const int64_t nn = (N - n0) < NR ? (N - n0) : NR;
        const int64_t ldb = K;
        float* bbuf = scratch_f32((size_t)NR * (size_t)K);
        const float* Bp = pack_rows(mb, n0, nn, K, bbuf, ldb);

        /* A folded mxfp4 weight (`b_ref` attached): fold 2^-dsh into each B element as an E4M3
         * magnitude -- libref's per-element fold -- and apply 2^(Wref-127) once per row as the
         * group scale. Only the packed branch reaches here (an e2m1 weight is never dense f32),
         * so `bbuf` holds the codes. */
        const bool folded = SB && SB->ref && mb.t->dtype == RAD_FP4E2M1;
        float brf[NR];
        if (folded) {
            float* bm = bbuf;
            for (int64_t j = 0; j < nn; ++j) {
                const int wref = SB->wref_at(n0 + j);
                brf[j] = SB->row_factor(wref);
                for (int64_t gi = 0; gi < ngrp; ++gi) {
                    const int64_t kk0 = gi * g, kk1 = (kk0 + g < K) ? kk0 + g : K;
                    const int dsh = SB->group_dsh(n0 + j, gi, wref);
                    for (int64_t k = kk0; k < kk1; ++k)
                        bm[j * ldb + k] = SB->fold_mag(bm[j * ldb + k], dsh);
                }
            }
        }

        for (int64_t m0 = 0; m0 < M; m0 += MR) {
            const int64_t mm = (M - m0) < MR ? (M - m0) : MR;
            float acc[MR][NR];
            for (int i = 0; i < MR; ++i)
                for (int j = 0; j < NR; ++j) acc[i][j] = 0.0f;

            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < K) ? k0 + g : K;
                if (mm == MR && nn == NR) {
                    float sub[MR][NR];
                    for (int i = 0; i < MR; ++i)
                        for (int j = 0; j < NR; ++j) sub[i][j] = 0.0f;
                    dot_block<MR, NR>(Ap + m0 * lda, lda, Bp, ldb, k0, k1, sub);
                    for (int j = 0; j < NR; ++j) {
                        const float sb = (SB && SB->p)
                                             ? (folded ? brf[j] : SB->grp(n0 + j, gi)) : 1.0f;
                        for (int i = 0; i < MR; ++i) {
                            const float sa = (SA && SA->p)
                                                 ? SA->grp(m0 + i, k0 / SA->group) : 1.0f;
                            acc[i][j] += sub[i][j] * sa * sb;
                        }
                    }
                } else {
                    for (int64_t j = 0; j < nn; ++j) {
                        const float sb = (SB && SB->p)
                                             ? (folded ? brf[j] : SB->grp(n0 + j, gi)) : 1.0f;
                        for (int64_t i = 0; i < mm; ++i) {
                            const float sa = (SA && SA->p)
                                                 ? SA->grp(m0 + i, k0 / SA->group) : 1.0f;
                            acc[i][j] +=
                                dot1(Ap + (m0 + i) * lda, Bp + j * ldb, k0, k1) * sa * sb;
                        }
                    }
                }
            }
            for (int64_t i = 0; i < mm; ++i)
                for (int64_t j = 0; j < nn; ++j) {
                    float v = acc[i][j];
                    if (bias) v += ldt(bias, (n0 + j) * laststride(bias));
                    stt(Y, (m0 + i) * ys_r + (n0 + j) * ys_c, v);
                }
        }
    }
    return RAD_OK;
}

/* Shared entry for gemm_nt / gemm_nt_bias / logits_gemm. `nkey`/`kkey` name which parameters carry
 * N and K, because logits_gemm spells them n_vocab and n_embd. */
/* gemm_nt_bias's epilogue (docs/OPS.md): an activation of the stored product, then a residual,
 * each step read back from and stored to the output's dtype -- the unfused chain, as libref has it. */
static void gemm_epilogue(const RadTensor* Y, const RadTensor* res, int act, int64_t M, int64_t N) {
    const int64_t ys = laststride(Y), yr = Y->rank >= 2 ? Y->stride[Y->rank - 2] : N;
    const int64_t rs = res ? laststride(res) : 0, rr = res && res->rank >= 2 ? res->stride[res->rank - 2] : N;
    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            const int64_t o = m * yr + n * ys;
            float v = ldt(Y, o);
            if (act == 1) { v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f)); stt(Y, o, v); }
            else if (act == 2) {
                v = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
                stt(Y, o, v);
            }
            if (res) stt(Y, o, ldt(res, m * rr + n * rs) + ldt(Y, o));
        }
}

static int gemm_nt_impl(const RadArgs* a, bool bias_operand, const char* nkey, const char* kkey) {
    /* gemm_nt_bias is a, b, bias, res?, y; the other two are a, b, y. */
    const int nt = bias_operand ? 5 : 3;
    if (a->n_t < nt || !a->t[0].data || !a->t[1].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[1], *Y = &a->t[nt - 1];
    const RadTensor* bias = bias_operand ? rad_arg_in(a, 2) : nullptr;
    const RadTensor* res  = bias_operand ? rad_arg_in(a, 3) : nullptr;
    static const char* kActs[] = { "none", "gelu", "gelu_tanh" };
    const int act = bias_operand ? p_enum(a, "act", kActs, 3, 0) : 0;

    const int64_t K = p_int(a, kkey, B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, nkey, B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    if (res && numel(res) < M * N) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K), mb = as_mat(B, K);
    const int st = gemm_drive(ma, mb, nullptr, nullptr, Y, bias, M, N, K);
    if (st < 0) return st;
    if (act || res) gemm_epilogue(Y, res, act, M, N);
    return RAD_OK;
}

static int gemm_ntq_impl(const RadArgs* a, bool bias_operand) {
    const int nt = bias_operand ? 6 : 5;
    if (a->n_t < nt || !a->t[0].data || !a->t[2].data || !a->t[nt - 1].data) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *Y = &a->t[nt - 1];
    const RadTensor* as_ = rad_arg_in(a, 1);
    const RadTensor* bs_ = rad_arg_in(a, 3);
    const RadTensor* bias = bias_operand ? rad_arg_in(a, 4) : nullptr;

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / N;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (bias && numel(bias) < N) return RAD_E_SHAPE;

    /* OCP-MX weights -- e2m1, e2m3 or an e4m3 plane with an e8m0 grid -- read b_scale as exponent
     * bytes; like libref's, the flag only bites when the stream is u8, so block-fp8's bf16
     * weight_scale_inv is unaffected. The e2m3/e4m3 codes are self-scaling and carry no b_ref, so
     * the folded path below stays keyed to RAD_FP4E2M1. Gated ops stay fp4-only this phase. */
    const bool mx = B->dtype == RAD_FP4E2M1 || B->dtype == RAD_FP6E2M3 || B->dtype == RAD_F8E4M3;
    Scales SA = make_scales(as_, M, K, false);
    Scales SB = make_scales(bs_, N, K, mx);
    /* The OCP-MX fold's reference exponent is gemm_nt_q's operand 6; the bias form has no such slot. */
    if (!bias_operand) scales_ref(SB, rad_arg_in(a, 6), N);
    /* `group` is what the SELECTOR matches on; the loop uses the group the scale stream actually
     * implies. They agree in every well-formed call, and where they do not the stream is the one
     * that describes the bytes -- so a disagreement is refused rather than silently resolved. */
    const int64_t gp = p_int(a, "group", 0);
    if (gp > 0 && SB.p && SB.group != gp) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K), mb = as_mat(B, K);
    return gemm_drive(ma, mb, &SA, &SB, Y, bias, M, N, K);
}

AVX_KERNEL(gemm_nt)        { return gemm_nt_impl(a, false, "N", "K"); }
AVX_KERNEL(gemm_nt_bias)   { return gemm_nt_impl(a, true,  "N", "K"); }
AVX_KERNEL(gemm_nt_q)      { return gemm_ntq_impl(a, false); }
AVX_KERNEL(gemm_nt_q_bias) { return gemm_ntq_impl(a, true); }

/* ------------------------------------------------------------------ gemm_nt_q_gated */
/* gemm_nt_q over a [N, K] gate_up plane -- N is the WEIGHT's rows, twice the fused width, gate half
 * first -- with gated_quant_fp8 as the epilogue; the output is [M, N/2].
 *
 * BYTE-IDENTICAL TO THE PAIR IT STANDS FOR, gemm_nt_q into a bf16 [M, N] tensor and then
 * gated_quant_fp8 over it. The product is gemm_drive's, the same driver at the same blocking; both
 * halves are narrowed to bf16 by the ABI's own converter, which is what gemm_nt_q's element store
 * does to a bf16 output; and the gate and the codes are avx_had.h's, which gated_quant_fp8 runs.
 * Taking either half off the f32 accumulator instead would be a different function.
 *
 * The f32 plane between the two is materialised whole, in a buffer the calling thread keeps: the
 * driver parallelises over the plane's columns, and the epilogue over its rows. `act` is REQUIRED
 * here where the quantiser defaults it -- an implementation may compile one gate, and a default
 * would let it resolve for the other. */
AVX_KERNEL(gemm_nt_q_gated) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *A = &a->t[0], *B = &a->t[2], *q = &a->t[4], *s = &a->t[5];
    const RadTensor* as_ = rad_arg_in(a, 1);
    const RadTensor* bs_ = rad_arg_in(a, 3);
    const RadTensor* ob  = rad_arg_in(a, 6);
    if (!A->data || !B->data || !q->data || !s->data) return RAD_E_INVAL;

    const int64_t K = p_int(a, "K", B->rank >= 2 ? B->shape[B->rank - 1] : 0);
    const int64_t N = p_int(a, "N", B->rank >= 2 ? B->shape[B->rank - 2] : 0);
    if (K <= 0 || N <= 0 || (N & 1)) return RAD_E_SHAPE;
    const int64_t n = N / 2;
    static const char* kActs[] = { "silu", "sigmoid" };
    const int act = p_enum(a, "act", kActs, 2, -1);
    if (act != 0 && act != 1) return RAD_E_INVAL;
    int64_t g = p_int(a, "group", 0);
    if (g <= 0 || g > n) g = n;
    const int64_t ngrp = (n + g - 1) / g;
    const int64_t M = numel(q) / n;
    if (M <= 0 || numel(A) < M * K || numel(B) < N * K) return RAD_E_SHAPE;
    if (numel(s) < M * ngrp) return RAD_E_SHAPE;
    if (ob && numel(ob) < M * n) return RAD_E_SHAPE;

    const bool mx = B->dtype == RAD_FP4E2M1;
    Scales SA = make_scales(as_, M, K, false);
    Scales SB = make_scales(bs_, N, K, mx);
    /* gemm_nt_q_gated_mx appends b_ref at operand 7; the fp8 form has none. */
    scales_ref(SB, rad_arg_in(a, 7), N);
    const Mat ma = as_mat(A, K), mb = as_mat(B, K);

    static thread_local std::vector<float> plane;
    if (plane.size() < (size_t)(M * N)) plane.resize((size_t)(M * N));
    RadTensor Y{};
    Y.data = plane.data();
    Y.dtype = RAD_F32;
    Y.rank = 2;
    Y.shape[0] = M; Y.shape[1] = N;
    Y.stride[0] = N; Y.stride[1] = 1;
    const int rc = gemm_drive(ma, mb, &SA, &SB, &Y, nullptr, M, N, K);
    if (rc != RAD_OK) return rc;

    const float* P = plane.data();
    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* row = scratch_f32((size_t)N);
        float* out = scratch_f32_b((size_t)n);
        float* tmp = scratch_f32_c((size_t)n);
        for (int64_t j = 0; j < N; ++j) row[j] = rad_bf16_to_f32(rad_f32_to_bf16(P[m * N + j]));
        gate_bf16(row, row + n, out, n, act, tmp);
        fp8_emit(out, tmp, n, g, q, s, ob, m);
    }
    return RAD_OK;
}

/* The OCP-MX (mxfp4a8) gated fold, a separate op for the reason libr4d's new op gives: the MX
 * weight carries an eighth `b_ref` operand, and the positional operand list has an exact arity
 * check, so it cannot ride on the seven-operand op. The arithmetic is gemm_nt_q_gated's -- the same
 * driver, the same bf16 narrowing of both halves, the same whole-row quantiser when `group` is
 * absent -- and the shared implementation attaches operand 7 (`b_ref`) through `scales_ref`, so the
 * MX block scale is the device's saturated fold 2^(Wref-127-dsh), as libref's is. */
AVX_KERNEL(gemm_nt_q_gated_mx) { return AVX_FN(avx_gemm_nt_q_gated)(a, stream); }

/* The same arithmetic as gemm_nt, and a separate op anyway: lm_head is its own access class
 * (RAD_ACCESS_VOCAB) and the sampler wants to fuse against it, so the two have to be selectable
 * apart even when one kernel serves both. N and K are spelled n_vocab and n_embd. */
AVX_KERNEL(logits_gemm) { return gemm_nt_impl(a, false, "n_vocab", "n_embd"); }

/* ================================================================== the vocabulary edges
 *
 * A gather, not a GEMM. A NEGATIVE token id writes a ZERO ROW -- the padding convention a batch
 * with a short sequence needs. `vocab_offset` makes the table this rank's slice of a vocab-parallel
 * embedding: ids stay GLOBAL, one outside the slice writes a zero row, and summing the ranks
 * reconstructs x.
 *
 * WITHOUT AN OFFSET AN ID AT OR ABOVE n_vocab STILL FAILS THE LAUNCH, because there it is a caller
 * bug rather than another rank's token and a plausible embedding is the worst possible answer.
 * With one, this kernel cannot know the global vocabulary, so the bound is the caller's. */
template <bool QUANT>
static int embed_impl(const RadArgs* a) {
    const int nt = QUANT ? 4 : 3;
    /* embed_lookup_q's scale (operand 2) is optional; everything else must be there. */
    for (int i = 0; i < nt; ++i)
        if (!(QUANT && i == 2) && !rad_arg_in(a, i)) return RAD_E_INVAL;
    const RadTensor* tok = &a->t[0];
    const RadTensor* wte = &a->t[1];
    const RadTensor* sc  = QUANT ? &a->t[2] : nullptr;
    const RadTensor* x   = &a->t[nt - 1];
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    const int64_t voff = p_int(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / n_embd;
    if (M <= 0 || numel(tok) < M || numel(wte) < n_vocab * n_embd) return RAD_E_SHAPE;
    /* An absent scale is one (ref_embed_lookup_q says why the op allows it). */
    const float scale = QUANT && sc->data && numel(sc) >= 1 ? ldt(sc, 0) : 1.0f;

    /* Validated before the loop, because an OpenMP body cannot return. */
    if (voff == 0)
        for (int64_t m = 0; m < M; ++m)
            if (ldi(tok, offlin(tok, m)) >= n_vocab) return RAD_E_INVAL;

    const int64_t ws = wte->rank >= 2 ? wte->stride[wte->rank - 2] : n_embd;
    const int64_t wc = wte->rank >= 2 ? wte->stride[wte->rank - 1] : 1;
    /* A ROW COPY WHEN NOTHING HAS TO CHANGE, which for the unquantised gather is the usual case:
     * the table and the activation are the same dtype and a memcpy moves the row at the width of
     * the machine instead of through an f32 round trip that would also re-round it. */
    const bool copy = !QUANT && wte->dtype == x->dtype && wc == 1 && laststride(x) == 1;
    const size_t rowb = copy ? (size_t)((rad_dtype_bits(x->dtype) * n_embd) / 8) : 0;

    AVX_PARALLEL_FOR_IF(M * n_embd >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t id = ldi(tok, offlin(tok, m)) - voff;
        const int64_t xb = rowoff(x, m, n_embd), xs = laststride(x);
        const bool have_row = id >= 0 && id < n_vocab;
        if (copy) {
            void* dst = (void*)byte_at(x->data, x->dtype, xb);
            if (have_row) std::memcpy(dst, byte_at(wte->data, wte->dtype, id * ws), rowb);
            else          std::memset(dst, 0, rowb);
            continue;
        }
        float* s = scratch_f32((size_t)n_embd);
        if (!have_row) {
            for (int64_t i = 0; i < n_embd; ++i) s[i] = 0.0f;
        } else if (wc == 1) {
            row_to_f32(wte->dtype, byte_at(wte->data, wte->dtype, id * ws), s, n_embd);
            if (QUANT) {
                const vf vs = vf_set1(scale);
                int64_t i = 0;
                for (; i + VF_N <= n_embd; i += VF_N) vf_storeu(s + i, vf_mul(vf_loadu(s + i), vs));
                for (; i < n_embd; ++i) s[i] *= scale;
            }
        } else {
            for (int64_t i = 0; i < n_embd; ++i)
                s[i] = ld_elem(wte->data, wte->dtype, id * ws + i * wc) * scale;
        }
        if (xs == 1) out_row(x, xb, n_embd, s);
        else for (int64_t i = 0; i < n_embd; ++i) stt(x, xb + i * xs, s[i]);
    }
    return RAD_OK;
}

AVX_KERNEL(embed_lookup)   { return embed_impl<false>(a); }

/* ================================================================== the E4M3 table into bf16
 *
 * The form a model serves -- an E4M3 table, a bf16 activation -- is a gather whose rows are usually
 * not in memory, and avx_ngram.cpp is where they come from: a row cache shared by every thread, a
 * direct read for each miss, a reader a step ahead. What stays here is the part that is this
 * level's: turning a row of codes into bf16 through the call's 256-entry table.
 *
 * AT AVX-512 THE TABLE LIVES IN EIGHT REGISTERS. 256 bf16 are 512 bytes; vpermt2w indexes 64 words
 * across a register pair, so four of them look every code up in its quarter and the code's top two
 * bits pick the quarter -- a dozen instructions for 32 codes, against 32 dependent loads. Below
 * AVX-512 the table has no in-register form (vpshufb indexes 16 bytes and a gather issues one load
 * a lane, the null result in README.md), so those levels look a code up at a time.
 *
 * EVERY ROW IS A CACHE MISS, so each row's lines are requested kAhead rows before its decode:
 * a batch is thousands of rows scattered over the row cache, and a decode that waited for each
 * row's lines in turn would spend its time on the memory latency rather than the lookups. */
static void decode_e4m3_rows(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                             int64_t n, const uint16_t* lut) {
    constexpr int64_t kAhead = 8;
#if AVX_LEVEL >= 3
    __m512i t[8];
    for (int k = 0; k < 8; ++k) t[k] = _mm512_loadu_si512((const void*)(lut + 32 * k));
    const __m512i bit6 = _mm512_set1_epi16(0x40), bit7 = _mm512_set1_epi16(0x80);
    auto pick = [&](__m512i ix) {
        const __m512i  q0 = _mm512_permutex2var_epi16(t[0], ix, t[1]);
        const __m512i  q1 = _mm512_permutex2var_epi16(t[2], ix, t[3]);
        const __m512i  q2 = _mm512_permutex2var_epi16(t[4], ix, t[5]);
        const __m512i  q3 = _mm512_permutex2var_epi16(t[6], ix, t[7]);
        const __mmask32 h6 = _mm512_test_epi16_mask(ix, bit6);
        const __mmask32 h7 = _mm512_test_epi16_mask(ix, bit7);
        return _mm512_mask_blend_epi16(h7, _mm512_mask_blend_epi16(h6, q0, q1),
                                       _mm512_mask_blend_epi16(h6, q2, q3));
    };
#endif
    for (int64_t r = 0; r < count; ++r) {
        if (r + kAhead < count) {
            const uintptr_t p = (uintptr_t)src[r + kAhead];
            for (uintptr_t q = p & ~(uintptr_t)63; q < p + (uintptr_t)n; q += 64)
                __builtin_prefetch((const void*)q);
        }
        const uint8_t* s = src[r];
        uint16_t*      d = dst[r];
#if AVX_LEVEL >= 3
        int64_t i = 0;
        for (; i + 32 <= n; i += 32) {
            const __m512i ix = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(s + i)));
            _mm512_storeu_si512((void*)(d + i), pick(ix));
        }
        if (i < n) {
            const __mmask32 k = (__mmask32)((1u << (n - i)) - 1u);
            const __m512i ix = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(k, s + i));
            _mm512_mask_storeu_epi16((void*)(d + i), k, pick(ix));
        }
#else
        for (int64_t i = 0; i < n; ++i) d[i] = lut[s[i]];
#endif
    }
}

/* A bf16 table's rows ARE their output: the decode is a copy, and the row cache in front of it is
 * the whole of the work, as it is for the E4M3 table. */
static void decode_bf16_rows(const uint8_t* const* src, uint16_t* const* dst, int64_t count,
                             int64_t n, const uint16_t*) {
    for (int64_t r = 0; r < count; ++r) std::memcpy(dst[r], src[r], (size_t)n * sizeof(uint16_t));
}

/* The call in avx_ngram.cpp's form, or false: i32 ids, a dense E4M3 table with its scale or a
 * dense bf16 table without one, bf16 rows at one pitch. */
static bool rows_form(const RadArgs* a, RowGather* g) {
    /* The scale is optional, so it is not part of what must be present. */
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 3)) return false;
    const RadTensor* tok = &a->t[0];
    const RadTensor* wte = &a->t[1];
    const RadTensor* sc  = &a->t[2];
    const RadTensor* x   = &a->t[3];
    if (tok->dtype != RAD_I32 || !rad_tensor_is_contiguous(tok)) return false;
    if (!rad_tensor_is_contiguous(wte)) return false;
    const bool e4m3 = wte->dtype == RAD_F8E4M3;
    if (!e4m3 && wte->dtype != RAD_BF16) return false;
    const bool scaled = sc->data && numel(sc) >= 1;
    if (x->dtype != RAD_BF16 || scaled != e4m3) return false;
    const int64_t n_embd = p_int(a, "n_embd", wte->rank >= 2 ? wte->shape[wte->rank - 1] : 0);
    const int64_t n_vocab = p_int(a, "n_vocab", wte->rank >= 2 ? wte->shape[wte->rank - 2] : 0);
    if (n_embd <= 0 || n_vocab <= 0) return false;
    int64_t ld = -1;
    if (rad_tensor_is_contiguous(x))                                        ld = n_embd;
    else if (x->rank == 2 && x->shape[1] == n_embd && x->stride[1] == 1)   ld = x->stride[0];
    if (ld < n_embd) return false;
    g->ids = (const int32_t*)tok->data;
    g->M = numel(x) / n_embd;
    g->tab = (const uint8_t*)wte->data;
    g->n_vocab = n_vocab;
    g->n_embd = n_embd;
    g->voff = p_int(a, "vocab_offset", 0);
    g->scale = e4m3 ? ldt(sc, 0) : 1.0f;
    g->row_bytes = e4m3 ? n_embd : 2 * n_embd;
    g->out = (uint16_t*)x->data;
    g->x_ld = ld;
    const RadTensor* ahd = rad_arg_in(a, 4);
    const bool ahead = ahd && ahd->dtype == RAD_I32 && rad_tensor_is_contiguous(ahd) && numel(ahd);
    g->ahead = ahead ? (const int32_t*)ahd->data : nullptr;
    g->n_ahead = ahead ? numel(ahd) : 0;
    g->rank = a->rank;
    g->world_size = a->world_size;
    return true;
}

AVX_KERNEL(embed_lookup_q) {
    RowGather g;
    if (!rows_form(a, &g)) return embed_impl<true>(a);
    if (g.voff < 0 || g.M <= 0 || numel(&a->t[0]) < g.M || numel(&a->t[1]) < g.n_vocab * g.n_embd)
        return RAD_E_SHAPE;
    /* Checked before the gather, not inside it: a refusal halfway through would leave the output
     * half written and the step would carry on over it. The bound is embed_impl's -- with an offset
     * an id outside this rank's slice is another rank's row and writes a zero one. */
    if (g.voff == 0)
        for (int64_t m = 0; m < g.M; ++m)
            if ((int64_t)g.ids[m] >= g.n_vocab) return RAD_E_INVAL;
    return row_gather(g, g.row_bytes == g.n_embd ? decode_e4m3_rows : decode_bf16_rows);
}
