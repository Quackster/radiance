// r4d_fused_quant_fp8_row.hip -- the per-row (OCP-MX) forms of the two fused producers, plus the
// gated-attention gate form. Same structure as r4d_fused_quant_fp8.hip, with the group=128
// quantiser replaced by quant_act_fp8_row's whole-row one.
//
// WHY THE FUSIONS EXIST is that file's argument and it is unchanged: a decode step issues ~300
// quantisers, one per activation per layer on both ranks, and the row is tiny against a launch. The
// MX trunk reads the SAME activations the fp8 trunk does, so it wants the SAME producers fused --
// dropping them here would hand back the launch this family was built to remove.
//
// THE QUANTISER IS quant_act_fp8_row's, NOT quant_act_fp8's, and that is the one structural
// difference. The MX GEMM's epilogue is `C = acc * 2^(Wref[n]-127) * As[m]`, so the activation
// carries ONE f32 scale per row (`As[m]`) rather than one per (row, 128 columns). The amax is
// therefore a BLOCK reduction over the whole row, not a wave reduction over a 128-group, and the
// kernel is one block a row like quant_act_fp8_row.
//
// BIT-IDENTICAL TO THE PAIR IT REPLACES. The unfused pair rounds the producer's output to bf16,
// writes it, and quant_act_fp8_row reads those bf16 bytes back; the amax and the codes are then
// taken over bf16-ROUNDED values. Every kernel here rounds to bf16 first and quantises the
// round-tripped float, which is the difference between "the same text" and "very nearly". The amax
// is a MAX, which is exact and partition-independent, so the fused block width is free; the SUM OF
// SQUARES is not, so the norm's pass one takes the same partition as rmsnorm_bf16 (see the
// kU-loop note below and r4d_act_threads).
//
// THE NORM RE-READS ITS ROW FOR PASS TWO rather than keeping it in registers or LDS, exactly as the
// group=128 form and r4d_rmsnorm_had_quant_i8 explain: the sum is normalised, not the row, and a
// row is up to 17408 wide so it does not fit LDS at M = 2048. The residual may alias `out_bf16` and
// every caller makes it do so; pass three reads `out_bf16` when it exists (which is where the pair
// it replaces read the norm output from) and recomputes from the source when it does not.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_common.h"
#include "r4d_ar_seq.h"   // the all-reduce sequence step the fused all-reduce + norm takes

namespace {

constexpr uint32_t kWave = 32;
// 448 is E4M3's largest finite value; the scale is a DEQUANT multiplier, the same sense as the
// checkpoint's weight_scale_inv, so the GEMM's fold multiplies both.
constexpr float kTop = 448.0f;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// The sum of squares across a whole row, block-level. NOT partition-free (fp32 addition is not
// associative), so the column order is the one rmsnorm_bf16 takes: `i = tid; i += THR` with the
// kU-wide unrolled stride below. The width itself comes from r4d_act_threads at the launcher.
template <uint32_t THR>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < THR / kWave; ++w) s += lds[w];
    return s;
}

// The whole-row amax. A MAX is exact and order-independent, so this partition is free -- it need
// not agree with quant_act_fp8_row's, only the VALUES it sees must be the same bf16-rounded ones.
template <uint32_t THR>
__device__ __forceinline__ float block_max(float m, float* red) {
    m = r4d_wave_max(m);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) red[wave] = m;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < THR / kWave) ? red[lane] : 0.0f;
        v = r4d_wave_max(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// Four bf16-rounded values -> four E4M3 bytes at a GIVEN inverse scale. The clamp is
// quant_act_fp8_row's and is not defensive: `amax * (1/scale)` is 448 in exact arithmetic but can
// land an ulp above it in float, and v_cvt_pk_fp8_f32 answers an out-of-range input with NaN.
__device__ __forceinline__ void quant4(const float v[4], float inv, uint8_t* __restrict__ qp) {
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -kTop), kTop),
                                         fminf(fmaxf(v[1] * inv, -kTop), kTop), pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -kTop), kTop),
                                         fminf(fmaxf(v[3] * inv, -kTop), kTop), pk, true);
    *(uint32_t*)qp = (uint32_t)pk;
}

// ---------------------------------------------------------------------------- rmsnorm -> quant
// One block a row. Pass one is the sum of squares; pass two computes the norm, stores it and takes
// the whole-row amax of the bf16 value; pass three quantises -- from `out_bf16` where it exists,
// which is where the pair being replaced read it, and by recomputation where it does not.
template <bool WF32, bool RES, uint32_t THR>
__global__ __launch_bounds__(THR) void rmsnorm_quant_fp8_row_kernel(
    const uint16_t* __restrict__ x, uint16_t* res, const void* __restrict__ w,
    uint16_t* y, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    float eps, float wadd, uint32_t x_ld, uint32_t r_ld, uint32_t y_ld, uint32_t q_ld,
    uint32_t s_ld) {
    __shared__ float lds[THR / kWave];
    __shared__ float red[THR / kWave];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
    const uint16_t* __restrict__ nr = RES ? rr : xr;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)row * q_ld;

    // FOUR STRIDES OF THE ROW ARE LOADED BEFORE ANY IS USED, and the sum takes this thread's
    // columns in the same order `i = tid; i += THR` does, which is rmsnorm_bf16's order and the
    // agreement the fusion's bit-identity needs. See r4d_fused_quant_fp8.hip for the reasoning.
    constexpr uint32_t kU = 4;
    float ss = 0.0f;
    for (uint32_t i0 = tid; i0 < n; i0 += kU * THR) {
        uint16_t xa[kU], ra[kU];
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR < n ? i0 + u * THR : i0;
            xa[u] = xr[i];
            if (RES) ra[u] = rr[i];
        }
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR;
            if (i < n) {
                float v;
                if (RES) {
                    const uint16_t h = f2bf(bf2f(ra[u]) + bf2f(xa[u]));
                    rr[i] = h;
                    v = bf2f(h);
                } else {
                    v = bf2f(xa[u]);
                }
                ss += v * v;
            }
        }
    }
    const float rs = rsqrtf(block_sum<THR>(ss, lds) / (float)n + eps);

    // Pass two: the norm, written to `y` and reduced to the whole-row amax of the bf16 value.
    float m = 0.0f;
    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            float g[4];
            uint16_t xb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                g[j] = WF32 ? ((const float*)w)[c0 + j] : bf2f(((const uint16_t*)w)[c0 + j]);
                xb[j] = nr[c0 + j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // THE BF16 ROUND TRIP IS THE POINT: the unfused quantiser reads these bytes.
                const uint16_t h = f2bf(bf2f(xb[j]) * rs * (g[j] + wadd));
                if (yr) yr[c0 + j] = h;
                v[j] = bf2f(h);
            }
        }
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    }
    const float amax = block_max<THR>(m, red);
    const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
    if (tid == 0) s[row * s_ld] = sc;
    const float inv = 1.0f / sc;

    // Pass three: the codes. `out_bf16` holds the norm output the pair being replaced would have
    // quantised; where it is absent, recompute the same bf16 value from the source.
    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            if (yr) {
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = bf2f(yr[c0 + j]);
            } else {
                float g[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    g[j] = WF32 ? ((const float*)w)[c0 + j]
                                : bf2f(((const uint16_t*)w)[c0 + j]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    v[j] = bf2f(f2bf(bf2f(nr[c0 + j]) * rs * (g[j] + wadd)));
            }
        }
        quant4(v, inv, qr + c0);
    }
}

// ---------------------------------------------------------------------------- gate -> quant
// silu(gate) * up, or sigmoid(gate) * up, then the whole-row quantiser. The reference is
// silu_mul_bf16, ONE kernel that keeps the activation in f32 through the product, so `a` below is
// f32 and only the product is rounded.
template <int ACT, uint32_t THR>
__global__ __launch_bounds__(THR) void gated_quant_fp8_row_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t gu_ld, uint32_t y_ld, uint32_t q_ld,
    uint32_t s_ld) {
    __shared__ float red[THR / kWave];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)row * q_ld;

    float m = 0.0f;
    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            uint16_t gb[4], ub[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { gb[j] = gr[c0 + j]; ub[j] = gr[n + c0 + j]; }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g = bf2f(gb[j]);
                const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
                const uint16_t h = f2bf(a * bf2f(ub[j]));
                if (yr) yr[c0 + j] = h;
                v[j] = bf2f(h);
            }
        }
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    }
    const float amax = block_max<THR>(m, red);
    const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
    if (tid == 0) s[row * s_ld] = sc;
    const float inv = 1.0f / sc;

    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            if (yr) {
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = bf2f(yr[c0 + j]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float g = bf2f(gr[c0 + j]);
                    const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
                    v[j] = bf2f(f2bf(a * bf2f(gr[n + c0 + j])));
                }
            }
        }
        quant4(v, inv, qr + c0);
    }
}

// ---------------------------------------------------------- gate . x -> quant, two source buffers
// The gate is a COLUMN SLICE of the [q|gate] projection and `x` is the attention output, two
// buffers with different pitches; `M` is a (token, head) pair. The reference is `sigmoid` then
// `mul` -- TWO kernels with a bf16 buffer between them -- so the ACTIVATION is rounded before the
// multiply, unlike gated_quant_fp8_row whose reference is one kernel. See r4d_fused_quant_fp8.hip.
template <int ACT, uint32_t THR>
__global__ __launch_bounds__(THR) void gate_quant_fp8_row_kernel(
    const uint16_t* __restrict__ g, const uint16_t* __restrict__ x, uint16_t* __restrict__ y,
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n, uint32_t g_ld, uint32_t x_ld,
    uint32_t y_ld, uint32_t q_ld, uint32_t s_ld) {
    __shared__ float red[THR / kWave];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t* __restrict__ gr = g + (size_t)row * g_ld;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)row * q_ld;

    float m = 0.0f;
    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            uint16_t gb[4], xb[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { gb[j] = gr[c0 + j]; xb[j] = xr[c0 + j]; }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float gv = bf2f(gb[j]);
                const uint16_t ab =
                    f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
                const uint16_t h = f2bf(bf2f(ab) * bf2f(xb[j]));
                if (yr) yr[c0 + j] = h;
                v[j] = bf2f(h);
            }
        }
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    }
    const float amax = block_max<THR>(m, red);
    const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
    if (tid == 0) s[row * s_ld] = sc;
    const float inv = 1.0f / sc;

    for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (c0 < n) {
            if (yr) {
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = bf2f(yr[c0 + j]);
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float gv = bf2f(gr[c0 + j]);
                    const uint16_t ab =
                        f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
                    v[j] = bf2f(f2bf(bf2f(ab) * bf2f(xr[c0 + j])));
                }
            }
        }
        quant4(v, inv, qr + c0);
    }
}

// -------------------------------------------------------- all-reduce -> rmsnorm -> per-row quant
//
// The per-row twin of r4d_fused_quant_fp8.hip's ar_rmsnorm_quant_fp8: the same two-rank all-reduce
// folded in front of the norm, with the group=128 quantiser replaced by quant_act_fp8_row's
// whole-row one. The wire, the handshake, the row partition and the reduced bytes in `x` are the
// fp8 sibling's; the norm and the quantiser are rmsnorm_quant_fp8_row's three passes, run here over
// every row a block owns.
//
// The 6-bit wire's arithmetic is duplicated (like r4d_ar_ln_had_quant_i8 explains) rather than
// included from r4d_ar_oneshot_2rank_wht6.h, because that header also carries the standalone
// kernel and its launcher. It is the same group of 64, amax-scaled 6 uniform bits plus a bf16
// scale, at a CHUNK of 512 elements so a chunk never straddles a row.
#define ARQR_GROUP 64
#define ARQR_BITS 6
#define ARQR_EPL 8
#define ARQR_LPG (ARQR_GROUP / ARQR_EPL)                    /* 8 lanes span one group */
#define ARQR_WV 32
#define ARQR_GPW (ARQR_WV * ARQR_EPL / ARQR_GROUP)          /* 4 groups a wave-iteration */
#define ARQR_CG 8                                           /* groups a chunk */
#define ARQR_CE (ARQR_CG * ARQR_GROUP)                      /* 512 elements a chunk */
#define ARQR_ITERS (ARQR_CE / (ARQR_WV * ARQR_EPL))         /* 2 iterations to cover a chunk */
#define ARQR_NWORD (ARQR_ITERS * ARQR_EPL * ARQR_BITS / 32) /* 3 dwords a lane holds for a chunk */
#define ARQR_CBYTES (ARQR_CE * ARQR_BITS / 8)               /* 384 wire bytes a chunk */
#define ARQR_QMAX ((1 << (ARQR_BITS - 1)) - 1)
#define ARQR_QMIN (-(1 << (ARQR_BITS - 1)))
#define ARQR_MASK ((1u << ARQR_BITS) - 1u)
#define ARQR_SPIN_MAX 4000000000ULL

// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`; bit-identical, the sign flip is one XOR.
template <int M>
__device__ __forceinline__ void arqr_xstage(float* v, int lig) {
    const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
    for (int i = 0; i < ARQR_EPL; ++i) {
        const float o = r4d_xor_lane<M>(v[i]);
        v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
    }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this also undoes it.
__device__ __forceinline__ void arqr_wht(float* v, int lig) {
    static_assert(ARQR_LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
    for (int h = 1; h < ARQR_EPL; h <<= 1)
#pragma unroll
        for (int i = 0; i < ARQR_EPL; ++i)
            if ((i & h) == 0) { const float a = v[i], c = v[i ^ h]; v[i] = a + c; v[i ^ h] = a - c; }
    arqr_xstage<1>(v, lig);
    arqr_xstage<2>(v, lig);
    arqr_xstage<4>(v, lig);
    const float nrm = rsqrtf((float)ARQR_GROUP);
#pragma unroll
    for (int i = 0; i < ARQR_EPL; ++i) v[i] *= nrm;
}

__device__ __forceinline__ void arqr_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arqr_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <bool WF32, bool RES, uint32_t THR, int WIRE>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_fp8_row_kernel(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint8_t* __restrict__ q, float* __restrict__ s,
    int4* peer_base, const int4* my_base, unsigned int* peer_flags, unsigned int* my_flags,
    unsigned int* seq_ctrs, char* loc_pack, long loc_scale_off, int M, uint32_t n,
    float eps, float wadd, uint32_t r_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld,
    int slot_stride16, int drain, int acq, unsigned int nbw) {
    __shared__ float lds[THR / kWave];
    __shared__ float red[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    /* Integer arithmetic only, so both ranks land on the same split. */
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    const int cpr = WIRE ? (int)(n / ARQR_CE) : 0;
    const int c00 = r0 * cpr, cpb = (r1 - r0) * cpr;
    const uint32_t lane = tid & (ARQR_WV - 1), wave = tid / ARQR_WV;
    const int lig = (int)(lane & (ARQR_LPG - 1)), gw = (int)(lane / ARQR_LPG);

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* const ms = my_base + (size_t)s_slot * slot_stride16;
    char* const p_pay = (char*)ps;
    const char* const m_pay = (const char*)ms;
    unsigned short* const p_sc =
        (unsigned short*)((char*)ps + (size_t)(slot_stride16 >> 1) * 16);
    const unsigned short* const m_sc =
        (const unsigned short*)((const char*)ms + (size_t)(slot_stride16 >> 1) * 16);
    unsigned int* const Lpk = (unsigned int*)loc_pack;
    float* const Lsc = (float*)(loc_pack + loc_scale_off);

    if constexpr (WIRE == 0) {
        /* Push my rows into the peer's slot as 16B words; x is a tight [M][n] of bf16. */
        const int wpr = (int)(n >> 3);
        const int wend = r1 * wpr;
        const int4* const in4 = (const int4*)x;
        for (int k = r0 * wpr + (int)tid; k < wend; k += (int)THR) ps[k] = in4[k];
    } else {
        /* Rotate, amax, quantise and pack straight into the peer's slot, keeping the same words in
         * the side buffer. n is a whole number of chunks and a block owns whole rows, so no chunk
         * is partial and every cross-lane stage is wholly active. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQR_WV)) {
            const int c = c00 + ci;
            unsigned int acc[ARQR_NWORD];
#pragma unroll
            for (int wi = 0; wi < ARQR_NWORD; ++wi) acc[wi] = 0u;
#pragma unroll
            for (int t = 0; t < ARQR_ITERS; ++t) {
                const int g = c * ARQR_CG + t * ARQR_GPW + gw;
                const long off = (long)g * ARQR_GROUP + (long)lig * ARQR_EPL;
                float v[ARQR_EPL];
#pragma unroll
                for (int i = 0; i < ARQR_EPL; ++i) v[i] = bf2f(x[off + i]);
                arqr_wht(v, lig);
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < ARQR_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
                a = fmaxf(a, r4d_xor_lane<1>(a));
                a = fmaxf(a, r4d_xor_lane<2>(a));
                a = fmaxf(a, r4d_xor_lane<4>(a));
                float sc = (a > 0.0f) ? bf2f(f2bf(a * (1.0f / (float)ARQR_QMAX))) : 1.0f;
                if (!(sc > 0.0f)) sc = 1.0f;
                if (lig == 0) {
                    p_sc[g] = (unsigned short)(__float_as_uint(sc) >> 16);
                    Lsc[g] = sc;
                }
                const float inv = 1.0f / sc;
#pragma unroll
                for (int i = 0; i < ARQR_EPL; ++i) {
                    const unsigned int sym = (unsigned int)(
                        min(max((int)rintf(v[i] * inv), ARQR_QMIN), ARQR_QMAX) - ARQR_QMIN);
                    const int bit = (t * ARQR_EPL + i) * ARQR_BITS, wi = bit >> 5, bo = bit & 31;
                    acc[wi] |= sym << bo;
                    if (bo + ARQR_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
                }
            }
            /* Flush lane-contiguously, widest region first, so each is one coalesced store. */
            char* const pb = p_pay + (size_t)c * ARQR_CBYTES;
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQR_NWORD; w4 += 4)
                *(uint4*)(pb + w4 * 4 * ARQR_WV + lane * 16) =
                    make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]);
            if constexpr (ARQR_NWORD % 4 >= 2)
                *(uint2*)(pb + (ARQR_NWORD & ~3) * 4 * ARQR_WV + lane * 8) =
                    make_uint2(acc[ARQR_NWORD & ~3], acc[(ARQR_NWORD & ~3) + 1]);
            if constexpr (ARQR_NWORD % 2 == 1)
                *(unsigned int*)(pb + (ARQR_NWORD - 1) * 4 * ARQR_WV + lane * 4) =
                    acc[ARQR_NWORD - 1];
#pragma unroll
            for (int wi = 0; wi < ARQR_NWORD; ++wi)
                Lpk[((size_t)c * ARQR_NWORD + wi) * ARQR_WV + lane] = acc[wi];
        }
    }
    /* The drain modes are the standalone all-reduce's, and the variant table picks between them;
     * the kernel only has to honour them identically. */
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arqr_store_rel(&peer_flags[b], sq);
        unsigned long long z = 0;
        while (r4d_ar_seq_behind(arqr_load_acq(&my_flags[b]), sq)) {
            if (++z > ARQR_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    if (acq) __threadfence_system();

    if constexpr (WIRE == 6) {
        /* REDUCE: unpack both halves, add in fp32, inverse-rotate, store the bf16 sum into `x` --
         * what the standalone compressed all-reduce wrote, so pass one below reads its bytes. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQR_WV)) {
            const int c = c00 + ci;
            const char* const pb = m_pay + (size_t)c * ARQR_CBYTES;
            unsigned int pac[ARQR_NWORD], lac[ARQR_NWORD];
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQR_NWORD; w4 += 4) {
                const uint4 pq = *(const uint4*)(pb + w4 * 4 * ARQR_WV + lane * 16);
                pac[w4] = pq.x; pac[w4 + 1] = pq.y; pac[w4 + 2] = pq.z; pac[w4 + 3] = pq.w;
            }
            if constexpr (ARQR_NWORD % 4 >= 2) {
                const uint2 pu = *(const uint2*)(pb + (ARQR_NWORD & ~3) * 4 * ARQR_WV + lane * 8);
                pac[ARQR_NWORD & ~3] = pu.x; pac[(ARQR_NWORD & ~3) + 1] = pu.y;
            }
            if constexpr (ARQR_NWORD % 2 == 1)
                pac[ARQR_NWORD - 1] =
                    *(const unsigned int*)(pb + (ARQR_NWORD - 1) * 4 * ARQR_WV + lane * 4);
#pragma unroll
            for (int wi = 0; wi < ARQR_NWORD; ++wi)
                lac[wi] = Lpk[((size_t)c * ARQR_NWORD + wi) * ARQR_WV + lane];
#pragma unroll
            for (int t = 0; t < ARQR_ITERS; ++t) {
                const int gl = t * ARQR_GPW + gw;
                const int g = c * ARQR_CG + gl;
                const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
                const float lsc = Lsc[g];
                float v[ARQR_EPL];
#pragma unroll
                for (int i = 0; i < ARQR_EPL; ++i) {
                    const int bit = (t * ARQR_EPL + i) * ARQR_BITS, wi = bit >> 5, bo = bit & 31;
                    unsigned int psy = pac[wi] >> bo, lsy = lac[wi] >> bo;
                    if (bo + ARQR_BITS > 32) {
                        psy |= pac[wi + 1] << (32 - bo);
                        lsy |= lac[wi + 1] << (32 - bo);
                    }
                    /* Opposite fold order on the two ranks; fp32 add commutes exactly. */
                    v[i] = (float)((int)(psy & ARQR_MASK) + ARQR_QMIN) * psc
                         + (float)((int)(lsy & ARQR_MASK) + ARQR_QMIN) * lsc;
                }
                arqr_wht(v, lig);
                const long off = (long)g * ARQR_GROUP + (long)lig * ARQR_EPL;
#pragma unroll
                for (int i = 0; i < ARQR_EPL; ++i) x[off + i] = f2bf(v[i]);
            }
        }
        /* Pass one below reads `x` across the whole row and a wave only wrote its own chunks. */
        __syncthreads();
    }

    /* The peer's rows are now in MY scratch, in the same [M][n] layout as x -- WIRE == 0 only; the
     * compressed wire has already reduced into `x` above. */
    const uint16_t* __restrict__ P = (const uint16_t*)ms;

    for (int row = r0; row < r1; ++row) {
        uint16_t* __restrict__ xr = x + (size_t)row * n;
        const uint16_t* __restrict__ pr = P + (size_t)row * n;
        uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
        const uint16_t* __restrict__ nr = RES ? rr : xr;
        uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
        uint8_t* __restrict__ qr = q + (size_t)row * q_ld;

        /* Pass one, in rmsnorm_quant_fp8_row's own column order (kU = 4 strides a thread) so the
         * sum of squares is the standalone norm's to the bit. WIRE == 0 reduces HERE, on the norm's
         * pass; WIRE == 6 has already reduced into `x`. */
        constexpr uint32_t kU = 4;
        float ss = 0.0f;
        for (uint32_t i0 = tid; i0 < n; i0 += kU * THR) {
            uint16_t xa[kU], ra[kU];
#pragma unroll
            for (uint32_t u = 0; u < kU; ++u) {
                const uint32_t i = i0 + u * THR < n ? i0 + u * THR : i0;
                xa[u] = xr[i];
                if (RES) ra[u] = rr[i];
            }
#pragma unroll
            for (uint32_t u = 0; u < kU; ++u) {
                const uint32_t i = i0 + u * THR;
                if (i < n) {
                    uint16_t a;
                    if constexpr (WIRE == 0) { a = f2bf(bf2f(xa[u]) + bf2f(pr[i])); xr[i] = a; }
                    else                     { a = xa[u]; }
                    float v;
                    if (RES) { const uint16_t h = f2bf(bf2f(ra[u]) + bf2f(a)); rr[i] = h; v = bf2f(h); }
                    else     { v = bf2f(a); }
                    ss += v * v;
                }
            }
        }
        const float rs = rsqrtf(block_sum<THR>(ss, lds) / (float)n + eps);

        /* Pass two: the norm, written to `y`, and the whole-row amax of the bf16 value. */
        float m = 0.0f;
        for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (c0 < n) {
                float g[4];
                uint16_t xb[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g[j] = WF32 ? ((const float*)w)[c0 + j] : bf2f(((const uint16_t*)w)[c0 + j]);
                    xb[j] = nr[c0 + j];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint16_t h = f2bf(bf2f(xb[j]) * rs * (g[j] + wadd));
                    if (yr) yr[c0 + j] = h;
                    v[j] = bf2f(h);
                }
            }
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
        }
        const float amax = block_max<THR>(m, red);
        const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
        if (tid == 0) s[(size_t)row * s_ld] = sc;
        const float inv = 1.0f / sc;

        /* Pass three: the codes, from `out_bf16` where it exists and by recomputation where not. */
        for (uint32_t c0 = tid * 4u; c0 < n; c0 += THR * 4u) {
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (c0 < n) {
                if (yr) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = bf2f(yr[c0 + j]);
                } else {
                    float g[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        g[j] = WF32 ? ((const float*)w)[c0 + j]
                                    : bf2f(((const uint16_t*)w)[c0 + j]);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        v[j] = bf2f(f2bf(bf2f(nr[c0 + j]) * rs * (g[j] + wadd)));
                }
            }
            quant4(v, inv, qr + c0);
        }
        /* lds/red are re-used by the next row this block owns; without this barrier a thread still
         * reading them for this row would race the next row's pass one. */
        __syncthreads();
    }
}

}  // namespace

// ============================================================== the launches
//
// The block width is r4d_act_threads for the reason the standalone norm gives: it decides which
// columns a thread accumulates into the sum of squares, so a fused kernel that picked a different
// one would produce a different sum in the last bits and a different token stream. The amax is not
// sensitive to it (a max is exact), but the same width serves both.

#define R4D_RQR_T(wf, rs, T)                                                                  \
    emit_rc = R4D_EMIT(sk, (rmsnorm_quant_fp8_row_kernel<wf, rs, T>), dim3((uint32_t)M),       \
                       dim3(T), 0, (const uint16_t*)x, (uint16_t*)res, (const void*)w,         \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n, eps, wadd,           \
                       (uint32_t)x_ld, (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)q_ld,         \
                       (uint32_t)s_ld)
#define R4D_RQR(wf, rs)                                                                       \
    do { if (thr == 1024)     R4D_RQR_T(wf, rs, 1024);                                        \
         else if (thr == 512) R4D_RQR_T(wf, rs, 512);                                         \
         else                 R4D_RQR_T(wf, rs, 256); } while (0)

//   x    [M][n]              bf16, `x_ld` elements a row (0 = n)
//   res  [M][n]              bf16, `r_ld` elements a row (0 = n), optional, READ AND WRITTEN
//   w    [n]                 f32 or bf16
//   y    [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q    [M][n]              E4M3, `q_ld` bytes a row (0 = n), rounded up to 4
//   s    [M][1]              f32,  `s_ld` elements a row (0 = 1)
static int rmsnorm_quant_fp8_row_launch(long x, long res, long w, long y, long q, long s, int M,
                                        int n, float eps, float wadd, int w_f32, int x_ld, int r_ld,
                                        int y_ld, int q_ld, int s_ld, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!x || !w || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;              // a thread owns four columns; see the kernel
    if (!x_ld) x_ld = n;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = 1;
    const int thr = r4d_act_threads(M, n);
    if (w_f32) { if (res) R4D_RQR(true, true);  else R4D_RQR(true, false); }
    else       { if (res) R4D_RQR(false, true); else R4D_RQR(false, false); }
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// The fused all-reduce. The IPC arguments are r4d_ar_oneshot_2rank_exact's -- the same
// double-buffered scratch, per-block flags and device-resident sequence counters -- and everything
// after them is rmsnorm_quant_fp8_row's, with two differences the caller has to know about:
//
//   * `x` HAS NO PITCH. It is the all-reduce's message as well as the norm's input, and a block
//     pushes its rows as a range of 16B words; a pitched x would make the two ranks disagree about
//     which bytes a block owns. Refused rather than silently strided.
//   * `nblocks` IS CLAMPED TO M, because a block owns whole rows.
//
// `res` is the fused_add_rms_norm form and behaves exactly as it does there. `wire` is 0 for the
// exact bf16 payload and 6 for the rotated 6-bit one, which additionally needs n to be a multiple
// of the wire's chunk so that a chunk never straddles a row. Returns 0, or a negative code.
extern "C" int r4d_ar_rmsnorm_quant_fp8_row_chunk_elems(void) { return ARQR_CE; }

extern "C" int r4d_ar_rmsnorm_quant_fp8_row(long peer_scratch, long my_scratch, long peer_flags,
                                            long my_flags, long seq_ctrs, long slot_stride16,
                                            long x, long res, long w, long y, long q, long s,
                                            int M, int n, float eps, float wadd, int w_f32,
                                            int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                            int nblocks, int nthreads, int drain, int acq,
                                            int wire, long loc_pack, long loc_scale_off) {
    if (!x || !w || !q || !s) return -1;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;              // a thread owns four columns; see the kernel
    if (n % 8) return -5;              // n bf16 is a whole number of 16B words; see the push
    if (wire != 0 && wire != 6) return -7;
    if (wire == 6 && (n % ARQR_CE)) return -8;
    if (wire == 6 && (!loc_pack || loc_scale_off <= 0)) return -9;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = 1;               // one f32 scale per ROW, not per 128-column group

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    const int nbw = r4d_ar_max_blocks();       // the counter capacity; see r4d_ar_seq.h
    if (nb > nbw) nb = nbw;
    /* The width is r4d_act_threads for the same reason the standalone row norm gives: it decides
     * which columns a thread accumulates into the sum of squares. Zero means "resolve it". */
    const int thr = nthreads ? nthreads : r4d_act_threads(M, n);
    if (thr != 256 && thr != 512 && thr != 1024) return -6;
#define R4D_ARQR_T(wf, rs, T, W)                                                                 \
    hipLaunchKernelGGL((ar_rmsnorm_quant_fp8_row_kernel<wf, rs, T, W>), dim3((uint32_t)nb),       \
                       dim3(T), 0, (hipStream_t)stream, (uint16_t*)x, (uint16_t*)res,            \
                       (const void*)w, (uint16_t*)y, (uint8_t*)q, (float*)s, (int4*)peer_scratch, \
                       (const int4*)my_scratch, (unsigned int*)peer_flags,                       \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, (char*)loc_pack,        \
                       (long)loc_scale_off, M, (uint32_t)n, eps, wadd, (uint32_t)r_ld,            \
                       (uint32_t)y_ld, (uint32_t)q_ld, (uint32_t)s_ld, (int)slot_stride16, drain, \
                       acq, (unsigned)nbw)
#define R4D_ARQR_W(wf, rs, T)                                                                    \
    do { if (wire == 6) R4D_ARQR_T(wf, rs, T, 6); else R4D_ARQR_T(wf, rs, T, 0); } while (0)
#define R4D_ARQR(wf, rs)                                                                         \
    do { if (thr == 1024)     R4D_ARQR_W(wf, rs, 1024);                                          \
         else if (thr == 512) R4D_ARQR_W(wf, rs, 512);                                           \
         else                 R4D_ARQR_W(wf, rs, 256); } while (0)
    if (w_f32) { if (res) R4D_ARQR(true, true);  else R4D_ARQR(true, false); }
    else       { if (res) R4D_ARQR(false, true); else R4D_ARQR(false, false); }
#undef R4D_ARQR
#undef R4D_ARQR_W
#undef R4D_ARQR_T
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

#define R4D_GQR_T(a, T)                                                                       \
    emit_rc = R4D_EMIT(sk, (gated_quant_fp8_row_kernel<a, T>), dim3((uint32_t)M), dim3(T), 0, \
                       (const uint16_t*)gu, (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n, \
                       (uint32_t)gu_ld, (uint32_t)y_ld, (uint32_t)q_ld, (uint32_t)s_ld)
#define R4D_GQR(a)                                                                            \
    do { if (thr == 1024)     R4D_GQR_T(a, 1024);                                             \
         else if (thr == 512) R4D_GQR_T(a, 512);                                              \
         else                 R4D_GQR_T(a, 256); } while (0)

//   gu [M][2n]             bf16, gate first then up, `gu_ld` elements a row (0 = 2n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3, `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][1]              f32,  `s_ld` elements a row (0 = 1)
// `mode` is 0 for silu and 1 for sigmoid, the spelling r4d_gated_quant_fp8 takes.
static int gated_quant_fp8_row_launch(long gu, long y, long q, long s, int M, int n, int mode,
                                      int gu_ld, int y_ld, int q_ld, int s_ld, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!gu || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = 1;
    const int thr = r4d_act_threads(M, n);
    if (mode == 0) R4D_GQR(0);
    else           R4D_GQR(1);
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

#define R4D_GTQR_T(a, T)                                                                      \
    emit_rc = R4D_EMIT(sk, (gate_quant_fp8_row_kernel<a, T>), dim3((uint32_t)M), dim3(T), 0, \
                       (const uint16_t*)g, (const uint16_t*)x, (uint16_t*)y, (uint8_t*)q,    \
                       (float*)s, (uint32_t)n, (uint32_t)g_ld, (uint32_t)x_ld,               \
                       (uint32_t)y_ld, (uint32_t)q_ld, (uint32_t)s_ld)
#define R4D_GTQR(a)                                                                           \
    do { if (thr == 1024)     R4D_GTQR_T(a, 1024);                                            \
         else if (thr == 512) R4D_GTQR_T(a, 512);                                             \
         else                 R4D_GTQR_T(a, 256); } while (0)

//   g  [M][.]              bf16, `g_ld` elements a row; the gate starts at element 0 of the row
//   x  [M][n]              bf16, `x_ld` elements a row (0 = n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3, `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][1]              f32,  `s_ld` elements a row (0 = 1)
static int gate_quant_fp8_row_emit(long g, long x, long y, long q, long s, int M, int n, int mode,
                                   int g_ld, int x_ld, int y_ld, int q_ld, int s_ld,
                                   const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!g || !x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (!g_ld) g_ld = n;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = 1;
    const int thr = r4d_act_threads(M, n);
    if (mode == 0) R4D_GTQR(0);
    else           R4D_GTQR(1);
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. The operand lists are the group=128 fusions' -- the
// same producer operands with the quantiser's two outputs appended -- and the ONE difference is
// the scale's extent: `[M][1]` rather than `[M][ceil(n/128)]`, which is what makes the MX GEMM's
// `As[m]` fold read it.
static int rad_gated_quant_fp8_row(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* gu = r4d_opt(a, GQ_GU);
    const RadTensor* q  = r4d_opt(a, GQ_Q);
    const RadTensor* sc = r4d_opt(a, GQ_S);
    const RadTensor* y  = r4d_opt(a, GQ_Y);
    if (!gu || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    if (gu->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = gu->shape[0], K = q->shape[1];
    /* The gated operand is [M][2K] and the halves are gate then up -- one buffer, because that is
     * how the fused gate_up projection writes it. */
    if (gu->shape[1] != 2 * K) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    const long long gp = r4d_row_pitch(gu), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    if (gp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The store is a 32-bit word per four columns; `n % 4` only aligns the chunk within a row. A
     * strided `q` whose pitch is not a multiple of 4 would put that word off a 4-byte boundary. */
    if (qp % 4) return R4D_NO(RAD_E_STRIDE);
    const char* act = r4d_gets(a, "act");
    int mode = (int)r4d_geti_or(a, "mode", 0);
    if (act) {
        if (!std::strcmp(act, "silu")) mode = 0;
        else if (!std::strcmp(act, "sigmoid")) mode = 1;
        else return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const int e = gated_quant_fp8_row_launch(r4d_addr(gu), y ? r4d_addr(y) : 0, r4d_addr(q),
                                             r4d_addr(sc), (int)M, (int)K, mode, (int)gp, (int)yp,
                                             (int)qp, (int)sp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_gated_quant_fp8_row, rad_gated_quant_fp8_row)

/* Elements between consecutive `width`-wide rows, the same rule r4d_fused_quant_fp8.hip's tq_pitch
 * uses: a column view of an interleaved projection. Duplicated rather than shared because the two
 * translation units may not link against each other. */
static long long tq_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* THE SAME EPILOGUE WITH THE GATE SOMEWHERE ELSE. A gated attention's gate is a COLUMN SLICE of
 * the [q|gate] projection, so this takes two operands where gated_quant_fp8_row takes one. `M` is
 * counted off `x` over the known width, because the row is a (token, head) pair that neither
 * operand spells out. */
static int rad_gate_quant_fp8_row_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* g  = r4d_opt(a, TQ_G);
    const RadTensor* x  = r4d_opt(a, TQ_X);
    const RadTensor* q  = r4d_opt(a, TQ_Q);
    const RadTensor* sc = r4d_opt(a, TQ_S);
    const RadTensor* y  = r4d_opt(a, TQ_Y);
    if (!g || !x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    const long long K = r4d_geti_or(a, "n", 0);
    if (K <= 0 || r4d_numel(x) % K) return R4D_NO(RAD_E_SHAPE);
    const long long M = r4d_numel(x) / K;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(g) < M * K) return R4D_NO(RAD_E_SHAPE);
    /* ONE SCALE PER ROW: `n` elements, not `ceil(n/128)` of them. */
    if (r4d_numel(q) < M * K || r4d_numel(sc) < M) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) < M * K) return R4D_NO(RAD_E_SHAPE);
    const long long gp = tq_row_pitch(g, K), xp = tq_row_pitch(x, K);
    const long long qp = tq_row_pitch(q, K), sp = tq_row_pitch(sc, 1);
    const long long yp = y ? tq_row_pitch(y, K) : 0;
    if (gp < 0 || xp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (qp % 4) return R4D_NO(RAD_E_STRIDE);
    const char* act = r4d_gets(a, "act");
    int mode = (int)r4d_geti_or(a, "mode", 1);
    if (act) {
        if (!std::strcmp(act, "silu")) mode = 0;
        else if (!std::strcmp(act, "sigmoid")) mode = 1;
        else return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const int e = gate_quant_fp8_row_emit(r4d_addr(g), r4d_addr(x), y ? r4d_addr(y) : 0,
                                          r4d_addr(q), r4d_addr(sc), (int)M, (int)K, mode,
                                          (int)gp, (int)xp, (int)yp, (int)qp, (int)sp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_gate_quant_fp8_row, rad_gate_quant_fp8_row_body)

/* rmsnorm_quant_fp8_row is defined above as a plain `extern "C"` launch; its describe twin is
 * generated here from the same body. */
static int rad_rmsnorm_quant_fp8_row_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x  = r4d_opt(a, RQ_X);
    const RadTensor* rs = r4d_opt(a, RQ_RES);
    const RadTensor* w  = r4d_opt(a, RQ_W);
    const RadTensor* q  = r4d_opt(a, RQ_Q);
    const RadTensor* sc = r4d_opt(a, RQ_S);
    const RadTensor* y  = r4d_opt(a, RQ_Y);
    if (!x || !w || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3) return R4D_NO(RAD_E_DTYPE);
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], n = x->shape[1];
    if (r4d_numel(w) != n) return R4D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (q->shape[0] != M || q->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != n)) return R4D_NO(RAD_E_SHAPE);
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != n)) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return R4D_NO(RAD_E_STRIDE);
    if (qp % 4) return R4D_NO(RAD_E_STRIDE);
    const int e = rmsnorm_quant_fp8_row_launch(
        r4d_addr(x), rs ? r4d_addr(rs) : 0, r4d_addr(w), y ? r4d_addr(y) : 0, r4d_addr(q),
        r4d_addr(sc), (int)M, (int)n, (float)r4d_getf(a, "eps", 1e-6),
        (float)r4d_getf(a, "wadd", 0.0), w->dtype == RAD_F32, (int)xp, (int)rp, (int)yp, (int)qp,
        (int)sp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}
R4D_ENTRY(r4d_rad_rmsnorm_quant_fp8_row, rad_rmsnorm_quant_fp8_row_body)
