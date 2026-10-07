// r4d_quant_act_fp8_row.hip -- per-row (per-token) symmetric E4M3 quantisation of a bf16
// activation, one f32 scale per row. The OCP-MX (mxfp4a8) GEMMs read their activation this way:
// the m64 kernel's epilogue is `C[m][n] = acc * 2^(Wref[n]-127) * As[m]`, one activation scale per
// token row, and there is no per-128 group on the activation side.
//
// WHY THIS IS A NEW OP AND NOT A PARAMETER ON quant_act_fp8. That kernel hardcodes kGroup=128
// (r4d_quant_act_fp8.hip) and the constraint system has no param-eq-param predicate, so
// `group == n` cannot be a row constraint; and a group of 128 would put a scale on a grid the MX
// GEMM does not read. The arithmetic is quant_act_fp8's, restated at a group of the whole row:
// 448, scale 1 for a zero row, and the clamp are the same three choices, and the output is plain
// row-major e4m3 with a single f32 per row.
//
// THE REDUCTION IS BLOCK-LEVEL, not wave-level. A wave owns a 128-column group in the 128-group
// kernel and reduces within the wave; here one scale covers the whole row, which runs to 17408
// wide, so the amax is a block reduction over every column. One block per row, 256 threads, each
// thread a 4-column chunk striding the row; the row is read twice (absmax, then quantise) rather
// than staged in LDS, because 17408 elements does not fit a block's LDS and the row is small
// against the model.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_common.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;

__global__ __launch_bounds__(kThreads) void quant_act_fp8_row_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    // A thread owns 4 consecutive columns per chunk: chunk t (thread t) covers columns t*4 ..
    // t*4+3 and the stride is kThreads*4, so the wave covers 128 contiguous columns per pass and
    // the whole block covers kThreads*4 = 1024. K is a multiple of 4 (the row's constraint), so a
    // chunk that starts inside the row is four whole elements inside it.
    float m = 0.0f;
    for (uint32_t c0 = tid * 4u; c0 < K; c0 += kThreads * 4u) {
        const uint16_t* p = x + xrow + c0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = __uint_as_float((uint32_t)p[j] << 16);
            m = fmaxf(m, fabsf(v));
        }
    }
    // Block-level max reduction: wave reduce, the wave leaders into LDS, then one wave.
    m = r4d_wave_max(m);
    __shared__ float red[kWaves];
    const uint32_t wave = tid / kWave, lane = tid & (kWave - 1);
    if (lane == 0) red[wave] = m;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < kWaves) ? red[lane] : 0.0f;
        v = r4d_wave_max(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float amax = red[0];
    // 448 is E4M3's largest finite value and the scale is amax/448 -- a DEQUANT multiplier, the
    // same sense as the checkpoint's weight_scale_inv, so the GEMM's fold multiplies both scales.
    // A zero row gets a scale of 1, because amax/448 would be zero and its reciprocal infinite.
    const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
    if (tid == 0) s[row * s_ld] = sc;
    const float inv = 1.0f / sc;

    // THE CLAMP IS NOT DEFENSIVE: amax * (1/scale) is 448 in exact arithmetic but can land an ulp
    // above it in float, and v_cvt_pk_fp8_f32 answers an out-of-range input with NaN rather than
    // saturating -- so without the clamp every row produces a NaN at its own maximum.
    for (uint32_t c0 = tid * 4u; c0 < K; c0 += kThreads * 4u) {
        const uint16_t* p = x + xrow + c0;
        const float v0 = __uint_as_float((uint32_t)p[0] << 16) * inv;
        const float v1 = __uint_as_float((uint32_t)p[1] << 16) * inv;
        const float v2 = __uint_as_float((uint32_t)p[2] << 16) * inv;
        const float v3 = __uint_as_float((uint32_t)p[3] << 16) * inv;
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v0, -448.0f), 448.0f),
                                             fminf(fmaxf(v1, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v2, -448.0f), 448.0f),
                                             fminf(fmaxf(v3, -448.0f), 448.0f), pk, true);
        *(uint32_t*)(q + qrow + c0) = (uint32_t)pk;
    }
}

}  // namespace

//   a  [M][K]                bf16, `x_ld` elements a row (0 = K)
//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K)
//   s  [M][1]                f32,  `s_ld` elements a row (0 = 1)
static int quant_act_fp8_row_launch(long a, long q, long s, int M, int K,
                                    int x_ld, int q_ld, int s_ld, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    if (!a || !q || !s) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;              // a thread owns four columns; see the kernel
    if (!x_ld) x_ld = K;
    if (!q_ld) q_ld = K;
    if (!s_ld) s_ld = 1;
    emit_rc = R4D_EMIT(sk, quant_act_fp8_row_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
                       (const uint16_t*)a, (uint8_t*)q, (float*)s,
                       (uint32_t)K, (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld);
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// ============================================================== the plugin ABI entry point
//
// The shared operand reader is r4d_fp8_args.h (QF_X/QF_Q/QF_S). See libr4d/r4d_args.h for why
// there is no adapter directory between the core and these launches.
static int rad_quant_act_fp8_row(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x  = r4d_opt(a, QF_X);
    const RadTensor* q  = r4d_opt(a, QF_Q);
    const RadTensor* sc = r4d_opt(a, QF_S);
    if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* ROWS ARE DIM 0 AND EVERYTHING AFTER IT IS THE ROW, the same reading r4d_rad_quant_act_fp8
     * takes: a rank-3 activation is one row per token, which is what the consuming GEMM forces.
     * `q` and `sc` stay rank 2 because this kernel writes them and they exist for the GEMM. */
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    /* ONE SCALE PER ROW: the whole point of this op over quant_act_fp8. A [M, n/128] plane here
     * would read a plausible number and be wrong on every token. */
    if (sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    /* THE STORE IS A 32-BIT WORD PER FOUR COLUMNS. `n % 4` only aligns the chunk within a row; a
     * strided `q` whose pitch is not a multiple of 4 would put that word off a 4-byte boundary. */
    if (qp % 4) return R4D_NO(RAD_E_STRIDE);
    /* ...and the base must be 4-byte aligned too, or the first word of every row is off the
     * boundary the `*(uint32_t*)(q + qrow + c0)` store requires (qrow is 4-aligned by `qp % 4`). */
    if (!r4d_aligned(q->data, 4)) return R4D_NO(RAD_E_ALIGN);
    const int e = quant_act_fp8_row_launch(r4d_addr(x), r4d_addr(q), r4d_addr(sc), (int)M, (int)K,
                                           (int)xp, (int)qp, (int)sp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_quant_act_fp8_row_body(const RadArgs* a, const R4dSink& sk) {
    return rad_quant_act_fp8_row(a, sk);
}
R4D_ENTRY(r4d_rad_quant_act_fp8_row, rad_quant_act_fp8_row_body)
