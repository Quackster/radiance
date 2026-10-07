// r4d_paroquant_quant_fp8_row.hip -- THE PATH-A PRODUCER. One row of a bf16 activation is turned
// into the domain a still-rotated ParoQuant weight is stored in, then quantised to E4M3 with one
// f32 scale per row.
//
// WHY THIS OP EXISTS AND WHY IT IS NOT quant_act_fp8_row. A ParoQuant checkpoint ships its trunk
// ROTATED: the stored weight v is w = R(v) . D for the base weight w (PARO-SERVING.md §1). A GEMM
// that reads v computes a . v^T, so to reproduce x . w^T the ACTIVATION must be rotated:
//
//     a = x . D . M_dec^T = (x / t.scale) . M_dec^T
//
// -- multiply each column by channel_scales (the container stores t.scale = 1/channel_scales, so
// this is a DIVIDE), then apply the Givens stages ASCENDING and TRANSPOSED, (a,b) -> (c*a + s*b,
// -s*a + c*b). The rotation is the generic, learned one: unlike the Walsh-Hadamard the fp8 family
// folds in, it mixes ARBITRARY column pairs within a 128-wide group, so it cannot be done wave-local
// in registers. Each row is read twice (amax, then quantise) exactly as quant_act_fp8_row reads it;
// the rotation between the two is redone rather than staged, because 17408 f32 does not fit LDS.
//
// THE TABLES ARE WEIGHT OPERANDS. `pairs` (i32 [S, n], column-global), `angle` (f32 [S, n/2], radians)
// and `t_scale` (f32 [1, n]) belong to the SAME logical weight the consuming GEMM reads, so they are
// declared as weight operands and cached with it rather than fed per call.
//
// NO NORM HERE. The RMS norm is a separate producer (the tree's rmsnorm form writing bf16); this op
// is the rotated quantiser that follows it, which keeps it independently testable and lets one norm
// feed several consumers. See PAROQUANT-PATH-A-PLAN.md.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_common.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;        /* the ParoQuant rotation-group width */
/* The whole-row-in-LDS path is ONE ROTATION PASS, and the width it can hold. Above it the row is
 * re-rotated for the quantiser (r4d_paroquant_quant_fp8_row_two_pass). K = 17408 is the down
 * projection and is the only one over this cap; every 5120/6144 projection is under it. */
constexpr int64_t  kLdsCap = 32 * 1024;

/* ONE ROTATION PASS over a row staged whole in LDS. Every stage's K/2 pairs are disjoint, so a
 * single barrier a stage is all the ordering the rotation needs -- eight barriers a row against the
 * two-pass kernel's hundreds -- and the amax and the quantiser read the rotated row instead of
 * rotating it again. This is the decode path: M is small and the fixed cost of a launch and of the
 * barrier chain is what the producer must not pay. */
__global__ __launch_bounds__(kThreads) void paroquant_quant_fp8_row_lds_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ pairs,
    const float* __restrict__ angle, const float* __restrict__ tscale,
    uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t S, int32_t k_base, uint32_t x_ld, uint32_t pair_ld, uint32_t angle_ld,
    uint32_t ts_ld, uint32_t q_ld, uint32_t s_ld) {
    extern __shared__ float g[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    for (uint32_t k = tid; k < K; k += kThreads)
        g[k] = __uint_as_float((uint32_t)x[xrow + k] << 16) / tscale[k];
    __syncthreads();
    /* THE GROUPS ARE INDEPENDENT, SO THEY RUN IN PARALLEL ACROSS WARPS. A 128-group is a warp's own
     * slab, its stages need only `__syncwarp`, and no barrier spans two warps until the amax --
     * against one block-wide barrier a stage when a single warp walked all the groups. This is what
     * the reference producer (par_kernels.h's pq_rotate_quant, one block a (group, row)) buys, and
     * it is the whole of the decode cost. `g` is the whole row staged once, so the quantiser reads
     * the rotation instead of repeating it. */
    const uint32_t G = K / kGroup;
    const uint32_t lane = tid & (kWave - 1), warp = tid / kWave;
    for (uint32_t grp = warp; grp < G; grp += kWaves) {
        const uint32_t b0 = grp * kGroup;
        for (uint32_t st = 0; st < S; ++st) {
            const int32_t* pr = pairs + (size_t)st * pair_ld + grp * (kGroup / 2) * 2;
            const float*   an = angle + (size_t)st * angle_ld + grp * (kGroup / 2);
#pragma unroll
            for (uint32_t t2 = 0; t2 < 2; ++t2) {
                const uint32_t t = lane + kWave * t2;
                const uint32_t a = (uint32_t)pr[2 * t] - k_base, b = (uint32_t)pr[2 * t + 1] - k_base;
                const float c = cosf(an[t]), sn = sinf(an[t]);
                const float xa = g[a], xb = g[b];
                g[a] = fmaf(c, xa, sn * xb);
                g[b] = fmaf(c, xb, -sn * xa);
            }
            __syncwarp();
        }
    }
    __syncthreads();

    float amax = 0.0f;
    for (uint32_t k = tid; k < K; k += kThreads) amax = fmaxf(amax, fabsf(g[k]));
    amax = r4d_wave_max(amax);
    __shared__ float red[kWaves];
    if (lane == 0) red[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < kWaves) ? red[lane] : 0.0f;
        v = r4d_wave_max(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float sc = red[0] > 0.0f ? red[0] * (1.0f / 448.0f) : 1.0f;
    if (tid == 0) s[(size_t)row * s_ld] = sc;
    const float inv = 1.0f / sc;
    for (uint32_t k = tid; k < K; k += kThreads) {
        const float v = fminf(fmaxf(g[k] * inv, -448.0f), 448.0f);
        const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
        q[qrow + k] = (uint8_t)pk;
    }
}

__global__ __launch_bounds__(kThreads) void paroquant_quant_fp8_row_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ pairs,
    const float* __restrict__ angle, const float* __restrict__ tscale,
    uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t S, int32_t k_base, uint32_t x_ld, uint32_t pair_ld, uint32_t angle_ld,
    uint32_t ts_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;
    const uint32_t ngroup = K / kGroup;

    /* A 128-wide group in LDS: the whole rotation unit. Threads 0..127 own one column each; the 64
     * pairs of a stage are spread over the block and the barrier between stages is what makes each
     * stage read its predecessor's writes. */
    __shared__ float g[kGroup];

    /* pass 1: rotate and keep every column's magnitude for the whole-row amax. */
    float amax = 0.0f;
    for (uint32_t gi = 0; gi < ngroup; ++gi) {
        const uint32_t b0 = gi * kGroup;
        if (tid < kGroup)
            g[tid] = __uint_as_float((uint32_t)x[xrow + b0 + tid] << 16) / tscale[b0 + tid];
        __syncthreads();
        for (uint32_t st = 0; st < S; ++st) {
            const int32_t* pr = pairs + (size_t)st * pair_ld + gi * kGroup;
            const float*   an = angle + (size_t)st * angle_ld + gi * (kGroup / 2);
            for (uint32_t p = tid; p < kGroup / 2; p += kThreads) {
                const int32_t a = pr[2 * p] - k_base - (int32_t)b0,
                              b = pr[2 * p + 1] - k_base - (int32_t)b0;
                const float t = an[p]; float c, sn; __sincosf(t, &sn, &c);
                const float xa = g[a], xb = g[b];
                g[a] = c * xa + sn * xb;
                g[b] = -sn * xa + c * xb;
            }
            __syncthreads();
        }
        if (tid < kGroup) amax = fmaxf(amax, fabsf(g[tid]));
        __syncthreads();
    }

    /* the whole-row amax, then amax/448 -- a DEQUANT multiplier, as in quant_act_fp8_row. A zero
     * row takes a scale of 1 so the reciprocal below is finite. */
    amax = r4d_wave_max(amax);
    __shared__ float red[kWaves];
    const uint32_t wave = tid / kWave, lane = tid & (kWave - 1);
    if (lane == 0) red[wave] = amax;
    __syncthreads();
    if (wave == 0) {
        float v = (lane < kWaves) ? red[lane] : 0.0f;
        v = r4d_wave_max(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float sc = red[0] > 0.0f ? red[0] * (1.0f / 448.0f) : 1.0f;
    if (tid == 0) s[(size_t)row * s_ld] = sc;
    const float inv = 1.0f / sc;

    /* pass 2: rotate again and emit the codes over the bf16-rounded (here f32) rotated value. */
    for (uint32_t gi = 0; gi < ngroup; ++gi) {
        const uint32_t b0 = gi * kGroup;
        if (tid < kGroup)
            g[tid] = __uint_as_float((uint32_t)x[xrow + b0 + tid] << 16) / tscale[b0 + tid];
        __syncthreads();
        for (uint32_t st = 0; st < S; ++st) {
            const int32_t* pr = pairs + (size_t)st * pair_ld + gi * kGroup;
            const float*   an = angle + (size_t)st * angle_ld + gi * (kGroup / 2);
            for (uint32_t p = tid; p < kGroup / 2; p += kThreads) {
                const int32_t a = pr[2 * p] - k_base - (int32_t)b0,
                              b = pr[2 * p + 1] - k_base - (int32_t)b0;
                const float t = an[p]; float c, sn; __sincosf(t, &sn, &c);
                const float xa = g[a], xb = g[b];
                g[a] = c * xa + sn * xb;
                g[b] = -sn * xa + c * xb;
            }
            __syncthreads();
        }
        if (tid < kGroup) {
            const float v = fminf(fmaxf(g[tid] * inv, -448.0f), 448.0f);
            const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
            q[qrow + b0 + tid] = (uint8_t)pk;
        }
        __syncthreads();
    }
}

}  // namespace

//   x       [M][K]        bf16, `x_ld` elements a row (0 = K)
//   pairs   [S][K]        i32, column-global Givens pairs, `pair_ld` elements a row (0 = K)
//   angle   [S][K/2]      f32 radians, `angle_ld` elements a row (0 = K/2)
//   t_scale [1][K]        f32, 1/channel_scales, `ts_ld` elements a row (0 = K)
//   q       [M][K]        E4M3, `q_ld` bytes a row (0 = K)
//   s       [M][1]        f32,  `s_ld` elements a row (0 = 1)
static int paroquant_quant_fp8_row_launch(long a, long pairs, long angle, long tscale, long q,
                                          long s, int M, int K, int S, int k_base, int x_ld,
                                          int pair_ld, int angle_ld, int ts_ld, int q_ld, int s_ld,
                                          const R4dSink& sk) {
    if (!a || !pairs || !angle || !tscale || !q || !s) return -1;
    if (M <= 0 || K <= 0 || S <= 0) return -2;
    if (K % (int)kGroup) return -3;
    if (!x_ld) x_ld = K;
    if (!pair_ld) pair_ld = K;
    if (!angle_ld) angle_ld = K / 2;
    if (!ts_ld) ts_ld = K;
    if (!q_ld) q_ld = K;
    if (!s_ld) s_ld = 1;
    /* THE ROW FITS LDS: rotate it once and read it back, the whole decode cost being one launch and
     * eight barriers. Otherwise the two-pass kernel, which pays a second rotation but needs no more
     * shared memory than one 128-group. */
    if ((int64_t)K * 4 <= kLdsCap) {
        const int emit_rc = R4D_EMIT(sk, paroquant_quant_fp8_row_lds_kernel, dim3((uint32_t)M),
                                     dim3(kThreads), (size_t)K * 4,
                                     (const uint16_t*)a, (const int32_t*)pairs, (const float*)angle,
                                     (const float*)tscale, (uint8_t*)q, (float*)s,
                                     (uint32_t)K, (uint32_t)S, (int32_t)k_base, (uint32_t)x_ld,
                                     (uint32_t)pair_ld, (uint32_t)angle_ld, (uint32_t)ts_ld,
                                     (uint32_t)q_ld, (uint32_t)s_ld);
        if (emit_rc != RAD_OK) return -4;
        return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
    }
    const int emit_rc = R4D_EMIT(sk, paroquant_quant_fp8_row_kernel, dim3((uint32_t)M),
                                 dim3(kThreads), 0,
                                 (const uint16_t*)a, (const int32_t*)pairs, (const float*)angle,
                                 (const float*)tscale, (uint8_t*)q, (float*)s,
                                 (uint32_t)K, (uint32_t)S, (int32_t)k_base, (uint32_t)x_ld,
                                 (uint32_t)pair_ld, (uint32_t)angle_ld, (uint32_t)ts_ld,
                                 (uint32_t)q_ld, (uint32_t)s_ld);
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// ============================================================== the plugin ABI entry point
enum { PQ_X = 0, PQ_PAIRS = 1, PQ_ANGLE = 2, PQ_TSCALE = 3, PQ_Q = 4, PQ_S = 5 };

static int rad_paroquant_quant_fp8_row(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x  = r4d_opt(a, PQ_X);
    const RadTensor* pr = r4d_opt(a, PQ_PAIRS);
    const RadTensor* an = r4d_opt(a, PQ_ANGLE);
    const RadTensor* ts = r4d_opt(a, PQ_TSCALE);
    const RadTensor* q  = r4d_opt(a, PQ_Q);
    const RadTensor* sc = r4d_opt(a, PQ_S);
    if (!x || !pr || !an || !ts || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* ROWS ARE DIM 0 AND EVERYTHING AFTER IT IS THE ROW, exactly as quant_act_fp8_row reads its
     * input: a rank-3 activation is one row per token. `q`/`sc` stay rank 2 because they exist for
     * the GEMM. */
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0];
    if (M < 1 || r4d_numel(x) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(x) / M;
    if (K < (long long)kGroup || K % (long long)kGroup) return R4D_NO(RAD_E_SHAPE);
    if (pr->rank != 2 || pr->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    const long long S = pr->shape[0];
    if (S < 1) return R4D_NO(RAD_E_SHAPE);
    if (an->rank != 2 || an->shape[0] != S || an->shape[1] != K / 2) return R4D_NO(RAD_E_SHAPE);
    if (ts->rank > 2 || r4d_numel(ts) != K) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), pp = r4d_row_pitch(pr), ap = r4d_row_pitch(an),
                    tp = r4d_row_pitch(ts), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || pp < 0 || ap < 0 || tp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    /* `k_base` is this rank's first column of the FULL K the Givens pairs are indexed against: a
     * row-parallel weight's activation is this rank's K-slice and its table a column slice, but the
     * pair values stay global, so the kernel rebases them. 0 for a K the world does not cut. */
    long long kb = 0;
    r4d_geti(a, "k_base", &kb);
    const int k_base = (int)kb;
    const int e = paroquant_quant_fp8_row_launch(
        r4d_addr(x), r4d_addr(pr), r4d_addr(an), r4d_addr(ts), r4d_addr(q), r4d_addr(sc),
        (int)M, (int)K, (int)S, k_base, (int)xp, (int)pp, (int)ap, (int)tp, (int)qp, (int)sp, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_paroquant_quant_fp8_row_body(const RadArgs* a, const R4dSink& sk) {
    return rad_paroquant_quant_fp8_row(a, sk);
}
R4D_ENTRY(r4d_rad_paroquant_quant_fp8_row, rad_paroquant_quant_fp8_row_body)
