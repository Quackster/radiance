// r4d_paroquant_fused_row.hip -- THE PATH-A PRODUCER, FUSED INTO THE OP THAT ALREADY RUNS.
//
// The standalone producer (r4d_paroquant_quant_fp8_row) rotates one bf16 activation into ONE
// consumer's stored domain, and the multi form (r4d_paroquant_quant_fp8_row_multi) does P consumers
// of a shared input in one launch -- but both still run ON TOP of the norm (or the gated product)
// that produced their input. Measured on the shipped Path-A container, that extra launch per
// consumer is ~100% of the decode gap to Plain-MXFP6: skipping the producers entirely restores Path
// B's rate, and no amount of intra-kernel width moves it, because the cost is the launch itself and
// the serial dependency at small M. The fix is to remove the launches, not the rotations.
//
// These two ops ARE the producers' consumers minus the extra launch. Each is the fused row op the
// MX family already runs -- `rmsnorm_quant_fp8_row` for the norm-fed linears, `gated_quant_fp8_row`
// for the silu/sigmoid-fed ones -- with the Givens rotation and the per-row E4M3 quantiser folded
// into the SAME launch, once per consumer's partition:
//
//   paroquant_rmsnorm_quant_fp8_rowP  norm(+residual add) -> P x (divide by that partition's
//                                     t_scale, Givens stages ascending+transposed, whole-row amax,
//                                     E4M3 codes + one f32 scale). Feeds every linear whose input is
//                                     a block norm: MLP gate/up, attention q/k/v, GDN qkv/z.
//   paroquant_gated_quant_fp8_rowP    silu(gate)*up (or sigmoid) -> P x rotated+quantised
//                                     activation. Feeds the MLP down projection's own rotation.
//
// NUMERIC CONTRACT: bit-identical to the pair it replaces, so Path A's quality is unchanged. The
// norm pass is rmsnorm_quant_fp8_row_kernel's (same kU=4 stride through r4d_act_threads, same bf16
// round trip of the normalised value), and the rotate+quant body is
// r4d_paroquant_quant_fp8_row_lds_kernel's (same divide, same fmaf/transposed pair op, same
// whole-row amax, same clamp + cvt_pk_fp8_f32). A device selftest gates the fused op against the
// standalone producer over the same bf16 input, byte for byte.
//
// LDS. The norm path keeps the bf16 normalised row in shared memory (K*2 bytes) and a scratch f32
// copy (K*4) it rotates in place per partition, so it serves a K whose K*6 fits the 32 KiB budget --
// every norm-fed projection on this model is n_embd = 5120 (30 KiB). The gated path does not stage
// the row: its input can be n_ff = 17408, so it walks one 128-group at a time and recomputes the
// product for the amax pass (the two-pass producer's shape).

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_common.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kGroup = 128;    /* the ParoQuant rotation-group width */
constexpr float    kTop = 448.0f;   /* E4M3's largest finite value */
constexpr int64_t  kLdsCap = 32 * 1024;
constexpr int      kMaxP = 3;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct PQPart {
    const int32_t* pairs;
    const float*   angle;
    const float*   tscale;
    uint8_t*       q;
    float*         s;
};
struct PQPartSet { PQPart p[kMaxP]; };

/* The whole-block max over THR threads, the reduction r4d_fused_quant_fp8_row.hip's block_max uses.
 * A max is exact and order-independent, so this partition is free. */
template <uint32_t THR>
__device__ __forceinline__ float pq_block_max(float m, float* red) {
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

/* Apply one partition's Givens stages to an f32 row staged whole in `g`, ascending and transposed,
 * exactly as the standalone LDS producer does. The groups are independent, so each runs on its own
 * warp with `__syncwarp` a stage. `wrote` receives the partition's whole-row scale. */
template <uint32_t THR>
__device__ __forceinline__ void pq_rotate_lds(float* g, uint32_t K, uint32_t S, int32_t k_base,
                                              uint32_t pair_ld, uint32_t angle_ld,
                                              const PQPart& pt, float* red, uint8_t* qrow,
                                              float* srow, bool rot) {
    const uint32_t G = K / kGroup;
    const uint32_t lane = threadIdx.x & (kWave - 1), warp = threadIdx.x / kWave;
    const uint32_t warps = THR / kWave;
    for (uint32_t grp = warp; rot && grp < G; grp += warps) {
        for (uint32_t st = 0; st < S; ++st) {
            const int32_t* pr = pt.pairs + (size_t)st * pair_ld + grp * (kGroup / 2) * 2;
            const float*   an = pt.angle + (size_t)st * angle_ld + grp * (kGroup / 2);
#pragma unroll
            for (uint32_t t2 = 0; t2 < 2; ++t2) {
                const uint32_t t = lane + kWave * t2;
                const uint32_t a = (uint32_t)pr[2 * t] - k_base,
                               b = (uint32_t)pr[2 * t + 1] - k_base;
                float c, sn;
                __sincosf(an[t], &sn, &c);
                const float xa = g[a], xb = g[b];
                g[a] = fmaf(c, xa, sn * xb);
                g[b] = fmaf(c, xb, -sn * xa);
            }
            __syncwarp();
        }
    }
    /* The groups are spread over the warps, so the amax and the quantiser read columns other warps
     * rotated: a block barrier here, exactly as the standalone LDS producer places one. */
    __syncthreads();
    /* The scale is written before the codes so `pq_quant_lds` can read the same reduction. */
    float m = 0.0f;
    for (uint32_t k = threadIdx.x; k < K; k += THR) m = fmaxf(m, fabsf(g[k]));
    const float amax = pq_block_max<THR>(m, red);
    const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
    if (threadIdx.x == 0) srow[0] = sc;
    const float inv = 1.0f / sc;
    for (uint32_t k = threadIdx.x; k < K; k += THR) {
        const float v = fminf(fmaxf(g[k] * inv, -kTop), kTop);
        const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
        qrow[k] = (uint8_t)pk;
    }
}

/* ------------------------------------------------------------ norm -> P rotated activations
 * One block a row. Pass one is the sum of squares (rmsnorm_quant_fp8_row_kernel's order); pass two
 * writes the bf16 normalised row into LDS; then each partition rotates that row in place and
 * quantises it. */
template <bool WF32, bool RES, uint32_t THR>
__global__ __launch_bounds__(THR) void paroquant_rmsnorm_quant_fp8_row_kernel(
    const uint16_t* __restrict__ x, uint16_t* res, const void* __restrict__ w, uint16_t* y,
    PQPartSet parts, uint32_t K, uint32_t S, int32_t k_base, uint32_t P, uint32_t x_ld, uint32_t r_ld,
    uint32_t y_ld, uint32_t pair_ld, uint32_t angle_ld, uint32_t ts_ld, uint32_t q_ld, uint32_t s_ld,
    float eps, float wadd, uint32_t no_rot) {
    extern __shared__ char sm[];
    uint16_t* nb = (uint16_t*)sm;                 /* [K] bf16 normalised row */
    float*    g  = (float*)(sm + (size_t)K * 2);  /* [K] f32 scratch, rotated per partition */
    __shared__ float red[THR / kWave];
    __shared__ float lds[THR / kWave];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (no_rot >= 2) return;   /* DIAGNOSTIC: launches, does nothing -- isolates launch vs work */
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
    const uint16_t* __restrict__ nr = RES ? rr : xr;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;

    constexpr uint32_t kU = 4;
    float ss = 0.0f;
    for (uint32_t i0 = tid; i0 < K; i0 += kU * THR) {
        uint16_t xa[kU], ra[kU];
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR < K ? i0 + u * THR : i0;
            xa[u] = xr[i];
            if (RES) ra[u] = rr[i];
        }
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR;
            if (i < K) {
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
    ss = r4d_wave_sum(ss);
    {
        const uint32_t lane = tid & (kWave - 1), wave = tid / kWave;
        if (lane == 0) lds[wave] = ss;
        __syncthreads();
        float s = 0.0f;
#pragma unroll
        for (uint32_t w2 = 0; w2 < THR / kWave; ++w2) s += lds[w2];
        ss = s;
    }
    const float rs = rsqrtf(ss / (float)K + eps);

    for (uint32_t c0 = tid * 4u; c0 < K; c0 += THR * 4u) {
        if (c0 < K) {
            float gg[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) gg[j] = WF32 ? ((const float*)w)[c0 + j]
                                                     : bf2f(((const uint16_t*)w)[c0 + j]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                nb[c0 + j] = f2bf(bf2f(nr[c0 + j]) * rs * (gg[j] + wadd));
            if (yr) {
#pragma unroll
                for (int j = 0; j < 4; ++j) yr[c0 + j] = nb[c0 + j];
            }
        }
    }
    __syncthreads();

    for (uint32_t p = 0; p < P; ++p) {
        const PQPart& pt = parts.p[p];
        for (uint32_t k = tid; k < K; k += THR) g[k] = bf2f(nb[k]) / pt.tscale[k];
        __syncthreads();
        uint8_t* qrow = pt.q + (size_t)row * q_ld;
        float*   srow = pt.s + (size_t)row * s_ld;
        pq_rotate_lds<THR>(g, K, S, k_base, pair_ld, angle_ld, pt, red, qrow, srow, !no_rot);
        __syncthreads();
    }
}

/* ------------------------------------------------------------ gated product -> P rotated activations
 * silu(gate)*up or sigmoid(gate)*up, then each partition's rotate+quant. The row is walked one
 * 128-group at a time and the product recomputed for the second pass, so n may exceed LDS. */
template <int ACT, uint32_t THR>
__global__ __launch_bounds__(THR) void paroquant_gated_quant_fp8_row_kernel(
    const uint16_t* __restrict__ gu, const uint16_t* __restrict__ gu2, PQPartSet parts, uint32_t n,
    uint32_t S, int32_t k_base, uint32_t P, uint32_t gu_ld, uint32_t gu2_ld, uint32_t pair_ld,
    uint32_t angle_ld, uint32_t ts_ld, uint32_t q_ld, uint32_t s_ld) {
    __shared__ float g[kGroup];
    __shared__ float red[THR / kWave];
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    /* TWO INPUTS. The gate is always `gu`; `up` is either the second half of the same [gate|up] row
     * (the merged form) or a separate [M,n] buffer (`gu2`), which is what an MLP's two GEMMs write
     * when there is no reason to `cast` them into one buffer first. */
    const uint16_t* __restrict__ gr2 = gu2 ? gu2 + (size_t)row * gu2_ld : nullptr;
    const uint32_t G = n / kGroup;

    /* The bf16 product value the standalone gated producer writes and the rotator reads: silu or
     * sigmoid of the gate, multiplied by up, rounded to bf16. */
    auto product = [&](uint32_t c) -> float {
        const float gv = bf2f(gr[c]);
        const float a = ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv));
        const float uv = gr2 ? bf2f(gr2[c]) : bf2f(gr[n + c]);
        return bf2f(f2bf(a * uv));
    };

    for (uint32_t p = 0; p < P; ++p) {
        const PQPart& pt = parts.p[p];
        uint8_t* qrow = pt.q + (size_t)row * q_ld;
        float*   srow = pt.s + (size_t)row * s_ld;
        /* pass 1: rotate and keep the whole-row amax. */
        float m = 0.0f;
        for (uint32_t gi = 0; gi < G; ++gi) {
            const uint32_t b0 = gi * kGroup;
            if (tid < kGroup) g[tid] = product(b0 + tid) / pt.tscale[b0 + tid];
            __syncthreads();
            for (uint32_t st = 0; st < S; ++st) {
                const int32_t* pr = pt.pairs + (size_t)st * pair_ld + gi * kGroup;
                const float*   an = pt.angle + (size_t)st * angle_ld + gi * (kGroup / 2);
                for (uint32_t q = tid; q < kGroup / 2; q += THR) {
                    const int32_t a = pr[2 * q] - k_base - (int32_t)b0,
                                  b = pr[2 * q + 1] - k_base - (int32_t)b0;
                    const float t = an[q]; float c, sn; __sincosf(t, &sn, &c);
                    const float xa = g[a], xb = g[b];
                    g[a] = c * xa + sn * xb;
                    g[b] = -sn * xa + c * xb;
                }
                __syncthreads();
            }
            if (tid < kGroup) m = fmaxf(m, fabsf(g[tid]));
            __syncthreads();
        }
        const float amax = pq_block_max<THR>(m, red);
        const float sc = amax > 0.0f ? amax * (1.0f / kTop) : 1.0f;
        if (tid == 0) srow[0] = sc;
        const float inv = 1.0f / sc;
        /* pass 2: rotate again and emit the codes over the bf16-rounded product. */
        for (uint32_t gi = 0; gi < G; ++gi) {
            const uint32_t b0 = gi * kGroup;
            if (tid < kGroup) g[tid] = product(b0 + tid) / pt.tscale[b0 + tid];
            __syncthreads();
            for (uint32_t st = 0; st < S; ++st) {
                const int32_t* pr = pt.pairs + (size_t)st * pair_ld + gi * kGroup;
                const float*   an = pt.angle + (size_t)st * angle_ld + gi * (kGroup / 2);
                for (uint32_t q = tid; q < kGroup / 2; q += THR) {
                    const int32_t a = pr[2 * q] - k_base - (int32_t)b0,
                                  b = pr[2 * q + 1] - k_base - (int32_t)b0;
                    const float t = an[q]; float c, sn; __sincosf(t, &sn, &c);
                    const float xa = g[a], xb = g[b];
                    g[a] = c * xa + sn * xb;
                    g[b] = -sn * xa + c * xb;
                }
                __syncthreads();
            }
            if (tid < kGroup) {
                const float v = fminf(fmaxf(g[tid] * inv, -kTop), kTop);
                const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
                qrow[b0 + tid] = (uint8_t)pk;
            }
            __syncthreads();
        }
    }
}

}  // namespace

// ============================================================== the norm-fed producer
//
//   x       [M][K]     bf16, `x_ld` a row (0 = K)
//   res     [M][K]     bf16, `r_ld` a row, optional, READ AND WRITTEN
//   w       [K]        f32 or bf16 norm gain
//   parts.p {pairs [S][K] i32, angle [S][K/2] f32, t_scale [1][K] f32, q [M][K] e4m3, s [M][1] f32}
//                      P partitions, all [S][K]-indexed against the SAME K
//   P       partitions (<= 3); K*6 bytes of LDS, so K <= 5461
static int paroquant_rmsnorm_quant_fp8_row_launch(long x, long res, long w, long y, const PQPartSet* set,
                                                  int M, int K, int S, int k_base, int P, int x_ld,
                                                  int r_ld, int y_ld, int pair_ld, int angle_ld,
                                                  int ts_ld, int q_ld, int s_ld, float eps, float wadd,
                                                  int no_rot, int w_f32, const R4dSink& sk) {
    if (!x || !w || !set || M <= 0 || K <= 0 || S <= 0 || P <= 0 || P > kMaxP) return -1;
    if (K % (int)kGroup) return -3;
    if ((int64_t)K * 6 > kLdsCap) return -3;
    if (!x_ld) x_ld = K;
    if (!r_ld) r_ld = K;
    if (!y_ld) y_ld = K;
    if (!pair_ld) pair_ld = K;
    if (!angle_ld) angle_ld = K / 2;
    if (!ts_ld) ts_ld = K;
    if (!q_ld) q_ld = K;
    if (!s_ld) s_ld = 1;
    const int thr = r4d_act_threads(M, K);
    const size_t lds = (size_t)K * 6;
    int emit_rc = RAD_OK;
#define R4D_PQR_T(wf, rs, T)                                                                    \
    emit_rc = R4D_EMIT(sk, (paroquant_rmsnorm_quant_fp8_row_kernel<wf, rs, T>),                  \
                       dim3((uint32_t)M), dim3(T), lds, (const uint16_t*)x, (uint16_t*)res,      \
                       (const void*)w, (uint16_t*)y, *set, (uint32_t)K, (uint32_t)S,             \
                       (int32_t)k_base, (uint32_t)P, (uint32_t)x_ld, (uint32_t)r_ld,             \
                       (uint32_t)y_ld, (uint32_t)pair_ld, (uint32_t)angle_ld, (uint32_t)ts_ld,    \
                       (uint32_t)q_ld, (uint32_t)s_ld, eps, wadd, (uint32_t)no_rot)
#define R4D_PQR(wf, rs)                                                                         \
    do { if (thr == 1024)      R4D_PQR_T(wf, rs, 1024);                                          \
         else if (thr == 512)  R4D_PQR_T(wf, rs, 512);                                           \
         else                  R4D_PQR_T(wf, rs, 256); } while (0)
    if (w_f32) { if (res) R4D_PQR(true, true);  else R4D_PQR(true, false); }
    else       { if (res) R4D_PQR(false, true); else R4D_PQR(false, false); }
#undef R4D_PQR
#undef R4D_PQR_T
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// ============================================================== the gated-fed producer
//
//   gu      [M][2n]    bf16, gate first (at column 0) then up (at column n), `gu_ld` a row (0 = 2n)
//   gu2     [M][n]     optional bf16 up in its OWN buffer (then `gu` is only [M][n] of gate)
//   parts.p as above, width n
//   P       partitions (<= 3); one 128-group of LDS, so n may exceed the LDS budget
static int paroquant_gated_quant_fp8_row_launch(long gu, long gu2, const PQPartSet* set, int M,
                                                int n, int S, int k_base, int P, int gu_ld,
                                                int gu2_ld, int pair_ld, int angle_ld, int ts_ld,
                                                int q_ld, int s_ld, int mode, const R4dSink& sk) {
    if (!gu || !set || M <= 0 || n <= 0 || S <= 0 || P <= 0 || P > kMaxP) return -1;
    if (n % (int)kGroup) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (!gu_ld) gu_ld = gu2 ? n : 2 * n;
    if (!gu2_ld) gu2_ld = n;
    if (!pair_ld) pair_ld = n;
    if (!angle_ld) angle_ld = n / 2;
    if (!ts_ld) ts_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = 1;
    const int thr = r4d_act_threads(M, n);
    int emit_rc = RAD_OK;
#define R4D_PGQ_T(a, T)                                                                         \
    emit_rc = R4D_EMIT(sk, (paroquant_gated_quant_fp8_row_kernel<a, T>), dim3((uint32_t)M),       \
                       dim3(T), 0, (const uint16_t*)gu, (const uint16_t*)gu2, *set, (uint32_t)n,  \
                       (uint32_t)S, (int32_t)k_base, (uint32_t)P, (uint32_t)gu_ld,                \
                       (uint32_t)gu2_ld, (uint32_t)pair_ld, (uint32_t)angle_ld, (uint32_t)ts_ld,  \
                       (uint32_t)q_ld, (uint32_t)s_ld)
#define R4D_PGQ(a)                                                                              \
    do { if (thr == 1024)      R4D_PGQ_T(a, 1024);                                               \
         else if (thr == 512)  R4D_PGQ_T(a, 512);                                                \
         else                  R4D_PGQ_T(a, 256); } while (0)
    if (mode == 0) R4D_PGQ(0);
    else           R4D_PGQ(1);
#undef R4D_PGQ
#undef R4D_PGQ_T
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

// ============================================================== the plugin ABI entry points
//
// Operand list (norm): x, res?, w, then P x {pairs, angle, t_scale}, then P x {q, scale}.
// Operand list (gated): gu, then P x {pairs, angle, t_scale}, then P x {q, scale}.
enum { PQR_X = 0, PQR_RES = 1, PQR_W = 2, PQR_W0 = 3 };
enum { PGQ_GU = 0, PGQ_W0 = 1 };

/* Read the P partitions' weight views and output views out of a positional operand list. The
 * weight base is `wb` and the output base `wb + 3P`. */
static bool pq_read_parts(const RadArgs* a, PQPartSet* set, int P, int wb, long long* pair_ld,
                          long long* angle_ld, long long* ts_ld, long long* q_ld, long long* s_ld) {
    for (int p = 0; p < P; ++p) {
        const RadTensor* pr = r4d_opt(a, wb + 3 * p + 0);
        const RadTensor* an = r4d_opt(a, wb + 3 * p + 1);
        const RadTensor* ts = r4d_opt(a, wb + 3 * p + 2);
        const RadTensor* q  = r4d_opt(a, wb + 3 * P + 2 * p + 0);
        const RadTensor* sc = r4d_opt(a, wb + 3 * P + 2 * p + 1);
        if (!pr || !an || !ts || !q || !sc) return false;
        if (pr->rank != 2 || an->rank != 2) return false;
        set->p[p].pairs  = (const int32_t*)r4d_addr(pr);
        set->p[p].angle  = (const float*)r4d_addr(an);
        set->p[p].tscale = (const float*)r4d_addr(ts);
        set->p[p].q      = (uint8_t*)r4d_addr(q);
        set->p[p].s      = (float*)r4d_addr(sc);
        *pair_ld  = r4d_row_pitch(pr);
        *angle_ld = r4d_row_pitch(an);
        *ts_ld    = r4d_row_pitch(ts);
        *q_ld     = r4d_row_pitch(q);
        *s_ld     = r4d_row_pitch(sc);
    }
    return true;
}

static int rad_paroquant_rmsnorm_quant_fp8_row(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x  = r4d_opt(a, PQR_X);
    const RadTensor* rs = r4d_opt(a, PQR_RES);
    const RadTensor* w  = r4d_opt(a, PQR_W);
    if (!x || !w || x->rank != 2) return R4D_NO(RAD_E_INVAL);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long M = x->shape[0], K = x->shape[1];
    if (M < 1 || K < (long long)kGroup || K % (long long)kGroup) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(w) != K) return R4D_NO(RAD_E_SHAPE);
    long long P = 0, S = 0, kb = 0;
    if (!r4d_geti(a, "P", &P) || P < 1 || P > kMaxP) return R4D_NO(RAD_E_INVAL);
    if (!r4d_geti(a, "stages", &S) || S < 1) return R4D_NO(RAD_E_INVAL);
    r4d_geti(a, "k_base", &kb);
    /* Every partition's tables are [S][K]-shaped against the same K; the outputs are [M][K]. Validate
     * against the first partition -- pq_read_parts passes the rest through positionally. */
    const RadTensor* p0 = r4d_opt(a, PQR_W0);
    const RadTensor* q0 = r4d_opt(a, PQR_W0 + 3 * (int)P);
    if (!p0 || !q0 || p0->rank != 2 || p0->shape[0] != S || p0->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (q0->shape[0] != M || q0->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    PQPartSet set{};
    long long pair_ld = K, angle_ld = K / 2, ts_ld = K, q_ld = K, s_ld = 1;
    if (!pq_read_parts(a, &set, (int)P, PQR_W0, &pair_ld, &angle_ld, &ts_ld, &q_ld, &s_ld))
        return R4D_NO(RAD_E_INVAL);
    const long long xp = r4d_row_pitch(x), rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || rp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The norm's bf16 output is an OPTIONAL trailing operand, so a consumer that reads the normed
     * row (the delta net's a/b gate) still finds it where the unfused norm left it. */
    const RadTensor* y = r4d_opt(a, PQR_W0 + 3 * (int)P + 2 * (int)P);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    const int e = paroquant_rmsnorm_quant_fp8_row_launch(
        r4d_addr(x), rs ? r4d_addr(rs) : 0, r4d_addr(w), y ? r4d_addr(y) : 0, &set, (int)M, (int)K,
        (int)S, (int)kb, (int)P, (int)xp, (int)rp, (int)yp, (int)pair_ld, (int)angle_ld,
        (int)ts_ld, (int)q_ld, (int)s_ld, (float)r4d_getf(a, "eps", 1e-6),
        (float)r4d_getf(a, "wadd", 0.0), (int)r4d_geti_or(a, "no_rot", 0), w->dtype == RAD_F32, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_paroquant_gated_quant_fp8_row(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* gu = r4d_opt(a, PGQ_GU);
    if (!gu || gu->rank != 2) return R4D_NO(RAD_E_INVAL);
    const long long M = gu->shape[0];
    long long P = 0, S = 0, kb = 0;
    if (!r4d_geti(a, "P", &P) || P < 1 || P > kMaxP) return R4D_NO(RAD_E_INVAL);
    if (!r4d_geti(a, "stages", &S) || S < 1) return R4D_NO(RAD_E_INVAL);
    r4d_geti(a, "k_base", &kb);
    /* `n` is the partition width (the product's width). `up` is the second half of `gu` when the
     * optional `gu2` is absent, and its own [M][n] buffer when it is present (an MLP's two GEMMs). */
    const RadTensor* q0 = r4d_opt(a, PGQ_W0 + 3 * (int)P);
    if (!q0 || q0->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long n = q0->shape[1];
    const RadTensor* gu2 = r4d_opt(a, PGQ_W0 + 3 * (int)P + 2 * (int)P);
    if (gu2) {
        if (gu->shape[1] != n || gu2->rank != 2 || gu2->shape[0] != M || gu2->shape[1] != n)
            return R4D_NO(RAD_E_SHAPE);
    } else if (gu->shape[1] != 2 * n) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (n % (long long)kGroup) return R4D_NO(RAD_E_SHAPE);
    PQPartSet set{};
    long long pair_ld = n, angle_ld = n / 2, ts_ld = n, q_ld = n, s_ld = 1;
    if (!pq_read_parts(a, &set, (int)P, PGQ_W0, &pair_ld, &angle_ld, &ts_ld, &q_ld, &s_ld))
        return R4D_NO(RAD_E_INVAL);
    const char* act = r4d_gets(a, "act");
    int mode = (int)r4d_geti_or(a, "mode", 0);
    if (act) {
        if (!std::strcmp(act, "silu")) mode = 0;
        else if (!std::strcmp(act, "sigmoid")) mode = 1;
        else return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const int e = paroquant_gated_quant_fp8_row_launch(
        r4d_addr(gu), gu2 ? r4d_addr(gu2) : 0, &set, (int)M, (int)n, (int)S, (int)kb, (int)P,
        (int)r4d_row_pitch(gu), gu2 ? (int)r4d_row_pitch(gu2) : 0, (int)pair_ld, (int)angle_ld,
        (int)ts_ld, (int)q_ld, (int)s_ld, mode, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_pqr_body(const RadArgs* a, const R4dSink& sk) {
    return rad_paroquant_rmsnorm_quant_fp8_row(a, sk);
}
static int rad_pgq_body(const RadArgs* a, const R4dSink& sk) {
    return rad_paroquant_gated_quant_fp8_row(a, sk);
}
R4D_ENTRY(r4d_rad_paroquant_rmsnorm_quant_fp8_row, rad_pqr_body)
R4D_ENTRY(r4d_rad_paroquant_gated_quant_fp8_row, rad_pgq_body)
