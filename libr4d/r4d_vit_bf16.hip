// r4d_vit_bf16.hip -- the three ops a vision tower needs that no language model here does: a layer
// norm with a bias, the learned-grid position embedding, and a biased GEMM with the linear layer's
// activation and residual folded into its store.
//
// THE ARITHMETIC IS libref's (ref_elementwise.cpp, ref_gemm.cpp), and where libref rounds, this
// rounds in the same place: every accumulation is f32, every store is round-to-nearest-even bf16,
// and the GEMM's epilogue rounds after the biased product, again after the activation and again
// after the residual -- the three tensors the unfused chain would have written. A fused form that
// kept f32 across those seams would be a little more accurate and a different function.
//
// WHY THE GEMM IS NOT gemm_bf16_nt_tiled. That kernel is byte-identical to the skinny one by
// construction (split-K slices summed in a fixed order) and serves K in whole 64-element tiles; a
// tower's MLP projects to 4304 and back, which is 16 elements past a tile, and its linear layers
// all carry a bias. So this is the same block shape -- eight waves over an LDS-staged 64-deep K
// tile, the next tile prefetched into registers -- with a K tail staged as zeros, no split, and the
// epilogue in the store. It serves tall activations: a tower runs one pass over thousands of
// patches, never a decode row.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#include <cstdint>

namespace {

constexpr int kThr = 256;

__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    __syncthreads();
    return s;
}

template <bool F32>
__device__ __forceinline__ float ldw(const void* p, uint32_t i) {
    return F32 ? ((const float*)p)[i] : bf2f(((const unsigned short*)p)[i]);
}

// ---------------------------------------------------------------------------- layernorm
// ref_layernorm: y = (x - mean) * rsqrt(var + eps) * w + b, biased variance, TWO PASSES over the
// row -- the mean first, then the squared deviations -- because the one-pass E[x^2] - E[x]^2 is
// catastrophically worse on a residual whose mean is large against its spread. One block a row.
template <bool WF32>
__global__ __launch_bounds__(kThr) void layernorm_kernel(
    const unsigned short* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b,
    unsigned short* __restrict__ y, uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kThr / 32];
    const uint32_t m = blockIdx.x;
    const unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    unsigned short* __restrict__ yr = y + (size_t)m * y_ld;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) s += bf2f(xr[i]);
    const float mean = block_sum<kThr / 32>(s, lds) / (float)n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        const float d = bf2f(xr[i]) - mean;
        ss += d * d;
    }
    const float r = rsqrtf(block_sum<kThr / 32>(ss, lds) / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float v = (bf2f(xr[i]) - mean) * r * ldw<WF32>(w, i);
        if (b) v += ldw<WF32>(b, i);
        yr[i] = f32_to_bf16(v);
    }
}

// ---------------------------------------------------------------------------- grid_embed
// ref_grid_embed: a side x side table of rows resampled bilinearly to row m's grid (corners on the
// table's corners, edge taps clamped), the four taps summed in f32 in libref's order, rounded,
// added to x and rounded again. `coord` is [4, planes] -- row, column, grid height, grid width.
template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    const uint32_t m = blockIdx.x;
    const int g[4] = { crd[m], crd[plane + m], crd[2 * plane + m], crd[3 * plane + m] };
    int th[2], tw[2];
    float wh[2], ww[2];
    auto axis = [&](int idx, int size, int* t, float* wt) {
        const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
        const float fl = floorf(src);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int tap = (int)fl + k;
            t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
            const float d = fabsf(src - fl - (float)k);
            wt[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
        }
    };
    axis(g[0], g[2], th, wh);
    axis(g[1], g[3], tw, ww);
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
#pragma unroll
            for (int kj = 0; kj < 2; ++kj)
                acc += ldw<TF32>(tab, (uint32_t)(th[ki] * side + tw[kj]) * n + i) *
                       (wh[ki] * ww[kj]);
        const float pe = bf2f(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// Half-pixel antialiased bilinear interpolation, used by SigLIP2. A table is a tensor in
// its own dtype: both filter-weight stages and each separable pass narrow to that dtype.
// This matters for BF16 even when upsampling, where the footprint is only two samples.
template <bool TF32>
__device__ __forceinline__ float grid_narrow(float v) {
    return TF32 ? v : bf2f(f32_to_bf16(v));
}
template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_antialias_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    extern __shared__ float weights[];
    float* wh = weights;
    float* ww = weights + side;
    const uint32_t m = blockIdx.x;
    if (threadIdx.x == 0) {
        for (int axis = 0; axis < 2; ++axis) {
            const int size = max(1, crd[(axis + 2) * plane + m]);
            const int at = max(0, min(size - 1, crd[axis * plane + m]));
            const float scale = (float)side / (float)size;
            const float support = fmaxf(scale, 1.0f), inv = 1.0f / support;
            const float center = scale * ((float)at + 0.5f);
            const int first = max(0, (int)(center - support + 0.5f));
            const int end = min(side, (int)(center + support + 0.5f));
            float* w = weights + axis * side;
            float total = 0.0f;
            for (int j = 0; j < side; ++j) w[j] = 0.0f;
            for (int j = first; j < end; ++j) {
                const float d = fabsf(((float)(j - first) + ((float)first - center) + 0.5f) * inv);
                const float value = fmaxf(0.0f, 1.0f - d);
                w[j] = grid_narrow<TF32>(value);
                total += value;
            }
            for (int j = first; j < end; ++j) w[j] = grid_narrow<TF32>(w[j] / total);
        }
    }
    __syncthreads();
    unsigned short* xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float value = 0.0f;
        for (int y = 0; y < side; ++y) {
            if (wh[y] == 0.0f) continue;
            float horizontal = 0.0f;
            for (int col = 0; col < side; ++col) {
                if (ww[col] == 0.0f) continue;
                horizontal += ldw<TF32>(tab, (uint32_t)(y * side + col) * n + i) * ww[col];
            }
            value += grid_narrow<TF32>(horizontal) * wh[y];
        }
        const float pe = bf2f(f32_to_bf16(grid_narrow<TF32>(value)));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- gemm_nt_bias
#define VB_BK 64
#define VB_PITCH (VB_BK + 8)      // 144-byte LDS rows: gemm_bf16_nt_tiled's bank argument

// TM x TN WMMA tiles a wave, eight waves as 2 x 4: BM = 2*TM*16, BN = 4*TN*16. ACT: 0 none,
// 1 exact GELU, 2 tanh GELU.
template <int TM, int TN, int ACT, bool RES, bool BF32>
__global__ __launch_bounds__(256) void vit_gemm_kernel(
    const unsigned short* __restrict__ A, unsigned lda, const unsigned short* __restrict__ W,
    unsigned ldw_, const void* __restrict__ bias, const unsigned short* R, unsigned ldr,
    unsigned short* C, unsigned ldc, int M, int K, int N) {
    constexpr int BM = 2 * TM * 16, BN = 4 * TN * 16;
    constexpr int CPR = VB_BK / 8;                      // 16-byte chunks a staged row
    constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
    static_assert(AL >= 1 && WL >= 1 && BM * CPR % 256 == 0 && BN * CPR % 256 == 0,
                  "a tile is whole 16-byte chunks for every thread");
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * VB_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * VB_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / 4, wn = wave % 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // Staging: chunk c = tid + 256 i is row c / CPR, elements (c % CPR) * 8 of it. Rows past M
    // or N re-read the last valid row and compute a result nobody stores.
    unsigned aoff[AL], woff[WL];
    int ae[AL], we[WL];
    unsigned short* adst[AL];
    unsigned short* wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = m0 + r < M ? m0 + r : M - 1;
        aoff[i] = (unsigned)gr * lda + (unsigned)e;
        ae[i] = e;
        adst[i] = sA + r * VB_PITCH + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = n0 + r < N ? n0 + r : N - 1;
        woff[i] = (unsigned)gr * ldw_ + (unsigned)e;
        we[i] = e;
        wdst[i] = sW + r * VB_PITCH + e;
    }
    const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * VB_PITCH + 4 * (lane >> 4);
    const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * VB_PITCH + 4 * (lane >> 4);

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

    // THE K TAIL IS STAGED AS ZEROS: K is a multiple of 8, so a 16-byte chunk is wholly inside K
    // or wholly past it, and a zero chunk contributes exactly nothing to the dot products.
    const uint4 zero = make_uint4(0, 0, 0, 0);
    uint4 ra[AL], rw[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = ae[i] < K ? *(const uint4*)(A + aoff[i]) : zero;
#pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = we[i] < K ? *(const uint4*)(W + woff[i]) : zero;

    const int nk = (K + VB_BK - 1) / VB_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < AL; ++i) *(uint4*)adst[i] = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) *(uint4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            const int k1 = (kt + 1) * VB_BK;
#pragma unroll
            for (int i = 0; i < AL; ++i)
                ra[i] = k1 + ae[i] < K ? *(const uint4*)(A + aoff[i] + k1) : zero;
#pragma unroll
            for (int i = 0; i < WL; ++i)
                rw[i] = k1 + we[i] < K ? *(const uint4*)(W + woff[i] + k1) : zero;
        }
#pragma unroll
        for (int s = 0; s < VB_BK / 16; ++s) {
            v8bf af[TM], bfr[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * VB_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bfr[j] = fragP(fw + j * 16 * VB_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bfr[j], acc[i][j]);
        }
    }

    // Lane l holds column (l & 15) of a tile and rows 8 (l >> 4) + e. The residual is read at the
    // element it is added to, by the thread that stores it, so `R` may be `C` itself.
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
            if (n >= N) continue;
            const float bv = ldw<BF32>(bias, (uint32_t)n);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + e;
                if (m >= M) continue;
                unsigned short t = f32_to_bf16(acc[i][j][e] + bv);
                if constexpr (ACT == 1) t = f32_to_bf16(gelu_erf(bf2f(t)));
                if constexpr (ACT == 2) t = f32_to_bf16(gelu_tanh(bf2f(t)));
                if constexpr (RES) t = f32_to_bf16(bf2f(R[(size_t)m * ldr + n]) + bf2f(t));
                C[(size_t)m * ldc + n] = t;
            }
        }
}

}  // namespace

// ============================================================== the plugin ABI entry points

/* ------------------------------------------------------------------ layernorm */
enum { LN_X = 0, LN_W, LN_B, LN_Y };

static int rad_layernorm_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x = r4d_opt(a, LN_X);
    const RadTensor* w = r4d_opt(a, LN_W);
    const RadTensor* b = r4d_opt(a, LN_B);
    const RadTensor* y = r4d_opt(a, LN_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    /* The gain and the bias share one width, read off the operand: f32 or bf16. */
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (b && b->dtype != w->dtype) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w) || (b && !r4d_contig(b))) return R4D_NO(RAD_E_STRIDE);

    const long long n = r4d_geti_or(a, "n", r4d_dim(w, (int)w->rank - 1));
    if (n <= 0 || r4d_numel(w) < n || (b && r4d_numel(b) < n)) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    const long long y_ld = r4d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-5);
    if (w->dtype == RAD_F32)
        return R4D_EMIT(sk, layernorm_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (const unsigned short*)r4d_p(x), r4d_p(w), r4d_p(b),
                        (unsigned short*)r4d_wp(y), (uint32_t)n, eps, (uint32_t)x_ld,
                        (uint32_t)y_ld);
    return R4D_EMIT(sk, layernorm_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (const unsigned short*)r4d_p(x), r4d_p(w), r4d_p(b),
                    (unsigned short*)r4d_wp(y), (uint32_t)n, eps, (uint32_t)x_ld, (uint32_t)y_ld);
}
R4D_ENTRY(r4d_rad_layernorm, rad_layernorm_body)

/* ------------------------------------------------------------------ grid_embed */
enum { GE_X = 0, GE_TAB, GE_CRD };

static int rad_grid_embed_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x   = r4d_opt(a, GE_X);
    const RadTensor* tab = r4d_opt(a, GE_TAB);
    const RadTensor* crd = r4d_opt(a, GE_CRD);
    if (!x || !tab || !crd) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(tab, (int)tab->rank - 1));
    const long long side = r4d_geti_or(a, "side", 0);
    const char* mode = r4d_gets(a, "mode");
    const bool antialias = mode && std::strcmp(mode, "bilinear_antialias") == 0;
    if (mode && !antialias && std::strcmp(mode, "bilinear") != 0) return R4D_NO(RAD_E_INVAL);
    if (n <= 0 || side <= 0 || r4d_numel(tab) < side * side * n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    /* Four planes of coordinates, dense: [4, planes] with planes >= M. */
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !r4d_contig(crd))
        return R4D_NO(RAD_E_SHAPE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    if (antialias) {
        // Only two side-long weight vectors are shared, independent of the patch count.
        if (side > 4096) return R4D_NO(RAD_E_SHAPE);
        const uint32_t shared = (uint32_t)(2 * side * sizeof(float));
        if (tab->dtype == RAD_F32)
            return R4D_EMIT(sk, grid_embed_antialias_kernel<true>, dim3((uint32_t)M), dim3(kThr), shared,
                            (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                            (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
        return R4D_EMIT(sk, grid_embed_antialias_kernel<false>, dim3((uint32_t)M), dim3(kThr), shared,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    }
    if (tab->dtype == RAD_F32)
        return R4D_EMIT(sk, grid_embed_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return R4D_EMIT(sk, grid_embed_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}
R4D_ENTRY(r4d_rad_grid_embed, rad_grid_embed_body)

/* ------------------------------------------------------------------ gemm_nt_bias */
enum { GB_A = 0, GB_W, GB_BIAS, GB_RES, GB_Y };

static int rad_gemm_nt_bias_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* A = r4d_opt(a, GB_A);
    const RadTensor* W = r4d_opt(a, GB_W);
    const RadTensor* B = r4d_opt(a, GB_BIAS);
    const RadTensor* R = r4d_opt(a, GB_RES);
    const RadTensor* Y = r4d_opt(a, GB_Y);
    if (!A || !W || !B || !Y) return R4D_NO(RAD_E_INVAL);
    if (A->dtype != RAD_BF16 || W->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_BF16 && B->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (R && R->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(W) || !r4d_contig(B)) return R4D_NO(RAD_E_STRIDE);

    const long long K = r4d_geti_or(a, "K", r4d_dim(W, (int)W->rank - 1));
    const long long N = r4d_geti_or(a, "N", r4d_numel(W) / (K > 0 ? K : 1));
    if (K <= 0 || N <= 0 || (K % 8) || r4d_numel(W) < N * K || r4d_numel(B) < N)
        return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mr = 0;
    const long long ldc = r4d_rows_pitch(Y, N, &M);
    const long long lda = r4d_rows_pitch(A, K, &Ma);
    const long long ldr = R ? r4d_rows_pitch(R, N, &Mr) : 0;
    if (ldc < 0 || lda < 0 || ldr < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || (R && Mr < M)) return R4D_NO(RAD_E_SHAPE);
    /* 16-byte staged chunks: the operand rows must start on 16-byte boundaries. */
    if ((lda % 8) || !r4d_aligned(r4d_p(A), 16) || !r4d_aligned(r4d_p(W), 16))
        return R4D_NO(RAD_E_STRIDE);
    if ((long long)M * lda >= (1ll << 32) || (long long)N * K >= (1ll << 32))
        return R4D_NO(RAD_E_SHAPE);

    const char* act_s = r4d_gets(a, "act");
    int act = 0;
    if (act_s && *act_s) {
        if (!std::strcmp(act_s, "gelu")) act = 1;
        else if (!std::strcmp(act_s, "gelu_tanh")) act = 2;
        else if (std::strcmp(act_s, "none")) return R4D_NO(RAD_E_SCHEMA);
    }
    const bool res = R != nullptr, bf32 = B->dtype == RAD_F32;

    /* The large tile once its grid reaches half the part, as gemm_bf16_nt_tiled chooses. */
    const long long g128 = ((N + 127) / 128) * ((M + 127) / 128);
    const bool big = g128 >= 32;
    const int bt = big ? 128 : 64;
    const dim3 grid((unsigned)((N + bt - 1) / bt), (unsigned)((M + bt - 1) / bt)), block(256);
    auto Ap = (const unsigned short*)r4d_p(A);
    auto Wp = (const unsigned short*)r4d_p(W);
    auto Rp = (const unsigned short*)r4d_p(R);
    auto Cp = (unsigned short*)r4d_wp(Y);
    const void* Bp = r4d_p(B);
#define VB_L(TM, TN, AC, RS, BF)                                                                  \
    return R4D_EMIT(sk, (vit_gemm_kernel<TM, TN, AC, RS, BF>), grid, block, 0, Ap, (unsigned)lda, \
                    Wp, (unsigned)K, Bp, Rp, (unsigned)ldr, Cp, (unsigned)ldc, (int)M, (int)K,     \
                    (int)N)
#define VB_ACT(TM, TN, RS, BF)                                                                    \
    do {                                                                                          \
        if (act == 1) VB_L(TM, TN, 1, RS, BF);                                                    \
        if (act == 2) VB_L(TM, TN, 2, RS, BF);                                                    \
        VB_L(TM, TN, 0, RS, BF);                                                                  \
    } while (0)
#define VB_RES(TM, TN, BF)                                                                        \
    do { if (res) VB_ACT(TM, TN, true, BF); VB_ACT(TM, TN, false, BF); } while (0)
    if (big) { if (bf32) VB_RES(4, 2, true); VB_RES(4, 2, false); }
    if (bf32) VB_RES(2, 1, true);
    VB_RES(2, 1, false);
#undef VB_RES
#undef VB_ACT
#undef VB_L
}
R4D_ENTRY(r4d_rad_gemm_nt_bias, rad_gemm_nt_bias_body)
