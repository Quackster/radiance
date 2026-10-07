// The small ops an encoder issues between its GEMMs and its attention: a gated activation over two
// operands, the QK(V)-norm-and-rotary preamble over a fused projection, mean pooling over
// cu_seqlens segments, and the bf16 -> f32 widening of a pooled row. r4d_model_bf16.hip's rules
// hold here too: each kernel equals a libref function, reductions accumulate in f32 (seg_mean in
// f64, as ref does) and only the store narrows, and the transcendental functions are the accurate
// libm forms.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float bf2f(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

// ---------------------------------------------------------------------------- act_mul
// ref_act_mul: y = round(act(a)) * b. The activation is rounded to bf16 before the product
// because the unfused pair stores it.
template <int ACT>
__device__ __forceinline__ float act_of(float g) {
    if (ACT == 0) return g / (1.0f + expf(-g));
    if (ACT == 1) return 0.5f * g * (1.0f + erff(g * 0.70710678118654752f));
    if (ACT == 3) return 1.0f / (1.0f + expf(-g));
    return 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}

// THE ROW AXIS IS GRID-STRIDED. A grid's y extent stops at 65535, and an elementwise op over the
// audio subsampler's (time, frequency) rows is 96000 of them: a launch of y = M runs nothing past
// row 65535. A row kernel takes M, launches min(M, kGridYMax) and walks the rest.
constexpr uint32_t kGridYMax = 65535;
static inline uint32_t grid_rows(int64_t M) {
    return (uint32_t)(M < (int64_t)kGridYMax ? M : (int64_t)kGridYMax);
}

template <int ACT>
__global__ __launch_bounds__(kThreads) void act_mul_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float g = bf2f(f32_to_bf16(act_of<ACT>(bf2f(a[(size_t)m * a_ld + i]))));
        y[(size_t)m * y_ld + i] = f32_to_bf16(g * bf2f(b[(size_t)m * b_ld + i]));
    }
}

// ---------------------------------------------------------------------------- seg_mean
// ref_seg_mean: row s of y is the mean of x's rows [cu[s], cu[s+1]), summed in f64 and rounded
// once. A workgroup is 32 columns by 8 row lanes; each lane sums every eighth row of its column
// and the eight partial sums meet in LDS. An empty segment writes zeros.
__global__ __launch_bounds__(kThreads) void seg_mean_kernel(
    const uint16_t* __restrict__ x, const int* __restrict__ cu, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    __shared__ double part[8][33];
    const uint32_t s = blockIdx.y;
    const uint32_t col = threadIdx.x & 31, rl = threadIdx.x >> 5;
    const uint32_t i = blockIdx.x * 32 + col;
    const int b0 = cu[s], b1 = cu[s + 1];
    double acc = 0.0;
    if (i < n)
        for (int r = b0 + (int)rl; r < b1; r += 8) acc += (double)bf2f(x[(size_t)r * x_ld + i]);
    part[rl][col] = acc;
    __syncthreads();
    if (rl == 0 && i < n) {
        double t = 0.0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) t += part[k][col];
        const int len = b1 - b0;
        y[(size_t)s * y_ld + i] = f32_to_bf16(len > 0 ? (float)(t / (double)len) : 0.0f);
    }
}

// ---------------------------------------------------------------------------- qk_norm_rope
// ref_qk_norm_rope over a fused [q heads | k heads | v heads] row, in place: each q and k head is
// RMS-normed against its gain and rounded to bf16 (the unfused pair stores the norm), then
// rotated; with v_norm each v head is RMS-normed with a unit gain and not rotated.
//
// ONE WORKGROUP A TOKEN, A WAVE A HEAD, THE HEAD IN REGISTERS. Lane l holds the head's elements
// l, l+32, l+64, ... (neox) or the pairs 2l, 2l+1, 2l+64, 2l+65, ... (gptj), which puts both
// halves of every rotary pair in the same lane: the norm is a wave sum and the rotation needs no
// exchange at all. The token's sin/cos are computed once into LDS for all its heads -- 4 KiB, so
// occupancy is not what LDS decides. A first form staged the token's whole fused row in LDS
// (12 KiB a workgroup at head_dim 512) and ran slower than a workgroup a (token, head).
//
// The layout needs head_dim a multiple of 32 and, for neox, rotary_dim / 2 a multiple of 32 (for
// gptj, head_dim a multiple of 64); the launcher refuses anything else.
constexpr uint32_t kQkThreads = 256;
constexpr uint32_t kQkMaxHalf = 512;    // rotary pairs a head
constexpr int      kQkMaxE    = 32;     // elements a lane: head_dim up to 1024

/* The `blocks` mode's sections: pairs a section, and the position component each one reads. */
struct QkBlocks {
    int n = 0;
    int sec[8] = {};
    int comp[8] = {};
};

// MODE: 0 neox, 1 gptj, 2 blocks. Positions are [M] for the first two and [C, M] (plane stride
// `pplane`) for blocks.
template <int MODE>
__global__ __launch_bounds__(kQkThreads) void qk_norm_rope_kernel(
    uint16_t* __restrict__ qkv, const int* __restrict__ pos, const uint16_t* __restrict__ qw,
    const uint16_t* __restrict__ kw, uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
    uint32_t rot, int v_norm, float theta, float eps, float rscale, float wadd, uint32_t ld,
    QkBlocks blk, uint32_t pplane) {
    __shared__ float cs_s[kQkMaxHalf], sn_s[kQkMaxHalf];
    const uint32_t m = blockIdx.x;
    const uint32_t n_rope = n_head + n_head_kv;
    const uint32_t n_norm = n_rope + (v_norm ? n_head_kv : 0);
    const uint32_t half = rot / 2;
    uint16_t* row = qkv + (size_t)m * ld;
    const float div = rscale != 0.0f ? rscale : 1.0f;

    if (MODE == 2) {
        // Pair i of section c is the (i - start_c)-th of its block, at theta^(-j / sec[c]).
        for (uint32_t i = threadIdx.x; i < half; i += kQkThreads) {
            int c = 0, s0 = 0;
            while (c + 1 < blk.n && (int)i >= s0 + blk.sec[c]) { s0 += blk.sec[c]; ++c; }
            const float p = (float)pos[(size_t)blk.comp[c] * pplane + m] / div;
            const float inv = 1.0f / powf(theta, (float)((int)i - s0) / (float)blk.sec[c]);
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
        }
    } else {
        const float p = (float)pos[m] / div;
        for (uint32_t i = threadIdx.x; i < half; i += kQkThreads) {
            const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rot);
            sincosf(p * inv, &sn_s[i], &cs_s[i]);
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int E = (int)(head_dim / 32);
    auto at = [&](int j) -> uint32_t {
        return MODE == 1 ? 64u * (uint32_t)(j >> 1) + 2u * lane + (uint32_t)(j & 1)
                         : lane + 32u * (uint32_t)j;
    };
    for (uint32_t hh = wave; hh < n_norm; hh += kQkThreads / 32) {
        uint16_t* h = row + hh * head_dim;
        float x[kQkMaxE];
        float ss = 0.0f;
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) { x[j] = bf2f(h[at(j)]); ss += x[j] * x[j]; }
        ss = r4d_wave_sum(ss);
        const float sc = 1.0f / sqrtf(ss / (float)head_dim + eps);
        const bool is_v = hh >= n_rope;
        const uint16_t* w = hh < n_head ? qw : kw;
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) {
                const float g = is_v ? 1.0f : bf2f(w[at(j)]) + wadd;
                x[j] = bf2f(f32_to_bf16(x[j] * sc * g));
            }
        if (!is_v) {
            if (MODE == 1) {
                #pragma unroll
                for (int j = 0; j < kQkMaxE; j += 2) {
                    const uint32_t i = at(j) / 2;
                    if (j < E && i < half) {
                        const float x0 = x[j], x1 = x[j + 1];
                        x[j]     = x0 * cs_s[i] - x1 * sn_s[i];
                        x[j + 1] = x0 * sn_s[i] + x1 * cs_s[i];
                    }
                }
            } else if (MODE == 0) {
                const int hj = (int)(half / 32);
                #pragma unroll
                for (int j = 0; j < kQkMaxE / 2; ++j)
                    if (j < hj) {
                        const uint32_t i = lane + 32u * (uint32_t)j;
                        const float x0 = x[j], x1 = x[j + hj];
                        x[j]      = x0 * cs_s[i] - x1 * sn_s[i];
                        x[j + hj] = x0 * sn_s[i] + x1 * cs_s[i];
                    }
            } else {
                // Each section's block starts on a 32-channel slice. A section of 16 pairs is
                // one slice, its partners in the other half-wave; a multiple of 32 pairs keeps
                // both halves of a pair in the lane. (The launcher refuses anything else.)
                int s0 = 0;
                for (int c = 0; c < blk.n; ++c) {
                    const int sec = blk.sec[c], j0 = s0 / 16;
                    if (sec == 16) {
                        #pragma unroll
                        for (int j = 0; j < kQkMaxE; ++j)
                            if (j == j0) {
                                const float mine = x[j], other = swap16(x[j]);
                                const uint32_t i = (uint32_t)s0 + (lane & 15u);
                                x[j] = lane < 16 ? mine * cs_s[i] - other * sn_s[i]
                                                 : other * sn_s[i] + mine * cs_s[i];
                            }
                    } else {
                        const int hj = sec / 32;
                        #pragma unroll
                        for (int j = 0; j < kQkMaxE; ++j)
                            if (j >= j0 && j < j0 + hj) {
                                const uint32_t i = (uint32_t)s0 + lane + 32u * (uint32_t)(j - j0);
                                const float x0 = x[j], x1 = x[j + hj];
                                x[j]      = x0 * cs_s[i] - x1 * sn_s[i];
                                x[j + hj] = x0 * sn_s[i] + x1 * cs_s[i];
                            }
                    }
                    s0 += sec;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < kQkMaxE; ++j)
            if (j < E) h[at(j)] = f32_to_bf16(x[j]);
    }
}

// ---------------------------------------------------------------------------- pool_rows
// ref_pool_rows: row r of y is the mean of x's rows [r*k, r*k + k), summed in f32 and rounded to
// bf16, then times `mult` and rounded again. A workgroup an output row, a thread a column.
__global__ __launch_bounds__(kThreads) void pool_rows_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t k, float mult,
    uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t r = blockIdx.y; r < M; r += gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < k; ++j) acc += bf2f(x[((size_t)r * k + j) * x_ld + i]);
        const float mean = bf2f(f32_to_bf16(acc * (1.0f / (float)k)));
        y[(size_t)r * y_ld + i] = f32_to_bf16(mean * mult);
    }
}

// ---------------------------------------------------------------------------- cast bf16 -> f32
__global__ __launch_bounds__(kThreads) void widen_kernel(
    const uint16_t* __restrict__ x, float* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y)
        y[(size_t)m * y_ld + i] = bf2f(x[(size_t)m * x_ld + i]);
}

// ---------------------------------------------------------------------------- the audio tower
// Gemma 4's USM conformer: an elementwise rectifier and clamp, the subsampler's segment-wise
// convolution, the light-conv block's causal depthwise convolution, the relative-position table,
// and the local attention with a relative term. An audio pass is at most a few thousand frames
// (750 tokens for 30 s), so each kernel is the plain form -- a workgroup a row, a thread an
// element -- and equal to its libref function in arithmetic: f32 throughout, one rounding.

// ref_relu.
__global__ __launch_bounds__(kThreads) void relu_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float v = bf2f(x[(size_t)m * x_ld + i]);
        y[(size_t)m * y_ld + i] = f32_to_bf16(v > 0.0f ? v : 0.0f);
    }
}

// ref_silu: x * sigmoid(x) with the sigmoid taken first, libref's act_silu term for term. The
// audio tower's feed-forward issues it on its own between two clipped linears.
__global__ __launch_bounds__(kThreads) void silu_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        const float v = bf2f(x[(size_t)m * x_ld + i]);
        y[(size_t)m * y_ld + i] = f32_to_bf16(v * (1.0f / (1.0f + expf(-v))));
    }
}

// ref_clamp: the two bounds are one-element weights, bf16 or f32.
__global__ __launch_bounds__(kThreads) void clamp_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ lo, const void* __restrict__ hi,
    int b_f32, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld, uint32_t M) {
    const uint32_t i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const float l = b_f32 ? ((const float*)lo)[0] : bf2f(((const uint16_t*)lo)[0]);
    const float h = b_f32 ? ((const float*)hi)[0] : bf2f(((const uint16_t*)hi)[0]);
    for (uint32_t m = blockIdx.y; m < M; m += gridDim.y) {
        float v = bf2f(x[(size_t)m * x_ld + i]);
        v = v < l ? l : v;
        v = v > h ? h : v;
        y[(size_t)m * y_ld + i] = f32_to_bf16(v);
    }
}

struct ConvGeom {
    int W, cin, cout, kh, kw, st, pad, cdiv, vst, Wo, S;
};

// ref_conv2d. A workgroup an OUTPUT time row of the pass: it finds the segment whose output rows
// hold it (a scan over cu_seqlens, which is a handful of entries), and a thread computes output
// elements (width, channel) of that row. Rows no segment covers are left alone, as libref does.
__global__ __launch_bounds__(kThreads) void conv2d_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    const int* __restrict__ valid, uint16_t* __restrict__ y, ConvGeom g, uint32_t x_ld,
    uint32_t y_ld) {
    const int r = (int)blockIdx.x;
    int t0 = -1, T = 0, to = 0;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || (c0 / g.cdiv) % g.st || c1 < c0) continue;
        const int a = c0 / g.cdiv, Ts = c1 / g.cdiv - a;
        if (Ts <= 0) continue;
        const int To = (Ts + 2 * g.pad - g.kh) / g.st + 1, o0 = a / g.st;
        if (r >= o0 && r < o0 + To) { t0 = a; T = Ts; to = r - o0; break; }
    }
    if (t0 < 0) return;
    const int nel = g.Wo * g.cout;
    for (int e = threadIdx.x; e < nel; e += kThreads) {
        const int wo = e / g.cout, co = e % g.cout;
        float acc = 0.0f;
        for (int i = 0; i < g.kh; ++i) {
            const int ti = to * g.st - g.pad + i;
            if (ti < 0 || ti >= T) continue;
            const int tg = t0 + ti;
            if (valid && valid[(size_t)tg * g.vst] == 0) continue;
            const uint16_t* xr = x + (size_t)tg * x_ld;
            for (int j = 0; j < g.kw; ++j) {
                const int wi = wo * g.st - g.pad + j;
                if (wi < 0 || wi >= g.W) continue;
                const uint16_t* xv = xr + (size_t)wi * g.cin;
                const uint16_t* wv = w + (((size_t)co * g.cin) * g.kh + i) * g.kw + j;
                for (int ci = 0; ci < g.cin; ++ci)
                    acc += bf2f(xv[ci]) * bf2f(wv[(size_t)ci * g.kh * g.kw]);
            }
        }
        y[(size_t)r * y_ld + e] = f32_to_bf16(acc);
    }
}

// conv2d_kernel's arithmetic with each kernel row's input row and weights staged in LDS: the form
// for the audio subsampler's second convolution, 64 x 128 channels in and 32 x 32 out a row,
// where the plain kernel re-reads every weight from global memory for every output and spends
// 6 ms on a 30 s clip. Weights sit [j][ci][co] so the lanes of a wave -- consecutive output
// channels -- read consecutive LDS words, and each input element is one broadcast. A thread owns
// kConvEpt outputs strided by the workgroup, all of one output channel. The sum runs over i, j,
// ci in the plain kernel's order.
constexpr int kConvEpt  = 4;
constexpr int kConvXs   = 8192;     /* W * cin, one input row */
constexpr int kConvWs   = 12288;    /* kw * cin * cout, one kernel row's weights */

__global__ __launch_bounds__(kThreads) void conv2d_lds_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    const int* __restrict__ valid, uint16_t* __restrict__ y, ConvGeom g, uint32_t x_ld,
    uint32_t y_ld) {
    __shared__ uint16_t xs[kConvXs];
    __shared__ uint16_t ws[kConvWs];
    const int r = (int)blockIdx.x;
    int t0 = -1, T = 0, to = 0;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || (c0 / g.cdiv) % g.st || c1 < c0) continue;
        const int a = c0 / g.cdiv, Ts = c1 / g.cdiv - a;
        if (Ts <= 0) continue;
        const int To = (Ts + 2 * g.pad - g.kh) / g.st + 1, o0 = a / g.st;
        if (r >= o0 && r < o0 + To) { t0 = a; T = Ts; to = r - o0; break; }
    }
    if (t0 < 0) return;
    const int nel = g.Wo * g.cout, nx = g.W * g.cin, nw = g.kw * g.cin * g.cout;
    const int tid = (int)threadIdx.x;
    float acc[kConvEpt];
    #pragma unroll
    for (int q = 0; q < kConvEpt; ++q) acc[q] = 0.0f;
    for (int i = 0; i < g.kh; ++i) {
        const int ti = to * g.st - g.pad + i;
        if (ti < 0 || ti >= T) continue;
        const int tg = t0 + ti;
        if (valid && valid[(size_t)tg * g.vst] == 0) continue;
        __syncthreads();
        const uint16_t* xr = x + (size_t)tg * x_ld;
        for (int e = tid; e < nx; e += kThreads) xs[e] = xr[e];
        for (int e = tid; e < nw; e += kThreads) {
            const int co = e % g.cout, rest = e / g.cout, ci = rest % g.cin, j = rest / g.cin;
            ws[e] = w[(((size_t)co * g.cin + ci) * g.kh + i) * g.kw + j];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < kConvEpt; ++q) {
            const int e = tid + q * kThreads;
            if (e >= nel) break;
            const int wo = e / g.cout, co = e % g.cout;
            float a = acc[q];
            for (int j = 0; j < g.kw; ++j) {
                const int wi = wo * g.st - g.pad + j;
                if (wi < 0 || wi >= g.W) continue;
                const uint16_t* xv = xs + wi * g.cin;
                const uint16_t* wv = ws + (size_t)j * g.cin * g.cout + co;
                for (int ci = 0; ci < g.cin; ++ci) a += bf2f(xv[ci]) * bf2f(wv[(size_t)ci * g.cout]);
            }
            acc[q] = a;
        }
    }
    #pragma unroll
    for (int q = 0; q < kConvEpt; ++q) {
        const int e = tid + q * kThreads;
        if (e < nel) y[(size_t)r * y_ld + e] = f32_to_bf16(acc[q]);
    }
}

// ref_conv1d_dw. A thread a (row, channel); the segment found by a scan over cu_seqlens.
__global__ __launch_bounds__(kThreads) void conv1d_dw_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const int* __restrict__ cu,
    uint16_t* __restrict__ y, int C, int K, int cdiv, int S, uint32_t x_ld, uint32_t y_ld,
    int rows) {
    const int c = (int)(blockIdx.x * kThreads + threadIdx.x);
    if (c >= C) return;
    for (int t = (int)blockIdx.y; t < rows; t += (int)gridDim.y) {
        int t0 = -1;
        for (int s = 0; s < S; ++s) {
            const int c0 = cu[s], c1 = cu[s + 1];
            if (c0 % cdiv || c1 % cdiv || c1 < c0) continue;
            if (t >= c0 / cdiv && t < c1 / cdiv) { t0 = c0 / cdiv; break; }
        }
        if (t0 < 0) continue;
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int tt = t - (K - 1) + j;
            if (tt < t0) continue;
            acc += bf2f(x[(size_t)tt * x_ld + c]) * bf2f(w[(size_t)c * K + j]);
        }
        y[(size_t)t * y_ld + c] = f32_to_bf16(acc);
    }
}

// ref_rel_sinusoid, the same f32 steps with the accurate libm forms.
__global__ __launch_bounds__(kThreads) void rel_sinusoid_kernel(
    uint16_t* __restrict__ y, uint32_t h, float inc, uint32_t y_ld) {
    const uint32_t d = blockIdx.x;
    for (uint32_t i = threadIdx.x; i < h; i += kThreads) {
        const float inv = expf((float)i * -inc);
        const float ang = (float)d * inv;
        y[(size_t)d * y_ld + i] = f32_to_bf16(sinf(ang));
        y[(size_t)d * y_ld + h + i] = f32_to_bf16(cosf(ang));
    }
}

// ref_attn_local_rel. A workgroup a (query row, head), a thread a head dimension; each of the at
// most left + 1 logits is a workgroup sum, then the softmax over them and the weighted sum of v.
constexpr uint32_t kLrThreads = 128;
constexpr int      kLrMaxKeys = 64;

struct LocalRelGeom {
    int D, H, L, cdiv, S, pds_f32;
    float cap, qs, ks;
    uint32_t q_ld, k_ld, v_ld, r_ld, o_ld;
};

__global__ __launch_bounds__(kLrThreads) void attn_local_rel_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const uint16_t* __restrict__ rk, const void* __restrict__ pds,
    const int* __restrict__ cu, uint16_t* __restrict__ o, LocalRelGeom g) {
    __shared__ float red[kLrThreads / 32];
    __shared__ float lg[kLrMaxKeys];
    const int i = (int)blockIdx.x, h = (int)blockIdx.y;
    int t0 = -1;
    for (int s = 0; s < g.S; ++s) {
        const int c0 = cu[s], c1 = cu[s + 1];
        if (c0 % g.cdiv || c1 % g.cdiv || c1 < c0) continue;
        if (i >= c0 / g.cdiv && i < c1 / g.cdiv) { t0 = c0 / g.cdiv; break; }
    }
    if (t0 < 0) return;
    const int j0 = i - g.L > t0 ? i - g.L : t0, nk = i - j0 + 1;
    const int tid = (int)threadIdx.x;
    for (int jj = 0; jj < nk; ++jj) {
        const int j = j0 + jj;
        float part = 0.0f;
        for (int d = tid; d < g.D; d += kLrThreads) {
            const float x = g.pds_f32 ? ((const float*)pds)[d] : bf2f(((const uint16_t*)pds)[d]);
            float sp = x > 20.0f ? x : (x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x)));
            if (!g.pds_f32) sp = bf2f(f32_to_bf16(sp));
            const float qf = bf2f(q[(size_t)i * g.q_ld + h * g.D + d]) * g.qs * sp;
            part += qf * (bf2f(k[(size_t)j * g.k_ld + h * g.D + d]) * g.ks);
            part += qf * bf2f(rk[(size_t)(i - j) * g.r_ld + h * g.D + d]);
        }
        part = r4d_wave_sum(part);
        if ((tid & 31) == 0) red[tid >> 5] = part;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (uint32_t w = 0; w < kLrThreads / 32; ++w) s += red[w];
            lg[jj] = g.cap * tanhf(s / g.cap);
        }
        __syncthreads();
    }
    float mx = -INFINITY;
    for (int jj = 0; jj < nk; ++jj) mx = lg[jj] > mx ? lg[jj] : mx;
    float sum = 0.0f;
    for (int jj = 0; jj < nk; ++jj) sum += expf(lg[jj] - mx);
    for (int d = tid; d < g.D; d += kLrThreads) {
        float acc = 0.0f;
        for (int jj = 0; jj < nk; ++jj)
            acc += expf(lg[jj] - mx) / sum * bf2f(v[(size_t)(j0 + jj) * g.v_ld + h * g.D + d]);
        o[(size_t)i * g.o_ld + h * g.D + d] = f32_to_bf16(acc);
    }
}

}  // namespace

// ============================================================== launchers

// act: 0 silu, 1 gelu (erf), 2 gelu_tanh, 3 sigmoid.
extern "C" int r4d_act_mul_bf16(long a, long b, long y, int M, int n, int act, int a_ld, int b_ld,
                                int y_ld, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0 || act < 0 || act > 3) return -2;
    const dim3 grid((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M));
#define R4D_AM(A)                                                                               \
    hipLaunchKernelGGL(act_mul_kernel<A>, grid, dim3(kThreads), 0, (hipStream_t)stream,         \
                       (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, (uint32_t)n,      \
                       (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, (uint32_t)M)
    if (act == 0)      R4D_AM(0);
    else if (act == 1) R4D_AM(1);
    else if (act == 2) R4D_AM(2);
    else               R4D_AM(3);
#undef R4D_AM
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_seg_mean_bf16(long x, long cu, long y, int S, int n, int x_ld, int y_ld,
                                 long stream) {
    if (!x || !cu || !y) return -1;
    if (S <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(seg_mean_kernel, dim3((uint32_t)((n + 31) / 32), (uint32_t)S),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const int*)cu, (uint16_t*)y, (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// mode: 0 neox, 1 gptj, 2 blocks (`blk` and the position plane stride `pplane` used only there).
extern "C" int r4d_qk_norm_rope_bf16(long qkv, long pos, long qw, long kw, int M, int head_dim,
                                     int n_head, int n_head_kv, int rot, int mode, int v_norm,
                                     float theta, float eps, float rscale, float wadd, int ld,
                                     const QkBlocks* blk, int pplane, long stream) {
    if (!qkv || !pos || !qw || !kw) return -1;
    if (M <= 0 || head_dim <= 0 || n_head <= 0 || n_head_kv <= 0 || rot <= 0 || rot > head_dim ||
        (rot & 1) || rot / 2 > (int)kQkMaxHalf || head_dim > 32 * kQkMaxE || head_dim % 32 ||
        mode < 0 || mode > 2)
        return -2;
    if (mode == 1 && head_dim % 64) return -2;
    if (mode == 0 && (rot / 2) % 32) return -2;
    QkBlocks b{};
    if (mode == 2) {
        if (!blk || blk->n <= 0 || blk->n > 8) return -2;
        b = *blk;
        int total = 0;
        for (int c = 0; c < b.n; ++c) {
            if ((2 * total) % 32 || (b.sec[c] != 16 && b.sec[c] % 32)) return -2;
            total += b.sec[c];
        }
        if (2 * total != rot) return -2;
    }
#define R4D_QN(MD)                                                                              \
    hipLaunchKernelGGL(qk_norm_rope_kernel<MD>, dim3((uint32_t)M), dim3(kQkThreads), 0,         \
                       (hipStream_t)stream, (uint16_t*)qkv, (const int*)pos, (const uint16_t*)qw, \
                       (const uint16_t*)kw, (uint32_t)head_dim, (uint32_t)n_head,               \
                       (uint32_t)n_head_kv, (uint32_t)rot, v_norm, theta, eps, rscale, wadd,    \
                       (uint32_t)ld, b, (uint32_t)pplane)
    if (mode == 1)      R4D_QN(1);
    else if (mode == 2) R4D_QN(2);
    else                R4D_QN(0);
#undef R4D_QN
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_pool_rows_bf16(long x, long y, int rows, int n, int k, float mult, int x_ld,
                                  int y_ld, long stream) {
    if (!x || !y) return -1;
    if (rows <= 0 || n <= 0 || k <= 0) return -2;
    hipLaunchKernelGGL(pool_rows_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads),
                                              grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)k, mult, (uint32_t)x_ld, (uint32_t)y_ld,
                       (uint32_t)rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_widen_bf16_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(widen_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (float*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_relu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(relu_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_silu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(silu_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)y,
                       (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_clamp_bf16(long x, long lo, long hi, int b_f32, long y, int M, int n, int x_ld,
                              int y_ld, long stream) {
    if (!x || !lo || !hi || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    hipLaunchKernelGGL(clamp_kernel, dim3((uint32_t)((n + kThreads - 1) / kThreads), grid_rows(M)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const void*)lo, (const void*)hi, b_f32, (uint16_t*)y, (uint32_t)n,
                       (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* `rows_out` is the output operand's rows: a workgroup each, every one a segment does not cover
 * returning at once. */
extern "C" int r4d_conv2d_bf16(long x, long w, long cu, long valid, long y, int rows_out,
                               const ConvGeom* g, int x_ld, int y_ld, long stream) {
    if (!x || !w || !cu || !y || !g) return -1;
    if (rows_out <= 0 || g->Wo <= 0 || g->S <= 0) return -2;
    const bool lds = g->W * g->cin <= kConvXs && g->kw * g->cin * g->cout <= kConvWs &&
                     g->Wo * g->cout <= kConvEpt * (int)kThreads;
    if (lds)
        hipLaunchKernelGGL(conv2d_lds_kernel, dim3((uint32_t)rows_out), dim3(kThreads), 0,
                           (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w,
                           (const int*)cu, (const int*)valid, (uint16_t*)y, *g, (uint32_t)x_ld,
                           (uint32_t)y_ld);
    else
        hipLaunchKernelGGL(conv2d_kernel, dim3((uint32_t)rows_out), dim3(kThreads), 0,
                           (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w,
                           (const int*)cu, (const int*)valid, (uint16_t*)y, *g, (uint32_t)x_ld,
                           (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_conv1d_dw_bf16(long x, long w, long cu, long y, int rows, int C, int K,
                                  int cdiv, int S, int x_ld, int y_ld, long stream) {
    if (!x || !w || !cu || !y) return -1;
    if (rows <= 0 || C <= 0 || K <= 0 || cdiv <= 0 || S <= 0) return -2;
    hipLaunchKernelGGL(conv1d_dw_kernel,
                       dim3((uint32_t)((C + kThreads - 1) / kThreads), grid_rows(rows)),
                       dim3(kThreads), 0, (hipStream_t)stream, (const uint16_t*)x,
                       (const uint16_t*)w, (const int*)cu, (uint16_t*)y, C, K, cdiv, S,
                       (uint32_t)x_ld, (uint32_t)y_ld, rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_rel_sinusoid_bf16(long y, int rows, int n, double max_timescale, int y_ld,
                                     long stream) {
    if (!y) return -1;
    if (rows <= 0 || n <= 0 || (n & 1)) return -2;
    const int h = n / 2;
    const float inc = (float)(std::log(max_timescale / 1.0) / (double)(h - 1 > 1 ? h - 1 : 1));
    hipLaunchKernelGGL(rel_sinusoid_kernel, dim3((uint32_t)rows), dim3(kThreads), 0,
                       (hipStream_t)stream, (uint16_t*)y, (uint32_t)h, inc, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_attn_local_rel_bf16(long q, long k, long v, long rk, long pds, long cu, long o,
                                       int rows, const LocalRelGeom* g, long stream) {
    if (!q || !k || !v || !rk || !pds || !cu || !o || !g) return -1;
    if (rows <= 0 || g->D <= 0 || g->H <= 0 || g->L < 0 || g->L + 1 > kLrMaxKeys) return -2;
    hipLaunchKernelGGL(attn_local_rel_kernel, dim3((uint32_t)rows, (uint32_t)g->H),
                       dim3(kLrThreads), 0, (hipStream_t)stream, (const uint16_t*)q,
                       (const uint16_t*)k, (const uint16_t*)v, (const uint16_t*)rk,
                       (const void*)pds, (const int*)cu, (uint16_t*)o, *g);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points

static inline int dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

/* ------------------------------------------------------------------ act_mul */
extern "C" int r4d_rad_act_mul(const RadArgs* a, RadStream s) {
    const RadTensor* A = r4d_opt(a, 0);
    const RadTensor* B = r4d_opt(a, 1);
    const RadTensor* Y = r4d_opt(a, 2);
    if (!A || !B || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(A));
    R4D_NEED(dt_bf16(B));
    R4D_NEED(dt_bf16(Y));
    const char* act = r4d_gets(a, "act");
    int ai = -1;
    if (act && !std::strcmp(act, "silu")) ai = 0;
    else if (act && !std::strcmp(act, "gelu")) ai = 1;
    else if (act && !std::strcmp(act, "gelu_tanh")) ai = 2;
    else if (act && !std::strcmp(act, "sigmoid")) ai = 3;
    if (ai < 0) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mb = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long a_ld = r4d_rows_pitch(A, n, &Ma);
    const long long b_ld = r4d_rows_pitch(B, n, &Mb);
    if (y_ld < 0 || a_ld < 0 || b_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || Mb < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_act_mul_bf16(r4d_addr(A), r4d_addr(B), r4d_addr(Y), (int)M, (int)n, ai, (int)a_ld,
                            (int)b_ld, (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                          : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ seg_mean */
extern "C" int r4d_rad_seg_mean(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* cu = r4d_opt(a, 1);
    const RadTensor* Y  = r4d_opt(a, 2);
    if (!X || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long S = r4d_numel(cu) - 1;
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < S) return R4D_NO(RAD_E_SHAPE);
    return r4d_seg_mean_bf16(r4d_addr(X), r4d_addr(cu), r4d_addr(Y), (int)S, (int)n, (int)x_ld,
                             (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ qk_norm_rope */
enum { QN_QKV = 0, QN_POS, QN_QW, QN_KW };

static int parse_ints(const char* v, int* out, int max) {
    int n = 0;
    while (v && *v && n < max) {
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v) break;
        out[n++] = (int)x;
        v = end;
    }
    return n;
}

extern "C" int r4d_rad_qk_norm_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r4d_opt(a, QN_QKV);
    const RadTensor* pos = r4d_opt(a, QN_POS);
    const RadTensor* qw  = r4d_opt(a, QN_QW);
    const RadTensor* kw  = r4d_opt(a, QN_KW);
    if (!qkv || !pos || !qw || !kw) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(qkv));
    R4D_NEED(dt_bf16(qw));
    R4D_NEED(dt_bf16(kw));
    if (pos->dtype != RAD_I32 || !r4d_contig(pos)) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(qw) || !r4d_contig(kw)) return R4D_NO(RAD_E_STRIDE);
    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nh  = r4d_geti_or(a, "n_head", 0);
    const long long nkv = r4d_geti_or(a, "n_head_kv", 0);
    const long long rot = r4d_geti_or(a, "rotary_dim", hd);
    if (hd <= 0 || nh <= 0 || nkv <= 0 || r4d_numel(qw) < hd || r4d_numel(kw) < hd)
        return R4D_NO(RAD_E_SHAPE);
    /* A head lives in a wave's registers with both halves of a rotary pair in one lane (or, for a
     * 16-pair block, in the two half-waves): head_dim a multiple of 32 up to 1024. */
    if (hd % 32 || hd > 1024 || rot > hd || (rot & 1)) return R4D_NO(RAD_E_SHAPE);
    const char* mode = r4d_gets(a, "mode");
    int md = 0;
    if (mode && !std::strcmp(mode, "gptj")) md = 1;
    else if (mode && !std::strcmp(mode, "blocks")) md = 2;
    else if (mode && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    QkBlocks blk{};
    long long M = 0, pplane = 0;
    if (md == 2) {
        blk.n = parse_ints(r4d_gets(a, "sections"), blk.sec, 8);
        if (blk.n <= 0) return R4D_NO(RAD_E_SHAPE);
        for (int c = 0; c < blk.n; ++c) blk.comp[c] = c;
        if (const char* cs = r4d_gets(a, "comps"))
            if (parse_ints(cs, blk.comp, 8) != blk.n) return R4D_NO(RAD_E_SHAPE);
        if (pos->rank != 2) return R4D_NO(RAD_E_SHAPE);
        for (int c = 0; c < blk.n; ++c)
            if (blk.comp[c] < 0 || blk.comp[c] >= pos->shape[0]) return R4D_NO(RAD_E_SHAPE);
        M = pos->shape[1];
        pplane = pos->stride[0];
    } else {
        M = r4d_numel(pos);
    }
    const long long width = (nh + 2 * nkv) * hd;
    long long Mq = 0;
    const long long ld = r4d_rows_pitch(qkv, width, &Mq);
    if (ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mq < M) return R4D_NO(RAD_E_SHAPE);
    const int e = r4d_qk_norm_rope_bf16(r4d_addr(qkv), r4d_addr(pos), r4d_addr(qw), r4d_addr(kw),
                                        (int)M, (int)hd, (int)nh, (int)nkv, (int)rot, md,
                                        (int)r4d_geti_or(a, "v_norm", 0),
                                        (float)r4d_getf(a, "theta", 10000.0),
                                        (float)r4d_getf(a, "eps", 1e-6),
                                        (float)r4d_getf(a, "scale", 1.0),
                                        (float)r4d_getf(a, "wadd", 0.0), (int)ld, &blk,
                                        (int)pplane, (long)(uintptr_t)s);
    if (e == -2) return R4D_NO(RAD_E_SHAPE);
    return e == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ pool_rows */
extern "C" int r4d_rad_pool_rows(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", 0), k = r4d_geti_or(a, "k", 0);
    if (n <= 0 || k <= 0) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx <= 0 || Mx % k || My < Mx / k) return R4D_NO(RAD_E_SHAPE);
    return r4d_pool_rows_bf16(r4d_addr(X), r4d_addr(Y), (int)(Mx / k), (int)n, (int)k,
                              (float)r4d_getf(a, "mult", 1.0), (int)x_ld, (int)y_ld,
                              (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ cast bf16 -> f32 */
extern "C" int r4d_rad_cast_widen(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* y = r4d_opt(a, 1);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(x));
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_widen_bf16_f32(r4d_addr(x), r4d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                              (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

/* ------------------------------------------------------------------ the audio tower */

extern "C" int r4d_rad_relu(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_relu_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                         (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_silu(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* Y = r4d_opt(a, 1);
    if (!X || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_silu_bf16(r4d_addr(X), r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                         (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_clamp(const RadArgs* a, RadStream s) {
    const RadTensor* X = r4d_opt(a, 0);
    const RadTensor* L = r4d_opt(a, 1);
    const RadTensor* H = r4d_opt(a, 2);
    const RadTensor* Y = r4d_opt(a, 3);
    if (!X || !L || !H || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Y));
    if (L->dtype != H->dtype || (L->dtype != RAD_BF16 && L->dtype != RAD_F32))
        return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(L) < 1 || r4d_numel(H) < 1) return R4D_NO(RAD_E_SHAPE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    const long long x_ld = r4d_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_clamp_bf16(r4d_addr(X), r4d_addr(L), r4d_addr(H), L->dtype == RAD_F32,
                          r4d_addr(Y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                          (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_conv2d(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* Wt = r4d_opt(a, 1);
    const RadTensor* cu = r4d_opt(a, 2);
    const RadTensor* V  = r4d_opt(a, 3);
    const RadTensor* Y  = r4d_opt(a, 4);
    if (!X || !Wt || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Wt));
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (V && (V->dtype != RAD_I32 || !r4d_contig(V))) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(Wt)) return R4D_NO(RAD_E_STRIDE);
    ConvGeom g{};
    g.W = (int)r4d_geti_or(a, "width", 0);
    g.cin = (int)r4d_geti_or(a, "cin", 0);
    g.cout = (int)r4d_geti_or(a, "cout", 0);
    g.kh = (int)r4d_geti_or(a, "kh", 0);
    g.kw = (int)r4d_geti_or(a, "kw", 0);
    g.st = (int)r4d_geti_or(a, "stride", 1);
    g.pad = (int)r4d_geti_or(a, "pad", 0);
    g.cdiv = (int)r4d_geti_or(a, "cu_div", 1);
    g.vst = (int)r4d_geti_or(a, "valid_stride", 1);
    if (g.W <= 0 || g.cin <= 0 || g.cout <= 0 || g.kh <= 0 || g.kw <= 0 || g.st <= 0 || g.pad < 0 ||
        g.cdiv <= 0 || g.vst <= 0)
        return R4D_NO(RAD_E_SHAPE);
    g.Wo = (g.W + 2 * g.pad - g.kw) / g.st + 1;
    g.S = (int)(r4d_numel(cu) - 1);
    if (g.Wo <= 0 || r4d_numel(Wt) < (long long)g.cout * g.cin * g.kh * g.kw)
        return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, (long long)g.W * g.cin, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, (long long)g.Wo * g.cout, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mx <= 0 || My <= 0) return R4D_NO(RAD_E_SHAPE);
    return r4d_conv2d_bf16(r4d_addr(X), r4d_addr(Wt), r4d_addr(cu), V ? r4d_addr(V) : 0,
                           r4d_addr(Y), (int)My, &g, (int)x_ld, (int)y_ld,
                           (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_conv1d_dw(const RadArgs* a, RadStream s) {
    const RadTensor* X  = r4d_opt(a, 0);
    const RadTensor* Wt = r4d_opt(a, 1);
    const RadTensor* cu = r4d_opt(a, 2);
    const RadTensor* Y  = r4d_opt(a, 3);
    if (!X || !Wt || !cu || !Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(X));
    R4D_NEED(dt_bf16(Wt));
    R4D_NEED(dt_bf16(Y));
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(Wt)) return R4D_NO(RAD_E_STRIDE);
    const long long C = r4d_geti_or(a, "n", 0), K = r4d_geti_or(a, "k", 0);
    const long long cdiv = r4d_geti_or(a, "cu_div", 1);
    if (C <= 0 || K <= 0 || cdiv <= 0 || r4d_numel(Wt) < C * K) return R4D_NO(RAD_E_SHAPE);
    long long Mx = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(X, C, &Mx);
    const long long y_ld = r4d_rows_pitch(Y, C, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My <= 0 || Mx < My) return R4D_NO(RAD_E_SHAPE);
    return r4d_conv1d_dw_bf16(r4d_addr(X), r4d_addr(Wt), r4d_addr(cu), r4d_addr(Y), (int)My,
                              (int)C, (int)K, (int)cdiv, (int)(r4d_numel(cu) - 1), (int)x_ld,
                              (int)y_ld, (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_rel_sinusoid(const RadArgs* a, RadStream s) {
    const RadTensor* Y = r4d_opt(a, 0);
    if (!Y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(Y));
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || (n & 1)) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    /* libref reads the parameter as a float before widening it; the same here, so a timescale
     * that is not a float is rounded in both. */
    const double mt = (double)(float)r4d_getf(a, "max_timescale", 10000.0);
    return r4d_rel_sinusoid_bf16(r4d_addr(Y), (int)M, (int)n, mt, (int)y_ld,
                                 (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_attn_local_rel(const RadArgs* a, RadStream s) {
    const RadTensor* q   = r4d_opt(a, 0);
    const RadTensor* k   = r4d_opt(a, 1);
    const RadTensor* v   = r4d_opt(a, 2);
    const RadTensor* rk  = r4d_opt(a, 3);
    const RadTensor* pds = r4d_opt(a, 4);
    const RadTensor* cu  = r4d_opt(a, 5);
    const RadTensor* o   = r4d_opt(a, 6);
    if (!q || !k || !v || !rk || !pds || !cu || !o) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_bf16(q));
    R4D_NEED(dt_bf16(k));
    R4D_NEED(dt_bf16(v));
    R4D_NEED(dt_bf16(rk));
    R4D_NEED(dt_bf16(o));
    if (pds->dtype != RAD_BF16 && pds->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu) || r4d_numel(cu) < 2) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(pds)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_geti_or(a, "right", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);
    LocalRelGeom g{};
    g.D = (int)r4d_geti_or(a, "head_dim", 0);
    g.H = (int)r4d_geti_or(a, "n_head", 0);
    g.L = (int)r4d_geti_or(a, "left", 0);
    g.cdiv = (int)r4d_geti_or(a, "cu_div", 1);
    g.cap = (float)r4d_getf(a, "softcap", 0.0);
    if (g.D <= 0 || g.H <= 0 || g.L < 0 || g.L + 1 > kLrMaxKeys || g.cdiv <= 0 || !(g.cap > 0.0f))
        return R4D_NO(RAD_E_SHAPE);
    const long long Wd = (long long)g.H * g.D;
    if (r4d_numel(pds) < g.D || r4d_numel(rk) < (long long)(g.L + 1) * Wd) return R4D_NO(RAD_E_SHAPE);
    long long Mq = 0, Mk = 0, Mv = 0, Mr = 0, Mo = 0;
    const long long q_ld = r4d_rows_pitch(q, Wd, &Mq), k_ld = r4d_rows_pitch(k, Wd, &Mk);
    const long long v_ld = r4d_rows_pitch(v, Wd, &Mv), r_ld = r4d_rows_pitch(rk, Wd, &Mr);
    const long long o_ld = r4d_rows_pitch(o, Wd, &Mo);
    if (q_ld < 0 || k_ld < 0 || v_ld < 0 || r_ld < 0 || o_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mo <= 0 || Mq < Mo || Mk < Mo || Mv < Mo) return R4D_NO(RAD_E_SHAPE);
    g.S = (int)(r4d_numel(cu) - 1);
    g.pds_f32 = pds->dtype == RAD_F32;
    g.qs = (float)(1.0 / std::sqrt((double)g.D) / std::log(2.0));
    g.ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    g.q_ld = (uint32_t)q_ld; g.k_ld = (uint32_t)k_ld; g.v_ld = (uint32_t)v_ld;
    g.r_ld = (uint32_t)r_ld; g.o_ld = (uint32_t)o_ld;
    return r4d_attn_local_rel_bf16(r4d_addr(q), r4d_addr(k), r4d_addr(v), r4d_addr(rk),
                                   r4d_addr(pds), r4d_addr(cu), r4d_addr(o), (int)Mo, &g,
                                   (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}
