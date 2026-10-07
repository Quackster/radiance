// Selected vocabulary projection. Keep the full-head kernel's 256-thread/eight-element
// partition and reduction order so dropping unused vocabulary rows does not change logits.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <limits>

namespace {
constexpr uint32_t kThreads = 256, kWaves = 8, kVec = 8;

__global__ __launch_bounds__(kThreads) void logits_select_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const int32_t* __restrict__ rows, const int32_t* __restrict__ tokens,
    float* __restrict__ y, uint32_t K, uint32_t V, uint32_t R, uint32_t xld, bool round) {
    __shared__ float partial[kWaves];
    const uint32_t i = blockIdx.x;
    const int32_t row = rows[i], token = tokens[i];
    if (row < 0 || (uint32_t)row >= R || token < 0 || (uint32_t)token >= V) {
        if (threadIdx.x == 0) y[i] = __builtin_nanf("");
        return;
    }
    const uint16_t* a = x + (size_t)row * xld;
    const uint16_t* b = w + (size_t)token * K;
    float sum = 0.0f;
    for (uint32_t k = threadIdx.x * kVec; k < K; k += kThreads * kVec) {
        float chunk = 0.0f;
        #pragma unroll
        for (uint32_t p = 0; p < kVec; ++p)
            chunk += bf16_to_f32(a[k + p]) * bf16_to_f32(b[k + p]);
        sum += chunk;
    }
    sum = r4d_wave_sum(sum);
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    if (!lane) partial[wave] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float value = 0.0f;
        #pragma unroll
        for (uint32_t q = 0; q < kWaves; ++q) value += partial[q];
        y[i] = round ? bf16_to_f32(f32_to_bf16(value)) : value;
    }
}

int logits_select_body(const RadArgs* a, const R4dSink& sink) {
    const auto *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *rows = r4d_opt(a, 2);
    const auto *tokens = r4d_opt(a, 3), *y = r4d_opt(a, 4);
    if (!x || !w || !rows || !tokens || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || rows->dtype != RAD_I32 ||
        tokens->dtype != RAD_I32 || y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    const long long K = r4d_geti_or(a, "n_embd", 0), V = r4d_geti_or(a, "n_vocab", 0);
    const long long round = r4d_geti_or(a, "round_bf16", 0), M = r4d_numel(rows);
    if (K <= 0 || K % kVec || V <= 0 || M <= 0 || (round != 0 && round != 1))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(rows) || !r4d_contig(tokens) || !r4d_contig(y))
        return R4D_NO(RAD_E_STRIDE);
    long long R = 0;
    const long long xld = r4d_rows_pitch(x, K, &R);
    if (xld < 0) return R4D_NO(RAD_E_STRIDE);
    if (R <= 0 || w->rank != 2 || r4d_dim(w, 0) != V || r4d_dim(w, 1) != K ||
        r4d_numel(tokens) != M || r4d_numel(y) != M) return R4D_NO(RAD_E_SHAPE);
    const auto limit = std::numeric_limits<int32_t>::max();
    if (K > limit || V > limit || R > limit || M > limit || xld > limit) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(sink, logits_select_kernel, dim3((uint32_t)M), dim3(kThreads), 0,
        (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w),
        (const int32_t*)r4d_addr(rows), (const int32_t*)r4d_addr(tokens), (float*)r4d_addr(y),
        (uint32_t)K, (uint32_t)V, (uint32_t)R, (uint32_t)xld, round != 0);
}
}  // namespace

R4D_ENTRY(r4d_rad_logits_select, logits_select_body)
