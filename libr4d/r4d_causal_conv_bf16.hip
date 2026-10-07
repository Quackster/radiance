// Depthwise causal convolution, with no activation or residual. The caller owns LFM2's gates.
//
// One thread owns a channel for a 32-token tile. Only tile zero can read history (at most seven
// tokens), so it can also update that channel's rolling state without a grid-wide barrier. The
// state is read before any write and each live sequence owns a distinct slot. This is the same
// prefill/verification state contract as ple_conv, with dilation fixed at one.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"

#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kTile = 32;
constexpr uint32_t kMaxWidth = 8;
static_assert(kMaxWidth - 1 < kTile, "only tile zero may read and write history");

__global__ __launch_bounds__(kThreads) void causal_conv_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ state, const int32_t* __restrict__ cu,
    const int32_t* __restrict__ slots, const int32_t* __restrict__ init,
    const int32_t* __restrict__ accepted, uint16_t* __restrict__ y,
    uint32_t channels, uint32_t width, uint32_t xld, uint32_t wld, uint32_t yld,
    uint32_t rows, uint32_t n_slots, uint32_t depth) {
    const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= channels) return;
    const uint32_t s = blockIdx.z;
    const int32_t begin = cu[s], end = cu[s + 1];
    if (begin < 0 || end <= begin || (uint32_t)end > rows) return;
    const int32_t slot = slots ? slots[s] : (int32_t)s;
    if (slot == -1) return;  // the only supported inactive-slot sentinel
    // Slots otherwise satisfy 0 <= slot < n_slots. Keep malformed device metadata bounded;
    // unlike the host oracle, a launch cannot report its contents without synchronizing.
    if (slot < -1 || (uint32_t)slot >= n_slots) return;
    const int32_t length = end - begin;
    const int32_t first = (int32_t)(blockIdx.y * kTile);
    if (first >= length) return;
    const uint32_t history = width - 1;
    const int32_t offset = accepted ? accepted[s] - 1 : 0;
    if (offset < 0 || (uint32_t)offset > depth - history ||
        (accepted && (uint32_t)length > depth - history + 1)) return;

    uint16_t* dst = state + ((size_t)slot * channels + c) * depth;
    const uint16_t* input = x + (size_t)begin * xld + c;
    uint16_t* output = y + (size_t)begin * yld + c;
    const bool warm = !init || init[s] != 0;
    uint16_t prior[kMaxWidth - 1] = {};
    float weights[kMaxWidth];
#pragma unroll
    for (uint32_t j = 0; j < kMaxWidth; ++j)
        weights[j] = j < width ? bf16_to_f32(w[(size_t)c * wld + j]) : 0.0f;
    if (first == 0 && warm) {
#pragma unroll
        for (uint32_t i = 0; i < kMaxWidth - 1; ++i)
            if (i < history) prior[i] = dst[offset + i];
    }

    const int32_t last = min(first + (int32_t)kTile, length);
    for (int32_t t = first; t < last; ++t) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kMaxWidth; ++j) {
            if (j >= width) break;
            const int32_t source = t - (int32_t)history + (int32_t)j;
            const uint16_t value = source < 0 ? prior[source + history]
                : input[(size_t)source * xld];
            sum += bf16_to_f32(value) * weights[j];
        }
        output[(size_t)t * yld] = f32_to_bf16(sum);
    }

    if (first != 0) return;
    if (accepted) {
        // Every candidate remains available until the next step selects an accepted prefix.
#pragma unroll
        for (uint32_t i = 0; i < kMaxWidth - 1; ++i)
            if (i + 1 < history) dst[i] = prior[i + 1];
        for (int32_t t = 0; t < length; ++t)
            dst[history - 1 + t] = input[(size_t)t * xld];
    } else {
        // Prefill commits the whole chunk; copy its tail, retaining older history if needed.
#pragma unroll
        for (uint32_t i = 0; i < kMaxWidth - 1; ++i) {
            if (i >= history) break;
            const int32_t source = length - (int32_t)history + (int32_t)i;
            dst[i] = source < 0 ? prior[source + history] : input[(size_t)source * xld];
        }
    }
}

int causal_conv_body(const RadArgs* a, const R4dSink& sink) {
    const auto *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *state = r4d_opt(a, 2);
    const auto *cu = r4d_opt(a, 3), *slots = r4d_opt(a, 4), *init = r4d_opt(a, 5);
    const auto *accepted = r4d_opt(a, 6), *y = r4d_opt(a, 7);
    if (!x || !w || !state || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || state->dtype != RAD_BF16 ||
        y->dtype != RAD_BF16 || cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long long channels = r4d_geti_or(a, "n", 0);
    const long long width = r4d_geti_or(a, "width", 0);
    if (channels <= 0 || width < 2) return R4D_NO(RAD_E_SHAPE);
    if (width > kMaxWidth) return R4D_NO(RAD_E_UNSUPPORTED);
    long long rows = 0, out_rows = 0, weight_rows = 0;
    const long long xld = r4d_rows_pitch(x, channels, &rows);
    const long long yld = r4d_rows_pitch(y, channels, &out_rows);
    const long long wld = r4d_rows_pitch(w, width, &weight_rows);
    if (xld < 0 || yld < 0 || wld < 0 || !r4d_contig(cu) || !r4d_contig(state))
        return R4D_NO(RAD_E_STRIDE);
    const long long sequences = r4d_numel(cu) - 1;
    if (rows <= 0 || rows != out_rows || weight_rows < channels || sequences <= 0 ||
        state->rank != 3 || r4d_dim(state, 0) <= 0 || r4d_dim(state, 1) != channels ||
        r4d_dim(state, 2) < width - 1 ||
        (!slots && r4d_dim(state, 0) < sequences)) return R4D_NO(RAD_E_SHAPE);
    for (const RadTensor* ids : {slots, init, accepted}) {
        if (!ids) continue;
        if (ids->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
        if (!r4d_contig(ids)) return R4D_NO(RAD_E_STRIDE);
        if (r4d_numel(ids) < sequences) return R4D_NO(RAD_E_SHAPE);
    }
    const auto max_index = std::numeric_limits<int32_t>::max();
    if (rows > max_index || channels > max_index || xld > max_index || wld > max_index ||
        yld > max_index || sequences > 65535 || r4d_dim(state, 0) > max_index ||
        r4d_dim(state, 2) > max_index) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(sink, causal_conv_kernel,
        dim3((uint32_t)((channels + kThreads - 1) / kThreads),
             (uint32_t)((rows + kTile - 1) / kTile), (uint32_t)sequences), dim3(kThreads), 0,
        (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w), (uint16_t*)r4d_addr(state),
        (const int32_t*)r4d_addr(cu), slots ? (const int32_t*)r4d_addr(slots) : nullptr,
        init ? (const int32_t*)r4d_addr(init) : nullptr,
        accepted ? (const int32_t*)r4d_addr(accepted) : nullptr,
        (uint16_t*)r4d_addr(y), (uint32_t)channels, (uint32_t)width,
        (uint32_t)xld, (uint32_t)wld, (uint32_t)yld, (uint32_t)rows,
        (uint32_t)r4d_dim(state, 0), (uint32_t)r4d_dim(state, 2));
}

}  /* namespace */

R4D_ENTRY(r4d_rad_causal_conv1d, causal_conv_body)
