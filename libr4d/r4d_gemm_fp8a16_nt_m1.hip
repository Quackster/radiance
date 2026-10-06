// BLOCK-SCALED FP8 weight, bf16 activation, ONE row. The decode form of the checkpoint's own dense
// format: `<linear>.weight` as F8_E4M3 [N][K] with `<linear>.weight_scale_inv` in BF16 on a
// [ceil(N/128)][ceil(K/128)] grid.
//
// WHY IT IS NOT r4d_gemm_fp8a8_nt_m16 WITH M = 1. That kernel's narrowest token tile is sixteen
// wide, so at one row it issues sixteen times the matrix work for the same weight bytes -- both are
// weight-bandwidth-bound, so that alone is survivable -- but it also needs the activation quantised
// to E4M3 first, which is a whole launch and a round trip through VRAM for one 5120-float vector.
// This reads the bf16 activation directly. The difference at decode is the launch, and decode is
// where launches are the budget.
//
// It is also the ONLY member of this family that does not need r4d_quant_act_fp8, which makes it
// the path for a caller with no fp8 quantiser in front of the linear.
//
// Three properties of the dot loops:
//
//   1. THE SCALE DTYPE is bf16, one per 128x128 tile: one shift and one bitcast to decode. See
//      r4d_fp8_wmma.h.
//   2. THE ACTIVATION IS bf16, not f32, as every activation in this library is, and the accessor is
//      where that lives rather than a second copy of the dot. It also halves the activation read,
//      which matters at RB > 1 where sixteen of them are held in registers per pass.
//   3. bf16 C, not f32, for the same reason every other GEMM here writes bf16.
//
// `rb`, the rows a wave, is a tuning argument and not a shape: blocking trades waves for
// memory-level parallelism, so where it lands depends on whether the kernel is waiting on loads or
// on the ALU -- a property of the shape AND of the inner loop, which means a value fitted against
// one inner loop does not carry to another. 0 takes the default rule (rb_default) so an untuned
// caller is fast rather than broken; 1, 2, 4 and 8 pin it.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.
//
// ---- THE MT LOOP IS ISSUE-BOUND AND v_dot2_f32_bf16 DOES NOT FIX IT (gfx1201) -------------------
//
// At MT = 8 this kernel reaches well under the rate M = 1 does on identical weight traffic, and the
// ISA says why: 394 instructions per 16 weight bytes a lane, ~136 of them FMA and ~89 doing nothing
// but widening bf16 activations to f32, because v_fma_f32 cannot read a bf16. At this part's
// 307 G wave-instructions/s that instruction count alone accounts for the whole of the rate.
//
// TWO WAYS OUT, AND NEITHER IS AVAILABLE ON THIS PART:
//
//   v_pk_fma_f32 does not exist on RDNA4. __builtin_elementwise_fma on a float2 lowers to two
//   scalar FMAs. It would be bit-preserving and free; it is simply not in the ISA.
//
//   v_dot2_f32_bf16 IS in the ISA and IS SLOWER. E4M3 -> bf16 is exact (three mantissa bits into
//   seven), so a bf16-pair dot multiplies exactly the same numbers, and a loop written around it is
//   215 instructions instead of 394 -- 64 v_dot2 and not one activation-widening op -- with register
//   pressure down from 57 VGPRs to 47. It is still about 30% slower: the DOT instructions on this
//   consumer part run at a fraction of v_fmac_f32's rate, so the scalar form is the right one here
//   however it reads. Do not re-derive this from the instruction count; the count is not the cost.
//
// THE WMMA PATH IS BELOW: lm_wmma_kernel decodes E4M3 to bf16 fragments on the fly and stages two
// scale blocks a round. It holds close to the card's stream rate from M = 1 to M = 16 where this
// loop has already fallen well below it, so `r4d_logits_gemm_fp8` sends everything past M = 2
// there. This matvec keeps M <= 2.

#include "r4d.h"
#include "r4d_fp8_args.h"
#include "r4d_args.h"
#include "r4d_fp8_lm.h"
#include "r4d_fp8_wmma.h"

#include <cstdlib>
#include <algorithm>

/* Local to this file, like r4d_gemm_fp8a8.hip's env_u32: a pin read once, with the rule below
 * deciding whenever it is unset, which is what a serve runs. */
static uint32_t env_u32_lm(const char* k) {
    const char* v = std::getenv(k);
    return v ? (uint32_t)std::strtoul(v, nullptr, 10) : 0u;
}

/* ROWS A WAVE; see rows_dot_mt for what RB is buying. The activation traffic is what MT costs and
 * RB divides it, but RB * MT is also the accumulator count, so the two pull opposite ways and the
 * best RB rises with MT and then falls back again:
 *
 *   - at MT <= 2 the activation is already amortised over the whole weight, so blocking only
 *     shortens the grid and RB = 1 wins;
 *   - through the middle of the band the activation re-read dominates and RB = 2 then RB = 4 win;
 *   - at MT = 32 RB = 4 is 128 accumulators a lane and falls off a cliff, so RB drops back to 2.
 *
 * R4D_LM_RB pins it. */
static uint32_t mvm_rb(int mt) {
    const uint32_t pin = env_u32_lm("R4D_LM_RB");
    if (pin == 1 || pin == 2 || pin == 4) return pin;
    if (mt <= 2) return 1u;
    if (mt <= 8) return 2u;
    if (mt <= 16) return 4u;
    return 2u;
}

/* HOW MUCH K ONE STAGING ROUND COVERS, by MT = ceil(M / 16); see lm_wmma_kernel for the argument.
 * Two scale blocks a round hide most of the cost of the stored row's 16-byte-only alignment, a
 * large win at MT = 1 and a smaller one at MT = 2; from MT = 3 the wider tile costs more LDS than
 * the straddle costs traffic and one block wins again. */
static uint32_t lm_fkb_rule(uint32_t mt) { return mt <= 2u ? 2u : 1u; }

namespace {

using namespace r4d_fp8;

constexpr int kWave = 32;
constexpr int kDenseWaves = 8;
constexpr int kDenseThreads = kDenseWaves * kWave;
constexpr uint32_t kBlock = 128;   // the scale block; see r4d_gemm_fp8a8.hip
// The widest token count the multi-row form serves. It is a REGISTER budget, not a shape: a lane
// holds MT accumulators and the decoded weight window, so past this the kernel spills and the
// 16-wide fp8a8 tile is the better answer anyway. 8 covers n_spec up to 7.
constexpr uint32_t kMaxMt = 8;

__device__ __forceinline__ float wave_add(float v) {
    v = r4d_wave_sum(v);
    return v;
}

// The activation, seen as f32 by the dot. `XV` is anything with `operator[](uint32_t) -> float`,
// which lets an RMS norm fold into the same loop without a second copy of it -- an
// `r4d_rmsnorm_fp8a16_nt_m1` would be exactly that.
struct Bf16V {
    const uint16_t* __restrict__ x;
    __device__ __forceinline__ float operator[](uint32_t i) const {
        return __uint_as_float((uint32_t)x[i] << 16);
    }
};

// One output row, reduced across the wave.
//
// 32 lanes x uint4 = 512 contiguous bytes a pass, fully coalesced. That width also fixes the
// reduction: a 512 B pass spans four scale tiles and 128/16 = 8, so lanes 0-7 land entirely inside
// the first tile, 8-15 the second, and so on. Every lane's 16 columns therefore share one scale, so
// a lane applies its own scale locally and the row needs a SINGLE wave reduction at the end rather
// than one per tile. Summation order changes -- per-lane then across, rather than across then
// per-tile -- but the value does not.
template <class XV>
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ W,
                                         const uint16_t* __restrict__ S, const XV& xv,
                                         uint32_t row, uint32_t cols, uint32_t w_ld,
                                         uint32_t sc_cols, int lane) {
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint8_t* w = W + (size_t)row * w_ld;
    const uint16_t* srow = S + (size_t)(row / kBlock) * sc_cols;
    float acc = 0.0f;
    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        const uint4 packed = *(const uint4*)(w + c);
        const uint32_t* pw = (const uint32_t*)&packed;
        const float scale = bscale(srow[c / kBlock]);
        float a0 = 0.0f, a1 = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const uint32_t p = pw[k];
            const uint32_t c4 = c + (uint32_t)k * 4u;
            // Two instructions for four bytes: v_cvt_pk_f32_fp8. No precondition: these are
            // arbitrary model weights and can be subnormal, which the HARDWARE handles correctly
            // where a software fast path does not.
            const v2f e01 = e4m3_pk<false>(p);
            const v2f e23 = e4m3_pk<true>(p);
            a0 = fmaf(e01.x, xv[c4 + 0], a0);
            a1 = fmaf(e01.y, xv[c4 + 1], a1);
            a0 = fmaf(e23.x, xv[c4 + 2], a0);
            a1 = fmaf(e23.y, xv[c4 + 3], a1);
        }
        acc = fmaf(a0 + a1, scale, acc);
    }
    // Tail, when K is not a multiple of 512. Same per-lane-scale rule, one column at a time.
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave)
        acc = fmaf(e4m3(w[c]) * xv[c], bscale(srow[c / kBlock]), acc);
    return wave_add(acc);
}

// The same dot, RB rows at a time, so one activation load feeds RB weight loads. The one-row form
// spends six memory instructions per 16 weight bytes and only ONE of them is the weight, which sits
// alone on the critical path; a narrow matrix runs too few inner iterations to hide that latency.
//
// RB consecutive rows share the activation window AND the scale row -- `r0` is RB-aligned and RB
// divides 128, so `row / 128` cannot change inside the block. What multiplies is the weight load,
// which is the point: RB of them issue back to back and are in flight together.
//
// Summation order is untouched: each row still accumulates per lane and reduces once at the end.
template <uint32_t RB, class XV>
__device__ __forceinline__ void rows_dot(const uint8_t* __restrict__ W,
                                         const uint16_t* __restrict__ S, const XV& xv,
                                         uint32_t row0, uint32_t nr, uint32_t cols, uint32_t w_ld,
                                         uint32_t sc_cols, int lane, float* __restrict__ acc) {
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    const uint8_t* w[RB];
    // Rows past `N` re-read row0 rather than branching: the result is discarded by the caller, and a
    // guarded load inside the inner loop would cost more than the duplicate traffic on the one
    // ragged wave of the grid.
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) w[r] = W + (size_t)(row0 + (r < nr ? r : 0)) * w_ld;
    const uint16_t* srow = S + (size_t)(row0 / kBlock) * sc_cols;
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) acc[r] = 0.0f;

    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        uint4 packed[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) packed[r] = *(const uint4*)(w[r] + c);
        float xr[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) xr[k] = xv[c + (uint32_t)k];
        const float scale = bscale(srow[c / kBlock]);
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            const uint32_t* pw = (const uint32_t*)&packed[r];
            float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t p = pw[k];
                const v2f e01 = e4m3_pk<false>(p);
                const v2f e23 = e4m3_pk<true>(p);
                a0 = fmaf(e01.x, xr[k * 4 + 0], a0);
                a1 = fmaf(e01.y, xr[k * 4 + 1], a1);
                a0 = fmaf(e23.x, xr[k * 4 + 2], a0);
                a1 = fmaf(e23.y, xr[k * 4 + 3], a1);
            }
            acc[r] = fmaf(a0 + a1, scale, acc[r]);
        }
    }
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
        const float sc = bscale(srow[c / kBlock]);
        const float xc = xv[c];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) acc[r] = fmaf(e4m3(w[r][c]) * xc, sc, acc[r]);
    }
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) acc[r] = wave_add(acc[r]);
}

// MT TOKEN ROWS AGAINST ONE WEIGHT ROW, which is what a speculative verify step is.
//
// A verify step runs at M = 1 + n_spec, and without this form one row past M = 1 sends every linear
// in the model to r4d_gemm_fp8a8_nt_m16, which is slower on the identical 24.9 GB of weights -- so
// turning speculation on would cost milliseconds a step before a single draft was proposed.
//
// The weight is read ONCE for all MT rows and that is the entire point: this loop is bound by the
// weight stream, six memory instructions per 16 weight bytes with only one of them the weight, and
// MT activation rows amortise it MT ways without reading another weight byte.
//
// The activations are a few KB, so they stay in L1 -- but they are read once per OUTPUT ROW, and
// past MT = 4 that L1 traffic is the kernel. rows_dot_mt below is the answer.
//
// TWO THINGS ARE DELIBERATE HERE:
//
//   - the E4M3 decode is hoisted out of the token loop. It is a property of the weight, and doing it
//     per token would multiply the only non-FMA work in the loop by MT -- four times the
//     v_cvt_pk_f32_fp8 count at MT = 4, for no extra arithmetic.
//   - the a0/a1 split keeps row_dot's summation order EXACTLY: even columns into a0, odd into a1,
//     per lane, one wave reduction at the end. So MT = 1 here is bit-identical to row_dot, and a
//     deployment that turns speculation on does not change the trunk's arithmetic.

// RB ROWS A WAVE, because the ACTIVATION is what makes MT expensive and not the weight.
//
// row_dot_mt above gives one output row to one wave, so every wave reads the whole of X for its
// own row: the weight stream is read once and X is read `rows` times. That is free at MT = 1 and
// it is the whole cost at MT = 32, where the rate over the stored weight has fallen to a fifth of
// what it is at M = 1.
//
// The weight is the same bytes at every M, so a rate that falls with M is not a memory-bound
// kernel -- and it is not registers either: no instantiation spills and every one reaches 16
// waves/SIMD. It is L1. Per 16 weight bytes a lane reads MT x 32 bytes of X, so on this model's
// lm_head (124160 x 5120) at MT = 32 the kernel pulls 40 GB through L1 to stream 0.65 GB from HBM,
// and that L1 traffic cannot hide under the weight stream.
//
// Giving a wave RB CONSECUTIVE rows divides the X traffic by RB: one load of X[m][c..c+15] now
// feeds RB rows instead of one. The weight is untouched -- RB rows means RB weight chunks, which
// is the same bytes in the same order.
//
// IT IS BIT-IDENTICAL. Each row keeps its own a0/a1 split, its own lane sum and its own
// wave_add, in the same order row_dot_mt used; RB only changes which wave owns a row and when the
// activation is fetched. RB = 1 compiles to the same arithmetic as before.
// THE INT8 HEAD'S CODES (logits_gemm_i8): a signed byte per weight, exact in f32 and in bf16, where
// the E4M3 head's are decoded by v_cvt_pk_f32_fp8. One form for both, picked at compile time.
template <bool I8, bool HI>
__device__ __forceinline__ v2f code_pk(uint32_t p) {
    if constexpr (I8) {
        v2f r;
        r.x = (float)(int8_t)(p >> (HI ? 16 : 0));
        r.y = (float)(int8_t)(p >> (HI ? 24 : 8));
        return r;
    } else {
        return e4m3_pk<HI>(p);
    }
}
template <bool I8>
__device__ __forceinline__ float code1(uint8_t b) {
    if constexpr (I8) return (float)(int8_t)b;
    else return e4m3(b);
}

template <uint32_t MT, uint32_t RB, bool SROW, bool I8 = false>
__device__ __forceinline__ void rows_dot_mt(const uint8_t* __restrict__ W,
                                            const uint16_t* __restrict__ S,
                                            const uint16_t* __restrict__ X, uint32_t a_ld,
                                            uint32_t row0, uint32_t rows, uint32_t cols,
                                            uint32_t w_ld, uint32_t sc_cols, uint32_t m_real,
                                            int lane, float* __restrict__ acc) {
    const uint32_t stride = kWave * 16u;
    const uint32_t full = (cols / stride) * stride;
    // THE ACTIVATION ROW A TOKEN SLOT READS. Instantiations above kMaxMt are launched only with M
    // rounded up to a multiple of four, and X holds `m_real` rows, not MT: a padded slot reads the
    // last real row again -- a duplicate read of live memory, like the clamped weight rows below --
    // and the epilogue drops its outputs. At or below kMaxMt every instantiation is exact, the
    // test folds away and the loop is unchanged.
    auto xrow = [m_real](uint32_t m) -> uint32_t {
        return (MT > kMaxMt && m >= m_real) ? m_real - 1u : m;
    };
    const uint8_t*  w[RB];
    const uint16_t* srow[RB];
#pragma unroll
    for (uint32_t r = 0; r < RB; ++r) {
        // CLAMPED, not skipped. A tail wave whose block runs past the last row computes the last
        // row again and the store below drops it -- a duplicate read of live memory, where a
        // branch would cost the whole loop its uniformity.
        const uint32_t rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        w[r] = W + (size_t)rr * w_ld;
        srow[r] = SROW ? (const uint16_t*)(w[r] + cols) : S + (size_t)(rr / kBlock) * sc_cols;
    }
#pragma unroll
    for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = 0.0f;

    for (uint32_t c = (uint32_t)lane * 16u; c < full; c += stride) {
        float wf[RB][16];
        float sc[RB];
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            const uint4 packed = *(const uint4*)(w[r] + c);
            const uint32_t* pw = (const uint32_t*)&packed;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const v2f e01 = code_pk<I8, false>(pw[k]);
                const v2f e23 = code_pk<I8, true>(pw[k]);
                wf[r][k * 4 + 0] = e01.x;
                wf[r][k * 4 + 1] = e01.y;
                wf[r][k * 4 + 2] = e23.x;
                wf[r][k * 4 + 3] = e23.y;
            }
            sc[r] = bscale(srow[r][c / kBlock]);
        }
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            const uint16_t* xm = X + (size_t)xrow(m) * a_ld + c;
            float xv[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) xv[k] = __uint_as_float((uint32_t)xm[k] << 16);
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                for (int k = 0; k < 16; k += 2) {
                    a0 = fmaf(wf[r][k],     xv[k],     a0);
                    a1 = fmaf(wf[r][k + 1], xv[k + 1], a1);
                }
                acc[r * MT + m] = fmaf(a0 + a1, sc[r], acc[r * MT + m]);
            }
        }
    }
    for (uint32_t c = full + (uint32_t)lane; c < cols; c += kWave) {
#pragma unroll
        for (uint32_t r = 0; r < RB; ++r) {
            const float wv = code1<I8>(w[r][c]), s = bscale(srow[r][c / kBlock]);
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m)
                acc[r * MT + m] =
                    fmaf(wv * __uint_as_float((uint32_t)X[(size_t)xrow(m) * a_ld + c] << 16), s,
                         acc[r * MT + m]);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < RB * MT; ++i) acc[i] = wave_add(acc[i]);
}

// bf16 for a GEMM output, f32 for a LOGITS one. The engine declares its logits buffer f32
// because a bf16 logit has eight bits of mantissa and a nucleus cut compares neighbours;
// r4d_model.cpp explains at length why writing bf16 into it is wrong.
template <class OT> __device__ __forceinline__ OT mvm_store(float v);
template <> __device__ __forceinline__ uint16_t mvm_store<uint16_t>(float v) { return f32_to_bf16(v); }
template <> __device__ __forceinline__ float mvm_store<float>(float v) { return v; }

template <uint32_t MT, bool SROW, class OT, uint32_t RB = 1, bool I8 = false>
__global__ __launch_bounds__(kDenseThreads) void mvm_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S,
    const uint16_t* __restrict__ X, uint32_t a_ld, OT* __restrict__ out, uint32_t c_ld,
    uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols, uint32_t m_real) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    const uint32_t step = gridDim.x * kDenseWaves * RB;
    for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
         r0 += step) {
        float acc[RB * MT];
        rows_dot_mt<MT, RB, SROW, I8>(W, S, X, a_ld, r0, rows, cols, w_ld, sc_cols, m_real, lane,
                                      acc);
        // C is [M][N], so MT lanes-0 write MT addresses `c_ld` apart. Scattered, and it is 2 bytes
        // per output row against a whole-weight read -- the same trade the AW epilogue makes in
        // r4d_gemm_fp8a8.hip, at a hundredth of the row count.
        // `m_real` is MT for every exact instantiation and smaller only where M was rounded up to
        // one -- the padded rows are computed and discarded, which is what buys ONE weight read
        // for the whole batch instead of one a chunk.
        if (lane == 0) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r) {
                if (r0 + r >= rows) break;      /* the clamped duplicate, dropped here */
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m)
                    if (m < m_real) out[(size_t)m * c_ld + r0 + r] = mvm_store<OT>(acc[r * MT + m]);
            }
        }
    }
}

template <uint32_t RB, class XV>
__global__ __launch_bounds__(kDenseThreads) void mv_rb_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S, XV xv,
    uint16_t* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    const uint32_t step = gridDim.x * kDenseWaves * RB;
    for (uint32_t r0 = ((uint32_t)blockIdx.x * kDenseWaves + (uint32_t)wave) * RB; r0 < rows;
         r0 += step) {
        const uint32_t nr = min(RB, rows - r0);
        float acc[RB];
        rows_dot<RB>(W, S, xv, r0, nr, cols, w_ld, sc_cols, lane, acc);
        if (lane == 0) {
#pragma unroll
            for (uint32_t r = 0; r < RB; ++r)
                if (r < nr) out[r0 + r] = f32_to_bf16(acc[r]);
        }
    }
}

template <class XV>
__global__ __launch_bounds__(kDenseThreads) void mv_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S, XV xv,
    uint16_t* __restrict__ out, uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t sc_cols) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wave = tid / kWave;
    for (uint32_t row = blockIdx.x * kDenseWaves + wave; row < rows;
         row += gridDim.x * kDenseWaves) {
        const float total = row_dot(W, S, xv, row, cols, w_ld, sc_cols, lane);
        if (lane == 0) out[row] = f32_to_bf16(total);
    }
}

// The default rule. Wide rows already have enough loads in flight, so blocking
// would only remove waves; narrow rows cannot hide a load behind two inner iterations, so block --
// harder the fewer rows there are to make waves from.
uint32_t rb_default(uint32_t rows, uint32_t cols) {
    if (cols > 2048u) return 1u;
    return rows >= 8192u ? 2u : 4u;
}

// ---------------------------------------------------------------------------- the lm head, on WMMA
//
// THE MATVEC ABOVE CANNOT SERVE A WIDE BATCH AND THE REASON IS NOT MEMORY. Its rate falls steeply
// with M (rows_dot_mt has the L1 argument) and row blocking recovers only part of it: what is left
// is that it is scalar FMA and the fp8a8 tile, which holds its rate across the same range on the
// identical weight, is not.
//
// This is that tile's shape for a bf16 activation. It costs one weight read at ANY M up to 64,
// which is what a served deployment needs: a step runs 1 + n_spec rows a SEQUENCE, so eight
// concurrent sequences at DFlash2 depth 7 put 64 rows through here, where the matvec would take
// two chunks of 32 and read the whole weight twice.
//
// IT LOSES NO PRECISION. WMMA's bf16 form takes bf16 A and bf16 B and accumulates in f32:
//
//   - the activation is ALREADY bf16, so A is exact;
//   - an E4M3 code has three mantissa bits and an exponent inside bf16's range, so the dequantised
//     code is exact in bf16 too -- f32_to_bf16 of a value whose low sixteen mantissa bits are zero
//     is a shift, and every E4M3 value has them zero;
//   - the BLOCK SCALE is applied AFTER the accumulation of its own 128-wide K block, not folded
//     into the operand, so it never rounds either.
//
// What changes is the ORDER of the sum -- WMMA's sixteen-wide reduction against the matvec's
// per-lane a0/a1 chains -- which is the same kind of change the split-K target already makes.
//
// THE FRAGMENT MAP IS WHY NO RE-LAYOUT IS NEEDED. r4d_fp8_wmma.h: for lane l, c = l % 16 and
// h = l / 16, the B operand of a row-major B[k][n] wants B[8h + j][c] for j = 0..7. Here
// B[k][n] = W[n][k], so a lane wants EIGHT CONSECUTIVE K of ONE weight row -- eight contiguous
// bytes at `wrow + k`, a single load out of the plain row-major plane this row already stores. The
// A operand wants A[c][8h + j], also eight consecutive K of one activation row, sixteen contiguous
// bytes. Both operands come from the same K labelling, which is rule 1 of that header.
//
// The accumulator is the other half of the map: slot j of lane l is D[8h + j][c]. So a lane owns
// ONE output column for all its slots -- one weight row, and therefore ONE block scale per K block
// rather than sixteen. That is what makes the scale a scalar multiply in the epilogue of each
// block instead of a per-element gather.
typedef short lm_s8v __attribute__((ext_vector_type(8)));

template <bool I8 = false>
__device__ __forceinline__ lm_s8v e4m3x8_to_bf16(uint32_t lo, uint32_t hi) {
    const v2f a01 = code_pk<I8, false>(lo), a23 = code_pk<I8, true>(lo);
    const v2f b01 = code_pk<I8, false>(hi), b23 = code_pk<I8, true>(hi);
    lm_s8v r;
    r[0] = (short)f32_to_bf16(a01.x); r[1] = (short)f32_to_bf16(a01.y);
    r[2] = (short)f32_to_bf16(a23.x); r[3] = (short)f32_to_bf16(a23.y);
    r[4] = (short)f32_to_bf16(b01.x); r[5] = (short)f32_to_bf16(b01.y);
    r[6] = (short)f32_to_bf16(b23.x); r[7] = (short)f32_to_bf16(b23.y);
    return r;
}

// MT is ceil(M / 16): how many 16-row activation tiles ride the one weight read.
//
// THE WEIGHT GOES THROUGH LDS, for the reason r4d_fp8_wmma.h's staging section gives and one more.
// A fragment read wants eight bytes of ONE weight row per lane, so a wave straight off global
// issues sixteen eight-byte requests a K step, each pulling a 128 B line that the next seven K
// steps then need to still be resident. Sixteen rows a wave times eight waves is a 16 KiB working
// set against a 32 KiB L1, which thrashes and costs roughly a third of the rate the fp8a8 tile
// reaches on the same bytes. Staged, the line is requested ONCE, by a cooperative load of
// contiguous bytes a thread, and every later read of it is an LDS read.
//
// The pitch is lds_pitch<FK>() and not FK, which is that header's padded-pitch argument applied
// unchanged: sixteen rows at a pitch of FK + 8 bytes land on sixteen distinct banks, where a pitch
// of FK would put all sixteen on one.
//
// ------------------------------------------------- FK, and it is the whole of what this tile tunes
//
// The obvious staging round is ONE 128-wide scale block, and in that form the tile's rate depends
// on MT while the weight stream does not -- which is not a memory-bound kernel.
//
// FK IS HOW MUCH K ONE STAGING ROUND COVERS, and widening it is the whole fix. The stored row is
// `[K bytes of E4M3][nkb bf16 scales]`, so at K = 5120 a row is 5200 BYTES -- sixteen-byte aligned
// and nothing more. A one-block round reads 128 bytes at `n * 5200 + kb`, which lands at byte 80,
// 160, 240, 64 ... inside a 128-byte line and therefore STRADDLES TWO LINES for every row of every
// round. The rest of each line is the SAME row's next block, wanted seven rounds later and long
// since evicted. FK = 256 makes it three lines a 256 bytes instead of four.
//
// The straddle's size is separable from FK: a K whose stored stride IS 128-aligned
// (`K + 2*K/128 == 0 (mod 128)`, which K = 8192 satisfies) runs ~39% faster at MT = 1 and ~29% at
// MT = 4 than a neighbouring K that straddles, and identically at MT = 2, which already reaches the
// aligned rate. FKB = 2 recovers most of that without a re-convert, and at MT <= 2 it brings the
// tile to within a few percent of the card's stream rate. From MT = 3 the wider tile loses: 33792
// bytes of LDS against 17408 is fewer workgroups a WGP, and MT >= 3 has the accumulators to want
// them. lm_fkb_rule above is that rule.
//
// FOUR ARMS THAT LOSE AND ARE NOT IN THE CODE:
//
//   ALDS, the activation tile staged in LDS beside the weight. Every wave in the workgroup reads
//   the SAME activation bytes -- the fragment address has no `wv` in it -- so a round pulls
//   MT x 2 x FK bytes through L1 eight times over, and staging it cooperatively would read it once.
//   It is worth a few percent at FKB = 1 and MT <= 2, and FK DOMINATES IT: at FKB = 2 it is a wash
//   at MT = 1 and a loss everywhere else. The amplification is not the constraint; the straddle
//   is. At MT = 4 and FKB = 2 the two tiles also want 67584 bytes, over the 64 KiB a workgroup
//   gets.
//
//   DOUBLE-BUFFERING THE STAGED TILE, which is the other way to cover a staging round and the one
//   that says what the straddle actually costs. Two buffers at FKB = 1 fit (34816 bytes) and turn
//   the round's two barriers into one, with the next tile's global loads issued BEFORE this tile's
//   WMMA. It recovers almost exactly what FK does WITHOUT MOVING A BYTE LESS, so the straddle's
//   cost is LATENCY and not traffic: a 128-byte read at byte 80 of a line is two memory
//   transactions instead of one, and with four loads a thread in flight the second one is on the
//   critical path. FK removes the transaction, double-buffering hides it, and FK is slightly better
//   because it does both. At MT >= 3 double-buffering is a small LOSS, and at FKB = 2 there is no
//   room for it, so it survives nowhere. If a K that is not a multiple of 256 ever ships, FKB falls
//   back to 1 and this is what to put back.
//
//   NW, weight tiles a wave -- the N-direction dual of MT, meant for MT = 1 where a K step is one
//   activation load, one LDS read, eight cvt and ONE wmma. NW = 2 is SLOWER at every M: it doubles
//   kRows and so the LDS and the staging round, and at MT = 4 it spills 60 registers.
//
//   `__builtin_nontemporal_load` on the weight stream -- the operand no block re-reads, and the
//   textbook case for the hint. It is about TWICE AS SLOW at every shape. On this part the hint
//   costs the stream its coalescing and there is no shape where it pays. (r4d_fp8_wmma.h's
//   stage8_load offers the same hint for the fragment-order planes; this says nothing about those,
//   whose rows ARE 128-aligned.)
//
// FK CHANGES NO ARITHMETIC. Every output element still accumulates its K in the same order and
// still applies its block scale after its own 128-wide block, which is why the scale loop below is
// nested INSIDE the staged tile rather than wrapped around it -- FK is a staging width and it must
// not reach the arithmetic.
//
// PADDING THE STORED STRIDE TO A 128-ALIGNED ONE IS NOT WORTH IT, and it would cost a re-convert.
// At K = 5120 the pad is 5200 -> 5376 bytes, +3.4% on every byte read, and FKB = 2 on the 5200-byte
// stride already beats an aligned stride at FKB = 1 at both ends of the M band. The straddle is
// cheaper than the padding.
//
// WHAT IS LEFT AT MT >= 3, AND IT IS FIVE THINGS IT IS NOT. MT <= 2 is within a few percent of the
// card's stream rate and is done; MT = 4 is well short of it, and every mechanism that explains the
// MT <= 2 gap has been tried there and falsified:
//
//   not the register file -- the rate does track waves a SIMD (14 at MT = 1 down to 6 at MT = 4),
//   and MT = 4 sits at 220 VGPRs against the 216 that would buy a seventh wave, so dropping the
//   K-step unroll from two to one is the one-line way to buy it. It buys it -- 200 VGPRs, seven
//   waves, no spill -- and it is about 12% SLOWER at MT = 3 and MT = 4 alike. Two K steps in flight
//   are worth more than the wave. Same shape as the prefill tile's null in r4d_gemm_fp8a8.hip: a
//   resource count is not a cost;
//   not the staging latency -- double-buffering is a wash to a small loss there (above);
//   not the straddle -- FKB = 2 is a loss there (above);
//   not the activation loads -- they scale with m_real and the time does not. M = 56 and M = 64 are
//   the same MT = 4 kernel doing the same WMMA, with 12% fewer activation loads at 56, and they
//   take the same time;
//   not the activation re-read, which is the one worth spelling out, below.
//
// It is also not the WMMA issue rate: the tile is far under the matrix pipe's throughput at MT = 4,
// and the E4M3 conversion is per WEIGHT fragment, so MT = 4 has four times MT = 1's arithmetic per
// cvt and not less. Whatever it is, five separate mechanisms have been priced and none of them is
// it, and the band is a small share of a wide-batch step.
//
// ---- WHY IT IS NOT THE ACTIVATION RE-READ ------------------------------------------------------
//
// r4d_gemm_fp8a8.hip's narrow tile records the fall past 16 rows as the activation being staged
// once per WORKGROUP, and its fix is a wider workgroup. The arithmetic here says the same thing
// at first: the activation is read from GLOBAL inside the K loop and every workgroup reads the
// whole [M][K], so a call moves
//
//     weight  rows * (K + 2*nkb)   +   activation  M * K * 2 * (rows / (NW*16))
//
// which at this shape is 646 MB against 636 MB at M = 64 -- the same size. Halving the workgroup
// count halves the second term at no cost in parallelism, since NW * (rows/(NW*16)) is rows/16
// whatever NW is. It does not help. NW = 16 is ~38% slower, and that part IS occupancy: the same
// 220 VGPRs and 6 waves a SIMD, but A WORKGROUP IS THE GRANULE -- 24 waves a WGP is three 8-wave
// groups or ONE 16-wave group, and the second does not tile. NW = 12 is the count that tiles
// exactly (two groups, 24 waves, 26112 B of LDS against 3 x 17408) and it is STILL a few percent
// slower while doing a third fewer activation reads. So the re-read is served out of cache and
// costs nothing measurable, and a wider workgroup makes every __syncthreads wait on more waves --
// the same reading the double-buffering result above gives.
//
// The band is five things it is not: not the register file, not staging latency, not the straddle,
// not the activation loads, not the activation re-read. It needs a new hypothesis, not another
// pass over these.
template <uint32_t MT, uint32_t FKB, bool I8 = false>
__global__ __launch_bounds__(kDenseThreads) void lm_wmma_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ A, uint32_t a_ld,
    float* __restrict__ C, uint32_t c_ld,
    uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t m_real) {
    constexpr uint32_t kFK    = FKB * kBlock;              /* K a staging round */
    constexpr uint32_t kRows  = kDenseWaves * 16u;         /* weight rows a workgroup covers */
    constexpr uint32_t kPitch = lds_pitch<kFK>();          /* BYTES a staged weight row */
    constexpr uint32_t kWUnits = kRows * (kFK / 16u);      /* 16-byte staging units, weight */

    __shared__ uint8_t s_w[kRows * kPitch];

    const uint32_t tid  = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & (kWave - 1u);
    const uint32_t wv   = tid / kWave;
    const uint32_t c    = lane & 15u;          /* this lane's weight row inside its 16-row tile */
    const uint32_t h    = lane >> 4;           /* which half of the K step */

    for (uint32_t base = (uint32_t)blockIdx.x * kRows; base < rows; base += gridDim.x * kRows) {
        /* Clamped, not skipped: a tail workgroup stages and computes
         * the last row again and the store drops it. */
        const uint32_t n   = base + wv * 16u + c;
        const uint32_t nc  = (n < rows) ? n : (rows - 1u);
        const uint16_t* scl = (const uint16_t*)(W + (size_t)nc * w_ld + cols);

        f8v acc[MT];
#pragma unroll
        for (uint32_t t = 0; t < MT; ++t) acc[t] = (f8v)(0.0f);

        for (uint32_t kt = 0; kt < cols; kt += kFK) {
            /* Before the store, because the previous tile's fragment reads are still in flight. */
            __syncthreads();
/* HOW MANY STAGING LOADS ARE IN FLIGHT, and it is a REGISTER decision, not a memory one. A round
 * is kFK/32 units a thread -- eight at FKB = 2 -- and unrolling it whole holds eight uint4 results
 * live across the stores, 32 VGPRs on top of MT accumulators and MT partials. At MT >= 3 unrolling
 * it whole puts FKB = 2 at 256 VGPRs with six spill slots, which is why the unroll narrows there. */
#pragma unroll MT >= 3 ? 4 : 8
            for (uint32_t i = 0; i < kWUnits / kDenseThreads; ++i) {
                const uint32_t u = tid + i * kDenseThreads;
                const uint32_t r = u / (kFK / 16u), off = (u % (kFK / 16u)) * 16u;
                const uint32_t g = base + r;
                const uint8_t* src = W + (size_t)((g < rows) ? g : (rows - 1u)) * w_ld + kt + off;
                *(uint4*)(s_w + (size_t)r * kPitch + off) = *(const uint4*)src;
            }
            __syncthreads();

            /* One scale block at a time INSIDE the staged tile, so the partial that the block
             * scale multiplies is exactly the one a one-block round forms. FK is a staging width and
             * nothing else; it must not reach the arithmetic. */
#pragma unroll
            for (uint32_t sb = 0; sb < FKB; ++sb) {
                f8v part[MT];
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) part[t] = (f8v)(0.0f);
/* Four K steps in flight is the right depth until MT = 4, where four accumulators, four partials
 * and four activation fragments an iteration reach 249 VGPRs and five waves a SIMD. Below MT = 4
 * the deeper unroll wins clearly; at MT = 4 the register pressure reverses it. */
#pragma unroll MT >= 4 ? 2 : 4
                for (uint32_t ks = 0; ks < kBlock; ks += 16u) {
                    const uint32_t kf = sb * kBlock + ks;          /* K inside the staged tile */
                    const i2v raw = frag<kFK>(s_w, wv * 16u, kf, lane);
                    const lm_s8v bf = e4m3x8_to_bf16<I8>((uint32_t)raw[0], (uint32_t)raw[1]);
#pragma unroll
                    for (uint32_t t = 0; t < MT; ++t) {
                        const uint32_t m = t * 16u + c;
                        const lm_s8v af =
                            (m < m_real) ? *(const lm_s8v*)(A + (size_t)m * a_ld + kt + kf + h * 8u)
                                         : (lm_s8v)(short)0;
                        part[t] =
                            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af, bf, part[t]);
                    }
                }
                const float sc = bscale(scl[(kt + sb * kBlock) / kBlock]);
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t][j] = fmaf(part[t][j], sc, acc[t][j]);
            }
        }

        if (n >= rows) continue;
#pragma unroll
        for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t m = t * 16u + h * 8u + j;
                if (m < m_real) C[(size_t)m * c_ld + n] = acc[t][j];
            }
    }
}

// ---------------------------------------------------------------- logits_gemm_i4, the int4 head
//
// lm_wmma_kernel at half the code bytes. The stored row is r4d_fp8::lm4_row_bytes': K/2 bytes of
// two's-complement nibbles (low nibble first), the row's own bf16 scale per 128 columns in its
// tail, the pad. Everything that is not the code width is lm_wmma_kernel's and is argued there:
// the weight staged through LDS at a padded pitch, ONE 128-wide scale block's partial at a time
// inside a staging round of FKB blocks, the activation read straight from global, f32 out.
//
// What the width changes: a staging round of FK columns is FK/2 BYTES a row, so the LDS tile is
// half the size at the same FKB, and FKB = 4 fits (128 rows x 264 bytes). A fragment is eight
// codes, which are four bytes and not eight -- one dword at (k/2) in the staged row, one bank a
// lane: the pitch FK/2 + 8 is an odd multiple of eight bytes at every FKB, so the sixteen rows of
// a tile and the two halves of a K step land on 32 distinct banks.
//
// A nibble is exact in bf16 (|v| <= 8), so the arithmetic is lm_wmma_kernel's: bf16 x bf16 WMMA
// into f32, the block scale applied to the block's partial.
__device__ __forceinline__ lm_s8v i4x8_to_bf16(uint32_t p) {
    lm_s8v r;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        r[i] = (short)f32_to_bf16((float)(((int)(p << (28 - 4 * i))) >> 28));
    return r;
}
// The asymmetric grid's eight codes: unsigned nibbles less the group's zero, |v| <= 15, exact.
__device__ __forceinline__ lm_s8v u4x8_to_bf16(uint32_t p, int z) {
    lm_s8v r;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        r[i] = (short)f32_to_bf16((float)((int)((p >> (4 * i)) & 15u) - z));
    return r;
}

// THE ASYMMETRIC CODES WITHOUT A CONVERSION. 128 + n for a nibble n is exact in bf16 and its bits
// are 0x4300 | n, so eight codes become four packed bf16 pairs with an AND, a shift and an OR each --
// where (float)(n - z) then f32_to_bf16 was two conversions a code and the reason this tile ran at
// ~465 GB/s while the int8-activation draft head on the same bytes ran at ~600. The offset is taken
// back per group: sum (128 + n) x = sum (n - z) x + (128 + z) sum x, and sum x over the group comes
// out of one more WMMA against a B fragment of ones, in the same slot layout as the product -- a
// lane holds one weight row, so its zero is a scalar.
typedef unsigned int lm_u4v __attribute__((ext_vector_type(4)));
__device__ __forceinline__ lm_s8v u4x8_to_bf16_off128(uint32_t p) {
    lm_u4v w;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (p >> (8 * i)) & 0xFFu;
        w[i] = 0x43004300u | (b & 0xFu) | ((b & 0xF0u) << 12);
    }
    return __builtin_bit_cast(lm_s8v, w);
}

// GS is the scale group along K (64 or 128) and ASYM the asymmetric grid, whose u8 zeros follow
// the scales in the row's tail. The group is what the inner partial covers, so a row's scale (and
// zero) multiplies exactly that group's sum -- the FK staging round is a multiple of 128 and so of
// either group, and stays a staging width only.
// NPW is how many 16-row weight tiles one WAVE owns (R4D_LM4_NPW; default 1 = the original kernel).
// The activation fragment and the group's sum-of-x WMMA depend only on the activation, so with
// NPW = 2 one A load and one xsum WMMA serve two weight tiles: per k step the wave issues MT loads
// and MT * (2 NPW + 1) WMMA where two NPW = 1 waves issue 2 MT loads and 4 MT WMMA. Every output
// element still sums the same WMMA partials in the same order, so the result is BIT-IDENTICAL to
// NPW = 1; only who computes the (identical) xsum, and how often A is loaded, changes.
// LEAN is a register-pressure knob and nothing else (R4D_LM4_LEAN; unset = auto: 1 from MT 2, 0 at
// MT 1; 0 = the original unrolling). At MT >= 3 the fully unrolled group loops make the compiler
// hoist every A fragment of the round and spill (ScratchSize 448 B a lane at MT 3, 748 at MT 4 for the live u4 g64 head: the
// 3 -> 4 row-tile cliff, 4.5 -> 8.3 ms). LEAN 1 keeps the group loop rolled, LEAN 2 the k-step loop
// too. The arithmetic and its order are untouched, so the output is bit-identical at every level.
template <uint32_t MT, uint32_t FKB, uint32_t GS = 128, bool ASYM = false, uint32_t NPW = 1,
          uint32_t LEAN = 0>
__global__ __launch_bounds__(kDenseThreads) void lm_wmma_i4_kernel(
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ A, uint32_t a_ld,
    float* __restrict__ C, uint32_t c_ld,
    uint32_t rows, uint32_t cols, uint32_t w_ld, uint32_t m_real) {
    constexpr uint32_t kFK     = FKB * kBlock;             /* K a staging round */
    constexpr uint32_t kFB     = kFK / 2u;                 /* code BYTES a row a round */
    constexpr uint32_t kRows   = kDenseWaves * 16u * NPW;  /* weight rows a workgroup covers */
    constexpr uint32_t kPitch  = kFB + 8u;                 /* bytes a staged row */
    constexpr uint32_t kWUnits = kRows * (kFB / 16u);      /* 16-byte staging units */
    static_assert(kWUnits % kDenseThreads == 0, "a staging round is whole units a thread");
    static_assert(kRows * kPitch <= 65536u, "staged rows must fit the 64 KiB workgroup LDS");

    __shared__ uint8_t s_w[kRows * kPitch];

    const uint32_t tid  = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & (kWave - 1u);
    const uint32_t wv   = tid / kWave;
    const uint32_t c    = lane & 15u;
    const uint32_t h    = lane >> 4;
    const uint32_t cb   = cols / 2u;                       /* code bytes a row */
    const uint32_t ng   = cols / GS;                       /* scale groups a row */

    for (uint32_t base = (uint32_t)blockIdx.x * kRows; base < rows; base += gridDim.x * kRows) {
        const uint16_t* scl[NPW];
        const uint8_t*  zr[NPW];
        uint32_t nn[NPW];
#pragma unroll
        for (uint32_t p = 0; p < NPW; ++p) {
            nn[p] = base + (wv * NPW + p) * 16u + c;
            const uint32_t nc = (nn[p] < rows) ? nn[p] : (rows - 1u);
            scl[p] = (const uint16_t*)(W + (size_t)nc * w_ld + cb);
            zr[p]  = W + (size_t)nc * w_ld + cb + 2u * ng;
        }

        f8v acc[NPW][MT];
#pragma unroll
        for (uint32_t p = 0; p < NPW; ++p)
#pragma unroll
            for (uint32_t t = 0; t < MT; ++t) acc[p][t] = (f8v)(0.0f);

        for (uint32_t kt = 0; kt < cols; kt += kFK) {
            __syncthreads();
#pragma unroll (LEAN >= 1 ? 2 : (MT >= 3 ? 4 : 8))
            for (uint32_t i = 0; i < kWUnits / kDenseThreads; ++i) {
                const uint32_t u = tid + i * kDenseThreads;
                const uint32_t r = u / (kFB / 16u), off = (u % (kFB / 16u)) * 16u;
                const uint32_t g = base + r;
                const uint8_t* src =
                    W + (size_t)((g < rows) ? g : (rows - 1u)) * w_ld + kt / 2u + off;
                *(uint4*)(s_w + (size_t)r * kPitch + off) = *(const uint4*)src;
            }
            __syncthreads();

#pragma unroll (LEAN >= 1 ? 1 : 16)
            for (uint32_t sb = 0; sb < kFK / GS; ++sb) {
                f8v part[NPW][MT], xsum[MT];
#pragma unroll
                for (uint32_t t = 0; t < MT; ++t) xsum[t] = (f8v)(0.0f);
#pragma unroll
                for (uint32_t p = 0; p < NPW; ++p)
#pragma unroll
                    for (uint32_t t = 0; t < MT; ++t) part[p][t] = (f8v)(0.0f);
                const uint32_t gi = (kt + sb * GS) / GS;
                int zg[NPW];
#pragma unroll
                for (uint32_t p = 0; p < NPW; ++p) zg[p] = ASYM ? (int)zr[p][gi] : 0;
                const lm_s8v ones = (lm_s8v)(short)0x3F80;
#pragma unroll (LEAN >= 2 ? 1 : (MT >= 4 ? 2 : 4))
                for (uint32_t ks = 0; ks < GS; ks += 16u) {
                    const uint32_t kf = sb * GS + ks;
                    lm_s8v af[MT];
#pragma unroll
                    for (uint32_t t = 0; t < MT; ++t) {
                        const uint32_t m = t * 16u + c;
                        af[t] = (m < m_real) ? *(const lm_s8v*)(A + (size_t)m * a_ld + kt + kf + h * 8u)
                                             : (lm_s8v)(short)0;
                    }
#pragma unroll
                    for (uint32_t p = 0; p < NPW; ++p) {
                        const uint32_t raw = *(const uint32_t*)(s_w +
                            (size_t)((wv * NPW + p) * 16u + c) * kPitch + kf / 2u + h * 4u);
                        const lm_s8v bf = ASYM ? u4x8_to_bf16_off128(raw) : i4x8_to_bf16(raw);
#pragma unroll
                        for (uint32_t t = 0; t < MT; ++t)
                            part[p][t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                af[t], bf, part[p][t]);
                    }
                    if constexpr (ASYM)
#pragma unroll
                        for (uint32_t t = 0; t < MT; ++t)
                            xsum[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                                af[t], ones, xsum[t]);
                }
#pragma unroll
                for (uint32_t p = 0; p < NPW; ++p) {
                    const float sc = bscale(scl[p][gi]);
                    const float off = ASYM ? (float)(128 + zg[p]) : 0.0f;
#pragma unroll
                    for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[p][t][j] = fmaf(ASYM ? fmaf(-off, xsum[t][j], part[p][t][j])
                                                     : part[p][t][j], sc, acc[p][t][j]);
                }
            }
        }

#pragma unroll
        for (uint32_t p = 0; p < NPW; ++p) {
            if (nn[p] >= rows) continue;
#pragma unroll
            for (uint32_t t = 0; t < MT; ++t)
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t m = t * 16u + h * 8u + j;
                    if (m < m_real) C[(size_t)m * c_ld + nn[p]] = acc[p][t][j];
                }
        }
    }
}
}  // namespace

extern "C" int r4d_gemm_fp8a16_nt_m1_block(void) { return (int)kBlock; }
extern "C" int r4d_gemm_fp8a16_nt_m1_max_m(void) { return (int)kMaxMt; }

// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T for M <= kMaxMt, one weight read for all M rows.
//
//   a    [M][K]                     bf16, `a_ld` elements a row (0 = K)
//   wq   [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws   [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c    [M][N]                     bf16, `c_ld` elements a row (0 = N)
//
// M IS A TEMPLATE ARGUMENT, not a loop bound, and the switch instantiates every value rather than
// rounding up to a power of two: a verify step at n_spec 2 is M = 3, and padding it to 4 would
// spend a third of the FMAs on a row whose output is discarded. The kernel is small enough that
// eight instantiations cost less than that would.
extern "C" int r4d_gemm_fp8a16_nt_mt(long a, long wq, long ws, long c,
                                     int M, int K, int N, int a_ld, int c_ld,
                                     int w_ld, int sw_ld, long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > (int)kMaxMt) return -3;
    if (K % (int)kBlock) return -4;
    if (!w_ld) w_ld = K;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    const int nkb = K / (int)kBlock;
    if (!sw_ld) sw_ld = nkb;

    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    const uint32_t rb = mvm_rb(M);
    const uint32_t blocks =
        std::min<uint32_t>(2048u, (rows + kDenseWaves * rb - 1) / (kDenseWaves * rb));
#define R4D_MVMT_RB(NM, RBV)                                                                    \
    hipLaunchKernelGGL((mvm_kernel<NM, false, uint16_t, RBV>), dim3(blocks), dim3(kDenseThreads), \
                       0,                                                                        \
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws,             \
                       (const uint16_t*)a, (uint32_t)a_ld, (uint16_t*)c, (uint32_t)c_ld,         \
                       rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld, (uint32_t)M)
#define R4D_MVMT(NM)                                                                            \
    do { if (rb == 4) R4D_MVMT_RB(NM, 4);                                                       \
         else if (rb == 2) R4D_MVMT_RB(NM, 2);                                                  \
         else R4D_MVMT_RB(NM, 1); } while (0)
    switch (M) {
        case 1: R4D_MVMT(1); break;
        case 2: R4D_MVMT(2); break;
        case 3: R4D_MVMT(3); break;
        case 4: R4D_MVMT(4); break;
        case 5: R4D_MVMT(5); break;
        case 6: R4D_MVMT(6); break;
        case 7: R4D_MVMT(7); break;
        case 8: R4D_MVMT(8); break;
        default: return -3;
    }
#undef R4D_MVMT
#undef R4D_MVMT_RB
    return hipGetLastError() == hipSuccess ? 0 : -6;
}


// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f32 out, with the SCALES IN EACH ROW'S OWN TAIL.
//
// This is the lm_head, the last bf16 weight in an otherwise fp8 model: 248320 rows by 5120, which
// is 1.27 GiB a rank a step and 10% of everything a decode step reads. As bf16 the projection runs
// at the card's stream rate on those bytes, so the only lever is FEWER BYTES.
//
// WHY THE SCALES RIDE IN THE ROW instead of the separate [ceil(N/128)][ceil(K/128)] plane every
// other fp8 weight here uses. `logits_gemm` has exactly ONE weight operand -- there is no second
// tensor in the schema to hand a scale plane through, and giving it one is a change to the op, to
// ref, and to every architecture that declares it. The alternative, appending the plane to the end
// of the same stream, does not survive TENSOR PARALLELISM: lm_head is ROW-sharded, and the loader
// takes a rank's rows as a byte range `rows * row_bytes` long, which would copy codes only and
// leave the scales behind. Interleaving makes every row self-contained, so the shard stays a plain
// byte range and needs to know nothing about the layout.
//
// It costs a duplicate: a 128-row block shares one scale per 128 columns and storing it per row
// repeats it 128 times. At K = 5120 that is 40 scales, 80 bytes against 5120 code bytes -- 1.6%,
// against the 50% the format saves. The kernel gets a slightly better deal than it pays: the row's
// scales are on the same pages as its codes.
//
//   a    [M][K]          bf16, `a_ld` elements a row (0 = K)
//   wq   [N][lm_row_bytes(K)]  E4M3 codes, nkb bf16 scales, pad; `w_ld` BYTES a row (0 = that)
//   c    [M][N]          f32,  `c_ld` elements a row (0 = N)
//
// M up to 32 is one pass: exact instantiations to 8, rounded up to a multiple of 4 above that
// with the padded rows discarded in the epilogue. Chunking re-reads the whole 636 MiB once a
// chunk, which is the entire saving, so the ceiling is as high as the accumulators allow.
//
// IT STILL HAS TO SERVE EVERY M, and that is not a preference. The bands of one op can resolve to
// different kernels, and they all read the same stored bytes -- so a row that re-layouts a weight
// and covers only part of the range leaves the rest to a kernel reading its codes as something
// else. The failure mode is not a slow path but a wrong one: if the band above 32 fell to the bf16
// projection it would read E4M3 codes as bf16, and the model would return EMPTY completions as soon
// as enough sequences batched together while every request under 32 rows stayed perfect.
// core/build/rad_builder.cpp refuses that at declare time; this covers the range so there is
// nothing to refuse. Above 32 rows the weight is read once
// a chunk, which is what the bf16 kernel cost anyway, and a batch that wide amortises it over
// thirty-two sequences' worth of tokens.
template <bool I8>
static int logits_gemm_8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                         int w_ld, long stream);

extern "C" int r4d_logits_gemm_fp8(long a, long wq, long c, int M, int K, int N,
                                   int a_ld, int c_ld, int w_ld, long stream) {
    return logits_gemm_8<false>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}

// The int8 head: the same stored row -- codes, the row's bf16 scales in its tail, the pad -- with
// i8 codes and each row's OWN scale per 128 columns where the E4M3 head repeats its block's.
extern "C" int r4d_logits_gemm_i8(long a, long wq, long c, int M, int K, int N,
                                  int a_ld, int c_ld, int w_ld, long stream) {
    return logits_gemm_8<true>(a, wq, c, M, K, N, a_ld, c_ld, w_ld, stream);
}

// The int4 head (logits_gemm_i4): r4d_fp8::lm4_row_bytes' row, the WMMA tile at every M -- the
// matvec's M <= 2 edge over the fp8 row is not worth a second int4 kernel for the one-row calls a
// speculative deployment makes only at prefill. FKB: four blocks a round at MT <= 2, two above,
// which is the fp8 rule at the same staged BYTES; R4D_LM4_FKB (1, 2 or 4) pins it.
// The common body of both int4 heads: `group` 128 symmetric (logits_gemm_i4), or 64 / 128
// asymmetric (logits_gemm_u4g64 / _u4g128).
extern "C" int r4d_logits_gemm_q4(long a, long wq, long c, int M, int K, int N,
                                  int a_ld, int c_ld, int w_ld, int group, int asym, long stream) {
    if (!a || !wq || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % (int)kBlock) return -4;
    if (!(group == 128 || (asym && group == 64))) return -3;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if (!w_ld) w_ld = (int)(asym ? r4d_fp8::lm4a_row_bytes(K, group) : r4d_fp8::lm4_row_bytes(K));
    if (w_ld % 16) return -5;
    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    const uint32_t fpin = env_u32_lm("R4D_LM4_FKB");
    for (int m0 = 0; m0 < M; m0 += 64) {
        const int Mc = (M - m0) < 64 ? (M - m0) : 64;
        const uint32_t mt = (uint32_t)((Mc + 15) / 16);
        uint32_t fkb = (fpin == 1u || fpin == 2u || fpin == 4u) ? fpin : (mt <= 2u ? 4u : 2u);
        while (fkb > 1u && (cols % (fkb * kBlock)) != 0u) fkb >>= 1;
        /* R4D_LM4_NPW=2: two weight tiles a wave (bit-identical, see the kernel). Needs the staged
         * rows to fit 64 KiB, so the round is at most two scale blocks. Default 1 = unchanged. */
        /* LEAN: unset = auto (1 from MT >= 2, where the default unrolling spills; 0 at MT = 1, where
         * it does not and the rolled loop is ~10 % slower), 0 = never, 1 / 2 = always. */
        const char* lenv = std::getenv("R4D_LM4_LEAN");
        const uint32_t lean = (lenv && *lenv) ? std::min<uint32_t>((uint32_t)std::strtoul(lenv, nullptr, 10), 2u)
                                              : (mt >= 2u ? 1u : 0u);
        uint32_t npw = (env_u32_lm("R4D_LM4_NPW") == 2u) ? 2u : 1u;
        if (!asym) npw = 1u;
        if (npw == 2u && fkb > 2u) fkb = 2u;
        const uint32_t per = kDenseWaves * 16u * npw;
        const uint32_t blocks = std::min<uint32_t>(4096u, (rows + per - 1u) / per);
        const uint16_t* ac = (const uint16_t*)a + (size_t)m0 * a_ld;
        float* cc = (float*)c + (size_t)m0 * c_ld;
#define R4D_LM4K(MTV, FKV, GSV, AV, NV, LV)                                                  \
    hipLaunchKernelGGL((lm_wmma_i4_kernel<MTV, FKV, GSV, AV, NV, LV>), dim3(blocks),         \
                       dim3(kDenseThreads), 0, (hipStream_t)stream, (const uint8_t*)wq, ac,  \
                       (uint32_t)a_ld, cc, (uint32_t)c_ld, rows, cols, (uint32_t)w_ld, (uint32_t)Mc)
        /* the knobs only exist on the asymmetric heads; the symmetric i4 head stays as it was */
#define R4D_LM4W3(MTV, FKV, GSV, AV)                                                         \
    do { if (!(AV) || (lean == 0u && npw == 1u)) R4D_LM4K(MTV, FKV, GSV, AV, 1, 0);          \
         else if (npw == 2u && (FKV) <= 2) {                                                  \
             if (lean == 0u) R4D_LM4K(MTV, FKV, GSV, AV, (FKV <= 2 ? 2 : 1), 0);             \
             else if (lean == 1u) R4D_LM4K(MTV, FKV, GSV, AV, (FKV <= 2 ? 2 : 1), 1);        \
             else R4D_LM4K(MTV, FKV, GSV, AV, (FKV <= 2 ? 2 : 1), 2); }                      \
         else if (lean == 0u) R4D_LM4K(MTV, FKV, GSV, AV, 1, 0);                             \
         else if (lean == 1u) R4D_LM4K(MTV, FKV, GSV, AV, 1, 1);                             \
         else R4D_LM4K(MTV, FKV, GSV, AV, 1, 2); } while (0)
#define R4D_LM4W2(MTV, FKV)                                                                  \
    do { if (!asym) R4D_LM4W3(MTV, FKV, 128, false);                                         \
         else if (group == 64) R4D_LM4W3(MTV, FKV, 64, true);                                \
         else R4D_LM4W3(MTV, FKV, 128, true); } while (0)
#define R4D_LM4W(MTV) do { if (fkb == 4u) R4D_LM4W2(MTV, 4); else if (fkb == 2u) R4D_LM4W2(MTV, 2); \
                           else R4D_LM4W2(MTV, 1); } while (0)
        switch (mt) {
            case 1:  R4D_LM4W(1); break;
            case 2:  R4D_LM4W(2); break;
            case 3:  R4D_LM4W(3); break;
            default: R4D_LM4W(4); break;
        }
#undef R4D_LM4W
#undef R4D_LM4W2
#undef R4D_LM4W3
#undef R4D_LM4K
        if (hipGetLastError() != hipSuccess) return -6;
    }
    return 0;
}

extern "C" int r4d_logits_gemm_i4(long a, long wq, long c, int M, int K, int N,
                                  int a_ld, int c_ld, int w_ld, long stream) {
    return r4d_logits_gemm_q4(a, wq, c, M, K, N, a_ld, c_ld, w_ld, 128, 0, stream);
}

template <bool I8>
static int logits_gemm_8(long a, long wq, long c, int M, int K, int N, int a_ld, int c_ld,
                         int w_ld, long stream) {
    if (!a || !wq || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (K % (int)kBlock) return -4;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if (!w_ld) w_ld = (int)r4d_fp8::lm_row_bytes(K);
    /* BOTH PATHS BELOW LOAD THE WEIGHT THROUGH A uint4, at `row * w_ld + 16k`, so an odd w_ld is
     * not a slow row -- it is a misaligned load on every second row. It is the layout's job to pad
     * (r4d_fp8_lm.h) and this is the floor that says so out loud, because the only thing that ever
     * gets this wrong is a caller passing a stride it derived itself. */
    if (w_ld % 16) return -5;

    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;

    /* WMMA FROM M = 3, THE MATVEC AT OR BELOW IT. The tile's rate is nearly flat in M across this
     * range, while the matvec's falls away from M = 4 up, so the crossover sits wherever the matvec
     * leaves its own best row. At M = 2 the matvec is marginally ahead and at M = 1 clearly so --
     * the activation is already amortised over the whole weight there -- so the matvec keeps the
     * bottom of the range and the tile takes the whole of what a speculative verify step issues. A
     * step runs 1 + n_spec rows a SEQUENCE, so at DFlash2 depth 7 every served M is 8, 16, 24 ...
     * R4D_LM_WMMA pins the choice (1 = tile, 2 = matvec). Both paths read the SAME stored bytes --
     * this is one kernel row with one layout, so there is no band and no claim to get wrong. */
    const uint32_t wpin = env_u32_lm("R4D_LM_WMMA");
    if (wpin == 1u || (wpin != 2u && M > 2)) {
        /* THE SHAPE OF THE STAGED TILE; see lm_wmma_kernel for the argument. FKB is how many
         * 128-wide scale blocks one staging round covers. R4D_LM_FKB pins it without a rebuild;
         * unset means the rule. It is bit-identical either way, so a pin can only change the
         * time. */
        const uint32_t fpin = env_u32_lm("R4D_LM_FKB");
        /* Sixty-four rows a pass, not thirty-two: the accumulator is four WMMA tiles and costs
         * thirty-two registers, so the chunk is bounded by nothing that matters and one weight
         * read covers eight concurrent sequences at DFlash2 depth 7. */
        for (int m0 = 0; m0 < M; m0 += 64) {
            const int Mc = (M - m0) < 64 ? (M - m0) : 64;
            const uint32_t mt = (uint32_t)((Mc + 15) / 16);
            /* FKB = 4 is not available: kRows * (512 + 8) is 66560 bytes of LDS against a 64 KiB
             * workgroup limit, and kRows is eight waves owning a 16-row tile each. */
            uint32_t fkb = (fpin >= 1u && fpin <= 2u) ? fpin : lm_fkb_rule(mt);
            /* FK has to divide K. Every real lm head is a multiple of 512, but the harness and the
             * reference shapes are not, and a staging round that runs off the end of a row would
             * read the next row's scales as codes. */
            while (fkb > 1u && (cols % (fkb * kBlock)) != 0u) fkb >>= 1;
            const uint32_t per = kDenseWaves * 16u;      /* weight rows a workgroup covers */
            const uint32_t blocks = std::min<uint32_t>(4096u, (rows + per - 1u) / per);
            const uint16_t* ac = (const uint16_t*)a + (size_t)m0 * a_ld;
            float* cc = (float*)c + (size_t)m0 * c_ld;
#define R4D_LMW2(MTV, FKV)                                                                  \
    hipLaunchKernelGGL((lm_wmma_kernel<MTV, FKV, I8>), dim3(blocks), dim3(kDenseThreads), 0, \
                       (hipStream_t)stream, (const uint8_t*)wq, ac, (uint32_t)a_ld, cc,      \
                       (uint32_t)c_ld, rows, cols, (uint32_t)w_ld, (uint32_t)Mc)
#define R4D_LMW(MTV) do { if (fkb == 2u) R4D_LMW2(MTV, 2); else R4D_LMW2(MTV, 1); } while (0)
            switch (mt) {
                case 1:  R4D_LMW(1); break;
                case 2:  R4D_LMW(2); break;
                case 3:  R4D_LMW(3); break;
                default: R4D_LMW(4); break;
            }
#undef R4D_LMW

#undef R4D_LMW2
            if (hipGetLastError() != hipSuccess) return -6;
        }
        return 0;
    }

  for (int m0 = 0; m0 < M; m0 += 32) {
    const int Mc = (M - m0) < 32 ? (M - m0) : 32;
    const uint32_t rb = mvm_rb(Mc);
    const uint32_t blocks =
        std::min<uint32_t>(2048u, (rows + kDenseWaves * rb - 1) / (kDenseWaves * rb));
    const uint16_t* ac = (const uint16_t*)a + (size_t)m0 * a_ld;
    float* cc = (float*)c + (size_t)m0 * c_ld;
#define R4D_LGF8_RB(NM, RBV)                                                                    \
    hipLaunchKernelGGL((mvm_kernel<NM, true, float, RBV, I8>), dim3(blocks), dim3(kDenseThreads), \
                       0,                                                                        \
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)nullptr,        \
                       ac, (uint32_t)a_ld, cc, (uint32_t)c_ld,                                   \
                       rows, cols, (uint32_t)w_ld, 0u, (uint32_t)Mc)
#define R4D_LGF8(NM)                                                                            \
    do { if (rb == 4) R4D_LGF8_RB(NM, 4);                                                       \
         else if (rb == 2) R4D_LGF8_RB(NM, 2);                                                  \
         else R4D_LGF8_RB(NM, 1); } while (0)
    switch (Mc) {
        case 1: R4D_LGF8(1); break;
        case 2: R4D_LGF8(2); break;
        case 3: R4D_LGF8(3); break;
        case 4: R4D_LGF8(4); break;
        case 5: R4D_LGF8(5); break;
        case 6: R4D_LGF8(6); break;
        case 7: R4D_LGF8(7); break;
        case 8: R4D_LGF8(8); break;
        default:
            if (Mc <= 12)      R4D_LGF8(12);
            else if (Mc <= 16) R4D_LGF8(16);
            else if (Mc <= 20) R4D_LGF8(20);
            else if (Mc <= 24) R4D_LGF8(24);
            else if (Mc <= 28) R4D_LGF8(28);
            else              R4D_LGF8(32);
            break;
    }
#undef R4D_LGF8
#undef R4D_LGF8_RB
    if (hipGetLastError() != hipSuccess) return -6;
  }
    return 0;
}

// C[1,N] = A[1,K] @ dequant(Wq)[N,K]^T.
//
//   a    [K]                        bf16
//   wq   [N][K]                     E4M3, row-major, `w_ld` bytes a row (0 = K)
//   ws   [ceil(N/128)][ceil(K/128)] bf16, `sw_ld` elements a row (0 = ceil(K/128))
//   c    [N]                        bf16
//
// NO LDS STAGING, deliberately. With uint4 loads a lane reads 16 consecutive activations, so 16
// lanes land on the same LDS bank and the conflicts cost more than the staging saves; `a` is small
// enough that every wave in the block keeps it in L1 regardless.
//
// Returns 0, or a negative code for a shape it does not serve.
extern "C" int r4d_gemm_fp8a16_nt_m1(long a, long wq, long ws, long c,
                                     int M, int K, int N, int w_ld, int sw_ld, int rb,
                                     long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > 1) return -3;
    // `srow[c / 128]` reads one scale per 128 columns; a ragged last block would read past the
    // plane. Every dense tensor in the target checkpoint has K a multiple of 1024, so this is a
    // guard rather than a path -- but a guard that is absent is a kernel that reads other people's
    // memory on the first model that is not this one.
    if (K % (int)kBlock) return -4;
    if (rb != 0 && rb != 1 && rb != 2 && rb != 4 && rb != 8) return -5;
    if (!w_ld) w_ld = K;
    const int nkb = K / (int)kBlock;
    if (!sw_ld) sw_ld = nkb;

    const Bf16V xv{ (const uint16_t*)a };
    const uint32_t rows = (uint32_t)N, cols = (uint32_t)K;
    uint32_t r = rb ? (uint32_t)rb : rb_default(rows, cols);

    if (r > 1u) {
        const uint32_t per = kDenseWaves * r;
        const uint32_t blocks = std::min<uint32_t>(2048u, (rows + per - 1) / per);
#define R4D_MVRB(NB)                                                                          \
    hipLaunchKernelGGL((mv_rb_kernel<NB, Bf16V>), dim3(blocks), dim3(kDenseThreads), 0,        \
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws, xv,       \
                       (uint16_t*)c, rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld)
        if (r == 2u)      { R4D_MVRB(2); }
        else if (r == 4u) { R4D_MVRB(4); }
        else              { R4D_MVRB(8); }
#undef R4D_MVRB
        return hipGetLastError() == hipSuccess ? 0 : -6;
    }
    const uint32_t blocks = std::min<uint32_t>(2048u, (rows + kDenseWaves - 1) / kDenseWaves);
    hipLaunchKernelGGL((mv_kernel<Bf16V>), dim3(blocks), dim3(kDenseThreads), 0,
                       (hipStream_t)stream, (const uint8_t*)wq, (const uint16_t*)ws, xv,
                       (uint16_t*)c, rows, cols, (uint32_t)w_ld, (uint32_t)sw_ld);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

// ============================================================== the plugin ABI entry point
// The helpers this uses are libr4d/r4d_args.h's.  See the note in r4d_model_bf16.hip.
enum { LG_X = 0, LG_W, LG_Y };

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)
static inline int lg_dt_bf16(const RadTensor* t) {
    return (t && t->dtype == RAD_BF16) ? RAD_OK : RAD_E_DTYPE;
}
/* ------------------------------------------------------------------ logits_gemm, fp8 weight
 *
 * The same projection over a quantised head. The lm_head is 248320 x 5120 on this model -- 1.27
 * GiB a rank a step, 10% of everything a decode step reads, and the last bf16 weight in an
 * otherwise fp8 container. The bf16 kernel above runs at the card's stream rate, so the only
 * lever is fewer bytes.
 *
 * The weight arrives as [n_vocab][r4d_fp8::lm_row_bytes(n_embd)]: E4M3 codes, the row's own
 * bf16 scales in its tail, and the 16-byte alignment pad after those.
 * r4d_rad_layout_fp8_logits explains why the scales are interleaved rather than kept in the
 * separate plane every other fp8 weight here uses -- it is tensor parallelism, not taste.
 */
static int rad_logits_gemm_8(const RadArgs* a, RadStream s, bool i8) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(lg_dt_bf16(x));
    if (w->dtype != (i8 ? RAD_I8 : RAD_F8E4M3)) return R4D_NO(RAD_E_DTYPE);
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);

    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", 0);
    if (n_vocab <= 0 || n_embd <= 0) return R4D_NO(RAD_E_SHAPE);
    if (n_embd % 128) return R4D_NO(RAD_E_SHAPE);
    const long long stride = r4d_fp8::lm_row_bytes(n_embd);
    /* THE DECLARED SHAPE, not the stored one, because the declared shape is what arrives.
     * core/runtime/ctx.cpp builds a weight's RadTensor from `decl.shape` with the layout hook's
     * `stored_dtype` -- so this operand comes through as [n_vocab][n_embd] E4M3 even though the
     * plane is [n_vocab][n_embd + 2*nkb]. The stored width therefore cannot be checked here and is
     * DERIVED, above, by the same rule the layout hook used. What guarantees the two agree is the
     * engine's layout-tag check, which refuses a container laid out for any other row before a
     * byte of it moves -- a better test than this one could be, and it runs earlier. */
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd)
        return R4D_NO(RAD_E_SHAPE);

    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);

    return (i8 ? r4d_logits_gemm_i8 : r4d_logits_gemm_fp8)(
               r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd, (int)n_vocab,
               (int)x_ld, (int)y_ld, (int)stride, (long)(uintptr_t)s) == 0 ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r4d_rad_logits_gemm_fp8(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_8(a, s, false);
}

/* The int8 head: the same operands and stored row, i8 codes (r4d_rad_layout_i8_logits). */
extern "C" int r4d_rad_logits_gemm_i8(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_8(a, s, true);
}

/* The int4 head: the same operands over r4d_fp8::lm4_row_bytes' row. The layout hook stores it as
 * bytes (RAD_U8), so that is the dtype the declared [n_vocab][n_embd] operand arrives with. */
static int rad_logits_gemm_q4(const RadArgs* a, RadStream s, int group, int asym) {
    const RadTensor* x = r4d_opt(a, LG_X);
    const RadTensor* w = r4d_opt(a, LG_W);
    const RadTensor* y = r4d_opt(a, LG_Y);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(lg_dt_bf16(x));
    if (w->dtype != RAD_U8) return R4D_NO(RAD_E_DTYPE);
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", 0);
    if (n_vocab <= 0 || n_embd <= 0 || n_embd % 128) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd)
        return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);
    if (n_embd % group) return R4D_NO(RAD_E_SHAPE);
    const long long stride = asym ? r4d_fp8::lm4a_row_bytes(n_embd, group)
                                  : r4d_fp8::lm4_row_bytes(n_embd);
    return r4d_logits_gemm_q4(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd,
                              (int)n_vocab, (int)x_ld, (int)y_ld, (int)stride, group, asym,
                              (long)(uintptr_t)s) == 0
               ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r4d_rad_logits_gemm_i4(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_q4(a, s, 128, 0);
}
/* The asymmetric int4 heads: (code - zero) * scale, the zeros after the scales in the row. */
extern "C" int r4d_rad_logits_gemm_u4g64(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_q4(a, s, 64, 1);
}
extern "C" int r4d_rad_logits_gemm_u4g128(const RadArgs* a, RadStream s) {
    return rad_logits_gemm_q4(a, s, 128, 1);
}
#undef R4D_NEED

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. See libr4d/r4d_args.h for why there is no adapter
// directory between the core and these launches.
/* ================================================================== the fp8 GEMMs */

extern "C" int r4d_rad_gemm_fp8a16_m1(const RadArgs* a, RadStream s) {
    Fp8Ops g;
    int rc = fp8_ops(a, &g, 0);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_fp8a16_nt_m1_max_m()) return R4D_NO(RAD_E_SHAPE);
    /* An activation scale is meaningless to this kernel -- it reads bf16 directly. A caller that
     * supplies one has asked for the wrong dtype string, and running anyway would silently drop a
     * scale the caller believes was applied. */
    if (g.ascale) return R4D_NO(RAD_E_INVAL);
    /* ONE ROW KEEPS THE ONE-ROW KERNEL, and that is not just thrift: it is the RB-blocked form with
     * its own tuning argument, it is what every non-speculative decode step runs, and routing it
     * through the multi-row entry would change the arithmetic of a deployment that never
     * speculates. The multi-row form is bit-identical at MT = 1 by construction, but "by
     * construction" is a claim and this is a guarantee.
     *
     * Above one row it is the multi-row form, which reads the weight ONCE for all M rows. The
     * alternative at M = 1 + n_spec is the 16-wide fp8a8 tile, which is slower here on identical
     * weight traffic. */
    if (g.M == 1) {
        R4dMvCfg mv;
        r4d_cfg_mv(a, &mv);
        const int e = r4d_gemm_fp8a16_nt_m1(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.bscale),
                                            r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N,
                                            (int)g.w_ld, (int)g.sw_ld, mv.rb, (long)(uintptr_t)s);
        return e == 0 ? RAD_OK : RAD_E_SHAPE;
    }
    /* `c_ld` is 0 because fp8_ops has already refused any output whose row pitch is not N. */
    const int e = r4d_gemm_fp8a16_nt_mt(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.bscale),
                                        r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N,
                                        (int)g.a_ld, 0, (int)g.w_ld, (int)g.sw_ld,
                                        (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

/* Split-K's partial plane. Called at declare with the band's upper bound, so the arena is sized for
 * the worst step in the band -- which is what makes the answer safe at every M inside it, since the
 * plane is ksplit*M*N and both factors only fall as M does.
 *
 * SIZED FROM THE PARAMETERS, not the operands. The scratch hook is a DECLARE-time question and the
 * geometry is what declare has; a hook that read `y->shape` would answer 0 whenever the core asked
 * before the buffers existed, and 0 is not an error here -- it means "run unsplit", which is correct
 * and slow. A correctness-preserving silent slowdown is the worst kind, because nothing reports it,
 * and split-K is a large share of decode on this shape. */
