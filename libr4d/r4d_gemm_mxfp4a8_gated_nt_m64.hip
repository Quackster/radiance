// r4d_gemm_mxfp4a8_gated_nt_m64.hip -- OCP-MXFP4 weight, per-row E4M3 activation, with the
// SwiGLU and the whole-row fp8 quantiser folded into the GEMM epilogue. This is the
// `gemm_nt_q_gated_mx` row (r4d_rows.cpp): the fused twin of `gemm_mxfp4a8_nt_m64` followed by
// `gated_quant_fp8_row`.
//
// WHY THIS IS ITS OWN OP AND NOT A dtype ON `gemm_nt_q_gated`. That op's schema has seven
// operands and no `b_ref`; the full operand list is positional and the core checks arity EXACTLY
// (core/runtime/issue.cpp), so appending an eighth operand would make every existing 7-operand
// caller -- arch/common/rad_block_mlp_fp8.h, rad_block_dflash2.h, rad_block_dspark.h and
// tests/ref_test.cpp -- a living arity error. The MX weight genuinely carries an eighth operand,
// the per-output-row reference exponent (r4d_layout.cpp's `Q_BREF`), so the right move is a new op
// with a schema of its own rather than an invisible extra slot. `gemm_nt_q` carries b_ref because
// it was written with the slot from the start (oGemmNtQ in r4d_rows.cpp).
//
// THE STRUCTURAL DIFFICULTY, AND HOW THIS KERNEL SOLVES IT. In the m64/tiled kernels a lane's
// accumulator owns columns scattered across the N tiles and the gate row j and the up row j + nout
// usually land in DIFFERENT lanes, so a straight port has nothing to pair. This kernel makes each
// wave own an OUTPUT column tile t and compute BOTH weight tiles t (gate) and nout/16 + t (up) in
// the same lane: the WMMA's fragment-to-column map does not depend on which weight rows were
// multiplied, so `acc_gate[j][i][e]` and `acc_up[j][i][e]` are two values for the SAME token m and
// the SAME output column n. The pairing is therefore free and exact.
//
// THE PER-ROW AMAX DOES NOT CLOSE INSIDE A BLOCK, unlike the fp8 gated fold whose 128-column group
// is exactly a block's output width. Here the group is the WHOLE row, which spans every N tile, so
// the fold is TWO LAUNCHES behind one entry point:
//
//   pass 1  the gated mxfp4a8 GEMM: computes gate and up for a row tile x output-column tile,
//           applies the fold `C = acc * 2^(Wref[row]-127) * As[m]`, rounds to bf16, computes
//           silu(gate)*up and rounds THAT to bf16, and stores the bf16 product. The bf16 product
//           plane is exactly `gemm_nt_q_gated`'s own `out_bf16`, and it is where the amax is taken.
//   pass 2  the per-row quantiser over that product plane (r4d_quant_act_fp8_row.hip's kernel,
//           restated here because that one lives in another translation unit). One block a row,
//           whole-row amax, scale = amax/448 (1 for a zero row), E4M3 codes.
//
// BYTE-IDENTITY IS THE POINT. The pair this replaces is
// `gemm_nt_q(mxfp4a8)` -> bf16 [M, 2*nout] -> `gated_quant_fp8_row`. Pass 1 reproduces the m64
// GEMM's accumulator bit-for-bit (same cfg_for launch law, same fragment-order weight read, same
// folded-magnitude unpack, same split-K reduction order) and rounds the scaled value to bf16 before
// the gate, exactly as the m64 epilogue does. Pass 2 is quant_act_fp8_row, which the selftest at
// r4d_selftest.cpp:3489 already certifies byte-identical to gated_quant_fp8_row's quantiser. So the
// whole fold is the pair, byte for byte.
//
// THE DEFAULT CONFIGURATION IS THE UNFUSED ROW'S. cfg_for() is called with the same arguments and
// this row declares no tunables, so the split, tile and wave count are the m64 row's own defaults
// and the two accumulations agree. Pinning a tuned split on `gemm_mxfp4a8_nt_m64` without pinning
// the same point here would move the unfused row's last bits; that is a deployment choice and is
// called out in r4d_rows.cpp rather than hidden.
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

/* K per E8M0 exponent, fixed by the OCP MX format. Bound to the one canonical enum (r4d_args.h) so
 * a drift from the layout hook's group width is a compile error, not a silent block-gain mis-scale. */
#define R4D_GATED_MX_GROUP R4D_MXFP4_GROUP
static_assert(R4D_MXFP4_GROUP == 32, "OCP-MX group width changed; revisit every MX kernel");
#define R4D_GATED_MX_MAX_M 64
#define WAVE32 32

typedef int      v2i    __attribute__((ext_vector_type(2)));

// Folded magnitudes: kGatedMag[d] is the eight e2m1 magnitudes scaled by 2^-d as packed e4m3
// bytes. Identical table and unpack to r4d_gemm_mxfp4a8_nt_m64.hip; copied (with a distinct name so
// the two translation units do not collide at device link) rather than shared, because the m64
// kernel's table is not exported and a header for it would be a second copy with the same problem.
__device__ __constant__ unsigned int kGatedMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

// Eight e2m1 codes (four packed bytes) -> eight e4m3 bytes in k order, block exponent folded in.
__device__ __forceinline__ v2i r4d_mxfp4_unpack8_g(unsigned int wv, int dsh) {
  const unsigned int t0 = kGatedMag[dsh][0], t1 = kGatedMag[dsh][1];
  const unsigned int ev = wv & 0x0F0F0F0Fu;
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  v2i out;
  out[0] = (int)__builtin_amdgcn_perm(bo, be, 0x05010400u);
  out[1] = (int)__builtin_amdgcn_perm(bo, be, 0x07030602u);
  return out;
}

// ------------------------------------------------------------------ pass 1: the gated GEMM
//
// `nout` is the FUSED output width; the weight has 2*nout rows and `Nf = 2*nout` is the width the
// e8m0 plane and the reference vector are indexed by. Operand 4 is the product plane, bf16
// [M][nout], and it is the only thing this kernel writes.
template <int MT, int NPW>
__global__ void r4d_gemm_mxfp4a8_gated_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, unsigned char* __restrict__ P,
    int M, int K, int nout, int Pld, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int Nf = 2 * nout;

  v8f accg[NPW][MT], accu[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) { accg[j][i] = (v8f)(0.f); accu[j][i] = (v8f)(0.f); }

  // Output column tiles this wave owns, CLAMPED to the last real one so the weight read never walks
  // off the end. The up tile is always nout/16 further on and is clamped with the same index.
  const int ntmax = nout / 16 - 1;
  const int utbase = nout / 16;
  int wtile[NPW], grow[NPW], urow[NPW];
  unsigned char wref_g[NPW], wref_u[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    grow[j] = wtile[j] * 16 + nlane;
    urow[j] = (utbase + wtile[j]) * 16 + nlane;
    wref_g[j] = Wref[grow[j]];
    wref_u[j] = Wref[urow[j]];
  }

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps = K / 16;

  // The K loop is the m64 kernel's, with the gate and up fragments in flight together. Same double
  // buffer, same loads, same WMMA order -- which is what keeps the two accumulators bit-identical
  // to two runs of the unfused GEMM.
  {
    const int kg0 = kbase;
    const int kstep0 = kg0 / 16;
    const int blk0 = kg0 / R4D_GATED_MX_GROUP;
    v2i af_cur[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + nlane;
      r = r < M - 1 ? r : M - 1;
      const unsigned int* pa = (const unsigned int*)(A + (size_t)r * K + kg0 + mhalf);
      af_cur[i][0] = (int)pa[0];
      af_cur[i][1] = (int)pa[1];
    }
    unsigned int wvg_cur[NPW], wvu_cur[NPW];
    int dg_cur[NPW], du_cur[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      wvg_cur[j] = Wp[((size_t)wtile[j] * ksteps + kstep0) * WAVE32 + lane];
      wvu_cur[j] = Wp[((size_t)(utbase + wtile[j]) * ksteps + kstep0) * WAVE32 + lane];
      int dg = (int)wref_g[j] - (int)Ws[(size_t)blk0 * Nf + grow[j]];
      int du = (int)wref_u[j] - (int)Ws[(size_t)blk0 * Nf + urow[j]];
      dg_cur[j] = dg < 0 ? 0 : (dg > 15 ? 15 : dg);
      du_cur[j] = du < 0 ? 0 : (du > 15 ? 15 : du);
    }
    for (int k = 0; k < Ks; k += 16) {
      v2i af_next[MT];
      unsigned int wvg_next[NPW], wvu_next[NPW];
      int dg_next[NPW], du_next[NPW];
      const int k1 = k + 16;
      if (k1 < Ks) {
        const int kg1 = kbase + k1;
        const int kstep1 = kg1 / 16;
        const int blk1 = kg1 / R4D_GATED_MX_GROUP;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          int r = (mt0 + i) * 16 + nlane;
          r = r < M - 1 ? r : M - 1;
          const unsigned int* pa = (const unsigned int*)(A + (size_t)r * K + kg1 + mhalf);
          af_next[i][0] = (int)pa[0];
          af_next[i][1] = (int)pa[1];
        }
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          wvg_next[j] = Wp[((size_t)wtile[j] * ksteps + kstep1) * WAVE32 + lane];
          wvu_next[j] = Wp[((size_t)(utbase + wtile[j]) * ksteps + kstep1) * WAVE32 + lane];
          int dg = (int)wref_g[j] - (int)Ws[(size_t)blk1 * Nf + grow[j]];
          int du = (int)wref_u[j] - (int)Ws[(size_t)blk1 * Nf + urow[j]];
          dg_next[j] = dg < 0 ? 0 : (dg > 15 ? 15 : dg);
          du_next[j] = du < 0 ? 0 : (du > 15 ? 15 : du);
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const v2i wfg = r4d_mxfp4_unpack8_g(wvg_cur[j], dg_cur[j]);
        const v2i wfu = r4d_mxfp4_unpack8_g(wvu_cur[j], du_cur[j]);
#pragma unroll
        for (int i = 0; i < MT; i++) {
          accg[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af_cur[i], wfg, accg[j][i]);
          accu[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af_cur[i], wfu, accu[j][i]);
        }
      }
      // Advance the double buffer -- only when a next step was actually loaded. On the final
      // K-step `k1 == Ks` leaves `af_next/wvg_next/wvu_next/dg_next/du_next` unset, and reading
      // them (as the carried `*_cur` values) would be an uninitialised read; nothing consumes them
      // after the loop, so the honest fix is to not advance (mirrors r4d_gemm_mxfp4a8_nt_m64.hip).
      if (k1 < Ks) {
#pragma unroll
        for (int i = 0; i < MT; i++) af_cur[i] = af_next[i];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          wvg_cur[j] = wvg_next[j]; wvu_cur[j] = wvu_next[j];
          dg_cur[j] = dg_next[j]; du_cur[j] = du_next[j];
        }
      }
    }
  }

  // Reduce the SK partials for gate and up in LDS, then fold and store the product. Each of the two
  // halves is a 256-element tile (32 lanes x 8 values) and they share a 512-wide slot so the two
  // reductions are one pass.
  const int ncols = WV * NPW;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int base = ((col_local * NPW + j) * SK + ksplit) * 512;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        red[base + lane * 8 + e] = accg[j][i][e];
        red[base + 256 + lane * 8 + e] = accu[j][i][e];
      }
    }
    __syncthreads();
    // In-place reduction to the s = 0 slot. Each (col, off) belongs to exactly one thread, and a
    // column's slots are disjoint from every other column's, so the writes cannot race a read.
    for (int t = tid; t < ncols * 512; t += blockDim.x) {
      const int col = t >> 9, off = t & 511;
      float v = 0.f;
      for (int ss = 0; ss < SK; ss++) v += red[(col * SK + ss) * 512 + off];
      red[(col * SK) * 512 + off] = v;
    }
    __syncthreads();
    for (int t = tid; t < ncols * 256; t += blockDim.x) {
      const int col = t >> 8, o = t & 255;
      const int n = (blockIdx.x * ncols + col) * 16 + ((o >> 3) & 15);
      const int m = (mt0 + i) * 16 + ((o >> 7) << 3) + (o & 7);
      if (n >= nout || m >= M) continue;
      // THE FOLD: the same `acc * 2^(Wref[row]-127) * As[m]` the unfused m64 epilogue applies,
      // rounded to bf16, then silu(gate)*up rounded to bf16. Two roundings, in the pair's order.
      const float g = red[(col * SK) * 512 + o];
      const float u = red[(col * SK) * 512 + 256 + o];
      const float frg = __int_as_float((int)Wref[n] << 23);
      const float fru = __int_as_float((int)Wref[nout + n] << 23);
      const float gg = bf16_to_f32(f32_to_bf16(g * frg * As[m]));
      const float uu = bf16_to_f32(f32_to_bf16(u * fru * As[m]));
      const float a = gg / (1.0f + expf(-gg));
      ((uint16_t*)P)[(size_t)m * Pld + n] = f32_to_bf16(a * uu);
    }
  }
}

// ------------------------------------------------------------------ pass 2: the per-row quantiser
//
// r4d_quant_act_fp8_row.hip's kernel, restated: one block a row, 256 threads, 4-column chunks,
// whole-row amax, scale = amax/448 (1 for a zero row), clamp +-448, cvt_pk_fp8_f32. The selftest at
// r4d_selftest.cpp:3489 certifies this arithmetic byte-identical to gated_quant_fp8_row's own
// quantiser, which is the second stage of the pair the fold stands for.
constexpr uint32_t kQThr = 256;
__global__ __launch_bounds__(kQThr) void mxfp4a8_gated_quant_row_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
  const uint32_t row = blockIdx.x, tid = threadIdx.x;
  const size_t xrow = (size_t)row * x_ld, qrow = (size_t)row * q_ld;

  float m = 0.0f;
  for (uint32_t c0 = tid * 4u; c0 < K; c0 += kQThr * 4u) {
    const uint16_t* p = x + xrow + c0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float v = __uint_as_float((uint32_t)p[j] << 16);
      m = fmaxf(m, fabsf(v));
    }
  }
  m = r4d_wave_max(m);
  __shared__ float red[kQThr / 32];
  const uint32_t wave = tid / 32, lane = tid & 31;
  if (lane == 0) red[wave] = m;
  __syncthreads();
  if (wave == 0) {
    float v = (lane < kQThr / 32) ? red[lane] : 0.0f;
    v = r4d_wave_max(v);
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  const float amax = red[0];
  const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
  if (tid == 0) s[row * s_ld] = sc;
  const float inv = 1.0f / sc;

  for (uint32_t c0 = tid * 4u; c0 < K; c0 += kQThr * 4u) {
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

// ============================================================== the plugin ABI entry points
extern "C" int64_t r4d_rad_gemm_mxfp4a8_gated_m64_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0);
    const long long N = r4d_geti_or(a, "N", 0);
    if (M < 1 || N < 2 || (N & 1)) return 0;
    const long long nout = N / 2;
    /* The product plane the fold stages is exactly `out_bf16` when the caller has one; only then
     * is there no scratch to size. Operand 6 is that optional output. */
    if (r4d_opt(a, 6) && r4d_opt(a, 6)->data) return 0;
    return (int64_t)M * nout * 2;
}

extern "C" int r4d_rad_gemm_mxfp4a8_gated_m64(const RadArgs* a, RadStream s) {
    const RadTensor* x   = r4d_opt(a, 0);   /* a       [M, K] e4m3            */
    const RadTensor* xs  = r4d_opt(a, 1);   /* a_scale [M, 1] f32             */
    const RadTensor* w   = r4d_opt(a, 2);   /* b       mxfp4 fragment [2nout,K] */
    const RadTensor* wsc = r4d_opt(a, 3);   /* b_scale [K/32, 2nout] u8 e8m0  */
    const RadTensor* q   = r4d_opt(a, 4);   /* q       [M, nout] e4m3         */
    const RadTensor* sc  = r4d_opt(a, 5);   /* scale   [M, 1] f32             */
    const RadTensor* y   = r4d_opt(a, 6);   /* out_bf16 [M, nout] bf16 (opt)  */
    const RadTensor* wref= r4d_opt(a, 7);   /* b_ref   [2nout] u8 e8m0        */
    if (!x || !xs || !w || !wsc || !q || !sc || !wref) return R4D_NO(RAD_E_INVAL);

    const long long N = r4d_geti_or(a, "N", 0);   /* the WEIGHT's rows = 2*nout */
    const long long K = r4d_geti_or(a, "K", 0);
    if (N < 2 || (N & 1) || K < 1) return R4D_NO(RAD_E_SHAPE);
    if (N % 16 || K % R4D_GATED_MX_GROUP) return R4D_NO(RAD_E_SHAPE);
    /* M IS THE ACTIVATION'S ROW COUNT, NOT THE GEOMETRY'S. The core resolves this row once per
     * BAND, and the geometry carries the band's ceiling -- `M <= 64` means the op parameter is
     * M = 64 -- while each issue narrows the operands to the rows the step actually holds (4 at a
     * decode step). Reading M from the parameter and demanding q->shape[0] == M therefore refuses
     * every call shorter than the band, which is what aborted the trunk MLP at runtime ("shape
     * this kernel does not serve" at M=4 against geometry M=64). The ungated row never sees this
     * because gemm_ops() derives M from the operands; this shim's own operand list has to do the
     * same. A contiguous rank-3 activation folds to (M, K) exactly as gemm_ops folds it. */
    if (x->rank < 2) return R4D_NO(RAD_E_SHAPE);
    long long M = 1;
    for (uint32_t d = 0; d + 1 < x->rank; ++d) M *= x->shape[d];
    if (M < 1 || M > R4D_GATED_MX_MAX_M) return R4D_NO(RAD_E_SHAPE);
    const long long nout = N / 2;
    if (q->rank != 2 || q->shape[0] != M || q->shape[1] != nout) return R4D_NO(RAD_E_SHAPE);
    if (sc->rank != 2 || sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(x) < M * K) return R4D_NO(RAD_E_SHAPE);
    /* THE KERNEL READS A WITH PITCH K (`A + r*K`), so a STRIDED activation whose row pitch is not
     * K is read as the wrong numbers -- `r4d_numel` counts elements and cannot see that, which is
     * exactly why it is not enough. The ungated rows get this from gemm_ops(); this shim has its
     * own operand list, so it makes the same contiguity and 16-byte-alignment checks here, plus the
     * pitch-K equality the kernel's addressing assumes (a contiguous rank-3 [M, a, b] with a*b == K
     * is the rank-2 [M, K] this reads, as gemm_ops folds it). */
    if (!r4d_contig(x)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_row_pitch(x) != K) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);
    if (r4d_numel(xs) < M) return R4D_NO(RAD_E_SHAPE);   /* [M,1] or a flat [M] */
    if (y && (y->rank != 2 || y->shape[0] != M || y->shape[1] != nout)) return R4D_NO(RAD_E_SHAPE);
    if (nout % 4) return R4D_NO(RAD_E_SHAPE);   /* the pass-2 quantiser reads four columns a store */

    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "silu")) return R4D_NO(RAD_E_UNSUPPORTED);

    /* The unfused row's own launch law, called with the same arguments, so the two accumulations
     * agree bit for bit. */
    Cfg c;
    int rc = cfg_for(a, K, M, R4D_GATED_MX_GROUP, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    if (c.npw != 1) return R4D_NO(RAD_E_UNSUPPORTED);   /* the fold is compiled for NPW = 1 */
    if (c.mb < 1 || c.mb > 4) return R4D_NO(RAD_E_UNSUPPORTED);

    /* The product plane. `out_bf16` when the caller holds one, otherwise the scratch arena: the
     * product IS the bf16 tensor the unfused pair passed between its two ops, so writing it to the
     * caller's own output is the correct place and costs no scratch. */
    unsigned char* P = y ? (unsigned char*)r4d_addr(y) : (unsigned char*)a->scratch;
    if (!P) return R4D_NO(RAD_E_INVAL);
    const long long need = M * nout * 2;
    if (!y && a->scratch_bytes < need) return R4D_NO(RAD_E_INVAL);
    /* The product plane's row pitch: the caller's `out_bf16` when it holds one (which may be
     * padded), otherwise the tightly-packed scratch. */
    const int Pld = y ? (int)r4d_row_pitch(y) : (int)nout;
    if (Pld < nout) return R4D_NO(RAD_E_STRIDE);

    const int ntiles = (int)(nout / 16), mtiles = (int)((M + 15) / 16);
    const int ncols = c.wv * 1;
    const dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + c.mb - 1) / c.mb, 1);
    const dim3 block(c.wv * c.sk * WAVE32);
    const size_t smem = (size_t)ncols * c.sk * 512 * sizeof(float);
    if (smem > 64 * 1024) return R4D_NO(RAD_E_UNSUPPORTED);

    hipStream_t st = (hipStream_t)s;
    auto Ap = (const unsigned char*)r4d_addr(x);
    auto Sp = (const float*)r4d_addr(xs);
    auto Wp = (const unsigned char*)r4d_addr(w);
    auto Zp = (const unsigned char*)r4d_addr(wsc);
    auto Rp = (const unsigned char*)r4d_addr(wref);
    auto Gp = (const unsigned char*)r4d_addr(q);
    auto Qs = (float*)r4d_addr(sc);

#define R4D_GATED_LAUNCH(MTv)                                                        \
    hipLaunchKernelGGL((r4d_gemm_mxfp4a8_gated_kernel<MTv, 1>), grid, block, smem, st, \
                       Ap, Sp, Wp, Zp, Rp, P, (int)M, (int)K, (int)nout, Pld, c.wv, c.sk)
    switch (c.mb) {
        case 1: R4D_GATED_LAUNCH(1); break;
        case 2: R4D_GATED_LAUNCH(2); break;
        case 3: R4D_GATED_LAUNCH(3); break;
        case 4: R4D_GATED_LAUNCH(4); break;
        default: return R4D_NO(RAD_E_UNSUPPORTED);
    }
#undef R4D_GATED_LAUNCH
    if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

    /* Pass 2 over the product plane, one block a row. Read off `q`'s pitch so a padded output is
     * handled, and the product plane is contiguous. The pitches are `long long` and tested for a
     * negative BEFORE the cast: `r4d_row_pitch` returns -1 for a tensor it cannot pitch, and
     * casting that to uint32 first makes the `< 0` test dead. */
    const long long qp = r4d_row_pitch(q);
    const long long sp = r4d_row_pitch(sc);
    if (qp < 0 || sp < 0 || (qp & 3)) return R4D_NO(RAD_E_STRIDE);
    hipLaunchKernelGGL((mxfp4a8_gated_quant_row_kernel), dim3((uint32_t)M), dim3(kQThr), 0, st,
                       (const uint16_t*)P, (uint8_t*)Gp, (float*)Qs,
                       (uint32_t)nout, (uint32_t)Pld, (uint32_t)qp, sp ? (uint32_t)sp : 1u);
    if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
