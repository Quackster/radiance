// r4d_gemm_mxfp4a8_decode.hip - r4d_gemm_mxfp4a8_decode: the OCP-MXFP4 x fp8 GEMM for the DECODE
// band, M <= 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T over the same operands, weight
// layout and fold as r4d_gemm_mxfp4a8_nt_m64 and r4d_gemm_mxfp4a8_tiled.
//
// WHY A THIRD MXFP4 KERNEL.  At decode the GEMM is a weight stream: a DFlash2 verify step is M = 8
// rows against 13 GB of MXFP4 trunk, and the only number that matters is how close the stream runs
// to the card's ~640 GB/s.  Measured inside a served Qwen3.8-27B (radiance --profile-ops, R9700),
// the skinny kernel reads gate_up at ~500 GB/s, down/inz/qg at ~430-490 and the 16 MB o/out
// projections at ~270 -- the trunk takes ~33 ms a verify step.  This kernel is the one that served
// the same model in SlyBase/vllm-sly-radiance (radiance_mxfp4_fp8_gemm_decode), where the same trunk
// took ~22.7 ms: split-K ACROSS workgroups with a fused last-arriver reduction, the weight unpacked
// through LDS, a 16-row A fragment per M tile, and streaming (non-temporal) weight loads.
//
// THE TILE.  A block is DWN = 8 waves over BND = 128 output columns, one 16-column fragment a wave,
// and DTM 16-row M fragments -- the SMALLEST that covers M (at M=20 DTM=2 beat DTM=3 by 12%: the
// wider tile computes rows nobody asked for).  A K slab of DBK bytes is staged: A (16*DTM rows) and
// the weight's 128 rows, unpacked to e4m3 with the block exponent folded in exactly as the other two
// MXFP4 kernels fold it (libr4d's kMag table, so all three agree to the byte).
//
// SPLIT-K IS NOT OPTIONAL.  With one 16-row M fragment the independent N fragments number N/16, so
// N=5120 is 320 waves -- 2.5 a SIMD however the N tile is chosen.  DKS blocks split the K slabs;
// each writes an fp32 partial [DKS][M][N] and the LAST to arrive (an agent-scope counter per column
// block) sums them in place, so there is no second launch.  The split is the smallest whose grid
// fills the card -- fill = 110 blocks at M <= 24, 78 above -- measured cell by cell, DRAM-fed, on
// the six production shapes (the tables are in the vllm-sly-radiance source this is ported from).
// DBK is 128 except at DTM=4 with no split, where LDS per block (16*DTM + 128)*(DBK+8) would halve
// the resident blocks; a K that is not a multiple of 128 takes 64 (and a 2-way split widens to 4).
// All three are tunables (`ks`, `bk`, `nt`) whose 0 is this rule.
//
// THE ENGINE PATH KEEPS NO STATE ON THE STREAM.  The split partial is the op's own scratch (the core
// sizes it from the scratch hook and treats it as read+write, so two ops that could overlap get
// disjoint slices) and the arrival counters are a small block keyed on the OUTPUT buffer (mxd_out_ctr),
// zero at entry and left zero by the reducer.  Never key them on the stream: a recorded pass is
// replayed with its recorded arguments, and a counter pointer that depended on the stream would be a
// pass that "issued differently from its recording".  Two launches writing the same output are ordered
// by that dependency; two that can overlap write different outputs.  The row is therefore concurrent
// (rad_kernel_concurrent), as gemm_fp8a8_nt_m16 is.
//
// The standalone entry (tools, benches) keeps the older persistent workspace, one a (device, stream),
// allocated ONCE at the band's largest extent (4 splits x 64 rows x 36864 columns, 37.7 MB) and never
// grown; the engine never calls it.
//
// Every thread fences its partial stores before the counter --
// a fence on thread 0 alone orders only thread 0's stores, and the reducer then reads whatever
// happened to drain.
//
// TWO ACTIVATION SCALE GRIDS, as in the tiled kernel: one f32 a row, or one a (row, 128 of K).  The
// per-block form carries each row's accumulator in the current block's scale and rescales it by
// s_prev / s_next at each 128 boundary, the ratios computed once a row into LDS.  A split's partial
// is brought back to absolute units before it is written, because the splits end in different
// blocks.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

#define R4D_MXD_GROUP   32      // K per E8M0 exponent, fixed by the OCP MX format
#define R4D_MXD_DWN     8       // waves per block; the N tile is DWN*16 = 128 wide
#define R4D_MXD_MTILE   16
#define R4D_MXD_MAX_TM  4       // M <= 64
#define R4D_MXD_PAD     8
#define R4D_MXD_MAX_KS  4
#define R4D_MXD_MAX_N   36864   // TP=1 gate_up is 34816

typedef int      mxd_v2i __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u2  __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u4  __attribute__((ext_vector_type(4)));

// libr4d's folded-magnitude table (r4d_gemm_mxfp4a8_nt_m64.hip), copied and NOT regenerated: the
// three MXFP4 kernels must fold every element to the same e4m3 byte.
__device__ __constant__ unsigned int kMxdMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

template <bool NT, typename T>
static __device__ __forceinline__ T mxd_ld(const T* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

template <int DBK, int DKS, int DTM, bool NT, bool ABLK>
__global__ __launch_bounds__(R4D_MXD_DWN * 32) void r4d_gemm_mxfp4a8_decode_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs) {
  constexpr int DWN = R4D_MXD_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  constexpr int STR = DBK + R4D_MXD_PAD;
  __shared__ unsigned char sA[ROWS * STR];
  __shared__ unsigned char sW[BND * STR];
  __shared__ float sR[ABLK ? ROWS : 1];
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);

  // ABLK: the scale row m's accumulator is carried in after block kb -- the last non-zero entry at
  // or before kb, or 1 when there is none.
  auto carried = [&](int m, int kb) -> float {
    const float* __restrict__ row = As + (size_t)m * as_rs;
    for (int j = kb; j >= 0; --j) {
      const float s = row[j];
      if (s > 0.f) return s;
    }
    return 1.f;
  };

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    // ---- a 128 boundary inside this split: one ratio a row into LDS (published by the barrier
    // after staging; the previous readers retired behind the last slab's closing barrier).
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) {
        const int m = tid < M ? tid : M - 1;
        const int kb = k0 >> 7;
        const float sc = As[(size_t)m * as_rs + kb];
        sR[tid] = sc > 0.f ? carried(m, kb - 1) / sc : 1.f;
      }
    }
    // ---- A: ROWS x DBK bytes, clamped rows, 16 B a thread
#pragma unroll
    for (int off = 0; off < ROWS * DBK; off += NTHR * 16) {
      const int idx = off + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;              // clamp, never predicate
        *(mxd_u4*)(&sA[r * STR + c]) = *(const mxd_u4*)(A + (size_t)rc * K + k0 + c);
      }
    }
    // ---- W: fragment-order slots, two adjacent rows a thread (an 8 B request), folded to e4m3
    {
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += NTHR * 2) {
        const int sl = off + tid * 2;
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // gn is even and N % 16 == 0: clamping to the last even row keeps the 8 B read aligned and
        // gc + 1 in bounds; the slot address is recomputed from the clamped row.
        const int gc = gn < N - 2 ? gn : N - 2;
        const int lanec = (lane_ & 16) | (gc & 15);
        const mxd_u2 wv =
            mxd_ld<NT>((const mxd_u2*)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        const int blk = (k0 + kst * 16) / R4D_MXD_GROUP;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMxdMag[d][0], t1 = kMxdMag[d][1];
          const unsigned int w = q ? wv[1] : wv[0];
          const unsigned int ev = w & 0x0F0F0F0Fu;
          const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(mxd_u2*)(&sW[(r + q) * STR + kloc]) =
              mxd_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                     __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
    __syncthreads();
    if constexpr (ABLK) {
      if (bnd) {
#pragma unroll
        for (int i = 0; i < DTM; ++i) {
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][e] *= sR[i * 16 + kb8 + e];
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      mxd_v2i af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char* pa = &sA[(i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)pa; af[i][1] = *(const int*)(pa + 4);
      }
      const unsigned char* pw = &sW[(wave * 16 + col) * STR + kk];
      wf[0] = *(const int*)pw; wf[1] = *(const int*)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
  }

  // ABLK: the scale each row ended this split in, one row a thread through LDS -- the K loop's
  // closing barrier retired the last ratio reads.
  if constexpr (ABLK) {
    if (tid < ROWS) {
      const int m = tid < M ? tid : M - 1;
      const int kb_end = s_hi > s_lo ? ((s_hi * DBK < K ? s_hi * DBK : K) - 1) >> 7 : 0;
      sR[tid] = carried(m, kb_end);
    }
    __syncthreads();
  }
  auto absval = [&](int i, int e) -> float {
    return ABLK ? acc[i][e] * sR[i * 16 + kb8 + e] : acc[i][e];
  };
  const int n = n0 + wave * 16 + col;
  if constexpr (DKS == 1) {
    // The accumulator holds the whole K range: no partial, no counter, no reduction.
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(absval(i, e) * rf * (ABLK ? 1.f : As[m]));
        }
    }
    return;
  }
  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = absval(i, e);
      }
  }
  // EVERY thread's partial stores are ordered before the counter, then the last arriver reduces.
  __threadfence();
  __syncthreads();
  if (tid == 0)
    s_last = __hip_atomic_fetch_add(&cnt[blockIdx.x], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (s_last != (unsigned)(DKS - 1)) return;
  if (tid == 0) cnt[blockIdx.x] = 0u;          // left clean for the next launch on this stream
  __threadfence();                             // acquire, before reading the other partials
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += NTHR) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__hip_bfloat16)(sum * rf * (ABLK ? 1.f : As[m]));
    }
  }
}

// THE ONE-FRAGMENT FORM (DTM = 1, M <= 16), which is every single-sequence verify step. The same tile,
// fold and split as above, with two changes that are worth ~3 % on the large shapes there and nothing
// at DTM >= 2 (measured over the six production shapes, R9700, weights rotated past the
// Infinity Cache): the NEXT slab's A, codes and exponents are loaded into registers while this slab
// computes (PF), so a block keeps its loads in flight across the WMMA phase instead of draining at
// every barrier; and a thread takes four adjacent slots, one 16 B load (VEC = 4) instead of two 8 B
// ones. Each row's reference exponent is read once, not once a slab. Bit-identical to the kernel above
// at every split: the K order, the fold and the reduction are unchanged.
template <int DBK, int DKS, int DTM, bool NT, bool ABLK, bool PF, int VEC>
__global__ __launch_bounds__(R4D_MXD_DWN * 32) void r4d_gemm_mxfp4a8_decode1_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs) {
  constexpr int DWN = R4D_MXD_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  constexpr int STR = DBK + R4D_MXD_PAD;
  constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
  constexpr int W_IT = NTILES_T * KSTEPS_T * 32 / (NTHR * VEC);
  constexpr int A_IT = (ROWS * DBK + NTHR * 16 - 1) / (NTHR * 16);
  __shared__ unsigned char sA[ROWS * STR];
  __shared__ unsigned char sW[BND * STR];
  __shared__ float sR[ABLK ? ROWS : 1];
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);
  auto carried = [&](int m, int kb) -> float {
    const float* __restrict__ row = As + (size_t)m * as_rs;
    for (int j = kb; j >= 0; --j) { const float s = row[j]; if (s > 0.f) return s; }
    return 1.f;
  };
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  // per-thread slot geometry, fixed across slabs
  int s_r[W_IT], s_kst[W_IT], s_kloc[W_IT], s_gc[W_IT], s_lanec[W_IT];
  int s_ref[W_IT][VEC];
#pragma unroll
  for (int it = 0; it < W_IT; ++it) {
    const int sl = it * NTHR * VEC + tid * VEC;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    const int r = ntl * 16 + (lane_ & 15);
    const int gn = n0 + r;
    const int gc = gn < N - VEC ? gn : N - VEC;
    s_r[it] = r; s_kst[it] = kst; s_kloc[it] = kst * 16 + (lane_ >> 4) * 8;
    s_gc[it] = gc; s_lanec[it] = (lane_ & 16) | (gc & 15);
#pragma unroll
    for (int q = 0; q < VEC; ++q) s_ref[it][q] = (int)Wref[gc + q];
  }
  typedef unsigned wv_t __attribute__((ext_vector_type(VEC)));
  wv_t rw[W_IT];
  int rd[W_IT][VEC];
  mxd_u4 ra[A_IT];

  auto load = [&](int s) {
    const int k0 = s * DBK;
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        ra[j] = *(const mxd_u4*)(A + (size_t)rc * K + k0 + c);
      }
    }
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < W_IT; ++it) {
      rw[it] = mxd_ld<NT>((const wv_t*)&Wp[((size_t)(s_gc[it] >> 4) * ksteps_g + kstep0 + s_kst[it]) * 32 + s_lanec[it]]);
      const int blk = (k0 + s_kst[it] * 16) / R4D_MXD_GROUP;
#pragma unroll
      for (int q = 0; q < VEC; ++q) rd[it][q] = (int)Ws[(size_t)blk * N + s_gc[it] + q];
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int j = 0; j < A_IT; ++j) {
      const int idx = j * NTHR * 16 + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        *(mxd_u4*)(&sA[r * STR + c]) = ra[j];
      }
    }
#pragma unroll
    for (int it = 0; it < W_IT; ++it) {
#pragma unroll
      for (int q = 0; q < VEC; ++q) {
        int d = s_ref[it][q] - rd[it][q];
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const unsigned int t0 = kMxdMag[d][0], t1 = kMxdMag[d][1];
        const unsigned int w = rw[it][q];
        const unsigned int ev = w & 0x0F0F0F0Fu;
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
        *(mxd_u2*)(&sW[(s_r[it] + q) * STR + s_kloc[it]]) =
            mxd_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u), __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
    }
  };

  if (s_lo < s_hi) load(s_lo);
  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) {
        const int m = tid < M ? tid : M - 1;
        const int kb = k0 >> 7;
        const float sc = As[(size_t)m * as_rs + kb];
        sR[tid] = sc > 0.f ? carried(m, kb - 1) / sc : 1.f;
      }
    }
    if constexpr (!PF) { if (s > s_lo) load(s); }
    store();
    __syncthreads();
    if constexpr (PF) { if (s + 1 < s_hi) load(s + 1); }
    if constexpr (ABLK) {
      if (bnd) {
#pragma unroll
        for (int i = 0; i < DTM; ++i) {
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][e] *= sR[i * 16 + kb8 + e];
        }
      }
    }
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      mxd_v2i af[DTM], wf;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char* pa = &sA[(i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)pa; af[i][1] = *(const int*)(pa + 4);
      }
      const unsigned char* pw = &sW[(wave * 16 + col) * STR + kk];
      wf[0] = *(const int*)pw; wf[1] = *(const int*)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
    }
    __syncthreads();
  }
  if constexpr (ABLK) {
    if (tid < ROWS) {
      const int m = tid < M ? tid : M - 1;
      const int kb_end = s_hi > s_lo ? ((s_hi * DBK < K ? s_hi * DBK : K) - 1) >> 7 : 0;
      sR[tid] = carried(m, kb_end);
    }
    __syncthreads();
  }
  auto absval = [&](int i, int e) -> float { return ABLK ? acc[i][e] * sR[i * 16 + kb8 + e] : acc[i][e]; };
  const int n = n0 + wave * 16 + col;
  if constexpr (DKS == 1) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(absval(i, e) * rf * (ABLK ? 1.f : As[m]));
        }
    }
    return;
  }
  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = absval(i, e);
      }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = __hip_atomic_fetch_add(&cnt[blockIdx.x], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (s_last != (unsigned)(DKS - 1)) return;
  if (tid == 0) cnt[blockIdx.x] = 0u;
  __threadfence();
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += NTHR) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__hip_bfloat16)(sum * rf * (ABLK ? 1.f : As[m]));
    }
  }
}


// THE PERSISTENT PARTIAL AND COUNTERS, one a (device, stream) -- see the header. Grown on demand,
// never freed; the counters are zeroed when allocated and left zero by every launch.
namespace {
constexpr int kMxdStreams = 4;
struct MxdSlot {
  bool used = false;
  hipStream_t key = nullptr;
  float* ws = nullptr;
  size_t ws_bytes = 0;
  unsigned* cnt = nullptr;
  size_t cnt_n = 0;
};
void mxd_reserve(hipStream_t stream, float** ws, unsigned** cnt) {
  const size_t ws_bytes = (size_t)R4D_MXD_MAX_KS * R4D_MXD_MTILE * R4D_MXD_MAX_TM * R4D_MXD_MAX_N *
                          sizeof(float);
  const size_t cnt_n = R4D_MXD_MAX_N / (R4D_MXD_DWN * 16);
  static std::mutex mu;
  static MxdSlot slot[16][kMxdStreams];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: no current device for the split workspace");
  std::lock_guard<std::mutex> lk(mu);
  MxdSlot* e = nullptr;
  for (MxdSlot& x : slot[dev])
    if (x.used && x.key == stream) { e = &x; break; }
  if (!e) {
    for (MxdSlot& x : slot[dev])
      if (!x.used) { e = &x; break; }
    if (!e) throw std::runtime_error("r4d gemm_mxfp4a8_decode: more streams than split workspaces");
    e->used = true;
    e->key = stream;
  }
  if (ws_bytes > e->ws_bytes) {
    if (e->ws) (void)hipFree(e->ws);
    e->ws = nullptr;
    e->ws_bytes = 0;
    if (hipMalloc((void**)&e->ws, ws_bytes) != hipSuccess)
      throw std::runtime_error("r4d gemm_mxfp4a8_decode: split workspace alloc failed");
    e->ws_bytes = ws_bytes;
  }
  if (cnt_n > e->cnt_n) {
    if (e->cnt) (void)hipFree(e->cnt);
    e->cnt = nullptr;
    e->cnt_n = 0;
    if (hipMalloc((void**)&e->cnt, cnt_n * sizeof(unsigned)) != hipSuccess)
      throw std::runtime_error("r4d gemm_mxfp4a8_decode: split counter alloc failed");
    (void)hipMemset(e->cnt, 0, cnt_n * sizeof(unsigned));
    e->cnt_n = cnt_n;
  }
  *ws = e->ws;
  *cnt = e->cnt;
}

// THE ENGINE'S PATH: the split partial in the op's own SCRATCH, the arrival counters keyed on the
// OUTPUT. A recorded pass is replayed with the arguments it was recorded with, and the scratch is
// part of the op as the core planned it, so it is the same pointer at record and at replay -- where
// a workspace looked up by stream is not, whenever the engine records a pass on one stream and plays
// it on another: the DFlash2 drafter's passes do exactly that ("draft pass ... issued differently
// from its recording ... byte 42", the partial's pointer). The counters cannot live in the scratch
// (other ops write the arena between two of ours, and they must be zero at entry), so they are a
// small device allocation of their own, one block an output buffer, taken once and kept zero by the
// reducer -- gemm_fp8a8_nt_m16's split_ctr, keyed on the output alone: two launches that write the
// same output are ordered by that dependency, and two that can overlap write different outputs.
unsigned* mxd_out_ctr(const void* out) {
  static std::mutex mu;
  static std::map<std::pair<int, const void*>, unsigned*> blocks;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return nullptr;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_pair(dev, out);
  if (auto it = blocks.find(key); it != blocks.end()) return it->second;
  const size_t n = R4D_MXD_MAX_N / (R4D_MXD_DWN * 16);
  unsigned* p = nullptr;
  if (hipMalloc((void**)&p, n * sizeof(unsigned)) != hipSuccess) return nullptr;
  if (hipMemset(p, 0, n * sizeof(unsigned)) != hipSuccess) { (void)hipFree(p); return nullptr; }
  blocks.emplace(key, p);
  return p;
}

// The smallest split whose grid fills the card; see the header for where the numbers come from.
int mxd_split_k(int nblk, int M) {
  const int fill = (M <= 24) ? 110 : 78;
  for (int k = 1; k < R4D_MXD_MAX_KS; k <<= 1)
    if (nblk * k >= fill) return k;
  return R4D_MXD_MAX_KS;
}
}  // namespace

extern "C" int r4d_gemm_mxfp4a8_decode_max_m(void) { return R4D_MXD_MTILE * R4D_MXD_MAX_TM; }

static void mxd_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K, int N,
                       int KS, int BK, int NT, int as_group, int as_rs, long stream, long scratch,
                       long scratch_bytes, bool engine);

// The standalone entry (tools, benches): the persistent per-stream workspace.
extern "C" void r4d_gemm_mxfp4a8_decode(long a, long ascale, long wq, long ws, long wref, long c,
                                        int M, int K, int N, int KS, int BK, int NT,
                                        int as_group, int as_rs, long stream) {
  mxd_launch(a, ascale, wq, ws, wref, c, M, K, N, KS, BK, NT, as_group, as_rs, stream, 0, 0, false);
}

// The split partial's size for a split of `ks` -- what the scratch hook asks the core for.
extern "C" long r4d_gemm_mxfp4a8_decode_scratch(int M, int K, int N, int KS) {
  if (M < 1 || N < 16 || K < 64) return 0;
  constexpr int BND = R4D_MXD_DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  int ks = KS > 0 ? KS : mxd_split_k(nblk, M);
  ks = ks > (KS > 0 ? KS : mxd_split_k(nblk, 1)) ? ks : (KS > 0 ? KS : mxd_split_k(nblk, 1));
  if (ks < 4 && (K % 128)) ks = 4;              /* bk falls back to 64 and a 2-way split widens */
  return ks > 1 ? (long)ks * M * N * (long)sizeof(float) : 0;
}

static void mxd_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K, int N,
                       int KS, int BK, int NT, int as_group, int as_rs, long stream, long scratch,
                       long scratch_bytes, bool engine) {
  if (M < 1 || M > R4D_MXD_MTILE * R4D_MXD_MAX_TM)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: M must be 1..64");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_decode: N must be a multiple of 16");
  if (K < 64 || K % 64) throw std::runtime_error("r4d gemm_mxfp4a8_decode: K must be a multiple of 64");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  constexpr int BND = R4D_MXD_DWN * 16;
  const int nblk = (N + BND - 1) / BND;
  const int tm = (M + R4D_MXD_MTILE - 1) / R4D_MXD_MTILE;
  int ks = KS > 0 ? KS : mxd_split_k(nblk, M);
  if (ks != 1 && ks != 2 && ks != 4) throw std::runtime_error("r4d gemm_mxfp4a8_decode: KS must be 1, 2 or 4");
  int bk = BK > 0 ? BK : ((tm == 4 && ks == 1) ? 64 : 128);
  if (bk != 64 && bk != 128) throw std::runtime_error("r4d gemm_mxfp4a8_decode: BK must be 64 or 128");
  if (K % bk) { bk = 64; if (ks == 2) ks = 4; }   // a slab may not run past K
  const bool nt = NT != 0;
  hipStream_t st = (hipStream_t)stream;
  float* P = nullptr;
  unsigned* cnt = nullptr;
  if (N > R4D_MXD_MAX_N) throw std::runtime_error("r4d gemm_mxfp4a8_decode: N above the workspace's 36864");
  if (engine) {
    /* A scratch too short for the split runs unsplit, which is correct and slower. */
    if (ks > 1 && scratch_bytes < (long)ks * M * N * (long)sizeof(float)) { ks = 1; bk = BK > 0 ? BK : ((tm == 4) ? 64 : 128); if (K % bk) bk = 64; }
    P = (float*)scratch;
    cnt = mxd_out_ctr((const void*)c);
    if (!cnt) throw std::runtime_error("r4d gemm_mxfp4a8_decode: split counter alloc failed");
  } else {
    mxd_reserve(st, &P, &cnt);   // every launch, split or not: one pointer for the life of the stream
  }
  dim3 grid(nblk, 1, ks), block(R4D_MXD_DWN * 32);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
#define MXD_L(BK_, KS_, TM_, NT_, AB_)                                                         \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_decode_kernel<BK_, KS_, TM_, NT_, AB_>), grid, block, 0, \
                     st, Ap, Sp, Wp, Zp, Rp, P, cnt, Cp, M, K, N, as_rs)
#define MXD_AB(BK_, KS_, TM_, NT_) { if (ablk) MXD_L(BK_, KS_, TM_, NT_, true); else MXD_L(BK_, KS_, TM_, NT_, false); }
#define MXD_NT(BK_, KS_, TM_) { if (nt) MXD_AB(BK_, KS_, TM_, true) else MXD_AB(BK_, KS_, TM_, false) }
#define MXD_L1(BK_, KS_, NT_, AB_)                                                              \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_decode1_kernel<BK_, KS_, 1, NT_, AB_, true, 4>), grid, block, \
                     0, st, Ap, Sp, Wp, Zp, Rp, P, cnt, Cp, M, K, N, as_rs)
#define MXD_AB1(BK_, KS_, NT_) { if (ablk) MXD_L1(BK_, KS_, NT_, true); else MXD_L1(BK_, KS_, NT_, false); }
#define MXD_NT1(BK_, KS_) { if (nt) MXD_AB1(BK_, KS_, true) else MXD_AB1(BK_, KS_, false) }
#define MXD_TM(BK_, KS_) { switch (tm) { case 1: MXD_NT1(BK_, KS_) break; case 2: MXD_NT(BK_, KS_, 2) break; \
                                         case 3: MXD_NT(BK_, KS_, 3) break; default: MXD_NT(BK_, KS_, 4) break; } }
  if (bk == 64) { if (ks == 1) MXD_TM(64, 1) else if (ks == 2) MXD_TM(64, 2) else MXD_TM(64, 4) }
  else          { if (ks == 1) MXD_TM(128, 1) else if (ks == 2) MXD_TM(128, 2) else MXD_TM(128, 4) }
#undef MXD_TM
#undef MXD_NT1
#undef MXD_AB1
#undef MXD_L1
#undef MXD_NT
#undef MXD_AB
#undef MXD_L
}

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_mxfp4a8_decode(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.K % 64) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp4a8_decode_max_m() || g.N > R4D_MXD_MAX_N) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale || !g.bref) return R4D_NO(RAD_E_INVAL);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs);
    if (rc != RAD_OK) return rc;
    R4dMxDecCfg t;
    r4d_cfg_mxfp4_decode(a, &t);
    R4D_TRY("gemm_mxfp4a8_decode",
        mxd_launch(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                   r4d_addr(g.bscale), r4d_addr(g.bref), r4d_addr(g.y),
                   (int)g.M, (int)g.K, (int)g.N, t.ks, t.bk, t.nt, as_group, as_rs,
                   (long)(uintptr_t)s, (long)(uintptr_t)a->scratch, (long)a->scratch_bytes, true));
    return RAD_OK;
}

/* The core sizes the op's scratch from this, at the band's widest M and over the split rule at
 * every M of the band (a narrower M can split wider). */
extern "C" int64_t r4d_rad_gemm_mxfp4a8_decode_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (M < 1 || N < 1 || K < 1 || M > r4d_gemm_mxfp4a8_decode_max_m()) return 0;
    R4dMxDecCfg t;
    r4d_cfg_mxfp4_decode(a, &t);
    return (int64_t)r4d_gemm_mxfp4a8_decode_scratch((int)M, (int)K, (int)N, t.ks);
}
