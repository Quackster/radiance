// r4d_gemm_mxfp4a8_decode.hip - r4d_gemm_mxfp4a8_decode: the OCP-MXFP4 x fp8 GEMM for the DECODE
// band, M <= 128, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T over the same operands, weight
// layout and fold as r4d_gemm_mxfp4a8_nt_m64 and r4d_gemm_mxfp4a8_tiled.
//
// WHY A THIRD MXFP4 KERNEL.  At decode the GEMM is a weight stream: a speculative verify step is
// a handful of rows against the whole 4-bit trunk, and the only number that matters is how close
// the stream runs to the card's DRAM bandwidth.  The skinny kernel splits K inside one workgroup
// and reads the weight straight into registers, which leaves the narrow projections well short of
// it.  This one splits K ACROSS workgroups with a fused last-arriver reduction, unpacks the weight
// through LDS, gives each block one 16-row A fragment per M tile, and streams the weight with
// non-temporal loads.
//
// THE TILE.  A block is DWN = 8 waves over BND = 128 output columns, one 16-column fragment a wave,
// and DTM 16-row M fragments -- the SMALLEST that covers M, because a wider tile computes rows
// nobody asked for.  A K slab of DBK bytes is staged: A (16*DTM rows) and the weight's 128 rows,
// unpacked to e4m3 with the block exponent folded in through r4d_mxfp4.h, as the other two MXFP4
// kernels fold it.
//
// SPLIT-K IS NOT OPTIONAL.  With one 16-row M fragment the independent N fragments number N/16, so
// N=5120 is 320 waves -- 2.5 a SIMD however the N tile is chosen.  DKS blocks split the K slabs;
// each writes an fp32 partial [DKS][M][N] and the LAST to arrive (an agent-scope counter per column
// block) sums them in place, so there is no second launch.
//
// THE SPLIT, SLAB AND LOAD HINT are a rule measured cell by cell on this kernel -- every split x
// slab x hint against the 27B's per-rank shapes at M = 8..128, DRAM-fed -- because no single
// fill target fits: the split is the smallest power of two whose grid reaches 48 blocks at M <= 16,
// 56 at M <= 96 and 40 above (an unsplit 40-block grid already streams a short-K weight at the
// card's rate there, and a split's partial traffic grows with M); a deep K (8192 or more, down at
// K=8704) keeps doubling while a split still has 24 or more 128-wide slabs, since 40-odd blocks
// do not keep its long slab loop fed; and no split goes past ks*M = 512, where the partials
// outweigh what the split buys.  The slab is 128 except where LDS per block, (16*DTM + 128) *
// (DBK+8) bytes, would cost a wide or deeply split grid its resident blocks (see mxd_plan); a K
// that is not a multiple of 128 takes 64 (and a 2-way split widens to 4).  Weight loads are non-temporal for weights of 4 MiB or more,
// plain below, where the weight is too small to evict anything the split partials need.  All three
// are tunables (`ks`, `bk`, `nt`) whose 0 is this rule.
//
// NO STATE IS KEYED ON THE STREAM.  The split partial is the op's own scratch -- the core sizes it
// from the scratch hook and treats it as read+write, so two ops that could overlap get disjoint
// slices -- and the arrival counters are a small block keyed on the OUTPUT buffer (mxd_out_ctr),
// zero at entry and left zero by the reducer.  A recorded pass is replayed with its recorded
// arguments, possibly on another stream than it was recorded on, so a pointer that depended on the
// stream would make the replay differ from its recording.  Two launches writing the same output are
// ordered by that dependency; two that can overlap write different outputs.  The row is therefore
// concurrent (rad_kernel_concurrent), as gemm_fp8a8_nt_m16 is.  A scratch shorter than the split
// needs is refused: the hook sizes it for every M of the band, so a short one is a caller's error.
//
// Every thread fences its partial stores before the counter: a fence on thread 0 alone orders
// only thread 0's stores, and the reducer would then read whatever happened to drain.
//
// TWO ACTIVATION SCALE GRIDS, as in the tiled kernel: one f32 a row, or one a (row, 128 of K).  The
// per-block form carries each row's accumulator in the current block's scale and rescales it by
// s_prev / s_next at each 128 boundary, the ratio computed once a row into LDS from scales loaded a
// boundary early; with the accumulator transposed a lane reads one ratio for each of its rows.  A split's partial is brought back
// to absolute units before it is written, because the splits end in different blocks.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>

#define R4D_MXD_DWN     8       // waves per block; the N tile is DWN*16 = 128 wide
#define R4D_MXD_MTILE   16
#define R4D_MXD_MAX_TM  8       // M <= 128
#define R4D_MXD_PAD     8
#define R4D_MXD_MAX_KS  16
#define R4D_MXD_MAX_N   36864   // the counter block's width; a 27B gate_up at TP=1 is 34816

typedef int      mxd_v2i __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u2  __attribute__((ext_vector_type(2)));
typedef unsigned mxd_u4  __attribute__((ext_vector_type(4)));
typedef float    mxd_f4  __attribute__((ext_vector_type(4)));

template <bool NT, typename T>
static __device__ __forceinline__ T mxd_ld(const T* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// ---- what both kernels share: the scale grid at a 128 boundary, and the epilogue.
//
// THE ACCUMULATOR IS C TRANSPOSED.  The WMMA takes the weight fragment as its A operand and the
// activation fragment as its B operand -- the two fragments have the same register layout, so
// this is only an argument order -- and computes C^T: a lane's accumulator for M fragment i holds
// ONE row, m = 16i + col, and the eight consecutive columns n0 + 16*wave + kb8 + e.  So a row's
// activation scale is one value a lane, not eight: the (row, 128) grid's boundary rescale reads
// DTM scales instead of 8*DTM, and the epilogue writes each lane's eight outputs as one 16-byte
// store (one 32-byte store of partials) instead of eight 2-byte ones.

// THE (ROW, 128) SCALE GRID, computed once a row: thread t < ROWS owns row t.  It loads the next
// 128 boundary's two scales -- the block that starts there and the one before it -- a boundary
// early, writes the row's ratio s_prev / s_next to LDS at the boundary (published by the staging
// barrier), and a lane then reads ONE ratio for each of its rows.  A zero block keeps the carried
// scale; a zero block before this one sends the carried scale back to the grid, which only an
// all-zero activation block ever does.
static __device__ __forceinline__ float mxd_ratio(float ns, float np, const float* __restrict__ row,
                                                  int kb) {
  return ns > 0.f ? (np > 0.f ? np : r4d_mxfp4_carried(row, kb - 2)) / ns : 1.f;
}

template <int DTM>
static __device__ __forceinline__ void mxd_rescale(v8f (&acc)[DTM], const float* sR, int col) {
#pragma unroll
  for (int i = 0; i < DTM; ++i) {
    const float r = sR[i * 16 + col];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] *= r;
  }
}

// The split's result: written straight to C when the accumulator holds the whole K range, else as
// an fp32 partial that the last block to arrive on this column block sums.  `kb_end` is the block
// the split's last slab ends in (ABLK: the scale the accumulators are carried in).
template <int DTM, int DKS, bool ABLK>
static __device__ __forceinline__ void mxd_epilogue(
    const v8f (&acc)[DTM], const float* __restrict__ As, int as_rs,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int N, int n0, int wave, int col, int kb8, int ks,
    int kb_end, unsigned* s_last, float* sR) {
  constexpr int NTHR = R4D_MXD_DWN * 32, BND = R4D_MXD_DWN * 16;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  const int tid = threadIdx.x;
  const int nb = n0 + wave * 16 + kb8;   // this lane's eight columns; N % 16 == 0 keeps all or none
  // Each row's final scale, once a row: the scale the split's accumulators end in (ABLK), or the
  // row's own.  The K loop's closing barrier retired the last ratio reads.
  if (tid < ROWS) {
    const int m = tid < M ? tid : M - 1;
    sR[tid] = ABLK ? r4d_mxfp4_carried(As + (size_t)m * as_rs, kb_end) : As[(size_t)m * as_rs];
  }
  __syncthreads();
  float sc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) sc[i] = sR[i * 16 + col];
  if constexpr (DKS == 1) {
    if (nb < N) {
      const mxd_u2 rb = *(const mxd_u2*)(Wref + nb);
      float rf[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) rf[e] = __int_as_float((int)((rb[e >> 2] >> (8 * (e & 3))) & 0xFF) << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const int m = i * 16 + col;
        if (m < M) {
          union { __hip_bfloat16 h[8]; mxd_u4 v; } o;
#pragma unroll
          for (int e = 0; e < 8; ++e)
            o.h[e] = (__hip_bfloat16)(ABLK ? acc[i][e] * sc[i] * rf[e] : acc[i][e] * rf[e] * sc[i]);
          *(mxd_u4*)(C + (size_t)m * N + nb) = o.v;
        }
      }
    }
    return;
  }
  if (nb < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i) {
      const int m = i * 16 + col;
      if (m < M) {
        mxd_f4 lo, hi;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          lo[e] = ABLK ? acc[i][e] * sc[i] : acc[i][e];
          hi[e] = ABLK ? acc[i][e + 4] * sc[i] : acc[i][e + 4];
        }
        mxd_f4* dst = (mxd_f4*)(P + ((size_t)ks * M + m) * N + nb);
        dst[0] = lo;
        dst[1] = hi;
      }
    }
  }
  // EVERY thread's partial stores are ordered before the counter, then the last arriver reduces.
  __threadfence();
  __syncthreads();
  if (tid == 0)
    *s_last = __hip_atomic_fetch_add(&cnt[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (*s_last != (unsigned)(DKS - 1)) return;
  if (tid == 0) cnt[blockIdx.x] = 0u;          // left clean for the next launch on this stream
  __threadfence();                             // acquire, before reading the other partials
  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += NTHR) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
#pragma unroll
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      C[(size_t)m * N + nn] = (__hip_bfloat16)(sum * rf * (ABLK ? 1.f : As[(size_t)m * as_rs]));
    }
  }
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
  __shared__ float sR[ROWS];     // ABLK: a boundary's ratio a row; then every row's final scale
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const int k_hi = (s_hi * DBK < K ? s_hi * DBK : K);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);
  const float* __restrict__ asrow = As + (size_t)(tid < M ? tid : M - 1) * as_rs;
  float ns = 0.f, np = 0.f;
  if constexpr (ABLK) {
    // a split that starts mid-block meets its first boundary one slab in
    const int kb = ((s_lo * DBK) >> 7) + 1;
    if (tid < ROWS && ((s_lo * DBK) & 127) && kb * 128 < k_hi) { ns = asrow[kb]; np = asrow[kb - 1]; }
  }

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) sR[tid] = mxd_ratio(ns, np, asrow, k0 >> 7);
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
            mxd_ld<NT>((const mxd_u2*)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 +
                                          lanec]);
        const int blk = (k0 + kst * 16) / R4D_MXFP4_GROUP;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int d = r4d_mxfp4_fold_d(Wref[gc + q], Ws[(size_t)blk * N + gc + q]);
          unsigned int lo, hi;
          r4d_mxfp4_unpack8(q ? wv[1] : wv[0], d, &lo, &hi);
          *(mxd_u2*)(&sW[(r + q) * STR + kloc]) = mxd_u2{lo, hi};
        }
      }
    }
    __syncthreads();
    if constexpr (ABLK) {
      if (bnd) mxd_rescale(acc, sR, col);
      if (tid < ROWS && (k0 & 127) == 0 && k0 + 128 < k_hi) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
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
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf, af[i], acc[i]);
    }
    __syncthreads();
  }
  const int kb_end = s_hi > s_lo ? (k_hi - 1) >> 7 : 0;
  mxd_epilogue<DTM, DKS, ABLK>(acc, As, as_rs, Wref, P, cnt, C, M, N, n0, wave, col, kb8, ks,
                               kb_end, &s_last, sR);
}

// THE ONE-FRAGMENT FORM (M <= 16), which is every single-sequence verify step. The same tile, fold
// and split as above, with two changes that pay only when one M fragment leaves the block waiting
// on memory: the NEXT slab's A, codes and exponents are loaded into registers while this slab
// computes, so a block keeps its loads in flight across the WMMA phase instead of draining at every
// barrier; and a thread takes four adjacent slots, one 16 B load instead of two 8 B ones. Each
// row's reference exponent is read once, not once a slab. Bit-identical to the kernel above at
// every split: the K order, the fold and the reduction are unchanged.
template <int DBK, int DKS, bool NT, bool ABLK>
__global__ __launch_bounds__(R4D_MXD_DWN * 32) void r4d_gemm_mxfp4a8_decode1_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, float* __restrict__ P, unsigned* __restrict__ cnt,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int as_rs) {
  constexpr int DTM = 1, VEC = 4;
  constexpr int DWN = R4D_MXD_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXD_MTILE * DTM;
  constexpr int STR = DBK + R4D_MXD_PAD;
  constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
  constexpr int W_IT = NTILES_T * KSTEPS_T * 32 / (NTHR * VEC);
  constexpr int A_IT = (ROWS * DBK + NTHR * 16 - 1) / (NTHR * 16);
  __shared__ unsigned char sA[ROWS * STR];
  __shared__ unsigned char sW[BND * STR];
  __shared__ float sR[ROWS];     // ABLK: a boundary's ratio a row; then every row's final scale
  __shared__ unsigned s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);
  const int k_hi = (s_hi * DBK < K ? s_hi * DBK : K);

  v8f acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) acc[i] = (v8f)(0.f);
  const float* __restrict__ asrow = As + (size_t)(tid < M ? tid : M - 1) * as_rs;
  float ns = 0.f, np = 0.f;
  if constexpr (ABLK) {
    const int kb = ((s_lo * DBK) >> 7) + 1;
    if (tid < ROWS && ((s_lo * DBK) & 127) && kb * 128 < k_hi) { ns = asrow[kb]; np = asrow[kb - 1]; }
  }
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
      const size_t slot =
          ((size_t)(s_gc[it] >> 4) * ksteps_g + kstep0 + s_kst[it]) * 32 + s_lanec[it];
      rw[it] = mxd_ld<NT>((const wv_t*)&Wp[slot]);
      const int blk = (k0 + s_kst[it] * 16) / R4D_MXFP4_GROUP;
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
        unsigned int lo, hi;
        r4d_mxfp4_unpack8(rw[it][q], r4d_mxfp4_fold_d(s_ref[it][q], rd[it][q]), &lo, &hi);
        *(mxd_u2*)(&sW[(s_r[it] + q) * STR + s_kloc[it]]) = mxd_u2{lo, hi};
      }
    }
  };

  if (s_lo < s_hi) load(s_lo);
  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > s_lo && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) sR[tid] = mxd_ratio(ns, np, asrow, k0 >> 7);
    }
    store();
    __syncthreads();
    if (s + 1 < s_hi) load(s + 1);           // the next slab's loads fly across this slab's WMMAs
    if constexpr (ABLK) {
      if (bnd) mxd_rescale(acc, sR, col);
      if (tid < ROWS && (k0 & 127) == 0 && k0 + 128 < k_hi) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
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
        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf, af[i], acc[i]);
    }
    __syncthreads();
  }
  const int kb_end = s_hi > s_lo ? (k_hi - 1) >> 7 : 0;
  mxd_epilogue<DTM, DKS, ABLK>(acc, As, as_rs, Wref, P, cnt, C, M, N, n0, wave, col, kb8, ks,
                               kb_end, &s_last, sR);
}

namespace {
// THE ARRIVAL COUNTERS, one block an output buffer -- see the header. The counters cannot live in
// the scratch: other ops write the arena between two of ours, and the counters must be zero at
// entry. So they are a small device allocation of their own, taken once per (device, output) and
// kept zero by the reducer, as gemm_fp8a8_nt_m16 keys its split_ctr.
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

// THE SPLIT, SLAB AND LOAD HINT A LAUNCH RUNS, from M, the shape and the tunables (0 = the rule).
// One function for the launch and the scratch hook, so the partial the hook sizes is the one
// written.  See the header for the rule and what it was measured on.
struct MxdPlan { int ks, bk; bool nt; };
MxdPlan mxd_plan(int M, int K, int N, int KS, int BK, int NT) {
  const int nblk = (N + R4D_MXD_DWN * 16 - 1) / (R4D_MXD_DWN * 16);
  const int tm = (M + R4D_MXD_MTILE - 1) / R4D_MXD_MTILE;
  MxdPlan p;
  int ks = 1;
  const int fill = M <= 16 ? 48 : (M <= 96 ? 56 : 40);
  while (ks < R4D_MXD_MAX_KS && nblk * ks < fill && 2 * ks * M <= 512) ks <<= 1;
  if (K >= 8192)                       // a deep K: a split of 24 or more slabs still waits on them
    while (ks < R4D_MXD_MAX_KS && (K / 128) / ks >= 24 && 2 * ks * M <= 512) ks <<= 1;
  p.ks = KS > 0 ? KS : ks;
  // 64-wide slabs where LDS per block, (16*DTM + 128)*(DBK+8) bytes, costs a wide or deeply split
  // grid its resident blocks: DTM 4 unsplit on 110+ column blocks, and DTM 8 on those or at a 4-way
  // split.
  const bool slab64 = (tm == 4 && p.ks == 1 && nblk >= 110) || (tm == 8 && (nblk >= 110 || p.ks >= 4));
  p.bk = BK > 0 ? BK : (slab64 ? 64 : 128);
  p.nt = NT == 1 || (NT == 0 && (long)N * K / 2 >= (4l << 20));
  if (K % p.bk) {                      // a slab may not run past K
    p.bk = 64;
    if (p.ks == 2) p.ks = 4;
  }
  return p;
}

long mxd_partial_bytes(int M, int N, int ks) {
  return ks > 1 ? (long)ks * M * N * (long)sizeof(float) : 0;
}

void mxd_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K, int N,
                int KS, int BK, int NT, int as_group, int as_rs, long stream, long scratch,
                long scratch_bytes) {
  if (M < 1 || M > R4D_MXD_MTILE * R4D_MXD_MAX_TM)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: M must be 1..128");
  if (N < 16 || N % 16 || N > R4D_MXD_MAX_N)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: N must be a multiple of 16 up to 36864");
  if (K < 64 || K % 64)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: K must be a multiple of 64");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: the activation scale is per row or per "
                             "128 of K");
  const MxdPlan pl = mxd_plan(M, K, N, KS, BK, NT);
  const int ks = pl.ks, bk = pl.bk;
  if (ks != 1 && ks != 2 && ks != 4 && ks != 8 && ks != 16)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: KS must be 1, 2, 4, 8 or 16");
  if (bk != 64 && bk != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: BK must be 64 or 128");
  if (scratch_bytes < mxd_partial_bytes(M, N, ks) || (ks > 1 && !scratch))
    throw std::runtime_error("r4d gemm_mxfp4a8_decode: the op's scratch is shorter than the split "
                             "partial; the scratch hook sizes it");
  float* P = (float*)scratch;
  unsigned* cnt = mxd_out_ctr((const void*)c);
  if (!cnt) throw std::runtime_error("r4d gemm_mxfp4a8_decode: split counter alloc failed");
  const bool ablk = as_group == 128, nt = pl.nt;
  const int nblk = (N + R4D_MXD_DWN * 16 - 1) / (R4D_MXD_DWN * 16);
  const int tm = (M + R4D_MXD_MTILE - 1) / R4D_MXD_MTILE;
  hipStream_t st = (hipStream_t)stream;
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
#define MXD_L1(BK_, KS_, NT_, AB_)                                                              \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_decode1_kernel<BK_, KS_, NT_, AB_>), grid, block, 0, st, \
                     Ap, Sp, Wp, Zp, Rp, P, cnt, Cp, M, K, N, as_rs)
#define MXD_AB(BK_, KS_, TM_, NT_) \
  { if (ablk) MXD_L(BK_, KS_, TM_, NT_, true); else MXD_L(BK_, KS_, TM_, NT_, false); }
#define MXD_NT(BK_, KS_, TM_) \
  { if (nt) MXD_AB(BK_, KS_, TM_, true) else MXD_AB(BK_, KS_, TM_, false) }
#define MXD_AB1(BK_, KS_, NT_) \
  { if (ablk) MXD_L1(BK_, KS_, NT_, true); else MXD_L1(BK_, KS_, NT_, false); }
#define MXD_NT1(BK_, KS_) { if (nt) MXD_AB1(BK_, KS_, true) else MXD_AB1(BK_, KS_, false) }
#define MXD_TM(BK_, KS_)                                                                       \
  { switch (tm) { case 1: MXD_NT1(BK_, KS_) break; case 2: MXD_NT(BK_, KS_, 2) break;          \
                  case 3: MXD_NT(BK_, KS_, 3) break; case 4: MXD_NT(BK_, KS_, 4) break;        \
                  case 5: MXD_NT(BK_, KS_, 5) break; case 6: MXD_NT(BK_, KS_, 6) break;        \
                  case 7: MXD_NT(BK_, KS_, 7) break; default: MXD_NT(BK_, KS_, 8) break; } }
#define MXD_KS(BK_)                                                                            \
  { switch (ks) { case 1: MXD_TM(BK_, 1) break; case 2: MXD_TM(BK_, 2) break;                  \
                  case 4: MXD_TM(BK_, 4) break; case 8: MXD_TM(BK_, 8) break;                  \
                  default: MXD_TM(BK_, 16) break; } }
  if (bk == 64) MXD_KS(64) else MXD_KS(128)
#undef MXD_KS
#undef MXD_TM
#undef MXD_NT1
#undef MXD_AB1
#undef MXD_NT
#undef MXD_AB
#undef MXD_L1
#undef MXD_L
}
}  // namespace

extern "C" int r4d_gemm_mxfp4a8_decode_max_m(void) { return R4D_MXD_MTILE * R4D_MXD_MAX_TM; }

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
                   (long)(uintptr_t)s, (long)(uintptr_t)a->scratch, (long)a->scratch_bytes));
    return RAD_OK;
}

/* The core sizes the op's scratch from this. `M` is the band's widest, and a narrower M can split
 * wider (the fill rule asks for more blocks at M <= 24), so the size is the widest split over the
 * band times its widest M -- an upper bound on every launch the band can make. */
extern "C" int64_t r4d_rad_gemm_mxfp4a8_decode_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0);
    const long long K = r4d_geti_or(a, "K", 0);
    if (M < 1 || N < 16 || K < 64 || M > r4d_gemm_mxfp4a8_decode_max_m()) return 0;
    R4dMxDecCfg t;
    r4d_cfg_mxfp4_decode(a, &t);
    int ks = 1;
    for (int m = 1; m <= (int)M; ++m) {
        const int k = mxd_plan(m, (int)K, (int)N, t.ks, t.bk, t.nt).ks;
        if (k > ks) ks = k;
    }
    return (int64_t)mxd_partial_bytes((int)M, (int)N, ks);
}
