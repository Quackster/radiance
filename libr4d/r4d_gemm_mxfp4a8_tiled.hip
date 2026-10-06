// r4d_gemm_mxfp4a8_tiled.hip - r4d_gemm_mxfp4a8_tiled: the OCP-MXFP4 x fp8 GEMM for PREFILL row
// counts, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the same operands, the same weight
// layout and the same fold as r4d_gemm_mxfp4a8_nt_m64 and r4d_gemm_mxfp4a8_decode -- the band
// above theirs.
//
// WHY A SEPARATE KERNEL.  The decode kernels give a block at most 128 rows, so at prefill they
// would re-read the whole weight ceil(M/128) times.  Here a block is a 256 x (32*TN) tile and each
// weight byte is unpacked once per K slab into LDS, where all four row-waves of the block read it.
//
// A COMES STRAIGHT FROM GLOBAL INTO REGISTERS, NOT THROUGH LDS.  Staging A through LDS costs a
// write, a barrier and a second read of the operand the block uses most, and a WMMA A fragment --
// lane l holds row l%16, k-bytes 8*(l/16)..+8 of a 16 x 16 tile -- is 16 rows apart in a row-major
// plane, so it cannot be loaded straight from one either.  So A is re-tiled first, into the op's
// scratch, in FRAGMENT ORDER:
//
//     AT[((m/16) * (K/16) + k/16) * 256 + ((m%16) + 16 * ((k/8)%2)) * 8 + k%8] = A[m][k]
//
// One 16 x 16 fragment is 256 contiguous bytes and a lane's eight are one aligned 8-byte load, so
// every A fragment a wave needs for a slab issues as one clause of global loads off a wave-uniform
// base (SGPR, the SADDR form) and is consumed at the WMMAs.  The re-tile is one read and one write
// of A -- tens of microseconds at a 2048-row chunk against the hundreds the A-direct loop saves on
// every large projection -- and it is the op's own, so no producer has to know about the layout.
// Rows past M in the last fragment are written as zeros; they only reach accumulators the
// epilogue drops.
//
// THE WEIGHT AND THE FOLD ARE THE SKINNY KERNEL'S.  It reads the fragment-order weight that
// r4d_rad_layout_mxfp4 stores and folds each block's exponent into the e4m3 value through the
// table in r4d_mxfp4.h, so all three MXFP4 kernels produce the same e4m3 byte for every element.
// The table is copied to LDS once per block: read from global it would be a dependent load issued
// behind the A fragments, and the in-order load counter would make the fold wait for all of them.
// 2^(Wref[n]-127) is applied once in the epilogue next to the activation's scale; see
// r4d_gemm_mxfp4a8_nt_m64.hip for the fold and its error analysis.
//
// THE TILE.  8 waves as 4 (M) x 2 (N); a wave owns TM=4 x TN accumulator fragments, so the block
// covers BMF = 4*4*16 = 256 rows and BNF = 2*TN*16 columns.  The tall tile is for reuse: every
// weight byte staged is multiplied against 256 rows.  TN is the axis that depends on M.  Widening
// N halves the column-block count while BMF still masks most of a short M, so TN=4 wins only on
// long prompts; on a 17408x5120 weight the two cross near M=2048.  Hence tn=0, the default, picks
// TN=4 from M >= 2048 and TN=2 below.  The K slab is 128 at TN=2 and 64 at TN=4 (and 64 at TN=2
// when K is not a multiple of 128): TN=4 holds twice the accumulators, and a 128 slab's A
// fragments on top of them would cost occupancy.
//
// BARRIERS FENCE LDS ONLY.  __syncthreads() is a workgroup acquire/release over every address
// space, which drains the global load counter at each barrier.  Nothing in global memory is
// shared between the block's waves, so the barrier is scoped to LDS and the next slab's A loads
// stay in flight across it.
//
// STAGING IS CLAMPED, NEVER PREDICATED.  A bounds-predicated load lands in its own exec-mask
// region and the compiler then cannot carry a counted s_wait_loadcnt across the merge.  Clamped
// (a tile past M reads the last real tile, a weight row past N the last aligned group), a slab's
// loads issue back to back; the epilogue drops what they computed.
//
// THE ACCUMULATOR IS C TRANSPOSED.  The WMMA takes the weight fragment as its A operand and the
// activation fragment as its B operand -- the two have the same register layout, so this is only
// an argument order -- and computes C^T: a lane's accumulator for (M fragment i, N fragment j)
// holds ONE row and eight consecutive columns.  So a row's activation scale is one value a lane
// for each of its TM rows, and the epilogue writes each (i, j) as one 16-byte store.
//
// TWO ACTIVATION SCALE GRIDS.  `a_scale` is either one f32 per row (ABLK=false) or one per (row,
// 128 of K) (ABLK=true: the block-scaled fp8 family's grid, which every fused producer in
// arch/common writes).  Per-block scales vary along K and so cannot wait for the epilogue.  The
// accumulator is carried IN UNITS OF THE CURRENT BLOCK'S SCALE: at each 128 boundary every row's
// partial sum is multiplied by s_prev / s_next, and the epilogue applies the last block's scale --
// one multiply per accumulator element per 128 of K, ~K/128 fp32 ulps of extra rounding.  The
// ratio is computed once a row: thread t owns row t of the block, loads the next boundary's two
// scales a boundary early (no slab waits on a scale load) and writes the ratio to LDS, and a lane
// reads one ratio for each of its TM rows.
//
// TRAP FOR ANYONE MEASURING THIS KERNEL: the same as the skinny kernel's.  A weight of tens of
// megabytes streamed in a loop is Infinity-Cache resident on this part, so a bandwidth figure
// taken that way reads above the DRAM roofline.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_MXT_TM    4         // 16-row fragments per wave
#define R4D_MXT_WM    4         // waves along M
#define R4D_MXT_WN    2         // waves along N
#define R4D_MXT_PAD   8         // LDS row pad: 16 lanes reading rows 64 B apart collide 8 ways
#define R4D_MXT_NTHR  (R4D_MXT_WM * R4D_MXT_WN * 32)
#define R4D_MXT_BMF   (R4D_MXT_WM * R4D_MXT_TM * 16)
#define R4D_MXT_KUNIT 64        // the narrowest K slab, so K must be a multiple of it
#define R4D_MXT_TN4_MIN_M 2048  // tn=0's crossover, see above

typedef int      mxt_v2i  __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u2   __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u4   __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ void mxt_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The re-tile: one thread per 8-byte slot of the fragment-ordered plane, so the writes are
// contiguous and each lane reads one 8-byte group of one row.
__global__ __launch_bounds__(256) void r4d_mxfp4a8_tile_a_kernel(
    const unsigned char* __restrict__ A, unsigned char* __restrict__ AT, int M, int K,
    long long slots) {
  const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
  if (o >= slots) return;
  const int ks = K / 16;
  const int lane = (int)(o & 31);
  const long long f = o >> 5;
  const int mt = (int)(f / ks), kst = (int)(f % ks);
  const int r = mt * 16 + (lane & 15), k = kst * 16 + (lane >> 4) * 8;
  mxt_u2 v = { 0u, 0u };
  if (r < M) v = *(const mxt_u2*)(A + (size_t)r * K + k);
  *(mxt_u2*)(AT + o * 8) = v;
}

template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(R4D_MXT_NTHR) void r4d_gemm_mxfp4a8_tiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs) {
  constexpr int TM = R4D_MXT_TM, WN = R4D_MXT_WN, NTHR = R4D_MXT_NTHR, BMF = R4D_MXT_BMF;
  constexpr int NS = LBK / 16, STR = LBK + R4D_MXT_PAD, BNF = WN * TN * 16;
  constexpr int NTILES_T = BNF / 16, TOT_SLOTS = NTILES_T * NS * 32;
  // 4 weight slots (one 16 B read) wherever the tile has enough of them, else 2 (8 B), else 1.
  constexpr int WSLOTS = (TOT_SLOTS >= NTHR * 4) ? 4 : (TOT_SLOTS >= NTHR * 2 ? 2 : 1);
  constexpr int NW = TOT_SLOTS / (NTHR * WSLOTS);
  static_assert(NW * NTHR * WSLOTS == TOT_SLOTS, "the W tile must split evenly over the block");
  __shared__ unsigned char sW[BNF * STR];
  __shared__ unsigned int sMag[32];
  __shared__ float sR[ABLK ? BMF : 1];   // ABLK: a boundary's ratio a row, then each row's final scale

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps = K / 16, Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = (&kMxfp4Mag[0][0])[tid];

  // One A base per M fragment: wave-uniform, so readfirstlane keeps it in an SGPR and every load
  // takes the SADDR form with a 32-bit lane offset.  A tile past M reads the last real one.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned int aoff = (unsigned int)lane * 8;

  // The W slots this thread stages, fixed across slabs: WSLOTS adjacent rows of one (n-tile,
  // k-step), so the wave reads contiguous bytes; the clamp keeps the vector read aligned.
  int wrow[NW], wkloc[NW], wrc[NW], wkst[NW], wlanec[NW];
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHR * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % NS, ntl = rest / NS;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wkloc[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
    wref[it] = WSLOTS == 4 ? *(const unsigned int*)(Wref + wrc[it])
             : WSLOTS == 2 ? *(const unsigned short*)(Wref + wrc[it])
                           : Wref[wrc[it]];
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  // ABLK, once a row: thread t owns row m0 + t (BMF == NTHR) and loads the next 128 boundary's two
  // scales -- the block that starts there and the one before it -- a boundary early.
  const int trow = m0 + tid < M ? m0 + tid : M - 1;
  const float* __restrict__ asrow = As + (size_t)trow * as_rs;
  float ns = 0.f, np = 0.f;
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  mxt_lds_barrier();   // sMag

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // ABLK: a new 128-wide activation block starts with this slab; each row's ratio goes to LDS,
    // published by the staging barrier below (sR's previous readers were behind the last slab's
    // closing barrier).  A zero block keeps the carried scale; a zero block before this one sends
    // the carried scale back to the grid, which only an all-zero activation block ever does.
    const bool bnd = ABLK && k0 && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd)
        sR[tid] = ns > 0.f ? (np > 0.f ? np : r4d_mxfp4_carried(asrow, (k0 >> 7) - 2)) / ns : 1.f;
    }
    // A fragments for the whole slab, one clause, consumed only at the WMMAs.
    mxt_v2i af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const mxt_v2i*)(abase[i] +
                                      (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    // W slab -> LDS, folded to e4m3.
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int* src =
          &Wp[((size_t)(wrc[it] >> 4) * ksteps + kstep0 + wkst[it]) * 32 + wlanec[it]];
      const int blk = (k0 + wkst[it] * 16) / R4D_MXFP4_GROUP;
      unsigned int wq[WSLOTS], wsw;
      if constexpr (WSLOTS == 4) {
        const mxt_u4 v = *(const mxt_u4*)src;
        wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        wsw = *(const unsigned int*)(Ws + (size_t)blk * N + wrc[it]);
      } else if constexpr (WSLOTS == 2) {
        const mxt_u2 v = *(const mxt_u2*)src;
        wq[0] = v[0]; wq[1] = v[1];
        wsw = *(const unsigned short*)(Ws + (size_t)blk * N + wrc[it]);
      } else {
        wq[0] = *src;
        wsw = Ws[(size_t)blk * N + wrc[it]];
      }
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const int d = r4d_mxfp4_fold_d((int)((wref[it] >> (8 * q)) & 0xFF),
                                       (int)((wsw >> (8 * q)) & 0xFF));
        unsigned int lo, hi;
        r4d_mxfp4_unpack8_t(wq[q], sMag[d * 2], sMag[d * 2 + 1], &lo, &hi);
        *(mxt_u2*)(&sW[(wrow[it] + q) * STR + wkloc[it]]) = mxt_u2{lo, hi};
      }
    }
    mxt_lds_barrier();

    if constexpr (ABLK) {
      // Re-express each row's partial sum in the new block's scale: one ratio a lane a row.
      if (bnd) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const float r = sR[wm * TM * 16 + i * 16 + col];
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] *= r;
        }
      }
      // The next boundary's scales, issued behind this slab's loads and needed a boundary later.
      if ((k0 & 127) == 0 && k0 + 128 < K) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      mxt_v2i wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      // One step's W fragments live at a time: without the fence the compiler hoists every
      // step's loads above the first WMMA and holds them all.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf[j], af[i][step],
                                                                          acc[i][j]);
    }
    mxt_lds_barrier();
  }

  // Epilogue.  A lane holds row m0 + 64*wm + 16i + col of M fragment i and eight consecutive columns of N fragment
  // j, so each (i, j) is one 16-byte store.  2^(Wref[n]-127) is exactly the float whose exponent
  // field is Wref[n]; N % 16 == 0 keeps a lane's eight columns all inside N or all outside it.
  float sc[TM];
  if constexpr (ABLK) {
    // the scale each row ends in, once a row; the K loop's closing barrier retired the ratio reads
    sR[tid] = r4d_mxfp4_carried(asrow, (K - 1) >> 7);
    mxt_lds_barrier();
#pragma unroll
    for (int i = 0; i < TM; ++i) sc[i] = sR[wm * TM * 16 + i * 16 + col];
  } else {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      sc[i] = As[(size_t)(m < M ? m : M - 1) * as_rs];
    }
  }
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int nb = n0 + wn * TN * 16 + j * 16 + kb8;
    if (nb >= N) continue;
    const mxt_u2 rb = *(const mxt_u2*)(Wref + nb);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int m = m0 + wm * TM * 16 + i * 16 + col;
      if (m >= M) continue;
      union { __hip_bfloat16 h[8]; mxt_u4 v; } o;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float rf = __int_as_float((int)((rb[e >> 2] >> (8 * (e & 3))) & 0xFF) << 23);
        o.h[e] = (__hip_bfloat16)(acc[i][j][e] * rf * sc[i]);
      }
      *(mxt_u4*)(C + (size_t)m * N + nb) = o.v;
    }
  }
}

extern "C" int r4d_gemm_mxfp4a8_tiled_group(void) { return R4D_MXFP4_GROUP; }

extern "C" long r4d_mxfp4a8_tile_a_bytes(int M, int K) {
  return M < 1 || K < 16 ? 0 : (long)((M + 15) / 16) * 16 * K;
}

extern "C" void r4d_mxfp4a8_tile_a(long a, long at, int M, int K, long stream) {
  if (M < 1) throw std::runtime_error("r4d mxfp4a8_tile_a: M must be >= 1");
  if (K < 16 || K % 16) throw std::runtime_error("r4d mxfp4a8_tile_a: K must be a multiple of 16");
  const long long slots = (long long)((M + 15) / 16) * (K / 16) * 32;
  hipLaunchKernelGGL(r4d_mxfp4a8_tile_a_kernel, dim3((unsigned)((slots + 255) / 256)), dim3(256), 0,
                     (hipStream_t)stream, (const unsigned char*)a, (unsigned char*)at, M, K, slots);
}

extern "C" void r4d_gemm_mxfp4a8_tiled(long at, long ascale, long wq, long ws, long wref, long c,
                                       int M, int K, int N, int TN, int as_group, int as_rs,
                                       long stream) {
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: N must be a multiple of 16");
  if (K < R4D_MXT_KUNIT || K % R4D_MXT_KUNIT)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: K must be a multiple of 64");
  // tn=0: TN=4 from the M crossover on, unless its 128-wide tile leaves the grid short of 128
  // blocks (a narrow N): there the TN=2 grid's extra blocks win.
  if (TN == 0)
    TN = M >= R4D_MXT_TN4_MIN_M && (long)((N + 127) / 128) * ((M + R4D_MXT_BMF - 1) / R4D_MXT_BMF) >= 128
             ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: activation scale rows shorter than ceil(K/128)");
  const bool wide_slab = TN == 2 && K % 128 == 0;
  const int bnf = R4D_MXT_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_MXT_BMF - 1) / R4D_MXT_BMF), block(R4D_MXT_NTHR);
  auto Ap = (const unsigned char*)at;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define MXT_LAUNCH(TNv, LBKv, ABv)                                                              \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_tiled_kernel<TNv, LBKv, ABv>), grid, block, 0, st, Ap, Sp, \
                     Wp, Zp, Rp, Cp, M, K, N, as_rs)
  if (TN == 4)        { if (ablk) MXT_LAUNCH(4, 64, true);  else MXT_LAUNCH(4, 64, false); }
  else if (wide_slab) { if (ablk) MXT_LAUNCH(2, 128, true); else MXT_LAUNCH(2, 128, false); }
  else                { if (ablk) MXT_LAUNCH(2, 64, true);  else MXT_LAUNCH(2, 64, false); }
#undef MXT_LAUNCH
}

// ============================================================== the plugin ABI entry point
// The row-major activation is re-tiled into the op's scratch, then the GEMM reads it from there.
extern "C" int r4d_rad_gemm_mxfp4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.K % R4D_MXT_KUNIT) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent, as in r4d_rad_gemm_mxfp4a8_m64: without it the fold has no
     * reference and every element would be scaled against zero. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs);
    if (rc != RAD_OK) return rc;
    if (!a->scratch || a->scratch_bytes < r4d_mxfp4a8_tile_a_bytes((int)g.M, (int)g.K))
        return R4D_NO(RAD_E_INVAL);
    R4dMxTileCfg t;
    r4d_cfg_mxfp4_tiled(a, &t);
    const long at = (long)(uintptr_t)a->scratch;
    R4D_TRY("gemm_mxfp4a8_tiled",
        r4d_mxfp4a8_tile_a(r4d_addr(g.a), at, (int)g.M, (int)g.K, (long)(uintptr_t)s);
        r4d_gemm_mxfp4a8_tiled(at, r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale),
                               r4d_addr(g.bref), r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N,
                               t.tn, as_group, as_rs, (long)(uintptr_t)s));
    return RAD_OK;
}

/* The core sizes the op's scratch from this, at the band's widest M: the fragment-tiled copy of
 * the activation, whole 16-row fragments. */
extern "C" int64_t r4d_rad_gemm_mxfp4a8_tiled_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), K = r4d_geti_or(a, "K", 0);
    return (int64_t)r4d_mxfp4a8_tile_a_bytes((int)M, (int)K);
}
