// r4d_gemm_mxfp4a8_tiled.hip - r4d_gemm_mxfp4a8_tiled: the OCP-MXFP4 x fp8 GEMM for PREFILL row
// counts, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the same operands, the same weight
// layout and the same fold as r4d_gemm_mxfp4a8_nt_m64 -- this is its large-M sibling.
//
// WHY A SECOND KERNEL.  The skinny kernel gives a block at most 64 rows and reads the weight
// straight into registers, so at prefill it re-reads the whole weight ceil(M/64) times: at M=8192
// on gate_up (17408 x 5120) that is 128 passes over 45 MB.  Here a block is a 256 x (32*TN) tile,
// A goes through LDS once per K slab and is reused across every column of the tile, and the
// weight is unpacked once per slab into LDS where all four row-waves read it.
//
// Ported from SlyBase/vllm-sly-radiance (radiance_mxfp4_fp8_gemm_folded, WPERM arm), where it
// served Qwen3.8-27B MXFP4 prefill on one R9700.  What changed on the way in:
//   - the folded-magnitude table is libr4d's (r4d_gemm_mxfp4a8_nt_m64.hip), so this kernel and
//     the skinny one fold every element to the SAME e4m3 byte.  The two tables differed at
//     d >= 9 (round-to-nearest-even there); a model that crosses M=64 between steps must not
//     change numerics with it.
//   - only the fragment-order weight path is kept: it is the layout r4d_rad_layout_mxfp4 stores,
//     and spec §"Agreement" requires every kernel reading a weight to ask for the same layout.
//   - the merged-linear partition select (three rotated copies of A in one launch) is gone; it
//     belonged to a ParoQuant path radiance does not have.
//   - TN, chosen by an environment variable there, is the tunable `tn` here.
//
// THE FOLD.  Same as the skinny kernel: take Wref[n] = max_k E8M0[n][k] once at load, push the
// difference d = Wref[n] - E8M0[n][blk] into the e4m3 value through a 16 x 8 table of packed e4m3
// bytes (v_perm_b32, no multiply), and apply 2^(Wref[n]-127) once in the epilogue next to the
// activation's per-row scale.  The table runs into e4m3 subnormals, which the gfx12 fp8 WMMA
// honours; see r4d_gemm_mxfp4a8_nt_m64.hip for the error analysis.
//
// THE TILE.  8 waves as 4 (M) x 2 (N); a wave owns TM=4 x TN accumulator fragments, so the block
// covers BMF = 4*4*16 = 256 rows and BNF = 2*TN*16 columns, with a K slab of BK = 64.  BM=256 won
// a correctness-gated sweep at every production shape: a taller tile buys more A reuse per weight
// byte.  TN is the axis that depends on M -- measured on gate_up 17408x5120, TN=4 against TN=2:
//     M=8192 +10.0%  M=4096 +8.5%  M=2048 +1.3%  M=1024 -4.2%  M=512 -8.8%
// Widening N amortises the A staging (ablated at 24% of runtime) over more columns, but below
// the crossover it halves the column-block count while BMF still masks most of a short M.  Hence
// tn=0 (the default) picks TN=4 from M >= 2048 and TN=2 below.
//
// STAGING IS CLAMPED, NEVER PREDICATED.  A bounds-predicated load lands in its own exec-mask
// region and the compiler then cannot carry a counted s_wait_loadcnt across the merge: it emits a
// full wait after every load.  Clamped, a slab's loads issue back to back under one wait.  Safe
// because the epilogue drops rows past M and columns past N, and a clamped read is real data.
//
// WEIGHT STAGING.  Fragment order is one 32-lane uint32 slot per (n-tile, k-step); adjacent slots
// differ only in the row, so a thread reads WSLOTS adjacent slots with one load (4 = 16 B at TN=4,
// 2 = 8 B at TN=2, where the tile has only half a block's worth of 16 B reads) and the wave still
// reads contiguous bytes.  The block and reference exponents for those rows are one dword or one
// short each (N % 16 == 0 keeps them aligned).
//
// TWO ACTIVATION SCALE GRIDS.  `a_scale` is either one f32 per row (ABLK=false: what a per-token
// quantiser writes) or one per (row, 128 of K) (ABLK=true: the block-scaled fp8 family's grid,
// which every fused producer in arch/common writes, so an architecture can put an MXFP4 weight
// behind any of them).  Per-block scales vary along K and so cannot wait for the epilogue; a
// second accumulator to fold them per block would double the 128 accumulator VGPRs at TN=4.
// Instead the accumulator is carried IN UNITS OF THE CURRENT BLOCK'S SCALE: at each 128 boundary
// every row's partial sum is multiplied by s_prev / s_next, and the epilogue applies the last
// block's scale.  That is one multiply per accumulator element per 128 of K and one division per
// row, in fp32, so the extra rounding is ~K/128 fp32 ulps -- far below the e4m3 operands' own.
// An all-zero activation block (scale 0) keeps the previous scale: its codes are zero, so the
// block contributes nothing whichever scale the accumulator is carried in.
//
// THE CARRIED SCALES ARE NOT HELD IN REGISTERS.  A lane owns TM*8 = 32 rows, and keeping a float
// for each took TN=4 from 192 VGPRs to 256 plus 264 B/lane of scratch (occupancy 8 -> 5) and TN=2
// from 140 to 230 -- measured at ~80 TF/s on a 34816x5120x8192 prefill.  Re-reading the grid per
// lane did not help either: the unrolled rescale hoists all 32 loads and holds them.  So the
// block's 256 threads compute the 256 rows' ratios ONCE, one row each, into 1 KiB of LDS, and a
// lane reads them back eight at a time -- one fragment row-group -- behind a scheduling fence,
// so at most eight are live.  A zero grid entry walks back to the last non-zero one, which only an
// all-zero activation block ever triggers.
//
// TRAP FOR ANYONE MEASURING THIS KERNEL: the same as the skinny kernel's.  A weight of tens of
// megabytes streamed in a loop is Infinity-Cache resident on this part, so a bandwidth figure
// taken that way reads above the DRAM roofline.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_MXT_GROUP 32        // K per E8M0 exponent, fixed by the OCP MX format
#define R4D_MXT_TM    4         // 16-row fragments per wave
#define R4D_MXT_WM    4         // waves along M
#define R4D_MXT_WN    2         // waves along N
#define R4D_MXT_BK    64        // K per slab
#define R4D_MXT_PAD   8         // LDS row pad: 16 lanes reading rows 64 B apart collide 8 ways
#define R4D_MXT_STR   (R4D_MXT_BK + R4D_MXT_PAD)
#define R4D_MXT_NTHR  (R4D_MXT_WM * R4D_MXT_WN * 32)
#define R4D_MXT_BMF   (R4D_MXT_WM * R4D_MXT_TM * 16)
#define R4D_MXT_TN4_MIN_M 2048  // tn=0's crossover, measured above

typedef int      mxt_v2i  __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u2   __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u4   __attribute__((ext_vector_type(4)));

// libr4d's folded-magnitude table, copied from r4d_gemm_mxfp4a8_nt_m64.hip and NOT regenerated:
// the two kernels must fold every element to the same byte.  kMag[d] is the eight e2m1
// magnitudes scaled by 2^-d as packed e4m3, low word then high word.
__device__ __constant__ unsigned int kMxtMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

template <int TN, bool ABLK>
__global__ __launch_bounds__(R4D_MXT_NTHR) void r4d_gemm_mxfp4a8_tiled_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs) {
  constexpr int TM = R4D_MXT_TM, WN = R4D_MXT_WN, BK = R4D_MXT_BK, STR = R4D_MXT_STR;
  constexpr int NTHREADS = R4D_MXT_NTHR, BMF = R4D_MXT_BMF;
  constexpr int BNF = WN * TN * 16;
  __shared__ unsigned char sA[BMF * STR];
  __shared__ unsigned char sW[BNF * STR];
  __shared__ float sR[ABLK ? BMF : 1];   // ABLK: this boundary's per-row ratio, then the final scale

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps_g = K / 16;

  // ABLK: the scale row m's accumulator is carried in after block kb -- the last non-zero
  // entry at or before kb, or 1 when there is none. Read from the grid, not held (see above).
  auto carried = [&](int m, int kb) -> float {
    const float* __restrict__ row = As + (size_t)m * as_rs;
    for (int j = kb; j >= 0; --j) {
      const float s = row[j];
      if (s > 0.f) return s;
    }
    return 1.f;
  };

  for (int k0 = 0; k0 < K; k0 += BK) {
    if constexpr (ABLK) {
      // A new 128-wide activation block starts with this slab (BK divides 128): re-express every
      // row's partial sum in the new block's scale before any of its products land.
      // One row a thread: BMF == NTHREADS. sR is free here -- its last readers were behind the
      // previous slab's closing barrier -- and the staging barrier below publishes it.
      if (k0 && (k0 & 127) == 0) {
        const int kb = k0 >> 7;
        const int m = m0 + tid < M ? m0 + tid : M - 1;
        const float s = As[(size_t)m * as_rs + kb];
        sR[tid] = s > 0.f ? carried(m, kb - 1) / s : 1.f;   // a zero block keeps the carried scale
      }
    }
    // Wave-uniform base, so the compiler keeps it in SGPRs and emits the SADDR form: one 32-bit
    // add per lane instead of a 64-bit add pair.  r*K < 256 * K fits an int at any real K.
    const unsigned char* __restrict__ Ab = A + (size_t)m0 * K + k0;
    // ---- A tile: BMF x BK bytes, 16 B per thread per iteration
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(mxt_u4*)(&sA[r * STR + c]) = *(const mxt_u4*)(Ab + (rc * K + c));
    }
    // ---- W tile: fragment-order slots, WSLOTS adjacent rows per 16 B read, folded into e4m3
    {
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      // 4 slots (one 16 B read) wherever the tile has enough of them -- TN=4 -- else 2 (8 B):
      // at TN=2 the tile is 512 slots, half a 256-thread block's worth of 16 B reads.
      constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
      static_assert(TOT_SLOTS % (NTHREADS * WSLOTS) == 0, "W tile must split evenly over the block");
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
        const int sl = off + tid * WSLOTS;              // first of the group; WSLOTS-aligned
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        // gn is WSLOTS-aligned and N % 16 == 0, so clamping to the last WSLOTS-aligned row keeps
        // the vector read aligned and in bounds; the slot address is recomputed from the clamp.
        const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
        const int lanec = (lane_ & 16) | (gc & 15);
        const unsigned int* src = &Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
        const int blk = (k0 + kst * 16) / R4D_MXT_GROUP;
        unsigned int wq[WSLOTS], refw, wsw;
        if constexpr (WSLOTS == 4) {
          const mxt_u4 v = *(const mxt_u4*)src;
          wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
          refw = *(const unsigned int*)(Wref + gc);
          wsw  = *(const unsigned int*)(Ws + (size_t)blk * N + gc);
        } else {
          const mxt_u2 v = *(const mxt_u2*)src;
          wq[0] = v[0]; wq[1] = v[1];
          refw = *(const unsigned short*)(Wref + gc);
          wsw  = *(const unsigned short*)(Ws + (size_t)blk * N + gc);
        }
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMxtMag[d][0], t1 = kMxtMag[d][1];
          const unsigned int ev = wq[q] & 0x0F0F0F0Fu;
          const unsigned int od = (wq[q] >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(mxt_u2*)(&sW[(r + q) * STR + kloc]) =
              mxt_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                     __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
    __syncthreads();

    if constexpr (ABLK) {
      if (k0 && (k0 & 127) == 0) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          float r[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) r[e] = sR[wm * TM * 16 + i * 16 + kb8 + e];
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] *= r[e];
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      mxt_v2i af[TM], wf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char* p = &sA[(wm * TM * 16 + i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)p; af[i][1] = *(const int*)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      // Without this fence the compiler hoists every step's fragment loads above the first WMMA
      // and holds four steps of fragments live (measured: 140 v_dual_mov_b32 of shuffling ahead of
      // a 64-WMMA run).  One step live at a time is what the register budget is sized for.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // ABLK: the scale every row ends in, one row a thread, through the same LDS row. The K loop's
  // closing barrier retired the last ratio reads; this one publishes the scales.
  if constexpr (ABLK) {
    const int m = m0 + tid < M ? m0 + tid : M - 1;
    sR[tid] = carried(m, (K - 1) >> 7);
    __syncthreads();
  }
  // Epilogue: 2^(Wref[n]-127) is exactly the float whose exponent field is Wref[n].
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  // A block entirely inside M and N needs no bounds test, and with a wave-uniform base each store
  // is one 32-bit offset add; on a real prefill only the last row and column block are ragged.
  // Bit-identical to the guarded path below: same operands, same order, same rounding.
  __hip_bfloat16* __restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
  const float* __restrict__ Asb = As + m0 + wm * TM * 16;
  const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                    (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
  if (full) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = i * 16 + kb8 + e;
          const int n = n0 + wn * TN * 16 + j * 16 + col;
          Cb[r * N + n] = (__hip_bfloat16)(acc[i][j][e] * rf[j] *
                                           (ABLK ? sR[wm * TM * 16 + r] : Asb[r]));
        }
    return;
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N)
          C[(size_t)m * N + n] =
              (__hip_bfloat16)(acc[i][j][e] * rf[j] *
                               (ABLK ? sR[wm * TM * 16 + i * 16 + kb8 + e] : As[m]));
      }
}

// ================================================================= the A-FRAGMENT-TILED variant
// r4d_gemm_mxfp4a8_atiled_kernel: the same tile, W staging, fold and epilogue as the kernel above,
// but the activation arrives in WMMA-FRAGMENT-TILED layout and goes straight from global memory
// into the registers the WMMA reads.  There is no sA, no A staging and no A fragment ds_read: only
// W goes through LDS (9 KB at TN=4), so three blocks fit a CU where the staged kernel fits two.
//
// Ported from SlyBase/vllm-sly-radiance (radiance_mxfp4_fp8_gemm_atiled, which is built on the
// ggz14 / StillDeadcode fragment-tiled kernel), where it measured +11-16 % prefill over the folded
// kernel this file's staged one is the sibling of.  What changed on the way in:
//   - only the fragment-order weight is kept (as in the staged kernel), no E6 arm, no merged-linear
//     partition select;
//   - the folded-magnitude table is THIS FILE's kMxtMag, copied into LDS once per block;
//   - the (row, 128 of K) activation grid (ABLK) is carried exactly as in the staged kernel: the
//     per-row ratio is computed into LDS at each 128 boundary and applied to the accumulators, so
//     the two kernels agree bit for bit.  vLLM's kernel had no such grid (per-row scale only).
//
// A LAYOUT, produced by r4d_rmsnorm_quant_fp8 / r4d_gated_quant_fp8 when the op carries `afrag`
// and M >= afrag.  Each 16 (m) x 16 (k) fp8 fragment is 256 contiguous bytes in lane order:
//     byte  ((mt * (K/16) + ks) * 256) + lane * 8 + j  =  A[mt*16 + lane%16][ks*16 + (lane/16)*8 + j]
// so one global_load_b64 per lane per fragment lands in the register the WMMA reads.  The storage
// is whole 16-row tiles; the rows past M in the last tile are never written and only reach
// accumulators the epilogue drops (WMMA rows are independent), so a NaN there is harmless.
//
// Codegen notes carried over from the source (all measured there, bit-identical each):
//   - the fragment bases are SGPRs (readfirstlane on the tile offset) with an UNSIGNED 32-bit lane
//     offset, so the loads take the SADDR form and issue as one s_clause;
//   - the magnitude table is read from LDS: a global load for it is a dependent load issued behind
//     the A fragments, and loadcnt is in order;
//   - the barriers fence LDS only.  __syncthreads() is a workgroup-scope acquire/release on every
//     address space (a global_inv and a full loadcnt drain, twice per slab), and nothing in global
//     memory is shared between waves here.
// LBK, K per slab: with no A tile a 128-deep W slab is affordable at TN=2 and halves the barriers
// there (measured +2-4 %); at TN=4 it costs registers (occupancy 5) and loses, so TN=4 stays at 64.
static __device__ __forceinline__ void mxt_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int TN, bool ABLK, int LBK>
__global__ __launch_bounds__(R4D_MXT_NTHR) void r4d_gemm_mxfp4a8_atiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs) {
  constexpr int TM = R4D_MXT_TM, WN = R4D_MXT_WN, BK = R4D_MXT_BK;
  constexpr int NTHREADS = R4D_MXT_NTHR, BMF = R4D_MXT_BMF;
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + R4D_MXT_PAD;
  constexpr int BNF = WN * TN * 16;
  static_assert(LBK % BK == 0, "slab is a multiple of the K unit");
  static_assert(!ABLK || BMF == NTHREADS, "ABLK: one ratio row per thread");
  __shared__ unsigned char sW[BNF * LWSTR];
  __shared__ unsigned int sMag[32];             // kMxtMag, 128 B: a ds_load keeps it off loadcnt
  __shared__ float sR[ABLK ? BMF : 1];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = ((const unsigned int*)kMxtMag)[tid];
  mxt_lds_barrier();

  // One A tile base per M fragment: wave-uniform, forced into SGPRs.  Tile-granular clamp, never
  // predicate: rows of a clamped tile only reach accumulators the epilogue drops.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted: see the staged kernel for the slot arithmetic.
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
  constexpr int NW = TOT_SLOTS / (NTHREADS * WSLOTS);
  static_assert(NW * NTHREADS * WSLOTS == TOT_SLOTS, "W slot groups must tile the block");
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHREADS * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wcol[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
  }
  unsigned int wref[NW];   // WSLOTS per-row reference exponents, packed
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int*)(Wref + wrc[it]);
    else                       wref[it] = *(const unsigned short*)(Wref + wrc[it]);
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  // ABLK: the scale row m's accumulator is carried in after block kb (see the staged kernel).
  auto carried = [&](int m, int kb) -> float {
    const float* __restrict__ row = As + (size_t)m * as_rs;
    for (int j = kb; j >= 0; --j) {
      const float s = row[j];
      if (s > 0.f) return s;
    }
    return 1.f;
  };

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  for (int k0 = 0; k0 < K; k0 += LBK) {
    if constexpr (ABLK) {
      // LBK divides 128 or is 128, so a 128 boundary is always a slab start.
      if (k0 && (k0 & 127) == 0) {
        const int kb = k0 >> 7;
        const int m = m0 + tid < M ? m0 + tid : M - 1;
        const float s = As[(size_t)m * as_rs + kb];
        sR[tid] = s > 0.f ? carried(m, kb - 1) / s : 1.f;
      }
    }
    // A fragments straight from global: one SADDR clause, consumed only at the WMMAs.
    mxt_v2i af[TM][NS];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const mxt_v2i*)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));

    // ---- W tile -> LDS, folded to e4m3 (kMxtMag[d] through v_perm, as the staged kernel does)
    mxt_u4 pk[NW];
    unsigned int wsv[NW];
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int* src = &Wp[((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]];
      if constexpr (WSLOTS == 4) pk[it] = *(const mxt_u4*)src;
      else { const mxt_u2 v = *(const mxt_u2*)src; pk[it] = mxt_u4{v[0], v[1], 0u, 0u}; }
      const int blk = (k0 + wkst[it] * 16) / R4D_MXT_GROUP;
      if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int*)(Ws + (size_t)blk * N + wrc[it]);
      else                       wsv[it] = *(const unsigned short*)(Ws + (size_t)blk * N + wrc[it]);
    }
#pragma unroll
    for (int it = 0; it < NW; ++it)
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const mxt_u2 t = *(const mxt_u2*)&sMag[d * 2];
        const unsigned int wq = pk[it][q];
        const unsigned int ev = wq & 0x0F0F0F0Fu;
        const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        *(mxt_u2*)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
            mxt_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                   __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
    mxt_lds_barrier();

    if constexpr (ABLK) {
      if (k0 && (k0 & 127) == 0) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          float r[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) r[e] = sR[wm * TM * 16 + i * 16 + kb8 + e];
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] *= r[e];
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      mxt_v2i wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int*)p; wf[j][1] = *(const int*)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
    }
    mxt_lds_barrier();
  }

  if constexpr (ABLK) {
    const int m = m0 + tid < M ? m0 + tid : M - 1;
    sR[tid] = carried(m, (K - 1) >> 7);
    mxt_lds_barrier();
  }
  // Epilogue: identical to the staged kernel's (bit-identical by construction).
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  {
    __hip_bfloat16* __restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float* __restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            Cb[r * N + n] = (__hip_bfloat16)(acc[i][j][e] * rf[j] *
                                             (ABLK ? sR[wm * TM * 16 + r] : Asb[r]));
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N)
          C[(size_t)m * N + n] =
              (__hip_bfloat16)(acc[i][j][e] * rf[j] *
                               (ABLK ? sR[wm * TM * 16 + i * 16 + kb8 + e] : As[m]));
      }
}

// Same operands and checks as r4d_gemm_mxfp4a8_tiled, with `a` in the fragment-tiled layout above
// (whole 16-row tiles; a pointer 16-byte aligned).  TN 0 = the same M rule as the staged kernel.
extern "C" void r4d_gemm_mxfp4a8_atiled(long a, long ascale, long wq, long ws, long wref, long c,
                                        int M, int K, int N, int TN, int as_group, int as_rs,
                                        long stream) {
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp4a8_atiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_atiled: N must be a multiple of 16");
  if (K < R4D_MXT_BK || K % R4D_MXT_BK)
    throw std::runtime_error("r4d gemm_mxfp4a8_atiled: K must be a multiple of 64");
  if (TN == 0) TN = M >= R4D_MXT_TN4_MIN_M ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("r4d gemm_mxfp4a8_atiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_atiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_atiled: activation scale rows shorter than ceil(K/128)");
  const int bnf = R4D_MXT_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_MXT_BMF - 1) / R4D_MXT_BMF), block(R4D_MXT_NTHR);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  // TN=2 takes a 128-deep slab when K allows; TN=4 stays at 64 (see the note above the kernel).
  const bool l128 = TN == 2 && K % 128 == 0;
#define MXT_ALAUNCH(TNv, ABv, LBv) hipLaunchKernelGGL((r4d_gemm_mxfp4a8_atiled_kernel<TNv, ABv, LBv>), \
                                        grid, block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs)
  if (TN == 4) { if (ablk) MXT_ALAUNCH(4, true, 64); else MXT_ALAUNCH(4, false, 64); }
  else if (l128) { if (ablk) MXT_ALAUNCH(2, true, 128); else MXT_ALAUNCH(2, false, 128); }
  else           { if (ablk) MXT_ALAUNCH(2, true, 64);  else MXT_ALAUNCH(2, false, 64); }
#undef MXT_ALAUNCH
}


extern "C" int r4d_gemm_mxfp4a8_tiled_group(void) { return R4D_MXT_GROUP; }
extern "C" int r4d_gemm_mxfp4a8_tiled_k_unit(void) { return R4D_MXT_BK; }

extern "C" void r4d_gemm_mxfp4a8_tiled(long a, long ascale, long wq, long ws, long wref, long c,
                                       int M, int K, int N, int TN, int as_group, int as_rs,
                                       long stream) {
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: N must be a multiple of 16");
  if (K < R4D_MXT_BK || K % R4D_MXT_BK)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: K must be a multiple of 64");
  if (TN == 0) TN = M >= R4D_MXT_TN4_MIN_M ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("r4d gemm_mxfp4a8_tiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_tiled: activation scale rows shorter than ceil(K/128)");
  const int bnf = R4D_MXT_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + R4D_MXT_BMF - 1) / R4D_MXT_BMF), block(R4D_MXT_NTHR);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define MXT_LAUNCH(TNv, ABv) hipLaunchKernelGGL((r4d_gemm_mxfp4a8_tiled_kernel<TNv, ABv>), grid, \
                                                 block, 0, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs)
  if (TN == 4) { if (ablk) MXT_LAUNCH(4, true); else MXT_LAUNCH(4, false); }
  else         { if (ablk) MXT_LAUNCH(2, true); else MXT_LAUNCH(2, false); }
#undef MXT_LAUNCH
}

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_mxfp4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.K % r4d_gemm_mxfp4a8_tiled_k_unit()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent, as in r4d_rad_gemm_mxfp4a8_m64: without it the fold has no
     * reference and every element would be scaled against zero. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs);
    if (rc != RAD_OK) return rc;
    R4dMxTileCfg t;
    r4d_cfg_mxfp4_tiled(a, &t);
    /* `afrag` = T > 0 is the ARCHITECTURE'S statement that its producers write this op's A in the
     * fragment-tiled layout whenever M >= T (rad_fp8.h ActFP8::frag), the same T they carry. The
     * decision is a function of the issue's M alone, so a producer and its consumers agree without
     * any state between them. Absent (every op today) it is the row-major kernel below. */
    const long long afrag = r4d_geti_or(a, "afrag", 0);
    if (afrag > 0 && g.M >= afrag) {
        if (afrag <= 64) return R4D_NO(RAD_E_INVAL);   /* the skinny/decode rows own M <= 64 */
        R4D_TRY("gemm_mxfp4a8_atiled",
            r4d_gemm_mxfp4a8_atiled(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                    r4d_addr(g.bscale), r4d_addr(g.bref), r4d_addr(g.y),
                                    (int)g.M, (int)g.K, (int)g.N, t.tn, as_group, as_rs,
                                    (long)(uintptr_t)s));
        return RAD_OK;
    }
    R4D_TRY("gemm_mxfp4a8_tiled",
        r4d_gemm_mxfp4a8_tiled(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                               r4d_addr(g.bscale), r4d_addr(g.bref), r4d_addr(g.y),
                               (int)g.M, (int)g.K, (int)g.N, t.tn, as_group, as_rs,
                               (long)(uintptr_t)s));
    return RAD_OK;
}
