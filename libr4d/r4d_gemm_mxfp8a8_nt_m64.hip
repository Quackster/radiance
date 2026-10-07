// r4d_gemm_mxfp8a8_nt_m64.hip - r4d_gemm_mxfp8a8_nt_m64: OCP-MXFP8 weight, fp8 activation skinny
// GEMM for M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row to
// e4m3 and W stored as OCP micro-scaling fp8: an e4m3 code per byte and one E8M0 exponent per 32
// elements along K.
//
// WHY THIS EXISTS ALONGSIDE r4d_gemm_w8a8_nt_m64.  That kernel's 8-bit code is an integer the
// library derives at load; this one's is a checkpoint format it does not get to choose.  An MXFP8
// checkpoint arrives with its own per-32 E8M0 exponents, and a serving stack that re-derives a
// scale has quantised twice.  The two also want the same matrix instruction -- both operands are
// already E4M3 bytes -- and on this part it runs at
//
//     v_wmma_f32_16x16x16_fp8_fp8   412 TF/s      2.99 cyc/instr
//
// so the format is the whole decision.
//
// THE PER-32 EXPONENT IS APPLIED ONCE PER GROUP, NOT FOLDED INTO A TABLE.  The naive MXFP8 dequant
// multiplies each element by 2^(E8M0-127), which is a per-32-block rescale.  Unlike MXFP4 there is
// no cheap fold: an e4m3 code has a full 3-bit mantissa, so scaling it by a power of two is not a
// table lookup -- it is a float multiply on every weight.  Instead the 32 products of a group are
// accumulated into a scratch accumulator, that sum is scaled by the group's ONE exponent
// 2^(sb-127), and the scaled sum is added into the running accumulator.  The scale is therefore
// read once per 32 K and applied once per 32 K, and the e4m3 codes themselves need no unpacking at
// all.
//
// THE E8M0 EXPONENT IS A PER-GROUP FACTOR AND CANNOT MOVE TO THE KERNEL EPILOGUE.  It multiplies
// an inner product over a K-subrange -- sum over 32 k of a*w -- not the whole row, so it has to be
// applied where that subrange closes.  The only factor left for the final epilogue is the
// activation's own per-row scale As[m]; there is no per-row weight reference as there is for MXFP4.
//
// THE B FRAGMENT IS THE STORED BYTES, UNCHANGED.  A lane's fragment is eight consecutive k, which
// is exactly the eight e4m3 bytes the fragment map puts at that lane's slot: bytes 0..3 are the
// first four k, bytes 4..7 the next four, i.e. the two 4-byte halves of the fp8 WMMA B operand in
// order.  Nothing is decoded, permuted or interleaved between the global read and the mma.
//
// SPLIT-K IS IN THE WORKGROUP, AS EVERYWHERE ELSE IN THIS LIBRARY.  With one 16-row M fragment the
// independent N fragments number N/16 -- 320 waves at N=5120 -- which does not fill the part, so
// the K range has to be split.  Doing it across workgroups needs an fp32 partial buffer in HBM and
// a counter to decide who reduces; doing it across waves of one workgroup reduces in LDS and needs
// neither.  The LDS form is also what keeps this kernel cudagraph-safe with no scratch to register:
// a caller can replay it without having handed the library any memory.
//
// THE WEIGHT IS PRE-PERMUTED INTO FRAGMENT ORDER, AS IT IS FOR w8a16/w8a8.  A B fragment maps lane
// l onto N row l&15, so in checkpoint order the sixteen lanes of a half-wave read sixteen rows K
// bytes apart -- a fully strided read on the operand that dominates traffic.  Reordering the weight
// once at load fixes it and is what lets this kernel skip LDS entirely.
//
// The layout is one 32-lane slot per (n-tile, k-step):
//
//     Wp[((n/16) * (K/16) + (k/16)) * 32 + lane]      one uint2 = 8 bytes = 8 elements
//
// where lane's row is 16*(n/16) + (lane&15) and its eight elements are the consecutive k at
// 16*(k/16) + 8*(lane>>4).  A wave then reads 256 contiguous bytes per step.  More precisely the
// byte index of element (n, k) is
//
//     (((n/16)*(K/16) + k/16)*32 + 16*((k%16)/8) + (n%16))*8 + (k%8)
//
// which is the frozen map in libr4d/r4d_layout.cpp (mxw_slot) exactly: lane = 16*((k%16)/8) +
// (n%16), and the lane's eight bytes are k%8 = 0..7.  In checkpoint terms, taking W as
// [N/16][16][K/16][8] bytes, slot l of tile (nt, ks) is W[nt][l&15][ks][8*(l>>4) .. +8].  This is
// the mxfp4 map widened from a nibble to a byte: same block/lane structure, one byte per code
// instead of two codes per byte, so a wave reads 256 bytes a step rather than 128.
//
// The E8M0 scales are stored transposed to [K/32][N] -- a half-wave reads sixteen consecutive bytes
// of those already -- and are decoded with one shift and bitcast: 2^(e-127) = __uint_as_float(e<<23).
//
// THE PERMUTE IS WHAT MAKES THE LDS-FREE FORM WORTH HAVING.  Reading checkpoint order directly,
// this kernel is slower than an LDS-staged one on every shape; with the permute it is faster than
// the LDS-staged form on every shape, at both ends of the M range.  It also moves the tuning: the
// fastest configs are the ones with deeper split-K (SK 4-8) and NPW 2, which only pay once the
// weight read coalesces.
//
// TRAP FOR ANYONE MEASURING THIS KERNEL: a weight of tens of megabytes streamed in a loop is
// Infinity-Cache resident on this part, so a bandwidth figure taken that way can read above the
// DRAM roofline.  It ranks kernels reading the same bytes; it is not achieved DRAM bandwidth.
//
// A IS NOT STAGED IN LDS.  A lane's A fragment is eight consecutive bytes of one row, so the
// global load already has the shape the fragment wants; staging it would cost LDS, a barrier and a
// second read for nothing.  Rows past M are CLAMPED rather than masked -- a tile past M recomputes
// the last valid row and the store discards it -- which keeps the inner loop branch-free.
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"   // lds_pitch / frag / Stage8: the A-side LDS staging the prefill kernel needs

#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#define R4D_GEMM_MXFP8_MAX_M 64
#define R4D_GEMM_MXFP8_GROUP 32       // K per E8M0 exponent, fixed by the OCP MX format
#define WAVE32 32

// E8M0 -> f32 including the two ends a plain `<<23` decode gets wrong: byte 0 is the subnormal
// 2^-127 (exact in f32) and byte 255 is the format's NaN. Same semantics as abi/rad_plugin.h's
// rad_e8m0_to_f32, which is host-only and so cannot be called from the kernel.
__device__ __forceinline__ float r4d_mxfp8_e8m0_to_f32(unsigned e) {
  return e == 0u   ? __uint_as_float(0x00400000u)      // 2^-127, the one subnormal
       : e == 0xffu ? __uint_as_float(0x7fc00000u)     // the format's NaN
                    : __uint_as_float(e << 23);
}

typedef int      v2i    __attribute__((ext_vector_type(2)));

template <int MT, int NPW>
__global__ void r4d_gemm_mxfp8a8_nt_m64_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  // Column tiles this wave owns, CLAMPED to the last real tile. grid.x rounds N/16 up to a
  // multiple of ncols = WV*NPW, so the tail block addresses tiles that do not exist -- at N=48
  // with ncols=4 it reaches tile 3 of 3. Clamping the TILE (not just the row) matters because the
  // weight is indexed by tile: an unclamped tile index walks off the end of the buffer, which is a
  // GPU memory fault rather than a wrong answer. The epilogue still drops these columns.
  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
  }

  const uint2* __restrict__ Wp = (const uint2*)W;
  const int ksteps = K / 16;

  // SOFTWARE-PIPELINED K LOOP. The decode form issued one weight read per K-step and waited on it
  // before the WMMA, so at prefill M -- where the block is the unit of work and the weight is the
  // cost -- the wave stalled on every K-step and the card sat at a fraction of its HBM bandwidth.
  // Here the NEXT k-step's A and W loads are issued before the CURRENT k-step's WMMA, so a K-tile's
  // memory requests are in flight while the previous one computes. The fragment-order weight read
  // (lane l's eight bytes at slot l, 256 contiguous bytes a wave) is unchanged; only the ordering
  // of the loads against the WMMA moves. The two in-flight fragment sets are the double buffer.
  //
  // The exponent is per 32 K, so the loop carries a group accumulator `gacc`: the two 16-k WMMAs of
  // a 32-group add into it, and when the group closes the sum is scaled by the group's E8M0 factor
  // and folded into `acc`. GROUP is fixed at 32 and Ks is a multiple of GROUP (the launcher
  // requires K % (SK*32) == 0), so the last group always closes on the last K-step.
  {
    const int kg0 = kbase;
    const int kstep0 = kg0 / 16;
    const int blk0 = kg0 / R4D_GEMM_MXFP8_GROUP;
    v2i af_cur[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + nlane;
      r = r < M - 1 ? r : M - 1;
      const unsigned int* pa = (const unsigned int*)(A + (size_t)r * K + kg0 + mhalf);
      af_cur[i][0] = (int)pa[0];
      af_cur[i][1] = (int)pa[1];
    }
    uint2 wv_cur[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++)
      wv_cur[j] = Wp[((size_t)wtile[j] * ksteps + kstep0) * WAVE32 + lane];
    v8f gacc[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++) gacc[j][i] = (v8f)(0.f);
    for (int k = 0; k < Ks; k += 16) {
      // Issue the NEXT k-step's loads (the in-flight buffer) before this k-step's WMMA.
      v2i af_next[MT];
      uint2 wv_next[NPW];
      const int k1 = k + 16;
      if (k1 < Ks) {
        const int kg1 = kbase + k1;
        const int kstep1 = kg1 / 16;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          int r = (mt0 + i) * 16 + nlane;
          r = r < M - 1 ? r : M - 1;
          const unsigned int* pa = (const unsigned int*)(A + (size_t)r * K + kg1 + mhalf);
          af_next[i][0] = (int)pa[0];
          af_next[i][1] = (int)pa[1];
        }
#pragma unroll
        for (int j = 0; j < NPW; j++)
          wv_next[j] = Wp[((size_t)wtile[j] * ksteps + kstep1) * WAVE32 + lane];
      }
      // Compute THIS k-step from the in-hand buffer. The stored 8 bytes ARE the B fragment.
      // A 32-lane fragment maps lane l onto column l&15, so byte j of the lane is B[k0 + j][l&15]
      // with k0 a multiple of 8 -- the fp8 WMMA's own K labelling, applied identically to A.
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        v2i wf;
        wf[0] = (int)wv_cur[j].x;    // k = 8*(lane>>4) + 0..3
        wf[1] = (int)wv_cur[j].y;    // k = 8*(lane>>4) + 4..7
#pragma unroll
        for (int i = 0; i < MT; i++)
          gacc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af_cur[i], wf, gacc[j][i]);
      }
      // Close the 32-group: (k+16) is a multiple of 32 exactly when the second half-step ends.
      // One E8M0 read per 32 K, one shift/bitcast, one scale-and-add into the persistent sum.
      if (((k + 16) & (R4D_GEMM_MXFP8_GROUP - 1)) == 0) {
        const int blk = blk0 + (k / R4D_GEMM_MXFP8_GROUP);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const float s = r4d_mxfp8_e8m0_to_f32((unsigned)Ws[(size_t)blk * N + wrow[j]]);
#pragma unroll
          for (int i = 0; i < MT; i++) {
#pragma unroll
            for (int e = 0; e < 8; e++) acc[j][i][e] += gacc[j][i][e] * s;
            gacc[j][i] = (v8f)(0.f);
          }
        }
      }
      // Advance the double buffer -- only when a next step was actually loaded. On the final
      // K-step `k1 == Ks` leaves `af_next/wv_next` unset, and reading them (as the carried `*_cur`
      // values) would be an uninitialised read; nothing consumes them after the loop, so the
      // honest fix is to not advance.
      if (k1 < Ks) {
#pragma unroll
        for (int i = 0; i < MT; i++) af_cur[i] = af_next[i];
#pragma unroll
        for (int j = 0; j < NPW; j++) wv_cur[j] = wv_next[j];
      }
    }
  }

  // Reduce the SK partial sums in LDS, one M tile at a time so the buffer stays WV*NPW*SK*256.
  // The E8M0 factor is already in `acc`; the epilogue's only factor is the activation's per-row
  // scale As[m], applied once.
  const int ncols = WV * NPW;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + lane * 8 + e] = acc[j][i][e];
    }
    __syncthreads();
    // One pass over the block's 256-element tiles; every thread takes a stride.
    for (int t = tid; t < ncols * 256; t += blockDim.x) {
      const int col = t >> 8, off = t & 255;
      // off = lane*8 + e. (off>>3)&15 recovers the lane's N index (lane&15) and off>>7 its M
      // half (lane>>4), so both half-waves of a 16-lane fragment land in the SAME 16-column tile
      // with no cross-column bleed at any M -- the map the mxfp4 template writes identically.
      const int n = (blockIdx.x * ncols + col) * 16 + ((off >> 3) & 15);
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * N + n] = (__hip_bfloat16)(v * As[m]);
    }
  }
}

extern "C" int r4d_gemm_mxfp8a8_nt_m64_max_m(void) { return R4D_GEMM_MXFP8_MAX_M; }
extern "C" int r4d_gemm_mxfp8a8_nt_m64_group(void) { return R4D_GEMM_MXFP8_GROUP; }

extern "C" void r4d_gemm_mxfp8a8_nt_m64(long a, long ascale, long wq, long ws, long c,
                                        int M, int K, int N, int WV, int SK, int MB, int NPW,
                                        long stream) {
  /* M is unbounded here: the kernel tiles M over blockIdx.y (rows past M are clamped and dropped),
   * so the same launch serves the prefill band. The DECODE entry (r4d_rad_gemm_mxfp8a8_m64) keeps
   * its own M <= R4D_GEMM_MXFP8_MAX_M gate; the PREFILL entry (r4d_rad_gemm_mxfp8a8_tiled) does not. */
  if (M < 1)
    throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_MXFP8_GROUP))
    throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: K must be divisible by SK*32");
  if (WV * SK * WAVE32 > 1024)
    throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: threads/block > 1024");
  /* MB is the 16-row M-fragments per wave (MT). The decode band uses 1..4 (<= 64 rows); the
   * prefill band uses 8 (a 128-row tile), which halves the weight traffic at the cost of a wider
   * accumulator (MT*NPW wmma fragments per lane). 5..7 are left out: they are slower than 4 or 8
   * for the same reason the fp8 GEMM's NT ladder skips holes -- the accumulator count is the cost. */
  if (MB < 1 || MB > 8) throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: MB must be 1..8");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024)
    throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MTv, NPv) hipLaunchKernelGGL((r4d_gemm_mxfp8a8_nt_m64_kernel<MTv, NPv>), grid, \
                                            block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NPv) switch (MB) { \
    case 1: LAUNCH(1, NPv); break; \
    case 2: LAUNCH(2, NPv); break; \
    case 3: LAUNCH(3, NPv); break; \
    case 4: LAUNCH(4, NPv); break; \
    case 8: LAUNCH(8, NPv); break; \
    default: throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: MB unsupported"); }
  switch (NPW) {
    case 1: LAUNCH_MB(1); break;
    case 2: LAUNCH_MB(2); break;
    case 4: LAUNCH_MB(4); break;
    case 8: LAUNCH_MB(8); break;
    default: throw std::runtime_error("r4d gemm_mxfp8a8_nt_m64: NPW unsupported");
  }
#undef LAUNCH_MB
#undef LAUNCH
}

// ====================================================== the PREFILL tiled kernel, A staged in LDS
//
// WHY THIS IS A SECOND KERNEL AND NOT THE ONE ABOVE. The m64 launch serves the prefill band by
// tiling M over blockIdx.y, and on CORRECTNESS that is enough. On TRAFFIC it is not: every one of
// the WV waves reads the SAME A rows from GLOBAL, because all the waves differ only in which N
// tile they own. With WV = 8 that is the A plane read EIGHT times per block, and A is a one-byte
// operand while the weight is also a byte -- so the redundant stream is the LARGER one. Measured on
// an idle card at M=1024, N=4096, K=8192 the m64 path moves ~2.1 GB (2 GB of it that redundant A)
// in 3.29 ms, and the fp8a8 tiled kernel moves 0.5 GB in 0.41 ms for the same output.
//
// THE FIX IS THE ONE fp8a8 ALREADY MAKES: stage the A K-tile into LDS ONCE, coalesced, and let every
// wave read its fragments out of LDS. A's global traffic drops from WV passes to one. The weight
// stays in fragment order read straight from global -- coalesced per wave, no LDS, no transpose --
// which is the m64 kernel's whole advantage and is unchanged here.
//
// THE REGISTER TILE IS KEPT AT MT = 8 (an M-fragment per wave, the m64 prefill band's shape) RATHER
// THAN TRANSPOSED. fp8a8 puts NT = 8 on a wave because its per-ROW activation scale makes a wide M
// tile cost registers; MX has no such fold -- the per-32 exponent is applied once per group -- so
// the binding cost here is the group close (a scale-and-add per 32 K), and that is amortised by MT:
// one group's work feeds MT M-fragments. At MT = 8 the group close is a sixteenth of the matrix
// work and all but disappears; at MT = 1 it is on the critical path for every WMMA.
//
// W is read one uint2 a lane a (n-tile, k-step) exactly as in the m64 kernel -- a 256-byte wave
// request -- and software-pipelined one k-step ahead across the whole K range, so the tile boundary
// is not a bubble. A is double-buffered in LDS: stage the next tile's registers while the current
// tile is multiplied out, one barrier a K-tile.
template <int MT, int NPW, int FK, int WV>
__global__ __launch_bounds__(WV * WAVE32)
void r4d_gemm_mxfp8a8_prefill_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C,
    int M, int K, int N) {
  using r4d_fp8::i2v;
  using r4d_fp8::f8v;
  using r4d_fp8::frag;
  using r4d_fp8::Stage8;
  using r4d_fp8::stage8_load;
  using r4d_fp8::stage8_store;
  using r4d_fp8::mma;
  using r4d_fp8::acc_zero;

  constexpr int FM = MT * 16;                 // M rows the block covers (a wave owns all of them)
  constexpr int kSeg = 32;                    // bytes a thread stages, contiguous
  constexpr int kTPR = FK / kSeg;             // threads covering one A row
  constexpr int kRPP = (WV * WAVE32) / kTPR;  // A rows one pass stages
  constexpr int kAP  = (FM + kRPP - 1) / kRPP;
  static_assert(kTPR >= 1 && kRPP >= 1, "FK must be at least 32 B and no wider than the block");

  __shared__ unsigned char s_a[2][FM * r4d_fp8::lds_pitch<FK>()];

  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int nlane = lane & 15;
  const int mt0 = blockIdx.y * FM;
  const int ksteps = K / 16;

  const int ntmax = N / 16 - 1;
  int ntile[NPW], nrow[NPW];
#pragma unroll
  for (int j = 0; j < NPW; ++j) {
    const int t = (blockIdx.x * WV + wave) * NPW + j;
    ntile[j] = t < ntmax ? t : ntmax;
    nrow[j] = ntile[j] * 16 + nlane;
  }

  const uint2* __restrict__ Wp = (const uint2*)W;
  const int sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

  // A rows past M are CLAMPED rather than masked, as in the m64 kernel: the load recomputes the
  // last valid row and the epilogue drops it, which keeps the staging branch-free.
  auto load_a = [&](Stage8 (&pa)[kAP], int c0) {
#pragma unroll
    for (int p = 0; p < kAP; ++p) {
      const int r = sn + p * kRPP;
      int rr = mt0 + r;
      rr = rr < M ? rr : M - 1;
      pa[p] = stage8_load(A + (size_t)rr * K + c0 + sk, r < FM);
    }
  };

  f8v acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[j][i] = acc_zero();

  // The per-32-group scratch accumulator, scaled by the group's E8M0 factor at each group close.
  // This prefill kernel has no split-K, so the groups walk from 0 with the K range.
  f8v gacc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
#pragma unroll
    for (int i = 0; i < MT; ++i) gacc[j][i] = acc_zero();

  Stage8 pa[kAP];
  load_a(pa, 0);

  // The weight for the first k-step is in hand before the loop, so the loop body only ever issues
  // the NEXT one.
  const int kstep0 = 0;
  uint2 wv_cur[NPW];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
    wv_cur[j] = Wp[((size_t)ntile[j] * ksteps + kstep0) * WAVE32 + lane];

  int buf = 0;
  for (int c0 = 0; c0 < K; c0 += FK) {
#pragma unroll
    for (int p = 0; p < kAP; ++p) {
      const int r = sn + p * kRPP;
      if (r < FM) stage8_store<FK>(s_a[buf], r, sk, pa[p]);
    }
    __syncthreads();
    if (c0 + FK < K) load_a(pa, c0 + FK);

#pragma unroll
    for (int ks = 0; ks < FK; ks += 16) {
      const int kg = c0 + ks;
      uint2 wv_next[NPW];
      if (kg + 16 < K) {
#pragma unroll
        for (int j = 0; j < NPW; ++j)
          wv_next[j] = Wp[((size_t)ntile[j] * ksteps + (kg + 16) / 16) * WAVE32 + lane];
      } else {
        /* The last k-step has nothing after it; carry the current values so the rotation below is
         * not an uninitialised read (the loop exits immediately after and they are never used). */
#pragma unroll
        for (int j = 0; j < NPW; ++j) wv_next[j] = wv_cur[j];
      }
      i2v af[MT];
#pragma unroll
      for (int i = 0; i < MT; ++i) af[i] = frag<FK>(s_a[buf], i * 16, ks, lane);
#pragma unroll
      for (int j = 0; j < NPW; ++j) {
        i2v wf;
        wf[0] = (int)wv_cur[j].x;
        wf[1] = (int)wv_cur[j].y;
#pragma unroll
        for (int i = 0; i < MT; ++i) gacc[j][i] = mma(af[i], wf, gacc[j][i]);
      }
      // Close the 32-group. FK is a multiple of 32 and c0 is a multiple of FK, so a group never
      // straddles a tile boundary; the final group closes on the final k-step because K % FK == 0
      // (enforced by the launcher) and K is a multiple of 32.
      if (((kg + 16) & (R4D_GEMM_MXFP8_GROUP - 1)) == 0) {
        const int blk = kg / R4D_GEMM_MXFP8_GROUP;
#pragma unroll
        for (int j = 0; j < NPW; ++j) {
          const float s = r4d_mxfp8_e8m0_to_f32((unsigned)Ws[(size_t)blk * N + nrow[j]]);
#pragma unroll
          for (int i = 0; i < MT; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j][i][e] += gacc[j][i][e] * s;
            gacc[j][i] = acc_zero();
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; ++j) wv_cur[j] = wv_next[j];
    }
    buf ^= 1;
  }

  // The epilogue is the m64 kernel's: the E8M0 factors are already folded into `acc` per 32-group,
  // so the only per-row factor left is the activation's own scale As[m], applied once.
#pragma unroll
  for (int j = 0; j < NPW; ++j) {
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = mt0 + i * 16 + (lane >> 4) * 8 + e;
        const int n = nrow[j];
        if (m < M && n < N)
          C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * As[m]);
      }
    }
  }
}

extern "C" void r4d_gemm_mxfp8a8_prefill(long a, long ascale, long wq, long ws, long c,
                                         int M, int K, int N, int WV, int MT, int NPW, int FK,
                                         long stream) {
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: N must be a multiple of 16");
  if (K % 32) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: K must be a multiple of 32");
  /* THE STAGED TILE AND THE K RANGE MUST AGREE. The kernel's A granule is FK bytes and it reads
   * c0+sk+32 with sk <= FK-32, so `K % FK` is what bounds the read at the last tile; and kTPR is
   * FK/kSeg, so FK must be a multiple of the granule. The ABI enforces both before it gets here, but
   * this is an extern "C" entry and a direct caller must be refused rather than overread. */
  if (FK % 32) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: FK must be a multiple of 32");
  if (K % FK) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: K must be a multiple of FK");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  /* CONFIG VALIDATION IS EXPLICIT, and the dispatch below is total over the instantiated set. An
   * out-of-range FK silently reduced to the 64-byte template would stage the wrong tile width and
   * read past A; an out-of-range MT/NPW silently reduced would corrupt the RELATIVE sweep the env
   * override exists for. */
  if (FK != 64 && FK != 128) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: FK must be 64 or 128");
  if (MT != 4 && MT != 8) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: MT must be 4 or 8");
  if (NPW != 1 && NPW != 2) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: NPW must be 1 or 2");
  if (WV != 8) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: WV must be 8");
  /* MT = 8 with NPW = 2 is 128 VGPRs of accumulator alone -- the cap r4d_gemm_fp8a8.hip calls
   * immediate spill -- so it is not instantiated; NPW = 2 is a MT = 4 shape. */
  if (MT == 8 && NPW == 2) throw std::runtime_error("r4d gemm_mxfp8a8_prefill: MT=8 needs NPW=1");
  const int ncols = WV * NPW;
  dim3 grid((N / 16 + ncols - 1) / ncols, (M + MT * 16 - 1) / (MT * 16));
  dim3 block(WV * WAVE32);
#define R4D_MXPF(MTv, NPv, FKv, WVv) \
  hipLaunchKernelGGL((r4d_gemm_mxfp8a8_prefill_kernel<MTv, NPv, FKv, WVv>), grid, block, 0, st, \
                     Ap, Sp, Wp, Zp, Cp, M, K, N)
  /* The shapes the tuning sweep reaches: FK = 64 and 128 (a wider staged tile coalesces the A fetch
   * better at the cost of LDS), MT = 8 or 4 (the group-close amortisation), WV = 8. R4D_MXCFG picks
   * one for a RELATIVE sweep; the default is the 128x128 tile at FK = 64. Every arm tests NPW, so a
   * config the sweep names is the shape it launches -- the FK = 128 arms must not match before NPW
   * is decided, or `R4D_MXCFG=4,2,128` would silently run NPW = 1 under a NPW = 2 label. */
  if      (FK == 128 && MT == 4 && NPW == 2) R4D_MXPF(4, 2, 128, 8);
  else if (FK == 128 && MT == 4)             R4D_MXPF(4, 1, 128, 8);
  else if (FK == 128)                        R4D_MXPF(8, 1, 128, 8);
  else if (MT == 4 && NPW == 2)              R4D_MXPF(4, 2, 64, 8);
  else if (MT == 4)                          R4D_MXPF(4, 1, 64, 8);
  else                                       R4D_MXPF(8, 1, 64, 8);
#undef R4D_MXPF
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
extern "C" int r4d_rad_gemm_mxfp8a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp8a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* MXFP8 is SELF-SCALED: the weight carries an E8M0 exponent per 32 K and there is no per-row
     * reference exponent, so there is NO b_ref operand and no fold. The layout hooks return
     * RAD_E_UNSUPPORTED for operand 6, so a caller that supplies one is refused upstream. */
    const int group = r4d_gemm_mxfp8a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_mxfp8a8_nt_m64",
        r4d_gemm_mxfp8a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                                (long)(uintptr_t)s));
    return RAD_OK;
}

// The PREFILL band of the same kernel, tuned for the case where the weight is the cost. It is the
// m64 launch with the M <= 64 gate lifted: SK = 1 (a prefill chunk has hundreds of M tiles to fill
// the part, so the split-K decode needs to fill a 64-CU part is pure overhead here -- its only
// effect is the LDS reduction), WV = 8 (a 256-thread block) gives each block eight N lanes to hide
// the group close behind, and cfg_for() is not used because it caps MB at 4, the decode band's
// ceiling and the wrong one for prefill. THE TILE IS NOT A FIXED 128x128: the launcher picks
// between two shapes split at M = 256 (below it MT = 4 / FK = 128 to double the shallow M grid;
// above it MT = 8 / FK = 64), and R4D_MXCFG can override the shape for a relative tuning sweep.
// When K % FK != 0 the staged-tile kernel is not ABI-valid, so the launcher falls back to the m64
// launch with MB = 8 -- named `gemm_mxfp8a8_nt_tiled` below -- which serves any 32-aligned K. The
// numerics -- fragment-order weight read, per-32-group E8M0 fold, per-row As[m] epilogue -- are the
// m64 kernel's, so this row and the decode row agree to the last bit at every M they share.
extern "C" int r4d_rad_gemm_mxfp8a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    /* The 128-wide N tile reads a clamped tail for N < 128, which is correct but wasted; the
     * mxfp8a8 GEMMs' N is the hidden size (thousands), so this never fires in the trunk. */
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* No b_ref for MXFP8; the weight is self-scaled per 32 K. */
    /* The prefill tile. R4D_MXCFG=MT,NPW,FK overrides the shape for tuning sweeps (the absolute
     * numbers are contention-noisy with another model on the card, so this is for RELATIVE
     * comparison of block shapes, not for recording a default). WV is fixed at 8.
     *
     * THE DEFAULT IS TWO SHAPES, split at M = 256. Below it the M grid is what fills the card --
     * ceil(M/128) is 1 or 2 blocks deep -- so a 64-row tile (MT = 4) doubles it, and the wider
     * staged tile (FK = 128) coalesces the A fetch the fp8a8 table measures at 626 against 422 GB/s
     * for the 64-byte form. Above it the grid is already deep and the narrower tile's lower LDS use
     * (and thus more resident blocks) wins. */
    int MT = 8, NPW = 1, FK = 64;
    if (g.M <= 256) { MT = 4; FK = (g.K % 128 == 0) ? 128 : 64; }
    /* The override is accepted only for the shapes the launcher has instantiated and validated --
     * MT in {4,8}, NPW in {1,2}, FK in {64,128}, and never MT=8/NPW=2 (128 accumulator VGPRs).
     * Anything else is ignored rather than silently reduced to a different shape, which would
     * corrupt the relative sweep this exists for.
     *
     * It is a benchmarking/tuning hook, never a production knob: it is honoured only when
     * R4D_ALLOW_MXCFG is ALSO set, and the first honouring logs the chosen shape once. A stale
     * R4D_MXCFG left in the environment of a served model otherwise silently changes the kernel
     * shape with no opt-in. */
    if (const char* e = std::getenv("R4D_MXCFG")) {
        if (std::getenv("R4D_ALLOW_MXCFG")) {
            int mtv, npwv, fkv;
            if (std::sscanf(e, "%d,%d,%d", &mtv, &npwv, &fkv) == 3 &&
                (mtv == 4 || mtv == 8) && (npwv == 1 || npwv == 2) && (fkv == 64 || fkv == 128) &&
                !(mtv == 8 && npwv == 2)) {
                MT = mtv; NPW = npwv; FK = fkv;
                static bool logged = false;
                if (!logged) {
                    std::fprintf(stderr, "radiance: R4D_MXCFG override active: MT=%d NPW=%d FK=%d\n",
                                 MT, NPW, FK);
                    logged = true;
                }
            }
        }
    }
    /* The staged-tile kernel needs K a multiple of the K-tile, and the whole 8-wave grid needs
     * N % 16 (the tail tile is clamped). K % 64 is a property of every trunk shape (hidden and
     * intermediate sizes are multiples of 128); a K that is only 32-aligned falls back to the m64
     * launch, which has no staged tile and serves any 32-aligned K. */
    if (g.K % FK == 0) {
        R4D_TRY("gemm_mxfp8a8_nt_prefill",
            r4d_gemm_mxfp8a8_prefill(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                     r4d_addr(g.bscale), r4d_addr(g.y),
                                     (int)g.M, (int)g.K, (int)g.N, 8, MT, NPW, FK,
                                     (long)(uintptr_t)s));
        return RAD_OK;
    }
    int WV = 8, SK = 1, MB = 8, PNPW = 1;
    R4D_TRY("gemm_mxfp8a8_nt_tiled",
        r4d_gemm_mxfp8a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, WV, SK, MB, PNPW,
                                (long)(uintptr_t)s));
    return RAD_OK;
}
