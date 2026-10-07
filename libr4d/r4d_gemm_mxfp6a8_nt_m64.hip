// r4d_gemm_mxfp6a8_nt_m64.hip - r4d_gemm_mxfp6a8_nt_m64: OCP-MXFP6 weight, fp8 activation skinny
// GEMM for M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row to
// e4m3 and W stored as OCP micro-scaling fp6: an e2m3 element per 6-bit code and one E8M0 exponent
// per 32 elements along K.
//
// THIS IS THE MXFP6 SIBLING OF r4d_gemm_mxfp4a8_nt_m64.  Same 32-lane fragment structure, same
// workgroup split-K, same LDS reduction and epilogue shape.  Two things differ, and both are
// format properties rather than tuning choices:
//
//   1. THE CODE IS SIX BITS, NOT FOUR.  E2M3 is a 1-2-3 float (sign, 2 exponent bits with bias 1,
//      3 mantissa bits) topping out at 7.5, and it is packed four codes per three bytes along the
//      row by the ABI's own sub-byte rule (rad_store_bits/rad_load_bits: lowest bits first, four
//      codes per three bytes little-endian).  A lane holds eight consecutive-k codes = two 4-code
//      groups = six bytes, and a wave reads 192 contiguous bytes per (n-tile, 16-k step).
//
//   2. THERE IS NO wref/dsh FOLD.  E4M3 (mxfp8) and E2M3 (mxfp6) codes are SELF-SCALED: the code
//      carries its own exponent, and the only scale is the E8M0 plane.  mxfp4's fold exists because
//      an e2m1 magnitude has no exponent to fold into; here there is one, and the fold would be a
//      no-op at best.  So instead of folding a per-row reference into a magnitude table, the kernel
//      promotes every code EXACTLY to e4m3 and applies the group's E8M0 factor to the block sum:
//
//          acc += block_sum * 2^(sb[n][g] - 127)   per 32-column group
//
//      The group is exactly two 16-k WMMA steps, so the two steps accumulate into a block
//      accumulator that is scaled and folded into the running accumulator once per group.  That is
//      one extra v8f per (NPW, MT) slot -- the price of no fold -- and it leaves the epilogue with
//      just the activation's own per-row scale to apply.
//
// THE PROMOTION e2m3 -> e4m3 IS EXACT.  E2M3 has three mantissa bits and exponent bias 1; its
// magnitudes are 0, M/8 (E=0), and 2^(E-1)*(1+M/8) for E in 1..3, all inside [0, 7.5].  E4M3 has
// the same three mantissa bits, normals from 2^-6 up, so every E2M3 magnitude is an exact E4M3
// value (values below 0.125 are normal in e4m3, with room to spare).  A 64-entry constexpr table
// indexed by the six-bit code gives the e4m3 byte directly; a host static_assert re-derives both
// formats independently and checks the table entry for entry.  No arithmetic, no clamping, no
// rounding: the promoted fragment is the operand the fp8 WMMA wants.
//
// THE WEIGHT IS PRE-PERMUTED INTO FRAGMENT ORDER, AS IT IS FOR w4a16/w4a8/mxfp4.  The byte map is
// the mxfp8 one used as an ELEMENT index into an N*K 6-bit stream:
//
//     idx = (((n/16)*(K/16) + k/16)*32 + 16*((k%16)/8) + (n%16))*8 + (k%8)
//
// and because 16*((k%16)/8) + (n%16) is exactly the lane id, lane l of (n-tile t, k-step ks) holds
// the eight element indices [((t*(K/16) + ks)*32 + l)*8, +8) -- eight consecutive-k codes, a 6-byte
// little-endian chunk.  So a lane reads six bytes at ((t*ksteps + ks)*32 + l)*6 and a wave's
// request is 192 contiguous bytes.  A2's frozen map (board_68a4f7e8) and libr4d/r4d_layout.cpp's
// mxw_slot()+rad_store_bits are the reference; this kernel matches them bit for bit.
//
// The E8M0 scales are left in [K/32][N] order -- a half-wave reads sixteen consecutive bytes of
// those already.
//
// SPLIT-K IS IN THE WORKGROUP, AS EVERYWHERE ELSE IN THIS LIBRARY.  With one 16-row M fragment the
// independent N fragments number N/16 -- 320 waves at N=5120 -- which does not fill the part, so the
// K range has to be split.  Doing it across workgroups needs an fp32 partial buffer in HBM and a
// counter; doing it across waves of one workgroup reduces in LDS and needs neither.  Because each
// split is a whole number of 32-groups, every split's block sums are already scaled by their own
// group exponents, so the LDS reduction is a plain sum with no further scale.
//
// A IS NOT STAGED IN LDS in the m64 kernel.  A lane's A fragment is eight consecutive bytes of one
// row, so the global load already has the shape the fragment wants.  Rows past M are CLAMPED rather
// than masked, which keeps the inner loop branch-free.  The prefill kernel below stages A through
// LDS because at prefill M every wave would otherwise re-read the same A rows.
//
// TRAP FOR ANYONE MEASURING THIS KERNEL: a weight of tens of megabytes streamed in a loop is
// Infinity-Cache resident on this part, so a bandwidth figure taken that way can read above the
// DRAM roofline.  It ranks kernels reading the same bytes; it is not achieved DRAM bandwidth.
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"   // lds_pitch / frag / Stage8: the A-side LDS staging the prefill kernel needs

#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <stdexcept>

#define R4D_GEMM_MXFP6_MAX_M 64
#define R4D_GEMM_MXFP6_GROUP 32       // K per E8M0 exponent, fixed by the OCP MX format
#define WAVE32 32
#define R4D_MXFP6_LANE_BYTES 6        // eight 6-bit codes = two 4-code groups = six bytes

typedef int      v2i    __attribute__((ext_vector_type(2)));

// The exact E2M3 -> E4M3 map.  Index is the 6-bit E2M3 code (sign bit5, exponent bits4:3 bias 1,
// mantissa bits2:0); the byte is the E4M3 value of the same number.  Generated by the arithmetic
// below and pinned by the static_assert after it -- the table is checked, not trusted.
#define R4D_MXFP6_PROM_ENTRIES                                                            \
  0x00u, 0x20u, 0x28u, 0x2Cu, 0x30u, 0x32u, 0x34u, 0x36u,  /* E=0, sign 0: M/8       */ \
  0x38u, 0x39u, 0x3Au, 0x3Bu, 0x3Cu, 0x3Du, 0x3Eu, 0x3Fu,  /* E=1, sign 0: (1+M/8)   */ \
  0x40u, 0x41u, 0x42u, 0x43u, 0x44u, 0x45u, 0x46u, 0x47u,  /* E=2, sign 0: 2*(1+M/8) */ \
  0x48u, 0x49u, 0x4Au, 0x4Bu, 0x4Cu, 0x4Du, 0x4Eu, 0x4Fu,  /* E=3, sign 0: 4*(1+M/8) */ \
  0x80u, 0xA0u, 0xA8u, 0xACu, 0xB0u, 0xB2u, 0xB4u, 0xB6u,  /* E=0, sign 1            */ \
  0xB8u, 0xB9u, 0xBAu, 0xBBu, 0xBCu, 0xBDu, 0xBEu, 0xBFu,  /* E=1, sign 1            */ \
  0xC0u, 0xC1u, 0xC2u, 0xC3u, 0xC4u, 0xC5u, 0xC6u, 0xC7u,  /* E=2, sign 1            */ \
  0xC8u, 0xC9u, 0xCAu, 0xCBu, 0xCCu, 0xCDu, 0xCEu, 0xCFu   /* E=3, sign 1            */

// One source of values, two views: the device constant-memory table the kernel indexes, and a host
// constexpr copy the static_assert reads.
constexpr unsigned char kE2M3ToE4M3Const[64] = { R4D_MXFP6_PROM_ENTRIES };
__device__ __constant__ unsigned char kE2M3ToE4M3[64] = { R4D_MXFP6_PROM_ENTRIES };

// E8M0 -> f32 including the ends a plain `<<23` decode gets wrong: byte 0 is the subnormal 2^-127
// and byte 255 is NaN. Same semantics as abi/rad_plugin.h's rad_e8m0_to_f32 (host-only).
__device__ __forceinline__ float r4d_mxfp6_e8m0_to_f32(unsigned e) {
  return e == 0u   ? __uint_as_float(0x00400000u)      // 2^-127, the one subnormal
       : e == 0xffu ? __uint_as_float(0x7fc00000u)     // the format's NaN
                    : __uint_as_float(e << 23);
}

namespace r4d_mxfp6_table {
// Independent decoders, so the check is of the format semantics and not of the encoder that made
// the table.  E2M3: bias 1; E==0 is M/8, else 2^(E-1)*(1+M/8).  E4M3: bias 7; E==0 is M/8 * 2^-6,
// else 2^(E-7)*(1+M/8).
constexpr double dec_e2m3(unsigned c) {
  const unsigned e = (c >> 3) & 3u, m = c & 7u;
  const double frac = (double)m / 8.0;
  const double mag = (e == 0u) ? frac : (1.0 + frac) * (double)(1u << (e - 1u));
  return (c & 0x20u) ? -mag : mag;
}
constexpr double dec_e4m3(unsigned b) {
  const unsigned e = (b >> 3) & 0xFu, m = b & 7u;
  double mag = 0.0;
  if (e == 0u) {
    mag = ((double)m / 8.0) * (1.0 / 64.0);
  } else {
    const int k = (int)e - 7;
    const double p = (k >= 0) ? (double)(1u << k) : (1.0 / (double)(1u << (-k)));
    mag = (1.0 + (double)m / 8.0) * p;
  }
  return (b & 0x80u) ? -mag : mag;
}
constexpr bool ok() {
  for (int c = 0; c < 64; ++c)
    if (dec_e4m3(kE2M3ToE4M3Const[c]) != dec_e2m3((unsigned)c)) return false;
  return true;
}
}  // namespace r4d_mxfp6_table
static_assert(r4d_mxfp6_table::ok(),
              "r4d mxfp6: kE2M3ToE4M3 is not the exact e2m3->e4m3 map");

// Six bytes a lane = eight 6-bit codes, little-endian along the row, four codes per three bytes:
//   v0 = b0 & 0x3F; v1 = ((b1 & 0xF) << 2) | (b0 >> 6); v2 = ((b2 & 3) << 4) | (b1 >> 4);
//   v3 = b2 >> 2; and the same for bytes 3..5.  Loaded explicitly as six bytes so the read never
// crosses the end of the N*K*6/8-byte stream.
__device__ __forceinline__ unsigned long long r4d_mxfp6_load6(const unsigned char* __restrict__ p) {
  const unsigned long long lo = (unsigned long long)p[0] | ((unsigned long long)p[1] << 8) |
                                ((unsigned long long)p[2] << 16) | ((unsigned long long)p[3] << 24);
  const unsigned long long hi = (unsigned long long)p[4] | ((unsigned long long)p[5] << 8);
  return lo | (hi << 32);
}

// The lane's 6-byte chunk in the fragment stream: slot = (ntile*ksteps + kstep)*32 + lane, six
// bytes a slot.
__device__ __forceinline__ const unsigned char* r4d_mxfp6_slot(
    const unsigned char* __restrict__ W, int ntile, int ksteps, int kstep, int lane) {
  return W + (((size_t)ntile * ksteps + kstep) * WAVE32 + (size_t)lane) * R4D_MXFP6_LANE_BYTES;
}

// Eight E2M3 codes (six bytes, packed in `w`) -> eight E4M3 bytes in k order, two dwords.
__device__ __forceinline__ v2i r4d_mxfp6_unpack8(unsigned long long w) {
  const unsigned int v0 = (unsigned int)(w & 0x3Fu);
  const unsigned int v1 = (unsigned int)((w >> 6) & 0x3Fu);
  const unsigned int v2 = (unsigned int)((w >> 12) & 0x3Fu);
  const unsigned int v3 = (unsigned int)((w >> 18) & 0x3Fu);
  const unsigned int v4 = (unsigned int)((w >> 24) & 0x3Fu);
  const unsigned int v5 = (unsigned int)((w >> 30) & 0x3Fu);
  const unsigned int v6 = (unsigned int)((w >> 36) & 0x3Fu);
  const unsigned int v7 = (unsigned int)((w >> 42) & 0x3Fu);
  v2i out;
  out[0] = (int)((unsigned int)kE2M3ToE4M3[v0] | ((unsigned int)kE2M3ToE4M3[v1] << 8) |
                 ((unsigned int)kE2M3ToE4M3[v2] << 16) | ((unsigned int)kE2M3ToE4M3[v3] << 24));
  out[1] = (int)((unsigned int)kE2M3ToE4M3[v4] | ((unsigned int)kE2M3ToE4M3[v5] << 8) |
                 ((unsigned int)kE2M3ToE4M3[v6] << 16) | ((unsigned int)kE2M3ToE4M3[v7] << 24));
  return out;
}

template <int MT, int NPW>
__global__ void r4d_gemm_mxfp6a8_nt_m64_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ A1, const float* __restrict__ As1,
    const unsigned char* __restrict__ A2, const float* __restrict__ As2,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int pb1, int pb2) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  /* THE PARTITION-SELECT ACTIVATION. `pb1`/`pb2` are 16-aligned column offsets; INT_MAX (or a null
   * pointer) means "absent", so a wave below every boundary reads the base A and a wave at or past a
   * boundary reads that partition's codes and per-row scale. The paro path pins NPW == 1, so one
   * wave owns exactly one 16-wide N tile and can never straddle a boundary. */
  const int col0 = nt0 * 16;
  const bool p1 = A1 && col0 >= pb1;
  const bool p2 = A2 && col0 >= pb2;
  const unsigned char* Al = p2 ? A2 : (p1 ? A1 : A);
  const float* Asl = p2 ? As2 : (p1 ? As1 : As);
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;
  const int ksteps = K / 16;

  // acc is the running sum; tmp is the current 32-group's block sum, scaled in once per group.
  v8f acc[NPW][MT], tmp[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) { acc[j][i] = (v8f)(0.f); tmp[j][i] = (v8f)(0.f); }

  // Column tiles this wave owns, CLAMPED to the last real tile. grid.x rounds N/16 up to a multiple
  // of ncols = WV*NPW, so the tail block addresses tiles that do not exist; clamping the TILE (not
  // just the row) keeps the weight index inside the buffer. The epilogue still drops these columns.
  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
  }

  // SOFTWARE-PIPELINED K LOOP. The next k-step's A and W loads are issued before the current
  // k-step's WMMA, so a K-tile's memory requests are in flight while the previous one computes.
  // The weight read (lane l's six bytes at slot l, 192 contiguous a wave) is unchanged; only the
  // ordering of the loads against the WMMA moves.
  {
    const int kg0 = kbase;
    const int kstep0 = kg0 / 16;
    v2i af_cur[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + nlane;
      r = r < M - 1 ? r : M - 1;
      const unsigned int* pa = (const unsigned int*)(Al + (size_t)r * K + kg0 + mhalf);
      af_cur[i][0] = (int)pa[0];
      af_cur[i][1] = (int)pa[1];
    }
    unsigned long long wv_cur[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++)
      wv_cur[j] = r4d_mxfp6_load6(r4d_mxfp6_slot(W, wtile[j], ksteps, kstep0, lane));

    for (int k = 0; k < Ks; k += 16) {
      // Issue the NEXT k-step's loads before this k-step's WMMA.
      v2i af_next[MT];
      unsigned long long wv_next[NPW];
      const int k1 = k + 16;
      if (k1 < Ks) {
        const int kg1 = kbase + k1;
        const int kstep1 = kg1 / 16;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          int r = (mt0 + i) * 16 + nlane;
          r = r < M - 1 ? r : M - 1;
          const unsigned int* pa = (const unsigned int*)(Al + (size_t)r * K + kg1 + mhalf);
          af_next[i][0] = (int)pa[0];
          af_next[i][1] = (int)pa[1];
        }
#pragma unroll
        for (int j = 0; j < NPW; j++)
          wv_next[j] = r4d_mxfp6_load6(r4d_mxfp6_slot(W, wtile[j], ksteps, kstep1, lane));
      }
      // Compute THIS k-step into the group accumulator.
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const v2i wf = r4d_mxfp6_unpack8(wv_cur[j]);
#pragma unroll
        for (int i = 0; i < MT; i++)
          tmp[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af_cur[i], wf, tmp[j][i]);
      }
      // Every second 16-k step closes a 32-group: scale its block sum by 2^(sb-127) and fold it
      // into acc, then reset the block accumulator. Ks is a multiple of 32, so a group always
      // closes inside the split (k & 16 is nonzero exactly on the second half of a group).
      if (k & 16) {
        const int blk = (kbase + k - 16) / R4D_GEMM_MXFP6_GROUP;
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const float s = r4d_mxfp6_e8m0_to_f32((unsigned)Ws[(size_t)blk * N + wrow[j]]);
#pragma unroll
          for (int i = 0; i < MT; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
              acc[j][i][e] += tmp[j][i][e] * s;
              tmp[j][i][e] = 0.f;
            }
        }
      }
      // Advance the double buffer -- only when a next step was actually loaded.
      if (k1 < Ks) {
#pragma unroll
        for (int i = 0; i < MT; i++) af_cur[i] = af_next[i];
#pragma unroll
        for (int j = 0; j < NPW; j++) wv_cur[j] = wv_next[j];
      }
    }
  }

  // Reduce the SK partial sums in LDS, one M tile at a time so the buffer stays WV*NPW*SK*256.
  // Each partial already carries its own groups' E8M0 scales, so this is a plain sum.
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
      const int n = (blockIdx.x * ncols + col) * 16 + ((off >> 3) & 15);
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * N + n] = (__hip_bfloat16)(v * Asl[m]);
    }
  }
}

extern "C" int r4d_gemm_mxfp6a8_nt_m64_max_m(void) { return R4D_GEMM_MXFP6_MAX_M; }
extern "C" int r4d_gemm_mxfp6a8_nt_m64_group(void) { return R4D_GEMM_MXFP6_GROUP; }

extern "C" void r4d_gemm_mxfp6a8_nt_m64(long a, long ascale, long wq, long ws, long c,
                                        int M, int K, int N, int WV, int SK, int MB, int NPW,
                                        long stream, long a1, long as1, long a2, long as2,
                                        int pb1, int pb2) {
  /* M is unbounded here: the kernel tiles M over blockIdx.y (rows past M are clamped and dropped),
   * so the same launch serves the prefill band. The DECODE entry (r4d_rad_gemm_mxfp6a8_m64) keeps
   * its own M <= R4D_GEMM_MXFP6_MAX_M gate; the PREFILL entry (r4d_rad_gemm_mxfp6a8_tiled) does not. */
  if (M < 1)
    throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_MXFP6_GROUP))
    throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: K must be divisible by SK*32");
  if (WV * SK * WAVE32 > 1024)
    throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 8) throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: MB must be 1..8");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB);
  dim3 block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024)
    throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Cp = (__hip_bfloat16*)c;
  auto A1p = (const unsigned char*)a1;
  auto S1p = (const float*)as1;
  auto A2p = (const unsigned char*)a2;
  auto S2p = (const float*)as2;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MTv, NPv) hipLaunchKernelGGL((r4d_gemm_mxfp6a8_nt_m64_kernel<MTv, NPv>), grid, \
                                            block, smem, st, Ap, Sp, A1p, S1p, A2p, S2p, Wp, Zp, \
                                            Cp, M, K, N, WV, SK, pb1, pb2)
#define LAUNCH_MB(NPv) switch (MB) { \
    case 1: LAUNCH(1, NPv); break; \
    case 2: LAUNCH(2, NPv); break; \
    case 3: LAUNCH(3, NPv); break; \
    case 4: LAUNCH(4, NPv); break; \
    case 8: LAUNCH(8, NPv); break; \
    default: throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: MB unsupported"); }
  switch (NPW) {
    case 1: LAUNCH_MB(1); break;
    case 2: LAUNCH_MB(2); break;
    case 4: LAUNCH_MB(4); break;
    case 8: LAUNCH_MB(8); break;
    default: throw std::runtime_error("r4d gemm_mxfp6a8_nt_m64: NPW unsupported");
  }
#undef LAUNCH_MB
#undef LAUNCH
}

// ====================================================== the PREFILL tiled kernel, A staged in LDS
//
// WHY THIS IS A SECOND KERNEL AND NOT THE ONE ABOVE. The m64 launch serves the prefill band by
// tiling M over blockIdx.y, and on CORRECTNESS that is enough. On TRAFFIC it is not: every one of
// the WV waves reads the SAME A rows from GLOBAL, because all the waves differ only in which N tile
// they own. With WV = 8 that is the A plane read eight times per block, and A is a one-byte operand
// while the weight is six bits -- so the redundant stream is the larger one. The fix is the one
// fp8a8/mxfp4 already make: stage the A K-tile into LDS ONCE, coalesced, and let every wave read
// its fragments out of LDS.
//
// THE WEIGHT STAYS IN FRAGMENT ORDER READ STRAIGHT FROM GLOBAL -- coalesced per wave, no LDS, no
// transpose. The register tile is kept at MT = 8 (a 128-row tile) because the binding cost is the
// 6-byte unpack and that is amortised by MT: one unpacked weight fragment feeds MT WMMAs. W is read
// one 6-byte chunk a lane a (n-tile, k-step) exactly as in the m64 kernel and software-pipelined
// one k-step ahead across the whole K range; A is double-buffered in LDS.
template <int MT, int NPW, int FK, int WV>
__global__ __launch_bounds__(WV * WAVE32)
void r4d_gemm_mxfp6a8_prefill_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N) {
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

  const int sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

  // A rows past M are CLAMPED rather than masked: the load recomputes the last valid row and the
  // epilogue drops it, which keeps the staging branch-free.
  auto load_a = [&](Stage8 (&pa)[kAP], int c0) {
#pragma unroll
    for (int p = 0; p < kAP; ++p) {
      const int r = sn + p * kRPP;
      int rr = mt0 + r;
      rr = rr < M ? rr : M - 1;
      pa[p] = stage8_load(A + (size_t)rr * K + c0 + sk, r < FM);
    }
  };

  f8v acc[NPW][MT], tmp[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
#pragma unroll
    for (int i = 0; i < MT; ++i) { acc[j][i] = acc_zero(); tmp[j][i] = acc_zero(); }

  Stage8 pa[kAP];
  load_a(pa, 0);

  const int kg0 = 0;
  const int kstep0 = kg0 / 16;
  unsigned long long wv_cur[NPW];
#pragma unroll
  for (int j = 0; j < NPW; ++j)
    wv_cur[j] = r4d_mxfp6_load6(r4d_mxfp6_slot(W, ntile[j], ksteps, kstep0, lane));

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
      unsigned long long wv_next[NPW];
      if (kg + 16 < K) {
#pragma unroll
        for (int j = 0; j < NPW; ++j)
          wv_next[j] = r4d_mxfp6_load6(r4d_mxfp6_slot(W, ntile[j], ksteps, (kg + 16) / 16, lane));
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
        const i2v wf = (i2v)r4d_mxfp6_unpack8(wv_cur[j]);
#pragma unroll
        for (int i = 0; i < MT; ++i) tmp[j][i] = mma(af[i], wf, tmp[j][i]);
      }
      // Close a 32-group: scale the block sum in and reset it. FK is a multiple of 32, so each tile
      // boundary is a group boundary and tmp is zero at the first k-step of every tile.
      if (kg & 16) {
        const int blk = (kg - 16) / R4D_GEMM_MXFP6_GROUP;
#pragma unroll
        for (int j = 0; j < NPW; ++j) {
          const float s = r4d_mxfp6_e8m0_to_f32((unsigned)Ws[(size_t)blk * N + nrow[j]]);
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              acc[j][i][e] += tmp[j][i][e] * s;
              tmp[j][i][e] = 0.f;
            }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; ++j) wv_cur[j] = wv_next[j];
    }
    buf ^= 1;
  }

  // The epilogue applies the activation's per-row scale once; the group exponents were folded into
  // acc inside the loop.
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

extern "C" void r4d_gemm_mxfp6a8_prefill(long a, long ascale, long wq, long ws, long c,
                                         int M, int K, int N, int WV, int MT, int NPW, int FK,
                                         long stream) {
  if (M < 1) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: N must be a multiple of 16");
  if (K % 32) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: K must be a multiple of 32");
  if (FK % 32) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: FK must be a multiple of 32");
  if (K % FK) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: K must be a multiple of FK");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  if (FK != 64 && FK != 128) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: FK must be 64 or 128");
  if (MT != 4 && MT != 8) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: MT must be 4 or 8");
  if (NPW != 1 && NPW != 2) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: NPW must be 1 or 2");
  if (WV != 8) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: WV must be 8");
  /* The block accumulator doubles the live register tile, so MT = 8 with NPW = 2 (128 accumulators
   * plus 128 block accumulators) is not instantiated; NPW = 2 is a MT = 4 shape. */
  if (MT == 8 && NPW == 2) throw std::runtime_error("r4d gemm_mxfp6a8_prefill: MT=8 needs NPW=1");
  const int ncols = WV * NPW;
  dim3 grid((N / 16 + ncols - 1) / ncols, (M + MT * 16 - 1) / (MT * 16));
  dim3 block(WV * WAVE32);
#define R4D_MXPF(MTv, NPv, FKv, WVv) \
  hipLaunchKernelGGL((r4d_gemm_mxfp6a8_prefill_kernel<MTv, NPv, FKv, WVv>), grid, block, 0, st, \
                     Ap, Sp, Wp, Zp, Cp, M, K, N)
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
// The operand reader and the split law are in r4d_gemm_args.h. Unlike mxfp4a8 there is NO b_ref:
// E2M3 codes are self-scaled and the fold is mxfp4's alone, so the shape hook and the layout hook
// both declare operand 6 (Q_BREF) absent for this dtype.
extern "C" int r4d_rad_gemm_mxfp6a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp6a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_mxfp6a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_mxfp6a8_nt_m64",
        r4d_gemm_mxfp6a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                                (long)(uintptr_t)s, 0L, 0L, 0L, 0L, INT_MAX, INT_MAX));
    return RAD_OK;
}

// The PREFILL band of the same kernel. Same structure as r4d_rad_gemm_mxfp4a8_tiled minus b_ref:
// the staged-tile prefill kernel when K % FK == 0, otherwise the m64 launch with MB = 8, which
// serves any 32-aligned K. The numerics -- fragment-order weight read, exact e2m3->e4m3 promotion,
// per-32-group E8M0 scale folded into the block sum, per-row As[m] epilogue -- are the m64 kernel's,
// so this row and the decode row agree to the last bit at every M they share.
extern "C" int r4d_rad_gemm_mxfp6a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The prefill tile. R4D_MXCFG=MT,NPW,FK overrides the shape for relative tuning sweeps, gated on
     * R4D_ALLOW_MXCFG. WV is fixed at 8. The default splits at M = 256: below it the narrower,
     * wider-staged tile doubles the shallow M grid; above it the narrower tile's lower LDS use wins. */
    int MT = 8, NPW = 1, FK = 64;
    if (g.M <= 256) { MT = 4; FK = (g.K % 128 == 0) ? 128 : 64; }
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
    if (g.K % FK == 0) {
        R4D_TRY("gemm_mxfp6a8_nt_prefill",
            r4d_gemm_mxfp6a8_prefill(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                     r4d_addr(g.bscale), r4d_addr(g.y),
                                     (int)g.M, (int)g.K, (int)g.N, 8, MT, NPW, FK,
                                     (long)(uintptr_t)s));
        return RAD_OK;
    }
    int WV = 8, SK = 1, MB = 8, PNPW = 1;
    R4D_TRY("gemm_mxfp6a8_nt_tiled",
        r4d_gemm_mxfp6a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, WV, SK, MB, PNPW,
                                (long)(uintptr_t)s, 0L, 0L, 0L, 0L, INT_MAX, INT_MAX));
    return RAD_OK;
}

// ================================================== the PARTITION-SELECT (paro) entry
//
// ONE GEMM launch that reads a DIFFERENT per-partition activation for each output-column partition
// of a merged weight. `pb1`/`pb2` are 16-aligned column offsets; `a` covers columns below pb1,
// `a2` the columns at or past pb1, and `a3` the columns at or past pb2. The paro path pins NPW == 1
// so a wave owns exactly one 16-wide tile and never straddles a boundary -- which is what lets the
// per-wave selection below be exact rather than approximate. Correctness first: the decode band
// resolves the same cfg the plain m64 entry does (with npw forced to 1); past M == 64 the tiled
// band uses the same WV=8/SK=1/MB=8/NPW=1 fallback the plain tiled row uses for an unstaged K.
extern "C" int r4d_rad_gemm_mxfp6a8_paro_m64(const RadArgs* a, RadStream s) {
    GemmOpsParo g;
    int rc = gemm_ops_paro(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.K % 32) return R4D_NO(RAD_E_SHAPE);
    /* The boundaries are column offsets into the merged output, and a wave is one 16-wide tile, so a
     * boundary that is not 16-aligned would make the selection straddle a tile and be wrong. */
    const long long pb1 = r4d_geti_or(a, "pb1", INT_MAX);
    const long long pb2 = r4d_geti_or(a, "pb2", INT_MAX);
    if (pb1 != INT_MAX && (pb1 < 0 || (pb1 % 16))) return R4D_NO(RAD_E_SHAPE);
    if (pb2 != INT_MAX && (pb2 < 0 || (pb2 % 16))) return R4D_NO(RAD_E_SHAPE);
    int WV, SK, MB;
    if (g.M <= r4d_gemm_mxfp6a8_nt_m64_max_m()) {
        const int group = r4d_gemm_mxfp6a8_nt_m64_group();
        Cfg c;
        rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
        if (rc != RAD_OK) return rc;
        WV = c.wv; SK = c.sk; MB = c.mb;
    } else {
        WV = 8; SK = 1; MB = 8;
    }
    R4D_TRY("gemm_mxfp6a8_paro",
        r4d_gemm_mxfp6a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, WV, SK, MB, 1,
                                (long)(uintptr_t)s,
                                r4d_addr(g.a2), r4d_addr(g.ascale2),
                                r4d_addr(g.a3), r4d_addr(g.ascale3),
                                (int)pb1, (int)pb2));
    return RAD_OK;
}
