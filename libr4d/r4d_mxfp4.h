// Shared device-side machinery for the OCP-MXFP4 GEMMs (r4d_gemm_mxfp4a8_nt_m64, _tiled, _decode):
// the folded-magnitude table and the eight-element unpack that reads it. One definition, because
// the three kernels must fold every weight element to the same e4m3 byte -- a model whose step
// crosses from one row's band into another's must not change what the weight is.
//
// r4d_gemm_mxfp4a8_nt_m64.hip explains the fold: d = Wref[n] - E8M0[n][blk] is pushed into the
// e4m3 value through this table, and 2^(Wref[n]-127) is applied once in the epilogue.
#pragma once

#include <hip/hip_runtime.h>

#define R4D_MXFP4_GROUP 32   // K per E8M0 exponent, fixed by the OCP MX format

// kMxfp4Mag[d] is the eight e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by 2^-d, as packed
// e4m3 bytes, low word then high word, indexed by the 3-bit magnitude through v_perm_b32.
// Every entry is the round-to-nearest-EVEN e4m3 of that product, which is exact for d <= 8 and a
// rounding for d >= 9 -- three entries there (d=9 mag 1.5, d=10 mag 3, d=11 mag 6) are the same
// exact tie 1.5*2^-9 and take the even neighbour 2^-8, not the smaller one. They matter little
// (such an element is 2.9e-3 of its row's reference), but the table is generated, not hand-rounded,
// so it agrees with the hardware's own conversion. d >= 13 is zero: below the row's range.
//
// Static: each kernel's code object carries its own copy, and no two define the same symbol.
static __device__ __constant__ unsigned int kMxfp4Mag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

// The exponent difference a block is folded at, clamped to the table: a block above its row's
// reference cannot occur (the reference is the row's maximum), and one 15 or more below is zero.
static __device__ __forceinline__ int r4d_mxfp4_fold_d(int ref, int e8) {
  const int d = ref - e8;
  return d < 0 ? 0 : (d > 15 ? 15 : d);
}

// Eight e2m1 codes (four packed bytes, element 2j in the low nibble of byte j) -> eight e4m3 bytes
// in k order, word 0 holding elements 0..3 and word 1 elements 4..7, with the block's exponent
// difference `d` already folded in. Splitting the nibbles gives the even and odd elements; each is
// mapped through the table by a v_perm_b32 indexed with the 3-bit magnitude, the sign rides along
// as bit 3 shifted into the e4m3 sign position, and two more v_perm_b32 interleave the halves.
//
// `_t` takes the table row (t0, t1) the caller already holds -- a kernel that copies the table to
// LDS reads it there, off the global load counter its weight and activation loads wait on.
static __device__ __forceinline__ void r4d_mxfp4_unpack8_t(unsigned int wv, unsigned int t0,
                                                           unsigned int t1, unsigned int* lo,
                                                           unsigned int* hi) {
  const unsigned int ev = wv & 0x0F0F0F0Fu;          // low nibble of each byte  -> even elements
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;   // high nibble             -> odd elements
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  *lo = __builtin_amdgcn_perm(bo, be, 0x05010400u);
  *hi = __builtin_amdgcn_perm(bo, be, 0x07030602u);
}
static __device__ __forceinline__ void r4d_mxfp4_unpack8(unsigned int wv, int d,
                                                         unsigned int* lo, unsigned int* hi) {
  r4d_mxfp4_unpack8_t(wv, kMxfp4Mag[d][0], kMxfp4Mag[d][1], lo, hi);
}

// THE ACTIVATION SCALE A ROW'S ACCUMULATOR IS CARRIED IN, for the per-(row, 128 of K) grid: the
// last non-zero entry of row `row` at or before block `kb`, or 1 when there is none. A block whose
// scale is zero has zero codes and adds nothing whichever scale the sum is carried in, so it keeps
// the previous one rather than dividing by zero.
static __device__ __forceinline__ float r4d_mxfp4_carried(const float* __restrict__ row, int kb) {
  for (int j = kb; j >= 0; --j) {
    const float s = row[j];
    if (s > 0.f) return s;
  }
  return 1.f;
}
