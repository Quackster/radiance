// r4d_fp8_lm.h -- the STORED ROW GEOMETRY of the fp8 lm_head, and nothing else. Pure C++: no HIP,
// no r4d.h, so the host layout hook (r4d_fp8_layout.cpp), the device entry point
// (r4d_gemm_fp8a16_nt_m1.hip) and the selftest all derive the row width from the same line instead
// of writing `K + 2 * (K / 128)` four times and drifting.
//
// WHY THE ROW LOOKS LIKE THIS AT ALL is r4d_rad_layout_fp8_logits' argument and is not repeated
// here: `logits_gemm` carries ONE weight operand, so the block scales have nowhere to go but the
// row's own tail, and that is what keeps a row-sharded lm_head a plain byte range under tensor
// parallelism.
//
// ---- THE PADDING, which is the only thing this header adds to `K + 2*nkb` ---------------------
//
// Both kernels that read this plane -- the M <= 2 matvec and the WMMA tile above it -- load the
// weight SIXTEEN BYTES AT A TIME through a uint4, at `row * w_ld + (a multiple of 16)`. So the row
// pitch has to be a multiple of 16 or every odd row is misaligned, and the unpadded width is a
// multiple of 16 only when
//
//     K + 2*(K/128) = 130m  (K = 128m)   is 0 mod 16   <=>   m is 0 mod 8   <=>   K is 0 mod 1024
//
// which is a property of the MODEL, not of anything the kernel chose. K = 5120 has it; K = 2560
// does not. Without the pad such a head cannot be served in fp8 at all: it falls to the bf16
// projection, which reads twice the bytes and is then the single largest op in the step -- for
// want of eight bytes a row.
//
// So pad. The cost is under 16 bytes on a row that is already K + K/64 long -- 0.3% at K = 2560,
// and ZERO at every K that is a multiple of 1024, where the rounding is a no-op and the plane is
// byte-for-byte the unpadded one. **The padding never changes a valid container's plane**, and
// that is not luck: the layout hook refuses a K without the property rather than writing a plane
// its own kernels would fault on, so every K a container can already hold is one the rounding
// leaves alone.
//
// This is NOT the padding r4d_gemm_fp8a16_nt_m1.hip's header prices and rejects. That one is about
// padding an ALREADY-ALIGNED 5200-byte row out to 5376 to make it 128-aligned, chasing the
// cacheline straddle, and it loses to FK = 256. This is 16-byte alignment, which is a correctness
// floor rather than a tuning knob, and it is what decides whether the row can be served at all.
#pragma once
#include <cstdint>

namespace r4d_fp8 {

// BYTES one stored lm_head row occupies: K E4M3 codes, then K/128 bf16 block scales, then the
// alignment tail. K must be a multiple of 128 (the scale block); the caller checks that.
inline int64_t lm_row_bytes(int64_t K) {
    const int64_t used = K + 2 * (K / 128);
    return (used + 15) & ~(int64_t)15;
}

// ...and where the pad starts, which is the only part of the row nothing reads. The transform
// zeroes it so that a container's bytes are a function of its weights and not of what the
// allocator happened to leave there.
inline int64_t lm_row_used(int64_t K) { return K + 2 * (K / 128); }

// THE INT4 HEAD'S ROW (logits_gemm_i4): the same row at half the code bytes -- K/2 bytes of
// two's-complement nibbles, element 2j in the low nibble of byte j (rad_store_bits' order), then the
// row's own nkb bf16 scales, then the pad to 16. At K = 5120 that is 2640 bytes against 5200.
inline int64_t lm4_row_bytes(int64_t K) {
    const int64_t used = K / 2 + 2 * (K / 128);
    return (used + 15) & ~(int64_t)15;
}

// ...AND ITS ASYMMETRIC FORM (logits_gemm_u4g64 / _u4g128): unsigned nibbles, then the row's bf16
// scale a group, then its u8 zero a group, then the pad -- (code - zero) * scale, libquant's
// `codes=u4 zero=u8`. At K = 5120 and group 64 that is 2560 + 160 + 80 = 2800 bytes.
inline int64_t lm4a_row_bytes(int64_t K, int64_t group) {
    const int64_t ng = K / group;
    const int64_t used = K / 2 + 3 * ng;
    return (used + 15) & ~(int64_t)15;
}

}  // namespace r4d_fp8
