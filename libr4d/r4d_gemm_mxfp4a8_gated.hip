// r4d_gemm_mxfp4a8_gated.hip - r4d_gemm_mxfp4a8_gated: gemm_nt_q_gated_mx, the OCP-MXFP4 x fp8 decode
// GEMM over a stacked [gate | up] weight with silu(gate) * up and the E4M3 quantiser in its epilogue.
//
// WHAT IT REPLACES.  gemm_mxfp4a8_decode over the [2n, K] gate_up plane (a bf16 [M, 2n] write), then
// gated_quant_fp8 over that plane (a read of it, the silu product, the per-128 amax, the codes and the
// scales).  Two launches, an ~8 us hole between them and a [M, 2n] round trip through memory.  Here the
// block that holds a gate fragment holds the matching up fragment, so the product is formed in the
// accumulator registers and only the codes (n bytes a row) and the scales leave the chip.
//
// THE TILE.  A block owns ONE 128-column scale group of the FUSED width: 8 waves, wave w streams gate
// fragment (b * 8 + w) and up fragment (b * 8 + w + n / 16) -- the weight stores the two halves
// stacked, gate rows [0, n) then up rows [n, 2n) -- and keeps two accumulator sets.  Both reduce over K
// in the order the decode kernel's unsplit form does (K slabs in order, 16 K a WMMA, the 128
// activation-scale boundary rescale included), so each accumulator is the SAME FLOAT the unfused GEMM
// holds before its bf16 store.  That is the whole bit-identity argument; it needs the unfused GEMM to
// be UNSPLIT (ks = 1), which the row's `N GE 14080` constraint states (the decode row's split rule gives
// ks = 1 from 110 column blocks of 128) and the tune file agrees with at the real shapes.
//
// THE EPILOGUE is gated_quant_fp8_kernel's arithmetic in its order: each accumulator -> *ref exponent
// (and the per-row activation scale if the grid is per row) -> bf16 (that is the GEMM's C store), silu
// in f32 on the bf16 gate, the product with the bf16 up value rounded to bf16, and the quantiser
// reading THAT.  A lane holds one column and 8 rows of a 16x16 accumulator, so a row's 16 columns of
// this wave are the 16 lanes of a half-wave (xor reduction, fmaxf: exact and order-free) and the other
// 112 are the 7 other waves (LDS).  The scale is amax * (1/448) or 1, the code is clamp(v * (1/scale))
// through v_cvt_pk_fp8_f32 -- quant_group's lines.
//
// NO PARTIAL, NO COUNTER, NO SCRATCH: split-K is not offered (the fold is the epilogue of a complete K
// range), so the row is stateless and a recorded pass replays with the arguments it recorded.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#define R4D_MXG_GROUP  32      // K per E8M0 exponent
#define R4D_MXG_DWN    8       // waves a block; the fused width a block owns is DWN * 16 = 128
#define R4D_MXG_MTILE  16
#define R4D_MXG_MAX_TM 4       // M <= 64, the decode band

typedef int      mxg_v2i __attribute__((ext_vector_type(2)));
typedef unsigned mxg_u2  __attribute__((ext_vector_type(2)));
typedef unsigned mxg_u4  __attribute__((ext_vector_type(4)));

// libr4d's folded-magnitude table, COPIED from r4d_gemm_mxfp4a8_decode.hip (and not regenerated): every
// MXFP4 kernel must fold an element to the same e4m3 byte.
__device__ __constant__ unsigned int kMxgMag[16][2] = {
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
static __device__ __forceinline__ T mxg_ld(const T* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

__device__ __forceinline__ float mxg_bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
// Round-to-nearest-even: the three lines of r4d_fused_quant_fp8.hip's f2bf, which is what the unfused
// pair's bf16 plane is written with (the GEMM's __hip_bfloat16 cast is the same rounding).
__device__ __forceinline__ uint16_t mxg_f2bf(float f) {
  const uint32_t u = __float_as_uint(f);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int DBK, int DTM, bool NT, bool ABLK>
__global__ __launch_bounds__(R4D_MXG_DWN * 32) void r4d_gemm_mxfp4a8_gated_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, unsigned char* __restrict__ Q, float* __restrict__ S,
    uint16_t* __restrict__ Y, int M, int K, int N, int as_rs, int q_ld, int s_ld, int y_ld) {
  constexpr int DWN = R4D_MXG_DWN, BND = DWN * 16, NTHR = DWN * 32;
  constexpr int ROWS = R4D_MXG_MTILE * DTM;
  constexpr int STR = DBK + 8;
  constexpr int WROWS = 2 * BND;                       // gate rows, then up rows
  __shared__ unsigned char sA[ROWS * STR];
  __shared__ unsigned char sW[WROWS * STR];
  __shared__ float sR[ABLK ? ROWS : 1];
  __shared__ float s_amx[DWN][ROWS];
  __shared__ float s_inv[ROWS];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int nff = N >> 1;                              // the fused width; the weight has 2 * nff rows
  const int c0 = blockIdx.x * BND;                     // first fused column of this block

  const int slabs = (K + DBK - 1) / DBK;

  v8f acc[DTM], accu[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i) { acc[i] = (v8f)(0.f); accu[i] = (v8f)(0.f); }

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

  for (int s = 0; s < slabs; ++s) {
    const int k0 = s * DBK;
    const bool bnd = ABLK && s > 0 && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd && tid < ROWS) {
        const int m = tid < M ? tid : M - 1;
        const int kb = k0 >> 7;
        const float sc = As[(size_t)m * as_rs + kb];
        sR[tid] = sc > 0.f ? carried(m, kb - 1) / sc : 1.f;
      }
    }
#pragma unroll
    for (int off = 0; off < ROWS * DBK; off += NTHR * 16) {
      const int idx = off + tid * 16;
      if (idx < ROWS * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        *(mxg_u4*)(&sA[r * STR + c]) = *(const mxg_u4*)(A + (size_t)rc * K + k0 + c);
      }
    }
    {
      constexpr int KSTEPS_T = DBK / 16, NTILES_T = WROWS / 16;
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += NTHR * 2) {
        const int sl = off + tid * 2;
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        // sW row r < BND is a gate row, r >= BND the matching up row, nff further down the plane.
        const int gc = (r < BND ? c0 + r : nff + c0 + (r - BND));
        const int lanec = (lane_ & 16) | (gc & 15);
        const mxg_u2 wv =
            mxg_ld<NT>((const mxg_u2*)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
        const int blk = (k0 + kst * 16) / R4D_MXG_GROUP;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMxgMag[d][0], t1 = kMxgMag[d][1];
          const unsigned int w = q ? wv[1] : wv[0];
          const unsigned int ev = w & 0x0F0F0F0Fu;
          const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(mxg_u2*)(&sW[(r + q) * STR + kloc]) =
              mxg_u2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
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
          for (int e = 0; e < 8; ++e) {
            const float r = sR[i * 16 + kb8 + e];
            acc[i][e] *= r;
            accu[i][e] *= r;
          }
          __builtin_amdgcn_sched_barrier(0);
        }
      }
    }
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      mxg_v2i af[DTM], wg, wu;
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        const unsigned char* pa = &sA[(i * 16 + col) * STR + kk];
        af[i][0] = *(const int*)pa; af[i][1] = *(const int*)(pa + 4);
      }
      const unsigned char* pg = &sW[(wave * 16 + col) * STR + kk];
      const unsigned char* pu = &sW[(BND + wave * 16 + col) * STR + kk];
      wg[0] = *(const int*)pg; wg[1] = *(const int*)(pg + 4);
      wu[0] = *(const int*)pu; wu[1] = *(const int*)(pu + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        acc[i]  = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wg, acc[i]);
        accu[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wu, accu[i]);
      }
    }
    __syncthreads();
  }

  if constexpr (ABLK) {
    if (tid < ROWS) {
      const int m = tid < M ? tid : M - 1;
      sR[tid] = carried(m, (K - 1) >> 7);
    }
    __syncthreads();
  }
  auto absval = [&](const v8f& v, int e, int i) -> float {
    return ABLK ? v[e] * sR[i * 16 + kb8 + e] : v[e];
  };

  // ---- the unfused pair's arithmetic, per element, then the per-(row, 128) amax
  const int nc = wave * 16 + col;                      // the column within the group
  const float rfg = __int_as_float((int)Wref[c0 + nc] << 23);
  const float rfu = __int_as_float((int)Wref[nff + c0 + nc] << 23);
  float v[DTM][8];
  uint16_t hb[DTM][8];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = i * 16 + kb8 + e;
      float val = 0.f;
      uint16_t h = 0;
      if (m < M) {
        const float asc = ABLK ? 1.f : As[m];
        const uint16_t gb = mxg_f2bf(absval(acc[i], e, i) * rfg * asc);
        const uint16_t ub = mxg_f2bf(absval(accu[i], e, i) * rfu * asc);
        const float g = mxg_bf2f(gb);
        const float a = g / (1.0f + expf(-g));
        h = mxg_f2bf(a * mxg_bf2f(ub));
        val = mxg_bf2f(h);
      }
      v[i][e] = val;
      hb[i][e] = h;
      float mx = fabsf(val);
      mx = r4d_xor_max<16>(mx);                        // the 16 columns of this wave, one row
      if (col == 0) s_amx[wave][m] = mx;
    }
  __syncthreads();
  if (tid < ROWS) {
    float mm = 0.f;
#pragma unroll
    for (int w = 0; w < DWN; ++w) mm = fmaxf(mm, s_amx[w][tid]);
    const float sc = mm > 0.0f ? mm * (1.0f / 448.0f) : 1.0f;
    s_inv[tid] = 1.0f / sc;
    if (tid < M) S[(size_t)tid * s_ld + blockIdx.x] = sc;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = i * 16 + kb8 + e;
      if (m >= M) continue;
      const float x = fminf(fmaxf(v[i][e] * s_inv[m], -448.0f), 448.0f);
      Q[(size_t)m * q_ld + c0 + nc] = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(x, x, 0, false) & 0xff);
      if (Y) Y[(size_t)m * y_ld + c0 + nc] = hb[i][e];
    }
}

extern "C" int r4d_gemm_mxfp4a8_gated_max_m(void) { return R4D_MXG_MTILE * R4D_MXG_MAX_TM; }

static void mxg_launch(long a, long ascale, long wq, long ws, long wref, long q, long sc, long y,
                       int M, int K, int N, int as_group, int as_rs, int q_ld, int s_ld, int y_ld,
                       int NT, long stream) {
  if (M < 1 || M > R4D_MXG_MTILE * R4D_MXG_MAX_TM)
    throw std::runtime_error("r4d gemm_mxfp4a8_gated: M must be 1..64");
  if (N < 256 || N % 256) throw std::runtime_error("r4d gemm_mxfp4a8_gated: N (both halves) must be a multiple of 256");
  if (K < 64 || K % 64) throw std::runtime_error("r4d gemm_mxfp4a8_gated: K must be a multiple of 64");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_gated: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  const int tm = (M + R4D_MXG_MTILE - 1) / R4D_MXG_MTILE;
  // The slab is a memory-layout choice (the K order does not depend on it): 64 where 128 would take
  // more than ~40 KB of LDS (DTM 4, K not a multiple of 128).
  const int bk = (tm == 4 || (K % 128)) ? 64 : 128;
  const bool nt = NT != 0;
  hipStream_t st = (hipStream_t)stream;
  dim3 grid((N / 2) / (R4D_MXG_DWN * 16)), block(R4D_MXG_DWN * 32);
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Qp = (unsigned char*)q;
  auto Scp = (float*)sc;
  auto Yp = (uint16_t*)y;
#define MXG_L(BK_, TM_, NT_, AB_)                                                              \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_gated_kernel<BK_, TM_, NT_, AB_>), grid, block, 0, st,   \
                     Ap, Sp, Wp, Zp, Rp, Qp, Scp, Yp, M, K, N, as_rs, q_ld, s_ld, y_ld)
#define MXG_AB(BK_, TM_, NT_) { if (ablk) MXG_L(BK_, TM_, NT_, true); else MXG_L(BK_, TM_, NT_, false); }
#define MXG_NT(BK_, TM_) { if (nt) MXG_AB(BK_, TM_, true) else MXG_AB(BK_, TM_, false) }
#define MXG_TM(BK_) { switch (tm) { case 1: MXG_NT(BK_, 1) break; case 2: MXG_NT(BK_, 2) break; \
                                    case 3: MXG_NT(BK_, 3) break; default: MXG_NT(BK_, 4) break; } }
  if (bk == 64) MXG_TM(64) else MXG_TM(128)
#undef MXG_TM
#undef MXG_NT
#undef MXG_AB
#undef MXG_L
}

// ============================================================== the plugin ABI entry point
// Operands (gemm_nt_q_gated_mx, oGemmNtQGatedMx in r4d_rows.cpp):
//   [0] a  [1] a_scale  [2] b  [3] b_scale  [4] q  [5] scale  [6] b_ref  [7] out_bf16?
// b and b_scale sit at gemm_nt_q's indices 2, 3 and b_ref at ITS 6, which is what r4d_rad_layout_mxfp4 keys on.
enum { MG_A = 0, MG_AS, MG_B, MG_BS, MG_Q, MG_S, MG_REF, MG_Y };

extern "C" int r4d_rad_gemm_mxfp4a8_gated(const RadArgs* a, RadStream s) {
  const RadTensor* x = r4d_opt(a, MG_A);
  const RadTensor* xs = r4d_opt(a, MG_AS);
  const RadTensor* w = r4d_opt(a, MG_B);
  const RadTensor* wsc = r4d_opt(a, MG_BS);
  const RadTensor* wr = r4d_opt(a, MG_REF);
  const RadTensor* q = r4d_opt(a, MG_Q);
  const RadTensor* sc = r4d_opt(a, MG_S);
  const RadTensor* y = r4d_opt(a, MG_Y);
  if (!x || !xs || !w || !wsc || !wr || !q || !sc) return R4D_NO(RAD_E_INVAL);
  if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
  if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
  const char* act = r4d_gets(a, "act");
  if (!act || std::strcmp(act, "silu")) return R4D_NO(RAD_E_UNSUPPORTED);

  const long long M = q->shape[0], nout = q->shape[1], K = x->shape[1];
  if (M < 1 || nout < 128 || nout % 128 || K < 64 || K % 64) return R4D_NO(RAD_E_SHAPE);
  if (M > r4d_gemm_mxfp4a8_gated_max_m()) return R4D_NO(RAD_E_SHAPE);
  if (x->shape[0] != M) return R4D_NO(RAD_E_SHAPE);
  if (sc->shape[0] != M || sc->shape[1] != nout / 128) return R4D_NO(RAD_E_SHAPE);
  if (y && (y->shape[0] != M || y->shape[1] != nout)) return R4D_NO(RAD_E_SHAPE);
  if (r4d_row_pitch(x) != K) return R4D_NO(RAD_E_STRIDE);       // the kernel reads A at pitch K
  const long long qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc), yp = y ? r4d_row_pitch(y) : 0;
  if (qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
  GemmOps g;
  std::memset(&g, 0, sizeof(g));
  g.ascale = xs; g.M = M; g.K = K;
  int as_group = 0, as_rs = 1;
  int rc = mx_ascale_grid(g, &as_group, &as_rs);
  if (rc != RAD_OK) return rc;
  R4D_TRY("gemm_mxfp4a8_gated",
      mxg_launch(r4d_addr(x), r4d_addr(xs), r4d_addr(w), r4d_addr(wsc), r4d_addr(wr),
                 r4d_addr(q), r4d_addr(sc), y ? r4d_addr(y) : 0, (int)M, (int)K, (int)(2 * nout),
                 as_group, as_rs, (int)qp, (int)sp, (int)yp, 0, (long)(uintptr_t)s));
  return RAD_OK;
}
