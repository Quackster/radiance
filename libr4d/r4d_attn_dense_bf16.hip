// R4D dense attention for an encoder: non-causal, varlen over cu_seqlens segments, grouped-query,
// with an optional two-sided window |i - j| < window. bf16 in and out, f16 operands, f32
// accumulate. Head widths 64, 256 and 512 are compiled; each (head_dim, gqa) pair is a row.
//
// The core is the vision kernel's (r4d_attn_vit_h72_bf16.hip): S^T = K.Q^T, so the wave32
// column-distributed accumulator gives each lane one query row, the softmax state is lane-private,
// and the C fragment of S^T is already the B fragment the PV matmul wants -- P never goes through
// LDS. What this file adds:
//
//   GQA IN THE WORKGROUP. A workgroup owns one KV head and every query head that reads it, so a
//   staged K/V tile is consumed by GQA times as many WMMAs as one head would give it. The grid is
//   (query blocks, kv heads, segments).
//
//   A WINDOW. A workgroup's key range is cut to the keys any of its queries can see, and the
//   per-element test runs only on tiles that straddle a window edge or the segment's end; an
//   interior tile pays nothing. Sliding layers at |i - j| <= 512 then cost O(T * window), not
//   O(T^2).
//
//   A HEAD SPLIT ACROSS WAVES (DSPLIT). One lane's output is head_dim / 2 floats, so head_dim 512
//   would need 256 accumulator VGPRs before the query fragments -- more than a lane has. At
//   DSPLIT 2 two waves own the same 16 queries and half the head each: each contracts its half
//   of q.k, the two partial scores are summed through LDS, both run the same softmax (a + b ==
//   b + a, so the two copies agree to the bit), and each accumulates its half of P.V.
//
// Every operand is read at its own token pitch, so q, k and v may be column slices of one fused
// projection; pitches and head offsets must be multiples of 8 elements for the 16-byte fetches.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#include <cmath>

template<int HD, int GQA, int NQB, int DSPLIT, int TILE>
__global__ __launch_bounds__(NQB * GQA * DSPLIT * 32)
void r4d_attn_dense_kernel(const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
                           const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
                           const int* __restrict__ cu, float scale, int window,
                           int q_ld, int k_ld, int v_ld, int o_ld)
{
    typedef DT16<1> D;
    typedef D::frag frag16;

    constexpr int NWARPS = NQB * GQA * DSPLIT;
    constexpr int NTHR   = NWARPS * 32;
    constexpr int BQ     = NQB * 16;            // query rows per head per workgroup
    constexpr int HDP    = HD / DSPLIT;         // the head dims one wave owns
    constexpr int NKS    = HDP / 16;            // its 16-wide k-tiles
    constexpr int MT     = TILE / 16;
    // Row strides padded by 8 shorts: 16-byte aligned, and off the bank period.
    constexpr int KSTR   = HD + 8;
    constexpr int VSTR   = TILE + 8;
    constexpr int DG     = HD / 8;              // eight-element dim groups in a row
    constexpr int KGRP   = TILE / 8;            // eight-key groups in a tile
    constexpr int KITEMS = TILE * DG;           // 16-byte K fetches a tile
    constexpr int KP     = (KITEMS + NTHR - 1) / NTHR;
    constexpr int NTR    = KGRP * DG;           // transposed V units, eight lanes each
    constexpr int LGRP   = NTHR / 8;
    constexpr int VP     = (NTR + LGRP - 1) / LGRP;
    constexpr float GROW = D::GROW;
    static_assert(HD % (16 * DSPLIT) == 0 && TILE % 16 == 0, "geometry");
    static_assert(DSPLIT == 1 || DSPLIT == 2, "a head is split across at most two waves");

    __shared__ __align__(16) uint16_t sK[TILE * KSTR];
    __shared__ __align__(16) uint16_t sV[HD * VSTR];
    __shared__ __align__(16) float sX[DSPLIT > 1 ? NWARPS * 32 * 8 : 4];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int dp  = warp % DSPLIT;
    const int g   = (warp / DSPLIT) % GQA;
    const int qbi = warp / (DSPLIT * GQA);
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const int hq  = kvh * GQA + g;

    const int s0 = cu[seq], slen = cu[seq + 1] - s0;
    const int q0 = blockIdx.x * BQ;
    if (slen <= 0 || q0 >= slen) return;
    const int qlast = min(q0 + BQ, slen) - 1;

    // The keys any query of this workgroup can see.
    int kb = 0, ke = slen;
    if (window > 0) {
        kb = max(0, q0 - (window - 1));
        ke = min(slen, qlast + window);
    }

    const float mul = scale * 1.44269504089f;
    const int  qpos = q0 + qbi * 16 + c;
    const bool live = qpos < slen;

    // ---- this wave's half (or whole) of its 16 query rows, scaled once -------------------------
    frag16 qf[NKS];
    {
        const int qrow = live ? qpos : slen - 1;
        const uint16_t* qp = gq + (long)(s0 + qrow) * q_ld + (long)hq * HD + dp * HDP;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint4 u = *(const uint4*)(qp + 16 * t + 8 * h);
            auto cv = [&](uint32_t w) {
                return D::pk(__builtin_bit_cast(float, w << 16) * mul,
                             __builtin_bit_cast(float, w & 0xffff0000u) * mul);
            };
            qf[t] = D::mk(make_uint2(cv(u.x), cv(u.y)), make_uint2(cv(u.z), cv(u.w)));
        }
    }

    float m_ref = -3.0e38f, soff = 3.0e38f, l_i = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    const int vj = tid & 7, lgi = tid >> 3;
    const uint16_t* kbase = gk + (long)kvh * HD;
    const uint16_t* vbase = gv + (long)kvh * HD;

    const int ntiles = (ke - kb + TILE - 1) / TILE;
    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = kb + ti * TILE;
        // Whether any (query, key) pair of this tile is outside the window or past the keys:
        // uniform over the workgroup, so the interior tiles skip the test entirely.
        const bool mask = (k0 + TILE > ke) ||
                          (window > 0 && (k0 < qlast - (window - 1) ||
                                          k0 + TILE - 1 > q0 + (window - 1)));

        uint4 kr[KP], vr[VP];
        #pragma unroll
        for (int p = 0; p < KP; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = i / DG, gg = i % DG;
                const int kk = (k0 + key < ke) ? k0 + key : ke - 1;
                kr[p] = *(const uint4*)(kbase + (long)(s0 + kk) * k_ld + gg * 8);
            }
        }
        // V transposed on the way in: the eight lanes of a group name eight key rows at the same
        // eight dims, and global_load_tr_b128 hands lane j dim d0+j across all eight.
        #pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                const int kk0 = k0 + kg * 8 + vj;
                const int kk = (kk0 < ke) ? kk0 : ke - 1;
                vr[r] = __builtin_bit_cast(uint4, load_tr_b128(vbase + (long)(s0 + kk) * v_ld +
                                                               dg * 8));
            }
        }

        lds_barrier();          // the previous tile's readers are done
        #pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                const uint4 w = vr[r];
                *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] =
                    make_uint4(D::from_bf16w(w.x), D::from_bf16w(w.y),
                               D::from_bf16w(w.z), D::from_bf16w(w.w));
            }
        }
        #pragma unroll
        for (int p = 0; p < KP; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const uint4 w = kr[p];
                *(uint4*)&sK[(i / DG) * KSTR + (i % DG) * 8] =
                    make_uint4(D::from_bf16w(w.x), D::from_bf16w(w.y),
                               D::from_bf16w(w.z), D::from_bf16w(w.w));
            }
        }
        lds_barrier();

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR + dp * HDP] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 kf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(krow + 16 * (t0 + j));
                            kf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) s = D::wmma(kf[j], qf[t0 + j], s);
                }
            }
            if (DSPLIT > 1) {
                // The other half of q.k, from the wave that owns it.
                float* mine = &sX[(warp * 32 + lane) * 8];
                const float* theirs = &sX[((warp ^ 1) * 32 + lane) * 8];
                *(float4*)mine       = make_float4(s[0], s[1], s[2], s[3]);
                *(float4*)(mine + 4) = make_float4(s[4], s[5], s[6], s[7]);
                lds_barrier();
                const float4 a = *(const float4*)theirs, b = *(const float4*)(theirs + 4);
                s[0] += a.x; s[1] += a.y; s[2] += a.z; s[3] += a.w;
                s[4] += b.x; s[5] += b.y; s[6] += b.z; s[7] += b.w;
                lds_barrier();
            }

            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = s[e];
            if (mask) {
                const int kq = k0 + mt * 16 + 8 * h;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int key = kq + e;
                    const int d = qpos - key;
                    const bool ok = key < ke && (window <= 0 || (d < window && -d < window));
                    sc[e] = ok ? sc[e] : -INFINITY;
                }
            }

            // The softmax against a reference max, rescaled only when the row max outgrows what
            // the f16 P can hold (r4d_dt16.h).
            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (wave_any(smax > m_ref + GROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int t = 0; t < NKS; ++t)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                soff  = -mnew;
            }

            uint32_t pw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2)
                pw[e >> 1] = D::pk(ex2(sc[e] + soff), ex2(sc[e + 1] + soff));
            // The row sum off the packed P, so the denominator is the values P.V used.
            const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
            const frag16 p16 = D::mk(make_uint2(pw[0], pw[1]), make_uint2(pw[2], pw[3]));

            {
                const uint16_t* vrow = &sV[(dp * HDP + c) * VSTR + mt * 16] + 8 * h;
                #pragma unroll
                for (int t0 = 0; t0 < NKS; t0 += 4) {
                    frag16 vf[4];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) {
                            const uint4 u = *(const uint4*)(vrow + (t0 + j) * 16 * VSTR);
                            vf[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        }
                    sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (t0 + j < NKS) acc[t0 + j] = D::wmma(vf[j], p16, acc[t0 + j]);
                }
            }
        }
    }

    if (!live) return;
    const float ls = l_i + swap16(l_i);
    const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
    uint16_t* op = go + (long)(s0 + qpos) * o_ld + (long)hq * HD + dp * HDP;
    #pragma unroll
    for (int t = 0; t < NKS; ++t) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[t][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[t][e + 1] * inv) << 16);
        *(uint4*)(op + t * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}

// ---- launch ----------------------------------------------------------------------------------
// One instantiation per (head_dim, gqa), at a workgroup shape picked for it:
//   h64  gqa1: 8 query blocks of 16, K/V tiles of 64 keys          (the vision towers' heads)
//   h256 gqa2: 8 query blocks x 2 heads, tiles of 32 (LDS: 37 KiB)
//   h512 gqa4: 2 query blocks x 4 heads x 2 head halves, tiles of 16 (LDS: 57 KiB)
template<int HD, int GQA, int NQB, int DSPLIT, int TILE>
static int r4d_attn_dense_launch(const void* q, const void* k, const void* v, void* o,
                                 const void* cu, int num_seqs, int max_seqlen, int n_head_kv,
                                 float scale, int window, int q_ld, int k_ld, int v_ld, int o_ld,
                                 hipStream_t s)
{
    constexpr int BQ = NQB * 16;
    dim3 grid((max_seqlen + BQ - 1) / BQ, n_head_kv, num_seqs);
    hipLaunchKernelGGL((r4d_attn_dense_kernel<HD, GQA, NQB, DSPLIT, TILE>), grid,
                       dim3(NQB * GQA * DSPLIT * 32), 0, s, (const uint16_t*)q, (const uint16_t*)k,
                       (const uint16_t*)v, (uint16_t*)o, (const int*)cu, scale, window, q_ld,
                       k_ld, v_ld, o_ld);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}

// q, o: [tokens, n_head * head_dim]; k, v: [tokens, n_head_kv * head_dim], each at its own token
// pitch in elements. cu_seqlens is [num_seqs + 1] int32 on the device; max_seqlen is a host-side
// bound on the segment lengths and sizes the grid. window 0 is the whole segment.
extern "C" int r4d_attn_dense_bf16(const void* q, const void* k, const void* v, void* o,
                                   const void* cu_seqlens, int num_seqs, int max_seqlen,
                                   int n_head, int n_head_kv, int head_dim, float scale,
                                   int window, int q_ld, int k_ld, int v_ld, int o_ld,
                                   void* stream)
{
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv) return -2;
    if ((q_ld | k_ld | v_ld | o_ld) & 7) return -3;
    if (window < 0) return -4;
    if (num_seqs <= 0 || max_seqlen <= 0) return 0;
    const int gqa = n_head / n_head_kv;
    hipStream_t s = (hipStream_t)stream;
    if (head_dim == 64 && gqa == 1)
        return r4d_attn_dense_launch<64, 1, 8, 1, 64>(q, k, v, o, cu_seqlens, num_seqs, max_seqlen,
                                                      n_head_kv, scale, window, q_ld, k_ld, v_ld,
                                                      o_ld, s);
    if (head_dim == 256 && gqa == 2)
        return r4d_attn_dense_launch<256, 2, 8, 1, 32>(q, k, v, o, cu_seqlens, num_seqs,
                                                       max_seqlen, n_head_kv, scale, window, q_ld,
                                                       k_ld, v_ld, o_ld, s);
    if (head_dim == 512 && gqa == 4)
        return r4d_attn_dense_launch<512, 4, 2, 2, 16>(q, k, v, o, cu_seqlens, num_seqs,
                                                       max_seqlen, n_head_kv, scale, window, q_ld,
                                                       k_ld, v_ld, o_ld, s);
    return -1;
}

// ============================================================== the plugin ABI entry point

enum { DN_Q = 0, DN_K, DN_V, DN_CU, DN_OUT };

extern "C" int r4d_rad_attn_dense(const RadArgs* a, RadStream s) {
    const RadTensor* q  = r4d_opt(a, DN_Q);
    const RadTensor* k  = r4d_opt(a, DN_K);
    const RadTensor* v  = r4d_opt(a, DN_V);
    const RadTensor* cu = r4d_opt(a, DN_CU);
    const RadTensor* o  = r4d_opt(a, DN_OUT);
    if (!q || !k || !v || !cu || !o) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_geti_or(a, "causal", 0) != 0) return R4D_NO(RAD_E_UNSUPPORTED);

    const long long head_dim = r4d_geti_or(a, "head_dim", 0);
    const long long gqa      = r4d_geti_or(a, "gqa", 1);
    if (head_dim <= 0 || gqa <= 0) return R4D_NO(RAD_E_SHAPE);
    /* [tokens, heads * head_dim] (or [tokens, heads, head_dim]) at the operand's own row pitch. */
    auto rows = [&](const RadTensor* t, long long* n_tok, long long* n_head, long long* ld) {
        long long width = 1;
        for (uint32_t i = 1; i < t->rank; ++i) width *= t->shape[i];
        if (t->rank < 2 || width % head_dim) return false;
        *n_head = width / head_dim;
        *ld = r4d_rows_pitch(t, width, n_tok);
        return *ld >= 0;
    };
    long long total = 0, nh = 0, q_ld = 0;
    long long tk = 0, hk = 0, k_ld = 0, tv = 0, hv = 0, v_ld = 0, to = 0, ho = 0, o_ld = 0;
    if (!rows(q, &total, &nh, &q_ld) || !rows(k, &tk, &hk, &k_ld) ||
        !rows(v, &tv, &hv, &v_ld) || !rows(o, &to, &ho, &o_ld))
        return R4D_NO(RAD_E_STRIDE);
    if (hk != hv || ho != nh || hk * gqa != nh) return R4D_NO(RAD_E_SHAPE);
    if (tk < total || tv < total || to < total) return R4D_NO(RAD_E_SHAPE);
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const int num_seqs = (int)(cu->shape[0] - 1);

    /* max_seqlen sizes the grid and is a bound, not a length: the kernel reads the device-side
     * cu_seqlens. Absent, the total token count is a valid bound. */
    long long msl = r4d_geti_or(a, "max_seqlen", 0);
    if (msl <= 0 || msl > total) msl = total;
    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)head_dim));
    const long long window = r4d_geti_or(a, "window", 0);

    const int e = r4d_attn_dense_bf16(q->data, k->data, v->data, o->data, cu->data, num_seqs,
                                      (int)msl, (int)nh, (int)hk, (int)head_dim, scale,
                                      (int)window, (int)q_ld, (int)k_ld, (int)v_ld, (int)o_ld, s);
    if (e == -1) return R4D_NO(RAD_E_SHAPE);
    return e == 0 ? RAD_OK : R4D_NO(e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
