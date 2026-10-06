/* r4d_fp8_layout.cpp -- what libr4d's FP8 kernels store for a weight, and the relayouts that
 * produce it from the weight's canonical planes (spec.md §4.3). Pure host C++: nothing here
 * includes r4d.h or <hip/hip_runtime.h>, so it compiles and can be exercised on a box with no
 * ROCm, which is where a packing bug is cheapest to find.
 *
 * ================================================================= what they read
 *
 * The block-scaled linears read the encoding a block-FP8 checkpoint ships, `fp8_e4m3*bf16[128x128]`:
 *
 *   codes   F8_E4M3, [N][K]                       -- exactly as safetensors stores `X.weight`
 *   scale   BF16,    [ceil(N/128)][ceil(K/128)]   -- exactly `X.weight_scale_inv`
 *
 * In a Qwen3.8-27B-FP8 checkpoint, `mlp.gate_proj.weight` is F8_E4M3 [17408, 5120] with
 * `mlp.gate_proj.weight_scale_inv` BF16 [136, 40], and 17408/136 = 5120/40 = 128. The kernels
 * index `S[(row / 128) * sc_cols + (c / 128)]`, which is that grid, so the scale plane is stored as
 * it is and only the codes move: into WMMA fragment order for the fp8a8 GEMMs, which read a lane's
 * own bytes straight from a coalesced load, and nowhere at all for the fp8a16 matvec, which reads
 * sixteen consecutive k of one row. The two cannot share a stored form, so a weight is declared
 * for one or the other and the builder refuses the pair.
 *
 * The lm_head carries each row's block scales in the row's own tail, because it has one operand
 * and is ROW-sharded: [K codes][K/128 bf16 scales], the row padded to 16 bytes so a uint4 load of
 * the next row is aligned (r4d_fp8_lm.h). hc_read's mixing matrices and mtp_enter's fc are the
 * same with an f32 scale per group of the row (r4d_hc_fp8.h).
 *
 * ================================================================= BF16 scales, not UE8M0
 *
 * Some block-FP8 checkpoints carry a UE8M0 exponent BYTE per tile rather than a bf16. Converting
 * to it is not free: UE8M0 stores an exponent and nothing else, so rounding a bf16 scale into it
 * changes the tile's gain, and correcting for that means requantising every E4M3 byte against the
 * new power-of-two scale. And nothing is gained -- decoding either is one shift and one bitcast.
 * The bf16 plane is 0.02% of its weight.
 *
 * Choosing the numbers is not this file's job: a checkpoint's own planes are served as they are,
 * and anything quantised -- a bf16 lm_head made fp8 -- was quantised by a quantiser before the
 * container was written (libquant's `rtn` over `fp8_e4m3` with a 128x128 bf16 block, whose rule is
 * the one these kernels' activation quantiser uses: amax/448 rounded to bf16, a zero block scaled
 * by one).
 */

#include "r4d_plugin.h"
#include "r4d_fp8_frag.h"
#include "r4d_fp8_lm.h"
#include "r4d_hc_fp8.h"
#include "r4d_enc.h"

#include <cmath>
#include <cstdint>
#include <cstring>

/* Operand indices for `gemm_nt_q`. Named rather than literal because the schema is positional and a
 * reorder in r4d_rows.cpp is otherwise a silent miscompile here. */
enum { FL_A = 0, FL_ASCALE = 1, FL_B = 2, FL_BSCALE = 3, FL_Y = 4 };

/* The k a fragment-order tile spans, which libr4d compiles in. */
enum { R4D_FP8_SUPER_K = r4d_fp8::kSuperK };

namespace {

/* THE WALK BOTH DIRECTIONS SHARE: the fragment-order plane front to back, eight bytes at a time,
 * each with the row-major offset of the [N][K] weight it holds. A fragment is eight consecutive k
 * of one row -- r4d_fp8_frag.h's frag_src over j = 0..7 -- so it moves as one word, and the
 * stored side is written or read in order. A byte at a time, with the divisions that place it,
 * the same permutation is most of the CPU a block-FP8 model's load spends. */
template <typename Move>
void frag_walk(long long N, long long K, Move move) {
    constexpr int kLanes = r4d_fp8::kTileBytes / r4d_fp8::kLaneBytes;
    constexpr int kFrags = r4d_fp8::kLaneBytes / 8;
    const long long ksuper = K / r4d_fp8::kSuperK;
    int64_t at = 0;
    for (long long rt = 0; rt < N / r4d_fp8::kFragRows; ++rt)
        for (long long ks = 0; ks < ksuper; ++ks)
            for (int lane = 0; lane < kLanes; ++lane)
                for (int t = 0; t < kFrags; ++t, at += 8) {
                    int64_t row, k;
                    r4d_fp8::frag_src(rt, ks, lane, t, 0, &row, &k);
                    move(at, row * K + k);
                }
}

struct FGeom { long long N, K, sn, sk; };

int fgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "N", 0);
    g->K = r4d_pgeti_or(p, n_p, "K", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    /* The block is compiled into the kernels as `>> 7`. A geometry that asks for another group is
     * asking for a format nothing here reads; the constraint table already refused it, and this is
     * the second line of defence for a tool calling the hook without going through selection. */
    if (r4d_pgeti_or(p, n_p, "group", R4D_FP8_BLOCK) != R4D_FP8_BLOCK) return RAD_E_SHAPE;
    g->sn = (g->N + R4D_FP8_BLOCK - 1) / R4D_FP8_BLOCK;
    g->sk = (g->K + R4D_FP8_BLOCK - 1) / R4D_FP8_BLOCK;
    return RAD_OK;
}

void ffill(RadLayout* out, const char* tag, uint32_t dt, long long d0, long long d1,
           long long bytes) {
    std::memset(out, 0, sizeof(*out));
    out->tag = tag;
    out->dtype = dt;
    out->rank = 2;
    out->shape[0] = d0;
    out->shape[1] = d1;
    out->align = 16;   /* 16-byte loads; see the padding note in r4d_fp8_lm.h */
    out->bytes = bytes;
}


/* `logits_gemm` names its geometry `n_vocab` and `n_embd` where the GEMMs say N and K. */
int lgeom(const RadParam* p, int n_p, FGeom* g) {
    g->N = r4d_pgeti_or(p, n_p, "n_vocab", 0);
    g->K = r4d_pgeti_or(p, n_p, "n_embd", 0);
    if (g->N <= 0 || g->K <= 0) return RAD_E_SHAPE;
    if (g->K % R4D_FP8_BLOCK) return RAD_E_SHAPE;
    g->sn = (g->N + R4D_FP8_BLOCK - 1) / R4D_FP8_BLOCK;
    g->sk = g->K / R4D_FP8_BLOCK;
    return RAD_OK;
}

/* Operand 1 is the weight; see oLogits in r4d_rows.cpp. */
enum { LGL_X = 0, LGL_W = 1, LGL_Y = 2 };

/* Operand indices of hc_read -- see oHcRead in r4d_rows.cpp. */
enum { HCL_DOWN = 2, HCL_UP = 3 };

struct HGeom { long long rows, K, G; };

int hgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    const long long n  = r4d_pgeti_or(p, n_p, "n", 0);
    const long long hc = r4d_pgeti_or(p, n_p, "hc", 0);
    const long long lr = r4d_pgeti_or(p, n_p, "lowrank", 0);
    if (n <= 0 || hc <= 0 || lr <= 0) return RAD_E_SHAPE;
    if (operand == HCL_DOWN)    { g->rows = lr;     g->K = hc * n; g->G = r4d_hc8::down_group(g->K); }
    else if (operand == HCL_UP) { g->rows = hc * n; g->K = lr;     g->G = r4d_hc8::up_group(g->K); }
    else return RAD_E_UNSUPPORTED;
    if (!r4d_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

enum { MTL_FCH = 4, MTL_FCE = 5 };   /* oMtpEnter in r4d_rows.cpp */

int mgeom(const RadParam* p, int n_p, int operand, HGeom* g) {
    if (operand != MTL_FCH && operand != MTL_FCE) return RAD_E_UNSUPPORTED;
    const long long n = r4d_pgeti_or(p, n_p, "n", 0);
    if (n <= 0) return RAD_E_SHAPE;
    g->rows = n; g->K = n; g->G = r4d_hc8::kDownGroup;
    if (!r4d_hc8::group_ok(g->K, g->G)) return RAD_E_SHAPE;
    return RAD_OK;
}

}  /* namespace */

/* ==================================================================== the hooks
 *
 * Each checks the encoding its kernel reads and refuses any other with RAD_E_DTYPE, which is the
 * declare-time error a mis-converted container meets; then it moves bits and chooses none. */

namespace {

/* gemm_nt_q's weight and scale over block fp8: e4m3 codes with a bf16 scale a 128x128 block. */
bool fp8_block_enc(const RadEncoding* e) {
    return r4d_enc::affine(e, RAD_F8E4M3, RAD_BF16, R4D_FP8_BLOCK, R4D_FP8_BLOCK);
}

int fp8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* pl, int n, RadLayout* out, bool frag) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    if (!fp8_block_enc(enc)) return RAD_E_DTYPE;
    if (operand == FL_BSCALE) {
        if (!r4d_enc::selects(enc, sel, n, "scale") || !r4d_enc::dims(pl[0], g.sn, g.sk))
            return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;                      /* the checkpoint's own bf16 plane */
    }
    if (!r4d_enc::selects(enc, sel, n, "codes") || !r4d_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    if (!frag) return RAD_E_UNSUPPORTED;               /* row-major, as the matvec reads it */
    if (g.K % R4D_FP8_SUPER_K || g.N % r4d_fp8::kFragRows) return RAD_E_SHAPE;
    ffill(out, "r4d.fp8.w.frag64", RAD_F8E4M3, g.N, g.K, r4d_fp8::frag_bytes(g.N, g.K));
    return RAD_OK;
}

}  /* namespace */

extern "C" int r4d_rad_layout_fp8_block(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* pl, int n, RadLayout* out) {
    return fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, true);
}

extern "C" int r4d_rad_layout_fp8_rowmajor(const RadParam* p, int n_p, int operand,
                                               const RadEncoding* enc, const int* sel,
                                               const RadTensor* pl, int n, RadLayout* out) {
    return fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, out, false);
}

extern "C" int r4d_rad_relayout_fp8_block(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    const int rc = fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, true);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)pl[0].data;
    uint8_t* d = (uint8_t*)dst;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + at, s + from, 8); });
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_fp8_block(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = fp8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, true);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + from, s + at, 8); });
    return RAD_OK;
}

/* ---- the int8 twin: i8 codes in the same fragment order, with a bf16 scale per ROW per 128 K.
 * The codes move exactly as E4M3 codes do -- the permutation does not read them.
 *
 * THE SCALE PLANE IS TILE-MAJOR, [N/16][K/128][16]: the sixteen rows of a fragment tile side by side
 * for one 128-K block. On the narrow path a lane's eight accumulator rows are eight CONSECUTIVE
 * rows of one tile, so their scales are one 16-byte load in the fold where the row-major plane
 * would need eight scattered ones -- which, if-converted into every k step, cost the decode
 * kernel 70% at M = 4. The prefill tile reads one row's scale a lane, from either form. */
namespace {
int i8_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    FGeom g;
    const int rc = fgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (operand != FL_B && operand != FL_BSCALE) return RAD_E_UNSUPPORTED;
    if (!r4d_enc::affine(enc, RAD_I8, RAD_BF16, 1, R4D_FP8_BLOCK)) return RAD_E_DTYPE;
    if (operand == FL_BSCALE) {
        if (!r4d_enc::selects(enc, sel, n, "scale") || !r4d_enc::dims(pl[0], g.N, g.sk))
            return RAD_E_SHAPE;
        if (g.N % r4d_fp8::kFragRows) return RAD_E_SHAPE;
        ffill(out, "r4d.i8.s.t16", RAD_BF16, g.N, g.sk, g.N * g.sk * 2);
        return RAD_OK;
    }
    if (!r4d_enc::selects(enc, sel, n, "codes") || !r4d_enc::dims(pl[0], g.N, g.K))
        return RAD_E_SHAPE;
    if (g.K % R4D_FP8_SUPER_K || g.N % r4d_fp8::kFragRows) return RAD_E_SHAPE;
    ffill(out, "r4d.i8.w.frag64", RAD_I8, g.N, g.K, r4d_fp8::frag_bytes(g.N, g.K));
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_i8_block(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    return i8_enc_layout(p, n_p, operand, enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_i8_block(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        const long long N = pl[0].shape[0], nkb = pl[0].shape[1];
        const uint16_t* s = (const uint16_t*)pl[0].data;
        uint16_t* d = (uint16_t*)dst;
        for (long long r = 0; r < N; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                d[((r / 16) * nkb + kb) * 16 + r % 16] = s[r * nkb + kb];
        return RAD_OK;
    }
    const uint8_t* s = (const uint8_t*)pl[0].data;
    uint8_t* d = (uint8_t*)dst;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + at, s + from, 8); });
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_i8_block(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    RadLayout L;
    const int rc = i8_enc_layout(p, n_p, operand, enc, sel, pl, n, &L);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data) return RAD_E_SHAPE;
    if (operand == FL_BSCALE) {
        const long long N = pl[0].shape[0], nkb = pl[0].shape[1];
        const uint16_t* s = (const uint16_t*)src;
        uint16_t* d = (uint16_t*)pl[0].data;
        for (long long r = 0; r < N; ++r)
            for (long long kb = 0; kb < nkb; ++kb)
                d[r * nkb + kb] = s[((r / 16) * nkb + kb) * 16 + r % 16];
        return RAD_OK;
    }
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    frag_walk(pl[0].shape[0], pl[0].shape[1],
              [&](int64_t at, int64_t from) { std::memcpy(d + from, s + at, 8); });
    return RAD_OK;
}

/* ---- the lm_head: codes and scales in one row, the scales of the row's 128-row block repeated
 * into its tail. */
namespace {
/* `i8`: the int8 head (logits_gemm_i8) -- i8 codes with a bf16 scale a row per 128 columns, the
 * same stored row with each row's OWN scales in its tail where the E4M3 head repeats its 128-row
 * block's. */
int lm_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, FGeom* gout, bool i8 = false) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != LGL_W) return RAD_E_UNSUPPORTED;
    FGeom g;
    const int rc = lgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (i8 ? !r4d_enc::affine(enc, RAD_I8, RAD_BF16, 1, R4D_FP8_BLOCK) : !fp8_block_enc(enc))
        return RAD_E_DTYPE;
    if (!r4d_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], g.N, g.K) || !r4d_enc::dims(pl[1], i8 ? g.N : g.sn, g.sk))
        return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm_row_bytes(g.K);
    ffill(out, i8 ? "r4d.i8lm.r128" : "r4d.fp8lm.b128", i8 ? RAD_I8 : RAD_F8E4M3, g.N, stride,
          g.N * stride);
    if (gout) *gout = g;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_fp8_logits(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const RadTensor* pl, int n, RadLayout* out) {
    return lm_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_fp8_logits(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, void* dst,
                                           int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm_row_bytes(g.K);
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + (r / R4D_FP8_BLOCK) * g.sk, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_fp8_logits(const RadParam* p, int n_p, int operand,
                                             const RadEncoding* enc, const int* sel,
                                             const void* src, int64_t src_bytes,
                                             const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm_row_bytes(g.K);
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    uint16_t* sc = (uint16_t*)pl[1].data;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy(codes + r * g.K, s + r * stride, (size_t)g.K);
        if (r % R4D_FP8_BLOCK == 0)
            std::memcpy(sc + (r / R4D_FP8_BLOCK) * g.sk, s + r * stride + g.K, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int r4d_rad_layout_i8_logits(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return lm_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, true);
}

extern "C" int r4d_rad_relayout_i8_logits(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, true);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm_row_bytes(g.K);
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * g.sk, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_i8_logits(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, true);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm_row_bytes(g.K);
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    uint16_t* sc = (uint16_t*)pl[1].data;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy(codes + r * g.K, s + r * stride, (size_t)g.K);
        std::memcpy(sc + r * g.sk, s + r * stride + g.K, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

/* ---- the INT4 head (logits_gemm_i4): i4 codes with a bf16 scale a row per 128 columns, stored
 * as r4d_fp8::lm4_row_bytes' row -- the K/2 code bytes as libquant packed them (low nibble
 * first), the row's own scales, the pad -- and handed to the kernel as BYTES. */
namespace {
int lm4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                   const int* sel, const RadTensor* pl, int n, RadLayout* out, FGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != LGL_W) return RAD_E_UNSUPPORTED;
    FGeom g;
    const int rc = lgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (!r4d_enc::affine(enc, RAD_I4, RAD_BF16, 1, R4D_FP8_BLOCK)) return RAD_E_DTYPE;
    if (!r4d_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (g.K % 2 || !r4d_enc::dims(pl[0], g.N, g.K) || !r4d_enc::dims(pl[1], g.N, g.sk))
        return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4_row_bytes(g.K);
    ffill(out, "r4d.i4lm.r128", RAD_U8, g.N, stride, g.N * stride);
    if (gout) *gout = g;
    return RAD_OK;
}
}  /* namespace */

extern "C" int r4d_rad_layout_i4_logits(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    return lm4_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r4d_rad_relayout_i4_logits(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, void* dst,
                                          int64_t dst_bytes) {
    RadLayout L;
    FGeom g;
    const int rc = lm4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4_row_bytes(g.K), cb = g.K / 2;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const uint16_t* sc = (const uint16_t*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * cb, (size_t)cb);
        std::memcpy(row + cb, sc + r * g.sk, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

extern "C" int r4d_rad_unrelayout_i4_logits(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const void* src, int64_t src_bytes,
                                            const RadTensor* pl, int n) {
    RadLayout L;
    FGeom g;
    const int rc = lm4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4_row_bytes(g.K), cb = g.K / 2;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    uint16_t* sc = (uint16_t*)pl[1].data;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy(codes + r * cb, s + r * stride, (size_t)cb);
        std::memcpy(sc + r * g.sk, s + r * stride + cb, (size_t)(2 * g.sk));
    }
    return RAD_OK;
}

/* ---- the ASYMMETRIC int4 heads (logits_gemm_u4g64 / _u4g128): libquant's `codes=u4 zero=u8
 * group=G scale=bf16`, stored as r4d_fp8::lm4a_row_bytes' row -- the K/2 code bytes, the row's
 * bf16 scale a group, its u8 zero a group, the pad. The planes are found by ROLE, three of them. */
namespace {
int lm4a_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                    const int* sel, const RadTensor* pl, int n, RadLayout* out, FGeom* gout,
                    int64_t group, int* ic, int* is, int* iz) {
    if (!out || !pl) return RAD_E_INVAL;
    if (operand != LGL_W) return RAD_E_UNSUPPORTED;
    FGeom g;
    const int rc = lgeom(p, n_p, &g);
    if (rc != RAD_OK) return rc;
    if (!r4d_enc::affine(enc, RAD_U4, RAD_BF16, 1, group, RAD_U8)) return RAD_E_DTYPE;
    if (n != 3) return RAD_E_SHAPE;
    *ic = *is = *iz = -1;
    for (int i = 0; i < 3; ++i) {
        const char* r = r4d_enc::role(enc, sel, i);
        if (!std::strcmp(r, "codes")) *ic = i;
        else if (!std::strcmp(r, "scale")) *is = i;
        else if (!std::strcmp(r, "zero")) *iz = i;
    }
    if (*ic < 0 || *is < 0 || *iz < 0) return RAD_E_SHAPE;
    const int64_t ng = g.K / group;
    if (g.K % group || !r4d_enc::dims(pl[*ic], g.N, g.K) || !r4d_enc::dims(pl[*is], g.N, ng) ||
        !r4d_enc::dims(pl[*iz], g.N, ng))
        return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4a_row_bytes(g.K, group);
    ffill(out, group == 64 ? "r4d.u4lm.g64" : "r4d.u4lm.g128", RAD_U8, g.N, stride, g.N * stride);
    if (gout) *gout = g;
    return RAD_OK;
}
int lm4a_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, void* dst, int64_t dst_bytes, int64_t group) {
    RadLayout L;
    FGeom g;
    int ic, is, iz;
    const int rc = lm4a_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, group, &ic, &is, &iz);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes || !pl[ic].data || !pl[is].data || !pl[iz].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4a_row_bytes(g.K, group), cb = g.K / 2, ng = g.K / group;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.N; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, (const uint8_t*)pl[ic].data + r * cb, (size_t)cb);
        std::memcpy(row + cb, (const uint16_t*)pl[is].data + r * ng, (size_t)(2 * ng));
        std::memcpy(row + cb + 2 * ng, (const uint8_t*)pl[iz].data + r * ng, (size_t)ng);
    }
    return RAD_OK;
}
int lm4a_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                    const int* sel, const void* src, int64_t src_bytes, const RadTensor* pl, int n,
                    int64_t group) {
    RadLayout L;
    FGeom g;
    int ic, is, iz;
    const int rc = lm4a_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g, group, &ic, &is, &iz);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes || !pl[ic].data || !pl[is].data || !pl[iz].data) return RAD_E_SHAPE;
    const long long stride = r4d_fp8::lm4a_row_bytes(g.K, group), cb = g.K / 2, ng = g.K / group;
    const uint8_t* s = (const uint8_t*)src;
    for (long long r = 0; r < g.N; ++r) {
        std::memcpy((uint8_t*)pl[ic].data + r * cb, s + r * stride, (size_t)cb);
        std::memcpy((uint16_t*)pl[is].data + r * ng, s + r * stride + cb, (size_t)(2 * ng));
        std::memcpy((uint8_t*)pl[iz].data + r * ng, s + r * stride + cb + 2 * ng, (size_t)ng);
    }
    return RAD_OK;
}
}  /* namespace */

#define R4D_U4LM_HOOKS(G)                                                                       \
extern "C" int r4d_rad_layout_u4g##G##_logits(const RadParam* p, int n_p, int operand,         \
                                              const RadEncoding* enc, const int* sel,          \
                                              const RadTensor* pl, int n, RadLayout* out) {    \
    int ic, is, iz;                                                                             \
    return lm4a_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr, G, &ic, &is, &iz);  \
}                                                                                               \
extern "C" int r4d_rad_relayout_u4g##G##_logits(const RadParam* p, int n_p, int operand,       \
                                                const RadEncoding* enc, const int* sel,        \
                                                const RadTensor* pl, int n, void* dst,         \
                                                int64_t dst_bytes) {                           \
    return lm4a_relayout(p, n_p, operand, enc, sel, pl, n, dst, dst_bytes, G);                 \
}                                                                                               \
extern "C" int r4d_rad_unrelayout_u4g##G##_logits(const RadParam* p, int n_p, int operand,     \
                                                  const RadEncoding* enc, const int* sel,      \
                                                  const void* src, int64_t src_bytes,          \
                                                  const RadTensor* pl, int n) {                \
    return lm4a_unrelayout(p, n_p, operand, enc, sel, src, src_bytes, pl, n, G);               \
}
R4D_U4LM_HOOKS(64)
R4D_U4LM_HOOKS(128)
#undef R4D_U4LM_HOOKS

/* ---- hc_read's mixing matrices and mtp_enter's fc: e4m3 codes with an f32 scale a group of the
 * row, the scales in the row's tail and the row padded to 16 bytes. */
namespace {
int hcrows_enc_layout(const HGeom& g, const char* tag, const RadEncoding* enc, const int* sel,
                      const RadTensor* pl, int n, RadLayout* out) {
    if (!out || !pl) return RAD_E_INVAL;
    if (!r4d_enc::affine(enc, RAD_F8E4M3, RAD_F32, 1, g.G)) return RAD_E_DTYPE;
    if (!r4d_enc::selects(enc, sel, n, "codes", "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], g.rows, g.K) || !r4d_enc::dims(pl[1], g.rows, g.K / g.G))
        return RAD_E_SHAPE;
    const long long stride = r4d_hc8::row_bytes(g.K, g.G);
    ffill(out, tag, RAD_F8E4M3, g.rows, stride, g.rows * stride);
    return RAD_OK;
}

int hcrows_relayout(const HGeom& g, const RadTensor* pl, void* dst, int64_t dst_bytes) {
    const long long stride = r4d_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (dst_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    std::memset(d, 0, (size_t)dst_bytes);
    for (long long r = 0; r < g.rows; ++r) {
        std::memcpy(d + r * stride, codes + r * g.K, (size_t)g.K);
        std::memcpy(d + r * stride + g.K, sc + r * ng, (size_t)(4 * ng));
    }
    return RAD_OK;
}

int hcrows_unrelayout(const HGeom& g, const void* src, int64_t src_bytes, const RadTensor* pl) {
    const long long stride = r4d_hc8::row_bytes(g.K, g.G), ng = g.K / g.G;
    if (src_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    for (long long r = 0; r < g.rows; ++r) {
        std::memcpy((uint8_t*)pl[0].data + r * g.K, s + r * stride, (size_t)g.K);
        std::memcpy((float*)pl[1].data + r * ng, s + r * stride + g.K, (size_t)(4 * ng));
    }
    return RAD_OK;
}

const char* hc_tag(int operand) {
    return operand == HCL_DOWN ? "r4d.hc.e4m3.g128" : "r4d.hc.e4m3.q4";
}
}  /* namespace */

extern "C" int r4d_rad_layout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel,
                                          const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = hgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int r4d_rad_unrelayout_hc_e4m3(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel, const void* src,
                                          int64_t src_bytes, const RadTensor* pl, int n) {
    HGeom g;
    RadLayout L;
    int rc = hgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, hc_tag(operand), enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_unrelayout(g, src, src_bytes, pl);
}

extern "C" int r4d_rad_layout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const RadTensor* pl, int n, RadLayout* out) {
    HGeom g;
    const int rc = mgeom(p, n_p, operand, &g);
    if (rc != RAD_OK) return rc;
    return hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, out);
}

extern "C" int r4d_rad_relayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst, int64_t dst_bytes) {
    HGeom g;
    RadLayout L;
    int rc = mgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_relayout(g, pl, dst, dst_bytes);
}

extern "C" int r4d_rad_unrelayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    HGeom g;
    RadLayout L;
    int rc = mgeom(p, n_p, operand, &g);
    if (rc == RAD_OK) rc = hcrows_enc_layout(g, "r4d.mtp.e4m3.g128", enc, sel, pl, n, &L);
    return rc != RAD_OK ? rc : hcrows_unrelayout(g, src, src_bytes, pl);
}

/* ================================================================= embed_lookup_q's table
 *
 * The gather reads the table AS IT IS, so this hook only answers whether it can: E4M3 codes with
 * ONE bf16 scale for the whole table (`fp8_e4m3*bf16[*x*]`, the n-gram table's recipe), or the
 * table as the checkpoint ships it, plain bf16 with no scale. The same two forms libavx's host
 * gather accepts (avx_layout_ngram), checked the same way, so a model whose table one of them
 * reads is never refused by the other at declare. */
extern "C" int r4d_rad_layout_embed_q(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, RadLayout* out) {
    (void)out;
    enum { EQ_WTE = 1, EQ_SCALE = 2 };
    if (operand != EQ_WTE && operand != EQ_SCALE) return RAD_E_UNSUPPORTED;
    const long long rows = rad_param_geti(p, n_p, "n_vocab", 0);
    const long long dim  = rad_param_geti(p, n_p, "n_embd", 0);
    if (rows <= 0 || dim <= 0) return RAD_E_SHAPE;
    if (rad_enc_is(enc, "plain") && enc->n_planes == 1 && enc->plane[0].dtype == RAD_BF16) {
        if (operand != EQ_WTE) return RAD_E_DTYPE;
        if (n != 1 || !pl || !r4d_enc::dims(pl[0], rows, dim)) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    if (!rad_enc_is(enc, "affine") || enc->n_planes != 2 || enc->transform[0] ||
        enc->plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 || sc->block[0] != 0 ||
        sc->block[1] != 0)
        return RAD_E_DTYPE;
    if (n != 1 || !pl || !sel) return RAD_E_SHAPE;
    if (!r4d_enc::selects(enc, sel, n, operand == EQ_WTE ? "codes" : "scale")) return RAD_E_SHAPE;
    if (!r4d_enc::dims(pl[0], operand == EQ_WTE ? rows : 1, operand == EQ_WTE ? dim : 1))
        return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;
}
