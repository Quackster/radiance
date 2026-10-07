// r4d_paroquant_quant_fp8_row_multi.hip -- ONE LAUNCH, P ROTATED ACTIVATIONS.
//
// The path-A producer (r4d_paroquant_quant_fp8_row) rotates a bf16 activation into ONE consumer's
// stored domain. But a block's input activation is shared by consumers with DIFFERENT rotations: an
// MLP's norm feeds gate_proj and up_proj, attention's norm feeds q, k and v, the delta net's norm
// feeds in_proj_qkv and in_proj_z. Serving each consumer with its own producer launch is what
// dominates path A's decode cost (measured: skipping the producers entirely puts path A at path B's
// rate, which is launch/execution bound and not arithmetic). This op rotates ONE input row into P
// domains in ONE launch.
//
// Each partition's arithmetic is the single producer's: the row is divided by that partition's
// `t_scale` (= 1/channel_scales), the S stages applied ascending and transposed, the whole-row amax
// taken, and E4M3 codes written with one f32 scale. Only the outer loop over partitions is new.
//
// The partitions share a shard: this op is used where the consumers' K is not cut (gate/up, q/k/v,
// in_proj_qkv/z are column-parallel), so one `k_base` covers them all.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_common.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;
constexpr int64_t  kLdsCap = 32 * 1024;
constexpr int      kMaxP = 4;

struct PQPart {
    const int32_t* pairs;
    const float*   angle;
    const float*   tscale;
    uint8_t*       q;
    float*         s;
};
struct PQPartSet { PQPart p[kMaxP]; };

__global__ __launch_bounds__(kThreads) void paroquant_quant_fp8_row_multi_kernel(
    const uint16_t* __restrict__ x, PQPartSet parts, uint32_t K, uint32_t S, int32_t k_base,
    uint32_t P, uint32_t x_ld, uint32_t pair_ld, uint32_t angle_ld, uint32_t ts_ld, uint32_t q_ld,
    uint32_t s_ld) {
    extern __shared__ float g[];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const size_t xrow = (size_t)row * x_ld;
    const uint32_t G = K / kGroup;
    const uint32_t lane = tid & (kWave - 1), warp = tid / kWave;

    for (uint32_t p = 0; p < P; ++p) {
        const PQPart& pt = parts.p[p];
        for (uint32_t k = tid; k < K; k += kThreads)
            g[k] = __uint_as_float((uint32_t)x[xrow + k] << 16) / pt.tscale[k];
        __syncthreads();
        /* The groups are independent, so they run across warps with `__syncwarp` a stage and only
         * the barriers the amax needs -- the single producer's shape, per partition. */
        for (uint32_t grp = warp; grp < G; grp += kWaves) {
            for (uint32_t st = 0; st < S; ++st) {
                const int32_t* pr = pt.pairs + (size_t)st * pair_ld + grp * (kGroup / 2) * 2;
                const float*   an = pt.angle + (size_t)st * angle_ld + grp * (kGroup / 2);
#pragma unroll
                for (uint32_t t2 = 0; t2 < 2; ++t2) {
                    const uint32_t t = lane + kWave * t2;
                    const uint32_t aa = (uint32_t)pr[2 * t] - k_base,
                                   bb = (uint32_t)pr[2 * t + 1] - k_base;
                    const float c = cosf(an[t]), sn = sinf(an[t]);
                    const float xa = g[aa], xb = g[bb];
                    g[aa] = fmaf(c, xa, sn * xb);
                    g[bb] = fmaf(c, xb, -sn * xa);
                }
                __syncwarp();
            }
        }
        __syncthreads();

        float amax = 0.0f;
        for (uint32_t k = tid; k < K; k += kThreads) amax = fmaxf(amax, fabsf(g[k]));
        amax = r4d_wave_max(amax);
        __shared__ float red[kWaves];
        if (lane == 0) red[warp] = amax;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < kWaves) ? red[lane] : 0.0f;
            v = r4d_wave_max(v);
            if (lane == 0) red[0] = v;
        }
        __syncthreads();
        const float sc = red[0] > 0.0f ? red[0] * (1.0f / 448.0f) : 1.0f;
        if (tid == 0) pt.s[(size_t)row * s_ld] = sc;
        const float inv = 1.0f / sc;
        for (uint32_t k = tid; k < K; k += kThreads) {
            const float v = fminf(fmaxf(g[k] * inv, -448.0f), 448.0f);
            const int pk = __builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false);
            pt.q[(size_t)row * q_ld + k] = (uint8_t)pk;
        }
        __syncthreads();
    }
}

}  // namespace

//   x            [M][K]   bf16 input, shared by every partition
//   parts.p[p]   { pairs [S][K] i32, angle [S][K/2] f32, t_scale [1][K] f32, q [M][K] e4m3,
//                  s [M][1] f32 }
//   P            partitions (<= 4); K*4 bytes of LDS, so K <= 8192
static int paroquant_quant_fp8_row_multi_launch(long a, const PQPartSet* set, int M, int K, int S,
                                                 int k_base, int P, int x_ld, int pair_ld,
                                                 int angle_ld, int ts_ld, int q_ld, int s_ld,
                                                 const R4dSink& sk) {
    if (!a || !set || M <= 0 || K <= 0 || S <= 0 || P <= 0 || P > kMaxP) return -1;
    if (K % (int)kGroup) return -3;
    if (!x_ld)   x_ld = K;
    if (!pair_ld) pair_ld = K;
    if (!angle_ld) angle_ld = K / 2;
    if (!ts_ld)  ts_ld = K;
    if (!q_ld)   q_ld = K;
    if (!s_ld)   s_ld = 1;
    const int64_t lds = (int64_t)K * 4;
    if (lds > kLdsCap) return -3;
    const int emit_rc = R4D_EMIT(sk, paroquant_quant_fp8_row_multi_kernel, dim3((uint32_t)M),
                                 dim3(kThreads), (size_t)lds,
                                 (const uint16_t*)a, *set, (uint32_t)K, (uint32_t)S,
                                 (int32_t)k_base, (uint32_t)P, (uint32_t)x_ld, (uint32_t)pair_ld,
                                 (uint32_t)angle_ld, (uint32_t)ts_ld, (uint32_t)q_ld,
                                 (uint32_t)s_ld);
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

/* The operand list is the schema's positionally: x (input), then P x {pairs, angle, t_scale}
 * (weights), then P x {q, scale} (outputs) -- inputs, then weights, then outputs, as every op's
 * list is. */
static int rad_paroquant_quant_fp8_row_multi(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* x = r4d_opt(a, 0);
    if (!x || x->rank < 2) return R4D_NO(RAD_E_INVAL);
    const long long M = x->shape[0];
    if (M < 1 || r4d_numel(x) % M) return R4D_NO(RAD_E_SHAPE);
    const long long K = r4d_numel(x) / M;
    if (K < (long long)kGroup || K % (long long)kGroup) return R4D_NO(RAD_E_SHAPE);
    long long P = 0, S = 0, kb = 0;
    if (!r4d_geti(a, "P", &P) || P < 1 || P > kMaxP) return R4D_NO(RAD_E_INVAL);
    if (!r4d_geti(a, "stages", &S) || S < 1) return R4D_NO(RAD_E_INVAL);
    r4d_geti(a, "k_base", &kb);
    const int wbase = 1, obase = 1 + 3 * (int)P;
    PQPartSet set{};
    long long pair_ld = K, angle_ld = K / 2, ts_ld = K, q_ld = K, s_ld = 1;
    for (long long p = 0; p < P; ++p) {
        const RadTensor* pr = r4d_opt(a, wbase + 3 * (int)p + 0);
        const RadTensor* an = r4d_opt(a, wbase + 3 * (int)p + 1);
        const RadTensor* ts = r4d_opt(a, wbase + 3 * (int)p + 2);
        const RadTensor* q  = r4d_opt(a, obase + 2 * (int)p + 0);
        const RadTensor* sc = r4d_opt(a, obase + 2 * (int)p + 1);
        if (!pr || !an || !ts || !q || !sc) return R4D_NO(RAD_E_INVAL);
        if (pr->rank != 2 || pr->shape[0] != S || pr->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
        if (an->rank != 2 || an->shape[0] != S || an->shape[1] != K / 2) return R4D_NO(RAD_E_SHAPE);
        if (r4d_numel(ts) != K) return R4D_NO(RAD_E_SHAPE);
        if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
        if (sc->shape[0] != M || sc->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
        set.p[p].pairs = (const int32_t*)r4d_addr(pr);
        set.p[p].angle = (const float*)r4d_addr(an);
        set.p[p].tscale = (const float*)r4d_addr(ts);
        set.p[p].q = (uint8_t*)r4d_addr(q);
        set.p[p].s = (float*)r4d_addr(sc);
        pair_ld = r4d_row_pitch(pr);
        angle_ld = r4d_row_pitch(an);
        ts_ld = r4d_row_pitch(ts);
        q_ld = r4d_row_pitch(q);
        s_ld = r4d_row_pitch(sc);
    }
    const int e = paroquant_quant_fp8_row_multi_launch(
        r4d_addr(x), &set, (int)M, (int)K, (int)S, (int)kb, (int)P, (int)r4d_row_pitch(x),
        (int)pair_ld, (int)angle_ld, (int)ts_ld, (int)q_ld, (int)s_ld, sk);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_paroquant_quant_fp8_row_multi_body(const RadArgs* a, const R4dSink& sk) {
    return rad_paroquant_quant_fp8_row_multi(a, sk);
}
R4D_ENTRY(r4d_rad_paroquant_quant_fp8_row_multi, rad_paroquant_quant_fp8_row_multi_body)
