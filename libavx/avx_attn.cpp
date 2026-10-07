/* avx_attn.cpp -- paged attention, dense attention, the KV write, and the two rotary forms.
 *
 * ============================== THE PAGED LAYOUT ==============================
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]      K then V inside a slot
 *
 * libr4d's, stated once for the whole project in libref/ref_ops.h, obeyed here. A flat slot index
 * s decomposes as block = s / block_size, offset = s % block_size; `block_table[seq, i]` holds the
 * block id of the sequence's i-th block and `slot_mapping[token]` the flat slot a token writes to.
 *
 * ============================== THE MASK ==============================
 *
 * `seqused[s]` is the sequence's TOTAL key count INCLUDING this step's query tokens, so query row
 * i of a q_len-row step sits at absolute position seqused[s] - q_len + i. That is what makes one
 * kernel serve a decode (q_len 1), a speculative verify (q_len <= 64) and a prefill chunk (q_len
 * in the thousands) with no second length parameter.
 *
 *   causal = 1   key j is visible when j <= pos
 *   causal = 0   key j is visible when j <  seqused[s]
 *   window > 0   causal: additionally pos - j < window; non-causal: |pos - j| < window, the
 *                two-sided reading libr4d's h128 kernel documents.
 *
 * ============================== ONE PASS OVER THE CACHE, NOT TWO ==============================
 *
 * This is the one place where this plugin deliberately computes the softmax in a DIFFERENT ORDER
 * from the oracle, and it is worth being explicit about the trade.
 *
 * libref makes two passes over the whole context: one to find the maximum score, one to
 * exponentiate and accumulate -- and it RECOMPUTES every dot product in the second pass, because
 * keeping them would need a buffer the size of the context and "the reference has time it does not
 * have memory". At a 128K context that is two full sweeps of the KV cache and twice the dot
 * products, and the KV cache is the largest thing a decode step touches.
 *
 * Here the softmax is ONLINE and tiled, the flash-attention arrangement: keys are taken in tiles,
 * a tile's scores are computed once, the running maximum is updated at the tile boundary and the
 * accumulator rescaled by exp(m_old - m_new) when it moves. One pass, one dot product per key, and
 * the memory is one tile of scores rather than one context of them.
 *
 * WHAT IT COSTS: the rescalings round. The result is mathematically identical -- the final running
 * maximum IS the global maximum, and every weight ends up as exp(s_j - m_final) -- but the
 * intermediate multiplications are extra roundings that libref does not perform, so this cannot be
 * bit-exact against it and does not claim to be. It lands at ~1e-7 relative against a tolerance of
 * 1e-4 (f32) / 2e-2 (bf16). Every other numerical choice in this file is exact by construction:
 *
 *   THE DESCALES ARE APPLIED PER ELEMENT, NOT TO THE DOT. libref computes
 *   `dot += q[d] * (k[d] * kds)`, which is not the same f32 number as `dot(q, k) * kds`. The K row
 *   is therefore scaled by `kds` as it is widened, which reproduces the reference's arithmetic
 *   exactly rather than to a tolerance. Same for V and `vds`.
 *
 *   AN EMPTY WINDOW IS A ZERO ROW, NOT A NaN. A softmax over no visible key has no defined answer
 *   and zero is what every kernel in this project produces. The checker compares the finiteness
 *   pattern before any norm, so getting this wrong is a hard failure rather than a tolerance
 *   question -- which is the right treatment.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <vector>
#include "rad_yarn.h"

using namespace avx;

/* The widest head this accumulates in one frame. MLA's 576 is the widest in circulation and
 * libr4d's widest is 256, so 1024 is headroom rather than a limit anyone meets; above it the op
 * says RAD_E_UNSUPPORTED by name instead of overflowing. Same number as libref's, so the two
 * refuse the same calls. */
enum { AVX_HD_MAX = 1024 };

/* Keys per online-softmax tile. Sized so a tile of scores and the key rows behind it stay in L1
 * beside the query head and the accumulator: at head_dim 128 that is 64 keys x 128 floats = 32 KiB
 * of K plus the same of V, which is the working set a Zen-class L1 holds. It is a constant rather
 * than a tunable because the cost curve is flat between 32 and 128 -- what matters is that it is
 * not 1 (a rescale per key) and not the whole context (libref's memory problem). */
enum { AVX_KV_TILE = 64 };

struct KVView {
    const RadTensor* t;
    int64_t sb, sh, so, sd;   /* strides: block, kv head, slot offset, element */
    int64_t head_dim;
};

static KVView kv_view(const RadTensor* c, int64_t head_dim, int64_t block_size, int64_t kv_heads) {
    KVView v{};
    v.t = c;
    v.head_dim = head_dim;
    if (c->rank == 4) {
        v.sb = c->stride[0]; v.sh = c->stride[1]; v.so = c->stride[2]; v.sd = c->stride[3];
    } else {
        v.sd = 1;
        v.so = 2 * head_dim;
        v.sh = block_size * v.so;
        v.sb = kv_heads * v.sh;
    }
    return v;
}

/* A head of an operand, widened into f32 and optionally scaled as it goes -- which is what makes
 * the descale exact against libref rather than close (see the header). */
static inline const float* head_f32(const RadTensor* t, int64_t base, int64_t ds, int64_t n,
                                    float mul, float* scr) {
    if (ds == 1) {
        if (mul == 1.0f) return row_f32_in(t->dtype, byte_at(t->data, t->dtype, base), scr, n);
        row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), scr, n);
    } else {
        for (int64_t d = 0; d < n; ++d) scr[d] = ldt(t, base + d * ds);
    }
    if (mul != 1.0f) {
        const vf vm = vf_set1(mul);
        int64_t i = 0;
        for (; i + VF_N <= n; i += VF_N) vf_storeu(scr + i, vf_mul(vf_loadu(scr + i), vm));
        for (; i < n; ++i) scr[i] *= mul;
    }
    return scr;
}

/* A dot product with four accumulators, which is what covers the FMA latency. Shared by both
 * attention forms and by nothing else, so it lives here rather than in the header. */
static inline float dotf(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
        a1 = vf_fma(vf_loadu(a + i + VF_N), vf_loadu(b + i + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + i + 2 * VF_N), vf_loadu(b + i + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + i + 3 * VF_N), vf_loadu(b + i + 3 * VF_N), a3);
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* acc += w * v, vectorised. */
static inline void axpy(float* acc, const float* v, float w, int64_t n) {
    const vf vw = vf_set1(w);
    int64_t i = 0;
    for (; i + VF_N <= n; i += VF_N)
        vf_storeu(acc + i, vf_fma(vw, vf_loadu(v + i), vf_loadu(acc + i)));
    for (; i < n; ++i) acc[i] += w * v[i];
}

/* ------------------------------------------------------------------ attn_paged */
/* Prefill form, taken when the queries are uniform (!cus) and long (q_len >= 32): the
 * whole context is shared across q_len rows, so widening K/V per q-row (the decode form in
 * AVX_KERNEL below) re-converts the context q_len times -- 512x over at the bench shape.
 * Each (seq, kv-head, q-block) unit widens the context ONCE into panels and runs its rows
 * off them. The per-row math is unchanged -- dots ascending, the online softmax with the
 * same rescale points, the same scalar exps -- so rows match the decode form bit for bit;
 * only the loads move. q-blocks of 64 keep the units many for the engine's thread pool.
 * Ragged (cus) inputs stay on the decode form, whatever their lengths. */
static int attn_paged_prefill(const RadArgs* a, [[maybe_unused]] RadStream stream) {
    const RadTensor *q = &a->t[0], *kv = &a->t[1], *bt = &a->t[2], *su = &a->t[3], *o = &a->t[7];
    const RadTensor* kdes = rad_arg_in(a, 4);
    const RadTensor* vdes = rad_arg_in(a, 5);
    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa = p_int(a, "gqa", 1);
    const int64_t block_size = p_int(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    const int causal = (int)p_int(a, "causal", 1);
    const int64_t window = p_int(a, "window", 0);
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > AVX_HD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t kv_heads = kv->rank == 4 ? kv->shape[1] : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = gqa * kv_heads;
    const int64_t n_seq = numel(su);
    /* THE QUERY LENGTH IS PER SEQUENCE. The operand holds every sequence's rows back to back, so
     * its row count is n_seq * q_len; taking the whole of it as one sequence's length puts
     * sequence s at rows s * rows and writes every sequence after the first past `out`. */
    const int64_t rows = numel(q) / (q_heads * head_dim);
    if (n_seq <= 0 || rows <= 0 || rows % n_seq != 0) return RAD_E_SHAPE;
    const int64_t q_len = rows / n_seq;
    const int64_t max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : max_blocks;
    const int64_t bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
    const KVView kvv = kv_view(kv, head_dim, block_size, kv_heads);
    /* Read as [rows, q_heads, head_dim] at whatever rank they arrived, libref's reading: a rank-2
     * q is a column slice whose row stride is its own (avx_common.h: row_stride). */
    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));

    enum { QB = 64 };
    const int64_t nqb = (q_len + QB - 1) / QB;
    AVX_PARALLEL_FOR_IF(n_seq * kv_heads * nqb >= 4)
    for (int64_t u = 0; u < n_seq * kv_heads * nqb; ++u) {
        const int64_t qb = u % nqb, kh = (u / nqb) % kv_heads, s = u / (nqb * kv_heads);
        const int64_t ctx = ldi(su, offlin(su, s));
        if (ctx <= 0) continue;
        const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
        const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;
        /* Panels plus one row frame: a single arena call, partitioned by hand. */
        float* mem = scratch_f32((size_t)2 * (size_t)ctx * (size_t)head_dim +
                                 (size_t)(3 * head_dim + AVX_KV_TILE) + 16);
        float* Kp = mem;
        float* Vp = mem + (size_t)ctx * (size_t)head_dim;
        float* frame = Vp + (size_t)ctx * (size_t)head_dim;
        float* qbuf = frame;
        float* kbuf = frame + head_dim;
        float* acc = frame + 2 * head_dim;
        float* sbuf = frame + 3 * head_dim;
        int64_t* kbv = (int64_t*)scratch_raw((size_t)ctx * sizeof(int64_t) + 64);
        /* Widen the context once. Rows past max_blocks stay invalid, exactly as the
         * per-row loop's break would leave them. */
        int64_t nkeys = 0;
        for (int64_t j = 0; j < ctx; ++j) {
            if (j / block_size >= max_blocks) break;
            const int64_t blk = ldi(bt, s * bt_rs + (j / block_size) * bt_cs);
            if (blk < 0) {
                kbv[j] = -1;
                continue;
            }
            kbv[j] = blk * kvv.sb + kh * kvv.sh + (j % block_size) * kvv.so;
            const float* kr = head_f32(kv, kbv[j], kvv.sd, head_dim, kds, kbuf);
            for (int64_t d = 0; d < head_dim; ++d) Kp[(size_t)j * head_dim + d] = kr[d];
            const float* vr = head_f32(kv, kbv[j] + head_dim * kvv.sd, kvv.sd, head_dim,
                                       vds, kbuf);
            for (int64_t d = 0; d < head_dim; ++d) Vp[(size_t)j * head_dim + d] = vr[d];
            nkeys = j + 1;
        }
        const int64_t i0 = qb * QB, i1 = (i0 + QB < q_len) ? i0 + QB : q_len;
        /* One gqa row at a time. A joint-rows form -- each key row loaded once for all gqa
         * dots -- measures identical on a long-query shape: the nest is compute-saturated
         * (dots, axpy, scalar exps) rather than bound by panel traffic, so sharing the loads
         * frees nothing. Do not switch to it without measuring. */
        for (int64_t g = 0; g < gqa; ++g) {
            const int64_t h = kh * gqa + g;
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t tok = s * q_len + i;
                const int64_t pos = ctx - q_len + i;
                const int64_t qb2 = tok * q_rs + h * q_hs;
                const int64_t ob = tok * o_rs + h * o_hs;
                const float* qv = head_f32(q, qb2, q_ds, head_dim, 1.0f, qbuf);
                for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
                float m = -INFINITY, l = 0.0f;
                int ntile = 0;
                int64_t kbase[AVX_KV_TILE];
                auto flush = [&]() {
                    if (ntile == 0) return;
                    float tmax = sbuf[0];
                    for (int t = 1; t < ntile; ++t) if (sbuf[t] > tmax) tmax = sbuf[t];
                    if (tmax > m) {
                        const float r = (m == -INFINITY) ? 0.0f : std::exp(m - tmax);
                        if (r != 1.0f) {
                            const vf vr = vf_set1(r);
                            int64_t d = 0;
                            for (; d + VF_N <= head_dim; d += VF_N)
                                vf_storeu(acc + d, vf_mul(vf_loadu(acc + d), vr));
                            for (; d < head_dim; ++d) acc[d] *= r;
                            l *= r;
                        }
                        m = tmax;
                    }
                    /* Scalar exps here, deliberately: the tile-vectorised form
                     * (vf_exp, as in the decode flush) measures neutral-to-slower on a
                     * long-query shape, because the nest is bound elsewhere. The values
                     * are the same either way. */
                    for (int t = 0; t < ntile; ++t) {
                        const float w = std::exp(sbuf[t] - m);
                        l += w;
                        axpy(acc, Vp + (size_t)kbase[t] * head_dim, w, head_dim);
                    }
                    ntile = 0;
                };
                for (int64_t j = 0; j < nkeys; ++j) {
                    if (causal ? (j > pos) : (j >= ctx)) continue;
                    if (window > 0) {
                        const int64_t d = pos - j;
                        if (causal ? (d >= window) : (d >= window || d <= -window)) continue;
                    }
                    if (kbv[j] < 0) continue;
                    sbuf[ntile] = dotf(qv, Kp + (size_t)j * head_dim, head_dim) * scale;
                    kbase[ntile] = j;
                    if (++ntile == AVX_KV_TILE) flush();
                }
                flush();
                if (m == -INFINITY) {
                    for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                    continue;
                }
                const float inv = l > 0.0f ? 1.0f / l : 0.0f;
                if (o_ds == 1) {
                    const vf vi = vf_set1(inv);
                    int64_t d = 0;
                    for (; d + VF_N <= head_dim; d += VF_N)
                        vf_storeu(acc + d, vf_mul(vf_loadu(acc + d), vi));
                    for (; d < head_dim; ++d) acc[d] *= inv;
                    out_row(o, ob, head_dim, acc);
                } else {
                    for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
                }
            }
        }
    }
    return RAD_OK;
}

AVX_KERNEL(attn_paged) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *kv = &a->t[1], *bt = &a->t[2], *su = &a->t[3], *o = &a->t[7];
    const RadTensor* kdes = rad_arg_in(a, 4);
    const RadTensor* vdes = rad_arg_in(a, 5);
    const RadTensor* cus  = rad_arg_in(a, 6);
    if (!q->data || !kv->data || !bt->data || !su->data || !o->data) return RAD_E_INVAL;

    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa = p_int(a, "gqa", 1);
    /* `block_size` is RAD_DERIVED (docs/OPS.md), so a caller who did not pin it and resolved to a
     * kernel that declares no RAD_C_EQ for it arrives here without the key. The cache tensor says
     * what the block size is; reading it off the operand is this plugin's rule anyway. */
    const int64_t block_size = p_int(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    const int causal = (int)p_int(a, "causal", 1);
    const int64_t window = p_int(a, "window", 0);
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > AVX_HD_MAX) return RAD_E_UNSUPPORTED;

    const int64_t kv_heads = kv->rank == 4 ? kv->shape[1] : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = gqa * kv_heads;
    if (q->rank >= 3 && q->shape[q->rank - 2] != q_heads) return RAD_E_SHAPE;
    const int64_t n_seq = numel(su);
    const int64_t rows = numel(q) / (q_heads * head_dim);
    if (n_seq <= 0 || rows <= 0) return RAD_E_SHAPE;
    if (!cus && rows % n_seq != 0) return RAD_E_SHAPE;
    if (cus && numel(cus) != n_seq + 1) return RAD_E_SHAPE;
    const int64_t q_len = cus ? 0 : rows / n_seq;
    if (numel(o) < rows * q_heads * head_dim) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[2] != block_size) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[3] < 2 * head_dim) return RAD_E_SHAPE;

    const int64_t max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : max_blocks;
    const int64_t bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
    const KVView kvv = kv_view(kv, head_dim, block_size, kv_heads);

    /* Row strides of q and out read as [rows, q_heads, head_dim], at whatever rank they arrived --
     * libref's reading. A rank-2 operand is a column slice whose row stride is its own. */
    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);

    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
    /* Uniform long queries share one context: the panel form widens it once instead of once
     * per row. Ragged (cus) inputs stay below whatever their lengths. */
    if (!cus && q_len >= 32) return attn_paged_prefill(a, stream);
    /* Rows run as (token, kv-head) groups: the gqa query rows of a group share the key
     * and value rows behind every key, so each is widened once for gqa dots/axpys rather
     * than once per row. Per-row arithmetic is untouched -- rows match bit for bit. */
    const int64_t total = rows * kv_heads;

    AVX_PARALLEL_FOR_IF(total >= 2)
    for (int64_t tidx = 0; tidx < total; ++tidx) {
        const int64_t kh = tidx % kv_heads;
        const int64_t tok = tidx / kv_heads;
        /* Which sequence owns this token and where inside its query it sits. Linear rather
         * than a binary search: n_seq is a handful. Once per group, not once per row. */
        int64_t s = 0, qlen = q_len, i = 0;
        if (cus) {
            while (s + 1 < n_seq && ldi(cus, offlin(cus, s + 1)) <= tok) ++s;
            const int64_t beg = ldi(cus, offlin(cus, s));
            qlen = ldi(cus, offlin(cus, s + 1)) - beg;
            i = tok - beg;
        } else {
            s = tok / q_len;
            i = tok - s * q_len;
        }
        if (qlen <= 0) continue;
        const int64_t ctx = ldi(su, offlin(su, s));
        const int64_t pos = ctx - qlen + i;
        const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
        const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;

        /* One arena call carves the whole frame, because a second one may reallocate and
         * invalidate the first pointer (avx_common.h). */
        float* frame = (float*)scratch_f32((size_t)gqa * (size_t)(2 * head_dim + AVX_KV_TILE + 2) +
                                           (size_t)(2 * head_dim) + 16);
        float* qpan = frame;                                 /* [gqa, head_dim] queries */
        float* krow = qpan + (size_t)gqa * head_dim;         /* one widened key row */
        float* vrow = krow + head_dim;                       /* one widened value row */
        float* accs = vrow + head_dim;                       /* [gqa, head_dim] */
        float* sbufs = accs + (size_t)gqa * head_dim;        /* [gqa, TILE] scores */
        float* mm = sbufs + (size_t)gqa * AVX_KV_TILE;
        float* ll = mm + gqa;
        int64_t kbase[AVX_KV_TILE];

        for (int64_t g = 0; g < gqa; ++g) {
            const int64_t h = kh * gqa + g;
            float* qd = qpan + g * head_dim;
            const float* qr = head_f32(q, tok * q_rs + h * q_hs, q_ds, head_dim, 1.0f, qd);
            if (qr != qd) std::memcpy(qd, qr, (size_t)head_dim * sizeof(float));
            float* ac = accs + g * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) ac[d] = 0.0f;
            mm[g] = -INFINITY;
            ll[g] = 0.0f;
        }
        int ntile = 0;

        /* Flush a tile: per row, update the running maximum, rescale what is already
         * accumulated, then add this tile's weighted values in ascending key order. The
         * value row is widened once per key and shared across the group's axpys. */
        auto flush = [&]() {
            if (ntile == 0) return;
            for (int64_t g = 0; g < gqa; ++g) {
                float* sb = sbufs + g * AVX_KV_TILE;
                float* ac = accs + g * head_dim;
                float m = mm[g], l = ll[g];
                float tmax = sb[0];
                for (int t = 1; t < ntile; ++t) if (sb[t] > tmax) tmax = sb[t];
                if (tmax > m) {
                    const float r = (m == -INFINITY) ? 0.0f : std::exp(m - tmax);
                    if (r != 1.0f) {
                        const vf vr = vf_set1(r);
                        int64_t d = 0;
                        for (; d + VF_N <= head_dim; d += VF_N)
                            vf_storeu(ac + d, vf_mul(vf_loadu(ac + d), vr));
                        for (; d < head_dim; ++d) ac[d] *= r;
                        l *= r;
                    }
                    m = tmax;
                }
                /* The tile's exps, vectorised: the accumulation keeps its order. */
                {
                    const vf vm = vf_set1(m);
                    int t = 0;
                    for (; t + VF_N <= ntile; t += VF_N)
                        vf_storeu(sb + t, vf_exp(vf_sub(vf_loadu(sb + t), vm)));
                    for (; t < ntile; ++t) sb[t] = std::exp(sb[t] - m);
                }
                mm[g] = m;
                ll[g] = l;
            }
            for (int t = 0; t < ntile; ++t) {
                /* One widening per key for the whole group, then gqa axpys. */
                const float* vv = head_f32(kv, kbase[t] + head_dim * kvv.sd, kvv.sd, head_dim,
                                           vds, vrow);
                for (int64_t g = 0; g < gqa; ++g) {
                    const float w = sbufs[g * AVX_KV_TILE + t];
                    ll[g] += w;
                    axpy(accs + g * head_dim, vv, w, head_dim);
                }
            }
            ntile = 0;
        };

        for (int64_t j = 0; j < ctx; ++j) {
            if (causal ? (j > pos) : (j >= ctx)) continue;
            if (window > 0) {
                const int64_t d = pos - j;
                if (causal ? (d >= window) : (d >= window || d <= -window)) continue;
            }
            if (j / block_size >= max_blocks) break;   /* the table is only this wide */
            const int64_t blk = ldi(bt, s * bt_rs + (j / block_size) * bt_cs);
            if (blk < 0) continue;
            const int64_t kb = blk * kvv.sb + kh * kvv.sh + (j % block_size) * kvv.so;
            /* One widening per key for the whole group, then gqa dots. */
            const float* kvrow = head_f32(kv, kb, kvv.sd, head_dim, kds, krow);
            for (int64_t g = 0; g < gqa; ++g)
                sbufs[g * AVX_KV_TILE + ntile] =
                    dotf(qpan + g * head_dim, kvrow, head_dim) * scale;
            kbase[ntile] = kb;
            if (++ntile == AVX_KV_TILE) flush();
        }
        flush();

        for (int64_t g = 0; g < gqa; ++g) {
            const int64_t h = kh * gqa + g;
            const int64_t ob = tok * o_rs + h * o_hs;
            float* ac = accs + g * head_dim;
            if (mm[g] == -INFINITY) {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                continue;
            }
            const float inv = ll[g] > 0.0f ? 1.0f / ll[g] : 0.0f;
            if (o_ds == 1) {
                const vf vi = vf_set1(inv);
                int64_t d = 0;
                for (; d + VF_N <= head_dim; d += VF_N)
                    vf_storeu(ac + d, vf_mul(vf_loadu(ac + d), vi));
                for (; d < head_dim; ++d) ac[d] *= inv;
                out_row(o, ob, head_dim, ac);
            } else {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, ac[d] * inv);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_dense, bf16 blocks */
/* AN ENCODER'S ATTENTION IS ALL PREFILL, and the per-query loop below is built for a decode row:
 * it parallelises over (segment, head) -- eight units for one long input on a sixteen-core part --
 * and widens every key row to f32 again for every query that reads it, so a 2000-token segment
 * pays four million row conversions a head. Measured at 130 GFLOPS on the sliding layers.
 *
 * Where q, k and v are bf16 with unit element stride and the CPU has AVX512_BF16, this blocked
 * form runs instead. A task is (segment, KV head, 32 query positions) and holds the rows of every
 * query head that reads that KV head, so one key tile serves gqa x 32 rows. Per tile of 64 keys:
 *
 *   S = Q K^T     vdpbf16ps against the tile packed once as key-interleaved bf16 pairs, each
 *                 query pair broadcast into 16-key vectors: no score is horizontally reduced
 *   the mask      causal, window and segment end, on the scores
 *   the softmax   online, the running max moved at tile boundaries (this file's arrangement)
 *   O += P V      V's tile packed once for the task as key-interleaved bf16 pairs, P rounded to
 *                 bf16 pairs along the keys (as PyTorch's CPU flash attention does), six rows of
 *                 64 output columns a block
 *
 * The key range of a task is cut to what its queries can see, so a sliding layer costs
 * O(T * window). Each output row is one task's, so the result does not depend on the thread
 * count. Same arithmetic class as the loop below: f32 scores (bf16 products are exact in f32),
 * f32 softmax and accumulator, one rounding at the store. */
#if AVX_LEVEL == 3
#define AVX_BF16_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))

/* S[r][0..63] for MB_ query rows against a key tile packed as bf16 PAIRS INTERLEAVED BY KEY:
 * kp[d2 * 64 + j] holds key j's dims 2*d2 and 2*d2+1. Each step broadcasts one row's pair and
 * multiplies it into four 16-key vectors, so a score is never horizontally reduced -- with a
 * 64-wide head a reduction per score cost more than the two products it closed. */
template <int MB_>
AVX_BF16_TARGET static inline void qk_rows(const uint16_t* const* qr, const uint32_t* kp,
                                           int64_t D2, float scale, float* S, int64_t ldS) {
    __m512 acc[MB_][4];
    #pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
        #pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) acc[i][v] = _mm512_setzero_ps();
    for (int64_t d2 = 0; d2 < D2; ++d2) {
        const uint32_t* kr = kp + d2 * 64;
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(kr));
        const __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 16));
        const __m512bh b2 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 32));
        const __m512bh b3 = (__m512bh)_mm512_loadu_si512((const void*)(kr + 48));
        #pragma GCC unroll 8
        for (int i = 0; i < MB_; ++i) {
            uint32_t pr;
            std::memcpy(&pr, qr[i] + 2 * d2, 4);
            const __m512bh a = (__m512bh)_mm512_set1_epi32((int)pr);
            acc[i][0] = _mm512_dpbf16_ps(acc[i][0], a, b0);
            acc[i][1] = _mm512_dpbf16_ps(acc[i][1], a, b1);
            acc[i][2] = _mm512_dpbf16_ps(acc[i][2], a, b2);
            acc[i][3] = _mm512_dpbf16_ps(acc[i][3], a, b3);
        }
    }
    const __m512 vs = _mm512_set1_ps(scale);
    #pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
        #pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(S + i * ldS + 16 * v, _mm512_mul_ps(acc[i][v], vs));
}

/* acc[r][c0 .. c0+63] += P[r] . V for MB_ rows, in bf16 pairs: pp[r * 32 + k2] holds row r's
 * probabilities for keys 2*k2 and 2*k2+1, vp[k2 * D + d] holds those two keys' values at column
 * d. PyTorch's CPU flash attention -- the reference's -- rounds P to bf16 for this product as
 * well; the softmax's sum is kept in f32 from the unrounded P. */
template <int MB_>
AVX_BF16_TARGET static inline void pv_rows(float* acc, int64_t D, const uint32_t* pp,
                                           const uint32_t* vp, int64_t K2, int64_t c0) {
    __m512 o[MB_][4];
    #pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
        #pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) o[i][v] = _mm512_loadu_ps(acc + i * D + c0 + 16 * v);
    for (int64_t k2 = 0; k2 < K2; ++k2) {
        const uint32_t* vr = vp + k2 * D + c0;
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512((const void*)(vr));
        const __m512bh b1 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 16));
        const __m512bh b2 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 32));
        const __m512bh b3 = (__m512bh)_mm512_loadu_si512((const void*)(vr + 48));
        #pragma GCC unroll 8
        for (int i = 0; i < MB_; ++i) {
            const __m512bh a = (__m512bh)_mm512_set1_epi32((int)pp[i * 32 + k2]);
            o[i][0] = _mm512_dpbf16_ps(o[i][0], a, b0);
            o[i][1] = _mm512_dpbf16_ps(o[i][1], a, b1);
            o[i][2] = _mm512_dpbf16_ps(o[i][2], a, b2);
            o[i][3] = _mm512_dpbf16_ps(o[i][3], a, b3);
        }
    }
    #pragma GCC unroll 8
    for (int i = 0; i < MB_; ++i)
        #pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(acc + i * D + c0 + 16 * v, o[i][v]);
}

AVX_BF16_TARGET static bool attn_dense_bf16(const RadTensor* q, const RadTensor* k,
                                            const RadTensor* v, const RadTensor* cu,
                                            const RadTensor* o, int64_t D, int64_t gqa,
                                            int64_t kv_heads, int causal, int64_t window,
                                            float scale, int64_t q_rs, int64_t q_hs, int64_t q_ds,
                                            int64_t k_rs, int64_t k_hs, int64_t k_ds, int64_t v_rs,
                                            int64_t v_hs, int64_t v_ds, int64_t o_rs, int64_t o_hs,
                                            int64_t o_ds) {
    static const bool have = __builtin_cpu_supports("avx512bf16");
    if (!have) return false;
    if (q->dtype != RAD_BF16 || k->dtype != RAD_BF16 || v->dtype != RAD_BF16) return false;
    if (q_ds != 1 || k_ds != 1 || v_ds != 1 || o_ds != 1 || D % 64 != 0 || gqa > 16) return false;

    constexpr int64_t QB = 32, KT = 64;
    const int64_t n_seq = numel(cu) - 1;
    /* The task list: (segment, kv head, query block), flattened through a prefix over segments. */
    std::vector<int64_t> first((size_t)n_seq + 1, 0);
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t len = ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s));
        first[(size_t)s + 1] = first[(size_t)s] + kv_heads * ((len + QB - 1) / QB);
    }
    const int64_t n_task = first[(size_t)n_seq];
    const uint16_t* Q = (const uint16_t*)q->data;
    const uint16_t* K = (const uint16_t*)k->data;
    const uint16_t* V = (const uint16_t*)v->data;
    const int64_t R_MAX = QB * gqa;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(avx_team_size())
#endif
    for (int64_t t = 0; t < n_task; ++t) {
        int64_t s = (int64_t)(std::upper_bound(first.begin(), first.end(), t) - first.begin()) - 1;
        const int64_t b0 = ldi(cu, offlin(cu, s)), len = ldi(cu, offlin(cu, s + 1)) - b0;
        const int64_t local = t - first[(size_t)s];
        const int64_t nqb = (len + QB - 1) / QB;
        const int64_t kh = local / nqb, q0 = (local % nqb) * QB;
        const int64_t q1 = std::min(q0 + QB, len), nq = q1 - q0, R = nq * gqa;

        float* frame = scratch_f32((size_t)(R_MAX * D + R_MAX * KT + 2 * R_MAX + KT * D / 2 +
                                            KT * D / 2 + R_MAX * KT / 2) + 64);
        float* acc = frame;
        float* S   = acc + R_MAX * D;
        float* m   = S + R_MAX * KT;
        float* l   = m + R_MAX;
        uint32_t* kpack = reinterpret_cast<uint32_t*>(l + R_MAX);
        uint32_t* vpack = kpack + KT * D / 2;
        uint32_t* ppack = vpack + KT * D / 2;
        const int64_t D2 = D / 2;
        const uint16_t* qrow[16 * QB];
        for (int64_t r = 0; r < R; ++r) {
            const int64_t qi = r / gqa, g = r % gqa;
            qrow[r] = Q + (b0 + q0 + qi) * q_rs + (kh * gqa + g) * q_hs;
            m[r] = -INFINITY;
            l[r] = 0.0f;
        }
        std::memset(acc, 0, sizeof(float) * (size_t)(R * D));

        int64_t jbeg = 0, jend = causal ? q1 : len;
        if (window > 0) {
            jbeg = std::max<int64_t>(0, q0 - window + 1);
            if (!causal) jend = std::min<int64_t>(len, q1 - 1 + window);
        }
        for (int64_t j0 = jbeg; j0 < jend; j0 += KT) {
            const int64_t kt = std::min(KT, jend - j0);
            /* The key tile packed once for every row of the task: pairs interleaved by key, the
             * keys past the tile's end zero (their scores are never read). */
            for (int64_t d2 = 0; d2 < D2; ++d2) {
                uint32_t* dst = kpack + d2 * KT;
                for (int64_t j = 0; j < kt; ++j)
                    std::memcpy(dst + j, K + (b0 + j0 + j) * k_rs + kh * k_hs + 2 * d2, 4);
                for (int64_t j = kt; j < KT; ++j) dst[j] = 0;
            }

            /* S, six rows a block against the whole packed tile. */
            int64_t r = 0;
            for (; r + 6 <= R; r += 6) qk_rows<6>(qrow + r, kpack, D2, scale, S + r * KT, KT);
            for (; r < R; ++r) qk_rows<1>(qrow + r, kpack, D2, scale, S + r * KT, KT);
            /* The mask, by position inside the segment. */
            if (causal || window > 0) {
                for (int64_t r = 0; r < R; ++r) {
                    const int64_t i = q0 + r / gqa;
                    for (int64_t j = 0; j < kt; ++j) {
                        const int64_t d = i - (j0 + j);
                        bool ok = !causal || d >= 0;
                        if (window > 0) ok = ok && (causal ? d < window : (d < window && -d < window));
                        if (!ok) S[r * KT + j] = -INFINITY;
                    }
                }
            }
            /* The online softmax, row by row, four vectors a row; S becomes P in place. The keys
             * past a short tile's end score -inf, so P is zero there and every row is full width. */
            if (kt < KT)
                for (int64_t r = 0; r < R; ++r)
                    for (int64_t j = kt; j < KT; ++j) S[r * KT + j] = -INFINITY;
            for (int64_t r = 0; r < R; ++r) {
                float* sr = S + r * KT;
                const float tmax = _mm512_reduce_max_ps(
                    _mm512_max_ps(_mm512_max_ps(_mm512_loadu_ps(sr), _mm512_loadu_ps(sr + 16)),
                                  _mm512_max_ps(_mm512_loadu_ps(sr + 32), _mm512_loadu_ps(sr + 48))));
                if (tmax == -INFINITY) {
                    for (int64_t j = 0; j < KT; ++j) sr[j] = 0.0f;
                    continue;
                }
                if (tmax > m[r]) {
                    const float alpha = m[r] == -INFINITY ? 0.0f : std::exp(m[r] - tmax);
                    if (alpha != 1.0f) {
                        float* ar = acc + r * D;
                        const __m512 va = _mm512_set1_ps(alpha);
                        for (int64_t d = 0; d < D; d += 16)
                            _mm512_storeu_ps(ar + d, _mm512_mul_ps(_mm512_loadu_ps(ar + d), va));
                        l[r] *= alpha;
                    }
                    m[r] = tmax;
                }
                const vf vm = vf_set1(m[r]);
                __m512 sum = _mm512_setzero_ps();
                for (int64_t j = 0; j < KT; j += 16) {
                    /* exp of -inf is the clamp's floor, ~1e-38: zero to every sum it joins. */
                    const vf e = vf_exp(vf_sub(vf_loadu(sr + j), vm));
                    vf_storeu(sr + j, e);
                    sum = _mm512_add_ps(sum, e);
                }
                l[r] += _mm512_reduce_add_ps(sum);
            }
            /* V's tile packed once for every row of the task, consecutive keys interleaved in
             * pairs (a key past the tile's end is zero), and each row's P rounded to bf16 pairs
             * along the keys. */
            const int64_t K2 = (kt + 1) / 2;
            {
                /* Two key rows' words interleaved, 32 columns a step: lanes 0..15 of the first
                 * output are (k0[d], k1[d]) for d = 0..15, the second's for d = 16..31. */
                const __m512i ilo = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10,
                                                     41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4, 35, 3,
                                                     34, 2, 33, 1, 32, 0);
                const __m512i ihi = _mm512_set_epi16(63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26,
                                                     57, 25, 56, 24, 55, 23, 54, 22, 53, 21, 52, 20,
                                                     51, 19, 50, 18, 49, 17, 48, 16);
                for (int64_t k2 = 0; k2 < K2; ++k2) {
                    const uint16_t* v0 = V + (b0 + j0 + 2 * k2) * v_rs + kh * v_hs;
                    const bool two = 2 * k2 + 1 < kt;
                    uint32_t* dst = vpack + k2 * D;
                    for (int64_t d = 0; d < D; d += 32) {
                        const __m512i a = _mm512_loadu_si512((const void*)(v0 + d));
                        const __m512i bb = two ? _mm512_loadu_si512((const void*)(v0 + v_rs + d))
                                               : _mm512_setzero_si512();
                        _mm512_storeu_si512((void*)(dst + d), _mm512_permutex2var_epi16(a, ilo, bb));
                        _mm512_storeu_si512((void*)(dst + d + 16),
                                            _mm512_permutex2var_epi16(a, ihi, bb));
                    }
                }
            }
            /* P to bf16, round to nearest even: a row's 64 values are its 32 key pairs in order. */
            for (int64_t r = 0; r < R; ++r) {
                const float* sr = S + r * KT;
                uint32_t* pr = ppack + r * 32;
                _mm512_storeu_si512((void*)pr, (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(sr + 16),
                                                                             _mm512_loadu_ps(sr)));
                _mm512_storeu_si512((void*)(pr + 16),
                                    (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(sr + 48),
                                                                 _mm512_loadu_ps(sr + 32)));
            }
            for (int64_t c0 = 0; c0 < D; c0 += 64) {
                int64_t r = 0;
                for (; r + 6 <= R; r += 6) pv_rows<6>(acc + r * D, D, ppack + r * 32, vpack, K2, c0);
                for (; r < R; ++r) pv_rows<1>(acc + r * D, D, ppack + r * 32, vpack, K2, c0);
            }
        }
        for (int64_t r = 0; r < R; ++r) {
            const int64_t qi = r / gqa, g = r % gqa;
            const int64_t ob = (b0 + q0 + qi) * o_rs + (kh * gqa + g) * o_hs;
            float* ar = acc + r * D;
            const float inv = (m[r] == -INFINITY || l[r] <= 0.0f) ? 0.0f : 1.0f / l[r];
            const __m512 vi = _mm512_set1_ps(inv);
            for (int64_t d = 0; d < D; d += 16)
                _mm512_storeu_ps(ar + d, _mm512_mul_ps(_mm512_loadu_ps(ar + d), vi));
            out_row(o, ob, D, ar);
        }
    }
    return true;
}
#endif

/* ------------------------------------------------------------------ attn_dense */
/* The vision tower's: nothing paged, nothing quantised, `cu_seqlens` bounding one image (or one
 * attention window) per entry so a whole batch is ONE launch rather than a per-segment loop and a
 * concatenate. Causal is a parameter because a dense decoder-only prefill wants the same kernel.
 *
 * Same online softmax as the paged form, and the tile is over the segment rather than the page
 * table -- so the K and V rows are consecutive in memory and the hardware prefetcher sees a single
 * stream, which is the one thing the dense form has that the paged one cannot. */
AVX_KERNEL(attn_dense) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3], *o = &a->t[4];

    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa = p_int(a, "gqa", 1);
    const int causal = (int)p_int(a, "causal", 0);
    /* libref's window: causal, i - j < window; non-causal, |i - j| < window; 0 none. */
    const int64_t window = p_int(a, "window", 0);
    if (head_dim <= 0 || gqa <= 0 || window < 0) return RAD_E_SHAPE;
    if (head_dim > AVX_HD_MAX) return RAD_E_UNSUPPORTED;

    /* A rank-2 operand is [tokens, heads * head_dim], so its head count is its width over the
     * head; reading it as one head would take every head of a multi-head slice for one long one. */
    const int64_t q_heads = q->rank >= 3 ? q->shape[q->rank - 2]
                          : (q->rank == 2 ? q->shape[1] / head_dim : 1);
    const int64_t kv_heads = q_heads / gqa;
    if (kv_heads <= 0 || q_heads % gqa != 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t total_q = numel(q) / (q_heads * head_dim);
    if (numel(o) < total_q * q_heads * head_dim) return RAD_E_SHAPE;
    if (numel(k) < total_q * kv_heads * head_dim) return RAD_E_SHAPE;
    if (numel(v) < total_q * kv_heads * head_dim) return RAD_E_SHAPE;

    /* libref's reading of all four, at whatever rank they arrived: a rank-2 operand is a column
     * slice whose row stride is its own (avx_common.h: row_stride). */
    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t v_rs = row_stride(v, kv_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);

    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
#if AVX_LEVEL == 3
    if (attn_dense_bf16(q, k, v, cu, o, head_dim, gqa, kv_heads, causal, window, scale, q_rs,
                        q_hs, q_ds, k_rs, k_hs, k_ds, v_rs, v_hs, v_ds, o_rs, o_hs, o_ds))
        return RAD_OK;
#endif

    AVX_PARALLEL_FOR_IF(n_seq * q_heads >= 2)
    for (int64_t sh = 0; sh < n_seq * q_heads; ++sh) {
        const int64_t s = sh / q_heads, h = sh % q_heads, kh = h / gqa;
        const int64_t b0 = ldi(cu, offlin(cu, s));
        const int64_t b1 = ldi(cu, offlin(cu, s + 1));
        float* frame = (float*)scratch_f32((size_t)(4 * head_dim + AVX_KV_TILE) + 16);
        float* qbuf = frame;
        float* kbuf = frame + head_dim;
        float* vbuf = frame + 2 * head_dim;
        float* acc  = frame + 3 * head_dim;
        float* sbuf = frame + 4 * head_dim;
        int64_t vbase[AVX_KV_TILE];

        for (int64_t i = b0; i < b1; ++i) {
            const int64_t qb = i * q_rs + h * q_hs, ob = i * o_rs + h * o_hs;
            int64_t jbeg = b0, jend = causal ? (i + 1) : b1;
            if (window > 0) {
                jbeg = std::max<int64_t>(b0, i - window + 1);
                if (!causal) jend = std::min<int64_t>(b1, i + window);
            }
            const float* qv = head_f32(q, qb, q_ds, head_dim, 1.0f, qbuf);
            for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            float m = -INFINITY, l = 0.0f;
            int ntile = 0;

            auto flush = [&]() {
                if (ntile == 0) return;
                float tmax = sbuf[0];
                for (int t = 1; t < ntile; ++t) if (sbuf[t] > tmax) tmax = sbuf[t];
                if (tmax > m) {
                    const float r = (m == -INFINITY) ? 0.0f : std::exp(m - tmax);
                    if (r != 1.0f) {
                        const vf vr = vf_set1(r);
                        int64_t d = 0;
                        for (; d + VF_N <= head_dim; d += VF_N)
                            vf_storeu(acc + d, vf_mul(vf_loadu(acc + d), vr));
                        for (; d < head_dim; ++d) acc[d] *= r;
                        l *= r;
                    }
                    m = tmax;
                }
                /* The tile's exps, vectorised (see the prefill flush above): the
                 * accumulation keeps its order. */
                {
                    const vf vm = vf_set1(m);
                    int t = 0;
                    for (; t + VF_N <= ntile; t += VF_N)
                        vf_storeu(sbuf + t, vf_exp(vf_sub(vf_loadu(sbuf + t), vm)));
                    for (; t < ntile; ++t) sbuf[t] = std::exp(sbuf[t] - m);
                }
                for (int t = 0; t < ntile; ++t) {
                    const float w = sbuf[t];
                    l += w;
                    axpy(acc, head_f32(v, vbase[t], v_ds, head_dim, 1.0f, vbuf), w, head_dim);
                }
                ntile = 0;
            };

            for (int64_t j = jbeg; j < jend; ++j) {
                const float* kvr = head_f32(k, j * k_rs + kh * k_hs, k_ds, head_dim, 1.0f, kbuf);
                sbuf[ntile] = dotf(qv, kvr, head_dim) * scale;
                vbase[ntile] = j * v_rs + kh * v_hs;
                if (++ntile == AVX_KV_TILE) flush();
            }
            flush();

            if (m == -INFINITY) {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                continue;
            }
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            if (o_ds == 1) {
                const vf vi = vf_set1(inv);
                int64_t d = 0;
                for (; d + VF_N <= head_dim; d += VF_N)
                    vf_storeu(acc + d, vf_mul(vf_loadu(acc + d), vi));
                for (; d < head_dim; ++d) acc[d] *= inv;
                out_row(o, ob, head_dim, acc);
            } else {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_local_rel */
/* libref's local attention with a relative-position term (Gemma 4's audio). A query sees at most
 * left + 1 keys -- thirteen at the checkpoint's geometry -- so the work is a few dot products a
 * (row, head) and the loop is libref's, vectorised along the head and parallel over (row, head). */
AVX_KERNEL(attn_local_rel) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *rk = &a->t[3];
    const RadTensor *pds = &a->t[4], *cu = &a->t[5], *o = &a->t[6];
    const int64_t D = p_int(a, "head_dim", 0), H = p_int(a, "n_head", 0);
    const int64_t L = p_int(a, "left", 0), R = p_int(a, "right", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1);
    const float cap = p_f32(a, "softcap", 0.0f);
    if (D <= 0 || H <= 0 || L < 0 || cdiv <= 0 || !(cap > 0.0f)) return RAD_E_SHAPE;
    if (R != 0) return RAD_E_UNSUPPORTED;
    if (D > AVX_HD_MAX || L + 1 > AVX_HD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t Wd = H * D;
    if (numel(pds) < D || numel(rk) < (L + 1) * Wd) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t Mq = numel(q) / Wd, Mk = numel(k) / Wd, Mv = numel(v) / Wd, Mo = numel(o) / Wd;
    const int64_t q_rs = row_stride(q, Wd), k_rs = row_stride(k, Wd), v_rs = row_stride(v, Wd);
    const int64_t r_rs = row_stride(rk, Wd), o_rs = row_stride(o, Wd);
    const int64_t q_ds = laststride(q), k_ds = laststride(k), v_ds = laststride(v);
    const int64_t r_ds = laststride(rk), o_ds = laststride(o);

    const float qs = (float)(1.0 / std::sqrt((double)D) / std::log(2.0));
    const float ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    /* softplus(pds), rounded to the weight's dtype as the reference evaluates it on the bf16
     * parameter (libref's reading), and the query's whole scale folded into one vector. */
    float* spd = (float*)scratch_raw((size_t)D * sizeof(float) + 64);
    for (int64_t d = 0; d < D; ++d) {
        const float x = ldt(pds, d * laststride(pds));
        const float sp = x > 20.0f ? x : (x > 0.0f ? x + std::log1p(std::exp(-x))
                                                   : std::log1p(std::exp(x)));
        float r = sp;
        if (pds->dtype == RAD_BF16) r = rad_bf16_to_f32(rad_f32_to_bf16(sp));
        else if (pds->dtype == RAD_F16) r = rad_f16_to_f32(rad_f32_to_f16(sp));
        spd[d] = r;
    }

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = ldi(cu, offlin(cu, s)), c1 = ldi(cu, offlin(cu, s + 1));
        if (c0 % cdiv || c1 % cdiv || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, t1 = c1 / cdiv;
        if (t1 > Mq || t1 > Mk || t1 > Mv || t1 > Mo) return RAD_E_SHAPE;

        AVX_PARALLEL_FOR_IF((t1 - t0) * H * (L + 1) * D >= AVX_PAR_MIN)
        for (int64_t ih = 0; ih < (t1 - t0) * H; ++ih) {
            const int64_t i = t0 + ih / H, h = ih % H;
            float* frame = scratch_f32((size_t)(4 * D + L + 1) + 16);
            float* qf = frame;
            float* kb = frame + D;
            float* acc = frame + 2 * D;
            float* tmp = frame + 3 * D;
            float* lg = frame + 4 * D;
            const float* qv = head_f32(q, i * q_rs + h * D * q_ds, q_ds, D, 1.0f, tmp);
            for (int64_t d = 0; d < D; ++d) qf[d] = qv[d] * qs * spd[d];
            const int64_t j0 = i - L > t0 ? i - L : t0;
            float mx = -INFINITY;
            for (int64_t j = j0; j <= i; ++j) {
                const float* kv = head_f32(k, j * k_rs + h * D * k_ds, k_ds, D, 1.0f, tmp);
                for (int64_t d = 0; d < D; ++d) kb[d] = kv[d] * ks;
                float ac = dotf(qf, kb, D);
                const float* rv = head_f32(rk, (i - j) * r_rs + h * D * r_ds, r_ds, D, 1.0f, tmp);
                ac += dotf(qf, rv, D);
                const float x = cap * std::tanh(ac / cap);
                lg[j - j0] = x;
                mx = x > mx ? x : mx;
            }
            float sum = 0.0f;
            for (int64_t j = j0; j <= i; ++j) {
                lg[j - j0] = std::exp(lg[j - j0] - mx);
                sum += lg[j - j0];
            }
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
            for (int64_t j = j0; j <= i; ++j)
                axpy(acc, head_f32(v, j * v_rs + h * D * v_ds, v_ds, D, 1.0f, tmp),
                     lg[j - j0] / sum, D);
            const int64_t ob = i * o_rs + h * D * o_ds;
            if (o_ds == 1) out_row(o, ob, D, acc);
            else for (int64_t d = 0; d < D; ++d) stt(o, ob + d * o_ds, acc[d]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ kv_store */
/* The cache operand is INOUT and not OUT: this op touches only the mapped slots and every other
 * slot in the pool must survive it, which is a liveness fact the buffer planner has to know.
 *
 * A NEGATIVE slot is SKIPPED, and that is a documented value rather than an accident: it is how a
 * padded batch row says "this token has no cache slot". libr4d treats slot <= 0 as null because
 * vLLM reserves block 0; a generic implementation must not steal a valid slot, so only the
 * strictly negative is skipped -- libref's rule, and the two have to agree or a padded step writes
 * different bytes under the two plugins. */
AVX_KERNEL(kv_store) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *v = &a->t[1], *sm = &a->t[2], *kv = &a->t[3];

    const int64_t head_dim = p_int(a, "head_dim", k->rank >= 3 ? k->shape[k->rank - 1] : 0);
    const int64_t block_size = p_int(a, "block_size", 0);
    const int64_t kv_heads = p_int(a, "n_head_kv", kv->rank == 4 ? kv->shape[1] : 0);
    if (head_dim <= 0 || block_size <= 0 || kv_heads <= 0) return RAD_E_SHAPE;
    if (kv->rank == 4 && (kv->shape[1] != kv_heads || kv->shape[2] != block_size ||
                          kv->shape[3] < 2 * head_dim)) return RAD_E_SHAPE;

    const int64_t M = numel(sm);
    if (M <= 0 || numel(k) < M * kv_heads * head_dim) return RAD_E_SHAPE;
    const int64_t n_blocks = kv->rank == 4 ? kv->shape[0] : 0;

    /* `k` AND `v` ARRIVE RANK 2 FROM AN ARCHITECTURE THAT DOES NOT DE-INTERLEAVE ITS PROJECTION --
     * a column slice of the fused QKV plane, whose row stride is the whole projection's width.
     * Deriving the stride from the shape puts token 0 in the right slot and fills every later
     * token's slot out of the previous token's query rows. libref's reading, through row_stride. */
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t v_rs = row_stride(v, kv_heads * head_dim);
    const KVView kvv = kv_view(kv, head_dim, block_size, kv_heads);

    /* Validated before the loop, because an OpenMP body cannot return. */
    for (int64_t t = 0; t < M; ++t) {
        const int64_t slot = ldi(sm, offlin(sm, t));
        if (slot >= 0 && n_blocks > 0 && slot / block_size >= n_blocks) return RAD_E_SHAPE;
    }

    AVX_PARALLEL_FOR_IF(M * kv_heads * head_dim >= AVX_PAR_MIN)
    for (int64_t t = 0; t < M; ++t) {
        const int64_t slot = ldi(sm, offlin(sm, t));
        if (slot < 0) continue;
        const int64_t blk = slot / block_size, off = slot % block_size;
        float* buf = scratch_f32((size_t)head_dim);
        for (int64_t h = 0; h < kv_heads; ++h) {
            const int64_t cb = blk * kvv.sb + h * kvv.sh + off * kvv.so;
            /* The head at a time, through the row converter, so a bf16 cache fed a bf16 activation
             * is a widen and a narrow of a whole head rather than 2*head_dim dtype switches. The
             * strided case falls back per element, which is what a non-unit cache stride means. */
            if (k_ds == 1 && kvv.sd == 1) {
                row_to_f32(k->dtype, byte_at(k->data, k->dtype, t * k_rs + h * k_hs), buf, head_dim);
                out_row(kv, cb, head_dim, buf);
            } else {
                for (int64_t d = 0; d < head_dim; ++d)
                    stt(kv, cb + d * kvv.sd, ldt(k, t * k_rs + h * k_hs + d * k_ds));
            }
            if (v_ds == 1 && kvv.sd == 1) {
                row_to_f32(v->dtype, byte_at(v->data, v->dtype, t * v_rs + h * v_hs), buf, head_dim);
                out_row(kv, cb + head_dim, head_dim, buf);
            } else {
                for (int64_t d = 0; d < head_dim; ++d)
                    stt(kv, cb + (head_dim + d) * kvv.sd, ldt(v, t * v_rs + h * v_hs + d * v_ds));
            }
        }
    }
    return RAD_OK;
}

/* ================================================================== rotary
 *
 * NeoX pairs element i with i + rotary_dim/2; GPT-J (the "interleaved" style) pairs 2i with 2i+1.
 * Which one a checkpoint wants is a property of the checkpoint, so it is a parameter and not a
 * guess -- and getting it wrong produces fluent text with a broken sense of position, which is the
 * hardest kind of wrong to notice.
 *
 * THE ANGLE TABLE IS THE WHOLE OPTIMISATION HERE. libref computes
 * `1/pow(theta, 2i/rot)` and then a cos and a sin FOR EVERY (token, head, i) -- three libm calls
 * per element, on an op that otherwise does two multiplies and an add. But the angle depends only
 * on (position, i): every head of a token wants the same cos/sin. Building the row's table once
 * and reusing it across all n_head + n_head_kv heads removes (heads - 1)/heads of the transcendental
 * work, which at 32 heads is 97% of it, and the values are identical because they are the same
 * computation performed once instead of many times -- so this is bit-identical, not merely close.
 *
 * The inverse frequencies depend only on `i` and are built ONCE PER LAUNCH for the same reason.
 *
 * What is NOT vectorised is the cos/sin themselves: a vector sincos is a polynomial with its own
 * accuracy, and `rope` feeds a position signal where a fraction of a ULP is exactly the error that
 * produces fluent text with wrong positions. The table makes them rare enough not to matter. */
/* The multi-component modes (docs/OPS.md, `rope`): which position component drives each of the
 * rot/2 frequencies, and for `axial` a ladder per section. libref's rope_freqs states the same
 * rule; the two are checked against each other by avx_check. */
enum { ROPE_NEOX = 0, ROPE_GPTJ = 1, ROPE_MROPE = 2, ROPE_IMROPE = 3, ROPE_AXIAL = 4 };

/* cos in [0, half), sin in [half, rot). Same layout `rope_table` writes, deliberately.
 *
 * NOT sincosf: it is bit-identical to the two separate calls over a sweep of millions of angles,
 * but measures SLOWER than them, because glibc's joint wrapper costs more than two fast-path
 * calls. */
static void angles(float* cs, const float* invf, int64_t half, float pos) {
    for (int64_t i = 0; i < half; ++i) {
        const float ang = pos * invf[i];
        cs[i] = std::cos(ang);
        cs[half + i] = std::sin(ang);
    }
}

/* ...and with a position per component, `comp[i]` naming the one frequency i reads. */
static void angles_multi(float* cs, const float* invf, const int* comp, int64_t half,
                         const float* pos) {
    for (int64_t i = 0; i < half; ++i) {
        const float ang = pos[comp[i]] * invf[i];
        cs[i] = std::cos(ang);
        cs[half + i] = std::sin(ang);
    }
}

static void rope_head(const RadTensor* t, int64_t base, int64_t ds, int64_t rot, int mode,
                      const float* cs, int64_t half, float* buf) {
    if (mode == ROPE_NEOX && ds == 1) {
        /* x0 and x1 are two contiguous runs of `half`, so the whole rotation is vector work. */
        row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), buf, rot);
        int64_t i = 0;
        for (; i + VF_N <= half; i += VF_N) {
            const vf x0 = vf_loadu(buf + i), x1 = vf_loadu(buf + half + i);
            const vf c = vf_loadu(cs + i), s = vf_loadu(cs + half + i);
            vf_storeu(buf + i, vf_sub(vf_mul(x0, c), vf_mul(x1, s)));
            vf_storeu(buf + half + i, vf_add(vf_mul(x0, s), vf_mul(x1, c)));
        }
        for (; i < half; ++i) {
            const float x0 = buf[i], x1 = buf[half + i];
            buf[i] = x0 * cs[i] - x1 * cs[half + i];
            buf[half + i] = x0 * cs[half + i] + x1 * cs[i];
        }
        out_row(t, base, rot, buf);
        return;
    }
    /* GPT-J's interleave, and every strided case. The pairs are adjacent, so a vector form needs a
     * de-interleave and a re-interleave for two multiplies -- not worth the code on a layout only
     * older checkpoints use. */
    for (int64_t i = 0; i < half; ++i) {
        const int64_t i0 = (mode == ROPE_GPTJ) ? 2 * i : i;
        const int64_t i1 = (mode == ROPE_GPTJ) ? 2 * i + 1 : i + half;
        const float x0 = ldt(t, base + i0 * ds), x1 = ldt(t, base + i1 * ds);
        const float c = cs[i], s = cs[half + i];
        stt(t, base + i0 * ds, x0 * c - x1 * s);
        stt(t, base + i1 * ds, x0 * s + x1 * c);
    }
}

static float* inv_freqs(int64_t half, int64_t rot, float theta) {
    float* f = (float*)scratch_raw((size_t)half * sizeof(float));
    for (int64_t i = 0; i < half; ++i)
        f[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)rot);
    return f;
}

AVX_KERNEL(rope) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *pos = &a->t[1];

    const int64_t head_dim = p_int(a, "head_dim", 0);
    const int64_t n_head = p_int(a, "n_head", 0);
    const int64_t n_head_kv = p_int(a, "n_head_kv", 0);
    const float theta = p_f32(a, "theta", 10000.0f);
    const float rscale = p_f32(a, "scale", 1.0f);
    static const char* kModes[] = { "neox", "gptj", "mrope", "imrope", "axial" };
    const int mode = p_enum(a, "mode", kModes, 5, -1);
    if (mode < 0) return RAD_E_UNSUPPORTED;
    const int64_t rot = p_int(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0 || rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;

    /* [M] one component a token, or [C, M] component-major with `sections`. */
    int64_t sec[8] = {};
    int n_sec = 0;
    for (const char* v = p_str(a, "sections", nullptr); v && *v && n_sec < 8;) {
        char* e = nullptr;
        const long long x = std::strtoll(v, &e, 10);
        if (e == v) break;
        if (x <= 0) return RAD_E_SHAPE;
        sec[n_sec++] = x;
        v = e;
    }
    const bool multi = pos->rank >= 2;
    const int64_t n_comp = multi ? pos->shape[0] : 1;
    if (multi && (mode <= ROPE_GPTJ || n_sec <= 0 || n_comp > 8 ||
                  n_comp < (mode == ROPE_IMROPE ? 3 : n_sec)))
        return RAD_E_SHAPE;

    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    const int64_t M = multi ? pos->shape[1] : numel(pos);
    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = qkv->rank >= 2 ? qkv->stride[qkv->rank - 2] : stride_row;
    const int64_t ds = laststride(qkv);
    const int64_t half = rot / 2;
    float* invf = inv_freqs(half, rot, theta);
    int comp[512] = {};
    if (half > 512) return RAD_E_UNSUPPORTED;
    if (multi) {
        int64_t start[8] = {}, total = 0;
        for (int c = 0; c < n_sec; ++c) { start[c] = total; total += sec[c]; }
        if ((mode == ROPE_MROPE || mode == ROPE_AXIAL) && total != half) return RAD_E_SHAPE;
        for (int64_t i = 0; i < half; ++i) {
            int c = 0;
            if (mode == ROPE_MROPE || mode == ROPE_AXIAL) {
                while (c + 1 < n_sec && i >= start[c + 1]) ++c;
                if (mode == ROPE_AXIAL)
                    invf[i] = 1.0f / std::pow(theta, (float)(2 * (i - start[c])) /
                                                     (float)(2 * sec[c]));
            } else if (mode == ROPE_IMROPE) {
                if (i % 3 == 1 && i < 3 * sec[1]) c = 1;
                else if (i % 3 == 2 && i < 3 * sec[2]) c = 2;
            }
            comp[i] = c;
        }
    }
    const int pair_mode = mode == ROPE_GPTJ ? ROPE_GPTJ : ROPE_NEOX;
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn)) return RAD_E_INVAL;
    if (yarn.factor != 0.0f) {
        if (mode == ROPE_AXIAL || rscale != 1.0f) return RAD_E_UNSUPPORTED;
        for (int64_t i = 0; i < half; ++i) invf[i] = yarn.frequency(invf[i], (int)i);
    }
    const int64_t pcs = multi ? pos->stride[0] : 0, pms = multi ? pos->stride[1] : 0;

    AVX_PARALLEL_FOR_IF(M * (n_head + n_head_kv) * rot >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const float div = rscale != 0.0f ? rscale : 1.0f;
        const int64_t row = m * rs;
        float* cs = scratch_f32((size_t)rot);
        float* buf = scratch_f32_b((size_t)rot);
        if (multi) {
            float pc[8];
            for (int64_t c = 0; c < n_comp; ++c) pc[c] = (float)ldi(pos, c * pcs + m * pms) / div;
            angles_multi(cs, invf, comp, half, pc);
        } else {
            angles(cs, invf, half, (float)ldi(pos, offlin(pos, m)) / div);
        }
        if (yarn.factor != 0.0f)
            for (int64_t i = 0; i < rot; ++i) cs[i] *= yarn.magnitude;
        /* Q heads then K heads. The V heads that follow are left alone, which is the whole reason
         * this op takes the fused qkv rather than two tensors. */
        for (int64_t h = 0; h < n_head + n_head_kv; ++h)
            rope_head(qkv, row + h * head_dim * ds, ds, rot, pair_mode, cs, half, buf);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qk_norm_rope */
/* An RMS norm over each head against its own gain, then RoPE.
 *
 * THE NORM OUTPUT IS ROUNDED TO THE TENSOR'S DTYPE BEFORE THE ROTATION. That is not an accident of
 * writing it back in place -- it is what the unfused pair does, because the norm stores a tensor
 * and the rotation reads it, and libr4d's r4d_qk_norm_rope_gate matches the unfused reference for
 * exactly this reason. A fused kernel that kept f32 across the seam would be half an ulp better
 * and would fail a comparison against this. So the norm is written with out_row and the rotation
 * re-reads it, rather than being carried in registers. */
AVX_KERNEL(qk_norm_rope) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *pos = &a->t[1], *qw = &a->t[2], *kw = &a->t[3];

    const int64_t head_dim = p_int(a, "head_dim", 0);
    const int64_t n_head = p_int(a, "n_head", 0);
    const int64_t n_head_kv = p_int(a, "n_head_kv", 0);
    const float theta = p_f32(a, "theta", 10000.0f);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float rscale = p_f32(a, "scale", 1.0f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const int v_norm = (int)p_int(a, "v_norm", 0);
    static const char* kModes[] = { "neox", "gptj", "blocks" };
    const int mode = p_enum(a, "mode", kModes, 3, ROPE_NEOX);
    if (mode < 0) return RAD_E_UNSUPPORTED;
    const bool blocks = mode == 2;
    const int64_t rot = p_int(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0 || rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;
    if (numel(qw) < head_dim || numel(kw) < head_dim) return RAD_E_SHAPE;

    /* BLOCKS (libref): section c is sec[c] pairs over its own 2 * sec[c] channels, paired
     * (start + j, start + sec[c] + j), turned by position component comps[c]. */
    int64_t sec[8] = {}, comp[8] = {};
    int n_sec = 0;
    if (blocks) {
        for (const char* v = p_str(a, "sections", nullptr); v && *v && n_sec < 8; ) {
            char* end = nullptr;
            const long long x = std::strtoll(v, &end, 10);
            if (end == v) break;
            if (x <= 0) return RAD_E_SHAPE;
            comp[n_sec] = n_sec;
            sec[n_sec++] = x;
            v = end;
        }
        int64_t total = 0;
        for (int c = 0; c < n_sec; ++c) total += sec[c];
        if (n_sec <= 0 || 2 * total != rot) return RAD_E_SHAPE;
        if (const char* cs = p_str(a, "comps", nullptr)) {
            int k = 0;
            for (const char* v = cs; *v && k < 8; ) {
                char* end = nullptr;
                const long long x = std::strtoll(v, &end, 10);
                if (end == v) break;
                comp[k++] = x;
                v = end;
            }
            if (k != n_sec) return RAD_E_SHAPE;
        }
    }
    const bool multi = pos->rank >= 2;
    const int64_t n_comp = multi ? pos->shape[0] : 1;
    if (multi && !blocks) return RAD_E_SHAPE;
    for (int c = 0; c < n_sec; ++c)
        if (comp[c] < 0 || comp[c] >= n_comp || n_comp > 8) return RAD_E_SHAPE;
    const int64_t pcs = multi ? pos->stride[0] : 0, pms = multi ? pos->stride[1] : 0;

    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    const int64_t M = multi ? pos->shape[1] : numel(pos);
    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = qkv->rank >= 2 ? qkv->stride[qkv->rank - 2] : stride_row;
    const int64_t ds = laststride(qkv);
    const int64_t half = rot / 2;

    /* One arena call for the launch-invariant tables: the inverse frequencies (and, for blocks,
     * each pair's component) and both widened gains. A second scratch_raw would reallocate under
     * the parallel region. */
    float* tab = (float*)scratch_raw((size_t)(2 * half + 2 * head_dim) * sizeof(float) + 64);
    float* invf = tab;
    float* pcomp = tab + half;
    float* gq = tab + 2 * half;
    float* gk = tab + 2 * half + head_dim;
    if (!blocks) {
        for (int64_t i = 0; i < half; ++i)
            invf[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)rot);
    } else {
        int64_t i = 0;
        for (int c = 0; c < n_sec; ++c)
            for (int64_t j = 0; j < sec[c]; ++j, ++i) {
                invf[i] = 1.0f / std::pow(theta, (float)j / (float)sec[c]);
                pcomp[i] = (float)comp[c];
            }
    }
    for (int64_t d = 0; d < head_dim; ++d) {
        gq[d] = ldt(qw, d * laststride(qw)) + wadd;
        gk[d] = ldt(kw, d * laststride(kw)) + wadd;
    }

    AVX_PARALLEL_FOR_IF(M * (n_head + n_head_kv) * head_dim >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const float div = rscale != 0.0f ? rscale : 1.0f;
        float pc[8];
        if (multi)
            for (int64_t c = 0; c < n_comp; ++c) pc[c] = (float)ldi(pos, c * pcs + m * pms) / div;
        else
            pc[0] = (float)ldi(pos, offlin(pos, m)) / div;
        const int64_t row = m * rs;
        float* cs = scratch_f32((size_t)rot);
        float* buf = scratch_f32_b((size_t)head_dim);
        float* rbuf = scratch_f32_c((size_t)head_dim);
        if (!blocks) angles(cs, invf, half, pc[0]);
        else
            for (int64_t i = 0; i < half; ++i) {
                const float ang = pc[(int)pcomp[i]] * invf[i];
                cs[i] = std::cos(ang);
                cs[half + i] = std::sin(ang);
            }
        for (int64_t h = 0; h < n_head + n_head_kv; ++h) {
            const float* g = h < n_head ? gq : gk;
            const int64_t base = row + h * head_dim * ds;
            const float* xv = head_f32(qkv, base, ds, head_dim, 1.0f, buf);
            const float sc = 1.0f / std::sqrt(row_sumsq(xv, head_dim) / (float)head_dim + eps);
            const vf vs = vf_set1(sc);
            int64_t d = 0;
            for (; d + VF_N <= head_dim; d += VF_N)
                vf_storeu(rbuf + d, vf_mul(vf_mul(vf_loadu(xv + d), vs), vf_loadu(g + d)));
            for (; d < head_dim; ++d) rbuf[d] = xv[d] * sc * g[d];
            /* Written back through the operand's dtype, then re-read by the rotation -- the
             * rounding across the seam is the contract. */
            if (ds == 1) out_row(qkv, base, head_dim, rbuf);
            else for (int64_t i = 0; i < head_dim; ++i) stt(qkv, base + i * ds, rbuf[i]);
            if (!blocks) {
                rope_head(qkv, base, ds, rot, mode, cs, half, rbuf);
                continue;
            }
            int64_t i = 0, start = 0;
            for (int c = 0; c < n_sec; ++c) {
                for (int64_t j = 0; j < sec[c]; ++j, ++i) {
                    const int64_t i0 = start + j, i1 = start + sec[c] + j;
                    const float x0 = ldt(qkv, base + i0 * ds), x1 = ldt(qkv, base + i1 * ds);
                    stt(qkv, base + i0 * ds, x0 * cs[i] - x1 * cs[half + i]);
                    stt(qkv, base + i1 * ds, x0 * cs[half + i] + x1 * cs[i]);
                }
                start += 2 * sec[c];
            }
        }
        /* v_norm: the v heads RMS-normed with a unit gain and not rotated (libref). */
        for (int64_t h = 0; v_norm && h < n_head_kv; ++h) {
            const int64_t base = row + (n_head + n_head_kv + h) * head_dim * ds;
            const float* xv = head_f32(qkv, base, ds, head_dim, 1.0f, buf);
            const float sc = 1.0f / std::sqrt(row_sumsq(xv, head_dim) / (float)head_dim + eps);
            for (int64_t d = 0; d < head_dim; ++d) rbuf[d] = xv[d] * sc;
            if (ds == 1) out_row(qkv, base, head_dim, rbuf);
            else for (int64_t d = 0; d < head_dim; ++d) stt(qkv, base + d * ds, rbuf[d]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope_table */
/* The cos/sin plane a fused prologue reads instead of theta, written at each position's OWN row of
 * a [max_ctx, rot] table: `rot` values a row, cos for the first rot/2 and sin for the second.
 *
 * THE ANGLES ARE `rope`'s, TO THE LAST OPERATION, and that is the whole contract: a fused rotation
 * reading a table built any other way would be a different model by a fraction of a ULP per
 * element -- fluent text, subtly wrong positions. So the two share `inv_freqs` and `angles` rather
 * than each computing the angle their own way. Only the rows `positions` names are written; a
 * negative position is a padded row and writes nothing. */
AVX_KERNEL(rope_table) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *pos = &a->t[0], *cs = &a->t[1];
    const int64_t rot = p_int(a, "rot", 0);
    const float theta = p_f32(a, "theta", 10000.0f);
    const float rscale = p_f32(a, "scale", 1.0f);
    if (rot < 2 || (rot & 1)) return RAD_E_SHAPE;
    const int64_t M = numel(pos);
    if (M <= 0 || numel(cs) % rot) return RAD_E_SHAPE;
    const int64_t rows = numel(cs) / rot;
    const int64_t half = rot / 2;
    float* invf = inv_freqs(half, rot, theta);
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && rscale != 1.0f)) return RAD_E_INVAL;
    if (yarn.factor != 0.0f)
        for (int64_t i = 0; i < half; ++i) invf[i] = yarn.frequency(invf[i], (int)i);

    AVX_PARALLEL_FOR_IF(M * rot >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t pp = ldi(pos, offlin(pos, m));
        if (pp < 0 || pp >= rows) continue;
        const float p = (float)pp / (rscale != 0.0f ? rscale : 1.0f);
        float* buf = scratch_f32((size_t)rot);
        angles(buf, invf, half, p);
        if (yarn.factor != 0.0f)
            for (int64_t i = 0; i < rot; ++i) buf[i] *= yarn.magnitude;
        /* `cs` may be strided, and offlin has to be evaluated per element there -- which is what
         * libref does unconditionally. The dense case is one row store. */
        if (rad_tensor_is_contiguous(cs)) {
            out_row(cs, pp * rot, rot, buf);
        } else {
            for (int64_t i = 0; i < rot; ++i) stt(cs, offlin(cs, pp * rot + i), buf[i]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_block_key */
/* QSA's compressed key (docs/OPS.md): a block's `ratio` rows of k averaged in f32 and rounded to
 * k's dtype, RMS-normed with gain (wadd + w), then NeoX-rotated over `rotary_dim` at the block's
 * position, YaRN included. libref's arithmetic in libref's order; what differs is that the angles
 * of a block come from one `angles` table, and the inverse frequencies are built once a launch. */
AVX_KERNEL(qsa_block_key) {
    const RadTensor* k = rad_arg_in(a, 0);
    const RadTensor* w = rad_arg_in(a, 1);
    const RadTensor* page = rad_arg_in(a, 2);
    const RadTensor* pos = rad_arg_in(a, 3);
    const RadTensor* out = rad_arg_in(a, 4);
    if (!k || !w || !out) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    const int64_t rot = p_int(a, "rotary_dim", 0), pos0 = p_int(a, "pos0", -1);
    if (n <= 0 || ratio <= 0 || (pos0 < 0 && !pos)) return RAD_E_INVAL;
    if (rot < 0 || rot > n || (rot & 1) || rot / 2 > 512) return RAD_E_SHAPE;
    if (p_int(a, "block_size", ratio) != ratio) return RAD_E_SHAPE;
    if (k->rank != 2 || out->rank < 2 || out->shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = page ? numel(page) : out->shape[0];
    if (M < 1 || numel(out) / out->shape[0] != n) return RAD_E_SHAPE;
    if (k->shape[0] != M * ratio || k->shape[1] != n || numel(w) < n) return RAD_E_SHAPE;
    const bool pos2 = pos && pos->rank == 2;
    if (pos && (pos2 ? pos->shape[0] < M : numel(pos) < M)) return RAD_E_SHAPE;

    const float eps = p_f32(a, "eps", 1e-6f), wadd = p_f32(a, "wadd", 1.0f);
    const float theta = p_f32(a, "theta", 10000.0f), rscale = p_f32(a, "scale", 1.0f);
    static const char* kModes[] = { "neox", "mrope", "imrope" };
    const int mi = p_enum(a, "mode", kModes, 3, 0);
    if (mi < 0) return RAD_E_UNSUPPORTED;
    int64_t sec[3] = {};
    int n_sec = 0;
    for (const char* v = p_str(a, "sections", nullptr); v && *v;) {
        char* e = nullptr;
        const long long x = std::strtoll(v, &e, 10);
        if (e == v) break;
        if (x <= 0 || n_sec >= 3) return RAD_E_SHAPE;
        sec[n_sec++] = x;
        v = e;
    }
    /* Several components only with a mode, its sections and three columns of `pos`. */
    const bool mc = mi != 0 && n_sec > 0 && pos2 && pos->shape[1] >= 3;
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && (rot <= 0 || rscale != 1.0f)))
        return RAD_E_INVAL;
    const int64_t half = rot / 2;
    float* invf = inv_freqs(half, rot, theta);
    int comp[512] = {};
    if (mc) {
        int64_t total = 0;
        for (int c = 0; c < n_sec; ++c) total += sec[c];
        if (mi == 1 && total != half) return RAD_E_SHAPE;
        for (int64_t i = 0; i < half; ++i) {
            int c = 0;
            if (mi == 1) {
                int64_t start = 0;
                while (c + 1 < n_sec && i >= start + sec[c]) { start += sec[c]; ++c; }
            } else if (n_sec >= 3) {
                if (i % 3 == 1 && i < 3 * sec[1]) c = 1;
                else if (i % 3 == 2 && i < 3 * sec[2]) c = 2;
            }
            comp[i] = c;
        }
    }
    for (int64_t i = 0; i < half; ++i) invf[i] = yarn.frequency(invf[i], (int)i);
    float* gain = (float*)scratch_f32_c((size_t)n);
    for (int64_t t = 0; t < n; ++t) gain[t] = wadd + ldt(w, offlin(w, t));

    AVX_PARALLEL_FOR_IF(M * ratio * n >= AVX_PAR_MIN)
    for (int64_t b = 0; b < M; ++b) {
        const int64_t orow = page ? ldi(page, offlin(page, b)) : b;
        if (orow < 0) continue;
        float* y = scratch_f32((size_t)n);
        float* cs = scratch_f32_b((size_t)(rot > 0 ? rot : 1));
        double ss = 0.0;
        for (int64_t t = 0; t < n; ++t) {
            float acc = 0.0f;
            for (int64_t j = 0; j < ratio; ++j) acc += ldt(k, offlin(k, (b * ratio + j) * n + t));
            uint64_t tmp = 0;
            st_elem(&tmp, k->dtype, 0, acc / (float)ratio);
            y[t] = ld_elem(&tmp, k->dtype, 0);
            ss += (double)y[t] * y[t];
        }
        const float s = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
        for (int64_t t = 0; t < n; ++t) y[t] = y[t] * s * gain[t];
        const float div = rscale != 0.0f ? rscale : 1.0f;
        float pc[3] = {};
        for (int c = 0; c < (mc ? 3 : 1); ++c)
            pc[c] = (float)(!pos ? pos0 + b * ratio
                                 : ldi(pos, offlin(pos, pos2 ? b * pos->shape[1] + c : b))) / div;
        if (mc) angles_multi(cs, invf, comp, half, pc);
        else    angles(cs, invf, half, pc[0]);
        for (int64_t i = 0; i < half; ++i) {
            const float c = cs[i] * yarn.magnitude, sn = cs[half + i] * yarn.magnitude;
            const float x0 = y[i], x1 = y[half + i];
            y[i] = x0 * c - x1 * sn;
            y[half + i] = x0 * sn + x1 * c;
        }
        for (int64_t t = 0; t < n; ++t) stt(out, offlin(out, orow * n + t), y[t]);
    }
    return RAD_OK;
}
