/* avx_harness.h -- opening a plugin, sizing its operands, drawing them, and the geometry table.
 *
 * SHARED BY rad-avx-check AND rad-avx-bench, AND THAT SHARING IS THE POINT. A benchmark taken at a
 * different extent from the correctness check is a measurement of something else, and two copies
 * of a geometry table drift the moment one of them gains a case. tools/opshapes.h makes exactly
 * this argument for rad-kbench and rad-tune; the same argument applies one level down.
 *
 * Header-only and included by two translation units, so everything here is `static` or `inline`.
 * Neither binary links either plugin: both dlopen by path, which is what makes the claim be about
 * the .so the engine would load rather than about a differently-inlined copy of the same source.
 */
#ifndef RAD_AVX_HARNESS_H
#define RAD_AVX_HARNESS_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ================================================================== a loaded plugin */
struct Lib {
    void* h = nullptr;
    std::string path;
    const RadPluginInfo* info = nullptr;
    int (*kernel_count)() = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    int (*schema_count)() = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*plugin_open)() = nullptr;
    /* libavx's own, and absent from any other plugin -- which is how a caller learns whether it
     * can sweep ISA levels at all rather than assuming it is looking at libavx. */
    int (*isa_force)(int) = nullptr;
    int (*isa_detect)() = nullptr;
    const char* (*isa_name)(int) = nullptr;

    bool open(const char* p) {
        path = p;
        h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", p, dlerror()); return false; }
        auto sym = [&](const char* n) { return dlsym(h, n); };
        auto abi = (uint32_t (*)())sym("rad_plugin_abi_version");
        auto inf = (const RadPluginInfo* (*)())sym("rad_plugin_info");
        kernel_count = (int (*)())sym("rad_kernel_count");
        kernel_at    = (const RadKernelInfo* (*)(int))sym("rad_kernel_at");
        schema_count = (int (*)())sym("rad_kernel_schema_count");
        schema_at    = (const RadOpSchema* (*)(int))sym("rad_kernel_schema_at");
        plugin_open  = (int (*)())sym("rad_plugin_open");
        isa_force    = (int (*)(int))sym("avx_isa_force");
        isa_detect   = (int (*)())sym("avx_isa_detect");
        isa_name     = (const char* (*)(int))sym("avx_isa_name");
        if (!abi || !inf || !kernel_count || !kernel_at || !schema_count || !schema_at) {
            std::fprintf(stderr, "%s: not a kernel plugin (missing exports)\n", p);
            return false;
        }
        if (abi() != RAD_ABI_VERSION) {
            /* THE LOADER'S RULE, REPRODUCED: a plugin that does not report exactly this version is
             * refused by name with both versions printed. A harness that shrugged at it would be
             * measuring a plugin the engine will not load. */
            std::fprintf(stderr, "%s: ABI %u, this tool is built for %u\n", p, abi(),
                         RAD_ABI_VERSION);
            return false;
        }
        info = inf();
        if (plugin_open && plugin_open() < 0) {
            std::fprintf(stderr, "%s: rad_plugin_open failed\n", p);
            return false;
        }
        return true;
    }

    const RadKernelInfo* row(const char* op) const {
        for (int i = 0, n = kernel_count(); i < n; ++i) {
            const RadKernelInfo* k = kernel_at(i);
            if (k && k->op && std::strcmp(k->op, op) == 0) return k;
        }
        return nullptr;
    }
    const RadOpSchema* schema(const char* op) const {
        for (int i = 0, n = schema_count(); i < n; ++i) {
            const RadOpSchema* s = schema_at(i);
            if (s && s->op && std::strcmp(s->op, op) == 0) return s;
        }
        return nullptr;
    }
};

/* ================================================================== the draw
 *
 * The laws are RadOpdDesc's (rad_abi.h) and they are not decoration. A gated-delta-net `g` is an
 * intra-chunk cumulative sum of log-decays; fed normals, the ORACLE overflows to 2e38 while the
 * kernel does something else and the comparison measures the test rather than the kernel. The same
 * is true of a sigmoid gate, a unit-triangular factor, a unit-norm key, and of every index
 * operand -- a KV slot indexes n_blocks * block_size and a block-table entry indexes n_blocks, and
 * drawing either from the wrong range tests out-of-range handling and calls the result
 * correctness. rad_abi.h and tools/kfixture.cpp both record having been bitten by each of these.
 */
static inline uint64_t hsplitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void fill_normal(float* v, int64_t n, uint64_t seed, float sigma) {
    uint64_t s = seed ? seed : 1;
    for (int64_t i = 0; i < n; i += 2) {
        /* Box-Muller off two splitmix draws. u1 is nudged off zero so the log is finite. */
        const double u1 = ((hsplitmix(s) >> 40) + 1) * (1.0 / 16777216.0);
        const double u2 = ((hsplitmix(s) >> 40)) * (1.0 / 16777216.0);
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 6.283185307179586 * u2;
        v[i] = (float)(r * std::cos(th)) * sigma;
        if (i + 1 < n) v[i + 1] = (float)(r * std::sin(th)) * sigma;
    }
}

struct Opd {
    RadOpdDesc d{};
    bool absent = false;
    int  role = RAD_OPD_IN;
    int64_t n = 0;
    std::vector<uint8_t> master, a, b;   /* the drawn bytes, then one copy per side */
};

static inline int64_t desc_numel(const RadOpdDesc& d) {
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    return n;
}

static inline size_t dt_bytes(uint32_t dt, int64_t n) {
    return (size_t)((n * rad_dtype_bits(dt) + 7) / 8);
}

/* W4 / W2 / MXFP4 / MXFP6 have no store side in the ABI's accessor, and the packed grids are WEIGHT
 * operands a fixture would lay out rather than draw. Writing the CODE directly is the honest
 * alternative: both plugins decode the same symmetric two's-complement nibble
 * (libref/ref_common.h), so the bytes mean the same thing to both sides, which is the only
 * property the comparison needs. */
static inline void h_narrow(const float* src, uint32_t dt, int64_t n, void* dst) {
    if (dt == RAD_FP6E2M3 || dt == RAD_FP6E3M2) {
        /* A drawn index IS the code: 0..63 covers all 64 e2m3 patterns, and every pattern is a
         * value -- sign, the subnormal M/8 arm, and the 7.5 top all occur. */
        uint8_t* b = (uint8_t*)dst;
        std::memset(b, 0, dt_bytes(dt, n));
        const int maxc = 63;
        for (int64_t i = 0; i < n; ++i) {
            int c = (int)std::lrintf(src[i]);
            c = c < 0 ? 0 : (c > maxc ? maxc : c);
            rad_store_bits(b, 6, i, (uint32_t)c);
        }
        return;
    }
    if (dt == RAD_I4 || dt == RAD_FP4E2M1) {
        uint8_t* b = (uint8_t*)dst;
        std::memset(b, 0, dt_bytes(dt, n));
        for (int64_t i = 0; i < n; ++i) {
            int c = (int)std::lrintf(src[i] * 3.0f);
            c = c < -8 ? -8 : (c > 7 ? 7 : c);
            b[i >> 1] |= (uint8_t)(((uint8_t)((c + 8) & 0xf)) << (4 * (i & 1)));
        }
        return;
    }
    if (dt == RAD_I2) {
        uint8_t* b = (uint8_t*)dst;
        std::memset(b, 0, dt_bytes(dt, n));
        for (int64_t i = 0; i < n; ++i) {
            int c = (int)std::lrintf(src[i]);
            c = c < -2 ? -2 : (c > 1 ? 1 : c);
            b[i >> 2] |= (uint8_t)(((uint8_t)((c + 2) & 0x3)) << (2 * (i & 3)));
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) rad_store_f32(dst, dt, i, src[i]);
}

static inline void h_widen(const void* src, uint32_t dt, int64_t n, float* dst) {
    if (dt == RAD_I4) {
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t by = ((const uint8_t*)src)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0xf);
            dst[i] = (float)((int)(nb ^ 0x8u) - 8);
        }
        return;
    }
    if (dt == RAD_I2) {
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t by = ((const uint8_t*)src)[i >> 2];
            const uint8_t c = (uint8_t)((by >> (2 * (i & 3))) & 0x3u);
            dst[i] = (float)((int)(c ^ 0x2u) - 2);
        }
        return;
    }
    if (dt == RAD_FP4E2M1) {
        static const float mag[8] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6 };
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t by = ((const uint8_t*)src)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0xf);
            dst[i] = (nb & 8u) ? -mag[nb & 7u] : mag[nb & 7u];
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) dst[i] = rad_load_f32(src, dt, i);
}

static inline void h_draw(std::vector<Opd>& o, size_t k, int64_t M, uint64_t seed) {
    Opd& d = o[k];
    const int64_t n = d.n;
    d.master.assign(dt_bytes(d.d.dtype, n > 0 ? n : 0), 0);
    if (n <= 0 || d.absent) return;
    if (d.role != RAD_OPD_IN && d.role != RAD_OPD_INOUT && d.role != RAD_OPD_WEIGHT) return;

    std::vector<float> tmp((size_t)n, 0.0f);
    const bool is_index = d.d.fill == RAD_FILL_INDEX;
    const bool idx_cu   = (d.d.flags & RAD_OPD_F_IDX_CU) != 0;
    const bool idx_uniq = (d.d.flags & RAD_OPD_F_IDX_UNIQUE) != 0;

    if (d.role == RAD_OPD_WEIGHT && !is_index) {
        /* 0.02 sigma, and it is not cosmetic: a checkpoint's weights are that size, and drawing
         * them at unit variance puts an fp8 quantiser's amax three orders of magnitude off where a
         * real one sits -- so the case would measure the quantiser on data no model produces. */
        fill_normal(tmp.data(), n, seed, 0.02f);
    } else if (is_index && idx_cu) {
        /* Cumulative sequence lengths, [0 .. T]: the batch's shape, not a draw. */
        const int64_t S = n > 1 ? n - 1 : 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)((i * M) / S);
    } else if (is_index && d.d.idx_const >= 0) {
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)d.d.idx_const;
    } else if (is_index && idx_uniq) {
        /* A scatter's destinations, distinct: two tokens on one slot is a race the two
         * implementations resolve differently, so the comparison would measure the scheduler. */
        uint64_t s2 = seed;
        const int64_t range = d.d.idx_max > 0 ? d.d.idx_max : n;
        std::vector<int32_t> pool((size_t)range);
        for (int64_t i = 0; i < range; ++i) pool[(size_t)i] = (int32_t)i;
        for (int64_t i = 0; i < n && i < range; ++i) {
            const int64_t j = i + (int64_t)(hsplitmix(s2) % (uint64_t)(range - i));
            std::swap(pool[(size_t)i], pool[(size_t)j]);
            tmp[(size_t)i] = (float)pool[(size_t)i];
        }
        for (int64_t i = range; i < n; ++i) tmp[(size_t)i] = -1.0f;
    } else if (is_index) {
        uint64_t s2 = seed;
        int64_t rows = 32000;
        if (k > 0 && o[k - 1].d.rank > 0) rows = o[k - 1].d.shape[0];
        if (d.d.idx_max > 0) rows = d.d.idx_max;
        if (rows <= 0) rows = 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)(hsplitmix(s2) % (uint64_t)rows);
    } else {
        fill_normal(tmp.data(), n, seed, 1.0f);
        if (d.d.fill == RAD_FILL_SIGMOID) {
            for (int64_t i = 0; i < n; ++i)
                tmp[(size_t)i] = 1.0f / (1.0f + std::exp(-tmp[(size_t)i]));
        } else if (d.d.fill == RAD_FILL_TRI_LOWER) {
            /* UNIT DIAGONAL, SMALL OFF-DIAGONAL. `A` is the chunked delta rule's intra-chunk
             * matrix I + tril(diag(beta) K K^T, -1), unit lower triangular by construction, and
             * its inverse is bounded only because of that. A normal on the diagonal makes the
             * inverse enormous, and what the comparison then reports is the condition number of
             * the TEST. */
            const int64_t ch = d.d.fill_chunk > 0 ? d.d.fill_chunk : 1;
            const int64_t heads = d.d.rank >= 3 ? d.d.shape[1] : 1;
            const int64_t rows = (heads * ch) != 0 ? n / (heads * ch) : n;
            for (int64_t t = 0; t < rows; ++t)
                for (int64_t h = 0; h < heads; ++h)
                    for (int64_t j = 0; j < ch; ++j) {
                        const size_t i = (size_t)((t * heads + h) * ch + j);
                        const int64_t diag = t % ch;
                        if (j > diag)       tmp[i] = 0.0f;
                        else if (j == diag) tmp[i] = 1.0f;
                        else                tmp[i] *= 0.25f;
                    }
        } else if (d.d.fill == RAD_FILL_GATE_CUMSUM) {
            const int64_t heads = d.d.rank >= 2 ? d.d.shape[d.d.rank - 1] : 1;
            const int64_t rows = heads > 0 ? n / heads : n;
            const int64_t ch = d.d.fill_chunk > 0 ? d.d.fill_chunk : rows;
            for (int64_t h = 0; h < heads; ++h) {
                float acc = 0.0f;
                for (int64_t t = 0; t < rows; ++t) {
                    if (t % ch == 0) acc = 0.0f;
                    acc -= std::log1p(std::exp(tmp[(size_t)(t * heads + h)]));
                    tmp[(size_t)(t * heads + h)] = acc;
                }
            }
        } else if (d.d.fill == RAD_FILL_UNIT_ROW) {
            const int64_t w = d.d.rank ? d.d.shape[d.d.rank - 1] : n;
            if (w > 0)
                for (int64_t r = 0; r + w <= n; r += w) {
                    double ss = 0;
                    for (int64_t i = 0; i < w; ++i)
                        ss += (double)tmp[(size_t)(r + i)] * (double)tmp[(size_t)(r + i)];
                    const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
                    for (int64_t i = 0; i < w; ++i) tmp[(size_t)(r + i)] *= inv;
                }
        }
    }
    h_narrow(tmp.data(), d.d.dtype, n, d.master.data());
}

/* ================================================================== the geometry table
 *
 * THE ONE PLACE THESE TOOLS KNOW ANYTHING ABOUT AN OP, and it is a PARAMETER list rather than a
 * shape: the shapes come from libref's own `opd_shape`. The values are chosen to be small but not
 * degenerate -- a row wider than any vector, a row count above one, a group that divides, and
 * where the op allows it a row length that does NOT divide the vector width, because the tail is
 * where a hand-vectorised kernel is wrong.
 */
struct P { const char* k; int kind; long long i; const char* s; double d; };

static inline RadParam h_mk(const P& p) {
    RadParam r{};
    r.key = p.k; r.kind = p.kind; r.ival = p.i; r.sval = p.s; r.dval = p.d;
    return r;
}
#define HI(k, v) P{ k, RAD_P_INT, (long long)(v), nullptr, 0.0 }
#define HS(k, v) P{ k, RAD_P_STR, 0, v, 0.0 }
#define HF(k, v) P{ k, RAD_P_F64, 0, nullptr, (double)(v) }

struct Case {
    const char* op;
    std::vector<P> p;
    /* Bytes moved and flops done, for the bench's rate columns. 0 means "do not report a rate":
     * an op whose cost is not a simple function of its geometry gets a time and no invented
     * denominator. */
    double bytes = 0, flops = 0;
    /* A case worth TIMING at a real model shape and not worth checking there. Correctness is a
     * property of the arithmetic and a 640-wide row exercises every path a 2048-wide one does, so
     * the checker would spend minutes inside libref to learn nothing; speed is not, because it is
     * the cache hierarchy that decides it and a shape that fits L1 answers a different question.
     * The same split tools/kfixture.h draws, and for the same reason. rad-avx-check skips these. */
    bool speed_only = false;
};

/* THE TOTAL TOKEN COUNT OF THE BATCH, which is what the cumulative-length draw divides across the
 * sequences and what no operand carries.
 *
 * IT IS NOT ALWAYS THE RANGED PARAMETER, and that distinction cost four silently skipped ops. An
 * op banded on `M` is banded on the TOTAL -- `gdn_conv_prep` sizes its activation [M, conv_dim] and
 * its cu spans M. An op banded on `q_len` is banded on the PER-SEQUENCE length: `gdn_conv_update`
 * sizes its activation [n_seq * q_len, conv_dim], because a decode step presents q_len candidate
 * rows for each of n_seq sequences. Drawing cu to sum to q_len there gives a batch whose cu says
 * two tokens over operands sized for four, and libref recomputes its head count from
 * numel(q) / (T * head_k) -- which comes out double, makes conv_dim double, and refuses the launch
 * with RAD_E_SHAPE. The case was then SKIPPED, and a skip with no reason printed is indistinguishable
 * from an op that has no case at all.
 *
 * `attn_paged` is also banded on `q_len` and is unaffected only because libref describes it at one
 * sequence; the rule below is right for both. */
static inline int64_t case_M(const Case& c) {
    int64_t n_seq = 1;
    for (const P& p : c.p) if (std::strcmp(p.k, "n_seq") == 0 && p.i > 0) n_seq = p.i;
    for (const P& p : c.p) if (std::strcmp(p.k, "M") == 0) return p.i;
    for (const P& p : c.p) if (std::strcmp(p.k, "q_len") == 0) return p.i * n_seq;
    for (const P& p : c.p) if (std::strcmp(p.k, "numel") == 0) return p.i;
    return 1;
}

static inline const char* case_dtype(const Case& c) {
    static const char* keys[] = { "dtype", "q_dtype", "kv_dtype", "to" };
    for (const char* k : keys)
        for (const P& p : c.p)
            if (std::strcmp(p.k, k) == 0 && p.kind == RAD_P_STR) return p.s;
    return "*";
}

static inline long long case_geti(const Case& c, const char* k, long long dflt) {
    for (const P& p : c.p) if (std::strcmp(p.k, k) == 0) return p.i;
    return dflt;
}

/* A one-line rendering of a case, so a failure names a geometry a reader can reproduce. */
static inline std::string case_str(const Case& c) {
    std::string s;
    for (const P& p : c.p) {
        char buf[96];
        if (p.kind == RAD_P_STR)      std::snprintf(buf, sizeof buf, "%s=%s ", p.k, p.s ? p.s : "");
        else if (p.kind == RAD_P_F64) std::snprintf(buf, sizeof buf, "%s=%g ", p.k, p.d);
        else                          std::snprintf(buf, sizeof buf, "%s=%lld ", p.k, p.i);
        s += buf;
    }
    if (!s.empty()) s.pop_back();
    return s;
}

/* ================================================================== a prepared case
 * Operands sized by the ORACLE's description, drawn once, and copied into one buffer per side.
 * The two plugins must see byte-identical inputs or the comparison is noise; drawing twice from
 * one seed would work too and would be one more place for the two to diverge. */
struct Prepared {
    bool ok = false;
    bool skipped = false;              /* the oracle declined this geometry -- a real answer */
    std::vector<RadParam> params;
    std::vector<Opd> opd;
    const RadOpSchema* schema = nullptr;
    std::vector<RadTensor> ta, tb;
    RadArgs aa{}, ab{};
    std::vector<uint8_t> sa, sb;
};

static inline void h_build_tensors(std::vector<Opd>& opd, bool side_a,
                                   std::vector<RadTensor>* out) {
    out->assign(opd.size(), RadTensor{});
    for (size_t i = 0; i < opd.size(); ++i) {
        RadTensor& x = (*out)[i];
        if (opd[i].absent) { x.data = nullptr; continue; }
        x.data = (side_a ? opd[i].a : opd[i].b).data();
        x.dtype = opd[i].d.dtype;
        x.rank = opd[i].d.rank;
        int64_t acc = 1;
        for (int k = (int)x.rank - 1; k >= 0; --k) {
            x.shape[k] = opd[i].d.shape[k];
            x.stride[k] = acc;
            acc *= x.shape[k];
        }
    }
}

/* `describe` is the kernel whose opd_shape is asked. A correctness check passes the ORACLE's (the
 * two sides must be handed the same bytes, and where they could disagree about an extent the
 * reference semantics define the op); a benchmark of a kernel with no oracle passes its own.
 * rad-kbench makes the same distinction for the same reason, in tools/opshapes.h. */
static inline Prepared h_prepare(const Lib& oracle, const RadKernelInfo* describe, const Case& c,
                                 const RadKernelInfo* ka, const RadKernelInfo* kb) {
    Prepared pr;
    pr.schema = oracle.schema(c.op);
    if (!pr.schema || !describe || !describe->opd_shape) { pr.skipped = true; return pr; }

    pr.params.reserve(c.p.size());
    for (const P& p : c.p) pr.params.push_back(h_mk(p));

    pr.opd.assign((size_t)pr.schema->n_operands, Opd{});
    for (int i = 0; i < pr.schema->n_operands; ++i) {
        RadOpdDesc d{};
        d.idx_const = -1;      /* the ABI's "draw", so a hook that leaves it alone means default */
        if (describe->opd_shape(pr.params.data(), (int)pr.params.size(), i, &d) < 0) {
            pr.skipped = true;
            return pr;
        }
        Opd& o = pr.opd[(size_t)i];
        o.d = d;
        o.absent = (d.flags & RAD_OPD_F_ABSENT) != 0;
        o.role = pr.schema->operands[i].role;
        o.n = o.absent ? 0 : desc_numel(d);
        /* A rank the ABI cannot express is a hook bug, not a declined geometry, and it must not
         * become a zero-extent buffer that every comparison then passes vacuously. */
        if (!o.absent && (d.rank == 0 || d.rank > RAD_MAX_RANK)) { pr.skipped = true; return pr; }
    }

    uint64_t seed = 0x9E3779B9ull;
    for (const P& p : c.p) {
        seed = seed * 1000003ull + (uint64_t)p.i;
        if (p.s) for (const char* s = p.s; *s; ++s) seed = seed * 131ull + (unsigned char)*s;
    }
    for (size_t k = 0; k < pr.opd.size(); ++k) {
        uint64_t s = seed ^ (0x100000001b3ull * (k + 1));
        h_draw(pr.opd, k, case_M(c), s ? s : 1);
        pr.opd[k].a = pr.opd[k].master;
        pr.opd[k].b = pr.opd[k].master;
    }

    h_build_tensors(pr.opd, true, &pr.ta);
    h_build_tensors(pr.opd, false, &pr.tb);

    pr.aa.t = pr.ta.data(); pr.aa.n_t = (int)pr.ta.size();
    pr.aa.p = pr.params.data(); pr.aa.n_p = (int)pr.params.size();
    pr.aa.rank = 0; pr.aa.world_size = 1;
    pr.ab = pr.aa;
    pr.ab.t = pr.tb.data();

    /* The scratch hook is honoured rather than assumed away: a kernel that grew one and was handed
     * null would fault, and the fault would read as a kernel bug. */
    if (ka && ka->scratch) {
        const int64_t n = ka->scratch(&pr.aa);
        if (n > 0) { pr.sa.assign((size_t)n, 0); pr.aa.scratch = pr.sa.data(); pr.aa.scratch_bytes = n; }
    }
    if (kb && kb->scratch) {
        const int64_t n = kb->scratch(&pr.ab);
        if (n > 0) { pr.sb.assign((size_t)n, 0); pr.ab.scratch = pr.sb.data(); pr.ab.scratch_bytes = n; }
    }
    pr.ok = true;
    return pr;
}

/* Restore both sides to the drawn bytes. A benchmark that loops a launch must do this between
 * iterations for any op with an INOUT operand, or it is timing a different computation each time
 * -- a twenty-iteration loop over `rope` leaves the tensor rotated twenty-one times, which a
 * checker then reports as a large error against a kernel that is correct (tools/opshapes.h and
 * rad-kbench make the same point). */
static inline void h_reset(Prepared& pr) {
    for (Opd& o : pr.opd) {
        if (o.absent || o.n <= 0) continue;
        if (o.role == RAD_OPD_INOUT || o.role == RAD_OPD_OUT) {
            std::memcpy(o.a.data(), o.master.data(), o.master.size());
            std::memcpy(o.b.data(), o.master.data(), o.master.size());
        }
    }
}

std::vector<Case> build_cases();

#endif /* RAD_AVX_HARNESS_H */
