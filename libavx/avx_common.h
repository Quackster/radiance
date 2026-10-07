/* avx_common.h -- what every kernel in this plugin needs and the ABI does not give it.
 *
 * Deliberately thin. rad_plugin.h already carries the parameter readers, the absent-operand rule,
 * the row-pitch helper and every narrowing conversion, and its header explains at length why those
 * belong in the ABI rather than once per plugin. This plugin calls them. What is here is the three
 * things that are genuinely this plugin's:
 *
 *   THE SCRATCH ARENA. Every kernel converts a row to f32 before it computes (avx_vec.h), so every
 *   kernel needs a row of scratch. RadArgs::scratch would be the ABI's answer and it is the wrong
 *   one here: it is sized once at declare from the band's upper bound, it is one buffer, and this
 *   plugin runs its outer loop under OpenMP -- so every thread needs its own, and the number of
 *   threads is not known at declare. A thread_local arena that grows to a high-water
 *   mark and is then reused is allocation-free on the hot path, which is what the ABI's "must not
 *   allocate" is protecting, and it is what libref does too (it holds std::vector in its kernels).
 *   The growth is logged under AVX_TRACE so a kernel that reallocates per call is visible.
 *
 *   THE OPERAND PREAMBLE. Sixty-eight launches open by fetching n operands, refusing if a required
 *   one is absent, and reading the extents off the tensors rather than off the parameters --
 *   libref/ref_ops.h explains why that rule exists and it is the same rule here: a ranged parameter
 *   is collapsed to a BAND, so a loop bounded by `M` runs to the band's upper bound and walks off
 *   the arena. Every extent in this plugin comes from a RadTensor.
 *
 *   THE THREADING DECISION, in one place. OpenMP over the OUTER dimension only, one thread owning
 *   whole output rows, no reduction() clause anywhere -- the same rule libref states, for a
 *   different reason. libref wants bit-identical answers across thread counts because it is the
 *   oracle. This plugin wants them because a kernel whose answer moves with the machine's load is
 *   a kernel no measurement can rank: two runs of the same binary would differ, and the difference
 *   would be read as whatever change was under test.
 */
#ifndef RAD_AVX_COMMON_H
#define RAD_AVX_COMMON_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
/* THE TEAM IS ONE THREAD A PHYSICAL CORE, not one a logical CPU. These kernels are FMA- or
 * bandwidth-bound, so an SMT sibling adds next to nothing to them -- and the engine runs threads
 * of its own beside the team (a worker per stream, the scheduler, the HTTP server). A team as
 * wide as the logical CPU count leaves those nowhere to run but on a core a team thread is
 * spinning on, and a team barrier then waits for a descheduled thread: measured on a 16-core,
 * 32-thread part, an encoder step of 576 tokens took 0.75 s at 32 threads and 0.09 s at 16.
 * Counted over the process's affinity mask, so a pinned or containerised engine sizes its team to
 * what it was given; OMP_NUM_THREADS, when set, is the operator's and wins, and a thread that
 * asked OpenMP for fewer (omp_set_num_threads) gets fewer. */
static inline int avx_team_size() {
    /* The physical core count is a property of the process and counted once; the OpenMP request
     * is the calling thread's and read every time, so a caller that asks for fewer threads
     * (omp_set_num_threads, as the benchmark does for its single-thread figures) gets them. */
    static const int cores = [] {
        if (const char* e = std::getenv("OMP_NUM_THREADS"); e && *e) return 0;
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
        int n = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            /* A CPU counts when it is the first of its core's siblings that is in the mask. */
            char path[96];
            std::snprintf(path, sizeof path,
                          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            int first = c;
            if (FILE* f = std::fopen(path, "r")) {
                int v = -1;
                if (std::fscanf(f, "%d", &v) == 1 && v >= 0 && v < CPU_SETSIZE && CPU_ISSET(v, &set))
                    first = v;
                std::fclose(f);
            }
            n += first == c;
        }
        return n;
    }();
    const int asked = omp_get_max_threads();
    return cores > 0 && cores < asked ? cores : asked;
}
#define AVX_PRAGMA(x) _Pragma(#x)
#define AVX_PARALLEL_FOR AVX_PRAGMA(omp parallel for schedule(static) num_threads(avx_team_size()))
/* THE OUTER LOOP IS WORTH A TEAM ONLY WHEN THERE IS ENOUGH WORK TO PAY FOR STARTING ONE, and the
 * `if` clause is how that is said without writing the serial loop twice. A fork/join costs a few
 * microseconds; an op below the threshold spends more time in the barrier than in the arithmetic,
 * which is the same shape of problem a small device dispatch has, one level down. The crossover is
 * measured by `rad-avx-bench --threads`.
 *
 * `cond` is pasted into the pragma text, so it must contain no comma. */
#define AVX_PARALLEL_FOR_IF(cond) \
    AVX_PRAGMA(omp parallel for schedule(static) if(cond) num_threads(avx_team_size()))
#else
#define AVX_PARALLEL_FOR
#define AVX_PARALLEL_FOR_IF(cond)
#endif

/* Elements of output a launch must touch before a parallel region pays for itself. Used as
 * AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN). */
#define AVX_PAR_MIN 16384

namespace avx {

/* ================================================================== scratch
 * Aligned to 64 so a 512-bit aligned load is legal off it at every level. Returns a buffer of at
 * least `n` floats, valid until the next call on this thread that asks for more. */
float*   scratch_f32(size_t n);
/* A second and third independent buffer, for kernels that need two rows live at once (every gated
 * op: gate and up, a and b, x and residual). Separate functions rather than an index so the call
 * site says which buffer it means. */
float*   scratch_f32_b(size_t n);
float*   scratch_f32_c(size_t n);
void*    scratch_raw(size_t bytes);

/* ================================================================== operand preamble */
/* Fetch the first `n` operands into `out`, returning 0 if any is absent. The array is indexed
 * positionally and an absent OPTIONAL operand past `n` is fetched with rad_arg_in directly. */
static inline int take(const RadArgs* a, int n, const RadTensor** out) {
    for (int i = 0; i < n; ++i) {
        out[i] = rad_arg_in(a, i);
        if (!out[i]) return 0;
    }
    return 1;
}

/* The (rows, lastdim) reading of a tensor, and the pitch between rows. Three lines every kernel
 * would otherwise write. */
struct Plane {
    void*   p;
    uint32_t dt;
    int64_t rows, n, pitch;
};

static inline Plane plane(const RadTensor* t) {
    Plane s;
    s.p     = t->data;
    s.dt    = t->dtype;
    s.rows  = rad_tensor_rows(t);
    s.n     = rad_tensor_lastdim(t);
    s.pitch = rad_tensor_row_pitch(t);
    return s;
}

/* Element offset of row r. Strides are in ELEMENTS (rad_types.h), so this is what a caller adds to
 * a typed pointer -- never a byte count, which is what makes it safe for W4/W2 where a byte count
 * would be fractional. */
static inline int64_t row_off(const Plane& s, int64_t r) { return r * s.pitch; }

/* ================================================================== libref's addressing, exactly
 *
 * THESE ARE COPIED FROM libref/ref_common.h AND THEY HAVE TO BE. They are not conveniences, they
 * are the ADDRESSING CONTRACT: which bytes an operand's element (r, i) lives at, how a rank-3
 * tensor is read as [rows, cols], and what a strided view means. A plugin that computed the right
 * arithmetic over the wrong bytes would be checked against the oracle and would fail in a way that
 * looks like a numerical bug, which is the most expensive kind to chase.
 *
 * `numel` returning 0 for a null-data tensor is part of it: that is how libref's shape checks
 * refuse an absent required operand, and a copy that returned the product of the shape instead
 * would accept the call and fault. */
static inline int64_t numel(const RadTensor* t) {
    if (!t || !t->data || t->rank == 0) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

static inline int64_t laststride(const RadTensor* t) { return t->rank ? t->stride[t->rank - 1] : 1; }

/* Element offset of the `lin`-th element in row-major logical order. */
static inline int64_t offlin(const RadTensor* t, int64_t lin) {
    if (t->rank == 0) return lin;
    int64_t acc = 1;
    bool c = true;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) { c = false; break; }
        acc *= t->shape[i];
    }
    if (c) return lin;
    int64_t off = 0;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        int64_t s = t->shape[i];
        if (s <= 0) return 0;
        off += (lin % s) * t->stride[i];
        lin /= s;
    }
    return off;
}

/* The element offset of logical row `r` read as [rows, cols]. The leading axes are walked with
 * their OWN strides, so a [T, H, V] tensor read as [T*H, V] lands correctly even when H's stride
 * is not V -- which is exactly the fused-projection column slice docs/OPS.md warns about. */
static inline int64_t rowoff(const RadTensor* t, int64_t r, int64_t cols) {
    if (t->rank >= 2 && t->shape[t->rank - 1] == cols) {
        int64_t off = 0;
        for (int i = (int)t->rank - 2; i >= 0; --i) {
            int64_t s = t->shape[i];
            if (s <= 0) return 0;
            off += (r % s) * t->stride[i];
            r /= s;
        }
        return off;
    }
    return r * cols;
}

static inline int64_t rows_of(const RadTensor* t, int64_t n) { return n > 0 ? numel(t) / n : 0; }

/* THE OPERAND'S EXTENT, WITH THE RANGED PARAMETER AS A CEILING. libref/ref_common.h's band_rows,
 * copied for the reason the header gives: `M`, `numel` and `q_len` arrive as the band's upper
 * bound, so the call's row count is the operand's and a count above the band is a caller bug. */
static inline int64_t band_rows(int64_t from_operand, int64_t band) {
    if (from_operand <= 0) return 0;
    return (band > 0 && from_operand > band) ? 0 : from_operand;
}

/* THE BLOCK A SCALE GRID OF `blocks` PIECES IMPLIES OVER `ext` ELEMENTS -- libref/ref_gemm.h's
 * grid_block, which says why it is the power of two whose ceiling division gives `blocks` and not
 * the quotient ext / blocks: a grid's last block may be partial. */
static inline int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* `row_stride`: which axis carries a row of `cols` elements, at whatever rank the operand arrived
 * in -- libref/ref_gemm.h's rule, copied for the header's reason. A rank-2 operand is not dense
 * because it is rank 2: the engine hands attention a COLUMN SLICE of the fused QKV plane,
 * [tokens, kv_heads * head_dim] with the whole projection's width as its row stride, and deriving
 * the stride from the shape reads every token after the first out of the previous token's row. */
static inline int64_t row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

/* ONE ELEMENT, AS f32 AND BACK, with the two things the ABI's accessors do not do.
 *
 * THE PACKED CODES. rad_load_f32 returns 0 for W4, W2 and MXFP4 -- its contract says a dtype it
 * cannot read is a refusal and not a value -- so every strided or odd-offset read of a packed
 * weight that falls off the row converter would decode as zeros. The codes are libref's
 * (ref_common.h: w4_code, w2_code, mxfp4_code), the same ones row_to_f32 unpacks in bulk.
 *
 * THE INTEGER STORE ROUNDS TO NEAREST-EVEN AND SATURATES, as libref's st_dt does, where
 * rad_store_f32 truncates and converts out-of-range values with a plain cast. A NaN stores 0. */
static inline float ld_elem(const void* p, uint32_t dt, int64_t i) {
    switch (dt) {
        case RAD_I4: {
            const uint8_t b = ((const uint8_t*)p)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
            return (float)((int)(nb ^ 0x8u) - 8);
        }
        case RAD_I2: {
            const uint8_t b = ((const uint8_t*)p)[i >> 2];
            const uint8_t c = (uint8_t)((b >> (2 * (i & 3))) & 0x3u);
            return (float)((int)(c ^ 0x2u) - 2);
        }
        case RAD_FP4E2M1: {
            static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const uint8_t b = ((const uint8_t*)p)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
            return (nb & 8u) ? -kMag[nb & 7u] : kMag[nb & 7u];
        }
        default: return rad_load_f32(p, dt, i);
    }
}

/* Round to nearest-even and clamp to [lo, hi], in double so that every bound of every integer
 * width up to 32 bits is exact. NaN is 0. */
static inline double round_sat(float v, double lo, double hi) {
    if (std::isnan(v)) return 0.0;
    const double r = (double)std::nearbyintf(v);
    return r < lo ? lo : (r > hi ? hi : r);
}

static inline void st_elem(void* p, uint32_t dt, int64_t i, float v) {
    switch (dt) {
        case RAD_I8:  ((int8_t*)p)[i]   = (int8_t)round_sat(v, -128.0, 127.0); break;
        case RAD_U8:  ((uint8_t*)p)[i]  = (uint8_t)round_sat(v, 0.0, 255.0); break;
        case RAD_I16: ((int16_t*)p)[i]  = (int16_t)round_sat(v, -32768.0, 32767.0); break;
        case RAD_I32: ((int32_t*)p)[i]  = (int32_t)round_sat(v, -2147483648.0, 2147483647.0); break;
        case RAD_U32: ((uint32_t*)p)[i] = (uint32_t)round_sat(v, 0.0, 4294967295.0); break;
        case RAD_I64: {
            /* 2^63 is the first float past the range, so the bound is a comparison, not a clamp
             * value a double could hold. */
            const double r = round_sat(v, -9223372036854775808.0, 9223372036854775808.0);
            ((int64_t*)p)[i] = r >= 9223372036854775808.0 ? INT64_MAX : (int64_t)r;
            break;
        }
        default: rad_store_f32(p, dt, i, v); break;
    }
}

static inline float ldt(const RadTensor* t, int64_t off) { return ld_elem(t->data, t->dtype, off); }
static inline void  stt(const RadTensor* t, int64_t off, float v) {
    st_elem(t->data, t->dtype, off, v);
}

/* IS THIS OPERAND'S ROW DENSE? The whole vectorised path is predicated on it: a row whose elements
 * are one apart can be converted with one call and then read as a flat f32 array, and a row whose
 * elements are not has to go through the strided gather. Almost every operand in a real step is
 * dense; the exceptions are named in docs/OPS.md and are a handful of sites. */
static inline bool dense_row(const RadTensor* t) { return laststride(t) == 1; }

/* ================================================================== parameters
 * Deliberately LAXER than rad_param_geti, and for libref's reason rather than a different one: a
 * tool or a test may call a launch directly with no declare in front of it to enforce the type, so
 * a range and a string spelling of an integer are both read rather than refused. The ABI's own
 * reader is right for a kernel reached through declare and wrong for one reached through a
 * checker, and this plugin is reached through both. */
static inline const RadParam* p_find(const RadArgs* a, const char* k) {
    return a ? rad_param_find(a->p, a->n_p, k) : nullptr;
}

static inline long long p_int(const RadArgs* a, const char* k, long long dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_INT)   return p->ival;
    if (p->kind == RAD_P_RANGE) return p->ihi;
    if (p->kind == RAD_P_STR && p->sval) return std::strtoll(p->sval, nullptr, 10);
    return dflt;
}

static inline const char* p_str(const RadArgs* a, const char* k, const char* dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) return p->sval;
    return dflt;
}

static inline float p_f32(const RadArgs* a, const char* k, float dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_F64)   return (float)p->dval;
    if (p->kind == RAD_P_STR && p->sval) return (float)std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT)   return (float)p->ival;
    if (p->kind == RAD_P_RANGE) return (float)p->ihi;
    return dflt;
}

static inline int p_enum(const RadArgs* a, const char* k, const char* const* names, int n,
                         int dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) {
        for (int i = 0; i < n; ++i)
            if (names[i] && std::strcmp(names[i], p->sval) == 0) return i;
        return dflt;
    }
    if (p->kind == RAD_P_INT) return (int)p->ival;
    return dflt;
}

static inline int64_t ld_int_dt(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32:  return (int64_t)((const int32_t*)p)[i];
        case RAD_U32:  return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64:  return ((const int64_t*)p)[i];
        case RAD_I16:  return (int64_t)((const int16_t*)p)[i];
        case RAD_I8:   return (int64_t)((const int8_t*)p)[i];
        case RAD_U8:   return (int64_t)((const uint8_t*)p)[i];
        case RAD_BOOL: return ((const uint8_t*)p)[i] ? 1 : 0;
        case RAD_F32:  return (int64_t)((const float*)p)[i];
        default:       return 0;
    }
}

static inline void st_int_dt(uint32_t dt, void* p, int64_t i, int64_t v) {
    switch (dt) {
        case RAD_I32: ((int32_t*)p)[i] = (int32_t)v; break;
        case RAD_U32: ((uint32_t*)p)[i] = (uint32_t)v; break;
        case RAD_I64: ((int64_t*)p)[i] = v; break;
        case RAD_I16: ((int16_t*)p)[i] = (int16_t)v; break;
        case RAD_I8:  ((int8_t*)p)[i]  = (int8_t)v; break;
        case RAD_U8:  ((uint8_t*)p)[i] = (uint8_t)v; break;
        default:      rad_store_f32(p, dt, i, (float)v); break;
    }
}

static inline int64_t ldi(const RadTensor* t, int64_t off) { return ld_int_dt(t->dtype, t->data, off); }

/* Bytes of element i for a dtype whose element is not a whole byte. Only the packed weight codes
 * need it and they are always read through row_to_f32, but `dequant` addresses a group start. */
static inline const void* byte_at(const void* base, uint32_t dt, int64_t i) {
    const uint8_t* b = (const uint8_t*)base;
    switch (dt) {
        case RAD_I4:    case RAD_FP4E2M1: return b + (i >> 1);
        case RAD_I2:                    return b + (i >> 2);
        default:                        return b + i * (size_t)(rad_dtype_bits(dt) / 8);
    }
}

}  /* namespace avx */
#endif /* RAD_AVX_COMMON_H */
