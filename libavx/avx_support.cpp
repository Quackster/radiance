/* avx_support.cpp -- the dtype-generic row and the row reductions, compiled once per ISA level.
 *
 * WHY THESE ARE OUT OF LINE AND NOT IN THE HEADER. They are called from every kernel with a
 * runtime dtype, so inlining them would paste a fifteen-way switch into sixty-eight call sites and
 * the switch is the thing being hoisted out of the loop in the first place. Out of line, the
 * branch is taken ONCE PER ROW and the body it lands in is a straight vector loop the hardware
 * prefetcher can see through.
 *
 * The conversion rates per level are measured by `rad-avx-bench --cvt` and recorded in
 * libavx/README.md rather than here, because they are a property of the machine the bench last ran
 * on and a number typed into a comment outlives the machine that produced it.
 *
 * The two fp8 decodes take different paths and avx_vec.h says why: e4m3 is a branchless bit
 * reconstruction, e5m2 a table gather, and the scalar level and every tail element read the table
 * in both cases.
 */
#include "avx_vec.h"
#include "avx_common.h"

namespace avx {
inline namespace AVX_SUFFIX {

/* ================================================================== packed weight codes
 * The conventions are libr4d's, because a .rad is specific to a kernel set (spec §4.2) and this
 * plugin has to read the same bytes: W4 is two nibbles a byte with element 2i LOW and a symmetric
 * two's-complement code ((n ^ 8) - 8), W2 four codes a byte with the same construction one radix
 * down, MXFP4 two e2m1 nibbles with the magnitude table below. libref/ref_common.h is the written
 * definition and this is the vectorised reading of it.
 *
 * THE W4 UNPACK IS A SHUFFLE, NOT A SHIFT PER ELEMENT. Sixteen nibble pairs are one 128-bit load;
 * the low nibbles are an AND and the high ones a shift, and the two interleave back with a
 * punpcklbw -- four instructions for 32 elements against 32 shifts and 32 masks scalar. W2 and
 * MXFP4 are left scalar: W2 appears only in the draft head's lm_head and MXFP4 only in an expert
 * grid this plugin does not serve at speed, so neither is on a path worth the code. */
static const float kMxfp4Mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

static void w4_to_f32(const uint8_t* src, float* dst, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL >= 2
    const __m128i lomask = _mm_set1_epi8(0x0f);
    for (; i + 32 <= n; i += 32) {
        __m128i b  = _mm_loadu_si128((const __m128i*)(src + (i >> 1)));
        __m128i lo = _mm_and_si128(b, lomask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lomask);
        lo = _mm_sub_epi8(_mm_xor_si128(lo, _mm_set1_epi8(8)), _mm_set1_epi8(8));
        hi = _mm_sub_epi8(_mm_xor_si128(hi, _mm_set1_epi8(8)), _mm_set1_epi8(8));
        __m128i e0 = _mm_unpacklo_epi8(lo, hi);   /* codes 0..15  */
        __m128i e1 = _mm_unpackhi_epi8(lo, hi);   /* codes 16..31 */
#if AVX_LEVEL >= 3
        _mm512_storeu_ps(dst + i,      _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(e0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(e1)));
#else
        _mm256_storeu_ps(dst + i,      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e0)));
        _mm256_storeu_ps(dst + i + 8,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e0, 8))));
        _mm256_storeu_ps(dst + i + 16, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(e1)));
        _mm256_storeu_ps(dst + i + 24,
                         _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(e1, 8))));
#endif
    }
#endif
    for (; i < n; ++i) {
        uint8_t b = src[i >> 1];
        uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
        dst[i] = (float)((int)(nb ^ 0x8u) - 8);
    }
}

static void w2_to_f32(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        uint8_t b = src[i >> 2];
        uint8_t c = (uint8_t)((b >> (2 * (i & 3))) & 0x3u);
        dst[i] = (float)((int)(c ^ 0x2u) - 2);
    }
}

static void mxfp4_to_f32(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        uint8_t b = src[i >> 1];
        uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
        float v = kMxfp4Mag[nb & 7u];
        dst[i] = (nb & 8u) ? -v : v;
    }
}

/* Four e2m3 codes per three bytes, little-endian LSB-first -- ref_common.h's mxfp6_code and the
 * ABI's rad_load_bits(base, 6, i) are the same selection written as one group unpack. Left scalar
 * for the mxfp4 reason: this plane is read by the oracle and the checker, not by a kernel this
 * plugin serves at speed. */
static void mxfp6_to_f32(const uint8_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* g = src + 3 * (i >> 2);
        uint8_t v;
        switch (i & 3) {
            case 0:  v = (uint8_t)(g[0] & 0x3fu); break;
            case 1:  v = (uint8_t)(((g[1] & 0x0fu) << 2) | (g[0] >> 6)); break;
            case 2:  v = (uint8_t)(((g[2] & 0x03u) << 4) | (g[1] >> 4)); break;
            default: v = (uint8_t)(g[2] >> 2); break;
        }
        const int e = (v >> 3) & 0x3u, m = v & 0x7u;
        const float f = (e == 0) ? (float)m / 8.0f : std::ldexp(1.0f + (float)m / 8.0f, e - 1);
        dst[i] = (v & 0x20u) ? -f : f;
    }
}

/* ================================================================== the generic row */
void row_to_f32(uint32_t dt, const void* src, float* dst, int64_t n) {
    switch (dt) {
        case RAD_F32:    std::memcpy(dst, src, (size_t)n * 4); break;
        case RAD_BF16:   cvt_bf16_to_f32((const uint16_t*)src, dst, n); break;
        case RAD_F16:    cvt_f16_to_f32((const uint16_t*)src, dst, n); break;
        case RAD_F8E4M3: cvt_fp8e4m3_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_F8E5M2: cvt_fp8e5m2_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_I8:     cvt_i8_to_f32((const int8_t*)src, dst, n); break;
        case RAD_I4:     w4_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_I2:     w2_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_FP4E2M1:  mxfp4_to_f32((const uint8_t*)src, dst, n); break;
        case RAD_FP6E2M3:  mxfp6_to_f32((const uint8_t*)src, dst, n); break;
        default:
            /* Everything left is an integer or a bool, none of them hot. The ABI's own accessor is
             * the definition and there is nothing to vectorise about a u8 widen nobody runs. */
            for (int64_t i = 0; i < n; ++i) dst[i] = rad_load_f32(src, dt, i);
            break;
    }
}

void row_from_f32(uint32_t dt, const float* src, void* dst, int64_t n) {
    switch (dt) {
        case RAD_F32:    std::memcpy(dst, src, (size_t)n * 4); break;
        case RAD_BF16:   cvt_f32_to_bf16(src, (uint16_t*)dst, n); break;
        case RAD_F16:    cvt_f32_to_f16(src, (uint16_t*)dst, n); break;
        case RAD_F8E4M3: cvt_f32_to_fp8e4m3(src, (uint8_t*)dst, n); break;
        case RAD_F8E5M2: cvt_f32_to_fp8e5m2(src, (uint8_t*)dst, n); break;
        case RAD_I8:     cvt_f32_to_i8(src, (int8_t*)dst, n); break;
        default:
            /* Every other integer width rounds and saturates as i8 does (avx_common.h: st_elem). */
            for (int64_t i = 0; i < n; ++i) st_elem(dst, dt, i, src[i]);
            break;
    }
}

void row_to_f32_strided(uint32_t dt, const void* src, int64_t stride, float* dst, int64_t n) {
    if (stride == 1) { row_to_f32(dt, src, dst, n); return; }
    /* A GATHER, AND IT IS LEFT SCALAR. vpgatherdd issues one cache line per lane where the scalar
     * loop's loads coalesce in the load queue, and every strided operand in this plugin is a
     * column slice of a fused projection (docs/OPS.md on `z`) -- a handful of rows per launch. Not
     * a hot path, and not worth a second code path that would need its own check. */
    for (int64_t i = 0; i < n; ++i) dst[i] = ld_elem(src, dt, i * stride);
}

/* ================================================================== row reductions
 * FIXED TREES. The accumulator count is written out rather than left to the unroller, because the
 * summation order has to be a property of the source and not of the optimiser: a rebuild that
 * moved the answer in the last ulp would show up in the checker as a tolerance move and be read as
 * whatever change was under test. Four accumulators is what it takes to cover the FMA latency on a modern core
 * -- the sweep that says so is `rad-avx-bench --reduce`. */
float row_sumsq(const float* x, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        vf v0 = vf_loadu(x + i), v1 = vf_loadu(x + i + VF_N);
        vf v2 = vf_loadu(x + i + 2 * VF_N), v3 = vf_loadu(x + i + 3 * VF_N);
        a0 = vf_fma(v0, v0, a0);
        a1 = vf_fma(v1, v1, a1);
        a2 = vf_fma(v2, v2, a2);
        a3 = vf_fma(v3, v3, a3);
    }
    for (; i + VF_N <= n; i += VF_N) { vf v = vf_loadu(x + i); a0 = vf_fma(v, v, a0); }
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += x[i] * x[i];
    return s;
}

float row_sum(const float* x, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        a0 = vf_add(a0, vf_loadu(x + i));
        a1 = vf_add(a1, vf_loadu(x + i + VF_N));
        a2 = vf_add(a2, vf_loadu(x + i + 2 * VF_N));
        a3 = vf_add(a3, vf_loadu(x + i + 3 * VF_N));
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_add(a0, vf_loadu(x + i));
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += x[i];
    return s;
}

float row_amax(const float* x, int64_t n) {
    vf m0 = vf_zero(), m1 = vf_zero();
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        m0 = vf_max(m0, vf_abs(vf_loadu(x + i)));
        m1 = vf_max(m1, vf_abs(vf_loadu(x + i + VF_N)));
    }
    for (; i + VF_N <= n; i += VF_N) m0 = vf_max(m0, vf_abs(vf_loadu(x + i)));
    float m = vf_hmax(vf_max(m0, m1));
    for (; i < n; ++i) { float av = x[i] < 0 ? -x[i] : x[i]; if (av > m) m = av; }
    return m;
}

float row_max(const float* x, int64_t n) {
    if (n <= 0) return -3.402823466e+38f;
    vf m0 = vf_set1(x[0]), m1 = vf_set1(x[0]);
    int64_t i = 0;
    for (; i + 2 * VF_N <= n; i += 2 * VF_N) {
        m0 = vf_max(m0, vf_loadu(x + i));
        m1 = vf_max(m1, vf_loadu(x + i + VF_N));
    }
    for (; i + VF_N <= n; i += VF_N) m0 = vf_max(m0, vf_loadu(x + i));
    float m = vf_hmax(vf_max(m0, m1));
    for (; i < n; ++i) if (x[i] > m) m = x[i];
    return m;
}

/* ================================================================== an operand's row, as f32 */
const float* in_row(const RadTensor* t, int64_t base, int64_t n, float* scr) {
    const int64_t st = laststride(t);
    if (st == 1) {
        /* SUB-BYTE DTYPES AT AN ODD ELEMENT OFFSET are the one case a byte pointer cannot express,
         * and they take the slow path rather than being silently truncated: `byte_at` maps element
         * 2i and 2i+1 to the same byte, so a W4 row starting at an odd element would read the whole
         * row shifted by one nibble -- a permutation, and the kind that produces plausible numbers.
         * Every real W4 row starts on a group boundary, so this costs nothing. */
        const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                      : (t->dtype == RAD_I2 || t->dtype == RAD_FP6E2M3) ? 3 : 0;
        if (!sub || (base & sub) == 0) {
            const void* p = byte_at(t->data, t->dtype, base);
            return row_f32_in(t->dtype, p, scr, n);
        }
    }
    for (int64_t i = 0; i < n; ++i) scr[i] = ldt(t, base + i * st);
    return scr;
}

void out_row(const RadTensor* t, int64_t base, int64_t n, const float* src) {
    const int64_t st = laststride(t);
    if (st == 1) {
        const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                      : (t->dtype == RAD_I2 || t->dtype == RAD_FP6E2M3) ? 3 : 0;
        if (!sub || (base & sub) == 0) {
            row_from_f32(t->dtype, src, (void*)byte_at(t->data, t->dtype, base), n);
            return;
        }
    }
    for (int64_t i = 0; i < n; ++i) stt(t, base + i * st, src[i]);
}

/* ================================================================== exact top-R
 *
 * ONE STEP OF AN EXACT SELECTION, descending by value and on a tie by the LOWER index. libref's
 * `topk_next` is the definition and this is the vectorised reading of it.
 *
 * THE ELIGIBILITY TEST SPLITS AT THE PREVIOUS PICK, and that is what makes it vectorisable at all.
 * The rule is "skip anything above the last pick, and skip the last pick itself and everything
 * before it that equals it" -- which reads as an index comparison and is not a lane predicate. But
 * it is equivalent to two PURE VALUE tests over two ranges:
 *
 *     i <= pi    eligible iff  !(v > pv)  and  v != pv
 *     i >  pi    eligible iff  !(v > pv)
 *
 * so each range is one compare and one blend, and the maximum over them is a vector max. Recovering
 * WHICH index held that maximum is a second pass that early-exits at the first match -- on real
 * data it reads a fraction of the row. libref does one scalar pass with two unpredictable branches
 * per element; at a 248K vocabulary and R = 16 that is eight million of them.
 *
 * `!(v > pv)` RATHER THAN `v <= pv` IS DELIBERATE and is why the comparisons below are the
 * unordered NGT form: for a NaN both are different answers, and libref's `if (v > *pv) continue;`
 * treats a NaN as eligible. Matching it costs nothing here.
 *
 * WHAT DOES NOT MATCH, stated rather than hidden: libref seeds its running best with the FIRST
 * eligible element and only replaces it on a strict `>`, so an eligible NaN encountered first stays
 * the winner for the whole row. A vector max cannot reproduce that, and this returns the largest
 * non-NaN instead. Both are arbitrary -- a NaN logit has no correct selection -- and no path in
 * this project produces one (a masked logit is -inf, which IS ordered and IS handled). */
bool topk_next_f32(const float* v, int64_t n, float* pv, int64_t* pi) {
    const float prev = *pv;
    const int64_t pidx = *pi;

#if AVX_LEVEL == 0
    /* ONE PASS AT THE SCALAR LEVEL, WHICH IS libref's LOOP. The two-pass form below -- a vector
     * maximum and then a scan for the index that held it -- pays only because the first pass is
     * vector work; with a vector of one float it is strictly two passes where libref does one, and
     * it measures slower than the reference it is meant to replace. A fallback that loses to the
     * thing it falls back from is not a fallback, so the scalar level gets its own loop. */
    float best = 0.0f;
    int64_t bi = -1;
    for (int64_t j = 0; j < n; ++j) {
        const float x = v[j];
        if (x > prev) continue;
        if (x == prev && j <= pidx) continue;
        if (bi < 0 || x > best) { best = x; bi = j; }
    }
    if (bi < 0) return false;
    *pv = best;
    *pi = bi;
    return true;
#else
    const float NEG = -INFINITY;
    /* The split point: indices at or below `pidx` must also differ from `prev`. */
    const int64_t split = (pidx + 1 > 0) ? (pidx + 1 < n ? pidx + 1 : n) : 0;
    float best = NEG;
    int64_t i = 0;

    {
        const vf vprev = vf_set1(prev), vneg = vf_set1(NEG);
        vf acc = vf_set1(NEG);
        /* Range one: below or at the previous pick -- equality is excluded here. */
        for (; i + VF_N <= split; i += VF_N) {
            const vf x = vf_loadu(v + i);
#if AVX_LEVEL == 3
            const __mmask16 ok = _kand_mask16(_mm512_cmp_ps_mask(x, vprev, _CMP_NGT_UQ),
                                              _mm512_cmp_ps_mask(x, vprev, _CMP_NEQ_UQ));
            acc = vf_max(acc, _mm512_mask_blend_ps(ok, vneg, x));
#else
            const __m256 ok = _mm256_and_ps(_mm256_cmp_ps(x, vprev, _CMP_NGT_UQ),
                                            _mm256_cmp_ps(x, vprev, _CMP_NEQ_UQ));
            acc = vf_max(acc, _mm256_blendv_ps(vneg, x, ok));
#endif
        }
        for (; i < split; ++i) {
            const float x = v[i];
            if (x > prev || x == prev) continue;
            if (x > best) best = x;
        }
        /* Range two: past the previous pick -- equality is allowed. */
        for (; i + VF_N <= n; i += VF_N) {
            const vf x = vf_loadu(v + i);
#if AVX_LEVEL == 3
            const __mmask16 ok = _mm512_cmp_ps_mask(x, vprev, _CMP_NGT_UQ);
            acc = vf_max(acc, _mm512_mask_blend_ps(ok, vneg, x));
#else
            const __m256 ok = _mm256_cmp_ps(x, vprev, _CMP_NGT_UQ);
            acc = vf_max(acc, _mm256_blendv_ps(vneg, x, ok));
#endif
        }
        const float vm = vf_hmax(acc);
        if (vm > best) best = vm;
    }
    for (; i < n; ++i) {
        const float x = v[i];
        if (x > prev) continue;
        if (x == prev && i <= pidx) continue;
        if (x > best) best = x;
    }

    /* THE LOWEST INDEX HOLDING THAT MAXIMUM -- "ties to the lower column" -- AND THIS PASS IS
     * VECTORISED TOO, because it costs as much as the maximum does when it is not.
     *
     * The maximum can sit anywhere in the row, so a scalar scan for it reads n/2 elements on
     * average: over a 32000-logit row that is sixteen thousand unpredictable compares, and it
     * dominates `sample_argmax`. Comparing a whole vector against `best` and taking the lowest set
     * bit of the mask turns that into n/2/VF_N iterations.
     *
     * THE LOW RANGE IS SCANNED FIRST AND THAT IS NOT A DETAIL. "Lowest index" means lowest over the
     * WHOLE row, so a match below the previous pick beats one above it -- and scanning the long
     * vector range first and only then the short scalar one returns the wrong index exactly when
     * the maximum occurs twice, once on each side. That failure only bites on ties: it does not
     * show at f32 or at the scalar level, and it shows at bf16, where a narrow row quantised to a
     * handful of distinct values makes a repeated maximum common.
     *
     * Below the previous pick the extra `x != prev` test applies, and that range is at most pidx+1
     * long -- zero for an argmax, R-1 for the R-th selection -- so it stays scalar. Past it,
     * eligibility is exactly `!(x > prev)` and `x == best` already implies it, since `best` was
     * drawn from eligible lanes and is therefore <= prev. */
    for (int64_t j = 0; j < split; ++j) {
        const float x = v[j];
        if (x != best || x == prev) continue;
        *pv = best;
        *pi = j;
        return true;
    }
    {
        int64_t j = split;
        const vf vbest = vf_set1(best);
        for (; j + VF_N <= n; j += VF_N) {
            const vf x = vf_loadu(v + j);
#if AVX_LEVEL == 3
            const unsigned msk = (unsigned)_mm512_cmpeq_ps_mask(x, vbest);
#else
            const unsigned msk =
                (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, vbest, _CMP_EQ_OQ));
#endif
            if (msk) { *pv = best; *pi = j + __builtin_ctz(msk); return true; }
        }
        for (; j < n; ++j)
            if (v[j] == best) { *pv = best; *pi = j; return true; }
    }
    return false;
#endif
}

}  /* inline namespace */
}  /* namespace avx */

/* ================================================================== the machine's FMA ceiling
 *
 * THE CEILING HAS TO BE MEASURED WITH THE FLAGS THE KERNELS ARE COMPILED WITH, which is why this
 * lives here -- in a file compiled once per ISA level -- and not in rad-avx-bench, which is
 * baseline-compiled like the dispatcher and can only ever emit SSE2. A bench that reported the
 * scalar chain as "the machine's peak" would be reporting something ten times below what a GEMM
 * at this level is up against, and every kernel would look like it was doing well.
 *
 * WHAT IT MEASURES: a dependency-free chain of independent FMA accumulators, all in registers,
 * nothing touching memory. What comes out is what the machine RETIRES, including whatever clock it
 * actually holds under a sustained vector load -- which on a part with an AVX-512 frequency offset
 * is not the advertised boost, and is exactly the number a kernel is competing with.
 *
 * THE ACCUMULATOR COUNT IS THE REGISTER FILE, and getting it wrong understates the ceiling rather
 * than overstating it -- which is worse, because every kernel then looks good against it. Sixteen
 * accumulators plus the two constants is EIGHTEEN live vectors, which fits AVX-512's 32 registers
 * and spills on AVX2's and AVX's 16. With sixteen everywhere the AVX2 peak comes out barely above
 * the AVX one -- a step of a few percent for turning two instructions into one, where the ports
 * say 2x -- because the spill is inside the loop. Ten accumulators covers a 4-5 cycle FMA latency
 * against 2 FMA ports with room to spare and leaves six registers free.
 *
 * `volatile` on the sink, so the whole loop is not deleted. The accumulators are read out through
 * it; without that the compiler is entirely within its rights to remove the work and report an
 * infinite ceiling, which is the classic way this measurement is wrong. */
extern "C" double AVX_FN(avx_peak_fma)(int64_t iters) {
    using namespace avx;
#if AVX_LEVEL == 1 || AVX_LEVEL == 2
    enum { NACC = 10 };          /* 16 architectural YMM registers */
#else
    enum { NACC = 16 };          /* 32 ZMM at AVX-512; scalar has no such limit */
#endif
    vf acc[NACC];
    for (int i = 0; i < NACC; ++i) acc[i] = vf_set1((float)(i + 1) * 1e-6f);
    const vf m = vf_set1(1.0000001f), b = vf_set1(1e-7f);
    for (int64_t it = 0; it < iters; ++it)
        for (int i = 0; i < NACC; ++i) acc[i] = vf_fma(acc[i], m, b);
    float sink = 0;
    for (int i = 0; i < NACC; ++i) sink += vf_hsum(acc[i]);
    volatile float keep = sink;
    (void)keep;
    /* 2 flops an FMA, NACC of them an iteration, VF_N lanes each. */
    return 2.0 * (double)iters * (double)NACC * (double)VF_N;
}
