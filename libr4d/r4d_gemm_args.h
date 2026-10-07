/* r4d_gemm_args.h -- the skinny GEMMs' operand list and their split law.
 *
 * The launch law -- ladders, caps, `pick_sk`, a bound rule per weight grid -- lives beside the
 * kernels it decides for, so a kernel library built out of tree carries its own rule instead of
 * depending on one held in the engine. The declared tunable axes (r4d_rows.cpp) are the half of
 * that a tuner walks; this is the half that answers when a shape is left on the axis defaults.
 *
 * The operand plumbing is here for the ordinary reason: eleven kernels across eleven translation
 * units read the same two operand lists -- [A, B, Y] for the bf16 grid and [A, a_scale, B,
 * b_scale, Y, a_sum, B_ref] for every quantised one -- because they implement two ops.
 */
#ifndef R4D_GEMM_ARGS_H
#define R4D_GEMM_ARGS_H

#include "r4d.h"
#include "r4d_args.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <exception>


/* The build passes the same -DR4D_GEMM_*_GROUP to libr4d's translation units and to this one, so
 * these cannot drift apart silently -- a weight file written for one group and read by a kernel
 * compiled for another is wrong by a factor of nothing visible. */
#ifndef R4D_GEMM_W4_GROUP
#define R4D_GEMM_W4_GROUP 128
#endif
#ifndef R4D_GEMM_W8_GROUP
#define R4D_GEMM_W8_GROUP 128
#endif
#ifndef R4D_GEMM_W4A8_GROUP
#define R4D_GEMM_W4A8_GROUP 128
#endif
#ifndef R4D_GEMM_W8A8_GROUP
#define R4D_GEMM_W8A8_GROUP 128
#endif
#ifndef R4D_GEMM_W2A8_GROUP
#define R4D_GEMM_W2A8_GROUP 128
#endif
static_assert(R4D_W4_GROUP == R4D_GEMM_W4_GROUP,   "w4 group differs from libr4d's build");
static_assert(R4D_W8_GROUP == R4D_GEMM_W8_GROUP,   "w8 group differs from libr4d's build");
static_assert(R4D_W4_GROUP == R4D_GEMM_W4A8_GROUP, "w4a8 group differs from libr4d's build");
static_assert(R4D_W8_GROUP == R4D_GEMM_W8A8_GROUP, "w8a8 group differs from libr4d's build");
static_assert(R4D_W2_GROUP == R4D_GEMM_W2A8_GROUP, "w2a8 group differs from libr4d's build");


static inline int gemm_caught(const char* who, const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[r4d] %s threw: %s\n", who, what ? what : "(null)");
    }
    return R4D_NO(RAD_E_DEVICE);
}

#define R4D_TRY(who, ...)                                                          \
    do {                                                                           \
        try { __VA_ARGS__; }                                                       \
        catch (const std::exception& e_) { return gemm_caught(who, e_.what()); }   \
        catch (...) { return gemm_caught(who, "unknown exception"); }              \
    } while (0)

/* ------------------------------------------------------------------ the split count
 *
 * libr4d's skinny GEMMs put split-K IN THE WORKGROUP: SK k-splits per column tile, reduced in LDS,
 * which is what fills a 64-CU part at M=1. The legality rule is K % (SK * unit) == 0, so no single
 * SK is legal at every K -- and a DEFAULT that is illegal at a shape would make an untuned install
 * FAIL rather than be slow, which is the wrong way round (spec §15).
 *
 * So the `sk` axis carries 0 as its DEFAULT and 0 means "the largest legal split from this K",
 * resolved here. The candidate ladder is the set of split counts that tuned configurations use, in
 * descending order, so an untuned install lands on a split count some shape was tuned to rather
 * than on 1. The cost of the sentinel is that two shapes with different K both running
 * `sk = 0` get different split counts; --debug-graph prints the chosen point, so a 0 there has to
 * be read as "the law decided" and not as a number.
 */
/* THE LAUNCH BOUND IS PART OF A SPLIT'S LEGALITY.
 *
 * libr4d's skinny kernels carry __launch_bounds__ that TIGHTEN as MB and NPW grow -- w2a8's is
 * (MB >= 4) ? 256 : ((MB >= 2 || NPW >= 4) ? 512 : 1024) -- and a block of WV*SK*32 threads wider
 * than its own bound is a launch failure, which surfaces later as a device error from whatever
 * happens to synchronise next. A split count picked against a flat 1024 is therefore not
 * necessarily legal.
 *
 * MB comes from M (`ceil(M/16)`, capped at 4), so this is an M-dependent trap that only springs
 * under CONCURRENCY. On the DFlash2 2-bit draft head, N = 248320: M >= 49 gives MB = 4 and a bound
 * of 256, where sk = 10 asks for 320 threads a block; M in [17, 48] gives MB = 2 and a bound of
 * 512, where sk = 20 asks for 640. Any bench that runs M <= 16 sees none of it. The engine refuses
 * the op at issue and fails the step, exactly as spec 17 says it must, so the symptom is
 * concurrent requests coming back empty rather than a wrong number.
 *
 * EACH CALLER NAMES ITS OWN RULE, because they genuinely differ and one conservative rule costs
 * real throughput: the bf16, w4a16, w8a16 and mxfp4 kernels declare a flat 1024 (or no
 * __launch_bounds__ at all, which is the same 1024), and capping them as if they were w2a8 costs
 * prefill throughput for nothing. w4a8 tightens on NPW alone and w8a8 on MT*NPW, both because the
 * bound is a REGISTER budget, which their own headers say at length. Where a kernel's macro is
 * switched off by a build flag the TIGHT branch is named here anyway: over-constraining a build
 * that relaxed its bound loses a little parallelism, and under-constraining one that did not is the
 * launch failure above.
 */
enum BoundRule { BR_FLAT = 0, BR_W2A8, BR_NPW, BR_MBNPW };

static inline int block_bound(BoundRule r, int mb, int npw) {
    switch (r) {
        case BR_W2A8:  return (mb >= 4) ? 256 : (((mb >= 2) || (npw >= 4)) ? 512 : 1024);
        case BR_NPW:   return (npw >= 8) ? 256 : ((npw >= 4) ? 512 : 1024);
        case BR_MBNPW: return (mb * npw >= 8) ? 256 : ((mb * npw >= 4) ? 512 : 1024);
        default:       return 1024;
    }
}

static inline int pick_sk(long long K, long long unit, int wv, int bound) {
    /* The ladder enumerates the candidate splits the law above can reach, so its top entry is the
     * largest split any untuned shape will be given. 20 is on it because K = 5120 -- every
     * projection in this model -- divides by it, and 20 is the fastest legal split at that K on the
     * GDN a|b projection, which runs forty-eight times a step. `cap` still bounds the block at 1024
     * threads, which is what keeps wv = 2 from reaching 20 at all.
     *
     * 32 IS LEFT OUT ON PURPOSE. It is legal at K = 5120 and it is slower there than 20, so
     * including it would move the default AWAY from the fastest split purely to reach a bigger
     * number. The ladder tops out where the fastest split is, not where legality ends. */
    static const int ladder[] = { 20, 16, 10, 8, 5, 4, 2, 1 };
    const int cap = (wv > 0) ? (bound / (wv * 32)) : 1;   /* threads/block <= this MB/NPW's bound */
    for (int s : ladder) {
        if (s > cap) continue;
        if (unit > 0 && (K % ((long long)s * unit)) == 0) return s;
    }
    return 1;
}

/* "THE LARGEST LEGAL SPLIT" IS THE WRONG LAW WHERE N ALREADY SUPPLIES THE PARALLELISM, and the
 * bf16 skinny GEMM is where that shows. The split exists to fill a 64-CU part at M = 1; a shape
 * with eighty N tiles is already full, and splitting it twenty ways is 1600 workgroups whose only
 * extra effect is the reduction. At K = 5120 the fastest split falls as N grows: at N = 48 (three
 * N tiles) the split IS the parallelism and the largest legal one wins; at N = 256 (sixteen tiles)
 * a split of 4 wins; at N = 1280 (eighty tiles) 1 or 2 does.
 *
 * So the law is the SMALLEST split whose `n_tiles * sk` reaches a target workgroup count, falling
 * back to the largest legal split at small N because nothing smaller reaches it. kFlatTileTarget
 * is that count; it is the same quantity `narrow_ksplit` targets in the fp8 kernel, where it is
 * called kSplitTarget.
 *
 * N = 48 IS THE ONLY TRUNK CALLER (the GDN a|b projection, forty-eight times a step); N = 1280 and
 * N = 256 are the DFlash2 drafter's. So at the trunk shape this law reduces to the largest legal
 * split, and the shapes it actually moves are the ones the drafter PROPOSES. */
enum { kFlatTileTarget = 128 };

static inline int pick_sk_for(long long n_tiles, long long K, long long unit, int wv, int bound) {
    static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20 };   /* ascending: the SMALLEST that fits */
    const int cap = (wv > 0) ? (bound / (wv * 32)) : 1;
    int best = 0;
    for (int s : ladder) {
        if (s > cap) continue;
        if (unit > 0 && (K % ((long long)s * unit))) continue;
        best = s;                                   /* legal; remember it in case none reaches */
        if (n_tiles * s >= (long long)kFlatTileTarget) return s;
    }
    return best > 0 ? best : 1;                     /* nothing reaches the target: the largest legal */
}

/* Resolve the tuned point into (WV, SK, MB, NPW, NT) for a shape whose split unit is `unit`. */
struct Cfg { int wv, sk, mb, npw, nt; };

/* `n_tiles` is how many 16-row N tiles the shape has, and it is what turns "the largest legal
 * split" into "enough split to fill the machine". Zero keeps the largest legal split, which is
 * right wherever the split is the only source of parallelism. See the note at pick_sk_for. */
static inline int cfg_for(const RadArgs* a, long long K, long long M, long long unit, BoundRule br,
                   Cfg* out, long long n_tiles = 0);

static inline int cfg_for(const RadArgs* a, long long K, long long M, long long unit, BoundRule br,
                   Cfg* out, long long n_tiles) {
    R4dGemmCfg cv;
    r4d_cfg_gemm(a, &cv);
    const R4dGemmCfg* c = &cv;
    out->wv  = c->wv  > 0 ? c->wv  : 1;
    /* MB IS HOW MANY ROW TILES ONE BLOCK COVERS, and mb <= 0 means "all of them", the same way
     * sk <= 0 means "the largest legal split". It matters far more than it looks: grid.y is
     * ceil(ceil(M/16)/MB), and every block in a grid.y column reads THE WHOLE WEIGHT for its N
     * tile. At these shapes the weight is the entire cost, so MB = 1 at M = 64 reads it four times
     * -- on the DFlash2 drafter's fc, [5120, 25600] and 250 MiB, that is a gibibyte moved at peak
     * bandwidth to compute a 64-row slice. Covering every row tile in one block is right whenever
     * the weight is bigger than the activation, which is every caller here. */
    out->mb  = c->mb  > 0 ? c->mb : (int)((M + 15) / 16);
    if (out->mb < 1) out->mb = 1;
    if (out->mb > 4) out->mb = 4;
    /* ...AND IN THE DECODE BAND IT IS THE OPPOSITE RULE, because there the weight is not the cost:
     * the GRID is. `grid.y` is ceil(ceil(M/16)/MB), so MB = ceil(M/16) makes it ONE -- and at the
     * GDN a|b projection, N = 48, that is a THREE-BLOCK grid on a 32-WGP part for the whole kernel.
     * MB = 1 makes it twelve at M = 64 and reads a 0.5 MiB weight four times, and those extra
     * reads cost far less than the idle machine does. Across the decode band MB = 1 is the fastest
     * choice at every (N, M) these callers run; MB = 2 edges it only at the widest N and the top of
     * the band, by less than a second threshold would be worth.
     *
     * FOUR ROW TILES IS THE BAND, and the bound is what makes it safe rather than a guess: a
     * PREFILL chunk is hundreds of row tiles, where grid.y is already large and MB = 1 would read
     * the weight a hundred and seventy times. `mtiles <= 4` is exactly M <= 64, which is exactly
     * the decode band this kernel's callers run (C * (1 + n_spec) at n_spec 7 tops out at 64), and
     * prefill is untouched by construction.
     *
     * BR_FLAT ONLY, and that is what leaves the output bits unchanged. `block_bound` ignores MB
     * for BR_FLAT, so the split `pick_sk_for` answers below does not move -- and with the same
     * split every output element is still one wave's accumulation over the same K slices in the
     * same ascending order. MB moves which block owns the element, not what it adds. On a rule
     * whose bound DOES read MB (BR_W2A8, BR_MBNPW) this would change the split and therefore the
     * bits, which is why it is gated rather than applied to every auto.
     *
     * `R4D_GEMM_MB1=0` disables this branch and leaves MB on the shape-derived default. */
    static const bool mb1 = [] { const char* e = std::getenv("R4D_GEMM_MB1");
                                 return !(e && e[0] == '0'); }();
    if (mb1 && c->mb <= 0 && br == BR_FLAT && ((M + 15) / 16) <= 4) out->mb = 1;
    /* ...AND PAST THE DECODE BAND WHEN N IS NARROW. Above 64 rows an N of 128 or more belongs to
     * the tiled row (r4d_gemm_bf16_nt_tiled.hip), and what stays here is the narrow projections --
     * the GDN a|b at 48, a shared-expert gate at 1 -- whose weight is a few hundred KiB against an
     * activation of megabytes, so the "weight is the cost" reading above is backwards for them.
     * Measured at K = 2560 over M 80..2048 and N 16/48/112, MB = 1 is 2-5x faster than covering
     * every row tile at every point (M 2048, N 48: 27.8 -> 14.9 us; N 16: 27.9 -> 8.8). The same
     * BR_FLAT argument as above keeps the bits. */
    if (c->mb <= 0 && br == BR_FLAT && n_tiles > 0 && n_tiles < 8) out->mb = 1;
    out->npw = c->npw > 0 ? c->npw : 1;
    out->nt  = c->nt;
    /* W2A8 PAST ONE ROW TILE NEEDS A WIDER BLOCK, not a taller one, and the shape-derived default
     * gives it a taller one. The DFlash2 draft head is the only w2a8 caller and it runs
     * M = n_seq * steps, so M crosses 16 at the SECOND concurrent sequence: a single-stream bench
     * never reaches this branch at all.
     *
     * At one row tile the kernel runs at memory bandwidth on its 170.5 MiB weight, N = 124160 and
     * K = 5120. Past one row tile, at NPW = 1, it falls to roughly a third of the bus even though
     * the weight read has not changed -- MB covers every row tile in one block throughout -- so the
     * collapse is entirely the block shape. Widening the block over N recovers most of it: NPW = 4
     * gives each block four N tiles to hide the dequantise behind, where NPW = 1 has only its own.
     * MB is pinned at 2 rather than left to grow because MB = 4 with NPW = 4 spills -- accumulators
     * are MT * NPW -- and reading the weight twice is cheaper than spilling, which is the same
     * conclusion the kernel's own header reaches from its sweep.
     *
     * Only mb = 0 is touched: a tuned mb is the tuner's decision and must survive. */
    /* AND THE SPLIT IS PINNED WITH THEM, because `pick_sk` takes the LARGEST split the bound
     * allows and the largest is not the fastest here. At K = 5120 the unit is 128, so the legal
     * counts are the divisors of 40 up to bound/(wv*32) = 16, and pick_sk answers 10. Eight is
     * uniformly faster than ten across the whole M band this branch covers. The one cost step
     * inside that band is where grid.y grows and the weight is read again; it sits in the same
     * place at either split count.
     *
     * MB = 4 WOULD REMOVE THE SECOND WEIGHT READ AND IS STILL WRONG: at NPW = 4 it spills, and
     * below NPW = 4 the block is too narrow to hide the dequantise. Reading the weight twice
     * really is cheaper than any block shape that reads it once, which is the same conclusion the
     * kernel's own header reaches. The lever that IS left is a vocabulary shard -- see
     * rad_block_dflash2.h.
     *
     * MB = 3 IS THE COUNT BETWEEN THEM AND IT IS ALSO WRONG, which is worth stating because the
     * analogous hole at three in the fp8 GEMM's NT ladder is a real loss there (r4d_gemm_fp8a8.hip,
     * "...AND THE TOKEN-TILE COUNT HAD A HOLE IN IT AT THREE"). At three row tiles -- M = 33..48 --
     * MB = 2 makes grid.y two and reads the whole weight twice while MB = 3 makes it one, and
     * MB = 3 is still slower: MT * NPW accumulators is twelve against eight, which costs more than
     * the read it saves. It is slower even at an M where grid.y and the bytes moved are identical
     * between them, so the accumulator count is the whole of it. The second weight read past 32
     * rows stands.
     *
     * A pin that this K cannot divide, or that overruns the bound, falls back to pick_sk rather
     * than becoming a refusal: this branch is the AUTO path and auto must serve every shape. */
    int sk_pin = 0;
    if (br == BR_W2A8 && c->mb <= 0 && out->mb >= 2) {
        out->mb  = 2;
        out->npw = 4;
        sk_pin   = 8;
    }
    const int bound = block_bound(br, out->mb, out->npw);
    if (sk_pin && (out->wv * sk_pin * 32 > bound ||
                   (unit > 0 && (K % ((long long)sk_pin * unit)))))
        sk_pin = 0;
    out->sk  = c->sk > 0 ? c->sk
                         : (sk_pin ? sk_pin
                                   : (n_tiles > 0 ? pick_sk_for(n_tiles, K, unit, out->wv, bound)
                                                  : pick_sk(K, unit, out->wv, bound)));
    /* A NAMED split that overruns the bound is a REFUSAL, not a device error. The tuner has to see
     * "does not serve this shape" so it never records the candidate, and the selector must not be
     * able to hand a cached name to a shape whose MB has since grown. */
    if (out->wv * out->sk * 32 > bound) return R4D_NO(RAD_E_UNSUPPORTED);
    /* A NAMED split count that this K cannot divide is a refusal, not a silent clamp: the tuner
     * has to see that this candidate does not serve this shape, or it will record a measurement
     * for a configuration that never ran. */
    if (unit > 0 && (K % ((long long)out->sk * unit))) return R4D_NO(RAD_E_SHAPE);
    if (out->wv * out->sk * 32 > 1024) return R4D_NO(RAD_E_SHAPE);
    /* R4D_GEMMCFG=1 prints each DISTINCT (tuned point, K, M, n_tiles) this resolved, once. The rule
     * above is shape-dependent and the harness cannot see which branch it took. */
    static const bool dbg = [] { const char* e = std::getenv("R4D_GEMMCFG");
                                 return e && e[0] && e[0] != '0'; }();
    if (dbg) {
        /* Once per DISTINCT key, to at most 64 of them: the engine resolves this on every issue
         * of every GEMM row and a line each would be the log. */
        static uint64_t seen[64];
        static int nseen = 0;
        const uint64_t key = ((uint64_t)(unsigned)(c->wv * 41 + c->sk * 7 + c->mb * 3 + c->npw) << 56)
                           ^ ((uint64_t)(unsigned)br << 52)
                           ^ ((uint64_t)(unsigned)n_tiles << 26) ^ (uint64_t)(unsigned)M
                           ^ ((uint64_t)(unsigned)K << 13);
        int i = 0;
        while (i < nseen && seen[i] != key) ++i;
        if (i == nseen && nseen < 64) {
            seen[nseen++] = key;
            std::fprintf(stderr,
                "[gemmcfg] K=%lld M=%lld tiles=%lld br=%d -> wv=%d sk=%d mb=%d npw=%d nt=%d\n",
                (long long)K, (long long)M, (long long)n_tiles, (int)br,
                out->wv, out->sk, out->mb, out->npw, out->nt);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ operand plumbing */

enum { G_A = 0, G_B, G_Y };                                     /* gemm_nt          */
enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y, Q_ASUM, Q_BREF };  /* gemm_nt_q       */
/* gemm_nt_q_paro -- the partition-select form. NEW NAMES AND NEW POSITIONS: the two extra
 * activation partitions and their scales sit between `a_scale` and the weight, so the list does not
 * overlap the Q_* indices above and neither can be read with the other's constants. */
enum { QP_A = 0, QP_ASCALE, QP_A2, QP_ASCALE2, QP_A3, QP_ASCALE3, QP_B, QP_BSCALE, QP_Y };

struct GemmOps {
    /* A flattened view of `a`, when the caller declared it rank 3; see gemm_ops. Held by value
     * because the operand array it was folded from belongs to the core. */
    RadTensor a_flat, y_flat;
    const RadTensor *a, *ascale, *b, *bscale, *y, *asum, *bref;
    long long M, N, K;
};

/* The operand view `gemm_nt_q_paro` reads. `a2`/`ascale2` and `a3`/`ascale3` are optional partitions;
 * an absent pointer means the previous partition extends to the next boundary. */
struct GemmOpsParo {
    RadTensor a_flat, y_flat;
    const RadTensor *a, *ascale, *a2, *ascale2, *a3, *ascale3, *b, *bscale, *y;
    long long M, N, K;
};

/* THE MXFP4 GEMMS' ACTIVATION SCALE GRID, read off the operand rather than assumed. One f32 a row
 * (`[M]` or `[M][1]`: a per-token quantiser) is group 0; one per (row, 128 of K) (`[M][ceil(K/128)]`:
 * the block-scaled fp8 family's quantisers, which every fused producer in arch/common writes) is
 * group 128, read at the operand's own row stride. Anything else is refused by shape: a grid read
 * at the wrong width is an answer that is merely wrong.
 *
 * RADIANCE'S OWN MXFP4 PATH IS ALWAYS GROUP 0: `a_scale` is one f32 per row (`[M,1]`), so the
 * per-(row,128) grid is only reachable from a caller that supplies it. `mx_ascale_grid` therefore
 * resolves group 0 for every row this plugin registers today, and the ABLK code path the MXFP4
 * kernels gate is inert for our path. */
static inline int mx_ascale_grid(const GemmOps& g, int* group, int* rs) {
    const RadTensor* s = g.ascale;
    if (!s || g.M <= 0) return R4D_NO(RAD_E_INVAL);
    long long cols = 1, rstride = 1;
    if (s->rank >= 2) {
        cols    = s->shape[s->rank - 1];
        rstride = s->stride[s->rank - 2];
        if (s->shape[s->rank - 2] < g.M) return R4D_NO(RAD_E_SHAPE);
    } else if (s->rank == 1) {
        if (s->shape[0] < g.M) return R4D_NO(RAD_E_SHAPE);
        cols = s->shape[0] / g.M;
        rstride = cols;
    }
    const long long kb = (g.K + 127) / 128;
    if (cols == 1 && (s->rank < 2 || rstride == 1)) { *group = 0; *rs = 1; return RAD_OK; }
    if (cols == kb && rstride >= kb) { *group = 128; *rs = (int)rstride; return RAD_OK; }
    return R4D_NO(RAD_E_SHAPE);
}

/* Common shape and layout checks. A is [M,K] row-major, C is [M,N] row-major and the weight is
 * an opaque packed blob whose element count the layout hook owns -- so the weight's SHAPE is not
 * checked here beyond being present, which is deliberate: the byte count that matters is the one
 * rad-convert wrote from RadLayout.bytes, and re-deriving it from N and K would be a second
 * source for one number. */
static inline int gemm_ops(const RadArgs* a, GemmOps* g, int quantised) {
    std::memset(g, 0, sizeof(*g));
    if (quantised) {
        g->a      = r4d_opt(a, Q_A);
        g->ascale = r4d_opt(a, Q_ASCALE);
        g->b      = r4d_opt(a, Q_B);
        g->bscale = r4d_opt(a, Q_BSCALE);
        g->y      = r4d_opt(a, Q_Y);
        g->asum   = r4d_opt(a, Q_ASUM);
        g->bref   = r4d_opt(a, Q_BREF);
        if (!g->a || !g->b || !g->bscale || !g->y) return R4D_NO(RAD_E_INVAL);
    } else {
        g->a = r4d_opt(a, G_A);
        g->b = r4d_opt(a, G_B);
        g->y = r4d_opt(a, G_Y);
        if (!g->a || !g->b || !g->y) return R4D_NO(RAD_E_INVAL);
    }
    /* A CONTIGUOUS RANK-3 ACTIVATION IS A RANK-2 ACTIVATION, and refusing it would refuse a real
     * model: the bf16 plugin's gated delta net writes its output into a [tokens, heads,
     * head_v] buffer -- rank 3 because `gdn_chunk_scan` declares its `o` operand that way -- and
     * hands the same bytes to `gemm_nt` for the out projection. libref computes that correctly,
     * because it takes K from the op's K PARAMETER and reads A as M x K whatever the declared rank;
     * this shim derived K from shape[1] instead, which is the HEAD COUNT, so it refused the rank
     * rather than computing with 48 where K is 3072. The fp8 plugin never reached it only because
     * its quantiser writes a separate rank-2 codes buffer.
     *
     * The fold is exact for a CONTIGUOUS tensor and gives the same M x K the parameter names. It is
     * held by value here rather than written back through g->a: those tensors are the core's
     * operand array, which every other kernel in the step also reads.
     *
     * A STRIDED rank-3 tensor is still refused, and must be: it is not a rank-2 one. */
    if (g->a->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (g->a->rank > 2) {
        if (!r4d_contig(g->a)) return R4D_NO(RAD_E_STRIDE);
        g->a_flat = *g->a;
        int64_t k = 1;
        for (uint32_t d = 1; d < g->a->rank; ++d) k *= g->a->shape[d];
        g->a_flat.rank = 2;
        g->a_flat.shape[1] = k;  g->a_flat.stride[1] = 1;
        g->a_flat.stride[0] = k;
        g->a = &g->a_flat;
    }
    if (g->y->rank < 2) return R4D_NO(RAD_E_SHAPE);
    /* ...AND THE SAME ON THE OUTPUT SIDE, for the same reason and from the same plugin. The gated
     * attention writes its query-and-gate projection into a [tokens, heads, 2*head_dim] buffer --
     * rank 3 so that q and gate are ordinary column slices of the last dimension, which is what
     * makes the interleaving addressable at all -- and N is the product of the trailing two. */
    if (g->y->rank > 2) {
        if (!r4d_contig(g->y)) return R4D_NO(RAD_E_STRIDE);
        g->y_flat = *g->y;
        int64_t n = 1;
        for (uint32_t d = 1; d < g->y->rank; ++d) n *= g->y->shape[d];
        g->y_flat.rank = 2;
        g->y_flat.shape[1] = n;  g->y_flat.stride[1] = 1;
        g->y_flat.stride[0] = n;
        g->y = &g->y_flat;
    }
    if (!r4d_contig(g->a) || !r4d_contig(g->y)) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 16)) return R4D_NO(RAD_E_ALIGN);

    g->M = g->a->shape[0];
    g->K = g->a->shape[1];
    g->N = g->y->shape[1];
    if (g->y->shape[0] != g->M) return R4D_NO(RAD_E_SHAPE);
    if (g->M < 1 || g->N < 1 || g->K < 1) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

/* gemm_nt_q_paro's operand reader. Same checks as gemm_ops for the required operands, plus the two
 * optional partitions: a partition present without its scale is refused rather than run against a
 * scale the caller believes was applied. The rank-3 fold is repeated so a merged activation handed
 * as [tokens, heads, width] is read as a rank-2 [tokens, heads*width], exactly as gemm_ops does. */
static inline int gemm_ops_paro(const RadArgs* a, GemmOpsParo* g) {
    std::memset(g, 0, sizeof(*g));
    g->a       = r4d_opt(a, QP_A);
    g->ascale  = r4d_opt(a, QP_ASCALE);
    g->a2      = r4d_opt(a, QP_A2);
    g->ascale2 = r4d_opt(a, QP_ASCALE2);
    g->a3      = r4d_opt(a, QP_A3);
    g->ascale3 = r4d_opt(a, QP_ASCALE3);
    g->b       = r4d_opt(a, QP_B);
    g->bscale  = r4d_opt(a, QP_BSCALE);
    g->y       = r4d_opt(a, QP_Y);
    if (!g->a || !g->ascale || !g->b || !g->bscale || !g->y) return R4D_NO(RAD_E_INVAL);
    if (g->a2 && !g->ascale2) return R4D_NO(RAD_E_INVAL);
    if (g->a3 && !g->ascale3) return R4D_NO(RAD_E_INVAL);
    if (g->a->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (g->a->rank > 2) {
        if (!r4d_contig(g->a)) return R4D_NO(RAD_E_STRIDE);
        g->a_flat = *g->a;
        int64_t k = 1;
        for (uint32_t d = 1; d < g->a->rank; ++d) k *= g->a->shape[d];
        g->a_flat.rank = 2;
        g->a_flat.shape[1] = k;  g->a_flat.stride[1] = 1;
        g->a_flat.stride[0] = k;
        g->a = &g->a_flat;
    }
    if (g->y->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (g->y->rank > 2) {
        if (!r4d_contig(g->y)) return R4D_NO(RAD_E_STRIDE);
        g->y_flat = *g->y;
        int64_t n = 1;
        for (uint32_t d = 1; d < g->y->rank; ++d) n *= g->y->shape[d];
        g->y_flat.rank = 2;
        g->y_flat.shape[1] = n;  g->y_flat.stride[1] = 1;
        g->y_flat.stride[0] = n;
        g->y = &g->y_flat;
    }
    if (!r4d_contig(g->a) || !r4d_contig(g->y)) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 16)) return R4D_NO(RAD_E_ALIGN);
    g->M = g->a->shape[0];
    g->K = g->a->shape[1];
    g->N = g->y->shape[1];
    if (g->y->shape[0] != g->M) return R4D_NO(RAD_E_SHAPE);
    if (g->M < 1 || g->N < 1 || g->K < 1) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}



/* The asymmetric grid is a different WEIGHT FORMAT, not a different call: the nibble is read
 * UNSIGNED and the stored per-group zero is corrected out with the activation's per-group row
 * sums. So it is selected by `dtype`, which is what the weight file was written for, and never by
 * whether a_sum happens to be present -- guessing from an operand would make a caller that forgot
 * to pass the sums silently compute the symmetric answer against asymmetric bytes. */
static inline int is_asym(const RadArgs* a) {
    const char* dt = r4d_gets(a, "dtype");
    return dt && std::strstr(dt, "_asym") != nullptr;
}

/* The tiled and prefill kernels take a single opaque tile VARIANT into libr4d's own tile switch.
 * The block width BN is a property of the variant, and N must be a multiple of it -- which the ROW
 * cannot express, because BN varies by variant. So the row constrains N % 256 (the default
 * variant's BN) and this is where a variant whose BN does not divide N is refused. */
static inline int tile_ok(const R4dTileCfg* t, long long N) {
    if (!t) return R4D_NO(RAD_E_INVAL);
    if (t->bn > 0 && (N % t->bn)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}


#endif  /* R4D_GEMM_ARGS_H */
