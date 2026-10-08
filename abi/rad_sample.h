/* rad_sample.h -- the per-sequence sampling row, and why it is part of the ABI.
 *
 * Every sampler op takes `params` as an OPERAND: an array of the struct below, one row per sampled
 * position, uploaded once per step. That makes its layout a wire format between the core and every
 * kernel plugin, so it belongs here beside RadTensor rather than in core/sample/ where it started
 * -- a kernel plugin cannot include a core header, and a struct the host writes and the device
 * reads is a struct whose layout has to be STATED, not inferred.
 *
 * ============================== WHY NOT PARAMETERS ==============================
 *
 * `rep`, `freq`, `pres`, `k` and `p` read like op PARAMETERS, and docs/OPS.md names them that way.
 * They cannot be. Parameters are geometry: the selector matches constraints against them and they
 * are frozen at declare (spec §2.2). A per-sequence value cannot live there, and the failure is
 * not subtle -- two requests in one batch with different top_p would be two different geometries
 * and therefore two different resolutions of the same op, which is not something a batch can
 * contain. A deployment would have one top_p.
 *
 * So the values are DATA and they travel as an operand. `seed` and `offset`, which docs/OPS.md
 * gives `sample_pick` as two more operands, are fields of the row for the same reason.
 *
 * core/sample/sampler.h carries the full schema table this implies; libref implements it.
 */
#ifndef RAD_SAMPLE_H
#define RAD_SAMPLE_H

#include "rad_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flags, so a kernel can skip a stage for a sequence without reading five floats to discover they
 * are the identity. One integer test beats five float comparisons per row per stage.
 *
 * RAD_SP_GREEDY OWNS THE ROW'S TOKEN. `sample_argmax` writes it, and every candidate stage after
 * it -- `sample_topk`, the narrowing stages, `sample_merge_topk` and `sample_pick` -- treats the
 * row as having no candidates and leaves `token` alone. A batch mixing greedy and sampled rows
 * issues both paths over every row, so a stage that ignored the flag would overwrite the argmax
 * with a draw. A greedy row that has to go through the candidate chain instead -- a split
 * vocabulary, where the argmax's column is not a token id -- is staged at top_k = 1 WITHOUT this
 * flag. */
enum {
    RAD_SP_GREEDY  = 1 << 0,   /* argmax path; the candidate chain is not run for this sequence */
    RAD_SP_PENALTY = 1 << 1,   /* any of rep / freq / pres is non-identity */
    RAD_SP_MASK    = 1 << 2,   /* a grammar is constraining this sequence this step */
    RAD_SP_XTC     = 1 << 3,
    RAD_SP_DRY     = 1 << 4,
    RAD_SP_LOGPROBS = 1 << 5   /* `sample_logprobs` summarises this row's raw logits */
};

/* One row per sampled position. Laid out so the 8-byte fields are aligned and the struct has no
 * padding anywhere, including at the tail.
 *
 * THE HISTORY WINDOW. `hist_off` and `hist_len` address the row's OWN row of the history plane,
 * [M, max_hist] int32 with row m belonging to sampled position m: the window is
 * history[m][hist_off .. hist_off + hist_len), clamped to the row. Token ids in it are GLOBAL
 * vocabulary ids; a stage over a vocabulary shard subtracts its `vocab_off` parameter.
 *
 * `penalty_last_n` and `dry_penalty_last_n` window the TAIL of that span: 0 turns the stage off
 * for the row, a negative value takes the whole span, and n > 0 takes the last n tokens -- the
 * meaning llama.cpp gives `repeat_last_n` and `dry_penalty_last_n`.
 *
 * `dry_rep_limit` is the longest repeat DRY may count for the row, fixed by the most recent
 * sequence breaker in the DRY window (llama.cpp's `rep_limit`), or 0 when no breaker occurs there
 * and repeats are not limited. The breakers themselves are resolved against the vocabulary on the
 * host, which is also where that limit is found; see `sample_dry`'s optional `breakers` operand
 * for the other half. */
typedef struct RadSampleParams {
    uint64_t seed;              /* the request's seed */
    uint64_t pos;               /* output position: the RNG's second coordinate */

    float    temp;
    float    top_p;
    float    min_p;
    float    typical_p;

    float    rep_penalty;
    float    freq_penalty;
    float    pres_penalty;
    float    xtc_probability;

    float    xtc_threshold;
    float    dry_multiplier;
    float    dry_base;

    int32_t  top_k;
    int32_t  penalty_last_n;
    int32_t  dry_allowed_length;
    int32_t  dry_penalty_last_n;

    int32_t  hist_off;          /* start of the window within this row's own history row */
    int32_t  hist_len;
    int32_t  mask_row;          /* row into the bitmask plane, or -1 when unconstrained */
    int32_t  flags;             /* RAD_SP_* */

    int32_t  dry_rep_limit;     /* the breaker-imposed repeat limit, 0 when there is none */
} RadSampleParams;

/* The row is uploaded as an array and addressed as 24 words by every plugin, so its size is fixed
 * and every byte of it is a named field. */
#ifdef __cplusplus
static_assert(sizeof(RadSampleParams) == 96, "RadSampleParams is 96 bytes on the wire");
static_assert(offsetof(RadSampleParams, dry_rep_limit) + sizeof(int32_t) == sizeof(RadSampleParams),
              "RadSampleParams must pack without a tail hole");
#endif

/* THE RANDOM DRAW IS SPECIFIED, not left to a library, because the same stream has to come out of
 * a host reference, a device kernel and a rerun (spec §17). splitmix64 over
 * (seed * golden ratio + pos), taking 24 bits as a float in [0, 1) -- exactly f32's mantissa, so
 * the draw is representable without a double anywhere in the chain.
 *
 * INDEPENDENT DRAWS AT ONE POSITION TAKE INDEPENDENT STREAMS, and a stream is the request's seed
 * XORed with a key before the draw: `rad_sample_uniform01(seed ^ key, pos)`. The pick draws on the
 * seed itself. XTC's "does it fire" coin has its own key, so a row whose XTC fired is not a row
 * whose pick is correlated with that fact; speculative acceptance and its residual draw have
 * theirs. Every implementation of a stage takes its key from here. */
#define RAD_SAMPLE_KEY_PICK   0x0ull
#define RAD_SAMPLE_KEY_XTC    0x58544311ull
#define RAD_SAMPLE_KEY_ACCEPT 0x41435054ull
#define RAD_SAMPLE_KEY_RESID  0x52534944ull

static inline uint64_t rad_sample_splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline float rad_sample_uniform01(uint64_t seed, uint64_t pos) {
    const uint64_t z = rad_sample_splitmix64(seed * 0x9E3779B97F4A7C15ull + pos);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_SAMPLE_H */
