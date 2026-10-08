/* sampler.h -- the sampler driver.
 *
 * SAMPLING RUNS ON DEVICE (spec §13). At a 151K vocab and batch 256 the logits are ~77 MiB a
 * step; moving them across the link to run a host chain costs both the bandwidth and a
 * synchronisation, on the step path, at exactly the concurrency the engine exists to serve. So
 * every sampler is a kernel, not a lift -- and this file does not implement one. It ISSUES the
 * declared sampler ops in order, exactly as an architecture plugin issues a GEMM: declared
 * through the ordinary builder, resolved through the ordinary hierarchy, serviced by
 * `libref` on the host domain or by a device plugin on the fast one.
 *
 * THE ORDER, and it is not a detail -- reordering changes which tokens survive:
 *
 *     penalties (repetition / frequency / presence, then DRY)
 *     temperature
 *     the grammar mask
 *     top-k
 *     top-p, min-p, typical
 *     XTC
 *     the pick
 *
 * TEMPERATURE MAY RUN AFTER THE CUTS INSTEAD, per sequence: llama.cpp's order, which a request
 * names with `samplers` (SamplingParams::temp_last). Such a row's temperature is staged at 1 for
 * the stage above, which skips it, and at its own value for a second `sample_temp` over the
 * candidate plane after XTC -- the same kernel at the candidate width, so no plugin needs a
 * second implementation of it. Every other row is 1 there.
 *
 * LOGPROBS ARE READ OFF THE RAW LOGITS, before the first stage: `sample_logprobs` summarises each
 * row that asks (RAD_SP_LOGPROBS) as its maximum, its normaliser and its kLpK largest (one more
 * than RAD_MAX_LOGPROBS, for the row's width below), and rank 0 copies that back behind the pick.
 * The emitted token's own logprob is its raw logit less the log normaliser. When it is not among the largest its raw logit is no longer on the card
 * -- every stage wrote the logits in place -- so it is recovered from the value the pick drew it
 * at, the candidate plane's, by undoing temperature, DRY and the penalties on the host
 * (logprobs_for). Such a row is never a greedy-flagged one: a greedy row asking for logprobs is
 * staged as a top-1 chain row, so that its token is drawn from a candidate plane too.
 *
 * A GREEDY SEQUENCE SKIPS ALL OF IT and takes `sample_argmax` -- after penalties and the mask,
 * because temperature 0 collapses every other logit to -inf and a mask applied afterwards would
 * leave nothing at all. host_ref.cpp makes the same choice for the same reason and the two must
 * agree, or a temperature-0 request stops matching a greedy run.
 *
 * PARAMETERS ARE DATA, NOT BRANCHES. Sequences in one batch have different sampling parameters,
 * so the parameters are a device-side per-sequence array built once per step from a pinned
 * staging buffer -- the same discipline the step batch uses (spec §8). A kernel that branched on
 * a host-side parameter would need one launch per distinct parameter set, which at batch 256 is
 * up to 256 launches for what is one kernel's worth of work.
 *
 * THE SCHEMA THIS DRIVER ISSUES AGAINST (docs/OPS.md, "Sampling"). No per-sequence value is an
 * op PARAMETER, because parameters are geometry -- the thing the selector matches constraints
 * against and freezes at declare. A per-sequence value cannot live there: two requests in one
 * batch with different top_p would be two different geometries and therefore two different
 * resolutions of the same op. The seed and output position `sample_pick` draws from are in the
 * parameter row too. So every sampler op takes the row array as an operand:
 *
 *   sample_penalties   M(range) n_vocab vocab_off?   logits, params, history  -> logits (inout)
 *   sample_dry         M(range) n_vocab vocab_off?   logits, params, history, breakers?
 *                                                                          -> logits (inout)
 *   sample_temp        M(range) n_vocab   logits, params                  -> logits (inout)
 *   sample_mask        M(range) n_vocab   logits, params, bitmask         -> logits (inout)
 *   sample_topk        M(range) n_vocab vocab_off?   logits, params       -> cand_idx, cand_val,
 *                                                                             pairs?
 *   sample_argmax      M(range) n_vocab   logits, params                  -> token
 *   sample_topp        M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_minp        M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_typical     M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_xtc         M(range) n_cand    cand_idx, cand_val, params      -> cand_idx, cand_val
 *   sample_pick        M(range) n_cand    cand_idx, cand_val, params      -> token
 *   sample_merge_topk  M(range) n_cand    gathered, params                -> cand_idx, cand_val
 *   sample_logprobs    M(range) n_vocab k vocab_off?   logits, params     -> stats
 *
 * `M` is the number of SAMPLED POSITIONS, not tokens: one per sequence at decode, n_spec+1 per
 * sequence at a verify. `params` is [M] rows of DeviceSampleParams below; `history` is
 * [M, max_hist] int32 of GLOBAL token ids, row m addressed by row m's hist_off/hist_len;
 * `breakers` is [M, max_hist] uint8 beside it, nonzero where that position's token is a whole DRY
 * sequence breaker; `bitmask` is [M, mask_words] uint32 addressed by the row's mask_row.
 * `vocab_off` is this rank's first vocabulary row and is declared only when the vocabulary is
 * split: the three ops that turn a token id into a column of this rank's logits subtract it.
 * Every one of these has a libref implementation: a sampler with no ref implementation has no
 * fallback and no oracle (spec §17).
 */
#pragma once
#include "rad_sample.h"

#include "../rad_core.h"
#include "grammar.h"
#include "host_ref.h"
#include "vocab_view.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ================================================================== the per-sequence row */

/* THE ROW AND ITS FLAGS LIVE IN abi/rad_sample.h, because they are the wire format of
 * an OPERAND and therefore part of the ABI: libref reads these rows and a kernel plugin
 * cannot include a core header. The alias keeps this file's spelling. */
using DeviceSampleParams = RadSampleParams;

/* ================================================================== configuration */

struct SamplerConfig {
    int64_t n_vocab       = 0;    /* the WHOLE vocabulary */
    int64_t n_vocab_local = 0;    /* THIS RANK's shard of it; == n_vocab when world_size == 1 */
    int64_t max_seqs      = 0;    /* the largest number of sequences one step may carry */
    int64_t max_hist      = 2048; /* the penalty window bound; a longer request is truncated */
    int     max_spec      = 0;
    int     rank          = 0;
    int     world_size    = 1;
    /* This rank's first vocabulary row, which sample_topk adds to every candidate id it packs
     * into the pair plane so the merge sees GLOBAL ids, and which sample_penalties and sample_dry
     * subtract from the GLOBAL ids in the history to find this rank's column. Zero unless lm_head
     * is row-sharded. */
    int64_t vocab_off     = 0;

    /* The candidate-set width the cuts work over. Every sampler after top-k operates on
     * candidates rather than the vocabulary, so this bounds the buffers they read and write. It
     * is a bound, not a policy: a request asking for a larger top-k than this is refused at
     * admission, by name, rather than silently narrowed. */
    int64_t max_candidates = 1024;
};

/* ================================================================== the driver */

class Sampler {
public:
    Sampler() = default;
    ~Sampler();

    /* ---------------------------------------------------------- declare phase */

    /* Declare every sampler op and every buffer the chain needs, and allocate the pinned staging.
     * Runs once, alongside the architecture plugin's declare, so the sampler's kernels resolve
     * through the same hierarchy at the same time as everything else and a missing one is in the
     * same missing-list (spec §3.1).
     *
     * A sampler op that does not resolve is NOT a declare failure: a deployment that never sends
     * `typical_p` does not need a typical kernel. It becomes a failure when a request asks for
     * it, which `check_request` reports by name. */
    int declare(RadBuilder* b, const SamplerConfig& cfg);

    /* The vocabulary view the grammar backends walk. Set after the tokeniser loads, before the
     * first request. */
    void attach_vocab(const VocabView* v) { vocab_ = v; }

    /* ---------------------------------------------------------- admission */

    /* Can this request's sampling parameters be served? Returns RAD_OK, or RAD_E_NOKERNEL naming
     * the sampler that did not resolve, or RAD_E_INVAL naming the parameter that is out of range.
     * Called on admission, so a request fails at the door rather than three steps in. */
    int check_request(const SamplingParams& sp, std::string* why) const;

    /* Build this request's grammar state, if it has a grammar. Runs on the scheduler thread, so
     * it must not compile: a request admitted through the server carries its grammar compiled
     * (SamplingParams::grammar_program), and this finds that program and positions a grammar in
     * it. A request that arrives without one is compiled here, which costs its mask plan. */
    int begin_request(uint64_t req_id, const SamplingParams& sp, std::string* err);
    void end_request(uint64_t req_id);
    /* Does this request already have state here? The step loop reaches a request on every step it
     * is scheduled in, and begin_request is not idempotent -- calling it twice would rebuild the
     * automaton and throw away everything it had accepted, which silently unconstrains a
     * generation from its second token onward. */
    bool has_request(uint64_t req_id) const;

    /* Requests build_step could not serve: a grammar that reached a state admitting no token.
     * Returned and cleared; the caller cancels them. Empty on almost every step. */
    std::vector<uint64_t> take_failed();

    /* Advance a request's grammar over the token it just emitted. */
    int accept_token(uint64_t req_id, int32_t token);

    /* May the scheduler speculate on this request? Yes, unless it carries a constraining grammar
     * whose backend cannot snapshot its state. The mask for draft position i depends on which of
     * positions 0..i-1 were accepted, so masking a window means walking the grammar forward over
     * the proposals and rewinding afterwards; a backend without push_state/rollback cannot, and
     * such a request decodes one token at a time rather than silently emitting drafts the grammar
     * forbids. */
    bool allows_speculation(uint64_t req_id) const;
    /* Is allows_speculation's answer fixed for this request? It changes only when a grammar whose
     * backend cannot rewind starts to constrain, which is a commit's doing; a request with no
     * grammar, or one whose backend rewinds, answers the same at every step. */
    bool speculation_fixed(uint64_t req_id) const;

    /* ---------------------------------------------------------- step path */

    /* Build the per-sequence parameter array, the packed penalty history and the grammar
     * bitmasks for this step, into pinned staging. Scheduler-thread work: it touches request
     * state and host grammar automata, and it must be done before the rank threads are released
     * at the barrier.
     *
     * Allocates nothing -- every buffer was sized at declare against max_seqs (spec §3.1). */
    int build_step(const std::vector<Request*>& reqs);

    /* THE SAME STAGING, FILLED FOR A GREEDY DRAFTER. `n` rows of argmax: no penalties, no
     * grammar mask, no history, and the RNG untouched -- so the chain resolves to op_argmax and
     * nothing else runs.
     *
     * It exists because a DRAFT round is not a request's own sampling. Its output is a
     * proposal the target then verifies, and the verifier is greedy (core/sample/accept.h): a
     * drafter that sampled at the request's temperature would spend the request's RNG stream
     * on tokens that may never be emitted, and would lower acceptance for nothing. Every MTP
     * head that ships inside a target checkpoint is greedy for the same reason. */
    int build_greedy(int64_t n);

    /* Upload the staging and issue the chain. Rank-thread work, on the compute stream, with no
     * host synchronisation in it.
     *
     * `logits` holds [n_seq, n_vocab_local] and `out_tokens` receives [n_seq] token ids. Under
     * tensor parallel the candidate merge (`sample_merge_topk`, or `row_topk_merge` fused with the
     * top-k) happens between the top-k and the cuts; full logits are never gathered. */
    int run(RadCtx* c, rad_buf logits, rad_buf out_tokens);

    /* THE EMITTED TOKEN'S LOGPROBS, after the step the last build_step staged has completed on the
     * card: row `row` must have been staged for a request with SamplingParams::n_probs >= 0, and
     * `token` is what the step sampled there. Rank 0's sampler only -- it is the one that copies
     * the summary back. RAD_E_STATE for a row that was not staged for logprobs; RAD_E_INVAL, with
     * `out->token` -1, for one whose numbers are not finite -- a row with no finite logit, or a
     * token drawn where none was -- which is a number no client should be sent. */
    int logprobs_for(int64_t row, int32_t token, TokenLogprobs* out) const;
    /* Did `sample_logprobs` resolve? What the server advertises as logprobs support. */
    bool has_logprobs() const { return op_logprobs_ != RAD_NULL_HANDLE; }

    /* The parameter rows the last build_step or build_greedy staged, and how many. What `run`
     * uploads, readable so the staging can be checked without a device behind it. */
    const DeviceSampleParams* staged_params(int64_t* n) const {
        if (n) *n = n_seq_;
        return cur_stage_;
    }

private:
    /* IS THE VOCABULARY ACTUALLY SPLIT? Not the same question as "is there more than one rank".
     * A tensor-parallel deployment whose plugin declares `output.weight` replicated gives every
     * rank the whole vocabulary, and then the candidate merge is not just unnecessary but wrong:
     * every rank would contribute the SAME k candidates and the gather would count them once per
     * rank. The declared width of this rank's logits is the fact, so that is what this asks. */
    bool vocab_parallel() const {
        return cfg_.world_size > 1 && cfg_.n_vocab_local < cfg_.n_vocab;
    }

    /* `dry` holds the request's resolved sequence breakers, empty when it sent none -- which is
     * not "DRY off": DRY runs whenever its multiplier is set, with or without breakers. */
    struct SeqState {
        std::unique_ptr<GrammarBackend> grammar;
        DryBreakers                     dry;
        /* Where the reply stands against its reasoning block, over `output[0, think_seen)`.
         * Advanced by build_step over what each commit appended, so a step costs the tokens it
         * added and not the reply's length. Read only when the request carries a budget. */
        ReasoningCount                  think;
        size_t                          think_seen = 0;
    };

    /* Mask staged row `row` to the reasoning block's close token: the budget is spent and the
     * sampled position is still inside the block. A grammar already masking the row keeps the
     * last word -- the close is forced only where the grammar admits it, and a row whose grammar
     * forbids it is left to the grammar and forced at a later position. */
    void force_close(DeviceSampleParams& p, int64_t row, int32_t close);

    SamplerConfig    cfg_{};
    const VocabView* vocab_ = nullptr;
    bool             declared_ = false;

    /* Declared op handles. RAD_NULL_HANDLE means the op did not resolve; the admission check
     * turns that into a named refusal rather than a silent skip. */
    rad_op op_penalties_ = RAD_NULL_HANDLE;
    rad_op op_dry_       = RAD_NULL_HANDLE;
    rad_op op_temp_      = RAD_NULL_HANDLE;
    rad_op op_temp_late_ = RAD_NULL_HANDLE;   /* sample_temp over the candidates (temp_last) */
    rad_op op_mask_      = RAD_NULL_HANDLE;
    rad_op op_topk_      = RAD_NULL_HANDLE;
    rad_op op_topp_      = RAD_NULL_HANDLE;
    rad_op op_minp_      = RAD_NULL_HANDLE;
    rad_op op_typical_   = RAD_NULL_HANDLE;
    rad_op op_xtc_       = RAD_NULL_HANDLE;
    rad_op op_pick_      = RAD_NULL_HANDLE;
    rad_op op_argmax_    = RAD_NULL_HANDLE;
    rad_op op_allgather_ = RAD_NULL_HANDLE;
    rad_op op_merge_     = RAD_NULL_HANDLE;   /* the candidate merge, under tensor parallel */
    rad_op op_logprobs_  = RAD_NULL_HANDLE;   /* the raw distribution's summary */
    rad_op op_lp_gather_ = RAD_NULL_HANDLE;   /* every rank's summary, under a split vocabulary */

    /* Device-side buffers, from the ordinary buffer plan. */
    rad_buf buf_params_  = RAD_NULL_HANDLE;
    rad_buf buf_params_late_ = RAD_NULL_HANDLE;   /* the late temperature's rows */
    rad_buf buf_hist_    = RAD_NULL_HANDLE;
    rad_buf buf_brk_     = RAD_NULL_HANDLE;   /* DRY's single-token breaker marks, beside history */
    rad_buf buf_mask_    = RAD_NULL_HANDLE;
    rad_buf buf_cand_i_  = RAD_NULL_HANDLE;
    rad_buf buf_cand_v_  = RAD_NULL_HANDLE;
    rad_buf buf_gather_  = RAD_NULL_HANDLE;
    rad_buf buf_pairs_   = RAD_NULL_HANDLE;   /* this rank's candidates as (global id, logit) */
    rad_buf buf_lp_      = RAD_NULL_HANDLE;   /* sample_logprobs' stats, [rows, lp_width()] */
    rad_buf buf_lp_g_    = RAD_NULL_HANDLE;   /* and every rank's side by side */

    /* Pinned staging. One allocation each, at declare, never on the step path. */
    DeviceSampleParams* params_stage_ = nullptr;
    /* The late temperature's rows: params_stage_'s, with each row's temperature at 1 unless it
     * runs last. Uploaded only by a step where some row does (any_late_). */
    DeviceSampleParams* late_stage_   = nullptr;
    bool                any_late_     = false;
    /* THE GREEDY ROWS HAVE STAGING OF THEIR OWN, written once. A draft pass is staged while the
     * step before it -- whose sampler uploads `params_stage_` when the card reaches it -- may not
     * have run yet; rows written into the same staging would reach that step's sampler instead.
     * Every greedy row is the same row, so this one is filled on first use and never again, and
     * no upload of it can ever see a later build. */
    DeviceSampleParams* greedy_stage_ = nullptr;
    bool                greedy_ready_ = false;
    const DeviceSampleParams* cur_stage_ = nullptr;   /* what the last build staged */
    int32_t*            hist_stage_   = nullptr;
    uint8_t*            brk_stage_    = nullptr;
    uint32_t*           mask_stage_   = nullptr;

    int64_t mask_words_ = 0;      /* words for THIS RANK's shard; what the mask op indexes */
    int64_t n_seq_      = 0;
    /* Does any DRY row this step carry breakers? Only then is the breaker plane uploaded and
     * handed to sample_dry; a batch whose DRY requests sent none issues it without the operand. */
    bool    any_brk_    = false;

    /* The whole-vocabulary mask the grammar fills, from which the staged shard is cut. Only used
     * when the head is sharded; see the note in declare(). */
    std::vector<uint32_t> mask_full_;

    /* The grammar mask's cost, accumulated for the Debug line in build_step. Not state the step
       path reads -- a diagnostic, and one worth carrying: the mask walk is scheduler-thread work
       large enough to decide the step time of a constrained request, so it has to be visible
       rather than assumed. */
    double  mask_us_       = 0.0;
    double  mask_worst_us_ = 0.0;
    int64_t mask_us_n_     = 0;
    int64_t mask_fast_     = 0;   /* memo hits */
    int64_t mask_slow_     = 0;   /* walks    */
    double  mask_walk_us_  = 0.0; /* and what only the walks cost */
    double  build_us_      = 0.0;
    int64_t build_n_       = 0;
    double  accept_us_     = 0.0;
    int64_t accept_n_      = 0;
    void shard_mask(uint32_t* dst) const;

    /* Requests whose grammar died this step. Drained by the engine, which cancels them -- one bad
     * request must not stop the step loop for everyone else. */
    std::vector<uint64_t> failed_;

    /* ---------------------------------------------------------- logprobs */
    /* THE SUMMARY'S WIDTH: RAD_MAX_LOGPROBS + 1 pairs, not RAD_MAX_LOGPROBS, so that a row is 44
     * floats -- whole 16-byte words, which is what the all-gather of a split vocabulary moves (and
     * libr4d's refuses anything else). The extra pair is never reported. */
    static constexpr int kLpK = RAD_MAX_LOGPROBS + 1;
    static int64_t lp_width() { return 2 + 2 * (int64_t)kLpK; }
    /* WHAT UNDOING A ROW'S CHAIN NEEDS, kept from build_step for each row staged with
     * RAD_SP_LOGPROBS: the staged row itself, the temperature the candidate value carries, the
     * history window both history stages read, and the request's DRY breakers. Copied rather
     * than pointed at, because the staging is rebuilt by the next step. */
    struct LpRow {
        int64_t            row = -1;
        int32_t            n_top = 0;
        float              temp = 1.0f;          /* the candidate value is l / temp */
        DeviceSampleParams p{};
        std::vector<int32_t> hist;               /* the staged history row, [0, hist_len) */
        const DryBreakers* dry = nullptr;        /* lives in seqs_ for the request's lifetime */
    };
    std::vector<LpRow> lp_rows_;
    /* Rank 0's copies, written behind the pick: the summaries ([rows, W * lp_width()], W ranks
     * side by side) and the candidate planes the pick drew from. */
    float*   lp_host_    = nullptr;
    int32_t* lp_ci_host_ = nullptr;
    float*   lp_cv_host_ = nullptr;
    int64_t  lp_lo_ = 0, lp_hi_ = 0;             /* the rows copied, [lo, hi) */
    /* A token's raw logit from the value the pick drew it at: temperature, DRY and the penalties
     * undone, in the reverse of the order the chain applied them. */
    float raw_logit(const LpRow& L, int32_t token, float v) const;

    /* Per-request grammar and DRY state, keyed by request id. Held here rather than in Request
     * because Request is a frozen struct shared across components and a unique_ptr in it would
     * make the scheduler's copies a compile error. */
    std::unordered_map<uint64_t, SeqState> seqs_;

    int stage_alloc();
    void stage_free();
};

/* SPLIT-VOCABULARY SUMMARIES, MERGED: `W` ranks' `sample_logprobs` rows of `k` pairs each, side by
 * side in `st` (W * (2 + 2k) floats). Sets the shared log normaliser and fills up to `cap` of the
 * largest (id, logit) pairs over every rank, descending, ties to the lower id; returns how many. A
 * rank whose maximum or normaliser is not finite contributes nothing, nor does a pair whose logit
 * is not finite, and `*log_z` comes back NaN when no rank contributes. One rank is the same
 * arithmetic. What Sampler::logprobs_for reads a step's summaries with. */
int merge_logprob_rows(const float* st, int W, int k, double* log_z, int32_t* ids, float* vals,
                       int cap);

}  /* namespace rad */
