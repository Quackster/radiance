/* sampler.cpp -- the sampler driver: declare the sampler ops, build the per-sequence parameter
 * array once per step, issue the chain. No sampler is implemented here; see sampler.h for why.
 */
#include "sampler.h"
#include "gbnf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rad {

/* ================================================================== grammar triggers
 *
 * A chat template names its triggers the way a person would -- "the call starts at the word
 * <tool_call>" -- while GbnfGrammar wants them in the two forms it checks as tokens arrive: token
 * ids to compare against the id just emitted, and literal words to find in the emitted bytes.
 * The translation needs a vocabulary, so it happens here and not in the server.
 *
 * A WORD IS A TOKEN WHEN THE VOCABULARY SAYS SO. The word is tokenised with special tokens
 * recognised, since a template's markers are usually special tokens and a split of `<tool_call>`
 * into `<`, `tool`, `_call`, `>` is not what the model emits. One token makes an id trigger, which
 * costs a comparison; anything longer is scanned for as bytes, because the tokeniser may split it
 * differently depending on what came before it. Getting this wrong does not fail loudly -- it
 * yields a trigger that never fires, so the call comes out unconstrained, which reads as a model
 * that writes malformed calls.
 *
 * A REGULAR-EXPRESSION TRIGGER IS REFUSED. Its matcher would run over the reply once per token
 * on the scheduler thread; the templates in service all trigger on words. A request whose
 * template needs one is refused at admission, naming the pattern. */
static int resolve_triggers(const VocabView& vocab,
                            const std::vector<SamplingParams::GrammarTrigger>& triggers,
                            std::vector<std::string>* words, std::vector<int32_t>* tokens,
                            std::string* err) {
    words->clear();
    tokens->clear();
    for (const auto& t : triggers) {
        if (t.value.empty()) continue;
        std::vector<int32_t> ids;
        switch (t.type) {
            case 0:      /* token: the spelling of exactly one token */
            case 1: {    /* word */
                const int st = vocab.tokenize_special(t.value, &ids);
                if (st < 0) {
                    if (err) *err = "grammar trigger '" + t.value + "' could not be tokenised";
                    return st;
                }
                if (ids.size() == 1) { tokens->push_back(ids[0]); break; }
                if (t.type == 0) {
                    if (err) *err = "grammar trigger token '" + t.value +
                                    "' is not a single token in this vocabulary";
                    return RAD_E_INVAL;
                }
                words->push_back(t.value);
                break;
            }
            case 2:      /* pattern */
            case 3:      /* pattern_full */
                if (err) *err = "this chat template triggers its tool-call grammar on the "
                                "regular expression '" + t.value + "'; only word and token "
                                "triggers are supported";
                return RAD_E_UNSUPPORTED;
            default:
                if (err) *err = "unknown grammar trigger type " + std::to_string(t.type);
                return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* ================================================================== small helpers */

/* Copy this rank's window out of a whole-vocabulary bitmask. `vocab_off` is word-aligned for
 * every shard this engine builds (the head is split on a multiple of 32), so the aligned arm is
 * the one that runs; the bit-shifting arm exists because a shard boundary is a placement decision
 * and not a promise, and a mask that is silently off by a few bits admits the wrong tokens. */
void Sampler::shard_mask(uint32_t* dst) const {
    const int64_t full = (int64_t)mask_full_.size();
    std::memset(dst, 0, (size_t)mask_words_ * 4);
    if ((cfg_.vocab_off & 31) == 0) {
        const int64_t w0 = cfg_.vocab_off >> 5;
        const int64_t n  = std::min(mask_words_, full - w0);
        if (n > 0) std::memcpy(dst, mask_full_.data() + w0, (size_t)n * 4);
        return;
    }
    for (int64_t i = 0; i < cfg_.n_vocab_local; ++i) {
        const int64_t g = cfg_.vocab_off + i;
        if (g >= cfg_.n_vocab) break;
        if (mask_full_[(size_t)(g >> 5)] & (1u << (uint32_t)(g & 31)))
            dst[i >> 5] |= 1u << (uint32_t)(i & 31);
    }
}

/* rad_runtime.h's RAD_B / RAD_W macros are C compound literals. They are legal C++ only as a
 * compiler extension, and this file is core code that has to build with -Wall -Wextra clean, so
 * the operands are built the ordinary way. */
/* NARROWED TO THIS STEP'S ROWS, always.  arch/common/rad_arch.h says why for the model's own
 * buffers and the reason is the sampler's too: a kernel cannot read the row count out of `M`,
 * because `M` is the band's upper bound, so every shim takes it off the operand.  Handed over at
 * its DECLARED extent -- max_seqs * (max_spec + 1) rows -- an operand makes every stage compute a
 * full candidate set for rows that have no request behind them, over params the upload never
 * wrote.  The extra rows are discarded, so nothing downstream is wrong and no bench can see it,
 * but each phantom row costs a vocabulary-wide reduction, and at one sequence every row but one
 * is a phantom.
 *
 * `rows` is dim 0 for every buffer here -- logits, params, history, mask, candidates, pairs, the
 * gather and the token output all count sampled POSITIONS first -- and the strides below it are
 * products of the trailing dimensions, so shortening the outermost one leaves them correct. */
static RadOperand opd_rows(rad_buf h, int64_t rows) {
    RadOperand o{};
    o.kind = RAD_OPK_BUF;
    o.handle = h;
    o.rows = rows;
    return o;
}

/* An optional operand this issue does not pass. The runtime holds every issue to its schema's
 * operand COUNT -- operands are positional -- so an absent trailing optional is spelled, not
 * dropped. */
static RadOperand opd_none() {
    RadOperand o{};
    o.kind = RAD_OPK_NONE;
    return o;
}

static int64_t rows_for(const SamplerConfig& c) {
    /* One row per SAMPLED POSITION, not per sequence: a speculative verify samples n_spec+1
     * positions for a sequence and each one has its own output position, hence its own RNG
     * coordinate. Sizing for the worst step is the same rule the arena follows (spec §3.1). */
    return c.max_seqs * (int64_t)(c.max_spec + 1);
}

/* ================================================================== declare */

int Sampler::declare(RadBuilder* b, const SamplerConfig& cfg) {
    if (!b) return RAD_E_INVAL;
    if (cfg.n_vocab <= 0 || cfg.max_seqs <= 0) {
        RAD_ERR("sampler: n_vocab=%lld max_seqs=%lld is not a configuration",
                (long long)cfg.n_vocab, (long long)cfg.max_seqs);
        return RAD_E_INVAL;
    }

    cfg_ = cfg;
    if (cfg_.n_vocab_local <= 0) cfg_.n_vocab_local = cfg_.n_vocab / cfg_.world_size;

    /* THE STAGED MASK IS THIS RANK'S SHARD, AND THE GRAMMAR'S IS THE WHOLE VOCABULARY. Those are
     * different lengths the moment the head is sharded, and conflating them is silent: every
     * vocab-scoped sampler op is declared over `n_vocab_local` (op_vocab, below) and `sample_mask`
     * indexes its bitmask from bit zero, so a full-vocabulary mask handed to rank 1 masks its
     * logits with the admissibility of tokens 0..n_local -- ids that belong to rank 0. The result
     * is not a diagnosable error, it is a rank sampling from the wrong permission set.
     *
     * So the grammar fills `mask_full_` over the whole vocabulary, which is also where the "no
     * token is admissible anywhere" check belongs, and only this rank's window of it is staged. */
    mask_words_ = grammar_mask_words(cfg_.n_vocab_local);
    mask_full_.assign((size_t)grammar_mask_words(cfg_.n_vocab), 0u);

    const int64_t rows = rows_for(cfg_);
    const int64_t nv   = cfg_.n_vocab_local;
    const int64_t ncand = cfg_.max_candidates;

    /* ---------------------------------------------------------- buffers */
    /* PERSIST, not TRANSIENT: the parameter, history and mask planes are written by an upload
     * that happens BEFORE the run phase issues anything, so liveness over the declared op list
     * would conclude they are dead where they are in fact being filled. A transient whose first
     * definition is invisible to the planner is a transient the planner will happily overwrite. */
    auto decl = [&](const char* name, uint32_t dt, int64_t d0, int64_t d1, int kind) -> rad_buf {
        RadBufDecl d{};
        d.dtype = dt;
        d.rank  = 2;
        d.shape[0] = d0;
        d.shape[1] = d1;
        d.kind = kind;
        d.domain = RAD_DOMAIN_DEVICE;
        return rad_decl_buffer(b, name, &d);
    };

    buf_params_ = decl("sampler.params", RAD_U32, rows,
                       (int64_t)(sizeof(DeviceSampleParams) / 4), RAD_BUF_PERSIST);
    buf_params_late_ = decl("sampler.params_late", RAD_U32, rows,
                            (int64_t)(sizeof(DeviceSampleParams) / 4), RAD_BUF_PERSIST);
    buf_hist_   = decl("sampler.history", RAD_I32, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    /* One byte beside every history token: nonzero where that token is a whole DRY sequence
     * breaker, which DRY never penalises. Resolving a token against the breaker strings needs the
     * vocabulary, so it happens here on the host and the kernel reads the answer. */
    buf_brk_    = decl("sampler.dry_breakers", RAD_U8, rows, cfg_.max_hist, RAD_BUF_PERSIST);
    buf_mask_   = decl("sampler.mask", RAD_U32, rows, mask_words_, RAD_BUF_PERSIST);
    buf_cand_i_ = decl("sampler.cand_idx", RAD_I32, rows, ncand, RAD_BUF_TRANSIENT);
    buf_cand_v_ = decl("sampler.cand_val", RAD_F32, rows, ncand, RAD_BUF_TRANSIENT);
    if (vocab_parallel()) {
        /* THE PAIR PLANE AND THE GATHERED PLANE. sample_topk packs this rank's candidates into
         * `pairs` as (u32 global id, f32 logit); the all-gather places every rank's pairs for a
         * row SIDE BY SIDE, which is the one layout the merge can read at a single row stride. */
        buf_pairs_  = decl("sampler.cand_pairs", RAD_U32, rows, ncand * 2, RAD_BUF_TRANSIENT);
        buf_gather_ = decl("sampler.cand_gathered", RAD_U32, rows,
                           ncand * cfg_.world_size * 2, RAD_BUF_TRANSIENT);
    }

    /* THE LOGPROBS SUMMARY, PERSIST for the reason the staging planes are: nothing in the
     * declared op list reads it -- rank 0 copies it to the host behind the pick -- so liveness
     * over the op list would hand its bytes to the candidate planes the stages after it write. */
    buf_lp_ = decl("sampler.logprobs", RAD_F32, rows, lp_width(), RAD_BUF_PERSIST);
    if (vocab_parallel())
        buf_lp_g_ = decl("sampler.logprobs_gathered", RAD_F32, rows,
                         lp_width() * cfg_.world_size, RAD_BUF_PERSIST);

    if (!buf_params_ || !buf_params_late_ || !buf_hist_ || !buf_brk_ || !buf_mask_ ||
        !buf_cand_i_ || !buf_cand_v_ || !buf_lp_) {
        RAD_ERR("sampler: a staging buffer failed to declare");
        return RAD_E_NOMEM;
    }

    /* ---------------------------------------------------------- ops */
    /* Every one of these is declared unconditionally and may fail to resolve. That is not an
     * error here: a deployment that never sends `typical_p` has no use for a typical kernel, and
     * failing declare over it would make an absent sampler an absent model. check_request turns
     * an unresolved sampler into a named refusal at admission instead. */
    /* `M` is the ranged parameter and it is the number of SAMPLED POSITIONS, not tokens: one row
     * at decode, n_spec+1 per sequence at a verify, and a kernel written for a single row is a
     * different kernel from one written for a batch of 256 (spec §2.2). */
    auto op_vocab = [&](const char* name) -> rad_op {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 2, nullptr, 0);
    };
    /* THE VOCABULARY OFFSET, on the three stages that cross between a token id and a column of
     * this rank's logits row: top-k turns a column into a token id, and the penalties and DRY turn
     * the history's token ids into columns. Without it a rank past the first penalises the
     * columns of whatever ids happen to be small, which are other tokens entirely. Declared only
     * when the vocabulary is sharded, so a single-rank graph carries no parameter it has no use
     * for. */
    auto op_vocab_off = [&](const char* name) -> rad_op {
        if (!vocab_parallel())
            return op_vocab(name);
        RadParam p[3] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
            RadParam{ "vocab_off", RAD_P_INT, (long long)cfg_.vocab_off, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 3, nullptr, 0);
    };
    /* The stages after top-k read a candidate set, not the vocabulary, so their geometry is the
     * candidate width. That is what makes them cheap: 1024 entries rather than 151K. */
    auto op_cand = [&](const char* name) -> rad_op {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_cand", RAD_P_INT, (long long)ncand, 0, nullptr },
        };
        return rad_decl_op(b, name, p, 2, nullptr, 0);
    };

    op_penalties_ = op_vocab_off("sample_penalties");
    op_dry_       = op_vocab_off("sample_dry");
    op_temp_      = op_vocab("sample_temp");
    op_mask_      = op_vocab("sample_mask");
    op_topk_      = op_vocab_off("sample_topk");
    op_argmax_    = op_vocab("sample_argmax");

    op_topp_      = op_cand("sample_topp");
    op_minp_      = op_cand("sample_minp");
    op_typical_   = op_cand("sample_typical");
    op_xtc_       = op_cand("sample_xtc");
    op_pick_      = op_cand("sample_pick");
    /* THE SAME TEMPERATURE KERNEL OVER THE CANDIDATE PLANE: its `n_vocab` is the candidate width,
     * which is all the kernel reads it as -- a row length. */
    {
        RadParam p[2] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)ncand, 0, nullptr },
        };
        op_temp_late_ = rad_decl_op(b, "sample_temp", p, 2, nullptr, 0);
    }

    if (vocab_parallel()) {
        /* lm_head is vocab-sharded, so each rank reduces ITS OWN shard and only the candidate
         * sets are combined -- k * world_size pairs, 2 KiB at k=64 and world_size=4, against the
         * ~77 MiB a full-logits gather would move (spec §9). */
        RadParam ag[4] = {
            RadParam{ "world_size", RAD_P_INT, (long long)cfg_.world_size, 0, nullptr },
            RadParam{ "numel", RAD_P_INT, (long long)(rows * ncand * 2), 0, nullptr },
            RadParam{ "dtype", RAD_P_STR, 0, 0, "f32" },
            /* ROW-INTERLEAVED. One sampled position's pairs from every rank have to land side by
             * side, because sample_merge_topk reads one row at one row stride. */
            RadParam{ "row", RAD_P_INT, (long long)(ncand * 2), 0, nullptr },
        };
        op_allgather_ = rad_decl_op(b, "all_gather", ag, 4, nullptr, 0);
        op_merge_     = op_cand("sample_merge_topk");
    }

    /* THE RAW DISTRIBUTION'S SUMMARY, at the one width every request's top_logprobs fits in. A
     * deployment whose kernels lack it serves every request that does not ask for logprobs;
     * check_request refuses the ones that do, by name. */
    {
        RadParam p[4] = {
            RadParam{ "M", RAD_P_RANGE, 1, rows, nullptr },
            RadParam{ "n_vocab", RAD_P_INT, (long long)nv, 0, nullptr },
            RadParam{ "k", RAD_P_INT, (long long)kLpK, 0, nullptr },
            RadParam{ "vocab_off", RAD_P_INT, (long long)cfg_.vocab_off, 0, nullptr },
        };
        op_logprobs_ = rad_decl_op(b, "sample_logprobs", p, vocab_parallel() ? 4 : 3, nullptr, 0);
    }
    if (vocab_parallel()) {
        /* Each rank summarised its own shard; rank 0 merges them on the host, so every rank's
         * row lands beside the others' as the candidate pairs do. */
        RadParam ag[4] = {
            RadParam{ "world_size", RAD_P_INT, (long long)cfg_.world_size, 0, nullptr },
            RadParam{ "numel", RAD_P_INT, (long long)(rows * lp_width()), 0, nullptr },
            RadParam{ "dtype", RAD_P_STR, 0, 0, "f32" },
            RadParam{ "row", RAD_P_INT, (long long)lp_width(), 0, nullptr },
        };
        op_lp_gather_ = rad_decl_op(b, "all_gather", ag, 4, nullptr, 0);
        if (!op_lp_gather_) op_logprobs_ = RAD_NULL_HANDLE;
    }

    RAD_TRY(stage_alloc());
    declared_ = true;

    RAD_INFO("sampler: %lld-token vocabulary (%lld this rank), %lld rows, %lld mask words "
             "(%s per row)", (long long)cfg_.n_vocab, (long long)nv, (long long)rows,
             (long long)mask_words_, humanb(mask_words_ * 4).c_str());
    return RAD_OK;
}

int Sampler::stage_alloc() {
    const int64_t rows = rows_for(cfg_);

    /* Pinned, because these are uploaded every step and a pageable source turns an async copy
     * into a staged one with a host bounce -- which is exactly the per-step cost the step batch
     * exists to remove (spec §8). */
    params_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_stage_ = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    late_stage_   = (DeviceSampleParams*)rad_dev_alloc(
        rows * (int64_t)sizeof(DeviceSampleParams), RAD_MEM_HOST_PINNED);
    greedy_ready_ = false;
    hist_stage_   = (int32_t*)rad_dev_alloc(rows * cfg_.max_hist * 4, RAD_MEM_HOST_PINNED);
    brk_stage_    = (uint8_t*)rad_dev_alloc(rows * cfg_.max_hist, RAD_MEM_HOST_PINNED);
    mask_stage_   = (uint32_t*)rad_dev_alloc(rows * mask_words_ * 4, RAD_MEM_HOST_PINNED);

    /* Rank 0's landing place for the logprobs copies, and only rank 0's: it is the rank that
     * reads them. */
    if (cfg_.rank == 0) {
        const int64_t w = lp_width() * (vocab_parallel() ? cfg_.world_size : 1);
        lp_host_    = (float*)rad_dev_alloc(rows * w * 4, RAD_MEM_HOST_PINNED);
        lp_ci_host_ = (int32_t*)rad_dev_alloc(rows * cfg_.max_candidates * 4, RAD_MEM_HOST_PINNED);
        lp_cv_host_ = (float*)rad_dev_alloc(rows * cfg_.max_candidates * 4, RAD_MEM_HOST_PINNED);
        if (!lp_host_ || !lp_ci_host_ || !lp_cv_host_) {
            RAD_ERR("sampler: could not pin %s for the logprobs readback",
                    humanb(rows * (w + 2 * cfg_.max_candidates) * 4).c_str());
            stage_free();
            return RAD_E_NOMEM;
        }
    }

    if (!params_stage_ || !greedy_stage_ || !late_stage_ || !hist_stage_ || !brk_stage_ ||
        !mask_stage_) {
        RAD_ERR("sampler: could not pin %s of staging",
                humanb(rows * (3 * (int64_t)sizeof(DeviceSampleParams) + cfg_.max_hist * 5 +
                               mask_words_ * 4)).c_str());
        stage_free();
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

void Sampler::stage_free() {
    if (params_stage_) rad_dev_free(params_stage_, RAD_MEM_HOST_PINNED);
    if (greedy_stage_) rad_dev_free(greedy_stage_, RAD_MEM_HOST_PINNED);
    if (late_stage_)   rad_dev_free(late_stage_,   RAD_MEM_HOST_PINNED);
    if (hist_stage_)   rad_dev_free(hist_stage_,   RAD_MEM_HOST_PINNED);
    if (brk_stage_)    rad_dev_free(brk_stage_,    RAD_MEM_HOST_PINNED);
    if (mask_stage_)   rad_dev_free(mask_stage_,   RAD_MEM_HOST_PINNED);
    if (lp_host_)      rad_dev_free(lp_host_,      RAD_MEM_HOST_PINNED);
    if (lp_ci_host_)   rad_dev_free(lp_ci_host_,   RAD_MEM_HOST_PINNED);
    if (lp_cv_host_)   rad_dev_free(lp_cv_host_,   RAD_MEM_HOST_PINNED);
    lp_host_      = nullptr;
    lp_ci_host_   = nullptr;
    lp_cv_host_   = nullptr;
    params_stage_ = nullptr;
    greedy_stage_ = nullptr;
    late_stage_   = nullptr;
    cur_stage_    = nullptr;
    hist_stage_   = nullptr;
    brk_stage_    = nullptr;
    mask_stage_   = nullptr;
}

Sampler::~Sampler() { stage_free(); }

/* ================================================================== admission */

/* WHEN THE TWO HISTORY STAGES RUN, in one place, because admission, request setup and the step
 * staging each ask and must not disagree. llama.cpp's meaning throughout: a window of 0 turns the
 * stage off, and DRY also needs a base of at least 1. The multiplier is taken as positive because
 * a negative one rewards repetition, which is not a request anyone makes on purpose. */
static bool penalties_on(const SamplingParams& sp) {
    return sp.penalty_last_n != 0 &&
           (sp.rep_penalty != 1.0f || sp.freq_penalty != 0.0f || sp.pres_penalty != 0.0f);
}

static bool dry_on(const SamplingParams& sp) {
    return sp.dry_multiplier > 0.0f && sp.dry_base >= 1.0f && sp.dry_penalty_last_n != 0;
}

int Sampler::check_request(const SamplingParams& sp, std::string* why) const {
    auto refuse = [&](const char* what, int st) {
        if (why) *why = what;
        return st;
    };

    if (sp.top_k < 0)                       return refuse("top_k is negative", RAD_E_INVAL);
    if (sp.top_p < 0.0f || sp.top_p > 1.0f) return refuse("top_p is outside [0,1]", RAD_E_INVAL);
    if (sp.min_p < 0.0f || sp.min_p > 1.0f) return refuse("min_p is outside [0,1]", RAD_E_INVAL);
    if (sp.typical_p < 0.0f || sp.typical_p > 1.0f)
        return refuse("typical_p is outside [0,1]", RAD_E_INVAL);
    if (sp.temp < 0.0f)                     return refuse("temperature is negative", RAD_E_INVAL);
    if (sp.top_k > cfg_.max_candidates)
        return refuse("top_k exceeds the sampler's candidate width (--max-candidates)",
                      RAD_E_INVAL);
    if (sp.n_probs > RAD_MAX_LOGPROBS) {
        if (why) *why = "at most " + std::to_string(RAD_MAX_LOGPROBS) +
                        " logprobs alternatives may be asked for";
        return RAD_E_INVAL;
    }
    /* Every breaker is a pass over the whole vocabulary on the scheduler thread, and every other
     * sequence on the server waits for it; see kDryMaxBreakers. Refused here, before any of that
     * work is queued, rather than truncated -- a request whose breakers quietly went missing is a
     * request whose output changed. */
    if (dry_on(sp) && sp.dry_seq_breakers.size() > kDryMaxBreakers) {
        if (why)
            *why = "dry_sequence_breakers has " + std::to_string(sp.dry_seq_breakers.size()) +
                   " entries; at most " + std::to_string(kDryMaxBreakers) + " are accepted";
        return RAD_E_INVAL;
    }

    /* An unresolved sampler is a named refusal, not a silent skip. A request whose min_p quietly
     * stopped being applied is a request whose output changed and nothing said so. */
    auto need = [&](bool wanted, rad_op h, const char* n) -> const char* {
        return (wanted && !h) ? n : nullptr;
    };
    /* A SPLIT VOCABULARY MOVES GREEDY ONTO THE CHAIN, so what a greedy request needs moves with
     * it: not the argmax, but the temperature stage, the top-k, the merge and the pick that
     * build_step's top_k = 1 sends it through -- run() issues the temperature stage for every
     * batch with a chain row, even though its kernel leaves a greedy row (temperature 0) as it
     * is. Admission has to check what will actually be issued. */
    /* AND SO DOES ONE ASKING FOR LOGPROBS, which build_step stages as a top-1 chain row so its
     * token comes off a candidate plane (sampler.h). */
    const bool lp = sp.n_probs >= 0;
    const bool greedy_chain = sp.greedy() && (vocab_parallel() || lp);
    const char* missing = nullptr;
    if (!missing) missing = need(lp, op_logprobs_, "sample_logprobs");
    if (!missing) missing = need(sp.greedy() && !greedy_chain, op_argmax_, "sample_argmax");
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_temp_, "sample_temp");
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_pick_, "sample_pick");
    /* EVERY CHAIN ROW NEEDS THE TOP-K, whatever its own top_k says: it is the stage that writes
     * the candidate set every later stage reads. top_k = 0 asks for the whole candidate width,
     * not for no candidates. */
    if (!missing) missing = need(!sp.greedy() || greedy_chain, op_topk_, "sample_topk");
    if (!missing) missing = need(sp.top_p < 1.0f, op_topp_, "sample_topp");
    if (!missing) missing = need(sp.min_p > 0.0f, op_minp_, "sample_minp");
    if (!missing) missing = need(sp.typical_p < 1.0f, op_typical_, "sample_typical");
    if (!missing) missing = need(sp.xtc_probability > 0.0f, op_xtc_, "sample_xtc");
    if (!missing) missing = need(!sp.greedy() && sp.temp_last, op_temp_late_,
                                 "sample_temp (over the candidates, for temperature last)");
    if (!missing) missing = need(dry_on(sp), op_dry_, "sample_dry");
    if (!missing) missing = need(!sp.grammar.empty() || sp.thinking_budget >= 0, op_mask_,
                                 "sample_mask");
    if (!missing) missing = need(penalties_on(sp), op_penalties_, "sample_penalties");
    if (missing) {
        if (why) *why = std::string("no kernel resolved for '") + missing + "'";
        return RAD_E_NOKERNEL;
    }

    if (!sp.grammar.empty() && !vocab_)
        return refuse("a grammar was requested before the vocabulary was attached", RAD_E_STATE);

    /* The triggers are resolved here as well as at begin_request, so a template whose triggers
     * this machine cannot run is a refusal at admission rather than a request cancelled on the
     * scheduler thread. */
    if (!sp.grammar.empty() && !sp.grammar_triggers.empty()) {
        std::vector<std::string> words;
        std::vector<int32_t>     tokens;
        std::string              e;
        const int st = resolve_triggers(*vocab_, sp.grammar_triggers, &words, &tokens, &e);
        if (st < 0) {
            if (why) *why = e;
            return st;
        }
    }

    return RAD_OK;
}

int Sampler::begin_request(uint64_t req_id, const SamplingParams& sp, std::string* err) {
    SeqState st;

    if (!sp.grammar.empty()) {
        if (!vocab_) {
            if (err) *err = "no vocabulary attached";
            return RAD_E_STATE;
        }
        GrammarSpec spec;
        spec.kind = GrammarKind::Gbnf;
        spec.text = sp.grammar;
        spec.root = "root";
        spec.lazy = sp.grammar_lazy;
        /* ONLY A NON-LAZY GRAMMAR IS POSITIONED AT THE START OF THE TURN. An eager grammar's root
         * describes the whole assistant turn and therefore opens with the generation prompt, which
         * is already in the prompt -- so it has to be replayed or the model is made to write it
         * out again. A LAZY one describes only the region after its trigger: its root begins at
         * `<tool_call>`, and feeding it a prefix it never claimed to match fails outright, taking
         * every tool-calling request with it. */
        spec.prefix = sp.grammar_lazy ? std::string() : sp.grammar_prefix;
        RAD_TRY(resolve_triggers(*vocab_, sp.grammar_triggers, &spec.trigger_words,
                                 &spec.trigger_tokens, err));
        RAD_TRY(grammar_create(spec, *vocab_, &st.grammar, err));
    }

    /* BREAKERS ARE OPTIONAL. DRY runs whenever the request turns it on, as llama.cpp's does with an
     * empty breaker list; the breakers only limit how far back a repeat may be counted and exempt
     * the tokens that are breakers themselves. */
    if (dry_on(sp) && !sp.dry_seq_breakers.empty()) {
        if (!vocab_) {
            if (err) *err = "DRY needs a vocabulary to resolve its sequence breakers";
            return RAD_E_STATE;
        }
        if (sp.dry_seq_breakers.size() > kDryMaxBreakers) {
            if (err) *err = "too many DRY sequence breakers";
            return RAD_E_INVAL;
        }
        /* Once per request, never per step: a full vocabulary scan per breaker string is
         * milliseconds at a 151K vocab and has no business on the step path. */
        const int s = dry_prepare_breakers(*vocab_, sp.dry_seq_breakers, &st.dry,
                                           kDryBreakerMaxTail);
        if (s < 0) {
            if (err) *err = "a DRY sequence breaker could not be resolved against the vocabulary";
            return s;
        }
    }

    st.think.start(sp.reasoning);
    seqs_[req_id] = std::move(st);
    return RAD_OK;
}

void Sampler::end_request(uint64_t req_id) { seqs_.erase(req_id); }
bool Sampler::has_request(uint64_t req_id) const { return seqs_.find(req_id) != seqs_.end(); }

std::vector<uint64_t> Sampler::take_failed() {
    std::vector<uint64_t> v;
    v.swap(failed_);
    return v;
}

int Sampler::accept_token(uint64_t req_id, int32_t token) {
    auto it = seqs_.find(req_id);
    if (it == seqs_.end()) return RAD_OK;      /* no grammar, nothing to advance */
    if (!it->second.grammar) return RAD_OK;
    /* TIMED for the same reason build_step is: this runs once per ACCEPTED token per rank, on the
     * step thread, so it is wall time on every constrained token. */
    if ((int)log_level() < (int)Log::Debug) return it->second.grammar->accept_token(token);
    const auto t0 = std::chrono::steady_clock::now();
    const int s = it->second.grammar->accept_token(token);
    accept_us_ += std::chrono::duration<double, std::micro>(
                      std::chrono::steady_clock::now() - t0).count();
    if (++accept_n_ == 256) {
        RAD_DEBUG("sampler: grammar accept_token %.3f ms mean over 256", accept_us_ / 256000.0);
        accept_us_ = 0; accept_n_ = 0;
    }
    return s;
}

/* A CONSTRAINING GRAMMAR DOES NOT MEAN NO SPECULATION. build_step masks draft position i
 * against the state reached by accepting drafts 0..i-1 and then rewinds, so a backend that can
 * push and pop its state speculates like any other request -- which is worth several times the
 * per-token cost on a constrained request. What IS refused is a backend that cannot rewind:
 * masking its window would mean masking against a state the sequence is not in. */
bool Sampler::allows_speculation(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    if (it == seqs_.end() || !it->second.grammar) return true;
    return !it->second.grammar->constrains() || it->second.grammar->can_speculate();
}

bool Sampler::speculation_fixed(uint64_t req_id) const {
    auto it = seqs_.find(req_id);
    return it != seqs_.end() && (!it->second.grammar || it->second.grammar->can_speculate());
}

/* ================================================================== the step path */

int Sampler::build_step(const std::vector<Request*>& reqs) {
    if (!declared_) return RAD_E_STATE;

    const auto t_build = std::chrono::steady_clock::now();
    const int64_t cap = rows_for(cfg_);
    n_seq_ = 0;
    any_brk_ = false;
    any_late_ = false;
    cur_stage_ = params_stage_;
    lp_rows_.clear();

    /* DRY'S HOST HALF, for one row whose history `h[0, take)` is already staged. The breakers are
     * strings resolved against the vocabulary, so what they mean for this window is worked out
     * here and handed to the kernel as data: the repeat limit the most recent breaker imposes
     * (llama.cpp's step 1), and one byte per position marking the tokens that are whole breakers
     * and therefore never penalised. The window is the one the kernel will take -- the tail of
     * the staged row, `dry_penalty_last_n` long when that is positive.
     *
     * Returns false when the breakers rule DRY out for the row altogether: a breaker closer to the
     * end than `allowed_length` leaves no repeat long enough to count. A row with no breakers
     * gets zero marks, because the plane is shared by the whole batch and another row may have
     * some. */
    auto stage_dry = [&](DeviceSampleParams& p, const int32_t* h, int64_t take,
                         const DryBreakers* brk, int64_t row) -> bool {
        const int64_t last_n  = p.dry_penalty_last_n;
        const int64_t win     = last_n > 0 ? std::min<int64_t>(take, last_n) : take;
        const int64_t w0      = take - win;
        const int32_t allowed = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;
        uint8_t* f = brk_stage_ + row * cfg_.max_hist + w0;
        if (!brk) {
            std::memset(f, 0, (size_t)win);
            return true;
        }
        if (win <= allowed) return true;     /* the kernel takes its own early out */
        const int rep = dry_rep_limit(*brk, h + w0, (int)win);
        if (rep < allowed) return false;
        p.dry_rep_limit = rep < win ? rep : 0;
        for (int64_t i = 0; i < win; ++i) f[i] = brk->is_single(h[w0 + i]) ? 1 : 0;
        any_brk_ = true;
        return true;
    };

    for (Request* r : reqs) {
        if (!r) continue;
        const SamplingParams& sp = r->sp;
        const int n_pos = r->n_draft + 1;

        if (n_pos > 1 && !allows_speculation(r->id)) {
            RAD_ERR("sampler: request %llu carries a constraining grammar its backend cannot "
                    "snapshot, and %d draft tokens. The mask for draft position i is the mask of "
                    "the state reached by accepting drafts 0..i-1, so masking the window needs "
                    "push_state/rollback. The scheduler must consult allows_speculation() and "
                    "decode this one a token at a time.",
                    (unsigned long long)r->id, r->n_draft);
            return RAD_E_UNSUPPORTED;
        }

        if (n_seq_ + n_pos > cap) {
            RAD_ERR("sampler: %lld sampled positions exceeds the %lld the arena was sized for",
                    (long long)(n_seq_ + n_pos), (long long)cap);
            return RAD_E_FULL;
        }

        const int64_t n_prompt = (int64_t)r->prompt.size();
        const int64_t n_out    = (int64_t)r->output.size();
        const int64_t n_total  = n_prompt + n_out;
        const bool    pen      = penalties_on(sp);
        const bool    dry      = dry_on(sp);

        auto it = seqs_.find(r->id);
        SeqState* st = it == seqs_.end() ? nullptr : &it->second;
        const DryBreakers* brk = st && !st->dry.empty() ? &st->dry : nullptr;

        /* THE WINDOW IS WALKED, THEN REWOUND. Position 0 is masked against the state the sequence
         * is really in; position k against the state reached by accepting drafts 0..k-1. The push
         * happens whatever the grammar's current mode, because a draft token can fire a lazy
         * grammar's trigger mid-window -- at which point the positions behind it DO need a mask,
         * and without the push they would be sampled unconstrained.
         *
         * `dead` is not a failure. A draft the grammar forbids simply cannot be accepted: the
         * target's own masked pick at that position will differ from it, so verification rejects
         * there and every position behind it is discarded. Masking stops and the rest of the
         * window is left unconstrained, which is what "discarded" needs it to be. */
        GrammarBackend* gr = st && st->grammar ? st->grammar.get() : nullptr;
        const bool walk = gr && n_pos > 1;
        if (walk) gr->push_state();
        bool dead = false;

        /* THE REASONING BUDGET, walked over the window as the grammar is. Position k is inside the
         * block or not according to the committed reply and drafts 0..k-1, which is the only
         * history under which its token is ever emitted; a draft that ends the block early is
         * accepted or rejected like any other. */
        const bool budget = st && sp.thinking_budget >= 0 && sp.reasoning.known();
        ReasoningCount think;
        if (budget) {
            if (st->think_seen > (size_t)n_out) {
                st->think.start(sp.reasoning);
                st->think_seen = 0;
            }
            for (; st->think_seen < (size_t)n_out; ++st->think_seen)
                st->think.push(sp.reasoning, r->output[st->think_seen]);
            think = st->think;
        }

        for (int k = 0; k < n_pos; ++k) {
            if (walk && k > 0 && !dead && gr->accept_token(r->draft[k - 1]) < 0) dead = true;
            if (budget && k > 0) think.push(sp.reasoning, r->draft[k - 1]);
            const int64_t row = n_seq_++;

            DeviceSampleParams& p = params_stage_[row];
            std::memset(&p, 0, sizeof p);

            p.seed = sp.seed;
            /* The output position, advanced across the speculative window: draft position k is
             * the (n_out + k)-th token this request will emit, so its randomness is the same
             * whether it arrived through speculation or one token at a time. */
            p.pos = (uint64_t)(n_out + k);

            p.temp        = sp.temp;
            p.top_p       = sp.top_p;
            p.min_p       = sp.min_p;
            p.typical_p   = sp.typical_p;
            p.rep_penalty = sp.rep_penalty;
            p.freq_penalty = sp.freq_penalty;
            p.pres_penalty = sp.pres_penalty;
            p.xtc_probability = sp.xtc_probability;
            p.xtc_threshold   = sp.xtc_threshold;
            p.dry_multiplier  = sp.dry_multiplier;
            p.dry_base        = sp.dry_base;
            p.top_k              = sp.top_k;
            p.penalty_last_n     = sp.penalty_last_n;
            p.dry_allowed_length = sp.dry_allowed_length;
            p.dry_penalty_last_n = sp.dry_penalty_last_n;
            p.mask_row = -1;

            /* GREEDY OVER A SPLIT VOCABULARY IS A TOP-1 CHAIN, not an argmax. sample_argmax reads a
             * whole logits row and writes the winning COLUMN, which under vocab parallelism is
             * this rank's local index into its own shard -- a fluent answer built from the wrong
             * token ids. Staging top_k = 1 sends the row through top-k, the pair plane, the
             * gather and the merge instead, which is where a local index becomes a global one;
             * sample_pick over a single live candidate returns it deterministically, so the
             * result is the argmax it was always meant to be.
             *
             * Such a row is a CHAIN row and does not carry RAD_SP_GREEDY: every candidate stage
             * skips a row that does, because in an unsplit batch that flag is what keeps the pick
             * from overwriting the argmax's answer with a draw. */
            /* A GREEDY ROW ASKING FOR LOGPROBS IS A TOP-1 CHAIN ROW TOO, for a reason of its own:
             * its token has to come off a candidate plane, which is where logprobs_for reads the
             * value it was drawn at. The argmax's tie rule is the top-k's, so the token is the
             * same. */
            const bool want_lp = sp.n_probs >= 0;
            if (sp.greedy()) {
                if (vocab_parallel() || want_lp) p.top_k = 1;
                else                             p.flags |= RAD_SP_GREEDY;
            }
            if (want_lp) p.flags |= RAD_SP_LOGPROBS;
            if (sp.xtc_probability > 0.0f) p.flags |= RAD_SP_XTC;
            /* TEMPERATURE LAST: the row's own temperature in the late stage's row, and 1 -- which
             * the temperature stage skips -- in whichever of the two it does not run in. Every row
             * gets a late row, because the late stage is issued over the whole batch. */
            const bool late = sp.temp_last && !sp.greedy() && sp.temp != 1.0f;
            DeviceSampleParams& lp = late_stage_[row];
            lp = p;
            lp.temp = late ? sp.temp : 1.0f;
            if (late) {
                p.temp = 1.0f;
                any_late_ = true;
            }

            /* THE HISTORY THE PENALTIES AND DRY READ, staged only for a row that runs one of them
             * -- the copy is scheduler-thread work and the upload is step-path bytes, and neither
             * buys anything for a row that reads none of it. It is the prompt, everything
             * generated so far and, at draft position k, the k drafted tokens in front of it: the
             * token sampled there is emitted only if those drafts are, so they are part of the
             * output it is penalised against, exactly as they would be one token at a time. Oldest
             * first, truncated to the declared window -- a bound on the buffer, not a policy on
             * the sampler; penalty_last_n still decides how much of it counts. The row owns its
             * own row of the plane, so the window starts at offset 0 within it. */
            p.hist_off = 0;
            p.hist_len = 0;
            if (pen || dry) {
                const int64_t len  = n_total + k;
                const int64_t take = std::min<int64_t>(len, cfg_.max_hist);
                int32_t* h = hist_stage_ + row * cfg_.max_hist;
                for (int64_t i = 0; i < take; ++i) {
                    const int64_t src = len - take + i;
                    h[i] = src < n_prompt ? r->prompt[(size_t)src]
                         : src < n_total  ? r->output[(size_t)(src - n_prompt)]
                                          : r->draft[src - n_total];
                }
                p.hist_len = (int32_t)take;
                if (pen) p.flags |= RAD_SP_PENALTY;
                if (dry && stage_dry(p, h, take, brk, row)) p.flags |= RAD_SP_DRY;
            }

            if (gr && !dead && gr->constrains()) {
                uint32_t* dst = mask_stage_ + row * mask_words_;
                int s;
                /* WHAT THE MASK COSTS, MEASURED RATHER THAN ASSUMED. The mask is scheduler-thread
                 * work with the rank threads waiting at the barrier, so it is wall time on every
                 * constrained token and would decide the step if it were slow. Reported at Debug,
                 * per 256 masks, mean and worst. */
                const bool   inst = (int)log_level() >= (int)Log::Debug;
                const auto   t0   = std::chrono::steady_clock::now();
                /* ASK THE GRAMMAR WHETHER IT COMPUTED THE MASK, rather than inferring it from how
                 * long the call took. No duration threshold separates the two: a mask computed
                 * from the plan can finish inside the time a memo hit costs on a large
                 * vocabulary, so a timing test counts computed masks as hits and any per-mask
                 * cost derived from the split is wrong by that factor. The counter is exact and
                 * costs a relaxed load. */
                const uint64_t w0 = inst ? st->grammar->masks_walked() : 0;
                if (!vocab_parallel()) {
                    s = st->grammar->fill_mask(dst, mask_words_);
                } else {
                    s = st->grammar->fill_mask(mask_full_.data(), (int64_t)mask_full_.size());
                    if (s >= 0) shard_mask(dst);
                }
                if (inst) {
                    const double us = std::chrono::duration<double, std::micro>(
                                          std::chrono::steady_clock::now() - t0).count();
                    mask_us_ += us; mask_worst_us_ = mask_us_n_ && us < mask_worst_us_
                                                   ? mask_worst_us_ : us;
                    if (st->grammar->masks_walked() != w0) { ++mask_slow_; mask_walk_us_ += us; }
                    else                                     ++mask_fast_;
                    if (++mask_us_n_ == 256) {
                        /* Mean per MASK and mean per COMPUTED mask are different questions and
                         * the second is the one that says what a new state costs: a window's
                         * masks are mostly memo hits, so the first tracks how many positions
                         * asked. */
                        RAD_DEBUG("sampler: grammar mask %.2f ms mean, %.2f ms worst over 256; "
                                  "%lld memo hits, %lld computed at %.2f ms mean (%lld vocab, "
                                  "%lld words)",
                                  mask_us_ / 256000.0, mask_worst_us_ / 1000.0,
                                  (long long)mask_fast_, (long long)mask_slow_,
                                  mask_slow_ ? mask_walk_us_ / (1000.0 * (double)mask_slow_) : 0.0,
                                  (long long)cfg_.n_vocab, (long long)mask_words_);
                        mask_us_ = 0; mask_us_n_ = 0; mask_worst_us_ = 0;
                        mask_fast_ = 0; mask_slow_ = 0; mask_walk_us_ = 0.0;
                    }
                }
                if (s < 0 && k > 0) {
                    /* A SPECULATIVE POSITION IS NOT A DEAD REQUEST. This is the state reached by
                     * accepting drafts the sequence has not committed to, so "nothing is
                     * admissible here" says the draft is wrong, not that the request is stuck.
                     * Verification will reject at or before this position; stop masking and leave
                     * the rest of the window unconstrained, since it is about to be discarded. */
                    dead = true;
                } else if (s < 0) {
                    /* A DEAD GRAMMAR FAILS ITS REQUEST, NOT THE STEP. Returning a status here
                     * would reach the step loop as an engine fault and stop it stepping, so every
                     * other client on the server would stop receiving tokens because one of them
                     * sent a schema the model painted itself out of. The row is left unmasked for
                     * this step -- it is about to be cancelled -- and the id goes to the engine
                     * to close. */
                    RAD_WARN("sampler: request %llu has no admissible token under its grammar; "
                             "failing that request", (unsigned long long)r->id);
                    failed_.push_back(r->id);
                } else {
                    p.mask_row = (int32_t)row;
                    p.flags |= RAD_SP_MASK;
                    if ((int)log_level() >= (int)Log::Trace) {
                        /* How many tokens the grammar allows THIS RANK to pick. Zero is legal
                         * under sharding -- the admissible set can live entirely on another rank
                         * -- and is the first thing to check when a grammar-constrained answer
                         * comes out unconstrained. */
                        int64_t bits = 0;
                        for (int64_t w = 0; w < mask_words_; ++w)
                            bits += __builtin_popcount(dst[w]);
                        RAD_TRACE("sampler: req %llu row %lld mask allows %lld of %lld local "
                                  "tokens (vocab_off %lld)", (unsigned long long)r->id,
                                  (long long)row, (long long)bits,
                                  (long long)cfg_.n_vocab_local, (long long)cfg_.vocab_off);
                    }
                }
            }

            if (budget && think.inside() && think.body >= sp.thinking_budget)
                force_close(p, row, sp.reasoning.close);

            if (want_lp) {
                /* The temperature the candidate value carries: the early stage's and the late
                 * one's, each only where its kernel runs (temp > 0 and not 1). */
                auto t_of = [](float t) { return t > 0.0f && t != 1.0f ? t : 1.0f; };
                LpRow L;
                L.row   = row;
                L.n_top = sp.n_probs;
                L.temp  = t_of(p.temp) * t_of(late_stage_[row].temp);
                L.p     = p;
                if (p.hist_len > 0) {
                    const int32_t* h = hist_stage_ + row * cfg_.max_hist;
                    L.hist.assign(h, h + p.hist_len);
                }
                L.dry = brk;
                lp_rows_.push_back(std::move(L));
            }
        }

        /* Back to where the sequence actually is. The tokens it really accepted are applied by
         * Sampler::accept_token after the step verifies, so anything the walk did must be gone
         * before then or the grammar would be ahead of the sequence by the whole window. */
        if (walk) RAD_TRY(gr->rollback());
    }

    /* THE WHOLE OF IT, not just the mask, because that is what the rank threads wait for. Without
     * this the mask's own number could fall to nothing and the step stay slow with no way to see
     * which side of the barrier the time was on. */
    if ((int)log_level() >= (int)Log::Debug) {
        build_us_ += std::chrono::duration<double, std::micro>(
                         std::chrono::steady_clock::now() - t_build).count();
        if (++build_n_ == 256) {
            RAD_DEBUG("sampler: build_step %.3f ms mean over 256 (%lld rows)",
                      build_us_ / 256000.0, (long long)n_seq_);
            build_us_ = 0; build_n_ = 0;
        }
    }
    return RAD_OK;
}

void Sampler::force_close(DeviceSampleParams& p, int64_t row, int32_t close) {
    uint32_t* dst = mask_stage_ + row * mask_words_;
    const int64_t at = (int64_t)close - cfg_.vocab_off;
    const bool local = at >= 0 && at < cfg_.n_vocab_local;
    if (p.flags & RAD_SP_MASK) {
        /* The grammar's mask for this row. Under a split vocabulary the whole one is still in
         * mask_full_, and the close may be another rank's column. */
        const bool admitted =
            vocab_parallel()
                ? close < (int64_t)mask_full_.size() * 32 &&
                      (mask_full_[(size_t)(close >> 5)] >> (uint32_t)(close & 31)) & 1u
                : local && (dst[at >> 5] >> (uint32_t)(at & 31)) & 1u;
        if (!admitted) return;
    }
    /* ONE ADMISSIBLE TOKEN, on whichever rank holds its column. The others mask the whole row
     * out, which the candidate merge reads as no candidate from them (the grammar's own masks do
     * the same when the admissible set lives on one shard). */
    std::memset(dst, 0, (size_t)mask_words_ * 4);
    if (local) dst[at >> 5] |= 1u << (uint32_t)(at & 31);
    p.mask_row = (int32_t)row;
    p.flags |= RAD_SP_MASK;
}

int Sampler::build_greedy(int64_t n) {
    if (!declared_) return RAD_E_STATE;
    if (n < 0) return RAD_E_INVAL;
    const int64_t cap = rows_for(cfg_);
    if (n > cap) {
        RAD_ERR("sampler: %lld greedy rows exceeds the %lld the arena was sized for",
                (long long)n, (long long)cap);
        return RAD_E_FULL;
    }
    n_seq_ = n;
    any_brk_ = false;
    cur_stage_ = greedy_stage_;
    if (greedy_ready_) return RAD_OK;
    for (int64_t i = 0; i < cap; ++i) {
        DeviceSampleParams& p = greedy_stage_[i];
        std::memset(&p, 0, sizeof p);
        p.temp     = 0.0f;
        p.top_k    = 1;
        p.hist_off = 0;
        p.hist_len = 0;
        p.mask_row = -1;
        /* Over a split vocabulary a greedy row is a top-1 CHAIN row, for the reason build_step
         * gives, and so it does not carry the flag every candidate stage skips. */
        p.flags    = vocab_parallel() ? 0 : RAD_SP_GREEDY;
    }
    greedy_ready_ = true;
    return RAD_OK;
}

int Sampler::run(RadCtx* c, rad_buf logits, rad_buf out_tokens) {
    if (!declared_) return RAD_E_STATE;
    if (!c || !logits || !out_tokens) return RAD_E_INVAL;
    if (n_seq_ == 0) return RAD_OK;

    const RadStream s = rad_stream(c);

    bool any_mask = false, any_greedy = false, any_chain = false;
    bool any_penalty = false, any_dry = false, any_xtc = false;
    bool any_topp = false, any_minp = false, any_typ = false;
    for (int64_t i = 0; i < n_seq_; ++i) {
        const DeviceSampleParams& p = cur_stage_[i];
        any_mask    |= (p.flags & RAD_SP_MASK)    != 0;
        any_penalty |= (p.flags & RAD_SP_PENALTY) != 0;
        any_dry     |= (p.flags & RAD_SP_DRY)     != 0;
        any_xtc     |= (p.flags & RAD_SP_XTC)     != 0;
        if (p.flags & RAD_SP_GREEDY) any_greedy = true;
        else {
            any_chain = true;
            any_topp |= p.top_p < 1.0f;
            any_minp |= p.min_p > 0.0f;
            any_typ  |= p.typical_p < 1.0f;
        }
    }

    /* A GREEDY ROW OVER A SPLIT VOCABULARY is staged as a top-1 chain row, never with the flag:
     * the argmax would return this rank's column for a token id. build_step and build_greedy both
     * hold that, so a flagged row here is a staging bug, and it is refused rather than served. */
    if (any_greedy && vocab_parallel()) {
        RAD_ERR("sampler: a greedy-flagged row reached a split vocabulary, where the argmax "
                "returns a column of this rank's shard rather than a token id");
        return RAD_E_STATE;
    }

    /* Async copies on the compute stream and no host synchronisation, and only of what some row
     * will read: the history and the breaker marks for the penalties and DRY, the mask for a
     * grammar. The mask upload is 19 KiB a constrained sequence at a 151K vocab, which is the
     * number that makes a host-side grammar affordable at all (spec §13). */
    RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_params_), cur_stage_,
                             n_seq_ * (int64_t)sizeof(DeviceSampleParams), s));
    /* The late temperature's rows only when this build staged one: a greedy build's rows come
     * from greedy_stage_, and late_stage_ then holds an older build's. */
    const bool late = any_late_ && cur_stage_ == params_stage_;
    if (late)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_params_late_), late_stage_,
                                 n_seq_ * (int64_t)sizeof(DeviceSampleParams), s));
    if (any_penalty || any_dry)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_hist_), hist_stage_,
                                 n_seq_ * cfg_.max_hist * 4, s));
    const bool pass_brk = any_dry && any_brk_;
    if (pass_brk)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_brk_), brk_stage_,
                                 n_seq_ * cfg_.max_hist, s));
    if (any_mask)
        RAD_TRY(rad_memcpy_async(rad_buf_ptr(c, buf_mask_), mask_stage_,
                                 n_seq_ * mask_words_ * 4, s));

    const RadOperand o_logits = opd_rows(logits, n_seq_);
    const RadOperand o_params = opd_rows(buf_params_, n_seq_);
    const RadOperand o_hist = opd_rows(buf_hist_, n_seq_);
    const RadOperand o_brk = opd_rows(buf_brk_, n_seq_);
    const RadOperand o_mask = opd_rows(buf_mask_, n_seq_);
    const RadOperand o_ci = opd_rows(buf_cand_i_, n_seq_);
    const RadOperand o_cv = opd_rows(buf_cand_v_, n_seq_);
    const RadOperand o_tok = opd_rows(out_tokens, n_seq_);

    auto issue = [&](rad_op op, std::initializer_list<RadOperand> ops) -> int {
        std::vector<RadOperand> v(ops);
        return rad_issue(c, op, v.data(), (int)v.size(), n_seq_);
    };

    /* THE LOGPROBS SUMMARY, ON THE LOGITS AS THE MODEL WROTE THEM: every stage below writes them in
     * place. Only for a trunk build that staged a logprobs row -- a draft pass's greedy build
     * stages none, and must not overwrite the summary rank 0 is about to copy back. */
    const bool lp = !lp_rows_.empty() && cur_stage_ == params_stage_ && op_logprobs_;
    const RadOperand o_lp = opd_rows(buf_lp_, n_seq_);
    if (lp) {
        RAD_TRY(issue(op_logprobs_, { o_logits, o_params, o_lp }));
        if (vocab_parallel())
            RAD_TRY(issue(op_lp_gather_, { o_lp, opd_rows(buf_lp_g_, n_seq_) }));
    }

    /* Penalties act on raw logits, before anything derives a probability from them. DRY takes the
     * breaker marks only when some row in the batch has breakers; its fourth operand is optional
     * and absent means no token is exempt. */
    if (any_penalty) RAD_TRY(issue(op_penalties_, { o_logits, o_params, o_hist }));
    if (any_dry)
        RAD_TRY(issue(op_dry_, { o_logits, o_params, o_hist, pass_brk ? o_brk : opd_none() }));

    /* The greedy rows take the argmax after penalties and the mask and never see temperature.
     * When the batch is mixed, BOTH the argmax and the chain are issued over the whole batch: the
     * argmax writes every row's token, and then every candidate stage skips the greedy rows
     * (RAD_SP_GREEDY) so the pick overwrites only the rows the chain owns. The cost is one extra
     * pass over the logits for the rows the other op owns -- ~77 MiB read at batch 256 -- against
     * one kernel launch per distinct parameter set, which at batch 256 is up to 256 launches. The
     * pass wins.
     *
     * Under vocab parallelism there is no greedy shortcut at all: build_step staged those rows at
     * top_k = 1, without the flag, so they take the chain, where the merge turns a local candidate
     * index into a global token id. */
    if (any_greedy) {
        if (any_mask) RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));
        RAD_TRY(issue(op_argmax_, { o_logits, o_params, o_tok }));
    }

    if (!any_chain) return RAD_OK;

    RAD_TRY(issue(op_temp_, { o_logits, o_params }));
    if (any_mask && !any_greedy)
        RAD_TRY(issue(op_mask_, { o_logits, o_params, o_mask }));

    /* top-k reduces the vocabulary to a candidate set, and everything after it works on that set.
     * Under tensor parallel this is the LOCAL top-k over this rank's shard; the candidate sets
     * are combined below and full logits are never gathered (spec §9).
     *
     * IT IS ISSUED WHENEVER THE CHAIN RUNS, not only when some row sets top_k. A row at
     * top_k = 0 asks for the whole candidate width, and the candidate planes are transient: a
     * step that skipped this would hand the pick whatever the planes last held -- the previous
     * step's candidates, or another op's bytes. */
    if (vocab_parallel()) {
        /* The fifth operand is the pair plane: this rank's candidates with GLOBAL ids, which is
         * what crosses the wire. cand_idx and cand_val still come back LOCAL and the merge
         * overwrites both, so nothing downstream ever sees a local id. */
        const RadOperand o_p = opd_rows(buf_pairs_, n_seq_);
        const RadOperand o_g = opd_rows(buf_gather_, n_seq_);
        RAD_TRY(issue(op_topk_,      { o_logits, o_params, o_ci, o_cv, o_p }));
        RAD_TRY(issue(op_allgather_, { o_p, o_g }));
        RAD_TRY(issue(op_merge_,     { o_g, o_params, o_ci, o_cv }));
    } else {
        /* No pair plane on a whole vocabulary, and it is passed as absent rather than left off:
         * the schema has five operands and an issue of four is refused. */
        RAD_TRY(issue(op_topk_, { o_logits, o_params, o_ci, o_cv, opd_none() }));
    }

    if (any_topp) RAD_TRY(issue(op_topp_,    { o_ci, o_cv, o_params }));
    if (any_minp) RAD_TRY(issue(op_minp_,    { o_ci, o_cv, o_params }));
    if (any_typ)  RAD_TRY(issue(op_typical_, { o_ci, o_cv, o_params }));
    if (any_xtc)  RAD_TRY(issue(op_xtc_,     { o_ci, o_cv, o_params }));
    if (late)
        RAD_TRY(issue(op_temp_late_, { o_cv, opd_rows(buf_params_late_, n_seq_) }));

    RAD_TRY(issue(op_pick_, { o_ci, o_cv, o_params, o_tok }));

    /* AND THE LOGPROBS ROWS ON THEIR WAY TO THE HOST, behind the pick on the same stream: the
     * summary, and the candidate planes the pick drew from (logprobs_for reads a token's value
     * there when the summary does not hold it). Only the rows between the first and the last
     * that asked, and only on rank 0, which is the rank collect_step asks. */
    if (lp && cfg_.rank == 0) {
        lp_lo_ = lp_rows_.front().row;
        lp_hi_ = lp_rows_.back().row + 1;
        const int64_t w = lp_width() * (vocab_parallel() ? cfg_.world_size : 1);
        const int64_t nc = cfg_.max_candidates, n = lp_hi_ - lp_lo_;
        const char* sp = (const char*)rad_buf_ptr(c, vocab_parallel() ? buf_lp_g_ : buf_lp_);
        const char* ci = (const char*)rad_buf_ptr(c, buf_cand_i_);
        const char* cv = (const char*)rad_buf_ptr(c, buf_cand_v_);
        if (!sp || !ci || !cv) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(lp_host_ + lp_lo_ * w, sp + lp_lo_ * w * 4, n * w * 4, s));
        RAD_TRY(rad_memcpy_async(lp_ci_host_ + lp_lo_ * nc, ci + lp_lo_ * nc * 4, n * nc * 4, s));
        RAD_TRY(rad_memcpy_async(lp_cv_host_ + lp_lo_ * nc, cv + lp_lo_ * nc * 4, n * nc * 4, s));
    }
    return RAD_OK;
}

/* ================================================================== logprobs */

float Sampler::raw_logit(const LpRow& L, int32_t token, float v) const {
    const DeviceSampleParams& p = L.p;
    double y = (double)v * (double)L.temp;
    /* DRY subtracted a penalty that depends on the token and the window alone, so the host
     * reference computes it for this one token: a single candidate at 0 comes back at minus the
     * penalty. */
    if (p.flags & RAD_SP_DRY) {
        static const DryBreakers kNone;
        CandidateSet s;
        s.c.push_back(Candidate{ token, 0.0f, 0.0f });
        DryParams dp;
        dp.multiplier     = p.dry_multiplier;
        dp.base           = p.dry_base;
        dp.allowed_length = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;
        dp.penalty_last_n = p.dry_penalty_last_n < 0 ? -1 : p.dry_penalty_last_n;
        host_dry(s, L.hist.data(), (int64_t)L.hist.size(), (int64_t)L.hist.size(), dp,
                 L.dry ? *L.dry : kNone);
        y -= (double)s.c[0].logit;
    }
    /* The penalties, inverted: presence and frequency added back, then the repetition penalty
     * undone on the side of zero the logit is on -- it never changes sign. The window is the
     * stage's: the tail of the staged history, penalty_last_n long when that is positive. */
    if (p.flags & RAD_SP_PENALTY) {
        int64_t n = (int64_t)L.hist.size(), o = 0;
        if (p.penalty_last_n > 0 && n > p.penalty_last_n) { o = n - p.penalty_last_n; n = p.penalty_last_n; }
        int64_t count = 0;
        for (int64_t i = 0; i < n; ++i) count += L.hist[(size_t)(o + i)] == token;
        if (count > 0) {
            y += (double)count * p.freq_penalty + p.pres_penalty;
            y = y > 0.0 ? y * p.rep_penalty : y / p.rep_penalty;
        }
    }
    return (float)y;
}

int merge_logprob_rows(const float* st, int W, int k, double* log_z, int32_t* ids, float* vals,
                       int cap) {
    const int64_t w = 2 + 2 * (int64_t)k;
    auto live = [&](int r) {
        const float m = st[r * w], z = st[r * w + 1];
        return std::isfinite(m) && std::isfinite(z) && z > 0.0f;
    };
    double mx = -INFINITY;
    for (int r = 0; r < W; ++r)
        if (live(r)) mx = std::max(mx, (double)st[r * w]);
    double sum = 0.0;
    for (int r = 0; r < W; ++r)
        if (live(r)) sum += (double)st[r * w + 1] * std::exp((double)st[r * w] - mx);
    *log_z = sum > 0.0 ? mx + std::log(sum) : NAN;

    /* On the stack for every split the engine runs (eight ranks of 21 pairs is 168); a call for
     * more takes the heap rather than a bound. Once per emitted token, on collect_step's path. */
    struct Pair { float v; int32_t id; };
    Pair buf[256];
    std::vector<Pair> heap;
    Pair* top = buf;
    if ((int64_t)W * k > (int64_t)(sizeof buf / sizeof buf[0])) {
        heap.resize((size_t)W * (size_t)k);
        top = heap.data();
    }
    int m = 0;
    for (int r = 0; r < W; ++r) {
        if (!live(r)) continue;
        for (int j = 0; j < k; ++j) {
            const int32_t id = (int32_t)st[r * w + 2 + 2 * j];
            const float   v  = st[r * w + 3 + 2 * j];
            if (id >= 0 && std::isfinite(v)) top[m++] = Pair{ v, id };
        }
    }
    std::sort(top, top + m, [](const Pair& a, const Pair& b) {
        return a.v > b.v || (a.v == b.v && a.id < b.id);
    });
    const int n = std::min(cap, m);
    for (int j = 0; j < n; ++j) {
        ids[j]  = top[j].id;
        vals[j] = top[j].v;
    }
    return n;
}

int Sampler::logprobs_for(int64_t row, int32_t token, TokenLogprobs* out) const {
    if (!out || !lp_host_ || row < lp_lo_ || row >= lp_hi_) return RAD_E_STATE;
    const LpRow* L = nullptr;
    for (const LpRow& x : lp_rows_)
        if (x.row == row) { L = &x; break; }
    if (!L) return RAD_E_STATE;

    /* Every rank's summary of its shard, merged: the maxima and normalisers by the usual
     * rescale, the largest by value with the lower id first. */
    const int W = vocab_parallel() ? cfg_.world_size : 1;
    double log_z = 0.0;
    int32_t ids[kLpK * 8];
    float   vals[kLpK * 8];
    const int n = merge_logprob_rows(lp_host_ + row * W * lp_width(), W, kLpK, &log_z, ids, vals,
                                     (int)(sizeof ids / sizeof ids[0]));
    /* A ROW WITH NO FINITE LOGIT HAS NO DISTRIBUTION to report against -- NaN logits out of the
     * model, or every one at -inf -- and any number sent for it would be invented. */
    if (!std::isfinite(log_z)) {
        RAD_WARN("sampler: row %lld has no finite logit; its logprobs cannot be computed",
                 (long long)row);
        out->token = -1;
        return RAD_E_INVAL;
    }

    out->token = token;
    out->n_top = std::min(L->n_top, n);
    for (int j = 0; j < out->n_top; ++j) {
        out->top_id[j] = ids[j];
        out->top_lp[j] = (float)((double)vals[j] - log_z);
    }

    /* The token's own raw logit: exact when it is among the largest, and otherwise recovered
     * from the value the pick drew it at. */
    double own = NAN;
    for (int j = 0; j < n && std::isnan(own); ++j)
        if (ids[j] == token) own = (double)vals[j];
    if (std::isnan(own)) {
        const int64_t nc = cfg_.max_candidates;
        const int32_t* ci = lp_ci_host_ + row * nc;
        const float*   cv = lp_cv_host_ + row * nc;
        for (int64_t c = 0; c < nc && std::isnan(own); ++c)
            if (ci[c] == token) own = (double)raw_logit(*L, token, cv[c]);
    }
    if (!std::isfinite(own)) {
        RAD_WARN("sampler: row %lld sampled token %d, which has no finite logit on the candidate "
                 "plane it was drawn from; its logprobs cannot be computed", (long long)row, token);
        out->token = -1;
        return RAD_E_INVAL;
    }
    out->logprob = (float)(own - log_z);
    return RAD_OK;
}

}  /* namespace rad */
