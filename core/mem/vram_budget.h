/* vram_budget.h -- how many bytes each VRAM pool gets, decided once and printed in full.
 *
 * ============================== WHY THIS IS NOT THE §6 VIOLATION IT LOOKS LIKE ==============================
 *
 * Spec §6 says the engine does not derive its budgets, and core/mem/pools.cpp still refuses to
 * re-optimise a split the operator stated. The reason given there is exact and worth repeating:
 * a budget derived from the FREE VRAM at startup is a benchmark that moves when somebody opens a
 * window on the machine's desktop.
 *
 * What that argument forbids is inference. It does not forbid arithmetic on numbers that are all
 * known, all printed, and all stable:
 *
 *   capacity        totalGlobalMem. A fixed property of the part.
 *   already held    capacity - the free VRAM this process can see, MEASURED once, after every
 *                   kernel plugin has initialised and its code objects are resident. This is the
 *                   driver's context and the loaded code, and it is the one term that is observed
 *                   rather than computed -- so it is printed on its own line, and a machine with a
 *                   desktop on the card shows up in it instead of silently eating a pool.
 *   --gpu-headroom-mib   what the operator wants left alone. Stated, defaulted to 96 MiB.
 *   arena           the activation arena and the scratch beside it, computed from the buffer plan
 *                   by arena_plan_bytes() -- the same call that later allocates it.
 *
 * What is left over is claimable, and it is spent in TWO STEPS, not one.
 *
 * THE STATIC WEIGHTS COME OFF THE TOP. Everything that is not an expert -- the attention
 * projections, the norms, the embedding, the head -- has to be resident for the model to run at a
 * sensible speed at all (the one exception is a token embedding --embedding-placement puts in the
 * pinned host pool, which a lookup reads a few rows of in place; weight_footprint counts it out of
 * the statics then), and under tensor parallel each rank already holds its own even share of
 * them (declare runs per rank with every dimension divided, spec §9). They are a FLOOR, not a
 * share of anything: dividing the weights as a whole would let the cache's side demote the
 * attention projections to host memory to make room for blocks nothing will fill. If they do not
 * fit, that is a refusal with the shortfall named.
 *
 * ONLY WHAT IS LEFT AFTER THAT IS DIVIDED: the expert plane against the KV cache. Where the cache
 * lends its idle blocks to the expert slab, that division has nothing to guess. The experts keep
 * on the card exactly what the pinned host pool cannot hold -- they have nowhere else to be -- and
 * the cache takes the rest, up to what it could ever address; every block no live request holds
 * is lent back to the experts in place. So the cache is as large as the work needs at the moment
 * and the experts have everything else, for a long-context server and a high-concurrency one
 * alike. Without a lender the cache would hold that memory empty, so there the experts come first
 * and the cache keeps one --max-model-len sequence. The cache never gets less than that sequence.
 *
 * BOTH SIDES ARE CAPPED AT WHAT THEY COULD USE, and whatever one side cannot take goes to the
 * other rather than nowhere. A cache cannot address past max_seqs x max_ctx tokens
 * (KVManager::ceiling), and an expert plane that fits entirely in VRAM has no use for more; when
 * both fit, both are whole.
 *
 * A stated --vram-weights-mib or --vram-kv-mib still wins outright and turns the whole of this
 * off for that pool. The point is that the DEFAULT is a full card rather than a guess: VRAM left
 * unbudgeted is VRAM no pool can use, and on a model whose experts do not fit, every unbudgeted
 * GiB costs measurable decode time.
 */
#pragma once
#include "../rad_core.h"

#include <string>
#include <vector>

namespace rad {

/* One rank's card and program, as the resolver needs to see them. Every field is either read from
 * the device or computed from the declared program; none of it is a policy. */
struct VramFacts {
    int64_t capacity   = 0;   /* RadDeviceProps::vram_bytes -- totalGlobalMem */
    int64_t free       = 0;   /* RadDeviceProps::vram_free, measured now */
    int64_t arena      = 0;   /* arena_plan_bytes() for this rank's program, total incl. scratch */
    int64_t staging    = 0;   /* BatchBuilder::device_bytes(): the step slab and the media staging,
                               * and TierExec::device_bytes() when a KV tier is configured */
    /* THE HARDWARE QUEUES this rank creates after `free` was read -- its compute stream, the expert
     * mover's, and the second compute lane where the program declares one -- at `queue_bytes`
     * each. MEASURED, by making one: a queue's kernel-argument ring is the engine's, and the
     * driver backs every queue with VRAM of its own beside it (33 MiB a queue on an R9700, eight
     * times the ring). */
    int     n_queues    = 0;
    int64_t queue_bytes = 0;
    int64_t kv_ceiling = 0;   /* KVManager::ceiling(max_ctx) for this rank's declared groups */
    int64_t kv_fixed   = 0;   /* KVManager::fixed_bytes(): the part of the ceiling that is state */
    /* weight_footprint() for this rank, which under tensor parallel is already this card's even
     * share: declare runs per rank with every dimension already divided. */
    int64_t statics    = 0;   /* non-expert weights -- the floor, not a share of anything */
    int64_t experts    = 0;   /* the expert plane: what trades against the KV cache */
    int64_t expert_layer_max = 0;   /* the most expert bytes one routed layer holds */
    /* THE EXPERTS THE PINNED HOST POOL CANNOT HOLD on this rank, with the slab's spare slots
     * beside them: they have nowhere to be but the card, so this is the floor under the expert
     * slab when the cache lends (expert_vram_floor, core/place/planner.h). */
    int64_t expert_floor = 0;
    int64_t kv_one     = 0;   /* the cache for ONE --max-model-len sequence: the least it may get */
    /* WILL THE CACHE LEND ITS IDLE BLOCKS TO THE EXPERT SLAB: a placement that moves units
     * (expert_tiered, or auto), not --deterministic, on a device with virtual memory management.
     * Decides the split: a cache that lends gives back what it does not use, so it can be sized
     * for what live requests could need; one that does not would hold it empty. */
    bool    lends      = false;
    /* Integrated graphics: what of the host's memory the card leaves to the host. Its device
     * memory IS the host's, so an unclaimed MiB is the operating system's and every other
     * process's to run in, not slack. 0 on a discrete card. */
    int64_t host_reserve = 0;
};

/* The resolved split, in bytes, beside every term it came from. Always filled, including when the
 * operator stated the budgets -- the placement report and the pool report both want the totals. */
struct VramBudget {
    int64_t capacity = 0, already_held = 0, headroom = 0, arena = 0, staging = 0;
    int64_t drift = 0;                /* kDriverDrift: what already_held moves by after it is read */
    int64_t queues = 0;               /* n_queues * queue_bytes, from the rank that needs most */
    int     n_queues = 0;
    int64_t queue_bytes = 0;
    int64_t claimable = 0;            /* after the fixed terms above */
    int64_t statics = 0;              /* off the top, before the split sees anything */
    int64_t elastic = 0;              /* claimable - statics: what experts and cache share */
    int64_t expert_floor = 0;         /* the experts the host pool cannot hold */
    bool    lends = false;            /* every rank's cache lends its idle blocks */
    int64_t kv_ceiling = 0, expert_ceiling = 0;
    int64_t kv_fixed = 0;             /* the state part of kv_ceiling, which no context uses */
    int64_t experts = 0, kv = 0;      /* the two shares, each capped at its ceiling */
    int64_t unusable = 0;             /* elastic neither side could take -- both were full */
    /* THE PREFILL STAGING REGION: two buffers, each one layer's share of the experts the budget
     * leaves off the card, taken out of the expert share and appended to the top of the activation
     * arena (core/place/stager.h). Zero when every expert is resident or the placement cannot move
     * any. */
    int64_t stage = 0;
    int64_t weights = 0;              /* statics + experts: what vram_weights is set to */
    bool    derived_weights = false, derived_kv = false;
    std::string report;       /* the block the engine prints at startup */
};

/* Writes the answer into cfg.vram_weights_mib / cfg.vram_kv_mib -- ONE resolved number that every
 * consumer downstream reads, so the planner, the KV manager and the pool report cannot disagree
 * (an uneven world's cheaper rank takes its own copy of it: vram_budget_rank_split below).
 * Takes the SMALLEST capacity and free across ranks and the LARGEST arena, because a split has to
 * fit on every card it is applied to.
 *
 * Refuses, by name and by how much, when the fixed terms alone exceed the card. */
int vram_budget_resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out);

/* The same split worked out on a copy of `cfg`, written nowhere and printed nowhere: for a caller
 * deciding between two programs before it resolves the one it keeps (Engine::plan's table
 * placement). */
int vram_budget_trial(const Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out);

/* AN UNEVEN WORLD'S CHEAPER RANK. The resolver above derives one split, the dearest rank's, and one
 * block table serves every rank -- so a rank whose cache costs less per token (an attention-zero
 * rank keeps no attention cache) would reserve the common KV pool for tokens configure_kv then caps
 * away. Given the common split, the MiB that rank's pool needs to page the SAME tokens, and what
 * every one of its weights would take resident (statics + all experts + their spare slots), this
 * moves the difference to its weights pool, no further than that cap; what the cap leaves stays
 * unclaimed. Nothing changes when the need is not below the common pool. */
struct RankSplit {
    int64_t weights_mib  = 0;
    int64_t kv_mib       = 0;
    int64_t unusable_mib = 0;   /* saved from the KV pool but past the weights cap: never allocated */
    bool    resplit      = false;
};
RankSplit vram_budget_rank_split(int64_t common_weights_mib, int64_t common_kv_mib,
                                 int64_t need_kv_mib, int64_t weights_cap_bytes);

}  /* namespace rad */
