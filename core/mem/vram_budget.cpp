/* vram_budget.cpp -- see vram_budget.h for why a derived budget is not the §6 violation it looks
 * like. This file is the arithmetic and the receipt. */
#include "mem/vram_budget.h"

#include <algorithm>
#include <cmath>

namespace rad {

namespace {
constexpr int64_t kMiB = 1024 * 1024;

/* DOWN, ALWAYS. The pools take MiB and multiply back up, so a budget rounded up is a budget that
 * asks the card for more than the arithmetic said was there -- which fails at the allocation
 * rather than here, with the driver's message instead of ours. */
int64_t to_mib(int64_t bytes) { return bytes > 0 ? bytes / kMiB : 0; }

/* UP, FOR THE ONE BUDGET THAT IS A FLOOR. The KV budget is a CAPACITY -- truncated, the cache
 * addresses a few fewer tokens and nothing else changes. The weights budget is `statics +
 * experts`, a figure the pool must COVER, and a byte short is a tensor with nowhere to land. An
 * expert share pads the number enough to hide that; a dense model is budgeted its static
 * footprint to the byte, and there the rounding direction decides whether it loads. The MiB this
 * claims comes out of the headroom, which is what the headroom is for. */
int64_t to_mib_up(int64_t bytes) { return bytes > 0 ? (bytes + kMiB - 1) / kMiB : 0; }

/* WHAT THE DRIVER'S OWN HOLDING MOVES BY AFTER IT IS READ. `already_held` is one reading, and on
 * ROCm 7.2 the same server started twice has read 4 MiB less than it later held (and once 30 MiB
 * more): a budget that split the difference away came up a few MiB short of the headroom the
 * first launches of kernels allocate from. Four times the drift seen, and charged before anything
 * is split. */
constexpr int64_t kDriverDrift = 16 * kMiB;
}  /* namespace */

/* WHAT THE CACHE BACKS, counted on its PAGED part. The ceiling is the recurrent and conv state of
 * max_seqs sequences plus their context, and the state is spent before a single token is cached --
 * on a hybrid model at a few dozen sequences it is gigabytes a card. A ratio over the whole
 * ceiling counts that state as context and reports full-length sessions the cache cannot hold. */
static double paged_share(const VramBudget& b) {
    const double ctx = (double)(b.kv_ceiling - b.kv_fixed);
    return ctx > 0 ? std::max(0.0, (double)(b.kv - b.kv_fixed)) / ctx : 0.0;
}

/* The resolver, with its log lines behind `log`: vram_budget_trial asks the same question
 * of a configuration it may not use, and a refusal printed for a split nobody takes reads as the
 * startup failing. */
static int resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out, bool log) {
    VramBudget b;
    if (per_rank.empty()) { if (out) *out = b; return RAD_OK; }

    /* THE SMALLEST CARD AND THE LARGEST PROGRAM, because one split is derived for every rank. Ranks
     * of one deployment normally hold the same shard shapes and the same program, so these folds
     * are no-ops -- they exist so that a mixed pair degrades to the card that binds instead of
     * overcommitting the smaller one on the first allocation. An uneven world's cheaper ranks are
     * re-split from this answer afterwards (vram_budget_rank_split, called by Engine::plan). */
    b.capacity   = per_rank[0].capacity;
    int64_t freeb = per_rank[0].free;
    int64_t layer_max = 0, kv_one = 0;
    for (const VramFacts& f : per_rank) {
        b.capacity   = std::min(b.capacity, f.capacity);
        freeb        = std::min(freeb, f.free);
        b.arena      = std::max(b.arena, f.arena);
        b.staging    = std::max(b.staging, f.staging);
        if ((int64_t)f.n_queues * f.queue_bytes > b.queues) {
            b.queues      = (int64_t)f.n_queues * f.queue_bytes;
            b.n_queues    = f.n_queues;
            b.queue_bytes = f.queue_bytes;
        }
        b.kv_ceiling = std::max(b.kv_ceiling, f.kv_ceiling);
        b.kv_fixed   = std::max(b.kv_fixed, f.kv_fixed);
        b.statics    = std::max(b.statics, f.statics);
        b.expert_ceiling = std::max(b.expert_ceiling, f.experts);
        layer_max        = std::max(layer_max, f.expert_layer_max);
        /* THE RANK THAT NEEDS MOST ON THE CARD SETS THE FLOOR. */
        b.expert_floor = std::max(b.expert_floor, f.expert_floor);
        kv_one = std::max(kv_one, f.kv_one);
    }
    b.lends = true;
    for (const VramFacts& f : per_rank) b.lends = b.lends && f.lends;
    b.headroom = std::max<int64_t>(cfg.gpu_headroom_mib, 0) * kMiB;
    int64_t host_reserve = 0;
    for (const VramFacts& f : per_rank) host_reserve = std::max(host_reserve, f.host_reserve);
    const bool shared = host_reserve > b.headroom;
    if (shared) b.headroom = host_reserve;

    /* A host backend has no VRAM to speak of and reports none; there is nothing here to resolve
     * and inventing a capacity would be exactly the inference §6 forbids. */
    if (b.capacity <= 0) {
        b.weights = cfg.vram_weights_mib * kMiB;
        b.kv      = cfg.vram_kv_mib * kMiB;
        if (out) *out = b;
        return RAD_OK;
    }

    /* MEASURED, NOT ASSUMED, AND IT IS THE ONLY TERM THAT IS. Everything the driver and the loaded
     * code objects already hold shows up here, including a desktop compositor on the same card --
     * which is a real configuration on a workstation and the reason this is a line in the report
     * rather than a constant folded into the headroom. Read after declare(), so every kernel
     * plugin's init has run and its code object is resident. */
    b.already_held = std::max<int64_t>(b.capacity - freeb, 0);

    const bool want_w  = cfg.vram_weights_mib <= 0;
    const bool want_kv = cfg.vram_kv_mib <= 0;
    b.derived_weights = want_w;
    b.derived_kv      = want_kv;

    /* What is left after the terms nobody gets to spend. */
    b.drift = (want_w || want_kv) ? kDriverDrift : 0;
    int64_t fixed = b.already_held + b.drift + b.headroom + b.arena + b.staging + b.queues;
    if (!want_w)  fixed += cfg.vram_weights_mib * kMiB;   /* a stated pool is fixed too */
    if (!want_kv) fixed += cfg.vram_kv_mib * kMiB;
    b.claimable = b.capacity - fixed;

    if (b.claimable < 0) {
        b.report = fmt(
            "VRAM does not cover the fixed costs before a single pool is sized:\n"
            "  card total         %10s\n"
            "  already held       %10s   (driver context, code objects, anything else on the card)\n"
            "  driver drift       %10s   (what that moves by after it is read)\n"
            "  --gpu-headroom-mib %10s\n"
            "  activation arena   %10s   (buffer plan + kernel scratch)\n"
            "  step staging       %10s   (block tables, media and KV tier staging)\n"
            "  hardware queues    %10s   (%d at %s each, measured)\n"
            "%s%s"
            "  ------------------------------\n"
            "  SHORT BY           %10s\n"
            "Lower --gpu-headroom-mib, lower --max-num-batched-tokens (the arena scales with it), "
            "or free the card.\n",
            humanb(b.capacity).c_str(), humanb(b.already_held).c_str(), humanb(b.drift).c_str(),
            humanb(b.headroom).c_str(), humanb(b.arena).c_str(), humanb(b.staging).c_str(),
            humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(),
            want_w  ? "" : fmt("  --vram-weights-mib %10s   (stated)\n",
                               humanb(cfg.vram_weights_mib * kMiB).c_str()).c_str(),
            want_kv ? "" : fmt("  --vram-kv-mib      %10s   (stated)\n",
                               humanb(cfg.vram_kv_mib * kMiB).c_str()).c_str(),
            humanb(-b.claimable).c_str());
        if (log) RAD_ERR("%s", b.report.c_str());
        if (out) *out = b;
        return RAD_E_NOMEM;
    }

    b.weights = want_w  ? 0 : cfg.vram_weights_mib * kMiB;
    b.kv      = want_kv ? 0 : cfg.vram_kv_mib * kMiB;

    /* ---- the static weights come off the top ------------------------------------------------
     *
     * They are a floor and not a share. Only a DERIVED weight budget pays them here: an operator
     * who stated --vram-weights-mib has already decided what the weights get, and second-guessing
     * that is the re-optimisation §6 forbids. */
    if (want_w) {
        b.statics = std::min(b.statics, b.claimable);
        if (b.statics >= b.claimable && b.claimable > 0) {
            /* Not fatal: the planner will demote what does not fit and the model still serves,
             * badly. Name it, because "why is this model 8x slower than it should be" has no other
             * answer at startup. */
            if (log) RAD_WARN("the static (non-expert) weights alone want %s of the %s this card has to "
                     "give. Every expert will stream from host memory and the KV cache has "
                     "nothing left -- this model wants more VRAM, fewer ranks of it, or a smaller "
                     "--max-num-batched-tokens.",
                     humanb(b.statics).c_str(), humanb(b.claimable).c_str());
        }
    }
    b.elastic = want_w ? b.claimable - b.statics : b.claimable;
    if (b.elastic < 0) b.elastic = 0;

    /* ---- and only what is left is divided --------------------------------------------------- */
    if (want_w && want_kv) {
        /* An unset cache ceiling (plan_groups failed) bounds nothing; zero experts is a ceiling of
         * zero, which weight_footprint always measures. */
        const int64_t kv_cap = b.kv_ceiling > 0 ? b.kv_ceiling : b.elastic;
        int64_t kv = 0, exp = 0;
        if (kv_cap + b.expert_ceiling <= b.elastic) {
            /* BOTH FIT: nothing trades. */
            kv  = kv_cap;
            exp = b.expert_ceiling;
        } else if (b.lends) {
            /* THE CACHE LENDS, so it is sized for what live requests could need and gives back
             * whatever they do not. The experts keep what the host pool cannot hold. */
            exp = std::min(b.expert_floor, b.elastic);
            kv  = std::min(kv_cap, std::max(b.elastic - exp, kv_one));
            exp = std::min(std::max(b.expert_ceiling, b.expert_floor), b.elastic - kv);
            /* THE ONE SEQUENCE WINS, and the experts it pushes off the card have no tier: the
             * planner refuses them by name. Said here too, with what moves it. */
            if (log && exp < b.expert_floor)
                RAD_WARN("one --max-model-len sequence's cache (%s) leaves the experts %s, short of "
                         "the %s the pinned host pool cannot hold: raise --host-pool-mib or lower "
                         "--max-model-len",
                         humanb(kv_one).c_str(), humanb(exp).c_str(),
                         humanb(b.expert_floor).c_str());
        } else {
            /* NO LENDER: a block the cache does not use would sit empty, so the experts come
             * first and the cache keeps one --max-model-len sequence. */
            kv  = std::min(kv_cap, std::max(b.elastic - b.expert_ceiling, kv_one));
            exp = std::min(b.expert_ceiling, b.elastic - kv);
        }
        if (kv > b.elastic) kv = b.elastic;
        if (exp < 0) exp = 0;
        b.experts  = exp;
        b.kv       = kv;
        b.weights  = b.statics + exp;
        b.unusable = std::max<int64_t>(b.elastic - kv - exp, 0);
    } else if (want_kv) {
        b.kv = b.kv_ceiling > 0 ? std::min(b.claimable, b.kv_ceiling) : b.claimable;
        b.unusable = b.claimable - b.kv;
    } else if (want_w) {
        /* The KV pool was stated, so `claimable` is already net of it; the weights take the rest,
         * still capped at what the model has to put there. */
        int64_t exp = b.elastic;
        if (exp > b.expert_ceiling) {
            b.unusable = exp - b.expert_ceiling;
            exp = b.expert_ceiling;
        }
        b.experts = exp;
        b.weights = b.statics + exp;
    }

    /* ---- THE PREFILL STAGING REGION, out of the expert share ---------------------------------
     *
     * A prefill chunk routes to nearly every expert of every layer, so the part of each layer the
     * budget leaves off the card is read across the link by the layer's GEMM -- on the critical
     * path, at the link's rate, while the rest of the layer's work leaves the link idle. Two
     * buffers of one layer's share of that let the next layer's copy run behind the current one
     * (core/place/stager.h). They come out of the expert share because they exist only for the
     * experts that do not fit, and they sit at the top of the activation arena, which every step
     * smaller than the full plan lends back to the expert slab: a decode step keeps the residency
     * they cost.
     *
     * The share is the budget's -- what the slab alone leaves off -- which is the most a step can
     * see: the loans the KV cache and the smaller arena levels make only ever add residency. A
     * layer whose pooled part is larger than a buffer stages what fits and streams the rest. */
    if (want_w && layer_max > 0 && b.expert_ceiling > b.experts && cfg.placement != "all_vram" &&
        cfg.placement != "layer_offload") {
        const double f = (double)(b.expert_ceiling - b.experts) / (double)b.expert_ceiling;
        const int64_t one = ((int64_t)std::ceil(f * (double)layer_max) + RAD_ALIGN_UNIT - 1) /
                            RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        if (b.lends && want_kv) {
            /* A LENDING CACHE PAYS FOR IT: the experts are at their floor and a byte less would
             * leave one with no tier, while the region is lent to the slab whenever the steps are
             * small -- the same memory the cache would have lent. */
            if (b.kv - 2 * one >= kv_one) {
                b.stage = 2 * one;
                b.kv   -= b.stage;
            }
        } else if (2 * one < b.experts) {
            b.stage    = 2 * one;
            b.experts -= b.stage;
            b.weights  = b.statics + b.experts;
        }
    }

    /* HOW MUCH OF THE ADVERTISED CONTEXT THE POOL BACKS, AND IT IS A PROVISIONING CHOICE RATHER
     * THAN A SHORTFALL.
     *
     * A KV share under max_seqs x max_ctx is not a warning. The argument that it would be -- that
     * the server accepts at the door what it cannot serve -- is wrong for a PAGED cache, and the
     * difference is the whole of vLLM's block manager: a sequence holds only the blocks its tokens
     * actually occupy (KVManager::ensure grows it by ceil_div(n_tokens, block_size) as it runs,
     * drops them at completion, and returns RAD_E_FULL for the scheduler to preempt rather than
     * failing the request). Every sequence reaching max_ctx at the same moment is the worst case,
     * not the workload -- real loads run well under half backing -- and sizing for the worst case
     * spends VRAM that a model whose experts do not fit would rather have for them.
     *
     * So it is reported and not warned about. What IS still checked, and belongs in the KV manager
     * because it is a real limit rather than a ratio, is that ONE max-length sequence fits: a
     * context no single request can complete is a configuration error however much slack the pool
     * has (core/engine_bringup.cpp, configure_kv). */
    if (log && b.derived_kv && b.kv_ceiling > 0 && cfg.max_ctx > 0)
        RAD_INFO("kv: the pool backs %.0f%% of the worst case (%lld sequences all at %lld tokens), "
                 "%.1f full-length sessions. That is concurrency at length before the scheduler "
                 "preempts, not a bound on any one request. WITH PREFIX CACHING a finished "
                 "sequence's blocks stay held so the next turn can reuse them, so the demand is "
                 "live SESSIONS and not running requests -- when they do not all fit, the cache "
                 "evicts prefixes the next turn wanted and that turn re-prefills its whole "
                 "context. --host-pool-mib moves it: the experts the host pool cannot hold are the "
                 "part of the card the cache cannot have.",
                 100.0 * paged_share(b),
                 (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                 (double)cfg.max_seqs * paged_share(b));
    /* Rounding the floor up claims up to a MiB the split did not hand out. Take it back from the
     * cache where the cache is ours to size -- a paged pool a MiB smaller addresses a handful
     * fewer tokens and nothing else -- so the headroom stays whole. Where the KV budget was
     * STATED it is not ours to shave and the MiB comes out of the headroom instead. */
    cfg.vram_budget_derived = want_w || want_kv;
    cfg.vram_weights_mib = to_mib_up(b.weights);
    const int64_t w_over = cfg.vram_weights_mib * kMiB - b.weights;
    cfg.vram_kv_mib      = want_kv ? to_mib(std::max<int64_t>(b.kv - w_over, 0)) : to_mib(b.kv);

    /* THE RECEIPT. Every term, in the order it was spent, so that a split anybody disagrees with
     * can be argued about against numbers instead of re-derived from sysfs by hand. The unclaimed
     * line is not padding: to_mib truncates, and what it drops is real VRAM that ends up beside
     * the headroom. */
    const int64_t granted   = cfg.vram_weights_mib * kMiB + cfg.vram_kv_mib * kMiB;
    const int64_t unclaimed = b.capacity - b.already_held - b.drift - b.arena - b.staging - b.queues -
                              b.stage - granted;
    const double  ela       = b.elastic > 0 ? (double)b.elastic : 1.0;
    b.report = fmt(
        "VRAM budget, per card:\n"
        "  card total         %10s\n"
        "  already held       %10s   measured: driver context, code objects, anything else resident\n"
        "  driver drift       %10s   what that moves by after it is read: kept free\n"
        "  activation arena   %10s   computed from the buffer plan\n"
        "  step staging       %10s   the step batch's block tables and media staging, and the\n"
        "                                  KV tiers' copy staging\n"
        "  hardware queues    %10s   %d a card at %s each, measured: the engine's streams\n"
        "  %-18s %10s   %s\n"
        "  ------------------------------\n"
        "  claimable          %10s\n"
        "  static weights     %10s   %s\n"
        "  ------------------------------\n"
        "  elastic            %10s   %s\n"
        "    experts          %10s   %.0f%% of it%s\n"
        "    kv cache         %10s   %.0f%% of it%s\n"
        "%s"
        "%s"
        "  ==============================\n"
        "  vram_weights       %10s   %s\n"
        "  vram_kv            %10s   %s\n"
        "  unclaimed          %10s   the headroom, the unusable above, and what MiB rounding\n"
        "                                  could not place\n",
        humanb(b.capacity).c_str(), humanb(b.already_held).c_str(), humanb(b.drift).c_str(),
        humanb(b.arena).c_str(), humanb(b.staging).c_str(),
        humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(),
        shared ? "left to the host" : "--gpu-headroom-mib", humanb(b.headroom).c_str(),
        shared ? "integrated graphics: this memory is the host's; --vram-kv-mib states a pool "
                 "outright"
               : "left unclaimed on purpose",
        humanb(b.claimable).c_str(),
        humanb(b.statics).c_str(),
        b.derived_weights ? "off the top: every rank's own shard, and not a share of anything"
                          : "(not charged here -- --vram-weights-mib was stated)",
        humanb(b.elastic).c_str(),
        b.lends ? "the experts keep what the host pool cannot hold; the cache takes the rest\n"
                  "                                  and lends what no live request holds to the "
                  "expert slab"
                : "no lender: the experts first, the cache at least one full-length sequence",
        humanb(b.experts).c_str(), 100.0 * (double)b.experts / ela,
        b.expert_ceiling == 0
            ? " -- a dense model, so there is no expert plane for this share to fill and the\n"
              "                                  cache took everything it could address"
        : (b.experts >= b.expert_ceiling)
            ? " -- the whole expert plane is resident, so more would do nothing" : "",
        humanb(b.kv).c_str(), 100.0 * (double)b.kv / ela,
        (b.kv_ceiling > 0 && b.kv >= b.kv_ceiling)
            ? fmt(" -- backs %lld sequences all at %lld tokens at once, which is the worst case and\n"
                  "                                  more would be unusable",
                  (long long)cfg.max_seqs, (long long)cfg.max_ctx).c_str()
            : (b.kv_ceiling > 0
                   /* SAID AS A SESSION COUNT, because that is the number to compare against
                    * --max-num-seqs, and "28%" makes the reader do the arithmetic.
                    *
                    * AND IT NAMES SESSIONS, NOT RUNNING REQUESTS. "Paged blocks are held only
                    * while a sequence occupies them" stops being true once prefix caching takes a
                    * KV reference on every block it indexes: a finished sequence's blocks stay
                    * held so the next turn can reuse them. So the demand of a multi-turn workload
                    * is the number of LIVE SESSIONS, and counting running requests instead
                    * under-counts by exactly the thing that makes agent turns cheap. When the pool
                    * cannot hold them all, the cache evicts prefixes the next turn wants and that
                    * turn re-prefills its whole context (radiance:prefix_cache_evictions_total). */
                   ? fmt(" -- %.0f%% backing of the %lld x %lld worst case: %.1f full-length\n"
                         "                                  sessions. With prefix caching a "
                         "finished sequence's blocks stay held\n"
                         "                                  for reuse, so the demand is LIVE "
                         "SESSIONS and not running requests",
                         100.0 * paged_share(b),
                         (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                         (double)cfg.max_seqs * paged_share(b)).c_str()
                   : ""),
        b.unusable > 0
            ? fmt("    unusable         %10s   neither side could take it: both are at their "
                  "ceiling\n", humanb(b.unusable).c_str()).c_str()
            : "",
        b.stage > 0
            ? fmt("    prefill staging  %10s   out of the %s: two buffers of one layer's "
                  "non-resident experts,\n"
                  "                                  at the arena's top and lent back to the "
                  "slab while steps are small\n", humanb(b.stage).c_str(),
                  b.lends ? "cache" : "experts").c_str()
            : "",
        humanb(cfg.vram_weights_mib * kMiB).c_str(),
        b.derived_weights ? "static + experts" : "stated by --vram-weights-mib",
        humanb(cfg.vram_kv_mib * kMiB).c_str(),
        b.derived_kv ? "the cache's share" : "stated by --vram-kv-mib",
        humanb(unclaimed).c_str());

    if (out) *out = b;
    return RAD_OK;
}

int vram_budget_resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out) {
    return resolve(cfg, per_rank, out, true);
}

int vram_budget_trial(const Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out) {
    Config c = cfg;
    return resolve(c, per_rank, out, false);
}

RankSplit vram_budget_rank_split(int64_t common_weights_mib, int64_t common_kv_mib,
                                 int64_t need_kv_mib, int64_t weights_cap_bytes) {
    RankSplit s;
    s.weights_mib = common_weights_mib;
    s.kv_mib      = common_kv_mib;
    if (need_kv_mib < 0 || need_kv_mib >= common_kv_mib) return s;
    const int64_t saved = common_kv_mib - need_kv_mib;
    /* NEVER LESS THAN THE COMMON WEIGHTS POOL, and never more than every weight resident: a weights
     * pool is one whole allocation, so MiB past what this rank could ever place are left unclaimed
     * rather than allocated empty. */
    const int64_t cap   = std::max(common_weights_mib, to_mib_up(weights_cap_bytes));
    const int64_t grown = std::min(common_weights_mib + saved, cap);
    s.kv_mib       = need_kv_mib;
    s.weights_mib  = grown;
    s.unusable_mib = saved - (grown - common_weights_mib);
    s.resplit      = true;
    return s;
}

}  /* namespace rad */
