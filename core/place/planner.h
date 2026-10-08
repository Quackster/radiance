/* planner.h -- the placement planner. Runs ONCE, after declare and before load, and assigns every
 * weight a tier and an execution site against explicit pool budgets (spec §5.3).
 *
 * ---- THE ONE IDEA IN THIS DIRECTORY ------------------------------------------------------------
 *
 * All-in-VRAM, layer offload to RAM/SSD, and expert offload to RAM/SSD are ONE MECHANISM AT THREE
 * GRANULARITIES. What separates them is not the tiering -- the tiers are the same four -- but
 * whether the access is PREDICTABLE, and that distinction is the shape of this code:
 *
 *   LAYER OFFLOAD IS A SCHEDULING PROBLEM. Declare already enumerated every op with its weight
 *   operands in order, so WeightInfo::first_use_op is the entire future access sequence, known
 *   statically. The planner emits an exact prefetch schedule off it (Plan::prefetch). No policy,
 *   no predictor, nothing to be wrong about.
 *
 *   EXPERT OFFLOAD IS A PREDICTION PROBLEM. Routing is data-dependent, so the only thing the
 *   planner can do is rank an initial residency and hand the rest to heat.h. This is why expert
 *   offload needs a heat engine and layer offload (llama.cpp's) does not.
 *
 *   ALL-IN-VRAM IS THE DEGENERATE CASE where every weight lands on tier 0 and the mover never
 *   runs. It is not a separate path; it is this one with a budget that fits.
 *
 * ---- TIER AND SITE ARE INDEPENDENT AXES (spec §5.1) --------------------------------------------
 *
 * Tier is where the bytes live. Site is where the arithmetic happens. All four combinations are
 * core behaviour:
 *
 *   VRAM       + Device          vLLM
 *   host/SSD   + DeviceStaged    staged into a slab slot, exact or predicted
 *   host       + DeviceZeroCopy  the GEMM streams the weight over the link as it computes; no
 *                                slab slot, so a read-once weight costs no residency at all
 *   host       + Host            llama.cpp -ngl
 *
 * Host-site execution needs a RAD_DOMAIN_HOST kernel. Where declare resolved none, that site is
 * UNAVAILABLE for that op, and the planner is told so by reading Band::dom[RAD_DOMAIN_HOST]
 * rather than by being passed a flag -- the resolution is the fact.
 */
#pragma once
#include "../rad_core.h"
#include "rad_format.h"

#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ strategy */
enum class Strategy { AllVram = 0, LayerOffload = 1, ExpertTiered = 2, Auto = 3 };

const char* strategy_name(Strategy s);
/* RAD_E_INVAL and *out untouched if the string is not one of the four. */
int         strategy_parse(const char* s, Strategy* out);

/* ------------------------------------------------------------------ budgets */
/* Every pool is budgeted explicitly. No profiling pass, no utilisation fraction, no inference
 * about what the workload will be (spec §6). The one thing a zero means is stated here rather
 * than guessed at each use: zero vram_weights is "no cap", which is answerable only under
 * all_vram -- any offload strategy needs a number to spill against and says so by name.
 */
struct Budgets {
    int64_t vram_weights = 0;    /* bytes; 0 = uncapped, legal only when everything fits */
    int64_t host_pool = 0;       /* bytes; the pinned host tier. 0 = no host tier */
    bool    ssd = false;         /* --weights-disk-tier was given */

    static Budgets from(const Config& c);
};

/* ------------------------------------------------------------------ the movement unit */
/* THE UNIT OF PLACEMENT IS A GROUP, NOT A TENSOR. An expert's gate/up/down share one
 * RadWeightGroup because a dispatch holding two of the three has nothing it can compute; a
 * layer's weights share one because layer offload is a contiguous run rather than a scatter
 * (spec §4.1). Everything below -- the slab's fixed stride, the mover's copy, heat's ranking --
 * is per unit, and a weight's placement is its unit's placement.
 *
 * UNDER TENSOR PARALLEL A UNIT IS THIS RANK'S SHARD OF THE GROUP. The plugin declares dimensions
 * already divided (spec §9), so `bytes` is already per-rank and the planner budgets against this
 * card. What the planner still owes TP is the harder half, and it is an invariant rather than an
 * arithmetic: THE RESIDENT SET MUST BE IDENTICAL ON EVERY RANK. Each rank holds its own half of
 * every resident expert, so a slab slot only means the same thing everywhere if every rank
 * decided the same experts were in it. That is why the ranking below is keyed on the container's
 * profile and the unit's identity and on nothing rank-local -- no free-VRAM query, no timing, no
 * device enumeration. The invariant is a property of how the ranking is computed rather than a
 * condition checked before placement may run.
 */
struct MoveUnit {
    int32_t layer = -1;          /* RadWeightGroup::layer; -1 for a model-level weight */
    int32_t expert = -1;         /* RadWeightGroup::expert; -1 for a non-expert group */
    int32_t slab_class = -1;     /* index into Plan::classes */
    int64_t bytes = 0;           /* THIS RANK's shard of the whole group */
    int32_t first_use_op = -1;   /* min over the group; the prefetch schedule keys on it */
    int32_t last_use_op = -1;
    int     access = RAD_ACCESS_PER_TOKEN;   /* the group's strongest access class */
    float   profile_share = 0.0f;            /* the container's popularity profile, or 0 */

    Tier    tier = Tier::VRAM;
    Site    site = Site::Device;
    int32_t slab_slot = -1;
    bool    movable = false;     /* may the heat engine relocate it after load */
    bool    host_forced = false; /* --embedding-placement ram put it in the pinned pool */

    /* PLACED BEFORE ANY STRATEGY RAN, so every assignment pass skips it and no VRAM total counts
     * it: the mapped tier (no device kernel can read it) and --embedding-placement ram. */
    bool preplaced() const { return tier == Tier::Mapped || host_forced; }
    const char* reason = "fits in vram";   /* a fixed string, so the report never guesses */

    /* The group's INTERNAL LAYOUT, computed once here so the mover never has to. `offset[i]` is
     * where weights[i] sits inside the unit, honouring that weight's declared alignment, and
     * `bytes` is the padded total. One definition, because the mover's copy size, the pool offset
     * it reads and the pointer it publishes must agree and they are written in three places. */
    std::vector<rad_weight> weights;
    std::vector<int64_t>    offset;

    /* WHAT THE UNIT ACTUALLY COSTS A POOL, which is its class's slot stride and not `bytes`. The
     * planner budgets against this and the mover allocates this, so the two cannot drift -- and
     * the direction they would drift in is an overrun of the weights pool. `bytes` is
     * still what the report attributes to the weights, because padding is not a weight. */
    int64_t slot_bytes = 0;

    /* Every weight of it lies in the container exactly as its kernel reads it (weight_file_direct),
     * so the file tier can stage it with a read. A unit that is not is never placed there. */
    bool    file_direct = true;
};

/* The strongest access class in a group decides the group's. Strongest means "read most surely":
 * a group holding one PER_TOKEN weight is read on every forward pass whatever the rest of it is,
 * and calling it conditional would let the heat engine demote something the next token needs.
 * VOCAB ranks with PER_TOKEN -- lm_head and the embedding table are read every token; what makes
 * them their own class is their size and the gather, not their certainty (spec §19.2). */
int access_strength(int access);

/* ------------------------------------------------------------------ the slab class */
/* Fixed-stride VRAM slab slots per weight class (spec §5.4). One stride per class, so a slot is
 * interchangeable within its class and the mover's bookkeeping is an integer rather than an
 * allocator. The stride is the class's largest unit rounded to the movement-unit alignment; the
 * padding that costs is the price of never fragmenting, and a fragmenting slab is a slab whose
 * hit rate depends on the order promotions happened to arrive in.
 */
struct SlabClass {
    std::string name;            /* "expert", "layer", "model" plus the byte size */
    int64_t     stride = 0;      /* bytes per slot, aligned */
    int64_t     n_units = 0;     /* units of this class in the model */
    int64_t     n_slots = 0;     /* resident slots the planner bought in VRAM */
    int64_t     n_shadow = 0;    /* spare slots for in-flight promotions; see mover.h */
    int64_t     n_stage = 0;     /* the staging ring; see RAD_PLACE_STAGE_RING */
    bool        movable = false;
};

/* THE STAGING RING. A DeviceStaged unit lives off the device and is copied into a slab slot
 * before its op runs, so a class holding staged units needs slots that belong to no unit. Two is
 * double buffering -- one slot being read by the op that is running, one landing for the op
 * after it -- and it is the minimum depth that overlaps anything at all.
 *
 * More is better hiding at more VRAM, and where the curve flattens is a measurement rather than a
 * principle; it is not a Config field because Config is frozen and this is the honest
 * default rather than a tuned one. If it becomes worth tuning it wants a --stage-ring flag beside
 * the other explicit budgets in spec §6, NOT a heuristic derived from free VRAM. */
#define RAD_PLACE_STAGE_RING 2

/* THE FILE TIER'S STAGING POOL, IN PINNED HOST SLOTS. A unit read from the container lands in a
 * pinned buffer the mover can DMA out of without a bounce (spec §5.4), and those buffers come out
 * of --host-pool-mib like everything else on the host side.
 *
 * It is defined HERE, beside the VRAM ring, because the planner has to charge the operator for it
 * and the mover has to allocate exactly what was charged. Were it a mover-side default instead,
 * the planner would fill the host pool with pinned units and the mover would then ask the
 * exhausted pool for these slots and refuse the plan -- the same shape as the slab overhead on
 * the VRAM side, and the same rule: whoever budgets a thing owns its number. */
#define RAD_PLACE_FILE_STAGE_SLOTS 4

/* THE WINDOWS A ROUTED LAYER IS READ THROUGH, when routed experts land on the file tier. A layer's
 * file-tier experts are read as long sequential runs of the container -- a few gigabytes a layer
 * for a bf16 model -- so the read does not go a unit at a time into the slots above: it fills one
 * pinned window while the window before it drains to the card, and the host waits on a window
 * only when it comes round again. Four of 32 MiB keep a drive at its sequential rate (a pread of
 * this size is ~5 ms at 7 GB/s, the copy out of it under 2) for 128 MiB of pinned memory, charged
 * out of --host-pool-mib like the slots. */
#define RAD_PLACE_FILE_WINDOWS 4
#define RAD_PLACE_FILE_WINDOW_BYTES ((int64_t)32 << 20)

/* SHADOW SLOTS: slab slots that are allocated but hold no unit, so a promotion writes into one
 * and only then publishes and never DMAs over a slot a kernel might still be reading (mover.h).
 *
 * THEY COME OUT OF THE WEIGHTS POOL, like everything else in the slab. Taking them from
 * whatever VRAM is left over after the slabs is not an option: spec §6 budgets every pool
 * explicitly and --gpu-headroom-mib is memory the engine promises NOT to touch. So dynamic
 * placement costs a few resident units, and the plan report prints exactly how many.
 *
 * 32 is far past where the knob stops mattering, and MORE is the wrong end of the problem: extra
 * shadows raise the move rate and run SLOWER, because more copies in flight contend with the
 * zero-copy tier for the same link. The quarter-of-the-slab cap only ever binds on a slab too
 * small for the number to be meaningful. */
#define RAD_PLACE_SHADOW_SLOTS 32

/* ------------------------------------------------------------------ per-weight result */
struct Placement {
    Tier    tier = Tier::VRAM;
    Site    site = Site::Device;
    int32_t slab_slot = -1;
    int32_t unit = -1;
    int64_t bytes = 0;
    /* Why this and not VRAM/device, as a fixed string so the report never has to guess. */
    const char* reason = "";
};

/* ------------------------------------------------------------------ the plan */
class Plan {
public:
    int      rank = 0, world_size = 1;
    Strategy strategy = Strategy::Auto;

    /* False under Config::deterministic. Dynamic placement sets which experts are resident and
     * residency sets the summation order, so identical inputs do not give identical outputs
     * (spec §17). This is that mode as a real property of the plan -- the heat engine reads it
     * and stays off, and the mover reads it and allocates no shadow slots -- rather than a flag
     * that is checked nowhere. */
    bool     dynamic = true;

    std::vector<Placement> weight;         /* by rad_weight; index 0 is the sentinel */
    std::vector<MoveUnit>  units;
    std::vector<int32_t>   unit_of_weight; /* by rad_weight; -1 for the sentinel */
    std::vector<SlabClass> classes;

    /* Host-site work, in EXECUTION ORDER, as runs. A host-site op leaves its activation in host
     * memory, so a host layer between two device layers costs two link crossings; a few hundred
     * KB is nothing and alternating per layer is not. llama.cpp gets this free because -ngl
     * assigns a contiguous suffix. Here it is an explicit objective, and this vector is the
     * evidence: one entry means one run, and `link_crossings` is what the plan costs a step. */
    struct HostRun { int32_t first_layer = 0, last_layer = 0; int64_t bytes = 0; };
    std::vector<HostRun> host_runs;
    int32_t link_crossings = 0;

    /* THE EXACT PREFETCH SCHEDULE. Layer offload is a scheduling problem: this is the whole of
     * the policy, and it is a sorted list rather than a predictor. `at_op` is the op index by
     * which the unit's bytes must have landed, already backed off by the configured lead. */
    struct Prefetch { int32_t at_op = 0; int32_t unit = 0; };
    std::vector<Prefetch> prefetch;

    struct Totals { int64_t bytes[kTierCount] = {0}; };   /* indexed by Tier */
    Totals  totals;
    /* `host_budget` is what is left for WEIGHTS after the file tier's staging pool is taken out of
     * --host-pool-mib, so the number the placement passes spend against is the number they may
     * actually have. `file_stage_reserve` is what was taken, reported so the operator can see
     * where the difference went: the unit slots, plus the read windows when a routed expert can
     * land on the file tier. */
    int64_t vram_budget = 0, host_budget = 0;
    int64_t file_stage_reserve = 0;
    int64_t file_windows = 0;          /* the read windows' part of it */
    /* --embedding-placement ram: the pool slots taken out of --host-pool-mib before any strategy
     * ran, like the file staging above, and reported beside it rather than among the pooled units.
     * `embedding_host_used` is the same tables' weight bytes, the unit the tier totals count in. */
    int64_t embedding_host_bytes = 0;
    int64_t embedding_host_used = 0;
    /* --ngram-placement ram: the n-gram tables the load phase copies into host memory. They stay
     * on the mapped tier as far as every pool is concerned; this is what the copy costs in RAM.
     * `ngram_copy_card` says which copy: pinned memory a device kernel reads in place over the
     * link, or locked memory the host gather reads. */
    int64_t ngram_copy_bytes = 0;
    bool    ngram_copy_card = false;
    /* ROUTED EXPERTS ON THE FILE TIER have no address of their own: a routed layer's are read into
     * one of two VRAM buffers before the layer's first expert op and published there until the next
     * routed layer starts (place/stager.h, FileStager). `file_layer_max` is the most one layer's
     * file-tier units take in a buffer, each at its own size rounded to RAD_ALIGN_UNIT, which is
     * how the stager packs them; `file_stage_vram` is the two buffers, taken out of
     * the weights pool before any expert is made resident. Both 0 when nothing routed is on the
     * file tier. */
    int64_t file_layer_max = 0;
    int64_t file_stage_vram = 0;
    /* VRAM the slab holds that no weight occupies: the shadow slots and the staging ring. It is
     * reported rather than folded into the tier total because it is the price of dynamic
     * placement and of offload respectively, and an operator comparing two runs needs to see
     * which of the two changed. */
    int64_t slab_overhead = 0;

    /* What did not fit. Two lists because there are two kinds of failure and they read
     * differently: `notes` is a whole-plan statement ("the budget is 3 GiB short"), and `unfit`
     * is per unit -- aggregated by class and reason, because a plan that strands nine hundred
     * experts must print one line about it and not nine hundred (spec §16). */
    std::vector<std::string> notes;
    struct Unfit {
        std::string what;    /* the weight class key */
        std::string why;
        int64_t     count = 0;
        int64_t     bytes = 0;
    };
    std::vector<Unfit> unfit;
    bool fits() const { return notes.empty() && unfit.empty(); }

    const MoveUnit* unit_for(rad_weight w) const {
        if (w >= unit_of_weight.size() || unit_of_weight[w] < 0) return nullptr;
        return &units[(size_t)unit_of_weight[w]];
    }

    /* THE PLACEMENT PLAN REPORT (spec §16): every weight's tier and site, the pool totals, and
     * what did not fit. One line per weight CLASS rather than per weight, because a model with
     * eleven thousand experts has eleven thousand weights and three interesting lines. */
    std::string report(const Program& p) const;
};

/* ------------------------------------------------------------------ the planner */
struct PlannerInput {
    const Program* program = nullptr;
    const Config*  config = nullptr;
    int rank = 0;
    int world_size = 1;

    /* The container's expert-popularity profile, derived from the imatrix's activation counts.
     * Null is fine and costs about a point of hit rate. See Planner::rank_units. */
    const RadFileProfile* profile = nullptr;
    int64_t               n_profile = 0;
    /* The weights are in a container, which the file and mapped tiers read in place. A checkpoint
     * served directly is neither grouped by movement unit nor aligned for O_DIRECT, so with this
     * false no unit is file-direct and both tiers are closed (spec §4.3). */
    bool                  container = true;
    /* No card is visible and the device backend is the host's own memory (device_is_host()), so a
     * host kernel can read the weights pool as well as the container's mapping. See
     * mapped_unit(). */
    bool                  host_backend = false;
};

/* DOES A UNIT STAY IN THE CONTAINER'S MAPPING because only a host kernel can read it? On a card,
 * always: a host kernel cannot read VRAM, and the mapping is the one host address its bytes
 * already have. On the host backend the weights pool is host memory too, and a unit whose stored
 * form is not the container's (a widening or a relayout its host kernel asked for at load, or a
 * checkpoint served directly) has no place in the mapping that looks the way its kernel reads it,
 * so there it takes a converted copy in the weights pool like any other weight. One rule, for the
 * planner and the budget alike. */
inline bool mapped_unit(bool host_only, bool direct, bool host_backend) {
    return host_only && (direct || !host_backend);
}

class Planner {
public:
    explicit Planner(const PlannerInput& in) : in_(in) {}

    /* Runs the whole plan. Returns RAD_OK, or a negative status with `out` still filled in far
     * enough to print -- a plan that does not fit is a plan the operator has to read. */
    int plan(Plan* out);

private:
    const PlannerInput& in_;

    /* Per-weight facts derived from declare, indexed by rad_weight. */
    std::vector<uint8_t> host_ok_;     /* every op reading it resolved a RAD_DOMAIN_HOST kernel */
    std::vector<uint8_t> device_ok_;   /* ...and a RAD_DOMAIN_DEVICE one. Cleared means the weight
                                        * can ONLY be read on the host, which is a decision and
                                        * not a preference -- see pin_mapped. */
    bool                 ssd_ = false; /* --weights-disk-tier was given */

    void resolve_sites(const Program& prog);
    void build_units(const Program& prog, Plan* p);
    void pin_mapped(const Program& prog, Plan* p);
    int  pin_embedding(const Program& prog, const Config& cfg, Plan* p) const;
    void build_classes(Plan* p);
    void rank_units(Plan* p, std::vector<int32_t>* order) const;
    int  assign(Plan* p);
    int  assign_all_vram(Plan* p);
    int  assign_layer_offload(Plan* p);
    int  assign_expert_tiered(Plan* p);
    void enforce_op_site_agreement(const Program& prog, Plan* p);
    void schedule_prefetch(Plan* p);
    void finish(const Program& prog, Plan* p);
    /* Fold one stranded unit into the aggregated list. Found-or-appended by (class, reason), so
     * the cost of reporting nine hundred stranded experts is one line and one counter. */
    void note_unfit(Plan* p, const MoveUnit& mu, const char* why) const;
};

/* Bytes of a declared weight as THIS RANK will hold them: the stored size once the layout pass
 * has run, the logical size before it. Exposed because the mover and the report both need the
 * same answer and two spellings of it is how two components disagree about a budget. */
int64_t weight_bytes(const WeightInfo& w);

/* ------------------------------------------------------------------ what the weights would cost
 *
 * ONE RANK'S DECLARED WEIGHTS, SPLIT INTO THE PART THAT IS NOT NEGOTIABLE AND THE PART THAT IS.
 *
 * The VRAM budget needs this BEFORE the planner runs, because the two questions are asked in that
 * order: how much room is there, and only then what goes in it. And the split matters more than
 * the total. A model's static weights -- everything that is not an expert -- have to be resident
 * for the model to run at a sensible speed at all, so they come off the top; the expert plane is
 * the only thing that trades against the KV cache. Dividing the two together would let the cache's
 * side demote the attention projections to host memory to make room for blocks nothing will fill.
 *
 * The three answers are per RANK, and under tensor parallel that is already the even split across
 * devices: declare runs once per rank with every dimension ALREADY DIVIDED (spec §9), so summing
 * this rank's declared weights is summing its shard.
 *
 * IT MUST AGREE WITH THE PLANNER AND THAT IS WHY IT LIVES HERE. The unit grouping, the alignment
 * accumulation and the mapped-tier rule are Planner::build_units', Planner::resolve_sites' and
 * Planner::pin_mapped's respectively; a second copy of any of them somewhere else would drift and
 * the failure would be a budget that overcommits the card by exactly the difference. */
struct WeightFootprint {
    int64_t statics = 0;   /* non-expert and device-readable -- the floor under any budget */
    int64_t experts = 0;   /* the elastic plane: what streams from host when it does not fit */
    int64_t expert_layer_max = 0;   /* the most expert bytes any one routed layer holds */
    int64_t expert_unit_max  = 0;   /* the largest expert unit's slot */
    /* THE SLAB'S SPARE SLOTS beside the residents of an expert-tiered plan, at their most: per
     * expert slab class, the shadow slots and the staging ring and the tail pad
     * (Planner::assign_expert_tiered). They come out of the same weight budget as the residents. */
    int64_t expert_spare     = 0;
    int64_t layer_unit_max   = 0;   /* the widest per-layer unit's slot: the file tier's staging */
    bool    routed           = false;   /* a layer has experts: the file tier's read windows */
    int64_t mapped  = 0;   /* no device kernel can read them; they never enter a pool at all */
    int64_t hosted  = 0;   /* --embedding-placement ram: in the pinned host pool, off the card */
    /* The tables the two placement flags move, wherever the buckets above count them: `embedding`
     * is inside statics or hosted, `ngram` inside statics when a device kernel reads it and it is
     * placed in VRAM, inside mapped otherwise. */
    int64_t embedding = 0;
    int64_t ngram     = 0;
    /* A device kernel reads the n-gram table, so it can be on the card or read over the link from
     * pinned host memory; without one only the host gather reads it, from RAM or the container. */
    bool    ngram_device = false;
    /* A host kernel gathers it, so it can stay in the container (--ngram-placement disk). */
    bool    ngram_host = false;
    int64_t n_units = 0;
};
/* `embedding` is what the token embedding's placement resolved to; only Ram moves it out of the
 * statics. `ngram` is the n-gram table's; only Vram puts it among them, and only where a device
 * kernel reads it. `host_backend` and `container` are PlannerInput's: whether a host kernel can
 * read the weights pool, and whether the weights lie in a container at all (a checkpoint served
 * directly has no mapping a unit could stay in). Stated by every caller, so the budget and the
 * planner cannot count it differently. */
void weight_footprint(const Program& prog, WeightFootprint* out, TablePlacement embedding,
                      TablePlacement ngram, bool host_backend, bool container);

/* THE PINNED STAGING THE FILE TIER TAKES OUT OF --host-pool-mib before any unit is placed in it:
 * RAD_PLACE_FILE_STAGE_SLOTS of the widest per-layer unit, and the read windows when a routed
 * layer can stream from the container. Planner::plan spends it; the budget counts it the same. */
int64_t file_stage_reserve(int64_t widest, bool routed);

/* THE EXPERT BYTES THE CARD MUST HOLD when `host_room` bytes of the pinned host pool are free for
 * experts: everything the pool cannot take, the slab's spare slots beside them, and one unit of
 * first-fit slack on each side -- the resident fill and the pool fill each skip a unit that does
 * not fit and can leave up to one unit's room unspent. At most every expert and the spare slots.
 * A weight budget of the statics and this much places every expert in VRAM or the host pool. */
int64_t expert_vram_floor(const WeightFootprint& wf, int64_t host_room);

/* THE PLAN AND ITS RANK'S WEIGHTS POOL READ ONE SPLIT. The planner places against
 * Plan::vram_budget and the mover fills the pool Pools::configure allocated, each from the Config
 * it was handed; a rank given two different Configs plans a slab the pool does not have, or leaves
 * part of the pool unplanned and empty, and nothing downstream would say so. RAD_E_STATE, naming
 * the rank and both sizes, when they differ. */
int plan_check_weights_pool(const Plan& p, int64_t pool_bytes);

/* THE TABLES THE PLACEMENT FLAGS MOVE, by what reads them and not by name: a model-level weight
 * that at least one op reads and that only `embed_lookup` reads is a token embedding table, and
 * one only `embed_lookup_q` reads is an n-gram table. A lookup reads M rows a step, which is what
 * makes a tier below the card usable for it at all; a weight a GEMM also reads -- an output head
 * tied to the embedding -- would stream whole over the link every step, and is neither. One
 * definition, because the budget resolver and the planner must pick the same weights. */
bool read_only_by(const Program& prog, rad_weight w, const char* op);
inline bool is_embedding_table(const Program& prog, rad_weight w) {
    return read_only_by(prog, w, "embed_lookup");
}
inline bool is_ngram_table(const Program& prog, rad_weight w) {
    return read_only_by(prog, w, "embed_lookup_q");
}

/* "blk.12.ffn_gate_exps.3.weight" -> "blk.*.ffn_gate_exps.*.weight". The report's grouping key:
 * one line per weight class rather than per weight (spec §16). */
std::string weight_class_key(const std::string& name);

}  /* namespace rad */
