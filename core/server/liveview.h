/* core/server/liveview.h -- the live view (spec §16).
 *
 * Per-card utilisation and VRAM, throughput, draft acceptance, and the expert plane coloured by
 * tier. It is a terminal view and nothing else: no web page, no port, no JSON.
 *
 * The rule that shapes the implementation is the last sentence of §16 -- piped or redirected it
 * prints ordinary lines instead. So there are two renderers, chosen once by isatty(), and the
 * non-interactive one emits no escape sequence at all. A log file full of cursor-home sequences
 * is a log file you cannot grep, and the failure is silent: it looks fine on the terminal where
 * it was written.
 *
 * The view owns no data. A snapshot function supplied by the engine fills a LiveSnapshot; this
 * component only draws it, which keeps the server free of any device dependency.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace server {

struct SchedMetrics;

struct CardStat {
    int         index = 0;
    std::string name;
    double      util = 0.0;        /* 0..1 */
    int64_t     vram_used = 0;
    int64_t     vram_total = 0;
    double      power_w = 0.0;     /* 0 = not sampled */
    double      temp_c = 0.0;      /* 0 = not sampled */
};

/* rad::kTierCount, restated. This header deliberately does not include the program description --
 * a tier arrives here as a byte -- so the size is written out and liveview.cpp static_asserts the
 * two against each other. A fifth tier that fits in four slots is an out-of-bounds write that
 * costs nothing until it costs everything. */
enum { kLiveTiers = 5 };

/* THE EXPERT PLANE, and what the placement engine has been doing to it.
 *
 * The plane itself -- one cell per (layer, expert), valued by rad::Tier -- is the picture. The
 * counters beside it are the things the picture cannot say. Residency per tier says how much of
 * the model is actually in VRAM; the per-layer counts say whether that residency is SPREAD, since
 * a fill that walks the model in order can starve the last layers and the draft head while the
 * global count still looks healthy; and the move counters say whether the traffic on the link is
 * the engine discovering a working set or re-admitting what it just evicted.
 *
 * Counted ONCE, here, rather than by each renderer: the terminal view and the web page are two
 * drawings of the same snapshot and two independent tallies are two chances to disagree.
 */
struct ExpertPlane {
    int                  n_layers = 0;
    int                  n_experts = 0;
    std::vector<uint8_t> tier;     /* row-major, size n_layers * n_experts */

    /* Residency, by rad::Tier. Sized by kTierCount and not by 4: folding an unknown tier into the
     * SSD bucket would report a weight as on disk when it is not. `bytes` is this rank's shard,
     * which is the number that has to add up against the VRAM bar beside it. */
    int64_t units[kLiveTiers] = {0};
    int64_t bytes[kLiveTiers] = {0};
    int64_t n_units = 0;           /* expert units in the plan; NOT n_layers * n_experts, which
                                    * counts cells a ragged model never had */
    int64_t total_bytes = 0;
    int64_t movable = 0;           /* how many of them the heat engine is allowed to relocate */

    /* Per layer, so a starved layer is visible as a number and not only as a dark band. */
    std::vector<int32_t> layer_units;
    std::vector<int32_t> layer_resident;

    /* What the mover did. Cumulative since start: a rate over a poll interval is the reader's
     * subtraction, and a counter is the only form that survives being read twice. */
    int64_t promotions = 0, demotions = 0;
    /* Is the churn PROGRESS or THRASH? Same move count, opposite meanings -- see HeatStats. */
    int64_t distinct_promoted = 0, readmits = 0;
    /* Why a dispatch moved nothing. `refused` is the guard declining, which is the healthy steady
     * state; the starved counts are the mover having nowhere to put the bytes. */
    int64_t dispatches = 0, refused = 0, no_candidate = 0;
    /* DID THE ROUTER FIND THE EXPERT IN VRAM. `routed` is expert selections seen by the heat
     * engine and `routed_resident` the subset that was already there; the difference is what the
     * MoE GEMM had to stream across the link, which LinkStat::stream_bytes prices. A residency
     * percentage says how much of the model is close; this says how much of the WORK is. */
    int64_t routed = 0, routed_resident = 0;
    int64_t no_record = 0, starved_slab = 0, starved_pool = 0, quarantined = 0;
    /* The prefill stager's copies (Mover::stage_units), rank 0: layers staged, units and bytes
     * copied onto the card, and pooled units a buffer had no room for. */
    int64_t stage_layers = 0, staged_units = 0, staged_bytes = 0, stage_overflow = 0;
    /* HOW MUCH OF THE SLAB IS BORROWED. `flex_bytes` is what the slab holds in VRAM the KV cache
     * lent it; `flex_capacity` is how much the cache is lending right now. The pair is what says
     * whether the boundary between the two pools is moving, and whether what was lent is being
     * used -- a gap between them is lent VRAM holding nothing. */
    int64_t flex_bytes = 0, flex_capacity = 0;
    bool    heat_enabled = false;  /* false = the plan is static and none of the above will move */
};

/* WHAT IS ON THE INTERCONNECT, and in what proportion.
 *
 * Three things share it and they are never discussed together: the tensor-parallel all-reduce,
 * which is a property of the model's width and the batch; the expert mover's promotions and
 * demotions, which are a property of the placement policy; and the MoE GEMM streaming an expert
 * the router asked for and placement had not put in VRAM, which is what the policy is spending
 * the moves to avoid. All three are cumulative byte counts from the same wall clock, so a reader
 * who subtracts two polls gets rates that can be compared -- which is the whole question "is the
 * link busy because the model is wide, or because placement is thrashing".
 *
 * `ar_bytes` is the message payload summed over issues; Ctx::ar_bytes documents exactly what that
 * is and is not. `h2d`/`d2h` are the mover's own measured copy sizes.
 */
struct LinkStat {
    bool    present = false;       /* false when nothing measured any of it */
    int     world = 1;
    int64_t ar_calls = 0, ar_bytes = 0;
    int64_t h2d_bytes = 0, d2h_bytes = 0;
    int64_t ssd_reads = 0, ssd_bytes = 0;
    /* The MoE GEMM reading a routed expert that was not resident. Not a move and not a
     * collective: the third consumer of the link, and the one a promotion is spent to remove. */
    int64_t stream_bytes = 0;
    int64_t prefetched = 0, prefetch_bytes = 0;
    /* The prefill stager's copies of the next layer's pooled experts (core/place/stager.h). Part
     * of `h2d_bytes`, like `prefetch_bytes`: the mover counts every copy it makes there, and a
     * reader that wants the promotions alone takes both out. */
    int64_t staged_bytes = 0;
};

/* HOW THE RANKS RAN THEIR PASSES, summed over ranks: Ctx::PassCounts says what each is. A played
 * pass costs the host a packet copy; every other kind walks the architecture's issue path, so the
 * share that is not `played` is where a step's host time goes. */
struct PassStat {
    bool    present = false;
    int64_t played = 0, recorded = 0, audited = 0;
    int64_t issued_unkeyed = 0, issued_refused = 0, issued_first = 0;
    int64_t dispatched = 0, unordered = 0;
    int64_t evicted = 0, tape_cap = 0;   /* tapes evicted (summed); the per-rank capacity */
};

struct LiveSnapshot {
    int64_t     step = 0;
    std::string model;
    /* Duplicated rather than referenced so that a snapshot is a value the drawing thread can hold
     * without racing the scheduler that produced it. */
    int64_t waiting = 0, running = 0, preempted = 0;
    double  kv_util = 0.0, prefix_hit_rate = 0.0;
    int64_t prefix_evictions = 0;
    double  prefill_tps = 0.0, decode_tps = 0.0, draft_acceptance = 0.0;
    int64_t total_requests = 0, preemptions = 0;

    std::vector<CardStat> cards;
    ExpertPlane           experts;
    LinkStat              link;
    PassStat              passes;
};

void snapshot_from(const SchedMetrics& m, LiveSnapshot* out);

class LiveView {
public:
    using Source = std::function<void(LiveSnapshot&)>;

    /* `out` defaults to stdout. `interval_ms` is the redraw period; 250 ms is fast enough to see
     * a stall and slow enough that the view costs nothing measurable. */
    LiveView(Source src, int interval_ms = 250, FILE* out = nullptr);
    ~LiveView();

    void start();
    void stop();


    /* Both renderers are public because they are the testable part: a snapshot in, a string out,
     * no terminal involved. */
    std::string render_screen(const LiveSnapshot& s) const;
    std::string render_line(const LiveSnapshot& s) const;


private:
    void loop();

    Source            src_;
    int               interval_ms_;
    FILE*             out_;
    bool              tty_ = false;
    int               cols_ = 100;
    std::atomic<bool> running_{false};
    std::thread       thread_;
};

}  /* namespace server */
}  /* namespace rad */
