/* kvtier.h -- where a finished conversation's KV is kept once VRAM has a better use for it
 * (spec §7.3).
 *
 * The prefix cache holds a KV reference on every block it indexes, so a finished turn's context
 * survives into the next one. That is the whole value of it, and it is also the whole cost: those
 * blocks are the most valuable thing in the pool while the session is live and the least valuable
 * thing in it once the session stops -- and on a card whose experts do not fit, VRAM holding an
 * idle conversation is VRAM every decode step pays for across the link.
 *
 * ============================== COPY ON IDLE, FREE ON DEMAND ====================================
 *
 * An entry is COPIED to host memory as soon as the request that published it finishes, and its
 * VRAM copy is kept. From then on the entry is CLEAN: the pool and the host hold the same bytes,
 * so giving the VRAM copy up costs nothing at all -- no transfer, no wait, only the cache's
 * reference -- and it is given up the moment something wants the memory: the expert slab's loan,
 * or an allocation the free list cannot serve. A conversation that comes back before either
 * happens resumes from VRAM for free; one that comes back after is restored from its host copy,
 * and KEEPS that copy, so the turn after it copies only the tokens that turn added.
 *
 *     published --(copy, at once)--> clean on device --(memory wanted)--> host
 *                                        ^                                  |
 *                                        +------------(a hit restores)------+
 *     host copy --(copy to disk, at once)--> clean on host --(a slot wanted)--> disk
 *
 * THE DISK TIER IS A COPY TOO, by the same argument one level down. Every host copy is written to
 * the disk store in the background as soon as it lands, and keeps its host slot. Once the write
 * has landed the host slot is as free to give up as a clean entry's VRAM: when a copy or a
 * restore needs a slot and the arena is full, the oldest entries the disk also holds give theirs
 * up, which is bookkeeping and nothing else. Moving an entry to disk only when its slot is wanted
 * would put a write on the path of whatever wanted it -- a restore out of disk needs a host slot to
 * read through, and an arena full of host-only entries has no slot to give it at all.
 *
 * THERE IS NO IDLE THRESHOLD ON THE WAY OUT OF VRAM, and that is the point of the design rather
 * than an omission. A threshold is a guess at when an entry stops being worth its VRAM, made in
 * seconds by an operator; a clean entry answers the question itself, because keeping it costs
 * nothing until the memory is wanted and releasing it costs nothing once it is.
 *
 * WHY MOVING BEATS DROPPING, stated once so the costs can be argued about in the right units:
 * dropping a 200K-token prefix costs a re-prefill, which is minutes. Restoring it costs a copy
 * over PCIe, which is a fraction of a second. The two are not the same order of magnitude.
 *
 * ============================== THIS OBJECT MOVES NO BYTES ======================================
 *
 * It decides WHAT should move and hands the work out as a list of TierJob. The engine executes
 * them on its transfer stream and reports back. That split is not decoration: `KVManager` already
 * works this way (BlockCopy, take_pending_copies) for the same reason -- core/mem has no stream,
 * and a synchronisation on the step path is what this design spends itself avoiding (spec §5.4).
 * A second mechanism for moving KV bytes would be a second place to get the pool's addressing
 * wrong.
 *
 * ============================== THE ADDRESSING, WHICH IS THE TRAP ==============================
 *
 * THE POOL IS LAYER-MAJOR (core/mem/kv.h). Layer i of a paged group owns
 * `[base + i*layer_stride, + layer_stride)`, and inside that a block is
 * `block_index * layer_bytes_per_block()`. A logical block is therefore NOT a contiguous extent:
 * it is `n_layers` fragments, one per layer, and on a production model a fragment is a few hundred
 * bytes. `base + block * bytes_per_block` addresses something else entirely -- roughly `n_layers`
 * consecutive blocks OF LAYER ZERO -- and a tier built on it moves other sequences' data and
 * restores a jumble. Corrupt KV does not read as babble; it reads as fluent, deterministic,
 * plausible, wrong text, and nothing short of a byte comparison against a known-good run finds it.
 *
 * So the unit of transfer here is not a block. It is A LAYER'S WORTH OF MANY BLOCKS, which is the
 * shape the pool is already in: within one layer the group's blocks are dense rows of
 * `layer_bytes_per_block()` each, so packing an arbitrary set of them is exactly `gather_rows`
 * and restoring them is exactly `scatter_rows`. One launch per (group, layer) per batch, however
 * many entries the batch holds.
 *
 * ============================== ONE RECORD, ONE TIER, ONE LIST ==================================
 *
 * An entry that can live in either of two tiers must not be on BOTH tiers' LRU lists: each tier's
 * eviction path then invalidates references the other holds, and every such pair is a
 * use-after-free. The rule here is that a key has exactly one Rec,
 * a Rec is in exactly one tier, and a Rec is in at most that tier's list. A CLEAN device record
 * is on no list: the copy it keeps off the card is what lets the VRAM copy go, and the tiers'
 * lists are for entries no longer in VRAM. A Rec with a transfer through the pool outstanding is
 * `pending` and is in no list either: it cannot be chosen as a victim, cannot be planned again,
 * and cannot be erased until the engine reports the job finished or failed. A disk write is not
 * such a transfer -- it reads host memory and nothing else -- and holds only the slots it reads.
 */
#pragma once
#include "mem/hostarena.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "rad_core.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rad {

class SSDTier;

enum class KVTier : uint8_t { Device = 0, Host = 1, Disk = 2 };
const char* kv_tier_name(KVTier t);

/* What the engine is being asked to do. One job is one entry moving one step. The block ids are
 * carried in the job rather than looked up when it runs, because the whole point of `pending` is
 * that nothing may touch the record in between. */
struct TierJob {
    enum class Kind : uint8_t {
        WriteBack,      /* gather_rows out of the pool into a host slot; the entry stays in VRAM */
        ToDisk,         /* a host slot written to the disk store; the host copy is kept */
        ToDevice,       /* allocate blocks, scatter_rows in; the host copy is kept */
        DropHost,       /* no disk tier and the arena is full: this record's bytes go */
        ToHost,         /* the disk store read into a host slot, ahead of the restore that wants it */
    };
    Kind      kind = Kind::WriteBack;
    BlockHash key;
    /* Per cached group, in PrefixCache::cached_groups() order. For a write-back these are the
     * blocks to read -- EMPTY when only the snapshot is being copied, because the blocks already
     * have their host copy; for a promotion they are filled in by the engine with the blocks it
     * allocated. */
    std::vector<int32_t> blocks;
    int32_t   host_slot = -1;
    int64_t   bytes = 0;

    /* ---- the linear half, on a hybrid model ----
     *
     * A session's recurrent state lives in a SNAPSHOT SLOT of the checkpoint store -- pinned host
     * memory, one contiguous KVManager::checkpoint_bytes() shard a rank -- rather than in blocks,
     * so it never crosses the link with them: a lookup restores it straight out of its slot. The
     * tiers only move it between that slot and the disk store, on the disk thread: a ToDisk job
     * reads `ck_slot`, a ToHost job writes it.
     *
     * `ck_slot` is an INPUT to both. A read off disk is planned with `ck_want` and the engine
     * allocates the slot before the job leaves its hands (it holds the store's free list); a
     * write names the slot the snapshot is in. */
    int32_t   ck_slot = -1;
    int64_t   ck_pos = 0;       /* the token position, which the disk store verifies on the way back */
    bool      ck_want = false;  /* a read off disk should bring the snapshot in too */
    int64_t   ck_bytes = 0;
    /* DID THE SNAPSHOT HALF SUCCEED. Separate from the job's own result because the two halves
     * fail for different reasons and are repaired differently: a snapshot whose disk write failed
     * is still correct in its slot, and one whose read failed costs the session a replay of its
     * linear layers, not its blocks. */
    bool      ck_ok = true;
    /* A PROMOTION THAT WAS NEVER STARTED, as opposed to one that failed part of the way through.
     * A restore reserves every block before it moves a byte, and a chain that did not fit is cut
     * where the reservation ran out: the entries past that point were not touched, their host
     * copies are exactly what they were, and they stay on the host tier. Only an entry whose
     * scatter may have run is dropped on failure. A read off disk that never ran -- its batch
     * could not be handed to the thread -- is the same: the record stays on disk. */
    bool      started = false;
};

/* ------------------------------------------------------------------ the policy */
class IdleTiers {
public:
    struct Config {
        int64_t host_cap_bytes  = 0;   /* 0 is off */
        int64_t disk_cap_bytes  = 0;
        int64_t entry_bytes     = 0;   /* one entry: sum over cached groups of bytes_per_block */
        /* ---- the linear half ----
         * `ckpt_bytes` is ONE SNAPSHOT, every rank's shard of it, laid out exactly like
         * `entry_bytes`. The snapshots live in the engine's checkpoint store, `ckpt_slots` deep,
         * which is reported here and not allocated: --prefix-cache-host-mib is the blocks' alone.
         * `ckpt_disk_slots` zero means snapshots never reach disk, and the blocks still do. */
        int64_t ckpt_bytes      = 0;
        int64_t ckpt_slots      = 0;   /* the checkpoint store's capacity, for the report and view */
        int64_t ckpt_disk_slots = 0;   /* how many the disk store was opened for */
    };

    int  configure(const Config& c, SSDTier* disk);
    void close();
    bool host_on() const { return cfg_.host_cap_bytes > 0 && arena_.enabled(); }
    bool disk_on() const { return cfg_.disk_cap_bytes > 0 && disk_ != nullptr; }
    bool any_on()  const { return host_on() || disk_on(); }
    std::string report() const;

    /* Called by PrefixCache whenever an entry is inserted. The blocks come with it because a copy
     * needs them and the map is the only place they would otherwise be read from -- at which point
     * the tiers would hold a second copy of the cache's own bookkeeping and the two could
     * disagree. A record with no host copy is queued for one here: publishing is the moment a
     * request has stopped writing the entry, which is the moment it can be copied. */
    void note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns);
    /* THE HIT HALF OF THAT, for a lookup that matched `n` blocks of a chain. Only the idle clock
     * moves: a hit carries no new block ids, and it creates no record for an entry the tiers were
     * never told about. One lock for the whole chain, since a long prefix is thousands of blocks. */
    void note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns);
    /* The entry is gone from the cache for good -- evicted, or its blocks released. Any tier
     * storage it held is returned. A record with work outstanding is marked instead of erased and
     * is cleaned up when the job reports. */
    void forget(const BlockHash& key);
    /* AN OFF-DEVICE ENTRY WAS RECOMPUTED. A request that could not reach the tier copy -- a
     * restore that did not fit, a chain past the point a promotion was cut -- computes the same
     * tokens again and publishes them, and the cache takes those blocks as the entry's device
     * copy. The tier copy goes: two copies of one entry computed by two different batches agree
     * only to the rounding the batch shape allows, and a record that called them the same bytes
     * would restore the older one over a pool that holds the newer. The record is a fresh device
     * record, queued for a copy of its own. */
    void rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns);

    /* WHERE IS IT. Device for anything this object has never been told about, which is the right
     * answer: an entry the cache holds and the tiers have not touched is in the pool. */
    KVTier tier_of(const BlockHash& key) const;
    bool   resident_off_device(const BlockHash& key) const;

    /* ---------------- copy on idle ----------------
     *
     * The records waiting for a host copy, up to `max_entries`, as WriteBack jobs with their host
     * slots already allocated. Oldest first, so a conversation that went idle before another is
     * copied before it. Blocks only: a snapshot is in host memory from the moment it is saved.
     *
     * Every record named by a returned job is `pending` until report() is called for it. An arena
     * too full to take the batch first gives up the host slots of the oldest entries the disk also
     * holds (make_room); what is still short stays queued, and with no disk tier the next
     * maintenance plan drops that many of the oldest host-only entries. */
    void plan_writeback(int64_t max_entries, std::vector<TierJob>* out);
    bool writeback_waiting() const;
    /* Has the arena refused a copy since the last maintenance plan? That plan is then due now. */
    bool starved() const;

    /* THE VRAM COPIES THAT CAN GO, as the entries that became clean since the last call. The
     * cache decides whether each one actually goes -- a block a running sequence also holds would
     * free nothing -- and reports what it did through drop_commit. A key handed out here and not
     * dropped is the cache's to ask about again. */
    void take_clean(std::vector<BlockHash>* out);
    /* Can this entry's blocks leave VRAM for nothing, right now: a copy off the card has landed --
     * in host memory or on disk -- and no job is out on the record. */
    bool droppable(const BlockHash& key) const;
    /* The cache released the blocks. The record is now a host record with its copy, on the host
     * tier's list like any other -- or a disk record, when its host slot was already given up.
     * Its snapshot stays in its slot: the store is not the pool, and the blocks leaving frees
     * nothing there. */
    void drop_commit(const BlockHash& key);

    /* ---------------- the snapshots' own pressure ----------------
     *
     * The checkpoint store is --checkpoint-slots deep for the whole server, and when it is full
     * the cache retires the least recently used snapshot to make room. One the disk also holds
     * need not be lost: the cache DETACHES it instead -- the slot goes back, the index entry
     * stays, and the next restore of the session reads the state back off disk (plan_fetch).
     * True when this key's snapshot is in a slot and has a finished disk copy that no write is
     * still producing. */
    bool snapshot_backed(const BlockHash& key) const;
    /* Is this key's snapshot in store slot `slot` by this object's record? */
    bool snapshot_in_slot(const BlockHash& key, int32_t slot) const;
    /* The cache gave the slot back; the disk copy is now the snapshot. */
    void snapshot_detached(const BlockHash& key);

    /* ---------------- the slow half ----------------
     *
     * The host copies waiting for their disk copy, in the order they landed, as ToDisk jobs --
     * blocks and snapshot halves together where both are waiting, and a resume point's snapshot
     * ahead of everything (note_snapshot). A ToDisk job does NOT make its record pending: it only
     * reads the host slots it names, which a restore may read at the same time, and a slot given
     * up while the write is out is freed when the write reports rather than under it. With no disk tier, the copies the arena refused since the last plan
     * are made room for by dropping that many of the oldest host-only entries. `budget` bounds
     * one plan so the IO a pass asks for is bounded too. */
    void plan(int budget, std::vector<TierJob>* out);
    /* Is a host copy waiting for its disk copy? The slow half is then due now rather than on its
     * once-a-second clock: an entry the disk does not hold yet is one whose host slot cannot be
     * given up for nothing. */
    bool disk_waiting() const;

    /* ROOM IN THE ARENA, FOR NOTHING. Up to `n` host slots of entries the disk also holds, given
     * up oldest first: host records first -- they become disk records -- and then the host copies
     * of clean VRAM entries, which stay clean through their disk copy. Nothing is written; an
     * entry whose disk copy has not landed is not a candidate. Returns how many were freed. */
    int64_t make_room(int64_t n);

    /* ---------------- reading a chain back off disk, before its request runs ----------------
     *
     * A RESTORE COPIES HOST MEMORY TO THE DEVICE AND NOTHING ELSE. It runs inside the scheduler's
     * step, and a disk read there -- thousands of entries for a long conversation -- would hold
     * every other request's next token for as long as the reads take. So the disk half of a chain
     * is read into host slots on the IO thread first, as ToHost jobs, while the request that wants
     * it waits outside admission and everything else keeps running; the restore then finds the
     * chain in host memory.
     *
     * `chain` is every key a prompt's prefix matched, in order. True when the request must wait:
     * a read for one of its entries was planned now, into `out`, or is already out. False when
     * none of the chain is on disk, or none of what is can be given a slot -- a chain larger than
     * the arena comes back as far as the arena holds, and the rest is prefilled.
     *
     * The chain's deepest snapshot, when it is only on disk, is read with it -- that is the state
     * the hit resumes from -- into the store slot `*ck_slot` the caller allocated after asking
     * snapshot_fetch_wanted. The slot is taken (and `*ck_slot` set to -1) only when the read is
     * planned; otherwise it is still the caller's to free. Every record named is `pending` until
     * the read reports, and what a read brought in is not given up to make room for anybody else
     * until the request has had the step it needs to claim it. */
    bool plan_fetch(const std::vector<BlockHash>& chain, std::vector<TierJob>* out,
                    uint64_t now_ns, int32_t* ck_slot = nullptr);
    /* Would plan_fetch read this chain's deepest snapshot off disk -- is it there and nowhere
     * else? Asked first, because the slot it reads into is the store's, which the caller owns. */
    bool snapshot_fetch_wanted(const std::vector<BlockHash>& chain) const;

    /* A promotion, asked for by a lookup that hit off-device. Returns false when the entry is not
     * in host memory -- a disk entry is read in by plan_fetch first -- or already has work
     * outstanding. */
    bool plan_promote(const BlockHash& key, TierJob* out);

    /* The engine finished (or failed) a job. On failure the record returns to the state it was in
     * with its storage intact, EXCEPT for a failed ToDevice that had started, which cannot leave a
     * half-scattered pool behind and so drops the entry -- and a disk read that failed is marked
     * started for that reason too: the store no longer holds what the record says it does. A
     * successful ToDevice restarts the record's clock at `now_ns` when one is given: the promotion
     * happened because something is about to use it. */
    void report(const TierJob& job, bool ok, uint64_t now_ns = 0);

    /* WAS THIS KEY FORGOTTEN WHILE ITS JOB WAS OUT -- or is it not known at all. The engine asks
     * before acting on a finished job's blocks: an entry the cache evicted mid-transfer had its
     * references released by the eviction, and anything done in its name afterwards would be done
     * to blocks somebody else now holds. */
    bool doomed(const BlockHash& key) const;

    /* ============================ WHEN THE BYTES ARE GONE, SO IS THE ENTRY ====================
     *
     * A record erased while its bytes are OFF-DEVICE -- a capacity drop, or a promotion that
     * failed -- leaves the prefix cache holding an entry whose block ids the pool has already
     * given to somebody else. `resident_off_device` then answers false, because there is no
     * record any more, and the cache would serve those ids as an ordinary hit: one conversation
     * gets another's context, as fluent and plausible text.
     *
     * So the tiers tell the cache, at every site that erases an off-device record, and the cache
     * drops the entry WITHOUT releasing blocks -- there are none of its own left to release. */
    void set_drop_sink(std::function<void(const BlockHash&)> f) { on_drop_ = std::move(f); }

    /* ---------------- the linear half of an idle session ----------------
     *
     * Told by PrefixCache::commit_checkpoint, for the same reason note_used carries the block ids:
     * a copy has to know what to move and the cache's index is the only place it is written down.
     * `slot` is a checkpoint store slot and `pos` the token position the disk store verifies a
     * read against.
     *
     * A record may hear this BEFORE it hears note_used -- a checkpoint is committed mid-request
     * and the chain is published when the request closes -- so this creates the record if it has
     * to. The snapshot is queued for its disk copy when the record is published, or at once when
     * it already was -- or AT ONCE AND AHEAD OF EVERYTHING when it is a resume point (`resume`):
     * the deepest checkpoint its prompt gets, which the next turn resumes from. Until it is on disk
     * the store can only retire it, and a burst of prefills churns the store through the whole
     * decode in between. */
    void note_snapshot(const BlockHash& key, int32_t slot, int64_t pos, bool resume = false);
    /* The cache is done with the snapshot at `key`, which is in store slot `slot`: return whatever
     * the tiers hold for it. The record's BLOCKS are untouched -- the two halves are retired by
     * different policies on different clocks, and a snapshot the retention policy drops says
     * nothing about the prefix.
     *
     * TRUE WHEN A DISK WRITE IS STILL READING THE SLOT, and then the slot is NOT the caller's to
     * free: a save into it would put half of another snapshot on disk under this key. The tiers
     * keep the slot and hand it back through `take_checkpoint_frees` once the write has reported.
     * Decided by the slot and not by the record, which may have gone while the write was out. */
    bool release_snapshot(const BlockHash& key, int32_t slot);
    /* Store slots whose release waited on a disk write, now safe to free. */
    void take_checkpoint_frees(std::vector<int32_t>* out);
    /* Is a disk write reading store slot `slot` right now? A save into it would tear the write. */
    bool slot_reading(int32_t slot) const;
    /* THE TIERS LOST A SNAPSHOT BUT KEPT THE BLOCKS: its only copy was on disk and the disk store
     * no longer has it. The cache has to stop offering an index entry whose bytes nobody holds.
     * Wired to PrefixCache::drop_snapshot; separate from set_drop_sink because losing half an
     * entry and losing all of it are different events with different repairs. Never said of a
     * snapshot still in its slot: that one is the cache's. */
    void set_snapshot_drop_sink(std::function<void(const BlockHash&)> f) { on_ck_drop_ = std::move(f); }
    /* Do snapshots reach disk at all? */
    bool     ckpt_disk_on() const { return disk_on() && cfg_.ckpt_bytes > 0 && cfg_.ckpt_disk_slots > 0; }
    int64_t  ckpt_bytes() const { return cfg_.ckpt_bytes; }
    int64_t  ckpt_disk_slots() const { return cfg_.ckpt_disk_slots; }
    int64_t  ckpt_slots() const { return cfg_.ckpt_slots; }

    /* ---------------- the session view ----------------
     *
     * WHAT A SESSION IS HERE, because it is not what the cache is keyed by. The cache is keyed by
     * BLOCK, and a 200K-token conversation is twelve thousand of them -- an enumeration nobody can
     * read and not what anyone means by "session". A session is one PUBLISHED CHAIN: the blocks
     * one request inserted, named by its tip. That is exactly the leaf of the prefix trie, so two
     * conversations sharing a system prompt are two sessions and the shared blocks belong to both.
     *
     * THE SIZES THEREFORE OVERLAP AND THE TOTAL IS NOT A SUM. Sharing is the point of the cache;
     * a view that hid it would make a server look like it was storing several times what it is.
     * Anything reporting these numbers says so.
     *
     * The model name is carried rather than looked up because the disk tier outlives the process:
     * a store written by one container and read back under another is refused by fingerprint, and
     * the name is what turns that refusal into something an operator can act on. */
    struct SessionInfo {
        uint64_t    id = 0;
        std::string model;
        std::string label;     /* the client's own name for the conversation, or "" */
        int64_t     n_tokens = 0;
        int64_t     bytes = 0;
        uint64_t    first_seen_ns = 0;
        uint64_t    last_used_ns = 0;
        int64_t     on_device = 0, on_host = 0, on_disk = 0;   /* blocks, by tier */
        /* LINEAR STATE, on a hybrid model. Counted separately from blocks because it is a
         * different resource with a different scarcity: a snapshot is thousands of blocks' worth
         * of bytes and there are only --checkpoint-slots of them in host memory. A session showing
         * blocks on disk and no snapshot anywhere is one that will come back and re-run every
         * recurrent layer over its whole transcript, which is not what "restored" should mean. */
        int64_t     snapshots = 0;                             /* this session holds, any tier */
        int64_t     snap_host = 0, snap_disk = 0;              /* in the store; only on disk */
        int64_t     snap_bytes = 0;
        /* WHAT A RETURN REUSES: the token position of the deepest snapshot on the chain, in any
         * tier. On a hybrid model the scheduler clamps a hit to the last linear checkpoint it can
         * reach, so this -- not n_tokens and not the blocks -- is the prefix the next turn skips,
         * and n_tokens minus it is what that turn prefills. Zero with no snapshot anywhere. */
        int64_t     resume_tokens = 0;
        bool        moving = false;                            /* a job is outstanding */
    };
    /* WHAT THE TIERS OFF THE CARD ARE HOLDING, blocks and snapshots together, each counted once.
     * The per-session figures cannot be summed into this: sessions sharing a prefix share its
     * blocks and its snapshots. Caps are what the operator configured, snapshots included. A
     * clean entry's host copy is counted: it occupies the arena whether or not the VRAM copy has
     * gone yet. */
    struct Occupancy {
        int64_t host_bytes = 0, disk_bytes = 0;
        int64_t host_cap_bytes = 0, disk_cap_bytes = 0;
    };
    /* Publish one request's chain. Called from PrefixCache::insert, which is the only place that
     * knows a chain was completed rather than merely touched.
     *
     * KEPT WHETHER OR NOT ANY TIER IS ON, because what it records is not a tiering fact. A server
     * with no --prefix-cache-host-mib still stores conversations -- it stores them all in VRAM, which
     * is the case the table most wants to show -- and gating the record on the mover would make
     * the view report an empty server to every operator who has not opted into tiering.
     *
     * ONE CONVERSATION IS ONE ROW, and the rule that makes it one is the trie: a session is a
     * LEAF, so a chain that a later turn extends has stopped being one and is absorbed into it.
     * Each turn publishes the chain it grew under its own request id, so without that the table
     * is a row per turn -- a table nobody can read, holding a copy of every prefix of the
     * transcript for as long as the newest. */
    void note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                      uint64_t now_ns, const std::string& label = std::string());
    void set_model(const std::string& m) { model_ = m; }
    /* Newest activity first. Recomputes each session's tier histogram from the records, so it is
     * a view and never a second copy of the truth. The walk takes the lock in slices and lets
     * every other taker go first between them (YieldingMutex), so rows read at different slices
     * may describe moments a few microseconds apart. */
    std::vector<SessionInfo> sessions(Occupancy* occ = nullptr) const;
    void forget_session(uint64_t id);

    struct Stats {
        /* `to_host` is entries COPIED to host memory; `dropped` is entries whose VRAM copy was
         * then given up. They are different events: a copy is link traffic and a drop is VRAM
         * handed to something else, and a conversation that returns before anything wants its
         * VRAM is copied and never dropped. */
        int64_t to_host = 0, dropped = 0, to_disk = 0, promoted = 0;
        /* `host_freed` is host slots given up because the disk holds the entry -- the host tier's
         * counterpart of `dropped`; `host_drops` is entries lost with no disk tier to hold them. */
        int64_t host_freed = 0, host_drops = 0, failures = 0;
        int64_t host_bytes = 0, disk_bytes = 0;
        int64_t host_hits = 0, disk_hits = 0;
        /* Entries and snapshots read off disk into host memory ahead of a restore. */
        int64_t fetched = 0, ck_fetched = 0;
        /* SNAPSHOTS MOVED, counted apart from blocks. A hybrid session can move its whole chain
         * off the device and still replay every linear layer from zero on the way back, and no
         * block counter can tell the two apart -- these are the numbers that can. `ck_drops` is
         * snapshots whose only copy was lost. */
        int64_t ck_to_disk = 0, ck_drops = 0;
        /* WHAT THE STEP THREAD SPENT ON THEM, under the scheduler's lock: restores issued at
         * admission, and copies landed synchronously because pressure could not wait for them.
         * Host time, not link time -- what every other request waits through. */
        int64_t restores = 0, restore_ns = 0, flushes = 0, flush_ns = 0;
    };
    /* The step thread's own time on the tiers (see Stats). */
    void note_restore(int64_t ns) { std::lock_guard lk(mu_); ++stats_.restores; stats_.restore_ns += ns; }
    void note_flush(int64_t ns)   { std::lock_guard lk(mu_); ++stats_.flushes;  stats_.flush_ns += ns; }
    /* A copy, taken under the lock: an HTTP thread reads these while the engine thread moves
     * records. */
    Stats        stats() const;
    HostArena&   arena() { return arena_; }
    /* The size of ONE SLOT, which under tensor parallelism holds every rank's shard. */
    int64_t      entry_bytes() const { return cfg_.entry_bytes; }

private:
    struct Rec {
        KVTier   tier = KVTier::Device;
        uint64_t used_ns = 0;
        /* THE HOST COPY. On a device record it is a copy that makes the record clean; on a host
         * record it is where the bytes are read from. -1 on a disk record always. */
        int32_t  host_slot = -1;
        /* THE DISK STORE HOLDS THESE BYTES, written from the host copy. What makes the host slot
         * free to give up, and a device record with no host copy clean. The store evicts on its own
         * and does not say so; a read that finds the entry gone drops it (report). */
        bool     on_disk = false;
        /* The host slot a disk write is reading right now, -1 when none. A slot given up while
         * this names it is freed when the write reports, not under it. */
        int32_t  disk_src = -1;
        bool     disk_q  = false;   /* in dk_q_: waiting for a disk copy */
        bool     pending = false;
        bool     fetching = false;  /* the pending job is a read off disk (plan_fetch) */
        bool     doomed  = false;   /* forget() arrived while pending */
        bool     queued  = false;   /* in wb_q_: waiting for a copy */
        bool     resume_q = false;  /* in resume_q_: a resume point waiting for its disk copy */
        bool     clean_q = false;   /* in clean_q_: a drop the cache has not been asked about */
        bool     listed  = false;   /* on host_lru_ or disk_lru_; `it` is valid */
        std::vector<int32_t> blocks;
        /* ---- the linear half ----
         * `ck_slot` is the snapshot's checkpoint store slot and `ck_on_disk` the disk store's
         * copy. A snapshot the store gave up has only the second. All clear when the record
         * carries no snapshot, which is the common case -- one block in every
         * (interval / block_size) is a checkpoint position. */
        int32_t  ck_slot = -1;
        bool     ck_on_disk = false;
        int64_t  ck_pos = 0;
        /* The store slot a disk write is reading, as `disk_src` is for the blocks. */
        int32_t  ck_disk_src = -1;
        /* WHEN A READ OFF DISK BROUGHT THIS RECORD IN, 0 when none did or its restore has run.
         * Until the request that asked for it has been admitted the slots are its and not the
         * arena's to give up: two conversations waiting on reads into a full arena would
         * otherwise each evict what the other had read, and neither would ever be restored. Held
         * for kFetchHoldNs at most, which bounds what a request that went away can pin. */
        uint64_t fetched_ns = 0;
        /* Set only while plan_fetch makes room: this record is part of the chain being read, and
         * giving its slot up to read another part of the same chain would be no progress. */
        bool     keep = false;
        std::list<BlockHash>::iterator it;
    };

    /* Does this record want a host copy it does not have yet? Its blocks are in VRAM with none. */
    bool wants_copy(const Rec& r) const;
    /* And a disk copy: a host copy of either half that the disk does not hold yet. */
    bool wants_disk(const Rec& r) const;
    bool wants_ck_disk(const Rec& r) const;
    void queue_copy(const BlockHash& key, Rec& r);
    void queue_clean(const BlockHash& key, Rec& r);
    void queue_disk(const BlockHash& key, Rec& r);
    /* GIVE UP A HOST SLOT, the record's or a finished job's. Every block-arena free goes through
     * `give_slot`, because a freed slot is what unblocks a write-back that found the arena full.
     * `free_host` is the record's own: a slot a disk write is reading is left
     * for that write's report to free. */
    void give_slot(int32_t slot);
    void free_host(Rec& r);
    void set_on_disk(Rec& r, bool v);
    int64_t make_room_locked(int64_t n);
    /* May this record's host copies be given up to make room right now? Not while a read put
     * them there for a request that has not claimed them yet, nor while the chain being read
     * includes it. */
    bool held_by_fetch(const Rec& r, uint64_t now_ns) const;
    void unlist(Rec& r);
    void relist(const BlockHash& key, Rec& r);
    std::list<BlockHash>& list_for(KVTier t) { return t == KVTier::Host ? host_lru_ : disk_lru_; }
    void release_storage(const BlockHash& key, Rec& r);
    /* The chain position of the deepest snapshot no job is out on, or chain.size(). */
    size_t deepest_snapshot(const std::vector<BlockHash>& chain) const;

    struct Session {
        int64_t  n_tokens = 0;
        uint64_t first_seen_ns = 0;
        uint64_t last_used_ns = 0;
        std::vector<BlockHash> chain;
        std::string label;
    };

    /* Erase one row and its root index entry together. */
    void drop_session(uint64_t id);
    /* The body of close(), for a caller that already holds the lock. */
    void reset();

    /* TWO THREADS REACH THIS OBJECT. The engine loop owns every mutation -- the cache publishes a
     * chain from the scheduler's close path, the maintenance pass moves records -- and an HTTP
     * thread drawing the sessions view walks the same containers through sessions(). An
     * unordered_map rehashed under an iterator is a crash rather than a stale number, so the
     * state the view reads is held under this and every mutator takes it.
     *
     * NOT A BOTTLENECK, and it is worth saying where the cost lands: the per-block calls on the
     * publish path (note_used) take it once each, uncontended, against a hash insert they were
     * already paying for. No method here calls another that takes it, which is what keeps a plain
     * mutex sufficient; the drop sinks are called while it is held and reach into the prefix
     * cache, which never calls back.
     *
     * AND THE VIEW'S WALK MUST NOT DELAY THE ENGINE. It visits every block of every stored
     * conversation, which is far longer than any engine-side hold, and the engine takes this lock
     * every step. So the walk takes it for a slice of records at a time and, between slices,
     * waits until nobody else is waiting for it: an engine-side taker that arrives during a slice
     * waits for that slice and never for the walk. A plain mutex cannot promise that, because
     * the walker re-locking the moment it unlocks can win against a woken waiter every time. */
    struct YieldingMutex {
        std::mutex       m;
        std::atomic<int> waiting{0};
        void lock() {
            waiting.fetch_add(1, std::memory_order_relaxed);
            m.lock();
            waiting.fetch_sub(1, std::memory_order_relaxed);
        }
        void unlock() { m.unlock(); }
        /* The view's: taken only while no other taker is waiting. */
        void lock_after_others() {
            for (;;) {
                while (waiting.load(std::memory_order_relaxed) > 0) std::this_thread::yield();
                m.lock();
                if (waiting.load(std::memory_order_relaxed) == 0) return;
                m.unlock();
            }
        }
    };
    mutable YieldingMutex mu_;

    Config   cfg_;
    SSDTier* disk_ = nullptr;
    HostArena arena_;
    std::unordered_map<BlockHash, Rec, BlockHashHash> recs_;
    std::list<BlockHash> host_lru_, disk_lru_;   /* front = oldest */
    /* THE WORK QUEUES, in the order the work arrived. Each holds a key at most once -- the Rec's
     * flag says whether it is in -- and a key whose record changed under it is skipped when it is
     * reached rather than searched for, so both stay a push and a pop. */
    std::deque<BlockHash> wb_q_;       /* records wanting a host copy */
    std::deque<BlockHash> resume_q_;   /* resume points wanting a disk copy, served first */
    std::deque<BlockHash> dk_q_;       /* records wanting a disk copy, in the order they landed */
    std::vector<BlockHash> clean_q_;   /* records whose VRAM copy may now go */
    /* STORE SLOTS A DISK WRITE IS READING, and those the cache let go of meanwhile. Kept by slot
     * and not on the record, because the record can be forgotten while the write is out and the
     * slot is the cache's either way. A few entries at most: one write batch is out at a time. */
    std::vector<int32_t> ck_reading_;
    std::vector<int32_t> ck_free_after_;
    std::vector<int32_t> ck_frees_;    /* store slots a finished write released */
    /* COPIES THE ARENA REFUSED since the last maintenance plan, which is what that plan frees
     * room for. A count and not a list: any host-only entry makes room for any copy. */
    int64_t  starved_ = 0;
    /* A WRITE-BACK FOUND THE ARENA FULL AND NOTHING TO GIVE UP. Waiting again changes nothing until
     * a slot is freed or an entry's disk copy lands, so the copy is not planned until one does --
     * otherwise every tick pops the queue, finds no slot and puts it back. */
    bool     wb_stuck_ = false;
    std::unordered_map<uint64_t, Session> sessions_;
    /* THE DEATH SIGNAL FOR A ROW, and the reason the table is bounded. A published chain is only
     * ever reachable from position zero, so the moment the cache evicts its FIRST block the
     * conversation cannot be hit at any length and the row describes blocks that are not there.
     * The cache announces every eviction through forget(), which is the only notice this object
     * gets, so the root is indexed and the row goes with it. Several sessions share a root when
     * they share a system prompt and all of them are unreachable together, which is why the value
     * is a list. */
    std::unordered_map<BlockHash, std::vector<uint64_t>, BlockHashHash> roots_;
    std::string model_;
    std::function<void(const BlockHash&)> on_drop_;
    std::function<void(const BlockHash&)> on_ck_drop_;
    Stats    stats_;
    std::string report_;
};

}  /* namespace rad */
