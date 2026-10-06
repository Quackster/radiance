/* prefix.cpp -- see prefix.h. */
#include "mem/prefix.h"

#include "mem/kvtier.h"

#include <algorithm>
#include <cstring>

namespace rad {

/* ------------------------------------------------------------------ hashing */
/* A 128-bit chained mix rather than a cryptographic hash. A prefix cache whose store is on disk,
 * shared and long-lived is usually keyed by SHA-256, because a cheap hash's collision would serve
 * one caller another caller's context. We get the same guarantee more cheaply: every entry keeps
 * its BLOCK'S TOKEN IDS and compares them on a hit, so a collision is a miss rather than a wrong
 * answer, and the hash only has to be good enough to keep the bucket chains short. The SSD tier,
 * which is such a store, carries the token ids into the slot header for the same reason
 * (sstier.h).
 *
 * splitmix64's finaliser, which is a full 64-bit bijection -- so folding one token cannot lose the
 * bits of the previous one, which is the failure mode of a plain multiply-xor chain. */
static inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

std::string BlockHash::str() const { return fmt("%016llx%016llx", (unsigned long long)hi,
                                                (unsigned long long)lo); }

uint64_t mm_content_hash(const void* embedding, int64_t bytes) {
    const uint8_t* p = (const uint8_t*)embedding;
    uint64_t h = 0xcbf29ce484222325ull ^ mix64((uint64_t)bytes);
    int64_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w; std::memcpy(&w, p + i, 8);
        h = mix64(h ^ w);
    }
    uint64_t tail = 0;
    for (int64_t k = 0; i + k < bytes; ++k) tail |= (uint64_t)p[i + k] << (8 * k);
    return mix64(h ^ tail);
}

BlockHash chain_block_hash(const BlockHash& parent, const int32_t* toks, int32_t n,
                           const MMContentKey* mm, int32_t n_mm, int64_t block_begin) {
    /* The parent seeds both lanes. That is the whole chained-hash property: a block with identical
     * tokens under a different prefix lands somewhere else, so a block in the middle of another
     * conversation cannot be served here. */
    BlockHash h;
    h.hi = mix64(parent.hi ^ 0x243f6a8885a308d3ull);
    h.lo = mix64(parent.lo ^ 0x13198a2e03707344ull);
    h.hi = mix64(h.hi ^ (uint64_t)block_begin);

    for (int32_t i = 0; i < n; ++i) {
        uint64_t t = (uint64_t)(uint32_t)toks[i];
        h.lo = mix64(h.lo ^ t);
        h.hi = mix64(h.hi + h.lo + t);
    }

    /* Multimodal spans that overlap this block, folded by CONTENT plus the overlap offsets, so the
     * same image at the same position hashes the same and a different image never aliases behind
     * the identical placeholder token ids (spec §11). */
    const int64_t lo_pos = block_begin, hi_pos = block_begin + n;
    for (int32_t i = 0; i < n_mm; ++i) {
        const MMContentKey& m = mm[i];
        if (m.tok_end <= lo_pos || m.tok_begin >= hi_pos) continue;
        h.lo = mix64(h.lo ^ m.hash);
        h.hi = mix64(h.hi ^ mix64(m.hash + (uint64_t)(m.tok_begin - lo_pos) * 1000003ull
                                  + (uint64_t)(m.tok_end - lo_pos)));
    }
    /* Never let a block hash to zero: a zeroed BlockHash is the chain root and must be unique. */
    if (!h.valid()) h.lo = 1;
    return h;
}

/* ------------------------------------------------------------------ retention policies */

void GeometricBackoff::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    keep->clear();
    if (pos.empty()) return;
    const int64_t tip = (int64_t)pos.size() - 1;
    std::vector<char> mark(pos.size(), 0);
    mark[(size_t)tip] = 1;                    /* THE TIP IS ALWAYS KEPT */
    for (int64_t d = 1; tip - d >= 0; d *= 2) mark[(size_t)(tip - d)] = 1;
    for (size_t i = 0; i < pos.size(); ++i) if (mark[i]) keep->push_back(pos[i]);
}

void KeepTipOnly::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    keep->clear();
    if (!pos.empty()) keep->push_back(pos.back());
}

void KeepAll::select(const std::vector<int64_t>& pos, std::vector<int64_t>* keep) const {
    *keep = pos;
}

/* ------------------------------------------------------------------ configure */

int PrefixCache::configure(const Config& cfg, KVManager* kv, int64_t checkpoint_interval) {
    kv_ = kv;
    enabled_ = false;
    has_linear_ = false;
    block_size_ = 0;
    interval_ = 0;
    groups_.clear();
    map_.clear(); lru_.clear();
    ck_by_prefix_.clear(); ck_by_session_.clear();
    /* THE POLICY COMES FROM THE CONFIG, and set_policy() overrides it -- which is the order a
     * test wants and the order an operator wants. The three names are checked in config_parse, so
     * an unknown one cannot reach here and silently become the default. */
    if (cfg.checkpoint_policy == "tip-only")      policy_ = std::make_unique<KeepTipOnly>();
    else if (cfg.checkpoint_policy == "keep-all") policy_ = std::make_unique<KeepAll>();
    else                                          policy_ = std::make_unique<GeometricBackoff>();
    report_.clear();

    if (!kv) return RAD_E_INVAL;
    if (!cfg.prefix_cache) {
        report_ = "prefix cache: disabled by --no-prefix-cache\n";
        return RAD_OK;
    }

    /* The cache's granularity is the block size of the first FULL-attention group, because that is
     * the group whose content a prefix hit actually reuses. A paged group with a different block
     * size cannot be indexed at that granularity and is named in the report rather than silently
     * skipped -- a group that quietly stops being prefix-cached is a hit-rate regression nobody
     * can attribute. */
    for (const auto& p : kv->plans()) {
        if (p.kind == RAD_KV_FULL && block_size_ == 0) block_size_ = p.block_size;
        if (p.stateful()) has_linear_ = true;
    }
    if (block_size_ == 0) {
        for (const auto& p : kv->plans()) if (p.paged()) { block_size_ = p.block_size; break; }
    }
    if (block_size_ <= 0) {
        report_ = "prefix cache: no paged KV group to cache; disabled\n";
        RAD_WARN("%s", report_.c_str());
        return RAD_OK;
    }
    std::string skipped;
    for (const auto& p : kv->plans()) {
        if (!p.paged()) continue;
        /* A WINDOW GROUP CANNOT BE SERVED A PREFIX. Its blocks below the window are recycled, so
         * the block list a hit hands over does not describe it -- adopt() would take a prefix's
         * worth of blocks into a table the manager only ever grows to window_blocks, and the slot
         * mapping is computed against a first_block_pos that a hit does not set. The group is
         * simply not prefix-cached, which is correct and costs the sequences that hit a drafter
         * or a windowed layer whose cache starts where the hit ended. */
        if (p.kind == RAD_KV_WINDOW) {
            skipped += fmt("  group '%s' is windowed (%lld blocks) -- NOT cached, a hit's blocks "
                           "do not describe a cache whose floor moves\n",
                           p.name.c_str(), (long long)p.window_blocks);
            continue;
        }
        if (p.block_size == block_size_) groups_.push_back(p.index);
        else skipped += fmt("  group '%s' block %lld != cache granularity %lld -- NOT cached\n",
                            p.name.c_str(), (long long)p.block_size, (long long)block_size_);
    }

    interval_ = checkpoint_interval;
    if (has_linear_) {
        if (interval_ <= 0) {
            RAD_ERR("checkpoint_interval is %lld but the model has linear-attention state. A "
                    "0 interval means no checkpoint is ever written and every hit replays the "
                    "linear layers from zero.", (long long)interval_);
            return RAD_E_INVAL;
        }
        if (interval_ % block_size_ != 0) {
            /* A checkpoint that does not land on an attention block boundary can never be found:
             * the lookup identifies a checkpoint by the chained hash at its position, and that
             * hash only exists at block boundaries. vLLM's answer is to inflate the block to the
             * interval, which is what costs it attention hit rate on every short prompt. Ours is
             * to refuse the misconfiguration by name. */
            RAD_ERR("the checkpoint interval resolves to %lld (from --checkpoint-interval %lld, "
                    "rounded up to the chunk quantum) and that is not a multiple of the attention "
                    "block size %lld. The linear checkpoint is found by the chained block hash at "
                    "its position, and that hash exists only on a block boundary, so this interval "
                    "would never produce a single hit (spec §7.3).",
                    (long long)interval_, (long long)cfg.checkpoint_interval,
                    (long long)block_size_);
            return RAD_E_INVAL;
        }
    }

    enabled_ = true;
    report_ = fmt("prefix cache: %zu paged group(s) at %lld-token blocks; linear checkpoints "
                  "every %lld tokens (%s a rank x %s, pinned host memory), retention '%s'\n",
                  groups_.size(), (long long)block_size_,
                  (long long)(has_linear_ ? interval_ : 0),
                  humanb(kv->checkpoint_bytes()).c_str(),
                  fmt("%lld slots", (long long)kv->checkpoint_slots()).c_str(),
                  policy_->name());
    report_ += skipped;
    return RAD_OK;
}

void PrefixCache::set_policy(std::unique_ptr<CheckpointPolicy> p) {
    if (p) policy_ = std::move(p);
}

std::string PrefixCache::report() const { return report_; }

/* ------------------------------------------------------------------ full attention */

std::vector<BlockHash> PrefixCache::chain_all(const std::vector<int32_t>& tokens,
                                              const std::vector<MMContentKey>& mm,
                                              int64_t n_blocks) const {
    std::vector<BlockHash> hs;
    hs.reserve((size_t)n_blocks);
    BlockHash parent;
    for (int64_t b = 0; b < n_blocks; ++b) {
        parent = chain_block_hash(parent, tokens.data() + b * block_size_, (int32_t)block_size_,
                                  mm.data(), (int32_t)mm.size(), b * block_size_);
        hs.push_back(parent);
    }
    return hs;
}

void PrefixCache::touch(const BlockHash& h) const {
    auto it = map_.find(h);
    if (it == map_.end() || !it->second.in_lru) return;
    lru_.splice(lru_.end(), lru_, it->second.lru);
}

CacheHit PrefixCache::lookup(const std::vector<int32_t>& tokens,
                             const std::vector<MMContentKey>& mm,
                             std::vector<std::vector<int32_t>>* blocks_out,
                             int64_t cap_tokens) const {
    CacheHit hit;
    if (blocks_out) { blocks_out->assign(groups_.size(), {}); }
    if (!enabled_ || tokens.empty()) return hit;
    ++stats_.lookups;

    const int64_t n_full = (int64_t)tokens.size() / block_size_;
    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);

    int64_t matched = 0;
    for (; matched < n_full; ++matched) {
        auto it = map_.find(hs[(size_t)matched]);
        if (it == map_.end()) break;
        /* AN ENTRY THE TIERS HOLD IS NOT A HIT, and this is the guard that makes demotion safe at
         * all. Moving an entry off the device releases its KV blocks; the map keeps the entry so
         * the bytes can be found again, but its `blocks` now name blocks the pool has given to
         * somebody else. Serving them would hand one conversation another's context -- exactly
         * the failure the token compare below exists to rule out, arrived at from the other side.
         *
         * The prefix simply stops here. The caller may promote and ask again (see
         * restorable_from), and until it does the answer is an honest shorter hit. An entry with
         * a promotion still out on it stops the prefix too: its ids are the ones it left with. */
        if (!it->second.in_lru) break;
        /* Compare the token ids, not just the hash. A 128-bit collision is improbable; serving one
         * conversation's context to another is not a failure anybody would ever diagnose, so it is
         * made impossible rather than unlikely. */
        const int32_t* src = tokens.data() + matched * block_size_;
        if ((int64_t)it->second.tokens.size() != block_size_ ||
            std::memcmp(it->second.tokens.data(), src, (size_t)block_size_ * sizeof(int32_t)) != 0)
            break;
    }

    /* `cap_tokens` >= 0 ends the hit no later than that many tokens (rounded down to a block). The
     * checkpoint walk below starts from what is left, so a capped hit lands on a checkpoint at or
     * below the cap and the recurrent state stays exact. See Scheduler::set_draft_refill. */
    if (cap_tokens >= 0 && matched > cap_tokens / block_size_) matched = cap_tokens / block_size_;

    /* At least one token must remain for the step to compute a logit from. A full hit is backed
     * off by one block rather than being handed to a caller that would then schedule a zero-token
     * forward pass. */
    if (matched * block_size_ >= (int64_t)tokens.size() && matched > 0) --matched;

    for (int64_t b = 0; b < matched; ++b) {
        touch(hs[(size_t)b]);
        if (!blocks_out) continue;
        const Entry& e = map_.find(hs[(size_t)b])->second;
        for (size_t g = 0; g < groups_.size(); ++g)
            (*blocks_out)[g].push_back(e.blocks[g]);
    }
    /* A HIT IS A USE, and the tiers' idle clock has to hear about it: it is what decides when an
     * entry leaves the device, and without this it runs from when the chain was published, so a
     * prefix hit on every turn -- or one just promoted back for this very lookup -- would read as
     * idle and be demoted from under the request using it. */
    if (tiers_ && matched > 0) tiers_->note_hits(hs.data(), matched, rad_mono_ns());
    hit.n_attn_tokens = (int32_t)(matched * block_size_);
    stats_.block_hits += matched;
    stats_.block_misses += n_full - matched;

    /* ---- the split: how far the LINEAR state can be reused ----
     * The attention blocks are reusable up to n_attn_tokens. The recurrent state is only reusable
     * at a checkpoint: the deepest snapshot at or below n_attn_tokens on this chain, in its store
     * slot. The scheduler resumes from there (Scheduler::admit_one) and recomputes the tokens
     * between the two. */
    if (!has_linear_) {
        /* No recurrent state means nothing to replay: the whole attention hit is directly usable. */
        hit.n_linear_tokens = hit.n_attn_tokens;
        return hit;
    }
    /* OVER THE SNAPSHOTS, NOT OVER THE POSITIONS. A checkpoint sits on the interval or at the
     * tail of a prompt (Scheduler::plan_chunk), which is any block boundary, so the deepest one
     * under the hit is found by asking each snapshot whether this chain passes through it: its
     * key is the chained hash at its own position, which matches only on the same prefix. The
     * index holds a few hundred snapshots against tens of thousands of block positions. */
    auto best = ck_by_prefix_.end();
    for (auto it = ck_by_prefix_.begin(); it != ck_by_prefix_.end(); ++it) {
        const Ckpt& c = it->second;
        if (!c.valid || c.detached || c.slot < 0 || c.pos <= 0 || c.pos > hit.n_attn_tokens ||
            c.pos % block_size_ != 0)
            continue;
        if (best != ck_by_prefix_.end() && c.pos <= best->second.pos) continue;
        if (hs[(size_t)(c.pos / block_size_ - 1)] != it->first) continue;
        best = it;
    }
    if (best == ck_by_prefix_.end()) {
        ++stats_.ckpt_misses;
        return hit;
    }
    hit.n_linear_tokens = (int32_t)best->second.pos;
    hit.checkpoint_slot = best->second.slot;
    best->second.used = ++clock_;
    handed_.push_back(best->first);
    ++stats_.ckpt_hits;
    return hit;
}

int PrefixCache::insert(uint64_t session,
                        const std::vector<int32_t>& tokens,
                        const std::vector<MMContentKey>& mm,
                        const std::vector<std::vector<int32_t>>& blocks,
                        int64_t n_tokens) {
    if (!enabled_) return RAD_OK;
    if (blocks.size() != groups_.size()) return RAD_E_INVAL;
    if (n_tokens > (int64_t)tokens.size()) return RAD_E_INVAL;

    const int64_t n_full = n_tokens / block_size_;
    if (n_full <= 0) return RAD_OK;
    for (const auto& bt : blocks) if ((int64_t)bt.size() < n_full) return RAD_E_INVAL;

    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);
    const uint64_t now = rad_mono_ns();
    for (int64_t b = 0; b < n_full; ++b) {
        const BlockHash& h = hs[(size_t)b];
        auto it = map_.find(h);
        if (it != map_.end() && it->second.in_lru) { touch(h); continue; }
        if (it != map_.end()) {
            /* AN ENTRY WHOSE ONLY COPY IS OFF THE CARD, AND THIS REQUEST JUST COMPUTED IT AGAIN --
             * a restore that did not fit, or a chain cut short. The request's blocks become the
             * entry's device copy and the tier copy is let go (IdleTiers::rebound says why). One
             * with a promotion still out on it is left alone: its record is the transfer's. */
            if (!tiers_ || !tiers_->resident_off_device(h)) continue;
            std::vector<int32_t> mine(groups_.size(), -1);
            size_t held = 0;
            for (; held < groups_.size(); ++held) {
                const int32_t blk = blocks[held][(size_t)b];
                if (kv_->retain_block(groups_[held], blk) < 0) break;
                mine[held] = blk;
            }
            if (held != groups_.size()) {
                for (size_t g = 0; g < held; ++g) kv_->release_block(groups_[g], mine[g]);
                return RAD_E_STATE;
            }
            Entry& e = it->second;
            e.blocks = std::move(mine);
            lru_.push_back(h);
            e.lru = std::prev(lru_.end());
            e.in_lru = true;
            tiers_->rebound(h, e.blocks, now);
            continue;
        }

        Entry e;
        e.blocks.assign(groups_.size(), -1);
        size_t held = 0;
        for (; held < groups_.size(); ++held) {
            int32_t blk = blocks[held][(size_t)b];
            if (kv_->retain_block(groups_[held], blk) < 0) break;
            e.blocks[held] = blk;
        }
        if (held != groups_.size()) {
            /* Unwind exactly what was taken. A half-referenced entry leaks a block forever, and a
             * released block that was never retained frees one out from under a live sequence --
             * both are silent and neither is diagnosable later. */
            for (size_t g = 0; g < held; ++g) kv_->release_block(groups_[g], e.blocks[g]);
            return RAD_E_STATE;
        }
        e.tokens.assign(tokens.begin() + b * block_size_,
                        tokens.begin() + (b + 1) * block_size_);
        lru_.push_back(h);
        e.lru = std::prev(lru_.end());
        e.in_lru = true;
        map_.emplace(h, std::move(e));
    }

    /* THE TIERS LEARN THE CHAIN HERE AND NOWHERE ELSE. insert() is the one call that knows a chain
     * was COMPLETED rather than merely touched -- which is what a session is -- and it is also the
     * only place every entry's block ids are in hand at once.
     *
     * AND THE IDS COME OUT OF THE MAP, NOT OUT OF `blocks`. The two are the same for an entry this
     * call just created and DIFFERENT for one that already existed, because the loop above keeps
     * the entry the cache already holds and lets this request's own copy of those blocks go. Both
     * block tables describe the same tokens, but only the cache's is REFERENCED: the other belongs
     * to a sequence that is about to be freed.
     *
     * Recording the wrong one is not a bookkeeping detail. The tiers demote what they were told,
     * so they would gather out of blocks the pool had already handed to somebody else -- putting a
     * stranger's KV in the tier under this prefix's key -- and then release a reference nobody
     * took. It shows as a flood of "double free of block N" in the log and, once the entry is
     * promoted back, as fluent text with one word changed. */
    if (tiers_) {
        for (int64_t b = 0; b < n_full; ++b) {
            auto it = map_.find(hs[(size_t)b]);
            if (it == map_.end()) continue;
            tiers_->note_used(hs[(size_t)b], it->second.blocks, now);
        }
        tiers_->note_session(session, hs, n_tokens, now);
    }

    /* A CONVERSATION'S SNAPSHOTS BELONG TO ITS LATEST TURN. The index keys a session by REQUEST
     * id, so each turn would otherwise keep its own tip forever: turn N-1's deepest snapshot stays
     * a "tip" after turn N has resumed from it and written deeper ones, and the retention policy
     * then guards a dozen dead tips across a long agent run while thinning the one conversation
     * that is live. Every snapshot on the chain this request just published is a prefix of it,
     * and is moved under this id, so the policy sees one session with one tip. */
    for (auto& kvp : ck_by_prefix_) {
        Ckpt& c = kvp.second;
        if (c.session == session || c.pos <= 0 || c.pos > n_full * block_size_ ||
            c.pos % block_size_ != 0 || hs[(size_t)(c.pos / block_size_ - 1)] != kvp.first)
            continue;
        auto old = ck_by_session_.find(c.session);
        if (old != ck_by_session_.end()) {
            old->second.erase(std::remove(old->second.begin(), old->second.end(), kvp.first),
                              old->second.end());
            if (old->second.empty()) ck_by_session_.erase(old);
        }
        c.session = session;
        ck_by_session_[session].push_back(kvp.first);
    }
    return RAD_OK;
}


/* ------------------------------------------------------------------ idle session tiers */

std::vector<BlockHash> PrefixCache::restorable_from(const std::vector<int32_t>& tokens,
                                                    const std::vector<MMContentKey>& mm,
                                                    std::vector<BlockHash>* chain) const {
    std::vector<BlockHash> out;
    if (chain) chain->clear();
    if (!enabled_ || !tiers_ || tokens.empty()) return out;

    /* THE BLOCKS A HIT CAN USE: one token must remain to compute a logit from, so a prompt that
     * ends on a block boundary never reuses its last block (lookup backs off by one) and neither
     * restoring it nor bringing back a snapshot keyed there would be read. */
    const int64_t n_full = ((int64_t)tokens.size() - 1) / block_size_;
    std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);

    /* Walk exactly as lookup does, and keep walking past the point it stopped for as long as the
     * chain holds. EVERY off-device block in it is collected, not just the first run of them.
     *
     * STOPPING AT THE FIRST RESIDENT BLOCK PAST A GAP, on the reasoning that "the run this call is
     * about has ended", would be wrong, and the way it is wrong is worth stating because the
     * symptom does not look like a caching bug at all. A hit needs the WHOLE prefix:
     * restoring blocks 0..99 and stopping leaves block 200 off-device, so the hit ends at 200 and
     * everything above it is re-prefilled -- and WHICH blocks are off-device at the moment this
     * runs depends on how far the maintenance pass has got. The same prompt then gets a different
     * hit length on each restore.
     *
     * That is not only a lost prefix. The hit length is what the scheduler chunks the remaining
     * prefill against, so two restores that recovered different amounts run different GEMM shapes
     * over the same tokens -- and greedy decoding is tie-level sensitive to that (§19). The
     * session's answer moves between restores, intermittently, with nothing wrong in the KV at
     * all. Restoring the whole chain makes the hit length a property of the prompt. */
    for (int64_t b = 0; b < n_full; ++b) {
        auto it = map_.find(hs[(size_t)b]);
        if (it == map_.end()) break;
        const int32_t* src = tokens.data() + b * block_size_;
        if ((int64_t)it->second.tokens.size() != block_size_ ||
            std::memcmp(it->second.tokens.data(), src, (size_t)block_size_ * sizeof(int32_t)) != 0)
            break;
        if (chain) chain->push_back(hs[(size_t)b]);
        if (tiers_->resident_off_device(hs[(size_t)b])) out.push_back(hs[(size_t)b]);
    }
    return out;
}

void PrefixCache::drop_offdevice(const BlockHash& key) {
    auto it = map_.find(key);
    if (it == map_.end()) return;
    /* NO release_block HERE, and that is the whole point. The entry has been off-device since it
     * was demoted; the ids in it name blocks the pool reassigned long ago, so releasing them
     * would free somebody else's context and serving them would hand it over. */
    if (it->second.in_lru) lru_.erase(it->second.lru);
    map_.erase(it);
    ++stats_.evictions;
    /* AND THE SNAPSHOT KEYED BY THE SAME HASH, when the disk was all it had: its record is gone,
     * and an index entry left behind would be a detached snapshot nobody can read back. One still
     * in its store slot stays -- no lookup reaches it past the missing block, the slot may be
     * under a disk write this sink cannot ask about, and the retention policy retires it through
     * the path that does ask (retire_ckpt). */
    auto ck = ck_by_prefix_.find(key);
    if (ck != ck_by_prefix_.end() && ck->second.detached) drop_snapshot(key);
}

int PrefixCache::rebind(const BlockHash& key, const std::vector<int32_t>& blocks) {
    auto it = map_.find(key);
    if (it == map_.end()) return RAD_E_NOTFOUND;
    if (blocks.size() != groups_.size()) return RAD_E_SHAPE;
    for (int32_t b : blocks) if (b < 0) return RAD_E_INVAL;
    /* THE RESERVATION'S REFERENCE BECOMES THE CACHE'S. KVManager::reserve_block left each block at
     * a refcount of one with no sequence attached; assigning them here is the handover, and the
     * cache's ordinary release path frees them from now on. Retaining and then releasing to "make
     * it explicit" would be a round trip to the same number and would only obscure which reference
     * is which. */
    Entry& e = it->second;
    e.blocks = blocks;
    for (size_t g = 0; g < groups_.size(); ++g) kv_->cache_reserved_block(groups_[g], blocks[g]);
    if (!e.in_lru) {
        lru_.push_back(key);
        e.lru = std::prev(lru_.end());
        e.in_lru = true;
    }
    return RAD_OK;
}

bool PrefixCache::shared_with_live(const Entry& e) const {
    for (size_t g = 0; g < groups_.size(); ++g)
        if (kv_->block_refcount(groups_[g], e.blocks[g]) > 1) return true;
    return false;
}

int64_t PrefixCache::evict_lru(int64_t n) {
    int64_t done = 0;
    /* ONE PASS OVER WHAT WAS ON THE LIST WHEN THE CALL BEGAN. An entry a live sequence also holds
     * goes to the back rather than being evicted: releasing the cache's reference on it frees no
     * block, and an entry that had no host copy would be erased for nothing. It is in use, which is
     * what the back of the list means, and moving it is what keeps a caller that asks again from
     * walking the same shared run every time. */
    for (int64_t seen = (int64_t)lru_.size(); done < n && seen > 0 && !lru_.empty(); --seen) {
        const BlockHash h = lru_.front();
        auto it = map_.find(h);
        if (it == map_.end()) { lru_.pop_front(); continue; }
        Entry& e = it->second;
        if (shared_with_live(e)) {
            lru_.splice(lru_.end(), lru_, lru_.begin());
            continue;
        }
        lru_.pop_front();
        e.in_lru = false;
        for (size_t g = 0; g < groups_.size(); ++g)
            kv_->release_block(groups_[g], e.blocks[g]);
        if (tiers_) {
            /* THE HOST COPY HAS LANDED: the entry leaves VRAM and stays in the index. */
            if (tiers_->droppable(h)) {
                tiers_->drop_commit(h);
                ++stats_.drops;
                ++done;
                continue;
            }
            tiers_->forget(h);
        }
        map_.erase(it);
        ++done;
        ++stats_.evictions;
    }
    return done;
}

int64_t PrefixCache::trim(int64_t n_blocks) {
    /* TIERS ATTACHED AND OFF are no tiers: nothing becomes droppable, and waiting for a copy that
     * nobody will make would leave the reserve unkept for good. */
    const bool keep = tiers_ && tiers_->any_on();
    int64_t done = 0;
    auto it = lru_.begin();
    for (int64_t seen = (int64_t)lru_.size(); it != lru_.end() && done < n_blocks && seen > 0;
         --seen) {
        const BlockHash h = *it;
        auto m = map_.find(h);
        if (m == map_.end()) { it = lru_.erase(it); continue; }
        Entry& e = m->second;
        if (shared_with_live(e)) {
            /* In use: to the back, as evict_lru does, and the walk carries on from the next. */
            auto next = std::next(it);
            lru_.splice(lru_.end(), lru_, it);
            it = next;
            continue;
        }
        if (keep && !tiers_->droppable(h)) { ++it; continue; }
        it = lru_.erase(it);
        e.in_lru = false;
        for (size_t g = 0; g < groups_.size(); ++g)
            kv_->release_block(groups_[g], e.blocks[g]);
        if (keep) {
            tiers_->drop_commit(h);
            ++stats_.drops;
        } else {
            if (tiers_) tiers_->forget(h);
            map_.erase(m);
            ++stats_.evictions;
        }
        ++done;
    }
    return done;
}

int64_t PrefixCache::drop_clean() {
    if (!tiers_ || !kv_) return 0;
    drop_scratch_.clear();
    tiers_->take_clean(&drop_scratch_);
    const uint64_t frees = kv_->seq_frees();
    if (frees != frees_seen_ || released_) {
        frees_seen_ = frees;
        released_ = false;
        drop_scratch_.insert(drop_scratch_.end(), drop_wait_.begin(), drop_wait_.end());
        drop_wait_.clear();
    }
    int64_t n = 0;
    for (const BlockHash& h : drop_scratch_) {
        auto it = map_.find(h);
        if (it == map_.end()) continue;
        Entry& e = it->second;
        e.drop_wait = false;
        if (!e.in_lru || !tiers_->droppable(h)) continue;
        /* A BLOCK A RUNNING SEQUENCE ALSO HOLDS WOULD FREE NOTHING, and its entry would then be
         * off-device in the index while its bytes sat in VRAM under somebody else's table -- the
         * next lookup would restore a second copy beside them. */
        bool shared = held_.count(h) > 0;
        for (size_t g = 0; g < groups_.size() && !shared; ++g)
            shared = kv_->block_refcount(groups_[g], e.blocks[g]) > 1;
        if (shared) {
            if (!e.drop_wait) { e.drop_wait = true; drop_wait_.push_back(h); }
            continue;
        }
        for (size_t g = 0; g < groups_.size(); ++g)
            kv_->release_block(groups_[g], e.blocks[g]);
        lru_.erase(e.lru);
        e.in_lru = false;
        ++stats_.drops;
        tiers_->drop_commit(h);
        ++n;
    }
    return n;
}

/* THE REFCOUNT IS WHAT SEPARATES A CACHE FROM A LEAK. A block the cache indexes and a running
 * sequence is reading is not memory the cache could give back; a block only the cache references
 * is exactly what the next eviction frees. Nothing else distinguishes them -- both are `used`,
 * both are in the index -- so the count is the manager's: it marks the blocks the cache holds and
 * keeps the number of marked blocks at a reference count of one current as the counts move.
 *
 * ONE GROUP, FOR THE REASON THE POOL'S OWN TOKEN FIGURES ARE ONE GROUP'S: the paged groups share
 * a token axis and an entry is one block in each of them, so counting every group would multiply
 * the answer by the group count.
 *
 * OFF-DEVICE ENTRIES COUNT FOR NOTHING. A demoted entry released its blocks, and the mark went
 * with the release; its bytes are on the host or on disk and it is holding none of the card. */
int64_t PrefixCache::unshared_tokens() const {
    if (!kv_ || groups_.empty() || block_size_ <= 0) return 0;
    return kv_->cache_only_blocks(groups_[0]) * block_size_;
}

void PrefixCache::clear() {
    for (auto& kv : map_) {
        if (kv.second.in_lru)
            for (size_t g = 0; g < groups_.size(); ++g)
                kv_->release_block(groups_[g], kv.second.blocks[g]);
        if (tiers_) tiers_->forget(kv.first);
        ++stats_.evictions;
    }
    map_.clear();
    lru_.clear();
    drop_wait_.clear();
}

std::vector<BlockHash> PrefixCache::lru_order() const {
    return std::vector<BlockHash>(lru_.begin(), lru_.end());
}

/* ------------------------------------------------------------------ checkpoints */

int64_t PrefixCache::next_checkpoint_after(int64_t pos) const {
    if (interval_ <= 0) return -1;
    if (pos < 0) pos = 0;
    return (pos / interval_ + 1) * interval_;
}

std::vector<int64_t> PrefixCache::checkpoints(uint64_t session) const {
    std::vector<int64_t> out;
    auto it = ck_by_session_.find(session);
    if (it == ck_by_session_.end()) return out;
    for (const BlockHash& h : it->second) {
        auto e = ck_by_prefix_.find(h);
        if (e != ck_by_prefix_.end()) out.push_back(e->second.pos);
    }
    std::sort(out.begin(), out.end());
    return out;
}

int64_t PrefixCache::retain(uint64_t session) {
    auto it = ck_by_session_.find(session);
    if (it == ck_by_session_.end()) return 0;

    /* SNAPSHOTS IN THE STORE ONLY. This policy exists to spend a fixed number of --checkpoint-
     * slots well, and it runs when the store is exhausted. A detached snapshot occupies none of
     * it, so retiring one would free nothing here and would cost a session the state it was
     * written to disk precisely in order to keep. The disk store bounds its own, oldest first,
     * which is the right policy for the resource it actually spends.
     *
     * It stays in `kept` rather than being skipped, or the session index would forget it. */
    std::vector<std::pair<int64_t, BlockHash>> all;
    std::vector<BlockHash> kept;
    for (const BlockHash& h : it->second) {
        auto e = ck_by_prefix_.find(h);
        if (e == ck_by_prefix_.end()) continue;
        if (e->second.detached) { kept.push_back(h); continue; }
        all.push_back({e->second.pos, h});
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<int64_t> pos;
    for (const auto& a : all) pos.push_back(a.first);
    std::vector<int64_t> keep;
    policy_->select(pos, &keep);

    int64_t freed = 0;
    for (const auto& a : all) {
        if (std::binary_search(keep.begin(), keep.end(), a.first)) { kept.push_back(a.second); continue; }
        auto e = ck_by_prefix_.find(a.second);
        if (e == ck_by_prefix_.end()) continue;
        /* A SNAPSHOT THE DISK HOLDS LEAVES THE SLOT AND STAYS IN THE INDEX: the policy is about
         * the slots, and that copy costs none of them. */
        if (tiers_ && tiers_->snapshot_backed(a.second)) {
            (void)detach_snapshot(a.second);
            tiers_->snapshot_detached(a.second);
            kept.push_back(a.second);
            ++freed;
            continue;
        }
        retire_ckpt(e);
        ++freed;
        ++stats_.ckpt_evictions;
    }
    /* ERASE THE SESSION WHEN NOTHING IS LEFT OF IT, and that is not tidiness. `session` is a
     * REQUEST id, so without an eraser this map gains an entry for every request that ever writes
     * a checkpoint: the vector empties, here and in the eviction below, and the key stays. The
     * cost is not the bytes -- it is that the loop in reserve_checkpoint walks EVERY key on every
     * allocation failure, which is the one path this cache exists to relieve. */
    if (kept.empty()) { ck_by_session_.erase(it); return freed; }
    it->second = std::move(kept);
    return freed;
}

int PrefixCache::reserve_checkpoint(uint64_t session, int64_t pos, const BlockHash& prefix,
                                    int32_t* slot_out) {
    if (!enabled_ || !has_linear_) return RAD_E_UNSUPPORTED;
    if (!slot_out) return RAD_E_INVAL;
    /* ON A BLOCK BOUNDARY, because the key is the chained hash of whole blocks. The scheduler
     * places checkpoints on the interval and at the tail of a prompt, both on its chunk quantum,
     * which is a multiple of the block. */
    if (interval_ <= 0 || pos <= 0 || pos % block_size_ != 0) {
        RAD_ERR("checkpoint at %lld is not on a %lld-token block boundary; the scheduler splits "
                "chunks on its quantum precisely so this cannot happen (spec §7.3)",
                (long long)pos, (long long)block_size_);
        return RAD_E_INVAL;
    }
    auto existing = ck_by_prefix_.find(prefix);
    if (existing != ck_by_prefix_.end()) {
        /* A DETACHED SNAPSHOT IS NOT ONE THIS CALL CAN HAND BACK. Returning its -1 slot would
         * look to the scheduler exactly like a refusal (it tests `cs >= 0`) and the chunk would
         * silently stop checkpointing; handing back a slot that is not in the store would be
         * worse. The state is about to be recomputed anyway -- that is what a reservation at this
         * prefix means -- so the disk copy is retired and a fresh slot allocated below. */
        /* NOR ONE A DISK WRITE IS READING. The save that follows this reservation writes the slot
         * while the write reads it, and the disk would keep a snapshot that is half of each. A
         * fresh slot costs one allocation. */
        if (!existing->second.detached &&
            !(tiers_ && tiers_->slot_reading(existing->second.slot))) {
            *slot_out = existing->second.slot;
            return RAD_OK;
        }
        const uint64_t vs = existing->second.session;
        retire_ckpt(existing);
        auto sit = ck_by_session_.find(vs);
        if (sit != ck_by_session_.end()) {
            sit->second.erase(std::remove(sit->second.begin(), sit->second.end(), prefix),
                              sit->second.end());
            if (sit->second.empty()) ck_by_session_.erase(sit);
        }
    }

    int32_t slot = -1;
    if (kv_->alloc_checkpoint(&slot) < 0) {
        /* Budgeted retention: run the policy over every session before giving up. The tip of each
         * session survives it by construction, because that is what an appending agent transcript
         * needs (spec §7.3).
         *
         * Over a SNAPSHOT of the keys, because retain() erases a session it empties and a
         * range-for over the map would be iterating a container it is erasing from. */
        std::vector<uint64_t> sessions;
        sessions.reserve(ck_by_session_.size());
        for (const auto& s : ck_by_session_) sessions.push_back(s.first);
        for (uint64_t s : sessions) retain(s);
        if (kv_->alloc_checkpoint(&slot) < 0) {
            /* Still nothing. Drop the globally least-recently-used snapshot, PREFERRING a non-tip
             * one -- but falling back to a tip rather than giving up.
             *
             * MAKING A TIP UNEVICTABLE OUTRIGHT WEDGES THE POOL PERMANENTLY. A session here is a
             * REQUEST id (PCBinding passes r.id), and a finished request's checkpoints are meant
             * to outlive it -- they are keyed by chained prefix hash, which is exactly how the
             * next turn of an agent transcript finds them. So every request that writes a
             * checkpoint leaves a tip behind and nothing ever retires one: after
             * --checkpoint-slots such requests, every slot holds a dead session's tip and this
             * would return RAD_E_FULL for the life of the process. The scheduler ignores the
             * failure (scheduler.cpp: `>= 0 && cs >= 0`), so linear prefix caching would simply
             * stop, with nothing in the log.
             *
             * Two passes rather than one comparison, because "prefer a non-tip" is a tier and not
             * a tiebreak: the oldest tip must lose to the newest non-tip. */
            const Ckpt* victim = nullptr;
            const Ckpt* tip_victim = nullptr;
            for (const auto& kvp : ck_by_prefix_) {
                /* A DETACHED SNAPSHOT IS NOT A CANDIDATE, because evicting it frees nothing HERE.
                 * Its slot went back to the store when it was detached, so picking it as the
                 * victim would retire a usable snapshot and then fail the retry anyway. */
                if (kvp.second.detached) continue;
                /* NOR ONE A DISK WRITE IS READING: its slot comes back only when the write
                 * reports, so retiring it now loses the snapshot and frees nothing yet. */
                if (tiers_ && tiers_->slot_reading(kvp.second.slot)) continue;
                auto sit = ck_by_session_.find(kvp.second.session);
                bool is_tip = false;
                if (sit != ck_by_session_.end()) {
                    int64_t tip = -1;
                    for (const BlockHash& h : sit->second) {
                        auto e = ck_by_prefix_.find(h);
                        if (e != ck_by_prefix_.end()) tip = std::max(tip, e->second.pos);
                    }
                    is_tip = (kvp.second.pos == tip);
                }
                const Ckpt*& best = is_tip ? tip_victim : victim;
                if (!best || kvp.second.used < best->used) best = &kvp.second;
            }
            if (!victim) victim = tip_victim;
            if (!victim) return RAD_E_FULL;
            BlockHash vh = victim->prefix;
            uint64_t vs = victim->session;
            /* A VICTIM THE DISK HOLDS IS DETACHED, NOT LOST. The slot goes back all the same; the
             * index entry stays, and the session's next restore reads the state back instead of
             * replaying every recurrent layer from an older point. */
            if (tiers_ && tiers_->snapshot_backed(vh)) {
                (void)detach_snapshot(vh);
                tiers_->snapshot_detached(vh);
                if (kv_->alloc_checkpoint(&slot) < 0) return RAD_E_FULL;
                Ckpt c;
                c.session = session; c.pos = pos; c.prefix = prefix; c.slot = slot;
                c.used = ++clock_;
                ck_by_prefix_.emplace(prefix, c);
                ck_by_session_[session].push_back(prefix);
                *slot_out = slot;
                return RAD_OK;
            }
            retire_ckpt(ck_by_prefix_.find(vh));
            ++stats_.ckpt_evictions;
            auto sit = ck_by_session_.find(vs);
            if (sit != ck_by_session_.end()) {
                sit->second.erase(std::remove(sit->second.begin(), sit->second.end(), vh),
                                  sit->second.end());
                if (sit->second.empty()) ck_by_session_.erase(sit);
            }
            if (kv_->alloc_checkpoint(&slot) < 0) return RAD_E_FULL;
        }
    }

    Ckpt c;
    c.session = session; c.pos = pos; c.prefix = prefix; c.slot = slot; c.used = ++clock_;
    ck_by_prefix_.emplace(prefix, c);
    ck_by_session_[session].push_back(prefix);
    *slot_out = slot;
    return RAD_OK;
}

void PrefixCache::commit_checkpoint(const BlockHash& prefix, bool resume) {
    auto it = ck_by_prefix_.find(prefix);
    if (it == ck_by_prefix_.end()) return;
    /* One that lost its slot between the reservation and now holds nothing to serve. */
    if (it->second.detached || it->second.slot < 0) return;
    it->second.valid = true;
    /* THE TIERS LEARN THE SNAPSHOT HERE, for the same reason note_used tells them the block ids:
     * a demotion has to know what to move and this index is the only place it is written down.
     * Before the commit there is nothing to move -- the slot still holds the previous tenant's
     * state -- so this is also the earliest moment at which telling them would be true. */
    if (tiers_) tiers_->note_snapshot(prefix, it->second.slot, it->second.pos, resume);
}


/* ------------------------------------------------------------------ the linear half of a session */

void PrefixCache::retire_ckpt(std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator it) {
    if (it == ck_by_prefix_.end()) return;
    /* THE TIERS ARE ASKED FIRST, because a disk write may be reading the slot: a save into it
     * before the write is done would put half of another snapshot on disk under this key. When
     * one is, the tiers keep the slot and hand it back once the write has reported. */
    const bool held = tiers_ && tiers_->release_snapshot(it->first, it->second.slot);
    if (it->second.slot >= 0 && !held) kv_->free_checkpoint(it->second.slot);
    /* TOLD EITHER WAY, AND NOT ONLY WHEN THE DISK HOLDS THE BYTES. The tiers also remember a
     * snapshot's slot -- note_snapshot records it at commit, because a disk write has to know
     * what to read and this index is the only place it is written down. Freeing that slot here
     * without saying so leaves the tiers naming a slot the store has since given to another
     * session, and the next write puts THAT session's recurrent state on disk as this one's. The
     * text that comes back is fluent, deterministic and about the wrong conversation, which is
     * the exact failure the whole tier design is arranged to make impossible.
     *
     * Not reachable from the tiers' own drop sink, which erases the index entry through
     * drop_snapshot instead -- so this never calls back into a record being erased. */
    ck_by_prefix_.erase(it);
}

int PrefixCache::detach_snapshot(const BlockHash& key) {
    auto it = ck_by_prefix_.find(key);
    if (it == ck_by_prefix_.end()) return RAD_E_NOTFOUND;
    if (it->second.detached) return RAD_OK;          /* already only on disk */
    /* Only ever asked of a snapshot the tiers call backed, which no write is reading. */
    if (it->second.slot >= 0) kv_->free_checkpoint(it->second.slot);
    it->second.slot = -1;
    it->second.detached = true;
    /* NOT SERVABLE FROM HERE ON, which is the same state a reserved-but-unwritten snapshot is in
     * and is handled by the same branch in lookup(). A hit that reaches this position simply walks
     * back to an older checkpoint, or to none -- shorter, never wrong. */
    it->second.valid = false;
    return RAD_OK;
}

int PrefixCache::reattach_snapshot(const BlockHash& key, int32_t slot) {
    auto it = ck_by_prefix_.find(key);
    if (it == ck_by_prefix_.end()) return RAD_E_NOTFOUND;
    if (slot < 0) return RAD_E_INVAL;
    /* AN ENTRY ALREADY IN A SLOT keeps it: taking the new one would leak the old. The caller gives
     * the new slot back. */
    if (!it->second.detached && it->second.slot >= 0) return it->second.slot == slot ? RAD_OK
                                                                                     : RAD_E_STATE;
    /* A SLOT THIS ENTRY DID NOT ASK FOR IS STILL ITS OWN. The read allocated it and filled it
     * from the disk copy of THIS key's state, so the identity that matters -- which prefix the
     * bytes describe -- is carried by the key and not by the slot number it happened to get. */
    it->second.slot = slot;
    it->second.detached = false;
    it->second.valid = true;
    it->second.used = ++clock_;
    return RAD_OK;
}

bool PrefixCache::free_checkpoint_for_restore() {
    if (!tiers_) return false;
    const Ckpt* victim = nullptr;
    for (const auto& kvp : ck_by_prefix_) {
        const Ckpt& c = kvp.second;
        if (c.detached || c.slot < 0 || !c.valid) continue;
        if (victim && c.used >= victim->used) continue;
        if (std::find(handed_.begin(), handed_.end(), kvp.first) != handed_.end()) continue;
        if (!tiers_->snapshot_backed(kvp.first)) continue;
        victim = &c;
    }
    if (!victim) return false;
    const BlockHash vh = victim->prefix;
    (void)detach_snapshot(vh);
    tiers_->snapshot_detached(vh);
    return true;
}

void PrefixCache::hold(uint64_t req, const std::vector<int32_t>& tokens,
                       const std::vector<MMContentKey>& mm) {
    if (!enabled_ || !tiers_ || tokens.empty() || holds_.count(req)) return;
    const int64_t n_full = ((int64_t)tokens.size() - 1) / block_size_;
    const std::vector<BlockHash> hs = chain_all(tokens, mm, n_full);
    std::vector<BlockHash>& mine = holds_[req];
    for (int64_t b = 0; b < n_full; ++b) {
        auto it = map_.find(hs[(size_t)b]);
        if (it == map_.end()) break;
        const int32_t* src = tokens.data() + b * block_size_;
        if ((int64_t)it->second.tokens.size() != block_size_ ||
            std::memcmp(it->second.tokens.data(), src, (size_t)block_size_ * sizeof(int32_t)) != 0)
            break;
        mine.push_back(hs[(size_t)b]);
        ++held_[hs[(size_t)b]];
    }
}

size_t PrefixCache::resident_prefix(const std::vector<BlockHash>& chain) const {
    size_t n = 0;
    for (; n < chain.size(); ++n) {
        auto it = map_.find(chain[n]);
        if (it == map_.end() || !it->second.in_lru) break;
    }
    return n;
}

void PrefixCache::release(uint64_t req) {
    auto it = holds_.find(req);
    if (it == holds_.end()) return;
    for (const BlockHash& h : it->second) {
        auto k = held_.find(h);
        if (k != held_.end() && --k->second <= 0) held_.erase(k);
    }
    holds_.erase(it);
    released_ = true;
}

void PrefixCache::drop_snapshot(const BlockHash& key) {
    auto it = ck_by_prefix_.find(key);
    if (it == ck_by_prefix_.end()) return;
    const uint64_t session = it->second.session;
    /* No release_snapshot on this path: the tiers are the caller, holding their lock, and the
     * bytes are already gone. They say this only of a detached snapshot, so there is normally no
     * slot to give back. */
    if (it->second.slot >= 0) kv_->free_checkpoint(it->second.slot);
    ck_by_prefix_.erase(it);
    ++stats_.ckpt_evictions;
    auto sit = ck_by_session_.find(session);
    if (sit == ck_by_session_.end()) return;
    sit->second.erase(std::remove(sit->second.begin(), sit->second.end(), key), sit->second.end());
    if (sit->second.empty()) ck_by_session_.erase(sit);
}

int64_t PrefixCache::snapshots_detached() const {
    int64_t n = 0;
    for (const auto& kvp : ck_by_prefix_) if (kvp.second.detached) ++n;
    return n;
}
}  /* namespace rad */
