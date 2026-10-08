/* rad_bufplan.cpp -- the buffer plan (spec §3.1).
 *
 * Activations are declared, liveness is computed over the declared op list, and non-overlapping
 * lifetimes are packed into one arena sized once. The alternative is an allocator on the hot
 * path, and an allocator on the hot path is the thing this engine exists to remove.
 *
 * Two conservatisms are deliberate and both are stated in the spec: the arena is sized for
 * max_tok, and liveness is computed over the declared list, which is a superset of what any given
 * step issues. A plan that is correct for the worst step is correct for every step.
 *
 * The packer is greedy interval colouring, largest first: for each buffer in descending size
 * order, take the lowest aligned offset that collides with no already-placed buffer whose
 * lifetime overlaps. It is O(n^2) in the buffer count at declare, which for a few hundred
 * activations is microseconds, and it is within a few percent of optimal on the shapes that
 * matter -- the one big activation is placed first and everything else fills in around it.
 * Optimal packing here is NP-hard and the difference is not worth a solver.
 */
#include "rad_build.h"
#include "rad_internal.h"

#include <algorithm>
#include <climits>

namespace rad {

/* Every buffer starts at a 256-byte boundary: the container's sub-block alignment, and wide
 * enough for any vector load an RDNA4 kernel issues (dwordx4 is 16 B). A kernel that needs more
 * than this from its scratch says so through its scratch hook, which is sized separately. */
static constexpr int64_t kBufAlign = (int64_t)RAD_ALIGN_SUB;
/* Keep the bulk activations on a 4 KiB boundary independently of small sampler/state planes.
 * Moving this base by a few sub-blocks can change device throughput even when every kernel and
 * tensor shape is unchanged. Individual buffers still need only the sub-block alignment above.
 * The padding stays in the fixed region so smaller-step plans and arena lending retain it. */
static constexpr int64_t kTransientBaseAlign = (int64_t)RAD_ALIGN_UNIT;

#include <cstdlib>

namespace {

struct Item {
    size_t  idx;      /* index into Program::buffers */
    int64_t bytes;    /* already rounded up to kBufAlign */
    int32_t lo, hi;   /* inclusive op-index lifetime */
    int64_t off = -1;
};

bool overlaps(const Item& a, const Item& b) { return a.lo <= b.hi && b.lo <= a.hi; }

/* Lowest aligned offset at or above 0 that fits `need` bytes without colliding with anything in
 * `placed` that is alive at the same time. */
int64_t first_fit(const Item& want, const std::vector<Item>& placed) {
    std::vector<std::pair<int64_t, int64_t>> occ;
    occ.reserve(placed.size());
    for (const Item& p : placed)
        if (p.off >= 0 && overlaps(p, want)) occ.emplace_back(p.off, p.off + p.bytes);
    std::sort(occ.begin(), occ.end());

    int64_t off = 0;
    for (auto& [s, e] : occ) {
        if (s >= off + want.bytes) break;      /* the gap below this block is big enough */
        off = std::max(off, align_up(e, kBufAlign));
    }
    return off;
}

}  /* namespace */

int plan_buffers(Program& p, BufPlanReport* rep) {
    BufPlanReport r;
    int status = RAD_OK;

    const int32_t last_op = (int32_t)p.ops.size() - 1;

    std::vector<Item> transient, persist, host;
    for (size_t i = 1; i < p.buffers.size(); ++i) {
        BufferInfo& b = p.buffers[i];
        if (b.bytes <= 0) b.bytes = buffer_bytes(b.decl);
        if (b.bytes <= 0) {
            RAD_ERR("buffer plan: '%s' cannot be sized (dtype %u, rank %u)", b.name.c_str(),
                    b.decl.dtype, b.decl.rank);
            status = RAD_E_INVAL;
            continue;
        }

        Item it{ i, align_up(b.bytes, kBufAlign), b.first_def, b.last_use };
        if (it.lo < 0 || it.hi < it.lo) {
            /* Live for the whole program: the sound assumption when nothing declared a use. */
            it.lo = 0;
            it.hi = last_op > 0 ? last_op : 0;
            if (b.decl.kind == RAD_BUF_TRANSIENT) ++r.n_unknown_live;
        }

        /* The two domains are coloured SEPARATELY and each offset is relative to its own arena.
         * A host-site activation cannot index device memory, so interleaving them in one offset
         * space would leave both arenas sparse -- correct, and paying VRAM for host activations.
         * Program::arena_bytes is the device high-water mark, host_arena_bytes the host one. */
        if (b.decl.domain == RAD_DOMAIN_HOST) host.push_back(it);
        else if (b.decl.kind == RAD_BUF_TRANSIENT) transient.push_back(it);
        else persist.push_back(it);
    }

    /* Largest first, ties broken by lifetime start and then by index, so the plan is the same on
     * every rank and every run -- a placement that varies between ranks is a debugging nightmare
     * for a bug that only shows up as a numerical difference. */
    auto by_size = [](const Item& a, const Item& b) {
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        if (a.lo != b.lo) return a.lo < b.lo;
        return a.idx < b.idx;
    };
    std::sort(transient.begin(), transient.end(), by_size);
    std::sort(host.begin(), host.end(), by_size);

    for (const Item& t : transient) r.sum_bytes += t.bytes;

    /* peak_live is what a perfect packer would reach, so the pair (peak_live, transient_bytes)
     * brackets the arena and says whether the greedy pass left anything on the table. */
    for (int32_t t = 0; t <= std::max<int32_t>(last_op, 0); ++t) {
        int64_t live = 0;
        for (const Item& it : transient) if (it.lo <= t && t <= it.hi) live += it.bytes;
        r.peak_live = std::max(r.peak_live, live);
    }

    /* PERSIST survives across steps and DERIVED is written once per step by the core before any
     * op runs, so neither may share with a transient or with each other: both are live for every
     * op that could read them.
     *
     * THEY GO AT THE BOTTOM, and the transients above them. A step smaller than the largest packs
     * its transients smaller (plan_level), and what it does not reach at the TOP of the arena is
     * lent to the expert slab; the fixed region has to be where no level moves it, because what
     * it holds outlives the step that wrote it. */
    int64_t off = 0;
    for (const Item& it : persist) {
        p.buffers[it.idx].arena_offset = off;
        off += it.bytes;
        off = align_up(off, kBufAlign);
    }
    const int64_t fixed = align_up(off, transient.empty() ? kBufAlign : kTransientBaseAlign);
    r.persist_bytes = fixed;

    std::vector<Item> placed;
    placed.reserve(transient.size());
    int64_t tr_end = 0;
    for (Item it : transient) {
        it.off = first_fit(it, placed);
        tr_end = std::max(tr_end, it.off + it.bytes);
        p.buffers[it.idx].arena_offset = fixed + it.off;
        placed.push_back(it);
    }
    r.transient_bytes = tr_end;

    p.arena_fixed_bytes = fixed;
    p.arena_bytes = fixed + tr_end;

    /* The host arena is its own address space and its own base pointer. Offsets below are relative
     * to it, not to Program::arena_bytes. */
    std::vector<Item> host_placed;
    int64_t host_end = 0;
    for (Item it : host) {
        /* One pass, same packer: a host PERSIST is live for the whole program by construction, so
         * first_fit already refuses to share it with anything. */
        it.off = first_fit(it, host_placed);
        host_end = std::max(host_end, it.off + it.bytes);
        p.buffers[it.idx].arena_offset = it.off;
        host_placed.push_back(it);
    }
    r.host_bytes = host_end;
    p.host_arena_bytes = host_end;

    if (rep) *rep = r;
    return status;
}

/* ------------------------------------------------------------------ a smaller step's plan */

int plan_level(const Program& full, const Program& probe, int64_t rows, ArenaLevel* out,
               std::string* why) {
    auto refuse = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_STATE;
    };
    if (probe.buffers.size() != full.buffers.size())
        return refuse(fmt("the sizing declare named %zu buffers and the real one %zu",
                          probe.buffers.size() - 1, full.buffers.size() - 1));

    ArenaLevel lv;
    lv.rows = rows;
    lv.offset.assign(full.buffers.size(), -1);
    lv.decl.assign(full.buffers.size(), RadBufDecl{});

    std::vector<Item> transient;
    for (size_t i = 1; i < full.buffers.size(); ++i) {
        const BufferInfo& fb = full.buffers[i];
        const BufferInfo& pb = probe.buffers[i];
        const RadBufDecl& f = fb.decl;
        const RadBufDecl& q = pb.decl;
        if (fb.name != pb.name || f.kind != q.kind || f.domain != q.domain ||
            f.dtype != q.dtype || f.rank != q.rank)
            return refuse(fmt("buffer %zu is '%s' in the real declare and '%s' in the sizing one",
                              i, fb.name.c_str(), pb.name.c_str()));
        /* ONLY THE ROWS MAY SHRINK. Every other extent is a stride some kernel was resolved
         * against at the real size, and a buffer narrower in any of them would be read at the
         * wrong pitch. */
        for (uint32_t d = 1; d < f.rank; ++d)
            if (f.shape[d] != q.shape[d])
                return refuse(fmt("buffer '%s' changes extent %u (%lld to %lld) with the step size",
                                  fb.name.c_str(), d, (long long)f.shape[d], (long long)q.shape[d]));
        if (f.rank > 0 && q.shape[0] > f.shape[0])
            return refuse(fmt("buffer '%s' is larger for a smaller step (%lld rows to %lld)",
                              fb.name.c_str(), (long long)f.shape[0], (long long)q.shape[0]));
        lv.decl[i] = q;
        /* THE FIXED REGION AND THE HOST ARENA ARE THE SAME AT EVERY LEVEL. A persistent buffer
         * holds what an earlier step wrote, so it cannot move -- and it keeps its full size too:
         * one the plugin sizes by the step (a per-token plane it writes from outside the op order)
         * is only ever smaller at a smaller step, and the real declare's extent covers it. */
        if (f.kind != RAD_BUF_TRANSIENT || f.domain == RAD_DOMAIN_HOST) {
            lv.decl[i] = f;
            lv.offset[i] = fb.arena_offset;
            continue;
        }
        /* The real declare's liveness: the op list is the same, and it is the one the kernels were
         * issued against. */
        Item it{ i, align_up(buffer_bytes(q), kBufAlign), fb.first_def, fb.last_use };
        if (it.lo < 0 || it.hi < it.lo) { it.lo = 0; it.hi = (int32_t)full.ops.size(); }
        transient.push_back(it);
    }

    auto by_size = [](const Item& a, const Item& b) {
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        if (a.lo != b.lo) return a.lo < b.lo;
        return a.idx < b.idx;
    };
    std::sort(transient.begin(), transient.end(), by_size);
    std::vector<Item> placed;
    int64_t tr_end = 0;
    for (Item it : transient) {
        it.off = first_fit(it, placed);
        tr_end = std::max(tr_end, it.off + it.bytes);
        lv.offset[it.idx] = full.arena_fixed_bytes + it.off;
        placed.push_back(it);
    }
    lv.end = full.arena_fixed_bytes + tr_end;
    if (lv.end > full.arena_bytes)
        return refuse(fmt("packing the smaller buffers took %lld bytes, more than the full plan's "
                          "%lld", (long long)lv.end, (long long)full.arena_bytes));
    *out = std::move(lv);
    return RAD_OK;
}

}  /* namespace rad */
