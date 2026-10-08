/* pools.cpp -- see pools.h. Every refusal in here names what did not fit and by how much. */
#include "mem/pools.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace rad {

/* ------------------------------------------------------------------ allocators */

namespace {

class DeviceAllocator final : public MemAllocator {
public:
    void* alloc(int64_t bytes, int kind) override { return rad_dev_alloc(bytes, kind); }
    void  free(void* p, int kind) override { rad_dev_free(p, kind); }
    const char* name() const override { return "device"; }
};

/* aligned_alloc, at the pool alignment. Used by the no-ROCm build and by the tests. A pool's base
 * is 2 MiB-aligned either way, because the .rad's pool alignment is 2 MiB and huge-page mappable
 * (spec §4.1) and a slab whose base is not aligned makes every fixed-stride slot inside it
 * misaligned as well. */
class MallocAllocator final : public MemAllocator {
public:
    void* alloc(int64_t bytes, int kind) override {
        (void)kind;
        if (bytes <= 0) return nullptr;
        int64_t n = align_up(bytes, (int64_t)RAD_ALIGN_POOL);
        void* p = ::aligned_alloc((size_t)RAD_ALIGN_POOL, (size_t)n);
        return p;
    }
    void free(void* p, int kind) override { (void)kind; ::free(p); }
    const char* name() const override { return "malloc"; }
};

}  /* namespace */

MemAllocator& rad_device_allocator() { static DeviceAllocator a; return a; }
MemAllocator& rad_malloc_allocator() { static MallocAllocator a; return a; }

/* ------------------------------------------------------------------ Shortfall */

std::string Shortfall::str() const {
    if (!refused.empty())
        return fmt("%s: %s of %s refused: %s", pool.c_str(), what.c_str(),
                   humanb(wanted).c_str(), refused.c_str());
    return fmt("%s: %s wanted %s, %s available -- short by %s",
               pool.c_str(), what.c_str(), humanb(wanted).c_str(),
               humanb(available).c_str(), humanb(missing()).c_str());
}

/* ------------------------------------------------------------------ Pool */

Pool::~Pool() { release(); }

void Pool::release() {
    /* ELASTIC FIRST, because an elastic pool has no allocator-owned base: vm_ owns the address
     * range and the pages in it, and handing that pointer to MemAllocator::free would be a free
     * of memory the allocator never handed out. */
    if (elastic_) {
        vm_.release();
        elastic_ = false;
    } else if (base_ && alloc_) {
        alloc_->free(base_, kind_);
    }
    base_ = nullptr;
    bytes_ = used_ = 0;
}

int Pool::init(std::string_view nm, int64_t bytes, int kind, MemAllocator* a, bool accounting) {
    release();
    name_ = std::string(nm);
    bytes_ = bytes;
    used_ = 0;
    kind_ = kind;
    accounting_ = accounting;
    alloc_ = a;
    shortfalls_.clear();
    res_.clear();
    if (accounting_ || bytes_ <= 0) return RAD_OK;

    base_ = a->alloc(bytes_, kind);
    if (!base_) {
        /* Refuse; do not retry smaller. An engine that quietly re-optimises the split is an
         * engine whose benchmarks do not reproduce (spec §6). The allocator's reason goes with
         * it: the device layer says what the driver answered. */
        const char* why = a == &rad_device_allocator() ? rad_dev_last_error() : nullptr;
        shortfalls_.push_back({name_, "backing allocation", bytes_, 0,
                               why && *why ? why : "the allocator returned nothing"});
        bytes_ = 0;
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

void* Pool::reserve(std::string_view what, int64_t bytes, int64_t align, int64_t* off_out) {
    if (bytes < 0 || align <= 0) return nullptr;
    int64_t off = align_up(used_, align);
    if (off + bytes > bytes_) {
        shortfalls_.push_back({name_, std::string(what), bytes, bytes_ - off});
        return nullptr;
    }
    used_ = off + bytes;
    res_.push_back({std::string(what), off, bytes});
    if (off_out) *off_out = off;
    if (accounting_) return nullptr;               /* headroom: accounted, never touched */
    return (char*)base_ + off;
}

/* ------------------------------------------------------------------ elastic */

int64_t Pool::granule() { return VMem::granularity(); }

int Pool::init_elastic(std::string_view nm, int64_t bytes, int kind, MemAllocator* a) {
    /* WITHOUT VIRTUAL MEMORY MANAGEMENT THIS IS init(). The caller does not branch: it commits
     * what it wants, the commits succeed against memory that was already there, and the pool
     * holds its maximum for the whole run. That is worse than elastic and identical to a fixed
     * pool, which is the right shape for a fallback. */
    if (!VMem::supported() || kind != RAD_MEM_DEVICE)
        return init(nm, bytes, kind, a, /*accounting=*/false);

    release();
    name_  = std::string(nm);
    bytes_ = bytes;
    used_  = 0;
    kind_  = kind;
    accounting_ = false;
    alloc_ = a;
    shortfalls_.clear();
    res_.clear();
    if (bytes_ <= 0) return RAD_OK;

    const int rc = vm_.reserve(bytes_, RAD_ALIGN_POOL);
    if (rc < 0) {
        /* Refuse; do not retry smaller. An address range is not memory -- a reservation that
         * fails on a 64-bit card is a bug and not a budget, and quietly halving it would hide the
         * bug behind a pool that cannot grow to what the operator asked for. */
        const char* why = rad_dev_last_error();
        shortfalls_.push_back({ name_, "address reservation", bytes_, 0,
                                *why ? why : "the device layer gave no reason" });
        bytes_ = 0;
        return rc;
    }
    base_ = vm_.base();
    /* THE RANGE ROUNDS UP AND THE BUDGET DOES NOT. vm_.size() is the reservation rounded to a
     * whole granule, which is a fact about the driver and not a licence to spend more than the
     * split allowed. Taking it as the budget lets a pool grow into the headroom by up to a
     * granule and, worse, GRANT a reservation the budget should have refused -- the same quiet
     * re-optimisation of the split that init() refuses to make when its allocation fails.
     *
     * The rounded tail stays addressable, and that is what it is for: the last reservation can
     * end mid-granule and its commit rounds outward into a range that is really reserved. */
    elastic_ = true;
    return RAD_OK;
}

int Pool::commit(int64_t off, int64_t bytes) {
    if (!elastic_) {
        /* A fixed pool is backed everywhere inside it and nowhere outside it, and saying so is
         * what keeps a caller written against the elastic contract correct here. */
        return (off >= 0 && bytes >= 0 && off + bytes <= bytes_) ? RAD_OK : RAD_E_INVAL;
    }
    return vm_.commit(off, bytes);
}

int Pool::decommit(int64_t off, int64_t bytes) {
    if (!elastic_) return RAD_OK;       /* nothing to give back, and that is not an error */
    return vm_.decommit(off, bytes);
}

bool Pool::is_committed(int64_t off, int64_t bytes) const {
    if (!elastic_) return off >= 0 && bytes >= 0 && off + bytes <= bytes_;
    return vm_.is_committed(off, bytes);
}

std::string Pool::report() const {
    std::string s = fmt("%-14s %10s budget, %10s used, %10s free%s\n",
                        name_.c_str(), humanb(bytes_).c_str(), humanb(used_).c_str(),
                        humanb(avail()).c_str(), accounting_ ? "   (accounted, not allocated)" : "");
    for (const auto& r : res_)
        s += fmt("    %-24s %10s @ +%lld\n", r.what.c_str(), humanb(r.bytes).c_str(),
                 (long long)r.off);
    for (const auto& f : shortfalls_) s += "    REFUSED " + f.str() + "\n";
    return s;
}

/* ------------------------------------------------------------------ Pools */

int Pools::configure(const Config& cfg, int64_t vram_capacity,
                     MemAllocator* device, MemAllocator* host) {
    configured_ = false;
    fit_report_.clear();

    if (!device) device = &rad_device_allocator();
    if (!host)   host   = device;

    const int64_t MiB = 1024 * 1024;
    const int64_t w  = cfg.vram_weights_mib  * MiB;
    const int64_t kv = cfg.vram_kv_mib       * MiB;
    const int64_t hp = (cfg.host_pool_mib + cfg.embedding_host_mib) * MiB;

    /* THE BUDGETS ARE RESOLVED BEFORE THIS RUNS AND THIS DOES NOT RE-DECIDE THEM.
     * core/mem/vram_budget.cpp derived them from the card, the measured driver residency and the
     * computed arena, and they arrive here as two numbers with a report behind them. This
     * component refuses rather than shrinking anything: an engine that quietly re-optimises a
     * split is an engine whose benchmarks do not reproduce (spec §6). A zero here means the
     * resolver was never called, which is a wiring bug and gets named as one. */
    std::string missing;
    if (cfg.vram_weights_mib <= 0) missing += "  vram_weights resolved to 0 MiB\n";
    if (cfg.vram_kv_mib      <= 0) missing += "  vram_kv resolved to 0 MiB\n";
    if (!missing.empty()) {
        fit_report_ = "VRAM pool budgets are empty: the budget resolver (core/mem/vram_budget.cpp) "
                      "did not run before the pools were configured:\n" + missing;
        RAD_ERR("%s", fit_report_.c_str());
        return RAD_E_INVAL;
    }

    const int64_t want = w + kv;
    if (vram_capacity > 0 && want > vram_capacity) {
        /* Name every pool and the total overage. Refuse rather than shrinking any of them. */
        fit_report_ = fmt(
            "VRAM budgets do not fit:\n"
            "  vram_weights   %10s\n"
            "  vram_kv        %10s\n"
            "  ------------------------\n"
            "  total          %10s\n"
            "  card total     %10s\n"
            "  OVER BY        %10s\n"
            "The resolved split is larger than the card, which the resolver does not produce: a "
            "wiring bug (spec §6).\n",
            humanb(w).c_str(), humanb(kv).c_str(),
            humanb(want).c_str(), humanb(vram_capacity).c_str(),
            humanb(want - vram_capacity).c_str());
        RAD_ERR("%s", fit_report_.c_str());
        return RAD_E_NOMEM;
    }

    int st = RAD_OK;
    int s;
    /* Order matters only for the diagnostic: allocate the two big ones first so a failure names
     * the one that actually could not be served. */
    if ((s = weights_.init("vram_weights", w,  RAD_MEM_DEVICE, device, false)) < 0) st = s;
    /* THE KV POOL IS THE ELASTIC ONE, ALWAYS, AND THERE IS NO OTHER MODE. Its budget is still the
     * budget: elasticity does not change how much of the card the cache may address, it changes
     * whether the part it is not addressing has VRAM behind it. What the cache releases is what
     * the expert plane can promote into, so the split between them stops being a guess made once
     * at startup and becomes the two pools' actual demand.
     *
     * THERE IS NO FLAG, BECAUSE A FLAG IS THE WRONG SHAPE FOR IT. A fixed pool holds every block
     * it was carved for whether or not a single token is in it, and an expert plane that does not
     * fit streams weights over the link for the whole run to pay for cache nobody is using. There
     * is no deployment that wants that. A device with no virtual memory management goes through
     * the same call -- init_elastic() allocates the range whole there, which is fixed-pool
     * behaviour under a contract the callers already meet. */
    if ((s = kv_.init_elastic("vram_kv", kv, RAD_MEM_DEVICE, device)) < 0) st = s;
    /* MAPPED, NOT MERELY PINNED, AND THE DIFFERENCE IS A WHOLE TIER.
     *
     * This one pool serves two jobs: the mover's staging buffers and O_DIRECT landing zone, which
     * want pinned memory so a transfer is a DMA with no driver bounce -- and the ZERO-COPY
     * execution site, row three of spec §5.1's table, where the bytes stay in host memory and the
     * kernel reads them over the link as it computes. The second needs the allocation to be
     * addressable FROM A KERNEL, and hipHostMallocDefault does not make it so.
     *
     * Getting it wrong is silent and total: Mover::init asks rad_dev_device_ptr for the pool's
     * base as the card sees it, an unmapped allocation answers null, and the fallback is the host
     * pointer -- so `assign_expert_tiered`, which routes EVERY pooled expert to
     * Site::DeviceZeroCopy, would hand a kernel a pointer it cannot dereference. There would be no
     * hybrid tier, only a plan that says there is one.
     *
     * Mapped memory is still pinned, so job one is unaffected; hipHostMallocMapped is strictly the
     * larger promise. On the host backend all four kinds are the same aligned pages. */
    if ((s = host_   .init("host_pool",   hp, RAD_MEM_HOST_MAPPED, host, false)) < 0) st = s;

    fit_report_  = weights_.report();
    fit_report_ += kv_.report();
    fit_report_ += host_.report();

    if (st < 0) {
        RAD_ERR("pool allocation refused:\n%s", fit_report_.c_str());
        return st;
    }
    configured_ = true;
    return RAD_OK;
}

std::vector<Shortfall> Pools::shortfalls() const {
    std::vector<Shortfall> out;
    for (const Pool* p : { &weights_, &kv_, &host_ })
        for (const auto& f : p->shortfalls()) out.push_back(f);
    return out;
}

}  /* namespace rad */
