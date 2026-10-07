/* oracle.cpp -- run the HOST kernel beside the device one, on the same bytes, every issue.
 *
 * ================================================================= why this exists
 *
 * rad-kbench falsifies one op at a time against libref, with operands rad-kbench invented. A build
 * can pass nearly every configuration it knows while the engine answers the same token to every
 * prompt. That gap is structural and not a gap in coverage: rad-kbench's operands are ITS shapes, dense and
 * rank-2 and freshly filled, and the engine's are row slices of a max_tok buffer, column slices of
 * a fused projection, rank-3 views a scan kernel required, and weights that came off a file. An op
 * can be correct at every shape rad-kbench knows and wrong at the one shape the model issues.
 *
 * So this checks the shape the model actually issued. For a matching op it
 *
 *   1. copies every operand to the host BEFORE the launch -- inputs, outputs, weights, all of it;
 *   2. lets the device kernel run and synchronises;
 *   3. copies every operand back into a second host buffer;
 *   4. puts every weight the device row stores RELAID back through that row's own
 *      RadUnrelayoutFn, because the kernel rearranged it at load and a dense reference reads the
 *      canonical planes (see oracle_unrelayout -- this step is not optional, it is the
 *      difference between checking a kernel and comparing a permutation against itself);
 *   5. runs the HOST kernel for the same (op, band) on the copy from step 1 -- same RadTensor
 *      shapes, same strides, same parameters, only the pointers differ;
 *   6. compares, per operand, the host result against the device result.
 *
 * Step 5 is the part that has no substitute. The host row is libref's, resolved by the same
 * selector into the same bucket table -- `issue_table_[plan_off + band*RAD_N_DOMAINS + HOST]` is
 * already sitting there beside the device plan, filled in at prepare and never used, because the
 * placement planner put every op on the device. It is a second implementation of the op, against
 * the same schema, and it is free.
 *
 * ================================================================= what it costs
 *
 * Everything. Two device-to-host copies of every operand -- a fused gate_up weight is 170 MiB --
 * a host synchronisation per issue, and a reference kernel that is not trying to be fast. A step
 * with the oracle on takes minutes. It is a debugging instrument and it is off unless asked.
 *
 * ================================================================= how to ask
 *
 *   RADIANCE_OP_ORACLE=gemm_nt_q,rmsnorm   op names, comma separated; `*` for every op
 *   RADIANCE_OP_ORACLE_MAX=32              stop after this many checks (default 32)
 *   RADIANCE_OP_ORACLE_FROM=0              first ISSUE INDEX to consider (default 0)
 *   RADIANCE_OP_ORACLE_TO=100000           last issue index to consider
 *   RADIANCE_OP_ORACLE_TOL=0.02            relative tolerance below which an operand is silent
 *   RADIANCE_OP_ORACLE_MAXBYTES            the largest operand a check will copy (default 4 GiB)
 *
 * The issue index is the same counter Ctx tracks internally and the same order --debug-graph
 * prints, so a failure names a position in the graph and not just an op.
 *
 * ================================================================= the repeat mode
 *
 * RADIANCE_OP_REPEAT asks a different question with the same machinery: not "does this kernel
 * agree with libref" but "does this kernel agree with ITSELF". It snapshots the operands, lets
 * the device kernel run, puts the operands back exactly as they were, runs the SAME DEVICE KERNEL
 * again, and byte-compares the two results.
 *
 * That is the one fault class every other instrument here is structurally blind to. The oracle,
 * rad-kbench and the residual probe all run an op ONCE per side, so a kernel with an unordered
 * reduction, a workgroup reading what another workgroup is writing, or a workspace it assumed was
 * zeroed passes them at exactly the rate it happens to be right. Byte-exact, no tolerance: the
 * same kernel on the same bytes has no licence to move, and one mantissa bit is a different token
 * a few hundred greedy steps later.
 *
 *   RADIANCE_OP_REPEAT=*                   op names, comma separated; `*` for every op
 *   RADIANCE_OP_REPEAT_MAX=1000000         stop after this many checks
 *   RADIANCE_OP_REPEAT_FROM / _TO          issue index window
 *   RADIANCE_OP_REPEAT_MAXBYTES=8388608    operands above this are not copied, restored or
 *                                          compared -- which is what keeps a 170 MiB fused weight
 *                                          from making the mode unaffordable, and is right only
 *                                          because such an operand is read-only.
 *
 * ================================================================= what it cannot see
 *
 * An op ISSUED ON A LANE THE SNAPSHOT DOES NOT SYNCHRONISE -- fixed for lane 1, which is the only
 * one the ABI has, and a third would need the same line.
 *
 * An op with no host kernel for the band -- the oracle skips it and says so once. An op whose
 * dtype the core cannot size (a plugin-private one) -- skipped, since a byte span is the one thing
 * this needs to know. An op whose device row re-lays a weight and publishes no inverse -- skipped
 * BY NAME, because the alternative is a page of false mismatches that drowns the true ones. And
 * an op that is wrong in the SAME WAY on both sides, which is what a shared misreading of the
 * schema would look like; that is what rad-kbench's independent operands
 * are for, and the two instruments are complementary rather than redundant.
 */
#include "ctx.h"
#include "format/encoding.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the state */
struct Ctx::OracleState {
    std::vector<std::string> want;        /* op names; a single "*" means every op */
    int64_t budget = 32;
    int64_t from = 0, to = INT64_MAX;
    double  tol = 2e-2;
    int64_t checked = 0, skipped_nohost = 0;
    bool    said_nohost = false;
    /* ---- repeat mode. `maxbytes` is what makes it affordable: a fused gate_up operand is 170 MiB
     * and a weight is read-only, so an operand past the cap is not snapshotted, not restored and
     * not compared. Every activation this graph carries at decode is far below it. */
    bool    repeat = false;
    int64_t maxbytes = 8 << 20;
    /* ---- AN OPERAND THE HOST CANNOT HOLD, which is the oracle mode's own cap and not the one
     * above. A snapshot is two host copies of every operand, plus a third for a weight stored
     * re-laid, so an operand of tens of gigabytes does not make a check slow -- it ends the
     * process with nothing reported. Four gibibytes clears a routed layer's whole expert table,
     * which is the largest operand a check has to hold. */
    int64_t maxopd = 4LL << 30;
    int64_t differed = 0, unchecked = 0;
    int     skip[MAX_OPERANDS] = {0};

    std::vector<uint8_t>              scratch;    /* the host kernel's, sized per launch */
    /* A re-laid weight inverted back into the form a dense reference reads, one buffer per
     * operand that needed it, and which operands those were. See oracle_unrelayout. */
    std::vector<std::vector<uint8_t>> logical;
    int                               unlaid[MAX_OPERANDS] = {0};
    std::vector<std::vector<uint8_t>> pre, post;  /* one snapshot per operand */
    int64_t                           span[MAX_OPERANDS] = {0};   /* elements, stacked */
    int64_t                           ent [MAX_OPERANDS] = {0};   /* bytes, ONE entry */
    /* Entries when the operand is a weight TABLE and 0 when it is a plane. See oracle_size: a
     * table is materialised stacked, because the operand itself is an array of pointers. */
    int64_t                           tab [MAX_OPERANDS] = {0};
    int                               alias[MAX_OPERANDS] = {0};   /* which operand owns the copy */
    RadTensor                         ht[MAX_OPERANDS]{};
};

namespace {

/* The number of ELEMENTS a tensor's strides can reach, which is what has to be copied. Not
 * numel: a row slice of a [max_tok, n] buffer is dense, but a column slice of a fused projection
 * is not, and copying numel elements of one of those copies the wrong bytes. */
int64_t span_of(const RadTensor* t) {
    if (!t->data || t->rank == 0) return 0;
    int64_t s = 1;
    for (uint32_t d = 0; d < t->rank; ++d) {
        if (t->shape[d] <= 0) return 0;
        const int64_t st = t->stride[d];
        if (st > 0) s += (t->shape[d] - 1) * st;
    }
    return s;
}

float f8e4m3_to_f32(uint8_t b) {
    const uint32_t sgn = (uint32_t)(b & 0x80u) << 24;
    const int      e   = (b >> 3) & 0x0F;
    const uint32_t m   = b & 0x07u;
    float f;
    uint32_t u;
    if (e == 0) {                                   /* subnormal: m * 2^-9 */
        if (!m) { u = sgn; std::memcpy(&f, &u, 4); return f; }
        int ex = -6; uint32_t mm = m;
        while (!(mm & 0x08u)) { mm <<= 1; --ex; }   /* normalise into 1.mmm */
        u = sgn | ((uint32_t)(ex + 127) << 23) | ((mm & 0x07u) << 20);
    } else if (e == 0x0F && m == 0x07) {            /* the two NaN encodings */
        u = sgn | 0x7FC00000u;
    } else {
        u = sgn | ((uint32_t)(e - 7 + 127) << 23) | (m << 20);
    }
    std::memcpy(&f, &u, 4);
    return f;
}

/* One element as a double, by dtype. Returns false for a dtype with no meaningful value -- the
 * comparison then falls back to a byte compare, which is the right answer for an index plane. */
bool elem(const uint8_t* p, uint32_t dt, int64_t i, double* out) {
    switch (dt) {
    case RAD_F32:  { float v;    std::memcpy(&v, p + i * 4, 4); *out = v; return true; }
    case RAD_BF16: { uint32_t u = (uint32_t)((uint16_t)(p[i * 2] | (p[i * 2 + 1] << 8))) << 16;
                     float v; std::memcpy(&v, &u, 4); *out = v; return true; }
    case RAD_F16:  { uint16_t h = (uint16_t)(p[i * 2] | (p[i * 2 + 1] << 8));
                     const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
                     const int      e = (h >> 10) & 0x1F;
                     const uint32_t m = h & 0x3FFu;
                     uint32_t u;
                     if (e == 0)        u = s | (m ? ((uint32_t)(127 - 15 + 1) << 23) | (m << 13) : 0u);
                     else if (e == 31)  u = s | 0x7F800000u | (m << 13);
                     else               u = s | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
                     float v; std::memcpy(&v, &u, 4); *out = v; return true; }
    case RAD_F8E4M3: *out = f8e4m3_to_f32(p[i]); return true;
    case RAD_I8:   *out = (double)(int8_t)p[i];  return true;
    case RAD_U8:
    case RAD_BOOL: *out = (double)p[i];          return true;
    case RAD_I16:  { int16_t v; std::memcpy(&v, p + i * 2, 2); *out = v; return true; }
    case RAD_I32:  { int32_t v; std::memcpy(&v, p + i * 4, 4); *out = v; return true; }
    case RAD_U32:  { uint32_t v; std::memcpy(&v, p + i * 4, 4); *out = v; return true; }
    case RAD_I64:  { int64_t v; std::memcpy(&v, p + i * 8, 8); *out = (double)v; return true; }
    default: return false;
    }
}

double row_at(const uint8_t* p, uint32_t dt, int64_t base, int64_t k, int64_t n) {
    double v = 0;
    if (base + k < n) elem(p, dt, base + k, &v);
    return v;
}

std::string shape_str(const RadTensor* t) {
    std::string s = "[";
    for (uint32_t d = 0; d < t->rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t->shape[d]);
    s += "]/[";
    for (uint32_t d = 0; d < t->rank; ++d)
        s += fmt("%s%lld", d ? "," : "", (long long)t->stride[d]);
    return s + "]";
}

}  /* namespace */

/* ------------------------------------------------------------------ open / close */
void Ctx::oracle_open() {
    /* REPEAT MODE IS THE SAME MACHINERY WITH THE SECOND SIDE SWAPPED, so it shares the arming,
     * the op filter and the snapshot and differs only in what it runs and how it compares. Named
     * separately because the two answer different questions: the oracle asks whether a kernel is
     * RIGHT, this asks whether it is a FUNCTION. */
    const char* spec = std::getenv("RADIANCE_OP_REPEAT");
    const bool  rep  = spec && *spec;
    if (!rep) spec = std::getenv("RADIANCE_OP_ORACLE");
    if (!spec || !*spec) return;

    oracle_ = new OracleState();
    oracle_->repeat = rep;
    for (const char* s = spec; *s; ) {
        const char* e = s;
        while (*e && *e != ',') ++e;
        if (e > s) oracle_->want.emplace_back(s, (size_t)(e - s));
        s = *e ? e + 1 : e;
    }
    auto geti = [](const char* k, int64_t d) {
        const char* v = std::getenv(k);
        return (v && *v) ? (int64_t)std::strtoll(v, nullptr, 10) : d;
    };
    oracle_->budget = geti(rep ? "RADIANCE_OP_REPEAT_MAX" : "RADIANCE_OP_ORACLE_MAX", rep ? 1000000 : 32);
    oracle_->maxbytes = geti("RADIANCE_OP_REPEAT_MAXBYTES", 8 << 20);
    oracle_->maxopd   = geti("RADIANCE_OP_ORACLE_MAXBYTES", 4LL << 30);
    oracle_->from   = geti(rep ? "RADIANCE_OP_REPEAT_FROM" : "RADIANCE_OP_ORACLE_FROM", 0);
    oracle_->to     = geti(rep ? "RADIANCE_OP_REPEAT_TO" : "RADIANCE_OP_ORACLE_TO", INT64_MAX);
    if (const char* t = std::getenv("RADIANCE_OP_ORACLE_TOL")) oracle_->tol = std::strtod(t, nullptr);

    if (oracle_->repeat) {
        RAD_WARN("op REPEAT armed for {%s}: issues [%lld,%lld], at most %lld checks, operands up "
                 "to %lld bytes. Every matching issue runs its DEVICE kernel TWICE on the same "
                 "bytes and byte-compares. A difference is a kernel that is not a function of its "
                 "inputs. This run is not a measurement of anything.",
                 spec, (long long)oracle_->from, (long long)oracle_->to,
                 (long long)oracle_->budget, (long long)oracle_->maxbytes);
        return;
    }
    RAD_WARN("op oracle ARMED for {%s}: issues [%lld,%lld], at most %lld checks, tolerance %.3g, "
             "operands up to %lld MiB. Every matching issue copies all its operands to the host "
             "twice and runs the host kernel beside the device one -- this run is not a "
             "measurement of anything.",
             spec, (long long)oracle_->from, (long long)oracle_->to,
             (long long)oracle_->budget, oracle_->tol, (long long)(oracle_->maxopd >> 20));
}

void Ctx::oracle_close() {
    if (!oracle_) return;
    if (oracle_->repeat)
        RAD_WARN("op REPEAT: %lld issue(s) run twice, %lld of them ANSWERED DIFFERENTLY; %lld "
                 "more were NOT JUDGED because an operand was over the copy cap (raise "
                 "RADIANCE_OP_REPEAT_MAXBYTES to cover those)",
                 (long long)oracle_->checked, (long long)oracle_->differed,
                 (long long)oracle_->unchecked);
    else
        RAD_WARN("op oracle: %lld op(s) checked, %lld skipped for want of a host kernel",
                 (long long)oracle_->checked, (long long)oracle_->skipped_nohost);
    delete oracle_;
    oracle_ = nullptr;
}

/* ------------------------------------------------------------------ the check */
bool Ctx::oracle_arm(rad_op h) const {
    OracleState& o = *oracle_;
    if (o.budget <= 0) return false;
    if (n_issues_ < o.from || n_issues_ > o.to) return false;
    const std::string& name = plans_[h].info->op;
    for (const std::string& w : o.want)
        if (w == "*" || w == name) return true;
    return false;
}

/* ------------------------------------------------------------------ sizing the snapshot
 *
 * WHAT IS BEING SIZED IS NOT ALWAYS WHAT THE OPERAND POINTS AT, and a routed mixture is where the
 * difference is total. A grouped GEMM's weight operand is a TABLE (Ctx::WeightTable): after
 * placement a layer's experts are in different places, so the operand handed to the kernel is a
 * device array of POINTERS with a zero leading stride, and `base + e*stride` describes only the
 * all-in-VRAM case, which stops being true the moment anything is offloaded.
 *
 * Read as a plane that operand is n*8 bytes pretending to be N*K of weight. Copying it as one
 * reads far past the pointer array, and the host kernel then dereferences DEVICE pointers -- so
 * the largest weight in the model would be the one the instrument cannot be pointed at without
 * reading out of bounds. What it gets instead is the STACKED form: every entry copied into one
 * contiguous host buffer behind a non-zero leading stride, which is the other reading libref's
 * expert_entry() already accepts and the one a fixture uses.
 *
 * A PLUGIN-PRIVATE DTYPE HAS NO BYTE COUNT IN THE CORE, by design -- a 4-bit expert plane is two
 * elements to a byte and the core deliberately knows nothing about the packing (spec 4.2). The
 * container does: the layout pass wrote the resolved kernel's own answer into WeightInfo, so an
 * operand the core cannot size is sized from the weight it is bound to, and only an operand that
 * is neither sizeable nor a weight is declined. */
void Ctx::oracle_size(rad_op h, int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;

    /* The stored byte count of the weight at each operand position, from the layout pass. Every
     * entry of a table maps to one position and they are proven identical at first issue, so the
     * last writer and the first agree. */
    int64_t wb[MAX_OPERANDS] = {0};
    int     wlaid[MAX_OPERANDS] = {0};
    if (program_)
        for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j) {
            const rad_weight wh = oi.weights[j];
            const int k = (int)oi.weight_opd[j];
            if (!wh || (size_t)wh >= program_->weights.size()) continue;
            if (k < 0 || k >= n_opd) continue;
            wb[k]    = program_->weights[(size_t)wh].stored_bytes;
            wlaid[k] = program_->weights[(size_t)wh].layout_tag.empty() ? 0 : 1;
        }

    for (int i = 0; i < n_opd; ++i) {
        const RadTensor& t = tbuf_[i];
        /* A ZERO LEADING STRIDE IS THE MARKER. resolve_weight_table() prepends the expert axis
         * that way precisely to say "this axis is not addressable by arithmetic". */
        o.tab[i] = (t.data && t.rank >= 2 && t.stride[0] == 0 && t.shape[0] > 0) ? t.shape[0] : 0;
        const int64_t es = span_of(&t);           /* one entry, or the whole plane when not a table */
        o.span[i] = o.tab[i] ? es * o.tab[i] : es;
        o.ent[i]  = es > 0 ? rad_dtype_bytes(t.dtype, es) : 0;
        if (o.ent[i] <= 0 && es > 0) o.ent[i] = wb[i];
        /* A RE-LAID WEIGHT IS AS BIG AS THE CONTAINER SAYS, NOT AS BIG AS ITS DECLARED PLANE, and
         * the two differ even where the dtype has a byte count. The fp8 lm_head pads each stored
         * row to a 16-byte boundary and carries that row's block scales inside it, so the plane
         * is 2608 bytes a row where n_embd * 1 says 2560 -- and the inverse, which checks the
         * byte count it is handed against the stride it wrote, refuses a snapshot sized from the
         * declared plane. The layout pass records the resolved kernel's own answer for exactly
         * this; where a weight is stored re-laid, it wins. */
        if (wlaid[i] && wb[i] > 0) o.ent[i] = wb[i];
        o.ht[i] = t;
        /* The stacked reading: the expert axis becomes addressable and its stride is one entry. */
        if (o.tab[i]) o.ht[i].stride[0] = es;

        /* ALIASED OPERANDS SHARE ONE HOST BUFFER. A residual add is issued {x, h} -> {x}: two
         * of its three operands are the same device bytes, and giving each its own copy makes
         * the host kernel write into a buffer the device kernel's own read-modify-write also
         * wrote -- which reports the op as a mismatch of exactly the size of the addend, every
         * time, on an op that is correct. The first operand with a given pointer owns the
         * copy and the rest point into it, which is what the device sees. */
        o.alias[i] = i;
        for (int j = 0; j < i; ++j)
            if (t.data && tbuf_[j].data == t.data) { o.alias[i] = o.alias[j]; break; }
    }
}

/* Bytes the snapshot of operand `i` occupies, which for a table is every entry of it. */
int64_t Ctx::oracle_opd_bytes(int i) const {
    const OracleState& o = *oracle_;
    if (o.span[i] <= 0 || o.ent[i] <= 0) return 0;
    return o.tab[i] ? o.ent[i] * o.tab[i] : o.ent[i];
}

void Ctx::oracle_snap(int n_opd, bool post) {
    OracleState& o = *oracle_;
    std::vector<std::vector<uint8_t>>& into = post ? o.post : o.pre;
    into.resize(MAX_OPERANDS);
    std::vector<void*> ptr;
    /* THE SECOND LANE IS DRAINED FIRST, AND THE ORDER IS THE WHOLE POINT. An architecture may
     * issue on lane 1 -- the MoE shared expert does -- and the copies below go on `stream_`, which
     * is not ordered against it. Draining lane 1 AFTER them leaves the copy racing the kernel
     * that writes what it is copying, which is not a fix: the copy has already returned its bytes.
     *
     * Both snapshots need it and for different reasons. In `post` the race reads an OUTPUT before
     * its kernel wrote it, and a correct kernel reports as having produced nothing. In `pre` it
     * reads an INPUT before its producer wrote it, and then the two sides are not even given the
     * same problem -- the host kernel gets a half-written operand, the device kernel gets the
     * whole one, and the operand the oracle prints is an INPUT that disagrees with itself. The
     * oracle already costs a host synchronisation per issue; one more stream is free. */
    if (lane1_) rad_stream_sync(lane1_);
    for (int i = 0; i < n_opd; ++i) {
        const RadTensor& t = tbuf_[i];
        if (o.alias[i] != i) continue;
        const int64_t bytes = oracle_opd_bytes(i);
        /* THE CAP IS A REPEAT-MODE THING ONLY. The oracle has to see every operand or its host
         * kernel has nothing to read; the repeat re-launches the device kernel on the operands
         * still in place, so an operand it declines to copy is simply one it cannot restore and
         * will not compare -- which is right for a weight and would be wrong for an output. */
        if (!post) o.skip[i] = o.repeat && bytes > o.maxbytes;
        if (o.skip[i]) { into[(size_t)i].clear(); continue; }
        /* A SLOT KEEPS THE HIGH-WATER CAPACITY OF EVERY OPERAND THAT EVER LANDED IN IT, and on a
         * large model that is not a rounding error: operand 2 of one op is a 170 MiB fused weight
         * and operand 2 of the next is a few kilobytes, so thirty-two slots across `pre`, `post`
         * and `logical` hold the sum of the largest thing each has ever seen -- on a container
         * of well over a hundred GiB, enough for the host OOM killer to end a sweep part way
         * through. Shrinking when the slot is grossly oversized bounds the footprint by the
         * largest SINGLE issue instead of by the sum over the graph; the reallocation costs
         * nothing against a synchronised copy. */
        into[(size_t)i].assign((size_t)bytes, 0);
        if (into[(size_t)i].capacity() > (size_t)bytes * 2 + (1u << 20))
            into[(size_t)i].shrink_to_fit();
        if (bytes <= 0) continue;
        if (!o.tab[i]) { rad_memcpy_async(into[(size_t)i].data(), t.data, bytes, stream_); continue; }
        /* The entries are wherever placement put them, so their pointers come back first and the
         * copies are issued against those. An entry may be device- or host-resident and the
         * direction is inferred from the pointer, which is what makes one call serve both. */
        ptr.assign((size_t)o.tab[i], nullptr);
        rad_memcpy_async(ptr.data(), t.data, o.tab[i] * (int64_t)sizeof(void*), stream_);
        rad_stream_sync(stream_);
        for (int64_t e = 0; e < o.tab[i]; ++e)
            if (ptr[(size_t)e])
                rad_memcpy_async(into[(size_t)i].data() + e * o.ent[i], ptr[(size_t)e],
                                 o.ent[i], stream_);
    }
    rad_stream_sync(stream_);
    if (!post)
        for (int i = 0; i < n_opd; ++i)
            o.ht[i].data = o.span[i] > 0 ? (void*)o.pre[(size_t)o.alias[i]].data() : nullptr;
}

/* ------------------------------------------------------------------ the stored layout
 *
 * THE HOST KERNEL READS A WEIGHT CANONICALLY AND THE ENGINE DOES NOT HOLD IT THAT WAY.
 *
 * A container holds each weight's encoding as canonical planes, and the kernel that reads it may
 * rearrange them at load (spec §4.3): libr4d's fp8a8 GEMMs claim WMMA FRAGMENT ORDER, the same N*K
 * bytes permuted so a wave's sixteen rows arrive in lane order off one coalesced load. libref
 * declares no layout for anything and reads [N, K] row-major. Handing it the operand pointer
 * unchanged therefore compares two readings of one permuted plane, which are uncorrelated tensors
 * -- a relative error around 1.35 on every operand from element zero, on a kernel that is not
 * wrong.
 *
 * So the stored bytes go back through the device row's own inverse to the canonical planes, and
 * from those to the form the HOST row reads: a single plane converted element by element -- an
 * expert's int4 codes as the integers the reference multiplies -- or a whole encoding decoded by
 * the core's decoder, which is what a reference reading an lm_head of codes and scales together
 * wants. Both sides then see the same values, expressed the way each of them reads.
 *
 * NO INVERSE MEANS NO CHECK, and it is a named skip rather than a tolerance: a comparison that
 * cannot be made honestly is worse than one not made, because a page of false failures drowns the
 * true ones. Returns false in that case, and the issue is not counted as checked.
 *
 * WHICH LOGICAL FORM, AND WHO SAYS SO. `opd_shape` describes the operand the kernel is HANDED --
 * for a relayed weight that is the stored form, which is the thing being inverted and cannot also
 * be the answer. The form wanted here is the one the READER reads, so the host row is asked first
 * and the device row is the fallback. libref publishes a shape for the whole conventional
 * vocabulary, so an op inside docs/OPS.md is describable even when the kernel under test
 * describes nothing.
 *
 * The extents come from the parameters AT THIS ISSUE, not from the weight's declaration: a
 * tensor-parallel shard's op carries the per-rank N and K. */
bool Ctx::oracle_unrelayout(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int64_t seq,
                            int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;
    for (int i = 0; i < n_opd; ++i) o.unlaid[i] = 0;

    const RadKernelInfo* ki = dip.row ? dip.row->info : nullptr;
    if (!ki || !ki->layout || !program_) return true;   /* every operand is read as it is */
    const RadKernelInfo* hk = hip.row ? hip.row->info : nullptr;
    const RadShapeFn shape = hk && hk->opd_shape ? hk->opd_shape : ki->opd_shape;

    o.logical.resize(MAX_OPERANDS);
    for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j) {
        const rad_weight wh = oi.weights[j];
        if (!wh || (size_t)wh >= program_->weights.size()) continue;
        const WeightInfo& w = program_->weights[(size_t)wh];
        if (w.layout_tag.empty()) continue;             /* stored as it is */
        const int i = oi.weight_opd[j];
        if (i < 0 || i >= n_opd || o.span[i] <= 0 || o.unlaid[i]) continue;

        if (!ki->unrelayout || !shape) {
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. '%s' is stored as %s's '%s', and no %s "
                     "can be had for it -- so the reference would read a permuted plane row-major "
                     "and disagree with a kernel that is right.",
                     (long long)seq, oi.op.c_str(), w.name.c_str(), ki->name,
                     w.layout_tag.c_str(), ki->unrelayout ? "RadShapeFn" : "RadUnrelayoutFn");
            return false;
        }
        RadOpdDesc d{};
        if (shape(dip.args.p, dip.args.n_p, i, &d) != RAD_OK || d.rank == 0) {
            /* A shape hook declines without a reason (RadShapeFn), so the geometry it declined is
             * printed, the dtype with it -- a description refuses a dtype it knows no grid for. */
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. nothing could describe the logical form "
                     "of '%s' at this geometry {%s}, so there is nothing to invert its layout "
                     "into.", (long long)seq, oi.op.c_str(), w.name.c_str(),
                     oi.base.str().c_str());
            return false;
        }
        RadTensor lg{};
        lg.dtype = d.dtype;
        lg.rank  = d.rank;
        int64_t n = 1, acc = 1;
        for (uint32_t k = 0; k < d.rank && k < RAD_MAX_RANK; ++k) { lg.shape[k] = d.shape[k]; n *= d.shape[k]; }
        for (int k = (int)d.rank - 1; k >= 0; --k) { lg.stride[k] = acc; acc *= lg.shape[k]; }
        const int64_t bytes = rad_dtype_bytes(lg.dtype, n);
        if (bytes <= 0) return false;
        /* A TABLE'S DESCRIPTION MUST BE THE STACKED PLANE. The two agree on every other axis, so
         * a disagreement here is a description that answered for one expert where the operand
         * carries the layer's whole set -- and inverting into it would write past the buffer. */
        if (o.tab[i] && (d.rank < 2 || lg.shape[0] != o.tab[i])) {
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. '%s' is a table of %lld experts and the "
                     "logical form offered for it leads with %lld, so the two are not the same "
                     "tensor.", (long long)seq, oi.op.c_str(), w.name.c_str(),
                     (long long)o.tab[i], (long long)(d.rank ? lg.shape[0] : 0));
            return false;
        }
        o.logical[(size_t)i].assign((size_t)bytes, 0);
        if (o.logical[(size_t)i].capacity() > (size_t)bytes * 2 + (1u << 20))
            o.logical[(size_t)i].shrink_to_fit();
        lg.data = o.logical[(size_t)i].data();

        /* ONE CALL PER ENTRY FOR A TABLE. The hook answers for a weight, and an expert IS the
         * weight -- the table is the engine's way of saying where the experts ended up, not a
         * shape the plugin ever agreed to. So each entry's stored bytes are inverted on their own
         * and land at their own offset in the stacked tensor. */
        const std::vector<uint8_t>& src = o.pre[(size_t)o.alias[i]];
        const int64_t n_ent = o.tab[i] ? o.tab[i] : 1;
        const int64_t ent_bytes = o.tab[i] ? o.ent[i] : (int64_t)src.size();
        const int64_t ent_n = n / n_ent;
        std::vector<std::vector<uint8_t>> pl((size_t)w.n_sel);
        RadTensor pt[RAD_ENC_MAX_PLANES];
        weight_planes(w, pt);
        int rc = RAD_OK;
        std::string why;
        for (int64_t e = 0; e < n_ent && rc >= 0; ++e) {
            for (int k = 0; k < w.n_sel; ++k) {
                pl[(size_t)k].assign((size_t)(w.sel_rows[k] *
                                              rad_enc_row_bytes(pt[k].dtype, w.sel_cols[k])), 0);
                pt[k].data = pl[(size_t)k].data();
            }
            rc = ki->unrelayout(dip.args.p, dip.args.n_p, i, &w.enc, w.sel,
                                src.data() + e * ent_bytes, ent_bytes, pt, w.n_sel);
            if (rc < 0) break;
            uint8_t* dst = (uint8_t*)lg.data + rad_dtype_bytes(lg.dtype, ent_n * e);
            const void* pd[RAD_ENC_MAX_PLANES] = {};
            for (int k = 0; k < w.n_sel; ++k) pd[k] = pl[(size_t)k].data();
            rc = enc_selection_logical(w.enc, w.sel, w.n_sel, w.sel_rows, w.sel_cols, pd,
                                       lg.dtype, ent_n, dst, &why);
        }
        if (rc < 0) {
            /* THE TWO NUMBERS IT DISAGREED ABOUT. An inverse refuses on a shape far more often
             * than on anything else, and "shape this kernel does not serve" without them says
             * only that one of the two sides is wrong -- which is the whole question. */
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. %s could not invert its own layout for "
                     "'%s' (%s%s%s): %lld stored bytes into %lld %s elements, so the reference has "
                     "nothing it can read.",
                     (long long)seq, oi.op.c_str(), ki->name, w.name.c_str(), rad_strerror(rc),
                     why.empty() ? "" : ": ", why.c_str(), (long long)ent_bytes, (long long)ent_n,
                     rad_dtype_name(lg.dtype));
            return false;
        }
        o.ht[i] = lg;
        o.unlaid[i] = 1;
    }
    return true;
}


void Ctx::oracle_run(rad_op h, const IssuePlan& hip, int64_t seq, int n_opd) {
    OracleState& o = *oracle_;
    const OpInfo& oi = *plans_[h].info;

    /* The host kernel writes into the PRE copy: same shapes, same strides, same parameters, and
     * pointers that are the only thing about the call that differs from the device's. */
    RadArgs a = hip.args;
    a.t   = o.ht;
    a.n_t = n_opd;
    o.scratch.assign((size_t)(a.scratch_bytes > 0 ? a.scratch_bytes : 0), 0);
    a.scratch = o.scratch.empty() ? nullptr : o.scratch.data();

    const int rc = hip.launch(&a, nullptr);
    if (rc < 0) {
        /* RAD_E_UNSUPPORTED IS NOT A FINDING. It is the reference saying "not this contract" --
         * an op whose parameters name a form it does not implement, which `gdn_recurrent_update`'s
         * two state layouts are the live example of. Reported as the coverage gap it is, because
         * the alternative is a reader treating a declined comparison as a passed one. Any OTHER
         * code is the reference refusing shapes it claims to serve, and that is about the engine's
         * operands rather than about the oracle. */
        if (rc == RAD_E_UNSUPPORTED)
            RAD_WARN("oracle #%lld op '%s': NOT CHECKED. the host kernel (%s) does not implement "
                     "what these parameters ask for {%s}, so there is no comparison to make.",
                     (long long)seq, oi.op.c_str(),
                     hip.row && hip.row->info ? hip.row->info->name : "?", oi.base.str().c_str());
        else {
            /* AND THE SHAPES IT REFUSED, since that is what this line says the finding is about.
             * An error name on its own leaves no way to tell an engine operand the reference
             * genuinely cannot serve from a geometry it was handed wrongly, which is the whole
             * thing a reader comes to this line for. */
            std::string sh;
            for (int i = 0; i < n_opd; ++i)
                if (o.ht[i].data)
                    sh += fmt("%s%d:%s", sh.empty() ? "" : " ", i, shape_str(&o.ht[i]).c_str());
            RAD_WARN("oracle #%lld op '%s': the host kernel (%s) REFUSED the engine's own operands "
                     "with %s. That is a finding about the shapes, not about the oracle: {%s} %s",
                     (long long)seq, oi.op.c_str(),
                     hip.row && hip.row->info ? hip.row->info->name : "?", rad_strerror(rc),
                     oi.base.str().c_str(), sh.c_str());
        }
        return;
    }

    /* Per operand: the largest disagreement, scaled by the device side's own magnitude. A pure
     * input comes out at zero because neither side touched it, which is the check that the
     * snapshot itself is sound. */
    std::string line;
    bool bad = false;
    for (int i = 0; i < n_opd; ++i) {
        if (o.span[i] <= 0) continue;
        /* AN INVERTED WEIGHT IS NOT COMPARED. `pre` holds the stored plane and `post` the device's
         * copy of the same read-only bytes, so there is nothing here the two sides could disagree
         * about -- what the host kernel actually read is the separate logical buffer, in a form
         * neither of those two holds. Printed rather than dropped, so that a reader can see the
         * operand was handled and know which shape the host side saw. */
        if (o.unlaid[i]) {
            line += fmt("  opd%d~ %-6s %-22s  read through %s's own inverse; not compared\n",
                        i, rad_dtype_name(o.ht[i].dtype), shape_str(&o.ht[i]).c_str(),
                        ip_kernel_name(h) ? ip_kernel_name(h) : "?");
            continue;
        }
        const uint32_t dt = o.ht[i].dtype;
        const uint8_t* hp = o.pre[(size_t)o.alias[i]].data();
        const uint8_t* dp = o.post[(size_t)o.alias[i]].data();
        const int64_t  n  = o.span[i];

        double worst = 0, scale = 0, hv = 0, dv = 0, sum = 0, hsum = 0;
        int64_t at = -1, nbad = 0;
        int64_t badix[4] = { -1, -1, -1, -1 };
        double x = 0, y = 0;
        if (!elem(hp, dt, 0, &x)) {
            const bool same = std::memcmp(hp, dp, o.pre[(size_t)o.alias[i]].size()) == 0;
            line += fmt("  opd%d %-6s %s %s\n", i, rad_dtype_name(dt),
                        shape_str(&o.ht[i]).c_str(), same ? "bytes equal" : "BYTES DIFFER");
            bad = bad || !same;
            continue;
        }
        /* ONE CODE STEP IN AN fp8 PLANE IS THE FORMAT, NOT A DISAGREEMENT.
         *
         * E4M3 has three mantissa bits, so the gap between neighbouring codes is between 1/16 and
         * 1/8 of the value -- four to eight times the tolerance a comparison of continuous values
         * is held to. Any difference at all in the bf16 the quantiser reads, down to a single ulp
         * from a different summation order, moves an element that sits near a rounding boundary
         * onto the next code, and the reported relative error is then a whole quantum on a kernel
         * that is right -- typically a handful of elements in ten thousand, one code apart (host
         * 176, device 192), on an fp8 output plane such as rmsnorm_quant_fp8's or hc_read's.
         *
         * So the two sides are compared in CODE space for these dtypes, where adjacent is a
         * question with an exact answer: same sign bit, magnitude codes one apart. Those are
         * COUNTED and reported and do not set the verdict; anything further apart is a real
         * disagreement and is measured like any other. A kernel whose arithmetic is actually
         * wrong does not land one code away on six elements in ten thousand. */
        const bool fp8 = (dt == RAD_F8E4M3 || dt == RAD_F8E5M2);
        int64_t ulp1 = 0;
        for (int64_t k = 0; k < n; ++k) {
            if (!elem(hp, dt, k, &x) || !elem(dp, dt, k, &y)) break;
            scale = std::max(scale, std::fabs(y));
            if (std::isfinite(y)) sum += y;
            if (std::isfinite(x)) hsum += x;
            if (fp8 && hp[k] != dp[k]) {
                const int da = (int)(hp[k] & 0x7Fu) - (int)(dp[k] & 0x7Fu);
                if (((hp[k] ^ dp[k]) & 0x80u) == 0 && (da == 1 || da == -1)) { ++ulp1; continue; }
            }
            const double d = std::fabs(x - y);
            if (std::isnan(x) != std::isnan(y) || d > worst) {
                worst = std::isnan(x) != std::isnan(y) ? INFINITY : d;
                at = k; hv = x; dv = y;
            }
            /* The individually bad elements, not just the worst: WHERE they sit says whether a
             * whole operand is wrong or one lane of it is, and those are different bugs. */
            if (d > 1e-30 && (std::isnan(x) != std::isnan(y) || d > o.tol * std::fabs(x))) {
                if (nbad < 4) badix[nbad] = k;
                ++nbad;
            }
        }
        const double rel = worst / (scale > 0 ? scale : 1.0);
        if (rel > o.tol) bad = true;
        double f[4] = {0, 0, 0, 0};
        for (int64_t k = 0; k < 4 && k < n; ++k) elem(dp, dt, k, &f[k]);
        /* PER-ROW SUMS, up to three rows. A whole-tensor sum is a weak fingerprint and the first
         * four values only ever describe row 0, so between them they can say "token 0 is exactly
         * right and the total is wrong" -- a statement about the OTHER tokens that neither column
         * can localise. A row here is dim 0, which is the token for every activation the graph
         * carries. */
        std::string rows;
        if (o.ht[i].rank >= 2 && o.ht[i].shape[0] > 1 && o.ht[i].stride[0] > 0) {
            const int64_t nr = o.ht[i].shape[0] < 3 ? o.ht[i].shape[0] : 3;
            const int64_t w  = o.ht[i].stride[0];
            for (int64_t r = 0; r < nr; ++r) {
                double rs = 0;
                for (int64_t k = r * w; k < (r + 1) * w && k < n; ++k) {
                    double y = 0;
                    if (!elem(dp, dt, k, &y)) break;
                    if (std::isfinite(y)) rs += y;
                }
                double t[3] = {0, 0, 0};
                const int64_t last = (r + 1) * w < n ? (r + 1) * w : n;
                for (int j = 0; j < 3; ++j)
                    if (last - 3 + j >= r * w) elem(dp, dt, last - 3 + j, &t[j]);
                /* HEAD AND TAIL OF EVERY ROW, which is exactly what llama.cpp's eval-callback
                 * prints -- three at each end and an ellipsis. A sum agrees between two vectors
                 * that are not the same vector, which is the shape these mismatches take. */
                rows += fmt("\n      row%lld sum %.6g head [%.5g %.5g %.5g] tail [%.5g %.5g %.5g]",
                            (long long)r, rs,
                            row_at(dp, dt, r * w, 0, n), row_at(dp, dt, r * w, 1, n),
                            row_at(dp, dt, r * w, 2, n), t[0], t[1], t[2]);
            }
        }
        /* WHERE the worst element is, and what each side said there. Without it a mismatch is a
         * number: "rel 2 on a gate cumsum" cannot distinguish a whole tensor that is wrong from
         * one padding lane that is, and those want different fixes. The index is flat over the
         * operand, which is the same order the row sums above walk. */
        std::string at_s;
        if (at >= 0 && rel > o.tol) {
            at_s = fmt("\n      worst at [%lld]: host %.9g  device %.9g  (diff %.4g)"
                       "\n      %lld of %lld element(s) differ; host sum %.6g vs device sum %.6g"
                       "\n      first bad at",
                       (long long)at, hv, dv, worst, (long long)nbad, (long long)n, hsum, sum);
            for (int b = 0; b < 4 && badix[b] >= 0; ++b) at_s += fmt(" %lld", (long long)badix[b]);
        }
        /* The one-code-step count is PRINTED whether or not anything else disagreed. Silently
         * absorbing them would make the operand look byte-equal, and a reader comparing two runs
         * of the same build wants to see that six elements moved. */
        const std::string u1 = ulp1 ? fmt("  %lld at one fp8 code step", (long long)ulp1)
                                    : std::string();
        line += fmt("  opd%d %-6s %-22s n=%-9lld amax %.4g sum %.6g [%.5g %.5g %.5g %.5g]%s"
                    "  rel %.4g%s%s%s\n",
                    i, rad_dtype_name(dt), shape_str(&o.ht[i]).c_str(), (long long)n, scale,
                    sum, f[0], f[1], f[2], f[3], rows.c_str(), rel, u1.c_str(),
                    rel > o.tol ? "  <-- MISMATCH" : "", at_s.c_str());
    }

    const char* dk = ip_kernel_name(h);
    RAD_WARN("oracle #%lld op '%s' %s [device %s vs host %s] {%s}\n%s",
             (long long)seq, oi.op.c_str(), bad ? "MISMATCH" : "agrees",
             dk, hip.row && hip.row->info ? hip.row->info->name : "?",
             oi.base.str().c_str(), line.c_str());
}

/* The device kernel's name for the report, looked up the same way abort_step does. */
const char* Ctx::ip_kernel_name(rad_op h) const {
    const OpPlan& p = plans_[h];
    const int dom = (int)op_domain_[h].load(std::memory_order_relaxed);
    for (int b = 0; b < p.n_bands; ++b) {
        const IssuePlan& ip = issue_table_[(size_t)p.plan_off + (size_t)b * RAD_N_DOMAINS + dom];
        if (ip.row && ip.row->info) return ip.row->info->name;
    }
    return "?";
}

/* The one entry point issue() calls. Returns the host plan to check against, or null. */
/* ------------------------------------------------------------------ the repeat check */
/* RUN THE DEVICE KERNEL A SECOND TIME ON THE BYTES THE FIRST ONE SAW, AND COMPARE.
 *
 * The oracle above asks whether a kernel agrees with libref. This asks a question no two-sided
 * comparison can: whether the kernel agrees with ITSELF. A kernel with an unordered reduction, a
 * workgroup that reads what another workgroup is writing, or a workspace it assumed was zeroed is
 * CORRECT on the run the oracle happens to look at and wrong on the next one, and every
 * instrument in this tree that runs an op once is blind to it.
 *
 * BYTE-EXACT AND NO TOLERANCE, deliberately. "Within 2%" is the right question for an
 * implementation and the wrong one here: the same kernel on the same bytes has no licence to move
 * at all, and a difference in the last mantissa bit is the finding, because greedy decoding
 * amplifies it into a different token a few hundred steps later.
 *
 * WHAT IT CANNOT SEE. A race between the two ranks of a tensor-parallel step: both Ctxs walk the
 * same graph and both re-launch, so a collective still meets itself and does not deadlock, but
 * what the second launch measures is the pair re-running together rather than either one alone.
 * And a kernel whose output depends on a buffer this declined to restore -- see the cap. */
void Ctx::oracle_repeat(rad_op h, const IssuePlan& dip, int64_t seq, int n_opd) {
    OracleState& o = *oracle_;

    /* AN OPERAND THIS COULD NOT COPY MAKES THE WHOLE ISSUE UNJUDGEABLE, and saying so is the
     * difference between an instrument and a rumour. The cap exists for read-only weights, but
     * nothing here knows that an operand IS a weight -- and a GDN recurrent state is both an input
     * and an output and is far over the default cap. Left unrestored it hands the second launch a
     * state the first one already advanced, and every GDN issue in the model then reports as
     * non-deterministic when what actually happened is that it was asked a different question.
     * Raise RADIANCE_OP_REPEAT_MAXBYTES to check those ops; do not guess about them. */
    for (int i = 0; i < n_opd; ++i)
        if (o.skip[i]) { ++o.unchecked; return; }

    /* The first result, before anything is put back. oracle_snap writes `post`, so it is moved
     * aside rather than copied: the second launch needs the same destination. */
    oracle_snap(n_opd, true);
    std::vector<std::vector<uint8_t>> first;
    first.swap(o.post);

    /* PUT BACK ONLY WHAT THE KERNEL ACTUALLY CHANGED, and decide that by comparing rather than by
     * believing the schema. An op issued {x, h} -> {x} has already overwritten one of its own
     * inputs, so without a restore the second launch is not given the same problem and every
     * in-place op in the graph reports as non-deterministic.
     *
     * Restoring everything SEGFAULTS instead: a small weight is a perfectly ordinary operand under
     * the size cap, and a weight on Tier::Mapped is the container's read-only mmap. A kernel never
     * writes an operand it only reads, so "changed" is both the safe test and the cheap one.
     *
     * A WEIGHT TABLE IS NEVER RESTORED WHATEVER THE COMPARISON SAYS. Its snapshot is the STACKED
     * plane behind an array of pointers (oracle_size), so a byte-for-byte write back would put
     * gigabytes of expert weight over a few kilobytes of pointers. No op writes one, which is
     * why the comparison alone is enough; if one ever did, a write back would take every weight
     * in the graph off its address. */
    bool wrote[MAX_OPERANDS] = {false};
    for (int i = 0; i < n_opd; ++i) {
        if (o.alias[i] != i || o.skip[i] || o.span[i] <= 0 || o.tab[i]) continue;
        const std::vector<uint8_t>& A = o.pre[(size_t)i];
        const std::vector<uint8_t>& B = first[(size_t)i];
        if (A.empty() || A.size() != B.size()) continue;
        if (std::memcmp(A.data(), B.data(), A.size()) == 0) continue;
        wrote[i] = true;
        if (tbuf_[i].data) rad_memcpy_async(tbuf_[i].data, A.data(), A.size(), stream_);
    }
    rad_stream_sync(stream_);

    /* THE SAME PLAN, THE SAME ARGS, THE SAME STREAM. dip.args.t still points at tbuf_ -- the
     * device pointers -- because the oracle's rebinding only ever touched its own `ht`. */
    RadArgs a = dip.args;
    a.n_t = n_opd;
    const int rc = dip.launch(&a, cur_stream());
    if (rc < 0) {
        RAD_WARN("repeat #%lld op '%s': the second launch was REFUSED with %s, so this issue says "
                 "nothing about determinism", (long long)seq, plans_[h].info->op.c_str(),
                 rad_strerror(rc));
        return;
    }
    oracle_snap(n_opd, true);

    std::string line;
    for (int i = 0; i < n_opd; ++i) {
        /* ONLY THE OPERANDS THE KERNEL WROTE. One it merely read is equal by construction, and
         * saying so for every input on every issue buries the finding. */
        if (!wrote[i]) continue;
        const std::vector<uint8_t>& A = first[(size_t)i];
        const std::vector<uint8_t>& B = o.post[(size_t)i];
        if (A.size() != B.size() || A.empty()) continue;
        if (std::memcmp(A.data(), B.data(), A.size()) == 0) continue;

        /* WHICH ELEMENTS AND BY HOW MUCH. One lane apart at the last bit and a whole tensor of
         * garbage are both "BYTES DIFFER", and they are different bugs: the first is a reduction
         * order, the second is a read of memory another workgroup owns. */
        const uint32_t dt = o.ht[i].dtype;
        const int64_t  n  = o.span[i];
        int64_t nbad = 0, at = -1;
        double  worst = 0, av = 0, bv = 0, scale = 0;
        for (int64_t k = 0; k < n; ++k) {
            double x = 0, y = 0;
            if (!elem(A.data(), dt, k, &x) || !elem(B.data(), dt, k, &y)) { nbad = -1; break; }
            if (x == y) continue;
            ++nbad;
            const double d = std::fabs(x - y);
            if (d > worst) { worst = d; at = k; av = x; bv = y; }
            scale = std::max(scale, std::fabs(y));
        }
        if (nbad < 0)
            line += fmt("  opd%d %-6s %s BYTES DIFFER (dtype not scalar)\n", i,
                        rad_dtype_name(dt), shape_str(&o.ht[i]).c_str());
        else
            line += fmt("  opd%d %-6s %s  %lld of %lld element(s) differ; worst at [%lld] "
                        "run1 %.9g run2 %.9g (diff %.4g, rel %.3g)\n",
                        i, rad_dtype_name(dt), shape_str(&o.ht[i]).c_str(),
                        (long long)nbad, (long long)n, (long long)at, av, bv, worst,
                        worst / (scale > 0 ? scale : 1.0));
    }
    if (line.empty()) return;
    ++o.differed;
    RAD_ERR("REPEAT #%lld op '%s' (kernel %s) IS NOT A FUNCTION OF ITS INPUTS: two launches on the "
            "same bytes disagreed.\n%s", (long long)seq, plans_[h].info->op.c_str(),
            ip_kernel_name(h) ? ip_kernel_name(h) : "?", line.c_str());
}

const IssuePlan* Ctx::oracle_before(rad_op h, int band, int n_opd) {
    if (!oracle_ || !oracle_arm(h)) return nullptr;
    oracle_size(h, n_opd);
    if (oracle_->repeat) {
        for (int i = 0; i < n_opd; ++i)
            if (tbuf_[i].data && oracle_opd_bytes(i) <= 0) return nullptr;
        oracle_snap(n_opd, false);
        /* REPEAT MODE NEEDS NO HOST KERNEL, and the issue path carries this pointer back only to
         * say "armed". oracle_after re-launches the DEVICE plan it is handed instead. */
        static const IssuePlan kRepeatArmed{};
        return &kRepeatArmed;
    }
    /* NOT GATED ON THE DEVICE DOMAIN. On the host backend the launch IS the host kernel,
     * so the comparison is host against host and agrees by construction -- but the per-operand
     * lines are still the only ordered trace of what the graph computed, and diffing those
     * against another engine's is what the whole-model reference exists for. */
    const OpPlan& p = plans_[h];
    const IssuePlan& hip =
        issue_table_[(size_t)p.plan_off + (size_t)band * RAD_N_DOMAINS + RAD_DOMAIN_HOST];
    if (!hip.launch) {
        ++oracle_->skipped_nohost;
        if (!oracle_->said_nohost) {
            oracle_->said_nohost = true;
            RAD_WARN("oracle: op '%s' has no host kernel in this band, so it cannot be checked. "
                     "A library with host kernels -- a reference one, or the host library -- "
                     "must be on the kernel search path for the oracle to see anything.", plans_[h].info->op.c_str());
        }
        return nullptr;
    }
    /* AN OPERAND WITH NO BYTE COUNT IS A NAMED SKIP, not a silent one. The count is missing when
     * the dtype is plugin-private and the operand is not a weight the layout pass sized -- a
     * packed plane the core is not meant to understand (spec 4.2). Dropping the whole issue
     * without saying so would report a routed model's expert GEMM as "0 checked" beside a page of
     * agreements, which reads as nothing to check rather than as a gap. */
    for (int i = 0; i < n_opd; ++i) {
        if (!tbuf_[i].data || oracle_opd_bytes(i) > 0) continue;
        RAD_WARN("oracle #%lld op '%s': NOT CHECKED. operand %d is %s and the core cannot size "
                 "it -- a plugin-private dtype is the plugin's to measure, and nothing bound "
                 "this operand to a weight whose stored size the container records.",
                 (long long)n_issues_, plans_[h].info->op.c_str(), i,
                 rad_dtype_name(tbuf_[i].dtype));
        return nullptr;
    }
    /* AND AN OPERAND THE HOST CANNOT HOLD IS A NAMED SKIP TOO. The snapshot is two host copies
     * of every operand and a third for a weight stored re-laid, so an operand of tens of
     * gigabytes is not a slow check -- it is the process being killed with nothing reported, and
     * no reason in the log because there is no chance to write one. A 47 GiB n-gram table read
     * at decode is such an operand. */
    for (int i = 0; i < n_opd; ++i) {
        if (!tbuf_[i].data || oracle_->alias[i] != i) continue;
        const int64_t b = oracle_opd_bytes(i);
        if (b <= oracle_->maxopd) continue;
        const OpInfo& oi = *plans_[h].info;
        const char* wn = nullptr;
        if (program_)
            for (size_t j = 0; j < oi.weights.size() && j < oi.weight_opd.size(); ++j)
                if ((int)oi.weight_opd[j] == i && oi.weights[j]
                    && (size_t)oi.weights[j] < program_->weights.size())
                    wn = program_->weights[(size_t)oi.weights[j]].name.c_str();
        RAD_WARN("oracle #%lld op '%s': NOT CHECKED. operand %d%s%s%s is %lld MiB against a cap of "
                 "%lld MiB, and two host copies of it would end the run instead of reporting on "
                 "it. RADIANCE_OP_ORACLE_MAXBYTES raises the cap.",
                 (long long)n_issues_, oi.op.c_str(), i, wn ? " ('" : "", wn ? wn : "",
                 wn ? "')" : "", (long long)(b >> 20), (long long)(oracle_->maxopd >> 20));
        return nullptr;
    }
    oracle_snap(n_opd, false);
    return &hip;
}

void Ctx::oracle_after(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int n_opd) {
    if (oracle_->repeat) {
        oracle_repeat(h, dip, n_issues_, n_opd);
        --oracle_->budget;
        ++oracle_->checked;
        return;
    }
    oracle_snap(n_opd, true);
    /* The pre snapshot's tensors still point into `pre`, which oracle_snap(false) bound -- except
     * for a weight the device row stores re-laid, which this repoints at its own inverse. A row
     * that cannot be inverted is not checked at all, and not counted as checked either. */
    if (!oracle_unrelayout(h, hip, dip, n_issues_, n_opd)) return;
    oracle_run(h, hip, n_issues_, n_opd);
    --oracle_->budget;
    ++oracle_->checked;
}

}  /* namespace rad */
