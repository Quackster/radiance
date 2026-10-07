/* engine_mm.cpp -- the encoder driver (spec §11).
 *
 * The scheduler holds back a request whose media is not encoded (Scheduler::media_ready) and hands
 * its items here. They are encoded BETWEEN steps, never inside one: a pass is issued with no step
 * in flight, its rows retained, and the item marked done, so the next step's admission finds its
 * features ready for the chunk that reaches them. Nothing on the
 * step path ever waits on the tower.
 *
 * A PASS IS WHOLE SEGMENTS. An image is one segment and a video one per temporal group, and the
 * encoder attends within a segment, so a pass may end between two segments of a video and never
 * inside one; the processor has already kept every segment within a pass. Several items share a
 * pass when they fit, which is what a request with a handful of small pictures wants.
 *
 * EVERY RANK RUNS THE WHOLE TOWER on its own card from its own copy of the patches. Rank zero's
 * result supplies the embeddings; other ranks contribute zero rows before the vocabulary
 * reduction. Retained features use the budgeted device pool, with host storage for overflow.
 */
#include "engine_priv.h"

#include "mm/processor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace rad {

int Engine::run_encodes() {
    std::vector<std::shared_ptr<mm::Item>> items;
    sched_.take_encodes(&items);
    if (items.empty()) return RAD_OK;

    const RadEncoderDecl& enc = ranks_[0]->program.encoder;
    const int64_t max_p = sched_.encoder_patches();
    const int64_t n_embd = enc.n_embd;
    Rank& r0 = *ranks_[0];
    void* out = r0.ctx.buf_ptr(enc.out);
    if (!out) {
        RAD_ERR("encoder '%s': its output buffer was declared but never planned",
                ranks_[0]->program.encoder_name.c_str());
        return RAD_E_STATE;
    }
    /* A DIAGNOSTIC: RADIANCE_DEBUG_MM_DUMP=<dir> writes each item's patches (`.pix.bf16`, the
     * processor's output in merge-block order) and its rows (`.bf16`) there as raw bf16, named by
     * its content key -- the tower's input and output, for comparison against a reference encoder
     * fed the same input. */
    static const char* const dump_dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
    auto dump = [&](const mm::Item& it, const char* ext, const uint16_t* p, size_t n) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/mm_%016llx%016llx_%dx%dx%d%s", dump_dir,
                      (unsigned long long)it.key_hi, (unsigned long long)it.key_lo, it.grid_t,
                      it.grid_h, it.grid_w, ext);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(p, sizeof(uint16_t), n, f);
            std::fclose(f);
            RAD_INFO("encoder: wrote %s", path);
        }
    };

    for (auto& ip : items) {
        if (ip->encoded) continue;
        RAD_TRY(sched_.reserve_media_rows(*ip));
        if (!ip->device_rows || (dump_dir && *dump_dir))
            ip->embd.assign((size_t)(ip->n_rows() * n_embd), 0);
        if (dump_dir && *dump_dir) dump(*ip, ".pix.bf16", ip->pixels.data(), ip->pixels.size());
    }

    size_t i = 0;
    int32_t seg = 0;
    while (i < items.size()) {
        std::vector<BatchBuilder::EncoderPart> parts;
        int64_t np = 0;
        while (i < items.size()) {
            mm::Item& it = *items[i];
            if (it.encoded) { ++i; seg = 0; continue; }
            const int32_t begin = seg;
            while (seg < it.n_seg) {
                const int64_t seg_p = it.segment_patches(seg);
                if (seg_p > max_p) {
                    RAD_ERR("encoder: a segment of %lld patches does not fit a pass of %lld",
                            (long long)seg_p, (long long)max_p);
                    return RAD_E_STATE;
                }
                if (np + seg_p > max_p) break;
                np += seg_p;
                ++seg;
            }
            if (seg == begin) break;
            parts.push_back({ &it, begin, seg - begin });
            if (seg == it.n_seg) { ++i; seg = 0; }
        }
        if (parts.empty()) break;

        const RadBatch* b = sched_.build_encoder(parts, enc_step_++);
        if (!b) return RAD_E_STATE;
        RAD_TRY(prepare_release(b));
        /* kIssueModel: the pass and nothing behind it -- no sampler, and no tiering tick, which
         * belongs to steps. */
        issue_what_.store(kIssueModel, std::memory_order_release);
        step_batch_.store(b, std::memory_order_release);
        barrier_->arrive_and_wait();
        barrier_->arrive_and_wait();
        const int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("encoder pass failed: %s -- the engine cannot continue", rad_strerror(st));
            return st;
        }
        /* Keep the rows on rank zero, whose embeddings are injected before the
         * vocabulary reduction. The host copy is only needed by the host backend,
         * pool overflow, or an explicit diagnostic dump. */
        int64_t row = 0;
        for (const BatchBuilder::EncoderPart& p : parts) {
            mm::Item& it = *const_cast<mm::Item*>(p.item);
            const int64_t first = it.row_at(p.seg0);
            const int64_t rows = it.row_at(p.seg0 + p.n_seg) - first;
            const int64_t bytes = rows * n_embd * (int64_t)sizeof(uint16_t);
            const char* src = static_cast<const char*>(out) +
                              row * n_embd * (int64_t)sizeof(uint16_t);
            if (it.device_rows) {
                int64_t copied = 0;
                for (const auto& span : it.device_rows->spans) {
                    const int64_t lo = std::max(first, span.row);
                    const int64_t hi = std::min(first + rows, span.row + span.count);
                    if (lo >= hi) continue;
                    RAD_TRY(rad_memcpy_async(
                        static_cast<char*>(span.data) + (lo - span.row) * n_embd * (int64_t)sizeof(uint16_t),
                        src + (lo - first) * n_embd * (int64_t)sizeof(uint16_t),
                        (hi - lo) * n_embd * (int64_t)sizeof(uint16_t), r0.ctx.stream()));
                    copied += hi - lo;
                }
                if (copied != rows) return RAD_E_STATE;
            }
            if (!it.embd.empty())
                RAD_TRY(rad_memcpy_async(it.embd.data() + first * n_embd,
                                         src, bytes, r0.ctx.stream()));
            row += rows;
        }
        /* Every rank's stream, not only rank 0's: the next pass rewrites the staging every rank's
         * copy reads, and the arena every rank's tower wrote. */
        for (auto& rk : ranks_) RAD_TRY(rad_stream_sync(rk->ctx.stream()));
    }

    for (auto& ip : items) {
        mm::Item& it = *ip;
        if (it.encoded) continue;
        it.encoded = true;
        it.pixels.clear();
        it.pixels.shrink_to_fit();
        if (dump_dir && *dump_dir) dump(it, ".bf16", it.embd.data(), it.embd.size());
    }
    RAD_DEBUG("encoder: %zu item(s) encoded", items.size());
    return RAD_OK;
}

}  /* namespace rad */
