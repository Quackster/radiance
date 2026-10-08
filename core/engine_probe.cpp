/* engine_probe.cpp -- RADIANCE_DUMP_BUF, the buffer probe.
 *
 * Debug-only and it reads everything, which is why it is not in the step loop's file: it is the
 * one part of the Engine that knows every dtype and every buffer's shape, and none of that
 * belongs beside code that runs a thousand times a second. See engine_priv.h.
 */
#include "engine_priv.h"

#include "build/rad_build.h"
#include "device/device.h"

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace rad {

/* ================================================================== the buffer probe
 *
 * RADIANCE_DUMP_BUF names declared buffers, comma separated, and every step prints the first row
 * of each: its mean, its largest magnitude and its first four elements. RADIANCE_DUMP_HEAD=<n>
 * adds the first n elements verbatim (integers included) and RADIANCE_DUMP_TOPK=<k> the k largest.
 *
 * It exists because nothing else can see INSIDE a step. rad-kbench falsifies one op against the
 * oracle and passes; --debug-graph shows what was declared and resolved; and between them sits a
 * forward pass that answers the same token to every prompt with every op individually correct.
 * The layer bisect narrows that to a block, and this narrows it to a tensor -- which is as far as
 * anything short of a second implementation can go.
 *
 * An environment variable and not a flag, and only at -vv, because it is a device-to-host copy and
 * a synchronisation per step: it changes what it measures, like every probe does. */
/* How much of a row to bring back. Eight kibibytes covers an ordinary hidden state; a row that is
 * a CONCATENATION -- the DFlash2 tap buffer is five hidden states wide -- needs more, and looking
 * at only the first of the five is how a probe reports a healthy tensor whose fifth field is
 * garbage. RADIANCE_DUMP_BYTES raises it. */
static int64_t dump_cap() {
    static const int64_t cap = [] {
        const char* s = std::getenv("RADIANCE_DUMP_BYTES");
        const long long v = s ? std::atoll(s) : 0;
        return v > 0 ? (int64_t)v : (int64_t)8192;
    }();
    return cap;
}

static void dump_row(int step, const char* phase, const char* name, uint32_t dtype,
                     const std::vector<uint8_t>& raw) {
    const int bits = rad_dtype_bits(dtype);
    const int64_t cnt = bits > 0 ? (int64_t)raw.size() * 8 / bits : 0;
    double sum = 0, amax = 0;
    /* NON-FINITE COUNT, beside the mean. A single NaN makes the mean NaN and leaves amax at zero,
     * which reads as "empty" and is the opposite of what happened; and the interesting question
     * about a row that has gone wrong is almost always WHERE, since a row that is finite for its
     * first tap and garbage for its fifth names the bug on its own. */
    int64_t bad = 0, bad_first = -1;
    float first[4] = {0, 0, 0, 0};
    for (int64_t k = 0; k < cnt; ++k) {
        float v = 0;
        if (dtype == RAD_F32) std::memcpy(&v, raw.data() + k * 4, 4);
        else if (dtype == RAD_BF16) {
            uint32_t u = (uint32_t)(raw[(size_t)(k * 2)] | (raw[(size_t)(k * 2 + 1)] << 8));
            u <<= 16; std::memcpy(&v, &u, 4);
        }
        else if (dtype == RAD_F8E4M3) {
            /* E4M3-fn, decoded here rather than skipped: a block-scaled fp8 weight IS the model,
             * and "is this the tensor the name map named" has to be answerable about it too. */
            const uint8_t c = raw[(size_t)k];
            const uint32_t sgn = (uint32_t)(c & 0x80u) << 24;
            const int      e   = (c >> 3) & 0x0F;
            uint32_t       m   = c & 0x07u, u;
            if (e == 0) {
                if (!m) u = sgn;
                else { int ex = -6; while (!(m & 0x08u)) { m <<= 1; --ex; }
                       u = sgn | ((uint32_t)(ex + 127) << 23) | ((m & 0x07u) << 20); }
            } else if (e == 0x0F && m == 0x07) u = sgn | 0x7FC00000u;
            else u = sgn | ((uint32_t)(e - 7 + 127) << 23) | (m << 20);
            std::memcpy(&v, &u, 4);
        }
        else continue;                        /* the index types have no meaningful mean */
        if (!std::isfinite(v)) { if (bad_first < 0) bad_first = k; ++bad; }
        else { sum += v; amax = std::max(amax, (double)std::fabs(v)); }
        if (k < 4) first[k] = v;
    }
    if (!cnt) return;
    /* A BIT-EXACT CHECKSUM OF THE RAW BYTES, beside the summary. The mean is what a reader wants
     * for "is this tensor plausible"; it is useless for "is this run the same as the last one",
     * which is the other question this probe gets asked: a single logit moving by 1e-3 shifts the
     * mean of a vocabulary's worth of them by 1e-8 and prints identical at %.6f. FNV-1a over the
     * bytes answers that one exactly, and it is the cheap half of the loop. */
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < raw.size(); ++i) { h ^= raw[i]; h *= 1099511628211ull; }
    const int64_t good = cnt - bad;
    RAD_TRACE("step %d%-8s %-28s %-5s n=%lld  hash %016llx  mean %.6f  amax %.5f  "
              "nonfinite %lld%s  [%.6f %.6f %.6f %.6f]",
              step, phase, name, rad_dtype_name(dtype), (long long)cnt, (unsigned long long)h,
              good > 0 ? sum / (double)good : 0.0, amax, (long long)bad,
              bad ? fmt(" from %lld", (long long)bad_first).c_str() : "",
              (double)first[0], (double)first[1], (double)first[2], (double)first[3]);

    /* ---- THE FIRST N ELEMENTS, IN THE ROW'S OWN TYPE ----------------------------------------
     *
     * `RADIANCE_DUMP_HEAD=<n>` prints n elements verbatim, integers included. The summary above
     * decodes only the float types, so an INDEX plane reports mean 0 and four zeros -- which is
     * a fair answer to "is this tensor plausible" and no answer at all to "which tokens".
     *
     * That second question is what the sampler's candidate planes exist to answer. After
     * sample_merge_topk, `sampler.cand_idx` holds GLOBAL token ids and `sampler.cand_val` their
     * logits, descending -- so the pair IS this engine's next-token distribution, read at the one
     * place it is ever assembled whole across the vocabulary shards.
     *
     * WHY THAT IS ENOUGH FOR A KL DIVERGENCE, with no log-sum-exp anywhere. Renormalised over a
     * fixed candidate set S, p~_i = exp(l_i) / sum_{j in S} exp(l_j) -- the partition function
     * over the WHOLE vocabulary cancels, in both engines independently. So a top-K KLD against
     * another engine's top-K logprobs needs only these logits. The OpenAI `logprobs` field carries
     * the normaliser (sample_logprobs) and only the RAW distribution's top 20; this reads the
     * plane the pick drew from, which is a probe, and it says so by being an environment variable
     * at -vv rather than a response field. */
    static const int head = [] {
        const char* s = std::getenv("RADIANCE_DUMP_HEAD");
        const long v = (s && *s) ? std::strtol(s, nullptr, 10) : 0;
        return (int)(v > 0 ? (v < 4096 ? v : 4096) : 0);
    }();
    if (head > 0) {
        const int64_t n = std::min<int64_t>(head, cnt);
        std::string hl;
        for (int64_t k = 0; k < n; ++k) {
            float f = 0;
            switch (dtype) {
            case RAD_I32: {
                int32_t v; std::memcpy(&v, raw.data() + k * 4, 4);
                hl += fmt(" %d", (int)v); continue;
            }
            case RAD_U32: {
                uint32_t v; std::memcpy(&v, raw.data() + k * 4, 4);
                hl += fmt(" %u", (unsigned)v); continue;
            }
            case RAD_F32: std::memcpy(&f, raw.data() + k * 4, 4); break;
            case RAD_BF16: {
                uint32_t u = (uint32_t)(raw[(size_t)(k * 2)] | (raw[(size_t)(k * 2 + 1)] << 8));
                u <<= 16; std::memcpy(&f, &u, 4); break;
            }
            default: hl += " ?"; continue;
            }
            hl += fmt(" %.6f", (double)f);
        }
        RAD_TRACE("step %d%-8s %-28s head%s", step, phase, name, hl.c_str());
    }

    /* ---- AND THE TOP K OF THE ROW, A GRADED COMPARISON FROM A SERVING ENGINE ---------------
     *
     * `RADIANCE_DUMP_TOPK=<k>` adds one `topk` line per dumped row: the k largest values and where
     * they are. The hash above answers "is this run the same as the last one" and nothing else,
     * and the mean of a vocabulary of logits is identical at %.6f long after the model has started
     * writing different code -- so the question a LOSSY LEVER asks ("how much worse, if at all")
     * needs an instrument of its own.
     *
     * WHY THIS SHAPE AND NOT A PERPLEXITY. The server's logprobs are the generated positions'
     * only -- never the prompt's -- and a served prefill produces logits only at the rows
     * `out_ids` names; the teacher-forced scoring pass is the KL mode's (core/kld.h), which runs in
     * place of the server. What a serving engine offers: send N distinct prompts at max_tokens = 1,
     * and every build under comparison sees IDENTICAL inputs at one row each. The top of that row,
     * over enough prompts, is a next-token distribution comparison -- top-1 agreement, rank
     * displacement, and the gap between the first and second logit, which is what decides whether a
     * perturbation can flip the token at all.
     *
     * WHY A GENERATED CONTINUATION IS NOT EVIDENCE. Divergence at temperature 0 is chaotic: two
     * numeric formats can agree for hundreds of tokens and then differ arbitrarily, so one
     * continuation cannot rank them and the difference it shows says nothing about which is
     * closer to the unquantised model. N prompts at one row each can. */
    static const int topk = [] {
        const char* s = std::getenv("RADIANCE_DUMP_TOPK");
        const long v = (s && *s) ? std::strtol(s, nullptr, 10) : 0;
        return (int)(v > 0 ? (v < 64 ? v : 64) : 0);
    }();
    if (topk <= 0 || cnt <= 0) return;
    /* A partial selection, not a sort: n is the whole vocabulary and this runs on the step path. */
    std::vector<std::pair<float, int64_t>> best;
    best.reserve((size_t)topk + 1);
    for (int64_t k = 0; k < cnt; ++k) {
        float v = 0;
        if (dtype == RAD_F32) std::memcpy(&v, raw.data() + k * 4, 4);
        else if (dtype == RAD_BF16) {
            uint32_t u = (uint32_t)(raw[(size_t)(k * 2)] | (raw[(size_t)(k * 2 + 1)] << 8));
            u <<= 16; std::memcpy(&v, &u, 4);
        } else continue;
        if (!std::isfinite(v)) continue;
        if ((int)best.size() < topk) {
            best.emplace_back(v, k);
            std::push_heap(best.begin(), best.end(), std::greater<>());
        } else if (v > best.front().first) {
            std::pop_heap(best.begin(), best.end(), std::greater<>());
            best.back() = { v, k };
            std::push_heap(best.begin(), best.end(), std::greater<>());
        }
    }
    std::sort(best.begin(), best.end(), std::greater<>());
    std::string line;
    for (const auto& b : best) line += fmt(" %lld:%.5f", (long long)b.second, (double)b.first);
    RAD_TRACE("step %d%-8s %-28s topk%s", step, phase, name, line.c_str());
}

/* Whether `name` answers to `want`: the whole name, or its last scope components -- "rank0.h" answers
 * to "h" -- and never a bare suffix, or "moe_logits" would answer to "logits" and, declared first,
 * be the one dumped. */
static bool answers_to(const std::string& name, const std::string& want) {
    if (name.size() < want.size() ||
        name.compare(name.size() - want.size(), want.size(), want) != 0) return false;
    return name.size() == want.size() || name[name.size() - want.size() - 1] == '.';
}

void dump_buffers(Ctx& ctx, int step, const char* names, const char* phase) {
    if (!names || !*names) return;
    const Program& p = *ctx.program();
    for (const char* s = names; *s; ) {
        const char* e = s;
        while (*e && *e != ',') ++e;
        const std::string want(s, (size_t)(e - s));
        s = *e ? e + 1 : e;
        if (want.empty()) continue;

        /* WEIGHTS ANSWER TO THE SAME NAMES. A declared weight is bytes the engine chose to put
         * somewhere, exactly as a buffer is, and "did the loader put THIS checkpoint tensor in
         * THIS declared slot" has no other answer short of reading the container by hand. The
         * name map is one line per weight and a wrong one is a model that runs and means nothing:
         * output.weight falling back to token_embd, say, gives a model that copies its prompt
         * fluently and knows nothing. */
        for (size_t i = 1; i < p.weights.size(); ++i) {
            const WeightInfo& wi = p.weights[i];
            if (!answers_to(wi.name, want)) continue;
            void* ptr = ctx.weight_ptr((rad_weight)i);
            if (!ptr) continue;
            /* A rank-1 weight IS one row -- a norm gain, A_log, dt_bias. Dividing by shape[0]
             * there shows a single element, which is exactly enough to miss a conversion that
             * got the first value right and every one after it wrong. */
            const int64_t row = wi.decl.rank >= 2
                                    ? wi.stored_bytes / (wi.decl.shape[0] ? wi.decl.shape[0] : 1)
                                    : wi.stored_bytes;
            const int64_t n = std::min<int64_t>(row > 0 ? row : wi.stored_bytes, dump_cap());
            std::vector<uint8_t> raw((size_t)n);
            if (n <= 0 || rad_memcpy_async(raw.data(), ptr, n, ctx.stream()) < 0) continue;
            rad_stream_sync(ctx.stream());
            dump_row(step, phase, wi.name.c_str(), wi.decl.dtype, raw);
            goto next;
        }

        for (size_t i = 1; i < p.buffers.size(); ++i) {
            const BufferInfo& b = p.buffers[i];
            if (!answers_to(b.name, want)) continue;
            void* ptr = ctx.buf_ptr((rad_buf)i);
            if (!ptr) continue;

            const int64_t row = b.decl.rank >= 1 ? b.bytes / (b.decl.shape[0] ? b.decl.shape[0] : 1)
                                                 : b.bytes;
            const int64_t n = std::min<int64_t>(row, dump_cap());
            std::vector<uint8_t> raw((size_t)n);
            if (rad_memcpy_async(raw.data(), ptr, n, ctx.stream()) < 0) continue;
            rad_stream_sync(ctx.stream());

            dump_row(step, phase, b.name.c_str(), b.decl.dtype, raw);
            break;
        }
        next:;
    }
}

}  /* namespace rad */
