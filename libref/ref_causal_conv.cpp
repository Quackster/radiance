/* Depthwise causal convolution without an activation. The rolling state uses the same
 * prefill/verification convention as ple_conv; callers supply the gates separately. */
#include "ref_common.h"
#include "ref_ops.h"

using namespace ref;

int ref_causal_conv1d(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 8) return RAD_E_INVAL;
    const auto *x = t_in(a, 0), *w = t_in(a, 1), *cs = t_in(a, 2), *cu = t_in(a, 3);
    const auto *idx = t_in(a, 4), *init = t_in(a, 5), *accepted = t_in(a, 6), *y = t_in(a, 7);
    if (!x || !w || !cs || !cu || !y) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), width = p_int(a, "width", 0);
    const int64_t rows = n > 0 ? band_rows(numel(x) / n, p_int(a, "M", 0)) : 0;
    if (n <= 0 || width <= 0 || rows <= 0 || cs->rank != 3 || !contig(cs))
        return RAD_E_SHAPE;
    constexpr int64_t max_history = 64;
    const int64_t history = width - 1, depth = cs->shape[2], sequences = numel(cu) - 1;
    if (history > max_history) return RAD_E_UNSUPPORTED;
    if (sequences < 1 || cs->shape[1] != n || depth < history ||
        numel(w) < n * width || numel(y) < rows * n) return RAD_E_SHAPE;
    if ((idx && numel(idx) < sequences) || (init && numel(init) < sequences) ||
        (accepted && numel(accepted) < sequences)) return RAD_E_SHAPE;
    const auto integer = [](const RadTensor* t, int64_t i) {
        return ld_int(t->dtype, t->data, offlin(t, i));
    };
    // A mixed batch passes the prefill suffix of cu while x/y still name all batch rows.
    if (integer(cu, 0) < 0 || integer(cu, sequences) > rows) return RAD_E_SHAPE;
    /* Validate every sequence before mutating any state. Duplicate live slots would race. */
    std::vector<int64_t> slots;
    for (int64_t s = 0; s < sequences; ++s) {
        const auto begin = integer(cu, s), end = integer(cu, s + 1);
        const auto slot = idx ? integer(idx, s) : s;
        const auto offset = accepted ? integer(accepted, s) - 1 : 0;
        if (begin < 0 || end < begin || end > rows || slot < -1 || slot >= cs->shape[0] ||
            offset < 0) return RAD_E_INVAL;
        if (offset > depth - history ||
            (accepted && history && end > begin && end - begin > depth - history + 1))
            return RAD_E_SHAPE;
        if (slot >= 0 && begin != end) {
            for (auto previous : slots) if (slot == previous) return RAD_E_INVAL;
            slots.push_back(slot);
        }
    }
    const auto xr = rowoff(x, 1, n), xc = laststride(x);
    const auto yr = rowoff(y, 1, n), yc = laststride(y);
    const auto wr = rowoff(w, 1, width), wc = laststride(w);
    // State is the conventional packed [slots, channels, rolling window] KV group.
    #pragma omp parallel for schedule(static)
    for (int64_t sc = 0; sc < sequences * n; ++sc) {
        const int64_t s = sc / n, c = sc % n;
        const int64_t begin = integer(cu, s), length = integer(cu, s + 1) - begin;
        const int64_t slot = idx ? integer(idx, s) : s;
        if (length == 0 || slot < 0) continue;
        const bool warm = !init || integer(init, s) != 0;
        const int64_t offset = accepted ? integer(accepted, s) - 1 : 0;
        const int64_t base = (slot * n + c) * depth;
        float prior[max_history];
        for (int64_t i = 0; i < history; ++i)
            prior[i] = warm ? ldt(cs, base + offset + i) : 0.0f;
        for (int64_t t = 0; t < length; ++t) {
            float sum = 0.0f;
            for (int64_t j = 0; j < width; ++j) {
                const int64_t source = t - history + j;
                const float v = source < 0 ? prior[source + history]
                                           : ldt(x, (begin + source) * xr + c * xc);
                sum += v * ldt(w, c * wr + j * wc);
            }
            stt(y, (begin + t) * yr + c * yc, sum);
        }
        if (!history) continue;
        if (accepted) {
            // Keep candidates in the rolling window until the next step selects its offset.
            for (int64_t i = 0; i + 1 < history; ++i) stt(cs, base + i, prior[i + 1]);
            for (int64_t t = 0; t < length; ++t)
                stt(cs, base + history - 1 + t, ldt(x, (begin + t) * xr + c * xc));
        } else {
            for (int64_t i = 0; i < history; ++i) {
                const int64_t source = length - history + i;
                stt(cs, base + i, source < 0 ? prior[source + history]
                                             : ldt(x, (begin + source) * xr + c * xc));
            }
        }
    }
    return RAD_OK;
}
