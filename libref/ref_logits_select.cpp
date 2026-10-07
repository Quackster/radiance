/* Project explicitly requested (hidden row, vocabulary token) pairs. The full vocabulary is
 * unnecessary for classifiers which normalize only their supplied candidates. */
#include "ref_common.h"
#include "ref_ops.h"

using namespace ref;

int ref_logits_select(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const auto *x = &a->t[0], *w = &a->t[1], *rows = &a->t[2], *tokens = &a->t[3], *y = &a->t[4];
    if (rows->dtype != RAD_I32 || tokens->dtype != RAD_I32 || y->dtype != RAD_F32)
        return RAD_E_DTYPE;
    const int64_t K = p_int(a, "n_embd", 0), V = p_int(a, "n_vocab", 0);
    const int64_t round = p_int(a, "round_bf16", 0);
    if (K <= 0 || V <= 0 || (round != 0 && round != 1)) return RAD_E_INVAL;
    const int64_t R = numel(x) / K, M = numel(rows);
    if (R <= 0 || M <= 0 || numel(tokens) != M || numel(y) != M || numel(w) < V * K)
        return RAD_E_SHAPE;
    auto index = [](const RadTensor* t, int64_t i) { return ld_int(t->dtype, t->data, offlin(t, i)); };
    /* Validate first: a bad pair must not publish a partly computed answer. */
    for (int64_t i = 0; i < M; ++i)
        if (index(rows, i) < 0 || index(rows, i) >= R || index(tokens, i) < 0 || index(tokens, i) >= V)
            return RAD_E_INVAL;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < M; ++i) {
        const int64_t xb = rowoff(x, index(rows, i), K), wb = rowoff(w, index(tokens, i), K);
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k)
            sum += ldt(x, xb + k * laststride(x)) * ldt(w, wb + k * laststride(w));
        if (round) {
            uint16_t value;
            st_dt(RAD_BF16, &value, 0, sum);
            sum = ld_dt(RAD_BF16, &value, 0);
        }
        stt(y, offlin(y, i), sum);
    }
    return RAD_OK;
}
