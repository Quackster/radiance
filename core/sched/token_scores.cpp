#include "token_scores.h"
#include "rad_internal.h"

namespace rad {

int TokenScoreReadback::prepare(const std::vector<TokenScoreRow>& rows, int64_t first_vocab,
                               int64_t width, int64_t row_capacity, int device,
                               bool compact, int64_t full_rows) {
    if (pending_ || staged_) return RAD_E_STATE;
    if (first_vocab < 0 || width < 0 || row_capacity < 0 || full_rows < 0 ||
        full_rows > row_capacity || width > INT32_MAX || row_capacity > INT32_MAX)
        return RAD_E_INVAL;
    copies_.clear();
    active_ = !rows.empty();
    device_ = device;
    compact_ = compact;
    full_rows_ = full_rows;
    for (const auto& r : rows) {
        if (!r.result || r.row < 0 || r.row >= row_capacity ||
            r.result->tokens.size() != r.result->logits.size()) return RAD_E_INVAL;
        for (size_t i = 0; i < r.result->tokens.size(); ++i) {
            const int64_t local = (int64_t)r.result->tokens[i] - first_vocab;
            if (local >= 0 && local < width)
                copies_.push_back({r.result, i, r.row * width + local,
                                   (int32_t)r.row, (int32_t)local});
        }
    }
    if (!active_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    if (copies_.size() > capacity_) {
        if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
        host_ = static_cast<float*>(rad_dev_alloc((int64_t)copies_.size() * 3 * sizeof(float),
                                                 RAD_MEM_HOST_PINNED));
        capacity_ = host_ ? copies_.size() : 0;
        if (!host_) return RAD_E_NOMEM;
    }
    if (compact_ && copies_.size() > device_capacity_) {
        if (device_storage_) rad_dev_free(device_storage_, RAD_MEM_DEVICE);
        device_storage_ = rad_dev_alloc((int64_t)copies_.size() * 3 * sizeof(float), RAD_MEM_DEVICE);
        device_capacity_ = device_storage_ ? copies_.size() : 0;
        if (!device_storage_) return RAD_E_NOMEM;
    }
    if (compact_ && !copies_.empty()) {
        auto* rows_h = reinterpret_cast<int32_t*>(host_ + capacity_);
        auto* tokens_h = reinterpret_cast<int32_t*>(host_ + 2 * capacity_);
        for (size_t i = 0; i < copies_.size(); ++i) {
            rows_h[i] = copies_[i].row;
            tokens_h[i] = copies_[i].token;
        }
    }
    /* The host reads device-written pinned memory after this event, so its release must have
     * system scope. A local event is only sufficient for ordering another device operation. */
    if (!ready_) RAD_TRY(rad_event_create(&ready_));
    return RAD_OK;
}

int TokenScoreReadback::bind(RadBatch* batch, RadStream stream) {
    if (!batch) return RAD_E_INVAL;
    batch->n_score_values = 0;
    batch->score_rows = batch->score_tokens = nullptr;
    batch->score_values = nullptr;
    batch->n_full_logits = batch->n_out;
    if (!active_ || !compact_ || copies_.empty()) return RAD_OK;
    if (pending_ || staged_ || !device_storage_ || !host_) return RAD_E_STATE;
    stream_ = stream;
    staged_ = true;
    auto* values = static_cast<float*>(device_storage_);
    auto* rows_d = reinterpret_cast<int32_t*>(values + device_capacity_);
    auto* tokens_d = reinterpret_cast<int32_t*>(values + 2 * device_capacity_);
    RAD_TRY(rad_memcpy_async(rows_d, host_ + capacity_, copies_.size() * sizeof(int32_t), stream));
    RAD_TRY(rad_memcpy_async(tokens_d, host_ + 2 * capacity_, copies_.size() * sizeof(int32_t), stream));
    batch->n_score_values = (int64_t)copies_.size();
    batch->score_rows = rows_d;
    batch->score_tokens = tokens_d;
    batch->score_values = values;
    batch->n_full_logits = full_rows_;
    return RAD_OK;
}

int TokenScoreReadback::issue(const float* logits, RadStream stream) {
    if (!active_) return RAD_OK;
    if ((!copies_.empty() && (!host_ || (compact_ ? !staged_ : !logits))) ||
        !ready_ || pending_) return RAD_E_STATE;
    stream_ = stream;
    pending_ = true;
    recorded_ = false;
    if (compact_ && !copies_.empty())
        RAD_TRY(rad_memcpy_async(host_, device_storage_, copies_.size() * sizeof(float), stream));
    else for (size_t i = 0; i < copies_.size(); ++i)
        RAD_TRY(rad_memcpy_async(host_ + i, logits + copies_[i].source, sizeof(float), stream));
    RAD_TRY(rad_event_record(ready_, stream));
    recorded_ = true;
    return RAD_OK;
}

int TokenScoreReadback::collect() {
    if (!active_) return RAD_OK;
    if (!pending_ || !recorded_) return RAD_E_STATE;
    RAD_TRY(rad_dev_set(device_));
    RAD_TRY(rad_event_sync(ready_));
    for (size_t i = 0; i < copies_.size(); ++i)
        copies_[i].result->logits[copies_[i].candidate] = host_[i];
    pending_ = false;
    staged_ = false;
    return RAD_OK;
}

void TokenScoreReadback::reset() {
    if (host_ || ready_ || device_storage_) {
        (void)rad_dev_set(device_);
        if (pending_ || staged_) (void)rad_stream_sync(stream_);
        if (ready_) rad_event_destroy(ready_);
        if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
        if (device_storage_) rad_dev_free(device_storage_, RAD_MEM_DEVICE);
    }
    host_ = nullptr;
    ready_ = nullptr;
    capacity_ = 0;
    device_storage_ = nullptr;
    device_capacity_ = 0;
    pending_ = recorded_ = staged_ = false;
    active_ = false;
    copies_.clear();
}

}  // namespace rad
