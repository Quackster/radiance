/* Selected raw logits, copied on the model's stream before the next pass can overwrite them. */
#pragma once

#include "rad_core.h"
#include "device/device.h"

namespace rad {

struct TokenScoreRow {
    std::shared_ptr<TokenScores> result;
    int64_t row = 0;
};

/* One per rank. prepare runs while the rank is parked; issue runs behind its forward pass;
 * collect runs before scheduler commit. Only the selected floats cross the device link. Each
 * rank owns a disjoint vocabulary interval (or only rank zero reads a replicated vocabulary). */
class TokenScoreReadback {
public:
    TokenScoreReadback() = default;
    ~TokenScoreReadback() { reset(); }
    TokenScoreReadback(const TokenScoreReadback&) = delete;
    TokenScoreReadback& operator=(const TokenScoreReadback&) = delete;

    int prepare(const std::vector<TokenScoreRow>& rows, int64_t first_vocab,
                int64_t width, int64_t row_capacity, int device,
                bool compact = false, int64_t full_rows = 0);
    /* Called on the rank's stream before its forward pass. The private batch copy keeps
     * score staging out of the scheduler's shared immutable batch. */
    int bind(RadBatch* batch, RadStream stream);
    int issue(const float* logits, RadStream stream);
    int collect();
    void reset();

private:
    struct Copy {
        std::shared_ptr<TokenScores> result;
        size_t candidate = 0;
        int64_t source = 0;
        int32_t row = 0;
        int32_t token = 0;
    };
    std::vector<Copy> copies_;
    float* host_ = nullptr;
    size_t capacity_ = 0;
    void* device_storage_ = nullptr;
    size_t device_capacity_ = 0;
    int64_t full_rows_ = 0;
    int device_ = 0;
    RadEvent ready_ = nullptr;
    RadStream stream_ = nullptr;
    bool pending_ = false;
    bool recorded_ = false;
    bool active_ = false;
    bool compact_ = false;
    bool staged_ = false;
};

}  // namespace rad
