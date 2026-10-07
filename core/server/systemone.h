#pragma once

#include "oai.h"

namespace rad {
namespace server {

/* d1's answer slot is an ordinary causal LM output. Keep its prompt and verbalizer separate
 * from the engine: the engine returns raw candidate logits and knows no question schema. */
struct DecisionQuestion {
    std::string name;
    std::string type;
    std::vector<std::string> labels;
    std::vector<std::string> descriptions;
    std::string prompt;
    std::vector<int32_t> tokens;
    int32_t shared_prefix_tokens = 0; // exact separately tokenized trunk boundary
    std::vector<int32_t> candidates;
    /* Candidate indices, one group of alternative token spellings per answer. */
    std::vector<std::vector<size_t>> groups;
};

struct DecisionRequest {
    std::vector<DecisionQuestion> questions;
    /* Images precede the shared state, so every question uses these same placements and items.
     * The scheduler queues each Item once and keeps it alive through cancellation/retirement. */
    std::shared_ptr<mm::PromptMedia> media;
    int64_t input_tokens = 0;
};

int parse_d1_request(const std::string& body, const ITokenizer& tok,
                     const std::string& model_id, int64_t max_ctx, int64_t max_questions,
                     DecisionRequest* out, ApiError* err, IMultimodal* mm = nullptr);

/* Max over each answer's token spellings, then softmax over answers, without sampling.
 * Returns an error for missing or non-finite logits rather than inventing probabilities. */
int d1_answer(const DecisionQuestion& q, const std::vector<float>& logits,
              json* out, ApiError* err);

}  /* namespace server */
}  /* namespace rad */
