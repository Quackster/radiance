#include "rad_test.h"
#include "sched/token_scores.h"
#include "server/adapters.h"

#include <algorithm>

using namespace rad;

namespace {
struct DeviceRows {
    RadStream stream = nullptr;
    float* logits = nullptr;
    std::vector<float> values;
    DeviceRows(int rows, int width) : values((size_t)rows * width) {
        for (size_t i = 0; i < values.size(); ++i) values[i] = (float)i * 0.25f - 3.0f;
        if (rad_stream_create(&stream, 0) < 0) return;
        logits = static_cast<float*>(rad_dev_alloc((int64_t)values.size() * sizeof(float), RAD_MEM_DEVICE));
        if (logits) (void)rad_memcpy_async(logits, values.data(),
                                         (int64_t)values.size() * sizeof(float), stream);
    }
    ~DeviceRows() {
        if (stream) (void)rad_stream_sync(stream);
        if (logits) rad_dev_free(logits, RAD_MEM_DEVICE);
        if (stream) rad_stream_destroy(stream);
    }
};

std::shared_ptr<TokenScores> request(std::initializer_list<int32_t> tokens) {
    auto r = std::make_shared<TokenScores>();
    r->tokens = tokens;
    r->logits.assign(tokens.size(), -999.0f);
    return r;
}
}  // namespace

TEST(selected_logits_are_captured_before_the_next_stream_operation_overwrites_them) {
    DeviceRows dev(3, 8);
    REQUIRE(dev.stream && dev.logits);
    auto a = request({7, 1, 7});
    auto b = request({0, 3});
    TokenScoreReadback scores;
    CHECK_OK(scores.prepare({{a, 2}, {b, 0}}, 0, 8, 3, 0));
    CHECK_OK(scores.issue(dev.logits, dev.stream));
    CHECK_EQ(a->logits[0], -999.0f);  // published only at collection
    CHECK_EQ(scores.prepare({}, 0, 8, 3, 0), RAD_E_STATE);
    CHECK_OK(rad_memset_async(dev.logits, 0, (int64_t)dev.values.size() * sizeof(float), dev.stream));
    CHECK_OK(scores.collect());
    CHECK_EQ(a->logits[0], dev.values[23]);
    CHECK_EQ(a->logits[1], dev.values[17]);
    CHECK_EQ(a->logits[2], dev.values[23]);
    CHECK_EQ(b->logits[0], dev.values[0]);
    CHECK_EQ(b->logits[1], dev.values[3]);
}

TEST(vocabulary_shards_write_only_the_candidates_they_own) {
    DeviceRows left(2, 4), right(2, 3);
    REQUIRE(left.stream && left.logits && right.stream && right.logits);
    auto result = request({6, 0, 4, 3, 4});
    TokenScoreReadback a, b;
    CHECK_OK(a.prepare({{result, 1}}, 0, 4, 2, 0));
    CHECK_OK(b.prepare({{result, 1}}, 4, 3, 2, 0));
    CHECK_OK(a.issue(left.logits, left.stream));
    CHECK_OK(b.issue(right.logits, right.stream));
    CHECK_OK(a.collect());
    CHECK_EQ(result->logits[0], -999.0f);
    CHECK_EQ(result->logits[1], left.values[4]);
    CHECK_EQ(result->logits[3], left.values[7]);
    CHECK_OK(b.collect());
    CHECK_EQ(result->logits[0], right.values[5]);
    CHECK_EQ(result->logits[2], right.values[3]);
    CHECK_EQ(result->logits[4], right.values[3]);
}

TEST(compact_scores_stage_local_vocabulary_pairs_and_preserve_mixed_generation_rows) {
    DeviceRows dev(3, 8);
    REQUIRE(dev.stream && dev.logits);
    auto result = request({11, 8, 11, 3});
    TokenScoreReadback scores;
    CHECK_OK(scores.prepare({{result, 2}}, 8, 4, 3, 0, true, 2));
    RadBatch batch{};
    batch.n_out = 3;
    CHECK_OK(scores.bind(&batch, dev.stream));
    CHECK_EQ(batch.n_full_logits, (int64_t)2);
    CHECK_EQ(batch.n_score_values, (int64_t)3);
    int32_t rows[3]{}, tokens[3]{};
    CHECK_OK(rad_memcpy_async(rows, batch.score_rows, sizeof(rows), dev.stream));
    CHECK_OK(rad_memcpy_async(tokens, batch.score_tokens, sizeof(tokens), dev.stream));
    CHECK_OK(rad_stream_sync(dev.stream));
    CHECK_EQ(rows[0], 2); CHECK_EQ(rows[1], 2); CHECK_EQ(rows[2], 2);
    CHECK_EQ(tokens[0], 3); CHECK_EQ(tokens[1], 0); CHECK_EQ(tokens[2], 3);
    CHECK_EQ(scores.prepare({}, 0, 4, 3, 0), RAD_E_STATE);
    const float projected[] = {0.75f, -2.0f, 0.75f};
    CHECK_OK(rad_memcpy_async(batch.score_values, projected, sizeof(projected), dev.stream));
    CHECK_OK(scores.issue(nullptr, dev.stream));
    CHECK_OK(rad_memset_async(batch.score_values, 0, sizeof(projected), dev.stream));
    CHECK_OK(scores.collect());
    CHECK_EQ(result->logits[0], 0.75f);
    CHECK_EQ(result->logits[1], -2.0f);
    CHECK_EQ(result->logits[2], 0.75f);
    CHECK_EQ(result->logits[3], -999.0f); // owned by another vocabulary shard
    CHECK_OK(scores.prepare({}, 0, 4, 3, 0, true, 0));
    CHECK_OK(scores.bind(&batch, dev.stream));
    CHECK_EQ(batch.n_score_values, (int64_t)0);
    CHECK_EQ(batch.n_full_logits, (int64_t)3);
    CHECK(batch.score_values == nullptr);
}

TEST(compact_rank_without_owned_candidates_uses_the_full_head_and_still_fences) {
    DeviceRows dev(1, 4);
    REQUIRE(dev.stream && dev.logits);
    auto result = request({7});
    TokenScoreReadback scores;
    CHECK_OK(scores.prepare({{result, 0}}, 0, 4, 1, 0, true, 0));
    RadBatch batch{};
    batch.n_out = 1;
    CHECK_OK(scores.bind(&batch, dev.stream));
    CHECK_EQ(batch.n_score_values, (int64_t)0);
    CHECK_EQ(batch.n_full_logits, (int64_t)1);
    CHECK_OK(scores.issue(nullptr, dev.stream));
    CHECK_OK(scores.collect());
    CHECK_EQ(result->logits[0], -999.0f);
}

TEST(a_readback_keeps_an_abandoned_requests_result_alive_until_the_copy_finishes) {
    DeviceRows dev(1, 4);
    REQUIRE(dev.stream && dev.logits);
    TokenScoreReadback scores;
    auto result = request({1});
    std::weak_ptr<TokenScores> lifetime = result;
    CHECK_OK(scores.prepare({{result, 0}}, 0, 4, 1, 0));
    CHECK_OK(scores.issue(dev.logits, dev.stream));
    result.reset();
    CHECK(!lifetime.expired());
    CHECK_OK(scores.collect());
    CHECK_EQ(lifetime.lock()->logits[0], dev.values[1]);
    CHECK_OK(scores.prepare({}, 0, 4, 1, 0));
    CHECK(lifetime.expired());
}

TEST(a_rank_with_no_selected_tokens_still_fences_its_forward_pass) {
    DeviceRows dev(1, 4);
    REQUIRE(dev.stream && dev.logits);
    auto result = request({7});
    TokenScoreReadback scores;
    CHECK_OK(scores.prepare({{result, 0}}, 0, 4, 1, 0));
    CHECK_OK(scores.issue(nullptr, dev.stream));
    CHECK_OK(scores.collect());
    CHECK_EQ(result->logits[0], -999.0f);
}

TEST(readback_refuses_invalid_rows_and_incomplete_request_storage) {
    auto result = request({0});
    TokenScoreReadback scores;
    CHECK_EQ(scores.prepare({{result, -1}}, 0, 4, 1, 0), RAD_E_INVAL);
    CHECK_EQ(scores.prepare({{result, 1}}, 0, 4, 1, 0), RAD_E_INVAL);
    result->logits.clear();
    CHECK_EQ(scores.prepare({{result, 0}}, 0, 4, 1, 0), RAD_E_INVAL);
}

TEST(a_scheduler_refusal_finishes_the_scoring_sink_instead_of_leaving_http_waiting) {
    Scheduler scheduler;
    server::SchedulerBridge bridge(&scheduler);
    server::Sink one, two, three;
    Request invalid;
    invalid.sink = &one;
    invalid.token_scores = request({1});
    CHECK_EQ(bridge.submit(std::move(invalid)), (uint64_t)0);  // empty prompt
    CHECK(one.done());
    CHECK_EQ((int)one.finish_reason(), (int)server::Finish::Error);
    std::vector<Request> batch(2);
    batch[0].sink = &two;
    batch[1].sink = &three;
    batch[0].token_scores = request({2});
    batch[1].token_scores = request({3});
    std::vector<uint64_t> ids;
    bridge.submit_all(batch, &ids);
    CHECK_EQ(ids.size(), (size_t)2);
    CHECK_EQ(ids[0], (uint64_t)0);
    CHECK_EQ(ids[1], (uint64_t)0);
    CHECK(two.done() && three.done());
}

RAD_TEST_MAIN()
