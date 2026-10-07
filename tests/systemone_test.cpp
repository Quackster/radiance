#include "rad_test.h"
#include "server/server.h"
#include "server/systemone.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>
#include <limits>
#include <map>

using namespace rad;
using namespace rad::server;
using json = nlohmann::ordered_json;

namespace {

struct DecisionTokenizer : ITokenizer {
    std::map<std::string, int32_t> forms = {
        {"yes", 501}, {"Yes", 502}, {"YES", 503},
        {"no", 504}, {"No", 505}, {"NO", 506},
        {" Å", 600}, {"Å", 601}, {" B", 602}, {" A", 603},
    };
    std::vector<int32_t> encode(const std::string& s, bool special, bool parse_special) const override {
        CHECK(!special);
        if (!parse_special) {
            const auto it = forms.find(s);
            if (it != forms.end()) return {it->second};
        }
        std::vector<int32_t> ids;
        for (unsigned char c : s) ids.push_back((int32_t)c);
        return ids;
    }
    std::string decode(const std::vector<int32_t>&) const override { return ""; }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return 1; }
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                             const std::vector<std::string>&) const override {
        return nullptr;  // decisions must not ask for a generation decoder
    }
};

int parse(const json& body, DecisionRequest* out, ApiError* err, int64_t ctx = 32768,
           int64_t questions = 256) {
    DecisionTokenizer tok;
    return parse_d1_request(body.dump(), tok, "d1", ctx, questions, out, err);
}

json request(const json& q, json state = "State") {
    return json{{"state", std::move(state)}, {"questions", {{"q", q}}}};
}

json choice() {
    return json{{"type", "choice"}, {"instructions", "Who handles this?"},
                {"criteria", {{"technical", "App faults"}, {"billing", "Charges"}}}};
}

json noul() { return json{{"type", "noul"}, {"instructions", "Refund?"}}; }

struct DecisionScheduler : IScheduler {
    bool support = true;
    bool defer = false;
    bool fail = false;
    bool malformed = false;
    uint64_t next = 1;
    std::map<uint64_t, Sink*> pending;
    std::vector<Request> submitted;
    std::vector<uint64_t> cancelled, reaped;
    bool supports_token_scores() const override { return support; }
    uint64_t submit(Request&& r) override {
        const uint64_t id = next++;
        Sink* sink = static_cast<Sink*>(r.sink);
        CHECK(r.token_scores != nullptr);
        CHECK_EQ(r.max_tokens, 0);
        if (r.token_scores) {
            for (int32_t token : r.token_scores->tokens)
                r.token_scores->logits.push_back(token == 602 || token == 502 ? 2.0f : 0.0f);
            if (malformed) r.token_scores->logits.clear();
        }
        sink->set_prompt_stats((int32_t)r.prompt.size(), 0);
        submitted.push_back(std::move(r));
        if (defer) pending[id] = sink;
        else if (fail) sink->fail(RAD_E_STATE, "fixture decision failure");
        else sink->finish(Finish::Stop);
        return id;
    }
    void cancel(uint64_t id) override {
        cancelled.push_back(id);
        auto it = pending.find(id);
        if (it != pending.end()) { it->second->finish(Finish::Cancelled); pending.erase(it); }
    }
    void reap(uint64_t id) override { reaped.push_back(id); }
    SchedMetrics metrics() const override { return {}; }
};

struct DecisionImages : IMultimodal {
    bool support = true;
    bool fail = false;
    int prepares = 0;
    std::vector<std::vector<uint8_t>> received;
    bool accepts(MediaKind kind) const override { return support && kind == MediaKind::Image; }
    int prepare(MediaKind kind, const std::vector<uint8_t>& bytes, const MediaOpts& opts,
                std::shared_ptr<mm::Item>* out, std::string* why) override {
        CHECK(kind == MediaKind::Image);
        CHECK_EQ(opts.image_tokens, 0);
        ++prepares;
        received.push_back(bytes);
        if (fail) { *why = "fixture image decode failure"; return RAD_E_INVAL; }
        auto item = std::make_shared<mm::Item>();
        item->grid_t = 1; item->grid_h = 2; item->grid_w = 4;
        item->n_seg = 1; item->seg_rows = 2; item->rows_h = 1; item->rows_w = 2;
        item->expand = {700, 701, 701, 702};
        item->seg_at = {1};
        item->key_lo = (uint64_t)prepares;
        *out = std::move(item);
        return RAD_OK;
    }
};

struct Fixture {
    DecisionTokenizer tok;
    DecisionScheduler sched;
    ServerOptions opt;
    IMultimodal* mm = nullptr;
    std::unique_ptr<Server> server;
    Fixture() { opt.model_id = "d1"; opt.decision_type = "d1"; opt.max_ctx = 32768; }
    void start() {
        Deps deps;
        deps.tok = &tok;
        deps.sched = &sched;
        deps.mm = mm;
        server.reset(new Server(deps, opt));
    }
    HttpResponse call(const json& body, bool disconnected = false) {
        if (!server) start();
        HttpRequest req;
        req.method = "POST";
        req.path = "/v1/systemone";
        req.body = body.dump();
        req.closed = [disconnected] { return disconnected; };
        return server->router().dispatch(req);
    }
};

}  /* namespace */

TEST(d1_prompt_matches_reference_text_and_order) {
    DecisionRequest r;
    ApiError err;
    REQUIRE_EQ(parse(request(choice(), "Paid twice"), &r, &err), RAD_OK);
    REQUIRE_EQ(r.questions.size(), 1u);
    CHECK_EQ(r.questions[0].prompt,
        "<|startoftext|><|im_start|>user\nPaid twice\n\n\nQUESTION:\n"
        "Who handles this?\n\nOptions:\nA App faults\nB Charges\n\n"
        "Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n");
    CHECK_EQ(r.questions[0].labels[0], "technical");
    CHECK_EQ(r.questions[0].labels[1], "billing");
    CHECK_EQ(r.input_tokens, (int64_t)r.questions[0].tokens.size());
}

TEST(d1_json_and_null_state_reference_prompts) {
    DecisionRequest r;
    ApiError err;
    const json state = {{"z", "É"}, {"a", json::array({true, nullptr})}};
    REQUIRE_EQ(parse(request(noul(), state), &r, &err), RAD_OK);
    CHECK(r.questions[0].prompt.find("{\n  \"z\": \"É\",\n  \"a\": [\n    true,\n    null\n  ]\n}") != std::string::npos);
    REQUIRE_EQ(parse(request(noul(), nullptr), &r, &err), RAD_OK);
    CHECK_EQ(r.questions[0].prompt,
        "<|startoftext|><|im_start|>user\nRefund?\n\nReply with yes or no only."
        "<|im_end|>\n<|im_start|>assistant\n");
}

TEST(d1_multiple_questions_preserve_the_reference_prefix_tokenization_boundary) {
    struct BoundaryTokenizer : DecisionTokenizer {
        std::vector<int32_t> encode(const std::string& text, bool special, bool parse_special) const override {
            if (!parse_special) return DecisionTokenizer::encode(text, special, false);
            // These are d1's actual whole-token IDs; a longer colon/newline token wins.
            const std::pair<std::string, int32_t> forms[] = {
                {":\n\n\n", 89289}, {":\n\n", 63990}, {":\n", 833}, {"\n\n", 8}
            };
            std::vector<int32_t> ids;
            for (size_t at = 0; at < text.size();) {
                bool found = false;
                for (const auto& form : forms)
                    if (text.compare(at, form.first.size(), form.first) == 0) {
                        ids.push_back(form.second);
                        at += form.first.size();
                        found = true;
                        break;
                    }
                if (!found) ids.push_back((unsigned char)text[at++]);
            }
            return ids;
        }
    } tok;
    json q = noul();
    q["instructions"] = "";
    json body = request(q);
    DecisionRequest r;
    ApiError err;
    REQUIRE_EQ(parse_d1_request(body.dump(), tok, "d1", 32768, 10, &r, &err), RAD_OK);
    CHECK(std::find(r.questions[0].tokens.begin(), r.questions[0].tokens.end(), 89289) != r.questions[0].tokens.end());
    body["questions"]["other"] = q;
    REQUIRE_EQ(parse_d1_request(body.dump(), tok, "d1", 32768, 10, &r, &err), RAD_OK);
    const auto& ids = r.questions[0].tokens;
    CHECK(std::find(ids.begin(), ids.end(), 89289) == ids.end());
    auto colon = std::find(ids.begin(), ids.end(), 833);
    REQUIRE(colon != ids.end() && colon + 1 != ids.end());
    CHECK_EQ(*(colon + 1), 8);
    CHECK_EQ(r.questions[0].shared_prefix_tokens, (int32_t)(colon - ids.begin() + 1));
    CHECK_EQ(r.questions[1].shared_prefix_tokens, r.questions[0].shared_prefix_tokens);
    const int64_t rowsize = (int64_t)ids.size();
    CHECK_EQ(r.input_tokens, rowsize * 2);
    REQUIRE_EQ(parse_d1_request(body.dump(), tok, "d1", rowsize, 10, &r, &err), RAD_OK);
    CHECK(parse_d1_request(body.dump(), tok, "d1", rowsize - 1, 10, &r, &err) < 0);
    CHECK_EQ(err.status, 400);
}

TEST(d1_noul_criteria_and_max_pooled_forms) {
    DecisionRequest r;
    ApiError err;
    json q = noul();
    q["criteria"] = {{"true", "Money back"}};
    REQUIRE_EQ(parse(request(q), &r, &err), RAD_OK);
    CHECK(r.questions[0].prompt.find("Refund?\nYes: Money back\nNo: None\n\n") != std::string::npos);
    const std::vector<float> logits = {-100.0f, 2.0f, 0.0f, 1.0f, -10.0f, -20.0f};
    json answer;
    REQUIRE_EQ(d1_answer(r.questions[0], logits, &answer, &err), RAD_OK);
    CHECK_NEAR(answer["noul"].get<double>(), 1.0 / (1.0 + std::exp(-1.0)), 1e-7);
    CHECK(!answer.contains("confidence"));
}

TEST(d1_choice_aliases_include_space_form_and_confidence_is_max_probability) {
    DecisionRequest r;
    ApiError err;
    REQUIRE_EQ(parse(request(choice()), &r, &err), RAD_OK);
    REQUIRE_EQ(r.questions[0].candidates.size(), 4u);
    // Bare A wins among bare codes, but leading-space B is the strongest form overall.
    json answer;
    REQUIRE_EQ(d1_answer(r.questions[0], {1.0f, 0.0f, -10.0f, 2.0f}, &answer, &err), RAD_OK);
    CHECK_EQ(answer["choice"].get<std::string>(), "billing");
    CHECK_NEAR(answer["confidence"].get<double>(), 1.0 / (1.0 + std::exp(-1.0)), 1e-7);
    CHECK_EQ(answer["probabilities"].begin().key(), "technical");
    REQUIRE_EQ(d1_answer(r.questions[0], {0, 0, 0, 0}, &answer, &err), RAD_OK);
    CHECK_NEAR(answer["confidence"].get<double>(), 0.5, 1e-8);
    CHECK_EQ(answer["choice"].get<std::string>(), "technical");
}

TEST(d1_native_unicode_letters_and_alias_fallbacks) {
    DecisionRequest r;
    ApiError err;
    json q = choice();
    q["criteria"] = {{" Å ", "First"}, {"B", "Second"}};
    REQUIRE_EQ(parse(request(q), &r, &err), RAD_OK);
    CHECK(r.questions[0].prompt.find("Options:\nÅ First\nB Second") != std::string::npos);
    CHECK_EQ(r.questions[0].labels[0], " Å ");
    q["criteria"] = json::object();
    for (int i = 0; i < 27; ++i) q["criteria"]["opt" + std::to_string(i)] = "Option";
    REQUIRE_EQ(parse(request(q), &r, &err), RAD_OK);
    // This tokenizer splits numeric pairs: the reference pool falls through A..Z then a.
    CHECK(r.questions[0].prompt.find("\na Option\n\nReply") != std::string::npos);
}

TEST(d1_score_returns_expected_level_and_legend) {
    DecisionRequest r;
    ApiError err;
    json q = {{"type", "score"}, {"instructions", "Urgency?"},
              {"criteria", json::array({"Later", "Today", "Now"})}};
    REQUIRE_EQ(parse(request(q), &r, &err), RAD_OK);
    CHECK(r.questions[0].prompt.find("Urgency?\n\n0 Later\n1 Today\n2 Now\n\nReply with a single digit 0-2 only.") != std::string::npos);
    json answer;
    REQUIRE_EQ(d1_answer(r.questions[0], {0, 0, (float)std::log(2.0)}, &answer, &err), RAD_OK);
    CHECK_NEAR(answer["score"].get<double>(), 1.25, 1e-7);
    CHECK_NEAR(answer["confidence"].get<double>(), 0.5, 1e-7);
    CHECK_EQ(answer["legend"]["2"].get<std::string>(), "Now");
}

TEST(d1_parser_rejects_invalid_unsupported_or_oversized_requests) {
    DecisionRequest r;
    ApiError err;
    for (const auto& bad : {
        json::array(), json{{"state", "x"}}, json{{"questions", {{"q", noul()}}}},
        request(json{{"type", "other"}, {"instructions", "x"}}),
        request(json{{"type", "choice"}, {"instructions", "x"}, {"criteria", json::object()}}),
        request(json{{"type", "score"}, {"instructions", "x"}, {"criteria", json::array({"one"})}}),
        request(json{{"type", "noul"}, {"instructions", 3}})}) {
        CHECK(parse(bad, &r, &err) < 0);
        CHECK_EQ(err.status, 400);
    }
    json body = request(noul());
    body["images"] = json::array({"data:image/png;base64,AA=="});
    CHECK(parse(body, &r, &err) < 0);
    CHECK_EQ(err.status, 501);
    body.erase("images"); body["stream"] = true;
    CHECK(parse(body, &r, &err) < 0);
    CHECK_EQ(err.param, "stream");
    body.erase("stream"); body["temperature"] = 0;
    CHECK(parse(body, &r, &err) < 0);
    CHECK_EQ(err.param, "temperature");
    CHECK(parse(request(noul()), &r, &err, 5) < 0);
    CHECK_EQ(err.status, 400);
    body = request(noul()); body["questions"]["other"] = noul();
    CHECK(parse(body, &r, &err, 32768, 1) < 0);
    CHECK_EQ(err.status, 429);
}

TEST(d1_invalid_logits_fail_instead_of_returning_a_distribution) {
    DecisionRequest r;
    ApiError err;
    REQUIRE_EQ(parse(request(choice()), &r, &err), RAD_OK);
    json answer;
    CHECK(d1_answer(r.questions[0], {}, &answer, &err) < 0);
    CHECK_EQ(err.status, 500);
    CHECK(d1_answer(r.questions[0], {0, 0, 0, std::numeric_limits<float>::quiet_NaN()}, &answer, &err) < 0);
    CHECK(d1_answer(r.questions[0], {0, 0, 0, std::numeric_limits<float>::infinity()}, &answer, &err) < 0);
}

TEST(systemone_endpoint_submits_scores_without_generating_and_reaps_every_question) {
    Fixture f;
    f.opt.max_batch_tokens = 1;
    json body = request(choice());
    body["questions"]["refund"] = noul();
    const auto response = f.call(body);
    REQUIRE_EQ(response.status, 200);
    CHECK(!response.streaming());
    const json result = json::parse(response.body);
    CHECK_EQ(result["answers"]["q"]["choice"].get<std::string>(), "billing");
    CHECK(result["answers"]["refund"]["noul"].get<double>() > 0.8);
    CHECK_EQ(result["usage"]["output_tokens"].get<int>(), 0);
    REQUIRE_EQ(f.sched.submitted.size(), 2u);
    REQUIRE(f.sched.submitted[0].shared_prefix != nullptr);
    CHECK(f.sched.submitted[0].shared_prefix == f.sched.submitted[1].shared_prefix);
    const size_t trunk = (size_t)f.sched.submitted[0].shared_prefix->n_tokens;
    REQUIRE(trunk > 0 && trunk < f.sched.submitted[0].prompt.size());
    CHECK(std::equal(f.sched.submitted[0].prompt.begin(), f.sched.submitted[0].prompt.begin() + trunk,
                     f.sched.submitted[1].prompt.begin()));
    CHECK_EQ(result["usage"]["input_tokens"].get<int64_t>(),
             (int64_t)(f.sched.submitted[0].prompt.size() + f.sched.submitted[1].prompt.size()));
    CHECK_EQ(f.sched.reaped.size(), 2u);
    CHECK_EQ(f.server->in_flight(), 0);
}

TEST(systemone_capability_and_model_gate_precede_submission) {
    Fixture f;
    f.opt.decision_type.clear();
    CHECK_EQ(f.call(request(noul())).status, 501);
    CHECK(f.sched.submitted.empty());
    Fixture unsupported;
    unsupported.sched.support = false;
    CHECK_EQ(unsupported.call(request(noul())).status, 501);
    CHECK(unsupported.sched.submitted.empty());
    Fixture wrong_model;
    json body = request(noul()); body["model"] = "other";
    CHECK_EQ(wrong_model.call(body).status, 400);
    CHECK(wrong_model.sched.submitted.empty());
}

TEST(systemone_disconnect_cancels_and_releases_admission) {
    Fixture f;
    f.sched.defer = true;
    CHECK_EQ(f.call(request(noul()), true).status, 499);
    CHECK_EQ(f.sched.cancelled.size(), 1u);
    CHECK_EQ(f.sched.reaped.size(), 1u);
    CHECK_EQ(f.server->in_flight(), 0);
}

TEST(systemone_failure_and_invalid_scores_release_every_request) {
    for (bool malformed : {false, true}) {
        Fixture f;
        f.sched.fail = !malformed;
        f.sched.malformed = malformed;
        CHECK_EQ(f.call(request(noul())).status, 500);
        CHECK_EQ(f.sched.reaped.size(), 1u);
        CHECK_EQ(f.server->in_flight(), 0);
    }
}

TEST(systemone_admission_bounds_questions_before_submitting) {
    Fixture f;
    f.opt.queue_depth = 1;
    json body = request(noul()); body["questions"]["second"] = noul();
    const auto response = f.call(body);
    CHECK_EQ(response.status, 429);
    CHECK(f.sched.submitted.empty());
    CHECK_EQ(f.server->in_flight(), 0);
}

TEST(d1_images_precede_a_null_or_json_state_and_keep_prepared_token_positions) {
    DecisionTokenizer tok;
    for (const json& state : {json(), json{{"message", "Look at the picture"}}}) {
        DecisionImages images;
        json body = request(noul(), state);
        body["images"] = json::array({"data:image/png;base64,AQID"});
        DecisionRequest r;
        ApiError err;
        REQUIRE_EQ(parse_d1_request(body.dump(), tok, "d1", 32768, 10, &r, &err, &images), RAD_OK);
        REQUIRE(r.media != nullptr);
        REQUIRE_EQ(r.media->items.size(), 1u);
        REQUIRE_EQ(r.questions.size(), 1u);
        CHECK_EQ(images.prepares, 1);
        CHECK(images.received[0] == std::vector<uint8_t>({1, 2, 3}));
        const std::string head = "<|startoftext|><|im_start|>user\n";
        const size_t first = tok.encode(head, false, true).size();
        CHECK_EQ(r.media->items[0].tok, (int32_t)first);
        const auto& tokens = r.questions[0].tokens;
        CHECK(std::vector<int32_t>(tokens.begin() + first, tokens.begin() + first + 4) ==
              std::vector<int32_t>({700, 701, 701, 702}));
        CHECK_EQ(r.questions[0].prompt.substr(0, head.size() + 7), head + "<image>");
        REQUIRE_EQ(r.media->rope.runs.size(), 1u);
        CHECK_EQ(r.media->rope.runs[0].tok, (int32_t)first + 1);
        CHECK_EQ(r.media->rope.runs[0].n, 2);
        if (state.is_null())
            CHECK(r.questions[0].prompt.find("<image>Refund?") != std::string::npos);
        else
            CHECK(r.questions[0].prompt.find("<image>{\n  \"message\": \"Look at the picture\"\n}\n\n\nQUESTION:\nRefund?") != std::string::npos);
        CHECK_EQ(r.input_tokens, (int64_t)tokens.size());
    }
}

TEST(systemone_questions_share_image_items_and_encoder_lifecycle) {
    DecisionImages images;
    Fixture f;
    f.opt.max_batch_tokens = 1;
    f.mm = &images;
    f.opt.allow_image = true;
    json body = request(choice());
    body["questions"]["refund"] = noul();
    body["images"] = json::array({"data:image/png;base64,AQID", "data:image/jpeg;base64,BAUG"});
    const auto response = f.call(body);
    REQUIRE_EQ(response.status, 200);
    REQUIRE_EQ(f.sched.submitted.size(), 2u);
    REQUIRE(f.sched.submitted[0].shared_prefix != nullptr);
    CHECK(f.sched.submitted[0].shared_prefix == f.sched.submitted[1].shared_prefix);
    const size_t trunk = (size_t)f.sched.submitted[0].shared_prefix->n_tokens;
    REQUIRE(trunk > 0 && trunk < f.sched.submitted[0].prompt.size());
    CHECK(std::equal(f.sched.submitted[0].prompt.begin(), f.sched.submitted[0].prompt.begin() + trunk,
                     f.sched.submitted[1].prompt.begin()));
    CHECK_EQ(images.prepares, 2);  // once per image, not once per question
    const auto& media = f.sched.submitted[0].media;
    REQUIRE(media != nullptr);
    CHECK(media == f.sched.submitted[1].media);
    REQUIRE_EQ(media->items.size(), 2u);
    CHECK(media->items[0].item != media->items[1].item);
    CHECK_EQ(media->items[1].tok, media->items[0].tok + 4);
    CHECK(!media->items[0].item->encoded);
    CHECK(!media->items[0].item->queued);  // only the real scheduler may mutate encoding state
    CHECK(images.received[1] == std::vector<uint8_t>({4, 5, 6}));
    const json answer = json::parse(response.body);
    CHECK_EQ(answer["usage"]["output_tokens"].get<int>(), 0);
    CHECK_EQ(answer["usage"]["input_tokens"].get<int64_t>(),
             (int64_t)(f.sched.submitted[0].prompt.size() + f.sched.submitted[1].prompt.size()));
    CHECK_EQ(f.sched.reaped.size(), 2u);
    CHECK_EQ(f.server->in_flight(), 0);
}

TEST(systemone_images_require_enabled_capability_and_valid_data_urls) {
    const std::string image = "data:image/png;base64,AQID";
    json body = request(noul(), nullptr);
    body["images"] = json::array({image});
    for (int mode = 0; mode < 3; ++mode) {
        DecisionImages images;
        Fixture f;
        if (mode != 0) f.mm = &images;
        f.opt.allow_image = mode != 1;
        images.support = mode != 2;
        CHECK_EQ(f.call(body).status, 501);
        CHECK_EQ(images.prepares, 0);
        CHECK(f.sched.submitted.empty());
    }
    for (const json& source : {json(17), json{{"url", image}},
                              json("https://example.com/photo.png"), json("data:image/png,raw"),
                              json("data:image/png;base64,***")}) {
        DecisionImages images;
        Fixture f;
        f.mm = &images; f.opt.allow_image = true;
        body["images"] = json::array({source});
        const auto response = f.call(body);
        CHECK_EQ(response.status, 400);
        CHECK_EQ(json::parse(response.body)["error"]["param"].get<std::string>(), "images[0]");
        CHECK_EQ(images.prepares, 0);
        CHECK(f.sched.submitted.empty());
    }
}

TEST(systemone_image_decode_or_context_failure_does_not_submit) {
    for (bool decode_failure : {false, true}) {
        DecisionImages images;
        images.fail = decode_failure;
        Fixture f;
        f.mm = &images; f.opt.allow_image = true;
        if (!decode_failure) f.opt.max_ctx = 34;
        json body = request(noul());
        body["images"] = json::array({"data:image/png;base64,AQID"});
        CHECK_EQ(f.call(body).status, 400);
        CHECK_EQ(images.prepares, 1);
        CHECK(f.sched.submitted.empty());
        CHECK_EQ(f.server->in_flight(), 0);
    }
}

TEST(systemone_stops_preparing_images_at_the_first_expanded_context_overflow) {
    DecisionTokenizer tok;
    const int64_t header = (int64_t)tok.encode(
        "<|startoftext|><|im_start|>user\n", false, true).size();
    for (int fit = 0; fit < 3; ++fit) {
        DecisionImages images;
        Fixture f;
        f.mm = &images; f.opt.allow_image = true;
        // Each image has two feature tokens and two boundary markers. The next
        // full expansion exceeds this budget by one; later images must not decode.
        f.opt.max_ctx = header + 4 * fit + 3;
        json body = request(noul(), nullptr);
        body["images"] = json::array();
        for (int i = 0; i < 5; ++i)
            body["images"].push_back("data:image/png;base64,AQID");
        const auto response = f.call(body);
        REQUIRE_EQ(response.status, 400);
        CHECK_EQ(json::parse(response.body)["error"]["param"].get<std::string>(),
                 "images[" + std::to_string(fit) + "]");
        CHECK_EQ(images.prepares, fit + 1);
        CHECK(f.sched.submitted.empty());
        CHECK_EQ(f.server->in_flight(), 0);
    }
}

TEST(systemone_image_disconnect_releases_every_question) {
    DecisionImages images;
    Fixture f;
    f.mm = &images; f.opt.allow_image = true; f.sched.defer = true;
    json body = request(noul(), nullptr);
    body["questions"]["other"] = choice();
    body["images"] = json::array({"data:image/png;base64,AQID"});
    CHECK_EQ(f.call(body, true).status, 499);
    CHECK_EQ(images.prepares, 1);
    CHECK_EQ(f.sched.cancelled.size(), 2u);
    CHECK_EQ(f.sched.reaped.size(), 2u);
    CHECK_EQ(f.server->in_flight(), 0);
}

RAD_TEST_MAIN()

TEST(systemone_keeps_short_shared_trunks_in_the_existing_batch) {
    for (int64_t batch : {0, 256, 4096}) {
        Fixture f;
        f.opt.max_batch_tokens = batch;
        json body = request(choice());
        body["questions"]["refund"] = noul();
        REQUIRE_EQ(f.call(body).status, 200);
        REQUIRE_EQ(f.sched.submitted.size(), 2u);
        CHECK(!f.sched.submitted[0].shared_prefix);
        CHECK(!f.sched.submitted[1].shared_prefix);
    }
}

TEST(systemone_large_batches_still_share_a_trunk_when_the_branches_span_batches) {
    for (int64_t batch : {2048, 4096}) {
        Fixture f;
        f.opt.max_batch_tokens = batch;
        json body = request(choice(), std::string(1200, 'x'));
        body["questions"]["refund"] = noul();
        REQUIRE_EQ(f.call(body).status, 200);
        REQUIRE_EQ(f.sched.submitted.size(), 2u);
        CHECK_EQ(f.sched.submitted[0].shared_prefix != nullptr, batch == 2048);
        CHECK(f.sched.submitted[0].shared_prefix == f.sched.submitted[1].shared_prefix);
    }
}
