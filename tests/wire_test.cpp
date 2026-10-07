/* wire_test.cpp -- the pieces of the HTTP and OpenAI surface that are not a route.
 *
 * server_test drives the server end to end and is where a handler's behaviour belongs. What it
 * cannot reach is the half-dozen small functions underneath: a target splitter, a percent
 * decoder, a base64 pair, the UTF-8 back-off that keeps a streamed delta from cutting a code
 * point in half, the bucket ladders every Prometheus quantile is computed against, and the body
 * builders. Each of them is reachable from exactly one route, so a fault in one shows up as one
 * endpoint being subtly wrong -- a query parameter that silently does not arrive, a stream a
 * client cannot decode, a histogram whose quantiles moved across a version boundary.
 *
 * WHAT MAKES THESE WORTH A TEST RATHER THAN A READING. Every one of them has a case that is not
 * the obvious one: base64 has two padding lengths, the decoder has to reject a character rather
 * than skip it, `+` means two different things in the two halves of a target, and `utf8_trunc`
 * exists precisely because the easy implementation is wrong.
 */
#include "rad_test.h"

#include <nlohmann/json.hpp>

#include "server/http.h"
#include "server/json_partial.h"
#include "server/metrics.h"
#include "server/oai.h"
#include "server/sink.h"

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::server;

/* ================================================================== the target */

TEST(a_target_splits_into_a_path_and_decoded_query_parameters) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/v1/models/foo?x=1&y=two", &path, &q);
    CHECK_EQ(path, std::string("/v1/models/foo"));
    CHECK_EQ(q.size(), 2u);
    CHECK_EQ(q["x"], std::string("1"));
    CHECK_EQ(q["y"], std::string("two"));

    /* No query at all leaves the map untouched rather than inventing an empty entry. */
    std::map<std::string, std::string> none;
    parse_target("/health", &path, &none);
    CHECK_EQ(path, std::string("/health"));
    CHECK_EQ(none.size(), 0u);

    /* A key with no value is present and empty -- "?stream" is a flag, and a caller asking
     * whether the key is there must be able to tell it from a key that was never sent. */
    std::map<std::string, std::string> flag;
    parse_target("/x?stream", &path, &flag);
    CHECK_EQ(flag.size(), 1u);
    CHECK(flag.count("stream") == 1);
    CHECK_EQ(flag["stream"], std::string(""));

    /* An empty query after the '?' contributes nothing. */
    std::map<std::string, std::string> empty;
    parse_target("/x?", &path, &empty);
    CHECK_EQ(path, std::string("/x"));
    CHECK_EQ(empty.size(), 0u);

    /* Both outputs are optional: the router asks for the path alone on the hot path. */
    std::string only;
    parse_target("/x?a=b", &only, nullptr);
    CHECK_EQ(only, std::string("/x"));
    std::map<std::string, std::string> qonly;
    parse_target("/x?a=b", nullptr, &qonly);
    CHECK_EQ(qonly["a"], std::string("b"));
}

TEST(query_values_are_percent_decoded_and_survive_an_embedded_equals) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/x?name=a%20b&eq=1%3D2&plus=a+b", &path, &q);
    CHECK_EQ(q["name"], std::string("a b"));
    /* Split on the FIRST '=' only, or a base64 value -- which ends in '=' padding -- loses its
     * tail every time. */
    CHECK_EQ(q["eq"], std::string("1=2"));
    CHECK_EQ(q["plus"], std::string("a b"));

    std::map<std::string, std::string> raw;
    parse_target("/x?v=a=b=c", &path, &raw);
    CHECK_EQ(raw["v"], std::string("a=b=c"));
}

/* `+` IS A SPACE IN A QUERY STRING AND A PLUS SIGN IN A PATH. The form encoding that gives it the
 * second meaning applies to form data, which is what a query string is and a path is not --
 * so /v1/models/gpt+4 must not become a request for "gpt 4". */
TEST(a_plus_in_the_path_stays_a_plus) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/v1/models/gpt+4?q=a+b", &path, &q);
    CHECK_EQ(path, std::string("/v1/models/gpt+4"));
    CHECK_EQ(q["q"], std::string("a b"));

    CHECK_EQ(url_decode("a+b"), std::string("a b"));
    CHECK_EQ(url_decode("a+b", false), std::string("a+b"));
    /* Percent-encoding works the same in both. */
    CHECK_EQ(url_decode("a%2Bb", false), std::string("a+b"));
}

TEST(url_decode_leaves_a_truncated_or_invalid_escape_alone) {
    CHECK_EQ(url_decode("plain"), std::string("plain"));
    CHECK_EQ(url_decode(""), std::string(""));
    CHECK_EQ(url_decode("%41%42"), std::string("AB"));
    CHECK_EQ(url_decode("%4a"), std::string("J"));        /* lower-case hex */
    CHECK_EQ(url_decode("%4A"), std::string("J"));
    /* A DECODER THAT CONSUMED A BAD ESCAPE WOULD EAT THE BYTES AFTER IT. Leaving it verbatim
     * makes the mistake visible in the value rather than shortening the string silently. */
    CHECK_EQ(url_decode("%zz"), std::string("%zz"));
    CHECK_EQ(url_decode("%4"), std::string("%4"));
    CHECK_EQ(url_decode("%"), std::string("%"));
    CHECK_EQ(url_decode("100%"), std::string("100%"));
    /* A percent-encoded percent. */
    CHECK_EQ(url_decode("%2541"), std::string("%41"));
}

/* ================================================================== base64 */

TEST(base64_round_trips_at_every_padding_length) {
    /* The three residues are the whole of the encoder's branching, and they are what the padding
     * length encodes: a decoder that got the two-byte tail wrong would return a trailing zero. */
    const char* cases[] = { "", "f", "fo", "foo", "foob", "fooba", "foobar" };
    const char* want[]  = { "", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy" };
    for (int i = 0; i < 7; ++i) {
        const std::string in = cases[i];
        const std::string enc = base64_encode(in.data(), in.size());
        CHECK_EQ(enc, std::string(want[i]));
        std::vector<uint8_t> back;
        CHECK(base64_decode(enc, &back));
        CHECK_EQ(back.size(), in.size());
        CHECK_EQ(std::string(back.begin(), back.end()), in);
    }
}

TEST(base64_carries_every_byte_value_including_zero) {
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i) all[(size_t)i] = (uint8_t)i;
    const std::string enc = base64_encode(all.data(), all.size());
    /* 256 bytes is not a multiple of three, so this exercises a padded tail on real data. */
    CHECK_EQ(enc.size(), (size_t)344);
    std::vector<uint8_t> back;
    CHECK(base64_decode(enc, &back));
    CHECK_EQ(back.size(), (size_t)256);
    for (int i = 0; i < 256; ++i) CHECK_EQ((int)back[(size_t)i], i);

    /* The alphabet is the standard one and not the URL-safe variant: a decoder on the other end
     * chooses by which characters it sees, and '+' and '/' are what it must see here. */
    bool plus = false, slash = false;
    for (char c : enc) { if (c == '+') plus = true; if (c == '/') slash = true; }
    CHECK(plus);
    CHECK(slash);
}

/* AN EMBEDDING ARRIVES AS BASE64 FROM A CLIENT, so the decoder is parsing untrusted input: a
 * character outside the alphabet has to be an error and not a byte skipped, or a corrupted
 * payload decodes to a shorter vector that the shape check then accepts or rejects for the
 * wrong reason. */
TEST(base64_decode_rejects_a_character_outside_the_alphabet) {
    std::vector<uint8_t> out;
    CHECK(!base64_decode("Zm9v!", &out));
    CHECK(!base64_decode("****", &out));
    CHECK(!base64_decode("Zm-9v", &out));        /* the URL-safe alphabet is not accepted */
    CHECK(!base64_decode("Zm_9v", &out));

    /* Whitespace and padding are skipped, because a wrapped MIME payload is still the value. */
    CHECK(base64_decode("Zm9v\nYmFy", &out));
    CHECK_EQ(std::string(out.begin(), out.end()), std::string("foobar"));
    CHECK(base64_decode("Zm9v YmFy\t\r\n", &out));
    CHECK_EQ(std::string(out.begin(), out.end()), std::string("foobar"));
    CHECK(base64_decode("", &out));
    CHECK_EQ(out.size(), 0u);

    /* The output is cleared first: a failed decode after a successful one must not leave the
     * previous payload behind for the caller to read as this one's. */
    CHECK(base64_decode("Zm9v", &out));
    CHECK_EQ(out.size(), 3u);
    CHECK(!base64_decode("!", &out));
    CHECK_EQ(out.size(), 0u);
}

/* Every length mod 3 round-trips; padding and whitespace are skipped wherever they stand, which
 * splits the four-digit blocks the decoder takes at once; anything else is refused. */
TEST(base64_round_trips_and_refuses_stray_bytes) {
    std::vector<uint8_t> bytes(1000);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t)((i * 2654435761u) >> 13);
    for (size_t n : { 0, 1, 2, 3, 4, 5, 999, 1000 }) {
        std::vector<uint8_t> got;
        CHECK(base64_decode(base64_encode(bytes.data(), n), &got));
        CHECK(got == std::vector<uint8_t>(bytes.begin(), bytes.begin() + (long)n));
    }
    std::string e = base64_encode(bytes.data(), 100), spaced;
    for (size_t i = 0; i < e.size(); ++i) { spaced += e[i]; if (i % 7 == 3) spaced += (i % 2 ? "\n" : " "); }
    std::vector<uint8_t> got;
    CHECK(base64_decode(spaced, &got));
    CHECK(got == std::vector<uint8_t>(bytes.begin(), bytes.begin() + 100));
    CHECK(!base64_decode("QUJD*REVG", &got));
    CHECK(!base64_decode("QUJDREVG\x80", &got));
}

/* ================================================================== ids and errors */

TEST(a_random_id_carries_its_prefix_and_does_not_repeat) {
    const std::string a = random_id("chatcmpl-");
    const std::string b = random_id("chatcmpl-");
    CHECK(a.rfind("chatcmpl-", 0) == 0);
    CHECK(b.rfind("chatcmpl-", 0) == 0);
    CHECK(a.size() > std::string("chatcmpl-").size());
    CHECK(a != b);                 /* ids collide in a client's logs, not in the server */
    CHECK_EQ(a.size(), b.size());  /* fixed width, so it sorts and pads predictably */

    const std::string c = random_id("cmpl-");
    CHECK(c.rfind("cmpl-", 0) == 0);
    /* The body is drawn from a printable set -- an id lands in a URL and in a log line. */
    for (size_t i = std::string("cmpl-").size(); i < c.size(); ++i)
        CHECK(c[i] > 0x20 && c[i] < 0x7f);
}

TEST(the_error_envelope_is_the_shape_a_client_parses) {
    const ApiError bad = err_bad("temperature", "must be in [0, 2]");
    CHECK_EQ(bad.status, 400);
    CHECK_EQ(bad.param, std::string("temperature"));
    const std::string d = bad.dump();
    /* OpenAI's envelope is {"error":{"message":...}} and a client reads message and param by
     * name; anything flatter is an error body a library reports as "unknown". */
    CHECK(d.find("\"error\"") != std::string::npos);
    CHECK(d.find("\"message\"") != std::string::npos);
    CHECK(d.find("temperature") != std::string::npos);
    CHECK(d.find("must be in") != std::string::npos);
    CHECK(json_complete(d));

    const ApiError uns = err_unsupported("logprobs", "not implemented");
    CHECK_EQ(uns.status, 400);
    CHECK_EQ(uns.param, std::string("logprobs"));
    CHECK(json_complete(uns.dump()));

    /* A CAPABILITY THIS BUILD DOES NOT HAVE IS 501 AND NOT 400, because 400 tells the caller to
     * fix their request and there is nothing in it to fix. */
    const ApiError ni = err_not_implemented("embeddings");
    CHECK_EQ(ni.status, 501);
    CHECK(ni.dump().find("embeddings") != std::string::npos);

    /* And a full queue is 429, which is what carries a Retry-After: the request is fine and the
     * server is momentarily not, so backing off and resending is the correct client behaviour.
     * 503 would say the service is down, which a client is entitled to treat as fatal. */
    const ApiError ov = err_overloaded("admission queue is full");
    CHECK_EQ(ov.status, 429);
    CHECK(json_complete(ov.dump()));

    /* An error about no particular field carries no param rather than an empty one. */
    CHECK(err_overloaded("busy").param.empty());
}

/* ================================================================== server-sent events */

TEST(an_sse_event_is_one_data_line_and_a_blank_line) {
    const std::string e = sse_event("{\"a\":1}");
    CHECK_EQ(e, std::string("data: {\"a\":1}\n\n"));
    /* THE BLANK LINE IS THE FRAME DELIMITER. Without it a client buffers the event forever and
     * the stream looks hung rather than broken. */
    CHECK(e.size() >= 2 && e.compare(e.size() - 2, 2, "\n\n") == 0);

    CHECK_EQ(std::string(SSE_DONE), std::string("data: [DONE]\n\n"));

    const std::string err = sse_error(err_bad("n", "too large"));
    CHECK(err.rfind("data: ", 0) == 0);
    CHECK(err.find("\"error\"") != std::string::npos);
    CHECK(err.find("too large") != std::string::npos);
    CHECK(err.compare(err.size() - 2, 2, "\n\n") == 0);
    /* The payload between the prefix and the delimiter is one complete JSON value, which is what
     * a client's per-event parse requires. */
    CHECK(json_complete(err.substr(6, err.size() - 8)));
}

/* ================================================================== finish reasons */

TEST(the_finish_reason_matches_vllms_spelling) {
    CHECK_EQ(std::string(finish_reason_oai(Finish::Stop)), std::string("stop"));
    CHECK_EQ(std::string(finish_reason_oai(Finish::Length)), std::string("length"));
    CHECK_EQ(std::string(finish_reason_oai(Finish::ToolCalls)), std::string("tool_calls"));
    /* CANCELLED HAS NO OPENAI SPELLING and vLLM reports "abort". Dashboards key off that string,
     * so inventing a better one here would make this server's traffic invisible in them. */
    CHECK_EQ(std::string(finish_reason_oai(Finish::Cancelled)), std::string("abort"));
    /* Every reason resolves to something non-empty: a missing finish_reason is a null in the
     * choice object, which clients treat as "still streaming". */
    for (Finish f : { Finish::Stop, Finish::Length, Finish::ToolCalls, Finish::Cancelled,
                      Finish::Error })
        CHECK(finish_reason_oai(f) && *finish_reason_oai(f));
}

/* ================================================================== the UTF-8 back-off */

TEST(utf8_trunc_backs_out_of_a_split_code_point_and_nothing_else) {
    /* Complete input is returned whole -- the common case, and the one a per-delta call takes. */
    CHECK_EQ(utf8_trunc("hello"), (size_t)5);
    CHECK_EQ(utf8_trunc(""), (size_t)0);
    CHECK_EQ(utf8_trunc("\xC3\xA9"), (size_t)2);              /* e-acute, complete */
    CHECK_EQ(utf8_trunc("\xE2\x82\xAC"), (size_t)3);          /* euro sign, complete */
    CHECK_EQ(utf8_trunc("\xF0\x9F\x98\x80"), (size_t)4);      /* an emoji, complete */

    /* A TRAILING PARTIAL SEQUENCE IS CUT. A delta ending mid-code-point is not decodable, and a
     * client that concatenates deltas before decoding sees a replacement character that never
     * goes away. */
    CHECK_EQ(utf8_trunc("ab\xC3"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xE2\x82"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xE2"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xF0\x9F\x98"), (size_t)2);
    CHECK_EQ(utf8_trunc("\xF0"), (size_t)0);

    /* A COMPLETE SEQUENCE FOLLOWED BY A PARTIAL ONE keeps the complete one. */
    CHECK_EQ(utf8_trunc("\xE2\x82\xAC\xC3"), (size_t)3);
    /* AND A PARTIAL SEQUENCE IN THE MIDDLE IS NOT THIS FUNCTION'S BUSINESS. It backs out of a
     * truncation rather than validating, so a lone lead byte followed by ASCII is left where it
     * is: only the tail is reconsidered, and the tail here is complete. */
    CHECK_EQ(utf8_trunc("\xC3 ok"), (size_t)4);
    CHECK_EQ(utf8_trunc("\xC3 ok\xE2\x82"), (size_t)4);   /* ... and the real tail still cuts */
}

/* ================================================================== histogram ladders */

/* vLLM'S LAYOUTS, VERBATIM. A dashboard computes quantiles by interpolating between bucket
 * bounds, so changing one of these silently moves every historical quantile across a version
 * boundary -- the graph does not break, it just stops meaning what it meant. */
TEST(the_bucket_ladders_are_sorted_positive_and_the_ones_vllm_publishes) {
    struct { const char* name; std::vector<double> b; size_t n; double first, last; } cases[] = {
        { "ttft",    buckets_ttft(),    22, 0.001, 2560.0 },
        { "tpot",    buckets_tpot(),    19, 0.01,  80.0 },
        { "latency", buckets_latency(), 21, 0.3,   7680.0 },
    };
    for (auto& c : cases) {
        CHECK_EQ(c.b.size(), c.n);
        CHECK_NEAR(c.b.front(), c.first, 1e-12);
        CHECK_NEAR(c.b.back(), c.last, 1e-9);
        for (size_t i = 0; i < c.b.size(); ++i) {
            CHECK(c.b[i] > 0.0);
            /* STRICTLY INCREASING. A repeated bound gives a bucket of zero width, which every
             * quantile interpolation divides by. */
            if (i) CHECK(c.b[i] > c.b[i - 1]);
        }
    }
}

TEST(the_token_ladder_is_mantissas_1_2_5_stopping_past_the_context) {
    /* vLLM's build_1_2_5_buckets: a 32K-context model gets 1..20000 and no more. */
    const std::vector<double> b = buckets_1_2_5(32768);
    CHECK(!b.empty());
    CHECK_NEAR(b.front(), 1.0, 1e-12);
    CHECK_NEAR(b.back(), 20000.0, 1e-9);
    CHECK_EQ(b.size(), 14u);
    for (size_t i = 1; i < b.size(); ++i) CHECK(b[i] > b[i - 1]);
    /* Every entry is a mantissa of 1, 2 or 5 at some decade. */
    for (double v : b) {
        double m = v;
        while (m >= 10.0) m /= 10.0;
        CHECK(std::fabs(m - 1.0) < 1e-9 || std::fabs(m - 2.0) < 1e-9 || std::fabs(m - 5.0) < 1e-9);
    }
    /* It stops at the first value PAST the bound rather than including it: nothing above the
     * context can be observed, so a bucket there would never be reached. */
    for (double v : b) CHECK(v <= 32768.0);

    const std::vector<double> big = buckets_1_2_5(200000);
    CHECK(big.size() > b.size());
    CHECK_NEAR(big.back(), 200000.0, 1e-9);        /* the bound itself is legal when it is on the
                                                    * ladder, because it can be observed */

    /* A DEGENERATE BOUND STILL GIVES A USABLE HISTOGRAM. Zero or negative is clamped to one
     * rather than producing an empty ladder, which would make every observation fall in the
     * overflow bucket and the metric say nothing. */
    CHECK_EQ(buckets_1_2_5(1).size(), 1u);
    CHECK_EQ(buckets_1_2_5(0).size(), 1u);
    CHECK_EQ(buckets_1_2_5(-5).size(), 1u);
}


/* ================================================================== the response bodies
 *
 * WHAT THESE HAVE TO GET RIGHT is not the content, which the routes already cover end to end, but
 * the SHAPE: a client library deserialises into generated types, and a field of the wrong type --
 * a null where a string was declared, a missing usage block, a number where an array belongs --
 * is a hard parse failure rather than a degraded answer. They are also the only place `usage` is
 * assembled, and usage is what a caller bills against.
 */

namespace {

OaiRequest a_request(const char* id, const char* model) {
    OaiRequest r;
    r.id = id;
    r.model = model;
    r.created = 1700000000;
    return r;
}

/* nlohmann is already a dependency of this translation unit, so the assertions read the body as
 * a client would rather than searching it for substrings. */
json parsed(const std::string& body) {
    CHECK(json_complete(body));
    return json::parse(body, nullptr, false);
}

}  /* namespace */

static ModelCard card(const char* id, int64_t created, int64_t max_model_len = 0) {
    ModelCard c;
    c.id = id;
    c.created = created;
    c.max_model_len = max_model_len;
    return c;
}

TEST(the_model_list_is_a_list_of_one_with_the_fields_clients_read) {
    const json o = parsed(models_body(card("q38-flashnext", 1700000000)));
    CHECK_EQ(o.value("object", std::string()), std::string("list"));
    CHECK(o["data"].is_array());
    CHECK_EQ(o["data"].size(), 1u);
    CHECK_EQ(o["data"][0].value("id", std::string()), std::string("q38-flashnext"));
    CHECK_EQ(o["data"][0].value("object", std::string()), std::string("model"));
    CHECK_EQ(o["data"][0].value("created", (int64_t)0), (int64_t)1700000000);
    /* `owned_by` is in the schema and generated clients deserialise it. */
    CHECK(o["data"][0].contains("owned_by"));
    /* No length known, none reported: a client falls back to its own default rather than
     * trusting a made-up one. */
    CHECK(!o["data"][0].contains("max_model_len"));
    CHECK(!o["data"][0].contains("root"));
}

TEST(the_model_entry_names_the_file_behind_the_served_name) {
    ModelCard c = card("q38-flashnext", 1700000000);
    c.root = "/models/qwen3.8-next-flash-fp8-iq4r-moe.rad";
    CHECK_EQ(parsed(model_body(c)).value("root", std::string()), c.root);
}

TEST(the_model_list_reports_the_context_the_server_runs_with) {
    const json o = parsed(models_body(card("q38-flashnext", 1700000000, 262144)));
    CHECK_EQ(o["data"][0].value("max_model_len", (int64_t)0), (int64_t)262144);
}

TEST(one_model_is_the_entry_itself_not_a_list_of_one) {
    const json m = parsed(model_body(card("q38-flashnext", 1700000000, 262144)));
    CHECK_EQ(m.value("object", std::string()), std::string("model"));
    CHECK_EQ(m.value("id", std::string()), std::string("q38-flashnext"));
    CHECK_EQ(m.value("max_model_len", (int64_t)0), (int64_t)262144);
    CHECK(!m.contains("data"));
}

TEST(a_chat_body_carries_usage_and_one_choice_per_completion) {
    ChoiceOut a;
    a.index = 0;
    a.msg.content = "hello";
    a.finish = Finish::Stop;
    ChoiceOut b;
    b.index = 1;
    b.msg.content = "world";
    b.finish = Finish::Length;

    UsageOut u;
    u.prompt = 10;
    u.completion = 4;
    u.cached = 6;

    const json o = parsed(chat_completion_body(a_request("chatcmpl-1", "m"), { a, b }, u));
    CHECK_EQ(o.value("object", std::string()), std::string("chat.completion"));
    CHECK_EQ(o.value("id", std::string()), std::string("chatcmpl-1"));
    CHECK_EQ(o["choices"].size(), 2u);
    CHECK_EQ(o["choices"][0]["message"].value("role", std::string()), std::string("assistant"));
    CHECK_EQ(o["choices"][0]["message"].value("content", std::string()), std::string("hello"));
    CHECK_EQ(o["choices"][0].value("finish_reason", std::string()), std::string("stop"));
    CHECK_EQ(o["choices"][1].value("finish_reason", std::string()), std::string("length"));
    CHECK_EQ(o["choices"][1].value("index", -1), 1);

    /* TOTAL IS THE SUM AND NOT A THIRD COUNTER. A caller reconciling a bill against the two
     * halves has to get the same answer both ways. */
    CHECK_EQ(o["usage"].value("prompt_tokens", (int64_t)0), (int64_t)10);
    CHECK_EQ(o["usage"].value("completion_tokens", (int64_t)0), (int64_t)4);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)14);
    /* The cached count sits where vLLM puts it and not at the top level. */
    CHECK_EQ(o["usage"]["prompt_tokens_details"].value("cached_tokens", (int64_t)0), (int64_t)6);

    /* logprobs is PRESENT AND NULL rather than absent, because a generated client deserialising
     * into an optional field distinguishes "not asked for" from "the key does not exist". */
    CHECK(o["choices"][0].contains("logprobs"));
    CHECK(o["choices"][0]["logprobs"].is_null());
}

/* A TOOL CALL MAKES CONTENT NULL RATHER THAN EMPTY. The schema says a message with tool_calls has
 * no content, and a client branching on `content == null` -- which several do -- reads an empty
 * string as an ordinary text reply and never looks at the calls. */
TEST(a_chat_body_with_a_tool_call_sends_a_null_content) {
    ChoiceOut c;
    c.msg.tool_calls.push_back(ToolCall{ "call_1", "get_weather", "{\"city\":\"Berlin\"}" });
    c.finish = Finish::ToolCalls;

    const json o = parsed(chat_completion_body(a_request("chatcmpl-2", "m"), { c }, UsageOut{}));
    const json& m = o["choices"][0]["message"];
    CHECK(m["content"].is_null());
    CHECK(m["tool_calls"].is_array());
    CHECK_EQ(m["tool_calls"].size(), 1u);
    CHECK_EQ(o["choices"][0].value("finish_reason", std::string()), std::string("tool_calls"));

    /* Text beside a call keeps the text: content is only null when there is none. */
    ChoiceOut both = c;
    both.msg.content = "checking";
    const json p = parsed(chat_completion_body(a_request("chatcmpl-3", "m"), { both }, UsageOut{}));
    CHECK_EQ(p["choices"][0]["message"].value("content", std::string()), std::string("checking"));

    /* Reasoning is carried under the key vLLM publishes, and only when there is some: an empty
     * reasoning_content on every reply is noise in a transcript. */
    CHECK(!p["choices"][0]["message"].contains("reasoning_content"));
    ChoiceOut think;
    think.msg.reasoning = "the user wants weather";
    think.msg.content = "it is cold";
    const json q = parsed(chat_completion_body(a_request("chatcmpl-4", "m"), { think }, UsageOut{}));
    CHECK_EQ(q["choices"][0]["message"].value("reasoning_content", std::string()),
             std::string("the user wants weather"));
}

TEST(a_text_completion_body_is_the_other_shape_entirely) {
    ChoiceOut c;
    c.text = "once upon a time";
    c.finish = Finish::Stop;
    UsageOut u;
    u.prompt = 3;
    u.completion = 4;

    const json o = parsed(text_completion_body(a_request("cmpl-1", "m"), { c }, u));
    CHECK_EQ(o.value("object", std::string()), std::string("text_completion"));
    /* A completion choice carries `text` and no `message`; mixing the two is what makes one
     * endpoint's response undeserialisable by the other endpoint's client. */
    CHECK_EQ(o["choices"][0].value("text", std::string()), std::string("once upon a time"));
    CHECK(!o["choices"][0].contains("message"));
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)7);
}

/* THE LOGPROB SHAPES DIFFER BETWEEN THE TWO ENDPOINTS and both are fixed by the schema: chat
 * nests an array of per-token objects under `content`, completions carries four parallel arrays.
 * text_offset is cumulative over the token texts, which is what makes it an offset into `text`. */
TEST(logprobs_take_each_endpoints_own_shape) {
    ChoiceOut c;
    c.text = "abcd";
    c.logprobs = { { 1, "ab", -0.5f }, { 2, "cd", -1.5f } };
    OaiRequest r = a_request("x", "m");
    r.want_logprobs = true;

    const json chat = parsed(chat_completion_body(r, { c }, UsageOut{}));
    const json& lp = chat["choices"][0]["logprobs"];
    CHECK(lp["content"].is_array());
    CHECK_EQ(lp["content"].size(), 2u);
    CHECK_EQ(lp["content"][0].value("token", std::string()), std::string("ab"));
    CHECK_NEAR(lp["content"][0].value("logprob", 0.0), -0.5, 1e-6);
    CHECK(lp["content"][0]["bytes"].is_array());
    CHECK_EQ(lp["content"][0]["bytes"].size(), 2u);

    const json txt = parsed(text_completion_body(r, { c }, UsageOut{}));
    const json& tl = txt["choices"][0]["logprobs"];
    CHECK_EQ(tl["tokens"].size(), 2u);
    CHECK_EQ(tl["token_logprobs"].size(), 2u);
    CHECK_EQ(tl["text_offset"].size(), 2u);
    CHECK_EQ(tl["text_offset"][0].get<int64_t>(), (int64_t)0);
    CHECK_EQ(tl["text_offset"][1].get<int64_t>(), (int64_t)2);
}

/* AN SSE EVENT WITH NO CHOICES AND ONLY USAGE, which is what stream_options.include_usage asks
 * for. Accepting the field and not emitting it leaves a client waiting for a chunk that never
 * arrives, so the empty choices array has to be present rather than omitted. */
TEST(the_completion_usage_chunk_is_an_event_with_an_empty_choices_array) {
    UsageOut u;
    u.prompt = 5;
    u.completion = 2;
    const std::string ev = text_usage_chunk(a_request("cmpl-9", "m"), u);
    CHECK(ev.rfind("data: ", 0) == 0);
    CHECK(ev.compare(ev.size() - 2, 2, "\n\n") == 0);

    const json o = json::parse(ev.substr(6, ev.size() - 8), nullptr, false);
    CHECK(!o.is_discarded());
    CHECK_EQ(o.value("object", std::string()), std::string("text_completion"));
    CHECK(o["choices"].is_array());
    CHECK_EQ(o["choices"].size(), 0u);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)7);
    CHECK_EQ(o.value("id", std::string()), std::string("cmpl-9"));
}

/* THE BASE64 ENCODING IS PER-REQUEST AND IT IS THE RAW FLOAT BUFFER, which is what every client
 * that asks for it unpacks. Sending the JSON array under that flag, or the base64 of a formatted
 * array, both deserialise to nonsense rather than failing. */
TEST(an_embeddings_body_honours_the_encoding_the_request_asked_for) {
    const std::vector<std::vector<float>> vecs = { { 1.0f, 2.0f }, { -0.5f, 0.25f } };
    UsageOut u;
    u.prompt = 8;

    OaiRequest plain = a_request("emb-1", "m");
    const json o = parsed(embeddings_body(plain, vecs, u));
    CHECK_EQ(o.value("object", std::string()), std::string("list"));
    CHECK_EQ(o["data"].size(), 2u);
    CHECK_EQ(o["data"][0].value("object", std::string()), std::string("embedding"));
    CHECK_EQ(o["data"][0].value("index", (int64_t)-1), (int64_t)0);
    CHECK_EQ(o["data"][1].value("index", (int64_t)-1), (int64_t)1);
    CHECK(o["data"][0]["embedding"].is_array());
    CHECK_NEAR(o["data"][0]["embedding"][1].get<double>(), 2.0, 1e-6);
    /* An embedding has no completion half, so total is the prompt. */
    CHECK_EQ(o["usage"].value("prompt_tokens", (int64_t)0), (int64_t)8);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)8);

    OaiRequest b64 = a_request("emb-2", "m");
    b64.base64_embeddings = true;
    const json e = parsed(embeddings_body(b64, vecs, u));
    CHECK(e["data"][0]["embedding"].is_string());
    std::vector<uint8_t> raw;
    CHECK(base64_decode(e["data"][0]["embedding"].get<std::string>(), &raw));
    CHECK_EQ(raw.size(), 2 * sizeof(float));
    float back[2];
    std::memcpy(back, raw.data(), sizeof back);
    CHECK_NEAR(back[0], 1.0f, 1e-6);
    CHECK_NEAR(back[1], 2.0f, 1e-6);

    /* No inputs is an empty list rather than a null or an error: a caller that sent no strings
     * gets no vectors, and the envelope is still the one its client deserialises. */
    const json z = parsed(embeddings_body(plain, {}, UsageOut{}));
    CHECK(z["data"].is_array());
    CHECK_EQ(z["data"].size(), 0u);
}

RAD_TEST_MAIN()
