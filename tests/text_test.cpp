/* text_test.cpp -- the text frontend.
 *
 * THE IMPORTANT TEST is `tokenize_against_llama_cpp`. Its expected token ids come from
 * llama.cpp's own tokeniser -- `llama_tokenize()` with add_special=false and parse_special=true
 * -- over two models with DIFFERENT pre-tokenisers (pre=qwen2 and pre=qwen35), and they are BAKED
 * IN, so the test needs llama.cpp only to regenerate them, never to run. Regenerating means
 * building llama-tokenize and running it over the same strings; the tables below are the record
 * of what it said.
 *
 * Beyond the tables, the same comparison holds per model on random strings over Latin, Cyrillic,
 * CJK, Hangul, kana, emoji, ZWJ sequences, combining marks and arbitrary bytes, and on real C++
 * source, with one deliberate divergence, which `does_not_promote_control_looking_tokens` pins
 * and explains. Arbitrary-byte strings on which llama.cpp THROWS out of llama_tokenize are the
 * reason vendor/patches/0001 exists.
 *
 * The vocabs themselves are too large to bake -- 151,669 and 248,320 tokens -- so the vocab tests
 * read the GGUFs. If one is not there its tests are SKIPPED, loudly, and everything that needs no
 * vocab still runs. RAD_TEST_GGUF and RAD_TEST_GGUF_QWEN35 override the paths.
 */
#include "rad_test.h"

#include "text/gguf.h"
#include "text/tokenizer.h"
#include "text/unicode_norm.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

using namespace rad;

/* ------------------------------------------------------------------ helpers */

/* The table stores strings in an escaped form so a control character and a lone 0xFF byte are
 * both readable in source. Same escapes as the harness that produced the expectations. */
static std::string unescape(const std::string& in) {
    std::string u;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '\\' || i + 1 >= in.size()) { u += in[i]; continue; }
        ++i;
        switch (in[i]) {
            case 'n': u += '\n'; break;
            case 't': u += '\t'; break;
            case 'r': u += '\r'; break;
            case '\\': u += '\\'; break;
            case 'x': {
                auto h = [](int c) {
                    return c >= '0' && c <= '9' ? c - '0'
                         : c >= 'a' && c <= 'f' ? c - 'a' + 10
                         : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0;
                };
                if (i + 2 < in.size()) { u += (char)(h(in[i + 1]) * 16 + h(in[i + 2])); i += 2; }
                break;
            }
            default: u += '\\'; u += in[i];
        }
    }
    return u;
}

/* Two models, two pre-tokenisers.
 *
 *   qwen3e  Qwen3-Embedding-0.6B, tokenizer.ggml.pre = "qwen2"
 *   qwen35  Qwen3.5-0.8B,         tokenizer.ggml.pre = "qwen35"
 *
 * The two differ in the letter-run rule (`[\p{L}\p{M}]+` versus `\p{L}+`), which is exactly the
 * difference that a hash-the-file pre-tokeniser gets wrong on a model it has not seen. Both are
 * pinned.
 *
 * Both are checkpoint files and not in the repo: RAD_TEST_GGUF and RAD_TEST_GGUF_QWEN35 name
 * them. Unset, the cases that need a vocab skip and say which file they wanted. */
static const char* gguf_path_qwen3e() {
    if (const char* p = std::getenv("RAD_TEST_GGUF")) return p;
    return "Qwen3-Embedding-0.6B-f16.gguf (set RAD_TEST_GGUF)";
}
static const char* gguf_path_qwen35() {
    if (const char* p = std::getenv("RAD_TEST_GGUF_QWEN35")) return p;
    return "Qwen3.5-0.8B-BF16.gguf (set RAD_TEST_GGUF_QWEN35)";
}

/* Loaded once each. Reading 150K-250K tokens is ~0.2 s and every test wants the same one. */
static std::shared_ptr<const Vocab> load_vocab(const char* path) {
    auto g = std::make_shared<GgufFile>();
    if (g->open(path) < 0) return nullptr;
    VocabBuild vb;
    if (vocab_from_gguf(*g, vb) < 0) return nullptr;
    auto out = std::make_shared<Vocab>();
    if (out->load(std::move(vb)) < 0) return nullptr;
    return out;
}

static std::shared_ptr<const Vocab> qwen3e_vocab() {
    static std::shared_ptr<const Vocab> v = load_vocab(gguf_path_qwen3e());
    return v;
}
static std::shared_ptr<const Vocab> qwen35_vocab() {
    static std::shared_ptr<const Vocab> v = load_vocab(gguf_path_qwen35());
    return v;
}

/* The vocab the non-table tests use: Qwen3-Embedding-0.6B (qwen3e). */
static std::shared_ptr<const Vocab> qwen_vocab() { return qwen3e_vocab(); }
static const char* gguf_path() { return gguf_path_qwen3e(); }

static bool need_vocab(const char* what) {
    if (qwen_vocab()) return true;
    fprintf(stderr, "  SKIP %s: no vocab at %s (set RAD_TEST_GGUF)\n", what, gguf_path());
    return false;
}

/* ------------------------------------------------------------------ the baked table */

/* `detok` is what the round trip must produce. It is null wherever that is the input itself,
 * which is every case whose input is valid UTF-8. The three that are not are lossy on purpose:
 * an invalid byte becomes U+FFFD when the text is decoded to codepoints, so it cannot come back.
 * llama.cpp's detokenizer does exactly the same thing, and these expectations are its output. */
struct Case { const char* text; std::vector<int32_t> ids; const char* detok = nullptr; };

static const std::vector<Case>& cases_qwen35() {
    static const std::vector<Case> k = {
    { "Hello, world!",
      { 9419,11,1814,0 } },
    { "The quick brown fox jumps over the lazy dog. 1234567890",
      { 760,3841,13477,37550,33075,888,279,15217,5388,13,220,16,17,18,19,20,21,22,23,24,15 } },
    { "   leading spaces",
      { 256,6187,12258 } },
    { "trailing spaces   \\t\\t",
      { 371,13868,12258,66227 } },
    { "\\n\\n  mixed\\t whitespace \\n",
      { 271,220,9238,197,35169,695 } },
    { "你好，世界！这是一个测试。",
      { 109266,3709,96748,6115,109455,99449,1710 } },
    { "こんにちは世界。テストです。",
      { 85951,96748,1710,181801,36298,1710 } },
    { "안녕하세요 세계",
      { 148924,154982,88005,155308 } },
    { "Здравствуй, мир",
      { 34279,161541,19587,79734,11,160202 } },
    { "🙂🚀✨",
      { 169171,9008,248,222,169379 } },
    { "👨‍👩‍👧‍👦 family zwj",
      { 9008,239,101,373,235,9008,239,102,373,235,9008,239,100,373,235,9008,239,99,2902,23624,73 } },
    { "🏳️‍🌈 flag zwj vs16",
      { 9008,237,111,29545,373,235,9008,234,230,5016,23624,73,5972,16,21 } },
    { "def main():\\n\\tif x == 1:\\n\\t\\tprint(\"hi\")\\n\\treturn 0\\n",
      { 727,1822,4406,198,720,830,606,220,16,25,198,197,6685,437,5834,871,198,827,220,15,198 } },
    { "```cpp\\nint main() {\\n\\tstd::cout << \"x\" << std::endl;\\n}\\n```",
      { 71093,10504,198,390,1822,363,313,198,6524,476,5914,1079,328,87,1,1079,1407,476,5168,26,198,92,198,71093 } },
    { "bad byte: \\xff here",
      { 13458,4763,25,28373,1532 },  "bad byte: \\xef\\xbf\\xbd here" },
    { "lone continuation \\x80\\x81 end",
      { 75,588,39811,220,9678,809 },  "lone continuation \\xef\\xbf\\xbd\\xef\\xbf\\xbd end" },
    { "truncated emoji \\xf0\\x9f\\x98",
      { 371,36751,40958,220,55413 },  "truncated emoji \\xef\\xbf\\xbd\\xef\\xbf\\xbd\\xef\\xbf\\xbd" },
    { "café naïve Ångström",
      { 895,56868,91603,571,76533,938,485,82628 } },
    { "It's the model's output, isn't it?",
      { 2064,579,279,1558,579,2468,11,4290,914,424,30 } },
    { "<|im_start|>system\\nYou are helpful.<|im_end|>\\n",
      { 248045,8678,198,2523,513,10631,13,248046,198 } },
    };
    return k;
}

/* Qwen3-Embedding-0.6B, tokenizer.ggml.pre = "qwen2". Same strings, different vocab AND a
 * different letter-run rule, so the two tables disagree on nearly every line -- which is the
 * point: a tokeniser that quietly used one pre-tokeniser for both would pass one table.
 */
static const std::vector<Case>& cases_qwen3e() {
    static const std::vector<Case> k = {
    { "Hello, world!",
      { 9707,11,1879,0 } },
    { "The quick brown fox jumps over the lazy dog. 1234567890",
      { 785,3974,13876,38835,34208,916,279,15678,5562,13,220,16,17,18,19,20,21,22,23,24,15 } },
    { "   leading spaces",
      { 256,6388,12621 } },
    { "trailing spaces   \\t\\t",
      { 376,14277,12621,68546 } },
    { "\\n\\n  mixed\\t whitespace \\n",
      { 271,220,9519,197,36372,715 } },
    { "你好，世界！这是一个测试。",
      { 108386,3837,99489,6313,105464,81705,1773 } },
    { "こんにちは世界。テストです。",
      { 89015,99489,1773,140592,37541,1773 } },
    { "안녕하세요 세계",
      { 126246,144370,91145,133196 } },
    { "Здравствуй, мир",
      { 35451,6949,26988,20200,82580,11,137144 } },
    { "🙂🚀✨",
      { 145080,145836,144232 } },
    { "👨‍👩‍👧‍👦 family zwj",
      { 145367,378,235,145233,378,235,145665,378,235,145988,2997,24396,73 } },
    { "🏳️‍🌈 flag zwj vs16",
      { 147233,30543,378,235,145354,5181,24396,73,6165,16,21 } },
    { "def main():\\n\\tif x == 1:\\n\\t\\tprint(\"hi\")\\n\\treturn 0\\n",
      { 750,1887,3932,743,856,621,220,16,510,197,6900,445,6023,1138,853,220,15,198 } },
    { "```cpp\\nint main() {\\n\\tstd::cout << \"x\" << std::endl;\\n}\\n```",
      { 73594,10821,198,396,1887,368,341,6736,486,6104,1115,330,87,1,1115,1460,486,5336,280,532,73594 } },
    { "bad byte: \\xff here",
      { 13855,4922,25,29333,1588 },  "bad byte: \\xef\\xbf\\xbd here" },
    { "lone continuation \\x80\\x81 end",
      { 75,603,41171,220,9973,835 },  "lone continuation \\xef\\xbf\\xbd\\xef\\xbf\\xbd end" },
    { "truncated emoji \\xf0\\x9f\\x98",
      { 376,38007,42365,220,57332 },  "truncated emoji \\xef\\xbf\\xbd\\xef\\xbf\\xbd\\xef\\xbf\\xbd" },
    { "café naïve Ångström",
      { 924,58858,94880,586,79252,968,495,85584 } },
    { "It's the model's output, isn't it?",
      { 2132,594,279,1614,594,2550,11,4436,944,432,30 } },
    { "<|im_start|>system\\nYou are helpful.<|im_end|>\\n",
      { 151644,8948,198,2610,525,10950,13,151645,198 } },
    };
    return k;
}

/* ================================================================== the important test */

static void check_table(const char* name, std::shared_ptr<const Vocab> v,
                        const std::vector<Case>& cases) {
    if (!v) {
        fprintf(stderr, "  SKIP tokenize_against_llama_cpp[%s]: vocab not available\n", name);
        return;
    }
    Tokenizer t;
    CHECK_OK(t.init(v));
    for (const Case& c : cases) {
        std::vector<int32_t> got;
        CHECK_OK(t.encode(unescape(c.text), got, /*add_special=*/false, /*parse_special=*/true));
        if (got != c.ids) {
            std::string w, g;
            for (int32_t i : c.ids) w += std::to_string(i) + ",";
            for (int32_t i : got)   g += std::to_string(i) + ",";
            ::radtest::fail(__FILE__, __LINE__,
                            std::string(name) + " \"" + c.text +
                            "\"\n      want " + w + "\n      got  " + g);
        }
    }
}

TEST(tokenize_against_llama_cpp) {
    check_table("qwen3e", qwen3e_vocab(), cases_qwen3e());
    check_table("qwen35", qwen35_vocab(), cases_qwen35());
}

/* A DELIBERATE DIVERGENCE FROM llama.cpp, pinned so nobody "fixes" it by accident.
 *
 * llama.cpp carries a hardcoded list of control-LOOKING token texts -- "<|eot_id|>", "<eos>",
 * "</s>" and a dozen more -- and forces any of them to CONTROL even when the container says the
 * token is NORMAL. In Qwen3-Embedding's vocab "</s>" is id 128247 with token_type NORMAL: an
 * ordinary BPE merge learnt from web text, not an added token. llama.cpp's list promotes it, so
 * llama.cpp's special-token partitioner isolates "</s>" in the middle of a document and emits it
 * as one token.
 *
 * HuggingFace does not: only `added_tokens` are isolated, "</s>" is not one of them, and the
 * pre-tokeniser splits it at the letter, so the reference tokenisation is three tokens. We follow
 * the vocab as declared, which means we agree with HuggingFace and disagree with llama.cpp on
 * exactly this input. On C++ source containing "</s>" literals, and on fuzzed text, it is the
 * only difference between the two.
 *
 * If this test starts failing because we now match llama.cpp, someone has added a recognise-the-
 * token heuristic, and that is the thing spec §12 exists to remove. */
TEST(does_not_promote_control_looking_tokens) {
    if (!qwen3e_vocab()) {
        fprintf(stderr, "  SKIP does_not_promote_control_looking_tokens: vocab not available\n");
        return;
    }
    const int32_t id = qwen3e_vocab()->find("</s>");
    CHECK(id >= 0);
    CHECK_EQ(qwen3e_vocab()->type(id), (uint8_t)RAD_TT_NORMAL);

    Tokenizer t;
    CHECK_OK(t.init(qwen3e_vocab()));
    std::vector<int32_t> got;
    CHECK_OK(t.encode("x </s> y", got, false, /*parse_special=*/true));
    CHECK(std::find(got.begin(), got.end(), id) == got.end());
}

/* A control token typed by a user must NOT become a role switch. With parse_special off,
 * "<|im_start|>" is ordinary text and tokenises into several ordinary tokens. */
TEST(special_tokens_are_not_parsed_from_user_text) {
    if (!need_vocab("special_tokens_are_not_parsed_from_user_text")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    std::vector<int32_t> as_special, as_text;
    CHECK_OK(t.encode("<|im_start|>", as_special, false, true));
    CHECK_OK(t.encode("<|im_start|>", as_text,    false, false));
    CHECK_EQ(as_special.size(), (size_t)1);
    CHECK(as_text.size() > 1);
    CHECK(as_text != as_special);
}

/* ================================================================== detokenisation */

TEST(detokenize_round_trip) {
    if (!need_vocab("detokenize_round_trip")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    for (const Case& c : cases_qwen3e()) {
        const std::string want = unescape(c.detok ? c.detok : c.text);
        /* Streaming, one token at a time -- the way the server drives it. The concatenation of
         * every push() plus flush() must equal the input, byte for byte, including the invalid
         * UTF-8 the byte-level alphabet carried through as bytes. */
        Detokenizer d(qwen_vocab(), {}, /*render_special=*/true);
        std::string got;
        for (int32_t id : c.ids) got += d.push(id);
        got += d.flush();
        CHECK_EQ(got, want);
    }
}

/* Every push() must return COMPLETE UTF-8. A three-byte codepoint that arrives as three separate
 * byte tokens must not appear until its last byte does. */
TEST(detokenize_holds_partial_utf8) {
    if (!need_vocab("detokenize_holds_partial_utf8")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    /* Build the worst case rather than hoping a string produces it: take a three-byte codepoint
     * and hand the detokeniser its three byte-level tokens one at a time. Every byte-level vocab
     * has all 256 of them, so this is the same test on any model. */
    /* U+4EFF, whose three UTF-8 bytes are E4 BB BF -- all of them in the 0xAE..0xFF range the
     * GPT-2 alphabet maps to ITSELF as a codepoint, so each token's text is that codepoint in
     * UTF-8 and the test does not have to reimplement the remapping. */
    const std::string cjk = "\xe4\xbb\xbf";
    std::vector<int32_t> ids;
    for (unsigned char b : cjk) {
        /* Looking the token up rather than encoding the byte is deliberate: a lone continuation
         * byte is not valid UTF-8 INPUT and encode() would replace it with U+FFFD long before it
         * reached the vocab. The point of this test is what the DECODER does with such a token. */
        CHECK(b >= 0xAE);
        std::string enc;
        enc += (char)(0xC0 | (b >> 6));
        enc += (char)(0x80 | (b & 0x3F));
        const int32_t id = qwen_vocab()->find(enc);
        CHECK(id >= 0);
        ids.push_back(id);
    }
    CHECK_EQ(ids.size(), (size_t)3);

    Detokenizer d(qwen_vocab());
    std::string all;
    bool saw_partial_hold = false;
    for (int32_t id : ids) {
        const std::string out = d.push(id);
        /* Whatever came out is complete UTF-8: no truncated trailing sequence. */
        size_t i = out.size(), back = 0;
        while (i > 0 && back < 4) {
            --i; ++back;
            const unsigned char ch = (unsigned char)out[i];
            if ((ch & 0xC0) == 0x80) continue;
            const size_t need = ch < 0x80 ? 1 : ch < 0xE0 ? 2 : ch < 0xF0 ? 3 : 4;
            CHECK_EQ(need, out.size() - i);
            break;
        }
        if (out.empty()) saw_partial_hold = true;
        all += out;
    }
    all += d.flush();
    CHECK_EQ(all, cjk);
    CHECK(saw_partial_hold);   /* at least one push held everything back */
}

/* The straddle: a stop string arriving across several tokens must never be partially emitted. */
TEST(detokenize_stop_string_straddle) {
    if (!need_vocab("detokenize_stop_string_straddle")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    const std::string body = "the answer is 42";
    const std::string stop = "</tool_call>";
    std::vector<int32_t> ids;
    CHECK_OK(t.encode(body + stop + " and this must never be seen", ids, false, true));

    Detokenizer d(qwen_vocab(), { stop });
    std::string emitted;
    for (int32_t id : ids) {
        const std::string out = d.push(id);
        emitted += out;
        /* Nothing emitted so far may contain any part of the stop string as a suffix, which is
         * the property that makes the emitted text safe to forward to a client immediately. */
        for (size_t n = 1; n < stop.size(); ++n) {
            if (emitted.size() >= n)
                CHECK(emitted.compare(emitted.size() - n, n, stop, 0, n) != 0);
        }
    }
    CHECK(d.stopped());
    CHECK_EQ(d.stop_hit(), stop);
    CHECK_EQ(emitted, body);
    /* flush() after a stop yields nothing: the tail was discarded, not buffered. */
    CHECK_EQ(d.flush(), std::string());
}

/* A reply that starts inside its reasoning block: a stop string written while thinking is text,
 * not a stop, and the first one after the closer cuts as usual. */
TEST(detokenize_stops_wait_for_the_reasoning_closer) {
    if (!need_vocab("detokenize_stops_wait_for_the_reasoning_closer")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    const std::string think = "restating it, Q: what is two and two</think>";
    std::vector<int32_t> ids;
    CHECK_OK(t.encode(think + "The answer is 4. Q: next", ids, false, true));
    Detokenizer d(qwen_vocab(), { "Q:" });
    d.defer_stops_until("</think>");
    std::string s;
    for (int32_t id : ids) {
        s += d.push(id);
        if (d.stopped()) break;
    }
    CHECK(d.stopped());
    CHECK_EQ(d.stop_hit(), std::string("Q:"));
    CHECK_EQ(s, think + "The answer is 4. ");

    /* Without the deferral the same text stops inside the reasoning. */
    Detokenizer plain(qwen_vocab(), { "Q:" });
    std::string p;
    for (int32_t id : ids) {
        p += plain.push(id);
        if (plain.stopped()) break;
    }
    CHECK_EQ(p, std::string("restating it, "));
}

/* A stop string that never completes must be released at flush() rather than swallowed. */
TEST(detokenize_incomplete_stop_is_flushed) {
    if (!need_vocab("detokenize_incomplete_stop_is_flushed")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    std::vector<int32_t> ids;
    CHECK_OK(t.encode("done </tool", ids, false, true));
    Detokenizer d(qwen_vocab(), { "</tool_call>" });
    std::string s;
    for (int32_t id : ids) s += d.push(id);
    CHECK(!d.stopped());
    s += d.flush();
    CHECK_EQ(s, std::string("done </tool"));
}

/* ================================================================== byte level */

/* The GPT-2 alphabet, spot-checked at every boundary of the three literal ranges plus the
 * remapped run. Getting one of these wrong costs 20% of the prompt length and nothing else,
 * which is why it is pinned rather than trusted. */
TEST(bytelevel_alphabet_is_exact) {
    if (!need_vocab("bytelevel_alphabet_is_exact")) return;
    Tokenizer t;
    CHECK_OK(t.init(qwen_vocab()));

    /* Space is byte 0x20, which is NOT in a literal range, so it is the first remapped byte and
     * maps to U+0100 'Ġ'. That single fact is the one every byte-level vocab is built on. */
    const int32_t g = qwen_vocab()->find("\xc4\xa0");           /* U+0120 = 'Ġ' */
    CHECK(g >= 0);
    std::vector<int32_t> ids;
    CHECK_OK(t.encode(" ", ids, false, true));
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], g);

    /* Newline is 0x0A, the eleventh remapped byte, U+010A 'Ċ'. llama.cpp prints it as the LF
     * token at id 198 for this vocab. */
    ids.clear();
    CHECK_OK(t.encode("\n", ids, false, true));
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], 198);
    CHECK_EQ(qwen_vocab()->text(198), std::string("\xc4\x8a"));  /* U+010A */

    /* '!' is 0x21, the first byte of the first literal range, and stands for itself. */
    CHECK(qwen_vocab()->find("!") >= 0);
    /* 0x7E '~' is the last of it. */
    CHECK(qwen_vocab()->find("~") >= 0);
}

/* ================================================================== normalisation */

/* Pinned vectors from UAX #15. The tables themselves are generated from ICU
 * (core/text/unicode_norm_data.cpp); these are the cases that would break first if the tables
 * were regenerated wrong. */
TEST(unicode_normalisation) {
    auto cpts = [](std::initializer_list<uint32_t> v) { return std::vector<uint32_t>(v); };

    /* Composed vs decomposed Angstrom: U+00C5 decomposes to A + ring, and U+212B (the Angstrom
     * SIGN) is a singleton that NFC folds onto U+00C5 rather than back onto itself. */
    CHECK(unorm_nfd(cpts({0x00C5})) == cpts({0x0041, 0x030A}));
    CHECK(unorm_nfc(cpts({0x0041, 0x030A})) == cpts({0x00C5}));
    CHECK(unorm_nfc(cpts({0x212B})) == cpts({0x00C5}));

    /* Canonical ordering: two marks of different combining class are sorted, and composition
     * then sees the one it can compose with. */
    CHECK(unorm_nfd(cpts({0x1E69})) == cpts({0x0073, 0x0323, 0x0307}));
    CHECK(unorm_nfc(cpts({0x0073, 0x0307, 0x0323})) == cpts({0x1E69}));

    /* A composition exclusion: U+0344 decomposes but never recomposes. */
    CHECK(unorm_nfd(cpts({0x0344})) == cpts({0x0308, 0x0301}));
    CHECK(unorm_nfc(cpts({0x0308, 0x0301})) == cpts({0x0308, 0x0301}));

    /* Hangul, which is arithmetic rather than tabular. */
    CHECK(unorm_nfd(cpts({0xAC01})) == cpts({0x1100, 0x1161, 0x11A8}));
    CHECK(unorm_nfc(cpts({0x1100, 0x1161, 0x11A8})) == cpts({0xAC01}));
    CHECK(unorm_nfc(cpts({0x1100, 0x1161})) == cpts({0xAC00}));

    /* Compatibility: the ligature and the fullwidth letter move only under the K forms. */
    CHECK(unorm_nfd (cpts({0xFB01})) == cpts({0xFB01}));
    CHECK(unorm_nfkd(cpts({0xFB01})) == cpts({0x0066, 0x0069}));
    CHECK(unorm_nfkc(cpts({0xFF21})) == cpts({0x0041}));
    CHECK(unorm_nfc (cpts({0xFF21})) == cpts({0xFF21}));

    /* The UTF-8 wrapper, including the ASCII fast path. */
    CHECK_EQ(unorm_utf8("plain ascii", true, true), std::string("plain ascii"));
    CHECK_EQ(unorm_utf8("A\xcc\x8a", false, true), std::string("\xc3\x85"));
    CHECK_EQ(unorm_utf8("\xc3\x85", false, false), std::string("A\xcc\x8a"));
}

/* ================================================================== gguf reader */

TEST(gguf_reader) {
    GgufFile g;
    if (g.open(gguf_path()) < 0) {
        fprintf(stderr, "  SKIP gguf_reader: no file at %s\n", gguf_path());
        return;
    }
    CHECK(g.version() >= 2);
    CHECK(!g.kv().empty());
    CHECK(!g.tensors().empty());

    /* The architecture names its own metadata prefix, which is the one GGUF convention a reader
     * has to know: `general.architecture` is "qwen3" and the dimensions are under "qwen3.*". */
    std::string arch;
    CHECK(g.get_str("general.architecture", &arch));
    CHECK(!arch.empty());

    int64_t layers = 0;
    CHECK(g.get_i64(arch + ".block_count", &layers));
    CHECK(layers > 0);

    std::vector<std::string> toks;
    CHECK(g.get_strs("tokenizer.ggml.tokens", &toks));
    CHECK(toks.size() > 100000);

    /* Every tensor's extent must be inside the file, and tensor_data() must agree. */
    for (const GgufTensor& t : g.tensors()) {
        CHECK(gguf_dtype(t.ggml_type) != nullptr);
        CHECK(t.bytes > 0);
        CHECK(g.tensor_data(t) != nullptr);
        CHECK((const uint8_t*)g.tensor_data(t) + t.bytes
              <= (const uint8_t*)g.base() + g.file_bytes());
    }

    /* Lookup by name agrees with the directory. */
    const GgufTensor& first = g.tensors()[0];
    CHECK(g.tensor(first.name) == &first);
    CHECK(g.tensor("this tensor does not exist") == nullptr);

    /* A missing key is a false return and not a fabricated default. */
    int64_t v = 12345;
    CHECK(!g.get_i64("no.such.key", &v));
    CHECK_EQ(v, (int64_t)12345);
}

TEST(gguf_rejects_non_gguf) {
    GgufFile g;
    /* Something that exists and is not a GGUF: this source file. */
    const int s = g.open(__FILE__);
    CHECK(s == RAD_E_FORMAT);
    CHECK(!g.is_open());

    GgufFile h;
    CHECK_EQ(h.open("/definitely/not/a/path/anywhere.gguf"), RAD_E_IO);
}

namespace {

/* A GGUF file byte by byte, for the shapes no producer writes on purpose. */
struct GgufBytes {
    std::vector<uint8_t> b;
    template <class T> void pod(T v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof v);
    }
    void str(const std::string& s) {
        pod<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void header(uint64_t n_tensors, uint64_t n_kv) {
        for (char c : std::string("GGUF")) b.push_back((uint8_t)c);
        pod<uint32_t>(3);
        pod<uint64_t>(n_tensors);
        pod<uint64_t>(n_kv);
    }
    /* One tensor record, then the data section padded to the default alignment. */
    void tensor(const std::vector<int64_t>& ne, uint32_t type, size_t data_bytes) {
        str("w");
        pod<uint32_t>((uint32_t)ne.size());
        for (int64_t n : ne) pod<int64_t>(n);
        pod<uint32_t>(type);
        pod<uint64_t>(0);
        while (b.size() % 32) b.push_back(0);
        b.insert(b.end(), data_bytes, 0);
    }
    int open(GgufFile* g) const {
        const char* dir = std::getenv("TMPDIR");
        const std::string path = std::string(dir && *dir ? dir : "/tmp") + "/radgguf_" +
                                 std::to_string((long)::getpid()) + ".gguf";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return RAD_E_IO;
        fwrite(b.data(), 1, b.size(), f);
        fclose(f);
        const int st = g->open(path.c_str());
        ::unlink(path.c_str());
        return st;
    }
};

}  /* namespace */

/* COUNTS AND DIMENSIONS IN A GGUF ARE THE FILE'S CLAIMS, NOT FACTS. A string array's count that
 * sizes a reservation before a single string is read ends the converter at 2^60; the product of
 * a tensor's dimensions can wrap, to a tensor of zero bytes that passes the bounds check while its
 * dimensions say it is vast. Each is refused against what the file can hold. */
TEST(gguf_refuses_counts_and_sizes_its_file_cannot_hold) {
    {
        GgufBytes f;                    /* a string array of 2^60 */
        f.header(0, 1);
        f.str("tokenizer.ggml.tokens");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(1ull << 60);
        f.str("a");
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* an array of a type GGUF does not define */
        f.header(0, 1);
        f.str("k");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(99); f.pod<uint64_t>(1ull << 60);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* 2^32 x 2^32: the element count wraps to zero */
        f.header(1, 0);
        f.tensor({ 1ll << 32, 1ll << 32 }, 0, 64);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* 2^62 f32: the count fits, its bytes wrap to zero */
        f.header(1, 0);
        f.tensor({ 1ll << 62 }, 0, 64);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* more KVs than the rest of the file can encode */
        f.header(0, 200);
        f.b.insert(f.b.end(), 100, 0);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        /* And a well-formed file with each of those shapes at a sane size opens and reads. */
        GgufBytes f;
        f.header(1, 2);
        f.str("general.name"); f.pod<uint32_t>(GGUF_STR); f.str("t");
        f.str("tokenizer.ggml.tokens");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(2);
        f.str("a"); f.str("b");
        f.tensor({ 4, 2 }, 0, 32);
        GgufFile g;
        CHECK_OK(f.open(&g));
        std::vector<std::string> toks;
        CHECK(g.get_strs("tokenizer.ggml.tokens", &toks));
        CHECK_EQ(toks.size(), (size_t)2);
        CHECK_EQ(g.tensors().size(), (size_t)1);
        CHECK_EQ(g.tensors()[0].bytes, (uint64_t)32);
    }
}

/* ================================================================== the vocab section */

namespace {
/* The minimum a StringSink has to be: rad-convert's real one dedupes; this one does not need to,
 * because the test is about the offsets being read back the way they were written. */
struct TestStrings : StringSink {
    std::string blob{ '\0' };   /* offset 0 is the empty string, per rad_format.h */
    rad_stroff intern(std::string_view s) override {
        if (s.empty()) return 0;
        const rad_stroff at = blob.size();
        blob.append(s);
        blob.push_back('\0');
        return at;
    }
};
}  /* namespace */

TEST(vocab_section_round_trip) {
    VocabBuild in;
    in.kind   = RAD_TOK_BPE;
    in.tokens = { "<unk>", "a", "b", "ab", "\xc4\xa0" };
    in.types  = { RAD_TT_UNKNOWN, RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_NORMAL };
    in.merges = { { 1, 2 } };
    in.bos = -1; in.eos = 2; in.eot = 2; in.unk = 0; in.pad = -1; in.sep = -1;
    in.add_bos = false;
    in.add_eos = true;
    in.chat_template = "{{ messages }}";
    VocabStep s0; s0.kind = RAD_NORM_NFC;             in.steps.push_back(s0);
    VocabStep s1; s1.kind = RAD_PRE_SPLIT; s1.arg0 = "\\p{N}"; s1.iarg = 1; in.steps.push_back(s1);
    VocabStep s2; s2.kind = RAD_PRE_BYTELEVEL; s2.flags = 2;  in.steps.push_back(s2);
    VocabStep s3; s3.kind = RAD_DEC_BYTELEVEL;        in.steps.push_back(s3);

    TestStrings strings;
    std::vector<uint8_t> section;
    /* Lay the file out the way rad-convert would: header, then the string blob, then the vocab
     * section. The offsets the serialiser writes are absolute, so the section has to know where
     * it lands before it is written -- and it lands 8-aligned, as the writer places it, because
     * the reader takes the header as a typed pointer into the file. */
    const uint64_t str_off = sizeof(RadFileHeader);
    CHECK_OK(vocab_serialize(in, strings, 0, section));   /* pass 1: sizes only */
    const uint64_t sec_off = (str_off + strings.blob.size() + 7) & ~(uint64_t)7;

    TestStrings strings2;
    section.clear();
    CHECK_OK(vocab_serialize(in, strings2, sec_off, section));

    std::vector<uint8_t> file(sec_off + section.size(), 0);
    RadFileHeader* fh = reinterpret_cast<RadFileHeader*>(file.data());
    fh->magic = RAD_MAGIC;
    fh->version = RAD_FORMAT_VER;
    fh->str_off = str_off;
    fh->str_bytes = strings2.blob.size();
    fh->vocab_off = sec_off;
    fh->vocab_bytes = section.size();
    std::memcpy(file.data() + str_off, strings2.blob.data(), strings2.blob.size());
    std::memcpy(file.data() + sec_off, section.data(), section.size());

    VocabBuild out;
    CHECK_OK(vocab_deserialize(file.data(), file.size(),
                               reinterpret_cast<const RadVocabHeader*>(file.data() + sec_off),
                               out));

    CHECK_EQ(out.kind, in.kind);
    CHECK(out.tokens == in.tokens);
    CHECK(out.types == in.types);
    CHECK(out.merges == in.merges);
    CHECK_EQ(out.eos, in.eos);
    CHECK_EQ(out.unk, in.unk);
    CHECK_EQ(out.add_eos, in.add_eos);
    CHECK_EQ(out.chat_template, in.chat_template);
    CHECK_EQ(out.steps.size(), in.steps.size());
    for (size_t i = 0; i < out.steps.size(); ++i) {
        CHECK_EQ(out.steps[i].kind, in.steps[i].kind);
        CHECK_EQ(out.steps[i].flags, in.steps[i].flags);
        CHECK_EQ(out.steps[i].arg0, in.steps[i].arg0);
        CHECK_EQ(out.steps[i].iarg, in.steps[i].iarg);
    }

    /* And the round-tripped vocab actually tokenises. */
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(out)));
    Tokenizer t;
    CHECK_OK(t.init(v));
    std::vector<int32_t> ids;
    CHECK_OK(t.encode("ab", ids, false, false));
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], 3);      /* the merge fired */
}

/* ================================================================== tokenizer.json */

TEST(parse_tokenizer_json_chain) {
    /* A miniature Qwen-shaped tokenizer.json: byte-level BPE with an explicit Split. The point
     * of the test is the CHAIN, not the vocab -- that the declared steps come out in the
     * declared order with the declared arguments. */
    const char* js = R"JSON({
      "added_tokens": [ {"id": 4, "content": "<|end|>", "special": true} ],
      "normalizer": { "type": "Sequence", "normalizers": [ {"type": "NFC"},
                      {"type": "Replace", "pattern": {"String": "  "}, "content": " "} ] },
      "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
          {"type": "Split", "pattern": {"Regex": "\\p{N}"}, "behavior": "Isolated", "invert": false},
          {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false} ] },
      "decoder": { "type": "ByteLevel" },
      "model": { "type": "BPE", "byte_fallback": false,
                 "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3, "<|end|>": 4},
                 "merges": ["a b"] }
    })JSON";

    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(js, "test.json", vb));
    CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_BPE);
    CHECK_EQ(vb.tokens.size(), (size_t)5);
    CHECK_EQ(vb.merges.size(), (size_t)1);
    CHECK_EQ(vb.types[4], (uint8_t)RAD_TT_CONTROL);

    CHECK_EQ(vb.steps.size(), (size_t)5);
    CHECK_EQ(vb.steps[0].kind, (uint32_t)RAD_NORM_NFC);
    CHECK_EQ(vb.steps[1].kind, (uint32_t)RAD_NORM_REPLACE);
    CHECK_EQ(vb.steps[1].arg0, std::string("  "));
    CHECK_EQ(vb.steps[1].arg1, std::string(" "));
    CHECK_EQ(vb.steps[2].kind, (uint32_t)RAD_PRE_SPLIT);
    CHECK_EQ(vb.steps[2].arg0, std::string("\\p{N}"));
    CHECK_EQ(vb.steps[3].kind, (uint32_t)RAD_PRE_BYTELEVEL);
    CHECK_EQ(vb.steps[3].flags, (uint32_t)0);      /* no prefix space, no inline regex */
    CHECK_EQ(vb.steps[4].kind, (uint32_t)RAD_DEC_BYTELEVEL);
}

/* A component we cannot express must FAIL rather than be dropped. That refusal is the whole
 * point of interpreting tokenizer.json instead of hashing it. */
TEST(parse_tokenizer_json_refuses_unknown_components) {
    const char* js = R"JSON({
      "normalizer": { "type": "Precompiled", "precompiled_charsmap": "AAAA" },
      "pre_tokenizer": { "type": "ByteLevel" },
      "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] }
    })JSON";
    VocabBuild vb;
    CHECK_EQ(parse_tokenizer_json_buf(js, "test.json", vb), RAD_E_UNSUPPORTED);
}

TEST(parse_tokenizer_json_rejects_bpe_without_pretokenizer) {
    const char* js = R"JSON({
      "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] }
    })JSON";
    VocabBuild vb;
    CHECK_EQ(parse_tokenizer_json_buf(js, "test.json", vb), RAD_E_FORMAT);
}

/* AN ID IS AN INDEX THE MOMENT IT IS READ. The token table is sized from the largest id and then
 * written at each one, so a negative id writes outside it, a huge one sizes it to tens of
 * gigabytes, and a value of the wrong JSON type throws out of the reader and ends the converter.
 * Every one of these is a malformed file and comes back as a refusal. */
TEST(parse_tokenizer_json_refuses_ids_and_values_it_cannot_place) {
    const std::string pre =
        R"("pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true})";
    auto bpe = [&](const std::string& vocab, const std::string& extra) {
        return "{" + pre + R"(, "model": {"type": "BPE", "vocab": )" + vocab +
               R"(, "merges": []})" + extra + "}";
    };
    const std::string bad[] = {
        bpe(R"({"a": 0, "b": -1})", ""),                                  /* negative */
        bpe(R"({"a": "x"})", ""),                                         /* not a number */
        bpe(R"({"a": 0, "b": 1.5})", ""),                                 /* not an integer */
        bpe(R"({"a": 2147483647})", ""),                                  /* past the file */
        bpe(R"(["a", "b"])", ""),                                         /* not {text: id} */
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": -1, "content": "x"}])"),
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": 4000000000, "content": "x"}])"),
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": 3, "content": 7}])"),
        R"({"model": {"type": "Unigram", "vocab": [["a", "x"]]}})",
        R"({"model": {"type": "Unigram", "vocab": [[5, 1.0]]}})",
        R"({"model": {"type": "Unigram", "unk_id": 9, "vocab": [["a", -1.0]]}})",
        R"({"model": {"type": "Unigram", "unk_id": -1, "vocab": [["a", -1.0]]}})",
        R"({"model": {"type": "WordPiece", "vocab": {"a": 0, "b": -7}}})",
        R"({"pre_tokenizer": {"type": "Split", "pattern": {"String": 5}}, )"
        R"("model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}})",
    };
    for (const std::string& js : bad) {
        VocabBuild vb;
        const int st = parse_tokenizer_json_buf(js, "test.json", vb);
        CHECK_EQ(st, RAD_E_FORMAT);
        if (st != RAD_E_FORMAT) fprintf(stderr, "    accepted: %s\n", js.c_str());
    }

    /* A merge that is not two strings is dropped like one naming a missing token, not fatal. */
    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(
        "{" + pre + R"(, "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2},)"
                    R"( "merges": [[1, 2], ["a", "b"]]}})", "test.json", vb));
    CHECK_EQ(vb.merges.size(), (size_t)1);

    /* And a sparse but sane table still loads: the gap is empty text. */
    CHECK_OK(parse_tokenizer_json_buf(bpe(R"({"a": 0, "b": 5})", ""), "test.json", vb));
    CHECK_EQ(vb.tokens.size(), (size_t)6);
    CHECK_EQ(vb.tokens[5], std::string("b"));
}

/* Unigram takes the other path through the model reader: an array vocab with scores, and Viterbi
 * rather than merges. */
TEST(unigram_viterbi) {
    const char* js = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "decoder": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram", "unk_id": 0,
                 "vocab": [["<unk>", 0.0], ["▁", -3.0], ["▁he", -1.0],
                           ["h", -6.0], ["e", -6.0], ["▁h", -5.0], ["llo", -2.0],
                           ["l", -7.0], ["o", -7.0]] }
    })JSON";
    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(js, "test.json", vb));
    CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_UNIGRAM);

    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    Tokenizer t;
    CHECK_OK(t.init(v));

    std::vector<int32_t> ids;
    CHECK_OK(t.encode("hello", ids, false, false));
    /* "_he" (-1) + "llo" (-2) = -3 beats "_h"(-5)+"e"(-6)+... by a wide margin. */
    CHECK_EQ(ids.size(), (size_t)2);
    CHECK_EQ(ids[0], 2);
    CHECK_EQ(ids[1], 6);
    CHECK_EQ(t.decode(ids), std::string(" hello"));
}

/* A UNIGRAM VOCABULARY WITH NO UNK TOKEN still routes text no piece covers through the unk
 * transition, and the id of that transition is then -1: a token id no vocabulary has, handed to
 * the model as input. It is spelled as byte tokens when the chain declares byte fallback, and
 * dropped otherwise -- what the BPE encoder does with a piece it cannot match. */
TEST(unigram_without_an_unk_token_never_emits_an_invalid_id) {
    auto encode = [](const char* js, const char* text) {
        std::vector<int32_t> ids;
        VocabBuild vb;
        if (parse_tokenizer_json_buf(js, "test.json", vb) < 0) return ids;
        auto v = std::make_shared<Vocab>();
        if (v->load(std::move(vb)) < 0) return ids;
        Tokenizer t;
        if (t.init(v) < 0) return ids;
        t.encode(text, ids, false, false);
        return ids;
    };
    /* "x" and the euro sign are in no piece. */
    const char* plain = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram",
                 "vocab": [["▁", -3.0], ["▁he", -1.0], ["h", -6.0], ["e", -6.0],
                           ["l", -7.0], ["o", -7.0]] } })JSON";
    const std::vector<int32_t> dropped = encode(plain, "hex\xe2\x82\xac" "lo");
    CHECK(dropped == (std::vector<int32_t>{ 1, 4, 5 }));

    const char* bytes = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram", "byte_fallback": true,
                 "vocab": [["▁", -3.0], ["▁he", -1.0], ["h", -6.0], ["e", -6.0],
                           ["l", -7.0], ["o", -7.0], ["<0x78>", -9.0], ["<0xE2>", -9.0],
                           ["<0x82>", -9.0], ["<0xAC>", -9.0]] } })JSON";
    const std::vector<int32_t> spelled = encode(bytes, "hex\xe2\x82\xac" "lo");
    CHECK(spelled == (std::vector<int32_t>{ 1, 6, 7, 8, 9, 4, 5 }));
}

/* A special id is emitted as it stands, so one naming no token -- which a GGUF or a container can
 * carry, since nothing else checks them -- is treated as absent rather than handed out. */
TEST(vocab_treats_an_out_of_range_special_id_as_absent) {
    VocabBuild vb;
    vb.kind   = RAD_TOK_BPE;
    vb.tokens = { "a", "b", "</s>" };
    vb.eos = 2; vb.unk = 7; vb.bos = -5; vb.pad = 3;
    vb.add_eos = true;
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    CHECK_EQ(v->eos(), 2);
    CHECK_EQ(v->unk(), -1);
    CHECK_EQ(v->bos(), -1);
    CHECK_EQ(v->pad(), -1);
}

/* ================================================================== chain steps */

TEST(step_chain_units) {
    /* A vocab whose tokens are single characters, so the test observes the SPLIT rather than the
     * merges. Each case installs one step and checks what the pre-tokeniser produced by looking
     * at how many tokens came out. */
    auto make = [](std::vector<VocabStep> steps) {
        VocabBuild vb;
        vb.kind = RAD_TOK_BPE;
        for (int c = 0; c < 128; ++c) vb.tokens.push_back(std::string(1, (char)c));
        vb.tokens.push_back("ab");
        vb.merges = { { 'a', 'b' } };    /* so "a" + "b" can reach id 128 */
        vb.steps = std::move(steps);
        auto v = std::make_shared<Vocab>();
        v->load(std::move(vb));
        return v;
    };

    /* Digits, individual: "a12b" -> "a", "1", "2", "b". */
    {
        VocabStep d; d.kind = RAD_PRE_DIGITS; d.flags = 1;
        Tokenizer t;
        CHECK_OK(t.init(make({ d })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("a12b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)4);
    }
    /* Punctuation, isolated: "a,b" -> "a", ",", "b" and the merge of "ab" cannot fire. */
    {
        VocabStep p; p.kind = RAD_PRE_PUNCTUATION; p.iarg = 1;
        Tokenizer t;
        CHECK_OK(t.init(make({ p })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("a,b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)3);
    }
    /* Lowercase normaliser. */
    {
        VocabStep n; n.kind = RAD_NORM_LOWERCASE;
        VocabStep w; w.kind = RAD_PRE_WHITESPACE;
        Tokenizer t;
        CHECK_OK(t.init(make({ n, w })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("AB", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)1);      /* "ab" is one token */
        CHECK_EQ(ids[0], 128);
    }
    /* Prepend + Strip. */
    {
        VocabStep pre; pre.kind = RAD_NORM_PREPEND; pre.arg0 = "a";
        VocabStep w;   w.kind = RAD_PRE_WHITESPACE;
        Tokenizer t;
        CHECK_OK(t.init(make({ pre, w })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)1);
        CHECK_EQ(ids[0], 128);                /* "a" + "b" merged into "ab" */
    }
}

RAD_TEST_MAIN()

/* ================================================================== utf8_sanitize */

/* A byte-fallback vocab can emit a byte that no continuation completes, and the completion goes
 * into a JSON string, where one bad byte costs the client the whole response rather than one
 * character. These are the WHATWG cases: one U+FFFD per maximal subpart, well-formed input
 * untouched. */
TEST(utf8_sanitize_repairs_only_what_is_broken) {
    const std::string fffd = "\xEF\xBF\xBD";

    /* Untouched: ASCII, and every well-formed length. */
    CHECK_EQ(utf8_sanitize("hello"), std::string("hello"));
    CHECK_EQ(utf8_sanitize("caf\xC3\xA9 \xE6\x97\xA5 \xF0\x9F\x98\x80"),
             std::string("caf\xC3\xA9 \xE6\x97\xA5 \xF0\x9F\x98\x80"));
    CHECK_EQ(utf8_sanitize(""), std::string(""));

    /* 0xC1 is a lead for an encoding that would be overlong, so it is never legal however it
     * continues -- and a single one of them reaching a JSON string is a 500 for the client. */
    CHECK_EQ(utf8_sanitize("\xC1"), fffd);
    CHECK_EQ(utf8_sanitize("a\xC1\x61z"), "a" + fffd + "az");

    /* A lone continuation byte, and a lead byte that is simply not a lead. */
    CHECK_EQ(utf8_sanitize("\x80"), fffd);
    CHECK_EQ(utf8_sanitize("\xFF"), fffd);

    /* Truncated three-byte character: ONE replacement for the whole subpart, not two. */
    CHECK_EQ(utf8_sanitize("\xE6\x97"), fffd);
    CHECK_EQ(utf8_sanitize("\xE6\x97 "), fffd + " ");

    /* Overlong, surrogate and out-of-range encodings are rejected at the second byte, so the
     * lead is one subpart and the rest decodes on its own. */
    CHECK_EQ(utf8_sanitize("\xE0\x80\xAF"), fffd + fffd + fffd);  /* overlong '/' */
    CHECK_EQ(utf8_sanitize("\xED\xA0\x80"), fffd + fffd + fffd);  /* U+D800 */
    CHECK_EQ(utf8_sanitize("\xF4\x90\x80\x80"), fffd + fffd + fffd + fffd);  /* > U+10FFFF */
}

/* ================================================================== the ggml dtype map
 *
 * THE BLOCK QUANTS ARE DELIBERATELY ABSENT, and that absence is the property under test.
 * Reinterpreting a q4_K block as RAD_I4 produces a tensor that loads, computes, and is wrong --
 * which is the failure class this project exists to remove -- so rad-convert has to be told
 * "no equivalent" and dequantise or refuse, rather than being handed a plausible code.
 */
TEST(the_ggml_dtype_map_carries_only_the_types_that_mean_the_same_thing) {
    CHECK_EQ(gguf_to_rad_dtype(0),  (uint32_t)RAD_F32);
    CHECK_EQ(gguf_to_rad_dtype(1),  (uint32_t)RAD_F16);
    CHECK_EQ(gguf_to_rad_dtype(24), (uint32_t)RAD_I8);
    CHECK_EQ(gguf_to_rad_dtype(25), (uint32_t)RAD_I16);
    CHECK_EQ(gguf_to_rad_dtype(26), (uint32_t)RAD_I32);
    CHECK_EQ(gguf_to_rad_dtype(27), (uint32_t)RAD_I64);
    CHECK_EQ(gguf_to_rad_dtype(30), (uint32_t)RAD_BF16);

    /* q4_0, q8_0, q4_K, q6_K, and the iq family: every one of them is a BLOCK layout with its
       own scales, and none has a RAD_* equivalent. */
    for (uint32_t t : { 2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
                        16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 28u, 29u })
        CHECK_EQ(gguf_to_rad_dtype(t), (uint32_t)RAD_DT_INVALID);

    /* And a type code from a newer ggml than this build knows is invalid, not a guess. */
    CHECK_EQ(gguf_to_rad_dtype(9999), (uint32_t)RAD_DT_INVALID);

    /* THE TABLE BESIDE IT IS BOUNDS CHECKED, because a truncated or hostile GGUF names its own
       type codes and "SIGSEGV in the loader" is not a diagnosis. */
    CHECK(gguf_dtype(0) != nullptr);
    CHECK(gguf_dtype(9999) == nullptr);
    const GgufDtype* f32 = gguf_dtype(0);
    if (f32) {
        CHECK(f32->name && *f32->name);
        CHECK(f32->blck > 0);
        CHECK(f32->block_bytes > 0);
    }
    /* Every type the table does carry describes a non-degenerate block, or the element count
       arithmetic that reads it divides by zero. */
    for (uint32_t t = 0; t < 64; ++t) {
        const GgufDtype* d = gguf_dtype(t);
        if (!d || !d->name || !*d->name) continue;
        CHECK(d->blck > 0);
        CHECK(d->block_bytes > 0);
    }
}

/* ================================================================== the step chain's names
 *
 * WHAT A STEP PRINTS IS THE CONTRACT rad-convert WRITES AGAINST. A chain this build cannot
 * express must be refused by name rather than silently approximated (spec §12), and the name is
 * what the refusal carries -- so a step whose kind this build does not know has to say exactly
 * that, with the number, rather than falling through to a neighbouring step's description.
 */
TEST(every_step_kind_describes_itself_and_an_unknown_one_names_its_number) {
    VocabStep s;
    s.kind = RAD_NORM_NFC;
    CHECK_EQ(step_describe(s), std::string("NFC"));
    s.kind = RAD_NORM_NFKD;
    CHECK_EQ(step_describe(s), std::string("NFKD"));

    /* The arguments are in the description because they are what distinguishes two declarations
       of the same step -- a Replace chain is several rows differing only in their strings. */
    s.kind = RAD_NORM_REPLACE;
    s.flags = 0;
    s.arg0 = " ";
    s.arg1 = "_";
    const std::string rep = step_describe(s);
    CHECK(rep.find("Replace") != std::string::npos);
    CHECK(rep.find("literal") != std::string::npos);
    CHECK(rep.find("_") != std::string::npos);
    s.flags = 1;
    CHECK(step_describe(s).find("regex") != std::string::npos);

    s = VocabStep{};
    s.kind = RAD_PRE_BYTELEVEL;
    s.flags = 3;
    const std::string bl = step_describe(s);
    CHECK(bl.find("ByteLevel") != std::string::npos);
    CHECK(bl.find("add_prefix_space=1") != std::string::npos);
    CHECK(bl.find("use_regex=1") != std::string::npos);

    /* A DECODER STEP SAYS SO. The three chains are one list in declared order, so a reader who
       cannot tell a normaliser from a decoder cannot tell where the list turns over. */
    s = VocabStep{};
    s.kind = RAD_DEC_BYTELEVEL;
    CHECK(step_describe(s).rfind("decode", 0) == 0);
    s.kind = RAD_DEC_FUSE;
    CHECK(step_describe(s).rfind("decode", 0) == 0);

    /* An unexpressible step names its number, which is what makes the refusal actionable. */
    s = VocabStep{};
    s.kind = 9999;
    const std::string unk = step_describe(s);
    CHECK(unk.find("unknown") != std::string::npos);
    CHECK(unk.find("9999") != std::string::npos);
}

/* THE KIND RANGE IS WHICH OF THE THREE CHAINS A ROW BELONGS TO, and the three are stored as ONE
 * list in declared order so that no reader has to reassemble them. A boundary off by one puts a
 * pre-tokeniser in the normaliser chain, which changes token boundaries and nothing else. */
TEST(the_kind_ranges_partition_the_three_chains) {
    CHECK(step_is_norm(RAD_NORM_NFC));
    CHECK(step_is_norm(RAD_NORM_PREPEND));
    CHECK(!step_is_norm(0));                       /* 0 is the unset sentinel, not a normaliser */
    CHECK(!step_is_norm(RAD_PRE_BYTELEVEL));
    CHECK(!step_is_norm(RAD_DEC_BYTELEVEL));

    /* Every kind is in at most one chain. */
    for (uint32_t k : { (uint32_t)RAD_NORM_NFC, (uint32_t)RAD_NORM_LOWERCASE,
                        (uint32_t)RAD_PRE_BYTELEVEL, (uint32_t)RAD_PRE_METASPACE,
                        (uint32_t)RAD_DEC_BYTELEVEL, (uint32_t)RAD_DEC_FUSE }) {
        const int in = (step_is_norm(k) ? 1 : 0) + (step_is_pre(k) ? 1 : 0) +
                       (step_is_dec(k) ? 1 : 0);
        CHECK_EQ(in, 1);
    }
}

/* ================================================================== the normalisation tables
 *
 * The four public forms are covered above; these are the two primitives under them, and they are
 * worth pinning on their own because the tables they read are GENERATED -- a table regenerated
 * from a newer Unicode is exactly the kind of change that alters token boundaries for one script
 * and nothing else, which no round-trip test over ASCII would notice.
 */
TEST(the_combining_class_table_holds_the_values_canonical_ordering_depends_on) {
    /* Starters are class 0; the canonical ordering algorithm only reorders non-starters. */
    CHECK_EQ(unorm_ccc('a'), 0u);
    CHECK_EQ(unorm_ccc(0x00E9), 0u);               /* precomposed e-acute is itself a starter */
    /* Combining acute is 230 (above), cedilla is 202 (below) -- and the ORDER of those two
       numbers is what makes "e + cedilla + acute" and "e + acute + cedilla" normalise to the
       same string. */
    CHECK_EQ(unorm_ccc(0x0301), 230u);
    CHECK_EQ(unorm_ccc(0x0327), 202u);
    CHECK(unorm_ccc(0x0327) < unorm_ccc(0x0301));
    /* A codepoint past everything the table covers is a starter rather than a read past it. */
    CHECK_EQ(unorm_ccc(0x10FFFF), 0u);
    CHECK_EQ(unorm_ccc(0), 0u);
}

TEST(decompose_and_compose_are_inverses_over_the_canonical_forms) {
    /* e-acute decomposes to e + combining acute and composes back. */
    std::vector<uint32_t> v = { 0x00E9 };
    unorm_decompose(v, /*compat=*/false);
    CHECK_EQ(v.size(), 2u);
    CHECK_EQ(v[0], (uint32_t)'e');
    CHECK_EQ(v[1], 0x0301u);
    unorm_compose(v);
    CHECK_EQ(v.size(), 1u);
    CHECK_EQ(v[0], 0x00E9u);

    /* CANONICAL ORDERING IS PART OF DECOMPOSITION. The two accents come back in ccc order
       whichever order they went in, which is the whole reason NFD is a normal form. */
    std::vector<uint32_t> a = { 'e', 0x0301, 0x0327 };
    std::vector<uint32_t> b = { 'e', 0x0327, 0x0301 };
    unorm_decompose(a, false);
    unorm_decompose(b, false);
    CHECK(a == b);
    CHECK_EQ(a[1], 0x0327u);                       /* 202 before 230 */

    /* COMPATIBILITY DECOMPOSITION IS NOT CANONICAL DECOMPOSITION, and a K form that behaved like
       the plain one would silently stop folding the ligatures and full-width forms a tokeniser
       declaring NFKC is counting on. */
    std::vector<uint32_t> lig = { 0xFB01 };        /* the fi ligature */
    unorm_decompose(lig, /*compat=*/false);
    CHECK_EQ(lig.size(), 1u);                      /* it has no CANONICAL decomposition */
    std::vector<uint32_t> ligk = { 0xFB01 };
    unorm_decompose(ligk, /*compat=*/true);
    CHECK_EQ(ligk.size(), 2u);
    CHECK_EQ(ligk[0], (uint32_t)'f');
    CHECK_EQ(ligk[1], (uint32_t)'i');
    /* And composition does not put it back: NFKC is lossy by design. */
    unorm_compose(ligk);
    CHECK_EQ(ligk.size(), 2u);

    /* Plain ASCII is untouched by either, which is the path almost every token takes. */
    std::vector<uint32_t> ascii = { 'h', 'e', 'l', 'l', 'o' };
    const std::vector<uint32_t> before = ascii;
    unorm_decompose(ascii, false);
    unorm_compose(ascii);
    CHECK(ascii == before);

    /* Empty input is empty output rather than a read of element zero. */
    std::vector<uint32_t> empty;
    unorm_decompose(empty, true);
    unorm_compose(empty);
    CHECK_EQ(empty.size(), 0u);

    /* A LONE COMBINING MARK HAS NO STARTER TO ATTACH TO and must survive rather than being
       dropped or folded into whatever precedes it in the buffer. */
    std::vector<uint32_t> lone = { 0x0301 };
    unorm_compose(lone);
    CHECK_EQ(lone.size(), 1u);
    CHECK_EQ(lone[0], 0x0301u);
}
