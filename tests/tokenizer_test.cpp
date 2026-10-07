/* tokenizer_test.cpp -- the encoder against its reference, and the parts under it.
 *
 * THE REFERENCE is tools/tokref: an encoder built on llama.cpp's structural splitters, std::regex
 * for every other pattern and a string-keyed merge. It shares no code with what it checks. The
 * differential tests here hold the engine's splitter and encoder (`encode_new`) to it over tens of
 * thousands of random strings, and every place the two are ALLOWED to differ is a place the
 * reference is wrong about HuggingFace -- tokref.h lists them -- so each one has its own test
 * saying exactly what the difference is and that the engine's side is HuggingFace's answer.
 *
 * Then the parts: the merge on small vocabularies whose answers are worked out by hand, the linear
 * and heap merges against each other, special-token priority, the NFC shortcut against the full
 * normaliser, and the segment cache -- cold, warm, evicted, across a growing conversation, and
 * under concurrent encodes.
 */
#include "rad_test.h"
#include "tokenizer_fixture.h"
#include "tokref.h"

#include "rad_internal.h"
#include "text/tokenizer_bpe.h"
#include "text/tokenizer_nfc.h"
#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <thread>

using namespace rad;
using namespace tokfix;

namespace {

/* ------------------------------------------------------------------ helpers */

/* Compiled once per pattern: compiling is a load-time cost and the tests split thousands of
 * strings with each one. */
const TokRegex* compiled(const char* pattern) {
    static std::map<std::string, std::unique_ptr<TokRegex>> cache;
    auto it = cache.find(pattern);
    if (it == cache.end()) {
        std::unique_ptr<TokRegex> re;
        std::string why;
        TokRegex::compile(pattern, &re, &why);
        it = cache.emplace(pattern, std::move(re)).first;
    }
    return it->second.get();
}

std::vector<std::string> new_split(const char* pattern, const std::string& word) {
    const TokRegex* re = compiled(pattern);
    if (!re) return { "<refused>" };
    std::string canon;
    std::string_view v = word;
    if (!utf8_is_canonical(v)) { utf8_canonicalize(v, canon); v = canon; }
    std::vector<TokSpan> out, scratch;
    tok_split(*re, v, TOK_SPLIT_ISOLATED, false, out, scratch);
    std::vector<std::string> pieces;
    for (const TokSpan& s : out) pieces.emplace_back(v.substr(s.begin, s.end - s.begin));
    return pieces;
}

std::string escaped(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7F && c != '\\') o += (char)c;
        else o += fmt("\\x%02x", c);
    }
    return o;
}

std::string show(const std::vector<std::string>& v) {
    std::string o;
    for (const std::string& p : v) o += "[" + escaped(p) + "]";
    return o;
}

std::string show_ids(const std::vector<int32_t>& v, size_t from = 0, size_t n = 12) {
    std::string o;
    for (size_t i = from; i < v.size() && i < from + n; ++i) o += fmt("%d ", v[i]);
    return o;
}

/* Where llama.cpp's category tables (Unicode 15.1) and the compiler's (Oniguruma, 16.0) put a
 * codepoint in different categories. */
bool tables_disagree(uint32_t c) {
    const auto f = unicode_cpt_flags_from_cpt(c);
    const uint32_t g = gc_of(c);
    const bool L = g <= ucd::Lo, M = g >= ucd::Mn && g <= ucd::Me, N = g >= ucd::Nd && g <= ucd::No;
    const bool P = g >= ucd::Pc && g <= ucd::Po, S = g >= ucd::Sm && g <= ucd::So;
    return f.is_letter != L || f.is_accent_mark != M || f.is_number != N ||
           f.is_punctuation != P || f.is_symbol != S;
}

std::vector<int32_t> encode_new(const Tokenizer& t, const std::string& s, bool parse_special = true) {
    std::vector<int32_t> ids;
    t.encode(s, ids, false, parse_special);
    return ids;
}

std::vector<int32_t> encode_ref(const Vocab& v, const std::string& s, bool parse_special = true) {
    std::vector<int32_t> ids;
    tokref::encode(v, s, ids, false, parse_special);
    return ids;
}

/* Compare the two encoders over `n` texts from `gen`; report the first few differences. */
template <class Gen>
int compare_encoders(const char* what, std::shared_ptr<const Vocab> v, int n, Gen gen,
                     bool parse_special = true) {
    Tokenizer t;
    t.init(v);
    t.set_segment_cache(false);
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const std::string s = gen(i);
        const std::vector<int32_t> a = encode_new(t, s, parse_special), b = encode_ref(*v, s, parse_special);
        if (a == b) continue;
        if (++bad <= 3) {
            size_t k = 0;
            while (k < a.size() && k < b.size() && a[k] == b[k]) ++k;
            std::fprintf(stderr, "    %s: differs at id %zu for '%s'\n      new %s\n      ref %s\n", what, k,
                         escaped(s.substr(0, 200)).c_str(), show_ids(a, k > 3 ? k - 3 : 0).c_str(),
                         show_ids(b, k > 3 ? k - 3 : 0).c_str());
        }
    }
    return bad;
}

/* The corpus the trained vocabularies learn from: code, prose, every script. */
std::string training_corpus() {
    TextGen g(7);
    std::string s;
    for (int i = 0; i < 400; ++i) s += g.text(60) + "\n";
    return s;
}

const VocabBuild& trained() {
    static const VocabBuild vb = train_bytelevel(training_corpus(), k_qwen35, 1200);
    return vb;
}

}  /* namespace */

/* ================================================================== splits against the reference */

/* THE SPLITTER AGAINST THE REFERENCE, on every pattern the reference implements: its structural
 * splitters (GPT-2, Llama-3, Qwen2, Qwen3.5, in both spellings llama.cpp keys them on,
 * and "[^\n]+|[\n]+") and patterns it runs through std::regex (MiniCPM5's two, the GGUF defaults,
 * the gpt-4o adaptation). The text has every general category, marks and ZWJ sequences, CJK,
 * Hangul and RTL, digit runs of every length, every kind of whitespace including CRLF and the
 * Unicode spaces, contractions in mixed case, and bytes that are not UTF-8.
 *
 * Two inputs are held out, each with its own test below: the codepoints the two category tables
 * disagree on, and -- for the `(?i)` spelling only -- an apostrophe before U+017F. */
TEST(splits_match_the_reference_splitters) {
    struct P { const char* pattern; bool ci; };
    const P patterns[] = {
        { "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)", false },
        { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", false },
        { k_llama3, true },
        { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", false },
        { "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", true },
        { "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", false },
        { k_qwen35, true },
        { "[^\\n]+|[\\n]+", false },
        { k_minicpm_digits, false },
        { k_minicpm, true },
        { "[\\p{P}\\$\\+<=>\\^~\\|]+", false },
        { "\\p{N}+", false },
        { "[0-9][0-9][0-9]", false },
        { "\\S+", false },
        { "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", false },
    };
    const std::string long_s = "'\xc5\xbf";
    size_t compared = 0;
    for (const P& p : patterns) {
        TextGen g(1234);
        int bad = 0;
        /* Few long strings rather than many short ones: the reference builds a std::regex per
         * call for the patterns it has no splitter for. */
        for (int i = 0; i < 160; ++i) {
            const std::string s = g.text(1 + g.pick(i < 80 ? 40 : 1200), /*junk=*/true, tables_disagree);
            if (p.ci && s.find(long_s) != std::string::npos) continue;
            const std::vector<std::string> a = new_split(p.pattern, s), b = tokref::split(p.pattern, s);
            compared += s.size();
            if (a == b) continue;
            if (++bad <= 3)
                std::fprintf(stderr, "    /%.60s/ on '%s'\n      new %s\n      ref %s\n", p.pattern,
                             escaped(s.substr(0, 300)).c_str(), show(a).c_str(), show(b).c_str());
        }
        CHECK_EQ(bad, 0);
    }
    CHECK(compared > (size_t)15 << 20 >> 4);
}

/* Digit runs of every length and whitespace of every shape, exhaustively, since the structural
 * splitters special-case both. */
TEST(splits_match_the_reference_on_every_digit_run_and_whitespace_run) {
    const char* patterns[] = { k_llama3, k_qwen35,
                               "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" };
    const char* ws[] = { " ", "\t", "\n", "\r", "\r\n", "\xc2\xa0", "\xe3\x80\x80", "\x0b", "\xe2\x80\xa8" };
    int bad = 0;
    for (const char* p : patterns) {
        for (int n = 1; n <= 24; ++n) {
            for (const char* before : { "", "a", " ", "x ", "!" }) {
                for (const char* after : { "", "a", " ", "\n", "," }) {
                    const std::string s = std::string(before) + std::string((size_t)n, '7') + after;
                    if (new_split(p, s) != tokref::split(p, s)) ++bad;
                }
            }
        }
        for (int mask = 1; mask < 4096; mask += 7) {
            std::string s = "w";
            for (int k = 0, m = mask; k < 4; ++k, m /= 9) s += ws[m % 9];
            for (const char* after : { "", "x", "!", "1", "\n" }) {
                const std::string t = s + after;
                if (new_split(p, t) != tokref::split(p, t)) ++bad;
            }
        }
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== where the reference is wrong */

/* THE REFERENCE REWRITES `(?i:'s|...)` INTO `'[sS]|...`, which loses U+017F: Oniguruma folds
 * LATIN SMALL LETTER LONG S to 's', so "'ſ" is a contraction to HuggingFace. The pieces below are
 * HuggingFace's. Only the (?i) spelling is affected; llama.cpp's expanded one has no ſ in it. */
TEST(long_s_is_a_contraction_under_case_folding_as_in_huggingface) {
    const std::string s = "x'\xc5\xbfy it'S";
    const std::vector<std::string> want = { "x", "'\xc5\xbf", "y", " it", "'S" };
    CHECK(new_split(k_qwen35, s) == want);
    CHECK(tokref::split(k_qwen35, s) != want);
    const char* expanded =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    CHECK(new_split(expanded, s) == tokref::split(expanded, s));
}

/* THE TWO CATEGORY TABLES DIFFER BY UNICODE 16.0 AND BY NOTHING ELSE: every codepoint they
 * disagree on is unassigned in llama.cpp's and assigned in Oniguruma's, and there are 5185 of
 * them. A split over one of them follows Oniguruma's category, as HuggingFace does -- U+10D40 GARAY
 * DIGIT ZERO is a digit and U+10D6E GARAY HYPHEN is punctuation. */
TEST(the_category_tables_differ_only_by_what_unicode_16_assigned) {
    size_t n = 0, bad = 0;
    for (uint32_t c = 0; c < 0x110000; ++c) {
        if (c >= 0xD800 && c <= 0xDFFF) continue;
        if (!tables_disagree(c)) continue;
        ++n;
        if (!unicode_cpt_flags_from_cpt(c).is_undefined || gc_of(c) == ucd::Cn) ++bad;
    }
    CHECK_EQ(n, (size_t)5185);
    CHECK_EQ(bad, (size_t)0);
    const std::string s = "a\xf0\x90\xb5\x80" "b!\xf0\x90\xb5\xae.";
    const std::vector<std::string> want = { "a", "\xf0\x90\xb5\x80", "b", "!\xf0\x90\xb5\xae." };
    CHECK(new_split(k_qwen35, s) == want);
    CHECK(tokref::split(k_qwen35, s) != want);
}

/* `\p{Han}+` IS NOT KIMI-K2's PATTERN, it is llama.cpp's key for a splitter approximating it.
 * The GGUF table carries the real pattern, whose letter classes split camel case. */
TEST(the_kimi_k2_entry_is_the_real_pattern) {
    const char* k2 =
        "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    const std::string s = "HelloWorld \xe4\xb8\xad\xe6\x96\x87" "ABc";
    const std::vector<std::string> want = { "Hello", "World", " ", "\xe4\xb8\xad\xe6\x96\x87", "ABc" };
    CHECK(new_split(k2, s) == want);
    CHECK(tokref::split("\\p{Han}+", s) != want);
}

/* A digit-grouping lookahead is refused rather than run through a rule that means something
 * else: llama.cpp's AFMoE splitter splits "12345abc", the pattern does not. */
TEST(a_wide_lookahead_is_refused_where_the_reference_approximated_it) {
    std::unique_ptr<TokRegex> re;
    std::string why;
    CHECK_EQ(TokRegex::compile("\\d{1,3}(?=(?:\\d{3})*\\b)", &re, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("lookaround") != std::string::npos);
    VocabBuild vb = trained();
    vb.steps = { split_step("\\d{1,3}(?=(?:\\d{3})*\\b)"), bytelevel_step() };
    auto v = std::make_shared<Vocab>();
    CHECK_EQ(v->load(std::move(vb)), RAD_E_UNSUPPORTED);
    CHECK(!v->ready());
}

/* ================================================================== BPE by hand */

namespace {

/* A byte-level vocabulary small enough to work every answer out on paper. "c c" merges to text
 * that is not a token, which blocks "c ab" -> "cab" behind it; "Ġab" is a user-defined token the
 * raw text can only reach through the byte-level mapping. */
const char* k_tiny = R"JSON({
  "added_tokens": [ {"id": 12, "content": "<|end|>", "special": true},
                    {"id": 13, "content": "Ġab", "special": false} ],
  "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": " ?\\p{L}+|\\p{N}|\\s+|[^\\s\\p{L}\\p{N}]+"}, "behavior": "Isolated"},
      {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false} ] },
  "decoder": { "type": "ByteLevel" },
  "model": { "type": "BPE",
    "vocab": {"a": 0, "b": 1, "c": 2, "Ġ": 3, "ab": 4, "bc": 5, "abc": 6, "Ġa": 7, "aa": 8, "1": 9,
              "cab": 10, "x": 11},
    "merges": ["a b", "b c", "ab c", "Ġ a", "a a", "c c", "c ab"] }
})JSON";

/* SentencePiece-shaped: Metaspace, codepoint symbols, "<0xNN>" byte fallback. */
const char* k_spm = R"JSON({
  "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false },
  "decoder": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
  "model": { "type": "BPE", "byte_fallback": true,
    "vocab": {"<unk>": 0, "▁": 1, "h": 2, "i": 3, "▁h": 4, "▁hi": 5, "<0xE2>": 6, "<0x82>": 7, "<0xAC>": 8},
    "merges": ["▁ h", "▁h i"] }
})JSON";

std::shared_ptr<const Vocab> from_json(const char* js, int32_t unk = -1) {
    VocabBuild vb;
    if (parse_tokenizer_json_buf(js, "test.json", vb) < 0) return nullptr;
    if (unk >= 0) vb.unk = unk;
    return load(std::move(vb));
}

}  /* namespace */

TEST(bpe_merges_as_worked_out_by_hand) {
    auto v = from_json(k_tiny);
    REQUIRE(v != nullptr);
    Tokenizer t;
    CHECK_OK(t.init(v));
    t.set_segment_cache(false);
    const struct { const char* text; bool parse_special; std::vector<int32_t> ids; } cases[] = {
        { "abc",       true,  { 6 } },          /* a b -> ab (rank 0), ab c -> abc (2) */
        { "bcab",      true,  { 5, 4 } },       /* a b first (0), then b c (1); bc ab has no rank */
        { "aaa",       true,  { 8, 0 } },       /* two equal ranks: the leftmost pair merges */
        { "cc",        true,  { 2, 2 } },       /* merged into text that is no token: spelled by byte */
        { "ccab",      true,  { 2, 2, 4 } },    /* and that merge (5) blocks c ab (6) */
        { " a",        true,  { 7 } },
        { " ab",       true,  { 13 } },         /* "Ġab" is user-defined: taken whole */
        { "abx",       true,  { 4, 11 } },
        { "1a",        true,  { 9, 0 } },
        { "a<|end|>b", true,  { 0, 12, 1 } },
        { "a<|end|>b", false, { 0, 1 } },       /* not parsed: its bytes have no tokens, dropped */
    };
    for (const auto& c : cases) {
        const std::vector<int32_t> got = encode_new(t, c.text, c.parse_special);
        CHECK(got == c.ids);
        CHECK(encode_ref(*v, c.text, c.parse_special) == c.ids);
        if (got != c.ids) std::fprintf(stderr, "    '%s': %s\n", c.text, show_ids(got).c_str());
    }
}

TEST(bpe_ignore_merges_takes_whole_pretokens_and_keeps_the_default_merge_order) {
    /* "abc" exists, but the only merge stops at "ab". Normalization and pre-tokenization
     * must still run before the optional whole-token lookup. Exercise both ByteLevel paths. */
    const char* chains[] = {
        R"({"type":"WhitespaceSplit"})",
        R"({"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]})",
        R"({"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false},{"type":"Split","pattern":{"String":" "},"behavior":"Removed"}]})"
    };
    for (const char* chain : chains)
        for (const char* flag : {"", "\"ignore_merges\":false,", "\"ignore_merges\":true,"}) {
            const bool ignore = std::strstr(flag, "true") != nullptr;
            const std::string js = std::string("{\"normalizer\":{\"type\":\"Lowercase\"},\"pre_tokenizer\":") +
                chain + ",\"model\":{\"type\":\"BPE\"," + flag +
                R"("vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4,"abc abc":5},"merges":["a b"]}})";
            auto v = from_json(js.c_str());
            REQUIRE(v != nullptr);
            CHECK_EQ(v->build().ignore_merges, ignore);
            Tokenizer t;
            CHECK_OK(t.init(v));
            for (int repeat = 0; repeat < 2; ++repeat) {
                CHECK(encode_new(t, "ABC") == (ignore ? std::vector<int32_t>{4} : std::vector<int32_t>{3, 2}));
                CHECK(encode_new(t, "ABCABC") == (std::vector<int32_t>{3, 2, 3, 2}));
                CHECK(encode_new(t, "ABC ABC") == (ignore ? std::vector<int32_t>{4, 4} : std::vector<int32_t>{3, 2, 3, 2}));
            }
        }
}

TEST(bpe_ignore_merges_does_not_enable_disabled_control_tokens) {
    auto v = from_json(R"JSON({
      "added_tokens":[{"id":4,"content":"abc","special":true}],
      "pre_tokenizer":{"type":"WhitespaceSplit"},
      "model":{"type":"BPE","ignore_merges":true,
               "vocab":{"a":0,"b":1,"c":2,"ab":3},"merges":["a b"]}
    })JSON");
    REQUIRE(v != nullptr);
    Tokenizer t;
    CHECK_OK(t.init(v));
    CHECK(encode_new(t, "abc", true) == (std::vector<int32_t>{4}));
    CHECK(encode_new(t, "abc", false) == (std::vector<int32_t>{3, 2}));
}

TEST(byte_fallback_and_unk_follow_the_declared_chain) {
    auto v = from_json(k_spm);
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    CHECK(encode_new(t, "hi") == (std::vector<int32_t>{ 5 }));
    CHECK(encode_new(t, "hi\xe2\x82\xac") == (std::vector<int32_t>{ 5, 6, 7, 8 }));   /* the euro sign, by byte */
    CHECK(encode_new(t, "hix") == (std::vector<int32_t>{ 5 }));                       /* no <0x78>, no unk: dropped */
    auto u = from_json(k_spm, 0);
    REQUIRE(u != nullptr);
    Tokenizer tu;
    tu.init(u);
    CHECK(encode_new(tu, "hix") == (std::vector<int32_t>{ 5, 0 }));                   /* no <0x78>: unk */
    for (const char* s : { "hi", "hi\xe2\x82\xac", "hix", "h i  hi", "\xe2\x82\xac\xe2\x82\xac" }) {
        CHECK(encode_new(t, s) == encode_ref(*v, s));
        CHECK(encode_new(tu, s) == encode_ref(*u, s));
    }
}

/* THE LINEAR AND HEAP MERGES PICK THE SAME PAIR AT EVERY STEP: lowest rank, then leftmost. Held
 * to each other on random words over a random merge table dense enough that most words merge
 * many times, with results that are and are not tokens. */
TEST(linear_and_heap_merges_agree) {
    std::mt19937 rng(99);
    MergeTable mt;
    const int n_ids = 24;
    for (uint32_t r = 0; r < 300; ++r) {
        const int32_t l = (int32_t)(rng() % n_ids), rr = (int32_t)(rng() % n_ids);
        const int32_t merged = rng() % 5 == 0 ? -1 : (int32_t)(rng() % n_ids);
        mt.insert(l, rr, r, merged);
    }
    int bad = 0;
    for (int i = 0; i < 20000; ++i) {
        std::vector<BpeSym> a;
        const size_t n = 1 + rng() % kBpeLinearMax;
        for (size_t k = 0; k < n; ++k) a.push_back({ (uint32_t)k, 1, (int32_t)(rng() % n_ids) });
        std::vector<BpeSym> b = a;
        bpe_merge(a, mt, false);
        bpe_merge(b, mt, true);
        bool same = a.size() == b.size();
        for (size_t k = 0; same && k < a.size(); ++k)
            same = a[k].start == b[k].start && a[k].len == b[k].len && a[k].id == b[k].id;
        if (!same) ++bad;
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the whole encoder against the reference */

/* Trained vocabularies, every chain shape the encoder takes a different route through: the
 * byte-level fast path (Split then ByteLevel, merged on raw bytes), two Splits, ByteLevel with
 * its own regex and prefix space, ByteLevel before a Split (the mapped text is split and merged),
 * NFC, special and user-defined tokens including one that is a prefix of another, SentencePiece's
 * Metaspace with byte fallback, WordPiece and Unigram. */
TEST(the_encoder_matches_the_reference_on_trained_vocabularies) {
    const std::vector<std::pair<std::string, uint8_t>> specials = {
        { "<|im_start|>", RAD_TT_CONTROL }, { "<|im_end|>", RAD_TT_CONTROL }, { "<|im", RAD_TT_CONTROL },
        { "XYZ", RAD_TT_CONTROL }, { "YZWV", RAD_TT_CONTROL }, { "<think>", RAD_TT_USER_DEFINED },
        { "</think>", RAD_TT_USER_DEFINED }, { "Wv", RAD_TT_USER_DEFINED },
    };
    auto with = [&](std::vector<VocabStep> steps, bool add_sp = true) {
        VocabBuild vb = trained();
        vb.steps = std::move(steps);
        if (add_sp) add_specials(vb, specials);
        return load(std::move(vb));
    };
    auto gen_for = [&](uint32_t seed, bool no_long_s) {
        auto g = std::make_shared<TextGen>(seed);
        return [g, no_long_s](int) {
            std::string s;
            const int parts = 1 + g->pick(5);
            for (int k = 0; k < parts; ++k) {
                s += g->text(g->pick(60), true, tables_disagree);
                static const char* inject[] = { "<|im_start|>", "<|im_end|>", "<|im", "XYZWV", "XYZ",
                                                "YZWV", "<think>", "</think>", "Wv", "WWv" };
                if (g->pick(2)) s += inject[g->pick(10)];
            }
            if (no_long_s) {
                for (size_t p; (p = s.find("'\xc5\xbf")) != std::string::npos; ) s.replace(p, 1, "_");
            }
            return s;
        };
    };
    VocabStep nfc = simple_step(RAD_NORM_NFC);
    struct Chain { const char* name; std::shared_ptr<const Vocab> v; bool ci; };
    const Chain chains[] = {
        { "split+bytelevel",     with({ split_step(k_qwen35), bytelevel_step() }), true },
        { "nfc+split+bytelevel", with({ nfc, split_step(k_qwen35), bytelevel_step() }), true },
        { "two splits",          with({ split_step(k_minicpm_digits), split_step(k_minicpm), bytelevel_step() }), true },
        { "bytelevel regex",     with({ bytelevel_step(true, true) }), false },
        { "bytelevel then split", with({ bytelevel_step(), split_step("\\S+|\\s+") }), false },
        { "gpt2 without \\s+",    with({ split_step("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"), bytelevel_step() }), false },
    };
    uint32_t seed = 500;
    for (const Chain& c : chains) {
        REQUIRE(c.v != nullptr);
        for (bool ps : { true, false })
            CHECK_EQ(compare_encoders(c.name, c.v, 500, gen_for(seed++, c.ci), ps), 0);
    }

    /* SentencePiece-shaped: codepoint symbols and byte fallback, no byte-level mapping. */
    {
        VocabBuild vb;
        vb.kind = RAD_TOK_BPE;
        for (int b = 0; b < 256; ++b) vb.tokens.push_back(fmt("<0x%02X>", b));
        vb.types.assign(256, (uint8_t)RAD_TT_BYTE);
        const std::vector<std::string> base = { "\xe2\x96\x81", "a", "b", "c", "d", "e", "h", "i",
                                                "l", "o", "t", "\xe4\xb8\x96", "\xe7\x95\x8c" };
        for (const std::string& t : base) { vb.tokens.push_back(t); vb.types.push_back(RAD_TT_NORMAL); }
        auto id = [&](const std::string& t) {
            for (size_t i = 0; i < vb.tokens.size(); ++i) if (vb.tokens[i] == t) return (uint32_t)i;
            return UINT32_MAX;
        };
        const std::vector<std::pair<std::string, std::string>> merges = {
            { "\xe2\x96\x81", "t" }, { "h", "e" }, { "\xe2\x96\x81t", "he" }, { "l", "l" },
            { "ll", "o" }, { "e", "llo" }, { "h", "ello" }, { "\xe2\x96\x81", "hello" },
            { "\xe4\xb8\x96", "\xe7\x95\x8c" }, { "a", "b" }, { "ab", "c" }, { "\xe2\x96\x81", "abc" },
        };
        for (const auto& [l, r] : merges) {
            if (id(l + r) == UINT32_MAX) { vb.tokens.push_back(l + r); vb.types.push_back(RAD_TT_NORMAL); }
            vb.merges.push_back({ id(l), id(r) });
        }
        VocabStep ms;
        ms.kind = RAD_PRE_METASPACE;
        ms.arg0 = "\xe2\x96\x81";
        ms.iarg = 2;
        vb.steps = { ms, simple_step(RAD_PRE_BYTE_FALLBACK) };
        add_specials(vb, specials);
        auto v = load(std::move(vb));
        REQUIRE(v != nullptr);
        CHECK_EQ(compare_encoders("spm", v, 800, gen_for(seed++, false)), 0);
    }

    /* WordPiece and Unigram over the trained vocabulary's pieces. */
    {
        VocabBuild vb;
        vb.kind = RAD_TOK_WORDPIECE;
        vb.tokens = { "[UNK]", "the", "cache", "##s", "##ing", "sh", "##ard", "val", "##ue", "a", "##b",
                      "\xe4\xb8\x96", "##\xe7\x95\x8c", ",", "." };
        vb.unk = 0;
        vb.steps = { simple_step(RAD_NORM_LOWERCASE), simple_step(RAD_PRE_WHITESPACE) };
        auto v = load(std::move(vb));
        REQUIRE(v != nullptr);
        CHECK_EQ(compare_encoders("wordpiece", v, 800, gen_for(seed++, false)), 0);
    }
    {
        VocabBuild vb = trained();
        vb.kind = RAD_TOK_UNIGRAM;
        vb.merges.clear();
        vb.scores.resize(vb.tokens.size());
        for (size_t i = 0; i < vb.tokens.size(); ++i) vb.scores[i] = -1.0f - 0.001f * (float)(i % 997);
        vb.tokens.push_back("<unk>");
        vb.types.push_back(RAD_TT_UNKNOWN);
        vb.scores.push_back(0.0f);
        vb.unk = (int32_t)vb.tokens.size() - 1;
        VocabStep ms;
        ms.kind = RAD_PRE_METASPACE;
        ms.arg0 = "\xe2\x96\x81";
        ms.iarg = 1;
        ms.flags = 1;
        vb.steps = { simple_step(RAD_NORM_STRIP), ms };
        auto v = load(std::move(vb));
        REQUIRE(v != nullptr);
        CHECK_EQ(compare_encoders("unigram", v, 400, gen_for(seed++, false)), 0);
    }
}

/* A TEXT THAT OPENS WITH BOS GETS NO SECOND ONE. A client rendering its own chat template writes
 * the BOS into the prompt; one the text spells anywhere else, or a second one at the start, is
 * the client's and stays. */
TEST(a_text_that_opens_with_bos_gets_no_second_one) {
    VocabBuild vb;
    REQUIRE(parse_tokenizer_json_buf(k_tiny, "test.json", vb) >= 0);
    vb.bos = 12;
    vb.add_bos = true;
    auto v = load(std::move(vb));
    REQUIRE(v != nullptr);
    Tokenizer t;
    CHECK_OK(t.init(v));
    auto enc = [&](const char* s, bool add_special, bool parse_special) {
        std::vector<int32_t> ids;
        t.encode(s, ids, add_special, parse_special);
        return ids;
    };
    CHECK(enc("ab", true, true) == (std::vector<int32_t>{ 12, 4 }));
    CHECK(enc("<|end|>ab", true, true) == (std::vector<int32_t>{ 12, 4 }));
    CHECK(enc("<|end|><|end|>ab", true, true) == (std::vector<int32_t>{ 12, 12, 4 }));
    CHECK(enc("ab<|end|>", true, true) == (std::vector<int32_t>{ 12, 4, 12 }));
    CHECK(enc("<|end|>ab", false, true) == (std::vector<int32_t>{ 12, 4 }));
    /* Not parsed, the spelling is not the token, so the BOS is still owed. */
    CHECK(enc("<|end|>x", true, false) == (std::vector<int32_t>{ 12, 11 }));
}

/* SPECIAL TOKENS ARE CLAIMED LONGEST FIRST, not leftmost: "bcde" wins over "ab" in "abcde" even
 * though "ab" starts earlier, and "<|im_start|>" is never cut by its prefix "<|im". */
TEST(special_tokens_are_claimed_by_priority) {
    VocabBuild vb = trained();
    vb.steps = { split_step(k_qwen35), bytelevel_step() };
    const int32_t base = (int32_t)vb.tokens.size();
    add_specials(vb, { { "ab", RAD_TT_CONTROL }, { "bcde", RAD_TT_CONTROL },
                       { "<|im", RAD_TT_CONTROL }, { "<|im_start|>", RAD_TT_CONTROL } });
    auto v = load(std::move(vb));
    REQUIRE(v != nullptr);
    Tokenizer t;
    t.init(v);
    const std::vector<int32_t> ids = encode_new(t, "abcde");
    REQUIRE(ids.size() == 2);
    CHECK_EQ(ids[1], base + 1);
    CHECK(ids == encode_ref(*v, "abcde"));
    CHECK(encode_new(t, "x<|im_start|>y<|im") == encode_ref(*v, "x<|im_start|>y<|im"));
    const std::vector<int32_t> im = encode_new(t, "<|im_start|>");
    CHECK(im == (std::vector<int32_t>{ base + 3 }));
    /* Not parsed, a control token is text. */
    CHECK(encode_new(t, "<|im_start|>", false) != im);
    CHECK(encode_new(t, "<|im_start|>", false) == encode_ref(*v, "<|im_start|>", false));
}

/* ================================================================== NFC */

/* NORMALISING ONLY AROUND WHAT CAN CHANGE IS NORMALISING THE WHOLE TEXT. Every codepoint on its
 * own, then random runs of starters, marks in and out of canonical order, Hangul jamo, composed
 * and decomposed forms, and bytes that are not UTF-8. */
TEST(nfc_between_boundaries_is_the_whole_text_normalised) {
    size_t bad = 0;
    for (uint32_t c = 0; c < 0x110000; ++c) {
        if (c >= 0xD800 && c <= 0xDFFF) continue;
        std::string s = cp_utf8(c);
        const std::string want = unorm_utf8(s, false, true);
        tok_nfc(s);
        if (s != want) ++bad;
    }
    CHECK_EQ(bad, (size_t)0);

    static const char* parts[] = {
        "a", "e", "o", "A", " ", "x", "\xcc\x81", "\xcc\x80", "\xcc\x88", "\xcc\xa3", "\xcc\x9b",
        "\xcd\x85", "\xe0\xb9\x88", "\xd6\xb0", "\xe1\x84\x80", "\xe1\x85\xa1", "\xe1\x86\xa8",
        "\xea\xb0\x80", "\xc3\xa9", "\xc3\x85", "\xe2\x84\xab", "\xef\xac\x81", "\xe2\x80\x8d",
        "\xf0\x90\xbd\x8a", "\xe0\xbb\x8b", "\xe0\xbd\xb3", "\xcd\x84", "\xe4\xb8\x96", "\xc3",
        "\x80", "\xed\xa0\x80", "\xc0\x80", "\xf4\x90\x80\x80", "abcdefgh",
    };
    std::mt19937 rng(5);
    for (int i = 0; i < 100000; ++i) {
        std::string s;
        const int n = 1 + rng() % 12;
        for (int k = 0; k < n; ++k) s += parts[rng() % (sizeof(parts) / sizeof(parts[0]))];
        const std::string want = unorm_utf8(s, false, true);
        std::string got = s;
        const bool touched = tok_nfc(got);
        /* Untouched means already normalised; touched text may still come out as it went in. */
        if (got != want || (!touched && want != s)) ++bad;
    }
    CHECK_EQ(bad, (size_t)0);
}

/* ================================================================== the segment cache */

namespace {

std::shared_ptr<const Vocab> chat_vocab() {
    static std::shared_ptr<const Vocab> v = [] {
        VocabBuild vb = trained();
        vb.steps = { simple_step(RAD_NORM_NFC), split_step(k_qwen35), bytelevel_step() };
        add_specials(vb, { { "<|im_start|>", RAD_TT_CONTROL }, { "<|im_end|>", RAD_TT_CONTROL },
                           { "<think>", RAD_TT_USER_DEFINED } });
        return load(std::move(vb));
    }();
    return v;
}

/* A conversation: every turn is every message so far, framed as a chat template frames them. */
std::vector<std::string> conversation(uint32_t seed, int turns) {
    TextGen g(seed);
    std::vector<std::string> prompts;
    std::string conv = "<|im_start|>system\nYou are a coding agent.<|im_end|>\n";
    for (int k = 0; k < turns; ++k) {
        const char* role = k % 2 ? "assistant" : "user";
        conv += std::string("<|im_start|>") + role + "\n" + (k % 3 == 1 ? "<think>\n" : "") +
                g.text(50 + g.pick(400)) + "<|im_end|>\n";
        prompts.push_back(conv + "<|im_start|>assistant\n");
    }
    return prompts;
}

}  /* namespace */

TEST(the_cache_returns_what_encoding_returns_cold_warm_and_evicted) {
    auto v = chat_vocab();
    REQUIRE(v != nullptr);
    Tokenizer plain, cached;
    plain.init(v);
    plain.set_segment_cache(false);
    cached.init(v);
    cached.clear_cache();
    cached.set_cache_limit(Tokenizer::kSegmentCacheDefault);

    const std::vector<std::string> prompts = conversation(11, 12);
    const std::string& last = prompts.back();
    const std::vector<int32_t> want = encode_new(plain, last);
    CHECK(encode_new(cached, last) == want);                     /* cold */
    const Tokenizer::CacheStats cold = cached.cache_stats();
    CHECK(cold.entries > 0);
    CHECK(encode_new(cached, last) == want);                     /* warm */
    const Tokenizer::CacheStats warm = cached.cache_stats();
    CHECK(warm.hits >= cold.entries);
    CHECK(warm.bytes <= warm.limit);

    /* A bound smaller than the conversation evicts, and what is left still answers exactly. */
    cached.set_cache_limit(cold.bytes / 3);
    const Tokenizer::CacheStats small = cached.cache_stats();
    CHECK(small.bytes <= small.limit);
    CHECK(small.entries < cold.entries);
    for (int k = 0; k < 3; ++k) CHECK(encode_new(cached, last) == want);
    CHECK(cached.cache_stats().bytes <= cold.bytes / 3);

    /* No room at all: every segment is encoded, nothing is kept. */
    cached.set_cache_limit(0);
    CHECK_EQ(cached.cache_stats().entries, (size_t)0);
    CHECK(encode_new(cached, last) == want);
    CHECK_EQ(cached.cache_stats().entries, (size_t)0);
    cached.set_cache_limit(Tokenizer::kSegmentCacheDefault);
}

/* A GROWING CONVERSATION costs its new turn: every earlier message is a hit, and the ids of every
 * turn are the uncached ids. */
TEST(the_cache_serves_a_growing_conversation_exactly) {
    auto v = chat_vocab();
    REQUIRE(v != nullptr);
    Tokenizer plain, cached;
    plain.init(v);
    plain.set_segment_cache(false);
    cached.init(v);
    cached.clear_cache();
    const std::vector<std::string> prompts = conversation(21, 30);
    uint64_t prev_hits = 0;
    for (size_t k = 0; k < prompts.size(); ++k) {
        CHECK(encode_new(cached, prompts[k]) == encode_new(plain, prompts[k]));
        const Tokenizer::CacheStats s = cached.cache_stats();
        /* Turn k re-sends k earlier messages; each long enough to cache is a hit. */
        if (k >= 2) CHECK(s.hits - prev_hits >= k / 2);
        prev_hits = s.hits;
    }
    /* A segment under the minimum is never cached. */
    cached.clear_cache();
    encode_new(cached, "<|im_start|>user\nhi<|im_end|>");
    CHECK_EQ(cached.cache_stats().entries, (size_t)0);
}

/* CONCURRENT ENCODES, sharing prefixes and evicting under each other, each get the uncached
 * answer. */
TEST(the_cache_is_exact_under_concurrent_encodes) {
    auto v = chat_vocab();
    REQUIRE(v != nullptr);
    Tokenizer plain;
    plain.init(v);
    plain.set_segment_cache(false);
    std::vector<std::vector<std::string>> convs;
    std::vector<std::vector<std::vector<int32_t>>> want;
    for (uint32_t c = 0; c < 4; ++c) {
        convs.push_back(conversation(100 + c, 16));
        want.emplace_back();
        for (const std::string& p : convs.back()) want.back().push_back(encode_new(plain, p));
    }
    Tokenizer shared;
    shared.init(v);
    shared.clear_cache();
    /* Small enough that the threads evict each other's segments. */
    shared.set_cache_limit(64 << 10);
    std::atomic<int> bad{ 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            const size_t c = (size_t)t % convs.size();
            for (int round = 0; round < 3; ++round)
                for (size_t k = 0; k < convs[c].size(); ++k)
                    if (encode_new(shared, convs[c][k]) != want[c][k]) ++bad;
        });
    }
    for (std::thread& th : threads) th.join();
    CHECK_EQ(bad.load(), 0);
    const Tokenizer::CacheStats s = shared.cache_stats();
    CHECK(s.bytes <= s.limit);
    CHECK(s.hits > 0);
    shared.set_cache_limit(Tokenizer::kSegmentCacheDefault);
}

RAD_TEST_MAIN()
