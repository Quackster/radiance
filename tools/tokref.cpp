/* tokref.cpp -- a reference encoder: llama.cpp's splitters, std::regex and a string-keyed merge.
 *
 * A REFERENCE, not a fallback: core/ does not link it. It runs a `Split` through
 * llama.cpp's unicode_regex_split -- a hand-written splitter for the patterns llama.cpp knows, and
 * std::regex over a category-collapsed copy of the text for every other one -- and it merges with
 * a string-keyed bigram queue. That is an implementation independent of everything in
 * core/text/tokenizer_regex.cpp and tokenizer_bpe.cpp, which is what makes it worth comparing
 * against: tokenizer_test holds core/text's encoder to it id for id, and rad-tokcheck does the
 * same over large corpora. Where the two disagree because this one is wrong -- see tokref.h -- the
 * test says so case by case rather than excluding the input.
 */
#include "tokref.h"

#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <queue>
#include <regex>

namespace rad {
namespace tokref {

namespace {

struct ByteLevelAlphabet {
    std::string to_utf8[256];                         /* byte -> its 1-2 byte UTF-8 spelling */
    std::unordered_map<std::string, uint8_t> from;    /* and back */
    uint32_t cpt_of[256];

    ByteLevelAlphabet() {
        bool taken[256] = { false };
        auto give = [&](int b, uint32_t cpt) {
            to_utf8[b] = unicode_cpt_to_utf8(cpt);
            cpt_of[b]  = cpt;
            from[to_utf8[b]] = (uint8_t)b;
            taken[b] = true;
        };
        for (int b = 0x21; b <= 0x7E; ++b) give(b, (uint32_t)b);   /* '!' .. '~'  */
        for (int b = 0xA1; b <= 0xAC; ++b) give(b, (uint32_t)b);   /* U+00A1..AC  */
        for (int b = 0xAE; b <= 0xFF; ++b) give(b, (uint32_t)b);   /* U+00AE..FF  */
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b) if (!taken[b]) give(b, 256 + n++);
    }
};

const ByteLevelAlphabet& alphabet() {
    static const ByteLevelAlphabet a;
    return a;
}

std::string byte_encode(std::string_view raw) {
    const ByteLevelAlphabet& a = alphabet();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += a.to_utf8[c];
    return out;
}

constexpr const char* k_metaspace = "\xe2\x96\x81";   /* U+2581 LOWER ONE EIGHTH BLOCK */

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    for (;;) {
        size_t j = s.find(from, i);
        if (j == std::string::npos) { out.append(s, i, std::string::npos); break; }
        out.append(s, i, j - i);
        out.append(to);
        i = j + from.size();
    }
    s.swap(out);
}

/* ================================================================== the pre-tokeniser regex
 *
 * WHICH PATTERNS ARE HANDLED STRUCTURALLY AND WHICH FALL THROUGH -- read this before changing
 * anything below it.
 *
 * llama.cpp carries hand-written splitters for the patterns that actually ship on models, in
 * vendor/llama/src/unicode.cpp: the GPT-2 pattern, the Llama-3 pattern, the Qwen2 pattern, the
 * Qwen3.5 pattern, Kimi-K2, the AFMoE digit rule, and "[^\n]+|[\n]+". Those are STRUCTURAL: a
 * codepoint-class state machine, no backtracking, and correct on the lookahead constructs
 * (`\s+(?!\S)`) that a general engine gets wrong or slow. `unicode_regex_split` dispatches on the
 * pattern string and picks one of them when it matches.
 *
 * Everything else FALLS THROUGH to std::regex over a collapsed representation of the text, where
 * each non-ASCII codepoint is replaced by one byte standing for its unicode category. That is
 * llama.cpp's trick and it is what makes `\p{L}` work under a std::regex that has no idea what
 * `\p{L}` is. Its limits, stated plainly, because they are real:
 *   - a pattern mixing `\p{...}` with a non-ASCII literal is refused, not approximated;
 *   - possessive quantifiers and atomic groups are not ECMAScript and are not supported;
 *   - std::regex backtracks, so a pathological pattern is slow rather than wrong.
 * A model whose pre-tokeniser needs more than that gets a hand-written splitter here, next to
 * the others, and the .rad still declares the pattern so the decision stays visible.
 *
 * The one rewrite we do apply before dispatching is the case-insensitive contraction group.
 * tokenizer.json writes `(?i:'s|'t|'re|'ve|'m|'ll|'d)`; ECMAScript has no inline flag group, so
 * llama.cpp expands it to explicit classes and keys its structural splitters off the EXPANDED
 * form. Rewriting here is what lets a pattern read straight out of tokenizer.json reach the
 * structural splitter instead of the fallback -- which for Qwen is the difference between a
 * correct split and a merely plausible one. */
const char* k_ci_contractions_in  = "(?i:'s|'t|'re|'ve|'m|'ll|'d)";
const char* k_ci_contractions_out = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

std::string canonicalise_pattern(const std::string& p) {
    std::string out = p;
    replace_all(out, k_ci_contractions_in, k_ci_contractions_out);
    return out;
}

/* Does `unicode_regex_split` have a structural splitter for this pattern? Used only for the
 * diagnostic line, so a slow model is diagnosable without a profiler.
 *
 * IT MUST CANONICALISE FIRST, for the same reason apply_split does: comparing the raw pattern
 * reports "[std::regex fallback]" for a pattern the split path does in fact handle structurally,
 * and a diagnostic that claims a correctness problem where there is none is worse than no
 * diagnostic at all. */
bool pattern_is_structural(const std::string& raw) {
    const std::string p = canonicalise_pattern(raw);
    static const char* k_structural[] = {
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "\\p{Han}+",
        "\\p{AFMoE_digits}",
        "[^\\n]+|[\\n]+",
        "\\d{1,3}(?=(?:\\d{3})*\\b)",
    };
    for (const char* s : k_structural) if (p == s) return true;
    return false;
}
/* --- normalisers ------------------------------------------------------------------------- */

/* A literal Replace is a plain find-and-replace. A regex Replace goes through std::regex, which
 * is acceptable HERE and not in the pre-tokeniser: normalisers run once over the whole prompt
 * rather than once per candidate split, and the patterns that show up in practice
 * (`\s+` -> ` `, `` -> `_`) are trivial. */
void apply_replace(std::string& text, const VocabStep& s) {
    if (s.flags & 1) {
        try {
            text = std::regex_replace(text, std::regex(s.arg0), s.arg1);
        } catch (const std::regex_error& e) {
            /* Refusing to normalise is worse than a slightly wrong normalisation only if you
             * think a silent difference is cheap. Say it, once, and leave the text alone. */
            RAD_WARN("tokenizer: Replace pattern '%s' did not compile (%s); step skipped",
                     s.arg0.c_str(), e.what());
        }
    } else {
        replace_all(text, s.arg0, s.arg1);
    }
}

void apply_norm(std::string& text, const VocabStep& s) {
    switch (s.kind) {
        case RAD_NORM_NFC:  text = unorm_utf8(text, false, true);  break;
        case RAD_NORM_NFKC: text = unorm_utf8(text, true,  true);  break;
        case RAD_NORM_NFD:  text = unorm_utf8(text, false, false); break;
        case RAD_NORM_NFKD: text = unorm_utf8(text, true,  false); break;
        case RAD_NORM_LOWERCASE: {
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            std::string out;
            out.reserve(text.size());
            for (uint32_t c : cpts) out += unicode_cpt_to_utf8(unicode_tolower(c));
            text.swap(out);
            break;
        }
        case RAD_NORM_STRIP: {
            /* Whitespace by the unicode definition, not isspace(): a non-breaking space in a
             * pasted prompt is whitespace to HuggingFace and would not be to isspace(). */
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            size_t a = 0, b = cpts.size();
            if (s.flags & 1) while (a < b && unicode_cpt_flags_from_cpt(cpts[a]).is_whitespace) ++a;
            if (s.flags & 2) while (b > a && unicode_cpt_flags_from_cpt(cpts[b-1]).is_whitespace) --b;
            std::string out;
            for (size_t i = a; i < b; ++i) out += unicode_cpt_to_utf8(cpts[i]);
            text.swap(out);
            break;
        }
        case RAD_NORM_REPLACE: apply_replace(text, s); break;
        case RAD_NORM_PREPEND: text = s.arg0 + text; break;
        default:
            RAD_WARN("tokenizer: normaliser step %u is not implemented; text passed through",
                     s.kind);
            break;
    }
}

/* --- pre-tokenisers ---------------------------------------------------------------------- */

enum { SPLIT_REMOVED = 0, SPLIT_ISOLATED = 1, SPLIT_MERGED_PREV = 2,
       SPLIT_MERGED_NEXT = 3, SPLIT_CONTIGUOUS = 4 };

void apply_split(std::vector<std::string>& words, const VocabStep& s) {
    const std::string pat = canonicalise_pattern(s.arg0);
    const std::vector<std::string> exprs = { pat };
    std::vector<std::string> out;
    out.reserve(words.size() * 2);
    for (const std::string& w : words) {
        if (w.empty()) continue;
        /* byte_encode=false: ByteLevel is its own declared step and applying it here as well
         * would encode twice. */
        std::vector<std::string> parts;
        try {
            parts = unicode_regex_split(w, exprs, /*byte_encode=*/false);
        } catch (const std::exception& e) {
            RAD_ERR("tokenizer: Split('%s') failed: %s", s.arg0.c_str(), e.what());
            parts = { w };
        }
        /* Behaviour only changes what happens to the DELIMITERS, and the patterns that ship on
         * real models are "Isolated" with a pattern that matches the pieces themselves, so
         * every part is kept. Removed drops empty parts only, which is already the case. */
        for (std::string& p : parts) if (!p.empty()) out.push_back(std::move(p));
    }
    words.swap(out);
}

void apply_bytelevel(std::vector<std::string>& words, const VocabStep& s) {
    const bool add_prefix_space = (s.flags & 1) != 0;
    const bool use_regex        = (s.flags & 2) != 0;

    if (add_prefix_space && !words.empty() && !words[0].empty() && words[0][0] != ' ') {
        words[0].insert(words[0].begin(), ' ');
    }
    if (use_regex) {
        /* HuggingFace's ByteLevel carries the GPT-2 pattern inside it. rad-convert normally
         * splits that out into an explicit Split step so the chain reads as what it does, but a
         * chain that declares it inline still has to work. */
        VocabStep split;
        split.kind = RAD_PRE_SPLIT;
        split.arg0 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
        split.iarg = SPLIT_ISOLATED;
        apply_split(words, split);
    }
    for (std::string& w : words) w = byte_encode(w);
}

void apply_metaspace(std::vector<std::string>& words, const VocabStep& s) {
    const std::string rep = s.arg0.empty() ? std::string(k_metaspace) : s.arg0;
    const int64_t prepend = s.iarg;            /* 0 never, 1 first, 2 always */
    const bool    split   = (s.flags & 1) != 0;

    for (size_t i = 0; i < words.size(); ++i) {
        std::string& w = words[i];
        if (prepend == 2 || (prepend == 1 && i == 0)) {
            if (w.empty() || w.compare(0, 1, " ") != 0) w.insert(0, " ");
        }
        replace_all(w, " ", rep);
    }
    if (!split) return;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        size_t i = 0;
        while (i < w.size()) {
            size_t j = w.find(rep, i + (w.compare(i, rep.size(), rep) == 0 ? rep.size() : 0));
            if (j == std::string::npos) { out.push_back(w.substr(i)); break; }
            out.push_back(w.substr(i, j - i));
            i = j;
        }
        if (w.empty()) out.push_back(w);
    }
    words.swap(out);
}

/* Whitespace pre-tokeniser: HuggingFace's is the regex \w+|[^\w\s]+ with Isolated behaviour.
 * Written structurally because \w is locale-dependent in std::regex and is not there. */
void apply_whitespace(std::vector<std::string>& words) {
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        size_t i = 0;
        while (i < cpts.size()) {
            const auto f = unicode_cpt_flags_from_cpt(cpts[i]);
            if (f.is_whitespace) { ++i; continue; }
            const bool word = f.is_letter || f.is_number || cpts[i] == '_';
            size_t j = i;
            while (j < cpts.size()) {
                const auto g = unicode_cpt_flags_from_cpt(cpts[j]);
                if (g.is_whitespace) break;
                const bool wj = g.is_letter || g.is_number || cpts[j] == '_';
                if (wj != word) break;
                ++j;
            }
            std::string piece;
            for (size_t k = i; k < j; ++k) piece += unicode_cpt_to_utf8(cpts[k]);
            out.push_back(std::move(piece));
            i = j;
        }
    }
    words.swap(out);
}

void apply_punctuation(std::vector<std::string>& words, const VocabStep& s) {
    const int64_t behavior = s.iarg;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        std::string acc;
        for (uint32_t c : cpts) {
            const auto f = unicode_cpt_flags_from_cpt(c);
            const bool punct = f.is_punctuation || f.is_symbol;
            if (punct) {
                if (!acc.empty()) { out.push_back(acc); acc.clear(); }
                if (behavior != SPLIT_REMOVED) out.push_back(unicode_cpt_to_utf8(c));
            } else {
                acc += unicode_cpt_to_utf8(c);
            }
        }
        if (!acc.empty()) out.push_back(acc);
    }
    words.swap(out);
}

void apply_digits(std::vector<std::string>& words, const VocabStep& s) {
    const bool individual = (s.flags & 1) != 0;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        std::string acc;
        bool acc_digit = false;
        for (uint32_t c : cpts) {
            const bool d = unicode_cpt_flags_from_cpt(c).is_number;
            if (acc.empty()) { acc_digit = d; acc += unicode_cpt_to_utf8(c); continue; }
            if (d != acc_digit || (d && individual)) {
                out.push_back(acc);
                acc.clear();
                acc_digit = d;
            }
            acc += unicode_cpt_to_utf8(c);
        }
        if (!acc.empty()) out.push_back(acc);
    }
    words.swap(out);
}

}  /* namespace */

std::vector<std::string> split(const std::string& pattern, const std::string& word) {
    VocabStep s;
    s.kind = RAD_PRE_SPLIT;
    s.arg0 = pattern;
    s.iarg = SPLIT_ISOLATED;
    std::vector<std::string> words = { word };
    apply_split(words, s);
    return words;
}

bool structural(const std::string& pattern) { return pattern_is_structural(pattern); }

namespace {

int run_pre(const Vocab& v, const std::string& text, std::vector<std::string>& words) {
    std::string t = text;
    for (const VocabStep& s : v.steps()) if (step_is_norm(s.kind)) apply_norm(t, s);

    words.assign(1, t);
    for (const VocabStep& s : v.steps()) {
        if (!step_is_pre(s.kind)) continue;
        switch (s.kind) {
            case RAD_PRE_SPLIT:       apply_split(words, s); break;
            case RAD_PRE_BYTELEVEL:   apply_bytelevel(words, s); break;
            case RAD_PRE_METASPACE:   apply_metaspace(words, s); break;
            case RAD_PRE_WHITESPACE:  apply_whitespace(words); break;
            case RAD_PRE_PUNCTUATION: apply_punctuation(words, s); break;
            case RAD_PRE_DIGITS:      apply_digits(words, s); break;
            case RAD_PRE_BYTE_FALLBACK:
                /* A declaration about what the MODEL does with an unknown piece, not a split.
                 * The fallback itself lives in bpe() below, where the miss happens. */
                break;
            default:
                RAD_ERR("tokenizer: pre-tokeniser step %u is not implemented", s.kind);
                return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

/* The BPE merge loop, in shape from llama.cpp's llm_tokenizer_bpe_session (MIT, 06938ac12).
 * A priority queue of candidate bigrams ordered by merge rank; pop the best, splice the two
 * symbols into one, and offer the two new neighbours. Stale entries are recognised by comparing
 * the recorded text against the symbols' current text rather than by deleting from the queue,
 * because deletion from a heap is the expensive operation and staleness is cheap to detect. */
void bpe(const Vocab& v, const std::vector<std::string>& words, std::vector<int32_t>& out) {
    struct Symbol { int prev, next; const char* text; size_t n; };
    struct Bigram {
        int left, right, rank;
        size_t size;
        std::string text;
        bool operator<(const Bigram& o) const {
            /* Lower rank wins, so the comparator is inverted for a max-heap; ties break on the
             * LEFTMOST position, which is what makes the result independent of heap order. */
            return rank > o.rank || (rank == o.rank && left > o.left);
        }
    };

    std::vector<Symbol> symbols, final_syms;
    std::priority_queue<Bigram> work;
    int final_prev = -1;

    /* Byte fallback is only reachable when the chain declares it; without it an unmatched piece
     * is dropped, which is llama.cpp's behaviour and is what a vocab with a full byte alphabet
     * relies on (there, nothing is unmatched). */
    bool have_byte_fallback = false;
    bool have_bytelevel = false;
    for (const VocabStep& s : v.steps()) {
        if (s.kind == RAD_PRE_BYTE_FALLBACK) have_byte_fallback = true;
        if (s.kind == RAD_PRE_BYTELEVEL)     have_bytelevel = true;
    }

    for (const std::string& word : words) {
        if (word.empty()) continue;
        work = std::priority_queue<Bigram>();
        symbols.clear();

        size_t offset = 0;
        int index = 0;
        /* A user-defined token is taken whole rather than re-split. Added tokens are exact
         * strings the vocabulary promises to round-trip; decomposing one into codepoints and
         * merging back can land on a different sequence, because the merge table was never
         * fitted over them.
         *
         * `ignore_merges` also takes ordinary vocabulary entries whole, after pre-tokenization. */
        if (v.find(word) >= 0 && ((v.build().ignore_merges && v.type(v.find(word)) != RAD_TT_CONTROL) ||
                                 v.type(v.find(word)) == RAD_TT_USER_DEFINED)) {
            symbols.push_back({ -1, -1, word.c_str(), word.size() });
            offset = word.size();
        }
        while (offset < word.size()) {
            Symbol sym;
            const size_t len = std::min(unicode_len_utf8(word[offset]), word.size() - offset);
            sym.text = word.c_str() + offset;
            sym.n    = len;
            offset  += len;
            sym.prev = index - 1;
            sym.next = offset == word.size() ? -1 : index + 1;
            ++index;
            symbols.push_back(sym);
        }

        auto offer = [&](int l, int r) {
            if (l == -1 || r == -1) return;
            const std::string lt(symbols[l].text, symbols[l].n);
            const std::string rt(symbols[r].text, symbols[r].n);
            const int32_t rank = v.merge_rank(lt, rt);
            if (rank < 0) return;
            work.push({ l, r, rank, lt.size() + rt.size(), lt + rt });
        };

        for (int i = 1; i < (int)symbols.size(); ++i) offer(i - 1, i);

        while (!work.empty()) {
            const Bigram b = work.top();
            work.pop();
            Symbol& l = symbols[b.left];
            Symbol& r = symbols[b.right];
            if (l.n == 0 || r.n == 0) continue;
            if (std::string(l.text, l.n) + std::string(r.text, r.n) != b.text) continue;
            l.n += r.n;
            r.n = 0;
            l.next = r.next;
            if (r.next >= 0) symbols[r.next].prev = b.left;
            offer(l.prev, b.left);
            offer(b.left, l.next);
        }

        for (Symbol& s : symbols) {
            if (s.n == 0) continue;
            s.prev = final_prev;
            s.next = -1;
            if (final_prev != -1) final_syms[final_prev].next = (int)final_syms.size();
            final_syms.push_back(s);
            final_prev = (int)final_syms.size() - 1;
        }
    }

    for (int i = final_syms.empty() ? -1 : 0; i != -1; i = final_syms[i].next) {
        const Symbol& s = final_syms[i];
        if (s.n == 0) continue;
        const std::string str(s.text, s.n);
        const int32_t id = v.find(str);
        if (id >= 0) { out.push_back(id); continue; }

        /* Miss. Decompose into single-byte tokens; how a byte is SPELT depends on the chain.
         * Under ByteLevel every byte is already its own mapped codepoint, so the piece splits at
         * codepoint boundaries; under byte fallback it is "<0xNN>". */
        if (have_bytelevel) {
            for (size_t k = 0; k < str.size(); ) {
                const size_t n = std::min(unicode_len_utf8(str[k]), str.size() - k);
                const int32_t b = v.find(str.substr(k, n));
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              str.substr(k, n).c_str());
                k += n;
            }
        } else if (have_byte_fallback) {
            static const char* hex = "0123456789ABCDEF";
            for (unsigned char c : str) {
                const char buf[7] = { '<', '0', 'x', hex[c >> 4], hex[c & 15], '>', 0 };
                const int32_t b = v.find(buf);
                if (b >= 0) out.push_back(b);
                else if (v.unk() >= 0) out.push_back(v.unk());
            }
        } else if (v.unk() >= 0) {
            out.push_back(v.unk());
        } else {
            RAD_WARN("tokenizer: piece '%s' is not in the vocab and the chain declares neither "
                     "ByteLevel nor ByteFallback; it is dropped", str.c_str());
        }
    }
}

/* Unigram, by Viterbi over a prefix trie -- SentencePiece's algorithm, in shape from
 * llama.cpp's llm_tokenizer_ugm_session. Scores are summed in double precision because the
 * reference implementation does and the argmax genuinely differs in float on long inputs. */
void unigram(const Vocab& v, const std::string& text, std::vector<int32_t>& out) {
    if (text.empty()) return;
    const size_t n = text.size();

    struct Best { int32_t id; size_t from; double score; };
    std::vector<Best> best(n + 1, { v.unk(), 0, -DBL_MAX });
    best[0] = { v.unk(), 0, 0.0 };

    /* The unknown-token penalty: SentencePiece uses (min score - 10). Recomputing it here rather
     * than storing it keeps the .rad from carrying a derived number that could disagree. */
    double unk_penalty = 0.0;
    if (!v.build().scores.empty()) {
        unk_penalty = *std::min_element(v.build().scores.begin(), v.build().scores.end()) - 10.0;
    }

    for (size_t i = 0; i < n; ) {
        const size_t cp_len = std::min(unicode_len_utf8(text[i]), n - i);
        bool matched_whole_cp = false;
        /* Longest-prefix scan. A trie would beat this on a large vocab; the substring lookup is
         * O(len) with a bounded len because no vocab entry is longer than the longest token, and
         * that bound is what keeps it linear. */
        const size_t max_len = std::min(n - i, (size_t)256);
        for (size_t l = 1; l <= max_len; ++l) {
            const int32_t id = v.find(std::string_view(text.data() + i, l));
            if (id < 0) continue;
            if (l == cp_len) matched_whole_cp = true;
            const double sc = v.type(id) == RAD_TT_USER_DEFINED ? 0.0 : (double)v.score(id);
            const double cand = best[i].score + sc;
            if (cand > best[i + l].score) best[i + l] = { id, i, cand };
        }
        if (!matched_whole_cp) {
            const double cand = best[i].score + unk_penalty;
            if (cand > best[i + cp_len].score) best[i + cp_len] = { v.unk(), i, cand };
        }
        i += cp_len;
    }

    /* A VOCABULARY WITH NO UNK TOKEN HAS NO ID FOR TEXT NO PIECE COVERS, and the Viterbi above
     * still routes such text through the unk transition -- whose id is then -1, which is not a
     * token and must never reach the output. The span is spelled as byte tokens when the chain
     * declares byte fallback and dropped otherwise, as bpe() does with a piece it cannot match.
     * Built back to front like the rest of the backtrack, so the bytes go in reversed. */
    const int32_t unk = v.unk();
    bool byte_fallback = false;
    if (unk < 0)
        for (const VocabStep& s : v.steps())
            if (s.kind == RAD_PRE_BYTE_FALLBACK) byte_fallback = true;

    const size_t start = out.size();
    size_t dropped = 0;
    bool prev_unk = false;
    for (size_t pos = n; ; ) {
        const Best& b = best[pos];
        const bool is_unk = (b.id == unk);
        if (is_unk && unk < 0) {
            static const char* hex = "0123456789ABCDEF";
            for (size_t k = pos; k-- > b.from; ) {
                const unsigned char c = (unsigned char)text[k];
                const char buf[7] = { '<', '0', 'x', hex[c >> 4], hex[c & 15], '>', 0 };
                const int32_t id = byte_fallback ? v.find(buf) : -1;
                if (id >= 0) out.push_back(id);
                else         ++dropped;
            }
        } else if (!(prev_unk && is_unk)) {
            out.push_back(b.id);
        }
        if (b.from == 0) break;
        prev_unk = is_unk;
        pos = b.from;
    }
    std::reverse(out.begin() + start, out.end());
    if (dropped)
        RAD_WARN("tokenizer: %zu byte(s) of this text have no piece in the vocab, which has no unk "
                 "token and %s; they are dropped", dropped,
                 byte_fallback ? "lacks a byte token for them" : "declares no byte fallback");
}

/* WordPiece: greedy longest-match-first with a continuation prefix. Kept because BERT-family
 * embedding models are a real deployment and they are all WordPiece; it is 30 lines. */
void wordpiece(const Vocab& v, const std::vector<std::string>& words, std::vector<int32_t>& out) {
    for (const std::string& w : words) {
        if (w.empty()) continue;
        size_t i = 0;
        std::vector<int32_t> piece;
        bool ok = true;
        while (i < w.size()) {
            size_t j = w.size();
            int32_t id = -1;
            for (; j > i; --j) {
                std::string sub = w.substr(i, j - i);
                if (i > 0) sub = "##" + sub;
                id = v.find(sub);
                if (id >= 0) break;
            }
            if (id < 0) { ok = false; break; }
            piece.push_back(id);
            i = j;
        }
        if (ok) out.insert(out.end(), piece.begin(), piece.end());
        else if (v.unk() >= 0) out.push_back(v.unk());
    }
}

}  /* namespace */

int encode(const Vocab& v, std::string_view text, std::vector<int32_t>& out,
           bool add_special, bool parse_special) {
    if (!v.ready()) return RAD_E_STATE;

    if (add_special && v.add_bos() && v.bos() >= 0) out.push_back(v.bos());

    /* Special-token partitioning. The text is cut into (raw, token, raw, token, ...) around any
     * declared special token that appears literally, before any other step touches it, because a
     * special token is not something the pre-tokeniser is allowed to split.
     *
     * `parse_special == false` means a control token typed by a user is ordinary text. That is
     * not a nicety: without it, user content containing "<|im_start|>system" reaches the model
     * as an actual role switch. */
    struct Frag { bool is_token; int32_t id; size_t off, len; };
    std::vector<Frag> frags;
    const std::string raw(text);
    frags.push_back({ false, -1, 0, raw.size() });

    for (int32_t sid : v.special_ids()) {
        const uint8_t t = v.type(sid);
        if (!parse_special && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN)) continue;
        const std::string& st = v.text(sid);
        if (st.empty()) continue;

        std::vector<Frag> next;
        next.reserve(frags.size());
        for (const Frag& f : frags) {
            if (f.is_token) { next.push_back(f); continue; }
            size_t pos = f.off;
            const size_t end = f.off + f.len;
            for (;;) {
                const size_t m = raw.find(st, pos);
                if (m == std::string::npos || m + st.size() > end) break;
                if (m > pos) next.push_back({ false, -1, pos, m - pos });
                next.push_back({ true, sid, m, st.size() });
                pos = m + st.size();
            }
            if (pos < end) next.push_back({ false, -1, pos, end - pos });
        }
        frags.swap(next);
    }

    for (const Frag& f : frags) {
        if (f.is_token) { out.push_back(f.id); continue; }
        if (f.len == 0) continue;
        const std::string chunk = raw.substr(f.off, f.len);

        std::vector<std::string> words;
        RAD_TRY(run_pre(v, chunk, words));

        switch (v.kind()) {
            case RAD_TOK_BPE:       bpe(v, words, out); break;
            case RAD_TOK_WORDPIECE: wordpiece(v, words, out); break;
            case RAD_TOK_UNIGRAM: {
                /* Unigram runs Viterbi over the whole normalised fragment; the pre-tokeniser's
                 * job there is only to insert the metaspace, so the words are rejoined. */
                std::string joined;
                for (const std::string& w : words) joined += w;
                unigram(v, joined, out);
                break;
            }
            default:
                RAD_ERR("tokenizer: kind %u has no encoder", v.kind());
                return RAD_E_UNSUPPORTED;
        }
    }

    if (add_special && v.add_eos() && v.eos() >= 0) out.push_back(v.eos());
    return RAD_OK;
}

}  /* namespace tokref */
}  /* namespace rad */
