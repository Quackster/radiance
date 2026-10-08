/* host_ref.cpp -- llama.cpp's sampling chain, ported to plain C++ over a float array.
 *
 * Source: llama.cpp `src/llama-sampler.cpp` at upstream commit 06938ac12 (MIT). Ported rather
 * than linked: this is the oracle the device samplers are checked against, and an oracle that
 * drags a second inference engine in is a dependency rather than a reference. See host_ref.h for
 * the two places this deliberately differs from upstream -- the chain order and the RNG.
 */
#include "host_ref.h"

#include "rad_sample.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>

namespace rad {

/* ================================================================== the candidate set */

void CandidateSet::from_logits(const float* logits, int64_t n_vocab) {
    c.resize((size_t)n_vocab);
    for (int64_t i = 0; i < n_vocab; ++i) {
        c[(size_t)i].id    = (int32_t)i;
        c[(size_t)i].logit = logits[i];
        c[(size_t)i].p     = 0.0f;
    }
    sorted = false;
    selected = -1;
}

/* Upstream sorts on the logit alone and leaves equal logits in whatever order the partial sort
 * happened to produce. That is fine for a single-threaded reference and useless for one that has
 * to agree with a device kernel, where the order equal elements land in is a property of the
 * block size. The id tie-break makes the sort a total order, so the two agree. */
void CandidateSet::sort_desc() {
    if (sorted) return;
    std::stable_sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
        if (a.logit != b.logit) return a.logit > b.logit;
        return a.id < b.id;
    });
    sorted = true;
}

/* ================================================================== reproducible randomness */

/* The stream's key, from abi/rad_sample.h. A stream this file does not know gets a key of its
 * own rather than the pick's, so it can never silently repeat the pick's draw. */
static uint64_t stream_key(uint32_t stream) {
    switch (stream) {
        case RAD_RNG_PICK:   return RAD_SAMPLE_KEY_PICK;
        case RAD_RNG_XTC:    return RAD_SAMPLE_KEY_XTC;
        case RAD_RNG_ACCEPT: return RAD_SAMPLE_KEY_ACCEPT;
        case RAD_RNG_RESID:  return RAD_SAMPLE_KEY_RESID;
        default:             return rad_sample_splitmix64((uint64_t)stream);
    }
}

/* rad_sample_uniform01's bits, before the float conversion: one splitmix64 round over
 * ((seed ^ key) * golden ratio + pos). The same handful of integer ops a sampler kernel runs,
 * which is what makes it affordable there where a Mersenne twister's state is not. */
uint64_t rng_bits(uint64_t seed, uint64_t pos, uint32_t stream) {
    return rad_sample_splitmix64((seed ^ stream_key(stream)) * 0x9E3779B97F4A7C15ull + pos);
}

float rng_uniform(uint64_t seed, uint64_t pos, uint32_t stream) {
    /* 24 bits, which is every bit a float can hold. Drawing 32 and dividing by 2^32 would round
     * to 1.0f for the top values, and a uniform that can return 1.0 walks off the end of an
     * inverse CDF. Written as rad_sample_uniform01 itself, so the two cannot drift apart. */
    return rad_sample_uniform01(seed ^ stream_key(stream), pos);
}

/* ================================================================== individual samplers */

void host_softmax(CandidateSet& s, bool do_sort) {
    if (s.c.empty()) return;
    if (do_sort) s.sort_desc();

    float max_l = s.c[0].logit;
    if (!s.sorted)
        for (size_t i = 1; i < s.c.size(); ++i) max_l = std::max(max_l, s.c[i].logit);

    double cum = 0.0;
    for (auto& e : s.c) {
        const float p = std::exp(e.logit - max_l);
        e.p = p;
        cum += p;
    }
    if (cum <= 0.0) return;   /* every candidate masked out; leave p at zero and let the pick say so */
    for (auto& e : s.c) e.p = (float)(e.p / cum);
}

void host_temp(CandidateSet& s, float temp) {
    if (s.c.empty()) return;

    /* temp <= 0 is greedy and stays a real path rather than a caller's special case: the argmax
     * it produces must use the SAME tie-break the argmax kernel uses (lowest id wins), or a
     * temperature-0 request stops matching a greedy run. */
    if (temp <= 0.0f) {
        size_t max_i = 0;
        for (size_t i = 1; i < s.c.size(); ++i) {
            const Candidate& e = s.c[i];
            if (e.logit > s.c[max_i].logit) max_i = i;
        }
        for (size_t i = 0; i < s.c.size(); ++i)
            if (i != max_i) s.c[i].logit = -INFINITY;
        return;
    }

    for (auto& e : s.c) e.logit /= temp;
}

void host_top_k(CandidateSet& s, int32_t k) {
    if (k <= 0) return;
    if ((size_t)k >= s.c.size()) { s.sort_desc(); return; }
    s.sort_desc();
    s.c.resize((size_t)k);
}

void host_top_p(CandidateSet& s, float p, size_t min_keep) {
    if (p >= 1.0f || s.c.empty()) return;

    s.sort_desc();
    host_softmax(s, false);

    /* Upstream has an adaptive partial sort that grows k until the nucleus closes; it is a
     * performance path over a 151K vocab and produces the same set as a full sort, so the
     * reference takes the full sort and the device kernel takes the radix threshold. */
    float  cum = 0.0f;
    size_t last_idx = s.c.size();
    for (size_t i = 0; i < s.c.size(); ++i) {
        cum += s.c[i].p;
        if (cum >= p && i + 1 >= min_keep) { last_idx = i + 1; break; }
    }
    s.c.resize(last_idx);
}

void host_min_p(CandidateSet& s, float p, size_t min_keep) {
    if (p <= 0.0f || s.c.empty()) return;

    /* Upstream has an unsorted fast path that filters in place and falls back to the sorted one
     * when too few survive. Both select the same SET and differ only in the order they leave it
     * in; the reference takes the sorted path because an order that depends on which path ran is
     * an order a device kernel cannot reproduce. */
    s.sort_desc();

    const float min_logit = s.c[0].logit + std::log(p);   /* p_i >= p * p_max */
    size_t i = 1;                                          /* the mode always survives */
    for (; i < s.c.size(); ++i)
        if (s.c[i].logit < min_logit && i >= min_keep) break;

    s.c.resize(i);
}

void host_typical(CandidateSet& s, float p, size_t min_keep) {
    if (p >= 1.0f || s.c.empty()) return;

    host_softmax(s, true);

    float entropy = 0.0f;
    for (const auto& e : s.c)
        if (e.p > 0.0f) entropy += -e.p * std::log(e.p);

    /* The distance between a candidate's surprisal and the distribution's entropy. Sorting by it
     * is what makes typical sampling different from every other cut here: it keeps the tokens
     * that are AS surprising as expected, not the least surprising ones. */
    std::vector<float> shifted(s.c.size());
    for (size_t i = 0; i < s.c.size(); ++i) {
        const float pi = s.c[i].p;
        shifted[i] = std::fabs(-(pi > 0.0f ? std::log(pi) : -INFINITY) - entropy);
    }

    std::vector<size_t> idx(s.c.size());
    std::iota(idx.begin(), idx.end(), (size_t)0);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (shifted[a] != shifted[b]) return shifted[a] < shifted[b];
        return s.c[a].id < s.c[b].id;      /* total order, same reason as sort_desc */
    });

    float  cum = 0.0f;
    size_t last_idx = idx.size();
    for (size_t i = 0; i < idx.size(); ++i) {
        cum += s.c[idx[i]].p;
        if (cum > p && (min_keep == 0 || i >= min_keep - 1)) { last_idx = i + 1; break; }
    }

    std::vector<Candidate> kept;
    kept.reserve(last_idx);
    for (size_t i = 0; i < last_idx; ++i) kept.push_back(s.c[idx[i]]);
    s.c = std::move(kept);
    s.sorted = false;                       /* surprisal order is not logit order */
}

void host_penalties(CandidateSet& s, const int32_t* history, int64_t n_history,
                    int32_t last_n, float rep, float freq, float pres) {
    if (last_n == 0) return;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;
    if (!history || n_history <= 0) return;

    const int64_t window = last_n < 0 ? n_history : std::min<int64_t>(last_n, n_history);
    std::unordered_map<int32_t, int> count;
    count.reserve((size_t)window * 2);
    for (int64_t i = n_history - window; i < n_history; ++i) count[history[i]]++;

    for (auto& e : s.c) {
        auto it = count.find(e.id);
        if (it == count.end()) continue;
        const int n = it->second;

        /* The paper divides. Dividing makes a token with a negative logit MORE likely, which is
         * obviously wrong, so everyone multiplies below zero instead. Upstream's comment says the
         * same thing and the reference has to reproduce the fix, not the paper. */
        if (e.logit <= 0.0f) e.logit *= rep;
        else                 e.logit /= rep;

        e.logit -= (float)n * freq + (n > 0 ? 1.0f : 0.0f) * pres;
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ DRY */

void DryBreakers::index(int32_t n_vocab) {
    const size_t words = n_vocab > 0 ? ((size_t)n_vocab + 31) / 32 : 0;
    heads.assign(words, 0u);
    singles.assign(words, 0u);
    for (const auto& e : seqs) {
        const int32_t t = e.first;
        if (t < 0 || t >= n_vocab) continue;
        heads[(size_t)(t >> 5)] |= 1u << ((uint32_t)t & 31u);
        if (e.second.empty()) singles[(size_t)(t >> 5)] |= 1u << ((uint32_t)t & 31u);
    }
}

bool DryBreakers::is_head(int32_t t) const {
    if (heads.empty()) return seqs.count(t) > 0;
    return bit(heads, t);
}

bool DryBreakers::is_single(int32_t t) const {
    if (singles.empty()) {
        auto range = seqs.equal_range(t);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second.empty()) return true;
        return false;
    }
    return bit(singles, t);
}

/* The longest prefix of `s` of at most `max` bytes that does not end inside a UTF-8 sequence.
 * Upstream truncates at the byte; cutting a character in half leaves a tail the tokeniser may
 * refuse, and a refused tail fails the whole request. */
static std::string utf8_prefix(const std::string& s, size_t max) {
    if (s.size() <= max) return s;
    size_t n = max;
    while (n > 0 && ((unsigned char)s[n] & 0xC0u) == 0x80u) --n;
    return s.substr(0, n);
}

int dry_prepare_breakers(const VocabView& vocab, const std::vector<std::string>& breakers,
                         DryBreakers* out, int max_tail_len) {
    if (!out) return RAD_E_INVAL;
    out->seqs.clear();
    out->heads.clear();
    out->singles.clear();
    if (breakers.empty()) return RAD_OK;
    if (breakers.size() > kDryMaxBreakers) {
        RAD_WARN("dry: %zu sequence breakers, more than the %zu the breaker scan accepts",
                 breakers.size(), kDryMaxBreakers);
        return RAD_E_INVAL;
    }

    /* Upstream: get_overlapping_token_sequences. A full vocabulary scan per breaker string, which
     * is why this runs once per request on admission and never on the step path. */
    const int32_t n = vocab.n_tokens();
    for (const std::string& given : breakers) {
        const std::string str = utf8_prefix(given, kDryBreakerMaxBytes);
        if (str.empty()) continue;

        /* THE TAIL DEPENDS ONLY ON WHERE THE BREAKER WAS SPLIT, not on which token covered its
         * head: every token whose suffix is the breaker's first i bytes is completed by the
         * tokenisation of the remaining bytes. So each split point is tokenised once rather than
         * once per matching token -- a breaker that starts with a common byte matches thousands
         * of tokens, and tokenising is the expensive half of the scan. */
        std::vector<std::vector<int32_t>> tail_at(str.size());
        std::vector<char> have(str.size(), 0);

        for (int32_t id = 0; id < n; ++id) {
            const std::string& word = vocab.token_piece(id);
            if (word.find(str) != std::string::npos) {
                out->seqs.emplace(id, std::vector<int32_t>());
                continue;
            }
            size_t pos = std::string::npos;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool   match = true;
                size_t i = 1;
                for (; i < str.size() && i + pos < word.size(); ++i) {
                    if (word[pos + i] != str[i]) { match = false; break; }
                }
                if (!match) continue;

                if (!have[i]) {
                    std::vector<int32_t>& fresh = tail_at[i];
                    const int st = vocab.tokenize(str.substr(i), &fresh);
                    if (st < 0) {
                        /* A breaker that cannot be tokenised is a breaker that silently stops
                         * working, and a DRY request whose breakers vanished is a DRY request
                         * that quietly changed behaviour. Say so. */
                        RAD_WARN("dry: sequence breaker \"%s\" does not tokenise (%s)",
                                 str.c_str(), rad_strerror(st));
                        out->seqs.clear();
                        return st;
                    }
                    if (max_tail_len >= 0 && (int)fresh.size() > max_tail_len)
                        fresh.resize((size_t)max_tail_len);
                    have[i] = 1;
                }
                const std::vector<int32_t>& tail = tail_at[i];

                bool dup = false;
                auto range = out->seqs.equal_range(id);
                for (auto it = range.first; it != range.second; ++it)
                    if (it->second == tail) { dup = true; break; }
                if (!dup) out->seqs.emplace(id, tail);
            }
        }
    }
    out->index(n);
    return RAD_OK;
}

int dry_rep_limit(const DryBreakers& breakers, const int32_t* window, int n) {
    if (!window || n <= 0) return n > 0 ? n : 0;
    if (breakers.empty()) return n;

    /* rat(i): the i-th token counting back from the end of the window. */
    auto rat = [&](int i) -> int32_t { return window[n - 1 - i]; };

    for (int i = 0; i < n; ++i) {
        const int32_t tok = rat(i);
        if (!breakers.is_head(tok)) continue;

        int longest = -1;
        auto its = breakers.seqs.equal_range(tok);
        for (auto it = its.first; it != its.second; ++it) {
            const int seq_len = (int)it->second.size();
            if (seq_len <= longest || seq_len > i) continue;
            bool match = true;
            for (int off = 0; off < seq_len; ++off) {
                if (it->second[(size_t)off] != rat(i - off - 1)) { match = false; break; }
            }
            if (match) longest = seq_len;
        }
        if (longest >= 0) return i - longest;
    }
    return n;
}

void host_dry(CandidateSet& s, const int32_t* history, int64_t n_history,
              int64_t total_context, const DryParams& p, const DryBreakers& breakers) {
    if (p.multiplier == 0.0f || p.base < 1.0f || p.penalty_last_n == 0) return;
    if (!history || n_history <= 0) return;
    if (total_context <= 0) total_context = n_history;

    const int64_t effective = (p.penalty_last_n == -1) ? total_context
                                                       : std::max(p.penalty_last_n, 0);
    const int n_rep = (int)std::min(std::min(n_history, effective), total_context);
    if (n_rep <= p.allowed_length) return;

    /* rat(i): the i-th token counting back from the end of the window. Upstream reads it off a
     * ring buffer; here the history is a flat array the scheduler already owns. */
    auto rat = [&](int i) -> int32_t { return history[n_history - 1 - i]; };

    std::vector<int> repeat_count((size_t)n_rep, 0);
    std::unordered_map<int32_t, int> max_token_repeat;

    /* Step 1: a restart sequence bounds how long a repetition may be counted. Walking back from
     * the end, the first token that heads a complete breaker sequence caps it. */
    const int rep_limit = dry_rep_limit(breakers, history + (n_history - n_rep), n_rep);
    if (rep_limit < p.allowed_length) return;

    /* Step 2: the Z-algorithm, run in reverse, giving for every position the length of the
     * suffix that also appears ending there. Linear despite the nested loops, because lt/rt bound
     * each token to one visit. */
    {
        const int last = n_rep - 1;
        int rt = 0, lt = 0;
        for (int k = 1; k < n_rep; ++k) {
            if (k > rt) {
                int n = 0;
                while (n + k < n_rep && rat(n) == rat(n + k)) ++n;
                repeat_count[(size_t)(last - k)] = std::min(n, rep_limit);
                if (n > 0) { lt = k; rt = k + n - 1; }
            } else {
                const int pi = k - lt;
                const int right_part_len = rt - k + 1;
                if (repeat_count[(size_t)(last - pi)] < right_part_len) {
                    repeat_count[(size_t)(last - k)] =
                        std::min(repeat_count[(size_t)(last - pi)], rep_limit);
                } else {
                    int i = rt + 1;
                    while (i < n_rep && rat(i) == rat(i - k)) ++i;
                    repeat_count[(size_t)(last - k)] = std::min(i - k, rep_limit);
                    lt = k; rt = i - 1;
                }
            }
        }
    }

    /* Step 3: for every position that ends a long-enough repeat, the NEXT token would extend it.
     * That token is the one to penalise, and by the length it would reach. */
    for (int i = 0; i < n_rep - 1; ++i) {
        const int len = repeat_count[(size_t)i];
        if (len < p.allowed_length) continue;
        const int32_t tok = rat(n_rep - 2 - i);
        auto it = max_token_repeat.find(tok);
        if (it == max_token_repeat.end() || it->second < len) max_token_repeat[tok] = len;
    }

    /* Step 4: an exponential penalty in the excess length, clamped so pow() cannot overflow a
     * float -- log(FLT_MAX) is 88.72, and past that the penalty is infinite anyway. */
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exponent = 0;
    if (p.base > 1.000001f) max_exponent = (int)(FLOAT_MAX_LOG / std::log(p.base));

    for (auto& e : s.c) {
        auto it = max_token_repeat.find(e.id);
        if (it == max_token_repeat.end()) continue;

        /* A token that is itself a single-token breaker is exempt: penalising it would penalise
         * the very thing that ends the repetition. */
        if (breakers.is_single(e.id)) continue;

        int exp_n = it->second - p.allowed_length;
        if (max_exponent > 0 && exp_n > max_exponent) exp_n = max_exponent;
        e.logit -= p.multiplier * std::pow(p.base, (float)exp_n);
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ XTC */

void host_xtc(CandidateSet& s, float probability, float threshold, size_t min_keep,
              uint64_t seed, uint64_t pos) {
    if (probability <= 0.0f || threshold > 0.5f || s.c.size() < 2) return;

    /* Its own RNG stream: XTC's coin and the pick must not be the same number, or a request
     * whose XTC fires becomes a request whose pick is correlated with that fact. It fires when
     * the uniform is BELOW the probability, the comparison `sample_xtc` makes, so the two agree
     * on the draw that lands exactly on it. */
    if (rng_uniform(seed, pos, RAD_RNG_XTC) >= probability) return;

    host_softmax(s, true);

    size_t pos_last = 0;
    for (size_t i = 0; i < s.c.size(); ++i) {
        if (s.c[i].p >= threshold) pos_last = i;
        else break;
    }

    /* Drop everything above the threshold EXCEPT the least probable of them. That is the whole
     * idea: remove the obvious continuation, not the unlikely ones. */
    if (s.c.size() - pos_last >= min_keep && pos_last > 0)
        s.c.erase(s.c.begin(), s.c.begin() + (std::ptrdiff_t)pos_last);
}

/* ------------------------------------------------------------------ the grammar mask */

void host_mask(CandidateSet& s, const uint32_t* bitmask, int64_t n_words) {
    if (!bitmask) return;
    for (auto& e : s.c) {
        const int64_t w = (int64_t)e.id >> 5;
        if (w >= n_words) { e.logit = -INFINITY; continue; }
        if (!(bitmask[w] & (1u << ((uint32_t)e.id & 31u)))) e.logit = -INFINITY;
    }
    s.sorted = false;
}

/* ------------------------------------------------------------------ the pick */

int32_t host_pick(CandidateSet& s, uint64_t seed, uint64_t pos) {
    if (s.c.empty()) { s.selected = -1; return -1; }

    float max_l = -INFINITY;
    for (const auto& e : s.c) max_l = std::max(max_l, e.logit);
    if (!(max_l > -INFINITY)) {
        /* Every candidate is -inf. That is a fully-masked distribution, which is a grammar that
         * admits nothing -- a real failure, not a token. The caller fails the request. */
        s.selected = -1;
        return -1;
    }
    if (s.c.size() == 1) { s.selected = 0; s.c[0].p = 1.0f; return s.c[0].id; }

    /* DESCENDING LOGIT, ties to the lower id, not the set's current order. Any fixed order
     * samples correctly, but a total order makes the answer a property of the uniform rather than
     * of how the previous stage happened to permute the set -- and this is the order sample_pick
     * walks, since sample_topk leaves its candidates in it. */
    std::vector<size_t> order(s.c.size());
    std::iota(order.begin(), order.end(), (size_t)0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (s.c[a].logit != s.c[b].logit) return s.c[a].logit > s.c[b].logit;
        return s.c[a].id < s.c[b].id;
    });

    double total = 0.0;
    for (size_t i : order) total += std::exp((double)s.c[i].logit - (double)max_l);
    if (!(total > 0.0)) { s.selected = -1; return -1; }

    const double want = (double)rng_uniform(seed, pos, RAD_RNG_PICK) * total;

    /* A candidate at -infinity is never chosen, and the fall-through is the last one above it: a
     * draw at the very top of the range can walk past every comparison in floating point, and
     * the candidates behind the last real one are the grammar-masked ones. */
    double  acc = 0.0;
    size_t  chosen = order.front();
    for (size_t i : order) {
        if (!(s.c[i].logit > -INFINITY)) continue;
        chosen = i;
        acc += std::exp((double)s.c[i].logit - (double)max_l);
        if (acc >= want) break;
    }

    for (size_t i = 0; i < s.c.size(); ++i)
        s.c[i].p = (float)(std::exp((double)s.c[i].logit - (double)max_l) / total);

    s.selected = (int)chosen;
    return s.c[chosen].id;
}

int32_t host_argmax(const float* logits, int64_t n_vocab) {
    if (!logits || n_vocab <= 0) return -1;
    int64_t best = 0;
    for (int64_t i = 1; i < n_vocab; ++i)
        if (logits[i] > logits[best]) best = i;      /* strict >, so a tie keeps the lowest id */
    return (int32_t)best;
}

/* ================================================================== the chain */

static int32_t argmax_of(const CandidateSet& s) {
    if (s.c.empty()) return -1;
    size_t best = 0;
    for (size_t i = 1; i < s.c.size(); ++i) {
        if (s.c[i].logit > s.c[best].logit) best = i;
        else if (s.c[i].logit == s.c[best].logit && s.c[i].id < s.c[best].id) best = i;
    }
    if (!(s.c[best].logit > -INFINITY)) return -1;   /* everything masked out */
    return s.c[best].id;
}

int32_t host_sample_into(CandidateSet& s, const SamplingParams& sp, const HostRefInput& in,
                         ChainOrder order) {
    /* Penalties first in both orders, because they act on raw logits and every cut below acts on
     * probabilities derived from them. */
    host_penalties(s, in.history, in.n_history, sp.penalty_last_n,
                   sp.rep_penalty, sp.freq_penalty, sp.pres_penalty);

    if (sp.dry_multiplier > 0.0f && in.dry_breakers) {
        DryParams dp;
        dp.multiplier     = sp.dry_multiplier;
        dp.base           = sp.dry_base;
        dp.allowed_length = sp.dry_allowed_length;
        dp.penalty_last_n = sp.dry_penalty_last_n;
        host_dry(s, in.history, in.n_history, in.total_context, dp, *in.dry_breakers);
    }

    /* THE GREEDY PATH SKIPS THE CHAIN, and it masks BEFORE it picks. Ordering matters here in a
     * way that is easy to get wrong: temperature 0 collapses every other logit to -inf, so a
     * mask applied afterwards can leave nothing at all. A greedy sequence is therefore
     * penalties -> mask -> argmax, not the chain with temp=0 threaded through it. */
    if (sp.greedy()) {
        host_mask(s, in.bitmask, in.n_words);
        const int32_t tok = argmax_of(s);
        for (size_t i = 0; i < s.c.size(); ++i) if (s.c[i].id == tok) { s.selected = (int)i; break; }
        return tok;
    }

    if (order == ChainOrder::Radiance) {
        /* `temp_last` moves temperature behind the cuts, as the device chain moves it onto the
         * candidate plane (Sampler::issue); the cuts keep this order either way. */
        if (!sp.temp_last) host_temp(s, sp.temp);
        host_mask(s, in.bitmask, in.n_words);
        host_top_k(s, sp.top_k);
        host_top_p(s, sp.top_p, 1);
        host_min_p(s, sp.min_p, 1);
        host_typical(s, sp.typical_p, 1);
        host_xtc(s, sp.xtc_probability, sp.xtc_threshold, 1, sp.seed, in.pos);
        if (sp.temp_last) host_temp(s, sp.temp);
    } else {
        /* llama.cpp's default order, for checking this port against llama.cpp itself. Note the
         * absence of top-n-sigma: it is not in SamplingParams and radiance does not ship it, so a
         * comparison that enables it upstream will disagree here and should. */
        host_mask(s, in.bitmask, in.n_words);
        host_top_k(s, sp.top_k);
        host_typical(s, sp.typical_p, 1);
        host_top_p(s, sp.top_p, 1);
        host_min_p(s, sp.min_p, 1);
        host_xtc(s, sp.xtc_probability, sp.xtc_threshold, 1, sp.seed, in.pos);
        host_temp(s, sp.temp);
    }

    return host_pick(s, sp.seed, in.pos);
}

int32_t host_sample(const float* logits, int64_t n_vocab, const SamplingParams& sp,
                    const HostRefInput& in, ChainOrder order) {
    if (!logits || n_vocab <= 0) return -1;
    CandidateSet s;
    s.from_logits(logits, n_vocab);
    return host_sample_into(s, sp, in, order);
}

}  /* namespace rad */
