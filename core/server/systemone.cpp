#include "systemone.h"

#include "mm/processor.h"
#include "llama/src/unicode.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace rad {
namespace server {
namespace {

bool only_keys(const json& value, std::initializer_list<const char*> allowed,
               const std::string& path, ApiError* err) {
    for (const auto& item : value.items()) {
        bool found = false;
        for (const char* key : allowed) if (item.key() == key) found = true;
        if (!found) {
            *err = err_unsupported(path + item.key(), "unsupported System One field");
            return false;
        }
    }
    return true;
}

bool is_space(uint32_t c) {
    return unicode_cpt_flags_from_cpt(c).is_whitespace || (c >= 0x1c && c <= 0x1f);
}

/* Python's strip + one-letter test, including non-ASCII native option labels. */
std::string letter_code(const std::string& label) {
    const auto cps = unicode_cpts_from_utf8(label);
    size_t a = 0, b = cps.size();
    while (a < b && is_space(cps[a])) ++a;
    while (b > a && is_space(cps[b - 1])) --b;
    return b == a + 1 && unicode_cpt_flags_from_cpt(cps[a]).is_letter
        ? unicode_cpt_to_utf8(cps[a]) : std::string();
}

std::string numbered_code(size_t i) {
    return (i < 10 ? "0" : "") + std::to_string(i);
}

const std::vector<std::string>& fallback_codes() {
    static const std::vector<std::string> codes = [] {
        std::vector<std::string> v;
        for (char c = 'A'; c <= 'Z'; ++c) v.emplace_back(1, c);
        for (size_t i = 0; i < 100; ++i) v.push_back(numbered_code(i));
        for (char c = 'a'; c <= 'z'; ++c) v.emplace_back(1, c);
        for (size_t i = 0; i < 200; ++i) v.push_back("#" + std::to_string(i));
        for (char a = 'A'; a <= 'Z'; ++a)
            for (char b = 'A'; b <= 'Z'; ++b) v.push_back(std::string{a, b});
        return v;
    }();
    return codes;
}

int32_t one_token(const ITokenizer& tok, const std::string& text) {
    const auto ids = tok.encode(text, false, false);
    return ids.size() == 1 ? ids[0] : -1;
}

void add_form(DecisionQuestion* q, std::vector<size_t>* group, int32_t id) {
    if (id < 0) return;
    const auto it = std::find(q->candidates.begin(), q->candidates.end(), id);
    const size_t index = (size_t)(it - q->candidates.begin());
    if (it == q->candidates.end()) q->candidates.push_back(id);
    if (std::find(group->begin(), group->end(), index) == group->end()) group->push_back(index);
}

bool choice_codes(const ITokenizer& tok, DecisionQuestion* q,
                   std::vector<std::string>* codes, ApiError* err) {
    std::vector<std::string> preferred;
    bool letters = true;
    for (const auto& name : q->labels) {
        preferred.push_back(letter_code(name));
        if (preferred.back().empty()) letters = false;
    }
    std::unordered_set<int32_t> used;
    for (size_t i = 0; i < q->labels.size(); ++i) {
        const auto desired = letters ? preferred[i] : q->labels.size() <= 26
            ? std::string(1, (char)('A' + i)) : numbered_code(i);
        int32_t id = -1;
        std::string code;
        auto take = [&](const std::string& candidate) {
            const int32_t t = one_token(tok, candidate);
            if (t < 0 || used.count(t)) return false;
            used.insert(t);
            code = candidate;
            id = t;
            return true;
        };
        if (!take(desired)) {
            for (const auto& candidate : fallback_codes()) if (take(candidate)) break;
        }
        if (id < 0) {
            *err = err_bad("questions." + q->name + ".criteria",
                           "the tokenizer has no distinct single-token alias left for these options");
            return false;
        }
        codes->push_back(code);
        q->groups.emplace_back();
        add_form(q, &q->groups.back(), id);
        add_form(q, &q->groups.back(), one_token(tok, " " + code));
    }
    return true;
}

bool description(const json& value, const std::string& path, ApiError* err) {
    if (value.is_string() || value.is_null()) return true;
    *err = err_bad(path, "must be a string or null");
    return false;
}

struct PromptPrefix {
    std::string text;
    /* Prepared image expansions include their own special-token boundaries. Text after the
     * last image must still join the question before tokenization for a single-question call. */
    std::vector<int32_t> image_tokens;
    std::string after_images;
};

int parse_question(const std::string& name, const json& source, const PromptPrefix& prefix,
                    const ITokenizer& tok, int64_t max_ctx, bool separate_prefix,
                    DecisionQuestion* q, ApiError* err) {
    const std::string path = "questions." + name;
    auto bad = [&](const std::string& field, const std::string& why) {
        *err = err_bad(path + field, why);
        return RAD_E_INVAL;
    };
    if (!source.is_object()) return bad("", "must be an object");
    if (!only_keys(source, {"type", "instructions", "criteria"}, path + ".", err)) return RAD_E_INVAL;
    if (!source.contains("instructions") || !source["instructions"].is_string())
        return bad(".instructions", "must be a string");
    if (!source.contains("type") || !source["type"].is_string())
        return bad(".type", "must be one of: noul, choice, score");
    q->name = name;
    q->type = source["type"].get<std::string>();
    const json criteria = source.value("criteria", json());
    std::string question = source["instructions"].get<std::string>();
    if (q->type == "noul") {
        if (!criteria.is_null() && !criteria.is_object()) return bad(".criteria", "must be an object or null");
        if (criteria.is_object()) {
            if (!only_keys(criteria, {"true", "false"}, path + ".criteria.", err)) return RAD_E_INVAL;
            for (const auto& item : criteria.items())
                if (!description(item.value(), path + ".criteria." + item.key(), err)) return RAD_E_INVAL;
        }
        q->labels = {"true", "false"};
        if (!criteria.empty()) {
            for (size_t i = 0; i < q->labels.size(); ++i) {
                const json value = criteria.value(q->labels[i], json());
                question += i == 0 ? "\nYes: " : "\nNo: ";
                question += value.is_null() ? "None" : value.get<std::string>();
            }
        }
        question += "\n\nReply with yes or no only.";
        for (const auto& forms : {std::vector<std::string>{"yes", "Yes", "YES"},
                                  std::vector<std::string>{"no", "No", "NO"}}) {
            q->groups.emplace_back();
            for (const auto& form : forms) add_form(q, &q->groups.back(), one_token(tok, form));
            if (q->groups.back().empty()) {
                *err = err_not_implemented("the d1 tokenizer needs single-token yes and no forms");
                return RAD_E_UNSUPPORTED;
            }
        }
    } else if (q->type == "choice") {
        if (!criteria.is_object() || criteria.empty()) return bad(".criteria", "must be a non-empty object");
        if (criteria.size() > fallback_codes().size()) return bad(".criteria", "too many options for the d1 alias codebook");
        for (const auto& item : criteria.items()) {
            if (!description(item.value(), path + ".criteria." + item.key(), err)) return RAD_E_INVAL;
            q->labels.push_back(item.key());
            std::string desc = item.value().is_null() ? "" : item.value().get<std::string>();
            if (desc.empty()) {
                desc = item.key();
                std::replace(desc.begin(), desc.end(), '_', ' ');
            }
            q->descriptions.push_back(std::move(desc));
        }
        std::vector<std::string> codes;
        if (!choice_codes(tok, q, &codes, err)) return RAD_E_INVAL;
        question += "\n\nOptions:\n";
        for (size_t i = 0; i < codes.size(); ++i) {
            if (i) question += '\n';
            question += codes[i] + " " + q->descriptions[i];
        }
        question += "\n\nReply with the option code only.";
    } else if (q->type == "score") {
        if (!criteria.is_array() || criteria.size() < 2 || criteria.size() > 10)
            return bad(".criteria", "must be an array of 2 to 10 level descriptions");
        question += "\n\n";
        for (size_t i = 0; i < criteria.size(); ++i) {
            if (!criteria[i].is_string()) return bad(".criteria", "each level description must be a string");
            q->labels.push_back(std::to_string(i));
            q->descriptions.push_back(criteria[i].get<std::string>());
            if (i) question += '\n';
            question += q->labels.back() + " " + q->descriptions.back();
            q->groups.emplace_back();
            add_form(q, &q->groups.back(), one_token(tok, q->labels.back()));
            if (q->groups.back().empty()) {
                *err = err_not_implemented("the d1 tokenizer needs single-token score digits");
                return RAD_E_UNSUPPORTED;
            }
        }
        question += "\n\nReply with a single digit 0-" + std::to_string(criteria.size() - 1) + " only.";
    } else return bad(".type", "must be one of: noul, choice, score");

    const std::string suffix = question + "<|im_end|>\n<|im_start|>assistant\n";
    q->prompt = prefix.text + suffix;
    if (!prefix.image_tokens.empty()) {
        q->tokens = prefix.image_tokens;
        auto append = [&](const std::string& text) {
            const auto ids = tok.encode(text, false, true);
            q->tokens.insert(q->tokens.end(), ids.begin(), ids.end());
        };
        if (separate_prefix) {
            append(prefix.after_images);
            q->shared_prefix_tokens = (int32_t)q->tokens.size();
            append(suffix);
        } else append(prefix.after_images + suffix);
    } else if (separate_prefix) {
        /* d1's multi-question reference tokenizes its shared trunk and each branch separately.
         * Merging the strings first changes BPE at the join, e.g. newline-leading instructions. */
        q->tokens = tok.encode(prefix.text, false, true);
        q->shared_prefix_tokens = (int32_t)q->tokens.size();
        const auto branch = tok.encode(suffix, false, true);
        q->tokens.insert(q->tokens.end(), branch.begin(), branch.end());
    } else q->tokens = tok.encode(q->prompt, false, true);
    if (q->tokens.empty()) return bad("", "the decision prompt tokenized to an empty sequence");
    if (max_ctx > 0 && (int64_t)q->tokens.size() > max_ctx)
        return bad("", "the decision prompt exceeds the model context length");
    return RAD_OK;
}

}  /* namespace */

int parse_d1_request(const std::string& body, const ITokenizer& tok,
                     const std::string& model_id, int64_t max_ctx, int64_t max_questions,
                     DecisionRequest* out, ApiError* err, IMultimodal* mm) {
    *out = DecisionRequest{};
    bool over = false, deep = false;
    const json source = parse_json_bounded(body, max_ctx, &over, &deep);
    auto bad = [&](const std::string& field, const std::string& why) {
        *err = err_bad(field, why);
        return RAD_E_INVAL;
    };
    if (over || deep) return bad("", "the request exceeds the JSON value or nesting limit");
    if (source.is_discarded() || !source.is_object()) return bad("", "request body must be a JSON object");
    if (!only_keys(source, {"state", "questions", "model", "images", "stream"}, "", err)) return RAD_E_INVAL;
    if (source.contains("model")) {
        if (!source["model"].is_string()) return bad("model", "must be a string");
        if (source["model"] != model_id) return bad("model", "must name the loaded model: " + model_id);
    }
    if (source.contains("stream") && (!source["stream"].is_boolean() || source["stream"].get<bool>()))
        return bad("stream", "System One does not support streaming");
    if (source.contains("images")) {
        if (!source["images"].is_array()) return bad("images", "must be an array");
        for (size_t i = 0; i < source["images"].size(); ++i)
            if (!source["images"][i].is_string())
                return bad("images[" + std::to_string(i) + "]", "must be a base64 data-URL string");
        if (!source["images"].empty() && (!mm || !mm->accepts(MediaKind::Image))) {
            *err = err_not_implemented("System One images require a loaded and enabled image encoder");
            return RAD_E_UNSUPPORTED;
        }
    }
    if (!source.contains("state")) return bad("state", "must be provided (a string, JSON value, or null)");
    if (!source.contains("questions") || !source["questions"].is_object() || source["questions"].empty())
        return bad("questions", "must be a non-empty object of named questions");
    if (max_questions > 0 && (int64_t)source["questions"].size() > max_questions) {
        *err = err_overloaded("the request has more questions than the admission queue can hold");
        return RAD_E_INVAL;
    }
    PromptPrefix prefix;
    prefix.text = "<|startoftext|><|im_start|>user\n";
    if (source.contains("images") && !source["images"].empty()) {
        prefix.image_tokens = tok.encode(prefix.text, false, true);
        out->media = std::make_shared<mm::PromptMedia>();
        for (size_t i = 0; i < source["images"].size(); ++i) {
            const std::string field = "images[" + std::to_string(i) + "]";
            std::vector<uint8_t> bytes;
            int rc = decode_media_url(source["images"][i].get<std::string>(), field, &bytes, err);
            if (rc < 0) return rc;
            std::shared_ptr<mm::Item> item;
            std::string why;
            rc = mm->prepare(MediaKind::Image, bytes, MediaOpts{}, &item, &why);
            if (rc < 0 || !item || item->expand.empty()) {
                *err = err_bad(field, "the image could not be used: " +
                    (why.empty() ? std::string(rc < 0 ? rad_strerror(rc) : "empty image expansion") : why));
                return rc < 0 ? rc : RAD_E_INVAL;
            }
            if (item->expand.size() > (size_t)std::numeric_limits<int32_t>::max() ||
                prefix.image_tokens.size() > (size_t)std::numeric_limits<int32_t>::max() - item->expand.size() ||
                (max_ctx > 0 && prefix.image_tokens.size() + item->expand.size() > (size_t)max_ctx))
                return bad(field, "the expanded images exceed the model context length");
            mm::Placed placed;
            placed.item = std::move(item);
            placed.tok = (int32_t)prefix.image_tokens.size();
            prefix.image_tokens.insert(prefix.image_tokens.end(),
                                       placed.item->expand.begin(), placed.item->expand.end());
            out->media->items.push_back(std::move(placed));
            prefix.text += "<image>";
        }
        out->media->layout();
    }
    const json& state = source["state"];
    if (!state.is_null()) {
        prefix.after_images = state.is_string() ? state.get<std::string>() : state.dump(2);
        prefix.after_images += "\n\n\nQUESTION:\n";
        prefix.text += prefix.after_images;
    }
    for (const auto& item : source["questions"].items()) {
        DecisionQuestion q;
        const int rc = parse_question(item.key(), item.value(), prefix, tok, max_ctx,
                                      source["questions"].size() > 1, &q, err);
        if (rc < 0) return rc;
        out->input_tokens += (int64_t)q.tokens.size();
        out->questions.push_back(std::move(q));
    }
    return RAD_OK;
}

int d1_answer(const DecisionQuestion& q, const std::vector<float>& logits, json* out, ApiError* err) {
    auto failed = [&] {
        *err = ApiError{};
        err->status = 500;
        err->type = "internal_server_error";
        err->message = "the engine returned invalid decision logits";
        return RAD_E_STATE;
    };
    if (logits.size() != q.candidates.size() || q.groups.empty() || q.groups.size() != q.labels.size())
        return failed();
    for (float value : logits) if (!std::isfinite(value)) return failed();
    std::vector<double> probs;
    for (const auto& group : q.groups) {
        if (group.empty()) return failed();
        double value = -std::numeric_limits<double>::infinity();
        for (size_t index : group) {
            if (index >= logits.size()) return failed();
            value = std::max(value, (double)logits[index]);
        }
        probs.push_back(value);
    }
    const double peak = *std::max_element(probs.begin(), probs.end());
    double total = 0.0;
    for (double& value : probs) total += value = std::exp(value - peak);
    for (double& value : probs) value /= total;
    *out = json{{"type", q.type}};
    if (q.type == "noul") {
        (*out)["noul"] = probs[0];
        return RAD_OK;
    }
    const size_t best = (size_t)(std::max_element(probs.begin(), probs.end()) - probs.begin());
    json distribution = json::object();
    for (size_t i = 0; i < probs.size(); ++i) distribution[q.labels[i]] = probs[i];
    if (q.type == "choice") (*out)["choice"] = q.labels[best];
    else {
        if (q.descriptions.size() != probs.size()) return failed();
        double expected = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < probs.size(); ++i) {
            expected += (double)i * probs[i];
            legend[q.labels[i]] = q.descriptions[i];
        }
        (*out)["score"] = expected;
        (*out)["legend"] = std::move(legend);
    }
    (*out)["confidence"] = probs[best];
    (*out)["probabilities"] = std::move(distribution);
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
