/* chat.h -- rendering a message list to a prompt, and what the template can do.
 *
 * This is a thin face over the lifted `common_chat_*` API (vendor/llama/common/chat.*, jinja/,
 * see vendor/UPSTREAM). It does three things the lift does not:
 *
 *   1. ERRORS ARE VALUES. Upstream throws -- a bad template, a message the template rejects, a
 *      malformed OAI request are all std::runtime_error, and a throw crossing the server's
 *      request loop is not a failure mode this project accepts. Everything here returns a
 *      negative RAD_E_* and logs once.
 *   2. It takes the template out of the .rad instead of out of a llama_model. RadVocabHeader
 *      carries the chat template, the BOS/EOS spellings and the add_bos/add_eos flags, which is
 *      exactly the four things common_chat_templates_init() would otherwise read off a model.
 *   3. It keeps the lifted headers out of the rest of core/. Only chat.cpp, chatfmt.cpp and
 *      chatparse.cpp include them, the last only to resolve schema types when a request's tools
 *      are bound, so a vendor refresh cannot ripple into the server.
 *
 * Messages and tools arrive as OpenAI-shaped JSON, because that is what the server has and the
 * lift already knows how to read it. Converting to a struct here and back inside would be a
 * translation nobody asked for.
 */
#pragma once
#include "rad_internal.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace rad {

class Vocab;

/* What the template supports, PROBED rather than assumed -- jinja/caps.cpp renders the template
 * against synthetic inputs and watches what it does with them (spec §12, "template capability
 * probe"). The server reports this on /props, and the message adapter uses it to decide whether
 * to hand the template a string content or a typed content array. */
struct ChatCaps {
    bool tools = true;
    bool tool_calls = true;
    bool system_role = true;
    bool parallel_tool_calls = true;
    bool preserve_reasoning = false;
    bool string_content = true;
    bool typed_content = false;
    bool object_arguments = false;

};

struct ChatOptions {
    bool        add_generation_prompt = true;
    bool        enable_thinking = true;
    bool        parallel_tool_calls = false;
    std::string tool_choice = "auto";           /* auto | none | required */
    std::string reasoning_format = "auto";      /* none | auto | deepseek | deepseek-legacy */
    std::string continue_final;                 /* "" | auto | content | reasoning_content */
    std::string grammar;                        /* a user-supplied GBNF, or empty */
    std::string json_schema;                    /* response_format json_schema, or empty */
    bool        add_bos = false;
    bool        add_eos = false;
    std::map<std::string, std::string> template_kwargs;
    bool        quiet = false;                  /* a refusal is not logged: a startup probe */
};

/* Everything the scheduler and the sampler need for one request, produced in one pass. */
struct ChatPrompt {
    std::string prompt;
    std::string generation_prompt;    /* the assistant prefix already inside `prompt` */

    std::string grammar;              /* GBNF; empty when the request is unconstrained */
    bool        grammar_lazy = false;
    /* Triggers for a lazy grammar: the grammar starts applying only once one of these is seen.
     * Kept as (type, text) so the sampler can tell a literal word from a pattern. */
    struct GrammarTrigger { int type; std::string value; };
    std::vector<GrammarTrigger> grammar_triggers;

    std::vector<std::string> preserved_tokens;    /* must survive the sampler's bans */
    std::vector<std::string> additional_stops;

    bool        supports_thinking = false;
    std::string thinking_start_tag, thinking_end_tag;

    /* The auto-parser's own verdict on this render: `format` is a common_chat_format value and
     * deliberately opaque here, `parser` the PEG grammar it derived. The engine reads replies
     * with chatparse.h instead; these are kept for the differential tests that compare the two. */
    int         format = 0;
    std::string parser;

    /* WHY THE TEMPLATE REFUSED, in the template's own words.
     *
     * A jinja `raise_exception` carries the one sentence that says what the caller got wrong --
     * "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
     * -- and it is carried out to the caller here rather than only into the log, so a 400 names
     * the field at fault instead of saying only that the template refused the request. Filled
     * only on a failure return; empty otherwise.
     *
     * It is the TEMPLATE'S text and therefore model-controlled, not attacker-controlled: it comes
     * from the container's own jinja, which the operator chose along with the weights. */
    std::string error;
};

class ChatTemplate {
public:
    ChatTemplate();
    ~ChatTemplate();

    /* Compile a Jinja source. `bos`/`eos` are the token TEXTS, which templates reference as
     * `bos_token` / `eos_token`. An empty source loads the ChatML fallback and says so at Warn:
     * a model with no template is a model whose prompt format we are guessing, and guessing
     * quietly is how a chat endpoint produces fluent nonsense. */
    int load(const std::string& jinja_source, const std::string& bos, const std::string& eos,
             bool add_bos = false, bool add_eos = false);
    /* The same, from a loaded vocab: template, bos text, eos text and the add flags all come off
     * RadVocabHeader. This is the path the engine uses. */
    int load_from_vocab(const Vocab& v);

    bool ready() const;
    const ChatCaps& caps() const;
    const std::string& source() const;

    /* Render. `messages` and `tools` are OpenAI-shaped JSON (`tools` may be null or an empty
     * array). Returns RAD_OK, or RAD_E_INVAL when the request does not parse, or
     * RAD_E_UNSUPPORTED when the template refuses what was asked of it -- a system message to a
     * template with no system role, tools to a template that has none. */
    int apply(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
              const ChatOptions& opt, ChatPrompt* out) const;

    /* ChatML, the fallback. Named rather than inlined so the server can report which template it
     * is running. */
    static const char* chatml_source();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  /* namespace rad */
