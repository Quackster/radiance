#include "text/chat.h"
#include "text/tokenizer.h"

/* The lifted headers live here and nowhere else in core/. See chat.h for why. */
#include "llama/common/chat.h"
/* For jinja::enable_debug only -- see RADIANCE_DEBUG_JINJA below. */
#include "llama/common/jinja/runtime.h"
#include "shim/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>

namespace rad {

using ojson = nlohmann::ordered_json;


struct ChatTemplate::Impl {
    common_chat_templates_ptr tmpls;
    ChatCaps                  caps;
    std::string               src;
    bool                      add_bos = false, add_eos = false;
};

ChatTemplate::ChatTemplate() : impl_(std::make_unique<Impl>()) {}
ChatTemplate::~ChatTemplate() = default;

const char* ChatTemplate::chatml_source() {
    /* Byte-identical to CHATML_TEMPLATE_SRC in the lift. Duplicated rather than exported because
     * exporting it would mean editing the lifted header; the string is stable upstream, and if it
     * ever moves this is the single place that has to follow it. */
    return
        "{%- for message in messages -%}\n"
        "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "  {{- '<|im_start|>assistant\n' -}}\n"
        "{%- endif -%}";
}

int ChatTemplate::load(const std::string& jinja_source, const std::string& bos,
                       const std::string& eos, bool add_bos, bool add_eos) {
    /* The vendored files' LOG_DBG goes to stderr behind this switch. It has to be set HERE, where
     * the template is first touched: the auto-parser analyses the template inside apply() and
     * again when the reply format is derived (chatfmt.cpp), and a switch set any later comes too
     * late to trace a format the analyzer could not crack. */
    rad_vendor_log::verbose() = (int)log_level() >= (int)Log::Debug;

    /* THE TEMPLATE ENGINE'S OWN TRACE, and there is no other way to get at this one.
     * common_chat_templates_init runs a capability PROBE: it renders the template with a
     * synthetic tool and a synthetic tool call and watches which values were read. It SWALLOWS
     * every exception -- "ignore exceptions during capability analysis" -- so a template that
     * genuinely has no tool support and a template that THROWS on the probe both arrive here as
     * tools=0, and the difference is the difference between a model that cannot call tools and a
     * gap in the engine. Loud: every statement of every render, the serving ones included. */
    if (const char* dbg = std::getenv("RADIANCE_DEBUG_JINJA"); dbg && *dbg && *dbg != '0')
        jinja::enable_debug(true);

    const bool fallback = jinja_source.empty();
    if (fallback) {
        /* A model with no template is a model whose prompt format we are guessing. ChatML is the
         * best guess there is and it is still a guess, so it is said out loud once rather than
         * discovered later from output that reads fine and is formatted wrong. */
        RAD_WARN("chat: the model declares no chat template; falling back to ChatML. "
                 "If the model was not trained on ChatML its replies will be subtly wrong.");
    }
    const std::string src = fallback ? std::string(chatml_source()) : jinja_source;

    try {
        /* model = nullptr: everything common_chat_templates_init() would read off a llama_model
         * is passed in as an override instead. See vendor/shim/llama.h. */
        impl_->tmpls = common_chat_templates_init(nullptr, src, bos, eos);
    } catch (const std::exception& e) {
        RAD_ERR("chat: template did not compile: %s", e.what());
        impl_->tmpls.reset();
        return RAD_E_FORMAT;
    }
    if (!impl_->tmpls) return RAD_E_FORMAT;

    impl_->src = src;
    impl_->add_bos = add_bos;
    impl_->add_eos = add_eos;

    const std::map<std::string, bool> m = common_chat_templates_get_caps(impl_->tmpls.get());
    auto get = [&](const char* k, bool dflt) {
        auto it = m.find(k);
        return it == m.end() ? dflt : it->second;
    };
    impl_->caps.tools               = get("supports_tools", true);
    impl_->caps.tool_calls          = get("supports_tool_calls", true);
    impl_->caps.system_role         = get("supports_system_role", true);
    impl_->caps.parallel_tool_calls = get("supports_parallel_tool_calls", true);
    impl_->caps.preserve_reasoning  = get("supports_preserve_reasoning", false);
    impl_->caps.string_content      = get("supports_string_content", true);
    impl_->caps.typed_content       = get("supports_typed_content", false);
    impl_->caps.object_arguments    = get("supports_object_arguments", false);

    RAD_INFO("chat: template loaded (%zu bytes)%s; tools=%d tool_calls=%d system=%d thinking=%d",
             src.size(), fallback ? " [ChatML fallback]" : "",
             (int)impl_->caps.tools, (int)impl_->caps.tool_calls,
             (int)impl_->caps.system_role,
             (int)common_chat_templates_support_enable_thinking(impl_->tmpls.get()));
    return RAD_OK;
}

int ChatTemplate::load_from_vocab(const Vocab& v) {
    /* The template references bos_token and eos_token by TEXT, not by id, so what goes in is the
     * piece -- and a control token's piece is its literal spelling, which is what Vocab::text()
     * returns for it. */
    const std::string bos = v.bos() >= 0 ? v.text(v.bos()) : std::string();
    const std::string eos = v.eos() >= 0 ? v.text(v.eos()) : std::string();
    return load(v.chat_template(), bos, eos, v.add_bos(), v.add_eos());
}

bool ChatTemplate::ready() const { return impl_ && impl_->tmpls != nullptr; }
const ChatCaps& ChatTemplate::caps() const { return impl_->caps; }
const std::string& ChatTemplate::source() const { return impl_->src; }

/* HOW DEEPLY A CLIENT'S SCHEMA MAY NEST, in objects and arrays. The tools' parameter schemas and
 * a response format's schema reach the JSON-schema converter, which recurses once per level and
 * names each rule after its whole path, so its output grows with the square of the depth; a few
 * thousand levels exhaust the memory or the stack of the thread rendering the request. At this
 * depth it still converts in tens of milliseconds, far deeper than a schema describing data
 * nests. grammar.cpp bounds the same converter on the path that bypasses the template. */
static constexpr size_t kMaxSchemaDepth = 512;

/* Walked with an explicit stack, because the values it exists to catch are the ones too deep to
 * recurse over. */
static size_t json_depth(const ojson& root) {
    std::vector<std::pair<const ojson*, size_t>> todo{ { &root, 1 } };
    size_t deepest = 0;
    while (!todo.empty()) {
        const auto [j, d] = todo.back();
        todo.pop_back();
        if (!j->is_structured()) continue;
        deepest = std::max(deepest, d);
        for (const auto& c : *j) todo.push_back({ &c, d + 1 });
    }
    return deepest;
}

int ChatTemplate::apply(const ojson& messages, const ojson& tools, const ChatOptions& opt,
                        ChatPrompt* out) const {
    if (!ready()) {
        RAD_ERR("chat: apply() before load()");
        return RAD_E_STATE;
    }
    if (!out) return RAD_E_INVAL;

    /* Checked before either schema reaches the converter. The response format arrives as text;
     * one that does not parse is left for the template to refuse in its own words. */
    {
        const ojson schema = opt.json_schema.empty()
                           ? ojson()
                           : ojson::parse(opt.json_schema, nullptr, /*allow_exceptions=*/false);
        if (json_depth(tools) > kMaxSchemaDepth || json_depth(schema) > kMaxSchemaDepth) {
            out->error = "a tool or response schema nests more than " +
                         std::to_string(kMaxSchemaDepth) + " levels deep";
            RAD_WARN("chat: %s", out->error.c_str());
            return RAD_E_INVAL;
        }
    }

    common_chat_templates_inputs in;
    try {
        in.messages = common_chat_msgs_parse_oaicompat(messages);
        if (!tools.is_null() && tools.is_array() && !tools.empty())
            in.tools = common_chat_tools_parse_oaicompat(tools);
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(opt.tool_choice);
    } catch (const std::exception& e) {
        /* A malformed request, not a broken engine. It must come back as a 400 rather than as a
         * throw through the request loop. */
        RAD_WARN("chat: request did not parse: %s", e.what());
        out->error = e.what();
        return RAD_E_INVAL;
    }
    if (!in.tools.empty() && !impl_->caps.tools) {
        RAD_WARN("chat: the model's template declares no tool support; refusing the request "
                 "rather than dropping the tools silently");
        out->error = "this model's chat template declares no tool support";
        return RAD_E_UNSUPPORTED;
    }

    in.grammar               = opt.grammar;
    in.json_schema           = opt.json_schema;
    in.add_generation_prompt = opt.add_generation_prompt;
    in.use_jinja             = true;
    in.parallel_tool_calls   = opt.parallel_tool_calls;
    in.enable_thinking       = opt.enable_thinking;
    in.chat_template_kwargs  = opt.template_kwargs;
    in.add_bos               = opt.add_bos || impl_->add_bos;
    in.add_eos               = opt.add_eos || impl_->add_eos;
    in.reasoning_format      = common_reasoning_format_from_name(opt.reasoning_format);
    in.continue_final_message =
        opt.continue_final.empty()          ? COMMON_CHAT_CONTINUATION_NONE
        : opt.continue_final == "content"   ? COMMON_CHAT_CONTINUATION_CONTENT
        : opt.continue_final == "reasoning_content" ? COMMON_CHAT_CONTINUATION_REASONING
                                            : COMMON_CHAT_CONTINUATION_AUTO;

    common_chat_params p;
    try {
        p = common_chat_templates_apply(impl_->tmpls.get(), in);
    } catch (const std::exception& e) {
        if (!opt.quiet) RAD_ERR("chat: template refused the request: %s", e.what());
        out->error = e.what();
        return RAD_E_UNSUPPORTED;
    }

    out->prompt            = std::move(p.prompt);
    out->generation_prompt = std::move(p.generation_prompt);
    out->grammar           = std::move(p.grammar);
    out->grammar_lazy      = p.grammar_lazy;
    out->preserved_tokens  = std::move(p.preserved_tokens);
    out->additional_stops  = std::move(p.additional_stops);
    out->supports_thinking = p.supports_thinking;
    out->thinking_start_tag = std::move(p.thinking_start_tag);
    out->thinking_end_tag   = std::move(p.thinking_end_tag);
    out->format            = (int)p.format;
    out->parser            = std::move(p.parser);
    out->grammar_triggers.clear();
    for (const common_grammar_trigger& t : p.grammar_triggers)
        out->grammar_triggers.push_back({ (int)t.type, t.value });

    RAD_DEBUG("chat: rendered %zu bytes, format %s, grammar %zu bytes%s, %zu extra stop(s)",
              out->prompt.size(), common_chat_format_name(p.format),
              out->grammar.size(), out->grammar_lazy ? " (lazy)" : "",
              out->additional_stops.size());
    if ((int)log_level() >= (int)Log::Trace) {
        /* THE PROMPT ITSELF, at Trace and nowhere else. A chat endpoint that answers fluently in
         * the wrong format looks exactly like a chat endpoint that works, and the only way to tell
         * the two apart is to read what the template actually produced -- control tokens included,
         * which is why this prints the raw bytes rather than the detokenised form. It is at Trace
         * because it is user content and does not belong in an ordinary server log. */
        RAD_TRACE("chat: prompt <<<%s>>>", out->prompt.c_str());
        if (!out->grammar.empty()) RAD_TRACE("chat: grammar <<<%s>>>", out->grammar.c_str());
        for (const auto& s : out->additional_stops) RAD_TRACE("chat: extra stop <<<%s>>>", s.c_str());
    }
    return RAD_OK;
}

}  /* namespace rad */
