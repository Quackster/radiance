/* loader.cpp -- dlopen, the ABI gate, the configured hierarchy, and the registration of one
 * plugin's schemas and kernel rows. spec.md §2, §18.
 */
#include "registry.h"
#include "../device/device.h"
#include "../format/encoding.h"

#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <climits>
#include <cerrno>
#include <cstdarg>
#include <cstring>

namespace rad {

namespace {

/* dlsym returns void*, which is not a function pointer in standard C++ and is one everywhere this
 * engine will ever run. The cast is isolated here so the rest of the file reads. */
template <class Fn>
Fn dl_fn(void* h, const char* name) {
    dlerror();
    return reinterpret_cast<Fn>(dlsym(h, name));
}

std::string stem_of(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base.size() > 3 && base.compare(base.size() - 3, 3, ".so") == 0)
        base.resize(base.size() - 3);
    if (base.compare(0, 3, "lib") == 0 && base.size() > 3) base.erase(0, 3);
    return base;
}

const char* or_empty(const char* s) { return s ? s : ""; }

/* Does a plugin's declared coverage include this device's architecture? The field is a list, so a
 * binary built for several is one entry per name, separated by spaces or commas. */
bool target_covers(const std::string& declared, const char* arch) {
    for (size_t i = 0; i < declared.size(); ) {
        size_t e = declared.find_first_of(" ,;", i);
        if (e == std::string::npos) e = declared.size();
        if (e > i && declared.compare(i, e - i, arch) == 0) return true;
        i = e + 1;
    }
    return false;
}

/* Two schemas for one op name are the SAME schema when every positional slot means the same thing.
 * Parameter keys, types and requiredness are load-bearing: they are the geometry the selector
 * matches on. Operand roles, count and optionality are load-bearing: arguments are positional.
 * Operand NAMES are not -- the ABI says they are for diagnostics -- so two plugins spelling slot 1
 * "b" and "w_gate" agree, and refusing that would be refusing a synonym. */
bool schema_same(const RadOpSchema& a, const RadOpSchema& b, std::string* diff) {
    if (a.n_params != b.n_params) {
        *diff = fmt("%d parameters vs %d", a.n_params, b.n_params);
        return false;
    }
    for (int i = 0; i < a.n_params; ++i) {
        const RadParamSpec& x = a.params[i];
        const RadParamSpec& y = b.params[i];
        if (std::strcmp(or_empty(x.key), or_empty(y.key)) != 0) {
            *diff = fmt("parameter %d is '%s' vs '%s'", i, or_empty(x.key), or_empty(y.key));
            return false;
        }
        if (x.type != y.type) {
            *diff = fmt("parameter '%s' is %s vs %s", or_empty(x.key),
                        x.type == RAD_P_STR ? "a string" : "an integer",
                        y.type == RAD_P_STR ? "a string" : "an integer");
            return false;
        }
        if (x.required != y.required) {
            *diff = fmt("parameter '%s' is %s vs %s", or_empty(x.key),
                        x.required ? "required" : "optional", y.required ? "required" : "optional");
            return false;
        }
    }
    if (a.n_operands != b.n_operands) {
        *diff = fmt("%d operands vs %d", a.n_operands, b.n_operands);
        return false;
    }
    for (int i = 0; i < a.n_operands; ++i) {
        if (a.operands[i].role != b.operands[i].role ||
            a.operands[i].optional != b.operands[i].optional) {
            *diff = fmt("operand %d ('%s' / '%s') has a different role", i,
                        or_empty(a.operands[i].name), or_empty(b.operands[i].name));
            return false;
        }
    }
    return true;
}

}  /* namespace */

/* ------------------------------------------------------------------ diagnostics */
void Registry::err_(const char* f, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(msg, sizeof msg, f, ap);
    va_end(ap);
    load_errors_.emplace_back(msg);
    RAD_ERR("%s", msg);
}

/* ------------------------------------------------------------------ staging */
/* dlopen and read the plugin's identity, without registering anything. Two phases, because the
 * hierarchy is ordered by plugin NAME and the name lives inside the .so: the whole directory must
 * be open before anything can be registered, and registration order is what fixes an op's schema. */
int Registry::stage_(const std::string& path, Staged* out) {
    out->path = path;
    out->file = stem_of(path);

    /* RTLD_LOCAL is not a preference. Every kernel plugin exports the same four symbol names --
     * rad_kernel_count and friends -- so RTLD_GLOBAL would let the first plugin loaded answer for
     * every plugin after it, and the failure would look like a plugin whose kernels vanished. We
     * dlsym against the handle instead, which is exactly what RTLD_LOCAL is for. */
    out->code_from = rad::dev_code_mark();
    out->handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    out->code_to = rad::dev_code_mark();
    if (!out->handle) {
        err_("plugin '%s' failed to load: %s", path.c_str(), or_empty(dlerror()));
        return RAD_E_IO;
    }

    auto abi = dl_fn<uint32_t (*)(void)>(out->handle, "rad_plugin_abi_version");
    if (!abi) {
        err_("'%s' is not a radiance plugin: it exports no rad_plugin_abi_version", path.c_str());
        unstage_(*out);
        return RAD_E_ABI;
    }
    const uint32_t v = abi();
    if (v != RAD_ABI_VERSION) {
        /* Named by FILE, not by RadPluginInfo::name: the struct is exactly the thing that may have
         * changed shape between ABI versions, so reading it here would be reading a different
         * layout than the one this build was compiled against. Both versions are printed because
         * "ABI mismatch" alone does not tell an operator which side is stale. */
        err_("plugin '%s' was built against ABI version %u; this engine is version %u -- refused",
             path.c_str(), v, (unsigned)RAD_ABI_VERSION);
        unstage_(*out);
        return RAD_E_ABI;
    }

    auto info_fn = dl_fn<const RadPluginInfo* (*)(void)>(out->handle, "rad_plugin_info");
    if (!info_fn || !(out->info = info_fn())) {
        err_("plugin '%s' exports no usable rad_plugin_info", path.c_str());
        unstage_(*out);
        return RAD_E_INVAL;
    }
    out->name = out->info->name && out->info->name[0] ? out->info->name : out->file;
    if (auto ref = dl_fn<RadPluginReferenceFn>(out->handle, "rad_plugin_reference"))
        out->reference = out->info->kind == RAD_PLUGIN_KERNEL && ref() != 0;
    return RAD_OK;
}

void Registry::unstage_(Staged& s) {
    if (s.handle) dlclose(s.handle);
    s.handle = nullptr;
    s.info = nullptr;
}

/* ------------------------------------------------------------------ registration */
int Registry::commit_(Staged& s, int order) {
    for (const Plugin& p : plugins_) {
        if (p.name == s.name) {
            err_("two plugins claim the name '%s': %s and %s -- refusing the second",
                 s.name.c_str(), p.path.c_str(), s.path.c_str());
            unstage_(s);
            return RAD_E_DUPLICATE;
        }
    }

    Plugin p;
    p.name         = s.name;
    p.file         = s.file;
    p.path         = s.path;
    p.version      = or_empty(s.info->version);
    p.description  = or_empty(s.info->description);
    p.build_target = or_empty(s.info->build_target);
    p.kind         = s.info->kind;
    p.reference    = s.reference;
    p.order        = order;
    p.handle       = s.handle;
    p.close        = dl_fn<RadPluginCloseFn>(s.handle, "rad_plugin_close");

    std::vector<const RadOpSchema*> staged_schemas;
    std::vector<std::pair<const RadKernelInfo*, int>> staged_rows;   /* the row and its index */
    std::vector<const RadQuantizerInfo*> staged_quants;
    int n_declared_schemas = 0;

    if (p.kind == RAD_PLUGIN_ARCH) {
        auto id_fn = dl_fn<const char* (*)(void)>(s.handle, "rad_arch_id");
        p.arch_declare = dl_fn<int (*)(RadBuilder*, const RadModelMeta*, const RadBuildCtx*)>(
                             s.handle, "rad_arch_declare");
        p.arch_step = dl_fn<void (*)(RadCtx*, const RadBatch*)>(s.handle, "rad_arch_step");
        if (!id_fn || !p.arch_declare || !p.arch_step) {
            err_("architecture plugin '%s' is missing rad_arch_id, rad_arch_declare or "
                 "rad_arch_step", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        p.arch_id = or_empty(id_fn());
        if (p.arch_id.empty()) {
            err_("architecture plugin '%s' returns an empty architecture id", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        /* The second half of the key. OPTIONAL: a plugin written before this export existed serves
         * an unquantised container, which is what "" means, so its absence is not an error. */
        if (auto q_fn = dl_fn<const char* (*)(void)>(s.handle, "rad_arch_quant"))
            p.arch_quant = or_empty(q_fn());

        /* ALSO OPTIONAL, and for the same reason: it answers what the core needs to know before
         * it can build the RadBuildCtx that declare runs against. Absent means every answer is
         * the zero -- see rad_builder.h. */
        p.arch_probe = dl_fn<int (*)(const RadModelMeta*, RadArchProbe*)>(s.handle,
                                                                         "rad_arch_probe");
        /* AND THE REPLY FORMAT, optional for the same reason: a plugin that says nothing has its
         * model's format read off the chat template. */
        p.arch_chat_format = dl_fn<const RadChatFormat* (*)(const RadModelMeta*)>(
            s.handle, "rad_arch_chat_format");

        /* One plugin implements one architecture id AT ONE QUANTISATION, and no two may claim the
         * same pair: the pair is what selects the plugin, so two claimants is a coin toss over
         * which graph a model gets built from (§2.4). Two plugins sharing an id at different
         * quantisations is the NORMAL case -- qwen35_bf16 and qwen35_fp8 are the same architecture
         * over two weight formats -- and refusing that would make §2.4 unimplementable. */
        for (const Plugin& q : plugins_) {
            if (q.arch_id == p.arch_id && q.arch_quant == p.arch_quant) {
                err_("architecture id '%s' at quantisation '%s' is claimed by both '%s' and '%s' "
                     "-- refusing the second", p.arch_id.c_str(),
                     p.arch_quant.empty() ? "(unquantised)" : p.arch_quant.c_str(),
                     q.name.c_str(), p.name.c_str());
                unstage_(s);
                return RAD_E_DUPLICATE;
            }
        }
    } else if (p.kind == RAD_PLUGIN_KERNEL) {
        auto sc_n  = dl_fn<int (*)(void)>(s.handle, "rad_kernel_schema_count");
        auto sc_at = dl_fn<const RadOpSchema* (*)(int)>(s.handle, "rad_kernel_schema_at");
        auto k_n   = dl_fn<int (*)(void)>(s.handle, "rad_kernel_count");
        auto k_at  = dl_fn<const RadKernelInfo* (*)(int)>(s.handle, "rad_kernel_at");
        if (!sc_n || !sc_at || !k_n || !k_at) {
            err_("kernel plugin '%s' is missing rad_kernel_schema_count/at or "
                 "rad_kernel_count/at", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }

        /* Schemas are checked WHOLE before any of them is committed. A plugin that conflicts on
         * its fourth op must not leave its first three registered under its name -- a half-loaded
         * plugin is a state nobody can reason about afterwards. */
        n_declared_schemas = sc_n();
        for (int i = 0; i < n_declared_schemas; ++i) {
            const RadOpSchema* sch = sc_at(i);
            if (!sch || !sch->op || !sch->op[0]) {
                err_("kernel plugin '%s' declares a schema with no op name at index %d",
                     p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            std::string diff;
            auto it = schemas_.find(sch->op);
            if (it != schemas_.end()) {
                if (!schema_same(*it->second.schema, *sch, &diff)) {
                    /* Refused at load with both plugins named. Since arguments are positional, two
                     * disagreeing schemas would make the same call mean different things depending
                     * on which plugin won selection, and that is not a diagnosable failure -- it
                     * is silent numerical garbage (§2.3). The whole plugin is refused, not just
                     * the op: a plugin built against a different vocabulary is stale everywhere. */
                    err_("op '%s' has two different schemas: '%s' fixed it first, '%s' disagrees "
                         "(%s) -- refusing '%s'",
                         sch->op, it->second.plugin.c_str(), p.name.c_str(), diff.c_str(),
                         p.name.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
                continue;   /* same-shape re-declaration: expected, every plugin declares its ops */
            }
            for (const RadOpSchema* q : staged_schemas) {
                if (std::strcmp(q->op, sch->op) == 0 && !schema_same(*q, *sch, &diff)) {
                    err_("kernel plugin '%s' declares op '%s' twice with different schemas (%s)",
                         p.name.c_str(), sch->op, diff.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            }
            staged_schemas.push_back(sch);
        }

        for (int i = 0, n = k_n(); i < n; ++i) {
            const RadKernelInfo* k = k_at(i);
            if (!k || !k->name || !k->op || !k->op[0]) {
                err_("kernel plugin '%s' declares a kernel with no name or op at index %d",
                     p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* A row with no entry point is a row that resolves and then cannot run, which is the
             * one failure this layer exists to make impossible. */
            if (!k->launch) {
                err_("kernel '%s' in plugin '%s' has no launch function", k->name, p.name.c_str());
                unstage_(s);
                return RAD_E_INVAL;
            }
            if (k->domain != RAD_DOMAIN_DEVICE && k->domain != RAD_DOMAIN_HOST) {
                err_("kernel '%s' in plugin '%s' declares domain %d, which is neither device nor "
                     "host", k->name, p.name.c_str(), k->domain);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* EVERY TUNABLE AXIS CARRIES A DEFAULT, AND THE DEFAULT IS ONE OF ITS OWN VALUES.
             *
             * This is what makes an untuned install fast rather than broken (spec §15): a machine
             * with no cache, a shape nobody measured, a cache from a different card -- all of them
             * land on the defaults, and there is no other fallback beneath them. A kernel that
             * declares an axis without a usable default has no answer at all for those cases, so
             * it is refused here rather than resolving and then launching a zero tile.
             *
             * Checked at LOAD and not at first use, for the same reason the missing launch
             * function above is: a library author finds out when the plugin loads, not when some
             * shape three hours into a serve happens to miss the cache. */
            for (int v = 0; v < k->n_tunables; ++v) {
                const RadTunable& t = k->tunables[v];
                if (!t.key || !t.key[0] || !t.values || t.n_values <= 0) {
                    err_("kernel '%s' in plugin '%s' declares tunable %d with no key or no legal "
                         "values", k->name, p.name.c_str(), v);
                    unstage_(s);
                    return RAD_E_INVAL;
                }
                bool ok = false;
                for (int j = 0; j < t.n_values; ++j) if (t.values[j] == t.deflt) ok = true;
                if (!ok) {
                    err_("kernel '%s' in plugin '%s': tunable '%s' defaults to %lld, which is not "
                         "one of the values it declares. The default is what runs when this "
                         "machine has no measurement for a shape, so it has to be a value the "
                         "kernel can actually launch",
                         k->name, p.name.c_str(), t.key, (long long)t.deflt);
                    unstage_(s);
                    return RAD_E_INVAL;
                }
                for (int w = 0; w < v; ++w)
                    if (k->tunables[w].key && std::strcmp(k->tunables[w].key, t.key) == 0) {
                        err_("kernel '%s' in plugin '%s' declares tunable '%s' twice", k->name,
                             p.name.c_str(), t.key);
                        unstage_(s);
                        return RAD_E_INVAL;
                    }
            }
            staged_rows.emplace_back(k, i);
        }

        /* A KERNEL LIBRARY WITH NO CODE FOR THIS CARD IS LEFT OUT OF THE CARD'S SELECTION. Its
         * device rows would otherwise resolve -- a row's constraints say nothing about which card
         * it was compiled for -- and the first launch would fail, so an installation carrying
         * libraries for several cards would serve none of them but the first in the hierarchy.
         * What the library carries is read out of the fat binaries its dlopen registered, which is
         * what the launch itself will consult (core/device/aql_code.cpp), not out of
         * build_target, which is prose. Its host rows run anywhere and stay. On the host backend
         * an engine that serves says no GPU code runs at all (set_no_device_code). */
        int n_device = 0;
        for (const auto& r : staged_rows) n_device += r.first->domain == RAD_DOMAIN_DEVICE;
        std::string why;
        if (no_device_code_)
            why = "no card is visible, so this process runs on the host backend and no GPU code "
                  "object runs here";
        if (n_device > 0 && s.code_to > s.code_from &&
            (no_device_code_ || rad::dev_code_runs(s.code_from, s.code_to, code_device_, &why) == 0)) {
            std::vector<std::pair<const RadKernelInfo*, int>> keep;
            for (const auto& r : staged_rows)
                if (r.first->domain != RAD_DOMAIN_DEVICE) keep.push_back(r);
            RAD_WARN("kernel plugin '%s' has no device code for device %d: %s. Its %d device "
                     "kernel(s) are left out of selection here%s.", p.name.c_str(), code_device_,
                     why.c_str(), n_device,
                     keep.empty() ? "" : fmt(" and its %zu host kernel(s) stay", keep.size()).c_str());
            staged_rows.swap(keep);
            p.device_code_absent = true;
        }
    } else if (p.kind == RAD_PLUGIN_QUANT) {
        auto q_n  = dl_fn<int (*)(void)>(s.handle, "rad_quant_count");
        auto q_at = dl_fn<const RadQuantizerInfo* (*)(int)>(s.handle, "rad_quant_at");
        if (!q_n || !q_at) {
            err_("quantiser plugin '%s' is missing rad_quant_count or rad_quant_at",
                 p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        /* Checked WHOLE before any is registered, for the reason the kernel schemas are: a plugin
         * refused on its third quantiser must not leave the first two answering to a recipe. */
        for (int i = 0, n = q_n(); i < n; ++i) {
            const RadQuantizerInfo* q = q_at(i);
            if (!q || !q->name || !q->name[0] || !q->encoding || !q->quantize) {
                err_("quantiser plugin '%s' declares a quantiser with no name, no encoding hook or "
                     "no quantize hook at index %d", p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* A NAME IS ONE QUANTISER. A recipe that names it would otherwise mean different
             * numbers depending on which plugin loaded first. */
            auto clash = [&](const char* other) { return std::strcmp(other, q->name) == 0; };
            for (const QuantRow& r : quants_)
                if (clash(r.info->name)) {
                    err_("quantiser '%s' is offered by both '%s' and '%s' -- refusing '%s'",
                         q->name, r.plugin.c_str(), p.name.c_str(), p.name.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            for (const RadQuantizerInfo* o : staged_quants)
                if (clash(o->name)) {
                    err_("quantiser plugin '%s' declares '%s' twice", p.name.c_str(), q->name);
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            staged_quants.push_back(q);
        }
    } else {
        err_("plugin '%s' declares kind %u, which is not a kernel, architecture or quantiser "
             "plugin", p.name.c_str(), p.kind);
        unstage_(s);
        return RAD_E_INVAL;
    }

    /* Once after the plugin is loaded and before any selection (rad_abi.h). A plugin that cannot
     * open -- no device, no config table -- is refused rather than asked to select. */
    if (auto open = dl_fn<RadPluginOpenFn>(s.handle, "rad_plugin_open")) {
        int st = open();
        /* RAD_E_UNSUPPORTED is a plugin saying this machine is not one it serves -- a library for
         * another card, in an installation that ships several. That is a skip, said once, and
         * not a failure: the next library in the hierarchy is the one that serves this card. */
        if (st == RAD_E_UNSUPPORTED) {
            RAD_INFO("plugin '%s' declines this machine and is not loaded", p.name.c_str());
            unstage_(s);
            return 1;
        }
        if (st < 0) {
            err_("plugin '%s' refused to open: %s", p.name.c_str(), rad_strerror(st));
            unstage_(s);
            return st;
        }
    }

    /* What the plugin DECLARES, not what it fixed: a plugin below another one re-declares the same
     * ops with the same shapes, and reporting 0 there would read as a plugin that declares
     * nothing. Which plugin fixed a given op is schema_owner()'s answer, not a count. */
    p.n_schemas = n_declared_schemas;
    p.n_kernels = (int)staged_rows.size();
    for (const RadOpSchema* sch : staged_schemas) schemas_[sch->op] = SchemaEntry{sch, p.name};
    for (const RadQuantizerInfo* q : staged_quants) {
        quants_.push_back(QuantRow{ q, p.name });
        /* Its own schemes are read through it, by everything that decodes a weight. */
        if (q->decode) enc_add_decoder(q->decode);
    }
    const auto concurrent = dl_fn<RadKernelConcurrentFn>(s.handle, "rad_kernel_concurrent");
    for (const auto& sr : staged_rows) {
        KernelRow row;
        row.info         = sr.first;
        row.plugin       = p.name;
        row.plugin_order = order;
        row.index        = sr.second;
        row.concurrent   = concurrent && concurrent(sr.second) != 0;
        row.reference    = p.reference;
        rows_.push_back(row);
    }

    /* A KERNEL LIBRARY BUILT FOR ANOTHER CARD FAILS AT EVERY LAUNCH, AND THIS IS THE ONE PLACE IT
     * CAN BE NAMED ONCE. An installed .so travelling to a machine it was not compiled for is
     * otherwise invisible until the first dispatch, where it surfaces as a driver error carrying a
     * kernel name and nothing about why.
     *
     * A WARNING AND NOT A REFUSAL, deliberately. build_target is text the plugin's author writes;
     * it is not derived from what the binary actually contains, so a library covering several
     * architectures may still name only one, and refusing on it would be refusing on a claim the
     * engine cannot verify. The launch is what is authoritative. This predicts it by name.
     *
     * "host" and "" are not claims about a device: the first is a library of CPU kernels, which
     * run anywhere, and the second is a plugin that declines to claim anything. */
    if (p.kind == RAD_PLUGIN_KERNEL && s.code_to == s.code_from && !p.build_target.empty() &&
        p.build_target != "host") {
        RadDeviceProps dp{};
        if (rad_dev_count() > code_device_ && rad_dev_props(code_device_, &dp) == RAD_OK &&
            dp.arch[0] && !target_covers(p.build_target, dp.arch))
            RAD_WARN("kernel plugin '%s' declares it was built for %s and this machine reports %s. "
                     "Its kernels will fail to launch; build it for %s, or leave it out with "
                     "--kernels.",
                     p.name.c_str(), p.build_target.c_str(), dp.arch, dp.arch);
    }

    RAD_DEBUG("plugin %d: %s %s (%s) -- %d kernels, %d schemas, %zu fat binaries, %s", order,
              p.name.c_str(), p.version.c_str(), p.path.c_str(), p.n_kernels, p.n_schemas,
              s.code_to - s.code_from, p.build_target.empty() ? "portable" : p.build_target.c_str());
    plugins_.push_back(std::move(p));
    index_rows_();
    s.handle = nullptr;   /* the Plugin owns it now */
    return RAD_OK;
}

/* The per-op index select() walks, in (hierarchy order, declaration order). Rebuilt whole on every
 * registration: loading is startup and the row count is in the hundreds, so the simple thing is
 * also the fast enough thing, and an index that can never be stale is one fewer invariant. */
void Registry::index_rows_() {
    by_op_.clear();
    for (const KernelRow& r : rows_) by_op_[r.info->op].push_back(&r);
    for (auto& kv : by_op_) {
        std::stable_sort(kv.second.begin(), kv.second.end(),
                         [](const KernelRow* a, const KernelRow* b) {
                             if (a->plugin_order != b->plugin_order)
                                 return a->plugin_order < b->plugin_order;
                             return a->index < b->index;
                         });
    }
}

/* ------------------------------------------------------------------ the public loading API */
int Registry::load_plugin(const std::string& path, int order) {
    Staged s;
    int st = stage_(path, &s);
    if (st < 0) return st;
    return commit_(s, order);
}

int Registry::load_dir(const std::string& dir, const std::vector<std::string>& hierarchy) {
    return load_dirs({ dir }, hierarchy);
}

int Registry::load_dirs(const std::vector<std::string>& dirs,
                        const std::vector<std::string>& hierarchy) {
    int first_error = RAD_OK;

    /* Directory by directory in the order given, and within one in filename order, so a set of
     * plugins nobody named still loads the same way twice. A file whose name an earlier directory
     * already supplied is SHADOWED rather than loaded: the first directory is the one the operator
     * put first -- their own plugins ahead of an installation's -- and loading both would be two
     * plugins claiming one name, which is refused. */
    std::vector<std::string> files, stems;
    std::string where;
    for (const std::string& dir : dirs) {
        DIR* d = opendir(dir.c_str());
        if (!d) {
            err_("plugin directory '%s': %s", dir.c_str(), std::strerror(errno));
            return RAD_E_IO;
        }
        where += (where.empty() ? "" : ", ") + dir;
        std::vector<std::string> here;
        while (const dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() < 4 || n.compare(n.size() - 3, 3, ".so") != 0) continue;
            const std::string path = dir + "/" + n;
            struct stat st {};
            if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            here.push_back(path);
        }
        closedir(d);
        std::sort(here.begin(), here.end());
        for (const std::string& f : here) {
            const std::string stem = stem_of(f);
            auto at = std::find(stems.begin(), stems.end(), stem);
            if (at != stems.end()) {
                RAD_INFO("plugin %s is shadowed by %s, which comes first on the search path",
                         f.c_str(), files[(size_t)(at - stems.begin())].c_str());
                continue;
            }
            files.push_back(f);
            stems.push_back(stem);
        }
    }
    const std::string dir = where;

    std::vector<Staged> staged;
    for (const std::string& f : files) {
        Staged s;
        int st = stage_(f, &s);
        if (st < 0) {
            if (first_error == RAD_OK) first_error = st;
            continue;
        }
        staged.push_back(s);
    }

    /* An entry names a library by its declared name or by its file, written as the operator sees
     * it on disk: `libr4d`, `r4d`, `libr4d.so` and a path to it are one library. */
    auto names = [](const std::string& h, const Staged& s) {
        return h == s.name || h == s.file || stem_of(h) == s.file;
    };
    std::string found;   /* every kernel library on the search path, for a list that names none */
    for (const Staged& s : staged)
        if (s.info && s.info->kind == RAD_PLUGIN_KERNEL)
            found += (found.empty() ? "" : ", ") + s.name;

    /* A HIERARCHY NAMES THE KERNEL LIBRARIES THAT SERVE, AND ONLY THOSE LOAD. Ordering alone
     * cannot keep a library out: selection runs per domain, so a library named first that has no
     * device row for an op leaves that op to whichever unnamed library does -- a GPU library on a
     * machine whose card it was not built for, say -- and an operator who listed what to use gets
     * something else. A kernel library on the search path that the list leaves out is closed again
     * before it registers anything, and said once. Architecture and quantiser plugins are not
     * kernel libraries and no list ranks them. */
    if (!hierarchy.empty()) {
        std::vector<Staged> kept;
        std::string left_out;
        for (Staged& s : staged) {
            const bool kernel = s.info && s.info->kind == RAD_PLUGIN_KERNEL;
            const bool named = std::find_if(hierarchy.begin(), hierarchy.end(),
                                            [&](const std::string& h) {
                                                return names(h, s);
                                            }) != hierarchy.end();
            if (!kernel || named) {
                kept.push_back(s);
                continue;
            }
            left_out += (left_out.empty() ? "" : ", ") + s.name;
            unstage_(s);
        }
        staged.swap(kept);
        if (!left_out.empty())
            RAD_INFO("kernel libraries on the search path that --kernels (or RADIANCE_KERNELS) "
                     "does not name, and so not loaded: %s", left_out.c_str());
    }

    /* The hierarchy is configured, not discovered: a library the operator named takes that
     * position. With no list every kernel library on the search path loads, so dropping an .so
     * into the directory adds it with no configuration.
     *
     * A REFERENCE LIBRARY IS PINNED BELOW EVERY OTHER UNNAMED PLUGIN. It declares a kernel for
     * every op it knows, so a plugin ranked behind it is unreachable -- an unnamed kernel library
     * for this machine's card would load, never be selected for anything, and leave the whole
     * model resolving to the naive implementation. Pinning it is what makes an unconfigured
     * drop-in work. Naming it in the hierarchy places it where it is named, which is how it is put
     * in front of a library to compare against. Which library is one is its own declaration
     * (RadPluginReferenceFn), not its name. */
    const size_t kUnnamed = (size_t)INT_MAX - 1;
    std::vector<size_t> rank(staged.size());
    for (size_t i = 0; i < staged.size(); ++i)
        rank[i] = staged[i].reference ? (size_t)INT_MAX : kUnnamed;
    for (size_t h = 0; h < hierarchy.size(); ++h) {
        bool hit = false;
        for (size_t i = 0; i < staged.size(); ++i) {
            if (names(hierarchy[h], staged[i])) {
                rank[i] = h;
                hit = true;
            }
        }
        /* The list is what loads, so an entry that matches nothing is a library that is not
         * there to select -- and a list none of whose entries match loads no kernel library. */
        if (!hit)
            RAD_WARN("--kernels (or RADIANCE_KERNELS) names '%s', which is no kernel library in "
                     "%s, so nothing loads in its place; the kernel libraries there are: %s",
                     hierarchy[h].c_str(), dir.c_str(), found.empty() ? "none" : found.c_str());
    }
    std::vector<size_t> idx(staged.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) { return rank[a] < rank[b]; });

    /* Orders continue from what is already loaded, so loading kernels/ and then architectures/
     * does not produce two plugins at position 0. */
    int order = 0;
    for (const Plugin& p : plugins_) order = std::max(order, p.order + 1);

    for (size_t i : idx) {
        int st = commit_(staged[i], order);
        if (st < 0) {
            if (first_error == RAD_OK) first_error = st;
            continue;   /* refused, and the next plugin still gets its chance */
        }
        if (st > 0) continue;   /* declined this machine */
        ++order;
    }

    int st = check_complete();
    if (st < 0 && first_error == RAD_OK) first_error = st;
    return first_error;
}

/* ------------------------------------------------------------------ the tuning cache (§15) */
int Registry::load_tune(const std::string& radiance_home, const std::string& machine) {
    const std::string key = machine.empty() ? rad_machine_key() : machine;
    const int s = tune_.load(radiance_home, key);
    if (s < 0) return s;
    /* A cache measured on another card is not a slower answer, it is a wrong one -- tunecache.h
     * makes the machine key line 2 of the file for exactly this. `load` already refuses a
     * mismatch, so reaching here with rows means they are this machine's. */
    return (int)tune_.rows().size();
}

/* Kernel plugins declare the ops they implement AND the parameter schema for each. A row whose op
 * has no schema anywhere in the loaded set is a plugin that supplied only half of that contract,
 * and the cost is exact: no architecture plugin's parameters for that op can ever be checked, so a
 * typo in them becomes a silent miss instead of a named one. Checked once the set is complete,
 * because an override plugin may legitimately rely on the schema of the plugin BELOW it, which
 * loads later. */
int Registry::check_complete() {
    int st = RAD_OK;
    for (const KernelRow& r : rows_) {
        if (schemas_.find(r.info->op) == schemas_.end()) {
            err_("kernel '%s' in plugin '%s' implements op '%s', which no loaded plugin declares a "
                 "schema for", r.info->name, r.plugin.c_str(), r.info->op);
            st = RAD_E_NOSCHEMA;
        }
    }
    return st;
}

/* ------------------------------------------------------------------ lookup and teardown */
const Plugin* Registry::plugin(std::string_view name) const {
    for (const Plugin& p : plugins_)
        if (p.name == name || p.file == name) return &p;
    return nullptr;
}

const Registry::QuantRow* Registry::quantizer(std::string_view name) const {
    for (const QuantRow& q : quants_)
        if (name == q.info->name) return &q;
    return nullptr;
}

const std::vector<const KernelRow*>* Registry::rows_for(std::string_view op) const {
    auto it = by_op_.find(std::string(op));
    return it == by_op_.end() ? nullptr : &it->second;
}

void Registry::close() {
    fini_instances();
    for (const QuantRow& q : quants_)
        if (q.info && q.info->decode) enc_remove_decoder(q.info->decode);
    /* Reverse order: a plugin lower in the hierarchy may have been opened against state a higher
     * one established, and unwinding in the order things were built is the only order that is
     * always safe. */
    for (auto it = plugins_.rbegin(); it != plugins_.rend(); ++it) {
        if (it->close) it->close();
        if (it->handle) dlclose(it->handle);
    }
    plugins_.clear();
    rows_.clear();
    quants_.clear();
    by_op_.clear();
    schemas_.clear();
    load_errors_.clear();
    clear_selections();
}

Registry::~Registry() { close(); }

}  /* namespace rad */
