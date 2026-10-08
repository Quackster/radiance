/* registry.h -- the plugin layer: dlopen, the configured hierarchy, the op-schema registry, the
 * constraint selector, bucket tables and the selection table. spec.md §2 and §16.
 *
 * Four rules live here and nowhere else, because every one of them is a rule somebody would
 * otherwise reinvent slightly differently in a second place:
 *
 *   THE HIERARCHY IS ABSOLUTE.  Plugins are queried in configured order and the first plugin with
 *   any matching kernel supplies it. A lower-priority plugin never outbids a higher one no matter
 *   how specialised its kernel -- that is what makes a user override an override (§2.1).
 *
 *   SPECIFICITY IS DECLARED, NOT DERIVED.  Within one plugin the highest `priority` among matching
 *   rows wins, ties broken by declaration order. Counting matched constraints gets it backwards:
 *   {head_dim<=1024, gqa>=1, block%16==0, causal in {0,1}} matches four and {head_dim==256, gqa==6}
 *   matches two, and the second is by far the more specialised.
 *
 *   SCHEMAS ARE FIXED BY FIRST DECLARATION.  The first plugin in hierarchy order to declare an op
 *   fixes its schema; a later plugin declaring the same op name with a different one is refused at
 *   load with both plugins named. Arguments are positional, so two disagreeing schemas make the
 *   same call mean different things depending on which plugin won selection -- silent numerical
 *   garbage rather than a diagnosable failure (§2.3).
 *
 *   THE BUCKET BOUNDARIES ARE THE CONSTRAINT VALUES THEMSELVES.  The union of the LE, GE and EQ
 *   values candidate kernels place on the ranged parameter, clamped to the declared range. No
 *   policy is chosen by anyone (§2.2).
 *
 *   A REFERENCE LIBRARY IS LAST BY CONSTRUCTION.  A library that declares itself one
 *   (RadPluginReferenceFn) matches every op it declares, so anything ranked behind it would be
 *   unreachable. An unnamed plugin sorts after every named one but still ahead of a reference,
 *   which is what makes dropping in a kernel library for a new card work with no configuration
 *   (§2.1). Naming it in the hierarchy still overrides that. The rule is keyed on the declaration,
 *   never on a name: the libraries that ship with radiance load exactly as a third party's do.
 *
 * LIFETIME. A Resolved, a Band and a KernelRow all point into a plugin's static tables. They are
 * valid exactly as long as the Registry that loaded them: close() dlcloses, and everything handed
 * out before it dangles after. The core keeps one Registry for the life of the process, so this
 * costs nothing; a tool that loads two kernel sets in turn has to finish with the first.
 *
 * THREADING. Loading is startup, single-threaded, before the rank threads exist. select() and
 * init_instance() are called from every rank thread during declare, so the selection log and the
 * instance list are behind one mutex. It is uncontended at declare and never touched on the step
 * path, which is the only place a lock would cost anything.
 */
#pragma once
#include "rad_core.h"
#include "rad_quant.h"
#include "../format/tunecache.h"

#include <deque>
#include <mutex>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rad {

inline const char* domain_name(int d) { return d == RAD_DOMAIN_HOST ? "host" : "device"; }


/* ------------------------------------------------------------------ one loaded .so */
struct Plugin {
    std::string name;          /* RadPluginInfo::name -- what the hierarchy list names */
    std::string file;          /* the .so's stem; the hierarchy list accepts this spelling too */
    std::string path;
    std::string version;
    std::string description;
    std::string build_target;

    uint32_t    kind   = 0;    /* RAD_PLUGIN_KERNEL | RAD_PLUGIN_ARCH */
    int         order  = 0;    /* hierarchy position; 0 is highest */
    void*       handle = nullptr;

    int n_kernels = 0;
    int n_schemas = 0;
    /* A kernel library whose fat binaries hold no code object the card runs: its device rows were
     * left out at load (loader.cpp, commit_), and only its host rows are selectable. */
    bool device_code_absent = false;
    /* A kernel library that declared itself a reference implementation (RadPluginReferenceFn). */
    bool reference = false;

    /* Architecture plugins are loaded and IDENTIFIED here; the build component drives them. The
     * core never calls declare or step from the registry -- it only guarantees that no two
     * plugins claim the same (architecture id, quantisation) PAIR (§2.4). The pair, not the id:
     * qwen35_bf16 and qwen35_fp8 are the same architecture at two weight formats and both answer
     * "qwen35", which is the whole reason §2.4 splits on quantisation at all. `arch_quant` is ""
     * for a plugin that serves an unquantised container, including one that predates the export. */
    std::string arch_id;
    std::string arch_quant;
    int  (*arch_declare)(RadBuilder*, const RadModelMeta*, const RadBuildCtx*) = nullptr;
    void (*arch_step)(RadCtx*, const RadBatch*) = nullptr;
    /* OPTIONAL (rad_builder.h): null when the plugin exports none, and every question then
     * answers zero. It is what the core asks BEFORE declare, so that a fact which is an input to
     * declaration and a property of the model is still read by the plugin and not by us. */
    int  (*arch_probe)(const RadModelMeta*, RadArchProbe*) = nullptr;
    /* OPTIONAL (rad_builder.h): null when the plugin exports none, and the reply format is then
     * derived from the chat template. */
    const RadChatFormat* (*arch_chat_format)(const RadModelMeta*) = nullptr;

    RadPluginCloseFn close = nullptr;
};

/* ------------------------------------------------------------------ one selection question */
/* What select() did, in enough detail to print §16's table and to answer "why did it not pick the
 * kernel I wrote". `candidates` is every row that was CONSIDERED -- rows of the other domain are
 * not, since they answer a different question, and rows behind a plugin that already supplied a
 * match are not either, because the hierarchy stopped there. */
struct SelectionTrace {
    struct Candidate {
        std::string plugin;
        std::string kernel;
        int         priority = 0;
        bool        ok = false;
        std::string why;        /* the constraint that refused it, or "ok" */
    };

    std::string op;
    std::string query;          /* the geometry as asked */
    int         domain = RAD_DOMAIN_DEVICE;

    std::string kernel;         /* empty on a miss */
    std::string plugin;
    int         priority = 0;
    std::string reason;         /* on a miss: the constraint(s) that refused it */

    /* A RAD_DERIVED parameter the CALLER did not supply: its constraints were skipped during
     * matching, and the winning kernel's RAD_C_EQ value for the key was written into the resolved
     * geometry afterwards. `known` is false when the winner has no EQ on that key -- ref's
     * `block_size >= 1` is exactly that case -- and an unset derived key is a condition the
     * caller has to report, not a zero to be read. */
    struct DerivedParam {
        std::string key;
        long long   value = 0;
        bool        known = false;
    };
    std::vector<DerivedParam> derived;

    std::vector<Candidate> candidates;
    bool other_domain_has_rows = false;   /* the op exists, just not in the domain asked for */
};

/* ------------------------------------------------------------------ the registry */
class Registry {
public:
    Registry() = default;
    ~Registry();
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    /* ---------------------------------------------------------- loading (§2, §18) */
    /* Scan a directory of .so files and load them in HIERARCHY ORDER. The hierarchy is configured,
     * not discovered: `hierarchy` is an ordered list of plugin names (RadPluginInfo::name, or the
     * file as the operator wrote it -- `libr4d`, `r4d`, `libr4d.so`), so substituting a kernel is
     * dropping an .so in a directory and naming it ahead of the default. A non-empty list is
     * EXCLUSIVE: a kernel library it does not name is closed before it registers anything, and an
     * entry that matches nothing is warned about by name. An empty list loads every kernel library
     * found, in filename order so the result is reproducible, with a reference library last.
     * Architecture and quantiser plugins are not ranked and always load.
     *
     * Always completes: every plugin that can be loaded is, every refusal is recorded in
     * load_errors(), and the FIRST negative status is returned. Reporting the whole list beats
     * reporting the first failure at the top -- an operator with two stale plugins wants to hear
     * about both in one run (§3.1). */
    int load_dir(const std::string& dir, const std::vector<std::string>& hierarchy);
    /* The same over several directories -- a $RADIANCE_HOME search path -- as one set: the
     * hierarchy orders plugins across all of them, and a file an earlier directory already holds
     * by the same name is shadowed, not loaded. */
    int load_dirs(const std::vector<std::string>& dirs, const std::vector<std::string>& hierarchy);

    /* One plugin at a known hierarchy position. rad-info and rad-kbench load a single .so; the
     * engine goes through load_dir. 1, and nothing registered, when the plugin declines this
     * machine (its rad_plugin_open answers RAD_E_UNSUPPORTED). */
    int load_plugin(const std::string& path, int order);

    /* Every registered kernel row's op must have a schema SOMEWHERE in the loaded set -- an
     * override plugin may legitimately ship kernels for an op whose schema the plugin below it
     * declared, and that plugin loads later, so the check can only run once the set is complete.
     * load_dir calls this; a caller assembling a set by hand calls it when it is done. */
    int check_complete();

    /* The device a kernel library's device code is checked against when it loads: the first card
     * the caller will run on. 0 unless the caller says otherwise -- the engine says so before it
     * loads anything, once it has chosen its cards. */
    void set_code_device(int device) { code_device_ = device; }

    /* THE ENGINE ABOUT TO SERVE ON THE HOST BACKEND SAYS SO: no card is visible, and no GPU code
     * object runs in this process at all. A kernel library that brought device code then has its
     * device rows left out of selection at load, its host rows staying, so every op takes its host
     * kernel. Only the engine says this. A tool that selects without serving -- rad-convert's
     * recipe check, rad-kbench's tables, a test of a library's constraints -- sees every row, as
     * it would on the machine the model is served on. */
    void set_no_device_code(bool v) { no_device_code_ = v; }
    /* WHAT THE ENGINE MEASURED ABOUT THE RANKS' DEVICES, for the constraints on "device." keys
     * (rad_types.h). Set before the first selection, by whoever chose the devices; a registry
     * nobody told anything has no facts, and a kernel constrained on one is not selected. */
    void set_device_fact(std::string_view key, long long v) { device_facts_.set_i(key, v); }
    const Geometry& device_facts() const { return device_facts_; }

    void close();                       /* fini instances, close hooks, dlclose, forget everything */

    const std::vector<Plugin>&      plugins() const { return plugins_; }
    const Plugin*                   plugin(std::string_view name) const;
    const std::vector<std::string>& load_errors() const { return load_errors_; }

    /* Every quantiser the loaded quantiser plugins offer, in load order. A name is unique across
     * them -- two plugins naming one quantiser is refused at load -- so the lookup is the answer
     * and not the first of several. Null when nothing offers it. */
    struct QuantRow {
        const RadQuantizerInfo* info = nullptr;
        std::string             plugin;
    };
    const std::deque<QuantRow>& quantizers() const { return quants_; }
    const QuantRow*             quantizer(std::string_view name) const;

    /* Every kernel row, in (hierarchy order, declaration order). rad-info prints this. */
    const std::deque<KernelRow>&    rows() const { return rows_; }
    /* The rows implementing one op, in the order select() considers them. Empty if none. */
    const std::vector<const KernelRow*>* rows_for(std::string_view op) const;

    /* ---------------------------------------------------------- op schemas (§2.3) */
    const RadOpSchema* schema(std::string_view op) const;
    const Plugin*      schema_owner(std::string_view op) const;
    std::vector<std::string> op_names() const;      /* sorted; the vocabulary this set implements */

    /* Check a declared parameter set against the op's schema. Without this, a typo and an
     * unimplemented op are the same diagnostic; with it, a misspelled op fails declare with the
     * near-miss named. Returns RAD_OK, RAD_E_NOSCHEMA (no such op) or RAD_E_SCHEMA, and fills
     * `err` with a sentence an architecture-plugin author can act on. */
    int validate(std::string_view op, const RadParam* p, int n_p, std::string* err) const;

    /* The closest known op name to `op` within a small edit distance, or "" if nothing is close.
     * Public because the builder wants it in its own messages too. */
    std::string near_miss(std::string_view op) const;

    /* ---------------------------------------------------------- the tuning cache (§15) */
    /* Load $RADIANCE_HOME/tune/<machine>.tune, which rad-tune wrote. resolve() then moves a shape
     * off each axis's declared default wherever there is a row for it. A missing file is not an
     * error and leaves every shape on that default, which is what makes an untuned install fast
     * rather than broken.
     *
     * THE CORE OWNS THE CACHE AND THE POLICY, so no plugin has to (tunecache.h; the choice is
     * applied in select.cpp). Without this call every kernel runs its axis defaults whatever the
     * file on disk says, which is slow rather than wrong -- but silently so.
     *
     * Returns the number of rows loaded, or a negative status. */
    int  load_tune(const std::string& radiance_home, const std::string& machine = std::string());
    const TuneCache& tune() const { return tune_; }

    /* ---------------------------------------------------------- selection (§2.1) */
    /* The first plugin in hierarchy order with any matching row supplies the kernel; within that
     * plugin the highest declared priority wins, ties on declaration order. Null on a miss, with
     * the refusing constraint in `why->reason`. Every call is recorded in the selection table.
     *
     * RAD_DERIVED parameters (rad_abi.h) are supplied by the KERNEL, so a constraint on one is
     * SKIPPED here when the caller did not name it, and the winner's RAD_C_EQ value for the key
     * comes back in `why->derived` -- resolve() is what writes it into the geometry. Without that
     * skip, spec §7.2 is circular: the block manager reads the block size off the resolved
     * attention kernel, but a kernel constrained to block_size == 16 can never be selected by a
     * query that omits block_size, because a constraint whose key the geometry lacks does not
     * hold. A caller that supplies the key anyway pins it and matches normally, which is how an
     * operator forces a block size and gets a refusal by name instead of a substitution. */
    /* `record` is false for a caller asking the selector a question the ENGINE did not ask -- the
     * graph dump replays every band to recover the candidate list it threw away. The selection
     * table is a record of what this deployment resolved, and a reporting tool inflating its
     * counts would make it a record of itself. */
    /* `accept`, when given, is asked of every row whose constraints hold, and an answer that is
     * not empty takes the row out as a candidate with that answer as the reason. It is how a
     * weight's encoding takes part in selection: a kernel whose layout hook does not read the
     * encoding a weight operand has (RAD_E_DTYPE) is not a candidate, and the next one is chosen
     * -- a bf16 lm_head goes to the bf16 GEMM instead of being refused by the fp8 one. */
    using RowFilter = std::function<std::string(const KernelRow&, const Geometry&)>;
    const KernelRow* select(std::string_view op, const Geometry& g, int domain,
                            SelectionTrace* why = nullptr, bool record = true,
                            const RowFilter* accept = nullptr);

    /* Selection plus the per-instance bookkeeping: the tuned axes, where they came from, and the frozen
     * geometry -- the query plus whatever the winning kernel supplied for a derived key, so
     * rad_kv_block_size() and RadArgs.p both see it. init() is NOT called here -- see
     * init_instance(). */
    Resolved resolve(std::string_view op, const Geometry& g, int domain, SelectionTrace* why = nullptr,
                     const RowFilter* accept = nullptr);

    /* ---------------------------------------------------------- bucket tables (§2.2) */
    /* Resolve a ranged parameter into a bucket table: one resolution per band, per domain, built
     * at declare and indexed by the actual value at issue. The boundaries are the constraint
     * values themselves. Each band is resolved AT ITS UPPER BOUND, which is the worst case in the
     * band and therefore the only value that is safe to size and select on.
     *
     * `ranged_key` empty means the op has no ranged parameter: exactly one band is returned, whose
     * hi is INT64_MAX so the issue path's band scan always finds it.
     *
     * Both domains are resolved wherever both exist, so the core can change execution site at
     * runtime without re-resolving. Absence is asymmetric: a band with no DEVICE kernel is fatal
     * and the caller reports it (this function does not abort -- declare always completes); a band
     * with no HOST kernel is a warning naming the placement options it just removed. */
    std::vector<Band> build_bands(std::string_view op, std::string_view ranged_key,
                                  int64_t lo, int64_t hi, const Geometry& base,
                                  const RowFilter* accept = nullptr);

    /* ---------------------------------------------------------- instances (§2.1) */
    /* init() once per resolved instance -- one op, one band, one domain -- with the frozen
     * geometry, the rank and the world size. libr4d's all-reduce needs its peer signal buffers and
     * IPC handles established across ranks before the first call, and a kernel that JITs or loads
     * a config table wants somewhere to do it that is not the hot path. A negative return aborts
     * startup and the caller names the kernel. */
    int  init_instance(Resolved& r, int rank, int world_size);
    void fini_instances();
    size_t n_instances() const;

    /* ---------------------------------------------------------- the selection table (§16) */
    /* Every DISTINCT question the selector was asked, in the order first asked, with what it
     * resolved to and, for a miss, the constraint that refused it. Repeats are folded and counted:
     * a 64-layer model asks the same question 64 times, and "asked 64 times" says how much of the
     * model resolved to that kernel. This is the record of why a fallback is running. */
    std::string selection_table() const;
    size_t      n_selections() const;
    void        clear_selections();

private:
    struct Staged {
        void*                handle = nullptr;
        std::string          path, file, name;
        const RadPluginInfo* info = nullptr;
        /* The device backend's fat-binary registrations before and after this plugin's dlopen:
         * [code_from, code_to) is the device code it brought in. */
        size_t               code_from = 0, code_to = 0;
        bool                 reference = false;   /* RadPluginReferenceFn answered nonzero */
    };

    struct SchemaEntry {
        const RadOpSchema* schema = nullptr;
        std::string        plugin;      /* who fixed it, for the conflict message */
    };

    struct SelectionRecord {
        std::string op, query, key, kernel, plugin, reason;
        std::string derived;      /* "block_size:=16", rendered apart from the query because it
                                   * was NOT part of the question the selector was asked */
        int         domain = 0;
        int         priority = 0;
        long long   count = 1;
    };

    int code_device_ = 0;
    bool no_device_code_ = false;
    Geometry device_facts_;
    TuneCache tune_;

    int  stage_(const std::string& path, Staged* out);
    int  commit_(Staged& s, int order);
    void unstage_(Staged& s);
    void index_rows_();
    void record_(const std::string& op, int domain, const Geometry& g, const SelectionTrace& t);
    void err_(const char* fmt, ...) __attribute__((format(printf, 2, 3)));

    std::vector<Plugin>      plugins_;
    std::deque<KernelRow>    rows_;          /* deque: KernelRow* handed out must stay valid */
    std::deque<QuantRow>     quants_;
    std::unordered_map<std::string, std::vector<const KernelRow*>> by_op_;
    std::unordered_map<std::string, SchemaEntry> schemas_;
    std::vector<std::string> load_errors_;

    mutable std::mutex           mu_;
    std::vector<SelectionRecord> log_;
    std::vector<std::pair<const RadKernelInfo*, void*>> instances_;
};

/* The value a kernel's own constraints place on one key, e.g. the paged block size the resolved
 * attention kernel is compiled for. §7.2: the block manager reads that off the resolved kernel and
 * sizes itself accordingly, because a core that picked its own block size would be a core that has
 * to be edited when a kernel changes. Returns false when the kernel says nothing about that key
 * with that operator -- which is not the same as a default, and the caller has to decide. */
bool kernel_constraint(const KernelRow& r, std::string_view key, int c_op, long long* out);

}  /* namespace rad */
