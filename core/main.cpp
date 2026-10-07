/* main.cpp -- the entry point. Everything it does is in engine.h's comment; this file only turns
 * a command line into that sequence and a status into an exit code.
 */
#include "engine.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

/* "owner/name" that is not a local path: exactly one slash, no shell-fodder, not a known file. */
static bool is_hf_repo_id(const std::string& s) {
    if (std::filesystem::exists(s)) return false;   /* a real existing local path always wins */
    if (s.empty() || s.front() == '/') return false;
    if (s.find_first_of("\\:*?\"<>|") != std::string::npos) return false;
    size_t slash = s.find('/');
    if (slash == std::string::npos || s.find('/', slash + 1) != std::string::npos) return false;
    std::string owner = s.substr(0, slash);
    std::string name  = s.substr(slash + 1);
    if (owner.empty() || name.empty()) return false;
    if (owner == ".." || name == "..") return false;
    static const std::string kFileSuffixes[] = {".rad", ".safetensors", ".gguf", "index.json"};
    for (const auto& suf : kFileSuffixes)
        if (name.size() >= suf.size() && name.compare(name.size() - suf.size(), suf.size(), suf) == 0)
            return false;
    return true;
}

/* hfget.sh: $RADIANCE_HFGET, then <exe_dir>/../scripts/, then ./scripts/. */
static std::string find_hfget() {
    if (const char* e = std::getenv("RADIANCE_HFGET"))
        if (*e && std::filesystem::exists(e)) return e;
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string p = (std::filesystem::path(buf).parent_path() / ".." / "scripts" / "hfget.sh").string();
        if (std::filesystem::exists(p)) return p;
    }
    if (std::filesystem::exists("./scripts/hfget.sh")) return "./scripts/hfget.sh";
    return "";
}

/* If cfg.model is an HF repo id, fetch it into $RADIANCE_HOME/hf/ via hfget.sh and rewrite the path. */
static bool resolve_hf_model(rad::Config& cfg) {
    if (!is_hf_repo_id(cfg.model)) return true;
    std::string script = find_hfget();
    if (script.empty()) {
        RAD_ERR("hfget.sh not found -- set RADIANCE_HFGET to its path");
        return false;
    }
    std::string dest = cfg.radiance_home + "/hf/" + cfg.model;
    std::filesystem::create_directories(dest);
    fprintf(stderr, "radiance: downloading %s -> %s\n", cfg.model.c_str(), dest.c_str());
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return false; }
    if (pid == 0) {
        const char* argv[] = { script.c_str(), cfg.model.c_str(), dest.c_str(), nullptr };
        execvp(script.c_str(), const_cast<char* const*>(argv));
        perror("execvp hfget.sh");
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        RAD_ERR("hfget.sh failed for %s (status %d) -- see output above", cfg.model.c_str(), status);
        return false;
    }
    cfg.model = dest;   /* rewrite to the cached local dir; downstream reads it as a checkpoint dir */
    return true;
}

int main(int argc, char** argv) {
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

    if (!resolve_hf_model(cfg)) return 2;

    rad::Engine engine(std::move(cfg));

    s = engine.start();
    if (s < 0) return 1;                       /* start() already said what and why */

    if (engine.config().debug_graph) return engine.dump();
    return engine.serve() < 0 ? 1 : 0;
}
