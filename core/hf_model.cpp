/* hf_model.cpp -- resolve a --model that is an HF repo id into a local cache dir.
 *
 * The entry point (main.cpp) stays thin: it calls rad::resolve_hf_model() once, between
 * config_parse and Engine construction, and the rest of bring-up sees a plain directory.
 * The fetching itself is scripts/hfget.sh (curl + jq, no Python), run as a child process.
 */
#include "engine.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <sys/wait.h>

namespace rad {

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
bool resolve_hf_model(Config& cfg) {
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

}  /* namespace rad */
