/* hf_model_test.cpp -- a --model that names a Hugging Face repo (core/hf_model.h).
 *
 * Two questions, both answered before anything is fetched, and both wrong in a way that costs a
 * download or a confusing refusal:
 *
 *   IS IT A REPO   a path that exists is never looked up, and a missing model FILE is reported as
 *                  missing rather than sent to the hub as a repo name
 *   WHAT TO SERVE  the hub's model repos hold one container beside a README, a recipe and a
 *                  licence; a checkpoint repo holds none and is served as the directory
 *
 * No network: the fetch itself is scripts/hfget.sh, which this does not run.
 */
#include "rad_test.h"

#include "hf_model.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace rad;
namespace fs = std::filesystem;

namespace {

/* A scratch directory of its own, removed with everything in it. */
struct TempDir {
    std::string path;
    TempDir() {
        char t[] = "/tmp/hf_model_test.XXXXXX";
        path = mkdtemp(t) ? t : "";
    }
    ~TempDir() {
        std::error_code ec;
        if (!path.empty()) fs::remove_all(path, ec);
    }
    void touch(const std::string& name) const {
        if (FILE* f = std::fopen((path + "/" + name).c_str(), "w")) std::fclose(f);
    }
};

}  /* namespace */

TEST(a_repo_id_is_two_names_of_the_hubs_characters) {
    CHECK(hf_repo_id("StillDeadcode/minicpm5-2b-fp8"));
    CHECK(hf_repo_id("StillDeadcode/qwen3.8-27b-mxfp4"));
    CHECK(hf_repo_id("Qwen/Qwen3.8-Flash-Next"));
    CHECK(hf_repo_id("hf-internal-testing/tiny_random.v2"));

    CHECK(!hf_repo_id(""));
    CHECK(!hf_repo_id("minicpm5-2b-fp8"));               /* no owner */
    CHECK(!hf_repo_id("/abs/model"));                     /* absolute */
    CHECK(!hf_repo_id("a/b/c"));                          /* a path, not a repo */
    CHECK(!hf_repo_id("owner/"));
    CHECK(!hf_repo_id("/name"));
    CHECK(!hf_repo_id("../x"));                           /* a relative path, not a name */
    CHECK(!hf_repo_id("./x"));
    CHECK(!hf_repo_id("owner/.hidden"));
    CHECK(!hf_repo_id("-x/y"));                           /* would read as an option */
    CHECK(!hf_repo_id("owner/name with space"));
    CHECK(!hf_repo_id("owner/na$me"));
    CHECK(!hf_repo_id("C:/models"));
}

TEST(a_missing_model_file_is_not_sent_to_the_hub) {
    CHECK(!hf_repo_id("models/qwen.rad"));
    CHECK(!hf_repo_id("ckpt/model.safetensors"));
    CHECK(!hf_repo_id("ckpt/model.gguf"));
    CHECK(!hf_repo_id("ckpt/model.safetensors.index.json"));
}

TEST(a_path_that_exists_is_never_a_repo) {
    TempDir t;
    CHECK(!t.path.empty());
    const std::string cwd = fs::current_path().string();
    CHECK_EQ(chdir(t.path.c_str()), 0);
    fs::create_directories("owner/name");
    CHECK(!hf_repo_id("owner/name"));                     /* exists: a checkpoint directory */
    CHECK(hf_repo_id("owner/other"));                     /* does not: a repo */
    CHECK_EQ(chdir(cwd.c_str()), 0);
}

TEST(the_one_container_in_a_repo_is_what_is_served) {
    TempDir t;
    t.touch("README.md");
    t.touch("LICENSE");
    t.touch("q38-27b.recipe");
    t.touch(".hfget-manifest");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), t.path);       /* no container: a checkpoint */

    t.touch("qwen3.8-27b-mxfp4.rad");
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");

    fs::create_directories(t.path + "/sub");
    t.touch("sub/other.rad");                             /* only the top level counts */
    CHECK_EQ(hf_served_path(t.path, &why), t.path + "/qwen3.8-27b-mxfp4.rad");
}

TEST(two_containers_are_refused_by_name) {
    TempDir t;
    t.touch("a.rad");
    t.touch("b.rad");
    std::string why;
    CHECK_EQ(hf_served_path(t.path, &why), std::string());
    CHECK(why.find("a.rad") != std::string::npos);
    CHECK(why.find("b.rad") != std::string::npos);
    CHECK(why.find("--model") != std::string::npos);
}

TEST(the_download_dir_defaults_to_the_xdg_cache) {
    const char* x = std::getenv("XDG_CACHE_HOME");
    const std::string saved = x ? x : "";
    setenv("XDG_CACHE_HOME", "/c", 1);
    CHECK_EQ(hf_default_download_dir(), std::string("/c/radiance/hf"));
    unsetenv("XDG_CACHE_HOME");
    if (const char* h = std::getenv("HOME"); h && *h)
        CHECK_EQ(hf_default_download_dir(), std::string(h) + "/.cache/radiance/hf");
    if (x) setenv("XDG_CACHE_HOME", saved.c_str(), 1);
}

RAD_TEST_MAIN()
