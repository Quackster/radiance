/* rad-tokenize -- text file -> token ids, with the model's own tokenizer, on the CPU.
 *
 * usage: rad-tokenize <tokenizer.json> <text-file> [out.ids] [--special]
 *   prints the token count; with out.ids also writes the ids (little-endian int32). --special parses control
 *   tokens in the text ("<|im_start|>" is the token), which is how a rendered chat prompt is encoded.
 * Used to check that a probe prompt renders to the length it is meant to (tests/lookup_*, lookup_probe.py).
 */
#include "text/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;

int main(int argc, char** argv) {
    std::vector<std::string> args;
    bool special = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--special") special = true; else args.push_back(a);
    }
    if (args.size() < 2) { std::fprintf(stderr, "usage: rad-tokenize <tokenizer.json> <text-file> [out.ids] [--special]\n"); return 2; }
    log_set_level(Log::Warn);
    VocabBuild vb;
    if (parse_tokenizer_json(args[0].c_str(), vb) < 0) return 1;
    auto vocab = std::make_shared<Vocab>();
    if (vocab->load(std::move(vb)) < 0) return 1;
    Tokenizer tok;
    if (tok.init(vocab) < 0) return 1;
    std::ifstream f(args[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", args[1].c_str()); return 1; }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::vector<int32_t> ids;
    if (tok.encode(ss.str(), ids, /*add_special=*/false, /*parse_special=*/special) < 0) return 1;
    std::printf("%zu\n", ids.size());
    if (args.size() > 2) {
        std::ofstream o(args[2], std::ios::binary);
        o.write((const char*)ids.data(), (std::streamsize)(ids.size() * 4));
    }
    return 0;
}
