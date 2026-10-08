/* main.cpp -- the entry point. Everything it does is in engine.h's comment; this file only turns
 * a command line into that sequence and a status into an exit code.
 */
#include "engine.h"
#include "hf_model.h"

int main(int argc, char** argv) {
    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    rad::Engine engine(std::move(cfg));

    s = engine.start();
    if (s < 0) return 1;                       /* start() already said what and why */

    if (engine.config().debug_graph) return engine.dump();
    return engine.serve() < 0 ? 1 : 0;
}
