# Radiance — user guide

Radiance is an inference server: it loads a model and answers OpenAI-compatible HTTP requests.
This guide takes you from an empty machine to a running server, and then through operating one.

There is no Python anywhere — not in the engine, not in the tools, not in the build. Everything
below is C++, CMake and shell.

**Where to go from here**

| If you want | Read |
|---|---|
| to get something running | this file, in order |
| every flag of one tool | [TOOLS.md](TOOLS.md) |
| to write a kernel library for your card | [PLUGIN.md](PLUGIN.md) |
| to know why it is built this way | `../spec.md` — the design, and it is binding |

---

## 0. The whole thing in five commands

```sh
./build.sh                                          # build and test
export RADIANCE_HOME=$PWD/build/radiance_home

scripts/hfget.sh Qwen/Qwen3.8-Flash-Next ~/ckpt     # get a checkpoint
build/bin/rad-convert ~/ckpt -o ~/m.rad \
    --recipe data/recipes/qwen4exp-w4.recipe        # quantise it into a container
build/bin/radiance --model ~/m.rad --port 8000      # serve it
```

A checkpoint served at its own precision needs no container at all: `radiance --model ~/ckpt`
reads it as it stands (§3.4).

Then:

```sh
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":50}' \
  | jq -r '.choices[0].message.content'
```

and open `http://localhost:8000/` for the dashboard. Everything below is the detail behind those
five lines.

---

## 1. Install

**Or skip this section and run the container image**: `docker/build.sh` builds the engine inside
the ROCm toolchain image, tests it and produces an image that needs only Docker and the `amdgpu`
driver on the host. [`DOCKER.md`](DOCKER.md) covers it, along with ready compose files per model.

### 1.1 What you need

| | |
|---|---|
| **Required** | Linux, a C++20 compiler, CMake 3.21+, Ninja, pthreads |
| **Optional** | ROCm with `hipcc`, for AMD device kernels. Without it you get the host backend. Set `ROCM_PATH` if it is not at `/opt/rocm` |
| **Optional** | OpenMP, used by the host kernel libraries if present |
| **For the shell harnesses** | `curl` and `jq` |

Everything else is vendored in `vendor/` — the HTTP server, the JSON library and the llama.cpp
lifts. There is nothing to `apt install` beyond a toolchain.

**A machine with no GPU is a supported configuration.** The host backend is a real implementation,
not a stub: the core, scheduler, planner, tokeniser, server and `libref` all run through it and
the full test suite passes. It is slow, and it will not serve a production model — but it is a
complete development and inspection environment.

### 1.2 Build

```sh
git clone <repo> radiance && cd radiance
./build.sh
```

That configures, builds and runs the tests. It detects your situation: if `hipcc` is present and a
card can be enumerated, device kernels are compiled for that card's architecture; otherwise you
get the host backend.

Options:

```sh
./build.sh -c          # wipe the build directory first
./build.sh --no-hip    # force the host backend even where hipcc exists
./build.sh --debug     # -DCMAKE_BUILD_TYPE=Debug
./build.sh --asan      # AddressSanitizer
./build.sh -b mybuild  # build into mybuild/ instead of build/
```

Or drive CMake yourself:

```sh
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### 1.3 Check that it works

```sh
ctest --test-dir build --output-on-failure        # everything (a few minutes)
ctest --test-dir build -LE slow                   # the fast inner loop (~1 s)
ctest --test-dir build -LE gpu                    # skip anything needing a card
```

Three of them are worth knowing by name, because they are the ones that actually check kernels:

- **`kbench` / `kbench_moe`** (label `slow`) replay a recorded fixture — every kernel every loaded
  library supplies, checked against `libref`'s recorded answers on seeded inputs. **No model file is
  needed**, which is what makes this runnable on a build machine. It also regression-tests `libref`
  itself, which a live oracle never can.
- **`r4d_selftest`** (labels `slow`, `gpu`) checks libr4d's fusion claims by `memcmp` against the
  unfused kernels it says it replaces.
- **`ops_doc`** (fast) checks that `docs/OPS.md` still agrees with what the plugins declare. A
  document that is wrong is worse than one that is missing, because it is read instead of the source.

You can also run the kernel check by hand:

```sh
build/bin/rad-kbench --home build/radiance_home --max-cases 1
```

### 1.4 Install to a prefix

```sh
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
sudo cmake --install build
```

What lands where:

```
<prefix>/bin/                      radiance, rad-convert, rad-info,
                                   rad-kbench, rad-tune, rad-schemas, rad-tokcheck
<prefix>/include/                  the ABI headers — all a plugin compiles against
<prefix>/lib/cmake/radiance/       the package config, for find_package(radiance)
<prefix>/share/radiance/           $RADIANCE_HOME: kernels/, architectures/, quantizers/, kernels*.rkb
```

### 1.5 `$RADIANCE_HOME`

This is the directory the engine and every tool scan at startup:

```
$RADIANCE_HOME/
  kernels/         kernel libraries       libr4d.so, libref.so, libavx.so, yours.so
  architectures/   architecture plugins   qwen4exp_fp8.so, llama_fp8.so, ...
  quantizers/      quantiser plugins      libquant.so
  kernels.rkb      the recorded kernel fixtures rad-kbench replays (and kernels_moe.rkb)
  tune/            per-machine tuning caches rad-tune writes
```

It defaults to the install prefix's `share/radiance`. Override it with `--radiance-home DIRS` or the
`RADIANCE_HOME` environment variable — a search path: homes separated by `:`, the first winning,
each holding any of the directories above. A directory of your own plugins goes ahead of an
installation's without copying it, and `rad-info --plugins` prints what the path offers:

```sh
RADIANCE_HOME=$HOME/radiance-plugins:/opt/radiance/share/radiance rad-info --plugins
```

**Running from a build tree**, the home is `build/radiance_home` and the plugins are already in it:

```sh
export RADIANCE_HOME=$PWD/build/radiance_home
build/bin/radiance --model m.rad
```

Adding support for a new card means dropping an `.so` into `kernels/` — not forking the engine.
The engine's own device code is built for every AMD GPU family, so one engine binary starts on any
card; a kernel library whose code objects hold nothing for the card is left out of selection when
it loads, and says so. `docs/PLUGIN.md` §10 is a library for a card the engine has never seen.

### 1.6 Building for a card you do not have

Device kernels are compiled for the architectures the machine reports. To target others:

```sh
cmake -S . -B build -G Ninja -DRAD_GPU_TARGETS='gfx1201;gfx1200'
```

Which of those actually get kernels is decided by the libraries present; each one declares the
architectures it serves and stays out of a build targeting anything else. The engine's own device
code is built for every family's generic target besides, whatever is named here.

---

## 2. Five things to know before you start

**1. A model is a checkpoint served as it stands, or a `.rad` container.** A checkpoint whose
weights are bf16, f16, f32 or block-scaled fp8 is served at that precision straight from its
directory. Anything that needs quantising is quantised once, by `rad-convert` and a recipe, into a
container. A container holds each weight as the planes of its *encoding* — codes, scales, zero
points, a rotation — in one canonical arrangement, plus the tokeniser, the chat template, the
metadata, the recipe it was quantised with and optionally a draft head. No kernel library is named
in either: the kernels rearrange the planes into their own layouts as the model loads, so a kernel
library can change without a reconvert.

**2. Kernels are plugins, and the order matters.** The engine resolves every op in the model
against an ordered list of kernel libraries: first one that matches an op wins. `--kernels
mylib,libref` states that order. With no list, the loader orders what it finds and puts a
library that declares itself a reference implementation last.

Three ship with the tree:

| library | |
|---|---|
| `libr4d` | the device library: AMD RDNA4 (`gfx12xx`). Paged and dense attention, gated delta net, P2P all-reduce, skinny and tiled quantised GEMM, rotate-and-quantise fusions. Built when ROCm is present and the build targets a `gfx12` card |
| `libref` | every op naively and generically. The last-resort fallback **and** the correctness oracle `rad-kbench` measures against |
| `libavx` | the host library: the same vocabulary in hand-written x86 SIMD, dispatched by CPU feature at load. Serves every op placed on the host, the n-gram embedding's gather among them. Built on x86-64 |

A library states the architectures it serves and stays out of a build targeting anything else, so a
tree configured for a card it does not cover still builds and still runs.

**3. Architectures are plugins too.** `arch/<id>/` declares a model family's graph. Six ship:

| plugin | serves |
|---|---|
| `llama_fp8`, `llama_dense_fp8` | LLaMA-shaped decoders — MiniCPM5, Llama 3, anything reporting `model_type: llama` |
| `qwen35_bf16` | Qwen3.5 / 3.6 / 3.8 dense (0.8B–27B), bf16 |
| `qwen35_fp8` | the same, block-scaled fp8 |
| `qwen35moe_fp8` | Qwen3.5 / 3.6 MoE (35B-A3B), 256 experts, 8 a token |
| `qwen4exp_fp8` | Qwen4-Exp / Qwen3.8-Flash-Next — hybrid Gated DeltaNet + gated attention, 512 experts |

**4. Falling back to the reference library is a fatal error, not a warning.** `libref` implements
every op naively and generically. It is the oracle and the last resort. If your model resolves any
op to it, the engine **refuses to start** and names the ops, because serving from it would be
orders of magnitude slow with nothing in the output to say so. That list of ops is exactly the
work a new kernel library has to do. `--debug-accept-reference-kernels` overrides it for bring-up.

**5. Every pool is budgeted explicitly.** There is no profiling pass and no utilisation fraction.
The trade between KV blocks and resident weights is real, workload-dependent and **yours to make**.
An engine that quietly re-optimises the split is an engine whose benchmarks do not reproduce.

---

## 3. Get a model in

### 3.1 Download a checkpoint

```sh
scripts/hfget.sh Qwen/Qwen3.8-Flash-Next ~/models/Qwen/Qwen3.8-Flash-Next
```

`curl` and `jq`, nothing else. Re-running resumes: a file whose local size already matches is
skipped and a short one is continued. It also skips duplicate weight formats — if a repo carries
`.safetensors` it will not also pull the `.bin`, the ONNX export and the GGUF.

Set `HF_TOKEN` for a gated or private repo. `HFGET_JOBS` sets concurrency (default 4).

### 3.2 Convert it

```sh
rad-convert ~/models/Qwen/Qwen3.8-27B-FP8 -o ~/models/rad/q38-27b-fp8.rad --home $RADIANCE_HOME
```

Input is a `.gguf`, a `.safetensors`, or a directory holding a checkpoint. What happens: the
architecture plugin declares the model's graph and its name map says which checkpoint tensors make
each weight; every op is resolved against the kernel hierarchy and every resolved kernel is asked
whether it reads the encoding each weight will have; then each weight is written once, as its
encoding's planes, and the tokeniser and chat template are baked in.

**A recipe decides what is quantised.** With none, every weight keeps the checkpoint's encoding,
which is right for a bf16 or block-fp8 checkpoint served at its own precision. A model whose served
form needs quantisation work — Flash-Next's four-bit experts, an fp8 head made from a bf16 one —
names it in rules from weight names to quantisers:

```sh
rad-convert ~/models/Qwen/Qwen3.8-Flash-Next -o ~/models/rad/flashnext-w4.rad \
    --recipe data/recipes/qwen4exp-w4.recipe --home $RADIANCE_HOME
```

`qwen4exp-w4.recipe` rounds the experts to nearest and needs nothing but the checkpoint. The
production Flash-Next container, `w4nl64-i8lm.rad`, is built by
`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe`, whose expert rule is GPTQ against recorded Grams:
it expands `$CALIB` from the environment, so it is run as `CALIB=DIR rad-convert ...` with `DIR`
holding a calibration run's files. The recipe's header and [MOE-W4.md](MOE-W4.md) say what that
directory must hold and how it is recorded.

```
# data/recipes/... -- PATTERN  QUANTISER  OPTIONS; the first match decides
blk.*.ffn_*_exps.*.weight   gptq  codes=i4 group=128 scale=bf16 transform=fwht128 calib=$CALIB
output.weight               rtn   codes=fp8_e4m3 block=128x128 scale=bf16
blk.*.weight                rtn   codes=fp8_e4m3 block=128x128 scale=bf16
```

Every weight it quantises is measured against the source and the worst are printed. The recipe
syntax, the quantisers and their options are in [TOOLS.md](TOOLS.md#recipes).

Because it runs the same declare phase the engine does, everything that can refuse has refused by
the time the I/O starts — including a recipe no loaded kernel can serve. On a large checkpoint that
I/O is an hour, so check the plan first:

```sh
rad-convert <checkpoint> -o out.rad --recipe r.recipe --plan-only -v
```

Useful variants:

```sh
# Merge a separate drafter checkpoint (DSpark, DFlash2) into the container
rad-convert $SRC -o out.rad --draft-model ~/models/openbmb/MiniCPM5-2B-DSpark

# Re-quantise some weights and copy the rest from an earlier container
rad-convert $SRC -o new.rad --recipe new.recipe --reuse old.rad
```

The conversions of record live in `scripts/convfn.sh` and `scripts/convdspark.sh`, with their
recipes in `data/recipes/`.

### 3.3 Look at what you built

```sh
rad-info ~/models/rad/flashnext-w4.rad
```

```
flashnext-w4.rad
  format        RAD2, 166.4 GiB, little-endian
  architecture  qwen4exp
  model         Qwen3.8-Flash-Next
  source quant  (none)
  recipe        14 line(s) -- --recipe prints it
  created by    rad-convert ...

  section          offset        bytes    count
  strings             256      1.2 MiB        -
  metadata        1258752      8.4 KiB      214
  ...
```

`rad-info` opens the file read-only and loads no plugins. It lists every weight with its encoding
and what made it — the checkpoint as it stood, or the recipe rule's quantiser and options.

```sh
rad-info --meta m.rad                  # just the metadata table
rad-info --vocab m.rad                 # vocab summary and normaliser chain
rad-info --recipe m.rad                # the recipe it was converted with
rad-info -v m.rad                      # every tensor
```

### 3.4 Or skip the container

```sh
radiance --model ~/models/Qwen/Qwen3.8-27B-FP8
```

`--model` takes a checkpoint directory, a `.safetensors`, an index or a `.gguf`, and serves it at
its own precision. Its `config.json` (or the GGUF header) is the metadata, the `tokenizer.json` and
chat template beside it are the tokeniser, and `generation_config.json` gives the sampling defaults.
The weights are read once, at load, by every relayout worker at once.

**Nothing is quantised at load.** A weight the architecture plugin reads in another encoding than
the checkpoint holds is named at startup, and the remedy is a container with a recipe that says
what it becomes. MiniCPM5-2B is one: OpenBMB ships it in bf16, and `llama_dense_fp8` reads fp8.

What a container has that a checkpoint does not:

| | |
|---|---|
| the file and mapped tiers | a checkpoint is neither grouped nor aligned for them, so every weight must fit VRAM or the host pool, and a weight only a host kernel reads (the n-gram table) cannot be served |
| a merged drafter | `--draft-model` is `rad-convert`'s; a checkpoint serves the draft head it ships, if any |
| a placement profile | worth about a point of expert hit rate |

A GGUF is read when its architecture plugin maps llama.cpp's tensor names; the shipped plugins map
HuggingFace's, so a model of a shipped architecture is converted from its safetensors release.

---

## 4. Serve it

### 4.1 The simplest run

```sh
radiance --model ~/models/rad/flashnext-w4.rad
```

It claims the card (less `--gpu-headroom-mib`, default 96 MiB), splits it between weights and KV
cache, loads, and serves on `http://0.0.0.0:8000`. The startup log states every budget before
anything is allocated.

```sh
radiance --model m.rad --port 8100 --host 127.0.0.1
```

### 4.2 Look before you leap

```sh
radiance --model m.rad --debug-graph
```

Prints the entire declared graph — every op, the kernel each one resolved to, and why — then exits
without serving. This is the first thing to run against a new container or a new kernel library.

```sh
radiance --model m.rad --debug-selection    # just the selection table
radiance --model m.rad --debug-placement    # just the placement plan
```

### 4.3 Memory: how the card gets split

The card is split into **resident weights** and the **paged KV pool**. You can state the split
outright:

```sh
radiance --model m.rad --vram-weights-mib 24000 --vram-kv-mib 6000
```

or leave both unset and let the engine derive it from the card. The non-expert weights come off
the top of what is claimed. The rest — the `elastic` line of the startup receipt — is shared by
the expert slab and the KV cache, and no flag divides it:

- **The cache lends** under `--placement auto` or `expert_tiered`, without `--deterministic`, on a
  device with virtual memory management. The experts keep on the card exactly what the pinned host
  pool (`--host-pool-mib`) cannot hold, plus one expert unit of packing slack; they have nowhere
  else to be. The cache takes everything else, up to what it could ever address
  (`--max-num-seqs` x `--max-model-len`), and every block no live request holds is lent back to the
  expert slab in place. The prefill staging region comes out of the cache's side.
- **Without a lender** — `--deterministic`, a stated `--placement all_vram` or `layer_offload`, or
  no virtual memory management — a block the cache does not use would sit empty, so the experts
  come first and the cache keeps one `--max-model-len` sequence.

The cache never gets less than one `--max-model-len` sequence, and when both sides fit, both are
whole. With a lending cache, `--host-pool-mib` is what moves the split: a larger pool takes experts
off the card and gives the cache more.

**The cache's size decides whether multi-turn works.** The KV side is not only concurrency: prefix
caching holds a finished sequence's blocks so the next turn can reuse them. When the pool cannot
hold the *live sessions*, the cache drops prefixes the next turn wanted and that turn re-prefills
its whole context. At long contexts that is two orders of magnitude of time-to-first-token.
Measured on Qwen3.8-Flash-Next at a 200K context with two ~184K-token sessions and byte-identical
text, the second turn took 173.0 s at a 0.0% prefix hit with a pool that could not hold both
sessions, and 3.0 s at 99.0% with one that could. The fall-off is sharp, and `kv_util` shows a
pool approaching it.

To size it yourself: work out what a token of context costs across **all** paged groups for your
model (the startup log prints this), multiply by the context you actually serve, multiply by the
sessions you want live at once, and compare that with the cache the receipt reports. The cache
already has everything the experts can do without; a larger `--host-pool-mib` (RAM permitting) or
`--kv-cache-dtype fp8` gives it more.

Worked, on Qwen3.8-Flash-Next: two paged groups share one token axis — `kv_attn` at 26.00 KiB and
`kv_qsa_bkey` at 3.25 KiB per 4-token block — so a token of context costs **7.32 KiB**, and a
200K-token sequence is 1.43 GiB. A prefix-cached session occupies roughly 1.8x its context in pool
tokens, so two live 184K sessions want ~662K pool-tokens. That is the number the KV side has to
clear.

What is claimed is not all of the card. The **activation arena** — scratch for the largest step
the engine may run — is taken off the budget *before* it is split, so it is paid for by both
sides. It grows with `--max-num-batched-tokens`, which is why raising the prefill chunk costs
resident experts (§4.5).

**Most of the arena is lent back while the steps are small.** A decode step is a few dozen rows
of the 2048 a prefill chunk carries, so the engine asks the architecture plugin to declare its
activations again at smaller step sizes (the decode rows rounded up to a power of two and at least
256, then four times that, up to the step budget) and packs each into the bottom of the arena. While the recent steps fit a smaller size, the top of the arena holds expert
weights; the first step that needs more takes it back before it runs — a change of address for
the units there, not a copy. Startup prints what each size lends (`activation arena: lends ...`).
A plugin opts in (`RadArchProbe::shape_probe_ok`); one that does not keeps its whole arena.

Related flags:

```sh
--gpu-headroom-mib 96      # VRAM left unclaimed on every card
--kv-cache-dtype fp8       # halve what a token of context costs (a real quantisation)
--host-pool-mib 8192       # pinned host tier for weights
--embedding-placement ram   # the token embedding off the card (below)
--ngram-placement ram       # an n-gram table in RAM instead of read from disk (below)
--weights-disk-tier       # stream what fits neither from the container; needs O_DIRECT
```

**Two lookup tables have a placement of their own**, because a step reads only M of their rows and
never the whole table, so a tier below the card serves them at little or no cost:

| | `vram` | `ram` | `disk` |
|---|---|---|---|
| `--embedding-placement` | on the card, with the other weights | the pinned host pool, read over the link in place | refused |
| `--ngram-placement` | on the card, every card holding all of it | a copy in RAM made at load, read by the card over the link | read from the container through a row cache |

Both default to `auto`, the fastest tier that has room once every other weight is placed. Startup
prints what each resolved to and why. A tier the flag names that cannot be honoured stops the
server with the reason; nothing falls back quietly.

**The token embedding** is the largest model-level weight a dense model has -- 248320 x 5120 in bf16
is 1.18 GiB a card at `--tp 2`. With `ram` it is placed in the pinned host pool before any strategy
runs and the lookup reads it over the link in place, so it costs no VRAM and the KV pool gets its
share. The pool must hold it (`--host-pool-mib`, every rank its own shard). `auto` keeps it on the
card unless the card cannot hold the other weights with it, or a derived KV pool could then not
hold one `--max-model-len` sequence; then it takes `ram` if the host pool can hold it. The table is
found by what reads it: a model-level weight only `embed_lookup` reads. A head tied to it is read by
a GEMM too and is never moved, since it would stream whole every step. `disk` is refused: the lookup
writes activations the card reads, and a host kernel reading the container cannot write them.

```sh
radiance --model m.rad --tp 2 --host-pool-mib 1280 --embedding-placement ram
```

**An n-gram table** (Qwen3.8-Flash-Next's is 47.7 GiB) is read a few rows a token, which is why it
can live off the card at all. `disk` reads its rows from the container through a row cache and
direct reads on the host; about half of a step's rows are first-seen, so steps wait on the drive.
`ram` copies the whole table into host memory at load -- 12 to 40 s more startup, depending on the
page cache -- and every gather reads memory: the card's own gather kernel reads a pinned copy over
the link, so a step has no host work in it at all. `vram` puts it on the card, every card holding
all of it, which needs a card that large. `ram` needs the table's size in memory after the pinned
host pools and the prefix cache's host tier, with 4 GiB to spare, and room under every memory cgroup
the process sits in; it refuses the start short of either and names which. `auto` takes the first
of `vram`, `ram` and `disk` that holds the table. Without a device kernel for the gather (a build
with no libr4d), the table is read on the host: `vram` is refused, and `ram` is a copy in locked
huge pages, which also needs a memlock limit that large (`--ulimit memlock=-1` in Docker,
`LimitMEMLOCK=infinity` under systemd).

**A model larger than the card and the host pool together still serves with
`--weights-disk-tier`** -- and every step is then paced by the drive, which is the trade the flag
names. A routed expert that fits neither VRAM nor the pinned pool lives in the container and
nowhere else; before each routed layer's first expert op, on every pass, the engine reads all of
that layer's on-disk experts into one of two VRAM buffers (a router's picks are not known on the
host, and a prefill chunk routes to nearly all of them anyway). The two buffers come out of the
weight budget -- the plan report's `file buffers` line -- and the reads go through four 32 MiB
pinned windows out of `--host-pool-mib` as long runs of the container, skipping resident experts.
What it is for: serving a checkpoint as it ships to calibrate or compare against, not a
production deployment. The bf16 Qwen3.8-Flash-Next (335 GiB) runs at two cards this way at about
28 s a step, and that step costs the same at one token as at 32K, so give it large prefill steps:

```sh
rad-convert ~/models/Qwen/Qwen3.8-Flash-Next -o flashnext-bf16.rad --max-ctx 4096   # no recipe: bf16
radiance --model flashnext-bf16.rad --tp 2 --placement expert_tiered --weights-disk-tier \
    --host-pool-mib 12288 --deterministic --gpu-headroom-mib 768 \
    --max-num-seqs 32 --max-num-batched-tokens 32768 --num-speculative-tokens 0
```

A production server leaves the flag off, so a budget that does not fit is refused at startup
rather than served from the drive.

**The KV pool is elastic on every run and there is no flag for it.** It holds the blocks it is
actually using plus a buffer it derives — one prefill chunk, or the largest shortfall the
admission path has met in the last few seconds, whichever is larger — and the expert plane grows
into whatever it releases. A pool that held every block it was carved for whether or not a token
was in it would, on a model whose experts do not fit, stream weight over the link for the whole
run to pay for cache nobody is using.

What it leans on is `--gpu-headroom-mib`. That reserve is what the card keeps for allocations
this engine does not make — chief among them a kernel's code object, which HIP loads on the
kernel's *first launch*, not at startup. A deployment whose budget settles *below* its stated
headroom has already spent it, and the slab growing into what is left collides with the first
kernel that has never run: a draft head's, a sampler chain's, whatever the early requests did not
reach. It fails its load and faults inside the HIP runtime, which arrives as a segfault in a
kernel that is correct, on a server that had been serving fine.

Startup measures that shortfall and holds it back on top of the reserve, so a run that did not
quite keep its headroom is safe rather than fatal — but it is costing you the VRAM twice. Raise
`--gpu-headroom-mib` past the number startup names, so the budget resolver sees it and the expert
plane is sized against it instead of trimmed afterwards.

What the engine itself allocates after the card is read is charged before the split, so it never
comes out of the headroom: the step staging, the KV tiers' copy staging, and its hardware queues.
A queue costs more than its ring — the driver backs each one with VRAM of its own, 33 MiB on an
R9700 — so startup measures one and the receipt's `hardware queues` line shows how many a card
holds at that price.

### 4.4 Long context

```sh
radiance --model m.rad --max-model-len 200000 --kv-cache-dtype fp8 --max-num-seqs 8
```

`--max-model-len` does not reserve anything per request — it sets the worst case the pool is sized
against. `0` takes the model's training context.

`--kv-cache-dtype fp8` is an E4M3 cache with per-(sequence, head) descales. It halves what a token
of context costs, which at a production context is most of the pool. It is a real quantisation and
the model computes a slightly different function with it, so it is stated rather than inferred.
Every architecture serves both widths, and the flag sets every attention cache of the deployment —
a drafter's (DSpark, DFlash2) as well as the trunk's. The delta-net state and the QSA indexer's
key tail are recurrent state rather than an attention cache and keep their own widths.

**Many sequences against a long context.** `--max-num-seqs` runs to 32 on every model here, and
what it costs is not context — a sequence holds only the blocks its tokens occupy — but what exists
per sequence whether it is used or not:

- **Recurrent state.** On a hybrid model every sequence owns its delta-net state, and with
  speculation two copies of it. Qwen3.8-Flash-Next keeps 54 MiB a copy on each card, so 32
  sequences take 3.4 GiB of the cache share before a single token is cached. The 27B in bf16,
  with no experts to give way, then holds about 117K tokens of attention cache, less than one 200K
  request, and startup warns that a longer one would be refused. The 27B in FP8 holds 2.8
  full-length sessions at 32, the 35B-A3B ten.
- **Block tables.** The step batch's tables are carved for every sequence at the full context.
  They are charged to the VRAM budget (`step staging` in the startup receipt), so they come out of
  the cache share and not out of `--gpu-headroom-mib`.

The startup line `the pool backs N% of the worst case ... M full-length sessions` counts only the
part of the pool that holds context, so M is the number to compare with the sessions you expect
to keep live. With `--tp-wire wht6`, a decode step above the row count startup names is
all-reduced lossily like a prefill chunk, which a busy step reaches: at 8 sequences and depth 3,
Flash-Next's decode steps are already over it.

### 4.5 Prefill speed: the chunk

A long prompt is prefilled in chunks, and `--max-num-batched-tokens` is how big one is. It is the
single biggest lever on time-to-first-token on a mixture-of-experts model, because a chunk sweeps
the whole expert plane once however many tokens are in it — so four times the chunk is a quarter
of the sweeps.

```sh
radiance --model m.rad --max-num-batched-tokens 2048
```

On a 48-layer MoE at 200K context on two cards, prompts of 8K and 24K tokens:

| chunk asked for | 512 | 1024 | **2048** | 4096 |
|---|---|---|---|---|
| chunk actually run | 512 | 1024 | **2048** | 2048 |
| prefill | 1750 tok/s | 2200 | **2706** | 2620 |

At full context the same change is the difference between waiting and waiting a lot: a
185,253-token prompt prefills in **91 s at a 2048 chunk against 138 s at 512**.

**The 4096 column is the trap, and the engine warns about it.** On a hybrid model the chunk is
clamped to `--checkpoint-interval` (2048 by default) *before* the token budget, so asking for 4096
runs 2048 chunks and pays for 4096 anyway: the activation arena grows with the budget and is
charged to the VRAM budget *before* it is split between weights and KV, so the expert plane loses
residency for a chunk it never got. Startup prints which of the two bound it:

```
chunk geometry: quantum 64 (attn block ..., state chunk ...), checkpoint interval 2048,
max_tok 4096 -> prefill chunks of at most 2048 (the checkpoint interval, not the budget)
```

Read that line, and `RADIANCE_LOG_STEPS=1` prints the token count of every step if you want to see
it per step.

Raising both — `--max-num-batched-tokens 4096 --checkpoint-interval 4096`, which does run 4096
chunks — does not help either: 2610 tok/s against 2748, because the arena takes
another 239 resident experts (9219 to 8980) and the extra amortisation does not pay for them. On
a model whose prefill is bound by the link carrying non-resident experts, the chunk stops winning
as soon as it starts costing residency.

**It does not cost the requests already running.** A longer chunk holds the device longer per
step, so a decode step queued behind one waits longer — but the prefill it is queued behind
finishes sooner, and that is the larger term. Four decoders with a long prompt landing beside
them finished in 29.5 s at a 2048 chunk against 31.9 s at 512.

**And check it did not cost you the prefix cache.** The arena it grows is charged before the
split (§4.3), so the KV pool shrinks a little — here 516,864 → 499,920 pool tokens, 3.3% — and
§4.3's cliff is sharp enough that 3% is worth confirming rather than assuming. On this shape it
costs nothing: two ~180K sessions, first turns back to back, then the first session's second turn.

| chunk | session A turn 1 | session B turn 1 | A turn 2 | cached | evictions |
|---|---|---|---|---|---|
| 512 | 137.7 s | 130.0 s | 1.42 s | 178,176 of 179,585 | 0 |
| 2048 | **90.6 s** | **93.2 s** | **0.97 s** | 178,176 of 179,585 | 0 |

Identical hit, no evictions, `kv_util` 0.70 → 0.73. Turn one is 34% faster and turn two stays
where it was. Do this on your own shape before you keep the change: `prefix_hit_rate` and
`prefix_evictions` on `/stats` are the two numbers, and the failure looks like turn two taking
as long as turn one.

**What it does cost is decode, and it is small.** The 239 resident experts the arena takes have to
come from somewhere, and steady-state decode is where. Decode step time from `scripts/cstep2.sh`,
which divides by the decode step counter rather than by wall clock, two passes each, generations
long enough that no sequence retires inside the window:

| chunk | one stream | eight streams |
|---|---|---|
| 512 | 22.35 / 22.29 ms/step | 39.52 / 40.52 ms/step |
| 2048 | 22.42 / 22.48 ms/step | 39.52 / 40.73 ms/step |

**+0.6% on a single-stream decode step**; at eight streams the difference is 0.11 ms, below what
the harness resolves. Against a prefill that is 55% faster and a first turn that goes from 138 s
to 91 s, that is the right side of the trade — but it is a real cost and it lands on every turn,
so measure it on your own shape rather than taking the sign on faith.

**Changing it changes the text.** A different chunk is a different batch shape, so the sums inside
a prefill run in a different order, and at a tie that is a different token. A fixed prompt at a
fixed seed gives one answer at 512 and a different one at 2048, and each is byte-identical across
runs and server restarts at its own setting. Reproducibility holds *at a configuration*, which is what it claims; this flag is part of the
configuration in the same way `--tp` and `--num-speculative-tokens` are. Pin it before you record
anything you intend to compare against later.


### 4.6 Speculative decoding

If the container carries a draft head, it is used by default at the depth the plugin states:

```sh
radiance --model m.rad                              # auto: the drafter's own operating point
radiance --model m.rad --num-speculative-tokens 0   # off
radiance --model m.rad --num-speculative-tokens 3   # a stated depth
```

There is **no `--draft-model`**. This engine loads its drafter from the container — merge one at
conversion time with `rad-convert --draft-model DIR`, or use an MTP head that already ships inside
the target checkpoint, which needs no flag at all. Passing `--draft-model` to the engine is an
error that says so rather than being quietly ignored.

When you measure speculation, **measure step time, not tokens per second.**

### 4.7 Two cards

```sh
radiance --model m.rad --tp 2
radiance --model m.rad --tp 2 --tp-wire wht6 --tp-wire-min-kb 128
```

`--tp-wire` chooses the cross-rank all-reduce payload. `exact` (default) is the bf16 all-reduce.
`wht6` accepts a lossy wire: a Walsh-Hadamard-rotated 6-bit payload, 2.4× smaller over PCIe. The
sum a rank receives is then not the sum it would have received and nothing downstream can tell,
which is why it is a named flag and not a heuristic.

`--tp-wire-min-kb` is the floor below which messages are still served exactly — 128 KiB leaves
single-sequence decode exact and compresses prefill, which is where the win is.

At four ranks `wht6` selects a different lossy wire with the same floor: tiered int8 over a
two-shot all-reduce, both hops quantised. Eight ranks have no lossy wire, and `--tp 8 --tp-wire
wht6` stops at startup naming the all-reduce it found no kernel for. The exact wire at four and
eight ranks is one-shot up to 1 MiB a message and two-shot above it, which sends
2·(ranks−1)/ranks of the message per rank instead of (ranks−1) times it; the two produce the same
bits.

On a model with a gated residual (Qwen3.8-Flash-Next) the all-reduce rides inside the residual's
write, with the MoE gather in front of it. At two ranks that holds on either wire: the rotated
forms of `ar_hc_write` and `ar_gather_hc_write` produce the same bytes as the unfused chain on the
same wire. There `wht6` is worth +5.6–6.3% of prefill at a 2048-token chunk (20K prompt 5556 →
5908 tok/s; a 23K extension at 81K depth 4998 → 5278) and −1.3% of the eight-stream decode step;
one-stream decode stays exact. At four ranks only the exact wire has a fused form, and it takes
the decode and verify steps — up to `--max-num-seqs` × (1 + `--num-speculative-tokens`) rows. A
prefill chunk, and a message that goes out on the tiered-int8 wire, runs the unfused chain, where
the all-reduce's own size choice between one-shot and two-shot applies.

**The cards must reach each other's memory directly** (peer to peer, P2P). The all-reduce
kernels push into the other card's memory. At launch radiance asks the driver, for every pair of
the cards it will use, whether that works, and logs the answer. When some pair cannot, those
kernels are not selected, and startup stops naming `device.p2p` instead of the first all-reduce
faulting or hanging. Whether P2P works is decided by the platform: the IOMMU setting, the PCIe
topology, and whether the cards sit behind a switch that allows peer traffic. The startup line
names the pairs that failed.

### 4.8 Finished conversations: copied to host, VRAM given up on demand

By default a finished conversation's KV blocks stay in VRAM, held by the prefix cache, until the
cache evicts them under pressure — and an evicted prefix is re-prefilled from scratch next turn.

With a host tier, a conversation is **copied** to host memory the moment its request finishes, and
keeps its VRAM copy. From then on it is *clean*: its VRAM can be given up for nothing, and it is,
as soon as something wants it — the expert slab's loan, or a request the pool cannot otherwise
serve. A conversation that comes back before that resumes from VRAM; one that comes back after is
restored from host memory and keeps its host copy, so its next turn copies only what that turn
added. There is no idle threshold on the way out of VRAM.

With a disk tier the same happens one level down: every host copy is written on to disk in the
background and keeps its host slot, and once the write has landed the slot can be given up for
nothing. A full host tier gives up its oldest slots when a new copy or a restore needs one, so a
conversation that no longer fits in host memory is on disk already rather than lost.

```sh
radiance --model m.rad \
    --prefix-cache-host-mib 8192 \
    --prefix-cache-disk-mib 65536 --prefix-cache-dir /var/cache/radiance
```

- `--prefix-cache-host-mib N` — **this is the switch.** Without a size, nothing is copied. It holds
  attention blocks only; a hybrid model's linear-state snapshots are in host memory already, in the
  `--checkpoint-slots` store. Size it for the conversations you expect to come back: one that does
  not fit comes back from disk instead.
- `--prefix-cache-disk-mib N` — the disk tier's size. Everything that reaches host memory is
  written here too, snapshots included, so a full snapshot store detaches its least recently used
  snapshot instead of losing it.
- `--prefix-cache-dir DIR` — where the disk tier lives. **Not `--weights-disk-tier`**, which is about
  *weights*. The disk tier holds conversation content unencrypted.
- Recovering a 200K-token turn costs a copy over PCIe; recomputing it costs minutes.

On a hybrid model the recurrent state matters as much as the attention blocks — restoring the
blocks alone leaves the session re-running every linear layer over its whole transcript, which
every hit-rate counter reports as a hit. The snapshots never take VRAM: each is copied to host
memory after the step that produced it and back before the step that resumes from it. `/sessions` shows what is stored, where, and whether the
linear half is being served.

### 4.9 What a request that leaves a field out gets

Every default below applies only to a request that does not name the field. A request that names
it wins. Without any of these flags, the server behaves exactly as it always has.

**Temperature, top-k, top-p, min-p.** First one that exists wins:

1. `--temp` / `--top-k` / `--top-p` / `--min-p` on the command line
2. `--generation-config PATH` — a `generation_config.json`
3. `<model>.generation.json` beside the container
4. the container's own `generation.*` metadata
5. the built-in defaults

```sh
radiance --model m.rad --temp 0.7 --top-p 0.8 --top-k 20
```

**The rest of the sampler** has a flag per field, spelled like the request field:
`--presence-penalty`, `--frequency-penalty`, `--repetition-penalty`, `--repeat-last-n`,
`--typical-p`, `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`, `--dry-penalty-last-n`,
`--dry-sequence-breakers` (a JSON array), `--xtc-probability`, `--xtc-threshold`. They are checked
against the same ranges a request is, at startup:

```sh
radiance --model m.rad --presence-penalty 1.5 --repetition-penalty 1.05
```

**How much a request may generate.**

- `--default-max-tokens N` sets `max_tokens` for a request that sends none. The default, `auto`,
  is whatever the context leaves after the prompt, as in vLLM. Many clients send no `max_tokens`,
  and a reasoning model can think for thousands of tokens before it answers: a small default ends
  those answers inside the reasoning, with a `finish_reason: "length"` that chat clients do not
  show.
- `--max-tokens-cap N` is the most any request may generate. A larger `max_tokens` is cut to it
  rather than refused, and the reply ends with `finish_reason: "length"`, the same as the context
  bound (§5.8).

```sh
radiance --model m.rad --default-max-tokens auto --max-tokens-cap 32768
```

**Chat template variables.** `--chat-template-kwargs` hands variables to the chat template on
every chat request, the way a request's own `chat_template_kwargs` does:

```sh
radiance --model m.rad --chat-template-kwargs '{"enable_thinking": false}'
radiance --model m.rad --chat-template-kwargs @template-vars.json
```

- It takes a JSON object, or `@FILE` to read one. Give it more than once and a later key replaces
  an earlier one.
- A key the request's own `chat_template_kwargs` names is the request's.
- Thinking is one setting with three spellings. A request that says anything about thinking
  (`enable_thinking`, at the top level or in `chat_template_kwargs`, or `reasoning_effort`)
  replaces the server's `enable_thinking`. One that names a `reasoning_effort` replaces the
  server's effort too.
- `messages`, `tools`, `bos_token`, `eos_token` and `add_generation_prompt` are refused: the
  server sets those itself.
- At startup a one-message chat is rendered with the variables. A value the template refuses
  stops startup with the template's own message, instead of failing every request.
- `/tokenize` renders `messages` with the same variables, so the count it gives is the prompt
  chat serves.

**`--reasoning-effort` is worth knowing about.** It is the default for chat requests that do not
set one, and the value goes to the model's chat template verbatim — so the legal set is the
template's. Qwen3.8 takes `xhigh|medium|low` and **defaults to `xhigh`**, which on a short task can
spend the entire token budget inside `<think>` and return a turn with empty content and
`finish_reason: "length"`. `"none"` turns thinking off. It cannot be combined with a
`reasoning_effort` or `enable_thinking` key in `--chat-template-kwargs`. A value the template
refuses is a warning at startup: requests that send their own effort still work.

```sh
radiance --model m.rad --reasoning-effort medium
```

**`--thinking-token-budget N`** caps the reasoning instead of its depth. At N tokens inside the
reasoning block the block is closed, by masking the next position to its end token
(`</think>`), and the answer follows. This makes a client's `max_tokens` buy an answer, where
otherwise the reasoning could use all of it. It is the default for chat requests that send no
`thinking_token_budget` of their own (§5.2). It needs a reasoning block that ends in a single
token, as Qwen's and MiniCPM's do. On a model whose block does not, startup stops and says why.

**`--reasoning-format none`** leaves a reasoning block inline in `content` instead of returning it
as `reasoning_content`, for a client that reads it there.

`/server_info` lists every effective default under `request_defaults` (§6.2), and startup prints
one line naming each one the command line moved.

### 4.10 Every flag

`radiance --help` prints this table. Grouped here by what you are doing:

**Model and plugins**

| flag | |
|---|---|
| `--model PATH` | the `.rad` container, or a checkpoint served as it stands (§3.4) (**required**) |
| `--radiance-home DIR` | where `kernels/` and `architectures/` live |
| `--kernels LIST` | kernel plugin hierarchy, comma-separated, first wins |

**Batch shape**

| flag | default | |
|---|---|---|
| `--max-num-batched-tokens N` | 8192 | largest token count one step may carry |
| `--max-num-seqs N` | 256 | largest number of sequences one step may carry |
| `--max-model-len N` | 0 | context bound; 0 takes the model's training context |

**Memory**

| flag | default | |
|---|---|---|
| `--gpu-headroom-mib N` | 96 | VRAM left unclaimed on every card |
| `--vram-weights-mib N` | derive | the resident weight slab |
| `--vram-kv-mib N` | derive | the paged KV pool |
| `--kv-cache-dtype T` | bf16 | `bf16` or `fp8` |
| `--host-pool-mib N` | 0 | pinned host tier (weights) |
| `--embedding-placement WHERE` | auto | `auto`, `vram` or `ram`: the token embedding on the card or in the pinned host pool, read in place; `disk` is refused. See §4.3 |
| `--ngram-placement WHERE` | auto | `auto`, `vram`, `ram` or `disk`: an n-gram table on the card, as a copy in RAM made at load and read by the card, or read from the container. See §4.3 |
| `--weights-disk-tier` | off | read weights that fit neither VRAM nor the host pool from the container (O_DIRECT); routed experts a layer at a time |

**Idle session tiers**

| flag | default | |
|---|---|---|
| `--prefix-cache-host-mib N` | 0 (off) | copy finished conversations' KV to host memory |
| `--prefix-cache-disk-mib N` | 0 (off) | and copy them on to disk |
| `--prefix-cache-dir DIR` | — | where the disk tier keeps them |

**Placement and state**

| flag | default | |
|---|---|---|
| `--placement MODE` | auto | `all_vram`, `layer_offload`, `expert_tiered`, `auto` |
| `--deterministic` | off | pin placement to the loader's plan, so the resident set cannot change under the heat engine |
| `--checkpoint-interval N` | 2048 | linear-attention state checkpoint interval, in tokens |
| `--checkpoint-slots N` | 8 | linear-state snapshots the server may hold at once, **in total**, not per sequence. They live in pinned host memory, not VRAM: one slot is a snapshot of every rank's shard (113 MB on Flash-Next at `--tp 2`), allocated at startup. A snapshot is copied off the card after the step that produced it and back before the step that resumes from it. With a disk tier a full store detaches its least recently used snapshot to disk instead of losing it |
| `--checkpoint-policy NAME` | geometric-backoff | also `tip-only`, `keep-all` |
| `--no-prefix-cache` | — | disable prefix caching |

**Parallel**

| flag | default | |
|---|---|---|
| `--tp N` | 1 | tensor-parallel ranks |
| `--tp-wire MODE` | exact | `exact` or `wht6`, the lossy wire (§4.7) |
| `--tp-wire-min-kb N` | 128 | smallest all-reduce the lossy wire is used for |

**Speculation**

| flag | default | |
|---|---|---|
| `--num-speculative-tokens N` | auto | speculative window; 0 disables |
| `--draft-model PATH` | — | **not implemented** — merge at convert time |

**Server**

| flag | default | |
|---|---|---|
| `--host ADDR` | 0.0.0.0 | listen address |
| `--port N` | 8000 | listen port |
| `--api-key KEY` | — | require `Authorization: Bearer KEY` on every request but `/health`, `/ping` and the dashboard page (§6.6) |
| `--served-model-name NAME` | container's name | the model id `/v1/models` lists, the metrics carry and a response names when its request named none. A request may name any model; the one loaded serves it |
| `--mm-max-patches N` | auto | patches one vision-encoder pass carries, and so the largest image (a patch is 16x16 pixels). `auto` is 16384 when the container carries a vision tower; 0 serves text only and keeps the tower off the card |
| `--override-chat-template PATH` | container's | a Jinja chat template file served in place of the one the container carries. The reply format (reasoning markers, tool calls) is derived from it too, unless the architecture plugin declares one. An unreadable or empty file stops startup |
| `--max-queued-requests N` | 8 × `--max-num-seqs` | requests admitted and unfinished at once; past it, 429 |
| `--http-threads N` | one per core, at least 4 | HTTP workers kept while idle |
| `--read-timeout S` | 30 | seconds a client may go silent while sending a request |
| `--write-timeout S` | 600 | seconds one write to a client may block |
| `--keep-alive-timeout S` | 5 | seconds an idle connection is held open between requests |
| `--max-body-mib N` | 512 | the largest request body; images and video arrive base64 inside it |
| `--retry-after S` | 1 | the `Retry-After` a 429 carries |
| `--no-cors` | CORS on | send no CORS headers (§6.6) |

**Request defaults and bounds** — what a request that leaves a field out gets (§4.9). A request
that names the field wins.

| flag | default | |
|---|---|---|
| `--generation-config PATH` | — | sampler defaults from a `generation_config.json` |
| `--temp F`, `--top-k N`, `--top-p F`, `--min-p F` | from container | sampler defaults |
| `--presence-penalty F`, `--frequency-penalty F` | 0 | [-2, 2] |
| `--repetition-penalty F`, `--repeat-last-n N` | 1, 64 | > 0; ≥ -1 (-1 is all) |
| `--typical-p F` | 1 | (0, 1] |
| `--dry-multiplier F`, `--dry-base F`, `--dry-allowed-length N`, `--dry-penalty-last-n N` | 0, 1.75, 2, -1 | DRY; a multiplier of 0 is off |
| `--dry-sequence-breakers JSON` | `["\n", ":", "\"", "*"]` | a JSON array of at most 16 strings |
| `--xtc-probability F`, `--xtc-threshold F` | 0, 0.1 | [0, 1] |
| `--default-max-tokens N` | `auto` | `max_tokens` for a request that sends none; `auto` is what the context leaves |
| `--max-tokens-cap N` | none | the most any request may generate; a larger `max_tokens` is cut to it, not refused |
| `--max-n N` | the larger of 8 and `--max-num-seqs` | the largest `n` a request may ask for |
| `--max-stop-strings N`, `--max-stop-bytes N` | 64, 4096 | how many stop strings a request may send, and how long each may be |
| `--chat-template-kwargs JSON` | — | variables for the chat template on every chat request; a JSON object or `@FILE`, repeatable |
| `--reasoning-effort S` | template's own | default `reasoning_effort` for chat |
| `--thinking-token-budget N` | none | default `thinking_token_budget` for chat: the reasoning block is closed at N tokens |
| `--reasoning-format MODE` | auto | `auto` returns reasoning as `reasoning_content`; `none` leaves it inline in `content` |

**Media** — each replaces what the container's preprocessor configuration states. They do nothing
for a model that takes no images or video.

| flag | default | |
|---|---|---|
| `--image-min-pixels N`, `--image-max-pixels N` | container's | the band an image is resized into; one encoder pass (`--mm-max-patches`) still bounds the top |
| `--video-min-pixels N`, `--video-max-pixels N` | container's | the same for a video's sampled frames together |
| `--video-fps F` | container's, else 2 | frames sampled per second of video |
| `--video-min-frames N`, `--video-max-frames N` | container's, else 4 and 768 | the fewest and most frames sampled from a video |
| `--video-max-frame-tokens N` | container's, else 768 | the most prompt tokens one frame may become; 0 is no cap |
| `--max-source-pixels N` | 67108864 | the largest image or frame the decoder accepts, before any resize |

**Quality measurement** — instead of serving; see [TOOLS.md](TOOLS.md#the-kl-mode)

| flag | |
|---|---|
| `--kld-record DIR` | run `--kld-corpus` through prefill and write this model's log-probabilities at every scored position to `DIR`: the reference a quantised model is measured against |
| `--kld-ref DIR` | score this model against the reference in `DIR`: mean, median, 99/99.9/99.99th percentile KL divergence and top-1 agreement |
| `--kld-corpus FILE` | JSONL, one `{"prompt", "score_from"?, "source"?}` a line; `--kld-record` only |
| `--kld-out FILE` | the `--kld-ref` report as JSON, and every position's numbers in `FILE.rows` |

**Diagnostics**

| flag | |
|---|---|
| `--debug-graph` | print the declared graph and exit |
| `--debug-selection` | print the selection table |
| `--debug-placement` | print the placement plan |
| `--debug-accept-reference-kernels` | serve even when ops resolve to `libref`. **Not a production configuration** |
| `--profile-ops` | per-op device timing |
| `--live` | the live terminal view |
| `-v` / `-vv` | verbose / very verbose |

---

## 5. Talk to it

### 5.1 The endpoints

| | |
|---|---|
| `POST /v1/chat/completions` | chat, with tools, reasoning and streaming |
| `POST /v1/completions` | raw text completion |
| `POST /v1/embeddings` | needs a model with an embedding head, else 501 |
| `GET /v1/models`, `/v1/models/:id` | what is loaded |
| `GET /health`, `/ping`, `POST /ping` | liveness |
| `GET /load` | current load |
| `GET /version` | server version |
| `GET /server_info` | the whole configuration of the running engine |
| `GET /metrics` | Prometheus |
| `POST /tokenize`, `POST /detokenize` | |
| `GET /get_tokenizer_info` | |
| `POST /reset_prefix_cache` | drop every cached prefix |
| `GET /` or `/dashboard` | the web dashboard |
| `GET /stats`, `/graph`, `/sessions` | the JSON the dashboard is drawn from |

Method spellings follow vLLM's, including the ones that look odd: `/tokenize` is POST because its
input is a body, and `/ping` answers both GET and POST because different load balancers send
different ones.

A text prompt to `/v1/completions`, `/v1/embeddings` or `/tokenize` reads the model's control
tokens as tokens, as vLLM and llama.cpp do: a client that renders its own chat template can send
`<|im_start|>user\n…` and the model sees the turn markers. The BOS the model declares is prepended
unless the prompt already opens with it.

### 5.2 Chat

```sh
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "m",
    "messages": [{"role": "user", "content": "Explain merge sort briefly."}],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
```

If the container carries no chat template this endpoint answers **501** and the startup log says
so — use `/v1/completions` instead.

**Turning thinking off.** Three spellings reach the server for one question; all work:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
{"enable_thinking": false}
{"reasoning_effort": "none"}
```

Reasoning text comes back in `choices[].message.reasoning_content`, separate from `content`.

**Reasoning counts against `max_tokens`**, as in OpenAI's API, vLLM and llama.cpp: every
generated token does. A reply that runs out inside its reasoning comes back with empty
`content` and `finish_reason: "length"`. `usage.completion_tokens_details.reasoning_tokens` says
how many of `completion_tokens` were reasoning, the block's end token included, so a client can
tell that case from a model that answered nothing.

**Capping the reasoning.** `thinking_token_budget` (vLLM's name; `thinking_budget`, Qwen's, is
read as the same field) is how many tokens the reasoning may run to. At that count the block is
closed and the model answers from what it has reasoned so far:

```json
{"max_tokens": 4096, "thinking_token_budget": 2048}
```

The count starts after the opening `<think>`. A reasoning block that ends sooner is not
affected. The close is forced by the sampler, so it holds under speculative decoding and at any
temperature. A `grammar` or `response_format` that does not allow the close at some position
keeps that position, and the close comes at the next position that allows it. The budget and
`reasoning_tokens` need the block's end to be one token. A model whose end is not one token
reports no `completion_tokens_details` and refuses a budget with a 400 that says why.

### 5.3 Streaming

```sh
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Count to ten."}],
       "max_tokens":100,"stream":true,
       "stream_options":{"include_usage":true}}'
```

Server-sent events, OpenAI's delta format, terminated by `data: [DONE]`.

### 5.4 Tool calls

```sh
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "m",
    "messages": [{"role":"user","content":"What is in /etc/hosts?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read",
        "description": "read a file",
        "parameters": {"type":"object",
                       "properties":{"path":{"type":"string"}},
                       "required":["path"]}
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 300
  }' | jq '.choices[0].message.tool_calls'
```

Calls come back in `message.tool_calls` with `finish_reason: "tool_calls"`. No `<function>` or
`<param>` markup ever reaches `content`. `scripts/mctools.sh` and `scripts/mcstream.sh` assert
exactly that, including on raw streamed fragments — content cannot be retracted once sent, so a
repair applied at the end is not a fix.

A tool's `parameters` schema shapes the grammar the call is written under. A property whose schema
accepts no value — `false`, or `{"not": {}}`, which is how TypeBox and Arktype export `never` — is
an argument no call writes. Any other `not` is refused with a 400 naming it: the grammar cannot
express a complement, and dropping the constraint silently would serve a call the tool forbids.

Length bounds are held exactly: `maxLength`, `minLength`, `maxItems` and `minItems` up to about
200,000, which costs the first request carrying the tool under a second to compile on a
quarter-million-token vocabulary and nothing per token after that. A bound past 2^31 (as
`Number.MAX_SAFE_INTEGER` is) is no bound, since no context holds that much. One in between is
refused with a 400 naming the property's rule and the count. The same holds for `response_format`
schemas and `{m,n}` in a GBNF grammar.

### 5.5 Constrained output

```sh
# JSON object
-d '{"model":"m","messages":[...],"response_format":{"type":"json_object"}}'

# a JSON schema
-d '{"model":"m","messages":[...],
     "response_format":{"type":"json_schema",
       "json_schema":{"schema":{"type":"object",
         "properties":{"city":{"type":"string"},"pop":{"type":"integer"}},
         "required":["city","pop"]}}}}'

# a GBNF grammar directly (cannot be combined with response_format)
-d '{"model":"m","prompt":"...","grammar":"root ::= \"yes\" | \"no\""}'
```

**On chat, a constraint passes through the chat template and may not survive it.** The template
gets to rewrite or drop `grammar` and `response_format` — a tool-calling template produces its own
grammar, for instance — so whether a constraint can be enforced is a property of the model's
template and not only of the server. When the one you asked for is not there at the end, the
request is **refused**:

```
grammar: this model's chat template produced no grammar for it, so the constraint could not
be enforced
```

That is the refused-not-ignored rule (§5.8) doing its job: an unconstrained 200 would leave you
parsing free prose against a schema you were told was enforced. `/v1/completions` has no template
in the way, so the same grammar works there — which is why the example above uses it.

### 5.6 Completions, embeddings, tokenising

```sh
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","prompt":"def fib(n):","max_tokens":128,"temperature":0}'

curl -s http://localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"m","input":"hello"}'

curl -s http://localhost:8000/tokenize -H 'Content-Type: application/json' \
  -d '{"prompt":"hello world","return_token_strs":true}'

curl -s http://localhost:8000/tokenize -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'    # applies the chat template

curl -s http://localhost:8000/detokenize -H 'Content-Type: application/json' \
  -d '{"tokens":[15339,1917]}'
```

`/tokenize` takes `prompt` **or** `messages`, not both.

### 5.7 Request fields

| field | notes |
|---|---|
| `model` | accepted and not enforced — one server serves one model |
| `messages` / `prompt` / `input` | chat / completion / embedding |
| `max_tokens`, `max_completion_tokens` | default: what the context leaves after the prompt (`--default-max-tokens`). Clamped to what the context leaves, and to `--max-tokens-cap`, rather than refused — see §5.8 |
| `temperature` | [0, 2] |
| `top_p` | (0, 1] |
| `top_k` | ≥ 0, 0 disables |
| `min_p`, `typical_p` | |
| `presence_penalty`, `frequency_penalty` | [-2, 2] |
| `repetition_penalty`, `repeat_last_n` | |
| `xtc_probability`, `xtc_threshold` | |
| `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` | |
| `seed` | absent means "pick one" — **not** seed 0 |
| `n` | 1 to `--max-n`, by default the larger of 8 and `--max-num-seqs`. Fan-out over the same prompt; the prefix cache makes it nearly free after the first |
| `priority` | |
| `stop` | a string or an array; at most 64 strings of 4096 bytes (`--max-stop-strings`, `--max-stop-bytes`) |
| `ignore_eos` | generate to `max_tokens` whatever the model emits |
| `stream`, `stream_options.include_usage` | |
| `logprobs`, `top_logprobs` | **not implemented** — refused, not returned as nulls. `/server_info` reports `logprobs: false` |
| `tools`, `tool_choice`, `parallel_tool_calls` | chat only |
| `add_generation_prompt`, `chat_template_kwargs` | chat only |
| `reasoning_effort`, `enable_thinking` | chat only |
| `thinking_token_budget`, `thinking_budget` | chat only. ≥ 0: the reasoning tokens after which the block is closed — see §5.2. Default `--thinking-token-budget`, else none |
| `preserve_thinking` | chat only. The chat template variable of that name, as llama.cpp maps it; Qwen3.6 and later keep earlier turns' reasoning in the prompt with it |
| `response_format` | chat only. Mutually exclusive with `grammar` |
| `grammar` | a GBNF grammar directly. On chat it passes through the chat template and a template that drops it makes the request a refusal — see §5.5 |
| `echo` | `/v1/completions` only |
| `encoding_format` | `/v1/embeddings` only: `float` or `base64`. `dimensions` is refused — the embedding width is fixed |

Any field the request omits keeps the deployment default (§4.9). A field the request names wins.

### 5.8 A parameter this server does not implement is **refused**, not ignored

This is the one place radiance deliberately differs from what an OpenAI client may expect, and it
will be the first thing you hit pointing an existing client at it:

```json
{"error": {"message": "mirostat: mirostat is not one of the device samplers (spec 13)", ...}}
```

The rule is that **a request that returns 200 means every field of it was honoured**. Silently
dropping a sampler setting and answering 200 is how a deployment ends up serving a model at
settings nobody chose.

Three lists decide it:

- **Accepted** — the table above.
- **Inert** — OpenAI metadata that genuinely does nothing anywhere, accepted and ignored so the
  official SDKs work: `user`, `metadata`, `store`, `service_tier`, `safety_identifier`,
  `prompt_cache_key`.
- **Refused with a reason** — known, not implemented, each named:

| refused | because |
|---|---|
| `logit_bias` | the device sampler has no per-token bias stage |
| `functions`, `function_call` | deprecated — send `tools` / `tool_choice` |
| `best_of` | sampling n candidates and returning the best is not implemented |
| `suffix` | infilling is not implemented |
| `mirostat`, `mirostat_tau`, `mirostat_eta` | not one of the device samplers |
| `top_n_sigma`, `dynatemp_range`, `dynatemp_exponent` | not one of the device samplers |
| `lora` | explicitly out of scope |
| `modalities`, `audio`, `prediction` | not implemented |
| `continue_final_message` | assistant prefill is not implemented |
| `cache_prompt` | prefix caching is a server-wide setting, not per request |

Anything else unrecognised is refused too, naming the key.

**The one exception is `max_tokens`.** If `prompt + max_tokens` would exceed the context, the
server clamps `max_tokens` to what is left and serves the request:

```
prompt 72,170 + max_tokens 128,000  >  200,000 context
                                    ->  served with max_tokens 127,830
```

It is an exception because the other keys are *settings* — dropping one serves a different model
than the caller asked for — and this one is a *bound*, whose whole purpose is to stop generation
early and which the context was going to stop earlier anyway. Refusing would not produce more
tokens than clamping does; it would produce none. The completion ends with `finish_reason:
"length"` and `usage` reports what it actually got, so the caller can still see what happened, and
`-v` logs the adjustment.

A prompt that does not fit **on its own** is still a 400 — there is no window to generate into, so
no `max_tokens` would have made it servable — and the error names how many tokens have to go:

```
prompt: 220470 prompt tokens does not fit the 200000 token context, which leaves nothing to
generate into; shorten the prompt by at least 20471 tokens
```

At the boundary, where an off-by-one would hide: a 198,729-token prompt asking for `max_tokens`
128,000 against a 200,000 context is served, generates **1,271** tokens — exactly the room left —
and ends `finish_reason: "length"`. `POST /tokenize` is how to size a prompt that
precisely without running the model.

---

### 5.9 What a response says about its own speed

Every chat and completion response carries a top-level `timings` object for the request. It uses
llama.cpp's field names, so a proxy or dashboard built against llama-server (llama-swap's activity
view, for one) reads it unchanged:

```json
"timings": {
  "cache_n": 1024, "prompt_n": 71, "prompt_ms": 284.0,
  "prompt_per_token_ms": 4.0, "prompt_per_second": 250.0,
  "predicted_n": 64, "predicted_ms": 711.0,
  "predicted_per_token_ms": 11.29, "predicted_per_second": 88.6,
  "draft_n": 78, "draft_n_accepted": 37
}
```

| field | meaning |
|---|---|
| `prompt_n` | prompt tokens this request computed. Tokens the prefix cache supplied are not in it |
| `cache_n` | prompt tokens the prefix cache supplied. `prompt_n + cache_n` is the prompt |
| `prompt_ms` | from the first step the scheduler gave the request to its first token: the prefill. The time it waited for admission is not in it |
| `prompt_per_second`, `prompt_per_token_ms` | `prompt_n` over `prompt_ms`. A prompt the cache mostly supplied computes few tokens, so this is the speed of the prefill that actually ran, not the prompt length over the time to first token |
| `predicted_n` | tokens generated, the same as `usage.completion_tokens` |
| `predicted_ms` | from the first generated token to the last |
| `predicted_per_second`, `predicted_per_token_ms` | the `predicted_n − 1` tokens after the first, over `predicted_ms`. The first token is the prefill's product. It is the same figure as `/metrics`' time per output token, and with speculative decoding it counts every token a step accepts |
| `draft_n`, `draft_n_accepted` | speculative tokens proposed for this request and how many were accepted. Present when any were proposed, as llama.cpp does |

Milliseconds are wall-clock milliseconds, and the rates are tokens per second. A rate whose count
or time is zero is `null`. A request with several choices (`n`, or a batch of prompts) sums the
counts, and each time runs from the earliest choice's start to the latest choice's end.

When streaming, each choice's final chunk (the one with `finish_reason`) carries its own
`timings`, whether or not the request asked for `stream_options.include_usage`. The usage chunk,
when asked for, carries the request's. `/metrics` stays the aggregate surface (§6.3). It sums
the same quantities over every request, so it cannot say which request they belong to.

## 6. Operate it

### 6.1 The dashboard

Open `http://<host>:<port>/` in a browser. Three tabs:

**Dashboard** — the live strip across the top (decode step time, decode and prefill throughput,
requests in flight, KV cache in VRAM, draft acceptance), then cards for the batch, memory and
interconnect, and the expert plane.

The **KV cache** meter splits what it is holding into two segments:

```
KV cache   3.3 GiB / 4.5 GiB · 390.1k tok · 75.5%
▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
20.8k tok in flight · 369.3k held for the next turn
```

The full-strength segment is live requests; the ghost segment is **prefix cache** — finished turns
kept so the next turn need not re-prefill. Held is not a second pool: it gives way the moment a
live request needs it. A high held figure with a high hit rate is the cache doing its job, not a
leak. Both segments take their warning colour from the *total* fill, since what causes preemption
is the pool being full.

**Sessions** — one row per stored conversation: how many tokens, which tier it is in, and whether
its linear half is still reachable. Sizes **overlap and cannot be added up**: two sessions sharing
a system prompt share the blocks for it, which is the whole point of a prefix cache.

On a hybrid model, a session with blocks but no reachable snapshot reuses **zero** tokens — the
scheduler clamps reuse to the last linear checkpoint. That verdict is the headline of the tab.

**Model** — the resolved graph: every op, which library serves it, and what did not resolve.

### 6.2 The JSON behind it

```sh
curl -s localhost:8000/stats | jq          # counters, memory, throughput
curl -s localhost:8000/sessions | jq       # what is stored and where
curl -s localhost:8000/graph | jq          # the resolved kernel graph
curl -s localhost:8000/server_info | jq    # model, cards, KV, draft depth, wire
```

`/server_info` is the operator's first question asked of a process that is already running: which
model, on how many cards, with how much KV, drafting how deeply, over which wire. All of it is in
the startup log too — but a startup log is on a machine, in a file, from a process that may have
been restarted since.

Its `request_defaults` object says what a request that leaves a field out is served with: every
sampler field, `max_tokens`, `reasoning_effort`, `chat_template_kwargs` and `reasoning_format`,
under the request's own spelling of each. `server.default_max_tokens` is `null` for `auto`, and
`server.max_tokens_cap` is `null` when there is none.

### 6.3 Metrics

`GET /metrics` is Prometheus text. Two families:

**`vllm:` — the compatibility set**, so existing dashboards work: `num_requests_running`,
`num_requests_waiting`, `num_preemptions_total`, `gpu_cache_usage_perc`, `kv_cache_usage_perc`,
`gpu_prefix_cache_hit_rate`, `prompt_tokens_total`, `generation_tokens_total`,
`time_to_first_token_seconds`, `time_per_output_token_seconds`, `inter_token_latency_seconds`,
`request_prompt_tokens`, `request_generation_tokens`, `request_success_total`, …

**`radiance:` — what this engine actually has**:

| | |
|---|---|
| `engine_steps_total`, `_decode_total`, `_prefill_total`, `_mixed_total` | step counts by kind |
| `engine_step_seconds_total`, `_decode_total` | **the honest step time denominator** |
| `decode_tokens_total`, `prefill_tokens_total` | |
| `decode_tokens_per_second`, `prefill_tokens_per_second` | |
| `draft_tokens_total`, `draft_accepted_total`, `draft_acceptance_ratio` | |
| `draft_position_reached_total`, `draft_position_acceptance_ratio` | acceptance by position, since start |
| `decode_rows_unspeculated_total` | rows that never drafted at all |
| `attn_cached_tokens_total`, `linear_cached_tokens_total` | prefix reuse, both halves |
| `linear_checkpoints_written_total`, `linear_prefix_cache_hit_rate` | |
| `prefix_cache_evictions_total`, `prefix_cache_ckpt_evictions_total` | |
| `num_requests_preempted` | |
| `http_queue_depth`, `http_responses_total`, `http_admission_rejected_total` | |
| `client_disconnect_total`, `stream_overflow_total` | |

Two traps worth naming:

- **`draft_acceptance_ratio` is conditional on having drafted.** It reads healthy while
  tokens-per-step collapses. The quantity that separates the two is blocks per step —
  `draft_tokens_total / depth` over `engine_steps_decode_total`. `scripts/mcdecl.sh` measures it.
- **Step time is not wall-clock over steps.** Use `engine_step_seconds_decode_total /
  engine_steps_decode_total`, which charges only decode. Wall-clock charges idle and prefill to it.

`deploy/grafana/radiance.json` charts both families over time, on Prometheus or VictoriaMetrics.

### 6.4 The live terminal view

```sh
radiance --model m.rad --live
```

Cards, throughput, draft acceptance and the expert plane, redrawn in place. **Redirected it prints
one greppable `key=value` line instead**, so it is usable from a script:

```sh
radiance --model m.rad --live >> serve.log 2>&1
```

The step counter is the field that tells a stalled loop from an idle one: throughput decays toward
zero on its own, but a step count frozen between two draws means the loop itself stopped.

### 6.5 Logging

```sh
RADIANCE_LOG=debug radiance --model m.rad     # error|warn|info|debug|trace
radiance --model m.rad -v                     # same as debug
radiance --model m.rad -vv                    # same as trace
```

### 6.6 Security

**There is no TLS, and without `--api-key` there is no authentication.** CORS is on, so a page on
any origin can call the API from a browser; `--no-cors` turns it off.

```sh
radiance --model m.rad --api-key "$KEY"
```

With a key, every request must carry `Authorization: Bearer KEY` or it is answered 401 —
`/tokenize` and `/server_info` included, since they hand out the vocabulary, the chat template and
the configuration. Three paths are exempt: `/health` and `/ping`, because a load balancer cannot
carry a bearer token and an authenticated probe is a server that reports itself down, and the
dashboard page itself (`/`, `/dashboard`), which holds no data — on a 401 it shows a key field,
keeps the key in the browser's storage and sends it on every fetch of data it makes.

The key is checked, not encrypted: on any network you do not control, bind to localhost and put a
TLS-terminating reverse proxy in front:

```sh
radiance --model m.rad --host 127.0.0.1 --port 8000 --api-key "$KEY"
```

Also note `--prefix-cache-disk-mib` writes **conversation content unencrypted** under `--prefix-cache-dir`.

### 6.7 Restarting and stopping

Nothing here is a daemon; run it under systemd, a supervisor or `nohup`. `scripts/fnstop.sh` stops
a running server and waits for the process to actually go — which matters, because VRAM is not
released until it does.

---

## 7. When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `N declared op(s) resolve to libref` and the engine exits | No kernel library covers those ops on this card. Serving from the reference implementation would be orders of magnitude slow with nothing to say so | Install a library that covers them, or accept the cost with `--debug-accept-reference-kernels`. The named ops are exactly what a new library must implement |
| `did not fit` at load, naming a weight class or a KV group | The split left one side too little. A routed expert cannot be served from the file tier, so it has nowhere else to go | If **weights** did not fit: add or raise `--host-pool-mib`, or lower `--max-model-len` or `--max-num-seqs`. If a **KV group** did not fit: `--kv-cache-dtype fp8`, or a larger `--host-pool-mib` (RAM permitting), which takes experts off the card. The plan report names which, by class, with the shortfall |
| `chat is 501` in the startup line | The container carries no chat template | Re-convert with `--tokenizer`, and check that a `chat_template.jinja` sits beside `tokenizer_config.json` in the checkpoint. Meanwhile use `/v1/completions` |
| `does not read '<weight>' encoded <encoding>` at startup | No loaded kernel takes that weight's encoding for the op that reads it | Convert with a recipe that writes an encoding a loaded kernel reads, or load the library that reads it |
| Prefix hit rate 0% and every turn re-prefills | The KV pool cannot hold the live sessions, so the cache evicts what the next turn wanted | The cache already has everything the experts can do without: a larger `--host-pool-mib` (RAM permitting), `--kv-cache-dtype fp8`, or fewer/shorter sessions. See §4.3 |
| The server dies with no log line, on a request it had been serving fine | A kernel whose code object had never been loaded could not be: the card has no room left. HIP faults inside the launch rather than returning an error, so there is nothing to print | A card that never met `--gpu-headroom-mib` — see §4.3. Raise the headroom past the shortfall startup names. `coredumpctl info` names the kernel; `AMD_LOG_LEVEL=1` prints `HSA_STATUS_ERROR_OUT_OF_RESOURCES` just above the fault |
| KV in VRAM stays high with one session and nothing moves | The idle tiers are **off** — they are sized, and with no size nothing is copied | Pass `--prefix-cache-host-mib N` |
| `out of memory` / `the carve reserved N of address space and the card would not back it` | Something else is holding VRAM | Check with `rocm-smi`; stop the other process. The budget is stated before allocation, so the log says what it asked for |
| `pool allocation refused` with `backing allocation of … refused` on the second card, after the first card's pools succeeded | The VRAM budget is measured before any rank allocates. A card that refuses its weights pool lost that memory to another process in between (the line below the refusal says how much is free now). A refused `host_pool` is pinned memory: **every rank pins its own `--host-pool-mib`**, so `--tp 2` pins it twice, and the amdgpu driver caps pinned memory system-wide at `ttm.pages_limit`, half of RAM by default | Stop the other process (`rocm-smi --showpids`). For the host pool, keep `--tp` × `--host-pool-mib` under the cap the refusal names, or raise it with the kernel parameter `ttm.pages_limit` |
| `host_pool: backing allocation of … refused: rad_dev_alloc(…): locking piece K of N`, and `failed to pin userptr` in `journalctl -k` | The IOMMU would not DMA-map one piece of the pinned host pool. Pinned host memory is locked in pieces of at most 1 GiB because an Intel VT-d host in scalable mode was seen to refuse any single pin above about 2.5 GiB; a piece failing means this machine refuses smaller pins still, or a total its IOMMU will not map | Keep IOMMU translation on: `iommu=pt` lets every pin through but on that host also removed peer access between the cards, and every multi-rank kernel requires it. Report the kernel version, the IOMMU mode and the refusal line |
| `--draft-model is not implemented` | This engine loads its drafter from the container | `rad-convert --draft-model DIR` at conversion time |
| `context_length_exceeded` on a request | The **prompt alone** does not fit the context. `max_tokens` is clamped rather than refused, so this fires only when there is no room left to generate at all | Shorten the prompt by the number of tokens the message names, or raise `--max-model-len` if the model allows it |
| `unknown option '--foo'` and the engine exits | Deliberate. A typo in a budget flag is the difference between the plan you asked for and one the planner invented | Check `radiance --help` |
| Empty `content` and `finish_reason: "length"` on short chat turns | The reasoning spent all of `max_tokens` inside `<think>`; `usage.completion_tokens_details.reasoning_tokens` equals `completion_tokens`. Qwen3.8's default `reasoning_effort` is `xhigh` | `--reasoning-effort medium`, or `"reasoning_effort": "none"` per request; or cap the reasoning with `thinking_token_budget` / `--thinking-token-budget` below `max_tokens` |
| `401` with `invalid api key` | The server was started with `--api-key` | Send `Authorization: Bearer KEY` (§6.6) |

First three things to run when a server will not start:

```sh
rad-info m.rad                            # is the container what you think it is?
radiance --model m.rad --debug-graph      # does every op resolve?
rad-kbench --home $RADIANCE_HOME --max-cases 1   # are the kernels correct at all?
```

---

## 8. Environment

**Operator-facing**

| | |
|---|---|
| `RADIANCE_HOME` | where `kernels/`, `architectures/` and `quantizers/` live; a `:`-separated search path, the first home winning. Same as `--radiance-home` |
| `RADIANCE_KERNELS` | default kernel hierarchy, comma-separated. Same as `--kernels`, read by the engine *and* every tool, so a plugin validated with `rad-kbench` can be served without editing launchers |
| `RADIANCE_LOG` | `error` \| `warn` \| `info` \| `debug` \| `trace` |
| `ROCM_PATH` | where ROCm lives, if not `/opt/rocm` |
| `HF_TOKEN` | for `scripts/hfget.sh` on a gated repo |

**Tooling**

| | |
|---|---|
| `RADIANCE_TUNE` | pin a kernel's tunable axes instead of reading the tune cache — `op:key=value[,key=value][;op:...]`. For comparing tunings, not for deployment |
| `RADIANCE_CALIB_DIR` | where an engine run writes calibration data for a recipe's `gptq calib=DIR` |
| `RADIANCE_CALIB_TOKENS` | with `RADIANCE_CALIB_DIR`, the tokens the calibration tap accumulates before it writes and switches itself off (default 65536) |
| `RADIANCE_PROFILE_EVERY` | with `--profile-ops`, dump the per-op table every N steps (default 32) |
| `RADIANCE_LOG_STEPS` | log one line per step: its phase, sequences, and each one's token count and context (§4.5) |
| `RADIANCE_DISPATCH_TRACE` | `N` (default unset: off): record per-dispatch start/end timestamps of up to N kernel dispatches and write them at exit to `RADIANCE_DISPATCH_TRACE_OUT` (default `/tmp/radiance-dispatch.csv`, rocprofv3 `kernel_trace` column names). Unlike `--profile-ops` it adds no barrier and no host wait between kernels, so the gaps between dependent kernels stay visible (a decode step runs about 830 kernels: N = 20000 covers ~24 steps); the trace itself costs ~2 ms a step. Use it where rocprofv3 cannot run on the AQL backend. Not for production |

**Performance switches** (opt-in; off or neutral unless set)

| | |
|---|---|
| `RADIANCE_TAPES` | the number of recorded decode/prefill passes ("tapes") the step keeps for replay. Default 24, range 1..4096. A pass is keyed by token count, longest query and speculative width, so many concurrent sequences with varying draft lengths can push the cache over its capacity and every miss costs a live issue. `/stats` `passes` shows `played`, `recorded`, `audited`, `dispatched`, `unordered`, `evicted` and `tape_cap`: `recorded` and `evicted` climbing means thrash, `played` climbing steady replay. **Recommended:** keep the default at one or two sequences; raise it (64 is a good start) at 8 sequences. Read once at start |
| `RADIANCE_LOOKUP_DRAFT` | `0` (default: off) / `1`: whole-context lookup draft for a **block drafter** (DFlash2). The drafter's sliding window (2048) cannot see a source further back, so a far copy, an edit or a re-emitted block drafts at ~3.3 tokens a step instead of ~7. With this on, a host-side suffix index over prompt + output finds the longest earlier occurrence (up to `_MAX_MATCH` tokens, most recent on a tie) of the sequence's current suffix whose continuation lies further back than the drafter window, and when the policy takes the step it replaces the drafter's proposal by the 7 tokens that followed it. The verify is unchanged, so output is lossless. Measured on gfx1201 (Qwen3.8-27B + DFlash2, 210 W): far copy +31..38 % TG at 8k/32k context, edits +7..11 % at 32k (-3..4 % at 8k), free generation, summaries and short prompts unchanged. Recommended for long-context copy/edit workloads (code edits, document rewrites); leave off for pure chat. Policy knobs: `RADIANCE_LOOKUP_ENTER` (match length that starts lookup mode, 8), `_STAY` (match length that keeps it going, 3; must be <= ENTER), `_HOT` (accepted draft tokens of the last lookup step that keep it going, 3), `_MAX_MATCH` (match length cap, 24, at most 64), `_MIN_DIST` (`auto` = the drafter window, or a token count; 0 = any distance), `_CHAIN` (candidates one search may visit, 1024). A malformed value keeps the default; an inconsistent set (not 1 <= STAY <= ENTER <= MAX_MATCH <= 64) turns the feature off with a warning. Metrics `radiance:lookup_draft_{taken,steps,tokens,accepted}_total`. Read once at start |
| `RADIANCE_DRAFT_REFILL` | tokens, default the drafter window (0 = off): on a prefix-cache hit with a **block drafter** (DFlash2), recompute the last N cached tokens so the drafter has context at once. The drafter's own windowed KV group is not prefix-cached, so a hit leaves it without K/V for the cached tokens and it declines the sequence until its window has slid past the hole: ~1 token per step in multi-turn chat. With this on, a hit is ended N tokens before the end of the prompt (down to a block and, on a hybrid model, to a linear-state checkpoint), so the trunk recomputes those tokens and the drafter's context pass covers its window. N is clamped to the window. A hit that would have to give back more than 2 N extra tokens (no checkpoint near enough; see `--checkpoint-slots`) is kept whole and the drafter waits as before. Measured (Qwen3.8-27B + DFlash2, turn 2 of a chat at ~11k context): 1.29 to ~4.0 tokens per step, 6.0 to 3.4 s for 256 tokens; cold turns unchanged. Lossless. Metrics `radiance:draft_refills_total`, `radiance:draft_refill_tokens_total`. Read once at start |
| `RADIANCE_GDN_STATE` | `f32` (default) / `f16`: the width of the delta net's recurrent state on the Qwen3.5 family. `f16` halves the largest item of the KV plan (144 -> 72 MiB a slot, two slots a sequence, and each checkpoint with it), which goes to the KV pool (about +37k to +55k tokens at 8 sequences), and changes the generated text slightly (round-trip error ~5e-4 relative). **Recommended: `f32` unless context capacity is the limit**; judge `f16` on KL and acceptance for your model before using it. Read at declare time |
| `RADIANCE_ADAPTIVE_K` | `0` (default: off) / `1`: per-sequence verify length for a **block drafter** (DFlash2). The step verifies only a prefix of each sequence's draft, chosen from a small length set by expected tokens per step time, so sequences the drafter predicts badly stop costing trunk rows. Output tokens are still the target's own choice; a single stream (below 32 total rows) is untouched and bit-identical. **Recommended: `1` when serving 4 or more concurrent sequences** (+5.7 % steady, +9.8 % with staggered arrivals at 8 sequences on gfx1201), together with `RADIANCE_TAPES=64`; no effect, and no need, for one stream. Optional knobs: `RADIANCE_ADAPTIVE_K_LENS` (candidate lengths, default `1,3,5,7`; the drafter depth is always added), `_ALPHA` (weight of the newest verify in a sequence's acceptance estimate, 0.25), `_PRIOR` (pseudo-positions pulling it toward the batch average, 2), `_COST_A` / `_COST_B` (step-time model `a + b*M` in ms, defaults 30.57 and 0.4286 from one gfx1201 card), `_COST_TABLE` (`M:ms,M:ms,..` measured on your card, ascending M, at least two points; overrides A/B), `_MIN_ROWS` (total rows below which nothing is shrunk, 32), `_QUANT` (total rows restricted to multiples of this to bound the distinct step shapes, 8; 0 = free). A malformed value keeps the default and warns. Read once at start |
| `R4D_LM4_LEAN` / `R4D_LM4_NPW` | libr4d launch pins for the int4 / u4 lm_head rows, read per launch. `R4D_LM4_LEAN`: unset = auto (default: 1 at M > 16, 0 at M <= 16) / `0` / `1` / `2`; level 1 keeps the scale-group loop rolled and removes the register spill the unrolled u4 g64 head has at 2 or more row tiles (M 17 and up). `R4D_LM4_NPW`: `1` (default) / `2` (two weight tiles a wave share each activation fragment; an experiment, no measured gain). `R4D_LM4_FKB`: `1`/`2`/`4` K staging round (default 4 at M <= 32, else 2). All bit-identical to each other. **Recommended: leave unset** (the auto LEAN is the measured optimum: u4 g64 lm_head at M=32 5971 to 2177 us, M=48 4659 to 2928 us at 210 W; M <= 16 stays on the unrolled form, the rolled one costs ~10 % there). Reading libr4d README, `lm_head launch pins` |

**Performance switches** (opt-in; off or neutral unless set)

| | |
|---|---|
| `RADIANCE_MXFP4_ATILED_MIN_M` | `0` (default: off) / a row count T. From T rows on, the MXFP4 trunk's block norm and SwiGLU product write their E4M3 activation plane fragment-tiled (`afrag` op parameter, OPS.md) and the MXFP4 GEMMs read it straight into the WMMA registers (libr4d `gemm_mxfp4a8_atiled`, no LDS staging of A). Dense MXFP4 models on one rank only; T must be above 64 (keep it above 256: the tuned wide decode form and a fragment-tiled plane must not meet; a `dec=1` tune line for a band whose upper bound is at or above T is refused by the tiled row's `tune_valid`, and that band runs the tiled kernel) and not above `--max-num-batched-tokens`, otherwise the value is refused with a warning. **Recommended: 513** for prefill-heavy serving (+5.6 % prefill at 2k, +6.2 % at 8k, +6.6 % at 32k on gfx1201, same-window A/B). `afrag` is part of the op tune key, so tune rows must be produced with the same value. Read at declare time; a recorded pass is bound to it |

There is also a family of instruments for engine development — `RADIANCE_DEBUG_*` (residual probes,
idle scrubs, routing and issue dumps), `RADIANCE_OP_ORACLE*` (named ops checked against `libref` on
the operands the model issued; `scripts/oracle.sh` drives it) and `RADIANCE_OP_REPEAT` (named ops
run a second time on the same inputs, to ask whether a kernel is a function of them). They are
instruments, not features; grep `getenv` if you need one, and read the comment beside it first.

---

## 9. Further reading

| | |
|---|---|
| `spec.md` | the design, and it is binding. Start here for *why* |
| [TOOLS.md](TOOLS.md) | every flag of every tool, and the shell harnesses |
| [PLUGIN.md](PLUGIN.md) | writing a kernel plugin: exports, constraints, hooks, out-of-tree build |
| [OPS.md](OPS.md) | the conventional op vocabulary. Held to the registries by `ctest -R ops_doc` |
| [ARCHITECTURES.md](ARCHITECTURES.md) | per-architecture geometry, and the traps that stay plausible when they are wrong |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | layout, build and conventions for working on the engine |
| [KERNELS.md](KERNELS.md) | what `rad-kbench` last measured |
| [QSA.md](QSA.md), [MOE-W4.md](MOE-W4.md) | the sparse-attention indexer; the four-bit expert format |
