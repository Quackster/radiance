# Radiance — implementation brief

Read `spec.md` first; it is the design and it is binding. This file is the mechanical part: who
owns what, how it builds, and the conventions that keep ten people's code looking like one
person's.

## The host backend is the primary target, not a fallback

Everything except the device kernels builds, tests and runs through the **host device backend**:
`cmake` with no ROCm must configure, build, test and run, and `RAD_HIP_FOUND` gates only the HIP
backend and the device kernels. A machine with no card is a complete development environment for
the core, the tools, the architecture plugins and `libref`, and the suite runs in full there.

## Layout and ownership

```
abi/              THE ABI, and all a plugin compiles against. Frozen — do not edit quietly
core/rad_internal.h   shared internals: logging, Geometry, constraint matching
core/util/        logging, error strings, dtype arithmetic, tensor helpers: the bedrock
core/device/      device backends: host, and aql (HSA queues the engine writes; HIP for memory)
core/plugin/      plugin loader, kernel registry, op schemas, selector, bucket tables
core/build/       the declare phase: builder, weight manifest, buffer plan, graph dump
core/runtime/     the run phase: RadCtx, rad_issue, the arena
core/mem/         pools, slab, KV block manager, prefix cache
core/sched/       scheduler, RadBatch construction, request lifecycle
core/place/       placement planner, the mover, the heat engine
core/sample/      device sampler dispatch + the host reference chain + grammar bitmask
core/text/        tokeniser, chat template, tool parsing, detokenisation
core/mm/          image and video decoding, and the processor that turns frames into patches
core/format/      the .rad reader and writer, safetensors, recipes, the imatrix, the tune cache
core/server/      HTTP: the OpenAI, Anthropic and llama-server endpoints, /metrics, the dashboard
core/kld.*        the KL mode: --kld-record / --kld-ref
core/engine*      the thing that ties them together, and `radiance` the binary
libref/           every op, naively, correct, host domain: the fallback and the oracle
libr4d/           the RDNA4 (gfx12xx) kernel library: kernels, rows, layouts, fusion claims
libavx/           the same vocabulary in x86 SIMD, host domain: the production host kernels,
                  built on every x86-64 build because an op placed on the host cannot run on libref
libquant/         the built-in quantisers rad-convert's recipes name: grids, transforms, rtn, gptq
arch/<id>/        one architecture plugin per directory
tools/            rad-info, rad-kbench, rad-convert, rad-tune, rad-schemas, rad-tokcheck,
                  rad-chatparse, rad-gbnf-check
cmake/            rad_add_plugin, and the package config an out-of-tree plugin find_package()s
scripts/          shell harnesses for serving, tracing and profiling a running engine
data/             kernels.rkb and kernels_moe.rkb: recorded references rad-kbench replays with
                  no model. Two, because no one container declares the whole vocabulary.
                  recipes/: the quantisation recipes of the containers this tree serves
docs/KERNELS.md   what rad-kbench last measured, generated and checked in
vendor/           llama.cpp lifts, tracked to an upstream commit, patched not edited
tests/            ctest
```

Each directory has its own `CMakeLists.txt` and each owner writes only inside their own. The
top-level file already exists; add `add_subdirectory` lines there only if your directory is new.

## Build

```sh
cmake -S . -B build -G Ninja           # host backend; no ROCm required
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Your directory must compile and its tests must pass **before you report done**. A component that
does not build is not a component.

## Conventions

**No Python in the engine, its tools, its build or its test suite.** This is the project's
founding constraint — `spec.md` opens with it. Tests are C++ and CTest. The one Python file under
`tests/`, `mkgolden.py`, is run by hand against transformers to regenerate the reference numbers in
`tests/mm_golden.h`; nothing in the build or the suite calls it.

**C ABI at plugin boundaries, C++20 inside.** `abi/` is C and must stay compilable as C.
Everything under `core/` is C++20 and may use the standard library freely.

**Errors are values.** Return `RAD_OK` or a negative `RAD_E_*`. Do not throw across a function
that a plugin or the run phase can call. `fatal()` is for broken invariants only.

**Nothing unimplemented returns success.** A function you have not written yet returns
`RAD_E_UNSUPPORTED` and logs once at Warn with its own name. A stub that returns 0 is how a
project ends with fluent wrong output and no way to bisect it (`spec.md` §17).

**No allocation and no synchronisation on the step path.** The arena is sized at declare. If you
find yourself wanting a `malloc` inside `rad_arch_step` or inside the scheduler's per-step work,
that is a declare-phase bug.

**`rad_memset_async` refuses `RAD_MEM_HOST` on a card, and the host backend cannot reproduce that.**
`RAD_MEM_HOST` is pageable memory the GPU has never been shown, so the device backend returns
`RAD_E_UNSUPPORTED` naming the rule (`core/device/aql.cpp`). On the host backend every kind of
memory is the same memory, so a memset added to a path that runs on host memory passes in a no-ROCm
build and fails on a card. Use `RAD_MEM_HOST_PINNED` or `RAD_MEM_HOST_MAPPED`, or memset on the
host. `rad_memcpy_async` accepts pageable memory but bounces it through pinned memory and WAITS,
so a pageable copy on the step path is a host stall; `aql_pageable_bytes()` counts them.

**Kernel plugins launch with `hipLaunchKernel` and the engine dispatches them.** On a card every
stream is an HSA queue the engine owns and writes AQL packets into itself (`core/device/aql.cpp`);
the executable defines `hipLaunchKernel` and HIP's kernel registration hooks, so a plugin built by
hipcc needs nothing special. What a plugin must NOT do with an engine stream is call any other HIP
stream API: `hipEventRecord` and `hipStreamWaitEvent` are refused by name, and anything else would
be handed to HIP as a stream HIP does not own. Use `rad_event_*` and `rad_memcpy_async`.

**Comments explain the decision, not the syntax.** The spec's voice: say why this and not the
obvious alternative, and name the cost when there is one. `libr4d/r4d_rows.cpp` is the model.
Do not write `// increment i`.

**Naming.** `rad_` prefix on everything with external linkage. Files are named for what they
provide. `snake_case` for functions and variables, `PascalCase` for types.
