# Writing a plugin

Three kinds of plugin extend the engine, each a shared object that exports a handful of C symbols
and links **nothing** of radiance's:

- a **kernel library** implements ops for some hardware — a card, a family of cards, the CPU;
- an **architecture** declares a model: its weights, its buffers, the ops it issues and in what
  order;
- a **quantiser** turns a checkpoint's weights into the planes of an encoding, for `rad-convert`.

The engine `dlopen`s them, asks what they can do, and calls them. That boundary is the whole design:
a plugin is compiled against the installed `abi/` headers and nothing else, so a third party can
ship one without building the engine. **The plugins radiance ships go through exactly this door.**
libr4d, libref, libavx, libquant and every architecture are built with the same `rad_add_plugin`, can
each be built on their own against an installed radiance, and are found and ranked by what they
declare, never by their names; `ctest -L oot` rebuilds all of them out of tree against an installed
prefix and requires the same result (`tests/oot_plugins.sh`).

Sections 1–9 are the kernel side; §10 is what a library for a card radiance has never seen needs;
§11 and §12 are architectures and quantisers; §13 is building, installing and the search path all
three share.

---

## 1. The shape of it

```c
#include "rad_abi.h"          /* required */
#include "rad_plugin.h"       /* optional helpers -- see §6 */

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo i = {
        RAD_PLUGIN_KERNEL, "mykernels", "0.1", "a GEMM or two", "gfx1100",
    };
    return &i;
}

int                rad_kernel_schema_count(void);   /* the ops you DEFINE, if any */
const RadOpSchema* rad_kernel_schema_at(int i);
int                rad_kernel_count(void);          /* the kernels you IMPLEMENT */
const RadKernelInfo* rad_kernel_at(int i);
```

`rad_plugin_abi_version` is checked at load and a mismatch is refused **by name, with both numbers
printed**. Do not hardcode the integer — return the macro, and rebuild when it moves. It moves for
any shape change in a header a plugin compiles against (`rad_abi.h` says which).

`build_target` is prose for a human: the targets you were compiled for (`"gfx1100"`,
`"gfx11-generic"`, several separated by spaces or commas), `"host"` for CPU kernels — which is what
both in-tree host libraries say — or `""` for a plugin that really is portable. It is printed, not
trusted: what the engine checks is the code objects your library actually carries (§10).

Four more exports are optional, each looked up by exactly this name:

| symbol | says |
|---|---|
| `int rad_plugin_open(void)` | called once after load. `RAD_E_UNSUPPORTED` **declines this machine** — the library is unloaded and the rest of the installation carries on; any other negative status refuses it as an error |
| `void rad_plugin_close(void)` | called once at shutdown |
| `int rad_kernel_concurrent(int i)` | nonzero: kernel `i` touches nothing but its operands and scratch, so independent launches may overlap |
| `int rad_plugin_reference(void)` | nonzero: this library is a **reference implementation** — ranked below every unnamed library, refused for serving unless `--debug-accept-reference-kernels`, and what `rad-kbench` checks the others against. libref says so through this, as any library would |

---

## 2. A kernel says what it can serve, not what it is

```c
static const RadConstraint c_gemm[] = {
    RAD_CLE("M", 64),              /* M <= 64            */
    RAD_CDIV("K", 128),            /* K divisible by 128 */
    RAD_CIN("dtype", "fp8a8 w4a8") /* dtype in that set  */
};

static const RadKernelInfo k[] = {{
    .name = "mygemm_m64", .op = "gemm_nt_q", .family = "gemm",
    .computes = "row-major A times column-major B, fp8 in, bf16 out",
    .shape    = "M<=64, K%128==0",
    .dtypes   = "fp8a8, w4a8",
    .domain   = RAD_DOMAIN_DEVICE,
    .priority = 50,
    .constraints = c_gemm, .n_constraints = 3,
    .launch = my_launch,
}};
```

The five operators are `RAD_CEQ`, `RAD_CLE`, `RAD_CGE`, `RAD_CDIV`, `RAD_CIN`. **The engine picks
bucket boundaries out of your `LE`/`GE`/`EQ` values** and resolves once per bucket, because a GEMM
sees one row at decode and thousands in a prefill chunk and they do not want the same kernel
(`spec.md` §2.2). `DIV` and `IN` cut nothing — a divisibility holds at every value in a range, and
set membership is not an interval — so neither can be a band edge.

Two consequences worth internalising:

- **Declare the band you actually serve.** A kernel that claims `M <= 2048` and is correct only to
  64 is not slow, it is wrong, and the engine believes you.
- **A band you claim and cannot serve is invisible to a single-sequence bench.** It only shows at
  concurrency, where M is larger. Run the whole ladder, not just C=1.

**A constraint can be on the machine instead of the geometry.** A key that starts with `device.`
is matched against what the engine measured about the ranks' cards at launch, not against the
op's parameters. `RAD_CEQ(RAD_FACT_P2P, 1)` (`device.p2p`) is how a kernel says it reads or writes
another card's memory. On a machine where some pair of the ranks' cards cannot reach each other,
such a kernel is not selected. The next row that does not need it takes the op. With no such row,
startup stops and the selection table names the fact (`device.p2p == 1, have 0 -- this machine,
measured at launch`). Every collective in libr4d carries it, and each has a `_host` twin that
requires `device.p2p == 0` and exchanges through pinned host memory, so exactly one of the pair
matches any machine. A machine fact is in no geometry, so
it changes no tuning key. A tool that selects without an engine has measured nothing, and such a
kernel is not selected there either.

`priority` breaks ties **within your plugin**, highest first, a tie going to declaration order.
Between plugins there is no contest: **the hierarchy is absolute.** The first plugin in `--kernels`
order with any matching kernel supplies it, and a later plugin never outbids it however
specialised its kernel — which is exactly what makes a user override an override rather than a
competitor.

---

## 3. Operand extents are a request, not a bound

`RadArgs.t` is positional, in the op schema's order. Each `RadTensor` carries the extent the
**caller narrowed it to**, which is generally smaller than the buffer.

> An operand handed over at its declared extent means a single-row decode computed over all 8192
> rows of an arena buffer — reading uninitialised memory and writing over the next op's input.

Size your loops and your grid from the operands you were given, never from a declared maximum.
Where an op takes `cu_seqlens`, derive the sequence count from **its own length** (`n = len - 1`)
rather than from a sibling array, and derive each sequence's rows from `cu[n]` and `cu[n+1]`, which
are absolute offsets into the token operands. Doing it that way is also what lets the engine hand
you a *sub-range* of the step — see §8.

An **absent optional operand is a null `RadTensor.data`, not a short operand count.** Read the
operand after it and every weight in the model shifts by one. `rad_arg_in()` in `rad_plugin.h` is the
one correct fetch.

---

## 4. The optional hooks, and what each one buys

None are required. Each one is a claim, and the engine will use it.

| hook | says |
|---|---|
| `init` / `fini` | per-resolved-instance state; `RadArgs.instance` hands it back |
| `scratch` | how many bytes you need; the engine provides at least that |
| `layout` | this weight is held in my order, not as the container stores it |
| `relayout` | how to produce that order from the container's planes, run by the loader at load |
| `unrelayout` | the inverse, so a tool holding the stored bytes can read the planes back |
| `opd_shape` | how to size and fill every operand, so a tool can call you cold |
| `replaces` | the chain of conventional ops I compute in one launch |
| `tunables` / `tune_valid` | the axes of my instantiation space, and which combinations are legal at a geometry |
| `describe` | this dispatch, described rather than performed, so a repeated sequence can be replayed as a graph |

`opd_shape` and `unrelayout` are what make your kernel **checkable**. Without them, tools that would
otherwise test you against the reference skip you *by name* rather than guessing — which is the
right outcome, and a quiet one.

`replaces` is a strong claim and the ABI demands exactness: it means "I stand in for this chain,
and you may check me against it." **A kernel that is merely close to the sequence it names should
publish no hook at all.** In exchange, the engine can recognise that your fused kernel and the
spelled-out chain are alternatives — which is how the Model view draws them as one branch instead
of four sequential nodes.

`tunables` is what makes your kernel tunable at all; without it, it is one fixed implementation
forever. **You declare the axes and their legal values, not a list of pre-named variants**
(`RadTunable` in `rad_abi.h`): the core takes the cross product, so a ladder cannot end up with a
hole in it that quietly runs the next tile up. `tune_valid` rejects the combinations a product of
independent lists cannot express — a block bound that couples two axes, a `K % (sk * unit)` rule —
and a rejected candidate is reported as skipped, not as a failure. `rad-tune` measures the
survivors at the shapes this model actually issues and writes the cache the selector consults at
declare. The chosen point then arrives as **ordinary parameters**: you read `sk` with the same
`rad_args_geti()` you read `M` with, so there is no variant index and no picker in your launch
adapter. `deflt` is what runs where this machine has no measurement for an instantiation, so it has
to be launchable — the loader checks it appears in `values` and refuses your plugin by name if it
does not.

`describe` buys nothing but speed: null means every issue of your kernel is submitted
individually, which is correct and merely slower, and non-null lets a repeatedly issued sequence
be replayed as a graph instead. It must not allocate, must not synchronise and must not touch the
device: it is called while a run is being RECORDED -- not on every step, and not instead of the
launch, which happens anyway on that pass. It takes the stream, because a launch whose arguments
depend on which stream it goes onto would otherwise be described as something it is not. An op
whose work is a memory copy rather than a launch has nothing to put in a `RadLaunchDesc` and should
leave `describe` null.

---

## 5. What a launch may do

On a card the `RadStream` a kernel is handed is a queue the engine writes packets into itself
(`core/device/aql.cpp`), not a `hipStream_t`. The engine binary defines the HIP entry points a
kernel library uses and answers them on its own queues:

- `hipLaunchKernel`, and therefore `<<<>>>` and `hipLaunchKernelGGL`;
- `hipMemcpyAsync`, `hipMemcpy2DAsync`, `hipMemsetAsync`;
- `hipGetLastError` / `hipPeekAtLastError`, with HIP's own semantics.

**Nothing else in HIP takes an engine stream.** `hipModuleLaunchKernel`, `hipExtLaunchKernel`,
cooperative launches, `hipLaunchHostFunc`, and any library that launches through them — rocBLAS,
hipBLASLt, MIOpen, a hipRTC module — would hand HIP a queue it has never heard of. Header-only
device code (Composable Kernel's device templates, rocPRIM's block primitives) compiled into your
own `.hip` launches through `hipLaunchKernel` and is fine. `hipEventRecord` and `hipStreamWaitEvent`
on an engine stream are refused by name; ordering between streams is the engine's.

The code object is read by the engine too, which puts limits on what a kernel may be:

- **no hostcall**: `printf` and `assert` in device code need a buffer the queues do not provide, and
  a kernel that takes one is refused at its first launch, by name. `rad_add_plugin` compiles device
  code with `-DNDEBUG` outside Debug for this reason;
- **no dynamic stack** — recursion, a variable-length array — since nothing sizes its scratch;
- **kernel arguments up to 768 bytes**, and static plus dynamic LDS up to 64 KiB a dispatch;
- **an uncompressed fat binary**. `rad_add_plugin` passes `--no-offload-compress`; a toolchain that
  compresses anyway is refused at the first launch.

A device allocation a kernel needs for itself comes from the `scratch` hook (§4), not `hipMalloc`:
the engine's allocator is the one that knows what the card has left.

---

## 6. `rad_plugin.h` is optional, and one part of it is not merely convenience

A plugin that includes only `rad_abi.h` is complete and correct. The helper header is shared
definitions of the dozen things every plugin needs — optional-operand fetch, parameter-with-default,
contiguity, row pitch, narrowing conversions.

Use its conversions. **The narrowing conversions define what a result IS**: the reference contract
is f32 accumulation, IEEE round-to-nearest-even, no FMA and no reassociation, and a plugin whose
bf16 rounding truncates instead of rounding to even does not reproduce the reference and therefore
cannot be checked against it.

---

## 7. Defining a new op

Only if the conventional vocabulary in `docs/OPS.md` genuinely has no name for it. That document
is the op vocabulary and the rules that come with adding to it: what a schema must declare, that
the first plugin in hierarchy order to declare an op **fixes its schema** and a later plugin
declaring it differently is refused at load with both plugins named, and why the refusal is worth
having. It is also mechanically checked against the registries by `ctest -R ops_doc`, which is the
reason the statement lives there and not in a second copy here that can drift away from it.

An op that stays inside `docs/OPS.md` is describable by the reference, and therefore checkable,
even when your kernel cannot describe itself.

---

## 8. The one contract an architecture plugin gets silently wrong

If you write an **architecture** plugin with a linear/recurrent block, read `spec.md` §10 before
you write its `step()`.

A step that carries a prefill chunk beside decode rows is what continuous batching produces
constantly — on an agentic load about a third of all steps, and *none of them prefill-only*,
because every chunk rides with decode rows. The obvious block issues one path for the whole step.
Its decode rows then land on the chunked path, which commits the recurrent state directly and takes
no `num_accepted`: a draft verified there cannot be rolled back, the rejected tokens stay folded
into the state, and the model is conditioned on text it never emitted. **The answer stays fluent
and drifts**, which is why it shows up as concurrent requests disagreeing with the same requests
run alone rather than as anything failing.

Issue the recurrent path over `RadBatch::n_seq_decode` sequences and the chunked path over the
rest. The core orders every step decode-first so each half is a contiguous run. It costs no kernel
change, for the reason in §3: `cu_seqlens` holds absolute offsets, so the prefill half is addressed
by passing `cu + n_seq_decode` with the token operands left at their base.

---

## 9. Checking your work

`rad-kbench` is the answer to "is my library correct, complete and fast", and it needs no model:

```sh
rad-kbench --kernels mylib,libr4d,libref --report mylib.md --bench
```

It replays a recorded fixture — `kernels.rkb` unless `--fixture` names another, and `data/` holds
a dense one and a routed-MoE one — a recording of libref's answers over a few hundred real model
geometries, and runs every kernel **your** library supplies against it. Your rows are found the
same way the engine finds them — your constraints, your priorities — but they are not *selected*:
every library that has a row for a case is run, so yours appears beside libr4d's rather than
instead of it. The report carries the three columns that matter separately:

- **correctness**, relative to the recorded reference. The recording keeps the full norm of every
  output plus 1024 sampled values, so a kernel that gets the sampled positions right and the total
  energy wrong still fails.
- **completeness**, the op-by-library matrix, read off the registries rather than off the run —
  so an op your library does not implement is a gap whether or not this fixture exercises it.
- **speed**, timed on the same bytes in the same pass.

Three things it does for you that are easy to get wrong on your own:

- **Red zones.** Every device allocation it hands you has a poisoned 256-byte margin either side,
  verified after the launch. An out-of-bounds write is the one fault that otherwise reports as a
  pass, because in a harness the bytes past your operand belong to nobody. AMD's device
  AddressSanitizer needs `xnack+` and RDNA has none, so this is what there is.
- **Fusion.** A row with `replaces` is run against the chain it names, out of *your* library, and
  the outputs compared byte for byte. That is a stronger claim than a tolerance against the
  oracle, and it is the claim a fusion actually makes.
- **Divergence.** If your `opd_shape` describes an output differently from libref's, you are told
  so by name with both descriptions printed, and the case is counted apart from the failures. Two
  libraries can mean different things by one op name; that is a conversation, not a bug in your
  arithmetic.

If your kernel is skipped rather than checked, the report names why. The usual cause is a missing
`opd_shape` (§4) — without it the tool will not invent an extent for you.

- `--debug-selection` prints the selection table: which kernel took each band, and the constraint
  that refused every other candidate.
- `/graph` on a running server carries the declared graph *and* which declarations the run phase
  has actually issued — the way to find a band you resolved for nothing.
- `--debug-accept-reference-kernels` is required to run with a reference fallback in place. Without
  it, falling back is a fatal error, on purpose: a reference kernel in production is a correctness
  result reported as a performance one.
- `rad-info --plugins` prints every plugin the search path offers, as the engine loads it: kind,
  version, what it declares, whether it is a reference, whether its device code runs on this card,
  and the file it came from.

---

## 10. A library for a card radiance has never seen

The engine holds no opinion about cards. Its own device code — the copies and fills every transfer
runs, the step's bookkeeping, the weight-table patches — is compiled for every AMD GPU family's
generic target (`gfx9-generic`, `gfx9-4-generic`, `gfx10-1-generic`, `gfx10-3-generic`,
`gfx11-generic`, `gfx12-generic`) beside whatever the build targeted, so one engine binary starts on
any of them. What it computes there is whatever kernel library you install.

**Build for the card, or for its family.** `-DRAD_GPU_TARGETS=gfx942` or `=gfx11-generic` when you
configure; the default is the targets the installed radiance was built for. When a code object is
needed the engine asks the runtime which targets the card accepts, in order — the chip itself, then
its family (`gfx1036`, then `gfx10-3-generic`) — each with the modes the card runs in
(`gfx942:sramecc+:xnack-`), and takes the best fit your fat binary holds: the exact chip over the
family, a code object built for the card's mode over one built for either.

**A library with nothing for this card is left out, not failed.** When a kernel library loads, the
fat binaries its `dlopen` registered are checked against the card the ranks will run on, the same
check its first launch would make. If none fits, its device rows are left out of selection and the
log says what it carries and what the card runs; its host rows stay. So one installation can carry
a library per card. A library that knows at `rad_plugin_open` that it does not serve the machine
can also decline it with `RAD_E_UNSUPPORTED`.

**Which devices.** Discrete cards are used when there are any, and of those the ones sharing the
first one's architecture; integrated graphics only when it is all there is. `ROCR_VISIBLE_DEVICES`
chooses explicitly. On integrated graphics the device memory is the host's: the budget is bounded by
what the host has available and leaves an eighth of the host's memory to it; `--vram-kv-mib` and
`--vram-weights-mib` state the pools outright.

**What the core assumes of a card: nothing about its wave size**, its LDS, its fp8 hardware or its
links — those are your kernels' business. Three things are worth knowing before you start:

- **The vocabulary is what you implement.** A model runs on your library when every op its
  architecture issues has a row in it (`docs/OPS.md`; `--debug-graph` lists them). A bf16 model
  in the tree issues only ops libref also implements — MiniCPM5-2B is 13 model ops and 11 sampler
  ops — so libref's host code is the specification and `rad-kbench` the check. Ops `docs/OPS.md`
  marks **device-only** are libr4d's own and have no reference yet: the fp8 models, and a drafter
  with a 2-bit head, issue some, and run on a library that implements them.
- **fp8 bytes are OCP E4M3** (`e4m3fn`, bias 7, max 448) everywhere the engine reads them: weights in
  a container, the KV cache at `--kv-cache-dtype fp8`, activations between ops. A card whose fp8
  is `e4m3fnuz` (gfx942) converts weights in its `relayout` hook — the codes are the same bits but
  0x80, the value half, so the scale doubles — and keeps activations and KV in OCP bytes, or
  converts at its own boundaries.
- **Tensor parallelism is ops you supply.** `all_reduce`, `all_gather` and the fused collectives
  are rows in a kernel library like any other; the engine inserts none and assumes no link.

---

## 11. Architecture plugins

An architecture plugin declares a model in two phases (`spec.md` §3): `rad_arch_declare` names
every weight, buffer and op once, against the model's metadata, and `rad_arch_step` issues the ops
for one step. The contract is `abi/rad_builder.h` and `abi/rad_runtime.h`; the helpers every
in-tree architecture is built from — weight declarations, the gated-attention and MoE blocks, the
MTP and DFlash2 drafters — are installed as `<arch/rad_arch.h>` and its siblings, and the in-tree
plugins include them by exactly that spelling.

```c
const char* rad_arch_id(void);          /* "llama" -- what a container's header names */
const char* rad_arch_quant(void);       /* optional: "" unquantised, or the descriptor served */
int         rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
void        rad_arch_step(struct RadCtx* c, const struct RadBatch* batch);
int         rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out);          /* optional */
const RadChatFormat* rad_arch_chat_format(const RadModelMeta* meta);             /* optional */
```

`rad_plugin_info` reports `RAD_PLUGIN_ARCH`. The engine picks the plugin whose
`(rad_arch_id, rad_arch_quant)` pair equals the container's — exactly, with no fallback — so one
model family at two weight formats is two plugins. A checkpoint's id is its `config.json`
`model_type` with the underscores removed; `rad-convert --arch ID` overrides it. Every key of the
checkpoint's configuration reaches the plugin in `RadModelMeta`, and the plugin's name map says
which checkpoint tensor each declared weight comes from, so `rad-convert` needs no knowledge of the
model.

What to know first:

- **Stay inside `docs/OPS.md` and the model runs anywhere.** An op libref implements has a host
  fallback, an oracle and a specification every kernel library can be checked against. An op you
  invent runs only on a library that implements it.
- **A fusion is an op name.** The engine does not rewrite your graph; a kernel library's
  `replaces` declares an equivalence for checking, not a substitution. Ask for a fused op where one
  exists and fall back to its parts where it does not resolve — an op no library serves is
  declared as `RAD_NULL_HANDLE` (`rad_op_resolved`), which is an error only if it is issued — the
  way the in-tree blocks do for `qk_norm_rope_gate`.
- **Read §8** if the model has a linear or recurrent block.
- **Images go through the Qwen-VL processor** (`core/mm`): patching, the pad-token expansion and
  three-part rotary positions are the core's. A vision model with another image pipeline needs a
  core change today.

---

## 12. Quantiser plugins

A quantiser turns a weight, handed to it in f32, into the planes of an **encoding**
(`abi/rad_encoding.h`) — the numbers, in one canonical row-major arrangement any kernel library can
lay out its own way at load. The contract is `abi/rad_quant.h`:

```c
int                     rad_quant_count(void);
const RadQuantizerInfo* rad_quant_at(int i);   /* name, options, encoding, row_block, quantize, decode */
```

`rad_plugin_info` reports `RAD_PLUGIN_QUANT`. `rad-convert` drives quantisers through a **recipe**,
ordered rules from a weight's logical name to a quantiser and its options:

```
# PATTERN                    QUANTISER  OPTIONS
blk.*.ffn_*_exps.*.weight    myquant    bits=3 group=64
*                            rtn        codes=fp8_e4m3 block=128x128 scale=bf16
```

`rad-convert --list-quantizers` prints every quantiser loaded and the options each takes; an option
a recipe gives that the quantiser does not declare is refused before anything is quantised. Two
plugins naming the same quantiser are refused, both named.

An encoding whose scheme is `plain` or `affine` is decoded by the core. A scheme of your own is
decoded by your `decode` hook, which the oracle, `rad-kbench` and `rad-convert`'s error report all
read through; answer `RAD_E_UNSUPPORTED` for a scheme you did not write. A container in a new
encoding then needs two more things to SERVE: a kernel library whose `layout` hook accepts it, and an
architecture that declares the weight at it. The in-tree architectures' linear and expert
declarations take the encodings their blocks were written for, so a format that is not one of
those is served by a kernel library and an architecture plugin of its own.

---

## 13. Building, installing, and the search path

```cmake
cmake_minimum_required(VERSION 3.21)
project(mykernels LANGUAGES CXX HIP)   # CXX alone for a host-only plugin
find_package(radiance REQUIRED)
rad_plugin_defaults()                  # C++20, an optimising build type unless one was asked for
rad_add_plugin(mykernels kernels SOURCES rows.cpp HIP_SOURCES kernels.hip)
```

The kind is `kernels`, `architectures` or `quantizers` — the directory the engine loads it from.
`HIP_SOURCES`, not `SOURCES`, for device code: only those are compiled as HIP, for
`RAD_GPU_TARGETS` (§10) or a list you pass as `GPU_TARGETS`, at `-O3 -DNDEBUG` outside Debug,
uncompressed, against the HIP runtime the installed radiance found. `rad_add_plugin` is the same
function radiance builds its own plugins with, deliberately not a copy, and every plugin directory
in radiance's tree opens with these lines guarded by `if(NOT COMMAND rad_add_plugin)`, which is
what lets each be built inside the tree or on its own.

`find_package` also gives you `RADIANCE_ABI_VERSION`, so you can refuse at configure time rather
than at `dlopen`, and `RADIANCE_HOME`, `RADIANCE_KERNELS_DIR`, `RADIANCE_ARCHITECTURES_DIR` and
`RADIANCE_QUANTIZERS_DIR`. `cmake --install` puts the plugin under the home of the radiance you
built against; set `RAD_PLUGIN_INSTALL_HOME` to install somewhere else. Before installing, the
build tree's own `radiance_home/` is a working home for the plugin alone.

**`$RADIANCE_HOME` is a search path**: homes separated by `:`, the first winning. Each may hold any
of `kernels/`, `architectures/` and `quantizers/`; a plugin file an earlier home holds shadows one
of the same name later on. So your own plugins go ahead of an installation without copying it:

```sh
RADIANCE_HOME=$HOME/radiance-plugins:/opt/radiance/share/radiance radiance --model ...
```

and the container image takes one the same way, mounted:
`-v /my/plugins:/plugins -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance`.

**The kernel hierarchy** is `--kernels` (or `$RADIANCE_KERNELS`), an ordered list: `--kernels
mykernels,libr4d` puts you in front. An entry matches a plugin's name or its file stem, and **the
list is comma-separated** — the tools also accept colons, the engine's `--kernels` and
`$RADIANCE_KERNELS` do not, so write commas everywhere. A library on the search path that nobody
named loads after everything named, and a reference library (§1) after that.
