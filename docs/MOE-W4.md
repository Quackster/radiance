# The four-bit MoE expert format

Qwen3.8-Flash-Next has 512 experts of 640 over 48 layers. At E4M3 that is **112.8 GiB of expert
weight** against two 32 GB cards. Splitting the experts over the two ranks halves the per-rank
share and the placement planner still refuses 26 GiB. A routed weight can be served from the file
tier (`--weights-disk-tier`), but it is then read a layer at a time and a step is paced by the
drive. Pinned host memory does not close it either on a host with 64 GB of RAM (2 x 34.78 GiB).

So four bits is not an optimisation here. It is what lets the model run from VRAM and host memory:
an expert unit is 2.42 MiB instead of 4.69, the per-rank share is ~29 GiB, and that fits VRAM plus
a few GiB of pinned host.

## What is served

Qwen3.8-Flash-Next is served as `w4nl64-i8lm.rad`, built by
`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` with `CALIB=<dir>` naming the calibration Grams
(below, "Calibration"): the routed experts at `w4nl64a8h` (the w4nl codebook below, an E4M3 scale
a row and group of 64, GPTQ and three sweeps of coordinate descent), ten protected experts bf16,
and the trunk's block linears and the lm_head **int8, served W8A8** -- `gemm_i8a8_nt_m16` /
`gemm_i8a8_tiled` and `logits_gemm_i8` (docs/OPS.md), every quantising producer writing the int8
codes its consumer reads, so an int8 trunk adds no quantiser launch. How each step moved the model, by the engine's KL mode against the
bf16 model over the same 83K positions (docs/TOOLS.md, "The KL mode"):

| | all: mean KL | all: top-1 | text + assistant: mean KL | text + assistant: top-1 |
|---|---|---|---|---|
| int4 `-8..7`, round to nearest, E4M3 trunk (`qwen4exp-w4-hc8m.recipe`) | 0.1357 | 90.35% | 0.0440 | 92.46% |
| `w4nla8h` experts, GPTQ | 0.1098 | 91.36% | 0.0363 | 93.30% |
| + int8 W8A8 trunk | 0.0841 | 92.46% | 0.0277 | 94.07% |
| + int8 lm_head | 0.0830 | 92.84% | 0.0265 | 94.54% |
| **+ E4M3 scales a 64, coordinate descent (`w4nl64a8h`) -- served** | **0.0789** | **92.85%** | **0.0239** | **94.57%** |
| + every down projection at five bits (`w5nl64a8h`) -- not served | 0.0717 | 93.34% | 0.0204 | 95.10% |
| a second bf16 implementation (the floor) | 0.0399 | 95.13% | 0.0111 | 96.44% |

The served container's tail: median 0.0037, p99 1.65, p99.9 5.33. **The floor is not zero and is
most of the overall mean**: the tool-documentation system prompts in the corpus are numerically
chaotic, and two bf16 implementations already disagree there at KL 0.142 -- so text and assistant
turns are the slice comparable with other quantisers' published numbers, and the overall mean of
~0.04 is the floor itself rather than a target a quantised model can reach.

What it costs, two cards, production flags, against `w4nla8h` with the same trunk: decode
15.28 -> 15.32 ms a step (+0.3%), a 26K-token prefill 6539 -> 6442 tokens a second (-1.5%). The
five-bit down projections cost decode +4% and prefill **-19%**: the experts are 8% larger, fewer of
them fit beside the KV cache (137.5 resident a layer against 149), and a prefill streams the rest.

**Where the rest of the loss is.** Emulated with the experts of some layers left bf16 and the rest
as served (an HF reference with the container's quantisers, which matches the engine to 0.001):

| | all: mean KL | all: top-1 | text + assistant | system / user |
|---|---|---|---|---|
| served | 0.0789 | 92.85% | 0.0239 | 0.2732 |
| every down projection bf16 | 0.0700 | 93.61% | 0.0203 | 0.2456 |
| layers 16-31 bf16 | 0.0631 | 93.51% | 0.0226 | 0.2064 |

Both are a third of the experts' bytes. Five-bit down projections already take ~80% of what bf16
down projections would; layers 16-31 hold more of the overall mean -- almost all of it on the
chaotic system and user turns -- and text and assistant turns are most sensitive in layers 24-39.
Round-to-nearest experts in one group of eight layers at a time, the rest bf16, put the excess
over the floor at .012, .017, .031, .026, .020 and .009 for layers 0-7 through 40-47.

**What did not pay**, measured: GPTQ's act order (no change once the
weight is rotated), its dampening at 0.003 and 0.03 (within 0.4%), a codebook of the layer's own
(Lloyd's fit is the same in every layer) or one refitted at groups of 64 or for the E4M3 grid
(worse), a power-of-two sub-scale a 32 under the 64-wide scale (0.0679 -> 0.0675),
a 256-word two-dimensional codebook a pair of weights (10% less error than the scalar table, for an
LDS lookup a byte), more protected bf16 experts (+4% bytes for 7% less error), and E4M3 scales a 32
(13% less error energy for +13-18% on the expert GEMMs).

**Further quality levers**, in order: five bits on both projections of layers
16-31 (the same +8% as the down projections, ~0.066 expected); fewer layers at five bits for a
smaller prefill cost; and `--expert-vs-cache-ratio`, which buys the residency back with KV
capacity. The formats, the kernel and a layer's own choice of form are all in the tree.

## The stored plane

Row-major packed nibbles, two's complement `-8..7`, with a **bf16 scale per (row, group of 128
along K)** -- not the fp8 path's 128x128 tile scale, which would spend all sixteen levels on the
largest row in a block. The kernel is `moe_gemm_w4a8` (libr4d), which is the fp8 kernel with the
LDS staging and the scale fold changed: the integers `-8..7` are exact in E4M3, so a code expands
losslessly and the existing fp8 WMMA loop is untouched.

The encodings are **not interchangeable**:

| encoding | kernel dtype | what it means |
|---|---|---|
| `i4*bf16[1x128]` | `w4a8` | int4 codes, a bf16 scale per row and group of 128 along K |
| `i4*bf16[1x128]/fwht128` | `w4a8h` | the same, with a 128-wide Walsh-Hadamard along K applied before quantising |
| `affine:codes=u4[1x1],scale=bf16[1x128],table=f32{1x16}/fwht128` | `w4nla8h` | the rotated plane with the codes indexing the `w4nl` codebook (below) |
| `affine:codes=u4[1x1],scale=fp8_e4m3[1x64],scale.1=f32[*x*],table=f32{1x16}/fwht128` | `w4nl64a8h` | the codebook with an E4M3 scale a row and group of **64**, under one fixed f32 (below) -- the same bytes |
| `affine:codes=u4[1x1],scale=fp8_e4m3[1x32],scale.1=f32[*x*],table=f32{1x16}/fwht128` | `w4nl32a8h` | the same at a group of **32**, +0.125 bits a weight |
| `affine:codes=u5[1x1],scale=fp8_e4m3[1x64],scale.1=f32[*x*],table=f32{1x32}/fwht128` | `w5nl64a8h` | `w4nl64a8h` with **five-bit** codes into the 32-level `w5nl` codebook (below), +1 bit a weight |

The first two have the same planes, extents and byte count and differ only in the transform; the
container says which encoding each expert has. The architecture plugin asks for the first expert's encoding (`rad_weight_encoding`) and
declares the matching kernels, and a kernel that does not read an encoding is not a candidate for
it -- a rotated container cannot be served unrotated. libr4d reorders each row's nibbles into the
order its unpack reads when the experts load; the container holds them in the canonical order. The
activation side is `had_quant_act_fp8` / `gated_had_quant_fp8`, which are the ordinary fp8
quantisers with `r4d_fwht_wave<128>` between the load and the absmax -- free, because that
kernel's mapping (one wave a 128-column group, four consecutive columns a lane) is already the
transform's.

**The residual factor is not split.** `H` is symmetric, so `<Hw, Ha> = 128 <w, a>`, and the whole
factor of 128 is folded into the **stored weight scale** by the quantiser (`bf16(amax/7) / 128`). The
int8 rotated ops fold `1/sqrt(group)` into each side; these fold nothing at runtime. That the fold
goes entirely into the stored scale rather than being split is normative in `docs/OPS.md`
(*Quantisation and rotation*, the `group` parameter under the `had_*` FP8 forms), which is also
where the cost of splitting it is stated.

## The w4nl codebook

A Hadamard-rotated expert weight is close to normal, and sixteen evenly spaced levels are not the
best sixteen for a normal value: the outer levels are rarely used and the inner ones are too far
apart. `w4nla8h` keeps everything about the rotated plane -- the bytes a code, the bf16 scale a row
and group of 128, the rotation, the activation quantiser, the GEMM -- and changes only what a code
stands for:

    codes 0..7    1.125  3.25  5.5  8  11  14  18  22
    codes 8..15   the same, negated

Lloyd's quantiser fitted to the rotated, amax-normalised experts of Qwen3.8-Flash-Next gives the
same levels in every layer and both projections once each is rounded to E4M3, and every one is
exact in E4M3 -- so the kernel expands a code to the byte the fp8 WMMA reads with no rounding, as
it does `-8..7`. The decode is the same three byte permutes with another four dwords in them: the
table is a kernel argument (`r4d_args.h`, `kW4IntPerm` / `kW4nlPerm`), so both formats are one
code path and the same fifteen VALU per eight weights.

In libquant it is the named codebook `table=w4nl`, which a recipe pairs with `rule=search` (each
group's scale searched for least error, from 0.70 to 1.10 of amax / 22 and each candidate's
least-squares refit) and `gptq`. **An encoding cannot say what its table holds**, so the codes view
selects the `table` plane beside the codes, and the relayout compares it with the kernel's value
for value before it serves a byte: a container quantised against any other table of the same shape
is refused at load.

Measured as held-out output error (`sqrt(tr(dW H dW^T) / tr(W H W^T))`, the layer's pooled gate_up
Gram and each expert's own down Gram, 6 experts in each of 7 layers), at the same 4.125 bits a
weight:

| | gate_up | down |
|---|---|---|
| `-8..7`, scale amax/8 | 0.0889 | 0.1048 |
| `-8..7`, searched scale | 0.0849 | 0.1005 |
| `w4nl`, searched scale | 0.0755 | 0.0896 |
| `-8..7` + GPTQ | 0.0629 | 0.0832 |
| `w4nl` + GPTQ | **0.0547** | **0.0720** |

`-8..7` at amax/7, what `w4a8h` containers were converted with, is a further ~10% worse than the
first row. Halving the group to 64 at bf16 scales (+3% of the expert bytes) does less than the
codebook does for nothing -- and at E4M3 scales it costs nothing at all (below).

## Finer groups at the same bytes: E4M3 scales under a fixed 2^-13

A bf16 scale a 128 is sixteen bits a 128; an E4M3 scale a 64 is the same sixteen. What an E4M3
scale lacks is range, and the experts do not need it: with the rotation's 1/128 folded in, the
scales of Qwen3.8-Flash-Next's routed experts at a 64 sit between 2^-15 and 2^-10.6 (every 37th
expert of eight layers), so under one fixed factor of **2^-13** each is a normal E4M3 value with
four binades to spare either side -- and E4M3's three mantissa bits are the same in every binade,
so a per-row second level would buy range nobody uses and no precision. The factor is libr4d's
`kW4G64Scale`; a container states it as libquant's fixed second level (`scale2=f32 block2=*x*
scale2_value=0.0001220703125`), and the scale view's relayout compares it with the kernel's before
it serves a byte, as the codes view does the table. libquant searches each group's scale **as
stored**: once the second level is known the first is searched again over the E4M3 values it can
take, or the rounding gives back most of what the search won.

The kernel stages a 128-wide tile's E4M3 scales as one 16- or 32-bit load, sums each scale group's
WMMA in its own accumulator -- folding a group as it ends would make the next group's WMMA wait on
its results -- and folds them together at the end of the tile. Held-out output error, GPTQ + `w4nl`
(layer 20 and 28's experts):

| | gate_up | down | bits a weight |
|---|---|---|---|
| bf16 a 128 (`w4nla8h`) | 0.0683 / 0.0708 | 0.0842 / 0.0957 | 4.125 |
| E4M3 a 64 under 2^-13 (`w4nl64a8h`) | 0.0655 / 0.0679 | 0.0807 / 0.0917 | 4.125 |
| E4M3 a 32 under 2^-13 (`w4nl32a8h`) | -- / 0.0632 | -- / 0.0850 | 4.25 |

and three sweeps of coordinate descent after GPTQ (libquant `cd=3`) take another ~2% off either.
What the finer groups cost in the GEMM, against `w4nla8h` at a decode step's 4 tokens and a
2048-token chunk: `w4nl64a8h` +4% / +2% and +7% / +4% (gate_up / down); `w4nl32a8h` +13% / +15%
and +18% / +11% -- the 32-wide form is for the layers that need it, not the model.

## Five bits where the error is: w5nl64a8h

The 64-wide form at one more bit a weight. libquant's `table=w5nl` is Lloyd's quantiser with
sixteen magnitudes, fitted to the rotated experts at groups of 64 as `w4nl` was -- the same to
three places in every layer and both projections -- and rounded to E4M3 at the top level (30) that
leaves the sixteen distinct and moves them least: 0.8125, 2.5, 4, 5.5, 7.5, 9, 11, 12, 14, 16, 18,
20, 22, 24, 26, 30, then their negatives. The order makes a code's high bit its sign, so the
kernel expands the low four bits exactly as a `w4nl` code -- out of the sixteen magnitudes -- and
sets each byte's sign from a bit plane stored after the row's nibbles. That plane is a dword a 32
elements, transposed so that a shift by m puts the signs of the expansion's output dword m on
E4M3's sign bits at once: two VALU per four weights.

Held-out output error under the production quantiser (GPTQ, an E4M3 scale a 64 under 2^-13, three
sweeps of coordinate descent), layers 20 and 28:

| | gate_up | down |
|---|---|---|
| `w4nl64a8h` | 0.0638 | 0.0943 |
| `w5nl64a8h` | 0.0335 | 0.0501 |

-- 72% less error energy for 25% more expert bytes on whatever it is applied to. In the GEMM
against `w4nl64a8h`, at a 2048-token chunk: +19% gate_up, +8% down, most of it the extra bytes
streamed. At a decode step the projection reads 25% more from DRAM. (`r4d_selftest --perfmoe` at a
decode step's 4 tokens times one routing again and again, so its experts sit in the 64 MB last-level
cache: there the five-bit gate_up falls out of it and reads 3x slower, which a served decode step,
reading ~3 GB of experts, never sees.)

Every down projection at five bits (`data/recipes/qwen4exp-w5dn64-i8-hc8m.recipe`) is measured in
"What is served" above: 0.0717 / 93.34% against 0.0789 / 92.85%, for -19% prefill.

**WHERE IT GOES IS A PER-PROJECTION, PER-LAYER CHOICE.** The codebook forms at E4M3 scales --
`w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h` -- share the rotated input, the scale views and the
accumulators and differ only in the GEMM's dtype, so a recipe may put any projection of any layer
in any of them, and the MoE block takes each layer's two from that layer's own experts.

## Rotation alone is expected to be worse

A rotation fixes **range**, and a group-128 absmax already handles range. Measured on the 27B,
Hadamard was **2x worse** for RTN int4, and e4m3 activations went 0.02024 -> 0.02254 KL under one.
The rotation is here because it is the substrate an error-feedback packer needs underneath it --
the pair is what gets measured, not this half: a recipe asks for `transform=fwht128` together with
`gptq`, not on its own.

## Calibration, and why it is a separate run

GPTQ needs `H = sum over tokens of x x^T` for the layer's real input. That needs the model
**running**, and a container has to be written before it can be served. So the order is a
bootstrap:

1. **convert with plain absmax** -- the experts quantised by `rtn`
2. **serve that** with `RADIANCE_CALIB_DIR` set, and push a corpus through prefill
3. **convert again**, the experts quantised by `gptq calib=<dir>`

The production recipe takes the second route below instead: its Grams are recorded from the bf16
checkpoint.

A second moment is robust to the small perturbation the first container's own error puts on the
activation, which is what makes step 2 legitimate.

**Or calibrate the checkpoint as it ships, with no bootstrap at all.** Converted without a recipe,
Flash-Next is a 335 GiB bf16 container, and `--weights-disk-tier` serves it at two cards by reading
each routed layer's non-resident experts from the container a pass at a time (GUIDE.md §4.3). Its
tap reads the bf16 activation the bf16 experts multiply -- no upstream layer quantised -- and writes
its Hessians in the model's own domain; a rotated recipe's GPTQ reads them rotated into its own.
Whole experts per rank, which a calibration run wants anyway. A step reads about 100 GiB a rank from
the drive and grows only slowly with what it carries -- 50 s at ~32K tokens, 59 s at ~55K -- so
drive it with large prefill steps: `--max-num-batched-tokens 65536 --max-num-seqs 64`, about 96
requests in flight, and `--expert-vs-cache-ratio 0.5` so the cache share holds 64 sequences'
recurrent state. That records ~930 tokens a second, ten million tokens in three hours.

**The corpus is what the Hessian is of, so it has to look like the traffic.** A few thousand
tokens of one repeated sentence gives a Gram whose energy sits in a handful of directions, and GPTQ
then hides its error in the directions the corpus never excites -- an in-sample number that says
nothing. Score on Hessians from a held-out slice of the corpus, never on the ones the weight was
fitted to.

**A million tokens is enough for the tap's Hessians; the held-out slice wants more.** The tap pools
a layer's Gram over every expert, and on Flash-Next one recorded from 10M tokens beat one from 1M by
under 1% on 1.8M held-out tokens (GPTQ on gate_up, layers 0-47), with no change on down or under
AutoRound. What the larger corpus does fix is the scoring: the gap between in-sample and held-out
error drops from up to 3% to under 1% on gate_up, and a 113K-token held-out slice put layer 47's
down GPTQ at 0.49x RTN where 1.8M tokens say 0.60x.

```sh
# 1. the uncalibrated container: the recipe's expert rule is rtn, int4 g128 rotated
rad-convert <checkpoint> -o model-absmax.rad --recipe data/recipes/qwen4exp-w4.recipe

# 2. the calibration run. Stops at RADIANCE_CALIB_TOKENS (default 65536) and writes once.
RADIANCE_CALIB_DIR=<calib dir> RADIANCE_CALIB_TOKENS=65536 \
  radiance --model model-absmax.rad ...
# ...then drive prompts at max_tokens=1 until it prints
#    radiance: calibration wrote layer Hessians after N tokens to <calib dir>

# 3. the calibrated convert: a --quant rule ahead of the recipe's own puts the experts through
#    GPTQ against calib=$CALIB (a recipe whose expert rule names calib=$CALIB itself, as the
#    production one does, needs no --quant)
CALIB=<calib dir> rad-convert <checkpoint> -o model-gptq.rad \
    --recipe data/recipes/qwen4exp-w4.recipe --reuse model-absmax.rad \
    --quant 'blk.*.ffn_*_exps.*.weight=gptq:codes=i4,group=128,scale=bf16,transform=fwht128,calib=$CALIB'
```

`--reuse` in step 3 copies every weight whose encoding, quantiser and options did not change -- all
but the experts -- from the first container. **The format is a property of the container**: each
expert's encoding is in its entry, the plugin reads it from there, and the recipe that made it is
in the header (`rad-info --recipe`), `calib=` included, so a reader can tell an error-feedback
plane from a plain absmax one.

The files are `gram.r<rank>.<weight stem>.bin`: a 32-byte `RADGRAM2` header (`k`, `rank`, `rows`, `domain`)
then `k*k` f32. `domain` is what the tap summed: 0 for the model's own activation, 128 for the
rotated one a `w4a8h` container's kernel multiplies; libquant brings every file into the domain the
recipe's grid quantises in before summing them, so a bf16 model's Hessians serve a rotated recipe.
**Named after the weight they will pack, minus the expert index**, because that is
the whole protocol between the engine and the quantiser -- `gptq` is given a directory and a weight
name, and strips `.<e>.weight` to find its file. A name of the tap's own
would have made the quantiser carry a per-architecture table.

**`down` gets a Hessian an expert.** A routed expert's `down` input is its own intermediate units,
so channel `i` of one expert has nothing to do with channel `i` of another, and a Gram pooled over a
layer's experts describes none of them: on Flash-Next a typical expert's own profile has a median
cosine of 0.15-0.87 with the pool, and in the last layers the pool *is* one expert. The tap also
sums each local expert's rows into its own Gram -- the same `gram_accum` once an expert over its run
of the sorted rows, into a layer-sized scratch that comes back to the host every pass (640^2 f32 for
256 experts a rank is 10 GB a rank as packed triangles) -- and writes
`gram.r<rank>.blk.L.ffn_down_exps.<e>.bin` beside the pooled file. libquant reads an expert's own
file when it holds at least 4K rows and the layer's otherwise. Fitted to its own Gram, a Flash-Next
expert's down projection comes out 20-55% below round-to-nearest on its own held-out Gram, where the
pooled Gram bought 0-10%; the super experts gain most (layer 47's expert 399: 0.204 rounded, 0.137
pooled, 0.065 own). Under ~2K rows the own fit fits its rows and can lose to rounding, which is why
the threshold: below it sit 0.6% of the routed rows. The per-layer sync it costs roughly
halves a disk-tier recording's throughput, which is why the per-expert sets are recorded from a
million tokens.

Three things about the tap worth knowing before reading its numbers:

- **The gate_up tap is the unsorted activation and loses nothing.** Every token is routed to
  exactly `top_k` experts, so the gathered distribution IS the token distribution times `top_k`,
  and GPTQ is invariant to a positive scale on H. `top_k` times cheaper for the same matrix.
- **It taps the quantiser's output, not the bf16 before it.** The Hessian wanted is of the operand
  the expert GEMM actually contracts -- and under `w4a8h` that is the only form the activation has.
- **Per rank, and the quantiser sums.** Under expert parallelism a rank's `down` input is only the
  tokens routed to its own experts. Its gate_up input is the whole token set on every rank, so
  summing those double-counts by `world` -- a positive scale, harmless. One rule beats two.

## Protected experts

**A few experts carry a large share of a layer's output, and those are kept plain bf16.** In
Flash-Next's last layers one expert's `down` input holds a sixth of the layer's energy from a third
of a percent of its routing, nearly all of it in one channel (layer 47, expert 445). Four bits cost
it the same *relative* error as any expert -- about 12% of its output under its own profile -- but its
output is that much larger, so the absolute error is too. unsloth's GGUFs upcast the whole layer's
`down` for this (a GGUF type is per tensor, and a tensor is all 512 experts); a container keeps an
expert's encoding per expert, so here only those experts are kept, at ~7 MB each.

A recipe names them with a rule ahead of the expert rule, `blk.47.ffn_*_exps.445.weight cast
dtype=bf16`, and the layer serves them apart: the sort places them last (`moe_scatter`'s `protect`),
the four-bit GEMMs serve the other experts, and a bf16 `moe_gemm` pair serves the trailing run of the
offsets, writing the routed rows the four-bit `down` GEMM left zero. Their weights are the layer's
static weights, not expert slots -- a container's experts in a layer are one fixed-size unit each --
so they are always resident and the file tier never reads them. Where every rank holds a slice of
every expert, it holds an even slice of these too -- 320 of the 640 intermediate units at two ranks,
since a bf16 expert has no scale blocks to respect -- and the block's all-reduce sums the partials.
The count must keep the rest even, because the four-bit GEMM takes them as two equal tables by parity.

## The quantiser

`gptq` in libquant (`libquant/lq_gptq.{h,cpp}`), over any grid `rtn` takes. H is damped by 1% of its mean diagonal, inverted, and re-factorised
into the upper triangle `U` with `U^T U = H^-1`; then columns are quantised left to right and each
column's rounding error is pushed into the columns to its right, weighted by `U`. Blocked 128
columns at a time so `U` is streamed once rather than once per row, and parallel over rows because
the recursion is independent in the row index.

**The scale grid is fixed from the weight before any code is chosen** -- the "static groups"
variant. The scales are then exactly the ones `rtn` writes for the same weight, which
`tests/relayout_parity_test.cpp` checks, and the grid does not depend on the order the columns are
visited in.

**Cost.** Budget tens of minutes of extra convert time for a 48-layer container, and ~1.3 GB of
factor cache held for its duration. The quantiser is OpenMP-parallel over rows, so the wall time
falls with core count; the cache is a property of the model, not of the machine. The factorisation
is per layer and projection, not per expert: 512 experts share one factor.

**A weight with no calibration file is packed the uncalibrated way and says so on stderr** -- so a
typo'd directory is a message, not a silent quality regression.

## What is measured

- `r4d_selftest --case moe_gemm_w4a8` -- the kernel against an independent host decode of the same
  planes, both M bands, through the relayout the loader runs.
- `quant_test` -- **GPTQ against the objective it minimises**. The metric is
  `||X (W - W')^T||^2 = sum over rows of e H e^T` and *not* rel_l2 on the weight: error feedback
  makes the weight worse in the plain sense, deliberately. The control is no Hessian at all, under
  which `gptq` must produce `rtn`'s planes **byte for byte**.
- `r4d_selftest --case had_quant_act_fp8` -- every format against the exact f32 dot product,
  which is the only way to see whether the quantiser's `fwht_inplace` and the device's
  `r4d_fwht_wave` are the same transform, and whether the kernel decodes the `w4nl` codes as their
  table entries. 0.2015 (`w4a8`), 0.1307 (`w4a8h`), 0.0991 (`w4nla8h`, round to nearest at
  amax / 22), 0.0761 (`w4nl64a8h`), 0.0625 (`w4nl32a8h`) and 0.0469 (`w5nl64a8h`) on a source
  with outlier channels; the negative control (rotate the activation, not the weight) measures
  5.32.
- `relayout_parity_test` -- the `w4nla8h` plane's stored bytes pinned, its inverse giving back the
  codes and the table, and a container quantised against another sixteen-entry table refused at
  relayout; the same for `w4nl64a8h`, `w4nl32a8h` and `w5nl64a8h`, with a second level other than
  2^-13 refused, a 32-entry table other than `w5nl` refused, and each refusing the others'
  encodings. `r4d_selftest --case moe_gemm_w4a8` also checks `w5nl64a8h`'s arranged row bit for
  bit against the statement in r4d_layout.cpp, and its inverse. `moe_gemm_q_mod4`'s four classes
  are held to `moe_gemm_q`'s stored bytes in every format, and for both ops what the op oracle
  reads through the inverse -- codes as their levels, an E4M3 scale times its second level -- is
  held to the quantiser's planes.
- `r4d_selftest --case gram_accum` -- the accumulator against an f64 host Gram, **launched twice
  over a seeded buffer**, because a kernel that stored instead of adding would agree with any
  single-call oracle.

**The end-to-end quality of a container against the bf16 model** is the engine's KL mode
([TOOLS.md](TOOLS.md#the-kl-mode)): the full-vocabulary KL at every position of a fixed corpus,
its mean and its tail percentiles, and the top-1 agreement. It needs the **full 48-layer
container**: the 4-layer bring-up slice cannot resolve a difference this small -- its residual
probe reads the same to within 1% for fp8, `w4a8`, `w4a8h` and GPTQ alike, which says only that
all four compute the same function.
