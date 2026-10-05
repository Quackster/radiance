# Architectures — the geometry, and the traps that stay plausible when wrong

One architecture plugin claims one **(architecture id, quantisation descriptor)** pair — that is the
selection key of `spec.md` §2.4, and it is why a family with two checkpoint formats is two `.so`
files rather than one that branches. `rad-convert` normalises whatever it read (safetensors or GGUF)
into the id an architecture plugin claims.

| plugin | id | quantisation | what it serves |
|---|---|---|---|
| `arch/llama_fp8` | `llama` | `fp8_e4m3` | the plain LLaMA decoder, from a checkpoint that already holds E4M3 codes and a `weight_scale_inv` plane |
| `arch/llama_dense_fp8` | `llama` | *(empty)* | the same graph from a bf16 checkpoint; `rad-convert` quantises. MiniCPM5-2B |
| `arch/qwen35_bf16` | `qwen35` | *(empty)* | the Qwen3.5/3.6/3.8 dense hybrid, served bf16 |
| `arch/qwen35_fp8` | `qwen35` | `fp8_e4m3` | the same, block-scaled FP8 weights and activations |
| `arch/qwen35moe_fp8` | `qwen35moe` | `fp8_e4m3` | `qwen35_fp8` with the feed-forward routed |
| `arch/qwen4exp_fp8` | `qwen4exp` | *(empty)* | Qwen4-Exp (Qwen3.8-Flash-Next), a bf16 checkpoint quantised at convert |

This document is the geometry of each, and the handful of readings that are **arithmetically
silent** — wrong in a way that produces fluent output and no error anywhere.

## The rule that applies to all of them

**`head_dim` is READ, never derived.** On the Qwen hybrids `head_dim` is 256 while
`hidden_size / num_attention_heads` is 213, so a derived value makes every downstream shape silently
wrong — the projection widths, the KV page size, the attention kernel's band. `core/engine_bringup.cpp`
takes it from the container's meta and says so at the line that does it.

---

# `llama` — the plain decoder

`model_type: llama` with `LlamaForCausalLM`, the most widely shared decoder shape there is. Every
layer is full attention; there is no gated delta net, no layer schedule, no output gate and no
per-head q/k norm, which is precisely what separates this plugin from `qwen35_fp8`.

MiniCPM5-2B is what it exists for: 42 layers, hidden 2048, 16 query heads over 2 KV heads at
`head_dim` 128, SwiGLU at 6144, RMS norms, full rotary at theta 5e6, an untied 130560-row head, and
no biases anywhere.

**Two checkpoint formats, one graph.** OpenBMB publishes MiniCPM5-2B in bf16 and nothing else, so
"MiniCPM5-2B FP8A8" is a statement about what this engine *runs*, not about what the checkpoint
holds. `llama_dense_fp8` declares the empty quantisation descriptor and the recipe
`data/recipes/minicpm5-2b-dspark.recipe` does the work at conversion — libquant's `rtn`, one amax
per 128x128 block, `scale = amax/448` rounded to bf16 — so the container holds the encoding a
checkpoint that shipped fp8 would. Both plugins declare the same ops over the same encoding, and the
kernels lay it out the same way at load, so nothing downstream can tell which one ran. For the same
reason the bf16 checkpoint itself is not served directly: its linears are not the fp8 this plugin
reads.

What is fp8: `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`, each with a bf16
`[N/128][K/128]` scale plane. What is not: every norm and the embedding. The lm_head is declared at
whatever the source holds — bf16 from a checkpoint, fp8 where the recipe made it so — and the
selection picks the GEMM that reads it. Both vocabulary edges are row-sharded.

---

# `qwen35` — the hybrid Gated DeltaNet / gated-attention family

Qwen3.5, Qwen3.6 and Qwen3.8 dense models share this architecture; HuggingFace calls it
`model_type: qwen3_5` (`Qwen3_5ForConditionalGeneration`) and GGUF calls it `qwen35`.

## Qwen3.8-27B-FP8, exactly

```
64 layers, full_attention_interval 4  ->  layers 3,7,...,63 are FULL ATTENTION (16 of them)
                                          the other 48 are GATED DELTA NET
hidden_size 5120      intermediate_size 17408      vocab_size 248320
num_attention_heads 24   num_key_value_heads 4   head_dim 256      ->  GQA 6
attn_output_gate true                                              ->  gated attention
partial_rotary_factor 0.25  ->  RoPE over 64 of the 256 head dims
mrope_interleaved true, mrope_section [11, 11, 10]                 ->  MROPE, not plain rope
rope_theta 1e7        rms_norm_eps 1e-6        max_position_embeddings 262144

linear_num_key_heads   16    linear_key_head_dim   128
linear_num_value_heads 48    linear_value_head_dim 128    ->  GDN GQA 3:1
linear_conv_kernel_dim 4     mamba_ssm_dtype float32      output_gate_type swish
mtp_num_hidden_layers 1      tie_word_embeddings false
quantization: fp8 e4m3, activation_scheme "dynamic", per-weight `weight_scale_inv`
vision: 27 blocks, hidden 1152, 16 heads -> head_dim 72, patch 16, out_hidden_size 5120
```

### Why this lines up with libr4d

Every geometry above lines up with a libr4d entry point, which is the strongest available evidence
that the kernel-plugin constraint model is the right shape.

| model geometry | libr4d entry point |
|---|---|
| head_dim 256, GQA 6, paged block 16 | `attn_prefill_h256_gqa6_*`, `attn_decode_h256_gqa6_*` |
| linear k128 / v128, chunk 64 | `gdn_chunk_scan_k128_v128_c64_bf16`, `gdn_kkt_solve_k128_c64_bf16` |
| conv width 4, k128 | `gdn_conv_prep_w4_h128_bf16`, `gdn_conv_update_w4_h128_bf16` |
| recurrent state fp32 | `gdn_recurrent_update_k128_v128_bf16_fp32state` |
| gated attention, interleaved q/gate | `qk_norm_rope_gate` (reads the q/gate interleave off the operand's row pitch) |
| GDN gated norm over an interleaved z | `gdn_gated_norm_had_quant_i8` (`zgrp`/`zblk`) |
| vision head_dim 72 | `attn_vit_h72_bf16` |

## Tensor names — the FP8 safetensors checkpoint

Sharded one file per layer (`layers-N.safetensors`), plus `outside.safetensors` and
`mtp.safetensors`. Every FP8 weight carries a companion `weight_scale_inv`; the norms, biases,
`A_log`, `dt_bias` and `conv1d` stay bf16.

**Full-attention layer** (`model.language_model.layers.N.`):
```
input_layernorm.weight              post_attention_layernorm.weight
self_attn.q_proj.weight  (+_scale_inv)   [5120, 12288]  = 24 heads x 256 x 2, query and GATE
self_attn.k_proj.weight  (+_scale_inv)   [5120,  1024]
self_attn.v_proj.weight  (+_scale_inv)   [5120,  1024]
self_attn.o_proj.weight  (+_scale_inv)   [6144,  5120]
self_attn.q_norm.weight [256]       self_attn.k_norm.weight [256]
mlp.{gate,up}_proj.weight (+_scale_inv)  [5120, 17408]
mlp.down_proj.weight      (+_scale_inv)  [17408, 5120]
```

**Linear (GDN) layer**:
```
input_layernorm.weight              post_attention_layernorm.weight
linear_attn.in_proj_qkv.weight (+_scale_inv)  [5120, 10240]  = 2*16*128 + 48*128
linear_attn.in_proj_z.weight   (+_scale_inv)  [5120,  6144]  = 48*128
linear_attn.in_proj_a.weight                  [5120,    48]
linear_attn.in_proj_b.weight                  [5120,    48]
linear_attn.conv1d.weight                     [10240, 1, 4]
linear_attn.A_log [48]    linear_attn.dt_bias [48]    linear_attn.norm.weight [128]
linear_attn.out_proj.weight (+_scale_inv)     [6144, 5120]
mlp.{gate,up,down}_proj  as above
```

**Model level**: `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`,
`lm_head.weight` (**not tied** on this model), and the MTP head: `mtp.fc.weight`,
`mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`.

## The full-attention layer — gated attention

`q_proj` produces the query **and a gate, interleaved per head**: for each of the 24 heads the
first 256 elements are the query and the next 256 are the gate, so both are strided views with
pitch 512 rather than a contiguous split.

```
qg      = gemm(x, q_proj)                  [n_tok, 12288]
k, v    = gemm(x, k_proj), gemm(x, v_proj) [n_tok, 1024]
q       = rmsnorm(view(qg, off 0,   pitch 512), q_norm)     per head, over 256
k       = rmsnorm(k, k_norm)                                per head
gate    = view(qg, off 256, pitch 512)
q, k    = mrope(q, k, positions, sections [11,11,10], rotary dims 64 of 256)
o       = attn_paged(q, k, v)              scale = 1/sqrt(256)
o       = o * sigmoid(gate)                <-- the gate. output_gate_type "swish" names the
                                               GDN gate, not this one; this one is a sigmoid
out     = gemm(o, o_proj)
```

`qk_norm_rope_gate` does the two norms, the rope and the gate extraction in one kernel and takes the
interleaving off the `q_gate` operand's own row pitch, so the projection is declared as it is laid
out rather than converted at load. Declare it first and fall back to the unfused sequence.

## The linear layer — gated delta net

```
qkv     = gemm(x, in_proj_qkv)             [n_tok, 10240]
z       = gemm(x, in_proj_z)               [n_tok,  6144]
b       = sigmoid(gemm(x, in_proj_b))      [n_tok, 48]     -> beta
a       = gemm(x, in_proj_a)               [n_tok, 48]
g       = softplus(a + dt_bias) * A_log_term
conv    = silu(causal_conv1d(qkv, conv1d, conv_state))     depthwise, width 4, 10240 channels
q,k,v   = strided views of conv:  q at 0 (16x128), k at 2048 (16x128), v at 4096 (48x128)
q, k    = l2_norm(q), l2_norm(k)           <-- L2, NOT rmsnorm
o       = delta rule: chunked scan at prefill, recurrent update at decode
o       = rmsnorm(o, linear_attn.norm) * silu(z)           the gated norm
out     = gemm(o, out_proj)
```

`linear_num_value_heads` (48) is three times `linear_num_key_heads` (16), so q and k are shared
across three value heads. libr4d's GDN kernels take `H` and `Hg` separately for exactly this.

## Things that stay plausible when wrong, so get them right

- **`q`/`k` use L2 normalisation, not RMS.** Different op, different arithmetic.
- **`A_log` is a log.** In the GGUF conversion `ssm_a` is already `-exp(A_log)`; in *this*
  safetensors checkpoint the tensor is named `A_log` and the exponent has **not** been taken. The
  converter is the component that decides which convention reaches the kernel, and it must say so.
- **The GDN output gate is `silu` here and `sigmoid` on Qwen4-Exp.** `RMSNormGated` defaults to silu
  and every Qwen3.5/3.6 release leaves it there; Qwen4-Exp states `output_gate_type: "sigmoid"` and
  means it. A gate of `sigmoid(z)` where the model wants `silu(z)` rescales every value head's
  output — a config read, not a kernel to write. `arch/common/rad_block_gdn_fp8.h` refuses anything
  but those two names.
- **Two readings the checkpoint cannot settle by inspection.** The rope convention (`neox`, halves,
  against `gptj`, adjacent pairs) and which half of each head's `q|gate` pair is the query are both
  self-consistent either way, and the sigmoid accepts either half — so no per-op check can reach
  them. The MTP head's concat order (hidden-first or embedding-first) is the third of the same kind.
  Each is a declared knob at the site that reads it, for the same reason.

## KV groups

Three of them, which is why hybrid KV groups are first-class rather than an extension:

| group | kind | geometry |
|---|---|---|
| full attention | `RAD_KV_FULL` | 16 layers, head_dim 256, 4 KV heads; block size from the resolved kernel |
| GDN recurrent state | `RAD_KV_LINEAR` | 48 layers, 48 states of 128 x 128 |
| GDN conv state | `RAD_KV_CONV` | 48 layers, 2*16 + 48 = 80 heads x 128 = 10240 channels, a rolling `conv_width - 1 + n_spec` window |

**The recurrent state is f32 by default**, which is 48 x 128 x 128 x 4 = 3 MiB a layer and
**144 MiB across the 48 linear layers, per sequence** — and it is what the prefix-cache checkpoint
of `spec.md` §7.3 snapshots. Its read and write are most of what the two GDN kernels do, so halving
its width halves one of the largest kernels after the fp8 GEMMs. libr4d compiles an f16 form and
argues the numerics where the kernel lives — the elements are O(0.1) with no range problem and
f16's eleven mantissa bits put the round-trip error at 5e-4 relative. It is **off by default
anyway**, because it changes generated text without being asked to; nothing about the container
changes between the two, so there is no re-convert. How a speculative step keeps the state it can
roll back is `docs/OPS.md`'s `state_form`.

## FP8

`quant_method: fp8`, `fmt: e4m3`, `activation_scheme: dynamic`, and a `weight_scale_inv` beside
every quantised weight — DeepSeek-style block scaling, 128x128. Activations are quantised at
runtime, which is what "dynamic" means and what makes the activation quantiser an op rather than a
constant. The vision tower's first several blocks are in `modules_to_not_convert` and stay bf16.

**libr4d reads those bytes directly, so `rad-convert` COPIES**: no permutation, no repacking, no
dequantisation, and a `.rad` costs what the checkpoint does. That is the whole reason this plugin
family exists — widening a 27B FP8 checkpoint to bf16 doubles the weight bytes; `qwen35_bf16` is the
plugin for a checkpoint that is bf16 to begin with.
The rows are `gemm_fp8a16_nt_m1`, `gemm_fp8a8_nt_m16`, `gemm_fp8a8_gated_nt_m16` and
`gemm_fp8a8_tiled`.

**The shipped path is `fp8a8`**, and the reason is a band and not a preference. `dtype` is ONE string
per declared op, so a linear is `fp8a16` or `fp8a8` and cannot be both. `fp8a16` reads a bf16
activation directly and needs no quantise launch, which is strictly better — and its kernel serves
`M <= 1`, which is a single-sequence decode and nothing else. `fp8a8`'s two kernels cover every M
between them, and the price is one `quant_act_fp8` launch per activation that feeds a GEMM — *per
activation*, not per linear, which is why the quantiser is its own struct rather than a step inside
the linear.

**The scale plane is a WEIGHT operand**: `gemm_nt_q` takes `a, a_scale, b, b_scale`, and `b` and
`b_scale` are two declarations over the planes of one logical weight (`RadWeightDecl::planes`).
Each is placed, moved and sharded on its own — the engine cuts a row-sharded scale plane on its
block boundary exactly as it cuts the codes beside it.

## `qwen35moe`

This architecture with the MLP replaced by a router plus experts, and *only* that: the gated
attention, the gated delta net, the layer schedule, the rope, the vocabulary edges and the MTP head
are identical, and the places where a routed feed-forward differs are behind one conditional in the
shared source. It is a second translation unit rather than a branch on `n_expert` because the id is
half the selection key — one `.so` cannot claim both, and an operator looking at what is installed
should see two plugins for two architectures.

---

# `qwen4exp` — Qwen4-Exp / Qwen3.8-Flash-Next

`model_type: qwen4_exp`, a subclass of `Qwen3_5MoeTextModel` — so most of it is composition over the
blocks above. What is genuinely new is small and named: the gated residual, the n-gram/PLE layer,
QSA's indexer, stacked dense expert tensors, GQA 12, and the delta net's sigmoid output gate.

```
48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE)
hidden_size 2560      hc_count 4  ->  the residual stream is 10240
512 experts of 640 at top-10, plus a shared expert of 640
24 query heads, 2 KV heads, head_dim 256     ->  GQA 12
vocab 248320          max_position_embeddings 262144
indexer_n_heads 4  indexer_head_dim 128  indexer_budget 2048  indexer_compress_ratio 4
mtp_num_hidden_layers 1        output_gate_type sigmoid
```

**The residual stream is four times as wide, and the blocks do not own their norms.** There is no
`input_layernorm`, no `post_attention_layernorm` and no final `norm` in this checkpoint. In their
place every block has a gated residual connection — a grouped norm over the 10240 stream, a low-rank
data-dependent read gate that mixes the four sub-streams down to 2560, and four data-dependent
scalars that gain the block's output back into all four. 96 of them, plus a 97th with no write half
that collapses the stream for the lm_head. The consequence for the blocks is that they run in their
"input is already normed and quantised" mode and therefore give up two fusions — the residual add
folded into the next norm, and the all-reduce folded into it.

**What each weight is stored as is the recipe's decision.** The checkpoint is bf16 throughout and
carries no `quantization_config` at all, so the plugin's descriptor is the empty one: `rad-convert`
quantises by a recipe, and the plugin reads each weight's encoding from the container and declares
the op that reads it. The block linears — `self_attn.{q,k,v,o}_proj`,
`linear_attn.{in_proj_qkv, in_proj_z, out_proj}`, the shared expert's three — are block fp8 or
int8 (served W8A8). Every routed expert's gate_up and down is fp8, bf16, or one of the four- and
five-bit forms of `docs/MOE-W4.md`.
Each connection's two mixing matrices and the MTP head's `fc_hidden` / `fc_embedding` are E4M3 rows
or bf16: the low-rank width is 320, which is 2.5 scale blocks, so they are scaled a row group at a
time rather than block-scaled. The lm_head is whatever a `logits_gemm` kernel reads. Every norm
gain, `A_log`, `dt_bias`, `conv1d`, `in_proj_a` and `in_proj_b` (48 rows is not a whole scale
block), the router, `shared_expert_gate` and `embed_tokens` stay bf16.
`data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe` is the served choice: four-bit codebook experts with
ten protected experts left bf16, int8 block linears and lm_head, E4M3 mixing and MTP fc matrices,
and the n-gram table at one fixed E4M3 scale.

**Tensor parallel at two ranks splits every expert's width, unevenly.** `moe_intermediate_size` is
640 = 5 x 128, so an even share at tp2 is 320, which is not a whole number of 128-column blocks —
and the fp8 scale block, the four-bit scale group and the rotation all sit on them. So each rank
takes three blocks of the experts of one parity and two of the other's (`moe_gemm_q`'s two tables in
`docs/OPS.md`), the router stays replicated so both ranks agree on the routing, each rank computes
its slice of every routed expert, and the block's existing end-of-block all-reduce sums the
partials. No all-to-all, no extra collective. The shared expert's 640 columns split the same way,
three blocks on rank 0 and two on rank 1.

**Five blocks do not divide evenly, and an even share is not required.** 5 is odd, so at tp4 an
even share is 160 columns, 1.25 blocks — but the split is TWO SLICES per expert, not one slice a
rank: each parity's blocks are cut contiguously across the ranks and the rank taking the odd block
alternates with the parity, so every rank's share is whole blocks, every block of every expert has
exactly one owner, and each rank still computes a slice of every routed expert. At four ranks and
five blocks an even-parity expert is `[0, 2)` blocks on rank 0, `[2, 3)` on rank 1, `[3, 4)` on
rank 2, `[4, 5)` on rank 3, and each rank's place on the odd parity shifts by one — rank 1 takes the
fat block there instead of rank 0 — so no rank takes the extra block of both parities. A world
above the block count is refused at declare, by name: a rank with an empty slice computes nothing
and still joins the all-reduce. The shared expert's 640 columns split as evenly as whole blocks
allow: five blocks over four ranks is rank 2 `[0, 256)` and one block each on ranks 3, 0 and 1.

**At four ranks the classes are four.** Parity gives the extra block only two places to go, so
ranks 0 and 1 would compute three blocks of every even/odd pair of routed experts and ranks 2 and 3
two, and every block's all-reduce would have 2 and 3 wait. So at four ranks the cut above is made
per class of the expert's number mod 4, the rank's place shifted by the class: rank `r` takes
`[0, 2)` blocks of class `r` and one block of each other class, five blocks of every four experts
on every rank. That is `moe_gemm_q_mod4`, four slice tables where `moe_gemm_q` takes two
(`docs/OPS.md`), and every slice is one a rank already took of some parity. A layer that keeps two
experts plain bf16 has 510 quantised ones, and its first two classes are one expert longer. What
does not divide elsewhere stays apart: the shared expert's extra block is rank 2's and the
protected bf16 experts' ten 64-wide tiles go 3, 3, 2, 2.
Expert parallelism — whole experts a rank — is what this replaced, and why it stays replaced: the
recipe protects its bf16 experts in pairs precisely so each rank's quantised count stays even, and
sharding experts by rank breaks that parity property the weight tables rest on.

**Attention splits by KV head, and replicates once the ranks outrun the heads.** There are 2 KV
heads of 256 rows, so each rank declares `n_head_kv 1` and reads one head's row-span of `k_proj` and
`v_proj`. At tp2 that is the ordinary row shard. At tp4 the heads cannot divide four ways, so they
are replicated in rank pairs: ranks {0,1} carry KV head 0, ranks {2,3} carry KV head 1, and both
ranks of a pair hold byte-identical KV weights — more read traffic, no extra VRAM. `n_head` is 12 a
rank at tp2 and 6 at tp4, so gqa is 12 at tp2 and 6 at tp4 and the attention kernel is a different
instantiation (`attn_{prefill,decode}_h256_gqa6_bs4`, the QSA page of 4) than the gqa12 pair tp2
resolves to, not the same kernel over fewer heads. Any world size above the KV head count must be a
multiple of it: 4 is a multiple of 2, 3 and 5 are not, and those widths are refused at declare
time.

**The fused all-reduce forms are two-rank.** At four ranks nothing fuses: the hyper-connection pair
runs the standalone sequence — `all_reduce` (libr4d's 4-rank oneshot/twoshot), then
`hc_write_bf16`, then `moe_gather` — 2 to 3 extra launches per pair against the two-rank fused
form. The unfused sequence is declared beside the probe, so nothing is unresolved at boot. The
attention's own fusion is not a collective and does carry over: the sparse decode with the output
gate and the quantiser in its merge (`attn_paged_gate_quant`) has gqa-6, page-4 rows. The per-step cost against tp2 on the
same cards is unmeasured.

**One collective does generalise, because it has to.** The lm_head is vocab-sharded, so the sampler
gathers the ranks' candidate sets rather than their logits (`core/sample/sampler.cpp`) and an
`all_gather` is on every step's sample path at any world size above one. libr4d shipped a two-rank
form only — one peer flag, a `1 - rank` peer index, a destination stride of `2 * row16` — so the
N-rank gather is a new kernel, not a fallback: at four ranks nothing but the reference
implementation resolves and the engine refuses to serve rather than run a naive gather. It is the
same one-shot push as the N-rank all-reduce with the reduction removed, and it moves the
(u32 id, f32 logit) pairs without ever adding them.

**Four-bit experts are what make it fit.** 512 experts of 640 over 48 layers is 112.8 GiB of expert
weight at E4M3 and twice that at bf16. `--weights-disk-tier` serves what fits neither VRAM nor the
host pool from the container file, but a routed expert is then read a layer at a time and a step is
paced by the drive. At four bits an expert unit is 2.42 MiB instead of 4.69 and the per-rank share
fits VRAM plus a few GiB of pinned host. `docs/MOE-W4.md` is the format.

**PLE / the n-gram embedding is 51B of the model's 180B parameters** — a 320,001,536-row table of
160-dim embeddings, gathered 16 rows a token by a hash of the last three token ids, read once at
layer 1 and injected into the wide stream. Without it this is a 125B model missing a third of its
parameters.

**QSA** picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens for each
query and attends to those alone. Below `indexer_budget + indexer_compress_ratio - 1` = 2051 tokens
of context every complete block is selected, so QSA is dense causal attention bit for bit — which is
why a prefill below the bound runs the cheaper dense attention, and is the property any change to it
is argued against. `docs/QSA.md` is the decomposition.

The vision tower is declared when the container carries one and `--mm-max-patches` is not 0
(`arch/common/rad_block_vit.h`, `spec.md` §11).

---

# Two rules about measuring these

**A bisect knob is not a deployment knob.** Several switches here ablate a block or a layer range;
they produce a DIFFERENT MODEL and exist so that a wrong distribution can be attributed to a block
when nothing compares a whole layer against an oracle. A serve that sets one is a serve that is not
running the model, and the plugin says so at declare time. **Knobs are documented where they are
read** — a table of them in a document goes stale silently and ends up naming variables nothing
reads.

**THE SAMPLE SIZE IS PART OF THE RESULT.** Top-1 agreement between two numeric formats is one
Bernoulli trial per prompt, so at p ≈ 0.93 its standard error is 2.8 points at N = 80 and 2.0 at
N = 160 — wide enough that formats which differ by less than that rank in any order. Rank with a
graded statistic that has thousands of samples rather than hundreds (the mean |delta| over the
logits row, or the per-position KL of `radiance --kld-ref`), and confirm any ordering at N ≥ 160
before acting on it.
