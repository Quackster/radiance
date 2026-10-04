# Three ranks: padding a checkpoint to a world its geometry does not divide

Radiance shards by dividing: `n_head`, `n_head_kv`, the GDN head counts and the vocabulary each
split evenly across the world, and a count that does not divide is **refused at declare, by name**
(`share.cpp`, `rad_arch.h`). Four ranks got through for Qwen3.8-Flash-Next by replicating the two
KV heads in rank pairs (1.0.7). Three ranks cannot: 24 query heads, 2 KV heads, 16 GDN K heads and
48 GDN V heads have no even share in 3, and replication needs a world that is a multiple of the
head count. The engine is right to refuse. What this document describes is how to **change the
checkpoint** so there is nothing to refuse, exactly, and what the engine gained so that the padded
checkpoint then serves at three ranks.

The regime is tcclaviger's `vllm/config/tp_padding.py`, which pads the same model to the same
shape at load time inside vLLM. Radiance has no load-time pad -- a container is served as it was
converted -- so the pad is done once, offline, before `rad-convert`.

## 1. What is padded, and why it is exact

Every padded weight gains **zeros**, placed where a zero contributes nothing to the forward pass:

| block | tensor | gains | exact because |
|---|---|---|---|
| attention | `k_proj`, `v_proj` | zero **rows** (new KV heads) | a zero KV head stores zeros and is read only by zero query heads |
| attention | `q_proj` (fused q+gate on Flash-Next: 512-row head blocks) | zero rows so every KV group has the new ratio | a zero query head attends uniformly, and its `o_proj` column is zero, so its output is 0 |
| attention | `o_proj` | zero **columns** in the same head order | the zero heads' outputs are multiplied by zero |
| GDN | `in_proj_qkvz`, `in_proj_ba` | zero rows per the fused layout (new K and V heads) | a zero head carries a zero state |
| GDN | `conv1d` | zero channels | zero input, zero output |
| GDN | `A_log`, `dt_bias` | zero heads | a zero head decays nothing |
| GDN | `out_proj` | zero columns | the zero heads are multiplied by zero |

Norms, embeddings, the lm_head, the router and the experts are untouched: their only width is
`n_embd`. The experts stay at the model's own 640 and the plugin slices them by parity across the
ranks (1.0.8: at three ranks the five 128-wide blocks go 2, 2, 1), so the +20 % expert bytes of a
640 → 768 pad are not paid.

For Qwen3.8-Flash-Next the pad is **24 → 36 query heads, 2 → 3 KV heads, GDN 16 → 18 K heads and
48 → 54 V heads** -- head_dim 256 for attention and 128 for GDN are unchanged, so every rank at
three declares 12 query heads, 1 KV head and gqa 12: **the same per-rank attention geometry the
two-rank serve declares**, and the same kernels resolve. Dummies go **last**, as the vLLM pad does,
so rank 2 carries them.

The cost is the dummies' arithmetic: 12 of 36 query heads and 2 of 18 GDN heads are zeros on the
last rank, so three ranks do 1.5× the attention work of two for 1.5× the cards.

## 2. The tool

`tools/pad_tp.py` (Python, numpy only; no torch) rewrites a Hugging Face safetensors checkpoint:

```sh
tools/pad_tp.py SRC DST --heads 36 --kv-heads 3 --gdn-k 18 --gdn-v 54 --sparse
tools/pad_verify.py SRC DST
```

It streams one tensor at a time (a 336 GiB checkpoint pads in a few GB of RAM), reads bf16 as raw
u16 (no arithmetic happens), rewrites `config.json` with the new counts and an explicit `head_dim`,
and handles the fused layouts: the Flash-Next `q_proj` that carries the attention gate beside each
query head, and the GDN `in_proj_qkvz` / `in_proj_ba` row order.

`--sparse` is for checkpoints larger than the free disk. Instead of copying, it **reflinks** every
original shard (btrfs, XFS; `cp --reflink`) and writes only the padded tensors into new shards named
`model-00000-of-NNNNN-tp3pad-*.safetensors`. Radiance's reader lists the index's shards sorted by
name and keeps the **first** tensor of a name (`checkpoint.cpp`, `emplace`), so the new shards
shadow the originals by sorting first. Flash-Next: 340 tensors, 6.6 GB of new bytes, the 131
original shards shared.

`tools/pad_verify.py` reads both checkpoints back and checks every tensor: an untouched tensor is
byte-identical, a padded one is the source in its place and zeros everywhere else. Run it before
`rad-convert`; a pad that passes it cannot change what the model computes.

Padding proved on Llama-3.2-3B (24/8 heads, gqa 3 → 36/9, gqa 4): a 28-layer numpy forward on the
padded weights gives the same argmax as the original with max |Δlogit| 2.1e-5 (fp32 summation
order), and on the engine TP1 on the padded checkpoint produces the same text as TP1 on the
original at temperature 0.

## 3. What the engine gained for three ranks

- **libr4d: `all_reduce` and `all_gather` at world 3.** `ar_oneshot_3rank_exact` is the wide
  one-shot family's template at 3; `ar_gather_3rank` the same for the gather the vocab-parallel
  sampler declares. Both share the wide family's instance (scratch, `[block][source]` flags, seq
  counters). There is no two-shot at 3: it shards the message `world` ways and a residual width
  such as 2560 does not divide by 3. `wht6` is a two-rank wire; three ranks take the exact one, as
  four do. Selftest: `--case ar_oneshot_3rank_exact` and `--case ar_gather_3rank`, bit-exact against
  the host sum / concatenation on three gfx1201 cards, with the two-rank controls unchanged.
- **The vocabulary at a world that does not divide it.** 248 320 rows do not divide by 3 and
  `share.cpp` refuses a row shard without a declared span. `qwen4exp` now declares the rank's span
  for `token_embd` and `output`, and the split rule (`rad_arch.h` `shard_extent` / `shard_begin`,
  restated in `core/engine.cpp`) keeps the even share whenever the world divides the vocabulary and
  otherwise rounds the per-rank count up to a **128-row block** -- a block-scaled fp8 head refuses a
  cut inside a tile. At three the ranks take 82 816, 82 816 and 82 688; two and four ranks divide
  248 320 evenly and are untouched.
- Everything else three ranks need was already there by 1.0.8: the width split of the experts at
  any world up to the block count, the uneven tile share of the plain bf16 (protected) experts, the
  N-rank gather's placement.

## 4. Serving

```sh
tools/pad_tp.py ~/models/Qwen3.8-Flash-Next ~/models/Qwen3.8-Flash-Next-tp3pad \
    --heads 36 --kv-heads 3 --gdn-k 18 --gdn-v 54 --sparse
tools/pad_verify.py ~/models/Qwen3.8-Flash-Next ~/models/Qwen3.8-Flash-Next-tp3pad
build/bin/rad-convert ~/models/Qwen3.8-Flash-Next-tp3pad -o qfn-tp3pad.rad \
    --recipe data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe
radiance --model qfn-tp3pad.rad --tp 3 --tp-wire exact --max-num-seqs 8 --max-model-len 200000 \
    --placement expert_tiered --host-pool-mib 8192 --gpu-headroom-mib 96 \
    --expert-vs-cache-ratio 0.82 --kv-cache-dtype fp8 --num-speculative-tokens 3 \
    --max-num-batched-tokens 2048
```

Measured on three Radeon AI PRO R9700 (gfx1201, PCIe Gen5 x16, Threadripper PRO 9955WX), the
uncensored Flash-Next fine-tune padded as above and converted with the shipped recipe
(`qwen4exp-w4nl64-i8-hc8m`, here without Hessians: nearest into the codebook), this branch over
1.0.8; the two-rank column is the shipped `qwen3.8-next-flash-fp8-iq4r-moe.rad` on the same box:

| | three ranks, padded | two ranks, shipped container |
|---|---|---|
| READY from cold | 45 s | 35 s |
| 1 / 4 / 8 streams, aggregate t/s | 173.5 / 332.2 / 410.9 | 156 / 320 / 404 |
| 4 streams at 100 K context, per stream | 41.8 | 41.2 |
| warm-prefix TTFT, 60 K prompt, second time | 0.3 s (cold 8.2 s) | — |
| HumanEval / GSM8K | 0.854 / 0.915 | 0.854 / 0.940 |

The GSM8K gap is the Hessians the shipped container was quantised against and this one was not;
the GPTQ'd conversion of the same padded checkpoint is the comparison that closes it. Hi-vocab
tokens (CJK, Cyrillic -- rank 2's lm_head shard, the three-rank gather) decode correctly; no engine
warnings; every band the run issued resolved to a device kernel.
