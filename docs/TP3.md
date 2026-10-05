# Any world size: the attention world, uneven delta-net heads, and nothing padded

Radiance shards by dividing: `n_head`, `n_head_kv`, the delta-net head counts and the vocabulary
each split across the world, and a count that does not divide is **refused at declare, by name**
(`share.cpp`, `rad_arch.h`). Four ranks got through for Qwen3.8-Flash-Next by replicating the two
KV heads in rank pairs (1.0.7); three did not: 24 query heads, 2 KV heads, 16 GDN K heads and 48
GDN V heads have no even share in 3, and replication needs a world that is a multiple of the head
count. 1.0.12 served three ranks by **padding the checkpoint** offline (`tools/pad_tp.py`, the
regime of tcclaviger's `vllm/config/tp_padding.py`): zero heads appended so that 36 / 3 / 18 / 54
divide, and a container converted from that. It worked, exactly, and it cost a second 115 GiB
container for every model and 1/3 of the attention and 2 of 18 delta-net heads computing zeros.

This document describes what replaced it: **the shipped container serves at any world size**, with
no tool and no second file. Two rules, both built from machinery the engine already had.

## 1. The attention world

`rad_arch.h` `geom_from` already handled a world *above* the KV head count by replication, when
the world is a multiple of the count. It now handles a world that is **neither a divisor nor a
multiple** of it:

> attention is served on the largest multiple of the KV head count under the world -- the
> **attention world** -- and the ranks past it are **attention-zero**.

Three ranks over two KV heads: the attention world is 2. Ranks 0 and 1 split the heads exactly as
the two-rank serve does -- 12 query heads, one KV head, gqa 12, the same kernels -- and rank 2
declares **the same shapes** and reads **nothing** for them: `rad_weight_shard_zero` tells the
loader that this rank's share of `attn_qg`, `attn_k`, `attn_v` and `attn_output` is zeros. Its
attention computes zeros from zeros and its output projection adds exactly nothing to the reduce.
Five ranks over two: attention world 4 (two replicated pairs), rank 4 attention-zero. Six: three
pairs, no zero rank.

Why the same shapes rather than no attention on that rank: the scheduler sees **one program**
(`core/sched/batch.cpp` indexes `kv_groups[i]` by position), so every rank declares every op and
every KV group in the same order. An attention-zero rank's KV group is as large as its peers'; the
cost is one KV head of cache on that rank, the same bytes the padded container spent on its dummy
head, and nothing is read from disk for its projections.

Under a smaller attention world the plain even shard over `world` is the wrong cut for every
attention projection, so `rad_block_attn_gated_fp8.h` declares **spans** (`rad_weight_shard_span`,
the same call the KV replication already used for k/v): q|gate rows and output columns
`[attn_rank * n_head, +n_head)` heads, k/v rows at `kv_off`. When the attention world is the world
-- every TP2 and TP4 deployment -- no span is declared and the block declares as it always did.

## 2. Uneven delta-net heads

The delta net has no KV-head constraint: its kernels are bound on head *widths* (`head_k`,
`head_v`, the chunk) and take any head count. So 16 K heads over three ranks are **6, 5 and 5**,
each rank's V heads the same multiple of its K heads (48 / 16 = 3: 18, 15, 15), every rank real
heads, nothing padded. `qwen4exp` computes the split (`base + (rank < rem)`) and
`rad_block_gdn_fp8.h` declares it as spans of the container's rows.

That needed one thing the span call could not say. `in_proj_qkv|z` is four projections stacked
along dim 0 -- q 2048, k 2048, v 6144, z 6144 rows -- and `rad_weight_shard_span` takes one
`[lo, hi)` applied to every stacked part, which assumes the parts are equal.
**`rad_weight_shard_span_parts`** (an additive export) takes a span per part together with each
part's whole extent in the container; `share.cpp` cuts every part by its own rows. The conv
channels `[q|k|v]`, the `[a|b]` projections, `A_log`, `dt_bias` and the output projection's
columns take the same rank's heads. The even split, when the counts divide, is untouched.

## 3. What this is built from

- `rad_weight_shard_zero(b, w)` -- this rank's share is zeros; `weight_share` returns the
  declaration's extents with no rectangles, the loader stages the weight with a worker that writes
  zeros (the route a checkpoint weight takes) and reads nothing.
- `rad_weight_shard_span_parts(b, w, lo, hi, total, n)` -- a span per stacked part.
- `Geom::attn_world / attn_rank / attn_zero`, set by `geom_from`; `GdnFP8::Config::k_first /
  v_first`, set by the plugin.
- Both exports are additive: `RAD_ABI_VERSION` does not move, and a plugin that does not call them
  declares as before. `tests/share_test.cpp` holds the per-part cut and the zero share.

## 4. Serving

```sh
radiance --model qwen3.8-next-flash-fp8-iq4r-moe.rad --tp 3 --tp-wire exact --max-num-seqs 8 \
    --max-model-len 262144 --placement expert_tiered --host-pool-mib 8192 \
    --gpu-headroom-mib 96 --expert-vs-cache-ratio 0.82 --kv-cache-dtype fp8 \
    --num-speculative-tokens 3 --max-num-batched-tokens 2048
```

The log says what the geometry did:

```
radiance: world_size = 3 over n_head_kv = 2: attention on 2 rank(s), rank(s) 2..2 carry zero attention weights and the delta net and experts of a full rank
radiance: qwen4exp: rank 0 takes delta-net K heads [0, 6) of 16 and V heads [0, 18) of 48
radiance: qwen4exp: rank 1 takes delta-net K heads [6, 11) of 16 and V heads [18, 33) of 48
radiance: qwen4exp: rank 2 takes delta-net K heads [11, 16) of 16 and V heads [33, 48) of 48
```

Measured on three Radeon AI PRO R9700 (gfx1201, PCIe Gen5 x16, Threadripper PRO 9955WX) with the
same container served at two and three ranks -- the one comparison where only the world changes:

| | two ranks | three ranks |
|---|---|---|
| READY from cold | | |
| 1 / 4 / 8 streams, aggregate t/s | | |
| 4 streams at 100 K context, per stream | | |
| HumanEval / GSM8K | | |

The three-rank text is the three-rank text of the padded container, token for token, at temperature
0 on the prompt set -- the two computations are the same arithmetic with the zeros in a different
place.

`tools/pad_tp.py` and `tools/pad_verify.py` remain in the tree: they are how a checkpoint is padded
for an engine that has no attention world, and how a pad is verified, and nothing here needs them.
