# Any world size: the attention world, uneven delta-net heads, and nothing padded

Radiance shards by dividing: `n_head`, `n_head_kv`, the delta-net head counts and the vocabulary
each split across the world, and a count that does not divide is **refused at declare, by name**
(`share.cpp`, `rad_arch.h`). Four ranks get through for Qwen3.8-Flash-Next by replicating the two
KV heads in rank pairs; three did not: 24 query heads, 2 KV heads, 16 GDN K heads and 48 GDN V
heads have no even share in 3, and replication needs a world that is a multiple of the head count.
Padding the checkpoint offline (`tools/pad_tp.py`: zero heads appended so that 36 / 3 / 18 / 54
divide) served three ranks exactly, at the cost of a second 115 GiB container for every model and
a third of the attention and 2 of 18 delta-net heads computing zeros.

This document describes what replaced it: **the shipped container serves at any world size**, with
no tool and no second file. Two rules, both built from machinery the engine already had.

## 1. The attention world

`rad_arch.h` `geom_from` handles a world *above* the KV head count by replication, when the world
is a multiple of the count. For a plugin that asks for it (`attn_world_ok`; qwen4exp does) it also
handles a world that is **neither a divisor nor a multiple** of it:

> attention is served on the largest multiple of the KV head count under the world -- the
> **attention world** -- and the ranks past it are **attention-zero**.

Three ranks over two KV heads: the attention world is 2. Ranks 0 and 1 split the heads exactly as
the two-rank serve does -- 12 query heads, one KV head, gqa 12, the same kernels -- and rank 2
**computes no attention**: `rad_block_attn_gated_fp8.h` declares no projections, no attention and
no cache store on it, and writes the block's output wire to zero with one `fill` (`docs/OPS.md`).
The all-reduce and the residual after it see what they always saw, and the zero adds exactly
nothing to the reduce. Five ranks over two: attention world 4 (two replicated pairs), rank 4
attention-zero. Six: three pairs, no zero rank. A plugin that does not ask is refused at a world
like these, by name, as before.

**The KV groups stay in the same order on every rank.** The scheduler sees one program
(`core/sched/batch.cpp` indexes `kv_groups[i]` by position) and one block table serves every rank,
so an attention-zero rank declares its attention group like its peers and marks it
`rad_kv_group_zero`: it carves nothing for it, holds the shared table's block count, lends none of
it, and the tiers skip it when they move an entry. Every rank carves the same block count -- each
is planned, asked what it could page (`KVManager::paged_tokens`; a rank whose only paged groups are
zero limits nobody) and capped at the least before any carves.

**The attention-zero rank's pools are its own.** The budget resolver derives one split for every
card, the dearest rank's -- and an attention-zero rank's cache costs a fraction of its peers' per
token (at TP3 on Flash-Next it keeps only the QSA block keys: 2.3 GiB at the worst case where an
attention rank's costs 15.2). Under `expert_tiered` with both pools derived, `Engine::plan` gives
such a rank a KV pool that pages exactly the common token count (`KVManager::pool_mib_for_tokens`,
the inverse of `paged_tokens`: the least whole MiB whose paged split reaches the count, exact
because the split is monotone in the budget and its alignment is among the fixed costs) and moves the
difference into its weights pool, no further than every one of its weights resident. Both readers
of the resolved sizes take the rank's own split -- `Pools::configure` and the planner -- so the
grown pool is planned into, and startup refuses a rank whose plan and weights pool disagree, or
whose re-sized KV pool pages fewer tokens than it was sized for. The token count, the carve, the
loan and the attention ranks' pools are unchanged, and `auto` keeps the common split (a fully
resident rank would choose all_vram while its peers spill). `--deterministic` and a device without virtual memory
management re-split like any other: the cache does not lend there, and the bytes the rank never
pages are as much its experts' as they are under a lending cache. The condition is the cost, not
the world: a rank is re-split only where its cache costs less per token than the dearest rank's,
which at TP2/TP4 no rank does. A rank that qualifies and keeps the common split names the reason on
its `kv: rank N's cache costs ...` line. The pinned host pool is not re-split: every rank allocates
its whole `--host-pool-mib`, and a re-split rank whose weights all fit on the card says so
(`its pinned host pool holds no weights`). The log line, and the mover on the third rank:

```
kv: rank 2's cache costs 2.27 GiB at the worst case where its dearest peer's costs 15.16 GiB, so its KV pool pages the common 1446302 tokens in 1811 MiB instead of 10881; its weights pool grows 19747 -> 21692 MiB (every weight resident wants 21.18 GiB), and 7125 MiB stay unclaimed
I[2] mover: 25142 slab slots (64 shadow), 0 pinned pool slots
```

The settle check's INFO line on that card names the unclaimed MiB and points at this line rather
than at the budget's `unusable` line alone.

Under a smaller attention world the plain even shard over `world` is the wrong cut for every
attention projection, so the attention ranks declare **spans** (`rad_weight_shard_span`, the same
call the KV replication uses for k/v): q|gate rows and output columns `[attn_rank * n_head,
+n_head)` heads, k/v rows at `kv_off`. When the attention world is the world -- every TP2 and TP4
deployment -- no span is declared and the block declares as it always did.

## 2. Uneven delta-net heads

The delta net has no KV-head constraint: its kernels are bound on head *widths* (`head_k`,
`head_v`, the chunk) and take any head count. So 16 K heads over three ranks are **5, 5 and 6**,
each rank's V heads the same multiple of its K heads (48 / 16 = 3: 15, 15, 18), every rank real
heads, nothing padded. The extra head goes to the rank without attention (`remainder_place` in
`rad_arch.h`): it has the time, and the rank that would otherwise hold the extra head AND attention
is the step's slowest. The experts' extra block stays on the parity rotation (`qwen4exp_fp8.cpp`
says why). `qwen4exp` computes the split and `rad_block_gdn_fp8.h` declares it as spans of the
container's rows. Each rank's recurrent state is its own size, so a fresh slot is cleared, a
snapshot taken and a tier entry laid out by each rank's own plan: a slot of the snapshot store or
the tiers is the sum of the ranks' shares, each rank's where the ones before it end.

That needed one thing the span call could not say. `in_proj_qkv|z` is four projections stacked
along dim 0 -- q 2048, k 2048, v 6144, z 6144 rows -- and `rad_weight_shard_span` takes one
`[lo, hi)` applied to every stacked part, which assumes the parts are equal.
**`rad_weight_shard_span_parts`** (an additive export) takes a span per part together with each
part's whole extent in the container; `share.cpp` cuts every part by its own rows. The conv
channels `[q|k|v]`, the `[a|b]` projections, `A_log`, `dt_bias` and the output projection's
columns take the same rank's heads. The even split, when the counts divide, is untouched.

## 3. What this is built from

- `rad_weight_shard_span_parts(b, w, lo, hi, total, n)` -- a span per stacked part.
- `rad_kv_group_zero(b, g)` -- this rank keeps no cache for a paged group.
- the `fill` op -- libr4d's `fill_bf16`, libref's `ref_fill`.
- `Geom::attn_world / attn_rank / attn_zero`, set by `geom_from` under `attn_world_ok`;
  `GdnFP8::Config::k_first / v_first`, set by the plugin.
- The exports are additive: `RAD_ABI_VERSION` does not move, and a plugin that does not call them
  declares as before. `tests/share_test.cpp` holds the per-part cut, `tests/arch_test.cpp` the
  three-rank declaration, `tests/mem_test.cpp` the capped carve, the zero group and a tier entry
  moved across ranks with unequal shares.

**A load check that came with it.** The relayout pool refuses to finish when any staged weight was
never delivered: a slot nothing wrote holds whatever the card had there -- zeros on a card whose
freed memory the driver wipes, garbage after a reset ("VRAM is lost"). The instrument that shows
such a read on any card:

    RADIANCE_DEBUG_POISON_ALLOC=255 radiance --model stock.rad --tp 3 ...

fills every fresh device allocation with 0xFF (NaN in every float width) before it is handed out;
the serve must answer exactly as without it. `RADIANCE_DEBUG_POISON_RANGE=lo:hi` narrows the fill
to a thread's allocation ordinals for a bisect (`core/device/dispatch.cpp`).

## 4. Serving

```sh
radiance --model qwen3.8-next-flash-fp8-iq4r-moe.rad --tp 3 --tp-wire exact --max-num-seqs 8 \
    --max-model-len 262144 --placement expert_tiered --host-pool-mib 8192 \
    --gpu-headroom-mib 96 --kv-cache-dtype fp8 \
    --num-speculative-tokens 3 --max-num-batched-tokens 2048
```

The exact all-reduce at three ranks is the wide family's one-shot up to 1 MiB a message (every
decode step) and its two-shot above it (every prefill chunk): reduce-scatter + all-gather, which
sends 4/3 of the message per rank instead of twice it. A message rarely divides into thirds -- a
2048-wide bf16 row is 256 16-byte words -- so the first `words % 3` shards take one word more.
Both rows fold in fp32 in ascending rank order, so the band edge between them changes no bits.
`--tp-wire wht6` at three ranks is the four-rank tiered-int8 wire over the same uneven two-shot,
above `--tp-wire-min-kb`; under it, the exact one-shot.

The log says what the geometry did:

```
radiance: world_size = 3 over n_head_kv = 2: attention on 2 rank(s), rank(s) 2..2 carry no attention and take the delta net's remainder first
radiance: qwen4exp: rank 0 takes delta-net K heads [6, 11) of 16 and V heads [18, 33) of 48
radiance: qwen4exp: rank 1 takes delta-net K heads [11, 16) of 16 and V heads [33, 48) of 48
radiance: qwen4exp: rank 2 takes delta-net K heads [0, 6) of 16 and V heads [0, 18) of 48
```

Measured by the contributor on three Radeon AI PRO R9700 (gfx1201, PCIe Gen5 x16) at an earlier
state of this design -- the attention-zero rank then ran its projections on zero weights, and the
extra delta-net head sat on rank 0 -- with the uncensored Flash-Next fine-tune. Hot = 1 / 4 / 8
streams, 8 K prompt, 1 K generated, aggregate tok/s; crowd = 4 streams at 100 K context, per
stream; HumanEval 164 and a 200-question GSM8K subset at temperature 0:

| container | `--tp` | hot c1 / c4 / c8 | crowd | HE | GSM8K |
|---|---|---|---|---|---|
| unpadded | 3 | 191 / 309 / 353 | 42.7 | 0.878 | 0.925 |
| unpadded, the same file | 2 | 177 / 308 / 377 | 33.8 | 0.884 | 0.915 |
| padded offline | 3 | 182 / 288 / 395 | 43.5 | 0.848 | 0.935 |

Three ranks buy the single stream and the long-context crowd, not the eight-stream aggregate. The
quality rows are within a rerun of each other (HumanEval moves by about 3 of 164 between runs of one
container); unpadded and padded at three ranks gave the same answers on 40 prompts at temperature
0, bit-identical on one -- the head regroup changes the order the all-reduce sums in, and near-ties
flip.

`tools/pad_tp.py` and `tools/pad_verify.py` remain in the tree: they are how a checkpoint is padded
for an engine that has no attention world, and how a pad is verified, and nothing here needs them.
