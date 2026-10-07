# Static YaRN for Qwen architectures

Increasing `--max-model-len` reserves a larger serving window; it does not change
the model's rotary embeddings. A Qwen checkpoint trained at 262144 positions can
explicitly request static YaRN for a 524288-token deployment. Whether that extension
preserves useful accuracy must be measured on the model and workload.

The Qwen architectures read the target's `text_config.rope_parameters`,
then `rope_parameters`, then their legacy `rope_scaling` spellings, in that order.
The first object with `rope_type` (or legacy `type`) is authoritative. The DFlash2
drafter reads its own `draft.rope_parameters` / `draft.rope_scaling`; it does not
inherit the target's scaling. The vision tower keeps its original axial rotations.

## Coverage

| Architecture | Rotary paths | Validation so far |
| --- | --- | --- |
| `qwen35_fp8` | Target direct/table; independent DFlash2 | 27B FP8, uncensored FP8 and MXFP4: text/tool/image checks, 499997-token retrieval and two overlapping 255981-token requests, TP4/BF16 KV |
| `qwen35_bf16` | Target direct rotary | Architecture declaration and shared numerical tests; no end-to-end BF16 model run yet |
| `qwen35moe_fp8` | Shared Qwen35 target and MTP geometry | 35B-A3B stock/YaRN smoke and retrieval, including 499997 tokens and paired 255981-token requests; TP2/BF16 KV |
| `qwen4exp_fp8` | Attention, indexer, compressed block keys and shared MTP geometry | Both Flash variants: stock/YaRN smoke and retrieval, including 499997 tokens and paired 255981-token requests; TP4, tiered experts, BF16 KV |

Radiance 1.2.0 also serves the published Qwen3.8-27B MXFP4 container through the
Qwen35 FP8 architecture's encoding dispatch. It passed the model checks above.
Request logs prove overlapping request lifetimes, not continuous simultaneous GPU
residency or a common decode batch. See [qualification and audit](YARN-VALIDATION.md)
for exact settings, timings, evidence boundaries and outstanding review findings.
Synthetic retrieval is evidence for those requests, not a general accuracy benchmark
or a capacity guarantee for other model sizes. Original-context quality should be
checked alongside extended-context quality when deploying static YaRN.

## Metadata

| Key within the rotary object | Meaning |
| --- | --- |
| `rope_type` | `yarn`; default/absent preserves stock behavior; `linear` divides positions uniformly |
| `factor` | finite number >= 1, required for YaRN |
| `original_max_position_embeddings` | positive integer, required; the **training** length, not the enlarged deployment/table length |
| `beta_fast`, `beta_slow` | defaults 32 and 1; finite, fast >= slow > 0 |
| `attention_factor` | positive finite cos/sin multiplier; default 1 + 0.1 ln(factor) for factor > 1, otherwise 1 |
| `truncate` | 1 (default): floor/ceil correction bounds; 0: retain fractional bounds |

`mscale` and `mscale_all_dim` variants are refused; specify `attention_factor`
explicitly. Other explicit rotary types are refused in this architecture. Rotary
width (including partial rotary), theta and M-RoPE section selection are unchanged.

## Convert a separate container

Use the original target and drafter checkpoints, the same quantization recipe,
and a **new output path**. `--reuse` is valid only for the same source weights and
recipe; it is not a way to mix weights from different model revisions.

```sh
rad-convert /checkpoints/Qwen3.8-27B-FP8 \
  --draft-model /checkpoints/Qwen3.8-27B-DFlash2 \
  --recipe data/recipes/q38-27b-fp8-df2.recipe \
  --reuse /models/qwen3.8-27b-fp8.rad \
  --set text_config.rope_parameters.rope_type=yarn \
  --set text_config.rope_parameters.factor=2 \
  --set text_config.rope_parameters.original_max_position_embeddings=262144 \
  --set draft.rope_parameters.rope_type=yarn \
  --set draft.rope_parameters.factor=2 \
  --set draft.rope_parameters.original_max_position_embeddings=262144 \
  --max-ctx 524288 --kv-cache-dtype bf16 \
  -o /models/qwen3.8-27b-fp8-yarn2.rad
```

Run with `--plan-only` first. `--max-ctx` is the converter's graph bound, not a
replacement for the explicit YaRN metadata or the server's `--max-model-len`.
The bundled `.rad` by itself is not a converter input; keep the source checkpoints.
Check both target and draft settings with `rad-info --meta` after conversion.

Serve on enough memory for the requested window:

```sh
radiance --model /models/qwen3.8-27b-fp8-yarn2.rad \
  --tp 4 --max-model-len 524288 --max-num-seqs 32 --kv-cache-dtype bf16
```

That cache flag controls **both** target and drafter KV. Startup capacity and
successful actual long requests must be checked; the command is not a promise
of concurrency or accuracy. Keep the unmodified model and serving setup until
long-context quality and speed meet the deployment's requirements.

## Kernel and test contract

`rope` and `rope_table` take optional resolved coefficients `yarn_factor`,
`yarn_low`, `yarn_high`, `yarn_magnitude`. Absent factor is the original path.
With YaRN, `scale` must be 1: inverse frequencies are blended rather than all
positions being divided. Axial rotary with YaRN is refused. Rebuild the bundled
plugins together because their shared op schemas must match.

The math follows the static YaRN formulation (Peng et al.) and is checked against
independent vLLM `YaRNScalingRotaryEmbedding` FP32 vectors for each backend. The fixture generator
is `tests/data/generate_yarn_reference.py`; it is not a runtime dependency.
Vectors cover target rotary width 64, draft width 128, an additional width 256,
and positions through 524287.
The CPU fixture uses torch CPU; the HIP fixture uses torch ROCm on gfx1201
(`generate_yarn_reference.py cuda`). CPU and ROCm `pow` differ by a few ULPs;
at 524K positions their cos/sin values can differ by 0.033 even within torch.
Compare each implementation against its backend's independent vectors.
The numerical check allows 0.004 absolute cos/sin error; fused versus unfused HIP rotation is checked
byte-for-byte separately. Existing stock rotary tests remain active.
