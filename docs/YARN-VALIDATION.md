# YaRN qualification and final-diff audit

## Decision

The six published Qwen containers passed the recorded functional checks. The host
regression suite passed 41/41 tests. Architecture routing is covered by source review,
declaration tests and model runs as distinguished below.

**Both compatibility findings are fixed and their focused regressions pass.**
The original findings and remedies are in section 5. The model runs predate these
fixes and do not exercise those edge cases.
Concurrency claims are limited to overlapping server request lifetimes; continuous
dual-context GPU residency has not been established by these logs.

Audit date: 2026-10-06. Reviewed implementation: `1d42304`, against released upstream
`e95d397` (Radiance 1.2.0), including `3d7f066` and merge `10721e6`.
No production code was changed during the initial audit. The subsequent compatibility
fixes change argument normalization and linear-factor validation as detailed below.
At the initial audit, all 24 changed non-YARN-doc files
matched the remote source/fixtures by SHA-256; the remote `docs/YARN.md` was older.
`git diff --check upstream/main...HEAD` passed. This is a direct source review,
not an independent second-reviewer sign-off.

## 1. Architecture audit

| Architecture / component | Route traced | Evidence and boundary |
| --- | --- | --- |
| `qwen35_fp8` | `yarn_from` resolves target coefficients; `decl_rope` handles Q/K; `decl_rotary` handles shared `rope_table`; fused prologues consume that table | Three 27B encodings passed actual model checks; direct/table and fused/unfused numerical checks |
| DFlash2 | Own `Geom`, theta and `draft.*` metadata; context K, draft Q and draft K all use `decl_rotary` | Independent target/draft parser checks; 128-wide numerical checks; model logs record DFlash2 passes |
| `qwen35_bf16` | Target metadata resolved after partial rotary width; gated attention uses `decl_rope` | Dedicated declaration test and shared kernels; no BF16-weight end-to-end model run |
| `qwen35moe_fp8` | Includes the Qwen35 FP8 implementation with routed-FFN macros | 35B-A3B model passed at TP2; TP4 stock fails unsupported BF16 attention geometry |
| Qwen35 MTP | `MtpBlockFP8::declare` copies target `Geom` and passes it to attention | Source trace plus actual 35B MTP startup and recorded draft passes; no isolated YaRN/MTP declaration regression |
| `qwen4exp_fp8` | Target direct/table; QSA query indexer and compressed keys receive target coefficients; indexer rotary width equals target `m.n_rot` | Dedicated declaration test, compressed-key GPU oracle and both Flash model runs |
| Qwen4 MTP | `MtpHcBlock` copies target `Geom`; attention and QSA reuse it | Declaration fixture requests speculation; actual Flash logs record MTP passes |
| Vision tower | Existing axial rotations do not receive target YaRN parameters | Source trace and red-image smoke checks; not a broad multimodal-quality evaluation |
| Llama plugins | Do not call `yarn_from`; shared rotary ops retain absent-parameter defaults | Outside architecture-level YaRN support; no new Llama support claimed |

Source anchors: `arch/qwen35_fp8/qwen35_fp8.cpp:343,639,957`,
`arch/qwen35moe_fp8/qwen35moe_fp8.cpp:20-24`,
`arch/qwen35_bf16/qwen35_bf16.cpp:240`,
`arch/qwen4exp_fp8/qwen4exp_fp8.cpp:367,700,1004,1397`,
`arch/common/rad_block_dflash2.h:541,715,723`,
`arch/common/rad_block_mtp_fp8.h:219,290`,
`arch/common/rad_block_mtp_hc.h:203-212,291-294`,
`arch/common/rad_qsa.h:162,230,272-281`.

### Compatibility contract

- Absent/default rotary metadata leaves the existing positional scale and adds no
  YaRN op parameters. Default op identities therefore retain their parameter lists.
- Static YaRN uses the explicit original training length, not runtime/table length.
  Target and DFlash2 configuration are independent. MTP shares target geometry.
- Both modern `rope_parameters` and legacy `rope_scaling` are read. Precedence is
  nested modern, top-level modern, nested legacy, top-level legacy; the first type
  found wins. Conflicting layouts are not merged. This precedence is documented,
  but lacks a dedicated conflicting-layout regression.
- Unknown explicit rotary types now fail instead of being silently ignored.
  `mscale` / `mscale_all_dim` are intentionally unsupported. This is an intentional
  compatibility change for previously ignored metadata, not universal stock parity.
- YaRN plus axial mode or positional scale other than one is rejected by rotary
  kernels. Model-level YaRN requires a positive even rotary width and finite inputs.
- Checkpoint JSON booleans are flattened to `1`/`0`
  (`core/format/checkpoint.cpp:64`), matching `truncate`'s parser.
- Optional op-schema parameters are consistent across reference, AVX and HIP
  rotary implementations. Compressed-key support is in the existing HIP-only op.
  Rebuild plugins together; no mixed-old/new-plugin compatibility is claimed.
- Existing stock tests and stock checkpoints passed. No bitwise comparison of
  complete upstream-versus-branch stock-model logits or throughput was performed.

## 2. Model evidence

All extended probes used static factor 2, original context 262144, BF16 KV and
prefix caching disabled. DFlash2 copies received their own factor/original keys.
Every single probe used **499997 prompt tokens**; each paired probe used
**255981 prompt tokens**, with three planted values at roughly 5%, 50%, 95% depth.
All returned exactly the requested values and `finish_reason=stop`.

| Model (all under `StillDeadcode`) | TP | Single seconds | Pair A / B seconds | Run evidence |
| --- | ---: | ---: | ---: | --- |
| `qwen3.8-27b-fp8` | 4 | 309.4 | 283.6 / 283.6 | `v12-media-qualification.json`, `v12-media-engine-audit.log` |
| `qwen3.8-27b-uncensored-fp8` | 4 | 309.8 | 283.0 / 282.9 | `matrix-20261006-181056` |
| `qwen3.8-27b-mxfp4` | 4 | 301.9 | 274.4 / 274.4 | `matrix-20261006-181056` |
| `qwen3.6-35b-a3b-fp8` | 2 | 197.3 | 121.5 / 121.5 | `matrix-20261006-185632` |
| `qwen3.8-next-flash-fp8-iq4r-moe` | 4 | 174.5 | 80.9 / 163.3 | `matrix-20261006-183853` |
| `qwen3.8-next-flash-uncensored-fp8-iq4r-moe` | 4 | 141.2 | 133.5 / 67.2 | `matrix-20261006-183853` |

Matrix runs include stock and YaRN arithmetic, tool-envelope, red-image and
approximately 20K/180K retrieval checks. The original 27B baseline was collected
separately (`ember-baseline*`, `baseline-retrieval.json`), not in this matrix;
the release-1.2 media-enabled YaRN run has its own smoke results.
These are synthetic functional checks, not a broad accuracy, perplexity or speed
comparison. Timings are single observations, not comparative benchmark medians.

Hardware: four Radeon AI PRO R9700, gfx1201, approximately 31.86 GiB each;
121 GiB host RAM; Linux `7.2.3-070203-generic`. The 35B run uses TP2.
HIP/media build uses `g++-14`, `RAD_WITH_FFMPEG=ON`, AQL device backend.
Recorded serving executable SHA-256:
`0c99111b9514e5f30d17b98ce590a588a8cc857373cd7b61d170d03adf16e911`.

Matrix launch options (exact per-model argv lives in each `command.json`):

```sh
radiance --model MODEL.rad --host 0.0.0.0 --port 8000 \
  --served-model-name Ember --tp 4 --max-num-seqs 8 \
  --max-model-len 524288 --kv-cache-dtype bf16 \
  --gpu-headroom-mib 256 --no-prefix-cache --reasoning-effort medium
```

Stock uses `--max-model-len 262144`. 35B uses `--tp 2`. Both Flash models add
`--placement expert_tiered --host-pool-mib 24576 --max-num-batched-tokens 2048`.
The original 27B media trial used `--max-num-seqs 32`. Do not assume identical
capacity across those configurations. The retained container argv includes device
mapping, memlock, source/model mounts and plugin home.

The five new YaRN artifacts were copied to separate files, then metadata appended;
verification checked all 21855 weights for 35B, 1101 for each new 27B, and 51489
for each Flash variant. Tokenizer, template and expert profile were identical.
This tested metadata-copy workflow is distinct from the documented checkpoint
converter command; the full published-model matrix did not rerun that converter.

### Stock artifact SHA-256

| Model suffix | SHA-256 |
| --- | --- |
| 27B FP8 | `e8f662be2be37312fb07b82d1aa6c58c6507856425ce0f28028bdd9440400737` |
| 27B uncensored FP8 | `fdb034bd0639d8fcc1c6d97d3b23939286ac7a7d114d1d53501653dd802fffe2` |
| 27B MXFP4 | `44aec7fb78d9515907c6ae46e10e506a8578155013af227247f22f3c63c997b2` |
| 35B-A3B FP8 | `deb3f9844cc5419ca8f546497b09fa7380f933f13e8244aa2af505bdfe593f36` |
| Flash FP8 IQ4R | `0af5e96244e80c21ac8edca1719dae49b8a201b94a2f3994c3e24026ceaa4d20` |
| Flash uncensored FP8 IQ4R | `19276213c4905100555508cf9d9383ae393424985f06c99b15935eaea5471f60` |

Download revisions and sizes are retained in `models/qwen-download-manifest.json`.

## 3. Concurrency verification

`audit-evidence.py` matched each response ID to its server request-start and
completion lines, rechecked the exact answers, and asserted that both starts
preceded either completion. All six pairs passed. Both requests report cache=0.
The resulting `concurrency-audit.json` records request IDs, line numbers, exact log
excerpts and SHA-256 hashes of the input logs/results.

| Model | Pair start lines | Pair completion lines | Engine log |
| --- | --- | --- | --- |
| 27B uncensored | 137, 138 | 148, 149 | corresponding `181056/*-yarn/engine.log` |
| 27B MXFP4 | 127, 128 | 138, 139 | corresponding `181056/*-yarn/engine.log` |
| 35B | 107, 108 | 109, 110 | corresponding `185632/*-yarn/engine.log` |
| Flash | 161, 162 | 163, 164 | corresponding `183853/*-yarn/engine.log` |
| Flash uncensored | 161, 162 | 163, 164 | corresponding `183853/*-yarn/engine.log` |

Original 27B IDs and line numbers are also in the machine-readable audit; its log
contains several later server restarts, so it is retained separately.

This proves overlapping HTTP/server request lifetimes and successful paired
completion. It does **not** identify scheduler admission, per-step sequence IDs,
preemption or simultaneous KV occupancy. The Flash completion timing is compatible
with substantial serialization. Dense 27B and 35B timings suggest overlapping
progress, but timing alone is not a GPU-residency measurement. A stronger claim
requires scheduler/batch traces or timestamped active-sequence/KV telemetry.

## 4. Regression and numerical evidence

- `final-host.log`: 41/41 CTest tests passed, 1633.32 seconds total, including
  `yarn_test`, `arch_test`, `ref_test`, `avx_check`, `ops_doc`, out-of-tree plugins,
  `kbench` and `kbench_moe`. Host suite success is not a full GPU-suite run.
- `v12-gpu.log`: explicit GPU YaRN table comparisons at rotary widths 64/128,
  fused/unfused Q/K/gate byte equality, and compressed-key reference checks passed.
- CPU fixture cases cover widths 64/128/256 and positions through 524287.
  GPU numerical oracle coverage is 64/128; presence of a 256-wide fixture is not
  evidence of a 256-wide GPU run. Cos/sin tolerance is 0.004; compressed BF16 keys
  use 0.008 absolute error.
- AVX checks compare against libref; they are not a separate direct-vLLM oracle.
- Independent fixtures use vLLM frequency computation and backend-specific torch
  FP32 trigonometry. Exact generating vLLM/torch revisions are not embedded in the
  fixture; pin/record them before calling fixture regeneration fully reproducible.

Observed run limitations are preserved: initial 35B TP4 stock failure; obsolete
Flash launch flag on the first attempt; Flash per-card headroom warnings of roughly
157–165 MiB below its stated target; five-second device synchronization diagnostics
during some successful long probes. No claim of warning-free production qualification
is made. Stock success uses the feature branch, not a clean upstream binary.

## 5. Final-diff findings

### Resolved P2 — disabled coefficients diverged across reference and optimized backends

`abi/rad_yarn.h:40-42` accepts factor zero without validating/normalizing magnitude.
`libref/ref_attn.cpp:500,597-598` applies `yarn.magnitude` unconditionally, while AVX
and HIP apply it only when factor is nonzero. With `{factor=0, magnitude=2}`, libref
scales the rotation by two and optimized backends preserve the stock rotation.
NaN magnitude can similarly contaminate only reference output.

Architecture-generated stock ops omit all four fields, so the six model runs do
not hit this. It remains a public optional-op-parameter contract inconsistency.
The fix normalizes disabled coefficients to stock defaults in `rad_yarn_args`, shared
by all three backends. Reference direct/table tests cover magnitude 2 and NaN with
disabled factor and NaN bounds; AVX-versus-reference cases cover magnitude 2 and NaN.

### Resolved P2 — linear factor validation happened before narrowing to float

`arch/common/rad_arch.h:1068-1072` accepts positive finite doubles then casts to
`float` without checking representability. A compiled probe against the real parser
returned `RAD_OK` for `1e300` with scale infinity, and for `1e-300` with scale zero.
The former collapses finite positions toward zero; the latter takes the kernel's
zero-scale fallback to unscaled positions. Both silently change the intended model.
The fix checks the narrowed value is finite and positive before assigning it.
The parser regression rejects overflow, underflow and invalid values, checks that
rejection preserves the previous scale, and accepts representable fractional scaling.

Reproducer: local `audit-edge-probe.cpp` includes the real architecture header and
supplies only its metadata lookup dependency. Its disabled-magnitude demonstration
evaluates the reviewed formulas; it is not a full kernel launch test.

### P3 — reproducibility and coverage hardening

- Pin vLLM/torch versions and record backend/version metadata in the fixture generator.
- Add explicit declaration assertions separating target and DFlash2 coefficients,
  plus MoE/MTP and conflicting-metadata precedence cases. Current evidence combines
  parser tests, shared source paths and model runs rather than isolated assertions.
- The rest of the traced YaRN propagation showed no missing rotary path.

### Fix verification

`compat-fixes-host.log` records a successful host rebuild and 5/5 focused CTest
targets: `yarn_test`, `ref_test`, `arch_test`, `avx_check` and `ops_doc`.
This includes the new parser and disabled-coefficient regressions.
The previous 41/41 suite and GPU/model results remain historical pre-fix evidence;
the GPU serving build was not rebuilt or restarted for this host-side verification.

## Evidence location and reproduction

Raw evidence is retained locally at
`~/repo/homelab/.local/experiments/radiance-20261006/evidence/`.
Trial tools are in its parent: `validate-qwen-matrix.py`, `smoke.py`, `retrieval.py`,
`qualify-yarn.py`, `relabel-rad.cpp`, `verify-rad-vocab.cpp`, `audit-evidence.py` and
`audit-edge-probe.cpp`. Raw logs are not bundled in this source change.

```sh
ctest --test-dir build-host --output-on-failure
python3 audit-evidence.py > concurrency-audit.json
python3 smoke.py http://127.0.0.1:8080 smoke.json
python3 retrieval.py http://127.0.0.1:8080 retrieval.json
python3 qualify-yarn.py http://127.0.0.1:8080 qualify-yarn.json
```

The last three commands require the intended model already serving and run real
inference. The matrix runner stops/starts trial containers; it is not a generic
upstream CI tool. Package the portable probes and sanitized result excerpts with
the PR if reviewers need to reproduce the model qualification outside this lab.
