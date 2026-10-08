# scripts/

Shell harnesses that drive a **running** engine: bring a model up, measure it, and assert on what
comes back over the wire. They are versioned with the engine because a measurement is only
reproducible if the thing that took it is, and because what each one encodes is an argument — why
a window is sampled the way it is, which counter is the honest one — that is expensive to
reconstruct from the number alone.

The scripts that start a server or a conversion expect the build at
`$HOME/workspace/radiance/build` and the containers under `$HOME/models/rad/`; every default is
an environment override (`M=` for the container, `P=` for the port, and so on — each script's
header lists its own). The ones that only attach to a live server need nothing but `curl` and
`jq`. Scripts resolve each other by `$(dirname "$0")`.

## Bringing a model up

| | |
|---|---|
| `mcserve.sh` | MiniCPM5-2B with its DSpark drafter (`minicpm5-2b-dspark.v2.rad`), left running on :8100. Records why `--max-model-len` is the container's `n_ctx_train` and why a dense model of this size needs no explicit VRAM split. |
| `fnserve.sh` | Qwen3.8-Flash-Next, the production container `w4nl64-i8lm.rad`, same contract. Records the production flags and why: the 200K context and how the expert/cache split sizes the prefix cache at it, the 2048-token chunk, the prefix-cache tiers and the lossy `wht6` cross-rank wire. |
| `fnstop.sh` | Stops whichever is running and waits for the process to go. |
| `convdspark.sh` | Rebuilds the container `mcserve.sh` serves, with the drafter merged. The conversion of record. |
| `convfn.sh` | Rebuilds the container `fnserve.sh` serves from `data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe`. Needs `CALIB=<dir>`: the recipe's expert rule runs GPTQ against the calibration Grams there (`docs/MOE-W4.md`). |
| `hfget.sh` | Fetches a checkpoint from Hugging Face with `curl` and `jq`. Resumable, and it leaves behind the duplicate weight formats a large repo often carries beside its safetensors. |

## Measuring

| | |
|---|---|
| `cstep2.sh` | Steady-state decode ms/step at a given concurrency, against a live server. Waits until exactly C sequences are running before sampling and re-checks at the end: a window whose edges carry fewer than C describes no concurrency at all. Model-agnostic (`P=` picks the port). |
| `dprofd.sh` | Per-op device timing (`--profile-ops`) at depth and concurrency, on the 27B dense model with its DFlash2 drafter (`q38-27b-fp8-df2.v2.rad`). The counters are cumulative and a run starts with prefill, so a single dump is a mixture of two populations — difference two dumps taken after prefill has ended. |
| `fnpf.sh` | Prefill throughput of whatever server is on `$P`, over the repository's own prose, with a fresh nonce a rep so the prefix cache cannot make a rep free. Prints card temperature and clock beside every rep. |
| `fnconc.sh` | What a long prefill costs the decoders running beside it, and the prefill's own time, on one clock: alone, then contended, with a warm pass discarded. |
| `fntrace.sh` | A Flash-Next prefill under `rocprofv3`, with the link counters from `/stats` across the same request, so the streamed-read rate of the MoE GEMM can be computed. |
| `mctrace.sh` | A MiniCPM5-2B decode trace under `rocprofv3`, formatted for `libr4d/isa/tsum.sh`. The `blocks` column is what the trace is for. |
| `ident.sh` | Run-to-run reproducibility at a fixed configuration: one hash per (temperature, question), to compare against another run of the same configuration. A speculative run is not expected to match a non-speculative one byte for byte — the split-K count is chosen from M, so the two take different reduction orders and can flip a near-tie late in a long answer — but a divergence within the first sentence or two is a defect. |
| `diskstall.py` | What restoring a conversation from the disk tier costs: the returning request's time to first token, and the longest stall it puts on a stream decoding beside it. |
| `cardclock.py` | What holds each card's clock under a sustained load, and which card a tensor-parallel step waits for. The step ends when its slowest rank finishes, and under a sustained load the firmware picks that rank's clock: the power limit, a temperature or current limit, or the top of its voltage/frequency curve. Reads every AMD card's `gpu_metrics` table ten times a second and counts a throttler bit only when the clock is lower while it is set (some firmware keeps a bit on at idle). The engine's own step counters sit beside it per window, so a slow window is told apart first: slower steps, fewer tokens per step (the text), or host time between steps. Drives its own greedy load, or with `--watch` only samples a server that real traffic is driving. `--compare` puts two saved runs side by side. Read-only on the cards. |
| `oracle.sh` | Every op compared against libref on the operands the model actually issued, which is the gap rad-kbench cannot close: its operands are its own, dense and rank-2, and an engine's are slices of arenas and weights off a file. Reads for two things — a MISMATCH, and the NOT CHECKED lines that name the rows whose weight layout has no published inverse and therefore cannot be compared at all. |

## Regression checks against a live server

These cover what `tests/chat_test.cpp` structurally cannot: the server, the wire format, and a
real model's output.

| | |
|---|---|
| `apikeys.sh` | Every request key on the lists in `core/server/oai.cpp`: a rejected key must be refused by name, an inert one accepted and ignored, an accepted one honoured — and a key the deployment cannot satisfy (`logprobs`, `grammar` on chat) refused rather than answered without it. |
| `fndrop.sh` | Clients that disconnect mid-request, at points derived from a timed full request: the server must stay up and keep answering a reference question identically. |
| `tierident.sh` | Whether a session that left VRAM comes back byte for byte, both halves of it. The VRAM repeat is the control: without it a differing hash proves nothing, because a cache hit could have been tie-divergent. A run where nothing moved is a failure, not a pass. On a hybrid model the linear-state checks are COUNTERS and not hashes — a lost snapshot is invisible in the text, because the engine replays from further back and produces the same answer. |
| `tiercross.sh` | The same question with TWO sessions in flight, which is the only way to see one being served the other's context or state — with a single conversation in the server there is no other conversation's bytes to be served. Compares each restore against that session's DEVICE-HIT run rather than its cold one, because a hybrid session's reuse is clamped to the last reachable checkpoint and a cold run reuses nothing by definition. |
| `mctools.sh` | The tool-call shapes that are hardest to parse — an array argument, two calls in one reply, prose before a call, plain chat with tools available. Asserts `finish_reason` and that no `<function`/`<param`/CDATA marker reaches content. |
| `mcstream.sh` | Tool-call replies **streamed**, asserting on raw SSE fragments rather than on the assembled message. Content cannot be retracted once sent, so a repair applied at the end is not a fix; this is the check that sees the difference. |
| `mcagent.sh` | What speculation is worth on an agentic workload — `/v1/chat/completions` with tools, which is a constrained decode. Speculation on and off alternate across restarts because the speculation depth is a server-level setting. |
| `mcdecl.sh` | How often the drafter actually proposes. `draft_acceptance_ratio` is conditional on having drafted, so it reads healthy while tokens-per-step collapses; blocks-per-step is the quantity that separates the two. |
