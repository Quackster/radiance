#!/bin/sh
# Every matching op compared against libref ON THE OPERANDS THE DEVICE KERNEL WAS HANDED.
#
# rad-kbench falsifies one op at a time on operands IT invented: dense, rank-2, freshly filled. The
# engine's are row slices of a max_tok buffer, column slices of a fused projection, rank-3 views a
# scan kernel required, and weights that came off a file. A kernel can be correct at every shape
# rad-kbench knows and wrong at the one the model issues, so this runs the host row beside the
# device one on the real thing, every issue, and reports per operand.
#
# IT IS NOT A MEASUREMENT OF ANYTHING. Two device-to-host copies of every operand -- a fused
# gate_up weight is 170 MiB -- a host synchronisation per issue on every lane, and a reference
# kernel that is not trying to be fast. Keep MAXN small and the prompt short.
#
# ON A LARGE CONTAINER IT CAN STILL BE OOM-KILLED PART WAY THROUGH AND THAT IS THE HOST, NOT A
# DEFECT. A 166 GiB container with a 12 GiB host pool leaves little room for two copies of every
# operand of every issue; the run prints the verdicts it reached before it died, and those are as
# good as any others. Lower HOSTPOOL, or window the sweep with FROM/TO, to reach further in.
#
# ONE OPERAND CAN DO IT ON ITS OWN, AND THAT ONE IS NAMED RATHER THAN FATAL. The Flash-Next PLE
# gather reads a 47.68 GiB n-gram table, which no snapshot can hold on a host of ordinary size.
# An operand over RADIANCE_OP_ORACLE_MAXBYTES (4 GiB) is a NOT CHECKED line naming the operand and
# its size, so the sweep goes on; raise the cap to include it if the host can take three copies.
#
# WHAT A DISAGREEMENT DOES NOT MEAN. The kernel that claims a weight relayouts its canonical
# planes at load, and libr4d's fp8a8 weights are held in WMMA fragment order -- the same bytes,
# permuted. The oracle puts those back through the device row's own RadUnrelayoutFn to the planes
# and decodes them before it compares; a row that relayouts a weight and publishes no inverse is
# NOT CHECKED and says so by name. Read those lines: they are coverage this instrument does not
# have. The fp8 rows, the int8 (i8a8) block rows, the MoE 4-bit expert plane and its E4M3 scale
# views (both grouped GEMMs, every class), the dense w4 rows and the mxfp4 rows answer; the
# w8a16, w8a8 and w2a8 rows do not.
#
# NOT EVERY GAP IS AN INVERSE SOMEBODY HAS NOT WRITTEN, AND THIS ONE IS A MISSING ZERO POINT.
# `gemm_w2a8_nt` -- the 2-bit draft head, live on EVERY speculating container, and the same gap
# under all three drafters (mtp.draft_head.weight, dspark.draft_head.weight,
# dflash.draft_head.weight) -- is stored on an UNSIGNED ASYMMETRIC grid, value = (code - zero) *
# scale per (row, group of 128).
#
# An inverse for the WEIGHT operand alone cannot close it, and not because the permutation is hard:
# undoing the fragment order to a plane of unsigned codes is the same work the fp8 rows already do.
# The obstruction is that `zero` is not in that operand. The weight's zero plane is taken by
# `b_scale` beside its scale plane (r4d.w2a8.sz.g128, a packed pair a dword), and the reference
# reads `b` as codes and `b_scale` as a symmetric scale -- it has no zero point to apply.
#
# What would close it is a zero point in `gemm_nt_q`'s schema and an asymmetric reader in libref,
# which is a FEATURE and not a missing hook. The row is covered instead by r4d_selftest's own
# gemm_w2a8_nt case, against a host decode of the canonical planes -- and what the draft
# head produces is a PROPOSAL the target verifies with its own head, so an error here costs
# acceptance and cannot change a token the model emits.
#
# A COLLECTIVE IS CHECKED ONLY WHERE BOTH RANKS ARM ON THE SAME ISSUE, AND AN OPS FILTER IS WHAT
# BREAKS THAT. libref's all_reduce is a RENDEZVOUS: it parks on a barrier until every rank has
# handed in its message. The oracle arms each rank on that rank's OWN issue counter and spends
# that rank's OWN budget, so the two sides have to arrive together. Unfiltered they mostly do --
# about one collective in ten is refused over a tp2 decode. A NARROW ops list is what pulls
# them apart -- the budget is then spent only on the named ops, the ranks do not issue the same
# ones (an expert-parallel split alone sees to that), and one rank reaches the collective armed
# while the other does not. What comes back then is
#
#   oracle #609 op 'all_reduce': the host kernel REFUSED ... with device error
#   oracle #620 op 'all_reduce': the host kernel REFUSED ... with shape this kernel does not serve
#
# which are the 30-second barrier timeout and, on the next attempt, the peer's stale message
# length. Neither is a finding about all_reduce. Sweep collectives with OPS=*, or at TP=1 where
# world_size is 1 and the reference answers in place; the two-rank wire also has its own tests
# (tests/wire_test.cpp and the tp2 determinism gate).
#
# A ROUTED EXPERT GEMM IS THE EXPENSIVE ONE. Its weight operand is a table of the layer's whole
# expert set and the snapshot materialises all of it, twice, plus the inverted copy -- some
# gigabytes an issue. MAXN=2 is a sweep, not a limitation of the instrument.
#
# WHAT HAS NO HOST KERNEL AT ALL, on top of the inverses above. Ten ops the architectures declare
# have no libref row, so naming one reports "0 op(s) checked, N skipped for want of a host kernel":
# the five QSA ops (qsa_block_key, qsa_score, qsa_select, qsa_work, qsa_tail_store), dflash_conv,
# gate_quant_fp8, router_topk_scatter, qk_norm_rope_gate and ar_rmsnorm_quant_fp8.
#
# THAT IS NOT THE SAME AS UNCHECKED. Every one of them has a numeric case in r4d_selftest -- the
# QSA ops against f64 evaluations of their own formulae, a host sort and a host walk; the four
# fusions against the unfused pair they replace, which is a BYTE-IDENTICAL check and stronger than
# any tolerance this instrument applies. What the missing rows cost is this instrument's own
# question: whether the kernel holds on the ENGINE'S operands rather than on invented ones.
#
# ACROSS THE FOUR ARCHITECTURES THE GAPS ARE THE SAME THREE. Flash-Next (four-bit MoE, QSA, PLE),
# Qwen3.6-35B-A3B (fp8 MoE), the 27B dense with the DFlash2 drafter, and MiniCPM5-2B with DSpark
# issue 42 distinct op types between them, and every comparison made agrees. What every
# speculating container leaves NOT CHECKED is `gdn_recurrent_update` under the anchor state form
# (which r4d_selftest covers instead), `quant_act_i8`'s asum arm, and the w2a8 draft head weight
# above. Nothing else.
#
# A NARROW SWEEP IS STILL WORTH RUNNING: at MAXN=400 the budget is spent inside the prefill of a
# deep model, so the drafter and the sampler chain are only reached with FROM past it or with an OPS
# list that skips the GEMMs. Both, for a full picture.
#
#   OPS=a,b   op names, or * for every op        MAXN=8    stop after this many checks
#   FROM/TO   issue-index window                 TOL=0.02  relative tolerance
#   M=path    container                          PROMPT/MT the request to run
P="${P:-8302}"
M="${M:-$HOME/models/rad/minicpm5-2b-dspark.v2.rad}"
pkill -f "port $P"; while pgrep -f "port $P" >/dev/null; do sleep 1; done
cd "$HOME/workspace/radiance/build" || exit 1
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 \
    RADIANCE_OP_ORACLE="${OPS:-*}" \
    RADIANCE_OP_ORACLE_MAX="${MAXN:-8}" \
    RADIANCE_OP_ORACLE_FROM="${FROM:-0}" \
    RADIANCE_OP_ORACLE_TOL="${TOL:-0.02}" \
    ${TO:+RADIANCE_OP_ORACLE_TO=$TO} \
nohup ./bin/radiance --model "$M" \
    --radiance-home "$HOME/workspace/radiance/build/radiance_home" \
    --host 127.0.0.1 --port "$P" \
    --max-num-seqs 1 --max-model-len "${MLEN:-8192}" --tp "${TP:-1}" \
    --gpu-headroom-mib "${HR:-512}" --kv-cache-dtype "${KVD:-bf16}" \
    --max-num-batched-tokens "${MNBT:-512}" $EXTRA \
    > "$HOME/oracle.log" 2>&1 &
for i in $(seq 1 90); do
  c=$(curl -s -m 3 -o /dev/null -w %{http_code} "http://127.0.0.1:$P/health" 2>/dev/null)
  [ "$c" = 200 ] && break
  pgrep -f "port $P" >/dev/null || { echo "DIED AT LOAD"; grep -E "^E " "$HOME/oracle.log" | tail -6; exit 1; }
  sleep 4
done
[ "$c" = 200 ] || { echo "NO START"; grep -E "^E " "$HOME/oracle.log" | tail -6; exit 1; }
curl -s -m 900 "http://127.0.0.1:$P/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"m\",\"prompt\":\"${PROMPT:-The capital of France is}\",\"max_tokens\":${MT:-2},\"temperature\":0}" \
  | head -c 300
echo
pkill -f "port $P"; while pgrep -f "port $P" >/dev/null; do sleep 1; done
echo "=== verdicts ==="
# The per-operand detail stays in the log; a MISMATCH line names the op and the kernel, and the
# NOT CHECKED lines are the coverage gaps. Both are what a sweep is read for.
grep -E "oracle #|op oracle|NOT CHECKED" "$HOME/oracle.log" | sed 's/^W //'
