#!/bin/sh
# Per-op device timing (--profile-ops) AT DEPTH and AT CONCURRENCY, on the 27B dense model with
# its DFlash2 drafter: C requests of a ~16K-token prompt (PROMPT=<file>), so every
# context-scaling op (attn_paged above all) shows the weight it carries at a real KV depth rather
# than at a few dozen tokens.
#
# THE COUNTERS ARE CUMULATIVE AND THE RUN STARTS WITH PREFILL, so a single table is a mixture:
# ~30 chunked-prefill steps at 4096 tokens sit under every decode step that follows. This dumps
# every EVERY steps (RADIANCE_PROFILE_EVERY) and leaves them all in the log; differencing two dumps
# that are both after prefill ended is the only way to get a decode-only table out of it.
P=8129
C="${C:-8}"
EVERY="${EVERY:-20}"
MT="${MT:-400}"
cd "$HOME/workspace/radiance/build" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 3
: > "$HOME/dprofd.log"
[ -n "$KVFP8" ] && export RADIANCE_QWEN35_KV_FP8=1   # read by nothing: the KV width is --kv-cache-dtype
GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 RADIANCE_PROFILE_EVERY="$EVERY" \
./bin/radiance --model "${M:-$HOME/models/rad/q38-27b-fp8-df2.v2.rad}" \
    --radiance-home "$HOME/workspace/radiance/build/radiance_home" \
    --max-num-seqs 8 --max-model-len 24000 --checkpoint-slots 0 --no-prefix-cache \
    --tp 2 --tp-wire wht6 --tp-wire-min-kb 128 \
    --max-num-batched-tokens 4096 --num-speculative-tokens ${DEPTH:-auto} \
    --profile-ops --port $P >> "$HOME/dprofd.log" 2>&1 &
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED"; grep -iE "^E " "$HOME/dprofd.log" | tail -5; exit 1; }
  sleep 5
done
jq -Rs "{model:\"m\",prompt:.,max_tokens:$MT,temperature:0}" < "${PROMPT:-$HOME/lpcode16k.txt}" > /tmp/dpd.json
PIDS=; i=0
while [ $i -lt $C ]; do
  curl -s -m 1800 http://localhost:$P/v1/completions -H "Content-Type: application/json" -d @/tmp/dpd.json -o /tmp/dpd_$i.out &
  PIDS="$PIDS $!"; i=$((i+1))
done
wait $PIDS
echo "C=$C kvfp8=${KVFP8:-0} completion_tokens $(jq -r '.usage.completion_tokens' /tmp/dpd_0.out)"
sleep 2
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done
grep -c "per-op device timing, rank 0" "$HOME/dprofd.log" | sed 's/^/dumps: /'
echo DPROFDDONE
