#!/bin/sh
# Bring up MiniCPM5-2B (fp8) as a long-lived server and LEAVE IT RUNNING.
#
# The sibling of scripts/fnserve.sh, for the small dense model. Same contract: start it, wait for
# /health, print the endpoints, exit with the server still up. Stop it with scripts/fnstop.sh.
#
# WHY 131072 AND NOT 200000. n_ctx_train is 131072 for this checkpoint; the 200K context
# scripts/fnserve.sh serves is a property of the Flash-Next model, which a 2B model cannot honour.
#
# KV is 21.0 KiB a token at fp8 (42 layers x 2 kv heads x 128 x 2 planes), halved per rank at
# tp 2, so a full-length sequence costs 1.38 GiB a rank and 8 of them do not fit -- the pool is
# a capacity, not a per-request reservation, and it backs however many the cache can hold.
P="${P:-8100}"
M="${M:-$HOME/models/rad/minicpm5-2b-dspark.v2.rad}"
cd "$HOME/workspace/radiance/build" || exit 1
pkill -f "bin/radianc[e]"; while pgrep -f "bin/radianc[e]" >/dev/null; do sleep 1; done; sleep 2
env GPU_MAX_ALLOC_PERCENT=100 GPU_MAX_HEAP_SIZE=100 $EXTRA_ENV \
nohup ./bin/radiance --model "$M" \
    --radiance-home "$HOME/workspace/radiance/build/radiance_home" \
    --host 0.0.0.0 --port "$P" \
    --max-num-seqs ${SEQS:-8} --max-model-len ${MLEN:-131072} \
    --tp ${TP:-2} \
    --gpu-headroom-mib ${HR:-160} --kv-cache-dtype ${KVD:-fp8} \
    --max-num-batched-tokens ${MNBT:-512} \
    $EXTRA > "$HOME/mcserve.log" 2>&1 &
echo "starting, log: ~/mcserve.log"
for i in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] && break
  pgrep -f "bin/radianc[e]" >/dev/null || { echo "DIED after ${i}x4s"; grep -iE "^E |error|refus" "$HOME/mcserve.log" | tail -20; exit 1; }
  sleep 4
done
[ "$(curl -s -m 3 -o /dev/null -w %{http_code} http://localhost:$P/health 2>/dev/null)" = 200 ] || {
  echo "DID NOT START in 10 min"; grep -iE "^E |error" "$HOME/mcserve.log" | tail -20; exit 1; }
IP=$(ip -4 -o addr show scope global | awk "{split(\$4,a,\"/\"); print a[1]; exit}")
echo "UP  pid $(pgrep -f "bin/radianc[e]" | tr "\n" " ")"
echo "  completions  http://$IP:$P/v1/completions"
echo "  chat         http://$IP:$P/v1/chat/completions"
echo "  dashboard    http://$IP:$P/"
echo "  metrics      http://$IP:$P/metrics"
grep -iE "serving on|listening on|KV cache|placement|budget" "$HOME/mcserve.log" | tail -10
