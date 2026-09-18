#!/bin/bash
# Is 25.88 tok/s llama.cpp's PRODUCTION best on this board, or a fallback?
# `-ngl 99` reports "offloaded 49/49 layers" and STILL leaves a 27,465.95 MiB CPU
# model buffer. Measure the alternatives instead of explaining the baseline away.
set -u
BIN=/tmp/q4exp-hip-src-D9K2/build/bin
GGUF=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
OUT=/workspace/q4exp-gfx1151-denominator/offload-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"; export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib"
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
( while sleep 120; do kill -0 $$ 2>/dev/null||exit 0; echo "### hb"; done ) & HB=$!; trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM
echo "free -g:"; free -g | head -2
echo "-- what llama-bench offers for placement --"
"$BIN/llama-bench" -h 2>&1 | grep -iE 'ot,|override-tensor|no-host|cpu-moe|ncmoe|-dev|--device|mmap' | head -20

probe(){ # probe <tag> <args...>
  local tag="$1"; shift
  say "PROBE $tag :: $*"
  timeout 1500 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 64 -ngl 99 -r 3 -v -o json "$@" \
     > "$OUT/$tag.json" 2> "$OUT/$tag.err" </dev/null
  local rc=$?
  echo "$tag rc=$rc"
  grep -E 'load_tensors: *(CPU|ROCm0|ROCm_Host) model buffer size|offloaded .*layers|cannot be used with preferred' "$OUT/$tag.err" | sed "s/^/$tag  /"
  python3 -c "
import json;d=json.load(open('$OUT/$tag.json'))[0]
print('$tag  avg_ts=%.4f stddev=%.4f samples=%s'%(d['avg_ts'],d['stddev_ts'],d['samples_ts']))" 2>&1
  [ $rc -ne 0 ] && tail -5 "$OUT/$tag.err"
  return 0
}
probe A_baseline
probe B_ot_exps_rocm -ot "exps=ROCm0"
probe C_ot_all_rocm  -ot ".*=ROCm0"
probe D_nohost       --no-host 1
probe E_nommap       --mmap 0

say "REPEAT the arm that changed the split, if any (2 more legs each for A and the best variant)"
for t in A2 A3; do probe "$t"; done
say "ARCHIVE"
gzip -f "$OUT"/*.err
echo "=====OFF_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====OFF_TAR_B64_END====="
say "DONE"
