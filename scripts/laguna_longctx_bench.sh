#!/usr/bin/env bash
# Two-length long-context slope harness for the Laguna-XS-2.1-NVFP4 decode levers
# (CLAIM-LAGUNA-LONGCTX-LEVERS). Measures the SLOPE, not one point: tok/s + nsys GPU-active
# ms/step at TWO generated lengths (~256 AND ~2048) for the A/B matrix of
#   Lever B = VT_LAGUNA_SWA_WINDOW (byte-exact window-bounded SWA reads, default ON)
#   Lever A = VT_LAGUNA_KV_BF16    (near-tie bf16 paged KV, default OFF / opt-in)
#
# RUN ON GB10 (dgx.casa), from the extracted build tree (git-archive transfer — NOT rsync),
# with the checkpoint at ~/laguna-xs-nvfp4. Serializes on GPU_LOCK, default $HOME/gpu.lock
# (never force).
# Park local-ai-worker before, restore after (see AGENTS box-safety). Median-of-3, drop-caches.
#
#   scripts/laguna_longctx_bench.sh <nvfp4-dir> <laguna-gen-binary>
#
# The graph/eager KV is fixed-capacity (pos + VT_LAGUNA_KV_HEADROOM rows); the 2048 arm needs
# headroom >= ~2048 or it trips the capacity VT_CHECK — this harness sets it per length.
set -euo pipefail

MODEL="${1:-$HOME/laguna-xs-nvfp4}"
BIN="${2:-$HOME/laguna-n4-build/build-cuda/examples/laguna-gen}"
IDS="2,785,9626,377,15360,395"          # prompt ids (same stream as prior Laguna gates)
BASE_ENV="VT_LAGUNA_RESIDENT_DECODE=1 VT_LAGUNA_MARLIN_MOE=1 VT_LAGUNA_DECODE_GRAPH=1"
LOCK="${GPU_LOCK:-$HOME/gpu.lock}"   # THE box mutex, one truth (#777)
OUT="$HOME/laguna-longctx-$(date +%s)"
mkdir -p "$OUT"

run_one() {  # $1=label $2=max_tokens $3=extra_env
  local label="$1" ntok="$2" extra="$3"
  local headroom=$(( ntok + 64 ))
  local log="$OUT/${label}.log"
  echo "== $label  (max_tokens=$ntok, headroom=$headroom) ==" | tee -a "$OUT/summary.txt"
  # drop caches for a paging-immune wall-clock; nsys per-node for the GPU-active ms/step anchor.
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' || true
  flock "$LOCK" -c "env $BASE_ENV $extra VT_LAGUNA_KV_HEADROOM=$headroom \
    nsys profile --cuda-graph-trace=node -o '$OUT/${label}.nsys' --force-overwrite true \
    '$BIN' --model '$MODEL' --token-ids '$IDS' --max-tokens '$ntok' --gpu \
    > '$log' 2>&1" || { echo "RUN FAILED: $label (see $log)"; return 1; }
  # gates: tok/s + the generated id stream (prove the code RAN + which tokens)
  grep -E 'decode_hp|generated ids' "$log" | tee -a "$OUT/summary.txt"
  # per-step GPU-active ms: parse a decode-only steady window from the nsys node trace
  nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/${label}.nsys.nsys-rep" \
    > "$OUT/${label}.kern.csv" 2>/dev/null || true
  echo "  (nsys kern sum: $OUT/${label}.kern.csv — attribute decode-only per nsys-aggregation-trap rule)" \
    | tee -a "$OUT/summary.txt"
}

# The matrix. Baseline = B off + A off (== origin/main byte stream, md5 anchor). Then B-alone,
# A-alone, B+A, each at 256 and 2048 — six deltas reveal the slope + whether it crosses
# 0.3 ms/step at ~2k.
for LEN in 256 2048; do
  run_one "baseline_${LEN}"  "$LEN" "VT_LAGUNA_SWA_WINDOW=0 VT_LAGUNA_KV_BF16=0"
  run_one "leverB_${LEN}"    "$LEN" "VT_LAGUNA_SWA_WINDOW=1 VT_LAGUNA_KV_BF16=0"
  run_one "leverA_${LEN}"    "$LEN" "VT_LAGUNA_SWA_WINDOW=0 VT_LAGUNA_KV_BF16=1"
  run_one "leverBA_${LEN}"   "$LEN" "VT_LAGUNA_SWA_WINDOW=1 VT_LAGUNA_KV_BF16=1"
done

# Byte-exact gate for Lever B: baseline vs leverB id streams MUST be identical at BOTH lengths.
for LEN in 256 2048; do
  b=$(grep 'generated ids' "$OUT/baseline_${LEN}.log" || true)
  l=$(grep 'generated ids' "$OUT/leverB_${LEN}.log" || true)
  if [ -n "$b" ] && [ "$b" = "$l" ]; then echo "LEVER B byte-exact @${LEN}: PASS" | tee -a "$OUT/summary.txt"
  else echo "LEVER B byte-exact @${LEN}: FAIL (compare $OUT/baseline_${LEN}.log vs leverB_${LEN}.log)" | tee -a "$OUT/summary.txt"; fi
done
echo "DONE — results in $OUT/summary.txt" | tee -a "$OUT/summary.txt"
touch "$OUT/.done"
