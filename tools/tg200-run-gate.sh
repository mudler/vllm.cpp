#!/bin/sh
# TG200 4B acceptance-gate arm: median of N warm reps, greedy, batch 1.
# Usage: run-gate.sh <reps> <tag>
set -eu
cd /home/ghazni/github/vllm.cpp/tg200
REPS="$1"; TAG="$2"
export LD_LIBRARY_PATH=/opt/rocm/lib
export VT_GEMV_MMVQ=1 VT_SKINNY_BF16=1 VT_NORM_QUANT_FUSED=1
echo "== uptime before window =="
uptime
./build-hip/examples/vllm-cli \
  --model /home/ghazni/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "$(cat tools/tg200-prompt.txt)" \
  --max-tokens 256 --temperature 0 --seed 0 \
  --repeat "$REPS" 2>&1 | tee "/tmp/tg200-${TAG}.log"
