#!/bin/sh
# Arm wrapper: $1=N reps $2=tag $3..=env assignments
set -eu
cd /home/ghazni/github/vllm.cpp/tg200
REPS="$1"; TAG="$2"; shift 2
export LD_LIBRARY_PATH=/opt/rocm/lib
for kv in "$@"; do export "$kv"; done
./build-hip/examples/vllm-cli \
  --model /home/ghazni/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "$(cat tools/tg200-prompt.txt)" \
  --max-tokens 256 --temperature 0 --seed 0 \
  --repeat "$REPS" 2>&1 | grep -E 'vllm-cli: run=' | sed "s/^/$TAG /"
