#!/usr/bin/env bash
# Exact local Qwen3.5-4B direct-load comparison. Invoke once under:
#   flock /tmp/gpu tools/bench/run_qwen35_4b_compare.sh \
#     /tmp/qwen35-transplant-4b-<commit>
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"
out=${1:?usage: run_qwen35_4b_compare.sh OUTPUT_DIR}
model=${MODEL:-$root/.hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
dataset=${DATASET:-/tmp/qwen35-4b-sharegpt-1024.json}
cpp=${CPP_BENCH:-$root/build-nix-cuda-transplant/examples/vllm-bench}
vllm_python=${VLLM_PYTHON:-$root/.venv-vllm/bin/python}
cmake_cache=${CMAKE_CACHE:-$(dirname "$(dirname "$cpp")")/CMakeCache.txt}

test ! -e "$out" || { echo "refusing to overwrite $out" >&2; exit 2; }
test -x "$cpp"
test -x "$vllm_python"
test -d "$model"
test -f "$dataset"
test -f "$cmake_cache"
git -C "$root" diff --quiet
mkdir -p "$out"

# Run vLLM in its host virtualenv (the Nix shell's CUDA runtime masks the
# system driver for this PyTorch build), while deriving the exact compiler/JIT
# tools and libstdc++ from the immutable CMake cache used for vllm.cpp.
ninja=$(sed -n 's/^CMAKE_MAKE_PROGRAM:[^=]*=//p' "$cmake_cache")
nvcc=$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "$cmake_cache")
host_cxx=$(sed -n 's/^CMAKE_CUDA_HOST_COMPILER:[^=]*=//p' "$cmake_cache")
cudart=$(sed -n 's/^CUDA_CUDART:[^=]*=//p' "$cmake_cache")
curand=$(sed -n 's/^CUDA_curand_LIBRARY:[^=]*=//p' "$cmake_cache")
test -x "$ninja"
test -x "$nvcc"
test -x "$host_cxx"
test -f "$cudart"
test -f "$curand"
cuda_home=$(dirname "$(dirname "$nvcc")")
cudart_home=$(dirname "$(dirname "$cudart")")
curand_store=$(dirname "$(dirname "$curand")")
curand_package=${curand_store#*-}
curand_package=${curand_package%-lib}
curand_header=$(find /nix/store -maxdepth 4 \
  -path "*-$curand_package-include/include/curand.h" -print -quit)
test -f "$curand_header"
curand_include=$(dirname "$curand_header")
# FlashInfer assumes a monolithic CUDA_HOME, while Nix provides nvcc, cudart,
# cuRAND and the driver as separate immutable outputs. Assemble a symlink-only
# view in /tmp so its generated -I/-L paths remain valid without copying files.
cuda_combined=${CUDA_COMBINED_HOME:-/tmp/vllm-cuda-combined-12.9}
mkdir -p "$cuda_combined/bin" "$cuda_combined/include" \
  "$cuda_combined/lib64/stubs"
for item in "$cuda_home/bin/"*; do
  ln -sfn "$item" "$cuda_combined/bin/$(basename "$item")"
done
for item in "$cuda_home/include/"* "$cudart_home/include/"* "$curand_include/"*; do
  ln -sfn "$item" "$cuda_combined/include/$(basename "$item")"
done
for item in "$cudart_home/lib/"* "$(dirname "$curand")/"* \
  /run/opengl-driver/lib/libcuda.so*; do
  ln -sfn "$item" "$cuda_combined/lib64/$(basename "$item")"
done
ln -sfn /run/opengl-driver/lib/libcuda.so \
  "$cuda_combined/lib64/stubs/libcuda.so"
if test -d "$cuda_home/nvvm"; then
  ln -sfn "$cuda_home/nvvm" "$cuda_combined/nvvm"
fi
libstdcpp=$($host_cxx -print-file-name=libstdc++.so.6)
test -f "$libstdcpp"
vllm_path=$(dirname "$ninja"):$(dirname "$nvcc"):$(dirname "$host_cxx"):$PATH
vllm_ld_library_path=$(dirname "$libstdcpp"):$(dirname "$cudart"):$(dirname "$curand"):/run/opengl-driver/lib

cpp_args=(
  --model "$model" --dataset-path "$dataset" --num-prompts 128
  --output-len 128 --concurrency 32 --temperature 0
  --max-num-batched-tokens 2048 --num-blocks 1280
)
vllm_args=(
  "$root/tools/bench/vllm_closed_loop_metrics.py"
  --model "$model" --dataset-path "$dataset" --num-prompts 128
  --output-len 128 --max-concurrency 32 --max-num-seqs 32
  --max-num-batched-tokens 2048 --max-model-len 4096
  --gpu-memory-utilization 0.88 --temperature 0
)
vllm_env=(
  PATH="$vllm_path"
  LD_LIBRARY_PATH="$vllm_ld_library_path"
  CUDA_HOME="$cuda_combined"
  CUDA_PATH="$cuda_combined"
  CPATH="$cudart_home/include:$curand_include"
  LIBRARY_PATH="$cudart_home/lib:$(dirname "$curand")"
  NIX_LDFLAGS="-L$cudart_home/lib -L$(dirname "$curand") -L/run/opengl-driver/lib"
  HF_HOME="$root/.hf-cache"
  XDG_CACHE_HOME="$root/.vllm-cache"
  TORCHINDUCTOR_CACHE_DIR="$root/.torchinductor-cache"
  TRITON_CACHE_DIR="$root/.triton-cache"
  TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
  FLASHINFER_WORKSPACE_BASE="$root/.flashinfer-cache"
  CUDA_VISIBLE_DEVICES=0
)

git -C "$root" rev-parse HEAD >"$out/commit.txt"
git -C "$root" status --porcelain=v1 >"$out/git-status.txt"
sha256sum "$dataset" "$cpp" >"$out/sha256.txt"
env "${vllm_env[@]}" "$vllm_python" -c \
  'import vllm; print(vllm.__version__)' >"$out/vllm-version.txt"

gpu_snapshot() {
  local path=$1
  nvidia-smi --query-gpu=name,driver_version,pstate,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader >"$path.csv"
  nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER >"$path.detail.txt"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >"$path.compute-apps.csv"
}

prepare_leg() {
  local name=$1
  python3 -m tools.bench.drop_file_cache --root "$model" \
    --output "$out/$name.cache-before.json" >"$out/$name.cache-before.log"
  gpu_snapshot "$out/$name.gpu-before"
  if test -s "$out/$name.gpu-before.compute-apps.csv"; then
    echo "GPU is not compute-idle before $name" >&2
    return 1
  fi
  awk '/^MemAvailable:/{print}' /proc/meminfo >"$out/$name.mem-before.txt"
  printf '%s\n' "$EPOCHREALTIME" >"$out/$name.started"
}

finish_leg() {
  local name=$1
  printf '%s\n' "$EPOCHREALTIME" >"$out/$name.finished"
  gpu_snapshot "$out/$name.gpu-after"
  awk '/^MemAvailable:/{print}' /proc/meminfo >"$out/$name.mem-after.txt"
}

run_cpp() {
  local phase=$1 mode=$2 rep=$3 direct=$4
  local name="$phase-cpp-$mode-r$rep"
  prepare_leg "$name"
  printf '%q ' env VT_RELEASE_HOST_WEIGHTS=1 VT_DIRECT_DEVICE_LOAD="$direct" \
    "$cpp" "${cpp_args[@]}" --output-token-ids "$out/$name.tokens.json" \
    >"$out/$name.command"
  printf '\n' >>"$out/$name.command"
  env VT_RELEASE_HOST_WEIGHTS=1 VT_DIRECT_DEVICE_LOAD="$direct" \
    "$cpp" "${cpp_args[@]}" --output-token-ids "$out/$name.tokens.json" \
    >"$out/$name.log" 2>&1 &
  local pid=$!
  if test "$phase" = memory; then
    python3 -m tools.bench.sample_process_memory --pid "$pid" \
      --output "$out/$name.memory.jsonl" --interval 0.2 --include-gpu \
      >"$out/$name.memory-summary.json"
  fi
  wait "$pid"
  finish_leg "$name"
}

run_vllm() {
  local phase=$1 rep=$2
  local name="$phase-vllm-r$rep"
  prepare_leg "$name"
  printf '%q ' env "${vllm_env[@]}" "$vllm_python" "${vllm_args[@]}" \
    --request-id-base "$((rep * 1000))" \
    --metrics-output "$out/$name.metrics.json" \
    --tokens-output "$out/$name.tokens.json" >"$out/$name.command"
  printf '\n' >>"$out/$name.command"
  env "${vllm_env[@]}" "$vllm_python" "${vllm_args[@]}" \
    --request-id-base "$((rep * 1000))" \
    --metrics-output "$out/$name.metrics.json" \
    --tokens-output "$out/$name.tokens.json" >"$out/$name.log" 2>&1 &
  local pid=$!
  if test "$phase" = memory; then
    python3 -m tools.bench.sample_process_memory --pid "$pid" \
      --output "$out/$name.memory.jsonl" --interval 0.2 --include-gpu \
      >"$out/$name.memory-summary.json"
  fi
  wait "$pid"
  finish_leg "$name"
}

gpu_snapshot "$out/series-before"
if test "${VLLM_PREFLIGHT:-0}" = 1; then
  env "${vllm_env[@]}" "$vllm_python" "${vllm_args[@]}" \
    --num-prompts 1 --output-len 2 --max-concurrency 1 --max-num-seqs 1 \
    --metrics-output "$out/preflight.metrics.json" \
    --tokens-output "$out/preflight.tokens.json" >"$out/preflight.log" 2>&1
  gpu_snapshot "$out/series-after"
  exit 0
fi
for rep in 1 2 3; do
  run_cpp memory on "$rep" 1
  run_vllm memory "$rep"
  run_cpp memory off "$rep" 0
done
for rep in 1 2 3; do
  run_cpp performance on "$rep" 1
  run_vllm performance "$rep"
  run_cpp performance off "$rep" 0
done
gpu_snapshot "$out/series-after"
