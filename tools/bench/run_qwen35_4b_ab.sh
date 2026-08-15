#!/usr/bin/env bash
# Same-binary A/B on the exact Qwen3.5-4B comparison workload.
#
# The comparison harness (run_qwen35_4b_compare.sh) answers "where are we versus
# vLLM". This answers "did THIS toggle move us", which needs a different shape:
# one binary, one lock across the WHOLE series, the two arms interleaved so a
# thermal or clock drift hits both arms equally, and the generated token ids
# captured per leg so an arm that changed the OUTPUT is caught rather than
# celebrated as a speedup.
#
# Invoke once, under the lock, from inside the CUDA dev shell:
#   flock "${GPU_LOCK:-$HOME/gpu.lock}" tools/bench/run_qwen35_4b_ab.sh /tmp/qwen35-ab-<name> \
#     A_NAME 'ENV=VAL ...' B_NAME 'ENV=VAL ...'
#
# Both arms get the identical request corpus, sampling and concurrency; the ONLY
# difference is the per-arm environment assignments.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"
out=${1:?usage: run_qwen35_4b_ab.sh OUTPUT_DIR A_NAME A_ENV B_NAME B_ENV}
a_name=${2:?missing A arm name}
a_env=${3?missing A arm env}
b_name=${4:?missing B arm name}
b_env=${5?missing B arm env}
reps=${REPS:-3}

model=${MODEL:-$root/.hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
dataset=${DATASET:-/tmp/qwen35-4b-sharegpt-1024.json}
cpp=${CPP_BENCH:-$root/build-nix-cuda-transplant-triton/examples/vllm-bench}
cmake_cache=${CMAKE_CACHE:-$(dirname "$(dirname "$cpp")")/CMakeCache.txt}

test ! -e "$out" || { echo "refusing to overwrite $out" >&2; exit 2; }
test -x "$cpp"
test -d "$model"
test -f "$dataset"
test -f "$cmake_cache"
test -e /run/opengl-driver/lib/libcuda.so.1
git -C "$root" diff --quiet
mkdir -p "$out"

# Same rationale as the comparison harness: the live host driver must precede
# Nix's link-only libcuda stub or a Triton AOT object silently misses CUDA
# registration and the arm measures a CPU fallback.
ld_path=/run/opengl-driver/lib
if test -n "${LD_LIBRARY_PATH:-}"; then
  ld_path="$ld_path:$LD_LIBRARY_PATH"
fi

cpp_args=(
  --model "$model" --dataset-path "$dataset" --num-prompts 128
  --output-len 128 --concurrency 32 --temperature 0
  --max-num-batched-tokens 2048 --num-blocks 1280
)

git -C "$root" rev-parse HEAD >"$out/commit.txt"
git -C "$root" status --porcelain=v1 >"$out/git-status.txt"
sha256sum "$dataset" "$cpp" "$cmake_cache" >"$out/sha256.txt"
grep -E '^(CMAKE_BUILD_TYPE|VLLM_CPP_CUDA_ARCHITECTURES|VLLM_CPP_FLASH_ATTN|VLLM_CPP_SANITIZE|VLLM_CPP_TRITON|VLLM_CPP_TRITON_REGEN|VLLM_CPP_TRITON_VENDORED_ARCH):' \
  "$cmake_cache" >"$out/cpp-build-config.txt"
printf 'A=%s env=[%s]\nB=%s env=[%s]\nreps=%s\n' \
  "$a_name" "$a_env" "$b_name" "$b_env" "$reps" >"$out/arms.txt"

gpu_snapshot() {
  nvidia-smi --query-gpu=name,driver_version,pstate,memory.used,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader >"$1.csv"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader >"$1.compute-apps.csv"
}

run_leg() {
  local name=$1 arm_env=$2 rep=$3
  local leg="$name-r$rep"
  # Cooldown FIRST, then snapshot. Without it the second arm inherits the first
  # arm's thermal and clock state and the A/B measures the order, not the toggle
  # - and the idle check below would read the previous leg's GPU still draining
  # and refuse to run at all, which is exactly what it did when this slept after
  # sampling instead of before.
  sleep "${COOLDOWN:-20}"
  gpu_snapshot "$out/$leg.gpu-before"
  if test -s "$out/$leg.gpu-before.compute-apps.csv"; then
    echo "GPU is not compute-idle before $leg" >&2
    return 1
  fi
  # Same hole the comparison harness had: --query-compute-apps lists CUDA
  # contexts only, so a graphics consumer keeps the GPU busy invisibly. An A/B is
  # if anything MORE sensitive to it than a comparison, because a drifting
  # background load can look exactly like an arm effect.
  local util
  util=$(cut -d, -f5 "$out/$leg.gpu-before.csv" | tr -dc '0-9')
  if test -n "$util" && test "$util" -gt "${GPU_IDLE_UTIL_MAX:-2}"; then
    echo "GPU is not idle before $leg: utilization ${util}% exceeds" \
      "${GPU_IDLE_UTIL_MAX:-2}%" >&2
    return 1
  fi
  printf '%s\n' "$arm_env" >"$out/$leg.env"
  # shellcheck disable=SC2086 # arm_env is a deliberate list of NAME=VALUE words
  env LD_LIBRARY_PATH="$ld_path" VT_RELEASE_HOST_WEIGHTS=1 \
    VT_DIRECT_DEVICE_LOAD=1 $arm_env \
    "$cpp" "${cpp_args[@]}" --output-token-ids "$out/$leg.tokens.json" \
    >"$out/$leg.log" 2>&1
  gpu_snapshot "$out/$leg.gpu-after"
}

gpu_snapshot "$out/series-before"
for rep in $(seq 1 "$reps"); do
  # Interleaved, and the order FLIPS on even repetitions, so neither arm is
  # systematically the one that runs on a warmer GPU.
  if test $((rep % 2)) -eq 1; then
    run_leg "$a_name" "$a_env" "$rep"
    run_leg "$b_name" "$b_env" "$rep"
  else
    run_leg "$b_name" "$b_env" "$rep"
    run_leg "$a_name" "$a_env" "$rep"
  fi
done
gpu_snapshot "$out/series-after"
echo "A/B series complete: $out"
