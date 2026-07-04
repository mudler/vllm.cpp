#!/usr/bin/env bash
# dgx-bringup.sh — validate the M1 stack on real GB10 hardware (sm_121).
#
# The dev box is CPU-only, so every CUDA kernel added across M1.6 (reshape_and_
# cache, paged_attention), M1.7 (the sampling ops), and the M1.8 batched runner
# is build-guarded and has only ever run on CPU. This script builds with CUDA ON
# and runs the full ctest suite, which un-skips all the HasCuda()-gated CPU-vs-
# CUDA parity tests, plus the checkpoint-gated 35B forward gate (RunQwen36Logits).
#
# Run ON dgx.casa (not the dev box):
#   ssh dgx.casa
#   cd ~/work/vllm.cpp && git pull && bash scripts/dgx-bringup.sh
#
# Prereqs on dgx.casa (see .agents/environment.md):
#   - CUDA toolkit 13.x; nvcc must be on PATH (non-interactive SSH does not add it).
#   - The 35B snapshot at ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4
#     (the RunQwen36Logits gate SKIPs cleanly if the checkpoint is absent).
set -euo pipefail

export PATH=/usr/local/cuda/bin:${PATH}

BUILD_DIR=${BUILD_DIR:-build-cuda}
JOBS=${JOBS:-$(nproc)}

echo "=== nvcc ==="; nvcc --version | tail -2 || { echo "nvcc not on PATH"; exit 1; }

echo "=== configure (CUDA ON, sm_121) ==="
cmake -S . -B "${BUILD_DIR}" -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo

echo "=== build ==="
cmake --build "${BUILD_DIR}" -j"${JOBS}"

# The full suite. On a CUDA build this runs the CPU-vs-CUDA parity cases that
# SKIP on the dev box: reshape_and_cache, paged_attention, the sampling ops
# (temperature/greedy/top-k/top-p/random/penalties/min-p/masks), plus the M1.6
# GDN/attention op goldens on GB10 and the checkpoint-gated 35B forward gate.
echo "=== ctest (full — un-skips the dgx-pending CUDA parity + 35B gate) ==="
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo
echo "=== dgx bring-up complete ==="
echo "If green: the M1 CUDA kernels are validated on GB10 and (with the 35B"
echo "snapshot present) the M0-exit dense forward gate reproduced."
echo
echo "STILL TODO (not covered by this script — needs a new test):"
echo "  * 35B greedy through the PAGED LLMEngine loop (ReshapeAndCache +"
echo "    PagedAttention + batched GDN + Sampler end-to-end) vs the M0-exit"
echo "    result. RunQwen36Logits today exercises ForwardDense, not the paged"
echo "    engine. See docs/superpowers/plans for the follow-up."
echo "  * M2 throughput benchmark vs pip-vLLM (~/venvs/vllm-oracle)."
