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
echo "If green, with the 35B snapshot present, the ctest above validated on GB10:"
echo "  * the M1.6/M1.7 CUDA kernels (reshape_and_cache, paged_attention, the"
echo "    sampling ops) via the CPU-vs-CUDA parity cases;"
echo "  * the M0-exit DENSE forward gate (RunQwen36Logits, ForwardDense);"
echo "  * the 35B greedy through the PAGED LLMEngine (test_qwen36_paged_engine:"
echo "    ReshapeAndCache + PagedAttention + batched GDN + Sampler end-to-end,"
echo "    token-for-token vs the M0-exit golden)."
echo
echo "STILL TODO (needs GB10 + separate work, not this script):"
echo "  * M2 throughput-parity-vs-vLLM benchmark (~/venvs/vllm-oracle) at large"
echo "    concurrency — the #1 MVP gate; needs the GPU perf kernels (FlashInfer-"
echo "    class attention, chunked-scan GDN, fused MoE, CUDA graphs)."
echo "  * The real APEX GGUF (~/work/apex/qwen36_35b/*.gguf) greedy parity — point"
echo "    --model at a Compact/Balanced .gguf (pure K-quant) through the same gate."
