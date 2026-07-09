#!/usr/bin/env bash
# dgx-bringup.sh — validate the gate stack on real GB10 hardware (sm_121a).
#
# The dev box may be CPU-only, so CUDA parity and real-model gates are validated
# on dgx.casa. This script builds with CUDA ON, sm_121a, and runs the full ctest
# suite, which un-skips HasCuda()-gated CPU-vs-CUDA tests plus checkpoint-gated
# 35B/27B paged-engine greedy gates when the model snapshots are present.
#
# Run ON dgx.casa (not the dev box):
#   ssh dgx.casa
#   cd ~/work/vllm.cpp && git pull && bash scripts/dgx-bringup.sh
#
# Prereqs on dgx.casa (see .agents/environment.md):
#   - CUDA toolkit 13.x; nvcc must be on PATH (non-interactive SSH does not add it).
#   - CUTLASS v4.4.2 at ~/cutlass_probe, or set VLLM_CPP_CUTLASS_DIR.
#   - The 35B snapshot at ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4
#   - The 27B snapshot at ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4
#     (checkpoint gates SKIP cleanly if the snapshots are absent).
set -euo pipefail

export PATH=/usr/local/cuda/bin:${PATH}

BUILD_DIR=${BUILD_DIR:-build-cuda}
JOBS=${JOBS:-8}
CUTLASS_DIR=${VLLM_CPP_CUTLASS_DIR:-${HOME}/cutlass_probe}

echo "=== nvcc ==="; nvcc --version | tail -2 || { echo "nvcc not on PATH"; exit 1; }

echo "=== configure (CUDA ON, sm_121a) ==="
cmake -S . -B "${BUILD_DIR}" -DVLLM_CPP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a \
      -DVLLM_CPP_CUTLASS_DIR="${CUTLASS_DIR}" -DCMAKE_BUILD_TYPE=RelWithDebInfo

echo "=== build ==="
cmake --build "${BUILD_DIR}" -j"${JOBS}"

# The full suite. On a CUDA build this runs the CPU-vs-CUDA parity cases, CUDA
# op goldens, and checkpoint-gated paged-engine model tests.
echo "=== ctest (full — un-skips CUDA parity + model gates) ==="
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo
echo "=== dgx bring-up complete ==="
echo "If green, with snapshots present, the ctest above validated on GB10:"
echo "  * CUDA kernels via CPU-vs-CUDA parity cases;"
echo "  * 35B MoE/W4A16 and 27B dense/W4A4 paged-engine greedy gates;"
echo "  * server/engine/library behavioral tests under the CUDA build."
echo
echo "STILL TODO (needs GB10 + separate work, not this script):"
echo "  * Fresh throughput-parity-vs-production-vLLM benchmark (~/venvs/vllm-oracle)"
echo "    at the gate workloads; record all axes and ratios in .agents/parity-ledger.md."
echo "  * The real APEX GGUF (~/work/apex/qwen36_35b/*.gguf) greedy parity — point"
echo "    --model at a Compact/Balanced .gguf (pure K-quant) through the same gate."
