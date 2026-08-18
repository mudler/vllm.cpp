#!/usr/bin/env bash
# dgx-bringup.sh — run the hardware gate stack on your CUDA gate box.
#
# The development box can be CPU-only, so the CUDA parity cases and the
# real-model gates run on the gate host your environment names. This script
# configures with CUDA ON at your DEVICE_ARCH and runs the full ctest suite,
# which un-skips the HasCuda()-gated CPU-vs-CUDA tests plus the
# checkpoint-gated paged-engine greedy gates when the snapshots are present.
#
# Run it ON the gate host, not on the development box:
#   ssh "${GATE_HOST}"
#   cd "${GATE_CHECKOUT}" && git pull && bash scripts/dgx-bringup.sh
#
# Every machine-specific value comes from the process environment first and
# from the repository `.env` second. Copy `.env.example` to `.env` and fill in
# what your setup has. This script never guesses a value, and it never falls
# back to another developer's path. Required here:
#
#   DEVICE_ARCH           vendor architecture name, for example 121a or gfx942
#   CUTLASS_DIR           CUTLASS checkout, passed as -DVLLM_CPP_CUTLASS_DIR
#
# Optional:
#
#   DEVICE_TOOLKIT_ROOT   prepended as $DEVICE_TOOLKIT_ROOT/bin to PATH, which
#                         a non-interactive SSH session usually needs because
#                         it does not put nvcc on PATH by itself
#
# The two required keys used to be hard-coded in this file, and the CUTLASS
# path this script defaulted to had already diverged from the one
# `.agents/environment.md` records as mandatory on the same box. A configure
# that does not find CUTLASS silently drops the sm120a NVFP4 GEMM and
# FlashAttention-2, which turns a SACRED greedy gate from 235/235 to 234/235
# with the source untouched. A wrong default is therefore a false green, not an
# inconvenience, and this script refuses instead of guessing (#1190).
#
# Prerequisites on the gate host are in `.agents/environment.md`:
#   - A CUDA toolkit whose nvcc matches DEVICE_ARCH.
#   - The gate model snapshots, under ${CHECKPOINT_ROOT} where your setup
#     agrees to keep them, or in the local HuggingFace cache. The two the
#     paged-engine greedy gates look for are `nvidia/Qwen3.6-35B-A3B-NVFP4`
#     and `unsloth/Qwen3.6-27B-NVFP4`, and docs/USAGE.md pins the revisions.
#     Checkpoint gates SKIP cleanly when a snapshot is absent.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# `.env` is untracked and per developer, and `set -a` matches the loader
# `.env.example` documents.
#
# The shell wins over the file, and that ordering needs code rather than a
# comment. A `.env` copied from `.env.example` declares EVERY key blank, so
# sourcing it assigns the empty string over a value the caller just exported,
# and this script then refuses a session that was correctly configured. Save
# what the shell answered, source the file, then put the shell's answers back.
PRESET_DEVICE_ARCH="${DEVICE_ARCH:-}"
PRESET_CUTLASS_DIR="${CUTLASS_DIR:-}"
PRESET_DEVICE_TOOLKIT_ROOT="${DEVICE_TOOLKIT_ROOT:-}"

if [ -f "${ROOT}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  . "${ROOT}/.env"
  set +a
fi

prefer_shell_value() {
  local key="$1" saved="$2"
  if [ -n "${saved}" ]; then
    export "${key}=${saved}"
  fi
}

prefer_shell_value DEVICE_ARCH "${PRESET_DEVICE_ARCH}"
prefer_shell_value CUTLASS_DIR "${PRESET_CUTLASS_DIR}"
prefer_shell_value DEVICE_TOOLKIT_ROOT "${PRESET_DEVICE_TOOLKIT_ROOT}"

require_env() {
  local key="$1" what="$2"
  if [ -n "${!key:-}" ]; then
    return 0
  fi
  echo "REFUSED: ${key} is unset. ${what}" >&2
  echo "Set it in ${ROOT}/.env (copy .env.example) or in this shell." >&2
  echo "Ask the developer for the value. Never substitute another" >&2
  echo "developer's host or path, and never guess a default." >&2
  exit 3
}

require_env DEVICE_ARCH \
  "It is the architecture this box builds for, in vendor naming."
require_env CUTLASS_DIR \
  "Without it the configure drops NVFP4 and FlashAttention-2 silently."

if [ -n "${DEVICE_TOOLKIT_ROOT:-}" ]; then
  export PATH="${DEVICE_TOOLKIT_ROOT}/bin:${PATH}"
fi

BUILD_DIR=${BUILD_DIR:-build-cuda}
JOBS=${JOBS:-8}

echo "=== nvcc ==="; nvcc --version | tail -2 || { echo "nvcc not on PATH"; exit 1; }

echo "=== configure (CUDA ON, ${DEVICE_ARCH}) ==="
cmake -S . -B "${BUILD_DIR}" -DVLLM_CPP_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${DEVICE_ARCH}" \
      -DVLLM_CPP_CUTLASS_DIR="${CUTLASS_DIR}" -DCMAKE_BUILD_TYPE=RelWithDebInfo

echo "=== build ==="
cmake --build "${BUILD_DIR}" -j"${JOBS}"

# The full suite. On a CUDA build this runs the CPU-vs-CUDA parity cases, CUDA
# op goldens, and checkpoint-gated paged-engine model tests.
echo "=== ctest (full — un-skips CUDA parity + model gates) ==="
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo
echo "=== gate bring-up complete ==="
echo "If green, with snapshots present, the ctest above validated on the GPU:"
echo "  * CUDA kernels via CPU-vs-CUDA parity cases;"
echo "  * MoE/W4A16 and dense/W4A4 paged-engine greedy gates;"
echo "  * server/engine/library behavioral tests under the CUDA build."
echo
echo "STILL TODO (needs the GPU + separate work, not this script):"
echo "  * Fresh throughput-parity-vs-production-vLLM benchmark (\${VLLM_ORACLE})"
echo "    at the gate workloads; record all axes and ratios in .agents/parity-ledger.md."
echo "  * The real APEX GGUF under \${CHECKPOINT_ROOT} greedy parity — point"
echo "    --model at a Compact/Balanced .gguf (pure K-quant) through the same gate."
