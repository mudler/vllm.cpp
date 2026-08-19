#!/usr/bin/env bash
# regen-triton-aot.sh — MAINTAINER task: regenerate the vendored Triton AOT
# artifacts (src/vt/cuda/triton_aot_vendored/<arch>/) from triton_kernels/*.py.
#
# Normal builds do NOT need this (nor Python/Triton at all): with
# -DVLLM_CPP_TRITON=ON the build consumes the vendored .c/.h artifacts with only
# a C compiler. Run this ONLY when the kernels / signatures / launch pins change
# (the vendored-path configure fails when they drift), or to add a new
# arch tree; then review + commit the git diff it prints.
#
# Requirements: CUDA toolkit (nvcc) and a Python with Triton installed (Triton
# compiles cubins with its OWN bundled ptxas). The target is derived from the
# vendored arch directory; workflow runtime validation still requires a GPU.
#
# Usage:
#   bash scripts/regen-triton-aot.sh [extra -D cmake args...]
#
# Every machine-specific value comes from the process environment first and from
# the repository `.env` second. Copy `.env.example` to `.env` and fill in what
# your setup has. Required here:
#
#   VLLM_ORACLE           venv root, or a Python executable, that has Triton
#   CUTLASS_DIR           CUTLASS checkout, passed as -DVLLM_CPP_CUTLASS_DIR
#
# Optional:
#
#   DEVICE_TOOLKIT_ROOT   prepended as $DEVICE_TOOLKIT_ROOT/bin to PATH, which a
#                         non-interactive SSH session usually needs because it
#                         does not put nvcc on PATH by itself
#   TRITON_PYTHON         an interpreter that overrides the one VLLM_ORACLE names
#   VLLM_CPP_CUTLASS_DIR  a CUTLASS checkout that overrides CUTLASS_DIR
#   BUILD_DIR             (default: build-triton-regen, which is nobody's path)
#
# All three used to be hard-coded: a CUTLASS checkout, an oracle interpreter, and
# a CUDA toolkit prefix, each one developer's. The campaign's worked example
# measured what the first of those costs:
# a configure that does not find CUTLASS silently drops the sm120a NVFP4 GEMM
# and FlashAttention-2, and `.agents/environment.md:388-400` records that turning
# those off moves the SACRED `test_qwen27_paged_engine` from 235/235 to 234/235
# with the source untouched. This script REGENERATES artifacts that then get
# committed, so a wrong toolchain here lands in the tree. It refuses instead of
# guessing (#1190).
#
# The regen happens at CONFIGURE time (execute_process); no build is required
# to refresh the vendored tree, but BUILD_DIR is left ready for `cmake --build`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# The shell wins over the file, and that ordering needs code rather than a
# comment. A `.env` copied from `.env.example` declares EVERY key blank, so
# sourcing it assigns the empty string over a value the caller just exported,
# and this script would then refuse a session that was correctly configured.
# Save what the shell answered, source the file, then put the shell's answers
# back. The worked example `scripts/dgx-bringup.sh` carries the same three steps.
PRESET_VLLM_ORACLE="${VLLM_ORACLE:-}"
PRESET_CUTLASS_DIR="${CUTLASS_DIR:-}"
PRESET_DEVICE_TOOLKIT_ROOT="${DEVICE_TOOLKIT_ROOT:-}"
PRESET_TRITON_PYTHON="${TRITON_PYTHON:-}"
PRESET_VLLM_CPP_CUTLASS_DIR="${VLLM_CPP_CUTLASS_DIR:-}"

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

prefer_shell_value VLLM_ORACLE "${PRESET_VLLM_ORACLE}"
prefer_shell_value CUTLASS_DIR "${PRESET_CUTLASS_DIR}"
prefer_shell_value DEVICE_TOOLKIT_ROOT "${PRESET_DEVICE_TOOLKIT_ROOT}"
prefer_shell_value TRITON_PYTHON "${PRESET_TRITON_PYTHON}"
prefer_shell_value VLLM_CPP_CUTLASS_DIR "${PRESET_VLLM_CPP_CUTLASS_DIR}"

refuse_env() {
  local key="$1" what="$2"
  echo "REFUSED: ${key} is unset. ${what}" >&2
  echo "Set it in ${ROOT}/.env (copy .env.example) or in this shell." >&2
  echo "Ask the developer for the value. Never substitute another" >&2
  echo "developer's host or path, and never guess a default." >&2
  exit 3
}

# `.env.example` declares VLLM_ORACLE as a venv ROOT or a Python executable, so
# both shapes resolve here rather than one being silently wrong.
if [ -z "${TRITON_PYTHON:-}" ]; then
  if [ -z "${VLLM_ORACLE:-}" ]; then
    refuse_env VLLM_ORACLE \
      "It is the interpreter that compiles the vendored Triton cubins."
  fi
  if [ -d "${VLLM_ORACLE}" ]; then
    TRITON_PYTHON="${VLLM_ORACLE}/bin/python"
  else
    TRITON_PYTHON="${VLLM_ORACLE}"
  fi
fi

CUTLASS_DIR="${VLLM_CPP_CUTLASS_DIR:-${CUTLASS_DIR:-}}"
if [ -z "${CUTLASS_DIR}" ]; then
  refuse_env CUTLASS_DIR \
    "Without it the configure drops NVFP4 and FlashAttention-2 silently."
fi

BUILD_DIR=${BUILD_DIR:-build-triton-regen}

if [ -n "${DEVICE_TOOLKIT_ROOT:-}" ]; then
  export PATH="${DEVICE_TOOLKIT_ROOT}/bin:${PATH}"
fi

echo "=== triton python: ${TRITON_PYTHON} ==="
if [ ! -x "${TRITON_PYTHON}" ]; then
  echo "error: TRITON_PYTHON=${TRITON_PYTHON} not found/executable" >&2
  echo "       point TRITON_PYTHON at a Python that has Triton installed," >&2
  echo "       or set VLLM_ORACLE to the venv that has it" >&2
  exit 1
fi
"${TRITON_PYTHON}" -c 'import triton; print("triton", triton.__version__)'

echo "=== configure with -DVLLM_CPP_TRITON_REGEN=ON (regen runs at configure time) ==="
cmake -S . -B "${BUILD_DIR}" \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUTLASS_DIR="${CUTLASS_DIR}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_CPP_TRITON=ON \
  -DVLLM_CPP_TRITON_REGEN=ON \
  -DVLLM_CPP_TRITON_PYTHON="${TRITON_PYTHON}" \
  "$@"

echo
echo "=== vendored tree diff (review + commit; regen is a maintainer task) ==="
git status --short -- src/vt/cuda/triton_aot_vendored/ || true
git --no-pager diff --stat -- src/vt/cuda/triton_aot_vendored/ || true
echo
echo "If the diff is empty, the vendored artifacts were already current"
echo "(byte-identical for the pinned target/toolchain; line info is disabled)."
echo "Untracked files above are NEW artifacts: add them."
