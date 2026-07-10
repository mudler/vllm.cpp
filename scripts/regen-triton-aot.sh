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
# Env overrides:
#   BUILD_DIR      (default: build-triton-regen)
#   TRITON_PYTHON  (default: ~/venvs/vllm-oracle/bin/python)
#   VLLM_CPP_CUTLASS_DIR (default: ~/cutlass_probe, like dgx-bringup.sh)
#
# The regen happens at CONFIGURE time (execute_process); no build is required
# to refresh the vendored tree, but BUILD_DIR is left ready for `cmake --build`.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build-triton-regen}
TRITON_PYTHON=${TRITON_PYTHON:-${HOME}/venvs/vllm-oracle/bin/python}
CUTLASS_DIR=${VLLM_CPP_CUTLASS_DIR:-${HOME}/cutlass_probe}
export PATH=/usr/local/cuda/bin:${PATH}

if [ ! -x "${TRITON_PYTHON}" ]; then
  echo "error: TRITON_PYTHON=${TRITON_PYTHON} not found/executable" >&2
  echo "       point TRITON_PYTHON at a Python that has Triton installed" >&2
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
