#!/usr/bin/env bash
# ADDITIVE-MODEL W4 dgx gate runner for OPT-125m (`OPTForCausalLM`).
#
# Runs the whole gate series under ONE `flock /tmp/gpu` so the measurements are
# uncontended and a concurrent agent queues rather than interleaves:
#   1. the OPT SACRED correctness gate + loader gate + the new-op unit test;
#   2. the NON-NEGOTIABLE regressions — 27B 235/235, 35B 315/315,
#      Qwen3-Coder 6/6, Qwen3-dense — which must be UNCHANGED because OPT is
#      new-files;
#   3. `compute-sanitizer memcheck` on the OPT engine path.
#
# Usage: scripts/opt-dgx-gate.sh <build-dir>
set -uo pipefail

BUILD="${1:?usage: opt-dgx-gate.sh <build-dir>}"
T="$BUILD/tests"
rc=0

run() {
  local name="$1"; shift
  echo "=================================================================="
  echo "### $name"
  echo "=================================================================="
  "$@" 2>&1 | tail -25
  local status=${PIPESTATUS[0]}
  echo "### $name -> exit $status"
  [ "$status" -ne 0 ] && rc=1
  return 0
}

echo "### goldens BEFORE:"
md5sum "$(dirname "$0")/../tests/parity/goldens/opt_greedy/"*.npy

# --- 1. OPT itself -----------------------------------------------------------
run "OPT new vt:: ops (LayerNorm/Relu/Add, CPU + CUDA)" "$T/test_ops_layernorm"
run "OPT W2 loader gate"                                "$T/test_opt_load"
run "OPT W4 SACRED correctness gate (STRICT)"           "$T/test_opt_paged_engine"

# --- 2. REGRESSIONS (non-negotiable, must be UNCHANGED) ----------------------
run "REGRESSION 27B  (expect 235/235)"        "$T/test_qwen27_paged_engine"
run "REGRESSION 35B  (expect 315/315)"        "$T/test_qwen36_paged_engine"
run "REGRESSION Qwen3-Coder (expect 6/6)"     "$T/test_qwen3coder_paged_engine"
run "REGRESSION Qwen3-dense"                  "$T/test_qwen3_paged_engine"
run "REGRESSION model registry"               "$T/test_model_registry"
run "REGRESSION pretokenizer (incl. kGpt2)"   "$T/test_pretokenizer"
run "REGRESSION gguf model loader"            "$T/test_model_loader_gguf"

# --- 3. memcheck on the OPT engine path --------------------------------------
run "memcheck: OPT engine path" \
  compute-sanitizer --tool memcheck --leak-check=full "$T/test_opt_paged_engine"

echo "### goldens AFTER (must be identical to BEFORE):"
md5sum "$(dirname "$0")/../tests/parity/goldens/opt_greedy/"*.npy

echo "=================================================================="
echo "### OPT dgx gate series overall exit: $rc"
exit "$rc"
