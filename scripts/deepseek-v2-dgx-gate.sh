#!/usr/bin/env bash
# MLA CAMPAIGN W8 dgx gate runner for DeepSeek-V2-Lite (`DeepseekV2ForCausalLM`),
# modelled on scripts/opt-dgx-gate.sh.
#
# Runs, in ONE series on an idle box:
#   1. the DeepSeek-V2 SACRED correctness gate (STRICT token-exact vs the vLLM
#      0.25.0 oracle) plus the loader/forward gates;
#   2. the NON-NEGOTIABLE regressions — 27B 235/235, 35B 315/315, Qwen3-Coder
#      6/6, Qwen3-dense 16/16, OPT 6/6 — which must be UNCHANGED. W8 touches
#      the model TU's diagnostic counters and adds tests; nothing in the shared
#      scheduler/runner path changed, and this is where that is PROVEN;
#   3. `compute-sanitizer` memcheck / racecheck / synccheck on the MLA path.
#      NOTE synccheck needs `--num-cuda-barriers 65536`: the default barrier
#      table overflows on a binary driving this many kernel families and the
#      tool then reports a BOGUS launch failure (W6 finding).
#
# SERIALIZE. Two things learned the hard way on 2026-07-22, both of which
# REBOOTED dgx: GB10's 119 GiB is UNIFIED memory, so (a) never run a big-model
# test alongside a parallel build, and (b) `ctest -j` can co-schedule two ~30 GiB
# model tests — run the big ones ONE AT A TIME, and re-run any big-model failure
# STANDALONE before calling it a regression.
#
# Usage: scripts/deepseek-v2-dgx-gate.sh <build-dir>
set -uo pipefail

BUILD="${1:?usage: deepseek-v2-dgx-gate.sh <build-dir>}"
T="$BUILD/tests"
G="$(dirname "$0")/../tests/parity/goldens/deepseek_v2_greedy"
rc=0

run() {
  local name="$1"; shift
  echo "=================================================================="
  echo "### $name"
  echo "=================================================================="
  "$@" 2>&1 | tail -40
  local status=${PIPESTATUS[0]}
  echo "### $name -> exit $status"
  [ "$status" -ne 0 ] && rc=1
  return 0
}

echo "### GPU must be idle (sole owner):"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true

echo "### goldens BEFORE:"
md5sum "$G"/* || true

# --- 1. DeepSeek-V2 itself ---------------------------------------------------
run "DSV2 W7 loader gate"                        "$T/test_deepseek_v2_load"
run "DSV2 W7 forward gate"                       "$T/test_deepseek_v2_forward"
run "DSV2 W8 SACRED correctness gate (STRICT)"   "$T/test_deepseek_v2_paged_engine"
run "DSV2 tokenizer special-token policy guard"  "$T/test_bpe"

# --- 2. REGRESSIONS (non-negotiable, must be UNCHANGED) ----------------------
run "REGRESSION 27B  (expect 235/235)"        "$T/test_qwen27_paged_engine"
run "REGRESSION 35B  (expect 315/315)"        "$T/test_qwen36_paged_engine"
run "REGRESSION Qwen3-Coder (expect 6/6)"     "$T/test_qwen3coder_paged_engine"
run "REGRESSION Qwen3-dense (expect 16/16)"   "$T/test_qwen3_paged_engine"
run "REGRESSION OPT (expect 6/6)"             "$T/test_opt_paged_engine"
run "REGRESSION runner / KV spec / engine"    "$T/test_runner"
run "REGRESSION kv_cache_interface"           "$T/test_kv_cache_interface"
run "REGRESSION llm_engine"                   "$T/test_llm_engine"

# --- 3. sanitizers on the MLA engine path ------------------------------------
run "memcheck: DSV2 MLA engine path" \
  compute-sanitizer --tool memcheck --leak-check=full "$T/test_deepseek_v2_forward"
run "racecheck: DSV2 MLA engine path" \
  compute-sanitizer --tool racecheck "$T/test_deepseek_v2_forward"
run "synccheck: DSV2 MLA engine path" \
  compute-sanitizer --tool synccheck --num-cuda-barriers 65536 \
  "$T/test_deepseek_v2_forward"

echo "### goldens AFTER (must be identical to BEFORE):"
md5sum "$G"/* || true

echo "=================================================================="
echo "### DeepSeek-V2 dgx gate series overall exit: $rc"
exit "$rc"
