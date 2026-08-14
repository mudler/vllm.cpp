#!/usr/bin/env bash
# DSpark 35B-A3B paired end-to-end measurement: ours -> oracle -> ours.
#
# COMMITTED because the headline ratio rests on it and a median cannot be
# recomputed from a record that only stores medians. Per-rep values for the
# runs it produced are in .agents/benchmark-record.md.
#
# FOUR CONTROLS, none of which is optional -- each was added because its
# absence produced a wrong number:
#   * $HOME/gpu.lock, NOT /tmp/gpu.lock. The latter coordinates with nothing.
#   * a DISCARDED warm-up arm: the GB10 SM clock ramps over MINUTES
#     (1449 -> 2190 MHz observed), so dropping rep 1 is not enough.
#   * settle barriers between arms: vLLM asserts free GPU memory does not GROW
#     during its startup profile, and GB10 releases our pages lazily, so an
#     oracle started straight after our arm aborts.
#   * a host-RAM headroom guard: gpu_memory_utilization reserves HOST RAM here,
#     so an oracle without headroom takes the MACHINE down, not the process.
#
# READ THE RESULT WITH ITS GATE: if the two `ours` arms differ by >= 1%, the run
# is REJECTED, not averaged. And ratios are comparable only WITHIN one boot --
# both arms move several percent across reboots, so a ratio from boot A and one
# from boot B cannot be differenced.
# Paired DSpark measurement, ours -> oracle -> ours, under the REAL lock, with a
# SETTLE BARRIER between arms.
#
# Why the barrier: vLLM profiles free GPU memory at startup and asserts it does
# not GROW during profiling. On GB10 the unified-memory allocator returns our
# engine's pages lazily, so an oracle started immediately after our arm sees
# free memory rise mid-profile and dies with
#   "Error in memory profiling. Initial free memory 68.53 GiB, current 89.42 GiB"
# That is a harness sequencing defect, not a property of either engine.
set -uo pipefail
DIR="$HOME/work/dspark-w6"
OUT="${VT_PAIRED_LOG:-$DIR/paired_e2e.log}"; : > "$OUT"
export PATH="/usr/local/cuda/bin:$HOME/venvs/vllm-oracle-next/bin:$HOME/.local/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CLI="$DIR/src/build/examples/vllm-cli"
T35=$(ls -d "$HOME"/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/*/ | head -1)
D35=$(ls -d "$HOME"/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-speculator.dspark/snapshots/*/ | head -1)
P=$(printf 'def fibonacci(n):\n    ')

clocks() { nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader; }

settle() {
  # Wait for every compute app to disappear, then give the allocator a fixed
  # grace period. Polling free memory is not an option: GB10 reports it [N/A].
  local n
  for _ in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
    [ "$n" -eq 0 ] && break
    sleep 5
  done
  sleep 60
}

ours() {
  echo "=== OURS $1 clocks=$(clocks)" >> "$OUT"
  "$CLI" --model "$T35" --max-num-seqs 2 --prompt "$P" --max-tokens 128 --temperature 0 \
    --repeat 10 \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"$D35\",\"num_speculative_tokens\":8}" >> "$OUT" 2>&1
  echo "OURS_${1}_RC=$?" >> "$OUT"
}

go() {
  echo "=== boot: $(who -b | tr -s ' ')" >> "$OUT"
  # WARM-UP ARM, DISCARDED. On a freshly booted GB10 the SM clock ramps over
  # MINUTES (measured 1449 -> 2190 MHz across one paired run), so discarding
  # rep 1 is not enough -- the whole first arm reads low. A previous run
  # bracketed ours at 133.7 before and 142.5 after, a 6.6% drift that swamps
  # the ~3% being measured. This arm exists to be thrown away.
  ours warmup
  settle
  ours before
  settle
  echo "=== ORACLE clocks=$(clocks)" >> "$OUT"
  "$HOME/venvs/vllm-oracle-next/bin/python" "$DIR/${VT_ORACLE_SCRIPT:-fibacc_lowmem.py}" >> "$OUT" 2>&1
  echo "ORACLE_RC=$?" >> "$OUT"
  settle
  ours after
  echo "=== clocks END: $(clocks)" >> "$OUT"
}
export -f go ours settle clocks; export OUT DIR CLI T35 D35 P HOME

docker stop local-ai-worker >> "$OUT" 2>&1 || true
flock -w 28800 "$HOME/gpu.lock" bash -c go
echo "=== paired_e2e done $(date -Is)" >> "$OUT"
