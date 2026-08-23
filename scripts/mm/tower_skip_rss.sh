#!/usr/bin/env bash
# ENG-MM-INPUT-PIPELINE wave L3, issue #607 — the tower-skip RSS gate.
#
# Spec: .agents/specs/multimodal-track.md §1.5 L3, "The RSS gate, its VEHICLE,
# and its threshold". THE THRESHOLD IS DECLARED THERE AND IN THIS FILE BEFORE ANY
# NUMBER EXISTS. Do not edit it after reading a result.
#
# WHAT THIS MEASURES. Peak resident set size of one `vllm-server` process that
# loads `MuseGlimmerForConditionalGeneration`, answers one fixed text-only
# completion, and exits — with and without `--language-model-only`. The
# perception encoder is 809 of the checkpoint's 1436 tensors, 3.580 GiB on disk,
# held by our loader as HOST f32, so 7.161 GiB resident. That is the quantity
# the skip removes.
#
# WHY PEAK AND NOT STEADY STATE. The tower's bytes are paid during load. A
# steady-state figure taken after the allocator has returned pages would report a
# saving the box never saw.
#
# WHY TWO BUILD DIRECTORIES. An A/B that reuses one build directory measures one
# binary twice; identical results are then the expected outcome rather than a
# finding. Both directories are configured from the SAME commit, and the script
# prints each binary's sha256 so the two arms are provably different files
# running the same source.
#
# WHY THE SERVER AND NOT A BESPOKE DRIVER. `--language-model-only` is a
# production flag on a production entry point, already gated by
# `test_serve_mm_limits`. A hand-written driver would measure the driver.
#
# ─── RUNNING IT ─────────────────────────────────────────────────────────────
#
#   scripts/mm/tower_skip_rss.sh --checkpoint /mnt/nas_share/checkpoints/muse-glimmer-30b \
#                                --out /workspace/tower-skip-rss
#
# On a fleet device this runs INSIDE an `rc` lease — never over `ssh`. Stage this
# script on the NAS and invoke it by path, because `rc run` submits die with the
# detaching client. Build in /tmp and copy out with `cp -rL`: /workspace is CIFS
# and holds no symlink. Use `-j 4`: unconstrained parallelism has OOM-rebooted
# the GB10.
#
# The checkpoint path is REQUIRED and is not defaulted. `.env` names
# CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints, which did not exist on
# mudler-ubuntu-box on 2026-08-19; the NAS was mounted at /mnt/nas_share. That is
# a host condition, and a script that guessed between the two would attribute a
# missing checkpoint to the wrong thing.
#
# `--report-only DIR` recomputes the verdict from a previous run's logs without
# re-running anything.
set -u -o pipefail

# ─── The declared threshold. Do not edit after seeing a number. ─────────────
TOWER_ONDISK_BYTES=3843691520          # 809 vision tensors, read from the shard headers
TOWER_RESIDENT_BYTES=$((TOWER_ONDISK_BYTES * 2))   # loader widens bf16 -> host f32
MIN_SAVING_FRACTION_PCT=90             # >= 90% of the resident tower must disappear
DEFAULT_ARM_DRIFT_PCT=2                # default arm within 2% of the pre-L3 binary

PROMPT='The capital of France is'
MAX_TOKENS=16
PORT=${PORT:-18607}

CHECKPOINT=""
OUT=""
REPORT_ONLY=""
BASELINE_REF="edbc47ce0"   # pre-L3 head, for the second half of the threshold

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --report-only) REPORT_ONLY=$2; shift 2 ;;
    --baseline-ref) BASELINE_REF=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

report() {
  local dir=$1
  local a b saving need pct
  a=$(grep -h 'Maximum resident set size' "$dir/default.time" | tr -dc '0-9')
  b=$(grep -h 'Maximum resident set size' "$dir/lmo.time" | tr -dc '0-9')
  if [ -z "$a" ] || [ -z "$b" ]; then
    echo "RESULT: VOID — no 'Maximum resident set size' in one of the two logs."
    echo "  A missing instrument line is not a measurement of zero saving."
    return 3
  fi
  a=$((a * 1024)); b=$((b * 1024))   # /usr/bin/time -v reports kilobytes
  saving=$((a - b))
  need=$((TOWER_RESIDENT_BYTES * MIN_SAVING_FRACTION_PCT / 100))
  pct=$((saving * 100 / TOWER_RESIDENT_BYTES))
  printf 'peak RSS default            %14d B  %8.3f GiB\n' "$a" "$(echo "$a" | awk '{print $1/1073741824}')"
  printf 'peak RSS language-model-only%14d B  %8.3f GiB\n' "$b" "$(echo "$b" | awk '{print $1/1073741824}')"
  printf 'saving                      %14d B  %8.3f GiB  (%d%% of the resident tower)\n' \
         "$saving" "$(echo "$saving" | awk '{print $1/1073741824}')" "$pct"
  printf 'threshold                   %14d B  (%d%% of %d B)\n' \
         "$need" "$MIN_SAVING_FRACTION_PCT" "$TOWER_RESIDENT_BYTES"
  # The postcondition, asserted rather than assumed: the arm that was supposed to
  # skip must SAY it skipped. Without this the gate cannot tell a real saving
  # from a flag that did nothing on a quieter box.
  if ! grep -q 'multimodal towers NOT loaded' "$dir/lmo.log"; then
    echo "RESULT: VOID — the language-model-only arm never reported a skipped tower."
    echo "  It measured something, but not this change."
    return 3
  fi
  if grep -q 'multimodal towers NOT loaded' "$dir/default.log"; then
    echo "RESULT: VOID — the DEFAULT arm reported a skipped tower. The two arms"
    echo "  are not different, so the difference between them is not the skip."
    return 3
  fi
  if [ "$saving" -ge "$need" ]; then
    echo "RESULT: MET (first half). Second half — default arm vs the $BASELINE_REF"
    echo "  binary, within ${DEFAULT_ARM_DRIFT_PCT}% — is a separate run and is NOT asserted here."
    return 0
  fi
  echo "RESULT: FAILING. The axis stays open. Do not renegotiate the threshold."
  return 1
}

if [ -n "$REPORT_ONLY" ]; then
  report "$REPORT_ONLY"
  exit $?
fi

[ -n "$CHECKPOINT" ] || { echo "--checkpoint is required (no default; see the header)" >&2; exit 2; }
[ -d "$CHECKPOINT" ] || { echo "no such checkpoint directory: $CHECKPOINT" >&2; exit 2; }
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
command -v /usr/bin/time >/dev/null || { echo "/usr/bin/time is missing; install time" >&2; exit 2; }

mkdir -p "$OUT"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SHA=$(git -C "$REPO" rev-parse HEAD)
echo "commit          $SHA"
echo "checkpoint      $CHECKPOINT"
echo "checkpoint root USED: the path above. .env's CHECKPOINT_ROOT was NOT consulted."
echo "host            $(uname -a)"
echo "load            $(cat /proc/loadavg)"
echo "free            $(free -g | sed -n 2p)"

# TWO build directories, same commit, so the A/B cannot measure one binary twice.
for arm in a b; do
  d="/tmp/tower-skip-build-$arm"
  cmake -S "$REPO" -B "$d" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_BUILD_EXAMPLES=OFF > "$OUT/cmake-$arm.log" 2>&1 \
    || { echo "configure failed for arm $arm; see $OUT/cmake-$arm.log" >&2; exit 4; }
  ninja -C "$d" -j 4 vllm-server > "$OUT/build-$arm.log" 2>&1 \
    || { echo "build failed for arm $arm; see $OUT/build-$arm.log" >&2; exit 4; }
done
BIN_A=$(find /tmp/tower-skip-build-a -name vllm-server -type f | head -1)
BIN_B=$(find /tmp/tower-skip-build-b -name vllm-server -type f | head -1)
[ -n "$BIN_A" ] && [ -n "$BIN_B" ] || { echo "vllm-server not found in one of the build trees" >&2; exit 4; }
SHA_A=$(sha256sum "$BIN_A" | cut -d' ' -f1)
SHA_B=$(sha256sum "$BIN_B" | cut -d' ' -f1)
echo "binary A        $SHA_A  $BIN_A"
echo "binary B        $SHA_B  $BIN_B"
# The two builds come from one commit with one set of flags, so they SHOULD be
# byte-identical. Not asserted: the build directory path is embedded (`__FILE__`,
# debug prefixes), so a difference here is usually the path and not the code, and
# a hard stop on it would fail a correct run. Printed instead, and decorrelated
# below, which is the property that actually matters.
if [ "$SHA_A" = "$SHA_B" ]; then
  echo "binaries        IDENTICAL"
else
  echo "binaries        DIFFER (expected only if the embedded build path leaks in;"
  echo "                the A-B-B-A order below keeps this off the arm axis anyway)"
fi

# One arm of the measurement: start the server, wait for it to answer, ask for
# exactly $MAX_TOKENS greedy tokens, stop it. `/usr/bin/time -v` wraps the SERVER,
# so the peak it reports is the load phase plus that one completion.
run_arm() {
  local bin=$1 tag=$2; shift 2
  /usr/bin/time -v -o "$OUT/$tag.time" \
    "$bin" --model "$CHECKPOINT" --port "$PORT" --device cpu "$@" \
    > "$OUT/$tag.log" 2>&1 &
  local pid=$!
  local ready=0 i
  for i in $(seq 1 900); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then ready=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    echo "arm $tag: the server never became ready; see $OUT/$tag.log" >&2
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 5
  fi
  curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
       -d "{\"model\":\"muse-glimmer-30b\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" \
       > "$OUT/$tag.completion.json" 2>>"$OUT/$tag.log"
  local crc=$?
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  [ $crc -eq 0 ] || { echo "arm $tag: the completion request failed" >&2; return 5; }
  return 0
}

# A discarded first run warms the page cache, so the two arms see the same I/O
# state rather than the first one paying for the second.
echo "== warming the page cache (this run is DISCARDED) =="
run_arm "$BIN_A" warmup || exit 5

# The arm-to-binary assignment SWAPS between the two pairs. Pinning `default` to
# BIN_A and `lmo` to BIN_B in both pairs makes binary identity perfectly
# correlated with the arm, so any difference between the two binaries — a stale
# object, a different toolchain resolution, one tree that did not rebuild —
# arrives as a difference between the arms and is indistinguishable from the
# saving being measured. Swapping puts each binary on each arm once, so a
# binary-shaped effect shows up as disagreement BETWEEN the pairs (which the
# spread report already prints) rather than as the result.
echo "== A-B then B-A: each binary runs each arm exactly once =="
run_arm "$BIN_A" default            || exit 5
run_arm "$BIN_B" lmo --language-model-only || exit 5
run_arm "$BIN_B" default2           || exit 5
run_arm "$BIN_A" lmo2 --language-model-only || exit 5

echo
echo "== first pair =="
report "$OUT"; first=$?
echo
echo "== repeat pair (default2/lmo2), for the spread =="
cp "$OUT/default2.time" "$OUT/pair2-default.time" 2>/dev/null
cp "$OUT/lmo2.time" "$OUT/pair2-lmo.time" 2>/dev/null
grep -h 'Maximum resident set size' "$OUT/default2.time" "$OUT/lmo2.time" || true
echo
echo "logs and both .time files are under $OUT"
exit $first
