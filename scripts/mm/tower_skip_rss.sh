#!/usr/bin/env bash
# ENG-MM-INPUT-PIPELINE wave L3, issue #607 — the tower-skip RSS gate.
#
# Spec: .agents/specs/multimodal-track.md §1.5 L3, "The RSS gate, its VEHICLE,
# and its threshold". EVERY THRESHOLD IS DECLARED THERE AND IN THIS FILE BEFORE ANY
# NUMBER EXISTS. Do not edit one after reading a result.
#
# WHAT THIS MEASURES. Peak resident set size of one `vllm-server` process that
# loads a multimodal checkpoint, runs one fixed workload, and exits — with and
# without `--language-model-only`. The difference is the tower the skip removes.
#
# TWO MODELS, TWO SEPARATE DECLARATIONS. `--model-kind` selects which. The
# figures do NOT transfer between them and neither does the vehicle: the two
# checkpoints have different towers, different storage costs and different
# text-path capabilities. Each kind's numbers are derived from its own
# safetensors headers below, and each is repeated in the spec.
#
#   muse-glimmer  `MuseGlimmerForConditionalGeneration`, 56 G.
#                 The perception encoder is 809 of the checkpoint's 1436
#                 tensors, 3.580 GiB on disk, held by our loader as HOST f32
#                 (`MuseGlimmerVisionWeights` is `std::vector<float>`,
#                 muse_glimmer_vision.h:106-118), so 7.161 GiB resident.
#                 Workload: load, then one fixed 16-token greedy text
#                 completion.
#
#   qwen3-vl      `Qwen3VLForConditionalGeneration`, 8.3 GiB
#                 (`Qwen/Qwen3-VL-4B-Instruct`, revision
#                 ebb281ec70b05090aa6165b016eac8ec08e71b17). Read from the two
#                 shard headers on 2026-08-23: 713 tensors, ALL BF16,
#                 8875631616 B = 8.266 GiB, of which 315 carry the
#                 `model.visual.` prefix and total 830695424 B = 0.7736 GiB
#                 (9.3593%). `LoadQwen3VLVisionWeights`
#                 (qwen3_vl.cpp:437-480) reads EXACTLY those 315 — the
#                 enumeration it walks under `w.vision_cfg`, whose defaults
#                 (qwen3_vl_vision.h:34-46: depth 24, three
#                 deepstack_visual_indexes) equal this checkpoint's own
#                 `vision_config` — 3 top-level + 24 blocks x 12 + 6 merger +
#                 3 x 6 deepstack = 315, with no vision tensor unread and no
#                 read name absent. Every one of those reads goes through
#                 `LoadVisionF32` (qwen3_vl.cpp:79-90), which VT_CHECKs
#                 `dtype == "BF16"` and returns `std::vector<float>`, and every
#                 field of `Qwen3VLVisionWeights` /`VisionBlockWeights` /
#                 `VisionMergerWeights` (qwen3_vl_vision.h:60-82) is a
#                 `std::vector<float>`. So the widening is not partial: the
#                 resident cost is 2 x 830695424 = 1661390848 B = 1.5473 GiB.
#                 Workload: load to `/health` ONLY — see the note below.
#
# THE WIDENING IS ITSELF A DEFECT (#1359), and it is HALF of what either
# threshold measures. On both kinds the resident figure is twice the on-disk
# one because the tower is held as host f32 where the checkpoint ships bf16.
# A large saving here is therefore partly a large widening, and is NOT a
# statement that the tower is that big. #1359 is filed and owed; narrowing the
# storage would change the very quantity these thresholds are stated against,
# which is why it is not done first.
#
# WHY qwen3-vl RUNS NO COMPLETION. It cannot:
# `ForwardQwen3VLForConditionalGeneration` (qwen3_vl_registry.cpp:124-130)
# VT_CHECKs `input.mm.has_value()` and names text-only Qwen3-VL through this
# arch a MM-ENGINE-FORWARD residual, so a text completion against this
# checkpoint throws by design rather than measuring anything. The load phase is
# where the tower's bytes are paid, and `/health` cannot answer until
# `LoadedEngine::FromModelDir` has returned (server_main.cpp:1328-1351 runs the
# load, prints the skip line, and only then builds and binds the handlers), so
# readiness is a sufficient postcondition for the quantity at stake. The
# difference from the muse-glimmer vehicle is IDENTICAL ON BOTH ARMS of the
# same kind, so it stays off the arm axis, which is the axis being measured.
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
#   scripts/mm/tower_skip_rss.sh --checkpoint /mnt/nas_share/checkpoints/qwen3-vl-4b-instruct \
#                                --out /workspace/tower-skip-rss-qwen3vl
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
# `--model-kind` is resolved from the checkpoint's own `config.json`
# `architectures` when it is not given. It is NEVER defaulted: an unrecognised
# architecture is refused, because applying one model's declared threshold to
# another model's tower is the failure this whole file exists to avoid.
#
# `--report-only DIR` recomputes the verdict from a previous run's logs without
# re-running anything. It reads the kind from the `model-kind` file the run
# wrote into DIR; pass `--model-kind` to override, and it refuses when it has
# neither rather than assuming one.
set -u -o pipefail

# ─── The declared thresholds. Do not edit after seeing a number. ────────────
#
# Per kind, resolved by `declare_model` below. `MIN_SAVING_FRACTION_PCT` and
# `DEFAULT_ARM_DRIFT_PCT` are shared because the ARGUMENT for them is shared:
# ninety per cent, not a hundred, leaves room for allocator granularity and for
# the tower geometry that is still parsed — the construct half of
# construct-without-initialise — and neither of those is a property of which
# checkpoint is loaded.
MIN_SAVING_FRACTION_PCT=90             # >= 90% of the resident tower must disappear
DEFAULT_ARM_DRIFT_PCT=2                # default arm within 2% of the pre-L3 binary

MUSE_GLIMMER_TOWER_ONDISK_BYTES=3843691520   # 809 vision tensors, read from the shard headers
QWEN3_VL_TOWER_ONDISK_BYTES=830695424        # 315 `model.visual.*` tensors, read from the shard headers

PROMPT='The capital of France is'
MAX_TOKENS=16
PORT=${PORT:-18607}

CHECKPOINT=""
OUT=""
REPORT_ONLY=""
MODEL_KIND=""
BASELINE_REF="edbc47ce0"   # pre-L3 head, for the second half of the threshold

# Set by `declare_model`.
TOWER_ONDISK_BYTES=""
TOWER_RESIDENT_BYTES=""
SERVED_MODEL_NAME=""
WORKLOAD=""

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --report-only) REPORT_ONLY=$2; shift 2 ;;
    --model-kind) MODEL_KIND=$2; shift 2 ;;
    --baseline-ref) BASELINE_REF=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Resolve the per-kind declarations. The `*)` arm refuses rather than falling
# back, for the reason the header gives.
declare_model() {
  case "$1" in
    muse-glimmer)
      TOWER_ONDISK_BYTES=$MUSE_GLIMMER_TOWER_ONDISK_BYTES
      SERVED_MODEL_NAME="muse-glimmer-30b"
      WORKLOAD="completion"
      ;;
    qwen3-vl)
      TOWER_ONDISK_BYTES=$QWEN3_VL_TOWER_ONDISK_BYTES
      SERVED_MODEL_NAME="qwen3-vl-4b-instruct"
      WORKLOAD="load-only"
      ;;
    *)
      echo "unknown --model-kind '$1'. Declared kinds: muse-glimmer, qwen3-vl." >&2
      echo "  Each kind carries its OWN measured tower size and its own vehicle;" >&2
      echo "  there is no default, because one model's threshold does not describe" >&2
      echo "  another model's tower." >&2
      return 2
      ;;
  esac
  # The loader widens bf16 -> host f32 on BOTH kinds (#1359). This factor is the
  # defect, not an estimate: see the header.
  TOWER_RESIDENT_BYTES=$((TOWER_ONDISK_BYTES * 2))
  return 0
}

# `architectures[0]` out of the checkpoint's own config.json, mapped onto a
# kind. Used only when `--model-kind` was not given.
kind_from_checkpoint() {
  local cfg="$1/config.json"
  [ -f "$cfg" ] || { echo ""; return 0; }
  if grep -q 'MuseGlimmerForConditionalGeneration' "$cfg"; then
    echo "muse-glimmer"; return 0
  fi
  if grep -q 'Qwen3VLForConditionalGeneration' "$cfg"; then
    echo "qwen3-vl"; return 0
  fi
  echo ""
}

# ─── One pair of arms ───────────────────────────────────────────────────────
#
# `$1` the run directory, `$2` the DEFAULT arm's tag, `$3` the
# language-model-only arm's tag. Sets `PAIR_SAVING` to the pair's saving in
# bytes on a non-VOID return. Returns 0 MET, 1 FAILING, 3 VOID.
PAIR_SAVING=0
report_pair() {
  local dir=$1 dtag=$2 ltag=$3
  local a b saving need pct
  PAIR_SAVING=0
  a=$(grep -h 'Maximum resident set size' "$dir/$dtag.time" 2>/dev/null | tr -dc '0-9')
  b=$(grep -h 'Maximum resident set size' "$dir/$ltag.time" 2>/dev/null | tr -dc '0-9')
  if [ -z "$a" ] || [ -z "$b" ]; then
    echo "RESULT: VOID — no 'Maximum resident set size' in one of $dtag.time / $ltag.time."
    echo "  A missing instrument line is not a measurement of zero saving."
    return 3
  fi
  a=$((a * 1024)); b=$((b * 1024))   # /usr/bin/time -v reports kilobytes
  saving=$((a - b))
  need=$((TOWER_RESIDENT_BYTES * MIN_SAVING_FRACTION_PCT / 100))
  pct=$((saving * 100 / TOWER_RESIDENT_BYTES))
  printf 'peak RSS default (%-8s)  %14d B  %8.3f GiB\n' "$dtag" "$a" \
         "$(echo "$a" | awk '{print $1/1073741824}')"
  printf 'peak RSS lang-model-only (%-4s) %14d B  %8.3f GiB\n' "$ltag" "$b" \
         "$(echo "$b" | awk '{print $1/1073741824}')"
  printf 'saving                      %14d B  %8.3f GiB  (%d%% of the resident tower)\n' \
         "$saving" "$(echo "$saving" | awk '{print $1/1073741824}')" "$pct"
  printf 'threshold                   %14d B  (%d%% of %d B)\n' \
         "$need" "$MIN_SAVING_FRACTION_PCT" "$TOWER_RESIDENT_BYTES"
  # The postcondition, asserted rather than assumed: the arm that was supposed to
  # skip must SAY it skipped. Without this the gate cannot tell a real saving
  # from a flag that did nothing on a quieter box.
  if ! grep -q 'multimodal towers NOT loaded' "$dir/$ltag.log" 2>/dev/null; then
    echo "RESULT: VOID — the $ltag arm never reported a skipped tower."
    echo "  It measured something, but not this change."
    return 3
  fi
  if grep -q 'multimodal towers NOT loaded' "$dir/$dtag.log" 2>/dev/null; then
    echo "RESULT: VOID — the $dtag arm reported a skipped tower. The two arms"
    echo "  are not different, so the difference between them is not the skip."
    return 3
  fi
  PAIR_SAVING=$saving
  if [ "$saving" -ge "$need" ]; then
    echo "RESULT: MET (this pair)."
    return 0
  fi
  echo "RESULT: FAILING (this pair). The axis stays open. Do not renegotiate the threshold."
  return 1
}

# ─── The verdict, over BOTH pairs ───────────────────────────────────────────
#
# WHY BOTH PAIRS AND NOT JUST THE FIRST. The run swaps the arm-to-binary
# assignment between the pairs (A-B then B-A) precisely so a binary-shaped bias
# `d` — a stale object, a different toolchain resolution, one tree that did not
# rebuild — cannot masquerade as a saving: pair 1 then yields `true + d` and
# pair 2 `true - d`. Reading only pair 1, which is what this reporter did until
# #607's L3 review, throws that design away: the decorrelation was performed and
# never reached the verdict, and `d` arrived as the result exactly as if the
# swap had never happened.
#
# WHY BOTH MUST MEET THE THRESHOLD, RATHER THAN THEIR MEAN. The mean is the
# better ESTIMATOR — `((true + d) + (true - d)) / 2 == true` cancels `d`
# exactly — and it is printed below for that reason. It is the worse GATE. A
# mean passes on `true` alone, so an arbitrarily large `d` is admissible to it
# as long as it cancels, and the run that produced it contains no pair whose
# number describes the machine; the two halves disagree and the gate cannot say
# which to believe. Requiring both halves to clear the threshold passes only
# when `true - |d| >= need`, which is conservative in the one direction that
# matters: it cannot be talked into a pass by a bias, and it fails loudly on the
# disagreement instead of averaging it away. That is also why no separate spread
# threshold is declared — a `d` large enough to matter already fails a half, and
# a second number would be a second thing to renegotiate.
#
# Returns 0 MET, 1 FAILING, 3 VOID. VOID outranks FAILING: an unmeasured pair is
# not a measurement of a small saving.
report() {
  local dir=$1
  local r1 r2 s1 s2 mean spread
  echo "== pair 1: default vs language-model-only (binary A then binary B) =="
  report_pair "$dir" default lmo; r1=$?; s1=$PAIR_SAVING
  echo
  echo "== pair 2: the SWAPPED assignment (binary B then binary A) =="
  report_pair "$dir" default2 lmo2; r2=$?; s2=$PAIR_SAVING
  echo
  if [ "$r1" -eq 3 ] || [ "$r2" -eq 3 ]; then
    echo "RESULT: VOID — a pair is unmeasured, so the decorrelated verdict does not exist."
    return 3
  fi
  mean=$(( (s1 + s2) / 2 ))
  spread=$(( s1 > s2 ? s1 - s2 : s2 - s1 ))
  printf 'pair 1 saving               %14d B  %8.3f GiB\n' "$s1" \
         "$(echo "$s1" | awk '{print $1/1073741824}')"
  printf 'pair 2 saving               %14d B  %8.3f GiB\n' "$s2" \
         "$(echo "$s2" | awk '{print $1/1073741824}')"
  printf 'mean (estimator, cancels d) %14d B  %8.3f GiB\n' "$mean" \
         "$(echo "$mean" | awk '{print $1/1073741824}')"
  printf 'spread = 2|d|               %14d B  %8.3f GiB\n' "$spread" \
         "$(echo "$spread" | awk '{print $1/1073741824}')"
  if [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]; then
    echo "RESULT: MET (first half, BOTH pairs). Second half — default arm vs the"
    echo "  $BASELINE_REF binary, within ${DEFAULT_ARM_DRIFT_PCT}% — is a separate run and is NOT"
    echo "  asserted here."
    return 0
  fi
  echo "RESULT: FAILING. Pair 1 $([ "$r1" -eq 0 ] && echo MET || echo FAILING),"
  echo "  pair 2 $([ "$r2" -eq 0 ] && echo MET || echo FAILING). The axis stays open."
  if [ "$r1" -ne "$r2" ]; then
    echo "  The two pairs DISAGREE across the swapped arm-to-binary assignment, so"
    echo "  a binary-shaped bias of about $((spread / 2)) B is in this run. That is"
    echo "  what the swap exists to expose; it is not a smaller saving."
  fi
  echo "  Do not renegotiate the threshold."
  return 1
}

if [ -n "$REPORT_ONLY" ]; then
  if [ -z "$MODEL_KIND" ] && [ -f "$REPORT_ONLY/model-kind" ]; then
    MODEL_KIND=$(cat "$REPORT_ONLY/model-kind")
  fi
  if [ -z "$MODEL_KIND" ]; then
    echo "--report-only needs a model kind: $REPORT_ONLY holds no 'model-kind' file" >&2
    echo "  and --model-kind was not given. Refusing to pick one, because the" >&2
    echo "  threshold is per model." >&2
    exit 2
  fi
  declare_model "$MODEL_KIND" || exit 2
  echo "model kind      $MODEL_KIND"
  echo "resident tower  $TOWER_RESIDENT_BYTES B (2 x $TOWER_ONDISK_BYTES B on disk; the x2 is #1359)"
  report "$REPORT_ONLY"
  exit $?
fi

[ -n "$CHECKPOINT" ] || { echo "--checkpoint is required (no default; see the header)" >&2; exit 2; }
[ -d "$CHECKPOINT" ] || { echo "no such checkpoint directory: $CHECKPOINT" >&2; exit 2; }
[ -n "$OUT" ] || { echo "--out is required" >&2; exit 2; }
command -v /usr/bin/time >/dev/null || { echo "/usr/bin/time is missing; install time" >&2; exit 2; }

if [ -z "$MODEL_KIND" ]; then
  MODEL_KIND=$(kind_from_checkpoint "$CHECKPOINT")
fi
if [ -z "$MODEL_KIND" ]; then
  echo "cannot resolve a model kind from $CHECKPOINT/config.json, and --model-kind" >&2
  echo "  was not given. Declared kinds: muse-glimmer, qwen3-vl. Refusing to guess." >&2
  exit 2
fi
declare_model "$MODEL_KIND" || exit 2

mkdir -p "$OUT"
printf '%s\n' "$MODEL_KIND" > "$OUT/model-kind"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SHA=$(git -C "$REPO" rev-parse HEAD)
echo "commit          $SHA"
echo "checkpoint      $CHECKPOINT"
echo "model kind      $MODEL_KIND"
echo "workload        $WORKLOAD"
echo "resident tower  $TOWER_RESIDENT_BYTES B (2 x $TOWER_ONDISK_BYTES B on disk; the x2 is #1359)"
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

# One arm of the measurement: start the server, wait for it to answer, run the
# kind's fixed workload, stop it. `/usr/bin/time -v` wraps the SERVER, so the peak
# it reports is the load phase plus that workload.
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
  # `load-only` stops here, and readiness is the postcondition: `/health` cannot
  # answer before `LoadedEngine::FromModelDir` returned, and the tower's bytes
  # are paid inside it. See the header for why qwen3-vl cannot run a completion.
  local crc=0
  if [ "$WORKLOAD" = "completion" ]; then
    curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
         -d "{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" \
         > "$OUT/$tag.completion.json" 2>>"$OUT/$tag.log"
    crc=$?
  fi
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
# binary-shaped effect shows up as disagreement BETWEEN the pairs, which
# `report` reads and folds into the verdict.
echo "== A-B then B-A: each binary runs each arm exactly once =="
run_arm "$BIN_A" default            || exit 5
run_arm "$BIN_B" lmo --language-model-only || exit 5
run_arm "$BIN_B" default2           || exit 5
run_arm "$BIN_A" lmo2 --language-model-only || exit 5

echo
report "$OUT"; verdict=$?
echo
echo "logs and all four .time files are under $OUT"
exit $verdict
