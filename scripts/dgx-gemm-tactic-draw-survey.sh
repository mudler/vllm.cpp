#!/bin/bash
# GEMM kernel-selection draw survey -- the lease-side half of #2750/#2751/#2752.
#
# WHAT THIS RUNS, AND WHAT IT DELIBERATELY DOES NOT
# -------------------------------------------------
# Three filed issues ask for one measurement each, and none of them asks for a
# feature. This script is the part that needs the device; the part that DECIDES
# is `tools/bench/gemm_tactic_draw_survey.py`, which is standard library only
# and is exercised without a GPU by `--dry-run`. That split is the polarity
# `gpu_clock_state.py` and `resumable_legs.py` already chose here, for the same
# reason: the half that judges must be checkable where the hardware is not.
#
#   #2750  KERNEL-GEMM-BF16        is cuBLASLt algo selection stable ACROSS
#                                  processes, or only within one?
#   #2751  KERNEL-GEMM-NVFP4-W4A4  how wide is the NVFP4 tactic draw spread, in
#                                  identity AND in speed?
#   #2752  KERNEL-GEMM-NVFP4-W4A4  which draw would we ship as a pinned GB10
#                                  artifact? (blocked on #2751 by its own text)
#
# BOTH INSTRUMENTS ALREADY EXIST AND NEITHER IS BUILT HERE.
# `VT_GEMM_ALGO_LOG=1` reaches `MaybeLogGemmAlgo` (src/vt/cuda/cuda_matmul.cu:248)
# and emits ONE line per unique (shape, dtype-combo, epilogue) selection, deduped
# within a process by `LogOncePerKey` (src/vt/cuda/gemm_algo_log.h). Diffing those
# line SETS across fresh processes IS #2750's experiment, and no new code is owed
# for it. `[VT_FP4_CACHE] prepared/complete/selected`
# (src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:610,657,673) reports the
# loaded/tuned/rejected/saved counts, the metadata fingerprint and the whole
# selected plan map; `VT_FP4_AUTOTUNE_CACHE_PATH` gives each process its own
# document and `VT_FP4_AUTOTUNE_CACHE_READONLY=1` freezes it
# (src/vt/cuda/nvfp4_persistent_cache.cpp:542-561).
#
# WHY THE DRAW IS ALREADY DISJOINT FROM THE SCORE
# -----------------------------------------------
# #2751 refuses a selector that picks a plan on the workload it will later be
# scored on, and so does `.agents/benchmarking.md`. On this lane the disjointness
# is STRUCTURAL rather than a discipline we have to keep: the tuning pass is the
# pre-serve NVFP4 warmup at `src/vllm/entrypoints/model_loader.cpp:2388-2410`,
# which submits ONE synthetic dummy request of `max_num_batched_tokens` repeated
# tokens and completes the tuner inside `Nvfp4AutotuneWarmupScope`. No prompt of
# the scoring workload has been seen when the draw is taken. What is NOT
# structural, and is enforced below, is that `--max-num-batched-tokens` be held
# constant across draws: it is what decides which plan keys get tuned, so a draw
# taken at a different value tuned a different key set and compares nothing.
#
# NO DRAW IS CHOSEN BY ITS OWN SPEED. `select_shipping_draw` in the Python half
# returns a draw only when the draws are performance-EQUIVALENT, and then it
# returns the first in draw order -- a rule fixed before any number existed.
#
# ONE TACTIC-SET ARM PER EVIDENCE ROOT, AND THE ARM IS SET RATHER THAN INHERITED
# -----------------------------------------------------------------------------
# `Fp4FullTacticsEnabled()` (`src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu:189-195`)
# reads `value == nullptr || value[0] != '0'`, so `VT_FP4_FULL_TACTICS` is
# DEFAULT ON. The shipped arm is 32 candidates chosen by pure argmin over
# per-candidate means (`:758-764`). The non-default `=0` arm is 4 candidates AND
# carries a variance damper (`:770-776`): a tactic displaces the fixed baseline
# only if it beats it by more than 1%, otherwise the baseline is kept.
#
# The arms therefore SELECT BY DIFFERENT RULES and their draw spreads are not
# one population. `--tactic-set full|w1` picks one, the Python half refuses a
# root that mixes them (exit 81), and the scoring legs are given the SAME arm as
# the draw they replay -- the tactic-set version is part of the cache metadata,
# so a frozen map replayed under the other arm is rejected rather than used.
#
# THE TUNER'S OWN TIMING IS A DIAGNOSTIC, NOT AN AXIS
# ---------------------------------------------------
# `VT_FP4_AUTOTUNE_VERBOSE=1` prints `-> id=%d %s (%.1f us)` per tuned key
# (`:777-788`). That figure is `timings[chosen] * 1000`: ONE mean over ten
# iterations, because `TimeCandidate` (`:322-363`) wraps all ten in a SINGLE
# `cudaEvent` pair and divides. No per-iteration sample exists in the tree, so
# no median or minimum over iterations is computable and a robust-statistic
# selector would have to ADD instrumentation rather than swap a reduction. This
# harness records the number and refuses to rank draws by it: it is produced by
# the instrument under suspicion, on the tuner's own warmup shapes, which is the
# workload the draw was taken on.
#
# THE HOST DECIDES THE SHAPE OF THIS SCRIPT
# -----------------------------------------
# `dgx.casa` went down four times in one session (#545) and has crashed roughly
# hourly under a long ladder, so a sequence long enough to answer these questions
# is longer than this host's MTBF. Every phase therefore has an on-disk marker,
# every draw has its own DONE file, the scoring legs go through
# `tools/bench/c8_leg_runner.py` (append-on-completion ledger, resume, cross-boot
# refusal, terminal control), and evidence is mirrored to the share as it lands.
# A re-run after a crash pays for what was lost and nothing else.
#
# FOUR LEASE FACTS SHAPE THE BUILD, AND THEY ARE MEASURED, NOT ASSUMED
# (.agents/environment.md):
#   * no CUDA toolkit is preinstalled, so [A] resolves or repairs one;
#   * `/workspace` is CIFS with `nounix`, so it stores NO symlink and serves
#     `file_mode=0664` -- a binary copied off it arrives non-executable and an
#     `nvcc` that cannot run reads as an `nvcc` that is absent;
#   * unconstrained parallelism has OOM-rebooted this box, so the build is -j 4;
#   * the SM clock CANNOT be pinned inside a lease (`LGC_RC=4`, #1354). It can
#     only be SAMPLED, so a pairing may be refused on within-run spread with no
#     lever to fix it. That refusal is a RESULT and this script records it.
#
# NO CHECKPOINT PATH IS DEFAULTED. `--model` is required and the script refuses
# without it. A checkpoint path written in a repository document is another
# run's resolved value, not a default.
#
# USAGE
# Submitted as one `rc` job, never inlined into `rc run --` beyond the `bash`
# call, because a detaching client kills the job:
#
#   rc run -d dgx:gpu0 --max-runtime 300m --idle-timeout 30m \
#     -- bash /workspace/gemm-draw-survey/run.sh
#
# where `run.sh` is a two-line staged wrapper that unpacks the source into /tmp
# and calls this file. The heartbeat below is what keeps `--idle-timeout` from
# firing during the silent build and load phases.
#
#   bash scripts/dgx-gemm-tactic-draw-survey.sh \
#        --evidence /workspace/gemm-draw-survey/<stamp> \
#        --src /workspace/gemm-draw-survey/src.tar.gz \
#        --model /workspace/ckpt/<nvfp4-checkpoint> \
#        [--draws 8] [--score-reps 3] [--concurrency 2] [--phase all] \
#        [--tactic-set full|w1]
#
# `--phase` is one of: all build draw score reduce. `--score-leg ARM` is the
# INTERNAL re-entry the leg runner calls; do not pass it by hand.
set -u

T0=$(date +%s)
say() { echo "[gtds +$(( $(date +%s) - T0 ))s] $*"; }
die() { echo "FATAL: $2" >&2; exit "$1"; }

# Exit codes. Named so a failed job says WHICH precondition failed in its status.
E_USAGE=2
E_SRC=31
E_CFG=33
E_BUILD=34
E_ARTEFACT=35
E_CUTLASS=36
E_CKPT=37
E_TOOLKIT=38
E_INSTRUMENT_ABSENT=40   # the SOURCE does not carry the instruments this measures
E_PREFLIGHT=41           # draw00 did not produce usable instrument output
E_LEG_NOT_FROZEN=78      # matches the Python half's EXIT_LEG_NOT_FROZEN
E_LEG_INCOMPLETE=80      # the leg ran and did not complete its requests

SELF="$(cd "$(dirname "$0")" 2>/dev/null && pwd)/$(basename "$0")"

EV_SHARE=""
SRC_IN=""
MODEL=""
DRAWS=8
# THREE, not two. The speed half's within-draw repeat spread is the floor a
# draw-to-draw gap is compared against, and a spread over two legs is one
# difference rather than an estimate of anything. `speed_spread` refuses a run
# with fewer than three legs per draw (INCOMPARABLE), so two would buy a lease
# and return no verdict.
SCORE_REPS=3
CHECK_ART=""
CONCURRENCY=2
NUM_PROMPTS=32
INPUT_LEN=512
OUTPUT_LEN=64
SEED=0
MAX_BATCHED=8192
PHASE=all
TACTIC_SET=full
SCORE_LEG=""
LOCAL_ROOT=/tmp/gtds

while [ $# -gt 0 ]; do
  case "$1" in
    --evidence)      EV_SHARE=${2:?}; shift 2 ;;
    --src)           SRC_IN=${2-}; shift 2 ;;
    --model)         MODEL=${2:?}; shift 2 ;;
    --draws)         DRAWS=${2:?}; shift 2 ;;
    --score-reps)    SCORE_REPS=${2:?}; shift 2 ;;
    --concurrency)   CONCURRENCY=${2:?}; shift 2 ;;
    --num-prompts)   NUM_PROMPTS=${2:?}; shift 2 ;;
    --input-len)     INPUT_LEN=${2:?}; shift 2 ;;
    --output-len)    OUTPUT_LEN=${2:?}; shift 2 ;;
    --seed)          SEED=${2:?}; shift 2 ;;
    --max-num-batched-tokens) MAX_BATCHED=${2:?}; shift 2 ;;
    --phase)         PHASE=${2:?}; shift 2 ;;
    --tactic-set)    TACTIC_SET=${2:?}; shift 2 ;;
    --score-leg)     SCORE_LEG=${2:?}; shift 2 ;;
    --check-artefacts) CHECK_ART=${2:?}; shift 2 ;;
    --local-root)    LOCAL_ROOT=${2:?}; shift 2 ;;
    -h|--help)       sed -n '2,120p' "$0"; exit 0 ;;
    *)               die "$E_USAGE" "unknown argument '$1'" ;;
  esac
done

[ -n "$EV_SHARE" ] || die "$E_USAGE" "--evidence is required"
[ -n "$MODEL" ] || die "$E_USAGE" "--model is required; this harness NEVER defaults a checkpoint path"
case "$TACTIC_SET" in full|w1) ;; *) die "$E_USAGE" "--tactic-set must be full or w1, not '$TACTIC_SET'" ;; esac
# `full` is the SHIPPED default (Fp4FullTacticsEnabled is on unless the value
# starts with '0'), so this mapping keeps the harness arm and the product arm
# the same thing under one name.
FULL_TACTICS=1; [ "$TACTIC_SET" = w1 ] && FULL_TACTICS=0

# THE EVIDENCE LIVES TWICE, ON PURPOSE.
# The engine publishes its cache document with mkstemp + fsync + atomic rename
# (nvfp4_persistent_cache.cpp:682-722, the mkstemp/write/fsync/rename body of
# WriteNativeCacheAtomically). That wants a real local filesystem, and
# `/workspace` is CIFS with `nounix`. So the run works under $EV_LOCAL and every
# phase MIRRORS to $EV_SHARE, which is what survives the box going down. A resume
# that finds a local tree missing restores it from the share first, so a crashed
# job restarts from the last completed draw rather than from zero.
EV_LOCAL="$LOCAL_ROOT/evidence"
BIN="$LOCAL_ROOT/bin"
SRC="$LOCAL_ROOT/src"
BLD="$LOCAL_ROOT/build"
PHASEDIR="$EV_LOCAL/phase"

mirror_out() {  # local -> share; `-L` because CIFS stores no symlink
  mkdir -p "$EV_SHARE" 2>/dev/null
  cp -rL "$EV_LOCAL"/. "$EV_SHARE"/ 2>/dev/null
}
mirror_in() {   # share -> local, on resume only
  [ -d "$EV_SHARE" ] || return 0
  mkdir -p "$EV_LOCAL"
  cp -rL "$EV_SHARE"/. "$EV_LOCAL"/ 2>/dev/null
  return 0
}
phase_done() { [ -f "$PHASEDIR/$1.ok" ]; }

# THE BINARY IS THE ARTEFACT; A SHARED LIBRARY IS OPTIONAL (#2912).
# The configuration this harness invokes (plain Release, no BUILD_SHARED_LIBS)
# emits a STATIC libvllm.a and links vllm-bench against it. The first real GB10
# run therefore built 855/855 with BUILD_RC=0 and then refused its own tree,
# because this predicate required a shared `libvllm.so.*` that no configuration
# here produces. Sets ART_BIN (required) and ART_LIB (empty when static).
resolve_artefacts() {
  ART_BIN=$(find "$1" -name vllm-bench -type f 2>/dev/null | head -1)
  ART_LIB=$(find "$1" -name 'libvllm.so.*' -type f 2>/dev/null | head -1)
  [ -n "$ART_BIN" ]
}

# `--check-artefacts DIR` is the INTERNAL gate handle: it runs the same
# predicate the build phase runs, so a test exercises the production code path
# rather than a transcription of it. The shell tests skip the build phase by
# writing phase/build.ok, which is why this branch shipped unreached.
if [ -n "$CHECK_ART" ]; then
  if resolve_artefacts "$CHECK_ART"; then
    echo "bin=$ART_BIN"
    echo "lib=${ART_LIB:-<none: vllm-bench is statically linked>}"
    exit 0
  fi
  die "$E_ARTEFACT" "build artefacts not found under $CHECK_ART: no vllm-bench"
fi
mark_phase() { mkdir -p "$PHASEDIR"; date -u +%Y-%m-%dT%H:%M:%SZ > "$PHASEDIR/$1.ok"; mirror_out; }

# ---------------------------------------------------------------------------
# INTERNAL RE-ENTRY: one scoring leg. `c8_leg_runner.py` calls this with {arm}
# substituted, and reads the metric out of what it prints.
#
# THE ARM OPENS ITS OWN CLOCK WINDOW. `.agents/benchmarking.md` is explicit that
# the window belongs to the arm and not to the shell that drives it, and #1657
# is the failure of getting that backwards. `exec` inside the subshell is
# load-bearing: without it `$!` is the SUBSHELL's pid, the TERM below kills the
# subshell, the sampler is orphaned into the NEXT leg and never writes its
# summary -- because the helper writes that summary only when the sampler STOPS
# (#2305).
# ---------------------------------------------------------------------------
if [ -n "$SCORE_LEG" ]; then
  ARM=$SCORE_LEG
  # THE SOURCE THE BUILD RECORDED, RESOLVED BEFORE ANYTHING USES IT. The leg is
  # re-entered with `--src ''` (the driver's own `--score-leg` command line
  # below passes exactly that), so without this `$SRC` is still the DEFAULT
  # `$LOCAL_ROOT/src` -- which only exists when the driver was given a `--src`
  # tarball. Run from an unpacked tree instead, the judge is somewhere else
  # entirely, `check-frozen` fails with "can't open file", and `|| exit
  # $E_LEG_NOT_FROZEN` reports that as 78: THE CODE THAT MEANS A LEG RE-TUNED.
  # A missing judge and a leg that re-tuned are different repairs.
  [ -f "$PHASEDIR/src.path" ] && SRC=$(cat "$PHASEDIR/src.path")
  SURVEY="$SRC/tools/bench/gemm_tactic_draw_survey.py"
  [ -f "$SURVEY" ] || die "$E_SRC" "the judging half is missing at $SURVEY; a leg cannot assert its own frozen control without it, and reporting that as $E_LEG_NOT_FROZEN would name the wrong fault"
  DRAWDIR="$EV_LOCAL/draws/$ARM"
  CACHE="$DRAWDIR/autotune_configs.json"
  [ -s "$CACHE" ] || die "$E_USAGE" "no cache document for arm '$ARM' at $CACHE"
  # The number of plans the frozen document must install. Read from the DRAW's
  # own emitted map, not from a constant: a constant would still be 64 on a
  # model whose shape set changed, and the frozen check would pass on a partial
  # install.
  PLANS=$(grep -c '\[VT_FP4_CACHE\] selected' "$DRAWDIR/stderr.log" 2>/dev/null)
  case "$PLANS" in ''|*[!0-9]*) PLANS=0 ;; esac
  [ "$PLANS" -gt 0 ] || die "$E_USAGE" "arm '$ARM' recorded no selected plans"

  # Globbed, not `ls`-counted: the leg index has to be the NEXT one after a
  # crash, and an `ls` that fails prints nothing, which would restart the count
  # at 1 and overwrite the leg that already ran.
  N=1
  for existing in "$EV_LOCAL/score/$ARM-"*; do
    [ -d "$existing" ] && N=$(( N + 1 ))
  done
  D="$EV_LOCAL/score/$ARM-$N"
  mkdir -p "$D"

  ( cd "$SRC" && exec python3 -m tools.bench.gpu_clock_state sample \
      --output "$D/clock.jsonl" --summary "$D/clock.json" --interval 2 \
      > "$D/clock.stdout" 2>&1 ) &
  CLK=$!

  VT_GEMM_ALGO_LOG=1 \
  VT_FP4_PERSISTENT_CACHE=1 \
  VT_FP4_AUTOTUNE_CACHE_PATH="$CACHE" \
  VT_FP4_AUTOTUNE_CACHE_READONLY=1 \
  VT_FP4_FULL_TACTICS="$FULL_TACTICS" \
  VT_FP4_AUTOTUNE_VERBOSE=1 \
  LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}" \
    "$BIN/vllm-bench" --model "$MODEL" \
      --num-prompts "$NUM_PROMPTS" --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" --concurrency "$CONCURRENCY" \
      --seed "$SEED" --temperature 0 \
      --max-num-batched-tokens "$MAX_BATCHED" > "$D/leg.log" 2>&1
  RC=$?
  kill -TERM "$CLK" 2>/dev/null; wait "$CLK" 2>/dev/null

  echo "leg arm=$ARM index=$N rc=$RC plans=$PLANS" > "$D/leg.status"
  [ -s "$D/clock.json" ] || echo "WARNING: the sampler wrote no clock summary for $ARM-$N (#2305)" >> "$D/leg.status"
  mirror_out
  [ "$RC" = 0 ] || { cat "$D/leg.status"; tail -30 "$D/leg.log"; exit "$RC"; }

  # THE FROZEN CONTROL, ON TWO WITNESSES. Without it this phase is N more DRAWS
  # wearing the label of a replay: a leg that re-tuned measured a plan map nobody
  # recorded, and its number would be attributed to the arm it was asked about.
  # `check-frozen` requires the runtime's own `tuned=0` AND zero
  # `[VT_FP4_AUTOTUNE]` selection lines, which is why the leg runs with
  # `VT_FP4_AUTOTUNE_VERBOSE=1` even though a frozen leg should print none.
  # The arm is passed through as well: the tactic-set version is part of the
  # cache metadata, so a map drawn under one arm and replayed under the other is
  # rejected by `ParseNativeCache` rather than silently used.
  # `--record` writes THIS LEG's control document, pass or fail, one file per
  # leg. `reduce` globs `score/*/frozen.json` and refuses (78) when a leg that
  # contributed a number carries no passing record -- without which the report
  # could carry EQUIVALENT and "ship draw00" over legs that re-tuned. One file
  # per writer, never one shared control file every leg has to append to.
  python3 "$SURVEY" check-frozen \
      --log "$D/leg.log" --expected-plans "$PLANS" \
      --record "$D/frozen.json" --leg "$ARM-$N"
  FZ=$?
  mirror_out
  [ "$FZ" = 0 ] || exit "$E_LEG_NOT_FROZEN"

  # EQUAL TIMES ARE NOISE; EQUAL COUNTS ARE IDENTITY (.agents/benchmarking.md).
  # A leg that completed fewer requests than it was given is not a slower leg,
  # it is a different workload, so it yields no number rather than a low one.
  GOT=$(grep -oE 'Successful requests:[[:space:]]+[0-9]+' "$D/leg.log" | grep -oE '[0-9]+$' | head -1)
  case "$GOT" in ''|*[!0-9]*) GOT=-1 ;; esac
  [ "$GOT" = "$NUM_PROMPTS" ] || { echo "leg completed $GOT of $NUM_PROMPTS requests"; exit "$E_LEG_INCOMPLETE"; }

  # The metric line the runner's regex reads. Printed only on a leg that passed
  # every control above, so a refused leg cannot contribute a number.
  grep -E 'Total token throughput|Output token throughput|Mean TPOT|Successful requests' "$D/leg.log"
  exit 0
fi

# ---------------------------------------------------------------------------
# THE HEARTBEAT, AND IT IS NOT OPTIONAL ON THIS FLEET.
# `rc run --idle-timeout` is OUTPUT-based, and `--idle-timeout 0` selects the
# DEVICE DEFAULT rather than disabling the kill, so the heartbeat is the remedy
# and the flag is not (`scripts/rc-sglang-oracle-lease.sh:88-92`, which is where
# this shape comes from). Every long phase of this driver is SILENT on stdout:
# `cmake` and `ninja` write to log files, a model load prints nothing here, and
# one draw is minutes of tuning. Without this loop the job is killed mid-build
# and the lease buys nothing.
#
# Started only on the driver path. A scoring leg runs INSIDE this job, so a leg
# that started its own would double the output and outlive the leg.
( while true; do
    sleep 60
    echo "### hb +$(( $(date +%s) - T0 ))s tmp_free=$(df -Pm /tmp 2>/dev/null | awk 'NR==2{print $4}')M"
  done ) &
HB=$!
cleanup_hb() { kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null; }
trap cleanup_hb EXIT INT TERM

mkdir -p "$EV_LOCAL" "$BIN"
mirror_in
mkdir -p "$PHASEDIR"
PROV="$EV_LOCAL/PROVENANCE"
{
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "rc_job=${RC_JOB_ID:-unknown} rc_device=${RC_DEVICE:-unknown}"
  echo "harness_sha256=$(sha256sum "$SELF" 2>/dev/null | awk '{print $1}')"
  echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
  echo "model=$MODEL"
  echo "draws=$DRAWS score_reps=$SCORE_REPS concurrency=$CONCURRENCY tactic_set=$TACTIC_SET"
  echo "num_prompts=$NUM_PROMPTS input_len=$INPUT_LEN output_len=$OUTPUT_LEN seed=$SEED"
  echo "max_num_batched_tokens=$MAX_BATCHED"
  nvidia-smi --query-gpu=name,driver_version,persistence_mode,clocks.max.sm --format=csv,noheader 2>/dev/null
} >> "$PROV"
say "EV_LOCAL=$EV_LOCAL EV_SHARE=$EV_SHARE"

# === [A] CUDA toolkit, resolved BEFORE anything probes it ==================
# Lifted from scripts/ltx25-dit-attn-flash-ab.sh, which records why each branch
# exists. The postcondition is asserted on BOTH names the linker can ask for:
# a staged toolkit that satisfies `find_package` can still fail 21 minutes later
# with 38 undefined references against `@libcudart.so.13`, because the SONAME is
# a THIRD name that CIFS did not carry (.agents/environment.md, #2220).
toolkit_ok() {
  [ -x "$1/bin/nvcc" ] || return 1
  local lib="$1/targets/sbsa-linux/lib"
  for stem in libcudart libcublasLt; do
    [ -e "$lib/$stem.so" ] || return 1
    ls "$lib/$stem.so."[0-9]* >/dev/null 2>&1 || return 1
  done
  return 0
}
resolve_toolkit() {
  local r
  for r in /usr/local/cuda /usr/local/cuda-13.0 "$LOCAL_ROOT/cudatk"; do
    if toolkit_ok "$r"; then TKLIB="$r"; return 0; fi
  done
  # Repair branch: rebuild the development link AND the SONAME link, letting
  # ldconfig read each object's own DT_SONAME so the name cannot disagree with
  # what the linker asks for. `ldconfig -n` does NOT create the `.so` link, so
  # both halves are needed.
  for r in /usr/local/cuda /usr/local/cuda-13.0; do
    [ -x "$r/bin/nvcc" ] || continue
    local lib="$r/targets/sbsa-linux/lib"
    [ -d "$lib" ] && [ -w "$lib" ] || continue
    ldconfig -n "$lib" 2>/dev/null
    ( cd "$lib" || exit 0
      for f in *.so.*; do
        case "$f" in *.a) continue ;; esac
        b=${f%%.so.*}
        [ -e "$b.so" ] || ln -sf "$f" "$b.so"
      done )
    if toolkit_ok "$r"; then TKLIB="$r"; return 0; fi
  done
  return 1
}

# === [B] source, and the PRECONDITION that it carries both instruments =====
stage_source() {
  if [ -z "$SRC_IN" ]; then
    # Running from an already-unpacked tree: this script's own repository.
    SRC="$(cd "$(dirname "$SELF")/.." && pwd)"
    return 0
  fi
  if [ -d "$SRC_IN" ]; then
    rm -rf "$SRC"; mkdir -p "$SRC"; cp -rL "$SRC_IN"/. "$SRC"/ || return 1
  else
    rm -rf "$SRC"; mkdir -p "$SRC"; tar xzf "$SRC_IN" -C "$SRC" || return 1
  fi
  return 0
}

# THE MARKER IS MIRRORED AND THE BINARY IS NOT, SO THE MARKER CAN OUTLIVE IT.
# `$BIN` and `$SRC` live under `$LOCAL_ROOT`, outside `$EV_LOCAL`, so `mirror_in`
# never restores them -- while `phase/build.ok` IS mirrored and comes back. A
# wiped `/tmp` under a surviving share therefore reads as "the build is done"
# with no binary anywhere, and the run dies further down on a missing file.
build_artefacts_ok() {
  [ -x "$BIN/vllm-bench" ] || return 1
  ls "$BIN"/libvllm.so.* >/dev/null 2>&1 || return 1
  local recorded
  recorded=$(cat "$PHASEDIR/src.path" 2>/dev/null) || return 1
  [ -n "$recorded" ] && [ -f "$recorded/tools/bench/gemm_tactic_draw_survey.py" ] || return 1
  return 0
}

if [ "$PHASE" = all ] || [ "$PHASE" = build ]; then
  if phase_done build && ! build_artefacts_ok; then
    # REFUSED RATHER THAN REBUILT, and the reason is the one-binary rule. Every
    # draw already recorded carries the previous binary's sha256, and the judge
    # refuses a run whose draws did not share one binary (exit 79). Rebuilding
    # into this evidence root would therefore spend the whole draw phase again
    # to be refused at the end. Remove the marker only when this root holds no
    # draws, or start a fresh evidence root.
    die "$E_ARTEFACT" "phase/build.ok came back from the share but $BIN/vllm-bench, its library, or the source at $(cat "$PHASEDIR/src.path" 2>/dev/null) did not: /tmp was wiped under a surviving evidence root. Draws already recorded here carry the previous binary's sha256 and the judge refuses two binaries (79), so start a FRESH --evidence root, or restore the binary, or delete $PHASEDIR/build.ok if this root holds no draws yet"
  fi
  if phase_done build; then
    say "=== [A-D] build already complete for this evidence root; skipping ==="
    SRC=$(cat "$PHASEDIR/src.path")
    TKLIB=$(cat "$PHASEDIR/toolkit.path")
  else
    say "=== [A] CUDA toolkit ==="
    TKLIB=""
    resolve_toolkit || die "$E_TOOLKIT" "no COMPLETE CUDA toolkit (nvcc + libcudart.so/.so.MAJOR + libcublasLt.so/.so.MAJOR)"
    export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
    say "CUDAToolkit_ROOT=$TKLIB"; nvcc --version | tail -2

    say "=== [B] source + instrument preconditions ==="
    stage_source || die "$E_SRC" "cannot stage the source from '$SRC_IN'"
    # BOTH instruments, because either one alone makes the run answer half a
    # question while looking complete. A tree without VT_GEMM_ALGO_LOG produces
    # zero lines, which the Python half refuses (exit 70) -- but it refuses it an
    # hour into the run rather than here.
    A_SITES=$(grep -c 'MaybeLogGemmAlgo' "$SRC/src/vt/cuda/cuda_matmul.cu" 2>/dev/null)
    F_SITES=$(grep -c 'VT_FP4_CACHE. selected' "$SRC/src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu" 2>/dev/null)
    C_SITES=$(grep -c 'VT_FP4_AUTOTUNE_CACHE_PATH' "$SRC/src/vt/cuda/nvfp4_persistent_cache.cpp" 2>/dev/null)
    case "$A_SITES" in ''|*[!0-9]*) A_SITES=0 ;; esac
    case "$F_SITES" in ''|*[!0-9]*) F_SITES=0 ;; esac
    case "$C_SITES" in ''|*[!0-9]*) C_SITES=0 ;; esac
    echo "  MaybeLogGemmAlgo sites:        $A_SITES (want >= 1)"
    echo "  [VT_FP4_CACHE] selected sites: $F_SITES (want >= 1)"
    echo "  cache-path override sites:     $C_SITES (want >= 1)"
    [ "$A_SITES" -ge 1 ] || die "$E_INSTRUMENT_ABSENT" "this tree has no cuBLASLt algo log; #2750 cannot be measured with it"
    [ "$F_SITES" -ge 1 ] || die "$E_INSTRUMENT_ABSENT" "this tree emits no selected-plan map; #2751 cannot be measured with it"
    [ "$C_SITES" -ge 1 ] || die "$E_INSTRUMENT_ABSENT" "this tree has no per-process cache-path override; the draws would share one document"

    say "=== [C] cutlass (resolved, never fetched) ==="
    CUT=""
    for c in /cutlass /workspace/cutlass "$LOCAL_ROOT/cutlass"; do
      [ -f "$c/include/cutlass/cutlass.h" ] && CUT="$c" && break
    done
    if [ -z "$CUT" ] && [ -f /workspace/cutlass-v4.5.0.tar.gz ]; then
      mkdir -p "$LOCAL_ROOT/cutlass" && tar xzf /workspace/cutlass-v4.5.0.tar.gz -C "$LOCAL_ROOT/cutlass" \
        && CUT="$LOCAL_ROOT/cutlass"
    fi
    [ -n "$CUT" ] && [ -f "$CUT/include/cutlass/cutlass.h" ] || die "$E_CUTLASS" "no CUTLASS tree; the NVFP4 arm cannot build"
    say "CUTLASS_DIR=$CUT"

    say "=== [D] configure + build (-j 4; unconstrained parallelism has OOM-rebooted this box) ==="
    cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
          -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" \
          > "$EV_LOCAL/configure.log" 2>&1
    CFG=$?; echo "CFG_RC=$CFG"
    grep -iE 'CUDA target arch|cutlass|CUDAToolkit' "$EV_LOCAL/configure.log" | head -8
    [ "$CFG" = 0 ] || { tail -40 "$EV_LOCAL/configure.log"; mirror_out; die "$E_CFG" "configure failed"; }

    ninja -C "$BLD" -j 4 vllm-bench > "$EV_LOCAL/build.log" 2>&1
    B=$?; echo "BUILD_RC=$B compile_errors=$(grep -ciE ' error: ' "$EV_LOCAL/build.log")"
    tail -15 "$EV_LOCAL/build.log"
    [ "$B" = 0 ] || { mirror_out; die "$E_BUILD" "build failed"; }

    resolve_artefacts "$BLD" || { mirror_out; die "$E_ARTEFACT" "build artefacts not found under $BLD: no vllm-bench"; }
    GEN=$ART_BIN; LIB=$ART_LIB
    cp -f "$GEN" "$BIN"/ && chmod 0755 "$BIN/vllm-bench"
    if [ -n "$LIB" ]; then
      cp -f "$LIB" "$BIN"/ && chmod 0755 "$BIN/$(basename "$LIB")"
      ( cd "$BIN" && SO=$(basename "$LIB"); ln -sf "$SO" libvllm.so.0; ln -sf "$SO" libvllm.so )
    else
      say "no shared libvllm under $BLD; vllm-bench is statically linked (#2912)"
    fi
    BINSHA=$(sha256sum "$BIN/vllm-bench" | awk '{print $1}')
    say "ONE BINARY for every draw and every leg: sha256=$BINSHA"
    { echo "binary_sha256=$BINSHA"; echo "cutlass=$CUT"; echo "toolkit=$TKLIB"; } >> "$PROV"
    echo "$SRC" > "$PHASEDIR/src.path"; echo "$TKLIB" > "$PHASEDIR/toolkit.path"
    mark_phase build
  fi
fi

[ -f "$PHASEDIR/src.path" ] && SRC=$(cat "$PHASEDIR/src.path")
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
SURVEY="$SRC/tools/bench/gemm_tactic_draw_survey.py"
[ -f "$SURVEY" ] || die "$E_SRC" "the judging half is missing at $SURVEY"

# === [E] checkpoint ========================================================
# `--model` is taken verbatim. When it names a path on the share, stage it once
# through the tree's own idempotent, manifest-verified copier: a 22 GB or 67 GB
# checkpoint read over CIFS pays the bandwidth on every shard on EVERY process,
# and this run starts N + N*R of them (#1807).
if [ "$PHASE" = all ] || [ "$PHASE" = draw ]; then
  case "$MODEL" in
    /workspace/*)
      LOCAL_CKPT="$LOCAL_ROOT/ckpt/$(basename "$MODEL")"
      if ! phase_done ckpt; then
        say "=== [E] staging the checkpoint off CIFS ==="
        bash "$SRC/scripts/rc-stage-checkpoint.sh" "$MODEL" "$LOCAL_CKPT" \
          || die "$E_CKPT" "checkpoint staging refused $MODEL"
        mark_phase ckpt
      fi
      MODEL="$LOCAL_CKPT"
      say "model staged at $MODEL"
      ;;
  esac
  [ -d "$MODEL" ] || [ -f "$MODEL" ] || die "$E_CKPT" "--model '$MODEL' does not exist"
fi

# === [P] preflight: ONE draw, then assert the instruments actually ran =====
# A silent instrument reads as a result. Paying for N draws and then discovering
# that VT_GEMM_ALGO_LOG produced nothing spends the lease to learn a fact one
# process could have told us.
if [ "$PHASE" = all ] || [ "$PHASE" = draw ]; then
  if ! phase_done preflight; then
    say "=== [P] preflight draw ==="
    ( cd "$SRC" && python3 "$SURVEY" draw --evidence "$EV_LOCAL" --bench "$BIN/vllm-bench" \
        --model "$MODEL" --draws 1 --num-prompts "$NUM_PROMPTS" --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" --concurrency "$CONCURRENCY" --seed "$SEED" \
        --max-num-batched-tokens "$MAX_BATCHED" --tactic-set "$TACTIC_SET" \
        --mirror "$EV_SHARE" )
    P=$?
    mirror_out
    [ "$P" = 0 ] || die "$E_PREFLIGHT" "the preflight draw did not produce usable instrument output (survey exit $P); nothing after this point could be trusted"
    mark_phase preflight
  fi

  if ! phase_done draws; then
    say "=== [G] $DRAWS independent draws, each in a fresh process ==="
    # `--mirror` makes the judge copy the evidence root to the share AFTER EACH
    # DRAW. Mirroring when the phase returns loses every completed draw to a
    # crash inside it, and the draw phase is hours of model loads on a box that
    # has gone down four times in one session (#545).
    ( cd "$SRC" && python3 "$SURVEY" draw --evidence "$EV_LOCAL" --bench "$BIN/vllm-bench" \
        --model "$MODEL" --draws "$DRAWS" --num-prompts "$NUM_PROMPTS" --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" --concurrency "$CONCURRENCY" --seed "$SEED" \
        --max-num-batched-tokens "$MAX_BATCHED" --tactic-set "$TACTIC_SET" \
        --mirror "$EV_SHARE" )
    G=$?
    mirror_out
    [ "$G" = 0 ] || die "$G" "the draw phase refused its own evidence (see draw-preconditions.json)"
    mark_phase draws
  fi
fi

# === [S] score each draw, frozen, interleaved, with a terminal control =====
# `c8_leg_runner.py` owns the plan/resume/fold/cross-boot rules; this supplies
# the arms and the command. A BLOCK of one arm followed by a block of the next
# measures the hour as much as the draw, which is why the plan interleaves and
# repeats the opening arm last.
if [ "$PHASE" = all ] || [ "$PHASE" = score ]; then
  mkdir -p "$EV_LOCAL/score"
  ARMS=""
  for d in "$EV_LOCAL/draws/draw"*; do
    [ -f "$d/DONE" ] && ARMS="$ARMS --arm $(basename "$d")"
  done
  [ -n "$ARMS" ] || die "$E_USAGE" "no completed draws to score"
  say "=== [S] scoring$ARMS, $SCORE_REPS legs per arm ==="
  # shellcheck disable=SC2086
  # `-m`, NOT a path. c8_leg_runner does `from tools.bench.resumable_legs import
  # ...`, so the repository ROOT must be on sys.path; `python3 tools/bench/...py`
  # puts `tools/bench` there instead and raises ModuleNotFoundError before it
  # reads an argument.
  ( cd "$SRC" && python3 -m tools.bench.c8_leg_runner \
      --ledger "$EV_LOCAL/score/legs.jsonl" \
      $ARMS --legs-per-arm "$SCORE_REPS" \
      --metric total_token_throughput \
      --metric-regex 'Total token throughput \(tok/s\):\s+([0-9.]+)' \
      --command "bash $SELF --score-leg {arm} --evidence $EV_SHARE --src '' --model $MODEL --local-root $LOCAL_ROOT --num-prompts $NUM_PROMPTS --input-len $INPUT_LEN --output-len $OUTPUT_LEN --concurrency $CONCURRENCY --seed $SEED --max-num-batched-tokens $MAX_BATCHED --tactic-set $TACTIC_SET" \
      > "$EV_LOCAL/score/leg-runner.log" 2>&1 )
  S=$?
  tail -40 "$EV_LOCAL/score/leg-runner.log"
  mirror_out
  # A NOT-ADMISSIBLE fold or a drifted terminal control is a RESULT, not a
  # harness failure: it says this box could not hold still long enough to answer
  # the speed half. It is recorded and the identity half still stands.
  [ "$S" = 0 ] || say "WARNING: the leg runner returned $S; the fold or the terminal control refused this sequence"
  mark_phase score
fi

# === [R] collect the clock windows and reduce ==============================
if [ "$PHASE" = all ] || [ "$PHASE" = reduce ]; then
  say "=== [R] clock windows + reduction ==="
  mkdir -p "$EV_LOCAL/score"
  python3 - "$EV_LOCAL" <<'PY' > "$EV_LOCAL/score/clock-windows.json"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "score"
legs = []
for summary in sorted(root.glob("draw*/clock.json")):
    try:
        legs.append({"leg": summary.parent.name,
                     "summary_path": str(summary),
                     "summary": json.loads(summary.read_text(encoding="utf-8"))})
    except (OSError, json.JSONDecodeError) as error:
        legs.append({"leg": summary.parent.name, "unreadable": str(error)})
missing = [str(p) for p in sorted(root.glob("draw*")) if not (p / "clock.json").is_file()]
print(json.dumps({
    "legs": legs,
    "legs_without_a_summary": missing,
    "note": ("Inside an rc lease the SM clock can be SAMPLED and not pinned "
             "(LGC_RC=4, #1354). A window refused on within-run spread has no "
             "lever to fix it, and that refusal is a result."),
}, indent=2, sort_keys=True))
PY
  ( cd "$SRC" && python3 "$SURVEY" reduce --evidence "$EV_LOCAL" \
      --out "$EV_LOCAL/REPORT.json" )
  R=$?
  mirror_out
  say "reduce exit $R; report at $EV_SHARE/REPORT.json"
  # MARKED ONLY ON A REDUCTION THAT PASSED. A marker written by a refused phase
  # makes the next run skip the work that did not happen, which is the same
  # polarity as the build marker that outlived its binary above.
  [ "$R" = 0 ] && mark_phase reduce
  exit "$R"
fi

mirror_out
say "done"
