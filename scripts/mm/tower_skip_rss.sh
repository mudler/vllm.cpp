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
# Run `--dry-run` first. It prints the exact cmake, ninja and server
# invocations this run will issue and refuses a plan that cannot produce a
# binary, and it costs nothing:
#
#   scripts/mm/tower_skip_rss.sh --dry-run --model-kind qwen3-vl
#
#   scripts/mm/tower_skip_rss.sh --checkpoint /workspace/ckpt/muse-glimmer-30b \
#                                --stage-to /tmp/tower-skip-ckpt \
#                                --out /workspace/tower-skip-rss
#
#   scripts/mm/tower_skip_rss.sh --checkpoint /workspace/ckpt/qwen3-vl-4b-instruct \
#                                --stage-to /tmp/tower-skip-ckpt \
#                                --out /workspace/tower-skip-rss-qwen3vl
#
# THE LEASED WORKER CANNOT SEE `/mnt/nas_share/checkpoints`, AND THIS FILE USED
# TO DOCUMENT THAT PATH. Measured on `dgx:gpu0` under a lease on 2026-08-23. The
# worker's only CIFS mount is `//192.168.68.102/Data on /workspace`, but what it
# exposes is NOT the share root: `ls /workspace` gives 81 entries and none of
# them is `checkpoints`, `datasets`, `models`, `bots`, `rc` or `loras`, the mount
# root cannot be escaped (`ls /workspace/../checkpoints` is empty), and there is
# no second NAS mount. Those 81 entries are the job directories that appear
# locally under `/mnt/nas_share/rc/`, confirmed by matching `a2d1` and `ckpt` on
# both sides — and this checkout's own host agrees: `/mnt/nas_share/rc/` holds
# exactly 81 entries, `//192.168.68.102/Data on /mnt/nas_share type cifs`. So:
#
#     worker `/workspace`  ==  local `/mnt/nas_share/rc`
#
# The staged checkpoints other sessions already use live at `/workspace/ckpt/`
# (locally `/mnt/nas_share/rc/ckpt/`), beside `manifests.log` and `*.copy.log`.
# That is the convention, not an invention of this file.
#
# THE MEASUREMENT MUST READ FROM LOCAL DISK, NOT CIFS. A run that streams weights
# over the mount measures the mount. `--checkpoint` therefore REFUSES a source on
# a CIFS/SMB filesystem, or under `/workspace`, unless `--stage-to DIR` is given;
# with it the tree is copied to local disk first and the run reads the copy.
#
# THE COPY IS VERIFIED BY A POSTCONDITION, NEVER BY `cp`'s EXIT STATUS. On this
# fleet a missing wrapper binary has already made a copy command print success
# and move nothing. `verify_stage` compares the two trees file by file — every
# regular file's path RELATIVE to its root and its byte SIZE, plus the file count
# and the total — and an empty destination fails it loudly rather than reading as
# a copy of a directory that happened to have no files.
#
# On a fleet device this runs INSIDE an `rc` lease — never over `ssh`. Stage this
# script on the NAS and invoke it by path, because `rc run` submits die with the
# detaching client. Build in /tmp and copy out with `cp -rL`: /workspace is CIFS
# and holds no symlink. Use `-j 4`: unconstrained parallelism has OOM-rebooted
# the GB10.
#
# The checkpoint path is REQUIRED and is not defaulted. `.env` names
# CHECKPOINT_ROOT=/usr/local/nas_share/checkpoints, which did not exist on
# mudler-ubuntu-box on 2026-08-19; the NAS was mounted at /mnt/nas_share. Neither
# path exists on a leased worker at all, which is the finding above. A script
# that guessed between them would attribute a missing checkpoint to the wrong
# thing, and on the worker it would attribute it to the wrong MACHINE.
#
# Three sub-modes exist so the parts of this script that would otherwise only
# ever execute under a lease are gateable without a checkpoint, a lease or a
# mount:
#
#   --check-source PATH     run only the local-disk refusal; 0 accept, 7 refuse
#   --stage-check SRC DST   run only the copy postcondition; 0 pass, 6 fail
#   --dry-run               resolve the kind, print the cmake/ninja/run_arm
#                           invocations this run WOULD issue, and assert that
#                           the ninja target those flags produce exists;
#                           0 plan is coherent, 4 it is not. Builds nothing,
#                           starts nothing, needs no checkpoint.
#
# WHY `--dry-run` EXISTS AT ALL. It is not a convenience. Until it was added the
# gated half of this script was the half that needs nothing — `--report-only`,
# `--check-source`, `--stage-check` — and the half that only ever runs on a
# leased box with a checkpoint (the configure, the build, `run_arm`, the
# `/health` poll, the kill/wait) was covered by nothing at all. That is how this
# file shipped configuring with `-DVLLM_CPP_BUILD_EXAMPLES=OFF` and then asking
# ninja for `vllm-server`, which is an `examples/` target: `ninja` answered
# `unknown target 'vllm-server'` and the run died at exit 4 before any RSS
# existed, while the reporter suite stayed 41/41 green. `--dry-run` asserts the
# one link between those two halves that a report-only test cannot see — that
# the flags this script passes `cmake` actually define the target it then asks
# `ninja` to build, and that the binary lands where `run_arm` looks for it.
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

# ─── The build the measurement runs, declared ONCE ──────────────────────────
#
# The real run and `--dry-run` read these same four values, so what the dry run
# prints is what the real run issues rather than a transcription of it. A
# transcription cannot gate the thing it transcribes: it agrees with itself.
#
# `VLLM_CPP_BUILD_EXAMPLES=ON` IS LOAD-BEARING, and it is the one this file
# shipped wrong. `vllm-server` is the OUTPUT_NAME of the `server` target
# (`examples/CMakeLists.txt:91,108`), and `examples/` is added only under
# `if(VLLM_CPP_BUILD_EXAMPLES)` (`CMakeLists.txt:2828`). With it OFF the
# configure succeeds, `ninja vllm-server` answers `unknown target`, and the run
# exits 4 having measured nothing. `docs/USAGE.md:54,95,128,204` names the
# binary `build/examples/vllm-server`, which is `$SERVER_RELPATH` below.
#
# Tests stay OFF: this is a measurement build, and the test targets are neither
# run nor timed here.
CMAKE_FLAGS=(-G Ninja -DCMAKE_BUILD_TYPE=Release
             -DVLLM_CPP_BUILD_TESTS=OFF -DVLLM_CPP_BUILD_EXAMPLES=ON)
NINJA_TARGET=vllm-server
NINJA_JOBS=4                 # unconstrained parallelism has OOM-rebooted the GB10
SERVER_RELPATH=examples/vllm-server
BUILD_DIR_PREFIX=${BUILD_DIR_PREFIX:-/tmp/tower-skip-build}

CHECKPOINT=""
OUT=""
REPORT_ONLY=""
MODEL_KIND=""
STAGE_TO=""
CHECK_SOURCE=""
STAGE_CHECK_SRC=""
STAGE_CHECK_DST=""
DRY_RUN=""
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
    --stage-to) STAGE_TO=$2; shift 2 ;;
    --check-source) CHECK_SOURCE=$2; shift 2 ;;
    --stage-check) STAGE_CHECK_SRC=$2; STAGE_CHECK_DST=$3; shift 3 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --build-dir-prefix) BUILD_DIR_PREFIX=$2; shift 2 ;;
    --baseline-ref) BASELINE_REF=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ─── The source must be on local disk ───────────────────────────────────────
#
# Returns 0 when PATH is a local-disk path this run may read weights from, and 7
# when it is not. Two prongs, because neither alone covers the case:
#
#   * the FILESYSTEM TYPE, which is the real question — `stat -f -c %T` reports
#     `cifs`, `smb2` or `smb3` for the NAS mount on both the workstation
#     (`//192.168.68.102/Data on /mnt/nas_share`) and the leased worker
#     (`//192.168.68.102/Data on /workspace`);
#   * the literal `/workspace` prefix, which holds even where `stat -f` cannot
#     answer, and which is the one path a leased job is given by convention.
#
# `nfs` is refused for the same reason as `cifs`. A filesystem this cannot
# identify is ACCEPTED: the refusal exists to stop a known-remote read, and
# refusing every unrecognised filesystem would stop ordinary local runs on hosts
# nobody has surveyed.
#
# THE PATH NEED NOT EXIST YET, AND THAT USED TO DEFEAT THE SECOND PRONG.
# `stat -f` fails on a path that is not there, which left `fstype` empty and the
# path ACCEPTED — measured: `--check-source /mnt/nas_share` returned 7 (smb2)
# while `--check-source /mnt/nas_share/no-such-dir` returned 0. This function is
# also what clears `--stage-to`, and `--stage-to` names a directory the run is
# about to CREATE, so the guard was blind in exactly the case it exists for: a
# staging directory on the NAS would have been accepted, created, and the
# checkpoint copied onto CIFS. The filesystem of a path that does not exist is
# the filesystem of the nearest ancestor that does, so that is what is probed,
# and the refusal names which path it read.
check_source_is_local() {
  local path=$1 fstype="" probe="" parent=""
  case "$path" in
    /workspace|/workspace/*)
      echo "--checkpoint '$path' is on the leased worker's CIFS mount." >&2
      echo "  /workspace is //192.168.68.102/Data over CIFS. Reading weights" >&2
      echo "  through it measures the mount, and the developer asked for local" >&2
      echo "  disk. Pass --stage-to <local dir> and the tree is copied first," >&2
      echo "  or point --checkpoint at a copy you staged yourself." >&2
      return 7
      ;;
  esac
  probe=$path
  while [ ! -e "$probe" ]; do
    parent=$(dirname "$probe")
    [ "$parent" = "$probe" ] && break
    probe=$parent
  done
  fstype=$(stat -f -c %T "$probe" 2>/dev/null || echo "")
  case "$fstype" in
    cifs|smb|smb2|smb3|nfs|nfs4)
      if [ "$probe" = "$path" ]; then
        echo "--checkpoint '$path' is on a $fstype filesystem." >&2
      else
        echo "--checkpoint '$path' does not exist yet, and its nearest existing" >&2
        echo "  ancestor '$probe' is on a $fstype filesystem, so that is where it" >&2
        echo "  would be created." >&2
      fi
      echo "  Reading weights over the network measures the network. Pass" >&2
      echo "  --stage-to <local dir>, or point --checkpoint at a local copy." >&2
      return 7
      ;;
  esac
  return 0
}

# ─── The copy postcondition ─────────────────────────────────────────────────
#
# Returns 0 when DST holds the same regular files as SRC, and 6 when it does
# not. `cp`'s exit status is NOT the evidence: on this fleet a missing wrapper
# binary has already made a copy command print success and move nothing, so the
# thing asserted is the RESOURCE. Every regular file's path relative to its own
# root and its byte size must match, and an empty source is itself a failure so
# that "copied nothing" cannot pass as "there was nothing to copy".
verify_stage() {
  local src=$1 dst=$2
  local sl dl sn dn sb db
  [ -d "$src" ] || { echo "stage check: source '$src' is not a directory" >&2; return 6; }
  [ -d "$dst" ] || { echo "stage check: destination '$dst' is not a directory" >&2; return 6; }
  sl=$(find "$src" -type f -printf '%P %s\n' | LC_ALL=C sort)
  dl=$(find "$dst" -type f -printf '%P %s\n' | LC_ALL=C sort)
  sn=$(printf '%s' "$sl" | grep -c '' )
  dn=$(printf '%s' "$dl" | grep -c '' )
  sb=$(printf '%s\n' "$sl" | awk '{t += $NF} END {print t + 0}')
  db=$(printf '%s\n' "$dl" | awk '{t += $NF} END {print t + 0}')
  echo "stage check: source      $sn files, $sb B  ($src)"
  echo "stage check: destination $dn files, $db B  ($dst)"
  if [ "$sn" -eq 0 ]; then
    echo "stage check: FAILED — the source holds no regular files at all." >&2
    echo "  An empty source is not a copy of anything, and passing it here" >&2
    echo "  would let a copy that moved nothing read as success." >&2
    return 6
  fi
  if [ "$sl" != "$dl" ]; then
    echo "stage check: FAILED — the two trees differ. First differences:" >&2
    diff <(printf '%s\n' "$sl") <(printf '%s\n' "$dl") | head -20 >&2
    echo "  This is the POSTCONDITION, not the copy command's exit status." >&2
    return 6
  fi
  echo "stage check: OK — every relative path and byte size matches."
  return 0
}

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

# ─── The legs, declared once ────────────────────────────────────────────────
#
# `TAG:BUILD-ARM:EXTRA-FLAG`. The real run and `--dry-run` both walk this table,
# so the plan the dry run prints is the plan the run executes rather than a copy
# of it that can drift away from it.
#
# The arm-to-binary assignment SWAPS between the two pairs. Pinning `default` to
# build a and `lmo` to build b in both pairs makes binary identity perfectly
# correlated with the arm, so any difference between the two binaries — a stale
# object, a different toolchain resolution, one tree that did not rebuild —
# arrives as a difference between the arms and is indistinguishable from the
# saving being measured. Swapping puts each binary on each arm once, so a
# binary-shaped effect shows up as disagreement BETWEEN the pairs, which
# `report` reads and folds into the verdict.
ARM_PLAN=(
  "warmup:a:"                              # DISCARDED; warms the page cache
  "default:a:"
  "lmo:b:--language-model-only"
  "default2:b:"
  "lmo2:a:--language-model-only"
)

arm_binary() { printf '%s\n' "$BUILD_DIR_PREFIX-$1/$SERVER_RELPATH"; }

# The exact argv one leg runs, into `ARM_CMD`. `run_arm` executes it and
# `--dry-run` prints it; neither restates the other.
ARM_CMD=()
arm_command() {
  local bin=$1 tag=$2; shift 2
  ARM_CMD=(/usr/bin/time -v -o "$OUT/$tag.time"
           "$bin" --model "$CHECKPOINT" --port "$PORT" --device cpu "$@")
}

# ─── Can these flags produce the binary the legs run? ───────────────────────
#
# Answered from CMake's OWN sources, so it costs no configure, needs no
# toolchain, and runs anywhere the checkout is. This is the assertion whose
# absence let `-DVLLM_CPP_BUILD_EXAMPLES=OFF` ship beside `ninja vllm-server`.
#
# Returns 0 coherent, 4 not.
assert_plan_builds_the_binary() {
  local repo=$1
  local defsites n defsite subdir guardline guard found=""
  defsites=$(grep -l -- "OUTPUT_NAME $NINJA_TARGET)" \
                "$repo/CMakeLists.txt" "$repo"/*/CMakeLists.txt 2>/dev/null)
  n=$(printf '%s' "$defsites" | grep -c '')
  if [ "$n" -ne 1 ]; then
    echo "cannot locate a unique CMake definition of '$NINJA_TARGET': $n found." >&2
    echo "  Looked for 'OUTPUT_NAME $NINJA_TARGET)' in $repo/CMakeLists.txt and" >&2
    echo "  $repo/*/CMakeLists.txt. Zero means the target was renamed; more than" >&2
    echo "  one means this check would have to guess which one the flags build." >&2
    return 4
  fi
  defsite=$defsites
  subdir=$(dirname "${defsite#"$repo"/}")
  if [ "$subdir" = "." ]; then
    # Defined at the top level: nothing gates it, and it lands at the build
    # directory root.
    if [ "$SERVER_RELPATH" != "$NINJA_TARGET" ]; then
      echo "'$NINJA_TARGET' is defined at the top level, so it lands at" >&2
      echo "  <build>/$NINJA_TARGET, but SERVER_RELPATH says $SERVER_RELPATH." >&2
      return 4
    fi
    echo "  target             '$NINJA_TARGET' is defined at the top level"
    return 0
  fi
  if [ "$SERVER_RELPATH" != "$subdir/$NINJA_TARGET" ]; then
    echo "'$NINJA_TARGET' is defined in $subdir/CMakeLists.txt, so ninja writes it" >&2
    echo "  to <build>/$subdir/$NINJA_TARGET, but SERVER_RELPATH says" >&2
    echo "  $SERVER_RELPATH — which is where the legs would look for it." >&2
    return 4
  fi
  guardline=$(awk -v d="add_subdirectory($subdir)" '
      /^[[:space:]]*if\(/ { last=$0 }
      index($0, d) { print last; exit }' "$repo/CMakeLists.txt")
  guard=$(printf '%s\n' "$guardline" | sed -n 's/^[[:space:]]*if(\([A-Za-z0-9_]*\)).*/\1/p')
  if [ -z "$guard" ]; then
    echo "  target             '$NINJA_TARGET' lives in $subdir/, added unconditionally"
    return 0
  fi
  for f in "${CMAKE_FLAGS[@]}"; do
    case "$f" in -D"$guard"=*) found=$f ;; esac
  done
  if [ "$found" != "-D$guard=ON" ]; then
    echo "the configure flags cannot produce '$NINJA_TARGET'." >&2
    echo "  It is defined in $subdir/CMakeLists.txt, and the root CMakeLists adds" >&2
    echo "  that directory only under 'if($guard)'. These flags pass" >&2
    echo "  '${found:-no -D$guard at all}', so cmake succeeds, ninja answers" >&2
    echo "  \"unknown target '$NINJA_TARGET'\", and the run exits 4 before any RSS" >&2
    echo "  exists. Pass -D$guard=ON." >&2
    return 4
  fi
  echo "  target             '$NINJA_TARGET' lives in $subdir/, gated by $guard, and"
  echo "                     these flags pass $found"
  echo "  binary path        <build>/$SERVER_RELPATH, which is what the legs run"
  return 0
}

# The same question asked of a build tree that is ALREADY configured. Skipped by
# name when there is none: this must not configure and must not build.
assert_target_in_build_dir() {
  local d=$1
  if [ ! -f "$d/build.ninja" ]; then
    echo "  live query         SKIPPED — no configured tree at $d"
    return 0
  fi
  if ! command -v ninja > /dev/null 2>&1; then
    echo "  live query         SKIPPED — ninja is not on PATH"
    return 0
  fi
  if ninja -C "$d" -t targets all 2>/dev/null | grep -q "^$NINJA_TARGET: "; then
    echo "  live query         '$NINJA_TARGET' IS a target in $d"
    return 0
  fi
  echo "'$NINJA_TARGET' is not a target in the configured tree $d." >&2
  echo "  'ninja -C $d -j $NINJA_JOBS $NINJA_TARGET' would answer" >&2
  echo "  \"unknown target '$NINJA_TARGET'\" and this run would exit 4." >&2
  return 4
}

# ─── One pair of arms ───────────────────────────────────────────────────────
#
# `$1` the run directory, `$2` the DEFAULT arm's tag, `$3` the
# language-model-only arm's tag. Sets `PAIR_SAVING` to the pair's saving in
# bytes on a non-VOID return. Returns 0 MET, 1 FAILING, 3 VOID.
# Peak RSS of one leg, in BYTES, or the empty string when the instrument line is
# absent. `/usr/bin/time -v` reports kilobytes.
rss_bytes() {
  local kbv
  kbv=$(grep -h 'Maximum resident set size' "$1/$2.time" 2>/dev/null | tr -dc '0-9')
  [ -n "$kbv" ] || return 1
  printf '%s' $((kbv * 1024))
}

PAIR_SAVING=0
report_pair() {
  local dir=$1 dtag=$2 ltag=$3
  local a b saving need pct
  PAIR_SAVING=0
  a=$(rss_bytes "$dir" "$dtag") || a=""
  b=$(rss_bytes "$dir" "$ltag") || b=""
  if [ -z "$a" ] || [ -z "$b" ]; then
    echo "RESULT: VOID — no 'Maximum resident set size' in one of $dtag.time / $ltag.time."
    echo "  A missing instrument line is not a measurement of zero saving."
    return 3
  fi
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
  local r1 r2 s1 s2 mean spread warm dflt legdiff
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
  printf 'spread |pair 1 - pair 2|    %14d B  %8.3f GiB\n' "$spread" \
         "$(echo "$spread" | awk '{print $1/1073741824}')"
  # WHAT THE SPREAD IS AND IS NOT. A binary-shaped bias `d` appears in it as
  # `2|d|`, which is why the swap exists — but ordinary run-to-run variance
  # lands in the same number, and this run takes ONE leg per cell, so it cannot
  # separate them. `.agents/benchmarking.md` asks for a noise band calibrated
  # from repeated identical legs BEFORE a delta is interpreted, and this harness
  # calibrates none. What it does have is the discarded warmup leg: same binary,
  # same arm, same flags as `default`, so `|warmup - default|` is one repeat of
  # one cell. It is COLD, so it is an UPPER BOUND on the leg-to-leg variation
  # rather than an estimate of it, and it is printed for scale, never gated.
  warm=$(rss_bytes "$dir" warmup) || warm=""
  dflt=$(rss_bytes "$dir" default) || dflt=""
  if [ -n "$warm" ] && [ -n "$dflt" ]; then
    legdiff=$(( warm > dflt ? warm - dflt : dflt - warm ))
    printf 'leg-to-leg |warmup-default| %14d B  %8.3f GiB\n' "$legdiff" \
           "$(echo "$legdiff" | awk '{print $1/1073741824}')"
    echo '  (same binary, same arm, one repeat; the warmup leg is COLD, so this is'
    echo '   an UPPER BOUND on run-to-run variation, not a calibrated noise band.)'
  else
    echo 'leg-to-leg |warmup-default|      NOT AVAILABLE — this directory carries no'
    echo '  warmup leg, so the spread above has nothing to be read against.'
  fi
  if [ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]; then
    echo "RESULT: MET (first half, BOTH pairs). Second half — default arm vs the"
    echo "  $BASELINE_REF binary, within ${DEFAULT_ARM_DRIFT_PCT}% — is a separate run and is NOT"
    echo "  asserted here."
    return 0
  fi
  echo "RESULT: FAILING. Pair 1 $([ "$r1" -eq 0 ] && echo MET || echo FAILING),"
  echo "  pair 2 $([ "$r2" -eq 0 ] && echo MET || echo FAILING). The axis stays open."
  if [ "$r1" -ne "$r2" ]; then
    echo "  The two pairs DISAGREE across the swapped arm-to-binary assignment."
    echo "  A binary-shaped bias d enters the spread as 2|d|, so IF the spread were"
    echo "  all bias, d would be about $((spread / 2)) B. It need not all be bias:"
    echo "  run-to-run variance lands in the same spread, one leg per cell cannot"
    echo "  tell the two apart, and the leg-to-leg figure above is a cold upper"
    echo "  bound rather than a calibrated band. What the disagreement does"
    echo "  establish is that the two halves do not agree, which is what the swap"
    echo "  exists to expose. It is not a smaller saving."
  fi
  echo "  Do not renegotiate the threshold."
  return 1
}

if [ -n "$CHECK_SOURCE" ]; then
  check_source_is_local "$CHECK_SOURCE"
  exit $?
fi

if [ -n "$STAGE_CHECK_SRC" ]; then
  verify_stage "$STAGE_CHECK_SRC" "$STAGE_CHECK_DST"
  exit $?
fi

# ─── --dry-run: print the plan, assert it can produce a binary ──────────────
#
# Nothing is configured, built, started or measured. `$OUT` and `$CHECKPOINT`
# are shown as placeholders when they were not given, so the printed argv keeps
# every flag in the position the real leg puts it in.
if [ -n "$DRY_RUN" ]; then
  REPO=$(cd "$(dirname "$0")/../.." && pwd)
  if [ -z "$MODEL_KIND" ] && [ -n "$CHECKPOINT" ]; then
    MODEL_KIND=$(kind_from_checkpoint "$CHECKPOINT")
  fi
  if [ -z "$MODEL_KIND" ]; then
    echo "--dry-run needs a model kind: pass --model-kind, or pass a --checkpoint" >&2
    echo "  whose config.json names a declared architecture. Refusing to pick one," >&2
    echo "  because the threshold is per model." >&2
    exit 2
  fi
  declare_model "$MODEL_KIND" || exit 2
  OUT=${OUT:-'<--out DIR>'}
  CHECKPOINT=${CHECKPOINT:-'<--checkpoint DIR>'}
  echo "DRY RUN — nothing is configured, built, started or measured."
  echo "repo            $REPO"
  echo "model kind      $MODEL_KIND"
  echo "workload        $WORKLOAD"
  echo "resident tower  $TOWER_RESIDENT_BYTES B (2 x $TOWER_ONDISK_BYTES B on disk; the x2 is #1359)"
  echo
  echo "planned build — one directory per arm, same commit:"
  for arm in a b; do
    echo "  cmake -S $REPO -B $BUILD_DIR_PREFIX-$arm ${CMAKE_FLAGS[*]}"
    echo "  ninja -C $BUILD_DIR_PREFIX-$arm -j $NINJA_JOBS $NINJA_TARGET"
    echo "  binary       $(arm_binary "$arm")"
  done
  echo
  echo "planned legs — each is one run_arm, /usr/bin/time -v wrapping the server:"
  for leg in "${ARM_PLAN[@]}"; do
    IFS=: read -r tag armb extra <<< "$leg"
    if [ -n "$extra" ]; then
      arm_command "$(arm_binary "$armb")" "$tag" "$extra"
    else
      arm_command "$(arm_binary "$armb")" "$tag"
    fi
    printf '  %-8s %s\n' "$tag" "${ARM_CMD[*]}"
  done
  echo "  readiness    curl -sf http://127.0.0.1:$PORT/health, then kill and wait"
  if [ "$WORKLOAD" = "completion" ]; then
    echo "  workload     POST http://127.0.0.1:$PORT/v1/completions model=$SERVED_MODEL_NAME"
    echo "               max_tokens=$MAX_TOKENS temperature=0"
  else
    echo "  workload     none beyond readiness; see the header for why this kind"
    echo "               cannot run a completion"
  fi
  echo
  echo "target check:"
  assert_plan_builds_the_binary "$REPO" || exit 4
  for arm in a b; do
    assert_target_in_build_dir "$BUILD_DIR_PREFIX-$arm" || exit 4
  done
  echo
  echo "DRY RUN: the plan is coherent. Nothing was built."
  exit 0
fi

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

# The model kind is resolved from the ORIGINAL path, before any staging, because
# `config.json` is small and reading it over the mount costs nothing that a
# measurement would notice.
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

# ─── Stage the checkpoint onto local disk, or refuse to read it remotely ─────
SOURCE_CHECKPOINT=$CHECKPOINT
if [ -n "$STAGE_TO" ]; then
  check_source_is_local "$STAGE_TO" || {
    echo "--stage-to '$STAGE_TO' is itself not local disk; staging there would" >&2
    echo "  move the problem rather than solve it." >&2
    exit 7
  }
  need_bytes=$(du -sb "$CHECKPOINT" | cut -f1)
  have_bytes=$(df --output=avail -B1 "$STAGE_TO" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -z "$have_bytes" ]; then
    mkdir -p "$STAGE_TO" || { echo "cannot create --stage-to '$STAGE_TO'" >&2; exit 6; }
    have_bytes=$(df --output=avail -B1 "$STAGE_TO" | tail -1 | tr -dc '0-9')
  fi
  echo "staging         $need_bytes B needed, $have_bytes B free under $STAGE_TO"
  if [ "$have_bytes" -lt "$need_bytes" ]; then
    echo "not enough local disk to stage the checkpoint; refusing rather than" >&2
    echo "  half-copying it and measuring the remainder over the mount" >&2
    exit 6
  fi
  STAGED="$STAGE_TO/$(basename "$CHECKPOINT")"
  mkdir -p "$STAGED" || { echo "cannot create '$STAGED'" >&2; exit 6; }
  # `cp` is invoked BARE. It is never wrapped in a timer or any other binary:
  # a missing wrapper has already made a copy print success and move nothing on
  # this fleet, and the postcondition below is what decides either way.
  cp -rL "$CHECKPOINT/." "$STAGED/" || echo "cp reported a failure; the postcondition below decides" >&2
  verify_stage "$CHECKPOINT" "$STAGED" || exit 6
  CHECKPOINT=$STAGED
  echo "staged to       $CHECKPOINT"
else
  check_source_is_local "$CHECKPOINT" || exit 7
fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SHA=$(git -C "$REPO" rev-parse HEAD)
echo "commit          $SHA"
echo "checkpoint      $CHECKPOINT"
echo "checkpoint src  $SOURCE_CHECKPOINT"
echo "checkpoint fs   $(stat -f -c %T "$CHECKPOINT" 2>/dev/null || echo unknown)"
echo "model kind      $MODEL_KIND"
echo "workload        $WORKLOAD"
echo "resident tower  $TOWER_RESIDENT_BYTES B (2 x $TOWER_ONDISK_BYTES B on disk; the x2 is #1359)"
echo "checkpoint root USED: the path above. .env's CHECKPOINT_ROOT was NOT consulted."
echo "host            $(uname -a)"
echo "load            $(cat /proc/loadavg)"
echo "free            $(free -g | sed -n 2p)"

# The flags must be able to produce the target, asserted BEFORE the configure
# rather than discovered by ninja after it. `--dry-run` runs this same check
# with nothing else attached, which is how it is gated in CI.
assert_plan_builds_the_binary "$REPO" || exit 4

# TWO build directories, same commit, so the A/B cannot measure one binary twice.
for arm in a b; do
  d="$BUILD_DIR_PREFIX-$arm"
  cmake -S "$REPO" -B "$d" "${CMAKE_FLAGS[@]}" > "$OUT/cmake-$arm.log" 2>&1 \
    || { echo "configure failed for arm $arm; see $OUT/cmake-$arm.log" >&2; exit 4; }
  assert_target_in_build_dir "$d" || exit 4
  ninja -C "$d" -j "$NINJA_JOBS" "$NINJA_TARGET" > "$OUT/build-$arm.log" 2>&1 \
    || { echo "build failed for arm $arm; see $OUT/build-$arm.log" >&2; exit 4; }
done
# NAMED, not searched for. `find ... -name vllm-server | head -1` picks one of
# whatever copies happen to be in the tree — a staged bundle, a previous
# layout — by luck, and prints nothing about which. The target lands at
# `$SERVER_RELPATH` because that is where CMake defines it, which is the
# property `assert_plan_builds_the_binary` just checked.
BIN_A=$(arm_binary a)
BIN_B=$(arm_binary b)
for bin in "$BIN_A" "$BIN_B"; do
  [ -x "$bin" ] || { echo "the build left no executable at $bin" >&2; exit 4; }
done
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
  arm_command "$bin" "$tag" "$@"
  "${ARM_CMD[@]}" > "$OUT/$tag.log" 2>&1 &
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

# The legs, in the declared order, out of `ARM_PLAN` — the same table
# `--dry-run` prints. A discarded first run warms the page cache, so the two
# arms see the same I/O state rather than the first one paying for the second;
# it is also a second `default` leg on binary A, which `report` folds in as the
# leg-to-leg figure.
for leg in "${ARM_PLAN[@]}"; do
  IFS=: read -r tag armb extra <<< "$leg"
  case "$tag" in
    warmup)  echo "== warming the page cache (this run is DISCARDED) ==" ;;
    default) echo "== A-B then B-A: each binary runs each arm exactly once ==" ;;
  esac
  if [ -n "$extra" ]; then
    run_arm "$(arm_binary "$armb")" "$tag" "$extra" || exit 5
  else
    run_arm "$(arm_binary "$armb")" "$tag" || exit 5
  fi
done

echo
report "$OUT"; verdict=$?
echo
echo "logs and all four .time files are under $OUT"
exit $verdict
