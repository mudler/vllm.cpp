#!/usr/bin/env bash
# TOKENGATE (#TBD): capture the OPT-125m greedy oracle golden at the CANDIDATE
# revision e126687a9a828d513c01a07cd69f025f27d63280 on `dgx:gpu0` (GB10,
# sm_121a) -- the SAME device the committed golden was captured on -- and diff
# it against that committed golden.
#
# WHY dgx AND NOT thor. The committed golden
# tests/parity/goldens/opt_greedy/greedy_ids.npy was captured on dgx/GB10. A
# capture at the candidate on any OTHER device moves two variables at once
# (revision AND silicon) and cannot attribute a difference, which is exactly
# what makes #2628's thor run informative rather than a gate. Holding the
# device fixed at GB10 leaves the revision as the only difference.
#
# RESUMABLE BY CONSTRUCTION. `dgx:gpu0` has crashed under long sequences, so
# this runs in two stages selected by $STAGE and each stage is a no-op when its
# output already exists on /workspace:
#   STAGE=build    stage sources, build the wheel, PERSIST it to /workspace
#   STAGE=capture  install the wheel, run scripts/opt-oracle-capture.py --runs 5
#   STAGE=all      both
#
# Every value that decides what was measured is printed. A step that cannot run
# exits non-zero; it never prints a partial success.
#
# THE EXIT CODE CARRIES THE VERDICT, and DRIFT has its own value. An earlier
# draft put DIFF_RC into a SUM line and fell through to exit 0, which made a
# real golden drift -- the one outcome this job exists to detect -- the only
# result `rc` could not tell apart from success. The map:
#
#   0  the capture ran and the candidate REPRODUCES the committed golden
#   2  wrong device: compute capability is not 12.1, nothing was measured
#   3  staging or build prerequisites missing
#   4  the vLLM source build failed
#   5  capture-stage setup failed (wheel, venv, checkpoint, oracle identity)
#   6  the capture itself failed to run
#   7  DRIFT: the capture ran and the candidate DISAGREES with the committed
#      golden, or is no longer self-deterministic. A FINDING, not a failure.
#   8  the differ could not compare (missing or malformed input) -- instrument
#   9  a staged input's sha256 does not match the tree it was staged from
#
# 2-6, 8 and 9 are the instrument or the environment failing. Only 0 and 7 are
# statements about the target.
set -uo pipefail

STAGE="${STAGE:-all}"
WS="${WS:-/workspace/tokengate-e126687}"
WORK="${WORK:-/tmp/tokengate-e126687}"
SRC_STAGE="${SRC_STAGE:-/workspace/runhalf-e126687/src}"
TARGET_SHA=e126687a9a828d513c01a07cd69f025f27d63280
ARCH_LIST="${ARCH_LIST:-12.1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$WS/out/$STAMP"
mkdir -p "$OUT" "$WS/wheel" "$WS/ccache-remote"

say() { echo "### $*"; }
sum() { echo "SUM $*"; }

say "TOKENGATE job start $STAMP  STAGE=$STAGE"
say "uname: $(uname -a)"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
say "COMPUTE_CAP=$CAP"
# The device assertion IS the point of this job. A capture on anything but GB10
# is not the measurement that was asked for, so refuse rather than produce a
# number wearing the wrong label.
if [ "$CAP" != "12.1" ]; then
  sum "DEVICE_RC=1  compute_cap=$CAP expected 12.1 (GB10). REFUSING."
  exit 2
fi
sum "DEVICE_RC=0  compute_cap=$CAP"

# ---------------------------------------------------------------- build stage
WHEEL_GLOB="$WS/wheel/vllm-*-cp312-cp312-linux_aarch64.whl"
have_wheel() { compgen -G "$WHEEL_GLOB" > /dev/null; }

if [ "$STAGE" = "build" ] || [ "$STAGE" = "all" ]; then
  if have_wheel; then
    say "wheel already persisted; build stage is a no-op"
    sum "SRCBUILD_RC=0 (cached) $(compgen -G "$WHEEL_GLOB" | head -1)"
  else
    say "--- toolchain ---"
    if ! command -v nvcc > /dev/null; then
      say "installing CUDA toolkit (nvcc absent)"
      apt-get update -qq && apt-get install -y -qq cuda-toolkit-13-0 > "$OUT/apt.log" 2>&1
      export PATH=/usr/local/cuda/bin:$PATH
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    nvcc --version | tail -2
    command -v git cmake ninja python3 || true
    apt-get install -y -qq python3-dev python3-venv git cmake ninja-build >> "$OUT/apt.log" 2>&1 || true

    say "--- stage sources from $SRC_STAGE (worker has NO github.com egress) ---"
    mkdir -p "$WORK/src"
    for t in cutlass flash-attention FlashMLA FlashKDA DeepGEMM MSA qutlass tml-fa4 triton; do
      if [ ! -d "$WORK/src/$t" ] && [ ! -d "$WORK/src/${t}"* ]; then
        say "extracting $t"
        tar xzf "$SRC_STAGE/$t.tar.gz" -C "$WORK/src" || { sum "STAGE_RC=1 $t"; exit 3; }
      fi
    done
    ls -1 "$WORK/src"

    say "--- restore the vLLM tree from its git bundle (setuptools_scm needs git) ---"
    if [ ! -d "$WORK/vllm/.git" ]; then
      rm -rf "$WORK/vllm"
      git clone -q "$SRC_STAGE/vllm-e126687.bundle" "$WORK/vllm" || { sum "CLONE_RC=1"; exit 3; }
      git -C "$WORK/vllm" checkout -q "$TARGET_SHA" || { sum "CHECKOUT_RC=1"; exit 3; }
    fi
    HEAD_SHA="$(git -C "$WORK/vllm" rev-parse HEAD)"
    TREE_SHA="$(git -C "$WORK/vllm" rev-parse HEAD^{tree})"
    say "HEAD_SHA=$HEAD_SHA"
    say "TREE_SHA=$TREE_SHA"
    say "GIT_DESCRIBE=$(git -C "$WORK/vllm" describe --tags 2>/dev/null || echo none)"
    say "REVCOUNT=$(git -C "$WORK/vllm" rev-list --count HEAD)"
    say "PORCELAIN_LINES=$(git -C "$WORK/vllm" status --porcelain | wc -l)"
    if [ "$HEAD_SHA" != "$TARGET_SHA" ]; then sum "TARGET_RC=1 got $HEAD_SHA"; exit 3; fi
    sum "TARGET_RC=0 CONFIRMED $TARGET_SHA"

    say "--- venv ---"
    if [ ! -x "$WORK/venv/bin/python" ]; then
      python3 -m venv "$WORK/venv" || { sum "VENV_RC=1"; exit 3; }
    fi
    "$WORK/venv/bin/pip" install -q -U pip wheel setuptools setuptools_scm ninja cmake numpy

    # THE BUILD REQUIREMENTS COME FROM THE TREE, NOT FROM THIS FILE. The wheel
    # build below passes `--no-build-isolation`, which makes pip SKIP
    # pyproject.toml's [build-system] requires ENTIRELY, so whatever this venv
    # lacks the build simply does not get. The line above used to be the only
    # answer to that, and it is a HAND-KEPT list: job
    # efc30c74-005e-4e80-bc28-bd34f5b76b77 reached the build stage and died in 3
    # seconds at setup.py:21, `from setuptools_rust.build import build_rust`,
    # because the pin's requires had moved past the list and `packaging`,
    # `setuptools-rust` and `jinja2` were all absent.
    #
    # pyproject.toml at $TARGET_SHA says its requires "Should be mirrored in
    # requirements/build/cuda.txt", and at e126687a9 that file is a superset of
    # them. Installing THAT file, out of the clone we just checked out, cannot
    # drift from the pin. A list retyped here silently can, which is the defect
    # above and the reason this is not simply three more names on line 121.
    BUILDREQ="$WORK/vllm/requirements/build/cuda.txt"
    if [ ! -f "$BUILDREQ" ]; then
      sum "BUILDREQ_RC=1  $BUILDREQ is absent at $TARGET_SHA"
      exit 3
    fi
    say "--- build requirements, read from the tree at $TARGET_SHA ---"
    sed 's/^/BUILDREQ /' "$BUILDREQ"
    "$WORK/venv/bin/pip" install -q -r "$BUILDREQ" > "$OUT/pipbuildreq.log" 2>&1 \
      || { sum "BUILDREQ_RC=1 see pipbuildreq.log"; tail -30 "$OUT/pipbuildreq.log"; exit 3; }

    "$WORK/venv/bin/pip" install -q -r "$WORK/vllm/requirements/cuda.txt" > "$OUT/pipreq.log" 2>&1 \
      || { sum "PIPREQ_RC=1 see pipreq.log"; tail -30 "$OUT/pipreq.log"; exit 3; }
    sum "PIPREQ_RC=0"

    # --- BUILD-REQUIRES ASSERTION begin ---
    # ASSERT what the build actually needs, read from the pin's OWN
    # pyproject.toml rather than from anything written here. "Should be mirrored
    # in requirements/build/cuda.txt" is upstream's intention, not a guarantee,
    # and the failure it hides is precisely the one that cost efc30c74: a name
    # in `requires` that nothing installed. Deriving the list means this cannot
    # go stale when the pin advances -- it will say which name is missing
    # instead of handing the next lease a traceback.
    #
    # It checks PRESENCE and not the version specifier. pip resolved the
    # versions when it installed the file above; what this catches is a NAME
    # that nothing installed, which is the ModuleNotFoundError class that
    # actually stops the build.
    #
    # `tests/scripts/test_tokengate_buildreq.py` extracts this program and runs
    # it against scratch trees, so the checker is falsifiable without a lease.
    "$WORK/venv/bin/python" - "$WORK/vllm/pyproject.toml" <<'PY'
import importlib.metadata as md
import re
import sys
import tomllib

path = sys.argv[1]
with open(path, "rb") as fh:
    requires = tomllib.load(fh).get("build-system", {}).get("requires", [])
if not requires:
    print("BUILDREQ ERROR no [build-system] requires in " + path)
    sys.exit(2)


def dist_name(spec):
    """The distribution a PEP 508 requirement names, without its constraints."""
    return re.split(r"[\s<>=!~;\[(]", spec.strip(), maxsplit=1)[0]


# `setuptools_rust` and `setuptools-rust` are ONE distribution, and pyproject
# and the mirror do not always agree on the spelling. No normalisation is
# written here because `importlib.metadata` already applies PEP 503 to the name
# it is given -- verified on this interpreter, where `annotated-doc`,
# `annotated_doc` and `annotated.doc` all resolve.
missing = []
for spec in requires:
    name = dist_name(spec)
    try:
        found = md.version(name)
    except md.PackageNotFoundError:
        print("BUILDREQ ABSENT  " + spec)
        missing.append(name)
    else:
        print("BUILDREQ PRESENT " + name + "==" + found + "  (" + spec + ")")
print("BUILDREQ REQUIRES=%d ABSENT=%d" % (len(requires), len(missing)))
if missing:
    print("BUILDREQ MISSING " + " ".join(missing))
    sys.exit(1)
PY
    BUILDREQ_RC=$?
    if [ "$BUILDREQ_RC" -ne 0 ]; then
      sum "BUILDREQ_RC=1  the venv does not satisfy [build-system] requires at $TARGET_SHA"
      exit 3
    fi
    sum "BUILDREQ_RC=0  every [build-system] requires distribution is installed"
    # --- BUILD-REQUIRES ASSERTION end ---

    say "--- build (MAX_JOBS=4; unconstrained parallelism has OOM-rebooted this fleet) ---"
    export VLLM_USE_PRECOMPILED=0 VLLM_TARGET_DEVICE=cuda
    export MAX_JOBS=4 NVCC_THREADS=1
    export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
    export CUDA_HOME=/usr/local/cuda
    export CCACHE_DIR="$WORK/ccache"
    # ccache on CIFS deadlocks (278 lock failures, measured); the REMOTE store
    # is the shape that works and it is what makes a crashed build resumable.
    export CCACHE_REMOTE_STORAGE="file:$WS/ccache-remote"
    export VLLM_CUTLASS_SRC_DIR="$WORK/src/cutlass"
    export VLLM_FLASH_ATTN_SRC_DIR="$WORK/src/flash-attention"
    export FLASH_MLA_SRC_DIR="$WORK/src/FlashMLA"
    export FLASH_KDA_SRC_DIR="$WORK/src/FlashKDA"
    export DEEPGEMM_SRC_DIR="$WORK/src/DeepGEMM"
    export FMHA_SM100_SRC_DIR="$WORK/src/MSA"
    export QUTLASS_SRC_DIR="$WORK/src/qutlass"
    export TML_FA4_SRC_DIR="$WORK/src/tml-fa4"
    export TRITON_KERNELS_SRC_DIR="$WORK/src/triton/python/triton_kernels/triton_kernels"
    for v in VLLM_USE_PRECOMPILED VLLM_TARGET_DEVICE MAX_JOBS NVCC_THREADS \
             TORCH_CUDA_ARCH_LIST CUDA_HOME CCACHE_REMOTE_STORAGE \
             VLLM_CUTLASS_SRC_DIR VLLM_FLASH_ATTN_SRC_DIR FLASH_MLA_SRC_DIR \
             FLASH_KDA_SRC_DIR DEEPGEMM_SRC_DIR FMHA_SM100_SRC_DIR \
             QUTLASS_SRC_DIR TML_FA4_SRC_DIR TRITON_KERNELS_SRC_DIR; do
      echo "BUILD_ENV $v=${!v}"
    done
    # Any GIT_REPOSITORY the tree still declares that we did NOT override names
    # itself here instead of costing another lease at configure time.
    say "--- declared GIT_REPOSITORY lines in the tree ---"
    grep -rn "GIT_REPOSITORY" "$WORK/vllm/CMakeLists.txt" "$WORK/vllm/cmake" 2>/dev/null | sed 's/^/GITREPO /'

    T0=$(date +%s)
    ( cd "$WORK/vllm" && "$WORK/venv/bin/pip" wheel --no-deps --no-build-isolation -w "$WORK/dist" . ) \
      > "$OUT/build.log" 2>&1
    SRCBUILD_RC=$?
    T1=$(date +%s)
    say "build wall seconds: $((T1-T0))"
    tail -40 "$OUT/build.log"
    if [ "$SRCBUILD_RC" -ne 0 ]; then sum "SRCBUILD_RC=1  see build.log"; exit 4; fi
    W="$(compgen -G "$WORK/dist/vllm-*.whl" | head -1)"
    say "WHEEL=$W  $(stat -c %s "$W") bytes  sha256 $(sha256sum "$W" | cut -d' ' -f1)"
    cp -f "$W" "$WS/wheel/" || { sum "PERSIST_RC=1"; exit 4; }
    sum "SRCBUILD_RC=0  persisted to $WS/wheel/"
  fi
fi

# -------------------------------------------------------------- capture stage
if [ "$STAGE" = "capture" ] || [ "$STAGE" = "all" ]; then
  if ! have_wheel; then sum "CAPTURE_RC=1  no wheel at $WS/wheel"; exit 5; fi
  W="$(compgen -G "$WHEEL_GLOB" | head -1)"

  # ---- STAGED-INPUT INTEGRITY --------------------------------------------
  # /workspace is a SHARED CIFS surface that other sessions write. The
  # checkpoint was already asserted below; these three were not, and
  # `goldens-committed` IS THE BAR -- a stale or wrong staged golden yields a
  # confident verdict against the wrong reference. The expected values are the
  # sha256 of the files in the tree at the commit that staged them, so this
  # also makes the "VERBATIM" claim about opt-oracle-capture.py checkable from
  # the job's own output instead of from a comment.
  assert_sha() {
    local path="$1" want="$2" got
    if [ ! -f "$path" ]; then echo "STAGED MISSING $path"; return 1; fi
    got="$(sha256sum "$path" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
      echo "STAGED MISMATCH $path"
      echo "STAGED   want $want"
      echo "STAGED   got  $got"
      return 1
    fi
    echo "STAGED OK $got  $path"
    return 0
  }
  stage_bad=0
  assert_sha "$WS/opt-oracle-capture.py"     c4b4b770671d3728faa3480d957bca392a68ec8d85b682fb24c918469d5342d3 || stage_bad=1
  assert_sha "$WS/tokengate-e126687-diff.py"     7334dab81079531051e7ab60dedab128ec1fb286d1c2cd2b44542dd9d1195d19 || stage_bad=1
  assert_sha "$WS/goldens-committed/greedy_ids.npy"     078d15930de5e498788922cad66aa83dcba3c24ad2a072b7a9447a0d32e90698 || stage_bad=1
  assert_sha "$WS/goldens-committed/greedy_dist.npy"     16e0ef356564bb36e2030004f68ee8e60b9109e609188f75c86835cf76918f45 || stage_bad=1
  assert_sha "$WS/goldens-committed/p0_prompt.i32"     7b14d31df1a3db82e271b9d4dae1844ca5f7270a9a32e63d294db98925df9d45 || stage_bad=1
  assert_sha "$WS/goldens-committed/p1_prompt.i32"     7c8d0ee78370f31d0d815ce6c594dac17f007da86d2332c2c670e595a7c81544 || stage_bad=1
  assert_sha "$WS/goldens-committed/p2_prompt.i32"     fab727c9ffd510284de8e2ae2b884f43ddd6ed25563bf38af43d97e24fadadc2 || stage_bad=1
  assert_sha "$WS/goldens-committed/p3_prompt.i32"     b791f64b0a95c4363655811636c2d5d3f688e8df54b05e0eac26ec4634ee4137 || stage_bad=1
  assert_sha "$WS/goldens-committed/p4_prompt.i32"     7e3be091ff6ab0c636e1b509502821e3e6565fba1699e87b7f80c6c46714d620 || stage_bad=1
  assert_sha "$WS/goldens-committed/p5_prompt.i32"     b8c6bec84f1ccaa7dc9d200be93da22fd533908df5cfc6c23d020f2b384090c0 || stage_bad=1
  if [ "$stage_bad" -ne 0 ]; then
    sum "STAGED_RC=1  a staged input does not match the tree it came from"
    exit 9
  fi
  sum "STAGED_RC=0  capture script, differ and all 8 committed goldens verified"
  say "--- install $W ---"
  if [ ! -x "$WORK/venv/bin/python" ]; then
    python3 -m venv "$WORK/venv" || { sum "VENV_RC=1"; exit 5; }
    "$WORK/venv/bin/pip" install -q -U pip numpy
  fi
  "$WORK/venv/bin/pip" install -q "$W" > "$OUT/pipwheel.log" 2>&1 \
    || { sum "PIPWHEEL_RC=1"; tail -30 "$OUT/pipwheel.log"; exit 5; }

  # NEVER read a gate model off /workspace: CIFS, and the rule is staged
  # NAS -> local ONCE. The checkpoint is the SAME materialized bf16 artifact the
  # committed golden was captured against, and its sha256s are asserted.
  say "--- stage the checkpoint /workspace -> /tmp and ASSERT its bytes ---"
  mkdir -p "$WORK/ckpt/opt-125m-bf16-st"
  cp -rL "$WS/ckpt/." "$WORK/ckpt/opt-125m-bf16-st/" || { sum "CKPT_RC=1"; exit 5; }
  ( cd "$WORK/ckpt/opt-125m-bf16-st" && sha256sum config.json generation_config.json \
      model.safetensors tokenizer_config.json tokenizer.json ) | sed 's/^/CKPT /'
  EXPECT_ST=d3eb4d4556e68b8a1d3721b1b3a4121a6a29cde36dc20f8bac71a0ccb2fd2255
  GOT_ST="$(sha256sum "$WORK/ckpt/opt-125m-bf16-st/model.safetensors" | cut -d' ' -f1)"
  if [ "$GOT_ST" != "$EXPECT_ST" ]; then sum "CKPT_RC=1 safetensors $GOT_ST != $EXPECT_ST"; exit 5; fi
  sum "CKPT_RC=0  model.safetensors $GOT_ST"

  say "--- oracle identity, read from a directory that is NOT the source tree ---"
  ( cd / && "$WORK/venv/bin/python" - <<'PY'
import torch, vllm
print("VLLM_FILE", vllm.__file__)
print("VLLM_VERSION", vllm.__version__)
print("TORCH_VERSION", torch.__version__)
print("CUDA_AVAIL", torch.cuda.is_available())
print("DEVICE", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
  ) | tee "$OUT/identity.log"
  VER="$(grep '^VLLM_VERSION' "$OUT/identity.log" | awk '{print $2}')"
  case "$VER" in
    *+ge126687a9*) sum "ORACLE_ID_RC=0  $VER" ;;
    *) sum "ORACLE_ID_RC=1  $VER does not carry +ge126687a9"; exit 5 ;;
  esac

  # Every value here is scripts/opt-oracle-capture.py's OWN default, passed
  # explicitly so the log records it: --runs is the only argument the committed
  # invocation in that script's docstring supplies. An earlier draft passed
  # --gpu-mem-util 0.10, which is RUNHALF's thor value picked for a
  # unified-memory box; 0.20 is what the committed golden was captured under,
  # and in a wave whose whole argument is that config deltas turn a gate into an
  # anecdote, an undeclared delta of my own is not defensible. 0.20 of this
  # box's 128 GB is far more than a 125M model and its KV pool need.
  say "--- CAPTURE: scripts/opt-oracle-capture.py VERBATIM, --runs 5 ---"
  mkdir -p "$OUT/goldens-candidate"
  ( cd "$WS" && "$WORK/venv/bin/python" "$WS/opt-oracle-capture.py" \
      --model "$WORK/ckpt/opt-125m-bf16-st" \
      --runs 5 --max-tokens 16 --gpu-mem-util 0.20 --max-model-len 2048 \
      --out-dir "$OUT/goldens-candidate" ) > "$OUT/capture.log" 2>&1
  CAPTURE_RC=$?
  cat "$OUT/capture.log"
  if [ "$CAPTURE_RC" -ne 0 ]; then sum "CAPTURE_RC=1  see capture.log"; exit 6; fi
  sum "CAPTURE_RC=0"

  say "--- DIFF candidate golden vs the COMMITTED golden ---"
  "$WORK/venv/bin/python" "$WS/tokengate-e126687-diff.py" \
    --committed "$WS/goldens-committed" --candidate "$OUT/goldens-candidate" \
    | tee "$OUT/diff.log"
  DIFF_RC=${PIPESTATUS[0]}
  sum "DIFF_RC=$DIFF_RC"
  case "$DIFF_RC" in
    0) sum "TOKENGATE=PASS   the candidate reproduces the committed golden" ;;
    1) sum "TOKENGATE=DRIFT  a FINDING: record it and re-gate, do not repair here" ;;
    *) sum "TOKENGATE=INSTRUMENT the differ could not compare; nothing is said about the target" ;;
  esac
fi

say "--- artifacts ---"
ls -laR "$OUT" | head -60
say "DONE_MARKER_TOKENGATE $STAMP"

# The verdict leaves this script in its EXIT STATUS, not only in a log line.
case "${DIFF_RC:-0}" in
  0) exit 0 ;;
  1) exit 7 ;;
  *) exit 8 ;;
esac
