#!/bin/bash
# RUNHALF, issue #2611, row UPSTREAM-SYNC-HEADPIN.
#
# ONE QUESTION: does vLLM e126687a9a828d513c01a07cd69f025f27d63280 SOURCE-BUILD
# and RUN a model on thor:gpu0? Every prior job at this target ran
# VLLM_USE_PRECOMPILED=1, which on aarch64 yields an editable wheel and NO
# vllm._C, so nothing could run. This job builds the extension from source.
#
# Every rc is reported separately and literally. A leg that could not run is an
# ABSENCE and is printed as one; it is never folded into a rc that means
# something else.
set -u
TARGET=e126687a9a828d513c01a07cd69f025f27d63280
S=/workspace/runhalf-e126687
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$S/out/$STAMP
ROOT=/tmp/runhalf-e126687
SRC=$ROOT/vllm-src
V=$ROOT/venv
mkdir -p "$RUN" "$ROOT"
echo "RUNHALF run dir: $RUN"
echo "SCRIPT_SHA256=$(sha256sum "$0" | cut -d' ' -f1)"

step() { echo; echo "########## $* ##########"; date -u +'%Y-%m-%dT%H:%M:%SZ'; echo "t=${SECONDS}s"; }

step "0. BOX -- and every value a later reader needs to know what this measured"
uname -m; echo "UNAME_RC=$?"
ARCH=$(uname -m); echo "ARCH=$ARCH"
NPROC=$(nproc); echo "NPROC=$NPROC"
hostname; echo "HOST=$(hostname)"
cat /proc/sys/kernel/random/boot_id
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
echo "SMI_RC=$?"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
echo "COMPUTE_CAP=${CC:-unknown}"
free -m | head -2
df -h /tmp /workspace | head -5
id

step "1. INPUTS -- staged from the dev box; a missing one is an ABSENCE"
ls -la "$S/src" "$S/models" 2>&1 | head -30
BUNDLE=$S/src/vllm-e126687.bundle
test -f "$BUNDLE"; echo "BUNDLE_PRESENT_RC=$?"
test -d "$S/models/opt-125m"; echo "OPT125M_PRESENT_RC=$?"
test -f "$S/models/qwen4exp-config/config.json"; echo "Q4EXPCFG_PRESENT_RC=$?"

step "2. TOOLCHAIN -- python headers and a CUDA toolkit"
export DEBIAN_FRONTEND=noninteractive
PYINC=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["include"])' 2>/dev/null)
echo "PYINC=$PYINC"
if [ -f "$PYINC/Python.h" ]; then echo "PYTHON_H_BEFORE=present"; else echo "PYTHON_H_BEFORE=absent"; fi
apt-get update -qq > "$RUN/apt.log" 2>&1; echo "APTUPD_RC=$?"
apt-get install -y -qq python3 python3-venv python3-dev git curl ca-certificates build-essential cmake ninja-build ccache >> "$RUN/apt.log" 2>&1
echo "APTDEV_RC=$?"
if [ -f "$PYINC/Python.h" ]; then echo "PYTHON_H_AFTER=present"; else echo "PYTHON_H_AFTER=absent"; fi

# The CUDA toolkit is the one thing the worker sheet names as absent. A source
# build needs far more than nvcc: cudart, cublas, curand, nvrtc and their
# headers. Probe first, because the worker image can carry one.
NVCC_FOUND=""
for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13.0/bin/nvcc /usr/bin/nvcc; do
  [ -x "$c" ] && NVCC_FOUND=$c && break
done
command -v nvcc >/dev/null 2>&1 && NVCC_FOUND=${NVCC_FOUND:-$(command -v nvcc)}
echo "NVCC_PREEXISTING=${NVCC_FOUND:-absent}"
if [ -z "$NVCC_FOUND" ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
  echo "CUDAKEYRING_CURL_RC=$?"
  dpkg -i /tmp/cuda-keyring.deb > "$RUN/cuda.log" 2>&1; echo "CUDAKEYRING_DPKG_RC=$?"
  apt-get update -qq >> "$RUN/cuda.log" 2>&1; echo "CUDAAPTUPD_RC=$?"
  # The full toolkit, not a hand-picked subset: a source build links against
  # parts a subset misses, and a missing one presents as a compile error deep
  # in the build rather than as a missing package.
  apt-get install -y -qq cuda-toolkit-13-0 >> "$RUN/cuda.log" 2>&1
  echo "CUDATOOLKIT_APT_RC=$?"
  tail -5 "$RUN/cuda.log"
  for c in /usr/local/cuda-13.0/bin/nvcc /usr/local/cuda/bin/nvcc; do
    [ -x "$c" ] && NVCC_FOUND=$c && break
  done
fi
echo "NVCC=${NVCC_FOUND:-absent}"
if [ -z "$NVCC_FOUND" ]; then
  echo "ABORT: no CUDA toolkit. Nothing below could be a measurement about the target."
  echo "SRCBUILD_RC=SKIPPED_NO_TOOLKIT EXT_PRESENT=SKIPPED RUN_RC=SKIPPED"
  echo "DONE_MARKER_RUNHALF"
  exit 20
fi
export CUDA_HOME=$(dirname "$(dirname "$NVCC_FOUND")")
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME"
"$NVCC_FOUND" --version > "$RUN/nvcc.txt" 2>&1; echo "NVCCV_RC=$?"; tail -2 "$RUN/nvcc.txt"
ls "$CUDA_HOME/include/cuda_runtime.h" > /dev/null 2>&1; echo "CUDART_HEADER_RC=$?"
ls "$CUDA_HOME"/lib64/libcudart.so* 2>/dev/null | head -2
ls "$CUDA_HOME"/lib64/libcublas.so* 2>/dev/null | head -2

step "3. THE SOURCE, restored from the staged bundle and ASSERTED against the target"
# A bundle rather than a clone: the worker's github egress is not guaranteed
# (.agents/environment.md, a thor job failed `git fetch` with a credential error
# on 2026-09-02), and the bundle carries the tags setuptools_scm needs. The
# identity is asserted from the restored tree, not from the file name.
if [ ! -d "$SRC/.git" ]; then
  cp "$BUNDLE" /tmp/vllm.bundle; echo "BUNDLE_COPY_RC=$?"
  git clone --quiet /tmp/vllm.bundle "$SRC" > "$RUN/clone.log" 2>&1; echo "CLONE_RC=$?"
  git -C "$SRC" checkout --quiet "$TARGET" >> "$RUN/clone.log" 2>&1; echo "CHECKOUT_RC=$?"
else
  echo "SRC_REUSED=1 (this worker container already holds the tree)"
  git -C "$SRC" checkout --quiet "$TARGET" > "$RUN/clone.log" 2>&1; echo "CHECKOUT_RC=$?"
fi
HEAD_SHA=$(git -C "$SRC" rev-parse HEAD); echo "HEAD_SHA=$HEAD_SHA"
if [ "$HEAD_SHA" != "$TARGET" ]; then
  echo "ABORT: tree is $HEAD_SHA, not the target $TARGET."
  echo "DONE_MARKER_RUNHALF"; exit 21
fi
echo "TARGET CONFIRMED $HEAD_SHA"
echo "TREE_SHA=$(git -C "$SRC" rev-parse HEAD^{tree})"
echo "SHALLOW=$(git -C "$SRC" rev-parse --is-shallow-repository)"
echo "REVCOUNT=$(git -C "$SRC" rev-list --count HEAD)"
echo "GIT_DESCRIBE=$(git -C "$SRC" describe --tags 2>&1)"
echo "PORCELAIN_LINES=$(git -C "$SRC" status --porcelain | wc -l)"
echo "QWEN4EXP_FILES=$(git -C "$SRC" ls-tree -r --name-only HEAD | grep -c '^vllm/models/qwen4_exp/')"

step "3b. EXTERNAL CMAKE SOURCES, staged so the build needs no github egress"
mkdir -p "$ROOT/ext"
# CMakeLists.txt includes NINE external projects for a CUDA build and each one
# clones from github at CONFIGURE time, before any arch gate can skip it. Job A
# (20260902T213816Z) staged four and died on the fifth, DeepGEMM, with
# `could not read Username for 'https://github.com'` -- the worker has NO github
# egress, which is exactly the mode .agents/environment.md records for a thor
# job on this date. The list below is the COMPLETE set; the probe after it
# prints every GIT_REPOSITORY the tree still declares, so a tenth one names
# itself instead of costing another lease.
for t in cutlass flash-attention FlashMLA FlashKDA DeepGEMM MSA qutlass tml-fa4 triton; do
  if [ -f "$S/src/$t.tar.gz" ] && [ ! -d "$ROOT/ext/$t" ]; then
    tar -C "$ROOT/ext" -xzf "$S/src/$t.tar.gz"; echo "UNTAR_${t}_RC=$?"
  fi
  if [ -d "$ROOT/ext/$t" ]; then echo "EXTSRC_$t=$ROOT/ext/$t"; else echo "EXTSRC_$t=absent (the build will try github and there is no egress)"; fi
done
[ -d "$ROOT/ext/cutlass" ]         && export VLLM_CUTLASS_SRC_DIR="$ROOT/ext/cutlass"
[ -d "$ROOT/ext/flash-attention" ] && export VLLM_FLASH_ATTN_SRC_DIR="$ROOT/ext/flash-attention"
[ -d "$ROOT/ext/FlashMLA" ]        && export FLASH_MLA_SRC_DIR="$ROOT/ext/FlashMLA"
[ -d "$ROOT/ext/FlashKDA" ]        && export FLASH_KDA_SRC_DIR="$ROOT/ext/FlashKDA"
[ -d "$ROOT/ext/DeepGEMM" ]        && export DEEPGEMM_SRC_DIR="$ROOT/ext/DeepGEMM"
[ -d "$ROOT/ext/MSA" ]             && export FMHA_SM100_SRC_DIR="$ROOT/ext/MSA"
[ -d "$ROOT/ext/qutlass" ]         && export QUTLASS_SRC_DIR="$ROOT/ext/qutlass"
[ -d "$ROOT/ext/tml-fa4" ]         && export TML_FA4_SRC_DIR="$ROOT/ext/tml-fa4"
# triton_kernels.cmake expects this to point DIRECTLY at the python package
# directory, not at the triton checkout root (its non-SRC_DIR branch appends
# python/triton_kernels/triton_kernels as SOURCE_SUBDIR and its SRC_DIR branch
# does not).
TKDIR="$ROOT/ext/triton/python/triton_kernels/triton_kernels"
[ -d "$TKDIR" ] && export TRITON_KERNELS_SRC_DIR="$TKDIR"
echo "TRITON_KERNELS_SRC_DIR=${TRITON_KERNELS_SRC_DIR:-absent}"

echo "--- every FetchContent GIT_REPOSITORY the target declares, and whether it is overridden ---"
grep -rhoE 'GIT_REPOSITORY[[:space:]]+[^[:space:]]+' "$SRC/CMakeLists.txt" "$SRC/cmake/" 2>/dev/null | sort -u | sed 's/^/FETCHREPO /'
for v in VLLM_CUTLASS_SRC_DIR VLLM_FLASH_ATTN_SRC_DIR FLASH_MLA_SRC_DIR FLASH_KDA_SRC_DIR \
         DEEPGEMM_SRC_DIR FMHA_SM100_SRC_DIR QUTLASS_SRC_DIR TML_FA4_SRC_DIR TRITON_KERNELS_SRC_DIR; do
  eval "echo \"OVERRIDE $v=\${$v:-UNSET}\""
done
# A previous failed configure leaves a half-populated FetchContent tree that can
# make a retry fail for the OLD reason.
rm -rf "$SRC/.deps"; echo "DEPS_CLEARED_RC=$?"

step "4. VENV AND DEPENDENCIES"
if [ ! -x "$V/bin/python" ]; then
  python3 -m venv "$V" > "$RUN/venv.log" 2>&1; echo "VENV_RC=$?"
fi
PY="$V/bin/python"
"$PY" -m pip install -q --upgrade pip setuptools wheel > "$RUN/pipup.log" 2>&1; echo "PIPUP_RC=$?"
cd "$SRC"
"$PY" -m pip install -r requirements/build/cuda.txt > "$RUN/builddeps.log" 2>&1; echo "BUILDDEPS_RC=$?"
tail -3 "$RUN/builddeps.log"
"$PY" -m pip install -r requirements/cuda.txt > "$RUN/rundeps.log" 2>&1; echo "RUNDEPS_RC=$?"
tail -5 "$RUN/rundeps.log"
"$PY" -m pip list > "$RUN/piplist.txt" 2>&1
grep -E '^(torch|transformers|flashinfer-python|flashinfer-cubin|instanttensor|nvidia-cuda-nvcc|numpy|triton) ' "$RUN/piplist.txt" | sed 's/^/PIPLIST /'

step "5. THE SOURCE BUILD -- the leg no job at this target has ever run"
# VLLM_USE_PRECOMPILED is what every prior job set. It downloads an x86_64 wheel
# that does not exist for aarch64 and leaves the package without vllm._C. This
# job compiles.
unset VLLM_USE_PRECOMPILED
export VLLM_USE_PRECOMPILED=0
export VLLM_TARGET_DEVICE=cuda
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=4          # AGENTS.md: unconstrained parallelism has OOM-rebooted a fleet box
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST="${CC:-11.0}"
echo "BUILD_ENV VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
echo "BUILD_ENV MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "BUILD_ENV CUDA_HOME=$CUDA_HOME"
echo "BUILD_ENV VLLM_CUTLASS_SRC_DIR=${VLLM_CUTLASS_SRC_DIR:-unset} VLLM_FLASH_ATTN_SRC_DIR=${VLLM_FLASH_ATTN_SRC_DIR:-unset}"
echo "BUILD_ENV FLASH_MLA_SRC_DIR=${FLASH_MLA_SRC_DIR:-unset} FLASH_KDA_SRC_DIR=${FLASH_KDA_SRC_DIR:-unset}"

WHEELDIR=$S/wheel
mkdir -p "$WHEELDIR"
EXISTING=$(ls "$WHEELDIR"/vllm-*.whl 2>/dev/null | head -1)
if [ -n "$EXISTING" ]; then
  echo "WHEEL_REUSED=$EXISTING  (a previous job in this wave already built it)"
  SRCBUILD_RC=0
  cp "$EXISTING" "$ROOT/"
  WHEEL="$ROOT/$(basename "$EXISTING")"
else
  # Heartbeat: rc kills a job that produces no output for the idle timeout, and
  # a 100-minute silent compile looks exactly like a hung job.
  ( while true; do
      sleep 120
      echo "HEARTBEAT t=${SECONDS}s load=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB last=$(tail -1 "$RUN/build.log" 2>/dev/null | tr -d '\r' | tail -c 160)"
    done ) &
  HB=$!
  "$PY" -m pip wheel --no-build-isolation --no-deps -w "$ROOT/dist" . > "$RUN/build.log" 2>&1
  SRCBUILD_RC=$?
  kill "$HB" 2>/dev/null
  echo "SRCBUILD_RC=$SRCBUILD_RC"
  echo "--- build log, the FIRST error and then the tail ---"
  grep -n -m 20 -E 'error:|Error |CMake Error|fatal error|No such file' "$RUN/build.log" | head -20
  echo "--- tail ---"
  tail -30 "$RUN/build.log"
  WHEEL=$(ls "$ROOT/dist"/vllm-*.whl 2>/dev/null | head -1)
  echo "WHEEL=${WHEEL:-none}"
  if [ -n "$WHEEL" ]; then
    ls -l "$WHEEL"; sha256sum "$WHEEL"
    cp "$WHEEL" "$WHEELDIR/"; echo "WHEEL_PERSIST_RC=$?"
  fi
fi
echo "SRCBUILD_RC=$SRCBUILD_RC"
if [ "$SRCBUILD_RC" -ne 0 ] || [ -z "${WHEEL:-}" ]; then
  echo "The source build did not produce a wheel. Everything below is an ABSENCE."
  echo "EXT_PRESENT=SKIPPED_NO_BUILD"
  echo "RUN_RC=SKIPPED_NO_BUILD"
  cp "$RUN/build.log" "$RUN/build-failed.log" 2>/dev/null
  echo "DONE_MARKER_RUNHALF"
  exit 22
fi

step "6. INSTALL AND IDENTITY, asserted from OUTSIDE any source tree"
"$PY" -m pip install "$WHEEL" > "$RUN/install.log" 2>&1; echo "INSTALLWHEEL_RC=$?"
tail -3 "$RUN/install.log"
cd /
"$PY" - <<'PY' > "$RUN/identity.txt" 2>&1
import vllm
print("VLLM_FILE", vllm.__file__)
print("VLLM_VERSION", vllm.__version__)
assert "site-packages" in vllm.__file__, f"not an installed package: {vllm.__file__}"
assert "e126687a9" in vllm.__version__, f"WRONG COMMIT: {vllm.__version__}"
print("IDENTITY OK")
PY
IMPORT_RC=$?
echo "IMPORT_RC=$IMPORT_RC"
cat "$RUN/identity.txt"
[ "$IMPORT_RC" -ne 0 ] && { echo "ABORT: instrument failure, not a statement about the model."; echo "DONE_MARKER_RUNHALF"; exit 23; }

step "6b. EXT_PRESENT -- the question every prior job answered NO by construction"
# Job B asked `import vllm._C` and got EXT_RC=1 while the wheel it had just
# built carried SEVEN compiled extensions. `_C` is the name every prior HEADPIN
# job probed and it does not exist at this revision. ext.py enumerates the real
# module names AND executes a compiled kernel, because a name is not an
# execution.
cp "$S/ext.py" "$ROOT/ext.py"; echo "EXTPY_COPY_RC=$?"
"$PY" "$ROOT/ext.py" > "$RUN/ext.txt" 2>&1
EXT_RC=$?
echo "EXT_RC=$EXT_RC"
cat "$RUN/ext.txt"
if [ "$EXT_RC" -eq 0 ]; then echo "EXT_PRESENT=True"; else echo "EXT_PRESENT=False"; fi
echo "--- what the wheel actually shipped ---"
"$PY" -c "
import os, vllm
d=os.path.dirname(vllm.__file__)
for r,_,fs in os.walk(d):
    for f in fs:
        if f.endswith('.so'):
            p=os.path.join(r,f)
            print('WHEEL_SO', os.path.relpath(p,d), os.path.getsize(p))
" 2>&1
"$PY" -c "import torch;print('TORCH', torch.__version__, 'CUDA_AVAIL', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" 2>&1
echo "TORCHCUDA_RC=$?"
"$PY" -c "
from vllm import ModelRegistry
a=ModelRegistry.get_supported_archs()
print('SUPPORTED_ARCH_COUNT', len(a))
for n in ('Qwen4ExpForCausalLM','Qwen4ExpForConditionalGeneration','Qwen4ExpMTP'):
    print('REG_'+n, n in a)
" 2>&1
echo "REG_RC=$?"

unset VLLM_CUTLASS_SRC_DIR VLLM_FLASH_ATTN_SRC_DIR FLASH_MLA_SRC_DIR FLASH_KDA_SRC_DIR \
      DEEPGEMM_SRC_DIR FMHA_SM100_SRC_DIR QUTLASS_SRC_DIR TML_FA4_SRC_DIR TRITON_KERNELS_SRC_DIR

step "7. THE RUN -- a model generating tokens greedily, watchdogged"
# rm first: `cp -rL src dst` NESTS when dst exists, and a second job in the
# same reused container silently built $ROOT/opt-125m/opt-125m.
rm -rf "$ROOT/opt-125m"
cp -rL "$S/models/opt-125m" "$ROOT/opt-125m"
echo "MODEL_COPY_RC=$?"
ls -la "$ROOT/opt-125m" | head -12
cp "$S/gen.py" "$ROOT/gen.py"; echo "GENPY_COPY_RC=$?"
sha256sum "$ROOT/gen.py"

FLOOR_MB=20000
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A $(awk '/^MemTotal/{print int($2/1024)}' /proc/meminfo)" >> "$RUN/mem.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      G=$(cat /tmp/runhalf.pgid 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing process GROUP $G"
      [ -n "$G" ] && kill -9 -- "-$G" 2>/dev/null
    fi
    sleep 1
  done ) > "$RUN/watchdog.log" 2>&1 &
WD=$!
echo "FLOOR_MB=$FLOOR_MB"

run_leg() {   # $1 = tag, $2..= env assignments
  local tag=$1; shift
  step "7.$tag"
  env "$@" MODEL="$ROOT/opt-125m" GMU="${GMU:-0.10}" MAXLEN="${MAXLEN:-2048}" MAXTOK=16 \
      OUTDIR="$RUN" setsid timeout 1800 "$PY" "$ROOT/gen.py" > "$RUN/gen.$tag.out" 2>&1 &
  local pid=$!
  echo "$pid" > /tmp/runhalf.pgid
  wait "$pid"; local rc=$?
  rm -f /tmp/runhalf.pgid
  echo "GEN_RC[$tag]=$rc"
  kill -9 -- "-$pid" 2>/dev/null
  sleep 5
  grep -E '^(GEN_|VLLM_|TOKENS|PROMPT|OUTPUT|TEXT|BACKEND|CFG|DONE_MARKER_GEN)' "$RUN/gen.$tag.out"
  grep -oE 'Using [A-Z_]+ attention backend|Using FlashAttention version [0-9]+' "$RUN/gen.$tag.out" | sort -u | sed 's/^/BACKEND_SELECTED /'
  if [ "$rc" -ne 0 ]; then
    echo "--- $tag did NOT complete. ABSENCE, not a measurement. First error then tail: ---"
    grep -n -m 8 -E 'Error|error:|Traceback|raise |assert' "$RUN/gen.$tag.out" | head -8
    tail -20 "$RUN/gen.$tag.out"
  fi
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  echo "MemAvailable now: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
  return $rc
}

# Leg 1: eager. The recorded failure mode on this fleet is a host consumed in
# the step AFTER torch.compile, so the first leg removes that variable.
LAST_BACKEND=default
run_leg eager VLLM_LEG=eager EAGER=1 BACKEND=default
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
  # vLLM's default FLASH_ATTN is built here from FA2_ARCHS "8.0+PTX", so on this
  # device it reaches sm_110 only by a driver JIT of compute_80 PTX. That is the
  # exact mode that failed on GB10 with cudaErrorUnsupportedPtxVersion
  # (/workspace/oracle-vllm/README-WHEELS.md), and the recorded workaround on
  # this fleet is TRITON_ATTN or FLASHINFER.
  LAST_BACKEND=TRITON_ATTN
  run_leg eager-triton VLLM_LEG=eager-triton EAGER=1 BACKEND=TRITON_ATTN
  RUN_RC=$?
fi
if [ "$RUN_RC" -ne 0 ]; then
  LAST_BACKEND=FLASHINFER
  run_leg eager-flashinfer VLLM_LEG=eager-flashinfer EAGER=1 BACKEND=FLASHINFER
  RUN_RC=$?
fi
echo "RUN_RC=$RUN_RC"
echo "RUN_BACKEND=$LAST_BACKEND"

# Leg 2: the production path, compiled and cudagraph-captured. Only attempted
# once an eager leg has proved the engine runs at all.
COMPILED_RC=SKIPPED
if [ "$RUN_RC" -eq 0 ]; then
  run_leg compiled VLLM_LEG=compiled EAGER=0 BACKEND="$LAST_BACKEND"
  COMPILED_RC=$?
fi
echo "COMPILED_RC=$COMPILED_RC"

step "8. STRETCH -- does the qwen4_exp GRAPH execute here, on RANDOM weights?"
# The published safetensors arms of Qwen3.8-Flash-Next exceed the largest fleet
# box, and upstream's own test registry marks all three architectures
# is_available_online=False. This leg is NOT a token gate and NOT a parity
# statement: it is a shrunk config with load_format="dummy", so the weights are
# random and the tokens are meaningless. What it can say is whether the
# qwen4_exp forward, its sampler and its KV path execute on this device.
Q4EXP_RC=SKIPPED
if [ "$RUN_RC" -eq 0 ]; then
  rm -rf "$ROOT/q4exp"; cp -rL "$S/models/qwen4exp-config" "$ROOT/q4exp"; echo "Q4EXP_COPY_RC=$?"
  cp "$S/q4exp_dummy.py" "$ROOT/q4exp_dummy.py"; echo "Q4EXPPY_COPY_RC=$?"
  # Two legs of the SAME binary and the SAME config. Only the batch differs,
  # and the batch is what selects the cooperative_topk cluster size.
  for NB in 1 6; do
    env NBATCH=$NB setsid timeout 1800 "$PY" "$ROOT/q4exp_dummy.py" "$ROOT/q4exp" > "$RUN/q4exp.n$NB.out" 2>&1 &
    qpid=$!; echo "$qpid" > /tmp/runhalf.pgid
    wait "$qpid"; rc=$?
    rm -f /tmp/runhalf.pgid; kill -9 -- "-$qpid" 2>/dev/null
    echo "Q4EXP_RC[nbatch=$NB]=$rc"
    grep -E '^(Q4|CFG|TOKENS|OUTPUT|TEXT|DONE_MARKER)' "$RUN/q4exp.n$NB.out"
    grep -oE 'cooperative_topk launch failed: [^"]*|cluster misconfiguration' "$RUN/q4exp.n$NB.out" | sort -u | sed 's/^/Q4 KERNELERR /'
    if [ "$rc" -ne 0 ]; then
      grep -E 'core.py:[0-9]+\]' "$RUN/q4exp.n$NB.out" | sed 's/.*core.py:[0-9]*\] //' | tail -6
    fi
    [ "$NB" = "6" ] && Q4EXP_RC=$rc
  done
fi
echo "Q4EXP_RC=$Q4EXP_RC"

kill -9 "$WD" 2>/dev/null

step "9. SUMMARY -- read every rc literally"
echo "SUM TARGET=$TARGET"
echo "SUM HEAD_SHA=$HEAD_SHA"
echo "SUM SRCBUILD_RC=$SRCBUILD_RC"
echo "SUM EXT_RC=$EXT_RC"
echo "SUM IMPORT_RC=$IMPORT_RC"
echo "SUM RUN_RC=$RUN_RC"
echo "SUM COMPILED_RC=$COMPILED_RC"
echo "SUM RUN_BACKEND=$LAST_BACKEND"
grep -h -E '^(VLLM_VERSION|TOKENS|OUTPUT_TOKEN_IDS|TEXT|BACKEND|CFG)' "$RUN"/gen.*.out "$RUN"/q4exp.out 2>/dev/null
awk 'NF>=3{if(min==""||$2<min)min=$2; tot=$3} END{if(min!="")print "MemTotal_MB="tot" minMemAvailable_MB="min" samples="NR}' "$RUN/mem.samples" 2>/dev/null
echo "ALL LOGS: $RUN"
echo "DONE_MARKER_RUNHALF"
