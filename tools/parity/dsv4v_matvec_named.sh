#!/bin/bash
# DeepSeek-V4 Flash Vision: the host GEMM's NAMED refusal, measured on thor:gpu0.
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411 and
# ISSUE-LOCAL-01M29KEXRT2GCS6C53DT2S3SPX.
#
#   rc cp ./base.tar.gz thor:gpu0:/workspace/dsv4v-matvec/
#   rc cp ./head.tar.gz thor:gpu0:/workspace/dsv4v-matvec/
#   rc run -d thor:gpu0 --max-runtime 240m -- \
#     bash /workspace/dsv4v-matvec/dsv4v_matvec_named.sh
#
# WHAT THIS JOB MEASURES, in order:
#   1. RED. The new `test_deepseek_v4_forward` case against the BASE tree, whose
#      host-GEMM guard is the anonymous `MatVec weight size mismatch`. The case
#      asserts the tensor, the layer and both geometries, so it MUST fail there.
#   2. GREEN. The same case against the HEAD tree, which names them. The two test
#      binaries' md5s MUST DIFFER: `-Werror` is on, and a mutant that fails to
#      compile leaves the OLD binary in place, so a re-run prints SUCCESS and
#      measures nothing. A verdict without the md5 comparison is VOID.
#   3. THE SERVED IMAGE. `test_deepseek_v4_mm_chat` on the HEAD CUDA build. Its
#      image branch dies in the host GEMM; this job captures WHAT THE REFUSAL NOW
#      NAMES, which is the tensor/layer/geometry nobody has ever seen.
#
# BOTH TREES ARE BUILT FOR CUDA sm_110. The refusal is host code, but its
# REACHABILITY is not: on a CPU build `ForwardDevice` refuses earlier at
# `VT_CHECK(V4DeviceKernelsAvailable(), kDevicePending)`, so only a build
# carrying the V4 device kernels reaches the served-image failure at all.
#
# thor is sm_110, OUTSIDE the vendored FlashAttention-2 arch set, so every
# FA-2-gated path can only REFUSE here. That is a property of the box.
#
# `pipefail` is required because a `cmd | tee f` pipeline otherwise reports TEE's
# status, and tee succeeds whenever it can write the file. There is no `set -e`,
# so this changes only the value the status readers see.
set -uo pipefail

W=/workspace/dsv4v-matvec
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$W/out-$STAMP
SRC=/tmp/dsv4v-mv
ARCH=110
NEED_GB=${NEED_GB:-60}
mkdir -p "$OUT"

free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/steps.txt"; }
: > "$OUT/steps.txt"

# DOCTEST READS ITS VERDICT OFF `test cases:`, NEVER OFF `Status:`. A `-tc`
# filter that matches nothing prints `test cases: 0` AND `Status: SUCCESS!`, so
# a green read off `Status:` is a run that measured nothing. This helper prints
# the counted line and returns non-zero when the filter selected no case.
doctest_line() {  # $1 log
  local line
  line=$(grep -E '^\[doctest\] test cases:' "$1" | tail -1)
  echo "    doctest: ${line:-<NO test cases LINE AT ALL>}"
  case "$line" in
    *"test cases:"*) : ;;
    *) return 2 ;;
  esac
  # `test cases: 2 | 2 passed | 0 failed |` -- the first number is the selection.
  local n
  n=$(echo "$line" | sed -E 's/.*test cases: *([0-9]+).*/\1/')
  [ "${n:-0}" -gt 0 ] || return 2
  return 0
}

cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G"; done ) &
HB=$!

echo "### identity $(date -u +%FT%TZ)"; uname -m; nproc; free -g | head -2
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
echo "OUTDIR=$OUT"

# --- DISK. The container is REUSED, so other jobs' trees share this overlay.
rm -rf "$SRC"
if [ "$(free_gb)" -lt "$NEED_GB" ]; then
  echo "REFUSING: /tmp has $(free_gb) GiB free, below the NEED_GB=${NEED_GB} floor."
  step disk 95; exit 95
fi

# --- ccache is REQUIRED on this fleet, and its cache lives on the NAS so the
# --- second tree reuses the first tree's objects for every TU they share.
command -v ccache >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq ccache; }
export CCACHE_DIR=/workspace/ccache
mkdir -p "$CCACHE_DIR"

# --- THE CUDA TOOLKIT IS NOT IN THE WORKER IMAGE. Install it UNCONDITIONALLY:
# --- the container is long-lived, so a leftover toolkit from another job is not
# --- a precondition this recipe may rely on. The box's system nvcc is 12.0,
# --- which CANNOT target sm_110 at all.
echo "### cuda toolkit"
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
dpkg -i /tmp/ck.deb >/dev/null 2>&1
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0; step toolkit_install $?
export PATH=/usr/local/cuda/bin:$PATH
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; step nvcc 90; exit 90; }
test -f "$CUDA_HOME/include/cuda_runtime.h" \
  || { echo "FATAL: nvcc present but no cuda_runtime.h"; step cudart_h 90; exit 90; }
nvcc --version | tail -2 | tee "$OUT/nvcc.txt"
case "$(nvcc --version | grep -o 'release [0-9]*' | head -1)" in
  "release 13") : ;;
  *) echo "FATAL: nvcc is not 13.x; sm_110 needs CUDA 13"; step nvcc_ver 90; exit 90 ;;
esac

# --- SOURCE. Both trees are staged by `rc cp` BEFORE the job starts.
mkdir -p "$SRC/base" "$SRC/head"
tar -xzf "$W/base.tar.gz" -C "$SRC/base" || { step untar_base 92; exit 92; }
tar -xzf "$W/head.tar.gz" -C "$SRC/head" || { step untar_head 92; exit 92; }
test -f "$SRC/base/CMakeLists.txt" && test -f "$SRC/head/CMakeLists.txt" \
  || { echo "FATAL: untar"; step untar 92; exit 92; }

# THE TWO TREES MUST DIFFER IN THE PRODUCT AND AGREE IN THE TEST. Asserted, not
# assumed: a staging slip that shipped the same tree twice would otherwise read
# as "the red went green", which is the loudest possible false pass.
echo "### tree delta"
diff -q "$SRC/base/src/vllm/model_executor/models/deepseek_v4.cpp" \
        "$SRC/head/src/vllm/model_executor/models/deepseek_v4.cpp" \
  > /dev/null 2>&1; [ $? -ne 0 ]; step trees_differ_product $?
diff -q "$SRC/base/tests/vllm/models/test_deepseek_v4_forward.cpp" \
        "$SRC/head/tests/vllm/models/test_deepseek_v4_forward.cpp" \
  > /dev/null 2>&1; step trees_share_test $?
grep -c 'MatVec weight size mismatch' \
  "$SRC/base/src/vllm/model_executor/models/deepseek_v4.cpp" | tee "$OUT/base-anon-count.txt"

build_tree() {  # $1 label  $2 tree  $3... targets
  local L=$1 T=$2; shift 2
  echo "### build $L"
  cmake -S "$T" -B "$T/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=$ARCH -DVLLM_CPP_TRITON=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    > "$OUT/configure-$L.log" 2>&1; local rc=$?; step configure_$L $rc
  [ $rc -eq 0 ] || { tail -30 "$OUT/configure-$L.log"; return 1; }
  cmake --build "$T/build-cuda" -j 4 --target "$@" \
    > "$OUT/build-$L.log" 2>&1; rc=$?; step build_$L $rc
  [ $rc -eq 0 ] || { grep -m15 -E 'error' "$OUT/build-$L.log"; return 1; }
  return 0
}

# ===========================================================================
# 1. THE RED, on the BASE tree.
# ===========================================================================
build_tree base "$SRC/base" test_deepseek_v4_forward || { echo "BASE BUILD FAILED"; }
BASE_BIN="$SRC/base/build-cuda/tests/test_deepseek_v4_forward"
if [ -x "$BASE_BIN" ]; then
  md5sum "$BASE_BIN" | tee "$OUT/md5-base.txt"
  "$BASE_BIN" -tc="deepseek-v4 W7-CUDA: the host GEMM names*" -s \
    > "$OUT/red.log" 2>&1; step red_run $?
  doctest_line "$OUT/red.log"; step red_selected $?
  echo "--- the RED message the base tree throws:"
  grep -iE 'MatVec|host GEMM|ERROR|FAILED' "$OUT/red.log" | head -20
else
  step red_binary_missing 94
fi

# ===========================================================================
# 2. THE GREEN, on the HEAD tree.
# ===========================================================================
build_tree head "$SRC/head" test_deepseek_v4_forward test_deepseek_v4_mm_chat \
  || { echo "HEAD BUILD FAILED"; }
HEAD_BIN="$SRC/head/build-cuda/tests/test_deepseek_v4_forward"
if [ -x "$HEAD_BIN" ]; then
  md5sum "$HEAD_BIN" | tee "$OUT/md5-head.txt"
  "$HEAD_BIN" -tc="deepseek-v4 W7-CUDA: the host GEMM names*" -s \
    > "$OUT/green.log" 2>&1; step green_run $?
  doctest_line "$OUT/green.log"; step green_selected $?
  echo "--- the NAMED message the head tree throws:"
  grep -iE 'host GEMM|tensor `|want \[N=' "$OUT/green.log" | head -20
  # THE BINARIES MUST DIFFER. Same md5 => the second build produced nothing and
  # the "green" is the RED binary run twice.
  A=$(awk '{print $1}' "$OUT/md5-base.txt" 2>/dev/null)
  B=$(awk '{print $1}' "$OUT/md5-head.txt" 2>/dev/null)
  echo "md5 base=$A head=$B"
  [ -n "$A" ] && [ -n "$B" ] && [ "$A" != "$B" ]; step md5_differ $?
  # The whole suite, so the named refusal did not break a sibling case.
  "$HEAD_BIN" > "$OUT/head-fullsuite.log" 2>&1; step head_fullsuite $?
  doctest_line "$OUT/head-fullsuite.log"
else
  step green_binary_missing 94
fi

# ===========================================================================
# 3. THE SERVED IMAGE, on the HEAD CUDA build.
# ===========================================================================
echo "### 3. the served image"
MM="$SRC/head/build-cuda/tests/test_deepseek_v4_mm_chat"
if [ -x "$MM" ]; then
  "$MM" -s > "$OUT/mm_chat.log" 2>&1; step mm_chat $?
  doctest_line "$OUT/mm_chat.log"
  echo "--- WHAT THE SERVED IMAGE NOW REPORTS (the point of this job):"
  grep -n 'image: ' "$OUT/mm_chat.log" | head -5
  grep -iE 'host GEMM|tensor `|want \[N=|MatVec|engine-fatal' "$OUT/mm_chat.log" | head -20
  # The aarch64 i8mm repack is a HOST-architecture effect, not a device one, and
  # thor is aarch64. Same binary, one env var, so the attribution is measured.
  VT_CPU_QUANT_REPACK=0 "$MM" -s > "$OUT/mm_chat-repack-off.log" 2>&1
  echo "repack OFF rc=$?"
  grep -n 'image: ' "$OUT/mm_chat-repack-off.log" | head -5
else
  step mm_chat_binary_missing 94
fi

echo "### /tmp free at end: $(free_gb) GiB"
echo "### steps"; cat "$OUT/steps.txt"

# READ THE STEPS BACK. Recording a status nothing reads is the same defect as
# not recording one. Counting non-zero lines is NOT enough either: an ABSENT
# step counts as zero failures, so the expected list is what makes a step that
# never ran distinguishable from one that passed.
#
# `red_run` IS EXPECTED TO BE NON-ZERO and is the only such step: the base tree
# does not name the tensor, so the new case must fail there. A ZERO from it
# falsifies the red and is reported, because a red-first test that passes before
# the repair measures nothing.
EXPECTED="toolkit_install trees_differ_product trees_share_test configure_base
          build_base red_run red_selected configure_head build_head green_run
          green_selected md5_differ head_fullsuite mm_chat"
step_rc() { sed -n "s/^$1 RC=\([0-9]*\)\$/\1/p" "$OUT/steps.txt" | tail -1; }

FAIL=0
for s in $EXPECTED; do
  rc=$(step_rc "$s")
  if [ -z "$rc" ]; then echo "### MISSING STEP: $s never ran"; FAIL=1; continue; fi
  case "$s" in
    red_run)
      if [ "$rc" = 0 ]; then
        echo "### FALSIFIED: red_run RC=0 -- the new case PASSES on the base tree,"
        echo "### so it does not detect the anonymous refusal and is not a red."
        FAIL=1
      else
        echo "### red_run RC=$rc: EXPECTED, this is the red"
      fi ;;
    mm_chat)
      # The served image is the OBSERVATION this job came for. Its exit status is
      # recorded and reported, but it is not asserted either way here: whether an
      # image serves is what the log says, not what this script wishes.
      echo "### mm_chat RC=$rc (recorded, not asserted -- read the 'image: ' line)" ;;
    *)
      [ "$rc" = 0 ] || { echo "### FAILING STEP: $s RC=$rc"; FAIL=1; } ;;
  esac
done
echo "### OVERALL FAIL=$FAIL"
echo "### MATVEC_NAMED_DONE OUTDIR=$OUT"
exit $FAIL
