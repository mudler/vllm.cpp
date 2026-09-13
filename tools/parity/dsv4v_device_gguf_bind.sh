#!/bin/bash
# DeepSeek-V4 Flash Vision: does a served image get PAST the host-GEMM refusal
# once `ForwardDevice` binds the keep-quant tower? Measured on thor:gpu0.
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411 and
# ISSUE-LOCAL-01M29KEXRT2GCS6C53DT2S3SPX.
#
#   rc cp ./head3.tar.gz thor:gpu0:/workspace/dsv4v-matvec/
#   rc run -d thor:gpu0 --max-runtime 180m -- \
#     bash -c 'TARBALL=head3.tar.gz bash /workspace/dsv4v-matvec/dsv4v_device_gguf_bind.sh'
#
# `TARBALL` names the staged tree to measure and defaults to `head2.tar.gz`,
# which is the tree the first run of this recipe measured.
#
# WHAT THIS MEASURES. The previous wave (out-20260912-223721) recorded, from the
# served image on a CUDA build:
#
#   vt: deepseek-v4 host GEMM: weight size mismatch: tensor `wq_a` layer 0
#   want [N=32,K=32] = 1024 elements, got 0 elements
#
# `got 0` because `ForwardDevice` never bound the GGUF tower, so every GEMM fell
# to a host arm the loader leaves empty on a GGUF load. This job runs the SAME
# served-image test against the tree that binds it, and gates on ONE question:
# does that refusal still name `wq_a` layer 0?
#
# THE GATE IS THE MESSAGE AND THE COUNTED LINE, NEVER THE RAW EXIT CODE.
# `test_deepseek_v4_mm_chat` used to carry a CPU-only-premise assertion -- it
# expected the served error to name `W7-device`, which a build carrying the V4
# device kernels can NEVER emit, because `kDevicePending` fires only when they
# are absent. Measured on this box before that was repaired, the suite read
# `test cases: 8 | 7 passed | 1 failed` with that one assertion as the failure.
# The expectation is now device-aware, so the suite must be GREEN here and
# `mm_chat_suite_green` gates on `0 failed` read off doctest's COUNTED line.
# The raw exit code is still not the gate, and `Status:` never is.
#
# `pipefail` so a `cmd | tee f` reports the command's status and not tee's.
set -uo pipefail

W=/workspace/dsv4v-matvec
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT=$W/bind-$STAMP; mkdir -p "$OUT"
SRC=/tmp/dsv4v-bind
ARCH=110
NEED_GB=${NEED_GB:-60}
# Which staged tree to measure. Named rather than hard-coded, so a later wave
# does not have to OVERWRITE an earlier wave's tarball to reuse this recipe --
# overwriting is how a run ends up measuring a tree nobody can identify after
# the fact.
TARBALL=${TARBALL:-head2.tar.gz}

free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/steps.txt"; }
: > "$OUT/steps.txt"

# doctest reads its verdict off `test cases:`, NEVER off `Status:` -- a `-tc`
# that matches nothing prints `test cases: 0` AND `SUCCESS!`.
doctest_line() {
  local line n
  line=$(grep -E '^\[doctest\] test cases:' "$1" | tail -1)
  echo "    doctest: ${line:-<NO test cases LINE AT ALL>}"
  case "$line" in *"test cases:"*) : ;; *) return 2 ;; esac
  n=$(echo "$line" | sed -E 's/.*test cases: *([0-9]+).*/\1/')
  [ "${n:-0}" -gt 0 ] || return 2
}

cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G"; done ) &
HB=$!

echo "### identity $(date -u +%FT%TZ)"; uname -m; nproc
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
echo "OUTDIR=$OUT"
rm -rf "$SRC"
[ "$(free_gb)" -ge "$NEED_GB" ] || { echo "REFUSING: /tmp $(free_gb)G < ${NEED_GB}G"; step disk 95; exit 95; }

command -v ccache >/dev/null 2>&1 || { apt-get update -qq; apt-get install -y -qq ccache; }
export CCACHE_DIR=/workspace/ccache; mkdir -p "$CCACHE_DIR"

echo "### cuda toolkit"
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
dpkg -i /tmp/ck.deb >/dev/null 2>&1
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0; step toolkit_install $?
export PATH=/usr/local/cuda/bin:$PATH
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
command -v nvcc >/dev/null || { echo "FATAL: no nvcc"; step nvcc 90; exit 90; }
case "$(nvcc --version | grep -o 'release [0-9]*' | head -1)" in
  "release 13") : ;;
  *) echo "FATAL: nvcc is not 13.x; sm_110 needs CUDA 13"; step nvcc_ver 90; exit 90 ;;
esac

mkdir -p "$SRC"
echo "### measuring tarball: $TARBALL"
tar -xzf "$W/$TARBALL" -C "$SRC" || { step untar 92; exit 92; }
test -f "$SRC/CMakeLists.txt" || { echo "FATAL: untar"; step untar 92; exit 92; }

# ASSERT THE TREE REALLY CARRIES THE FIX. A staging slip that shipped the
# previous head would reproduce the old message and read as "the fix does
# nothing", which is the loudest possible wrong conclusion.
grep -q 'has_gguf_weights) dev_be.gguf' "$SRC/src/vllm/model_executor/models/deepseek_v4.cpp"
step tree_carries_fix $?

cmake -S "$SRC" -B "$SRC/b" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=$ARCH -DVLLM_CPP_TRITON=OFF \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > "$OUT/configure.log" 2>&1
RC=$?; step configure $RC
[ $RC -eq 0 ] || { tail -30 "$OUT/configure.log"; exit 91; }

cmake --build "$SRC/b" -j 4 --target test_deepseek_v4_mm_chat test_deepseek_v4_forward \
  test_deepseek_v4_mm_reach > "$OUT/build.log" 2>&1
RC=$?; step build $RC
[ $RC -eq 0 ] || { grep -m15 -E 'error' "$OUT/build.log"; exit 94; }
md5sum "$SRC/b/tests/test_deepseek_v4_mm_chat" | tee "$OUT/md5-mm_chat.txt"

# ===========================================================================
# THE MEASUREMENT: what does the served image say now?
# ===========================================================================
echo "### the served image, with the tower bound"
"$SRC/b/tests/test_deepseek_v4_mm_chat" -s > "$OUT/mm_chat.log" 2>&1
echo "mm_chat exit=$? (RECORDED, NOT GATED -- see the header)"
doctest_line "$OUT/mm_chat.log"; step mm_chat_ran $?

# THE SUITE MUST BE GREEN once the image expectation is device-aware. Before
# that repair this suite read `test cases: 8 | 7 passed | 1 failed` on this box,
# the single failure being the unsatisfiable `W7-device` assertion. Read the
# COUNTED line; `Status:` prints SUCCESS even when a filter selected nothing.
if grep -E '^\[doctest\] test cases:' "$OUT/mm_chat.log" | tail -1 | grep -q '0 failed'; then
  step mm_chat_suite_green 0
else
  echo "mm_chat is NOT green; the failing assertions are:"
  grep -E 'ERROR:' "$OUT/mm_chat.log" | head -10
  step mm_chat_suite_green 1
fi
# `MESSAGE: image: ` is the CASE'S OWN line -- mm_chat prints
# `MESSAGE("image: " << (error.empty() ? "served" : error))`. A bare `image: `
# ALSO matches the PNG and data-URI residual INFO lines, which read
# `chat image: ...` and are NOT the engine's stop. Measured: on the run that
# repaired this, the bare pattern printed three container-decode residuals while
# the real stop sat 70 lines further down. Quoting one of those as "the blocker"
# is how a job reports a cause that is not the cause.
echo "--- the image: line (the case's OWN MESSAGE) ---"
grep -nE 'MESSAGE: image: ' "$OUT/mm_chat.log" | head -5
echo "--- the engine's own stop, if any ---"
grep -nE 'host GEMM|engine-fatal' "$OUT/mm_chat.log" | head -10

# THE GATE. The previous wave died naming `wq_a` layer 0. If that exact refusal
# is still on the image line, binding the tower did not move the blocker.
if grep -q 'tensor `wq_a` layer 0' "$OUT/mm_chat.log"; then
  echo "UNMOVED: the served image still dies in wq_a layer 0"
  step image_past_wq_a 1
else
  echo "MOVED: the wq_a layer 0 refusal is GONE from this run"
  step image_past_wq_a 0
fi
# Did it SERVE, or did it stop somewhere new? Both are results; neither is
# assumed. The line is printed either way and the next blocker, if any, is named.
if grep -qE 'image: served' "$OUT/mm_chat.log"; then
  echo "SERVED: the image request completed"
  step image_served 0
else
  echo "NOT SERVED: the image stopped. THE NEXT BLOCKER, verbatim:"
  grep -nE 'MESSAGE: image: |engine-fatal' "$OUT/mm_chat.log" | head -3
  step image_served 1
fi

# No regression on the suites that were green.
"$SRC/b/tests/test_deepseek_v4_forward" > "$OUT/forward.log" 2>&1; step forward_suite $?
doctest_line "$OUT/forward.log"
VT_CPU_QUANT_REPACK=0 "$SRC/b/tests/test_deepseek_v4_mm_reach" > "$OUT/mm_reach.log" 2>&1
echo "mm_reach (repack OFF) exit=$?"; doctest_line "$OUT/mm_reach.log"; step mm_reach $?

echo "### steps"; cat "$OUT/steps.txt"
# Read the steps back. An ABSENT step is not a pass, so the expected list is
# what separates "never ran" from "ran and returned 0". `image_served` is
# RECORDED rather than required: whether an image serves is what the log says.
EXPECTED="toolkit_install tree_carries_fix configure build mm_chat_ran
          image_past_wq_a mm_chat_suite_green forward_suite mm_reach"
step_rc() { sed -n "s/^$1 RC=\([0-9]*\)\$/\1/p" "$OUT/steps.txt" | tail -1; }
FAIL=0
for s in $EXPECTED; do
  rc=$(step_rc "$s")
  [ -n "$rc" ] || { echo "### MISSING STEP: $s never ran"; FAIL=1; continue; }
  [ "$rc" = 0 ] || { echo "### FAILING STEP: $s RC=$rc"; FAIL=1; }
done
echo "### image_served RC=$(step_rc image_served) (recorded, not gated)"
echo "### OVERALL FAIL=$FAIL"
echo "### DEVICE_GGUF_BIND_DONE OUTDIR=$OUT"
exit $FAIL
