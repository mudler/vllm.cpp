#!/bin/bash
# DeepSeek-V4 Flash Vision W7-CUDA: the device path, measured on thor:gpu0.
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W7-CUDA, issue #2411.
#
#   rc run -d thor:gpu0 --max-runtime 240m -- \
#     bash /workspace/dsv4-vision/w7-out/src/tools/parity/dsv4v_w7_cuda.sh
#
# thor is sm_110, OUTSIDE the vendored FlashAttention-2 arch set, so every
# FA-2-gated path can only REFUSE here. That is a property of the box and is
# recorded rather than worked around.
#
# What this job measures, in order:
#   1. the row's device-capable suites, with every SKIP named. A skip nobody
#      reads is a gate that measured nothing (#463).
#   2. the W6 probe on a CUDA queue against the SAME mmproj-BF16.gguf, compared
#      with the CPU block W6 left in w6-parity/ by the SAME compare script.
#   3. the three device refusals, driven rather than reasoned about.
#
# Everything lands in /workspace/dsv4-vision/w7-out/ as well as on stdout,
# because rc logs age out within a day.
#
# `pipefail` is required for the same reason `dsv4v_w6_parity.sh` states: a
# `cmd | tee f` pipeline otherwise reports TEE's status, and tee succeeds
# whenever it can write the file. There is no `set -e` here, so this changes
# only the value the status readers see. THIS DRIVER WAS LEFT OUT of the
# 2026-09-12 repair that gave the other three `pipefail` and a steps readback,
# and it is the driver that produced the 2.884% / 0.99939 / 1.872% figures the
# spec quotes: it recorded steps nothing ever read and ended on its DONE banner
# whatever they said.
set -uo pipefail
W=/workspace/dsv4-vision
OUT=$W/w7-out; mkdir -p "$OUT"
P6=$W/w6-parity
MMPROJ=$W/mmproj-BF16.gguf
SRC=/tmp/dsv4v-w7-src
NEED_GB=${NEED_GB:-60}
ARCH=110

free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/steps.txt"; }
cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
: > "$OUT/steps.txt"
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G"; done ) &
HB=$!

echo "### identity $(date -u +%FT%TZ)"; uname -m; nproc; free -g | head -2
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
df -h /tmp

# --- DISK. The container is REUSED, so other jobs' trees share this overlay.
rm -rf /tmp/dsv4v-w7-src /tmp/dsv4v-w6*-src /tmp/dsv4v-w6*-llama
du -sh /tmp/* 2>/dev/null | sort -rh | head -10
if [ "$(free_gb)" -lt "$NEED_GB" ]; then
  echo "REFUSING: /tmp has $(free_gb) GiB free, below the NEED_GB=${NEED_GB} floor."
  step disk 95; exit 95
fi

# --- THE CUDA TOOLKIT IS NOT IN THE WORKER IMAGE. Install it UNCONDITIONALLY:
# --- the container is long-lived, so a leftover /usr/local/cuda-13.0 from
# --- another job is not a precondition this recipe may rely on. A partial
# --- leftover also puts nvcc on PATH with no include/lib64 behind it.
# --- The box carries a system nvcc 12.0, which CANNOT target sm_110 at all.
echo "### cuda toolkit"
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
dpkg -i /tmp/ck.deb >/dev/null 2>&1
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0; step toolkit_install $?
export PATH=/usr/local/cuda/bin:$PATH
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# --- Assert the POSTCONDITION the build needs, not merely the binary.
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; step nvcc 90; exit 90; }
test -f "$CUDA_HOME/include/cuda_runtime.h" \
  || { echo "FATAL: nvcc present but no cuda_runtime.h under $CUDA_HOME"; step cudart_h 90; exit 90; }
ls "$CUDA_HOME"/targets/*/lib/libcudart.so* >/dev/null 2>&1 \
  || ls "$CUDA_HOME"/lib64/libcudart.so*    >/dev/null 2>&1 \
  || { echo "FATAL: no libcudart under $CUDA_HOME"; step cudart 90; exit 90; }
nvcc --version | tail -2 | tee "$OUT/nvcc.txt"
# The version that will actually run, not the one that happened to be on PATH.
case "$(nvcc --version | grep -o 'release [0-9]*' | head -1)" in
  "release 13") : ;;
  *) echo "FATAL: nvcc is not 13.x; sm_110 needs CUDA 13"; step nvcc_ver 90; exit 90 ;;
esac

apt-get install -y -qq cuda-cuobjdump-13-0
if command -v cuobjdump >/dev/null; then CUOBJ=1; else
  CUOBJ=0; echo "### cuobjdump ABSENT after install -- cubin proof stays OWED"
fi

# --- SOURCE. Staged by `rc cp` from the W7 worktree into $W, NOT into $OUT:
# --- $OUT is created by this script, so a tarball addressed there could not
# --- have been copied in before the job started. The SHA is printed so the
# --- measurement names the tree it came from.
mkdir -p "$SRC"
tar -xzf "$W/w7-src.tar.gz" -C "$SRC" || { step untar 92; exit 92; }
test -f "$SRC/CMakeLists.txt" || { echo "FATAL: untar"; step untar 92; exit 92; }
BASE_SHA=$(cat "$W/w7-BASE_SHA" 2>/dev/null)
echo "### base sha: $BASE_SHA"
echo "$BASE_SHA" > "$OUT/BASE_SHA"

# --- BUILD, CUDA ON, sm_110.
cmake -S "$SRC" -B "$SRC/build-cuda" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=$ARCH -DVLLM_CPP_TRITON=OFF \
  > "$OUT/configure.log" 2>&1; RC=$?; step configure $RC
grep -E 'CUDA feature|CUDA compiler identification|FA2|fa2' "$OUT/configure.log" | tee "$OUT/cuda-features.txt"
[ $RC -eq 0 ] || { tail -30 "$OUT/configure.log"; exit 91; }

# The probe is NOT in the default build; W6 added it to a scratch copy the same way.
printf '\nadd_executable(dsv4v-w6-probe ${CMAKE_SOURCE_DIR}/tools/parity/dsv4v_w6_probe.cpp)\ntarget_link_libraries(dsv4v-w6-probe PRIVATE vllm::vllm)\n' >> "$SRC/examples/CMakeLists.txt"
cmake -S "$SRC" -B "$SRC/build-cuda" >> "$OUT/configure.log" 2>&1; step reconfigure $?

TESTS=$(cd "$SRC/build-cuda" && ctest -N -R 'deepseek_v4|clip_mmproj_gguf' 2>/dev/null \
        | grep -oP 'Test\s+#\d+:\s+\K\S+' | sort -u \
        | grep -v 'test_deepseek_v4_exl3_forward_loop_arm' | tr '\n' ' ')
echo "### test targets: $TESTS"
cmake --build "$SRC/build-cuda" -j 4 --target dsv4v-w6-probe $TESTS \
  > "$OUT/build.log" 2>&1; RC=$?; step build $RC
tail -5 "$OUT/build.log"
[ $RC -eq 0 ] || { grep -m15 -E 'error' "$OUT/build.log"; exit 94; }

# --- PROVE IT IS A CUDA BUILD before believing any result below.
#
# `ldd` MUST target a linked EXECUTABLE, not the library. CMakeLists.txt:732 is
# `add_library(vllm STATIC ...)`, so this tree produces `libvllm.a` and there is
# no `libvllm.so` to inspect -- the first run of this recipe asked for one and
# got "No such file or directory", which is a broken proof line rather than a
# finding about the build. The cubin histogram below is the independent proof
# and it stood on that run: 41 `.cu.o`, all sm_110.
LDD_TARGET=$(find "$SRC/build-cuda/tests" -maxdepth 1 -name 'test_cuda_deepseek_v4' -type f | head -1)
if [ -n "$LDD_TARGET" ]; then
  ldd "$LDD_TARGET" | grep -Ei 'cudart|cublas' | tee "$OUT/ldd.txt"
  test -s "$OUT/ldd.txt" || echo "### WARNING: no cudart/cublas in $LDD_TARGET -- NOT a CUDA link"
else
  echo "### WARNING: no test_cuda_deepseek_v4 executable to ldd; linkage proof OWED"
fi
find "$SRC/build-cuda" -name '*.cu.o' | wc -l | tee "$OUT/cu-objects.txt"
if [ "$CUOBJ" -eq 1 ]; then
  for o in $(find "$SRC/build-cuda" -name '*.cu.o'); do
    echo "== $o"; cuobjdump --list-elf "$o"
  done > "$OUT/cubin.log" 2>&1
  grep -o 'sm_[0-9]*' "$OUT/cubin.log" | sort | uniq -c | tee "$OUT/cubin-arch.txt"
  echo "objects scanned: $(grep -c '^== ' "$OUT/cubin.log")"
fi

# ===========================================================================
# 1. THE DEVICE SUITES, with every skip named.
# ===========================================================================
echo "### 1. device suites"
( cd "$SRC/build-cuda" && ctest -R 'deepseek_v4|clip_mmproj_gguf' -j1 --timeout 1800 \
    --output-on-failure ) > "$OUT/ctest-cuda.log" 2>&1; step ctest_cuda $?
tail -25 "$OUT/ctest-cuda.log"
echo "--- SKIPPED cases, and the reason each one printed:"
grep -nE 'SKIPPED|Skipped|no CUDA|skip' "$OUT/ctest-cuda.log" | tee "$OUT/skips.txt"

# The CUDA kernel suite by itself, verbosely, so each case's own SKIP message is
# attributable rather than summarised.
( cd "$SRC/build-cuda" && ./tests/test_cuda_deepseek_v4 -s ) \
  > "$OUT/test_cuda_deepseek_v4.log" 2>&1; step cuda_kernels $?
tail -20 "$OUT/test_cuda_deepseek_v4.log"

# ===========================================================================
# 2. THE REAL TOWER ON THE DEVICE, against the CPU block W6 recorded.
# ===========================================================================
echo "### 2. the real tower on a CUDA queue"
PROBE=$(find "$SRC/build-cuda" -name dsv4v-w6-probe -type f | head -1)
test -n "$PROBE" || { echo "FATAL: no probe binary"; step probe_missing 94; exit 94; }

# A CPU control from THIS binary first. It must reproduce W6's block byte for
# byte, which is what makes the device comparison below a device result rather
# than a difference between two builds.
"$PROBE" "$MMPROJ" "$P6/img392.rgb" 392 392 0 "$OUT" cpuctl \
  > "$OUT/probe-cpuctl.log" 2>&1; step probe_cpu $?
tail -3 "$OUT/probe-cpuctl.log"
# THIS CONTROL IS LOAD-BEARING, so it records a STEP like every other check.
# It previously wrote the bare line "cpu_control identical" into steps.txt --
# not `<name> RC=<n>`, so no readback could ever parse it -- and the DIFFERS
# branch recorded nothing at all, which made "this build is not W6's" a message
# on stdout rather than a result. Every device number below rests on this
# comparison holding.
if cmp -s "$OUT/ours-cpuctl-block.f32" "$P6/ours-lp0-block.f32"; then
  echo "CPU_CONTROL_IDENTICAL: this build reproduces W6's CPU block byte for byte"
  step cpu_control 0
else
  echo "CPU_CONTROL_DIFFERS: this build's CPU block is NOT W6's. Any device"
  echo "comparison below would be against THIS build's CPU arm, not against W6's."
  step cpu_control 1
fi

# The device arm, at every lead_pad rung W6 measured.
for LP in 0 1 2 3; do
  DSV4V_PROBE_DEVICE=cuda "$PROBE" "$MMPROJ" "$P6/img392.rgb" 392 392 $LP "$OUT" cuda-lp$LP \
    > "$OUT/probe-cuda-lp$LP.log" 2>&1; RC=$?
  echo "--- lead_pad $LP rc=$RC"
  tail -5 "$OUT/probe-cuda-lp$LP.log"
  if [ $RC -ne 0 ]; then
    # A REFUSAL IS A RECORDED RESULT ON THIS BOX, NOT A FAILING STEP. thor is
    # sm_110, outside the vendored FA-2 arch set, so an FA-2-gated path can only
    # refuse here and the message is what this job came to collect. It is
    # recorded under its OWN step name so that the readback below still fails on
    # a non-zero exit nobody can explain: an unexplained crash must not be
    # filed as "the expected refusal".
    echo "DEVICE ARM REFUSED at lead_pad $LP -- the message is the result:"
    grep -iE 'refus|device|FATAL|what|share one device' "$OUT/probe-cuda-lp$LP.log" | head -5
    if grep -qiE 'refus|unsupported|share one device|must be' "$OUT/probe-cuda-lp$LP.log"; then
      step probe_cuda_lp${LP}_refused 0
    else
      echo "UNEXPLAINED non-zero exit $RC with no refusal message in the log."
      step probe_cuda_lp${LP}_unexplained "$RC"
    fi
    continue
  fi
  step probe_cuda_lp$LP 0
  # Compare the DEVICE block against W6's CPU block with W6's own script and
  # statistics, so the number is judged against the recorded bound and not a
  # fresh one. "oracle" here is the CPU arm; the file names say so.
  #
  # EVERY COPY IS CHECKED, and the destination is removed first. $OUT is a
  # persistent NAS directory nothing clears, so a failed `cp` used to leave the
  # PREVIOUS run's file in place and the comparator judged that instead --
  # silently, against a stale artefact this run never produced.
  CPRC=0
  for s in block vit cells input; do
    rm -f "$OUT/oracle-cuda-lp$LP-$s.f32"
    cp "$P6/ours-lp$LP-$s.f32" "$OUT/oracle-cuda-lp$LP-$s.f32" || CPRC=1
  done
  step stage_copy_cuda_lp$LP $CPRC
  [ $CPRC -eq 0 ] || { echo "FATAL: staging W6's lp$LP files failed; refusing to compare"; continue; }
  python3 "$SRC/tools/parity/dsv4v_w6_compare.py" "$OUT" cuda-lp$LP $LP 10 10 \
    > "$OUT/compare-cuda-lp$LP.txt" 2>&1; step compare_cuda_lp$LP $?
  cat "$OUT/compare-cuda-lp$LP.txt"
done

# ===========================================================================
# 3. THE THREE DEVICE REFUSALS, driven rather than reasoned about.
# ===========================================================================
echo "### 3. the device refusals"
# (a) dev_attn: needs VT_V4_DEVICE_ATTN=1 + a CUDA queue + the V4 kernels
#     together, which is why NO CPU BUILD can execute it and W4 recorded it
#     unmeasured. This is the run that can.
( cd "$SRC/build-cuda" && VT_V4_DEVICE_ATTN=1 ./tests/test_cuda_deepseek_v4 -s ) \
  > "$OUT/dev-attn-on.log" 2>&1; step dev_attn_on $?
grep -iE 'sliding_window|image span|DEVICE decode|2411|refus' "$OUT/dev-attn-on.log" \
  | head -20 | tee "$OUT/dev-attn-refusal.txt"

# (b) the two device routers, and (c) the paged image-span arm, through the
#     registered forward on a CUDA build.
for T in test_deepseek_v4_mm_reach test_deepseek_v4_forward test_deepseek_v4_dsa; do
  if [ -x "$SRC/build-cuda/tests/$T" ]; then
    ( cd "$SRC/build-cuda" && VT_V4_DEVICE_ATTN=1 VT_V4_DEVICE_GLUE=1 ./tests/$T -s ) \
      > "$OUT/$T-deviceflags.log" 2>&1; step ${T}_deviceflags $?
    echo "--- $T under the device flags:"; tail -12 "$OUT/$T-deviceflags.log"
  fi
done

# ===========================================================================
# 4. ATTRIBUTE THE SUITE FAILURES, rather than inferring them.
# ===========================================================================
# The first W7-CUDA run had 8 of 20 `mm_reach` cases fail with
#   "keep-quant expert/group slice requires non-repacked blocks
#    (disable VT_CPU_QUANT_REPACK for the stacked-expert weights)"
# That predicate is `vt::cpu::QuantRepackActive()`, which is true only on an
# aarch64 i8mm HOST -- so the cause is the host architecture, not the device,
# and thor is aarch64 while the devbox is x86-64 (where these cases are green).
# Reading that off the message is a hypothesis. Running the same binary with the
# repack disabled is the measurement, and it is one env var.
echo "### 4. aarch64 repack attribution (same binary, VT_CPU_QUANT_REPACK=0)"
for T in test_deepseek_v4_mm_reach test_deepseek_v4_mm_chat; do
  [ -x "$SRC/build-cuda/tests/$T" ] || continue
  ( cd "$SRC/build-cuda" && ./tests/$T ) > "$OUT/$T-repack-on.log" 2>&1
  echo "$T repack ON  rc=$?"
  ( cd "$SRC/build-cuda" && VT_CPU_QUANT_REPACK=0 ./tests/$T ) \
    > "$OUT/$T-repack-off.log" 2>&1
  echo "$T repack OFF rc=$?"
  echo "--- $T: repack ON vs OFF, doctest totals"
  grep -E "^\[doctest\] test cases:" "$OUT/$T-repack-on.log" | tail -1
  grep -E "^\[doctest\] test cases:" "$OUT/$T-repack-off.log" | tail -1
  echo "--- $T: the served image error under repack OFF (mm_chat only)"
  grep -n "image: " "$OUT/$T-repack-off.log" | head -3
done

du -sh "$SRC" "$SRC/build-cuda"
echo "### /tmp free at end: $(free_gb) GiB"
echo "### steps"; cat "$OUT/steps.txt"

# READ THE STEPS BACK, AND KNOW WHICH ONES WERE EXPECTED. Recording a status
# nothing reads is the same defect as not recording one, and this driver did not
# read its own steps.txt at all: every leg could fail and the job still ended on
# `W7_CUDA_DONE` with rc 0.
#
# COUNTING NON-ZERO LINES IS NOT ENOUGH, which is the second half. `awk
# '!/ RC=0$/' | wc -l` counts an ABSENT step as zero failures, so a steps.txt
# holding only `configure RC=0` and `build RC=0` -- every comparison never
# having run -- passed, and so did an empty file. The expected list below is
# what makes a step that never ran distinguishable from one that passed.
EXPECTED="toolkit_install configure reconfigure build ctest_cuda cuda_kernels
          probe_cpu cpu_control dev_attn_on"
BAD=0
if [ ! -s "$OUT/steps.txt" ]; then
  echo "### FATAL: steps.txt is empty or absent -- NOTHING was recorded"; BAD=1
else
  if grep -qvE '^[A-Za-z0-9_]+ RC=[0-9]+$' "$OUT/steps.txt"; then
    echo "### MALFORMED STEP LINES (a line no readback can parse):"
    grep -vE '^[A-Za-z0-9_]+ RC=[0-9]+$' "$OUT/steps.txt"; BAD=1
  fi
  if awk '!/ RC=0$/' "$OUT/steps.txt" | grep -q .; then
    echo "### FAILING STEPS:"; awk '!/ RC=0$/' "$OUT/steps.txt"; BAD=1
  fi
  for s in $EXPECTED; do
    grep -qE "^$s RC=" "$OUT/steps.txt" \
      || { echo "### MISSING EXPECTED STEP: $s -- it never ran"; BAD=1; }
  done
  for LP in 0 1 2 3; do
    grep -qE "^(compare_cuda_lp$LP|probe_cuda_lp${LP}_refused) RC=" "$OUT/steps.txt" \
      || { echo "### MISSING lead_pad $LP: neither a comparison nor a recorded refusal"; BAD=1; }
  done
fi
[ "$BAD" -eq 0 ] || { echo "### W7_CUDA_FAILED"; exit 1; }
echo "### W7_CUDA_DONE failed_steps=0"
