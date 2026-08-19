#!/usr/bin/env bash
# A2-Q1 (#810, .agents/specs/nemotron-h-a2q1-fp8-mamba.md) — the GB10 gate for
# the FP8 W8A8 Mamba2 device arm, recorded as a script so the recipe is
# reproducible rather than retyped (AGENTS.md §Gates: "Record the exact build and
# run recipe").
#
# RUN IT INSIDE A LEASE, NEVER OVER ssh:
#   rc run -d dgx:gpu0 --max-runtime 6h -- bash -lc \
#     'git clone --depth 50 -b row/A2-Q1 https://github.com/mudler/vllm.cpp /root/src \
#      && bash /root/src/scripts/nemotron-h-a2q1-dgx-gate.sh'
#
# THREE ENVIRONMENT FACTS THIS SCRIPT ENCODES RATHER THAN REDISCOVERS:
#   * the CUDA lane on this box is `sbsa`, not `arm64`, and an UNANCHORED
#     `cuda-toolkit-13*` match selects `cuda-toolkit-13-config-common`, which
#     installs cleanly, ships no compiler and returns 0;
#   * `nvcc --version` is NOT a sufficient postcondition — a partial CUDA install
#     printed every feature line and then failed to link `CUDA::cublasLt`, so the
#     toolkit is proved by an actual link;
#   * `/workspace` is CIFS and refuses symlinks, so the build lives on local disk
#     and only the log is copied out.
set -u -o pipefail

LOG_ROOT=${LOG_ROOT:-/workspace/a2q1}
CKPT=${CKPT:-/workspace/a3/ckpt-stage}
SRC=${SRC:-/root/src}
# 121a is the GB10; 110 is Thor. The spec gates BOTH hosts, and the FP8 W8A8 arm
# on Thor is the whole point of #960/#991, so the arch is a parameter.
ARCH=${ARCH:-121a}
BUILD=${BUILD:-/root/build-cuda}
CUTLASS=${CUTLASS:-/root/cutlass}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$LOG_ROOT/$STAMP
mkdir -p "$RUN"
echo "A2Q1 run dir: $RUN"

step() { echo; echo "=== $* ==="; }
# Every gate command runs BARE and echoes its own status. Never pipe a command
# whose exit code matters -- a pipeline reports the LAST stage, which is how a
# failing `mount` once reported rc=0.
rc() { "$@"; local r=$?; echo "RC[$*]=$r"; return $r; }

step "0. the box, before anything is installed"
rc uname -m
rc id -u
rc nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
rc df -h /root /workspace
rc free -m

step "1. contention -- a timing number measured beside another job is VOID"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee "$RUN/contention.txt"
echo "RC[nvidia-smi compute-apps]=${PIPESTATUS[0]}"

step "2. toolchain"
export DEBIAN_FRONTEND=noninteractive
rc apt-get update -qq
rc apt-get install -y -qq git cmake ninja-build g++ curl ca-certificates python3 python3-dev
if ! command -v nvcc >/dev/null 2>&1; then
  # ANCHORED package names on the sbsa lane. `cuda-nvcc-13-0` is the compiler;
  # the libraries this build links are named individually so a metapackage that
  # ships nothing cannot satisfy the check.
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
  echo "RC[curl keyring]=$?"
  rc dpkg -i /tmp/cuda-keyring.deb
  rc apt-get update -qq
  rc apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0 \
     cuda-nvrtc-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 libcurand-dev-13-0
fi
export PATH=/usr/local/cuda/bin:$PATH
rc nvcc --version

step "2b. the toolkit is proved by a LINK, not by --version"
cat > /tmp/probe.cu <<'EOF'
#include <cublasLt.h>
#include <cstdio>
int main() {
  cublasLtHandle_t h = nullptr;
  const auto s = cublasLtCreate(&h);
  std::printf("cublasLtCreate=%d\n", static_cast<int>(s));
  return s == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
EOF
rc nvcc -arch=sm_$ARCH /tmp/probe.cu -o /tmp/probe -lcublasLt
TOOLCHAIN_OK=$?
if [ "$TOOLCHAIN_OK" -ne 0 ]; then
  echo "VOID: the CUDA toolkit does not link cublasLt; every number below would be a lie"
  exit 2
fi
rc /tmp/probe

step "3. cutlass (the fp8/fp4 fast-path cells resolve from it)"
if [ ! -f "$CUTLASS/include/cutlass/cutlass.h" ]; then
  rc git clone --depth 1 --branch v4.5.0 https://github.com/NVIDIA/cutlass "$CUTLASS"
fi
rc test -f "$CUTLASS/include/cutlass/cutlass.h"

step "4. configure"
rc cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
  -DVLLM_CPP_CUTLASS_DIR="$CUTLASS" 2>&1 | tee "$RUN/configure.log"
echo "RC[cmake configure]=${PIPESTATUS[0]}"

step "4b. the feature table -- a DISABLED or wrong-arch cell VOIDS the run"
grep -E "CUDA feature .*(ENABLED|DISABLED)" "$RUN/configure.log" | tee "$RUN/features.txt"
BAD=$(grep -cE "CUDA feature .*DISABLED" "$RUN/features.txt")
ENABLED=$(grep -cE "ENABLED for \[$ARCH\]" "$RUN/features.txt")
echo "feature cells ENABLED for [$ARCH]: $ENABLED ; DISABLED cells: $BAD"

step "5. build (-j 4: unconstrained parallelism has OOM-REBOOTED this box)"
rc cmake --build "$BUILD" -j 4 2>&1 | tail -40 | tee "$RUN/build.tail"
echo "RC[cmake build]=${PIPESTATUS[0]}"

step "6. the focused device gate -- A2-Q1's own cases"
"$BUILD/tests/test_nemotron_h_mamba_device" -s 2>&1 | tee "$RUN/mamba_device.log"
echo "RC[test_nemotron_h_mamba_device]=${PIPESTATUS[0]}"
grep -E "test cases:|assertions:|Status:" "$RUN/mamba_device.log"

step "6b. the neighbouring suites the arm can break"
for t in test_nemotron_h_forward test_nemotron_h_paged_forward test_nemotron_h_loader \
         test_nemotron_h_moe_device test_ops_mamba2_ssd test_ops_fp8_cpu; do
  if [ -x "$BUILD/tests/$t" ]; then
    "$BUILD/tests/$t" > "$RUN/$t.log" 2>&1
    echo "RC[$t]=$?"
    grep -E "test cases:|assertions:|Status:" "$RUN/$t.log"
  else
    echo "MISSING BINARY: $t"
  fi
done

step "7. the A3 gate + the GPU busy fraction, device mamba ON"
# The acceptance test of this unit is NOT a ratio: the GPU busy fraction must
# RISE from the 6.31% baseline, so it is sampled on a loop for the whole decode
# and the SAMPLE COUNT is reported beside it. A fraction with no denominator is
# not a measurement.
run_gate() {   # $1 = label, $2 = VT_NEMOTRON_H_DEVICE_MAMBA value
  local label=$1 flag=$2
  local log="$RUN/a3_$label.log"
  : > "$log"

  # ★ THE SAMPLING WINDOW IS THE DECODE, NOT THE WHOLE PROCESS.
  #
  # The first Thor run sampled from process start, so the window contained the
  # multi-minute 20.1 GiB engine load, which is GPU-IDLE. That dilutes both arms
  # toward each other and toward zero: it read 15.33% with the arm on and 14.73%
  # with it off, a 0.59-point difference that says almost nothing about what the
  # DECODE does, and it is not comparable to the 6.31% figure either. Mixing a
  # load phase into a decode measurement is the same defect as summing prefill
  # and decode in one profile.
  #
  # So the driver starts FIRST, and the sampler starts only once the driver has
  # printed `engine loaded in Ns`. Everything after that line is decode.
  VT_NEMOTRON_H_DEVICE_MAMBA=$flag "$BUILD/examples/nemotron-h-gen" \
      --model "$CKPT" \
      --golden "$SRC/tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json" \
      > "$log" 2>&1 &
  local pid=$!
  local waited=0
  local loaded=0
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "engine loaded in" "$log" 2>/dev/null; then loaded=1; break; fi
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 5400 ]; then break; fi
  done
  if [ "$loaded" -ne 1 ]; then
    # Say so rather than sampling a window whose meaning is unknown. A fraction
    # over the wrong window is worse than no fraction, because it still formats
    # like a measurement.
    echo "$label: the engine never reported a load in ${waited}s -- the decode window is UNKNOWN, so NO busy fraction is sampled"
  fi
  local t0=$(date +%s.%N)
  ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 0.1; done ) \
    > "$RUN/util_$label.txt" 2>/dev/null &
  local sampler=$!
  wait "$pid"
  local r=$?
  local t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null

  echo "RC[a3 $label]=$r"
  grep -E "STRICT|PASS|FAIL|DIVERGENCE|REFUSING|SHORT|mode=|engine loaded in" "$log" | tail -20
  # The busy fraction WITH its denominator, over the DECODE window alone. The
  # 6.31% baseline is a GB10 number, so it is quoted ONLY on GB10: printing it
  # beside a Thor figure invites a cross-silicon comparison that answers a
  # different question, and a number quoted often enough becomes treated as
  # measured.
  python3 - "$RUN/util_$label.txt" "$label" "$ARCH" "$loaded" <<'PY'
import sys
vals = [int(x) for x in open(sys.argv[1]).read().split() if x.strip().isdigit()]
label, arch, loaded = sys.argv[2], sys.argv[3], sys.argv[4]
if loaded != "1":
    print(f"{label}: busy fraction NOT REPORTED -- the decode window was never identified")
elif not vals:
    print(f"{label}: NO SAMPLES -- the busy fraction is unmeasured, not 0")
else:
    busy = sum(1 for v in vals if v > 0)
    pct = 100.0 * busy / len(vals)
    note = ("baseline 6.31% on this same GB10 workload" if arch == "121a"
            else f"NO baseline for arch {arch}: the 6.31% figure is GB10's, so it "
                 f"is not comparable here; use the ON/OFF A/B on this box")
    print(f"{label}: GPU busy in {busy} of {len(vals)} DECODE samples = {pct:.2f}% busy ({note})")
PY
  # The per-output-token time over the SAME decode window, with the terms shown.
  python3 "$SRC/scripts/nemotron-h-a2q1-per-token.py" "$log" "$label" "$t0" "$t1" "$ARCH"
  # The reference tier is numerically CORRECT, so a pass obtained on it is
  # invisible in the numbers and only this line separates them (spec R2).
  echo "reference-tier lines in $label: $(grep -c 'reference-tier' "$log")"
}

if [ ! -d "$CKPT" ]; then
  echo "NO CHECKPOINT AT $CKPT -- the A3 gate and the GPU busy fraction are NOT MEASURED on this host."
  echo "That is a stated absence, not a pass. Steps 7-9 are skipped."
  echo "ALL LOGS: $RUN"
  exit 0
fi

( while true; do free -m | awk '/^Mem:/{print $3}'; sleep 1; done ) > "$RUN/rss.txt" 2>/dev/null &
MEMPID=$!
run_gate on 1

step "8. the same binary with the arm OFF -- the A/B this unit is measured by"
run_gate off 0
kill "$MEMPID" 2>/dev/null
echo "peak host MiB used during the run: $(sort -n "$RUN/rss.txt" | tail -1)"

step "9. contention, after"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "ALL LOGS: $RUN"
