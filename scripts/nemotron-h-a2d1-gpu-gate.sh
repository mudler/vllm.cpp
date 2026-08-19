#!/usr/bin/env bash
# A2-D1 (#1311, .agents/specs/nemotron-h-a2d1-mamba-decode-step.md) — the GPU
# gate for the SINGLE-STEP recurrent decode arm.
#
# It is a SIBLING of scripts/nemotron-h-a2q1-dgx-gate.sh, not an edit of it:
# that script owns A2-Q1's device-vs-host A/B (VT_NEMOTRON_H_DEVICE_MAMBA) and
# is a live surface on another branch. This one owns A2-D1's A/B
# (VT_NEMOTRON_H_MAMBA_DECODE_STEP), with the device mamba arm ON in BOTH legs,
# so the only thing that differs between them is which recurrent kernels the
# DECODE rows take. One file per row, read with a glob (AGENTS.md §Records).
#
# RUN IT INSIDE A LEASE, NEVER OVER ssh:
#   rc run -d dgx:gpu0 --max-runtime 6h -- bash -lc \
#     'git clone --depth 50 -b row/A2-D1-mamba-decode-step \
#        https://github.com/mudler/vllm.cpp /root/src \
#      && bash /root/src/scripts/nemotron-h-a2d1-gpu-gate.sh'
#
# The environment facts (sbsa lane, anchored cuda package names, the toolkit
# proved by a LINK and not by --version, /workspace refusing symlinks, -j 4)
# are the a2q1 script's and are re-encoded here rather than re-discovered.
set -u -o pipefail

LOG_ROOT=${LOG_ROOT:-/workspace/a2d1}
CKPT=${CKPT:-/workspace/a3/ckpt-stage}
SRC=${SRC:-/root/src}
ARCH=${ARCH:-121a}          # 121a = GB10, 110 = Thor
BUILD=${BUILD:-/root/build-cuda-a2d1}
CUTLASS=${CUTLASS:-/root/cutlass}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$LOG_ROOT/$STAMP
mkdir -p "$RUN"
echo "A2D1 run dir: $RUN"

step() { echo; echo "=== $* ==="; }
# Every gate command runs BARE and echoes its own status. Never pipe a command
# whose exit code matters -- a pipeline reports the LAST stage.
rcx() { "$@"; local r=$?; echo "RC[$*]=$r"; return $r; }

step "0. the box"
rcx uname -m
rcx nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
rcx df -h /root /workspace
rcx free -m

step "1. contention -- a timing number measured beside another job is VOID"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee "$RUN/contention.txt"
echo "RC[nvidia-smi compute-apps]=${PIPESTATUS[0]}"

step "2. toolchain"
export DEBIAN_FRONTEND=noninteractive
rcx apt-get update -qq
rcx apt-get install -y -qq git cmake ninja-build g++ curl ca-certificates python3 python3-dev
if ! command -v nvcc >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
  echo "RC[curl keyring]=$?"
  rcx dpkg -i /tmp/cuda-keyring.deb
  rcx apt-get update -qq
  rcx apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0 \
     cuda-nvrtc-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 libcurand-dev-13-0
fi
export PATH=/usr/local/cuda/bin:$PATH
rcx nvcc --version

step "2b. the toolkit is proved by a LINK, not by --version"
cat > /tmp/probe_a2d1.cu <<'EOF'
#include <cublasLt.h>
#include <cstdio>
int main() {
  cublasLtHandle_t h = nullptr;
  const auto s = cublasLtCreate(&h);
  std::printf("cublasLtCreate=%d\n", static_cast<int>(s));
  return s == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
EOF
rcx nvcc -arch=sm_$ARCH /tmp/probe_a2d1.cu -o /tmp/probe_a2d1 -lcublasLt
if [ $? -ne 0 ]; then
  echo "VOID: the CUDA toolkit does not link cublasLt; every number below would be a lie"
  exit 2
fi
rcx /tmp/probe_a2d1

step "3. cutlass"
if [ ! -f "$CUTLASS/include/cutlass/cutlass.h" ]; then
  rcx git clone --depth 1 --branch v4.5.0 https://github.com/NVIDIA/cutlass "$CUTLASS"
fi
rcx test -f "$CUTLASS/include/cutlass/cutlass.h"

step "4. configure"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=ON \
  -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
  -DVLLM_CPP_CUTLASS_DIR="$CUTLASS" > "$RUN/configure.log" 2>&1
echo "RC[cmake configure]=$?"

step "4b. the feature table -- a DISABLED or wrong-arch cell VOIDS the run"
grep -E "CUDA feature .*(ENABLED|DISABLED)" "$RUN/configure.log" | tee "$RUN/features.txt"
echo "feature cells ENABLED for [$ARCH]: $(grep -cE "ENABLED for \[$ARCH\]" "$RUN/features.txt") ; DISABLED cells: $(grep -cE 'DISABLED' "$RUN/features.txt")"

step "5. build (-j 4: unconstrained parallelism has OOM-REBOOTED this box)"
# OPS_ONLY=1 builds and runs ONLY the op-level equivalence suite and stops.
# It exists for a small board that can host the CUDA kernels but not the 20.1
# GiB checkpoint: the driver-group equivalence is the one piece of evidence that
# needs real CUDA and does NOT need the model, so it should not queue behind a
# box that can run the whole gate.
OPS_ONLY=${OPS_ONLY:-0}
if [ "$OPS_ONLY" = "1" ]; then
  cmake --build "$BUILD" -j 4 --target test_ops_mamba2_state_update > "$RUN/build.log" 2>&1
else
  cmake --build "$BUILD" -j 4 > "$RUN/build.log" 2>&1
fi
echo "RC[cmake build]=$?"
tail -5 "$RUN/build.log"

step "6. the OP-LEVEL equivalence this swap rests on, at n_groups=8"
# ★ THE PREMISE, GATED BEFORE THE SWAP IS TRUSTED. The only pre-existing
# decode-vs-prefill case ran heads_per_group = 2. NemotronH runs 8.
"$BUILD/tests/test_ops_mamba2_state_update" > "$RUN/state_update.log" 2>&1
echo "RC[test_ops_mamba2_state_update]=$?"
grep -E "test cases:|assertions:|Status:" "$RUN/state_update.log"
grep -E "driver-group equivalence" "$RUN/state_update.log"

if [ "$OPS_ONLY" = "1" ]; then
  echo
  echo "OPS_ONLY=1: the op-level equivalence ran and NOTHING ELSE did. The A3 token"
  echo "gate and the decode-window numbers are NOT MEASURED on this host. That is a"
  echo "stated absence, not a pass."
  echo "ALL LOGS: $RUN"
  exit 0
fi

step "6b. the neighbouring suites this arm can break"
for t in test_nemotron_h_mamba_device test_nemotron_h_paged_forward test_nemotron_h_forward \
         test_nemotron_h_loader test_nemotron_h_moe_device test_ops_mamba2_ssd; do
  if [ -x "$BUILD/tests/$t" ]; then
    "$BUILD/tests/$t" > "$RUN/$t.log" 2>&1
    echo "RC[$t]=$?"
    grep -E "test cases:|assertions:|Status:" "$RUN/$t.log"
  else
    echo "MISSING BINARY: $t"
  fi
done

if [ ! -d "$CKPT" ]; then
  echo
  echo "NO CHECKPOINT AT $CKPT -- the A3 gate and the decode-window numbers are NOT MEASURED on this host."
  echo "That is a stated absence, not a pass."
  echo "ALL LOGS: $RUN"
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# The A/B. ONE binary, device mamba ON in both legs, only the decode arm differs.
# ─────────────────────────────────────────────────────────────────────────────
# $1 = label, $2 = VT_NEMOTRON_H_MAMBA_DECODE_STEP, $3 = VT_NEMOTRON_H_DEVICE_MAMBA
# (defaults to 1; leg 3 sets it to 0 -- see the CONFOUND note above step 7).
run_leg() {
  local label=$1 flag=$2 devmamba=${3:-1}
  local log="$RUN/a3_$label.log"
  : > "$log"

  # ★ THE SAMPLING WINDOW IS THE DECODE, NOT THE WHOLE PROCESS. A sampler
  # started with the process contains the multi-minute 20.1 GiB engine load,
  # which is GPU-IDLE, and dilutes both arms toward each other. That already
  # produced one void number on this row. The driver starts FIRST and the
  # sampler starts only once it has printed `engine loaded in Ns`.
  # ★ VT_NEMOTRON_H_ARM_TRACE, *NOT* VT_NEMOTRON_H_DIAG. The latter downloads
  # the carry and the residual per layer per step, so a timed leg under it
  # measures the diagnostic. The arm trace is one fprintf of six resident
  # counters per step, so the reachability evidence and the timing come from
  # the SAME run rather than from two runs that might differ.
  VT_NEMOTRON_H_DEVICE_MAMBA=$devmamba VT_NEMOTRON_H_MAMBA_DECODE_STEP=$flag \
  VT_NEMOTRON_H_ARM_TRACE=1 "$BUILD/examples/nemotron-h-gen" \
      --model "$CKPT" \
      --golden "$SRC/tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json" \
      > "$log" 2>&1 &
  local pid=$!
  local waited=0 loaded=0
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "engine loaded in" "$log" 2>/dev/null; then loaded=1; break; fi
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge 5400 ]; then break; fi
  done
  if [ "$loaded" -ne 1 ]; then
    echo "$label: the engine never reported a load in ${waited}s -- the decode window is UNKNOWN, so NO busy fraction is sampled"
  fi
  local t0=$(date +%s.%N)
  ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 0.1; done ) \
    > "$RUN/util_$label.txt" 2>/dev/null &
  local sampler=$!
  wait "$pid"; local r=$?
  local t1=$(date +%s.%N)
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null

  echo "RC[a3 $label]=$r"
  grep -E "STRICT|PASS|FAIL|DIVERGENCE|REFUSING|SHORT|mode=|engine loaded in" "$log" | tail -20

  # ★ ON A DIVERGENCE, PRINT THE TOKENS. The driver already emits `got:` and
  # `exp:` for any row that mismatches (nemotron_h_gen/main.cpp:385), and the
  # verdict grep above throws them away -- so a `95/96 DIVERGENCE` arrived with
  # no way to tell a wrong recurrent carry from a benign bf16 near-tie without
  # spending a second lease. A gate that reports a failure it cannot triage has
  # not finished reporting. Also print the per-prompt row lines, which carry
  # which prompt was short and by how much.
  if grep -q "DIVERGENCE" "$log"; then
    echo "--- DIVERGENCE detail ($label): per-prompt rows, then got/exp ---"
    grep -E "^\s*prompt |matched=|wall=" "$log" | tail -10
    grep -E "^\s+(got|exp):" "$log" | tail -20
  fi

  # ★ WHICH KERNELS THE DECODE STEPS ACTUALLY LAUNCHED. The two arms compute the
  # same recurrence, so the tokens agree and cannot separate them; this is the
  # only line that can. A counter that reads 0 on BOTH legs means the recorder
  # is broken, not that no kernel ran -- test_nemotron_h_paged_forward's
  # "recurrent arm recorder" case is the triage for that.
  echo "--- decode-step arm counters ($label) ---"
  # Match on the FIELD TEXT, not on a field INDEX. The line is
  # `[NH-DIAG] ARM step T=1 nd=1 np=0  state_update_rows=...`, so nd is field 5
  # and an `$4` test silently selects NOTHING -- which prints as a clean empty
  # result and reads exactly like "no decode steps ran".
  grep -E 'ARM step .* nd=[1-9]' "$log" | head -3
  grep -E 'ARM step .* nd=[1-9]' "$log" | tail -1
  echo "ARM lines total: $(grep -c 'ARM step' "$log")"
  echo "ARM lines with a DECODE row: $(grep -cE 'ARM step .* nd=[1-9]' "$log")"
  echo "ARM lines with a PREFILL row: $(grep -cE 'ARM step .* np=[1-9]' "$log")"
  # A count of ZERO decode lines is an instrument failure, not a result.
  if [ "$(grep -cE 'ARM step .* nd=[1-9]' "$log")" -eq 0 ]; then
    echo "$label: NO decode-step ARM lines were recorded -- the counters say NOTHING about this run, and that is not a pass"
  fi

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
    print(f"{label}: GPU busy in {busy} of {len(vals)} DECODE samples = {100.0*busy/len(vals):.2f}% "
          f"(arch {arch}; compare ONLY against the other leg of this same binary)")
PY
  python3 "$SRC/scripts/nemotron-h-a2q1-per-token.py" "$log" "$label" "$t0" "$t1" "$ARCH"
  echo "reference-tier lines in $label: $(grep -c 'reference-tier' "$log")"
}

step "7. A3 gate + decode window, SINGLE-STEP decode arm (the default, = vLLM)"
( while true; do free -m | awk '/^Mem:/{print $3}'; sleep 1; done ) > "$RUN/rss.txt" 2>/dev/null &
MEMPID=$!
# ★ THE CONFOUND THIS THIRD LEG EXISTS TO REMOVE.
#
# Legs 1 and 2 vary the mamba KERNEL (single-step vs chunk scan) and BOTH sit on
# top of A2-Q1's FP8 W8A8 device projections, because both run inside the
# `mamba_on_device` branch. `main` carries NO device mamba arm at all --
# `NemotronHMamba2MixerDevice`, `MambaIsFp8` and even `VT_NEMOTRON_H_DEVICE_MAMBA`
# are absent there -- so a `95/96` seen on BOTH of legs 1 and 2 is equally
# consistent with:
#
#   (a) the divergence being the HOST's (arch-specific), or
#   (b) the divergence being A2-Q1's FP8 PROJECTIONS, which neither leg turns off.
#
# Leg 3 discriminates by routing the whole mamba block back to the host
# reference on the SAME binary and box. Its counters are the check that it
# really took that path: the four kernel counters must read 0 while
# gathers/scatters stay non-zero, because the host branch still gathers and
# scatters but never enters the instrumented device mixer.
#
#   leg 3 `96/96` => the divergence is A2-Q1's FP8 arm, NOT the architecture.
#   leg 3 `95/96` => the host owns it and the arch-specific framing stands.
run_leg on 1

step "8. the same binary with the decode arm OFF -- the chunk scan on decode rows"
run_leg off 0

step "8b. THE DISCRIMINATOR -- device mamba arm OFF, so A2-Q1's FP8 projections are OUT"
run_leg hostmamba 1 0

kill "$MEMPID" 2>/dev/null
echo "peak host MiB used during the run: $(sort -n "$RUN/rss.txt" | tail -1)"

step "8c. the three-way verdict"
for L in on off hostmamba; do
  printf '%-10s %s\n' "$L" "$(grep -E 'TOKEN MATCH' "$RUN/a3_$L.log" 2>/dev/null | tail -1)"
done
echo "leg 3 (hostmamba) MUST show the four kernel counters at 0 with gathers/scatters non-zero;"
echo "anything else means it did NOT take the host path and the discrimination is VOID."
grep -E 'ARM step .* nd=[1-9]' "$RUN/a3_hostmamba.log" 2>/dev/null | tail -1

step "9. contention, after"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "ALL LOGS: $RUN"
