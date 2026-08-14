#!/usr/bin/env bash
# PERF-GDN-PACKED-BRIDGE (#365) — dgx selection probe + decode-only A/B.
#
# Prerequisites VERIFIED read-only on dgx 2026-08-12 (no lock taken): the
# nvidia@0893e160 snapshot, $HOME/cutlass-4.5.0 and /usr/local/cuda-13.0/bin/nvcc
# all present.
#
# NOT RUN BY ITS AUTHOR. The implementing worktree has no nvcc and no GPU, and
# on dgx the GPU lock was held by row/GATE-27B-FP8-TOWER-GOLDEN, which outranks
# this row. Everything below is therefore UNEXECUTED and must be treated as a
# proposal until it prints its own DONE marker.
#
# WHAT IT DOES NOT DO: it makes no correctness claim. SACRED
# test_qwen27_paged_engine pins unsloth@890bdef7, a BF16-tower checkpoint that
# cannot execute the fp8 path this row changes. The token gate is owed by
# row/GATE-27B-FP8-TOWER-GOLDEN. This script measures SELECTION and SPEED only.
set -euo pipefail

REPO=${REPO:-$HOME/work/vllm.cpp-gdn-packed-bridge}
BUILD=${BUILD:-$REPO/build-triton}
CKPT=${CKPT:-$HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404}
OUT=${OUT:-$HOME/work/gdn-packed-bridge-ab}
STEPS=${STEPS:-128}
mkdir -p "$OUT"

# ---------------------------------------------------------------- preconditions
# The box OOM-rebooted on 2026-08-12 because a large oracle ran alongside a
# build: on GB10 `gpu_memory_utilization` reserves HOST RAM. Refuse to start if
# someone else is resident.
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 100 ]; then
  echo "REFUSE: only ${avail} GiB available (<100). Someone else is active." >&2
  exit 3
fi
# THE box mutex, resolved once. `${GPU_LOCK:-$HOME/gpu.lock}` is the repo's one
# truth (#777): a hardcoded path here would take a different file from whoever
# followed the docs, and `flock` succeeds on it, so neither side would ever know.
GPU_LOCK="${GPU_LOCK:-$HOME/gpu.lock}"
if ! /usr/bin/flock -n "$GPU_LOCK" true; then
  echo "REFUSE: $GPU_LOCK is held. Holder:" >&2
  fuser -v "$GPU_LOCK" >&2 || true
  exit 3
fi

run_locked() {  # everything GPU-touching runs inside ONE lock for the whole run
  systemd-run --user --pipe --wait --collect \
    --unit="gdn-bridge-$$" \
    /usr/bin/flock "$GPU_LOCK" -c "$1"
}

# ------------------------------------------------------------------------ build
# -DVLLM_CPP_TRITON=ON is MANDATORY, not an optimisation: RecordGdnPackedDecode-
# TritonLaunch and its only call site are both inside `#ifdef VLLM_CPP_TRITON`
# (src/vt/cuda/cuda_gdn.cu), so without it `triton_launches` reads 0 and that is
# INDISTINGUISHABLE from "the cubin was rejected". A previous NVFP4 sweep was
# invalidated exactly this way.
cmake -S "$REPO" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_CPP_TRITON=ON \
      -DVLLM_CPP_CUTLASS_DIR="${CUTLASS_DIR:-$HOME/cutlass-4.5.0}" \
      > "$OUT/configure.log" 2>&1

# Verify the fast path actually configured; a degraded build silently falls back
# to slow WMMA and drifts every number below.
grep -qiE 'cutlass.*sm120a|CUTLASS-sm120a' "$OUT/configure.log" \
  || { echo "REFUSE: CUTLASS sm120a absent from configure log" >&2; exit 4; }
grep -qiE 'flash.?attn|FA2' "$OUT/configure.log" \
  || { echo "REFUSE: FA2 absent from configure log" >&2; exit 4; }
grep -qiE 'triton' "$OUT/configure.log" \
  || { echo "REFUSE: Triton AOT absent from configure log" >&2; exit 4; }

cmake --build "$BUILD" -j "$(nproc)" > "$OUT/build.log" 2>&1
echo "BUILD_RC=0" >> "$OUT/build.log"

# ------------------------------------------------------- 1. SELECTION, not speed
# Counters are HOST-dispatch counts. CUDA graph REPLAY performs no host dispatch
# (cuda_gdn_internal.h), so they increment during capture/eager only -- read them
# from the instrumented engine test, which steps eagerly, NOT from a graphed
# throughput run. Discriminator:
#   packed_launches == 48 && triton_launches == 48 -> cubin fired (the goal)
#   packed_launches == 48 && triton_launches == 0  -> hand kernel; cubin REJECTED
#   packed_launches == 0                           -> model never selected packed
for arm in off on; do
  if [ "$arm" = on ]; then
    envs="VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1"
  else
    envs=""
  fi
  run_locked "cd $BUILD && env $envs VT_QWEN27_SNAPSHOT=$CKPT \
      VT_GDN_DIAG_STEP_LOG=1 ./tests/test_qwen27_paged_engine" \
      > "$OUT/selection-$arm.log" 2>&1 || true
  echo "--- arm=$arm ---"
  grep -E 'VT_GDN_DIAG|packed|triton' "$OUT/selection-$arm.log" | head -20
done

# --------------------------------------------------------------- 2. decode-only
# Expected payoff, so an implausible result is caught: ours GdnDecodeFusedKernel
# 28.08 us/call x48 vs vLLM's packed 19.21 us -> about +0.425 ms/step of a
# +1.81 ms/step decode deficit, plus GdnPostConvFastKernel (+0.131 ms/step),
# which the packed kernel absorbs. Anything much larger is a measurement defect.
for arm in off on; do
  if [ "$arm" = on ]; then
    envs="VT_GDN_PACKED_DECODE_FP8_TOWER=1 VT_GDN_FP8_IN_BF16=1"
  else
    envs=""
  fi
  run_locked "cd $BUILD && env $envs nsys profile -t cuda --cuda-graph-trace=node \
      -o $OUT/decode-$arm --stats=false \
      ./tests/test_qwen27_paged_engine" > "$OUT/nsys-$arm.log" 2>&1 || true
  nsys stats --force-export=true --report cuda_gpu_kern_sum \
      "$OUT/decode-$arm.nsys-rep" > "$OUT/kern-$arm.txt" 2>&1 || true
  echo "--- kernels arm=$arm ---"
  grep -iE 'GdnDecodeFused|GdnPostConvFast|gdn_decode|packed' "$OUT/kern-$arm.txt" | head
done

echo "DONE gdn-packed-bridge-ab $(date -Is)" | tee "$OUT/DONE"
