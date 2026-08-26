#!/usr/bin/env bash
# Stage and run the pinned SGLang oracle from PyPI wheels inside an `rc` lease.
#
# Row `SGLANG-ORACLE-LEASE-WHEEL`, spec `.agents/specs/sglang-wheel-in-lease.md`,
# issue #1265. This is the NEW DRIVER that spec owes.
# `scripts/dgx-sglang-low-concurrency.sh` is NOT patched: it hard-codes a
# container image digest and passes it as `--image`, which is the forbidden path,
# and it has no execution half at all.
#
# HOW TO RUN IT. Stage this file plus `sglang_lease_identity.py` and
# `.agents/specs/sglang-wheel-in-lease.json` into `$NAS/sglang-w2/`, which the
# leased worker sees as `/workspace/sglang-w2/`, then:
#
#   rc run -d dgx:gpu0 --max-runtime 150m --idle-timeout 20m \
#     -- bash /workspace/sglang-w2/run.sh install
#   rc run -d dgx:gpu0 --max-runtime 200m --idle-timeout 20m \
#     -- bash /workspace/sglang-w2/run2.sh serve
#
# Those are the exact invocations of 2026-08-23. `--max-runtime` is the bound
# that matters and is not optional; the heartbeat below is what stops
# `--idle-timeout` firing inside it, because `--idle-timeout 0` selects the
# DEVICE DEFAULT rather than disabling the kill.
#
# **The second phase is staged under a DIFFERENT FILENAME on purpose.** bash
# reads a script lazily, so overwriting the file a running job is executing
# corrupts that job. Stage `run2.sh` beside `run.sh` rather than replacing it,
# and record both sha256 values.
#
# NEVER inline the work into `rc run --`: a detaching client kills the job and
# the ten-minute tool cap makes detaching the normal case. NEVER `ssh` to a
# fleet device; the lease is the only path.
#
# WHAT THE PHASES ARE FOR. `install` ends at the identity gate and touches no
# model, so a failed identity costs no model time. `serve` re-asserts identity
# before it loads a weight, because the worker container is reused and the
# virtual environment in /tmp can be gone.
#
# THE FIVE EXIT CONDITIONS this drives (spec `## The exit criterion`):
#   1. both wheels installed and their sha256 REPORTED FROM INSIDE THE JOB
#   2. IDENTITY_RC=0 against the committed manifest, asserted from `cd /`
#   3. the server reaches readiness and the log NAMES the resolved attention
#      backend and the resolved MoE runner
#   4. one leg completes with zero errors and a recorded output-token count
#   5. the `evidence` key moves off `#1265` -- a records edit, not a job step
set -uo pipefail

PHASE="${1:-}"
case "$PHASE" in
  install|serve) ;;
  *) echo "usage: $0 <install|serve>"; exit 2 ;;
esac

W=/workspace/sglang-w2
OUT="$W/out/$PHASE-$(date -u +%Y%m%dT%H%M%SZ)"
VENV=/tmp/sgenv
WHL=/tmp/sgwhl
CKPT_SRC=/workspace/ckpt/qwen3.8-27b-hf
CKPT_LOCAL=/tmp/ckpt38
PORT=8811
# 0.80 x MemTotal is reserved by design. The floor below is the WATCHDOG's, not
# an engine knob. This box REBOOTS rather than OOM-killing, so a sampler that
# only records the descent is a post-mortem.
FLOOR_MB="${FLOOR_MB:-5000}"
# Space the venv + wheels need in /tmp before the checkpoint copy is even
# considered. Derived from nothing measured on this box; it is a REFUSAL floor,
# not a prediction, and the end-of-run `du` prints the real figure.
NEED_GB="${NEED_GB:-45}"
# Extra space wanted on top of NEED_GB before copying the 52 GiB checkpoint into
# /tmp. Below it the server reads the checkpoint straight off CIFS, which is a
# STATED DIFFERENCE and is recorded as one.
CKPT_GB="${CKPT_GB:-60}"

# --- the two artifacts this row's identity rests on, from the committed spec.
SGLANG_WHL_NAME=sglang-0.5.15-cp312-cp312-manylinux_2_34_aarch64.whl
SGLANG_WHL_SHA=1c2d2602b4ba04c6a71d2f3bf2e3654da53987536f0d65dbe4f57cdc65c9812e
KERNEL_WHL_NAME=sglang_kernel-0.4.4-cp310-abi3-manylinux2014_aarch64.whl
KERNEL_WHL_SHA=727e4bc53abeade20260186f99199200320b9fa51f8de7af90c01524cff73e5d

step() { echo; echo "########## $* ##########"; date -u; }
free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }

mkdir -p "$OUT" || { echo "FATAL: cannot write $OUT"; exit 96; }
exec > >(tee -a "$OUT/job.log") 2>&1

# A quiet phase is the normal case here: a gigabyte-scale download and a 52 GiB
# copy both print nothing for minutes, and `--idle-timeout` counts the job's own
# stdout. `--idle-timeout 0` selects the DEVICE DEFAULT rather than disabling
# the kill, so the heartbeat is the remedy, not the flag.
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) tmp=$(free_gb)G"; done ) &
HB=$!
cleanup_hb() { kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null; }
trap cleanup_hb EXIT INT TERM

step "0. WHERE AM I"
hostname; date -u; uname -m; id -u
cat /proc/sys/kernel/random/boot_id
free -g | head -2 | tail -1
df -h /tmp /workspace
test -d "$W" || { echo "FATAL: $W not visible -- the stage did not land"; exit 96; }
test -f "$W/sglang-wheel-in-lease.json" || { echo "FATAL: no manifest staged"; exit 96; }
test -f "$W/identity.py" || { echo "FATAL: no identity gate staged"; exit 96; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "NVSMI_RC=$?"
python3 -VV

step "0b. IS THE BOX IDLE"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
COTENANTS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+')
echo "COTENANTS_AT_START=$COTENANTS"
if [ "$PHASE" = "serve" ] && [ "$COTENANTS" != "0" ]; then
  echo "ABORT: the box is NOT idle at series start. Reap, then rerun."
  echo "IDLE_BOX=NO"
  exit 38
fi

step "1. DISK, before anything else -- the container is REUSED and the free space is shared"
du -sh /tmp/* 2>/dev/null | sort -rh | head -10
echo "TMP_FREE_GB=$(free_gb)"
if [ "$(free_gb)" -lt "$NEED_GB" ]; then
  echo "REFUSING: /tmp has $(free_gb) GiB free, below the NEED_GB=${NEED_GB} floor."
  echo "That floor is a REFUSAL floor, not a measurement. Raise it deliberately."
  exit 95
fi

step "2. ENVIRONMENT REPAIR -- UNCONDITIONAL, then assert the postcondition"
# The worker container is reused between jobs. A `command -v` guard skips the
# repair on the second job, reports success, and then fails for the reason the
# guard was meant to remove.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
echo "APT_UPDATE_RC=$?"
apt-get install -y -qq python3-dev python3-venv wget ca-certificates gnupg curl
echo "APT_BASE_RC=$?"
# `Python.h` is absent from the worker image and a package that needs it fails
# late. Assert the header, not the package.
PYINC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
test -f "$PYINC/Python.h" || { echo "FATAL: no Python.h under $PYINC"; exit 90; }
echo "PYTHON_H_OK=$PYINC/Python.h"

# flashinfer JITs at run time, so the toolkit is a requirement and not a
# convenience. `cuda-toolkit-13-0` is NOT in Ubuntu's own archive; add NVIDIA's
# sbsa (aarch64) repo first. A bare `cuda-nvcc-13-0` satisfies `command -v nvcc`
# and then fails for want of headers and libraries, which is why the assertions
# below check the POSTCONDITION rather than the binary.
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
echo "KEYRING_DL_RC=$?"
dpkg -i /tmp/ck.deb
echo "KEYRING_RC=$?"
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0
echo "CUDA_APT_RC=$?"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; exit 90; }
test -f "$CUDA_HOME/include/cuda_runtime.h" || { echo "FATAL: nvcc but no cuda_runtime.h under $CUDA_HOME"; exit 90; }
ls "$CUDA_HOME"/targets/*/lib/libcudart.so* >/dev/null 2>&1 \
  || ls "$CUDA_HOME"/lib64/libcudart.so* >/dev/null 2>&1 \
  || { echo "FATAL: no libcudart under $CUDA_HOME"; exit 90; }
nvcc --version | tail -2
echo "NVCC_POSTCONDITION_OK=1"

if [ "$PHASE" = "install" ] || [ ! -x "$VENV/bin/python" ]; then
  step "3. THE VIRTUAL ENVIRONMENT -- in container-local /tmp, NEVER on /workspace"
  # /workspace is CIFS with nounix: it stores no symlink and presents
  # file_mode=0664, so a venv placed there loses its links and cannot run its
  # own entry points.
  if [ ! -x "$VENV/bin/python" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    echo "VENV_CREATE_RC=$?"
  else
    echo "VENV_REUSED=$VENV"
  fi
  test -x "$VENV/bin/python" || { echo "FATAL: no venv interpreter"; exit 91; }
  "$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel
  echo "PIP_BOOTSTRAP_RC=$?"

  step "4. THE TWO WHEELS -- downloaded, HASHED IN THE JOB, then installed"
  mkdir -p "$WHL"
  "$VENV/bin/pip" download --no-deps --only-binary=:all: -d "$WHL" \
      "sglang==0.5.15" "sglang-kernel==0.4.4"
  echo "WHEEL_DOWNLOAD_RC=$?"
  ls -la "$WHL"
  # Verify SEMANTICALLY, in the job, against the values the committed spec
  # carries. A remote hash proves nothing about the bytes that landed here.
  for pair in "$SGLANG_WHL_NAME:$SGLANG_WHL_SHA" "$KERNEL_WHL_NAME:$KERNEL_WHL_SHA"; do
    n="${pair%%:*}"; want="${pair##*:}"
    test -f "$WHL/$n" || { echo "FATAL: pip resolved no $n"; ls "$WHL"; exit 93; }
    got=$(sha256sum "$WHL/$n")
    got=${got%% *}
    echo "WHEEL $n"
    echo "  size=$(stat -c%s "$WHL/$n")"
    echo "  sha256=$got"
    echo "  expected=$want"
    [ "$got" = "$want" ] || { echo "FATAL: sha256 mismatch for $n"; exit 93; }
    echo "  WHEEL_SHA_OK=1"
  done

  step "5. INSTALL -- upstream's own order and index (docker/Dockerfile:210,242)"
  "$VENV/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu130 "$WHL/$SGLANG_WHL_NAME"
  echo "SGLANG_INSTALL_RC=$?"
  # `--force-reinstall --no-deps` is upstream's own invocation at :210 and it is
  # what pins the kernel wheel to the exact artifact hashed above.
  "$VENV/bin/pip" install --force-reinstall --no-deps "$WHL/$KERNEL_WHL_NAME"
  echo "KERNEL_INSTALL_RC=$?"

  step "5b. FLASHINFER JIT CACHE -- close the difference, do not accept it"
  # Without a warm cache the first request compiles kernels in-process and the
  # first leg measures the JIT compiler rather than the model. The image's own
  # cache stage is `ARG INSTALL_FLASHINFER_JIT_CACHE=0` by default and nobody
  # has read sha256:d0a667e, so this is closable rather than acceptable.
  "$VENV/bin/pip" install --no-deps \
      --extra-index-url https://flashinfer.ai/whl/cu130 \
      "flashinfer-jit-cache==0.6.12"
  JITC_RC=$?
  echo "JITCACHE_RC=$JITC_RC"
  if [ "$JITC_RC" != 0 ]; then
    echo "JITCACHE_STATE=ABSENT -- the first leg would measure the JIT compiler."
    echo "  A WARMUP leg is run and DISCARDED below, which is the fallback."
  else
    echo "JITCACHE_STATE=INSTALLED"
  fi

  step "6. FREEZE -- the whole resolved stack, so this run is reproducible"
  "$VENV/bin/pip" freeze > "$OUT/pip-freeze.txt"
  echo "FREEZE_RC=$?  lines=$(wc -l < "$OUT/pip-freeze.txt")"
  grep -iE '^(sglang|sglang-kernel|torch|transformers|flashinfer|triton|nvidia-cutlass-dsl|sgl-deep-gemm)' "$OUT/pip-freeze.txt"
fi

step "7. THE IDENTITY GATE -- FIRST, from `cd /`, before anything touches the GPU"
cd /
"$VENV/bin/python" "$W/identity.py" --manifest "$W/sglang-wheel-in-lease.json"
IDENTITY_RC=$?
echo "IDENTITY_RC=$IDENTITY_RC"
[ "$IDENTITY_RC" = 0 ] || { echo "ABORT: installed tree is not the pin"; exit 94; }

step "8. THE ENVIRONMENT THE NUMBER WILL CARRY"
cd /
"$VENV/bin/python" - <<'PY'
import torch, sglang
print("sglang.__version__ =", sglang.__version__)
print("sglang.__file__    =", sglang.__file__)
print("torch.__version__  =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda_available     =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name        =", torch.cuda.get_device_name(0))
    print("capability         =", torch.cuda.get_device_capability(0))
import sgl_kernel
print("sgl_kernel.__file__ =", sgl_kernel.__file__)
# WHICH shared object actually loaded, read from the process map rather than
# re-derived from `load_utils`. `_load_architecture_specific_ops` takes `sm90/`
# only at compute capability exactly 90, so GB10 is expected to take `sm100/` --
# but a prediction from source is not an observation, and this is the
# observation.
# `sgl_kernel/__init__.py` binds the loaded extension to `common_ops` at import
# time, so its `__file__` IS the object that loaded.
co = getattr(sgl_kernel, "common_ops", None)
print("sgl_kernel.common_ops.__file__ =", getattr(co, "__file__", "UNBOUND"))
with open("/proc/self/maps") as fh:
    mapped = sorted({
        line.split()[-1]
        for line in fh
        if "sgl_kernel" in line and line.rstrip().endswith(".so")
    })
for m in mapped:
    print("sgl_kernel MAPPED:", m)
print("sgl_kernel arch dir  =", "sm100/" if any("/sm100/" in m for m in mapped)
      else ("sm90/" if any("/sm90/" in m for m in mapped) else "UNRESOLVED"))
PY
echo "ENV_RC=$?"
cd /
"$VENV/bin/python" - <<'PY'
from sglang.srt.utils.common import is_sm100_supported, is_sm120_supported
print("is_sm100_supported =", is_sm100_supported())
print("is_sm120_supported =", is_sm120_supported())
try:
    from sglang.srt.utils.common import is_flashinfer_available
    print("is_flashinfer_available =", is_flashinfer_available())
except Exception as exc:  # the import path can move; do not fail the gate on it
    print("is_flashinfer_available = UNREAD:", exc)
PY
echo "BACKEND_PREDICATE_RC=$?"

nvidia-smi -lgc 2190 > /tmp/lgc.log 2>&1
echo "LGC_RC=$?"
tail -2 /tmp/lgc.log
echo "  (LGC_RC=4 is the recorded lease refusal, #1354. Clocks are SAMPLED, never pinned.)"

if [ "$PHASE" = "install" ]; then
  step "9. INSTALL PHASE COMPLETE -- no model was touched"
  du -sh "$VENV" "$WHL" 2>/dev/null
  echo "TMP_FREE_GB_AT_END=$(free_gb)"
  echo "EVIDENCE_DIR=$OUT"
  echo "DONE_MARKER_SGLANG_W2_INSTALL"
  exit 0
fi

########################  serve  ########################

step "10. THE CHECKPOINT"
test -f "$CKPT_SRC/config.json" || { echo "FATAL: no checkpoint at $CKPT_SRC"; exit 50; }
CKPT="$CKPT_SRC"
CKPT_PLACE=cifs
if [ "$(free_gb)" -ge "$CKPT_GB" ]; then
  if [ ! -f "$CKPT_LOCAL/config.json" ]; then
    t0=$SECONDS
    # `-L` is the part that matters: CIFS stores no symlink, so a plain copy of
    # a link target is what actually lands.
    cp -rL "$CKPT_SRC" "$CKPT_LOCAL"
    echo "CKPT_COPY_RC=$? secs=$((SECONDS-t0))"
  else
    echo "CKPT_REUSED=$CKPT_LOCAL"
  fi
  if [ -f "$CKPT_LOCAL/config.json" ]; then
    CKPT="$CKPT_LOCAL"; CKPT_PLACE=tmp
  else
    echo "CKPT_COPY_INCOMPLETE -- falling back to the CIFS path"
  fi
else
  echo "CKPT_STAYS_ON_CIFS: /tmp has $(free_gb) GiB, below CKPT_GB=$CKPT_GB."
fi
echo "CKPT=$CKPT  CKPT_PLACE=$CKPT_PLACE"
du -sb "$CKPT" | sed 's/^/CKPT_BYTES=/'
"$VENV/bin/python" -c "import json,sys; d=json.load(open('$CKPT/config.json')); print('ARCH=',d.get('architectures')); print('MODEL_TYPE=',d.get('model_type'))"

step "11. WATCHDOG -- this box REBOOTS rather than OOM-killing"
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A" >> "$OUT/mem.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      SP=$(cat "$OUT/server.pid" 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing pgid=${SP:-none} to save the box"
      # Kill the process GROUP. SGLang spawns scheduler and detokenizer
      # processes; killing the parent by name or pid STRANDS them holding
      # device memory, which is how 23 GB was held across three jobs.
      [ -n "$SP" ] && kill -9 -- "-$SP" 2>/dev/null
      touch "$OUT/watchdog.fired"
      sleep 30
    fi
    sleep 2
  done ) > "$OUT/watchdog.log" 2>&1 &
WATCHDOG=$!
trap 'kill -9 $WATCHDOG 2>/dev/null; kill "${CLK:-}" 2>/dev/null; cleanup_hb' EXIT INT TERM

step "12. SERVE"
awk '/MemAvailable/{print "MemAvailable_MB_before_server="int($2/1024)}' /proc/meminfo
SRV_LOG="$OUT/sglang-server.log"
t0=$SECONDS
set -m
"$VENV/bin/python" -m sglang.launch_server \
  --model-path "$CKPT" \
  --served-model-name gate \
  --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static 0.80 \
  --max-running-requests 32 \
  --context-length 2048 \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --random-seed 0 \
  --log-level info \
  > "$SRV_LOG" 2>&1 &
SRV=$!
set +m
echo "$SRV" > "$OUT/server.pid"
echo "SERVER_PID=$SRV (process-group leader)"

ready=0
while [ $((SECONDS-t0)) -lt 3000 ]; do
  curl -sf "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1 && { ready=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED during startup"; break; }
  sleep 5
done
echo "READY=$ready  SECONDS_TO_READY=$((SECONDS-t0))"
if [ "$ready" != 1 ]; then
  echo "LEG ABORTED: the server never became ready"
  echo "--- last 120 lines of the server log ---"
  tail -120 "$SRV_LOG"
  kill -9 -- "-$SRV" 2>/dev/null; kill -9 $SRV 2>/dev/null
  sleep 10
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
  echo "EVIDENCE_DIR=$OUT"
  echo "DONE_MARKER_SGLANG_W2_SERVE_FAILED"
  exit 60
fi

step "13. WHAT THE SERVER SAYS IT RESOLVED -- read from ITS log, never assumed"
# Exit condition 3 asks for the resolved attention backend and MoE runner AS
# THE LOG NAMES THEM. `entrypoints/engine.py:225` logs the whole ServerArgs
# dataclass, so both are in one line; the greps below pull them out and the
# whole line is kept in the evidence directory.
grep -oE "attention_backend='[^']*'|attention_backend=None" "$SRV_LOG" | head -3
grep -oE "moe_runner_backend='[^']*'|moe_runner_backend=None" "$SRV_LOG" | head -3
grep -oE "linear_attn_backend='[^']*'|linear_attn_decode_backend='[^']*'|linear_attn_prefill_backend='[^']*'|mamba_backend='[^']*'" "$SRV_LOG" | head -8
grep -oE "prefill_attention_backend='[^']*'|decode_attention_backend='[^']*'" "$SRV_LOG" | head -4
grep -iE "Attention backend|attention backend|cuda graph|Capture cuda graph|KV Cache is allocated|max_total_num_tokens|Load weight end|memory pool" "$SRV_LOG" | head -20
curl -s "http://127.0.0.1:$PORT/get_model_info"; echo
echo "SERVER_LOG=$SRV_LOG"

run_leg () {   # $1 concurrency  $2 label
  local C="$1" LABEL="$2" NP=$(( 6 * $1 ))
  cd /
  # `sglang-oai` posts to /v1/completions, the SAME OpenAI-completions path the
  # 2026-07-28 image run drove and the same one a vLLM arm uses. `sglang`
  # (native /generate) would be a different client path and not comparable to
  # anything already recorded.
  # SGLang's --random-range-ratio is the INVERSE of vLLM's: `compute_random_lens`
  # draws from [full_len*ratio, full_len], so 1 is the fixed-length case and 0 is
  # the widest spread. vLLM's --random-range-ratio 0 means fixed. Copying the
  # vLLM number here would silently randomise every length.
  # ignore_eos defaults ON (`--disable-ignore-eos` turns it off), so the expected
  # output-token count is exactly num_prompts * random-output-len.
  "$VENV/bin/python" -m sglang.benchmark.serving \
    --backend sglang-oai --base-url "http://127.0.0.1:$PORT" \
    --model gate --tokenizer "$CKPT" \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --random-range-ratio 1 \
    --num-prompts "$NP" --max-concurrency "$C" --request-rate inf \
    --temperature 0 --seed 0 \
    --output-details --output-file "$OUT/$LABEL.jsonl" \
    --disable-tqdm > "$OUT/$LABEL.log" 2>&1
  echo "CLIENT_RC=$?"
  tail -40 "$OUT/$LABEL.log"
}

# The SM clock cannot be PINNED inside a lease (#1354, LGC_RC=4 above), so it is
# SAMPLED and the samples travel with the number. This is attribution, not
# control: a leg taken here is an ABSOLUTE and no ratio may be divided out of it.
( while true; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null)"
    sleep 2
  done ) > "$OUT/clocks.samples" 2>&1 &
CLK=$!

step "14. WARMUP LEG -- DISCARDED. Without a warm cache the first leg measures the JIT compiler"
run_leg 1 "warmup-c1"

for C in 1 8; do
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "SERVER IS DEAD -- abandoning the remaining legs rather than recording completed=0"
    [ -f "$OUT/watchdog.fired" ] && echo "CAUSE=watchdog"
    break
  fi
  step "15. TIMED LEG c$C"
  run_leg "$C" "sglang-c$C"
  "$VENV/bin/python" - "$OUT/sglang-c$C.jsonl" $(( 6 * C )) <<'PY'
import json, sys, collections
f, want = sys.argv[1], int(sys.argv[2])
recs = [json.loads(l) for l in open(f) if l.strip()]
d = recs[-1]
comp = d.get("completed")
errs = d.get("errors")
failed = [e for e in errs if e] if isinstance(errs, list) else None
print(f"leg={f}")
print(f"completed={comp} expected={want}")
print(f"failed={len(failed) if failed is not None else 'UNKNOWN'}")
if failed:
    for t, n in collections.Counter(str(e)[:200] for e in failed).most_common(5):
        print(f"  x{n}: {t}")
for k in ("total_input_tokens", "total_output_tokens", "output_throughput",
          "request_throughput", "median_ttft_ms", "median_tpot_ms",
          "median_itl_ms", "median_e2el_ms", "duration"):
    if k in d:
        print(f"{k}={d[k]}")
print("LEG_COMPLETE_OK" if comp == want and failed == [] else "LEG_VOID")
PY
  echo "LEG_SUMMARY_RC=$?"
done

step "15b. CLOCK SAMPLES -- derived from the RAW file, never from a summary"
kill "$CLK" 2>/dev/null; wait "$CLK" 2>/dev/null
"$VENV/bin/python" - "$OUT/clocks.samples" <<'PYCLK'
import sys, pathlib, collections
rows = []
reasons = collections.Counter()
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        continue
    head = parts[0].split()
    if len(head) < 2 or not head[0].isdigit():
        continue
    try:
        sm = int(head[1]); temp = int(parts[1])
    except ValueError:
        continue
    rows.append((sm, temp))
    reasons[parts[2]] += 1
if not rows:
    print("clock_samples=0 -- NO attribution for any number in this run")
    raise SystemExit(0)
sms = sorted(r[0] for r in rows); temps = [r[1] for r in rows]
med = sms[len(sms)//2]
spread = (max(sms) - min(sms)) / med * 100 if med else 0.0
print(f"clock_samples={len(rows)}")
print(f"sm_clock_mhz median={med} min={min(sms)} max={max(sms)} spread={spread:.2f}%")
print("WITHIN_5PCT" if spread <= 5.0 else "BREACHES_5PCT")
print(f"gpu_temp_c max={max(temps)} min={min(temps)}")
# Every distinct throttle string with its count. A DECIMATED summary hides the
# rows it appears to rule out, so this enumerates instead of sampling.
for r, n in reasons.most_common():
    print(f"throttle_reason x{n}: {r}")
PYCLK
echo "CLOCK_SUMMARY_RC=$?"

step "16. TEARDOWN, then ASSERT the resource came back"
kill -TERM -- "-$SRV" 2>/dev/null || kill -TERM "$SRV" 2>/dev/null
for _ in $(seq 1 90); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
kill -9 -- "-$SRV" 2>/dev/null; kill -9 "$SRV" 2>/dev/null
rm -f "$OUT/server.pid"
REMAIN=1
for _ in $(seq 1 45); do
  REMAIN=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+')
  [ "$REMAIN" = "0" ] && break
  sleep 2
done
echo "COMPUTE_APPS_AFTER_TEARDOWN=$REMAIN"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
if [ "$REMAIN" = "0" ]; then
  echo "TEARDOWN_VERDICT=CLEAN -- the GPU was returned"
else
  echo "TEARDOWN_VERDICT=NOT_CLEAN -- REAP IT before the next series"
fi
kill -9 "$WATCHDOG" 2>/dev/null
awk 'NF>=2{if(min==""||$2<min)min=$2} END{print "minMemAvailable_MB="min"  samples="NR}' "$OUT/mem.samples" 2>/dev/null
free -g | head -2 | tail -1
echo "TMP_FREE_GB_AT_END=$(free_gb)"
ls -la "$OUT"
echo "EVIDENCE_DIR=$OUT"
echo "DONE_MARKER_SGLANG_W2_SERVE"
