#!/usr/bin/env bash
# #2497 -- the llama.cpp DENOMINATOR for Qwen3.8-27B Q4_K_M on gfx1151.
#
# Spec: .agents/specs/bench-rocm-llamacpp-denominator.md
# Row:  BACKEND-GATE-ROCM-LLAMACPP
#
# ONE ENGINE. This job runs the pinned llama.cpp oracle and nothing else.
# vllm.cpp is NOT built, NOT loaded and NOT timed here, because that arm's
# declared token gate reads FAIL at 3 of 6 and AGENTS.md Gates admits no
# performance result from it. No ratio is computed anywhere in this job.
#
# HSA_ENABLE_SDMA is retired (#2511) and is not set anywhere below.
set -uo pipefail

W=/workspace/rocm-denom-2497
LOCAL=/tmp/rocm-strix-q4k             # shared staging with the earlier campaign
JOBLOCAL=/tmp/rocm-denom-2497
CCSETUP_DIR=/workspace/rocm-strix-q4k
CCSETUP=$CCSETUP_DIR/ccache-setup.sh
IMG=vllmcpp-rocm-build:7.2.4
P=(podman --storage-driver=vfs --root /tmp/podman-pr66-root-vfs --runroot /tmp/podman-pr66-run-vfs)

GGUF_NAME=Qwen3.8-27B-Q4_K_M.gguf
GGUF_NAS=/workspace/ckpt/qwen38-27b-q4km/$GGUF_NAME
GGUF=$LOCAL/models/$GGUF_NAME
GGUF_SIZE=17106775008
GGUF_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169

LLAMA_PIN=10bf611e533d81f739128304991c5e133c6aebd8
LLAMA_LABEL=b10451
# LC_ALL=C is part of the DEFINITION: `sort` collates by locale, and a devbox on
# en_US.UTF-8 against a container on C refused a CORRECT tree once already.
LLAMA_MANIFEST=56c26d15c2acf11b8621ac26663b4316dc29719d765ba1d95231ffacaddf3cda
manifest_of() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z \
                 | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }

# THE DESIGN. N is declared here and the fold is told it. It is never counted
# out of the log, which is tee'd and would read every leg twice.
LEGS=6
NGL=99
NGEN=64
REPS=3

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$JOBLOCAL" "$LOCAL/models" "$LOCAL/out" || exit 90
exec > >(tee -a "$JOBLOCAL/job-$TAG.log") 2>&1
( while true; do cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT

fail() { echo "FATAL: $*"; echo "JOB_VERDICT=FAIL"; exit 1; }
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

CENV=(-e "HOME=$JOBLOCAL")
podrun() {   # podrun <timeout|-> <entrypoint> [args...]
  local tmo=$1 ep=$2; shift 2
  local -a pre=()
  [ "$tmo" != "-" ] && pre=(timeout --foreground "$tmo")
  "${pre[@]}" "${P[@]}" run --rm --entrypoint "$ep" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    "${CENV[@]}" -v "$LOCAL:$LOCAL:rw" -v "$LOCAL:/local:rw" \
    -v "$JOBLOCAL:$JOBLOCAL:rw" -v "$W:$W:ro" -v "$CCSETUP_DIR:$CCSETUP_DIR:ro" \
    -v /workspace/ccache:/workspace/ccache:rw \
    "$IMG" "$@"
}

step "0. worker identity and contention state"
echo "hostname=$(hostname)"
uname -a
echo "host_arch=$(uname -m)"
echo "nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "HSA_ENABLE_SDMA is deliberately NOT set on any leg of this job (retired, #2511)"
env | grep -i '^HSA_\|^VT_\|^GGML_\|^ROCR_' | sed 's/^/inherited_env /' || echo "inherited_env NONE"
free -g; df -h /tmp /workspace | head -5
{ uname -a; echo "host_arch=$(uname -m)"; echo "nproc=$(nproc)";
  cat /proc/sys/kernel/random/boot_id; } > "$JOBLOCAL/worker.txt" 2>&1
cp -f "$JOBLOCAL/worker.txt" "$OUT/worker.txt" 2>/dev/null

step "1. the artifact, verified ON the worker"
if [ ! -f "$GGUF" ] || [ "$(stat -c %s "$GGUF")" != "$GGUF_SIZE" ]; then
  echo "staging from the share (CIFS mmap is not a run surface)"
  cp -f "$GGUF_NAS" "$GGUF" || fail "stage failed"
fi
echo "gguf_path=$GGUF"
echo "gguf_bytes=$(stat -c %s "$GGUF")  expected=$GGUF_SIZE"
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "gguf_sha256=$GOT"
echo "gguf_sha256_expected=$GGUF_SHA"
[ "$GOT" = "$GGUF_SHA" ] || fail "artifact sha256 mismatch on the worker"
echo "ARTIFACT_OK"

step "2. the container image"
if "${P[@]}" image exists "$IMG"; then
  echo "image_present=YES"
else
  echo "image_present=NO -- rebuilding from /workspace/rocm-strix-dflash2/build-image.sh"
  bash /workspace/rocm-strix-dflash2/build-image.sh > "$JOBLOCAL/build-image.log" 2>&1
  echo "build_image_rc=$?"; tail -5 "$JOBLOCAL/build-image.log"
  "${P[@]}" image exists "$IMG" || fail "the build image is absent and could not be rebuilt"
fi

step "3. the pinned llama.cpp SOURCE identity, as content and not as framing"
if [ ! -d "$LOCAL/src-llamacpp" ]; then
  echo "llamacpp_source=RESTAGED from the share tarball (the pod dropped /tmp)"
  tar -xzf "$CCSETUP_DIR/llamacpp-b10451.tar.gz" -C "$LOCAL" \
    && mv "$LOCAL/llamacpp-b10451" "$LOCAL/src-llamacpp"
  cp "$CCSETUP_DIR/llamacpp-b10451.VERSION" "$LOCAL/src-llamacpp.VERSION"
else
  echo "llamacpp_source=ALREADY STAGED"
fi
[ -d "$LOCAL/src-llamacpp" ] || fail "no staged llama.cpp source at $LOCAL/src-llamacpp"
echo "llama_pin=$LLAMA_PIN label=$LLAMA_LABEL"
cat "$LOCAL/src-llamacpp.VERSION" 2>/dev/null || echo "VERSION file absent"
echo "llama_src_files=$(find "$LOCAL/src-llamacpp" -type f | wc -l)  expected=3425"
echo "llama_src_manifest_collation=LC_ALL=C (locale-independent by construction)"
MAN=$(manifest_of "$LOCAL/src-llamacpp")
echo "llama_src_manifest_sha256=$MAN"
echo "llama_src_manifest_expected=$LLAMA_MANIFEST"
[ "$MAN" = "$LLAMA_MANIFEST" ] || fail "the staged llama.cpp tree is not the pinned content"
echo "SOURCE_OK"

step "4. build llama-bench and llama-cli for gfx1151, or prove them present"
LB=$LOCAL/build-llamacpp/bin
HAVE_LIB=$(find "$LOCAL/build-llamacpp" -name 'libllama.so*' 2>/dev/null | head -1)
if [ ! -x "$LB/llama-bench" ] || [ ! -x "$LB/llama-cli" ] || [ -z "$HAVE_LIB" ]; then
  echo "llamacpp_build=BUILDING (a target is absent on this worker)"
  podrun - bash -c "
    set -uo pipefail
    source $CCSETUP
    if [ ! -e /opt/rocm/include/rocblas/rocblas.h ]; then
      apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends rocblas-dev hipblas-dev
      echo \"apt_rc=\$?\"
    fi
    cmake -S /local/src-llamacpp -B /local/build-llamacpp \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=OFF \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4 \
      \"\${CCACHE_ARGS[@]}\" \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON \
      || { echo CONFIGURE_FAILED; exit 10; }
    cmake --build /local/build-llamacpp -j 4 --target llama-bench llama-cli || { echo BUILD_FAILED; exit 11; }
  " > "$JOBLOCAL/build-llamacpp.log" 2>&1
  echo "llamacpp_build_rc=$?"; tail -20 "$JOBLOCAL/build-llamacpp.log"
  cp -f "$JOBLOCAL/build-llamacpp.log" "$OUT/build-llamacpp.log" 2>/dev/null
else
  echo "llamacpp_build=ALREADY PRESENT"
fi
[ -x "$LB/llama-bench" ] || fail "llama-bench is not available"
[ -x "$LB/llama-cli" ]   || fail "llama-cli is not available"
printf 'llama_bench '; sha256sum "$LB/llama-bench"
printf 'llama_cli '; sha256sum "$LB/llama-cli"
echo "== every libllama/libggml shared object this run links =="
find "$LOCAL/build-llamacpp" \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -type f \
  | LC_ALL=C sort | xargs -r sha256sum
MAN2=$(manifest_of "$LOCAL/src-llamacpp")
echo "llama_src_manifest_after_build=$MAN2"
[ "$MAN2" = "$LLAMA_MANIFEST" ] || fail "the build dirtied the pinned llama.cpp tree"
# The binaries link *-impl.so BESIDE them. Without this they exit 127.
LDP="$LOCAL/build-llamacpp/bin:$LOCAL/build-llamacpp/lib:/opt/rocm/lib"
CENV=(-e "HOME=$JOBLOCAL" -e "LD_LIBRARY_PATH=$LDP")
echo "LD_LIBRARY_PATH=$LDP"

step "5. the oracle STARTS, and sees the GPU"
# llama-bench at this pin has no --version; -h is its identity probe. The
# build_commit/build_number the pin is checked against come out of the leg JSON.
podrun 3m "$LB/llama-bench" -h > "$OUT/bench-help.txt" 2>&1 < /dev/null
echo "llama-bench -h rc=$?"; head -3 "$OUT/bench-help.txt"
grep -qi 'usage' "$OUT/bench-help.txt" || \
  fail "llama-bench printed no usage; the entrypoint did not become the process"
printf 'llama_cli_version '; podrun 3m "$LB/llama-cli" --version 2>&1 < /dev/null | head -3
podrun 3m "$LB/llama-bench" --list-devices > "$OUT/list-devices.txt" 2>&1 < /dev/null
echo "llama-bench --list-devices rc=$?"; cat "$OUT/list-devices.txt"
grep -qi 'rocm\|hip\|gfx' "$OUT/list-devices.txt" || \
  fail "llama-bench enumerates no ROCm device; this build would decode on the CPU"

step "6. system_info, from the oracle itself"
# -v reaches common.cpp:417, which is where common_params_get_system_info is
# emitted at trace level.
#
# STDOUT GOES TO /dev/null AND THAT IS THE POINT. `llama-cli` at b10451 is a TUI
# in front of an in-process server: it draws a spinner and a prompt with
# backspace redraws, and with a closed stdin it never stops redrawing even after
# generation finishes cleanly. A first submission of this job wrote 3.9 GB to the
# share in four minutes that way, which is the 24.9 GB incident again with
# -no-cnv already set. -no-cnv does not prevent it, so discarding the stream
# does. Everything this step is for -- system_info, the device list, the buffer
# types, the memory breakdown -- is on STDERR, which terminates.
podrun 5m "$LB/llama-cli" -m "$GGUF" -ngl "$NGL" -n 1 -p "The capital of France is" \
  -no-cnv --temp 0 --seed 1 -v > /dev/null 2> "$JOBLOCAL/system-info.err" < /dev/null
echo "llama-cli system_info leg rc=$? (a timeout here is EXPECTED: the TUI does not exit)"
head -c 2000000 "$JOBLOCAL/system-info.err" > "$OUT/system-info.err"
echo "system_info_err_bytes=$(stat -c %s "$JOBLOCAL/system-info.err")"
echo "-- system_info line --"
grep -m1 'system_info:' "$OUT/system-info.err" || echo "system_info: NOT EMITTED at this verbosity"
echo "-- the buffer types the tensors actually landed in (this is use_extra_bufts, observed) --"
grep -E 'load_tensors:|repack|CPU_REPACK|buffer size' "$OUT/system-info.err" | head -20
echo "-- what the oracle declined to load (the blk.64 / MTP question) --"
grep -iE 'not used|unused|n_layer|blk\.64|nextn|tensors' "$OUT/system-info.err" | head -12

step "7. the measured legs: $LEGS legs, -ngl $NGL, n_gen $NGEN, -r $REPS each"
echo "design: LEGS=$LEGS REPS=$REPS NGEN=$NGEN NGL=$NGL"
echo "one leg = one llama-bench process = one model load + its own warmup + $REPS timed generations"
for i in $(seq 1 "$LEGS"); do
  echo "--- leg $i of $LEGS (order index $i) ---"
  clk=$LOCAL/out/clock-denom-leg$i.jsonl; rm -f "$clk"
  python3 "$W/job/amd_clock_sample.py" --output "$clk" --interval 0.25 &
  cpid=$!
  sleep 1
  podrun 25m "$LB/llama-bench" -m "$GGUF" -p 0 -n "$NGEN" -ngl "$NGL" -r "$REPS" -o json \
    > "$OUT/leg$i.json" 2> "$OUT/leg$i.err" < /dev/null
  rc=$?
  echo "$rc" > "$OUT/leg$i.rc"
  kill -TERM "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  cp -f "$clk" "$OUT/clock-leg$i.jsonl" 2>/dev/null
  echo "leg$i rc=$rc clock_samples=$(wc -l < "$clk" 2>/dev/null || echo 0) finished_utc=$(date -u +%FT%TZ)"
  grep -oE '"(avg_ts|stddev_ts|n_gen|n_gpu_layers|n_threads)": [^,}]*' "$OUT/leg$i.json" 2>/dev/null \
    | sed "s/^/leg$i /"
  grep -iE 'GPU Hang|HW Exception|Memory access fault|aborting' "$OUT/leg$i.err" | head -3
done

step "8. fold, with N taken from the design"
cp -f "$W/job/fold.py" "$JOBLOCAL/fold.py"
python3 "$JOBLOCAL/fold.py" --evidence "$OUT" --legs "$LEGS" > "$JOBLOCAL/RESULT.json.stdout" 2>&1
FOLD_RC=$?
cat "$JOBLOCAL/RESULT.json.stdout"
echo "fold_rc=$FOLD_RC"

step "9. what this job did NOT do"
echo "vllmcpp_binaries_built=0"
echo "vllmcpp_legs_run=0"
echo "ratios_computed=0"
echo "reason=this arm's declared token gate reads FAIL at 3 of 6; AGENTS.md Gates"
echo "reason=admits no performance result from it, and #2497 already retracted one"

cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null
echo "JOB_VERDICT=$([ "$FOLD_RC" = 0 ] && echo OK || echo INCOMPLETE)"
echo "=== DENOMINATOR JOB DONE ==="
