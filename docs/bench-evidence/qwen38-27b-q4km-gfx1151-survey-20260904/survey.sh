#!/usr/bin/env bash
# THE MULTI-ENGINE THROUGHPUT SURVEY on strix:gpu0 (gfx1151), for #2497.
# Three engines -- vllm.cpp, llama.cpp b10451 and the PINNED vLLM -- on one
# board, one artifact, one prompt and one token count, inside one lease.
#
# WHAT THIS IS: a survey, published together with its correctness status.
# WHAT IT IS NOT: a parity claim, or a statement that the token gate passes.
# The arm's declared token gate reads TOKEN_GATE=FAIL at 3 of 6 prompts and no
# deterministic denominator exists on this path. The developer was told that and
# reaffirmed the request (developer-preferences.md, 2026-09-04), so it proceeds
# with the correctness state ON THE FACE of the table rather than under it.
set -uo pipefail

# --- RATIFICATION GUARD begin ---
# Same contract as .agents/scripts/rocm-strix-ourarm-staged.sh: the variable must
# NAME a decision, and naming means an issue reference. `=1` asserts nothing.
# The `#[0-9]+` pattern is known-weak (undetected mutation M15), so the value
# passed here names a real, recorded decision rather than satisfying a regex.
GUARD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --guard-only) GUARD_ONLY=1 ;;
    *) printf 'REFUSED: unknown argument %s\n' "$arg" >&2; exit 3 ;;
  esac
done
RATIFIED_BY="${STRIX_ARM_SPEED_RATIFIED_BY:-}"
if [ "${#RATIFIED_BY}" -lt 12 ] || ! printf '%s' "$RATIFIED_BY" | grep -Eq '#[0-9]+'; then
  cat <<'REASON'
REFUSED: this survey is not ratified, so it will not run.
  Missing: STRIX_ARM_SPEED_RATIFIED_BY, naming the decision that ratified it.
  The failing gate: TOKEN_GATE=FAIL, 3 of 6, in
    docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md
REASON
  printf 'REFUSED\n'; exit 3
fi
printf 'RATIFICATION_OK: %s\n' "$RATIFIED_BY"
[ "$GUARD_ONLY" = 1 ] && { echo "guard-only probe; nothing staged, loaded or measured."; exit 0; }
# --- RATIFICATION GUARD end ---

W=/workspace/strix-survey-2497
LOCAL=/tmp/rocm-strix-q4k
VLLMLOCAL=/tmp/vllm-gfx1151-2740
JOBLOCAL=/tmp/strix-survey-2497
IMG=vllmcpp-rocm-build:7.2.4
P=(podman --storage-driver=vfs --root /tmp/podman-pr66-root-vfs --runroot /tmp/podman-pr66-run-vfs)

GGUF_NAME=Qwen3.8-27B-Q4_K_M.gguf
GGUF_NAS=/workspace/ckpt/qwen38-27b-q4km/$GGUF_NAME
GGUF=$LOCAL/models/$GGUF_NAME
GGUF_SIZE=17106775008
GGUF_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
LLAMA_MANIFEST=56c26d15c2acf11b8621ac26663b4316dc29719d765ba1d95231ffacaddf3cda
manifest_of() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z \
                 | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }

TOKDIR=/workspace/ckpt/qwen3.8-27b-hf
MMPROJ=/workspace/ggufplugin/mmproj-BF16.gguf
PY=$VLLMLOCAL/venv/bin/python

# THE DESIGN, declared here and handed to the fold. Never derived from a tally.
ROUNDS="${ROUNDS:-4}"      # legs per arm
REPEAT=4                   # our arm and vLLM: completions per leg (1 discarded)
REPS=3                     # llama-bench's own repetitions per leg
NGEN=64
NGL=99
PROMPT='The capital of France is'
COLD_RUNS=1                # run 1 of each in-process arm is the cold run
DRYRUN="${DRYRUN:-0}"      # 1 = harness validation, NOT a published leg

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$JOBLOCAL" "$LOCAL/models" "$LOCAL/out" || exit 90
exec > >(tee -a "$JOBLOCAL/job-$TAG.log") 2>&1
# The log is on worker-local disk and mirrored to the share, so a board fault
# mid-run does not take the record of what had already happened with it.
( while true; do cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT

fail() { echo "FATAL: $*"; echo "JOB_VERDICT=FAIL"; exit 1; }
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

step "0. worker identity, inherited environment, contention state"
echo "ratified_by=$RATIFIED_BY"
echo "DRYRUN=$DRYRUN  (1 means this run is harness validation and no leg of it is published)"
hostname; uname -a; echo "host_arch=$(uname -m) nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "uptime_s=$(cut -d' ' -f1 /proc/uptime)"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}   <- must read UNSET"
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || fail "HSA_OVERRIDE_GFX_VERSION is set; this would measure a different device"
if env | grep -qE '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)'; then
  env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)' | sed 's/^/inherited_env /'
  fail "an HSA_/ROCR_/PYTORCH_/HIP_/GGML_/VT_ variable was inherited into this job"
else
  echo "inherited_env NONE"
fi
free -g | head -3; df -h /tmp /workspace | head -5

step "1. the artifact, verified ON the worker before any timing"
if [ ! -f "$GGUF" ] || [ "$(stat -c %s "$GGUF")" != "$GGUF_SIZE" ]; then
  cp -f "$GGUF_NAS" "$GGUF" || fail "stage failed"
fi
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "gguf_bytes=$(stat -c %s "$GGUF") gguf_sha256=$GOT"
[ "$GOT" = "$GGUF_SHA" ] || fail "artifact sha256 mismatch on the worker"

step "2. the three engines, pinned as executed bytes"
VC=$LOCAL/build-vllmcpp/examples/vllm-cli
LB=$LOCAL/build-llamacpp/bin/llama-bench
LC=$LOCAL/build-llamacpp/bin/llama-cli
[ -x "$VC" ] || fail "vllm.cpp binary absent"
[ -x "$LB" ] || fail "llama-bench absent"
[ -x "$LC" ] || fail "llama-cli absent"
[ -x "$PY"  ] || fail "the vLLM venv interpreter is absent"
printf 'vllm_cli '; sha256sum "$VC"
printf 'llama_bench '; sha256sum "$LB"
printf 'llama_cli '; sha256sum "$LC"
printf 'vllm_python '; sha256sum "$PY"
find "$LOCAL/build-llamacpp" \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -type f \
  | LC_ALL=C sort | xargs -r sha256sum
find "$LOCAL/build-vllmcpp" -maxdepth 1 -name 'lib*.so*' -type f | LC_ALL=C sort | xargs -r sha256sum
MAN=$(manifest_of "$LOCAL/src-llamacpp")
echo "llama_src_manifest_sha256=$MAN (LC_ALL=C)"
[ "$MAN" = "$LLAMA_MANIFEST" ] || fail "the staged oracle tree is not the pinned content"
echo "vllmcpp_source_revision=$(git -C "$LOCAL/src-vllmcpp" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
"$PY" -c "import vllm,torch;print('VLLM',vllm.__version__);print('TORCH',torch.__version__,'hip',torch.version.hip);print('ARCH',torch.cuda.get_device_properties(0).gcnArchName)" \
  || fail "the vLLM venv does not import"
"$PY" -c "import vllm_gguf_plugin._C_gguf as c;print('PLUGIN_C_EXT',c.__file__)" || echo "PLUGIN probe failed"

step "3. interleaved legs: $ROUNDS rounds, all three arms in every round"
echo "design: ROUNDS=$ROUNDS REPEAT=$REPEAT REPS=$REPS NGEN=$NGEN NGL=$NGL COLD_RUNS=$COLD_RUNS"
echo "legs by design, per arm = $ROUNDS  (the fold is told this; it counts nothing)"
echo "arm order rotates by round so drift cannot be read as a difference"

LDP_OURS="$LOCAL/build-vllmcpp:/opt/rocm/lib"
LDP_ORACLE="$LOCAL/build-llamacpp/bin:$LOCAL/build-llamacpp/lib:/opt/rocm/lib"
EXTRA_ENV=()
podrun() {   # podrun <timeout> <ld_library_path> <entrypoint> [args...]
  local tmo=$1 ldp=$2 ep=$3; shift 3
  timeout --foreground "$tmo" "${P[@]}" run --rm --entrypoint "$ep" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e "HOME=$JOBLOCAL" -e "LD_LIBRARY_PATH=$ldp" "${EXTRA_ENV[@]}" \
    -v "$LOCAL:$LOCAL:rw" -v "$JOBLOCAL:$JOBLOCAL:rw" -v "$W:$W:ro" \
    "$IMG" "$@"
}

run_leg() {   # run_leg <arm> <round>
  local arm=$1 round=$2 tag rc cpid clk
  tag="${arm}-r${round}"
  if [ -f "$OUT/$tag.done" ]; then echo "$tag SKIP (already complete in this evidence dir)"; return; fi
  clk="$LOCAL/out/clock-$tag.jsonl"; rm -f "$clk"
  # The sampler writes to WORKER-LOCAL disk. A 4 Hz flush against CIFS stalls the
  # sampler and distorts the very sample spacing the window is judged on.
  python3 "$W/amd_clock_sample.py" --output "$clk" --interval 0.25 &
  cpid=$!
  sleep 1
  echo "--- leg $tag start $(date -u +%FT%TZ) ---"
  case "$arm" in
    vllmcpp)
      # gfx1151 is INTEGRATED, so the portable CPU reference tier is reachable.
      # Without this the fallback is invisible in the throughput number.
      EXTRA_ENV=(-e VT_OP_PROVIDER_STATS=1)
      podrun 25m "$LDP_OURS" "$VC" \
        --model "$GGUF" --prompt "$PROMPT" --max-tokens "$NGEN" \
        --temperature 0 --repeat "$REPEAT" --max-num-seqs 1 \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err" < /dev/null
      rc=$?
      ;;
    llamacpp)
      EXTRA_ENV=()
      podrun 25m "$LDP_ORACLE" "$LB" \
        -m "$GGUF" -p 0 -n "$NGEN" -ngl "$NGL" -r "$REPS" -o json \
        > "$OUT/$tag.json" 2> "$OUT/$tag.err" < /dev/null
      rc=$?
      ;;
    llamacli)
      # THE END-TO-END CONTROL, and the reason it exists. `llama-bench -p 0` is a
      # PURE DECODE figure: it excludes the prompt and every per-request cost.
      # `vllm-cli` and the vLLM leg time a whole completion. Comparing those two
      # definitions across engines would charge one engine for work the other's
      # number never contained. This leg decodes the SAME prompt for the SAME 64
      # tokens through llama.cpp's ordinary request path, so the table has one
      # row per engine on each of the two definitions rather than a mixture.
      EXTRA_ENV=()
      podrun 25m "$LDP_ORACLE" "$LC" \
        -m "$GGUF" -p "$PROMPT" -n "$NGEN" -ngl "$NGL" --temp 0 -no-cnv --seed 1 \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err" < /dev/null
      rc=$?
      ;;
    vllm)
      # vLLM runs on the HOST out of its own venv, not in the pod: that is where
      # it was built and where its compile cache lives. Both paths open /dev/kfd
      # directly, and the asymmetry is recorded rather than hidden.
      # PRODUCTION CONFIGURATION: EAGER=0. AGENTS.md Gates forbids
      # --enforce-eager as a denominator, so the compiled arm is the headline.
      env -u HSA_OVERRIDE_GFX_VERSION \
        MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf \
        GMU=0.75 MAXLEN=4096 EAGER=0 NGEN="$NGEN" REPEAT="$REPEAT" \
        PROMPT="$PROMPT" LEG_TAG="$tag" OUT_JSON="$OUT/$tag.json" \
        setsid timeout 45m "$PY" "$W/bench_vllm.py" \
        > "$OUT/$tag.err" 2>&1 < /dev/null
      rc=$?
      ;;
  esac
  echo "$rc" > "$OUT/$tag.rc"
  kill -TERM "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  cp -f "$clk" "$OUT/clock-$tag.jsonl" 2>/dev/null
  echo "$tag rc=$rc clock_samples=$(wc -l < "$clk" 2>/dev/null || echo 0) finished_utc=$(date -u +%FT%TZ)"
  case "$arm" in
    vllmcpp)
      # `grep -c` PRINTS its own 0 and EXITS 1 on no match, so a `|| echo 0`
      # fallback yields "0\n0" and makes an ABSENT capture read as a clean run.
      if [ -r "$OUT/$tag.err" ]; then
        echo "$tag reference_tier_notices=$(grep -c '\[vt reference-tier\]' "$OUT/$tag.err")"
      else
        echo "$tag reference_tier_notices=UNREAD (no stderr capture at $OUT/$tag.err)"
      fi
      grep -E 'run=[0-9]+/[0-9]+ finish_reason' "$OUT/$tag.err" | sed "s/^/$tag /"
      ;;
    llamacpp)
      grep -oE '"(avg_ts|stddev_ts|n_gen|n_prompt|n_gpu_layers|backend)": [^,}]*' "$OUT/$tag.json" | sed "s/^/$tag /"
      ;;
    llamacli)
      grep -E 'llama_perf_context_print|llama_perf_sampler_print' "$OUT/$tag.err" | sed "s/^/$tag /"
      ;;
    vllm)
      grep -E 'ENGINE_UP|vllm: run=|one_token_secs|VLLM_VERSION|ON_GFX1151|PRIMARY_CONFIG_FAILED|DONE_MARKER' "$OUT/$tag.err" | sed "s/^/$tag /"
      ;;
  esac
  grep -iE 'GPU Hang|HW Exception|Memory access fault' "$OUT/$tag.err" | head -3
  [ "$rc" = 0 ] && touch "$OUT/$tag.done"
  EXTRA_ENV=()
}

# Order rotates so no arm sits permanently in the warmest or coldest slot.
orders=("vllmcpp llamacpp llamacli vllm" \
        "llamacpp llamacli vllm vllmcpp" \
        "llamacli vllm vllmcpp llamacpp" \
        "vllm vllmcpp llamacpp llamacli")
for r in $(seq 1 "$ROUNDS"); do
  order="${orders[$(( (r - 1) % ${#orders[@]} ))]}"
  echo; echo "--- round $r of $ROUNDS, order: $order ---"
  for arm in $order; do run_leg "$arm" "$r"; done
  cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null
done

step "4. the fold: each arm's own figures, and the ratios between them"
python3 "$W/fold-survey.py" --evidence "$OUT" --rounds "$ROUNDS" \
  --arms vllmcpp,llamacpp,llamacli,vllm \
  --cold-runs "$COLD_RUNS" > "$OUT/RESULT.json" 2> "$OUT/fold.err"
FOLD_RC=$?
cat "$OUT/RESULT.json"
echo "fold_rc=$FOLD_RC"; cat "$OUT/fold.err"

step "5. what a reader must carry away with any ratio above"
cat <<'CLOSING'
TOKEN_GATE=FAIL. This is a SURVEY, not a parity result.

  vllm.cpp ROCm arm   vs llama.cpp b10451     3 of 6 prompts divergent
  vllm.cpp ROCm arm   vs vLLM (compiled)      5 of 6 prompts divergent
  vLLM (compiled)     vs llama.cpp b10451     3 of 6 prompts divergent
  vLLM (eager)        vs llama.cpp b10451     4 of 6 prompts divergent

Every divergence is a near-tie at about 0.125 nats, one bf16 ULP.

NO DETERMINISTIC DENOMINATOR EXISTS ON THIS PATH. llama.cpp's greedy decode is
not stable across its own kernel paths (.agents/oracles/llama-cpp.md), and vLLM
disagrees with ITSELF across enforce_eager on both 27B models tried (#2740,
#2915). Limb 3 of the ratified methodology is unsatisfied and may be
unsatisfiable here (#2864, #2884).

The ratio and the caveat are one object. A reader must not be able to take the
first without the second.
CLOSING
echo "DRYRUN=$DRYRUN"
echo "OUT=$OUT"
cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null
echo "fold_rc=$FOLD_RC"
echo "=== SURVEY JOB DONE ==="
