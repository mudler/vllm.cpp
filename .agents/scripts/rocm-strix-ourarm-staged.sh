#!/usr/bin/env bash
# STAGED, AND IT REFUSES TO RUN. The paired Qwen3.8-27B Q4_K_M decode
# measurement on strix:gpu0 (gfx1151), for issue #2497, row
# BACKEND-GATE-ROCM-LLAMACPP, spec
# .agents/specs/bench-rocm-llamacpp-denominator.md.
#
# WHY IT IS HERE AND WHY IT IS INERT
# ----------------------------------
# AGENTS.md Gates admits a performance result from an arm only after that arm's
# declared token-exact gate passes. This arm's gate reads TOKEN_GATE=FAIL at
# 3 of 6, so no throughput figure for this engine is admissible today and no
# ratio may be computed from one. #2497 has already had one measurement
# retracted for being taken ahead of that gate; this file exists so the retake
# costs one lease and not a design session, and it refuses to start so that it
# cannot become the same defect a second time.
#
# The refusal is the first executable thing in the file, before any path is
# read, any byte staged or any device opened. Everything below it is the design:
# order-alternated rounds, the declared leg count, a clock window per leg, and
# the assertion that no portable CPU reference-tier op ran.
#
# THE DENOMINATOR IS ALREADY MEASURED. llama.cpp b10451 on gfx1151 on this exact
# artifact was taken alone, because one engine's own speed is not a cross-engine
# claim: docs/bench-evidence/rocm-strix-llamacpp-denominator-20260902.md. This
# job re-takes it inside the same lease anyway, because two arms compared across
# two leases are two measurements of two boxes.
set -uo pipefail

# --- RATIFICATION GUARD begin ---
# STRIX_ARM_SPEED_RATIFIED_BY must NAME the decision that ratified this arm's
# speed axis, and naming means an issue reference. `=1` asserts nothing, and a
# variable that can be satisfied by a reflex is not a gate.
GUARD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --guard-only) GUARD_ONLY=1 ;;   # probe the guard and stop. Never a bypass.
    *) printf 'REFUSED: unknown argument %s\n' "$arg" >&2; exit 3 ;;
  esac
done

guard_refuse() {
  cat <<'REASON'
REFUSED: this measurement is not ratified, so it will not run.

  What is missing:   STRIX_ARM_SPEED_RATIFIED_BY, naming the decision that
                     ratified this arm's speed axis (an issue reference, not a 1)
  Why:               AGENTS.md Gates admits a performance result from an arm only
                     after that arm's declared token-exact gate passes
  The failing gate:  TOKEN_GATE=FAIL, 3 of 6 prompts, recorded in
                     docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md
  Who owns the retake: https://github.com/mudler/vllm.cpp/issues/2497, which has
                     already retracted one measurement taken ahead of this gate

  When the gate passes, set the variable to the decision, for example:
    STRIX_ARM_SPEED_RATIFIED_BY='ratified <date> by <who> on #2497, token gate PASS'
REASON
  printf 'REFUSED\n'
}

RATIFIED_BY="${STRIX_ARM_SPEED_RATIFIED_BY:-}"
# Two conditions, and each one is independently falsifiable. A bare `-z` test
# sat here until a mutation showed nothing could detect its removal: the length
# floor already refuses the empty and the unset value, so the `-z` clause was a
# third description of a case the other two decide. An assertion no mutation can
# break is not a guarantee, so it is gone rather than kept for its comment value.
#   - the floor refuses a bare reference: `#2497` carries an issue number and
#     still names no decision, no date and nobody
#   - the reference refuses a long assertion that points at nothing
if [ "${#RATIFIED_BY}" -lt 12 ] \
   || ! printf '%s' "$RATIFIED_BY" | grep -Eq '#[0-9]+'; then
  guard_refuse
  exit 3
fi
printf 'RATIFICATION_OK: %s\n' "$RATIFIED_BY"
printf 'The ratification is an assertion by a person. This script does not verify it,\n'
printf 'and it still computes no ratio: see the closing section.\n'
if [ "$GUARD_ONLY" = 1 ]; then
  printf 'guard-only probe; nothing was staged, loaded or measured.\n'
  exit 0
fi
# --- RATIFICATION GUARD end ---

W=/workspace/rocm-strix-ourarm
LOCAL=/tmp/rocm-strix-q4k
JOBLOCAL=/tmp/rocm-strix-ourarm
CCSETUP_DIR=/workspace/rocm-strix-q4k
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

# THE DESIGN. Both arms' leg counts are declared here and handed to the fold.
# They are never derived by counting log lines, and the reason is RE-EMISSION and
# not `tee`, which writes each line once: a job log carries each leg's figure in
# its own step line AND again inside the RESULT.json the job echoes back, so a
# grep tally over it answers whatever the grep term happens to hit. On the
# oracle-only run of 2026-09-02 that tally read 13 against six legs.
ROUNDS=4
REPEAT=4          # our arm loads once and runs REPEAT completions per leg
REPS=3            # the oracle's own repetitions per leg
NGEN=64
NGL=99
PROMPT='The capital of France is'
# Discard policy, declared in advance: run 1 of each of our legs is the
# in-process cold run and is discarded for that named cause. No other leg is
# discarded without a cause printed beside it.
COLD_RUNS=1

# HSA_ENABLE_SDMA is retired (#2511) and is set nowhere in this file.

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$JOBLOCAL" "$LOCAL/models" "$LOCAL/out" || exit 90
exec > >(tee -a "$JOBLOCAL/job-$TAG.log") 2>&1
printf 'ratified_by=%s\n' "$RATIFIED_BY"

fail() { echo "FATAL: $*"; echo "JOB_VERDICT=FAIL"; exit 1; }
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

LDP_OURS="$LOCAL/build-vllmcpp:/opt/rocm/lib"
LDP_ORACLE="$LOCAL/build-llamacpp/bin:$LOCAL/build-llamacpp/lib:/opt/rocm/lib"
EXTRA_ENV=()
podrun() {   # podrun <timeout|-> <ld_library_path> <entrypoint> [args...]
  local tmo=$1 ldp=$2 ep=$3; shift 3
  local -a pre=()
  [ "$tmo" != "-" ] && pre=(timeout --foreground "$tmo")
  "${pre[@]}" "${P[@]}" run --rm --entrypoint "$ep" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -e "HOME=$JOBLOCAL" -e "LD_LIBRARY_PATH=$ldp" "${EXTRA_ENV[@]}" \
    -v "$LOCAL:$LOCAL:rw" -v "$LOCAL:/local:rw" -v "$JOBLOCAL:$JOBLOCAL:rw" \
    -v "$W:$W:ro" -v "$CCSETUP_DIR:$CCSETUP_DIR:ro" \
    -v /workspace/ccache:/workspace/ccache:rw \
    "$IMG" "$@"
}

step "0. worker identity and contention state"
hostname; uname -a; echo "host_arch=$(uname -m) nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
env | grep -i '^HSA_\|^VT_\|^GGML_\|^ROCR_' | sed 's/^/inherited_env /' || echo "inherited_env NONE"
free -g; df -h /tmp /workspace | head -5

step "1. the artifact, verified ON the worker"
# /workspace is CIFS and is never a run surface.
if [ ! -f "$GGUF" ] || [ "$(stat -c %s "$GGUF")" != "$GGUF_SIZE" ]; then
  cp -f "$GGUF_NAS" "$GGUF" || fail "stage failed"
fi
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "gguf_bytes=$(stat -c %s "$GGUF") gguf_sha256=$GOT"
[ "$GOT" = "$GGUF_SHA" ] || fail "artifact sha256 mismatch on the worker"

step "2. both binaries, and what they are"
VC=$LOCAL/build-vllmcpp/examples/vllm-cli
LB=$LOCAL/build-llamacpp/bin/llama-bench
[ -x "$VC" ] || fail "our arm's binary is absent; build it in this lease before measuring"
[ -x "$LB" ] || fail "the oracle binary is absent; build it in this lease before measuring"
printf 'vllm_cli '; sha256sum "$VC"
printf 'llama_bench '; sha256sum "$LB"
find "$LOCAL/build-llamacpp" \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -type f \
  | LC_ALL=C sort | xargs -r sha256sum
MAN=$(manifest_of "$LOCAL/src-llamacpp")
echo "llama_src_manifest_sha256=$MAN (LC_ALL=C)"
[ "$MAN" = "$LLAMA_MANIFEST" ] || fail "the staged oracle tree is not the pinned content"

step "3. order-alternated legs: $ROUNDS rounds, both arms in every round"
echo "design: ROUNDS=$ROUNDS REPEAT=$REPEAT REPS=$REPS NGEN=$NGEN NGL=$NGL COLD_RUNS=$COLD_RUNS"
echo "our arm legs by design = $ROUNDS; oracle legs by design = $ROUNDS"

run_leg() {   # run_leg <arm> <round>
  local arm=$1 round=$2 tag rc cpid clk
  tag="${arm}-r${round}"
  # The sampler writes to WORKER-LOCAL disk. A 4 Hz flush against CIFS stalls the
  # sampler and distorts the very sample spacing the window is judged on.
  clk="$LOCAL/out/clock-$tag.jsonl"; rm -f "$clk"
  python3 "$W/amd_clock_sample.py" --output "$clk" --interval 0.25 &
  cpid=$!
  sleep 1
  case "$arm" in
    ours)
      # gfx1151 is INTEGRATED, so the portable CPU reference tier is reachable.
      # Without this the fallback is invisible in the throughput number, and the
      # assertion below would be reading a counter nothing ever wrote.
      EXTRA_ENV=(-e VT_OP_PROVIDER_STATS=1)
      podrun 25m "$LDP_OURS" "$VC" \
        --model "$GGUF" --prompt "$PROMPT" --max-tokens "$NGEN" \
        --temperature 0 --repeat "$REPEAT" --max-num-seqs 1 \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err" < /dev/null
      ;;
    oracle)
      EXTRA_ENV=()
      podrun 25m "$LDP_ORACLE" "$LB" \
        -m "$GGUF" -p 0 -n "$NGEN" -ngl "$NGL" -r "$REPS" -o json \
        > "$OUT/$tag.json" 2> "$OUT/$tag.err" < /dev/null
      ;;
  esac
  rc=$?
  echo "$rc" > "$OUT/$tag.rc"
  kill -TERM "$cpid" 2>/dev/null; wait "$cpid" 2>/dev/null
  cp -f "$clk" "$OUT/clock-$tag.jsonl" 2>/dev/null
  echo "$tag rc=$rc clock_samples=$(wc -l < "$clk" 2>/dev/null || echo 0) finished_utc=$(date -u +%FT%TZ)"
  EXTRA_ENV=()
  if [ "$arm" = ours ]; then
    # `grep -c` PRINTS its own 0 and EXITS 1 on no match, so a `|| echo 0`
    # fallback runs on top of that count and yields the two-line value "0\n0".
    # Worse, it makes an ABSENT stderr capture read as 0 -- a clean reference-tier
    # result -- which is the one reading this assertion exists to refuse.
    if [ -r "$OUT/$tag.err" ]; then
      echo "$tag reference_tier_notices=$(grep -c '\[vt reference-tier\]' "$OUT/$tag.err")"
    else
      echo "$tag reference_tier_notices=UNREAD (no stderr capture at $OUT/$tag.err)"
    fi
    grep -E 'run=[0-9]+/[0-9]+ finish_reason' "$OUT/$tag.err" | sed "s/^/$tag /"
  else
    grep -oE '"(avg_ts|stddev_ts|n_gen|n_gpu_layers)": [^,}]*' "$OUT/$tag.json" | sed "s/^/$tag /"
  fi
  grep -iE 'GPU Hang|HW Exception|Memory access fault' "$OUT/$tag.err" | head -3
}

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="ours oracle"; else order="oracle ours"; fi
  echo "--- round $r order: $order ---"
  for arm in $order; do run_leg "$arm" "$r"; done
done

step "4. the numbers, per arm, and no ratio between them"
python3 "$W/fold-paired.py" --evidence "$OUT" --rounds "$ROUNDS" \
  --cold-runs "$COLD_RUNS" > "$JOBLOCAL/RESULT.json.stdout" 2>&1
FOLD_RC=$?
cat "$JOBLOCAL/RESULT.json.stdout"

step "5. what this job deliberately did not compute"
cat <<'CLOSING'
ratios_computed=0

Dividing two medians is arithmetic anyone can do. Deciding that the division is
ADMISSIBLE is not, and that decision is exactly what STRIX_ARM_SPEED_RATIFIED_BY
asserts a person made. Printing both medians and stopping keeps the assertion
where the person made it, instead of laundering it into a number this script
produced.

Two terms the reader must carry before dividing anything:
  - the clock offset between the arms, per leg, from the compute windows above;
    throughput is an integral over the window, so the MEANS are what transfer
  - the residency difference: the oracle declines blk.64, the drafter layer,
    while our loader holds it. Both arms run 64 trunk blocks per token, so that
    belongs in the memory column and not in a throughput ratio.

    NO SIZE IS WRITTEN HERE, ON PURPOSE. This row's spec forbids it producing,
    recording or quoting ANY throughput, latency or memory figure for vllm.cpp,
    with no exception written, because this arm's declared token gate reads
    TOKEN_GATE=FAIL at 3 of 6 -- and "how much our loader holds" is a memory
    figure for vllm.cpp however casually it is phrased. A residency figure
    attributed to our loader stood on this line and is deleted, not restated
    here: repeating it to explain its removal is still quoting it. The warning
    is what a future reader needs. The number is what this row may not give
    them, and it is recoverable from `git log -S` if the gate ever passes.
CLOSING
echo "JOB_VERDICT=$([ "$FOLD_RC" = 0 ] && echo OK || echo INCOMPLETE)"
echo "=== PAIRED JOB DONE ==="
