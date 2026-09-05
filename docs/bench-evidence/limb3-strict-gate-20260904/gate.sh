#!/usr/bin/env bash
# #2884 -- LIMB 3 OF THE RATIFIED NEAR-TIE METHODOLOGY, RUN ON A VEHICLE.
#
# Spec: .agents/specs/limb3-vehicle-strict-gate.md
#
# #2864 found NO admissible limb-3 vehicle among the 77 GGUFs this fleet held,
# and its §7 option 2 said what one would be: a second dense `qwen35` k-quant
# GGUF that is not Qwen3.8-27B. `unsloth/Qwen3.6-27B-GGUF` @ 82d411acf is that
# artifact, pinned and header-verified before it was fetched.
#
# WHAT THIS SCORES: a STRICT, FREE-RUNNING token-exact pass of our ROCm k-quant
# path against the PINNED vLLM, on the vehicle, on the SIX pre-registered
# prompts at 48 tokens. Strict means every token of every prompt.
#
# WHAT IT DOES NOT DO: no throughput, latency or memory figure and no
# cross-engine ratio. AGENTS.md Gates admits no performance result from an arm
# whose declared token gate has not passed, and #2497 already carries one
# retraction for exactly that. STRIX_ARM_SPEED_RATIFIED_BY is not set here and
# .agents/scripts/rocm-strix-ourarm-staged.sh stays refusing.
#
# HSA_OVERRIDE_GFX_VERSION is set NOWHERE. That knob makes the runtime report a
# different device, which is the one thing an oracle measurement cannot survive.
set -uo pipefail

W=/workspace/limb3-2884
LOCAL=/tmp/rocm-strix-q4k                 # the 2026-09-01..03 staging, reused
JOBLOCAL=/tmp/limb3-2884
BLD2=$LOCAL/build-vllmcpp-v2              # the binaries the arm's gate ran
VENV=/tmp/vllm-gfx1151-2740/venv          # the pinned vLLM built for gfx1151
PY=$VENV/bin/python

GGUF_NAME=Qwen3.6-27B-Q4_K_M.gguf
GGUF_NAS=/workspace/ckpt/qwen36-27b-q4km/$GGUF_NAME
GGUF=$LOCAL/models/$GGUF_NAME
GGUF_SIZE=16817244384
GGUF_SHA="${GGUF_SHA:?the staged sha256, measured on the devbox, must be passed in}"
TOK=/workspace/ckpt/qwen3.6-27b-hf
MMPROJ=/workspace/ckpt/qwen36-27b-q4km/mmproj-BF16.gguf
MMPROJ_SIZE=931146304

# The binaries this reuses, by the sha256 the arm's own gate recorded. Reusing
# a build is only admissible if it is provably THE build, so each is asserted
# against the value committed in
# docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md.
WANT_BENCH=d7292e3895fb4887149ec3c5902cc08fa080f88e0f8851d61848753466b6501d
WANT_TOKENIZE=8673794406f1ef3a6239941c0a48a52e3d68a23b78303ac4e7b07335b589ce77
WANT_LIB=2c3ac67eae458763a2c282e1fc601b60d46d408296a7d56955a36d876c9b0b65

PROMPTS_SHA=c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e
N_PREDICT=48
LEGS=${LEGS:-3}
IMG=vllmcpp-rocm-build:7.2.4
CCSETUP_DIR=/workspace/rocm-strix-q4k
P=(podman --storage-driver=vfs --root /tmp/podman-pr66-root-vfs --runroot /tmp/podman-pr66-run-vfs)

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$JOBLOCAL" "$LOCAL/models" || exit 90
exec > >(tee -a "$JOBLOCAL/job-$TAG.log") 2>&1
( while true; do cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT

fail() { echo "FATAL: $*"; echo "JOB_VERDICT=FAIL"; echo "STRICT_LIMB3=NOT_MEASURED"; exit 1; }
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

# podrun <timeout|-> <entrypoint> [args...]
# The timeout is an ARGUMENT, not a prefix: `timeout` is an external binary and
# cannot see a shell function, and this campaign has already lost a submission
# to a helper whose name also names the OCI runtime.
CENV=()
podrun() {
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

step "0. worker identity, and the environment this run REFUSES to inherit"
echo "hostname=$(hostname) nproc=$(nproc)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || fail "HSA_OVERRIDE_GFX_VERSION is set"
INH=$(env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|VT_)' || true)
if [ -n "$INH" ]; then echo "$INH" | sed 's/^/inherited_env /'; fail "the job inherited HSA_/ROCR_/PYTORCH_/HIP_/VT_ state"; fi
echo "inherited_env NONE"
t=$(type -t podrun); echo "podrun_resolves_to=$t"; [ "$t" = function ] || fail "podrun is not our function"
free -g | head -2; df -h /tmp /workspace | head -4

step "1. THE VEHICLE, verified ON the worker"
if [ ! -f "$GGUF" ] || [ "$(stat -c %s "$GGUF")" != "$GGUF_SIZE" ]; then
  echo "staging $GGUF_NAS -> $GGUF"
  cp -f "$GGUF_NAS" "$GGUF" || fail "stage failed"
fi
echo "gguf_bytes=$(stat -c %s "$GGUF")  expected=$GGUF_SIZE"
[ "$(stat -c %s "$GGUF")" = "$GGUF_SIZE" ] || fail "size mismatch"
GOT=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "gguf_sha256=$GOT"
echo "gguf_sha256_expected=$GGUF_SHA"
[ "$GOT" = "$GGUF_SHA" ] || fail "vehicle sha256 mismatch: the bytes on the board are not the pinned bytes"
echo "vehicle_repo=unsloth/Qwen3.6-27B-GGUF"
echo "vehicle_revision=82d411acf4a06cfb8d9b073a5211bf410bfc29bf"
ls -la "$TOK" | head -12
[ -f "$MMPROJ" ] || fail "the vehicle's vision tower is absent: $MMPROJ"
echo "mmproj_bytes=$(stat -c %s "$MMPROJ")  expected=$MMPROJ_SIZE"
[ "$(stat -c %s "$MMPROJ")" = "$MMPROJ_SIZE" ] || fail "mmproj size mismatch"
echo "mmproj_sha256=$(sha256sum "$MMPROJ" | cut -d' ' -f1)"

step "2. THE BINARY UNDER TEST IS THE ONE THE ARM'S GATE RAN"
# Reusing a build is admissible only if it is provably THE build. These three
# values are committed in the token-gate v2 evidence; a rebuild would not
# reproduce them and would be visible here rather than silent.
for pair in "vllm-bench:$BLD2/examples/vllm-bench:$WANT_BENCH" \
            "tokenize:$BLD2/examples/tokenize:$WANT_TOKENIZE" \
            "libvllm.so:$BLD2/libvllm.so:$WANT_LIB"; do
  nm=${pair%%:*}; rest=${pair#*:}; path=${rest%%:*}; want=${rest#*:}
  [ -e "$path" ] || fail "missing build product $path"
  got=$(sha256sum "$path" | cut -d' ' -f1)
  printf '%-12s %s\n' "$nm" "$got"
  [ "$got" = "$want" ] || fail "$nm is not the gated binary (want $want)"
done
echo "BINARIES_ARE_THE_GATED_ONES=yes"

step "3. THE PROMPTS ARE PRE-REGISTERED, AND HASHED BEFORE ANYTHING IS SCORED"
cp -f "$W/prompts.txt" "$JOBLOCAL/prompts.txt" || fail "no prompts"
cat -A "$JOBLOCAL/prompts.txt"
GOTP=$(sha256sum "$JOBLOCAL/prompts.txt" | cut -d' ' -f1)
echo "prompts_sha256=$GOTP"
echo "prompts_sha256_expected=$PROMPTS_SHA"
[ "$GOTP" = "$PROMPTS_SHA" ] || fail "the prompt set is not the pre-registered one"
NPROMPTS=$(grep -c . "$JOBLOCAL/prompts.txt")
echo "n_prompts=$NPROMPTS n_predict=$N_PREDICT"
echo "PRE_REGISTERED=yes  (this set has been scored by this campaign since 2026-08-23;"
echo "                     its hash is committed in three evidence documents that predate this run)"

step "4. OUR tokenizer, off the VEHICLE's own GGUF vocab"
CENV=(-e "LD_LIBRARY_PATH=$BLD2:/opt/rocm/lib" -e "HOME=$JOBLOCAL")
podrun 10m "$BLD2/examples/tokenize" "$GGUF" "$JOBLOCAL/prompts.txt" \
  > "$JOBLOCAL/ours_prompt_ids.txt" 2> "$JOBLOCAL/ours_tokenize.stderr" < /dev/null
echo "tokenize_rc=$?"
cp -f "$JOBLOCAL/ours_prompt_ids.txt" "$OUT/ours_prompt_ids.txt" 2>/dev/null
cat "$JOBLOCAL/ours_prompt_ids.txt"
[ -s "$JOBLOCAL/ours_prompt_ids.txt" ] || { tail -20 "$JOBLOCAL/ours_tokenize.stderr"; fail "our tokenizer produced nothing on the vehicle"; }
python3 - "$JOBLOCAL/ours_prompt_ids.txt" "$JOBLOCAL/feed_ids.json" <<'PY'
import json, sys
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    rows.append([] if line == "EMPTY" else [int(x) for x in line.split()])
json.dump(rows, open(sys.argv[2], "w"))
print("FEED_IDS n=%d lens=%s" % (len(rows), [len(r) for r in rows]))
PY
cp -f "$JOBLOCAL/feed_ids.json" "$OUT/feed_ids.json" 2>/dev/null

step "5. OUR ROCm generation, $LEGS INDEPENDENT legs, SHIPPED DEFAULT, no knobs"
python3 - "$JOBLOCAL/prompts.txt" "$JOBLOCAL/prompts.sharegpt.json" <<'PY'
import json, sys
lines = [l.rstrip("\n").rstrip("\r") for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
json.dump([{"conversations": [{"value": p}]} for p in lines], open(sys.argv[2], "w", encoding="utf-8"))
print("sharegpt_prompts=%d" % len(lines))
PY
OK_LEGS=0; FAULT_LEGS=0; HARN_LEGS=0; FIRST_OK=""
for L in $(seq 1 "$LEGS"); do
  IDS=$JOBLOCAL/ours_gen_ids_$L.json
  rm -f "$IDS" "$OUT/ours_gen_ids_$L.json"
  CENV=(-e "LD_LIBRARY_PATH=$BLD2:/opt/rocm/lib" -e "HOME=$JOBLOCAL" -e VT_OP_PROVIDER_STATS=1)
  podrun 40m "$BLD2/examples/vllm-bench" --model "$GGUF" \
    --dataset-path "$JOBLOCAL/prompts.sharegpt.json" \
    --num-prompts "$NPROMPTS" --input-len 64 --output-len "$N_PREDICT" \
    --concurrency 1 --seed 1234 --temperature 0 --num-blocks 256 \
    --output-token-ids "$IDS" \
    > "$JOBLOCAL/ours_bench_$L.stdout" 2> "$JOBLOCAL/ours_bench_$L.stderr" < /dev/null
  RC=$?
  cp -f "$IDS" "$OUT/ours_gen_ids_$L.json" 2>/dev/null
  cp -f "$JOBLOCAL/ours_bench_$L.stderr" "$OUT/ours_bench_$L.stderr" 2>/dev/null
  REF=$(grep -c '\[vt reference-tier\]' "$JOBLOCAL/ours_bench_$L.stderr" 2>/dev/null); REF=${REF:-0}
  DEV=$(grep -c 'device=5' "$JOBLOCAL/ours_bench_$L.stderr" 2>/dev/null); DEV=${DEV:-0}
  FAULT=$(grep -cE 'GPU Hang|Memory access fault|HW Exception|Aborted|core dumped' "$JOBLOCAL/ours_bench_$L.stderr" 2>/dev/null); FAULT=${FAULT:-0}
  if [ "$RC" = 0 ] && [ -s "$OUT/ours_gen_ids_$L.json" ]; then
    OK_LEGS=$((OK_LEGS+1)); [ -z "$FIRST_OK" ] && FIRST_OK=$L
    echo "MLEG $L rc=$RC status=OK rocm_device_lines=$DEV reference_tier_hits=$REF fault_lines=$FAULT"
    [ "$REF" = 0 ] || fail "leg $L took $REF reference-tier ops; it measured the CPU tier wearing a ROCm label"
    [ "$DEV" -gt 0 ] || fail "leg $L never reported a kROCM (device=5) op selection; it did not run on the board"
  elif [ "$FAULT" -gt 0 ] || [ "$RC" = 139 ] || [ "$RC" = 134 ]; then
    FAULT_LEGS=$((FAULT_LEGS+1))
    echo "MLEG $L rc=$RC status=BOARD_FAULT rocm_device_lines=$DEV reference_tier_hits=$REF fault_lines=$FAULT"
    grep -aE 'GPU Hang|Memory access fault|HW Exception' "$JOBLOCAL/ours_bench_$L.stderr" | head -3
  else
    HARN_LEGS=$((HARN_LEGS+1))
    echo "MLEG $L rc=$RC status=HARNESS_ERROR rocm_device_lines=$DEV reference_tier_hits=$REF fault_lines=$FAULT"
    tail -12 "$JOBLOCAL/ours_bench_$L.stderr"
  fi
done
echo "OUR_LEGS_OK=$OK_LEGS OUR_LEGS_BOARD_FAULT=$FAULT_LEGS OUR_LEGS_HARNESS_ERROR=$HARN_LEGS OUR_LEGS_TOTAL=$LEGS"
if [ "$OK_LEGS" -lt 2 ]; then
  echo "STRICT_LIMB3=NOT_MEASURED"
  echo "REASON=our arm produced $OK_LEGS clean legs on the vehicle; a lone survivor is not a verdict, because survival is not independent of correctness while anything can corrupt memory"
  echo "=== JOB DONE ==="; exit 0
fi

step "5b. do OUR own clean legs reproduce each other on the vehicle"
python3 - "$OUT" "$LEGS" <<'PY'
import json, os, sys
out, legs = sys.argv[1], int(sys.argv[2])
got = {}
for L in range(1, legs + 1):
    p = os.path.join(out, "ours_gen_ids_%d.json" % L)
    if os.path.exists(p) and os.path.getsize(p):
        got[L] = json.load(open(p))
ks = sorted(got)
same = True
for k in ks[1:]:
    if got[k] != got[ks[0]]:
        same = False
        for i, (a, b) in enumerate(zip(got[ks[0]], got[k])):
            if a != b:
                d = next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)
                print("  leg %d vs %d prompt %d first differs at %d: %d vs %d"
                      % (ks[0], k, i, d, a[d], b[d]))
print("OUR_LEGS_REPRODUCE_EACH_OTHER=%s over legs %s" % (same, ks))
PY
OURS=$OUT/ours_gen_ids_$FIRST_OK.json
echo "scoring leg $FIRST_OK: $OURS"

step "6. THE PINNED vLLM: is the phase-2 artifact still here, and is it the pin?"
[ -x "$PY" ] || fail "the #2740 venv is gone; this job cannot rebuild vLLM inside its budget"
cd /
"$PY" - <<'PY' || fail "the vLLM artifact does not satisfy its own identity assertions"
import vllm, torch
print("VLLM", vllm.__version__, vllm.__file__)
print("TORCH", torch.__version__, "hip", torch.version.hip)
print("ARCH", torch.cuda.get_device_properties(0).gcnArchName)
assert vllm.__version__.startswith("0.26.0.dev0+g5559679229"), vllm.__version__
assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151")
import vllm._C, vllm._rocm_C  # noqa
from vllm.platforms import current_platform
print("CURRENT_PLATFORM", type(current_platform).__name__)
assert type(current_platform).__name__ == "RocmPlatform"
import vllm_gguf_plugin, vllm_gguf_plugin._C_gguf as cg
print("PLUGIN", vllm_gguf_plugin.__file__)
print("PLUGIN_C_EXT", cg.__file__)
print("VLLM_ARTIFACT_OK")
PY

step "7. THE ORACLE ON THE VEHICLE: eager x2, compiled x2, fed OUR prompt ids"
run_leg() {   # run_leg <name> <eager 0|1> <prefill 0|1>
  local nm=$1 eg=$2 pf=$3
  echo "--- leg $nm eager=$eg prefill_check=$pf ---"
  ( cd /tmp && env -u HSA_OVERRIDE_GFX_VERSION \
      MODEL="$GGUF" TOK="$TOK" FEED_IDS="$JOBLOCAL/feed_ids.json" MMPROJ="$MMPROJ" \
      OUT_JSON="$JOBLOCAL/vllm-$nm.json" QUANT=gguf GMU=0.60 MAXLEN=2048 \
      EAGER="$eg" NTOK="$N_PREDICT" PREFILL_CHECK="$pf" \
      timeout --foreground 60m "$PY" "$W/gen_vehicle.py" ) \
      > "$JOBLOCAL/gen-$nm.out" 2>&1
  local rc=$?
  echo "GEN_RC[$nm]=$rc"
  cp -f "$JOBLOCAL/gen-$nm.out" "$OUT/gen-$nm.out" 2>/dev/null
  cp -f "$JOBLOCAL/vllm-$nm.json" "$OUT/vllm-$nm.json" 2>/dev/null
  grep -E '^(VLLM_VERSION|DEVICE|ON_GFX1151|PROMPT_IDS_MATCH_OURS|PREFILL_ARGMAX_TOTAL_MISMATCHES|PREFILL_AGREES_WITH_DECODE|DONE_MARKER)' \
    "$JOBLOCAL/gen-$nm.out" || tail -25 "$JOBLOCAL/gen-$nm.out"
  return $rc
}
run_leg eager    1 1; RC_E1=$?
run_leg eager-2  1 0; RC_E2=$?
run_leg compiled 0 1; RC_C1=$?
run_leg compiled-2 0 0; RC_C2=$?
echo "GEN_RCS eager=$RC_E1 eager2=$RC_E2 compiled=$RC_C1 compiled2=$RC_C2"

step "8. THE ORACLE'S OWN REPRODUCIBILITY on the vehicle"
python3 - "$JOBLOCAL" <<'PY'
import json, os, sys
j = sys.argv[1]
def ids(n):
    p = os.path.join(j, "vllm-%s.json" % n)
    if not (os.path.exists(p) and os.path.getsize(p)):
        return None
    return [r["gen_ids"] for r in json.load(open(p))["records"]]
e1, e2, c1, c2 = ids("eager"), ids("eager-2"), ids("compiled"), ids("compiled-2")
print("EAGER1_EQ_EAGER2       =", None if (e1 is None or e2 is None) else e1 == e2)
print("COMPILED1_EQ_COMPILED2 =", None if (c1 is None or c2 is None) else c1 == c2)
print("EAGER1_EQ_COMPILED1    =", None if (e1 is None or c1 is None) else e1 == c1)
if e1 and c1 and e1 != c1:
    for i, (a, b) in enumerate(zip(e1, c1)):
        for k, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print("  prompt %d: first diff at %d  eager=%d compiled=%d" % (i, k, x, y))
                break
PY

step "9. THE STRICT VERDICT"
python3 "$W/score_strict.py" "$JOBLOCAL/prompts.txt" "$OURS" \
  "$JOBLOCAL/vllm-eager.json" "$JOBLOCAL/vllm-compiled.json" "$N_PREDICT" \
  "$JOBLOCAL/vllm-eager-2.json" "$JOBLOCAL/vllm-compiled-2.json" \
  > "$JOBLOCAL/strict.txt" 2>&1
echo "score_rc=$?"
cp -f "$JOBLOCAL/strict.txt" "$OUT/strict.txt" 2>/dev/null
cat "$JOBLOCAL/strict.txt"

step "10. done"
cp -f "$JOBLOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== JOB DONE ==="
