#!/bin/bash
# Wave MOESEL (row MODEL-MM-QWEN4-EXP, #2552) -- DOES THE TOP-K SELECTION FLIP
# AT A FORWARD WHERE THE ARMS ACTUALLY DISAGREE?
#
# THE OPEN CLAUSE. `qwen4_exp` emits
#     cpu   11751 13 15767 411 2029 11 1092 369
#     cuda  11751 13 15767 411 1928 11 628  567
# Five of the eight ids agree. The three that differ are the 5th, 7th and 8th,
# and with max_tokens=8 the k-th sampled id is produced by model forward k-1, so
# the disagreements are at forwards 4, 6 and 7.
#
# WHY NO INSTRUMENT HAS SEEN ONE YET. `VT_Q4EXP_LAYER_FP` returns early at
# `s.step >= s.budget` (qwen4_exp_forward.cpp:118) and its budget is counted in
# FORWARDS, so the `=3` every prior wave ran fingerprints forwards 0, 1 and 2 --
# three forwards whose sampled ids AGREE. Every reading this row has taken from
# those taps is from the agreeing region.
#
# `VT_MOE_SEL_FP` IS ON A DIFFERENT BUDGET AXIS, AND THAT IS THE MECHANISM.
# `MoeSelFpCall()` counts MoE BLOCK INVOCATIONS, not forwards (qwen3_5.cpp:7091).
# This model runs 48 MoE blocks per forward, measured -- not assumed -- from
# wave MOEDIV's own logs, whose 384 digests split 48 prefill (T=5) + 336 decode
# (T=1) = 8 forwards of 48. So a budget of 384 already spans forwards 0..7, and
# forward 4 begins at call 192. The budget here is 512, which is margin: `selfwd.py`
# DERIVES the blocks-per-forward from the data and refuses to quote a per-forward
# verdict if the structure is not exactly uniform and contiguous.
#
# WHY THIS IS THE SAME IMAGE AND NOT A REBUILD. Wave ARMTOKENS' server, sha256
# 1d129fa0..., is the binary that produced BOTH id sequences above (its out2/
# results.txt records them). It already carries `VT_MOE_SEL_FP`. So this job
# changes exactly one thing about that measurement -- an env var that turns on a
# readback -- and asserts the ids come back unchanged. A rebuild would have put a
# second difference between this reading and the divergence it is about.
#
# THE DIFF HAS A NEGATIVE CONTROL. A null result is a live outcome here, and
# "zero flips" is also what a broken comparison prints. Arm C re-runs arm A's
# exact configuration; A-vs-C must report ZERO flipped slots. If it does not,
# the arms are not deterministic and NO A-vs-B reading may be quoted.
#
# EXIT CODES ARE SEPARATED BY KIND, because an instrument whose failure looks
# like a result is this row's defining trap:
#   0 ok | 10,11,12 binary | 20 artifact | 30 an arm never served
#   40 instrument did not run | 41 structure not derivable
#   50 TOKEN DRIFT (an arm's ids are not the recorded ones)
#   51 DETERMINISM CONTROL FAILED (A vs C flipped something)
set -u
IN=/workspace/moesel-2552-fwd4
OUT=$IN/out; mkdir -p "$OUT"
SRC=/workspace/armtokens-2612
CKPT=/workspace/ckpt/qwen4exp-flash-next-iq1s
STAGE=/tmp/q4exp-UD-IQ1_S
S1F=Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
SHARD1_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
WANT_HEAD=b767ebda4e55122b1a5473b9aa4027da67f77b75
BIN_SHA_WANT=1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f
SERVER=/tmp/moesel-server
PORT=8177; MAXTOK=8; BUDGET=512
CTRL_CPU="11751 13 15767 411 2029 11 1092 369"
CTRL_CUDA="11751 13 15767 411 1928 11 628 567"

say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
sum(){ echo "SUM $*"; echo "SUM $*" >> "$OUT/results.txt"; }
rm -f "$OUT/results.txt" "$OUT/DONE"
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "=== A. ENV ==="
hostname; date -u; uname -m
res "HOST: $(hostname) arch=$(uname -m) cpus=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 | tail -1 | tee "$OUT/gpu.txt"
res "GPU: $(tail -1 "$OUT/gpu.txt")"
df -h /tmp | tail -1 > "$OUT/df-before.txt"; res "DF /tmp before: $(cat "$OUT/df-before.txt")"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl binutils python3 >/dev/null 2>&1
A_RC=0
for t in curl strings python3 sha256sum; do command -v $t >/dev/null || { res "MISSING TOOL: $t"; A_RC=1; }; done
sum "ENV_RC=$A_RC"
[ $A_RC -eq 0 ] || exit 10

say "=== B. THE BINARY IS WAVE ARMTOKENS', BY DIGEST ==="
cp -L "$SRC/vllm-server-$WANT_HEAD" "$SERVER"; CP_RC=$?
sum "BINCOPY_RC=$CP_RC"
[ $CP_RC -eq 0 ] || { res "FATAL: cannot copy the ARMTOKENS server"; ls -la "$SRC"; exit 10; }
chmod +x "$SERVER"
B=$(sha256sum "$SERVER" | cut -d' ' -f1)
res "BINARY sha256=$B (expect $BIN_SHA_WANT)"
if [ "$B" = "$BIN_SHA_WANT" ]; then BSHA_RC=0; else BSHA_RC=1; fi
sum "BINSHA_RC=$BSHA_RC"
[ $BSHA_RC -eq 0 ] || { res "FATAL: this is NOT the image that produced the divergence"; exit 11; }
"$SERVER" --help > "$OUT/help.txt" 2>&1; res "BINARY EXECS: rc=$? help_lines=$(wc -l < "$OUT/help.txt")"
# POSITIVE and NEGATIVE control on the grep shape itself: absence from a failed
# grep is not evidence, so a string known to be present and one known to be
# absent are counted in the SAME command shape.
I_SEL=$(strings -a "$SERVER" | grep -c 'VT_MOE_SEL_FP')
I_FMT=$(strings -a "$SERVER" | grep -c 'moesel call=')
I_POS=$(strings -a "$SERVER" | grep -c 'VT_Q4EXP_LAYER_FP')
I_NEG=$(strings -a "$SERVER" | grep -c 'VT_THIS_STRING_DOES_NOT_EXIST_XYZZY')
res "BINARY CARRIES: VT_MOE_SEL_FP=$I_SEL(>=1) moesel_fmt=$I_FMT(>=1) poscontrol=$I_POS(>=1) negcontrol=$I_NEG(==0)"
if [ "$I_SEL" -ge 1 ] && [ "$I_FMT" -ge 1 ] && [ "$I_POS" -ge 1 ] && [ "$I_NEG" -eq 0 ]; then BI_RC=0; else BI_RC=1; fi
sum "BININSTR_RC=$BI_RC"
[ $BI_RC -eq 0 ] || { res "FATAL: the image lacks the tap, or the grep shape is broken"; exit 12; }

say "=== C. THE ARTIFACT IS THE SAME ONE, STAGED WORKER-LOCAL ==="
S1=$(sha256sum "$STAGE/$S1F" 2>/dev/null | cut -d' ' -f1)
if [ "$S1" = "$SHARD1_SHA" ]; then
  res "ARTIFACT STAGE REUSED (no copy)"
  ST_RC=0
else
  res "ARTIFACT not staged (saw '$S1'); copying from $CKPT"
  T0=$(date +%s); mkdir -p "$STAGE"
  cp -rL "$CKPT"/*.gguf "$STAGE"/; ST_RC=$?
  res "ARTIFACT COPY rc=$ST_RC wall=$(( $(date +%s) - T0 ))s"
  S1=$(sha256sum "$STAGE/$S1F" 2>/dev/null | cut -d' ' -f1)
fi
res "ARTIFACT shard1 sha256=$S1 (expect $SHARD1_SHA)"
if [ "$S1" = "$SHARD1_SHA" ] && [ $ST_RC -eq 0 ]; then ART_RC=0; else ART_RC=1; fi
res "ARTIFACT shards=$(ls "$STAGE"/*.gguf 2>/dev/null | wc -l) staged_bytes=$(du -sb "$STAGE" 2>/dev/null | cut -f1)"
sum "ARTIFACT_RC=$ART_RC"
[ $ART_RC -eq 0 ] || { res "FATAL: artifact absent or wrong"; exit 20; }

DRIFT=0
run_arm(){ # $1 tag  $2 device  $3 expected-ids  $4.. env
  local TAG=$1 DEVICE=$2 WANT=$3; shift 3
  local AOUT=$OUT/$TAG; mkdir -p "$AOUT"
  say "=== ARM $TAG (device=$DEVICE env='$*') ==="
  res "ARM $TAG START device=$DEVICE env='$*' budget=$BUDGET"
  local L0; L0=$(date +%s)
  env VT_MOE_SEL_FP=$BUDGET "$@" "$SERVER" --model "$STAGE/$S1F" \
      --device "$DEVICE" --host 127.0.0.1 --port $PORT \
      --block-size 16 --num-blocks 128 --max-model-len 256 \
      --served-model-name qwen4exp --verbose > "$AOUT/server.log" 2>&1 & local SRV=$!
  local UP=0 i
  for i in $(seq 1 900); do
    kill -0 $SRV 2>/dev/null || { say "ARM $TAG: SERVER EXITED after $(( $(date +%s) - L0 ))s"; break; }
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { UP=1; break; }
    sleep 15
  done
  res "ARM $TAG HEALTH up=$UP load_wall=$(( $(date +%s) - L0 ))s"
  sum "ARM_${TAG}_UP=$UP"
  if [ $UP -ne 1 ]; then
    res "ARM $TAG FIRST ERROR LINE: $(grep -m1 -inE 'engine-fatal|vt cuda:|illegal|terminate called|what\(\)' "$AOUT/server.log")"
    kill -9 $SRV 2>/dev/null; sleep 10
    return 30
  fi
  local REQ='{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":'$MAXTOK',"temperature":0,"logprobs":1}'
  local R0; R0=$(date +%s)
  local HTTP; HTTP=$(curl -s --max-time 5400 -o "$AOUT/c1.json" -w '%{http_code}' \
      -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/completions")
  local CRC=$?
  res "ARM $TAG REQ http=$HTTP curl_rc=$CRC wall=$(( $(date +%s) - R0 ))s"
  local IDS; IDS=$(grep -o 'token_id:[0-9]*' "$AOUT/c1.json" | sed 's/token_id://' | tr '\n' ' '); IDS=$(echo $IDS)
  echo "$IDS" > "$AOUT/ids.txt"
  res "ARM $TAG TOKEN IDS: $IDS"
  res "ARM $TAG EXPECTED  : $WANT"
  if [ "$IDS" = "$WANT" ]; then
    res "ARM $TAG DRIFT: NONE (the tap did not perturb the sampled ids)"
    sum "ARM_${TAG}_DRIFT=0"
  else
    res "ARM $TAG DRIFT: YES -- this arm is NOT the recorded one"
    sum "ARM_${TAG}_DRIFT=1"
    DRIFT=1
  fi
  grep '^moesel ' "$AOUT/server.log" > "$AOUT/sel.txt" 2>/dev/null
  local NL; NL=$(wc -l < "$AOUT/sel.txt")
  res "ARM $TAG MOESEL LINES: $NL  digests=$(grep -c ' END$' "$AOUT/sel.txt")"
  sum "ARM_${TAG}_SELLINES=$NL"
  res "ARM $TAG LAST DIGEST: $(grep ' END$' "$AOUT/sel.txt" | tail -1)"
  res "ARM $TAG steps completed: $(grep -c 'core-step end' "$AOUT/server.log")"
  res "ARM $TAG FIRST ERROR LINE: $(grep -m1 -inE 'engine-fatal|vt cuda:|illegal|terminate called|what\(\)' "$AOUT/server.log")"
  kill -9 $SRV 2>/dev/null; sleep 10
  res "ARM $TAG END"
  return 0
}

# A and C are the SAME configuration. C is the determinism control for the diff.
run_arm A-CPU  cpu  "$CTRL_CPU"  VT_CPU_QUANT_REPACK=0; RA=$?
sum "ARM_A_RC=$RA"
run_arm B-CUDA cuda "$CTRL_CUDA" VT_CPU_QUANT_REPACK=0; RB=$?
sum "ARM_B_RC=$RB"
run_arm C-CPU2 cpu  "$CTRL_CPU"  VT_CPU_QUANT_REPACK=0; RC2=$?
sum "ARM_C_RC=$RC2"
if [ $RA -ne 0 ] || [ $RB -ne 0 ] || [ $RC2 -ne 0 ]; then
  res "FATAL: an arm never served"; sum "ARMS_RC=30"; exit 30
fi
sum "ARMS_RC=0"
sum "TOKENDRIFT_RC=$DRIFT"

say "=== D. THE NEGATIVE CONTROL: A vs C MUST FLIP NOTHING ==="
python3 "$IN/selfwd.py" "$OUT/A-CPU/sel.txt" "$OUT/C-CPU2/sel.txt" "CONTROL-A-vs-C" \
    "$OUT/ctrl.json" > "$OUT/cmp-control.txt" 2>&1
CTRL_RC=$?
sum "CONTROL_CMP_RC=$CTRL_RC"
grep '^RESULT ' "$OUT/cmp-control.txt" >> "$OUT/results.txt"
sed -n '1,40p' "$OUT/cmp-control.txt"
CTRL_FLIPS=$(grep -o 'total_flipped_slots=[0-9]*' "$OUT/cmp-control.txt" | tail -1 | cut -d= -f2)
CTRL_FLIPS=${CTRL_FLIPS:-UNKNOWN}
res "CONTROL A-vs-C total_flipped_slots=$CTRL_FLIPS (MUST be 0)"
sum "CONTROL_FLIPS=$CTRL_FLIPS"

say "=== E. THE MEASUREMENT: A vs B, GROUPED BY FORWARD ==="
python3 "$IN/selfwd.py" "$OUT/A-CPU/sel.txt" "$OUT/B-CUDA/sel.txt" "MEASURE-CPU-vs-CUDA" \
    "$OUT/measure.json" > "$OUT/cmp-measure.txt" 2>&1
MEAS_RC=$?
sum "MEASURE_CMP_RC=$MEAS_RC"
grep '^RESULT ' "$OUT/cmp-measure.txt" >> "$OUT/results.txt"
cat "$OUT/cmp-measure.txt"

say "=== E2. THE PAIR IS THE BRACKETED ONE, ASSERTED ON A SECOND INSTRUMENT ==="
# The MoE tap's `x` axis is the SAME tensor as the `VT_Q4EXP_LAYER_FP` tap's
# `L00 mhc.mix`: MOEDIV's first digest reads x=3613.82031 and PREFILLDIV's
# CPU-CTRL `L00 mhc.mix` reads 3613.82031. ARMTOKENS run 2 recorded the matched
# pair as 3615.47142 (CPU-CHUNKED) vs 3615.62777 (CUDA-CTRL), whose rel is the
# 4.324e-05 that sits inside #2552's bracket. So these two numbers coming back
# unchanged is what says THIS run is that pair and not a neighbouring one.
X_CPU_WANT=3615.47142
X_CUDA_WANT=3615.62777
X_CPU=$(grep -m1 ' END$' "$OUT/A-CPU/sel.txt" | tr ' ' '\n' | grep '^x=' | cut -d= -f2)
X_CUDA=$(grep -m1 ' END$' "$OUT/B-CUDA/sel.txt" | tr ' ' '\n' | grep '^x=' | cut -d= -f2)
res "CALL0 x A-CPU =$X_CPU (recorded L00 mhc.mix CPU-CHUNKED = $X_CPU_WANT)"
res "CALL0 x B-CUDA=$X_CUDA (recorded L00 mhc.mix CUDA-CTRL  = $X_CUDA_WANT)"
if [ "$X_CPU" = "$X_CPU_WANT" ] && [ "$X_CUDA" = "$X_CUDA_WANT" ]; then XM=0; else XM=1; DRIFT=1; fi
res "PAIR IDENTITY: $( [ $XM -eq 0 ] && echo 'CONFIRMED -- this is the 4.324e-05 matched pair' || echo 'NOT CONFIRMED -- the layer-0 input is not the recorded one' )"
sum "PAIRIDENT_RC=$XM"

say "=== F. VERDICT INPUTS ==="
res "IDS A-CPU : $(cat "$OUT/A-CPU/ids.txt" 2>/dev/null)"
res "IDS B-CUDA: $(cat "$OUT/B-CUDA/ids.txt" 2>/dev/null)"
res "IDS C-CPU2: $(cat "$OUT/C-CPU2/ids.txt" 2>/dev/null)"
df -h /tmp | tail -1 > "$OUT/df-after.txt"; res "DF /tmp after: $(cat "$OUT/df-after.txt")"

# The ordered gate. Drift and instrument failure are DIFFERENT exits.
sum "DRIFT_RC=$DRIFT"
FINAL=0
if [ "$MEAS_RC" -eq 40 ] || [ "$CTRL_RC" -eq 40 ]; then FINAL=40
elif [ "$MEAS_RC" -eq 41 ] || [ "$CTRL_RC" -eq 41 ]; then FINAL=41
elif [ "$DRIFT" -ne 0 ]; then FINAL=50
elif [ "$CTRL_FLIPS" != "0" ]; then FINAL=51
fi
sum "FINAL_RC=$FINAL"
say "=== ALL RESULTS ==="; cat "$OUT/results.txt"
if [ $FINAL -eq 0 ]; then echo "MOESEL-DONE-OK $(date -u +%FT%TZ)" > "$OUT/DONE"
else echo "MOESEL-DONE-FAIL rc=$FINAL $(date -u +%FT%TZ)" > "$OUT/DONE"; fi
say "=== DONE marker: $(cat "$OUT/DONE") ==="
exit $FINAL
