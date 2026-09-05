#!/bin/bash
# Wave ARMTOKENS, run 2 (#2858) -- IS THE CHUNKED CPU ARM REACHED AT ALL?
#
# Run 1 answered the headline question and produced an AMBIGUITY it cannot
# resolve. All three CPU arms -- default, VT_CPU_QUANT_REPACK=0, and
# VT_GDN_CHUNKED=0 -- emitted the SAME eight ids:
#     11751 13 15767 411 2029 11 1092 369
# Two different worlds produce that reading and they are not the same finding:
#   (a) the chunked arm RUNS and its ~6e-5 perturbation flips none of the eight
#       argmaxes, or
#   (b) the chunked arm is NOT REACHED on this path -- q/k/v are not bf16, so
#       vt::GdnUseChunkedPrefill is false and BOTH settings run the SEQUENTIAL
#       recurrence.
# The spec's own T4 says a flag whose two arms coincide reads as a pass either
# way. NOTHING in run 1 separates them, and no log line names the arm.
#
# THE DISCRIMINATOR IS A COMMITTED INSTRUMENT, NOT A PATCH. VT_Q4EXP_LAYER_FP=3
# (qwen4_exp_forward.cpp:95, 15 taps) prints one line per tap per layer. Under
# (a) the two CPU arms must DIFFER at decoder layer 0's `blk` tap -- the Gated
# DeltaNet block output, the exact tap PREFILLDIV isolated. Under (b) they are
# bit-identical everywhere.
#
# AND THE DIFF NEEDS A POSITIVE CONTROL, because "no difference" is also what a
# broken diff prints. Arm H is CUDA, which PREFILLDIV measured at rel 3.525e-04
# on that same L00 blk tap against the CPU sequential arm. If H does not light
# up, the instrument or the diff is the thing that is broken and neither F-vs-G
# reading may be quoted.
#
# SAME BINARY AS RUN 1, BY DIGEST. Not a rebuild: run 1's server is on the share
# and its sha256 is asserted here, so these arms and run 1's arms are one image.
set -u
IN=/workspace/armtokens-2612
OUT=$IN/out2; mkdir -p "$OUT"
STAGE=/tmp/q4exp-UD-IQ1_S
S1F=Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
SHARD1_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
WANT_HEAD=b767ebda4e55122b1a5473b9aa4027da67f77b75
BIN_SHA_WANT=1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f
SERVER=/tmp/armtok-server
PORT=8173; MAXTOK=8
RUN1_CPU_SEQ="11751 13 15767 411 2029 11 1092 369"
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
rm -f "$OUT/results.txt"
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "=== A. ENV ==="
hostname; date -u; uname -m
res "HOST: $(hostname) arch=$(uname -m) cpus=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 | tail -1
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl binutils python3 >/dev/null 2>&1

say "=== B. THE BINARY IS RUN 1'S, BY DIGEST ==="
cp -L "$IN/vllm-server-$WANT_HEAD" "$SERVER" || { res "FATAL: cannot copy the run-1 server"; ls -la "$IN"; exit 20; }
chmod +x "$SERVER"
B=$(sha256sum "$SERVER" | cut -d' ' -f1)
res "BINARY sha256=$B (expect $BIN_SHA_WANT)"
[ "$B" = "$BIN_SHA_WANT" ] || { res "FATAL: this is NOT run 1's binary"; exit 21; }
"$SERVER" --help > "$OUT/help.txt" 2>&1
res "BINARY EXECS: rc=$? help_lines=$(wc -l < "$OUT/help.txt")"
I1=$(strings -a "$SERVER" | grep -c 'VT_Q4EXP_LAYER_FP')
I2=$(strings -a "$SERVER" | grep -c 'the chunked arm needs q/k/v in one dtype')
res "BINARY CARRIES: VT_Q4EXP_LAYER_FP=$I1(>=1) chunked_vtcheck=$I2(>=1)"
[ "$I1" -ge 1 ] && [ "$I2" -ge 1 ] || { res "FATAL: the image lacks the instrument or the chunked arm"; exit 22; }

say "=== C. THE ARTIFACT IS THE SAME ONE ==="
S1=$(sha256sum "$STAGE/$S1F" 2>/dev/null | cut -d' ' -f1)
[ "$S1" = "$SHARD1_SHA" ] || { res "FATAL: artifact absent or wrong: '$S1'"; exit 42; }
res "ARTIFACT VERIFIED: shard1 sha256=$S1 staged_bytes=$(du -sb "$STAGE" | cut -f1)"

run_arm(){ # $1 tag  $2 device  $3.. env
  local TAG=$1 DEVICE=$2; shift 2
  local AOUT=$OUT/$TAG; mkdir -p "$AOUT"
  say "=== ARM $TAG (device=$DEVICE env='$*') ==="
  res "ARM $TAG START device=$DEVICE env='$*'"
  local L0; L0=$(date +%s)
  env VT_Q4EXP_LAYER_FP=3 "$@" "$SERVER" --model "$STAGE/$S1F" \
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
  if [ $UP -eq 1 ]; then
    local REQ='{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":'$MAXTOK',"temperature":0,"logprobs":1}'
    local HTTP; HTTP=$(curl -s --max-time 5400 -o "$AOUT/c1.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/completions")
    res "ARM $TAG REQ http=$HTTP curl_rc=$?"
    local IDS; IDS=$(grep -o 'token_id:[0-9]*' "$AOUT/c1.json" | sed 's/token_id://' | tr '\n' ' '); IDS=$(echo $IDS)
    res "ARM $TAG TOKEN IDS: $IDS"
    echo "$IDS" > "$AOUT/ids.txt"
    res "ARM $TAG vs RUN1 CPU SEQ: $( [ "$IDS" = "$RUN1_CPU_SEQ" ] && echo IDENTICAL || echo DIFFERENT )"
  fi
  grep '^q4fp ' "$AOUT/server.log" > "$AOUT/fp.txt" 2>/dev/null
  # THE INSTRUMENT ASSERTS THAT IT RAN. PREFILLDIV counted 1314 lines and 437
  # taps per step on all three of its arms; an arm that printed none and an arm
  # whose taps agreed are indistinguishable in a diff and are not here.
  res "ARM $TAG FINGERPRINT LINES: $(wc -l < "$AOUT/fp.txt") (PREFILLDIV read 1314)"
  res "ARM $TAG TAPS END: $(grep 'taps=' "$AOUT/fp.txt" | tr '\n' ' ')"
  res "ARM $TAG steps completed: $(grep -c 'core-step end' "$AOUT/server.log")"
  res "ARM $TAG FIRST ERROR LINE: $(grep -m1 -inE 'engine-fatal|vt cuda:|illegal|terminate called|what\(\)' "$AOUT/server.log")"
  kill -9 $SRV 2>/dev/null; sleep 10
  res "ARM $TAG END"
}

# REPACK IS HELD AT 0 ON ALL THREE, so F-vs-G differs in VT_GDN_CHUNKED alone.
run_arm CPU-CHUNKED cpu  VT_CPU_QUANT_REPACK=0
run_arm CPU-SEQ     cpu  VT_CPU_QUANT_REPACK=0 VT_GDN_CHUNKED=0
run_arm CUDA-CTRL   cuda VT_CPU_QUANT_REPACK=0

say "=== D. DIFF THE TAPS ==="
cat > "$OUT/diff.py" <<'PYEOF'
import sys, os
def load(p):
    rows={}; order=[]
    if not os.path.exists(p): return rows, order
    for ln in open(p):
        if not ln.startswith('q4fp ') or ' taps=' in ln: continue
        f={}
        for tok in ln.split():
            if '=' in tok:
                k,v=tok.split('=',1); f[k]=v
        key=(f.get('step'), f.get('L'), f.get('tag'))
        if key in rows: continue
        rows[key]=f; order.append(key)
    return rows, order
def rel(a,b):
    a=float(a); b=float(b); m=max(abs(a),abs(b))
    return 0.0 if m==0.0 else abs(a-b)/m
base_p, other_p, label = sys.argv[1], sys.argv[2], sys.argv[3]
A,orderA=load(base_p); B,_=load(other_p)
print("=== %s : rows base=%d other=%d ===" % (label,len(A),len(B)))
if not A or not B:
    print("VOID: one side has NO fingerprint rows"); sys.exit(0)
nz=0; first=None; worst=(0.0,None)
print("%-5s %-4s %-9s %-6s %14s %14s %11s" % ("step","L","tag","dtype","sumabs_base","sumabs_other","rel_sumabs"))
for k in orderA:
    if k not in B:
        print("MISSING IN OTHER:",k); continue
    a=A[k]; b=B[k]; r=rel(a['sumabs'],b['sumabs'])
    if r>0.0:
        nz+=1
        if first is None: first=k
        if r>worst[0]: worst=(r,k)
    if k[0]=='0' and k[1] in (None,'+00','0') or (k[0]=='0'):
        print("%-5s %-4s %-9s %-6s %14s %14s %11.3e" % (k[0],k[1],k[2],a['dtype'],a['sumabs'],b['sumabs'],r))
print("--- %s SUMMARY ---" % label)
print("TAPS COMPARED                 :", len(orderA))
print("TAPS WITH rel(sumabs) != 0    :", nz)
print("FIRST TAP THAT DIFFERS        :", first)
print("LARGEST rel(sumabs)           : %.6e at %s" % (worst[0], worst[1]))
blk0=[k for k in orderA if k[0]=='0' and k[2]=='blk']
if blk0:
    k=blk0[0]
    print("L00 blk rel(sumabs)           : %.6e  (base=%s other=%s)" % (rel(A[k]['sumabs'],B[k]['sumabs']), A[k]['sumabs'], B[k]['sumabs']))
else:
    print("L00 blk rel(sumabs)           : NO SUCH TAP -- the instrument did not print it")
PYEOF
for pair in CPU-SEQ CUDA-CTRL; do
  python3 "$OUT/diff.py" "$OUT/CPU-CHUNKED/fp.txt" "$OUT/$pair/fp.txt" "CPU-CHUNKED vs $pair" > "$OUT/diff-$pair.txt" 2>&1
  say "--- DIFF CPU-CHUNKED vs $pair (step 0 taps) ---"; head -25 "$OUT/diff-$pair.txt"
  grep -E '^TAPS|^FIRST TAP|^LARGEST|^L00 blk|^VOID' "$OUT/diff-$pair.txt" | while read -r l; do res "DIFF vs $pair | $l"; done
done

say "=== E. THE VERDICT INPUTS ==="
res "SEQ CPU-CHUNKED: $(cat "$OUT/CPU-CHUNKED/ids.txt" 2>/dev/null)"
res "SEQ CPU-SEQ    : $(cat "$OUT/CPU-SEQ/ids.txt" 2>/dev/null)"
res "SEQ CUDA-CTRL  : $(cat "$OUT/CUDA-CTRL/ids.txt" 2>/dev/null)"
say "=== ALL RESULTS ==="; cat "$OUT/results.txt"
say "=== DONE ==="
