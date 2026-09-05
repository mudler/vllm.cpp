#!/bin/bash
# Wave ARMTOKENS (#2612, row KERNEL-GDN-CHUNKED-MIRROR) -- WHAT does each arm
# emit now that BOTH run the chunked decomposition?
#
# #2849 landed a chunked CPU GDN prefill mirroring vLLM's bf16 intermediate
# placement and made it the DEFAULT for bf16. Before it the CPU arm ran the
# sequential recurrence and the CUDA arm the chunked one -- two ALGORITHMS --
# and the released UD-IQ1_S artifact emitted
#   CPU  11751 13 15767 411 2029 11 1092 369
#   CUDA 11751 13 15767 411 1928 11  628 567   (5 of 8)
# The port wave DEFERRED re-deriving these because it needed the artifact and a
# lease. NOBODY HAS MEASURED EITHER ARM UNDER THE NEW DEFAULT. This job does.
#
# NOTHING HERE PREDICTS AN ANSWER. The spec says explicitly that agreement is an
# argmax over near-ties and is not monotone in residual. The arms report ids.
#
# ARMS, ordered so a mid-sequence box crash (#545) costs the headline least:
#   A  CPU-DEFAULT   --device cpu,  no env.            THE HEADLINE (production)
#   B  CUDA-DEFAULT  --device cuda, no env.            THE HEADLINE (production)
#   C  CPU-GDNSEQ    --device cpu,  VT_GDN_CHUNKED=0 VT_CPU_QUANT_REPACK=0
#                    THE POSITIVE CONTROL. This is the exact configuration and
#                    algorithm that produced the 2029/1092/369 sequence. If it
#                    does NOT reproduce it, this harness is not measuring the
#                    thing the prior run measured and NO arm here may be quoted.
#   D  CPU-REPACK0   --device cpu,  VT_CPU_QUANT_REPACK=0.  The repack check: a
#                    dropped repack marker once put a NaN in layer 0 on this
#                    aarch64 box. =0 must equal A byte for byte.
#   E  CUDA-REPACK0  --device cuda, VT_CPU_QUANT_REPACK=0.  Reproduces the prior
#                    CUDA run's env exactly, so B-vs-E separates "the default
#                    moved" from "the env differed".
#
# EVERY STEP ASSERTS THAT IT RAN. An instrument whose failure looks like a result
# is this row's recurring trap. A degenerate all-zero logit row yields token id 0
# eight times and has ALREADY once read as a real answer here.
set -u
IN=/workspace/armtokens-2612
OUT=$IN/out; mkdir -p "$OUT"
STAGE=/tmp/q4exp-UD-IQ1_S
S1F=Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
SHARD1_SHA=88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd
TAR=$IN/src.tar.gz
TAR_SHA=9a4a2934652d4cc9cfe63511eb8fd043f945f4f977bd1ca74412277f569a45d1
WANT_HEAD=b767ebda4e55122b1a5473b9aa4027da67f77b75
SRC=/tmp/armtok-src; BLD=/tmp/armtok-bld
PORT=8171; MAXTOK=8
PRIOR_CPU_SEQ="11751 13 15767 411 2029 11 1092 369"
PRIOR_CUDA_SEQ="11751 13 15767 411 1928 11 628 567"
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
res(){ echo "RESULT $*"; echo "RESULT $*" >> "$OUT/results.txt"; }
rm -f "$OUT/results.txt"
# SELF-TERMINATING heartbeat: one that outlives the script holds the job's stdout
# pipe open and the lease with it.
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "=== A. ENV ==="
hostname; date -u; uname -m; nproc; free -g | head -2
res "HOST: $(hostname) arch=$(uname -m) cpus=$(nproc)"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 | tee "$OUT/gpu.txt"
res "GPU: $(sed -n 2p "$OUT/gpu.txt")"
df -h /tmp /workspace 2>&1 | tee "$OUT/df-before.txt"
res "DF /tmp before: $(df -h /tmp | tail -1 | tr -s ' ')"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq wget ca-certificates gnupg ccache cmake ninja-build git curl binutils python3 >/dev/null 2>&1
if ! command -v nvcc >/dev/null; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
  dpkg -i /tmp/ck.deb >/dev/null 2>&1; apt-get update -qq
  apt-get install -y -qq cuda-toolkit-13-0 >/dev/null 2>&1
fi
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CCACHE_DIR=/workspace/ccache; mkdir -p "$CCACHE_DIR"; export CCACHE_MAXSIZE=40G
command -v nvcc >/dev/null || { res "FATAL: no nvcc"; exit 90; }
res "NVCC: $(nvcc --version | tail -2 | tr '\n' ' ')"
# The whole repack chain is gated on vt::cpu::QuantRepackActive(), which is false
# off aarch64 i8mm. Arm D is only a discriminator where the chain is LIVE.
ISA=$(grep -m1 -o 'i8mm' /proc/cpuinfo || echo ABSENT)
res "HOST ISA i8mm: $ISA (repack chain live iff PRESENT)"

say "=== B. UNPACK, AND ASSERT THIS IS THE TREE THE QUESTION NAMES ==="
T=$(sha256sum "$TAR" | cut -d' ' -f1)
res "SOURCE TARBALL sha256=$T (expect $TAR_SHA)"
[ "$T" = "$TAR_SHA" ] || { res "FATAL: tarball is not the staged one"; exit 11; }
H=$(cat "$IN/HEAD_SHA")
res "SOURCE HEAD: $H (expect $WANT_HEAD)"
[ "$H" = "$WANT_HEAD" ] || { res "FATAL: HEAD_SHA mismatch"; exit 12; }
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$TAR" -C "$SRC" || { res "FATAL: untar"; exit 14; }
test -f "$SRC/CMakeLists.txt" || { res "FATAL: untar produced no CMakeLists.txt"; exit 92; }
# The three source properties that make this tree the post-#2849 one. A binary
# built from a tree LACKING them cannot answer the question, and a build that
# succeeded is not evidence that it carries them.
CPUO=$SRC/src/vt/cpu/cpu_ops.cpp
OPSC=$SRC/src/vt/ops.cpp
P1=$(grep -c 'GdnChunkedHeadPrefill' "$CPUO")                 # expect 2 (def + call)
P2=$(grep -c 'GdnUseChunkedPrefill' "$CPUO")                  # expect >=1 (CPU routes on it)
P3=$(grep -c 'q_dtype == DType::kBF16' "$OPSC")               # expect 1 (D0 predicate)
# P4: #2861's repair. main at d2b1bda2b FAILED TO COMPILE here (nvcc 177-D on a
# dead static ChunkedPrefillEnabled), so run 1 produced no arm at all. The dead
# function must be ABSENT, or this tarball is the one that cannot build.
P4=$(grep -c 'bool ChunkedPrefillEnabled()' "$SRC/src/vt/cuda/cuda_gdn.cu")
res "CHUNKED CPU ARM IN SOURCE: GdnChunkedHeadPrefill=$P1(2) GdnUseChunkedPrefill_in_cpu=$P2(>=1) bf16_predicate=$P3(1) dead_cuda_wrapper=$P4(0)"
[ "$P1" -eq 2 ] && [ "$P2" -ge 1 ] && [ "$P3" -eq 1 ] && [ "$P4" -eq 0 ] || {
  res "FATAL: this source does not carry the chunked CPU arm, or still carries #2861. REFUSING to measure it."; exit 19; }

say "=== C. STAGE THE ARTIFACT ==="
S1=$(sha256sum "$STAGE/$S1F" 2>/dev/null | cut -d' ' -f1)
if [ "$S1" != "$SHARD1_SHA" ]; then
  say "stage absent or wrong -- copying from the share (SLOW PATH, ~2700s)"
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  SRCART=/workspace/ckpt/qwen4exp-flash-next-iq1s
  [ -f "$SRCART/$S1F" ] || SRCART=/workspace/q4exp-bench/UD-IQ1_S
  echo "copying from $SRCART"
  cp -rL "$SRCART"/. "$STAGE"/ || { res "FATAL: cannot stage from $SRCART"; ls /workspace; exit 41; }
  S1=$(sha256sum "$STAGE/$S1F" 2>/dev/null | cut -d' ' -f1)
  res "ARTIFACT STAGED FRESH from $SRCART"
else
  res "ARTIFACT STAGE REUSED (no copy)"
fi
[ "$S1" = "$SHARD1_SHA" ] || { res "FATAL: artifact sha mismatch: got '$S1'"; exit 42; }
BYTES=$(du -sb "$STAGE" | cut -f1)
res "ARTIFACT VERIFIED: shard1 sha256=$S1 staged_bytes=$BYTES shards=$(ls "$STAGE"/*.gguf | wc -l)"
ls -la "$STAGE" | tee "$OUT/artifact-ls.txt"

say "=== D. BUILD sm_110, CUDA ON ==="
t0=$(date +%s)
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES=110 -DVLLM_CPP_TRITON=OFF \
  -DVLLM_CPP_METAL=OFF -DVLLM_CPP_VULKAN=OFF -DVLLM_CPP_HIP=OFF \
  -DVLLM_CPP_TENSTORRENT=OFF -DVLLM_CPP_BUILD_TESTS=OFF \
  -DVLLM_CPP_BUILD_EXAMPLES=ON -DVLLM_CPP_HF_DOWNLOAD=OFF > "$OUT/cmake.log" 2>&1
crc=$?; res "CMAKE rc=$crc"
[ $crc -eq 0 ] || { tail -40 "$OUT/cmake.log"; exit 15; }
ninja -C "$BLD" -j 4 > "$OUT/build.log" 2>&1
brc=$?; res "BUILD rc=$brc wall=$(( $(date +%s) - t0 ))s objects=$(grep -c 'Building ' "$OUT/build.log")"
[ $brc -eq 0 ] || { tail -60 "$OUT/build.log"; exit 16; }
# ninja links examples/vllm-server, NOT $BLD/vllm-server. A predecessor job on
# this row measured NOTHING at rc 0 because it looked in the wrong place.
SERVER=$(find "$BLD" -name 'vllm-server' -type f -perm -u+x | head -1)
[ -x "${SERVER:-}" ] || { res "FATAL: no server binary after a rc=0 build"
  find "$BLD" -maxdepth 3 -type f -executable | head -30; exit 17; }
BIN_SHA=$(sha256sum "$SERVER" | cut -d' ' -f1)
res "BINARY: $SERVER sha256=$BIN_SHA built=$(date -u -r "$SERVER" +%FT%TZ)"
"$SERVER" --help > "$OUT/help.txt" 2>&1
res "BINARY EXECS: rc=$? help_lines=$(wc -l < "$OUT/help.txt")"
# BINARY-LEVEL proof, independent of the source grep. This VT_CHECK message is
# emitted ONLY from inside the chunked branch of GdnPrefillKernel, so its
# presence in the linked image proves the chunked CPU arm was compiled in.
B1=$(strings -a "$SERVER" 2>/dev/null | grep -c 'the chunked arm needs q/k/v in one dtype')
B2=$(strings -a "$SERVER" 2>/dev/null | grep -c 'VT_GDN_CHUNKED')
res "BINARY CARRIES THE CHUNKED CPU ARM: chunked_vtcheck_literal=$B1(>=1) VT_GDN_CHUNKED=$B2(>=1)"
[ "$B1" -ge 1 ] && [ "$B2" -ge 1 ] || { res "FATAL: linked image lacks the chunked arm's own strings"; exit 18; }
cp "$SERVER" "$IN/vllm-server-$WANT_HEAD" 2>/dev/null

run_arm(){ # $1 tag  $2 device  $3.. env (may be empty)
  local TAG=$1 DEVICE=$2; shift 2
  local AOUT=$OUT/$TAG; mkdir -p "$AOUT"
  say "=== ARM $TAG (device=$DEVICE env='$*') ==="
  res "ARM $TAG START bin_sha=$BIN_SHA device=$DEVICE env='$*'"
  local L0; L0=$(date +%s)
  env "$@" "$SERVER" --model "$STAGE/$S1F" \
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
  local IDS=""
  if [ $UP -eq 1 ]; then
    local RT0; RT0=$(date +%s)
    # logprobs:1 is how the prior run READ the ids (the "token_id:N" fallback
    # string). Greedy at temperature 0; keeping it keeps the conditions equal.
    local REQ='{"model":"qwen4exp","prompt":"The capital of France is","max_tokens":'$MAXTOK',"temperature":0,"logprobs":1}'
    local HTTP CRC
    HTTP=$(curl -s --max-time 5400 -o "$AOUT/c1.json" -w '%{http_code}' \
        -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/completions")
    CRC=$?
    res "ARM $TAG REQ http=$HTTP curl_rc=$CRC wall=$(( $(date +%s) - RT0 ))s"
    IDS=$(grep -o 'token_id:[0-9]*' "$AOUT/c1.json" | sed 's/token_id://' | tr '\n' ' '); IDS=$(echo $IDS)
    local NID; NID=$(echo "$IDS" | wc -w)
    res "ARM $TAG TOKEN IDS: $IDS"
    res "ARM $TAG N IDS: $NID (expect $MAXTOK)"
    res "ARM $TAG TEXT: $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["choices"][0]["text"])' "$AOUT/c1.json" 2>&1 | head -1)"
    echo "$IDS" > "$AOUT/ids.txt"
    # DEGENERACY REFUSAL. All-zero logits give id 0 eight times and that once
    # read as a real answer on this row. Flag it loudly; do not silently pass.
    if [ "$IDS" = "0 0 0 0 0 0 0 0" ]; then res "ARM $TAG DEGENERATE: ALL IDS ARE 0 -- NOT A RESULT"; fi
    if [ "$NID" -ne "$MAXTOK" ]; then res "ARM $TAG SHORT: got $NID ids, not $MAXTOK -- NOT A RESULT"; fi
    res "ARM $TAG vs PRIOR CPU SEQ: $( [ "$IDS" = "$PRIOR_CPU_SEQ" ] && echo IDENTICAL || echo DIFFERENT )"
    res "ARM $TAG vs PRIOR CUDA SEQ: $( [ "$IDS" = "$PRIOR_CUDA_SEQ" ] && echo IDENTICAL || echo DIFFERENT )"
  fi
  res "ARM $TAG steps completed: $(grep -c 'core-step end' "$AOUT/server.log")"
  res "ARM $TAG FIRST ERROR LINE: $(grep -m1 -inE 'engine-fatal|vt cuda:|illegal|terminate called|what\(\)|nan' "$AOUT/server.log")"
  # Any line the server itself prints about the resolved dtype, captured rather
  # than inferred. Reported whatever it says, including nothing.
  grep -inE 'dtype|bf16|bfloat' "$AOUT/server.log" | head -8 > "$AOUT/dtype.txt"
  res "ARM $TAG DTYPE LINES: $(wc -l < "$AOUT/dtype.txt")"
  sed -n '1,8p' "$AOUT/dtype.txt" | while read -r l; do echo "    dtype| $l"; done
  echo "--- $TAG server.log tail 20 ---"; tail -20 "$AOUT/server.log"
  kill -9 $SRV 2>/dev/null; sleep 10
  res "ARM $TAG END"
}

run_arm CPU-DEFAULT  cpu
run_arm CUDA-DEFAULT cuda
run_arm CPU-GDNSEQ   cpu  VT_GDN_CHUNKED=0 VT_CPU_QUANT_REPACK=0
run_arm CPU-REPACK0  cpu  VT_CPU_QUANT_REPACK=0
run_arm CUDA-REPACK0 cuda VT_CPU_QUANT_REPACK=0

say "=== E. THE COMPARISON ==="
A=$(cat "$OUT/CPU-DEFAULT/ids.txt" 2>/dev/null || echo MISSING)
B=$(cat "$OUT/CUDA-DEFAULT/ids.txt" 2>/dev/null || echo MISSING)
C=$(cat "$OUT/CPU-GDNSEQ/ids.txt" 2>/dev/null || echo MISSING)
D=$(cat "$OUT/CPU-REPACK0/ids.txt" 2>/dev/null || echo MISSING)
E=$(cat "$OUT/CUDA-REPACK0/ids.txt" 2>/dev/null || echo MISSING)
res "SEQ CPU-DEFAULT : $A"
res "SEQ CUDA-DEFAULT: $B"
res "SEQ CPU-GDNSEQ  : $C"
res "SEQ CPU-REPACK0 : $D"
res "SEQ CUDA-REPACK0: $E"
python3 - "$A" "$B" "$C" "$D" "$E" "$PRIOR_CPU_SEQ" "$PRIOR_CUDA_SEQ" <<'PYEOF' | tee -a "$OUT/results.txt"
import sys
A,B,C,D,E,PC,PU = sys.argv[1:8]
def sp(s): return s.split()
def agree(x,y):
    x,y=sp(x),sp(y)
    if len(x)!=8 or len(y)!=8: return "VOID (an arm has %d/%d ids, not 8)"%(len(x),len(y))
    idx=[i for i in range(8) if x[i]==y[i]]
    return "%d of 8 (indices %s)"%(len(idx), ",".join(map(str,idx)) if idx else "none")
print("RESULT AGREEMENT CPU-DEFAULT vs CUDA-DEFAULT : %s" % agree(A,B))
print("RESULT AGREEMENT CPU-DEFAULT vs PRIOR CPU SEQ: %s" % agree(A,PC))
print("RESULT AGREEMENT CUDA-DEFAULT vs PRIOR CUDA  : %s" % agree(B,PU))
print("RESULT CONTROL CPU-GDNSEQ REPRODUCES PRIOR CPU SEQ: %s" % ("YES" if sp(C)==sp(PC) else "NO -- THE HARNESS IS NOT MEASURING THE PRIOR THING; NO ARM MAY BE QUOTED"))
print("RESULT REPACK CHECK CPU-DEFAULT == CPU-REPACK0: %s" % ("YES (identical)" if sp(A)==sp(D) else "NO -- %s vs %s" % (A,D)))
print("RESULT FLAG ROUTES ON CPU (default != VT_GDN_CHUNKED=0): %s" % ("YES" if sp(A)!=sp(C) else "NO -- default and =0 coincide; either the flag is read and ignored or q/k/v are not bf16"))
print("RESULT CUDA env sensitivity (DEFAULT == REPACK0): %s" % ("YES (identical)" if sp(B)==sp(E) else "NO -- %s vs %s" % (B,E)))
PYEOF

say "=== DF AFTER ==="; df -h /tmp /workspace 2>&1 | tee "$OUT/df-after.txt"
say "=== ALL RESULTS ==="; cat "$OUT/results.txt"
say "=== DONE ==="
