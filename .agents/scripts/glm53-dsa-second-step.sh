#!/usr/bin/env bash
# MODEL-TEXT-GLM-MOE-DSA / ENG-EXPERT-STREAM-DEVICE -- drive GLM-5.3 (`glm-dsa`,
# NON-FLASH, GlmMoeDsaForCausalLM) on `dgx:gpu0` against the real 201.83 GiB
# UD-IQ1_S artifact WITH EXPERT STREAMING, for FOUR tokens.
#
# WHAT IS ALREADY KNOWN AND IS NOT THIS JOB'S QUESTION. A ONE-token streamed
# generation already landed on main (`b5d82659c`, 2026-08-31): stdout ` Paris`,
# `[expert-stream] ON slots=4096`, `steps=1 hits=0 misses=6399 fills=4096
# exhausted=2303`. That run emitted its token out of PREFILL and stopped. The
# lane was therefore never asked for a SECOND step, and every counter that only
# a second step can move -- `hits`, `evictions`, and the step guard's hotness
# clock -- reads zero in the landed evidence.
#
# THE QUESTION THIS JOB ASKS: does multi-token DECODE run at all on this model
# with the streaming lane on, and do the lane's counters show slot REUSE once a
# second step exists?
#
# NOT `set -e`. A refusal is a RESULT here. Every step captures its own rc, and
# never `$?` after a pipe.
set -u

W=/workspace/glm53-stream4
CKPT=${CKPT:-/workspace/ckpt/GLM-5.3-UD-IQ1_S}
DERIVED=${DERIVED:-/workspace/glm53-firstload/derived}
SHARD1=$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf
SRC=/tmp/glm53s4/src
BUILD=/tmp/glm53s4/build-cuda
CUT=/tmp/glm53s4/cutlass
BOX=$(hostname)
PROMPT=${PROMPT:-The capital of France is}
MAX_TOKENS=${MAX_TOKENS:-4}
SLOTS=${SLOTS:-8192}

SRCSHA=$(sha256sum "$W/src.tar.gz" 2>/dev/null | cut -c1-12)
RECIPE=s4r1
OUT=$W/out/$BOX-${SRCSHA:-nosrc}-$RECIPE
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

say(){ printf '\n\n========== %s ==========  %s\n' "$1" "$(date -u +%FT%TZ)"; }
note(){ echo "### RC $1=$2"; }
stamp(){ [ -f "$OUT/stamp.$1" ]; }
mark(){ : > "$OUT/stamp.$1"; }

say "IDENTITY -- WHICH BOX EVERY NUMBER BELOW CAME FROM"
hostname; id -un; uname -m
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv 2>&1 | head -4
echo "cores: $(grep -c ^processor /proc/cpuinfo)"
free -m 2>&1 | head -2
echo "OUT=$OUT  (keyed on src.tar.gz $SRCSHA, recipe $RECIPE)"

say "WORKSPACE MOUNT GUARD -- a failed CIFS mount is an EMPTY LOCAL DIR and this job would 'succeed' writing nowhere"
if [ ! -s "$W/SENTINEL" ]; then echo "FATAL: /workspace is not the NAS -- no $W/SENTINEL"; exit 90; fi
cat "$W/SENTINEL"
df -h /workspace 2>&1 | tail -2

say "THE ARTIFACT -- asserted present and NON-EMPTY before a lease minute is spent on a build"
for n in 1 2 3 4 5 6; do
  f=$DERIVED/GLM-5.3-UD-IQ1_S-0000$n-of-00006.gguf
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  echo "shard $n: $sz bytes  $f"
  [ "$sz" -gt 0 ] || { echo "FATAL: shard $n missing or empty"; exit 91; }
done
echo "### DERIVED shard 1 sha256 (the indexer-schedule repair, NOT the published shard 1):"
sha256sum "$SHARD1"

say "TOOLCHAIN"
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg git cmake ninja-build binutils >/dev/null 2>&1
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-13-0
fi
export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; exit 90; }
nvcc --version | tail -2

say "SOURCE"
[ -s "$W/src.tar.gz" ] || { echo "FATAL: no $W/src.tar.gz"; exit 92; }
echo "src.tar.gz sha256: $(sha256sum "$W/src.tar.gz" | cut -d' ' -f1)"
echo "EXPECTED:          $(cat "$W/src.tar.gz.sha256" 2>/dev/null)"
echo "BASE_SHA:          $(cat "$W/BASE_SHA" 2>/dev/null)"
if ! stamp src; then
  rm -rf "$SRC" "$BUILD"; mkdir -p "$SRC"
  tar -xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: extract failed"; exit 92; }
  test -f "$SRC/CMakeLists.txt" || { echo "FATAL: no CMakeLists.txt at the archive root"; exit 92; }
  find "$SRC" -exec touch {} + 2>/dev/null
  mark src
fi

say "CUDA ARCH -- READ off the device, never assumed"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
case "$CC" in
  121) ARCH=121a ;;
  110) ARCH=110a ;;
  "")  echo "FATAL: compute_cap unreadable -- refusing to guess an arch"; exit 93 ;;
  *)   ARCH=$CC ;;
esac
echo "### DEVICE compute_cap=$CC -> CUDA arch $ARCH"

say "CUTLASS -- FlashAttention-2 needs its headers, and without them it compiles for NO ARCH"
if [ ! -f "$CUT/include/cutlass/cutlass.h" ]; then
  rm -rf "$CUT"; mkdir -p "$CUT"
  [ -s /workspace/cutlass-v4.5.0.tar.gz ] || { echo "FATAL: no CUTLASS tarball on the share"; exit 93; }
  tar -xzf /workspace/cutlass-v4.5.0.tar.gz -C "$CUT" || { echo "FATAL: cutlass untar"; exit 93; }
fi
grep -aE '#define CUTLASS_(MAJOR|MINOR|PATCH)' "$CUT/include/cutlass/version.h" 2>/dev/null \
  || { echo "FATAL: no CUTLASS version header under $CUT"; exit 93; }

say "CONFIGURE -- -DVLLM_CPP_FLASH_ATTN=ON is PASSED, not read back from a cache"
if ! stamp cfg; then
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
        -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_CUTLASS_DIR="$CUT" \
        -DVLLM_CPP_FLASH_ATTN=ON > "$OUT/cmake.log" 2>&1
  rc=$?; note CONFIGURE $rc
  tail -12 "$OUT/cmake.log"
  [ "$rc" -ne 0 ] && { echo "FATAL: configure failed"; exit 93; }
  mark cfg
fi
grep -aE '^(VLLM_CPP_CUDA|VLLM_CPP_CUDA_ARCHITECTURES|VLLM_CPP_TRITON|VLLM_CPP_FLASH_ATTN|VLLM_CPP_CUTLASS_DIR|CMAKE_BUILD_TYPE):' "$BUILD/CMakeCache.txt"

# THE COMPILED FEATURE SET, ASSERTED BEFORE ANY RESULT IS BELIEVED. MLA prefill
# on sm_121 IS FlashAttention-2 and the upstream selector has no fallback below
# it (mla/prefill/selector.py:191-194), so an empty manifest guarantees a throw
# at the first forward -- AFTER a ~450 s load. Reading it here spends 0 s
# instead. An empty manifest is not an error line, which is why it is read out
# loud rather than discovered at the throw.
FA2LINE=$(grep -a 'FA2 compiled-arch manifest' "$OUT/cmake.log" | tail -1)
case "$FA2LINE" in
  *"manifest: []"*|"")
    echo "FATAL: FA2 is NOT compiled in (manifest line: '${FA2LINE:-ABSENT}')."
    echo "  MLA prefill on $ARCH IS FlashAttention and has no fallback below it."
    exit 94 ;;
esac
echo "### $FA2LINE"

say "BUILD vllm-cli -- -j 4, because unconstrained parallelism has OOM-REBOOTED this box"
# Check the ARTIFACT, not the stamp: a re-run landing on a different worker
# inherits 'already built' and an empty /tmp.
if stamp build && [ -z "$(find "$BUILD" -maxdepth 3 -name vllm-cli -type f -perm -u+x 2>/dev/null | head -1)" ]; then
  echo "### stamped as built, but no vllm-cli under $BUILD -- a different worker. Rebuilding."
  rm -f "$OUT"/stamp.src "$OUT"/stamp.cfg "$OUT"/stamp.build
  rm -rf "$SRC" "$BUILD"; mkdir -p "$SRC"
  tar -xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: extract failed"; exit 92; }
  find "$SRC" -exec touch {} + 2>/dev/null
  mark src
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DVLLM_CPP_CUDA=ON -DVLLM_CPP_CUDA_ARCHITECTURES="$ARCH" \
        -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_CUTLASS_DIR="$CUT" \
        -DVLLM_CPP_FLASH_ATTN=ON > "$OUT/cmake.log" 2>&1
  rc=$?; note RECONFIGURE $rc
  [ "$rc" -ne 0 ] && { tail -20 "$OUT/cmake.log"; echo "FATAL: reconfigure failed"; exit 93; }
  mark cfg
fi
if ! stamp build; then
  # A HEARTBEAT, because the compile prints nothing to stdout for up to an hour
  # and an idle-timeout cannot tell a working build from a hung one.
  ( while :; do sleep 120
      printf '  [build heartbeat] %s  %s\n' "$(date -u +%TZ)" \
        "$(tail -1 "$OUT/build.log" 2>/dev/null | tr -d '\r' | cut -c1-110)"
    done ) & hb=$!
  cmake --build "$BUILD" -j 4 --target vllm-cli > "$OUT/build.log" 2>&1
  rc=$?; kill "$hb" 2>/dev/null; note BUILD $rc
  if [ "$rc" -ne 0 ]; then tail -60 "$OUT/build.log"; echo "FATAL: build failed"; exit 94; fi
  mark build
fi
tail -2 "$OUT/build.log"
df -h / /tmp | tail -3

say "BINARY IDENTITY -- the executable AND every .so beside it"
CLI=$(find "$BUILD" -maxdepth 3 -name vllm-cli -type f -perm -u+x | head -1)
[ -n "$CLI" ] || { echo "FATAL: no vllm-cli under $BUILD"; exit 95; }
echo "CLI=$CLI"
sha256sum "$CLI" | tee "$OUT/binary.sha256"
find "$BUILD" -maxdepth 3 -name '*.so*' -type f -print0 | sort -z | xargs -0 -r sha256sum | tee -a "$OUT/binary.sha256"
scan(){ if command -v strings >/dev/null 2>&1; then strings -a "$1" 2>/dev/null; else cat "$1"; fi; }
found=no
for obj in "$CLI" "$(dirname "$CLI")"/*.so* "$BUILD"/*.so*; do
  [ -e "$obj" ] || continue; [ -d "$obj" ] && continue
  if scan "$obj" | grep -a -q 'GlmMoeDsaForCausalLM'; then found=yes; echo "GlmMoeDsaForCausalLM present in $(basename "$obj")"; fi
done
[ "$found" = yes ] || { echo "FATAL: GlmMoeDsaForCausalLM is in neither vllm-cli nor any .so beside it"; exit 95; }


# ─── THE READ-KEY, REGISTERED BEFORE THE RUN ─────────────────────────────────
# This is a PREDICTION, written into the script that produces the numbers, so it
# cannot be fitted to them afterwards. It is the same key that convicted the
# sibling `Glm5NextForConditionalGeneration` in the comment on issue #2544.
#
#   Leg C = `--device cuda`, expert streaming ON, DEFAULT async device mirror.
#   Leg E = byte-for-byte the same, plus `VT_ASYNC_DEVICE_MIRROR=0`.
#
# One variable moves between them, and it is not the streaming lane.
#
#   IF C and E emit DIFFERENT bytes  => `glm_moe_dsa` reads `token_ids` and
#       ignores `device_token_ids`, so every decode step after the first is
#       generated from token id 0. #2544 lists this model as a candidate on a
#       grep; differing bytes convict it, on the same isolation the sibling used.
#   IF C and E emit the SAME bytes   => the candidacy is FALSIFIED for this
#       model, and #2544's grep-based list is wrong about it.
#
# Neither outcome is a statement about expert streaming, which is ON in BOTH
# legs. Whichever leg emits text, that text is streamed text.
#
# Leg F (`--max-tokens 1`) is NOT re-run here: it already landed on main at
# `b5d82659c` (` Paris`, `[expert-stream] steps=1 ... exhausted=2303`), and a
# lease minute is better spent on the step that has never run.

say "THE LEGS -- $MAX_TOKENS tokens each, --device cuda, expert streaming ON in BOTH"
# `--device cuda` is REQUIRED for the streaming lane: model_loader.cpp builds it
# only under `needs_weight_staging() && host_memory_is_device_addressable()`,
# true on GB10 and false on every CPU. This is STREAMING (a byte-budgeted slot
# cache filled on router output), not PLACEMENT: VT_CPU_MOE is set nowhere in
# this script, and the check below reads that rather than asserting it.
echo "VT_CPU_MOE is ${VT_CPU_MOE:-UNSET}  (streaming, not placement, requires it UNSET)"

run_leg(){
  leg=$1; shift
  if stamp "leg$leg"; then echo "### leg $leg already done on this OUT -- skipping"; return 0; fi
  say "LEG $leg"
  # The leg's environment is the job's, plus the lane knobs, plus whatever this
  # leg adds in "$@". VT_CPU_MOE is UNSET rather than assumed unset, so a leaked
  # value from any earlier step cannot turn this into a placement run wearing a
  # streaming label. The resulting VT_* set is DUMPED, not described.
  unset VT_CPU_MOE VT_N_CPU_MOE VT_PLACEMENT_OVERRIDES VT_PLACEMENT_FIT
  ( env VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS="$SLOTS" \
        VT_MOE_EXPERT_STREAM_STATS_EVERY=1 VT_KV_ALLOC_LOG=1 "$@" \
        env | grep -aE '^VT_' | sort ) > "$OUT/leg$leg.env"
  echo "--- the VT_* environment this leg runs under ---"; cat "$OUT/leg$leg.env"
  echo "prompt: [$PROMPT]  max-tokens: $MAX_TOKENS  slots: $SLOTS"

  ( env VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS="$SLOTS" \
        VT_MOE_EXPERT_STREAM_STATS_EVERY=1 VT_KV_ALLOC_LOG=1 "$@" \
        "$CLI" --model "$SHARD1" \
        --device cuda --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 \
        > "$OUT/leg$leg.stdout" 2> "$OUT/leg$leg.stderr" ) &
  pid=$!
  hwm=0; gpumax=0; t0=$(date +%s); n=0
  : > "$OUT/leg$leg.mem"
  while kill -0 "$pid" 2>/dev/null; do
    v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
    [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    case "${g:-}" in ''|*[!0-9]*) g=NA ;; esac
    [ "$g" != NA ] && [ "$g" -gt "$gpumax" ] && gpumax=$g
    line="$(( $(date +%s) - t0 )) vmhwm_kb=$hwm gpu_used_mib=$g"
    echo "$line" >> "$OUT/leg$leg.mem"
    n=$((n + 1))
    if [ $((n % 8)) -eq 1 ]; then
      printf '  [leg %s heartbeat] %s | %s\n' "$leg" "$line" \
        "$(grep -a '\[expert-stream\]' "$OUT/leg$leg.stderr" 2>/dev/null | tail -1 | cut -c1-110)"
    fi
    sleep 15
  done
  wait "$pid"; rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  note "LEG$leg" $rc
  echo "### LEG${leg}_RC=$rc  wall=${elapsed}s"
  awk -v k="$hwm" 'BEGIN{printf "### VmHWM peak = %d kB = %.2f GiB (page-cache pressure, NOT residency -- glm-dsa spec O9)\n", k, k/1048576}'
  echo "### nvidia-smi memory.used peak = ${gpumax} MiB"
  echo "--- STDOUT, VERBATIM (cat -A: trailing space and newline visible) ---"
  cat -A "$OUT/leg$leg.stdout"
  echo "--- STDOUT as BYTES (the sibling's ' Paris.' and ' Paris Paris' differ LATE) ---"
  od -An -tx1 "$OUT/leg$leg.stdout" | tr -s ' '
  echo "--- stdout byte count: $(wc -c < "$OUT/leg$leg.stdout") ---"
  echo "--- stderr (head 30) ---"; head -30 "$OUT/leg$leg.stderr"
  echo "--- stderr (tail 60) ---"; tail -60 "$OUT/leg$leg.stderr"
  echo "--- every [expert-stream] line ---"
  grep -a '\[expert-stream\]' "$OUT/leg$leg.stderr"
  { echo "leg=$leg"; echo "rc=$rc"; echo "wall_s=$elapsed"; echo "vmhwm_kb=$hwm";
    echo "gpu_used_mib_peak=$gpumax"; } > "$OUT/leg$leg.result.env"
  mark "leg$leg"
}

# LEG C -- the goal run exactly as briefed: default configuration, nothing extra.
run_leg C

# LEG E -- the discriminator. One variable, and it is the async device mirror.
run_leg E VT_ASYNC_DEVICE_MIRROR=0

say "DERIVED VERDICTS -- computed from the captured bytes, never asserted"
# No conclusion string below is a constant. Every line is an expression over
# bytes read back off disk. A hardcoded verdict that never reads its input is how
# a job on this box reported '3 runs all rc=0' over its own captured 'rc=1'.
python3 - "$OUT" "$MAX_TOKENS" <<'PY'
import re, os, sys, hashlib
out, want = sys.argv[1], int(sys.argv[2])

def rd(p):
    try:  return open(p, 'rb').read()
    except FileNotFoundError: return None

def lane(errb):
    err = errb.decode('utf-8', 'replace')
    ban = re.findall(r'\[expert-stream\] ON [^\n]*', err)
    st  = re.findall(r'\[expert-stream\] steps=(\d+) hits=(\d+) misses=(\d+) '
                     r'evictions=(\d+) fills=(\d+) bytes=(\d+) exhausted=(\d+) advised=(\d+)', err)
    comp = re.findall(r'completion_tokens=(\d+)', err)
    return ban, st, comp

legs = {}
for L in ('C', 'E'):
    so, se = rd(f'{out}/leg{L}.stdout'), rd(f'{out}/leg{L}.stderr')
    if so is None or se is None:
        print(f'LEG {L}: ABSENT (did not run)'); continue
    ban, st, comp = lane(se)
    legs[L] = dict(stdout=so, banner=ban, steps=st, comp=comp)
    print(f'--- LEG {L} ---')
    print('  STDOUT_REPR       =', repr(so.decode('utf-8', 'replace')))
    print('  STDOUT_HEX        =', so.hex(' '))
    print('  STDOUT_SHA256     =', hashlib.sha256(so).hexdigest()[:32])
    print('  BANNER_COUNT      =', len(ban))
    for b in ban: print('  BANNER            =', b)
    print('  LANE_STAT_LINES   =', len(st))
    if st:
        print('     steps hits misses evictions fills bytes exhausted advised')
        for s in st: print('     ' + ' '.join(s))
        last = st[-1]
        print('  FINAL steps=%s hits=%s misses=%s evictions=%s fills=%s bytes=%s exhausted=%s'
              % (last[0], last[1], last[2], last[3], last[4], last[5], last[6]))
    print('  COMPLETION_TOKENS =', comp[-1] if comp else 'ABSENT')
    # Per-leg verdicts, each an expression over the values above.
    nsteps = int(st[-1][0]) if st else 0
    exh    = int(st[-1][6]) if st else -1
    hits   = int(st[-1][1]) if st else -1
    ntok   = int(comp[-1]) if comp else -1
    print('  VERDICT lane_banner_present =', len(ban) > 0)
    print('  VERDICT steps_reported      =', nsteps, '(>1 means DECODE ran, not prefill alone)')
    print('  VERDICT lane_is_streaming   =', (nsteps > 0 and exh == 0),
          '(docs/ENVIRONMENT.md: steps==0 or exhausted>0 means the lane is NOT streaming)')
    print('  VERDICT slot_reuse_observed =', hits > 0, '(hits =', hits, ')')
    print('  VERDICT tokens_emitted      =', ntok, 'of', want)

print()
print('=== THE READ-KEY, RESOLVED ===')
if 'C' in legs and 'E' in legs:
    same = legs['C']['stdout'] == legs['E']['stdout']
    print('C_BYTES =', legs['C']['stdout'].hex(' '))
    print('E_BYTES =', legs['E']['stdout'].hex(' '))
    print('BYTES_IDENTICAL =', same)
    if same:
        print('READ-KEY OUTCOME: the two legs agree. The async device-mirror input path is')
        print('  NOT a cause of divergence for glm_moe_dsa on this workload, so #2544\'s')
        print('  grep-based candidacy is FALSIFIED for this model by this measurement.')
    else:
        print('READ-KEY OUTCOME: the two legs DIFFER on one variable, VT_ASYNC_DEVICE_MIRROR.')
        print('  glm_moe_dsa reads `token_ids` and ignores `device_token_ids`, so every')
        print('  decode step after the first is generated from token id 0. This CONVICTS')
        print('  the model named as a candidate in #2544, on the same isolation that')
        print('  convicted Glm5NextForConditionalGeneration.')
        print('  Leg E is the arm whose text is trustworthy; leg C\'s is not.')
else:
    print('READ-KEY UNRESOLVED: both legs did not produce output. Nothing is concluded.')
PY

echo "--- leg result.env files ---"
for L in C E; do [ -f "$OUT/leg$L.result.env" ] && { echo "== leg $L =="; cat "$OUT/leg$L.result.env"; }; done
{ echo "box=$BOX"; echo "slots=$SLOTS"; echo "max_tokens=$MAX_TOKENS"; echo "arch=$ARCH";
  echo "base_sha=$(cat "$W/BASE_SHA" 2>/dev/null)"; } > "$OUT/result.env"
cat "$OUT/result.env"

say "DONE"
