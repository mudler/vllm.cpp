#!/usr/bin/env bash
# MODEL-TEXT-GLM-MOE-DSA -- drive the FIRST LOAD and the FIRST TOKEN of
# GLM-5.3 (`glm-dsa`, non-Flash) against the real 201.83 GiB UD-IQ1_S artifact,
# WITH EXPERT STREAMING. Spec `.agents/specs/glm-dsa-latest-deepseek.md`
# §3.3/§3.4/§3.6/§3.7 W7+W9, owed O7, O9, O14, O15, O17, O23, O29, O30.
# Issue https://github.com/mudler/vllm.cpp/issues/2214.
#
# NOT `set -e`. A refusal is a RESULT here; losing the log to an early exit is
# how a lease gets spent for nothing. Every step captures its own rc, and never
# `$?` after a pipe.
set -u

W=/workspace/glm53-firstload
CKPT=${CKPT:-/workspace/ckpt/GLM-5.3-UD-IQ1_S}
SHARD1=$CKPT/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf
SRC=/tmp/glm53fl/src
BUILD=/tmp/glm53fl/build-cuda
BOX=$(hostname)
PROMPT=${PROMPT:-The capital of France is}
MAX_TOKENS=${MAX_TOKENS:-1}
SLOTS=${SLOTS:-4096}

# THE OUTPUT DIRECTORY IS KEYED ON THE SOURCE ARCHIVE'S HASH, not on the box
# alone. The stamps below make this script resumable, and a resumable script
# whose stamps outlive a source change is a stale-binary generator: it would skip
# the extract and the build and then report the OLD binary's result under the NEW
# tree's name. Keying the stamps to the source makes that impossible to express.
SRCSHA=$(sha256sum "$W/src.tar.gz" 2>/dev/null | cut -c1-12)
# The RECIPE version rides in the key beside the source hash, because the stamps
# also outlive a change to the BUILD (r2 adds CUTLASS, r3 adds the CPU arm; without CUTLASS
# FlashAttention-2 compiles for no arch and MLA prefill refuses at the first
# step). A configure-time change that reused a stamp would report the previous
# recipe's binary under this one's name.
RECIPE=r3
OUT=$W/out/$BOX-${SRCSHA:-nosrc}-$RECIPE

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

say(){ printf '\n\n========== %s ==========  %s\n' "$1" "$(date -u +%FT%TZ)"; }
note(){ echo "### RC $1=$2"; }
stamp(){ [ -f "$OUT/stamp.$1" ]; }
mark(){ : > "$OUT/stamp.$1"; }
verdict(){ grep -aE '^\[doctest\] (assertions|test cases)' "$1" | tail -4; }

say "IDENTITY -- WHICH BOX EVERY NUMBER BELOW CAME FROM"
hostname; id -un; uname -m
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,compute_cap --format=csv 2>&1 | head -4
echo "cores: $(grep -c ^processor /proc/cpuinfo)"
free -g 2>&1 | head -2
echo "OUT=$OUT  (keyed on src.tar.gz $SRCSHA)"

say "WORKSPACE MOUNT GUARD -- a failed CIFS mount is an EMPTY LOCAL DIR and this job would 'succeed' writing nowhere"
if [ ! -s "$W/SENTINEL" ]; then echo "FATAL: /workspace is not the NAS -- no $W/SENTINEL"; exit 90; fi
sha256sum "$W/SENTINEL"
df -h /workspace 2>&1 | tail -2

say "DISK BEFORE"
df -h / /tmp /workspace 2>&1 | tail -6

say "THE ARTIFACT -- present, complete, and the exact byte count"
TOTAL_EXPECT=216715365893
if [ ! -d "$CKPT" ]; then echo "FATAL: no checkpoint directory at $CKPT"; exit 91; fi
n=$(ls -1 "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf 2>/dev/null | wc -l)
echo "shards found: $n"
if [ "$n" -ne 6 ]; then echo "FATAL: $n of 6 shards; a partial set is not a model"; exit 91; fi
total=$(du -bc "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf | tail -1 | cut -f1)
echo "total bytes: $total (expected $TOTAL_EXPECT)"
if [ "$total" -ne "$TOTAL_EXPECT" ]; then echo "FATAL: byte count disagrees; a short shard reads as a corrupt model"; exit 91; fi
echo "### ARTIFACT OK"
# The six sha256 were recorded by the fetch script as each shard landed
# (FETCH_FAIL=0) and were RE-VERIFIED independently off the same CIFS mount from
# the devbox on 2026-08-31, all six identical. Re-reading 201.83 GiB inside the
# lease would spend the lease on IO, so they are transcribed and the byte count
# above is what this job checks.
cat <<'HASHES'
### SHARD SHA256 (unsloth/GLM-5.3-GGUF @ 346b3591c7f28d1a23716f97a065ecf12ec14771, UD-IQ1_S)
00001 ff3adab0853dfb00bdf3889ec3f5556196f56b65783115720d57767bbd760dd9
00002 659d04cf4fc0b6026944f34c0b590a635803bff06c1775361e28490db7b168f8
00003 433302bac0e2d54da64c7c2f28509fa1b235aeccdf5b215a8a446ebaad1b5b27
00004 d0a6f19452d5b5cd498e1eb8fbe856e00aed7da1f80c27c095301eabe81e9bc1
00005 2ea1537ffab40fa8b8584a8647ec10fbaa6199dfed45e4019b822da2b319db37
00006 42a76ef04ffc5e321e1240f4e572b6fa6fc3315da5bea22fb598d7460db210fe
HASHES

say "TOOLCHAIN -- unconditional, the container is reused between jobs"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget ca-certificates gnupg git cmake ninja-build binutils >/dev/null 2>&1
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
  dpkg -i /tmp/ck.deb >/dev/null 2>&1
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-13-0
fi
export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; exit 90; }
nvcc --version | tail -2

say "SOURCE -- from the staged archive; no network dependency, no --prefix"
if [ ! -s "$W/src.tar.gz" ]; then echo "FATAL: no $W/src.tar.gz"; exit 92; fi
sha256sum "$W/src.tar.gz"
echo "EXPECTED_SRC_SHA256 $(cat "$W/src.tar.gz.sha256" 2>/dev/null)"
echo "BASE_SHA (recorded at archive time): $(cat "$W/BASE_SHA" 2>/dev/null)"
if ! stamp src; then
  # BOTH trees go, and that is deliberate. `git archive` stamps every file with
  # the COMMIT's time, which can be OLDER than an object file left by a previous
  # job -- ninja would then consider the object up to date and link the previous
  # source's binary under this source's name.
  rm -rf "$SRC" "$BUILD"; mkdir -p "$SRC"
  tar -xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: extract failed"; exit 92; }
  test -f "$SRC/CMakeLists.txt" || { echo "FATAL: no CMakeLists.txt at the archive root -- the archive was made with a --prefix"; exit 92; }
  find "$SRC" -exec touch {} + 2>/dev/null
  mark src
fi

say "CUDA ARCH -- READ off the device, never assumed"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
case "$CC" in
  121) ARCH=121a ;;   # GB10 (dgx)
  110) ARCH=110a ;;   # thor
  "")  ARCH=121a; echo "### WARNING: compute_cap unreadable, defaulting to $ARCH" ;;
  *)   ARCH=$CC ;;
esac
echo "### DEVICE compute_cap=$CC -> CUDA arch $ARCH"

say "CUTLASS -- FlashAttention-2 needs its headers, and without them it compiles for NO ARCH"
# The first run of this script omitted CUTLASS. The configure then printed
# `CUDA FA2 compiled-arch manifest: []` -- a line that says the feature is absent
# and does not say the word "error" -- and the load got all the way to the first
# step before throwing `cuda mla_prefill_attention: built without the vendored
# FlashAttention-2`. MLA prefill on this arch IS FlashAttention and the upstream
# selector has no fallback below it, so the omission is fatal at step 1 rather
# than slow. The tarball is staged on the share; nothing is downloaded.
CUT=/tmp/glm53fl/cutlass
if [ ! -f "$CUT/include/cutlass/cutlass.h" ]; then
  rm -rf "$CUT"; mkdir -p "$CUT"
  if [ -s /workspace/cutlass-v4.5.0.tar.gz ]; then
    tar -xzf /workspace/cutlass-v4.5.0.tar.gz -C "$CUT" || { echo "FATAL: cutlass untar"; exit 93; }
  else
    echo "### WARNING: no CUTLASS on the share -- FA2 will compile for no arch and MLA prefill WILL refuse"
  fi
fi
grep -aE '#define CUTLASS_(MAJOR|MINOR|PATCH)' "$CUT/include/cutlass/version.h" 2>/dev/null || echo "### no CUTLASS version header"

say "CONFIGURE"
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
# THE COMPILED FEATURE SET, ASSERTED BEFORE ANY RESULT IS BELIEVED. An empty FA2
# manifest is not an error line and it decides whether this model can take a
# single step, so it is read out loud here rather than discovered at the throw.
FA2LINE=$(grep -a 'FA2 compiled-arch manifest' "$OUT/cmake.log" | tail -1)
case "$FA2LINE" in
  *"manifest: []"*|"")
    echo "FATAL: FA2 is NOT compiled in ($FA2LINE)."
    echo "  MLA prefill on sm_121 IS FlashAttention and has no fallback below it,"
    echo "  so loading 201.83 GiB first would only reach the same throw 866s later."
    exit 94 ;;
esac
echo "### ${FA2LINE:-no FA2 manifest line in the configure log}"
case "$FA2LINE" in
  *'manifest: []'*)
    echo "### FA2 COMPILED FOR NO ARCH. On this build MLA prefill CANNOT run: it is FlashAttention"
    echo "### and the upstream selector has no fallback below it. Expect a refusal at the first step."
    echo "### On sm_110 (thor) that is the arch table, not the recipe: fa2 covers 8.0,8.6,8.7,8.9,12.0a,12.1a." ;;
  *) echo "### FA2 IS COMPILED for the arch(es) named above." ;;
esac

say "BUILD vllm-cli -- -j 4, because unconstrained parallelism has OOM-REBOOTED this box"
# THE STAMPS ARE ON THE SHARE AND THE BUILD IS IN THE WORKER'S /tmp, so a re-run
# that lands on a DIFFERENT worker inherits "already built" and an empty build
# tree. Check the artifact, not the stamp: an absent binary clears the stamps and
# rebuilds, instead of reaching the identity guard's refusal with the lease spent.
if stamp build && [ -z "$(find "$BUILD" -maxdepth 3 -name vllm-cli -type f -perm -u+x 2>/dev/null | head -1)" ]; then
  echo "### stamped as built, but no vllm-cli under $BUILD -- a different worker. Rebuilding."
  rm -f "$OUT"/stamp.src "$OUT"/stamp.cfg "$OUT"/stamp.build "$OUT"/stamp.suites
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
  cmake --build "$BUILD" -j 4 --target vllm-cli > "$OUT/build.log" 2>&1
  rc=$?; note BUILD $rc
  if [ "$rc" -ne 0 ]; then tail -60 "$OUT/build.log"; echo "FATAL: build failed"; exit 94; fi
  mark build
fi
tail -2 "$OUT/build.log"
df -h / /tmp | tail -3

say "BINARY IDENTITY GUARD -- the executable AND every .so beside it"
CLI=$(find "$BUILD" -maxdepth 3 -name vllm-cli -type f -perm -u+x | head -1)
[ -n "$CLI" ] || { echo "FATAL: no vllm-cli under $BUILD"; exit 95; }
echo "CLI=$CLI"
scan(){ if command -v strings >/dev/null 2>&1; then strings -a "$1" 2>/dev/null; else cat "$1"; fi; }
found=no
for obj in "$CLI" "$(dirname "$CLI")"/*.so* "$BUILD"/*.so*; do
  [ -e "$obj" ] || continue; [ -d "$obj" ] && continue
  if scan "$obj" | grep -a -q 'GlmMoeDsaForCausalLM'; then
    echo "identity OK: GlmMoeDsaForCausalLM present in $(basename "$obj")"; found=yes
  fi
done
[ "$found" = yes ] || { echo "FATAL: GlmMoeDsaForCausalLM is in neither vllm-cli nor any .so beside it"; exit 95; }
# THE FIX UNDER TEST IS IN THIS BINARY. `grep` on the source is not the question;
# the question is what got linked, and the sentinel below is a string only the
# post-fix loader carries.
for obj in "$CLI" "$(dirname "$CLI")"/*.so* "$BUILD"/*.so*; do
  [ -e "$obj" ] || continue; [ -d "$obj" ] && continue
  scan "$obj" | grep -a -q 'routed to an EXPAND residency' && echo "loader sentinel OK in $(basename "$obj")"
done
sha256sum "$CLI" | tee "$OUT/binary.sha256"
find "$BUILD" -maxdepth 3 -name '*.so*' -type f -print0 | sort -z | xargs -0 -r sha256sum | tee -a "$OUT/binary.sha256"

say "SPEC O17 -- THE PUBLISHED FILE STATES NO INDEXER SCHEDULE; the repair is to the FILE"
DERIVED=$W/derived
if ! stamp derived; then
  mkdir -p "$DERIVED"
  python3 - "$SRC/tests/vllm/models/glm_moe_dsa_config_glm53.inc" "$OUT/GLM-5.3-config.json" <<'PY'
import sys
s = open(sys.argv[1]).read()
a = s.index('R"GLM53(') + len('R"GLM53(')
b = s.index(')GLM53"')
open(sys.argv[2], 'w').write(s[a:b] + "\n")
PY
  echo "config.json extracted: $(wc -c < "$OUT/GLM-5.3-config.json") bytes  sha256=$(sha256sum "$OUT/GLM-5.3-config.json" | cut -d' ' -f1)"
  # IDEMPOTENT. `$DERIVED` is shared across runs and boxes, so the links are
  # usually already there; `ln -f` onto an existing link to the SAME inode fails
  # on this share, and a second job doing this at the same moment is a race. A
  # link that is already correct -- same size as the source -- is left alone, and
  # only a missing or wrong one is (re)made.
  for f in "$CKPT"/GLM-5.3-UD-IQ1_S-*-of-00006.gguf; do
    b=$(basename "$f")
    [ "$b" = "GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" ] && continue
    want=$(stat -c %s "$f")
    have=$(stat -c %s "$DERIVED/$b" 2>/dev/null || echo 0)
    if [ "$have" != "$want" ]; then
      rm -f "$DERIVED/$b"
      if ! ln "$f" "$DERIVED/$b" 2>/dev/null && ! cp -l "$f" "$DERIVED/$b" 2>/dev/null; then
        # A PEER JOB WON THE RACE. `$DERIVED` is shared across boxes and this
        # script is submitted to dgx and thor together, so between the `rm` and
        # the `ln` the other worker can recreate the same link. That is the
        # correct outcome, not a failure -- re-read the size and accept it.
        have=$(stat -c %s "$DERIVED/$b" 2>/dev/null || echo 0)
        if [ "$have" = "$want" ]; then
          echo "  $b: a peer job linked it first, size matches -- accepted"
        else
          echo "FATAL: cannot hardlink $b into $DERIVED (have=$have want=$want)"; exit 96
        fi
      fi
    fi
  done
  ls -la "$DERIVED" | head -8
  python3 "$SRC/scripts/glm-dsa-write-indexer-types.py" \
      --shard "$SHARD1" --from-config "$OUT/GLM-5.3-config.json" \
      --out "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" --force > "$OUT/indexer_types.log" 2>&1
  rc=$?; note INDEXER_REPAIR $rc
  cat "$OUT/indexer_types.log"
  [ "$rc" -ne 0 ] && { echo "FATAL: the metadata repair refused"; exit 96; }
  mark derived
fi
echo "### PUBLISHED shard 1 sha256:"; sha256sum "$SHARD1"
echo "### DERIVED  shard 1 sha256 (NOT unsloth/GLM-5.3-GGUF's shard 1):"; sha256sum "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf"

say "LEG 1 -- THE STREAMED LOAD AND THE FIRST TOKEN, on --device cuda"
# `--device cuda` is REQUIRED for the streaming lane: model_loader.cpp builds it
# only under `needs_weight_staging() && host_memory_is_device_addressable()`,
# which is a GB10/Thor and false on every CPU. On --device cpu the 228 towers
# would be read IN PLACE out of a 201.83 GiB CIFS-backed mmap, which is a
# page-cache number wearing a streaming label (spec §3.3).
export VT_MOE_EXPERT_STREAM=1
export VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS
export VT_MOE_EXPERT_STREAM_STATS_EVERY=1
export VT_KV_ALLOC_LOG=1
echo "VT_MOE_EXPERT_STREAM=1 VT_MOE_EXPERT_STREAM_SLOTS=$SLOTS VT_MOE_EXPERT_STREAM_STATS_EVERY=1 VT_KV_ALLOC_LOG=1"
echo "prompt: [$PROMPT]  max-tokens: $MAX_TOKENS"
echo "NOTE: the reads are served from CIFS (/workspace is //192.168.68.102/Data), NOT local NVMe."
echo "      Spec §3.7 W7's stop condition binds: any read-rate number below is a CIFS number (O7)."

( "$CLI" --model "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" \
         --device cuda --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 \
         > "$OUT/load.stdout" 2> "$OUT/load.stderr" ) &
pid=$!
hwm=0; gpumax=0; t0=$(date +%s)
: > "$OUT/mem.samples"
while kill -0 "$pid" 2>/dev/null; do
  v=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
  [ -n "${v:-}" ] && [ "$v" -gt "$hwm" ] && hwm=$v
  g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "${g:-}" in ''|*[!0-9]*) g=NA ;; esac
  [ "$g" != NA ] && [ "$g" -gt "$gpumax" ] && gpumax=$g
  printf '%s vmhwm_kb=%s gpu_used_mib=%s\n' "$(( $(date +%s) - t0 ))" "$hwm" "$g" >> "$OUT/mem.samples"
  sleep 15
done
wait "$pid"; rc=$?
elapsed=$(( $(date +%s) - t0 ))
note LOAD $rc

say "RESULT"
echo "### LOAD_RC=$rc   wall=${elapsed}s"
awk -v k="$hwm" 'BEGIN{printf "### VmHWM peak = %d kB = %.2f GiB\n", k, k/1048576}'
echo "### nvidia-smi memory.used peak = ${gpumax} MiB (NA on a box whose driver does not report it)"
echo "    VmHWM is NOT a residency measurement while the towers are mmap-resident: it tracks"
echo "    page-cache pressure (spec O9)."
echo "--- stdout (the emitted text, verbatim) ---"
cat "$OUT/load.stdout"
echo "--- stderr (head 60) ---"; head -60 "$OUT/load.stderr"
echo "--- stderr (tail 120) ---"; tail -120 "$OUT/load.stderr"
echo "--- expert-stream lines ---"
grep -a '\[expert-stream\]' "$OUT/load.stderr" | head -4
grep -a '\[expert-stream\]' "$OUT/load.stderr" | tail -20
echo "--- kv-alloc / residency lines ---"
grep -aiE 'kv-alloc|resident|arena|slot|fit|GiB' "$OUT/load.stderr" | head -60
echo "--- memory samples (first 5, last 5) ---"
head -5 "$OUT/mem.samples"; echo ...; tail -5 "$OUT/mem.samples"
{ echo "box=$BOX"; echo "rc=$rc"; echo "wall_s=$elapsed"; echo "vmhwm_kb=$hwm";
  echo "gpu_used_mib_peak=$gpumax"; echo "slots=$SLOTS"; echo "max_tokens=$MAX_TOKENS";
  echo "arch=$ARCH"; } > "$OUT/result.env"
cat "$OUT/result.env"
[ "$rc" -ne 0 ] && say "THE LOAD DID NOT COMPLETE -- the message above is the FINDING and is recorded verbatim"

say "LEG 1b -- THE CPU ARM. IT IS NOT A STREAMING MEASUREMENT AND IS LABELLED ONE"
# RUN ONLY WHEN THE CUDA ARM DID NOT PRODUCE A TOKEN, so a box that CAN stream
# spends its lease on the arm the goal names rather than on this one.
#
# READ WHAT THIS ARM IS BEFORE READING ITS NUMBER. `--device cpu` makes
# `needs_weight_staging()` false, so `model_loader.cpp` never builds the
# streamed-expert lane at all: `expert_stream::ExpertSlice` takes the RESIDENT
# fallback and every routed-expert slice is read IN PLACE out of the 201.83 GiB
# mmap, over CIFS. That is the page-cache path spec §3.3 refuses to publish under
# a streaming label, and nothing below may be quoted as an expert-streaming
# result. What it CAN answer is the other half of the question -- whether this
# port computes a token from this checkpoint at all -- on a queue whose MLA
# prefill is not FlashAttention and therefore does not need FA2.
if [ "$rc" -ne 0 ]; then
  export VT_MOE_EXPERT_STREAM=0
  echo "VT_MOE_EXPERT_STREAM=0 (the lane is not built on a CPU queue; saying so rather than implying one)"
  ( "$CLI" --model "$DERIVED/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf" \
           --device cpu --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature 0 \
           > "$OUT/cpu.stdout" 2> "$OUT/cpu.stderr" ) &
  cpid=$!
  chwm=0; c0=$(date +%s)
  : > "$OUT/cpu.mem.samples"
  while kill -0 "$cpid" 2>/dev/null; do
    v=$(awk '/VmHWM/{print $2}' "/proc/$cpid/status" 2>/dev/null)
    [ -n "${v:-}" ] && [ "$v" -gt "$chwm" ] && chwm=$v
    printf '%s vmhwm_kb=%s\n' "$(( $(date +%s) - c0 ))" "$chwm" >> "$OUT/cpu.mem.samples"
    sleep 15
  done
  wait "$cpid"; crc=$?
  celapsed=$(( $(date +%s) - c0 ))
  note CPU_LOAD $crc
  echo "### CPU_RC=$crc   wall=${celapsed}s"
  awk -v k="$chwm" 'BEGIN{printf "### CPU VmHWM peak = %d kB = %.2f GiB\n", k, k/1048576}'
  echo "--- CPU stdout (the emitted text, verbatim) ---"
  cat "$OUT/cpu.stdout"
  echo "--- CPU stderr (tail 60) ---"; tail -60 "$OUT/cpu.stderr"
  echo "--- CPU expert-stream lines (EXPECTED ABSENT: no lane is built on a CPU queue) ---"
  grep -a '\[expert-stream\]' "$OUT/cpu.stderr" | tail -10
  { echo "cpu_rc=$crc"; echo "cpu_wall_s=$celapsed"; echo "cpu_vmhwm_kb=$chwm"; } >> "$OUT/result.env"
  export VT_MOE_EXPERT_STREAM=1
else
  echo "### SKIPPED: the CUDA arm succeeded, so this box spent its lease on the arm the goal names."
fi

say "LEG 2 -- THE FOCUSED C++ SUITES, BY HAND, WITH THEIR COUNTS"
# AFTER the load, so a truncated lease still carries the primary result.
# `test_glm_moe_dsa_forward` needs VT_MOE_EXPERT_STREAM=1: StreamRequested() is
# read ONCE into a function-local static, so without it the binary reports 127
# assertions and one failure instead of 5,258 and green.
# `test_gguf_keep_quant` is the REGRESSION CONTROL for the new `prefault`
# parameter's DEFAULT: it carries the L7 prefault A/B and must be unmoved.
SUITES="test_glm_moe_dsa_gguf_load test_glm_moe_dsa_forward test_glm_moe_dsa_config test_glm_moe_dsa_schedule test_glm_moe_dsa_gguf_census test_expert_stream_wiring test_expert_stream_capacity test_gguf_keep_quant"
if ! stamp suites; then
  cmake --build "$BUILD" -j 4 --target $SUITES > "$OUT/build_suites.log" 2>&1
  brc=$?; note BUILD_SUITES $brc
  [ "$brc" -ne 0 ] && tail -40 "$OUT/build_suites.log"
  mark suites
fi
# The census gate is a statement about the PUBLISHED artifact, so it reads the
# published shard 1. The slot/stats knobs LEG 1 set are unset: a suite run under
# a non-default budget measures that budget.
unset VT_MOE_EXPERT_STREAM_SLOTS VT_MOE_EXPERT_STREAM_STATS_EVERY
export VT_GLM_DSA_GGUF="$SHARD1"
for t in $SUITES; do
  bin=$(find "$BUILD" -maxdepth 3 -name "$t" -type f -perm -u+x | head -1)
  if [ -z "$bin" ]; then echo "### SUITE $t: NOT BUILT"; continue; fi
  ( cd "$BUILD" && "$bin" ) > "$OUT/$t.log" 2>&1
  trc=$?; echo "### SUITE_RC($t)=$trc"; verdict "$OUT/$t.log"
done

say "LEG 3 -- THE MUTATION: put the tower prefault back, and the new case must RED"
# IMP-MUTATE. The fix is one argument at one call site, so the mutation is to
# restore the unconditional prefault there. A mutation killed by the COMPILER
# would be weaker than one killed by an assertion, so this one still compiles.
MUT=src/vllm/model_executor/models/glm_moe_dsa_loader.cpp
cp "$SRC/$MUT" "$OUT/mut_backup.cpp"
BEFORE=$(sha256sum "$SRC/$MUT" | cut -d' ' -f1); echo "pre-mutation sha256=$BEFORE"
python3 - "$SRC/$MUT" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "kGlmMoeDsaQuantRepack, /*cuda_align=*/false,\n                              /*prefault=*/false);"
b = "kGlmMoeDsaQuantRepack, /*cuda_align=*/false,\n                              /*prefault=*/true);"
assert a in s, "MUTATION DID NOT APPLY -- a mutation that never applied reads as a passing test"
s = s.replace(a, b, 1)
c = "/*elem_kn_repack=*/false, /*prefault=*/false);"
d = "/*elem_kn_repack=*/false, /*prefault=*/true);"
assert c in s, "MUTATION DID NOT APPLY (f16 arm)"
s = s.replace(c, d, 1)
open(p, 'w').write(s)
print("MUTATION APPLIED")
PY
mrc=$?; note MUTATION_APPLY $mrc
if [ "$mrc" -eq 0 ]; then
  cmake --build "$BUILD" -j 4 --target test_glm_moe_dsa_gguf_load > "$OUT/build_mut.log" 2>&1
  mbrc=$?; note MUTATION_BUILD $mbrc
  if [ "$mbrc" -ne 0 ]; then
    tail -30 "$OUT/build_mut.log"
    echo "### THE MUTATION DID NOT BUILD -- a build failure reads as a passing test, so this proves NOTHING"
  else
    bin=$(find "$BUILD" -maxdepth 3 -name test_glm_moe_dsa_gguf_load -type f -perm -u+x | head -1)
    ( cd "$BUILD" && "$bin" ) > "$OUT/mut_glm_load.log" 2>&1
    echo "### MUTATION_TEST_RC=$?  (NON-ZERO is the PASS: the mutation must red the case)"
    verdict "$OUT/mut_glm_load.log"
    grep -aE 'W10|prefaulted|ERROR' "$OUT/mut_glm_load.log" | head -12
  fi
fi
cp "$OUT/mut_backup.cpp" "$SRC/$MUT"
AFTER=$(sha256sum "$SRC/$MUT" | cut -d' ' -f1); echo "post-restore sha256=$AFTER"
[ "$BEFORE" = "$AFTER" ] && echo "### RESTORE OK -- byte-for-byte" || echo "### RESTORE FAILED"
cmake --build "$BUILD" -j 4 --target test_glm_moe_dsa_gguf_load > "$OUT/build_restore.log" 2>&1
note RESTORE_BUILD $?
bin=$(find "$BUILD" -maxdepth 3 -name test_glm_moe_dsa_gguf_load -type f -perm -u+x | head -1)
( cd "$BUILD" && "$bin" ) > "$OUT/restored_glm_load.log" 2>&1
echo "### RESTORED_TEST_RC=$?  (ZERO expected)"; verdict "$OUT/restored_glm_load.log"

say "DISK AFTER"
df -h / /tmp /workspace | tail -6

say "SUMMARY"
grep -aE '^### (RC|LOAD_RC|VmHWM|SUITE_RC|MUTATION|RESTORED)' "$OUT/run.log" | tail -40
exit "$rc"
