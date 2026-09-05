#!/bin/bash
# LTX25-RENDER-CONFIRM (#2457) -- the connector repair's RENDER, measured rather
# than projected, on a binary that takes #1864's correctness verdict inside this
# same job.
#
# WHY THIS FILE EXISTS AT ALL, AND WHY IT IS NOT A FLAG ON A NEIGHBOUR.
# `ltx25-render-speed-repeat.sh` times the WALL N times and REFUSES to build
# (exit 51), because its subject is the tree that `rc` job 4b0666ee established
# `VERDICT PASS` on and a rebuilt binary is a different measurement subject.
# That guard is correct. `LTX25-CONNECTOR-REPAIR` therefore could not re-measure
# the render at all, and its projection -- connector compute 224.882 s -> ~87.9 s,
# the render 516.751 s -> ~380 s, the oracle gap 5.51x -> ~4.05x -- is a
# component ratio carried onto a system, which is exactly the inference this
# repository has a standing note against.
#
# The way past that guard is to EARN it, not to remove it. This job builds the
# head, takes #1864's blockiness verdict on THAT binary, and only then times it.
# One build, one digest, both obligations closed over the same bytes. Running
# `ltx25-oracle-absolute-render.sh` and then `ltx25-render-speed-repeat.sh` in
# two leases would verify one binary and time another one built from the same
# source, which is the "an A/B that reuses one build dir measures one binary
# twice" defect from its mirror side.
#
# THE REQUEST IS THE MANIFEST'S, byte for byte, and identical to both
# neighbours'. A decomposition of a different request measures the request.
#
#   prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
#   320x192, 25 frames, 8 inference steps, seed 42
#   the four BF16 checkpoints of tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json
#
# ORDER: CORRECTNESS, THEN TIMING, and the order is enforced rather than
# intended. The CUDA unit gate runs before any render; render 1 is compared
# against #1864's reference the moment it finishes; a non-zero verdict EXITS
# before renders 2..N exist. A harness that collected the wall anyway would
# invite quoting a number whose picture was never judged.
#
# WHAT THE VERDICT CANNOT SEE, printed beside it rather than left implicit.
# `blockiness_grid8` / `blockiness_grid32` are ONE-SIDED ceilings and
# `.agents/specs/ltx25-oracle-absolute.md` records that our render is already
# less blocky, less sharp and less clipped than upstream's. A ceiling is blind
# to further smoothing, so phase [I] prints the tool's `reported_statistics`
# panel for both arms with the direction each moved.
#
# EXIT STATUS. 0 and 1 are the comparison's own verdict for render 1 and this
# job exits with it when the comparison decides; 2 is the comparison's
# UNREADABLE and is never a pass. Everything else is a refusal before a verdict:
#   23 a checkpoint sha256 that is not the manifest's, or staging failed
#   25 ltx2-gen will not exec        31 source tarball
#   33 configure   34 build   35 artefacts   36 no CUTLASS
#   37 ccache is mandatory on this host and could not be obtained
#   38 no complete CUDA toolkit
#   39 MemAvailable is below the start floor and stayed there
#   43 the comparison tool is not in this source
#   44 the CUDA unit gate FAILED     45 the CUDA unit gate BINARY IS ABSENT
#   48 a render produced the wrong number of frames, no audio, or no phase table
#   49 a render exited non-zero      50 the reference frames are not where this job expects them
#   52 a phase table covers less than 99% of its own wall
#   53 a render did not run 8 steps
set -u

T0=$SECONDS
say() { echo "[$(date -u +%H:%M:%S) +$((SECONDS-T0))s] $*"; }

W=${W:-/workspace/ltx25-render-confirm}
FULL=${FULL:-/workspace/ltx25-fullmodel}          # the DiT and the two VAEs
CKROOT=${CKROOT:-/workspace/ckpt/ltx-2.5}         # the BF16 text encoder
REFDIR=${REFDIR:-/workspace/ltx2-oracle/out/upstream_frames}
SRC=/root/src
BLD=/root/build-confirm
BIN=/root/confirmbin
CK=/root/ckpt
CACHE=$W/confirm-bin
N=${N:-3}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$W/run/$RUN_ID
mkdir -p "$OUT" "$CK" "$BIN" "$CACHE"

PROMPT='A red fox walks slowly through a snowy pine forest at sunrise, cinematic.'
WW=320; HH=192; FRAMES=25; STEPS=8; SEED=42
TOK=$(( (WW/32) * (HH/32) * (((FRAMES-1)/8) + 1) ))

export DEBIAN_FRONTEND=noninteractive

{
  echo "run_id=$RUN_ID"
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness=$0"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "renders_requested=$N"
  echo "geometry=${WW}x${HH}/${FRAMES}f steps=$STEPS seed=$SEED video_tokens=$TOK"
  echo "prompt_sha256=$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
} >> "$OUT/PROVENANCE"

# HEARTBEAT ON STDOUT, and `HEARTBEAT=$!` rather than a command substitution --
# `.agents/oracles/ltx-2.md` records 2h37m of a lease lost to the substitution
# form holding the pipe open. Signals too, not only EXIT: `rc` reclaiming a
# device sends SIGTERM and a bare EXIT trap leaves the heartbeat orphaned.
( while true; do sleep 120; echo "[hb +$((SECONDS-T0))s] alive"; done ) &
HEARTBEAT=$!
cleanup() { kill "$HEARTBEAT" 2>/dev/null; }
trap cleanup EXIT
for sig in HUP INT TERM; do
  trap "cleanup; exit \$((128 + \$(kill -l $sig)))" "$sig"
done

say "=== [0] the box ==="
uname -m; nproc; free -g; uptime
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,clocks.sm,persistence_mode --format=csv 2>&1 | head -3
df -h / /root /workspace 2>&1 | head -6

mem_avail_gib() { awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo; }
contention() {
  echo "  loadavg=$(cut -d' ' -f1-3 /proc/loadavg) memavail=$(mem_avail_gib)GiB"
  nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/  gpu=/'
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  gpuproc=/'
}

# 78 GiB is the neighbouring harnesses' DERIVED floor for this exact resident
# set: the 42 GB DiT plus the 26.3 GB BF16 tower plus the VAEs plus slack, on a
# box whose memory is unified so the device allocation comes out of the same
# pool. #1709 is why it is checked: this box has reported 5.0 GiB available
# across four leases while `rc` kept handing it out.
MEM_START_FLOOR_GIB=${MEM_START_FLOOR_GIB:-78.0}
MEM_START_WAIT_S=${MEM_START_WAIT_S:-1200}
say "=== [0b] MemAvailable start gate: floor ${MEM_START_FLOOR_GIB} GiB ==="
waited=0
while :; do
  avail=$(mem_avail_gib)
  say "  memavail=${avail} GiB after ${waited}s"
  awk -v a="$avail" -v f="$MEM_START_FLOOR_GIB" 'BEGIN{exit !(a+0 >= f+0)}' && break
  [ "$waited" -ge "$MEM_START_WAIT_S" ] && {
    echo "FATAL: MemAvailable ${avail} GiB stayed below ${MEM_START_FLOOR_GIB} GiB for ${waited}s"; exit 39; }
  sleep 30; waited=$((waited + 30))
done
echo "mem_available_at_start_gib=$(mem_avail_gib)" >> "$OUT/PROVENANCE"

say "=== [1] tools ==="
apt-get update -qq > /root/apt.log 2>&1 || say "  apt-get update returned non-zero; probing anyway"
apt-get install -y -qq ffmpeg python3-numpy ccache >> /root/apt.log 2>&1 || say "  apt returned non-zero; probing anyway"
for t in ffmpeg python3 cmake ninja; do command -v "$t" >/dev/null || { echo "FATAL: no $t"; exit 38; }; done
python3 -c 'import numpy' || { echo "FATAL: no numpy, and the comparison tool needs it"; exit 38; }

# CCACHE IS MANDATORY ON THIS HOST, and `rc describe dgx:gpu0`'s usage sheet says
# so in those words. It is a REFUSAL rather than a warning because the failure it
# prevents has already been paid repeatedly: a cold 20-25 minute build inside a
# lease whose subject is a render.
#
# THE CACHE MUST NOT LIVE ON THE NAS, and the usage sheet's instruction to keep
# it there is what made this whole block a no-op until 2026-09-01 (#2473).
# `/workspace` is CIFS mounted `nounix`; ccache 4.9.1 takes every cache AND stats
# lock by creating a symlink; symlink(2) on that mount returns EOPNOTSUPP. So
# nothing is stored, no counter is written, and `ccache -s` reports zero hits,
# zero misses and zero stores. That reading is the trap: a cache consulted and
# empty records MISSES, so zero-of-everything looks like a launcher that never
# ran. `rc` job 93a60151 configured all three launchers correctly and paid its
# 1404 s build in full. It is also WORSE than a no-op, because each refused lock
# costs a retry timeout and `ccache -s` walks 256 buckets. It is the same CIFS
# limitation that destroys the staged CUDA toolkit's SONAME link (#2220).
#
# CCACHE_DIR therefore goes on LOCAL disk, and persistence across the container
# comes from ccache's own remote storage, which MAY live on the NAS: the `file`
# backend stores through open plus rename, which this mount serves, and takes no
# lock. Measured: a fresh empty local cache against a populated remote store went
# 18 s -> 5 s at 8/8 remote hits (`rc` job e4793984).
command -v ccache >/dev/null || { echo "FATAL: ccache is mandatory for a C/C++/CUDA build on this host and is not installed"; exit 37; }
export CCACHE_DIR=${CCACHE_DIR:-/root/ccache}
mkdir -p "$CCACHE_DIR" || { echo "FATAL: cannot create $CCACHE_DIR"; exit 37; }
# THE ONE CAPABILITY THAT DECIDES IT, asserted in milliseconds rather than
# discovered 23 minutes later in a counter nobody reads.
if ! ln -sf . "$CCACHE_DIR/.symlink-probe" 2>/dev/null; then
  echo "FATAL: $CCACHE_DIR cannot hold a symlink, so ccache cannot take its locks there (#2473)"
  echo "       Point CCACHE_DIR at local disk and keep only CCACHE_REMOTE_STORAGE on /workspace."
  exit 37
fi
rm -f "$CCACHE_DIR/.symlink-probe"
export CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE:-file:/workspace/ccache-remote}
mkdir -p "${CCACHE_REMOTE_STORAGE#file:}" 2>/dev/null || true
export CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}
ccache --version | head -1
ccache -s > "$OUT/ccache-before.txt" 2>&1 || true
echo "ccache_dir=$CCACHE_DIR ccache_remote=$CCACHE_REMOTE_STORAGE ccache_version=$(ccache --version | head -1)" >> "$OUT/PROVENANCE"

say "=== [A] CUDA toolkit ==="
# THE SONAME IS WHAT MUST EXIST, AND IT IS WHAT CIFS DESTROYS (#2220).
# `/workspace` stores no symlink, so a staged toolkit carries only the versioned
# regular file `libcudart.so.13.3.29`; `nvcc` compiles happily against headers
# and the failure lands 21 minutes later at the consumer link. This checks what
# the LINKER needs, and it is asserted in seconds before the build.
soname_ok() {
  local target major
  target=$(readlink -f "$1/$2.so" 2>/dev/null) || return 1
  [ -e "$target" ] || return 1
  major=$(basename "$target"); major=${major#*.so.}; major=${major%%.*}
  [ -n "$major" ] || return 1
  [ -e "$1/$2.so.$major" ]
}
need_ok() {
  [ -x "$1/bin/nvcc" ] || return 1
  soname_ok "$1/targets/sbsa-linux/lib" libcudart &&
    soname_ok "$1/targets/sbsa-linux/lib" libcublasLt
}
TKLIB=""
for c in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
  if need_ok "$c"; then TKLIB=$c; break; fi
done
if [ -z "$TKLIB" ] && [ -d /workspace/a3/cuda-staged ]; then
  say "  staging the toolkit from /workspace/a3/cuda-staged (CIFS holds no symlink and serves 0664)"
  cp -a /workspace/a3/cuda-staged /root/cudatk || { echo "FATAL: cannot stage the toolkit"; exit 38; }
  chmod -R 0755 /root/cudatk/bin /root/cudatk/nvvm/bin 2>/dev/null
  L=/root/cudatk/targets/sbsa-linux/lib
  ldconfig -n "$L" 2>/dev/null
  ( cd "$L" 2>/dev/null && for f in *.so.*; do
      case "$f" in *.so.*.*) ;; *) continue;; esac
      b=${f%%.so.*}; v=${f#*.so.}
      ln -sf "$f" "$b.so"
      [ -e "$b.so.${v%%.*}" ] || ln -sf "$f" "$b.so.${v%%.*}"
    done ) 2>/dev/null
  need_ok /root/cudatk && TKLIB=/root/cudatk
fi
[ -n "$TKLIB" ] || {
  echo "FATAL: no CUDA toolkit whose libcudart/libcublasLt SONAME links resolve (#2220)"
  for d in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
    [ -d "$d" ] || continue
    echo "  $d/targets/sbsa-linux/lib:"
    ls -la "$d/targets/sbsa-linux/lib" 2>/dev/null | grep -E "libcudart|libcublasLt" | head -8
  done
  exit 38; }
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "  toolkit $TKLIB, $(nvcc --version | tail -1)"
for s in libcudart libcublasLt; do
  t=$(readlink -f "$TKLIB/targets/sbsa-linux/lib/$s.so")
  m=$(basename "$t"); m=${m#*.so.}; m=${m%%.*}
  say "  $s.so -> $(basename "$t"), SONAME link $s.so.$m present"
  echo "toolkit_soname $s.so.$m -> $(basename "$t")" >> "$OUT/PROVENANCE"
done

say "=== [B] source ==="
[ -s "$W/src.tar.gz" ] || { echo "FATAL: no $W/src.tar.gz"; exit 31; }
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 31; }
WANT_SHA=$(cat "$W/src.sha" 2>/dev/null)
TAR_SHA=$(sha256sum "$W/src.tar.gz" | awk '{print $1}')
{ echo "source_sha=$WANT_SHA"; echo "source_tarball_sha256=$TAR_SHA"; } >> "$OUT/PROVENANCE"
[ -n "$WANT_SHA" ] || { echo "FATAL: $W/src.sha is empty; the cache key would be the empty string"; exit 31; }
[ -s "$SRC/scripts/ltx25-render-compare.py" ] || { echo "FATAL: the comparison tool is not in this source"; exit 43; }
grep -q -- '--reference' "$SRC/scripts/ltx25-render-compare.py" || {
  echo "FATAL: this source's comparison tool has no --reference, so it predates the gate this job runs"; exit 43; }
grep -q -- '"--steps"' "$SRC/examples/ltx2_gen/main.cpp" || {
  echo "FATAL: this source's ltx2-gen has no --steps, so the render cannot be step-matched (#2130)"; exit 43; }

say "=== [C] CUTLASS ==="
CUT=""
for c in /cutlass /workspace/cutlass /root/cutlass; do
  [ -f "$c/include/cutlass/cutlass.h" ] && CUT=$c && break
done
if [ -z "$CUT" ] && [ -s /workspace/cutlass-v4.5.0.tar.gz ]; then
  # NO `--strip-components`: the staged tarball's first member is `include/`.
  say "  unpacking the staged cutlass"
  mkdir -p /root/cutlass && tar xzf /workspace/cutlass-v4.5.0.tar.gz -C /root/cutlass
  [ -f /root/cutlass/include/cutlass/cutlass.h ] && CUT=/root/cutlass
fi
[ -n "$CUT" ] || { echo "FATAL: no CUTLASS"; exit 36; }

say "=== [D] build ==="
BUILT_FROM=cache
# ALL-OR-NOTHING. A cache staged without `test_ltx2_device` would satisfy a
# per-file build skip and then leave nothing to run the correctness gate with.
if [ -s "$CACHE/ltx2-gen" ] && [ -s "$CACHE/libvllm.so.0.0.3" ] && [ -s "$CACHE/test_ltx2_device" ] && \
   [ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SHA" ]; then
  say "  cache hit on SRC_SHA=$WANT_SHA"
  cp -f "$CACHE/ltx2-gen" "$CACHE/libvllm.so.0.0.3" "$CACHE/test_ltx2_device" "$BIN"/
  chmod 0755 "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
else
  BUILT_FROM=in-lease
  BT0=$SECONDS
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
        -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache > "$OUT/configure.log" 2>&1 \
        || { echo "FATAL: configure failed"; tail -30 "$OUT/configure.log"; exit 33; }
  # NAMED TARGETS AND -j 4. A bare `ninja` links every test binary and writes
  # 9.4 GiB, and unconstrained parallelism has OOM-rebooted this box.
  ninja -C "$BLD" -j 4 ltx2-gen test_ltx2_device > "$OUT/build.log" 2>&1 \
        || { echo "FATAL: build failed"; tail -40 "$OUT/build.log"; exit 34; }
  BUILD_S=$((SECONDS-BT0))
  say "  built in ${BUILD_S}s"
  echo "build_seconds=$BUILD_S" >> "$OUT/PROVENANCE"
  GEN=$(find "$BLD" -name ltx2-gen -type f | head -1)
  LIB=$(find "$BLD" -name 'libvllm.so.0.0.3' -type f | head -1)
  TD=$(find "$BLD" -name test_ltx2_device -type f | head -1)
  for f in "$GEN" "$LIB" "$TD"; do [ -s "$f" ] || { echo "FATAL: missing build artefact"; exit 35; }; done
  cp -f "$GEN" "$LIB" "$TD" "$BIN"/
  chmod 0755 "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  cp -f "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device" "$CACHE"/ 2>/dev/null
  echo "$WANT_SHA" > "$CACHE/SRC_SHA"
fi
# WHETHER CCACHE ACTUALLY HIT, REFUSED rather than recorded. Configuring the
# launcher and getting no hits is the shape of a cache that is present and
# useless, and it reads identically to a fast build in a log that does not say.
# This line used to print the counters and continue; that is exactly what #2473
# did for a month while every lease paid a full build.
#
# The assertion is CACHEABLE CALLS, not hits. A genuinely first build on a new
# source has every right to zero hits, and demanding one would red an honest
# cold run. Zero cacheable calls after a build that compiled something means the
# cache was never able to participate at all, which is the defect.
ccache -s -v > "$OUT/ccache-after.txt" 2>&1 || true
say "  ccache after: $(grep -iE 'cacheable calls|^ +hits|^ +misses' "$OUT/ccache-after.txt" | tr '\n' ' ')"
if [ "$BUILT_FROM" = "in-lease" ]; then
  CALLS=$(sed -n 's/^Cacheable calls: *\([0-9]*\).*/\1/p' "$OUT/ccache-after.txt" | head -1)
  [ -n "$CALLS" ] && [ "$CALLS" -gt 0 ] 2>/dev/null || {
    echo "FATAL: this lease compiled and ccache recorded ${CALLS:-no} cacheable calls, so the"
    echo "       cache took no part in it. That is the #2473 shape: the launchers are"
    echo "       configured, every write is refused, and the build is paid in full."
    echo "       CCACHE_DIR=$CCACHE_DIR CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE:-none}"
    head -20 "$OUT/ccache-after.txt"
    exit 37; }
  say "  ccache took part: $CALLS cacheable calls"
  echo "ccache_cacheable_calls=$CALLS" >> "$OUT/PROVENANCE"
fi
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
# BOTH HASHES, AND THE LIBRARY IS THE ONE THAT MATTERS (#1881): `ltx2-gen` is a
# small launcher whose digest has been byte-identical across builds hundreds of
# commits apart while the libraries differed by megabytes.
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
LIBSHA=$(sha256sum "$BIN/libvllm.so.0.0.3" | awk '{print $1}')
{ echo "binary_sha256=$BINSHA"; echo "library_sha256=$LIBSHA"; echo "binary_built=$BUILT_FROM"; } >> "$OUT/PROVENANCE"
"$BIN/ltx2-gen" --help >/dev/null 2>&1 || {
  echo "FATAL: ltx2-gen will not exec (126 = no exec bit, 127 = missing lib)"; ldd "$BIN/ltx2-gen" | head; exit 25; }
say "  ltx2-gen=$BINSHA lib=$LIBSHA built=$BUILT_FROM src=$WANT_SHA"
say "  THE PIN THIS RUN CAN EARN: WANT_BIN_SHA=$BINSHA WANT_LIB_SHA=$LIBSHA WANT_SRC_SHA=$WANT_SHA"

say "=== [E] checkpoints, staged LOCAL and CHECKED AGAINST THE MANIFEST ==="
# NEVER RUN A MODEL OFF /workspace: it is CIFS on the house NAS, `rc describe`
# says so, and a 34-83 MiB/s network read inside the `load` phase would make the
# load number a statement about the NAS. Staged to /root, which is local.
#
# THE DIRECTORY IS STABLE RATHER THAN `mktemp`, and the sha256 is what makes
# that safe: every file's digest is recomputed against the manifest on EVERY
# run, so a reused stage is proved to be the right bytes rather than assumed to
# be. A fresh `mktemp` per job would re-copy 70 GB and verify no more than this.
declare -A SRCOF=(
  [ltx-2.5-22b-dev-transformer-bf16.safetensors]="$FULL/ckpt"
  [ltx-2.5-video-vae-conv-bf16.safetensors]="$FULL/ckpt"
  [ltx-2.5-audio-vae-bf16.safetensors]="$FULL/ckpt"
  [gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]="$CKROOT/text_encoders"
)
declare -A WANTSZ=(
  [ltx-2.5-22b-dev-transformer-bf16.safetensors]=42018190584
  [ltx-2.5-video-vae-conv-bf16.safetensors]=1452269922
  [ltx-2.5-audio-vae-bf16.safetensors]=364866540
  [gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]=26263858182
)
declare -A WANTSHA=(
  [ltx-2.5-22b-dev-transformer-bf16.safetensors]=792a2bad501ca03262c0bc2ce7a2949e85b142ce18e30894aad5bc849c8e7584
  [ltx-2.5-video-vae-conv-bf16.safetensors]=685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d
  [ltx-2.5-audio-vae-bf16.safetensors]=c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5
  [gemma4-12b-with-proj-ltx-2.5-bf16.safetensors]=ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1
)
NEED_K=$(( (42018190584 + 1452269922 + 364866540 + 26263858182) / 1024 + 8388608 ))
FREE_K=$(df -k --output=avail /root | tail -1)
[ "$FREE_K" -gt "$NEED_K" ] || {
  echo "FATAL: /root has ${FREE_K}K free against ${NEED_K}K needed, and reading the checkpoints over CIFS would put the NAS inside the load phase this job decomposes"; exit 23; }
for f in "${!WANTSZ[@]}"; do
  s="${SRCOF[$f]}/$f"; want=${WANTSZ[$f]}; wsha=${WANTSHA[$f]}
  got=$(stat -c %s "$s" 2>/dev/null || echo 0)
  [ "$got" = "$want" ] || { echo "FATAL: source $f is $got bytes, the manifest says $want"; exit 23; }
  d=$CK/$f
  if [ -s "$d" ] && [ "$(stat -c %s "$d")" = "$want" ]; then
    say "  already staged $f"
  else
    # `.part` RENAMED ONLY ON A SIZE MATCH: `oracle-ltx-2-pin.md` records a
    # `cp: Resource temporarily unavailable` off this soft CIFS mount that a
    # plain `cp` swallowed, and the half-written file that followed.
    t=$SECONDS; rm -f "$d" "$d.part"
    cp -- "$s" "$d.part" || { echo "FATAL: cannot stage $f"; exit 23; }
    [ "$(stat -c %s "$d.part")" = "$want" ] || { echo "FATAL: short stage of $f"; exit 23; }
    mv -f "$d.part" "$d"
    say "  staged $f $want bytes in $((SECONDS-t))s"
  fi
  t=$SECONDS
  gsha=$(sha256sum "$d" | awk '{print $1}')
  [ "$gsha" = "$wsha" ] || { echo "FATAL: $f sha256 $gsha, the manifest says $wsha"; exit 23; }
  say "  sha256 OK $f ($((SECONDS-t))s)"
  echo "checkpoint_sha256 $f $gsha" >> "$OUT/PROVENANCE"
done

say "=== [F] the CUDA unit gate, BEFORE any render ==="
# A doctest binary that skips everything also exits 0, so the case and assertion
# counts are recorded rather than the status alone.
[ -s "$BIN/test_ltx2_device" ] || { echo "FATAL: the CUDA unit gate binary is absent"; exit 45; }
"$BIN/test_ltx2_device" > "$OUT/test_ltx2_device.log" 2>&1 || {
  echo "FATAL: the CUDA unit gate FAILED; correctness comes before a render"
  tail -30 "$OUT/test_ltx2_device.log"; exit 44; }
grep -E 'test cases:|assertions:' "$OUT/test_ltx2_device.log" | tee -a "$OUT/PROVENANCE"

[ -d "$REFDIR" ] || { echo "FATAL: the reference frames are not at $REFDIR"; exit 50; }

say "=== [G] ${N} renders, and the verdict on the FIRST one ==="
for i in $(seq 1 "$N"); do
  D=$OUT/r$i
  rm -rf "$D"; mkdir -p "$D"
  LOG=$OUT/render-$i.log
  say "  --- render $i of $N ---"
  { echo "== render $i pre-state"; contention; } | tee -a "$OUT/PROVENANCE"
  {
    echo "[arm] label=ours render=$i"
    echo "[arm] binary=$BIN/ltx2-gen sha256=$BINSHA src_sha=$WANT_SHA"
    echo "[arm] library=$BIN/libvllm.so.0.0.3 sha256=$LIBSHA"
    echo "[arm] geometry=${WW}x${HH}/${FRAMES}f steps=$STEPS tokens=$TOK seed=$SEED"
  } >> "$LOG"

  # `exec`, and it is the fix for a defect the neighbouring harness shipped with
  # (#2305). Without it `$!` is the SUBSHELL's pid, the TERM below kills the
  # subshell, and the python sampler is orphaned -- it keeps sampling into the
  # NEXT render and never writes its summary, because the helper writes that
  # only when the sampler STOPS.
  ( cd "$SRC" && exec python3 -m tools.bench.gpu_clock_state sample \
      --output "$OUT/clock-$i.jsonl" --summary "$OUT/clock-$i.json" --interval 2 \
      > "$OUT/clock-$i.stdout" 2>&1 ) &
  CLK=$!

  t=$SECONDS
  VT_OP_PROVIDER_STATS=1 stdbuf -oL -eL "$BIN/ltx2-gen" \
    --pipeline-kind one_stage \
    --checkpoint-class full \
    --dit "$CK/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
    --video-vae "$CK/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae "$CK/ltx-2.5-audio-vae-bf16.safetensors" \
    --encoder "$CK/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "$WW" --height "$HH" --steps "$STEPS" --seed "$SEED" \
    --device cuda --workdir "$D" >> "$LOG" 2>&1
  RC=$?
  SEC=$((SECONDS-t))
  kill -TERM "$CLK" 2>/dev/null; wait "$CLK" 2>/dev/null
  [ -s "$OUT/clock-$i.json" ] || say "  WARNING: the clock sampler wrote no summary for render $i (#2305)"

  say "  render $i rc=$RC in ${SEC}s"
  echo "render_${i}_rc=$RC render_${i}_seconds=$SEC" >> "$OUT/PROVENANCE"
  { echo "== render $i post-state"; contention; } >> "$OUT/PROVENANCE"
  [ "$RC" = 0 ] || { echo "FATAL: render $i exited $RC"; tail -40 "$LOG"; exit 49; }

  NF=$(ls "$D"/frame_*.ppm 2>/dev/null | wc -l)
  [ "$NF" = "$FRAMES" ] && [ -s "$D/audio.wav" ] || {
    echo "FATAL: render $i is incomplete ($NF of $FRAMES frames)"; tail -40 "$LOG"; exit 48; }
  [ -s "$D/phase-log.json" ] || { echo "FATAL: render $i wrote no phase table"; exit 48; }

  # THE SCHEDULE IS READ BACK FROM THE RENDER'S OWN LINES, not from the flag.
  # `one_stage` is GUIDED and runs three DiT forwards per step, so the DISTINCT
  # set of denominators is the schedule and the line count measures the guider.
  OBS=$(grep -oE 'step [0-9]+/[0-9]+' "$LOG" | awk -F/ '{print $2}' | sort -u | tr '\n' ',' )
  FWD=$(grep -cE 'step [0-9]+/[0-9]+' "$LOG")
  echo "render_${i}_steps_observed={${OBS%,}} render_${i}_dit_forwards=$FWD" >> "$OUT/PROVENANCE"
  [ "${OBS%,}" = "$STEPS" ] || { echo "FATAL: render $i observed steps {${OBS%,}}, not $STEPS"; exit 53; }

  # A TABLE THAT STOPPED COVERING ITS RENDER IS REFUSED RATHER THAN SUMMED.
  python3 - "$D/phase-log.json" <<'PY' || exit 52
import json, sys
d = json.load(open(sys.argv[1]))
wall = d["wall_seconds"]; un = d["unaccounted_seconds"]
frac = un / wall if wall else 1.0
print(f"  coverage wall={wall:.3f}s unaccounted={un:.4f}s ({frac*100:.3f}%)")
sys.exit(0 if frac < 0.01 else 1)
PY
  cp -f "$D/phase-log.json" "$OUT/phase-log-$i.json"

  if [ "$i" = 1 ]; then
    say "=== [H] #1864's verdict, on the binary that just produced render 1 ==="
    # THE EXACT FORM FIRST: the 25 PPM frames upstream's own decode wrote, each
    # checked against the committed SHA256SUMS by the tool itself. The mp4 form
    # is the cross-check, because the claim that the two agree to better than
    # 0.08% is a number somebody wrote down unless something re-runs it.
    python3 "$SRC/scripts/ltx25-render-compare.py" \
      --a "$D" --label-a ours \
      --reference "$REFDIR" \
      --json "$OUT/absolute-vs-reference.json" 2>&1 | tee "$OUT/compare.log"
    CMP_RC=${PIPESTATUS[0]}
    say "  comparison against the PPM frames: exit $CMP_RC"
    python3 "$SRC/scripts/ltx25-render-compare.py" \
      --a "$D" --label-a ours \
      --reference "$SRC/tests/parity/goldens/ltx2_oracle/upstream-render.mp4" \
      --json "$OUT/absolute-vs-committed-mp4.json" > "$OUT/compare-mp4.log" 2>&1
    MP4_RC=$?
    say "  comparison against the committed mp4: exit $MP4_RC"
    echo "compare_exit_ppm=$CMP_RC compare_exit_mp4=$MP4_RC" >> "$OUT/PROVENANCE"
    if [ "$CMP_RC" != "$MP4_RC" ]; then
      say "  NOTE: the two reference forms DISAGREE on the verdict ($CMP_RC vs $MP4_RC)."
      say "  That is a finding about the mp4's usability as a reference and belongs in the spec."
    fi
    # THE REPORTED-ONLY PANEL, printed because the GATED pair cannot see it.
    # `blockiness_grid8` / `grid32` are ONE-SIDED ceilings and our render is
    # already less blocky, less sharp and less clipped than upstream's, so a
    # PASS beside an unremarked sharpness collapse would be a green gate on a
    # worse render.
    python3 - "$OUT/absolute-vs-reference.json" <<'PY' | tee "$OUT/reported-panel.txt"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  reported panel UNREADABLE: {e}")
    raise SystemExit(0)
ours = d.get("absolute_quality", {}).get("ours", {})
ref = d.get("reference", {}).get("bounds", {})
print(f"  reading             = {d.get('reading')}")
print(f"  checked_statistics  = {ours.get('checked_statistics')}")
print(f"  reported_statistics = {ours.get('reported_statistics')}")
print(f"  {'statistic':32s}{'ours':>14s}{'ref mean':>14s}{'ref min':>14s}{'ref max':>14s}  direction")
for k in ("blockiness_grid8", "blockiness_grid32", "sharpness_mean",
          "clipped_fraction", "audio_rms_mean"):
    o = ours.get(k)
    b = ref.get(k, {})
    if o is None:
        continue
    rm = b.get("mean")
    gated = k in (ours.get("checked_statistics") or [])
    tag = "GATED" if gated else "reported-only"
    if rm is None:
        print(f"  {k:32s}{o:14.6f}{'-':>14s}{'-':>14s}{'-':>14s}  {tag}, no reference bound")
        continue
    d_pct = (o - rm) / rm * 100 if rm else 0.0
    arrow = "LOWER" if d_pct < 0 else "HIGHER"
    print(f"  {k:32s}{o:14.6f}{rm:14.6f}{b.get('frame_min', float('nan')):14.6f}"
          f"{b.get('frame_max', float('nan')):14.6f}  {arrow} than upstream by {abs(d_pct):.2f}% [{tag}]")
for k in ("blockiness_grid8_collapsed_bands", "blockiness_grid32_collapsed_bands",
          "blockiness_grid8_bands", "blockiness_grid32_bands"):
    if k in ours:
        print(f"  {k:32s} {ours[k]}")
print("  NOTE: the two GATED statistics are ONE-SIDED ceilings. A render that is")
print("  smoother than upstream passes them by construction, so a sharpness that")
print("  fell further than upstream's own frame_min is a finding this gate cannot")
print("  make, and is why this panel is printed rather than summarised.")
PY
    if [ "$CMP_RC" != 0 ]; then
      # CORRECTNESS BEFORE TIMING, and this is the branch that enforces it. The
      # remaining renders are not run: a blockiness failure is the headline and
      # the wall is not reportable.
      say "  THE BLOCKINESS GATE DID NOT PASS. Renders 2..$N are NOT run and no"
      say "  timing is quotable from this job. The verdict is the result."
      echo "renders_completed=1 verdict=FAIL" >> "$OUT/PROVENANCE"
      say "evidence in $OUT"
      exit "$CMP_RC"
    fi
    say "  VERDICT PASS on the PPM frames; timing continues on this same binary"
  fi
  # The frames are 4.4 MB a render and only render 1 is judged; keep its frames
  # so a later reader can still check that something was drawn.
  #
  # KEEP_FRAMES=1 RETAINS EVERY RENDER'S FRAMES, and it exists because this
  # deletion is why the row's headline reading has no error bar (#2514).
  # `.agents/specs/ltx25-prompt-adherence.md` `## Owed` records it exactly:
  # only render 1 survives, so the S1 adherence margin of -0.7368 is n = 1, and
  # "a reading that moves by 0.74 between runs would make the S1 verdict a coin
  # toss rather than a finding". That bullet also names the two ways to close
  # it -- score inside the loop, or retain past it -- and says the choice was
  # not made. This is the SECOND one, because it is the one that does not touch
  # the verdict: the `if [ "$i" = 1 ]` compare block above is untouched, so the
  # gate this harness enforces is byte-for-byte the gate it enforced before,
  # and the extra frames are evidence a later CPU pass can score. The cost is
  # the 4.4 MB a render that the deletion was written to save, which is why it
  # is a knob and not the default.
  if [ "${KEEP_FRAMES:-0}" = 1 ]; then
    NKEPT=$(ls "$D"/frame_*.ppm 2>/dev/null | wc -l)
    echo "render_${i}_frames_retained=$NKEPT" >> "$OUT/PROVENANCE"
    say "  KEEP_FRAMES=1: retained $NKEPT frames of render $i for a later scoring pass"
  else
    [ "$i" = 1 ] || rm -f "$D"/frame_*.ppm
  fi
done

say "=== [I] the decomposition, over $N runs ==="
python3 - "$N" "$OUT" <<'PY' | tee "$OUT/SUMMARY.txt"
import json, sys, statistics
n = int(sys.argv[1]); out = sys.argv[2]
runs = []
for i in range(1, n + 1):
    d = json.load(open(f"{out}/phase-log-{i}.json"))
    agg = {}
    for r in d["phases"]:
        if r["span"]:
            k = "<span> " + r["name"]
        elif r["nested"]:
            # THE NESTED ROWS ARE PRINTED, not skipped. `conditioning.connector`
            # and `guiders.connector` each split into `.weights` and `.compute`
            # SubPhases, and `.compute` is the leaf this row's whole prediction
            # is about. A fold that dropped them would answer the question with
            # the sum of the thing and its parts.
            k = "<sub> " + r["name"]
        else:
            k = r["name"]
        agg[k] = agg.get(k, 0.0) + r["duration_seconds"]
    agg["<wall>"] = d["wall_seconds"]
    agg["<unaccounted>"] = d["unaccounted_seconds"]
    runs.append(agg)
names = sorted({k for r in runs for k in r},
               key=lambda k: -statistics.mean([r.get(k, 0.0) for r in runs]))
w0 = statistics.mean([r["<wall>"] for r in runs])
print(f"n = {n} renders, 320x192/25f/8 steps, seed 42")
print(f"{'phase':44s}" + "".join(f"{'r'+str(i+1):>10s}" for i in range(n)) +
      f"{'mean':>10s}{'spread%':>9s}{'%wall':>8s}")
for k in names:
    v = [r.get(k, 0.0) for r in runs]
    m = statistics.mean(v)
    if m < 0.01 and not k.startswith("<wall") and not k.startswith("<unacc"):
        continue
    sp = (max(v) - min(v)) / m * 100 if m else 0.0
    print(f"{k:44s}" + "".join(f"{x:10.3f}" for x in v) + f"{m:10.3f}{sp:9.2f}{m/w0*100:8.2f}")
# THE COMPARISON THIS ROW EXISTS TO MAKE, computed here so nobody has to do it
# from a table by hand. The baseline is #2296's measured mean and the
# denominator is the PINNED ORACLE'S OWN render_seconds, n = 1, which has no
# spread of its own -- a limit of the comparison and not a footnote.
BASE = 518.398   # ltx25-render-speed-parity.md, phase-log wall_seconds, n = 3
ORACLE = 93.8    # ltx2_oracle_manifest.json result.render_seconds, n = 1
PROJ = 380.0     # ltx25-connector-repair.md's projection
walls = [r["<wall>"] for r in runs]
m = statistics.mean(walls)
print()
print(f"wall mean            {m:10.3f} s   (runs: {' '.join(f'{x:.3f}' for x in walls)})")
print(f"wall spread          {(max(walls)-min(walls))/m*100:10.2f} %")
print(f"baseline #2296       {BASE:10.3f} s   change {BASE/m:.3f}x faster" if m else "")
print(f"oracle render_s      {ORACLE:10.3f} s   RATIO ours/oracle = {m/ORACLE:.3f}x  (baseline was {BASE/ORACLE:.2f}x)")
print(f"projection           {PROJ:10.3f} s   measured/projected = {m/PROJ:.3f}x")
PY

say "=== [J] done ==="
echo "renders_completed=$N verdict=PASS" >> "$OUT/PROVENANCE"
say "evidence in $OUT"
exit 0
