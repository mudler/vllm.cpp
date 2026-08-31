#!/bin/bash
# LTX25-TEXT-COND-DEVICE (#2354) -- TWO ARMS of the oracle's own request, so the
# connector's load/compute split is read on the same box, in the same lease, as
# the before/after of the repair that split blames.
#
# WHY NOT `ltx25-render-speed-repeat.sh`. That harness REFUSES to build (exit 51)
# and asserts the binary is `4b0666ee`'s, because its subject is the tree that
# took the correctness verdict. This job's subject is a CHANGE to that tree, so
# it must build -- and it builds TWICE, into two directories, because an A/B that
# reuses one build directory measures one binary twice. The identity assertion is
# inverted rather than dropped: the two arms' libraries must DIFFER, and a run
# where they hash the same is a run that measured one arm twice and exits 54.
#
# THE REQUEST IS THE MANIFEST'S, byte for byte, and identical to both neighbouring
# harnesses. A decomposition of a different request measures the request.
#
#   prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
#   320x192, 25 frames, 8 inference steps, seed 42
#
# CORRECTNESS COMES BEFORE THE SPEED RESULT, and it is checked in two forms
# because one of them is blind in the direction this change could move.
# `ltx25-render-compare.py --reference` is a ONE-SIDED blockiness bound against
# #1864's committed frames, and #1864 itself records that our render is already
# smoother than upstream's -- so a further smoothing PASSES it. The second form
# is the same-arm one: arm B's frames against arm A's, byte for byte. The repair
# under test reuses an identical weight bag rather than re-materializing it, so
# BYTE EQUALITY is the prediction, not a tolerance.
#
# EXIT STATUS.
#   0  both arms completed, the pixels are identical and the blockiness gate passed
#   23 a checkpoint sha256 that is not the manifest's, or staging failed
#   25 ltx2-gen will not exec
#   33 configure failed              34 build failed          35 missing artefact
#   36 no CUTLASS
#   38 no complete CUDA toolkit
#   39 MemAvailable is below the start floor and stayed there
#   43 a source tarball predates the tools this job runs
#   44 a CUDA unit gate FAILED       45 a CUDA unit gate binary is ABSENT
#   48 a render produced the wrong number of frames, or no audio
#   49 a render exited non-zero, or the reference frames are absent
#   51 a source tarball is absent
#   52 a phase table covers less than 99% of its own wall
#   53 a render did not run 8 steps / 32 DiT forwards
#   54 the two arms' libraries hash the SAME, so one binary was measured twice
#   55 the two arms' frames DIFFER, and the repair claimed byte equality
#   56 the blockiness gate FAILED on the changed arm
set -u

T0=$SECONDS
say() { echo "[$(date -u +%H:%M:%S) +$((SECONDS-T0))s] $*"; }

W=${W:-/workspace/ltx25-text-cond}
FULL=${FULL:-/workspace/ltx25-fullmodel}
CKROOT=${CKROOT:-/workspace/ckpt/ltx-2.5}
REFDIR=${REFDIR:-/workspace/ltx2-oracle/out/upstream_frames}
CK=/root/ckpt
N=${N:-3}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$W/run/$RUN_ID
mkdir -p "$OUT" "$CK"

PROMPT='A red fox walks slowly through a snowy pine forest at sunrise, cinematic.'
WW=320; HH=192; FRAMES=25; STEPS=8; SEED=42
export DEBIAN_FRONTEND=noninteractive

{
  echo "run_id=$RUN_ID"
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness=$0"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "renders_per_arm=$N"
  echo "geometry=${WW}x${HH}/${FRAMES}f steps=$STEPS seed=$SEED"
  echo "prompt_sha256=$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
} >> "$OUT/PROVENANCE"

# `HEARTBEAT=$!` rather than a command substitution: `.agents/oracles/ltx-2.md`
# records 2h37m of a lease lost to the substitution form holding the pipe open.
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
apt-get install -y -qq ffmpeg python3-numpy > /root/apt.log 2>&1 || say "  apt returned non-zero; probing anyway"
for t in python3 cmake ninja; do command -v "$t" >/dev/null || { echo "FATAL: no $t"; exit 38; }; done
python3 -c 'import numpy' || { echo "FATAL: no numpy, and the comparison tool needs it"; exit 38; }

say "=== [A] CUDA toolkit ==="
# The SONAME is what must exist and it is what CIFS destroys (#2220): a staged
# copy carries only `libcudart.so.13.3.29`, `nvcc` compiles happily against
# headers, and the failure lands 21 minutes later at the consumer link.
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
  soname_ok "$1/targets/sbsa-linux/lib" libcudart && soname_ok "$1/targets/sbsa-linux/lib" libcublasLt
}
TKLIB=""
for c in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
  if need_ok "$c"; then TKLIB=$c; break; fi
done
if [ -z "$TKLIB" ] && [ -d /workspace/a3/cuda-staged ]; then
  say "  staging the toolkit from /workspace/a3/cuda-staged"
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
[ -n "$TKLIB" ] || { echo "FATAL: no CUDA toolkit whose libcudart/libcublasLt SONAME links resolve (#2220)"; exit 38; }
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "  toolkit $TKLIB, $(nvcc --version | tail -1)"
for s in libcudart libcublasLt; do
  t=$(readlink -f "$TKLIB/targets/sbsa-linux/lib/$s.so")
  m=$(basename "$t"); m=${m#*.so.}; m=${m%%.*}
  echo "toolkit_soname $s.so.$m -> $(basename "$t")" >> "$OUT/PROVENANCE"
done

say "=== [C] CUTLASS ==="
CUT=""
for c in /cutlass /workspace/cutlass /root/cutlass; do
  [ -f "$c/include/cutlass/cutlass.h" ] && CUT=$c && break
done
if [ -z "$CUT" ] && [ -s /workspace/cutlass-v4.5.0.tar.gz ]; then
  # NO `--strip-components`: the staged tarball's first member is `include/`.
  mkdir -p /root/cutlass && tar xzf /workspace/cutlass-v4.5.0.tar.gz -C /root/cutlass
  [ -f /root/cutlass/include/cutlass/cutlass.h ] && CUT=/root/cutlass
fi
[ -n "$CUT" ] || { echo "FATAL: no CUTLASS"; exit 36; }

say "=== [D] checkpoints, staged and CHECKED AGAINST THE MANIFEST ==="
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
# 24 GiB of slack rather than 8: this job carries TWO build directories beside
# the checkpoints, and an ENOSPC here writes FALSE policy refusals into records.
NEED_K=$(( (42018190584 + 1452269922 + 364866540 + 26263858182) / 1024 + 25165824 ))
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

# ── one arm: unpack, CLEAN build, unit gate, N renders ──────────────────────
declare -A LIBSHA_OF

run_arm() {
  local arm=$1
  local SRC=/root/src-$arm BLD=/root/build-$arm BIN=/root/bin-$arm
  local ARMOUT=$OUT/$arm
  mkdir -p "$ARMOUT" "$BIN"

  say "=== [$arm/1] source ==="
  [ -s "$W/src$arm.tar.gz" ] || { echo "FATAL: no $W/src$arm.tar.gz"; exit 51; }
  rm -rf "$SRC"; mkdir -p "$SRC"
  tar xzf "$W/src$arm.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack src$arm"; exit 51; }
  local ssha tsha
  ssha=$(cat "$W/src$arm.sha" 2>/dev/null)
  tsha=$(sha256sum "$W/src$arm.tar.gz" | awk '{print $1}')
  { echo "arm_${arm}_source_sha=$ssha"; echo "arm_${arm}_source_tarball_sha256=$tsha"; } >> "$OUT/PROVENANCE"
  [ -s "$SRC/scripts/ltx25-render-compare.py" ] || { echo "FATAL: arm $arm has no comparison tool"; exit 43; }
  grep -q -- '--reference' "$SRC/scripts/ltx25-render-compare.py" || {
    echo "FATAL: arm $arm's comparison tool has no --reference"; exit 43; }

  say "=== [$arm/2] CLEAN build (two arms, two directories) ==="
  rm -rf "$BLD"
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
        -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" > "$ARMOUT/configure.log" 2>&1 \
        || { echo "FATAL: arm $arm configure failed"; tail -30 "$ARMOUT/configure.log"; exit 33; }
  # NAMED TARGETS ONLY, `-j 4`. A bare `ninja` links every test binary and writes
  # 9.4 GiB, and unconstrained parallelism has OOM-rebooted this box.
  ninja -C "$BLD" -j 4 ltx2-gen test_ltx2_device > "$ARMOUT/build.log" 2>&1 \
        || { echo "FATAL: arm $arm build failed"; tail -40 "$ARMOUT/build.log"; exit 34; }
  local GEN LIB TD
  GEN=$(find "$BLD" -name ltx2-gen -type f | head -1)
  LIB=$(find "$BLD" -name 'libvllm.so.0.0.3' -type f | head -1)
  TD=$(find "$BLD" -name test_ltx2_device -type f | head -1)
  for f in "$GEN" "$LIB" "$TD"; do [ -s "$f" ] || { echo "FATAL: arm $arm missing build artefact"; exit 35; }; done
  cp -f "$GEN" "$LIB" "$TD" "$BIN"/
  chmod 0755 "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib"
  local BINSHA LIBSHA
  BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
  LIBSHA=$(sha256sum "$BIN/libvllm.so.0.0.3" | awk '{print $1}')
  LIBSHA_OF[$arm]=$LIBSHA
  { echo "arm_${arm}_binary_sha256=$BINSHA"; echo "arm_${arm}_library_sha256=$LIBSHA"; } >> "$OUT/PROVENANCE"
  say "  arm $arm ltx2-gen=$BINSHA lib=$LIBSHA"
  "$BIN/ltx2-gen" --help >/dev/null 2>&1 || {
    echo "FATAL: arm $arm ltx2-gen will not exec"; ldd "$BIN/ltx2-gen" | head; exit 25; }

  say "=== [$arm/3] the CUDA unit gate, BEFORE any timing ==="
  [ -x "$BIN/test_ltx2_device" ] || { echo "FATAL: arm $arm has no unit gate binary"; exit 45; }
  "$BIN/test_ltx2_device" > "$ARMOUT/test_ltx2_device.log" 2>&1 || {
    echo "FATAL: arm $arm's CUDA unit gate FAILED"; tail -30 "$ARMOUT/test_ltx2_device.log"; exit 44; }
  # A doctest binary that skips everything also exits 0, so the counts are the
  # record and not the status.
  grep -E 'test cases:|assertions:' "$ARMOUT/test_ltx2_device.log" | sed "s/^/arm_${arm}_unit /" | tee -a "$OUT/PROVENANCE"

  say "=== [$arm/4] $N renders ==="
  local i D LOG t SEC RC NF OBS FWD CLK
  for i in $(seq 1 "$N"); do
    D=$ARMOUT/r$i
    rm -rf "$D"; mkdir -p "$D"
    LOG=$ARMOUT/render-$i.log
    say "  --- arm $arm render $i of $N ---"
    { echo "== arm $arm render $i pre-state"; contention; } | tee -a "$OUT/PROVENANCE"

    # `exec`, so the subshell is REPLACED by the sampler and the TERM below
    # reaches it. Without it `$!` is the subshell and the sampler is orphaned
    # into the next render (#2305).
    ( cd "$SRC" && exec python3 -m tools.bench.gpu_clock_state sample \
        --output "$ARMOUT/clock-$i.jsonl" --summary "$ARMOUT/clock-$i.json" --interval 2 \
        > "$ARMOUT/clock-$i.stdout" 2>&1 ) &
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
    [ -s "$ARMOUT/clock-$i.json" ] || say "  WARNING: no clock summary for arm $arm render $i (#2305)"

    say "  arm $arm render $i rc=$RC in ${SEC}s"
    echo "arm_${arm}_render_${i}_rc=$RC arm_${arm}_render_${i}_seconds=$SEC" >> "$OUT/PROVENANCE"
    { echo "== arm $arm render $i post-state"; contention; } >> "$OUT/PROVENANCE"
    [ "$RC" = 0 ] || { echo "FATAL: arm $arm render $i exited $RC"; tail -40 "$LOG"; exit 49; }

    NF=$(ls "$D"/frame_*.ppm 2>/dev/null | wc -l)
    [ "$NF" = "$FRAMES" ] && [ -s "$D/audio.wav" ] || {
      echo "FATAL: arm $arm render $i is incomplete ($NF of $FRAMES frames)"; tail -40 "$LOG"; exit 48; }
    [ -s "$D/phase-log.json" ] || { echo "FATAL: arm $arm render $i wrote no phase table"; exit 48; }

    OBS=$(grep -oE 'step [0-9]+/[0-9]+' "$LOG" | awk -F/ '{print $2}' | sort -u | tr '\n' ',' )
    FWD=$(grep -cE 'step [0-9]+/[0-9]+' "$LOG")
    echo "arm_${arm}_render_${i}_steps_observed={${OBS%,}} arm_${arm}_render_${i}_dit_forwards=$FWD" >> "$OUT/PROVENANCE"
    [ "${OBS%,}" = "$STEPS" ] || { echo "FATAL: arm $arm render $i observed steps {${OBS%,}}, not $STEPS"; exit 53; }

    python3 - "$D/phase-log.json" <<'PY' || exit 52
import json, sys
d = json.load(open(sys.argv[1]))
wall = d["wall_seconds"]; un = d["unaccounted_seconds"]
frac = un / wall if wall else 1.0
print(f"  coverage wall={wall:.3f}s unaccounted={un:.4f}s ({frac*100:.3f}%)")
sys.exit(0 if frac < 0.01 else 1)
PY
    cp -f "$D/phase-log.json" "$ARMOUT/phase-log-$i.json"
    # Render 1's frames are KEPT on both arms: they are the pixel comparison.
    [ "$i" = 1 ] || rm -f "$D"/frame_*.ppm
  done
  # The comparison tool and the reference live in the source; remember arm A's.
  echo "$SRC" > "$ARMOUT/SRCDIR"
}

run_arm A
run_arm B

say "=== [E] the two arms are two BINARIES ==="
# An A/B that measured one binary twice is the failure this asserts against.
if [ "${LIBSHA_OF[A]}" = "${LIBSHA_OF[B]}" ]; then
  echo "FATAL: both arms' libvllm hash ${LIBSHA_OF[A]}; one binary was measured twice"
  exit 54
fi
say "  A=${LIBSHA_OF[A]}"
say "  B=${LIBSHA_OF[B]}"

say "=== [F] SAME-ARM PIXELS: arm B against arm A, byte for byte ==="
# The repair reuses an identical weight bag instead of re-materializing it, so
# every float downstream is the same float. Byte equality is the PREDICTION.
# This is also the check the blockiness gate cannot make: that gate is one-sided
# and #1864 records our render as already smoother than upstream's, so a further
# smoothing passes it.
PIXDIFF=0
for f in "$OUT/A/r1"/frame_*.ppm; do
  b="$OUT/B/r1/$(basename "$f")"
  if ! cmp -s "$f" "$b"; then PIXDIFF=$((PIXDIFF+1)); say "  DIFFERS: $(basename "$f")"; fi
done
cmp -s "$OUT/A/r1/audio.wav" "$OUT/B/r1/audio.wav" || { PIXDIFF=$((PIXDIFF+1)); say "  DIFFERS: audio.wav"; }
echo "pixel_files_differing=$PIXDIFF" >> "$OUT/PROVENANCE"
say "  files differing: $PIXDIFF"

say "=== [G] the blockiness gate on BOTH arms (#1864's committed reference) ==="
SRCA=$(cat "$OUT/A/SRCDIR"); SRCB=$(cat "$OUT/B/SRCDIR")
[ -d "$REFDIR" ] || { echo "FATAL: the reference frames are not at $REFDIR"; exit 49; }
python3 "$SRCA/scripts/ltx25-render-compare.py" --a "$OUT/A/r1" --label-a armA \
  --reference "$REFDIR" --json "$OUT/A-vs-reference.json" > "$OUT/compare-A.log" 2>&1
CMP_A=$?
python3 "$SRCB/scripts/ltx25-render-compare.py" --a "$OUT/B/r1" --label-a armB \
  --reference "$REFDIR" --json "$OUT/B-vs-reference.json" > "$OUT/compare-B.log" 2>&1
CMP_B=$?
say "  arm A vs reference: exit $CMP_A"
say "  arm B vs reference: exit $CMP_B"
tail -20 "$OUT/compare-B.log"
echo "compare_exit_A=$CMP_A compare_exit_B=$CMP_B" >> "$OUT/PROVENANCE"

say "=== [H] the decomposition, per arm, over $N runs ==="
python3 - "$N" "$OUT" <<'PY' | tee "$OUT/SUMMARY.txt"
import json, sys, statistics
n = int(sys.argv[1]); out = sys.argv[2]

def load(arm):
    runs = []
    for i in range(1, n + 1):
        d = json.load(open(f"{out}/{arm}/phase-log-{i}.json"))
        agg = {}
        for r in d["phases"]:
            if r["span"]:
                continue
            # NESTED records are the sub-phases this row added. They are kept and
            # PREFIXED rather than dropped, because they are the whole point --
            # and they are never mixed into the leaf sum, which is what excludes
            # them from `sum_leaf_seconds` upstream of here.
            key = ("~" if r["nested"] else "") + r["name"]
            agg[key] = agg.get(key, 0.0) + r["duration_seconds"]
        for r in d["phases"]:
            if r["span"]:
                agg["<span>" + r["name"]] = r["duration_seconds"]
        agg["<wall>"] = d["wall_seconds"]
        agg["<unaccounted>"] = d["unaccounted_seconds"]
        runs.append(agg)
    return runs

tables = {a: load(a) for a in ("A", "B")}
names = sorted({k for rs in tables.values() for r in rs for k in r},
               key=lambda k: -statistics.mean([r.get(k, 0.0) for r in tables["A"]] or [0.0]))
wA = statistics.mean([r["<wall>"] for r in tables["A"]])
wB = statistics.mean([r["<wall>"] for r in tables["B"]])
print(f"n = {n} renders per arm, 320x192/25f/8 steps, seed 42")
print("A = the split instrument only.  B = A plus one materialization per render.")
print("`~` prefixes a NESTED sub-leaf: it decomposes the leaf above it and is")
print("never added into sum_leaf_seconds.\n")
hdr = f"{'phase':38s}" + "".join(f"{'A r'+str(i+1):>10s}" for i in range(n)) + \
      f"{'A mean':>10s}{'A sp%':>7s}" + "".join(f"{'B r'+str(i+1):>10s}" for i in range(n)) + \
      f"{'B mean':>10s}{'B sp%':>7s}{'B-A':>10s}"
print(hdr)
for k in names:
    va = [r.get(k, 0.0) for r in tables["A"]]
    vb = [r.get(k, 0.0) for r in tables["B"]]
    ma = statistics.mean(va); mb = statistics.mean(vb)
    if max(ma, mb) < 0.01 and k[0] != "<":
        continue
    sa = (max(va) - min(va)) / ma * 100 if ma else 0.0
    sb = (max(vb) - min(vb)) / mb * 100 if mb else 0.0
    print(f"{k:38s}" + "".join(f"{x:10.3f}" for x in va) + f"{ma:10.3f}{sa:7.2f}" +
          "".join(f"{x:10.3f}" for x in vb) + f"{mb:10.3f}{sb:7.2f}{mb-ma:10.3f}")
print(f"\nwall  A {wA:.3f}s   B {wB:.3f}s   delta {wB-wA:+.3f}s ({(wB-wA)/wA*100:+.2f}%)")
print("Oracle render_seconds = 93.8 (n = 1, its own load included).")
print(f"ratio  A {wA/93.8:.2f}x   B {wB/93.8:.2f}x")
PY

say "=== [I] verdict ==="
echo "renders_completed_per_arm=$N" >> "$OUT/PROVENANCE"
say "evidence in $OUT"
[ "$PIXDIFF" = 0 ] || { echo "FATAL: $PIXDIFF files differ between the arms; the repair claimed byte equality"; exit 55; }
[ "$CMP_B" = 0 ] || { echo "FATAL: the blockiness gate FAILED on the changed arm (exit $CMP_B)"; exit 56; }
exit 0
