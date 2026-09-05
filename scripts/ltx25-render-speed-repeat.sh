#!/bin/bash
# LTX25-RENDER-SPEED-PARITY (#2296) -- N renders of the ORACLE'S OWN REQUEST, on
# the binary that took the correctness reading, so the phase decomposition has a
# spread beside it instead of one number.
#
# THE PIN ADVANCED ON 2026-09-01, AND IT WAS EARNED RATHER THAN EDITED
# (LTX25-RENDER-CONFIRM, #2457). It named `4b0666ee`'s pre-repair binary, which
# is why `LTX25-CONNECTOR-REPAIR` could only PROJECT the post-repair render. The
# way past that guard is not to relax it: `scripts/ltx25-render-confirm.sh` built
# the head, re-took #1864's blockiness verdict on THAT binary, and only then
# timed it. `rc` job 93a60151 returned VERDICT PASS with margins +0.113 and
# +0.124 on the two gated ratios, and 302.954 s at n = 3. The digests below are
# that binary's. The guard itself is unchanged: still exit 51 on anything else,
# still never builds.
#
# WHAT THIS IS NOT. It is not `ltx25-oracle-absolute-render.sh` with a loop
# around it. That job's subject is the PICTURE and it renders once; this job's
# subject is the WALL, and a wall quoted from one run is an anecdote. It also
# never builds: it REFUSES on a cache miss (51), because a rebuilt binary is a
# different measurement subject and the whole point is to time a tree that has
# itself established `VERDICT PASS`.
#
# THE REQUEST IS THE MANIFEST'S, byte for byte, and identical to the neighbouring
# harness's. A decomposition of a different request measures the request.
#
#   prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
#   320x192, 25 frames, 8 inference steps, seed 42
#   the four BF16 checkpoints of tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json
#
# WHY THE CLOCK MATTERS LESS HERE THAN USUAL, AND IS STILL RECORDED. #2296
# measures that 61.8% of this render is host-side f32 work on a hard-coded CPU
# queue (ltx2_video.cpp:2344, :3074), so the axis that transfers into these
# numbers is HOST load, not the SM clock. `.agents/benchmarking.md` still says a
# number is quotable only with the clock it was taken at, and a lease cannot pin
# one (`nvidia-smi -lgc` -> LGC_RC=4), so this job SAMPLES a window per render
# through the one helper that folds it, and records loadavg and MemAvailable
# beside it because those are the terms this particular render is sensitive to.
#
# EXIT STATUS.
#   0  every render completed and every precondition held
#   23 a checkpoint sha256 that is not the manifest's, or staging failed
#   25 ltx2-gen will not exec
#   38 no complete CUDA toolkit
#   39 MemAvailable is below the start floor and stayed there
#   44 the CUDA unit gate FAILED     45 the CUDA unit gate BINARY IS ABSENT
#   48 a render produced the wrong number of frames, or no audio
#   49 a render exited non-zero
#   51 the build cache is absent, or its binary is not the VERIFIED one
#   52 a phase table covers less than 99% of its own wall
#   53 a render did not run 8 steps / 32 DiT forwards
set -u

T0=$SECONDS
say() { echo "[$(date -u +%H:%M:%S) +$((SECONDS-T0))s] $*"; }

# THE CACHE MOVED WITH THE PIN. The verified artefacts now live where the job
# that verified them wrote them, so the digests below and the files they name
# cannot drift apart.
W=${W:-/workspace/ltx25-render-confirm}       # the cache and the source this row reuses
OUTROOT=${OUTROOT:-/workspace/ltx25-render-speed}
FULL=${FULL:-/workspace/ltx25-fullmodel}
CKROOT=${CKROOT:-/workspace/ckpt/ltx-2.5}
SRC=/root/src
BIN=/root/speedbin
CK=/root/ckpt
CACHE=$W/confirm-bin
N=${N:-3}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$OUTROOT/run/$RUN_ID
mkdir -p "$OUT" "$CK" "$BIN"

# THE MEASUREMENT SUBJECT, asserted rather than assumed. These digests are `rc`
# job 93a60151's own PROVENANCE, the run whose comparison returned VERDICT PASS
# against #1864's reference on the SAME binary it then timed. A harness that
# rebuilds and then reports a decomposition has timed a binary nobody verified.
#
# The previous values were 4b0666ee's -- ltx2-gen 7b1f4367, libvllm 9e3dc6f4,
# source 0002ddfba -- and they are recorded here because a pin that changes
# without saying what it replaced is a pin nobody can audit.
WANT_BIN_SHA=${WANT_BIN_SHA:-600cf798c48ebabebc1fa25fb4891fe0b550f31f995501105aea856cced4c54d}
WANT_LIB_SHA=${WANT_LIB_SHA:-c4692db9025899463122b64b26cc40486dd445d37f9b7835e06cbf26daf83492}
WANT_SRC_SHA=${WANT_SRC_SHA:-790c582bbba45ab0f7b74aafee361e4557a84bf2}

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
# form holding the pipe open.
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
apt-get install -y -qq python3-numpy > /root/apt.log 2>&1 || say "  apt returned non-zero; probing anyway"
command -v python3 >/dev/null || { echo "FATAL: no python3"; exit 38; }

say "=== [A] CUDA runtime, for LD_LIBRARY_PATH only ==="
# NO BUILD HAPPENS HERE, so nvcc is not required -- what IS required is the
# runtime the cached library links against. #2220 is why this is checked by
# SONAME rather than by the toolkit's existence: a staged toolkit off CIFS
# carries no symlinks, `libcudart.so.13` is what `ld.so` resolves, and a
# `Found CUDAToolkit` that cannot be linked against is the failure that cost
# 21 minutes there.
TKLIB=""
for c in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
  [ -f "$c/targets/sbsa-linux/lib/libcudart.so.13" ] && TKLIB=$c && break
done
if [ -z "$TKLIB" ]; then
  for c in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
    [ -d "$c/targets/sbsa-linux/lib" ] || continue
    ( cd "$c/targets/sbsa-linux/lib" && for f in *.so.*.*; do
        b=${f%%.so.*}; maj=${f#*.so.}; maj=${maj%%.*}
        ln -sf "$f" "$b.so"; ln -sf "$f" "$b.so.$maj"; done ) 2>/dev/null
    [ -f "$c/targets/sbsa-linux/lib/libcudart.so.13" ] && TKLIB=$c && break
  done
fi
[ -n "$TKLIB" ] || { echo "FATAL: no libcudart.so.13 anywhere the cached library can link against"; exit 38; }
say "  runtime $TKLIB"
{ echo "cuda_runtime_dir=$TKLIB"
  ls -l "$TKLIB/targets/sbsa-linux/lib/libcudart.so.13" | sed 's/^/toolkit_soname /'; } >> "$OUT/PROVENANCE"

say "=== [B] source, for the clock helper and the phase reader ==="
[ -s "$W/src.tar.gz" ] || { echo "FATAL: no $W/src.tar.gz"; exit 51; }
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 51; }
GOT_SRC_SHA=$(cat "$W/src.sha" 2>/dev/null)
echo "source_sha=$GOT_SRC_SHA source_tarball_sha256=$(sha256sum "$W/src.tar.gz" | awk '{print $1}')" >> "$OUT/PROVENANCE"

say "=== [C] the binary: TAKEN FROM CACHE, NEVER BUILT ==="
for f in ltx2-gen libvllm.so.0.0.3 test_ltx2_device; do
  [ -s "$CACHE/$f" ] || { echo "FATAL: $CACHE/$f is absent; this job does not build (see the header)"; exit 51; }
done
[ "$GOT_SRC_SHA" = "$WANT_SRC_SHA" ] || {
  echo "FATAL: the staged source is $GOT_SRC_SHA, this measurement is about $WANT_SRC_SHA"; exit 51; }
[ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SRC_SHA" ] || {
  echo "FATAL: the cache was built from $(cat "$CACHE/SRC_SHA" 2>/dev/null), not $WANT_SRC_SHA"; exit 51; }
cp -f "$CACHE/ltx2-gen" "$CACHE/libvllm.so.0.0.3" "$CACHE/test_ltx2_device" "$BIN"/
chmod 0755 "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device"
( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
LIBSHA=$(sha256sum "$BIN/libvllm.so.0.0.3" | awk '{print $1}')
{ echo "binary_sha256=$BINSHA"; echo "library_sha256=$LIBSHA"; } >> "$OUT/PROVENANCE"
# BOTH, AND THE LIBRARY IS THE ONE THAT MATTERS (#1881): `ltx2-gen` is a small
# launcher whose digest has been byte-identical across builds hundreds of commits
# apart while the libraries differed by megabytes. Checking only the launcher
# would pass a completely different engine.
[ "$BINSHA" = "$WANT_BIN_SHA" ] || { echo "FATAL: ltx2-gen is $BINSHA, the verified binary's is $WANT_BIN_SHA"; exit 51; }
[ "$LIBSHA" = "$WANT_LIB_SHA" ] || { echo "FATAL: libvllm is $LIBSHA, the verified binary's is $WANT_LIB_SHA"; exit 51; }
say "  identity holds: ltx2-gen=$BINSHA lib=$LIBSHA"
"$BIN/ltx2-gen" --help >/dev/null 2>&1 || {
  echo "FATAL: ltx2-gen will not exec (126 = no exec bit, 127 = missing lib)"; ldd "$BIN/ltx2-gen" | head; exit 25; }

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
NEED_K=$(( (42018190584 + 1452269922 + 364866540 + 26263858182) / 1024 + 8388608 ))
FREE_K=$(df -k --output=avail /root | tail -1)
# STAGING IS NOT OPTIONAL FOR THIS ROW. The neighbouring harness falls back to
# reading over CIFS when /root is short; here that fallback would put a 34-83
# MiB/s network read inside the `load` phase this job is decomposing, and the
# load number would be about the NAS. Refuse instead.
[ "$FREE_K" -gt "$NEED_K" ] || {
  echo "FATAL: /root has ${FREE_K}K free against ${NEED_K}K needed, and reading the checkpoints over CIFS would put the NAS inside the load phase this job measures"; exit 23; }
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

say "=== [E] the CUDA unit gate, BEFORE any timing ==="
# Correctness comes before a performance result, and a doctest binary that skips
# everything also exits 0 -- so the case and assertion counts are recorded, not
# just the status.
"$BIN/test_ltx2_device" > "$OUT/test_ltx2_device.log" 2>&1 || {
  echo "FATAL: the CUDA unit gate FAILED"; tail -30 "$OUT/test_ltx2_device.log"; exit 44; }
grep -E 'test cases:|assertions:' "$OUT/test_ltx2_device.log" | tee -a "$OUT/PROVENANCE"

say "=== [F] ${N} renders ==="
RC_ANY=0
for i in $(seq 1 "$N"); do
  D=$OUT/r$i
  rm -rf "$D"; mkdir -p "$D"
  LOG=$OUT/render-$i.log
  say "  --- render $i of $N ---"
  { echo "== render $i pre-state"; contention; } | tee -a "$OUT/PROVENANCE"

  # The clock window is opened by the ARM and closed after it
  # (.agents/benchmarking.md: a window that does not span the work it describes
  # measures nothing). The helper writes its summary only when the sampler
  # STOPS, so nothing reads it before then.
  # `exec`, and it is the fix for a defect this harness shipped with (#2305).
  # Without it `$!` is the SUBSHELL's pid, the TERM below kills the subshell, and
  # the python sampler is orphaned -- it keeps sampling into the NEXT render and
  # never writes its summary, because the helper writes that only when the
  # sampler STOPS. The observable was a widening window (757, 512, 257 samples
  # for three equal renders) and three absent `clock-N.json` files. With `exec`
  # the subshell is REPLACED by python, so `$!` is the sampler and the signal
  # reaches it.
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
  # THE SUMMARY IS THE INSTRUMENT'S OWN PRECONDITION, so its absence is reported
  # rather than discovered later in a table that has no clock beside it. Not
  # fatal: this render is 88% host-side (#2296) and a missing clock fold does not
  # void it. Silence would.
  [ -s "$OUT/clock-$i.json" ] || say "  WARNING: the clock sampler wrote no summary for render $i (#2305)"

  say "  render $i rc=$RC in ${SEC}s"
  echo "render_${i}_rc=$RC render_${i}_seconds=$SEC" >> "$OUT/PROVENANCE"
  { echo "== render $i post-state"; contention; } >> "$OUT/PROVENANCE"
  [ "$RC" = 0 ] || { echo "FATAL: render $i exited $RC"; tail -40 "$LOG"; exit 49; }

  NF=$(ls "$D"/frame_*.ppm 2>/dev/null | wc -l)
  [ "$NF" = "$FRAMES" ] && [ -s "$D/audio.wav" ] || {
    echo "FATAL: render $i is incomplete ($NF of $FRAMES frames)"; tail -40 "$LOG"; exit 48; }
  [ -s "$D/phase-log.json" ] || { echo "FATAL: render $i wrote no phase table"; exit 48; }

  # THE SCHEDULE IS READ BACK FROM THE SAMPLER'S OWN LINES, not from the flag.
  # `steps_observed` is the set of distinct denominators of `step k/M`; a phase
  # carrying its own sigmas with `allow_request_sigmas` would keep its schedule,
  # ignore the override and put a different denoise budget into this average.
  OBS=$(grep -oE 'step [0-9]+/[0-9]+' "$LOG" | awk -F/ '{print $2}' | sort -u | tr '\n' ',' )
  FWD=$(grep -cE 'step [0-9]+/[0-9]+' "$LOG")
  echo "render_${i}_steps_observed={${OBS%,}} render_${i}_dit_forwards=$FWD" >> "$OUT/PROVENANCE"
  [ "${OBS%,}" = "$STEPS" ] || { echo "FATAL: render $i observed steps {${OBS%,}}, not $STEPS"; exit 53; }

  # A TABLE THAT STOPPED COVERING THE RENDER IS REFUSED RATHER THAN SUMMED.
  # `unaccounted_seconds` is time no phase named; averaging a run whose residue
  # grew would put that unnamed time into whichever phase sits next to it.
  python3 - "$D/phase-log.json" <<'PY' || exit 52
import json, sys
d = json.load(open(sys.argv[1]))
wall = d["wall_seconds"]; un = d["unaccounted_seconds"]
frac = un / wall if wall else 1.0
print(f"  coverage wall={wall:.3f}s unaccounted={un:.4f}s ({frac*100:.3f}%)")
sys.exit(0 if frac < 0.01 else 1)
PY
  cp -f "$D/phase-log.json" "$OUT/phase-log-$i.json"
  # The frames are 4.4 MB a render and this row judges no picture; the phase
  # table and the log are the evidence. Kept: one render's frames, so a later
  # reader can still check that something was drawn.
  [ "$i" = 1 ] || rm -f "$D"/frame_*.ppm
done

say "=== [G] the decomposition, over $N runs ==="
python3 - "$N" "$OUT" <<'PY' | tee "$OUT/SUMMARY.txt"
import json, sys, statistics
n = int(sys.argv[1]); out = sys.argv[2]
runs = []
for i in range(1, n + 1):
    d = json.load(open(f"{out}/phase-log-{i}.json"))
    agg = {}
    for r in d["phases"]:
        if r["span"] or r["nested"]:
            continue
        agg[r["name"]] = agg.get(r["name"], 0.0) + r["duration_seconds"]
    # The two spans are kept separately: they are the row's headline numbers and
    # they are NOT leaves, so they are read by name rather than summed.
    for r in d["phases"]:
        if r["span"]:
            agg["<span>" + r["name"]] = r["duration_seconds"]
    agg["<wall>"] = d["wall_seconds"]
    agg["<unaccounted>"] = d["unaccounted_seconds"]
    runs.append(agg)
names = sorted({k for r in runs for k in r}, key=lambda k: -statistics.mean([r.get(k, 0.0) for r in runs]))
w0 = statistics.mean([r["<wall>"] for r in runs])
print(f"n = {n} renders, 320x192/25f/8 steps, seed 42")
print(f"{'phase':34s}" + "".join(f"{'r'+str(i+1):>10s}" for i in range(n)) +
      f"{'mean':>10s}{'spread%':>9s}{'%wall':>8s}")
for k in names:
    v = [r.get(k, 0.0) for r in runs]
    m = statistics.mean(v)
    if m < 0.01 and k[0] != "<":
        continue
    sp = (max(v) - min(v)) / m * 100 if m else 0.0
    print(f"{k:34s}" + "".join(f"{x:10.3f}" for x in v) + f"{m:10.3f}{sp:9.2f}{m/w0*100:8.2f}")
PY

say "=== [H] done ==="
echo "renders_completed=$N" >> "$OUT/PROVENANCE"
say "evidence in $OUT"
exit 0
