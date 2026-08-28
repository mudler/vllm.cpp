#!/bin/bash
# LTX25-ORACLE-ABSOLUTE (#1854) -- ONE render, at #1864's exact request, judged
# against #1864's own reference.
#
# This is the job the row's `## Gates` item 5 names, and it is the only part of
# the row that needs a GPU. Everything else -- the criterion, the bound's
# derivation, the tool and its red-before suite -- is committed before this runs,
# so this file reads a number rather than choosing one.
#
# WHY THE FILE IS COMMITTED RATHER THAN TYPED INTO A LEASE. `oracle-ltx-2-pin.md`
# made the same call for the same reason: four earlier attempts at that row ran an
# inline CLI inside a throwaway job, and the manifest they produced was a claim
# about a script that no longer existed. The file that produced the evidence is
# the file that lands.
#
# THE REQUEST IS NOT FREE-FORM AND NOT A DEFAULT. Every value below is read off
# `tests/parity/goldens/ltx2_oracle/ltx2_oracle_manifest.json`, which records what
# upstream was asked at `fd4ded7f`. Matching it is the whole point: a comparison
# against a reference rendered from another request measures the request.
#
#   prompt   "A red fox walks slowly through a snowy pine forest at sunrise, cinematic."
#   320x192, 25 frames, 8 inference steps, seed 42
#   transformer  ltx-2.5-22b-dev-transformer-bf16.safetensors
#   text encoder gemma4-12b-with-proj-ltx-2.5-bf16.safetensors   <-- BF16, NOT nvfp4
#   video vae    ltx-2.5-video-vae-conv-bf16.safetensors          <-- the CONV vae
#   audio vae    ltx-2.5-audio-vae-bf16.safetensors
#
# THE TEXT ENCODER IS THE ONE THING THAT DIFFERS FROM EVERY PREVIOUS RENDER HERE.
# `ltx25-dit-attn-flash-pixel-ab.sh` passes the NVFP4 torchao tower, and
# `docs/models/ltx-2-5.md` says in as many words that "this project's own engine
# does not load" the bf16 one. The loader disagrees with that sentence --
# `ltx2_text_encoder.cpp` takes a BF16 tower module directly, and the bf16 file
# carries its own `__metadata__["gemma_config"]` so `--encoder-config` must be
# OMITTED beside it -- but nothing in this tree has ever run it. Phase [G] is the
# first time, and a refusal there is a RESULT to report, not a reason to swap in
# the NVFP4 tower: comparing a quantized arm against a bf16 reference would
# measure the arm.
#
# --steps 8 EXISTS BECAUSE OF #2130, filed and fixed in the same flow. Without it
# `one_stage` on model version 2.5 runs 30, and a 30-step render against an
# 8-step reference carries a 3.75x denoise-budget confound in the direction that
# flatters us.
#
# EXIT STATUS. 0 and 1 are the comparison's own verdict and this job exits with
# it: 0 the render is no worse than the reference on both blockiness ratios,
# 1 a check failed. 2 is the comparison's UNREADABLE and is never a pass -- with
# a digest-verified reference it means the reference itself was refused.
# Everything else is a refusal before any verdict exists:
#   23 checkpoint staging or a sha256 that is not the manifest's
#   25 ltx2-gen will not exec        31 source tarball
#   33 configure   34 build   35 artefacts   36 no CUTLASS
#   38 no complete CUDA toolkit
#   39 MemAvailable is below the start floor and stayed there
#   43 the comparison tool is not in this source
#   44 the CUDA unit gate FAILED     45 the CUDA unit gate BINARY IS ABSENT
#   48 the render produced the wrong number of frames, or no audio
#   49 the reference is not where this job expects it
set -u

T0=$SECONDS
say() { echo "[$(date -u +%H:%M:%S) +$((SECONDS-T0))s] $*"; }

W=${W:-/workspace/ltx25-oracle-absolute}
FULL=${FULL:-/workspace/ltx25-fullmodel}          # the DiT and the two VAEs
CKROOT=${CKROOT:-/workspace/ckpt/ltx-2.5}         # the BF16 text encoder
REFDIR=${REFDIR:-/workspace/ltx2-oracle/out/upstream_frames}
SRC=/root/src
BLD=/root/build-absref
BIN=/root/absrefbin
CK=/root/ckpt
CACHE=$W/absref-bin
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$W/run/$RUN_ID
mkdir -p "$OUT" "$CK" "$BIN" "$CACHE"

# THE REQUEST, byte for byte from the manifest.
PROMPT='A red fox walks slowly through a snowy pine forest at sunrise, cinematic.'
WW=320; HH=192; FRAMES=25; STEPS=8; SEED=42
TOK=$(( (WW/32) * (HH/32) * (((FRAMES-1)/8) + 1) ))

export DEBIAN_FRONTEND=noninteractive

{
  echo "run_id=$RUN_ID"
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness=$0"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "geometry=${WW}x${HH}/${FRAMES}f steps=$STEPS seed=$SEED video_tokens=$TOK"
  echo "prompt_sha256=$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
} >> "$OUT/PROVENANCE"

# HEARTBEAT ON STDOUT every 120 s, and `HEARTBEAT=$!` rather than a command
# substitution. `--idle-timeout` counts the job's own stdout, a build redirected
# to a file is silent, and `.agents/oracles/ltx-2.md` records 2h37m of a lease
# lost to a heartbeat written as `HB=$(heartbeat setup)`, which held the
# substitution open.
( while true; do sleep 120; echo "[hb +$((SECONDS-T0))s] alive"; done ) &
HEARTBEAT=$!
cleanup() { kill "$HEARTBEAT" 2>/dev/null; }
trap cleanup EXIT
# SIGNALS TOO, not only EXIT: `rc` reclaiming a device sends SIGTERM, and a bare
# EXIT trap left the heartbeat orphaned.
for sig in HUP INT TERM; do
  trap "cleanup; exit \$((128 + \$(kill -l $sig)))" "$sig"
done

say "=== [0] the box ==="
uname -m; nproc; free -g
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>&1 | head -3
df -h / /root /workspace 2>&1 | head -6

mem_avail_gib() {
  awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo
}

# THE START FLOOR IS DERIVED, NOT COPIED. `ltx25-dit-attn-flash-pixel-ab.sh` uses
# 60.0 GiB for a run that holds a 42 GB DiT plus the 7.4 GB NVFP4 tower. This run
# holds the same DiT plus the 26.3 GB BF16 tower, which is ~19 GB more resident at
# once, and GB10's memory is unified so the device allocation comes out of the
# same pool. 78 GiB is 42 + 26.3 + the VAEs + ~8 GiB of slack. It is HIGHER than
# the neighbouring harness's floor rather than lower, and #1709 is why it is
# checked at all: this box has reported 5.0 GiB available across four leases while
# `rc` kept handing it out.
MEM_START_FLOOR_GIB=${MEM_START_FLOOR_GIB:-78.0}
MEM_START_WAIT_S=${MEM_START_WAIT_S:-1200}
MEM_START_POLL_S=${MEM_START_POLL_S:-30}
say "=== [0b] MemAvailable start gate: floor ${MEM_START_FLOOR_GIB} GiB ==="
waited=0
while :; do
  avail=$(mem_avail_gib)
  say "  memavail=${avail} GiB after ${waited}s"
  awk -v a="$avail" -v f="$MEM_START_FLOOR_GIB" 'BEGIN{exit !(a+0 >= f+0)}' && break
  [ "$waited" -ge "$MEM_START_WAIT_S" ] && {
    echo "FATAL: MemAvailable ${avail} GiB stayed below ${MEM_START_FLOOR_GIB} GiB for ${waited}s"
    exit 39; }
  sleep "$MEM_START_POLL_S"; waited=$((waited + MEM_START_POLL_S))
done
echo "mem_available_at_start_gib=$(mem_avail_gib)" >> "$OUT/PROVENANCE"

say "=== [1] tools ==="
apt-get install -y -qq ffmpeg python3-numpy > /root/apt.log 2>&1 || say "  apt returned non-zero; probing anyway"
for t in ffmpeg python3 cmake ninja; do command -v "$t" >/dev/null || { echo "FATAL: no $t"; exit 38; }; done
python3 -c 'import numpy' || { echo "FATAL: no numpy, and the comparison tool needs it"; exit 38; }

say "=== [A] CUDA toolkit ==="
need_ok() { [ -x "$1/bin/nvcc" ] && [ -f "$1/targets/sbsa-linux/lib/libcublasLt.so" ]; }
TKLIB=""
for c in /usr/local/cuda /usr/local/cuda-13.0 /root/cudatk; do
  if need_ok "$c"; then TKLIB=$c; break; fi
done
if [ -z "$TKLIB" ] && [ -d /workspace/a3/cuda-staged ]; then
  say "  staging the toolkit from /workspace/a3/cuda-staged (CIFS holds no symlink and serves 0664)"
  cp -a /workspace/a3/cuda-staged /root/cudatk || { echo "FATAL: cannot stage the toolkit"; exit 38; }
  chmod -R 0755 /root/cudatk/bin /root/cudatk/nvvm/bin 2>/dev/null
  ( cd /root/cudatk/targets/sbsa-linux/lib 2>/dev/null && for f in *.so.*.*; do
      b=${f%%.so.*}; ln -sf "$f" "$b.so"; ln -sf "$f" "$b.so.${f#*.so.}"; done ) 2>/dev/null
  need_ok /root/cudatk && TKLIB=/root/cudatk
fi
[ -n "$TKLIB" ] || { echo "FATAL: no complete CUDA toolkit (nvcc + libcublasLt)"; exit 38; }
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "  toolkit $TKLIB, $(nvcc --version | tail -1)"

say "=== [B] source ==="
[ -s "$W/src.tar.gz" ] || { echo "FATAL: no $W/src.tar.gz"; exit 31; }
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 31; }
WANT_SHA=$(cat "$W/src.sha" 2>/dev/null)
# THE TARBALL IS HASHED TOO. `src.sha` is a `git rev-parse HEAD` the job never
# checks against the bytes it unpacked, so a stale `.sha` beside a fresh tarball
# is undetectable. One line removes that class.
TAR_SHA=$(sha256sum "$W/src.tar.gz" | awk '{print $1}')
{ echo "source_sha=$WANT_SHA"; echo "source_tarball_sha256=$TAR_SHA"; } >> "$OUT/PROVENANCE"
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
  # NO `--strip-components`. The staged tarball's first member is `include/`, not
  # a versioned top-level directory, so stripping one component throws the
  # `include` away and leaves `cutlass.h` two levels from where every consumer
  # looks. Measured: `rc` job 54e29063 exited 36 here in 11 seconds. This is the
  # same `tar xzf "$TB" -C /root/cutlass` that
  # `ltx25-dit-attn-flash-pixel-ab.sh` has used all along.
  say "  unpacking the staged cutlass"
  mkdir -p /root/cutlass && tar xzf /workspace/cutlass-v4.5.0.tar.gz -C /root/cutlass
  [ -f /root/cutlass/include/cutlass/cutlass.h ] && CUT=/root/cutlass
fi
[ -n "$CUT" ] || { echo "FATAL: no CUTLASS"; exit 36; }

say "=== [D] build ==="
BUILT_FROM=cache
if [ -s "$CACHE/ltx2-gen" ] && [ -s "$CACHE/libvllm.so.0.0.3" ] && [ -s "$CACHE/test_ltx2_device" ] && \
   [ -n "$WANT_SHA" ] && [ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SHA" ]; then
  say "  cache hit on SRC_SHA=$WANT_SHA"
  cp -f "$CACHE/ltx2-gen" "$CACHE/libvllm.so.0.0.3" "$CACHE/test_ltx2_device" "$BIN"/
  chmod 0755 "$BIN/ltx2-gen" "$BIN/libvllm.so.0.0.3" "$BIN/test_ltx2_device"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
else
  # THE CACHE IS ALL-OR-NOTHING HERE, and that is the repair of a known failure.
  # `ltx25-dit-attn-flash-pixel-ab.sh` copies `test_ltx2_device` only `[ -s ]`, so
  # a cache staged without it satisfies the build skip and then has no correctness
  # gate to run. This condition requires all three, so a partial cache rebuilds
  # instead of rendering ungated.
  BUILT_FROM=in-lease
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
        -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" > "$OUT/configure.log" 2>&1 \
        || { echo "FATAL: configure failed"; tail -30 "$OUT/configure.log"; exit 33; }
  # NAMED TARGETS ONLY. A bare `ninja -C build` links every test binary and writes
  # 9.4 GiB, and the ENOSPC that follows makes checkers emit false refusals.
  ninja -C "$BLD" -j 4 ltx2-gen test_ltx2_device > "$OUT/build.log" 2>&1 \
        || { echo "FATAL: build failed"; tail -40 "$OUT/build.log"; exit 34; }
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
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
# BOTH HASHES, AND THE LIBRARY IS THE ONE THAT MATTERS (#1881): `ltx2-gen` is a
# small launcher whose sha256 has been byte-identical across builds hundreds of
# commits apart while the libraries differed by megabytes.
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
LIBSHA=$(sha256sum "$BIN/libvllm.so.0.0.3" | awk '{print $1}')
{ echo "binary_sha256=$BINSHA"; echo "library_sha256=$LIBSHA"; echo "binary_built=$BUILT_FROM"; } >> "$OUT/PROVENANCE"
"$BIN/ltx2-gen" --help >/dev/null 2>&1 || {
  echo "FATAL: ltx2-gen will not exec (126 = no exec bit, 127 = missing lib)"; ldd "$BIN/ltx2-gen" | head; exit 25; }
say "  ltx2-gen=$BINSHA lib=$LIBSHA built=$BUILT_FROM"

say "=== [E] checkpoints, staged and CHECKED AGAINST THE MANIFEST ==="
# THE SHA256 IS THE POINT, not the size. This repository has run a gate against
# the wrong checkpoint and not noticed, and a size match would not have caught it.
# Every value below is `ltx2_oracle_manifest.json`'s, so agreeing with it is
# agreeing with the bytes upstream actually loaded.
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
CKUSE=$CK
if [ "$FREE_K" -le "$NEED_K" ]; then
  # READING OVER CIFS IS SLOWER AND MEASURED SO (34-83 MiB/s, a 2.5x spread), but
  # a stage that does not fit is a stage that fails halfway. Recorded rather than
  # silently chosen, because it is the difference between a load time that can be
  # compared with another run's and one that cannot.
  say "  /root has ${FREE_K}K free against ${NEED_K}K needed: reading over CIFS instead"
  CKUSE=""
fi
for f in "${!WANTSZ[@]}"; do
  s="${SRCOF[$f]}/$f"; want=${WANTSZ[$f]}; wsha=${WANTSHA[$f]}
  got=$(stat -c %s "$s" 2>/dev/null || echo 0)
  [ "$got" = "$want" ] || { echo "FATAL: source $f is $got bytes, the manifest says $want"; exit 23; }
  if [ -n "$CKUSE" ]; then
    d=$CK/$f
    if [ -s "$d" ] && [ "$(stat -c %s "$d")" = "$want" ]; then
      say "  already staged $f"
    else
      # `.part` RENAMED ONLY ON A SIZE MATCH. `oracle-ltx-2-pin.md` records a
      # `cp: Resource temporarily unavailable` off this same soft CIFS mount whose
      # failure a plain `cp` swallowed, and the half-written file that followed.
      t=$SECONDS
      rm -f "$d" "$d.part"
      cp -- "$s" "$d.part" || { echo "FATAL: cannot stage $f"; exit 23; }
      [ "$(stat -c %s "$d.part")" = "$want" ] || { echo "FATAL: short stage of $f"; exit 23; }
      mv -f "$d.part" "$d"
      say "  staged $f $want bytes in $((SECONDS-t))s"
    fi
    u=$d
  else
    u=$s
  fi
  t=$SECONDS
  gsha=$(sha256sum "$u" | awk '{print $1}')
  [ "$gsha" = "$wsha" ] || { echo "FATAL: $f sha256 $gsha, the manifest says $wsha"; exit 23; }
  say "  sha256 OK $f ($((SECONDS-t))s)"
  echo "checkpoint_sha256 $f $gsha" >> "$OUT/PROVENANCE"
done
DIT=${CKUSE:+$CK}; DIT=${DIT:-$FULL/ckpt}
TE=${CKUSE:+$CK}; TE=${TE:-$CKROOT/text_encoders}
echo "checkpoint_dir_dit=$DIT checkpoint_dir_te=$TE" >> "$OUT/PROVENANCE"

say "=== [F] the CUDA unit gate, BEFORE any render ==="
[ -s "$BIN/test_ltx2_device" ] || { echo "FATAL: the CUDA unit gate binary is absent"; exit 45; }
"$BIN/test_ltx2_device" > "$OUT/test_ltx2_device.log" 2>&1 || {
  echo "FATAL: the CUDA unit gate FAILED; correctness comes before a render"
  tail -30 "$OUT/test_ltx2_device.log"; exit 44; }
say "  $(tail -3 "$OUT/test_ltx2_device.log" | tr '\n' ' ')"

say "=== [G] the render ==="
D=$OUT/ours
rm -rf "$D"; mkdir -p "$D"
LOG=$OUT/render.log
{
  echo "[arm] label=ours"
  echo "[arm] harness=$0 sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "[arm] binary=$BIN/ltx2-gen sha256=$BINSHA src_sha=$WANT_SHA"
  echo "[arm] library=$BIN/libvllm.so.0.0.3 sha256=$LIBSHA"
  echo "[arm] geometry=${WW}x${HH}/${FRAMES}f steps=$STEPS tokens=$TOK seed=$SEED"
  echo "[arm] prompt=<<$PROMPT>>"
  echo "[arm] dit=$DIT te=$TE"
} >> "$LOG"
t=$SECONDS
VT_OP_PROVIDER_STATS=1 stdbuf -oL -eL "$BIN/ltx2-gen" \
  --pipeline-kind one_stage \
  --checkpoint-class full \
  --dit "$DIT/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
  --video-vae "$DIT/ltx-2.5-video-vae-conv-bf16.safetensors" \
  --audio-vae "$DIT/ltx-2.5-audio-vae-bf16.safetensors" \
  --encoder "$TE/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
  --prompt "$PROMPT" \
  --frames "$FRAMES" --width "$WW" --height "$HH" --steps "$STEPS" --seed "$SEED" \
  --device cuda --workdir "$D" >> "$LOG" 2>&1
RENDER_RC=$?
RENDER_S=$((SECONDS-t))
say "  render rc=$RENDER_RC in ${RENDER_S}s"
echo "render_rc=$RENDER_RC render_seconds=$RENDER_S" >> "$OUT/PROVENANCE"

# COMPLETENESS IS DEFINED, not eyeballed. Exactly the expected frame count and a
# non-empty wav. A partial render that reached the comparison would produce a
# blockiness number over whatever frames survived.
# --steps 8 ARRIVED, OBSERVED RATHER THAN INFERRED FROM THE FLAG BEING PASSED.
# The row's `## Owed` records that every link in `main.cpp` -> `vllm_c.cpp` ->
# `ltx2_video.cpp` is verified by INSPECTION and none by execution, because the
# lease that would have executed it refused at the checkpoint load 76 s in.
# `VLLM_RENDER_PROGRESS` is ON by default and writes one
# `[render]   dit forward N  phase P step k/N  t=.. last=..` per DiT forward
# (docs/ENVIRONMENT.md), so the denominator in `step k/N` IS the resolved step
# count. Extracted here into its own file so the proof is an artefact of the run
# rather than something a later reader has to find in a log.
#
# WHY THE DENOMINATOR AND NOT THE LINE COUNT. `one_stage` is GUIDED and runs
# three DiT forwards per step, so counting lines measures the guider. The
# distinct set of denominators is the schedule, and a set with anything but a
# single 8 in it is the finding, not a formatting detail.
grep -oE 'step [0-9]+/[0-9]+' "$LOG" | awk -F/ '{print $2}' | sort -u > "$OUT/steps-observed.txt"
STEPS_SEEN=$(tr '\n' ',' < "$OUT/steps-observed.txt" | sed 's/,$//')
FORWARDS=$(grep -cE 'step [0-9]+/[0-9]+' "$LOG")
say "  --steps: requested $STEPS, denominators observed at runtime {${STEPS_SEEN:-none}}, $FORWARDS DiT forwards"
echo "steps_requested=$STEPS steps_observed={${STEPS_SEEN:-none}} dit_forwards=$FORWARDS" >> "$OUT/PROVENANCE"
if [ "$STEPS_SEEN" != "$STEPS" ]; then
  # NOT FATAL, and deliberately so: the comparison's verdict is the row's
  # deliverable and a step count that did not arrive is a SECOND finding rather
  # than a reason to discard the first. It is said loudly and it is recorded.
  say "  WARNING: the sampler did not run $STEPS steps. #2130's flag is wired and this run"
  say "  did NOT observe it arrive; the comparison below carries a denoise-budget confound."
fi

NF=$(ls "$D"/frame_*.ppm 2>/dev/null | wc -l)
say "  frames=$NF expected=$FRAMES audio=$(stat -c %s "$D/audio.wav" 2>/dev/null || echo 0) bytes"
if [ "$NF" != "$FRAMES" ] || [ ! -s "$D/audio.wav" ]; then
  echo "FATAL: the render is incomplete ($NF of $FRAMES frames); nothing is compared"
  tail -40 "$LOG"
  exit 48
fi
grep -E '^\[op\]|op=' "$LOG" | sort | uniq -c | head -20 >> "$OUT/PROVENANCE"

say "=== [H] the absolute comparison (#1854) ==="
# THE EXACT FORM FIRST: the 25 PPM frames upstream's own decode wrote, each one
# checked against the committed SHA256SUMS by the tool itself. The mp4 form is run
# second as a cross-check, because the row's spec claims the two agree on the
# gated bound to better than 0.08% and a claim in a spec that nothing re-runs is a
# number somebody wrote down.
[ -d "$REFDIR" ] || { echo "FATAL: the reference frames are not at $REFDIR"; exit 49; }
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
if [ "$CMP_RC" != "$MP4_RC" ]; then
  say "  NOTE: the two reference forms DISAGREE on the verdict ($CMP_RC vs $MP4_RC)."
  say "  That is a finding about the mp4's usability as a reference and belongs in the spec."
fi

say "=== [I] verdict ==="
echo "compare_exit_ppm=$CMP_RC compare_exit_mp4=$MP4_RC" >> "$OUT/PROVENANCE"
say "evidence in $OUT"
# THE JOB EXITS ON THE VERDICT, not on "the script finished".
exit "$CMP_RC"
