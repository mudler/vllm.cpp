#!/bin/bash
# LTX25-DIT-ATTN-FLASH (#1612) -- the PIXEL A/B, and the speed A/B it also settles.
#
# `scripts/ltx25-dit-attn-flash-ab.sh` is this file's sibling and answers a
# different question. It caps each arm at 13 forwards, which is enough for a
# per-forward median and produces NO FRAMES. #1549 shipped on that: a kernel
# swap that is explicitly not bit-identical on CUDA, gated only by a
# reduced-dimension host-vs-device parity case, with nothing at all said about
# what the model RENDERS. A diffusion model has no token gate, so there is no
# discrete output to fall back on. This file renders both arms to completion and
# compares the pixels.
#
# THREE RENDERS, and the third is the point.
#
#   1. flash      VLLM_LTX2_DIT_FLASH_ATTN=flash   the #1549 arm
#   2. naive      VLLM_LTX2_DIT_FLASH_ATTN=0       the arm #1549 replaced
#   3. flash-ctl  VLLM_LTX2_DIT_FLASH_ATTN=flash   flash AGAIN, same binary, same seed
#
# THE FLASH ARMS SAID `=1` UNTIL #1751, AND THAT VALUE STOPPED MEANING FLASH.
# When this file was written the knob was BINARY: `=0` selected `vt::Attention`
# and any other value selected `vt::AttentionDenseFlash`, so `=1` was the flash
# arm and the recorded run's own table shows it announcing `op=21` (spec
# `ltx25-dit-attn-flash.md` section 10). #1551 made the knob THREE-WAY and gave
# the flash rung the exact spelling `flash`; from that commit `=1` matched no arm
# and fell through to the FA-2 default, so these two arms named a rung they no
# longer selected. `arm_report` below would have caught it -- `op=21` absent is
# ROUTING_BAD and exit 46 -- but only after the render, which is an hour of a
# four-hour lease to learn that a literal went stale. #1751 makes an unrecognised
# value refuse at the first DiT forward instead, and
# `tests/scripts/test_ltx2_dit_attn_knob_arms.py` holds every value this file
# sets against the arms the dispatch actually parses.
#
# Without (3) an arm-to-arm difference cannot be attributed to the kernel. Two
# runs of one configuration measure what the BOX does on its own -- cuBLAS split
# reductions, allocator-dependent kernel selection, anything nondeterministic
# anywhere in a 120-forward denoise plus a VAE decode. If (3) is bit-identical
# to (1) the noise floor is exactly zero and every bit of the flash-vs-naive
# delta is the swapped op. If (3) differs from (1) by as much as (2) does, the
# swap changed nothing the machine does not change by itself. Either reading is
# an answer; neither is available from two renders.
#
# ORDER: flash, naive, flash-ctl. The naive arm is ~6x the wall clock of a flash
# arm and it is the one whose loss leaves no A/B at all, so it is taken second,
# while the box is known good, rather than last. The control is last because it
# is the only one recoverable in a short follow-up lease: the build cache below
# is keyed on the source sha, so a resumed run reaches a render in minutes.
#
# The 20260821T092516Z attempt lost the worker at forward 20 with no memory
# trace and no guard, so it could not say afterwards what had happened. This one
# writes a MemAvailable trace per arm and stops an arm that crosses a floor.
# It does NOT run under `runguard.py --stack-period`: that sampler ptrace-stops
# every thread and cost the recorded 47.84 s denominator ~3.2% (spec section
# 7.1). Both arms here are instrumented identically and neither is sampled, so
# the ratio needs no correction.
#
# BEFORE YOU SPEND A LEASE ON THIS, read these three paragraphs.
#
#   EXIT STATUS. 0, 1, 2 and 3 are the pixel comparison's own verdict and this
#   job exits with it: 0 every threshold held, 1 a threshold failed, 2 an input
#   could not be read and NOTHING was compared, 3 the treatment passed and the
#   CONTROL failed its own content checks, so the pass cannot be READ. A 2 is
#   never a pass and neither is a 3. Everything else is a refusal before the
#   verdict exists:
#     23 checkpoint staging      25 ltx2-gen will not exec
#     31 source tarball          33 configure      34 build      35 artefacts
#     36 no CUTLASS              38 no complete CUDA toolkit
#     39 MemAvailable is below the start floor and stayed there
#     40 the swapped op is not in this source     41 the A/B knob is not either
#     43 the comparison tool is not in this source
#     44 the CUDA unit gate FAILED       45 the CUDA unit gate BINARY IS ABSENT
#     46 an arm did not route as its knob asked
#
#   A 45 IS USUALLY THE STAGED BINARY CACHE, and the fix is one command. Phase
#   [D] reuses `$W/pixab-bin` when its `SRC_SHA` matches, and it copies
#   `test_ltx2_device` only `[ -s ]` -- a cache staged without it therefore
#   satisfies the build skip and then has no correctness gate to run. This job
#   refuses rather than rendering, because a routing assertion that cannot run,
#   inside a job whose whole purpose is proving which kernel executed, is a
#   silent hole. Clear `$W/pixab-bin` and the build regenerates both.
#
#   RESUMING. `RUN_ID` is an environment override. Pass the same one and this
#   run lands in the same `$OUT` and skips every arm that is already complete
#   there. A reused arm's per-forward samples came from the earlier lease, so it
#   is recorded as `timing_source=an-earlier-lease` and phase [H] states that the
#   speed pair is not a same-lease pair. Its routing is still proved, from the
#   log that arm already has.
set -u
T0=$(date +%s)
say() { echo "[pixab +$(( $(date +%s) - T0 ))s] $*"; }
W=/workspace/ltx25-attnflash
FULL=/workspace/ltx25-fullmodel        # checkpoints and the text-encoder config
SRC=/root/src-pixab
BLD=/root/build-pixab
CK=/root/ckpt
# RUN_ID IS OVERRIDABLE, and that is the whole resume mechanism. `dgx:gpu0` has
# lost its worker three times, and this job renders three arms over about four
# hours, so a lease lost after the ~2 h naive arm used to throw that arm away
# and start a new timestamped directory. A resumed lease passes the same RUN_ID,
# lands in the same $OUT, and every arm that is already COMPLETE there is
# skipped. It composes with the binary cache in phase [D], which is keyed on the
# source sha, so a resumed run reaches its first missing render in minutes.
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$W/pixel-ab/$RUN_ID
mkdir -p "$OUT" "$CK"
export DEBIAN_FRONTEND=noninteractive
say "RUN_ID=$RUN_ID OUT=$OUT"
# APPENDED, never truncated: a resumed lease writes its own block, and the
# earlier lease's binary sha and start time are part of what the resumed run's
# evidence rests on.
{
  echo "--- lease $(date -Is) ---"
  echo "run_id=$RUN_ID"
  echo "rc_job=${RC_JOB_ID:-unknown}"
  echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
  echo "started=$(date -Is)"
} >> "$OUT/PROVENANCE"

# A LIVENESS LINE, and nothing more. The build redirects to a file and a single
# 42 GB checkpoint copy takes minutes, so this job can produce no stdout at all
# for the better part of an hour, and a lease with an idle timeout would kill a
# healthy run. It prints ONLY the elapsed time: it reports no count and no
# progress, because a line emitted on a fixed cadence cannot distinguish work
# from a hang and must not be read as if it could. The render loop below prints
# the forward count separately, and that one CAN stop advancing.
( while :; do sleep 120; echo "[pixab-alive +$(( $(date +%s) - T0 ))s]"; done ) &
HEARTBEAT=$!
# EXIT ALONE DOES NOT COVER A LEASE KILL. `rc` reclaiming a device sends SIGTERM,
# and a bash trap on EXIT does not run for a signal that has no handler, so the
# heartbeat subshell survived its parent and kept printing into a job nobody was
# reading. Each signal cleans up and then exits with 128+signo, which is the
# status the shell would have reported had the trap not existed.
cleanup() { kill "$HEARTBEAT" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 129' HUP
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# BEGIN pixab-helpers -- extracted verbatim by
# tests/scripts/test_ltx25_pixel_ab_harness.py, which runs both branches of the
# memory gate against a fabricated meminfo. Everything between these markers
# must depend on nothing but `say`, $MEMINFO and coreutils.
#
# ONE READER for MemAvailable, because two gates that disagree about what they
# measured are worse than one gate. The start gate below and the render
# watchdog in render() both call this, and neither reads `free`'s "available"
# column: that is a second source for the same quantity, and phase [0] printing
# one number while the watchdog acts on another is how this job walked into a
# box with 5 GiB free while printing that it had 5 GiB free.
MEMINFO=${MEMINFO:-/proc/meminfo}
mem_avail_gib() {
  awk '/^MemAvailable:/{printf "%.1f", $2/1048576; f=1; exit} END{if(!f) print ""}' \
    "$MEMINFO" 2>/dev/null
}

# WAIT, then refuse. A lease spent waiting and refusing costs a lease; a lease
# spent building into an out-of-memory kill costs the lease AND leaves a record
# that says nothing about why. Each poll logs its value so a reader can tell a
# recovering box from a flat one.
wait_for_memory() {  # $1 floor GiB, $2 budget s, $3 poll s
  local floor=$1 budget=$2 poll=$3 avail waited=0
  while :; do
    avail=$(mem_avail_gib)
    if [ -z "$avail" ]; then
      echo "FATAL: cannot read MemAvailable from $MEMINFO"
      return 39
    fi
    if awk -v a="$avail" -v f="$floor" 'BEGIN{exit !(a>=f)}'; then
      say "  MemAvailable ${avail} GiB >= floor ${floor} GiB after ${waited}s: proceeding"
      return 0
    fi
    if [ "$waited" -ge "$budget" ]; then
      echo "FATAL: MemAvailable ${avail} GiB is below the ${floor} GiB start floor" \
           "after ${waited}s of waiting (budget ${budget}s). The box was already" \
           "occupied when this lease started; nothing was built and nothing was rendered."
      return 39
    fi
    say "  MemAvailable ${avail} GiB < ${floor} GiB, waited ${waited}s of ${budget}s"
    sleep "$poll"
    waited=$((waited + poll))
  done
}

# A COMPLETE ARM, and nothing weaker. Exactly the expected frame count and a
# non-empty wav. A partial arm is re-rendered from scratch rather than resumed
# mid-flight: the engine deletes stale frame_*.ppm in its own output directory
# but nothing else there, so a half-arm's leftovers would outlive it.
#
# TWO CHECKS, and there used to be a third. `[ -d "$d" ] || return 1` stood on
# the first line and no mutation could reach it: deleting it left this suite
# green, because the glob does not expand for a directory that is not there, so
# `ls ... | wc -l` reports 0 and the frame count refuses the arm anyway. That is
# the same argument section 10.4 already made against the fourth C0 check in
# `ltx25-render-compare.py` -- a guard that cannot be observed to fail is a
# decoration -- so it is removed rather than kept for the look of it. An absent
# directory is still refused, by the line below, and
# `test_an_absent_directory_is_not_complete` says which line does it.
arm_is_complete() {  # $1 dir, $2 wanted frame count
  local d=$1 want=$2 n
  n=$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)
  [ "$n" = "$want" ] || return 1
  [ -s "$d/audio.wav" ] || return 1
  return 0
}
# END pixab-helpers

# BEGIN memwatch-helpers
# ONE spelling, extracted verbatim by tests/scripts/test_ltx25_ab_memwatch.py,
# which asserts these bytes are IDENTICAL in ltx25-dit-attn-fa2-hd128-ab.sh,
# ltx25-dit-attn-flash-ab.sh and ltx25-dit-attn-flash-pixel-ab.sh. #1734 was one
# idiom written three ways, two of them wrong, in three files in this directory;
# a block that has to match byte-for-byte is what stops a fourth spelling from
# appearing beside them. The marker lines carry no trailing prose, because the
# extractor splits on them and anything after them becomes a bash command.

# How many `last=` lines the engine log carries so far, as ONE integer.
#
# TWO GUARDS, AND THE SECOND ONE IS THE LOAD-BEARING ONE. `grep -c` prints `0`
# AND exits 1 when it matches nothing, so `grep -c ... || echo 0` runs the
# fallback ON TOP of grep's own count and yields the two-line string `0\n0`.
# That is #1734: the poll's tab-separated record went out through `echo`, the
# embedded newline split it across two lines on disk -- 170 of
# `watch-flash.tsv`'s 186 records in the 20260822T203535Z run -- and the same
# string reached the sample cap's `[ "$n" -ge ... ]`, where bash answers
# `integer expression expected` and returns 2.
#
#   `head -1`  drops the fallback's second line. It is the belt, and it is what
#              #1734 asked for -- but with the floor below it in front, NO test
#              detects its removal, because the floor already refuses `0\n0`.
#              It stays as the guard for the fallback being written back, and
#              that limit is stated rather than left for the next reader to
#              re-derive.
#   `case`     floors anything that is not a run of digits to `0`. It is the
#              braces, it subsumes the newline case, and it is the ONLY thing
#              covering the reads where grep prints NOTHING at all.
#
# MEASURED, on GNU grep 3.11, which is what `bash -c` resolves here and in the
# lease. A DIRECTORY answers `0` on stdout with status 2 and an EMPTY file
# answers `0` with status 1, so neither needs the floor. An ABSENT file and a
# PERMISSION-DENIED file write only to stderr and answer with NOTHING, and
# those two are what the floor is for: `head -1` cannot take a line that was
# never printed, so without the floor `$n` reaches an integer test empty.
sample_count() {  # $1 = engine log
  local n
  n=$(grep -c 'last=' "$1" 2>/dev/null | head -1)
  case "$n" in
    ''|*[!0-9]*) echo 0 ;;
    *)           echo "$n" ;;
  esac
}

# The lowest `memavail_gib=` reading in a watch TSV, WITH its unit, or the words
# `NO READINGS`.
#
# MATCHED ON ITS KEY, never on a field number. The reducer this replaces read
# `$4` and stripped the `memavail_gib=` prefix off `$4` alone, so it depended on
# a position it could not rely on: any record whose shape moved gave it the
# empty string, which sorts FIRST under `sort -n`, so `head -1` returned it and
# the report printed `memavail low-water:  GiB`.
#
# `\b` anchors the key so that a DIFFERENT key ending in these characters is not
# read as this one. Without it a record carrying `gpu_memavail_gib=2.0` beside
# `memavail_gib=99.0` reports 2.0, which is a low-water mark for a quantity
# nobody asked about. `LC_ALL=C` pins the sort: under a locale whose thousands
# separator is `.`, `sort -n` can read 1234.5 as 12345.
#
# It returns the unit or the words, never a bare value, for the same reason:
# a missing measurement must not be able to print as a measured one. A blank
# where a number belongs reads as "measured, and fine", and these harnesses back
# published attention ratios whose memory low-water is part of the evidence.
memavail_low_water() {  # $1 = watch TSV
  local v
  v=$(grep -ohE '\bmemavail_gib=[0-9]+(\.[0-9]+)?' "$1" 2>/dev/null \
        | cut -d= -f2 | LC_ALL=C sort -n | head -1)
  if [ -n "$v" ]; then echo "$v GiB"; else echo "NO READINGS"; fi
}
# END memwatch-helpers

say "=== [0] the box ==="
uname -m; nproc; free -g | head -2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | head -3
df -h / /root /workspace 2>&1 | head -6

say "=== [0b] the MemAvailable PRECONDITION, gated rather than printed ==="
# THE NUMBER COMES FROM THE BASELINE, not from taste. The recorded 20260820
# render at this geometry started from baseline_used 4.643 GiB, peaked at
# 79.503 GiB and had a MemAvailable low-water of 40.13 GiB, so this job needs
# most of the box and cannot start on the tail of it.
#
# This gate exists because phase [0] above PRINTED
#   Mem: total 119 used 114 free 1 shared 0 buff/cache 4 available 5
# at +0s on 2026-08-22 (rc job 5fb9399f-4f4e-417c-adbd-4d741a2e18e4) and the run
# proceeded anyway: cmake, then ninja -j 4 on CUDA, then a lost worker at ~+728s
# with no binary cached and nothing measured. The box was already at 114 of 119
# GiB before this job allocated a byte, so the harness cannot fix the cause --
# but it must not walk into it while printing the exact number that says not to.
MEM_START_FLOOR_GIB=${MEM_START_FLOOR_GIB:-60.0}
MEM_START_WAIT_S=${MEM_START_WAIT_S:-1200}
MEM_START_POLL_S=${MEM_START_POLL_S:-30}
say "start floor ${MEM_START_FLOOR_GIB} GiB, wait budget ${MEM_START_WAIT_S}s, poll ${MEM_START_POLL_S}s"
wait_for_memory "$MEM_START_FLOOR_GIB" "$MEM_START_WAIT_S" "$MEM_START_POLL_S" || exit 39
echo "mem_available_at_start_gib=$(mem_avail_gib)" >> "$OUT/PROVENANCE"

say "=== [1] tools ==="
# ffmpeg matters beyond the mp4: ltx2-gen exits 127 from its absence AFTER every
# frame and the wav are on disk (#1149), so its absence reads as a failed render
# even though the artifacts this file compares are already written. numpy is
# what scripts/ltx25-render-compare.py needs and it is the only dependency it
# has; the PPM and WAV readers there are written out precisely so that no image
# or audio library has to be present in a leased worker.
apt-get update -qq >/root/apt.log 2>&1
apt-get install -y -qq ffmpeg python3-numpy >>/root/apt.log 2>&1
echo "apt_rc=$?"
for t in ffmpeg ffprobe python3; do printf '%s=' "$t"; command -v "$t" || echo MISSING; done
python3 -c 'import numpy;print("numpy",numpy.__version__)' 2>&1 | tail -1

say "=== [A] CUDA toolkit, resolved BEFORE anything probes it ==="
CUDATK=/root/cudatk
STAGED=/workspace/a3/cuda-staged
need_ok() { [ -x "$1/bin/nvcc" ] && [ -f "$1/targets/sbsa-linux/lib/libcublasLt.so" ]; }
TKLIB=""
for r in /usr/local/cuda /usr/local/cuda-13.0 "$CUDATK"; do
  need_ok "$r" && TKLIB="$r" && say "complete toolkit: $r" && break
done
if [ -z "$TKLIB" ]; then
  for r in /usr/local/cuda /usr/local/cuda-13.0; do
    [ -x "$r/bin/nvcc" ] || continue
    LD="$r/targets/sbsa-linux/lib"
    [ -d "$LD" ] && [ -w "$LD" ] || continue
    for lib in libcublasLt libcublas; do
      f=$(ls "$STAGED"/targets/sbsa-linux/lib/$lib.so.13* 2>/dev/null | grep -v static | head -1)
      [ -n "$f" ] || continue
      say "repairing $r: adding $(basename "$f")"
      cp -f "$f" "$LD"/ && ( cd "$LD" && b=$(basename "$f"); ln -sf "$b" "$lib.so.13"; ln -sf "$b" "$lib.so" )
    done
    need_ok "$r" && TKLIB="$r" && break
  done
fi
if [ -z "$TKLIB" ] && [ -d "$STAGED/bin" ]; then
  if ! need_ok "$CUDATK"; then
    say "staging a COMPLETE toolkit locally (~4.9G)"
    mkdir -p "$CUDATK"; cp -a "$STAGED"/. "$CUDATK"/ 2>/dev/null
    # The share serves file_mode=0664: every binary copied off it arrives
    # NON-EXECUTABLE, and an nvcc that cannot run reads as an nvcc that is absent.
    find "$CUDATK/bin" "$CUDATK/nvvm/bin" -type f -exec chmod 0755 {} + 2>/dev/null
    # CIFS stores no symlinks; find_package wants libX.so and libX.so.MAJOR.
    ( cd "$CUDATK/targets/sbsa-linux/lib" 2>/dev/null || exit 0
      for f in *.so.*; do
        case "$f" in *.a) continue;; esac
        b=${f%%.so.*}; rest=${f#*.so.}; maj=${rest%%.*}
        [ -e "$b.so.$maj" ] || ln -sf "$f" "$b.so.$maj"
        [ -e "$b.so" ] || ln -sf "$f" "$b.so"
      done )
  fi
  need_ok "$CUDATK" && TKLIB="$CUDATK"
fi
[ -n "$TKLIB" ] || { echo "FATAL: no COMPLETE CUDA toolkit"; exit 38; }
# cmake finds nvcc via PATH, not CUDAToolkit_ROOT alone.
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "CUDAToolkit_ROOT=$TKLIB"
nvcc --version | tail -2

say "=== [B] source, and the PRECONDITION that it carries both arms ==="
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/pixab-src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 31; }
WANT_SHA=$(cat "$W/pixab-src.sha" 2>/dev/null)
echo "  built_from=$WANT_SHA"
echo "source_sha=$WANT_SHA" >> "$OUT/PROVENANCE"
# BOTH sides, because a half-applied tree satisfies either alone: the swapped op
# without the knob makes the naive arm a second flash arm, and the knob without
# the swap makes the flash arm a second naive one. Either way the A/B renders one
# configuration three times and still prints three columns.
NEWOP=$(grep -c 'vt::AttentionDenseFlash' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
KNOB=$(grep -c 'VLLM_LTX2_DIT_FLASH_ATTN' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
echo "  AttentionDenseFlash call sites: $NEWOP (want >= 1)"
echo "  A/B knob sites:                 $KNOB (want >= 1)"
[ "$NEWOP" -ge 1 ] || { echo "FATAL: #1549 is NOT in this source tree"; exit 40; }
[ "$KNOB"  -ge 1 ] || { echo "FATAL: the A/B knob is NOT in this source tree; both arms would be one arm"; exit 41; }
CMP="$SRC/scripts/ltx25-render-compare.py"
[ -s "$CMP" ] || { echo "FATAL: the comparison tool is not in this source tree"; exit 43; }

say "=== [C] cutlass (resolved, never fetched) ==="
CUT=""
for c in /cutlass /workspace/cutlass /root/cutlass; do
  [ -f "$c/include/cutlass/cutlass.h" ] && CUT="$c" && break
done
TB=/workspace/cutlass-v4.5.0.tar.gz
if [ -z "$CUT" ] && [ -f "$TB" ]; then
  say "unpacking staged cutlass"
  mkdir -p /root/cutlass && tar xzf "$TB" -C /root/cutlass && CUT=/root/cutlass
fi
[ -n "$CUT" ] && [ -f "$CUT/include/cutlass/cutlass.h" ] || { echo "FATAL: no CUTLASS tree"; exit 36; }
say "CUTLASS_DIR=$CUT"

say "=== [D] configure + build (-j 4, per the GB10 recipe) ==="
# The binary is REUSED only when the cached source sha matches the tarball's, so
# a resumed run can never render a tree other than the one it claims. That is
# the failure this guard prevents, not the rebuild.
CACHE="$W/pixab-bin"
BIN=/root/pixabbin; mkdir -p "$BIN"
SKIP_BUILD=0
if [ -s "$CACHE/ltx2-gen" ] && [ -s "$CACHE/libvllm.so.0.0.3" ] && \
   [ -n "$WANT_SHA" ] && [ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SHA" ]; then
  say "REUSING the staged binary: cached SRC_SHA matches $WANT_SHA"
  cp -f "$CACHE/ltx2-gen" "$BIN"/ && chmod 0755 "$BIN/ltx2-gen"
  cp -f "$CACHE/libvllm.so.0.0.3" "$BIN"/ && chmod 0755 "$BIN/libvllm.so.0.0.3"
  cp -f "$CACHE/ltx2_gemma4_text_config.json" "$BIN"/ 2>/dev/null || \
    cp -f "$FULL/bin/ltx2_gemma4_text_config.json" "$BIN"/
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  [ -s "$CACHE/test_ltx2_device" ] && cp -f "$CACHE/test_ltx2_device" "$BIN"/ && chmod 0755 "$BIN/test_ltx2_device"
  SKIP_BUILD=1
fi
if [ "$SKIP_BUILD" = 0 ]; then
  cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_CUDA=ON \
        -DVLLM_CPP_CUTLASS_DIR="$CUT" -DCUDAToolkit_ROOT="$TKLIB" > /root/cfg.log 2>&1
  CFG=$?; echo "CFG_RC=$CFG"
  grep -iE 'CUDA target arch|cutlass|flashattention' /root/cfg.log | head -8
  if [ "$CFG" != 0 ]; then
    awk '/CMake (Error|Warning)/,/^$/' /root/cfg.log | head -60; tail -40 /root/cfg.log
    cp -f /root/cfg.log "$OUT/configure-fail.log"; echo "FATAL: configure failed"; exit 33
  fi
  cp -f /root/cfg.log "$OUT/configure.log"
  # Unconstrained parallelism has OOM-rebooted this box.
  #
  # THE MEMORY AT THE MOMENT THE BUILD STARTS, logged here and not only at [0b].
  # The 2026-08-22 worker was lost during exactly this command and the log could
  # not say what memory it began with, because the only reading was taken 38 s
  # earlier under a different tenant's allocation.
  say "  ninja -j 4 starting with MemAvailable $(mem_avail_gib) GiB"
  echo "mem_available_at_build_gib=$(mem_avail_gib)" >> "$OUT/PROVENANCE"
  ninja -C "$BLD" -j 4 ltx2-gen test_ltx2_device > /root/build.log 2>&1
  B=$?; echo "BUILD_RC=$B"
  echo "  compile_errors=$(grep -ciE ' error: ' /root/build.log)"
  tail -15 /root/build.log
  cp -f /root/build.log "$OUT/build.log"
  [ "$B" = 0 ] || { echo "FATAL: build failed"; exit 34; }
  GEN=$(find "$BLD" -name ltx2-gen -type f | head -1)
  LIB=$(find "$BLD" -name 'libvllm.so.0.0.3' | head -1)
  [ -n "$GEN" ] && [ -n "$LIB" ] || { echo "FATAL: build artefacts not found"; exit 35; }
  cp -f "$GEN" "$BIN"/ && chmod 0755 "$BIN/ltx2-gen"
  cp -f "$LIB" "$BIN"/ && chmod 0755 "$BIN/libvllm.so.0.0.3"
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  cp -f "$SRC/tests/vllm/models/ltx2_gemma4_text_config.json" "$BIN"/ 2>/dev/null || \
    cp -f "$FULL/bin/ltx2_gemma4_text_config.json" "$BIN"/
  T=$(find "$BLD" -name test_ltx2_device -type f | head -1)
  [ -n "$T" ] && cp -f "$T" "$BIN"/ && chmod 0755 "$BIN/test_ltx2_device"
  mkdir -p "$CACHE"
  cp -f "$BIN/ltx2-gen" "$CACHE"/ && cp -f "$BIN/libvllm.so.0.0.3" "$CACHE"/
  cp -f "$BIN/ltx2_gemma4_text_config.json" "$CACHE"/ 2>/dev/null
  [ -s "$BIN/test_ltx2_device" ] && cp -f "$BIN/test_ltx2_device" "$CACHE"/
  echo "$WANT_SHA" > "$CACHE/SRC_SHA"
  say "staged the binary for a resumed run"
fi
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
say "ONE BINARY, all three renders: sha256=$BINSHA"
{ echo "binary_sha256=$BINSHA"; echo "binary_built=$([ "$SKIP_BUILD" = 1 ] && echo cache || echo in-lease)"; } >> "$OUT/PROVENANCE"
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
# sha256 and ldd both pass on a file with no execute bit. Ask the binary itself.
"$BIN/ltx2-gen" --help >/dev/null 2>&1 || { echo "FATAL: ltx2-gen will not exec (126 = no exec bit, 127 = missing lib)"; ldd "$BIN/ltx2-gen" | head; exit 25; }
say "EXECUTABLE_OK"

say "=== [E] checkpoints, staged to LOCAL disk ==="
# Measured: 589-1446 s per load over CIFS at 34-83 MiB/s, against ~32 s from
# local disk. Three renders pay that three times, so the ~9 minute copy is not
# an optimisation, it is most of the difference between fitting in a lease and
# not. Each file is matched on EXACT BYTE SIZE, so a half-written stage is
# refused rather than loaded.
declare -A WANT=(
  [ltx-2.5-22b-dev-transformer-bf16.safetensors]=42018190584
  [ltx-2.5-video-vae-conv-bf16.safetensors]=1452269922
  [ltx-2.5-audio-vae-bf16.safetensors]=364866540
  [gemma4-12b-with-proj-nvfp4-torchao.safetensors]=7423624178
)
FREE_K=$(df -k --output=avail /root | tail -1)
NEED_K=$(( (42018190584 + 1452269922 + 364866540 + 7423624178) / 1024 + 8388608 ))
say "local free ${FREE_K}K, need ${NEED_K}K"
CKUSE=$CK
if [ "$FREE_K" -le "$NEED_K" ]; then
  say "NOT staging (insufficient local disk); reading weights over CIFS"
  CKUSE=$FULL/ckpt
else
  for f in "${!WANT[@]}"; do
    s=$FULL/ckpt/$f; d=$CK/$f; want=${WANT[$f]}
    got=$(stat -c %s "$s" 2>/dev/null || echo 0)
    [ "$got" = "$want" ] || { echo "FATAL: source $f is $got bytes, want $want"; exit 23; }
    if [ -s "$d" ] && [ "$(stat -c %s "$d")" = "$want" ]; then say "  already staged $f"; continue; fi
    t=$SECONDS; cp "$s" "$d" || { echo "FATAL: cannot stage $f"; exit 23; }
    [ "$(stat -c %s "$d")" = "$want" ] || { echo "FATAL: short stage of $f"; exit 23; }
    say "  staged $f $want bytes in $((SECONDS-t))s"
  done
fi
# THE FULL/DEV TRANSFORMER, 42,018,190,584 B. Never the distilled one: nothing
# validates checkpoint class (#1137) and it would render plausibly in the wrong
# regime.
say "checkpoints from $CKUSE"
echo "checkpoint_dir=$CKUSE" >> "$OUT/PROVENANCE"

say "=== [F] CORRECTNESS FIRST: the CUDA unit gate, before any render ==="
# AGENTS.md: establish the correctness gate before accepting a performance
# result. `assertions: 0` is a skip wearing a pass and a thrown case shows up
# only on the `Status:` line, so both are printed rather than a grep for ok.
#
# AND IT REFUSES. It used to print its status and carry on, which made
# "correctness first" a heading rather than a gate: a red unit case, or a binary
# with no unit case at all, reached three renders and a published pixel verdict
# with nothing between them.
if [ -x "$BIN/test_ltx2_device" ]; then
  "$BIN/test_ltx2_device" > "$OUT/test_ltx2_device.log" 2>&1
  UNIT_RC=$?
  echo "  test_ltx2_device_RC=$UNIT_RC"
  grep -E 'assertions:|test cases:|Status:|SKIP' "$OUT/test_ltx2_device.log" | tail -8
  [ "$UNIT_RC" = 0 ] || {
    echo "FATAL: the CUDA unit gate FAILED (rc=$UNIT_RC); see $OUT/test_ltx2_device.log."
    echo "       Correctness comes before a render, so no arm is taken on this binary."
    exit 44
  }
else
  echo "FATAL: $BIN/test_ltx2_device is absent, so the correctness gate cannot run."
  echo "       A cached binary staged without it is the usual cause; rebuild with"
  echo "       an empty $CACHE to restore it."
  exit 45
fi

say "=== [G] the three renders ==="
# The prompt, seed and geometry of the recorded 49-frame baseline render
# (out/20260820T223701Z/768x448-49f/render.log line 1), copied byte-for-byte so
# that this pair is additionally comparable to it. The primary evidence is the
# same-binary pair below; the older render was built from a50c57d69, which is an
# ANCESTOR of the swap and therefore a different binary lineage, so it is a
# cross-check and never the control.
PROMPT='A golden retriever shakes water from its coat on a sunlit lawn, droplets flying outward in a bright arc around its head and shoulders, wet fur rippling and separating into strands from shoulders to tail, ears flapping, muscles moving under the coat. Crisp midday light, shallow depth of field, vivid green grass behind. The dog barks once, water patters onto the grass, and a light breeze moves through the trees.'
FRAMES=${FRAMES:-49}; WW=${WW:-768}; HH=${HH:-448}; SEED=${SEED:-20260820}
TOK=$(( (WW/32) * (HH/32) * (((FRAMES-1)/8) + 1) ))
MEM_FLOOR_GIB=${MEM_FLOOR_GIB:-8.0}
say "geometry ${WW}x${HH}/${FRAMES}f = $TOK video tokens, seed $SEED"
say "MemAvailable floor ${MEM_FLOOR_GIB} GiB (the recorded baseline's low-water was 40.13 GiB)"
{
  echo "geometry=${WW}x${HH}/${FRAMES}f"
  echo "video_tokens=$TOK"
  echo "seed=$SEED"
  echo "prompt_sha256=$(printf '%s' "$PROMPT" | sha256sum | awk '{print $1}')"
} >> "$OUT/PROVENANCE"

REUSED_ARMS=""
render() {  # $1 = label, $2 = knob value, $3 = hard timeout seconds
  local label=$1 knob=$2 tmo=$3
  local d="$OUT/$label"
  local log="$d/render.log"
  # RESUME, and say so. A skipped arm is loud, because a speed pair assembled
  # from two leases is NOT a same-binary same-lease pair, and the report must
  # not be able to claim one silently. The routing proof below still runs for a
  # skipped arm, from the log that arm already has: a reused arm whose routing
  # was never proved is worse than no arm at all.
  local timing_from="this-lease"
  if arm_is_complete "$d" "$FRAMES"; then
    say "--- render $label SKIPPED: $d already holds $FRAMES frames and an audio.wav"
    say "    (resumed run RUN_ID=$RUN_ID; its timings were taken in an EARLIER lease)"
    timing_from="an-earlier-lease"
    REUSED_ARMS="$REUSED_ARMS $label"
  else
    if [ -d "$d" ]; then
      say "  $label is INCOMPLETE ($(ls "$d"/frame_*.ppm 2>/dev/null | wc -l) of $FRAMES"\
          "frames, audio=$([ -s "$d/audio.wav" ] && echo yes || echo no)); re-rendering from scratch"
      rm -rf "${d:?}"
    fi
    mkdir -p "$d"
    render_arm "$label" "$knob" "$tmo" "$d" "$log"
  fi
  echo "arm=$label timing_source=$timing_from" >> "$OUT/PROVENANCE"
  echo "  timing_source=$timing_from" | tee -a "$d/ARM"
  arm_report "$label" "$knob" "$d" "$log"
}

render_arm() {  # $1 label, $2 knob, $3 tmo, $4 dir, $5 log
  local label=$1 knob=$2 tmo=$3 d=$4 log=$5
  say "--- render $label (VLLM_LTX2_DIT_FLASH_ATTN=$knob, hard cap ${tmo}s) ---"
  : > "$log"
  # EVERY RENDER STATES ITS OWN INVOCATION on line 1 of its own log, the way the
  # recorded baseline's render.log does and the way the withdrawn 7.680 s arm did
  # not. An arm whose log cannot say what it ran is not evidence, whatever number
  # it contains.
  {
    echo "[arm] label=$label knob=$knob tmo=${tmo}s"
    echo "[arm] harness=$0 sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
    echo "[arm] binary=$BIN/ltx2-gen sha256=$BINSHA src_sha=$WANT_SHA"
    echo "[arm] geometry=${WW}x${HH}/${FRAMES}f tokens=$TOK seed=$SEED ckpt=$CKUSE"
    echo "[arm] prompt=<<$PROMPT>>"
    echo "[arm] cmd: $BIN/ltx2-gen --pipeline-kind one_stage --dit $CKUSE/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
         "--video-vae $CKUSE/ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae $CKUSE/ltx-2.5-audio-vae-bf16.safetensors" \
         "--checkpoint-class full --encoder $CKUSE/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
         "--encoder-config $BIN/ltx2_gemma4_text_config.json --prompt <see above>" \
         "--frames $FRAMES --width $WW --height $HH --seed $SEED --device cuda" \
         "--workdir $d --out $d/video.mp4"
  } >> "$log"
  # VT_OP_PROVIDER_STATS=1 makes each op announce itself once when it resolves:
  # op=18 is kAttention, op=21 is kAttentionDenseFlash, device=1 is kCUDA.
  # Without it the only evidence of which arm ran is the wall clock, which is one
  # of the things being measured.
  export VLLM_LTX2_DIT_FLASH_ATTN="$knob"
  VT_OP_PROVIDER_STATS=1 timeout -s INT "$tmo" stdbuf -oL -eL "$BIN/ltx2-gen" \
    --pipeline-kind one_stage \
    --dit "$CKUSE/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
    --video-vae "$CKUSE/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae "$CKUSE/ltx-2.5-audio-vae-bf16.safetensors" \
    --checkpoint-class full \
    --encoder "$CKUSE/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
    --encoder-config "$BIN/ltx2_gemma4_text_config.json" \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "$WW" --height "$HH" --seed "$SEED" \
    --device cuda --workdir "$d" --out "$d/video.mp4" >> "$log" 2>&1 &
  local pid=$!
  # MEMORY FLOOR ONLY. No sample cap: the point of this run is a COMPLETED
  # render, and a cap is what left the previous attempt with no frames. No stack
  # sampler either: it would ptrace-stop the process and move the very per-forward
  # times the same run is reducing.
  local stopped_by="none" tick=0
  while kill -0 "$pid" 2>/dev/null; do
    local n avail
    n=$(sample_count "$log")
    avail=$(mem_avail_gib)   # the SAME reader phase [0b] gates on
    echo "$(date -u +%H:%M:%S)	$label	forwards=$n	memavail_gib=$avail" >> "$d/watch.tsv"
    # HEARTBEAT ON STDOUT, every ~2 minutes. The engine writes to its own log, so
    # without this the job produces NOTHING on stdout for up to two hours during
    # the naive render, and a lease with an idle timeout would kill a healthy run
    # that is doing exactly what it was asked to do. It doubles as progress: a
    # forward count that stops advancing is visible before the deadline, not
    # after it.
    tick=$((tick + 1))
    [ $((tick % 12)) = 1 ] && say "  [$label] forward $n, MemAvailable ${avail} GiB"
    if [ -n "$avail" ] && awk -v a="$avail" -v f="$MEM_FLOOR_GIB" 'BEGIN{exit !(a<f)}'; then
      stopped_by="memory-floor"; kill -INT "$pid" 2>/dev/null; break
    fi
    sleep 10
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  unset VLLM_LTX2_DIT_FLASH_ATTN
  local nf; nf=$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)
  say "render $label exit=$rc stopped_by=$stopped_by frames=$nf"
}

arm_report() {  # $1 label, $2 knob, $3 dir, $4 log -- runs for a RENDERED and a RESUMED arm
  local label=$1 knob=$2 d=$3 log=$4
  # THE TWO-SIDED ROUTING PROOF, per arm, from that arm's own log. One-sided
  # counting cannot tell a routed call from an added one: the flash arm must show
  # op=21 AND NOT op=18, and the naive arm the reverse. LTX's cross-attentions use
  # op=19 in both, which is why it is printed rather than asserted on.
  echo "--- op-provider selections (18 kAttention / 19 kAttentionCross / 21 kAttentionDenseFlash, device=1 CUDA) ---" | tee -a "$d/ARM"
  grep -E 'op-provider.*op=(18|19|20|21) device=1' "$log" | sort -u | sed 's/^/  /' | tee -a "$d/ARM"
  local n18 n21 routing
  n18=$(grep -cE 'op-provider.*op=18 device=1' "$log")
  n21=$(grep -cE 'op-provider.*op=21 device=1' "$log")
  echo "  op18_naive=$n18 op21_flash=$n21" | tee -a "$d/ARM"
  # THE VERDICT IS COMPUTED INTO A VARIABLE FIRST, and only then printed. It used
  # to be echoed inside a `case ... | tee` pipeline, where an `exit` would have
  # left the subshell and not the run -- so ROUTING_BAD was a word in a log and
  # nothing more. If the knob is not read, BOTH arms are flash, the two renders
  # come out bit-identical, and this file publishes PASS with all four thresholds
  # vacuous: the strongest positive verdict it can produce, from an experiment
  # that had one arm.
  case "$knob" in
    0) if [ "$n18" -ge 1 ] && [ "$n21" = 0 ]; then routing="OK"; else routing="BAD"; fi;;
    *) if [ "$n21" -ge 1 ] && [ "$n18" = 0 ]; then routing="OK"; else routing="BAD"; fi;;
  esac
  local want; case "$knob" in 0) want="op18>=1 and op21==0";; *) want="op21>=1 and op18==0";; esac
  if [ "$routing" = OK ]; then
    echo "  ROUTING_OK=$label (knob=$knob, want $want, saw op18=$n18 op21=$n21)" | tee -a "$d/ARM"
  else
    echo "  ROUTING_BAD=$label (knob=$knob, want $want, saw op18=$n18 op21=$n21)" | tee -a "$d/ARM"
    echo "FATAL: arm $label did not route as its knob asked. Both arms may be the" \
         "same arm, and an A/B of one configuration against itself reads as a" \
         "perfect match. Nothing further is measured on this run."
    exit 46
  fi
  # Per-forward MEDIAN from the engine's own `last=` lines. Never the governor,
  # which has reported 1.00 s, 69.1 s, 162 s and 396.9 s for this one quantity.
  grep -ohE 'last=[0-9.]+s' "$log" | sed 's/last=//;s/s$//' > "$d/samples.txt"
  sort -n "$d/samples.txt" | awk -v L="$label" '
    {a[NR]=$1; s+=$1}
    END{ if(!NR){print "  " L ": NO SAMPLES"; exit}
         m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2;
         printf "  %s: n=%d median=%.3fs mean=%.3fs min=%.3fs max=%.3fs\n", L, NR, m, s/NR, a[1], a[NR] }' | tee -a "$d/ARM"
  echo "  memavail low-water: $(memavail_low_water "$d/watch.tsv")" | tee -a "$d/ARM"
  echo "  frames=$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)" \
       "audio=$([ -s "$d/audio.wav" ] && stat -c %s "$d/audio.wav" || echo 0)" | tee -a "$d/ARM"
}

render flash     flash "${TMO_FLASH:-3600}"
render naive     0     "${TMO_NAIVE:-10800}"
render flash-ctl flash "${TMO_FLASH:-3600}"

say "=== [H] the speed pair, same binary, neither arm sampled ==="
# SAME LEASE IS A CLAIM, so it is stated rather than assumed. A resumed run
# reuses an arm's `last=` samples from the lease that produced them, and a ratio
# across two leases is not the same-binary same-lease pair section 7.1 owes.
if [ -n "$REUSED_ARMS" ]; then
  say "  NOT a same-lease pair: reused arms:$REUSED_ARMS (their per-forward samples were taken earlier)"
  echo "speed_pair_same_lease=no reused_arms=$REUSED_ARMS" >> "$OUT/PROVENANCE"
else
  say "  same binary, same lease: every arm was rendered in this run"
  echo "speed_pair_same_lease=yes" >> "$OUT/PROVENANCE"
fi
python3 - "$OUT/flash/samples.txt" "$OUT/naive/samples.txt" "$OUT/flash-ctl/samples.txt" <<'PY'
import sys, statistics
def med(p):
    try: v=[float(x) for x in open(p).read().split()]
    except OSError: v=[]
    return v, (statistics.median(v) if v else None)
f,fm = med(sys.argv[1]); n,nm = med(sys.argv[2]); c,cm = med(sys.argv[3])
for lab,v,m in (("flash",f,fm),("naive",n,nm),("flash-ctl",c,cm)):
    print(f"  {lab}: n={len(v)} median={m}")
if fm and nm:
    print(f"  SPEEDUP (naive median / flash median) = {nm/fm:.3f}x")
else:
    print("  INCOMPLETE: an arm produced no samples; report that, do not impute")
PY

say "=== [I] the pixel comparison ==="
# The tool is `$SRC/scripts/ltx25-render-compare.py`, so the thresholds it
# applies travel with the sources the binary was built from rather than being a
# copy that drifted from them. WHICH SOURCES THOSE ARE IS A FACT ABOUT THE RUN,
# not about this checkout: phase [B] unpacks `$W/pixab-src.tar.gz`, a tarball
# staged on the share, and records what it was in `source_sha`. Read that line
# before quoting a run's PIXEL_COMPARE_RC -- a tarball staged before exit 3
# existed carries a tool that cannot return one, and a degenerate control is a
# plain 0 there.
#
# Its exit status is the gate, and phase [L] below turns it into this job's:
# 0 every threshold held, 1 a threshold failed, 2 an input could not be read and
# nothing was compared, 3 the thresholds held and the control failed its own
# content checks. A 2 is never a pass, and a 3 is never a pass either: it is a
# result that exists and that nobody may READ.
#
# ARM A IS FLASH, and that is not cosmetic. The control is a repeat of FLASH, and
# the tool compares the control against the arm named by --control-of. This call
# used to pass `--a naive --b flash --control flash-ctl`, which made the
# "run-to-run noise floor" a SECOND naive-vs-flash comparison: it necessarily
# read about the same size as the treatment, and section 10.5's second branch
# would then have published "indistinguishable from run-to-run nondeterminism"
# whatever the kernel did. --control-of is passed explicitly rather than left to
# its default, so the wiring states its own intent.
python3 "$CMP" \
  --a "$OUT/flash" --b "$OUT/naive" --control "$OUT/flash-ctl" --control-of a \
  --label-a flash --label-b naive --label-control flash-ctl \
  --json "$OUT/pixel-compare.json" 2>&1 | tee "$OUT/pixel-compare.txt"
PIXEL_RC=${PIPESTATUS[0]}
echo "PIXEL_COMPARE_RC=$PIXEL_RC"
echo "pixel_compare_rc=$PIXEL_RC" >> "$OUT/PROVENANCE"

say "=== [J] the cross-check against the recorded 20260820 baseline ==="
# A different binary lineage (a50c57d69, an ancestor of the swap), so this is
# never the control and never the A/B. It answers one narrow question: whether a
# naive render is reproducible ACROSS builds, which bounds how much of any delta
# above could be everything-else-on-main rather than the kernel.
OLD=$FULL/out/20260820T223701Z/768x448-49f
if [ -d "$OLD" ]; then
  python3 "$CMP" --a "$OLD" --b "$OUT/naive" \
    --label-a baseline-20260820 --label-b naive \
    --json "$OUT/cross-check.json" 2>&1 | tee "$OUT/cross-check.txt"
  echo "CROSS_CHECK_RC=${PIPESTATUS[0]}"
else
  echo "  the recorded baseline is not on this share; cross-check SKIPPED"
fi

say "=== [K] artefacts ==="
for d in "$OUT"/flash "$OUT"/naive "$OUT"/flash-ctl; do
  [ -d "$d" ] || continue
  printf "  %-24s frames=%s audio=%s mp4=%s\n" "$(basename "$d")" \
    "$(ls "$d"/frame_*.ppm 2>/dev/null | wc -l)" \
    "$([ -s "$d/audio.wav" ] && echo yes || echo no)" \
    "$([ -s "$d/video.mp4" ] && echo yes || echo no)"
done
say "=== [L] the verdict, which is this job's exit status ==="
# THE GATE IS THE EXIT STATUS, and it used to be a line of text. [I]'s comment
# already said "its exit status is the gate ... a 2 is never a pass", and then
# the script printed PIXEL_COMPARE_RC, carried on through [J] and [K], and ended
# on DONE with status 0. A failing pixel verdict, an unreadable input and a
# zero-frame render all exited 0. [J] and [K] still run before this line, so the
# artefacts and the cross-check are produced either way -- what changes is that
# the run ends on the verdict rather than on the fact that it finished.
case "$PIXEL_RC" in
  0) say "PIXEL VERDICT: PASS -- every registered threshold held (see $OUT/pixel-compare.txt)";;
  1) say "PIXEL VERDICT: FAIL -- a registered threshold was not met. Section 10.5: this is a"
     say "               finding about a change already on main, and it does not owe a widened gate";;
  2) say "PIXEL VERDICT: UNREADABLE -- nothing was compared. This is never a pass";;
  3) say "PIXEL VERDICT: CONTROL DEGENERATE -- every registered threshold held, and the"
     say "               control rendered no picture, so it is not a noise floor and section"
     say "               10.5's R cannot be read. The renders are a PASS nobody may READ:"
     say "               re-take the control arm, and do not publish a reading from this run";;
  *) say "PIXEL VERDICT: UNKNOWN -- the comparison exited $PIXEL_RC, which it does not define";;
esac
say "DONE OUT=$OUT RUN_ID=$RUN_ID PIXEL_COMPARE_RC=$PIXEL_RC"
exit "$PIXEL_RC"
