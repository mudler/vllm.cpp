#!/bin/bash
# LTX25-DIT-ATTN-FLASH (#1549) -- build ONE binary and measure BOTH arms with it.
#
# COMMITTED HERE, and the reason is a defect in the run that produced this row's
# only number. The 7.680 s flash arm was taken by an EARLIER version of this file
# living only at /mnt/nas_share/rc/ltx25-attnflash/job/ab.sh, which was then
# EDITED IN PLACE after the run; its `arm-flash.log` opens at `[render] + load`
# with no command line, its working directory is empty and it wrote no
# phase-log.json, so the geometry, prompt, seed and sample cap behind that number
# are not recoverable from its own artifacts. The 47.84 s denominator has its
# full command line as line 1 of its render.log and this arm does not, which is
# the asymmetry. A mutable path on a share is not a record. This file is, and
# `.agents/specs/ltx25-dit-attn-flash.md` §7.1 names the sha256 of the exact
# revision each future number was taken with.
#
# Arm A (flash, the arm #1549 shipped):          VLLM_LTX2_DIT_FLASH_ATTN=flash
# Arm B (naive, the denominator):                VLLM_LTX2_DIT_FLASH_ATTN=0
#
# ARM A SAID `unset` UNTIL #1751, AND UNSET STOPPED MEANING FLASH. At #1549 the
# knob was BINARY and flash WAS the default, so leaving the variable unset was
# the flash arm and this line was true. #1551 made the knob THREE-WAY, moved the
# default up a rung to `vt::AttentionDenseFa2` and gave the flash rung the exact
# spelling `flash` -- so from that commit this arm rendered FA-2 and reported it
# under the name `flash`. This file, unlike `ltx25-dit-attn-fa2-hd128-ab.sh`,
# has NO `assert_arm_op`: phase [F] PRINTS the op-provider selections and asserts
# nothing about them, so the substitution was silent and the ratio it published
# would have been FA-2 against naive while its own header said flash against
# naive. `tests/scripts/test_ltx2_dit_attn_knob_arms.py` now holds every value
# this file sets against the arms the dispatch actually parses, and #1751 makes
# an unrecognised value refuse by name rather than default.
#
# Same binary, same geometry, same seed, same prompt, same checkpoint, one lease.
# The statistic is the per-forward MEDIAN reduced from the engine's own `last=`
# lines. The governor/estimator output is NOT used: it has reported 1.00 s,
# 69.1 s, 162 s and 396.9 s for this one quantity.
#
# Phases A-D are lifted from ltx25-fullmodel/job/build_and_render.sh, which
# records why each guard exists. They are not re-derived here.
set -u
T0=$(date +%s)
say() { echo "[ab +$(( $(date +%s) - T0 ))s] $*"; }

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

W=/workspace/ltx25-attnflash
FULL=/workspace/ltx25-fullmodel        # checkpoints and the text-encoder config
SRC=/root/src-attnflash
BLD=/root/build-attnflash
OUT=$W/out/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
export DEBIAN_FRONTEND=noninteractive
say "OUT=$OUT"
echo "rc_job=${RC_JOB_ID:-unknown}" > "$OUT/PROVENANCE"
# The HARNESS is part of the provenance, not context the reader is assumed to
# have. Its previous copy was edited after the run it describes, so a number was
# left with no recoverable recipe (#1549).
echo "harness_sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')" >> "$OUT/PROVENANCE"

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
export PATH="$TKLIB/bin:$PATH" CUDAToolkit_ROOT="$TKLIB"
say "CUDAToolkit_ROOT=$TKLIB"
nvcc --version | tail -2

say "=== [B] source, and the PRECONDITION that it carries this change ==="
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$W/src.tar.gz" -C "$SRC" || { echo "FATAL: cannot unpack source"; exit 31; }
cat "$W/src.sha" 2>/dev/null | sed 's/^/  built_from=/'
# BOTH sides, because a half-applied tree satisfies either alone: the swapped op
# without the knob makes the naive arm a second flash arm, and the knob without
# the swap makes the flash arm a second naive one. Either way the A/B measures
# one thing twice and still prints two columns.
#
# THERE IS NO THIRD PRECONDITION, and there used to be. It grepped `cuda_ops.cu`
# for a shared-memory opt-in that #1549 has since REVERTED (the row's section
# 4.3: LTX renders bf16 at head_dim 128, whose K/V tile is 32,768 B and fits the
# 49,152 B every CUDA architecture gives without an opt-in, so the raise was
# never what this measurement needed). It also grepped for `FlashTileSmemOptIn`,
# a spelling that never existed in any revision of that change -- the function
# was called `SetDynamicSmemOptIn` -- so the check counted 0 and `exit 42`-ed on
# a correct tree as readily as on a wrong one. A precondition that cannot pass is
# not a stricter precondition.
NEWOP=$(grep -c 'vt::AttentionDenseFlash' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
KNOB=$(grep -c 'VLLM_LTX2_DIT_FLASH_ATTN' "$SRC/src/vllm/model_executor/models/ltx2_device.cpp")
echo "  AttentionDenseFlash call sites: $NEWOP (want >= 1)"
echo "  A/B knob sites:                 $KNOB (want >= 1)"
[ "$NEWOP" -ge 1 ] || { echo "FATAL: #1549 is NOT in this source tree"; exit 40; }
[ "$KNOB"  -ge 1 ] || { echo "FATAL: the A/B knob is NOT in this source tree; both arms would be one arm"; exit 41; }

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
# RESUME PATH. The 20260821T092516Z run lost the worker MID-MEASUREMENT after
# paying an 18-minute build, and a retry that pays it again spends the lease on
# a compile rather than on the number. The staged binary is reused only when the
# CACHED SOURCE SHA MATCHES the tarball's, so a resumed run can never measure a
# tree other than the one it claims -- which is the failure this guard exists to
# prevent, not the rebuild.
CACHE="$W/bin"
WANT_SHA=$(cat "$W/src.sha" 2>/dev/null)
BIN=/root/abbin; mkdir -p "$BIN"
if [ -s "$CACHE/ltx2-gen" ] && [ -s "$CACHE/libvllm.so.0.0.3" ] && \
   [ -n "$WANT_SHA" ] && [ "$(cat "$CACHE/SRC_SHA" 2>/dev/null)" = "$WANT_SHA" ]; then
  say "REUSING the staged binary: cached SRC_SHA matches $WANT_SHA"
  # The share serves file_mode=0664, so a binary copied off it arrives
  # NON-EXECUTABLE. chmod, or the run dies with exit 126 and reads as "missing".
  cp -f "$CACHE/ltx2-gen" "$BIN"/ && chmod 0755 "$BIN/ltx2-gen"
  cp -f "$CACHE/libvllm.so.0.0.3" "$BIN"/ && chmod 0755 "$BIN/libvllm.so.0.0.3"
  cp -f "$CACHE/ltx2_gemma4_text_config.json" "$BIN"/ 2>/dev/null || \
    cp -f "$FULL/bin/ltx2_gemma4_text_config.json" "$BIN"/
  ( cd "$BIN" && ln -sf libvllm.so.0.0.3 libvllm.so.0 && ln -sf libvllm.so.0.0.3 libvllm.so )
  for t in test_ops_attention test_ltx2_device; do
    [ -s "$CACHE/$t" ] && cp -f "$CACHE/$t" "$BIN"/ && chmod 0755 "$BIN/$t"
  done
  BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
  say "ONE BINARY, both arms: sha256=$BINSHA (cached)"
  echo "binary_sha256=$BINSHA" >> "$OUT/PROVENANCE"
  echo "binary_source=cache" >> "$OUT/PROVENANCE"
  export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
  SKIP_BUILD=1
else
  SKIP_BUILD=0
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
# Unconstrained parallelism has OOM-rebooted this box.
ninja -C "$BLD" -j 4 ltx2-gen test_ops_attention test_ltx2_device > /root/build.log 2>&1
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
BINSHA=$(sha256sum "$BIN/ltx2-gen" | awk '{print $1}')
say "ONE BINARY, both arms: sha256=$BINSHA"
echo "binary_sha256=$BINSHA" >> "$OUT/PROVENANCE"
export LD_LIBRARY_PATH="$BIN:$TKLIB/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$CACHE"
cp -f "$BIN/ltx2-gen" "$CACHE"/ && cp -f "$BIN/libvllm.so.0.0.3" "$CACHE"/
cp -f "$BIN/ltx2_gemma4_text_config.json" "$CACHE"/ 2>/dev/null
for t in test_ops_attention test_ltx2_device; do
  E=$(find "$BLD" -name "$t" -type f | head -1)
  [ -n "$E" ] && cp -f "$E" "$CACHE"/ && cp -f "$E" "$BIN"/
done
echo "$WANT_SHA" > "$CACHE/SRC_SHA"
echo "$BINSHA" > "$CACHE/BIN_SHA256"
say "staged the binary for a resumed run"
fi

say "=== [E] CORRECTNESS FIRST: the CUDA unit gates, before any speed number ==="
# AGENTS.md: establish the correctness gate before accepting a performance result.
for t in test_ops_attention test_ltx2_device; do
  EXE=$(find "$BLD" -name "$t" -type f 2>/dev/null | head -1)
  [ -n "$EXE" ] || { [ -x "$BIN/$t" ] && EXE="$BIN/$t"; }
  if [ -z "$EXE" ]; then echo "  MISSING $t"; continue; fi
  say "--- $t ---"
  "$EXE" > "$OUT/$t.log" 2>&1
  RC=$?
  # `assertions: 0` is a skip wearing a pass, and a `Status:` line is the only
  # place a thrown case shows up, so both are printed rather than a grep for ok.
  grep -E 'assertions:|test cases:|Status:|SKIP' "$OUT/$t.log" | tail -8
  echo "  ${t}_RC=$RC"
done

say "=== [F] the A/B ==="
CK=/root/ckpt
[ -d "$CK" ] || CK=$FULL/ckpt
say "checkpoints from $CK"
PROMPT='A golden retriever shakes water from its coat on a sunlit lawn.'
FRAMES=${FRAMES:-49}; WW=${WW:-768}; HH=${HH:-448}; SEED=${SEED:-20260820}
# tokens = (w/32) * (h/32) * ((frames-1)/8 + 1)
TOK=$(( (WW/32) * (HH/32) * (((FRAMES-1)/8) + 1) ))
say "geometry ${WW}x${HH}/${FRAMES}f = $TOK video tokens (the 47.84 s denominator's geometry)"
WANT_SAMPLES=${WANT_SAMPLES:-13}
MEM_FLOOR_GIB=${MEM_FLOOR_GIB:-12.0}
say "sample cap ${WANT_SAMPLES} per arm; MemAvailable floor ${MEM_FLOOR_GIB} GiB"

run_arm() {  # $1 = label, $2 = timeout seconds, $3 = knob value ("" = unset)
  local label=$1 tmo=$2 knob=$3
  local log="$OUT/arm-$label.log"
  say "--- arm $label (VLLM_LTX2_DIT_FLASH_ATTN=${knob:-<unset>}, hard cap ${tmo}s, stop at ${WANT_SAMPLES} samples) ---"
  # VT_OP_PROVIDER_STATS=1 makes each arm state, in its OWN log, which op it
  # actually resolved: op=18 is kAttention, op=21 is kAttentionDenseFlash,
  # device=1 is kCUDA. Without it the only evidence of which arm ran is the
  # timing, which is the thing being measured.
  if [ -n "$knob" ]; then export VLLM_LTX2_DIT_FLASH_ATTN="$knob"; else unset VLLM_LTX2_DIT_FLASH_ATTN; fi
  local wd="$OUT/wd-$label"; mkdir -p "$wd"
  : > "$log"
  # EVERY ARM STATES ITS OWN INVOCATION, on line 1 of its own log, the way the
  # 47.84 s denominator's render.log does. The previous version of this script
  # did not, so the geometry, prompt, seed and sample cap behind its 7.680 s were
  # unverifiable from the artifacts it left and had to be inferred from a
  # SEPARATE, mutable copy of the harness that was edited after the run. An
  # arm whose log cannot say what it ran is not evidence, whatever number it
  # contains. `set -x` is deliberately not used: it would interleave the trace
  # with the engine's own progress lines, which are the samples being reduced.
  {
    echo "[arm] label=$label knob=${knob:-<unset>} tmo=${tmo}s want_samples=$WANT_SAMPLES"
    echo "[arm] harness=$0 sha256=$(sha256sum "$0" 2>/dev/null | awk '{print $1}')"
    echo "[arm] binary=$BIN/ltx2-gen sha256=${BINSHA:-unknown} src_sha=${WANT_SHA:-unknown}"
    echo "[arm] geometry=${WW}x${HH}/${FRAMES}f tokens=$TOK seed=$SEED ckpt=$CK"
    echo "[arm] prompt=<<$PROMPT>>"
    echo "[arm] cmd: $BIN/ltx2-gen --pipeline-kind one_stage --dit $CK/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
         "--video-vae $CK/ltx-2.5-video-vae-conv-bf16.safetensors --audio-vae $CK/ltx-2.5-audio-vae-bf16.safetensors" \
         "--checkpoint-class full --encoder $CK/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
         "--encoder-config $BIN/ltx2_gemma4_text_config.json --prompt <see above>" \
         "--frames $FRAMES --width $WW --height $HH --seed $SEED --device cuda" \
         "--workdir $wd --out $wd/video.mp4"
  } >> "$log"
  VT_OP_PROVIDER_STATS=1 timeout -s INT "$tmo" stdbuf -oL -eL "$BIN/ltx2-gen" \
    --pipeline-kind one_stage \
    --dit "$CK/ltx-2.5-22b-dev-transformer-bf16.safetensors" \
    --video-vae "$CK/ltx-2.5-video-vae-conv-bf16.safetensors" \
    --audio-vae "$CK/ltx-2.5-audio-vae-bf16.safetensors" \
    --checkpoint-class full \
    --encoder "$CK/gemma4-12b-with-proj-nvfp4-torchao.safetensors" \
    --encoder-config "$BIN/ltx2_gemma4_text_config.json" \
    --prompt "$PROMPT" \
    --frames "$FRAMES" --width "$WW" --height "$HH" --seed "$SEED" \
    --device cuda --workdir "$wd" --out "$wd/video.mp4" >> "$log" 2>&1 &
  local pid=$!

  # TWO WATCHDOGS, and the first one is why this run exists at all. The previous
  # attempt (20260821T092516Z) had neither: the flash arm was producing clean
  # samples at forward 20 and then the WORKER WAS LOST -- `rc` reported
  # dgx:gpu0 `unhealthy (no contact)` -- so the naive arm never ran and there was
  # no A/B, only one arm. GB10 shares host RAM with the GPU and an unconstrained
  # job has OOM-rebooted this box before, so a render that keeps running past the
  # samples it needs is spending risk for nothing.
  #
  #   (1) SAMPLE CAP. Stop the arm the moment it has WANT_SAMPLES `last=` lines.
  #       A median needs a dozen samples, not a finished render.
  #   (2) MEMORY FLOOR. Stop the arm if MemAvailable falls under MEM_FLOOR_GIB.
  #
  # SIGINT, not SIGKILL, and the PID is the one bash gave us -- matching on the
  # program NAME would also hit another session's copy on a shared box.
  local stopped_by="none"
  while kill -0 "$pid" 2>/dev/null; do
    local n avail
    n=$(sample_count "$log")
    avail=$(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo 2>/dev/null)
    echo "$(date -u +%H:%M:%S)	$label	samples=$n	memavail_gib=$avail" >> "$OUT/watch-$label.tsv"
    if [ "${n:-0}" -ge "$WANT_SAMPLES" ]; then stopped_by="sample-cap"; kill -INT "$pid" 2>/dev/null; break; fi
    if [ -n "$avail" ] && awk -v a="$avail" -v f="$MEM_FLOOR_GIB" 'BEGIN{exit !(a<f)}'; then
      stopped_by="memory-floor"; kill -INT "$pid" 2>/dev/null; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  unset VLLM_LTX2_DIT_FLASH_ATTN
  say "arm $label exit=$rc stopped_by=$stopped_by"
  say "  a non-zero exit is EXPECTED: the arm is capped at enough forwards for a"
  say "  median and interrupted, never run to a finished render."
  echo "--- op-provider selections (op=18 naive / op=21 flash, device=1 CUDA) ---"
  grep -E 'op-provider.*op=(18|19|20|21) device=1' "$log" | sort -u | sed 's/^/  /'
  echo "--- per-forward last= ---"
  grep -ohE 'last=[0-9.]+s' "$log" | sed 's/last=//;s/s$//' > "$OUT/samples-$label.txt"
  sort -n "$OUT/samples-$label.txt" | awk -v L="$label" '
    {a[NR]=$1; s+=$1}
    END{ if(!NR){print "  " L ": NO SAMPLES"; exit}
         m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2;
         printf "  %s: n=%d median=%.3fs mean=%.3fs min=%.3fs max=%.3fs\n", L, NR, m, s/NR, a[1], a[NR] }'
  echo "  raw sorted: $(sort -n "$OUT/samples-$label.txt" | tr '\n' ' ')"
  echo "  memavail low-water: $(memavail_low_water "$OUT/watch-$label.tsv")"
}

# NAIVE FIRST this time. The previous attempt ran the cheap arm first and lost
# the worker before the expensive one, which left one arm and no A/B. The arm
# that is most likely to be cut short is the one that should be taken while the
# box is known to be up.
run_arm naive "${TMO_NAIVE:-1500}" "0"
run_arm flash "${TMO_FLASH:-1200}" "flash"

say "=== [G] the ratio ==="
python3 - "$OUT/samples-flash.txt" "$OUT/samples-naive.txt" <<'PY'
import sys, statistics
def med(p):
    v=[float(x) for x in open(p).read().split()]
    return v, (statistics.median(v) if v else None)
f,fm = med(sys.argv[1]); n,nm = med(sys.argv[2])
print(f"  flash: n={len(f)} median={fm}")
print(f"  naive: n={len(n)} median={nm}")
if fm and nm:
    print(f"  SPEEDUP (naive median / flash median) = {nm/fm:.2f}x")
    print(f"  and against the recorded 47.84 s denominator = {47.84/fm:.2f}x")
else:
    print("  INCOMPLETE: one arm produced no samples; report that, do not impute")
PY
say "DONE OUT=$OUT"
