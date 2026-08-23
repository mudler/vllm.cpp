#!/bin/bash
# LTX25-DIT-ATTN-FA2-HD128 (#1551) -- build ONE binary and measure BOTH rungs with it.
#
# DERIVED FROM scripts/ltx25-dit-attn-flash-ab.sh (#1549) and not re-derived.
# Phases A-D, the two watchdogs, the resume path, the CIFS chmod and the
# per-arm invocation header are that file's, together with the reasons each one
# exists. Read it for those. What is NEW here is only the ladder:
#
#   Arm A (fa2,   the default this change ships):   VLLM_LTX2_DIT_FLASH_ATTN unset
#   Arm B (flash, the DENOMINATOR, #1549 default):  VLLM_LTX2_DIT_FLASH_ATTN=flash
#   Arm C (naive, the 47.84 s floor, OPT-IN):       VLLM_LTX2_DIT_FLASH_ATTN=0
#
# Arm C is opt-in (ARMS="fa2 flash naive") because it costs ~1500 s for 13
# samples and its number already exists at n=119. The RATIO this row claims is
# B/A, and both of its arms run here every time.
#
# EACH ARM CALLS A DIFFERENT vt:: OP, which is the point of the three-way knob.
# `VT_OP_PROVIDER_STATS=1` therefore makes every arm state in its OWN log which
# rung it resolved -- op=18 kAttention, op=21 kAttentionDenseFlash, op=22
# kAttentionDenseFa2 -- so which kernel ran is READ rather than inferred from
# the timing, which is the quantity under measurement and cannot also be the
# evidence for itself.
#
# Same binary, same geometry, same seed, same prompt, same checkpoint, one lease.
# The statistic is the per-forward MEDIAN reduced from the engine's own `last=`
# lines. The governor/estimator output is NOT used: it has reported 1.00 s,
# 69.1 s, 162 s and 396.9 s for this one quantity.
set -u
T0=$(date +%s)
say() { echo "[ab +$(( $(date +%s) - T0 ))s] $*"; }
W=/workspace/ltx25-fa2hd128
FULL=/workspace/ltx25-fullmodel        # checkpoints and the text-encoder config
SRC=/root/src-fa2hd128
BLD=/root/build-fa2hd128
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
# SIX PRECONDITIONS, because a half-applied tree satisfies any five of them and
# still measures one thing twice while printing two columns. This is the #1549
# harness's rule with the ladder's extra rungs added, not a new rule.
#
#   1. the model calls the FA-2 op at all;
#   2. the knob COMPARES against "flash", or the denominator is a second FA-2 arm;
#   3. the flash arm CALLS vt::AttentionDenseFlash, which 2 alone does not imply:
#      the comparison can survive while the body it guards does not;
#   4. the knob is read at all, or the naive floor is unreachable;
#   5. the hd-128 INSTANTIATION is in the tree, and
#   6. it is in the build list -- without it `vt::AttentionDenseFa2` falls through
#      to AttentionDenseFlash by its own TOTAL contract, silently, and arm A
#      becomes a second arm B. That is the one failure this A/B cannot detect
#      from its own numbers, because a 1.00x ratio is also what "no speedup"
#      looks like.
#
# STATIC PRECONDITIONS ARE NOT SUFFICIENT ON THEIR OWN, whatever they count.
# They read the SOURCE, and what the ratio depends on is which op each arm
# RESOLVED AT RUN TIME. Phase [F] asserts that separately, per arm, from the
# arm's own log.
#
# COUNTED OVER CODE, NEVER OVER COMMENTS, and that distinction is not pedantry.
# The previous version of this block counted the bare string `"flash"`, which
# `ltx2_device.cpp` mentions in PROSE as well as in the one place that matters.
# It read 3 on a clean tree and STILL READ 2 with the entire
# `else if (std::strcmp(arm, "flash") == 0)` arm deleted, so its `-ge 1` passed
# on exactly the tree it exists to reject (review of #1551). `FA2OP` and `KNOB`
# had the same defect. Each count below now names the CALL EXPRESSION, over a
# copy with every `//` comment stripped, so only code can raise a count.
LTXSRC="$SRC/src/vllm/model_executor/models/ltx2_device.cpp"
[ -f "$LTXSRC" ] || { echo "FATAL: $LTXSRC is absent"; exit 40; }
CODE=$(sed 's|//.*||' "$LTXSRC")
FA2OP=$(printf '%s\n' "$CODE" | grep -c 'vt::AttentionDenseFa2(')
KFLASH=$(printf '%s\n' "$CODE" | grep -c 'strcmp(arm, "flash")')
KFLASHOP=$(printf '%s\n' "$CODE" | grep -c 'vt::AttentionDenseFlash(')
KNOB=$(printf '%s\n' "$CODE" | grep -c 'getenv("VLLM_LTX2_DIT_FLASH_ATTN")')
HD128SRC=0
[ -f "$SRC/src/vt/cuda/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu" ] && HD128SRC=1
HD128CM=$(grep -c 'flash_fwd_hdim128_bf16_sm80.cu' "$SRC/CMakeLists.txt")
echo "  vt::AttentionDenseFa2( call sites:      $FA2OP    (want >= 1)"
echo "  strcmp(arm, flash) comparisons:         $KFLASH   (want >= 1)"
echo "  vt::AttentionDenseFlash( call sites:    $KFLASHOP (want >= 1)"
echo "  getenv(VLLM_LTX2_DIT_FLASH_ATTN) sites: $KNOB     (want >= 1)"
echo "  hd-128 instantiation TU present:        $HD128SRC (want 1)"
echo "  hd-128 TU in the build list:            $HD128CM  (want >= 1)"
[ "$FA2OP"    -ge 1 ] || { echo "FATAL: #1551 is NOT in this source tree; no vt::AttentionDenseFa2( call site survives the comment strip"; exit 40; }
[ "$KFLASH"   -ge 1 ] || { echo "FATAL: the knob has no flash arm; the denominator would be a second FA-2 arm, and the ratio would read about 1.00x, which is also what 'no speedup' looks like"; exit 41; }
[ "$KFLASHOP" -ge 1 ] || { echo "FATAL: nothing calls vt::AttentionDenseFlash(; the denominator would be a second FA-2 arm"; exit 41; }
[ "$KNOB"     -ge 1 ] || { echo "FATAL: the A/B knob is NOT in this source tree; every arm would be one arm"; exit 41; }
[ "$HD128SRC" = 1 ]   || { echo "FATAL: flash_fwd_hdim128_bf16_sm80.cu is absent; FA-2 would fall through to flash"; exit 43; }
[ "$HD128CM"  -ge 1 ] || { echo "FATAL: the hd-128 TU is not in CMakeLists; FA-2 would fall through to flash"; exit 43; }

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
  for t in test_ops_attention test_ltx2_device test_ops_attention_dense_fa2; do
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
ninja -C "$BLD" -j 4 ltx2-gen test_ops_attention test_ltx2_device test_ops_attention_dense_fa2 > /root/build.log 2>&1
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
for t in test_ops_attention test_ltx2_device test_ops_attention_dense_fa2; do
  E=$(find "$BLD" -name "$t" -type f | head -1)
  [ -n "$E" ] && cp -f "$E" "$CACHE"/ && cp -f "$E" "$BIN"/
done
echo "$WANT_SHA" > "$CACHE/SRC_SHA"
echo "$BINSHA" > "$CACHE/BIN_SHA256"
say "staged the binary for a resumed run"
fi

say "=== [E] CORRECTNESS FIRST: the CUDA unit gates, before any speed number ==="
# AGENTS.md: establish the correctness gate before accepting a performance result.
for t in test_ops_attention test_ltx2_device test_ops_attention_dense_fa2; do
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

say "=== [F0] stage the checkpoints onto LOCAL disk ==="
# NEW relative to the #1549 harness, and it pays for itself in one arm. The four
# artefacts are 47.74 GiB and `/workspace` is CIFS: a load READ FROM THE SHARE
# has been measured at 589-1446 s (34-83 MiB/s), against ~32 s from the local
# overlay. This A/B pays a load PER ARM, so two arms off the share is 20-48
# minutes of lease spent on file copying rather than on the number. The copy is
# ~9 minutes, once.
#
# `/root` sits on the same 3.6T overlay `.agents/environment.md` records as
# writable with a real exec bit; `$FULL/ckpt` on the share is the fallback and
# is never modified. The staged copy is REUSED across runs, and each file is
# accepted only when its size matches the share's EXACTLY -- a partial copy from
# an interrupted run is otherwise indistinguishable from a complete one and
# would fail inside the loader minutes later.
CKSRC=$FULL/ckpt
CK=/root/ckpt
CKFILES="ltx-2.5-22b-dev-transformer-bf16.safetensors ltx-2.5-video-vae-conv-bf16.safetensors ltx-2.5-audio-vae-bf16.safetensors gemma4-12b-with-proj-nvfp4-torchao.safetensors"
NEED=0
for f in $CKFILES; do
  [ -f "$CKSRC/$f" ] || { echo "FATAL: $CKSRC/$f is absent"; exit 45; }
  NEED=$(( NEED + $(stat -c %s "$CKSRC/$f") ))
done
FREE=$(( $(df -k --output=avail /root | tail -1) * 1024 ))
say "checkpoints need $(( NEED / 1073741824 )) GiB; /root has $(( FREE / 1073741824 )) GiB free"
if [ "$FREE" -gt $(( NEED + 10737418240 )) ]; then
  mkdir -p "$CK"
  for f in $CKFILES; do
    want=$(stat -c %s "$CKSRC/$f")
    have=$(stat -c %s "$CK/$f" 2>/dev/null || echo 0)
    if [ "$have" = "$want" ]; then say "  cached $f"; continue; fi
    say "  copying $f ($(( want / 1073741824 )) GiB)"
    cp -f "$CKSRC/$f" "$CK/$f" || { echo "FATAL: copy of $f failed"; exit 46; }
    got=$(stat -c %s "$CK/$f")
    [ "$got" = "$want" ] || { echo "FATAL: $f staged $got want $want"; exit 46; }
  done
  say "checkpoints staged locally"
else
  say "NOT ENOUGH LOCAL SPACE: reading from the share, expect 10-24 min per load"
  CK=$CKSRC
fi

say "=== [F] the A/B ==="
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
  # op=22 is kAttentionDenseFa2, device=1 is kCUDA. Without it the only evidence
  # of which arm ran is the timing, which is the thing being measured.
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
    n=$(grep -c 'last=' "$log" 2>/dev/null || echo 0)
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
  echo "--- per-forward last= ---"
  grep -ohE 'last=[0-9.]+s' "$log" | sed 's/last=//;s/s$//' > "$OUT/samples-$label.txt"
  sort -n "$OUT/samples-$label.txt" | awk -v L="$label" '
    {a[NR]=$1; s+=$1}
    END{ if(!NR){print "  " L ": NO SAMPLES"; exit}
         m = (NR%2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2;
         printf "  %s: n=%d median=%.3fs mean=%.3fs min=%.3fs max=%.3fs\n", L, NR, m, s/NR, a[1], a[NR] }'
  echo "  raw sorted: $(sort -n "$OUT/samples-$label.txt" | tr '\n' ' ')"
  echo "  memavail low-water: $(awk -F'\t' '{gsub(/memavail_gib=/,"",$4); print $4}' "$OUT/watch-$label.tsv" 2>/dev/null | sort -n | head -1) GiB"
}

# WHICH RUNG RAN IS NOW ASSERTED, not printed. The previous version printed the
# `[vt op-provider]` lines under a header calling them the answer and then
# computed the ratio in phase [G] without consulting them, so an arm that had
# quietly resolved another arm's op produced a number that read as a result. It
# is the composed failure the review of #1551 named: if the flash arm regresses
# to the FA-2 default, arm B becomes a second arm A, the ratio reads about 1.00x,
# and that is indistinguishable from "no speedup" -- the A/B cannot report its
# own failure from its own numbers, so something outside the numbers has to.
#
# `vt::Announce` (src/vt/op_provider.cpp:262-269) fires ONCE per (op, device)
# slot on first resolution, so the presence of a line is the observation and its
# count is not. Each arm must resolve EXACTLY its own op among the three rungs:
# op=18 kAttention, op=21 kAttentionDenseFlash, op=22 kAttentionDenseFa2, all at
# device=1 (kCUDA). Resolving another rung's op is the failure; resolving NONE of
# them means the DiT self-attention was never reached at all, which is equally
# fatal to the number and reads identically in the timing.
#
# It runs AFTER `run_arm` returns, so the arm's samples and median are already
# written to $OUT and survive the exit.
assert_arm_op() {  # $1 = label, $2 = the op id this arm MUST resolve, $3 = the arm's log
  local label=$1 want=$2 log=$3
  local n18 n21 n22 seen
  n18=$(grep -cE '\[vt op-provider\] op=18 device=1' "$log" 2>/dev/null || true)
  n21=$(grep -cE '\[vt op-provider\] op=21 device=1' "$log" 2>/dev/null || true)
  n22=$(grep -cE '\[vt op-provider\] op=22 device=1' "$log" 2>/dev/null || true)
  seen=""
  [ "${n18:-0}" -gt 0 ] && seen="$seen 18"
  [ "${n21:-0}" -gt 0 ] && seen="$seen 21"
  [ "${n22:-0}" -gt 0 ] && seen="$seen 22"
  seen=${seen# }
  echo "  op-provider on device=1: resolved [${seen:-none}], want exactly [$want]" \
       "(18=kAttention 21=kAttentionDenseFlash 22=kAttentionDenseFa2)"
  if [ "$seen" != "$want" ]; then
    echo "FATAL: arm '$label' resolved [${seen:-none}] on device=1 and must resolve exactly [$want]."
    echo "FATAL:   18=kAttention  21=kAttentionDenseFlash  22=kAttentionDenseFa2"
    echo "FATAL: an arm that resolved another arm's op IS that arm, so the ratio it feeds is"
    echo "FATAL: 1.00x by construction -- which is also what 'no speedup' looks like. Resolving"
    echo "FATAL: none of the three means the DiT self-attention was never reached."
    echo "FATAL: the [vt op-provider] lines this arm did emit:"
    grep -E '\[vt op-provider\]' "$log" | sort -u | sed 's/^/FATAL:   /'
    exit 47
  fi
}

# THE DENOMINATOR FIRST, and for the reason the #1549 harness records: the arm
# most likely to be cut short is the one to take while the box is known to be
# up. Here that is `flash`, which is the slower of the two rungs the ratio needs
# -- lose it and there is no ratio, only a number. `naive` is opt-in and runs
# last because its value already exists at n=119 and it costs ~1500 s.
ARMS="${ARMS:-flash fa2}"
say "arms: $ARMS"
for a in $ARMS; do
  case "$a" in
    flash) run_arm flash "${TMO_FLASH:-1200}" "flash"; assert_arm_op flash 21 "$OUT/arm-flash.log" ;;
    fa2)   run_arm fa2   "${TMO_FA2:-1200}"   ""     ; assert_arm_op fa2   22 "$OUT/arm-fa2.log"   ;;
    naive) run_arm naive "${TMO_NAIVE:-1800}" "0"    ; assert_arm_op naive 18 "$OUT/arm-naive.log" ;;
    *)     echo "FATAL: unknown arm '$a' (want flash|fa2|naive)"; exit 44 ;;
  esac
done

say "=== [G] the ratio ==="
python3 - "$OUT" <<'PY'
import os, sys, statistics
out = sys.argv[1]
def med(name):
    p = os.path.join(out, "samples-%s.txt" % name)
    if not os.path.exists(p):
        return None, None
    v = [float(x) for x in open(p).read().split()]
    return v, (statistics.median(v) if v else None)
res = {}
for name in ("naive", "flash", "fa2"):
    v, m = med(name)
    if v is None:
        continue
    res[name] = (v, m)
    print("  %-5s: n=%d median=%s" % (name, len(v), m))
# THE CLAIM this row makes is flash/fa2, measured here, on one binary, in one
# lease. Nothing else is a claim: the 47.84 s line below is a COMPARISON against
# a number taken in another run and is labelled as such, never as an A/B.
fm = res.get("flash", (None, None))[1]
am = res.get("fa2", (None, None))[1]
nm = res.get("naive", (None, None))[1]
if fm and am:
    print("  SPEEDUP, THIS A/B (flash median / fa2 median) = %.2fx" % (fm / am))
else:
    print("  INCOMPLETE: the flash/fa2 pair is not both present; report that, do not impute")
if nm and am:
    print("  and naive/fa2, same binary, same lease            = %.2fx" % (nm / am))
if am:
    print("  vs the RECORDED 47.84 s naive number (NOT this A/B) = %.2fx" % (47.84 / am))
PY
say "DONE OUT=$OUT"
