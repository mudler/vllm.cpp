#!/bin/bash
# DeepSeek-V4 Flash Vision W6: is the gap to llama.cpp COMPUTE DTYPE?
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.
# Runs after dsv4v_w6_floor.sh, which leaves oracle-f32in-*.f32 (the oracle on
# f32 input, exact vit_out capture) in w6-parity/.
#
#   rc run -d thor:gpu0 --max-runtime 2h --idle-timeout 20m -- \
#     bash /workspace/dsv4-vision/w6-parity/src/tools/parity/dsv4v_w6_f32.sh
#
# A SCRATCH MEASUREMENT. W2 refuses `compute_dtype != bf16`, so this script
# deletes that guard IN THE EXTRACTED COPY ONLY and runs the probe's f32 arm:
# the same tower with f32 weights and f32 activations. Nothing here is a
# product change and nothing here is committed to src/.
#
# A same-binary bf16 CONTROL runs too and must reproduce the first W6 run's
# block byte for byte, which proves deleting the guard changed nothing on the
# production bf16 path.
#
# `pipefail` is required for the same reason `dsv4v_w6_parity.sh` states: a
# `cmd | tee f` pipeline otherwise reports TEE's status to `$?`, and tee
# succeeds whenever it can write the file. No `set -e` here, so this changes
# only the value the status readers below see.
set -uo pipefail
W=/workspace/dsv4-vision
OUT=$W/w6-parity
OVL=$OUT/src/tools/parity
TAR=$W/dsv4v-src-4993c72b2.tar
TAR_SHA=ca7dddc141fbb11c8f004fd62682f0484ad1ae4b7f7d5a7ff3f0f00c5c121ede
MMPROJ=$W/mmproj-BF16.gguf
SRC=/tmp/dsv4v-w6x-src
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/f32-steps.txt"; }
cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
: > "$OUT/f32-steps.txt"
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S)"; done ) &
HB=$!
test -f "$OUT/oracle-f32in-block.f32" || { echo "FATAL: run dsv4v_w6_floor.sh first"; step prereq 97; exit 97; }

test "$(sha256sum "$TAR" | awk '{print $1}')" = "$TAR_SHA" || { step tarsha 91; exit 91; }
rm -rf "$SRC"; mkdir -p "$SRC" && tar -xf "$TAR" -C "$SRC" || { step untar 92; exit 92; }
cp "$OVL"/dsv4v_w6_* "$SRC/tools/parity/"
sha256sum "$SRC"/tools/parity/dsv4v_w6_* | tee "$OUT/f32-overlay.sha256"; step overlay_sha "${PIPESTATUS[0]}"
printf '\nadd_executable(dsv4v-w6-probe ${CMAKE_SOURCE_DIR}/tools/parity/dsv4v_w6_probe.cpp)\ntarget_link_libraries(dsv4v-w6-probe PRIVATE vllm::vllm)\n' >> "$SRC/examples/CMakeLists.txt"

echo "### scratch patch: delete the bf16-only guard"
python3 - "$SRC/src/vllm/model_executor/models/deepseek_v4_vision.cpp" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
guard = ('  if (config.compute_dtype != DType::kBF16) {\n'
         '    Invalid("DeepSeek-V4 vision compute dtype must be bf16");\n'
         '  }\n')
assert s.count(guard) == 1, "guard not found exactly once"
open(p, "w").write(s.replace(guard, "  // W6 SCRATCH: bf16-only guard deleted for the f32 measurement\n"))
print("guard deleted")
PY
step patch $?

cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF \
  > "$OUT/f32-configure.log" 2>&1; step configure $?
cmake --build "$SRC/build" --target dsv4v-w6-probe -j 4 > "$OUT/f32-build.log" 2>&1; RC=$?; step build $RC
[ $RC -eq 0 ] || { grep -m12 'error' "$OUT/f32-build.log"; exit 94; }
PROBE=$(find "$SRC/build" -name dsv4v-w6-probe -type f | head -1)

echo "### control: the production bf16 path from the patched binary"
"$PROBE" "$MMPROJ" "$OUT/img392.rgb" 392 392 0 "$OUT" bf16ctl > "$OUT/ours-bf16ctl.log" 2>&1; step control $?
# THIS CONTROL IS LOAD-BEARING and now records a step. It is what proves the
# scratch patch that deletes the bf16-only guard changed NOTHING on the
# production bf16 path; without it the f32 number below is measured by a binary
# nobody has shown to be equivalent on the shipped arm.
cmp "$OUT/ours-bf16ctl-block.f32" "$OUT/ours-lp0-block.f32"; step control_identical $?
echo "CONTROL_IDENTICAL (rc above): bf16 path unchanged by the scratch patch"

echo "### f32 arm"
DSV4V_PROBE_F32=1 "$PROBE" "$MMPROJ" "$OUT/img392.rgb" 392 392 0 "$OUT" f32 > "$OUT/ours-f32.log" 2>&1; step f32 $?
tail -3 "$OUT/ours-f32.log"
# EVERY COPY IS CHECKED and the destination removed first: $OUT is a persistent
# NAS directory nothing clears, so a failed `cp` left the PREVIOUS run's file
# for the comparator to judge.
CPRC=0
for s in block input vit cells; do
  rm -f "$OUT/oracle-f32-$s.f32"
  cp "$OUT/oracle-f32in-$s.f32" "$OUT/oracle-f32-$s.f32" || CPRC=1
done
step copy_f32 $CPRC
python3 "$OVL/dsv4v_w6_compare.py" "$OUT" f32 0 10 10 > "$OUT/compare-f32.txt" 2>&1; step compare_f32 $?
cat "$OUT/compare-f32.txt"
# And ours-f32 against ours-bf16: how far our own dtype moves our own output.
CPRC=0
for s in block input vit cells; do
  rm -f "$OUT/oracle-selfdt-$s.f32" "$OUT/ours-selfdt-$s.f32"
  cp "$OUT/ours-f32-$s.f32" "$OUT/oracle-selfdt-$s.f32" || CPRC=1
  cp "$OUT/ours-lp0-$s.f32" "$OUT/ours-selfdt-$s.f32" || CPRC=1
done
step copy_selfdt $CPRC
python3 "$OVL/dsv4v_w6_compare.py" "$OUT" selfdt 0 10 10 > "$OUT/compare-selfdt.txt" 2>&1; step compare_selfdt $?
cat "$OUT/compare-selfdt.txt"
echo "### steps"; cat "$OUT/f32-steps.txt"
# READ THE STEPS BACK; see the same block in dsv4v_w6_parity.sh. `compare_f32`
# carries condition (2), the f32 arm inside the oracle's own floor, which is the
# condition that tests the function -- so a regression there fails this job.
# COUNTING NON-ZERO LINES IS NOT ENOUGH: `awk '!/ RC=0$/' | wc -l` counts an
# ABSENT step as zero failures, so a steps file in which `compare_f32` never ran
# at all still passed, and so did an empty file.
EXPECTED="overlay_sha patch configure build control control_identical f32
          copy_f32 compare_f32 copy_selfdt compare_selfdt"
BAD=0
if [ ! -s "$OUT/f32-steps.txt" ]; then
  echo "### FATAL: f32-steps.txt is empty or absent -- NOTHING was recorded"; BAD=1
else
  if grep -qvE '^[A-Za-z0-9_]+ RC=[0-9]+$' "$OUT/f32-steps.txt"; then
    echo "### MALFORMED STEP LINES:"; grep -vE '^[A-Za-z0-9_]+ RC=[0-9]+$' "$OUT/f32-steps.txt"; BAD=1
  fi
  if awk '!/ RC=0$/' "$OUT/f32-steps.txt" | grep -q .; then
    echo "### FAILING STEPS:"; awk '!/ RC=0$/' "$OUT/f32-steps.txt"; BAD=1
  fi
  for s in $EXPECTED; do
    grep -qE "^$s RC=" "$OUT/f32-steps.txt" \
      || { echo "### MISSING EXPECTED STEP: $s -- it never ran"; BAD=1; }
  done
fi
echo "### W6_F32_DONE failed_steps=$BAD"
[ "$BAD" -eq 0 ] || exit 1
