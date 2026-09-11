#!/bin/bash
# DeepSeek-V4 Flash Vision W6 follow-up: the oracle's own NOISE FLOOR, and the
# vit-stage comparison the first run could not make.
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.
# Runs after dsv4v_w6_parity.sh, on the files it left in w6-parity/:
#
#   rc run -d thor:gpu0 --max-runtime 2h --idle-timeout 20m -- \
#     bash /workspace/dsv4-vision/w6-parity/src/tools/parity/dsv4v_w6_floor.sh
#
# Our processor narrows the normalised pixels to bf16 and llama.cpp keeps them
# f32, so the two towers never see the same input. The floor is
# oracle(f32 input) against oracle(bf16-rounded input): the distance the input
# rounding ALONE moves the oracle's output through 32 blocks. ours-vs-oracle is
# judged against that number rather than against a bound picked to pass.
set -u
W=/workspace/dsv4-vision
OUT=$W/w6-parity
OVL=$OUT/src/tools/parity
LC_PIN=9400c8946e4da5e7694f2c26d6d4e50e14b690fa
MMPROJ=$W/mmproj-BF16.gguf
LC=/tmp/dsv4v-w6f-llama
THREADS=${THREADS:-8}
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/floor-steps.txt"; }
cleanup() { rm -rf "$LC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
: > "$OUT/floor-steps.txt"
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S)"; done ) &
HB=$!
sha256sum "$OVL"/dsv4v_w6_* | tee "$OUT/floor-overlay.sha256"

git clone -q https://github.com/ggml-org/llama.cpp "$LC" && git -C "$LC" checkout -q "$LC_PIN"; step clone $?
[ "$(git -C "$LC" rev-parse HEAD)" = "$LC_PIN" ] || { step pin 96; exit 96; }
cp "$OVL/dsv4v_w6_oracle_dump.cpp" "$LC/tools/mtmd/"
printf '\nadd_executable(dsv4v-oracle-dump dsv4v_w6_oracle_dump.cpp)\ntarget_link_libraries(dsv4v-oracle-dump PRIVATE mtmd ggml)\n' >> "$LC/tools/mtmd/CMakeLists.txt"
cmake -S "$LC" -B "$LC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  > "$OUT/floor-configure.log" 2>&1; step configure $?
cmake --build "$LC/build" --target dsv4v-oracle-dump -j 4 > "$OUT/floor-build.log" 2>&1; RC=$?; step build $RC
[ $RC -eq 0 ] || { grep -m12 -E 'error' "$OUT/floor-build.log"; exit 93; }
ODUMP=$(find "$LC/build" -name dsv4v-oracle-dump -type f | head -1)

# A: the oracle again on f32 input (exact vit_out capture this time), to check
# it reproduces the first run's block bit for bit.
"$ODUMP" "$MMPROJ" "$OUT/img392.rgb" 392 392 0 "$OUT" f32in $THREADS > "$OUT/oracle-f32in.log" 2>&1; step oracle_f32in $?
grep -E 'captured|block|FATAL' "$OUT/oracle-f32in.log"
cmp "$OUT/oracle-f32in-block.f32" "$OUT/oracle-lp0-block.f32" && echo "REPRODUCIBLE: oracle block identical to the first run"
# B: the oracle on bf16-rounded input.
DSV4V_ROUND_INPUT_BF16=1 "$ODUMP" "$MMPROJ" "$OUT/img392.rgb" 392 392 0 "$OUT" bf16in $THREADS > "$OUT/oracle-bf16in.log" 2>&1; step oracle_bf16in $?
grep -E 'rounded|captured|block|FATAL' "$OUT/oracle-bf16in.log"

# FLOOR: "ours" := oracle(bf16 input), "oracle" := oracle(f32 input).
for s in block input vit cells; do
  cp "$OUT/oracle-bf16in-$s.f32" "$OUT/ours-floor-$s.f32"
  cp "$OUT/oracle-f32in-$s.f32" "$OUT/oracle-floor-$s.f32"
done
python3 "$OVL/dsv4v_w6_compare.py" "$OUT" floor 0 10 10 > "$OUT/compare-floor.txt" 2>&1; step compare_floor $?
cat "$OUT/compare-floor.txt"
# OURS vs the oracle on the SAME bf16 input: the tower difference with the
# input rounding taken out.
for s in block input vit cells; do
  cp "$OUT/ours-lp0-$s.f32" "$OUT/ours-samein-$s.f32"
  cp "$OUT/oracle-bf16in-$s.f32" "$OUT/oracle-samein-$s.f32"
done
python3 "$OVL/dsv4v_w6_compare.py" "$OUT" samein 0 10 10 > "$OUT/compare-samein.txt" 2>&1; step compare_samein $?
cat "$OUT/compare-samein.txt"
# The four lead_pad rungs again, now with the vit stage and the structure lines.
for LP in 0 1 2 3; do
  python3 "$OVL/dsv4v_w6_compare.py" "$OUT" lp$LP $LP 10 10 > "$OUT/compare-lp$LP.txt" 2>&1; step recompare_lp$LP $?
  grep -E '^\[vit\]|^structure|^\[block' "$OUT/compare-lp$LP.txt"
done
echo "### steps"; cat "$OUT/floor-steps.txt"
echo "### W6_FLOOR_DONE"
