#!/bin/bash
# DeepSeek-V4 Flash Vision W6 real-weight parity job, run on thor:gpu0 through rc.
#
# Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.
#
#   rc run -d thor:gpu0 --max-runtime 4h --idle-timeout 20m -- \
#     bash /workspace/dsv4-vision/w6-parity/src/tools/parity/dsv4v_w6_parity.sh
#
# Both sides run on the CPU. The vision tower is compared, not a device path.
# Our side is the row head tarball pinned by sha256, with the W6 probe files
# overlaid from w6-parity/src (their sha256s are printed). The oracle side is a
# fresh clone of ggml-org/llama.cpp at release b10766, built CPU-only and static.
set -u
W=/workspace/dsv4-vision
OUT=$W/w6-parity; mkdir -p "$OUT"
OVL=$OUT/src
TAR=$W/dsv4v-src-4993c72b2.tar
TAR_SHA=ca7dddc141fbb11c8f004fd62682f0484ad1ae4b7f7d5a7ff3f0f00c5c121ede
LC_PIN=9400c8946e4da5e7694f2c26d6d4e50e14b690fa
MMPROJ=$W/mmproj-BF16.gguf
LANG=$W/UD-IQ1_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_S-00001-of-00003.gguf
SRC=/tmp/dsv4v-w6p-src
LC=/tmp/dsv4v-w6p-llama
NEED_GB=${NEED_GB:-30}
THREADS=${THREADS:-8}
RUN_CLI=${RUN_CLI:-1}
free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
step() { echo "### STEP $1 RC=$2"; echo "$1 RC=$2" >> "$OUT/steps.txt"; }
cleanup() { rm -rf "$SRC" "$LC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM
: > "$OUT/steps.txt"
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G mem=$(free -g | awk '/Mem:/{print $3}')G"; done ) &
HB=$!

echo "### identity $(date -u +%FT%TZ)"; uname -m; nproc; free -g | head -2
df -h /tmp; rm -rf "$SRC" "$LC"
[ "$(free_gb)" -ge "$NEED_GB" ] || { echo "REFUSING: /tmp $(free_gb)G < ${NEED_GB}G"; step disk 95; exit 95; }

echo "### our source: pinned tarball + W6 overlay"
test "$(sha256sum "$TAR" | awk '{print $1}')" = "$TAR_SHA" || { step tarsha 91; exit 91; }
mkdir -p "$SRC" && tar -xf "$TAR" -C "$SRC" || { step untar 92; exit 92; }
mkdir -p "$SRC/tools/parity"
for f in dsv4v_w6_probe.cpp dsv4v_w6_oracle_dump.cpp dsv4v_w6_image.py dsv4v_w6_compare.py dsv4v_w6_parity.sh; do
  cp "$OVL/tools/parity/$f" "$SRC/tools/parity/$f" || { step overlay 92; exit 92; }
done
sha256sum "$SRC"/tools/parity/dsv4v_w6_* | tee "$OUT/overlay.sha256"
printf '\nadd_executable(dsv4v-w6-probe ${CMAKE_SOURCE_DIR}/tools/parity/dsv4v_w6_probe.cpp)\ntarget_link_libraries(dsv4v-w6-probe PRIVATE vllm::vllm)\n' >> "$SRC/examples/CMakeLists.txt"

echo "### image"
python3 "$SRC/tools/parity/dsv4v_w6_image.py" "$OUT/img392" | tee "$OUT/image.txt"; step image $?

echo "### oracle: llama.cpp b10766, CPU, static"
git clone -q https://github.com/ggml-org/llama.cpp "$LC" && git -C "$LC" checkout -q "$LC_PIN"; step clone $?
HEAD_SHA=$(git -C "$LC" rev-parse HEAD); echo "oracle HEAD $HEAD_SHA"
[ "$HEAD_SHA" = "$LC_PIN" ] || { echo "FATAL oracle head mismatch"; step pin 96; exit 96; }
git -C "$LC" describe --tags --exact-match 2>/dev/null | tee "$OUT/oracle-tag.txt"
cp "$SRC/tools/parity/dsv4v_w6_oracle_dump.cpp" "$LC/tools/mtmd/"
printf '\nadd_executable(dsv4v-oracle-dump dsv4v_w6_oracle_dump.cpp)\ntarget_link_libraries(dsv4v-oracle-dump PRIVATE mtmd ggml)\n' >> "$LC/tools/mtmd/CMakeLists.txt"
cmake -S "$LC" -B "$LC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  > "$OUT/oracle-configure.log" 2>&1; step oracle_configure $?
cmake --build "$LC/build" --target dsv4v-oracle-dump llama-mtmd-cli -j 4 > "$OUT/oracle-build.log" 2>&1; RC=$?; step oracle_build $RC
tail -3 "$OUT/oracle-build.log"; [ $RC -eq 0 ] || { grep -m12 -E 'error|Error' "$OUT/oracle-build.log"; exit 93; }
ODUMP=$(find "$LC/build" -name dsv4v-oracle-dump -type f | head -1)
OCLI=$(find "$LC/build" -name llama-mtmd-cli -type f | head -1)

echo "### ours: CPU build of the probe"
cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TRITON=OFF -DVLLM_CPP_SERVER=OFF \
  > "$OUT/ours-configure.log" 2>&1; step ours_configure $?
cmake --build "$SRC/build" --target dsv4v-w6-probe -j 4 > "$OUT/ours-build.log" 2>&1; RC=$?; step ours_build $RC
tail -3 "$OUT/ours-build.log"; [ $RC -eq 0 ] || { grep -m12 'error' "$OUT/ours-build.log"; exit 94; }
PROBE=$(find "$SRC/build" -name dsv4v-w6-probe -type f | head -1)

echo "### parity sweep over every lead_pad"
for LP in 0 3 1 2; do
  T=lp$LP
  "$ODUMP" "$MMPROJ" "$OUT/img392.rgb" 392 392 $LP "$OUT" $T $THREADS > "$OUT/oracle-$T.log" 2>&1; step oracle_$T $?
  grep -E 'hparams|preprocess|block|captured|FATAL' "$OUT/oracle-$T.log"
  "$PROBE" "$MMPROJ" "$OUT/img392.rgb" 392 392 $LP "$OUT" $T > "$OUT/ours-$T.log" 2>&1; step ours_$T $?
  grep -E 'config|processor|feature|block|FATAL|what' "$OUT/ours-$T.log"
  python3 "$SRC/tools/parity/dsv4v_w6_compare.py" "$OUT" $T $LP 10 10 > "$OUT/compare-$T.txt" 2>&1; step compare_$T $?
  cat "$OUT/compare-$T.txt"
done

if [ "$RUN_CLI" = 1 ]; then
  echo "### oracle end to end: llama-mtmd-cli with the 82 GB language model"
  rm -f "$OUT/oracle-cli-block.f32"
  MTMD_DEBUG_EMBEDDINGS="$OUT/oracle-cli-block.f32" timeout 5400 "$OCLI" -m "$LANG" --mmproj "$MMPROJ" \
    --image "$OUT/img392.png" -p "Describe this image in one sentence." -n 32 --temp 0 -c 4096 \
    -t $THREADS --no-mmproj-offload > "$OUT/oracle-cli.log" 2>&1; step oracle_cli $?
  grep -E 'deepseek4v|lead|image|MTMD_DEBUG|Shape|error|failed' "$OUT/oracle-cli.log" | head -20
  tail -8 "$OUT/oracle-cli.log"
  if [ -f "$OUT/oracle-cli-block.f32" ]; then
    N=$(python3 -c "import struct;print(struct.unpack('<i',open('$OUT/oracle-cli-block.f32','rb').read(4))[0])")
    LP=$((N - 114)); echo "cli block rows=$N lead_pad=$LP"
    for s in block input vit cells; do cp "$OUT/ours-lp$LP-$s.f32" "$OUT/ours-cli-$s.f32"; done
    python3 "$SRC/tools/parity/dsv4v_w6_compare.py" "$OUT" cli $LP 10 10 > "$OUT/compare-cli.txt" 2>&1; step compare_cli $?
    cat "$OUT/compare-cli.txt"
  fi
fi
echo "### steps"; cat "$OUT/steps.txt"
echo "### W6_PARITY_DONE"
