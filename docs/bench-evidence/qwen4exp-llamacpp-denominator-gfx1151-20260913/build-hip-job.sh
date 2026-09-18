#!/bin/bash
# Build llama.cpp at the `llama-cpp-qwen4exp` pin (PR #27742, 035e2273) with HIP
# for gfx1151, inside an `rc` lease on strix:gpu0.
# HIP equivalent of scripts/qwen4exp-llamacpp-build-cuda.sh.
set -u
PIN=035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
SRC=/tmp/q4exp-hip-src-D9K2
BLD=$SRC/build
OUTDIR=/workspace/q4exp-gfx1151-denominator
NEED_GB=${NEED_GB:-40}
free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
( while sleep 60; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G load=$(cut -d' ' -f1-3 /proc/loadavg)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "0. IDENTITY"
hostname; date -u; uname -m; nproc; free -g | head -2
echo "-- df --"; df -h /tmp /workspace
echo "-- /tmp top --"; du -sh /tmp/* 2>/dev/null | sort -rh | head -8
if [ "$(free_gb)" -lt "$NEED_GB" ]; then echo "REFUSING: /tmp has $(free_gb)G free < NEED_GB=$NEED_GB (ENOSPC risk)"; exit 95; fi

say "1. ROCm TOOLCHAIN (assert, do not assume)"
ls -d /opt/rocm*; hipcc --version 2>&1 | head -4
rocminfo 2>&1 | grep -m2 -E 'Name:.*gfx'
echo "-- rocblas/hipblas headers --"
ls /opt/rocm/include/rocblas/rocblas.h /opt/rocm/include/hipblas/hipblas.h 2>&1
if [ ! -e /opt/rocm/include/rocblas/rocblas.h ] || [ ! -e /opt/rocm/include/hipblas/hipblas.h ]; then
  echo "installing rocblas-dev hipblas-dev"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends rocblas-dev hipblas-dev
  echo "apt_rc=$?"
fi
ls /opt/rocm/include/rocblas/rocblas.h /opt/rocm/include/hipblas/hipblas.h 2>&1
apt-get install -y -qq git cmake ninja-build ccache curl jq binutils >/dev/null 2>&1
export CCACHE_DIR=/tmp/q4exp-hip-ccache-D9K2; mkdir -p "$CCACHE_DIR"; export CCACHE_MAXSIZE=20G
cmake --version | head -1; ninja --version

say "2. FRESH FETCH BY SHA INTO AN EMPTY DIRECTORY"
rm -rf "$SRC"; mkdir -p "$SRC"; cd "$SRC" || exit 92
git init -q .
git remote add origin https://github.com/ggml-org/llama.cpp
git fetch -q --depth 1 origin "$PIN" || { echo "FATAL: fetch $PIN"; exit 93; }
git checkout -q FETCH_HEAD || { echo "FATAL: checkout"; exit 93; }
HEAD_SHA=$(git rev-parse HEAD)
echo "HEAD = $HEAD_SHA"
echo "PIN  = $PIN"
[ "$HEAD_SHA" = "$PIN" ] || { echo "FATAL: HEAD is not the pin"; exit 94; }
PORC=$(git status --porcelain | wc -c)
echo "git status --porcelain bytes = $PORC"
[ "$PORC" -eq 0 ] || { echo "FATAL: dirty tree at the pin"; exit 94; }
echo "-- the pin defines qwen4exp, which no released llama.cpp does --"
git ls-files | grep -i qwen4exp
echo "-- source content manifest (locale-independent) --"
MAN=$(cd "$SRC" && find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
echo "llama_src_manifest_sha256=$MAN"
echo "llama_src_files=$(find "$SRC" -path "$SRC/.git" -prune -o -type f -print | wc -l)"

say "3. CONFIGURE (HIP, gfx1151)"
cmake -S "$SRC" -B "$BLD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON \
  2>&1 | tee /tmp/q4exp-hip-configure.log | tail -30
CFG_RC=${PIPESTATUS[0]}; echo "configure rc=$CFG_RC"
[ "$CFG_RC" -eq 0 ] || { echo "FATAL: configure failed at the pin with HIP"; grep -iE 'error|fatal' /tmp/q4exp-hip-configure.log | head -30; exit 96; }
grep -iE 'hip|gfx1151|rocm' /tmp/q4exp-hip-configure.log | head -20

say "4. BUILD -j 4 (unconstrained parallelism has OOM-rebooted these boxes)"
cmake --build "$BLD" -j 4 --target llama-bench llama-cli llama-server 2>&1 | tee /tmp/q4exp-hip-build.log | tail -40
BLD_RC=${PIPESTATUS[0]}; echo "build rc=$BLD_RC"
if [ "$BLD_RC" -ne 0 ]; then
  echo "FATAL: HIP BUILD FAILED AT THE PIN -- first errors:"
  grep -iE 'error:|Error [0-9]|FAILED:' /tmp/q4exp-hip-build.log | head -40
  cp -f /tmp/q4exp-hip-build.log "$OUTDIR/build-hip-FAILED.log" 2>/dev/null
  exit 97
fi
echo "-- tree dirtied by the build? --"
MAN2=$(cd "$SRC" && find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
echo "llama_src_manifest_after_build=$MAN2 (expect $MAN)"

say "5. ASSERT THE COMPILED FEATURE SET BEFORE ANYTHING IS TIMED"
BIN=$(dirname "$(find "$BLD" -name llama-bench -type f | head -1)")
echo "BIN=$BIN"; ls -la "$BIN" | head -20
export LD_LIBRARY_PATH="$BIN:$BLD/bin:$BLD/lib:/opt/rocm/lib"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
for b in llama-bench llama-cli llama-server; do
  echo "-- ldd $b --"; ldd "$BIN/$b" 2>&1 | grep -Ei 'hip|rocblas|hipblas|hsa|ggml|llama|not found'
done
echo "-- qwen4exp strings in the shipped objects --"
for f in $(find "$BLD" \( -name 'llama-server' -o -name 'libllama.so*' -o -name 'libggml*.so*' \) -type f | LC_ALL=C sort); do
  echo "$(strings -a "$f" 2>/dev/null | grep -c qwen4exp)  $f"
done
echo "-- binary sha256 --"
sha256sum "$BIN/llama-bench" "$BIN/llama-cli" "$BIN/llama-server" 2>&1
find "$BLD" \( -name 'libllama.so*' -o -name 'libggml*.so*' \) -type f | LC_ALL=C sort | xargs -r sha256sum
echo "-- hip device code objects present? --"
find "$BLD" -name '*.hip.o' | wc -l
HIPO=$(find "$BLD" -name '*.hip.o' | head -1); echo "sample=$HIPO"
[ -n "$HIPO" ] && (roc-obj-ls "$HIPO" 2>/dev/null | head -5; llvm-objdump --offloading "$HIPO" 2>/dev/null | head -8; strings -a "$HIPO" | grep -o 'gfx11[0-9]*' | sort | uniq -c | head)
echo "-- gfx1151 code objects in the built ggml-hip library --"
GH=$(find "$BLD" -name 'libggml-hip.so*' -type f | head -1); echo "libggml-hip=$GH"
[ -n "$GH" ] && strings -a "$GH" | grep -o 'gfx11[0-9]*' | sort | uniq -c | head

say "6. THE ORACLE STARTS AND SEES THE GPU"
"$BIN/llama-bench" -h > /tmp/bench-help.txt 2>&1 </dev/null; echo "llama-bench -h rc=$?"; head -3 /tmp/bench-help.txt
"$BIN/llama-bench" --list-devices > /tmp/list-devices.txt 2>&1 </dev/null; echo "--list-devices rc=$?"; cat /tmp/list-devices.txt
grep -qiE 'rocm|hip|gfx' /tmp/list-devices.txt || echo "WARNING: no ROCm device enumerated -- this build would decode on the CPU"

say "7. ARCHIVE THE BUILD (rc logs age out; /workspace strips the exec bit and find -type f drops SONAME symlinks, so TAR)"
mkdir -p "$OUTDIR"
cd "$BLD" || exit 98
tar czf /tmp/q4exp-hip-bin.tar.gz bin 2>/dev/null || tar czf /tmp/q4exp-hip-bin.tar.gz . 
ls -la /tmp/q4exp-hip-bin.tar.gz; sha256sum /tmp/q4exp-hip-bin.tar.gz
cp -f /tmp/q4exp-hip-bin.tar.gz "$OUTDIR/llamacpp-pr27742-hip-gfx1151-bin.tar.gz" && echo "tar copied to $OUTDIR"
sha256sum "$OUTDIR/llamacpp-pr27742-hip-gfx1151-bin.tar.gz"
for f in /tmp/q4exp-hip-configure.log /tmp/q4exp-hip-build.log /tmp/list-devices.txt /tmp/bench-help.txt; do cp -f "$f" "$OUTDIR/" 2>/dev/null; done
echo "$PIN" > "$OUTDIR/LLAMA_PIN"; echo "$MAN" > "$OUTDIR/LLAMA_SRC_MANIFEST"
echo "SRC_TREE_KEPT=$SRC  BUILD_KEPT=$BLD"
ls -la "$OUTDIR"
say "BUILD DONE rc=0"
