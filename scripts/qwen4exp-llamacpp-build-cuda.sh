#!/bin/bash
# Build llama.cpp at the `llama-cpp-qwen4exp` pin (PR #27742, 035e2273) with CUDA
# for GB10 (sm_121a), inside an `rc` lease on dgx:gpu0.
#
# THIS FILE IS IN THE TREE, and it was not on 29 August 2026.  The CUDA build
# recorded in docs/bench-evidence/qwen4exp-llamacpp-ladder-arm-20260829.md ran
# from a copy on the NAS, so the build that flipped an oracle to
# `gateable = yes` was not reproducible from the repository.  A recipe that
# lives only beside its own output is a recipe nobody else can run.
#
# Contract: .agents/oracles/llama-cpp-qwen4exp.md ("Assert the tree, not only the
# commit") and .agents/environment.md §"the whole recipe".
#   - fresh clone into an EMPTY dir, never a checkout somebody develops in
#   - `git status --porcelain` asserted empty, recorded
#   - the CUDA toolkit is NOT in the worker image: install it unconditionally,
#     and then ASSERT the version, because `apt-get install cuda-toolkit-13-0`
#     is not a pin: a rerun gets whatever apt serves that day and, before this
#     assertion existed, nothing noticed.  EXPECT_NVCC carries the version the
#     recorded evidence was measured with; a mismatch refuses rather than
#     silently producing a build the evidence does not describe.
#   - build in /tmp (CIFS /workspace holds no symlink), `cp -rL` the result out
#   - -j 4: unconstrained parallelism has OOM-rebooted this box
#   - print a heartbeat: --idle-timeout counts the job's OWN stdout
set -u

PIN=035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
# The toolchain the committed evidence was measured with.  Override deliberately
# and then re-measure the evidence; do not widen it to make a rerun pass.
EXPECT_NVCC=${EXPECT_NVCC:-13.0.88}
SRC=/tmp/llamacpp-q4exp-src
OUT=/workspace/q4exp-bench/llamacpp-pr27742-cuda
NEED_GB=${NEED_GB:-30}

free_gb() { df -BG --output=avail /tmp | tail -1 | tr -dc '0-9'; }
cleanup() { rm -rf "$SRC"; kill "${HB:-}" 2>/dev/null; wait "${HB:-}" 2>/dev/null; }
trap cleanup EXIT INT TERM

( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S) disk=$(free_gb)G"; done ) &
HB=$!

echo "=== IDENTITY ==="
hostname; date -u +%Y-%m-%dT%H:%M:%SZ; whoami
nvidia-smi -L 2>&1 | head -3
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv 2>&1

echo "=== WHAT THIS LEASE CAN SEE (probe, not an assumption) ==="
echo "-- ls / --";           ls / | tr '\n' ' '; echo
echo "-- ls /workspace --";  ls /workspace 2>&1 | head -20
echo "-- /workspace/.. --";  ls /workspace/.. 2>&1 | head -20
echo "-- /mnt --";           ls /mnt 2>&1 | head -20
echo "-- nas_share? --";     ls -d /mnt/nas_share /usr/local/nas_share 2>&1
echo "-- findmnt /workspace --"; findmnt /workspace 2>&1 | head -5
echo "-- df -h /workspace /tmp --"; df -h /workspace /tmp 2>&1
echo "-- the staged quant, as reached from this lease --"
ls -la /workspace/q4exp-bench/UD-IQ1_S 2>&1 | head -10

echo "=== DISK ==="
df -h /tmp
du -sh /tmp/* 2>/dev/null | sort -rh | head -10
rm -rf "$SRC"
if [ "$(free_gb)" -lt "$NEED_GB" ]; then
  echo "REFUSING: /tmp has $(free_gb) GiB free, below NEED_GB=${NEED_GB}."
  echo "A CUDA build that runs out of space fails as unrelated compile errors."
  exit 95
fi

echo "=== CUDA TOOLKIT (not in the worker image; install unconditionally) ==="
apt-get update -qq || { echo "FATAL: apt-get update"; exit 91; }
apt-get install -y -qq wget ca-certificates gnupg git cmake ninja-build || { echo "FATAL: base pkgs"; exit 91; }
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb || { echo "FATAL: keyring fetch"; exit 91; }
dpkg -i /tmp/ck.deb
apt-get update -qq
apt-get install -y -qq cuda-toolkit-13-0 || { echo "FATAL: cuda-toolkit-13-0"; exit 91; }
apt-get install -y -qq cuda-cuobjdump-13-0
export PATH=/usr/local/cuda/bin:$PATH
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

command -v nvcc >/dev/null || { echo "FATAL: no nvcc after install"; exit 90; }
test -f "$CUDA_HOME/include/cuda_runtime.h" || { echo "FATAL: nvcc but no cuda_runtime.h"; exit 90; }
{ ls "$CUDA_HOME"/targets/*/lib/libcudart.so* >/dev/null 2>&1 || ls "$CUDA_HOME"/lib64/libcudart.so* >/dev/null 2>&1; } \
  || { echo "FATAL: no libcudart under $CUDA_HOME"; exit 90; }
nvcc --version | tail -2
# The toolchain assertion.  `cuda-toolkit-13-0` is a channel, not a version.
NVCC_V=$(nvcc --version | grep -Eo 'V[0-9]+\.[0-9]+\.[0-9]+' | head -1 | tr -d 'V')
echo "nvcc version    = ${NVCC_V:-UNREADABLE} (expected $EXPECT_NVCC)"
if [ "$NVCC_V" != "$EXPECT_NVCC" ]; then
  echo "REFUSING: apt served nvcc '$NVCC_V', not the '$EXPECT_NVCC' the committed"
  echo "evidence was measured with.  'cuda-toolkit-13-0' pins a channel, not a"
  echo "version.  Either install $EXPECT_NVCC, or re-measure the evidence file at"
  echo "the new toolchain and move EXPECT_NVCC in the same change."
  exit 89
fi
if command -v cuobjdump >/dev/null; then CUOBJ=1; else CUOBJ=0; echo "### cuobjdump ABSENT -- cubin proof stays OWED"; fi

echo "=== FRESH CLONE AT THE PIN (never a developed-in tree) ==="
mkdir -p "$SRC" && cd "$SRC" || { echo "FATAL: mkdir $SRC"; exit 92; }
git init -q .
git remote add origin https://github.com/ggml-org/llama.cpp
git fetch -q --depth 1 origin "$PIN" || { echo "FATAL: fetch $PIN (container egress to github.com?)"; exit 93; }
git checkout -q FETCH_HEAD || { echo "FATAL: checkout"; exit 93; }
HEAD_SHA=$(git rev-parse HEAD)
echo "HEAD            = $HEAD_SHA"
echo "PIN             = $PIN"
[ "$HEAD_SHA" = "$PIN" ] || { echo "FATAL: HEAD is not the pin"; exit 94; }
PORC=$(git status --porcelain | wc -c)
echo "git status --porcelain bytes = $PORC"
[ "$PORC" -eq 0 ] || { echo "FATAL: tree is dirty at the pin"; exit 94; }
echo "-- the pin defines qwen4exp, which no released llama.cpp does --"
git ls-files | grep -i qwen4exp
git rev-parse --short HEAD

echo "=== CONFIGURE ==="
# We ask for `121`; llama.cpp's own CMake rewrites it, printing
# `Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a`, and the cubins come out
# `sm_121a`.  Measured, not assumed: see the histogram below and the evidence
# file's "It is a CUDA build, proven three ways".
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  2>&1 | tee /tmp/configure.log | tail -25
CFG_RC=${PIPESTATUS[0]}
echo "configure rc=$CFG_RC"
[ "$CFG_RC" -eq 0 ] || { echo "FATAL: configure"; exit 96; }

echo "=== BUILD (named targets only, -j 4) ==="
cmake --build "$SRC/build" -j 4 --target llama-server llama-cli llama-bench 2>&1 | tail -30
BLD_RC=${PIPESTATUS[0]}
echo "build rc=$BLD_RC"
[ "$BLD_RC" -eq 0 ] || { echo "FATAL: build"; exit 97; }

echo "=== PROVE IT IS A CUDA BUILD, AND THAT IT KNOWS qwen4exp ==="
find "$SRC/build" -name 'llama-server' -o -name 'llama-cli' -o -name 'llama-bench' | sort
SRV=$(find "$SRC/build" -name llama-server -type f | head -1)
echo "-- ldd --"; ldd "$SRV" | grep -Ei 'cuda|cublas|ggml|llama'
echo "-- qwen4exp strings in the shipped objects --"
for f in $(find "$SRC/build" -name 'llama-server' -o -name 'libllama.so*' -o -name 'libggml*.so*' | sort); do
  echo "$(strings -a "$f" 2>/dev/null | grep -c qwen4exp)  $f"
done
echo "-- .cu.o count (the DENOMINATOR) --"
find "$SRC/build" -name '*.cu.o' | wc -l
if [ "$CUOBJ" -eq 1 ]; then
  for o in $(find "$SRC/build" -name '*.cu.o' | head -400); do
    echo "== $o"; cuobjdump --list-elf "$o"
  done > /tmp/cubin.log 2>&1
  echo "-- cubin arch histogram --"
  # `a` INCLUDED.  `grep -o 'sm_[0-9]*'` cannot print the architecture-specific
  # suffix, so on 29 August it turned 142 `sm_121a` cubins into a histogram
  # reading `142 sm_121` -- and that transcription then went into the oracle
  # record.  The error was in this project's own disfavour: the histogram is the
  # second, independent witness that the `121` asked for in CMAKE_CUDA_ARCHITECTURES
  # became the `121a` the GB10 wants, and the truncated regex made it say nothing.
  grep -o 'sm_[0-9]*a\?' /tmp/cubin.log | sort | uniq -c
  echo "objects scanned: $(grep -c '^== ' /tmp/cubin.log)"
fi

echo "=== SMOKE: the binary runs and reports CUDA ==="
"$SRV" --version 2>&1 | head -20
LC=$(find "$SRC/build" -name llama-cli -type f | head -1)
"$LC" --version 2>&1 | head -20

echo "=== COPY OUT (cp -rL: CIFS holds no symlink) ==="
mkdir -p "$OUT"
rm -rf "$OUT"/bin "$OUT"/lib
mkdir -p "$OUT/bin"
for b in llama-server llama-cli llama-bench; do
  p=$(find "$SRC/build" -name "$b" -type f | head -1)
  [ -n "$p" ] && cp -L "$p" "$OUT/bin/"
done
find "$SRC/build" \( -name 'lib*.so' -o -name 'lib*.so.*' \) -type f -exec cp -L {} "$OUT/bin/" \;
cp -L /tmp/configure.log "$OUT/" 2>/dev/null
[ "$CUOBJ" -eq 1 ] && cp -L /tmp/cubin.log "$OUT/" 2>/dev/null
{
  echo "pin            $PIN"
  echo "head           $HEAD_SHA"
  echo "porcelain      $PORC bytes"
  echo "built_on       $(hostname) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nvcc           $(nvcc --version | tail -2 | tr '\n' ' ')"
  echo "cuda_arch      121 (llama.cpp rewrites it to 121a; cubins are sm_121a)"
  echo "expect_nvcc    $EXPECT_NVCC (asserted, not transcribed)"
  echo "cmake_flags    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF"
  echo "targets        llama-server llama-cli llama-bench (-j 4)"
  echo "--- sha256 ---"
  ( cd "$OUT/bin" && sha256sum * )
} > "$OUT/BUILD-RECORD.txt"
cat "$OUT/BUILD-RECORD.txt"
ls -la "$OUT/bin"
du -sh "$SRC" "$SRC/build"
echo "### /tmp free at end: $(free_gb) GiB"
echo "=== BUILD JOB DONE rc=0 ==="
