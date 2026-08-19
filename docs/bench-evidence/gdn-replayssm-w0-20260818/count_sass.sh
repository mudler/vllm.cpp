#!/usr/bin/env bash
# W0b confirmation job for KERNEL-GDN-REPLAYSSM (#1171): per-kernel SASS
# instruction counts.
#
# WHY THIS EXISTS, SEPARATELY FROM build_w0.sh. The W0 review closed the
# question "is REG:42 a stub the compiler folded away?" by inference -- three
# independent arguments from the source and from the other kernels' numbers.
# That reasoning stands, but a count answers it outright: a kernel that ptxas
# had emptied would carry a handful of instructions, not hundreds. This script
# does nothing but produce that count.
#
# It is a SEPARATE script rather than two more lines inside build_w0.sh because
# build_w0.sh's sha256 is the recorded provenance of job1.log and job2.log
# (`## Outcome`). Editing it would retroactively invalidate that record for
# runs it did not produce. Nothing here re-measures a W0 number; every register,
# stack and spill figure still comes from job1/job2 and is untouched.
#
# Compile-only, same recipe as build_w0.sh: same toolkit packages, same
# `-arch=sm_121a`, same probe.cu. cuobjdump reads a compiled object and never
# executes a kernel.
set -uo pipefail
W="${W:-/workspace/replayssm-w0b}"
OUT="$W/out-sass"
mkdir -p "$OUT"
exec > >(tee "$OUT/job.log") 2>&1

echo "=== identity ==="
id -u; uname -m; hostname
sha256sum "$W/probe.cu" "$0"

echo "=== toolkit ==="
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  PKGS="cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-crt-13-0 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0"
  apt-get update -qq 2>/dev/null
  if ! apt-cache show cuda-nvcc-13-0 >/dev/null 2>&1; then
    REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa
    curl -fsSL -o /tmp/cuda-keyring.deb "$REPO/cuda-keyring_1.1-1_all.deb" \
      && dpkg -i /tmp/cuda-keyring.deb >/dev/null 2>&1
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] $REPO/ /" \
      > /etc/apt/sources.list.d/cuda-sbsa.list
    apt-get update -qq 2>/dev/null
  fi
  apt-get install -y -qq --no-install-recommends $PKGS >/dev/null 2>&1 \
    || { echo "ABORT: cuda install failed"; exit 20; }
fi
export PATH=/usr/local/cuda-13.0/bin:$PATH
command -v nvcc >/dev/null 2>&1 || { echo "ABORT: no nvcc"; exit 21; }
nvcc --version 2>&1 | tail -2 | sed 's/^/NVCC: /'
ptxas --version 2>&1 | tail -2 | sed 's/^/PTXAS: /'

echo "=== compile ==="
B=/tmp/replayssm-w0b-sass
rm -rf "$B"; mkdir -p "$B"
cp "$W/probe.cu" "$B/probe.cu" || { echo "ABORT: no probe.cu staged"; exit 22; }
ARCH="${ARCH:-sm_121a}"
CMD="nvcc -std=c++20 -O3 -arch=$ARCH -Xptxas -v -c $B/probe.cu -o $B/probe.o"
echo "COMPILE_CMD=$CMD"
$CMD > "$OUT/ptxas.log" 2>&1
CRC=$?
echo "COMPILE_RC=$CRC"
cat "$OUT/ptxas.log"
if [ "$CRC" != 0 ] || [ ! -s "$B/probe.o" ]; then
  echo "ABORT: no object produced -- this is NO EVIDENCE"; exit 23
fi
echo "OBJECT_BYTES=$(stat -c %s "$B/probe.o")"

echo "=== arch verification ==="
cuobjdump -res-usage "$B/probe.o" > "$OUT/res.log" 2>&1
echo "ARCH_HITS_IN_RES_USAGE=$(grep -c "arch = $ARCH" "$OUT/res.log")"

echo "=== sass ==="
cuobjdump -sass "$B/probe.o" > "$OUT/sass.txt" 2>&1
echo "SASS_RC=$?"
echo "SASS_BYTES=$(stat -c %s "$OUT/sass.txt")"
echo "SASS_SHA256=$(sha256sum "$OUT/sass.txt" | awk '{print $1}')"
echo "sass_functions=$(grep -c 'Function : ' "$OUT/sass.txt")"

echo "=== per-function SASS instruction count ==="
# Every SASS instruction line carries its own /*ffff*/ hex address comment, so
# counting those lines per `Function :` header counts instructions, not the
# scheduling-control /* 0x... */ continuation lines that follow each one.
awk '/Function : /{f=$0} /\/\*[0-9a-f]{4}\*\//{c[f]++} END{for(k in c) print c[k], k}' \
  "$OUT/sass.txt" | sort -rn

echo "=== sass.txt, gzip+base64, for retrieval off a box with no share ==="
echo "SASS_GZ_B64_BEGIN"
gzip -9 -c "$OUT/sass.txt" | base64 -w0
echo
echo "SASS_GZ_B64_END"
echo "=== JOB DONE ==="
