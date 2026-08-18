#!/usr/bin/env bash
# W0 register-pressure probe job for KERNEL-GDN-REPLAYSSM (#1171), run under an
# `rc run` lease. Compile-only: cuobjdump reads a compiled object and never
# executes the kernel, so this needs the CUDA toolkit and -arch=sm_121a
# cross-compilation, not the GB10 device.
#
# It answers two questions, and the second one is why this script grew a SASS
# pass. (1) What does each of the six kernels in probe.cu cost in registers,
# stack, spill traffic and shared memory, all from ONE ptxas invocation.
# (2) WHERE the spill instructions sit -- `ptxas -v` reports a static byte count
# for the whole kernel, and a static count cannot say whether the local-memory
# traffic is paid on every step or only inside the 1-in-16 flush branch. So the
# object is recompiled with -lineinfo and every LDL/STL is printed against the
# probe.cu source line it came from.
#
# Encodes the recipe proven by /workspace/qwen38-gate/build.sh: the container has
# no CUDA toolkit; the NVIDIA apt repo may be absent, and the ubuntu2404/sbsa lane
# is the one that carries an aarch64 toolkit (the arm64 lane answers 200 and
# carries none); install the targeted packages rather than the 4 GB meta-package.
# No -j is needed here -- it is six kernels in one translation unit.
#
# orin:gpu0 has LOCAL storage only and cannot see the NAS, so probe.cu and this
# script are staged by base64 on the `rc run` command line. $W is overridable so
# the same script runs unchanged on a box that does mount the share.
set -uo pipefail
W="${W:-/workspace/replayssm-w0b}"
OUT="$W/out"
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
    || { echo "ABORT: cuda install failed"; echo "TOOLKIT_RC=20" > "$OUT/RESULT.txt"; exit 20; }
fi
export PATH=/usr/local/cuda-13.0/bin:$PATH
command -v nvcc >/dev/null 2>&1 || { echo "ABORT: no nvcc"; echo "TOOLKIT_RC=21" > "$OUT/RESULT.txt"; exit 21; }
NVCC_V="$(nvcc --version 2>&1 | tail -2 | tr '\n' ' ')"
PTXAS_V="$(ptxas --version 2>&1 | tail -2 | tr '\n' ' ')"
echo "NVCC_VERSION=$NVCC_V"
echo "PTXAS_VERSION=$PTXAS_V"

echo "=== compile (container-local /tmp; /workspace may be a network share) ==="
B=/tmp/replayssm-w0b
rm -rf "$B"; mkdir -p "$B"
cp "$W/probe.cu" "$B/probe.cu" || { echo "ABORT: no probe.cu staged"; exit 22; }
ARCH="${ARCH:-sm_121a}"
CMD="nvcc -std=c++20 -O3 -arch=$ARCH -Xptxas -v -c $B/probe.cu -o $B/probe.o"
echo "COMPILE_CMD=$CMD"
$CMD > "$OUT/ptxas.log" 2>&1
CRC=$?
# A build that produces no object is NO EVIDENCE, not a pass. The rc is printed
# and recorded before anything is read out of the log.
echo "COMPILE_RC=$CRC"
cat "$OUT/ptxas.log"
if [ "$CRC" != 0 ] || [ ! -s "$B/probe.o" ]; then
  echo "ABORT: no object produced -- this is NO EVIDENCE"
  { echo "COMPILE_RC=$CRC"; echo "OBJECT=absent"; echo "NVCC_VERSION=$NVCC_V"; } > "$OUT/RESULT.txt"
  exit 23
fi

echo "=== cuobjdump -res-usage ==="
cuobjdump -res-usage "$B/probe.o" > "$OUT/cuobjdump.log" 2>&1
ORC=$?
echo "CUOBJDUMP_RC=$ORC"
cat "$OUT/cuobjdump.log"

echo "=== arch verification (the -arch flag must have taken effect) ==="
# A silent fallback to another target would make every number above answer a
# question nobody asked. Assert the requested arch appears in the ELF section.
ARCH_HITS=$(grep -c "arch = $ARCH" "$OUT/cuobjdump.log")
echo "ARCH_REQUESTED=$ARCH"
echo "ARCH_HITS_IN_RES_USAGE=$ARCH_HITS"
cuobjdump -elf "$B/probe.o" 2>/dev/null | grep -m4 -E '^arch|sm_[0-9]+' | sed 's/^/ELF: /'
if [ "$ARCH_HITS" -lt 1 ]; then
  echo "ARCH_VERIFIED=NO -- the requested arch is absent from the object"
else
  echo "ARCH_VERIFIED=yes"
fi

echo "=== sass: function inventory (all six kernels must be present) ==="
cuobjdump -sass "$B/probe.o" > "$OUT/sass.txt" 2>&1
grep -c 'Function : ' "$OUT/sass.txt" | sed 's/^/sass_functions=/'
grep 'Function : ' "$OUT/sass.txt"

echo "=== compile again with -lineinfo, for SASS attribution ==="
# -lineinfo adds a debug section; it must not move the numbers. The REG/STACK
# lines of the two objects are diffed below, and a difference invalidates the
# attribution rather than being explained away.
LCMD="nvcc -std=c++20 -O3 -arch=$ARCH -lineinfo -Xptxas -v -c $B/probe.cu -o $B/probe_li.o"
echo "LINEINFO_COMPILE_CMD=$LCMD"
$LCMD > "$OUT/ptxas_lineinfo.log" 2>&1
LRC=$?
echo "LINEINFO_COMPILE_RC=$LRC"
if [ "$LRC" != 0 ] || [ ! -s "$B/probe_li.o" ]; then
  echo "LINEINFO_OBJECT=absent -- SASS attribution NOT AVAILABLE"
else
  cuobjdump -res-usage "$B/probe_li.o" > "$OUT/cuobjdump_lineinfo.log" 2>&1
  echo "--- res-usage diff, plain object vs -lineinfo object (REG/STACK lines only) ---"
  # The mangled name carries an anonymous-namespace hash that nvcc re-rolls per
  # translation unit; it is not codegen, so the comparison keys on the short
  # kernel name and the resource line only.
  RESKEY='/^ Function /{ if (match($0, /Probe[A-Za-z]*Kernel/)) k=substr($0, RSTART, RLENGTH); next }
          /^  REG:/{ print k, $0 }'
  awk "$RESKEY" "$OUT/cuobjdump.log"          > "$OUT/res_plain.txt"
  awk "$RESKEY" "$OUT/cuobjdump_lineinfo.log" > "$OUT/res_li.txt"
  if diff -u "$OUT/res_plain.txt" "$OUT/res_li.txt" > "$OUT/resusage.diff" 2>&1; then
    echo "RESUSAGE_IDENTICAL=yes  (-lineinfo did not change codegen)"
  else
    echo "RESUSAGE_IDENTICAL=NO   (attribution below is against a DIFFERENT codegen)"
    cat "$OUT/resusage.diff"
  fi

  # cuobjdump -sass carries no line markers even on a -lineinfo object, so the
  # cubin is extracted and disassembled with `nvdisasm -g`, which does. Reading
  # LDL/STL placement off a listing with no line info is how a static byte count
  # gets mistaken for a per-step cost.
  echo "=== nvdisasm -g (source-attributed SASS) ==="
  rm -rf "$B/xelf"; mkdir -p "$B/xelf"
  ( cd "$B/xelf" && cuobjdump -xelf all "$B/probe_li.o" > "$OUT/xelf.log" 2>&1 )
  echo "XELF_RC=$?"
  ls -1 "$B/xelf"/*.cubin 2>/dev/null | sed 's/^/cubin: /'
  CUBIN="$(ls -1 "$B/xelf"/*.cubin 2>/dev/null | head -1)"
  if [ -z "$CUBIN" ]; then
    echo "NVDISASM=skipped -- no cubin extracted"
  else
    nvdisasm -g -c "$CUBIN" > "$OUT/sass_lineinfo.txt" 2>&1
    NRC=$?
    if [ "$NRC" != 0 ]; then
      # -c is not accepted by every nvdisasm; -g alone carries the line info.
      nvdisasm -g "$CUBIN" > "$OUT/sass_lineinfo.txt" 2>&1
      NRC=$?
      echo "NVDISASM_FALLBACK=used (-g alone)"
    fi
    echo "NVDISASM_RC=$NRC"
    echo "LINE_MARKERS=$(grep -c '## File' "$OUT/sass_lineinfo.txt")"
    if [ "$NRC" != 0 ] || [ "$(grep -c '## File' "$OUT/sass_lineinfo.txt")" -lt 1 ]; then
      echo "ATTRIBUTION=UNAVAILABLE -- nvdisasm produced no line markers"
    else
      echo "--- route line map in probe.cu (so an attributed line can be classified) ---"
      grep -nE 'ProbeReplaySsm[A-Za-z]*Kernel\(PROBE|^__global__|if \(is_flush\)|\} else \{|// Flush route|// Non-flush route|// Streamed checkpoint|// Reconstruct, persist' "$B/probe.cu"

      echo "--- every LDL/STL with the probe.cu line it came from ---"
      awk '
        NR==FNR { src[FNR]=$0; next }
        /\.text\.|Function : / { if (match($0, /Probe[A-Za-z]*Kernel/))
                                   short=substr($0, RSTART, RLENGTH);
                                 line=0 }
        /## File/                { m=split($0, a, ","); t=a[m]; gsub(/[^0-9]/, "", t);
                                   if (t != "") line=t+0; next }
        /[ \t](LDL|STL)[.\t ]/  { txt=$0; sub(/^[ \t]*/, "", txt); sub(/\/\* 0x.*/, "", txt);
                                   body=src[line]; sub(/^[ \t]*/, "", body);
                                   printf "%-30s probe.cu:%-5d | %-42s | %s\n", short, line, txt, body }
      ' "$B/probe.cu" "$OUT/sass_lineinfo.txt"

      # The flush route of the first-run kernel begins at its `if (is_flush) {`.
      # Everything before that line executes on EVERY step; everything at or
      # after it executes on 1 step in L. This is the line the "pays
      # local-memory traffic on every step" claim actually turns on.
      FLUSH_LINE=$(grep -n 'if (is_flush) {' "$B/probe.cu" | head -1 | cut -d: -f1)
      echo "FIRST_RUN_KERNEL_FLUSH_BRANCH_STARTS_AT=probe.cu:$FLUSH_LINE"

      # The split line is the first-run kernel's. The control has no flush route
      # at all, so every instruction of it is correctly counted as every-step;
      # the four route-isolated kernels are each one route by construction.
      echo "--- LDL/STL count per kernel, split at that line ---"
      awk '
        /\.text\.|Function : / { if (match($0, /Probe[A-Za-z]*Kernel/))
                                   short=substr($0, RSTART, RLENGTH);
                                 if (!(short in seen)) { seen[short]=1; order[++n]=short }
                                 line=0 }
        /## File/                { m=split($0, a, ","); t=a[m]; gsub(/[^0-9]/, "", t);
                                   if (t != "") line=t+0; next }
        /[ \t](LDL|STL)[.\t ]/  { op = ($0 ~ /[ \t]LDL[.\t ]/) ? "ldl" : "stl";
                                   where = (line >= FL+0 && line > 0) ? "flush" : "everystep";
                                   c[short "|" op "|" where]++;
                                   if (line == 0) unattributed[short]++ }
        END { for (i=1; i<=n; i++) { k=order[i];
                printf "%-30s STL everystep=%-4d STL flush=%-4d LDL everystep=%-4d LDL flush=%-4d unattributed=%-4d\n",
                       k, c[k "|stl|everystep"]+0, c[k "|stl|flush"]+0,
                       c[k "|ldl|everystep"]+0, c[k "|ldl|flush"]+0, unattributed[k]+0 } }
      ' FL="$FLUSH_LINE" "$OUT/sass_lineinfo.txt"
    fi
  fi
fi

{
  echo "ARCH=$ARCH"
  echo "ARCH_HITS_IN_RES_USAGE=$ARCH_HITS"
  echo "COMPILE_RC=$CRC"
  echo "CUOBJDUMP_RC=$ORC"
  echo "LINEINFO_COMPILE_RC=$LRC"
  echo "OBJECT_BYTES=$(stat -c %s "$B/probe.o")"
  echo "NVCC_VERSION=$NVCC_V"
  echo "PTXAS_VERSION=$PTXAS_V"
  echo "COMPILE_CMD=$CMD"
  echo "PROBE_SHA256=$(sha256sum "$B/probe.cu" | awk '{print $1}')"
  echo "SCRIPT_SHA256=$(sha256sum "$0" | awk '{print $1}')"
  echo "--- ptxas -v ---"
  grep -E 'ptxas info|registers|stack frame|spill' "$OUT/ptxas.log"
  echo "--- cuobjdump -res-usage ---"
  cat "$OUT/cuobjdump.log"
} > "$OUT/RESULT.txt" 2>&1
echo "=== RESULT.txt ==="
cat "$OUT/RESULT.txt"
echo "=== W0 JOB DONE ==="
