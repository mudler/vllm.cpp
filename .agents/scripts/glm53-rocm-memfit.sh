#!/usr/bin/env bash
set -u
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
echo "=== MEMFIT $(date -Is) host=$(hostname) ==="
mountpoint -q /workspace && echo "WORKSPACE_MOUNTED=yes" || { echo "WORKSPACE_MOUNTED=NO -- ABORT"; exit 2; }
free -g; echo
B=/tmp/memfit; mkdir -p $B; cp "${SRC:-/workspace/glm53-rocm-probe/memfit.hip}" $B/memfit.hip || exit 2
echo "source sha256: $(sha256sum $B/memfit.hip)"
echo "--- compile for gfx1151 (FAIL HERE, not at the throw) ---"
hipcc --offload-arch=gfx1151 -O2 -o $B/memfit $B/memfit.hip 2>&1 | tail -20
rc=${PIPESTATUS[0]}
if [ ! -x $B/memfit ]; then echo "COMPILE_FAILED rc=$rc -- ABORT"; exit 3; fi
echo "COMPILE_OK; binary sha256: $(sha256sum $B/memfit)"
echo "--- compiled arch manifest: ASSERT, and FAIL HERE if it is empty ---"
MAN=$(llvm-objdump --offloading $B/memfit 2>/dev/null | grep -oE 'amdhsa--gfx[0-9a-z]+' | sed 's/.*--//' | sort -u)
echo "compiled device targets: [${MAN:-}]"
if [ -z "${MAN:-}" ]; then
  echo "EMPTY ARCH MANIFEST -- the binary carries no device code. ABORT before the run,"
  echo "because a host-only binary would allocate happily and measure nothing about the GPU."
  exit 4
fi
case "$MAN" in
  *gfx1151*) echo "ARCH_ASSERT_OK: gfx1151 present" ;;
  *) echo "ARCH MISMATCH: expected gfx1151, got [$MAN] -- ABORT"; exit 5 ;;
esac
echo
echo "--- run ---"
$B/memfit
echo "=== MEMFIT-END rc=$? ==="
free -g
