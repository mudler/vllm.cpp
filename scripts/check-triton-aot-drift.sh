#!/usr/bin/env bash
# check-triton-aot-drift.sh — verify the vendored Triton AOT artifacts are in
# sync with their generating sources, WITHOUT Python/Triton/GPU (pure sha256
# text compare, safe on any host incl. GitHub-hosted CI).
#
# For every src/vt/cuda/triton_aot_vendored/<arch>/MANIFEST, compare each
#   source <kernel>.py sha256=<hash>
# line against sha256sum of the current triton_kernels/<kernel>.py.
# Exit 0 = in sync; exit 1 = drift (stale vendored artifacts) with a report.
#
# This is the CI face of the staleness guard that cmake/TritonAOT.cmake also
# enforces at configure time on CUDA builds. Regenerating is a maintainer/GPU
# task: scripts/regen-triton-aot.sh (see that header).
set -euo pipefail
cd "$(dirname "$0")/.."

KERNELS_DIR=triton_kernels
VENDORED_DIR=src/vt/cuda/triton_aot_vendored
status=0
found_manifest=0

for manifest in "${VENDORED_DIR}"/*/MANIFEST; do
  [ -f "${manifest}" ] || continue
  found_manifest=1
  arch_dir=$(dirname "${manifest}")
  echo "== ${manifest}"
  while read -r _ src expected; do
    expected="${expected#sha256=}"
    if [ ! -f "${KERNELS_DIR}/${src}" ]; then
      echo "DRIFT  ${arch_dir}: source ${src} listed in MANIFEST but missing from ${KERNELS_DIR}/"
      status=1
      continue
    fi
    actual=$(sha256sum "${KERNELS_DIR}/${src}" | awk '{print $1}')
    if [ "${actual}" != "${expected}" ]; then
      echo "DRIFT  ${arch_dir}: ${src} changed since regen (manifest ${expected:0:12}… vs current ${actual:0:12}…)"
      status=1
    else
      echo "ok     ${src}"
    fi
  done < <(grep '^source ' "${manifest}")
  # Kernels added without any regen at all:
  for py in "${KERNELS_DIR}"/*.py; do
    base=$(basename "${py}")
    if ! grep -q "^source ${base} " "${manifest}"; then
      echo "DRIFT  ${arch_dir}: ${base} exists in ${KERNELS_DIR}/ but is absent from MANIFEST (never vendored for this arch)"
      status=1
    fi
  done
done

if [ "${found_manifest}" = 0 ]; then
  echo "error: no ${VENDORED_DIR}/*/MANIFEST found" >&2
  exit 2
fi

if [ "${status}" != 0 ]; then
  cat >&2 <<'EOF'

Vendored Triton AOT artifacts are STALE relative to triton_kernels/*.py.
Regenerate (maintainer task, needs Python+Triton+GPU, e.g. dgx.casa):
    scripts/regen-triton-aot.sh
then review + commit the vendored-tree diff. See .agents/porting-inventory.md
and cmake/TritonAOT.cmake for the full contract.
EOF
fi
exit "${status}"
