#!/usr/bin/env bash
# check-triton-aot-drift.sh — verify the vendored Triton AOT artifacts are in
# sync with their generating sources and with the build-declared dispatcher ABI,
# WITHOUT Python/Triton/GPU (CMake script evaluation + text/hash checks, safe on
# any host including GitHub-hosted CI).
#
# For every src/vt/cuda/triton_aot_vendored/<arch>/MANIFEST:
#   * compare source hashes;
#   * compare the exact base/signature set with cmake/TritonAOTKernels.cmake;
#   * require each stable dispatcher .c/.h and every per-specialization .c/.h
#     pair, with no undeclared artifact bases.
# Exit 0 = in sync; exit 1 = drift (stale vendored artifacts) with a report.
#
# This is the CI face of the staleness guard that cmake/TritonAOT.cmake also
# enforces at configure time on CUDA builds. Regenerating is a maintainer/GPU
# task: scripts/regen-triton-aot.sh (see that header).
set -euo pipefail
ROOT_DIR=${TRITON_AOT_CHECK_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "${ROOT_DIR}"

KERNELS_DIR=triton_kernels
VENDORED_DIR=src/vt/cuda/triton_aot_vendored
status=0
found_manifest=0
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/vllm-triton-aot-check.XXXXXX")
trap 'rm -rf "${tmpdir}"' EXIT

sha256_file() {
  cmake -E sha256sum "$1" | awk '{print $1}'
}

expected_bases="${tmpdir}/expected-bases"
cmake -DOUTPUT="${expected_bases}" -P cmake/DumpTritonAOTContract.cmake
if [ ! -s "${expected_bases}" ]; then
  echo "error: canonical Triton AOT contract is empty" >&2
  exit 2
fi

for manifest in "${VENDORED_DIR}"/*/MANIFEST; do
  [ -f "${manifest}" ] || continue
  found_manifest=1
  arch_dir=$(dirname "${manifest}")
  arch=$(basename "${arch_dir}")
  echo "== ${manifest}"

  capability=${arch#sm_}
  capability=${capability%a}
  case "${capability}" in
    ''|*[!0-9]*)
      echo "DRIFT  ${arch_dir}: cannot derive a CUDA capability from ${arch}"
      status=1
      ;;
    *)
      expected_target="cuda:${capability}:32"
      if ! grep -q "^arch ${arch}$" "${manifest}"; then
        echo "DRIFT  ${arch_dir}: MANIFEST arch does not match ${arch}"
        status=1
      fi
      if ! grep -q "^triton_target ${expected_target}$" "${manifest}"; then
        echo "DRIFT  ${arch_dir}: MANIFEST target must be ${expected_target}"
        status=1
      fi
      if ! grep -q '^line_info disabled$' "${manifest}"; then
        echo "DRIFT  ${arch_dir}: MANIFEST must record path-independent line_info disabled"
        status=1
      fi
      generator_hash=$(sha256_file scripts/triton-aot-compile.py)
      if ! grep -q "^generator scripts/triton-aot-compile.py sha256=${generator_hash}$" \
        "${manifest}"; then
        echo "DRIFT  ${arch_dir}: generator shim hash differs"
        status=1
      fi
      ;;
  esac

  actual_bases="${tmpdir}/$(basename "${arch_dir}").actual-bases"
  grep '^base ' "${manifest}" | LC_ALL=C sort >"${actual_bases}" || true
  if ! cmp -s "${expected_bases}" "${actual_bases}"; then
    echo "DRIFT  ${arch_dir}: MANIFEST base/signature set differs from the build contract"
    diff -u "${expected_bases}" "${actual_bases}" || true
    status=1
  fi

  while read -r _ src expected; do
    expected="${expected#sha256=}"
    if [ ! -f "${KERNELS_DIR}/${src}" ]; then
      echo "DRIFT  ${arch_dir}: source ${src} listed in MANIFEST but missing from ${KERNELS_DIR}/"
      status=1
      continue
    fi
    actual=$(sha256_file "${KERNELS_DIR}/${src}")
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


  while IFS= read -r line; do
    rest=${line#base }
    base=${rest%% *}
    signature=${line#* signature=}
    expected_specs=$(awk -F'|' '{print NF}' <<<"${signature}")

    for ext in c h; do
      if [ ! -f "${arch_dir}/${base}.${ext}" ]; then
        echo "DRIFT  ${arch_dir}: ${base} is declared but stable dispatcher ${base}.${ext} is missing"
        status=1
      fi
    done

    shopt -s nullglob
    spec_cs=("${arch_dir}/${base}."*.c)
    spec_hs=("${arch_dir}/${base}."*.h)
    shopt -u nullglob
    if [ "${#spec_cs[@]}" -ne "${expected_specs}" ] ||
       [ "${#spec_hs[@]}" -ne "${expected_specs}" ]; then
      echo "DRIFT  ${arch_dir}: ${base} expects ${expected_specs} specialization(s), found ${#spec_cs[@]} .c and ${#spec_hs[@]} .h"
      status=1
    fi
    for spec_c in "${spec_cs[@]}"; do
      stem=${spec_c%.c}
      if [ ! -f "${stem}.h" ]; then
        echo "DRIFT  ${arch_dir}: specialization $(basename "${spec_c}") has no matching .h"
        status=1
      fi
    done
    for spec_h in "${spec_hs[@]}"; do
      stem=${spec_h%.h}
      if [ ! -f "${stem}.c" ]; then
        echo "DRIFT  ${arch_dir}: specialization $(basename "${spec_h}") has no matching .c"
        status=1
      fi
    done
  done <"${expected_bases}"

  shopt -s nullglob
  artifacts=("${arch_dir}"/*.c "${arch_dir}"/*.h)
  shopt -u nullglob
  for artifact in "${artifacts[@]}"; do
    name=$(basename "${artifact}")
    base=${name%%.*}
    if ! grep -q "^base ${base} " "${expected_bases}"; then
      echo "DRIFT  ${arch_dir}: undeclared artifact ${name}"
      status=1
    fi
  done

  manifest_artifacts="${tmpdir}/$(basename "${arch_dir}").manifest-artifacts"
  actual_artifacts="${tmpdir}/$(basename "${arch_dir}").actual-artifacts"
  : >"${manifest_artifacts}"
  while read -r _ name expected; do
    expected=${expected#sha256=}
    echo "${name}" >>"${manifest_artifacts}"
    if [ ! -f "${arch_dir}/${name}" ]; then
      echo "DRIFT  ${arch_dir}: MANIFEST artifact ${name} is missing"
      status=1
      continue
    fi
    actual=$(sha256_file "${arch_dir}/${name}")
    if [ "${actual}" != "${expected}" ]; then
      echo "DRIFT  ${arch_dir}: artifact ${name} hash differs"
      status=1
    fi
  done < <(grep '^artifact ' "${manifest}" || true)
  printf '%s\n' "${artifacts[@]##*/}" | sed '/^$/d' | LC_ALL=C sort >"${actual_artifacts}"
  LC_ALL=C sort -o "${manifest_artifacts}" "${manifest_artifacts}"
  if ! cmp -s "${actual_artifacts}" "${manifest_artifacts}"; then
    echo "DRIFT  ${arch_dir}: MANIFEST artifact inventory does not match generated files"
    diff -u "${manifest_artifacts}" "${actual_artifacts}" || true
    status=1
  fi
done

if [ "${found_manifest}" = 0 ]; then
  echo "error: no ${VENDORED_DIR}/*/MANIFEST found" >&2
  exit 2
fi

if [ "${status}" != 0 ]; then
  cat >&2 <<'EOF'

Vendored Triton AOT artifacts are STALE relative to the build contract or
triton_kernels/*.py.
Regenerate (maintainer task, needs Python+Triton+GPU, e.g. dgx.casa):
    scripts/regen-triton-aot.sh
then review + commit the vendored-tree diff. See .agents/porting-inventory.md
and cmake/TritonAOT.cmake for the full contract.
EOF
fi
exit "${status}"
