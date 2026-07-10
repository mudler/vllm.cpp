#!/usr/bin/env bash
# Integration test for scripts/check-triton-aot-drift.sh. The fixture copies the
# real build contract and vendored tree, then proves every independently stale
# surface fails without Python, Triton, CUDA, or a GPU.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/vllm-triton-aot-test.XXXXXX")
trap 'rm -rf "${tmp}"' EXIT

base="${tmp}/base"
mkdir -p "${base}/scripts" "${base}/cmake" "${base}/triton_kernels" \
  "${base}/src/vt/cuda/triton_aot_vendored"
cp "${repo}/scripts/check-triton-aot-drift.sh" "${base}/scripts/"
cp "${repo}/scripts/triton-aot-compile.py" "${base}/scripts/"
cp "${repo}/cmake/TritonAOTKernels.cmake" \
  "${repo}/cmake/DumpTritonAOTContract.cmake" "${base}/cmake/"
cp "${repo}"/triton_kernels/*.py "${base}/triton_kernels/"
cp -R "${repo}/src/vt/cuda/triton_aot_vendored/sm_121a" \
  "${base}/src/vt/cuda/triton_aot_vendored/"

check() {
  TRITON_AOT_CHECK_ROOT="$1" bash "$1/scripts/check-triton-aot-drift.sh"
}

expect_fail() {
  local name=$1
  local root=$2
  if check "${root}" >"${tmp}/${name}.out" 2>&1; then
    echo "expected drift check to fail: ${name}" >&2
    cat "${tmp}/${name}.out" >&2
    exit 1
  fi
}

check "${base}" >/dev/null

case_root="${tmp}/missing-base"
cp -R "${base}" "${case_root}"
manifest="${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST"
awk 'index($0, "base gdn_chunko_bf16_h32 ") != 1' "${manifest}" \
  >"${manifest}.tmp"
mv "${manifest}.tmp" "${manifest}"
expect_fail missing-base "${case_root}"

case_root="${tmp}/signature-drift"
cp -R "${base}" "${case_root}"
manifest="${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST"
awk '{ if (index($0, "base gdn_chunko_bf16_h48 ") == 1) sub(/warps=4/, "warps=8"); print }' \
  "${manifest}" >"${manifest}.tmp"
mv "${manifest}.tmp" "${manifest}"
expect_fail signature-drift "${case_root}"

case_root="${tmp}/target-drift"
cp -R "${base}" "${case_root}"
manifest="${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST"
awk '{ if (index($0, "triton_target ") == 1) $0="triton_target cuda:90:32"; print }' \
  "${manifest}" >"${manifest}.tmp"
mv "${manifest}.tmp" "${manifest}"
expect_fail target-drift "${case_root}"

case_root="${tmp}/line-info-drift"
cp -R "${base}" "${case_root}"
manifest="${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/MANIFEST"
awk '$0 != "line_info disabled"' "${manifest}" >"${manifest}.tmp"
mv "${manifest}.tmp" "${manifest}"
expect_fail line-info-drift "${case_root}"

case_root="${tmp}/generator-drift"
cp -R "${base}" "${case_root}"
printf '\n# drift fixture\n' >>"${case_root}/scripts/triton-aot-compile.py"
expect_fail generator-drift "${case_root}"

case_root="${tmp}/missing-dispatcher"
cp -R "${base}" "${case_root}"
rm "${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/gdn_chunko_bf16_h32.h"
expect_fail missing-dispatcher "${case_root}"

case_root="${tmp}/missing-specialization"
cp -R "${base}" "${case_root}"
spec=
for candidate in \
  "${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a"/gdn_chunko_bf16_h48.*.c; do
  if [ -f "${candidate}" ]; then
    spec=${candidate}
    break
  fi
done
[ -n "${spec}" ] || { echo "fixture specialization not found" >&2; exit 1; }
rm "${spec}"
expect_fail missing-specialization "${case_root}"

case_root="${tmp}/unexpected-artifact"
cp -R "${base}" "${case_root}"
touch "${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/not_declared.c"
expect_fail unexpected-artifact "${case_root}"

case_root="${tmp}/artifact-hash-drift"
cp -R "${base}" "${case_root}"
printf '\n/* drift fixture */\n' >> \
  "${case_root}/src/vt/cuda/triton_aot_vendored/sm_121a/gdn_chunko_bf16_h32.c"
expect_fail artifact-hash-drift "${case_root}"

case_root="${tmp}/source-drift"
cp -R "${base}" "${case_root}"
printf '\n# drift fixture\n' >>"${case_root}/triton_kernels/chunk_o.py"
expect_fail source-drift "${case_root}"

echo "Triton AOT drift contract tests passed"
