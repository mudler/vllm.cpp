#!/usr/bin/env bash
# P1 campaign planner for BACKEND-BENCH-CUDA-SGLANG-PREFLIGHT.
#
# Pinned by .agents/specs/cuda-sglang-low-concurrency.md.  This checkpoint only
# supports --dry-run: it emits a fail-closed evidence manifest without pulling
# an image, starting a container, or acquiring /tmp/gpu.  P2 will add the exact
# checkpoint-load execution behind one whole-campaign flock.
set -euo pipefail

readonly SG_IMAGE_DEFAULT='docker.io/lmsysorg/sglang:v0.5.13-cu130-runtime@'\
'sha256:9631280f57d95503ed64cf3892de72190aafbfe6e58e90718a019fa775113bfb'

usage() {
  echo "usage: $0 --dry-run [--claim-root DIR] [--image DIGEST] [--vllm-cpp-sha SHA]" >&2
}

dry_run=0
claim_root="${HOME}/work/vllm.cpp-sglang-low-c"
image="${SG_IMAGE_DEFAULT}"
vllm_cpp_sha=""

while (($#)); do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --claim-root)
      (($# >= 2)) || { usage; exit 2; }
      claim_root=$2
      shift 2
      ;;
    --image)
      (($# >= 2)) || { usage; exit 2; }
      image=$2
      shift 2
      ;;
    --vllm-cpp-sha)
      (($# >= 2)) || { usage; exit 2; }
      vllm_cpp_sha=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ((dry_run == 0)); then
  echo "P2 GPU execution is not implemented in this P1 checkpoint; refusing to touch dgx" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z ${vllm_cpp_sha} ]]; then
  vllm_cpp_sha=$(git -C "${repo_root}" rev-parse HEAD)
fi
evidence_root="${claim_root}/evidence/${vllm_cpp_sha}"
manifest="${evidence_root}/manifest.json"

PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${repo_root}/tools/bench/run_serve_low.py" plan \
    --claim-root "${claim_root}" \
    --vllm-cpp-sha "${vllm_cpp_sha}" \
    --image "${image}" \
    --output "${manifest}"

echo "dry-run manifest: ${manifest}" >&2
