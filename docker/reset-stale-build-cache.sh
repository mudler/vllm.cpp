#!/usr/bin/env bash
# Drop a lane's BuildKit cache mount when the toolchain under it changed.
#
# The lane builds keep their build directory in a `--mount=type=cache`, which
# survives across image builds and across base-image bumps. CMake caches
# ABSOLUTE toolchain paths, so moving the CUDA builder from 12.9 to 13.3 left a
# cache whose link line still pointed at /usr/local/cuda-12.9/targets/.../
# libcudart.so, and the build failed with:
#
#   ninja: error: '/usr/local/cuda-12.9/.../libcudart.so', needed by
#   'examples/vllm-server', missing and no known rule to make it
#
# CMake cannot notice this on its own here: the compiler is reached through the
# stable /usr/local/cuda symlink, so its path is unchanged while everything it
# resolves to moved. A stamp of the actual toolchain versions is what changes.
#
# Usage: docker/reset-stale-build-cache.sh BUILD_DIR
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
stamp="${build_dir}/.toolchain-stamp"

want="$(c++ --version | head -n 1)"
want+=$'\n'"$(cmake --version | head -n 1)"
if command -v nvcc >/dev/null 2>&1; then
  want+=$'\n'"$(nvcc --version | tail -n 1)"
  # Resolve the symlink too: /usr/local/cuda is stable while its target moves.
  want+=$'\n'"$(readlink -f /usr/local/cuda 2>/dev/null || echo none)"
fi

mkdir -p "${build_dir}"
if [[ -f "${stamp}" ]] && [[ "$(cat "${stamp}")" == "${want}" ]]; then
  exit 0
fi

if [[ -e "${build_dir}/CMakeCache.txt" ]]; then
  echo "toolchain changed under the build cache; discarding ${build_dir}" >&2
fi
find "${build_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
printf '%s' "${want}" > "${stamp}"
