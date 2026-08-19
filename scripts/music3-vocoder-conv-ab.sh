#!/usr/bin/env bash
# Two-tree A/B for MiniMax-Music3's vocoder decode window (#672, #1334).
#
# THE SHAPE IS THE POINT. Spec §18.8 binds this row to arms that are two SOURCE
# TREES in two BUILD DIRECTORIES whose binaries have DIFFERENT sha256, because
# §16.6a was voided when both arms turned out to be the same binary and the tell
# was identical call counts rather than equal times. This script therefore
# HARD-FAILS when the two harness binaries hash the same, before it times
# anything.
#
# It builds from a clone rather than from a working tree so that the arms cannot
# share an object cache, and it builds in /tmp rather than on any network mount.
# It is CPU-only: the change under test is a host kernel.
#
# usage: music3-vocoder-conv-ab.sh <git-url> <after-ref> <before-ref> [workdir]
#
#   after-ref   the ref carrying the change
#   before-ref  the ref to take `src/vt/cpu/cpu_conv1d_general.cpp` FROM. Only
#               that file is taken, so the two trees differ in the kernel and in
#               nothing else — the harness, the config and the build flags are
#               byte-identical across the arms.
set -euo pipefail

url=${1:?git url}
after=${2:?after ref}
before=${3:?before ref}
work=${4:-/tmp/music3-vocoder-ab}
lengths=${LENGTHS:-20,40,86,172,344}
repeats=${REPEATS:-3}
rounds=${ROUNDS:-3}
jobs=${JOBS:-8}
kernel=src/vt/cpu/cpu_conv1d_general.cpp

say() { printf '\n=== %s\n' "$*"; }

say "BOX"
uname -a
nproc
grep -m1 'model name\|Model' /proc/cpuinfo || true
echo "uptime(before-build): $(uptime)"

rm -rf "$work"
mkdir -p "$work"
say "CLONE $after"
git clone --quiet "$url" "$work/after"
git -C "$work/after" checkout --quiet "$after"
after_sha=$(git -C "$work/after" rev-parse HEAD)
cp -a "$work/after" "$work/before"
git -C "$work/before" checkout --quiet "$before" -- "$kernel"
before_sha=$(git -C "$work/before" rev-parse HEAD)

say "THE ONLY SOURCE DIFFERENCE BETWEEN THE ARMS"
diff -rq "$work/before/src" "$work/after/src" || true
echo "kernel sha256:"
sha256sum "$work/before/$kernel" "$work/after/$kernel"
if [[ "$(sha256sum <"$work/before/$kernel" | cut -d' ' -f1)" == \
      "$(sha256sum <"$work/after/$kernel" | cut -d' ' -f1)" ]]; then
  echo "FATAL: the two trees carry the SAME kernel; there is no A/B here" >&2
  exit 4
fi

for arm in before after; do
  say "BUILD $arm"
  cmake -S "$work/$arm" -B "$work/build-$arm" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DVLLM_CPP_BUILD_TESTS=ON \
    -DVLLM_CPP_CUDA=OFF -DVLLM_CPP_TRITON=OFF >"$work/cfg-$arm.log" 2>&1
  ninja -C "$work/build-$arm" -j "$jobs" vllm_music3_vocoder_conv_ab \
    >"$work/build-$arm.log" 2>&1
  echo "built $arm: rc=$?"
done

say "BINARY IDENTITY — the assertion that voided spec 16.6a"
sha256sum "$work/build-before/vllm_music3_vocoder_conv_ab" \
          "$work/build-after/vllm_music3_vocoder_conv_ab"
a=$(sha256sum <"$work/build-before/vllm_music3_vocoder_conv_ab" | cut -d' ' -f1)
b=$(sha256sum <"$work/build-after/vllm_music3_vocoder_conv_ab" | cut -d' ' -f1)
if [[ "$a" == "$b" ]]; then
  echo "FATAL: both arms are the SAME BINARY; no measurement is possible" >&2
  exit 5
fi

say "CORRECTNESS on the AFTER arm, before any speed number is read"
ninja -C "$work/build-after" -j "$jobs" test_ops_conv1d_general test_host_parallel \
  test_vocoder1d test_bigvgan >>"$work/build-after.log" 2>&1
for t in test_ops_conv1d_general test_host_parallel test_vocoder1d test_bigvgan; do
  echo "--- $t"
  "$work/build-after/tests/$t" 2>&1 | grep -E 'test cases:|assertions:|Status:|SKIP' || true
  echo "rc=${PIPESTATUS[0]}"
done

say "SWEEP — arms ALTERNATED, $rounds rounds, best-of-$repeats per point"
echo "uptime(before-sweep): $(uptime)"
for round in $(seq 1 "$rounds"); do
  for arm in before after; do
    echo "--- round $round arm $arm"
    "$work/build-$arm/vllm_music3_vocoder_conv_ab" --lengths="$lengths" --repeats="$repeats"
  done
done
echo "uptime(after-sweep): $(uptime)"

say "PROVENANCE"
echo "after  ref=$after  sha=$after_sha"
echo "before ref=$before sha=$before_sha (kernel only)"
