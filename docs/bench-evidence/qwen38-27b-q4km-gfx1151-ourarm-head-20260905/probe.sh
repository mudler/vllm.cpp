#!/usr/bin/env bash
# PROBE ONLY. Decides which build route the measurement job can take on this
# node. It runs no model, times nothing, and publishes nothing.
set -uo pipefail
W=/workspace/strix-arm-2933
OUT=$W/out
mkdir -p "$OUT"
exec > >(tee "$OUT/probe.log") 2>&1
echo "=== IDENTITY ==="
hostname; date -u +%FT%TZ
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "uptime_s=$(cut -d' ' -f1 /proc/uptime)"
echo "rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo NONE)"
nproc; free -g | head -2; df -h / /tmp /workspace | tail -4

echo "=== ENV HYGIENE (must be clean) ==="
if env | grep -qE '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)'; then
  env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|GGML_|VT_)' | sed 's/^/inherited_env /'
else
  echo "inherited_env NONE"
fi
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}"

echo "=== ROUTE A: the podman image the survey and #2511 used ==="
P=(podman --storage-driver=vfs --root /tmp/podman-pr66-root-vfs --runroot /tmp/podman-pr66-run-vfs)
if command -v podman >/dev/null 2>&1; then
  echo "podman=$(command -v podman) $(podman --version 2>&1)"
  ls -d /tmp/podman-pr66-root-vfs 2>&1
  "${P[@]}" images 2>&1 | head -10
  echo "--- can the image run and see the board? ---"
  "${P[@]}" run --rm --entrypoint bash --device=/dev/kfd --device=/dev/dri --group-add video \
    vllmcpp-rocm-build:7.2.4 -lc 'cat /opt/rocm/.info/version; ls /opt/rocm/lib/llvm/bin/clang++; command -v ccache || echo no-ccache; /opt/rocm/bin/rocminfo 2>/dev/null | grep -m2 gfx' 2>&1 | head -20
  echo "podman_probe_rc=$?"
else
  echo "podman ABSENT"
fi

echo "=== ROUTE B: native toolchain on this node ==="
ls -la /opt/rocm/lib/libamdhip64.so* 2>&1 | head -4
ls -d /opt/rocm/lib/cmake/hipblaslt /opt/rocm/lib/cmake/hipblas 2>&1
ls -la /opt/rocm/llvm/bin/clang++ 2>&1

echo "=== WHAT IS ALREADY STAGED ON THIS NODE ==="
for d in /tmp/rocm-strix-q4k /tmp/rocm-strix-managed /tmp/strix-survey-2497; do
  echo "--- $d ---"; ls -la "$d" 2>&1 | head -12
done
echo "--- staged gguf ---"
for g in /tmp/rocm-strix-q4k/models/Qwen3.8-27B-Q4_K_M.gguf /tmp/rocm-strix-managed/models/Qwen3.8-27B-Q4_K_M.gguf; do
  if [ -f "$g" ]; then echo "PRESENT $g bytes=$(stat -c %s "$g")"; else echo "ABSENT $g"; fi
done
echo "--- the pre-fix binary #2933 names, if it is still here ---"
for b in /tmp/rocm-strix-q4k/build-vllmcpp/examples/vllm-cli; do
  if [ -f "$b" ]; then printf 'PRESENT '; sha256sum "$b"; else echo "ABSENT $b"; fi
done
echo "--- source trees ---"
for s in /tmp/rocm-strix-q4k/src-vllmcpp /tmp/rocm-strix-managed/src; do
  [ -d "$s/.git" ] && echo "$s HEAD=$(git -C "$s" rev-parse HEAD 2>/dev/null)" || echo "$s absent"
done
echo "=== ccache ==="
command -v ccache && ccache -s 2>&1 | head -5 || echo "ccache absent on the node"
du -sh /workspace/ccache 2>/dev/null | head -1
echo "=== board ==="
rocm-smi 2>&1 | head -12
echo "=== PROBE DONE ==="
