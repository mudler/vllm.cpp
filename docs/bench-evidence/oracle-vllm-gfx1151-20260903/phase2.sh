#!/usr/bin/env bash
# #2740 -- PHASE 2: build the PINNED vLLM for gfx1151 and prove the extensions
# load and the architecture resolves. No model is run here.
#
# Identity first, build second. A build of an unasserted tree proves nothing
# about the pin.
#
# NO HSA_OVERRIDE_GFX_VERSION anywhere.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

PIN=5559679229bc961848b121ccdeaa8fa5d79bec98
ARCHIVE_SHA=7d8bd182057aa17d227caacb7c343258a39af1b8e320fb2bd484c0358b3a98e5
W=/workspace/vllm-gfx1151-2740
LOCAL=/tmp/vllm-gfx1151-2740
SRC=$LOCAL/vllm-pin
VENV=$LOCAL/venv
PY=$VENV/bin/python
TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$LOCAL" || exit 90
exec > >(tee -a "$LOCAL/phase2-$TAG.log") 2>&1
( while true; do cp -f "$LOCAL/phase2-$TAG.log" "$OUT/phase2.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }
fail() { echo "FATAL: $*"; echo "PHASE2_VERDICT=FAIL"; exit 1; }

step "0. identity"
echo "hostname=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
free -g | head -2; df -h /tmp | tail -1; nproc

step "1. the venv phase 1b built must still be the torch we measured"
[ -x "$PY" ] || fail "venv absent -- phase 1b's /tmp did not survive; rerun phase 1b first"
"$PY" - <<'PY' || exit 1
import torch, sys
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("arch_list         =", torch.cuda.get_arch_list())
assert torch.__version__.startswith("2.13.0+rocm7.2"), torch.__version__
assert "gfx1151" in torch.cuda.get_arch_list(), "gfx1151 not in this torch"
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
print("TORCH_IDENTITY=OK")
PY

step "1b. python3-dev -- MEASURED PRECONDITION, not a convenience"
# Attempt 1 (job 153a0563) failed here and nowhere else:
#   CMake Error at cmake/utils.cmake:10: Unable to find python matching ...
#   -- Could NOT find Python (missing: Python_INCLUDE_DIRS ... Development.SABIModule)
# The SAME missing header stopped Triton's AMD driver compiling hip_utils.c.
# Neither is a statement about gfx1151.
if [ ! -f /usr/include/python3.12/Python.h ] || [ ! -d /usr/include/libdrm ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq  > "$LOCAL/apt.log" 2>&1; echo "apt_update_rc=$?"
  # libdrm-dev: attempt 3 (job 6a0cb05f) configured cleanly in 256.1s and then
  # failed with `Imported target "torch" includes non-existent path
  # /usr/include/libdrm`. torch's own INTERFACE_INCLUDE_DIRECTORIES names it.
  # Another absent package, and again not a statement about gfx1151.
  apt-get install -y -qq python3-dev build-essential libdrm-dev pkg-config \
      libnuma-dev libelf-dev git >> "$LOCAL/apt.log" 2>&1
  echo "apt_install_rc=$?"; tail -5 "$LOCAL/apt.log"
fi
ls -l /usr/include/python3.12/Python.h || fail "python3-dev still absent; the build would fail for a reason that is not the device"
ls -d /usr/include/libdrm || fail "libdrm-dev still absent; torch's own include path would not resolve"

step "2. the ROCm compiler"
for t in hipcc hipconfig amdclang++ cmake ninja ccache; do
  printf '%-12s %s\n' "$t" "$(command -v $t || echo ABSENT)"
done
hipconfig --version 2>&1 | head -2
cat /opt/rocm/.info/version

step "2b. two phase-1b instrument repairs, rerun here"
# (a) llvm-objdump was piped into `head`, so it died of SIGPIPE (rc=141) after
#     80 lines. Run it to completion and summarise the DISTINCT targets.
TLIB=$("$PY" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
llvm-objdump --offloading "$TLIB/libtorch_hip.so" > "$LOCAL/offload.txt" 2>&1
echo "offload_rc=$?  lines=$(wc -l < "$LOCAL/offload.txt")"
echo "-- DISTINCT offload targets in libtorch_hip.so (complete, not truncated) --"
grep -oE 'hipv4-amdgcn-amd-amdhsa--gfx[0-9a-z:+-]+' "$LOCAL/offload.txt" | sort -u
echo "OFFLOAD_BUNDLES_TOTAL=$(grep -c 'Extracting offload bundle' "$LOCAL/offload.txt")"
echo "OFFLOAD_GFX1151_BUNDLES=$(grep -c 'amdhsa--gfx1151$' "$LOCAL/offload.txt")"
cp -f "$LOCAL/offload.txt" "$OUT/offload.txt" 2>/dev/null
# (b) the Triton probe was written on stdin, and triton.jit needs a real FILE.
#     That ValueError was my harness, not the device. numpy was also absent.
"$PY" -m pip install -q numpy
cat > "$LOCAL/triton_probe.py" <<'PYEOF'
import torch, triton, triton.language as tl
@triton.jit
def add1(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1.0)
print("triton.__version__ =", triton.__version__)
x = torch.arange(256, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
add1[(1,)](x, y, 256)
torch.cuda.synchronize()
ok = bool((y == x + 1).all().item())
print("TRITON_JIT_ON_GFX1151 =", "PASS" if ok else "FAIL")
assert ok
PYEOF
"$PY" "$LOCAL/triton_probe.py"; echo "triton_probe_rc=$?"

step "2c. the ROCm MATH libraries torch's LoadHIP.cmake requires"
# Attempt 2 (job e97bd631) failed here and nowhere else:
#   CMake Error at torch/share/cmake/Caffe2/public/LoadHIP.cmake:79 (find_package):
#   Could not find a package configuration file provided by "rocrand"
# The worker image carries a RUNTIME ROCm, not a development one. Again: an
# absent package, not a statement about gfx1151.
echo "-- cmake package configs present BEFORE (this is what was missing) --"
ls /opt/rocm/lib/cmake 2>/dev/null | tr '\n' ' '; echo
if [ ! -d /opt/rocm/lib/cmake/rocrand ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-cache policy rocm-libs rocrand-dev 2>&1 | head -12
  apt-get update -qq >> "$LOCAL/apt.log" 2>&1
  apt-get install -y -qq --no-install-recommends rocm-libs >> "$LOCAL/apt.log" 2>&1
  echo "apt_rocm_libs_rc=$?"
  if [ ! -d /opt/rocm/lib/cmake/rocrand ]; then
    apt-get install -y -qq --no-install-recommends \
      rocrand-dev hiprand-dev rocblas-dev hipblas-dev hipblaslt-dev \
      miopen-hip-dev rocfft-dev hipfft-dev rocsparse-dev hipsparse-dev \
      rocsolver-dev hipsolver-dev rocprim-dev hipcub-dev rocthrust-dev \
      rccl-dev >> "$LOCAL/apt.log" 2>&1
    echo "apt_rocm_individual_rc=$?"
  fi
  tail -20 "$LOCAL/apt.log"
fi
echo "-- cmake package configs present AFTER --"
ls /opt/rocm/lib/cmake 2>/dev/null | tr '\n' ' '; echo
for pkg in hip hsa-runtime64 amd_comgr rocrand hiprand rocblas hipblas hipblaslt miopen rccl rocprim hipcub rocthrust hipsparse rocsparse hipfft rocfft hipsolver rocsolver; do
  [ -d "/opt/rocm/lib/cmake/$pkg" ] && printf 'ROCM_CMAKE %-14s PRESENT\n' "$pkg" || printf 'ROCM_CMAKE %-14s ABSENT\n' "$pkg"
done
cp -f "$LOCAL/apt.log" "$OUT/apt.log" 2>/dev/null

step "3. stage the PINNED source and ASSERT it"
if [ ! -d "$SRC" ]; then
  echo "archive sha256:"; sha256sum "$W/vllm-pin.tar.gz"
  got=$(sha256sum "$W/vllm-pin.tar.gz" | cut -d' ' -f1)
  [ "$got" = "$ARCHIVE_SHA" ] || fail "archive sha mismatch: $got != $ARCHIVE_SHA"
  mkdir -p "$LOCAL/x" && rm -rf "$LOCAL/x"/* && tar -xzf "$W/vllm-pin.tar.gz" -C "$LOCAL" || fail "extract"
fi
echo "src=$SRC files=$(find "$SRC" -type f | wc -l)"
echo "src_manifest_LC_ALL_C=$( (cd "$SRC" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1) )"
echo "-- the four anchors this campaign was opened on, re-read ON THE WORKER --"
sed -n '52p;72p' "$SRC/CMakeLists.txt"
grep -n '"0x1586"' "$SRC/vllm/platforms/rocm.py"
grep -n '_ON_GFX1151 = ' "$SRC/vllm/platforms/rocm.py"
grep -n 'Qwen3_5ForConditionalGeneration": ("qwen3_5"' "$SRC/vllm/model_executor/models/registry.py"
grep -n 'Tuned for RDNA 3.5' "$SRC/vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
grep -n '_CAST_DOT_TO_K_DTYPE = on_gfx1x()' "$SRC/vllm/third_party/flash_linear_attention/ops/chunk_scaled_dot_kkt.py"
grep -q 'gfx1151' "$SRC/CMakeLists.txt" || fail "HIP_SUPPORTED_ARCHS lacks gfx1151"

step "4. build + runtime dependencies (NEVER --no-deps: that trap is already paid for)"
export PIP_CACHE_DIR=$LOCAL/pipcache; mkdir -p "$PIP_CACHE_DIR"
# A constraint so nothing drags the CUDA torch in over the ROCm one.
printf 'torch==2.13.0+rocm7.2\n' > "$LOCAL/constraints.txt"
"$PY" -m pip install -c "$LOCAL/constraints.txt" \
  'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' \
  'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel jinja2 || fail "build deps"
echo "-- amdsmi from the ROCm install (vllm/platforms/rocm.py:28 imports it) --"
if [ -d /opt/rocm/share/amd_smi ]; then
  "$PY" -m pip install /opt/rocm/share/amd_smi 2>&1 | tail -3
else echo "amd_smi source ABSENT in /opt/rocm/share"; fi
"$PY" -c 'import amdsmi;print("amdsmi OK")' 2>&1 | tail -2

step "5. BUILD vLLM for gfx1151 ONLY"
export CCACHE_DIR=/workspace/ccache-2740
mkdir -p "$CCACHE_DIR" 2>/dev/null || { export CCACHE_DIR=$LOCAL/ccache; mkdir -p "$CCACHE_DIR"; }
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_HIP_COMPILER_LAUNCHER=ccache
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx1151
export MAX_JOBS=4          # unconstrained parallelism has OOM-rebooted these boxes
export SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0+g5559679229  # git archive carries no .git;
# setup.py:1026 joins with "." when the version already has a "+", so the built
# version reads 0.26.0.dev0+g5559679229.rocm724 and CARRIES the pin.
export VLLM_USE_PRECOMPILED=0
echo "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH MAX_JOBS=$MAX_JOBS"
ccache -z 2>/dev/null; ccache -s 2>/dev/null | head -5
cd "$SRC" || fail cd
( while true; do
    cp -f "$LOCAL/build.log" "$OUT/build.log" 2>/dev/null
    echo "[heartbeat $(date -u +%FT%TZ)] free=$(free -g|awk 'NR==2{print $7}')G build_lines=$(wc -l < "$LOCAL/build.log" 2>/dev/null || echo 0) $(tail -1 "$LOCAL/build.log" 2>/dev/null | cut -c1-110)"
    sleep 60; done ) & HB=$!
"$PY" -m pip install --no-build-isolation -c "$LOCAL/constraints.txt" -e . -v > "$LOCAL/build.log" 2>&1
BUILD_RC=$?
kill $HB 2>/dev/null
echo "BUILD_RC=$BUILD_RC   (read from the program, not through a pipe)"
cp -f "$LOCAL/build.log" "$OUT/build.log" 2>/dev/null
echo "-- build.log tail --"; tail -60 "$LOCAL/build.log"
echo "-- gfx targets the build actually asked for --"
grep -oE '\-\-offload-arch=[a-z0-9]+|amdgpu-target=[a-z0-9]+|gfx[0-9]{4}' "$LOCAL/build.log" | sort | uniq -c | sort -rn | head -10
ccache -s 2>/dev/null | head -8
[ "$BUILD_RC" != "0" ] && { echo "PHASE2_VERDICT=BUILD_FAILED"; grep -nE 'error:|Error|FAILED' "$LOCAL/build.log" | tail -40; exit 2; }

step "6. do the compiled extensions LOAD, and does the architecture RESOLVE?"
cd "$LOCAL"      # never import vllm from its own source dir
"$PY" - <<'PY'
import importlib, json, sys
res = {}
for m in ("vllm._C", "vllm._moe_C", "vllm._rocm_C", "vllm._C_stable_libtorch",
          "vllm._moe_C_stable_libtorch"):
    try:
        importlib.import_module(m); res[m] = "LOADED"
    except Exception as e:
        res[m] = f"FAILED: {type(e).__name__}: {e}"
for k, v in res.items(): print(f"EXT {k:32s} {v}")

import vllm, torch
print("vllm.__version__ =", vllm.__version__)
from vllm.platforms import current_platform
print("current_platform =", current_platform.device_name, current_platform.__class__.__name__)
from vllm.platforms.rocm import on_gfx1151, on_gfx1x, on_gfx11
print("on_gfx1151() =", on_gfx1151(), " on_gfx1x() =", on_gfx1x(), " on_gfx11() =", on_gfx11())
print("get_device_name =", current_platform.get_device_name(0))
print("get_device_capability =", current_platform.get_device_capability(0))

from vllm.model_executor.models.registry import ModelRegistry
arch = "Qwen3_5ForConditionalGeneration"
print("is_text_generation_model =", ModelRegistry.is_text_generation_model([arch]))
cls, name = ModelRegistry.resolve_model_cls([arch])
print("RESOLVED", arch, "->", name, cls)
# the linear-attention path this model uses, and its RDNA switch
from vllm.third_party.flash_linear_attention.ops import chunk_scaled_dot_kkt as k
print("_CAST_DOT_TO_K_DTYPE =", k._CAST_DOT_TO_K_DTYPE, "(True is the RDNA WMMA path)")
print("REGISTRY_RESOLVE=OK")
PY
echo "import_rc=$?"

step "7. a custom op on the device -- the extension must COMPUTE, not just load"
"$PY" - <<'PY'
import torch, traceback
try:
    import vllm._C  # noqa
    from vllm import _custom_ops as ops
    d = torch.device("cuda:0")
    x = torch.randn(4, 512, device=d, dtype=torch.bfloat16)
    w = torch.randn(512, device=d, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ops.rms_norm(out, x, w, 1e-6)
    ref = (x.float() / (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()) * w.float()
    print("CUSTOM_OP rms_norm max_abs_err =", (out.float() - ref).abs().max().item())
    y = torch.randn(4, 1024, device=d, dtype=torch.bfloat16)
    o2 = torch.empty(4, 512, device=d, dtype=torch.bfloat16)
    ops.silu_and_mul(o2, y)
    r2 = torch.nn.functional.silu(y[..., :512].float()) * y[..., 512:].float()
    print("CUSTOM_OP silu_and_mul max_abs_err =", (o2.float() - r2).abs().max().item())
    torch.cuda.synchronize()
    print("CUSTOM_OPS=PASS")
except Exception:
    traceback.print_exc(); print("CUSTOM_OPS=FAIL")
PY
echo "customop_rc=$?"

step "8. done"
"$PY" -m pip freeze > "$OUT/pip-freeze.txt" 2>/dev/null
cp -f "$LOCAL/phase2-$TAG.log" "$OUT/phase2.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== PHASE2 DONE ==="
