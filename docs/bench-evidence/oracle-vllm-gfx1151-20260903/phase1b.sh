#!/usr/bin/env bash
# #2740 -- PHASE 1b: does the PyTorch the pinned vLLM needs carry DEVICE CODE for
# gfx1151? Measured on the board, never inferred.
#
# 1a established, on this worker: python 3.12.3, egress OK, /opt/rocm present
# with rocminfo/roc-obj-ls/clang-offload-bundler/llvm-objdump (NOT on PATH), and
# the kernel driver reporting gfx_target_version=110501 on kfd node 1.
# 1a's index probe was a BROKEN INSTRUMENT: it grepped for `linux_x86_64` where
# the index writes `manylinux_2_28_x86_64`, so it read "no wheel" off a working
# index. That is fixed here and the fix is asserted, not assumed.
#
# NO HSA_OVERRIDE_GFX_VERSION anywhere. It makes the runtime lie about the
# device, and a result taken under it cannot pin an oracle.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

W=/workspace/vllm-gfx1151-2740
LOCAL=/tmp/vllm-gfx1151-2740
TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$LOCAL" || exit 90
exec > >(tee -a "$LOCAL/phase1b-$TAG.log") 2>&1
( while true; do cp -f "$LOCAL/phase1b-$TAG.log" "$OUT/phase1b.log" 2>/dev/null; sleep 15; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

step "0. identity"
echo "hostname=$(hostname)"; echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|GPU_)' | sed 's/^/inherited_env /' || echo "inherited_env NONE"
free -g | head -2; df -h /tmp | tail -1

step "1. the ROCm userspace in this container, and the device"
cat /opt/rocm/.info/version 2>/dev/null | sed 's/^/rocm_info_version /'
ls -d /opt/rocm-* 2>/dev/null
rocminfo 2>&1 | grep -E 'Name:|Marketing Name:|gfx|Compute Unit|Uuid' | head -40
echo "-- rocminfo agent gfx targets (verbatim) --"
rocminfo 2>&1 | grep -oE 'gfx[0-9]{3,4}[a-z:+-]*' | sort | uniq -c

step "2. THE INDEX, read with a regex that is proven to match"
PYTAG=$(python3 -c 'import sys;print("cp%d%d"%sys.version_info[:2])')
echo "local_python_tag=$PYTAG"
for idx in rocm7.2 rocm7.1 rocm7.0 rocm6.4; do
  f="$LOCAL/idx-$idx.html"
  code=$(curl -sL -o "$f" -w '%{http_code}' --max-time 60 "https://download.pytorch.org/whl/$idx/torch/")
  vers=$(grep -oE "torch-[0-9][^\"<>]*-${PYTAG}-${PYTAG}-manylinux_2_28_x86_64\.whl" "$f" \
         | sed "s/^torch-//; s/-${PYTAG}.*//; s/%2B/+/" | sort -u -V | tr '\n' ' ')
  echo "INDEX $idx http=$code bytes=$(stat -c%s "$f") ${PYTAG}_manylinux_2_28_x86_64: $vers"
done
# The instrument must be shown to discriminate: a version that cannot be there.
grep -qE "torch-2\.13\.0\+rocm7\.2-${PYTAG}-${PYTAG}-manylinux_2_28_x86_64\.whl" "$LOCAL/idx-rocm7.2.html" \
  && echo "INSTRUMENT_CHECK positive_control=FOUND torch-2.13.0+rocm7.2" \
  || echo "INSTRUMENT_CHECK positive_control=MISSING"
grep -qE "torch-9\.9\.9\+rocm7\.2-${PYTAG}" "$LOCAL/idx-rocm7.2.html" \
  && echo "INSTRUMENT_CHECK negative_control=FOUND (BROKEN)" \
  || echo "INSTRUMENT_CHECK negative_control=absent (the grep discriminates)"

step "3. install torch 2.13.0+rocm7.2 -- the exact version vLLM CMakeLists.txt:72 names"
export PIP_CACHE_DIR=$LOCAL/pipcache   # NOT /workspace: CIFS ownership disables the cache
mkdir -p "$PIP_CACHE_DIR"
VENV=$LOCAL/venv
rm -rf "$VENV"; python3 -m venv "$VENV" || { echo "FATAL venv"; exit 91; }
"$VENV/bin/python" -m pip install -q -U pip wheel setuptools
"$VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 "torch==2.13.0+rocm7.2"
INST_RC=$?
echo "pip_install_rc=$INST_RC   (read from the program, not through a pipe)"
[ "$INST_RC" != "0" ] && { echo "PHASE1_VERDICT=TORCH_INSTALL_FAILED"; exit 93; }
"$VENV/bin/python" -m pip show torch | head -6

step "4. THE MEASUREMENT: gfx targets in the installed libtorch_hip.so"
TLIB=$("$VENV/bin/python" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
echo "torch_lib=$TLIB"
ls -la "$TLIB" | grep -E 'libtorch_hip|amdhip|libhsa|hipblas' | head
SO="$TLIB/libtorch_hip.so"
if [ ! -f "$SO" ]; then echo "libtorch_hip.so ABSENT -- not a ROCm build"; ls "$TLIB" | head -40; echo "PHASE1_VERDICT=NOT_A_ROCM_BUILD"; exit 94; fi
sha256sum "$SO"; stat -c 'libtorch_hip.so bytes=%s' "$SO"

echo; echo "--- llvm-objdump --offloading (VERBATIM) ---"
llvm-objdump --offloading "$SO" 2>&1 | head -80
echo "objdump_rc=$?"

echo; echo "--- roc-obj-ls (VERBATIM) ---"
roc-obj-ls "$SO" 2>&1 | head -80
echo "rocobjls_rc=$?"

echo; echo "--- clang-offload-bundler --list (VERBATIM) ---"
clang-offload-bundler --type=o --input="$SO" --list 2>&1 | head -80
echo "cob_rc=$?"

echo; echo "--- cross-check only (weakest instrument): gfx strings in the .so ---"
strings -a "$SO" | grep -oE 'gfx[0-9]{3,4}[a-z:+-]*' | sort | uniq -c | sort -rn | head -30
echo "GFX1151_IN_LIBTORCH_HIP=$(strings -a "$SO" | grep -c 'gfx1151')"

step "5. what the torch build says about ITSELF, by execution"
"$VENV/bin/python" - <<'PY'
import torch
print("torch.__version__          =", torch.__version__)
print("torch.version.hip          =", torch.version.hip)
print("torch.version.cuda         =", torch.version.cuda)
try:
    print("torch.cuda.get_arch_list() =", torch.cuda.get_arch_list())
except Exception as e:
    print("get_arch_list FAILED:", type(e).__name__, e)
print("torch.cuda.is_available()  =", torch.cuda.is_available())
print("device_count               =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"device[{i}] name={p.name!r} gcnArchName={getattr(p,'gcnArchName',None)!r} "
          f"total_memory={p.total_memory} CUs={p.multi_processor_count}")
PY
echo "self_report_rc=$?"

step "6. DOES IT COMPUTE? real kernels on the real device"
"$VENV/bin/python" - <<'PY'
import os, torch, traceback
assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None, "override set: result would be conditional"
if not torch.cuda.is_available():
    print("SMOKE=SKIP no device visible to torch"); raise SystemExit(0)
d = torch.device("cuda:0")
try:
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        a = torch.randn(1024, 1024, device=d, dtype=dt)
        b = torch.randn(1024, 1024, device=d, dtype=dt)
        c = (a @ b).float().cpu()
        ref = a.float().cpu() @ b.float().cpu()
        rel = ((c - ref).abs().max() / ref.abs().max()).item()
        print(f"SMOKE_MATMUL {str(dt):>16} 1024^3 max_rel_err = {rel:.3e}")
    x = torch.randn(4096, device=d, dtype=torch.bfloat16)
    print("SMOKE_SOFTMAX sum =", torch.softmax(x.float(), 0).sum().item())
    print("SMOKE_SDPA       =", torch.nn.functional.scaled_dot_product_attention(
        *[torch.randn(1,4,128,64, device=d, dtype=torch.float16)]*3).float().abs().mean().item())
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"MEM free={free} total={total}")
    print("SMOKE=PASS")
except Exception:
    traceback.print_exc(); print("SMOKE=FAIL")
PY
echo "smoke_rc=$?"

step "7. Triton: vLLM's RDNA paths on this board are Triton kernels"
"$VENV/bin/python" - <<'PY'
try:
    import triton, torch
    print("triton.__version__ =", triton.__version__)
    import triton.language as tl
    @triton.jit
    def add1(X, Y, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Y + i, tl.load(X + i) + 1.0)
    x = torch.arange(256, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add1[(1,)](x, y, 256)
    torch.cuda.synchronize()
    ok = bool((y == x + 1).all().item())
    print("TRITON_JIT_ON_GFX1151 =", "PASS" if ok else "FAIL")
except Exception as e:
    import traceback; traceback.print_exc(); print("TRITON_JIT_ON_GFX1151 = FAIL", type(e).__name__)
PY
echo "triton_rc=$?"

step "8. done"
cp -f "$LOCAL/phase1b-$TAG.log" "$OUT/phase1b.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== PHASE1B DONE ==="
