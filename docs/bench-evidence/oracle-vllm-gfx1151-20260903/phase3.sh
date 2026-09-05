#!/usr/bin/env bash
# #2740 -- PHASE 3: run Qwen3.8-27B on the PINNED vLLM on gfx1151 and record
# exactly what it emits. Route 1 is GGUF Q4_K_M through vllm-gguf-plugin,
# because that is the arm actually under gate and it is 17 GB, not 54.
#
# CORRECTNESS ONLY. No timing is taken. AGENTS.md Gates admits no performance
# result from an arm whose declared token gate has not passed, and #2497 has
# already had one measurement retracted for exactly that.
#
# NO HSA_OVERRIDE_GFX_VERSION anywhere.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

W=/workspace/vllm-gfx1151-2740
LOCAL=/tmp/vllm-gfx1151-2740
VENV=$LOCAL/venv
PY=$VENV/bin/python
PLUGIN_SHA=d4c1f0d082fc7cd4350da56689109a01c1f29d6c
PLUGIN_TGZ_SHA=9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b
CKPT=/workspace/ckpt/qwen38-27b-q4km
GGUF=$CKPT/Qwen3.8-27B-Q4_K_M.gguf
GGUF_BYTES=17106775008
GGUF_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
MMPROJ=/workspace/ggufplugin/mmproj-BF16.gguf
TOKDIR=/workspace/ckpt/qwen3.8-27b-hf
FLOOR_MB=6000

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$LOCAL" || exit 90
exec > >(tee -a "$LOCAL/phase3-$TAG.log") 2>&1
( while true; do cp -f "$LOCAL/phase3-$TAG.log" "$OUT/phase3.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }
fail() { echo "FATAL: $*"; echo "PHASE3_VERDICT=INSTRUMENT_FAILURE"; exit 1; }

step "0. identity"
echo "hostname=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_|VLLM_)' | sed 's/^/inherited_env /' || echo "inherited_env NONE"
free -g | head -2; df -h /tmp | tail -1
[ -x "$PY" ] || fail "the phase-2 venv is gone; /tmp did not survive"
cd /
"$PY" - <<'PY' || exit 1
import vllm, torch
print("VLLM", vllm.__version__, vllm.__file__)
print("TORCH", torch.__version__, "hip", torch.version.hip)
print("ARCH", torch.cuda.get_device_properties(0).gcnArchName)
assert "site-packages" in vllm.__file__ or "vllm-pin" in vllm.__file__, vllm.__file__
assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151")
import vllm._C, vllm._rocm_C  # noqa
print("PHASE2_ARTIFACT_OK")
PY

step "0a. torchvision must be the ROCm build, or transformers cannot be imported"
# Attempt 1 (job d77ac9ac) got this far and then died on
#   RuntimeError: operator torchvision::nms does not exist
# raised from transformers/image_utils.py:54, which vllm.transformers_utils
# imports unconditionally. vLLM's dependency install pulled the DEFAULT PyPI
# torchvision, whose compiled ops are built against a different torch. The
# package imports; only its operator library is absent. Again: not gfx1151.
"$PY" - <<'PYEOF'
import importlib.util as u
print("torchvision_present =", u.find_spec("torchvision") is not None)
PYEOF
"$VENV/bin/pip" install --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/rocm7.2 torchvision 2>&1 | tail -4
echo "pip_torchvision_rc=${PIPESTATUS[0]}"
"$PY" - <<'PYEOF'
import torch, torchvision
print("TORCHVISION =", torchvision.__version__, "against torch", torch.__version__)
torch.ops.torchvision.nms  # the operator whose absence stopped attempt 1
print("TORCHVISION_OPS=OK")
PYEOF
echo "torchvision_rc=$?"

step "0b. amdsmi -- vLLM RESOLVES ITS PLATFORM THROUGH IT"
# Phase 2 measured this and it is the sharpest instrument trap of the campaign:
# vllm/platforms/__init__.py:110-128 decides "am I ROCm?" by importing amdsmi and
# counting processor handles. With amdsmi absent the answer is NO, vLLM resolves
# UnspecifiedPlatform, and current_platform.get_device_name raises
# NotImplementedError -- on a box whose ROCm stack is otherwise fully working. A
# run under that condition would read as "vLLM cannot run here" and would be a
# lie. So this is ASSERTED, not attempted.
if ! "$PY" -c 'import amdsmi' 2>/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  if [ ! -d /opt/rocm/share/amd_smi ]; then
    apt-get update -qq >> "$LOCAL/apt.log" 2>&1
    apt-get install -y -qq amd-smi-lib >> "$LOCAL/apt.log" 2>&1; echo "apt_amdsmi_rc=$?"
  fi
  ls -d /opt/rocm/share/amd_smi 2>&1
  "$VENV/bin/pip" install -q /opt/rocm/share/amd_smi 2>&1 | tail -3
  echo "pip_amdsmi_rc=${PIPESTATUS[0]}"
fi
"$PY" "$W/probe_platform.py" || fail "vLLM does not resolve RocmPlatform; every result below would be about the wrong platform"
echo "platform_rc=$?"

step "0c. the compiled extensions, probed correctly this time"
# Phase 2 imported only vllm._C and then called torch.ops._C.rms_norm.
# csrc/libtorch_stable/torch_bindings.cpp:10 registers into the _C NAMESPACE from
# the _C_stable_libtorch MODULE, so the op is unreachable until that module is
# imported. The AttributeError phase 2 recorded was the probe, not the build.
"$PY" "$W/probe_ext.py"; echo "customop_rc=$?"

step "0d. does the ARCHITECTURE resolve, now that the platform does?"
"$PY" "$W/probe_registry.py"; echo "registry_rc=$?"

step "1. python3-dev -- Triton's AMD driver COMPILES hip_utils.c at import time"
# Phase 2 measured this: without Python.h the driver raises inside
# triton.runtime.build and every Triton kernel on this board is unreachable.
# vLLM's RDNA paths ARE Triton kernels, so this is a precondition of the run and
# not a convenience.
if ! [ -f /usr/include/python3.12/Python.h ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq > "$LOCAL/apt.log" 2>&1; echo "apt_update_rc=$?"
  apt-get install -y -qq python3-dev >> "$LOCAL/apt.log" 2>&1; echo "apt_install_rc=$?"
fi
ls -l /usr/include/python3.12/Python.h 2>&1
cat > "$LOCAL/triton_probe.py" <<'PYEOF'
import torch, triton, triton.language as tl
@triton.jit
def add1(X, Y, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(Y + i, tl.load(X + i) + 1.0)
print("triton.__version__ =", triton.__version__)
x = torch.arange(256, device="cuda", dtype=torch.float32)
y = torch.empty_like(x); add1[(1,)](x, y, 256); torch.cuda.synchronize()
ok = bool((y == x + 1).all().item())
print("TRITON_JIT_ON_GFX1151 =", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PYEOF
"$PY" "$LOCAL/triton_probe.py"; echo "triton_probe_rc=$?"

step "2. inputs exist -- a missing one is an ABSENCE, never a result"
for f in "$GGUF" "$CKPT/config.json" "$MMPROJ" "$TOKDIR/tokenizer_config.json" \
         "$W/ggufplugin-src.tar.gz" "$W/gen_rocm.py"; do
  [ -f "$f" ] && echo "PRESENT $f" || { echo "ABSENT  $f"; fail "missing input $f"; }
done
SZ=$(stat -c %s "$GGUF"); echo "GGUF_BYTES=$SZ EXPECTED=$GGUF_BYTES"
[ "$SZ" = "$GGUF_BYTES" ] || fail "wrong artifact size"
if [ -f "$LOCAL/gguf.sha256" ]; then
  GOT=$(cat "$LOCAL/gguf.sha256"); echo "GGUF sha256 from this worker's earlier verification: $GOT"
else
  echo "hashing the GGUF on the worker (17 GB, a few minutes)"
  GOT=$(sha256sum "$GGUF" | cut -d' ' -f1); echo "$GOT  $GGUF"
  echo "$GOT" > "$LOCAL/gguf.sha256"
fi
[ "$GOT" = "$GGUF_SHA" ] || fail "GGUF sha mismatch $GOT"
echo "GGUF_SHA_OK"
GOT=$(sha256sum "$W/ggufplugin-src.tar.gz" | cut -d' ' -f1)
[ "$GOT" = "$PLUGIN_TGZ_SHA" ] || fail "plugin archive is not the staged one"
echo "PLUGIN_TGZ_OK (git-archive of $PLUGIN_SHA; the sha256 is what binds it)"

step "3. build the GGUF plugin FOR gfx1151"
SRC=$LOCAL/gplug-src
rm -rf "$SRC" "$LOCAL/gplug-wheel"; mkdir -p "$SRC"
tar -xzf "$W/ggufplugin-src.tar.gz" -C "$SRC" || fail untar
ls -1 "$SRC/vllm_gguf_plugin/csrc/gguf/"
grep -n "is_rocm" "$SRC/setup.py"
export PYTORCH_ROCM_ARCH=gfx1151      # torch cpp_extension does NOT derive this
export MAX_JOBS=4
echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH MAX_JOBS=$MAX_JOBS"
cd "$SRC"
"$VENV/bin/pip" wheel . --no-build-isolation --no-deps -w "$LOCAL/gplug-wheel" \
  > "$LOCAL/plugin_build.log" 2>&1
PLUGIN_RC=$?
echo "PLUGIN_BUILD_RC=$PLUGIN_RC"
cp -f "$LOCAL/plugin_build.log" "$OUT/plugin_build.log" 2>/dev/null
tail -25 "$LOCAL/plugin_build.log"
ls -l "$LOCAL/gplug-wheel" 2>/dev/null
if [ "$PLUGIN_RC" -ne 0 ]; then
  echo "ROUTE1_GGUF=PLUGIN_BUILD_FAILED"
  grep -nE 'error:|Error' "$LOCAL/plugin_build.log" | tail -30
else
  cp -L "$LOCAL"/gplug-wheel/*.whl "$OUT/" 2>/dev/null; sha256sum "$OUT"/vllm_gguf_plugin-*.whl 2>/dev/null
fi
cd /

step "4. install the plugin AND its declared dependencies"
# NEVER --no-deps for the deps: pyproject declares gguf and huggingface_hub, and
# starving the plugin of `gguf` reads as a plugin defect (#2624, 20260903T004110Z).
# --force-reinstall IS load-bearing: the wheel version is always 0.0.5, so a plain
# install into a reused venv leaves the PREVIOUS object in place.
if [ "$PLUGIN_RC" -eq 0 ]; then
  "$VENV/bin/pip" install -q "gguf>=0.17.0" "huggingface_hub>=1.26.0" 2>&1 | tail -3
  echo "deps_rc=${PIPESTATUS[0]}"
  "$VENV/bin/pip" install --force-reinstall --no-deps "$LOCAL"/gplug-wheel/*.whl 2>&1 | tail -4
  echo "plugin_install_rc=${PIPESTATUS[0]}"
  # ★ INTERROGATE THE INSTALLED OBJECT, not the wheel. Those two have disagreed.
  # 2>/dev/null, not 2>&1: attempt 2 captured a ROCm deprecation WARNING into
  # this variable, so the -f test failed and the arch check silently did not run.
  INST_SO=$("$PY" -c "import vllm_gguf_plugin._C_gguf as c; print(c.__file__)" 2>/dev/null | tail -1)
  echo "INSTALLED_SO=$INST_SO"
  if [ -f "$INST_SO" ]; then
    llvm-objdump --offloading "$INST_SO" > "$LOCAL/plugin-offload.txt" 2>&1
    echo "plugin_offload_rc=$?"
    grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z:+-]+' "$LOCAL/plugin-offload.txt" | sort | uniq -c
    cp -f "$LOCAL/plugin-offload.txt" "$OUT/plugin-offload.txt" 2>/dev/null
    if grep -q 'amdhsa--gfx1151' "$LOCAL/plugin-offload.txt"; then
      echo "INSTALLED_PLUGIN_ARCH_OK: gfx1151 is in the object that will be called"
    else
      echo "INSTALLED_PLUGIN_ARCH_MISSING: every launch would fail with 'no kernel image',"
      echo "which reads like a model failure and is not one."
    fi
  fi
  "$PY" - <<'PY'
import importlib.metadata as md
import vllm_gguf_plugin, torch
print("PLUGIN_FILE =", vllm_gguf_plugin.__file__)
eps = list(md.entry_points(group="vllm.general_plugins"))
print("GENERAL_PLUGIN_ENTRY_POINTS =", [(e.name, e.value) for e in eps])
assert any(e.name == "gguf" for e in eps), "the gguf entry point is not registered"
import gguf, huggingface_hub  # noqa
print("REGISTRATION OK")
PY
  echo "registration_rc=$?"
fi

step "5. GENERATION (watchdogged: this fleet has OOM-REBOOTED)"
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A" >> "$LOCAL/mem.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      G=$(cat "$LOCAL/gen.pgid" 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing group $G"
      [ -n "$G" ] && kill -9 -- "-$G" 2>/dev/null
    fi
    sleep 2
  done ) > "$LOCAL/watchdog.log" 2>&1 &
WD=$!

run_leg() {  # run_leg <name> <extra env assignments...>
  local name=$1; shift
  echo; echo "--- LEG $name ---"; date -u +%FT%TZ
  env "$@" OUT_JSON="$LOCAL/tokens-$name.json" \
    setsid timeout 3000 "$PY" "$W/gen_rocm.py" > "$LOCAL/gen-$name.out" 2>&1 &
  local p=$!; echo "$p" > "$LOCAL/gen.pgid"
  wait "$p"; local r=$?
  rm -f "$LOCAL/gen.pgid"; kill -9 -- "-$p" 2>/dev/null
  echo "GEN_RC[$name]=$r"
  cp -f "$LOCAL/gen-$name.out" "$OUT/gen-$name.out" 2>/dev/null
  cp -f "$LOCAL/tokens-$name.json" "$OUT/tokens-$name.json" 2>/dev/null
  grep -E "VLLM_VERSION|TORCH |DEVICE |ON_GFX1151|PLUGIN_C_EXT|PROMPT_LENS|PROMPT_IDS_MATCH_ORACLE|ENGINE_UP|GEN_IDS|GEN_TEXT|GEN_LEN|WROTE|DONE_MARKER" "$LOCAL/gen-$name.out"
  echo "--- tail of LEG $name, in case it died ---"; tail -30 "$LOCAL/gen-$name.out"
}

if [ "$PLUGIN_RC" -eq 0 ]; then
  run_leg gguf-eager MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf \
          GMU=0.60 MAXLEN=2048 EAGER=1
  if grep -q DONE_MARKER_VLLM_GFX1151_GEN "$LOCAL/gen-gguf-eager.out"; then
    run_leg gguf-eager-2 MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf \
            GMU=0.60 MAXLEN=2048 EAGER=1
    run_leg gguf-compiled MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf \
            GMU=0.60 MAXLEN=2048 EAGER=0
  fi
fi
# Route 2 only if route 1 never emitted a token. 53.8 GB against a 64 GiB carve.
if ! grep -qs DONE_MARKER_VLLM_GFX1151_GEN "$LOCAL"/gen-gguf-*.out; then
  echo "ROUTE1 emitted nothing -- falling back to bf16 safetensors (quant-MISMATCHED:"
  echo "it can validate the MODEL IMPLEMENTATION, never the Q4_K arm)."
  run_leg bf16-eager MODEL="$TOKDIR" TOK="$TOKDIR" GMU=0.92 MAXLEN=2048 EAGER=1
fi

kill -9 "$WD" 2>/dev/null
step "6. postcondition"
free -g | head -2
awk '{if(min==""||$2<min)min=$2} END{print "minMemAvailable_MB="min" samples="NR}' "$LOCAL/mem.samples" 2>/dev/null
tail -3 "$LOCAL/watchdog.log" 2>/dev/null
for f in "$LOCAL"/tokens-*.json; do [ -f "$f" ] && { echo "TOKENS $f"; sha256sum "$f"; }; done
dmesg 2>/dev/null | grep -iE 'gpu|amdgpu|ring|reset' | tail -10 || echo "dmesg unavailable in this container"

step "7. done"
cp -f "$LOCAL/phase3-$TAG.log" "$OUT/phase3.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== PHASE3 DONE ==="
