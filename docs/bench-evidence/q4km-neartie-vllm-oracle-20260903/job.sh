#!/usr/bin/env bash
# #2809 -- teacher-force the PINNED vLLM 5559679229 on gfx1151 with the Q4_K_M
# arm's own token stream, and score the four conjuncts of the ratified near-tie
# band (multimodal-speed.md §12.2) against BOTH of the oracle's supported
# configurations, eager and compiled.
#
# The build recipe is #2788's, reused verbatim in shape and not re-derived:
# phase1b's venv + torch 2.13.0+rocm7.2, phase2's apt packages and vLLM build,
# phase3's plugin build/install. Every one of those steps exists because a
# specific absent package once read as "vLLM does not run on gfx1151".
#
# CORRECTNESS ONLY. No timing is taken. AGENTS.md §Gates admits no performance
# result from an arm whose declared token gate has not passed, and #2497 has
# already had one measurement retracted for exactly that.
#
# NO HSA_OVERRIDE_GFX_VERSION anywhere.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

PIN=5559679229bc961848b121ccdeaa8fa5d79bec98
ARCHIVE_SHA=7d8bd182057aa17d227caacb7c343258a39af1b8e320fb2bd484c0358b3a98e5
PLUGIN_TGZ_SHA=9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b
S=/workspace/vllm-gfx1151-2740          # #2788's staged archives, read-only here
W=/workspace/q4km-neartie-2809
LOCAL=/tmp/q4km-neartie-2809
SRC=$LOCAL/vllm-pin
VENV=$LOCAL/venv
PY=$VENV/bin/python
CKPT=/workspace/ckpt/qwen38-27b-q4km
GGUF=$CKPT/Qwen3.8-27B-Q4_K_M.gguf
GGUF_BYTES=17106775008
GGUF_SHA=7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169
MMPROJ=/workspace/ggufplugin/mmproj-BF16.gguf
TOKDIR=/workspace/ckpt/qwen3.8-27b-hf
OURS_SHA=8b542c718fd38721d5dd3286a77c91ed30ab495b0c604783b9a2681fcc1ad107
EAGER_SHA=350b5fae5de4a25a62fdc2a839eefbfc69e2538f1eadfaa463b60a4c5d39434a
COMPILED_SHA=034a1e303fc39ac72044e2d4bfa79b084cc10b2867b9ec701288c1ab2978de25
FLOOR_MB=6000

TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" "$LOCAL" || exit 90
exec > >(tee -a "$LOCAL/job-$TAG.log") 2>&1
( while true; do cp -f "$LOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }
fail() { echo "FATAL: $*"; echo "JOB_VERDICT=INSTRUMENT_FAILURE"; exit 1; }

step "0. identity, and the one knob that would void every number below"
echo "hostname=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] || fail "HSA_OVERRIDE_GFX_VERSION is set; the runtime would lie about the device"
if env | grep -qE '^(HSA_|ROCR_|PYTORCH_|HIP_)'; then
  env | grep -E '^(HSA_|ROCR_|PYTORCH_|HIP_)' | sed 's/^/inherited_env /'
else
  echo "inherited_env NONE"
fi
free -g | head -2; df -h /tmp | tail -1; nproc; uname -m

step "1. inputs -- an absent one is an ABSENCE, never a result"
for f in "$GGUF" "$CKPT/config.json" "$MMPROJ" "$TOKDIR/tokenizer_config.json" \
         "$S/vllm-pin.tar.gz" "$S/ggufplugin-src.tar.gz" \
         "$W/neartie_vllm.py" "$W/ours_gen_ids_1.json" "$W/oracle_hip.txt" \
         "$W/tokens-gguf-eager.json" "$W/tokens-gguf-compiled.json"; do
  [ -f "$f" ] && echo "PRESENT $f" || { echo "ABSENT  $f"; fail "missing input $f"; }
done
SZ=$(stat -c %s "$GGUF"); echo "GGUF_BYTES=$SZ EXPECTED=$GGUF_BYTES"
[ "$SZ" = "$GGUF_BYTES" ] || fail "wrong artifact size"
for pair in "$W/ours_gen_ids_1.json:$OURS_SHA" "$W/tokens-gguf-eager.json:$EAGER_SHA" \
            "$W/tokens-gguf-compiled.json:$COMPILED_SHA"; do
  f=${pair%:*}; want=${pair##*:}
  got=$(sha256sum "$f" | cut -d' ' -f1)
  echo "SHA $(basename "$f") $got"
  [ "$got" = "$want" ] || fail "$f is not the committed stream ($got != $want)"
done
echo "RECORDED_STREAMS_OK  (our arm's ids, and both of #2788's vLLM legs)"

step "2. venv + torch 2.13.0+rocm7.2 (CMakeLists.txt:72's TORCH_SUPPORTED_VERSION_ROCM)"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV" || fail "venv"
  "$PY" -m pip install -q -U pip wheel setuptools
  "$PY" -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 "torch==2.13.0+rocm7.2"
  echo "torch_install_rc=$?"
fi
"$PY" - <<'PY' || fail "torch identity"
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("arch_list         =", torch.cuda.get_arch_list())
assert torch.__version__.startswith("2.13.0+rocm7.2"), torch.__version__
assert "gfx1151" in torch.cuda.get_arch_list(), "gfx1151 not in this torch"
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
p = torch.cuda.get_device_properties(0)
print("device[0]", p.name, p.gcnArchName, p.total_memory)
assert p.gcnArchName.startswith("gfx1151"), p.gcnArchName
print("TORCH_IDENTITY=OK")
PY

step "3. the packages whose absence has three times read as a device verdict"
export DEBIAN_FRONTEND=noninteractive
if [ ! -f /usr/include/python3.12/Python.h ] || [ ! -d /usr/include/libdrm ]; then
  apt-get update -qq > "$LOCAL/apt.log" 2>&1; echo "apt_update_rc=$?"
  apt-get install -y -qq python3-dev build-essential libdrm-dev pkg-config \
      libnuma-dev libelf-dev git ccache >> "$LOCAL/apt.log" 2>&1
  echo "apt_install_rc=$?"
fi
ls -l /usr/include/python3.12/Python.h || fail "python3-dev absent; the build would fail for a reason that is not the device"
ls -d /usr/include/libdrm || fail "libdrm-dev absent; torch's own include path would not resolve"
if [ ! -d /opt/rocm/lib/cmake/rocrand ]; then
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
fi
[ -d /opt/rocm/lib/cmake/rocrand ] || fail "rocrand cmake config absent; LoadHIP.cmake:79 would stop the build"
if ! "$PY" -c 'import amdsmi' 2>/dev/null; then
  [ -d /opt/rocm/share/amd_smi ] || apt-get install -y -qq amd-smi-lib >> "$LOCAL/apt.log" 2>&1
  "$PY" -m pip install -q /opt/rocm/share/amd_smi 2>&1 | tail -2
fi
# vllm/platforms/__init__.py:110-128 decides "am I ROCm?" through amdsmi. With it
# absent vLLM resolves UnspecifiedPlatform on a working box, and the run reads
# exactly like "vLLM cannot run here". ASSERTED, not attempted.
"$PY" -c 'import amdsmi; amdsmi.amdsmi_init(); print("AMDSMI_HANDLES =", len(amdsmi.amdsmi_get_processor_handles()))' || fail "amdsmi"
cp -f "$LOCAL/apt.log" "$OUT/apt.log" 2>/dev/null

step "4. stage the PINNED source and ASSERT it"
if [ ! -d "$SRC" ]; then
  got=$(sha256sum "$S/vllm-pin.tar.gz" | cut -d' ' -f1)
  echo "vllm-pin.tar.gz sha256 = $got"
  [ "$got" = "$ARCHIVE_SHA" ] || fail "archive sha mismatch"
  tar -xzf "$S/vllm-pin.tar.gz" -C "$LOCAL" || fail "extract"
fi
echo "src=$SRC files=$(find "$SRC" -type f | wc -l)"
echo "src_manifest_LC_ALL_C=$( (cd "$SRC" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1) )"
sed -n '52p;72p' "$SRC/CMakeLists.txt"
grep -n '_ON_GFX1151 = ' "$SRC/vllm/platforms/rocm.py"
grep -q 'gfx1151' "$SRC/CMakeLists.txt" || fail "HIP_SUPPORTED_ARCHS lacks gfx1151"

step "5. BUILD vLLM for gfx1151 ONLY"
export PIP_CACHE_DIR=$LOCAL/pipcache; mkdir -p "$PIP_CACHE_DIR"
printf 'torch==2.13.0+rocm7.2\n' > "$LOCAL/constraints.txt"
"$PY" -m pip install -q -c "$LOCAL/constraints.txt" \
  'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=77.0.3,<81.0.0' \
  'setuptools-scm>=8.0' 'setuptools-rust>=1.9.0' wheel jinja2 numpy || fail "build deps"
export CCACHE_DIR=/workspace/ccache-2740
mkdir -p "$CCACHE_DIR" 2>/dev/null || { export CCACHE_DIR=$LOCAL/ccache; mkdir -p "$CCACHE_DIR"; }
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_HIP_COMPILER_LAUNCHER=ccache
export VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4 VLLM_USE_PRECOMPILED=0
export SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0.dev0+g5559679229
echo "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH MAX_JOBS=$MAX_JOBS CCACHE_DIR=$CCACHE_DIR"
cd "$SRC" || fail cd
( while true; do
    cp -f "$LOCAL/build.log" "$OUT/build.log" 2>/dev/null
    echo "[heartbeat $(date -u +%FT%TZ)] free=$(free -g|awk 'NR==2{print $7}')G build_lines=$(wc -l < "$LOCAL/build.log" 2>/dev/null || echo 0)"
    sleep 60; done ) & HB=$!
"$PY" -m pip install --no-build-isolation -c "$LOCAL/constraints.txt" -e . -v > "$LOCAL/build.log" 2>&1
BUILD_RC=$?
kill $HB 2>/dev/null
echo "BUILD_RC=$BUILD_RC   (read from the program, not through a pipe)"
cp -f "$LOCAL/build.log" "$OUT/build.log" 2>/dev/null
grep -oE 'gfx[0-9]{4}' "$LOCAL/build.log" | sort | uniq -c | sort -rn | head -6
[ "$BUILD_RC" = "0" ] || { echo "JOB_VERDICT=BUILD_FAILED"; grep -nE 'error:|FAILED' "$LOCAL/build.log" | tail -40; exit 2; }
cd /
"$PY" - <<'PY' || fail "the built extensions do not load"
import vllm, torch
import vllm._C, vllm._rocm_C, vllm._C_stable_libtorch  # noqa
print("vllm.__version__ =", vllm.__version__)
assert "5559679229" in vllm.__version__, vllm.__version__
from vllm.platforms import current_platform
print("CURRENT_PLATFORM =", current_platform.__class__.__name__)
print("DEVICE_NAME      =", current_platform.get_device_name(0))
print("DEVICE_CAP       =", current_platform.get_device_capability(0))
assert current_platform.__class__.__name__ == "RocmPlatform"
from vllm.platforms.rocm import on_gfx1151
assert on_gfx1151(), "on_gfx1151() is False; every RDNA branch would be dead"
print("PLATFORM_OK")
PY

step "6. torchvision must be the ROCm build, or transformers cannot be imported"
"$VENV/bin/pip" install -q --force-reinstall --no-deps \
  --index-url https://download.pytorch.org/whl/rocm7.2 torchvision 2>&1 | tail -2
"$PY" -c 'import torch, torchvision; torch.ops.torchvision.nms; print("TORCHVISION_OPS=OK", torchvision.__version__)' || fail torchvision

step "7. build and install the GGUF plugin FOR gfx1151"
got=$(sha256sum "$S/ggufplugin-src.tar.gz" | cut -d' ' -f1)
[ "$got" = "$PLUGIN_TGZ_SHA" ] || fail "plugin archive is not the staged one"
echo "PLUGIN_TGZ_OK $got"
PSRC=$LOCAL/gplug-src
rm -rf "$PSRC" "$LOCAL/gplug-wheel"; mkdir -p "$PSRC"
tar -xzf "$S/ggufplugin-src.tar.gz" -C "$PSRC" || fail untar
export PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4
cd "$PSRC"
"$VENV/bin/pip" wheel . --no-build-isolation --no-deps -w "$LOCAL/gplug-wheel" > "$LOCAL/plugin_build.log" 2>&1
PLUGIN_RC=$?
cd /
echo "PLUGIN_BUILD_RC=$PLUGIN_RC"
cp -f "$LOCAL/plugin_build.log" "$OUT/plugin_build.log" 2>/dev/null
[ "$PLUGIN_RC" = "0" ] || { tail -30 "$LOCAL/plugin_build.log"; fail "plugin build"; }
sha256sum "$LOCAL"/gplug-wheel/*.whl
"$VENV/bin/pip" install -q "gguf>=0.17.0" "huggingface_hub>=1.26.0" 2>&1 | tail -2
# --force-reinstall is load-bearing: the wheel version is always 0.0.5.
"$VENV/bin/pip" install -q --force-reinstall --no-deps "$LOCAL"/gplug-wheel/*.whl 2>&1 | tail -2
INST_SO=$("$PY" -c "import vllm_gguf_plugin._C_gguf as c; print(c.__file__)" 2>/dev/null | tail -1)
echo "INSTALLED_SO=$INST_SO"
[ -f "$INST_SO" ] || fail "the plugin extension did not install"
sha256sum "$INST_SO"
llvm-objdump --offloading "$INST_SO" > "$LOCAL/plugin-offload.txt" 2>&1
grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z:+-]+' "$LOCAL/plugin-offload.txt" | sort | uniq -c
cp -f "$LOCAL/plugin-offload.txt" "$OUT/plugin-offload.txt" 2>/dev/null
grep -q 'amdhsa--gfx1151' "$LOCAL/plugin-offload.txt" || fail "the installed plugin object carries no gfx1151; every launch would fail with 'no kernel image'"
echo "INSTALLED_PLUGIN_ARCH_OK"
"$PY" - <<'PY' || fail "plugin registration"
import importlib.metadata as md
eps = list(md.entry_points(group="vllm.general_plugins"))
print("GENERAL_PLUGIN_ENTRY_POINTS =", [(e.name, e.value) for e in eps])
assert any(e.name == "gguf" for e in eps)
import gguf, huggingface_hub  # noqa
print("REGISTRATION OK")
PY

step "8. THE MEASUREMENT -- teacher-force, both oracle configurations"
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A" >> "$LOCAL/mem.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      G=$(cat "$LOCAL/gen.pgid" 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing group $G"
      [ -n "$G" ] && kill -9 -- "-$G" 2>/dev/null
    fi
    sleep 2; done ) > "$LOCAL/watchdog.log" 2>&1 &
WD=$!

run_leg() {   # run_leg <name> <EAGER 0|1> <self-tokens json>
  local name=$1 eager=$2 selfj=$3
  echo; echo "--- LEG $name (enforce_eager=$eager) ---"; date -u +%FT%TZ
  env MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf GMU=0.60 MAXLEN=2048 \
      EAGER="$eager" TOPK=20 \
      OURS_JSON="$W/ours_gen_ids_1.json" SELF_JSON="$selfj" \
      LLAMACPP_TXT="$W/oracle_hip.txt" OUT_JSON="$LOCAL/neartie-$name.json" \
    setsid timeout 3000 "$PY" "$W/neartie_vllm.py" > "$LOCAL/nt-$name.out" 2>&1 &
  local p=$!; echo "$p" > "$LOCAL/gen.pgid"
  wait "$p"; local r=$?
  rm -f "$LOCAL/gen.pgid"; kill -9 -- "-$p" 2>/dev/null
  echo "NEARTIE_RC[$name]=$r"
  cp -f "$LOCAL/nt-$name.out" "$OUT/nt-$name.out" 2>/dev/null
  cp -f "$LOCAL/neartie-$name.json" "$OUT/neartie-$name.json" 2>/dev/null
  echo "--- LEG $name: the four conjuncts ---"
  sed -n '/=== §12.2 NEAR-TIE BAND/,$p' "$LOCAL/nt-$name.out"
  echo "--- tail of LEG $name, in case it died ---"; tail -25 "$LOCAL/nt-$name.out"
}

run_leg eager    1 "$W/tokens-gguf-eager.json"
run_leg compiled 0 "$W/tokens-gguf-compiled.json"

kill -9 "$WD" 2>/dev/null
step "9. postcondition"
awk '{if(min==""||$2<min)min=$2} END{print "minMemAvailable_MB="min" samples="NR}' "$LOCAL/mem.samples" 2>/dev/null
for f in "$LOCAL"/neartie-*.json; do [ -f "$f" ] && sha256sum "$f"; done
cp -f "$LOCAL/job-$TAG.log" "$OUT/job.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== Q4KM_NEARTIE_2809 DONE ==="
