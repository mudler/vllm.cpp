#!/bin/bash
# ORACLE-VLLM-GGUF-QWEN35 / issue #2624, on dgx:gpu0 inside an rc lease.
#
# QUESTION: can the PINNED vLLM (5559679229) plus the first-party
# vllm-gguf-plugin load /workspace/ckpt/qwen38-27b-q4km/Qwen3.8-27B-Q4_K_M.gguf
# and EMIT TOKENS? That is AGENTS.md's gateability bar for an oracle.
#
# CORRECTNESS ONLY. No timing is taken and no number in this log is a
# performance result.
#
# Reuses what is already measured on this box rather than rediscovering it:
#   * the FLASHINFER-ONLY wheel at /workspace/oracle-vllm (README-WHEELS.md)
#     -- vLLM's default FLASH_ATTN carries no sm_12x SASS here;
#   * the toolchain, venv, watchdog and postcondition shape of
#     /workspace/a2q1-neartie/job.sh, which ran green on this box.
#
# ORDERING IS DELIBERATE. The plugin wheel is copied to /workspace the moment
# it exists, BEFORE anything loads a model, so a job killed at the ceiling
# still leaves a durable artifact and a registration proof behind.
set -u
OUT=/workspace/ggufplugin
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$OUT/$STAMP
mkdir -p "$RUN"
echo "GGUFPLUGIN run dir: $RUN"
V=/tmp/ggufplug-venv
W=/workspace/oracle-vllm
SRC=/tmp/ggufplug-src
CKPT=/workspace/ckpt/qwen38-27b-q4km
GGUF=$CKPT/Qwen3.8-27B-Q4_K_M.gguf
GGUF_BYTES=17106775008
TOK=/workspace/ckpt/qwen3.8-27b-hf
MMPROJ=$OUT/mmproj-BF16.gguf
PLUGIN_SHA=d4c1f0d082fc7cd4350da56689109a01c1f29d6c
SRC_TGZ_SHA=9e15c20e0b75f75bbf886966df07843c4b70a7952fad4b80e8e8183e2f70743b
FLOOR_MB=15000
export PIP_CACHE_DIR=/workspace/oracle-vllm/pip-cache

step() { echo; echo "########## $* ##########"; date -u +'%Y-%m-%dT%H:%M:%SZ'; }
rcx() { "$@"; local r=$?; echo "RC[$*]=$r"; return $r; }

step "0. BOX"
rcx uname -m
rcx nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
free -m | head -2
cat /proc/sys/kernel/random/boot_id
df -h /tmp /workspace | head -5

step "1. INPUTS EXIST -- a missing one is an ABSENCE, never a result"
rcx test -f "$GGUF"
SZ=$(stat -c %s "$GGUF" 2>/dev/null || echo 0)
echo "GGUF_BYTES=$SZ EXPECTED=$GGUF_BYTES MATCH=$([ "$SZ" = "$GGUF_BYTES" ] && echo yes || echo NO)"
[ "$SZ" = "$GGUF_BYTES" ] || { echo "ABORT: wrong artifact size"; exit 10; }
rcx test -f "$CKPT/config.json"; sha256sum "$CKPT/config.json"
rcx test -f "$TOK/tokenizer_config.json"
rcx test -f "$MMPROJ"; stat -c '%s %n' "$MMPROJ"
rcx test -f "$OUT/ggufplugin-src.tar.gz"
echo -n "SRC_TGZ "; sha256sum "$OUT/ggufplugin-src.tar.gz"
GOT=$(sha256sum "$OUT/ggufplugin-src.tar.gz" | cut -d' ' -f1)
[ "$GOT" = "$SRC_TGZ_SHA" ] || { echo "ABORT: plugin source archive is not the staged one"; exit 13; }
WHEEL=$(ls "$W"/vllm-*.whl 2>/dev/null | grep -v -e REJECTED -e '\.cu13[1-9]')
N=$(printf '%s\n' "$WHEEL" | grep -c 'vllm-')
echo "WHEEL_CANDIDATES=$N CHOSEN=$(basename "${WHEEL:-none}")"
[ "$N" = "1" ] || { echo "ABORT: expected exactly ONE eligible vLLM wheel, found $N"; exit 11; }
sha256sum "$WHEEL"

step "2. TOOLCHAIN -- nvcc is REQUIRED and its absence would read as a MODEL failure"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > "$RUN/apt.log" 2>&1; echo "RC[apt-get update]=$?"
apt-get install -y -qq python3 python3-venv python3-dev git curl ca-certificates \
  >> "$RUN/apt.log" 2>&1; echo "RC[apt-get install]=$?"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
  echo "RC[curl keyring]=$?"
  dpkg -i /tmp/cuda-keyring.deb > "$RUN/cuda.log" 2>&1; echo "RC[dpkg keyring]=$?"
  apt-get update -qq >> "$RUN/cuda.log" 2>&1; echo "RC[apt update cuda]=$?"
  apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcublas-dev-13-0 \
     cuda-nvrtc-dev-13-0 cuda-nvtx-13-0 cuda-profiler-api-13-0 libcurand-dev-13-0 \
     >> "$RUN/cuda.log" 2>&1
  echo "RC[apt install cuda]=$?"
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2; echo "RC[nvcc --version]=${PIPESTATUS[0]}"

step "3. VENV FROM THE PINNED WHEEL"
VENV_VER=$("$V/bin/python" -c "import vllm;print(vllm.__version__)" 2>/dev/null || echo none)
echo "VENV_EXISTING_VERSION=$VENV_VER"
case "$VENV_VER" in
  *555967922*) echo "REUSING the venv this worker already built"; SKIP_INSTALL=1 ;;
  *) SKIP_INSTALL=0; rm -rf "$V" ;;
esac
if [ "$SKIP_INSTALL" = "0" ]; then
  python3 -m venv "$V" > "$RUN/venv.log" 2>&1; echo "RC[venv]=$?"
  "$V/bin/pip" install -q --upgrade pip >> "$RUN/venv.log" 2>&1; echo "RC[pip upgrade]=$?"
  "$V/bin/pip" install -q torch==2.13.0 > "$RUN/pip_torch.log" 2>&1
  echo "RC[pip torch]=$?"; tail -3 "$RUN/pip_torch.log"
  # The staged wheel's name has SEVEN `-` parts; PEP 427 allows five, so pip
  # refuses it outright. Copy to a conforming name; do not rename the artifact
  # in place. (#1416, /workspace/oracle-vllm/README-WHEELS.md)
  WHEEL_OK=/tmp/vllm-0.1.dev1+g555967922-cp312-cp312-linux_aarch64.whl
  cp "$WHEEL" "$WHEEL_OK"; echo "RC[cp wheel to a PEP 427 name]=$?"
  "$V/bin/pip" install -q "$WHEEL_OK" > "$RUN/pip_wheel.log" 2>&1
  echo "RC[pip wheel]=$?"; tail -5 "$RUN/pip_wheel.log"
fi

step "4. IDENTITY, asserted from OUTSIDE any source tree"
cd /
"$V/bin/python" - <<'PY'
import vllm
print("vllm.__file__    =", vllm.__file__)
print("vllm.__version__ =", vllm.__version__)
assert "site-packages" in vllm.__file__, f"not an installed package: {vllm.__file__}"
assert "555967922" in vllm.__version__, f"WRONG COMMIT: {vllm.__version__}"
assert ".cu133" not in vllm.__version__, f"TOOLKIT THE DRIVER CANNOT RUN: {vllm.__version__}"
print("IDENTITY OK")
PY
IDENTITY_RC=$?
echo "IDENTITY_RC=$IDENTITY_RC"
[ "$IDENTITY_RC" -eq 0 ] || { echo "ABORT: the oracle did not install. INSTRUMENT failure, not a verdict about the model."; echo "ALL LOGS: $RUN"; exit 12; }
"$V/bin/python" -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo "RC[torch cuda]=$?"

step "5. BUILD THE PLUGIN FROM SOURCE, at a named revision"
rm -rf "$SRC"; mkdir -p "$SRC"
tar -xzf "$OUT/ggufplugin-src.tar.gz" -C "$SRC"; echo "RC[untar]=$?"
echo "PLUGIN_SHA_CLAIMED=$PLUGIN_SHA (the archive is git-archive of that object; the sha256 above is what binds it)"
cd "$SRC"
ls -1 vllm_gguf_plugin/csrc/gguf/
# ★ TORCH_CUDA_ARCH_LIST MUST BE SET, and this is measured rather than assumed.
# Run 20260903T010058Z left it unset on the theory that torch's cpp_extension
# derives the arch from the present device. It does not, or not for this
# capability: the wheel built and installed and the GGUF loaded (16.3 GiB,
# 745 s), and the FIRST quantized GEMM then died with
#   RuntimeError: CUDA error: no kernel image is available for execution on the
#   device
# inside torch.ops._C_gguf.ggml_mul_mat_a8 -- the PLUGIN's own kernel, built in
# that same job on that same box. A missing arch flag therefore presents as a
# model failure twelve minutes after the job looks healthy.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
[ -n "$CAP" ] || { echo "ABORT: cannot read compute_cap, so the arch flag would be a guess"; exit 14; }
export TORCH_CUDA_ARCH_LIST="${CAP}+PTX"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
time "$V/bin/pip" wheel . --no-build-isolation --no-deps -w /tmp/ggufplug-wheel \
  > "$RUN/plugin_build.log" 2>&1
PLUGIN_BUILD_RC=$?
echo "PLUGIN_BUILD_RC=$PLUGIN_BUILD_RC"
tail -25 "$RUN/plugin_build.log"
ls -l /tmp/ggufplug-wheel/ 2>/dev/null
if [ "$PLUGIN_BUILD_RC" -ne 0 ]; then
  echo "STOP: the plugin does not build here. That is the answer to step 1 and"
  echo "nothing below it could run. Full log: $RUN/plugin_build.log"
  cp "$RUN/plugin_build.log" "$RUN/PLUGIN_BUILD_FAILED.log"
  echo "DONE_MARKER_GGUFPLUGIN_JOB"
  exit 0
fi
# DURABLE FIRST. Everything after this can be killed at the ceiling; the wheel
# and this log survive on /workspace.
# Persist into the RUN dir, never into $OUT: two devices produce wheels with the
# SAME filename and different SASS, and one overwriting the other would leave a
# wheel nobody can attribute to a device.
cp -L /tmp/ggufplug-wheel/*.whl "$RUN/" ; echo "RC[persist plugin wheel]=$?"
sha256sum "$RUN"/vllm_gguf_plugin-*.whl
uname -m > "$RUN/built-on.arch"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader >> "$RUN/built-on.arch" 2>/dev/null
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" >> "$RUN/built-on.arch"
cat "$RUN/built-on.arch"
# ★ A FLAG IS NOT A POSTCONDITION. Ask the object what it carries.
# NOTE: this checks the WHEEL. The installed copy is checked again after the
# install below, because those two have already disagreed once.
# zipfile rather than unzip: the worker image is not guaranteed to carry unzip,
# and a missing tool would make this check silently not run.
SO=$("$V/bin/python" - <<'PYX'
import glob, os, zipfile
out = ""
for whl in glob.glob("/tmp/ggufplug-wheel/*.whl"):
    with zipfile.ZipFile(whl) as z:
        for n in z.namelist():
            if "_C_gguf" in n and n.endswith(".so"):
                os.makedirs("/tmp/sochk", exist_ok=True)
                z.extract(n, "/tmp/sochk")
                out = os.path.join("/tmp/sochk", n)
print(out)
PYX
)
echo "EXTENSION_SO=$SO"
if [ -n "$SO" ]; then
  cuobjdump --list-elf "$SO" > "$RUN/cubin-arch.txt" 2>&1
  echo "--- ELF sections the extension actually carries ---"
  cat "$RUN/cubin-arch.txt"
  SMWANT="sm_$(echo "$CAP" | tr -d '.')"
  if grep -q "$SMWANT" "$RUN/cubin-arch.txt"; then
    echo "CUBIN_ARCH_OK: $SMWANT present"
  else
    echo "CUBIN_ARCH_MISSING: $SMWANT is NOT in the object; a launch will fail with"
    echo "'no kernel image is available for execution on the device'."
  fi
else
  echo "CUBIN_ARCH_UNCHECKED: could not extract the extension from the wheel"
fi
cd /
# The plugin's own runtime dependencies, INSTALLED EXPLICITLY. Run
# 20260903T004110Z installed the wheel with --no-deps and the job then died on
# `ModuleNotFoundError: No module named 'gguf'` at the registration step -- an
# INSTRUMENT failure that read like a plugin defect. pyproject.toml declares
# gguf>=0.17.0, huggingface_hub>=1.26.0, vllm and torch; the last two are
# already installed from the pinned wheel and must NOT be re-resolved, which is
# why they are named here rather than dropping --no-deps.
"$V/bin/pip" install -q "gguf>=0.17.0" "huggingface_hub>=1.26.0" > "$RUN/plugin_deps.log" 2>&1
echo "RC[pip install plugin deps]=$?"; tail -3 "$RUN/plugin_deps.log"
# ★ --force-reinstall IS LOAD-BEARING. The wheel version is always 0.0.5, so a
# plain install into a REUSED venv prints "Requirement already satisfied" and
# leaves the PREVIOUS run's extension in place. Run 20260903T011701Z built a
# wheel whose object carried sm_110 (CUBIN_ARCH_OK) and still died in
# ggml_mul_mat_a8 with "no kernel image", because the .so being called was the
# arch-less one run 20260903T010058Z had installed.
"$V/bin/pip" install --force-reinstall --no-deps /tmp/ggufplug-wheel/*.whl > "$RUN/plugin_install.log" 2>&1
echo "RC[pip install plugin]=$?"; tail -5 "$RUN/plugin_install.log"
"$V/bin/python" -c "import gguf, huggingface_hub; print('DEPS OK gguf', gguf.__file__)"
echo "RC[deps import]=$?"

# ★ THE OBJECT THAT WILL ACTUALLY BE CALLED. Everything above inspected the
# wheel; this inspects site-packages, which is what torch.ops._C_gguf loads.
INST_SO=$("$V/bin/python" -c "import vllm_gguf_plugin._C_gguf as c; print(c.__file__)" 2>/dev/null)
echo "INSTALLED_SO=$INST_SO"
if [ -n "$INST_SO" ]; then
  cuobjdump --list-elf "$INST_SO" > "$RUN/cubin-arch-installed.txt" 2>&1
  cat "$RUN/cubin-arch-installed.txt"
  SMWANT="sm_$(echo "$CAP" | tr -d '.')"
  if grep -q "$SMWANT" "$RUN/cubin-arch-installed.txt"; then
    echo "INSTALLED_CUBIN_ARCH_OK: $SMWANT present in the object that will be called"
  else
    echo "ABORT: the INSTALLED extension carries no $SMWANT. Every launch below"
    echo "would fail with 'no kernel image is available for execution on the"
    echo "device', which reads like a model failure and is not one."
    echo "DONE_MARKER_GGUFPLUGIN_JOB"
    exit 15
  fi
fi

step "6. THE PLUGIN IS REGISTERED, and its CUDA extension IMPORTS"
"$V/bin/python" - <<'PY'
import importlib.metadata as md
import vllm_gguf_plugin
print("PLUGIN_FILE =", vllm_gguf_plugin.__file__)
import vllm_gguf_plugin._C_gguf as c
print("PLUGIN_C_EXT =", c.__file__)
eps = [e for e in md.entry_points(group="vllm.general_plugins")]
print("GENERAL_PLUGIN_ENTRY_POINTS =", [(e.name, e.value) for e in eps])
assert any(e.name == "gguf" for e in eps), "the gguf entry point is not registered"
import torch  # noqa: F401
print("HAS_GGUF_OPS =", [n for n in dir(torch.ops._C_gguf)][:12] if hasattr(torch.ops, "_C_gguf") else "no torch.ops._C_gguf namespace")
print("REGISTRATION OK")
PY
echo "REGISTRATION_RC=$?"

step "7. GENERATION (watchdogged: this box has OOM-REBOOTED)"
echo "FLOOR_MB=$FLOOR_MB" > "$RUN/watchdog.floor"
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A $(awk '/^MemTotal/{print int($2/1024)}' /proc/meminfo)" >> "$RUN/mem.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      G=$(cat /tmp/ggufplug.pgid 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing process GROUP $G to save the box"
      [ -n "$G" ] && kill -9 -- "-$G" 2>/dev/null
    fi
    sleep 1
  done ) > "$RUN/watchdog.log" 2>&1 &
WD=$!

MODEL="$GGUF" TOK="$TOK" MMPROJ="$MMPROJ" OUT_JSON="$RUN/tokens.json" GMU=0.30 MAXLEN=2048 \
  setsid timeout 2700 "$V/bin/python" "$OUT/gen.py" > "$RUN/gen.out" 2>&1 &
GPID=$!
echo "$GPID" > /tmp/ggufplug.pgid
wait "$GPID"; GEN_RC=$?
rm -f /tmp/ggufplug.pgid
echo "GEN_RC=$GEN_RC"
kill -9 -- "-$GPID" 2>/dev/null
kill -9 "$WD" 2>/dev/null
sleep 5
echo "--- postcondition ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "MemAvailable now: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"

step "8. WHAT IT SAID"
grep -E "VLLM_FILE|VLLM_VERSION|PLUGIN_FILE|PLUGIN_C_EXT|PROMPT_LENS|PROMPT_IDS|ENGINE_UP|GEN_IDS|GEN_TEXT|GEN_LEN|WROTE|DONE_MARKER" "$RUN/gen.out"
echo "--- tail, in case it died ---"
tail -40 "$RUN/gen.out"
tail -3 "$RUN/watchdog.log" 2>/dev/null
awk 'NF>=3{if(min==""||$2<min)min=$2; tot=$3} END{if(min!="")print "MemTotal_MB="tot" minMemAvailable_MB="min" peakUsed_MB="tot-min" samples="NR}' "$RUN/mem.samples" 2>/dev/null
echo "ALL LOGS: $RUN"
echo "DONE_MARKER_GGUFPLUGIN_JOB"
