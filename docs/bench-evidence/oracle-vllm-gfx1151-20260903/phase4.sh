#!/usr/bin/env bash
# #2740 -- PHASE 4: close the two gaps phase 3 left.
#   (a) the COMPILED leg is n=1, and it is the leg closest to vLLM's default.
#   (b) the installed plugin object's gfx targets were never read: phase 3
#       captured a ROCm deprecation WARNING into INST_SO with 2>&1, so the -f
#       test failed and the check silently did not run.
#   (c) the registry and second-custom-op probes, whose phase-3 failures were
#       both wrong API calls of mine.
#
# NOTHING IN THIS FILE IS EDITED WHILE IT RUNS. Phase 3 was edited mid-flight
# and bash, which reads a script by byte offset as it executes, hit
# `syntax error near unexpected token fi` in a file that parses cleanly. Every
# leg had already finished, but the postcondition section never ran.
#
# CORRECTNESS ONLY. No timing is taken.
set -uo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
W=/workspace/vllm-gfx1151-2740
LOCAL=/tmp/vllm-gfx1151-2740
VENV=$LOCAL/venv
PY=$VENV/bin/python
CKPT=/workspace/ckpt/qwen38-27b-q4km
GGUF=$CKPT/Qwen3.8-27B-Q4_K_M.gguf
MMPROJ=/workspace/ggufplugin/mmproj-BF16.gguf
TOKDIR=/workspace/ckpt/qwen3.8-27b-hf
FLOOR_MB=6000
TAG="${TAG:-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT=$W/out/$TAG
mkdir -p "$OUT" || exit 90
exec > >(tee -a "$LOCAL/phase4-$TAG.log") 2>&1
( while true; do cp -f "$LOCAL/phase4-$TAG.log" "$OUT/phase4.log" 2>/dev/null; sleep 20; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null' EXIT
step() { echo; echo "===== $* ====="; date -u +%FT%TZ; }

step "0. identity"
echo "hostname=$(hostname) boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "RC_JOB_ID=${RC_JOB_ID:-UNSET} RC_DEVICE=${RC_DEVICE:-UNSET}"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-UNSET}  <- must be UNSET"
"$PY" -c "import vllm,torch;print('VLLM',vllm.__version__);print('TORCH',torch.__version__,'hip',torch.version.hip);print('ARCH',torch.cuda.get_device_properties(0).gcnArchName)"
free -g | head -2

step "1. the INSTALLED plugin object's gfx targets"
INST_SO=$("$PY" -c "import vllm_gguf_plugin._C_gguf as c; print(c.__file__)" 2>/dev/null | tail -1)
echo "INSTALLED_SO=$INST_SO"
if [ -f "$INST_SO" ]; then
  sha256sum "$INST_SO"
  llvm-objdump --offloading "$INST_SO" > "$LOCAL/plugin-offload.txt" 2>&1
  echo "plugin_offload_rc=$?  lines=$(wc -l < "$LOCAL/plugin-offload.txt")"
  echo "-- DISTINCT offload targets, verbatim --"
  grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-z:+-]+' "$LOCAL/plugin-offload.txt" | sort | uniq -c
  echo "PLUGIN_GFX1151_BUNDLES=$(grep -c 'amdhsa--gfx1151$' "$LOCAL/plugin-offload.txt")"
  cp -f "$LOCAL/plugin-offload.txt" "$OUT/plugin-offload.txt" 2>/dev/null
  if grep -q 'amdhsa--gfx1151' "$LOCAL/plugin-offload.txt"; then
    echo "INSTALLED_PLUGIN_ARCH_OK: gfx1151 is in the object that was called"
  else
    echo "INSTALLED_PLUGIN_ARCH_MISSING"
  fi
  echo "-- the same instrument on an object that must NOT carry it, as a control --"
  llvm-objdump --offloading "$VENV/lib/python3.12/site-packages/torch/lib/libc10.so" 2>&1 | head -5
else
  echo "INSTALLED_SO not a file; the check did not run"
fi

step "2. the two probes phase 3 called wrongly"
"$PY" "$W/probe_ext.py"; echo "probe_ext_rc=$?"
"$PY" "$W/probe_registry.py"; echo "probe_registry_rc=$?"

step "3. a SECOND compiled leg -- the phase-3 compiled result is n=1"
( while true; do
    A=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    echo "$(date +%s) $A" >> "$LOCAL/mem4.samples"
    if [ "$A" -lt "$FLOOR_MB" ]; then
      G=$(cat "$LOCAL/gen4.pgid" 2>/dev/null)
      echo "WATCHDOG: MemAvailable ${A}MB < ${FLOOR_MB}MB -- killing group $G"
      [ -n "$G" ] && kill -9 -- "-$G" 2>/dev/null
    fi
    sleep 2
  done ) > "$LOCAL/watchdog4.log" 2>&1 &
WD=$!
run_leg() {
  local name=$1; shift
  echo; echo "--- LEG $name ---"; date -u +%FT%TZ
  env "$@" OUT_JSON="$LOCAL/tokens-$name.json" \
    setsid timeout 3000 "$PY" "$W/gen_rocm.py" > "$LOCAL/gen-$name.out" 2>&1 &
  local p=$!; echo "$p" > "$LOCAL/gen4.pgid"
  wait "$p"; local r=$?
  rm -f "$LOCAL/gen4.pgid"; kill -9 -- "-$p" 2>/dev/null
  echo "GEN_RC[$name]=$r"
  cp -f "$LOCAL/gen-$name.out" "$OUT/gen-$name.out" 2>/dev/null
  cp -f "$LOCAL/tokens-$name.json" "$OUT/tokens-$name.json" 2>/dev/null
  grep -E "VLLM_VERSION|DEVICE |ON_GFX1151|ENGINE_UP|GEN_IDS|GEN_LEN|WROTE|DONE_MARKER" "$LOCAL/gen-$name.out"
  echo "--- tail of LEG $name ---"; tail -20 "$LOCAL/gen-$name.out"
}
run_leg gguf-compiled-2 MODEL="$GGUF" TOK="$TOKDIR" MMPROJ="$MMPROJ" QUANT=gguf \
        GMU=0.60 MAXLEN=2048 EAGER=0
kill -9 "$WD" 2>/dev/null

step "4. is the compiled leg self-reproducible, and does it differ from eager?"
"$PY" - <<'PYEOF'
import json, os
L = "/tmp/vllm-gfx1151-2740"
def ids(n):
    p = f"{L}/tokens-{n}.json"
    return [r["gen_ids"] for r in json.load(open(p))["records"]] if os.path.exists(p) else None
e1, e2 = ids("gguf-eager"), ids("gguf-eager-2")
c1, c2 = ids("gguf-compiled"), ids("gguf-compiled-2")
def cmp(a, b, na, nb):
    if a is None or b is None:
        print(f"{na} vs {nb}: MISSING"); return
    print(f"{na}_EQ_{nb} = {a == b}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            k = next(j for j, (p, q) in enumerate(zip(x, y)) if p != q)
            print(f"   prompt {i}: first diff at {k}  {na}={x[k]}  {nb}={y[k]}")
cmp(e1, e2, "EAGER1", "EAGER2")
cmp(c1, c2, "COMPILED1", "COMPILED2")
cmp(e1, c1, "EAGER1", "COMPILED1")
PYEOF
echo "reproducibility_rc=$?"

step "5. postcondition"
free -g | head -2
awk '{if(min==""||$2<min)min=$2} END{print "minMemAvailable_MB="min" samples="NR}' "$LOCAL/mem4.samples" 2>/dev/null
tail -3 "$LOCAL/watchdog4.log" 2>/dev/null
for f in "$LOCAL"/tokens-*.json; do [ -f "$f" ] && sha256sum "$f"; done
cp -f "$LOCAL"/tokens-*.json "$OUT/" 2>/dev/null
cp -f "$LOCAL/phase4-$TAG.log" "$OUT/phase4.log" 2>/dev/null
echo "OUT=$OUT"
echo "=== PHASE4 DONE ==="
