#!/bin/bash
# MODEL-MM-QWEN4-EXP -- WAVE 3: M1c only.
# Wave 1's M1 was a semantic no-op; wave 2's M1b did not COMPILE (it dropped the
# last use of `nwave` into -Werror=unused-variable), so it measured nothing
# either and was reported VOID rather than as a survivor. This one keeps `nwave`
# live and still collapses the cross-wave fold to wave 0's partial.
set -u
FIX=63925eb7f417882c0b682ab850d9135873978fab
KERN=src/vt/rocm/rocm_qwen4_exp.hip
ROOT=/tmp/vllmcpp-hcnorm3-$$
OUT=/workspace/q4exp-hcnorm/w3-$(date -u +%Y%m%dT%H%M%SZ)
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
trap 'rm -rf "$ROOT" 2>/dev/null' EXIT INT TERM
ulimit -c 0
export LD_LIBRARY_PATH="$(dirname "$(find /opt -name 'libamdhip64.so.7' 2>/dev/null|head -1)")":/opt/rocm/lib
say CLONE; git clone -q --no-checkout https://github.com/mudler/vllm.cpp "$ROOT" || exit 90
cd "$ROOT" || exit 90; git fetch -q origin "$FIX"; git checkout -q "$FIX" || exit 91
git submodule update -q --init --recursive 2>&1|tail -1
cmake -S "$ROOT" -B "$ROOT/b" -G Ninja -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH=/opt/rocm -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 2>&1 | grep -i "ROCm backend: ENABLED"
BINR="$ROOT/b/tests/test_qwen4_exp_rocm_reductions"
bld(){ ninja -C "$ROOT/b" -j 4 test_qwen4_exp_rocm_reductions > /tmp/b3.log 2>&1; }
T0=$SECONDS; bld; echo "build rc=$? in $((SECONDS-T0))s"
mkdir -p "$OUT" || exit 92
FIXMD5=$(md5sum $BINR|cut -c1-12); echo "unmutated md5=$FIXMD5"
SEL="$ROOT/scripts/run-doctest-selected.sh"
run(){ "$SEL" "$BINR" -tc='*ROCM W7*' > "$OUT/$1.log" 2>&1; echo "$1 rc=$?"
  grep -E '^\[MEASURED\]|test cases:|assertions:|Status:|selected=' "$OUT/$1.log" | sed "s/^/$1  /"; return 0; }
run FIX_control

say "MUTATION M1c :: collapse the cross-wave fold to wave 0's partial"
python3 - "$KERN" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="      float v = (lane < nwave) ? s_part[lane] : 0.0f;"
new="      float v = (lane < nwave && lane == 0) ? s_part[lane] : 0.0f;"
assert old in s, "anchor missing"
open(p,'w').write(s.replace(old,new,1)); print("M1c applied")
PY
if ! bld; then echo "M1c BUILD FAILED -- VOID"; tail -12 /tmp/b3.log; else
  M=$(md5sum $BINR|cut -c1-12); echo "M1c md5=$M (unmutated=$FIXMD5)"
  [ "$M" = "$FIXMD5" ] && echo "M1c BINARY UNCHANGED -- MEASURED NOTHING"
  run M1c_battery
fi
git checkout "$FIX" -- "$KERN"; bld
echo "restored md5=$(md5sum $BINR|cut -c1-12)"; git diff --exit-code -- "$KERN" >/dev/null 2>&1; echo "tree-clean rc=$?"
run FIX_recheck
say ARCHIVE; ls -la "$OUT"
echo "=====HCN3_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====HCN3_TAR_B64_END====="
say "DONE $OUT"
