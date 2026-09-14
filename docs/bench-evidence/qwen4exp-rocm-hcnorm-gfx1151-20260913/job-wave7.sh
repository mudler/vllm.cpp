#!/bin/bash
# MODEL-MM-QWEN4-EXP -- WAVE 7: the F1 red-first AND the declared gate, re-run on the POST-MERGE head 0515493e8 (origin/main 7588c1e6f). Wave 6 measured 7b737f882, which is now a stale base. Wave 5 VOIDED it -- the poison orphaned `eps` and -Wextra -Werror refused the build. This one keeps `eps` live.
#  F1 red-then-green: a kernel mutation that poisons the grouped norm to all-NaN
#  must leave the PRE-repair case 5 GREEN and the POST-repair case 5 RED.
#  Then the full declared gate on the repaired head.
set -u
HEAD_SHA=0515493e8589dcefdfb63b78e686609f8929c6e8
PRE_SHA=a449eca85          # parent of the F1 commit: pre-repair test files
KERN=src/vt/rocm/rocm_qwen4_exp.hip
T1=tests/vllm/models/test_qwen4_exp_rocm_reductions.cpp
T2=tests/vllm/models/qwen4_exp_hc_synth.h
ROOT=/tmp/vllmcpp-hcnorm-w7-$$
OUT=/workspace/q4exp-hcnorm/w7-$(date -u +%Y%m%dT%H%M%SZ)
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
trap 'rm -rf "$ROOT" 2>/dev/null' EXIT INT TERM
ulimit -c 0
export LD_LIBRARY_PATH="$(dirname "$(find /opt -name 'libamdhip64.so.7' 2>/dev/null|head -1)")":/opt/rocm/lib
echo "HOST=$(hostname) JOB=${RC_JOB_ID:-none}"; df -h /tmp | tail -1

say "CLONE $ROOT"
git clone -q --no-checkout https://github.com/mudler/vllm.cpp "$ROOT" || { echo FATAL clone; exit 90; }
cd "$ROOT" || exit 90
git fetch -q origin "$HEAD_SHA" 2>&1 | tail -2
git checkout -q "$HEAD_SHA" || { echo FATAL checkout; exit 91; }
echo "HEAD=$(git rev-parse HEAD)"
git submodule update -q --init --recursive 2>&1 | tail -2
mkdir -p "$OUT" || { echo "FATAL cannot write $OUT"; exit 92; }

cfg(){ cmake -S "$ROOT" -B "$ROOT/b" -G Ninja -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DROCM_PATH=/opt/rocm -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 2>&1 | tail -3; }
bld(){ ninja -C "$ROOT/b" -j 4 test_backend_cross_device test_qwen4_exp_rocm_reductions > /tmp/w7b.log 2>&1; }
SEL="$ROOT/scripts/run-doctest-selected.sh"
BINR="$ROOT/b/tests/test_qwen4_exp_rocm_reductions"
BINX="$ROOT/b/tests/test_backend_cross_device"
run(){ local tag="$1" bin="$2" filt="$3"
  say "RUN $tag :: $(basename $bin) -tc='$filt'  md5=$(md5sum $bin|cut -c1-12)"
  "$SEL" "$bin" -tc="$filt" > "$OUT/$tag.log" 2>&1; local rc=$?
  grep -E '^\[MEASURED\]|^\[SKIP\]|test cases:|assertions:|Status:|selected=' "$OUT/$tag.log" | sed "s/^/$tag  /"
  echo "$tag rc=$rc"
  grep -iE 'illegal memory access|HSA_STATUS_ERROR|Memory access fault|core dumped' "$OUT/$tag.log" | head -3 | sed "s/^/$tag  CRASHSIG /"
  return 0; }

say "CONFIGURE + BUILD (repaired head)"; cfg; T0=$SECONDS; bld; echo "build rc=$? in $((SECONDS-T0))s"
[ -x "$BINR" ] || { echo FATAL no rocm reductions binary; tail -20 /tmp/w7b.log; exit 93; }
CTRLMD5=$(md5sum $BINR|cut -c1-12); echo "repaired-control md5=$CTRLMD5"

# ══ F1 RED-FIRST ════════════════════════════════════════════════════════════
MAGF='*MAGNITUDE-SEPARATED*'
say "F1 step 0 :: repaired case 5 on the CLEAN kernel (must PASS)"
run F1_0_clean_repaired "$BINR" "$MAGF"

say "F1 mutation :: poison HcGroupedNormKernel's reciprocal to NaN"
python3 - "$KERN" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old="        s_r = __frcp_rn(sqrtf(__fadd_rn(__fdiv_rn(v, static_cast<float>(H)), eps)));"
new="        s_r = __fmul_rn(__fadd_rn(__fdiv_rn(v, static_cast<float>(H)), eps), __builtin_nanf(\"\"));  // NaN POISON (F1 red-first): every operand of the real line stays LIVE, so -Wextra -Werror cannot refuse the build and void the mutation"
assert old in s, "ANCHOR MISSING"
open(p,'w').write(s.replace(old,new,1)); print("NaN poison applied")
PY
[ $? -eq 0 ] || { echo "FATAL poison anchor"; exit 94; }

say "F1 step 1 :: PRE-repair test files + poisoned kernel (EXPECT GREEN = the defect)"
git checkout "$PRE_SHA" -- "$T1" "$T2" || { echo FATAL pre checkout; exit 95; }
if ! bld; then echo "F1_1 BUILD FAILED -- VOID"; tail -20 /tmp/w7b.log; else
  M=$(md5sum $BINR|cut -c1-12); echo "F1_1 md5=$M (clean=$CTRLMD5)"
  [ "$M" = "$CTRLMD5" ] && echo "F1_1 BINARY UNCHANGED -- MEASURED NOTHING"
  run F1_1_prerepair_poisoned "$BINR" "$MAGF"
fi

say "F1 step 2 :: REPAIRED test files + the SAME poisoned kernel (EXPECT RED)"
git checkout "$HEAD_SHA" -- "$T1" "$T2" || { echo FATAL head checkout; exit 96; }
if ! bld; then echo "F1_2 BUILD FAILED -- VOID"; tail -20 /tmp/w7b.log; else
  M=$(md5sum $BINR|cut -c1-12); echo "F1_2 md5=$M"
  run F1_2_repaired_poisoned "$BINR" "$MAGF"
fi

say "F1 step 3 :: does the poison also red case 4's separation probe?"
run F1_3_repaired_poisoned_sep "$BINR" '*broadcast cannot hide*'

say "RESTORE the kernel"
git checkout "$HEAD_SHA" -- "$KERN"
git diff --exit-code >/dev/null 2>&1; echo "tree-clean rc=$?"
bld; echo "restored-build rc=$?"
RMD5=$(md5sum $BINR|cut -c1-12); echo "restored md5=$RMD5 (clean=$CTRLMD5)"
[ "$RMD5" = "$CTRLMD5" ] && echo "RESTORE BYTE-IDENTICAL" || echo "RESTORE DIFFERS -- INVESTIGATE"

# ══ THE DECLARED GATE ═══════════════════════════════════════════════════════
run GATE_battery   "$BINR" '*ROCM W7*'
run GATE_crossdev  "$BINX" '*'
run GATE_dsa       "$BINX" '*DSA*'

say ARCHIVE; ls -la "$OUT"
echo "=====W7_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====W7_TAR_B64_END====="
say "DONE $OUT"
