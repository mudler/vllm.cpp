#!/bin/bash
# MODEL-MM-QWEN4-EXP / ISSUE-LOCAL-01M2CJXQMV9R9JGRSKZMW4W21F -- WAVE 2
# (a) M1 and M5 REDONE: wave 1's versions were NO-OPS and their survival measured
#     nothing about the gate. (b) decode tok/s A/B, first generation excluded.
# (c) rocprofv3 kernel ranking BEFORE and AFTER, same tool, same workload.
set -u
FIX=63925eb7f417882c0b682ab850d9135873978fab
BASE=60990ee784101f74f6d1775575e9e89dfb26f73a
KERN=src/vt/rocm/rocm_qwen4_exp.hip
ROOT=/tmp/vllmcpp-hcnorm2-$$
OUT=/workspace/q4exp-hcnorm/w2-$(date -u +%Y%m%dT%H%M%SZ)
MODEL=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
( while sleep 300; do kill -0 $$ 2>/dev/null||exit 0; echo "### hb $(date -u +%H:%M:%S) free=$(df -BG --output=avail /tmp|tail -1)"; done ) & HB=$!
trap 'kill -9 $HB 2>/dev/null; rm -rf "$ROOT" 2>/dev/null' EXIT INT TERM
ulimit -c 0

say IDENTITY; uname -m; nproc; free -g|head -2; df -h /tmp|tail -1; cat /proc/sys/kernel/random/boot_id
HIPDIR=$(dirname "$(find /opt -name 'libamdhip64.so.7' 2>/dev/null | head -1)")
export LD_LIBRARY_PATH="$HIPDIR:/opt/rocm/lib"
echo "HIPDIR=$HIPDIR"
say "ARTIFACT"; ls -la "$MODEL" 2>&1 | tail -2 || echo "MODEL ABSENT at $MODEL"

say "CLONE + BUILD $ROOT"
git clone -q --no-checkout https://github.com/mudler/vllm.cpp "$ROOT" || { echo FATAL clone; exit 90; }
cd "$ROOT" || exit 90
git fetch -q origin "$FIX" "$BASE"; git checkout -q "$FIX" || exit 91
git submodule update -q --init --recursive 2>&1 | tail -1
echo "HEAD=$(git rev-parse HEAD)"
cmake -S "$ROOT" -B "$ROOT/b" -G Ninja -DVLLM_CPP_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  -DROCM_PATH=/opt/rocm -DVLLM_CPP_HIP_ARCHITECTURES=gfx1151 2>&1 | grep -iE "ROCm backend|Error" | head -3
bld(){ ninja -C "$ROOT/b" -j 4 test_qwen4_exp_rocm_reductions vllm-cli > /tmp/b.log 2>&1; local r=$?; [ $r -ne 0 ] && tail -12 /tmp/b.log; return $r; }
T0=$SECONDS; bld; echo "build rc=$? in $((SECONDS-T0))s"
BINR="$ROOT/b/tests/test_qwen4_exp_rocm_reductions"
CLI=$(find "$ROOT/b" -name vllm-cli -type f | head -1); echo "CLI=$CLI"
export LD_LIBRARY_PATH="$ROOT/b:$ROOT/b/bin:$LD_LIBRARY_PATH"
mkdir -p "$OUT" /tmp/rpt || exit 92
SEL="$ROOT/scripts/run-doctest-selected.sh"
FIXMD5=$(md5sum $BINR|cut -c1-12); echo "FIX battery binary md5=$FIXMD5"

run(){ local tag="$1" filt="$2"
  say "RUN $tag :: -tc='$filt'  md5=$(md5sum $BINR|cut -c1-12)"
  "$SEL" "$BINR" -tc="$filt" > "$OUT/$tag.log" 2>&1; echo "$tag rc=$?"
  grep -E '^\[MEASURED\]|test cases:|assertions:|Status:|selected=' "$OUT/$tag.log" | sed "s/^/$tag  /"
  return 0; }

# ── (a) M1 and M5, REDONE ───────────────────────────────────────────────────
# WAVE 1's M1 wrote `s_part[0] = part` from thread 0, which is lane 0 of wave 0
# and had ALREADY written that slot -- a no-op. Its M5 broadcast `s_r` across a
# wave in which every lane already held the same value -- also a no-op. Both
# SURVIVED because the mutation did nothing, not because the gate is weak.
mut(){ local tag="$1" py="$2"
  say "MUTATION $tag"
  python3 - "$KERN" <<PY || { echo "$tag ANCHOR MISSING -- VOID"; git checkout "$FIX" -- "$KERN"; return 0; }
import sys
p=sys.argv[1]; s=open(p).read()
$py
PY
  bld || { echo "$tag BUILD FAILED -- VOID"; git checkout "$FIX" -- "$KERN"; bld; return 0; }
  local m=$(md5sum $BINR|cut -c1-12)
  echo "$tag binary md5=$m  (unmutated=$FIXMD5)"
  if [ "$m" = "$FIXMD5" ]; then echo "$tag BINARY UNCHANGED -- THIS MUTATION MEASURED NOTHING"; fi
  run ${tag}_battery '*ROCM W7*'
  git checkout "$FIX" -- "$KERN"; bld
  echo "$tag restored md5=$(md5sum $BINR|cut -c1-12)"; git diff --exit-code -- "$KERN" >/dev/null 2>&1; echo "$tag tree-clean rc=$?"
  return 0; }

# M1b: COLLAPSE the cross-wave fold -- wave 0's partial becomes the whole sum.
mut M1b 'old="""      float v = (lane < nwave) ? s_part[lane] : 0.0f;"""
new="""      float v = (lane == 0) ? s_part[0] : 0.0f;"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

# M5b: BROADCAST group 0 -- the reduction reads token t group 0 for every group,
# which is exactly the defect the separation probe is aimed at.
mut M5b 'old="""      const float v = LoadAt(hyper, hyper_tag, base + h);
      part = __fadd_rn(part, __fmul_rn(v, v));"""
new="""      const float v = LoadAt(hyper, hyper_tag, t * flat + h);
      part = __fadd_rn(part, __fmul_rn(v, v));"""
assert old in s
open(p,"w").write(s.replace(old,new,1))'

say "RE-VERIFY the FIX after both restores"; run FIX_battery_recheck '*ROCM W7*'

# ── (b) decode tok/s, FIX vs BASE, interleaved, first generation excluded ────
[ -f "$MODEL" ] || { echo "MODEL ABSENT -- speed and profile arms SKIPPED, nothing is claimed"; exit 0; }
[ -n "$CLI" ] || { echo "NO vllm-cli -- speed arm SKIPPED"; exit 0; }
sha256sum "$MODEL" | sed 's/^/gguf /' &
decode(){ local tag="$1"
  say "DECODE $tag  cli_md5=$(md5sum $CLI|cut -c1-12)"
  timeout 2400 "$CLI" --model "$MODEL" --prompt 'The capital of France is' \
    --max-tokens 64 --temperature 0 --max-num-seqs 1 --repeat 4 \
    > "$OUT/$tag.log" 2>&1 </dev/null; echo "$tag rc=$?"
  grep -a "tok_s=" "$OUT/$tag.log" | sed "s/^/$tag  /"
  grep -aiE "illegal memory access|Memory access fault|HSA_STATUS_ERROR|core dumped" "$OUT/$tag.log" | head -2 | sed "s/^/$tag  CRASHSIG /"
  return 0; }
bldcli(){ ninja -C "$ROOT/b" -j 4 vllm-cli >/tmp/bc.log 2>&1 || tail -8 /tmp/bc.log; }

decode FIX_speed_r1
git checkout "$BASE" -- "$KERN"; bldcli; echo "BASE cli md5=$(md5sum $CLI|cut -c1-12)"
decode BASE_speed_r1
git checkout "$FIX" -- "$KERN"; bldcli
decode FIX_speed_r2
git checkout "$BASE" -- "$KERN"; bldcli
decode BASE_speed_r2

# ── (c) rocprofv3 ranking, BOTH arms, same tool, same workload ───────────────
# cwd=/tmp and ROCPROF_TMPDIR are load-bearing: from `/` the spill never reads
# back and the profiler aborts with mmap(len=0) EINVAL. Never pipe into `head`.
SLICE="$ROOT/docs/bench-evidence/rocm-kernel-attrib-gfx1151-20260913/slice-decode-window.py"
prof(){ local tag="$1"
  say "PROFILE $tag  cli_md5=$(md5sum $CLI|cut -c1-12)"
  rm -rf /tmp/cap-$tag; mkdir -p /tmp/cap-$tag
  ( cd /tmp && ROCPROF_TMPDIR=/tmp/rpt timeout -s KILL 2400 /opt/rocm/bin/rocprofv3 \
      --kernel-trace --output-format csv --output-directory /tmp/cap-$tag \
      -- "$CLI" --model "$MODEL" --prompt 'The capital of France is' \
         --max-tokens 64 --temperature 0 --max-num-seqs 1 --repeat 3 ) \
      > "$OUT/$tag-cap.log" 2>&1 </dev/null
  echo "$tag capture_rc=$?"
  grep -a "tok_s=" "$OUT/$tag-cap.log" | sed "s/^/$tag  /"
  echo -n "$tag swap_warnings="; grep -ac "swap" "$OUT/$tag-cap.log"
  local KT=$(find /tmp/cap-$tag -name '*kernel_trace.csv' | head -1)
  if [ -z "$KT" ] || [ "$(stat -c%s "$KT" 2>/dev/null || echo 0)" -lt 1000 ]; then echo "$tag NO_USABLE_TRACE"; return 0; fi
  echo "$tag KT_ROWS=$(wc -l < "$KT")"
  gzip -c "$KT" > "$OUT/$tag-kernel-trace.csv.gz"
  python3 "$SLICE" "$KT" /tmp/$tag-window.csv > "$OUT/$tag-slice.log" 2>&1
  echo "$tag slice_rc=$?"; grep -E "selected_stretch|wrote" "$OUT/$tag-slice.log" | sed "s/^/$tag  /"
  python3 "$ROOT/scripts/rocm-rank-kernels.py" /tmp/$tag-window.csv --sampler ArgmaxK --skip-first 0 --top 12 \
      > "$OUT/$tag-ranked.txt" 2>&1
  echo "$tag rank_rc=$?"; sed "s/^/$tag  /" "$OUT/$tag-ranked.txt" | head -25
  return 0; }

git checkout "$FIX" -- "$KERN"; bldcli; prof FIX_prof
git checkout "$BASE" -- "$KERN"; bldcli; prof BASE_prof
git checkout "$FIX" -- .; bldcli

wait
say ARCHIVE; du -sh "$OUT"; ls -la "$OUT"
echo "=====HCN2_TAR_B64_BEGIN====="; tar czf - -C "$OUT" --exclude='*kernel-trace.csv.gz' . | base64 -w 200; echo "=====HCN2_TAR_B64_END====="
say "DONE $OUT"
