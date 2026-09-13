#!/bin/bash
# MODEL-MM-QWEN4-EXP -- WAVE 4: re-rank the two archived wave-2 traces DEEPER.
# `HcGroupedNormKernel` fell out of the top 12 on the FIX arm, so its NEW SHARE
# was not printed. A share derived by subtracting two totals is arithmetic, not a
# reading; this reads it. No build, no GPU work, no model load.
set -u
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
ulimit -c 0
W2=$(ls -d /workspace/q4exp-hcnorm/w2-* 2>/dev/null | tail -1)
say "SOURCE $W2"; ls -la "$W2" | head -20
[ -n "$W2" ] || { echo FATAL no wave2 dir; exit 90; }
R=/tmp/rank4-$$; mkdir -p $R; trap 'rm -rf $R' EXIT
git clone -q --depth 1 https://github.com/mudler/vllm.cpp $R/src || exit 91
SLICE=$R/src/docs/bench-evidence/rocm-kernel-attrib-gfx1151-20260913/slice-decode-window.py
RANK=$R/src/scripts/rocm-rank-kernels.py
OUT=/workspace/q4exp-hcnorm/w4-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
for tag in BASE_prof FIX_prof; do
  KTZ="$W2/$tag-kernel-trace.csv.gz"
  say "RERANK $tag :: $KTZ"
  [ -f "$KTZ" ] || { echo "$tag TRACE ABSENT"; continue; }
  echo "$tag sha256=$(sha256sum "$KTZ" | cut -c1-16)"
  gunzip -c "$KTZ" > $R/$tag.csv
  # The #3040 bound needs the run's OWN swap-warning count. Read it from the
  # capture log rather than leaving the tool to say UNKNOWN a second time.
  #
  # THIS LINE IS NOT WHAT RAN, AND THE DIFFERENCE IS A BUG THIS REPOSITORY HAS A
  # GATE FOR. The lease executed
  #   SW=$(grep -aic "swap" "$W2/$tag-cap.log" 2>/dev/null || echo 0)
  # and `tests/scripts/test_ltx25_ab_memwatch.py` refuses it by name: `grep -c`
  # prints its count AND exits 1 when the count is zero, so the `|| echo 0`
  # fires IN ADDITION to the "0" grep already printed and `SW` becomes two
  # lines. It would have been passed to `--swap-warnings` and the arm would have
  # died reporting nothing. IT DID NOT FIRE: both logs matched, at 615 and 626,
  # so the numbers in the evidence are the ones grep printed. The idiom is
  # repaired here rather than left in the record for the next reader to copy.
  SW=$(grep -aic "swap" "$W2/$tag-cap.log" 2>/dev/null); SW=${SW:-0}
  echo "$tag swap_warning_lines_in_capture_log=$SW"
  python3 "$SLICE" $R/$tag.csv $R/$tag-win.csv > "$OUT/$tag-slice.log" 2>&1
  grep -E "selected_stretch|wrote" "$OUT/$tag-slice.log" | sed "s/^/$tag  /"
  python3 "$RANK" $R/$tag-win.csv --sampler ArgmaxK --skip-first 0 --top 60 \
      --swap-warnings "$SW" > "$OUT/$tag-ranked-top60.txt" 2>&1
  echo "$tag rank_rc=$?"
  sed "s/^/$tag  /" "$OUT/$tag-ranked-top60.txt"
done
say "THE ONE LINE THIS JOB EXISTS FOR"
grep -H "HcGroupedNormKernel" "$OUT"/*-ranked-top60.txt || echo "HcGroupedNormKernel ABSENT FROM BOTH TABLES"
say ARCHIVE; ls -la "$OUT"
echo "=====HCN4_TAR_B64_BEGIN====="; tar czf - -C "$OUT" . | base64 -w 200; echo "=====HCN4_TAR_B64_END====="
say "DONE $OUT"
