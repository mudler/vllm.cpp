#!/usr/bin/env bash
# WHY. Wave 2 established that rocprofv3 --kernel-trace --output-format csv
# cannot deliver a trace from our binary on gfx1151: the 27B ABORTED (rc=134,
# 37 s) and qwen4_exp COMPLETED THE WORKLOAD and then hung in output generation
# until SIGKILL at 1860 s. Both wrote a ZERO-ROW CSV.
#
# The workload itself is fine under the profiler -- decode measured 4.804 tok/s
# profiled against 4.823 unprofiled, a 0.4% difference -- so what fails is the
# WRITER, not the tracing. rocprofv3 1.1.0's native output is rocpd (SQLite) and
# csv is a converter on top of it. This job tests the native path.
set -u
if [ "${KA_STAGED:-0}" != "1" ]; then
  cp -f "$0" /tmp/ka5-staged.sh && chmod +x /tmp/ka5-staged.sh && exec env KA_STAGED=1 bash /tmp/ka5-staged.sh
  exit 80
fi
L=/tmp/kattrib; BLD=$L/build
B27=/tmp/kattrib-27b/Qwen3.8-27B-Q4_K_M.gguf
Q4E=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
OUT=/workspace/rocm-kernel-attrib/w3-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
exec 2>&1
( while true; do echo "[hb] $(date -u +%H:%M:%S)"; sleep 120; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT INT TERM
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
say(){ echo; echo "===== $* ====="; date -u +%FT%TZ; }
CLI=$(find "$BLD" -name vllm-cli -type f -perm -u+x|head -1)
[ -n "$CLI" ] || { echo "FATAL no vllm-cli"; exit 87; }
echo "CLI=$CLI"; ldd "$CLI"|grep -c 'not found'|sed 's/^/not_found_libs: /'

say "A. WHY THE CSV WRITER DIED -- the archived stderr from wave 2"
for f in /workspace/rocm-kernel-attrib/w2-*/w2-C27p-stderr.txt.gz /workspace/rocm-kernel-attrib/w2-*/w2-Q4Ep-stderr.txt.gz; do
  [ -f "$f" ] || continue
  echo "--- $f (last 30 lines) ---"; zcat "$f" | tail -30
done

say "B. rocprofv3 capability surface"
rocprofv3 --version 2>&1 | head -6
rocprofv3 --help 2>&1 | grep -iE "output-format|rocpd|otf2|pftrace|kernel-trace" | head -12

fmt_leg(){ # fmt_leg <tag> <format> <model> <ngen> <rep> <timeout>
  local tag=$1 fmt=$2 model=$3 ngen=$4 rep=$5 tmo=$6
  local D=$L/w3-$tag; rm -rf "$D"; mkdir -p "$D/trace"
  say "LEG[$tag] --output-format $fmt  model=$(basename "$model") ngen=$ngen rep=$rep tmo=${tmo}s"
  local S=$(date +%s)
  timeout -k 30 --foreground "$tmo" rocprofv3 --kernel-trace --output-format "$fmt" \
      --output-directory "$D/trace" -- "$CLI" --model "$model" --device auto \
      --prompt 'The capital of France is' --max-tokens "$ngen" --temperature 0 \
      --max-num-seqs 1 --repeat "$rep" > "$D/out.txt" 2> "$D/err.txt" </dev/null
  local r=$?
  echo "LEG[$tag] rc=$r wall=$(( $(date +%s) - S ))s"
  case $r in 0) echo "LEG[$tag] VERDICT: EXITED 0";; 124) echo "LEG[$tag] VERDICT: TIMED OUT";;
    137) echo "LEG[$tag] VERDICT: SIGKILLED, it ignored SIGTERM";;
    134) echo "LEG[$tag] VERDICT: SIGABRT";; *) echo "LEG[$tag] VERDICT: exited $r";; esac
  grep -a "tok_s=" "$D/err.txt"|sed "s/^/$tag  /"
  echo "$tag  files produced:"; find "$D/trace" -type f -printf '   %s bytes  %p\n' 2>/dev/null | head -12
  echo "$tag  stderr tail:"; tail -12 "$D/err.txt" | sed "s/^/   /"
  cp -r "$D/trace" "$OUT/$tag-trace" 2>/dev/null
  gzip -9 -c "$D/err.txt" > "$OUT/$tag-err.txt.gz"
}

say "C. THE NATIVE WRITER on the SMALL artifact first"
fmt_leg R27 rocpd "$B27" 40 3 900

say "D. READ WHATEVER THE NATIVE WRITER PRODUCED"
DB=$(find "$L/w3-R27/trace" -name '*.db' -o -name '*.rocpd' 2>/dev/null | head -1)
if [ -n "$DB" ]; then
  echo "DB=$DB size=$(stat -c%s "$DB")"
  python3 - "$DB" <<'PY'
import sqlite3,sys
db=sys.argv[1]; c=sqlite3.connect(db)
tabs=[r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print("tables:", len(tabs))
for t in tabs:
    n=c.execute(f'select count(*) from "{t}"').fetchone()[0]
    if n: print(f"  {n:>9}  {t}")
for t in tabs:
    if 'dispatch' in t.lower() or 'kernel' in t.lower():
        cols=[d[1] for d in c.execute(f'pragma table_info("{t}")')]
        print(f"-- {t}: {cols}")
PY
else
  echo "NO DATABASE PRODUCED"; find "$L/w3-R27/trace" -type f | head
fi

say "E. THE QWEN4_EXP ARM, native writer, small window"
fmt_leg RQ4 rocpd "$Q4E" 12 3 1500

say "F. READ THE QWEN4_EXP DATABASE"
DB2=$(find "$L/w3-RQ4/trace" -name '*.db' -o -name '*.rocpd' 2>/dev/null | head -1)
if [ -n "$DB2" ]; then
  echo "DB2=$DB2 size=$(stat -c%s "$DB2")"
  gzip -9 -c "$DB2" > "$OUT/RQ4.db.gz"; echo "archived $(stat -c%s "$OUT/RQ4.db.gz") bytes"
else
  echo "NO DATABASE PRODUCED FOR QWEN4_EXP"
fi
[ -n "${DB:-}" ] && gzip -9 -c "$DB" > "$OUT/R27.db.gz"
ls -la "$OUT"
say "DONE /workspace $OUT"
