set -u
ulimit -c 0
run(){ NAME=$1; WD=$2; shift 2
  echo "--- ARM $NAME wd=$WD env=$* ---"
  rm -rf /tmp/o-$NAME; mkdir -p /tmp/o-$NAME
  ( cd "$WD" && env "$@" timeout -s KILL 150 /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv --output-directory /tmp/o-$NAME -- /tmp/tiny ) > /tmp/l-$NAME.txt 2>&1
  echo "rc=$?  cwd_was=$WD"
  grep -aE "ring_buffer|mmap failed" /tmp/l-$NAME.txt | head -2
  for f in $(find /tmp/o-$NAME -name '*kernel_trace.csv'); do echo "KT_BYTES=$(stat -c%s $f) ROWS=$(wc -l < $f)"; done
}
mkdir -p /tmp/cdtest /tmp/rpt
run B1 /       ROCPROF_TMPDIR=/tmp/rpt
run A1 /       X=1
run C1 /tmp/cdtest X=1
run A2 /       X=1
run A3 /       X=1
run B2 /       ROCPROF_TMPDIR=/tmp/rpt
echo "=== J7_DONE ==="
