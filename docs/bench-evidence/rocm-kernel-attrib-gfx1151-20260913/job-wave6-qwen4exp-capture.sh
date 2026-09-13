RANK_B64='<gzip+base64 of scripts/rocm-rank-kernels.py, stripped for the record>'
set -u
ulimit -c 0
echo "$RANK_B64" | base64 -d | gunzip > /tmp/rank_committed.py
OUT=/workspace/rocmprof-kattrib-20260913
mkdir -p "$OUT" /tmp/rpt
HIPDIR=$(dirname $(find /opt -name libamdhip64.so.7 2>/dev/null | head -1))
export LD_LIBRARY_PATH="/tmp/kattrib/build:/tmp/kattrib/build/bin:$HIPDIR:/opt/rocm/lib"
echo "### HIPDIR=$HIPDIR"
MODEL=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
echo "### ENV-CHECK ###"; env | grep -E '^(HSA_|ROCR_|HIP_|GGML_|VT_)' || echo "inherited_env NONE"
echo "### DOUBLE-SLASH-PROBE ###"
( mkdir -p "//.rocprofv3" && echo "MKDIR_DBL_OK" && dd if=/dev/zero of="//.rocprofv3/p.dat" bs=1M count=1 2>&1|tail -1 && ls -l "//.rocprofv3" ) || echo "MKDIR_DBL_FAIL"
rm -rf "/.rocprofv3"
echo "### CAPTURE ###"
rm -rf /tmp/cap; mkdir -p /tmp/cap
date
( cd /tmp && ROCPROF_TMPDIR=/tmp/rpt timeout -s KILL 1500 /opt/rocm/bin/rocprofv3 \
    --kernel-trace --output-format csv --output-directory /tmp/cap \
    -- /tmp/kattrib/build/examples/vllm-cli --model "$MODEL" \
       --prompt 'The capital of France is' --max-tokens 64 --temperature 0 \
       --max-num-seqs 1 --repeat 3 ) > /tmp/cap.log 2>&1
echo "capture_rc=$?"
date
echo "--- tok_s ---"; grep -a "tok_s=" /tmp/cap.log | head -10
echo "--- errors ---"; grep -aE "ring_buffer|mmap failed|Opened result|caught signal" /tmp/cap.log | head -6
echo "--- files ---"; find /tmp/cap -type f -printf '%s %p\n'
KT=$(find /tmp/cap -name '*kernel_trace.csv' | head -1)
if [ -n "$KT" ] && [ "$(stat -c%s "$KT")" -gt 1000 ]; then
  echo "KT_ROWS=$(wc -l < "$KT")"
  gzip -c "$KT" > "$OUT/qwen4exp-iq1s-kernel-trace.csv.gz"
  AI=$(find /tmp/cap -name '*agent_info.csv' | head -1); [ -n "$AI" ] && gzip -c "$AI" > "$OUT/qwen4exp-iq1s-agent-info.csv.gz"
  cp /tmp/cap.log "$OUT/capture.log"
  sha256sum "$OUT"/* | sed 's|.*/||' > "$OUT/SHA256SUMS.txt"; cat "$OUT/SHA256SUMS.txt"
  echo "### RANK (ArgmaxK, skip-first 1) ###"
  python3 /tmp/rank_committed.py "$KT" --sampler ArgmaxK --skip-first 1 --top 30 > "$OUT/ranked.txt" 2>&1
  echo "rank_rc=$?"; cat "$OUT/ranked.txt"
else
  echo "NO_USABLE_TRACE"
fi
echo "=== J8_DONE ==="
