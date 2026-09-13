#!/bin/bash
# Close the one gap run 2 left: WHERE the tensors landed. A -ngl 99 that silently
# fell back to the CPU still prints a number. Also the KV size #2261 owes.
set -u
BIN=/tmp/q4exp-hip-src-D9K2/build/bin
GGUF=/tmp/ckpt-iq1s/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
OUT=/workspace/q4exp-gfx1151-denominator/placement-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib"
say(){ echo; echo "### $(date -u +%H:%M:%S) $*"; }
( while sleep 120; do kill -0 $$ 2>/dev/null || exit 0; echo "### hb"; done ) & HB=$!; trap 'kill -9 $HB 2>/dev/null' EXIT INT TERM

say "0. why the last llama-cli leg exited 1 in ONE second -- read the error, do not guess"
timeout 120 "$BIN/llama-cli" -m "$GGUF" -ngl 99 -n 4 -p "The capital of France is" -no-cnv --temp 0 --seed 1 -v > /dev/null 2> "$OUT/cli-probe.err" </dev/null
echo "llama-cli rc=$?"; echo "-- first 40 lines of its stderr --"; head -40 "$OUT/cli-probe.err"

say "1. llama-bench -v: the loader chatter on the SAME binary that produced every leg"
timeout 1200 "$BIN/llama-bench" -m "$GGUF" -p 0 -n 4 -ngl 99 -r 1 -v -o json > "$OUT/v.json" 2> "$OUT/v.err" </dev/null
echo "llama-bench -v rc=$?"
echo "=== load_tensors / buffer sizes (THE placement question) ==="
grep -E 'load_tensors|buffer size|offloa|n_gpu_layers|ROCm|CPU_Mapped|CPU model' "$OUT/v.err" | head -40
echo "=== KV cache sizing (owed by ISSUE-GH-2261) ==="
grep -iE 'kv cache|kv_cache|KV self|k_cache|v_cache|kv buffer|n_ctx' "$OUT/v.err" | head -20
echo "=== compute buffers ==="
grep -iE 'compute buffer|graph nodes|graph splits' "$OUT/v.err" | head -10
echo "=== system_info ==="; grep -m1 'system_info' "$OUT/v.err"
echo "=== anything the loader declined ==="; grep -iE 'missing|unused|skip|warn' "$OUT/v.err" | head -10
echo "=== stderr size ==="; wc -c "$OUT/v.err"
echo "=== FULL stderr (it is small) ==="; head -c 40000 "$OUT/v.err"

say "2. ARCHIVE both runs' small evidence to stdout"
gzip -f "$OUT"/*.err 2>/dev/null
echo "=====RUN1_TAR_B64_BEGIN====="
tar czf - -C /workspace/q4exp-gfx1151-denominator $(cd /workspace/q4exp-gfx1151-denominator && ls -d run-* 2>/dev/null | head -1) 2>/dev/null | base64 -w 200
echo "=====RUN1_TAR_B64_END====="
echo "=====PLACE_TAR_B64_BEGIN====="
tar czf - -C "$OUT" . 2>/dev/null | base64 -w 200
echo "=====PLACE_TAR_B64_END====="
say "DONE"
