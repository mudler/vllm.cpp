#!/bin/bash
# GATEABILITY, the RUN half: does llama.cpp at the `llama-cpp-qwen4exp` pin
# actually LOAD and DECODE the staged Qwen3.8-Flash-Next artifact?
#
# `AGENTS.md` §"Measure gateability": an oracle is gateable only once it
# demonstrably BUILDS and RUNS the model.  The build half was recorded on
# 2026-08-27 (docs/bench-evidence/oracle-llamacpp-qwen4exp-pr27742-build-20260827.md).
# #2060 owes this half.  A build proves the architecture is DECLARED; it says
# nothing about whether the graph it builds produces coherent text.
#
# This is NOT a benchmark.  One prompt, greedy, no clock window, no contention
# control, no repetitions.  Nothing here may be quoted as a number.
#
# THIS FILE IS IN THE TREE, and it was not on 29 August 2026.  The run that
# discharged #2060 and flipped the oracle to `gateable = yes` was executed from a
# copy on the NAS, so the proof was not reproducible from the repository.
#
# ONE THING THIS JOB MEASURED, WORTH READING BEFORE TRUSTING A KV GUARD.  The
# server log it captures -- the complete, unfiltered output of the redirect
# below -- contains NO `KV self size` line and no allocation summary at all: 16
# minutes pass between `load_model:` and `threadpool init` with nothing printed
# in between, and `/props` carries no KV bytes either.
# `scripts/qwen4exp-llamacpp-ladder.sh` used to grep for that line and default
# the term to zero when it was absent, which made its memory guard weightless on
# the only server it will ever face.  It now refuses instead.  If a future pin
# starts printing a KV size, this capture is where you will see it first.
set -u

# `/workspace` is CIFS mounted with `file_mode=0664`, which STRIPS the execute
# bit from everything the build copied out -- `llama-server` lands `-rw-rw-r--`
# no matter what mode `cp` asked for, and `chmod` on that mount does not stick.
# So the binaries are staged onto the worker's own overlay before they are run,
# which also keeps the hot path off a network filesystem.
# The object this proof is a proof ABOUT.  It is asserted below against the
# build's own BUILD-RECORD.txt rather than assumed from the directory name: a
# gateability claim that cannot say which revision answered is not a claim.
PIN=035e22731a7fd70b9854b3a2d64ec68e9b1a45d3
SRC_BIN=/workspace/q4exp-bench/llamacpp-pr27742-cuda/bin
BIN=/tmp/q4exp-bin
MODEL=/workspace/q4exp-bench/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
OUT=/workspace/q4exp-bench/decode-proof
PORT=8077

cleanup() {
  [ -n "${SRV_PID:-}" ] && kill "${SRV_PID}" 2>/dev/null
  kill "${HB:-}" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM
( while true; do sleep 60; echo "### hb $(date -u +%H:%M:%S)"; done ) &
HB=$!

# This job is QUEUED behind the build.  If the build did not finish, exit at
# once rather than hold a fleet device to discover that: a job that fails fast
# gives the box back, and a missing binary is not a finding worth a lease.
for b in llama-cli llama-server; do
  [ -s "$SRC_BIN/$b" ] || { echo "REFUSING: $SRC_BIN/$b is absent or empty -- the build did not produce it"; exit 11; }
done
rm -rf "$BIN"; mkdir -p "$BIN"
cp -L "$SRC_BIN"/* "$BIN"/ || { echo "FATAL: staging the binaries onto /tmp"; exit 12; }
chmod +x "$BIN"/llama-cli "$BIN"/llama-server "$BIN"/llama-bench
for b in llama-cli llama-server; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b still not executable after chmod"; exit 12; }
done
# The build copied REGULAR FILES only (`find -type f`), so every SONAME symlink
# -- libllama-common.so.0 -> libllama-common.so.0.3.0 -- was left behind, and
# the loader wants the SONAME. That is what this job found on its first run:
# `error while loading shared libraries: libllama-common.so.0`, rc=127, twice.
# Rebuild the links here, where /tmp is a real filesystem and CIFS's
# no-symlink limitation does not apply.
( cd "$BIN" && for f in lib*.so.*; do
    case "$f" in *.so) continue ;; esac
    base=${f%%.so.*}; ver=${f#*.so.}; major=${ver%%.*}
    [ "$f" = "$base.so.$major" ] || ln -sf "$f" "$base.so.$major"
    [ -e "$base.so" ] || ln -sf "$f" "$base.so"
  done )

# ASSERT the precondition rather than discover it after a 67 GiB load. An
# unresolved library is `not found` in ldd and rc=127 at exec, and rc=127 an
# hour into a job looks nothing like a linkage problem in a log.
echo "--- ldd, every dependency must resolve ---"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
MISSING=0
for b in llama-cli llama-server; do
  ldd "$BIN/$b" > /tmp/ldd-$b.txt 2>&1
  n=$(grep -c 'not found' /tmp/ldd-$b.txt)
  echo "$b: $n unresolved"
  [ "$n" -eq 0 ] || { grep 'not found' /tmp/ldd-$b.txt; MISSING=$((MISSING + n)); }
done
[ "$MISSING" -eq 0 ] || { echo "FATAL: $MISSING unresolved shared libraries"; exit 13; }

echo "--- IDENTITY: the binaries are the ones built AT THE PIN ---"
RECORD="$SRC_BIN/../BUILD-RECORD.txt"
[ -s "$RECORD" ] || { echo "FATAL: no BUILD-RECORD.txt beside $SRC_BIN"; exit 14; }
REC_PIN=$(awk '$1 == "pin" {print $2}' "$RECORD")
echo "BUILD-RECORD pin = ${REC_PIN:-ABSENT} (expected $PIN)"
[ "$REC_PIN" = "$PIN" ] || { echo "FATAL: these binaries were built at '$REC_PIN', not the pin"; exit 14; }
# The staged copy, hashed and compared -- not printed for a human to compare.
# A binary that was replaced on the share between the build and this job would
# otherwise answer the prompt and be recorded as the pin's answer.
DRIFT=0
for b in llama-cli llama-server llama-bench; do
  have=$(sha256sum "$BIN/$b" | cut -d' ' -f1)
  want=$(awk -v n="$b" '$2 == n {print $1}' "$RECORD")
  echo "$b  staged=$have  record=${want:-ABSENT}"
  [ -n "$want" ] && [ "$have" = "$want" ] || DRIFT=$((DRIFT + 1))
done
[ "$DRIFT" -eq 0 ] || { echo "FATAL: $DRIFT staged binary(ies) do not match BUILD-RECORD.txt"; exit 14; }

mkdir -p "$OUT"
echo "=== IDENTITY ==="
hostname; date -u +%Y-%m-%dT%H:%M:%SZ
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1
free -g | head -2
cat "$SRC_BIN/../BUILD-RECORD.txt" 2>&1 | head -20

echo "=== ARTIFACT (every shard, sized) ==="
ls -la /workspace/q4exp-bench/UD-IQ1_S/
for f in /workspace/q4exp-bench/UD-IQ1_S/*.gguf; do
  [ -s "$f" ] || { echo "FATAL: empty shard $f"; exit 10; }
done

"$BIN/llama-cli" --version 2>&1 | head -5

echo "=== 1. llama-cli: load and decode 64 greedy tokens ==="
# --no-warmup keeps the proof to ONE load; -no-cnv makes it a plain completion.
timeout 3600 "$BIN/llama-cli" \
  -m "$MODEL" \
  -ngl 99 -c 4096 -n 64 --temp 0 --seed 0 --no-warmup -no-cnv \
  -p "The capital of France is" \
  > "$OUT/llama-cli.log" 2>&1
CLI_RC=$?
echo "llama-cli rc=$CLI_RC (124 = the 3600s timeout fired)"
echo "--- architecture and tensor accounting ---"
grep -Ei 'arch|n_layer|n_embd|n_expert|qwen4exp|tensor|KV self|not (found|supported)|error|unknown' \
  "$OUT/llama-cli.log" | head -40
echo "--- the generated text ---"
tail -30 "$OUT/llama-cli.log"

echo "=== 2. llama-server: the path the LADDER actually uses ==="
timeout 3600 "$BIN/llama-server" \
  -m "$MODEL" --alias gate --host 127.0.0.1 --port "$PORT" \
  -ngl 99 -c 4096 -np 1 --cont-batching --flash-attn on \
  > "$OUT/llama-server.log" 2>&1 &
SRV_PID=$!
waited=0
ready=0
while [ "$waited" -lt 3600 ]; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "server EXITED after ${waited}s"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && { ready=1; echo "server READY after ${waited}s"; break; }
  sleep 5; waited=$((waited + 5))
done
if [ "$ready" -eq 1 ]; then
  curl -s "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"gate","prompt":"The capital of France is","max_tokens":64,"temperature":0,"seed":0}' \
    > "$OUT/completion.json" 2>&1
  echo "--- /v1/completions ---"
  head -c 2000 "$OUT/completion.json"; echo
  curl -s "http://127.0.0.1:$PORT/props" > "$OUT/props.json" 2>&1
  echo "--- server KV / load lines ---"
  grep -Ei 'KV self|n_ctx|slots|arch|qwen4exp|error' "$OUT/llama-server.log" | head -30
else
  echo "SERVER NEVER READY within ${waited}s"
fi
kill "$SRV_PID" 2>/dev/null
echo "=== server log tail ==="
tail -40 "$OUT/llama-server.log"
echo "=== DECODE PROOF DONE cli_rc=$CLI_RC server_ready=$ready ==="
