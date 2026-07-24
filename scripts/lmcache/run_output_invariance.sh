#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# KV-EXTERNAL-CACHE (LMCache) connector-ON full-model OUTPUT-INVARIANCE +
# throughput gate (spec gate 4/6 of
# .agents/specs/lmcache-cpp-client-connector.md).
#
# Stands up a REAL headless `lmcache.v1.server` (lm:// CPU remote-store, the W2
# recipe) and drives `test_lmcache_output_invariance` — a real OPT-125m
# generation loop — against it in BOTH modes:
#   1. mode "full"     — store -> restart (fresh engine) -> load, ONE process:
#                        connector-ON tokens must == connector-OFF tokens.
#   2. mode "loadonly" — a genuinely COLD second process that never stored, yet
#                        still HITS the server (populated by mode 1) and matches
#                        the OFF run token-for-token.
#
# The lm:// server is CPU-only (CUDA_VISIBLE_DEVICES= empty). Only the model
# forward touches the GPU, so the TEST runs are serialized under
# `flock $HOME/gpu.lock` (one series, one lock). Async scheduling is pinned OFF
# (VT_ASYNC_SCHED=0) so schedule->execute stays strictly paired for the
# connector's synchronous load/store timing; both arms use the SAME setting, so
# the A/B is apples-to-apples.
#
# Env (dgx defaults):
#   LMVENV       venv whose bin/python runs the server (default
#                ~/venvs/lmcache-throwaway)
#   LMCACHE_SRC  LMCache checkout, pinned 8570aad (default ~/lmcache-src-8570aad)
#   TEST_BIN     built test_lmcache_output_invariance (REQUIRED)
#   OPT_DIR      OPT-125m bf16 dir (default the test's ~/models/opt-125m-bf16-st)
#   GPU_LOCK     flock path (default $HOME/gpu.lock)
set -euo pipefail

LMVENV="${LMVENV:-$HOME/venvs/lmcache-throwaway}"
LMCACHE_SRC="${LMCACHE_SRC:-$HOME/lmcache-src-8570aad}"
TEST_BIN="${TEST_BIN:?set TEST_BIN to the built test_lmcache_output_invariance}"
GPU_LOCK="${GPU_LOCK:-$HOME/gpu.lock}"
PY="$LMVENV/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -x "$PY" ] || { echo "no venv python at $PY"; exit 1; }
[ -f "$LMCACHE_SRC/lmcache/v1/server/__main__.py" ] || {
  echo "no lmcache source at $LMCACHE_SRC"; exit 1; }

PORT="$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
LOG="$(mktemp)"

echo "== launching real lmcache.v1.server on 127.0.0.1:$PORT (CPU) =="
( cd /tmp && CUDA_VISIBLE_DEVICES= PYTHONPATH="$LMCACHE_SRC" \
    "$PY" "$HERE/lm_server.py" 127.0.0.1 "$PORT" >"$LOG" 2>&1 ) &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  grep -q "LISTENING" "$LOG" 2>/dev/null && break
  sleep 0.5
done
if ! grep -q "LISTENING" "$LOG"; then
  echo "SERVER FAILED TO START:"; cat "$LOG"; exit 1
fi
echo "server up."

OPT_ENV=()
[ -n "${OPT_DIR:-}" ] && OPT_ENV=(VLLM_CPP_OPT_MODEL_DIR="$OPT_DIR")

RC=0
run_mode() {
  local mode="$1"
  echo "== output-invariance: mode=$mode =="
  flock "$GPU_LOCK" env "${OPT_ENV[@]}" \
    VT_ASYNC_SCHED=0 \
    VT_LMCACHE_OI_HOST=127.0.0.1 VT_LMCACHE_OI_PORT="$PORT" \
    VT_LMCACHE_OI_MODE="$mode" \
    "$TEST_BIN" \
    -tc="LMCache connector-ON output-invariance in a real OPT-125m loop" \
    -s || RC=1
}

# mode 1: full (store -> restart -> load, one process).
run_mode full
# mode 2: loadonly (a genuinely cold second process that only hits the server).
run_mode loadonly

if [ "$RC" -eq 0 ]; then
  echo "OUTPUT-INVARIANCE GATE PASSED (both modes: full + cold-process loadonly)"
else
  echo "OUTPUT-INVARIANCE GATE FAILED"
fi
exit "$RC"
