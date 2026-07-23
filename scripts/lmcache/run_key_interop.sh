#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# W4 key-agreement-OVER-THE-WIRE gate (peer -> us). Stands up a REAL
# lmcache.v1.server, has a REAL lmcache ChunkedTokenDatabase derive a chunk key
# from tokens and PUT KV under it (scripts/lmcache/lm_key_interop.py), then runs
# the C++ test_lmcache_key_agreement LIVE case, which independently re-derives
# the SAME key with our ChunkedTokenDatabase and GETs the peer-written bytes:
# proof our C++ key agrees with a real vLLM+LMCache peer over the actual wire and
# the peer's KV is found + unpacked by us.
#
# Env:
#   LMVENV       throwaway venv (its bin/python); needs cbor2 + the server deps
#   VLLM_SHIM    dir with the vLLM sha256_cbor + init_none_hash shim on PYTHONPATH
#                (or a real vLLM install; the two leaf fns are all that is used)
#   LMCACHE_SRC  LMCache checkout, pinned 8570aad (default /home/mudler/_git/lmcache-src)
#   TEST_BIN     built test_lmcache_key_agreement
set -euo pipefail

LMVENV="${LMVENV:?set LMVENV to the throwaway venv path}"
VLLM_SHIM="${VLLM_SHIM:?set VLLM_SHIM to the vLLM-hashing shim dir}"
LMCACHE_SRC="${LMCACHE_SRC:-/home/mudler/_git/lmcache-src}"
TEST_BIN="${TEST_BIN:?set TEST_BIN to the built test_lmcache_key_agreement}"
PY="$LMVENV/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONHASHSEED=0

PORT="$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
LOG="$(mktemp)"; SPEC="$(mktemp --suffix=.json)"

echo "== launching real lmcache.v1.server on 127.0.0.1:$PORT =="
( cd /tmp && PYTHONPATH="$LMCACHE_SRC" "$PY" "$HERE/lm_server.py" 127.0.0.1 "$PORT" >"$LOG" 2>&1 ) &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true; rm -f "$SPEC"' EXIT
for _ in $(seq 1 60); do grep -q "LISTENING" "$LOG" 2>/dev/null && break; sleep 0.5; done
grep -q "LISTENING" "$LOG" || { echo "SERVER FAILED:"; cat "$LOG"; exit 1; }
echo "server up."

echo "== [Python real lmcache ChunkedTokenDatabase] derive key + PUT KV =="
PYTHONPATH="$VLLM_SHIM:$LMCACHE_SRC" "$PY" "$HERE/lm_key_interop.py" \
  --port "$PORT" --model "meta-llama/Llama-3.1-8B" --dtype bfloat16 \
  --chunk 256 --ntokens 256 --payload-bytes 512 --out "$SPEC"

echo "== [C++] re-derive the SAME key + GET the peer-written KV =="
VT_LMCACHE_LIVE_SPEC="$SPEC" "$TEST_BIN" \
  -tc="lmcache key-agreement: LIVE peer->us over the wire" -s

echo "KEY-AGREEMENT-OVER-THE-WIRE PASSED (peer key == our key; peer KV read by us)"
