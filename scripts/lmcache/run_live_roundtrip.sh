#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# W2 live interop gate: stand up a REAL lmcache.v1.server (lm:// CPU store) and
# drive the C++ LmcacheRemoteClient against it, proving BIDIRECTIONAL wire
# interop with LMCache's own protocol codec:
#   A. C++ client <-> real server round-trip (PUT/GET byte-identical, EXIST,
#      absent-GET, KV_2LTD repack) — test_lmcache_client's REAL-server case.
#   B. Python (real lmcache codec) PUT -> C++ GET byte-identical (Python->C++).
#   C. C++ PUT -> Python (real lmcache codec) GET byte-identical (C++->Python).
#
# Requires a throwaway venv with: torch (CPU), py-cpuinfo, aiohttp, cachetools,
# prometheus_client, psutil, requests, sortedcontainers, pyyaml, msgspec; and a
# LMCache source checkout (pinned 8570aad).  The server is CPU-only; no GPU.
#
# Env:
#   LMVENV       path to the venv (its bin/python is used)
#   LMCACHE_SRC  path to the LMCache checkout (default /home/mudler/_git/lmcache-src)
#   TEST_BIN     path to the built test_lmcache_client
set -euo pipefail

LMVENV="${LMVENV:?set LMVENV to the throwaway venv path}"
LMCACHE_SRC="${LMCACHE_SRC:-/home/mudler/_git/lmcache-src}"
TEST_BIN="${TEST_BIN:?set TEST_BIN to the built test_lmcache_client}"
PY="$LMVENV/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"

# A free ephemeral port.
PORT="$("$PY" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
LOG="$(mktemp)"

echo "== launching real lmcache.v1.server on 127.0.0.1:$PORT =="
( cd /tmp && PYTHONPATH="$LMCACHE_SRC" "$PY" "$HERE/lm_server.py" 127.0.0.1 "$PORT" >"$LOG" 2>&1 ) &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  grep -q "LISTENING" "$LOG" 2>/dev/null && break
  sleep 0.5
done
if ! grep -q "LISTENING" "$LOG"; then
  echo "SERVER FAILED TO START:"; cat "$LOG"; exit 1
fi
echo "server up."

run_py() { PYTHONPATH="$LMCACHE_SRC" "$PY" "$HERE/lm_interop_client.py" --port "$PORT" "$@"; }

# Direction B setup: Python (real codec) PUTs a fixed payload for C++ to read.
PY_KEY="vtinteroppy@1@0@cafe@half"
PY_HEX="deadbeefcafe0102030405060708090a0b0c0d0e0f107f00ff"
echo "== [Python real-codec] PUT $PY_KEY =="
run_py --op put --key "$PY_KEY" --hex "$PY_HEX"
# Python round-trips its own PUT (real codec <-> server sanity).
run_py --op get --key "$PY_KEY" --hex "$PY_HEX"

# Direction C setup: C++ will PUT this; Python GETs it afterwards.
CPP_KEY="vtinteropcpp@1@0@c0ffee@half"
CPP_HEX="0102037f5645524c4c4d435050494e5445524f50feed00ff11"

echo "== [C++] round-trip A + read Python's key (B) + write CPP key (C) =="
VT_LMCACHE_LIVE_HOST=127.0.0.1 VT_LMCACHE_LIVE_PORT="$PORT" \
  VT_LMCACHE_PY_KEY="$PY_KEY" VT_LMCACHE_PY_HEX="$PY_HEX" \
  VT_LMCACHE_CPP_KEY="$CPP_KEY" VT_LMCACHE_CPP_HEX="$CPP_HEX" \
  "$TEST_BIN" -tc="lmcache LmcacheRemoteClient round-trip vs REAL lmcache.v1.server"

echo "== [Python real-codec] GET the C++-written key (C) =="
run_py --op get --key "$CPP_KEY" --hex "$CPP_HEX"

echo "ALL LIVE INTEROP DIRECTIONS PASSED (A: C++<->server, B: Python->C++, C: C++->Python)"
