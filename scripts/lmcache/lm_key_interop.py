#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# W4 key-agreement-over-the-wire peer. Derives a CacheEngineKey from tokens using
# the REAL lmcache ChunkedTokenDatabase (token_database.py:298-449 @ 8570aad) —
# the SAME production path a real vLLM+LMCache instance keys chunks on — with
# vLLM's own sha256_cbor + init_none_hash (pinned source), then PUTs a
# deterministic KV_2LTD payload under that key via LMCache's OWN lm:// protocol
# codec (lmcache.v1.protocol) against a running lmcache.v1.server.
#
# It writes a JSON spec {tokens, key, payload_hex, ...} the C++ side reads to
# independently re-derive the SAME key (test_lmcache_key_agreement live case) and
# GET the peer-written bytes: proof that our C++ key agrees with a real peer over
# the actual wire, and the peer's KV is found + unpacked by us.
#
# torch is imported before lmcache (device-detect circular import); lmcache.c_ops
# is stubbed (unused by the lm:// CPU store). Run with the vLLM-hashing shim +
# lmcache-src on PYTHONPATH and PYTHONHASHSEED=0.
import argparse
import hashlib
import json
import socket
import sys
import types

import torch  # noqa: F401  (import order matters)

_stub = types.ModuleType("lmcache.c_ops")
_stub.__getattr__ = lambda n: (lambda *a, **k: None)
sys.modules["lmcache.c_ops"] = _stub

from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.protocol import (  # noqa: E402
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)
from lmcache.v1.token_database import ChunkedTokenDatabase  # noqa: E402

DTYPE_BY_NAME = {"bfloat16": torch.bfloat16, "half": torch.float16,
                 "float": torch.float32}


def _recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_BY_NAME))
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--ntokens", type=int, default=256)
    ap.add_argument("--payload-bytes", type=int, default=512)
    ap.add_argument("--out", required=True, help="path to write the JSON spec")
    args = ap.parse_args()

    tokens = [(i * 2654435761) % 32000 for i in range(args.ntokens)]

    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=args.chunk, pre_caching_hash_algorithm="sha256_cbor",
        save_unfull_chunk=False)
    md = LMCacheMetadata(
        model_name=args.model, world_size=args.world, local_world_size=args.world,
        worker_id=args.worker, local_worker_id=args.worker,
        kv_dtype=DTYPE_BY_NAME[args.dtype], kv_shape=(1, 2, args.chunk, 1, 1),
        chunk_size=args.chunk)
    db = ChunkedTokenDatabase(cfg, md)

    # The first full chunk's key = the real peer key for this prefix.
    first = None
    for start, end, key in db.process_tokens(tokens):
        first = (start, end, key)
        break
    if first is None:
        sys.stderr.write("no full chunk produced\n")
        return 1
    _start, _end, key = first
    key_str = key.to_string()

    # A deterministic payload keyed on the key string (so C++ can predict/verify).
    payload = bytearray()
    seed = key_str.encode()
    while len(payload) < args.payload_bytes:
        seed = hashlib.sha256(seed).digest()
        payload.extend(seed)
    payload = bytes(payload[:args.payload_bytes])

    # PUT via LMCache's OWN protocol codec.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    try:
        hdr = ClientMetaMessage(
            ClientCommand.PUT, key, len(payload), MemoryFormat.KV_2LTD,
            DTYPE_BY_NAME[args.dtype], torch.Size([2, 1, 1, 1])).serialize()
        sock.sendall(hdr)
        sock.sendall(payload)
    finally:
        sock.close()

    spec = {
        "host": args.host, "port": args.port, "model": args.model,
        "world": args.world, "worker": args.worker, "dtype": args.dtype,
        "chunk": args.chunk, "tokens": tokens, "key": key_str,
        "payload_hex": payload.hex(),
    }
    with open(args.out, "w") as f:
        json.dump(spec, f)
    print(f"PUT_BY_TOKENS_OK key={key_str} payload={len(payload)}B", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
