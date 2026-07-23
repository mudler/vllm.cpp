#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# A minimal lm:// client that uses LMCache's OWN protocol codec classes
# (lmcache.v1.protocol.ClientMetaMessage / ServerMetaMessage, lmcache.utils.
# CacheEngineKey, lmcache.v1.memory_management.MemoryFormat) to PUT/GET against a
# running lmcache.v1.server.  It proves TRUE wire interop with the C++
# LmcacheRemoteClient (W2): a chunk written by the real Python codec is read back
# byte-identically by C++, and vice-versa.
#
# Deliberately does NOT construct the full LMCache LMCServerConnector (which
# needs a LocalCPUBackend + asyncio loop + GPU paths); it speaks the same wire
# with the same serialization classes over a plain blocking socket.
#
# torch MUST be imported before lmcache (lmcache's lazy device-detect `import
# torch` otherwise trips a torch-internal circular import), and the compiled
# lmcache.c_ops native ext is stubbed (unused by the lm:// path).
import argparse
import socket
import sys
import types

import torch  # noqa: F401  (import order matters — see module docstring)

_stub = types.ModuleType("lmcache.c_ops")
_stub.__getattr__ = lambda n: (lambda *a, **k: None)
sys.modules["lmcache.c_ops"] = _stub

from lmcache.utils import CacheEngineKey  # noqa: E402
from lmcache.v1.memory_management import MemoryFormat  # noqa: E402
from lmcache.v1.protocol import (  # noqa: E402
    ClientCommand,
    ClientMetaMessage,
    ServerMetaMessage,
    ServerReturnCode,
)


def _recv_all(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


def _key(s):
    return CacheEngineKey.from_string(s)


def do_put(sock, key_str, payload):
    key = _key(key_str)
    hdr = ClientMetaMessage(
        ClientCommand.PUT,
        key,
        len(payload),
        MemoryFormat.KV_2LTD,
        torch.float16,
        torch.Size([2, 1, 1, 1]),
    ).serialize()
    sock.sendall(hdr)
    sock.sendall(payload)


def do_get(sock, key_str):
    key = _key(key_str)
    hdr = ClientMetaMessage(
        ClientCommand.GET,
        key,
        0,
        MemoryFormat.KV_2LTD,
        torch.float16,
        torch.Size([0, 0, 0, 0]),
    ).serialize()
    sock.sendall(hdr)
    meta = ServerMetaMessage.deserialize(_recv_all(sock, ServerMetaMessage.packlength()))
    if meta.code != ServerReturnCode.SUCCESS:
        return None
    return _recv_all(sock, meta.length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--op", choices=["put", "get"], required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--hex", help="payload hex for put; expected hex for get")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    try:
        if args.op == "put":
            payload = bytes.fromhex(args.hex)
            do_put(sock, args.key, payload)
            print(f"PUT_OK {args.key} {len(payload)}B", flush=True)
        else:
            got = do_get(sock, args.key)
            if got is None:
                print(f"GET_ABSENT {args.key}", flush=True)
                sys.exit(2)
            print(f"GET_OK {args.key} {len(got)}B hex={got.hex()}", flush=True)
            if args.hex is not None:
                want = bytes.fromhex(args.hex)
                if got != want:
                    print(f"MISMATCH got={got.hex()} want={want.hex()}", flush=True)
                    sys.exit(3)
                print("BYTE_IDENTICAL", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
