#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Launches the REAL lmcache.v1.server.LMCacheServer (the lm:// remote-store CPU
# cache) headless, for the C++ LmcacheRemoteClient interop round-trip (W2).
#
# Two headless workarounds, both documented in the W2 record:
#  1. torch is imported BEFORE lmcache — lmcache's lazy device-detect `import
#     torch` otherwise trips a torch-internal circular import.
#  2. the compiled `lmcache.c_ops` native extension is stubbed — it is NOT
#     published as a CPU wheel and the lm:// CPU store (LMSLocalBackend, which
#     only stores bytearrays) never calls its pinned-memory alloc/free ops.
# Everything else is the unmodified upstream LMCacheServer.  Usage:
#   PYTHONPATH=<lmcache-src> python lm_server.py <host> <port>
import sys
import types

import torch  # noqa: F401  (import order matters — see module docstring)

_stub = types.ModuleType("lmcache.c_ops")
_stub.__getattr__ = lambda n: (lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("lmcache.c_ops native ext stubbed (unused by lm:// CPU store)")))
sys.modules["lmcache.c_ops"] = _stub

from lmcache.v1.server.__main__ import LMCacheServer  # noqa: E402

if __name__ == "__main__":
    host, port = sys.argv[1], int(sys.argv[2])
    srv = LMCacheServer(host, port, "cpu")
    print(f"LISTENING {host}:{port}", flush=True)
    srv.run()
