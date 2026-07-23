#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dump ground-truth LMCache CacheEngineKey strings from the REAL lmcache
`ChunkedTokenDatabase`, for the C++ key-agreement gate (test_lmcache_key_agreement).

This drives lmcache/v1/token_database.py:298-449 (ChunkedTokenDatabase) @ LMCache
8570aad UNMODIFIED — its production `lm://` / in-process key-derivation path — with
vLLM's own `sha256_cbor` + `init_none_hash` supplied from the pinned vLLM source
(vllm/utils/hashing.py:43, vllm/v1/core/kv_cache_utils.py:99-114 @ e24d1b24). The
key derivation touches ONLY those two vLLM leaf functions (token_database.py:88-99),
so a full vLLM install produces byte-identical keys; the shim exists only to avoid
importing the whole vLLM engine on a CPU dev box.

pre_caching_hash_algorithm = "sha256_cbor": the portable, non-Python-dependent
hash LMCache documents for cross-implementation / distributed caching
(hashing.py:43-57). PYTHONHASHSEED must be set for a reproducible NONE_HASH
(kv_cache_utils.py:111-114); the interop peer sets PYTHONHASHSEED=0.

Usage (throwaway venv, real lmcache-src + the vLLM-hashing shim on PYTHONPATH):
  PYTHONHASHSEED=0 PYTHONPATH="$SHIM:/home/mudler/_git/lmcache-src" \
    python scripts/lmcache/gen_key_agreement_fixtures.py > \
    tests/fixtures/lmcache/key_agreement_fixtures.json
"""
import json
import os
import sys

import torch  # noqa: F401  (lmcache imports torch at module scope)

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.v1.token_database as td
from lmcache.v1.token_database import ChunkedTokenDatabase

DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "half": torch.float16,
    "float": torch.float32,
}


def make_db(chunk_size, hash_algo, save_unfull, model, world, worker, dtype_name):
    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        pre_caching_hash_algorithm=hash_algo,
        save_unfull_chunk=save_unfull,
    )
    md = LMCacheMetadata(
        model_name=model,
        world_size=world,
        local_world_size=world,
        worker_id=worker,
        local_worker_id=worker,
        kv_dtype=DTYPE_BY_NAME[dtype_name],
        kv_shape=(1, 2, chunk_size, 1, 1),
        chunk_size=chunk_size,
    )
    return ChunkedTokenDatabase(cfg, md), cfg, md


def dump_case(name, tokens, chunk_size, hash_algo, save_unfull, model, world,
              worker, dtype_name):
    db, _cfg, _md = make_db(chunk_size, hash_algo, save_unfull, model, world,
                            worker, dtype_name)
    entries = []
    for start, end, key in db.process_tokens(list(tokens)):
        entries.append({
            "start": start,
            "end": end,
            "chunk_hash_hex": key.chunk_hash_hex,
            "key": key.to_string(),
        })
    return {
        "name": name,
        "tokens": list(tokens),
        "chunk_size": chunk_size,
        "hash_algo": hash_algo,
        "save_unfull_chunk": save_unfull,
        "none_hash": td.NONE_HASH,
        "none_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "model_name": model,
        "world_size": world,
        "worker_id": worker,
        "dtype": dtype_name,
        "entries": entries,
    }


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        sys.stderr.write("ERROR: run with PYTHONHASHSEED=0 for reproducibility\n")
        return 1
    cases = []
    # The interop default: chunk_size 256, sha256_cbor, bf16 (Llama-3.1-8B style).
    cases.append(dump_case(
        "default_256_full", range(1, 257), 256, "sha256_cbor", False,
        "meta-llama/Llama-3.1-8B", 1, 0, "bfloat16"))
    # A two-full-chunk prefix (512 tokens) at chunk 256.
    cases.append(dump_case(
        "two_chunks_512", range(1000, 1512), 256, "sha256_cbor", False,
        "meta-llama/Llama-3.1-8B", 1, 0, "bfloat16"))
    # Non-monotone token ids (realistic prompt), 300 tokens: one full chunk only
    # (save_unfull=False drops the 44-token tail).
    seq = [(i * 2654435761) % 32000 for i in range(300)]
    cases.append(dump_case(
        "prompt_300_drop_tail", seq, 256, "sha256_cbor", False,
        "meta-llama/Llama-3.1-8B", 1, 0, "bfloat16"))
    # save_unfull_chunk=True keeps the trailing partial chunk (300 -> 256 + 44).
    cases.append(dump_case(
        "prompt_300_keep_tail", seq, 256, "sha256_cbor", True,
        "meta-llama/Llama-3.1-8B", 1, 0, "bfloat16"))
    # Small chunk_size for dense boundary coverage (partial + identity fields).
    cases.append(dump_case(
        "chunk4_world2_worker1_half", [10, 11, 12, 13, 14, 15], 4,
        "sha256_cbor", True, "m", 2, 1, "half"))
    # float32 dtype-string ("float"), single chunk.
    cases.append(dump_case(
        "chunk8_float", list(range(8)), 8, "sha256_cbor", False,
        "some/model", 4, 3, "float"))
    # Empty / sub-chunk (no full chunk, save_unfull=False -> no entries).
    cases.append(dump_case(
        "subchunk_none", [7, 8, 9], 256, "sha256_cbor", False,
        "meta-llama/Llama-3.1-8B", 1, 0, "bfloat16"))

    out = {
        "provenance": {
            "lmcache_pin": "8570aad",
            "vllm_pin": "e24d1b24",
            "driver": "real lmcache ChunkedTokenDatabase.process_tokens()",
            "hash": "vLLM sha256_cbor + init_none_hash (pinned source, verbatim)",
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "cases": cases,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
