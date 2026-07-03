#!/usr/bin/env python3
"""Python truth side for examples/dump_container (M0.4 e2e container gate).

Prints the exact same manifest as the C++ CLI, one line per tensor sorted by
name:

    <name> <dtype> [d0,d1,...] <nbytes> <sha256 of first min(nbytes, 64KiB)>

followed by "TOTAL <count> tensors". `diff <(python) <(c++)` must be empty.

- .safetensors: the header is parsed directly (u64 LE header length + JSON),
  so the hashed bytes come raw from the file with no framework in between.
  Dependency-free (stdlib only).
- .gguf: uses gguf-py from mudler's killgate llama.cpp fork (knows the NVFP4 /
  Q1_0 extension type ids). ReaderTensor.data is a raw view over the on-disk
  bytes (quantized types stay as their packed block bytes, never dequantized);
  we re-flatten it to uint8 before hashing. Override the gguf-py location with
  VLLM_GGUF_PY (default: ~/llama-phase84-attn-only-source/gguf-py).
"""

import hashlib
import json
import os
import struct
import sys

HASH_PREFIX_BYTES = 65536


def shape_str(dims):
    return "[" + ",".join(str(int(d)) for d in dims) + "]"


def dump_safetensors(path):
    rows = []
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            nbytes = end - begin
            f.seek(data_start + begin)
            first = f.read(min(nbytes, HASH_PREFIX_BYTES))
            rows.append(
                (
                    name,
                    info["dtype"],
                    shape_str(info["shape"]),
                    nbytes,
                    hashlib.sha256(first).hexdigest(),
                )
            )
    return rows


def dump_gguf(path):
    gguf_py = os.environ.get(
        "VLLM_GGUF_PY",
        os.path.expanduser("~/llama-phase84-attn-only-source/gguf-py"),
    )
    sys.path.insert(0, gguf_py)
    import numpy as np  # noqa: E402
    from gguf import GGUFReader  # noqa: E402

    reader = GGUFReader(path, "r")
    rows = []
    for t in reader.tensors:
        # t.data is a view over the mmap'd file: packed block bytes for
        # quantized types, scalar dtypes for F16/F32/... — flatten back to the
        # raw on-disk byte stream either way.
        raw = t.data.reshape(-1).view(np.uint8)
        nbytes = int(raw.size)
        if nbytes != int(t.n_bytes):
            raise RuntimeError(
                f"{t.name}: data span {nbytes} != n_bytes {t.n_bytes}"
            )
        first = raw[: min(nbytes, HASH_PREFIX_BYTES)].tobytes()
        rows.append(
            (
                t.name,
                t.tensor_type.name,
                # gguf-py keeps dims in on-disk ggml order (fastest-varying
                # first); reverse into torch row-major order like the C++
                # reader does.
                shape_str(reversed(t.shape.tolist())),
                nbytes,
                hashlib.sha256(first).hexdigest(),
            )
        )
    return rows


def main():
    if len(sys.argv) != 2:
        print("usage: verify_containers.py <path.safetensors|path.gguf>",
              file=sys.stderr)
        return 2
    path = sys.argv[1]
    if path.endswith(".safetensors"):
        rows = dump_safetensors(path)
    elif path.endswith(".gguf"):
        rows = dump_gguf(path)
    else:
        print(f"verify_containers.py: unsupported suffix: {path}",
              file=sys.stderr)
        return 2
    rows.sort(key=lambda r: r[0])
    for name, dtype, shape, nbytes, digest in rows:
        print(f"{name} {dtype} {shape} {nbytes} {digest}")
    print(f"TOTAL {len(rows)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
