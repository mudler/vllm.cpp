#!/usr/bin/env python3
"""Emit a Qwen3.8-Flash-Next (`qwen4exp`) GGUF tensor manifest as a C++ fixture.

The `unsloth/Qwen3.8-Flash-Next-GGUF` repo ships `UD-IQ1_S` as THREE shards,
67.56 GiB in total, and the split is not a detail this manifest can ignore:

    ...-00001-of-00003.gguf   10.9 MB   67 kv,    0 tensors  (metadata only)
    ...-00002-of-00003.gguf   ~25 GB     3 kv,  595 tensors
    ...-00003-of-00003.gguf   ~25 GB     3 kv,  629 tensors

Shard 1 carries every metadata key and NO tensor at all; shards 2 and 3 carry
3 keys each (`split.no`, `split.count`, `split.tensors.count`) and the whole
1224-tensor table between them. So this script takes all three and emits ONE
manifest with the tensors it can only get from 2 and 3.

It reads only the GGUF **headers** — names, ggml dims, type ids — so the loader's
name map, its shape resolution and its ggml-type coverage are gated against the
real checkpoint without checking in (or even downloading) the payload. The
header is self-delimiting and lives at the front of each file, so shard 1 is
fetched whole and a range request over the first few MB is enough for the rest:

    B=https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/8bdc666649440e9bdc97e16f3f75782c98478ff5/UD-IQ1_S
    for i in 1 2 3; do  # a REVISION, not `main`: this repo is being uploaded
      curl -sL -r 0-16777215 -o shard$i.head \\
        "$B/Qwen3.8-Flash-Next-UD-IQ1_S-0000$i-of-00003.gguf"
    done
    python3 scripts/gen-qwen4-exp-gguf-manifest.py shard1.head shard2.head \\
      shard3.head > tests/vllm/models/qwen4_exp_gguf_manifest.inc

Self-contained: deliberately does NOT use gguf-py, so it runs anywhere the file
does (same rationale as scripts/gen-muse-glimmer-gguf-manifest.py, which this is
modelled on).
"""

import struct
import sys

(
    UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
    UINT64, INT64, FLOAT64,
) = range(13)

FMT = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

# ggml_type ids (ggml.h). Anything untabulated is emitted numerically so the
# fixture never silently mislabels an encoding.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
    30: "BF16",
}


# The artifact this manifest freezes. A repo id alone is NOT a pin — GGUFs get
# re-quantized in place under an unchanged name — and this repo in particular was
# mid-upload when the headers were read, so the revision is part of the record.
REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "8bdc666649440e9bdc97e16f3f75782c98478ff5"


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        data = self.fh.read(n)
        if len(data) != n:
            raise SystemExit("short read: the header is larger than the fetched prefix")
        return data

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, type_id):
        if type_id in FMT:
            fmt, size = FMT[type_id]
            return struct.unpack(fmt, self.raw(size))[0]
        if type_id == STRING:
            return self.string()
        if type_id == ARRAY:
            elem = self.u32()
            count = self.u64()
            return [self.value(elem) for _ in range(count)]
        raise SystemExit(f"unknown GGUF value type {type_id}")


def read_shard(path):
    """Return (version, kv, tensors) for one shard header."""
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.u32())
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            type_id = r.u32()
            r.u64()  # offset — not part of the shape contract
            tensors.append((name, dims, type_id, path))
    return version, kv, tensors


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <shard.head> [<shard.head> ...]")
    sym = "Qwen4ExpGguf"

    version = None
    kv = {}
    tensors = []
    shard_of = {}
    for path in sys.argv[1:]:
        v, k, ts = read_shard(path)
        if version is None:
            version = v
        elif v != version:
            raise SystemExit("shards disagree on the GGUF version")
        # Shard 1 holds every real key; the tensor shards hold only the three
        # split keys. `split.no` is per-shard BY DEFINITION and is the one key
        # that must differ, so it is excluded from the agreement check rather
        # than merged; every other shared key has to match across shards.
        for key, val in k.items():
            if key == "split.no":
                continue
            if key in kv and kv[key] != val:
                raise SystemExit(f"shards disagree on {key}")
            kv[key] = val
        for t in ts:
            shard_of.setdefault(t[0], len(shard_of))
        tensors.extend(ts)

    names = [t[0] for t in tensors]
    if len(set(names)) != len(names):
        raise SystemExit("a tensor name appears in more than one shard")
    declared = kv.get("split.tensors.count")
    if declared is not None and int(declared) != len(tensors):
        raise SystemExit(
            f"split.tensors.count says {declared} but {len(tensors)} were read; "
            "a shard header is missing or truncated"
        )

    tensors.sort(key=lambda t: t[0])
    arch = kv.get("general.architecture", "")
    shards = int(kv.get("split.count", len(sys.argv) - 1))
    out = sys.stdout
    out.write(
        "// GENERATED by scripts/gen-qwen4-exp-gguf-manifest.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// The tensor manifest of the REAL Qwen3.8-Flash-Next GGUF from\n"
        f"// `{REPO}` @ revision {REVISION},\n"
        "// path `UD-IQ1_S`, read live 2026-08-26 by HTTP range request over the shard\n"
        "// headers. The revision is pinned rather than `main` for the usual reason and for\n"
        "// a sharper one: the repo's `lastModified` was 15:54:43Z and the read was at\n"
        "// 15:56Z, so the upload was still in progress and `main` is a moving target here.\n"
        f"// (GGUF v{version}, {len(tensors)} tensors across {shards} shards, architecture "
        f"{arch!r},\n"
        f"// file_type {kv.get('general.file_type', '?')}). Names, ggml dims and type ids only —\n"
        "// no weight bytes, so CI gates the loader's name map, its shape resolution and\n"
        "// its ggml-type coverage against the real 67.56 GiB checkpoint with no asset.\n"
        "//\n"
        "// This is the ONLY published artifact of this model that fits any device this\n"
        "// project owns: bf16 is ~360 GB, the official FP8 ~180 GB and NVFP4 ~128 GB,\n"
        "// against ~119.6 GiB usable on GB10. See .agents/specs/qwen4-exp-flash-next.md\n"
        "// and issue #1989.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )
    out.write(f"inline constexpr int64_t k{sym}TensorCount = {len(tensors)};\n")
    out.write(f"inline constexpr int64_t k{sym}Version = {version};\n")
    out.write(f"inline constexpr int64_t k{sym}ShardCount = {shards};\n")
    out.write(f'inline constexpr const char* k{sym}Architecture = "{arch}";\n\n')

    out.write(f"struct {sym}Tensor {{\n"
              "  const char* name;\n"
              "  int64_t dims[4];   // GGUF ne order (reversed vs torch), 0-padded\n"
              "  int32_t n_dims;\n"
              "  uint32_t ggml_type;\n"
              "  int32_t shard;     // 1-based shard the tensor was read from\n"
              "};\n\n")
    order = {path: i + 1 for i, path in enumerate(sys.argv[1:])}
    out.write(f"inline constexpr {sym}Tensor k{sym}Tensors[] = {{\n")
    for name, dims, type_id, path in tensors:
        padded = list(dims) + [0] * (4 - len(dims))
        type_name = GGML_TYPE_NAMES.get(type_id, str(type_id))
        out.write(
            f'    {{"{name}", {{{", ".join(str(v) for v in padded)}}}, {len(dims)}, '
            f"{type_id}u, {order[path]}}},  // {type_name}\n"
        )
    out.write("};\n\n")
    out.write("}  // namespace vllm_test\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
