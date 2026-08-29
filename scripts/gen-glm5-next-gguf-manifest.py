#!/usr/bin/env python3
"""Emit a GLM-5.3-Flash (`glm5next`) GGUF tensor manifest as a C++ fixture.

`unsloth/GLM-5.3-Flash-GGUF` ships `UD-Q2_K_XL` as FOUR shards, 101.2535 GiB in
total, and the split is not a detail this manifest can ignore:

    ...-00001-of-00004.gguf     9.4 MB   72 kv,    0 tensors  (metadata only)
    ...-00002-of-00004.gguf    ~45.9 GiB  3 kv,  676 tensors
    ...-00003-of-00004.gguf    ~46.5 GiB  3 kv,  622 tensors
    ...-00004-of-00004.gguf     ~8.8 GiB  3 kv,  114 tensors

Shard 1 carries every metadata key and NO tensor at all; shards 2-4 carry 3 keys
each (`split.no`, `split.count`, `split.tensors.count`) and the whole 1412-tensor
table between them. So this script takes all four and emits ONE manifest with the
tensors it can only get from 2, 3 and 4.

It reads only the GGUF **headers** — names, ggml dims, type ids — so the loader's
name map, its shape resolution and its ggml-type coverage are gated against the
real checkpoint without checking in (or even downloading) the 101 GiB of payload.
The header is self-delimiting and lives at the front of each file, so shard 1 is
fetched whole and a range request over the first few MB is enough for the rest:

    B=https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/d425e572fb9686125831f476129e51cea34bc5b4/UD-Q2_K_XL
    for i in 1 2 3 4; do  # a REVISION, not `main`
      curl -sL -r 0-16777215 -o shard$i.head \\
        "$B/GLM-5.3-Flash-UD-Q2_K_XL-0000$i-of-00004.gguf"
    done
    python3 scripts/gen-glm5-next-gguf-manifest.py shard1.head shard2.head \\
      shard3.head shard4.head > tests/vllm/models/glm5_next_gguf_manifest.inc

The staged copy under `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` works as
well and is what produced the committed file; only the first few MB of each shard
is read either way.

Self-contained: deliberately does NOT use gguf-py, so it runs anywhere the file
does (same rationale as scripts/gen-qwen4-exp-gguf-manifest.py, which this is
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
# re-quantized in place under an unchanged name — so the revision is part of the
# record, and it is the revision `.agents/oracles/llama-cpp-glm5next.md` names
# as the file whose `general.architecture` settled that oracle's own choice.
REPO = "unsloth/GLM-5.3-Flash-GGUF"
REVISION = "d425e572fb9686125831f476129e51cea34bc5b4"


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
    sym = "Glm5NextGguf"

    version = None
    kv = {}
    tensors = []
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
    blocks = int(kv.get("glm5next.block_count", 0))
    mtp = int(kv.get("glm5next.nextn_predict_layers", 0))
    # The per-layer schedule the file states, emitted beside the tensors so a
    # gate can assert the topology against the SAME bytes the tensor table came
    # from rather than against a transcription of it.
    head_kv = kv.get("glm5next.attention.head_count_kv")
    if not isinstance(head_kv, list):
        raise SystemExit(
            "glm5next.attention.head_count_kv is not the per-layer ARRAY form; "
            "this manifest exists to freeze that schedule and cannot be written "
            "from a file that states it as a scalar"
        )
    order = {path: i + 1 for i, path in enumerate(sys.argv[1:])}
    out = sys.stdout
    out.write(
        "// GENERATED by scripts/gen-glm5-next-gguf-manifest.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// The tensor manifest of the REAL GLM-5.3-Flash GGUF from\n"
        f"// `{REPO}` @ revision {REVISION},\n"
        "// path `UD-Q2_K_XL`, read 2026-08-29 from the staged copy under\n"
        "// `/mnt/nas_share/rc/ckpt/GLM-5.3-Flash-UD-Q2_K_XL/` — GGUF HEADERS ONLY, no\n"
        "// weight byte. This is the ONE published artifact of this model that fits any\n"
        "// device this project owns: the safetensors arms are FP8 305.78 GiB and BF16\n"
        "// 598.53 GiB, and the NVFP4 arm 181.32 GiB, against ~119.63 GiB on GB10.\n"
        f"// (GGUF v{version}, {len(tensors)} tensors across {shards} shards, architecture "
        f"{arch!r},\n"
        f"// file_type {kv.get('general.file_type', '?')}, block_count {blocks}, "
        f"nextn_predict_layers {mtp}).\n"
        "//\n"
        "// Names, ggml dims and type ids only, so CI gates the loader's name map, its\n"
        "// shape resolution and its ggml-type coverage against the real 101.2535 GiB\n"
        "// checkpoint with no asset. `kGlm5NextGgufHeadCountKv` is the file's own\n"
        f"// {len(head_kv)}-entry per-layer schedule (0 = a KDA `linear_attention` block,\n"
        "// non-zero = a DSA/MLA attention block); block " + str(len(head_kv) - 1) + " is the\n"
        "// multi-token-prediction block the reference discards.\n"
        "//\n"
        "// See .agents/specs/glm5-next-flash.md and issue #2242.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )
    out.write(f"inline constexpr int64_t k{sym}TensorCount = {len(tensors)};\n")
    out.write(f"inline constexpr int64_t k{sym}Version = {version};\n")
    out.write(f"inline constexpr int64_t k{sym}ShardCount = {shards};\n")
    out.write(f"inline constexpr int64_t k{sym}BlockCount = {blocks};\n")
    out.write(f"inline constexpr int64_t k{sym}NextnPredictLayers = {mtp};\n")
    out.write(f'inline constexpr const char* k{sym}Architecture = "{arch}";\n\n')
    out.write(f"inline constexpr int32_t k{sym}HeadCountKv[] = {{\n    ")
    out.write(", ".join(str(int(v)) for v in head_kv))
    out.write("};\n\n")

    out.write(f"struct {sym}Tensor {{\n"
              "  const char* name;\n"
              "  int64_t dims[4];   // GGUF ne order (reversed vs torch), 0-padded\n"
              "  int32_t n_dims;\n"
              "  uint32_t ggml_type;\n"
              "  int32_t shard;     // 1-based shard the tensor was read from\n"
              "};\n\n")
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
