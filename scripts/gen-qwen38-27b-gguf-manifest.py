#!/usr/bin/env python3
"""Emit a Qwen3.8-27B GGUF tensor+metadata manifest as a C++ fixture.

`unsloth/Qwen3.8-27B-GGUF` @ revision `fe1e2a23d973adb629709749dc4f6756df66ef10`
ships the two files this row gates:

    Qwen3.8-27B-Q4_K_M.gguf   arch 'qwen35'   866 tensors, 51 kv
    mmproj-BF16.gguf          arch 'clip'     334 tensors, 35 kv

This reads only the GGUF **header** — tensor names, ggml dims and type ids, plus
the SCALAR metadata keys — and freezes it into a C++ fixture, so the loader's
tensor accounting is gated against the real checkpoints with no asset in CI. The
header is self-delimiting and sits at the front of the file, so a range request
over the first few MB is enough for a remote artifact:

    curl -sL -r 0-16777215 -o q4km.head \\
      https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q4_K_M.gguf
    python3 scripts/gen-qwen38-27b-gguf-manifest.py q4km.head Qwen38Q4KM \\
      > tests/vllm/models/qwen38_27b_q4km_gguf_manifest.inc

ARRAY-VALUED and long STRING kvs are deliberately NOT emitted. The language
file's `tokenizer.ggml.tokens`, `.merges` and `.token_type` alone are 743,494
entries and its `tokenizer.chat_template` is a multi-kilobyte Jinja document;
none of them is a tensor-accounting fact, and freezing them here would make a
tensor manifest the place a tokenizer change has to be edited.

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

INT_TYPES = {UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64}
FLOAT_TYPES = {FLOAT32, FLOAT64}

# ggml_type ids (ggml.h). Anything untabulated is emitted numerically so the
# fixture never silently mislabels an encoding.
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
    30: "BF16",
}

# A string kv longer than this is a document rather than a datum (the chat
# template), and is summarised by its length instead of frozen.
MAX_STRING_KV = 256


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


def c_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main() -> int:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {sys.argv[0]} <gguf-or-header-prefix> [symbol-prefix]")
    path = sys.argv[1]
    sym = sys.argv[2] if len(sys.argv) == 3 else "Qwen38Q4KM"
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != b"GGUF":
            raise SystemExit("not a GGUF file")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        ints, floats, strings = [], [], []
        arch = ""
        for _ in range(n_kv):
            key = r.string()
            type_id = r.u32()
            val = r.value(type_id)
            if key == "general.architecture":
                arch = val
            if type_id in INT_TYPES:
                ints.append((key, int(val)))
            elif type_id == BOOL:
                ints.append((key, 1 if val else 0))
            elif type_id in FLOAT_TYPES:
                floats.append((key, float(val)))
            elif type_id == STRING and len(val) <= MAX_STRING_KV:
                strings.append((key, val))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            type_id = r.u32()
            r.u64()  # offset — not part of the shape contract
            tensors.append((name, dims, type_id))

    tensors.sort(key=lambda t: t[0])
    ints.sort()
    floats.sort()
    strings.sort()

    # The language file states its vocabulary ONLY through the embedding's
    # shape: it carries no `<arch>.vocab_size` kv, which is why
    # HfConfigFromGguf falls back to token_embd. GgufTensorInfo::shape is the
    # REVERSE of the ggml ne order, so the vocab is the LAST ggml dim.
    vocab = 0
    for name, dims, _ in tensors:
        if name == "token_embd.weight" and len(dims) == 2:
            vocab = dims[-1]

    out = sys.stdout
    out.write(
        "// GENERATED by scripts/gen-qwen38-27b-gguf-manifest.py — DO NOT EDIT BY HAND.\n"
        "//\n"
        "// The tensor + scalar-metadata manifest of a REAL Qwen3.8-27B GGUF from\n"
        "// `unsloth/Qwen3.8-27B-GGUF` @ revision\n"
        "// fe1e2a23d973adb629709749dc4f6756df66ef10\n"
        f"// (GGUF v{version}, {n_tensors} tensors, {n_kv} kv, architecture {arch!r}).\n"
        "// Names, ggml dims, type ids and the SCALAR kvs only — no weight bytes, and\n"
        "// no array kv (the tokenizer's 743,494 entries are not an accounting fact).\n"
        "// This is what gates the loader's tensor accounting against a real k-quant\n"
        "// checkpoint with no asset in CI.\n"
        "// See .agents/specs/qwen38-27b-quant-arms.md and .agents/porting-a-model.md §2.\n"
        "#pragma once\n\n"
        "#include <cstdint>\n\n"
        "namespace vllm_test {\n\n"
    )
    out.write(f"inline constexpr int64_t k{sym}TensorCount = {len(tensors)};\n")
    out.write(f"inline constexpr int64_t k{sym}KvCount = {n_kv};\n")
    out.write(f"inline constexpr int64_t k{sym}Version = {version};\n")
    out.write(f'inline constexpr const char* k{sym}Architecture = "{c_escape(arch)}";\n')
    if vocab:
        out.write(
            "// token_embd.weight's OUTER dim. The file carries no `<arch>.vocab_size`\n"
            "// kv, so this shape is the only place it states its vocabulary.\n"
            f"inline constexpr int64_t k{sym}VocabSize = {vocab};\n"
        )
    out.write("\n")

    out.write(f"struct {sym}Tensor {{\n"
              "  const char* name;\n"
              "  int64_t dims[4];   // GGUF ne order (reversed vs torch), 0-padded\n"
              "  int32_t n_dims;\n"
              "  uint32_t ggml_type;\n"
              "};\n\n")
    out.write(f"inline constexpr {sym}Tensor k{sym}Tensors[] = {{\n")
    for name, dims, type_id in tensors:
        padded = list(dims) + [0] * (4 - len(dims))
        type_name = GGML_TYPE_NAMES.get(type_id, str(type_id))
        out.write(
            f'    {{"{c_escape(name)}", {{{", ".join(str(v) for v in padded)}}}, {len(dims)}, '
            f"{type_id}u}},  // {type_name}\n"
        )
    out.write("};\n\n")

    out.write(f"struct {sym}IntKv {{ const char* key; int64_t value; }};\n")
    out.write(f"inline constexpr {sym}IntKv k{sym}IntKv[] = {{\n")
    for key, val in ints:
        out.write(f'    {{"{c_escape(key)}", {val}}},\n')
    out.write("};\n\n")

    out.write(f"struct {sym}FloatKv {{ const char* key; double value; }};\n")
    out.write(f"inline constexpr {sym}FloatKv k{sym}FloatKv[] = {{\n")
    for key, val in floats:
        out.write(f'    {{"{c_escape(key)}", {val!r}}},\n')
    out.write("};\n\n")

    out.write(f"struct {sym}StringKv {{ const char* key; const char* value; }};\n")
    out.write(f"inline constexpr {sym}StringKv k{sym}StringKv[] = {{\n")
    for key, val in strings:
        out.write(f'    {{"{c_escape(key)}", "{c_escape(val)}"}},\n')
    out.write("};\n\n")

    out.write("}  // namespace vllm_test\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
