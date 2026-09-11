#!/usr/bin/env python3
"""Create a separate two-layer recurrent control without changing gate fixtures.

The existing dense fixture supplies the vocabulary, embedding, output, and
full-attention layer. A deterministic recurrent layer precedes that layer.
Shapes and names follow both pinned secondary models/qwen35.cpp loaders.
This model is a control for the allocation overlay, not a native token gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import gguf
import numpy as np


class ControlWriter(gguf.GGUFWriter):
    def _pack_val(self, value, value_type, add_vtype, sub_type=None):
        # Preserve the original zero-length BPE merge array. The pinned Python
        # writer refuses empty arrays although both pinned C++ readers accept it.
        if value_type == gguf.GGUFValueType.ARRAY and len(value) == 0:
            if sub_type is None:
                raise RuntimeError("empty control metadata needs its original element type")
            prefix = struct.pack("<I", int(value_type)) if add_vtype else b""
            return prefix + struct.pack("<IQ", int(sub_type), 0)
        return super()._pack_val(value, value_type, add_vtype, sub_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_dense", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("control output must be new")
    source_hash = hashlib.sha256(args.original_dense.read_bytes()).hexdigest()
    reader = gguf.GGUFReader(args.original_dense)
    writer = ControlWriter(str(args.output), "qwen35")
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        value = field.contents()
        if field.name == "general.name":
            value = "rocm-gather-recurrent-allocation-control"
        if field.name in ("qwen35.block_count", "qwen35.full_attention_interval"):
            value = 2
        subtype = field.types[-1] if len(field.types) > 1 else None
        if field.types[0] == gguf.GGUFValueType.ARRAY and len(value) == 0:
            subtype = gguf.GGUFValueType(int(field.parts[3][0]))
        writer.add_key_value(field.name, value, field.types[0], sub_type=subtype)
    copied = []
    for tensor in reader.tensors:
        name = tensor.name.replace("blk.0.", "blk.1.")
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
        copied.append(name)

    # The control stays deterministic across NumPy versions with integer data.
    counter = 0
    added = {}

    def add(name, shape, *, constant=None):
        nonlocal counter
        count = int(np.prod(shape))
        if constant is None:
            integers = ((np.arange(count, dtype=np.int64) + counter) * 1103515245 + 12345) % 65536
            data = (integers.astype(np.float32) - 32768) / np.float32(2097152)
            counter += count
        else:
            data = np.full(count, constant, dtype=np.float32)
        data = data.reshape(shape)
        writer.add_tensor(name, data)
        added[name] = {"shape": list(shape), "dtype": "float32",
                       "sha256": hashlib.sha256(data.tobytes()).hexdigest()}

    add("blk.0.attn_norm.weight", (256,), constant=1)
    add("blk.0.post_attention_norm.weight", (256,), constant=1)
    add("blk.0.attn_qkv.weight", (192, 256))
    add("blk.0.attn_gate.weight", (64, 256))
    add("blk.0.ssm_conv1d.weight", (192, 4))
    add("blk.0.ssm_dt.bias", (1,), constant=1)
    add("blk.0.ssm_a", (1,), constant=-1)
    add("blk.0.ssm_beta.weight", (1, 256))
    add("blk.0.ssm_alpha.weight", (1, 256))
    add("blk.0.ssm_norm.weight", (64,), constant=1)
    add("blk.0.ssm_out.weight", (256, 64))
    for name in ("ffn_gate", "ffn_up", "ffn_down"):
        add("blk.0." + name + ".weight", (256, 256))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if hashlib.sha256(args.original_dense.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("original gate fixture changed")
    report = {"purpose": "separate actual recurrent-layer control", "layers": ["recurrent", "full_attention"],
              "original_sha256": source_hash, "control_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "copied_tensors": copied, "added_tensors": added,
              "source": "stock 10bf611e and fork 36fe8e1c models/qwen35.cpp::load_arch_tensors"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"control_sha256": report["control_sha256"], "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
