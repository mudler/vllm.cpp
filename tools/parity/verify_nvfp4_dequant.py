#!/usr/bin/env python3
"""Verify the C++ NVFP4 W4A16 -> bf16 dequant against the pinned modelopt code.

THE GATE for M0.9 real-weight loading (M0.8 Task 4). Runs on dgx:

  1. Reads one real modelopt W4A16_NVFP4 expert weight (weight U8,
     weight_scale F8_E4M3, weight_scale_2 F32) from the 35B checkpoint shard.
  2. Dequants it in Python with the PINNED modelopt E2M1 decode
     (nvfp4_emulation_utils.break_fp4_bytes @ e24d1b24) composed exactly as
     dequantize_to_dtype(..., swizzle=False) does (modelopt scales are LINEAR
     fp8-e4m3fn; global scale weight_scale_2 multiplied, no reciprocation).
  3. Runs the C++ `dequant_nvfp4` tool on the same tensor and diffs the bf16
     bit patterns. Expected: max abs diff 0 (bit-exact after bf16 round).

Usage:
  python tools/parity/verify_nvfp4_dequant.py \
      --tool build/examples/dequant_nvfp4 \
      --shard ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/<snap>/model-00001-of-*.safetensors \
      --prefix model.language_model.layers.0.mlp.experts.0.down_proj

If --prefix is omitted, the first '*.weight' tensor that has sibling
'*.weight_scale' + '*.weight_scale_2' F8_E4M3/F32 tensors is auto-selected.
"""

import argparse
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from safetensors import safe_open

# Pinned modelopt E2M1 decode (the authoritative nibble->float table + packing).
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)

GROUP_SIZE = 16


def ref_dequant_bf16(weight_u8, weight_scale_fp8, weight_scale_2):
    """Pinned reference: matches dequantize_to_dtype(swizzle=False), block 16.

    weight_u8         : uint8  torch tensor [out, in/2]
    weight_scale_fp8  : float8_e4m3fn torch tensor [out, in/16]
    weight_scale_2    : python float (per-tensor global scale, multiplied)
    returns           : uint16 numpy array [out*in] of bf16 bit patterns
    """
    out_dim, packed = weight_u8.shape
    k = packed * 2
    # E2M1 decode via pinned break_fp4_bytes -> f32 [out, k].
    tensor_f32 = break_fp4_bytes(weight_u8, torch.float32)
    tensor_f32 = tensor_f32.reshape(out_dim, k // GROUP_SIZE, GROUP_SIZE)
    # Per-group scale in f32: f8_e4m3 -> f32, times the global scale.
    sf = weight_scale_fp8.to(torch.float32) * float(weight_scale_2)
    out = (tensor_f32 * sf.unsqueeze(-1)).reshape(out_dim, k)
    out_bf16 = out.to(torch.bfloat16)
    # bf16 bit patterns as uint16.
    return out_bf16.view(torch.uint16).cpu().numpy().reshape(-1)


def pick_prefix(f, keys):
    for name in keys:
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        s1 = base + ".weight_scale"
        s2 = base + ".weight_scale_2"
        if s1 in keys and s2 in keys:
            if f.get_slice(name).get_dtype() == "U8":
                return base
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, help="path to dequant_nvfp4 binary")
    ap.add_argument("--shard", required=True, help="path to .safetensors shard")
    ap.add_argument("--prefix", default=None, help="tensor prefix (no .weight)")
    args = ap.parse_args()

    with safe_open(args.shard, framework="pt") as f:
        keys = set(f.keys())
        prefix = args.prefix or pick_prefix(f, sorted(keys))
        if prefix is None:
            print("no W4A16_NVFP4 weight found in shard", file=sys.stderr)
            return 1
        w_name = prefix + ".weight"
        s_name = prefix + ".weight_scale"
        s2_name = prefix + ".weight_scale_2"
        print(f"tensor: {w_name}")
        weight_u8 = f.get_tensor(w_name)
        weight_scale = f.get_tensor(s_name)
        weight_scale_2 = f.get_tensor(s2_name)

    print(f"  weight {tuple(weight_u8.shape)} {weight_u8.dtype}")
    print(f"  weight_scale {tuple(weight_scale.shape)} {weight_scale.dtype}")
    ws2 = float(weight_scale_2.flatten()[0].item())
    print(f"  weight_scale_2 = {ws2!r}")

    ref = ref_dequant_bf16(weight_u8, weight_scale, ws2)
    out_dim, packed = weight_u8.shape
    in_dim = packed * 2

    # C++ side.
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        raw_path = tmp.name
    res = subprocess.run(
        [args.tool, args.shard, w_name, s_name, s2_name, raw_path],
        capture_output=True,
        text=True,
        check=True,
    )
    cpp_dims = tuple(int(x) for x in res.stdout.split())
    assert cpp_dims == (out_dim, in_dim), (cpp_dims, (out_dim, in_dim))
    with open(raw_path, "rb") as fh:
        cpp = np.frombuffer(fh.read(), dtype=np.uint16)
    assert cpp.size == ref.size, (cpp.size, ref.size)

    # Compare bf16 bit patterns exactly.
    ndiff = int(np.count_nonzero(cpp != ref))
    if ndiff == 0:
        print(f"OK: {ref.size} bf16 values bit-exact (max diff 0)")
        return 0

    # Investigate: report ulp distribution (bf16 monotone in bit pattern for
    # same-sign; use signed-magnitude aware ulp for a real gap measure).
    a = cpp.astype(np.int32)
    b = ref.astype(np.int32)
    # Map to a monotone ordering: flip negatives (sign bit 0x8000).
    def mono(x):
        neg = (x & 0x8000) != 0
        return np.where(neg, 0x8000 - (x & 0x7FFF), x | 0)
    ulp = np.abs(mono(a) - mono(b))
    print(f"MISMATCH: {ndiff}/{ref.size} differ; max ulp = {int(ulp.max())}")
    bad = np.nonzero(cpp != ref)[0][:8]
    for i in bad:
        print(f"  idx {int(i)}: cpp=0x{cpp[i]:04x} ref=0x{ref[i]:04x}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
