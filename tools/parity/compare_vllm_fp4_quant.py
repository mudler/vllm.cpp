#!/usr/bin/env python3
"""Compare a vllm.cpp BF16 NVFP4 quant dump with pinned vLLM byte-for-byte.

The dump is produced by ``test_ops_nvfp4_fp4`` with
``VT_FP4_TEST_BF16_INPUT=1`` and ``VT_FP4_TEST_DUMP_DIR``.  This invokes the
same ``scaled_fp4_quant`` custom op used by the pinned FlashInfer-CUTLASS
linear path, including its native SM12 reciprocal and FP4 conversion rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vllm._custom_ops import scaled_fp4_quant


def load(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(f"{path}: expected {expected} elements, found {array.size}")
    return array.reshape(shape)


def first_difference(actual: np.ndarray, expected: np.ndarray) -> str:
    indices = np.argwhere(actual != expected)
    if not indices.size:
        return ""
    index = tuple(int(value) for value in indices[0])
    return f" first={index} ours=0x{int(expected[index]):02x} vllm=0x{int(actual[index]):02x}"


def normalize_fp4_negative_zero(packed: np.ndarray) -> np.ndarray:
    """Canonicalize E2M1 +/-0 nibbles; their GEMM values are identical."""
    low = packed & 0x0F
    high = packed >> 4
    low = np.where((low & 0x07) == 0, 0, low)
    high = np.where((high & 0x07) == 0, 0, high)
    return (low | (high << 4)).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    args = parser.parse_args()

    m, _n, k, mp, _npad, kp = map(
        int, (args.dump / "shape.txt").read_text(encoding="utf-8").split()
    )
    input_bits = load(args.dump / "input_bf16.bin", np.uint16, (m, k))
    input_tensor = torch.from_numpy(input_bits.copy()).view(torch.bfloat16).cuda()
    global_scale = torch.from_numpy(
        load(args.dump / "input_global_scale.bin", np.float32, (1,)).copy()
    ).cuda()
    expected_packed = load(args.dump / "a_packed.bin", np.uint8, (m, k // 2))
    expected_scale = load(args.dump / "a_sf_sw.bin", np.uint8, (mp, kp))

    actual_packed, actual_scale = scaled_fp4_quant(
        input_tensor,
        global_scale,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )
    torch.cuda.synchronize()
    actual_packed_np = actual_packed.cpu().numpy().view(np.uint8).reshape(m, k // 2)
    actual_scale_np = actual_scale.view(torch.uint8).cpu().numpy().reshape(mp, kp)

    packed_differing = int(np.count_nonzero(actual_packed_np != expected_packed))
    actual_numeric = normalize_fp4_negative_zero(actual_packed_np)
    expected_numeric = normalize_fp4_negative_zero(expected_packed)
    numeric_differing = int(np.count_nonzero(actual_numeric != expected_numeric))
    scale_differing = int(np.count_nonzero(actual_scale_np != expected_scale))
    print(
        f"shape={m}x{k} packed_differing={packed_differing}/{expected_packed.size} "
        f"numeric_differing={numeric_differing}/{expected_packed.size}"
        f"{first_difference(actual_packed_np, expected_packed)}"
    )
    print(
        f"shape={m}x{k} scale_differing={scale_differing}/{expected_scale.size}"
        f"{first_difference(actual_scale_np, expected_scale)}"
    )


if __name__ == "__main__":
    main()
