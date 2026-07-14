#!/usr/bin/env python3
"""Emit the exact vLLM BF16 GDN-BA projection oracle for GB10.

The production vLLM path resolves MergedColumnParallelLinear.apply through
default_unquantized_gemm to torch.nn.functional.linear.  This maintainer tool
uses that exact dispatch with deterministic BF16 bit patterns at the real
Qwen3.6-27B BA shape.  Its JSON is checked into the parity fixture and consumed
bit-for-bit by the C++ CUDA test; it is not part of the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
import torch
import vllm
from vllm.model_executor.layers.utils import default_unquantized_gemm


TARGET_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"


def bf16_pattern(count: int, seed: int, device: torch.device) -> torch.Tensor:
    mixed = np.arange(count, dtype=np.uint32)
    mixed += np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    sign = ((mixed & np.uint32(1)) << np.uint32(15)).astype(np.uint16)
    exponent = (
        (np.uint32(124) + ((mixed >> np.uint32(1)) % np.uint32(5)))
        << np.uint32(7)
    ).astype(np.uint16)
    mantissa = ((mixed >> np.uint32(8)) & np.uint32(127)).astype(np.uint16)
    bits = sign | exponent | mantissa
    return torch.from_numpy(bits).view(torch.bfloat16).to(device)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().contiguous().view(torch.uint16).cpu().numpy()
    return values.astype("<u2", copy=False).tobytes()


def sha256_hex(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def cbor_bytes(value: bytes) -> bytes:
    length = len(value)
    if length < 24:
        return bytes([0x40 | length]) + value
    if length <= 0xFF:
        return bytes([0x58, length]) + value
    if length <= 0xFFFF:
        return bytes([0x59]) + length.to_bytes(2, "big") + value
    if length <= 0xFFFFFFFF:
        return bytes([0x5A]) + length.to_bytes(4, "big") + value
    return bytes([0x5B]) + length.to_bytes(8, "big") + value


def cbor_sha256_hex(tensor: torch.Tensor) -> str:
    return hashlib.sha256(cbor_bytes(tensor_bytes(tensor))).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, action="append", dest="ms")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--require-vllm-version", default="0.25.0")
    parser.add_argument("--target-commit", default=TARGET_COMMIT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the binding projection oracle")
    if vllm.__version__ != args.require_vllm_version:
        raise SystemExit(
            f"expected vLLM {args.require_vllm_version}, got {vllm.__version__}"
        )
    if args.repetitions < 2:
        raise SystemExit("at least two repetitions are required for stability")

    device = torch.device("cuda:0")
    ms = args.ms or [1, 2, 4, 16, 32]
    if len(ms) != len(set(ms)) or any(m <= 0 for m in ms):
        raise SystemExit("--m values must be unique positive integers")

    k = 5120
    n = 96
    input_seed = 7
    weight_seed = 19
    weight = bf16_pattern(n * k, weight_seed, device).reshape(n, k)

    cases: list[dict[str, Any]] = []
    for m in ms:
        hidden = bf16_pattern(m * k, input_seed, device).reshape(m, k)
        outputs = [default_unquantized_gemm(None, hidden, weight) for _ in range(args.repetitions)]
        torch.cuda.synchronize(device)
        first = outputs[0]
        if first.dtype != torch.bfloat16 or tuple(first.shape) != (m, n):
            raise SystemExit(
                f"unexpected output for M={m}: dtype={first.dtype}, shape={tuple(first.shape)}"
            )
        if any(not torch.equal(first, other) for other in outputs[1:]):
            raise SystemExit(f"vLLM projection is not bit-stable for M={m}")
        cases.append(
            {
                "m": m,
                "input_sha256": sha256_hex(hidden),
                "output_sha256": sha256_hex(first),
                "output_cbor_sha256": cbor_sha256_hex(first),
            }
        )

    props = torch.cuda.get_device_properties(device)
    result = {
        "op": "gdn_ba_projection_bf16",
        "oracle": {
            "vllm_version": vllm.__version__,
            "vllm_target_commit": args.target_commit,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "dispatch": "vllm.model_executor.layers.utils.default_unquantized_gemm",
            "implementation": "torch.nn.functional.linear",
            "repetitions": args.repetitions,
        },
        "shape": {"k": k, "n": n},
        "pattern": {
            "input_seed": input_seed,
            "weight_seed": weight_seed,
            "formula": "x=u32(i+seed*0x9e3779b9); x=(x^(x>>16))*0x7feb352d; x=(x^(x>>15))*0x846ca68b; x^=x>>16; sign=(x&1)<<15; exponent=(124+((x>>1)%5))<<7; mantissa=(x>>8)&127",
        },
        "weight_sha256": sha256_hex(weight),
        "cases": cases,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
