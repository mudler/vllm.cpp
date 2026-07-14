#!/usr/bin/env python3
"""Dump the vLLM 0.25 BF16 packed GDN decode semantic oracle.

This maintainer-only tool executes both upstream paths covered by
``tests/kernels/test_fused_recurrent_packed_decode.py``: the packed decode
kernel and its explicit recurrent reference.  The fixture is accepted only
when repeated packed launches are bit-stable and the explicit reference stays
inside the upstream BF16 tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import vllm
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)


TARGET_COMMIT = "702f4814fe54fabff350d43cb753ae3e47c0c276"


def bf16_pattern(count: int, seed: int, exponent_span: int = 4) -> np.ndarray:
    mixed = np.arange(count, dtype=np.uint32)
    mixed += np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF)
    mixed ^= mixed >> np.uint32(16)
    mixed *= np.uint32(0x7FEB352D)
    mixed ^= mixed >> np.uint32(15)
    mixed *= np.uint32(0x846CA68B)
    mixed ^= mixed >> np.uint32(16)
    sign = ((mixed & np.uint32(1)) << np.uint32(15)).astype(np.uint16)
    exponent = (
        (np.uint32(123) + ((mixed >> np.uint32(1)) % np.uint32(exponent_span)))
        << np.uint32(7)
    ).astype(np.uint16)
    mantissa = ((mixed >> np.uint32(8)) & np.uint32(127)).astype(np.uint16)
    return sign | exponent | mantissa


def bf16_tensor(
    shape: tuple[int, ...], seed: int, device: torch.device
) -> torch.Tensor:
    bits = bf16_pattern(int(np.prod(shape)), seed).reshape(shape)
    return torch.from_numpy(bits).view(torch.bfloat16).to(device)


def tensor_numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).cpu().numpy()
    return value.cpu().numpy()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_numpy(tensor).tobytes()).hexdigest()


def diff_stats(got: torch.Tensor, want: torch.Tensor) -> dict[str, float | int | bool]:
    return {
        "bit_identical": torch.equal(got, want),
        "bitdiff": torch.count_nonzero(
            got.view(torch.uint16) != want.view(torch.uint16)
        ).item(),
        "max_abs": (got.float() - want.float()).abs().max().item(),
        "mean_abs": (got.float() - want.float()).abs().mean().item(),
    }


def save_tensor(
    root: Path, manifest: dict[str, Any], name: str, tensor: torch.Tensor
) -> None:
    array = tensor_numpy(tensor)
    np.save(root / f"{name}.npy", array)
    dtype = {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.int32: "i32",
    }[tensor.dtype]
    manifest["tensors"][name] = {
        "file": f"{name}.npy",
        "dtype": dtype,
        "shape": list(tensor.shape),
        "sha256": tensor_sha256(tensor),
    }


def run_packed(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_in: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state_in.clone()
    batch, value_heads = a.shape
    value_dim = state.shape[-2]
    out = torch.empty(
        (batch, 1, value_heads, value_dim),
        device=mixed_qkv.device,
        dtype=mixed_qkv.dtype,
    )
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=out,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--require-vllm-version", default="0.25.0")
    parser.add_argument("--target-commit", default=TARGET_COMMIT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the packed-decode oracle")
    if vllm.__version__ != args.require_vllm_version:
        raise SystemExit(
            f"expected vLLM {args.require_vllm_version}, got {vllm.__version__}"
        )
    if args.repetitions < 2:
        raise SystemExit("at least two repetitions are required")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch, key_heads, value_heads = 2, 1, 3
    key_dim = value_dim = 128
    qkv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    scale = key_dim**-0.5

    mixed_qkv = bf16_tensor((batch, qkv_dim), 11, device)
    a = bf16_tensor((batch, value_heads), 13, device)
    b = bf16_tensor((batch, value_heads), 17, device)
    a_log = torch.tensor([-1.25, -0.5, 0.25], device=device)
    dt_bias = torch.tensor([0.125, -0.25, 0.75], device=device)
    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    state_in = bf16_tensor(
        (batch + 1, value_heads, value_dim, key_dim), 23, device
    )
    state_in[0].zero_()

    packed_runs = [
        run_packed(
            mixed_qkv,
            a,
            b,
            a_log,
            dt_bias,
            state_in,
            state_indices,
            scale,
        )
        for _ in range(args.repetitions)
    ]
    torch.cuda.synchronize(device)
    packed_out, packed_state = packed_runs[0]
    for index, (out, state) in enumerate(packed_runs[1:], start=1):
        if not torch.equal(out, packed_out) or not torch.equal(state, packed_state):
            raise SystemExit(f"packed decode is not bit-stable at repetition {index}")

    q, k, v = torch.split(
        mixed_qkv,
        [key_heads * key_dim, key_heads * key_dim, value_heads * value_dim],
        dim=-1,
    )
    q = q.view(batch, key_heads, key_dim).unsqueeze(1).contiguous()
    k = k.view(batch, key_heads, key_dim).unsqueeze(1).contiguous()
    v = v.view(batch, value_heads, value_dim).unsqueeze(1).contiguous()
    x = a.float() + dt_bias.float()
    softplus = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(a_log.float()) * softplus).unsqueeze(1)
    beta_full = torch.sigmoid(b.float()).unsqueeze(1)
    beta_packed = torch.sigmoid(b.float()).to(dtype).unsqueeze(1)

    def run_reference(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state_in.clone()
        out, _ = fused_recurrent_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=None,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return out, state

    reference_out, reference_state = run_reference(beta_packed)
    full_beta_out, full_beta_state = run_reference(beta_full)
    rounded_stats = {
        "out": diff_stats(reference_out, packed_out),
        "state": diff_stats(reference_state, packed_state),
    }
    full_stats = {
        "out": diff_stats(full_beta_out, packed_out),
        "state": diff_stats(full_beta_state, packed_state),
    }
    if rounded_stats["out"]["max_abs"] > 2e-2 or rounded_stats["state"]["max_abs"] > 2e-2:
        raise SystemExit(
            "rounded-beta reference exceeds the upstream BF16 tolerance: "
            f"{rounded_stats}"
        )
    if rounded_stats["out"]["bitdiff"] >= full_stats["out"]["bitdiff"]:
        raise SystemExit(
            "fixture does not distinguish BF16-rounded beta at the output: "
            f"rounded={rounded_stats} full={full_stats}"
        )
    if rounded_stats["state"]["bitdiff"] >= full_stats["state"]["bitdiff"]:
        raise SystemExit(
            "fixture does not distinguish BF16-rounded beta in state: "
            f"rounded={rounded_stats} full={full_stats}"
        )

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "op": "gdn_packed_decode_bf16",
        "args": {
            "batch": batch,
            "Hk": key_heads,
            "Hv": value_heads,
            "Dk": key_dim,
            "Dv": value_dim,
            "scale": scale,
            "use_qk_l2norm_in_kernel": True,
            "softplus_threshold": 20.0,
        },
        "tensors": {},
        "oracle": {
            "vllm_version": vllm.__version__,
            "vllm_target_commit": args.target_commit,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "packed_callable": (
                "vllm.model_executor.layers.fla.ops."
                "fused_recurrent_gated_delta_rule_packed_decode"
            ),
            "reference_callable": (
                "vllm.model_executor.layers.fla.ops."
                "fused_recurrent_gated_delta_rule"
            ),
            "upstream_test": (
                "tests/kernels/test_fused_recurrent_packed_decode.py::"
                "test_fused_recurrent_packed_decode_matches_reference"
            ),
            "repetitions": args.repetitions,
            "rounded_beta_reference": rounded_stats,
            "full_f32_beta_reference": full_stats,
            "semantic_boundary": (
                "sigmoid(b.float()).to(torch.bfloat16) before recurrence; "
                "q/k L2 normalization remains in-kernel"
            ),
        },
    }
    for name, tensor in (
        ("mixed_qkv", mixed_qkv),
        ("q", q.squeeze(1)),
        ("k", k.squeeze(1)),
        ("v", v.squeeze(1)),
        ("a", a),
        ("b", b),
        ("A_log", a_log),
        ("dt_bias", dt_bias),
        ("state_indices", state_indices),
        ("state_in", state_in),
        ("g", g.squeeze(1)),
        ("beta_full_f32", beta_full.squeeze(1)),
        ("beta_packed_bf16", beta_packed.squeeze(1)),
        ("beta_packed_f32", beta_packed.float().squeeze(1)),
        ("out", packed_out.squeeze(1)),
        ("state_out", packed_state),
    ):
        save_tensor(root, manifest, name, tensor)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["oracle"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
