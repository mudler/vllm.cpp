#!/usr/bin/env python3
"""Generate reduced DeepSeek-V4 Flash Vision stage goldens.

The module definitions below are a direct transcription of inference/vision.py
from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp at revision
86f746b36186f0e567729a5c06a8c918caba82a9. The committed fixture records the
same revision and the torch version that executed these formulas.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

REVISION = "86f746b36186f0e567729a5c06a8c918caba82a9"
SOURCE = "inference/vision.py"


@dataclass(frozen=True)
class Args:
    vision_patch_size: int
    vision_dim: int
    vision_n_heads: int
    vision_n_layers: int
    vision_inter_dim: int
    vision_rope_theta: float
    vision_downsample_ratio: int
    dim: int


def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward_with_stages(self, patches: torch.Tensor, n_h: int, n_w: int):
        stages: dict[str, Any] = {}
        x = self.patch_embed(patches)
        stages["patch_embedding"] = x
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        stages["rope_cos"] = cos.squeeze(1)
        stages["rope_sin"] = sin.squeeze(1)
        blocks = []
        for block in self.blocks:
            x = block(x, cos, sin)
            blocks.append(x)
        stages["blocks"] = blocks
        x = self.norm(x)
        stages["vision"] = x
        return x, stages


class Aligner(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward_with_stages(self, x: torch.Tensor, n_h: int, n_w: int):
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        unfolded = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        hidden = self.w1(unfolded)
        gelu = F.gelu(hidden)
        return self.w2(gelu), {"unfold": unfolded, "aligner_hidden": hidden, "gelu": gelu}


def tensor_values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().reshape(-1).tolist()


def initialize(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(
                    torch.empty(parameter.shape, dtype=parameter.dtype).uniform_(
                        0.75, 1.25, generator=generator
                    )
                )
            else:
                parameter.copy_(
                    torch.empty(parameter.shape, dtype=parameter.dtype).uniform_(
                        -0.2, 0.2, generator=generator
                    )
                )


def export_weights(vit: ViT, aligner: Aligner) -> dict[str, Any]:
    blocks = []
    for block in vit.blocks:
        blocks.append(
            {
                "norm1": tensor_values(block.norm1.weight),
                "qkv_weight": tensor_values(block.attn.wqkv.weight),
                "qkv_bias": tensor_values(block.attn.wqkv.bias),
                "out_weight": tensor_values(block.attn.wo.weight),
                "out_bias": tensor_values(block.attn.wo.bias),
                "norm2": tensor_values(block.norm2.weight),
                "mlp_w1": tensor_values(block.mlp.w1.weight),
                "mlp_w2": tensor_values(block.mlp.w2.weight),
            }
        )
    return {
        "patch_weight": tensor_values(vit.patch_embed.proj.weight),
        "patch_bias": tensor_values(vit.patch_embed.proj.bias),
        "blocks": blocks,
        "final_norm": tensor_values(vit.norm.weight),
        "aligner_w1_weight": tensor_values(aligner.w1.weight),
        "aligner_w1_bias": tensor_values(aligner.w1.bias),
        "aligner_w2_weight": tensor_values(aligner.w2.weight),
        "aligner_w2_bias": tensor_values(aligner.w2.bias),
    }


def make_fixture(name: str, args: Args, seed: int, grids: list[tuple[int, int]]) -> dict[str, Any]:
    torch.manual_seed(seed)
    vit = ViT(args)
    aligner = Aligner(args)
    initialize(vit, seed)
    initialize(aligner, seed + 1)
    cases = []
    patch_dim = 3 * args.vision_patch_size**2
    for case_index, (n_h, n_w) in enumerate(grids):
        generator = torch.Generator(device="cpu").manual_seed(seed + 100 + case_index)
        patches = torch.empty((n_h * n_w, patch_dim), dtype=torch.bfloat16).uniform_(
            -1.0, 1.0, generator=generator
        )
        vision, vision_stages = vit.forward_with_stages(patches, n_h, n_w)
        output, aligner_stages = aligner.forward_with_stages(vision, n_h, n_w)
        cases.append(
            {
                "name": f"{n_h}x{n_w}",
                "grid": [n_h, n_w],
                "patches": tensor_values(patches),
                "expected": {
                    "rope_cos": tensor_values(vision_stages["rope_cos"]),
                    "rope_sin": tensor_values(vision_stages["rope_sin"]),
                    "patch_embedding": tensor_values(vision_stages["patch_embedding"]),
                    "blocks": [tensor_values(x) for x in vision_stages["blocks"]],
                    "vision": tensor_values(vision),
                    "unfold": tensor_values(aligner_stages["unfold"]),
                    "aligner_hidden": tensor_values(aligner_stages["aligner_hidden"]),
                    "gelu": tensor_values(aligner_stages["gelu"]),
                    "output": tensor_values(output),
                },
            }
        )
    return {
        "name": name,
        "seed": seed,
        "config": {
            "patch_size": args.vision_patch_size,
            "hidden_size": args.vision_dim,
            "num_heads": args.vision_n_heads,
            "depth": args.vision_n_layers,
            "intermediate_size": args.vision_inter_dim,
            "rope_theta": args.vision_rope_theta,
            "downsample_ratio": args.vision_downsample_ratio,
            "output_size": args.dim,
            "norm_epsilon": 1.0e-6,
            "compute_dtype": "bf16",
        },
        "weights": export_weights(vit, aligner),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/parity/goldens/deepseek_v4_vision/goldens.json"),
    )
    options = parser.parse_args()

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        fixtures = [
            make_fixture(
                "heads2_depth2",
                Args(2, 8, 2, 2, 12, 10000.0, 3, 10),
                2411,
                [(2, 5), (3, 3)],
            ),
            make_fixture(
                "heads4_depth1",
                Args(2, 16, 4, 1, 20, 1234.0, 3, 12),
                2412,
                [(3, 4)],
            ),
        ]
        gelu_probe_input = torch.tensor(
            [-5.5, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.15625, 3.0, 5.5],
            dtype=torch.bfloat16,
        )
        gelu_probe_output = F.gelu(gelu_probe_input)
    finally:
        torch.set_default_dtype(previous_dtype)

    document = {
        "oracle": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "revision": REVISION,
        "source": SOURCE,
        "torch_version": torch.__version__,
        "gelu_probe": {
            "input": tensor_values(gelu_probe_input),
            "expected": tensor_values(gelu_probe_output),
        },
        "fixtures": fixtures,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
