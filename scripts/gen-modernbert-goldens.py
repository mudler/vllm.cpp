#!/usr/bin/env python3
"""Emit tests/vllm/models/modernbert_goldens.inc — the ModernBERT encoder oracle.

Laya's backbone is a ModernBERT-large encoder (answerdotai/ModernBERT-large,
see .agents/specs/laya.md, Phase 1). This generator produces the goldens the
C++ port is gated against.

WHY A RESTATEMENT. vLLM has a modernbert.py at the pin, but this project's
pattern restates the model in plain PyTorch from a deterministic PRNG so no
weight byte is checked in — the same technique gen-deberta-v2-goldens.py and
gen-gpt2-goldens.py use. Both sides rebuild every weight from the same
FNV-1a -> splitmix64 stream.

Upstream anchors (vllm/model_executor/models/modernbert.py @ pin e126687a9a,
which mirrors transformers/models/modernbert/modeling_modernbert.py):
  ModernBertEmbeddings          modernbert.py:35-61
  ModernBertAttention           modernbert.py:64-145
  ModernBertMLP                 modernbert.py:148-165
  ModernBertLayer               modernbert.py:168-206
  ModernBertModel               modernbert.py:236-300

THE THINGS THIS ARCHITECTURE GETS WRONG QUIETLY are: (1) swapping the local
and global RoPE theta — the model still runs and emits plausible hidden states;
(2) omitting the sliding window — the local layers silently attend to all
positions; (3) applying LayerNorm before attention on layer 0 where the
upstream uses Identity. Each is gated by a perturbation test in the C++ test.

Usage:
    python3 scripts/gen-modernbert-goldens.py --out tests/vllm/models/modernbert_goldens.inc

Needs torch + numpy (CPU only).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_MASK64 = (1 << 64) - 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for byte in name.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & _MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def rand(name: str, count: int) -> np.ndarray:
    """`count` values uniform in [-1, 1), reproducible from `name` alone."""
    seed = fnv1a64(name)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = splitmix64((seed + i) & _MASK64)
        out[i] = ((u >> 11) * (2.0**-53)) * 2.0 - 1.0
    return out


def param(name: str, shape, scale: float) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    values = rand(name, count) * scale
    return torch.tensor(values, dtype=torch.float64).reshape(shape).to(torch.float32)


def param_plus_one(name: str, shape, scale: float) -> torch.Tensor:
    return param(name, shape, scale) + 1.0


# -- config ------------------------------------------------------------------


class Config:
    """The subset of ModernBertConfig the encoder reads."""

    def __init__(self) -> None:
        self.vocab_size = 33
        self.hidden_size = 24
        self.num_hidden_layers = 6
        self.num_attention_heads = 4
        self.head_dim = 6        # explicit, like ModernBERT-large
        self.intermediate_size = 48
        self.local_attention = 8   # sliding window = 4
        self.global_attn_every_n_layers = 3  # layers 0, 3 global; 1, 2, 4, 5 local
        self.local_rope_theta = 10000.0
        self.global_rope_theta = 160000.0
        self.layer_norm_eps = 1e-5

    def is_global_layer(self, layer: int) -> bool:
        return self.global_attn_every_n_layers > 0 and layer % self.global_attn_every_n_layers == 0

    @property
    def sliding_window(self) -> int:
        return self.local_attention // 2 if self.local_attention > 0 else 0


# -- weights -----------------------------------------------------------------


class Weights:
    """Checkpoint tensors in nn.Linear [out, in] orientation (same as C++).

    Weight names match the Laya checkpoint format exactly:
      encoder.embeddings.tok_embeddings.weight
      encoder.embeddings.norm.weight
      encoder.final_norm.weight
      encoder.layers.{i}.attn.Wqkv.weight
      encoder.layers.{i}.attn.Wo.weight
      encoder.layers.{i}.attn_norm.weight   (layers 1+, NOT layer 0)
      encoder.layers.{i}.mlp.Wi.weight
      encoder.layers.{i}.mlp.Wo.weight
      encoder.layers.{i}.mlp_norm.weight
    """

    def __init__(self, cfg: Config) -> None:
        h = cfg.hidden_size
        inter = cfg.intermediate_size
        self.t: dict[str, torch.Tensor] = {
            "encoder.embeddings.tok_embeddings.weight": param(
                "encoder.embeddings.tok_embeddings.weight", (cfg.vocab_size, h), 0.5),
            "encoder.embeddings.norm.weight": param_plus_one(
                "encoder.embeddings.norm.weight", (h,), 0.1),
            "encoder.final_norm.weight": param_plus_one(
                "encoder.final_norm.weight", (h,), 0.1),
        }
        for i in range(cfg.num_hidden_layers):
            p = f"encoder.layers.{i}."
            self.t[p + "attn.Wqkv.weight"] = param(p + "attn.Wqkv.weight", (3 * h, h), 0.3)
            self.t[p + "attn.Wo.weight"] = param(p + "attn.Wo.weight", (h, h), 0.3)
            if i > 0:
                self.t[p + "attn_norm.weight"] = param_plus_one(p + "attn_norm.weight", (h,), 0.1)
            self.t[p + "mlp.Wi.weight"] = param(p + "mlp.Wi.weight", (2 * inter, h), 0.3)
            self.t[p + "mlp.Wo.weight"] = param(p + "mlp.Wo.weight", (h, inter), 0.3)
            self.t[p + "mlp_norm.weight"] = param_plus_one(p + "mlp_norm.weight", (h,), 0.1)


# -- the model, restated from vLLM/transformers ------------------------------


def layer_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """torch.nn.LayerNorm without bias (ModernBERT norm_bias=False)."""
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=None, eps=eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU (erf-based), matching ACT2FN['gelu'] in HF."""
    return F.gelu(x)


def apply_rope(x: torch.Tensor, pos: int, theta: float, head_dim: int) -> torch.Tensor:
    """Rotate-half RoPE (GPT-J convention), matching transformers RotaryEmbedding.

    x: [..., head_dim]
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / head_dim))
    angles = float(pos) * freqs
    cos = torch.cos(angles).to(torch.float32)   # [half]
    sin = torch.sin(angles).to(torch.float32)
    cos_full = torch.cat([cos, cos], dim=-1)     # [head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * cos_full + rotate_half * sin_full


def attention(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """ModernBertAttention @ vLLM e126687a9a.

    x: [seq, hidden] — already normed (or Identity for layer 0)
    Returns: [seq, hidden] — attention output after Wo, BEFORE residual add
    """
    h = cfg.hidden_size
    heads = cfg.num_attention_heads
    hd = cfg.head_dim
    seq = x.shape[0]
    is_global = cfg.is_global_layer(layer)
    theta = cfg.global_rope_theta if is_global else cfg.local_rope_theta
    sw = cfg.sliding_window
    p = f"encoder.layers.{layer}."

    # Combined QKV projection (no bias)
    qkv = F.linear(x, w.t[p + "attn.Wqkv.weight"])  # [seq, 3*h]
    q, k, v = qkv.split(h, dim=-1)                   # each [seq, h]

    # Reshape to [seq, heads, head_dim]
    q = q.view(seq, heads, hd)
    k = k.view(seq, heads, hd)
    v = v.view(seq, heads, hd)

    # Apply RoPE to Q and K (per head, per position)
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    for t in range(seq):
        for hd_idx in range(heads):
            q_rot[t, hd_idx] = apply_rope(q[t, hd_idx], t, theta, hd)
            k_rot[t, hd_idx] = apply_rope(k[t, hd_idx], t, theta, hd)

    # Per-head scores: [heads, seq, seq]
    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q_rot.permute(1, 0, 2), k_rot.permute(1, 2, 0)) * scale  # [heads, seq, seq]

    # Sliding window mask for local layers
    if not is_global and sw > 0:
        positions = torch.arange(seq)
        dist = torch.abs(positions[:, None] - positions[None, :])  # [seq, seq]
        mask = dist > sw  # True where masked
        scores = scores.masked_fill(mask.unsqueeze(0), -1e30)

    # Softmax (bidirectional, no causal mask)
    probs = F.softmax(scores, dim=-1)  # [heads, seq, seq]

    # Weighted sum of V
    attn_out = torch.matmul(probs, v.permute(1, 0, 2))  # [heads, seq, head_dim]
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(seq, h)  # [seq, hidden]

    # Output projection Wo (no bias)
    out = F.linear(attn_out, w.t[p + "attn.Wo.weight"])  # [seq, hidden]
    return out


def mlp(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """ModernBertMLP (GeGLU) @ vLLM e126687a9a.

    x: [seq, hidden] — already normed
    Returns: [seq, hidden] — MLP output after Wo, BEFORE residual add
    """
    inter = cfg.intermediate_size
    p = f"encoder.layers.{layer}."

    # Gate-up projection: Wi [2*inter, hidden] -> split input + gate
    gate_up = F.linear(x, w.t[p + "mlp.Wi.weight"])  # [seq, 2*inter]
    inp, gate = gate_up.split(inter, dim=-1)          # each [seq, inter]

    # GeGLU: GELU(input) * gate
    act = gelu(inp) * gate

    # Down projection Wo [hidden, inter] (no bias)
    out = F.linear(act, w.t[p + "mlp.Wo.weight"])  # [seq, hidden]
    return out


def forward(cfg: Config, w: Weights, ids: list[int]) -> torch.Tensor:
    """Full encoder forward: embeddings -> N layers -> hidden states [seq, hidden]."""
    h = cfg.hidden_size
    eps = cfg.layer_norm_eps

    # Embeddings: token lookup -> LayerNorm (no bias, no position embeddings)
    x = w.t["encoder.embeddings.tok_embeddings.weight"][torch.tensor(ids)]  # [seq, hidden]
    x = layer_norm(x, w.t["encoder.embeddings.norm.weight"], eps)

    for layer in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{layer}."

        # Attention pre-norm: Identity for layer 0, LayerNorm for layers 1+
        if layer == 0:
            normed = x
        else:
            normed = layer_norm(x, w.t[p + "attn_norm.weight"], eps)

        attn_out = attention(cfg, w, layer, normed)
        x = x + attn_out  # residual

        # MLP pre-norm
        normed = layer_norm(x, w.t[p + "mlp_norm.weight"], eps)
        mlp_out = mlp(cfg, w, layer, normed)
        x = x + mlp_out  # residual

    # Final norm
    x = layer_norm(x, w.t["encoder.final_norm.weight"], eps)
    return x


# -- emit --------------------------------------------------------------------


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}F" for v in flat[i : i + 6])
        out.write(f"    {chunk},\n")
    out.write("};\n\n")


def emit_i64(out, name: str, values) -> None:
    out.write(f"inline constexpr int64_t {name}[] = {{\n")
    joined = ", ".join(str(int(v)) for v in values)
    out.write(f"    {joined},\n")
    out.write("};\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = Config()
    w = Weights(cfg)

    ids = [3, 17, 0, 31, 8, 8, 12, 5, 20, 1]

    hidden = forward(cfg, w, ids)

    path = Path(args.out)
    with path.open("w", encoding="utf-8") as out:
        out.write("// GENERATED by scripts/gen-modernbert-goldens.py -- do not edit.\n")
        out.write("//\n")
        out.write("// ModernBERT encoder goldens: a restatement of vLLM\n")
        out.write("// model_executor/models/modernbert.py @ pin e126687a9a (mirrors\n")
        out.write("// transformers/models/modernbert/modeling_modernbert.py),\n")
        out.write("// executed at reduced dimensions on CPU. Weights are rebuilt on\n")
        out.write("// both sides from the same FNV-1a -> splitmix64 stream, so no\n")
        out.write("// weight byte is checked in. The dual RoPE theta, the sliding\n")
        out.write("// window, and the layer-0 Identity norm are features this port\n")
        out.write("// has to get right; each is gated by a perturbation test rather\n")
        out.write("// than left to be implied by the forward.\n")
        out.write("#pragma once\n\n#include <cstdint>\n\n")
        out.write("namespace modernbert_goldens {\n\n")
        out.write(f"inline constexpr int64_t kVocab = {cfg.vocab_size};\n")
        out.write(f"inline constexpr int64_t kHidden = {cfg.hidden_size};\n")
        out.write(f"inline constexpr int64_t kLayers = {cfg.num_hidden_layers};\n")
        out.write(f"inline constexpr int64_t kHeads = {cfg.num_attention_heads};\n")
        out.write(f"inline constexpr int64_t kHeadDim = {cfg.head_dim};\n")
        out.write(f"inline constexpr int64_t kInter = {cfg.intermediate_size};\n")
        out.write(f"inline constexpr int64_t kLocalAttention = {cfg.local_attention};\n")
        out.write(f"inline constexpr int64_t kGlobalAttnEveryN = {cfg.global_attn_every_n_layers};\n")
        out.write(f"inline constexpr double kLocalRopeTheta = {cfg.local_rope_theta!r};\n")
        out.write(f"inline constexpr double kGlobalRopeTheta = {cfg.global_rope_theta!r};\n")
        out.write(f"inline constexpr double kLayerNormEps = {cfg.layer_norm_eps!r};\n\n")
        emit_i64(out, "kInputIds", ids)
        emit_f32(out, "kHiddenStates", hidden.detach().numpy())
        out.write("}  // namespace modernbert_goldens\n")

    print(f"wrote {path} (seq={len(ids)}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
