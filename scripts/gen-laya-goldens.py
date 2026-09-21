#!/usr/bin/env python3
"""Emit tests/vllm/models/laya_goldens.inc — the Laya decision-head oracle.

Laya (convaiinnovations/laya) is a System 1 decision model that layers a
decision head on top of a ModernBERT-large encoder. This generator covers
the head only (Phase 2 of .agents/specs/laya.md): it feeds synthetic encoder
hidden states through the type_emb addition, two standard pre-norm
transformer encoder layers (ReLU), the scorer, and the act_head with
confidence features, and emits the golden (logits, act_logits) the C++ port
is gated against.

WHY A RESTATEMENT. The upstream head uses nn.TransformerEncoderLayer, whose
internal mask handling and F.multi_head_attention_forward dispatch obscure
the exact computation. This generator restates the forward in plain PyTorch
from a deterministic PRNG so no weight byte is checked in — the same
technique gen-modernbert-goldens.py and gen-deberta-v2-goldens.py use. Both
sides rebuild every weight from the same FNV-1a -> splitmix64 stream.

Upstream anchors (laya/common.py DecisionModel.forward):
  type_emb addition         common.py: forward, h = h + type_emb(qtype)
  head layers               nn.TransformerEncoderLayer(norm_first=True,
                            activation='relu'), 2 layers
  scorer                    Sequential(LayerNorm, Linear, GELU, Linear)
  act_head                  Sequential(Linear(d+4, 256), GELU, Linear(256, 2))
  confidence features       [top1, top1-top2, entropy/log(k), k/255]

THE THINGS THIS HEAD GETS WRONG QUIETLY are: (1) skipping the type_emb
addition — the model still runs and emits plausible logits; (2) using
post-norm instead of pre-norm in the head layers — plausible but wrong;
(3) using GELU instead of ReLU in the head FFN — plausible but wrong; (4)
skipping the key_padding_mask in head attention — padded positions silently
contribute. Each is gated by a perturbation test in the C++ test.

Usage:
    python3 scripts/gen-laya-goldens.py --out tests/vllm/models/laya_goldens.inc

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
    """The subset of DecisionModel config the head reads (reduced dims)."""

    def __init__(self) -> None:
        self.hidden_size = 24
        self.num_heads = 4
        self.head_layers = 2
        self.dim_ff = 48           # 2 * hidden (smaller than 4*hidden for speed)
        self.n_act = 2
        self.act_hidden = 16
        self.layer_norm_eps = 1e-5
        self.seq = 8
        self.k_max = 4
        self.qtype = 0             # choice

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


# -- weights -----------------------------------------------------------------


class Weights:
    """Checkpoint tensors in nn.Linear [out, in] orientation (same as C++).

    Weight names match the Laya checkpoint format exactly:
      type_emb.weight
      head.layers.{i}.self_attn.in_proj_weight / in_proj_bias
      head.layers.{i}.self_attn.out_proj.weight / out_proj.bias
      head.layers.{i}.linear1.weight / linear1.bias
      head.layers.{i}.linear2.weight / linear2.bias
      head.layers.{i}.norm1.weight / norm1.bias
      head.layers.{i}.norm2.weight / norm2.bias
      scorer.0.weight / scorer.0.bias (LayerNorm)
      scorer.1.weight / scorer.1.bias (Linear)
      scorer.3.weight / scorer.3.bias (Linear)
      act_head.0.weight / act_head.0.bias (Linear)
      act_head.2.weight / act_head.2.bias (Linear)
      temperature [3]
    """

    def __init__(self, cfg: Config) -> None:
        d = cfg.hidden_size
        ff = cfg.dim_ff
        act_in = d + 4
        self.t: dict[str, torch.Tensor] = {
            "type_emb.weight": param("type_emb.weight", (3, d), 0.3),
            "temperature": param("temperature", (3,), 0.3) + 1.0,
        }
        for i in range(cfg.head_layers):
            p = f"head.layers.{i}."
            self.t[p + "self_attn.in_proj_weight"] = param(p + "self_attn.in_proj_weight", (3 * d, d), 0.3)
            self.t[p + "self_attn.in_proj_bias"] = param(p + "self_attn.in_proj_bias", (3 * d,), 0.3)
            self.t[p + "self_attn.out_proj.weight"] = param(p + "self_attn.out_proj.weight", (d, d), 0.3)
            self.t[p + "self_attn.out_proj.bias"] = param(p + "self_attn.out_proj.bias", (d,), 0.3)
            self.t[p + "linear1.weight"] = param(p + "linear1.weight", (ff, d), 0.3)
            self.t[p + "linear1.bias"] = param(p + "linear1.bias", (ff,), 0.3)
            self.t[p + "linear2.weight"] = param(p + "linear2.weight", (d, ff), 0.3)
            self.t[p + "linear2.bias"] = param(p + "linear2.bias", (d,), 0.3)
            self.t[p + "norm1.weight"] = param_plus_one(p + "norm1.weight", (d,), 0.1)
            self.t[p + "norm1.bias"] = param(p + "norm1.bias", (d,), 0.1)
            self.t[p + "norm2.weight"] = param_plus_one(p + "norm2.weight", (d,), 0.1)
            self.t[p + "norm2.bias"] = param(p + "norm2.bias", (d,), 0.1)

        self.t["scorer.0.weight"] = param_plus_one("scorer.0.weight", (d,), 0.1)
        self.t["scorer.0.bias"] = param("scorer.0.bias", (d,), 0.1)
        self.t["scorer.1.weight"] = param("scorer.1.weight", (d, d), 0.3)
        self.t["scorer.1.bias"] = param("scorer.1.bias", (d,), 0.3)
        self.t["scorer.3.weight"] = param("scorer.3.weight", (1, d), 0.3)
        self.t["scorer.3.bias"] = param("scorer.3.bias", (1,), 0.3)

        self.t["act_head.0.weight"] = param("act_head.0.weight", (cfg.act_hidden, act_in), 0.3)
        self.t["act_head.0.bias"] = param("act_head.0.bias", (cfg.act_hidden,), 0.3)
        self.t["act_head.2.weight"] = param("act_head.2.weight", (cfg.n_act, cfg.act_hidden), 0.3)
        self.t["act_head.2.bias"] = param("act_head.2.bias", (cfg.n_act,), 0.3)


# -- helpers (restated from C++, matching byte-for-byte) ---------------------


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """torch.nn.LayerNorm with bias (biased 1/N variance)."""
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU (erf-based), matching nn.GELU() default."""
    return F.gelu(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """nn.Linear forward: y = x @ w.T + b. w is [out, in]."""
    return F.linear(x, w, b)


# -- the model, restated from laya/common.py DecisionModel.forward ----------


def head_layer_forward(
    cfg: Config, w: Weights, layer: int,
    h: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    """One nn.TransformerEncoderLayer (norm_first=True, ReLU, batch_first).

    h: [seq, hidden]
    attention_mask: [seq] (1=valid, 0=padding)
    """
    d = cfg.hidden_size
    heads = cfg.num_heads
    hd = cfg.head_dim
    seq = h.shape[0]
    eps = cfg.layer_norm_eps
    p = f"head.layers.{layer}."

    # ── Attention pre-norm ──────────────────────────────────────────────
    normed = layer_norm(h, w.t[p + "norm1.weight"], w.t[p + "norm1.bias"], eps)

    # Combined QKV projection
    qkv = linear(normed, w.t[p + "self_attn.in_proj_weight"], w.t[p + "self_attn.in_proj_bias"])
    q, k, v = qkv.split(d, dim=-1)  # each [seq, hidden]

    # Reshape to [seq, heads, head_dim]
    q = q.view(seq, heads, hd)
    k = k.view(seq, heads, hd)
    v = v.view(seq, heads, hd)

    # Per-head scores: [heads, seq, seq]
    scale = 1.0 / math.sqrt(hd)
    scores = torch.matmul(q.permute(1, 0, 2), k.permute(1, 2, 0)) * scale

    # key_padding_mask: attention_mask == 0 → masked
    mask = (attention_mask == 0)  # [seq], True where padding
    # Expand to [1, seq] → broadcast over heads and query positions
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e30)

    # Softmax over keys
    probs = F.softmax(scores, dim=-1)  # [heads, seq, seq]

    # Weighted sum of V
    attn_out = torch.matmul(probs, v.permute(1, 0, 2))  # [heads, seq, head_dim]
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(seq, d)  # [seq, hidden]

    # Output projection + residual
    proj = linear(attn_out, w.t[p + "self_attn.out_proj.weight"], w.t[p + "self_attn.out_proj.bias"])
    h = h + proj

    # ── FFN pre-norm + ReLU ─────────────────────────────────────────────
    normed = layer_norm(h, w.t[p + "norm2.weight"], w.t[p + "norm2.bias"], eps)
    ff_inter = linear(normed, w.t[p + "linear1.weight"], w.t[p + "linear1.bias"])
    ff_inter = relu(ff_inter)
    proj = linear(ff_inter, w.t[p + "linear2.weight"], w.t[p + "linear2.bias"])
    h = h + proj

    return h


def forward(
    cfg: Config, w: Weights,
    hidden_states: torch.Tensor, attention_mask: torch.Tensor,
    marker_pos: list[int], marker_mask: list[int], qtype: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decision head forward (batch=1).

    hidden_states: [seq, hidden] (encoder output)
    attention_mask: [seq] (1=valid, 0=padding)
    marker_pos: [k_max] (positions of MASK tokens; negative clamped to 0)
    marker_mask: [k_max] (1=valid, 0=padding)
    qtype: 0=choice, 1=score, 2=noul
    Returns: (logits [k_max], act_logits [n_act])
    """
    d = cfg.hidden_size
    eps = cfg.layer_norm_eps
    k_max = len(marker_pos)

    # 1. type_emb addition
    h = hidden_states.clone()
    h = h + w.t["type_emb.weight"][qtype]  # broadcast over all positions

    # 2. Head layers
    for layer in range(cfg.head_layers):
        h = head_layer_forward(cfg, w, layer, h, attention_mask)

    # 3. Gather marker positions
    pos_clamped = [max(0, p) for p in marker_pos]
    m = h[pos_clamped]  # [k_max, hidden]

    # 4. Scorer: LayerNorm → Linear → GELU → Linear(1)
    normed = layer_norm(m, w.t["scorer.0.weight"], w.t["scorer.0.bias"], eps)
    inter = linear(normed, w.t["scorer.1.weight"], w.t["scorer.1.bias"])
    inter = gelu(inter)
    logits = linear(inter, w.t["scorer.3.weight"], w.t["scorer.3.bias"]).squeeze(-1)  # [k_max]

    # 5. Mask invalid markers
    for k in range(k_max):
        if marker_mask[k] == 0:
            logits[k] = -1e4

    # 6. Confidence features from softmax(logits)
    p = F.softmax(logits, dim=-1)  # [k_max]

    # k = max(2, sum(marker_mask))
    k_count = max(2, sum(marker_mask))

    # entropy / log(k)
    log_pk = torch.log(p.clamp(min=1e-9))
    ent = -(p * log_pk).sum() / math.log(float(k_count))

    # top2
    topk_vals = p.topk(2).values
    top1 = topk_vals[0].item()
    top2 = topk_vals[1].item()

    feats = torch.tensor([top1, top1 - top2, ent, float(k_count) / 255.0], dtype=torch.float32)

    # 7. CLS pool + act_head
    pooled = h[0]  # [hidden]

    act_in = torch.cat([pooled, feats])  # [hidden + 4]
    act_inter = linear(act_in, w.t["act_head.0.weight"], w.t["act_head.0.bias"])
    act_inter = gelu(act_inter)
    act_logits = linear(act_inter, w.t["act_head.2.weight"], w.t["act_head.2.bias"])  # [n_act]

    return logits, act_logits


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

    # Synthetic encoder hidden states [seq, hidden]
    hidden_states = param("input.hidden_states", (cfg.seq, cfg.hidden_size), 0.5)
    attention_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.int64)
    marker_pos = [1, 3, 0, 0]      # 2 valid markers at pos 1, 3
    marker_mask = [1, 1, 0, 0]     # 2 valid, 2 padding
    qtype = 0

    logits, act_logits = forward(cfg, w, hidden_states, attention_mask,
                                 marker_pos, marker_mask, qtype)

    path = Path(args.out)
    with path.open("w", encoding="utf-8") as out:
        out.write("// GENERATED by scripts/gen-laya-goldens.py -- do not edit.\n")
        out.write("//\n")
        out.write("// Laya decision head goldens: a restatement of\n")
        out.write("// laya/common.py DecisionModel.forward (convaiinnovations/laya),\n")
        out.write("// executed at reduced dimensions on CPU. Weights are rebuilt on\n")
        out.write("// both sides from the same FNV-1a -> splitmix64 stream, so no\n")
        out.write("// weight byte is checked in. The type_emb addition, pre-norm\n")
        out.write("// head layers with ReLU, the scorer, the key_padding_mask, and\n")
        out.write("// the confidence features are features this port has to get\n")
        out.write("// right; each is gated by a perturbation test rather than left\n")
        out.write("// to be implied by the forward.\n")
        out.write("#pragma once\n\n#include <cstdint>\n\n")
        out.write("namespace laya_goldens {\n\n")
        out.write(f"inline constexpr int64_t kHidden = {cfg.hidden_size};\n")
        out.write(f"inline constexpr int64_t kHeads = {cfg.num_heads};\n")
        out.write(f"inline constexpr int64_t kHeadLayers = {cfg.head_layers};\n")
        out.write(f"inline constexpr int64_t kDimFf = {cfg.dim_ff};\n")
        out.write(f"inline constexpr int64_t kNAct = {cfg.n_act};\n")
        out.write(f"inline constexpr int64_t kActHidden = {cfg.act_hidden};\n")
        out.write(f"inline constexpr int64_t kSeq = {cfg.seq};\n")
        out.write(f"inline constexpr int64_t kKMax = {cfg.k_max};\n")
        out.write(f"inline constexpr int64_t kQtype = {qtype};\n")
        out.write(f"inline constexpr double kLayerNormEps = {cfg.layer_norm_eps!r};\n\n")
        emit_i64(out, "kAttentionMask", attention_mask.tolist())
        emit_i64(out, "kMarkerPos", marker_pos)
        emit_i64(out, "kMarkerMask", marker_mask)
        emit_f32(out, "kHiddenStates", hidden_states.detach().numpy())
        emit_f32(out, "kLogits", logits.detach().numpy())
        emit_f32(out, "kActLogits", act_logits.detach().numpy())
        out.write("}  // namespace laya_goldens\n")

    print(f"wrote {path} (seq={cfg.seq}, hidden={cfg.hidden_size}, "
          f"head_layers={cfg.head_layers}, k_max={cfg.k_max})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
