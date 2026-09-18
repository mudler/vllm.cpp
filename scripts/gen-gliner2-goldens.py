#!/usr/bin/env python3
"""Emit tests/vllm/models/gliner2_goldens.inc — the GLiNER2 boundary head oracle.

GLiNER2.5-multi-v1's NER head is a boundary architecture (BoundaryEncoder +
BoundaryQueryHead), ported from the GLiNER2 library (github.com/fastino-ai/
GLiNER2). See .agents/specs/gliner2.5.md, Phase 3.

This generator restates the boundary encoder and query head in plain PyTorch
from a deterministic FNV-1a -> splitmix64 PRNG, the same technique
gen-deberta-v2-goldens.py uses. Both sides rebuild every weight from the same
stream, so no weight byte is checked in.

Upstream anchors (GLiNER2 library, boundary_head package):
  BoundaryEncoder.forward          boundary_head/model.py:BoundaryEncoder
  BoundaryQueryHead.forward        boundary_head/heads.py:BoundaryQueryHead
  BoundaryAttentionBlock           boundary_head/model.py:BoundaryAttentionBlock
  ResidualSwiGLu                   boundary_head/model.py:ResidualSwiGLu
  MASK_LOGIT                       boundary_head/constants.py

THE THING THIS ARCHITECTURE GETS WRONG QUIETLY is dropping the sliding window
mask in the boundary attention. Without it the model still runs and emits
plausible boundary states. The window is gated by a perturbation test in the
C++ test rather than left to be implied by the forward.

Usage:
    python3 scripts/gen-gliner2-goldens.py --out tests/vllm/models/gliner2_goldens.inc

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


# -- config -----------------------------------------------------------------


class Config:
    H = 12          # encoder hidden (matches deberta_v2 goldens)
    d = 8           # boundary_dim
    heads = 2       # boundary_attention_heads
    attn_layers = 1 # boundary_attention_layers
    attn_window = 4 # boundary_attention_window
    refine_layers = 1  # boundary_refinement_layers
    ffn_mult = 2.0
    ln_eps = 1e-5
    seq_len = 5
    num_queries = 2

    @property
    def ffn_dim(self) -> int:
        return int(self.d * self.ffn_mult)

    @property
    def head_dim(self) -> int:
        return self.d // self.heads

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.d)


# -- weights ----------------------------------------------------------------


def build_weights(cfg: Config) -> dict[str, torch.Tensor]:
    H, d = cfg.H, cfg.d
    ffn = cfg.ffn_dim
    half_ffn = ffn // 2
    p = "boundary_head.boundary_encoder."

    t: dict[str, torch.Tensor] = {
        p + "left_projection.weight": param(p + "left_projection.weight", (d, H), 0.3),
        p + "left_projection.bias": param(p + "left_projection.bias", (d,), 0.2),
        p + "right_projection.weight": param(p + "right_projection.weight", (d, H), 0.3),
        p + "right_projection.bias": param(p + "right_projection.bias", (d,), 0.2),
        p + "output_projection.weight": param(p + "output_projection.weight", (d, 2 * d), 0.3),
        p + "output_projection.bias": param(p + "output_projection.bias", (d,), 0.2),
        p + "layer_norm.weight": param_plus_one(p + "layer_norm.weight", (d,), 0.1),
        p + "layer_norm.bias": param(p + "layer_norm.bias", (d,), 0.1),
        p + "bos_state": param(p + "bos_state", (H,), 0.3),
        p + "eos_state": param(p + "eos_state", (H,), 0.3),
    }

    for i in range(cfg.attn_layers):
        b = p + f"attention_blocks.{i}."
        t.update({
            b + "norm.weight": param_plus_one(b + "norm.weight", (d,), 0.1),
            b + "norm.bias": param(b + "norm.bias", (d,), 0.1),
            b + "qkv_projection.weight": param(b + "qkv_projection.weight", (3 * d, d), 0.3),
            b + "qkv_projection.bias": param(b + "qkv_projection.bias", (3 * d,), 0.2),
            b + "output_projection.weight": param(b + "output_projection.weight", (d, d), 0.3),
            b + "output_projection.bias": param(b + "output_projection.bias", (d,), 0.2),
        })

    for i in range(cfg.refine_layers):
        b = p + f"refinement_blocks.{i}."
        t.update({
            b + "norm.weight": param_plus_one(b + "norm.weight", (d,), 0.1),
            b + "norm.bias": param(b + "norm.bias", (d,), 0.1),
            b + "input_projection.weight": param(b + "input_projection.weight", (ffn, d), 0.3),
            b + "input_projection.bias": param(b + "input_projection.bias", (ffn,), 0.2),
            b + "output_projection.weight": param(b + "output_projection.weight", (d, half_ffn), 0.3),
            b + "output_projection.bias": param(b + "output_projection.bias", (d,), 0.2),
        })

    q = "boundary_head.boundary_query_head."
    t.update({
        q + "start_boundary_projection.weight": param(q + "start_boundary_projection.weight", (d, d), 0.3),
        q + "start_boundary_projection.bias": param(q + "start_boundary_projection.bias", (d,), 0.2),
        q + "end_boundary_projection.weight": param(q + "end_boundary_projection.weight", (d, d), 0.3),
        q + "end_boundary_projection.bias": param(q + "end_boundary_projection.bias", (d,), 0.2),
        q + "start_query_projection.weight": param(q + "start_query_projection.weight", (d, H), 0.3),
        q + "start_query_projection.bias": param(q + "start_query_projection.bias", (d,), 0.2),
        q + "end_query_projection.weight": param(q + "end_query_projection.weight", (d, H), 0.3),
        q + "end_query_projection.bias": param(q + "end_query_projection.bias", (d,), 0.2),
        q + "inside_text_projection.weight": param(q + "inside_text_projection.weight", (d, H), 0.3),
        q + "inside_text_projection.bias": param(q + "inside_text_projection.bias", (d,), 0.2),
        q + "inside_query_projection.weight": param(q + "inside_query_projection.weight", (d, H), 0.3),
        q + "inside_query_projection.bias": param(q + "inside_query_projection.bias", (d,), 0.2),
    })

    return t


# -- the model, restated from GLiNER2 --------------------------------------


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def boundary_encoder_forward(cfg: Config, t: dict[str, torch.Tensor],
                             text_states: torch.Tensor) -> torch.Tensor:
    """BoundaryEncoder.forward restated.

    text_states: [L, H] -> boundary_states: [L+1, d]
    """
    H, d = cfg.H, cfg.d
    B = cfg.seq_len + 1
    heads = cfg.heads
    hd = cfg.head_dim
    window = cfg.attn_window
    eps = cfg.ln_eps
    p = "boundary_head.boundary_encoder."

    # Build left/right states [B, H]
    left_states = torch.zeros(B, H)
    right_states = torch.zeros(B, H)
    for i in range(B):
        if i == 0:
            left_states[i] = t[p + "bos_state"]
        else:
            left_states[i] = text_states[i - 1]
        if i == cfg.seq_len:
            right_states[i] = t[p + "eos_state"]
        else:
            right_states[i] = text_states[i]

    # Projections
    left_proj = F.linear(left_states, t[p + "left_projection.weight"],
                         t[p + "left_projection.bias"])
    right_proj = F.linear(right_states, t[p + "right_projection.weight"],
                          t[p + "right_projection.bias"])
    concat = torch.cat([left_proj, right_proj], dim=-1)  # [B, 2d]
    out = F.linear(concat, t[p + "output_projection.weight"],
                   t[p + "output_projection.bias"])
    out = layer_norm(out, t[p + "layer_norm.weight"],
                     t[p + "layer_norm.bias"], eps)

    # Attention blocks
    for blk_i in range(cfg.attn_layers):
        b = p + f"attention_blocks.{blk_i}."
        normed = layer_norm(out, t[b + "norm.weight"], t[b + "norm.bias"], eps)
        qkv = F.linear(normed, t[b + "qkv_projection.weight"],
                       t[b + "qkv_projection.bias"])  # [B, 3d]
        q = qkv[:, :d].reshape(B, heads, hd)
        k = qkv[:, d:2 * d].reshape(B, heads, hd)
        v = qkv[:, 2 * d:3 * d].reshape(B, heads, hd)

        attn_scale = 1.0 / math.sqrt(hd)
        # scores: [heads, B, B]
        scores = torch.einsum("ibd,jbd->bij", q, k) * attn_scale
        # Sliding window mask
        for i in range(B):
            for j in range(B):
                if abs(i - j) > window // 2:
                    scores[:, i, j] = float("-inf")
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.einsum("bij,jbd->ibd", probs, v).reshape(B, d)

        proj = F.linear(attn_out, t[b + "output_projection.weight"],
                        t[b + "output_projection.bias"])
        out = out + proj

    # Refinement blocks (pre-norm ResidualSwiGLU)
    half_ffn = cfg.ffn_dim // 2
    for blk_i in range(cfg.refine_layers):
        b = p + f"refinement_blocks.{blk_i}."
        normed = layer_norm(out, t[b + "norm.weight"], t[b + "norm.bias"], eps)
        ffn = F.linear(normed, t[b + "input_projection.weight"],
                       t[b + "input_projection.bias"])  # [B, ffn]
        gate = silu(ffn[:, :half_ffn])
        val = ffn[:, half_ffn:]
        gated = gate * val  # [B, half_ffn]
        proj = F.linear(gated, t[b + "output_projection.weight"],
                        t[b + "output_projection.bias"])
        out = out + proj

    return out  # [B, d]


def boundary_query_head_forward(cfg: Config, t: dict[str, torch.Tensor],
                                boundary_states: torch.Tensor,
                                text_states: torch.Tensor,
                                query_states: torch.Tensor) -> dict:
    """BoundaryQueryHead.forward restated.

    boundary_states: [L+1, d]
    text_states: [L, H]
    query_states: [Q, H]
    Returns dict with start_logits, end_logits, inside_logits, inside_prefix.
    """
    H, d = cfg.H, cfg.d
    B = cfg.seq_len + 1
    L = cfg.seq_len
    Q = cfg.num_queries
    scale = cfg.scale
    q = "boundary_head.boundary_query_head."

    start_boundary = F.linear(boundary_states, t[q + "start_boundary_projection.weight"],
                              t[q + "start_boundary_projection.bias"])
    end_boundary = F.linear(boundary_states, t[q + "end_boundary_projection.weight"],
                            t[q + "end_boundary_projection.bias"])
    start_query = F.linear(query_states, t[q + "start_query_projection.weight"],
                           t[q + "start_query_projection.bias"])
    end_query = F.linear(query_states, t[q + "end_query_projection.weight"],
                         t[q + "end_query_projection.bias"])
    inside_text = F.linear(text_states, t[q + "inside_text_projection.weight"],
                           t[q + "inside_text_projection.bias"])
    inside_query = F.linear(query_states, t[q + "inside_query_projection.weight"],
                            t[q + "inside_query_projection.bias"])

    start_logits = (start_query @ start_boundary.T) * scale  # [Q, B]
    end_logits = (end_query @ end_boundary.T) * scale  # [Q, B]
    inside_logits = (inside_query @ inside_text.T) * scale  # [Q, L]

    # Inside prefix: cumsum, fp32, mean-centered per query
    inside_prefix = torch.zeros(Q, B)
    for qi in range(Q):
        prefix_sum = 0.0
        for ti in range(L):
            prefix_sum += float(inside_logits[qi, ti])
            inside_prefix[qi, ti + 1] = prefix_sum
        mean = float(inside_prefix[qi].mean())
        inside_prefix[qi] -= mean

    return {
        "start_logits": start_logits,
        "end_logits": end_logits,
        "inside_logits": inside_logits,
        "inside_prefix": inside_prefix,
    }


# -- emit -------------------------------------------------------------------


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    out.write(f"inline constexpr float {name}[] = {{\n")
    for i in range(0, len(flat), 6):
        chunk = ", ".join(f"{v:.9e}F" for v in flat[i : i + 6])
        out.write(f"    {chunk},\n")
    out.write("};\n\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = Config()
    t = build_weights(cfg)

    text_states = param("input.text_states", (cfg.seq_len, cfg.H), 0.5)
    query_states = param("input.query_states", (cfg.num_queries, cfg.H), 0.5)

    boundary_states = boundary_encoder_forward(cfg, t, text_states)
    marg = boundary_query_head_forward(cfg, t, boundary_states, text_states,
                                       query_states)

    path = Path(args.out)
    with path.open("w", encoding="utf-8") as out:
        out.write("// GENERATED by scripts/gen-gliner2-goldens.py -- do not edit.\n")
        out.write("//\n")
        out.write("// GLiNER2 boundary head goldens: a restatement of the GLiNER2\n")
        out.write("// library boundary architecture (BoundaryEncoder + BoundaryQueryHead),\n")
        out.write("// executed at reduced dimensions on CPU. Weights and inputs are rebuilt\n")
        out.write("// on both sides from the same FNV-1a -> splitmix64 stream, so no weight\n")
        out.write("// byte is checked in. The sliding window mask in the boundary attention\n")
        out.write("// is the feature this port has to get right; it is gated by a perturbation\n")
        out.write("// test rather than left to be implied by the forward.\n")
        out.write("#pragma once\n\n#include <cstdint>\n\n")
        out.write("namespace gliner2_goldens {\n\n")
        out.write(f"inline constexpr int64_t kHidden = {cfg.H};\n")
        out.write(f"inline constexpr int64_t kBoundaryDim = {cfg.d};\n")
        out.write(f"inline constexpr int64_t kHeads = {cfg.heads};\n")
        out.write(f"inline constexpr int64_t kAttnLayers = {cfg.attn_layers};\n")
        out.write(f"inline constexpr int64_t kAttnWindow = {cfg.attn_window};\n")
        out.write(f"inline constexpr int64_t kRefineLayers = {cfg.refine_layers};\n")
        out.write(f"inline constexpr int64_t kFfnDim = {cfg.ffn_dim};\n")
        out.write(f"inline constexpr double kLayerNormEps = {cfg.ln_eps!r};\n")
        out.write(f"inline constexpr int64_t kSeqLen = {cfg.seq_len};\n")
        out.write(f"inline constexpr int64_t kNumQueries = {cfg.num_queries};\n")
        out.write(f"inline constexpr int64_t kB = {cfg.seq_len + 1};\n\n")
        emit_f32(out, "kTextStates", text_states.detach().numpy())
        emit_f32(out, "kQueryStates", query_states.detach().numpy())
        emit_f32(out, "kBoundaryStates", boundary_states.detach().numpy())
        emit_f32(out, "kStartLogits", marg["start_logits"].detach().numpy())
        emit_f32(out, "kEndLogits", marg["end_logits"].detach().numpy())
        emit_f32(out, "kInsideLogits", marg["inside_logits"].detach().numpy())
        emit_f32(out, "kInsidePrefix", marg["inside_prefix"].detach().numpy())
        out.write("}  // namespace gliner2_goldens\n")

    print(f"wrote {path} (seq={cfg.seq_len}, d={cfg.d}, H={cfg.H}, "
          f"heads={cfg.heads}, attn_layers={cfg.attn_layers}, "
          f"refine_layers={cfg.refine_layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
