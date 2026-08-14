#!/usr/bin/env python3
"""Emit tests/vllm/models/gpt2_goldens.inc — the GPT-2 backbone parity oracle.

IndexTTS-2.5's stage-0 talker is a GPT-2 autoregressive backbone (see
.agents/specs/indextts-2-5.md, W2). This generator produces the goldens the C++
port is gated against.

WHY A RESTATEMENT RATHER THAN AN IMPORT. Upstream's module
(vllm/model_executor/models/gpt2.py @ the parity pin 555967922) cannot be
executed standalone: `GPT2Attention` builds a paged `Attention` layer that wants
a KV cache and a device, and every projection is a {QKV,Column,Row}ParallelLinear
that wants an initialized distributed environment. At tensor_parallel_size=1 each
of those degenerates to a plain nn.Linear, so the restatement below is
line-for-line faithful and every class cites the upstream lines it mirrors. This
is the same technique `gen-minimax-h3-goldens.py` uses for the H3 DiT, and for
the same reason.

Upstream anchors (vllm/model_executor/models/gpt2.py @ 555967922):
  GPT2Attention.__init__/forward  -> gpt2.py:61-110   (fused c_attn, scale, c_proj)
  GPT2MLP.__init__/forward        -> gpt2.py:113-143  (c_fc -> act -> c_proj)
  GPT2Block.forward               -> gpt2.py:165-180  (pre-LN, two residuals)
  GPT2Model.forward               -> gpt2.py:217-240  (wte + wpe, blocks, ln_f)
  GPT2LMHeadModel                 -> gpt2.py:250-...  (lm_head, tied embeddings)
  _transpose_conv1d               -> gpt2.py:242-254  (HF Conv1D stores W^T)

THE CONV1D TRANSPOSE is the trap this port has to get right and is therefore
gated explicitly: HF's GPT-2 stores c_attn/c_proj/c_fc as Conv1D, whose 2D weight
is [in, out] rather than the [out, in] a Linear expects. Loading it untransposed
produces a model that runs and is wrong. The goldens below are emitted in the
UPSTREAM (Conv1D, [in, out]) orientation, exactly as a checkpoint carries them,
so the C++ loader has to perform the transpose to reproduce the output.

Usage:
    python3 scripts/gen-gpt2-goldens.py --out tests/vllm/models/gpt2_goldens.inc

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
    """`count` values uniform in [-1, 1), reproducible from `name` alone.

    The C++ side rebuilds this byte-for-byte; both use the same 53-bit mantissa
    construction so the doubles are bit-identical.
    """
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


# ── the model, restated at TP=1 ──────────────────────────────────────────────


class Config:
    """The subset of GPT2Config the backbone reads."""

    def __init__(self) -> None:
        self.vocab_size = 33
        self.n_positions = 16
        self.hidden_size = 8
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.n_inner = 16
        self.layer_norm_epsilon = 1e-5


class Weights:
    """Checkpoint tensors in UPSTREAM orientation (Conv1D weights are [in, out])."""

    def __init__(self, cfg: Config) -> None:
        h, inner = cfg.hidden_size, cfg.n_inner
        self.t: dict[str, torch.Tensor] = {
            "wte.weight": param("wte.weight", (cfg.vocab_size, h), 0.5),
            "wpe.weight": param("wpe.weight", (cfg.n_positions, h), 0.25),
            "ln_f.weight": param("ln_f.weight", (h,), 0.1) + 1.0,
            "ln_f.bias": param("ln_f.bias", (h,), 0.1),
        }
        for i in range(cfg.num_hidden_layers):
            p = f"h.{i}."
            self.t.update(
                {
                    p + "ln_1.weight": param(p + "ln_1.weight", (h,), 0.1) + 1.0,
                    p + "ln_1.bias": param(p + "ln_1.bias", (h,), 0.1),
                    p + "ln_2.weight": param(p + "ln_2.weight", (h,), 0.1) + 1.0,
                    p + "ln_2.bias": param(p + "ln_2.bias", (h,), 0.1),
                    # Conv1D: [in, out]. c_attn fuses q|k|v so out = 3*h.
                    p + "attn.c_attn.weight": param(p + "attn.c_attn.weight", (h, 3 * h), 0.3),
                    p + "attn.c_attn.bias": param(p + "attn.c_attn.bias", (3 * h,), 0.2),
                    p + "attn.c_proj.weight": param(p + "attn.c_proj.weight", (h, h), 0.3),
                    p + "attn.c_proj.bias": param(p + "attn.c_proj.bias", (h,), 0.2),
                    p + "mlp.c_fc.weight": param(p + "mlp.c_fc.weight", (h, inner), 0.3),
                    p + "mlp.c_fc.bias": param(p + "mlp.c_fc.bias", (inner,), 0.2),
                    p + "mlp.c_proj.weight": param(p + "mlp.c_proj.weight", (inner, h), 0.3),
                    p + "mlp.c_proj.bias": param(p + "mlp.c_proj.bias", (h,), 0.2),
                }
            )

    def linear(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (W^T, b) — the transpose upstream applies at load time.

        gpt2.py:242-254 `_transpose_conv1d`: HF stores these as Conv1D, so the
        2D weight is transposed relative to what a Linear expects.
        """
        return self.t[name + ".weight"].t(), self.t[name + ".bias"]


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's `activation_function` is gelu_new (the tanh approximation)."""
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


def attention(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """gpt2.py:102-110. Fused c_attn -> chunk(3) -> causal attention -> c_proj."""
    p = f"h.{layer}.attn."
    weight, bias = w.linear(p + "c_attn")
    qkv = F.linear(x, weight, bias)
    q, k, v = qkv.chunk(3, dim=-1)

    heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // heads
    scale = head_dim**-0.5  # gpt2.py:76

    seq = x.shape[0]

    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(seq, heads, head_dim).transpose(0, 1)  # [H, T, D]

    qh, kh, vh = split(q), split(k), split(v)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)  # [H, T, D]
    out = out.transpose(0, 1).reshape(seq, cfg.hidden_size)

    weight, bias = w.linear(p + "c_proj")
    return F.linear(out, weight, bias)


def mlp(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """gpt2.py:139-143. c_fc -> act -> c_proj."""
    p = f"h.{layer}.mlp."
    weight, bias = w.linear(p + "c_fc")
    h = F.linear(x, weight, bias)
    h = gelu_new(h)
    weight, bias = w.linear(p + "c_proj")
    return F.linear(h, weight, bias)


def block(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """gpt2.py:165-180. Pre-LN with two residual adds."""
    p = f"h.{layer}."
    residual = x
    h = layer_norm(x, w.t[p + "ln_1.weight"], w.t[p + "ln_1.bias"], cfg.layer_norm_epsilon)
    h = attention(cfg, w, layer, h)
    x = h + residual

    residual = x
    h = layer_norm(x, w.t[p + "ln_2.weight"], w.t[p + "ln_2.bias"], cfg.layer_norm_epsilon)
    h = mlp(cfg, w, layer, h)
    return residual + h


def forward(cfg: Config, w: Weights, ids: list[int], positions: list[int]) -> torch.Tensor:
    """gpt2.py:217-240. wte + wpe, the block stack, then ln_f."""
    inputs_embeds = w.t["wte.weight"][torch.tensor(ids)]
    position_embeds = w.t["wpe.weight"][torch.tensor(positions)]
    x = inputs_embeds + position_embeds
    for layer in range(cfg.num_hidden_layers):
        x = block(cfg, w, layer, x)
    return layer_norm(x, w.t["ln_f.weight"], w.t["ln_f.bias"], cfg.layer_norm_epsilon)


def logits(cfg: Config, w: Weights, hidden: torch.Tensor) -> torch.Tensor:
    """Tied lm_head: GPT-2 ties lm_head to wte (gpt2.py, tie_word_embeddings)."""
    return F.linear(hidden, w.t["wte.weight"])


# ── emit ─────────────────────────────────────────────────────────────────────


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

    ids = [3, 17, 0, 31, 8, 8, 12]
    positions = list(range(len(ids)))

    hidden = forward(cfg, w, ids, positions)
    lg = logits(cfg, w, hidden)
    argmax = torch.argmax(lg, dim=-1)

    path = Path(args.out)
    with path.open("w", encoding="utf-8") as out:
        out.write("// GENERATED by scripts/gen-gpt2-goldens.py -- do not edit.\n")
        out.write("//\n")
        out.write("// GPT-2 backbone goldens: a TP=1 restatement of vLLM's\n")
        out.write("// vllm/model_executor/models/gpt2.py @ 555967922 (the parity pin),\n")
        out.write("// executed at reduced dimensions on CPU. Weights are rebuilt on both\n")
        out.write("// sides from the same FNV-1a -> splitmix64 stream, so no weight byte is\n")
        out.write("// checked in. Conv1D weights are emitted in UPSTREAM [in, out]\n")
        out.write("// orientation: the loader must transpose them, and a port that does not\n")
        out.write("// still runs while producing the wrong tokens.\n")
        out.write("#pragma once\n\n#include <cstdint>\n\n")
        out.write("namespace gpt2_goldens {\n\n")
        out.write(f"inline constexpr int64_t kVocab = {cfg.vocab_size};\n")
        out.write(f"inline constexpr int64_t kPositions = {cfg.n_positions};\n")
        out.write(f"inline constexpr int64_t kHidden = {cfg.hidden_size};\n")
        out.write(f"inline constexpr int64_t kLayers = {cfg.num_hidden_layers};\n")
        out.write(f"inline constexpr int64_t kHeads = {cfg.num_attention_heads};\n")
        out.write(f"inline constexpr int64_t kInner = {cfg.n_inner};\n")
        out.write(f"inline constexpr double kLayerNormEps = {cfg.layer_norm_epsilon!r};\n\n")
        emit_i64(out, "kInputIds", ids)
        emit_i64(out, "kPositions0", positions)
        emit_f32(out, "kHiddenStates", hidden.detach().numpy())
        emit_f32(out, "kLogits", lg.detach().numpy())
        emit_i64(out, "kArgmax", argmax.detach().numpy())
        out.write("}  // namespace gpt2_goldens\n")

    print(f"wrote {path} (seq={len(ids)}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
