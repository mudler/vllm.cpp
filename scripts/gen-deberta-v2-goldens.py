#!/usr/bin/env python3
"""Emit tests/vllm/models/deberta_v2_goldens.inc — the DeBERTa v2 encoder oracle.

GLiNER2.5-multi-v1's backbone is a DeBERTa v2 encoder (mdeberta-v3-base, see
.agents/specs/gliner2.5.md, Phase 2). This generator produces the goldens the
C++ port is gated against.

WHY A RESTATEMENT. Upstream vLLM has no DeBERTa support (PRs #42094, #20215
unmerged), so there is no vLLM module to run. The HuggingFace
`DebertaV2Model` (transformers v4.44.2, the secondary oracle) CAN run
standalone, but this project's pattern restates the model in plain PyTorch
from a deterministic PRNG so no weight byte is checked in — the same
technique gen-gpt2-goldens.py uses. Both sides rebuild every weight from the
same FNV-1a -> splitmix64 stream.

Upstream anchors (transformers/models/deberta_v2/modeling_deberta_v2.py
@ v4.44.2):
  make_log_bucket_position          modeling_deberta_v2.py:make_log_bucket_position
  build_relative_position           modeling_deberta_v2.py:build_relative_position
  DebertaV2Embeddings.forward       DebertaV2Embeddings
  DebertaV2Encoder.get_rel_embedding  DebertaV2Encoder
  DisentangledSelfAttention.forward   DisentangledSelfAttention
  DisentangledSelfAttention.disentangled_attention_bias  (same class)
  DebertaV2SelfOutput.forward       DebertaV2SelfOutput
  DebertaV2Intermediate.forward    DebertaV2Intermediate
  DebertaV2Output.forward           DebertaV2Output

THE THING THIS ARCHITECTURE GETS WRONG QUIETLY is dropping the disentangled
attention bias. Without c2p and p2c the model still runs and emits plausible
hidden states, so the bias is gated by a perturbation test in the C++ test
rather than left to be implied by the forward.

Usage:
    python3 scripts/gen-deberta-v2-goldens.py --out tests/vllm/models/deberta_v2_goldens.inc

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


# ── config ───────────────────────────────────────────────────────────────────


class Config:
    """The subset of DebertaV2Config the encoder reads."""

    def __init__(self) -> None:
        self.vocab_size = 33
        self.hidden_size = 12
        self.num_hidden_layers = 2
        self.num_attention_heads = 2
        self.intermediate_size = 24
        self.max_position_embeddings = 16
        self.position_buckets = 8
        self.layer_norm_eps = 1e-7
        self.position_biased_input = False
        self.type_vocab_size = 0
        self.norm_rel_ebd = True          # "layer_norm"
        self.share_att_key = True
        self.use_c2p = True
        self.use_p2c = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def max_rel_pos(self) -> int:
        return self.max_position_embeddings

    @property
    def pos_ebd_size(self) -> int:
        return self.position_buckets * 2

    @property
    def att_span(self) -> int:
        return self.position_buckets

    @property
    def scale_factor(self) -> int:
        sf = 1
        if self.use_c2p:
            sf += 1
        if self.use_p2c:
            sf += 1
        return sf


# ── weights ──────────────────────────────────────────────────────────────────


class Weights:
    """Checkpoint tensors in nn.Linear [out, in] orientation (same as C++)."""

    def __init__(self, cfg: Config) -> None:
        h = cfg.hidden_size
        inter = cfg.intermediate_size
        pe = cfg.pos_ebd_size
        self.t: dict[str, torch.Tensor] = {
            "encoder.embeddings.word_embeddings.weight": param(
                "encoder.embeddings.word_embeddings.weight", (cfg.vocab_size, h), 0.5),
            "encoder.embeddings.LayerNorm.weight": param(
                "encoder.embeddings.LayerNorm.weight", (h,), 0.1) + 1.0,
            "encoder.embeddings.LayerNorm.bias": param(
                "encoder.embeddings.LayerNorm.bias", (h,), 0.1),
            "encoder.encoder.rel_embeddings.weight": param(
                "encoder.encoder.rel_embeddings.weight", (pe, h), 0.3),
            "encoder.encoder.LayerNorm.weight": param(
                "encoder.encoder.LayerNorm.weight", (h,), 0.1) + 1.0,
            "encoder.encoder.LayerNorm.bias": param(
                "encoder.encoder.LayerNorm.bias", (h,), 0.1),
        }
        for i in range(cfg.num_hidden_layers):
            p = f"encoder.encoder.layer.{i}."
            self.t.update({
                p + "attention.self.query_proj.weight": param(p + "attention.self.query_proj.weight", (h, h), 0.3),
                p + "attention.self.query_proj.bias": param(p + "attention.self.query_proj.bias", (h,), 0.2),
                p + "attention.self.key_proj.weight": param(p + "attention.self.key_proj.weight", (h, h), 0.3),
                p + "attention.self.key_proj.bias": param(p + "attention.self.key_proj.bias", (h,), 0.2),
                p + "attention.self.value_proj.weight": param(p + "attention.self.value_proj.weight", (h, h), 0.3),
                p + "attention.self.value_proj.bias": param(p + "attention.self.value_proj.bias", (h,), 0.2),
                p + "attention.output.dense.weight": param(p + "attention.output.dense.weight", (h, h), 0.3),
                p + "attention.output.dense.bias": param(p + "attention.output.dense.bias", (h,), 0.2),
                p + "attention.output.LayerNorm.weight": param(p + "attention.output.LayerNorm.weight", (h,), 0.1) + 1.0,
                p + "attention.output.LayerNorm.bias": param(p + "attention.output.LayerNorm.bias", (h,), 0.1),
                p + "intermediate.dense.weight": param(p + "intermediate.dense.weight", (inter, h), 0.3),
                p + "intermediate.dense.bias": param(p + "intermediate.dense.bias", (inter,), 0.2),
                p + "output.dense.weight": param(p + "output.dense.weight", (h, inter), 0.3),
                p + "output.dense.bias": param(p + "output.dense.bias", (h,), 0.2),
                p + "output.LayerNorm.weight": param(p + "output.LayerNorm.weight", (h,), 0.1) + 1.0,
                p + "output.LayerNorm.bias": param(p + "output.LayerNorm.bias", (h,), 0.1),
            })


# ── the model, restated from HF v4.44.2 ──────────────────────────────────────


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU (erf-based), matching ACT2FN["gelu"] in HF."""
    return F.gelu(x)


def make_log_bucket_position(relative_pos: torch.Tensor, bucket_size: int, max_position: int) -> torch.Tensor:
    """HF modeling_deberta_v2.py:make_log_bucket_position @ v4.44.2."""
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1, dtype=relative_pos.dtype),
        torch.abs(relative_pos),
    )
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.to(log_pos.dtype), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size: int, key_size: int, bucket_size: int, max_position: int) -> torch.Tensor:
    """HF modeling_deberta_v2.py:build_relative_position @ v4.44.2.
    Returns [query_size, key_size] (the squeeze(0) of upstream's [1, q, k])."""
    q_ids = torch.arange(0, query_size)
    k_ids = torch.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    return rel_pos_ids.to(torch.long)


def attention(cfg: Config, w: Weights, layer: int, x: torch.Tensor,
              rel_emb: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
    """DisentangledSelfAttention.forward + DebertaV2SelfOutput @ v4.44.2.

    x:          [seq, hidden]
    rel_emb:    [pos_ebd, hidden]  (already LayerNorm'd)
    rel_pos:    [seq, seq]         (bucketed, NOT shifted)
    Returns:    [seq, hidden]      (attention output after SelfOutput)
    """
    h = cfg.hidden_size
    heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    seq = x.shape[0]
    p = f"encoder.encoder.layer.{layer}.attention.self."

    # Q, K, V projections (nn.Linear: weight [out, in])
    qw = w.t[p + "query_proj.weight"]
    qb = w.t[p + "query_proj.bias"]
    kw = w.t[p + "key_proj.weight"]
    kb = w.t[p + "key_proj.bias"]
    vw = w.t[p + "value_proj.weight"]
    vb = w.t[p + "value_proj.bias"]

    q = F.linear(x, qw, qb)  # [seq, hidden]
    k = F.linear(x, kw, kb)
    v = F.linear(x, vw, vb)

    # Position projections (share_att_key=True: same query_proj/key_proj)
    pos_q = F.linear(rel_emb, qw, qb)  # [pos_ebd, hidden]
    pos_k = F.linear(rel_emb, kw, kb)  # [pos_ebd, hidden]

    # Reshape to [heads, seq, head_dim]
    def split(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(t.shape[0], heads, head_dim).permute(1, 0, 2).contiguous()

    qh = split(q)  # [heads, seq, head_dim]
    kh = split(k)
    vh = split(v)
    pos_qh = split(pos_q)  # [heads, pos_ebd, head_dim]
    pos_kh = split(pos_k)

    sf = cfg.scale_factor
    scale = math.sqrt(head_dim * sf)

    # Content-content scores
    scores = torch.matmul(qh, kh.transpose(-1, -2)) / scale  # [heads, seq, seq]

    # c2p: query @ pos_key^T, gathered by c2p_pos
    att_span = cfg.att_span
    if cfg.use_c2p:
        c2p_att = torch.matmul(qh, pos_kh.transpose(-1, -2))  # [heads, seq, pos_ebd]
        c2p_pos = torch.clamp(rel_pos + att_span, 0, att_span * 2 - 1)  # [seq, seq]
        # gather: c2p_att[h, i, j] = c2p_att_raw[h, i, c2p_pos[i, j]]
        c2p_pos_exp = c2p_pos.unsqueeze(0).expand(heads, seq, seq)
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos_exp)
        scores = scores + c2p_att / scale

    # p2c: key @ pos_query^T, gathered by p2c_pos, then transposed
    if cfg.use_p2c:
        p2c_att = torch.matmul(kh, pos_qh.transpose(-1, -2))  # [heads, seq, pos_ebd]
        p2c_pos = torch.clamp(-rel_pos + att_span, 0, att_span * 2 - 1)  # [seq, seq]
        # gather: p2c_att_gathered[h, j, i] = p2c_att_raw[h, j, p2c_pos[j, i]]
        # (index is p2c_pos, shape [seq, seq], expanded to [heads, seq, seq])
        p2c_pos_exp = p2c_pos.unsqueeze(0).expand(heads, seq, seq)
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos_exp)
        # transpose: result[h, i, j] = p2c_att_gathered[h, j, i]
        p2c_att = p2c_att.transpose(-1, -2)
        scores = scores + p2c_att / scale

    # Softmax (bidirectional, no mask)
    probs = F.softmax(scores, dim=-1)

    # Weighted sum of V
    attn_out = torch.matmul(probs, vh)  # [heads, seq, head_dim]
    attn_out = attn_out.permute(1, 0, 2).contiguous().reshape(seq, h)  # [seq, hidden]

    # DebertaV2SelfOutput: dense -> LayerNorm(dense(attn) + x)
    op = f"encoder.encoder.layer.{layer}.attention.output."
    dw = w.t[op + "dense.weight"]
    db = w.t[op + "dense.bias"]
    lw = w.t[op + "LayerNorm.weight"]
    lb = w.t[op + "LayerNorm.bias"]
    out = F.linear(attn_out, dw, db)
    out = layer_norm(out + x, lw, lb, cfg.layer_norm_eps)
    return out


def mlp(cfg: Config, w: Weights, layer: int, x: torch.Tensor) -> torch.Tensor:
    """DebertaV2Intermediate + DebertaV2Output @ v4.44.2."""
    ip = f"encoder.encoder.layer.{layer}.intermediate."
    op = f"encoder.encoder.layer.{layer}.output."
    h = cfg.hidden_size
    inter = cfg.intermediate_size

    # Intermediate: dense -> gelu
    inter_out = F.linear(x, w.t[ip + "dense.weight"], w.t[ip + "dense.bias"])
    inter_out = gelu(inter_out)

    # Output: dense -> LayerNorm(dense(inter) + x)
    out = F.linear(inter_out, w.t[op + "dense.weight"], w.t[op + "dense.bias"])
    out = layer_norm(out + x, w.t[op + "LayerNorm.weight"], w.t[op + "LayerNorm.bias"],
                     cfg.layer_norm_eps)
    return out


def forward(cfg: Config, w: Weights, ids: list[int]) -> torch.Tensor:
    """Full encoder forward: embeddings -> N layers -> hidden states [seq, hidden]."""
    h = cfg.hidden_size
    seq = len(ids)

    # Embeddings: word_embeddings -> LayerNorm (no position, no token type)
    x = w.t["encoder.embeddings.word_embeddings.weight"][torch.tensor(ids)]  # [seq, hidden]
    x = layer_norm(x, w.t["encoder.embeddings.LayerNorm.weight"],
                   w.t["encoder.embeddings.LayerNorm.bias"], cfg.layer_norm_eps)

    # Rel embeddings: rel_embeddings -> LayerNorm (if norm_rel_ebd)
    rel_emb = w.t["encoder.encoder.rel_embeddings.weight"]  # [pos_ebd, hidden]
    if cfg.norm_rel_ebd:
        rel_emb = layer_norm(rel_emb, w.t["encoder.encoder.LayerNorm.weight"],
                             w.t["encoder.encoder.LayerNorm.bias"], cfg.layer_norm_eps)

    # Relative position buckets
    rel_pos = build_relative_position(seq, seq, cfg.position_buckets, cfg.max_rel_pos)

    for layer in range(cfg.num_hidden_layers):
        attn_out = attention(cfg, w, layer, x, rel_emb, rel_pos)
        x = mlp(cfg, w, layer, attn_out)

    return x


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

    hidden = forward(cfg, w, ids)

    path = Path(args.out)
    with path.open("w", encoding="utf-8") as out:
        out.write("// GENERATED by scripts/gen-deberta-v2-goldens.py -- do not edit.\n")
        out.write("//\n")
        out.write("// DeBERTa v2 encoder goldens: a restatement of HuggingFace\n")
        out.write("// transformers/models/deberta_v2/modeling_deberta_v2.py @ v4.44.2,\n")
        out.write("// executed at reduced dimensions on CPU. Weights are rebuilt on both\n")
        out.write("// sides from the same FNV-1a -> splitmix64 stream, so no weight byte\n")
        out.write("// is checked in. The disentangled attention (c2p + p2c biases) is the\n")
        out.write("// feature this port has to get right; it is gated by a perturbation test\n")
        out.write("// rather than left to be implied by the forward.\n")
        out.write("#pragma once\n\n#include <cstdint>\n\n")
        out.write("namespace deberta_v2_goldens {\n\n")
        out.write(f"inline constexpr int64_t kVocab = {cfg.vocab_size};\n")
        out.write(f"inline constexpr int64_t kHidden = {cfg.hidden_size};\n")
        out.write(f"inline constexpr int64_t kLayers = {cfg.num_hidden_layers};\n")
        out.write(f"inline constexpr int64_t kHeads = {cfg.num_attention_heads};\n")
        out.write(f"inline constexpr int64_t kInter = {cfg.intermediate_size};\n")
        out.write(f"inline constexpr int64_t kMaxPos = {cfg.max_position_embeddings};\n")
        out.write(f"inline constexpr int64_t kPosBuckets = {cfg.position_buckets};\n")
        out.write(f"inline constexpr double kLayerNormEps = {cfg.layer_norm_eps!r};\n")
        out.write(f"inline constexpr int64_t kPosEbdSize = {cfg.pos_ebd_size};\n")
        out.write(f"inline constexpr int64_t kAttSpan = {cfg.att_span};\n")
        out.write(f"inline constexpr bool kShareAttKey = {'true' if cfg.share_att_key else 'false'};\n")
        out.write(f"inline constexpr bool kUseC2p = {'true' if cfg.use_c2p else 'false'};\n")
        out.write(f"inline constexpr bool kUseP2c = {'true' if cfg.use_p2c else 'false'};\n")
        out.write(f"inline constexpr bool kNormRelEbd = {'true' if cfg.norm_rel_ebd else 'false'};\n\n")
        emit_i64(out, "kInputIds", ids)
        emit_f32(out, "kHiddenStates", hidden.detach().numpy())
        out.write("}  // namespace deberta_v2_goldens\n")

    print(f"wrote {path} (seq={len(ids)}, hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
