#!/usr/bin/env python3
"""Generate golden logits for the cua-s1 TinyTransformerScorer test.

The generator and the C++ test rebuild the same weights from the same
FNV-1a -> splitmix64 stream, so no weight byte is checked in.  The golden
file contains only config constants, the input tensors, and the expected
logits.  The packed-QKV split order, the safe-mask force at position 0,
and the finfo.min fill are features this port has to get right; each is
gated by a perturbation test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
    for s in shape:
        count *= s
    return torch.tensor(rand(name, count).reshape(shape) * scale, dtype=torch.float32)


def param_plus_one(name: str, shape, scale: float) -> torch.Tensor:
    return param(name, shape, scale) + 1.0


# ── Config (reduced dims for fast test) ────────────────────────────────
WIDTH = 16
RANK = 16
CONTEXT_TOKENS = 8
OPTION_TOKENS = 4
LAYERS = 2
HEADS = 4
DIM_FF = WIDTH * 4
VOCAB = 257
MAX_POS = max(CONTEXT_TOKENS, OPTION_TOKENS)


def build_model() -> nn.Module:
    model = TinyTransformerScorer(
        WIDTH, RANK, CONTEXT_TOKENS, OPTION_TOKENS, LAYERS, HEADS, dropout=0.0
    )
    model.eval()
    return model


class AttentionHead(nn.Module):
    def __init__(self, input_width: int, rank: int) -> None:
        super().__init__()
        self.context_norm = nn.LayerNorm(input_width)
        self.option_norm = nn.LayerNorm(input_width)
        self.query = nn.Linear(input_width, rank, bias=False)
        self.key = nn.Linear(input_width, rank, bias=False)
        self.value = nn.Linear(input_width, rank, bias=False)
        self.rank = rank

    def forward(self, context, context_mask, options, option_mask):
        context = self.context_norm(context.float())
        options = self.option_norm(options.float())
        query = self.query(options)
        key = self.key(context)
        value = self.value(context)
        scores = torch.einsum("bnr,blr->bnl", query, key) / math.sqrt(self.rank)
        scores = scores.masked_fill(~context_mask[:, None, :], torch.finfo(scores.dtype).min)
        attended = torch.einsum("bnl,blr->bnr", scores.softmax(-1), value)
        logits = (query * attended).sum(-1) / math.sqrt(self.rank)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)


class TinyTransformerScorer(nn.Module):
    def __init__(self, width, rank, context_tokens, option_tokens, layers=2, heads=4, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.position = nn.Embedding(max(context_tokens, option_tokens), width)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        option_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout,
            batch_first=True, norm_first=True,
        )
        self.option_encoder = nn.TransformerEncoder(option_layer, 1)
        self.head = AttentionHead(width, rank)

    def _embed(self, ids):
        positions = torch.arange(ids.shape[-1], device=ids.device)
        return self.embedding(ids) + self.position(positions)

    def forward(self, batch):
        context_mask = batch["context_mask"]
        safe_context_mask = context_mask.clone()
        safe_context_mask[:, 0] = True
        context = self.encoder(
            self._embed(batch["context_ids"]),
            src_key_padding_mask=~safe_context_mask,
        )
        option_ids = batch["option_ids"]
        batch_size, option_count, token_count = option_ids.shape
        flat_ids = option_ids.reshape(batch_size * option_count, token_count)
        flat_mask = batch["option_token_mask"].reshape(batch_size * option_count, token_count)
        safe_mask = flat_mask.clone()
        safe_mask[:, 0] = True
        hidden = self.option_encoder(self._embed(flat_ids), src_key_padding_mask=~safe_mask)
        weights = flat_mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        options = pooled.reshape(batch_size, option_count, -1)
        return self.head(
            context, context_mask, options, batch["option_mask"],
        )


def load_state_dict(model: nn.Module) -> None:
    """Fill model parameters from the PRNG, matching the C++ test's Rand()."""
    sd = {}
    sd["embedding.weight"] = param("embedding.weight", (VOCAB, WIDTH), 0.5)
    sd["position.weight"] = param("position.weight", (MAX_POS, WIDTH), 0.5)

    for i in range(LAYERS):
        p = f"encoder.layers.{i}."
        sd[p + "self_attn.in_proj_weight"] = param(p + "self_attn.in_proj_weight", (3 * WIDTH, WIDTH), 0.3)
        sd[p + "self_attn.in_proj_bias"] = param(p + "self_attn.in_proj_bias", (3 * WIDTH,), 0.3)
        sd[p + "self_attn.out_proj.weight"] = param(p + "self_attn.out_proj.weight", (WIDTH, WIDTH), 0.3)
        sd[p + "self_attn.out_proj.bias"] = param(p + "self_attn.out_proj.bias", (WIDTH,), 0.3)
        sd[p + "linear1.weight"] = param(p + "linear1.weight", (DIM_FF, WIDTH), 0.3)
        sd[p + "linear1.bias"] = param(p + "linear1.bias", (DIM_FF,), 0.3)
        sd[p + "linear2.weight"] = param(p + "linear2.weight", (WIDTH, DIM_FF), 0.3)
        sd[p + "linear2.bias"] = param(p + "linear2.bias", (WIDTH,), 0.3)
        sd[p + "norm1.weight"] = param_plus_one(p + "norm1.weight", (WIDTH,), 0.1)
        sd[p + "norm1.bias"] = param(p + "norm1.bias", (WIDTH,), 0.1)
        sd[p + "norm2.weight"] = param_plus_one(p + "norm2.weight", (WIDTH,), 0.1)
        sd[p + "norm2.bias"] = param(p + "norm2.bias", (WIDTH,), 0.1)

    p = "option_encoder.layers.0."
    sd[p + "self_attn.in_proj_weight"] = param(p + "self_attn.in_proj_weight", (3 * WIDTH, WIDTH), 0.3)
    sd[p + "self_attn.in_proj_bias"] = param(p + "self_attn.in_proj_bias", (3 * WIDTH,), 0.3)
    sd[p + "self_attn.out_proj.weight"] = param(p + "self_attn.out_proj.weight", (WIDTH, WIDTH), 0.3)
    sd[p + "self_attn.out_proj.bias"] = param(p + "self_attn.out_proj.bias", (WIDTH,), 0.3)
    sd[p + "linear1.weight"] = param(p + "linear1.weight", (DIM_FF, WIDTH), 0.3)
    sd[p + "linear1.bias"] = param(p + "linear1.bias", (DIM_FF,), 0.3)
    sd[p + "linear2.weight"] = param(p + "linear2.weight", (WIDTH, DIM_FF), 0.3)
    sd[p + "linear2.bias"] = param(p + "linear2.bias", (WIDTH,), 0.3)
    sd[p + "norm1.weight"] = param_plus_one(p + "norm1.weight", (WIDTH,), 0.1)
    sd[p + "norm1.bias"] = param(p + "norm1.bias", (WIDTH,), 0.1)
    sd[p + "norm2.weight"] = param_plus_one(p + "norm2.weight", (WIDTH,), 0.1)
    sd[p + "norm2.bias"] = param(p + "norm2.bias", (WIDTH,), 0.1)

    sd["head.context_norm.weight"] = param_plus_one("head.context_norm.weight", (WIDTH,), 0.1)
    sd["head.context_norm.bias"] = param("head.context_norm.bias", (WIDTH,), 0.1)
    sd["head.option_norm.weight"] = param_plus_one("head.option_norm.weight", (WIDTH,), 0.1)
    sd["head.option_norm.bias"] = param("head.option_norm.bias", (WIDTH,), 0.1)
    sd["head.query.weight"] = param("head.query.weight", (RANK, WIDTH), 0.3)
    sd["head.key.weight"] = param("head.key.weight", (RANK, WIDTH), 0.3)
    sd["head.value.weight"] = param("head.value.weight", (RANK, WIDTH), 0.3)

    model.load_state_dict(sd)


# ── ByteCollator (ported from model.py:48-99) ────────────────────────


def _byte_ids(text: str, length: int) -> list[int]:
    return [byte + 1 for byte in text.encode("utf-8", errors="replace")[:length]]


from dataclasses import dataclass


@dataclass(frozen=True)
class ChoiceExample:
    context: str
    options: tuple[str, ...]
    label: int


class ByteCollator:
    def __init__(self, context_tokens: int, option_tokens: int) -> None:
        self.context_tokens = context_tokens
        self.option_tokens = option_tokens

    def __call__(self, examples):
        contexts = [_byte_ids(item.context, self.context_tokens) for item in examples]
        option_rows = [
            [_byte_ids(opt, self.option_tokens) for opt in item.options] for item in examples
        ]
        return _tensor_batch(examples, contexts, option_rows, pad_id=0)


def _tensor_batch(examples, contexts, option_rows, pad_id):
    batch = len(examples)
    max_context = max(1, max(map(len, contexts)))
    max_options = max(len(row) for row in option_rows)
    max_option_tokens = max(1, max(len(tokens) for row in option_rows for tokens in row))
    context_ids = torch.full((batch, max_context), pad_id, dtype=torch.long)
    option_ids = torch.full((batch, max_options, max_option_tokens), pad_id, dtype=torch.long)
    option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
    for row, tokens in enumerate(contexts):
        if tokens:
            context_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    for row, options in enumerate(option_rows):
        option_mask[row, : len(options)] = True
        for column, tokens in enumerate(options):
            if tokens:
                option_ids[row, column, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return {
        "context_ids": context_ids,
        "context_mask": context_ids.ne(pad_id),
        "option_ids": option_ids,
        "option_token_mask": option_ids.ne(pad_id),
        "option_mask": option_mask,
    }


def emit_f32(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.float32).ravel()
    out.append(f"inline constexpr float {name}[] = {{")
    for i, v in enumerate(flat):
        if i % 6 == 0:
            out.append("\n    ")
        out.append(f"{v:.9e}F, ")
    out.append("\n};\n\n")


def emit_i64(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.int64).ravel()
    out.append(f"inline constexpr int64_t {name}[] = {{")
    for i, v in enumerate(flat):
        if i % 10 == 0:
            out.append("\n    ")
        out.append(f"{v}, ")
    out.append("\n};\n\n")


def emit_u8(out, name: str, values) -> None:
    flat = np.asarray(values, dtype=np.uint8).ravel()
    out.append(f"inline constexpr uint8_t {name}[] = {{")
    for i, v in enumerate(flat):
        if i % 12 == 0:
            out.append("\n    ")
        out.append(f"{int(v)}, ")
    out.append("\n};\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/vllm/models/cua_s1_goldens.inc"))
    args = parser.parse_args()

    model = build_model()
    load_state_dict(model)

    # Test case 1: 3 options, context with some padding.
    context_ids_1 = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]])  # 4 valid, 4 pad
    context_mask_1 = torch.tensor([[True, True, True, True, False, False, False, False]])
    option_ids_1 = torch.tensor([[[5, 6, 0, 0], [7, 8, 9, 0], [10, 0, 0, 0]]])
    option_tok_mask_1 = torch.tensor([[[True, True, False, False], [True, True, True, False], [True, False, False, False]]])
    option_mask_1 = torch.tensor([[True, True, True]])

    batch_1 = {
        "context_ids": context_ids_1,
        "context_mask": context_mask_1,
        "option_ids": option_ids_1,
        "option_token_mask": option_tok_mask_1,
        "option_mask": option_mask_1,
    }
    with torch.no_grad():
        logits_1 = model(batch_1).squeeze(0)

    # Test case 2: 2 options, full context (no padding).
    context_ids_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    context_mask_2 = torch.tensor([[True, True, True, True, True, True, True, True]])
    option_ids_2 = torch.tensor([[[9, 10, 11, 12], [13, 14, 15, 0]]])
    option_tok_mask_2 = torch.tensor([[[True, True, True, True], [True, True, True, False]]])
    option_mask_2 = torch.tensor([[True, True]])

    batch_2 = {
        "context_ids": context_ids_2,
        "context_mask": context_mask_2,
        "option_ids": option_ids_2,
        "option_token_mask": option_tok_mask_2,
        "option_mask": option_mask_2,
    }
    with torch.no_grad():
        logits_2 = model(batch_2).squeeze(0)

    # Emit goldens.
    out = []
    out.append("// GENERATED by scripts/gen-cua-s1-goldens.py -- do not edit.\n")
    out.append("//\n")
    out.append("// cua-s1 TinyTransformerScorer goldens: a restatement of\n")
    out.append("// trycua/cua libs/cua-s1/python/src/cua_s1/model.py @ 9bbfa7dd,\n")
    out.append("// executed at reduced dimensions on CPU. Weights are rebuilt on\n")
    out.append("// both sides from the same FNV-1a -> splitmix64 stream, so no\n")
    out.append("// weight byte is checked in.\n")
    out.append("#pragma once\n\n")
    out.append("#include <cstdint>\n\n")
    out.append("namespace cua_s1_goldens {\n\n")

    out.append(f"inline constexpr int64_t kWidth = {WIDTH};\n")
    out.append(f"inline constexpr int64_t kRank = {RANK};\n")
    out.append(f"inline constexpr int64_t kContextTokens = {CONTEXT_TOKENS};\n")
    out.append(f"inline constexpr int64_t kOptionTokens = {OPTION_TOKENS};\n")
    out.append(f"inline constexpr int64_t kLayers = {LAYERS};\n")
    out.append(f"inline constexpr int64_t kHeads = {HEADS};\n\n")

    # Test case 1.
    out.append("// Test case 1: 3 options, partial context padding.\n")
    emit_i64(out, "kContextIds1", context_ids_1.numpy())
    emit_u8(out, "kContextMask1", context_mask_1.numpy())
    emit_i64(out, "kOptionIds1", option_ids_1.numpy())
    emit_u8(out, "kOptionTokMask1", option_tok_mask_1.numpy())
    emit_u8(out, "kOptionMask1", option_mask_1.numpy())
    emit_f32(out, "kLogits1", logits_1.numpy())
    out.append(f"inline constexpr int64_t kNumOptions1 = {logits_1.shape[0]};\n\n")

    # Test case 2.
    out.append("// Test case 2: 2 options, full context.\n")
    emit_i64(out, "kContextIds2", context_ids_2.numpy())
    emit_u8(out, "kContextMask2", context_mask_2.numpy())
    emit_i64(out, "kOptionIds2", option_ids_2.numpy())
    emit_u8(out, "kOptionTokMask2", option_tok_mask_2.numpy())
    emit_u8(out, "kOptionMask2", option_mask_2.numpy())
    emit_f32(out, "kLogits2", logits_2.numpy())
    out.append(f"inline constexpr int64_t kNumOptions2 = {logits_2.shape[0]};\n\n")

    # ── Phase 2: ByteCollator + full pipeline ───────────────────────────
    #
    # Test case 3: string inputs through ByteCollator → model → softmax.
    # Verifies the full inference path: text → byte ids → tensors → logits → probs.
    context_str_3 = "hi"
    options_str_3 = ["a", "bc"]

    collator = ByteCollator(CONTEXT_TOKENS, OPTION_TOKENS)
    example_3 = ChoiceExample(context=context_str_3, options=tuple(options_str_3), label=0)
    batch_3 = collator([example_3])
    with torch.no_grad():
        logits_3 = model(batch_3).squeeze(0)
        probs_3 = torch.softmax(logits_3, dim=-1)

    # Emit the collated tensors so the C++ ByteCollator test can compare.
    out.append("// Test case 3: ByteCollator + full pipeline (string inputs).\n")
    emit_i64(out, "kContextIds3", batch_3["context_ids"].numpy())
    emit_u8(out, "kContextMask3", batch_3["context_mask"].numpy())
    emit_i64(out, "kOptionIds3", batch_3["option_ids"].numpy())
    emit_u8(out, "kOptionTokMask3", batch_3["option_token_mask"].numpy())
    emit_u8(out, "kOptionMask3", batch_3["option_mask"].numpy())
    emit_f32(out, "kProbs3", probs_3.numpy())
    out.append(f"inline constexpr int64_t kNumOptions3 = {probs_3.shape[0]};\n")
    out.append(f"inline constexpr int64_t kCtxLen3 = {batch_3['context_ids'].shape[1]};\n")
    out.append(f"inline constexpr int64_t kOptLen3 = {batch_3['option_ids'].shape[2]};\n\n")

    out.append("}  // namespace cua_s1_goldens\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
