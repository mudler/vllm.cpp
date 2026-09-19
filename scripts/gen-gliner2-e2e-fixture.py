#!/usr/bin/env python3
"""Generate the committed tiny BoundaryExtractor e2e fixture (MODEL-GLINER25).

Writes tests/vllm/models/fixtures/gliner2_e2e/:
  config.json        — architectures ["BoundaryExtractor"], with hidden_size /
                       num_hidden_layers (LoadHfConfig requires them) and the
                       boundary_head sub-object the weight loader reads.
  model.safetensors  — deterministic tiny F32 checkpoint with every tensor name
                       deberta_v2::Load and gliner2::LoadBoundaryHead expect,
                       at reduced dimensions so the committed fixture is tiny.

The weight loader (gliner2_weights.cpp) infers the DeBERTa encoder config from
weight shapes, so the config's hidden_size / num_hidden_layers are only there
to satisfy LoadHfConfig's RequireKey — the loader does not read them.

Deterministic: fixed seed, no torch. Re-running reproduces byte-identical files.

Usage: python3 scripts/gen-gliner2-e2e-fixture.py
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests/vllm/models/fixtures/gliner2_e2e"

# Fixture dimensions — small enough to keep the committed fixture tiny, large
# enough to exercise every code path (multi-layer encoder, attention blocks,
# refinement blocks, query head projections).
H = 24           # hidden_size
VOCAB = 32
LAYERS = 2
HEADS = 4        # head_dim = 6
INTERMEDIATE = 48
POS_BUCKETS = 8  # rel_embeddings = [16, H]

D = 16           # boundary_dim
BD_HEADS = 2     # boundary head_dim = 8
BD_ATTN_LAYERS = 1
BD_ATTN_WINDOW = 8
BD_REFINE_LAYERS = 1
FFN_MULT = 2.0   # ffn_dim = 32, half_ffn = 16


class Rng:
    """Deterministic xorshift32 in [-scale, scale) — no numpy dependency."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF or 1

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x
        return x

    def uniform(self, scale: float = 0.08) -> float:
        return (self.next_u32() / 2**32 * 2.0 - 1.0) * scale


def f32_tensor(shape: list[int], seed: int, scale: float = 0.08) -> bytes:
    rng = Rng(seed)
    numel = 1
    for s in shape:
        numel *= s
    return b"".join(struct.pack("<f", rng.uniform(scale)) for _ in range(numel))


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]]) -> None:
    header: dict[str, dict] = {}
    offset = 0
    for name, (shape, data) in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    hdr = json.dumps(header, sort_keys=True).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for _, (_, data) in tensors.items():
            f.write(data)


def encoder_tensors() -> dict[str, tuple[list[int], bytes]]:
    t: dict[str, tuple[list[int], bytes]] = {}
    s = 1

    # Embeddings
    t["encoder.embeddings.word_embeddings.weight"] = ([VOCAB, H], f32_tensor([VOCAB, H], s))
    s += 1
    t["encoder.embeddings.LayerNorm.weight"] = ([H], f32_tensor([H], s))
    s += 1
    t["encoder.embeddings.LayerNorm.bias"] = ([H], f32_tensor([H], s, 0.02))
    s += 1

    # Relative embeddings [2*pos_buckets, H]
    rel_sz = 2 * POS_BUCKETS
    t["encoder.encoder.rel_embeddings.weight"] = ([rel_sz, H], f32_tensor([rel_sz, H], s))
    s += 1
    t["encoder.encoder.LayerNorm.weight"] = ([H], f32_tensor([H], s))
    s += 1
    t["encoder.encoder.LayerNorm.bias"] = ([H], f32_tensor([H], s, 0.02))
    s += 1

    # Per-layer weights
    for i in range(LAYERS):
        p = f"encoder.encoder.layer.{i}."
        t[p + "attention.self.query_proj.weight"] = ([H, H], f32_tensor([H, H], s))
        s += 1
        t[p + "attention.self.query_proj.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "attention.self.key_proj.weight"] = ([H, H], f32_tensor([H, H], s))
        s += 1
        t[p + "attention.self.key_proj.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "attention.self.value_proj.weight"] = ([H, H], f32_tensor([H, H], s))
        s += 1
        t[p + "attention.self.value_proj.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "attention.output.dense.weight"] = ([H, H], f32_tensor([H, H], s))
        s += 1
        t[p + "attention.output.dense.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "attention.output.LayerNorm.weight"] = ([H], f32_tensor([H], s))
        s += 1
        t[p + "attention.output.LayerNorm.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "intermediate.dense.weight"] = ([INTERMEDIATE, H], f32_tensor([INTERMEDIATE, H], s))
        s += 1
        t[p + "intermediate.dense.bias"] = ([INTERMEDIATE], f32_tensor([INTERMEDIATE], s, 0.02))
        s += 1
        t[p + "output.dense.weight"] = ([H, INTERMEDIATE], f32_tensor([H, INTERMEDIATE], s))
        s += 1
        t[p + "output.dense.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1
        t[p + "output.LayerNorm.weight"] = ([H], f32_tensor([H], s))
        s += 1
        t[p + "output.LayerNorm.bias"] = ([H], f32_tensor([H], s, 0.02))
        s += 1

    return t


def boundary_head_tensors() -> dict[str, tuple[list[int], bytes]]:
    t: dict[str, tuple[list[int], bytes]] = {}
    s = 1000
    ffn_dim = int(D * FFN_MULT)
    half_ffn = ffn_dim // 2
    e = "boundary_head.boundary_encoder."

    t[e + "left_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[e + "left_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[e + "right_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[e + "right_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[e + "output_projection.weight"] = ([D, 2 * D], f32_tensor([D, 2 * D], s))
    s += 1
    t[e + "output_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[e + "layer_norm.weight"] = ([D], f32_tensor([D], s))
    s += 1
    t[e + "layer_norm.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[e + "bos_state"] = ([H], f32_tensor([H], s))
    s += 1
    t[e + "eos_state"] = ([H], f32_tensor([H], s))
    s += 1

    # Attention blocks
    for i in range(BD_ATTN_LAYERS):
        p = f"{e}attention_blocks.{i}."
        t[p + "norm.weight"] = ([D], f32_tensor([D], s))
        s += 1
        t[p + "norm.bias"] = ([D], f32_tensor([D], s, 0.02))
        s += 1
        t[p + "qkv_projection.weight"] = ([3 * D, D], f32_tensor([3 * D, D], s))
        s += 1
        t[p + "qkv_projection.bias"] = ([3 * D], f32_tensor([3 * D], s, 0.02))
        s += 1
        t[p + "output_projection.weight"] = ([D, D], f32_tensor([D, D], s))
        s += 1
        t[p + "output_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
        s += 1

    # Refinement blocks
    for i in range(BD_REFINE_LAYERS):
        p = f"{e}refinement_blocks.{i}."
        t[p + "norm.weight"] = ([D], f32_tensor([D], s))
        s += 1
        t[p + "norm.bias"] = ([D], f32_tensor([D], s, 0.02))
        s += 1
        t[p + "input_projection.weight"] = ([ffn_dim, D], f32_tensor([ffn_dim, D], s))
        s += 1
        t[p + "input_projection.bias"] = ([ffn_dim], f32_tensor([ffn_dim], s, 0.02))
        s += 1
        t[p + "output_projection.weight"] = ([D, half_ffn], f32_tensor([D, half_ffn], s))
        s += 1
        t[p + "output_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
        s += 1

    # BoundaryQueryHead
    q = "boundary_head.boundary_query_head."
    t[q + "start_boundary_projection.weight"] = ([D, D], f32_tensor([D, D], s))
    s += 1
    t[q + "start_boundary_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[q + "end_boundary_projection.weight"] = ([D, D], f32_tensor([D, D], s))
    s += 1
    t[q + "end_boundary_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[q + "start_query_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[q + "start_query_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[q + "end_query_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[q + "end_query_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[q + "inside_text_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[q + "inside_text_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1
    t[q + "inside_query_projection.weight"] = ([D, H], f32_tensor([D, H], s))
    s += 1
    t[q + "inside_query_projection.bias"] = ([D], f32_tensor([D], s, 0.02))
    s += 1

    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    config = {
        "architectures": ["BoundaryExtractor"],
        "model_type": "extractor",
        # LoadHfConfig requires these fields; the weight loader infers the
        # real values from weight shapes and ignores the config values.
        "hidden_size": H,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "token_pooling": "first",
        "config_version": 3,
        "boundary_head": {
            "boundary_dim": D,
            "boundary_attention_heads": BD_HEADS,
            "boundary_attention_layers": BD_ATTN_LAYERS,
            "boundary_attention_window": BD_ATTN_WINDOW,
            "boundary_refinement_layers": BD_REFINE_LAYERS,
            "boundary_ffn_multiplier": FFN_MULT,
        },
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    # Minimal BPE tokenizer matching vocab_size: ▁ + a-z + 0-4 = 32 entries.
    # The NER pipeline (Gliner2NerInference) tokenizes text and labels through
    # this; Metaspace pre-tokenizer, no merges (each character is one token).
    vocab = {"\u2581": 0}
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[ch] = i + 1
    for i, ch in enumerate("01234"):
        vocab[ch] = 27 + i
    tokenizer = {
        "version": "1.0",
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always",
            "split": True,
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "\u2581",
            "prepend_scheme": "always",
            "split": True,
        },
        "model": {
            "type": "BPE",
            "unk_token": None,
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": [],
    }
    with open(OUT / "tokenizer.json", "w") as f:
        json.dump(tokenizer, f, indent=2)
        f.write("\n")

    tensors = {**encoder_tensors(), **boundary_head_tensors()}
    write_safetensors(OUT / "model.safetensors", tensors)

    # Report tensor count for verification
    print(f"Wrote {len(tensors)} tensors to {OUT / 'model.safetensors'}")
    total_bytes = sum(len(data) for _, (_, data) in tensors.items())
    print(f"Total payload: {total_bytes} bytes ({total_bytes / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
