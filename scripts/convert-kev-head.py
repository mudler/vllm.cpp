#!/usr/bin/env python3
"""Convert kev head.pt (torch pickle) to safetensors + JSON for C++ loading.

Usage:
    python3 convert-kev-head.py <head.pt> <output_dir>

Outputs:
    <output_dir>/head.safetensors  -- q.weight, q.bias, k.weight, k.bias (F32)
    <output_dir>/meta.json         -- temperature, head_dim, base, base_revision, lora, etc.

Ported from jaredpalmer/kev kev/checkpoint.py @ 19dcae9b:
    head.pt stores a dict with keys "head" (state_dict of PointerHead),
    "temperature", "head_dim", "base", "base_revision", "lora",
    "option_isolation", "special_embeddings", "weights_dtype".
"""
import json
import os
import sys

import torch
from safetensors.torch import save_file


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <head.pt> <output_dir>", file=sys.stderr)
        sys.exit(1)

    head_pt_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    state = torch.load(head_pt_path, map_location="cpu", weights_only=False)

    head = state["head"]
    tensors = {}
    for name, tensor in head.items():
        tensors[name] = tensor.contiguous().to(torch.float32)

    safetensors_path = os.path.join(output_dir, "head.safetensors")
    save_file(tensors, safetensors_path)
    print(f"Wrote {safetensors_path} ({len(tensors)} tensors)")
    for name, tensor in tensors.items():
        print(f"  {name}: {list(tensor.shape)} {tensor.dtype}")

    meta = {
        "temperature": state["temperature"],
        "head_dim": state["head_dim"],
        "base": state["base"],
        "base_revision": state["base_revision"],
        "lora": state["lora"],
        "option_isolation": state["option_isolation"],
        "special_embeddings": state["special_embeddings"],
        "weights_dtype": state["weights_dtype"],
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
