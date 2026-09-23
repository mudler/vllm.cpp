#!/usr/bin/env python3
"""Convert a kev checkpoint to a self-contained C++-loadable model directory.

Usage:
    python3 convert-kev.py <checkpoint_dir> [--output-dir <dir>] [--base-model-dir <dir>]

Inputs (in checkpoint_dir):
    head.pt                   -- torch pickle: PointerHead weights + metadata
    adapter_config.json       -- LoRA config (r, lora_alpha, base_model_name_or_path)
    adapter_model.safetensors -- LoRA adapter weights (F32)

Outputs (in output_dir, default: checkpoint_dir):
    model.safetensors         -- merged Qwen3.5 weights (BF16, LoRA applied at convert time)
    head.safetensors          -- PointerHead weights (F32: q.weight, q.bias, k.weight, k.bias)
    config.json               -- model_type=qwen3_5, architectures=["KevModel"], kev_head_dim, kev_temperature
    meta.json                 -- temperature, head_dim, base, lora, etc.
    tokenizer.json            -- copied from base model
    tokenizer_config.json     -- copied from base model
    merges.txt                -- copied from base model
    vocab.json                -- copied from base model

The LoRA merge (W' = W + scaling * (lora_B @ lora_A)) is applied HERE, at
convert time, not at C++ load time.  This keeps the C++ registry simple: it
loads a single set of safetensors shards (model + head) with no runtime LoRA
math, no BF16<->F32 conversion, and no two-directory loading.

Ported from jaredpalmer/kev kev/checkpoint.py @ 19dcae9b6e3e1a48200c5825aad9fc200d31e20a
and the PEFT merge_and_unload formula.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def resolve_base_model_dir(base_model_name_or_path, explicit_dir=None):
    """Resolve the base model directory from the HF cache or an explicit path."""
    if explicit_dir:
        return explicit_dir
    if os.path.isdir(base_model_name_or_path):
        return base_model_name_or_path
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    org, name = base_model_name_or_path.split("/", 1)
    repo_dir = f"models--{org}--{name}"
    # Flat layout (config.json at root) or snapshots/<rev>/ layout
    flat = os.path.join(cache, repo_dir)
    if os.path.isfile(os.path.join(flat, "config.json")):
        return flat
    snapshots = os.path.join(flat, "snapshots")
    if os.path.isdir(snapshots):
        revs = os.listdir(snapshots)
        if revs:
            return os.path.join(snapshots, revs[0])
    raise FileNotFoundError(
        f"Base model '{base_model_name_or_path}' not found in HF cache. "
        f"Pass --base-model-dir explicitly.")


def find_safetensors(model_dir):
    """Find all .safetensors files in a model directory."""
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return [os.path.join(model_dir, f) for f in files]


def convert_head(head_pt_path, output_dir):
    """Convert head.pt to head.safetensors + meta.json (same as convert-kev-head.py)."""
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
    return state


def merge_lora(base_dir, adapter_path, adapter_config, output_dir):
    """Merge LoRA adapter into base model weights and save as model.safetensors."""
    r = adapter_config["r"]
    lora_alpha = adapter_config["lora_alpha"]
    scaling = lora_alpha / r
    print(f"LoRA merge: r={r}, alpha={lora_alpha}, scaling={scaling}")

    # Load all base model safetensors shards.
    shard_paths = find_safetensors(base_dir)
    merged = {}
    for sp in shard_paths:
        print(f"  Loading base shard: {os.path.basename(sp)}")
        tensors = load_file(sp)
        merged.update(tensors)

    # Load adapter weights.
    print(f"  Loading adapter: {os.path.basename(adapter_path)}")
    adapter = load_file(adapter_path)

    # Group adapter tensors by module: base_model.model.layers.{N}.{path}.lora_{A,B}.weight
    # Base tensor name: model.language_model.layers.{N}.{path}.weight
    modules = {}  # base_name -> {"A": tensor, "B": tensor}
    for name, tensor in adapter.items():
        if not name.startswith("base_model.model.layers."):
            continue
        if not name.endswith(".weight"):
            continue
        # Strip prefix and suffix
        stripped = name[len("base_model.model.layers."):]
        # stripped = "0.linear_attn.in_proj_qkv.lora_A.weight"
        parts = stripped.rsplit(".lora_", 1)
        if len(parts) != 2:
            continue
        module_path = parts[0]  # "0.linear_attn.in_proj_qkv"
        variant = parts[1].replace(".weight", "")  # "A" or "B"

        base_name = f"model.language_model.layers.{module_path}.weight"
        if base_name not in modules:
            modules[base_name] = {}
        modules[base_name][variant] = tensor

    print(f"  Found {len(modules)} LoRA target modules")

    # Apply merge: W' = W + scaling * (lora_B @ lora_A)
    # All math in F32 for precision, then convert back to original dtype.
    merged_count = 0
    for base_name, pair in modules.items():
        if "A" not in pair or "B" not in pair:
            print(f"  WARNING: {base_name} missing lora_A or lora_B, skipping")
            continue
        if base_name not in merged:
            print(f"  WARNING: base tensor {base_name} not found, skipping")
            continue

        W = merged[base_name]
        A = pair["A"]  # [rank, in]
        B = pair["B"]  # [out, rank]

        orig_dtype = W.dtype
        W_f32 = W.float()
        A_f32 = A.float()
        B_f32 = B.float()

        # delta = scaling * (B @ A)  -> [out, in]
        delta = scaling * (B_f32 @ A_f32)
        W_merged = (W_f32 + delta).to(orig_dtype)
        merged[base_name] = W_merged
        merged_count += 1

    print(f"  Merged {merged_count}/{len(modules)} modules")

    # Save merged weights.  Drop non-backbone tensors (visual, mtp) that the
    # dense-only kev forward does not need, keeping the checkpoint small.
    keep = {}
    dropped = 0
    for name, tensor in merged.items():
        if name.startswith("model.visual.") or name.startswith("mtp."):
            dropped += 1
            continue
        keep[name] = tensor.contiguous()
    if dropped:
        print(f"  Dropped {dropped} non-backbone tensors (visual/mtp)")

    out_path = os.path.join(output_dir, "model.safetensors")
    save_file(keep, out_path)
    total_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Wrote {out_path} ({len(keep)} tensors, {total_size:.1f} MiB)")


def generate_config(base_dir, state, output_dir):
    """Generate config.json from base model config + kev metadata."""
    base_config_path = os.path.join(base_dir, "config.json")
    with open(base_config_path) as f:
        config = json.load(f)

    # Change architecture to KevModel
    config["architectures"] = ["KevModel"]

    # Add kev-specific fields (read by kev_registry.cpp)
    config["kev_head_dim"] = state["head_dim"]
    config["kev_temperature"] = state["temperature"]

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")


def copy_tokenizer(base_dir, output_dir):
    """Copy tokenizer files from base model to output directory."""
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json",
                       "merges.txt", "vocab.json"]
    for fname in tokenizer_files:
        src = os.path.join(base_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            print(f"  Copied {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert kev checkpoint to a self-contained C++-loadable model directory.")
    parser.add_argument("checkpoint_dir",
                        help="Directory containing head.pt, adapter_config.json, adapter_model.safetensors")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: checkpoint_dir)")
    parser.add_argument("--base-model-dir", default=None,
                        help="Base model directory (default: resolved from adapter_config.json via HF cache)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load adapter config
    adapter_config_path = os.path.join(args.checkpoint_dir, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    # 2. Resolve base model directory
    base_dir = resolve_base_model_dir(
        adapter_config["base_model_name_or_path"], args.base_model_dir)
    print(f"Base model: {base_dir}")

    # 3. Convert head.pt to head.safetensors + meta.json
    head_pt_path = os.path.join(args.checkpoint_dir, "head.pt")
    state = convert_head(head_pt_path, output_dir)

    # 4. Merge LoRA adapter into base model weights
    adapter_path = os.path.join(args.checkpoint_dir, "adapter_model.safetensors")
    merge_lora(base_dir, adapter_path, adapter_config, output_dir)

    # 5. Generate config.json
    generate_config(base_dir, state, output_dir)

    # 6. Copy tokenizer files
    print("Copying tokenizer files:")
    copy_tokenizer(base_dir, output_dir)

    print("\nDone. Output directory:", output_dir)
    print("Files:", os.listdir(output_dir))


if __name__ == "__main__":
    main()
