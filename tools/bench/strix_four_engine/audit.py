#!/usr/bin/env python3
"""Audit the exact Qwen3-4B BF16 conversion, without loading a model (#3053).

Manifest: model {revision,directory,files:{relative_path:sha256}}, converter
{revision,directory,inventory_sha256}. Inventory hashes use canonical JSON
(sorted keys, comma/colon separators), excluding .git and __pycache__ entries.
The converter inventory is also checked against the independently pinned tree.
"""
import argparse
import ctypes
from contextlib import ExitStack
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CONVERTER_REVISION = "10bf611e533d81f739128304991c5e133c6aebd8"
# All 3425 regular files in the stock git archive at CONVERTER_REVISION.
CONVERTER_INVENTORY = "ad7a105b10602373f7936b15fe9ff76c4ba349b982c985115e2520b9172e0ad7"
CHUNK_ELEMENTS = 1 << 20


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique_object)


def source_inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError("converter symlink: " + str(relative))
        if path.is_file():
            result[relative.as_posix()] = digest(path)
    return result


def mapping():
    result = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight"}
    # llama.cpp conversion/qwen.py:155 and conversion/base.py:623 at the pin.
    pairs = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
             "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
             "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
             "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
    for layer in range(36):
        for source, target in pairs.items():
            result[f"model.layers.{layer}.{source}.weight"] = f"blk.{layer}.{target}.weight"
    return result


def verify_bindings(manifest):
    model, converter = manifest["model"], manifest["converter"]
    if model["revision"] != MODEL_REVISION or converter["revision"] != CONVERTER_REVISION:
        raise ValueError("revision mismatch")
    native, tree = Path(model["directory"]), Path(converter["directory"])
    if not native.is_absolute() or not tree.is_absolute() or not native.is_dir() or not tree.is_dir():
        raise ValueError("absolute model and converter directories required")
    files = model["files"]
    if not files or not {"config.json", "model.safetensors.index.json"} <= files.keys():
        raise ValueError("missing source file bindings")
    for name, expected in files.items():
        path = native / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(native.resolve()):
            raise ValueError("source path escapes model directory")
        if not path.is_file() or digest(path) != expected:
            raise ValueError("source hash mismatch: " + name)
    inventory = source_inventory(tree)
    actual = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != converter["inventory_sha256"] or actual != CONVERTER_INVENTORY:
        raise ValueError("converter inventory mismatch")
    return native, tree


def pinned_gguf(tree):
    package = (tree / "gguf-py" / "gguf").resolve()
    # Refuse an already imported package from another installation.
    for name, module in tuple(sys.modules.items()):
        if name == "gguf" or name.startswith("gguf."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(package):
                raise ValueError("GGUF import outside pinned converter")
    sys.path.insert(0, str(package.parent))
    try:
        module = importlib.import_module("gguf")
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError("GGUF import outside pinned converter")
        return module
    finally:
        sys.path.pop(0)


def audit_model(manifest: dict, gguf_path: Path) -> dict:
    """Return PASS only after all tensors and both final file bindings match."""
    import numpy as np
    from safetensors import safe_open

    native, tree = verify_bindings(manifest)
    gguf = pinned_gguf(tree)
    gguf_path = Path(gguf_path)
    gguf_sha = digest(gguf_path)
    config = read_json(native / "config.json")
    if (config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("model_type") != "qwen3"
            or config.get("num_hidden_layers") != 36 or config.get("tie_word_embeddings") is not True):
        raise ValueError("Qwen3 tied-embedding architecture mismatch")
    names = mapping()
    index = read_json(native / "model.safetensors.index.json")["weight_map"]
    if set(index) != set(names):
        raise ValueError("source tensor inventory mismatch")
    shards = set(index.values())
    if not shards <= manifest["model"]["files"].keys():
        raise ValueError("unbound source shard")
    if {p.name for p in native.glob("*.safetensors")} != shards:
        raise ValueError("unexpected source shard")
    reader = gguf.GGUFReader(str(gguf_path))
    architecture = reader.get_field("general.architecture")
    if architecture is None or architecture.contents() != "qwen3":
        raise ValueError("GGUF architecture mismatch")
    destinations = {}
    for tensor in reader.tensors:
        if tensor.name in destinations:
            raise ValueError("duplicate GGUF tensor")
        destinations[tensor.name] = tensor
    if set(destinations) != set(names.values()):
        raise ValueError("GGUF tensor inventory mismatch")
    records = []
    with ExitStack() as stack:
        sources = {}
        for shard in sorted(shards):
            path = native / shard
            handle = stack.enter_context(safe_open(str(path), framework="numpy"))
            with path.open("rb") as raw:
                length = int.from_bytes(raw.read(8), "little")
                header = json.loads(raw.read(length), object_pairs_hook=unique_object)
            for name in handle.keys():
                if name in sources or index.get(name) != shard:
                    raise ValueError("duplicate or misplaced source tensor")
                entry = header[name]
                view = handle.get_slice(name)
                if view.get_dtype() != "BF16" or entry["dtype"] != "BF16":
                    raise ValueError("source dtype must be BF16: " + name)
                shape = view.get_shape()
                expected_rank = 1 if "norm" in names[name] else 2
                if len(shape) != expected_rank or any(d <= 0 for d in shape):
                    raise ValueError("unsupported source shape")
                count = math.prod(shape)
                begin, end = entry["data_offsets"]
                if end - begin != count * 2:
                    raise ValueError("source storage length mismatch")
                sources[name] = (shape, np.memmap(path, dtype="<u2", mode="r", offset=8 + length + begin, shape=(count,)))
        if set(sources) != set(names):
            raise ValueError("source shard tensor inventory mismatch")
        for name, target in names.items():
            shape, source = sources[name]
            tensor = destinations[target]
            if list(reversed(tensor.shape.tolist())) != shape:
                raise ValueError("logical shape mismatch: " + name)
            dtype = "F32" if len(shape) == 1 else "BF16"
            if tensor.tensor_type != getattr(gguf.GGMLQuantizationType, dtype):
                raise ValueError("destination dtype mismatch: " + name)
            destination = tensor.data.reshape(-1)
            if dtype == "BF16":
                destination = destination.view("<u2")
            if destination.size != source.size:
                raise ValueError("element count mismatch: " + name)
            for start in range(0, source.size, CHUNK_ELEMENTS):
                stop = start + CHUNK_ELEMENTS
                expected = (source[start:stop].astype(np.uint32) << 16).view(np.float32)
                actual = destination[start:stop]
                if dtype == "BF16":
                    actual = (actual.astype(np.uint32) << 16).view(np.float32)
                if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError("non-finite tensor: " + name)
                if not np.array_equal(expected, actual):
                    raise ValueError("exact value mismatch: " + name)
            records.append(dict(source=name, destination=target, shape=shape, elements=int(source.size),
                                source_dtype="BF16", destination_dtype=dtype, exact=True,
                                promotion="BF16 to F32 exact" if dtype == "F32" else None))
    verify_bindings(manifest)
    if digest(gguf_path) != gguf_sha:
        raise ValueError("GGUF changed during audit")
    return dict(result="PASS", model=manifest["model"], converter=manifest["converter"],
                gguf_sha256=gguf_sha, tensor_count=len(records), tensors=records)


def publish_result(result, output):
    """Publish a closed record atomically without replacing another output.

    Linux renameat2 is required. Pending files are not audit outputs, even when
    their bytes are complete. Only the requested final path is consumable.
    """
    if sys.platform != "linux":
        raise ValueError("atomic no-replace publication requires Linux renameat2")
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise ValueError("atomic no-replace publication requires Linux renameat2") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    pending = output.parent / (".audit-pending-" + uuid.uuid4().hex)
    created = False
    try:
        with pending.open("x", encoding="utf-8") as handle:
            created = True
            json.dump(result, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # AT_FDCWD=-100, RENAME_NOREPLACE=1. No hardlink or overwrite fallback.
        if rename(-100, os.fsencode(pending), -100, os.fsencode(output), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, "atomic no-replace publication failed: " + os.strerror(code))
        # Nothing fallible runs after publication, including temporary cleanup.
    except Exception:
        if created:
            try:
                pending.unlink()
            except OSError as error:
                print("AUDIT_PENDING_CLEANUP_FAIL: " + str(error), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("output already exists")
        result = audit_model(read_json(args.manifest), args.gguf)
        publish_result(result, args.output)
        return 0
    except Exception as error:
        print("AUDIT_FAIL: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
