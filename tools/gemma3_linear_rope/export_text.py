#!/usr/bin/env python3
"""Export a Gemma3ForCausalLM fixture without changing retained tensor bytes."""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provenance", type=Path,
                        default=Path(__file__).with_name("source-provenance.json"))
    args = parser.parse_args()
    provenance = json.loads(args.provenance.read_text())
    expected = {row["rfilename"]: row for row in provenance["files"]}
    args.output.mkdir(exist_ok=False)
    config = json.loads((args.source / "config.json").read_text())
    text = config["text_config"].copy()
    text["architectures"] = ["Gemma3ForCausalLM"]
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings"):
        if key in config:
            text[key] = config[key]
    (args.output / "config.json").write_text(json.dumps(text, indent=2) + "\n")
    manifest = {"source": provenance, "tensors": {}, "shards": {}}
    weight_map = {}
    total = 0
    for source in sorted(args.source.glob("*.safetensors")):
        digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
        if digest != expected[source.name]["lfs"]["sha256"]:
            raise ValueError(f"source checksum differs: {source}")
        with source.open("rb") as src:
            header_size = struct.unpack("<Q", src.read(8))[0]
            header = json.loads(src.read(header_size))
            selected = {key: value for key, value in header.items() if key.startswith("language_model.")}
            if not selected:
                continue
            out_header = {"__metadata__": {"format": "pt"}}
            offset = 0
            for key, value in selected.items():
                name = key.removeprefix("language_model.")
                size = value["data_offsets"][1] - value["data_offsets"][0]
                out_header[name] = {**value, "data_offsets": [offset, offset + size]}
                offset += size
                weight_map[name] = source.name
            encoded = json.dumps(out_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            output = args.output / source.name
            with output.open("wb") as dst:
                dst.write(struct.pack("<Q", len(encoded)))
                dst.write(encoded)
                for key, value in selected.items():
                    start, end = value["data_offsets"]
                    src.seek(8 + header_size + start)
                    remaining, h = end - start, hashlib.sha256()
                    while remaining:
                        chunk = src.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(f"truncated tensor: {key}")
                        dst.write(chunk)
                        h.update(chunk)
                        remaining -= len(chunk)
                    name = key.removeprefix("language_model.")
                    manifest["tensors"][name] = {"source_name": key, "sha256": h.hexdigest(),
                                                    "dtype": value["dtype"], "shape": value["shape"]}
            total += offset
            manifest["shards"][source.name] = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    if not weight_map:
        raise ValueError("no text tensors exported")
    for source in args.source.iterdir():
        if source.name.startswith(("tokenizer", "special_tokens", "added_tokens", "chat_template", "generation_config")):
            shutil.copyfile(source, args.output / source.name)
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
    (args.output / "text-export-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(weight_map)} tensors, {total} unchanged data bytes to {args.output}")


if __name__ == "__main__":
    main()
