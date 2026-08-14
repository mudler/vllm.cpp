#!/usr/bin/env python3
"""Convert IndexTTS-2.5's torch `.pth` checkpoints to safetensors.

This tree has no torch-pickle reader and deliberately does not grow one: every
other lane loads safetensors or GGUF, and adding a pickle interpreter to the
engine would mean executing a serialization format designed to run arbitrary
code, inside the process that serves users. Upstream ships `.pth`, so the
conversion happens OFFLINE, once, and the engine loads the result through the
reader it already has.

The mapping is flat and lossless: nested state dicts are joined with '.', which
is exactly the naming `tests/vllm/models/indextts2_pth_manifest.json` records,
so the converted names are the manifest's names and can be checked against it.

Usage:
  python3 scripts/convert-indextts2-checkpoint.py \
      --checkpoint $CHECKPOINT_ROOT/IndexTTS-2.5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = ("gpt.pth", "codec.pth", "s2mel.pth")

# The auxiliary tensors, which are bare tensors or small dicts rather than state
# dicts. They are TINY and they are on the render path: feat1/feat2 are the
# speaker and emotion matrices `emovec::Select` consumes, and the stats are what
# normalizes the w2v-bert features. Converting them here means the engine never
# has to read pickle for them either.
AUX = ("feat1.pt", "feat2.pt", "wav2vec2bert_stats.pt")

# `codec.pth` ships optimizer state alongside the weights. It is training
# residue, cannot be loaded by anything here, and is a large fraction of the
# file, so it is dropped -- loudly, with a count, never silently.
DROP_PREFIXES = ("optimizer.",)


def flatten(obj, prefix: str = ""):
    """Yield (dotted_name, tensor) for every tensor in a nested state dict."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(value, child)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="optional indextts2_pth_manifest.json to verify against")
    a = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    a.out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in SOURCES:
        src = a.checkpoint / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        state = torch.load(src, map_location="cpu", weights_only=False)

        kept, dropped = {}, 0
        for key, tensor in flatten(state):
            if key.startswith(DROP_PREFIXES):
                dropped += 1
                continue
            # safetensors refuses shared storage; contiguous clones are the
            # documented fix and keep the values identical.
            kept[key] = tensor.detach().cpu().contiguous().clone()

        dst = a.out / (name.replace(".pth", ".safetensors"))
        save_file(kept, str(dst), metadata={"source": name, "model": "IndexTTS-2.5"})
        summary[name] = {"kept": len(kept), "dropped": dropped,
                         "out": dst.name, "bytes": dst.stat().st_size}
        print(f"{name:<12} -> {dst.name:<22} {len(kept):5d} tensors"
              f"  {dropped:4d} dropped (optimizer state)")

    if a.manifest:
        man = json.loads(a.manifest.read_text())
        for name in SOURCES:
            expected = man[name]["tensors"]
            got = summary[name]["kept"] + summary[name]["dropped"]
            status = "MATCH" if got == expected else "MISMATCH"
            print(f"  manifest {name:<12} recorded={expected:5d} seen={got:5d} {status}")
            if got != expected:
                raise SystemExit(
                    f"{name}: converted {got} tensors but the manifest records "
                    f"{expected}; the checkpoint changed under the record")

    # A bare tensor has no name of its own, so it takes the file's stem.
    aux = {}
    for name in AUX:
        src = a.checkpoint / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        obj = torch.load(src, map_location="cpu", weights_only=False)
        stem = name.replace(".pt", "")
        if hasattr(obj, "shape"):
            aux[stem] = obj.detach().cpu().contiguous().clone()
        else:
            for key, value in flatten(obj, stem):
                aux[key] = value.detach().cpu().contiguous().clone()
    dst = a.out / "aux.safetensors"
    save_file(aux, str(dst), metadata={"source": ",".join(AUX), "model": "IndexTTS-2.5"})
    summary["aux"] = {"kept": len(aux), "dropped": 0, "out": dst.name,
                      "bytes": dst.stat().st_size,
                      "names": sorted(aux)}
    print(f"{'aux':<12} -> {dst.name:<22} {len(aux):5d} tensors  "
          f"({', '.join(sorted(aux))})")

    (a.out / "conversion.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
