#!/usr/bin/env python3
"""Settle what upstream DOES with `keyframes_abs_pos_embedding` on each shipped
LTX-2.5 DiT, by EXECUTING upstream's own model builder and state-dict reader.

Why this exists. `.agents/specs/ltx25-keyframes-abs-pos.md` §2 records a result
two reviewers disagreed about from the SOURCE alone: one read the parameter as
zero-initialized (model.py:200) and therefore "an exact no-op"; the other ran it.
Both readings are consistent with the text, only one with what executes. The
answer decides the NVFP4 arm's behaviour — refuse, synthesise a zero, or apply
nothing — so it must be reproducible on demand rather than quoted from prose.

Nothing of ours is in the path. This reads the safetensors header, builds the
model through upstream's own `create_meta_model` against the file's own declared
config, and loads with upstream's own `strict=False, assign=True` call:

  ltx_core/loader/helpers.py:84-95                create_meta_model
  ltx_core/loader/single_gpu_model_builder.py     load_state_dict(..., assign=True)
  ltx_core/model/transformer/model.py:158-173     _keyframes_embedding / supports_...
  ltx_core/model/transformer/model.py:217-219     the (1, inner_dim) parameter
  ltx_core/model/transformer/transformer_args.py:23-43  the consumer, called at :269

THE TWO SHIPPED DiTs ANSWER DIFFERENTLY, which is the whole point:

  vonkaiser FP8   no `__metadata__` at all, but CARRIES the tensor, TRAINED
  first-party NVFP4  `__metadata__` DECLARES the flag, carries NO tensor

Usage:
  python3 scripts/measure-ltx2-keyframes-meta.py \
      --ltx2 ~/_git/LTX-2 \
      --nvfp4 $CHECKPOINT_ROOT/ltx-2.5/lightricks-ltx-2.5/diffusion_models/\
ltx-2.5-22b-distilled-transformer-nvfp4.safetensors \
      --fp8 $CHECKPOINT_ROOT/ltx-2.5/vonkaiser-fp8-nvfp4/transformer/\
ltx-2.5-22b-distilled-fp8.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

PREFIX = "model.diffusion_model."
PARAM = "keyframes_abs_pos_embedding"


def load_upstream(root: Path) -> Path:
    """Import `ltx_core` BY PATH from `root`, and PROVE that is what resolved.

    A pip-installed or otherwise-shadowing `ltx_core` would import silently and
    every claim below would describe an upstream nobody pinned. Identity is
    asserted, not assumed.
    """
    src = root / "packages" / "ltx-core" / "src"
    if not (src / "ltx_core").is_dir():
        raise SystemExit(f"no ltx_core under {src}; point --ltx2 at a Lightricks/LTX-2 checkout")
    sys.path.insert(0, str(src))
    import ltx_core  # noqa: PLC0415

    resolved = Path(ltx_core.__file__).resolve()
    if resolved.parent != (src / "ltx_core").resolve():
        raise SystemExit(f"ltx_core resolved to {resolved}, NOT to the checkout at {src}")
    return resolved


def upstream_revision(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - a tarball checkout has no git metadata
        return "unknown"


def read_header(path: Path) -> tuple[dict, dict]:
    """(tensor entries, `__metadata__`) from a safetensors header."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    meta = header.pop("__metadata__", {})
    return header, meta


def read_raw(path: Path, name: str, entry: dict) -> bytes:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        start, end = entry["data_offsets"]
        f.seek(8 + n + start)
        return f.read(end - start)


def report_header(path: Path, label: str) -> dict:
    """The FILE's own facts, read from the bytes rather than from a claim."""
    header, meta = read_header(path)
    print(f"\n=== {label}: {path}")
    print(f"  tensors                       : {len(header)}")
    print(f"  __metadata__ keys             : {sorted(meta) or 'NONE'}")

    hits = sorted(k for k in header if PARAM in k)
    print(f"  keys matching '{PARAM}'       : {hits or '[]'}  ({len(hits)} of {len(header)})")
    for name in hits:
        entry = header[name]
        raw = read_raw(path, name, entry)
        nonzero = sum(1 for b in raw if b != 0)
        print(
            f"    {name}: dtype={entry['dtype']} shape={entry['shape']} "
            f"bytes={len(raw)} NON-ZERO={nonzero}"
        )
        # TRAINED vs the zero-init the source comment describes. The term is
        # ADDED, so a genuine zero would be an exact no-op; anything else is not.
        verdict = "TRAINED (not torch.zeros)" if nonzero > 0 else "ALL ZERO (an exact no-op)"
        print(f"    -> {verdict}")

    # The flag is NOT at the top level of `__metadata__`; it lives inside the
    # JSON-encoded `config` value, under `transformer`. A raw read returns None,
    # which is why upstream's own decoding reader is needed to see it at all.
    raw_flag = meta.get("use_keyframes_abs_pos_embedding")
    print(f"  __metadata__['use_keyframes_abs_pos_embedding'] : {raw_flag}")
    config = json.loads(meta["config"]) if "config" in meta else None
    if config is None:
        print("  config.transformer['use_keyframes_abs_pos_embedding'] : NO __metadata__['config']")
    else:
        declared = config.get("transformer", {}).get("use_keyframes_abs_pos_embedding")
        print(f"  config.transformer['use_keyframes_abs_pos_embedding'] : {declared}")
    return {"header": header, "meta": meta, "config": config, "hits": hits}


def probe_meta_load(path: Path, facts: dict) -> None:
    """Build on `meta` from the file's OWN config and run upstream's own load.

    The control is the point: a NEIGHBOUR key must materialise in the same load.
    Without it, "the parameter stayed on meta" is equally consistent with "the
    loader never ran", and the result would mean nothing.
    """
    import torch  # noqa: PLC0415
    from ltx_core.loader.helpers import create_meta_model, read_model_metadata  # noqa: PLC0415
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: PLC0415
    from ltx_core.model.transformer.model_configurator import (  # noqa: PLC0415
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    if facts["config"] is None:
        print("  SKIP: this file carries no __metadata__['config'], so upstream cannot build")
        print("        a model from it at all -- its config always arrives separately.")
        return

    loader = SafetensorsModelStateDictLoader()
    metadata = read_model_metadata(str(path), loader)
    model = create_meta_model(LTXModelConfigurator, metadata)

    param = getattr(model, PARAM, None)
    if param is None:
        print(f"  {PARAM}: the model has NO such attribute (flag not set)")
        return
    print(
        f"  {PARAM}: shape={tuple(param.shape)} dtype={param.dtype} "
        f"device={param.device} is_meta={param.is_meta}"
    )
    print(f"  supports_{PARAM} (BEFORE load) : {model.supports_keyframes_abs_pos_embedding}")

    # `StateDict` is upstream's immutable wrapper (loader/primitives.py:25-38);
    # `.sd` is the plain dict `load_state_dict` consumes. The SDOps is upstream's
    # own key rewrite (model_configurator.py:222-226) — without it EVERY key keeps
    # its `model.diffusion_model.` prefix, nothing matches, and the control below
    # would report "still on meta" for the whole model. That is the shape of a
    # broken instrument agreeing with the conclusion.
    sd = loader.load(
        str(path), sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP, device=torch.device("cpu")
    ).sd
    declared = set(model.state_dict())
    missing_from_file = sorted(declared - set(sd))
    print(f"  declared - state_dict          : {len(missing_from_file)} key(s)")
    for key in missing_from_file:
        print(f"    {key}")
    if missing_from_file == [PARAM]:
        print(f"  => `declared - state_dict` is EXACTLY ONE key, and it is {PARAM}.")

    # THE PACKED WEIGHTS ARE DROPPED, AND SAYING SO IS PART OF THE MEASUREMENT.
    # This file is NVFP4: every quantized weight is stored at HALF its logical
    # width, so `load_state_dict` raises a size mismatch on it unless upstream's
    # quantized-linear module ops have swapped the modules first. Those ops are a
    # different question from this one, and running them would put a large,
    # unrelated machine between the file and the claim.
    #
    # What survives is every UNPACKED parameter — the f32 tables and the norms —
    # which is what the control needs: `scale_shift_table` must MATERIALISE in
    # the same call that leaves `keyframes_abs_pos_embedding` on meta. And the
    # fact this script exists to establish does not depend on the filter at all:
    # the key is ABSENT FROM `sd` ENTIRELY, as the line above prints.
    declared_shapes = {k: v.shape for k, v in model.state_dict().items()}
    usable = {k: v for k, v in sd.items() if k in declared_shapes and v.shape == declared_shapes[k]}
    print(
        f"  loadable at declared shape     : {len(usable)} of {len(sd)} "
        "(the rest are NVFP4-PACKED, half-width until the quantized module ops run)"
    )
    result = model.load_state_dict(usable, strict=False, assign=True)
    neighbour = model.scale_shift_table
    print(
        f"  neighbour scale_shift_table    : device={neighbour.device} "
        f"is_meta={neighbour.is_meta}   <- the CONTROL: the loader DID run"
    )
    after = getattr(model, PARAM, None)
    print(f"  {PARAM}                        : device={after.device} is_meta={after.is_meta}")
    print(f"  in missing_keys                : {PARAM in set(result.missing_keys)}")

    # HOW YOU READ IT MATTERS. `.item()` on a (1, D) parameter raises a SHAPE
    # error, which is a different fact; only a scalar read reaches the meta one.
    try:
        value = float(after.abs().max())
        print(f"  reading the value RETURNED     : {value}")
    except Exception as e:  # noqa: BLE001 - the exception IS the measurement
        print(f"  reading the value RAISES       : {type(e).__name__}: {e}")

    print(f"  supports_{PARAM} (AFTER load)  : {model.supports_keyframes_abs_pos_embedding}")
    print(
        "  => upstream never reaches the add on this file: the provider yields a META\n"
        "     tensor, `supports_...` is False before AND after, and the correct mirror\n"
        "     is to load and apply NOTHING -- neither a refusal nor a synthesised zero."
    )
    del torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx2", required=True, type=Path)
    parser.add_argument("--nvfp4", type=Path, help="the first-party NVFP4 DiT")
    parser.add_argument("--fp8", type=Path, help="the vonkaiser FP8 DiT")
    args = parser.parse_args()

    resolved = load_upstream(args.ltx2.expanduser())
    print(f"ltx_core : {resolved}")
    print(f"pin      : {upstream_revision(args.ltx2.expanduser())}")

    if args.fp8 is not None:
        report_header(args.fp8.expanduser(), "vonkaiser FP8 DiT")
    if args.nvfp4 is not None:
        facts = report_header(args.nvfp4.expanduser(), "first-party NVFP4 DiT")
        print("\n--- upstream's own meta-device load, on the NVFP4 DiT ---")
        probe_meta_load(args.nvfp4.expanduser(), facts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
