#!/usr/bin/env python3
"""Measure the LTX-2.5 prompt-AdaLN timestep term against the static per-block
table, ON THE SHIPPED CHECKPOINT'S OWN WEIGHTS.

Why this exists. `scripts/gen-ltx2-goldens.py` reports the same ratio at reduced
dimensions, but there the table AND the prompt-AdaLN MLP are both drawn from
`param_spec`'s `scale=0.05`, so the ratio it prints is a property of that fixture
and moves with the init scale. It is a gate FLOOR, not a statement about the
trained model. This script answers the other question — how much of the prompt
K/V modulation is timestep-conditioned on the model people actually run — and its
output is what `.agents/specs/ltx25-prompt-adaln.md` §Outcome quotes (issue #644).

Nothing of ours is in the numeric path. It reads the safetensors header, pulls
six tensors per stream plus the 96 per-block tables, and runs UPSTREAM's own
module:

  ltx_core/model/transformer/adaln.py:19-45          AdaLayerNormSingle
  ltx_core/model/transformer/model.py:223-227,253-257  built at embedding_coefficient=2
  ltx_core/model/transformer/transformer_args.py:274-278  driven by SIGMA, not timesteps
  ltx_core/model/transformer/transformer_args.py:177      scaled by timestep_scale_multiplier
  ltx_core/model/transformer/transformer.py:441-446       table + term, then context*(1+scale)+shift
  ltx_core/components/schedulers.py:60-88                  the sampler the file's config names

Usage:
  python3 scripts/measure-ltx2-prompt-adaln.py \
      --ltx2 ~/_git/LTX-2 \
      --checkpoint $CHECKPOINT_ROOT/ltx-2.5/lightricks-ltx-2.5/diffusion_models/\
ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PREFIX = "model.diffusion_model."
ADALN_KEYS = (
    "emb.timestep_embedder.linear_1.weight",
    "emb.timestep_embedder.linear_1.bias",
    "emb.timestep_embedder.linear_2.weight",
    "emb.timestep_embedder.linear_2.bias",
    "linear.weight",
    "linear.bias",
)


def load_upstream(root: Path):
    """Import `ltx_core` BY PATH from `root`, and PROVE that is what resolved.

    A pip-installed or otherwise-shadowing `ltx_core` would import silently and
    every number below would describe an upstream nobody pinned — the failure
    mode `.agents/specs/ltx-2-5.md` records at its "(b) byte-identical goldens"
    entry. Identity is asserted, not assumed.
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


def read_header(path: Path) -> dict:
    """The safetensors header alone: an 8-byte length prefix and its JSON. No payload."""
    with path.open("rb") as fh:
        length = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(length))


def rms(t) -> float:
    return float(t.pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ltx2", required=True, type=Path, help="path to a Lightricks/LTX-2 checkout")
    ap.add_argument("--checkpoint", required=True, type=Path, help="a shipped LTX-2.5 DiT .safetensors")
    ap.add_argument("--steps", type=int, default=8, help="step count for the shipped sampler arm")
    args = ap.parse_args()

    resolved = load_upstream(args.ltx2.expanduser())
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    from ltx_core.components.schedulers import LinearQuadraticScheduler  # noqa: PLC0415
    from ltx_core.model.transformer.adaln import AdaLayerNormSingle  # noqa: PLC0415

    ckpt = args.checkpoint.expanduser()
    print(f"ltx_core:   {resolved}")
    print(f"checkpoint: {ckpt}")

    header = read_header(ckpt)
    meta = header.get("__metadata__")
    if not meta or "config" not in meta:
        raise SystemExit(f"{ckpt} declares no __metadata__ config; this script needs the file's own geometry")
    config = json.loads(meta["config"])
    tcfg = config["transformer"]
    mult = tcfg.get("timestep_scale_multiplier", 1000)
    blocks = tcfg["num_layers"]
    dims = {
        "video": (tcfg["num_attention_heads"] * tcfg["attention_head_dim"],
                  "prompt_adaln_single.", "prompt_scale_shift_table"),
        "audio": (tcfg["audio_num_attention_heads"] * tcfg["audio_attention_head_dim"],
                  "audio_prompt_adaln_single.", "audio_prompt_scale_shift_table"),
    }
    print(f"config: num_layers={blocks} timestep_scale_multiplier={mult} "
          f"cross_attention_adaln={tcfg.get('cross_attention_adaln')} "
          f"use_prompt_adaln_single={tcfg.get('use_prompt_adaln_single', '<absent -> upstream default True>')} "
          f"sampler={config.get('scheduler', {}).get('sampler')}")

    arms = {
        "uniform sigma [0,1] x101": torch.linspace(0.0, 1.0, 101, dtype=torch.float32),
        f"shipped LinearQuadratic, {args.steps} steps": LinearQuadraticScheduler().execute(args.steps),
    }

    torch.set_grad_enabled(False)
    with safe_open(str(ckpt), framework="pt") as fh:
        for stream, (dim, sub, table_name) in dims.items():
            module = AdaLayerNormSingle(dim, embedding_coefficient=2)
            module.load_state_dict(
                {k: fh.get_tensor(PREFIX + sub + k).float() for k in ADALN_KEYS}, strict=True
            )
            module.eval()
            tables = torch.stack([
                fh.get_tensor(f"{PREFIX}transformer_blocks.{i}.{table_name}").float()
                for i in range(blocks)
            ])  # [blocks, 2, dim]; row 0 shift, row 1 scale (transformer.py:444)

            t_rms, t_max = rms(tables), float(tables.abs().max())
            print(f"\n=== {stream}  dim={dim}  blocks={blocks}")
            print(f"  rms|table|={t_rms:.6f}  max|table|={t_max:.6f}")
            for arm, sigmas in arms.items():
                term = module(sigmas * mult, hidden_dtype=torch.float32)[0]
                term = term.reshape(sigmas.numel(), 2, dim)  # transformer.py:443
                e_rms, e_max = rms(term), float(term.abs().max())
                print(f"  [{arm}] rms|term|={e_rms:.6f} -> {100.0 * e_rms / t_rms:.0f}% of rms|table|; "
                      f"max|term|={e_max:.6f} -> {100.0 * e_max / t_max:.0f}% of max|table|")
                for row, name in ((0, "shift"), (1, "scale")):
                    print(f"      {name}: rms|table|={rms(tables[:, row]):.6f} "
                          f"rms|term|={rms(term[:, row]):.6f} "
                          f"-> {100.0 * rms(term[:, row]) / rms(tables[:, row]):.0f}%")
                # What is actually consumed (transformer.py:446): context*(1+scale)+shift.
                # For a context of unit rms with independent entries the modulated
                # context has rms sqrt(mean((1+scale)^2) + mean(shift^2)).
                def modulated(scale, shift) -> float:
                    return float(((1.0 + scale).pow(2).mean() + shift.pow(2).mean()).sqrt())

                static = modulated(tables[:, 1], tables[:, 0])
                full = modulated(tables[:, 1].unsqueeze(0) + term[:, 1].unsqueeze(1),
                                 tables[:, 0].unsqueeze(0) + term[:, 0].unsqueeze(1))
                print(f"      modulated context on a UNIT-rms context: static-only={static:.4f} "
                      f"upstream={full:.4f}  ({100.0 * (full - static) / static:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
