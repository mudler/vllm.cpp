#!/usr/bin/env python3
"""Emit C++ goldens for the S2Mel DiT FRONT END: how x_in is built.

Upstream `indextts/s2mel/modules/diffusion_transformer.py:243-253`, index-tts
@4f8792ff120cd3ea470dd511e997a17c86cddd10, under the shipped config
(`long_skip_connection: true`, `final_layer_type: wavenet`):

    if long_skip_connection: x_res = skip_linear(cat([x_res, x], dim=-1))
    x  = conv1(x_res)                       # Linear D -> wavenet hidden
    x  = x.transpose(1, 2)                  # [B, H, T]
    t2 = t_embedder2(t)
    x  = wavenet(x, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + res_projection(x_res)
    x  = final_layer(x, t1).transpose(1, 2)
    x  = conv2(x)                           # Conv1d H -> in_channels, kernel 1

The DiT is constructed for real at reduced dims, so every module here is
upstream's own; only the SEQUENCE is restated, and it is restated once, next to
the upstream line numbers it copies.

Usage: DIT_SRC=<path to indextts/s2mel/modules> python3 \
           scripts/gen-dit-front-goldens.py --out tests/vllm/models/dit_front_goldens.inc
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def rnd(name: str, n: int, scale: float = 1.0) -> list:
    h = 0xCBF29CE484222325
    for ch in name.encode():
        h = ((h ^ ch) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(n):
        h = (h + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = h
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        out.append(((z >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0) * scale)
    return out


def tensor(name: str, shape, scale: float = 1.0) -> torch.Tensor:
    n = 1
    for d in shape:
        n *= d
    return torch.tensor(rnd(name, n, scale), dtype=torch.float64).reshape(shape).float()


def load_dit(src: Path):
    sys.path.insert(0, str(src.parents[2]))
    for name in ("munch",):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.Munch = dict
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location(
        "indextts.s2mel.modules.diffusion_transformer", src / "diffusion_transformer.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fmt(values) -> str:
    lines, row = [], []
    for v in values:
        row.append(f"{float(v):.9e}F")
        if len(row) == 6:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    return "\n".join(lines)


# Reduced dims. The shipped model is hidden 512 / wavenet 512 / in_channels 80 /
# 8 wavenet layers; the RATIOS that matter (skip_linear takes hidden + in_channels,
# conv2 maps wavenet hidden -> in_channels) are preserved.
#
# WN_HIDDEN MUST EQUAL HIDDEN. `final_layer` is built at the wavenet width but is
# called with `t1`, which the DiT embeds at ITS hidden width, so the wavenet
# final-layer path only composes when the two are equal. They both happen to be
# 512 upstream, which hides the coupling; setting them differently here raised
# `mat1 and mat2 shapes cannot be multiplied (1x8 and 6x12)` from upstream's own
# module. The C++ port asserts it rather than inheriting a silent coincidence.
HIDDEN, WN_HIDDEN, IN_CH, HEADS, DEPTH = 8, 8, 4, 2, 1
STYLE = 6
WN_LAYERS, WN_KERNEL, WN_DILATION, FRAMES = 2, 3, 1, 7


def build_args():
    dit = SimpleNamespace(
        time_as_token=False, style_as_token=False, uvit_skip_connection=True,
        depth=DEPTH, num_heads=HEADS, hidden_dim=HIDDEN, block_size=128,
        in_channels=IN_CH, content_type="discrete", content_codebook_size=16,
        content_dim=HIDDEN, is_causal=False, final_layer_type="wavenet",
        style_condition=True, class_dropout_prob=0.0, long_skip_connection=True,
        target="mel", f0_condition=False, n_f0_bins=8, content_codebooks=1,
        zero_prompt_speech_token=False, add_resblock_in_transformer=False,
    )
    wavenet = SimpleNamespace(
        hidden_dim=WN_HIDDEN, num_layers=WN_LAYERS, kernel_size=WN_KERNEL,
        dilation_rate=WN_DILATION, p_dropout=0.0, style_condition=True,
    )
    style_encoder = SimpleNamespace(dim=STYLE)
    return SimpleNamespace(DiT=dit, wavenet=wavenet, style_encoder=style_encoder)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_dit(Path(os.environ["DIT_SRC"]))

    torch.manual_seed(0)
    dit = m.DiT(build_args()).eval()

    # Every parameter on the tail comes from the shared stream, so the C++ side
    # rebuilds them without a fixture.
    tail_prefixes = ("cond_projection.", "cond_x_merge_linear.")
    with torch.no_grad():
        for pname, p in sorted(dit.named_parameters()):
            if pname.startswith(tail_prefixes):
                p.copy_(tensor("front." + pname, list(p.shape), 0.5))

    x = tensor("front.x", [1, IN_CH, FRAMES])            # channel-major, as upstream
    prompt_x = tensor("front.prompt_x", [1, IN_CH, FRAMES])
    cond = tensor("front.cond", [1, FRAMES, HIDDEN])
    style = tensor("front.style", [1, STYLE])

    with torch.no_grad():
        # ---- upstream diffusion_transformer.py:206-226, verbatim order ----
        cond_p = dit.cond_projection(cond)
        xt = x.transpose(1, 2)
        pt = prompt_x.transpose(1, 2)
        x_in = torch.cat([xt, pt, cond_p], dim=-1)
        x_in = torch.cat([x_in, style[:, None, :].repeat(1, FRAMES, 1)], dim=-1)
        cat864 = x_in.clone()
        merged = dit.cond_x_merge_linear(x_in)

        # the CFG unconditional branch: everything past in_channels zeroed
        x_in_u = cat864.clone()
        x_in_u[..., IN_CH:] = x_in_u[..., IN_CH:] * 0
        merged_u = dit.cond_x_merge_linear(x_in_u)
        # -------------------------------------------------------------------

    names = sorted(n for n, _ in dit.named_parameters() if n.startswith(tail_prefixes))

    body = [
        "// GENERATED by scripts/gen-dit-front-goldens.py -- do not edit.",
        "// Oracle: diffusion_transformer.py:243-253 (DiT tail), index-tts",
        "// @4f8792ff120cd3ea470dd511e997a17c86cddd10, under the SHIPPED config",
        "// long_skip_connection: true, final_layer_type: wavenet.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace dit_front_goldens {",
        "",
        f"inline constexpr int64_t kHidden = {HIDDEN};",
        f"inline constexpr int64_t kInChannels = {IN_CH};",
        f"inline constexpr int64_t kStyle = {STYLE};",
        f"inline constexpr int64_t kFrames = {FRAMES};",
        "",
        "inline constexpr const char* kParamNames[] = {",
    ]
    body += [f'    "front.{n}",' for n in names]
    body += [
        "};",
        "",
        "// The 864-wide concatenation before the merge -- [kFrames, 864].",
        "inline constexpr float kCat[] = {",
        fmt(cat864.reshape(-1).tolist()),
        "};",
        "",
        "// cond_x_merge_linear(cat) -- [kFrames, kHidden].",
        "inline constexpr float kMerged[] = {",
        fmt(merged.reshape(-1).tolist()),
        "};",
        "",
        "// The CFG UNCONDITIONAL branch: columns past kInChannels zeroed first.",
        "inline constexpr float kMergedUncond[] = {",
        fmt(merged_u.reshape(-1).tolist()),
        "};",
        "",
        "}  // namespace dit_front_goldens",
        "",
    ]

    Path(a.out).write_text("\n".join(body))
    print(f"wrote {a.out}: {len(names)} front params, merged {tuple(merged.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
