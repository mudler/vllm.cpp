#!/usr/bin/env python3
"""Emit C++ goldens for the S2Mel DiT TAIL: everything after the transformer.

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
           scripts/gen-dit-tail-goldens.py --out tests/vllm/models/dit_tail_goldens.inc
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
    style_encoder = SimpleNamespace(dim=IN_CH)
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
    tail_prefixes = (
        "skip_linear.", "conv1.", "conv2.", "t_embedder2.", "wavenet.",
        "final_layer.", "res_projection.",
    )
    with torch.no_grad():
        for pname, p in sorted(dit.named_parameters()):
            if pname.startswith(tail_prefixes):
                p.copy_(tensor("tail." + pname, list(p.shape), 0.5))

    x_res = tensor("tail.x_res", [1, FRAMES, HIDDEN])
    x = tensor("tail.x", [1, FRAMES, IN_CH])
    t = torch.tensor([0.37], dtype=torch.float32)
    t1 = tensor("tail.t1", [1, HIDDEN])
    x_mask = torch.ones(1, 1, FRAMES)

    with torch.no_grad():
        # ---- upstream diffusion_transformer.py:243-253, verbatim order ----
        xr = dit.skip_linear(torch.cat([x_res, x], dim=-1))
        h = dit.conv1(xr)
        after_conv1 = h.reshape(-1).tolist()
        h = h.transpose(1, 2)
        t2 = dit.t_embedder2(t)
        h = dit.wavenet(h, x_mask, g=t2.unsqueeze(2)).transpose(1, 2) + dit.res_projection(xr)
        h = dit.final_layer(h, t1).transpose(1, 2)
        out = dit.conv2(h)
        # -------------------------------------------------------------------

    names = sorted(n for n, _ in dit.named_parameters() if n.startswith(tail_prefixes))

    body = [
        "// GENERATED by scripts/gen-dit-tail-goldens.py -- do not edit.",
        "// Oracle: diffusion_transformer.py:243-253 (DiT tail), index-tts",
        "// @4f8792ff120cd3ea470dd511e997a17c86cddd10, under the SHIPPED config",
        "// long_skip_connection: true, final_layer_type: wavenet.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace dit_tail_goldens {",
        "",
        f"inline constexpr int64_t kHidden = {HIDDEN};",
        f"inline constexpr int64_t kWnHidden = {WN_HIDDEN};",
        f"inline constexpr int64_t kInChannels = {IN_CH};",
        f"inline constexpr int64_t kWnLayers = {WN_LAYERS};",
        f"inline constexpr int64_t kWnKernel = {WN_KERNEL};",
        f"inline constexpr int64_t kWnDilation = {WN_DILATION};",
        f"inline constexpr int64_t kFrames = {FRAMES};",
        f"inline constexpr float kT = {float(t.item()):.9e}F;",
        "",
        "inline constexpr const char* kParamNames[] = {",
    ]
    body += [f'    "tail.{n}",' for n in names]
    body += [
        "};",
        "",
        "// conv1(skip_linear(cat)) BEFORE the wavenet -- [kFrames, kWnHidden].",
        "inline constexpr float kAfterConv1[] = {",
        fmt(after_conv1),
        "};",
        "",
        "// The tail's output -- [kInChannels, kFrames], channel-major.",
        "inline constexpr float kOut[] = {",
        fmt(out.reshape(-1).tolist()),
        "};",
        "",
        "}  // namespace dit_tail_goldens",
        "",
    ]

    Path(a.out).write_text("\n".join(body))
    print(f"wrote {a.out}: {len(names)} tail params, out {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
