"""How big IS the tiled-vs-untiled gap, and does it shrink with tile size?

Control first: a tiling config that resolves to ONE tile must reproduce
`forward` exactly through `tiled_decode`. Then a sweep over tile sizes.
"""
import sys

sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-core/src")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import ltx_core  # noqa: E402

assert "/home/mudler/_git/LTX-2" in ltx_core.__file__
from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder  # noqa: E402
from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType  # noqa: E402
from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig  # noqa: E402

BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_all", {"multiplier": 2, "residual": True}),
    ("res_x_y", {"num_layers": 1, "multiplier": 2}),
    ("compress_space", {"multiplier": 1}),
    ("compress_time", {"multiplier": 1}),
    ("res_x", {"num_layers": 1}),
]
DEC = dict(
    convolution_dimensions=3,
    in_channels=6,
    out_channels=3,
    patch_size=2,
    timestep_conditioning=False,
    base_channels=8,
)
# scale factors: time 4, h/w 8

torch.manual_seed(7)
for causal in (True, False):
    dec = ConvVideoDecoder(
        decoder_blocks=BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        causal=causal,
        **DEC,
    ).eval()
    for lat_shape in [(1, 6, 5, 4, 4), (1, 6, 9, 12, 12)]:
        latent = torch.randn(lat_shape)
        with torch.no_grad():
            untiled = dec(latent)
        scale = float(untiled.abs().max())
        print(f"\ncausal={causal} latent={lat_shape[2:]} out={tuple(untiled.shape[2:])} |out|max={scale:.4f}")
        for f_tile, f_ovl, s_tile, s_ovl in [
            (10000, 0, 10000, 0),  # control: no split (tile bigger than the axis)
            (12, 4, 16, 8),
            (16, 4, 24, 8),
            (20, 8, 32, 8),
            (24, 8, 48, 16),
        ]:
            cfg = TileSizeConfig(
                frames=DimensionSizeConfig(tile_size=f_tile, overlap=f_ovl),
                height=DimensionSizeConfig(tile_size=s_tile, overlap=s_ovl),
                width=DimensionSizeConfig(tile_size=s_tile, overlap=s_ovl),
            )
            with torch.no_grad():
                chunks = list(dec.tiled_decode(latent, cfg))
            tiled = torch.cat(chunks, dim=2)
            if tiled.shape != untiled.shape:
                print(f"  tile f={f_tile}/{f_ovl} s={s_tile}/{s_ovl}: SHAPE {tuple(tiled.shape)} != {tuple(untiled.shape)}")
                continue
            d = (untiled - tiled).abs()
            print(
                f"  tile f={f_tile}/{f_ovl} s={s_tile}/{s_ovl}: chunks={len(chunks)} "
                f"max={float(d.max()):.6g} mean={float(d.mean()):.6g} "
                f"rel_max={float(d.max())/scale:.4f}"
            )
