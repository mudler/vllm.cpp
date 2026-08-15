"""Same sweep, on a ladder shaped like the SHIPPED one: res_x + compress_* only.

`res_x_y` builds `norm3 = nn.GroupNorm(num_groups=1, ...)` (resnet.py:91-97),
a GLOBAL reduction over all of (C, T, H, W). A decoder containing one cannot have
a local tiled approximation at all. The shipped ltx-2.5-video-vae-conv ladder has
no `res_x_y`.
"""
import sys

sys.path.insert(0, "/home/mudler/_git/LTX-2/packages/ltx-core/src")
import torch  # noqa: E402
from ltx_core.model.video_vae.conv_video_decoder import ConvVideoDecoder  # noqa: E402
from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType  # noqa: E402
from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig  # noqa: E402

BLOCKS = [
    ("res_x", {"num_layers": 1}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 1}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 1}),
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

torch.manual_seed(7)
for causal in (True, False):
    dec = ConvVideoDecoder(
        decoder_blocks=BLOCKS,
        norm_layer=NormLayerType.PIXEL_NORM,
        decoder_spatial_padding_mode=PaddingModeType.REFLECT,
        causal=causal,
        **DEC,
    ).eval()
    for shape in [(1, 6, 5, 4, 4), (1, 6, 9, 8, 8)]:
        lat = torch.randn(shape)
        with torch.no_grad():
            u = dec(lat)
        s = float(u.abs().max())
        print(f"\ncausal={causal} latent={shape[2:]} out={tuple(u.shape[2:])} |out|max={s:.4f}")
        for ft, fo, st, so in [(10000, 0, 10000, 0), (12, 4, 16, 8), (16, 4, 24, 8), (20, 8, 32, 16)]:
            cfg = TileSizeConfig(
                frames=DimensionSizeConfig(tile_size=ft, overlap=fo),
                height=DimensionSizeConfig(tile_size=st, overlap=so),
                width=DimensionSizeConfig(tile_size=st, overlap=so),
            )
            with torch.no_grad():
                ch = list(dec.tiled_decode(lat, cfg))
            t = torch.cat(ch, dim=2)
            if t.shape != u.shape:
                print(f"  f={ft}/{fo} s={st}/{so}: SHAPE {tuple(t.shape)} != {tuple(u.shape)}")
                continue
            d = (u - t).abs()
            print(
                f"  f={ft}/{fo} s={st}/{so}: chunks={len(ch)} max={float(d.max()):.6g} "
                f"mean={float(d.mean()):.6g} rel={float(d.max()) / s:.4f}"
            )
