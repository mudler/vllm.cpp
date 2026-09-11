#!/usr/bin/env python3
"""Deterministic 392x392 RGB test image for the DeepSeek-V4 vision W6 parity gate.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm` W6, issue #2411.

Why 392x392: it is a multiple of the 14-pixel patch, its area (153,664) is above
the 147,456 `image_min_pixels` floor, and its 10x10 aligner grid gives a
114-token block under the 381-token budget. So neither the pinned llama.cpp
preprocessor (`mtmd-image.cpp::mtmd_image_preprocessor_deepseek4v::preprocess`,
whose `img_tool::resize` copies when source and target sizes are equal) nor ours
(`DeepSeekV4ImageProcessor::ProcessImage`, which transforms only when
`height != best_height || width != best_width`) resamples it. The comparison
therefore measures the tower, not two resamplers.

The content is smooth gradients plus sharp shapes, so the 2-D RoPE and the
3x3 unfold order both have spatial structure to act on. No asymmetry is
accidental: a transposed or row-swapped image would not match itself.

Writes <out>.rgb (raw HWC uint8) and <out>.png (the SAME bytes, lossless) using
only the standard library, so the worker needs no PIL.
"""
import hashlib
import math
import struct
import sys
import zlib

W = H = 392


def pixel(x, y):
    r = int(255 * x / (W - 1))
    g = int(255 * y / (H - 1))
    b = int(127.5 + 127.5 * math.sin(math.hypot(x - 120, y - 260) / 9.0))
    # Filled rectangle, top-left quadrant, asymmetric.
    if 40 <= x < 150 and 30 <= y < 90:
        r, g, b = 250, 20, 30
    # Disc, lower right.
    if (x - 290) ** 2 + (y - 300) ** 2 < 55 ** 2:
        r, g, b = 10, 200, 40
    # A diagonal bar that crosses aligner-cell boundaries.
    if abs((x - y) - 60) < 4 and x > 180:
        r, g, b = 0, 0, 0
    # A one-pixel white grid every 42 px (3 patches), off-phase by 7.
    if x % 42 == 7 or y % 42 == 7:
        r, g, b = 255, 255, 255
    return r, g, b


def main():
    out = sys.argv[1]
    rows = []
    raw = bytearray()
    for y in range(H):
        row = bytearray()
        for x in range(W):
            row.extend(pixel(x, y))
        raw.extend(row)
        rows.append(b"\x00" + bytes(row))
    with open(out + ".rgb", "wb") as f:
        f.write(raw)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    png += chunk(b"IEND", b"")
    with open(out + ".png", "wb") as f:
        f.write(png)
    print("rgb sha256", hashlib.sha256(raw).hexdigest(), len(raw))
    print("png sha256", hashlib.sha256(png).hexdigest(), len(png))


if __name__ == "__main__":
    main()
