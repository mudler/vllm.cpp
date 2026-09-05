#!/usr/bin/env python3
"""Map each fat_elf block of a CUDA extension to its archs and its kernel names."""
import struct, sys, re
import zstandard
from elfsec import read_sections

MAGIC = struct.pack("<I", 0xBA55ED50)
NAME = re.compile(rb"_Z[0-9A-Za-z_]{4,200}")


def blocks(path):
    data = read_sections(path, (".nv_fatbin",))[0][1]
    n = len(data)
    off = data.find(MAGIC)
    idx = 0
    dc = zstandard.ZstdDecompressor()
    while off != -1:
        magic, ver, hsize, size = struct.unpack_from("<IHHQ", data, off)
        end = off + hsize + size
        if hsize < 16 or end > n:
            off = data.find(MAGIC, off + 1)
            continue
        idx += 1
        archs, names = [], set()
        e = off + hsize
        while e + 0x20 <= end:
            kind, _v, ehsize = struct.unpack_from("<HHI", data, e)
            (psize,) = struct.unpack_from("<Q", data, e + 8)
            (arch,) = struct.unpack_from("<I", data, e + 0x1c)
            if ehsize < 0x20 or ehsize > 0x400 or e + ehsize + psize > end:
                break
            archs.append(("PTX" if kind == 1 else "ELF", arch))
            blob = data[e + ehsize:e + ehsize + psize]
            if blob[:4] == b"\x28\xb5\x2f\xfd":
                try:
                    blob = dc.decompress(blob, max_output_size=64 << 20)
                except Exception:
                    pass
            names.update(m.group(0).decode("ascii", "replace")
                         for m in NAME.finditer(blob))
            e += ehsize + psize
        yield idx, archs, names
        off = data.find(MAGIC, end)


path = sys.argv[1]
want = sys.argv[2:]
for idx, archs, names in blocks(path):
    hit = [w for w in want if any(w in nm for nm in names)]
    if hit or not want:
        print("block %3d  archs=%s  matches=%s" % (idx, archs, hit))
        if hit:
            for nm in sorted(n for n in names if any(w in n for w in hit))[:4]:
                print("      %s" % nm[:150])
