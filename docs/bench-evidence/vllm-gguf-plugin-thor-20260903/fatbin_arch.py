#!/usr/bin/env python3
"""List the SM architectures a CUDA fatbinary carries, per section.

Byte-scans every fat_elf_header (magic 0xBA55ED50) in the named ELF sections,
then walks that block's entries. Entry layout: kind uint16 at 0x00 (1 = PTX,
2 = ELF/SASS), header_size uint32 at 0x04, payload_size uint64 at 0x08,
SM version uint32 at 0x1c.
"""
import collections, struct, sys
from elfsec import read_sections

MAGIC = struct.pack("<I", 0xBA55ED50)


def parse(data):
    out = collections.Counter()
    blocks = 0
    covered = 0
    n = len(data)
    off = data.find(MAGIC)
    while off != -1:
        if off + 16 > n:
            break
        magic, ver, hsize, size = struct.unpack_from("<IHHQ", data, off)
        end = off + hsize + size
        if hsize < 16 or end > n:
            off = data.find(MAGIC, off + 1)
            continue
        blocks += 1
        covered += hsize + size
        e = off + hsize
        while e + 0x20 <= end:
            kind, _v, ehsize = struct.unpack_from("<HHI", data, e)
            (psize,) = struct.unpack_from("<Q", data, e + 8)
            (arch,) = struct.unpack_from("<I", data, e + 0x1c)
            if ehsize < 0x20 or ehsize > 0x400 or e + ehsize + psize > end:
                break
            out[("PTX" if kind == 1 else "ELF" if kind == 2 else "k%d" % kind,
                 arch)] += 1
            e += ehsize + psize
        off = data.find(MAGIC, end)
    return out, blocks, covered, n


for path in sys.argv[1:]:
    print(path)
    total = collections.Counter()
    for name, data in read_sections(path, (".nv_fatbin", ".nvFatBinSegment")):
        counts, blocks, covered, n = parse(data)
        total.update(counts)
        print("  %-16s %10d bytes, %d fat_elf blocks, %d bytes covered (%.1f%%)"
              % (name, n, blocks, covered, 100.0 * covered / n if n else 0.0))
    for (kind, arch), c in sorted(total.items()):
        print("    %-4s sm_%-4s entries=%d" % (kind, arch, c))
    print("  SASS archs: %s" % sorted({a for (k, a) in total if k == "ELF"}))
    print("  PTX  archs: %s" % sorted({a for (k, a) in total if k == "PTX"}))
