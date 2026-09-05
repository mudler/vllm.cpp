#!/usr/bin/env python3
"""Read sections out of a 64-bit little-endian ELF, host-arch independent."""
import struct


def read_sections(path, wanted):
    f = open(path, "rb")
    ident = f.read(16)
    assert ident[:4] == b"\x7fELF", "not an ELF"
    assert ident[4] == 2 and ident[5] == 1, "expect ELF64 little-endian"
    f.seek(0x28)
    (e_shoff,) = struct.unpack("<Q", f.read(8))
    f.seek(0x3A)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack("<HHH", f.read(6))
    f.seek(e_shoff + e_shstrndx * e_shentsize)
    sh = f.read(e_shentsize)
    str_off, str_size = struct.unpack_from("<QQ", sh, 0x18)
    f.seek(str_off)
    strtab = f.read(str_size)
    out = []
    for i in range(e_shnum):
        f.seek(e_shoff + i * e_shentsize)
        sh = f.read(e_shentsize)
        (name_off,) = struct.unpack_from("<I", sh, 0)
        name = strtab[name_off:strtab.index(b"\x00", name_off)].decode()
        if name in wanted:
            off, size = struct.unpack_from("<QQ", sh, 0x18)
            f.seek(off)
            out.append((name, f.read(size)))
    return out
