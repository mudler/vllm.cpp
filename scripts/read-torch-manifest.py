#!/usr/bin/env python3
"""Read a torch .pth's tensor manifest by HTTP range request, no weights.

A torch save file is a ZIP holding one small pickle (`*/data.pkl`) that names
every tensor and its shape/dtype, plus one uncompressed blob per storage. The
pickle is kilobytes; the blobs are gigabytes. Fetching the central directory and
then just the pickle reads the whole manifest for a few hundred KB.
"""

import io
import pickle
import struct
import subprocess
import sys


def get(url: str, start: int, end: int) -> bytes:
    out = subprocess.run(
        ["curl", "-sL", "-r", f"{start}-{end}", url],
        capture_output=True, check=True, timeout=180,
    )
    return out.stdout


def size(url: str) -> int:
    out = subprocess.run(
        ["curl", "-sIL", url], capture_output=True, check=True, text=True, timeout=60
    )
    lengths = [l for l in out.stdout.splitlines() if l.lower().startswith("content-length")]
    return int(lengths[-1].split(":")[1])


class _StatefulDict(dict):
    """A dict that tolerates pickle BUILD, which plain dict does not."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


class ShapeRecorder(pickle.Unpickler):
    """Unpickle the manifest without torch: every rebuild becomes a record."""

    def find_class(self, module, name):
        def stub(*args, **kwargs):
            if name == "_rebuild_from_type_v2":
                # (func, new_type, args, state) -> func(*args)
                func, _new_type, fargs = args[0], args[1], args[2]
                return func(*fargs) if callable(func) else f"<{func}>"
            if name.startswith("_rebuild_tensor"):
                # (storage, offset, size, stride, ...)
                storage = args[0]
                shape = list(args[2])
                return {"shape": shape, "dtype": storage[1] if isinstance(storage, tuple) else "?"}
            if name in ("OrderedDict", "dict"):
                return dict(*args, **kwargs)
            return f"<{module}.{name}>"
        if name == "OrderedDict":
            return _StatefulDict
        return stub

    def persistent_load(self, pid):
        # ('storage', <dtype class>, key, location, numel)
        if isinstance(pid, tuple) and len(pid) >= 2:
            dt = pid[1]
            return ("storage", getattr(dt, "__name__", str(dt)))
        return ("storage", "?")


def manifest(url: str) -> dict:
    total = size(url)
    tail = get(url, max(0, total - 200_000), total - 1)
    # Locate the End Of Central Directory record in the tail we fetched.
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise SystemExit("no EOCD; zip64 or a bigger tail needed")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12 : eocd + 20])
    cd = get(url, cd_off, cd_off + cd_size - 1)

    # Walk central-directory entries for the one ending in data.pkl.
    pos, target = 0, None
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        comp_size, uncomp_size = struct.unpack("<II", cd[pos + 20 : pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        lho = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + nlen].decode()
        if name.endswith("data.pkl"):
            target = (name, lho, comp_size, uncomp_size)
        pos += 46 + nlen + elen + clen
    if target is None:
        raise SystemExit("no data.pkl in the archive")

    name, lho, comp, uncomp = target
    head = get(url, lho, lho + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    data_off = lho + 30 + nlen + elen
    blob = get(url, data_off, data_off + comp - 1)
    obj = ShapeRecorder(io.BytesIO(blob)).load()
    return {"file": url.rsplit("/", 1)[-1], "bytes": total, "pickle": name, "obj": obj}


if __name__ == "__main__":
    import json
    import re

    for url in sys.argv[1:]:
        m = manifest(url)
        obj = m["obj"]
        while isinstance(obj, dict) and len(obj) == 1 and not any(
            isinstance(v, dict) and "shape" in v for v in obj.values()
        ):
            obj = next(iter(obj.values()))
        tensors = {k: v for k, v in obj.items() if isinstance(v, dict) and "shape" in v} \
            if isinstance(obj, dict) else {}
        pat = {}
        for k, v in tensors.items():
            pat.setdefault(re.sub(r"\.\d+\.", ".N.", k), v)
        print(f"=== {m['file']}  {m['bytes']/2**30:.2f} GiB  {len(tensors)} tensors "
              f"({len(pat)} patterns) ===")
        for k, v in sorted(pat.items()):
            print(f"  {k:<62} {v['shape']} {v['dtype']}")
