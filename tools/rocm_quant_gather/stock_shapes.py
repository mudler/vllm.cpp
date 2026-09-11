#!/usr/bin/env python3
"""Export applicable pinned stock GET_ROWS geometry from the frozen corpus.

Stock b10451 tests/test-backend-ops.cpp:8371-8376 uses width256, rows5,
four IDs, batch1 and a contiguous view. Keep those dimensions for the three
secondary formats. Fix IDs to first/last/middle/repeated and reuse recorded
packed bytes, adapting upstream's unrecorded random values for reproduction.
Batched tables and strided IDs are refused through the shared local ABI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from primary import seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.manifest.read_text())
    result = {"version": 1, "source": "stock-GET_ROWS-applicable-shapes",
              "source_manifest": seal(args.manifest), "cases": []}
    for case in original["cases"]:
        if case["type"] not in (15, 39, 66) or case["width"] != 256:
            continue
        packed_path = args.manifest.parent / case["packed"]
        packed = packed_path.read_bytes()
        if case["rows"] < 5 or len(packed) % case["rows"] != 0:
            raise ValueError("source table cannot supply five complete rows")
        exported = dict(case, rows=5, source_packed=seal(packed_path))
        (args.output / exported["packed"]).write_bytes(packed[:5 * (len(packed) // case["rows"])])
        (args.output / exported["ids"]).write_bytes(struct.pack("<4i", 0, 4, 2, 0))
        result["cases"].append(exported)
    if len(result["cases"]) != 3:
        raise ValueError("source manifest must supply all three secondary codecs")
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
