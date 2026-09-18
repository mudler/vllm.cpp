#!/usr/bin/env python3
"""Compare every native gather capture with its independently executed oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def seal(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def values(data: bytes, dtype: str) -> list[float]:
    if dtype == "f32":
        return [item[0] for item in struct.iter_unpack("<f", data)]
    return [struct.unpack("<f", struct.pack("<I", item[0] << 16))[0]
            for item in struct.iter_unpack("<H", data)]


def bf16_from_f32(data: bytes) -> bytes:
    result = bytearray()
    for (bits,) in struct.iter_unpack("<I", data):
        if bits & 0x7F800000 == 0x7F800000 and bits & 0x7FFFFF:
            word = (bits >> 16) | 0x40
        else:
            word = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
        result.extend(struct.pack("<H", word))
    return bytes(result)


def compare(manifest: Path, native: Path, primary: Path, secondary: Path, output: Path,
            scope: str = "synthetic operation parity; original fixture and model gates remain separate") -> None:
    cases = json.loads(manifest.read_text())["cases"]
    native_report = json.loads((native / "report.json").read_text())
    needs_primary = any(case["type"] not in (15, 39, 66) for case in cases)
    primary_report = json.loads((primary / "report.json").read_text()) if needs_primary else {"cases": []}
    native_entries = {(entry["name"], entry["dtype"]): entry for entry in native_report["cases"]}
    primary_entries = {(entry["name"], entry["dtype"]): entry for entry in primary_report["cases"]}
    if len(native_entries) != 2 * len(cases) or len(native_report["cases"]) != len(native_entries):
        raise ValueError("native report must contain exactly two outputs per input case")
    results = []
    for case in cases:
        count = len((manifest.parent / case["ids"]).read_bytes()) // 4 * case["width"]
        is_secondary = case["type"] in (15, 39, 66)
        for dtype, element_bytes in (("f32", 4), ("bf16", 2)):
            entry = native_entries[(case["name"], dtype)]
            actual_path = native / entry["result"]
            actual = actual_path.read_bytes()
            if len(actual) != count * element_bytes or entry["output_bytes"] != len(actual):
                raise ValueError("native output byte count mismatch")
            if is_secondary:
                expected_path = secondary / f"{case['name']}-f32.bin"
                expected = expected_path.read_bytes()
                if dtype == "bf16":
                    expected = bf16_from_f32(expected)
                oracle = "fork" if case["type"] == 66 else "stock"
                tolerance = "byte-exact, with explicit f32-to-bf16 RNE for bf16 output"
            else:
                oracle_entry = primary_entries[(case["name"], dtype)]
                if oracle_entry["packed"] != seal(manifest.parent / case["packed"]):
                    raise ValueError("primary capture used different packed bytes")
                if oracle_entry["ids"] != seal(manifest.parent / case["ids"]):
                    raise ValueError("primary capture used different selected IDs")
                expected_path = primary / oracle_entry["result"]
                if oracle_entry["seal"] != seal(expected_path):
                    raise ValueError("primary output changed after capture")
                expected = expected_path.read_bytes()
                oracle = "primary-plugin"
                tolerance = "atol=0.01 rtol=0.04, pinned upstream embedding test"
            if len(expected) != len(actual):
                raise ValueError("oracle output byte count mismatch")
            a_values = values(actual, dtype)
            b_values = values(expected, dtype)
            finite = all(math.isfinite(a) and math.isfinite(b) for a, b in zip(a_values, b_values))
            mismatches = sum(not (abs(a - b) <= 0.01 + 0.04 * abs(b))
                             for a, b in zip(a_values, b_values))
            passed = finite and (actual == expected if is_secondary else mismatches == 0)
            results.append({"name": case["name"], "dtype": dtype, "oracle": oracle,
                            "elements": count, "table_bytes": entry["table_bytes"],
                            "output_bytes": len(actual), "scratch_bytes": entry["scratch_bytes"],
                            "tolerance": tolerance, "byte_exact": actual == expected,
                            "max_absolute_error": max(abs(a - b) for a, b in zip(a_values, b_values)),
                            "finite": finite, "pass": passed,
                            "native": {"file": str(actual_path), **seal(actual_path)},
                            "oracle_output": {"file": str(expected_path), **seal(expected_path)}})
    passed = all(result["pass"] for result in results)
    output.write_text(json.dumps({"manifest": seal(manifest), "cases": results,
                                 "outputs": len(results), "all_pass": passed,
                                 "scope": scope},
                                indent=2) + "\n")
    if not passed:
        raise SystemExit("one or more native outputs differ from the pinned oracle")
    print(f"PASS: {len(results)} outputs, {sum(result['byte_exact'] for result in results)} byte-exact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scope", default="synthetic operation parity; original fixture and "
                        "model gates remain separate",
                        help="Report scope line describing the inputs this run compared.")
    args = parser.parse_args()
    compare(args.manifest, args.native, args.primary, args.secondary, args.output, args.scope)


if __name__ == "__main__":
    main()
