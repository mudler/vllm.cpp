#!/usr/bin/env python3
"""Assert the identity of an INSTALLED SGLang tree against the committed manifest.

Row `SGLANG-ORACLE-LEASE-WHEEL`, spec `.agents/specs/sglang-wheel-in-lease.md`,
issue #1265.

`sglang/_version.py` in the PyPI wheel sets `__commit_id__ = None`, so the
installed package carries NO runtime assertion of the commit it was built from.
This script is that assertion. It re-derives a per-file sha256 of the installed
`sglang/` tree and compares it against
`.agents/specs/sglang-wheel-in-lease.json`, aborting non-zero on ANY mismatch,
missing file, or extra file.

Run it from `/` so it reads the installed package and never a source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MAX_OFFENDERS = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive(root: Path, prefix: str, excludes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        # Prune excluded directories in place so os.walk does not descend.
        dirnames[:] = [
            d for d in dirnames if not any(e.rstrip("/") == d for e in excludes)
        ]
        for name in filenames:
            rel = (rel_dir / name).as_posix()
            if rel.startswith("./"):
                rel = rel[2:]
            key = f"{prefix}{rel}" if rel != "." else prefix
            out[key] = sha256_file(Path(dirpath) / name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to sglang-wheel-in-lease.json")
    args = ap.parse_args(argv)

    if Path.cwd() != Path("/"):
        print(f"FATAL: run this from / , not {Path.cwd()}", file=sys.stderr)
        return 3

    manifest = json.loads(Path(args.manifest).read_text())
    prefix = manifest["root"]  # "sglang/"
    excludes = manifest["exclude"]
    expected: dict[str, str] = manifest["files"]

    import sglang  # noqa: E402  (deliberately after the cwd assertion)

    pkg_file = Path(sglang.__file__).resolve()
    pkg_dir = pkg_file.parent
    print(f"sglang.__file__ = {pkg_file}")
    print(f"sglang.__version__ = {getattr(sglang, '__version__', 'MISSING')}")
    if "site-packages" not in str(pkg_dir):
        print(f"FATAL: not an installed package: {pkg_dir}", file=sys.stderr)
        return 4
    if pkg_dir.name != prefix.rstrip("/"):
        print(f"FATAL: package dir {pkg_dir} does not end in {prefix}", file=sys.stderr)
        return 4

    actual = derive(pkg_dir, prefix, excludes)

    print(f"manifest_files={len(expected)} derived_files={len(actual)}")

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differing = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])

    print(f"missing={len(missing)} extra={len(extra)} differing={len(differing)}")
    for label, items in (("MISSING", missing), ("EXTRA", extra), ("DIFFERING", differing)):
        for k in items[:MAX_OFFENDERS]:
            print(f"  {label}: {k}")
        if len(items) > MAX_OFFENDERS:
            print(f"  {label}: ... and {len(items) - MAX_OFFENDERS} more")

    if missing or extra or differing:
        print("IDENTITY FAILED")
        return 1

    print(f"IDENTITY OK: {len(actual)} files match the manifest for pin {manifest['pin']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
