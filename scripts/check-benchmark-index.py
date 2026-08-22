#!/usr/bin/env python3
"""Validate the mechanical shape of the public benchmark index."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "BENCHMARKS.md"
DETAILS = ROOT / "docs" / "benchmarks"
ROW = re.compile(
    r"^\|\s*`(?P<id>[a-z0-9]+(?:-[a-z0-9]+)*)`\s*\|[^\n]*\|\s*"
    r"\[[^]]+\]\(benchmarks/(?P<file>[a-z0-9]+(?:-[a-z0-9]+)*)\.md\)\s*\|$"
)


def benchmark_index_errors(text: str, detail_files: set[str]) -> list[str]:
    """Return only link, identity, and ownership errors for the index."""
    rows = []
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            rows.append((match.group("id"), f"{match.group('file')}.md"))

    errors: list[str] = []
    counts = Counter(benchmark_id for benchmark_id, _ in rows)
    for benchmark_id, count in sorted(counts.items()):
        if count > 1:
            errors.append(f"duplicate benchmark ID `{benchmark_id}`")

    linked_files: set[str] = set()
    for benchmark_id, filename in rows:
        linked_files.add(filename)
        stem = filename.removesuffix(".md")
        if benchmark_id != stem:
            errors.append(
                f"benchmark ID `{benchmark_id}` does not match detail file stem `{stem}`"
            )
        if filename not in detail_files:
            errors.append(f"benchmark `{benchmark_id}` has missing detail file {filename}")

    for filename in sorted(detail_files - linked_files):
        errors.append(f"orphan benchmark detail file {filename}")
    return errors


def validate_root(root: Path) -> list[str]:
    index = root / "docs" / "BENCHMARKS.md"
    details = root / "docs" / "benchmarks"
    if not index.is_file():
        return ["docs/BENCHMARKS.md is missing"]
    if not details.is_dir():
        return ["docs/benchmarks is missing"]
    files = {path.name for path in details.glob("*.md")}
    return benchmark_index_errors(index.read_text(encoding="utf-8"), files)


def main() -> int:
    errors = validate_root(ROOT)
    for error in errors:
        print(f"benchmark index error: {error}", file=sys.stderr)
    if errors:
        return 1
    print("benchmark index OK: every ID owns exactly one detail file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
