#!/usr/bin/env python3
"""#785 P1 kernel-trace classifier.

Family vs exact specialization (Researcher 2609):
  A: exact WMMA <2,8,16,32,false>; no SharedK family; no other WMMA.
  B: exact scalar <2,8,32,32>; no WMMA family; no other scalar.
  Wrong BM/BN, wrong qg/d, mixed, or none => UNKNOWN.

WMMA names contain the SharedK prefix; scalar family is scored only on
lines that do not contain SharedKWmma.

Exit 0 prints `arm=A|B|UNKNOWN` plus marker hits.
Exit 2 = unreadable/empty input (fail closed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXACT_WMMA = (
    "PagedAttnPrefillSharedKWmma<2, 8, 16, 32, false>",
    "PagedAttnPrefillSharedKWmma<2,8,16,32,false>",
    "PagedAttnPrefillSharedKWmmaILi2ELi8ELi16ELi32ELb0E",
)
EXACT_SCALAR = (
    "PagedAttnPrefillSharedK<2, 8, 32, 32>",
    "PagedAttnPrefillSharedK<2,8,32,32>",
    "PagedAttnPrefillSharedKILi2ELi8ELi32ELi32EE",
)


def collect_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(errors="replace")
    chunks: list[str] = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".json", ".txt", ".log", ".out", ""}:
            if p.stat().st_size > 8_000_000:
                continue
        try:
            chunks.append(p.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _scalar_blob(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "SharedKWmma" not in line)


def _strip_exact(text: str, markers: tuple[str, ...]) -> str:
    out = text
    for m in markers:
        out = out.replace(m, "")
    return out


def classify(text: str) -> tuple[str, list[str], list[str]]:
    scalar_blob = _scalar_blob(text)
    wmma_hits = [m for m in EXACT_WMMA if m in text]
    scalar_hits = [m for m in EXACT_SCALAR if m in scalar_blob]
    wmma_family = "PagedAttnPrefillSharedKWmma" in text
    scalar_family = "PagedAttnPrefillSharedK" in scalar_blob
    other_wmma = "PagedAttnPrefillSharedKWmma" in _strip_exact(text, EXACT_WMMA)
    other_scalar = "PagedAttnPrefillSharedK" in _strip_exact(scalar_blob, EXACT_SCALAR)
    if wmma_hits and not scalar_family and not other_wmma:
        arm = "A"
    elif scalar_hits and not wmma_family and not other_scalar:
        arm = "B"
    else:
        arm = "UNKNOWN"
    return arm, wmma_hits, scalar_hits


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="rocprofv3 output file or directory")
    ap.add_argument("--expect", choices=("A", "B"), help="fail closed if arm mismatches")
    args = ap.parse_args(argv)
    src = Path(args.path)
    if not src.exists():
        print("ERROR: missing trace path", src, file=sys.stderr)
        return 2
    text = collect_text(src)
    if not text.strip():
        print("ERROR: empty trace", src, file=sys.stderr)
        return 2
    arm, wmma_hits, scalar_hits = classify(text)
    print(f"arm={arm}")
    print("wmma_hits=" + (",".join(wmma_hits) if wmma_hits else "-"))
    print("scalar_hits=" + (",".join(scalar_hits) if scalar_hits else "-"))
    if args.expect and arm != args.expect:
        print(f"ERROR: expected arm={args.expect} got {arm}", file=sys.stderr)
        return 1
    if args.expect == "A" and not wmma_hits:
        print("ERROR: A missing exact WMMA specialization", file=sys.stderr)
        return 1
    if args.expect == "B" and (wmma_hits or not scalar_hits):
        print("ERROR: B must be exact scalar-only", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
