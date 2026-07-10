#!/usr/bin/env python3
"""Validate the canonical .agents tabular record without third-party packages."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"

MATRICES = {
    # The model summary points into the exhaustive, per-target spike table.
    "MODEL": (AGENTS / "specs/model-family-inventory.md", 323),
    "QUANT": (AGENTS / "quantization-matrix.md", 76),
    "KERNEL": (AGENTS / "kernel-matrix.md", 30),
    "BACKEND": (AGENTS / "backend-matrix.md", 48),
}

ENGINE_MATRIX = AGENTS / "engine-matrix.md"
ENGINE_PREFIXES = ("ENG", "KV", "PAR", "SAMPLE", "TOOLS", "SPEC", "SERVE", "LORA", "ATTN", "LOAD")
ENGINE_ROWS = 81

REQUIRED = [
    ROOT / "AGENTS.md",
    AGENTS / "roadmap_v1.md",
    AGENTS / "coordination.md",
    AGENTS / "feature-matrix.md",
    ENGINE_MATRIX,
    AGENTS / "specs/feature-anchor-backfill.md",
    *(path for path, _ in MATRICES.values()),
]

STATES = {
    "INVENTORIED",
    "SPIKE",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "DONE",
    "BLOCKED",
    "ANCHOR-BACKFILL",
    "BUILD-ONLY",
    "UNTRACED",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return [ROOT / "AGENTS.md", ROOT / "README.md", *sorted(AGENTS.rglob("*.md"))]


def check_links(errors: list[str]) -> None:
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (source.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{source.relative_to(ROOT)}: dangling link {raw_target}")


def table_ids(prefix: str, path: Path, errors: list[str]) -> list[str]:
    row_id = re.compile(rf"^\|\s*`({prefix}-[A-Za-z0-9_.-]+)`\s*\|")
    ids: list[str] = []
    state_table = False
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|"):
            state_table = False
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if any(cell.lower().endswith("state") for cell in cells):
            state_table = True
        match = row_id.match(line)
        if not match or not state_table:
            continue
        item_id = match.group(1)
        if item_id in ids:
            errors.append(f"{path.relative_to(ROOT)}:{line_no}: duplicate ID {item_id}")
        ids.append(item_id)
        if not any(f"`{state}`" in line for state in STATES):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_no}: {item_id} has no canonical state"
            )
    return ids


def check_matrices(errors: list[str]) -> None:
    for prefix, (path, minimum) in MATRICES.items():
        ids = table_ids(prefix, path, errors)
        if len(ids) != minimum:
            errors.append(
                f"{path.relative_to(ROOT)}: {len(ids)} {prefix} rows; expected {minimum}"
            )

    engine_ids = [
        item_id
        for prefix in ENGINE_PREFIXES
        for item_id in table_ids(prefix, ENGINE_MATRIX, errors)
    ]
    if len(engine_ids) != ENGINE_ROWS:
        errors.append(
            f"{ENGINE_MATRIX.relative_to(ROOT)}: {len(engine_ids)} engine rows; "
            f"expected {ENGINE_ROWS}"
        )


def check_model_invariants(errors: list[str]) -> None:
    path = AGENTS / "specs/model-family-inventory.md"
    rows: list[tuple[list[str], str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `MODEL-"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        aliases = re.findall(r"`([^`]+)`", cells[1])
        targets = [value for value in re.findall(r"`([^`]+)`", cells[2]) if "::" in value]
        if aliases and targets:
            rows.append((aliases, targets[-1]))

    actual = {
        "rows": len(rows),
        "memberships": sum(len(aliases) for aliases, _ in rows),
        "architectures": len({alias for aliases, _ in rows for alias in aliases}),
        "targets": len({target for _, target in rows}),
        "modules": len({target.split("::", 1)[0] for _, target in rows}),
    }
    expected = {
        "rows": 321,
        "memberships": 370,
        "architectures": 353,
        "targets": 307,
        "modules": 258,
    }
    if actual != expected:
        errors.append(f"{path.relative_to(ROOT)}: model inventory {actual}, expected {expected}")


def check_table_shapes(errors: list[str]) -> None:
    paths = [
        AGENTS / "roadmap_v1.md",
        AGENTS / "coordination.md",
        ENGINE_MATRIX,
        *(path for path, _ in MATRICES.values()),
        AGENTS / "model-matrix.md",
        AGENTS / "specs/quantization-coverage.md",
        AGENTS / "specs/kernel-family-inventory.md",
        AGENTS / "specs/cuda-architecture-inventory.md",
        AGENTS / "specs/feature-anchor-backfill.md",
    ]
    for path in paths:
        expected_pipes: int | None = None
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.startswith("|"):
                expected_pipes = None
                continue
            pipes = line.count("|")
            if expected_pipes is None:
                expected_pipes = pipes
            elif pipes != expected_pipes:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no}: table has {pipes} pipes; "
                    f"expected {expected_pipes}"
                )


def check_spec_location(errors: list[str]) -> None:
    misplaced = re.compile(r"(?:spec|scoping|semantics|feasibility|notes)", re.I)
    allowed = {"benchmark-protocol.md"}
    for path in AGENTS.glob("*.md"):
        if path.name not in allowed and misplaced.search(path.stem):
            errors.append(f"{path.relative_to(ROOT)}: feature spec/scoping file belongs in .agents/specs/")


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.is_file():
            errors.append(f"missing canonical record: {path.relative_to(ROOT)}")

    if not errors:
        check_links(errors)
        check_matrices(errors)
        check_model_invariants(errors)
        check_table_shapes(errors)
        check_spec_location(errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts = [
        "ENGINE="
        + str(
            sum(
                len(table_ids(prefix, ENGINE_MATRIX, []))
                for prefix in ENGINE_PREFIXES
            )
        )
    ]
    for prefix, (path, _) in MATRICES.items():
        counts.append(f"{prefix}={len(table_ids(prefix, path, []))}")
    print("agent record OK: " + " ".join(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
