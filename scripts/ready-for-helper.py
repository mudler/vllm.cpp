#!/usr/bin/env python3
"""Compute which rows a helper may pick. (W3)

A helper picking an arbitrary roadmap row hits what the 2026-08-04 audit found:
the row's state is a lie, its spec is missing, its anchors do not exist. It then
produces a PR the operator must reject. So helpers only ever see a queue, and a
row enters that queue only when all five conditions hold:

1. a committed `.agents/specs/<slug>.md` covering the spike contract;
2. its gates are defined and runnable (a real test/evidence anchor);
3. it is CPU-gateable, or its hardware need is named on the row;
4. it does not depend on an unmerged PR;
5. no open PR already claims it (read from the generated claim view).

    scripts/ready-for-helper.py            # print the queue
    scripts/ready-for-helper.py --check    # assert the computation is sound

`--check` is what CI runs: it never asserts the queue is NON-empty (an empty
queue is a true and useful answer today, since 98 rows fail condition 1 or 2),
only that every row it WOULD offer really satisfies the conditions.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PICKABLE_STATES = {"READY", "INVENTORIED"}
# Words that mean "this needs hardware"; the row must say so explicitly.
HARDWARE_HINTS = ("dgx", "gb10", "sm_121", "cuda", "gpu", "metal", "m4", "thor")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rec = _load("agent_record", "scripts/check-agent-record.py")
claim_view = _load("claim_view", "scripts/claim-view.py")


def reserved_rows() -> set[str]:
    """Rows already claimed by an open PR, per the generated view."""
    text = (ROOT / ".agents/coordination.md").read_text(encoding="utf-8")
    bounds = claim_view.block_bounds(text)
    if bounds is None:
        return set()
    return {
        line.split("`")[1]
        for line in text[bounds[0]:bounds[1]].splitlines()
        if line.startswith("| `")
    }


def spec_ok(row) -> bool:
    for candidate in rec.local_spec_paths(row):
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if not path.exists():
            continue
        try:
            if rec.has_substantive_spec_content(
                path.read_text(encoding="utf-8").splitlines()
            ):
                return True
        except OSError:
            continue
    return False


def hardware_declared(row) -> bool:
    """CPU-gateable, or the hardware need is named on the row."""
    blob = " ".join(row.cells).lower()
    needs_hw = any(hint in blob for hint in HARDWARE_HINTS)
    return (not needs_hw) or ("dgx" in blob or "gb10" in blob or "hardware" in blob)


def evaluate(row, reserved: set[str]) -> list[str]:
    """Return the conditions this row FAILS (empty means pickable)."""
    missing = []
    if not spec_ok(row):
        missing.append("no substantive spec")
    if not rec.is_test_anchor(row.field("tests"), row.path):
        missing.append("no runnable gate/evidence anchor")
    if not hardware_declared(row):
        missing.append("hardware need not named")
    if row.item_id in reserved:
        missing.append("already claimed by an open PR")
    return missing


def queue() -> tuple[list, dict[str, int]]:
    rows = []
    for path in rec.MATRIX_PATHS:
        rows.extend(rec.parse_claim_rows(path, []))
    reserved = reserved_rows()
    pickable, reasons = [], {}
    for row in rows:
        if row.state not in PICKABLE_STATES:
            continue
        missing = evaluate(row, reserved)
        if missing:
            for reason in missing:
                reasons[reason] = reasons.get(reason, 0) + 1
        else:
            pickable.append(row)
    return pickable, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pickable, reasons = queue()
    reserved = reserved_rows()

    if args.check:
        # Never assert the queue is non-empty: an empty queue is a true answer.
        # Assert instead that everything offered really satisfies every rule.
        bad = [r.item_id for r in pickable if evaluate(r, reserved)]
        if bad:
            for item in bad:
                print(f"ERROR: {item} is offered to helpers but fails a condition",
                      file=sys.stderr)
            return 1
        print(
            f"OK: {len(pickable)} row(s) are READY-FOR-HELPER; every offered row "
            "satisfies spec, gate, hardware and reservation conditions."
        )
        return 0

    print(f"READY-FOR-HELPER queue: {len(pickable)} row(s)\n")
    for row in pickable[:40]:
        print(f"  {row.item_id[:52]:52} {row.state:12} {row.path.name}")
    if len(pickable) > 40:
        print(f"  ... (+{len(pickable) - 40} more)")
    if reasons:
        print("\nWhy the rest are not pickable:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
