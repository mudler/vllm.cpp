#!/usr/bin/env python3
"""Keep the architecture-support checklist in `.agents/model-matrix.md` in sync
with the detailed per-architecture row lifecycle states, so it can never drift.

Per AGENTS.md the model matrix carries an at-a-glance **Architecture-support
checklist**: a rollup of how many architecture rows sit at each lifecycle state,
plus one line per *engaged* architecture (any row past `INVENTORIED`) with a
support mark. This checker fails if the checklist claims support the detailed
rows do not back, omits an engaged architecture, or its rollup counts drift from
the real per-state row counts. It mirrors check-readme-structure.py: the logic is
a pure `checklist_errors(text) -> list[str]` so it is unit- and mutation-testable
(see tests/scripts/test_check_model_checklist.py).

The invariant this enforces: a change that advances (or regresses) a model's
lifecycle state must update its checklist entry AND the rollup in the same commit,
exactly like check-doc-checkpoint.py forces the README/BENCHMARKS refresh.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / ".agents" / "model-matrix.md"

# The canonical lifecycle states (superset shared with check-agent-record.py).
STATES = (
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
)
STATE_RE = re.compile(r"`(" + "|".join(re.escape(s) for s in STATES) + r")`")
ROW_ID_RE = re.compile(r"MODEL-[A-Za-z0-9_.-]+")

# Support marks used on checklist entries and the lifecycle states each mark is
# allowed to stand for. `⬜` (inventoried) is deliberately NOT an entry mark: the
# ~300 inventoried rows are represented as a rollup COUNT, never as one line each.
#   ✅ correctness-complete + gated : an implemented, gate-on-record row.
#   🚧 active / in progress         : implemented or scoped, gate or speed pending.
#   📋 spiked                       : scoped in a spec, not implemented.
#   🚫 blocked                      : HW/dependency-blocked, no runnable gate.
MARK_ALLOWED_STATES: dict[str, frozenset[str]] = {
    "✅": frozenset({"DONE", "ACTIVE", "GATING", "PARTIAL"}),
    "🚧": frozenset({"ACTIVE", "GATING", "READY", "PARTIAL"}),
    "📋": frozenset({"SPIKE"}),
    "🚫": frozenset({"BLOCKED"}),
}
MARKS = tuple(MARK_ALLOWED_STATES)
INVENTORIED_MARK = "⬜"

CHECKLIST_HEADING = "## Architecture-support checklist"


def _split_cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", body)]


def _normalize(value: str) -> str:
    value = value.replace("`", "").replace("*", "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def parse_matrix_states(text: str) -> dict[str, str]:
    """Map every detailed MODEL-* row's stable ID to its single canonical state.

    Header-aware, exactly like check-agent-record.py: a table is a claim table
    only when its header has an `id` column AND a `state` column, so the separate
    checklist tables above (which have neither) are never mistaken for rows.
    """
    states: dict[str, str] = {}
    header: tuple[str, ...] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = _split_cells(line)
        normalized = tuple(_normalize(cell) for cell in cells)
        if "id" in normalized and any(
            value == "state" or value.endswith(" state") for value in normalized
        ):
            header = normalized
            continue
        if header is None or _is_separator(cells):
            continue
        item_id = cells[0].strip().strip("`")
        if not ROW_ID_RE.fullmatch(item_id):
            continue
        state_index = next(
            (
                i
                for i, value in enumerate(header)
                if value == "state" or value.endswith(" state")
            ),
            None,
        )
        state_cell = cells[state_index] if state_index is not None else ""
        matches = STATE_RE.findall(state_cell)
        if len(matches) == 1:
            states[item_id] = matches[0]
    return states


def _checklist_section(text: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if line.strip() == CHECKLIST_HEADING:
                inside = True
                continue
            if inside:
                break
        if inside:
            out.append(line)
    return out


def parse_rollup(section: list[str]) -> tuple[dict[str, int], int | None]:
    """Parse the rollup state-count table: rows `| STATE | N |` plus `| Total | N |`."""
    counts: dict[str, int] = {}
    total: int | None = None
    canonical = {s.lower(): s for s in STATES}
    for line in section:
        if not line.strip().startswith("|"):
            continue
        cells = _split_cells(line)
        if len(cells) < 2 or _is_separator(cells):
            continue
        key = cells[0].replace("*", "").strip().strip("`").lower()
        value = cells[1].replace("*", "").strip().strip("`")
        if not re.fullmatch(r"\d+", value):
            continue
        if key == "total":
            total = int(value)
        elif key in canonical:
            counts[canonical[key]] = int(value)
    return counts, total


def parse_entries(section: list[str]) -> list[tuple[str, str | None, str]]:
    """Return (mark, row_id_or_None, raw_line) for every checklist entry line.

    An entry is a table row whose first cell is a single support mark. The row's
    stable ID is the MODEL-* token found in its cells (conventionally the last).
    """
    entries: list[tuple[str, str | None, str]] = []
    for line in section:
        if not line.strip().startswith("|"):
            continue
        cells = _split_cells(line)
        if not cells:
            continue
        mark = cells[0].strip()
        if mark not in MARKS and mark != INVENTORIED_MARK:
            continue
        ids = ROW_ID_RE.findall(" ".join(cells[1:]))
        row_id = ids[-1] if ids else None
        entries.append((mark, row_id, line.strip()))
    return entries


def checklist_errors(text: str) -> list[str]:
    """Return a list of human-readable drift/consistency problems."""
    errors: list[str] = []
    states = parse_matrix_states(text)
    if not states:
        return ["no detailed MODEL-* rows parsed; matrix is malformed"]

    actual_counts: dict[str, int] = {}
    for state in states.values():
        actual_counts[state] = actual_counts.get(state, 0) + 1

    section = _checklist_section(text)
    if not section:
        return [f"missing required section: '{CHECKLIST_HEADING}'"]

    # --- Rollup counts must match the real per-state row counts, exactly. ---
    rollup, total = parse_rollup(section)
    for state, actual in sorted(actual_counts.items()):
        if state not in rollup:
            errors.append(
                f"rollup omits state {state} (matrix has {actual} such rows)"
            )
        elif rollup[state] != actual:
            errors.append(
                f"rollup count {state}={rollup[state]} but matrix has {actual}"
            )
    for state, claimed in sorted(rollup.items()):
        if state not in actual_counts and claimed != 0:
            errors.append(
                f"rollup claims {state}={claimed} but matrix has 0 such rows"
            )
    if total is None:
        errors.append("rollup has no Total row")
    elif total != len(states):
        errors.append(
            f"rollup Total={total} but matrix has {len(states)} architecture rows"
        )

    # --- Every checklist entry's mark must be backed by its row's state. ---
    entries = parse_entries(section)
    covered: dict[str, int] = {}
    for mark, row_id, raw in entries:
        where = raw if len(raw) <= 80 else raw[:77] + "..."
        if mark == INVENTORIED_MARK:
            errors.append(
                f"inventoried entries are a rollup COUNT, not a checklist line: {where}"
            )
            continue
        if row_id is None:
            errors.append(f"checklist entry has no MODEL-* row ID: {where}")
            continue
        covered[row_id] = covered.get(row_id, 0) + 1
        state = states.get(row_id)
        if state is None:
            errors.append(f"checklist entry references unknown row {row_id}")
            continue
        if state not in MARK_ALLOWED_STATES[mark]:
            errors.append(
                f"{mark} on {row_id} claims support its state `{state}` does not "
                f"back (allowed states for {mark}: "
                f"{', '.join(sorted(MARK_ALLOWED_STATES[mark]))})"
            )

    for row_id, times in sorted(covered.items()):
        if times > 1:
            errors.append(f"row {row_id} listed {times} times in the checklist")

    # --- Every engaged (non-INVENTORIED) architecture must be listed. ---
    for row_id, state in sorted(states.items()):
        if state == "INVENTORIED":
            if row_id in covered:
                errors.append(
                    f"INVENTORIED row {row_id} must not be an individual checklist "
                    "entry (it is part of the rollup count)"
                )
            continue
        if row_id not in covered:
            errors.append(
                f"engaged row {row_id} (state `{state}`) is missing from the checklist"
            )

    return errors


def main() -> int:
    if not MATRIX.exists():
        print(f"ERROR: {MATRIX.relative_to(ROOT)} is missing", file=sys.stderr)
        return 1
    errors = checklist_errors(MATRIX.read_text(encoding="utf-8"))
    if errors:
        print(
            "ERROR: the architecture-support checklist has drifted from the rows:",
            file=sys.stderr,
        )
        for err in dict.fromkeys(errors):
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("OK: architecture-support checklist matches the detailed row states.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
