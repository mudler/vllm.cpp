#!/usr/bin/env python3
"""Render the live position on demand instead of storing it in a shared file.

`.agents/NOW.md` used to carry a per-row live-claims table, and
every lifecycle move once had to edit it. That made one
file a surface EVERY row-advancing PR writes, which is a lock: it conflicted in 5
of the 16 conflicting open PRs measured at origin/main d928e2c3, and it violated
the `AGENTS.md` invariant "no surface that every PR must write".

Nothing here is authored. Every line is assembled from a record that already has
a per-row home and one writer:

  * the area matrices        -> which rows are SPIKE/ACTIVE, and their owner
  * .agents/claims/CLAIM-*.md -> who holds each claim (one file per claim, #364)
  * each row spec's `## Now`  -> what that row's next step is
  * open pull requests        -> live branch state

So there is no file to conflict on. The authored half -- the current gate, the
cross-row next actions -- stays in `.agents/NOW.md`, which only the operator
edits and which no per-row PR is required to touch.

OFFLINE-FIRST, deliberately. The pull-request half is the ONLY part that needs a
network, and it degrades to REMOTE_UNVERIFIED rather than failing, exactly as
`claim-view.py` does. A digest that cannot print without a network would be worse
than the file it replaces, so an unreachable remote is reported, never fatal, and
never silently rendered as "no claims".
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".agents"

MATRICES = (
    "roadmap_v1.md",
    "backend-matrix.md",
    "engine-matrix.md",
    "feature-matrix.md",
    "kernel-matrix.md",
    "model-matrix.md",
    "quantization-matrix.md",
)

LIVE_STATES = {"SPIKE", "ACTIVE"}

ROW_ID = re.compile(r"^\|\s*`([A-Z0-9][A-Za-z0-9_.-]*)`")
STATE_CELL = re.compile(r"`(INVENTORIED|SPIKE|READY|PARTIAL|ACTIVE|GATING|DONE|ANCHOR-BACKFILL)`")
SPEC_LINK = re.compile(r"\.agents/specs/([A-Za-z0-9_.-]+\.md)|\(specs/([A-Za-z0-9_.-]+\.md)\)")
CLAIM_ID = re.compile(r"CLAIM-[A-Za-z0-9_.-]+")
NOW_SECTION = re.compile(r"^##\s+Now\s*$", re.MULTILINE)


def split_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def live_rows() -> list[dict[str, str]]:
    """Every SPIKE/ACTIVE row across the matrices, with its spec and owner."""
    rows: list[dict[str, str]] = []
    for name in MATRICES:
        path = AGENTS / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            identifier = ROW_ID.match(line)
            if not identifier:
                continue
            states = STATE_CELL.findall(line)
            if not states or states[-1] not in LIVE_STATES:
                continue
            spec = SPEC_LINK.search(line)
            cells = split_cells(line)
            rows.append(
                {
                    "id": identifier.group(1),
                    "state": states[-1],
                    "matrix": name,
                    "spec": (spec.group(1) or spec.group(2)) if spec else "",
                    "owner": cells[-1] if cells else "",
                }
            )
    rows.sort(key=lambda r: r["id"])
    return rows


def spec_now(slug: str) -> str:
    """The row's own live position: the first line under its `## Now`."""
    if not slug:
        return ""
    path = AGENTS / "specs" / slug
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    section = NOW_SECTION.search(text)
    if section is None:
        return ""
    body = text[section.end():].split("\n##", 1)[0].strip()
    return " ".join(body.split())


def claims() -> dict[str, list[str]]:
    """Map row ID -> claim IDs, from the one-file-per-claim directory (#364)."""
    holders: dict[str, list[str]] = {}
    claims_dir = AGENTS / "claims"
    if not claims_dir.is_dir():
        return holders
    for path in sorted(claims_dir.glob("CLAIM-*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `CLAIM-"):
                continue
            cells = split_cells(line)
            match = CLAIM_ID.search(cells[0])
            if match is None or len(cells) < 2:
                continue
            for row in re.findall(r"`([A-Z0-9][A-Za-z0-9_.-]*)`", cells[1]):
                holders.setdefault(row, []).append(match.group(0))
    return holders


def open_prs() -> tuple[dict[str, list[str]], str | None]:
    """Map row ID -> open PR numbers. Returns (map, degraded_reason).

    The ONLY network call. An unreachable remote returns REMOTE_UNVERIFIED and an
    empty map, never an exception and never a silent "no PRs" -- unknown is
    neither absence nor success.
    """
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--limit", "200",
             "--json", "number,headRefName"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"REMOTE_UNVERIFIED: {exc.__class__.__name__}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return {}, f"REMOTE_UNVERIFIED: {detail[-1] if detail else 'gh failed'}"
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return {}, "REMOTE_UNVERIFIED: unparseable gh output"

    by_row: dict[str, list[str]] = {}
    for entry in payload:
        branch = entry.get("headRefName", "")
        if not branch.startswith("row/"):
            continue
        row = branch[len("row/"):]
        by_row.setdefault(row, []).append(f"#{entry['number']}")
    return by_row, None


def render(*, offline: bool = False) -> str:
    rows = live_rows()
    held = claims()
    prs, degraded = ({}, "REMOTE_UNVERIFIED: --offline") if offline else open_prs()

    lines = [
        "# NOW — derived, not stored",
        "",
        "Rendered by `scripts/now.py` from the matrices, `.agents/claims/`, each",
        "row spec's `## Now`, and open PRs. Nothing here is written by hand, so",
        "there is no shared file for concurrent PRs to conflict on.",
        "",
        f"## Live rows ({len(rows)})",
        "",
    ]
    if degraded:
        lines += [f"> {degraded} — pull-request column omitted, rows are complete.", ""]

    lines += ["| Row | State | Claim | PR | Next step |", "|---|---|---|---|---|"]
    for row in rows:
        claim = ", ".join(held.get(row["id"], [])) or "—"
        pr = ", ".join(prs.get(row["id"], [])) or "—"
        step = spec_now(row["spec"]) or "—"
        if len(step) > 110:
            step = step[:107] + "..."
        lines.append(
            f"| `{row['id']}` | {row['state']} | {claim} | {pr} | {step} |"
        )

    authored = AGENTS / "NOW.md"
    if authored.is_file():
        lines += ["", "## Authored context", "",
                  f"Operator-cadence priorities live in `{authored.relative_to(ROOT)}`.", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the pull-request lookup; everything else still renders",
    )
    args = parser.parse_args(argv)
    sys.stdout.write(render(offline=args.offline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
