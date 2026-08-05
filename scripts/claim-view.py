#!/usr/bin/env python3
"""Derive the claim view from open PRs instead of maintaining it by hand. (W2)

`coordination.md`'s hand-written claim table rotted for a structural reason:
claiming was free and never expired, while releasing cost anchor work. On
2026-08-04 it carried 106 rows asserting "implementation in flight" with nobody
flying them.

A PR is already a self-evidently live-or-dead claim: open = reserved, merged or
closed = released, no upkeep. So the claim view is GENERATED from PR state into a
delimited block, and the block is a REPORT that is never hand-edited.

Two modes, deliberately split so CI never needs the network:

    scripts/claim-view.py --apply    # query GitHub, rewrite the block
    scripts/claim-view.py --check    # offline: the block is well-formed,
                                     # its row IDs exist, and it is not stale

The legacy hand-maintained table above the block stays until the protocol cuts
over, because `check-agent-record.py` still requires every SPIKE/ACTIVE row's
owner to appear there. Removing it before those rows have PRs would strand them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COORD = ROOT / ".agents/coordination.md"

BEGIN = "<!-- claim-view:begin -->"
END = "<!-- claim-view:end -->"
GENERATED = re.compile(r"<!--\s*claim-view:generated\s+(\S+)\s*-->")

# A claim with no PR behind it expires: this is the TTL that stops the rot.
STALE_AFTER_DAYS = 14

ROW_BRANCH = re.compile(r"^row/([A-Za-z0-9_.-]+)$")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def known_row_ids() -> set[str]:
    """Every stable row ID the matrices define."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "agent_record", ROOT / "scripts/check-agent-record.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_record"] = module
    spec.loader.exec_module(module)
    rows: set[str] = set()
    for path in module.MATRIX_PATHS:
        rows |= {r.item_id for r in module.parse_claim_rows(path, [])}
    return rows


def fetch_open_prs() -> list[dict]:
    out = subprocess.check_output(
        ["gh", "pr", "list", "--state", "open", "--limit", "200",
         "--json", "number,headRefName,isDraft,title,author,updatedAt"],
        cwd=ROOT, text=True,
    )
    return json.loads(out)


def render(prs: list[dict], stamp: str) -> str:
    rows = []
    for pr in sorted(prs, key=lambda p: p["number"]):
        match = ROW_BRANCH.match(pr.get("headRefName", ""))
        if not match:
            continue
        state = "draft" if pr.get("isDraft") else "ready"
        author = (pr.get("author") or {}).get("login", "?")
        rows.append(
            f"| `{match.group(1)}` | #{pr['number']} | {state} | {author} | "
            f"{pr.get('updatedAt', '')[:10]} |"
        )
    body = "\n".join(rows) if rows else "| _none_ | | | | |"
    return "\n".join([
        BEGIN,
        f"<!-- claim-view:generated {stamp} -->",
        "",
        "GENERATED from open pull requests by `scripts/claim-view.py --apply`.",
        "Do not hand-edit: an open `row/<ROW-ID>` PR IS the reservation, and it",
        "is released by merging or closing it.",
        "",
        "| Row | PR | State | Agent | Updated |",
        "|---|---|---|---|---|",
        body,
        "",
        END,
    ])


def block_bounds(text: str) -> tuple[int, int] | None:
    start, end = text.find(BEGIN), text.find(END)
    if start == -1 or end == -1 or end < start:
        return None
    return start, end + len(END)


def check_errors(text: str, rows: set[str]) -> list[str]:
    """Offline validation of the generated block."""
    bounds = block_bounds(text)
    if bounds is None:
        return [
            f".agents/coordination.md is missing the claim view block "
            f"({BEGIN} ... {END}); regenerate with scripts/claim-view.py --apply"
        ]
    block = text[bounds[0]:bounds[1]]
    errors: list[str] = []

    stamp = GENERATED.search(block)
    if not stamp:
        errors.append("the claim view has no <!-- claim-view:generated ... --> stamp")
    else:
        try:
            age = time.time() - time.mktime(time.strptime(stamp.group(1), "%Y-%m-%d"))
            if age > STALE_AFTER_DAYS * 86400:
                errors.append(
                    f"the claim view was generated {int(age // 86400)} days ago, over "
                    f"the {STALE_AFTER_DAYS}-day TTL; a claim with no live PR behind "
                    "it must expire rather than rot. Re-run --apply"
                )
        except ValueError:
            errors.append(f"unparseable claim-view stamp {stamp.group(1)!r}")

    for line in block.splitlines():
        if not line.startswith("| `"):
            continue
        row = line.split("`")[1]
        if row not in rows:
            errors.append(f"the claim view references unknown row {row}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="query GitHub and rewrite")
    mode.add_argument("--check", action="store_true", help="offline validation (default)")
    args = parser.parse_args()

    text = COORD.read_text(encoding="utf-8")

    if args.apply:
        try:
            prs = fetch_open_prs()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"ERROR: could not query GitHub ({exc}); --apply needs `gh`",
                  file=sys.stderr)
            return 1
        stamp = time.strftime("%Y-%m-%d")
        rendered = render(prs, stamp)
        bounds = block_bounds(text)
        if bounds:
            text = text[:bounds[0]] + rendered + text[bounds[1]:]
        else:
            anchor = "## Handoff queue"
            if anchor not in text:
                print("ERROR: no place to insert the claim view", file=sys.stderr)
                return 1
            text = text.replace(anchor, rendered + "\n\n" + anchor, 1)
        COORD.write_text(text, encoding="utf-8")
        claimed = sum(1 for line in rendered.splitlines() if line.startswith("| `"))
        print(f"claim view regenerated: {claimed} row(s) reserved by open PRs")
        return 0

    failures = check_errors(text, known_row_ids())
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("OK: the claim view is present, well-formed and inside its TTL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
