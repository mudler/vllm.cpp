#!/usr/bin/env python3
"""Derive the issue index from GitHub instead of storing it in a shared file.

`.agents/issue-index.md` used to be a tracked, append-only table that every pull
request had to write. That made one file the repository's dominant lock. Measured
at origin/main e541be98 on 2026-08-29: 115 of the last 200 commits wrote it, it
held 854 rows, and 16 of 21 open pull requests reported CONFLICTING. For #2267,
#2248, #1726 and #1703 it was the ONLY conflicting path.

The `merge=union` attribute did not save it and could not. GitHub does not run
`.gitattributes` merge drivers, so the driver resolved the collision on a
developer's machine and the forge conflicted anyway (#883). A pull request born
conflicted is never scheduled at all, so it carries zero check-runs and reads as
unverified rather than red -- #2248 is the measured case.

Nothing here is authored. GitHub already holds every column the table had:

  * `gh issue list`      -> the number, the title, the open/closed state
  * the issue body's `Row:` line -> the owning roadmap or area-matrix row
  * a spec's `## Owed`   -> the dash case, resolved offline by check-agent-record

So there is no file to conflict on, and no pull request writes this one: the
snapshot is untracked, and `.gitignore` carries it.

OFFLINE-FIRST, deliberately, on the contract `scripts/now.py:127` has served
since #374. The `gh` call is the only part that needs a network and it degrades
to REMOTE_UNVERIFIED rather than failing. A FAILED REFRESH WRITES NOTHING: a
partial snapshot is worse than an absent one, because a consumer cannot tell a
truncated table from a complete one, and "unknown" must never render as "no
issues". Consumers SKIP on absence rather than passing.
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
AGENTS = ROOT / ".agents"
SNAPSHOT = AGENTS / "issue-index.generated.md"

# The owning row, written in the issue body. A dash, or no line at all, means the
# issue claims no row and must instead appear under some spec's `## Owed` --
# which `check-agent-record.owed_issues()` resolves from a glob, network-free.
ROW_LINE = re.compile(r"^\s*Row:\s*`?([A-Z0-9][A-Za-z0-9_.-]*|-)`?\s*$", re.MULTILINE)

# Kept deliberately in step with `check-agent-record.ISSUE_ROW`, which parses the
# rows this file writes. The snapshot exists to be read by that regex, so the two
# move together or the gate goes blind.
REPO_URL = "https://github.com/mudler/vllm.cpp"

# A snapshot older than this reports its age and SKIPs rather than gating. It is
# not a correctness bound -- GitHub is the record -- only the point past which a
# reader should be told the copy is old.
STALE_SECONDS = 24 * 60 * 60

PREAMBLE = f"""# Issue index (GENERATED -- do not edit, do not commit)

Written by `scripts/agent-issue-index.py --refresh` from `gh issue list`. GitHub
is the index; this file is a cache so the record gates can run offline. It is
untracked, so no pull request writes it and it cannot conflict.

`Row` is the owning roadmap block or area-matrix row, taken from a `Row:` line in
the issue body, or a dash when the issue is instead listed under a spec's
`## Owed`.

Refresh it with:

    python3 scripts/agent-issue-index.py --refresh

| Issue | Row | Title | Kind |
|---:|---|---|---|"""


def _kind(labels: list[dict]) -> str:
    """The issue's kind, from its labels, defaulting to `task`.

    The retired table carried this as a free-text column an author chose. Labels
    are the same information with a closed vocabulary and one writer.
    """

    names = {label.get("name", "").lower() for label in labels}
    for kind in ("bug", "enhancement", "documentation"):
        if kind in names:
            return "feature" if kind == "enhancement" else kind
    return "task"


def _row(body: str | None) -> str | None:
    """The owning row ID from the body's `Row:` line, or None for the dash case."""

    match = ROW_LINE.search(body or "")
    if match is None or match.group(1) == "-":
        return None
    return match.group(1)


def fetch(limit: int = 1000) -> tuple[list[dict] | None, str | None]:
    """Open issues from GitHub. Returns (issues, degraded_reason).

    The ONLY network call. An unreachable remote returns REMOTE_UNVERIFIED and
    None -- never an exception, and never an empty list, because an empty list is
    indistinguishable from a repository with no open issues.
    """

    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--limit", str(limit),
             "--json", "number,title,body,labels"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"REMOTE_UNVERIFIED: {exc.__class__.__name__}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return None, f"REMOTE_UNVERIFIED: {detail[-1] if detail else 'gh failed'}"
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return None, "REMOTE_UNVERIFIED: unparseable gh output"
    if not isinstance(payload, list):
        return None, "REMOTE_UNVERIFIED: gh returned no issue array"
    return payload, None


def render(issues: list[dict]) -> str:
    """The snapshot text. Rows sorted by issue number, so a refresh is stable."""

    lines = [PREAMBLE]
    for issue in sorted(issues, key=lambda item: item["number"]):
        number = issue["number"]
        row = _row(issue.get("body"))
        # A newline or a pipe in a title would break the table the consumers
        # parse, so both are neutralised here rather than in four readers.
        title = " ".join((issue.get("title") or "").split()).replace("|", "\\|")
        cell = f"`{row}`" if row else "—"
        lines.append(
            f"| [#{number}]({REPO_URL}/issues/{number}) | {cell} | {title} "
            f"| {_kind(issue.get('labels') or [])} |"
        )
    return "\n".join(lines) + "\n"


def refresh(limit: int = 1000) -> tuple[int, str]:
    """Write the snapshot. Returns (exit_code, message). Writes nothing on failure."""

    issues, degraded = fetch(limit)
    if degraded is not None:
        return 3, (
            f"{degraded}\n"
            "  The snapshot was NOT written, and any existing one is unchanged.\n"
            "  Unknown is neither absence nor success: a partial index would read\n"
            "  as a complete one. Record gates will SKIP until this succeeds."
        )
    assert issues is not None
    if not issues:
        return 3, (
            "REMOTE_UNVERIFIED: gh returned zero open issues\n"
            "  Refusing to write an empty index. A repository with no open issue\n"
            "  and a query that silently matched nothing look identical here."
        )
    SNAPSHOT.write_text(render(issues), encoding="utf-8")
    unowned = sum(1 for issue in issues if _row(issue.get("body")) is None)
    # Reported, not gated. Ownership is diff-scoped: a change owes a `Row:` only
    # for the issues it REFERENCES, so a large historical tail carrying no line
    # is the expected migration state and not a backlog anybody must pay down.
    return 0, (
        f"wrote {SNAPSHOT.relative_to(ROOT)}: {len(issues)} open issues, "
        f"{unowned} carrying no `Row:` line yet (backfilled per issue as changes "
        "reference them; not a gate)"
    )


# --- one-time migration -------------------------------------------------------
#
# The retired `.agents/issue-index.md` carried the row association in a column.
# This writes that association into the issue body, where the refresher reads it,
# so the retired file can be frozen without losing what it knew. Committed rather
# than run as a throwaway because it makes ~369 outward writes to the tracker: it
# has to be reviewable, idempotent and resumable, and a throwaway is none of
# those. It is a migration, not a gate; nothing calls it after the cutover.

RETIRED_INDEX = AGENTS / "issue-index.md"


def legacy_rows(path: Path | None = None) -> dict[str, str]:
    """{issue number -> row ID} from the retired index. Dash rows are omitted."""

    source = path if path is not None else RETIRED_INDEX
    if not source.is_file():
        return {}
    pattern = re.compile(
        r"^\|\s*\[#(\d+)\]\(https://github\.com/[^)]+/issues/(\d+)\)\s*\|"
        r"\s*`([A-Z0-9][A-Za-z0-9_.-]*)`\s*\|"
    )
    found: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        # A row whose link points at a DIFFERENT issue than its number is
        # corrupt; the retired file had no gate against that, so skip rather
        # than write a wrong association into the tracker.
        if match and match.group(1) == match.group(2):
            found[match.group(1)] = match.group(3)
    return found


def backfill(*, apply: bool, limit: int = 1000) -> tuple[int, str]:
    """Prepend the retired index's `Row:` line to each open issue that lacks one.

    Idempotent and resumable: an issue whose body already carries a `Row:` line is
    skipped, so a re-run after an interruption costs nothing and cannot double a
    line. Never writes an empty body.
    """

    mapping = legacy_rows()
    if not mapping:
        return 3, f"{RETIRED_INDEX.relative_to(ROOT)}: no legacy rows to migrate"
    issues, degraded = fetch(limit)
    if degraded is not None:
        return 3, degraded
    assert issues is not None

    planned: list[tuple[int, str, str]] = []
    already = 0
    unmapped = 0
    for issue in sorted(issues, key=lambda item: item["number"]):
        number = issue["number"]
        body = issue.get("body") or ""
        row = mapping.get(str(number))
        if row is None:
            unmapped += 1
            continue
        if ROW_LINE.search(body):
            already += 1
            continue
        planned.append((number, row, body))

    head = (
        f"{len(planned)} to write, {already} already carry a Row line, "
        f"{unmapped} open issues the retired index does not cover"
    )
    if not apply:
        sample = "\n".join(
            f"  #{number} -> Row: `{row}`  ({len(body)} byte body preserved)"
            for number, row, body in planned[:8]
        )
        return 0, f"DRY RUN: {head}\n{sample}\n  ... (--backfill --apply to write)"

    written = 0
    failed: list[str] = []
    for number, row, body in planned:
        # Prepend, never replace: the existing body is carried through verbatim.
        new_body = f"Row: `{row}`\n\n{body}" if body.strip() else f"Row: `{row}`\n"
        result = subprocess.run(
            ["gh", "issue", "edit", str(number), "--body", new_body],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            failed.append(f"#{number}: {result.stderr.strip().splitlines()[-1:]}")
            continue
        written += 1
        time.sleep(0.5)  # the tracker rate-limits; a resumable run can afford it
    tail = f"\n  {len(failed)} FAILED (re-run to resume): {failed[:5]}" if failed else ""
    return (1 if failed else 0), f"wrote {written} of {len(planned)}{tail}"


def snapshot_state() -> tuple[str | None, str | None]:
    """(text, skip_reason). Absence and age are reported, never silently passed."""

    if not SNAPSHOT.is_file():
        return None, (
            f"{SNAPSHOT.relative_to(ROOT)} is absent; run "
            "`python3 scripts/agent-issue-index.py --refresh`"
        )
    age = time.time() - SNAPSHOT.stat().st_mtime
    text = SNAPSHOT.read_text(encoding="utf-8")
    if age > STALE_SECONDS:
        return text, (
            f"{SNAPSHOT.relative_to(ROOT)} is {age / 3600:.0f}h old; run "
            "`python3 scripts/agent-issue-index.py --refresh`"
        )
    return text, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true",
        help="query GitHub and rewrite the snapshot",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report whether a usable snapshot exists, without a network call",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="one-time migration: write the retired index's Row into issue bodies",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="with --backfill, actually write; without it the run is a dry run",
    )
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.backfill:
        code, message = backfill(apply=args.apply, limit=args.limit)
        print(message, file=sys.stderr if code else sys.stdout)
        return code

    if args.refresh:
        code, message = refresh(args.limit)
        print(message, file=sys.stderr if code else sys.stdout)
        return code

    if args.check:
        _, reason = snapshot_state()
        if reason is not None:
            print(f"SKIP: {reason}", file=sys.stderr)
            return 3
        print(f"OK: {SNAPSHOT.relative_to(ROOT)} is present and fresh")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
