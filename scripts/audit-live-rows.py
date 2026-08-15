#!/usr/bin/env python3
"""Audit the live-state matrix rows against Git reality. (P0)

54 rows claim ACTIVE at once, which cannot be true: a stale ACTIVE cell inside
a several-hundred-row table is invisible rot. This tool makes it visible.

It PROPOSES and REPORTS. It never rewrites a matrix -- corrections are applied
by a human/agent in reviewable per-matrix commits, because a state transition
carries contract obligations (a spec link, evidence anchors) that only a reader
of the row can satisfy.

Row parsing is imported from scripts/check-agent-record.py rather than
reimplemented, so the audit and the gate can never disagree about what a row is.

    scripts/audit-live-rows.py                 # markdown report
    scripts/audit-live-rows.py --json          # machine-readable
    scripts/audit-live-rows.py --check         # exit 1 if an ACTIVE row is abandoned
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


record = _load("agent_record", "scripts/check-agent-record.py")

LIVE_STATES = frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"})

# check-agent-record.py's MATRIX_PATHS omits feature-matrix.md and
# sglang-matrix.md. All 11 of the live rows they add come from
# feature-matrix.md; sglang-matrix.md contributes ZERO, because its lifecycle
# column is `Class` (FUSED / SGLANG-DISTINCT / OUT-OF-SCOPE), not `State`, so
# no table in it parses as a claim table today. Keeping it here is still
# right: it costs nothing now and the coverage becomes automatic the day it
# grows a `State` column. The audit covers all seven matrices so no live row
# escapes it, but deliberately does NOT widen MATRIX_PATHS itself: that
# governs a repo-wide CI gate whose row contract these two files have never
# been held to.
AUDIT_MATRIX_PATHS = [
    *record.MATRIX_PATHS,
    record.AGENTS / "feature-matrix.md",
    record.AGENTS / "sglang-matrix.md",
]


def live_rows(errors: list[str] | None = None) -> list[record.ClaimRow]:
    """Every row in the audited matrices whose state is in LIVE_STATES.

    Parse errors are appended to `errors` when a list is supplied. They must
    not be swallowed: parse_claim_rows DROPS a row it cannot parse, so a
    malformed row would vanish from a census whose whole point is
    completeness -- and it bites hardest on feature-matrix.md and
    sglang-matrix.md, which no CI gate parses today.
    """
    sink = errors if errors is not None else []
    rows = []
    for path in AUDIT_MATRIX_PATHS:
        for row in record.parse_claim_rows(path, sink):
            if row.state in LIVE_STATES:
                rows.append(row)
    return rows


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else ""


def row_branches() -> dict[str, list[str]]:
    """Map row ID -> every local or remote branch named row/<ID>."""
    mapping: dict[str, list[str]] = {}
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes")
    for line in out.splitlines():
        name = line.strip()
        if name.startswith("row/"):
            item_id = name[len("row/") :]
        elif "/row/" in name:
            item_id = name.split("/row/", 1)[1]
        else:
            continue
        mapping.setdefault(item_id, []).append(name)
    return mapping


def id_grep_pattern(item_id: str) -> str:
    """POSIX-ERE pattern matching this row ID as a whole token, never a prefix."""
    return r"(^|[^A-Za-z0-9_-])" + re.escape(item_id) + r"([^A-Za-z0-9_-]|$)"


def main_commits(item_id: str) -> list[str]:
    """Commits on origin/main whose message mentions this row ID as a whole token.

    The match is ANCHORED on ID boundaries, never a substring: 55 pairs of live
    row IDs are prefixes of longer ones (MODEL-MM of seven MODEL-MM-* rows,
    LOAD-SAFETENSORS of LOAD-SAFETENSORS-DIRECT-DENSE, ...). A substring match
    would credit the short row with the long row's commits, and the classifier
    calls any commit LANDED -- so an abandoned row would silently report as
    finished, the exact false negative this tool exists to prevent.
    """
    out = git(
        "log", "--oneline", "-E", f"--grep={id_grep_pattern(item_id)}", "-n", "20",
        "origin/main",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def head_commits(item_id: str) -> list[str]:
    """Commits on HEAD but not yet on origin/main whose message names this row.

    The other half of the IN-FLIGHT evidence, and the only half that survives a
    FORK. A contributor's `row/<ID>` branch lives on their own repository, so no
    refspec `origin` can be given will ever produce that ref in the runner's
    checkout; on a pull request HEAD is the merge commit GitHub builds, and it
    already contains those commits. Without this arm, every PR that moves a row
    to ACTIVE and carries its implementation reads ABANDONED (#726).

    The range is `origin/main..HEAD`, never bare HEAD: on the push-to-main lane
    HEAD *is* origin/main, and a bare HEAD would re-report every landed commit
    as live work, so no ACTIVE row could ever be reported stale again.

    Matched by the same anchored, whole-token pattern as main_commits, and by
    the same `--grep` over the whole message rather than a filter on `--oneline`
    output: a row ID often appears only in the commit BODY (`25df7468f` is the
    shipped example), which a subject-only filter would miss.
    """
    out = git(
        "log", "--oneline", "-E", f"--grep={id_grep_pattern(item_id)}", "-n", "20",
        "origin/main..HEAD",
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def require_origin_main() -> None:
    """Abort unless origin/main resolves.

    git() maps any failure to "", which downstream is indistinguishable from
    "no evidence". An unfetched or missing origin/main would therefore make
    EVERY row look abandoned and the audit would propose downgrading all 54
    ACTIVE rows at once. Absence of work and absence of information must never
    look the same.
    """
    if not git("rev-parse", "--verify", "--quiet", "origin/main").strip():
        raise SystemExit(
            "origin/main does not resolve -- run `git fetch origin main` first. "
            "Without it every row reports no Git evidence and this audit would "
            "propose downgrading every ACTIVE row."
        )


def require_branch_information() -> None:
    """Abort unless the checkout holds at least one `row/<ID>` ref.

    The guard require_origin_main has, applied to the input it never covered.
    row_branches() returning {} means either "nobody holds a branch" or "this
    checkout was never told about branches", and classify_active reads both as
    the first. CI fetched `main` and nothing else, so the IN-FLIGHT verdict was
    literally unreachable there and in-flight work wore abandoned work's face
    (#726). Absence of work and absence of information must never look the same.

    A PRECONDITION, not a check on the result, because the damage is not
    confined to the ABANDONED verdict: with no branch refs a row that is
    IN-FLIGHT *and* has landed groundwork reports LANDED instead -- a live claim
    reported as finished, the false negative classify_active's own comment
    names. The whole census is untrustworthy, not just its abandoned rows.

    HEAD deliberately does not satisfy this. It carries evidence about the row
    the current pull request advances and no other, so counting it would make
    the guard vacuous -- HEAD always resolves -- while every other row stayed
    silently misclassified.
    """
    if not row_branches():
        raise SystemExit(
            "no `row/<ID>` ref resolves -- run `git fetch origin "
            "'+refs/heads/row/*:refs/remotes/origin/row/*'` first. Without them "
            "the IN-FLIGHT verdict is unreachable, so this audit would report "
            "every row whose work is on a branch as abandoned or as finished."
        )


def unmerged(branch: str) -> list[str]:
    """Commits on branch that are not yet on origin/main."""
    out = git("log", "--oneline", f"origin/main..{branch}")
    return [line.strip() for line in out.splitlines() if line.strip()]


VERDICTS = frozenset({"IN-FLIGHT", "LANDED", "ABANDONED"})


def classify_active(
    branches: list[str],
    unmerged_by_branch: dict[str, list[str]],
    commits: list[str],
    head_commits: list[str],
) -> tuple[str, str]:
    """Classify one ACTIVE row from already-gathered evidence.

    IN-FLIGHT wins over LANDED whenever both are present: a row can have landed
    groundwork and still have open follow-up work, and calling that finished
    would silently steal a live claim. That is why BOTH in-flight arms -- the
    branch and HEAD -- sit above both landed arms.

    `head_commits` is required, not defaulted. A default would let a caller
    gather the branch evidence, forget this, and lose the only source that
    survives a fork, with every test here still green.
    """
    # Indexed, never .get(): `branches` is the authority for which keys must
    # exist, so a missing key is a CALLER bug, not data. .get() would return
    # None -> falsy -> the row reports LANDED, a live claim reported as
    # finished. Absence of work and absence of information must never look the
    # same; a KeyError at the audit's own boundary is the loud alternative.
    live_branches = [b for b in branches if unmerged_by_branch[b]]
    if live_branches:
        joined = ", ".join(sorted(live_branches))
        return "IN-FLIGHT", f"unmerged commits on {joined}"
    # Below the branch arm because a named branch is the more specific evidence
    # -- both are IN-FLIGHT, so only the REASON can tell a reader which it was.
    if head_commits:
        return "IN-FLIGHT", f"unmerged commits on HEAD: {head_commits[0]}"
    if branches:
        joined = ", ".join(sorted(branches))
        return "LANDED", f"branch {joined} exists and is fully merged into main"
    if commits:
        return "LANDED", f"on main: {commits[0]}"
    return "ABANDONED", "no branch, no commit on main mentioning the row ID"


GAP_MARKERS = (
    "missing",
    "not yet",
    "unsupported",
    "pending",
    "only",
    "absent",
    "gap",
    "no",
    "without",
    "blocked",
    "todo",
)

def gap_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    """Compile markers into a whole-word, case-insensitive alternation.

    Whole words, never substrings: "only" must not match "commonly" and "no"
    must not match "node". A substring match would silently mark a vague row as
    explicit, which is the exact failure this flag exists to catch.

    Markers are ESCAPED before interpolation. GAP_MARKERS invites human tuning,
    and a raw marker fails two ways: "fp4(" raises re.error at IMPORT time and
    takes this whole module down with it, while "not.yet" compiles silently
    into a wildcard that also matches "notXyet". re.escape("not yet") is
    "not\\ yet", so widening that escaped space to \\s+ still works.

    Taking the markers as an argument is what makes the escaping testable: no
    shipped marker needs escaping today, so an inline expression could drop
    re.escape with nothing to notice.
    """
    alternation = "|".join(
        re.escape(marker).replace("\\ ", r"\s+") for marker in markers
    )
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


GAP_RE = gap_pattern(GAP_MARKERS)

# check mode fails on abandoned ACTIVE rows and nothing else. The PARTIAL flag
# is a keyword heuristic for human review; gating on it would be the fragile
# checker the protocol warns against.
CHECK_FAILS_ON = frozenset({"ACTIVE"})


def names_missing_modes(row_text: str) -> bool:
    """True when a PARTIAL row states what is NOT supported."""
    return GAP_RE.search(row_text) is not None


def matched_marker(row_text: str) -> str:
    """The gap marker that fired, or "" -- so a human can discount a bad hit.

    The heuristic under-flags: 11 of the 48 shipped rows it reads as explicit
    qualify only via bare `no` or `gap`, on prose asserting GOODNESS rather
    than absence ("no longer double-resides", "max gap 0.0 nats", "CLOSED the
    CPU RSS gap"). Naming the marker lets a reviewer dismiss those at a glance
    instead of trusting the verdict. Over-flagging costs a glance;
    under-flagging ships a vague public issue.
    """
    match = GAP_RE.search(row_text)
    return match.group(0) if match else ""


# Every PARTIAL row carries a flag string now -- either the marker that fired
# or this. So "needs review" is THIS string, never merely a non-empty flag;
# counting non-empty flags would report all 68 PARTIAL rows as vague.
VAGUE_FLAG = "does not name its missing modes"


def duplicate_live_ids(rows: list) -> dict[str, list[str]]:
    """Row IDs that appear live in more than one matrix.

    BACKEND-CUDA-SM121 and BACKEND-CPU are PARTIAL in BOTH backend-matrix.md
    and feature-matrix.md, so 188 live rows carry only 186 unique IDs.
    check-agent-record.py's duplicate check only walks MATRIX_PATHS, so it has
    never seen these. Left unresolved, the backfill would mint two issues for
    one item and this audit would report each twice with identical evidence.
    """
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row.item_id, []).append(f"{row.path.name}:{row.line_no}")
    return {k: v for k, v in seen.items() if len(v) > 1}


def audit() -> list[dict]:
    """One record per live row, with verdict (ACTIVE) and flag (PARTIAL)."""
    require_origin_main()
    require_branch_information()
    parse_errors: list[str] = []
    rows = live_rows(parse_errors)
    if parse_errors:
        raise SystemExit(
            "the matrices do not parse cleanly, so the census is incomplete:\n"
            + "\n".join(parse_errors)
        )
    duplicates = duplicate_live_ids(rows)
    branches_by_id = row_branches()
    records: list[dict] = []
    for row in rows:
        verdict = ""
        reason = ""
        flag = ""
        if row.state == "ACTIVE":
            branches = branches_by_id.get(row.item_id, [])
            unmerged_by_branch = {b: unmerged(b) for b in branches}
            verdict, reason = classify_active(
                branches,
                unmerged_by_branch,
                main_commits(row.item_id),
                head_commits(row.item_id),
            )
        if row.state == "PARTIAL":
            marker = matched_marker(row.raw)
            flag = f"explicit via {marker!r}" if marker else VAGUE_FLAG
        records.append(
            {
                "duplicate": ", ".join(duplicates.get(row.item_id, [])),
                "id": row.item_id,
                "state": row.state,
                "path": str(row.path.relative_to(ROOT)),
                "line": row.line_no,
                "verdict": verdict,
                "reason": reason,
                "flag": flag,
            }
        )
    return records


def _cell(value: object) -> str:
    """Table cells never contain a raw pipe -- it would split the row."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(records: list[dict]) -> str:
    lines = [
        "| Row | State | Location | Verdict | Evidence | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for item in records:
        lines.append(
            "| `{id}` | `{state}` | {path}:{line} | {verdict} | {reason} | {flag} |".format(
                id=_cell(item["id"]),
                state=_cell(item["state"]),
                path=_cell(item["path"]),
                line=_cell(item["line"]),
                verdict=_cell(item["verdict"]) or "-",
                reason=_cell(item["reason"]) or "-",
                flag=_cell(item["flag"]) or "-",
            )
        )
    return "\n".join(lines)


def exit_code(records: list[dict], check: bool) -> int:
    if not check:
        return 0
    abandoned = [
        item
        for item in records
        if item["state"] in CHECK_FAILS_ON and item["verdict"] == "ABANDONED"
    ]
    return 1 if abandoned else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any ACTIVE row is abandoned",
    )
    args = parser.parse_args(argv)

    records = audit()
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        print(render_markdown(records))
        stale = [i for i in records if i["verdict"] == "ABANDONED"]
        vague = [i for i in records if i["flag"] == VAGUE_FLAG]
        dupes = sorted({i["id"] for i in records if i["duplicate"]})
        print(
            f"\n{len(records)} live rows; {len(stale)} abandoned ACTIVE; "
            f"{len(vague)} PARTIAL rows to review; "
            f"{len(dupes)} IDs live in two matrices: {', '.join(dupes) or 'none'}."
        )
    return exit_code(records, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
