#!/usr/bin/env python3
"""Answer "what is the last SHA where `main` was fully green, and what failed?"

Issue #274, spec .agents/specs/main-verifiability.md.

Two things had to be true before that question had an answer, and neither was.

FIRST, a run has to finish. Every expensive job in ci.yml carries a job-level
concurrency group keyed on `github.ref` with `cancel-in-progress: true`. For a
push to `main` that ref is the constant `refs/heads/main`, so consecutive pushes
cancel each other's long jobs -- 26 of the last 40 `main` runs were `cancelled`,
and the cancel instant equals the next push's start instant, 3 for 3 on the runs
checked. That is deliberate for the push lane and simply incompatible with ever
finishing, so the baseline runs on its own `schedule`/`workflow_dispatch` lane
whose groups discriminate on the event and never cancel.

SECOND, and this is why nothing here reads `run.conclusion`: `sanitize-cpu` is
`continue-on-error: true`, so its failure does not fail the run. The single
completed `main` run in that window -- 31448896841 at 5812b8b6 -- reports
`conclusion: success` with BOTH sanitizer lanes `failure`. A tool that trusted
the run conclusion would publish that SHA as the known-good baseline. The
verdict is computed from PER-JOB conclusions, always.

DERIVED AT READ TIME. Nothing is written to the repository. AGENTS.md admits a
record surface in three shapes -- one file per row, append-only, or derived --
and a committed `LAST_GREEN.md` would be the worst possible instance of the
shape it forbids: one line that every merge wants to rewrite. `scripts/now.py`
is the in-repo precedent and this mirrors it, including degrading to
REMOTE_UNVERIFIED on an unreachable remote rather than rendering an absence as
"nothing failed".

    scripts/main-baseline.py                    # last green SHA + newest failures
    scripts/main-baseline.py --limit 20
    scripts/main-baseline.py --json
    scripts/main-baseline.py --run-id N --emit-summary   # used by CI
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_FILE = "ci.yml"
WORKFLOW_PATH = ROOT / ".github/workflows" / WORKFLOW_FILE
BRANCH = "main"
# The lane's own verdict job. It is still `in_progress` while it queries, so
# counting it would make every baseline permanently non-green.
SUMMARY_JOB = "baseline-summary"
# The events that constitute a baseline run. A `push` run is NOT one: its long
# jobs are cancellable by design, so its absence of failures means nothing.
BASELINE_EVENTS = ("schedule", "workflow_dispatch")
# Conclusions that mean "this job did not run", as opposed to "it ran and was
# not green". Reported separately and never counted either way.
NOT_RUN_CONCLUSIONS = frozenset({"skipped"})
GREEN = "success"

# THE JOBS A BASELINE MUST COVER, and why a set is needed at all.
#
# Without one the verdict was purely subtractive: whatever the API happened to
# return got graded, and nothing noticed what it did NOT return. Called with
# eight of the nine covered jobs simply absent, it printed GREEN with `jobs
# covered: 1`. So a job renamed, deleted or accidentally `if:`-guarded off the
# lane by an unrelated PR would silently narrow the baseline while it kept
# publishing green -- the same shape as the `continue-on-error` trap above, just
# reached by omission instead of by a masked conclusion.
#
# Pinned here and cross-checked against `baseline-summary`'s `needs:` list in
# ci.yml by `tests/scripts/test_main_baseline.py`, so renaming a job reds the
# suite in the PR that renames it rather than quietly shrinking the baseline.
EXPECTED_JOBS = (
    "agent-record",
    "build-test-cpu",
    "build-test-cpu-arm64",
    "build-test-vulkan",
    "cuda-arch-features",
    "cuda-fat-build",
    "device-leakage",
    # Joined on 2026-08-23 (#1765). It is the only lane anywhere that
    # compiles the four Metal translation units, and it runs post-merge, so
    # the baseline is where its verdict has to arrive. Leaving it out would
    # repeat #503 exactly: a compiling gate the baseline never graded.
    "macos-metal-mlx",
    "sanitize-cpu",
    "vulkan-spirv-freshness",
    # Joined on 2026-08-17 (#503). These two were `if: github.event_name ==
    # 'pull_request'` and so were never DEFINED for the baseline lane's events:
    # not skipped, not missing, not covered -- absent from the payload entirely,
    # which is the one shape the `missing` arm above exists to catch and the one
    # it could not see, because the expectation did not name them. The verdict
    # therefore printed GREEN while `main` did not compile under MSVC (#503,
    # #1068). The workflow now runs them on `schedule`/`workflow_dispatch`.
    "windows-msvc-cpu",
    "windows-msvc-vulkan",
)


def expected_jobs_from_workflow(path: Path = WORKFLOW_PATH) -> tuple[str, ...]:
    """`baseline-summary`'s `needs:` list, read from the workflow text.

    The workflow is the authority on which jobs the lane runs; `EXPECTED_JOBS`
    is the pin that a test compares against this, so the two cannot drift.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    block = re.search(
        rf"(?ms)^  {re.escape(SUMMARY_JOB)}:$\n(.*?)(?=^  [A-Za-z0-9_-]+:$|\Z)", text
    )
    if block is None:
        return ()
    # No DOTALL here, deliberately: with it, `      - .+\n` swallows the whole
    # rest of the job and the "expected jobs" set becomes shell fragments.
    needs = re.search(r"(?m)^    needs:$\n((?:      - .+\n)+)", block.group(1))
    if needs is None:
        return ()
    return tuple(
        sorted(
            line.strip()[1:].strip()
            for line in needs.group(1).splitlines()
            if line.strip()
        )
    )


def job_matches(expected: str, name: str) -> bool:
    """Does an API job name belong to expected job id `expected`?

    A matrix job reports as `sanitize-cpu (address,undefined)`, one entry per
    lane; a plain job reports as its own id.
    """

    return name == expected or name.startswith(f"{expected} (")


@dataclass
class Verdict:
    """One baseline run's per-job verdict."""

    sha: str
    created_at: str
    run_id: int
    green: bool
    failing: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    url: str = ""

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "sha": self.sha,
            "created_at": self.created_at,
            "green": self.green,
            "failing": self.failing,
            "covered": self.covered,
            "not_run": self.not_run,
            "pending": self.pending,
            "missing": self.missing,
            "url": self.url,
        }


def verdict(run: dict, jobs: list[dict], expected: tuple[str, ...] | None = None) -> Verdict:
    """Per-job verdict for one run. NEVER reads `run["conclusion"]`.

    `continue-on-error: true` on `sanitize-cpu` means the run conclusion can be
    `success` while that job is `failure`; run 31448896841 is exactly that. Fails
    closed three ways: a job that ran and was not green is `failing`, a job that
    has not finished is `pending`, and an EXPECTED job the payload never
    mentioned (or only mentions as skipped) is `missing`. None of the three is
    green, and the third is the one absence could otherwise wear green's face.
    """
    if expected is None:
        expected = EXPECTED_JOBS
    covered: list[str] = []
    failing: list[str] = []
    not_run: list[str] = []
    pending: list[str] = []
    for entry in jobs:
        name = entry.get("name", "")
        if name == SUMMARY_JOB:
            continue
        conclusion = entry.get("conclusion")
        if conclusion in NOT_RUN_CONCLUSIONS:
            not_run.append(name)
            continue
        if conclusion is None:
            # Queued or still running. Fail-closed like everything else here,
            # but it is not a FAILURE and must not be reported as one.
            pending.append(name)
            continue
        covered.append(name)
        if conclusion != GREEN:
            failing.append(name)
    # A skipped job is present in the payload and did NOT run, so it cannot
    # discharge an expectation: `skipped` reads as absent here, deliberately.
    ran = [
        entry.get("name", "")
        for entry in jobs
        if entry.get("conclusion") not in NOT_RUN_CONCLUSIONS
    ]
    missing = [
        name
        for name in expected
        if not any(job_matches(name, actual) for actual in ran)
    ]
    return Verdict(
        sha=run.get("head_sha", ""),
        created_at=run.get("created_at", ""),
        run_id=run.get("id", 0),
        green=bool(covered) and not failing and not pending and not missing,
        failing=failing,
        covered=covered,
        not_run=not_run,
        pending=pending,
        missing=missing,
        url=run.get("html_url", ""),
    )


def last_green(verdicts: list[Verdict]) -> Verdict | None:
    """Newest green verdict, or None. Input is newest-first."""
    for item in verdicts:
        if item.green:
            return item
    return None


# --------------------------------------------------------------------------
# Remote. The ONLY network path, and it degrades rather than failing.
# --------------------------------------------------------------------------


def gh_api(path: str) -> tuple[object | None, str | None]:
    try:
        result = subprocess.run(
            ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"REMOTE_UNVERIFIED: {exc.__class__.__name__}"
    if result.returncode != 0:
        detail = [line for line in result.stderr.splitlines() if line.strip()]
        return None, f"REMOTE_UNVERIFIED: {detail[-1] if detail else 'gh api failed'}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError:
        return None, "REMOTE_UNVERIFIED: unparseable gh output"


def repository() -> str:
    env = os.environ.get("GITHUB_REPOSITORY")
    if env:
        return env
    try:
        return subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "mudler/vllm.cpp"


def jobs_for(repo: str, run_id: int) -> tuple[list[dict] | None, str | None]:
    payload, degraded = gh_api(
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&filter=latest"
    )
    if degraded:
        return None, degraded
    if not isinstance(payload, dict):
        return None, "REMOTE_UNVERIFIED: unexpected jobs payload"
    return payload.get("jobs", []), None


def baseline_runs(repo: str, limit: int) -> tuple[list[dict], str | None]:
    collected: list[dict] = []
    for event in BASELINE_EVENTS:
        payload, degraded = gh_api(
            f"repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
            f"?branch={BRANCH}&event={event}&status=completed&per_page={limit}"
        )
        if degraded:
            return [], degraded
        if isinstance(payload, dict):
            collected.extend(payload.get("workflow_runs", []))
    collected.sort(key=lambda run: run.get("created_at", ""), reverse=True)
    return collected[:limit], None


def collect(limit: int) -> tuple[list[Verdict], str | None]:
    repo = repository()
    runs, degraded = baseline_runs(repo, limit)
    if degraded:
        return [], degraded
    verdicts: list[Verdict] = []
    for run in runs:
        jobs, degraded = jobs_for(repo, run.get("id", 0))
        if degraded:
            return verdicts, degraded
        verdicts.append(verdict(run, jobs or []))
    return verdicts, None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render(verdicts: list[Verdict], degraded: str | None) -> str:
    lines = [
        "main baseline -- the scheduled/dispatched lane of .github/workflows/ci.yml",
        "Verdicts are PER-JOB: sanitize-cpu is continue-on-error, so a run's own",
        "conclusion can say success while it is red. See issue #274.",
        "",
    ]
    if degraded:
        lines += [
            degraded,
            "",
            "The remote could not be queried. Unknown is neither absence nor",
            "success: this is NOT a statement that main is green.",
        ]
        return "\n".join(lines)

    if not verdicts:
        lines += [
            "No completed baseline run found on main.",
            "",
            "If this change has only just landed, the first scheduled run has not",
            "happened yet. Trigger one with:",
            "    gh workflow run ci.yml --ref main",
        ]
        return "\n".join(lines)

    for item in verdicts:
        state = "GREEN" if item.green else "RED"
        lines.append(f"{item.created_at}  {item.sha[:12]}  {state:5s}  run {item.run_id}")
        if item.failing:
            lines.append(f"    failed:    {', '.join(item.failing)}")
        if item.pending:
            lines.append(f"    pending:   {', '.join(item.pending)}")
        if item.missing:
            lines.append(f"    missing:   {', '.join(item.missing)}")
        if item.not_run:
            lines.append(f"    not run:   {', '.join(item.not_run)}")
    lines.append("")

    newest = verdicts[0]
    if newest.green:
        lines.append(f"NEWEST BASELINE: GREEN at {newest.sha}")
    else:
        lines.append(f"NEWEST BASELINE: RED at {newest.sha}")
        if newest.failing:
            lines.append(f"  failing: {', '.join(newest.failing)}")
        if newest.pending:
            lines.append(f"  pending: {', '.join(newest.pending)}")
        if newest.missing:
            lines.append(f"  missing (expected, never ran): {', '.join(newest.missing)}")

    green = last_green(verdicts)
    if green is None:
        lines.append(
            f"LAST FULLY GREEN: none -- no fully green baseline in the last "
            f"{len(verdicts)} run(s)."
        )
    else:
        lines.append(f"LAST FULLY GREEN: {green.sha}  ({green.created_at})")
    return "\n".join(lines)


def print_json(verdicts: list[Verdict], degraded: str | None) -> None:
    green = last_green(verdicts)
    json.dump(
        {
            "degraded": degraded,
            "runs": [item.as_dict() for item in verdicts],
            "last_green": green.as_dict() if green else None,
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


def emit_summary(item: Verdict, stream) -> int:
    """Write one run's verdict as Markdown; return the process exit code.

    Non-zero when any covered job is not green, sanitizers included. This job
    runs ONLY on schedule/workflow_dispatch, so it can never block a PR -- it
    makes the BASELINE honest without making the sanitizer binding for
    contributors. Reporting green while a covered job is red would rebuild, in a
    new place, the exact defect #274 was filed about.
    """
    state = "GREEN" if item.green else "RED"
    stream.write(f"# main baseline: {state}\n\n")
    stream.write(f"- commit: `{item.sha}`\n")
    stream.write(f"- run: {item.run_id}\n")
    stream.write(f"- jobs covered: {len(item.covered)} of {len(EXPECTED_JOBS)} expected\n")
    if item.not_run:
        stream.write(f"- not run (skipped on this lane): {', '.join(item.not_run)}\n")
    stream.write("\n")
    if item.failing:
        stream.write("## Failing\n\n")
        for name in item.failing:
            stream.write(f"- `{name}`\n")
        stream.write(
            "\nThe verdict is per-job. `sanitize-cpu` is `continue-on-error`, so "
            "the run's own conclusion may still read `success`.\n"
        )
    if item.pending:
        stream.write("\n## Still running\n\n")
        for name in item.pending:
            stream.write(f"- `{name}`\n")
    if item.missing:
        stream.write("\n## Expected but never ran\n\n")
        for name in item.missing:
            stream.write(f"- `{name}`\n")
        stream.write(
            "\nA baseline that does not cover every expected job certifies less "
            "than it claims, so this is RED. Renaming or removing a job re-pins "
            "`EXPECTED_JOBS` in the same change.\n"
        )
    if item.green:
        stream.write("Every expected job ran and is green.\n")
    stream.write(
        "\nRead the history with `scripts/main-baseline.py` (issue #274, spec "
        "`.agents/specs/main-verifiability.md`).\n"
    )
    return 0 if item.green else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=10, help="baseline runs to read")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--run-id", type=int, help="verdict for one specific run")
    parser.add_argument(
        "--emit-summary",
        action="store_true",
        help="write GITHUB_STEP_SUMMARY and exit non-zero when not green",
    )
    parser.add_argument("--offline", action="store_true", help="skip every network call")
    args = parser.parse_args(argv)

    if args.run_id:
        if args.offline:
            print("REMOTE_UNVERIFIED: --offline", file=sys.stderr)
            return 1
        repo = repository()
        run, degraded = gh_api(f"repos/{repo}/actions/runs/{args.run_id}")
        jobs = None
        if not degraded:
            jobs, degraded = jobs_for(repo, args.run_id)
        if degraded or not isinstance(run, dict):
            print(degraded or "REMOTE_UNVERIFIED: unexpected run payload", file=sys.stderr)
            return 1
        item = verdict(run, jobs or [])
        if args.as_json:
            print_json([item], degraded=None)
            return 0 if item.green else 1
        if args.emit_summary:
            summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary_path:
                with open(summary_path, "a", encoding="utf-8") as handle:
                    emit_summary(item, handle)
            return emit_summary(item, sys.stdout)
        print(render([item], degraded=None))
        return 0 if item.green else 1

    verdicts, degraded = ([], "REMOTE_UNVERIFIED: --offline") if args.offline else collect(args.limit)
    if args.as_json:
        print_json(verdicts, degraded)
    else:
        print(render(verdicts, degraded))
    # Absence exits NON-ZERO, for the same reason REMOTE_UNVERIFIED does. "No
    # completed baseline run found on main" is the strongest possible statement
    # that main is unverified, and exiting 0 let `main-baseline.py && echo ok`
    # read it as a pass -- the exact confusion between absence and success this
    # module was written to prevent.
    return 1 if (degraded or not verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
