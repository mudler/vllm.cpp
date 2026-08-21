#!/usr/bin/env python3
"""Public documents change when the project's claims change -- not when a file moves.

WHY THIS WAS REWRITTEN. The previous version classified by WHICH DIRECTORY a
change touched: anything under src/, include/ or tests/ became a
"feature_checkpoint" and owed docs/STATUS.md + docs/BENCHMARKS.md +
.agents/NOW.md. A one-line compile fix owed three public-doc edits, so this gate
produced 16 of the last 20 red CI runs, and it had accreted SIX hardcoded
exact-path-set escape hatches -- one per legitimate change it had blocked.

The trigger was simply wrong. STATUS.md and BENCHMARKS.md mirror what the
project CLAIMS, and that changes when a row changes lifecycle state or gains a
measurement -- not when someone edits a file. So that is what this now asks.

Editing src/ alone owes nothing. Moving a row READY -> ACTIVE -> DONE owes the
checkpoint surfaces. All six escape hatches deleted with the wrong trigger,
because they only ever existed to compensate for it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

STATUS = "docs/STATUS.md"
BENCHMARKS = "docs/BENCHMARKS.md"
FEATURES = "docs/FEATURES.md"
USAGE = "docs/USAGE.md"
README = "README.md"
NOW = ".agents/NOW.md"

PUBLIC_SURFACES = frozenset({STATUS, BENCHMARKS, FEATURES, USAGE, README})

# Rows live in these keyed tables. A lifecycle move in any of them is a claim
# about the project, which is exactly what the public surfaces project.
ROW_TABLES = (
    ".agents/roadmap_v1.md",
    ".agents/backend-matrix.md",
    ".agents/engine-matrix.md",
    ".agents/feature-matrix.md",
    ".agents/kernel-matrix.md",
    ".agents/model-matrix.md",
    ".agents/quantization-matrix.md",
)

# An accepted or retracted measurement is a claim too, even without a state move.
MEASUREMENT_RECORDS = (".agents/benchmark-record.md",)

# WHAT COUNTS AS A LIFECYCLE STATE. This tuple is the whole definition, and a
# state missing from it is not mislabelled -- row_states DROPS the row, and both
# lifecycle_moves and moved_rows iterate the AFTER map, so leaving the matched
# set is silent by construction.
#
# 2026-08-21 (GATE-DOC-CHECKPOINT-STATES, #1434): +PARTIAL. It was absent while
# being the second most used state in the matrices -- 118 backticked cells at
# 947e5f648 against 77 for DONE -- so READY -> PARTIAL and PARTIAL -> READY both
# returned rc 0, and PARTIAL -> ACTIVE red for the wrong reason, calling a row
# that had existed for months `added as ACTIVE`. Admitting it takes the resolved
# population over ROW_TABLES from 153 rows to 226.
#
# ANCHOR-BACKFILL is DELIBERATELY NOT HERE, although .agents/feature-matrix.md
# names the two together. This tuple triggers REQUIRED["lifecycle"], which is
# (STATUS, BENCHMARKS) and carries all of them or none, so the question is what
# docs/STATUS.md projects. `Partial` is a term on that page (docs/STATUS.md:39).
# ANCHOR-BACKFILL is a property of the RECORD -- "a legacy implemented row
# without exact code, test and real-spec anchors" -- the capability is already
# implemented, the page carries no matching term, and a DONE <-> ANCHOR-BACKFILL
# move would demand a public-document edit with nothing true to write. That is
# the shape this file's header records as the reason for the rewrite. The
# `## Now` half it genuinely owes needs a spec-only class in REQUIRED and is
# filed under `## Owed` in .agents/specs/doc-checkpoint-lifecycle-states.md.
#
# INVENTORIED and SPIKE stay out for the older reason: they are pre-claim, which
# is the same polarity as the {ACTIVE, GATING, DONE} new-row set below.
STATES = (
    "TODO",
    "READY",
    "ACTIVE",
    "GATING",
    "PARTIAL",
    "BLOCKED",
    "DONE",
    "DROPPED",
    "N/A",
)
STATE_CELL = re.compile(r"`(" + "|".join(re.escape(s) for s in STATES) + r")`")
ROW_ID = re.compile(r"^\|\s*`([A-Z0-9][A-Za-z0-9_.-]*)`")

# Support-surface triggers. These four records ARE the claim, so editing one is
# path-derived. Model SOURCE is not: see registration_changes() (#595), which
# asks whether the registry set moved rather than whether a file was touched.
FEATURE_SURFACE_FILES = frozenset(
    {
        ".agents/backend-matrix.md",
        ".agents/feature-matrix.md",
        ".agents/model-matrix.md",
        ".agents/quantization-matrix.md",
    }
)
FEATURE_SURFACE_PREFIXES = ("src/vllm/model_executor/models/",)

# The registry's own entry point. `feature_surface` keys off a change to the SET
# of these in a model TU, not off the path -- see registration_changes() (#595).
REGISTRATION = re.compile(r"REGISTER_VLLM_MODEL\(\s*([A-Za-z0-9_:\"]+)")

# Exact user-facing configuration/build/install entrypoints. Deliberately NOT
# all of cmake/: toolchain internals do not change installation instructions.
USER_USAGE_FILES = frozenset(
    {
        ".env.example",
        "CMakeLists.txt",
        "cmake/install.cmake",
        "examples/CMakeLists.txt",
        "examples/cli/main.cpp",
        "examples/server/main.cpp",
        "include/vllm.h",
    }
)
USER_USAGE_PREFIXES = (
    "include/vllm/",
    "src/vllm/entrypoints/",
    "examples/cli/",
    "examples/server/",
)

# README permission and obligation come only from underlying landing sources.
# Co-edited public projections can NEVER justify README churn -- that rule is
# deliberate and directly tested.
#
# docs/QUICKSTART.md joined the set on 2026-08-20 (#1520). Every other member is
# something the README QUOTES: the mission, the build entry point, the demo
# numbers, the two example mains. The quickstart page is the same relation with
# the direction made explicit -- the README `## Quickstart` block stopped
# carrying the commands and now points at that page, so the claim "this is where
# a reader starts" changed BECAUSE the page exists. It is a source, not a
# projection: nothing else records what it says, and the README defers to it.
#
# This admits exactly one document and no class. docs/BUILD.md, docs/STATUS.md
# and every other page under docs/ still cannot license a README claim change,
# which tests/scripts/test_doc_checkpoint.py pins directly.
LANDING_SOURCE_FILES = frozenset(
    {
        ".agents/mission.md",
        "CMakeLists.txt",
        "benchmarks/demo/footprint_gb10.json",
        "benchmarks/demo/qwen36_27b_c1_c32.json",
        "benchmarks/demo/vulkan_27b_llamacpp.json",
        "docs/QUICKSTART.md",
        "examples/cli/main.cpp",
        "examples/server/main.cpp",
    }
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def blob(revision: str, path: str) -> str:
    try:
        return git("show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return ""


def row_states(text: str) -> dict[str, str]:
    """Map row ID -> lifecycle state for every keyed row in a table."""
    states: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        identifier = ROW_ID.match(line)
        if not identifier:
            continue
        cells = STATE_CELL.findall(line)
        if cells:
            # The state cell is the last backticked state on the row; earlier
            # ones belong to prose in the evidence columns.
            states[identifier.group(1)] = cells[-1]
    return states


SPEC_LINK = re.compile(r"\.agents/specs/([A-Za-z0-9_.-]+\.md)|\(specs/([A-Za-z0-9_.-]+\.md)\)")
NOW_SECTION = re.compile(r"^##\s+Now\s*$", re.MULTILINE)


def moved_rows(paths: set[str], before: str, after: str) -> dict[str, str]:
    """Map row ID -> the table it moved in, for every lifecycle move."""
    moved: dict[str, str] = {}
    for path in sorted(paths & set(ROW_TABLES)):
        old = row_states(blob(before, path))
        new = row_states(blob(after, path))
        for row, state in sorted(new.items()):
            previous = old.get(row)
            if previous is None:
                if state in {"ACTIVE", "GATING", "DONE"}:
                    moved[row] = path
            elif previous != state:
                moved[row] = path
    return moved


def spec_for_row(text: str, row: str) -> str | None:
    """Return the spec path a matrix row links, or None when it links none."""
    for line in text.splitlines():
        identifier = ROW_ID.match(line)
        if not identifier or identifier.group(1) != row:
            continue
        match = SPEC_LINK.search(line)
        if match is None:
            return None
        slug = match.group(1) or match.group(2)
        return f".agents/specs/{slug}"
    return None


def spec_now_errors(paths: set[str], before: str, after: str) -> list[str]:
    """The relocated freshness obligation (ENG-NOW-DERIVED, #374).

    A row that moves lifecycle state must say, in ITS OWN spec, what the next
    step now is. That is what .agents/NOW.md's per-row line used to carry, and
    moving it here is the whole point: a spec has ONE writer, so the requirement
    stops serialising every concurrent PR through one shared digest.

    Reported, never inferred: a row whose matrix line links no spec is an error
    naming the row, not a silent pass.
    """
    errors: list[str] = []
    for row, table in sorted(moved_rows(paths, before, after).items()):
        spec = spec_for_row(blob(after, table), row)
        if spec is None:
            errors.append(
                f"{table}: {row} moved lifecycle state but its row links no spec, "
                "so there is nowhere to record what happens next; add the "
                "`Spike/spec` link"
            )
            continue
        if spec not in paths:
            errors.append(
                f"{spec} is not in this change: {row} moved lifecycle state, so "
                "its spec owes an updated `## Now` line. This replaced the old "
                ".agents/NOW.md requirement -- the live position is recorded per "
                "ROW now, in a file only this row's author writes"
            )
            continue
        body = blob(after, spec)
        section = NOW_SECTION.search(body)
        if section is None:
            errors.append(
                f"{spec} has no `## Now` section; {row} moved lifecycle state and "
                "the spec is where its live position is recorded"
            )
            continue
        rest = body[section.end():]
        following = rest.split("\n##", 1)[0].strip()
        if not following:
            errors.append(
                f"{spec}: `## Now` is empty; {row} moved lifecycle state, so say "
                "what the next command or step actually is"
            )
    return errors


def lifecycle_moves(paths: set[str], before: str, after: str) -> list[str]:
    """Return 'ROW: OLD -> NEW' for every row that changed lifecycle state."""
    moves: list[str] = []
    for path in sorted(paths & set(ROW_TABLES)):
        old = row_states(blob(before, path))
        new = row_states(blob(after, path))
        for row, state in sorted(new.items()):
            previous = old.get(row)
            if previous is None:
                # A brand new row only counts once it is making a claim.
                if state in {"ACTIVE", "GATING", "DONE"}:
                    moves.append(f"{path}: {row} added as {state}")
            elif previous != state:
                moves.append(f"{path}: {row} {previous} -> {state}")
    return moves


MARKDOWN_LINK_TARGET = re.compile(r"\]\([^)]*\)")


def claims_changed(before_text: str, after_text: str) -> bool:
    """Whether a file's CLAIMS changed, ignoring pure link-target repairs.

    Moving a file forces every document that links it to be edited. Those edits
    assert nothing new, so treating them as claims makes structural work
    impossible without a fake benchmark update -- which is precisely how the old
    gate grew six hardcoded escape hatches. Normalising link TARGETS away (link
    TEXT is preserved, so a reworded link still counts) keeps this general
    instead of exempting named paths.
    """

    return MARKDOWN_LINK_TARGET.sub("]()", before_text) != MARKDOWN_LINK_TARGET.sub(
        "]()", after_text
    )


def measurement_changes(paths: set[str], before: str, after: str) -> list[str]:
    return sorted(
        path
        for path in paths & set(MEASUREMENT_RECORDS)
        if claims_changed(blob(before, path), blob(after, path))
    )


def registrations(text: str) -> set[str]:
    """The architectures a model TU registers, by registry name."""
    return set(REGISTRATION.findall(text))


def registration_changes(paths: set[str], before: str, after: str) -> list[str]:
    """Model files whose set of REGISTER_VLLM_MODEL entries changed (#595).

    Keying `feature_surface` off the PATH made every edit under
    `src/vllm/model_executor/models/` owe docs/FEATURES.md, which is the same
    "classify by directory" defect this file's header says the rewrite removed
    for src/, include/ and tests/. A one-line compile fix there owed a public
    doc edit with nothing true to say; #1054 answered that demand with prose,
    the prose crossed the check-public-doc-tables budgets, and because that
    checker also runs pre-push it blocked EVERY branch in the repo (#1055).

    What the project supports is what the registry registers, which is the same
    set scripts/check-supported-models.py already gates FEATURES.md against. So
    adding, removing or renaming an architecture is a claim and still owes the
    surface; editing the internals of one already registered is not.
    """

    return sorted(
        path
        for path in paths
        if path.startswith(FEATURE_SURFACE_PREFIXES)
        and registrations(blob(before, path)) != registrations(blob(after, path))
    )


def classify(paths: set[str], before: str, after: str) -> tuple[set[str], list[str]]:
    """Return (change classes, human-readable reasons)."""
    classes: set[str] = set()
    reasons: list[str] = []

    moves = lifecycle_moves(paths, before, after)
    if moves:
        classes.add("lifecycle")
        reasons.extend(moves)

    measured = measurement_changes(paths, before, after)
    if measured:
        classes.add("lifecycle")
        reasons.extend(f"{path}: measurement recorded" for path in measured)

    registered = registration_changes(paths, before, after)
    if registered:
        classes.add("feature_surface")
        reasons.extend(f"{path}: model registration changed" for path in registered)

    for path in sorted(paths):
        if path in PUBLIC_SURFACES:
            continue
        if path in FEATURE_SURFACE_FILES:
            classes.add("feature_surface")
        if path in USER_USAGE_FILES or path.startswith(USER_USAGE_PREFIXES):
            classes.add("user_usage")
        if path in LANDING_SOURCE_FILES:
            classes.add("landing_page")

    return classes, reasons


# NOW LEFT THIS TRIPLE 2026-08-11 (ENG-NOW-DERIVED, #374). Requiring it here is
# what made .agents/NOW.md a surface EVERY row-advancing PR writes: the gate
# marched them into one shared file, and it conflicted in 5 of the 16 conflicting
# open PRs measured at d928e2c3. That is a lock under the AGENTS.md invariant
# "no surface that every PR must write", which #364 added while this line stood.
#
# The obligation is RELOCATED, not dropped. "The live position must be current"
# is now paid in the moved row's OWN spec, as a `## Now` line -- see
# spec_now_errors below. A spec has one writer, so the same requirement no longer
# serialises everyone. .agents/NOW.md keeps only what is authored at operator
# cadence and is otherwise rendered by scripts/now.py.
REQUIRED = {
    "lifecycle": (STATUS, BENCHMARKS),
    "feature_surface": (FEATURES,),
    "user_usage": (USAGE,),
    "landing_page": (README,),
}


def errors_for(paths: set[str], before: str, after: str) -> list[str]:
    classes, reasons = classify(paths, before, after)

    errors = spec_now_errors(paths, before, after)

    required: set[str] = set()
    for change_class in classes:
        # landing_page PERMITS a README change; it does not demand one.
        if change_class == "landing_page":
            continue
        required.update(REQUIRED[change_class])

    missing = sorted(required - paths)
    if missing:
        detail = "; ".join(reasons) if reasons else ", ".join(sorted(classes))
        why = (
            "A lifecycle move or a new measurement is a claim about the "
            "project, and these surfaces are how that claim reaches a reader."
            if "lifecycle" in classes
            else "This change alters what the project supports or how it is "
            "used, which its purpose-specific document has to reflect."
        )
        errors.append(
            f"changed {detail} but did not update {', '.join(missing)}. {why}"
        )

    if (
        README in paths
        and "landing_page" not in classes
        and claims_changed(blob(before, README), blob(after, README))
    ):
        errors.append(
            f"changed {README} without touching a landing source "
            f"({', '.join(sorted(LANDING_SOURCE_FILES))}). The README is the "
            "landing page; routine checkpoints belong in the purpose-specific "
            "docs. Co-edited public projections never justify README churn."
        )

    return errors


def commit_paths(commit: str) -> set[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        output = git("diff", "--name-only", parents[0], commit)
    else:
        output = git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit)
    return {line for line in output.splitlines() if line}


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    output = git("rev-list", "--reverse", "--no-merges", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", default=None, help="check one commit")
    source.add_argument("--staged", action="store_true", help="check the staged change")
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    return args


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    if args.staged:
        paths = {p for p in git("diff", "--cached", "--name-only").splitlines() if p}
        failures.extend(
            f"staged change: {error}" for error in errors_for(paths, "HEAD", "")
        )
    elif args.base is not None:
        for commit in commits_in_range(args.base, args.head):
            paths = commit_paths(commit)
            parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
            before = parents[0] if parents else EMPTY_TREE
            short = git("rev-parse", "--short", commit)
            failures.extend(
                f"commit {short}: {error}"
                for error in errors_for(paths, before, commit)
            )
    elif args.commit is not None:
        paths = commit_paths(args.commit)
        parents = git("rev-list", "--parents", "-n", "1", args.commit).split()[1:]
        before = parents[0] if parents else EMPTY_TREE
        short = git("rev-parse", "--short", args.commit)
        failures.extend(
            f"commit {short}: {error}"
            for error in errors_for(paths, before, args.commit)
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("OK: public documents match the claims this change makes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
