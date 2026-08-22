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

# WHAT COUNTS AS A LIFECYCLE STATE. These two tuples are the whole definition,
# and a state missing from BOTH is not mislabelled -- row_states DROPS the row
# and transitions() iterates the AFTER map, so leaving the resolved set is
# silent by construction.
#
# 2026-08-21 (GATE-DOC-CHECKPOINT-STATES, #1434): +PARTIAL. It was absent while
# being the second most used state in the matrices -- 118 backticked cells at
# 947e5f648 against 77 for DONE -- so READY -> PARTIAL and PARTIAL -> READY both
# returned rc 0, and PARTIAL -> ACTIVE red for the wrong reason, calling a row
# that had existed for months `added as ACTIVE`. Admitting it takes the resolved
# population over ROW_TABLES from 153 rows to 226.
#
# STATES are the CLAIM states: a move between two of them is something
# docs/STATUS.md carries a term for. `Partial` is on that page (docs/STATUS.md:39).
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

# RECORD states are lifecycle positions of the ROW'S RECORD rather than of the
# CAPABILITY. .agents/feature-matrix.md:14-20 names all three: INVENTORIED and
# SPIKE are pre-claim, and ANCHOR-BACKFILL is a property of the RECORD -- "a
# legacy implemented row without exact code, test and real-spec anchors" -- so
# the capability is already implemented and only the row's anchors are missing.
#
# 2026-08-21, W3: the criterion W2 wrote here -- "real lifecycle positions that
# the public pages carry NO term for" -- is FALSE, and it was derived for one
# state and then applied to three. docs/STATUS.md:42 defines
# `| Inventoried | The gap has a stable record but no accepted implementation |`
# and :56 uses it in a live projection cell ("Distributed execution | Inventoried
# or partial by lane"). In the other direction READY, TODO, BLOCKED, DROPPED and
# N/A are all in STATES and docs/STATUS.md carries no term for any of them, grep
# count 0 each. Word-on-the-page is therefore neither necessary nor sufficient,
# and tests/scripts/test_doc_checkpoint.py pins both halves of that so the wrong
# criterion cannot be reintroduced by quoting this file.
#
# Re-derived per state, separately, on what REQUIRED["lifecycle"] actually
# demands -- (STATUS, BENCHMARKS), carried all or none:
#
#   INVENTORIED -- ADMITTED to the page, DECLINED for the tuple. STATUS.md's own
#     definition describes the RECORD ("the gap has a stable record") together
#     with the ABSENCE of an implementation, so carrying the word does not make
#     arriving there a capability claim. Neither obvious cost decides it, and
#     both were measured rather than argued: moving it into STATES changes 0 of
#     793 row resolutions, and the 400-commit replay ending at e2a9e035d is
#     identical either way. What decides it is that INVENTORIED is where a
#     PRE-CLAIM row and a DEMOTED row both sit, so admitting it makes
#     SPIKE -> INVENTORIED demand docs/STATUS.md and docs/BENCHMARKS.md for a row
#     that has never claimed anything -- the public-document edit with nothing
#     true to write that this file's header records as the reason for the
#     rewrite. The half that genuinely goes unpaid is a DEMOTION out of a claim
#     state into it; that is listed under `## Owed` in the row's spec, and it is
#     0 of the 17 non-arrival transitions in the same 400 commits.
#
#   SPIKE -- pre-claim by protocol: .agents/feature-matrix.md gives a SPIKE row a
#     `CLAIM-*` and not a spec, and docs/STATUS.md carries no term (grep 0).
#     Neither limb of the criterion argues for admitting it.
#
#   ANCHOR-BACKFILL -- W1's ruling, re-derived independently and unchanged. It is
#     a property of the record BY DEFINITION, docs/STATUS.md carries no term
#     (grep 0), and a DONE <-> ANCHOR-BACKFILL move changes nothing a reader of
#     that page could be told.
#
# 2026-08-21, W2 of the same row: they are RESOLVED but not CLASSIFIED as claims.
# The earlier reading (recorded in doc-checkpoint-lifecycle-states.md `## Owed`)
# was that paying the `## Now` half needed REQUIRED to carry a spec-only class.
# That premise was wrong, and measuring it is what closed the residual:
# spec_now_errors() is called from errors_for() DIRECTLY, never through classify()
# or REQUIRED, so the spec-only obligation already existed. The only reason an
# ANCHOR-BACKFILL move was unobserved is that row_states DROPPED the row before
# any of it ran. So the fix is resolution, not a new REQUIRED class: these three
# states resolve, a move that touches one is a record move, and a record move
# owes its spec's `## Now` and NOT (STATUS, BENCHMARKS) -- which would be a
# public-document edit with nothing true to write, the exact shape this file's
# header records as the reason for the rewrite.
#
# Resolution takes the population over ROW_TABLES from 226 rows to 793.
RECORD_STATES = ("INVENTORIED", "SPIKE", "ANCHOR-BACKFILL")

LIFECYCLE_STATES = STATES + RECORD_STATES

STATE_CELL = re.compile(r"`(" + "|".join(re.escape(s) for s in STATES) + r")`")

# A record state counts only where it OPENS a markdown cell, which a claim state
# is not required to do. That asymmetry is measured, not stylistic. row_states
# takes the LAST backticked state on the line because earlier ones belong to the
# evidence columns, and the evidence prose in this repository narrates record
# states far more often than claim ones -- "the row stays `SPIKE`" appears mid
# sentence on two model rows whose real State cells read `BLOCKED` and `ACTIVE`.
# Matching those anywhere on the line resolved both rows to `SPIKE` and would
# have turned a correct resolution into a wrong one. Cell-anchored, the same two
# rows keep their true states, and the two rows the widening REPAIRS
# (engine-matrix KV-EXTERNAL-CACHE ACTIVE -> ANCHOR-BACKFILL, quantization-matrix
# QUANT-GGUF-PRESETS READY -> INVENTORIED) still resolve correctly. Measured over
# all 793 rows against a column-position proxy: 12 of 226 resolutions disagreed
# with it before, 10 of 793 after, and no previously-correct row regresses.
#
# THE OUTRIGHT WIN IS A LATENT HAZARD, recorded here so the next reader does not
# have to rediscover it. row_states takes the record cell and `continue`s past
# STATE_CELL, so an EVIDENCE cell that OPENS with a backticked record state
# demotes a claim row no matter where the real State column sits. Zero rows do
# that today, and the sized repair is already measured: taking whichever of the
# last RECORD_CELL and the last STATE_CELL match ends LATER in the line changes
# 0 of the same 793 resolutions. It is not made here because it is a separate
# semantic change to the resolver and wants its own red-before case; it is
# listed under `## Owed` in .agents/specs/doc-checkpoint-lifecycle-states.md.
RECORD_CELL = re.compile(
    r"\|\s*`(" + "|".join(re.escape(s) for s in RECORD_STATES) + r")`"
)
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
#
# include/vllm.h joined the set on 2026-08-22 (#1655). It is the QUOTES relation
# again, and it was the only such source missing: the README's `## Use it as a
# library (C API)` block quotes `VLLM_ABI_VERSION` straight out of the header.
# The header was already a USER_USAGE_FILES member, so an ABI change owed
# docs/USAGE.md -- but not being a landing source meant no edit to the header
# could ever license repairing the claim the header itself invalidated. The
# README consequently sat at `21` against a header reading `23`, with no legal
# change that could fix it. Admitting the file closes that trap; it does not
# admit `include/` as a class, and it demands nothing, so an ordinary ABI change
# still owes only docs/USAGE.md.
LANDING_SOURCE_FILES = frozenset(
    {
        ".agents/mission.md",
        "CMakeLists.txt",
        "include/vllm.h",
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
    """Map row ID -> lifecycle state for every keyed row in a table.

    A cell-anchored RECORD state wins outright, because a row whose State cell
    reads `INVENTORIED`, `SPIKE` or `ANCHOR-BACKFILL` is by definition making no
    claim, and resolving it to a claim state mentioned later in its evidence
    prose is the one error that makes this gate demand the wrong surfaces. See
    the RECORD_CELL comment for the measurement behind the asymmetry.
    """
    states: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        identifier = ROW_ID.match(line)
        if not identifier:
            continue
        record = RECORD_CELL.findall(line)
        if record:
            states[identifier.group(1)] = record[-1]
            continue
        cells = STATE_CELL.findall(line)
        if cells:
            # The state cell is the last backticked state on the row; earlier
            # ones belong to prose in the evidence columns.
            states[identifier.group(1)] = cells[-1]
    return states


SPEC_LINK = re.compile(r"\.agents/specs/([A-Za-z0-9_.-]+\.md)|\(specs/([A-Za-z0-9_.-]+\.md)\)")
NOW_SECTION = re.compile(r"^##\s+Now\s*$", re.MULTILINE)


# A row absent from the BEFORE map is only a claim once it arrives in one of
# these.
#
# 2026-08-21, W2 (#1434): +PARTIAL, which OVERTURNS the exclusion the first wave
# pinned. That exclusion rested on an unmeasured premise -- "the dominant real
# cause of a row being absent from BEFORE is a record RELOCATION between
# matrices" -- and the premise is false here. Replayed over the 400 non-merge
# commits ending at e2a9e035d, 126 of which touch a row table: 45 arrivals, ALL
# of them genuinely new (the row ID was in no other ROW_TABLE beforehand), ZERO
# relocations, and ZERO departures. Two of the 45 arrived as PARTIAL, so the
# case the exclusion protected does not occur and the case it hid does.
#
# `Partial` is a docs/STATUS.md term (:39), so a row arriving in it asserts
# exactly what that page projects: a usable path with named missing behavior.
#
# The record states stay out for the opposite and still-sound reason: a row
# arriving as INVENTORIED or SPIKE has no prior position and claims nothing.
CLAIM_ON_ARRIVAL = frozenset({"ACTIVE", "GATING", "DONE", "PARTIAL"})

CLAIM = "claim"
RECORD = "record"


def transitions(
    paths: set[str], before: str, after: str
) -> list[tuple[str, str, str | None, str, str]]:
    """Every lifecycle move, as (table, row, previous, state, kind).

    `kind` is keyed on the DESTINATION alone. A move INTO a claim state is a
    CLAIM. A move into a record state is a RECORD move, which is still a
    lifecycle move -- AGENTS.md `## Public documents` owes the moved row spec's
    `## Now` for ANY state change -- it simply owes nothing the public pages
    could truthfully say.

    2026-08-21, W3 (#1434). W2 keyed on BOTH endpoints, and that LOOSENED the
    gate on nine transitions. Before W2 a row sitting in INVENTORIED, SPIKE or
    ANCHOR-BACKFILL was absent from the BEFORE map, so `previous` was None and
    the ARRIVAL rule fired, pulling in REQUIRED["lifecycle"]. Resolving the
    record states made `previous` resolve, which made the arrival rule
    unreachable, and the both-endpoints test then sent the move down the RECORD
    branch and dropped (STATUS, BENCHMARKS). So every member of
    {INVENTORIED, SPIKE, ANCHOR-BACKFILL} x {ACTIVE, GATING, DONE} went RED ->
    GREEN across W2, with the spec `## Now` paid and the public surfaces
    withheld: 9 of 9 rc 1 at e2a9e035d, 9 of 9 rc 0 at ba4634204, measured on
    scratch commits. 569 of the 793 resolved rows sit in a record state, so that
    is the exit path of most of the tree.

    Destination-keying closes all nine, keeps W1's ANCHOR-BACKFILL ruling
    (DONE -> ANCHOR-BACKFILL is still RECORD), and matches the polarity
    CLAIM_ON_ARRIVAL already uses -- a rule that reads the destination and
    ignores where the row came from. Replayed over the 400 non-merge commits
    ending at e2a9e035d: 3 newly red against the pre-W2 gate and 0 newly green,
    each of the three a genuinely unpaid surface (67e53e716, 33f570ea9,
    ab6e65216).

    THE REPLAY CANNOT BE THE WHOLE EVIDENCE, which is the lesson W2 missed. It
    measures newly-GREEN over commits that landed UNDER the obligation being
    removed, so no such commit can exist and "0 newly green" is an artifact
    rather than a safety result. The three real commits that exercise these
    transitions -- 2a976eb9f, 678fc672c, 7a0e6c82b -- are green under every
    variant BECAUSE they paid both surfaces; withhold those two paths from the
    same tree change and W2 alone reports no lifecycle move at all. The nine
    transitions are therefore pinned as CONSTRUCTED cases in
    tests/scripts/test_doc_checkpoint.py, not by replay.
    """
    moves: list[tuple[str, str, str | None, str, str]] = []
    for path in sorted(paths & set(ROW_TABLES)):
        old = row_states(blob(before, path))
        new = row_states(blob(after, path))
        for row, state in sorted(new.items()):
            previous = old.get(row)
            if previous is None:
                if state in CLAIM_ON_ARRIVAL:
                    moves.append((path, row, None, state, CLAIM))
                continue
            if previous == state:
                continue
            # Destination-keyed. The both-endpoints form is what W2
            # loosened; see the docstring for the nine transitions.
            kind = CLAIM if state in STATES else RECORD
            moves.append((path, row, previous, state, kind))
    return moves


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

    Reported, never inferred for a CLAIM move: a row whose matrix line links no
    spec is an error naming the row, not a silent pass. A RECORD move is the one
    exception, and the loop below carries the count behind it.
    """
    errors: list[str] = []
    for table, row, _previous, _state, kind in transitions(paths, before, after):
        spec = spec_for_row(blob(after, table), row)
        if spec is None:
            if kind == RECORD:
                # A pre-claim row links a `CLAIM-*`, not a spec: 416 of the 463
                # INVENTORIED rows link no spec at all, by protocol rather than
                # by omission. Demanding one here would demand a document
                # .agents/feature-matrix.md says does not exist yet. The `## Now`
                # obligation still binds every record row that DOES link a spec,
                # which is 50 of 56 ANCHOR-BACKFILL rows and 50 of 50 SPIKE rows
                # (re-derived over the seven ROW_TABLES at 503e45900; W2 recorded
                # 52 of 52 and the population has moved since).
                continue
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
    """Return 'ROW: OLD -> NEW' for every CLAIM move.

    Record moves are deliberately absent: this list is what makes classify() add
    the "lifecycle" class, and that class is (STATUS, BENCHMARKS). They are
    reported by spec_now_errors instead, which owes only the row's own spec.
    """
    return [
        f"{path}: {row} added as {state}"
        if previous is None
        else f"{path}: {row} {previous} -> {state}"
        for path, row, previous, state, kind in transitions(paths, before, after)
        if kind == CLAIM
    ]


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
