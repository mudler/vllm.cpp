#!/usr/bin/env python3
"""Enforce that the public keyed-table docs stay human-readable.

Per AGENTS.md, docs/BENCHMARKS.md and docs/FEATURES.md are KEYED TABLES in the
same class as .agents/roadmap_v1.md and the area matrices: one row per subject,
updated in place. They are NOT append logs. Left unchecked they drift back into
one a checkpoint at a time (BENCHMARKS.md reached 11,405 lines and 171 sections
before the 2026-08-04 conversion), which is exactly what makes them unreadable
to users.

This checker fails if either page loses a required user-facing section, grows
past its budget, accumulates sections or prose instead of rows, or stops
pointing at the record it relocates detail into. When BENCHMARKS.md fails
because sections accumulated, `scripts/roll-benchmark-record.py` moves them into
.agents/benchmark-record.md.

The validation logic is the pure functions `benchmarks_errors(text)` and
`features_errors(text)` so they are unit-testable and mutation-testable (see
tests/scripts/test_check_public_doc_tables.py), mirroring
check-readme-structure.py and check-doc-checkpoint.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "docs/BENCHMARKS.md"
FEATURES = ROOT / "docs/FEATURES.md"
RECORD = ROOT / ".agents/benchmark-record.md"

# A table cell longer than this is the wall-of-prose smell: forensic detail
# belongs in the record, not in a keyed-table cell. Shared by both pages.
MAX_CELL_CHARS = 220

# Likewise shared: a single paragraph past this is narrative, not a caption.
MAX_PARAGRAPH_CHARS = 700

# The scoreboard must point at the record, and the record must exist, otherwise
# "move it to the archive" silently loses the evidence.
RECORD_LINK = ".agents/benchmark-record.md"


class PageRules:
    """The budget for one public keyed-table page.

    Every limit is set from what the page measured when it was written, with
    headroom for genuinely new subjects but not for accumulated entries. That
    is the whole mechanism: growth that is new ROWS passes, growth that is new
    SECTIONS or new PROSE fails.
    """

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        required_sections: tuple[tuple[str, tuple[str, ...]], ...],
        max_h2_sections: int,
        max_prose_paragraphs: int,
        max_chars: int,
        min_table_rows: int,
        required_links: tuple[str, ...],
        canonical_sections: tuple[str, ...] | None = None,
    ) -> None:
        self.name = name
        self.kind = kind
        self.required_sections = required_sections
        self.max_h2_sections = max_h2_sections
        self.max_prose_paragraphs = max_prose_paragraphs
        self.max_chars = max_chars
        self.min_table_rows = min_table_rows
        self.required_links = required_links
        # When set, EVERY H2 must match one of these substrings. This is the
        # sharp form of the append-log rule: the FIRST appended section fails,
        # rather than the seventh once a count budget fills up. Left None for a
        # page where new sections are a normal, legitimate way to grow.
        self.canonical_sections = canonical_sections

    def is_canonical(self, title: str) -> bool:
        if self.canonical_sections is None:
            return True
        lowered = title.lower()
        return any(matcher in lowered for matcher in self.canonical_sections)


# docs/BENCHMARKS.md: 10 sections, 24 prose paragraphs, 12,078 chars, 68 rows
# when it was converted from an 11,405-line append log on 2026-08-04. Required
# sections are the four things a reader arriving from the README badge needs:
# the summary, the method behind the numbers, what is NOT measured, and how to
# re-run it.
BENCHMARKS_RULES = PageRules(
    name="docs/BENCHMARKS.md",
    kind="scoreboard",
    required_sections=(
        ("At a glance", ("at a glance", "summary", "scoreboard")),
        ("How we measure", ("how we measure", "methodology", "method")),
        ("Open gaps", ("open gaps", "not measured", "pending")),
        ("Reproduce", ("reproduce", "reproduction")),
    ),
    max_h2_sections=16,
    max_prose_paragraphs=35,
    max_chars=45000,
    min_table_rows=40,
    required_links=(RECORD_LINK,),
    # The scoreboard's sections ARE its schema: a reference engine, a resource
    # axis, or one of the four required reader sections. Anything else is a
    # per-attempt entry. Adding a genuinely new comparison subject means adding
    # one line here, which is a deliberate act; appending an entry is not.
    canonical_sections=(
        "at a glance",
        "vllm",
        "llama.cpp",
        "mlx",
        "dwarfstar",
        "memory",
        "speculative decoding",
        "how we measure",
        "open gaps",
        "reproduce",
    ),
)

# docs/FEATURES.md: 13 sections, 10 prose paragraphs, 8,275 chars, 104 rows when
# it was written on 2026-08-04. It must keep pointing at the two surfaces that
# carry the detail it deliberately does not: the lifecycle caveats behind each
# mark, and the measured speed behind each supported row.
FEATURES_RULES = PageRules(
    name="docs/FEATURES.md",
    kind="feature matrix",
    required_sections=(
        ("At a glance", ("at a glance", "summary", "overview")),
        ("Not supported yet", ("not supported", "gaps", "limitations")),
        ("How to read this page", ("how to read", "legend", "reading")),
    ),
    max_h2_sections=20,
    max_prose_paragraphs=20,
    max_chars=30000,
    min_table_rows=60,
    required_links=("STATUS.md", "BENCHMARKS.md"),
)


def _h2_headers(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell) <= set("-: ") for cell in cells)


def _prose_paragraphs(text: str) -> list[tuple[int, str]]:
    """Yield (start_line, paragraph) for prose only.

    Fenced code blocks, tables, headings, and list items are excluded: the rule
    targets the narrative paragraph, not legitimate tables or code samples.
    """
    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append((start, " ".join(current)))
            current = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        is_prose = bool(stripped) and not (
            stripped.startswith("|")
            or stripped.startswith("#")
            or stripped.startswith("-")
            or stripped.startswith("*")
            or stripped.startswith(">")
        )
        if is_prose:
            if not current:
                start = lineno
            current.append(stripped)
        else:
            flush()
    flush()
    return paragraphs


def _table_rows(text: str) -> list[tuple[int, list[str]]]:
    """Yield (line_number, cells) for every non-separator table row."""
    rows: list[tuple[int, list[str]]] = []
    in_fence = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not _is_separator_row(cells):
                rows.append((lineno, cells))
    return rows


def page_errors(text: str, rules: PageRules) -> list[str]:
    """Return human-readable problems with one keyed-table page."""
    errors: list[str] = []

    headers = _h2_headers(text)
    headers_lower = [h.lower() for h in headers]
    for label, matchers in rules.required_sections:
        if not any(any(m in h for m in matchers) for h in headers_lower):
            errors.append(f"missing required user-facing section: {label}")

    non_canonical = [h for h in headers if not rules.is_canonical(h)]
    if non_canonical:
        listed = "; ".join(non_canonical[:3])
        if len(non_canonical) > 3:
            listed += f"; ... (+{len(non_canonical) - 3})"
        errors.append(
            f"{rules.name} has {len(non_canonical)} non-canonical H2 "
            f"section(s) ({listed}); it is a KEYED TABLE, so a checkpoint "
            "updates its ROW, it does not append a section. Run "
            "scripts/roll-benchmark-record.py --apply to move them verbatim "
            f"into {RECORD_LINK}, or add a genuinely new comparison subject to "
            "CANONICAL_SECTIONS deliberately"
        )

    if len(headers) > rules.max_h2_sections:
        errors.append(
            f"{rules.name} has {len(headers)} H2 sections, over the "
            f"{rules.max_h2_sections}-section budget; it is a KEYED TABLE, so "
            "a checkpoint updates its ROW, it does not append a section"
        )

    if "—" in text:  # em-dash
        count = text.count("—")
        errors.append(
            f"{rules.name} contains {count} em-dash(es); house style forbids "
            "them (use commas, periods, parentheses, or hyphens)"
        )

    if len(text) > rules.max_chars:
        errors.append(
            f"{rules.name} is {len(text)} chars, over the {rules.max_chars}-char "
            f"{rules.kind} budget; per-attempt detail belongs in the record and "
            "the lifecycle surfaces, not here"
        )

    for link in rules.required_links:
        if link not in text:
            errors.append(
                f"{rules.name} does not link to {link}; the {rules.kind} must "
                "point at the surface carrying the detail it omits, so nothing "
                "it drops becomes unreachable"
            )

    paragraphs = _prose_paragraphs(text)
    if len(paragraphs) > rules.max_prose_paragraphs:
        errors.append(
            f"{rules.name} has {len(paragraphs)} prose paragraphs, over the "
            f"{rules.max_prose_paragraphs} budget; content belongs in table "
            "ROWS and prose only explains them"
        )
    for lineno, para in paragraphs:
        if len(para) > MAX_PARAGRAPH_CHARS:
            errors.append(
                f"line {lineno}: prose paragraph of {len(para)} chars exceeds "
                f"{MAX_PARAGRAPH_CHARS} (wall-of-prose smell; put the result "
                f"in a table row and the detail in {RECORD_LINK})"
            )

    rows = _table_rows(text)
    if len(rows) < rules.min_table_rows:
        errors.append(
            f"{rules.name} has {len(rows)} table rows, under the "
            f"{rules.min_table_rows} minimum; the {rules.kind} carries its "
            "content as keyed table rows, not as prose"
        )
    for lineno, cells in rows:
        for cell in cells:
            if len(cell) > MAX_CELL_CHARS:
                errors.append(
                    f"line {lineno}: table cell of {len(cell)} chars exceeds "
                    f"{MAX_CELL_CHARS} (wall-of-prose smell; move the detail "
                    f"to {RECORD_LINK})"
                )
    return errors


def benchmarks_errors(text: str) -> list[str]:
    """Return problems with docs/BENCHMARKS.md, the measured scoreboard."""
    return page_errors(text, BENCHMARKS_RULES)


def features_errors(text: str) -> list[str]:
    """Return problems with docs/FEATURES.md, the feature comparison matrix."""
    return page_errors(text, FEATURES_RULES)


# docs/STATUS.md is the per-capability ledger. It is legitimately larger than a
# scoreboard, and it is ALSO the page BENCHMARKS.md's forensics were told to
# move to, so a hard budget today would either fail on landing or push detail
# somewhere worse. But it is currently in the exact shape BENCHMARKS.md was in
# before its conversion (291k chars, 91 paragraphs over the prose budget, 47
# oversized cells, one cell of 16,181 chars), and it is the only public surface
# with no size gate at all - which is precisely how BENCHMARKS.md reached 11,127
# lines unnoticed.
#
# So this is a RATCHET, not a budget: every limit is pinned to what the page
# measured on 2026-08-04 and may only go DOWN. New rows and tighter prose pass;
# growth fails. The same mechanism as the device-leakage DSR ratchet in CI.
# Lowering these numbers as the page is compacted is the gate closing.
STATUS = ROOT / "docs/STATUS.md"
STATUS_RATCHET = {
    # 283470 since 2026-08-07 (measured 283457): the `OpenAI server` row's cell
    # was 1223 chars of endpoint prose, the single largest wall-of-prose cell on
    # the page and exactly what MAX_CELL_CHARS targets. Collapsed to the binding
    # result (which endpoints exist, which are flag-gated, which lack live
    # backing) with the per-endpoint flag narrative kept in docs/USAGE.md and
    # .agents/engine-matrix.md, which is what paid for the `/v1/videos` line
    # this row owes the page. Set a few chars above the measurement rather than
    # exactly on it: a ratchet pinned to the byte turns every concurrently
    # merged row's one-line status edit into a spurious failure. Still strictly
    # DOWN from 284062, the only direction this number may move.
    #
    # 283455 since 2026-08-07 (measured 283433): the MiniMax-H3 row had to carry
    # a new claim (the ORIGINAL bf16 13-shard DiT release is now indexable), and
    # it was paid for inside the same cell rather than out of the page - the
    # ref2va activation-diff narrative collapsed to its binding result, with the
    # full guilty-class audit kept in .agents/specs/minimax-h3.md 8.12. Net -6.
    #
    # 281700 since 2026-08-07 (measured 281668): the wide-x86 elementwise GEMM
    # tiers owe this page a status line, and it was paid for OUT of the page.
    # The DeepSeek-V4 last-mile Bricks 0 and 1 paragraphs were duplicated here
    # in an abridged form; the fuller originals already live in
    # .agents/benchmark-record.md, so the duplicates collapsed to their binding
    # result plus a pointer. Nothing was lost, only de-duplicated. Net -1755.
    #
    # 279200 since 2026-08-07 (measured 279167): the transpose-free [K,N] row
    # owes this page a status line, paid for the same way and out of the same
    # duplication. Bricks 1b and 2 folded into the same one-line binding result
    # as Bricks 0 and 1; the fuller originals stay in
    # .agents/benchmark-record.md. Net -2501.
    #
    # 283446 since 2026-08-07 (measured 283446): the Vulkan row swapped a
    # SUPERSEDED number for two shipped ones. "71 on CPU tier" was a
    # registry-coverage count being read as a hot-path one - a profile showed
    # exactly ONE reference-tier op fires per run (kRopeCosSinCache, at setup) -
    # so it went out and "argmax 18.9x" came in. The per-shader time profile that
    # replaces it, and the 55%-of-GPU-time GEMM finding it points at, are detail
    # and live in .agents/benchmark-record.md rather than on this page. Net -9.

    #
    #
    # 283442 since 2026-08-07 (measured 283442): the Vulkan row now carries the GEMV
    # tactic (1.8x) and drops the coopmat microbenchmark, which is a COMPONENT
    # number and lives in .agents/benchmark-record.md. The llama.cpp gap stayed
    # on the page deliberately - it was the first thing that fit when trimming,
    # and dropping the unflattering number to satisfy a size limit is how a
    # status page starts reading better than the code. Net -4.

    #

    #
    # 283442 since 2026-08-07 (measured 283442): the Vulkan row records the VOID
    # subgroup-reduction outcome, paid for by dropping the native-kernel COUNT,
    # which is derivable from the linked campaign doc while a void result is not
    # derivable from anywhere. Net -0.
    #
    # 279150 since 2026-08-08 (measured 279111 after the #122 rebase): the ONE-SURFACE ROW 8 device
    # knob owes this page a status line, paid for by removing a STALE MERGE
    # DUPLICATE in the metrics paragraph (the same "matching vLLM's own
    # mapping ... behavioural CPU gate ... remaining work" narrative appeared
    # twice back to back; the longer, newer version stays). Nothing was lost,
    # only de-duplicated. Net -92.
    #
    # 279130 since 2026-08-08 (measured 279120 after the #145/#151 rebases): the ROCm
    # entry in "Not supported yet" swapped "HIP never compiled" (falsified by
    # the four #41 community build reports) for the current binding state — W0
    # community-green, approach-(b) F6 fix in blind-unverified, M2 on
    # verification — paid for inside the same parenthetical; the board/arch
    # detail lives in docs/ROCM.md and .agents/backend-matrix.md. Net -12.
    #
    # 276960 since 2026-08-08 (measured 276945): the container-image lane
    # (`ENG-RELEASE-CONTAINERS`, issue #170) owes this page a status line, paid
    # for OUT of the page. The Tier-A1 fold paragraph was a run-by-run log of a
    # 2026-07-30 branch - down to a literal `commit <this>, NOT pushed` - on a
    # page whose contract is ONE binding current-state line per capability. It
    # collapsed to its binding result plus pointers, with the run detail already
    # present in .agents/benchmark-record.md, .agents/state.md and
    # specs/arch-fusion-fold-plan-2026-07-30.md. The unflattering parts were kept
    # ON the page deliberately: only OLMo-2 has a committed golden, Granite and
    # StableLM skip rather than gate, dflash and deepseek_v2 are build-verified
    # only. Nothing was lost, only de-duplicated. Net -2185.
    # 244486 since 2026-08-09 (measured 244486): the Laguna-S-2.1 MoE row was a
    # SINGLE 33,211-char table row - an 18,215-char accumulated benchmark history
    # and a 14,941-char implementation narrative - on a page whose contract is ONE
    # binding current-state line per capability, and against a 220-char cell bound
    # it was 150x over. It had also become the reason this page could not satisfy
    # its own shrink-only ratchet: main sat 253 chars over with no other block
    # large enough to pay for anything. Both cells were MOVED VERBATIM, the
    # benchmark half to .agents/benchmark-record.md and the implementation half to
    # .agents/state.md, leaving the binding result (87% of vLLM, root cause, the
    # device-resident fix, default-ON) and the architecture summary ON the page
    # with pointers. Nothing was rewritten, condensed or dropped. Net -32728, and
    # the ratchet is tightened to the measured value in the same change so the
    # headroom cannot be silently re-spent. oversized_cells 47 -> 44 and
    # long_paragraphs 89 -> 82 fall out of the same move.
    #
    # 244126 since 2026-08-09 (measured 244101): the release-binary manifest
    # milestone updated the OpenAI server status cell. Its endpoint-by-endpoint
    # prose already lives in docs/USAGE.md, so the cell now keeps only the
    # capability groups and binding limitations. Net -366 from rebased main.
    #
    # 244015 since 2026-08-09 (measured 244015): #189 moved the server TU into
    # the shared layer, which made "All six record checkers now green on main"
    # false -- check-device-leakage went 32 -> 37 on 5 VT_BENCH_PROFILE_CONTROL
    # guards. Correcting that claim costs chars, so it is paid for by collapsing
    # the env-doc hygiene sentence beside it to its binding result: it spelled
    # out which of two env vars went to ENVIRONMENT.md and which to the
    # allowlist, and both files state that authoritatively. Net -111, and
    # re-pinned byte-tight to match the convention this ratchet was set with
    # rather than quietly keeping the slack as headroom.
    #
    # 243761 since 2026-08-09 (measured 243756): rebasing the structured-state
    # compaction onto current main reduced the live page below the inherited
    # cap. The cap remains byte-tight and within the <=25 convention, so that
    # reduction cannot become untracked growth headroom.
    # 243694 since 2026-08-09 (measured 243694): the W6 claim checkpoint
    # replaces the longer container-only paragraph with the current binary and
    # container dependency state. Re-pinned byte-tight after the structured
    # state migration so the reduction cannot become untracked growth headroom.
    # 243632 since 2026-08-09 (measured 243632): the W1 candidate records local
    # gencode gates while keeping the real ten-SM archive audit pending.
    #
    # 243600 since 2026-08-09 (measured 243600): the NVFP4 re-verification
    # rewrites the 27B, 35B and Kimi-Linear rows onto their pinned revisions and
    # their re-measured ratios, and drops the superseded narrative those rows
    # carried. Net -32 against the rebased page, re-pinned byte-tight so the
    # reduction cannot become untracked growth headroom.
    #
    # 243590 since 2026-08-09 (measured 243590): `sm_110` is removed from the
    # BUILD-supported, portable-kernels-only list it never belonged in — the
    # page already states two paragraphs earlier that it is RUNTIME-VERIFIED on
    # a Jetson Thor board, and the contradiction cost a contributor time
    # (issue #168). Net -10, re-pinned byte-tight so the reduction cannot become
    # untracked growth headroom.
    #
    # 243588 since 2026-08-09 (measured 243588): BACKEND-TENSTORRENT reaches W2
    # and needs a status line; its per-op detail stays in backend-matrix.md and
    # the spec, so the page carries the binding result plus a pointer. Paid for
    # by collapsing superseded narrative in the GGUF row (a "REPORTED, not
    # gated" preamble, a "superseding the gap recorded here" aside, and three
    # restatements). Every measured number and binding claim kept verbatim, and
    # re-pinned byte-tight after rebasing onto the newer page.
    #
    # 243584 since 2026-08-09 (measured 243584): #219 makes VLLM_CPP_TRITON a
    # computed default, which the page owes one binding paragraph. It is paid
    # for in the same change by collapsing the two paragraphs the flip
    # supersedes -- the 2026-07-28 per-arch vendoring note, whose operative
    # claim ("the cross-family arch builds ship -DVLLM_CPP_TRITON=OFF") the new
    # default makes false, and the 27B SACRED-gate note, which spelled out a
    # flag the build now sets for you. Both keep their binding result, their
    # measured numbers and their links; only the superseded framing goes. Net
    # -4 against the Tenstorrent page, re-pinned byte-tight so the reduction
    # cannot become untracked growth headroom.
    # 243578 since 2026-08-10 (measured 243578): the developer ruled Triton
    # default-ON for EVERY CUDA arch, which the page owes a disclosure -- the
    # default now enables five never-runtime-verified cubin trees, one of which
    # (sm_80) has open #193. Paid for inside the same block it belongs to: the
    # per-arch vendoring note and the default note were collapsed to their
    # binding results and the RISK split out beside them, so the new claim costs
    # the page nothing. Net -6, re-pinned byte-tight.
    #
    # 2026-08-10 (SPEC-DSPARK): the DSpark paragraph moves to the measured state
    # (works on the 35B gate model, spec-on output token-identical to spec-off,
    # 1.15x warm against upstream's 1.41x). Re-pinned byte-tight below after the
    # merge, so the reduction cannot become untracked growth headroom.
    # 243554 since 2026-08-10 (measured 243554): row PERF-27B-LMHEAD-FP4 adds
    # the packed NVFP4 lm_head to the 27B cell. Paid for by collapsing that
    # cell's superseded narrative (the ModelOpt-FP8-tower aside and the
    # ~100% GPU-busy reading) into the binding result, both of which are now
    # carried durably by docs/BENCHMARKS.md's canonical rows. Strictly DOWN
    # from 243556, the only direction this number may move.
    #
    # 243512 since 2026-08-10 (measured 243512, #277): the metrics paragraph and
    # the OpenAI-server row both carried a claim that had become FALSE -- that
    # the async serving path has no live metric backing. Wiring it (the AsyncLLM
    # output handler now folds each step's stats into the logger) RETIRES a
    # residual rather than adding one, so the correction is a net deletion:
    # "metrics and cache reset lack live async backing" loses its first subject,
    # and "the async production-serving path wiring" leaves the remaining-work
    # list. Net -30 against whatever main's concurrent re-pins leave (243554 at
    # each rebase); re-pinned byte-tight to the merged measurement.
    #
    # 243455 since 2026-08-10 (measured, #213): the 35B row folded the
    # superseded VT_ASYNC_EXECUTOR Option A negative A/B into the ledger and
    # collapsed the mid-band narrative to its binding result. Strictly DOWN.
    # 243431 since 2026-08-11 (measured 243431): the #213 post-lever binding
    # numbers replace the pre-lever narrative in the 27B cell. Strictly DOWN
    # from 243451.
    #
    # 243399 since 2026-08-10 (measured, #213): the gemv build-verify row folded a
    # superseded allowlist clause into the binding result. Strictly DOWN.
    #
    # 243389 since 2026-08-10 (measured, #223): the Sampling row gains prompt
    # logprobs computed on the runner; paid for by collapsing the beam-search
    # restatements that row carried. Strictly DOWN.
    #
    # 243377 since 2026-08-11 (measured 243377, #238): the logprobs_mode row needs
    # one binding line -- three of four modes were runtime-refused stubs and now
    # work. Paid for by collapsing the best_of cell's upstream RATIONALE ("vLLM
    # 0.26 itself has dropped best_of from its live path..."), which is a why, not
    # a current state, and belongs in the row's spec. A DIFFERENT collapse from
    # #223's beam-search one directly above: both collapses and both additions are
    # in the merged page, which is why this pin is RE-MEASURED against it rather
    # than carried from either PR (#223 measured 243389, #238 measured 243559 --
    # both stale the moment the other landed). Every measured number and binding
    # claim kept verbatim. Strictly DOWN.
    #
    # 243356 since 2026-08-11 (measured 243356): main re-pinned to 243378 for the
    # #323 async-serving correction while #223 and #238 were landing. RE-MEASURED
    # against the page that carries all three -- 243378 less #223's 10 and #238's
    # 12 -- rather than carried from any one of them. Strictly DOWN.
    # 243309 since 2026-08-11 (measured 243309): the 35B cell carries the
    # canonical c1-c32 grid instead of the superseded ad-hoc narrative.
    # Strictly DOWN from 243378.
    #
    # 243285 since 2026-08-11 (measured 243285): the 27B row collapsed to the binding
    # canonical 6-point result; the superseded 0.838 c1 and the two-grid
    # disagreement moved to .agents/benchmark-record.md (#349). Prior note:
    # 243287 since 2026-08-11: main re-pinned to 243309 for the
    # 35B canonical grid while #223 and #238 were landing. RE-MEASURED against the
    # page carrying all three -- 243309 less #223's 10 and #238's 12 -- not carried
    # from any one of them. Strictly DOWN.
    "chars": 243285,
    "h2_sections": 11,
    "long_paragraphs": 82,
    "oversized_cells": 44,
}
STATUS_REQUIRED = (
    ("Parity pin", ("parity pin",)),
    ("Capability status", ("capability status",)),
    ("Not supported yet", ("not supported yet", "not yet supported")),
)


def status_errors(text: str) -> list[str]:
    """Return ratchet violations for docs/STATUS.md, the per-capability ledger."""
    errors: list[str] = []

    headers = _h2_headers(text)
    lowered = [h.lower() for h in headers]
    for label, matchers in STATUS_REQUIRED:
        if not any(any(m in h for m in matchers) for h in lowered):
            errors.append(
                f"docs/STATUS.md is missing its '{label}' section; it is the "
                "per-capability status surface every checkpoint updates"
            )

    measured = {
        "chars": len(text),
        "h2_sections": len(headers),
        "long_paragraphs": sum(
            1 for _, para in _prose_paragraphs(text) if len(para) > MAX_PARAGRAPH_CHARS
        ),
        "oversized_cells": sum(
            1
            for _, cells in _table_rows(text)
            for cell in cells
            if len(cell) > MAX_CELL_CHARS
        ),
    }
    for key, cap in STATUS_RATCHET.items():
        if measured[key] > cap:
            errors.append(
                f"docs/STATUS.md {key.replace('_', ' ')} is {measured[key]}, over "
                f"the {cap} ratchet: this page may only shrink. Collapse the "
                "superseded narrative into the binding result and move the "
                f"detail to {RECORD_LINK} or structured state evidence, then lower the "
                "ratchet in the same change"
            )

    return errors


def record_errors(path: Path) -> list[str]:
    """Return problems with the append-only benchmark record."""
    if not path.exists():
        return [
            f"{RECORD_LINK} is missing; it is the append-only record the "
            "scoreboard relocates forensics into, and losing it loses the "
            "evidence"
        ]
    return []


def main() -> int:
    errors: list[str] = []
    for path, check in (
        (BENCHMARKS, benchmarks_errors),
        (FEATURES, features_errors),
        (STATUS, status_errors),
    ):
        if not path.exists():
            errors.append(f"{path.relative_to(ROOT)} is missing")
            continue
        errors += check(path.read_text(encoding="utf-8"))
    errors += record_errors(RECORD)
    if errors:
        print(
            "ERROR: the public keyed-table docs are not valid:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(
        "OK: docs/BENCHMARKS.md and docs/FEATURES.md are human-readable keyed "
        f"tables, docs/STATUS.md is inside its size ratchet, and {RECORD_LINK} "
        "carries the append-only record."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
