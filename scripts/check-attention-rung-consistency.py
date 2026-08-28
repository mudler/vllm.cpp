#!/usr/bin/env python3
r"""Fail if a model names the NAIVE attention kernel without saying why.

`vt::Attention` resolves `OpId::kAttention` (`src/vt/ops.cpp`) straight to
`AttentionKernel` (`src/vt/cuda/cuda_ops.cu`), self-described there as
"Correctness-grade (M0.9)": one 256-thread block per (query, head), a 256-wide
shared-memory tree reduction for EVERY key, no K/V tiling, K and V re-read from
global once per (query, head). Which kernel a model gets is decided by the C++
function name its author typed, and nothing ever routes `kAttention` up. A model
whose author never heard of `vt::AttentionDenseFlash` / `vt::AttentionDenseFast` /
`vt::AttentionDenseFa2` therefore pays up to ~500x with correct output, no warning
and no gate (issue #1544; measured at 47.84 s for one LTX-2.5 DiT forward).

The freeze is NOT the defect. `kAttention` is frozen so text decode stays
byte-identical, and six model sites use the naive kernel deliberately: as the
reference arm of a numeric gate, or as the `VT_*_EAGER` rung of a same-binary A/B.
The defect is that nothing distinguishes those six from an author who simply did
not know. So this checker does not reroute anything and cannot: it requires the
CHOICE to be recorded next to the call.

The record is a marker comment on the call line or in the 20 lines above it:

    // VT-ATTN-NAIVE: reference arm of the paged/dense equivalence gate; the fast
    // rungs are not bit-identical to this one, so rerouting deletes the golden.
    vt::Attention(q, out, qq, kk, vv, args);

`whisper_audio.cpp` and `qwen3_vl_vision.cpp` are the intended pattern (#1544):
both DEFAULT to a fast rung and expose the naive one behind an env knob.

Why a marker and not one shared registry: AGENTS.md forbids a record surface every
pull request must write. The marker lives in the file that owns the call, so the
ordinary case touches no shared file. `scripts/attention-rung-allowlist.txt`
carries only stems whose naive call is being REMOVED by a row already in flight —
editing the very lines those changes replace would conflict for no gain. An
allowlisted stem whose sites are all marked or gone is reported as STALE and does
NOT fail HERE, so the CHECKER never forces the row that cleans it up to edit this
file. A test does: the expected stem set is pinned in
tests/scripts/test_check_attention_rung_consistency.py, which the allowlist's own
header says, so growth of that file stays a review decision.

Text the compiler never sees is not a call site: the scan runs over
`scripts/checker_text.py::normalize_source`, so a commented-out, `#if 0`-ed or
`if (false)`-ed `vt::Attention(` is a deletion here, exactly as it is to nvcc.
That normalization is position-preserving, so every reported `file:line` still
describes the original file.

What this checker DETECTS, stated as a limit rather than implied by a green:
one literal spelling, `vt::Attention(`, anywhere under `src/`, `include/` or
`examples/`. Four spellings reach the same kernel and are NOT detected, each
verified to leave the checker green with a live unmarked call:

    using vt::Attention;   then a bare  Attention(...)
    namespace vv = vt;     then         vv::Attention(...)
    #define MUT_ATTN vt::Attention
    auto* fn = &vt::Attention;          then a call through `fn`

None exists in this tree, and the repository does not write attention calls that
way, so this is a stated bound and not a live hole. Widening the regex is not the
repair: `\bAttention\s*\(` also matches every fast rung's suffix-free form and
would demand a marker beside exactly the calls this checker wants people to make,
and no regex reaches a call through a function pointer at all. What closes it is a
compiler-side population — the CUDA op registry, or a clang tooling pass over the
real translation unit — which is a different instrument, not a longer pattern. A
green here therefore means "no unmarked `vt::Attention(` call in the scanned
population", never "no model is on the naive rung".

THE POPULATION IS THE COMPILED TREE, and it was not always (#1552). This scan
read `src/vllm/model_executor/models` and its include sibling, NON-recursively,
over `*.cpp` and `*.h` only. Three shapes escaped it, and two were measured on
`f9af269f9` by appending a live unmarked call and running this file: an unmarked
`vt::Attention(` in `src/vllm/v1/attention/backend.cpp`, and one in a new
subdirectory of the model directory, each left rc=0 with the OK line still
reporting the same 8 sites. Neither shape is exotic — `src/vllm/multimodal/`
already drives the LTX-2.5 denoise loop from outside `models/`, and a model that
grows a kernels subdirectory takes its call out of the population by moving a
file. So the roots are now `src/`, `include/` and `examples/`, walked
recursively over every C++ suffix this repository compiles.

`tests/` is excluded BY NAME, at any depth. The suite for this checker writes
unmarked `vt::Attention(` calls as fixtures, and a checker that refuses its own
tests cannot be run. `tests/scripts/test_check_attention_rung_consistency.py`
::PopulationTests asserts the exclusion together with a sibling that IS scanned,
because an exclusion asserted alone also passes on a scanner that reads nothing.

`MODEL_DIRS` survives the widening and still means what it meant: it is how the
allowlist resolves a model STEM to a source file, which is a model-scoped
question and not the scan population.

The validation logic is pure functions (`scan_file`, `drift_sites`,
`stale_allowlist_entries`) so it is unit- and mutation-testable
(tests/scripts/test_check_attention_rung_consistency.py), mirroring
check-fusion-consistency.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from checker_text import normalize_source  # noqa: E402

# Both halves of a model's sources. NOT the scan population any more (#1552, see
# the module docstring): this pair is how `stale_allowlist_entries`' companion
# test resolves an allowlisted STEM to a real source file, which is a
# model-scoped question. Widening it would change which stems are resolvable,
# not which files are read.
MODEL_DIRS = (
    ROOT / "src/vllm/model_executor/models",
    ROOT / "include/vllm/model_executor/models",
)

# THE SCAN POPULATION. Every root the compiler sees, walked recursively. A header
# is scanned for the same reason it always was — a call moved into an inline
# function or a template would otherwise leave the checker green — and a `.cu` is
# scanned because the naive/fast distinction this file guards lives in one.
SCAN_ROOTS = (
    ROOT / "src",
    ROOT / "include",
    ROOT / "examples",
)
SOURCE_SUFFIXES = (".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp")

# Excluded at any depth, by directory NAME. `tests/` writes unmarked calls as
# fixtures on purpose; so does this checker's own suite. A checker that refuses
# its own tests cannot be run, and that is the whole of the exclusion — it is not
# a general escape hatch, and adding a name here removes a directory from every
# green this file ever prints.
EXCLUDED_DIR_NAMES = frozenset({"tests"})

ALLOWLIST = ROOT / "scripts/attention-rung-allowlist.txt"

# `vt::Attention(` and nothing else. The word boundary is load-bearing: without it
# this matches `vt::AttentionDenseFlash(`, `vt::AttentionDenseFast(`,
# `vt::AttentionDenseFa2(` and `vt::AttentionCross(` — every FAST rung — and the
# checker would demand a marker beside exactly the calls it wants people to make.
_NAIVE_CALL = re.compile(r"\bvt::Attention\s*\(")

# The marker, in a `//` comment. A reason is required after the colon.
_MARKER = re.compile(r"//.*\bVT-ATTN-NAIVE:\s*(\S.*?)\s*$")

# How far above a call the marker may sit. A call reached through several lines of
# tensor-view setup still reads as one statement to a human, and forcing the
# marker onto the call line itself would push it past any sane column.
MARKER_WINDOW_LINES = 20

# A reason must be long enough to BE one. This is a floor against `// VT-ATTN-NAIVE: x`
# and nothing more; the checker cannot judge whether a reason is true, and does not
# try. That is a reviewer's job, exactly as it is for the reasons on
# scripts/fusion-consistency-allowlist.txt.
MIN_REASON_CHARS = 16


def marker_reason(line: str) -> str | None:
    """The reason recorded by a marker comment on this line, or None."""
    match = _MARKER.search(line)
    return match.group(1) if match is not None else None


def has_marker(lines: list[str], call_line: int) -> bool:
    """True if a marker with a substantive reason covers the call at `call_line`.

    `call_line` is 1-based. The window is the call line itself and the
    MARKER_WINDOW_LINES lines above it.
    """
    first = max(1, call_line - MARKER_WINDOW_LINES)
    for number in range(first, call_line + 1):
        reason = marker_reason(lines[number - 1])
        if reason is not None and len(reason) >= MIN_REASON_CHARS:
            return True
    return False


def scan_file(text: str) -> list[tuple[int, bool]]:
    """Every live `vt::Attention(` site in one translation unit.

    Returns (1-based line number, marker_present) per site, in source order.
    """
    live = normalize_source(text)
    raw_lines = text.splitlines()
    out: list[tuple[int, bool]] = []
    for match in _NAIVE_CALL.finditer(live):
        line = live.count("\n", 0, match.start()) + 1
        out.append((line, has_marker(raw_lines, line)))
    return out


def scan_models(roots=SCAN_ROOTS) -> dict[str, list[tuple[int, bool]]]:
    """Map repo-relative source path -> its `vt::Attention(` sites.

    Walks each root RECURSIVELY over SOURCE_SUFFIXES, skipping any path with an
    EXCLUDED_DIR_NAMES component. Before #1552 this globbed two directories
    non-recursively over two suffixes, and a call one directory to the side was
    invisible to it; the module docstring records the two shapes that were
    measured escaping.

    Keyed on the PATH, not the stem: `ltx2.cpp` and `ltx2.h` share a stem and would
    otherwise overwrite each other, reporting one file's sites under the other's
    name. The allowlist still matches on the stem, so one entry covers a model's
    whole translation unit.

    The key is repo-relative when the file is inside the repository and absolute
    when it is not, so a synthetic root under a temporary directory reports a
    usable path instead of raising. Every shipped root is inside ROOT, so nothing
    the gate reads takes the second branch.
    """
    out: dict[str, list[tuple[int, bool]]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            if EXCLUDED_DIR_NAMES.intersection(path.parts):
                continue
            sites = scan_file(path.read_text(encoding="utf-8", errors="ignore"))
            if sites:
                try:
                    key = str(path.relative_to(ROOT))
                except ValueError:
                    key = str(path)
                out[key] = sites
    return out


def allowlisted_names(text: str) -> set[str]:
    """Model stems accepted as in-flight (one per line, # comments ignored) —
    mirrors check-fusion-consistency.py."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def drift_sites(
    scanned: dict[str, list[tuple[int, bool]]], allowlisted: set[str]
) -> list[tuple[str, int]]:
    """(path, line) for every unmarked naive-attention call in a file whose stem is
    not allowlisted. Empty == the check passes."""
    out: list[tuple[str, int]] = []
    for path in sorted(scanned):
        if Path(path).stem in allowlisted:
            continue
        out.extend((path, line) for line, marked in scanned[path] if not marked)
    return out


def stale_allowlist_entries(
    scanned: dict[str, list[tuple[int, bool]]], allowlisted: set[str]
) -> list[str]:
    """Allowlisted stems that would pass on their own merit — their naive calls are
    gone or now carry a marker. Reported, never fatal: the row that removes the
    call must not be forced to edit this file to stay green."""
    return sorted(
        stem
        for stem in allowlisted
        if all(
            marked
            for path, sites in scanned.items()
            if Path(path).stem == stem
            for _, marked in sites
        )
    )


def main() -> int:
    scanned = scan_models()
    allowlisted = (
        allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        if ALLOWLIST.exists()
        else set()
    )
    drift = drift_sites(scanned, allowlisted)

    for stem in stale_allowlist_entries(scanned, allowlisted):
        print(
            f"STALE (not a failure): {stem} is on "
            "scripts/attention-rung-allowlist.txt but no longer has an unmarked "
            "vt::Attention call. Delete its entry."
        )

    if drift:
        print(
            # "call site(s)", not "model forward(s)": since #1552 the population
            # is `src/`, `include/` and `examples/`, so this message can now name
            # a file that is not a model forward at all.
            "ERROR: call site(s) name vt::Attention — the naive, "
            "correctness-grade attention kernel (up to ~500x the cost of "
            "vt::AttentionDenseFlash; issue #1544) — with no recorded reason:",
            file=sys.stderr,
        )
        for path, line in drift:
            print(f"  - {path}:{line}", file=sys.stderr)
        print(
            "If the fast rung is what you wanted, call vt::AttentionDenseFlash "
            "(shared-memory tiled), vt::AttentionDenseFast (warp-per-query, no "
            "shared memory) or vt::AttentionDenseFa2 (bf16 head_dim 64 or 128, "
            "non-causal, tensor cores). If the NAIVE kernel is what you meant — a "
            "reference arm, or the eager rung of a same-binary A/B — record that "
            "on the call line or within "
            f"{MARKER_WINDOW_LINES} lines above it:\n"
            "    // VT-ATTN-NAIVE: <why this call must stay on kAttention>\n"
            "See src/vllm/model_executor/models/whisper_audio.cpp and "
            "qwen3_vl_vision.cpp for the intended pattern.",
            file=sys.stderr,
        )
        return 1

    sites = sum(len(v) for v in scanned.values())
    marked = sum(1 for v in scanned.values() for _, m in v if m)
    # The number a reader of a green actually needs: sites that carry NO reason and
    # pass only because their stem is allowlisted. `sites - marked` is not it, because
    # a marked call inside an allowlisted file counts in `marked`. This is the debt the
    # green is hiding, so it is printed even when it is zero.
    excused = sum(
        1
        for path, sites_in_file in scanned.items()
        if Path(path).stem in allowlisted
        for _, m in sites_in_file
        if not m
    )
    print(
        f"OK (attention rung): {sites} vt::Attention call site(s) in "
        f"{len(scanned)} source file(s) under {len(SCAN_ROOTS)} scanned root(s); "
        f"{marked} carry a recorded reason, "
        f"{excused} unmarked and excused by "
        f"{len(allowlisted)} allowlisted in-flight stem(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
