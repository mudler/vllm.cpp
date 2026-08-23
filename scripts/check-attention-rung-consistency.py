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
one literal spelling, `vt::Attention(`, in a model `.cpp` or `.h`. Four spellings
reach the same kernel and are NOT detected, each verified to leave the checker
green with a live unmarked call:

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
green here therefore means "no unmarked `vt::Attention(` call", never "no model is
on the naive rung".

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

# Both halves of a model's sources. A header is scanned too, because a call moved
# into an inline function or a template there would otherwise leave the checker
# green — the population is what makes a green meaningful.
MODEL_DIRS = (
    ROOT / "src/vllm/model_executor/models",
    ROOT / "include/vllm/model_executor/models",
)
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


def scan_models(model_dirs=MODEL_DIRS) -> dict[str, list[tuple[int, bool]]]:
    """Map repo-relative model source path -> its `vt::Attention(` sites.

    Keyed on the PATH, not the stem: `ltx2.cpp` and `ltx2.h` share a stem and would
    otherwise overwrite each other, reporting one file's sites under the other's
    name. The allowlist still matches on the stem, so one entry covers a model's
    whole translation unit.
    """
    out: dict[str, list[tuple[int, bool]]] = {}
    for models_dir in model_dirs:
        if not models_dir.is_dir():
            continue
        for pattern in ("*.cpp", "*.h"):
            for path in sorted(models_dir.glob(pattern)):
                sites = scan_file(path.read_text(encoding="utf-8", errors="ignore"))
                if sites:
                    out[str(path.relative_to(ROOT))] = sites
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
            "ERROR: model forward(s) call vt::Attention — the naive, "
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
        f"{len(scanned)} model source file(s); {marked} carry a recorded reason, "
        f"{excused} unmarked and excused by "
        f"{len(allowlisted)} allowlisted in-flight stem(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
