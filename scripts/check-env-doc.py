#!/usr/bin/env python3
"""Fail if a production env var is undocumented, or documented and unread.

This gate runs in BOTH directions, because a one-directional gate lets a knob
outlive its implementation and says nothing:

FORWARD (`undocumented_env_vars`). Every `VT_*` / `VLLM_*` environment name read
from `src/` and `include/` must be either documented in `docs/ENVIRONMENT.md`
(the user-facing/behavior-changing knobs) or listed on
`scripts/env-doc-allowlist.txt` (the kernel-internal tuning tail). This stops a
new production env var from landing silently, the way the 153-variable blind
spot did before docs/ENVIRONMENT.md existed.

REVERSE (`unread_documented_vars`, #2389). Every variable in a user-facing TABLE
of `docs/ENVIRONMENT.md` must be read by at least one compiled file, as more
than a comment. Before this direction existed, deleting the last reader of a
documented knob left the doc row -- with its default, its formula and its tuning
advice -- promising behaviour the tree does not implement, and the failure was
invisible: the operator sets the variable, nothing happens, and nothing says
why. `VT_QWEN35_STAGE_MIN_FREE_FRAC` (reader deleted, comment-only hit) and
`VT_GEMMA4_MLP_MOE_PARALLEL` (never wired at all) are the two instances that
motivated it.

Three details decide whether the reverse direction reports the truth, and each
one produces a WRONG answer if it is skipped:

1. A COMMENT IS NOT A READ. `VT_QWEN35_STAGE_MIN_FREE_FRAC` occurs exactly once
   in compiled code, in a `//` comment in qwen3_5_weights.h, so a name grep
   passes it. Read sites are therefore harvested from comment-STRIPPED text and
   only as a quoted string literal, which is how an env name reaches any lookup.
2. A SHIPPED BINARY OUTSIDE `src/` STILL COUNTS. `VT_BENCH_PRETOKENIZE` is read
   in `examples/bench/bench_core.h`, which is the `vllm-bench` binary its doc
   row scopes it to. `READ_SITE_ROOTS` therefore adds `examples/` -- the third
   tree CMake compiles into shipped targets -- so a live knob is not called dead.
   It does NOT add `benchmarks/` or `tools/`, which no CMake target builds.
3. THE READ NEED NOT BE A LITERAL `getenv`. `VT_GGUF_KEEP_QUANT` is read through
   the `EnvOnOr("VT_GGUF_KEEP_QUANT", ...)` helper. Matching the quoted NAME
   rather than the calling function keeps every helper-mediated read visible.

`UNREAD_EXCEPTIONS` is the declared escape, and it is self-clearing: an entry
must carry a stated reason, and an entry whose variable has left the table or
gained a reader is reported as STALE. An exception is visible debt with an
expiry, not a permanent hole.

The validation logic is a set of pure functions so it is unit-testable and
mutation-testable (see tests/scripts/test_check_env_doc.py), mirroring
check-readme-structure.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("src", "include")
SCAN_SUFFIXES = (".cpp", ".cc", ".cu", ".cuh", ".c", ".h", ".hpp", ".hh", ".hip")
ENV_DOC = ROOT / "docs/ENVIRONMENT.md"
ALLOWLIST = ROOT / "scripts/env-doc-allowlist.txt"

# A production env var is read as a quoted string literal, e.g. getenv("VT_FOO").
# Match the quoted name so a bare mention in a comment is not counted as a read.
_QUOTED = re.compile(r'"((?:VT_|VLLM_)[A-Z0-9_]+)"')
# A name token, used to harvest the documented set from the markdown / allowlist.
_TOKEN = re.compile(r'\b((?:VT_|VLLM_)[A-Z0-9_]+)\b')

# The trees CMake compiles into shipped targets. `src/` and `include/` are the
# library; `examples/` is `add_subdirectory(examples)` under
# VLLM_CPP_BUILD_EXAMPLES and holds vllm-cli and vllm-bench. `benchmarks/` and
# `tools/` are deliberately absent: no CMake target builds them, so a knob read
# only there is read by nothing anybody can run.
READ_SITE_ROOTS = ("src", "include", "examples")

# A char literal, so `'` is not mistaken for the opening quote of one. Without
# this a C++14 digit separator (`1'000'000`) would swallow the rest of the line.
_CHAR_LITERAL = re.compile(r"'(?:\\.|[^\\'])'")

# Variables documented in a user-facing table that no compiled file reads, with
# the reason each one is tolerated. An entry is NOT a permanent hole: it must
# carry a non-empty reason, and `stale_unread_exceptions` reports it the moment
# its variable leaves the table or gains a reader, so the debt cannot outlive
# the condition that justified it.
UNREAD_EXCEPTIONS: dict[str, str] = {
    # EMPTY, and that is the point rather than an oversight. The one entry this
    # gate shipped with -- `VT_QWEN35_STAGE_MIN_FREE_FRAC` -- existed only so
    # this checker could land without editing a doc row another row owned. That
    # row is gone (#2385, landed), `stale_unread_exceptions` reported the entry
    # by name the moment it went, and the entry was deleted rather than
    # rewritten. The escape hatch working once, end to end, is the evidence it
    # is self-clearing.
}


def scan_env_names(root: Path) -> set[str]:
    """Return every VT_/VLLM_ env name read as a string literal in src/+include/."""
    names: set[str] = set()
    for rel in SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in SCAN_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            names.update(_QUOTED.findall(text))
    return names


def documented_names(text: str) -> set[str]:
    """Names mentioned in docs/ENVIRONMENT.md."""
    return set(_TOKEN.findall(text))


def allowlisted_names(text: str) -> set[str]:
    """Names on the kernel-internal allowlist (one per line, # comments ignored)."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def undocumented_env_vars(
    scanned: set[str], documented: set[str], allowlisted: set[str]
) -> list[str]:
    """The scanned names covered by neither surface. Empty == the check passes."""
    return sorted(scanned - documented - allowlisted)


def strip_comments(text: str) -> str:
    """Return `text` with C/C++ comments removed, string literals preserved.

    Complication 1 of #2389. A name that occurs only inside a comment is not a
    read, and a comment can quote the name it discusses -- `// getenv("VT_X")`
    is prose, not a lookup -- so the quoted-literal regex alone is not enough.
    String and char literals are walked through first, so a `//` or `/*` inside
    one is data and does not open a comment.
    """

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i : i + 2])
                    i += 2
                    continue
                out.append(text[i])
                i += 1
                if text[i - 1] == '"':
                    break
            continue
        if ch == "'":
            match = _CHAR_LITERAL.match(text, i)
            if match is not None:
                out.append(match.group(0))
                i = match.end()
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def scan_read_sites(root: Path) -> set[str]:
    """Every VT_/VLLM_ env name a compiled file READS, comments excluded.

    Reads across `READ_SITE_ROOTS` (complication 2), matches the quoted NAME
    rather than a `getenv` call so helper-mediated reads count (complication 3),
    and strips comments first so a comment-only mention does not (complication 1).
    """

    names: set[str] = set()
    for rel in READ_SITE_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in SCAN_SUFFIXES or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            names.update(_QUOTED.findall(strip_comments(text)))
    return names


def table_documented_names(text: str) -> set[str]:
    """Names in the FIRST column of a markdown table row in docs/ENVIRONMENT.md.

    The user-facing surface is the tables: a table row states a default and an
    effect, and that is the promise an operator acts on. Prose that merely names
    a variable is not held to the reverse rule, and neither is the deferred
    kernel-internal section, which names families rather than variables.
    """

    names: set[str] = set()
    for line in text.splitlines():
        row = line.strip()
        if not row.startswith("|"):
            continue
        first = row.strip("|").split("|")[0]
        names.update(_TOKEN.findall(first))
    return names


def unread_documented_vars(
    table: set[str], read: set[str], exceptions: dict[str, str]
) -> list[str]:
    """Documented table names that no compiled file reads. Empty == passes."""
    return sorted(table - read - set(exceptions))


def stale_unread_exceptions(
    table: set[str], read: set[str], exceptions: dict[str, str]
) -> list[str]:
    """Exceptions whose justification is gone: not in a table, or now read."""
    return sorted(name for name in exceptions if name not in table or name in read)


def unreasoned_unread_exceptions(exceptions: dict[str, str]) -> list[str]:
    """Exceptions carrying no stated reason. An unexplained hole is not allowed."""
    return sorted(name for name, reason in exceptions.items() if not reason.strip())


def main() -> int:
    if not ENV_DOC.exists():
        print("ERROR: docs/ENVIRONMENT.md is missing", file=sys.stderr)
        return 1
    doc_text = ENV_DOC.read_text(encoding="utf-8")
    scanned = scan_env_names(ROOT)
    documented = documented_names(doc_text)
    allowlisted = (
        allowlisted_names(ALLOWLIST.read_text(encoding="utf-8"))
        if ALLOWLIST.exists()
        else set()
    )
    failed = False

    missing = undocumented_env_vars(scanned, documented, allowlisted)
    if missing:
        failed = True
        print(
            "ERROR: production env var(s) read from src/+include/ are neither "
            "documented in docs/ENVIRONMENT.md nor on "
            "scripts/env-doc-allowlist.txt:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print(
            "Document it in docs/ENVIRONMENT.md (if it is a user-facing / "
            "behavior-changing knob) or add it to scripts/env-doc-allowlist.txt "
            "(if it is a kernel-internal tuning switch).",
            file=sys.stderr,
        )

    table = table_documented_names(doc_text)
    read = scan_read_sites(ROOT)

    unreasoned = unreasoned_unread_exceptions(UNREAD_EXCEPTIONS)
    if unreasoned:
        failed = True
        print(
            "ERROR: UNREAD_EXCEPTIONS entr(ies) in scripts/check-env-doc.py "
            "state no reason:",
            file=sys.stderr,
        )
        for name in unreasoned:
            print(f"  - {name}", file=sys.stderr)

    unread = unread_documented_vars(table, read, UNREAD_EXCEPTIONS)
    if unread:
        failed = True
        print(
            "ERROR: env var(s) documented in a user-facing table of "
            "docs/ENVIRONMENT.md are read by no compiled file under "
            f"{'/, '.join(READ_SITE_ROOTS)}/ (a comment is not a read):",
            file=sys.stderr,
        )
        for name in unread:
            print(f"  - {name}", file=sys.stderr)
        print(
            "The row promises behaviour the tree does not implement. Delete the "
            "row, restore the reader, or add the name to UNREAD_EXCEPTIONS in "
            "scripts/check-env-doc.py with the reason it is tolerated.",
            file=sys.stderr,
        )

    stale = stale_unread_exceptions(table, read, UNREAD_EXCEPTIONS)
    if stale:
        failed = True
        print(
            "ERROR: UNREAD_EXCEPTIONS entr(ies) in scripts/check-env-doc.py are "
            "stale -- the variable has left the user-facing table or gained a "
            "reader, so the exception no longer describes anything:",
            file=sys.stderr,
        )
        for name in stale:
            print(f"  - {name}", file=sys.stderr)
        print("Delete the entry.", file=sys.stderr)

    if failed:
        return 1
    print(
        f"OK: all {len(scanned)} production env vars are documented "
        "or classified kernel-internal."
    )
    print(
        f"OK: all {len(table)} env vars documented in a user-facing table are "
        f"read by compiled code ({len(UNREAD_EXCEPTIONS)} declared exception(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
