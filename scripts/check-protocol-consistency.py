#!/usr/bin/env python3
"""Keep the protocol prose and the checkers that enforce it in agreement.

The failure this exists to prevent is real and already happened: the
same-change public-document obligation was migrated from README.md to
docs/STATUS.md in scripts/check-doc-checkpoint.py, AGENTS.md was updated, and
`.agents/workflow.md` -- the operating manual an agent is told to follow every
session -- was not. For a while the manual instructed agents to update README.md
at every checkpoint, which is exactly the drift the migration removed. Prose and
checker disagreed, and the prose is what agents actually read.

So the obligated surfaces are declared ONCE, as a machine-readable contract
block that both documents carry verbatim, and this gate asserts the block equals
the constants in scripts/check-doc-checkpoint.py. Changing the checker without
changing the prose (or the reverse) is a red build, not a silent divergence.

The contract block looks like this, and is a normal Markdown table to a reader:

    <!-- doc-obligation-contract:begin -->
    | Public surface | Owed by |
    |---|---|
    | `docs/STATUS.md` | every feature/iteration checkpoint |
    <!-- doc-obligation-contract:end -->
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Both documents must carry the contract, because both are read as normative:
# AGENTS.md is the canonical index, workflow.md is the session operating manual.
CONTRACT_DOCUMENTS = ("AGENTS.md", ".agents/workflow.md")

BEGIN = "<!-- doc-obligation-contract:begin -->"
END = "<!-- doc-obligation-contract:end -->"

# A path in a table cell, e.g. `docs/STATUS.md`.
CELL_PATH = re.compile(r"`([^`]+\.md)`")

# README.md is a landing page, not a checkpoint surface. Naming it inside the
# contract is the specific regression this gate was built after, so it earns a
# targeted message instead of a bare set-difference.
FORBIDDEN_IN_CONTRACT = {
    "README.md": (
        "README.md is a user-facing landing page, not a per-checkpoint status "
        "surface; it changes only when a user-visible headline shifts. The "
        "per-capability obligation belongs to docs/STATUS.md"
    ),
}


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def obligated_surfaces() -> tuple[str, ...]:
    """Return the surfaces check-doc-checkpoint.py actually enforces."""
    checkpoint = _load("doc_checkpoint", "scripts/check-doc-checkpoint.py")
    return tuple(checkpoint.PUBLIC_CHECKPOINTS) + (checkpoint.FEATURE_CHECKPOINT,)


def contract_paths(text: str) -> list[str] | None:
    """Return the paths declared in the contract block, or None if absent."""
    start = text.find(BEGIN)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        return None
    block = text[start + len(BEGIN) : end]
    paths: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        found = CELL_PATH.findall(stripped)
        if found:
            paths.append(found[0])
    return paths


def document_errors(name: str, text: str, expected: tuple[str, ...]) -> list[str]:
    """Return contract problems for one normative document."""
    paths = contract_paths(text)
    if paths is None:
        return [
            f"{name} is missing the doc-obligation contract block "
            f"({BEGIN} ... {END}); it must declare the surfaces that "
            "scripts/check-doc-checkpoint.py enforces so prose cannot drift "
            "from the gate"
        ]

    errors: list[str] = []
    for path in paths:
        if path in FORBIDDEN_IN_CONTRACT:
            errors.append(f"{name} contract names {path}: {FORBIDDEN_IN_CONTRACT[path]}")

    if tuple(paths) != expected:
        errors.append(
            f"{name} contract declares {paths!r} but "
            f"scripts/check-doc-checkpoint.py enforces {list(expected)!r}; "
            "update the prose and the checker in the same change"
        )
    return errors


def main() -> int:
    expected = obligated_surfaces()
    failures: list[str] = []
    blocks: dict[str, list[str] | None] = {}

    for name in CONTRACT_DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            failures.append(f"{name} does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        blocks[name] = contract_paths(text)
        failures.extend(document_errors(name, text, expected))

    present = {name: paths for name, paths in blocks.items() if paths is not None}
    if len(present) == len(CONTRACT_DOCUMENTS):
        distinct = {tuple(paths) for paths in present.values()}
        if len(distinct) > 1:
            failures.append(
                "the doc-obligation contract differs between "
                f"{' and '.join(CONTRACT_DOCUMENTS)}; both must carry the same "
                "block verbatim"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "The obligated public surfaces are defined by PUBLIC_CHECKPOINTS and "
            "FEATURE_CHECKPOINT in scripts/check-doc-checkpoint.py. Mirror them "
            "in the contract block of every document listed in "
            "CONTRACT_DOCUMENTS.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: the doc-obligation contract in "
        f"{' and '.join(CONTRACT_DOCUMENTS)} matches "
        "scripts/check-doc-checkpoint.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
