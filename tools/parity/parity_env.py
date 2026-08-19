"""Resolve the pinned vLLM checkout every parity dumper reads, or refuse by name.

Issue #1190, row `ENV-AGNOSTIC-W1-TOOLING`. The dumpers here defaulted their
`--pin` / `--pinned-vllm` flag to one operator's checkout. That default did not
fail for a second developer; it resolved to whatever happened to sit at the same
path, and the dump then executed ORACLE MATH from the wrong tree and recorded
the wrong provenance in the manifest. A goldens file cannot show that, so an
unset value refuses here instead.

Standard library only, and no torch import, so the refusal is reachable and
testable on a machine that has neither torch nor a GPU.
"""

from __future__ import annotations

import os
import pathlib
import sys


#: `.env.example` declares this key as "the pinned vLLM checkout used for 1:1
#: porting and every parity citation", which is exactly what a dumper needs.
KEY = "VLLM_SOURCE"

#: Distinct from 1 (a dumper that ran and failed) and 2 (argparse), so a caller
#: can tell an unresolved environment from a real error.
EXIT_UNRESOLVED = 3


def resolve_pinned_source(explicit: str | None, flag: str) -> pathlib.Path:
    """The explicit flag, else `${VLLM_SOURCE}`, else refuse.

    The flag wins so a maintainer can dump from a second checkout without
    editing anything, which is what the old default was really being used for.
    """
    for value in (explicit, os.environ.get(KEY)):
        if value and value.strip():
            resolved = pathlib.Path(value.strip()).expanduser()
            # Say which tree the oracle math will execute from. A dumper writes
            # the pin into its manifest, and a run whose source nobody saw is
            # the same defect class as an unasserted oracle identity (#520).
            print(f"pinned vLLM source: {resolved}", file=sys.stderr)
            return resolved
    refuse(flag)


def refuse(flag: str) -> None:
    """Name the key, the file, and the flag, then stop before any dump."""
    root = pathlib.Path(__file__).resolve().parents[2]
    print(
        f"REFUSED: {KEY} is unset. It is the pinned vLLM checkout the oracle "
        "math executes from, and a wrong tree dumps wrong goldens without "
        "failing.",
        file=sys.stderr,
    )
    print(f"Set it in {root}/.env (copy .env.example) or in this shell,", file=sys.stderr)
    print(f"or pass {flag} explicitly.", file=sys.stderr)
    print("Ask the developer for the value. Never substitute another", file=sys.stderr)
    print("developer's host or path, and never guess a default.", file=sys.stderr)
    raise SystemExit(EXIT_UNRESOLVED)
