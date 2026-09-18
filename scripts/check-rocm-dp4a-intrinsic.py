#!/usr/bin/env python3
"""Fail if the ROCm Dp4a function does not use the hardware dot-product intrinsic.

The `Dp4a` function in `src/vt/rocm/rocm_grouped_gemm.hip` must CALL
`__ockl_sdot4`, which emits the `v_dot4_i32_i8` instruction on gfx1100.
The scalar expansion — four int8 multiplies plus four adds — is bit-identical
but ~1.4x slower on the KQuantGemmK prefill path.  A CPU-only `ctest` gate
stays green with either form, because the ROCm kernel is not compiled on the
CPU tier.  This checker reads the source and fails when the call is absent,
so the performance lever cannot regress silently.

A `#if __has_builtin(__ockl_sdot4)` guard around the call is fine, and so is
a scalar fallback after it: `__ockl_sdot4` is missing from some ROCm installs
(commit 2bde17f6c).  Neither is a substitute for the call, and this checker
enforces exactly that difference.  It looks for a genuine CALL SITE: the
name followed by an argument list, in code rather than in a comment.  The
probe does not match, because inside `__has_builtin(__ockl_sdot4)` the
intrinsic is followed by `)` — it is NAMED there, not called.

That distinction is the whole gate.  Between 2bde17f6c and this commit the
checker tested for the bare substring `__ockl_sdot4` and, as a second
disjunct, for the probe itself.  Both matched the probe line, so a `Dp4a`
whose guarded branch had been replaced by the scalar expansion — zero call
sites — passed.  `tests/scripts/test_check_rocm_dp4a_intrinsic.py` had
asserted that mutation red the whole time and was simply failing.

Mutation proof: that test file replaces the `__ockl_sdot4` call in the REAL
source with the scalar expansion and asserts this checker goes red.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "src/vt/rocm/rocm_grouped_gemm.hip"

# The intrinsic that emits v_dot4_i32_i8 on gfx1100.
INTRINSIC = "__ockl_sdot4"

# `// ...` and `/* ... */`.  An intrinsic named in prose is not a call, and
# this file has three such mentions above `Dp4a` alone.
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)

# A genuine call: the name followed by an argument list.  This single
# pattern is also what rejects the `__has_builtin(__ockl_sdot4)` availability
# probe, in which the intrinsic is followed by `)`.  An explicit probe-
# stripping pass was written first and then removed: a mutation run proved it
# unreachable, since no body that reaches this point can satisfy one and not
# the other.  `test_has_builtin_probe_alone_fails` pins the behaviour.
_CALL = re.compile(re.escape(INTRINSIC) + r"\s*\(")


def _extract_dp4a(text: str) -> str | None:
    """Return the body of the `Dp4a` function, or None if not found."""
    # Match: __device__ ... int Dp4a( ... ) { ... }
    # Balanced-brace scan from the opening brace after the signature.
    pattern = re.compile(
        r"__device__\s+__forceinline__\s+int\s+Dp4a\s*\([^)]*\)\s*\{",
        re.M,
    )
    match = pattern.search(text)
    if match is None:
        return None
    start = match.end()  # just past '{'
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None  # unbalanced


def _has_call(body: str) -> bool:
    """True when `body` contains a genuine call to the intrinsic.

    Comments are stripped first, so a commented-out call does not count.
    The `__has_builtin` probe needs no special handling; see `_CALL`.
    """
    return _CALL.search(_COMMENTS.sub(" ", body)) is not None


def check(root: Path = REPO) -> list[str]:
    """Return a list of error strings; empty means the gate passes."""
    source = root / "src/vt/rocm/rocm_grouped_gemm.hip"
    if not source.exists():
        return [f"{source.relative_to(root)}: source not found"]
    text = source.read_text(encoding="utf-8")
    body = _extract_dp4a(text)
    if body is None:
        return ["Dp4a function not found in rocm_grouped_gemm.hip"]
    if not _has_call(body):
        return [
            "Dp4a does not CALL the hardware dot-product intrinsic "
            f"({INTRINSIC}). A `__has_builtin({INTRINSIC})` guard around the "
            "call, or a scalar fallback after it, does not substitute for the "
            "call. The scalar expansion is bit-identical but ~1.4x slower on "
            "the KQuantGemmK prefill path."
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args()
    errors = check(root=args.root)
    if errors:
        print("check-rocm-dp4a-intrinsic: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("check-rocm-dp4a-intrinsic: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
