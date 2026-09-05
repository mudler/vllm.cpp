#!/usr/bin/env python3
"""Fail if the ROCm Dp4a function does not use the hardware dot-product intrinsic.

The `Dp4a` function in `src/vt/rocm/rocm_grouped_gemm.hip` must call
`__ockl_sdot4` (which emits the `v_dot4_i32_i8` instruction on gfx1100).
The scalar expansion — four int8 multiplies plus four adds — is bit-identical
but ~1.4x slower on the KQuantGemmK prefill path.  A CPU-only `ctest` gate
stays green with either form, because the ROCm kernel is not compiled on the
CPU tier.  This checker reads the source and fails when the intrinsic is
absent, so the performance lever cannot regress silently.

Mutation proof: `tests/scripts/test_check_rocm_dp4a_intrinsic.py` replaces
the `__ockl_sdot4` call with the scalar expansion and asserts this checker
goes red.
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

# The scalar expansion that the intrinsic replaced.  If this pattern appears
# in the Dp4a body INSTEAD of the intrinsic, the performance lever has
# regressed.
_SCALAR_MARKERS = (
    re.compile(r"\ba\s*\*\s*b", re.M),  # int8 multiply
)


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


def check(root: Path = REPO) -> list[str]:
    """Return a list of error strings; empty means the gate passes."""
    source = root / "src/vt/rocm/rocm_grouped_gemm.hip"
    if not source.exists():
        return [f"{source.relative_to(root)}: source not found"]
    text = source.read_text(encoding="utf-8")
    body = _extract_dp4a(text)
    if body is None:
        return ["Dp4a function not found in rocm_grouped_gemm.hip"]
    if INTRINSIC not in body:
        return [
            "Dp4a does not use the hardware dot-product intrinsic "
            f"({INTRINSIC}). The scalar expansion is bit-identical but "
            "~1.4x slower on the KQuantGemmK prefill path."
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
