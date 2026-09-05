#!/usr/bin/env python3
"""Mutation tests for scripts/check-rocm-dp4a-intrinsic.py.

The checker verifies that the `Dp4a` function in `rocm_grouped_gemm.hip`
uses the `__ockl_sdot4` hardware intrinsic.  Each mutation below replaces
the intrinsic with the scalar expansion and asserts the checker goes red,
proving the gate detects the regression.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-rocm-dp4a-intrinsic.py"
SOURCE = ROOT / "src/vt/rocm/rocm_grouped_gemm.hip"

SPEC = importlib.util.spec_from_file_location("check_rocm_dp4a_intrinsic", CHECKER)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

# The live Dp4a body, captured once so each mutation starts from the real
# source rather than a miniature.
_LIVE_TEXT = SOURCE.read_text(encoding="utf-8")

_SCALAR_BODY = """\
  int sum = 0;
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
  for (int i = 0; i < 4; ++i)
    sum += pa[i] * pb[i];
  return acc + sum;
"""

# A Dp4a body that uses the intrinsic — a miniature of the live source.
_INTRINSIC_BODY = """\
  using char4_native = char __attribute__((ext_vector_type(4)));
  char4_native va = *reinterpret_cast<const char4_native*>(&a);
  char4_native vb = *reinterpret_cast<const char4_native*>(&b);
  return __ockl_sdot4(va, vb, acc, false);
"""


def _make_source(dp4a_body: str) -> str:
    """Build a minimal HIP source whose Dp4a body is `dp4a_body`."""
    return (
        "// minimal test source\n"
        "__device__ __forceinline__ int Dp4a(int a, int b, int acc) {\n"
        f"{dp4a_body}"
        "}\n"
    )


class FakeTree:
    """A scratch directory with a miniature rocm_grouped_gemm.hip."""

    def __init__(self, dp4a_body: str) -> None:
        self.dir = tempfile.mkdtemp(prefix="rocm-dp4a-gate-")
        root = Path(self.dir)
        (root / "src/vt/rocm").mkdir(parents=True)
        (root / "src/vt/rocm/rocm_grouped_gemm.hip").write_text(
            _make_source(dp4a_body), encoding="utf-8"
        )

    def __enter__(self) -> Path:
        return Path(self.dir)

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class TestRocmDp4aIntrinsic(unittest.TestCase):
    def test_live_tree_passes(self) -> None:
        errors = checker.check(root=ROOT)
        self.assertEqual(errors, [], errors)

    def test_intrinsic_body_passes(self) -> None:
        with FakeTree(_INTRINSIC_BODY) as root:
            errors = checker.check(root=root)
            self.assertEqual(errors, [], errors)

    def test_scalar_body_fails(self) -> None:
        with FakeTree(_SCALAR_BODY) as root:
            errors = checker.check(root=root)
            self.assertEqual(len(errors), 1, errors)
            self.assertIn("__ockl_sdot4", errors[0])

    def test_missing_dp4a_function_fails(self) -> None:
        with FakeTree("") as root:
            # Empty body still has the function signature, so the body is
            # empty and the intrinsic is absent.
            errors = checker.check(root=root)
            self.assertEqual(len(errors), 1, errors)

    def test_missing_source_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            errors = checker.check(root=root)
            self.assertEqual(len(errors), 1, errors)

    def test_live_scalar_mutation_fails(self) -> None:
        """Replace __ockl_sdot4 in the REAL source and verify the checker
        catches it.  This is the mutation the reviewer asked for: prove the
        gate fails when v_dot4_i32_i8 is not emitted."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src/vt/rocm").mkdir(parents=True)
            # Replace the intrinsic call with the scalar expansion in the
            # live source text.
            mutated = _LIVE_TEXT.replace(
                "  return __ockl_sdot4(va, vb, acc, false);",
                _SCALAR_BODY.rstrip(),
            )
            self.assertNotEqual(mutated, _LIVE_TEXT, "mutation did not apply")
            (root / "src/vt/rocm/rocm_grouped_gemm.hip").write_text(
                mutated, encoding="utf-8"
            )
            errors = checker.check(root=root)
            self.assertEqual(len(errors), 1, errors)
            self.assertIn("__ockl_sdot4", errors[0])


if __name__ == "__main__":
    unittest.main()
