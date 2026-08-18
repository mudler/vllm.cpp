#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-cuda-op-arch-gate.py.

The checker is the only mechanism that bites at PR time when a CUDA op that must
exist on every CUDA arch is moved back into a feature-gated translation unit
(issue #960). The runtime pin — `tests/vt/test_ops_fp8_cpu.cpp` G4 — is the
stronger claim but can only speak on a CUDA build WITHOUT cutlass-fp8, which no
CI job produces. So these cases prove the structural gate detects each way the
invariant can be broken, and then run it against the LIVE tree so a refactor that
makes the invariant unreachable cannot pass silently.

EVERY MUTATION BREAKS EXACTLY ONE CLAUSE. A mutation that breaks two proves only
that the union fires. The four clauses (HOME / REGISTERED / UNGUARDED /
EXCLUSIVE) each get their own case, plus the two "text the compiler never sees"
disguises that the earlier generation of checkers in this tree passed on.

THE VACUITY CASE MATTERS MOST. A checker that reports OK because it parsed
nothing is worse than no checker: it is a green light attached to no measurement.
`test_empty_source_list_is_not_a_pass` pins that.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-cuda-op-arch-gate.py"
SPEC = importlib.util.spec_from_file_location("check_cuda_op_arch_gate", CHECKER)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)

# BOUND AS A MODULE, NOT AS THREE NAMES. `scripts/check-pr-size.py` proves a new
# checker red-before by replacing it with a disabled stub and re-running THIS
# module; pulling the functions out at import time would make that an ImportError
# rather than a run of failing cases, and the evidence contract reads an
# unimportable module as "executed no tests" instead of as a red. Every reference
# below goes through `checker.` so the stub fails each case on its own.

HOME = "src/vt/cuda/cuda_quant_fp8.cu"
GATED = "src/vt/cuda/cuda_matmul_fp8_cutlass.cu"
REGISTRATION = (
    "    RegisterOp(OpId::kQuantFp8Static, DeviceType::kCUDA,\n"
    "               reinterpret_cast<void*>("
    "static_cast<QuantFp8StaticFn>(&QuantFp8StaticKernelCuda)));\n"
)
# The miniature describes the REAL tree, and the real TU now registers a second
# op from the same Registrar (#1189 M1, kQuantFp8Group). Without this line the
# baseline miniature is red for a reason that has nothing to do with the
# mutation each case applies, which would hide what those cases measure. Every
# mutation below still targets REGISTRATION, so nothing they assert is widened.
GROUP_REGISTRATION = (
    "    RegisterOp(OpId::kQuantFp8Group, DeviceType::kCUDA,\n"
    "               reinterpret_cast<void*>("
    "static_cast<QuantFp8GroupFn>(&QuantFp8GroupKernelCuda)));\n"
)


class FakeTree:
    """A miniature checkout: CMakeLists.txt plus the two CUDA TUs, arranged
    exactly as the real tree is, so a mutation can be applied to one of them."""

    def __init__(self, cmake: str, home_src: str, gated_src: str) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="cuda-op-arch-gate-"))
        (self.dir / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
        cuda = self.dir / "src/vt/cuda"
        cuda.mkdir(parents=True)
        (cuda / "cuda_quant_fp8.cu").write_text(home_src, encoding="utf-8")
        (cuda / "cuda_matmul_fp8_cutlass.cu").write_text(gated_src, encoding="utf-8")

    def __enter__(self) -> Path:
        return self.dir

    def __exit__(self, *exc: object) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


BASE_CMAKE = """\
project(mini)
if(VLLM_CPP_CUDA)
  target_sources(vllm PRIVATE
    src/vt/cuda/cuda_matmul.cu
    src/vt/cuda/cuda_quant_fp8.cu
    src/vt/cuda/cuda_ops.cu)
  if(VLLM_CPP_CUTLASS)
    set(_FP8_CUTLASS_SOURCES)
    if(VT_CUTLASS_FP8_ARCHS)
      set(_FP8_CUTLASS_SOURCES src/vt/cuda/cuda_matmul_fp8_cutlass.cu)
    endif()
    target_sources(vllm PRIVATE ${_FP8_CUTLASS_SOURCES})
  endif()
endif()
"""

BASE_HOME = f"""\
#include "vt/ops.h"
namespace vt::cuda {{
namespace {{
void QuantFp8StaticKernelCuda(Queue&, Tensor&, const Tensor&, float) {{}}
void QuantFp8GroupKernelCuda(Queue&, Tensor&, Tensor&, const Tensor&, int) {{}}
struct Registrar {{
  Registrar() {{
{REGISTRATION}{GROUP_REGISTRATION}  }}
}};
Registrar g_registrar;
}}
}}
"""

BASE_GATED = """\
#include "vt/ops.h"
namespace vt::cuda {
namespace {
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulFp8Cutlass, DeviceType::kCUDA,
               reinterpret_cast<void*>(&MatmulFp8CutlassKernelCuda));
  }
};
Registrar g_registrar;
}
}
"""


def run(cmake: str = BASE_CMAKE, home: str = BASE_HOME, gated: str = BASE_GATED) -> list[str]:
    with FakeTree(cmake, home, gated) as root:
        return checker.check(root=root)


class TestCmakeParse(unittest.TestCase):
    def test_reads_the_unconditional_list_only(self) -> None:
        srcs = checker.unconditional_cuda_sources(BASE_CMAKE)
        self.assertIn("src/vt/cuda/cuda_quant_fp8.cu", srcs)
        # The cutlass TU is added under a NESTED if(), never at depth [VLLM_CPP_CUDA].
        self.assertNotIn("src/vt/cuda/cuda_matmul_fp8_cutlass.cu", srcs)
        # ...and a variable expansion is not a literal home.
        self.assertNotIn("${_FP8_CUTLASS_SOURCES}", srcs)

    def test_else_branch_is_not_unconditional(self) -> None:
        cmake = BASE_CMAKE.replace(
            "if(VLLM_CPP_CUDA)\n  target_sources",
            "if(VLLM_CPP_HIP)\nelse()\n  target_sources",
        )
        self.assertNotIn("src/vt/cuda/cuda_quant_fp8.cu", checker.unconditional_cuda_sources(cmake))

    def test_cmake_comment_is_not_a_source(self) -> None:
        cmake = BASE_CMAKE.replace(
            "    src/vt/cuda/cuda_quant_fp8.cu\n",
            "    # src/vt/cuda/cuda_quant_fp8.cu\n",
        )
        self.assertNotIn("src/vt/cuda/cuda_quant_fp8.cu", checker.unconditional_cuda_sources(cmake))


class TestMutations(unittest.TestCase):
    def test_baseline_miniature_is_green(self) -> None:
        # Non-vacuity for every case below: they must differ from a passing state.
        self.assertEqual(run(), [])

    def test_HOME_moving_the_TU_under_the_cutlass_gate_goes_red(self) -> None:
        # THE ORIGINAL DEFECT, reproduced: the TU is compiled only when the
        # cutlass-fp8 arch set is non-empty.
        cmake = BASE_CMAKE.replace("    src/vt/cuda/cuda_quant_fp8.cu\n", "").replace(
            "set(_FP8_CUTLASS_SOURCES src/vt/cuda/cuda_matmul_fp8_cutlass.cu)",
            "set(_FP8_CUTLASS_SOURCES src/vt/cuda/cuda_matmul_fp8_cutlass.cu"
            " src/vt/cuda/cuda_quant_fp8.cu)",
        )
        problems = run(cmake=cmake)
        self.assertTrue(any("unconditional CUDA source list" in p for p in problems), problems)

    def test_REGISTERED_deleting_the_registration_goes_red(self) -> None:
        problems = run(home=BASE_HOME.replace(REGISTRATION, ""))
        self.assertTrue(any("expected exactly ONE" in p for p in problems), problems)

    def test_UNGUARDED_wrapping_the_registration_in_ifdef_goes_red(self) -> None:
        # The subtle regression: the TU stays in the unconditional list, so clause
        # (a) is satisfied, and the registration is still textually present, so a
        # naive grep passes -- but the arch gate is exactly back.
        guarded = BASE_HOME.replace(
            REGISTRATION, "#ifdef VT_CUTLASS_FP8\n" + REGISTRATION + "#endif\n"
        )
        problems = run(home=guarded)
        self.assertTrue(any("conditional depth" in p for p in problems), problems)

    def test_EXCLUSIVE_a_second_gated_registration_goes_red(self) -> None:
        problems = run(gated=BASE_GATED.replace(
            "    RegisterOp(OpId::kMatmulFp8Cutlass", REGISTRATION + "    RegisterOp(OpId::kMatmulFp8Cutlass"
        ))
        self.assertTrue(any("ALSO registered for kCUDA" in p for p in problems), problems)

    def test_disguised_deletion_by_comment_goes_red(self) -> None:
        # A commented-out registration is a deletion to the compiler. It must read
        # as one here too -- the failure mode this tree has paid for before.
        commented = BASE_HOME.replace(
            REGISTRATION,
            "".join("//" + ln + "\n" for ln in REGISTRATION.splitlines()),
        )
        problems = run(home=commented)
        self.assertTrue(any("expected exactly ONE" in p for p in problems), problems)

    def test_disguised_deletion_by_if_zero_goes_red(self) -> None:
        disabled = BASE_HOME.replace(REGISTRATION, "#if 0\n" + REGISTRATION + "#endif\n")
        problems = run(home=disabled)
        self.assertTrue(any("expected exactly ONE" in p for p in problems), problems)

    def test_empty_source_list_is_not_a_pass(self) -> None:
        # A parser that matches nothing must FAIL, not report OK. A green light
        # attached to no measurement is the worst outcome available to a gate.
        problems = run(cmake="project(mini)\n")
        self.assertTrue(any("found NO unconditional" in p for p in problems), problems)


class TestLiveTree(unittest.TestCase):
    def test_live_tree_passes(self) -> None:
        self.assertEqual(checker.check(root=ROOT), [])

    def test_live_registration_is_where_the_checker_says(self) -> None:
        # Pins the checker to the REAL file rather than only to miniatures: if the
        # kernel is renamed or the TU disappears, this fails rather than drifting.
        regs = checker.cuda_registrations("kQuantFp8Static", ROOT)
        self.assertEqual(list(regs), [HOME], regs)
        self.assertEqual([depth for _, depth in regs[HOME]], [0], regs)
        self.assertFalse((ROOT / GATED).read_text(encoding="utf-8").count("kQuantFp8Static,"))

    def test_checker_cli_exits_zero_on_the_live_tree(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(CHECKER), "--report"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("check-cuda-op-arch-gate: OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
