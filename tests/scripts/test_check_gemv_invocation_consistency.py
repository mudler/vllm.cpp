#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-gemv-invocation-consistency.py."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-gemv-invocation-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_gemv_invocation_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

output_layout_sites = mod.output_layout_sites
hardcoded_output_dtype_sites = mod.hardcoded_output_dtype_sites
requested_algo_args = mod.requested_algo_args
bare_algo_count_sites = mod.bare_algo_count_sites


class OutputDtypeInvariantTests(unittest.TestCase):
    """Invariant A(1): the C/D (output) layout dtype must be the out_type variable."""

    def test_helper_lc_out_type_passes(self) -> None:
        text = "MakeRowMajor(lc, out_type, m, n);"
        self.assertEqual(output_layout_sites(text), [(1, "out_type")])
        self.assertEqual(hardcoded_output_dtype_sites(text), [])

    def test_helper_lc_hardcoded_literal_fails(self) -> None:
        # Mutation (1): a hardcoded CUDA_R_32F output layout on a bf16 GEMM must trip.
        text = "MakeRowMajor(lc, CUDA_R_32F, m, n);"
        self.assertEqual(hardcoded_output_dtype_sites(text), [(1, "CUDA_R_32F")])

    def test_batched_helper_lc_hardcoded_fails(self) -> None:
        text = "MakeRowMajorBatched(lc, CUDA_R_32F, m, n, s, b, bs);"
        self.assertEqual(hardcoded_output_dtype_sites(text), [(1, "CUDA_R_32F")])

    def test_raw_lc_hardcoded_literal_fails(self) -> None:
        text = "cublasLtMatrixLayoutCreate(&lc.v, CUDA_R_32F, n, m, n);"
        self.assertEqual(hardcoded_output_dtype_sites(text), [(1, "CUDA_R_32F")])

    def test_raw_plc_out_type_passes(self) -> None:
        # The fp8 plan builds p.lc — the `p.` prefix must still resolve to the lc role.
        text = "cublasLtMatrixLayoutCreate(&p.lc, out_type, n, m, n);"
        self.assertEqual(output_layout_sites(text), [(1, "out_type")])
        self.assertEqual(hardcoded_output_dtype_sites(text), [])

    def test_ab_operand_literals_out_of_scope(self) -> None:
        # A/B operand layouts (la/lb) are NOT the template-selecting output dtype, so a
        # literal there is out of scope for this invariant — it must not be flagged.
        text = (
            "MakeRowMajor(la, CUDA_R_32F, m, k);\n"
            "cublasLtMatrixLayoutCreate(&lb.v, CUDA_R_16BF, k, n, k);\n"
        )
        self.assertEqual(output_layout_sites(text), [])
        self.assertEqual(hardcoded_output_dtype_sites(text), [])

    def test_helper_definition_head_not_matched(self) -> None:
        # The helper DEFINITION `MakeRowMajor(LayoutGuard& l, cudaDataType_t t, ...)` is
        # not a create site (no comma after the first token) and must not be scanned.
        text = "void MakeRowMajor(LayoutGuard& l, cudaDataType_t t, int64_t rows) {"
        self.assertEqual(output_layout_sites(text), [])

    def test_shipped_tree_is_green_out_dtype(self) -> None:
        text = mod.MATMUL_CU.read_text(encoding="utf-8")
        sites = output_layout_sites(text)
        # The sweep really found the five C/D layout sites (Matmul, MatmulBT,
        # BatchedMatmul, fp8 plan, and GetOrQueryGemmHeuristic's shared layout
        # after FIX-CUBLASLT-CAPTURE-1732 routed the three bf16/f32 lanes
        # through one cached query) — a non-vacuous green.
        self.assertEqual(len(sites), 5)
        self.assertTrue(all(dtype == "out_type" for _, dtype in sites))
        self.assertEqual(hardcoded_output_dtype_sites(text), [])

    def test_mutated_tree_would_fail_out_dtype(self) -> None:
        # Mutation on the real file text: hardcode the fp8 plan's C/D layout to f32.
        text = mod.MATMUL_CU.read_text(encoding="utf-8")
        mutated = text.replace(
            "cublasLtMatrixLayoutCreate(&p.lc, out_type",
            "cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F",
            1,
        )
        self.assertNotEqual(mutated, text)  # the anchor really exists
        self.assertTrue(hardcoded_output_dtype_sites(mutated))


class AlgoPolicyInvariantTests(unittest.TestCase):
    """Invariant A(2): requestedAlgoCount must be a named constant, not a bare literal."""

    def test_named_constant_passes(self) -> None:
        text = "..., pref.v, /*requestedAlgoCount=*/kGemvHeuristicAlgos, &heur, &n),"
        self.assertEqual(requested_algo_args(text), [(1, "kGemvHeuristicAlgos")])
        self.assertEqual(bare_algo_count_sites(text), [])

    def test_bare_literal_fails(self) -> None:
        # Mutation (2): a bare requestedAlgoCount literal must trip.
        text = "..., pref.v, /*requestedAlgoCount=*/1, &heur, &n),"
        self.assertEqual(bare_algo_count_sites(text), [(1, "1")])

    def test_unmarked_call_is_counted(self) -> None:
        # A heuristic call without the marker cannot smuggle a literal past the gate.
        text = "cublasLtMatmulAlgoGetHeuristic(h, d, la, lb, lc, lc, p, 1, &heur, &n);"
        self.assertEqual(mod.unmarked_heuristic_calls(text), 1)

    def test_constant_declaration_detected(self) -> None:
        self.assertTrue(mod.declares_algo_constant("constexpr int kGemvHeuristicAlgos = 1;"))
        self.assertFalse(mod.declares_algo_constant("int kGemvHeuristicAlgos = 1;"))

    def test_shipped_tree_is_green_algo(self) -> None:
        text = mod.MATMUL_CU.read_text(encoding="utf-8")
        args = requested_algo_args(text)
        # Two heuristic queries — GetOrQueryGemmHeuristic (the single cached
        # query the three bf16/f32 lanes share since FIX-CUBLASLT-CAPTURE-1732)
        # and the fp8 plan build — all through the named constant; a
        # non-vacuous green.
        self.assertEqual(len(args), 2)
        self.assertTrue(all(arg == "kGemvHeuristicAlgos" for _, arg in args))
        self.assertEqual(bare_algo_count_sites(text), [])
        self.assertEqual(mod.unmarked_heuristic_calls(text), 0)
        self.assertTrue(mod.declares_algo_constant(text))

    def test_mutated_tree_would_fail_algo(self) -> None:
        # Mutation on the real file text: revert one site to a bare literal.
        text = mod.MATMUL_CU.read_text(encoding="utf-8")
        mutated = text.replace(
            "/*requestedAlgoCount=*/kGemvHeuristicAlgos", "/*requestedAlgoCount=*/1", 1
        )
        self.assertNotEqual(mutated, text)
        self.assertTrue(bare_algo_count_sites(mutated))


class MainTests(unittest.TestCase):
    def test_main_passes_on_head(self) -> None:
        self.assertEqual(mod.main(), 0)


if __name__ == "__main__":
    unittest.main()
