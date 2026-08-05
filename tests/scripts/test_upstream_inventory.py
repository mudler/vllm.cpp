#!/usr/bin/env python3
"""Unit and mutation checks for scripts/upstream-inventory.py (W1).

The parsing is what must be right: an off-by-one in the arch floor would declare
supported arches out of scope, and a sloppy registry regex would invent or hide
uninventoried architectures — the exact error that made the first hand count say
62 when it is 43.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inv = _load("upstream_inventory", "scripts/upstream-inventory.py")

CMAKE = '''cmake_minimum_required(VERSION 3.26)
if (A)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;9.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;9.0;10.0;12.0")
endif()
'''

REGISTRY = '''_TEXT = {
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),
    "BertForMaskedLM": ("bert", "BertForMaskedLM"),
}
_EMBED = {
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),
    "ColPaliForRetrieval": ("colpali", "ColPali"),
}
'''


class ArchFloor(unittest.TestCase):
    def test_widest_list_wins(self) -> None:
        archs, line = inv.parse_supported_archs(CMAKE)
        self.assertEqual(archs, ["7.5", "8.0", "8.6", "9.0", "10.0", "12.0"])
        self.assertEqual(line, 5)  # the wider list is on line 5

    def test_arch_key_ordering(self) -> None:
        self.assertEqual(inv.arch_key("7.5"), 75)
        self.assertEqual(inv.arch_key("10.0"), 100)
        self.assertLess(inv.arch_key("7.5"), inv.arch_key("8.0"))

    def test_rows_below_the_floor_are_found(self) -> None:
        rows = {"BACKEND-CUDA-SM060", "BACKEND-CUDA-SM070", "BACKEND-CUDA-SM075",
                "BACKEND-CUDA-SM090"}
        self.assertEqual(
            inv.below_floor(rows, ["7.5", "8.0"]),
            ["BACKEND-CUDA-SM060", "BACKEND-CUDA-SM070"],
        )

    def test_supported_arch_is_never_called_out_of_scope(self) -> None:
        self.assertEqual(inv.below_floor({"BACKEND-CUDA-SM075"}, ["7.5", "8.0"]), [])

    def test_no_archs_means_no_verdict(self) -> None:
        self.assertEqual(inv.below_floor({"BACKEND-CUDA-SM060"}, []), [])


class Registry(unittest.TestCase):
    def test_archs_and_lines_are_extracted(self) -> None:
        found = inv.registry_archs(REGISTRY)
        self.assertEqual(found["BertForMaskedLM"], 3)
        self.assertEqual(found["ColPaliForRetrieval"], 7)

    def test_duplicates_keep_their_first_line(self) -> None:
        self.assertEqual(inv.registry_archs(REGISTRY)["AquilaForCausalLM"], 2)

    def test_nested_values_are_not_mistaken_for_archs(self) -> None:
        self.assertNotIn("llama", inv.registry_archs(REGISTRY))
        self.assertNotIn("LlamaForCausalLM", inv.registry_archs(REGISTRY))


class LiveTree(unittest.TestCase):
    def test_report_runs_with_or_without_references(self) -> None:
        self.assertEqual(inv.main.__module__, "upstream_inventory")
        data = inv.build()
        self.assertIn("available", data)

    def test_check_is_clean_or_skips(self) -> None:
        sys.argv = ["upstream-inventory.py", "--check"]
        self.assertEqual(inv.main(), 0)


if __name__ == "__main__":
    unittest.main()
