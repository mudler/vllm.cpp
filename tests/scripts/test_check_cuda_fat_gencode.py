#!/usr/bin/env python3
"""Mutation tests for the W1 ten-SM fat-CUDA compile audit."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-cuda-fat-gencode.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_cuda_fat_gencode", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()
ALL = ("80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a")


def command(source: str, sms: tuple[str, ...]) -> dict[str, str]:
    flags = " ".join(
        f"-gencode arch=compute_{sm},code=sm_{sm}" for sm in sms
    )
    return {"file": str(ROOT / source), "command": f"nvcc {flags} -c {source}"}


def valid_commands() -> list[dict[str, str]]:
    return [
        command("src/vt/cuda/cuda_matmul_nvfp4.cu", ALL),
        command("src/vt/cuda/cuda_nvfp4_sm12x.cu", ("120a", "121a")),
        command("src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu", ("120a", "121a")),
        command("src/vt/cuda/cuda_matmul_fp8_cutlass.cu", ("120a", "121a")),
        command("src/vt/cuda/cuda_matmul_nvfp4_sm100.cu", ("100a",)),
        command("src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu", ("90a",)),
        command("src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu", ("100a",)),
        # Marlin's set is the feature table's, not a literal here: restating it
        # is exactly what drifted in #394.
        command("src/vt/cuda/cuda_moe_marlin.cu", checker.expected_sms("src/vt/cuda/cuda_moe_marlin.cu")),
        command(
            "src/vt/cuda/cuda_flash_attn_fa2.cu",
            ("80", "86", "87", "89", "120a", "121a"),
        ),
    ]


class FatGencodeContract(unittest.TestCase):
    def run_checker(
        self, entries: list[dict[str, str]], archive_sms: tuple[str, ...] = ALL
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commands = root / "compile_commands.json"
            commands.write_text(json.dumps(entries), encoding="utf-8")
            listing = root / "cuobjdump.txt"
            listing.write_text(
                "\n".join(f"Fatbin elf code: sm_{sm}" for sm in archive_sms),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--compile-commands",
                    str(commands),
                    "--cuobjdump-list",
                    str(listing),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_exact_source_and_archive_matrix_passes(self) -> None:
        entries = valid_commands()
        entries.append(
            {
                "file": "/tmp/cutlass/tools/profiler/src/cublas_helpers.cu",
                "command": "nvcc -c cublas_helpers.cu",
            }
        )
        result = self.run_checker(entries)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_portable_source_missing_one_sm_fails(self) -> None:
        entries = valid_commands()
        entries[0] = command("src/vt/cuda/cuda_matmul_nvfp4.cu", ALL[:-1])
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cuda_matmul_nvfp4.cu", result.stdout + result.stderr)

    def test_sm12x_source_rejects_incompatible_sm(self) -> None:
        entries = valid_commands()
        entries[1] = command(
            "src/vt/cuda/cuda_nvfp4_sm12x.cu", ("90a", "120a", "121a")
        )
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cuda_nvfp4_sm12x.cu", result.stdout + result.stderr)

    def test_archive_missing_one_sm_fails(self) -> None:
        result = self.run_checker(valid_commands(), ALL[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("archive", result.stdout + result.stderr)

    def test_archive_rejects_undeclared_sm(self) -> None:
        result = self.run_checker(valid_commands(), ALL + ("999",))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("undeclared", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()


class MarlinTracksTheFeatureTable(unittest.TestCase):
    """#394: the checker restated a fact cmake already owns, and they drifted.

    `cmake/CudaArchFeatures.cmake` declares the marlin-nvfp4 arch set. sm_110
    was added there for Thor (#325) and the checker's hardcoded SM12X was not
    updated, so every ten-SM fat build failed its own audit on 14 Marlin TUs
    while the build itself was correct.

    RED-first: with `return SM12X` these assertions fail for 110.
    """

    def feature_table_marlin_archs(self) -> set[str]:
        text = (ROOT / "cmake/CudaArchFeatures.cmake").read_text(encoding="utf-8")
        row = re.search(r'"marlin-nvfp4\|([^|]+)\|', text)
        self.assertIsNotNone(row, "the marlin-nvfp4 feature row must exist")
        return {
            arch.strip().replace(".", "", 1) + ""
            for arch in row.group(1).split(",")
        }

    def test_marlin_expectation_equals_the_feature_table(self):
        table = self.feature_table_marlin_archs()
        for source in (
            "src/vt/cuda/cuda_moe_marlin.cu",
            "src/vt/cuda/cuda_marlin_dense.cu",
            "src/vt/cuda/cuda_marlin_repack.cu",
            "src/vt/cuda/marlin/libtorch_stable/quantization/marlin/marlin_mm_dense.cu",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    set(checker.expected_sms(source)),
                    table,
                    "the checker and cmake/CudaArchFeatures.cmake must agree; "
                    "restating the arch set is what caused #394",
                )

    def test_sm110_is_expected_for_marlin(self):
        """The specific regression, named: Thor's SM must not be audited away."""
        self.assertIn("110", checker.expected_sms("src/vt/cuda/cuda_moe_marlin.cu"))
