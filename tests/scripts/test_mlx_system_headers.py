#!/usr/bin/env python3
"""Compiler-level regression tests for the MLX CMake dependency boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class MlxSystemHeadersTest(unittest.TestCase):
    def test_dependency_target_marks_mlx_headers_as_system(self) -> None:
        dependency = (ROOT / "cmake" / "MLXDependency.cmake").read_text(
            encoding="utf-8"
        )
        self.assertIn("INTERFACE_SYSTEM_INCLUDE_DIRECTORIES", dependency)

    def test_mlx_includes_have_source_scoped_clang_warning_suppression(self) -> None:
        provider = (ROOT / "src" / "vt" / "metal" / "metal_mlx_provider.mm").read_text(
            encoding="utf-8"
        )
        required_pragmas = (
            "#pragma clang diagnostic push",
            '#pragma clang diagnostic ignored "-Wgnu-folding-constant"',
            "#pragma clang diagnostic pop",
        )
        for pragma in required_pragmas:
            self.assertIn(pragma, provider)
        self.assertEqual(provider.count("#if defined(__clang__)"), 2)
        self.assertIn(
            "#if defined(__clang__)\n"
            "#pragma clang diagnostic push\n"
            '#pragma clang diagnostic ignored "-Wgnu-folding-constant"\n'
            "#endif\n",
            provider,
        )
        self.assertIn(
            '#include "mlx/transforms.h"\n'
            "#if defined(__clang__)\n"
            "#pragma clang diagnostic pop\n"
            "#endif\n",
            provider,
        )
        push = provider.index("#pragma clang diagnostic push")
        ignored = provider.index(
            '#pragma clang diagnostic ignored "-Wgnu-folding-constant"'
        )
        first_mlx_include = provider.index('#include "mlx/allocator.h"')
        last_mlx_include = provider.index('#include "mlx/transforms.h"')
        pop = provider.index("#pragma clang diagnostic pop")
        first_project_include = provider.index('#include "metal_buffers.h"')
        self.assertLess(push, ignored)
        self.assertLess(ignored, first_mlx_include)
        self.assertLess(last_mlx_include, pop)
        self.assertLess(pop, first_project_include)

    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("clang++") and shutil.which("ar"),
        "needs CMake, Clang, and ar",
    )
    def _build(self, project_warning: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mlx_root = root / "mlx"
            (mlx_root / "include" / "mlx").mkdir(parents=True)
            (mlx_root / "lib").mkdir()
            (mlx_root / "include" / "mlx" / "array.h").write_text(
                "inline void mlx_warning() { const int n = 4; int values[n]; (void)values; }\n",
                encoding="utf-8",
            )
            library = mlx_root / "lib" / "libmlx.a"
            subprocess.run(["ar", "rcs", str(library)], check=True)
            warning = (
                "const int n = 4; int project_values[n]; (void)project_values;"
                if project_warning
                else "mlx_warning();"
            )
            (root / "main.cpp").write_text(
                f'#include "mlx/array.h"\nint main() {{ {warning} return 0; }}\n',
                encoding="utf-8",
            )
            (root / "CMakeLists.txt").write_text(
                "\n".join(
                    [
                        "cmake_minimum_required(VERSION 3.20)",
                        "project(mlx_system_boundary LANGUAGES CXX)",
                        f'list(APPEND CMAKE_MODULE_PATH "{ROOT / "cmake"}")',
                        "include(MLXDependency)",
                        f'vllm_cpp_import_mlx("{mlx_root}" "{library}")',
                        "add_executable(probe main.cpp)",
                        "target_link_libraries(probe PRIVATE vllm_cpp::mlx)",
                        "target_compile_options(probe PRIVATE -Wall -Wextra -Werror)",
                    ]
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["CXX"] = shutil.which("clang++") or "clang++"
            configured = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            if configured.returncode:
                return configured
            return subprocess.run(
                ["cmake", "--build", str(root / "build"), "--verbose"],
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

    def test_mlx_header_warning_is_not_promoted_by_project_werror(self) -> None:
        result = self._build(project_warning=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_project_warning_remains_fatal(self) -> None:
        result = self._build(project_warning=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("gnu-folding-constant", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
