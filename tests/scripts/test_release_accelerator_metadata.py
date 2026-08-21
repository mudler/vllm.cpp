#!/usr/bin/env python3
"""W10/W11 Linux accelerator release metadata contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/release_accelerator_metadata.py"
BUILD_SCRIPT = ROOT / "scripts/build-linux-accelerator-release.sh"
CUDA_STUB_PREP = ROOT / "scripts/prepare-cuda-driver-stub.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"
SMS = ["80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a"]


def load():
    spec = importlib.util.spec_from_file_location("release_accelerator_metadata", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AcceleratorMetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load()

    def fixture(self, scratch: Path, artifact_id: str):
        build = scratch / "build"
        stage = scratch / "stage"
        output = scratch / "metadata"
        build.mkdir()
        (stage / "bin").mkdir(parents=True)
        shutil.copy2("/bin/true", stage / "bin/vllm-server")
        cuda = "cuda" in artifact_id
        vulkan = "vulkan" in artifact_id
        cache = {
            "MLX_ROOT": ("PATH", ""),
            "VLLM_CPP_BUILD_EXAMPLES": ("BOOL", "ON"),
            "VLLM_CPP_BUILD_TESTS": ("BOOL", "ON"),
            "VLLM_CPP_CUDA": ("BOOL", "ON" if cuda else "OFF"),
            "VLLM_CPP_CUDA_ARCHITECTURES": ("STRING", ";".join(SMS) if cuda else ""),
            "VLLM_CPP_HIP": ("BOOL", "OFF"),
            "VLLM_CPP_HIP_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_LITERAL_STATIC": ("BOOL", "OFF"),
            "VLLM_CPP_METAL": ("BOOL", "OFF"),
            "VLLM_CPP_MLX": ("BOOL", "OFF"),
            "VLLM_CPP_SERVER": ("BOOL", "ON"),
            "VLLM_CPP_TRITON": ("BOOL", "ON" if cuda else "OFF"),
            "VLLM_CPP_VULKAN": ("BOOL", "ON" if vulkan else "OFF"),
        }
        (build / "CMakeCache.txt").write_text(
            "".join(f"{key}:{kind}={value}\n" for key, (kind, value) in cache.items()),
            encoding="utf-8",
        )
        return argparse.Namespace(
            abi_version="2.39",
            artifact_id=artifact_id,
            build_dir=build,
            c_abi_version=17,
            channel="preview",
            compiler="GNU C++ 13.2.0",
            evidence_url="https://github.com/mudler/vllm.cpp/actions/runs/1",
            output_dir=output,
            repo_root=ROOT,
            source_clean=True,
            source_commit=SHA,
            stage_dir=stage,
            toolchain="cmake-3.30+ninja-1.12",
            version="0.0.1",
        )

    def test_cuda_manifest_carries_all_sms_aot_and_external_driver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-cuda")
            manifest = self.tool.prepare_accelerator_metadata(args)
            self.assertEqual(manifest["cuda"]["compiled_sms"], SMS)
            self.assertEqual(
                [row["aot_available"] for row in manifest["cuda"]["sm_evidence"]],
                [True, True, False, True, True, True, False, False, False, True],
            )
            driver = next(row for row in manifest["dependencies"] if row["name"] == "nvidia-driver")
            self.assertEqual((driver["linkage"], driver["bundled"]), ("external", False))

    def test_vulkan_manifest_has_all_three_external_runtime_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-vulkan")
            manifest = self.tool.prepare_accelerator_metadata(args)
            names = {row["name"] for row in manifest["dependencies"]}
            self.assertTrue({"vulkan-loader", "vulkan-icd", "vulkan-driver"} <= names)
            self.assertNotIn("cuda", manifest)

    def test_aarch64_vulkan_is_registered_preview_and_claims_no_evidence(self) -> None:
        # Issue #1547. `docker/Dockerfile:145-146` selects this id for every
        # non-amd64 container build, and before the fix the script refused it
        # with `unsupported Linux accelerator artifact`, which failed the
        # `vulkan` lane's arm64 leg, skipped `manifest`, and left the pushed
        # `cpu` digests with no tag.
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-aarch64-glibc-vulkan")
            manifest = self.tool.prepare_accelerator_metadata(args)
            self.assertEqual(manifest["host"]["arch"], "aarch64")
            self.assertEqual(manifest["backend"]["name"], "vulkan")
            self.assertNotIn("cuda", manifest)
            names = {row["name"] for row in manifest["dependencies"]}
            self.assertTrue({"vulkan-loader", "vulkan-icd", "vulkan-driver"} <= names)
            # The lane is unbuilt and ungated, and the metadata must say so.
            self.assertEqual(manifest["artifact"]["channel"], "preview")
            for axis in ("correctness", "runtime", "performance"):
                self.assertEqual(manifest["evidence"][axis]["state"], "absent")

    def test_aarch64_vulkan_is_not_a_downloadable_release_artifact(self) -> None:
        # The build tuple above is NOT a release download. `release_pipeline.py`
        # requires `release/release-matrix.json` to equal its primary tuple
        # policy exactly, and `.github/workflows/release.yml` builds no aarch64
        # Vulkan tarball, so declaring one there would claim an archive that
        # never exists.
        matrix = json.loads(
            (ROOT / "release/release-matrix.json").read_text(encoding="utf-8")
        )
        ids = {item["id"] for item in matrix["artifacts"]}
        self.assertNotIn("linux-aarch64-glibc-vulkan", ids)
        self.assertIn("linux-x86_64-glibc-vulkan", ids)

    def test_cuda_refuses_partial_sm_or_disabled_triton_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "linux-x86_64-glibc-cuda")
            cache = (args.build_dir / "CMakeCache.txt").read_text()
            for before, after in (
                ("80;86;87;89;90a;100a;103a;110;120a;121a", "80;86"),
                ("VLLM_CPP_TRITON:BOOL=ON", "VLLM_CPP_TRITON:BOOL=OFF"),
            ):
                (args.build_dir / "CMakeCache.txt").write_text(cache.replace(before, after), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.tool.prepare_accelerator_metadata(args)
                (args.build_dir / "CMakeCache.txt").write_text(cache, encoding="utf-8")

    def test_cuda_archive_smoke_uses_external_cleaned_driver_stub_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            repo = scratch / "repo"
            scripts = repo / "scripts"
            fake_bin = scratch / "bin"
            log = scratch / "log"
            external_runtime = scratch / "external-runtime"
            scripts.mkdir(parents=True)
            fake_bin.mkdir()
            log.mkdir()
            shutil.copy2(BUILD_SCRIPT, scripts / BUILD_SCRIPT.name)
            (repo / "include").mkdir()
            (repo / "include/vllm.h").write_text(
                "#define VLLM_ABI_VERSION 17\n",
                encoding="utf-8",
            )

            shims = {
                "cmake": """#!/bin/sh
if [ "${1:-}" = --version ]; then
  echo 'cmake version 3.30.0'
fi
""",
                "ninja": """#!/bin/sh
echo '1.12.0'
""",
                "c++": """#!/bin/sh
echo 'c++ (GCC) 13.2.0'
""",
                "mktemp": """#!/bin/sh
printf '%s\n' "$*" > "$TEST_LOG/mktemp.args"
mkdir -p "$TEST_EXTERNAL_RUNTIME"
printf '%s\n' "$TEST_EXTERNAL_RUNTIME"
""",
                "python3": """#!/bin/sh
if [ "${1:-}" = scripts/validate-release-archive.py ]; then
  printf '%s\n' "$@" > "$TEST_LOG/validate.args"
  printf '%s\n' "${LD_LIBRARY_PATH:-}" > "$TEST_LOG/validate.ld-library-path"
  exit 23
fi
""",
            }
            for name, content in shims.items():
                path = fake_bin / name
                path.write_text(content, encoding="utf-8")
                path.chmod(0o755)

            stub_prep = scripts / CUDA_STUB_PREP.name
            stub_prep.write_text(
                """#!/bin/sh
printf '%s\n' "$2" > "$TEST_LOG/stub-runtime.arg"
mkdir -p "$2/canonical-runtime"
realpath "$2/canonical-runtime"
""",
                encoding="utf-8",
            )
            stub_prep.chmod(0o755)

            build_dir = "build-release-cuda-x86"
            env = dict(os.environ)
            env.update(
                {
                    "EVIDENCE_URL": "https://github.com/mudler/vllm.cpp/actions/runs/1",
                    "LD_LIBRARY_PATH": "/host/runtime",
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "SOURCE_DATE_EPOCH": "0",
                    "SOURCE_SHA": SHA,
                    "TEST_EXTERNAL_RUNTIME": str(external_runtime),
                    "TEST_LOG": str(log),
                    "VERSION": "0.0.1",
                }
            )
            result = subprocess.run(
                [
                    "bash",
                    f"scripts/{BUILD_SCRIPT.name}",
                    "linux-x86_64-glibc-cuda",
                    "cuda",
                    build_dir,
                ],
                cwd=repo,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 23, result.stderr)
            mktemp_log = log / "mktemp.args"
            self.assertTrue(
                mktemp_log.exists(),
                "release driver did not allocate the CUDA validation runtime with mktemp",
            )
            self.assertEqual(mktemp_log.read_text(encoding="utf-8").strip(), "-d")
            self.assertEqual(
                (log / "stub-runtime.arg").read_text(encoding="utf-8").strip(),
                str(external_runtime),
            )
            self.assertFalse(external_runtime.exists(), "validation runtime was not cleaned on failure")
            self.assertEqual(
                (log / "validate.ld-library-path").read_text(encoding="utf-8").strip(),
                f"{external_runtime}/canonical-runtime:/host/runtime",
            )
            validate_args = (log / "validate.args").read_text(encoding="utf-8").splitlines()
            self.assertEqual(validate_args[-2:], ["--forbid-path", str(repo / build_dir)])

    def test_cuda_driver_stub_preparation_adds_runtime_soname_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            versioned_root = scratch / "cuda-13.3"
            stub = versioned_root / "targets/sbsa-linux/lib/stubs/libcuda.so"
            stub.parent.mkdir(parents=True)
            stub.write_bytes(b"external CUDA driver stub")
            cuda_root = scratch / "cuda"
            cuda_root.symlink_to(versioned_root, target_is_directory=True)
            runtime_dir = scratch / "runtime"

            result = subprocess.run(
                ["bash", str(CUDA_STUB_PREP), str(cuda_root), str(runtime_dir)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(runtime_dir))
            soname_alias = runtime_dir / "libcuda.so.1"
            self.assertTrue(soname_alias.is_symlink())
            self.assertEqual(soname_alias.resolve(strict=True), stub.resolve(strict=True))
            self.assertFalse((stub.parent / "libcuda.so.1").exists())

    def test_cuda_driver_stub_preparation_fails_without_external_stub(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            cuda_root = scratch / "cuda"
            cuda_root.mkdir()
            result = subprocess.run(
                ["bash", str(CUDA_STUB_PREP), str(cuda_root), str(scratch / "runtime")],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "CUDA driver stub is required for archive smoke validation",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
