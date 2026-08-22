#!/usr/bin/env python3
"""W9 CPU release metadata and packaged-sidecar contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "scripts/release_metadata.py"
PACKAGE = ROOT / "scripts/package-server.py"
VALIDATOR = ROOT / "scripts/validate-release-archive.py"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passed(command: str) -> dict[str, str]:
    return {
        "command": command,
        "reason": "",
        "result": "exit 0",
        "state": "passed",
        "url": "https://github.com/mudler/vllm.cpp/actions/runs/1",
    }


class ReleaseMetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = load(METADATA, "release_metadata")
        cls.package = load(PACKAGE, "package_server")
        # #1487: the fixture stages the HOST's /bin/true as the server, so the
        # declared artifact must follow the staged ELF's machine instead of a
        # hardcoded x86_64 — on every other host the fixture builds a manifest
        # that lies about its own payload and the validator rightly refuses it
        # (ELF host architecture does not match manifest). CPU_TIER_POLICY
        # carries both linux CPU entries, so the tier report follows the same
        # arch and the whole contract runs identically per host.
        machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(
            platform.machine(), platform.machine()
        )
        if machine not in cls.metadata.release_manifest.CPU_TIER_POLICY:
            raise unittest.SkipTest(
                f"host machine {machine!r} has no declared linux CPU artifact "
                "to fixture"
            )
        cls.machine = machine
        cls.artifact_id = f"linux-{machine}-glibc-cpu"
        cls.policy = cls.metadata.release_manifest.CPU_TIER_POLICY[machine]

    def fixture(self, scratch: Path):
        build = scratch / "build"
        stage = scratch / "stage"
        metadata = scratch / "metadata"
        build.mkdir(parents=True)
        (stage / "bin").mkdir(parents=True)
        shutil.copy2("/bin/true", stage / "bin/vllm-server")
        (stage / "bin/vllm-server").chmod(0o755)
        cache = {
            "MLX_ROOT": ("PATH", ""),
            "VLLM_CPP_BUILD_EXAMPLES": ("BOOL", "ON"),
            "VLLM_CPP_BUILD_TESTS": ("BOOL", "ON"),
            "VLLM_CPP_CUDA": ("BOOL", "OFF"),
            "VLLM_CPP_CUDA_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_HIP": ("BOOL", "OFF"),
            "VLLM_CPP_HIP_ARCHITECTURES": ("STRING", ""),
            "VLLM_CPP_LITERAL_STATIC": ("BOOL", "OFF"),
            "VLLM_CPP_METAL": ("STRING", "OFF"),
            "VLLM_CPP_MLX": ("BOOL", "OFF"),
            "VLLM_CPP_SERVER": ("BOOL", "ON"),
            "VLLM_CPP_TRITON": ("STRING", "OFF"),
            "VLLM_CPP_VULKAN": ("STRING", "OFF"),
        }
        (build / "CMakeCache.txt").write_text(
            "".join(f"{key}:{kind}={value}\n" for key, (kind, value) in cache.items()),
            encoding="utf-8",
        )
        policy = self.policy
        report = {
            "commands": [f"VT_CPU_MATMUL_TIER={name} test_ops_matmul_elem" for name in policy["tiers"]],
            "schema": "vllm.cpp.cpu-tier-report.v1",
            "selected_tier": policy["tiers"][-1],
            "tiers": {name: passed(f"force {name}") for name in policy["tiers"]},
        }
        report_path = scratch / "tier-report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        args = argparse.Namespace(
            abi_version="2.31",
            artifact_id=self.artifact_id,
            build_dir=build,
            c_abi_version=17,
            channel="stable",
            compiler="GNU C++ 13.2.0",
            evidence_url="https://github.com/mudler/vllm.cpp/actions/runs/1",
            output_dir=metadata,
            repo_root=ROOT,
            source_clean=True,
            source_commit=SHA,
            stage_dir=stage,
            tier_report=report_path,
            toolchain="cmake-3.30+ninja-1.12",
            version="0.0.1",
        )
        return args, report

    def test_stable_cpu_metadata_is_schema_valid_and_byte_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = self.fixture(Path(temporary))
            manifest = self.metadata.prepare_cpu_metadata(args)
            self.assertEqual(manifest["artifact"]["c_abi_version"], 17)
            self.assertEqual(manifest["artifact"]["channel"], "stable")
            self.assertEqual(
                [tier["execution_evidence"]["state"] for tier in manifest["cpu"]["compiled_tiers"]],
                ["passed"] * len(self.policy["tiers"]),
            )
            sbom = json.loads((args.output_dir / "sbom.spdx.json").read_text())
            self.assertEqual(sbom["spdxVersion"], "SPDX-2.3")
            self.assertTrue((args.output_dir / "share/licenses/vllm.cpp/LICENSE").is_file())

    def test_stable_metadata_refuses_a_missing_or_unexecuted_tier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, report = self.fixture(Path(temporary))
            deprived = self.policy["tiers"][-1]
            del report["tiers"][deprived]
            args.tier_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.metadata.prepare_cpu_metadata(args)
            args, report = self.fixture(Path(temporary) / "second")
            report["tiers"][deprived] = {
                "command": "",
                "reason": f"runner lacks {deprived}",
                "result": "",
                "state": "absent",
                "url": "",
            }
            args.tier_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.metadata.prepare_cpu_metadata(args)

    def test_metadata_is_installed_archived_and_validated_with_final_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            args, _ = self.fixture(scratch)
            self.metadata.prepare_cpu_metadata(args)
            packaged = scratch / "packaged"
            shutil.copytree(args.stage_dir, packaged)
            self.package.install_metadata(args.output_dir, packaged)
            archive = scratch / f"vllm.cpp-0.0.1-{args.artifact_id}.tar.gz"
            self.package.write_archive(packaged, archive, 0, "tar.gz")
            self.package.write_archive_sidecars(archive, packaged)
            result = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--archive",
                    str(archive),
                    "--archive-format",
                    "tar.gz",
                    "--checksum",
                    f"{archive}.sha256",
                    "--provenance",
                    f"{archive}.provenance.json",
                    "--repo-root",
                    str(ROOT),
                    "--skip-version-smoke",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            first = Path(f"{archive}.sha256").read_text()
            self.package.write_archive(packaged, archive, 0, "tar.gz")
            self.package.write_archive_sidecars(archive, packaged)
            self.assertEqual(first, Path(f"{archive}.sha256").read_text())

    def test_metadata_install_rejects_unallowlisted_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            args, _ = self.fixture(scratch)
            self.metadata.prepare_cpu_metadata(args)
            (args.output_dir / "model.safetensors").write_bytes(b"forbidden")
            with self.assertRaises(SystemExit):
                self.package.install_metadata(args.output_dir, scratch / "stage-copy")


if __name__ == "__main__":
    unittest.main()
