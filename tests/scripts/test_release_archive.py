#!/usr/bin/env python3
"""Executable W7 contract for extracted release archives and sidecars."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "validate-release-archive.py"
PACKAGER = ROOT / "scripts" / "package-server.py"
FIXTURE = ROOT / "tests/scripts/fixtures/release_manifest/v1/cpu-manifest.json"
CUDA_FIXTURE = ROOT / "tests/scripts/fixtures/release_manifest/v1/cuda-manifest.json"


def load_tool():
    spec = importlib.util.spec_from_file_location("validate_release_archive", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_packager():
    spec = importlib.util.spec_from_file_location("package_server", PACKAGER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PACKAGER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 W7 fixture")
class ReleaseArchiveContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_tool()
        cls.packager = load_packager()

    def manifest(self, executable: Path) -> dict[str, object]:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        dynamic = subprocess.run(
            ["readelf", "-dW", str(executable)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        needed = self.tool.parse_elf_needed(dynamic)
        manifest["dependencies"] = [
            {
                "name": name,
                "version": "system",
                "kind": "library",
                "linkage": "dynamic",
                "bundled": False,
                "role": "runtime",
            }
            for name in needed
        ]
        return manifest

    def make_release(self, scratch: Path, mutate=None) -> tuple[Path, Path, Path]:
        stage = scratch / "stage"
        (stage / "bin").mkdir(parents=True)
        executable = stage / "bin/vllm-server"
        shutil.copy2("/bin/true", executable)
        executable.chmod(0o755)
        manifest = self.manifest(executable)
        version = {
            "version": manifest["artifact"]["version"],
            "commit": manifest["build"]["source_commit"],
            "artifact_id": manifest["artifact"]["id"],
            "backend": manifest["backend"]["name"],
            "host_os": manifest["host"]["os"],
            "host_arch": manifest["host"]["arch"],
            "host_abi": manifest["host"]["abi"],
            "source_clean": "true",
            "c_abi_version": "17",
        }
        binary_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": manifest["artifact"]["id"],
            "documentNamespace": "https://github.com/mudler/vllm.cpp/spdx/fixture",
            "creationInfo": {
                "created": "1970-01-01T00:00:00Z",
                "creators": ["Organization: vllm.cpp"],
            },
            "files": [
                {
                    "fileName": "./bin/vllm-server",
                    "SPDXID": "SPDXRef-File-vllm-server",
                    "checksums": [{"algorithm": "SHA256", "checksumValue": binary_digest}],
                    "licenseConcluded": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                }
            ],
        }
        (stage / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "VERSION").write_text(
            "".join(f"{key}={value}\n" for key, value in version.items()), encoding="utf-8"
        )
        (stage / "sbom.spdx.json").write_text(
            json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "THIRD_PARTY_NOTICES").write_text(
            "vllm.cpp release dependency notices\n", encoding="utf-8"
        )
        licenses = stage / "share/licenses/vllm.cpp"
        licenses.mkdir(parents=True)
        shutil.copy2(ROOT / "LICENSE", licenses / "LICENSE")
        if mutate is not None:
            mutate(stage, manifest, sbom)
        archive = scratch / (
            f"vllm.cpp-{manifest['artifact']['version']}-"
            f"{manifest['artifact']['id']}.tar.gz"
        )
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(stage.rglob("*")):
                bundle.add(path, arcname=path.relative_to(stage))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = Path(f"{archive}.sha256")
        checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
        provenance = Path(f"{archive}.provenance.json")
        provenance.write_text(
            json.dumps(
                {
                    "_type": "https://in-toto.io/Statement/v1",
                    "predicateType": "https://slsa.dev/provenance/v1",
                    "subject": [{"name": archive.name, "digest": {"sha256": digest}}],
                    "predicate": {
                        "buildDefinition": {
                            "externalParameters": {
                                "artifact_id": manifest["artifact"]["id"],
                                "source_commit": manifest["build"]["source_commit"],
                            }
                        }
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return archive, checksum, provenance

    def run_validator(self, archive: Path, checksum: Path, provenance: Path):
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--archive",
                str(archive),
                "--archive-format",
                "zip" if archive.name.endswith(".zip") else "tar.gz",
                "--checksum",
                str(checksum),
                "--provenance",
                str(provenance),
                "--repo-root",
                str(ROOT),
                "--skip-version-smoke",
                "--forbid-path",
                str(archive.parent / "build-secret"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_valid_extracted_cpu_archive_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_release(Path(temporary))
            result = self.run_validator(*paths)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unversioned_archive_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            archive, checksum, provenance = self.make_release(scratch)
            artifact_id = self.manifest(scratch / "stage/bin/vllm-server")["artifact"]["id"]
            unversioned = scratch / f"{artifact_id}.tar.gz"
            archive.rename(unversioned)
            checksum.unlink()
            statement = json.loads(provenance.read_text(encoding="utf-8"))
            provenance.unlink()
            statement["subject"][0]["name"] = unversioned.name
            digest = hashlib.sha256(unversioned.read_bytes()).hexdigest()
            unversioned_checksum = Path(f"{unversioned}.sha256")
            unversioned_checksum.write_text(
                f"{digest}  {unversioned.name}\n", encoding="utf-8"
            )
            unversioned_provenance = Path(f"{unversioned}.provenance.json")
            unversioned_provenance.write_text(
                json.dumps(statement), encoding="utf-8"
            )
            result = self.run_validator(
                unversioned, unversioned_checksum, unversioned_provenance
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical archive name", result.stdout + result.stderr)

    def test_extra_source_object_and_missing_metadata_fail(self) -> None:
        mutations = (
            lambda stage, manifest, sbom: (stage / "source.cpp").write_text("int x;"),
            lambda stage, manifest, sbom: (stage / "object.o").write_bytes(b"ELF"),
            lambda stage, manifest, sbom: (stage / "VERSION").unlink(),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                paths = self.make_release(Path(temporary), mutation)
                result = self.run_validator(*paths)
                self.assertNotEqual(result.returncode, 0)

    def test_wrong_checksum_and_provenance_subject_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, checksum, provenance = self.make_release(Path(temporary))
            checksum.write_text(f"{'0' * 64}  {archive.name}\n", encoding="utf-8")
            self.assertNotEqual(self.run_validator(archive, checksum, provenance).returncode, 0)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
            statement = json.loads(provenance.read_text())
            statement["subject"][0]["digest"]["sha256"] = "f" * 64
            provenance.write_text(json.dumps(statement), encoding="utf-8")
            self.assertNotEqual(self.run_validator(archive, checksum, provenance).returncode, 0)

    def test_sbom_binary_digest_and_version_manifest_agreement_are_live(self) -> None:
        def bad_sbom(stage, manifest, sbom):
            sbom["files"][0]["checksums"][0]["checksumValue"] = "0" * 64
            (stage / "sbom.spdx.json").write_text(json.dumps(sbom), encoding="utf-8")

        def bad_version(stage, manifest, sbom):
            path = stage / "VERSION"
            path.write_text(path.read_text().replace("version=0.1.0-test", "version=9.9.9"))

        for mutation in (bad_sbom, bad_version):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                paths = self.make_release(Path(temporary), mutation)
                result = self.run_validator(*paths)
                self.assertNotEqual(result.returncode, 0)

    def test_prerelease_manifest_rejects_numeric_only_server_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "bin/vllm-server"
            server.parent.mkdir(parents=True)
            server.write_bytes(b"ELF fixture")
            (root / "VERSION").write_text(
                "version=0.0.3-pre.1\nc_abi_version=17\n", encoding="utf-8"
            )
            manifest = self.manifest(Path("/bin/true"))
            manifest["artifact"]["version"] = "0.0.3-pre.1"

            def fake_run(command):
                if command[0] == "file":
                    return 0, "ELF 64-bit LSB executable, x86-64"
                if command[0] == "readelf":
                    return 0, ""
                if command[0] == "ldd":
                    return 0, "statically linked"
                if command[-1] == "--help":
                    return 0, "usage: vllm-server"
                if command[-1] == "--version":
                    return 0, "vllm.cpp 0.0.3 c-abi=17"
                raise AssertionError(command)

            with mock.patch.object(self.tool.shutil, "which", return_value="tool"), \
                 mock.patch.object(self.tool, "run", side_effect=fake_run):
                errors = self.tool.inspect_linux(server, manifest, [], False)
            self.assertIn(
                "extracted vllm-server --version disagrees with VERSION/manifest",
                errors,
            )

    def test_build_path_and_undeclared_dynamic_dependency_fail(self) -> None:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        errors = self.tool.validate_linux_dynamic(
            manifest,
            ["libc.so.6", "libundeclared.so.1"],
            ["/tmp/build-secret/lib"],
            "/lib64/ld-linux-x86-64.so.2",
            ["/tmp/build-secret"],
        )
        self.assertTrue(any("undeclared" in error for error in errors), errors)
        self.assertTrue(any("RPATH" in error for error in errors), errors)
        self.assertEqual(
            self.tool.validate_macho_install_id("@rpath/libmlx.dylib", ["/tmp/build"]),
            [],
        )
        self.assertTrue(
            self.tool.validate_macho_install_id("/tmp/build/libmlx.dylib", ["/tmp/build"])
        )

    def test_literal_static_policy_rejects_any_dynamic_boundary(self) -> None:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        manifest["artifact"]["static_boundary"] = "literal-static"
        manifest["host"]["abi"] = "musl"
        manifest["dependencies"] = []
        self.assertEqual(self.tool.validate_linux_dynamic(manifest, [], [], "", []), [])
        errors = self.tool.validate_linux_dynamic(
            manifest, ["libc.so"], ["$ORIGIN/lib"], "/lib/ld-musl-x86_64.so.1", []
        )
        self.assertTrue(any("literal-static" in error for error in errors), errors)

    def test_static_ldd_wording_covers_gnu_and_musl(self) -> None:
        for output in (
            "not a dynamic executable\n",
            "statically linked\n",
            "/lib/ld-musl-x86_64.so.1: ./vllm-server: Not a valid dynamic program\n",
        ):
            with self.subTest(output=output):
                self.assertTrue(self.tool.ldd_reports_static(output))
        self.assertFalse(self.tool.ldd_reports_static("libc.so.6 => /lib/libc.so.6\n"))

    def test_large_file_scanner_catches_a_credential_across_chunk_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large-binary"
            path.write_bytes(b"x" * (1024 * 1024 - 8) + b"AKIA0123456789ABCDEF")
            found, aws_key = self.tool.scan_file(path, [b"not-present"])
            self.assertEqual(found, set())
            self.assertTrue(aws_key)

    def test_cuda_inventory_requires_all_sms_and_exact_aot_symbols(self) -> None:
        manifest = json.loads(CUDA_FIXTURE.read_text(encoding="utf-8"))
        images = [f"sm_{sm}" for sm in self.tool.PRIMARY_CUDA_SMS]
        symbols = [f"vt_aot_sm_{sm}_gdn_default" for sm in self.tool.AOT_SMS]
        self.assertEqual(self.tool.validate_cuda_inventory(manifest, images, symbols), [])
        for removed in self.tool.PRIMARY_CUDA_SMS:
            with self.subTest(missing_sm=removed):
                errors = self.tool.validate_cuda_inventory(
                    manifest, [image for image in images if image != f"sm_{removed}"], symbols
                )
                self.assertTrue(any(removed in error for error in errors), errors)
        errors = self.tool.validate_cuda_inventory(
            manifest,
            images,
            [symbol.replace("sm_80", "sm_87") for symbol in symbols],
        )
        self.assertTrue(any("AOT" in error for error in errors), errors)

    def test_macho_dependencies_install_names_and_rpaths_fail_closed(self) -> None:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        manifest["host"].update({"os": "macos", "arch": "aarch64", "abi": "macos"})
        manifest["backend"]["name"] = "mlx"
        manifest["dependencies"] = [
            {"name": "libmlx.dylib", "linkage": "dynamic"},
            {"name": "libc++.1.dylib", "linkage": "dynamic"},
            {"name": "libSystem.B.dylib", "linkage": "dynamic"},
            {"name": "Metal.framework", "linkage": "external"},
            {"name": "Foundation.framework", "linkage": "external"},
        ]
        dependencies = [
            "@rpath/libmlx.dylib",
            "/usr/lib/libc++.1.dylib",
            "/usr/lib/libSystem.B.dylib",
            "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
            "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation",
        ]
        self.assertEqual(
            self.tool.validate_macho_dynamic(
                manifest, dependencies, ["@loader_path/../lib"], ["/tmp/build-secret"]
            ),
            [],
        )
        errors = self.tool.validate_macho_dynamic(
            manifest,
            [*dependencies, "/tmp/build-secret/libbad.dylib"],
            ["/tmp/build-secret"],
            ["/tmp/build-secret"],
        )
        self.assertTrue(any("install name" in error for error in errors), errors)
        self.assertTrue(any("RPATH" in error for error in errors), errors)

    def test_tar_traversal_is_rejected_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            archive = scratch / "bad.tar.gz"
            source = scratch / "payload"
            source.write_text("escape", encoding="utf-8")
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source, arcname="../escape")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            checksum = Path(f"{archive}.sha256")
            checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
            provenance = Path(f"{archive}.provenance.json")
            provenance.write_text(
                json.dumps(
                    {
                        "_type": "https://in-toto.io/Statement/v1",
                        "predicateType": "https://slsa.dev/provenance/v1",
                        "subject": [
                            {"name": archive.name, "digest": {"sha256": digest}}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = self.run_validator(archive, checksum, provenance)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe archive path", result.stdout + result.stderr)

    def test_zip_is_deterministic_and_rejects_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            (stage / "bin").mkdir(parents=True)
            (stage / "bin/vllm-server.exe").write_bytes(b"PE fixture")
            (stage / "VERSION").write_text("version=fixture\n", encoding="utf-8")
            first = root / "first.zip"
            second = root / "second.zip"
            self.packager.write_archive(stage, first, 315532800, "zip")
            self.packager.write_archive(stage, second, 315532800, "zip")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as bundle:
                names = bundle.namelist()
                self.assertEqual(names, sorted(names))
                self.assertTrue(all("\\" not in name for name in names))
            (stage / "link").symlink_to("VERSION")
            with self.assertRaises(ValueError):
                self.packager.write_archive(stage, root / "linked.zip", 315532800, "zip")

    def test_archive_writer_requires_an_explicit_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            stage.mkdir()
            with self.assertRaises(TypeError):
                self.packager.write_archive(stage, root / "implicit.tar.gz", 0)

    def test_zip_traversal_drive_backslash_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../escape", "C:/escape", "..\\escape"):
                with self.subTest(name=name):
                    archive = root / (hashlib.sha256(name.encode()).hexdigest() + ".zip")
                    with zipfile.ZipFile(archive, "w") as bundle:
                        bundle.writestr(name, b"escape")
                    errors = self.tool.safe_extract(archive, root / "extract", "zip")
                    self.assertTrue(any("unsafe archive path" in error for error in errors), errors)
            archive = root / "symlink.zip"
            info = zipfile.ZipInfo("bin/link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(info, "vllm-server.exe")
            errors = self.tool.safe_extract(archive, root / "extract-link", "zip")
            self.assertTrue(any("link" in error for error in errors), errors)

    def test_explicit_archive_format_controls_extraction_and_must_match_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "payload.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("VERSION", b"ok")
            errors = self.tool.safe_extract(archive, root / "wrong", "tar.gz")
            self.assertTrue(any("format" in error for error in errors), errors)
            errors = self.tool.safe_extract(archive, root / "right", "zip")
            self.assertEqual(errors, [])

    def test_zip_noncanonical_collisions_and_special_types_fail_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mutations = (
                ("bin//vllm-server.exe", None),
                ("bin/./vllm-server.exe", None),
                ("bin/vllm-server.exe", 0o020666 << 16),
            )
            for index, (name, attributes) in enumerate(mutations):
                archive = root / f"bad-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    if attributes is not None:
                        info.external_attr = attributes
                    bundle.writestr(info, b"bad")
                    if index == 0:
                        bundle.writestr("bin/vllm-server.exe", b"collision")
                destination = root / f"extract-{index}"
                errors = self.tool.safe_extract(archive, destination, "zip")
                self.assertTrue(errors, (name, errors))
                self.assertEqual(list(destination.rglob("*")) if destination.exists() else [], [])

    def test_zip_directory_markers_must_match_unix_type_bits_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("bin/", stat.S_IFREG | 0o755),
                ("other", stat.S_IFDIR | 0o755),
            )
            for index, (name, mode) in enumerate(cases):
                archive = root / f"type-mismatch-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = mode << 16
                    bundle.writestr(info, b"")
                destination = root / f"type-mismatch-out-{index}"
                errors = self.tool.safe_extract(archive, destination, "zip")
                self.assertTrue(any("type" in error for error in errors), errors)
                self.assertFalse(destination.exists())

    def test_zip_entries_without_unix_type_bits_use_the_path_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "portable-types.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("python-file", b"python")
                bundle.writestr("python-dir/", b"")
                for name, content in (("windows-file", b"windows"), ("windows-dir/", b"")):
                    info = zipfile.ZipInfo(name)
                    info.create_system = 0
                    info.external_attr = 0
                    bundle.writestr(info, content)
            destination = root / "portable-types-out"
            self.assertEqual(self.tool.safe_extract(archive, destination, "zip"), [])
            self.assertEqual((destination / "python-file").read_bytes(), b"python")
            self.assertEqual((destination / "windows-file").read_bytes(), b"windows")
            self.assertTrue((destination / "python-dir").is_dir())
            self.assertTrue((destination / "windows-dir").is_dir())

    def test_zip_windows_aliases_fail_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("bin/X", "BIN/x"), ("bin/name. ",), ("bin/name:stream",),
                ("bin/CON",), ("bin/prn.txt",), ("bin/AUX",), ("bin/nul.log",),
                ("bin/COM1.txt",), ("bin/lpt9",),
            )
            for index, names in enumerate(cases):
                archive = root / f"alias-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    for name in names:
                        bundle.writestr(name, b"bad")
                destination = root / f"alias-out-{index}"
                errors = self.tool.safe_extract(archive, destination, "zip")
                self.assertTrue(errors, names)
                self.assertFalse(destination.exists())

    def test_pe_audit_rejects_wrong_machine_imports_crt_debug_and_paths(self) -> None:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        manifest["host"].update({"os": "windows", "arch": "x86_64", "abi": "msvc"})
        manifest["dependencies"] = [
            {"name": name, "linkage": "dynamic"}
            for name in ("KERNEL32.dll",)
        ]
        self.assertEqual(
            self.tool.validate_pe_audit(
                manifest, "8664", ["KERNEL32.dll"], [], []
            ),
            [],
        )
        errors = self.tool.validate_pe_audit(
            manifest,
            "14C",
            ["KERNEL32.dll", "msys-2.0.dll", "VCRUNTIME140D.dll"],
            [r"C:\\agent\\_work\\build\\vllm-server.pdb"],
            [r"C:\\agent\\_work\\build"],
        )
        joined = "\n".join(errors)
        for needle in ("AMD64", "undeclared", "MSYS", "debug CRT", "debug path", "build path"):
            self.assertIn(needle, joined)

        api_crt = "api-ms-win-crt-runtime-l1-1-0.dll"
        errors = self.tool.validate_pe_audit(
            manifest, "8664", ["KERNEL32.dll", api_crt], [], []
        )
        self.assertTrue(any("static-CRT" in error for error in errors), errors)
        api_manifest = json.loads(json.dumps(manifest))
        api_manifest["dependencies"].append({"name": "api-ms-win-core-file-l1-1-0.dll", "linkage": "dynamic"})
        self.assertEqual(
            self.tool.validate_pe_audit(
                api_manifest, "8664", ["KERNEL32.dll", "api-ms-win-core-file-l1-1-0.dll"], [], []
            ),
            [],
        )

    def test_pe_audit_allows_declared_non_system_import(self) -> None:
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        manifest["host"].update({"os": "windows", "arch": "x86_64", "abi": "msvc"})
        manifest["dependencies"] = [
            {"name": name, "linkage": "dynamic"}
            for name in ("KERNEL32.dll", "LIBCRYPTO-3-x64.dll", "LIBSSL-3-x64.dll")
        ]
        self.assertEqual(
            self.tool.validate_pe_audit(
                manifest,
                "8664",
                ["KERNEL32.dll", "LIBCRYPTO-3-x64.dll", "LIBSSL-3-x64.dll"],
                [],
                [],
            ),
            [],
        )
        undeclared = json.loads(json.dumps(manifest))
        undeclared["dependencies"] = [
            {"name": "KERNEL32.dll", "linkage": "dynamic"},
        ]
        errors = self.tool.validate_pe_audit(
            undeclared,
            "8664",
            ["KERNEL32.dll", "LIBCRYPTO-3-x64.dll"],
            [],
            [],
        )
        self.assertTrue(any("non-system PE import" in e for e in errors), errors)
        crt_declared = json.loads(json.dumps(manifest))
        crt_declared["dependencies"].append(
            {"name": "VCRUNTIME140.dll", "linkage": "dynamic"}
        )
        errors = self.tool.validate_pe_audit(
            crt_declared,
            "8664",
            ["KERNEL32.dll", "VCRUNTIME140.dll"],
            [],
            [],
        )
        self.assertTrue(any("static-CRT" in e for e in errors), errors)

    def test_windows_layout_rejects_every_lib_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("lib/foo.lib", "LIB/arbitrary.data", "lib/Foo.LIB"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"inventoried but forbidden")
            errors = self.tool.validate_windows_layout(root)
            self.assertEqual(len(errors), 3, errors)


if __name__ == "__main__":
    unittest.main()
