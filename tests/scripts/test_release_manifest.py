#!/usr/bin/env python3
"""Executable W5 contract for the versioned binary-release manifest."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "release_manifest.py"
SCHEMA = ROOT / "release" / "manifest-v1.schema.json"
FIXTURES = ROOT / "tests" / "scripts" / "fixtures" / "release_manifest" / "v1"
PRIMARY_SMS = ["80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a"]
AOT_SMS = {"80", "86", "89", "90a", "100a", "121a"}
EVIDENCE_KEYS = (
    "build",
    "archive_smoke",
    "dependency_audit",
    "runtime",
    "correctness",
    "performance",
)


def evidence(state: str) -> dict[str, str]:
    values = {
        "absent": {
            "reason": "synthetic fixture has not run this gate",
            "command": "",
            "result": "",
            "url": "",
        },
        "not-applicable": {
            "reason": "synthetic cross-build has no matching runtime host",
            "command": "",
            "result": "",
            "url": "",
        },
        "failed": {
            "reason": "synthetic dependency audit failure",
            "command": "lddtree bin/vllm-server",
            "result": "exit 1: undeclared libsynthetic.so",
            "url": "https://example.invalid/evidence/failed",
        },
        "passed": {
            "reason": "",
            "command": "cmake --build build --target vllm",
            "result": "exit 0",
            "url": "https://example.invalid/evidence/passed",
        },
    }
    return {"state": state, **values[state]}


def flags(backend: str, sms: list[str] | None = None) -> dict[str, object]:
    return {
        "MLX_ROOT": "",
        "VLLM_CPP_BUILD_EXAMPLES": True,
        "VLLM_CPP_BUILD_TESTS": True,
        "VLLM_CPP_CUDA": backend == "cuda",
        "VLLM_CPP_CUDA_ARCHITECTURES": list(sms or []),
        "VLLM_CPP_HIP": False,
        "VLLM_CPP_HIP_ARCHITECTURES": [],
        "VLLM_CPP_LITERAL_STATIC": False,
        "VLLM_CPP_METAL": backend in {"metal", "mlx"},
        "VLLM_CPP_MLX": backend == "mlx",
        "VLLM_CPP_SERVER": True,
        "VLLM_CPP_TRITON": backend == "cuda",
        "VLLM_CPP_VULKAN": backend == "vulkan",
    }


def build_metadata(resolved_flags: dict[str, object]) -> dict[str, object]:
    return {
        "source_commit": "0123456789abcdef0123456789abcdef01234567",
        "source_clean": True,
        "compiler": "GNU C++ 13.2.0",
        "toolchain": "cmake-3.30+ninja-1.12",
        "resolved_cmake_options": copy.deepcopy(resolved_flags),
        "test_commands": ["cmake --build build --target vllm", "ctest --test-dir build"],
    }


def supply_chain() -> dict[str, object]:
    return {
        "archive_checksum": evidence("absent"),
        "sbom": evidence("absent"),
        "provenance": evidence("absent"),
        "licenses": evidence("absent"),
    }


def cpu_facts() -> dict[str, object]:
    resolved_flags = flags("cpu")
    tiers = [
        {
            "name": "portable-sse2",
            "kernel_families": ["matmul-elem-f32-bf16"],
            "required_cpu_bits": ["sse2"],
            "required_os_state": [],
            "execution_evidence": evidence("passed"),
        },
        {
            "name": "sse2-f16c",
            "kernel_families": ["matmul-elem-f16"],
            "required_cpu_bits": ["avx", "f16c", "osxsave", "sse2"],
            "required_os_state": ["xcr0:xmm", "xcr0:ymm"],
            "execution_evidence": evidence("failed"),
        },
        {
            "name": "avx2-f16c",
            "kernel_families": ["matmul-elem-f32-bf16-f16"],
            "required_cpu_bits": ["avx", "avx2", "f16c", "osxsave", "sse2"],
            "required_os_state": ["xcr0:xmm", "xcr0:ymm"],
            "execution_evidence": evidence("absent"),
        },
        {
            "name": "avx512f",
            "kernel_families": ["matmul-elem-f32-bf16"],
            "required_cpu_bits": [
                "avx",
                "avx2",
                "avx512bw",
                "avx512f",
                "avx512vl",
                "f16c",
                "osxsave",
                "sse2",
            ],
            "required_os_state": [
                "xcr0:xmm",
                "xcr0:ymm",
                "xcr0:opmask",
                "xcr0:zmm_hi256",
                "xcr0:hi16_zmm",
            ],
            "execution_evidence": evidence("not-applicable"),
        },
    ]
    return {
        "artifact": {
            "id": "linux-x86_64-glibc-cpu",
            "version": "0.1.0-test",
            "c_abi_version": 17,
            "channel": "preview",
            "kind": "primary",
            "static_boundary": "static-core",
        },
        "host": {
            "os": "linux",
            "arch": "x86_64",
            "abi": "glibc",
            "abi_version": "2.31",
        },
        "backend": {
            "name": "cpu",
            "flags": resolved_flags,
            "gpu_driver_boundary": "not-applicable",
        },
        "dependencies": [
            {
                "name": "glibc",
                "version": "2.31",
                "kind": "library",
                "linkage": "dynamic",
                "bundled": False,
                "role": "runtime",
            }
        ],
        "cpu": {
            "baseline": "portable-sse2",
            "selected_tier": "portable-sse2",
            "compiled_tiers": tiers,
        },
        "build": build_metadata(resolved_flags),
        "supply_chain": supply_chain(),
        "evidence": {
            "build": evidence("passed"),
            "archive_smoke": evidence("passed"),
            "dependency_audit": evidence("passed"),
            "runtime": evidence("not-applicable"),
            "correctness": evidence("absent"),
            "performance": evidence("absent"),
        },
    }


def cuda_facts() -> dict[str, object]:
    resolved_flags = flags("cuda", PRIMARY_SMS)
    sm_evidence = []
    for sm in PRIMARY_SMS:
        available = sm in AOT_SMS
        sm_evidence.append(
            {
                "sm": sm,
                "aot_available": available,
                "portable_fallback": not available,
                "aot_evidence": evidence("absent" if available else "not-applicable"),
                "runtime_evidence": evidence("absent"),
            }
        )
    return {
        "artifact": {
            "id": "linux-x86_64-glibc-cuda",
            "version": "0.1.0-test",
            "c_abi_version": 17,
            "channel": "preview",
            "kind": "primary",
            "static_boundary": "static-core",
        },
        "host": {
            "os": "linux",
            "arch": "x86_64",
            "abi": "glibc",
            "abi_version": "2.31",
        },
        "backend": {
            "name": "cuda",
            "flags": resolved_flags,
            "gpu_driver_boundary": "external-host-never-bundled",
        },
        "dependencies": [
            {
                "name": "nvidia-driver",
                "version": ">=580",
                "kind": "driver",
                "linkage": "external",
                "bundled": False,
                "role": "external-runtime",
            },
            {
                "name": "glibc",
                "version": "2.31",
                "kind": "library",
                "linkage": "dynamic",
                "bundled": False,
                "role": "runtime",
            },
        ],
        "cuda": {"compiled_sms": list(PRIMARY_SMS), "sm_evidence": sm_evidence},
        "build": build_metadata(resolved_flags),
        "supply_chain": supply_chain(),
        "evidence": {
            key: (
                evidence("passed")
                if key in {"build", "archive_smoke", "dependency_audit"}
                else evidence("absent")
            )
            for key in EVIDENCE_KEYS
        },
    }


def load_tool():
    spec = importlib.util.spec_from_file_location("release_manifest", TOOL)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = None
        cls.schema = None

    def setUp(self) -> None:
        self.assertTrue(SCHEMA.is_file(), f"missing versioned schema: {SCHEMA}")
        self.assertTrue(TOOL.is_file(), f"missing build-time generator: {TOOL}")
        if self.__class__.tool is None:
            self.__class__.tool = load_tool()
            self.__class__.schema = self.tool.load_schema(SCHEMA)

    def generated(self, facts: dict[str, object]) -> dict[str, object]:
        manifest = self.tool.generate_manifest(copy.deepcopy(facts), ROOT, self.schema)
        errors = self.tool.validate_manifest(manifest, self.schema, ROOT)
        self.assertEqual(errors, [])
        return manifest

    def assert_invalid(self, manifest: dict[str, object], needle: str = "") -> None:
        errors = self.tool.validate_manifest(manifest, self.schema, ROOT)
        self.assertTrue(errors, "mutation unexpectedly validated")
        if needle:
            self.assertTrue(any(needle in error for error in errors), errors)

    def test_cpu_and_cuda_generation_is_deterministic_and_matches_goldens(self) -> None:
        for stem, facts in (("cpu", cpu_facts()), ("cuda", cuda_facts())):
            manifest = self.generated(facts)
            canonical = self.tool.canonical_json(manifest)
            self.assertEqual(canonical, self.tool.canonical_json(manifest))
            self.assertEqual(canonical, (FIXTURES / f"{stem}-manifest.json").read_text())
            self.assertEqual(facts, json.loads((FIXTURES / f"{stem}-input.json").read_text()))

    def test_cli_generate_and_validate_use_only_stdlib_build_time_tooling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "manifest.json"
            generated = subprocess.run(
                [sys.executable, str(TOOL), "generate", "--input", str(FIXTURES / "cpu-input.json"), "--output", str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            self.assertEqual(output.read_text(), (FIXTURES / "cpu-manifest.json").read_text())
            validated = subprocess.run(
                [sys.executable, str(TOOL), "validate", str(output)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)

    def test_w5_suite_registration_is_fail_closed(self) -> None:
        checker_path = ROOT / "scripts" / "check-release-binary-contract.py"
        spec = importlib.util.spec_from_file_location("release_contract_checker_w5", checker_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        preflight = (ROOT / "scripts" / "agent-preflight.sh").read_text()
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        self.assertEqual(checker.wiring_errors(preflight, ci), [])
        mutations = (
            (preflight.replace("  test_release_manifest\n", "", 1), ci, "preflight"),
            (preflight, ci.replace("          python3 tests/scripts/test_release_manifest.py\n", "", 1), "CI"),
            (preflight.replace("  test_release_windows_metadata\n", "", 1), ci, "Windows preflight"),
            (preflight, ci.replace("          python3 tests/scripts/test_release_windows_metadata.py\n", "", 1), "Windows CI"),
        )
        for mutated_preflight, mutated_ci, reason in mutations:
            with self.subTest(reason=reason):
                if reason.startswith("Windows"):
                    errors = checker.wiring_errors(mutated_preflight, mutated_ci)
                    self.assertTrue(any("W15 Windows metadata suite" in error for error in errors), errors)
                else:
                    errors = checker.wiring_errors(mutated_preflight, mutated_ci)
                    self.assertTrue(any("W5 manifest suite" in error for error in errors), errors)

    def test_schema_required_keys_enums_and_additional_properties_are_live(self) -> None:
        base = self.generated(cpu_facts())
        required_paths = [
            ((), "schema"), ((), "schema_version"), ((), "artifact"),
            ((), "host"), ((), "backend"), ((), "dependencies"), ((), "build"),
            ((), "supply_chain"), ((), "evidence"),
            (("artifact",), "id"), (("artifact",), "version"),
            (("artifact",), "channel"), (("artifact",), "kind"),
            (("artifact",), "static_boundary"), (("host",), "os"), (("host",), "arch"),
            (("host",), "abi"), (("host",), "abi_version"), (("backend",), "name"),
            (("backend",), "flags"), (("backend",), "resolved_features"),
            (("backend",), "gpu_driver_boundary"), (("cpu",), "baseline"),
            (("cpu",), "selected_tier"), (("cpu",), "compiled_tiers"),
            (("build",), "source_commit"), (("build",), "source_clean"),
            (("build",), "compiler"), (("build",), "toolchain"),
            (("build",), "resolved_cmake_options"), (("build",), "test_commands"),
            (("supply_chain",), "archive_checksum"), (("supply_chain",), "sbom"),
            (("supply_chain",), "provenance"), (("supply_chain",), "licenses"),
            (("dependencies", 0), "name"),
            (("dependencies", 0), "version"), (("dependencies", 0), "kind"),
            (("dependencies", 0), "linkage"), (("dependencies", 0), "bundled"),
            (("dependencies", 0), "role"),
        ]
        for path, key in required_paths:
            with self.subTest(path=path, key=key):
                mutant = copy.deepcopy(base)
                target = mutant
                for part in path:
                    target = target[part]
                del target[key]
                self.assert_invalid(mutant, "required")
        for path, value in (
            (("artifact", "channel"), "nightly"),
            (("artifact", "kind"), "universal"),
            (("artifact", "static_boundary"), "fully-static-gpu"),
            (("host", "os"), "windows"),
            (("host", "arch"), "riscv64"),
            (("backend", "name"), "rocm"),
        ):
            with self.subTest(path=path):
                mutant = copy.deepcopy(base)
                mutant[path[0]][path[1]] = value
                self.assert_invalid(mutant)
        for path in ((), ("artifact",), ("host",), ("backend",), ("backend", "flags"), ("build",), ("supply_chain",), ("cpu",), ("cpu", "compiled_tiers", 0), ("dependencies", 0), ("evidence", "build")):
            with self.subTest(unknown_at=path):
                mutant = copy.deepcopy(base)
                target = mutant
                for part in path:
                    target = target[part]
                target["unknown_w5_field"] = "must fail"
                self.assert_invalid(mutant, "unknown")

    def test_schema_type_enum_and_const_semantics_are_independently_live(self) -> None:
        self.assertTrue(
            self.tool._schema_errors("not-an-integer", {"type": "integer"}, {}),
            "schema type enforcement is not live",
        )
        self.assertTrue(
            self.tool._schema_errors("not-in-enum", {"enum": ["allowed"]}, {}),
            "schema enum enforcement is not live",
        )
        self.assertTrue(
            self.tool._schema_errors(2, {"const": 1}, {}),
            "schema const enforcement is not live",
        )
        manifest = self.generated(cpu_facts())
        manifest["schema_version"] = True
        self.assert_invalid(manifest, "schema_version")

    def test_artifact_c_abi_version_is_mandatory_and_numeric(self) -> None:
        manifest = self.generated(cpu_facts())
        self.assertEqual(manifest["artifact"]["c_abi_version"], 17)
        for value in (None, True, 0, -1, "17"):
            with self.subTest(value=value):
                mutant = copy.deepcopy(manifest)
                if value is None:
                    del mutant["artifact"]["c_abi_version"]
                else:
                    mutant["artifact"]["c_abi_version"] = value
                self.assert_invalid(mutant, "c_abi_version")

    def test_boolean_is_neither_an_integer_type_nor_integer_json_constant(self) -> None:
        self.assertFalse(
            self.tool._type_matches(True, "integer"),
            "JSON boolean must not satisfy the integer schema type",
        )
        self.assertFalse(
            self.tool._json_equal(True, 1),
            "JSON boolean and integer constants must remain distinct",
        )

    def test_evidence_states_remain_four_distinct_fail_closed_values(self) -> None:
        manifest = self.generated(cpu_facts())
        states = {manifest["evidence"][key]["state"] for key in EVIDENCE_KEYS}
        states.update(tier["execution_evidence"]["state"] for tier in manifest["cpu"]["compiled_tiers"])
        self.assertEqual(states, {"absent", "not-applicable", "failed", "passed"})
        for key in EVIDENCE_KEYS:
            for field in ("state", "reason", "command", "result", "url"):
                with self.subTest(evidence=key, missing=field):
                    mutant = copy.deepcopy(manifest)
                    del mutant["evidence"][key][field]
                    self.assert_invalid(mutant)
        semantic_mutations = (
            ("passed", "command", ""), ("passed", "result", ""), ("passed", "url", ""),
            ("failed", "reason", ""), ("failed", "command", ""), ("failed", "result", ""),
            ("absent", "reason", ""), ("not-applicable", "reason", ""),
            ("absent", "command", "should-not-run"),
            ("not-applicable", "result", "truthy-collapse"),
        )
        for state, field, value in semantic_mutations:
            with self.subTest(state=state, field=field):
                mutant = copy.deepcopy(manifest)
                mutant["evidence"]["build"] = evidence(state)
                mutant["evidence"]["build"][field] = value
                self.assert_invalid(mutant, state)
        mutant = copy.deepcopy(manifest)
        mutant["evidence"]["build"]["state"] = False
        self.assert_invalid(mutant)

    def test_identity_build_supply_chain_and_duplicate_json_are_fail_closed(self) -> None:
        manifest = self.generated(cpu_facts())
        for path, value in (
            (("artifact", "version"), ""),
            (("build", "source_commit"), "not-a-commit"),
            (("build", "compiler"), ""),
            (("build", "toolchain"), ""),
            (("build", "test_commands"), []),
        ):
            with self.subTest(path=path):
                mutant = copy.deepcopy(manifest)
                mutant[path[0]][path[1]] = value
                self.assert_invalid(mutant)
        mutant = copy.deepcopy(manifest)
        mutant["build"]["resolved_cmake_options"]["VLLM_CPP_BUILD_TESTS"] = False
        self.assert_invalid(mutant, "resolved CMake")
        duplicate_dep = copy.deepcopy(manifest)
        duplicate_dep["dependencies"].append(copy.deepcopy(duplicate_dep["dependencies"][0]))
        self.assert_invalid(duplicate_dep, "unique")
        duplicate_test = copy.deepcopy(manifest)
        duplicate_test["build"]["test_commands"].append(duplicate_test["build"]["test_commands"][0])
        self.assert_invalid(duplicate_test, "unique")
        with tempfile.TemporaryDirectory() as td:
            duplicate_json = Path(td) / "duplicate.json"
            text = self.tool.canonical_json(manifest)
            duplicate_json.write_text(text.replace('  "schema": ', '  "schema": "duplicate",\n  "schema": ', 1))
            result = subprocess.run(
                [sys.executable, str(TOOL), "validate", str(duplicate_json)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key", result.stderr)

    def test_generator_has_no_cwd_environment_locale_or_timestamp_leakage(self) -> None:
        facts = cpu_facts()
        before = self.tool.canonical_json(self.generated(facts))
        old_cwd = Path.cwd()
        old_env = os.environ.get("W5_MANIFEST_LEAK_SENTINEL")
        try:
            with tempfile.TemporaryDirectory() as td:
                os.chdir(td)
                os.environ["W5_MANIFEST_LEAK_SENTINEL"] = "must-not-appear"
                after = self.tool.canonical_json(self.generated(facts))
        finally:
            os.chdir(old_cwd)
            if old_env is None:
                os.environ.pop("W5_MANIFEST_LEAK_SENTINEL", None)
            else:
                os.environ["W5_MANIFEST_LEAK_SENTINEL"] = old_env
        self.assertEqual(after, before)
        self.assertNotIn("must-not-appear", after)
        self.assertNotIn(str(old_cwd), after)
        self.assertNotIn("timestamp", after.lower())

    def test_every_current_x86_tier_and_its_contract_are_mandatory(self) -> None:
        manifest = self.generated(cpu_facts())
        tiers = manifest["cpu"]["compiled_tiers"]
        for index, tier in enumerate(tiers):
            with self.subTest(remove=tier["name"]):
                mutant = copy.deepcopy(manifest)
                del mutant["cpu"]["compiled_tiers"][index]
                self.assert_invalid(mutant, "compiled tiers")
            for field in ("name", "kernel_families", "required_cpu_bits", "required_os_state", "execution_evidence"):
                with self.subTest(tier=tier["name"], missing=field):
                    mutant = copy.deepcopy(manifest)
                    del mutant["cpu"]["compiled_tiers"][index][field]
                    self.assert_invalid(mutant)
        mutant = copy.deepcopy(manifest)
        mutant["cpu"]["compiled_tiers"][1]["name"] = "amx-unavailable"
        self.assert_invalid(mutant, "compiled tiers")
        mutant = copy.deepcopy(manifest)
        mutant["cpu"]["selected_tier"] = "amx-unavailable"
        self.assert_invalid(mutant, "selected tier")
        for field in ("kernel_families", "required_cpu_bits"):
            mutant = copy.deepcopy(manifest)
            mutant["cpu"]["compiled_tiers"][1][field] = []
            self.assert_invalid(mutant)
        mutant = copy.deepcopy(manifest)
        mutant["cpu"]["compiled_tiers"][1]["required_os_state"] = []
        self.assert_invalid(mutant, "OS-state")

    def test_literal_static_flag_is_exactly_scoped_to_the_musl_cpu_tuple(self) -> None:
        glibc = self.generated(cpu_facts())
        glibc["backend"]["flags"]["VLLM_CPP_LITERAL_STATIC"] = True
        glibc["build"]["resolved_cmake_options"]["VLLM_CPP_LITERAL_STATIC"] = True
        self.assert_invalid(glibc, "must agree with the artifact static boundary")

        facts = cpu_facts()
        facts["artifact"].update(
            {
                "id": "linux-x86_64-musl-cpu-static",
                "channel": "experimental-preview",
                "static_boundary": "literal-static",
            }
        )
        facts["host"].update({"abi": "musl", "abi_version": "1.2.5"})
        facts["backend"]["flags"]["VLLM_CPP_LITERAL_STATIC"] = True
        facts["build"]["resolved_cmake_options"]["VLLM_CPP_LITERAL_STATIC"] = True
        facts["dependencies"] = [
            {
                "name": "musl-libc",
                "version": "1.2.5",
                "kind": "library",
                "linkage": "static",
                "bundled": True,
                "role": "build-time",
            }
        ]
        self.generated(facts)

    def test_cpu_tier_inventory_is_exact_on_x86_and_aarch64(self) -> None:
        manifest = self.generated(cpu_facts())
        for index, tier in enumerate(manifest["cpu"]["compiled_tiers"]):
            for field, extra in (
                ("kernel_families", "fabricated-kernel-family"),
                ("required_cpu_bits", "fabricated-cpu-bit"),
                ("required_os_state", "fabricated:os-probe"),
            ):
                with self.subTest(tier=tier["name"], extra=field):
                    mutant = copy.deepcopy(manifest)
                    mutant["cpu"]["compiled_tiers"][index][field].append(extra)
                    self.assertTrue(
                        any(
                            field in error
                            for error in self.tool._cpu_policy(mutant)
                        ),
                        self.tool._cpu_policy(mutant),
                    )
            for field in (
                "kernel_families",
                "required_cpu_bits",
                "required_os_state",
            ):
                if not tier[field]:
                    continue
                with self.subTest(tier=tier["name"], missing=field):
                    mutant = copy.deepcopy(manifest)
                    del mutant["cpu"]["compiled_tiers"][index][field][0]
                    self.assertTrue(
                        any(
                            field in error
                            for error in self.tool._cpu_policy(mutant)
                        ),
                        self.tool._cpu_policy(mutant),
                    )

        arm_kernel_families = {
            "portable-neon": ["matmul-elem-f32-bf16-f16"],
            "dotprod": ["quant-dot-q8_0-q8_0"],
            "i8mm": ["quant-dot-q4_0-q8_0-q4_K-q6_K", "quant-repack-q8_0"],
        }
        for os_name, abi, artifact_id, dotprod_probe, i8mm_probe in (
            (
                "linux",
                "glibc",
                "linux-aarch64-glibc-cpu",
                "getauxval:AT_HWCAP:HWCAP_ASIMDDP",
                "getauxval:AT_HWCAP2:HWCAP2_I8MM",
            ),
            (
                "macos",
                "macos",
                "macos-arm64-cpu-policy-probe",
                "sysctl:hw.optional.arm.FEAT_DotProd",
                "sysctl:hw.optional.arm.FEAT_I8MM",
            ),
        ):
            arm = {
                "artifact": {"id": artifact_id},
                "host": {"os": os_name, "arch": "aarch64", "abi": abi},
                "backend": {"name": "cpu"},
                "cpu": {
                    "baseline": "portable-neon",
                    "selected_tier": "portable-neon",
                    "compiled_tiers": [
                        {
                            "name": "portable-neon",
                            "kernel_families": arm_kernel_families["portable-neon"],
                            "required_cpu_bits": ["neon"],
                            "required_os_state": [],
                        },
                        {
                            "name": "dotprod",
                            "kernel_families": arm_kernel_families["dotprod"],
                            "required_cpu_bits": ["dotprod", "neon"],
                            "required_os_state": [dotprod_probe],
                        },
                        {
                            "name": "i8mm",
                            "kernel_families": arm_kernel_families["i8mm"],
                            "required_cpu_bits": ["dotprod", "i8mm", "neon"],
                            "required_os_state": [dotprod_probe, i8mm_probe],
                        },
                    ],
                },
            }
            with self.subTest(aarch64_os=os_name):
                self.assertEqual(self.tool._cpu_policy(arm), [])
                wrong_probe = copy.deepcopy(arm)
                wrong_probe["cpu"]["compiled_tiers"][2]["required_os_state"] = [
                    "fabricated:os-probe"
                ]
                self.assertTrue(
                    any(
                        "required_os_state" in error
                        for error in self.tool._cpu_policy(wrong_probe)
                    ),
                    self.tool._cpu_policy(wrong_probe),
                )

    def test_primary_cuda_requires_every_sm_and_independent_aot_runtime_evidence(self) -> None:
        manifest = self.generated(cuda_facts())
        self.assertEqual(manifest["cuda"]["compiled_sms"], PRIMARY_SMS)
        for index, sm in enumerate(PRIMARY_SMS):
            with self.subTest(remove_sm=sm):
                mutant = copy.deepcopy(manifest)
                del mutant["cuda"]["compiled_sms"][index]
                del mutant["cuda"]["sm_evidence"][index]
                mutant["backend"]["flags"]["VLLM_CPP_CUDA_ARCHITECTURES"] = mutant["cuda"]["compiled_sms"]
                self.assert_invalid(mutant, "primary CUDA")
            for field in ("sm", "aot_available", "portable_fallback", "aot_evidence", "runtime_evidence"):
                with self.subTest(sm=sm, missing=field):
                    mutant = copy.deepcopy(manifest)
                    del mutant["cuda"]["sm_evidence"][index][field]
                    self.assert_invalid(mutant)
            mutant = copy.deepcopy(manifest)
            mutant["cuda"]["sm_evidence"][index]["aot_available"] = sm not in AOT_SMS
            self.assert_invalid(mutant, "AOT")
        independent = copy.deepcopy(manifest)
        independent["cuda"]["sm_evidence"][0]["aot_evidence"] = evidence("passed")
        independent["cuda"]["sm_evidence"][0]["runtime_evidence"] = evidence("absent")
        self.assertEqual(self.tool.validate_manifest(independent, self.schema, ROOT), [])
        mutant = copy.deepcopy(manifest)
        mutant["cuda"]["compiled_sms"][0] = "75"
        mutant["backend"]["flags"]["VLLM_CPP_CUDA_ARCHITECTURES"][0] = "75"
        mutant["cuda"]["sm_evidence"][0]["sm"] = "75"
        self.assert_invalid(mutant, "unsupported CUDA SM")

    def test_cuda_fallback_and_aot_state_rules_are_independently_live(self) -> None:
        manifest = self.generated(cuda_facts())
        unavailable = PRIMARY_SMS.index("87")
        fallback = copy.deepcopy(manifest)
        fallback["cuda"]["sm_evidence"][unavailable]["portable_fallback"] = False
        self.assertTrue(
            any("portable fallback" in error for error in self.tool._cuda_policy(fallback)),
            self.tool._cuda_policy(fallback),
        )
        aot_state = copy.deepcopy(manifest)
        aot_state["cuda"]["sm_evidence"][unavailable]["aot_evidence"] = evidence(
            "absent"
        )
        self.assertTrue(
            any("unavailable AOT" in error for error in self.tool._cuda_policy(aot_state)),
            self.tool._cuda_policy(aot_state),
        )
        available = PRIMARY_SMS.index("80")
        available_aot = copy.deepcopy(manifest)
        available_aot["cuda"]["sm_evidence"][available]["aot_evidence"] = evidence(
            "not-applicable"
        )
        self.assertTrue(
            any(
                "available AOT" in error
                for error in self.tool._cuda_policy(available_aot)
            ),
            self.tool._cuda_policy(available_aot),
        )

    def test_backend_channel_static_and_dependency_policies_fail_closed(self) -> None:
        cpu = self.generated(cpu_facts())
        cuda = self.generated(cuda_facts())
        stable = copy.deepcopy(cpu)
        stable["artifact"]["channel"] = "stable"
        self.assert_invalid(stable, "stable")
        for key in ("build", "archive_smoke", "dependency_audit", "runtime", "correctness"):
            stable["evidence"][key] = evidence("passed")
        self.assert_invalid(stable, "supply-chain")
        for key in ("archive_checksum", "sbom", "provenance", "licenses"):
            stable["supply_chain"][key] = evidence("passed")
        stable["build"]["source_clean"] = False
        self.assert_invalid(stable, "clean source")
        stable["build"]["source_clean"] = True
        self.assertEqual(self.tool.validate_manifest(stable, self.schema, ROOT), [])
        for key in ("archive_checksum", "sbom", "provenance", "licenses"):
            with self.subTest(stable_supply_chain=key):
                mutant = copy.deepcopy(stable)
                mutant["supply_chain"][key] = evidence("absent")
                self.assert_invalid(mutant, key)
        wrong_preview = copy.deepcopy(cuda)
        wrong_preview["artifact"]["channel"] = "experimental-preview"
        self.assert_invalid(wrong_preview, "channel")
        musl_gpu = copy.deepcopy(cuda)
        musl_gpu["artifact"]["id"] = "linux-x86_64-musl-cpu-static"
        musl_gpu["artifact"]["static_boundary"] = "literal-static"
        musl_gpu["artifact"]["channel"] = "experimental-preview"
        musl_gpu["host"]["abi"] = "musl"
        self.assert_invalid(musl_gpu, "musl")
        bundled_driver = copy.deepcopy(cuda)
        bundled_driver["dependencies"][0]["bundled"] = True
        bundled_driver["dependencies"][0]["linkage"] = "static"
        self.assert_invalid(bundled_driver, "driver")
        bad_boundary = copy.deepcopy(cuda)
        bad_boundary["backend"]["gpu_driver_boundary"] = "bundled"
        self.assert_invalid(bad_boundary, "driver boundary")
        bad_flags = copy.deepcopy(cpu)
        bad_flags["backend"]["flags"]["VLLM_CPP_VULKAN"] = True
        self.assert_invalid(bad_flags, "backend flags")

    def test_stable_runtime_and_correctness_rules_are_independently_live(self) -> None:
        stable = self.generated(cpu_facts())
        stable["artifact"]["channel"] = "stable"
        for key in ("build", "archive_smoke", "dependency_audit", "correctness"):
            stable["evidence"][key] = evidence("passed")
        for key in ("archive_checksum", "sbom", "provenance", "licenses"):
            stable["supply_chain"][key] = evidence("passed")
        runtime_errors = self.tool._publication_policy(stable)
        self.assertTrue(any("runtime" in error for error in runtime_errors), runtime_errors)
        stable["evidence"]["runtime"] = evidence("passed")
        stable["evidence"]["correctness"] = evidence("absent")
        correctness_errors = self.tool._publication_policy(stable)
        self.assertTrue(
            any("correctness" in error for error in correctness_errors),
            correctness_errors,
        )

    def test_preview_publication_requires_clean_staged_archive_evidence(self) -> None:
        preview = self.generated(cpu_facts())
        self.assertEqual(self.tool.validate_manifest(preview, self.schema, ROOT), [])
        for key in ("build", "archive_smoke", "dependency_audit"):
            with self.subTest(preview_evidence=key):
                mutant = copy.deepcopy(preview)
                mutant["evidence"][key] = evidence("absent")
                self.assert_invalid(mutant, key)
        dirty = copy.deepcopy(preview)
        dirty["build"]["source_clean"] = False
        self.assert_invalid(dirty, "clean source")
        for key in ("runtime", "correctness", "performance"):
            expected = "not-applicable" if key == "runtime" else "absent"
            self.assertEqual(preview["evidence"][key]["state"], expected)
        for key in ("archive_checksum", "sbom", "provenance", "licenses"):
            self.assertEqual(preview["supply_chain"][key]["state"], "absent")

    def test_cuda_manifest_requires_the_nvidia_driver_dependency(self) -> None:
        manifest = self.generated(cuda_facts())
        manifest["dependencies"] = [
            dependency
            for dependency in manifest["dependencies"]
            if dependency["name"] != "nvidia-driver"
        ]
        errors = self.tool.validate_manifest(manifest, self.schema, ROOT)
        self.assertIn(
            "$.dependencies: cuda requires external nvidia-driver declaration",
            errors,
        )

    def test_backend_dependency_and_static_boundaries_are_fail_closed(self) -> None:
        cpu = self.generated(cpu_facts())
        musl = copy.deepcopy(cpu)
        musl["artifact"].update({
            "id": "linux-x86_64-musl-cpu-static",
            "channel": "experimental-preview",
            "static_boundary": "literal-static",
        })
        musl["host"].update({"abi": "musl", "abi_version": "1.2.5"})
        musl["backend"]["flags"]["VLLM_CPP_LITERAL_STATIC"] = True
        musl["build"]["resolved_cmake_options"]["VLLM_CPP_LITERAL_STATIC"] = True
        musl["dependencies"] = [{
            "name": "musl", "version": "1.2.5", "kind": "library",
            "linkage": "static", "bundled": True, "role": "runtime",
        }]
        self.assertEqual(self.tool.validate_manifest(musl, self.schema, ROOT), [])
        dynamic_musl = copy.deepcopy(musl)
        dynamic_musl["dependencies"][0].update({
            "linkage": "dynamic", "bundled": False,
        })
        self.assert_invalid(dynamic_musl, "literal-static")

        cpu_driver = copy.deepcopy(cpu)
        cpu_driver["dependencies"].append({
            "name": "synthetic-gpu-driver", "version": "1", "kind": "driver",
            "linkage": "external", "bundled": False, "role": "external-runtime",
        })
        self.assert_invalid(cpu_driver, "CPU")

        cuda = self.generated(cuda_facts())
        wrong_cuda_driver = copy.deepcopy(cuda)
        wrong_cuda_driver["dependencies"][0]["name"] = "arbitrary-driver"
        self.assert_invalid(wrong_cuda_driver, "NVIDIA")

        def accelerator(
            artifact_id: str,
            backend_name: str,
            host: dict[str, str],
            dependencies: list[dict[str, object]],
        ) -> dict[str, object]:
            manifest = copy.deepcopy(cpu)
            manifest["artifact"]["id"] = artifact_id
            manifest["host"].update(host)
            manifest["backend"].update({
                "name": backend_name,
                "flags": flags(backend_name),
                "gpu_driver_boundary": "external-host-never-bundled",
            })
            if backend_name == "mlx":
                manifest["backend"]["flags"]["MLX_ROOT"] = "/opt/mlx"
            manifest["build"]["resolved_cmake_options"] = copy.deepcopy(
                manifest["backend"]["flags"]
            )
            manifest["dependencies"] = dependencies
            del manifest["cpu"]
            return manifest

        vulkan = accelerator(
            "linux-x86_64-glibc-vulkan", "vulkan", {}, [
                {"name": "vulkan-loader", "version": "1.3", "kind": "library", "linkage": "external", "bundled": False, "role": "external-runtime"},
                {"name": "vulkan-icd", "version": "1.3", "kind": "library", "linkage": "external", "bundled": False, "role": "external-runtime"},
                {"name": "vulkan-driver", "version": "tested-host", "kind": "driver", "linkage": "external", "bundled": False, "role": "external-runtime"},
            ],
        )
        self.assertEqual(self.tool.validate_manifest(vulkan, self.schema, ROOT), [])
        for name in ("vulkan-loader", "vulkan-icd", "vulkan-driver"):
            mutant = copy.deepcopy(vulkan)
            mutant["dependencies"] = [
                dep for dep in mutant["dependencies"] if dep["name"] != name
            ]
            self.assert_invalid(mutant, name)
            mutant = copy.deepcopy(vulkan)
            dependency = next(
                dep for dep in mutant["dependencies"] if dep["name"] == name
            )
            dependency.update({
                "linkage": "dynamic", "bundled": True, "role": "runtime",
            })
            self.assert_invalid(mutant, name)

        # Issue #1547. The container `vulkan` lane publishes a multi-arch
        # manifest, so its arm64 leg produces this tuple, and before the fix no
        # policy row named it. The tuple is PREVIEW ONLY: no arm64 Vulkan leg
        # has been built or gated here, so `stable` must stay refused.
        vulkan_arm64 = accelerator(
            "linux-aarch64-glibc-vulkan",
            "vulkan",
            {"arch": "aarch64"},
            copy.deepcopy(vulkan["dependencies"]),
        )
        self.assertEqual(
            self.tool.validate_manifest(vulkan_arm64, self.schema, ROOT), []
        )
        claimed_stable = copy.deepcopy(vulkan_arm64)
        claimed_stable["artifact"]["channel"] = "stable"
        self.assert_invalid(
            claimed_stable, "wrong channel for linux-aarch64-glibc-vulkan"
        )
        wrong_arch = copy.deepcopy(vulkan_arm64)
        wrong_arch["host"]["arch"] = "x86_64"
        self.assert_invalid(wrong_arch, "tuple policy mismatch")

        frameworks = [
            {"name": name, "version": "macOS-system", "kind": "framework", "linkage": "external", "bundled": False, "role": "external-runtime"}
            for name in ("Metal.framework", "Foundation.framework")
        ]
        metal = accelerator(
            "macos-arm64-metal", "metal",
            {"os": "macos", "arch": "aarch64", "abi": "macos", "abi_version": "14"},
            frameworks,
        )
        self.assertEqual(self.tool.validate_manifest(metal, self.schema, ROOT), [])
        for name in ("Metal.framework", "Foundation.framework"):
            mutant = copy.deepcopy(metal)
            mutant["dependencies"] = [
                dep for dep in mutant["dependencies"] if dep["name"] != name
            ]
            self.assert_invalid(mutant, name)
            mutant = copy.deepcopy(metal)
            dependency = next(
                dep for dep in mutant["dependencies"] if dep["name"] == name
            )
            dependency.update({
                "linkage": "dynamic", "bundled": True, "role": "runtime",
            })
            self.assert_invalid(mutant, name)

        mlx_dependencies = frameworks + [
            {"name": "libmlx.dylib", "version": "0.29.1", "kind": "library", "linkage": "dynamic", "bundled": True, "role": "runtime"},
            {"name": "mlx.metallib", "version": "0.29.1", "kind": "library", "linkage": "dynamic", "bundled": True, "role": "runtime"},
        ]
        mlx = accelerator(
            "macos-arm64-metal-mlx", "mlx",
            {"os": "macos", "arch": "aarch64", "abi": "macos", "abi_version": "14"},
            mlx_dependencies,
        )
        self.assert_invalid(mlx, "license")
        mlx["supply_chain"]["licenses"] = evidence("passed")
        self.assertEqual(self.tool.validate_manifest(mlx, self.schema, ROOT), [])
        unversioned_mlx = copy.deepcopy(mlx)
        unversioned_mlx["dependencies"][2]["version"] = "unknown"
        self.assert_invalid(unversioned_mlx, "versioned")
        arbitrary_bundle = copy.deepcopy(mlx)
        arbitrary_bundle["dependencies"].append({
            "name": "arbitrary.dylib", "version": "1", "kind": "library",
            "linkage": "dynamic", "bundled": True, "role": "runtime",
        })
        self.assert_invalid(arbitrary_bundle, "bundled")

    def test_windows_cpu_and_vulkan_require_pinned_msvc_ucrt_contract(self) -> None:
        cpu = self.generated(cpu_facts())
        windows = copy.deepcopy(cpu)
        windows["artifact"].update({
            "id": "windows-x86_64-msvc-cpu", "channel": "preview"
        })
        windows["host"].update({
            "os": "windows", "abi": "msvc", "abi_version": "14.38",
            "toolset_version": "14.38.33130", "ucrt_version": "10.0.20348.0",
        })
        windows["build"].update({
            "compiler": "MSVC 19.38.33135", "toolchain": "Visual Studio 2022 v143 /MT"
        })
        windows["dependencies"] = [{
            "name": "KERNEL32.dll", "version": "windows-2022", "kind": "library",
            "linkage": "dynamic", "bundled": False, "role": "runtime",
        }]
        self.assertEqual(self.tool.validate_manifest(windows, self.schema, ROOT), [])
        for field in ("toolset_version", "ucrt_version"):
            mutant = copy.deepcopy(windows)
            del mutant["host"][field]
            self.assert_invalid(mutant, field)
        dynamic_crt = copy.deepcopy(windows)
        dynamic_crt["dependencies"].append({
            "name": "VCRUNTIME140.dll", "version": "system", "kind": "library",
            "linkage": "dynamic", "bundled": False, "role": "runtime",
        })
        self.assert_invalid(dynamic_crt, "static CRT")

        vulkan = copy.deepcopy(windows)
        vulkan["artifact"]["id"] = "windows-x86_64-msvc-vulkan"
        vulkan["backend"].update({
            "name": "vulkan",
            "flags": flags("vulkan"),
            "gpu_driver_boundary": "external-host-never-bundled",
        })
        vulkan["build"]["resolved_cmake_options"] = copy.deepcopy(vulkan["backend"]["flags"])
        del vulkan["cpu"]
        vulkan["dependencies"].extend([
            {"name": name, "version": "host", "kind": kind, "linkage": "external", "bundled": False, "role": "external-runtime"}
            for name, kind in (("vulkan-loader", "library"), ("vulkan-icd", "library"), ("vulkan-driver", "driver"))
        ])
        vulkan["evidence"]["runtime"] = evidence("absent")
        self.assertEqual(self.tool.validate_manifest(vulkan, self.schema, ROOT), [])

    def test_cuda_feature_resolution_is_derived_from_current_cmake_table(self) -> None:
        manifest = self.generated(cuda_facts())
        names = [feature["name"] for feature in manifest["backend"]["resolved_features"]]
        self.assertEqual(names, [
            "fp4-mma", "cutlass-nvfp4", "cutlass-nvfp4-sm100", "cutlass-fp8",
            "scaledmm-c3x-sm90", "scaledmm-c3x-sm100", "marlin-nvfp4", "fa2",
        ])
        mutant = copy.deepcopy(manifest)
        mutant["backend"]["resolved_features"][0]["compiled_sms"] = ["121a"]
        self.assert_invalid(mutant, "resolved CUDA feature table")


if __name__ == "__main__":
    unittest.main()
