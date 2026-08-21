#!/usr/bin/env python3
"""Generate and validate vllm.cpp release manifests using only the stdlib.

This is build-time tooling. Release archives contain the generated JSON, not
this Python program or a Python runtime. CUDA feature resolution is derived from
the same VT_CUDA_FEATURE_TABLE consumed by CMake, so the manifest cannot invent
a capability independently of the build definition.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_ID = "vllm.cpp.release-manifest.v1"
SCHEMA_VERSION = 1
DEFAULT_SCHEMA = Path("release/manifest-v1.schema.json")
FEATURE_TABLE = Path("cmake/CudaArchFeatures.cmake")

PRIMARY_CUDA_SMS = (
    "80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a"
)
AOT_AVAILABILITY = {
    "80": True, "86": True, "87": False, "89": True, "90a": True,
    "100a": True, "103a": False, "110": False, "120a": False, "121a": True,
}
PUBLISHED_EVIDENCE = ("build", "archive_smoke", "dependency_audit")
STABLE_EVIDENCE = ("runtime", "correctness")
STABLE_SUPPLY_CHAIN_EVIDENCE = (
    "archive_checksum", "sbom", "provenance", "licenses"
)

# Current compiled CPU inventory. This is intentionally narrower than the ISA
# names a compiler understands: a tier is listed only when a real kernel TU and
# runtime selector exist. x86 anchors: CMakeLists.txt:912-923 and
# src/vt/cpu/cpu_matmul_elem.cpp:533-612; Arm anchors: CMakeLists.txt:926-933,
# src/vt/cpu/cpu_quant_dot_arm.cpp:1-40,67-83 and
# src/vt/cpu/cpu_quant_repack_arm.cpp:1-22,52-68.
CPU_TIER_POLICY = {
    "x86_64": {
        "baseline": "portable-sse2",
        "tiers": (
            "portable-sse2", "sse2-f16c", "avx2-f16c", "avx512f"
        ),
        "kernel_families": {
            "portable-sse2": {"matmul-elem-f32-bf16"},
            "sse2-f16c": {"matmul-elem-f16"},
            "avx2-f16c": {"matmul-elem-f32-bf16-f16"},
            "avx512f": {"matmul-elem-f32-bf16"},
        },
        "bits": {
            "portable-sse2": {"sse2"},
            "sse2-f16c": {"sse2", "avx", "f16c", "osxsave"},
            "avx2-f16c": {"sse2", "avx", "avx2", "f16c", "osxsave"},
            "avx512f": {
                "sse2", "avx", "f16c", "avx2", "avx512f", "avx512bw",
                "avx512vl", "osxsave",
            },
        },
        "os_state": {
            "portable-sse2": set(),
            "sse2-f16c": {"xcr0:xmm", "xcr0:ymm"},
            "avx2-f16c": {"xcr0:xmm", "xcr0:ymm"},
            "avx512f": {
                "xcr0:xmm", "xcr0:ymm", "xcr0:opmask",
                "xcr0:zmm_hi256", "xcr0:hi16_zmm",
            },
        },
    },
    "aarch64": {
        "baseline": "portable-neon",
        "tiers": ("portable-neon", "dotprod", "i8mm"),
        "kernel_families": {
            "portable-neon": {"matmul-elem-f32-bf16-f16"},
            "dotprod": {"quant-dot-q8_0-q8_0"},
            "i8mm": {
                "quant-dot-q4_0-q8_0-q4_K-q6_K",
                "quant-repack-q8_0",
            },
        },
        "bits": {
            "portable-neon": {"neon"},
            "dotprod": {"neon", "dotprod"},
            "i8mm": {"neon", "dotprod", "i8mm"},
        },
        "os_state": {
            "portable-neon": {"linux": set(), "macos": set()},
            "dotprod": {
                "linux": {"getauxval:AT_HWCAP:HWCAP_ASIMDDP"},
                "macos": {"sysctl:hw.optional.arm.FEAT_DotProd"},
            },
            "i8mm": {
                "linux": {
                    "getauxval:AT_HWCAP:HWCAP_ASIMDDP",
                    "getauxval:AT_HWCAP2:HWCAP2_I8MM",
                },
                "macos": {
                    "sysctl:hw.optional.arm.FEAT_DotProd",
                    "sysctl:hw.optional.arm.FEAT_I8MM",
                },
            },
        },
    },
}


class ManifestError(ValueError):
    """Raised when generation input cannot produce a valid manifest."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def load_schema(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ManifestError(f"schema {path} must be a JSON object")
    return value


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ManifestError(f"unsupported non-local schema reference {ref!r}")
    value: Any = root
    for component in ref[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        value = value[component]
    if not isinstance(value, dict):
        raise ManifestError(f"schema reference {ref!r} is not an object")
    return value


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    raise ManifestError(f"unsupported schema type {expected!r}")


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality collapse."""
    if type(left) is not type(right):
        return False
    return left == right


def _schema_errors(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
) -> list[str]:
    if "$ref" in schema:
        return _schema_errors(value, _resolve_ref(schema["$ref"], root), root, path)
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(value, expected_type):
        return [f"{path}: expected {expected_type}, got {type(value).__name__}"]
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: string is shorter than minLength")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: required property {key!r} is missing")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unknown property {key!r}")
        for key, child in value.items():
            if key in properties:
                errors.extend(
                    _schema_errors(child, properties[key], root, f"{path}.{key}")
                )
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: array has fewer than minItems entries")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: array entries must be unique")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(
                    _schema_errors(item, item_schema, root, f"{path}[{index}]")
                )
    return errors


def _release_arch(value: str) -> str:
    value = value.replace("+PTX", "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)(a?)", value)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    if re.fullmatch(r"\d+(?:a)?", value):
        return value
    raise ManifestError(f"cannot normalize CUDA architecture {value!r}")


def parse_cuda_feature_table(repo_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    path = repo_root / FEATURE_TABLE
    text = path.read_text(encoding="utf-8")
    rows: list[tuple[str, tuple[str, ...]]] = []
    in_table = False
    for line in text.splitlines():
        if line.strip() == "set(VT_CUDA_FEATURE_TABLE":
            in_table = True
            continue
        if in_table and line.strip() == ")":
            break
        if not in_table:
            continue
        match = re.match(r'^\s*"([^|]+)\|([^|]*)\|', line)
        if not match:
            continue
        arches = tuple(_release_arch(item) for item in match.group(2).split(",") if item)
        rows.append((match.group(1), arches))
    if not rows:
        raise ManifestError(f"{path}: VT_CUDA_FEATURE_TABLE is empty or unreadable")
    names = [name for name, _ in rows]
    if len(names) != len(set(names)):
        raise ManifestError(f"{path}: duplicate CUDA feature name")
    return rows


def resolve_cuda_features(
    compiled_sms: list[str], repo_root: Path
) -> list[dict[str, Any]]:
    requested = set(compiled_sms)
    return [
        {"name": name, "compiled_sms": [sm for sm in compiled_sms if sm in arches]}
        for name, arches in parse_cuda_feature_table(repo_root)
    ]


def _evidence_objects(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    objects: list[tuple[str, dict[str, Any]]] = []
    evidence_set = manifest.get("evidence")
    if isinstance(evidence_set, dict):
        for key, value in evidence_set.items():
            if isinstance(value, dict):
                objects.append((f"$.evidence.{key}", value))
    supply_chain = manifest.get("supply_chain")
    if isinstance(supply_chain, dict):
        for key, value in supply_chain.items():
            if isinstance(value, dict):
                objects.append((f"$.supply_chain.{key}", value))
    cpu = manifest.get("cpu")
    if isinstance(cpu, dict):
        for index, tier in enumerate(cpu.get("compiled_tiers", [])):
            if isinstance(tier, dict) and isinstance(tier.get("execution_evidence"), dict):
                objects.append(
                    (f"$.cpu.compiled_tiers[{index}].execution_evidence", tier["execution_evidence"])
                )
    cuda = manifest.get("cuda")
    if isinstance(cuda, dict):
        for index, sm in enumerate(cuda.get("sm_evidence", [])):
            if not isinstance(sm, dict):
                continue
            for key in ("aot_evidence", "runtime_evidence"):
                if isinstance(sm.get(key), dict):
                    objects.append((f"$.cuda.sm_evidence[{index}].{key}", sm[key]))
    return objects


def _validate_evidence(path: str, value: dict[str, Any]) -> list[str]:
    state = value.get("state")
    reason = value.get("reason")
    command = value.get("command")
    result = value.get("result")
    url = value.get("url")
    if not all(isinstance(item, str) for item in (reason, command, result, url)):
        return []  # the schema reports the type error
    errors: list[str] = []
    if state == "passed":
        if reason:
            errors.append(f"{path}: passed evidence reason must be empty")
        for field, content in (("command", command), ("result", result), ("url", url)):
            if not content:
                errors.append(f"{path}: passed evidence requires non-empty {field}")
    elif state == "failed":
        for field, content in (("reason", reason), ("command", command), ("result", result)):
            if not content:
                errors.append(f"{path}: failed evidence requires non-empty {field}")
    elif state in {"absent", "not-applicable"}:
        if not reason:
            errors.append(f"{path}: {state} evidence requires a reason")
        for field, content in (("command", command), ("result", result), ("url", url)):
            if content:
                errors.append(f"{path}: {state} evidence must leave {field} empty")
    return errors


def _artifact_policy(manifest: dict[str, Any]) -> list[str]:
    artifact = manifest.get("artifact", {})
    host = manifest.get("host", {})
    backend = manifest.get("backend", {})
    if not all(isinstance(value, dict) for value in (artifact, host, backend)):
        return []
    artifact_id = artifact.get("id")
    name = backend.get("name")
    if not isinstance(artifact.get("c_abi_version"), int) or isinstance(
        artifact.get("c_abi_version"), bool
    ) or artifact.get("c_abi_version", 0) <= 0:
        return ["$.artifact.c_abi_version: must be a positive integer"]
    policies = {
        "linux-x86_64-glibc-cpu": ("linux", "x86_64", "glibc", "cpu", "static-core", {"preview", "stable"}),
        "linux-aarch64-glibc-cpu": ("linux", "aarch64", "glibc", "cpu", "static-core", {"preview", "stable"}),
        "linux-x86_64-glibc-cuda": ("linux", "x86_64", "glibc", "cuda", "static-core", {"preview", "stable"}),
        "linux-aarch64-glibc-cuda": ("linux", "aarch64", "glibc", "cuda", "static-core", {"preview", "stable"}),
        "macos-arm64-metal": ("macos", "aarch64", "macos", "metal", "static-core", {"preview", "stable"}),
        "macos-arm64-metal-mlx": ("macos", "aarch64", "macos", "mlx", "static-core", {"preview"}),
        "linux-x86_64-glibc-vulkan": ("linux", "x86_64", "glibc", "vulkan", "static-core", {"preview"}),
        # Issue #1547. The container `vulkan` lane publishes a multi-arch
        # manifest, so its arm64 leg produces this tuple. PREVIEW ONLY, and the
        # narrower set is the point: no arm64 Vulkan leg has ever been built or
        # gated here, `.agents/roadmap_v1.md` records both arm64 container legs
        # as unbuilt, and `scripts/build-linux-accelerator-release.sh` passes
        # `--channel preview` for every accelerator artifact. A `stable` entry
        # would claim evidence that does not exist.
        "linux-aarch64-glibc-vulkan": ("linux", "aarch64", "glibc", "vulkan", "static-core", {"preview"}),
        "linux-x86_64-musl-cpu-static": ("linux", "x86_64", "musl", "cpu", "literal-static", {"experimental-preview"}),
        "windows-x86_64-msvc-cpu": ("windows", "x86_64", "msvc", "cpu", "static-core", {"preview"}),
        "windows-x86_64-msvc-vulkan": ("windows", "x86_64", "msvc", "vulkan", "static-core", {"preview"}),
    }
    policy = policies.get(artifact_id)
    if policy is None and artifact.get("kind") == "diagnostic" and name == "cuda":
        if not isinstance(artifact_id, str) or not re.fullmatch(
            r"linux-(?:x86_64|aarch64)-glibc-cuda-sm(?:80|86|87|89|90a|100a|103a|110|120a|121a)",
            artifact_id,
        ):
            return ["$.artifact.id: unknown diagnostic CUDA artifact tuple"]
        return []
    if policy is None:
        return ["$.artifact.id: unknown release artifact tuple"]
    os_name, arch, abi, expected_backend, static_boundary, channels = policy
    errors: list[str] = []
    actual = (host.get("os"), host.get("arch"), host.get("abi"), name, artifact.get("static_boundary"))
    expected = (os_name, arch, abi, expected_backend, static_boundary)
    if actual != expected:
        errors.append(f"$.artifact: tuple policy mismatch; expected {expected!r}, got {actual!r}")
    if artifact.get("channel") not in channels:
        errors.append(f"$.artifact.channel: wrong channel for {artifact_id}")
    if artifact.get("kind") != "primary":
        errors.append(f"$.artifact.kind: matrix artifact {artifact_id} must be primary")
    if os_name == "windows":
        for field in ("toolset_version", "ucrt_version"):
            if not isinstance(host.get(field), str) or not host[field]:
                errors.append(f"$.host.{field}: Windows artifacts require a pinned value")
        build = manifest.get("build", {})
        if not isinstance(build, dict) or "/MT" not in str(build.get("toolchain", "")):
            errors.append("$.build.toolchain: Windows artifacts require the /MT static CRT")
    return errors


def _backend_policy(manifest: dict[str, Any], repo_root: Path) -> list[str]:
    backend = manifest.get("backend", {})
    artifact = manifest.get("artifact", {})
    if not isinstance(backend, dict):
        return []
    if not isinstance(artifact, dict):
        artifact = {}
    name = backend.get("name")
    flags = backend.get("flags", {})
    if not isinstance(flags, dict):
        return []
    expected_switches = {
        "VLLM_CPP_CUDA": name == "cuda",
        "VLLM_CPP_HIP": False,
        "VLLM_CPP_METAL": name in {"metal", "mlx"},
        "VLLM_CPP_MLX": name == "mlx",
        "VLLM_CPP_VULKAN": name == "vulkan",
    }
    errors: list[str] = []
    for flag, expected in expected_switches.items():
        if flags.get(flag) is not expected:
            errors.append(f"$.backend.flags.{flag}: inconsistent backend flags for {name}")
    expects_literal_static = artifact.get("static_boundary") == "literal-static"
    if flags.get("VLLM_CPP_LITERAL_STATIC") is not expects_literal_static:
        errors.append(
            "$.backend.flags.VLLM_CPP_LITERAL_STATIC: must agree with the artifact static boundary"
        )
    if flags.get("VLLM_CPP_SERVER") is not True or flags.get("VLLM_CPP_BUILD_EXAMPLES") is not True:
        errors.append("$.backend.flags: release backend flags must build server and examples")
    if flags.get("VLLM_CPP_HIP_ARCHITECTURES") != []:
        errors.append("$.backend.flags.VLLM_CPP_HIP_ARCHITECTURES: HIP is blocked")
    if name == "cuda":
        if flags.get("VLLM_CPP_TRITON") is not True:
            errors.append("$.backend.flags.VLLM_CPP_TRITON: CUDA release input must resolve Triton explicitly")
        compiled = manifest.get("cuda", {}).get("compiled_sms", []) if isinstance(manifest.get("cuda"), dict) else []
        expected_features = resolve_cuda_features(compiled, repo_root)
        if backend.get("resolved_features") != expected_features:
            errors.append("$.backend.resolved_features: does not match resolved CUDA feature table")
        if flags.get("VLLM_CPP_CUDA_ARCHITECTURES") != compiled:
            errors.append("$.backend.flags.VLLM_CPP_CUDA_ARCHITECTURES: must equal compiled CUDA SMs")
        if backend.get("gpu_driver_boundary") != "external-host-never-bundled":
            errors.append("$.backend.gpu_driver_boundary: CUDA driver boundary must stay external")
    else:
        if flags.get("VLLM_CPP_CUDA_ARCHITECTURES") != []:
            errors.append("$.backend.flags.VLLM_CPP_CUDA_ARCHITECTURES: non-CUDA artifact cannot claim SMs")
        if backend.get("resolved_features") != []:
            errors.append("$.backend.resolved_features: non-CUDA artifact cannot claim CUDA features")
        if name == "cpu" and flags.get("VLLM_CPP_TRITON") is not False:
            errors.append("$.backend.flags.VLLM_CPP_TRITON: CPU artifact cannot enable Triton")
    if name == "mlx":
        if not flags.get("MLX_ROOT"):
            errors.append("$.backend.flags.MLX_ROOT: MLX preview requires an explicit root")
    elif flags.get("MLX_ROOT") != "":
        errors.append("$.backend.flags.MLX_ROOT: only MLX artifacts may set MLX_ROOT")
    if name in {"cuda", "metal", "mlx", "vulkan"}:
        if backend.get("gpu_driver_boundary") != "external-host-never-bundled":
            errors.append("$.backend.gpu_driver_boundary: accelerator driver boundary must stay external")
    elif backend.get("gpu_driver_boundary") != "not-applicable":
        errors.append("$.backend.gpu_driver_boundary: CPU driver boundary must be not-applicable")
    return errors


def _dependency_policy(manifest: dict[str, Any]) -> list[str]:
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    backend = manifest.get("backend", {})
    artifact = manifest.get("artifact", {})
    supply_chain = manifest.get("supply_chain", {})
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    literal_static = (
        isinstance(artifact, dict)
        and artifact.get("static_boundary") == "literal-static"
    )
    errors: list[str] = []
    names: list[Any] = []
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            continue
        dependency_name = dependency.get("name")
        names.append(dependency_name)
        if literal_static and dependency.get("role") != "build-time":
            if (
                dependency.get("kind") == "driver"
                or dependency.get("linkage") != "static"
                or dependency.get("bundled") is not True
                or dependency.get("role") == "external-runtime"
            ):
                errors.append(
                    f"$.dependencies[{index}]: literal-static runtime dependency "
                    "must be a bundled static non-driver"
                )
        if backend_name == "cpu" and dependency.get("kind") == "driver":
            errors.append(f"$.dependencies[{index}]: CPU artifact cannot declare a GPU driver")
        if dependency.get("kind") == "driver":
            if dependency.get("bundled") is not False:
                errors.append(f"$.dependencies[{index}]: GPU driver must never be bundled")
            if dependency.get("linkage") != "external" or dependency.get("role") != "external-runtime":
                errors.append(f"$.dependencies[{index}]: GPU driver must be an external runtime boundary")
        if dependency.get("bundled") and dependency.get("linkage") == "dynamic":
            allowed_mlx = backend_name == "mlx" and dependency_name in {
                "libmlx.dylib", "mlx.metallib"
            }
            if not allowed_mlx:
                errors.append(
                    f"$.dependencies[{index}]: bundled dynamic dependency is not permitted"
                )
        if (
            manifest.get("host", {}).get("os") == "windows"
            and re.match(r"(?i)^(?:vcruntime|msvcp|msvcr|ucrtbase|concrt).*\.dll$", str(dependency_name))
        ):
            errors.append(f"$.dependencies[{index}]: Windows release requires the static CRT")
    if len(names) != len(set(names)):
        errors.append("$.dependencies: dependency names must be unique")

    by_name = {
        dependency.get("name"): dependency
        for dependency in dependencies
        if isinstance(dependency, dict)
    }

    def require_external(name: str, kind: str) -> None:
        dependency = by_name.get(name)
        if not isinstance(dependency, dict) or (
            dependency.get("kind") != kind
            or dependency.get("linkage") != "external"
            or dependency.get("bundled") is not False
            or dependency.get("role") != "external-runtime"
        ):
            errors.append(
                f"$.dependencies: {backend_name} requires external {name} declaration"
            )

    if backend_name == "cuda":
        require_external("nvidia-driver", "driver")
        for index, dependency in enumerate(dependencies):
            if (
                isinstance(dependency, dict)
                and dependency.get("kind") == "driver"
                and dependency.get("name") != "nvidia-driver"
            ):
                errors.append(
                    f"$.dependencies[{index}]: CUDA accepts only the NVIDIA driver boundary"
                )
    elif backend_name == "vulkan":
        require_external("vulkan-loader", "library")
        require_external("vulkan-icd", "library")
        require_external("vulkan-driver", "driver")
    elif backend_name in {"metal", "mlx"}:
        require_external("Metal.framework", "framework")
        require_external("Foundation.framework", "framework")

    if backend_name == "mlx":
        for name in ("libmlx.dylib", "mlx.metallib"):
            dependency = by_name.get(name)
            if not isinstance(dependency, dict) or (
                dependency.get("kind") != "library"
                or dependency.get("linkage") != "dynamic"
                or dependency.get("bundled") is not True
                or dependency.get("role") != "runtime"
                or dependency.get("version") in {"", "unknown", "unversioned"}
            ):
                errors.append(f"$.dependencies: MLX requires bundled versioned {name}")
        licenses = supply_chain.get("licenses", {}) if isinstance(supply_chain, dict) else {}
        if not isinstance(licenses, dict) or licenses.get("state") != "passed":
            errors.append("$.dependencies: bundled MLX dylib/metallib require passed license evidence")
    return errors


def _build_policy(manifest: dict[str, Any]) -> list[str]:
    build = manifest.get("build")
    backend = manifest.get("backend")
    if not isinstance(build, dict) or not isinstance(backend, dict):
        return []
    errors: list[str] = []
    commit = build.get("source_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        errors.append("$.build.source_commit: must be a full lowercase 40-hex commit")
    if build.get("resolved_cmake_options") != backend.get("flags"):
        errors.append("$.build.resolved_cmake_options: must equal resolved CMake backend flags")
    commands = build.get("test_commands")
    if isinstance(commands, list) and len(commands) != len(set(commands)):
        errors.append("$.build.test_commands: commands must be unique")
    return errors


def _cpu_policy(manifest: dict[str, Any]) -> list[str]:
    backend = manifest.get("backend", {})
    host = manifest.get("host", {})
    cpu = manifest.get("cpu")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    if backend_name != "cpu":
        return ["$.cpu: only CPU artifacts may carry compiled CPU tiers"] if cpu is not None else []
    if not isinstance(cpu, dict) or not isinstance(host, dict):
        return ["$.cpu: CPU artifact requires compiled CPU tiers"]
    policy = CPU_TIER_POLICY.get(host.get("arch"))
    if policy is None:
        return ["$.host.arch: no compiled CPU tier inventory for host architecture"]
    errors: list[str] = []
    if cpu.get("baseline") != policy["baseline"]:
        errors.append(f"$.cpu.baseline: expected {policy['baseline']!r}")
    tiers = cpu.get("compiled_tiers", [])
    if not isinstance(tiers, list):
        return errors
    names = [tier.get("name") for tier in tiers if isinstance(tier, dict)]
    if names != list(policy["tiers"]):
        errors.append(f"$.cpu.compiled_tiers: compiled tiers must equal {list(policy['tiers'])!r}")
    if cpu.get("selected_tier") not in names:
        errors.append("$.cpu.selected_tier: selected tier must be one of the compiled tiers")
    for index, tier in enumerate(tiers):
        if not isinstance(tier, dict) or tier.get("name") not in policy["tiers"]:
            continue
        name = tier["name"]
        kernel_families = set(tier.get("kernel_families", []))
        expected_kernel_families = set(policy["kernel_families"][name])
        if kernel_families != expected_kernel_families:
            errors.append(
                f"$.cpu.compiled_tiers[{index}].kernel_families: {name} "
                f"must equal {sorted(expected_kernel_families)!r}"
            )
        bits = set(tier.get("required_cpu_bits", []))
        expected_bits = set(policy["bits"][name])
        if bits != expected_bits:
            errors.append(
                f"$.cpu.compiled_tiers[{index}].required_cpu_bits: {name} "
                f"must equal {sorted(expected_bits)!r}"
            )
        state_policy = policy["os_state"][name]
        if isinstance(state_policy, dict):
            required_state = set(state_policy.get(host.get("os"), set()))
        else:
            required_state = set(state_policy)
        actual_state = set(tier.get("required_os_state", []))
        if actual_state != required_state:
            errors.append(
                f"$.cpu.compiled_tiers[{index}].required_os_state: OS-state "
                f"probes for {name} "
                f"must equal {sorted(required_state)!r}"
            )
    return errors


def _cuda_policy(manifest: dict[str, Any]) -> list[str]:
    backend = manifest.get("backend", {})
    cuda = manifest.get("cuda")
    backend_name = backend.get("name") if isinstance(backend, dict) else None
    if backend_name != "cuda":
        return ["$.cuda: only CUDA artifacts may carry SM evidence"] if cuda is not None else []
    if not isinstance(cuda, dict):
        return ["$.cuda: CUDA artifact requires compiled SM and per-SM evidence"]
    compiled = cuda.get("compiled_sms", [])
    rows = cuda.get("sm_evidence", [])
    if not isinstance(compiled, list) or not isinstance(rows, list):
        return []
    errors: list[str] = []
    unsupported = [sm for sm in compiled if sm not in PRIMARY_CUDA_SMS]
    if unsupported:
        errors.append(f"$.cuda.compiled_sms: unsupported CUDA SM claim {unsupported!r}")
    artifact = manifest.get("artifact", {})
    kind = artifact.get("kind") if isinstance(artifact, dict) else None
    if kind == "primary" and compiled != list(PRIMARY_CUDA_SMS):
        errors.append("$.cuda.compiled_sms: primary CUDA artifact requires all ten supported SMs")
    if kind == "diagnostic" and len(compiled) != 1:
        errors.append("$.cuda.compiled_sms: diagnostic CUDA artifact requires exactly one SM")
    row_sms = [row.get("sm") for row in rows if isinstance(row, dict)]
    if row_sms != compiled:
        errors.append("$.cuda.sm_evidence: rows must match compiled SMs exactly and in order")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("sm") not in AOT_AVAILABILITY:
            continue
        sm = row["sm"]
        available = AOT_AVAILABILITY[sm]
        if row.get("aot_available") is not available:
            errors.append(f"$.cuda.sm_evidence[{index}]: AOT availability for sm_{sm} is incorrect")
        if row.get("portable_fallback") is not (not available):
            errors.append(f"$.cuda.sm_evidence[{index}]: portable fallback for sm_{sm} is incorrect")
        aot = row.get("aot_evidence", {})
        if isinstance(aot, dict):
            state = aot.get("state")
            if not available and state != "not-applicable":
                errors.append(f"$.cuda.sm_evidence[{index}]: unavailable AOT must be not-applicable")
            if available and state == "not-applicable":
                errors.append(f"$.cuda.sm_evidence[{index}]: available AOT cannot be not-applicable")
    return errors


def _publication_policy(manifest: dict[str, Any]) -> list[str]:
    artifact = manifest.get("artifact", {})
    build = manifest.get("build", {})
    evidence_set = manifest.get("evidence", {})
    supply_chain = manifest.get("supply_chain", {})
    if (
        not isinstance(artifact, dict)
        or not isinstance(build, dict)
        or not isinstance(evidence_set, dict)
        or not isinstance(supply_chain, dict)
    ):
        return []
    channel = artifact.get("channel")
    if channel not in {"preview", "experimental-preview", "stable"}:
        return []
    errors = []
    for key in PUBLISHED_EVIDENCE:
        evidence = evidence_set.get(key, {})
        if not isinstance(evidence, dict) or evidence.get("state") != "passed":
            errors.append(
                f"$.artifact.channel: {channel} publication requires passed {key} evidence"
            )
    if build.get("source_clean") is not True:
        errors.append(f"$.artifact.channel: {channel} publication requires a clean source tree")
    if channel != "stable":
        return errors
    for key in STABLE_EVIDENCE:
        evidence = evidence_set.get(key, {})
        if not isinstance(evidence, dict) or evidence.get("state") != "passed":
            errors.append(f"$.artifact.channel: stable requires passed {key} evidence")
    for key in STABLE_SUPPLY_CHAIN_EVIDENCE:
        evidence = supply_chain.get(key, {})
        if not isinstance(evidence, dict) or evidence.get("state") != "passed":
            errors.append(
                "$.artifact.channel: stable supply-chain requires passed "
                f"{key} evidence"
            )
    cuda = manifest.get("cuda")
    if isinstance(cuda, dict):
        for index, row in enumerate(cuda.get("sm_evidence", [])):
            if not isinstance(row, dict):
                continue
            if row.get("runtime_evidence", {}).get("state") != "passed":
                errors.append(f"$.artifact.channel: stable CUDA requires passed runtime evidence for row {index}")
            if row.get("aot_available") and row.get("aot_evidence", {}).get("state") != "passed":
                errors.append(f"$.artifact.channel: stable CUDA requires passed AOT evidence for row {index}")
    return errors


def validate_manifest(
    manifest: dict[str, Any], schema: dict[str, Any], repo_root: Path
) -> list[str]:
    errors = _schema_errors(manifest, schema, schema)
    for path, value in _evidence_objects(manifest):
        errors.extend(_validate_evidence(path, value))
    errors.extend(_artifact_policy(manifest))
    errors.extend(_backend_policy(manifest, repo_root))
    errors.extend(_dependency_policy(manifest))
    errors.extend(_build_policy(manifest))
    errors.extend(_cpu_policy(manifest))
    errors.extend(_cuda_policy(manifest))
    errors.extend(_publication_policy(manifest))
    return errors


def generate_manifest(
    facts: dict[str, Any], repo_root: Path, schema: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(facts, dict):
        raise ManifestError("generation input must be a JSON object")
    manifest = copy.deepcopy(facts)
    manifest["schema"] = SCHEMA_ID
    manifest["schema_version"] = SCHEMA_VERSION
    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        raise ManifestError("generation input requires backend object")
    if "resolved_features" in backend:
        raise ManifestError("resolved_features is generated, not accepted as input")
    if backend.get("name") == "cuda":
        cuda = manifest.get("cuda")
        if not isinstance(cuda, dict) or not isinstance(cuda.get("compiled_sms"), list):
            raise ManifestError("CUDA generation input requires cuda.compiled_sms")
        backend["resolved_features"] = resolve_cuda_features(cuda["compiled_sms"], repo_root)
    else:
        backend["resolved_features"] = []
    errors = validate_manifest(manifest, schema, repo_root)
    if errors:
        raise ManifestError("\n".join(errors))
    return manifest


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_reject_duplicate_keys)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="validate build facts and write canonical manifest JSON")
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate an existing release manifest")
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = args.schema if args.schema.is_absolute() else repo_root / args.schema
    try:
        schema = load_schema(schema_path)
        if args.command == "generate":
            manifest = generate_manifest(_read_json(args.input), repo_root, schema)
            _write_text(args.output, canonical_json(manifest))
            print(f"wrote {args.output}")
        else:
            manifest = _read_json(args.manifest)
            errors = validate_manifest(manifest, schema, repo_root)
            if errors:
                raise ManifestError("\n".join(errors))
            print(f"valid: {args.manifest}")
    except (OSError, json.JSONDecodeError, KeyError, ManifestError) as error:
        print(f"release manifest error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
