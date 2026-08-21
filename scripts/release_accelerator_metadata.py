#!/usr/bin/env python3
"""Generate Linux CUDA/Vulkan release metadata from staged bytes and build facts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import release_manifest  # noqa: E402
import release_metadata  # noqa: E402


# The (arch, backend) tuple each Linux accelerator artifact id names.
#
# `linux-aarch64-glibc-vulkan` is issue #1547. `docker/Dockerfile:145-146`
# selected it for every non-amd64 container build while nothing registered it,
# so the `vulkan` lane's arm64 leg failed with `unsupported Linux accelerator
# artifact`, and the failed leg skipped `manifest` and left the pushed `cpu`
# digests with no tag pointing at them.
#
# This registry names a BUILD tuple, not a downloadable release archive. The
# download matrix is `release/release-matrix.json`, which
# `scripts/release_pipeline.py:137` requires to equal `PRIMARY_ARTIFACT_FORMATS`
# exactly, and `.github/workflows/release.yml` builds no aarch64 Vulkan tarball.
# The id is deliberately absent there.
ARTIFACTS = {
    "linux-x86_64-glibc-cuda": ("x86_64", "cuda"),
    "linux-aarch64-glibc-cuda": ("aarch64", "cuda"),
    "linux-x86_64-glibc-vulkan": ("x86_64", "vulkan"),
    "linux-aarch64-glibc-vulkan": ("aarch64", "vulkan"),
}


def external_dependency(name: str, kind: str, version: str) -> dict[str, Any]:
    return {
        "bundled": False,
        "kind": kind,
        "linkage": "external",
        "name": name,
        "role": "external-runtime",
        "version": version,
    }


def dependency_rows(server: Path, backend: str, abi_version: str) -> list[dict[str, Any]]:
    rows = release_metadata.dependencies(server, "glibc", abi_version)
    if backend == "cuda":
        rows.append(external_dependency("nvidia-driver", "driver", ">=580"))
    else:
        rows.extend(
            (
                external_dependency("vulkan-loader", "library", ">=1.3"),
                external_dependency("vulkan-icd", "library", "vendor-matched"),
                external_dependency("vulkan-driver", "driver", "vendor-matched"),
            )
        )
    return rows


def cuda_evidence(url: str) -> dict[str, Any]:
    rows = []
    for sm in release_manifest.PRIMARY_CUDA_SMS:
        available = release_manifest.AOT_AVAILABILITY[sm]
        rows.append(
            {
                "aot_available": available,
                "aot_evidence": (
                    release_metadata.passed(
                        f"check exact embedded vt_aot_sm_{sm}_ namespace", url
                    )
                    if available
                    else release_metadata.evidence(
                        "not-applicable",
                        "",
                        "",
                        "",
                        f"no complete Triton AOT tree exists for sm_{sm}",
                    )
                ),
                "portable_fallback": not available,
                "runtime_evidence": release_metadata.absent(
                    f"preview archive has no matching sm_{sm} runtime gate"
                ),
                "sm": sm,
            }
        )
    return {
        "compiled_sms": list(release_manifest.PRIMARY_CUDA_SMS),
        "sm_evidence": rows,
    }


def prepare_accelerator_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_id not in ARTIFACTS:
        raise ValueError(f"unsupported Linux accelerator artifact {args.artifact_id!r}")
    arch, backend = ARTIFACTS[args.artifact_id]
    if args.channel != "preview":
        raise ValueError("Linux accelerator bundles remain preview until matching runtime evidence exists")
    server = args.stage_dir / "bin/vllm-server"
    if not server.is_file():
        raise ValueError("staged bin/vllm-server is missing")
    flags = release_metadata.backend_flags(
        release_metadata.parse_cache(args.build_dir / "CMakeCache.txt")
    )
    expected_sms = list(release_manifest.PRIMARY_CUDA_SMS) if backend == "cuda" else []
    if flags["VLLM_CPP_CUDA"] is not (backend == "cuda"):
        raise ValueError("configured CUDA flag does not match artifact backend")
    if flags["VLLM_CPP_VULKAN"] is not (backend == "vulkan"):
        raise ValueError("configured Vulkan flag does not match artifact backend")
    if flags["VLLM_CPP_CUDA_ARCHITECTURES"] != expected_sms:
        raise ValueError("CUDA release architecture cache must equal the exact ten-SM policy")
    if flags["VLLM_CPP_TRITON"] is not (backend == "cuda"):
        raise ValueError("CUDA release requires Triton and non-CUDA release forbids it")
    dependencies = dependency_rows(server, backend, args.abi_version)
    build_audit = (
        "check-cuda-fat-gencode.py && check-triton-aot-multiarch.py"
        if backend == "cuda"
        else "test_vulkan_backend && test_backend_cross_device"
    )
    facts: dict[str, Any] = {
        "artifact": {
            "c_abi_version": args.c_abi_version,
            "channel": args.channel,
            "id": args.artifact_id,
            "kind": "primary",
            "static_boundary": "static-core",
            "version": args.version,
        },
        "backend": {
            "flags": flags,
            "gpu_driver_boundary": "external-host-never-bundled",
            "name": backend,
        },
        "build": {
            "compiler": args.compiler,
            "resolved_cmake_options": flags,
            "source_clean": args.source_clean,
            "source_commit": args.source_commit,
            "test_commands": [build_audit, "python3 scripts/validate-release-archive.py"],
            "toolchain": args.toolchain,
        },
        "dependencies": dependencies,
        "evidence": {
            "archive_smoke": release_metadata.passed(
                "extracted vllm-server --help && --version", args.evidence_url
            ),
            "build": release_metadata.passed("cmake --build <build> --target server", args.evidence_url),
            "correctness": release_metadata.absent(
                "preview accelerator tuple has no matching-hardware model correctness gate"
            ),
            "dependency_audit": release_metadata.passed(
                "readelf -dW && ldd/lddtree", args.evidence_url
            ),
            "performance": release_metadata.absent("preview artifact makes no performance claim"),
            "runtime": release_metadata.absent(
                "preview accelerator tuple has no matching-hardware runtime gate"
            ),
        },
        "host": {
            "abi": "glibc",
            "abi_version": args.abi_version,
            "arch": arch,
            "os": "linux",
        },
        "supply_chain": {
            "archive_checksum": release_metadata.passed("sha256sum <final-archive>", args.evidence_url),
            "licenses": release_metadata.passed("validate notices and licenses", args.evidence_url),
            "provenance": release_metadata.passed("validate detached SLSA subject", args.evidence_url),
            "sbom": release_metadata.passed("validate SPDX-2.3 inventory", args.evidence_url),
        },
    }
    if backend == "cuda":
        facts["cuda"] = cuda_evidence(args.evidence_url)
    schema = release_manifest.load_schema(args.repo_root / "release/manifest-v1.schema.json")
    manifest = release_manifest.generate_manifest(facts, args.repo_root, schema)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-manifest.json").write_text(
        release_metadata.canonical_json(manifest), encoding="utf-8"
    )
    version_values = {
        "version": args.version,
        "commit": args.source_commit,
        "artifact_id": args.artifact_id,
        "backend": backend,
        "host_os": "linux",
        "host_arch": arch,
        "host_abi": "glibc",
        "source_clean": "true" if args.source_clean else "false",
        "c_abi_version": str(args.c_abi_version),
    }
    (output / "VERSION").write_text(
        "".join(f"{key}={value}\n" for key, value in version_values.items()),
        encoding="utf-8",
    )
    (output / "sbom.spdx.json").write_text(
        release_metadata.canonical_json(
            release_metadata.spdx_document(
                args.artifact_id,
                args.version,
                args.source_commit,
                server,
                dependencies,
            )
        ),
        encoding="utf-8",
    )
    notices = ["vllm.cpp release dependency notices", ""]
    notices.extend(
        f"- {row['name']} {row['version']} ({row['linkage']})" for row in dependencies
    )
    (output / "THIRD_PARTY_NOTICES").write_text("\n".join(notices) + "\n", encoding="utf-8")
    license_dir = output / "share/licenses/vllm.cpp"
    license_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.repo_root / "LICENSE", license_dir / "LICENSE")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--channel", choices=("preview",), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--c-abi-version", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-clean", action="store_true")
    parser.add_argument("--abi-version", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--evidence-url", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_accelerator_metadata(args)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, release_manifest.ManifestError) as exc:
        print(f"accelerator release metadata error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
