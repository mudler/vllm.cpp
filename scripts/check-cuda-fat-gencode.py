#!/usr/bin/env python3
"""Audit W1 per-source CUDA gencode and the linked ten-SM archive."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ALL_SMS = ("80", "86", "87", "89", "90a", "100a", "103a", "110", "120a", "121a")
SM12X = ("120a", "121a")


def _feature_table_archs(feature: str) -> tuple[str, ...]:
    """Read one feature's arch set out of cmake/CudaArchFeatures.cmake.

    #394: this checker used to restate the Marlin arch set as SM12X. When
    sm_110 was added to marlin-nvfp4 for Thor (#325) only cmake learned it, so
    every ten-SM fat build failed this audit on 14 correctly-compiled TUs.
    CMake owns the fact; deriving it here is what stops the two drifting again.

    The rows look like:
        "marlin-nvfp4|11.0,12.0a,12.1a|vendored Marlin NVFP4 ..."
    and the arch spellings differ between the two files (11.0 vs 110).
    """
    table = PROJECT_ROOT / "cmake/CudaArchFeatures.cmake"
    row = re.search(rf'"{re.escape(feature)}\|([^|]+)\|', table.read_text(encoding="utf-8"))
    if row is None:
        raise SystemExit(f"{table}: no feature row for {feature!r}")
    archs = tuple(arch.strip().replace(".", "", 1) for arch in row.group(1).split(","))
    if not archs:
        raise SystemExit(f"{table}: feature {feature!r} declares no architectures")
    return archs
FA2_SMS = ("80", "86", "87", "89", "120a", "121a")

REQUIRED_SOURCES = (
    "src/vt/cuda/cuda_matmul_nvfp4.cu",
    "src/vt/cuda/cuda_nvfp4_sm12x.cu",
    "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu",
    "src/vt/cuda/cuda_matmul_fp8_cutlass.cu",
    "src/vt/cuda/cuda_matmul_fp8_block_cutlass.cu",
    "src/vt/cuda/cuda_matmul_nvfp4_sm100.cu",
    "src/vt/cuda/cuda_scaled_mm_c3x_sm90.cu",
    "src/vt/cuda/cuda_scaled_mm_c3x_sm100.cu",
    "src/vt/cuda/cuda_moe_marlin.cu",
    "src/vt/cuda/cuda_flash_attn_fa2.cu",
)

ARCH_RE = re.compile(r"arch=compute_([0-9]+[af]?),code=sm_([0-9]+[af]?)")
ARCHIVE_RE = re.compile(r"\bsm_([0-9]+[af]?)\b")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def relative_source(value: str, directory: str = "") -> str | None:
    source = Path(value)
    if not source.is_absolute():
        source = Path(directory) / source
    try:
        relative = source.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    normalized = relative.as_posix()
    return normalized if normalized.startswith("src/") else None


def expected_sms(source: str) -> tuple[str, ...]:
    name = Path(source).name
    if name == "cuda_matmul_nvfp4_sm100.cu":
        return ("100a",)
    if name == "cuda_scaled_mm_c3x_sm90.cu":
        return ("90a",)
    if name == "cuda_scaled_mm_c3x_sm100.cu":
        return ("100a",)
    if "flash_attn/" in source or name == "cuda_flash_attn_fa2.cu":
        return FA2_SMS
    # Marlin is DERIVED from the feature table (#394); the nvfp4/fp8 CUTLASS
    # sources below are genuinely sm12x-only and stay literal. Collapsing the
    # two into one branch would silently widen those to Marlin's arch set.
    if (
        name in {
            "cuda_moe_marlin.cu",
            "cuda_marlin_repack.cu",
            "cuda_marlin_dense.cu",
        }
        or "/marlin/" in source
    ):
        return _feature_table_archs("marlin-nvfp4")
    if (
        name == "cuda_nvfp4_sm12x.cu"
        or name == "cuda_matmul_nvfp4_cutlass.cu"
        or name.startswith("cuda_nvfp4_tactics_")
        or name == "cuda_matmul_fp8_cutlass.cu"
        or name == "cuda_matmul_fp8_block_cutlass.cu"
    ):
        return SM12X
    return ALL_SMS


def command_sms(command: str) -> tuple[str, ...]:
    found: list[str] = []
    for compute, code in ARCH_RE.findall(command):
        if compute != code:
            continue
        if compute not in found:
            found.append(compute)
    return tuple(found)


def validate_compile_commands(path: Path) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    commands: dict[str, str] = {}
    for entry in entries:
        source = relative_source(
            str(entry.get("file", "")), str(entry.get("directory", ""))
        )
        if source is None or not source.endswith(".cu"):
            continue
        command = entry.get("command")
        if command is None:
            command = " ".join(str(value) for value in entry.get("arguments", []))
        commands[source] = str(command)

    errors: list[str] = []
    for required in REQUIRED_SOURCES:
        if required not in commands:
            errors.append(f"missing CUDA compile command for {required}")
    for source, command in sorted(commands.items()):
        actual = command_sms(command)
        expected = expected_sms(source)
        if set(actual) != set(expected) or len(actual) != len(expected):
            errors.append(
                f"{source}: gencode {list(actual)} != expected {list(expected)}"
            )
    return errors


def validate_archive_listing(listing: str) -> list[str]:
    actual = set(ARCHIVE_RE.findall(listing))
    missing = [sm for sm in ALL_SMS if sm not in actual]
    extra = sorted(actual.difference(ALL_SMS))
    errors: list[str] = []
    if missing:
        errors.append(f"archive is missing CUDA SMs: {','.join(missing)}")
    if extra:
        errors.append(f"archive has undeclared CUDA SMs: {','.join(extra)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-commands", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--library", type=Path)
    source.add_argument("--cuobjdump-list", type=Path)
    args = parser.parse_args()

    errors = validate_compile_commands(args.compile_commands)
    if args.library is not None:
        result = subprocess.run(
            ["cuobjdump", "--list-elf", str(args.library)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            errors.append(f"cuobjdump failed: {result.stderr.strip()}")
            listing = ""
        else:
            listing = result.stdout
    else:
        listing = args.cuobjdump_list.read_text(encoding="utf-8")
    errors.extend(validate_archive_listing(listing))

    if errors:
        print("CUDA fat gencode audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("CUDA fat gencode audit: all ten SMs and per-source intersections OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
