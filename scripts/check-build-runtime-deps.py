#!/usr/bin/env python3
"""Refuse a build environment that cannot compile the feature it ships.

Issue #1517. `docker/Dockerfile` installed `libssl3` in its RUNTIME stage and
never installed `libssl-dev` in the stage that COMPILES. `CMakeLists.txt`
answers a failed `find_package(OpenSSL)` with a DOWNGRADE -- a warning,
`VLLM_CPP_HF_DOWNLOAD` forced OFF, and a green build -- so every image the file
produced shipped a `vllm-server` that could not speak https, while carrying the
runtime library that made it look as if it could. `ldd` named four libraries and
none of them was `libssl`. The same omission was in `.github/workflows/
release.yml`'s CUDA jobs, which build inside `nvidia/cuda:...-devel` and so
published archives with the fetch disabled.

Nothing in this tree related the two halves. Every other gate was green: the
Dockerfile was well-formed, the matrix agreed with it, the archive audit passed,
and the token and unit suites never touch a build option. The defect is only
visible as a RELATION between two files, which is what this checker reads.

Two assertions, one per surface:

  Dockerfile   Every stage that runs a `scripts/build-*-release.sh` must reach
               `libssl-dev` through an apt install in itself or an ancestor
               stage, and some runtime stage must install `libssl3`. Either
               half alone is the #1517 shape.

  release.yml  Every job that runs a release build script is classified: it
               either installs `libssl-dev` or is named in EXEMPT_LANES with a
               reason. A NEW lane belongs to neither set and reddens this gate,
               which is the property that keeps the list from rotting.

Usage: scripts/check-build-runtime-deps.py [--dockerfile PATH] [--workflow PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker/Dockerfile"
WORKFLOW = ROOT / ".github/workflows/release.yml"

# The development files that make `find_package(OpenSSL)` succeed, and the
# runtime library the resulting binary loads. Debian splits them; the split is
# the whole reason #1517 could exist.
DEV_PACKAGE = "libssl-dev"
RUNTIME_PACKAGE = "libssl3"

# A stage or job "compiles the shipped server" when it invokes one of these.
# Matching the SCRIPT rather than a stage name means a renamed stage is still
# covered, and a stage that only copies artefacts is still exempt. The pattern
# deliberately spans EVERY release lane, macOS and Windows included, so those
# two do not fall silently outside the gate: they appear in EXEMPT_LANES with
# the reason each cannot use an apt package, which keeps their recorded TLS debt
# visible here rather than only in a spec.
BUILD_SCRIPTS = re.compile(r"scripts/build-[a-z0-9-]+-release\.(?:sh|ps1)")

# Release jobs that invoke a build script and are NOT required to carry
# libssl-dev. Each entry is a decision with a recorded reason, not an oversight.
# A job that invokes a build script and appears in neither this mapping nor the
# libssl-dev-carrying set is an error, so adding a Linux lane reddens the gate.
EXEMPT_LANES = {
    "cpu_musl": (
        "literal-static musl. scripts/build-cpu-release.sh:20-45 sets "
        "VLLM_CPP_HF_DOWNLOAD=OFF for this artifact on purpose, because "
        "validate-release-archive.py refuses any ELF dependency on a "
        "literal-static archive. The binary REFUSES org/repo with a message "
        "naming the build options, which is the documented disposition"
    ),
    "metal_arm64": (
        "macOS, and no apt. The lane needs an OPENSSL_ROOT_DIR hint for Homebrew's "
        "keg-only openssl, and is recorded UNRESOLVED under `## Owed` in "
        ".agents/specs/hf-model-download.md"
    ),
    "mlx_arm64": (
        "macOS. Same lane and the same recorded debt as metal_arm64"
    ),
    "cpu_windows": (
        "Windows MSVC, and no apt. cpp-httplib has no Schannel backend, so this "
        "lane needs an OpenSSL for find_package to point at rather than a Debian "
        "package. Recorded UNRESOLVED under `## Owed` in "
        ".agents/specs/hf-model-download.md"
    ),
    "vulkan_windows": (
        "Windows MSVC. Same lane and the same recorded debt as cpu_windows"
    ),
}


def _load_lexer():
    """Reuse check-container-matrix.py's Dockerfile lexer.

    It already joins continuations and drops comments, and
    tests/scripts/test_check_container_matrix.py pins all three behaviours. A
    second hand-written copy would be a second thing to keep correct.
    """
    path = ROOT / "scripts/check-container-matrix.py"
    spec = importlib.util.spec_from_file_location("_ccm_lexer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dockerfile_instructions


dockerfile_instructions = _load_lexer()

FROM_AS = re.compile(r"^(?P<base>\S+)(?:\s+AS\s+(?P<name>[A-Za-z0-9_.-]+))?", re.IGNORECASE)


def parse_stages(text: str) -> list[dict]:
    """Split a Dockerfile into stages: name, parent stage, RUN bodies.

    `parent` is the stage this one derives from, or None when the base is an
    external image. An `ARG`-substituted base like `${UBUNTU_BASE}` is external.
    """
    stages: list[dict] = []
    by_name: dict[str, dict] = {}
    for keyword, body in dockerfile_instructions(text):
        if keyword == "FROM":
            match = FROM_AS.match(body.strip())
            base = match.group("base") if match else body.strip()
            name = match.group("name") if match else None
            stage = {
                "name": name,
                "base": base,
                "parent": base if base in by_name else None,
                "runs": [],
            }
            stages.append(stage)
            if name:
                by_name[name] = stage
        elif keyword == "RUN" and stages:
            stages[-1]["runs"].append(body)
    return stages


def apt_packages(stage: dict) -> set[str]:
    """Packages the stage apt-installs, whitespace-split from the install body."""
    found: set[str] = set()
    for body in stage["runs"]:
        for match in re.finditer(r"apt-get\s+install\b(?P<rest>.*)", body):
            for token in match.group("rest").split():
                if token.startswith("-") or token in {"&&", "||", ";"}:
                    continue
                if token in {"apt-get", "rm", "rf"}:
                    break
                found.add(token)
    return found


def reachable_packages(stage: dict, by_name: dict[str, dict]) -> set[str]:
    """Packages this stage has, including everything an ancestor installed."""
    seen: set[str] = set()
    current: dict | None = stage
    guard = 0
    while current is not None and guard < 64:
        seen |= apt_packages(current)
        parent = current["parent"]
        current = by_name.get(parent) if parent else None
        guard += 1
    return seen


def check_dockerfile(text: str, label: str = "docker/Dockerfile") -> list[str]:
    errors: list[str] = []
    stages = parse_stages(text)
    by_name = {stage["name"]: stage for stage in stages if stage["name"]}

    compiling = [
        stage
        for stage in stages
        if any(BUILD_SCRIPTS.search(body) for body in stage["runs"])
    ]
    if not compiling:
        errors.append(
            f"{label}: no stage runs a release build script, so this checker "
            "would pass vacuously. Either a stage was renamed out of "
            "BUILD_SCRIPTS or the file stopped building the server"
        )

    for stage in compiling:
        name = stage["name"] or f"<unnamed stage from {stage['base']}>"
        if DEV_PACKAGE not in reachable_packages(stage, by_name):
            errors.append(
                f"{label}: stage {name!r} compiles the server but neither it "
                f"nor any ancestor apt-installs {DEV_PACKAGE}. "
                "CMakeLists.txt DOWNGRADES a failed find_package(OpenSSL) to a "
                "warning and ships a server that cannot fetch from "
                "huggingface.co, so this build is green and wrong (#1517)"
            )

    runtime_carriers = [
        stage["name"] or stage["base"]
        for stage in stages
        if RUNTIME_PACKAGE in apt_packages(stage)
    ]
    if not runtime_carriers:
        errors.append(
            f"{label}: no stage installs {RUNTIME_PACKAGE}, so a server linked "
            f"against {DEV_PACKAGE} would fail to start in the published image. "
            "The two halves ship together or not at all"
        )
    return errors


def job_blocks(text: str) -> dict[str, str]:
    """Map job name to its literal block from a workflow's `jobs:` mapping."""
    body = text.split("\njobs:\n", 1)[-1]
    blocks: dict[str, str] = {}
    for match in re.finditer(
        r"(?ms)^  (?P<name>[A-Za-z0-9_-]+):\n(?P<block>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        body,
    ):
        blocks[match.group("name")] = match.group("block")
    return blocks


def installs_dev_package(block: str) -> bool:
    """True when an apt install in this block names the development package.

    Shell line continuations are joined FIRST. The CUDA jobs write their package
    list across a `\\` continuation, and a matcher that stops at the newline
    reports those jobs as carrying nothing -- a gate that reddens a correct lane
    is as useless as one that passes a broken one.
    """
    joined = re.sub(r"\\\n\s*", " ", block)
    for match in re.finditer(r"apt-get\s+install\b(?P<rest>[^\n]*)", joined):
        if re.search(rf"(?:^|\s){re.escape(DEV_PACKAGE)}(?:\s|$)", match.group("rest")):
            return True
    return False


def check_workflow(text: str, label: str = ".github/workflows/release.yml") -> list[str]:
    errors: list[str] = []
    blocks = job_blocks(text)
    building = {
        name: block for name, block in blocks.items() if BUILD_SCRIPTS.search(block)
    }
    if not building:
        errors.append(
            f"{label}: no job runs a release build script, so this checker would "
            "pass vacuously"
        )

    for name, block in sorted(building.items()):
        if name in EXEMPT_LANES:
            if installs_dev_package(block):
                errors.append(
                    f"{label}: job {name!r} is listed in EXEMPT_LANES but now "
                    f"installs {DEV_PACKAGE}. Remove the exemption and its "
                    "reason rather than carrying a stale one"
                )
            continue
        if not installs_dev_package(block):
            errors.append(
                f"{label}: job {name!r} builds a release artifact but never "
                f"apt-installs {DEV_PACKAGE}, and it is not in EXEMPT_LANES. "
                "It either inherits the package from a runner image by accident "
                "or ships an archive whose --model org/repo is disabled (#1517). "
                "Name the package, or add the job to EXEMPT_LANES with a reason"
            )

    for name in sorted(EXEMPT_LANES):
        if name not in blocks:
            errors.append(
                f"{label}: EXEMPT_LANES names job {name!r}, which no longer "
                "exists. A stale exemption hides the next lane that needs one"
            )
        elif name not in building:
            errors.append(
                f"{label}: EXEMPT_LANES names job {name!r}, which no longer "
                "runs a release build script. Drop the exemption"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockerfile", type=Path, default=DOCKERFILE)
    parser.add_argument("--workflow", type=Path, default=WORKFLOW)
    args = parser.parse_args()

    errors: list[str] = []
    for path, checker, label in (
        (args.dockerfile, check_dockerfile, str(args.dockerfile)),
        (args.workflow, check_workflow, str(args.workflow)),
    ):
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"{label} does not exist")
            continue
        errors.extend(checker(text, label))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"{len(errors)} build/runtime dependency defect(s)", file=sys.stderr
        )
        return 1
    print(
        "build/runtime deps OK: every compiling stage and lane carries "
        f"{DEV_PACKAGE}, the runtime carries {RUNTIME_PACKAGE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
