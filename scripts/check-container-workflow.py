#!/usr/bin/env python3
"""Static least-privilege gate for .github/workflows/containers.yml.

The container publish workflow can write to a public registry. Everything that
makes that safe is structural -- which job holds which permission, which stages
a tag gates, what runs before a push -- and none of it is visible at review time
unless something reads the file and refuses.

Deliberately text-based, like scripts/check-release-workflow.py: the repository
has no PyYAML dependency and this gate is not worth adding one for.

Usage: scripts/check-container-workflow.py [--workflow PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/containers.yml"

# Jobs that may hold registry write, and the single job that may hold OIDC.
REGISTRY_WRITE_JOBS = frozenset({"publish", "manifest", "promote"})
OIDC_JOBS = frozenset({"attest"})
# `promote` moves :latest* and is RELEASE-ONLY. publish/manifest/attest also run
# for a main publish, which ships moving :main-<lane> tags -- so they are gated
# on `publishes`, which is true for a tag or main but never for a pull request.
RELEASE_ONLY_JOBS = ("promote",)
PUBLISH_JOBS = ("publish", "manifest", "attest")
TAG_GATED_JOBS = PUBLISH_JOBS + RELEASE_ONLY_JOBS
RELEASE_GUARD = "if: needs.plan.outputs.is_release == 'true'"
PUBLISH_GUARD = "if: needs.plan.outputs.publishes == 'true'"

# The two jobs that run `docker buildx build`. They build the IDENTICAL image,
# so anything true of the build in one has to be true in the other: a cap that
# held in verify and not in publish would let publish die on exactly the build
# verify had just proved.
BUILDING_JOBS = ("verify", "publish")
# Issue #1548. Each .cu in the cuda lane is compiled for the ten device
# architectures scripts/build-linux-accelerator-release.sh sets, so one compiler
# process there holds many times the resident set of a cpu or vulkan
# translation unit. Run 32447481128 handed that lane $(nproc) and died at object
# 512 of 787 with exit 143.
#
# This gate is on the SHAPE and not on the number. Lane-awareness is the
# contract. The value is a measurement that a future run may move, and a gate
# that reds on tuning it would be the defect rather than the discipline.
CUDA_JOBS_CAP = re.compile(
    r'if \[ "\$\{\{ matrix\.lane \}\}" = "cuda" \]; then jobs=(\d+); fi'
)
UNCAPPED_JOBS = 'JOBS=$(nproc)'
JOB_TIMEOUT = re.compile(r"(?m)^    timeout-minutes: (\d+)$")


def job_names(text: str) -> list[str]:
    return re.findall(r"(?m)^  ([a-zA-Z0-9_-]+):\n", text.split("\njobs:\n", 1)[-1])


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    return match.group(1) if match else ""


def step_order(block: str, needle: str) -> int | None:
    """Index of the step containing `needle`, or None."""
    starts = [match.start() for match in re.finditer(r"(?m)^      - ", block)]
    starts.append(len(block))
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        if needle in block[start:end]:
            return index
    return None


def validate(text: str) -> list[str]:
    errors: list[str] = []

    for fragment in (
        "  push:\n    tags: ['v*']",
        "  pull_request:",
        # A manual entry point must exist; whether it takes inputs is not this
        # gate's business, so match the key rather than the empty-mapping form.
        "  workflow_dispatch:",
        "permissions:\n  contents: read",
    ):
        if fragment not in text:
            errors.append(f"workflow is missing required global contract: {fragment!r}")

    if "continue-on-error" in text:
        errors.append(
            "containers workflow may not continue after an error: a lane that failed "
            "its gate must not reach a push"
        )

    names = job_names(text)
    for required in ("plan", "verify", *TAG_GATED_JOBS):
        if required not in names:
            errors.append(f"workflow is missing the {required!r} job")
    if errors:
        return errors

    blocks = {name: job_block(text, name) for name in names}

    for name, block in blocks.items():
        has_packages_write = re.search(r"(?m)^\s+packages: write$", block) is not None
        if has_packages_write and name not in REGISTRY_WRITE_JOBS:
            errors.append(
                f"job {name!r} holds packages: write; only {sorted(REGISTRY_WRITE_JOBS)} "
                "may write to the registry"
            )
        if name in REGISTRY_WRITE_JOBS and not has_packages_write:
            errors.append(f"job {name!r} must declare packages: write to publish")

        has_oidc = re.search(r"(?m)^\s+id-token: write$", block) is not None
        if has_oidc and name not in OIDC_JOBS:
            errors.append(
                f"job {name!r} holds id-token: write; only {sorted(OIDC_JOBS)} may sign"
            )
        if name in OIDC_JOBS and not has_oidc:
            errors.append(f"job {name!r} must hold id-token: write to attest")

        if re.search(r"(?m)^\s+contents: write$", block):
            errors.append(
                f"job {name!r} holds contents: write; publishing an image never needs "
                "to write the repository"
            )

    verify_block = blocks["verify"]
    if "docker/login-action" in verify_block or "docker push" in verify_block:
        errors.append(
            "the verify job must not log in to or push to a registry: it is the job a "
            "pull request runs"
        )
    if "validate-container-image.py" not in verify_block:
        errors.append("the verify job must run scripts/validate-container-image.py")

    for name in RELEASE_ONLY_JOBS:
        if RELEASE_GUARD not in blocks[name]:
            errors.append(
                f"job {name!r} must be gated on {RELEASE_GUARD!r}: :latest follows a "
                "RELEASE, and a main publish must never move it"
            )
    for name in PUBLISH_JOBS:
        if PUBLISH_GUARD not in blocks[name]:
            errors.append(
                f"job {name!r} must be gated on {PUBLISH_GUARD!r}: it publishes, so a "
                "pull request must not reach it"
            )

    # A main publish must not be able to write a version tag or :latest.
    plan_text = blocks["plan"]
    if 'GITHUB_EVENT_NAME}" != "pull_request"' not in plan_text:
        errors.append(
            "the plan job must exclude pull_request from is_main: a fork PR has no "
            "credentials and must never be classified as a publish"
        )
    manifest = blocks["manifest"]
    if "main-${lane}" not in manifest:
        errors.append(
            "the manifest job must tag a main publish :main-<lane>, never a version"
        )
    promote_block = blocks["promote"]
    if "main-" in promote_block:
        errors.append(
            "the promote job must not touch main tags; it moves :latest* only"
        )

    publish = blocks["publish"]
    immutable_guard = step_order(publish, "already exists; version tags are immutable")
    validate_step = step_order(publish, "validate-container-image.py")
    push_step = step_order(publish, "docker push")
    if immutable_guard is None:
        errors.append(
            "the publish job must refuse to overwrite an existing version tag: GHCR "
            "does not enforce immutability for us"
        )
    if validate_step is None:
        errors.append("the publish job must validate the image it pushes")
    if push_step is None:
        errors.append("the publish job must push by digest")
    if None not in (immutable_guard, push_step) and immutable_guard > push_step:
        errors.append(
            "the immutable-tag guard runs AFTER the push; it has to run before"
        )
    if None not in (validate_step, push_step) and validate_step > push_step:
        errors.append(
            "the publish job pushes BEFORE validating: the bytes that ship must be the "
            "bytes that passed"
        )

    if "plan" in blocks:
        plan = blocks["plan"]
        for checker in ("check-container-matrix.py", "check-container-workflow.py"):
            if checker not in plan:
                errors.append(f"the plan job must run scripts/{checker}")
        if 'test "${GITHUB_REF_NAME}" = "v${version}"' not in plan:
            errors.append(
                "the plan job must verify the tag matches the tree's version: a tag is "
                "untrusted input until it does"
            )

    # The reduced pull-request matrix must never reach a push. publish takes the
    # full release matrix, computed with --release, so a lane skipped on PRs is
    # still built and validated before it ships.
    plan_block = blocks["plan"]
    # Match the publish= assignment exactly. Looking for "--build-matrix
    # --release" anywhere in the job passes on the verify branch's own use of
    # the flag, which is how the first version of this check failed to notice
    # the publish matrix losing it.
    if 'publish=$(python3 scripts/container_tags.py --build-matrix --release)' not in plan_block:
        errors.append(
            "the plan job must compute the publish matrix with --release: the "
            "reduced pull-request matrix must never decide what gets published"
        )
    if "fromJSON(needs.plan.outputs.publish_matrix)" not in publish:
        errors.append(
            "the publish job must consume publish_matrix, not the reduced "
            "verify_matrix"
        )
    if "fromJSON(needs.plan.outputs.verify_matrix)" not in verify_block:
        errors.append("the verify job must consume verify_matrix")

    # Build parallelism and a stated time budget, both issue #1548.
    for name in BUILDING_JOBS:
        block = blocks[name]
        if UNCAPPED_JOBS in block:
            errors.append(
                f"job {name!r} hands JOBS=$(nproc) to the build: the ten-SM fat cuda "
                "lane compiles every .cu for ten device architectures and cannot take "
                "the whole runner"
            )
        if CUDA_JOBS_CAP.search(block) is None:
            errors.append(
                f"job {name!r} must lower build parallelism for the cuda lane and only "
                "that lane. The cpu and vulkan lanes are not failing and keep "
                "$(nproc)"
            )
        if JOB_TIMEOUT.search(block) is None:
            errors.append(
                f"job {name!r} must declare timeout-minutes: under the six-hour default "
                "a hang, a reclaimed runner and an exhausted one all report the same "
                "exit 143, and nothing separates them"
            )

    promote = blocks["promote"]
    if "container_tags.py" not in promote:
        errors.append(
            "the promote job must derive moving tags from release/container-matrix.json "
            "via scripts/container_tags.py, not from a hand-written list"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=WORKFLOW)
    args = parser.parse_args()

    try:
        text = args.workflow.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {args.workflow} does not exist")
        return 1

    errors = validate(text)
    if errors:
        print("ERROR: the container workflow does not meet its least-privilege contract:")
        for error in errors:
            print(f"  - {error}")
        return 1

    caps = sorted({match.group(1) for match in CUDA_JOBS_CAP.finditer(text)})
    budgets = sorted({match.group(1) for match in JOB_TIMEOUT.finditer(text)})
    print(
        "container workflow OK: least-privilege stages, tag-gated publish, validated "
        f"push, cuda lane at -j {','.join(caps)}, build budget {','.join(budgets)} min"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
