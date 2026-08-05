#!/usr/bin/env python3
"""Require public documentation updates for every feature checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The per-capability status surface is docs/STATUS.md, NOT README.md: pointing
# this obligation at the README is what drifted it from a landing page into a
# status log. README.md changes only when a user-visible headline shifts, which
# is a judgement call this checker deliberately does not force.
PUBLIC_CHECKPOINTS = ("docs/STATUS.md", "docs/BENCHMARKS.md")

# The public feature surface is docs/FEATURES.md: the user-facing comparison of
# what we support against vLLM, SGLang and llama.cpp. It is NOT owed by every
# checkpoint (most commits move no feature row), so it has its own narrower
# trigger set: the area matrices that define feature/model/backend/quant state,
# and the model implementations themselves. AGENTS.md already requires those
# matrices to move in the same change as the code, so this only mirrors that
# obligation onto the public surface.
FEATURE_CHECKPOINT = "docs/FEATURES.md"
FEATURE_TRIGGER_PREFIXES = ("src/vllm/model_executor/models/",)
FEATURE_TRIGGER_FILES = {
    ".agents/backend-matrix.md",
    ".agents/feature-matrix.md",
    ".agents/model-matrix.md",
    ".agents/quantization-matrix.md",
}

CHECKPOINT_PREFIXES = (
    ".agents/completed/",
    ".agents/specs/",
    ".github/workflows/",
    "cmake/",
    "examples/",
    "include/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)
CHECKPOINT_FILES = {
    "CMakeLists.txt",
    ".agents/backend-matrix.md",
    ".agents/coordination.md",
    ".agents/engine-matrix.md",
    ".agents/feature-matrix.md",
    ".agents/kernel-matrix.md",
    ".agents/model-matrix.md",
    ".agents/parity-ledger.md",
    ".agents/porting-inventory.md",
    ".agents/quantization-matrix.md",
    ".agents/roadmap_v1.md",
    ".agents/state.md",
}


def is_checkpoint_path(path: str) -> bool:
    """Return whether a changed path advances a feature/iteration checkpoint."""
    return path in CHECKPOINT_FILES or path.startswith(CHECKPOINT_PREFIXES)


def is_feature_path(path: str) -> bool:
    """Return whether a changed path can move a row in the public feature matrix."""
    return path in FEATURE_TRIGGER_FILES or path.startswith(FEATURE_TRIGGER_PREFIXES)


def _preview(triggers: list[str]) -> str:
    preview = ", ".join(triggers[:5])
    if len(triggers) > 5:
        preview += f", ... (+{len(triggers) - 5})"
    return preview


def checkpoint_errors(paths: set[str]) -> list[str]:
    """Return missing-public-document errors for one atomic change."""
    errors: list[str] = []

    triggers = sorted(path for path in paths if is_checkpoint_path(path))
    if triggers:
        errors += [
            "feature/iteration checkpoint changed "
            f"({_preview(triggers)}) but did not update {path} in the same change"
            for path in PUBLIC_CHECKPOINTS
            if path not in paths
        ]

    feature_triggers = sorted(path for path in paths if is_feature_path(path))
    if feature_triggers and FEATURE_CHECKPOINT not in paths:
        errors.append(
            "feature/model/backend/quantization surface changed "
            f"({_preview(feature_triggers)}) but did not update "
            f"{FEATURE_CHECKPOINT} in the same change; update the row this "
            "moves (support mark, gate, or the Not-supported-yet table) so the "
            "public feature matrix cannot drift from the area matrices"
        )

    return errors


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def commit_paths(commit: str) -> set[str]:
    parents = git("rev-list", "--parents", "-n", "1", commit).split()[1:]
    if parents:
        output = git("diff", "--name-only", parents[0], commit)
    else:
        output = git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", commit
        )
    return {line for line in output.splitlines() if line}


def commits_in_range(base: str, head: str) -> list[str]:
    try:
        git("cat-file", "-e", f"{base}^{{commit}}")
    except subprocess.CalledProcessError:
        return [head]
    output = git("rev-list", "--reverse", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--commit", default=None, help="check one commit")
    source.add_argument(
        "--staged", action="store_true", help="check the current staged change"
    )
    parser.add_argument("--base", help="check every commit after this revision")
    parser.add_argument("--head", help="range endpoint (requires --base)")
    args = parser.parse_args()
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    if args.base is not None and (args.commit is not None or args.staged):
        parser.error("a revision range cannot be combined with --commit/--staged")
    return args


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    if args.staged:
        paths = set(git("diff", "--cached", "--name-only").splitlines())
        failures.extend(f"staged change: {error}" for error in checkpoint_errors(paths))
    else:
        commits = (
            commits_in_range(args.base, args.head)
            if args.base is not None
            else [args.commit or "HEAD"]
        )
        for commit in commits:
            short = git("rev-parse", "--short", commit)
            failures.extend(
                f"commit {short}: {error}"
                for error in checkpoint_errors(commit_paths(commit))
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "Update docs/STATUS.md and docs/BENCHMARKS.md with the current "
            "stage/result, including explicit pending or void outcomes, and "
            "docs/FEATURES.md when the feature/model/backend/quantization "
            "surface moves.",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: feature checkpoints update docs/STATUS.md and docs/BENCHMARKS.md, "
        "and feature-surface changes update docs/FEATURES.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
