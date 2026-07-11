#!/usr/bin/env python3
"""Require public documentation updates for every feature checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CHECKPOINTS = ("README.md", "docs/BENCHMARKS.md")

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


def checkpoint_errors(paths: set[str]) -> list[str]:
    """Return missing-public-document errors for one atomic change."""
    triggers = sorted(path for path in paths if is_checkpoint_path(path))
    if not triggers:
        return []
    missing = [path for path in PUBLIC_CHECKPOINTS if path not in paths]
    if not missing:
        return []
    preview = ", ".join(triggers[:5])
    if len(triggers) > 5:
        preview += f", ... (+{len(triggers) - 5})"
    return [
        "feature/iteration checkpoint changed "
        f"({preview}) but did not update {path} in the same change"
        for path in missing
    ]


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
            "Update README.md and docs/BENCHMARKS.md with the current "
            "stage/result, including explicit pending or void outcomes.",
            file=sys.stderr,
        )
        return 1

    print("OK: feature checkpoints update README.md and docs/BENCHMARKS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
