#!/usr/bin/env python3
"""Validate the writing style of commit messages in an exact Git range."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_PROTOCOL_MARKER = "FOLLOWING_AGENTS_PROTOCOL"
TRAILER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:[ \t].+$")
CONTINUATION_LINE = re.compile(r"^[ \t]+\S")


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return re.split(r"\n[ \t]*\n+", normalized) if normalized else []


def _is_trailer_paragraph(paragraph: str) -> bool:
    lines = [line for line in paragraph.splitlines() if line.strip()]
    if not lines or TRAILER_LINE.fullmatch(lines[0]) is None:
        return False
    return all(
        TRAILER_LINE.fullmatch(line) is not None
        or CONTINUATION_LINE.fullmatch(line) is not None
        for line in lines[1:]
    )


def _has_authored_body(message: str) -> bool:
    normalized = message.replace("\r\n", "\n").replace("\r", "\n")
    _, separator, remainder = normalized.partition("\n")
    if not separator:
        return False
    paragraphs = _paragraphs(remainder)
    while paragraphs and _is_trailer_paragraph(paragraphs[-1]):
        paragraphs.pop()
    if paragraphs and paragraphs[-1] == RAW_PROTOCOL_MARKER:
        paragraphs.pop()
    return any(paragraph.strip() for paragraph in paragraphs)


def validate_commit_message(message: str) -> list[str]:
    """Return writing-style errors for one complete commit message."""

    errors: list[str] = []
    subject = message.replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0]
    if subject.rstrip().endswith("."):
        errors.append("subject must not end in a period")
    if not _has_authored_body(message):
        errors.append("authored body must not be empty")
    return errors


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Git command failed"
        raise ValueError(detail)
    return result.stdout.strip()


def _resolve_commit(repo: Path, revision: str) -> str:
    if not revision or "\x00" in revision or "\n" in revision:
        raise ValueError(f"invalid revision {revision!r}")
    if not revision.startswith("refs/"):
        candidates = (
            f"refs/heads/{revision}",
            f"refs/tags/{revision}",
            f"refs/remotes/{revision}",
        )
        matches = 0
        for candidate in candidates:
            result = subprocess.run(
                ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", candidate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                matches += 1
            elif result.returncode != 1:
                raise ValueError(f"could not resolve revision {revision!r}")
        if matches > 1:
            raise ValueError(f"ambiguous revision {revision!r}")
    resolved = _git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).splitlines()
    if len(resolved) != 1 or re.fullmatch(r"[0-9a-f]{40}", resolved[0]) is None:
        raise ValueError(f"revision {revision!r} did not resolve to one commit")
    return resolved[0]


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("could not establish commit ancestry")
    return result.returncode == 0


def validate_range(
    repo: Path,
    base: str,
    head: str,
    *,
    cutover: str | None,
) -> list[str]:
    """Validate the non-merge commits in an exact ``BASE..HEAD`` set."""

    base_oid = _resolve_commit(repo, base)
    head_oid = _resolve_commit(repo, head)
    if not _is_ancestor(repo, base_oid, head_oid):
        raise ValueError("range base must be an ancestor of range head")
    cutover_oid = _resolve_commit(repo, cutover) if cutover is not None else None
    if cutover_oid is not None and not _is_ancestor(repo, cutover_oid, head_oid):
        raise ValueError("cutover must be reachable from range head")

    commits = _git(repo, "rev-list", "--reverse", f"{base_oid}..{head_oid}")
    failures: list[str] = []
    for commit in (line for line in commits.splitlines() if line):
        parents = _git(repo, "show", "-s", "--format=%P", commit).split()
        if len(parents) > 1:
            continue
        if cutover_oid is not None:
            if _is_ancestor(repo, cutover_oid, commit):
                pass
            elif _is_ancestor(repo, commit, cutover_oid):
                continue
            else:
                raise ValueError(f"commit {commit} is incomparable with cutover")

        message = _git(repo, "show", "-s", "--format=%B", commit) + "\n"
        for error in validate_commit_message(message):
            failures.append(f"{commit[:12]}: {error}")
    return failures


def _range(value: str) -> tuple[str, str]:
    if value.count("..") != 1 or "..." in value:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    base, head = value.split("..", 1)
    if not base or not head:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    return base, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="revision_range", type=_range, required=True)
    parser.add_argument("--cutover")
    args = parser.parse_args()
    try:
        failures = validate_range(
            ROOT,
            *args.revision_range,
            cutover=args.cutover,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"commit style check FAILED: {exc}", file=sys.stderr)
        return 1
    if failures:
        print("commit style check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("OK: commit writing style")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
