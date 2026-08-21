#!/usr/bin/env python3
"""Keep every recipe on docs/QUICKSTART.md spelled the way the tree ships it.

The quickstart page is the first thing a new reader runs, and it is the one
public document whose lines are meant to be pasted rather than read. Two of its
spellings are decided elsewhere in this tree and can drift without anybody
noticing, because a wrong one still renders as valid Markdown:

  1. THE IMAGE TAG. `release/container-matrix.json` decides which tags the
     publish workflow pushes. A page naming `:latest-gpu` or
     `ghcr.io/mudler/vllm-cpp` looks right and pulls nothing.
  2. THE `--model` VALUE. `ParseModelReference` in
     `src/vllm/transformers_utils/model_resolver.cpp` decides which values the
     loader accepts. `Qwen3-0.6B` with the organisation dropped is not a
     repository identifier, and the reader learns that from a refusal instead
     of from this gate.

Both sets are DERIVED here rather than restated: the tags come out of the
container matrix, and the grammar mirrors `IsValidHfRepoId`
(`src/vllm/transformers_utils/hf_hub.cpp:254`) plus the last-colon tag split
(`model_resolver.cpp:384`).

WHAT `PENDING(#N)` DOES, AND WHAT IT DELIBERATELY DOES NOT DO. The page is
allowed to carry a recipe before the thing it names exists: the first container
publish has not happened, and a table row cannot be filled in before somebody
runs it. A `PENDING(#N)` marker declares exactly that, in the reader's view, with
the issue that owes the work. It excuses a MISSING value. It never excuses a
WRONG one:

  - a `--model` cell may read `PENDING` only inside the scope of a marker, and
    every other `--model` value is still parsed;
  - an image tag is checked WHATEVER the markers say, because a tag that is
    merely unpublished is spelled the same as the one that will be published,
    and a tag that is misspelled is wrong today and wrong after the publish.

A marker's scope is the fenced code block it sits in, or, outside a fence, the
single line it sits on (a Markdown table row is one line).

    scripts/check-quickstart-recipes.py
    scripts/check-quickstart-recipes.py --page docs/QUICKSTART.md

Row ROAD-V1-QUICKSTART, issue #1281.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/QUICKSTART.md"
MATRIX = ROOT / "release/container-matrix.json"

# `PENDING(#1511)`. The issue number is REQUIRED: an escape hatch nobody owns is
# how a placeholder becomes permanent.
PENDING_MARKER_RE = re.compile(r"\bPENDING\(#\d+\)")

# The placeholder a marker authorises. Deliberately NOT the marker itself, so a
# `--model PENDING(#1511)` cannot authorise its own placeholder: the excuse has
# to be written somewhere a reader will see it as a note rather than as a value.
PENDING_PLACEHOLDER = "PENDING"

# Every registry reference on the page, tag or no tag, so a misspelled package
# name is caught beside a misspelled tag.
GHCR_REF_RE = re.compile(r"ghcr\.io/[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?")

# `--model VALUE` and `--model=VALUE`.
MODEL_FLAG_RE = re.compile(r"--model(?:=|\s+)(\S+)")

# Markdown and shell punctuation that wraps a value without being part of it.
VALUE_TRIM = "`'\"|,;.*_ \\"


# ---------------------------------------------------------------------------
# The `--model` grammar, mirrored from the C++ that decides it.
# ---------------------------------------------------------------------------


def is_valid_hf_repo_id(repo_id: str) -> bool:
    """Mirror of `IsValidHfRepoId`, `src/vllm/transformers_utils/hf_hub.cpp:254`.

    Base characters `[A-Za-z0-9_]` are always valid, the special characters
    `[/.-]` must sit between two base characters, and exactly one `/` is
    required. ASCII only, because the C++ asks `IsAlphanumeric` and a non-ASCII
    letter is not one.
    """
    if not repo_id or len(repo_id) > 256:
        return False
    slashes = 0
    previous_was_special = True
    for char in repo_id:
        if char.isascii() and (char.isalnum() or char == "_"):
            previous_was_special = False
        elif char in "/.-":
            if previous_was_special:
                return False
            slashes += char == "/"
            previous_was_special = True
        else:
            return False
    return not previous_was_special and slashes == 1


def _looks_like_local_path(value: str) -> bool:
    """The two local shapes, decided by SHAPE because a page cannot be stat'ed.

    `ParseModelReference` probes the file system first and this checker cannot,
    so it accepts the spellings that can only ever be a path. A bare
    `models/qwen` is NOT one of them: that is also a well-formed repository
    identifier, and it is read as one.
    """
    if value.startswith(("/", "./", "../", "~")):
        return True
    if "\\" in value:
        return True
    return bool(re.match(r"^[A-Za-z]:[\\/]", value))


def classify_model_value(value: str) -> str | None:
    """Which of the shapes `value` is, or None when the loader refuses it.

    The order mirrors `ParseModelReference`, `model_resolver.cpp:365-402`:
    the local shapes first, then the last-colon tag split, then the bare
    repository identifier.
    """
    if not value:
        return None
    if _looks_like_local_path(value):
        return "local gguf file" if value.endswith(".gguf") else "local path"
    colon = value.rfind(":")
    if colon != -1 and colon + 1 < len(value):
        if is_valid_hf_repo_id(value[:colon]):
            return "hub gguf file"
    if is_valid_hf_repo_id(value):
        return "hub snapshot"
    return None


# ---------------------------------------------------------------------------
# The published tag set, derived from the container matrix.
# ---------------------------------------------------------------------------


def published_tags(matrix: dict) -> tuple[set[str], list[tuple[str, re.Pattern[str]]]]:
    """The moving and main tags by name, and one pattern per version tag.

    A version tag is matched by PATTERN rather than by name because the page
    outlives any one version: what this gate owns is the lane suffix, which is
    the half that gets misspelled.
    """
    exact: set[str] = set()
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for lane in matrix["lanes"]:
        exact.update(lane["moving_tags"])
        exact.add(lane["main_tag"])
        template = lane["version_tag"]
        body = re.escape(template).replace(
            re.escape("{version}"), r"\d+\.\d+\.\d+(?:[.\-][0-9A-Za-z.]+)?"
        )
        patterns.append((template, re.compile(rf"^{body}$")))
    return exact, patterns


# ---------------------------------------------------------------------------
# Marker scope.
# ---------------------------------------------------------------------------


def pending_scope(text: str) -> list[bool]:
    """One flag per line: is a `PENDING(#N)` marker in scope for this line?

    Inside a fenced block the marker covers the WHOLE block, because a shell
    comment sits above the command it excuses rather than on it. Outside a
    fence it covers only its own line, which for a Markdown table is exactly
    one row.
    """
    lines = text.splitlines()
    scope = [False] * len(lines)
    fence_start: int | None = None
    fence_marked = False
    for index, raw in enumerate(lines):
        if raw.strip().startswith("```"):
            if fence_start is None:
                fence_start, fence_marked = index, False
            else:
                if fence_marked:
                    for i in range(fence_start, index + 1):
                        scope[i] = True
                fence_start = None
            continue
        if fence_start is not None:
            fence_marked = fence_marked or bool(PENDING_MARKER_RE.search(raw))
        elif PENDING_MARKER_RE.search(raw):
            scope[index] = True
    return scope


# ---------------------------------------------------------------------------
# The two rules.
# ---------------------------------------------------------------------------


def image_errors(text: str, matrix: dict) -> list[str]:
    """Rule 1. Every registry reference names the package and a published tag."""
    package = matrix["package"]
    exact, patterns = published_tags(matrix)
    known = ", ".join(sorted(exact) + [t for t, _ in patterns])
    errors: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        for reference in GHCR_REF_RE.findall(raw):
            repository, _, tag = reference.partition(":")
            if repository != package:
                errors.append(
                    f"line {lineno}: image `{reference}` names the repository "
                    f"`{repository}`, and release/container-matrix.json publishes "
                    f"`{package}`"
                )
                continue
            if not tag:
                continue
            if tag in exact or any(pattern.match(tag) for _, pattern in patterns):
                continue
            errors.append(
                f"line {lineno}: image `{reference}` names the tag `{tag}`, which "
                f"release/container-matrix.json does not publish. It publishes: "
                f"{known}. A PENDING marker does not excuse a tag: an unpublished "
                f"tag is spelled exactly like the published one."
            )
    return errors


def model_errors(text: str) -> list[str]:
    """Rule 2. Every `--model` value parses, or is a marked placeholder."""
    scope = pending_scope(text)
    errors: list[str] = []
    for index, raw in enumerate(text.splitlines()):
        lineno = index + 1
        for match in MODEL_FLAG_RE.findall(raw):
            value = match.strip(VALUE_TRIM)
            if not value:
                errors.append(f"line {lineno}: `--model` carries no value")
                continue
            if value == PENDING_PLACEHOLDER:
                if scope[index]:
                    continue
                errors.append(
                    f"line {lineno}: `--model {value}` is a placeholder with no "
                    f"`PENDING(#N)` marker in scope. A row enters this page after "
                    f"somebody ran it; until then, say so and name the issue that "
                    f"owes the run."
                )
                continue
            if classify_model_value(value) is None:
                errors.append(
                    f"line {lineno}: `--model {value}` does not parse. The loader "
                    f"accepts a local path, a `.gguf` file, `org/repo`, or "
                    f"`org/repo:TAG` (ParseModelReference, "
                    f"src/vllm/transformers_utils/model_resolver.cpp:365)"
                )
    return errors


def page_errors(text: str, matrix: dict) -> list[str]:
    return image_errors(text, matrix) + model_errors(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    args = parser.parse_args()

    if not args.page.exists():
        print(
            f"ERROR: {args.page} is missing. It is the page a new reader is sent "
            "to, and README.md points at it.",
            file=sys.stderr,
        )
        return 1
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    errors = page_errors(args.page.read_text(encoding="utf-8"), matrix)
    if errors:
        print(f"ERROR: {args.page} carries a recipe nobody can run:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"OK: every image tag and `--model` value on {args.page.name} is spelled "
          "the way this tree ships it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
