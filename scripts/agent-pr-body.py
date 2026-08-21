#!/usr/bin/env python3
"""Hold a pull request body to the commit contract BEFORE the squash lands it.

This repository sets ``squash_merge_commit_message = PR_BODY``, so the body IS
the message of the commit that reaches ``main``. One guard already reads it, in
``.github/workflows/ci.yml`` (#848), and that guard is correct. It is also not a
precondition of merging: on #1257 it was still ``pending`` when the merge went
ahead, the body carried a malformed attribution value, and the result is a
message on ``main`` that can never be repaired, because ``main`` is never
force-pushed (#1262). This command is the operator's own local read of the same
bytes, in their shell instead of in a queue (#1263).

THE RULE IS NOT HERE. ``scripts/check-commit-trailers.py`` owns it, and this
tool passes it the body on standard input with the same ``--message-file -
--filled`` the CI step uses. One implementation, two callers, and nothing to
drift.

WHY THIS IS NOT A ONE-LINE PIPE. Measured before it was written::

    gh pr view 999999 --json body --jq .body | check-commit-trailers.py --message-file - --filled
    -> commit trailer check FAILED: the message is empty.

The forge had answered "could not resolve to a pull request". A pipe throws away
``gh``'s exit status, so an unreachable forge rendered as a verdict about a body
nobody had read. Here ``gh`` runs with its status captured and an unread body is
reported as ``REMOTE_UNVERIFIED``, which is the protocol's answer to unknown. A
skip is not.

THIS IS NOT A CHECKER. It reaches the network, so no gate may depend on it and
``scripts/agent-preflight.sh`` must never run it. Its OFFLINE half is
``--body-file``, which validates a message already on disk and never invokes
``gh``; that is the half the suite drives.

    scripts/agent-pr-body.py --pr 1263          # read the live body
    scripts/agent-pr-body.py --body-file draft  # the same contract, offline

Exit statuses: 0 the body will land clean, 1 the body was read and fails the
contract, 3 the body was NOT read. 2 belongs to ``argparse`` and is a usage
error. Refusing to conflate 1 and 3 is the whole point: this row exists because
"nobody looked" was allowed to pass for "looked and fine".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check-commit-trailers.py"
REPOSITORY = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")

EXIT_OK = 0
EXIT_CONTRACT = 1
EXIT_UNVERIFIED = 3


class Unverified(ValueError):
    """No verdict could be rendered. Carries its own reported prefix."""


def _load_ready():
    """`scripts/agent-ready.py` as a module, for ONE reading of `origin`.

    The identity of this repository is already derived there and is already
    tested there. Copying the expression would give the two commands two answers
    to the same question, which is the defect this whole tool exists to avoid at
    a different scale.
    """

    spec = importlib.util.spec_from_file_location(
        "agent_ready_for_pr_body", ROOT / "scripts/agent-ready.py"
    )
    if spec is None or spec.loader is None:
        raise Unverified("UNVERIFIED: cannot load scripts/agent-ready.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repository() -> str:
    """``owner/name`` for ``origin``, so the answer never depends on the cwd."""

    ready = _load_ready()
    result = subprocess.run(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise Unverified(
            f"REMOTE_UNVERIFIED: {result.stderr.strip() or 'no origin remote'}"
        )
    try:
        return ready.repository_identity(result.stdout.strip())
    except ready.RemoteUnverified as exc:
        raise Unverified(f"REMOTE_UNVERIFIED: {exc}") from exc


def fetch_body(repo: str, number: int) -> str:
    """The live body, or ``Unverified``. Never an empty string standing in.

    ``--json body`` rather than ``--jq .body``: the JSON is parsed here, so an
    absent or null field is distinguishable from a body that is genuinely the
    empty string, and the answer does not depend on how one build of ``gh``
    renders a null.
    """

    try:
        result = subprocess.run(
            ["gh", "pr", "view", str(number), "--repo", repo, "--json", "body"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise Unverified(f"REMOTE_UNVERIFIED: cannot run gh: {exc}") from exc
    if result.returncode:
        detail = result.stderr.strip() or f"gh exited {result.returncode}"
        raise Unverified(f"REMOTE_UNVERIFIED: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Unverified(f"REMOTE_UNVERIFIED: gh returned malformed JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("body"), str):
        raise Unverified("REMOTE_UNVERIFIED: gh returned no body field for this pull request")
    return payload["body"]


def read_body_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise Unverified(f"UNVERIFIED: cannot read {path}: {exc}") from exc


def validate(body: str) -> int:
    """Delegate to the checker. Its verdict is the verdict, and its report the report."""

    if not CHECKER.is_file():
        raise Unverified(f"UNVERIFIED: {CHECKER} is missing")
    result = subprocess.run(
        [sys.executable, str(CHECKER), "--message-file", "-", "--filled"],
        input=body, text=True, cwd=ROOT,
    )
    if result.returncode not in {0, 1}:
        raise Unverified(
            "UNVERIFIED: the trailer checker rendered no verdict "
            f"(exit {result.returncode})"
        )
    return EXIT_CONTRACT if result.returncode else EXIT_OK


def pull_request_number(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("a pull request number is a positive integer")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=pull_request_number, help="pull request number")
    parser.add_argument(
        "--body-file", type=Path, help="validate a message already on disk, offline"
    )
    parser.add_argument("--repo", help="owner/name; derived from origin when omitted")
    args = parser.parse_args()

    if (args.pr is None) == (args.body_file is None):
        parser.error("pass exactly one of --pr or --body-file")
    if args.repo is not None and REPOSITORY.fullmatch(args.repo) is None:
        parser.error("--repo must be owner/name")
    if args.repo is not None and args.pr is None:
        parser.error("--repo is meaningless without --pr")

    try:
        if args.body_file is not None:
            body = read_body_file(args.body_file)
        else:
            body = fetch_body(args.repo or repository(), args.pr)
        return validate(body)
    except Unverified as exc:
        print(exc, file=sys.stderr)
        print(
            "  Unknown is not absence and not success. Nothing was read, so "
            "nothing is known about the message this pull request would land.",
            file=sys.stderr,
        )
        return EXIT_UNVERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
