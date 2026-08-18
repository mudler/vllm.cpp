#!/usr/bin/env python3
"""Prove that the current helper head has one live, green pull request."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GH_FIELDS = (
    "number,state,isDraft,headRefName,headRefOid,headRepository,"
    "baseRefName,baseRefOid,statusCheckRollup,reviewDecision,reviews"
)
EXPECTED_KEYS = frozenset({"repository", "base", "base_oid", "head_branch", "head_sha"})
PAYLOAD_KEYS = frozenset({"expected", "prs"})


class RemoteUnverified(ValueError):
    """Remote evidence is absent, malformed, or unavailable."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RemoteUnverified(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def repository_identity(url: str) -> str:
    match = re.search(r"(?:github\.com[:/])([^/\s]+/[^/\s]+?)(?:\.git)?$", url)
    if match is None:
        raise RemoteUnverified("cannot derive owner/repository from origin")
    return match.group(1)


def local_expected(base: str = "main") -> dict[str, str]:
    return {
        "repository": repository_identity(_git("remote", "get-url", "origin")),
        "base": base,
        "base_oid": _git("rev-parse", f"origin/{base}^{{commit}}"),
        "head_branch": _git("symbolic-ref", "--short", "HEAD"),
        "head_sha": _git("rev-parse", "HEAD^{commit}"),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RemoteUnverified(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteUnverified(f"cannot read PR fixture: {exc}") from exc
    if not isinstance(value, dict) or set(value) != PAYLOAD_KEYS:
        raise RemoteUnverified("PR payload must contain exactly expected and prs")
    if not isinstance(value["expected"], dict) or set(value["expected"]) != EXPECTED_KEYS:
        raise RemoteUnverified("PR payload expected identity has the wrong shape")
    if not isinstance(value["prs"], list):
        raise RemoteUnverified("PR payload prs must be a list")
    return value


def query_remote(expected: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        ["gh", "pr", "list", "--repo", expected["repository"], "--state", "open",
         "--limit", "100", "--json", GH_FIELDS],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RemoteUnverified(result.stderr.strip() or "GitHub query failed")
    try:
        prs = json.loads(result.stdout, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RemoteUnverified) as exc:
        raise RemoteUnverified(f"GitHub returned malformed JSON: {exc}") from exc
    if not isinstance(prs, list):
        raise RemoteUnverified("GitHub PR response is not a list")
    return {"expected": expected, "prs": prs}


def ready_errors(payload: dict[str, Any], expected: dict[str, str]) -> list[str]:
    errors: list[str] = []
    if payload.get("expected") != expected:
        errors.append("fixture identity does not match the current repository/head/base")
    prs = payload.get("prs")
    if not isinstance(prs, list):
        return errors + ["PR payload prs must be a list"]
    candidates = [pr for pr in prs if isinstance(pr, dict) and pr.get("headRefName") == expected["head_branch"]]
    if len(candidates) != 1:
        return errors + [f"expected exactly one live PR for {expected['head_branch']}; found {len(candidates)}"]
    pr = candidates[0]
    repository = pr.get("headRepository")
    repository_name = repository.get("nameWithOwner") if isinstance(repository, dict) else None
    checks = (
        (repository_name == expected["repository"], "PR repository does not match"),
        (pr.get("state") == "OPEN", "PR is not OPEN"),
        (isinstance(pr.get("isDraft"), bool), "PR draft state is malformed"),
        (pr.get("headRefOid") == expected["head_sha"], "PR head SHA is stale or wrong"),
        (pr.get("baseRefName") == expected["base"], "PR base branch is wrong"),
        (pr.get("baseRefOid") == expected["base_oid"], "PR base SHA is stale or wrong"),
    )
    errors.extend(message for ok, message in checks if not ok)
    rollup = pr.get("statusCheckRollup")
    if not isinstance(rollup, list) or not rollup:
        errors.append("PR has no current CI results")
    else:
        for check in rollup:
            if not isinstance(check, dict):
                errors.append("PR CI result is malformed")
                continue
            if check.get("status") != "COMPLETED" or check.get("conclusion") != "SUCCESS":
                errors.append(f"PR check {check.get('name', '<unknown>')} is not completed-success")
    return errors


def run_local_preflight() -> bool:
    """Run preflight, and refuse a run that SKIPPED a gate as well as one that failed.

    This reads preflight by exit status alone, which carries two of the three
    states preflight reports. Without `--fail-on-skip` a run whose base does not
    resolve, or whose branch is behind `origin/main`, exits 0 with up to five
    gates never executed, and this function returned True for it. The line below
    then printed the word "green" over a trailer check that had not run (#998).

    The flag is opt-in precisely so a human running preflight on a branch behind
    `main` still gets exit 0. This caller is not that human. It is the documented
    gate before a remote handoff, so an unknown here has to read as "not ready"
    rather than as success.
    """

    return subprocess.run(
        [str(ROOT / "scripts/agent-preflight.sh"), "--quiet", "--fail-on-skip"],
        cwd=ROOT,
    ).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr-json", type=Path)
    args = parser.parse_args()
    if not run_local_preflight():
        print(
            "READY FAILED: local preflight did not report every gate green. "
            "A gate FAILED, or a gate was SKIPPED and reported nothing about "
            "this tree. Its report above says which, and why.",
            file=sys.stderr,
        )
        return 1
    try:
        expected = local_expected()
        payload = load_payload(args.pr_json) if args.pr_json else query_remote(expected)
        errors = ready_errors(payload, expected)
    except RemoteUnverified as exc:
        print(f"REMOTE_UNVERIFIED: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"READY FAILED: {error}", file=sys.stderr)
        return 1
    print("READY: local and live PR/CI evidence are green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
