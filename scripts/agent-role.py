#!/usr/bin/env python3
"""Declare, materialize and re-derive an agent session's role. (W0)

The role CANNOT be derived at session start: the common case is one operator and
several helpers all launched from the SAME checkout, indistinguishable until a
role has already been taken. So the role is DECLARED, immediately MATERIALIZED
into a fact, and only then re-derived. See
.agents/specs/operator-helper-protocol.md.

Two identities make that work, both verified real rather than assumed:

* **session** - `VLLM_CPP_AGENT_SESSION` when set, else the parent process id,
  which is the agent CLI process and is stable across tool calls within a
  session (measured) while differing between concurrently running sessions.
* **worktree** - `git rev-parse --git-dir`, which is per-worktree
  (`.git/worktrees/<name>`), so a materialized helper is distinguishable from
  the primary checkout without any bookkeeping.

The operator lock lives in the git COMMON dir, not the working tree: it is
shared by every worktree of the repo (the correct scope for "one operator per
repo") and can never be committed by accident.

    scripts/agent-role.py show                  # resolve; exit 3 if undeclared
    scripts/agent-role.py claim operator
    scripts/agent-role.py claim helper --row ENG-FOO
    scripts/agent-role.py heartbeat
    scripts/agent-role.py release
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROLES = ("operator", "helper")

# A lock older than this with no heartbeat is stale: a crashed operator must not
# block everyone forever. Breaking one is always logged, never silent.
LOCK_TTL_SECONDS = 2 * 60 * 60

UNDECLARED_EXIT = 3


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def session_id() -> str:
    """Stable within one agent session, distinct between concurrent ones."""
    explicit = os.environ.get("VLLM_CPP_AGENT_SESSION")
    return explicit if explicit else f"ppid:{os.getppid()}"


def marker_path() -> Path:
    """Per-worktree, so a materialized helper carries its own role."""
    return Path(git("rev-parse", "--absolute-git-dir")) / "vllm-cpp-agent-role"


def lock_path() -> Path:
    """Shared by every worktree of this repo, and never inside the work tree."""
    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "vllm-cpp-operator.lock"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def lock_is_stale(record: dict) -> bool:
    beat = record.get("heartbeat", record.get("claimed_at", 0))
    return (time.time() - float(beat)) > LOCK_TTL_SECONDS


def current_branch() -> str:
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return ""


def resolve() -> dict:
    """Return the resolved role for THIS session, or {'role': None, ...}."""
    me = session_id()
    marker = read_json(marker_path())
    lock = read_json(lock_path())

    # A marker written by a DIFFERENT session sharing this checkout is not ours.
    if marker and marker.get("session") == me and marker.get("role") in ROLES:
        role = marker["role"]
        if role == "operator":
            if not lock or lock.get("session") != me:
                return {
                    "role": None,
                    "session": me,
                    "reason": "operator marker without a held lock; re-claim",
                    "branch": current_branch(),
                }
        return {
            "role": role,
            "row": marker.get("row"),
            "session": me,
            "branch": current_branch(),
            "reason": "declared",
        }

    # Not declared by us. Report what else is going on so the caller can decide.
    held_by_other = bool(lock and lock.get("session") != me and not lock_is_stale(lock))
    return {
        "role": None,
        "session": me,
        "branch": current_branch(),
        "operator_held_by_other": held_by_other,
        "reason": "undeclared",
    }


def cmd_show(args: argparse.Namespace) -> int:
    state = resolve()
    if args.json:
        print(json.dumps(state))
    elif state["role"]:
        row = f" row={state['row']}" if state.get("row") else ""
        print(f"role={state['role']}{row} session={state['session']} branch={state['branch']}")
    else:
        print(f"role=UNDECLARED session={state['session']} branch={state['branch']}")
        if state.get("operator_held_by_other"):
            print("  note: the operator lock is held by another live session")
    return 0 if state["role"] else UNDECLARED_EXIT


def cmd_claim(args: argparse.Namespace) -> int:
    me = session_id()
    role = args.role
    if role == "helper" and not args.row:
        print("ERROR: a helper claims one row: --row <ROW-ID>", file=sys.stderr)
        return 2

    if role == "operator":
        path = lock_path()
        record = {"session": me, "claimed_at": time.time(), "heartbeat": time.time(),
                  "host": os.uname().nodename, "pid": os.getpid()}
        try:
            # Create-exclusive: a second self-declared operator FAILS here rather
            # than racing on main, which is the whole point of the lock.
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle)
        except FileExistsError:
            existing = read_json(path) or {}
            if existing.get("session") == me:
                pass  # already ours, idempotent
            elif lock_is_stale(existing):
                age = int(time.time() - float(existing.get("heartbeat", 0)))
                print(
                    f"NOTE: breaking a STALE operator lock held by "
                    f"{existing.get('session')} on {existing.get('host')} "
                    f"({age}s without heartbeat, TTL {LOCK_TTL_SECONDS}s)",
                    file=sys.stderr,
                )
                path.write_text(json.dumps(record), encoding="utf-8")
            else:
                print(
                    f"ERROR: the operator role is already held by session "
                    f"{existing.get('session')} on {existing.get('host')}. "
                    "This session cannot be the operator; take the helper role "
                    "instead (scripts/agent-role.py claim helper --row <ROW-ID>).",
                    file=sys.stderr,
                )
                return 1

    marker_path().write_text(
        json.dumps({"role": role, "row": args.row, "session": me, "at": time.time()}),
        encoding="utf-8",
    )
    print(f"claimed role={role}" + (f" row={args.row}" if args.row else ""))
    return 0


def cmd_heartbeat(_: argparse.Namespace) -> int:
    state = resolve()
    if state["role"] != "operator":
        print("not the operator; nothing to heartbeat")
        return 0
    path = lock_path()
    record = read_json(path) or {}
    record["heartbeat"] = time.time()
    path.write_text(json.dumps(record), encoding="utf-8")
    print("heartbeat updated")
    return 0


def cmd_release(_: argparse.Namespace) -> int:
    me = session_id()
    lock = read_json(lock_path())
    if lock and lock.get("session") == me:
        lock_path().unlink(missing_ok=True)
        print("released the operator lock")
    marker_path().unlink(missing_ok=True)
    print("released the role marker")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="resolve this session's role")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_show)

    claim = sub.add_parser("claim", help="declare and materialize a role")
    claim.add_argument("role", choices=ROLES)
    claim.add_argument("--row", help="the row ID a helper claims")
    claim.set_defaults(func=cmd_claim)

    sub.add_parser("heartbeat", help="keep the operator lock alive").set_defaults(
        func=cmd_heartbeat
    )
    sub.add_parser("release", help="drop the role and any held lock").set_defaults(
        func=cmd_release
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
