#!/usr/bin/env python3
"""Route every agent session into role declaration, preflight, and resume.

This command is deliberately read-only and non-interactive.  The onboarding
probe owns state discovery, agent-role.py owns declarations, and preflight owns
gates.  This module only renders truthful next actions.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_onboard():
    path = ROOT / "scripts/agent-onboard.py"
    spec = importlib.util.spec_from_file_location("agent_onboard_for_start", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_onboard_for_start"] = module
    spec.loader.exec_module(module)
    return module


onboard = _load_onboard()


WELCOME = """+------------------------------------------------------------------+
|                         Welcome to vllm.cpp                      |
+------------------------------------------------------------------+
| Helper   One focused contribution that moves the project forward.|
| Operator Maintain the project or run a long multi-agent campaign.|
| Looking  Read, review, or ask questions without claiming work.   |
+------------------------------------------------------------------+"""

WELCOME_BEGIN = "--- WELCOME: RELAY VERBATIM ---"
WELCOME_END = "--- END WELCOME ---"
ACTIONS_BEGIN = "--- AGENT NEXT ACTIONS ---"
ACTIONS_END = "--- END ACTIONS ---"


def _status_lines(state: dict) -> list[str]:
    """Render status labels only; never echo environment values or keys."""
    lines = [
        f"environment: {state.get('env') or 'unavailable'}",
        f"branch: {state.get('branch') or 'unavailable'}",
        f"worktree: {state.get('worktree') or 'unavailable'}",
    ]
    # Who else is coordinating. Until issue #285 this router turned the same
    # fact into "BLOCKED: the operator lock is held by another live worktree"
    # and told the session not to run "a known-failing claim". That claim no
    # longer fails: several coordinators may run at once, so a peer is status,
    # never a blocker. Kept SHORT -- the welcome route is width-checked.
    peers = state.get("operator_peers") or []
    if peers:
        lines.append(f"other coordinators: {len(peers)} recorded (claim is allowed)")
    return lines


def _environment_actions(state: dict) -> list[str]:
    """Route an unresolved .env to ask-and-record, never to a default.

    Status was the whole report until issue #1190: a session read
    `environment: missing` and got no next action, so what it fell back to in
    practice was a host name copied out of a document. The route never names
    WHICH key is unset, because that is the caller's own state and can carry a
    secret, and it never writes the file here, because the value has to come
    from the developer. agent-onboard.py owns the write and refuses any key
    .env.example does not declare.
    """
    status = state.get("env")
    if status == "present":
        return []
    return [
        f"NO ENVIRONMENT: .env is {status or 'unavailable'}.",
        "  Ask the developer for the ONE value the current gate needs,",
        "  then record it, and leave every other key empty:",
        "  scripts/agent-onboard.py --env-set KEY=VALUE",
        "  An unanswered key stays empty and its gate stays PENDING.",
        "  Never infer a value, and never take a host or a path from a",
        "  document instead of asking.",
    ]


def _claim_command(intent: str, row: str | None, headless: bool) -> str:
    parts = ["scripts/agent-role.py", "claim", intent]
    if intent == "helper" and row:
        parts.extend(("--row", row))
    if headless:
        parts.append("--headless")
    return " ".join(parts)


def _declared_actions(
    state: dict, intent: str | None, headless: bool
) -> list[str]:
    role = state["role"]
    lines = [f"role: {role}"]
    if state.get("row"):
        lines.append(f"row: {state['row']}")
    lines.extend(
        [
            f"mode: {state.get('mode') or 'unavailable'}",
            *_status_lines(state),
        ]
    )

    if intent is not None and intent != role:
        lines.extend(
            [
                f"MISMATCH: intent {intent} conflicts with declared role {role}.",
                "1. Stop and obtain direction before changing materialized state.",
                "2. Resolve current lock/worktree ownership, then re-declare",
                "   through scripts/agent-role.py and rerun this entrypoint.",
                "3. No worktree or PR was created by this command.",
            ]
        )
        return lines

    lines.extend(
        [
            "1. Confirm this inherited role fits the current request.",
            "2. Run scripts/agent-preflight.sh.",
            "3. Use the printed .agents/NOW.md as the live snapshot.",
            "4. Read .agents/developer-preferences.md, and create it from",
            "   .agents/developer-preferences.example.md when it is absent.",
            "5. Resume the row from .agents/coordination.md and",
            "   structured state event anchors that apply to this claim.",
        ]
    )
    if headless:
        lines.append("NOTE: declared marker mode wins; --headless changed nothing.")
    return lines


def _helper_without_row(state: dict) -> list[str]:
    lines = [
        "1. Identify or create one scoped row before claiming helper.",
    ]
    if state.get("queue_error"):
        lines.append(f"   READY queue unavailable: {state['queue_error']}")
    elif state.get("queue"):
        lines.append("   READY queue:")
        lines.extend(f"   - {item}" for item in state["queue"])
    else:
        lines.append("   READY queue is empty; create and spike a scoped row.")
    lines.extend(
        [
            "2. Rerun scripts/agent-start.py with --intent helper and --row",
            "   set to the selected row's actual ID.",
            "3. Run the printed claim command, then rerun agent-start.",
            "4. Run scripts/agent-preflight.sh after the role is declared.",
        ]
    )
    return lines


def _undeclared_actions(
    state: dict, intent: str | None, row: str | None, headless: bool
) -> list[str]:
    lines = _status_lines(state)

    if intent is None:
        lines.extend(
            [
                "1. Relay only the welcome block above verbatim.",
                "2. Then ask what the contributor is here to do.",
                "3. Use the matching scripts/agent-role.py claim command.",
                "4. After claiming, rerun scripts/agent-start.py.",
                "5. Then run scripts/agent-preflight.sh.",
            ]
        )
        return lines

    if intent == "helper" and not row:
        lines.extend(_helper_without_row(state))
        return lines

    command = _claim_command(intent, row, headless)
    lines.extend(
        [
            f"1. Run: {command}",
            "2. After claiming, rerun scripts/agent-start.py.",
            "3. Then run scripts/agent-preflight.sh.",
        ]
    )
    return lines


def render_route(
    state: dict, intent: str | None, row: str | None, headless: bool
) -> str:
    """Pure rendering of a probed state and caller-supplied intent."""
    sections: list[str] = []
    if state.get("role") is None and intent is None:
        sections.append(f"{WELCOME_BEGIN}\n{WELCOME}\n{WELCOME_END}")

    if state.get("role") is not None:
        actions = _declared_actions(state, intent, headless)
    else:
        actions = _undeclared_actions(state, intent, row, headless)
    # Appended to every route, declared or not, because an unresolved .env
    # blocks the same gates whatever role the session holds.
    actions.extend(_environment_actions(state))
    sections.append(f"{ACTIONS_BEGIN}\n" + "\n".join(actions) + f"\n{ACTIONS_END}")
    return "\n\n".join(sections)


def _argument_error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the truthful next actions for this agent session."
    )
    parser.add_argument("--intent", choices=("operator", "helper", "read-only"))
    parser.add_argument("--row", metavar="ROW-ID")
    parser.add_argument("--headless", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)

    if args.row and args.intent != "helper":
        return _argument_error("--row is valid only with --intent helper")
    if args.headless and args.intent is None:
        return _argument_error("--headless requires an explicit --intent")

    try:
        state = onboard.probe()
        rendered = render_route(state, args.intent, args.row, args.headless)
    except Exception as error:
        print(f"ERROR: unable to route session: {error}", file=sys.stderr)
        return 1
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
