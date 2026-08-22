#!/usr/bin/env python3
"""Fail closed while ENG-RELEASE-WINDOWS is ACTIVE and unpublished."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PENDING_ANCHOR = "<!-- ENG-RELEASE-WINDOWS: state=ACTIVE publication=pending artifact=unpublished -->"
REQUIRED_SURFACES = (
    ".agents/engine-matrix.md",
    ".agents/roadmap_v1.md",
    ".agents/specs/windows-binary-release.md",
)


def read_surfaces(root: Path) -> dict[str, str]:
    return {relative: (root / relative).read_text(encoding="utf-8") for relative in REQUIRED_SURFACES}


def validate_texts(texts: dict[str, str]) -> list[str]:
    errors: list[str] = []
    if set(texts) != set(REQUIRED_SURFACES):
        errors.append("Windows release truth surfaces are missing or unexpected")
        return errors
    for relative in REQUIRED_SURFACES:
        if texts[relative].count(PENDING_ANCHOR) != 1:
            errors.append(f"{relative} must carry exactly one ACTIVE/pending/unpublished anchor")
    spec = texts[".agents/specs/windows-binary-release.md"]
    required_spec = ("Status: `ACTIVE`", "## Outcome", "## Now", "remain pending", "exact 32-asset API audit")
    if any(fragment not in spec for fragment in required_spec):
        errors.append("Windows spec Status/Outcome/Now must retain explicit pending publication gates")
    engine = texts[".agents/engine-matrix.md"]
    if "| `ENG-RELEASE-WINDOWS` |" not in engine or "no Windows ZIP exists yet" not in engine or "| `ACTIVE` |" not in engine:
        errors.append("engine row must remain ACTIVE with Windows ZIP unpublished")
    roadmap = texts[".agents/roadmap_v1.md"]
    if "v0.0.3-pre.1` publication and 32-asset audit remain pending" not in roadmap:
        errors.append("roadmap must retain the pending Windows publication/audit gate")
    return errors


def validate_root(root: Path) -> list[str]:
    errors = validate_texts(read_surfaces(root))
    preflight = (root / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    audit_workflow = (root / ".github/workflows/release-postpublish-audit.yml").read_text(encoding="utf-8")
    errors.extend(validate_wiring(preflight, ci, audit_workflow))
    return errors


def validate_wiring(preflight: str, ci: str, audit_workflow: str) -> list[str]:
    errors: list[str] = []
    for fragment in (
        "  check-windows-release-state\n",
        "  test_check_windows_release_state\n",
        "  test_release_postpublish_audit\n",
    ):
        if preflight.count(fragment) != 1:
            errors.append(f"preflight must execute exactly one {fragment.strip()}")
    for command in (
        "python3 scripts/check-windows-release-state.py",
        "python3 tests/scripts/test_check_windows_release_state.py",
        "python3 tests/scripts/test_release_postpublish_audit.py",
    ):
        if ci.count(command) != 1:
            errors.append(f"CI must execute exactly one {command}")
    required_audit = (
        "workflow_run:", "workflows: [release]", "types: [completed]",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'", "actions: read",
        "attestations: read",
        "ref: ${{ github.event.workflow_run.head_sha }}",
        "python3 scripts/release_postpublish_audit.py",
        '--repo "${GITHUB_REPOSITORY}"', "--tag v0.0.3-pre.1",
        '--sha "${{ github.event.workflow_run.head_sha }}"',
        '--run-id "${{ github.event.workflow_run.id }}"',
        "--release-version release/release-version.json",
        "--matrix release/release-matrix.json",
    )
    for fragment in required_audit:
        if audit_workflow.count(fragment) != 1:
            errors.append(f"post-publication workflow must bind exactly one {fragment}")
    if audit_workflow.count("contents: read") != 2:
        errors.append("post-publication workflow must be globally and job-locally contents-read-only")
    return errors


def main() -> int:
    errors = validate_root(ROOT)
    if errors:
        for error in errors:
            print(f"windows release state error: {error}", file=sys.stderr)
        return 1
    print("Windows release remains truthfully ACTIVE, pending, and unpublished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
