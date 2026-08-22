#!/usr/bin/env python3
"""Mutation suite for the Windows release's pre-publication truth state."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-windows-release-state.py"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
CI = ROOT / ".github/workflows/ci.yml"
AUDIT_WORKFLOW = ROOT / ".github/workflows/release-postpublish-audit.yml"


def load():
    spec = importlib.util.spec_from_file_location("check_windows_release_state", CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CHECKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WindowsReleaseStateContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load()

    def test_repository_truth_contract_passes(self) -> None:
        self.assertEqual(self.checker.validate_root(ROOT), [])

    def test_each_surface_rejects_coordinated_false_publication(self) -> None:
        texts = self.checker.read_surfaces(ROOT)
        for relative in self.checker.REQUIRED_SURFACES:
            with self.subTest(relative=relative):
                mutant = dict(texts)
                mutant[relative] = mutant[relative].replace(
                    self.checker.PENDING_ANCHOR,
                    "<!-- ENG-RELEASE-WINDOWS: state=DONE publication=complete artifact=published -->",
                )
                self.assertTrue(self.checker.validate_texts(mutant))

    def test_missing_or_duplicated_anchor_fails_closed(self) -> None:
        texts = self.checker.read_surfaces(ROOT)
        relative = self.checker.REQUIRED_SURFACES[0]
        for replacement in ("", self.checker.PENDING_ANCHOR * 2):
            with self.subTest(replacement=replacement):
                mutant = dict(texts)
                mutant[relative] = mutant[relative].replace(
                    self.checker.PENDING_ANCHOR, replacement
                )
                self.assertTrue(self.checker.validate_texts(mutant))

    def test_truth_checker_and_mutations_are_wired_into_preflight_and_ci(self) -> None:
        preflight = PREFLIGHT.read_text(encoding="utf-8")
        ci = CI.read_text(encoding="utf-8")
        for fragment in (
            "  check-windows-release-state\n",
            "  test_check_windows_release_state\n",
            "  test_release_postpublish_audit\n",
        ):
            self.assertEqual(preflight.count(fragment), 1)
        for command in (
            "python3 scripts/check-windows-release-state.py",
            "python3 tests/scripts/test_check_windows_release_state.py",
            "python3 tests/scripts/test_release_postpublish_audit.py",
        ):
            self.assertEqual(ci.count(command), 1)

    def test_completed_release_workflow_runs_authenticated_remote_audit(self) -> None:
        workflow = AUDIT_WORKFLOW.read_text(encoding="utf-8")
        required = (
            "workflow_run:",
            "workflows: [release]",
            "types: [completed]",
            "github.event.workflow_run.conclusion == 'success'",
            "github.event.workflow_run.event == 'push'",
            "actions: read",
            "attestations: read",
            "ref: ${{ github.event.workflow_run.head_sha }}",
            "python3 scripts/release_postpublish_audit.py",
            "--repo \"${GITHUB_REPOSITORY}\"",
            "--tag v0.0.3-pre.1",
            "--sha \"${{ github.event.workflow_run.head_sha }}\"",
            "--run-id \"${{ github.event.workflow_run.id }}\"",
            "--release-version release/release-version.json",
            "--matrix release/release-matrix.json",
        )
        for fragment in required:
            self.assertEqual(workflow.count(fragment), 1, fragment)
        self.assertEqual(workflow.count("contents: read"), 2)

    def test_every_gate_wiring_binding_fails_under_deletion_or_weakening(self) -> None:
        preflight = PREFLIGHT.read_text(encoding="utf-8")
        ci = CI.read_text(encoding="utf-8")
        workflow = AUDIT_WORKFLOW.read_text(encoding="utf-8")
        mutations = []
        for fragment in (
            "  check-windows-release-state\n",
            "  test_check_windows_release_state\n",
            "  test_release_postpublish_audit\n",
        ):
            mutations.append((preflight.replace(fragment, "", 1), ci, workflow))
        for command in (
            "python3 scripts/check-windows-release-state.py",
            "python3 tests/scripts/test_check_windows_release_state.py",
            "python3 tests/scripts/test_release_postpublish_audit.py",
        ):
            mutations.append((preflight, ci.replace(command, "echo disabled", 1), workflow))
        for fragment in (
            "github.event.workflow_run.conclusion == 'success'",
            "github.event.workflow_run.event == 'push'",
            "attestations: read",
            "ref: ${{ github.event.workflow_run.head_sha }}",
            "python3 scripts/release_postpublish_audit.py",
            "--tag v0.0.3-pre.1",
            '--sha "${{ github.event.workflow_run.head_sha }}"',
            '--run-id "${{ github.event.workflow_run.id }}"',
        ):
            mutations.append((preflight, ci, workflow.replace(fragment, "disabled", 1)))
        mutations.append((preflight, ci, workflow.replace("contents: read", "contents: write", 1)))
        for index, mutant in enumerate(mutations):
            with self.subTest(index=index):
                self.assertTrue(self.checker.validate_wiring(*mutant))


if __name__ == "__main__":
    unittest.main()
