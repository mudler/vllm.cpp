#!/usr/bin/env python3
"""Mutation tests for explicit PR path classes and budgets."""

from __future__ import annotations

import datetime as dt
import importlib.util
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_pr_size", ROOT / "scripts/check-pr-size.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class PathClassification(unittest.TestCase):
    def test_each_mutable_surface_has_an_explicit_class(self) -> None:
        expected = {
            "src/vt/x.cpp": "product",
            "scripts/check-release-binary-contract.py": "product",
            "scripts/check-policy.py": "governance_checker",
            "tests/scripts/test_policy_contract.py": "governance_test",
            ".agents/policy.csv": "policy",
            ".agents/state.md": "append_only_record",
            "docs/STATUS.md": "public_document",
            ".github/workflows/ci.yml": "ci",
            "src/vt/vulkan/vulkan_spirv.cpp": "generated",
            "release/manifest-v1.schema.json": "configuration",
            "scripts/env-doc-allowlist.txt": "configuration",
            "tests/scripts/fixtures/release_manifest/v1/cpu-input.json": "asset",
        }
        for path, path_class in expected.items():
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), path_class)

    def test_generated_class_does_not_swallow_its_own_sources(self) -> None:
        # The generator and the GLSL it compiles are the REVIEWABLE surface and
        # must keep their own tighter budgets. If either ever classified as
        # `generated`, a shader change could arrive unreviewed behind the blob.
        self.assertEqual(
            checker.classify_path("src/vt/vulkan/shaders/vt_matmul_vec.comp"), "product"
        )
        self.assertEqual(checker.classify_path("scripts/gen-vulkan-spirv.py"), "product")

    def test_generated_files_are_actually_generated_and_gate_verified(self) -> None:
        # The class is only sound while every member is machine-emitted and
        # reproduced by a gate. Assert the self-declaration at the head of each
        # file so a hand-written file cannot be parked here to dodge review.
        root = Path(checker.ROOT)
        self.assertTrue(checker.GENERATED_FILES, "the class must not be empty")
        for rel in checker.GENERATED_FILES:
            with self.subTest(path=rel):
                head = (root / rel).read_text(errors="replace")[:400]
                self.assertIn("GENERATED FILE - DO NOT EDIT BY HAND", head)

    def test_unknown_and_noncanonical_paths_fail_closed(self) -> None:
        for path in (
            "unknown.bin",
            "/scripts/check-policy.py",
            "scripts/../src/x.cpp",
            "scripts//check-policy.py",
            r"scripts\check-policy.py",
            "./scripts/check-policy.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_every_tracked_and_current_change_path_is_classified(self) -> None:
        paths = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        )
        paths.update(
            subprocess.check_output(
                ["git", "diff", "--name-only", "origin/main"], cwd=ROOT, text=True
            ).splitlines()
        )
        failures = []
        for path in sorted(paths):
            try:
                checker.classify_path(path)
            except ValueError:
                failures.append(path)
        self.assertEqual(failures, [])

    def test_similar_names_do_not_enter_governance_classes(self) -> None:
        self.assertNotEqual(
            checker.classify_path("scripts/checkpoint-runner.py"),
            "governance_checker",
        )
        self.assertNotEqual(
            checker.recognized_evidence("scripts/check-policy.py"),
            "tests/scripts/test_policy_contract_extra.py",
        )
        self.assertEqual(
            checker.recognized_evidence("scripts/check-agent-record.py"),
            "tests/scripts/test_agent_record.py",
        )


class BudgetEnforcement(unittest.TestCase):
    def change(self, path: str, lines: int) -> checker.ChangedPath:
        return checker.ChangedPath(path, lines, 0)

    def test_every_class_has_a_finite_positive_budget(self) -> None:
        self.assertEqual(set(checker.PATH_CLASS_BUDGETS), set(checker.PATH_CLASSES))
        self.assertTrue(all(0 < value < 10000 for value in checker.PATH_CLASS_BUDGETS.values()))

    def test_budget_boundary_passes_and_one_over_fails_on_every_branch(self) -> None:
        path = "scripts/check-pr-size.py"
        limit = checker.PATH_CLASS_BUDGETS["governance_checker"]
        evidence = self.change("tests/scripts/test_check_pr_size.py", 1)
        self.assertEqual(
            checker.change_errors([self.change(path, limit), evidence]), []
        )
        errors = checker.change_errors([self.change(path, limit + 1), evidence])
        self.assertTrue(any("governance_checker" in error for error in errors), errors)

    def test_oversized_policy_and_governance_test_changes_fail(self) -> None:
        for path, path_class in (
            (".agents/policy.csv", "policy"),
            ("tests/scripts/test_policy_contract.py", "governance_test"),
        ):
            with self.subTest(path=path):
                errors = checker.change_errors(
                    [self.change(path, checker.PATH_CLASS_BUDGETS[path_class] + 1)]
                )
                self.assertTrue(errors)

    def test_only_one_exact_pr_waiver_covers_an_over_budget_class(self) -> None:
        change = self.change(
            "AGENTS.md", checker.PATH_CLASS_BUDGETS["procedure"] + 1
        )
        waiver = checker.Waiver(
            waiver_id="WAIVER-PR-SIZE-001",
            rule_id="POL-PR-SIZE",
            scope="pr:128",
            owner="maintainer",
            reason="bounded migration",
            evidence="PR-128",
            expires=dt.date(2026, 8, 15),
        )
        self.assertEqual(
            checker.change_errors(
                [change], waivers=(waiver,), waiver_scope="pr:128"
            ),
            [],
        )
        for rule_id, scope in (
            ("POL-PATH-CLASSIFICATION", "pr:128"),
            ("POL-PR-SIZE", "pr:129"),
            ("POL-PR-SIZE", ""),
        ):
            with self.subTest(rule_id=rule_id, scope=scope):
                wrong = checker.Waiver(
                    **{**waiver.__dict__, "rule_id": rule_id, "scope": scope}
                )
                self.assertTrue(
                    checker.change_errors(
                        [change], waivers=(wrong,), waiver_scope="pr:128"
                    )
                )

        duplicate = checker.Waiver(
            **{**waiver.__dict__, "waiver_id": "WAIVER-PR-SIZE-002"}
        )
        with self.assertRaisesRegex(ValueError, "duplicate applicable waivers"):
            checker.change_errors(
                [change],
                waivers=(waiver, duplicate),
                waiver_scope="pr:128",
            )

    def test_binary_changes_fail_closed_instead_of_becoming_free(self) -> None:
        errors = checker.change_errors([checker.ChangedPath("docs/image.png", None, None)])
        self.assertTrue(any("binary" in error for error in errors), errors)

    def test_checker_change_requires_its_recognized_mutation_test(self) -> None:
        changed = [
            self.change("scripts/check-pr-size.py", 5),
            self.change("tests/scripts/test_unrelated.py", 50),
        ]
        errors = checker.change_errors(changed, evidence_results={})
        self.assertTrue(any("mutation evidence" in error for error in errors), errors)
        changed.append(self.change("tests/scripts/test_check_pr_size.py", 5))
        proof = checker.EvidenceResult(
            checker="scripts/check-pr-size.py",
            test_module="tests.scripts.test_check_pr_size",
            rule_ids=("POL-PATH-CLASSIFICATION", "POL-PR-SIZE"),
            head_tests=1,
            head_passed=True,
            base_tests=1,
            base_failed=True,
        )
        self.assertEqual(
            checker.change_errors(
                changed, evidence_results={"scripts/check-pr-size.py": proof}
            ),
            [],
        )

    def test_every_created_checker_has_closed_bootstrap_evidence(self) -> None:
        expected = {
            "scripts/check-commit-trailers.py",
            "scripts/check-policy.py",
            "scripts/check-pr-size.py",
            "scripts/check-prompt-contract.py",
        }
        self.assertEqual(set(checker.CREATION_MUTATIONS), expected)
        for path, mutation in checker.CREATION_MUTATIONS.items():
            with self.subTest(path=path):
                compile(mutation, path, "exec")
                self.assertTrue(checker.recognized_evidence(path).endswith(".py"))

    def test_unchanged_or_mode_only_test_is_not_semantic_mutation_evidence(self) -> None:
        errors = checker.change_errors(
            [
                self.change("scripts/check-pr-size.py", 1),
                self.change("tests/scripts/test_check_pr_size.py", 0),
            ]
        )
        self.assertTrue(any("mutation evidence" in error for error in errors), errors)

    def test_fake_or_comment_only_test_edit_cannot_prove_checker_semantics(self) -> None:
        changes = [
            self.change("scripts/check-pr-size.py", 1),
            self.change("tests/scripts/test_check_pr_size.py", 1),
        ]
        fake = checker.EvidenceResult(
            checker="scripts/check-pr-size.py",
            test_module="tests.scripts.test_check_pr_size",
            rule_ids=("POL-PATH-CLASSIFICATION",),
            head_tests=1,
            head_passed=True,
            base_tests=1,
            base_failed=False,
        )
        errors = checker.change_errors(
            changes, evidence_results={"scripts/check-pr-size.py": fake}
        )
        self.assertTrue(any("BASE checker stayed green" in error for error in errors), errors)

    def test_missing_policy_binding_or_unexecuted_test_fails_closed(self) -> None:
        changes = [
            self.change("scripts/check-pr-size.py", 1),
            self.change("tests/scripts/test_check_pr_size.py", 1),
        ]
        for rules, count in (((), 1), (("POL-PR-SIZE",), 0)):
            with self.subTest(rules=rules, count=count):
                proof = checker.EvidenceResult(
                    checker="scripts/check-pr-size.py",
                    test_module="tests.scripts.test_check_pr_size",
                    rule_ids=rules,
                    head_tests=count,
                    head_passed=True,
                    base_tests=1,
                    base_failed=True,
                )
                self.assertTrue(
                    checker.change_errors(
                        changes,
                        evidence_results={"scripts/check-pr-size.py": proof},
                    )
                )

    def test_arbitrary_test_filename_cannot_claim_mutation_evidence(self) -> None:
        errors = checker.change_errors(
            [
                self.change("scripts/check-policy.py", 1),
                self.change("tests/scripts/test_check_policy.py", 1),
            ]
        )
        self.assertTrue(any("test_policy_contract.py" in error for error in errors), errors)

    def test_numstat_parser_rejects_malformed_negative_and_duplicate_paths(self) -> None:
        for text in (
            "1\t2\n",
            "-1\t2\tsrc/x.cpp\n",
            "1\t2\tsrc/x.cpp\n3\t4\tsrc/x.cpp\n",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    checker.parse_numstat(text)

    def test_real_git_rename_is_delete_plus_add_with_no_rename_syntax(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "old.cpp").write_text("line\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(repo), "mv", "src/old.cpp", "src/new.cpp"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "rename"], check=True)
            head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            changes = checker.changed_paths(base, head, repo=repo)
            self.assertEqual(
                [(change.path, change.added, change.removed) for change in changes],
                [("src/new.cpp", 1, 0), ("src/old.cpp", 0, 1)],
            )
            self.assertFalse(any("{" in change.path or "=>" in change.path for change in changes))

    def test_missing_and_nonancestor_objects_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "base.cpp").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--orphan", "side"], check=True)
            subprocess.run(["git", "-C", str(repo), "rm", "-qrf", "."], check=True)
            (repo / "src").mkdir()
            (repo / "src" / "side.cpp").write_text("side\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "side"], check=True)
            side = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            with self.assertRaises(ValueError):
                checker.changed_paths("missing", side, repo=repo)
            with self.assertRaisesRegex(ValueError, "ancestor"):
                checker.changed_paths(base, side, repo=repo)

    def test_production_pr_classifier_covers_every_governed_path_class(self) -> None:
        governed = (
            "src/x.cpp",
            ".agents/policy.csv",
            "scripts/check-policy.py",
            "tests/scripts/test_policy_contract.py",
            ".agents/workflow.md",
            ".agents/state.md",
            ".agents/NOW.md",
            "docs/STATUS.md",
            ".github/workflows/ci.yml",
            "scripts/agent-role.py",
            ".env.example",
            "release/manifest-v1.schema.json",
        )
        for path in governed:
            with self.subTest(path=path):
                self.assertTrue(checker.requires_reviewed_pr(path))

    def test_production_role_check_requires_pr_for_records_docs_ci_and_support(self) -> None:
        role = checker.load_role_discipline()
        for path in (
            ".agents/state.md",
            "docs/STATUS.md",
            ".github/workflows/ci.yml",
            "scripts/agent-role.py",
            ".env.example",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    role.policy_commit_violations(
                        "abc123", ["parent"], "direct", "", [path]
                    )
                )

    def test_pending_pr_range_requires_the_exact_event_head(self) -> None:
        role = checker.load_role_discipline()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        base = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True
        ).strip()
        self.assertIn(head, role.pending_pr_commits(base, head, head))
        for pending in ("", head[:-1], head.upper(), "0" * 40):
            with self.subTest(pending=pending):
                with self.assertRaises(ValueError):
                    role.pending_pr_commits(base, head, pending)


if __name__ == "__main__":
    unittest.main()
