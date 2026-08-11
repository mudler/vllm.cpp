#!/usr/bin/env python3
"""Mutation tests for explicit PR path classes and the checker contracts.

The per-class line budgets were retired 2026-08-10; the tests that pinned them
are gone and `test_no_line_budget_is_enforced_for_any_class` pins their absence."""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
# root on sys.path, so provide it before executing the module.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("check_pr_size", ROOT / "scripts/check-pr-size.py")
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


class CheckerEvidenceMapping(unittest.TestCase):
    # A checker whose evidence path does not exist can never satisfy the
    # governance_checker rule: every change to it fails with "requires semantic
    # mutation evidence in <file that is not there>". That is how
    # check-device-leakage.py became unmodifiable -- its suite predates the
    # test_check_<name> convention, so the derived path missed by one word.
    KNOWN_UNTESTED = {
        # No suite at all, and not wired into ci.yml either. Named here so the
        # gap is visible in a test rather than invisible in a naming rule.
        "scripts/check-dsv4-gguf-namemap.py",
    }

    def test_every_checker_maps_to_an_evidence_file_that_exists(self) -> None:
        root = Path(checker.__file__).resolve().parents[1]
        missing = []
        for path in sorted(root.glob("scripts/check-*.py")):
            rel = f"scripts/{path.name}"
            if rel in self.KNOWN_UNTESTED:
                continue
            evidence = checker.recognized_evidence(rel)
            if not (root / evidence).is_file():
                missing.append(f"{rel} -> {evidence}")
        self.assertEqual(missing, [])


class PathClassification(unittest.TestCase):
    def test_state_migration_manifest_archives_are_evidence(self) -> None:
        for path in (
            ".agents/completed/state-migration-manifest.csv",
            ".agents/completed/state-migration-manifest-f921.csv",
            ".agents/completed/state-migration-manifest-release-v1.2.csv",
        ):
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), "evidence")

    def test_completed_csv_near_misses_fail_closed(self) -> None:
        for path in (
            ".agents/completed/unrelated.csv",
            ".agents/completed/state-migration-manifests-f921.csv",
            ".agents/completed/state-migration-manifest-.csv",
            ".agents/completed/state-migration-manifest-f921-.csv",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_each_mutable_surface_has_an_explicit_class(self) -> None:
        expected = {
            "CLAUDE.md": "procedure",
            "MANIFESTO.md": "public_document",
            "src/vt/x.cpp": "product",
            "scripts/check-release-binary-contract.py": "product",
            "scripts/check-doc-checkpoint.py": "governance_checker",
            "tests/scripts/test_policy_contract.py": "governance_test",
            ".agents/policy.csv": "policy",
            ".agents/state-index/2026-08-001.csv": "append_only_record",
            ".agents/state-events/2026-08/STATE-20260808T120000-001.md": "append_only_record",
            ".agents/completed/state-migration-manifest.csv": "evidence",
            "docs/STATUS.md": "public_document",
            "website/hugo.toml": "public_document",
            "website/layouts/_default/baseof.html": "public_document",
            "website/assets/css/site.css": "public_document",
            "website/static/fonts/sora-700.woff2": "asset",
            "website/static/logo.svg": "asset",
            ".github/workflows/ci.yml": "ci",
            "src/vt/vulkan/vulkan_spirv.cpp": "generated",
            "release/manifest-v1.schema.json": "configuration",
            "release/release-matrix.json": "configuration",
            "release/container-matrix.json": "configuration",
            "scripts/env-doc-allowlist.txt": "configuration",
            # The container lane images. `docker/Dockerfile.arm64` already
            # classified through the suffixed pattern, but the unsuffixed
            # multi-lane Dockerfile and its healthcheck did not, and
            # classify_path FAILS CLOSED -- so adding them to the tree made the
            # PR-size gate reject the change outright.
            "docker/Dockerfile": "configuration",
            "docker/Dockerfile.arm64": "configuration",
            "docker/healthcheck.sh": "configuration",
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
            "/scripts/check-doc-checkpoint.py",
            "scripts/../src/x.cpp",
            "scripts//check-doc-checkpoint.py",
            r"scripts\check-doc-checkpoint.py",
            "./scripts/check-doc-checkpoint.py",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    checker.classify_path(path)

    def test_the_model_porting_checklist_is_procedure_and_agents_is_not_blanket(
        self,
    ) -> None:
        """The new per-model checklist classifies, and `.agents/` stays explicit.

        `.agents/porting-a-model.md` (#318) is a task guide like its siblings, so
        it belongs to the procedure class. It is listed by exact path rather than
        by widening a glob over `.agents/`, because AGENTS.md forbids hiding
        mutable files behind a blanket directory exemption.

        Both halves matter. Without the first the gate fails closed on the guide
        -- which is how this was found. Without the second, adding the entry as a
        directory pattern would silently classify every future `.agents/` file,
        including ones nobody reviewed, so the check asserts an unknown sibling
        still fails to classify.
        """
        self.assertEqual(
            checker.classify_path(".agents/porting-a-model.md"), "procedure"
        )
        # Its siblings are unchanged.
        self.assertEqual(checker.classify_path(".agents/porting.md"), "procedure")

        # `.agents/` is NOT a blanket exemption. An unlisted path there does not
        # quietly inherit a class -- the classifier refuses it outright, which is
        # what makes the gate fail closed on every new file rather than only on
        # the ones someone remembered to think about.
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/not-a-real-guide-xyz.md")

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
            checker.recognized_evidence("scripts/check-doc-checkpoint.py"),
            "tests/scripts/test_doc_checkpoint_extra.py",
        )
        self.assertEqual(
            checker.recognized_evidence("scripts/check-agent-record.py"),
            "tests/scripts/test_agent_record.py",
        )


class RetiredSurfaces(unittest.TestCase):
    """Deleted paths keep a class on purpose; the table says so in one place.

    `classify_path` fails closed, and a deleted file still appears in the diff of
    the commit that removes it, so a retired surface with no class reds the very
    change that retires it. The retention was previously spread across three
    live groups and two module-level regexes with the reason written in only one
    of them, and was read as abandoned scaffolding.
    """

    LIVE_GROUPS = (
        "POLICY_FILES",
        "APPEND_ONLY_FILES",
        "PROJECT_RECORD_FILES",
        "PROCEDURE_FILES",
        "GOVERNANCE_SUPPORT_FILES",
        "PRODUCT_CHECKER_FILES",
        "PUBLIC_DOCUMENT_FILES",
        "GENERATED_FILES",
    )

    def test_retired_paths_keep_the_class_they_had_while_live(self) -> None:
        """The budget a historical diff spends must not move under this table."""
        patterns = {
            ".agents/state-index/2026-08-001.csv": "append_only_record",
            ".agents/state-events/2026-08/STATE-20260808T120000-001.md": "append_only_record",
        }
        exact = {
            # Retired by #281: the waiver registry became a commit-message
            # argument recorded in git, so these three left the tree together.
            ".agents/waivers.csv": "policy",
            "scripts/waivers.py": "governance_support",
            "tests/scripts/test_waivers.py": "checker_test",
            ".agents/policy.csv": "policy",
            ".agents/policy-cutover": "policy",
            ".agents/state.md": "project_record",
            ".agents/state.csv": "project_record",
            ".agents/ai-coding-assistants.md": "procedure",
            ".agents/benchmark-protocol.md": "procedure",
            ".agents/directives.md": "procedure",
            ".agents/discipline.md": "procedure",
            ".agents/gates.md": "procedure",
            ".agents/test-porting.md": "procedure",
        }
        for path, path_class in {**exact, **patterns}.items():
            with self.subTest(path=path):
                self.assertEqual(checker.classify_path(path), path_class)
        self.assertEqual(dict(checker.RETIRED_PATHS), exact)

    def test_every_retired_path_is_really_gone_from_the_tree(self) -> None:
        """A live path parked here would take the retired budget and no review.

        If one is ever re-added it must move back to a live group, so this fails
        rather than letting the table quietly govern a live surface.
        """
        tracked = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
        )
        self.assertEqual(sorted(set(checker.RETIRED_PATHS) & tracked), [])
        still_here = [
            path
            for path in tracked
            for pattern, _ in checker.RETIRED_PATTERNS
            if pattern.fullmatch(path)
        ]
        self.assertEqual(still_here, [])

    def test_the_live_archive_does_not_classify_as_retired(self) -> None:
        """`.agents/completed/state-events/` was MOVED, not deleted: 160 files."""
        archived = ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md"
        self.assertIsNone(checker.retired_class(archived))
        self.assertEqual(checker.classify_path(archived), "procedure")
        self.assertEqual(
            checker.classify_path(".agents/completed/state-migration-manifest.csv"),
            "evidence",
        )

    def test_retired_and_live_groups_are_disjoint(self) -> None:
        """One home per path: the split that made this look like dead scaffolding."""
        retired = set(checker.RETIRED_PATHS)
        for name in self.LIVE_GROUPS:
            with self.subTest(group=name):
                self.assertEqual(sorted(retired & set(getattr(checker, name))), [])

    def test_a_surface_that_never_EXISTED_fails_closed(self) -> None:
        """`.agents/governance-tasks.csv` was never added and never deleted.

        It sat in POLICY_FILES as a speculative entry, so a file by that name
        could have arrived and spent the policy budget without anyone choosing
        its class. Retirement is for paths git actually removed; this one is
        simply unknown, and unknown fails closed.
        """
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/governance-tasks.csv")
        self.assertNotIn(".agents/governance-tasks.csv", checker.RETIRED_PATHS)


class BudgetEnforcement(unittest.TestCase):
    def change(self, path: str, lines: int) -> checker.ChangedPath:
        return checker.ChangedPath(path, lines, 0)

    def test_no_line_budget_is_enforced_for_any_class(self) -> None:
        """Budgets were retired 2026-08-10; size is a review judgement now.

        RED against the previous checker on both halves: it exported a
        PATH_CLASS_BUDGETS table, and a 100,000-line product change tripped its
        900-line limit. Either surviving would mean the retirement did not land.
        """
        self.assertFalse(hasattr(checker, "PATH_CLASS_BUDGETS"))
        huge = self.change("src/vllm/model_executor/models/qwen3_5.cpp", 100_000)
        self.assertEqual(checker.change_errors([huge]), [])

    def test_retiring_the_budget_did_not_retire_the_other_contracts(self) -> None:
        """The three rules that share this checker must still bite.

        Dropping a size gate is not licence to drop classification, the binary
        guard, or checker-evidence with it, which is exactly the kind of thing
        that goes unnoticed when a constant is deleted.
        """
        unknown = checker.ChangedPath("no/such/surface.txt", 1, 0)
        self.assertTrue(checker.change_errors([unknown]))
        # Asserted on the ERROR, not its wording: this is a regression guard
        # that must hold on both sides of the retirement, so it must not be
        # coupled to a message string that the retirement itself reworded.
        binary = checker.ChangedPath("assets/logo.png", None, None)
        self.assertTrue(checker.change_errors([binary]))
        lone_checker = self.change("scripts/check-pr-size.py", 10)
        self.assertTrue(
            any("mutation evidence" in e for e in checker.change_errors([lone_checker]))
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
            "scripts/check-arm-isa-build.py",
            "scripts/check-commit-trailers.py",
            "scripts/check-cpu-isa-build.py",
            "scripts/check-cuda-fat-gencode.py",
            "scripts/check-release-workflow.py",
            "scripts/validate-release-archive.py",
            "scripts/check-pr-size.py",
            "scripts/check-prompt-contract.py",
            "scripts/check-triton-aot-multiarch.py",
            # 2026-08-10: the docs-site content guard (#224). A checker created
            # in the same PR has no BASE version to mutate, so it registers the
            # disabled form its own tests must reject.
            "scripts/check-site.py",
            # 2026-08-10: the container-image gates (#170). Both suites load the
            # checker as a module and call into it, so the disabled stub fails
            # every case rather than quietly passing a reduced one.
            "scripts/check-container-matrix.py",
            "scripts/check-container-workflow.py",
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
            head_tests=1,
            head_passed=True,
            base_tests=1,
            base_failed=False,
        )
        errors = checker.change_errors(
            changes, evidence_results={"scripts/check-pr-size.py": fake}
        )
        self.assertTrue(any("BASE checker stayed green" in error for error in errors), errors)

    def test_unexecuted_test_fails_closed(self) -> None:
        changes = [
            self.change("scripts/check-pr-size.py", 1),
            self.change("tests/scripts/test_check_pr_size.py", 1),
        ]
        for count in (0,):
            with self.subTest(count=count):
                proof = checker.EvidenceResult(
                    checker="scripts/check-pr-size.py",
                    test_module="tests.scripts.test_check_pr_size",
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
                self.change("scripts/check-doc-checkpoint.py", 1),
                self.change("tests/scripts/test_check_doc_checkpoint.py", 1),
            ]
        )
        self.assertTrue(any("test_doc_checkpoint.py" in error for error in errors), errors)

    def test_agent_record_checker_uses_its_existing_mutation_suite(self) -> None:
        self.assertEqual(
            checker.recognized_evidence("scripts/check-agent-record.py"),
            "tests/scripts/test_agent_record.py",
        )



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
            ".agents/state-index/2026-08-001.csv",
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

    def test_production_role_check_requires_pr_for_product_paths(self) -> None:
        """Feature code still needs a reviewed row/* PR."""
        role = checker.load_role_discipline()
        for path in (
            "src/vt/cuda/cuda_backend.cu",
            "include/vllm.h",
            "examples/cli/main.cpp",
            "tests/vt/test_backend.cpp",
            "CMakeLists.txt",
            ".env.example",
        ):
            with self.subTest(path=path):
                self.assertTrue(
                    role.policy_commit_violations(
                        "abc123", ["parent"], "direct", "", [path]
                    )
                )

    def test_integration_trees_may_reach_main_directly(self) -> None:
        """The documented operator escape hatch, now actually implemented.

        check-role-discipline.py's docstring has always said scripts/, .agents/,
        docs/ and .github/ may be pushed straight to main so a gate or a record
        can be repaired without a round trip -- but only an explicit FILE list
        implemented it, so policy_commit_violations governed every path. A spec
        commit under docs/ could not reach main at all. This pins the documented
        behaviour so the code and its docstring cannot diverge again.
        """
        role = checker.load_role_discipline()
        for path in (
            "docs/STATUS.md",
            "docs/superpowers/specs/2026-08-09-policy-simplification-design.md",
            ".github/workflows/ci.yml",
            "scripts/agent-role.py",
            ".agents/NOW.md",
            "AGENTS.md",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    role.policy_commit_violations(
                        "abc123", ["parent"], "direct", "", [path]
                    ),
                    [],
                )

    def test_role_checker_classifies_archived_state_as_integration(self) -> None:
        role = checker.load_role_discipline()
        for path in (
            ".agents/completed/state-events/2026-08/STATE-20260808T120000-001.md",
            ".agents/completed/state-migration-manifest.csv",
        ):
            with self.subTest(path=path):
                self.assertTrue(role.is_integration_path(path))

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

    def test_a_merge_landed_pr_carries_the_commits_it_brings_in(self) -> None:
        """Arrival is judged ONCE, on the commit that lands the change.

        A PR landed with a real merge commit pushes the merge AND its branch
        commits in one range. The merge names the PR; the branch commits under it
        never had to, so judging each on its own message called every merge-
        landed PR a direct push -- main went red for `6603356a` (#178),
        `e73cbbae` (#204) and `1a02ab4f` (#196), in a gate about arriving through
        exactly the PR that had just been merged.

        The exhaustive cases live in tests/scripts/test_agent_role.py; this is the
        evidence CHECKER_EVIDENCE_OVERRIDES names for the role-discipline checker,
        so it pins the rule and the hole it must not open: only the SIDE parents
        count, and a merge naming no row and no PR exempts nothing.
        """
        role = checker.load_role_discipline()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args: str) -> str:
                return subprocess.check_output(
                    ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
                ).strip()

            def commit(message: str, path: str) -> str:
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"{message}\n")
                git("add", path)
                git("commit", "-q", "-m", message)
                return git("rev-parse", "HEAD")

            git("init", "-q", "-b", "main")
            git("config", "user.email", "t@example.com")
            git("config", "user.name", "T")
            commit("docs: seed", "docs/STATUS.md")
            pushed = commit("perf: hand-edit a kernel", "src/vt/cuda/x.cu")
            git("checkout", "-q", "-b", "row/ENG-FOO")
            reviewed = commit("perf: faster kernel", "src/vllm/a.cpp")
            git("checkout", "-q", "main")
            git("merge", "-q", "--no-ff", "-m",
                "Merge pull request #12 from mudler/row/ENG-FOO", "row/ENG-FOO")
            merge = git("rev-parse", "HEAD")

            with mock.patch.object(role, "ROOT", repo):
                content = role.merged_pr_content([pushed, merge])
                self.assertIn(reviewed, content)
                # Merging a PR on top must not launder a direct push below it.
                self.assertNotIn(pushed, content)

                git("checkout", "-q", "-b", "wip", merge)
                commit("perf: hand-edit again", "src/vllm/b.cpp")
                git("checkout", "-q", "main")
                git("merge", "-q", "--no-ff", "-m", "Merge branch 'wip'", "wip")
                self.assertEqual(
                    role.merged_pr_content([git("rev-parse", "HEAD")]), frozenset()
                )




class PerClaimPathClass(unittest.TestCase):
    """`.agents/claims/CLAIM-*.md` is a classified path (#364).

    One file per active claim, added because the claims TABLE in
    coordination.md is insert-at-one-anchor and was the largest single conflict
    source measured -- 8 of 16 conflicting open PRs. classify_path FAILS CLOSED
    on an unknown path, so without this the directory cannot be added at all.
    """

    def test_a_claim_file_is_classified(self) -> None:
        self.assertEqual(
            checker.classify_path(".agents/claims/CLAIM-EXAMPLE.md"),
            checker.classify_path(".agents/specs/example.md"),
            "a per-claim file is the same class as the per-row spec it mirrors",
        )

    def test_the_directory_readme_is_classified(self) -> None:
        checker.classify_path(".agents/claims/README.md")

    def test_dropping_the_pattern_fails_closed(self) -> None:
        """MUTATION: without the CLAIM pattern the path is unclassified.

        Proves the added clause is load-bearing rather than shadowed by some
        broader rule that already accepted the path.
        """
        with mock.patch.object(checker, "CLAIM", re.compile(r"(?!)")):
            with self.assertRaises(ValueError):
                checker.classify_path(".agents/claims/CLAIM-EXAMPLE.md")

    def test_a_non_claim_file_in_the_directory_is_still_rejected(self) -> None:
        """The pattern must not become a blanket exemption for the directory.

        AGENTS.md forbids hiding mutable files behind a directory exemption, so
        a non-markdown path here must still fail closed.
        """
        with self.assertRaises(ValueError):
            checker.classify_path(".agents/claims/CLAIM-EXAMPLE.sh")


if __name__ == "__main__":
    unittest.main()
