#!/usr/bin/env python3
"""Mutation suite for scripts/check-container-workflow.py.

Each case takes the shipped workflow, breaks exactly one property the guard
claims to enforce, and asserts the guard fails for that reason. The shipped
workflow passing is a single case here, not the point of the file: a guard whose
body was deleted would still pass that one.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/check-container-workflow.py"
WORKFLOW = ROOT / ".github/workflows/containers.yml"


def load_module():
    spec = importlib.util.spec_from_file_location("check_container_workflow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_module()
SHIPPED = WORKFLOW.read_text(encoding="utf-8")


def errors_for(text: str) -> list[str]:
    return guard.validate(text)


def assert_flags(case, text: str, needle: str):
    errors = errors_for(text)
    case.assertTrue(
        any(needle in error for error in errors),
        f"expected an error containing {needle!r}, got {errors}",
    )


class ShippedWorkflowTests(unittest.TestCase):
    def test_the_shipped_workflow_passes_its_own_guard(self):
        self.assertEqual(errors_for(SHIPPED), [])

    def test_the_guard_finds_every_declared_job(self):
        names = guard.job_names(SHIPPED)
        for expected in ("plan", "verify", "publish", "manifest", "attest", "promote"):
            self.assertIn(expected, names)


class PermissionMutationTests(unittest.TestCase):
    def test_registry_write_in_the_verify_job_is_rejected(self):
        text = SHIPPED.replace(
            "  verify:\n    needs: plan\n    permissions:\n      contents: read\n",
            "  verify:\n    needs: plan\n    permissions:\n      contents: read\n      packages: write\n",
        )
        assert_flags(self, text, "only ['manifest', 'promote', 'publish']")

    def test_oidc_outside_the_attest_job_is_rejected(self):
        text = SHIPPED.replace(
            "  manifest:\n    needs: [plan, publish]",
            "  manifest:\n    needs: [plan, publish]\n    x-marker: id-token: write",
        ).replace(
            "      contents: read\n      packages: write\n    runs-on: ubuntu-latest\n    outputs:\n      digests:",
            "      contents: read\n      packages: write\n      id-token: write\n    runs-on: ubuntu-latest\n    outputs:\n      digests:",
        )
        assert_flags(self, text, "may sign")

    def test_contents_write_anywhere_is_rejected(self):
        text = SHIPPED.replace(
            "  promote:\n    needs: [plan, attest]\n    if:",
            "  promote:\n    needs: [plan, attest]\n    if:",
        ).replace(
            "      contents: read\n      packages: write\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: docker/login-action@v3",
            "      contents: write\n      packages: write\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: docker/login-action@v3",
        )
        assert_flags(self, text, "contents: write")

    def test_a_publish_job_without_registry_write_is_rejected(self):
        text = SHIPPED.replace(
            "    permissions:\n      contents: read\n      packages: write\n    strategy:",
            "    permissions:\n      contents: read\n    strategy:",
        )
        assert_flags(self, text, "must declare packages: write")


class TagGateMutationTests(unittest.TestCase):
    def test_an_ungated_publish_job_is_rejected(self):
        text = SHIPPED.replace(
            "  publish:\n    needs: [plan, verify]\n    if: needs.plan.outputs.is_release == 'true'\n",
            "  publish:\n    needs: [plan, verify]\n",
        )
        assert_flags(self, text, "nothing publishes between tags")

    def test_an_ungated_promote_job_is_rejected(self):
        text = SHIPPED.replace(
            "  promote:\n    needs: [plan, attest]\n    if: needs.plan.outputs.is_release == 'true'\n",
            "  promote:\n    needs: [plan, attest]\n",
        )
        assert_flags(self, text, "nothing publishes between tags")

    def test_dropping_the_tag_version_check_is_rejected(self):
        text = SHIPPED.replace('test "${GITHUB_REF_NAME}" = "v${version}"', "true")
        assert_flags(self, text, "untrusted input")


class PublishOrderMutationTests(unittest.TestCase):
    def test_pushing_before_validating_is_rejected(self):
        """The ordering property, not merely the presence of both steps."""
        publish = guard.job_block(SHIPPED, "publish")
        validate_step = publish[
            publish.index("      - name: Validate the image immediately") : publish.index(
                "      - name: Push by digest"
            )
        ]
        push_step = publish[publish.index("      - name: Push by digest") : publish.index(
            "      - name: Record the exact digest"
        )]
        swapped = publish.replace(validate_step + push_step, push_step + validate_step)
        text = SHIPPED.replace(publish, swapped)
        assert_flags(self, text, "pushes BEFORE validating")

    def test_removing_the_immutable_tag_guard_is_rejected(self):
        text = SHIPPED.replace("already exists; version tags are immutable", "ok to clobber")
        assert_flags(self, text, "refuse to overwrite an existing version tag")

    def test_removing_validation_from_publish_is_rejected(self):
        publish = guard.job_block(SHIPPED, "publish")
        text = SHIPPED.replace(
            publish, publish.replace("validate-container-image.py", "true #")
        )
        assert_flags(self, text, "must validate the image it pushes")

    def test_removing_validation_from_verify_is_rejected(self):
        verify = guard.job_block(SHIPPED, "verify")
        text = SHIPPED.replace(verify, verify.replace("validate-container-image.py", "true #"))
        assert_flags(self, text, "must run scripts/validate-container-image.py")


class PullRequestSafetyTests(unittest.TestCase):
    def test_a_registry_login_in_verify_is_rejected(self):
        verify = guard.job_block(SHIPPED, "verify")
        poisoned = verify.replace(
            "      - uses: docker/setup-buildx-action@v3",
            "      - uses: docker/setup-buildx-action@v3\n      - uses: docker/login-action@v3",
        )
        text = SHIPPED.replace(verify, poisoned)
        assert_flags(self, text, "must not log in to or push to a registry")

    def test_dropping_the_pull_request_trigger_is_rejected(self):
        text = SHIPPED.replace("  pull_request:\n", "")
        assert_flags(self, text, "pull_request")

    def test_dropping_the_manual_entry_point_is_rejected(self):
        """The matcher was widened from `workflow_dispatch: {}` to the key alone
        so the dispatch could take a full_matrix input. What it still has to
        enforce is that a manual entry point EXISTS -- a dry run is how a lane
        outside the pull-request matrix gets proved."""
        text = SHIPPED.replace("  workflow_dispatch:\n", "")
        assert_flags(self, text, "workflow_dispatch")

    def test_continue_on_error_is_rejected(self):
        text = SHIPPED.replace(
            "  verify:\n    needs: plan\n", "  verify:\n    continue-on-error: true\n    needs: plan\n"
        )
        assert_flags(self, text, "may not continue after an error")


class PlanAndPromoteTests(unittest.TestCase):
    def test_dropping_a_checker_from_plan_is_rejected(self):
        text = SHIPPED.replace("python3 scripts/check-container-matrix.py", "true")
        assert_flags(self, text, "check-container-matrix.py")

    def test_hand_written_moving_tags_are_rejected(self):
        promote = guard.job_block(SHIPPED, "promote")
        text = SHIPPED.replace(
            promote,
            promote.replace(
                "python3 scripts/container_tags.py",
                "echo ghcr.io/mudler/vllm.cpp:latest #",
            ),
        )
        assert_flags(self, text, "not from a hand-written list")

    def test_a_missing_job_is_reported_before_anything_else(self):
        text = SHIPPED.replace("  attest:\n", "  attest-disabled:\n")
        assert_flags(self, text, "missing the 'attest' job")


class BuildMatrixTests(unittest.TestCase):
    """The reduced PR matrix is a cost decision; it must not become a publish gap."""

    def test_publishing_from_the_reduced_matrix_is_rejected(self):
        text = SHIPPED.replace(
            "fromJSON(needs.plan.outputs.publish_matrix)",
            "fromJSON(needs.plan.outputs.verify_matrix)",
        )
        assert_flags(self, text, "must consume publish_matrix")

    def test_dropping_release_from_the_publish_matrix_is_rejected(self):
        text = SHIPPED.replace(
            'echo "publish=$(python3 scripts/container_tags.py --build-matrix --release)"',
            'echo "publish=$(python3 scripts/container_tags.py --build-matrix)"',
        )
        assert_flags(self, text, "must compute the publish matrix with --release")

    def test_verify_consumes_the_verify_matrix(self):
        text = SHIPPED.replace(
            "fromJSON(needs.plan.outputs.verify_matrix)",
            "fromJSON(needs.plan.outputs.publish_matrix)",
        )
        assert_flags(self, text, "must consume verify_matrix")


class TagResolutionTests(unittest.TestCase):
    """scripts/container_tags.py is what promote trusts; prove it, don't assume it."""

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "container_tags", ROOT / "scripts/container_tags.py"
        )
        self.tags = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tags)
        import json

        self.matrix = json.loads(
            (ROOT / "release/container-matrix.json").read_text(encoding="utf-8")
        )

    def test_immutable_tags_carry_the_version_and_the_lane(self):
        tags = self.tags.immutable_tags(self.matrix, "9.9.9")
        self.assertIn("ghcr.io/mudler/vllm.cpp:9.9.9-cpu", tags)
        self.assertIn("ghcr.io/mudler/vllm.cpp:9.9.9-cuda", tags)

    def test_bare_latest_resolves_to_the_cpu_lane(self):
        pairs = dict(
            (target, source) for source, target in self.tags.moving_pairs(self.matrix, "9.9.9")
        )
        self.assertEqual(
            pairs["ghcr.io/mudler/vllm.cpp:latest"], "ghcr.io/mudler/vllm.cpp:9.9.9-cpu"
        )

    def test_the_release_build_matrix_covers_every_lane_on_both_arches(self):
        entries = self.tags.build_matrix(self.matrix, release=True)
        self.assertEqual(len(entries), 6)
        for lane in ("cpu", "vulkan", "cuda"):
            arches = {e["platform"] for e in entries if e["lane"] == lane}
            self.assertEqual(arches, {"linux/amd64", "linux/arm64"})

    def test_the_pull_request_matrix_is_a_strict_subset(self):
        pr = self.tags.build_matrix(self.matrix, release=False)
        full = self.tags.build_matrix(self.matrix, release=True)
        self.assertTrue(pr, "a pull request must still build something")
        for entry in pr:
            self.assertIn(entry, full)
        self.assertLess(len(pr), len(full))

    def test_the_cuda_lane_is_not_built_on_every_pull_request(self):
        pr = self.tags.build_matrix(self.matrix, release=False)
        self.assertNotIn("cuda", {e["lane"] for e in pr})

    def test_every_lane_gets_its_own_moving_pointer(self):
        pairs = dict(
            (target, source) for source, target in self.tags.moving_pairs(self.matrix, "9.9.9")
        )
        for lane in ("cpu", "vulkan", "cuda"):
            self.assertEqual(
                pairs[f"ghcr.io/mudler/vllm.cpp:latest-{lane}"],
                f"ghcr.io/mudler/vllm.cpp:9.9.9-{lane}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
