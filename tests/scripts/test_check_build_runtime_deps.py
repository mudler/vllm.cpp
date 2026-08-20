#!/usr/bin/env python3
"""Mutation suite for scripts/check-build-runtime-deps.py.

Every case mutates one guarantee the checker claims and asserts it fails FOR
THAT REASON. A suite that only proved the shipped files pass would stay green
against a checker whose body had been emptied, which is the shape #1509 tracks.

The load-bearing case is `test_the_state_this_gate_exists_for`: the exact
docker/Dockerfile and .github/workflows/release.yml this repository carried at
`141402e6c`, before #1517 was fixed. If the checker stops refusing that pair, it
has stopped being the gate it claims to be.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/check-build-runtime-deps.py"
DOCKERFILE = ROOT / "docker/Dockerfile"
WORKFLOW = ROOT / ".github/workflows/release.yml"


def load_module():
    spec = importlib.util.spec_from_file_location("check_build_runtime_deps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_module()


def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


class ShippedFilesTests(unittest.TestCase):
    def test_the_shipped_dockerfile_passes(self):
        self.assertEqual(checker.check_dockerfile(dockerfile_text()), [])

    def test_the_shipped_workflow_passes(self):
        self.assertEqual(checker.check_workflow(workflow_text()), [])

    def test_the_shipped_dockerfile_actually_has_a_compiling_stage(self):
        """Guards against a vacuous pass: the gate must have something to check."""
        stages = checker.parse_stages(dockerfile_text())
        compiling = [
            stage["name"]
            for stage in stages
            if any(checker.BUILD_SCRIPTS.search(run) for run in stage["runs"])
        ]
        self.assertEqual(sorted(compiling), ["build-cpu", "build-cuda", "build-vulkan"])

    def test_the_shipped_workflow_actually_has_building_jobs(self):
        blocks = checker.job_blocks(workflow_text())
        building = sorted(
            name for name, block in blocks.items() if checker.BUILD_SCRIPTS.search(block)
        )
        self.assertEqual(
            building,
            [
                "cpu_arm64",
                "cpu_musl",
                "cpu_windows",
                "cpu_x86",
                "cuda_arm64",
                "cuda_x86",
                "metal_arm64",
                "mlx_arm64",
                "vulkan_windows",
                "vulkan_x86",
            ],
        )


class DockerfileMutationTests(unittest.TestCase):
    def test_dropping_libssl_dev_from_the_shared_toolchain_is_refused(self):
        text = dockerfile_text().replace("      libssl-dev \\\n", "", 1)
        errors = checker.check_dockerfile(text)
        self.assertTrue(
            any("'build-cpu'" in e and "libssl-dev" in e for e in errors), errors
        )
        self.assertTrue(any("'build-vulkan'" in e for e in errors), errors)

    def test_dropping_libssl_dev_from_the_cuda_toolchain_is_refused(self):
        # The CUDA base cannot inherit builder-toolchain, so its list is its own
        # and losing it is a separate defect from the case above.
        text = dockerfile_text()
        self.assertEqual(text.count("      libssl-dev \\\n"), 2)
        head, sep, tail = text.rpartition("      libssl-dev \\\n")
        errors = checker.check_dockerfile(head + tail)
        names = [e for e in errors if "'build-cuda'" in e]
        self.assertEqual(len(names), 1, errors)
        self.assertNotIn("'build-cpu'", " ".join(errors))

    def test_dropping_the_runtime_library_is_refused(self):
        text = dockerfile_text().replace("      libssl3 \\\n", "", 1)
        errors = checker.check_dockerfile(text)
        self.assertTrue(any("libssl3" in e for e in errors), errors)

    def test_a_dev_package_in_an_ancestor_stage_satisfies_a_child(self):
        text = (
            "FROM scratch AS base\n"
            "RUN apt-get install --yes libssl-dev\n"
            "FROM base AS build\n"
            "RUN scripts/build-cpu-release.sh a b c d\n"
            "FROM scratch AS rt\n"
            "RUN apt-get install --yes libssl3\n"
        )
        self.assertEqual(checker.check_dockerfile(text), [])

    def test_a_dev_package_in_a_SIBLING_stage_does_not_satisfy_a_child(self):
        """Ancestry, not presence anywhere in the file. A package installed in
        an unrelated stage is not on the compiling stage's filesystem."""
        text = (
            "FROM scratch AS other\n"
            "RUN apt-get install --yes libssl-dev\n"
            "FROM scratch AS build\n"
            "RUN scripts/build-cpu-release.sh a b c d\n"
            "FROM scratch AS rt\n"
            "RUN apt-get install --yes libssl3\n"
        )
        errors = checker.check_dockerfile(text)
        self.assertTrue(any("'build'" in e for e in errors), errors)

    def test_a_file_with_no_compiling_stage_is_refused_rather_than_passed(self):
        text = "FROM scratch AS rt\nRUN apt-get install --yes libssl3\n"
        errors = checker.check_dockerfile(text)
        self.assertTrue(any("vacuously" in e for e in errors), errors)

    def test_a_package_name_hidden_across_a_continuation_is_still_seen(self):
        text = (
            "FROM scratch AS build\n"
            "RUN apt-get install --yes \\\n      libssl-dev \\\n      cmake\n"
            "RUN scripts/build-cpu-release.sh a b c d\n"
            "FROM scratch AS rt\n"
            "RUN apt-get install --yes libssl3\n"
        )
        self.assertEqual(checker.check_dockerfile(text), [])

    def test_libssl3_does_not_count_as_the_development_package(self):
        """The #1517 shape exactly: the runtime library present, the dev files
        absent, everything else well-formed."""
        text = (
            "FROM scratch AS build\n"
            "RUN apt-get install --yes libssl3\n"
            "RUN scripts/build-cpu-release.sh a b c d\n"
        )
        errors = checker.check_dockerfile(text)
        self.assertTrue(any("'build'" in e and "libssl-dev" in e for e in errors), errors)


class WorkflowMutationTests(unittest.TestCase):
    def test_dropping_the_package_from_a_container_job_is_refused(self):
        text = workflow_text().replace(
            "binutils ca-certificates cmake file g++ git libssl-dev ninja-build python3",
            "binutils ca-certificates cmake file g++ git ninja-build python3",
        )
        errors = checker.check_workflow(text)
        self.assertTrue(any("'cuda_x86'" in e for e in errors), errors)
        self.assertTrue(any("'cuda_arm64'" in e for e in errors), errors)

    def test_dropping_the_package_from_a_runner_job_is_refused(self):
        text = workflow_text().replace(
            "          sudo apt-get install --yes libssl-dev\n"
            "          scripts/install-intel-sde.sh",
            "          scripts/install-intel-sde.sh",
        )
        errors = checker.check_workflow(text)
        self.assertTrue(any("'cpu_x86'" in e for e in errors), errors)

    def test_a_new_linux_lane_that_names_nothing_is_refused(self):
        text = workflow_text() + (
            "  rocm_x86:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: scripts/build-linux-accelerator-release.sh a rocm b\n"
        )
        errors = checker.check_workflow(text)
        self.assertTrue(any("'rocm_x86'" in e for e in errors), errors)

    def test_a_stale_exemption_for_a_deleted_job_is_refused(self):
        text = workflow_text().replace("  cpu_musl:\n", "  cpu_musl_renamed:\n", 1)
        errors = checker.check_workflow(text)
        self.assertTrue(any("'cpu_musl'" in e and "no longer" in e for e in errors), errors)
        self.assertTrue(any("'cpu_musl_renamed'" in e for e in errors), errors)

    def test_an_exempt_job_that_gains_the_package_is_refused(self):
        text = workflow_text().replace(
            "          sudo chown -R \"$(id -u):$(id -g)\" build-release-cpu-musl",
            "          sudo apt-get install --yes libssl-dev\n"
            "          sudo chown -R \"$(id -u):$(id -g)\" build-release-cpu-musl",
        )
        errors = checker.check_workflow(text)
        self.assertTrue(
            any("'cpu_musl'" in e and "EXEMPT_LANES" in e for e in errors), errors
        )

    def test_a_workflow_with_no_building_job_is_refused_rather_than_passed(self):
        errors = checker.check_workflow("name: x\njobs:\n  plan:\n    runs-on: x\n")
        self.assertTrue(any("vacuously" in e for e in errors), errors)

    def test_every_exempt_lane_records_a_reason(self):
        for name, reason in checker.EXEMPT_LANES.items():
            with self.subTest(lane=name):
                self.assertGreater(len(reason), 40, name)


class TheStateThisGateExistsForTests(unittest.TestCase):
    """The pre-#1517 tree, verbatim from git, must be refused."""

    PRE_FIX = "141402e6cc72976a46adfbfafdab9a1a2a7386e1"

    def _blob(self, path: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{self.PRE_FIX}:{path}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def test_the_state_this_gate_exists_for(self):
        dockerfile = self._blob("docker/Dockerfile")
        workflow = self._blob(".github/workflows/release.yml")
        if dockerfile is None or workflow is None:
            self.skipTest(f"commit {self.PRE_FIX} is not in this checkout")
        docker_errors = checker.check_dockerfile(dockerfile, "PRE-FIX Dockerfile")
        workflow_errors = checker.check_workflow(workflow, "PRE-FIX release.yml")
        self.assertEqual(
            sorted(
                name
                for name in ("build-cpu", "build-vulkan", "build-cuda")
                if any(f"'{name}'" in e for e in docker_errors)
            ),
            ["build-cpu", "build-cuda", "build-vulkan"],
            docker_errors,
        )
        self.assertEqual(
            sorted(
                name
                for name in ("cpu_x86", "cpu_arm64", "cuda_x86", "cuda_arm64", "vulkan_x86")
                if any(f"'{name}'" in e for e in workflow_errors)
            ),
            ["cpu_arm64", "cpu_x86", "cuda_arm64", "cuda_x86", "vulkan_x86"],
            workflow_errors,
        )


class CommandLineTests(unittest.TestCase):
    def test_the_shipped_pair_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build/runtime deps OK", result.stdout)

    def test_a_defective_pair_exits_one_and_names_the_issue(self):
        with TemporaryDirectory() as tmp:
            broken = Path(tmp) / "Dockerfile"
            broken.write_text(
                "FROM scratch AS build\nRUN scripts/build-cpu-release.sh a b c d\n"
                "FROM scratch AS rt\nRUN apt-get install --yes libssl3\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dockerfile",
                    str(broken),
                    "--workflow",
                    str(WORKFLOW),
                ],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("#1517", result.stderr)

    def test_a_missing_file_is_an_error_not_a_pass(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--dockerfile", "/nonexistent/Dockerfile"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
