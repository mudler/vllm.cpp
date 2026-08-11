#!/usr/bin/env python3
"""Offline suite for the `main` baseline reader and its workflow lane (#274).

Two halves, both network-free.

The VERDICT half fixes the trap that made this row necessary. `sanitize-cpu`
carries `continue-on-error: true`, so the Actions API reports the RUN as
`success` while the job is `failure`. Run 31448896841 at 5812b8b6 is exactly
that shape and is the only completed `main` run in the 40-run window this row
measured: a reader of `run.conclusion` would publish it as the known-good SHA.
`test_run_level_success_with_red_sanitizers_is_not_green` is that mutation, and
it is the one test an implementation reading `run.conclusion` fails.

The WORKFLOW half asserts the lane can actually finish: every job-level
concurrency group must discriminate on the event, or a push to `main` cancels
the scheduled baseline exactly as it cancels the previous push -- which is the
measured cause of 26 cancelled runs out of 40. It also asserts the two
properties a reviewer most needs held constant: `push`/`pull_request` grouping
is unchanged, and the baseline verdict job can never run on a contributor's PR.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def load_module():
    path = ROOT / "scripts/main-baseline.py"
    spec = importlib.util.spec_from_file_location("main_baseline", path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `@dataclass` under `from __future__ import
    # annotations` resolves its field types through sys.modules[__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_module()


def job(name, conclusion):
    return {"name": name, "conclusion": conclusion, "status": "completed"}


# The real 31448896841 shape: run says success, both sanitizer lanes are red.
RUN_31448896841 = {
    "id": 31448896841,
    "head_sha": "5812b8b667eabcb7fe6f12767176a48c04ff7b26",
    "created_at": "2026-08-11T01:18:27Z",
    "conclusion": "success",
    "event": "push",
}
JOBS_31448896841 = [
    job("agent-record", "success"),
    job("build-test-cpu", "success"),
    job("build-test-cpu-arm64", "success"),
    job("build-test-vulkan", "success"),
    job("cuda-arch-features", "success"),
    job("cuda-fat-build", "success"),
    job("device-leakage", "success"),
    job("vulkan-spirv-freshness", "success"),
    job("pr-size", "skipped"),
    job("sanitize-cpu (address,undefined)", "failure"),
    job("sanitize-cpu (thread)", "failure"),
]

ALL_GREEN_JOBS = [
    job(name, "success")
    for name in (
        "agent-record",
        "build-test-cpu",
        "build-test-cpu-arm64",
        "build-test-vulkan",
        "cuda-arch-features",
        "cuda-fat-build",
        "device-leakage",
        "vulkan-spirv-freshness",
        "sanitize-cpu (address,undefined)",
        "sanitize-cpu (thread)",
    )
] + [job("pr-size", "skipped")]


class VerdictTests(unittest.TestCase):
    def test_run_level_success_with_red_sanitizers_is_not_green(self) -> None:
        """The mutation this row exists for: run.conclusion is not the verdict.

        An implementation that reads RUN_31448896841["conclusion"] passes every
        other test here and fails this one.
        """
        verdict = baseline.verdict(RUN_31448896841, JOBS_31448896841)
        self.assertFalse(verdict.green)
        self.assertEqual(
            verdict.failing,
            ["sanitize-cpu (address,undefined)", "sanitize-cpu (thread)"],
        )
        self.assertEqual(verdict.sha, RUN_31448896841["head_sha"])

    def test_all_green_is_green(self) -> None:
        verdict = baseline.verdict(RUN_31448896841, ALL_GREEN_JOBS)
        self.assertTrue(verdict.green)
        self.assertEqual(verdict.failing, [])

    def test_skipped_is_reported_as_not_covered_never_as_a_pass(self) -> None:
        verdict = baseline.verdict(RUN_31448896841, ALL_GREEN_JOBS)
        self.assertIn("pr-size", verdict.not_run)
        self.assertNotIn("pr-size", verdict.covered)

    def test_cancelled_is_not_green(self) -> None:
        """A cancelled long job is the defect, so it can never read as a pass."""
        jobs = [job("agent-record", "success"), job("cuda-fat-build", "cancelled")]
        verdict = baseline.verdict(RUN_31448896841, jobs)
        self.assertFalse(verdict.green)
        self.assertIn("cuda-fat-build", verdict.failing)

    def test_missing_and_in_progress_conclusions_are_not_green(self) -> None:
        for conclusion in (None, "", "timed_out", "action_required", "neutral"):
            with self.subTest(conclusion=conclusion):
                jobs = [job("agent-record", "success"), job("build-test-cpu", conclusion)]
                self.assertFalse(baseline.verdict(RUN_31448896841, jobs).green)

    def test_no_covered_jobs_fails_closed(self) -> None:
        """An empty or all-skipped job list is unknown, never green."""
        self.assertFalse(baseline.verdict(RUN_31448896841, []).green)
        self.assertFalse(
            baseline.verdict(RUN_31448896841, [job("pr-size", "skipped")]).green
        )

    def test_summary_job_is_excluded_from_its_own_verdict(self) -> None:
        """It queries while it is still running; counting itself is never green."""
        jobs = ALL_GREEN_JOBS + [
            {"name": baseline.SUMMARY_JOB, "conclusion": None, "status": "in_progress"}
        ]
        verdict = baseline.verdict(RUN_31448896841, jobs)
        self.assertTrue(verdict.green)
        self.assertNotIn(baseline.SUMMARY_JOB, verdict.covered)


class LastGreenTests(unittest.TestCase):
    def verdicts(self):
        newest = baseline.verdict(
            {**RUN_31448896841, "id": 3, "head_sha": "c" * 40,
             "created_at": "2026-08-11T12:00:00Z"},
            JOBS_31448896841,
        )
        middle = baseline.verdict(
            {**RUN_31448896841, "id": 2, "head_sha": "b" * 40,
             "created_at": "2026-08-11T08:00:00Z"},
            ALL_GREEN_JOBS,
        )
        oldest = baseline.verdict(
            {**RUN_31448896841, "id": 1, "head_sha": "a" * 40,
             "created_at": "2026-08-11T04:00:00Z"},
            ALL_GREEN_JOBS,
        )
        return [newest, middle, oldest]

    def test_last_green_is_the_newest_green_not_the_newest_run(self) -> None:
        self.assertEqual(baseline.last_green(self.verdicts()).sha, "b" * 40)

    def test_absence_of_any_green_is_explicit(self) -> None:
        reds = [v for v in self.verdicts() if not v.green]
        self.assertIsNone(baseline.last_green(reds))
        text = baseline.render(reds, degraded=None)
        self.assertIn("no fully green baseline", text.lower())
        self.assertNotIn("None", text.splitlines()[-1])

    def test_render_names_the_last_green_sha_and_the_newest_failures(self) -> None:
        text = baseline.render(self.verdicts(), degraded=None)
        self.assertIn("b" * 40, text)
        self.assertIn("sanitize-cpu (address,undefined)", text)

    def test_degraded_remote_is_reported_not_rendered_as_green(self) -> None:
        text = baseline.render([], degraded="REMOTE_UNVERIFIED: no network")
        self.assertIn("REMOTE_UNVERIFIED", text)
        self.assertNotIn("no fully green baseline", text.lower())


class EmitSummaryTests(unittest.TestCase):
    def test_exit_code_is_nonzero_for_the_red_shape(self) -> None:
        stream = io.StringIO()
        code = baseline.emit_summary(
            baseline.verdict(RUN_31448896841, JOBS_31448896841), stream
        )
        self.assertNotEqual(code, 0)
        self.assertIn("sanitize-cpu (thread)", stream.getvalue())

    def test_exit_code_is_zero_when_every_covered_job_is_green(self) -> None:
        stream = io.StringIO()
        code = baseline.emit_summary(
            baseline.verdict(RUN_31448896841, ALL_GREEN_JOBS), stream
        )
        self.assertEqual(code, 0)
        self.assertIn("5812b8b6", stream.getvalue())

    def test_json_output_carries_sha_green_and_failures(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            baseline.print_json(
                [baseline.verdict(RUN_31448896841, JOBS_31448896841)], degraded=None
            )
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["runs"][0]["green"])
        self.assertIsNone(payload["last_green"])
        self.assertIn(
            "sanitize-cpu (thread)", payload["runs"][0]["failing"]
        )


# ---------------------------------------------------------------------------
# Workflow lane
# ---------------------------------------------------------------------------

JOB_HEADER = re.compile(r"(?m)^  ([a-zA-Z0-9_-]+):$")
GROUP_LINE = re.compile(r"(?m)^      group: (.+)$")
CANCEL_LINE = re.compile(r"(?m)^      cancel-in-progress: (.+)$")

# Every job that carried a job-level group at 0eb049f7, the base of this row.
GROUPED_JOBS = (
    "agent-record",
    "cuda-arch-features",
    "cuda-fat-build",
    "vulkan-spirv-freshness",
    "build-test-vulkan",
    "device-leakage",
    "build-test-cpu",
    "build-test-cpu-arm64",
    "sanitize-cpu",
)
# Diff-scoped per-push gates. They must never regain a group (ci.yml:16-24) and
# they are not part of the baseline, whose subject is the tree.
UNGROUPED_JOBS = ("documentation-checkpoint", "commit-protocol-tag")


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:$\n(.*?)(?=^  [a-zA-Z0-9_-]+:$|\Z)", text
    )
    assert match, f"job {name} not found in ci.yml"
    return match.group(1)


class WorkflowLaneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = workflow_text()

    def test_baseline_triggers_exist(self) -> None:
        header = self.text.split("\njobs:\n", 1)[0]
        self.assertIn("schedule:", header)
        self.assertIn("workflow_dispatch:", header)
        self.assertRegex(header, r"- cron: ['\"][^'\"]+['\"]")

    def test_every_job_level_group_discriminates_on_the_event(self) -> None:
        """Without this a push to main cancels the scheduled baseline.

        `github.ref` is `refs/heads/main` for BOTH a push and a scheduled run,
        so a ref-only key puts them in one group and the baseline dies exactly
        as the previous push's jobs do -- 26 of 40 runs in the measured window.
        """
        for name in GROUPED_JOBS:
            with self.subTest(job=name):
                block = job_block(self.text, name)
                groups = GROUP_LINE.findall(block)
                self.assertEqual(len(groups), 1, f"{name} must have exactly one group")
                self.assertIn("github.event_name", groups[0])
                self.assertIn("github.ref", groups[0])

    def test_push_and_pull_request_grouping_is_unchanged(self) -> None:
        """The added token is a constant for push and for PR, so the partition
        of push runs (and of PR runs) into groups is exactly what it was."""
        for name in GROUPED_JOBS:
            with self.subTest(job=name):
                group = GROUP_LINE.findall(job_block(self.text, name))[0]
                for event in ("push", "pull_request"):
                    resolved = group.replace("${{ github.event_name }}", event)
                    self.assertNotIn(
                        "github.event_name",
                        resolved,
                        "the event token must resolve to a constant per event",
                    )
                    self.assertNotIn(
                        "github.run_id",
                        resolved,
                        "a run-unique key would disable push dedupe entirely",
                    )

    def test_the_baseline_lane_is_never_cancellable(self) -> None:
        for name in GROUPED_JOBS:
            with self.subTest(job=name):
                block = job_block(self.text, name)
                cancels = CANCEL_LINE.findall(block)
                self.assertEqual(len(cancels), 1)
                expression = cancels[0]
                self.assertIn("schedule", expression)
                self.assertIn("workflow_dispatch", expression)
                self.assertNotEqual(expression.strip(), "true")

    def test_diff_scoped_jobs_still_carry_no_concurrency_group(self) -> None:
        for name in UNGROUPED_JOBS:
            with self.subTest(job=name):
                self.assertNotIn("concurrency:", job_block(self.text, name))

    def test_diff_scoped_jobs_are_skipped_on_the_baseline_lane(self) -> None:
        """They range over `github.event.before..github.sha`; a schedule payload
        has no `before`, so running them there would gate an empty range."""
        for name in UNGROUPED_JOBS:
            with self.subTest(job=name):
                block = job_block(self.text, name)
                condition = re.search(r"(?m)^    if: (.+)$", block)
                self.assertIsNotNone(condition, f"{name} needs an event guard")
                self.assertIn("schedule", condition.group(1))
                self.assertIn("workflow_dispatch", condition.group(1))

    def test_summary_job_can_never_run_on_a_contributors_pr(self) -> None:
        block = job_block(self.text, baseline.SUMMARY_JOB)
        condition = re.search(r"(?m)^    if: (.+)$", block)
        self.assertIsNotNone(condition)
        guard = condition.group(1)
        self.assertIn("always()", guard)
        self.assertIn("schedule", guard)
        self.assertIn("workflow_dispatch", guard)
        self.assertNotIn("pull_request", guard)

    def test_summary_job_waits_on_every_grouped_job(self) -> None:
        block = job_block(self.text, baseline.SUMMARY_JOB)
        needs = re.search(r"(?ms)^    needs:.*?$\n(?:      - .+$\n)+", block)
        self.assertIsNotNone(needs, "baseline-summary must declare needs as a list")
        for name in GROUPED_JOBS:
            with self.subTest(job=name):
                self.assertIn(f"      - {name}\n", needs.group(0))

    def test_summary_job_reads_the_api_and_holds_actions_read(self) -> None:
        block = job_block(self.text, baseline.SUMMARY_JOB)
        self.assertIn("actions: read", block)
        self.assertIn("scripts/main-baseline.py", block)
        self.assertIn("--emit-summary", block)
        self.assertIn("github.run_id", block)
        self.assertNotIn(
            "needs.sanitize-cpu.result",
            block,
            "continue-on-error makes needs.<job>.result report success; use the API",
        )

    def test_sanitize_cpu_stays_continue_on_error_for_the_push_and_pr_lanes(self) -> None:
        """Out of scope for this row: it is the closing step of the hardening
        row. The baseline lane gets its bindingness from baseline-summary."""
        self.assertIn("continue-on-error: true", job_block(self.text, "sanitize-cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
