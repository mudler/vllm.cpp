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

WHY THE WORKFLOW HALF EVALUATES INSTEAD OF GREPPING. Its first version checked
substrings, and four planted defects walked straight through it:

  a. inverting `cancel-in-progress` to
     `== 'schedule' || == 'workflow_dispatch'` keeps every substring the old
     assertion looked for while turning PR cancellation OFF and baseline
     self-cancellation ON. `CancellationPolicyTests` now resolves the
     expression to a BOOLEAN per event.
  b. adding `${{ github.sha }}` to a job group key destroys push/PR dedupe.
     The old test only blocked `github.run_id`. `GroupKeyTests` pins the
     resolved key against a literal derived from the base revision's key.
  c/d. `|| true` on the verdict step, or `continue-on-error: true` on the
     verdict job, rebuild the exact `sanitize-cpu` defect this row is about one
     level up. `VerdictJobCannotSwallowFailureTests` refuses both.

And `AgentRecordDiffRangeTests` EXECUTES the two `agent-record` step bodies that
consume `github.event.before` under a `python3` shim: on a `schedule` payload
that variable renders empty, and before the guard landed the step died under
`set -eu` with `--range: range must be exactly BASE..HEAD`, so `agent-record`
could never be green and the baseline could never publish GREEN at all.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"


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


# The step whose conclusion IS `commit-protocol-tag`'s verdict. Named once so
# the payload builders and the assertions cannot drift apart.
GATE_STEP = "Every new commit satisfies the strict trailer contract"


def job_with_steps(name, conclusion, steps):
    """A jobs-API entry carrying its `steps` array, as the real payload does.

    `steps[].conclusion` is the only field that says whether a given step RAN.
    A job that concludes `failure` because step 3 failed reports steps 4..N as
    `skipped`, and the job's own conclusion cannot tell the two apart.
    """

    return {
        "name": name,
        "conclusion": conclusion,
        "status": "completed",
        "steps": [
            {
                "name": step_name,
                "conclusion": step_conclusion,
                "status": "completed",
                "number": position + 1,
            }
            for position, (step_name, step_conclusion) in enumerate(steps)
        ],
    }


def job(name, conclusion, steps=None):
    """A jobs-API entry.

    `steps` defaults to the ORDINARY case: one step, named `GATE_STEP`, that
    reached the same conclusion the job did. The interesting case is the one
    where they DIFFER, and `job_with_steps` states that one out loud.
    """

    return job_with_steps(
        name, conclusion, [(GATE_STEP, conclusion)] if steps is None else steps
    )



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
        "windows-msvc-cpu",
        "windows-msvc-vulkan",
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

    def test_an_expected_job_the_payload_never_mentions_is_red(self) -> None:
        """The narrowing mutation: eight of nine covered jobs simply absent.

        Without an expected-job set the verdict is purely subtractive and this
        prints GREEN with `jobs covered: 1` -- a job renamed or deleted by an
        unrelated PR silently shrinks the baseline while it keeps publishing
        green.
        """
        verdict = baseline.verdict(RUN_31448896841, [job("agent-record", "success")])
        self.assertFalse(verdict.green)
        self.assertEqual(verdict.covered, ["agent-record"])
        self.assertEqual(
            sorted(verdict.missing),
            sorted(n for n in baseline.EXPECTED_JOBS if n != "agent-record"),
        )
        self.assertEqual(verdict.failing, [], "absent is not the same as failed")

    def test_a_matrix_job_discharges_its_expectation_under_the_bare_id(self) -> None:
        """The API reports `sanitize-cpu (thread)`; the expectation is the id."""
        verdict = baseline.verdict(RUN_31448896841, ALL_GREEN_JOBS)
        self.assertEqual(verdict.missing, [])
        self.assertTrue(verdict.green)

    def test_an_expected_job_that_was_skipped_counts_as_never_ran(self) -> None:
        jobs = [j for j in ALL_GREEN_JOBS if j["name"] != "device-leakage"]
        jobs.append(job("device-leakage", "skipped"))
        verdict = baseline.verdict(RUN_31448896841, jobs)
        self.assertFalse(verdict.green)
        self.assertIn("device-leakage", verdict.missing)

    def test_expected_jobs_is_pinned_against_the_workflow_needs_list(self) -> None:
        """The workflow is the authority; the constant is the pin.

        A PR that renames a covered job reds THIS assertion instead of quietly
        narrowing every future baseline.
        """
        self.assertEqual(
            tuple(sorted(baseline.EXPECTED_JOBS)),
            baseline.expected_jobs_from_workflow(),
        )
        # 9 until 2026-08-17, then 11: `windows-msvc-cpu` and
        # `windows-msvc-vulkan` joined the lane (#503). The literal is here so
        # that DROPPING a job cannot be spelled as an edit to one list -- the
        # equality above is satisfied by narrowing both sides together, and this
        # is not.
        self.assertEqual(len(baseline.EXPECTED_JOBS), 11)

    def test_an_unfinished_job_is_pending_not_failed(self) -> None:
        """Fail-closed is right; calling it a FAILURE is a wrong label."""
        jobs = [dict(entry) for entry in ALL_GREEN_JOBS]
        for entry in jobs:
            if entry["name"] == "build-test-cpu":
                entry["conclusion"] = None
                entry["status"] = "in_progress"
        verdict = baseline.verdict(RUN_31448896841, jobs)
        self.assertFalse(verdict.green)
        self.assertIn("build-test-cpu", verdict.pending)
        self.assertNotIn("build-test-cpu", verdict.failing)
        self.assertNotIn("build-test-cpu", verdict.covered)
        self.assertNotIn("build-test-cpu", verdict.missing)


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

    def test_a_narrowed_run_reports_red_and_names_what_never_ran(self) -> None:
        stream = io.StringIO()
        code = baseline.emit_summary(
            baseline.verdict(RUN_31448896841, [job("agent-record", "success")]), stream
        )
        self.assertNotEqual(code, 0)
        text = stream.getvalue()
        self.assertIn("main baseline: RED", text)
        self.assertIn("never ran", text)
        self.assertIn("cuda-fat-build", text)


class ExitCodeTests(unittest.TestCase):
    """Absence is not success -- the confusion the module docstring names."""

    def run_main(self, argv, collected):
        original = baseline.collect
        baseline.collect = lambda limit: collected
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = baseline.main(argv)
        finally:
            baseline.collect = original
        return code, buffer.getvalue()

    def run_anchor(self, argv, anchor):
        original = baseline.gate_anchor
        baseline.gate_anchor = lambda *args, **kwargs: anchor
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                code = baseline.main(argv)
        finally:
            baseline.gate_anchor = original
        return code, buffer.getvalue()

    def test_the_anchor_CLI_exits_3_on_a_degraded_read(self) -> None:
        """3 is REMOTE_UNVERIFIED and 1 is a clean absence.

        Collapsing them is F3: the caller cannot tell "this branch has no gated
        history, so `PUSH_BASE` is honest" from "the forge did not answer, so
        the base is unknown". The second must SKIP the gate, because a narrowed
        pass would advance the step's own anchor past what the narrowing
        dropped. `ci.yml` branches on exactly this number.
        """
        code, _ = self.run_anchor(
            ["--gate-anchor", "documentation-checkpoint", "--gate-step", "s"],
            baseline.Anchor(sha="", source="none", degraded="REMOTE_UNVERIFIED: x"),
        )
        self.assertEqual(code, 3)

    def test_the_anchor_CLI_exits_1_on_a_CLEAN_absence(self) -> None:
        code, _ = self.run_anchor(
            ["--gate-anchor", "documentation-checkpoint", "--gate-step", "s"],
            baseline.Anchor(sha="", source="none", degraded=None),
        )
        self.assertEqual(code, 1)

    def test_the_anchor_CLI_exits_3_when_told_to_stay_OFFLINE(self) -> None:
        """`--offline` is the same class of answer: the forge was not asked."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = baseline.main(
                ["--gate-anchor", "x", "--gate-step", "s", "--offline"]
            )
        self.assertEqual(code, 3)

    def test_the_anchor_CLI_REFUSES_without_a_gate_step(self) -> None:
        """Not a default. Exiting 2 is what stops a future caller re-creating
        the job-granularity hole by simply omitting the flag."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = baseline.main(["--gate-anchor", "documentation-checkpoint"])
        self.assertEqual(code, 2)

    def test_no_completed_baseline_run_exits_nonzero(self) -> None:
        """`main-baseline.py && echo ok` must not print ok when nothing ran."""
        code, text = self.run_main([], ([], None))
        self.assertNotEqual(code, 0)
        self.assertIn("No completed baseline run found on main.", text)

    def test_no_completed_baseline_run_exits_nonzero_in_json_mode_too(self) -> None:
        code, text = self.run_main(["--json"], ([], None))
        self.assertNotEqual(code, 0)
        self.assertEqual(json.loads(text)["runs"], [])

    def test_a_degraded_remote_exits_nonzero(self) -> None:
        code, text = self.run_main([], ([], "REMOTE_UNVERIFIED: no network"))
        self.assertNotEqual(code, 0)
        self.assertIn("REMOTE_UNVERIFIED", text)

    def test_a_real_baseline_history_exits_zero(self) -> None:
        code, _ = self.run_main(
            [], ([baseline.verdict(RUN_31448896841, ALL_GREEN_JOBS)], None)
        )
        self.assertEqual(code, 0)


# ---------------------------------------------------------------------------
# Workflow lane
# ---------------------------------------------------------------------------

JOB_HEADER = re.compile(r"(?m)^  ([a-zA-Z0-9_-]+):$")
GROUP_LINE = re.compile(r"(?m)^      group: (.+)$")
CANCEL_LINE = re.compile(r"(?m)^      cancel-in-progress: (.+)$")

# Every GitHub event this workflow can be triggered by. The resolved-expression
# assertions below cover all four, because a polarity mutation is only visible
# as a DISAGREEMENT between the baseline events and the contributor events.
EVENTS = ("push", "pull_request", "schedule", "workflow_dispatch")
BASELINE_EVENTS = ("schedule", "workflow_dispatch")

# Every job that carried a job-level group at 0eb049f7, the base of this row,
# mapped to THE EXACT KEY IT CARRIED THERE (`git show 0eb049f7:.github/
# workflows/ci.yml`). The keys below are the whole content of the "grouping is
# unchanged" claim: this row's key must be the base key with the constant
# `-<event>` inserted, and nothing else. Inserting a run-varying token such as
# `${{ github.sha }}` -- which would destroy push and PR dedupe entirely --
# fails the equality, which a substring check could not see.
BASE_GROUPS = {
    "agent-record": "ci-agent-record-${{ github.ref }}-${{ github.repository }}",
    "cuda-arch-features": "ci-cuda-arch-features-${{ github.ref }}-${{ github.repository }}",
    "cuda-fat-build": "ci-cuda-fat-build-${{ github.ref }}-${{ github.repository }}",
    "vulkan-spirv-freshness": "ci-vulkan-spirv-freshness-${{ github.ref }}-${{ github.repository }}",
    "build-test-vulkan": "ci-build-test-vulkan-${{ github.ref }}-${{ github.repository }}",
    "device-leakage": "ci-device-leakage-${{ github.ref }}-${{ github.repository }}",
    "build-test-cpu": "ci-build-test-cpu-${{ github.ref }}-${{ github.repository }}",
    "build-test-cpu-arm64": "ci-build-test-cpu-arm64-${{ github.ref }}-${{ github.repository }}",
    "sanitize-cpu": "ci-sanitize-cpu-${{ matrix.lane }}-${{ github.ref }}-${{ github.repository }}",
}
GROUPED_JOBS = tuple(BASE_GROUPS)
# Any of these in a concurrency key means the key varies per RUN, which is the
# same thing as having no key: every run gets its own group and dedupe stops.
RUN_VARYING_TOKENS = (
    "github.run_id",
    "github.run_number",
    "github.run_attempt",
    "github.sha",
    "github.event.head_commit.id",
    "github.event.pull_request.head.sha",
)
# Diff-scoped per-push gates. They must never regain a group (ci.yml:16-24) and
# they are not part of the baseline, whose subject is the tree.
UNGROUPED_JOBS = ("documentation-checkpoint", "commit-protocol-tag")

EXPRESSION_TOKEN = re.compile(
    r"""\s*(?:
        (?P<wrap>\$\{\{|\}\})
      | (?P<op><=|>=|&&|\|\||==|!=|\(|\))
      | (?P<ctx>github\.event_name)
      | (?P<text>'[^']*')
      | (?P<literal>true|false)
    )\s*""",
    re.VERBOSE,
)


def resolve_boolean(expression: str, event: str) -> bool:
    """Evaluate the GitHub-expression subset these keys use, for one event.

    A real evaluation, not a substring check, because the mutation that matters
    keeps every substring. Inverting a key to
    `github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'`
    still "contains schedule", still "contains workflow_dispatch", still is not
    the literal `true` -- and it turns PR/push cancellation OFF and baseline
    self-cancellation ON, which is the precise opposite of the intent.

    Any token outside the recognised subset raises rather than being guessed at:
    an unreadable key must red the suite, never quietly resolve to something.
    """

    rendered: list[str] = []
    position = 0
    while position < len(expression):
        match = EXPRESSION_TOKEN.match(expression, position)
        if match is None:
            raise AssertionError(
                f"unrecognised token in {expression!r} at {expression[position:]!r}"
            )
        position = match.end()
        if match.group("wrap"):
            continue
        if match.group("ctx"):
            rendered.append(repr(event))
        elif match.group("op"):
            rendered.append(
                {"&&": " and ", "||": " or "}.get(match.group("op"), match.group("op"))
            )
        elif match.group("text"):
            rendered.append(match.group("text"))
        else:
            rendered.append(match.group("literal").capitalize())
    source = "".join(rendered).strip()
    if not source:
        raise AssertionError(f"empty expression: {expression!r}")
    value = eval(source, {"__builtins__": {}}, {})  # noqa: S307 -- fixed subset above
    if not isinstance(value, bool):
        raise AssertionError(f"{expression!r} did not resolve to a boolean for {event}")
    return value


# ---------------------------------------------------------------------------
# Group keys, resolved to a VALUE
# ---------------------------------------------------------------------------
#
# `resolve_boolean` above answers "does this expression mean true for this
# event". A group key is not a boolean, and since #274's follow-on it is not a
# constant either: the baseline events carry a conditional that admits the run
# identity, and every other event must resolve to exactly the key it resolved to
# before. Half-resolving the string cannot see that, so the resolver below
# evaluates the same grammar over VALUES, with GitHub's truthiness for `&&` and
# `||` (`null`, `false`, `0` and `''` are falsy, and both operators return an
# OPERAND rather than a boolean).
#
# Contexts split into two classes, and the split is the whole point. A
# RUN-VARYING context takes a different value in two different runs of the same
# workflow on the same ref. A STABLE one does not. Resolving one key twice, once
# per synthetic run, turns "does this key carry `github.run_id`" into the
# property that actually matters: does this key put two runs in the same group.

RUN_A = {
    "github.run_id": "5000000001",
    "github.run_number": "801",
    "github.run_attempt": "1",
    "github.sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "github.event.head_commit.id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "github.event.pull_request.head.sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
}
RUN_B = {
    "github.run_id": "5000000002",
    "github.run_number": "802",
    "github.run_attempt": "2",
    "github.sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "github.event.head_commit.id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "github.event.pull_request.head.sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
}
assert set(RUN_A) == set(RUN_B) == set(RUN_VARYING_TOKENS), (
    "the run-varying contexts and the by-name blocklist must name the same set"
)
assert all(RUN_A[k] != RUN_B[k] for k in RUN_A), "two runs must differ everywhere"

# Stable for a given event: the same in run A and run B. `matrix.lane` is stable
# because it identifies the lane, not the run, which is exactly why
# `sanitize-cpu`'s two lanes are allowed to hold separate groups.
STABLE_CONTEXTS = {
    "push": {
        "github.event_name": "push",
        "github.ref": "refs/heads/main",
        "github.ref_name": "main",
        "github.repository": "mudler/vllm.cpp",
        "github.event.pull_request.number": None,
        "matrix.lane": "thread",
    },
    "pull_request": {
        "github.event_name": "pull_request",
        "github.ref": "refs/pull/1234/merge",
        "github.ref_name": "1234/merge",
        "github.repository": "mudler/vllm.cpp",
        "github.event.pull_request.number": 1234,
        "matrix.lane": "thread",
    },
    "schedule": {
        "github.event_name": "schedule",
        "github.ref": "refs/heads/main",
        "github.ref_name": "main",
        "github.repository": "mudler/vllm.cpp",
        "github.event.pull_request.number": None,
        "matrix.lane": "thread",
    },
    "workflow_dispatch": {
        "github.event_name": "workflow_dispatch",
        "github.ref": "refs/heads/main",
        "github.ref_name": "main",
        "github.repository": "mudler/vllm.cpp",
        "github.event.pull_request.number": None,
        "matrix.lane": "thread",
    },
}
assert set(STABLE_CONTEXTS) == set(EVENTS)

VALUE_TOKEN = re.compile(
    r"""\s*(?:
        (?P<op>&&|\|\||==|!=|!|\(|\))
      | (?P<text>'[^']*')
      | (?P<literal>true|false|null)
      | (?P<number>[0-9]+)
      | (?P<ctx>[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)
    )\s*""",
    re.VERBOSE,
)


def _truthy(value) -> bool:
    """GitHub's truthiness. `null`, `false`, `0` and the empty string are false."""
    return not (value is None or value is False or value == 0 or value == "")


def _render(value) -> str:
    """How GitHub substitutes a resolved value back into the key text."""
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


class _Expression:
    """Recursive descent over the subset these keys use.

    An unrecognised token or an unknown context RAISES. A key this suite cannot
    read must red it, never resolve to something plausible -- the failure mode
    the workflow half of this file exists to refuse.
    """

    def __init__(self, source: str, contexts: dict) -> None:
        self.source = source
        self.contexts = contexts
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(source):
            match = VALUE_TOKEN.match(source, position)
            if match is None:
                raise AssertionError(
                    f"unrecognised token in {source!r} at {source[position:]!r}"
                )
            position = match.end()
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else (None, None)

    def take(self):
        token = self.peek()
        self.index += 1
        return token

    def parse(self):
        value = self.parse_or()
        if self.index != len(self.tokens):
            raise AssertionError(f"trailing tokens in {self.source!r}")
        return value

    def parse_or(self):
        value = self.parse_and()
        while self.peek() == ("op", "||"):
            self.take()
            right = self.parse_and()
            value = value if _truthy(value) else right
        return value

    def parse_and(self):
        value = self.parse_compare()
        while self.peek() == ("op", "&&"):
            self.take()
            right = self.parse_compare()
            value = right if _truthy(value) else value
        return value

    def parse_compare(self):
        left = self.parse_primary()
        kind, text = self.peek()
        if kind == "op" and text in ("==", "!="):
            self.take()
            right = self.parse_primary()
            return (left == right) if text == "==" else (left != right)
        return left

    def parse_primary(self):
        kind, text = self.take()
        if (kind, text) == ("op", "("):
            value = self.parse_or()
            if self.take() != ("op", ")"):
                raise AssertionError(f"unbalanced parenthesis in {self.source!r}")
            return value
        if (kind, text) == ("op", "!"):
            # Unary negation binds tighter than `==`, as it does on the forge.
            # It is here for `!cancelled()`, the condition that keeps a gate
            # step running after an earlier step failed (#1776).
            return not _truthy(self.parse_primary())
        if kind == "text":
            return text[1:-1]
        if kind == "number":
            return int(text)
        if kind == "literal":
            return {"true": True, "false": False, "null": None}[text]
        if kind == "ctx":
            if self.peek() == ("op", "("):
                # A STATUS FUNCTION -- `cancelled()`, `success()`, `failure()`,
                # `always()`. It is resolved from `contexts` under its called
                # name, so an unmodelled one raises exactly as an unmodelled
                # context does. None of them takes an argument in this subset.
                self.take()
                if self.take() != ("op", ")"):
                    raise AssertionError(
                        f"{text}(...) takes no argument in {self.source!r}"
                    )
                text = f"{text}()"
            if text not in self.contexts:
                raise AssertionError(
                    f"unknown context {text!r} in {self.source!r}; add it to "
                    "STABLE_CONTEXTS or to the run-varying set, and say which"
                )
            return self.contexts[text]
        raise AssertionError(f"unexpected token {text!r} in {self.source!r}")


def resolve_group(expression: str, event: str, run: dict) -> str:
    """Render a concurrency group key the way GitHub renders it for one run."""

    contexts = dict(STABLE_CONTEXTS[event])
    contexts.update(run)
    out: list[str] = []
    position = 0
    while position < len(expression):
        start = expression.find("${{", position)
        if start < 0:
            out.append(expression[position:])
            break
        out.append(expression[position:start])
        end = expression.find("}}", start)
        if end < 0:
            raise AssertionError(f"unterminated ${{{{ in {expression!r}")
        out.append(_render(_Expression(expression[start + 3 : end], contexts).parse()))
        position = end + 2
    return "".join(out)


def varies_per_run(expression: str, event: str) -> bool:
    """Do two runs of this workflow, same event and same ref, get SEPARATE groups?

    True means the group can never make one run wait for another, so GitHub can
    never discard it while it is pending. False means the group is shared, which
    is what `cancel-in-progress: true` needs to have anything to cancel.
    """

    return resolve_group(expression, event, RUN_A) != resolve_group(
        expression, event, RUN_B
    )


def resolve_condition(expression: str, state: dict) -> bool:
    """Resolve a step-level `if:` to the boolean GitHub decides to run on.

    A step guard is not a concurrency key, so it gets its own entry point, but
    it uses the same evaluator for the same reason: the mutation that matters
    keeps every substring. Appending `|| true` to the guards of all five gate
    steps leaves `!cancelled()`, both `steps.*.outcome == 'success'` clauses and
    `env.GATE_ANCHOR_DEGRADED != 'true'` byte-for-byte intact, and turns the
    whole conjunction into the constant `true` -- which restores exactly the
    degraded-read narrowing this row exists to remove. A substring assertion
    cannot see that. A resolved boolean cannot miss it.

    `state` supplies every context AND every status function the guard names.
    Anything it does not model raises, so a guard that grows a new term reds
    this suite instead of resolving to something plausible.
    """

    inner = expression.strip()
    if inner.startswith("${{") and inner.endswith("}}"):
        inner = inner[3:-2]
    elif "${{" in inner:
        raise AssertionError(f"partial interpolation in a step guard: {expression!r}")
    return _truthy(_Expression(inner, state).parse())


SHELL_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "fi", "while", "do", "done",
     "case", "esac", "in", "{", "}", "!"}
)

# What a gate step's body may run BEFORE its gate. Every entry computes the
# range or narrates it, and none of them can abort the step in a way that would
# make its conclusion a statement about a gate that never executed:
# `set` configures the shell, `[` is a conditional whose failure the surrounding
# `if`/`||` consumes, and `echo` writes to the step log.
RANGE_PRELUDE_COMMANDS = frozenset({"set", "[", "echo", ":"})


def commands_in(body: str) -> list[tuple[str, str]]:
    """`(command word, its statement)` for every command a step body runs.

    Whole-line comments are dropped, backslash continuations are joined, and
    each logical line is split on the operators that separate statements. Shell
    keywords and assignments are not commands. `for` and `case` headers are
    skipped whole, because their word list is data rather than a call.
    """

    logical: list[str] = []
    pending = ""
    for raw in body.splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if pending:
            text = f"{pending} {text}"
            pending = ""
        if text.endswith("\\"):
            pending = text[:-1].strip()
            continue
        logical.append(text)
    if pending:
        logical.append(pending)

    found: list[tuple[str, str]] = []
    for line in logical:
        for piece in re.split(r"(?:&&|\|\||;|\|)", line):
            words = piece.split()
            if words and words[0] in ("for", "case"):
                continue
            while words and words[0] in SHELL_KEYWORDS:
                words = words[1:]
            if not words:
                continue
            head = words[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", head):
                continue
            found.append((head, piece.strip()))
    return found


def concurrency_blocks(ci: dict) -> list[tuple[str, dict]]:
    """Every concurrency block in the workflow, workflow level first.

    Enumerated from the parsed file rather than from a fixed list, so a job that
    joins later is covered by the pull request that adds it.
    """

    blocks = []
    if ci.get("concurrency"):
        blocks.append(("<workflow>", ci["concurrency"]))
    for name, job in ci["jobs"].items():
        if job.get("concurrency"):
            blocks.append((name, job["concurrency"]))
    return blocks


def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def job_block(text: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:$\n(.*?)(?=^  [a-zA-Z0-9_-]+:$|\Z)", text
    )
    assert match, f"job {name} not found in ci.yml"
    return match.group(1)


def code_lines(block: str) -> list[str]:
    """Drop whole-line YAML comments.

    `baseline-summary`'s own prose explains why `sanitize-cpu` is
    `continue-on-error`, so a naive substring assertion about that key would
    trip over the comment that documents it.
    """

    return [line for line in block.splitlines() if not line.strip().startswith("#")]


def steps_of(job: str) -> list[list[str]]:
    """Split a job block into its `steps:` sequence entries, indentation kept."""

    lines = job.splitlines()
    try:
        first = next(i for i, line in enumerate(lines) if line == "    steps:") + 1
    except StopIteration:
        return []
    starts = [i for i in range(first, len(lines)) if lines[i].startswith("      - ")]
    steps = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        steps.append(lines[start:end])
    return steps


def step_env(step: list[str]) -> dict[str, str]:
    try:
        start = next(i for i, line in enumerate(step) if line == "        env:") + 1
    except StopIteration:
        return {}
    mapping: dict[str, str] = {}
    for line in step[start:]:
        match = re.fullmatch(r" {10}([A-Za-z_][A-Za-z0-9_]*): (.*)", line)
        if match is None:
            break
        mapping[match.group(1)] = match.group(2)
    return mapping


def step_run_body(step: list[str]) -> str | None:
    """The literal `run: |` block of a step, dedented to column zero."""

    try:
        start = next(i for i, line in enumerate(step) if line == "        run: |") + 1
    except StopIteration:
        return None
    body: list[str] = []
    for line in step[start:]:
        if not line.strip():
            body.append("")
            continue
        if not line.startswith(" " * 10):
            break
        body.append(line[10:])
    return "\n".join(body) + "\n"


def run_shimmed(
    body: str, environment: dict[str, str], python_exit: int = 0
) -> tuple[int, list[list[str]], str]:
    """Execute a step body with `python3` replaced by an argv recorder.

    No checker actually runs; what is under test is the SHELL logic that decides
    which checkers get invoked, and with which range. `python_exit` makes the
    recorder fail on demand, which is how the REMOTE_UNVERIFIED path -- rc 3 --
    gets executed rather than described.
    """

    with tempfile.TemporaryDirectory(prefix="vllm-baseline-step-") as temporary:
        root = Path(temporary)
        shim = root / "shim"
        shim.mkdir()
        trace = root / "argv.log"
        recorder = shim / "python3"
        recorder.write_text(
            "#!/bin/sh\n"
            "{\n"
            "  printf 'ARGV'\n"
            '  for a in "$@"; do printf \'\\t%s\' "$a"; done\n'
            "  printf '\\n'\n"
            '} >> "$VLLM_BASELINE_ARGV"\n'
            f"exit {python_exit}\n",
            encoding="utf-8",
        )
        recorder.chmod(0o700)
        script = root / "step.sh"
        script.write_text(body, encoding="utf-8")
        env = dict(os.environ)
        env["PATH"] = f"{shim}{os.pathsep}{env.get('PATH', '')}"
        env["VLLM_BASELINE_ARGV"] = str(trace)
        env.update(environment)
        result = subprocess.run(
            ["bash", str(script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        raw = trace.read_text(encoding="utf-8") if trace.exists() else ""
    invocations = [
        line.split("\t")[1:] for line in raw.splitlines() if line.startswith("ARGV")
    ]
    return result.returncode, invocations, result.stdout + result.stderr


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

    def test_each_resolved_group_is_the_base_key_plus_the_event_constant(self) -> None:
        """The whole "grouping is unchanged" claim, pinned as an equality.

        Substituting the event and comparing against the base revision's key
        with `-<event>` inserted proves two things at once: the partition of
        `push` runs (and of PR runs) into groups is bijective with what it was,
        and NOTHING ELSE was added to the key. Its predecessor compared against
        no baseline at all and blocked only `github.run_id`, so adding
        `${{ github.sha }}` -- which gives every run its own group and kills
        dedupe outright -- walked straight through it.
        """
        marker = "-${{ github.ref }}"
        expected_shape = "-${{ github.event_name }}" + marker
        for name in GROUPED_JOBS:
            base = BASE_GROUPS[name]
            self.assertEqual(base.count(marker), 1, f"{name}: base key shape changed")
            expected = base.replace(marker, expected_shape)
            groups = GROUP_LINE.findall(job_block(self.text, name))
            self.assertEqual(len(groups), 1, f"{name} must have exactly one group")
            for event in ("push", "pull_request"):
                with self.subTest(job=name, event=event):
                    self.assertEqual(
                        resolve_group(groups[0], event, RUN_A),
                        resolve_group(expected, event, RUN_A),
                        f"{name}: the {event} key no longer resolves to the base "
                        "key plus the event constant, so this lane's runs are "
                        "partitioned into different groups than they were",
                    )
            for event in BASELINE_EVENTS:
                with self.subTest(job=name, event=event):
                    self.assertTrue(
                        varies_per_run(groups[0], event),
                        f"{name}: two {event} runs share one group, and a group "
                        "that never cancels holds ONE pending run -- the third "
                        "arrival discards the second before it starts a job",
                    )

    def test_a_baseline_group_is_unique_per_run_and_a_contributor_group_is_not(
        self,
    ) -> None:
        """An independent statement of the same defect, in both directions.

        Its predecessor blocked `github.run_id` by SUBSTRING, for every event.
        That was right about the push and pull request lanes and wrong about the
        baseline lane, where a shared key is not dedupe but a one-slot queue
        that discards its own contents: runs 32140419182 and 32206456661 each
        returned `startedAt: null` for EVERY job and were cancelled the second
        their successor was created.

        The assertion is now the resolved property rather than the token.
        Resolving one key against two synthetic runs asks the question the token
        was standing in for: do two runs of this workflow, same event and same
        ref, land in the same group.
        """
        for name in GROUPED_JOBS:
            group = GROUP_LINE.findall(job_block(self.text, name))[0]
            for event in ("push", "pull_request"):
                with self.subTest(job=name, event=event):
                    self.assertFalse(
                        varies_per_run(group, event),
                        f"{name}: two {event} runs get separate groups, so "
                        "cancel-in-progress has nothing to cancel and dedupe "
                        "is off",
                    )
            for event in BASELINE_EVENTS:
                with self.subTest(job=name, event=event):
                    self.assertTrue(
                        varies_per_run(group, event),
                        f"{name}: two {event} runs share one group, so the "
                        "second waits and the third discards it",
                    )

    def test_cancellation_resolves_to_the_right_boolean_for_every_event(self) -> None:
        """push/PR still cancel; the baseline lane never cancels itself.

        Resolved, not grepped. The polarity inversion
        (`== 'schedule' || == 'workflow_dispatch'`) satisfies every substring
        assertion the previous version made and reverses all nine keys.
        """
        for name in GROUPED_JOBS:
            cancels = CANCEL_LINE.findall(job_block(self.text, name))
            self.assertEqual(len(cancels), 1, f"{name} must have exactly one key")
            for event in EVENTS:
                with self.subTest(job=name, event=event):
                    self.assertEqual(
                        resolve_boolean(cancels[0], event),
                        event not in BASELINE_EVENTS,
                        f"{name} cancel-in-progress is wrong for {event}",
                    )

    def test_the_workflow_level_group_cancels_prs_and_pushes_only(self) -> None:
        """#822 ratified latest-only for BOTH the pull request and main push
        lanes. It was previously pull-request-only, because a cancelled push
        left its diff-scoped range permanently unwalked. That is now safe: the
        gates base on the last SUCCESSFULLY gated commit, so a cancelled run's
        commits are covered by the next one (#863).

        `schedule` and `workflow_dispatch` stay non-cancellable. The baseline
        lane exists to answer "is main green", and a cancelled baseline answers
        nothing (#274).
        """
        header = self.text.split("\njobs:\n", 1)[0]
        cancel = re.search(r"(?m)^  cancel-in-progress: (.+)$", header)
        self.assertIsNotNone(cancel)
        for event in EVENTS:
            with self.subTest(event=event):
                self.assertEqual(
                    resolve_boolean(cancel.group(1), event),
                    event in ("pull_request", "push"),
                )

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

    def test_the_windows_proofs_run_on_the_baseline_lane_and_not_on_push(self) -> None:
        """#503: `main` could establish neither green nor red under MSVC.

        Both jobs were `if: github.event_name == 'pull_request'`, so the
        schedule/dispatch lane never DEFINED them -- and a job that is not
        defined for an event is not `skipped`, it is absent, so it appeared in
        no list `main-baseline.py` prints. The verdict read GREEN because it
        graded a set that excluded them. Measured on the deliberate baseline run
        32044993401 (`conclusion=success`): both jobs `skipped`. #1068 then
        stopped `main` compiling under MSVC and surfaced on an unrelated
        contributor's pull request rather than on `main`.

        RESOLVED per event, not grepped, for the reason `resolve_boolean`
        exists: `always()` and `github.event_name != 'push'` both admit the
        baseline lane and both are wrong, and only the `push` half separates
        them from the intended condition. The `push` assertion is not a style
        preference -- that lane's jobs cancel one another by construction (26 of
        40 runs in the window this suite measures) and 55 pushes/day times two
        `windows-2022` runners buys nothing a baseline does not already have.
        """
        for name in ("windows-msvc-cpu", "windows-msvc-vulkan"):
            block = job_block(self.text, name)
            conditions = re.findall(r"(?m)^    if: (.+)$", block)
            self.assertEqual(len(conditions), 1, f"{name} must have exactly one if:")
            for event in EVENTS:
                with self.subTest(job=name, event=event):
                    self.assertEqual(
                        resolve_boolean(conditions[0], event),
                        event != "push",
                        f"{name} must run on {event}" if event != "push"
                        else f"{name} must not run on push",
                    )

    def test_the_windows_proofs_are_covered_by_the_published_verdict(self) -> None:
        """Running is half of it; the verdict has to GRADE them.

        `baseline-summary` waits on its `needs:` list and `EXPECTED_JOBS` is
        read from it, so a job that runs on the lane but is absent from both
        would fail without moving the verdict -- the `continue-on-error` shape
        of #274 reached by omission instead of by a masked conclusion.
        """
        needs = job_block(self.text, baseline.SUMMARY_JOB)
        for name in ("windows-msvc-cpu", "windows-msvc-vulkan"):
            with self.subTest(job=name):
                self.assertIn(f"      - {name}\n", needs)
                self.assertIn(name, baseline.EXPECTED_JOBS)

    def test_a_red_windows_proof_makes_the_baseline_red(self) -> None:
        """The consequence, executed rather than asserted about the workflow.

        This is the state on the day it lands: #584 fast-fails
        `test_openai_api_server.exe` with 0xC0000409 on every run of both lanes,
        so the first baseline that can see them is RED. That is the correct
        first verdict -- it was GREEN before only because it never ran them.
        """
        jobs = [
            entry for entry in ALL_GREEN_JOBS if entry["name"] != "windows-msvc-cpu"
        ] + [job("windows-msvc-cpu", "failure")]
        item = baseline.verdict(RUN_31448896841, jobs)
        self.assertFalse(item.green)
        self.assertIn("windows-msvc-cpu", item.failing)
        self.assertEqual(item.missing, [], "it ran; it is failing, not absent")

    def test_a_windows_proof_skipped_back_off_the_lane_is_red_not_green(self) -> None:
        """The exact regression #503 is about, as an executable statement.

        Reverting the `if:` makes the API report the job `skipped`, which
        `NOT_RUN_CONCLUSIONS` deliberately reads as absent rather than as a
        pass. Before this row that absence discharged no expectation because
        there was no expectation, and the verdict printed GREEN.
        """
        jobs = [
            entry for entry in ALL_GREEN_JOBS if entry["name"] != "windows-msvc-vulkan"
        ] + [job("windows-msvc-vulkan", "skipped")]
        item = baseline.verdict(RUN_31448896841, jobs)
        self.assertFalse(item.green)
        self.assertIn("windows-msvc-vulkan", item.missing)
        self.assertNotIn("windows-msvc-vulkan", item.failing)

    def test_sanitize_cpu_stays_continue_on_error_for_the_push_and_pr_lanes(self) -> None:
        """Out of scope for this row: it is the closing step of the hardening
        row. The baseline lane gets its bindingness from baseline-summary."""
        self.assertIn("continue-on-error: true", job_block(self.text, "sanitize-cpu"))


class GroupEvictionTests(unittest.TestCase):
    """A group that never cancels has ONE remaining behaviour: it queues.

    GitHub's concurrency contract has two halves. `cancel-in-progress` is the
    half this file already resolved per event. The other half is the queue, and
    it holds exactly ONE pending run: when a third run joins a group that has
    one run in progress and one pending, GitHub cancels the pending one.

    That half discarded two of the last 39 scheduled baselines. Runs
    32140419182 and 32206456661 each returned `startedAt: null` for EVERY job
    and were cancelled at 16:46:51 and 04:49:54, the seconds their successors
    32162114781 and 32217173498 were created -- 2 out of 2. `cancel-in-progress`
    resolves to false for `schedule`, so it cancelled nothing; the shared key
    `ci-schedule-refs/heads/main-mudler/vllm.cpp` did. The suite reached queue
    depth two because run 32118587477 took 9 h 28 min with no predecessor to
    wait for, on 345 job-minutes of work, so the four-hour cron laps it.

    The invariant below is derived from `cancel-in-progress` rather than from a
    list of jobs, and it runs over every concurrency block the file declares:

      cancel-in-progress false for an event  =>  the group MUST vary per run
      cancel-in-progress true  for an event  =>  the group MUST NOT vary

    Both directions are load-bearing and neither implies the other. Dropping the
    first discards a baseline. Dropping the second gives every push its own
    group, which turns cancellation off while every substring still reads right
    -- the `${{ github.sha }}` mutation `WorkflowLaneTests` was rebuilt for.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import yaml

        cls.ci = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.blocks = concurrency_blocks(cls.ci)

    def test_the_enumeration_finds_every_block_this_suite_knows_about(self) -> None:
        """A resolver that reads no block passes every assertion below."""
        names = [name for name, _ in self.blocks]
        self.assertEqual(len(names), len(set(names)), "duplicate block name")
        self.assertIn("<workflow>", names)
        for name in GROUPED_JOBS:
            self.assertIn(name, names)
        self.assertGreaterEqual(len(names), len(GROUPED_JOBS) + 1)
        for name, block in self.blocks:
            with self.subTest(block=name):
                self.assertIn("group", block, f"{name}: concurrency without a group")
                self.assertIn(
                    "cancel-in-progress",
                    block,
                    f"{name}: a group with no cancel key defaults to FALSE, so it "
                    "queues, so it can be evicted, and no assertion here can tell",
                )

    def test_no_non_cancellable_group_can_be_evicted(self) -> None:
        """THE invariant. Red at 250db75a2 on all 11 blocks x 2 baseline events."""
        for name, block in self.blocks:
            group = str(block["group"])
            cancel = str(block["cancel-in-progress"])
            for event in EVENTS:
                if resolve_boolean(cancel, event):
                    continue
                with self.subTest(block=name, event=event):
                    self.assertTrue(
                        varies_per_run(group, event),
                        f"{name} never cancels on {event}, so it QUEUES on "
                        f"{resolve_group(group, event, RUN_A)!r}. That queue "
                        "holds one pending run and discards it when a third "
                        "arrives, which cost runs 32140419182 and 32206456661 "
                        "their whole verdict without executing a single job. "
                        "Admit the run identity into the key for this event.",
                    )

    def test_a_cancellable_group_is_shared_so_it_has_something_to_cancel(self) -> None:
        """The opposite direction, and it is not implied by the first.

        A key that varies per run cannot cancel anything. Latest-only on the
        push lane (#822) and pull request dedupe are exactly this property.
        """
        for name, block in self.blocks:
            group = str(block["group"])
            cancel = str(block["cancel-in-progress"])
            for event in EVENTS:
                if not resolve_boolean(cancel, event):
                    continue
                with self.subTest(block=name, event=event):
                    self.assertFalse(
                        varies_per_run(group, event),
                        f"{name} cancels in progress on {event} but every run "
                        "gets its own group, so it cancels nothing",
                    )

    def test_the_workflow_level_group_is_unique_per_baseline_run(self) -> None:
        """Named separately because the run level is where the loss happened.

        Both discarded runs were discarded as RUNS, before any job existed to
        carry a job-level key. A fix applied only to the job blocks would leave
        this file green and the lane broken in exactly the measured way.
        """
        block = self.ci["concurrency"]
        for event in BASELINE_EVENTS:
            with self.subTest(event=event):
                self.assertFalse(
                    resolve_boolean(str(block["cancel-in-progress"]), event)
                )
                self.assertTrue(varies_per_run(str(block["group"]), event))
        for event in ("push", "pull_request"):
            with self.subTest(event=event):
                self.assertFalse(varies_per_run(str(block["group"]), event))

    def test_the_baseline_key_is_the_contributor_key_with_the_run_admitted(self) -> None:
        """The run identity enters on the baseline events and NOWHERE else.

        Pinned as a pair of equalities against the keys resolved for `push` and
        `pull_request`, so widening the conditional to admit a third event, or
        replacing `github.ref` outright, is red here rather than only in the
        base-key equality one class up.
        """
        for name, block in self.blocks:
            group = str(block["group"])
            for event in ("push", "pull_request"):
                with self.subTest(block=name, event=event):
                    self.assertEqual(
                        resolve_group(group, event, RUN_A),
                        resolve_group(group, event, RUN_B),
                        f"{name}: the {event} key moved between two runs",
                    )
            for event in BASELINE_EVENTS:
                with self.subTest(block=name, event=event):
                    resolved = resolve_group(group, event, RUN_A)
                    self.assertIn(
                        RUN_A["github.run_id"],
                        resolved,
                        f"{name}: the {event} key varies per run without "
                        "carrying the run id, so say which token it varies on",
                    )



class VerdictJobCannotSwallowFailureTests(unittest.TestCase):
    """`baseline-summary`'s own non-zero exit must reach the run.

    Both mutations here rebuild the `sanitize-cpu` defect this row is about one
    level up: the verdict job would compute RED, print RED, exit non-zero, and
    the lane would still report green.
    """

    def setUp(self) -> None:
        self.block = job_block(workflow_text(), baseline.SUMMARY_JOB)

    def test_the_verdict_job_is_not_continue_on_error(self) -> None:
        for line in code_lines(self.block):
            self.assertNotRegex(
                line,
                r"^\s*continue-on-error\s*:",
                "continue-on-error on the verdict job makes every baseline green",
            )

    def test_the_verdict_step_never_masks_its_own_exit_status(self) -> None:
        steps = [
            step
            for step in steps_of(self.block)
            if "--emit-summary" in "\n".join(step)
        ]
        self.assertEqual(len(steps), 1, "expected exactly one --emit-summary step")
        body = step_run_body(steps[0])
        self.assertIsNotNone(body)
        for line in code_lines(body):
            for masker in ("||", "; true", "set +e", "continue-on-error"):
                with self.subTest(masker=masker):
                    self.assertNotIn(
                        masker,
                        line,
                        f"`{masker}` would discard the RED verdict's exit code",
                    )
        for line in code_lines("\n".join(steps[0])):
            self.assertNotRegex(line, r"^\s+continue-on-error\s*:")


class AgentRecordDiffRangeTests(unittest.TestCase):
    """`agent-record` must be able to be GREEN on the baseline lane.

    It is in `baseline-summary`'s `needs:`, and it carries diff-scoped checkers
    keyed on `github.event.before` -- which a `schedule` or `workflow_dispatch`
    payload does not have. Before the guard landed, replaying the step body with
    EVENT_NAME=schedule and PUSH_BASE="" died under `set -eu`:

        range=[..36fa56d0...]
        check-commit-trailers.py: error: argument --range:
            range must be exactly BASE..HEAD
        STEP_EXIT=2

    so the lane could never publish anything but RED. These tests execute the
    real step bodies rather than reading them, and they assert BOTH directions:
    no range-scoped call on the baseline lane, and the range-scoped calls still
    happen on `push` and `pull_request`. Deleting the checkers instead of
    guarding them fails the second half.
    """

    # The strict trailer walk moved to `commit-protocol-tag` (#863), which opts
    # out of the baseline lane at the job level and so has no in-step guard to
    # test here. `ConcurrencySemanticsTests` pins its placement and its base.
    RANGE_SCOPED = ("check-role-discipline.py",)
    FAKE_BASE = "1" * 40
    FAKE_HEAD = "2" * 40

    # ONLY agent-record. There are two valid ways to keep a diff-scoped checker
    # off the baseline lane, and this class tests one of them. `agent-record`
    # RUNS on that lane because most of it is tree-scoped, so its diff-scoped
    # calls must be guarded IN THE STEP. `documentation-checkpoint` and
    # `commit-protocol-tag` are diff-scoped end to end and opt out at the JOB
    # level instead, so they carry no in-step guard and replaying their bodies
    # here would fail for the wrong reason.
    #
    # The strict trailer walk moved OUT of this job to `commit-protocol-tag`
    # (#863), which is why the counts below dropped by one.
    DIFF_SCOPED_JOBS = ("agent-record",)

    def setUp(self) -> None:
        text = workflow_text()
        self.job = "\n".join(job_block(text, name) for name in self.DIFF_SCOPED_JOBS)
        self.steps = [
            step
            for name in self.DIFF_SCOPED_JOBS
            for step in steps_of(job_block(text, name))
            if step_env(step).get("PUSH_BASE") == "${{ github.event.before }}"
        ]

    def test_every_before_consumer_is_one_of_the_steps_under_test(self) -> None:
        """No other route into `github.event.before` may exist in this job.

        Counted over CODE only -- the guards' own comments name the variable
        while explaining why it is empty on this lane, and a comment consumes
        nothing.
        """
        # Role discipline is the only diff-scoped call left in this job.
        self.assertEqual(len(self.steps), 1)
        job_code = "\n".join(code_lines(self.job))
        accounted = "\n".join(
            "\n".join(code_lines("\n".join(step))) for step in self.steps
        )
        self.assertEqual(
            job_code.count("github.event.before"),
            accounted.count("github.event.before"),
            "an unaccounted github.event.before consumer in a diff-scoped job",
        )
        self.assertEqual(accounted.count("github.event.before"), 1)

    def test_the_baseline_lane_invokes_no_range_scoped_checker_and_exits_zero(self) -> None:
        for event in BASELINE_EVENTS:
            for step in self.steps:
                body = step_run_body(step)
                self.assertIsNotNone(body)
                code, invocations, output = run_shimmed(
                    body,
                    {
                        "EVENT_NAME": event,
                        "PR_BASE": "",
                        "PR_HEAD": "",
                        "PUSH_BASE": "",
                        "PUSH_HEAD": self.FAKE_HEAD,
                    },
                )
                with self.subTest(event=event, step=step[0].strip()):
                    self.assertEqual(code, 0, output)
                    for argv in invocations:
                        for checker in self.RANGE_SCOPED:
                            self.assertNotIn(
                                checker,
                                " ".join(argv),
                                "a diff-scoped checker ran with no diff range",
                            )
                        self.assertNotIn(f"..{self.FAKE_HEAD}", argv)
                        self.assertNotIn("", argv, "an empty argument means an empty base")

    def test_push_and_pull_request_still_get_the_full_range_scoped_checks(self) -> None:
        """The anti-overcorrection half: guarding is not deleting."""
        cases = {
            "push": {
                "EVENT_NAME": "push",
                "PR_BASE": "",
                "PR_HEAD": "",
                "PUSH_BASE": self.FAKE_BASE,
                "PUSH_HEAD": self.FAKE_HEAD,
            },
            "pull_request": {
                "EVENT_NAME": "pull_request",
                "PR_BASE": self.FAKE_BASE,
                "PR_HEAD": self.FAKE_HEAD,
                "PUSH_BASE": "",
                "PUSH_HEAD": self.FAKE_HEAD,
            },
        }
        for event, environment in cases.items():
            invoked: set[str] = set()
            for step in self.steps:
                code, invocations, output = run_shimmed(
                    step_run_body(step) or "", environment
                )
                self.assertEqual(code, 0, output)
                for argv in invocations:
                    joined = " ".join(argv)
                    for checker in self.RANGE_SCOPED:
                        if checker not in joined:
                            continue
                        # `check-commit-trailers.py --message-file` validates the
                        # pull request BODY, which under `PR_BODY` becomes the
                        # landed commit message (#848). That invocation is not
                        # diff-scoped and has no range to carry. It does not
                        # count as the range-scoped run this case demands, so
                        # `invoked` stays untouched and the assertion below still
                        # requires the real range walk to have happened.
                        if "--message-file" in joined:
                            continue
                        invoked.add(checker)
                        self.assertIn(
                            f"{self.FAKE_BASE}..{self.FAKE_HEAD}"
                            if checker == "check-commit-trailers.py"
                            else self.FAKE_BASE,
                            joined,
                            f"{checker} got the wrong range on {event}",
                        )
            with self.subTest(event=event):
                self.assertEqual(invoked, set(self.RANGE_SCOPED))


class SuiteRegistrationTests(unittest.TestCase):
    """This file guards every claim in the row; nothing was running it.

    `grep -rn test_main_baseline .` returned exactly one hit -- this file -- so
    all 24 tests above ran on no machine, in no lane, ever. That is the same
    class of defect as an unregistered CTest target, one layer up.
    `scripts/check-test-registration.py` cannot see it: its `REQUIRED_TESTS` is
    a fixed set naming one C++ target. Issue #408 tracks whether a checker
    should cover the class; this pins the instance.
    """

    NAME = "test_main_baseline"

    def test_registered_in_the_preflight_suite_array(self) -> None:
        text = PREFLIGHT.read_text(encoding="utf-8")
        block = re.search(r"(?ms)^SUITES=\(\n(.*?)^\)$", text)
        self.assertIsNotNone(block, "preflight SUITES array not found")
        suites = block.group(1).split()
        self.assertIn(self.NAME, suites)

    def test_registered_in_the_agent_record_ci_job(self) -> None:
        job = job_block(workflow_text(), "agent-record")
        self.assertIn(f"python3 tests/scripts/{self.NAME}.py", job)

    def test_the_agent_record_job_is_unconditional(self) -> None:
        """A CI registration behind an `if:` is not a registration.

        #865 replaced this assertion with one permitting a condition, and three
        pre-existing checkers went RED on `main` (#873): `check-release-binary-
        contract.py` and `check-test-registration.py` credit a checker to CI
        only through `_unconditional_ci_run_blocks`, which drops every job
        carrying an `if:` at all. So this is not a style pin -- ANY condition
        here, including the closed-pull-request one, un-registers every checker
        this job owns.

        #822's closed-PR skip is still enforced, by `needs:`. `last-gated-
        commit` excludes the closed action, and a skipped dependency skips this
        job with it, which `ConcurrencySemanticsTests.test_a_closed_pull_
        request_executes_no_gate` asserts. `always()` would defeat that, so its
        absence is asserted here too.
        """
        job = job_block(workflow_text(), "agent-record")
        self.assertNotRegex(job, r"(?m)^    if:")
        self.assertRegex(job, r"(?m)^    needs: \[last-gated-commit\]$")


class ConcurrencySemanticsTests(unittest.TestCase):
    """#822 lets a superseded run be cancelled. #863 is why that was unsafe.

    A diff-scoped gate walks `base..head` once. If its run is cancelled, no
    later run re-covers that range, because the next run's `before` is this
    run's `sha`. The strict trailer walk sat in `agent-record`, which carries a
    cancellable group keyed on `github.ref` -- constant for every push to main.
    Measured on run 31851003245: `agent-record` cancelled, the commit fails the
    strict walk, CI reported nothing.

    Cancellation is safe only while BOTH hold: the diff-scoped gates carry no
    group, and their base is the last SUCCESSFULLY gated commit rather than the
    previous push. Each is asserted here, because reverting either one alone
    silently reopens the hole.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import yaml
        cls.ci = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        cls.ci_text = WORKFLOW.read_text(encoding="utf-8")
        containers = ROOT / ".github/workflows/containers.yml"
        cls.containers = yaml.safe_load(containers.read_text(encoding="utf-8"))

    def owning_job(self, needle: str) -> str:
        for name, job in self.ci["jobs"].items():
            for step in job.get("steps", []) or []:
                if needle in str(step.get("run", "")):
                    return name
        raise AssertionError(f"no job runs {needle!r}")

    def test_the_strict_trailer_walk_lives_in_a_group_free_job(self) -> None:
        owner = self.owning_job("check-commit-trailers.py --range")
        self.assertIsNone(
            self.ci["jobs"][owner].get("concurrency"),
            f"{owner} walks a diff range and carries a concurrency group; a "
            "cancelled run would skip that range forever (#863)",
        )

    def test_every_diff_scoped_step_bases_on_its_own_jobs_anchor(self) -> None:
        """THE invariant, and it is not group-freeness.

        A workflow-level `cancel-in-progress` cancels every job in the run,
        including jobs that carry no group of their own. So moving a gate to a
        group-free job protects nothing once the push lane is latest-only. What
        protects it is the base: `github.event.before` is the previous push
        whether or not it was gated, and the anchor is not.

        RE-PINNED, not relaxed (#1773). The assertion used to name `LAST_GREEN`,
        the one shared string `last-gated-commit` resolved from a RUN-level
        `status=success`. That value was wrong in both directions (#274) and it
        froze whenever a gate was red, so the range widened on every push. It
        also could not be resolved for three jobs at once: they have three
        different cancellation profiles, and an `if: always()` consumer outlives
        its own resolver, which is how run 32625264281 passed this gate over a
        single push. The demand here is now STRICTER -- each such step consumes
        an anchor its OWN job resolved -- and
        `AnchorStepTests.test_each_diff_scoped_job_resolves_its_own_anchor`
        asserts the resolution names that job.
        """
        for name, job in self.ci["jobs"].items():
            for step in job.get("steps") or []:
                env = {k: str(v) for k, v in (step.get("env") or {}).items()}
                if "github.event.before" not in " ".join(env.values()):
                    continue
                with self.subTest(job=name, step=step.get("name")):
                    self.assertIn(
                        "GATE_ANCHOR", " ".join(env),
                        "a diff-scoped step that does not consume an anchor; a "
                        "cancelled push skips its range forever",
                    )
                    self.assertIn('GATE_ANCHOR:-', str(step.get("run", "")))
                    resolvers = [
                        s for s in job.get("steps") or []
                        if f"--gate-anchor {name}" in str(s.get("run", ""))
                    ]
                    self.assertEqual(
                        len(resolvers), 1,
                        f"{name} consumes an anchor it does not resolve for "
                        "itself; a shared anchor is the race #1773 measured",
                    )

    def test_agent_record_no_longer_walks_a_diff_range(self) -> None:
        """It keeps its cancellable group, so it must hold nothing diff-scoped."""
        body = " ".join(
            str(s.get("run", "")) for s in self.ci["jobs"]["agent-record"]["steps"]
        )
        self.assertNotIn("--range", body)
        self.assertIsNotNone(self.ci["jobs"]["agent-record"].get("concurrency"))

    def test_the_diff_scoped_base_is_this_jobs_own_anchor(self) -> None:
        """The enabling half. Without it, cancelling a push loses the range.

        `last-gated-commit` is still asserted to exist, because `agent-record`
        inherits its closed-pull-request skip through `needs:` on it (#873). It
        no longer resolves the base: see the re-pinning note above.
        """
        self.assertIn("last-gated-commit", self.ci["jobs"])
        owner = self.owning_job("check-commit-trailers.py --range")
        walk = next(
            s for s in self.ci["jobs"][owner]["steps"]
            if "check-commit-trailers.py --range" in str(s.get("run", ""))
        )
        self.assertIn("GATE_ANCHOR", str(walk.get("env")))
        self.assertTrue(
            any(
                f"--gate-anchor {owner}" in str(s.get("run", ""))
                for s in self.ci["jobs"][owner]["steps"]
            ),
            f"{owner} does not resolve its own anchor",
        )

    def test_the_guard_job_resolves_no_shared_value(self) -> None:
        """A shared base resolved elsewhere is the race, not the fix."""
        guard = self.ci["jobs"]["last-gated-commit"]
        self.assertIsNone(guard.get("outputs"))
        body = " ".join(str(s.get("run", "")) for s in guard["steps"])
        self.assertNotIn("gh api", body)

    def test_the_base_falls_back_when_no_successful_run_is_found(self) -> None:
        """A failed or rate-limited query must degrade to today's behaviour,
        never to an empty range that passes vacuously."""
        owner = self.owning_job("check-commit-trailers.py --range")
        walk = next(
            s for s in self.ci["jobs"][owner]["steps"]
            if "check-commit-trailers.py --range" in str(s.get("run", ""))
        )
        self.assertIn('base="$PUSH_BASE"', str(walk["run"]))

    def test_the_push_lane_is_latest_only(self) -> None:
        group = self.ci["concurrency"]["group"]
        cancel = str(self.ci["concurrency"]["cancel-in-progress"])
        self.assertIn("github.ref", group)
        self.assertNotIn("github.sha", group)
        self.assertIn("github.event_name == 'push'", cancel)

    def test_the_baseline_lane_stays_non_cancellable(self) -> None:
        """#274's schedule lane exists to answer 'is main green'. A cancelled
        baseline answers nothing."""
        cancel = str(self.ci["concurrency"]["cancel-in-progress"])
        self.assertNotIn("schedule", cancel)
        self.assertIn("github.event_name", self.ci["concurrency"]["group"])

    # The exception, pinned so a third job cannot join it silently. The ENTIRE
    # job mapping of these two is compared for equality against a literal by
    # `scripts/check-release-workflow.py::validate_pr_ci` -- the read-only
    # native Windows PR proof schema, which is how the PR lane proves it holds
    # no release, upload, write-token or OIDC authority (#117). The schema
    # admits no extra key, so `needs:` is rejected, and it fixes the `if:`
    # string, so a closed-action clause is rejected. #865 added one anyway and
    # left that checker and `test_release_pipeline.py` RED on `main` (#873).
    # The pinned authority schema outranks a cost optimisation; the residual --
    # two Windows runners started per closed pull request -- is #874.
    UNGUARDABLE_JOBS = ("windows-msvc-cpu", "windows-msvc-vulkan")

    def test_every_unguardable_job_is_one_the_pinned_schema_owns(self) -> None:
        """The list above is an ALLOWLIST, and prose is not a ratchet.

        The exemption is justified by one mechanical fact: `validate_pr_ci`
        compares these jobs' WHOLE mapping against a literal, so neither
        `needs:` nor a closed clause can be added to them. Assert that fact
        rather than the two names, and a job can only be exempted by actually
        being in that schema. Without this, appending a name to the tuple
        exempts any job at all and every gate stays green -- measured.

        `contracts` is a local inside the checker, so this PARSES the checker
        instead of importing it. Exposing it would mean editing `scripts/`,
        and the whole subject of this row is that a test may not reshape the
        thing it pins in order to pin it.
        """
        import ast

        checker = ROOT / "scripts/check-release-workflow.py"
        pinned = {
            entry.elts[0].value
            for function in ast.walk(ast.parse(checker.read_text(encoding="utf-8")))
            if isinstance(function, ast.FunctionDef)
            and function.name == "validate_pr_ci"
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "contracts" for t in node.targets)
            for entry in node.value.elts
        }
        self.assertTrue(
            pinned,
            f"no `contracts` tuple found in {checker.name}::validate_pr_ci; the "
            "exemption below cannot be justified against a schema this cannot read",
        )
        self.assertLessEqual(
            set(self.UNGUARDABLE_JOBS), pinned,
            "UNGUARDABLE_JOBS may only name jobs whose whole mapping is pinned "
            f"byte-for-byte by validate_pr_ci (it pins {sorted(pinned)}); every "
            "other job can carry the closed guard and must (#874)",
        )

    # GitHub skips a job whose `needs:` dependency was skipped -- UNLESS the
    # job's own `if:` calls a status check function. `always()` is not the only
    # one: `!cancelled()`, `failure()` and `success() || failure()` resurrect a
    # dependent exactly the same way, and a `documentation-checkpoint` written
    # as `${{ !cancelled() && (...) }}` with no closed clause at all passed
    # every gate before this landed.
    #
    # So this permits a SHAPE instead of forbidding four names: a job leaning on
    # the transitive form may carry no expression call at all. Everything that
    # defeats skip propagation is a call, which makes this a superset of
    # `success` / `failure` / `cancelled` / `always` and keeps it correct if
    # GitHub ever adds a fifth. It fails CLOSED -- a harmless `contains(...)` is
    # refused too -- and the answer to that is to carry the closed clause
    # directly, which every job but the two pinned Windows proofs can do.
    _CALLS_AN_EXPRESSION_FUNCTION = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\(")

    def _skipped_on_a_closed_pull_request(self, name: str, seen: frozenset) -> bool:
        """A job executes no gate on a closed PR if it says so itself, or if it
        `needs:` a job that does -- a skipped dependency skips its dependents.

        The transitive form is only load-bearing while the dependent's own `if:`
        cannot override the skip. A status check function is exactly what
        overrides it, so a job leaning on `needs:` must call no function at all,
        and this returns False when it does.
        """
        if name in seen:
            return False
        job = self.ci["jobs"][name]
        condition = str(job.get("if", ""))
        if "closed" in condition:
            return True
        if self._CALLS_AN_EXPRESSION_FUNCTION.search(condition):
            return False
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        return any(
            self._skipped_on_a_closed_pull_request(dependency, seen | {name})
            for dependency in needs
        )

    def test_a_closed_pull_request_executes_no_gate(self) -> None:
        on = self.ci[True] if True in self.ci else self.ci["on"]
        pr = on.get("pull_request") or {}
        self.assertIn(
            "closed", (pr.get("types") or []),
            "the pull_request trigger must list `closed`, or a closed PR never "
            "enters the concurrency group and its run is never superseded",
        )
        for name in self.ci["jobs"]:
            with self.subTest(job=name):
                if name in self.UNGUARDABLE_JOBS:
                    self.assertNotIn(
                        "closed", str(self.ci["jobs"][name].get("if", "")),
                        f"{name} carries the guard, so it is no longer the "
                        "byte-exact Windows PR proof schema (#874)",
                    )
                    continue
                self.assertTrue(
                    self._skipped_on_a_closed_pull_request(name, frozenset()),
                    f"{name} runs on a closed pull request: neither its own "
                    "`if:` nor any job it needs excludes that action",
                )

    def test_no_workflow_has_a_duplicate_mapping_key(self) -> None:
        """PyYAML keeps the LAST duplicate key and says nothing. GitHub rejects
        the whole file.

        This is not hypothetical: an edit to this row added `LAST_GREEN` twice
        to one env block. Every yaml.safe_load in this suite passed, and GitHub
        refused to parse ci.yml at all -- the run carried the workflow's PATH
        instead of its name, ran zero jobs, and reported failure. No PyYAML
        based test can see that, so the check has to be structural.
        """
        import yaml

        class Strict(yaml.SafeLoader):
            pass

        def no_duplicates(loader, node, deep=False):
            seen = set()
            for key_node, _ in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in seen:
                    raise AssertionError(
                        f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                    )
                seen.add(key)
            return yaml.SafeLoader.construct_mapping(loader, node, deep)

        Strict.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, no_duplicates
        )
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            with self.subTest(workflow=path.name):
                yaml.load(path.read_text(encoding="utf-8"), Loader=Strict)

    def test_containers_has_a_concurrency_policy(self) -> None:
        """#822 names it the largest source of queued duplicates, and it had
        none at all."""
        self.assertIsNotNone(self.containers.get("concurrency"))

    def test_a_container_tag_run_is_never_cancelled(self) -> None:
        """publish/manifest/attest/promote push by digest. A cancelled publish
        can leave a manifest half-joined."""
        cancel = str(self.containers["concurrency"]["cancel-in-progress"])
        self.assertIn("refs/tags/", cancel)


# --------------------------------------------------------------------------
# GATE-ANCHOR-PER-JOB (#1773). The base a diff-scoped gate walks from.
# --------------------------------------------------------------------------


def push_run(run_id: int, sha: str, conclusion: str | None, created_at: str) -> dict:
    """One entry of the workflow-runs payload for the push lane."""

    return {
        "id": run_id,
        "head_sha": sha,
        "conclusion": conclusion,
        "status": "completed" if conclusion else "in_progress",
        "created_at": created_at,
    }


def run_level_anchor(runs: list[dict]) -> str:
    """The rule this row replaces, modelled exactly.

    `last-gated-commit` asked the API for
    `...?branch=main&event=push&status=success&per_page=1` and took
    `.workflow_runs[0].head_sha`. `status=success` filters on the RUN's own
    conclusion, so this is "the newest push run whose conclusion is success",
    and nothing else. Modelled here rather than queried so the cycle can be
    constructed offline.
    """

    for run in runs:
        if run.get("conclusion") == "success":
            return run.get("head_sha", "")
    return ""


class GateAnchorTests(unittest.TestCase):
    """`gate_anchor` reads PER-JOB conclusions, like everything else here.

    Run 32625264281 is the shape that makes this necessary and it is not
    hypothetical: its own conclusion is `cancelled` while
    `documentation-checkpoint` inside it concluded `success`. A run-level query
    cannot see that job, so the anchor it returns was eleven days and 484
    commits older than the newest commit the gate had actually cleared.
    """

    def fetcher(self, mapping):
        return lambda run_id: mapping[run_id]

    def test_a_cancelled_run_whose_job_concluded_is_the_anchor(self) -> None:
        runs = [
            push_run(3, "c" * 40, "cancelled", "2026-08-23T07:39:25Z"),
            push_run(2, "b" * 40, "cancelled", "2026-08-23T07:19:38Z"),
            push_run(1, "a" * 40, "success", "2026-08-12T23:53:24Z"),
        ]
        jobs = {
            3: [job("build-test-cpu", "cancelled")],
            2: [job("documentation-checkpoint", "success")],
            1: [job("documentation-checkpoint", "success")],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "b" * 40)
        self.assertEqual(anchor.source, "verdict")

    def test_a_job_that_concluded_FAILURE_is_still_the_anchor(self) -> None:
        """The change that breaks the cycle, asserted on its own.

        A commit on `main` is immutable, so a violation can never be repaired by
        a later push. Anchoring on `success` therefore turns one violation into
        a permanent block, and the gate stops being able to report anything
        about new commits. A verdict is a verdict.
        """
        runs = [
            push_run(2, "b" * 40, "failure", "2026-08-23T07:19:38Z"),
            push_run(1, "a" * 40, "success", "2026-08-12T23:53:24Z"),
        ]
        jobs = {
            2: [job("documentation-checkpoint", "failure")],
            1: [job("documentation-checkpoint", "success")],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "b" * 40)

    def test_cancelled_skipped_absent_and_pending_jobs_do_not_anchor(self) -> None:
        runs = [
            push_run(5, "e" * 40, None, "2026-08-23T08:49:48Z"),
            push_run(4, "d" * 40, "cancelled", "2026-08-23T08:15:39Z"),
            push_run(3, "c" * 40, "cancelled", "2026-08-23T07:46:52Z"),
            push_run(2, "b" * 40, "cancelled", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "failure", "2026-08-23T07:19:38Z"),
        ]
        jobs = {
            5: [job("documentation-checkpoint", None)],       # this run, in flight
            4: [job("documentation-checkpoint", "cancelled")],
            3: [job("documentation-checkpoint", "skipped")],
            2: [job("build-test-cpu", "cancelled")],           # absent entirely
            1: [job("documentation-checkpoint", "failure")],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "a" * 40)

    def test_a_run_never_anchors_itself(self) -> None:
        """The current run's own job is `in_progress`, so its conclusion is
        `null`. If that qualified, every gate would walk an empty range and pass
        vacuously forever -- the loudest possible version of this defect.

        THIS TEST USED TO ASSERT THE HARM ITS OWN DOCSTRING NAMES. It checked
        that the VERDICT path refused the in-flight run and then asserted the
        floor returned `a * 40` -- the head being pushed. `base..head` with both
        ends at that commit is the empty range, so the gate passed over nothing,
        the step concluded on it, and the conclusion advanced the anchor.
        Refusing to anchor on the run and then flooring onto it is the same
        answer by a different route. Found by the second review of #1776.

        With one run there is no floor to name, so the honest answer is the
        clean absence: `main` exits 1 and `ci.yml` uses `$PUSH_BASE`. Reachable
        on a fork's first push; on `mudler/vllm.cpp`, whose window is always
        full, it is latent.
        """
        runs = [push_run(1, "a" * 40, None, "2026-08-23T08:49:48Z")]
        jobs = {1: [job("documentation-checkpoint", None)]}
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertNotEqual(anchor.source, "verdict")
        self.assertEqual(anchor.source, "none")
        self.assertEqual(anchor.sha, "")
        self.assertIsNone(anchor.degraded)

    def test_the_floor_is_NEVER_the_head_being_pushed(self) -> None:
        """The property behind the case above, over every window size.

        An anchor equal to the head makes `base..head` empty. Whatever else a
        degraded window does, it may not produce a gate that reports success
        over no commits AND concludes, because the conclusion is what advances
        the anchor.
        """
        for count in range(1, 6):
            for window in (1, 2, 3, 20):
                with self.subTest(runs=count, window=window):
                    runs = [
                        push_run(i, chr(ord("a") + i) * 40, None,
                                 f"2026-08-23T0{9 - i}:00:00Z")
                        for i in range(count)
                    ]
                    anchor = baseline.resolve_gate_anchor(
                        runs, "documentation-checkpoint", (GATE_STEP,),
                        lambda run_id: [job("documentation-checkpoint", None)],
                        window=window,
                    )
                    self.assertNotEqual(
                        anchor.sha, runs[0]["head_sha"],
                        "the anchor is the head being pushed, so the range is empty",
                    )

    def test_a_SHORT_history_floors_on_the_oldest_run_and_EXCLUDES_it(self) -> None:
        """The bound, stated as a property rather than as a claim.

        `window + 1` runs put a run PAST the candidates in hand, and its head is
        a base that covers every candidate. A branch with fewer push runs than
        the window has no such run: the oldest available run is itself a
        candidate, its head becomes the base, and `base..head` therefore does
        NOT contain that commit. It cannot be repaired from this payload -- the
        parent is what would be needed and a `workflow_run` object carries only
        `head_sha`.

        Asserted here so the residual cannot quietly grow or quietly be denied.
        """
        runs = [
            push_run(3, "c" * 40, None, "2026-08-23T09:00:00Z"),
            push_run(2, "b" * 40, "failure", "2026-08-23T08:00:00Z"),
            push_run(1, "a" * 40, "failure", "2026-08-23T07:00:00Z"),
        ]
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,),
            lambda run_id: [job("documentation-checkpoint", None)],
            window=20,
        )
        self.assertEqual(anchor.source, "floor")
        # The OLDEST run's own head, so run 1's commit is outside the range it
        # bases. That is the residual, and it is what a longer branch does not
        # have: with more than `window` runs the floor comes from `runs[window]`,
        # a run past every candidate.
        self.assertEqual(anchor.sha, "a" * 40)

    def test_a_matrix_job_anchors_only_when_every_lane_concluded(self) -> None:
        runs = [
            push_run(2, "b" * 40, "cancelled", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "cancelled", "2026-08-23T07:19:38Z"),
        ]
        jobs = {
            2: [
                job("sanitize-cpu (address,undefined)", "failure"),
                job("sanitize-cpu (thread)", "cancelled"),
            ],
            1: [
                job("sanitize-cpu (address,undefined)", "failure"),
                job("sanitize-cpu (thread)", "success"),
            ],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "sanitize-cpu", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "a" * 40)

    def test_no_qualifying_run_returns_the_WINDOW_FLOOR(self) -> None:
        """The floor, and what it actually bounds.

        It bounds the RANGE: a range never widens past the window however long
        `main` has been red. It does NOT bound the loss, and the earlier claim
        that "degrading toward more coverage is the only degradation a coverage
        gate may have" was false of this code -- past the window, commits roll
        off the back permanently. `test_past_the_window_commits_roll_off` states
        that trade as an executable fact rather than denying it.
        """
        runs = [
            push_run(3, "c" * 40, "cancelled", "2026-08-23T07:46:52Z"),
            push_run(2, "b" * 40, "cancelled", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "cancelled", "2026-08-23T07:19:38Z"),
        ]
        jobs = {n: [job("documentation-checkpoint", "cancelled")] for n in (1, 2, 3)}
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "a" * 40)
        self.assertEqual(anchor.source, "floor")

    def test_a_SKIPPED_gate_step_in_a_CONCLUDED_job_does_not_anchor(self) -> None:
        """F1, as one assertion. The live shape of runs 32601353990..32623377380.

        GitHub concludes a job `failure` as soon as any step fails and marks
        every remaining step `skipped`. The job says `failure` either way, so
        the job-level question cannot tell a gate that REFUSED the range from a
        gate that never ran. Run 2 here is the second, and anchoring on it
        would walk past every commit run 2 introduced.
        """
        runs = [
            push_run(2, "b" * 40, "failure", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "success", "2026-08-23T07:19:38Z"),
        ]
        jobs = {
            2: [job("commit-protocol-tag", "failure",
                    [("Some earlier step", "failure"), (GATE_STEP, "skipped")])],
            1: [job("commit-protocol-tag", "success",
                    [("Some earlier step", "success"), (GATE_STEP, "success")])],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "commit-protocol-tag", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "a" * 40)
        self.assertEqual(anchor.source, "verdict")

    def test_a_gate_step_ABSENT_from_the_payload_does_not_anchor(self) -> None:
        """A renamed, unreached or truncated step is not a verdict.

        This is the mutation route a future rename would take: change the step
        name in `ci.yml` and leave the `--gate-step` value behind, and a rule
        that treated "not found" as "fine" would anchor on every run forever.
        """
        runs = [push_run(1, "a" * 40, "failure", "2026-08-23T07:19:38Z")]
        jobs = {1: [job("commit-protocol-tag", "failure", [("Other", "failure")])]}
        anchor = baseline.resolve_gate_anchor(
            runs, "commit-protocol-tag", (GATE_STEP,), self.fetcher(jobs)
        )
        self.assertNotEqual(anchor.source, "verdict")

    def test_every_named_step_must_conclude_not_just_one(self) -> None:
        """`documentation-checkpoint` runs TWO diff-scoped gates, and half a
        job's gates concluding is half a verdict."""
        runs = [
            push_run(2, "b" * 40, "failure", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "failure", "2026-08-23T07:19:38Z"),
        ]
        jobs = {
            2: [job("documentation-checkpoint", "failure",
                    [("first", "failure"), ("second", "skipped")])],
            1: [job("documentation-checkpoint", "failure",
                    [("first", "failure"), ("second", "failure")])],
        }
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", ("first", "second"), self.fetcher(jobs)
        )
        self.assertEqual(anchor.sha, "a" * 40)

    def test_an_anchor_with_no_named_step_is_REFUSED(self) -> None:
        """Not defaulted to the job. A job-level anchor is #863 one level down,
        so the flag-less form has to be unavailable rather than lenient."""
        with self.assertRaises(ValueError):
            baseline.steps_concluded([job("x", "success")], "x", ())

    def test_the_floor_covers_the_oldest_CANDIDATES_own_head(self) -> None:
        """F2's off-by-one. The floor is the run PAST the candidate window.

        With the floor set to the oldest CANDIDATE's own head, that head sat on
        the wrong side of `base..head` and its commit was excluded from the very
        range the floor exists to guarantee. Reading one extra run costs nothing
        -- it is the same single runs call -- and puts every candidate inside.
        """
        runs = [
            push_run(5, "e" * 40, "cancelled", "2026-08-23T07:52:11Z"),
            push_run(4, "d" * 40, "cancelled", "2026-08-23T07:49:52Z"),
            push_run(3, "c" * 40, "cancelled", "2026-08-23T07:46:52Z"),
            push_run(2, "b" * 40, "cancelled", "2026-08-23T07:39:25Z"),
            push_run(1, "a" * 40, "cancelled", "2026-08-23T07:19:38Z"),
        ]
        jobs = {n: [job("documentation-checkpoint", "cancelled")]
                for n in (1, 2, 3, 4, 5)}
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs),
            window=3,
        )
        self.assertEqual(anchor.source, "floor")
        # The candidates are e, d, c. The floor is runs[window], `b`, so `c`'s
        # own head is INSIDE `b..HEAD`. There are deliberately MORE runs than
        # `window + 1` here: with exactly `window + 1`, `runs[window]` and
        # `runs[-1]` are the same entry and the assertion cannot tell the fixed
        # code from the off-by-one.
        self.assertEqual(anchor.sha, "b" * 40)

    def test_past_the_window_commits_roll_off_PERMANENTLY(self) -> None:
        """F2, stated rather than denied.

        The window is a cost bound: walking further is an unbounded number of
        API calls per push, on three jobs. The price is that a job which
        returns no verdict for more than `window` consecutive pushes loses the
        commits older than the window, and no later push ever sees them again.
        Asserted here so the trade is a measured property of the code and not a
        sentence in a spec that could quietly stop being true.
        """
        window = 3
        heads = ["a", "b", "c", "d", "e", "f"]
        runs = [
            push_run(n + 1, heads[n] * 40, "cancelled", f"2026-08-23T0{n}:00:00Z")
            for n in range(len(heads))
        ]
        runs.reverse()
        jobs = {n: [job("documentation-checkpoint", "cancelled")]
                for n in range(1, len(heads) + 1)}
        anchor = baseline.resolve_gate_anchor(
            runs, "documentation-checkpoint", (GATE_STEP,), self.fetcher(jobs),
            window=window,
        )
        # Six non-qualifying pushes, a window of three: the floor is runs[3],
        # which is `c`. `a` and `b` are older than the base of every range this
        # job will ever walk again.
        self.assertEqual(anchor.sha, "c" * 40)
        self.assertEqual(anchor.source, "floor")

    def test_gate_anchor_reads_one_run_PAST_the_window(self) -> None:
        """Where the off-by-one actually lives.

        `resolve_gate_anchor` can only put the oldest candidate inside the floor
        range if it was GIVEN a run older than the window. Fetching exactly
        `window` runs makes `runs[window]` and `runs[-1]` the same entry, and
        the floor silently becomes the oldest candidate again -- the defect,
        restored, with the slicing still looking correct.
        """
        seen = {}
        original_push_runs = baseline.push_runs
        original_jobs_for = baseline.jobs_for
        original_repository = baseline.repository
        baseline.repository = lambda: "mudler/vllm.cpp"
        baseline.jobs_for = lambda repo, run_id: ([], None)

        def recording(repo, limit, branch):
            seen["limit"] = limit
            return [], None

        baseline.push_runs = recording
        try:
            baseline.gate_anchor("documentation-checkpoint", (GATE_STEP,), "main", 20)
        finally:
            baseline.push_runs = original_push_runs
            baseline.jobs_for = original_jobs_for
            baseline.repository = original_repository
        self.assertEqual(seen["limit"], 21)

    def test_an_empty_window_resolves_to_nothing(self) -> None:
        anchor = baseline.resolve_gate_anchor(
            [], "documentation-checkpoint", (GATE_STEP,), None
        )
        self.assertEqual(anchor.sha, "")
        self.assertEqual(anchor.source, "none")

    def test_the_workflow_no_longer_queries_a_RUN_level_success(self) -> None:
        """The defect itself, pinned out of the file.

        `status=success` on the runs endpoint filters on the run's conclusion.
        `sanitize-cpu` is `continue-on-error`, so that field can read success
        over a red job; and a run that was cancelled reads `cancelled` over a
        job that ran to completion. It answers a different question in both
        directions.
        """
        # `code_lines` drops whole-line YAML comments: the job comments RECORD
        # the retired query, and a substring test that could not tell a comment
        # from a call would forbid documenting what was fixed.
        self.assertNotIn(
            "status=success", "\n".join(code_lines(workflow_text()))
        )


class PushRunsPayloadTests(unittest.TestCase):
    """`push_runs` against every shape the forge hands back, through `main`.

    NOTHING EXECUTED `push_runs` BEFORE #1776's second review. Every anchor test
    replaced it with a stand-in, and `gh_api` appeared in no test at all -- the
    M9/M10 shape one layer down, with the stand-in moved from the shell shim to
    the module function. So the chain is driven here end to end: real `main`,
    real `gate_anchor`, real `push_runs`, real `jobs_for`, and only `gh_api`
    faked, because it is the one function that opens a socket.

    What it holds is the rc SPLIT `ci.yml` acts on, and the split is the whole
    point. rc 3 is REMOTE_UNVERIFIED and makes the gate step SKIP. rc 1 is a
    clean absence and makes it fall back to `$PUSH_BASE`. Three unreadable
    payloads used to exit 1: a list, a null, and GitHub's `{"message": "Not
    Found"}` error object. Each of them made a diff gate walk one push, pass,
    conclude, and advance its own anchor past the whole span the narrowing
    dropped.
    """

    JOB = "documentation-checkpoint"
    STEP = "Every feature checkpoint updates STATUS, BENCHMARKS and NOW"

    def anchor_exit(self, reply) -> tuple[int, str, str]:
        """Run the real `--gate-anchor` CLI with `gh_api` faked.

        `reply` is either a fixed `(payload, degraded)` pair or a callable of
        the request path, which is how the readable case answers the runs
        endpoint and the jobs endpoint differently.
        """

        original_gh_api = baseline.gh_api
        original_repository = baseline.repository
        seen: list[str] = []

        def fake(path: str):
            seen.append(path)
            return reply(path) if callable(reply) else reply

        baseline.gh_api = fake
        baseline.repository = lambda: "mudler/vllm.cpp"
        out, err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                code = baseline.main(
                    ["--gate-anchor", self.JOB, "--gate-step", self.STEP]
                )
        finally:
            baseline.gh_api = original_gh_api
            baseline.repository = original_repository
        self.assertTrue(seen, "the anchor never asked the forge anything")
        self.assertIn("event=push", seen[0])
        self.assertNotIn("status=", seen[0])
        return code, out.getvalue(), err.getvalue()

    def test_a_degraded_gh_call_exits_3(self) -> None:
        """The case that was already right, pinned so the fix cannot regress it."""
        code, stdout, stderr = self.anchor_exit(
            (None, "REMOTE_UNVERIFIED: gh api failed")
        )
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout.strip(), "")
        self.assertIn("REMOTE_UNVERIFIED", stderr)

    def test_a_LIST_runs_payload_is_REMOTE_UNVERIFIED_not_an_empty_window(self) -> None:
        code, stdout, stderr = self.anchor_exit(([], None))
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout.strip(), "")
        self.assertIn("REMOTE_UNVERIFIED", stderr)

    def test_a_NULL_runs_payload_is_REMOTE_UNVERIFIED_not_an_empty_window(self) -> None:
        code, stdout, stderr = self.anchor_exit((None, None))
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout.strip(), "")
        self.assertIn("REMOTE_UNVERIFIED", stderr)

    def test_an_ERROR_OBJECT_is_REMOTE_UNVERIFIED_not_an_empty_window(self) -> None:
        """`gh api` exits 0 and prints this body for a 404 on some paths, so a
        dict is not on its own evidence that the question was answered."""
        code, stdout, stderr = self.anchor_exit(({"message": "Not Found"}, None))
        self.assertEqual(code, 3, stderr)
        self.assertEqual(stdout.strip(), "")
        self.assertIn("REMOTE_UNVERIFIED", stderr)

    def test_a_GENUINELY_empty_window_stays_a_CLEAN_absence(self) -> None:
        """rc 1 and rc 3 may not be collapsed in either direction.

        A branch whose workflow has never run on a push has no anchor and never
        will; skipping there would gate nothing forever. This is the case the
        fix must NOT convert into REMOTE_UNVERIFIED.
        """
        code, stdout, stderr = self.anchor_exit(
            ({"total_count": 0, "workflow_runs": []}, None)
        )
        self.assertEqual(code, 1, stderr)
        self.assertEqual(stdout.strip(), "")
        self.assertNotIn("REMOTE_UNVERIFIED", stderr)

    def test_a_READABLE_window_with_a_verdict_exits_0_and_prints_the_sha(self) -> None:
        """The positive control. Without it the four refusals above are also
        satisfied by a `push_runs` that refuses everything."""
        runs = {
            "total_count": 1,
            "workflow_runs": [
                {"id": 11, "head_sha": "a" * 40, "created_at": "2026-08-23T09:00:00Z"}
            ],
        }
        jobs = {
            "jobs": [
                job_with_steps(
                    self.JOB, "success", [(self.STEP, "success")]
                )
            ]
        }

        def reply(path: str):
            return (jobs, None) if "/jobs" in path else (runs, None)

        code, stdout, stderr = self.anchor_exit(reply)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stdout.strip(), "a" * 40)

    def test_the_anchor_query_goes_through_gh_api_and_nowhere_else(self) -> None:
        """The stand-in hazard, closed at its source.

        If `push_runs` ever grew its own transport, every test above would keep
        passing while measuring a function the workflow no longer calls.
        """
        original_gh_api = baseline.gh_api
        baseline.gh_api = lambda path: (_ for _ in ()).throw(
            AssertionError(f"unexpected second transport for {path!r}")
        )
        try:
            with self.assertRaises(AssertionError):
                baseline.push_runs("mudler/vllm.cpp", 21, "main")
        finally:
            baseline.gh_api = original_gh_api


class AnchorCycleConstructionTests(unittest.TestCase):
    """The feedback loop of #1773, CONSTRUCTED rather than read.

    Six pushes P1..P6. P1's run is fully green. P2 lands a commit that a
    diff-scoped gate flags, so from P2 onward no run is ever `success` again --
    which is the premise, not an assumption: the gate is what would have made it
    success. Under the run-level rule the anchor is pinned at P1 forever and
    each push walks a wider range that re-includes P2. Under the per-job rule
    the gate returns a verdict on every push, so the anchor advances and P2 is
    reported once.
    """

    PUSHES = ["p1", "p2", "p3", "p4", "p5", "p6"]

    def runs_after(self, index: int) -> list[dict]:
        """The payload as it stands when push number `index` starts, newest first.

        Every run from P2 onward concludes `failure`, because the diff gate in
        it is red. P1 is the last `success`.
        """
        entries = []
        for position in range(index):
            conclusion = "success" if position == 0 else "failure"
            entries.append(
                push_run(position + 1, self.PUSHES[position], conclusion,
                         f"2026-08-23T0{position}:00:00Z")
            )
        return list(reversed(entries))

    def jobs_after(self, index: int) -> dict:
        return {
            position + 1: [
                job("documentation-checkpoint",
                    "success" if position == 0 else "failure")
            ]
            for position in range(index)
        }

    def commits_after(self, base: str, head: str) -> list[str]:
        """`base..head` over the linear push sequence."""
        order = self.PUSHES
        start = order.index(base) + 1 if base in order else 0
        return order[start:order.index(head) + 1]

    def test_run_level_anchor_widens_across_pushes(self) -> None:
        """THE CYCLE. The anchor never moves and the range grows without bound.

        This is the rule being removed, so it passes before and after; what it
        pins is the SHAPE of the defect the replacement has to break.
        """
        widths, always_reincluded = [], []
        for index in range(1, len(self.PUSHES)):
            head = self.PUSHES[index]
            base = run_level_anchor(self.runs_after(index))
            self.assertEqual(base, "p1")
            commits = self.commits_after(base, head)
            widths.append(len(commits))
            always_reincluded.append("p2" in commits)
        self.assertEqual(widths, [1, 2, 3, 4, 5])
        self.assertTrue(all(always_reincluded))

    def test_per_job_anchor_reports_the_violation_once(self) -> None:
        """The exit condition no longer requires the thing it blocks."""
        reports = []
        for index in range(1, len(self.PUSHES)):
            head = self.PUSHES[index]
            runs = self.runs_after(index)
            jobs = self.jobs_after(index)
            anchor = baseline.resolve_gate_anchor(
                runs, "documentation-checkpoint", (GATE_STEP,),
                lambda run_id: jobs[run_id]
            )
            commits = self.commits_after(anchor.sha, head)
            self.assertEqual(len(commits), 1, f"range at {head}: {commits}")
            if "p2" in commits:
                reports.append(head)
        self.assertEqual(reports, ["p2"])

    # (job conclusion, gate-step conclusion) for P1..P6.
    #
    # P2 onward is the LIVE shape of `commit-protocol-tag` over runs
    # 32601353990..32623377380: an earlier step of the job failed, GitHub
    # concluded the JOB `failure` and marked every remaining step `skipped`,
    # and the diff-scoped gate never executed. P4 is #863's original hole, a
    # cancelled job, so both shapes are modelled in one sequence.
    STEP_PROFILE = [
        ("success", "success"),
        ("failure", "skipped"),
        ("failure", "skipped"),
        ("cancelled", "skipped"),
        ("failure", "failure"),
        ("failure", "failure"),
    ]

    def profile_runs(self, index: int) -> list[dict]:
        entries = [
            push_run(
                position + 1,
                self.PUSHES[position],
                self.STEP_PROFILE[position][0],
                f"2026-08-23T0{position}:00:00Z",
            )
            for position in range(index)
        ]
        return list(reversed(entries))

    def profile_jobs(self, index: int) -> dict:
        return {
            position + 1: [
                job_with_steps(
                    "commit-protocol-tag",
                    self.STEP_PROFILE[position][0],
                    [
                        ("Resolve this job's own diff anchor (#1773)", "success"),
                        (GATE_STEP, self.STEP_PROFILE[position][1]),
                    ],
                )
            ]
            for position in range(index)
        }

    @staticmethod
    def anchor(runs, job_name, step_names, fetch):
        """The rule under test, called through one adapter.

        The adapter exists so the red-before and the green-after assert BYTE
        IDENTICAL properties: what changes between them is whether the anchor
        is allowed to see `step_names`, which is the whole fix.
        """

        return baseline.resolve_gate_anchor(runs, job_name, step_names, fetch)

    def test_no_commit_is_ever_skipped(self) -> None:
        """Every commit gets a verdict from a run that actually RAN the gate.

        The union of the ranges is NOT the property. A job-level anchor keeps
        that union whole while the gate step is skipped on every push, because
        `failure` advances the anchor whether or not the gate executed. The
        property is narrower and is the one #863 is about: for every commit
        there is at least one push whose range contains it AND whose gate step
        returned a verdict. A range walked by a step that never ran measures
        nothing.
        """

        covered: set[str] = set()
        for index in range(1, len(self.PUSHES)):
            head = self.PUSHES[index]
            jobs = self.profile_jobs(index)
            anchor = self.anchor(
                self.profile_runs(index),
                "commit-protocol-tag",
                (GATE_STEP,),
                lambda run_id: jobs[run_id],
            )
            if anchor.source == "none":
                continue
            if self.STEP_PROFILE[index][1] not in {"success", "failure"}:
                # The gate did not EXECUTE on this push. Its range is not a
                # verdict about anything, so it may not count as coverage.
                continue
            covered.update(self.commits_after(anchor.sha, head))
        self.assertEqual(
            covered,
            set(self.PUSHES[1:]),
            "commits with no verdict from any run that ran the gate: "
            f"{sorted(set(self.PUSHES[1:]) - covered)}",
        )


class AnchorStepTests(unittest.TestCase):
    """The three diff-scoped jobs resolve an anchor NAMED AFTER THEMSELVES.

    One shared anchor cannot be right for three jobs with three different
    cancellation profiles: `agent-record` carries a cancellable group of its
    own, and the other two do not. It is also what made run 32625264281
    pathological -- `last-gated-commit` was cancelled while its `if: always()`
    consumers ran, so the anchor rendered EMPTY, the step fell back to
    `PUSH_BASE`, and the gate passed over a single push. The verdict on `main`
    was decided by which job won a cancellation race. Resolving inside the
    consuming job removes the race by removing the second job.
    """

    JOBS = ("agent-record", "documentation-checkpoint", "commit-protocol-tag")

    def anchor_steps(self, job_name: str) -> list[list[str]]:
        block = job_block(workflow_text(), job_name)
        return [
            step for step in steps_of(block)
            if "--gate-anchor" in "\n".join(step)
        ]

    def test_each_diff_scoped_job_resolves_its_own_anchor(self) -> None:
        for job_name in self.JOBS:
            with self.subTest(job=job_name):
                steps = self.anchor_steps(job_name)
                self.assertTrue(steps, f"{job_name} resolves no anchor of its own")
                body = "\n".join(steps[0])
                self.assertIn(f"--gate-anchor {job_name}", body)

    def test_the_anchor_step_runs_the_query_and_exports_it(self) -> None:
        block = job_block(workflow_text(), "documentation-checkpoint")
        step = next(s for s in steps_of(block) if "--gate-anchor" in "\n".join(s))
        body = step_run_body(step)
        self.assertIsNotNone(body)
        code, argv, output = run_shimmed(
            body,
            {
                "EVENT_NAME": "push",
                "BRANCH": "main",
                "GITHUB_ENV": "/dev/null",
            },
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(
            argv,
            [[
                "scripts/main-baseline.py",
                "--gate-anchor",
                "documentation-checkpoint",
                "--branch",
                "main",
                "--gate-step",
                "Every feature checkpoint updates STATUS, BENCHMARKS and NOW",
                "--gate-step",
                "Every commit in the range arrived on a task branch",
            ]],
            output,
        )

    def test_the_anchor_step_does_not_query_on_the_pull_request_lane(self) -> None:
        """A pull request has no gated `main` history to anchor on, and its own
        base is authoritative. Querying there would spend an API call to
        resolve a value nothing reads."""
        block = job_block(workflow_text(), "documentation-checkpoint")
        step = next(s for s in steps_of(block) if "--gate-anchor" in "\n".join(s))
        code, argv, output = run_shimmed(
            step_run_body(step),
            {"EVENT_NAME": "pull_request", "BRANCH": "main", "GITHUB_ENV": "/dev/null"},
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(argv, [], output)

    # ------------------------------------------------------------------
    # The STEP is the unit (#1776)
    # ------------------------------------------------------------------

    def gate_step_names(self, job_name: str) -> list[str]:
        """The step names this job's anchor query claims a verdict from."""

        body = "\n".join(self.anchor_steps(job_name)[0])
        return re.findall(r'--gate-step "([^"]+)"', body)

    def named_steps(self, job_name: str) -> dict[str, list[str]]:
        block = job_block(workflow_text(), job_name)
        found = {}
        for step in steps_of(block):
            match = re.match(r"      - name: (.*)", step[0])
            if match:
                found[match.group(1)] = step
        return found

    def test_every_named_gate_step_exists_in_its_own_job(self) -> None:
        """The rename hazard, closed. A `--gate-step` value that names nothing
        would make `steps_concluded` refuse every run and pin the anchor at the
        floor forever -- silent, and only visible as a slowly widening range."""
        for job_name in self.JOBS:
            with self.subTest(job=job_name):
                names = self.named_steps(job_name)
                wanted = self.gate_step_names(job_name)
                self.assertTrue(wanted, f"{job_name} names no gate step")
                for step_name in wanted:
                    self.assertIn(step_name, names)

    def test_every_step_that_READS_the_anchor_is_NAMED_by_it(self) -> None:
        """The population, not a sample.

        A diff-scoped step the anchor does not name is a gate whose verdict
        nothing waits for: the anchor advances on the steps it does name, and
        this one's range moves out from under it. Selecting the population by
        "reads GATE_ANCHOR" is what makes a newly added diff-scoped gate fail
        here instead of landing uncovered.
        """
        for job_name in self.JOBS:
            with self.subTest(job=job_name):
                wanted = set(self.gate_step_names(job_name))
                for name, step in self.named_steps(job_name).items():
                    body = step_run_body(step) or ""
                    if "GATE_ANCHOR" not in body or "--gate-anchor" in body:
                        continue
                    self.assertIn(
                        name, wanted,
                        f"{job_name}: step {name!r} walks the anchor's range but "
                        "no --gate-step names it, so nothing waits for its verdict",
                    )

    def guard_of(self, job_name: str, step_name: str) -> str:
        """One gate step's `if:`, from the PARSED workflow.

        Parsed rather than grepped out of the raw lines, because a guard is
        written over three physical lines and YAML folds it; reassembling it by
        hand is how a test ends up holding a shape rather than a value.
        """

        import yaml

        ci = yaml.safe_load(workflow_text())
        for step in ci["jobs"][job_name]["steps"]:
            if step.get("name") == step_name:
                condition = step.get("if")
                self.assertIsNotNone(
                    condition,
                    f"{job_name}: gate step {step_name!r} carries no `if:` at all, "
                    "so it runs on a degraded read and narrows the range",
                )
                return condition
        raise AssertionError(f"{job_name}: no step named {step_name!r}")

    # Every state a gate step's guard has to decide, and what it must decide.
    # `cancelled()` is the run's cancellation, the two outcomes are the steps
    # this one depends on, and the environment value is what the anchor step
    # exports. A gate step may run in exactly ONE of these sixteen states.
    GUARD_STATES = tuple(
        (
            {
                "cancelled()": cancelled,
                "steps.checkout.outcome": checkout,
                "steps.anchor.outcome": anchor,
                "env.GATE_ANCHOR_DEGRADED": degraded,
            },
            not cancelled
            and checkout == "success"
            and anchor == "success"
            and degraded != "true",
        )
        for cancelled in (False, True)
        for checkout in ("success", "failure")
        for anchor in ("success", "failure")
        for degraded in ("", "true")
    )

    def test_every_gate_step_SKIPS_rather_than_narrows(self) -> None:
        """The guard RESOLVED, over every state it has to decide.

        This test checked four substrings until #1776's second review, and the
        mutation that walked through it is the one the module docstring already
        names one level up: append `|| true` to all five guards. Every substring
        survives byte-for-byte, the whole conjunction becomes the constant
        `true`, and the gate step then runs on a degraded read -- `GATE_ANCHOR`
        empty, so `base` falls back to `$PUSH_BASE`, a one-push range passes,
        the step concludes `success`, and the anchor advances past the entire
        span the narrowing dropped. That is the defect this row exists to
        remove, restored in full and invisible.

        So resolve it. Sixteen states, one boolean each:

        - `cancelled()` false, checkout `success`, anchor `success`, degraded
          not `'true'` -- and ONLY then -- the gate runs.
        - anything else -- the run is being cancelled, there is no tree to walk,
          the resolver failed, or the base is UNKNOWN -- the gate SKIPS. A skip
          is not a verdict, so the anchor does not advance and the next readable
          run walks the span whole.

        `!cancelled()` is the term that closes the original F1: without it an
        earlier step's failure marks this one `skipped`, and a `skipped` step
        never returns the verdict the anchor waits for while the JOB concludes
        `failure` and a job-level rule walks straight past. It is now asserted
        by its EFFECT -- drop it and the eight `cancelled()`-true states resolve
        the wrong way.

        The evaluator raises on any context or status function `GUARD_STATES`
        does not model, so a guard that grows a new term reds here rather than
        resolving to something plausible.
        """
        for job_name in self.JOBS:
            names = self.named_steps(job_name)
            for step_name in self.gate_step_names(job_name):
                condition = self.guard_of(job_name, step_name)
                self.assertIn(step_name, names)
                for state, expected in self.GUARD_STATES:
                    with self.subTest(job=job_name, step=step_name, state=state):
                        self.assertEqual(
                            resolve_condition(condition, state),
                            expected,
                            f"{job_name}/{step_name}: guard {condition!r} resolves "
                            f"{not expected} in state {state}",
                        )

    def test_one_diff_scoped_checker_per_gate_step(self) -> None:
        """`documentation-checkpoint`'s regression, pinned shut.

        Its gate ran `check-now-current.py` and `check-role-discipline.py` in
        ONE step under `set -eu`. A `check-now-current` failure aborted before
        the arrival gate ran, and the step still concluded -- so the step's
        conclusion was not a verdict about the checker the anchor was waiting
        for. Two gates in one step is that hazard, whatever the two are.
        """
        for job_name in self.JOBS:
            names = self.named_steps(job_name)
            for step_name in self.gate_step_names(job_name):
                with self.subTest(job=job_name, step=step_name):
                    body = step_run_body(names[step_name]) or ""
                    checkers = set(re.findall(r"scripts/(check-[a-z0-9-]+\.py)", body))
                    self.assertLessEqual(len(checkers), 1, sorted(checkers))
                    if not checkers:
                        continue
                    first = re.search(r"python3 (\S+)", body)
                    self.assertIsNotNone(first)
                    self.assertEqual(
                        f"scripts/{checkers.pop()}", first.group(1),
                        f"{step_name}: a DIFFERENT python3 gate runs before this "
                        "step's own, so the step can conclude without it",
                    )

    def test_nothing_fallible_PRECEDES_the_gate_in_its_own_body(self) -> None:
        """The half of the shape above that it did not hold, and two mutations.

        Spec §4.3 item 3 claimed this test held "that nothing fallible precedes
        the checker in its body". It did not. It read the first `python3 <arg>`
        and compared it with the step's one checker, so anything that was not
        `python3` was invisible, and a step that runs no `scripts/check-*.py` at
        all left the population through `if not checkers: continue`. Both halves
        were confirmed by mutation on #1776:

        - insert `git fetch -q origin +refs/heads/main:refs/remotes/origin/main`
          before `python3 scripts/check-now-current.py` -- 99 tests, OK;
        - insert the same line into `Every new commit carries
          FOLLOWING_AGENTS_PROTOCOL`, whose whole body IS an inline shell gate
          and which named no checker -- 99 tests, OK.

        Under `set -eu` the inserted command fails, the step aborts, GitHub
        concludes it `failure`, and `steps_concluded` reads that as a verdict --
        the anchor then advances over a range the gate never looked at. That is
        the `documentation-checkpoint` defect again, inside a single step.

        The property, held rather than described: a gate step's body runs the
        RANGE PRELUDE and then its GATE, and nothing else in between. The
        prelude may only use `set`, `[`, `echo` and `:` -- a shell option, a
        conditional whose exit status its `if` or `||` consumes, and a message.
        None can abort the step. The first command that is NOT one of those is
        the gate, and it must CONSUME the range: reference `$base`. A command
        that runs before the gate and does not touch the range is by
        construction not part of computing it, and both mutations are exactly
        that shape.

        The population is every gate step, INCLUDING the one whose gate is
        inline shell rather than a checker. `commit-protocol-tag`'s
        `FOLLOWING_AGENTS_PROTOCOL` walk resolves its gate to
        `git cat-file -e "${base}^{commit}"`, and `commit-protocol-tag`'s strict
        trailer step to the same probe -- both consume the range, both pass, and
        neither is exempt any more.

        RESIDUAL, stated rather than hidden: a command inserted before the gate
        that itself references `$base` is not distinguishable from the gate by
        this test. Command substitution is refused in the prelude so a fallible
        call cannot hide inside an allowed one, and one gate per step is
        asserted above; a range-consuming impostor is what remains.
        """
        for job_name in self.JOBS:
            names = self.named_steps(job_name)
            for step_name in self.gate_step_names(job_name):
                with self.subTest(job=job_name, step=step_name):
                    body = step_run_body(names[step_name]) or ""
                    self.assertTrue(body, f"{step_name}: gate step has no run body")
                    gate = None
                    for head, statement in commands_in(body):
                        if head not in RANGE_PRELUDE_COMMANDS:
                            gate = (head, statement)
                            break
                        self.assertNotIn(
                            "$(", statement,
                            f"{job_name}/{step_name}: command substitution in the "
                            "range prelude hides a fallible call inside an "
                            "allowed one",
                        )
                        self.assertNotIn(
                            "`", statement,
                            f"{job_name}/{step_name}: backtick substitution in the "
                            "range prelude hides a fallible call inside an "
                            "allowed one",
                        )
                    self.assertIsNotNone(
                        gate, f"{job_name}/{step_name}: no gate runs in this step"
                    )
                    head, statement = gate
                    self.assertRegex(
                        statement, r"\$\{?base\b",
                        f"{job_name}/{step_name}: {head!r} runs before the gate and "
                        "does not consume the range, so this step can abort under "
                        "`set -eu` and CONCLUDE without its gate ever executing -- "
                        "and the anchor then advances over commits nothing checked",
                    )

    def test_a_DEGRADED_query_skips_the_gate_instead_of_narrowing_it(self) -> None:
        """F3, executed. rc 3 is REMOTE_UNVERIFIED and it is not a pass.

        `|| true` used to swallow it: the anchor rendered empty, the gate fell
        back to `PUSH_BASE`, passed over one push, and its success advanced this
        step's own anchor past every commit the narrowing dropped. The anchor
        step now exports `GATE_ANCHOR_DEGRADED=true`, which the gate steps'
        `if:` turns into a SKIP.
        """
        block = job_block(workflow_text(), "documentation-checkpoint")
        step = next(s for s in steps_of(block) if "--gate-anchor" in "\n".join(s))
        with tempfile.TemporaryDirectory(prefix="vllm-gate-env-") as temporary:
            github_env = Path(temporary) / "env"
            github_env.write_text("", encoding="utf-8")
            code, _, output = run_shimmed(
                step_run_body(step),
                {"EVENT_NAME": "push", "BRANCH": "main",
                 "GITHUB_ENV": str(github_env)},
                python_exit=3,
            )
            exported = github_env.read_text(encoding="utf-8")
        self.assertEqual(code, 0, output)
        self.assertIn("GATE_ANCHOR=\n", exported)
        self.assertIn("GATE_ANCHOR_DEGRADED=true", exported)
        self.assertIn("REMOTE_UNVERIFIED", output)

    def test_a_CLEAN_absence_still_falls_back_to_push_base(self) -> None:
        """rc 1 is not rc 3, and collapsing them would be its own defect.

        A branch with no gated history has no anchor and never will; skipping
        there would gate nothing forever. `PUSH_BASE` is the honest base in that
        case, and the run is not degraded.
        """
        block = job_block(workflow_text(), "documentation-checkpoint")
        step = next(s for s in steps_of(block) if "--gate-anchor" in "\n".join(s))
        with tempfile.TemporaryDirectory(prefix="vllm-gate-env-") as temporary:
            github_env = Path(temporary) / "env"
            github_env.write_text("", encoding="utf-8")
            code, _, output = run_shimmed(
                step_run_body(step),
                {"EVENT_NAME": "push", "BRANCH": "main",
                 "GITHUB_ENV": str(github_env)},
                python_exit=1,
            )
            exported = github_env.read_text(encoding="utf-8")
        self.assertEqual(code, 0, output)
        self.assertIn("GATE_ANCHOR=\n", exported)
        self.assertIn("GATE_ANCHOR_DEGRADED=\n", exported)

    def test_every_anchor_job_may_read_the_actions_api(self) -> None:
        import yaml
        ci = yaml.safe_load(workflow_text())
        for job_name in self.JOBS:
            with self.subTest(job=job_name):
                permissions = ci["jobs"][job_name].get("permissions")
                self.assertIsNotNone(
                    permissions,
                    f"{job_name} queries the Actions API and declares no scope",
                )
                self.assertEqual(permissions.get("actions"), "read")

    def test_the_gate_still_falls_back_when_the_query_returns_nothing(self) -> None:
        """A rate-limited or failed query degrades to today's behaviour, never
        to an empty range that passes vacuously."""
        block = job_block(workflow_text(), "documentation-checkpoint")
        step = next(
            s for s in steps_of(block) if "check-now-current.py" in "\n".join(s)
        )
        self.assertIn('base="$PUSH_BASE"', step_run_body(step))


if __name__ == "__main__":
    unittest.main(verbosity=2)
