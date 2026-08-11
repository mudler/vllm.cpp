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
from contextlib import redirect_stdout
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
        self.assertEqual(len(baseline.EXPECTED_JOBS), 9)

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


def run_shimmed(body: str, environment: dict[str, str]) -> tuple[int, list[list[str]], str]:
    """Execute a step body with `python3` replaced by an argv recorder.

    No checker actually runs; what is under test is the SHELL logic that decides
    which checkers get invoked, and with which range.
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
            '} >> "$VLLM_BASELINE_ARGV"\n',
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
        for name in GROUPED_JOBS:
            base = BASE_GROUPS[name]
            self.assertEqual(base.count(marker), 1, f"{name}: base key shape changed")
            groups = GROUP_LINE.findall(job_block(self.text, name))
            self.assertEqual(len(groups), 1, f"{name} must have exactly one group")
            for event in EVENTS:
                with self.subTest(job=name, event=event):
                    resolved = groups[0].replace("${{ github.event_name }}", event)
                    self.assertEqual(
                        resolved, base.replace(marker, f"-{event}{marker}")
                    )

    def test_no_group_key_carries_a_run_varying_token(self) -> None:
        """An independent statement of the same defect, by name."""
        for name in GROUPED_JOBS:
            group = GROUP_LINE.findall(job_block(self.text, name))[0]
            for token in RUN_VARYING_TOKENS:
                with self.subTest(job=name, token=token):
                    self.assertNotIn(
                        token,
                        group,
                        f"{token} makes {name}'s key unique per run, disabling dedupe",
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

    def test_the_workflow_level_group_cancels_prs_and_nothing_else(self) -> None:
        header = self.text.split("\njobs:\n", 1)[0]
        cancel = re.search(r"(?m)^  cancel-in-progress: (.+)$", header)
        self.assertIsNotNone(cancel)
        for event in EVENTS:
            with self.subTest(event=event):
                self.assertEqual(
                    resolve_boolean(cancel.group(1), event), event == "pull_request"
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

    def test_sanitize_cpu_stays_continue_on_error_for_the_push_and_pr_lanes(self) -> None:
        """Out of scope for this row: it is the closing step of the hardening
        row. The baseline lane gets its bindingness from baseline-summary."""
        self.assertIn("continue-on-error: true", job_block(self.text, "sanitize-cpu"))


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

    RANGE_SCOPED = ("check-commit-trailers.py", "check-role-discipline.py")
    FAKE_BASE = "1" * 40
    FAKE_HEAD = "2" * 40

    def setUp(self) -> None:
        self.job = job_block(workflow_text(), "agent-record")
        self.steps = [
            step
            for step in steps_of(self.job)
            if step_env(step).get("PUSH_BASE") == "${{ github.event.before }}"
        ]

    def test_every_before_consumer_is_one_of_the_steps_under_test(self) -> None:
        """No other route into `github.event.before` may exist in this job.

        Counted over CODE only -- the guards' own comments name the variable
        while explaining why it is empty on this lane, and a comment consumes
        nothing.
        """
        self.assertEqual(len(self.steps), 2)
        job_code = "\n".join(code_lines(self.job))
        accounted = "\n".join(
            "\n".join(code_lines("\n".join(step))) for step in self.steps
        )
        self.assertEqual(
            job_code.count("github.event.before"),
            accounted.count("github.event.before"),
            "an unaccounted github.event.before consumer in agent-record",
        )
        self.assertEqual(accounted.count("github.event.before"), 2)

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
                        if checker in joined:
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
        """A CI registration behind an `if:` is not a registration."""
        job = job_block(workflow_text(), "agent-record")
        self.assertNotRegex(job, r"(?m)^    if:")


if __name__ == "__main__":
    unittest.main(verbosity=2)
