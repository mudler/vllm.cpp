# Orchestration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the operator's loop a written, gated protocol — an independent reviewer that mutates rather than reads, and a gate command that can actually fail.

**Architecture:** Three deliverables, each independently useful. The **reviewer and implementer prompts** become tracked artifacts under `.agents/prompts/`, so the highest-value piece stops being folklore. `scripts/check-gate-commands.py` classifies every `READY`-or-later row's gate command and ships as a **ratchet**, not a demand — 67 of 97 rows cannot state one today, so a gate requiring them would be red on arrival. The **loop itself** lands in `.agents/workflow.md` with `check-protocol-consistency.py` asserting it, exactly as subsystem A landed the role interview.

**Tech Stack:** Python 3 standard library only, `argparse`, `importlib.util` for hyphenated module loading, `unittest`. Markdown for the tracked prompts. Matches the house style of `scripts/check-agent-record.py`.

## Global Constraints

Copied from `AGENTS.md` and `.agents/specs/orchestration-harness.md`. Every task's requirements implicitly include this section.

- **Every commit carries `FOLLOWING_AGENTS_PROTOCOL`** plus `Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]`. **Never** `Signed-off-by` or `Co-Authored-By` from an AI.
- **A role must be declared before committing** — subsystem A shipped, so `scripts/agent-preflight.sh` fails without one. Run `python3 scripts/agent-role.py show`; this worktree should already hold `helper row=HARNESS-B`. Claim before you commit, never soften the gate.
- **Run `bash scripts/agent-preflight.sh` before committing; it must exit 0.** Never pipe it — redirect to a file and check `$?`.
- **Every commit touching `scripts/`, `tests/` or `.agents/specs/` also updates `docs/STATUS.md` and `docs/BENCHMARKS.md` in the SAME commit.** Verify the **committed** form with `python3 scripts/check-doc-checkpoint.py --commit <sha>` — preflight runs that checker `--staged` only, which passes vacuously after committing, and `--staged` can also fail spuriously once the docs are already committed. `--commit` on the final SHA is the authoritative check.
- **Doc budgets are tight and must be measured, never assumed.** At the merge base: `docs/STATUS.md` 283,992 of a 284,081 cap (**89 chars**), `.agents/NOW.md` 5,973 of 6,000 (**27 chars**), and every `docs/BENCHMARKS.md` prose paragraph is near its 700-char limit. Measure with:
  ```bash
  python3 -c "import importlib.util,sys; s=importlib.util.spec_from_file_location('c','scripts/check-public-doc-tables.py'); m=importlib.util.module_from_spec(s); sys.modules['c']=m; s.loader.exec_module(m); t=open('docs/STATUS.md').read(); print(len(t), m.STATUS_RATCHET['chars'])"
  ```
- **Use a ROLLING doc surface.** Task 1 adds ONE short line to each page; every later task **edits that line digit-only** (`step 1/5` → `2/5` → … → `step 5/5`) so the pages do not grow. **Never rewrite an unrelated paragraph to buy room** — that was done twice on earlier branches and introduced a factual error both times. If your entry does not fit, shorten **your own** sentence.
- **Python standard library only.** `from __future__ import annotations`, type hints, house style.
- **Never weaken a checker, raise a cap, or relax a budget to make something pass. Repair the record.**
- Stage explicit paths. Never `git add -A`.
- **For every test you write, mutate the line it names and confirm it goes red, and report the result.** Eleven times across the two preceding branches a test passed with its subject deleted — including a gate's own default and a probe's five fields. This is the single most reliable defect class in this repo.

**Existing interfaces you build on** (read, do not reimplement):

- `scripts/check-agent-record.py`: `ClaimRow` (fields `path`, `line_no`, `item_id`, `state`, `header`, `cells`, `raw`; method `field(name)`), `parse_claim_rows(path, errors) -> list[ClaimRow]`, `MATRIX_PATHS`, `AGENTS`, `ROOT`, `local_spec_paths(row) -> list[Path]` (resolves a row's `Spike/spec` links to real files under `.agents/specs/`).
- `scripts/check-protocol-consistency.py`: `INTERVIEW_MARKER` (line 56), `INTERVIEW_REQUIRED` (57), `interview_errors(text) -> list[str]` (133), `main()` (144).
- `scripts/agent-preflight.sh`: `CHECKERS=(` at line 57, `SUITES=(` at line 73.
- `.github/workflows/ci.yml`: lines 94–95 run `test_agent_role.py` and `test_agent_onboard.py`.

**Measured baseline (merge base `35f7cb94`), the number that shapes this plan:** of **97** `READY`-or-later rows — 30 have a `Gates` section containing a runnable command, **46 have a `Gates` section with no command**, **20 have a spec with no `Gates` section**, 1 has no resolving spec. **67 of 97 (69%) cannot state a runnable gate command today.**

---

## File Structure

| File | Responsibility |
|---|---|
| `.agents/prompts/reviewer.md` (create) | The reviewer contract: mutate, don't read; don't trust the report; a plan-mandated finding is still a finding |
| `.agents/prompts/implementer.md` (create) | The implementer contract: TDD, commit in the worktree, report honestly, escalate rather than guess |
| `scripts/check-gate-commands.py` (create) | Classify each `READY`+ row's gate command; report mode, then a ratchet |
| `tests/scripts/test_check_gate_commands.py` (create) | Unit + mutation suite |
| `.agents/specs/gate-command-audit-2026-08-06.md` (create) | The 67-row debt, recorded honestly |
| `scripts/check-protocol-consistency.py` (modify) | Assert the prompts exist and carry their binding instruction; assert the loop is in `workflow.md` |
| `scripts/agent-preflight.sh`, `.github/workflows/ci.yml` (modify) | Wire the checker and its suite |
| `AGENTS.md`, `.agents/workflow.md`, `.agents/specs/operator-helper-protocol.md` (modify) | The loop, moved with its gate |

---

### Task 1: The tracked prompts

**Files:**
- Create: `.agents/prompts/reviewer.md`, `.agents/prompts/implementer.md`
- Modify: `scripts/check-protocol-consistency.py`
- Test: `tests/scripts/test_check_protocol_consistency.py`

**Interfaces:**
- Consumes: `interview_errors(text) -> list[str]` and `main()` from `check-protocol-consistency.py`.
- Produces: `PROMPT_REQUIRED: dict[str, tuple[str, ...]]` mapping each prompt path to phrases it must contain; `prompt_errors() -> list[str]`.

**Why this is Task 1:** it is the highest-value deliverable and depends on nothing. Across two branches, **every Important finding came from an independent reviewer and none from an implementer's self-review**, and the reviewers found them by mutating code, not by reading diffs. A prompt that lives only in an operator's head is not a protocol.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_check_protocol_consistency.py`, above its `if __name__` block:

```python
class PromptArtifactTests(unittest.TestCase):
    def test_both_prompts_exist_and_are_tracked(self):
        for name in ("reviewer.md", "implementer.md"):
            path = ROOT / ".agents/prompts" / name
            self.assertTrue(path.is_file(), f"{name} must exist")

    def test_the_reviewer_prompt_carries_the_mutation_instruction(self):
        # The instruction IS the deliverable. A reviewer told only to "review"
        # reads the diff, and reading found none of the eleven tests that
        # passed with their subject deleted.
        text = (ROOT / ".agents/prompts/reviewer.md").read_text(encoding="utf-8")
        for needle in ("mutate", "delete or invert", "stays green"):
            self.assertIn(needle, text.lower(), needle)

    def test_the_reviewer_prompt_refuses_to_defer_to_the_plan(self):
        text = (ROOT / ".agents/prompts/reviewer.md").read_text(encoding="utf-8")
        self.assertIn("plan-mandated", text.lower())

    def test_checker_rejects_a_prompt_missing_its_instruction(self):
        errors = consistency.prompt_errors({"nonexistent-prompt.md": ("mutate",)})
        self.assertTrue(errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_check_protocol_consistency.py -v`
Expected: FAIL — the prompt files do not exist, and `AttributeError: module … has no attribute 'prompt_errors'`.

- [ ] **Step 3: Write the prompts**

Create `.agents/prompts/reviewer.md`:

```markdown
# Reviewer prompt

You review one change. You did not write it and you will not fix it.

## The binding instruction: mutate, don't read

For each test in the change, **delete or invert the line it names and re-run
the suite. A test that stays green is a finding**, regardless of how it reads.

This is not a style preference. Across two branches of this project, eleven
tests passed with the thing they named deleted — including a gate's own
default (an unrelated line satisfied the assertion), a probe with five
hardcoded fields, and `assertIn("merged", reason)` where the string was
`"unmerged"`. **None was visible by reading the diff.** A reviewer who reads
and comments on style adds nothing this project has not already paid for.

## Do not trust the report

Treat the implementer's report as unverified claims about the code. Verify each
against the change. A stated rationale — "kept it simple deliberately", "left
it per YAGNI" — is the implementer grading its own work and **never** downgrades
a finding's severity. On two branches, three implementer reports asserted
something false in good faith; each was caught by reproducing the claim rather
than accepting it.

## A plan-mandated finding is still a finding

Roughly half of all Important findings on the preceding branches were defects in
the **plan text**, not the implementation. A reviewer that treats the plan as
authority cannot find them. Report them, labelled `plan-mandated`, and let the
human decide which governs.

## Severity

- **Critical** — corrupts the record, weakens a gate, or leaves a false claim in
  a document agents read.
- **Important** — the change cannot be trusted until fixed: wrong or fragile
  behavior, a missed requirement, a test that asserts nothing.
- **Minor** — polish.

Cite `file:line` for every finding and for any check you would otherwise answer
with a bare "yes". Acknowledge what was done well before listing issues.

## What you may not do

- Never fix what you found. Findings go back to a fresh implementer.
- Never mutate the reviewed worktree, its index, HEAD or branch state. Work in a
  scratch copy.
- Never re-run the full suite just to confirm the report; run a focused check
  when reading raises a specific doubt.
```

Create `.agents/prompts/implementer.md`:

```markdown
# Implementer prompt

You implement one task. A different agent will review it by mutating your code.

## Method

1. Write the failing test first. Run it. Confirm it fails **for the stated
   reason** — a test that fails for the wrong reason pins nothing.
2. Implement the minimum that makes it pass.
3. **Mutate every test you wrote**: delete the line it names, confirm red,
   restore. Report the results. If a briefed test does not pin what it claims,
   fix it and say so — four implementers before you did exactly that and were
   right every time.
4. Run the project gate (`scripts/agent-preflight.sh`, redirected to a file,
   never piped) and confirm `EXIT=0`.
5. Commit in your worktree with the required trailers, and return the SHA.

## Honesty rules

- **Never let a failure and an absence look the same.** Every recorded defect
  class in this repo is that bug: a substring `--grep` crediting a row with
  another row's commits, `.get()` on a missing key reporting a live claim as
  finished, a git failure mapped to `""` and read as "no evidence".
- **Report what you did not do.** An empty concerns section is itself a claim.
- **Escalate rather than guess.** Report `BLOCKED` or `NEEDS_CONTEXT` with
  specifics. Bad work is worse than no work, and you will not be penalised for
  stopping.
- **Never weaken a checker, a budget or a test to make something pass.** If the
  gate is red, repair the record.

## Deviating from the brief

You may deviate when the brief is wrong — and it sometimes is. State the
deviation explicitly in your report with the evidence that justifies it. Silent
scope expansion is a defect; a disclosed, argued correction is not.
```

- [ ] **Step 4: Write the checker**

In `scripts/check-protocol-consistency.py`, after `INTERVIEW_REQUIRED` (line 57):

```python
# The reviewer prompt's value is the MUTATION instruction; a reviewer told only
# to "review" reads the diff, and reading found none of the eleven tests that
# passed with their subject deleted. Pin the instruction, not the file.
PROMPT_REQUIRED = {
    ".agents/prompts/reviewer.md": (
        "mutate",
        "delete or invert",
        "stays green",
        "plan-mandated",
    ),
    ".agents/prompts/implementer.md": (
        "failing test first",
        "mutate every test",
        "escalate rather than guess",
    ),
}


def prompt_errors(required: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Each tracked prompt exists and carries its binding instruction."""
    errors: list[str] = []
    for relative, needles in (required or PROMPT_REQUIRED).items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative} is missing; the prompt is the protocol")
            continue
        text = path.read_text(encoding="utf-8").lower()
        errors.extend(
            f"{relative} omits {needle!r}" for needle in needles if needle not in text
        )
    return errors
```

Call `prompt_errors()` from `main()` and add its output to the error list, exactly as `interview_errors` is called.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 tests/scripts/test_check_protocol_consistency.py -v` → PASS.
Run: `python3 scripts/check-protocol-consistency.py; echo "EXIT=$?"` → `EXIT=0`.

- [ ] **Step 6: Mutate**

Delete the `"delete or invert"` line from `reviewer.md` → the suite must go red. Delete the `prompt_errors()` call from `main()` → `test_checker_rejects_a_prompt_missing_its_instruction` must go red. Restore both and confirm green. Report both results.

- [ ] **Step 7: Doc surfaces, preflight, commit**

Add ONE short rolling line to `docs/STATUS.md` (89 chars of headroom — keep it under 80) and one short clause to a `docs/BENCHMARKS.md` prose paragraph that has room; measure first. Both must say `step 1/5`.

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add .agents/prompts/reviewer.md .agents/prompts/implementer.md \
        scripts/check-protocol-consistency.py tests/scripts/test_check_protocol_consistency.py \
        docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
protocol(prompts): the reviewer contract becomes a tracked artifact (B step 1)

Across two branches every Important finding came from an independent reviewer
and none from an implementer's self-review, and the reviewers found them by
MUTATING code rather than reading diffs. Eleven tests passed with the thing they
named deleted; none was visible by reading. A prompt that lives only in an
operator's head is not a protocol, so the instruction is tracked and gated.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 2: Classify gate commands (report only)

**Files:**
- Create: `scripts/check-gate-commands.py`
- Test: `tests/scripts/test_check_gate_commands.py`

**Interfaces:**
- Consumes: `check-agent-record.py` — `ClaimRow`, `parse_claim_rows`, `MATRIX_PATHS`, `AGENTS`, `ROOT`, `local_spec_paths`.
- Produces: `GATED_STATES: frozenset[str]`; `AUDITED_MATRIX_PATHS: list[Path]`; `gates_section(text: str) -> str | None`; `runnable_commands(section: str) -> list[str]`; `classify_row(row) -> tuple[str, str]` returning one of `"runnable"`, `"gates-no-command"`, `"no-gates-section"`, `"no-spec"` plus a detail string; `audit() -> list[dict]`; `main(argv=None) -> int`.

**This task ships NO gate.** It classifies and reports. The ratchet is Task 4, after the debt is recorded — the same ordering the live-state audit used, and for the same reason: a gate wired before the record is repaired has to be relaxed to pass, and a relaxed gate is worse than none.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_check_gate_commands.py`:

```python
#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-gate-commands.py.

A gate command that cannot fail collapses "done" into the implementer's opinion
of its own work. This classifier's only job is to tell a runnable command from
prose that looks like one.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gates = _load("check_gate_commands", "scripts/check-gate-commands.py")


class GatesSectionTests(unittest.TestCase):
    def test_finds_the_gates_heading_at_any_level(self):
        for heading in ("## Gates", "### Gates", "#### Gates and evidence"):
            text = f"# Spec\n\nintro\n\n{heading}\n\nrun `ctest -R foo`\n\n## Next\n\ntail\n"
            section = gates.gates_section(text)
            self.assertIsNotNone(section, heading)
            self.assertIn("ctest", section)
            self.assertNotIn("tail", section, "must stop at the next heading")

    def test_returns_none_when_there_is_no_gates_section(self):
        self.assertIsNone(gates.gates_section("# Spec\n\n## Scope\n\nnothing here\n"))

    def test_is_not_fooled_by_the_word_gates_in_prose(self):
        # "the gates are green" is not a section heading.
        self.assertIsNone(gates.gates_section("# Spec\n\nAll the gates are green.\n"))


class RunnableCommandTests(unittest.TestCase):
    def test_recognises_a_real_command(self):
        for body in [
            "run `ctest -R test_foo`",
            "`python3 scripts/check-agent-record.py`",
            "```\nbash scripts/agent-preflight.sh\n```",
            "`cmake --build build -j`",
        ]:
            self.assertTrue(gates.runnable_commands(body), body)

    def test_rejects_prose_that_merely_mentions_gating(self):
        for body in [
            "Correctness, e2e and performance gates apply.",
            "The SACRED gate must pass on GB10.",
            "`docs/BENCHMARKS.md`",
        ]:
            self.assertEqual(gates.runnable_commands(body), [], body)

    def test_rejects_a_command_that_cannot_fail(self):
        # These are the exact shapes the spec forbids: a Verify that always
        # succeeds turns "done" into an opinion.
        for body in ["`true`", "`echo ok`", "`:`", "`echo done && true`"]:
            self.assertEqual(gates.runnable_commands(body), [], body)

    def test_rejects_a_piped_command(self):
        # `cmd | tail` reports tail's exit status, so the gate cannot fail.
        self.assertEqual(gates.runnable_commands("`ctest -R foo | tail -5`"), [])


class ShippedRecordTests(unittest.TestCase):
    def test_the_audit_covers_every_gated_state(self):
        self.assertEqual(
            gates.GATED_STATES,
            frozenset({"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"}),
        )

    def test_all_seven_matrices_are_audited(self):
        names = {p.name for p in gates.AUDITED_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertIn("sglang-matrix.md", names)
        self.assertEqual(len(names), 7)

    def test_every_record_carries_a_known_verdict(self):
        known = {"runnable", "gates-no-command", "no-gates-section", "no-spec"}
        records = gates.audit()
        self.assertTrue(records)
        for item in records:
            self.assertIn(item["verdict"], known)

    def test_report_mode_exits_zero_even_with_debt(self):
        # 67 of 97 rows cannot state a command today. Report mode must still
        # exit 0 -- the ratchet is step 4, after the debt is recorded.
        self.assertEqual(gates.main([]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_check_gate_commands.py -v`
Expected: FAIL — `FileNotFoundError`/`AssertionError` from `_load`; the script does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/check-gate-commands.py`:

```python
#!/usr/bin/env python3
"""Classify each gated row's Gates section: does it name a command that can FAIL?

A row's `Gates` field promises "exact commands", and nothing has ever checked
that one exists or that it can fail. A gate that is `true`, `echo ok`, or piped
into another command collapses "done" into the implementer's opinion of its own
work.

This ships as a CLASSIFIER first and a ratchet second, deliberately: 67 of 97
gated rows cannot state a runnable command today, so a gate demanding one would
be red on arrival and would have to be relaxed to pass. A relaxed gate is worse
than no gate.

    scripts/check-gate-commands.py            # report
    scripts/check-gate-commands.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


record = _load("agent_record", "scripts/check-agent-record.py")

# DONE is included: a row that lost its gate command is exactly the regression
# this exists to catch, and DONE rows are the ones people stop looking at.
GATED_STATES = frozenset({"READY", "ACTIVE", "GATING", "DONE", "BLOCKED"})

# check-agent-record.py's MATRIX_PATHS covers 5 of the 7 matrices. Audit all
# seven, without widening that constant -- it governs a repo-wide CI gate whose
# row contract these two files have never been held to.
AUDITED_MATRIX_PATHS = [
    *record.MATRIX_PATHS,
    record.AGENTS / "feature-matrix.md",
    record.AGENTS / "sglang-matrix.md",
]

_GATES_HEADING = re.compile(r"(?im)^#{1,6}\s*gates\b.*$")
_HEADING = re.compile(r"(?m)^#{1,6}\s")

# A command is recognised by an executable-looking leading token.
_COMMAND = re.compile(
    r"(?:^|\s)(ctest|pytest|python3?|cmake|bash|sh|make|nsys|ncu|git|gh|scripts/|\./|tests/)"
)
# Shapes that cannot fail, so they are not gates at all.
_CANNOT_FAIL = re.compile(r"^\s*(true|:|echo\b)")


def gates_section(text: str) -> str | None:
    """The body under the first `Gates` HEADING, or None. Prose does not count."""
    match = _GATES_HEADING.search(text)
    if not match:
        return None
    rest = text[match.end() :]
    nxt = _HEADING.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _candidates(section: str) -> list[str]:
    inline = re.findall(r"`([^`\n]+)`", section)
    fenced = re.findall(r"```[a-z]*\n(.*?)```", section, re.S)
    for block in fenced:
        inline.extend(line for line in block.splitlines() if line.strip())
    return [c.strip() for c in inline if c.strip()]


def runnable_commands(section: str) -> list[str]:
    """Commands in this section that could actually fail."""
    good = []
    for candidate in _candidates(section):
        if not _COMMAND.search(" " + candidate):
            continue
        if _CANNOT_FAIL.match(candidate):
            continue
        if "|" in candidate:  # `cmd | tail` reports tail's status
            continue
        good.append(candidate)
    return good


def classify_row(row) -> tuple[str, str]:
    specs = [p for p in record.local_spec_paths(row) if p.is_file()]
    if not specs:
        return "no-spec", "no resolving .agents/specs/ link"
    text = specs[0].read_text(encoding="utf-8", errors="replace")
    section = gates_section(text)
    if section is None:
        return "no-gates-section", specs[0].name
    commands = runnable_commands(section)
    if not commands:
        return "gates-no-command", specs[0].name
    return "runnable", commands[0]


def audit() -> list[dict]:
    records = []
    for path in AUDITED_MATRIX_PATHS:
        errors: list[str] = []
        for row in record.parse_claim_rows(path, errors):
            if row.state not in GATED_STATES:
                continue
            verdict, detail = classify_row(row)
            records.append(
                {
                    "id": row.item_id,
                    "state": row.state,
                    "path": str(row.path.relative_to(ROOT)),
                    "line": row.line_no,
                    "verdict": verdict,
                    "detail": detail,
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify gated rows' gate commands.")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)

    records = audit()
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    counts: dict[str, int] = {}
    for item in records:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    for verdict in ("runnable", "gates-no-command", "no-gates-section", "no-spec"):
        print(f"  {counts.get(verdict, 0):4d}  {verdict}")
    print(f"\n{len(records)} gated rows; {counts.get('runnable', 0)} carry a command that can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make it executable: `chmod +x scripts/check-gate-commands.py`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 tests/scripts/test_check_gate_commands.py -v` → PASS, 11 tests.

- [ ] **Step 5: Smoke-test against the real record**

Run: `python3 scripts/check-gate-commands.py`
Expected: roughly `30 runnable`, `46 gates-no-command`, `20 no-gates-section`, `1 no-spec` over 97 rows. Report the actual numbers — the record moves, and Task 3 consumes them.

- [ ] **Step 6: Mutate**

Confirm each goes red, then restore: drop the `"|"` check in `runnable_commands`; drop the `_CANNOT_FAIL` check; make `gates_section` match the word `gates` anywhere rather than at a heading; remove `feature-matrix.md` from `AUDITED_MATRIX_PATHS`. Report all four.

- [ ] **Step 7: Roll the docs to `step 2/5`, preflight, commit**

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/check-gate-commands.py tests/scripts/test_check_gate_commands.py \
        docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
tools(gates): classify whether a gated row names a command that can FAIL (B step 2)

Report only. 67 of 97 gated rows cannot state a runnable command today, so a
gate demanding one would be red on arrival and would have to be relaxed to
pass -- and a relaxed gate is worse than none. The ratchet is step 4, after
step 3 records the debt.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 3: Record the debt

**Files:**
- Create: `.agents/specs/gate-command-audit-2026-08-06.md`

**Interfaces:**
- Consumes: `scripts/check-gate-commands.py --json`.
- Produces: the artifact Task 4's ratchet baseline cites.

**No code and no matrix is edited in this task.** The debt lands before the gate, so the reasoning is reviewable separately from the enforcement — and so the ratchet's baseline is a recorded decision rather than a number someone pasted.

- [ ] **Step 1: Capture the audit**

```bash
python3 scripts/check-gate-commands.py --json > /tmp/gates.json
python3 scripts/check-gate-commands.py
python3 -c "
import json, collections
r = json.load(open('/tmp/gates.json'))
print(collections.Counter(x['verdict'] for x in r))
print('by matrix:', collections.Counter(x['path'] for x in r if x['verdict'] != 'runnable'))
"
```

- [ ] **Step 2: Hand-verify a sample before trusting the classifier**

Pick **three** rows it called `runnable` and **three** it called `gates-no-command`. Open each row's spec, read the `Gates` section, and confirm the verdict matches what a human would say. A classifier wrong on a sample is wrong on all 97 — if you find a mismatch, stop and report it rather than writing the artifact. Record the sample and its outcome.

- [ ] **Step 3: Write the artifact**

Create `.agents/specs/gate-command-audit-2026-08-06.md` with these sections:

- **Scope** — the gated rows at `origin/main` @ `<SHA>`; what the classifier decides and what it cannot.
- **Method** — `scripts/check-gate-commands.py`, the classification rules verbatim, and the Step 2 sample with its outcome.
- **Findings** — the four counts, the per-matrix breakdown, and the full list of rows in each non-`runnable` bucket.
- **What this does NOT mean** — a row without a runnable gate command is not ungated work; many carry real evidence in prose or in `.agents/parity-ledger.md`. The finding is that **the gate cannot be checked mechanically**, not that the work is unverified. Say this plainly; the opposite reading would slander a lot of landed work.
- **The ratchet baseline** — the exact count of `runnable` rows, which Task 4 pins. State that the baseline may only rise.
- **Risks/decisions** — every row the classifier could not decide, and the human call made.

- [ ] **Step 4: Roll the docs to `step 3/5`, preflight, commit**

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add .agents/specs/gate-command-audit-2026-08-06.md docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
record(gates): the gate-command debt, measured before anything enforces it (B step 3)

67 of 97 gated rows cannot state a command that can fail. That is a finding
about MECHANICAL checkability, not about whether the work was verified -- many
of those rows carry real evidence in prose and in the parity ledger, and the
artifact says so plainly.

Recorded before the ratchet so the baseline is a decision, not a pasted number.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 4: The ratchet

> **Superseded on 2026-08-06 by the whole-branch review, in two places. This
> plan text is left as written; where it disagrees with the list below, the list
> governs.**
>
> 1. **"Shrink-only" is wrong** — here, in Task 2's `AUDITED_MATRIX_PATHS`
>    comment, in "Done when", and wherever else this plan says it. What shipped
>    is an **exact pin**: `RUNNABLE_BASELINE` must EQUAL the audited `runnable`
>    set, so growth is red too and any movement re-pins the set in the same
>    change. See `.agents/specs/gate-command-audit-2026-08-06.md` § The ratchet
>    baseline.
> 2. **`test_all_seven_matrices_are_audited` is wrong**, and its
>    `assertIn("sglang-matrix.md", names)` was plan-mandated. `sglang-matrix.md`
>    carries a classification column, not a lifecycle state, and contributed 0
>    rows of 87 — listed and empty, which is this repo's recorded defect class.
>    It is **dropped** from the audited set with the reasoning recorded; six
>    matrices are audited, and the test now pins the justification (zero rows AND
>    zero parse errors) instead of the membership. See that artifact's risk 6.

**Files:**
- Modify: `scripts/check-gate-commands.py`
- Modify: `scripts/agent-preflight.sh` (`CHECKERS=(` line 57, `SUITES=(` line 73)
- Modify: `.github/workflows/ci.yml` (near lines 94–95)
- Test: `tests/scripts/test_check_gate_commands.py`

**Interfaces:**
- Consumes: `audit()` from Task 2.
- Produces: `RUNNABLE_BASELINE: frozenset[str]` — the SET of row IDs carrying a runnable gate command, **not a count**; `ratchet_errors(records: list[dict]) -> list[str]`; `--check` on `main()`.

**Why a ratchet and not a demand:** 72 of 97 rows cannot satisfy a demand today, and this repo already uses shrink-only ratchets for exactly this shape (`STATUS_RATCHET`, the device-leakage ratchet). It ships green today and gets stricter every time someone fixes a row.

**Pin the SET of row IDs, never a count** (specified by step 3's artifact, § the ratchet baseline). A bare integer cannot distinguish *a row lost its gate command* from *a row legitimately left the population* — and the population already moved 3 rows mid-branch, which is why 97 was deliberately never pinned. A count would go red on a legitimate record edit, and the author would "fix" it by lowering the number, which is the gate erasing its own finding.

**The rule, verbatim from the artifact:** a drop is a regression **only if** the row still exists and is still in `GATED_STATES`. If the row was deleted, merged, or transitioned out, the baseline is re-pinned **in the same change**, naming the row and the reason. `ratchet_errors` must therefore report *which* IDs left and why it could or could not tell.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_check_gate_commands.py`, above `if __name__`:

```python
class RatchetTests(unittest.TestCase):
    def test_the_baseline_matches_the_shipped_record(self):
        runnable = {r["id"] for r in gates.audit() if r["verdict"] == "runnable"}
        self.assertEqual(runnable, set(gates.RUNNABLE_BASELINE))

    def test_a_row_that_loses_its_command_is_refused(self):
        # Still present, still gated, no longer runnable -- a real regression.
        victim = sorted(gates.RUNNABLE_BASELINE)[0]
        records = [{"verdict": "gates-no-command", "id": victim, "state": "READY",
                    "path": "p", "line": 1, "detail": "d"}]
        errors = gates.ratchet_errors(records)
        self.assertTrue(errors)
        self.assertIn(victim, errors[0])
        self.assertIn("Repair the row", errors[0])

    def test_a_row_that_left_the_population_reports_differently(self):
        # Deleted or transitioned out: legitimate, but must re-pin. The two
        # cases MUST be distinguishable -- that is why the baseline is a set.
        errors = gates.ratchet_errors([])
        self.assertTrue(errors)
        self.assertTrue(any("left the gated population" in e for e in errors))
        self.assertFalse(any("Repair the row" in e for e in errors))

    def test_an_improvement_is_allowed(self):
        records = [
            {"verdict": "runnable", "id": rid, "state": "READY",
             "path": "p", "line": 1, "detail": "d"}
            for rid in sorted(gates.RUNNABLE_BASELINE)
        ] + [{"verdict": "runnable", "id": "NEW-ROW", "state": "READY",
              "path": "p", "line": 2, "detail": "d"}]
        self.assertEqual(gates.ratchet_errors(records), [])

    def test_check_mode_passes_on_the_shipped_record(self):
        # The gate ships GREEN. It was wired after the debt was recorded, so it
        # never had to be relaxed to pass.
        self.assertEqual(gates.main(["--check"]), 0)

    def test_the_checker_is_wired_into_preflight_and_ci(self):
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("check-gate-commands", preflight)
        self.assertIn("test_check_gate_commands", preflight)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/check-gate-commands.py --check", ci)
        self.assertIn("tests/scripts/test_check_gate_commands.py", ci)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_check_gate_commands.py -v`
Expected: FAIL with `AttributeError: … has no attribute 'RUNNABLE_RATCHET'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/check-gate-commands.py`, above `main()`:

```python
# Shrink-only, like STATUS_RATCHET in check-public-doc-tables.py -- but a SET of
# row IDs, not a count. A count cannot tell "this row lost its gate command"
# from "this row left the population", and the population moves: 3 rows moved
# mid-branch while step 2 was being written. Pinning a count would go red on a
# legitimate record edit, and the natural "fix" is to lower the number, which is
# the gate erasing its own finding.
RUNNABLE_BASELINE = frozenset({
    # the exact row IDs step 3 recorded as `runnable`
})


def ratchet_errors(records: list[dict]) -> list[str]:
    """A row may not silently lose its gate command.

    Leaving the gated population is legitimate; losing the command is not. So
    only IDs still PRESENT and still gated can be a regression -- anything else
    is a record edit that must re-pin the baseline in the same change.
    """
    runnable = {item["id"] for item in records if item["verdict"] == "runnable"}
    present = {item["id"] for item in records}
    lost = sorted((RUNNABLE_BASELINE - runnable) & present)
    departed = sorted(RUNNABLE_BASELINE - runnable - present)
    errors = []
    if lost:
        errors.append(
            "these rows still exist and are still gated but no longer name a "
            f"command that can fail: {', '.join(lost)}. Repair the row, never "
            "the baseline."
        )
    if departed:
        errors.append(
            f"these baseline rows left the gated population: {', '.join(departed)}. "
            "If that is a legitimate record edit, re-pin RUNNABLE_BASELINE in the "
            "SAME change, naming each row and the reason."
        )
    return errors
```

In `main()`, add the flag and return its errors:

```python
    parser.add_argument("--check", action="store_true", help="fail on a ratchet regression")
```

```python
    if args.check:
        errors = ratchet_errors(records)
        for line in errors:
            print(f"ERROR: {line}", file=sys.stderr)
        return 1 if errors else 0
```

- [ ] **Step 4: Wire preflight and CI**

Add `check-gate-commands` to `CHECKERS=(` (line 57) — note `claim-view` shows the pattern for a checker needing an argument; this one needs `--check`, so follow the `claim-view` branch shape. Add `test_check_gate_commands` to `SUITES=(` (line 73).

In `.github/workflows/ci.yml`, beside the existing script-suite lines (94–95):

```yaml
          python3 scripts/check-gate-commands.py --check
          python3 tests/scripts/test_check_gate_commands.py
```

- [ ] **Step 5: Run tests and both gates**

```bash
python3 tests/scripts/test_check_gate_commands.py -v          # PASS, 16 tests
python3 scripts/check-gate-commands.py --check; echo "EXIT=$?" # 0
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"  # 0
```

- [ ] **Step 6: Mutate**

Confirm each goes red, then restore: delete a real gate command from a spec whose row is in the baseline (the regression the gate exists to catch); replace the `& present` intersection with nothing, so a departed row is misreported as a loss; delete the `check-gate-commands` line from `CHECKERS`; delete the CI line. Report all four. **If `--check` is red for any reason other than your own mutation, the record regressed — repair the row, never the baseline.**

- [ ] **Step 7: Roll the docs to `step 4/5`, preflight, commit**

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add scripts/check-gate-commands.py tests/scripts/test_check_gate_commands.py \
        scripts/agent-preflight.sh .github/workflows/ci.yml docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
gate(gates): a row may never lose its runnable gate command (B step 4)

Shrink-only ratchet, the same shape as STATUS_RATCHET: the count of gated rows
carrying a command that can FAIL may never fall. It ships green because it was
wired AFTER step 3 recorded the debt, so it never had to be relaxed to pass,
and it gets stricter every time someone fixes a row.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

### Task 5: The loop, moved with its gate

**Files:**
- Modify: `.agents/workflow.md`, `AGENTS.md`, `.agents/specs/operator-helper-protocol.md`
- Modify: `scripts/check-protocol-consistency.py`
- Test: `tests/scripts/test_check_protocol_consistency.py`

**Interfaces:**
- Consumes: `prompt_errors()` from Task 1.
- Produces: `LOOP_MARKER = "<!-- orchestration-loop:begin -->"`; `loop_errors(text: str) -> list[str]`.

**Why the checker moves in the same commit:** `check-protocol-consistency.py` exists because an obligation was once migrated in `AGENTS.md` and the checker but not in the manual, which went on instructing agents to do the thing the migration removed. Subsystem A landed the role interview this way; the loop lands the same way.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_check_protocol_consistency.py`, above `if __name__`:

```python
class OrchestrationLoopTests(unittest.TestCase):
    def test_workflow_carries_the_loop(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(consistency.LOOP_MARKER, text)

    def test_the_loop_names_the_reviewer_and_the_gate(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8").lower()
        for needle in ("reviewer", "mutate", "run the gate yourself", "never fix"):
            self.assertIn(needle, text, needle)

    def test_the_loop_points_at_the_tracked_prompts(self):
        text = (ROOT / ".agents/workflow.md").read_text(encoding="utf-8")
        self.assertIn(".agents/prompts/reviewer.md", text)

    def test_checker_rejects_a_workflow_without_the_loop(self):
        self.assertTrue(consistency.loop_errors("# workflow\n\nno loop here\n"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/scripts/test_check_protocol_consistency.py -v`
Expected: FAIL with `AttributeError: … has no attribute 'LOOP_MARKER'`.

- [ ] **Step 3: Write the prose**

In `.agents/workflow.md`, after the role-interview block, insert:

```markdown
<!-- orchestration-loop:begin -->
### Running a row through sub-agents

Decompose, dispatch, verify, integrate. You do not write the feature.

For each task, serially — never two implementers in one worktree:

1. Dispatch a **fresh** implementer ([prompt](../../../.agents/prompts/implementer.md)). It works
   TDD and commits in the worktree, and returns the SHA.
2. **Run the row's gate yourself.** This is the one failure mode nothing else
   catches: if "done" is the implementer's opinion of its own work, the loop has
   no floor.
3. Dispatch a **fresh reviewer** ([prompt](../../../.agents/prompts/reviewer.md)) — never the
   agent that wrote the code. Its binding instruction is to **mutate, not read**:
   delete the line each test names and re-run. A test that stays green is a
   finding. Eleven such tests shipped on the two branches that built this
   protocol, and none was visible by reading a diff.
4. Findings go back to the implementer, then a **scoped re-review** of the fix
   diff only. **Never fix findings yourself** — a controller fix pollutes the
   context that exists to coordinate, and skips review entirely.

A gate command must exit nonzero on failure. Never `true`, never `echo ok`,
never piped — `cmd | tail` reports `tail`'s status.
`scripts/check-gate-commands.py` ratchets this.

Interactive is the default. In a **declared** headless run, decide rather than
ask, record every decision as immutable `.agents/state-events/` evidence with
its ordered row in the writable `.agents/state-index/` shard, refresh
`.agents/NOW.md`, and run `python3 scripts/check-state-record.py`; park what will
not go green, and never merge. `.agents/state.csv` is the structured-record
manifest.
<!-- orchestration-loop:end -->
```

In `scripts/check-protocol-consistency.py`, add:

```python
LOOP_MARKER = "<!-- orchestration-loop:begin -->"
LOOP_REQUIRED = ("prompts/reviewer.md", "mutate", "run the row's gate yourself")


def loop_errors(text: str) -> list[str]:
    """The operator's loop must live where agents read it, not in a prompt."""
    if LOOP_MARKER not in text:
        return [".agents/workflow.md is missing the orchestration-loop block"]
    lowered = text.lower()
    return [
        f".agents/workflow.md loop omits {needle!r}"
        for needle in LOOP_REQUIRED
        if needle.lower() not in lowered
    ]
```

Call `loop_errors` from `main()` against `.agents/workflow.md`.

Update `AGENTS.md`'s operator bullet to point at the loop and the tracked prompts. Update `.agents/specs/operator-helper-protocol.md` to record that the operator drives feature work through sub-agents with an independent reviewer, and link both prompts.

- [ ] **Step 4: Run tests and the checker**

```bash
python3 tests/scripts/test_check_protocol_consistency.py -v   # PASS
python3 scripts/check-protocol-consistency.py; echo "EXIT=$?" # 0
```

- [ ] **Step 5: Mutate**

Confirm each goes red, then restore: delete the loop block from `workflow.md`; delete the `loop_errors()` call from `main()`; empty `LOOP_REQUIRED`. Report all three.

- [ ] **Step 6: Roll the docs to `step 5/5`, preflight, commit**

```bash
bash scripts/agent-preflight.sh > /tmp/pf.log 2>&1; echo "EXIT=$?"
git add AGENTS.md .agents/workflow.md .agents/specs/operator-helper-protocol.md \
        scripts/check-protocol-consistency.py tests/scripts/test_check_protocol_consistency.py \
        docs/STATUS.md docs/BENCHMARKS.md
git commit -F - <<'EOF'
docs(protocol): the operator's loop ships with the gate that asserts it (B step 5)

check-protocol-consistency.py exists because an obligation was once migrated in
AGENTS.md and the checker but not in the manual, which went on instructing
agents to do the thing the migration removed. Prose is what agents read, so the
loop lands in workflow.md and the checker asserts it is there -- the same way
the role interview landed.

FOLLOWING_AGENTS_PROTOCOL
Assisted-by: Claude Code:claude-opus-5 [ClaudeCode]
EOF
python3 scripts/check-doc-checkpoint.py --commit "$(git rev-parse HEAD)"; echo "doc-checkpoint EXIT=$?"
```

---

## Done when

- `.agents/prompts/reviewer.md` and `implementer.md` are tracked, and `check-protocol-consistency.py` fails without their binding instructions.
- `scripts/check-gate-commands.py --check` exits 0 and is wired into preflight and CI as a shrink-only ratchet.
- `.agents/specs/gate-command-audit-2026-08-06.md` records the debt honestly, including what it does *not* mean.
- The loop is in `.agents/workflow.md` and the checker fails without it.
- Every commit passes `python3 scripts/check-doc-checkpoint.py --commit <sha>`.

## Out of scope

Backfilling gate commands for the 67 rows that lack one — that is per-row work needing per-row knowledge, and the ratchet makes it incremental rather than a flag day. Automating the dispatch loop itself: this protocol is executed by an operator reading `workflow.md`, not by a program. Any change to the roles, the interview, or `.env` handling — that was subsystem A and it has shipped.
