#!/usr/bin/env python3
"""Unit and mutation checks for scripts/audit-live-rows.py.

The audit only helps if it is honest in both directions: it must not call a
live row abandoned when work is really in flight, and it must not call an
abandoned row live because a branch name happens to exist.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = _load("audit_live_rows", "scripts/audit-live-rows.py")


class LiveRowLoadingTests(unittest.TestCase):
    def test_live_states_are_exactly_the_six(self):
        self.assertEqual(
            audit.LIVE_STATES,
            frozenset({"SPIKE", "READY", "ACTIVE", "GATING", "PARTIAL", "BLOCKED"}),
        )

    def test_all_seven_matrices_are_audited(self):
        names = {path.name for path in audit.AUDIT_MATRIX_PATHS}
        self.assertIn("feature-matrix.md", names)
        self.assertIn("sglang-matrix.md", names)
        self.assertEqual(len(names), 7)

    def test_newly_covered_feature_matrix_actually_yields_live_rows(self):
        # Asserting the path is in a list proves nothing: every other assertion
        # here still passes if feature-matrix.md contributes zero rows. This is
        # what the seventh-matrix commit actually bought.
        rows = audit.live_rows()
        self.assertTrue([r for r in rows if r.path.name == "feature-matrix.md"])

    def test_shipped_record_parses_without_errors(self):
        # parse_claim_rows DROPS a row it cannot parse. If a malformed row ever
        # lands, the census silently shrinks -- so the sink must be surfaced,
        # and it must be empty today.
        errors: list[str] = []
        audit.live_rows(errors)
        self.assertEqual(errors, [])

    def test_shipped_matrices_yield_only_live_rows(self):
        rows = audit.live_rows()
        self.assertTrue(rows, "the shipped matrices must contain live rows")
        for row in rows:
            self.assertIn(row.state, audit.LIVE_STATES)

    def test_every_live_state_is_represented_in_the_shipped_matrices(self):
        # Guards the loader against silently dropping a whole state: if a
        # header rename made one state unparseable, its count would go to
        # zero while the other five still looked healthy.
        rows = audit.live_rows()
        present = {row.state for row in rows}
        self.assertEqual(present, set(audit.LIVE_STATES))
        self.assertGreater(len(rows), 100, "the live set is ~188 rows")


class IdGrepPatternTests(unittest.TestCase):
    """The ID match must be a whole-token match, never a prefix match."""

    def test_pattern_matches_the_id_as_a_whole_token(self):
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "MODEL-MM",
            "feat(mm): MODEL-MM decoder lands",
            "closes MODEL-MM.",
            "(MODEL-MM) golden captured",
        ):
            self.assertTrue(pattern.search(message), message)

    def test_pattern_rejects_a_longer_id_that_merely_starts_with_it(self):
        # 55 pairs of live row IDs are prefixes of longer ones. A substring
        # match would credit MODEL-MM with MODEL-MM-voxtral's commits, and the
        # classifier calls any commit LANDED -- so an abandoned row would
        # report as finished, the exact false negative this tool prevents.
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        for message in (
            "feat(mm): MODEL-MM-voxtral audio tower lands",
            "MODEL-MM-QWEN3VL golden captured",
            "record(mm): MODEL-MM_SUFFIX bookkeeping",
        ):
            self.assertIsNone(pattern.search(message), message)


class ClassifierTests(unittest.TestCase):
    def test_unmerged_branch_commits_mean_in_flight(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=[],
            head_commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/ENG-FOO", reason)

    def test_fully_merged_branch_means_landed(self):
        verdict, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": []},
            commits=[],
            head_commits=[],
        )
        self.assertEqual(verdict, "LANDED")
        # "unmerged" contains "merged", so a substring check on the bare word
        # would pass for the IN-FLIGHT reason too.
        self.assertIn("fully merged", reason)

    def test_main_commits_without_branch_mean_landed(self):
        verdict, reason = audit.classify_active(
            branches=[],
            unmerged_by_branch={},
            commits=["def5678 feat(eng): ENG-FOO"],
            head_commits=[],
        )
        self.assertEqual(verdict, "LANDED")
        self.assertIn("def5678", reason)

    def test_no_evidence_at_all_means_abandoned(self):
        verdict, reason = audit.classify_active(
            branches=[], unmerged_by_branch={}, commits=[], head_commits=[]
        )
        self.assertEqual(verdict, "ABANDONED")
        self.assertIn("no branch", reason.lower())

    def test_in_flight_wins_over_landed_when_both_present(self):
        # A row can have landed groundwork AND active follow-up work.
        # Claiming it is finished would silently steal an open claim.
        verdict, _ = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["abc1234 wip"]},
            commits=["def5678 feat(eng): ENG-FOO groundwork"],
            head_commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")

    def test_reason_names_only_the_branches_with_unmerged_commits(self):
        # With more than one branch, the reason must name the live one and not
        # the merged one, and must be order-independent so a re-run does not
        # produce a spuriously different report.
        verdict, reason = audit.classify_active(
            branches=["row/B-LIVE", "row/A-MERGED"],
            unmerged_by_branch={"row/B-LIVE": ["abc1234 wip"], "row/A-MERGED": []},
            commits=[],
            head_commits=[],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertIn("row/B-LIVE", reason)
        self.assertNotIn("row/A-MERGED", reason)

    def test_reason_is_order_independent(self):
        # The mixed case above has exactly ONE live branch, so sorted() is a
        # no-op there and deleting it survives. Two branches on the same side
        # of the filter are what pin determinism, on both reason paths: a
        # report that reshuffles its own evidence between runs cannot be
        # diffed by the human who has to act on it.
        for label, by_branch in [
            ("in-flight", {"row/B": ["abc1234 wip"], "row/A": ["def5678 wip"]}),
            ("landed", {"row/B": [], "row/A": []}),
        ]:
            with self.subTest(label):
                forward = audit.classify_active(["row/A", "row/B"], by_branch, [], [])
                reverse = audit.classify_active(["row/B", "row/A"], by_branch, [], [])
                self.assertEqual(forward, reverse)
                self.assertIn("row/A, row/B", forward[1])

    def test_missing_branch_key_is_a_loud_caller_bug(self):
        # Silently treating an ungathered branch as merged would report a live
        # claim as finished -- the exact false negative this tool prevents.
        with self.assertRaises(KeyError):
            audit.classify_active(
                branches=["row/NEVER-GATHERED"],
                unmerged_by_branch={},
                commits=[],
                head_commits=[],
            )

    def test_every_verdict_is_declared(self):
        for branches, by_branch, commits, head in [
            (["row/X"], {"row/X": ["a b"]}, [], []),
            (["row/X"], {"row/X": []}, [], []),
            ([], {}, ["a b"], []),
            ([], {}, [], ["a b"]),
            ([], {}, [], []),
        ]:
            verdict, _ = audit.classify_active(branches, by_branch, commits, head)
            self.assertIn(verdict, audit.VERDICTS)


class HeadEvidenceClassifierTests(unittest.TestCase):
    """HEAD carries the work of the pull request being built (#726).

    A fork's `row/<ID>` branch is a ref `origin` can never hold, so on the PR
    that actually does the work HEAD -- the merge commit -- is the ONLY place
    the row's commits exist in the checkout. Without this arm every PR that
    moves a row to ACTIVE and carries its implementation reads ABANDONED.
    """

    def test_head_commits_alone_mean_in_flight(self):
        verdict, reason = audit.classify_active(
            branches=[],
            unmerged_by_branch={},
            commits=[],
            head_commits=["abc1234 feat(eng): ENG-FOO lands"],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        # The reason must say WHERE, or a reader cannot tell this apart from
        # branch evidence when chasing why a row was not flagged.
        self.assertIn("HEAD", reason)
        self.assertIn("abc1234", reason)

    def test_head_evidence_beats_a_landed_main_commit(self):
        # Same rule the branch arm already obeys: a row with landed groundwork
        # and follow-up work in THIS pull request is in flight, not finished.
        verdict, reason = audit.classify_active(
            branches=[],
            unmerged_by_branch={},
            commits=["def5678 feat(eng): ENG-FOO groundwork"],
            head_commits=["abc1234 feat(eng): ENG-FOO follow-up"],
        )
        self.assertEqual(verdict, "IN-FLIGHT")
        self.assertNotIn("def5678", reason)

    def test_head_evidence_beats_a_fully_merged_branch(self):
        # Ordering pin: putting the HEAD arm AFTER the merged-branch LANDED arm
        # still passes every test above, and reports a row whose follow-up is
        # in this very PR as finished.
        verdict, _ = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": []},
            commits=[],
            head_commits=["abc1234 feat(eng): ENG-FOO follow-up"],
        )
        self.assertEqual(verdict, "IN-FLIGHT")

    def test_a_live_branch_still_outranks_head(self):
        # Both are IN-FLIGHT, so the verdict cannot discriminate; the REASON
        # must, and the branch is the more specific evidence.
        _, reason = audit.classify_active(
            branches=["row/ENG-FOO"],
            unmerged_by_branch={"row/ENG-FOO": ["9999999 wip"]},
            commits=[],
            head_commits=["abc1234 feat(eng): ENG-FOO follow-up"],
        )
        self.assertIn("row/ENG-FOO", reason)
        self.assertNotIn("HEAD", reason)

    def test_a_pull_request_carrying_no_evidence_rescues_nothing(self):
        # The whole point of gathering HEAD evidence PER ROW. If the arm keyed
        # on "this checkout is ahead of main" instead of "a commit names this
        # row", every unrelated PR would silently clear every abandoned row and
        # the gate would never fire again.
        verdict, _ = audit.classify_active(
            branches=[], unmerged_by_branch={}, commits=[], head_commits=[]
        )
        self.assertEqual(verdict, "ABANDONED")


class PartialGapTests(unittest.TestCase):
    def test_explicit_gap_language_is_recognised(self):
        for text in [
            "Works for bf16; fp8 is missing",
            "Prefill only, decode not yet ported",
            "Dense path supported, MoE unsupported",
            "Image works; audio pending",
        ]:
            self.assertTrue(audit.names_missing_modes(text), text)

    def test_row_without_gap_language_is_flagged(self):
        self.assertFalse(audit.names_missing_modes("Ported and gated on GB10"))

    def test_detection_is_case_insensitive(self):
        self.assertTrue(audit.names_missing_modes("FP8 IS MISSING"))

    def test_markers_match_whole_words_not_substrings(self):
        # "commonly" contains "only" and "node" contains "no". A substring
        # match would mark these rows explicit and hide them from review.
        self.assertFalse(audit.names_missing_modes("Commonly used decode node"))
        # Both halves need their marker pinned as live, or the assertFalse
        # above passes for the wrong reason: dropping "no" from GAP_MARKERS
        # entirely also stops "node" matching, and nothing would notice.
        self.assertTrue(audit.names_missing_modes("Decode only"))
        self.assertTrue(audit.names_missing_modes("No fp8 path"))

    def test_flag_is_advisory_and_never_gates(self):
        # check mode fails only on abandoned ACTIVE rows, never on a vague
        # PARTIAL row -- the detector is a keyword heuristic.
        self.assertNotIn("PARTIAL", audit.CHECK_FAILS_ON)

    def test_check_fails_on_active_and_nothing_else(self):
        # assertNotIn above passes for frozenset() -- a check mode that fails
        # on NOTHING -- and even for the bare string "ACTIVE", since "PARTIAL"
        # is not a substring of it. Neither is what "report-only" means: the
        # flag must be excluded from a set that still gates something. Pin the
        # membership exactly, and pin that the gated state is a real live one.
        self.assertEqual(audit.CHECK_FAILS_ON, frozenset({"ACTIVE"}))
        self.assertTrue(audit.CHECK_FAILS_ON <= audit.LIVE_STATES)

    def test_matched_marker_names_the_marker_that_fired(self):
        # The heuristic under-flags: 11 of the 48 shipped rows it reads as
        # explicit qualify only via bare "no" or "gap", on prose asserting
        # GOODNESS rather than absence. Naming the hit is what lets a reviewer
        # discount those at a glance instead of trusting the verdict.
        self.assertEqual(
            audit.matched_marker("Works for bf16; fp8 is missing"), "missing"
        )
        self.assertEqual(
            audit.matched_marker("the mirror build no longer double-resides"), "no"
        )
        self.assertEqual(audit.matched_marker("max gap 0.0 nats, 0 divergent"), "gap")
        # The text AS WRITTEN, not the canonical marker, so the reviewer reads
        # the row's own words back.
        self.assertEqual(audit.matched_marker("FP8 IS MISSING"), "MISSING")
        # No hit is "", never None: a vague row must not be reported through
        # the same falsy channel as a row whose marker failed to render.
        self.assertEqual(audit.matched_marker("Ported and gated on GB10"), "")

    def test_a_marker_needing_escaping_is_treated_literally(self):
        # GAP_MARKERS invites human tuning, and an unescaped marker fails two
        # ways. "not.yet" compiles to a wildcard that also matches "notXyet",
        # silently widening the flag...
        literal = audit.gap_pattern(("not.yet",))
        self.assertTrue(literal.search("decode not.yet ported"))
        self.assertIsNone(literal.search("decode notXyet ported"))
        # ...and "fp4(" raises re.error at IMPORT time, taking the whole
        # module -- loader, classifier and all -- down with it.
        audit.gap_pattern(("fp4(",))
        # Escaping must not cost the multi-word widening: re.escape("not yet")
        # is "not\\ yet", and that escaped space is what gets widened.
        self.assertTrue(audit.gap_pattern(("not yet",)).search("decode not  yet"))
        # The shipped regex must be the one this builder returns. Mutation
        # testing shows what this does NOT buy: rebuilding GAP_RE inline
        # WITHOUT re.escape survives, because no shipped marker needs escaping
        # today, so both spellings compile to the identical pattern. It pins
        # the marker set and the structure, not the escaping -- and that
        # unobservability is exactly why gap_pattern takes its markers as an
        # argument instead of closing over GAP_MARKERS.
        self.assertEqual(
            audit.GAP_RE.pattern, audit.gap_pattern(audit.GAP_MARKERS).pattern
        )


class ReportTests(unittest.TestCase):
    RECORDS = [
        {
            "id": "ENG-FOO",
            "state": "ACTIVE",
            "path": ".agents/engine-matrix.md",
            "line": 42,
            "verdict": "ABANDONED",
            "reason": "no branch, no commit on main mentioning the row ID",
            "flag": "",
        },
        {
            "id": "MODEL-BAR",
            "state": "PARTIAL",
            "path": ".agents/model-matrix.md",
            "line": 7,
            "verdict": "",
            "reason": "",
            "flag": "does not name its missing modes",
        },
    ]

    def test_markdown_lists_every_record(self):
        out = audit.render_markdown(self.RECORDS)
        self.assertIn("ENG-FOO", out)
        self.assertIn("MODEL-BAR", out)
        self.assertIn("ABANDONED", out)
        self.assertIn("does not name its missing modes", out)

    def test_markdown_cells_do_not_break_the_table(self):
        records = [dict(self.RECORDS[0], reason="a | b")]
        out = audit.render_markdown(records)
        body = [ln for ln in out.splitlines() if "ENG-FOO" in ln]
        self.assertEqual(len(body), 1)
        self.assertNotIn("a | b", body[0])

    def test_check_mode_fails_when_an_active_row_is_abandoned(self):
        self.assertEqual(audit.exit_code(self.RECORDS, check=True), 1)

    def test_check_mode_passes_when_no_active_row_is_abandoned(self):
        clean = [dict(self.RECORDS[0], verdict="IN-FLIGHT")] + self.RECORDS[1:]
        self.assertEqual(audit.exit_code(clean, check=True), 0)

    def test_only_the_vague_flag_counts_as_needing_review(self):
        # Every PARTIAL row carries a flag: the marker that fired, or the vague
        # string. Counting non-empty flags would report all 68 as vague.
        explicit = dict(self.RECORDS[1], flag="explicit via 'missing'")
        self.assertNotEqual(explicit["flag"], audit.VAGUE_FLAG)
        self.assertEqual(self.RECORDS[1]["flag"], audit.VAGUE_FLAG)

    def test_report_mode_always_exits_zero(self):
        self.assertEqual(audit.exit_code(self.RECORDS, check=False), 0)

    def test_vague_partial_alone_never_fails_check_mode(self):
        only_flag = [self.RECORDS[1]]
        self.assertEqual(audit.exit_code(only_flag, check=True), 0)


class SummaryTests(unittest.TestCase):
    """The counting the summary line actually does, not just the constant.

    test_only_the_vague_flag_counts_as_needing_review above compares two
    literals and never calls the code: rewriting the summary to count every
    non-empty flag leaves it green while the report claims all 68 PARTIAL rows
    need review instead of the ~20 that do. This drives the real counter.
    """

    VAGUE = {
        "id": "MODEL-VAGUE",
        "state": "PARTIAL",
        "path": ".agents/model-matrix.md",
        "line": 7,
        "verdict": "",
        "reason": "",
        "flag": "does not name its missing modes",
        "duplicate": "",
    }
    EXPLICIT = dict(VAGUE, id="MODEL-EXPLICIT", flag="explicit via 'missing'")

    def run_main(self, records: list[dict], argv: list[str]) -> tuple[int, str]:
        buffer = io.StringIO()
        with mock.patch.object(audit, "audit", lambda: records):
            with contextlib.redirect_stdout(buffer):
                code = audit.main(argv)
        return code, buffer.getvalue()

    def test_summary_counts_only_the_vague_partial_rows(self):
        code, out = self.run_main([self.VAGUE, self.EXPLICIT], [])
        self.assertEqual(code, 0)
        self.assertIn("2 live rows; 0 abandoned ACTIVE", out)
        self.assertIn("1 PARTIAL rows to review", out)

    def test_summary_names_the_ids_living_in_two_matrices(self):
        both = dict(self.VAGUE, id="BACKEND-CPU", duplicate="backend-matrix.md:12")
        _, out = self.run_main([both, self.EXPLICIT], [])
        self.assertIn("1 IDs live in two matrices: BACKEND-CPU", out)

    def test_json_mode_emits_every_record_and_no_report_table(self):
        code, out = self.run_main([self.VAGUE, self.EXPLICIT], ["--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [self.VAGUE, self.EXPLICIT])
        self.assertNotIn("PARTIAL rows to review", out)


class AuditGuardTests(unittest.TestCase):
    """audit() must abort rather than emit a quietly wrong census."""

    def test_origin_main_is_verified_before_any_row_is_read(self):
        # git() maps every failure to "", so an unfetched origin/main makes
        # each row look ABANDONED and the audit would propose downgrading all
        # 54 ACTIVE rows at once. The guard has to fire first, not eventually.
        calls: list[str] = []

        def guard() -> None:
            calls.append("guard")
            raise SystemExit("origin/main does not resolve")

        def rows(errors=None):
            calls.append("rows")
            return []

        with mock.patch.object(audit, "require_origin_main", guard), mock.patch.object(
            audit, "live_rows", rows
        ):
            with self.assertRaises(SystemExit):
                audit.audit()
        self.assertEqual(calls, ["guard"])

    def test_a_row_that_fails_to_parse_aborts_the_audit(self):
        # parse_claim_rows DROPS a row it cannot parse. A census whose whole
        # point is completeness must not quietly return one row short.
        def broken(errors=None):
            if errors is not None:
                errors.append(".agents/engine-matrix.md:9: ENG-X has 4 cells")
            return []

        with mock.patch.object(
            audit, "require_origin_main", lambda: None
        ), mock.patch.object(audit, "live_rows", broken):
            with self.assertRaises(SystemExit) as caught:
                audit.audit()
        self.assertIn("ENG-X", str(caught.exception))


class HeadCommitGatheringTests(unittest.TestCase):
    """What `head_commits` actually asks git."""

    @staticmethod
    def _capture(output: str) -> tuple[list, list]:
        calls: list = []

        def fake_git(*args: str) -> str:
            calls.append(args)
            return output

        return calls, [fake_git]

    def test_it_asks_only_for_the_range_not_yet_on_main(self):
        # `origin/main..HEAD`, never bare HEAD: on the push-to-main lane HEAD
        # IS origin/main, so a bare HEAD would re-report every landed commit as
        # in-flight work and no ACTIVE row could ever be reported stale.
        calls, (fake_git,) = self._capture("abc1234 feat(eng): ENG-FOO\n")
        with mock.patch.object(audit, "git", fake_git):
            got = audit.head_commits("ENG-FOO")
        self.assertEqual(got, ["abc1234 feat(eng): ENG-FOO"])
        self.assertEqual(len(calls), 1)
        self.assertIn("origin/main..HEAD", calls[0])

    def test_it_matches_the_id_as_a_whole_token_like_the_main_arm(self):
        # 55 pairs of live row IDs are prefixes of longer ones. The HEAD arm
        # returns IN-FLIGHT, so a prefix match here would clear an abandoned
        # MODEL-MM because the PR happened to touch MODEL-MM-voxtral.
        calls, (fake_git,) = self._capture("")
        with mock.patch.object(audit, "git", fake_git):
            audit.head_commits("MODEL-MM")
        grep = [arg for arg in calls[0] if arg.startswith("--grep=")]
        self.assertEqual(grep, ["--grep=" + audit.id_grep_pattern("MODEL-MM")])
        self.assertIn("-E", calls[0])
        pattern = re.compile(audit.id_grep_pattern("MODEL-MM"))
        self.assertIsNone(pattern.search("feat: MODEL-MM-voxtral lands"))

    def test_a_failed_git_call_yields_no_evidence(self):
        # git() maps failure to "". That must read as "no HEAD evidence" and
        # fall through to the other arms, never as a crash or a phantom commit.
        with mock.patch.object(audit, "git", lambda *a: ""):
            self.assertEqual(audit.head_commits("ENG-FOO"), [])


class BranchInformationGuardTests(unittest.TestCase):
    """The guard `require_origin_main` has and the branch side never had (#726).

    `row_branches()` returning {} means either "nobody holds a branch" or "this
    checkout was never told about branches". The classifier reads both as the
    first, which is how CI -- fetching `main` and nothing else -- called every
    in-flight row ABANDONED.
    """

    @staticmethod
    def _refs(*names: str):
        return lambda *args: "".join(name + "\n" for name in names)

    def test_it_raises_when_the_checkout_holds_no_row_ref(self):
        with mock.patch.object(audit, "git", self._refs("main", "origin/main")):
            with self.assertRaises(SystemExit) as caught:
                audit.require_branch_information()
        message = str(caught.exception)
        # Actionable, like the origin/main guard: name the refspec, or the
        # reader is told only that something is missing.
        self.assertIn("refs/heads/row/*", message)

    def test_a_single_remote_row_ref_is_enough(self):
        with mock.patch.object(
            audit, "git", self._refs("main", "origin/main", "origin/row/ENG-FOO")
        ):
            audit.require_branch_information()

    def test_a_local_row_branch_is_enough(self):
        with mock.patch.object(audit, "git", self._refs("row/ENG-FOO")):
            audit.require_branch_information()

    def test_head_does_not_satisfy_it(self):
        # HEAD carries evidence about the row THIS pull request advances and no
        # other, so counting it would make the guard vacuous -- it always
        # resolves -- while every other row stayed silently misclassified.
        with mock.patch.object(audit, "git", self._refs("HEAD", "main")):
            with self.assertRaises(SystemExit):
                audit.require_branch_information()


class GuardOrderingTests(unittest.TestCase):
    def test_both_guards_run_before_any_row_is_read(self):
        # A guard that fires after the census is built still emits the wrong
        # census on the way there, and `live_rows` is the expensive step.
        calls: list[str] = []

        with mock.patch.object(
            audit, "require_origin_main", lambda: calls.append("origin")
        ), mock.patch.object(
            audit, "require_branch_information", lambda: calls.append("branches")
        ), mock.patch.object(
            audit, "live_rows", lambda errors=None: calls.append("rows") or []
        ):
            audit.audit()
        self.assertEqual(calls[:2], ["origin", "branches"])
        self.assertEqual(calls, ["origin", "branches", "rows"])


class AuditWiringTests(unittest.TestCase):
    """audit() must actually hand the HEAD evidence to the classifier.

    Adding the arm to `classify_active` and forgetting to gather it leaves
    every classifier test above green and #726 entirely unfixed.
    """

    def _row(self, item_id: str):
        return audit.record.ClaimRow(
            path=audit.record.AGENTS / "engine-matrix.md",
            line_no=9,
            item_id=item_id,
            state="ACTIVE",
            header=(),
            cells=(),
            raw="",
        )

    def test_head_evidence_reaches_the_verdict(self):
        with mock.patch.object(
            audit, "require_origin_main", lambda: None
        ), mock.patch.object(
            audit, "require_branch_information", lambda: None
        ), mock.patch.object(
            audit, "live_rows", lambda errors=None: [self._row("ENG-FOO")]
        ), mock.patch.object(
            audit, "row_branches", dict
        ), mock.patch.object(
            audit, "main_commits", lambda item_id: []
        ), mock.patch.object(
            audit, "head_commits", lambda item_id: ["abc1234 feat(eng): ENG-FOO"]
        ):
            records = audit.audit()
        self.assertEqual([r["verdict"] for r in records], ["IN-FLIGHT"])
        self.assertEqual(audit.exit_code(records, check=True), 0)

    def test_without_head_evidence_the_same_row_still_fails_the_check(self):
        # The paired half: the fix must not make --check unable to fail. This
        # is the same row, same absent branch, same absent main commit -- only
        # the HEAD evidence is gone.
        with mock.patch.object(
            audit, "require_origin_main", lambda: None
        ), mock.patch.object(
            audit, "require_branch_information", lambda: None
        ), mock.patch.object(
            audit, "live_rows", lambda errors=None: [self._row("ENG-FOO")]
        ), mock.patch.object(
            audit, "row_branches", dict
        ), mock.patch.object(
            audit, "main_commits", lambda item_id: []
        ), mock.patch.object(
            audit, "head_commits", lambda item_id: []
        ):
            records = audit.audit()
        self.assertEqual([r["verdict"] for r in records], ["ABANDONED"])
        self.assertEqual(audit.exit_code(records, check=True), 1)


class DuplicateLiveIdTests(unittest.TestCase):
    @staticmethod
    def _row(item_id: str, matrix: str, line_no: int):
        return audit.record.ClaimRow(
            path=audit.record.AGENTS / matrix,
            line_no=line_no,
            item_id=item_id,
            state="PARTIAL",
            header=(),
            cells=(),
            raw="",
        )

    def test_only_ids_live_in_more_than_one_matrix_are_reported(self):
        # A row ID living in two matrices would mint two issues for one item
        # and report the same item twice with identical evidence. Every other
        # ID must stay out of the result, or the report calls all 186 duplicates.
        dupes = audit.duplicate_live_ids(
            [
                self._row("BACKEND-CPU", "backend-matrix.md", 12),
                self._row("BACKEND-CPU", "feature-matrix.md", 40),
                self._row("ENG-SOLO", "engine-matrix.md", 5),
            ]
        )
        self.assertEqual(list(dupes), ["BACKEND-CPU"])
        self.assertEqual(
            dupes["BACKEND-CPU"], ["backend-matrix.md:12", "feature-matrix.md:40"]
        )


class GateWiringTests(unittest.TestCase):
    def test_preflight_runs_the_audit_suite(self):
        # The INVOCATION, not the substring. `audit-live-rows.py` also appears
        # in the explanatory comment above the gate, so asserting the bare name
        # stays green with the `run` line DELETED -- a test that passes with its
        # subject removed, guarding the one thing this task exists to install.
        text = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        self.assertIn("test_audit_live_rows", text)
        self.assertIn(
            'run "audit-live-rows" python3 scripts/audit-live-rows.py --check', text
        )

    def test_ci_runs_the_gate_and_its_suite(self):
        text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/audit-live-rows.py --check", text)
        self.assertIn("tests/scripts/test_audit_live_rows.py", text)

    def test_ci_fetches_the_branch_refs_the_classifier_reads(self):
        # #726: the step fetched `main` and nothing else, so no `row/<ID>` ref
        # existed and the IN-FLIGHT verdict was UNREACHABLE -- in-flight work
        # and abandoned work produced the same verdict. With the guard in place
        # dropping this refspec no longer misreports, it aborts, so this pins
        # the step that keeps the gate able to run at all.
        text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("+refs/heads/row/*:refs/remotes/origin/row/*", text)

    def test_shipped_record_has_no_abandoned_active_row(self):
        records = audit.audit()
        stale = [r["id"] for r in records if r["verdict"] == "ABANDONED"]
        self.assertEqual(stale, [], f"stale ACTIVE rows remain: {stale}")


if __name__ == "__main__":
    unittest.main()
