"""The draw survey's judgement, gated on the CPU.

WHY THESE ARE THE TESTS
-----------------------
The harness this covers answers #2750, #2751 and #2752 from ONE leased run, and
every way it can be wrong is a way in which the run still LOOKS finished:

* a draw phase that logged zero `[VT_GEMM_ALGO]` lines produces a perfectly
  consistent, perfectly empty stability report that reads as `STABLE` to anyone
  who does not count the keys;
* a "draw" that LOADED a plan map rather than tuning one is a copy of an earlier
  draw wearing a new label, and N copies of one draw report as N draws;
* a scoring leg that re-tuned measured a plan map nobody recorded, and its
  number is then attributed to the arm it was asked about;
* a `[[:space:]]` metric regex matches nothing in Python, so every leg records
  VOID while the runner exits 0.

So the assertions below are mostly NEGATIVE: each one deletes or inverts a
guarantee and requires the verdict to change. A predicate whose mutation leaves
the verdict green measures nothing.

The module under test is standard library only and has no device, which is why
the whole battery runs here rather than under a lease.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import inspect
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Sequence

import tools.bench.gemm_tactic_draw_survey as survey

from tools.bench.gemm_tactic_draw_survey import (
    EXIT_ALGO_KEYSET_DIFFERS,
    EXIT_ARM_MIXED,
    EXIT_ALGO_NO_BF16,
    EXIT_ALGO_SILENT,
    EXIT_BINARY_DIFFERS,
    EXIT_CACHE_MISSING,
    EXIT_DRAW_FAILED,
    EXIT_DRAW_NOT_INDEPENDENT,
    EXIT_FINGERPRINT_DIFFERS,
    EXIT_FP4_SILENT,
    EXIT_KEYSET_DIFFERS,
    EXIT_LEG_NOT_FROZEN,
    EXIT_OK,
    algo_key,
    algo_selection,
    algo_stability,
    check_draw_preconditions,
    check_frozen_leg,
    draw_identity,
    kv_tokens,
    main,
    parse_algo_lines,
    parse_autotune_lines,
    parse_bench_report,
    read_draw_records,
    run_draw,
    reduce_evidence,
    select_shipping_draw,
    selection_time_spread,
    speed_spread,
)


def quiet_main(argv: list[str]) -> int:
    """Run the CLI with its progress lines swallowed.

    The per-draw progress print is deliberate on a lease -- a job that says
    nothing for forty minutes is indistinguishable from a hung one -- and it is
    equally deliberately not wanted in a unit suite's output.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        return main(argv)


ALGO_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt m=1 n=3072 k=2048 a=bf16 b=bf16 c=bf16 "
    "epilogue=rowmajor-NN algoId=21 tile=15 stages=4 splitK=1 wsSize=4194304"
)


# ---------------------------------------------------------------------------
# `[VT_GEMM_ALGO]` HAS SIX EMIT SHAPES AND ONLY ONE OF THEM IS A SELECTION.
#
# Every one is behind the same `GemmAlgoLogEnabled()` flag, so a run that turns
# the instrument on gets all six. Byte-exact from the format strings, each cited
# at the `std::cerr` that writes it:
#
#   src/vt/cuda/cuda_matmul.cu:270  the SELECTION  (MaybeLogGemmAlgo)
#   src/vt/cuda/cuda_matmul.cu:805  REFUSED        (MaybeLogFp8PlanRefusal)
#   src/vt/cuda/cuda_matmul.cu:842  DENSE-CANDIDATES, query failed
#   src/vt/cuda/cuda_matmul.cu:858  DENSE-CANDIDATE, one per ranked candidate
#   src/vt/cuda/cuda_matmul.cu:883  CANDIDATES, query failed (fp8)
#   src/vt/cuda/cuda_matmul.cu:900  CANDIDATE rank=, one per ranked candidate (fp8)
#
# The dense candidate dump is reached from the TN-bt lane
# (`src/vt/cuda/cuda_matmul.cu:582`), which is the bf16 lane #2750 is about, so
# these lines are not a hypothetical shape in some other workload -- they are in
# the same stderr as the selections this harness compares.
# ---------------------------------------------------------------------------
SELECTION_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt m=1 n=3072 k=2048 a=bf16 b=bf16 c=bf16 "
    "epilogue=TN-bt algoId=21 tile=15 stages=4 splitK=1 wsSize=4194304"
)
REFUSED_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt REFUSED m=1 n=3072 k=2048 scale_mode=0 "
    "reason=no-heuristic pointerModeCapMask=0"
)
DENSE_CANDIDATES_FAILED_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt DENSE-CANDIDATES lane=TN-bt m=1 n=3072 "
    "k=2048 query failed: CUBLAS_STATUS_INTERNAL_ERROR"
)
DENSE_CANDIDATE_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt DENSE-CANDIDATE lane=TN-bt rank=0/6 m=1 "
    "n=3072 k=2048 algoId=31 tile=20 stages=6 splitK=1 wsSize=0 waves=1.5"
)
FP8_CANDIDATES_FAILED_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt CANDIDATES m=1 n=3072 k=2048 scale_mode=0 "
    "query failed: CUBLAS_STATUS_NOT_SUPPORTED"
)
FP8_CANDIDATE_LINE = (
    "[VT_GEMM_ALGO] backend=cublasLt CANDIDATE rank=1/8 m=1 n=3072 k=2048 "
    "scale_mode=0 algoId=41 tile=22 stages=3 splitK=2 wsSize=1024 waves=2.0"
)
DIAGNOSTIC_LINES = (
    REFUSED_LINE,
    DENSE_CANDIDATES_FAILED_LINE,
    DENSE_CANDIDATE_LINE,
    FP8_CANDIDATES_FAILED_LINE,
    FP8_CANDIDATE_LINE,
)
SELECTION_KEY = "cublasLt|m=1 n=3072 k=2048|a=bf16 b=bf16 c=bf16|TN-bt"


class AlgoLineShapeTest(unittest.TestCase):
    """Five of the six shapes are DIAGNOSTICS and none of them is a selection.

    Reading one as a selection is not a cosmetic parse error. The candidate
    dumps repeat one shape once per ranked candidate, so a first-wins map keeps
    RANK 0 -- and rank 0 of a heuristic LIST is not what the matmul ran. Two
    processes whose candidate lists came back in a different order would then
    report `UNSTABLE`, which is the verdict that, per
    `.agents/specs/cublaslt-cross-process-algo-stability.md`, turns #2750 into a
    benchmark-validity repair and licenses building a persistent cuBLASLt cache
    that #2750 says must be built only in its case 4.
    """

    def test_only_the_selection_line_is_read_as_a_selection(self) -> None:
        parsed = parse_algo_lines("\n".join((SELECTION_LINE,) + DIAGNOSTIC_LINES))
        self.assertEqual(sorted(k for k in parsed if not k.startswith("_")), [SELECTION_KEY])

    def test_each_diagnostic_shape_on_its_own_yields_no_selection(self) -> None:
        for line in DIAGNOSTIC_LINES:
            with self.subTest(shape=line.split()[2]):
                parsed = parse_algo_lines(line)
                self.assertEqual([k for k in parsed if not k.startswith("_")], [])

    def test_a_diagnostic_never_becomes_a_synthetic_unknown_key(self) -> None:
        # The old shape produced `...|a=? b=? c=?|?`, one key holding whatever
        # the first unparsed line happened to say.
        parsed = parse_algo_lines("\n".join(DIAGNOSTIC_LINES))
        self.assertNotIn("cublasLt|m=1 n=3072 k=2048|a=? b=? c=?|?", parsed)

    def test_a_candidate_dump_cannot_manufacture_an_unstable_verdict(self) -> None:
        # The SELECTION agrees across both processes and only the candidate
        # LIST moved. That is not an unstable selection.
        one = parse_algo_lines(SELECTION_LINE + "\n" + DENSE_CANDIDATE_LINE)
        two = parse_algo_lines(
            SELECTION_LINE + "\n" + DENSE_CANDIDATE_LINE.replace("algoId=31", "algoId=57")
        )
        result = algo_stability({"draw00": one, "draw01": two})
        self.assertEqual(result["verdict"], "STABLE")
        self.assertEqual(result["keys_common"], 1)

    def test_diagnostics_do_not_inflate_the_duplicate_count(self) -> None:
        # `_duplicates` is the report's only witness for `LogOncePerKey`
        # misbehaving. Five candidate lines sharing one shape are not that.
        parsed = parse_algo_lines("\n".join((SELECTION_LINE,) + DIAGNOSTIC_LINES))
        self.assertNotIn("_duplicates", parsed)

    def test_a_repeated_selection_line_is_still_counted_as_a_duplicate(self) -> None:
        parsed = parse_algo_lines(SELECTION_LINE + "\n" + SELECTION_LINE)
        self.assertEqual(parsed["_duplicates"]["count"], "1")

    def test_the_skipped_diagnostics_are_counted_rather_than_dropped_silently(self) -> None:
        # A run whose instrument emitted ONLY candidate dumps has zero
        # selections, and "zero lines" and "zero selections among 40 lines" are
        # different diagnoses of a failed run.
        parsed = parse_algo_lines("\n".join(DIAGNOSTIC_LINES))
        self.assertEqual(parsed["_diagnostics"]["count"], str(len(DIAGNOSTIC_LINES)))


class AlgoKeyTest(unittest.TestCase):
    """The dedupe key must not contain the answer.

    Mutation M10 -- folding `algoId` into `algo_key` -- survived all 60 tests of
    the first revision of this suite, because nothing called `algo_key` or
    `algo_selection` directly and the dry-run fixture held `algoId` constant
    across draws. With the id in the key a genuinely unstable run reports
    `INCOMPARABLE` (the two processes look like they saw different keys) rather
    than `UNSTABLE`, for ever, and #2750 would never be answerable at all.
    """

    BODY = SELECTION_LINE[len("[VT_GEMM_ALGO]"):]

    def moved(self) -> dict:
        return kv_tokens(
            self.BODY.replace("algoId=21", "algoId=99").replace("tile=15", "tile=20")
            .replace("stages=4", "stages=6").replace("splitK=1", "splitK=3")
        )

    def test_two_processes_that_selected_differently_still_share_one_key(self) -> None:
        self.assertEqual(algo_key(kv_tokens(self.BODY)), algo_key(self.moved()))

    def test_the_selection_tuple_is_what_moves(self) -> None:
        self.assertNotEqual(algo_selection(kv_tokens(self.BODY)), algo_selection(self.moved()))
        self.assertEqual(algo_selection(kv_tokens(self.BODY)), ("21", "15", "4", "1"))

    def test_the_key_is_the_shape_the_dedupe_uses(self) -> None:
        # `LogOncePerKey`'s key at src/vt/cuda/cuda_matmul.cu:266-268, rebuilt.
        self.assertEqual(algo_key(kv_tokens(self.BODY)), SELECTION_KEY)

    def test_the_workspace_estimate_is_not_part_of_the_selection(self) -> None:
        # The same algo reports a different wsSize under a different workspace
        # budget; folding it in would report a change that never happened.
        other = kv_tokens(self.BODY.replace("wsSize=4194304", "wsSize=8388608"))
        self.assertEqual(algo_selection(kv_tokens(self.BODY)), algo_selection(other))


class TokenizerTest(unittest.TestCase):
    def test_an_empty_value_does_not_swallow_the_next_key(self) -> None:
        # `prepared` prints two std::filesystem::path values, and an unset
        # FlashInfer path emits the bare token `flashinfer=`.
        fields = kv_tokens("mode=rw native= flashinfer= loaded=64 metadata=abc")
        self.assertEqual(fields["native"], "")
        self.assertEqual(fields["flashinfer"], "")
        self.assertEqual(fields["loaded"], "64")
        self.assertEqual(fields["metadata"], "abc")

    def test_the_parenthesised_breakdown_does_not_overwrite_the_paths(self) -> None:
        # `loaded=%llu (flashinfer=%llu native=%llu)` repeats two outer key
        # names as COUNTS. A flat last-wins map replaces the path with a number
        # and nothing says so.
        fields = kv_tokens(
            "native=/a/b.json flashinfer= loaded=64 (flashinfer=0 native=64) metadata=z"
        )
        self.assertEqual(fields["native"], "/a/b.json")
        self.assertEqual(fields["in_native"], "64")
        self.assertEqual(fields["in_flashinfer"], "0")

    def test_a_repeated_key_keeps_its_first_value(self) -> None:
        # Defensive, and stated as such: no shipped format string repeats an
        # outer key today. It is the second belt behind the `in_` prefix, so
        # that a future field added beside an existing name cannot silently
        # replace the one the parsers already read.
        self.assertEqual(kv_tokens("mode=first mode=second")["mode"], "first")

    def test_parsing_survives_a_timestamp_and_ansi_prefix(self) -> None:
        # An anchored grep over this tree's logs has failed on exactly this.
        prefixed = "\x1b[0m2026-09-03T10:00:00Z worker | " + ALGO_LINE
        self.assertEqual(len(parse_algo_lines(prefixed)), 1)


class BenchReportTest(unittest.TestCase):
    REPORT = (
        "Successful requests:                       32\n"
        "Total generated tokens:                    2048\n"
        "Total token throughput (tok/s):            1840.55\n"
        "Mean TPOT (ms):                            9.77\n"
    )

    def test_reads_the_fields_the_gate_turns_on(self) -> None:
        parsed = parse_bench_report(self.REPORT)
        self.assertEqual(parsed["total_token_throughput"], 1840.55)
        self.assertEqual(parsed["successful_requests"], 32.0)

    def test_a_missing_field_is_absent_not_zero(self) -> None:
        # Equal times are noise; equal COUNTS are identity. A request count that
        # did not parse must not read as a completed run of zero requests.
        parsed = parse_bench_report("Total token throughput (tok/s):            1.00\n")
        self.assertNotIn("successful_requests", parsed)

    def test_the_posix_bracket_class_is_not_a_python_regex(self) -> None:
        # The shell driver passes `\s+`, not `[[:space:]]+`. Python reads the
        # latter as the set {[,:,s,p,a,c,e}, which contains no space, so the
        # pattern matches NOTHING and every leg silently records VOID.
        import re
        import warnings

        line = "Total token throughput (tok/s):            1840.00"
        with warnings.catch_warnings():
            # Python itself warns "possible nested set" on the broken pattern.
            # That warning IS the defect; catching it keeps the suite's output
            # clean without hiding the assertion below it.
            warnings.simplefilter("ignore", FutureWarning)
            self.assertIsNone(
                re.search(r"Total token throughput \(tok/s\):[[:space:]]+([0-9.]+)", line)
            )
        self.assertIsNotNone(
            re.search(r"Total token throughput \(tok/s\):\s+([0-9.]+)", line)
        )


def _draw_record(
    label: str,
    *,
    tactic_offset: int = 0,
    algo_id: str = "21",
    tactic_set: str = "full",
    mean_us: float = 120.0,
) -> dict:
    algo = {}
    for n in (3072, 2048):
        key = f"cublasLt|m=1 n={n} k=2048|a=bf16 b=bf16 c=bf16|rowmajor-NN"
        algo[key] = {
            "backend": "cublasLt", "m": "1", "n": str(n), "k": "2048",
            "a": "bf16", "b": "bf16", "c": "bf16", "epilogue": "rowmajor-NN",
            "algoId": algo_id, "tile": "15", "stages": "4", "splitK": "1",
        }
    selected = {f"8,{n},2048": (n + tactic_offset) % 32 for n in (3072, 2048)}
    autotune = {
        "selections": {
            key: {"tactic_id": tactic, "name": f"tactic_{tactic}",
                  "mean_us": mean_us, "m": 8, "set": tactic_set}
            for key, tactic in selected.items()
        },
        "sets": [tactic_set],
        "repeat_selections": 0,
    }
    return {
        "label": label,
        "rc": 0,
        "tactic_set": tactic_set,
        "autotune": autotune,
        "algo": algo,
        "fp4": {
            "prepared": {"metadata": "fp1", "mode": "read-write"},
            "complete": {"loaded": "0", "tuned": str(len(selected)), "metadata": "fp1"},
            "selected": selected,
        },
        "bench": {"total_token_throughput": 1840.0},
        "cache_sha256": "abc",
        "cache_bytes": 12,
        "binary_sha256": "bin",
    }


class DrawPreconditionTest(unittest.TestCase):
    """Every case here is a run that looks finished and measured nothing."""

    def setUp(self) -> None:
        self.records = [
            _draw_record("draw00", tactic_offset=0),
            _draw_record("draw01", tactic_offset=7),
            _draw_record("draw02", tactic_offset=14),
        ]

    def code(self, mutate=None) -> int:
        records = copy.deepcopy(self.records)
        if mutate is not None:
            mutate(records)
        return check_draw_preconditions(records)[0]

    def test_the_unmutated_fixture_passes(self) -> None:
        self.assertEqual(self.code(), EXIT_OK)

    def test_a_failed_draw_process(self) -> None:
        self.assertEqual(self.code(lambda r: r[1].__setitem__("rc", 3)), EXIT_DRAW_FAILED)

    def test_zero_algo_lines_refuses_instead_of_reporting_stable(self) -> None:
        self.assertEqual(self.code(lambda r: r[2].__setitem__("algo", {})), EXIT_ALGO_SILENT)

    def test_no_bf16_input_says_nothing_about_the_bf16_row(self) -> None:
        def mutate(records):
            for fields in records[0]["algo"].values():
                fields["a"] = "f32"

        self.assertEqual(self.code(mutate), EXIT_ALGO_NO_BF16)

    def test_a_missing_fp4_complete_line(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1]["fp4"].__setitem__("complete", None)),
            EXIT_FP4_SILENT,
        )

    def test_a_draw_that_loaded_instead_of_tuning_is_a_copy(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[0]["fp4"]["complete"].__setitem__("tuned", "0")),
            EXIT_DRAW_NOT_INDEPENDENT,
        )

    def test_divergent_plan_key_sets_compare_nothing(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[2]["fp4"]["selected"].pop("8,3072,2048")),
            EXIT_KEYSET_DIFFERS,
        )

    def test_divergent_metadata_fingerprints(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1]["fp4"]["prepared"].__setitem__("metadata", "other")),
            EXIT_FINGERPRINT_DIFFERS,
        )

    def test_processes_that_saw_different_shapes(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[0]["algo"].pop(next(iter(r[0]["algo"])))),
            EXIT_ALGO_KEYSET_DIFFERS,
        )

    def test_two_binaries_are_not_one_experiment(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[2].__setitem__("binary_sha256", "deadbeef")),
            EXIT_BINARY_DIFFERS,
        )

    def test_a_draw_that_published_no_document_cannot_be_pinned(self) -> None:
        self.assertEqual(
            self.code(lambda r: r[1].__setitem__("cache_sha256", None)),
            EXIT_CACHE_MISSING,
        )


class AlgoStabilityTest(unittest.TestCase):
    def runs(self) -> dict:
        return {
            record["label"]: record["algo"]
            for record in (
                _draw_record("draw00"), _draw_record("draw01"), _draw_record("draw02")
            )
        }

    def test_one_config_per_key_across_processes_is_stable(self) -> None:
        result = algo_stability(self.runs())
        self.assertEqual(result["verdict"], "STABLE")
        self.assertEqual(result["keys_bf16_input"], 2)

    def test_one_moved_algo_id_makes_it_unstable(self) -> None:
        runs = self.runs()
        key = sorted(runs["draw01"])[0]
        runs["draw01"][key]["algoId"] = "99"
        result = algo_stability(runs)
        self.assertEqual(result["verdict"], "UNSTABLE")
        self.assertIn(key, result["unstable_keys"])

    def test_a_moved_tile_alone_is_also_unstable(self) -> None:
        # The selection is the four-tuple, not the id: two algos can share an id
        # and differ in tile/stages/splitK.
        runs = self.runs()
        runs["draw02"][sorted(runs["draw02"])[0]]["tile"] = "20"
        self.assertEqual(algo_stability(runs)["verdict"], "UNSTABLE")

    def test_a_key_missing_from_one_run_is_incomparable_not_stable(self) -> None:
        runs = self.runs()
        runs["draw02"].pop(sorted(runs["draw02"])[0])
        result = algo_stability(runs)
        self.assertEqual(result["verdict"], "INCOMPARABLE")
        self.assertEqual(len(result["keys_partial"]), 1)

    def test_one_process_cannot_answer_a_cross_process_question(self) -> None:
        self.assertEqual(
            algo_stability({"draw00": self.runs()["draw00"]})["verdict"], "INCOMPARABLE"
        )

    def test_no_runs_at_all(self) -> None:
        self.assertEqual(algo_stability({})["verdict"], "INCOMPARABLE")

    def test_the_verdict_carries_its_own_scope(self) -> None:
        # Four shapes are four shapes. The report must not read as a claim about
        # the cuBLASLt heuristic.
        self.assertIn("not a claim", algo_stability(self.runs())["scope"])


class DuplicateWitnessTest(unittest.TestCase):
    """`within_process_duplicate_lines` is the report's only witness for
    `LogOncePerKey` misbehaving, and it was structurally unable to fire: the
    only production caller stripped every `_`-prefixed key before
    `algo_stability` could see `_duplicates`, so the field was hard-wired to 0
    in every report this harness will ever write.

    A duplicate also means the parser KEPT THE FIRST line and dropped the rest,
    so the cross-process comparison would run over a first-wins subset of what
    the process actually said. That is not a stable run with a footnote; it is a
    run whose evidence has a hole in it.
    """

    def evidence_with_duplicates(self, tmp: str, count: int) -> pathlib.Path:
        evidence = pathlib.Path(tmp) / "ev"
        for index, label in enumerate(("draw00", "draw01")):
            record = _draw_record(label, tactic_offset=index * 7)
            if count and label == "draw00":
                record["algo"]["_duplicates"] = {"count": str(count)}
            home = evidence / "draws" / label
            home.mkdir(parents=True, exist_ok=True)
            (home / "record.json").write_text(json.dumps(record), encoding="utf-8")
        return evidence

    def test_a_duplicate_count_reaches_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = reduce_evidence(
                self.evidence_with_duplicates(tmp, 3),
                metric="total_token_throughput", ratification_bar=1.02,
            )
            self.assertEqual(code, EXIT_OK)
            block = report["issue_2750_draw_processes"]
            self.assertEqual(block["within_process_duplicate_lines"], 3)

    def test_a_duplicate_refuses_a_verdict_over_a_first_wins_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, report = reduce_evidence(
                self.evidence_with_duplicates(tmp, 3),
                metric="total_token_throughput", ratification_bar=1.02,
            )
            block = report["issue_2750_draw_processes"]
            self.assertEqual(block["verdict"], "INCOMPARABLE")
            self.assertIn("LogOncePerKey", block["reason"])

    def test_the_same_evidence_without_duplicates_is_stable(self) -> None:
        # The control. Without it the case above cannot tell "the duplicate
        # refused it" from "this fixture never read STABLE".
        with tempfile.TemporaryDirectory() as tmp:
            _, report = reduce_evidence(
                self.evidence_with_duplicates(tmp, 0),
                metric="total_token_throughput", ratification_bar=1.02,
            )
            block = report["issue_2750_draw_processes"]
            self.assertEqual(block["verdict"], "STABLE")
            self.assertEqual(block["within_process_duplicate_lines"], 0)


class SilentInstrumentDiagnosisTest(unittest.TestCase):
    def test_zero_selections_among_many_diagnostics_says_so(self) -> None:
        # A run whose instrument emitted only candidate dumps has no selection
        # to compare, and "the flag never reached MaybeLogGemmAlgo" is a
        # different repair from "the flag reached it and no cuBLASLt GEMM ran".
        records = [_draw_record("draw00"), _draw_record("draw01", tactic_offset=7)]
        records[0]["algo"] = {"_diagnostics": {"count": "40"}}
        code, problems = check_draw_preconditions(records)
        self.assertEqual(code, EXIT_ALGO_SILENT)
        self.assertIn("40", problems[0])


class DrawIdentityTest(unittest.TestCase):
    def test_identical_draws_share_every_key(self) -> None:
        one = _draw_record("draw00")["fp4"]["selected"]
        result = draw_identity({"draw00": dict(one), "draw01": dict(one)})
        self.assertEqual(result["pairwise_shared_min"], result["keys"])
        self.assertEqual(result["keys_with_multiple_tactics"], 0)

    def test_disjoint_draws_share_none(self) -> None:
        result = draw_identity(
            {
                "draw00": _draw_record("draw00", tactic_offset=0)["fp4"]["selected"],
                "draw01": _draw_record("draw01", tactic_offset=7)["fp4"]["selected"],
            }
        )
        self.assertEqual(result["pairwise_shared_max"], 0)

    def test_different_key_sets_are_incomparable(self) -> None:
        a = _draw_record("draw00")["fp4"]["selected"]
        b = dict(a)
        b.pop(sorted(b)[0])
        self.assertEqual(draw_identity({"a": a, "b": b})["verdict"], "INCOMPARABLE")

    def test_one_draw_is_incomparable(self) -> None:
        one = _draw_record("draw00")["fp4"]["selected"]
        self.assertEqual(draw_identity({"draw00": one})["verdict"], "INCOMPARABLE")


class SpeedSpreadTest(unittest.TestCase):
    def test_a_gap_inside_the_repeat_spread_is_not_a_gap(self) -> None:
        result = speed_spread({"a": [100.0, 100.2, 100.4], "b": [100.2, 100.1, 100.3]})
        self.assertEqual(result["verdict"], "EQUIVALENT")

    def test_a_gap_above_the_spread_but_under_the_bar(self) -> None:
        result = speed_spread({"a": [100.0, 100.02, 100.05], "b": [101.0, 101.02, 101.05]})
        self.assertEqual(result["verdict"], "SEPARATED_BELOW_BAR")

    def test_a_gap_over_the_bar_escalates_rather_than_selecting(self) -> None:
        result = speed_spread({"a": [100.0, 100.02, 100.05], "b": [105.0, 105.02, 105.05]})
        self.assertEqual(result["verdict"], "ABOVE_BAR")
        self.assertGreater(result["ratio"], 1.02)

    def test_the_bar_is_a_parameter_and_moving_it_moves_the_verdict(self) -> None:
        legs = {"a": [100.0, 100.02, 100.05], "b": [101.0, 101.02, 101.05]}
        self.assertEqual(speed_spread(legs, ratification_bar=1.005)["verdict"], "ABOVE_BAR")

    def test_one_leg_per_draw_cannot_separate_them(self) -> None:
        # With no repeat there is no within-draw spread to compare against, so a
        # difference cannot be told from the noise it might be.
        self.assertEqual(speed_spread({"a": [100.0], "b": [105.0]})["verdict"], "INCOMPARABLE")

    def test_two_legs_per_draw_is_one_difference_and_not_an_estimate(self) -> None:
        # A within-draw spread over n=2 is a single difference. The run's noise
        # floor cannot be estimated from it, and the spec's own gate command
        # used to ask for exactly two.
        result = speed_spread({"a": [100.0, 100.4], "b": [100.2, 100.1]})
        self.assertEqual(result["verdict"], "INCOMPARABLE")
        self.assertIn("Three legs per draw", result["reason"])

    def test_one_draw_is_incomparable(self) -> None:
        self.assertEqual(speed_spread({"a": [100.0, 100.05, 100.1]})["verdict"], "INCOMPARABLE")


class SpeedNoiseFloorTest(unittest.TestCase):
    """The noise floor cannot be the single worst draw in the run.

    The `EQUIVALENT` branch used the worst WITHIN-draw pair anywhere in the run
    as the floor, at `--score-reps 2`, where each within-draw figure is one
    difference over two points. One unrelated noisy draw then certified every
    other draw as equivalent -- and `EQUIVALENT` is the verdict that closes
    #2751 as "no divergence warranted" AND unblocks #2752 to pin a draw.

    This is not an exotic case on this host. One unchanged binary has read 36.82
    and 78.86 tok/s at c8 here (`.agents/specs/c8-measurement-admissibility.md`,
    #2154), so a large within-draw spread is the expected reading of a box that
    did not hold still, not a rarity.
    """

    def test_one_noisy_draw_cannot_certify_the_others_as_equivalent(self) -> None:
        # ~5.9% within `a`, and a 3% draw-to-draw gap -- half again the 1.02x
        # ratification bar -- which the old floor read as EQUIVALENT.
        result = speed_spread({"a": [100.0, 103.0, 106.0], "b": [100.0, 100.0, 100.0]})
        self.assertEqual(result["verdict"], "INCOMPARABLE")
        self.assertIn("ceiling", result["reason"])
        self.assertGreater(result["worst_within_draw_spread"], 1.02)

    def test_a_run_over_the_ceiling_ships_nothing(self) -> None:
        # The consequence, stated where #2752 reads it.
        result = speed_spread({"a": [100.0, 103.0, 106.0], "b": [100.0, 100.0, 100.0]})
        self.assertIsNone(select_shipping_draw(result, ["a", "b"])["ship"])

    def test_the_ceiling_is_a_named_parameter(self) -> None:
        legs = {"a": [100.0, 103.0, 106.0], "b": [100.0, 100.0, 100.0]}
        loosened = speed_spread(legs, noise_ceiling=1.10)
        self.assertNotEqual(loosened["verdict"], "INCOMPARABLE")

    def test_the_floor_is_pooled_over_the_draws_and_not_the_worst_one(self) -> None:
        # Two silent draws and one that is merely restless but still admissible.
        # Against the WORST draw the 0.83% gap reads EQUIVALENT and #2752
        # unblocks; against the pooled estimate it is a separation below the bar
        # and nothing ships.
        legs = {
            "a": [100.0, 100.0, 100.0],
            "b": [100.0, 100.0, 100.0],
            "c": [100.0, 101.5, 101.5],
        }
        result = speed_spread(legs)
        self.assertEqual(result["verdict"], "SEPARATED_BELOW_BAR")
        self.assertEqual(result["pooled_within_draw_spread"], 1.0)
        self.assertAlmostEqual(result["worst_within_draw_spread"], 1.015)
        self.assertIsNone(select_shipping_draw(result, ["a", "b", "c"])["ship"])

    def test_an_equivalent_run_still_reads_equivalent(self) -> None:
        # The control. Without it every case above passes on a predicate that
        # can only ever refuse.
        result = speed_spread({"a": [100.0, 100.2, 100.4], "b": [100.2, 100.1, 100.3]})
        self.assertEqual(result["verdict"], "EQUIVALENT")


class ShippingRuleTest(unittest.TestCase):
    ORDER = ["draw00", "draw01", "draw02"]

    def test_equivalent_draws_ship_the_first_in_draw_order(self) -> None:
        spread = speed_spread({"a": [100.0, 100.2, 100.4], "b": [100.2, 100.1, 100.3]})
        result = select_shipping_draw(spread, self.ORDER)
        self.assertEqual(result["ship"], "draw00")

    def test_the_rule_is_not_the_fastest_draw(self) -> None:
        # The whole point of #2752's refusal: a draw picked BY its own speed on
        # the workload it will be scored on is measuring around the harness.
        spread = speed_spread(
            {"draw00": [100.0, 100.2, 100.4], "draw01": [100.5, 100.2, 100.3]}
        )
        result = select_shipping_draw(spread, self.ORDER)
        self.assertEqual(result["ship"], "draw00")
        self.assertNotEqual(result["ship"], spread["best_draw"])

    def test_separated_draws_ship_nothing(self) -> None:
        spread = speed_spread({"a": [100.0, 100.02, 100.05], "b": [105.0, 105.02, 105.05]})
        self.assertIsNone(select_shipping_draw(spread, self.ORDER)["ship"])

    def test_below_bar_but_separated_also_ships_nothing(self) -> None:
        spread = speed_spread({"a": [100.0, 100.02, 100.05], "b": [101.0, 101.02, 101.05]})
        self.assertIsNone(select_shipping_draw(spread, self.ORDER)["ship"])


class FrozenLegTest(unittest.TestCase):
    OK = (
        "[VT_FP4_CACHE] complete mode=read-only loaded=64 tuned=0 rejected=0 "
        "saved=0 selected=64 metadata=x"
    )

    def test_a_frozen_replay_passes(self) -> None:
        ok, _ = check_frozen_leg(self.OK, 64)
        self.assertTrue(ok)

    def test_a_leg_that_retuned_is_refused(self) -> None:
        ok, why = check_frozen_leg(self.OK.replace("tuned=0", "tuned=3"), 64)
        self.assertFalse(ok)
        self.assertIn("re-tuned", why)

    def test_a_partial_install_is_refused(self) -> None:
        ok, _ = check_frozen_leg(self.OK.replace("loaded=64", "loaded=60"), 64)
        self.assertFalse(ok)

    def test_a_leg_that_announced_a_selection_is_refused(self) -> None:
        # The SECOND witness. `tuned=0` is the runtime's count; this is the
        # tuner's own voice, and a control with one witness cannot be
        # cross-checked.
        text = self.OK + (
            "\n[VT_FP4_AUTOTUNE] set=full M=8(bucket=8) N=3072 K=2048 device=0 "
            "sm=121 delay_us=5000 -> id=17 t (123.4 us), workspace=0"
        )
        ok, why = check_frozen_leg(text, 64)
        self.assertFalse(ok)
        self.assertIn("announced", why)

    def test_a_leg_with_no_complete_line_is_refused(self) -> None:
        ok, why = check_frozen_leg("nothing ran", 64)
        self.assertFalse(ok)
        self.assertIn("did not run", why)


class FrozenControlRecordTest(unittest.TestCase):
    """The frozen control has to REACH the report, and `reduce` has to refuse.

    `reduce_evidence` read `score/frozen-checks.json`, which nothing in this
    repository ever wrote -- so `reduce` had no path to exit 78 at all, and a
    `REPORT.json` could carry `EQUIVALENT` plus "ship draw00" with no record
    that any scoring leg had been frozen. The frozen control is the only thing
    that keeps a scoring leg from re-tuning, which would make the scoring phase
    N MORE DRAWS wearing the label of a replay, each number attributed to a plan
    map nobody recorded.

    One file per leg, read with a glob: a single shared control file would be a
    surface every leg has to write, which the protocol names as a lock.
    """

    OK_LOG = (
        "[VT_FP4_CACHE] complete mode=read-only loaded=2 tuned=0 rejected=0 "
        "saved=0 selected=2 metadata=fp1"
    )

    def evidence(self, tmp: str, *, legs: int = 3, controls: int | None = None,
                 failing: bool = False) -> pathlib.Path:
        """Two draws, `legs` scoring legs each, and their per-leg controls."""

        root = pathlib.Path(tmp) / "ev"
        arms = ("draw00", "draw01")
        for index, label in enumerate(arms):
            home = root / "draws" / label
            home.mkdir(parents=True, exist_ok=True)
            (home / "record.json").write_text(
                json.dumps(_draw_record(label, tactic_offset=index * 7)), encoding="utf-8"
            )
        ledger = root / "score" / "legs.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for leg in range(legs):
            for arm_index, arm in enumerate(arms):
                rows.append(json.dumps({
                    "arm": arm, "boot_id": "b", "rc": 0,
                    "total_token_throughput": 100.0 + arm_index * 0.1 + leg * 0.05,
                }))
        ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")

        written = legs * len(arms) if controls is None else controls
        count = 0
        for leg in range(legs):
            for arm in arms:
                if count >= written:
                    break
                home = root / "score" / f"{arm}-{leg + 1}"
                home.mkdir(parents=True, exist_ok=True)
                (home / "frozen.json").write_text(json.dumps({
                    "leg": f"{arm}-{leg + 1}",
                    "frozen": not (failing and count == 0),
                    "why": "frozen: tuned=0 loaded=2" if not (failing and count == 0)
                           else "tuned=3: the leg re-tuned instead of replaying the frozen draw",
                    "expected_plans": 2,
                }), encoding="utf-8")
                count += 1
        return root

    def reduce(self, root: pathlib.Path):
        return reduce_evidence(
            root, metric="total_token_throughput", ratification_bar=1.02
        )

    def test_a_complete_control_reaches_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = self.reduce(self.evidence(tmp))
            self.assertEqual(code, EXIT_OK)
            control = report["frozen_leg_control"]
            self.assertEqual(control["state"], "PASS")
            self.assertEqual(control["legs_in_ledger"], 6)
            self.assertEqual(control["legs_with_a_passing_control"], 6)
            self.assertIn(report["issue_2751_speed"]["verdict"],
                          ("EQUIVALENT", "SEPARATED_BELOW_BAR", "ABOVE_BAR"))

    def test_a_ledger_with_no_control_at_all_refuses_78(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = self.reduce(self.evidence(tmp, controls=0))
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)
            self.assertEqual(report["frozen_leg_control"]["state"], "REFUSED")

    def test_a_leg_that_re_tuned_refuses_78(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = self.reduce(self.evidence(tmp, failing=True))
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)
            self.assertIn("re-tuned", json.dumps(report["frozen_leg_control"]))

    def test_a_leg_without_a_control_record_refuses_78(self) -> None:
        # Five controls for six legs: one leg contributed a number that no
        # control ever covered.
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = self.reduce(self.evidence(tmp, controls=5))
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)

    def test_a_refused_control_ships_nothing_and_reports_no_speed_verdict(self) -> None:
        # The whole point: without this, `EQUIVALENT` + "ship draw00" could be
        # written over legs that re-tuned.
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self.reduce(self.evidence(tmp, controls=0))
            self.assertEqual(report["issue_2751_speed"]["verdict"], "REFUSED")
            self.assertIsNone(report["issue_2752"]["ship"])

    def test_no_ledger_needs_no_control(self) -> None:
        # The identity half stands on its own; a draw-only root is not refused
        # for lacking a control over legs that were never run.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.evidence(tmp)
            (root / "score" / "legs.jsonl").unlink()
            code, report = self.reduce(root)
            self.assertEqual(code, EXIT_OK)
            self.assertEqual(report["issue_2751_speed"]["verdict"], "NOT RUN")

    def test_check_frozen_writes_its_own_per_leg_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "leg.log"
            log.write_text(self.OK_LOG, encoding="utf-8")
            record = pathlib.Path(tmp) / "frozen.json"
            rc = quiet_main([
                "check-frozen", "--log", str(log), "--expected-plans", "2",
                "--record", str(record), "--leg", "draw00-1",
            ])
            self.assertEqual(rc, EXIT_OK)
            written = json.loads(record.read_text(encoding="utf-8"))
            self.assertTrue(written["frozen"])
            self.assertEqual(written["leg"], "draw00-1")

    def test_a_refused_leg_still_writes_a_record_saying_so(self) -> None:
        # A control that only writes itself on success is a control whose
        # failure looks like a leg that never ran.
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "leg.log"
            log.write_text(self.OK_LOG.replace("tuned=0", "tuned=3"), encoding="utf-8")
            record = pathlib.Path(tmp) / "frozen.json"
            rc = quiet_main([
                "check-frozen", "--log", str(log), "--expected-plans", "2",
                "--record", str(record), "--leg", "draw00-1",
            ])
            self.assertEqual(rc, EXIT_LEG_NOT_FROZEN)
            written = json.loads(record.read_text(encoding="utf-8"))
            self.assertFalse(written["frozen"])
            self.assertIn("re-tuned", written["why"])

    def test_the_noise_ceiling_reaches_the_speed_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.evidence(tmp)
            _, report = reduce_evidence(
                root, metric="total_token_throughput", ratification_bar=1.02,
                noise_ceiling=1.05,
            )
            self.assertEqual(report["issue_2751_speed"]["noise_ceiling"], 1.05)


class AutotuneSelectionParseTest(unittest.TestCase):
    """The tuner's own selection line, which is a DIAGNOSTIC and not an axis."""

    LINE = (
        "[VT_FP4_AUTOTUNE] set=full M=8(bucket=8) N=3072 K=2048 device=0 sm=121 "
        "delay_us=5000 -> id=17 sm121a_bf16_128x128 (123.4 us), workspace=4194304"
    )

    def test_reads_the_bucket_keyed_selection(self) -> None:
        parsed = parse_autotune_lines(self.LINE)
        self.assertEqual(parsed["sets"], ["full"])
        entry = parsed["selections"]["8,3072,2048"]
        self.assertEqual(entry["tactic_id"], 17)
        self.assertEqual(entry["mean_us"], 123.4)

    def test_the_key_joins_the_selected_plan_map(self) -> None:
        # `[VT_FP4_CACHE] selected` prints plan.m_bucket under the name `M`, so
        # the two maps must key identically or the diagnostic joins nothing.
        record = _draw_record("draw00")
        self.assertEqual(
            set(record["autotune"]["selections"]), set(record["fp4"]["selected"])
        )

    def test_a_tactic_name_with_spaces_is_read_whole(self) -> None:
        line = self.LINE.replace("sm121a_bf16_128x128", "baseline tactic name")
        entry = parse_autotune_lines(line)["selections"]["8,3072,2048"]
        self.assertEqual(entry["name"], "baseline tactic name")
        self.assertEqual(entry["mean_us"], 123.4)

    def test_a_log_prefix_does_not_hide_the_line(self) -> None:
        parsed = parse_autotune_lines("\x1b[0m2026-09-03Z pod | " + self.LINE)
        self.assertEqual(len(parsed["selections"]), 1)

    def test_a_repeated_key_is_counted_and_the_first_is_kept(self) -> None:
        # A key tuned twice is a lazy miss after the pre-serve warmup. The two
        # readings must DIFFER for this to discriminate: two identical lines
        # cannot tell "keep the first" from "keep the last", which is a test
        # that asserts nothing.
        second = self.LINE.replace("id=17", "id=29").replace("123.4", "456.7")
        parsed = parse_autotune_lines(self.LINE + "\n" + second)
        self.assertEqual(parsed["repeat_selections"], 1)
        self.assertEqual(len(parsed["selections"]), 1)
        entry = parsed["selections"]["8,3072,2048"]
        self.assertEqual(entry["tactic_id"], 17)
        self.assertEqual(entry["mean_us"], 123.4)

    def test_the_w1_arm_is_reported_under_its_own_name(self) -> None:
        parsed = parse_autotune_lines(self.LINE.replace("set=full", "set=w1"))
        self.assertEqual(parsed["sets"], ["w1"])


class SelectionTimeDiagnosticTest(unittest.TestCase):
    def draws(self, *means: float) -> dict:
        return {
            f"draw{i:02d}": _draw_record(f"draw{i:02d}", mean_us=mean)["autotune"]["selections"]
            for i, mean in enumerate(means)
        }

    def test_it_reports_a_state_and_never_a_verdict(self) -> None:
        # STRUCTURAL GUARD. `select_shipping_draw` reads `verdict`; this block
        # deliberately has none, so a selection-time result cannot be handed to
        # the shipping rule and be mistaken for an end-to-end one.
        result = selection_time_spread(self.draws(120.0, 130.0))
        self.assertNotIn("verdict", result)
        self.assertEqual(result["state"], "DIAGNOSTIC")
        self.assertIsNone(select_shipping_draw(result, ["draw00", "draw01"])["ship"])

    def test_it_carries_the_reason_it_cannot_gate(self) -> None:
        result = selection_time_spread(self.draws(120.0, 130.0))
        self.assertIn("per-iteration", result["not_a_gate"])

    def test_it_reports_the_per_key_ratio_and_the_distinct_ids(self) -> None:
        result = selection_time_spread(self.draws(100.0, 110.0))
        self.assertAlmostEqual(result["max_over_min_max"], 1.1)
        self.assertEqual(result["keys"], 2)

    def test_one_draw_is_incomparable(self) -> None:
        self.assertEqual(selection_time_spread(self.draws(120.0))["state"], "INCOMPARABLE")


class TacticSetArmTest(unittest.TestCase):
    def test_one_root_may_not_hold_two_arms(self) -> None:
        # The arms select by different rules: 32 candidates by pure argmin
        # versus 4 behind a >1% stickiness damper. A pooled spread names no rule.
        records = [
            _draw_record("draw00", tactic_set="full"),
            _draw_record("draw01", tactic_set="w1", tactic_offset=7),
        ]
        code, problems = check_draw_preconditions(records)
        self.assertEqual(code, EXIT_ARM_MIXED)
        self.assertIn("DIFFERENT RULES", problems[0])

    def test_a_record_without_an_arm_is_refused(self) -> None:
        # An unset VT_FP4_FULL_TACTICS is default-ON, so a draw whose arm was
        # never recorded is a draw nobody can attribute afterwards.
        records = [_draw_record("draw00"), _draw_record("draw01", tactic_offset=7)]
        records[1].pop("tactic_set")
        self.assertEqual(check_draw_preconditions(records)[0], EXIT_ARM_MIXED)

    def test_a_single_arm_root_passes(self) -> None:
        records = [
            _draw_record("draw00", tactic_set="w1"),
            _draw_record("draw01", tactic_set="w1", tactic_offset=7),
        ]
        self.assertEqual(check_draw_preconditions(records)[0], EXIT_OK)


class DryRunEndToEndTest(unittest.TestCase):
    """The record / resume / refuse path, with no device and no subprocess."""

    def test_draw_records_resume_and_reduce(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            rc = quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "3", "--dry-run",
            ])
            self.assertEqual(rc, EXIT_OK)
            first = read_draw_records(evidence)
            self.assertEqual([r["label"] for r in first], ["draw00", "draw01", "draw02"])

            # A resumed run replays what is done and only adds what is owed.
            stamp = (evidence / "draws" / "draw00" / "record.json").stat().st_mtime_ns
            rc = quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "5", "--dry-run",
            ])
            self.assertEqual(rc, EXIT_OK)
            self.assertEqual(len(read_draw_records(evidence)), 5)
            self.assertEqual(
                (evidence / "draws" / "draw00" / "record.json").stat().st_mtime_ns, stamp
            )

            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_OK)
            # The fixture must never be mistakable for a measurement.
            self.assertTrue(report["dry_run"])
            # THE FIXTURE MOVES ONE SHAPE'S SELECTION WITH THE DRAW INDEX, and
            # holds the other three still. A fixture in which every draw agreed
            # let mutation M10 -- folding `algoId` into `algo_key` -- survive all
            # 60 tests of this suite's first revision: with the id constant, a
            # tainted key still agreed. With one shape moving, the id in the key
            # makes that shape look like a DIFFERENT key per draw, and the
            # verdict collapses to INCOMPARABLE instead of naming the shape.
            stability = report["issue_2750_draw_processes"]
            self.assertEqual(stability["verdict"], "UNSTABLE")
            self.assertEqual(stability["keys_common"], 4)
            self.assertEqual(stability["keys_partial"], [])
            self.assertEqual(len(stability["unstable_keys"]), 1)
            self.assertEqual(stability["within_process_duplicate_lines"], 0)
            self.assertEqual(report["tactic_set"], ["full"])
            self.assertEqual(report["issue_2751_selection_time"]["state"], "DIAGNOSTIC")
            self.assertEqual(report["issue_2751_speed"]["verdict"], "NOT RUN")
            self.assertIsNone(report["issue_2752"]["ship"])
            self.assertEqual(report["clock_windows"]["state"], "ABSENT")

    def test_reduce_refuses_an_empty_evidence_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, report = reduce_evidence(
                pathlib.Path(tmp), metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertNotEqual(code, EXIT_OK)
            self.assertEqual(report["verdict"], "REFUSED")

    def test_a_scoring_ledger_reaches_the_speed_and_shipping_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            quiet_main([
                "draw", "--evidence", str(evidence), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "2", "--dry-run",
            ])
            ledger = evidence / "score" / "legs.jsonl"
            ledger.parent.mkdir(parents=True, exist_ok=True)
            ledger.write_text(
                "\n".join(
                    json.dumps({"arm": arm, "boot_id": "b", "rc": 0,
                                "total_token_throughput": value})
                    for arm, value in (
                        ("draw00", 100.0), ("draw01", 105.0),
                        ("draw00", 100.1), ("draw01", 105.1),
                        ("draw00", 100.05), ("draw01", 105.05),
                    )
                ) + "\n",
                encoding="utf-8",
            )
            # A ledger ALONE reaches no speed verdict: the frozen control is
            # what says these legs replayed a draw rather than drawing again.
            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)
            self.assertEqual(report["issue_2751_speed"]["verdict"], "REFUSED")

            for arm in ("draw00", "draw01"):
                for index in (1, 2, 3):
                    home = evidence / "score" / f"{arm}-{index}"
                    home.mkdir(parents=True, exist_ok=True)
                    (home / "frozen.json").write_text(
                        json.dumps({"leg": f"{arm}-{index}", "frozen": True,
                                    "why": "frozen: tuned=0 loaded=8"}),
                        encoding="utf-8",
                    )
            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_OK)
            self.assertEqual(report["issue_2751_speed"]["verdict"], "ABOVE_BAR")
            self.assertIsNone(report["issue_2752"]["ship"])


DRAW_CFG = {
    "num_prompts": 4, "input_len": 8, "output_len": 4, "concurrency": 1,
    "seed": 0, "max_num_batched_tokens": 64,
}


class DrawResumeTest(unittest.TestCase):
    """The two ways this harness loses a crashed run's work.

    `dgx.casa` has crashed roughly hourly under a long ladder (#545) and the
    whole shape of this driver -- per-phase markers, per-draw DONE files, a
    mirrored evidence root -- exists to survive that. Both cases below defeated
    it while the header claimed per-draw resume.
    """

    def test_a_draw_mirrors_as_it_lands_and_not_at_the_end_of_the_phase(self) -> None:
        # The share is what survives the box going down. Mirroring after the
        # whole `draw` subcommand returns loses EVERY completed draw to a crash
        # inside the phase, which is the phase that takes hours.
        with tempfile.TemporaryDirectory() as tmp:
            local = pathlib.Path(tmp) / "local"
            share = pathlib.Path(tmp) / "share"
            record = run_draw(
                0, local, pathlib.Path("/bin/true"), "/nonexistent", DRAW_CFG,
                dry_run=True, mirror=share,
            )
            self.assertEqual(record["rc"], 0)
            self.assertTrue((share / "draws" / "draw00" / "record.json").is_file())
            self.assertTrue((share / "draws" / "draw00" / "DONE").is_file())

    def test_the_draw_loop_mirrors_every_draw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local = pathlib.Path(tmp) / "local"
            share = pathlib.Path(tmp) / "share"
            rc = quiet_main([
                "draw", "--evidence", str(local), "--bench", "/bin/true",
                "--model", "/nonexistent", "--draws", "3", "--dry-run",
                "--mirror", str(share),
            ])
            self.assertEqual(rc, EXIT_OK)
            for label in ("draw00", "draw01", "draw02"):
                self.assertTrue((share / "draws" / label / "record.json").is_file(), label)

    def test_a_retried_draw_does_not_load_the_document_it_published(self) -> None:
        # A draw that published its cache and then died before writing DONE
        # leaves the document behind. On the retry the fresh process LOADS it,
        # tunes nothing, and the judge refuses with `73` -- whose message says
        # the draw is "a copy of an earlier draw", which misdescribes a crash.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            home = evidence / "draws" / "draw00"
            home.mkdir(parents=True)
            cache = home / "autotune_configs.json"
            cache.write_text('{"plans": ["a stale draw"]}\n', encoding="utf-8")
            seen: dict[str, bool] = {}

            def fake_run(command, **kwargs):
                seen["cache_present"] = cache.exists()
                return subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(survey.subprocess, "run", fake_run):
                run_draw(0, evidence, pathlib.Path("/bin/true"), "/model", DRAW_CFG)
            self.assertFalse(seen["cache_present"])

    def test_a_completed_draw_is_replayed_and_not_re_run(self) -> None:
        # The control on the unlink above: a draw with its DONE file must not
        # lose the document it published.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "ev"
            run_draw(0, evidence, pathlib.Path("/bin/true"), "/m", DRAW_CFG, dry_run=True)
            cache = evidence / "draws" / "draw00" / "autotune_configs.json"
            before = cache.read_bytes()
            calls: list[Any] = []
            with mock.patch.object(survey.subprocess, "run", lambda *a, **k: calls.append(a)):
                run_draw(0, evidence, pathlib.Path("/bin/true"), "/m", DRAW_CFG)
            self.assertEqual(calls, [])
            self.assertEqual(cache.read_bytes(), before)


class FrozenControlJoinTest(unittest.TestCase):
    """The control has to answer for THE LEGS THAT CONTRIBUTED, one by one.

    `frozen_control_state` used to compare two counts: how many legs the ledger
    held, and how many passing control records existed anywhere under `score/`.
    Six passing records satisfied six ledger legs even when all six described
    ONE arm and the other arm had none -- so an arm whose legs were never
    checked read as checked, which is the docstring's own question answered
    wrongly. The shell driver writes one record per leg it ran and cannot
    produce that state today, so this is a claim the function made rather than a
    bug anyone has seen; the repair is to make the claim true instead of
    narrowing it.
    """

    def evidence_with_controls_for(self, tmp: str, arms: Sequence[str]) -> pathlib.Path:
        evidence = pathlib.Path(tmp) / "ev"
        quiet_main([
            "draw", "--evidence", str(evidence), "--bench", "/bin/true",
            "--model", "/nonexistent", "--draws", "2", "--dry-run",
        ])
        ledger = evidence / "score" / "legs.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for arm, base in (("draw00", 100.0), ("draw01", 100.5)):
            for index in range(1, 4):
                rows.append(json.dumps({"arm": arm, "boot_id": "b", "rc": 0,
                                        "total_token_throughput": base + index * 0.01}))
        ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")
        # SIX passing records, placed wherever the caller says. The COUNT is
        # always right; only the join can tell the two placements apart.
        for arm in arms:
            for index in range(1, 4):
                home = evidence / "score" / f"{arm}-{index}"
                home.mkdir(parents=True, exist_ok=True)
                (home / "frozen.json").write_text(
                    json.dumps({"leg": f"{arm}-{index}", "frozen": True,
                                "why": "frozen: tuned=0 loaded=8"}),
                    encoding="utf-8",
                )
        return evidence

    def test_six_controls_for_one_arm_do_not_answer_for_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Both blocks land under draw00, so the total is six either way.
            evidence = self.evidence_with_controls_for(tmp, ["draw00", "draw00x"])
            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)
            self.assertEqual(report["frozen_leg_control"]["state"], "REFUSED")
            self.assertIn("draw01", report["frozen_leg_control"]["reason"])
            self.assertEqual(report["issue_2751_speed"]["verdict"], "REFUSED")

    def test_the_same_six_controls_spread_over_both_arms_pass(self) -> None:
        # THE CONTROL, and the count is identical: six records, six legs. Only
        # which arm they name differs.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self.evidence_with_controls_for(tmp, ["draw00", "draw01"])
            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_OK)
            self.assertEqual(report["frozen_leg_control"]["state"], "PASS")

    def test_a_failing_record_still_refuses_by_name(self) -> None:
        # The pre-existing polarity must survive the join.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self.evidence_with_controls_for(tmp, ["draw00", "draw01"])
            record = evidence / "score" / "draw01-2" / "frozen.json"
            record.write_text(
                json.dumps({"leg": "draw01-2", "frozen": False,
                            "why": "re-tuned: tuned=8"}),
                encoding="utf-8",
            )
            code, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            self.assertEqual(code, EXIT_LEG_NOT_FROZEN)
            self.assertIn("draw01-2", report["frozen_leg_control"]["reason"])

class NonPositiveLegTest(unittest.TestCase):
    """A draw the ceiling CANNOT SEE still moved the ratio.

    `within` is `None` when a draw's minimum leg is `<= 0`, and a `None` is
    dropped before the ceiling and the pooled floor are computed. The draw's
    own mean is not dropped, so the wildest draw in the run was exempt from
    both guards while still deciding which draw was worst. That is the same
    failure the ceiling was added for, reached by the other door.

    A `0.0` leg is not hypothetical here: `parse_bench_report` reads the metric
    out of a bench report, and a leg whose run produced no tokens records the
    field as zero rather than omitting it.
    """

    # From the second-round review, verbatim. Before the repair this returned
    # ABOVE_BAR at ratio 1.999 while reporting a worst within-draw spread of
    # 1.001 -- a figure computed from `d0` alone, with `d1` silently absent.
    COUNTEREXAMPLE = {"d0": [100.0, 100.1, 100.05], "d1": [0.0, 300.0, 300.0]}

    def test_a_draw_with_a_zero_leg_is_incomparable_not_above_the_bar(self) -> None:
        verdict = speed_spread(self.COUNTEREXAMPLE)
        self.assertEqual(verdict["verdict"], "INCOMPARABLE")
        self.assertIn("d1", verdict["reason"])

    def test_the_reported_spread_says_which_draws_it_left_out(self) -> None:
        # The number that made the old verdict readable as a measurement. Both
        # spread figures are pooled over the draws that HAVE a spread, so a
        # report that does not name the others quotes 1.001 as this run's noise
        # when it is `d0`'s alone.
        verdict = speed_spread(self.COUNTEREXAMPLE)
        self.assertEqual(verdict["draws_without_a_spread"], ["d1"])

    def test_a_run_every_draw_of_which_has_a_spread_leaves_none_out(self) -> None:
        verdict = speed_spread(
            {"d0": [100.0, 100.1, 100.05], "d1": [299.9, 300.0, 300.0]}
        )
        self.assertEqual(verdict["draws_without_a_spread"], [])

    def test_a_negative_leg_is_refused_the_same_way(self) -> None:
        verdict = speed_spread(
            {"d0": [100.0, 100.1, 100.05], "d1": [-1.0, 300.0, 300.0]}
        )
        self.assertEqual(verdict["verdict"], "INCOMPARABLE")

    def test_the_same_draws_with_a_positive_minimum_still_answer(self) -> None:
        # THE CONTROL. Without it the refusal above would also pass on a
        # predicate that calls every run INCOMPARABLE.
        verdict = speed_spread(
            {"d0": [100.0, 100.1, 100.05], "d1": [299.9, 300.0, 300.0]}
        )
        self.assertEqual(verdict["verdict"], "ABOVE_BAR")

class NoiseCeilingDefaultTest(unittest.TestCase):
    """The ceiling THE LEASE RUN ACTUALLY USES, not the one a unit call passes.

    `speed_spread` takes the ceiling as a keyword, and its own default was the
    only copy under test. Two more copies existed: `reduce_evidence`'s keyword
    default, and the `reduce` subparser's `--noise-ceiling` default -- and the
    subparser's is the one the spec's published gate command exercises, because
    that command passes no `--noise-ceiling` at all. Widening either of those
    two to 1.5 left the whole suite green, so the constant governing the real
    run was the unasserted one and the three could drift apart in silence.

    The fixture below is built so the ceiling BITES: at 1.02x it refuses the run
    and at any wider value it returns a verdict. Recording the number is not
    enough -- a report can carry a ceiling that decided nothing.
    """

    # d0's own repeats span 1.03x, which is over the ceiling and under the
    # mutation. d1 is quiet, so the pooled floor cannot hide the gap either:
    # at 1.5x this fixture reads ABOVE_BAR rather than INCOMPARABLE.
    LEDGER = (("draw00", (100.0, 103.0, 101.0)), ("draw01", (105.0, 105.1, 105.05)))

    def evidence_with_a_noisy_draw(self, tmp: str) -> pathlib.Path:
        evidence = pathlib.Path(tmp) / "ev"
        quiet_main([
            "draw", "--evidence", str(evidence), "--bench", "/bin/true",
            "--model", "/nonexistent", "--draws", "2", "--dry-run",
        ])
        ledger = evidence / "score" / "legs.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for arm, values in self.LEDGER:
            for index, value in enumerate(values, start=1):
                rows.append(json.dumps({"arm": arm, "boot_id": "b", "rc": 0,
                                        "total_token_throughput": value}))
                home = evidence / "score" / f"{arm}-{index}"
                home.mkdir(parents=True, exist_ok=True)
                (home / "frozen.json").write_text(
                    json.dumps({"leg": f"{arm}-{index}", "frozen": True,
                                "why": "frozen: tuned=0 loaded=8"}),
                    encoding="utf-8",
                )
        ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return evidence

    def test_the_published_reduce_command_runs_at_the_ceiling_it_declares(self) -> None:
        # THE CLI, END TO END, WITH NO `--noise-ceiling` -- which is the shape
        # `.agents/specs/nvfp4-persistent-plan-cache.md` publishes as the gate
        # command. A test that calls `speed_spread` directly cannot see this
        # copy of the number, and that is exactly how it drifted.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self.evidence_with_a_noisy_draw(tmp)
            out = pathlib.Path(tmp) / "report.json"
            quiet_main(["reduce", "--evidence", str(evidence), "--out", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))
            speed = report["issue_2751_speed"]
            self.assertEqual(speed["noise_ceiling"], 1.02)
            self.assertEqual(speed["verdict"], "INCOMPARABLE")
            self.assertIn("noise ceiling", speed["reason"])

    def test_reduce_evidence_called_without_a_ceiling_uses_the_same_number(self) -> None:
        # The middle copy. `main` always passes `--noise-ceiling` through, so
        # `reduce_evidence`'s own default is invisible to the CLI test above and
        # needs its own caller.
        with tempfile.TemporaryDirectory() as tmp:
            evidence = self.evidence_with_a_noisy_draw(tmp)
            _, report = reduce_evidence(
                evidence, metric="total_token_throughput", ratification_bar=1.02
            )
            speed = report["issue_2751_speed"]
            self.assertEqual(speed["noise_ceiling"], 1.02)
            self.assertEqual(speed["verdict"], "INCOMPARABLE")

    def test_speed_spread_called_without_a_ceiling_uses_the_same_number(self) -> None:
        # The third copy, and the only one the suite already covered.
        verdict = speed_spread(dict(self.LEDGER), ratification_bar=1.02)
        self.assertEqual(verdict["noise_ceiling"], 1.02)
        self.assertEqual(verdict["verdict"], "INCOMPARABLE")

    def test_a_wider_ceiling_would_change_this_fixture_s_verdict(self) -> None:
        # THE FIXTURE'S OWN CONTROL. Without it the three cases above would pass
        # on a run the ceiling never decided, and the mutation they exist to
        # catch would be invisible to them too.
        verdict = speed_spread(
            dict(self.LEDGER), ratification_bar=1.02, noise_ceiling=1.5
        )
        self.assertEqual(verdict["verdict"], "ABOVE_BAR")

class ShellDriverTest(unittest.TestCase):
    """The lease-side driver, on a host with no device and no toolkit.

    Every case here is a resume shape. The share survives a crash and `/tmp`
    does not, so the driver is repeatedly asked to restart from a state where
    the two disagree.
    """

    SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/dgx-gemm-tactic-draw-survey.sh"
    SURVEY = pathlib.Path(__file__).resolve().parents[2] / "tools/bench/gemm_tactic_draw_survey.py"

    FROZEN_LOG = (
        "[VT_FP4_CACHE] complete mode=read-only loaded=2 tuned=0 rejected=0 "
        "saved=0 selected=2 metadata=fp1"
    )

    def fake_tree(
        self, tmp: str, *, with_binary: bool = True, local_evidence: bool = True
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """A resumed local root plus its share, with no compiler in sight.

        `local_evidence=False` is the wiped-`/tmp` shape: the share survived the
        box going down and `$LOCAL_ROOT` did not, which is the state `mirror_in`
        exists to restore from.
        """

        root = pathlib.Path(tmp) / "gtds"
        share = pathlib.Path(tmp) / "share"
        src = pathlib.Path(tmp) / "src"
        (src / "tools" / "bench").mkdir(parents=True)
        shutil.copy2(self.SURVEY, src / "tools" / "bench" / "gemm_tactic_draw_survey.py")

        phase = share / "phase"
        phase.mkdir(parents=True)
        (phase / "build.ok").write_text("2026-09-04T00:00:00Z\n", encoding="utf-8")
        (phase / "src.path").write_text(str(src) + "\n", encoding="utf-8")
        (phase / "toolkit.path").write_text("/nonexistent/cuda\n", encoding="utf-8")

        draw = share / "draws" / "draw00"
        draw.mkdir(parents=True)
        (draw / "autotune_configs.json").write_text('{"plans": []}\n', encoding="utf-8")
        (draw / "stderr.log").write_text(
            "[VT_FP4_CACHE] selected M=8 N=3072 K=2048 tactic=1\n"
            "[VT_FP4_CACHE] selected M=8 N=2048 K=6144 tactic=2\n",
            encoding="utf-8",
        )
        (draw / "DONE").write_text("ok\n", encoding="utf-8")

        if local_evidence:
            # The ordinary state: the draw phase ran in this job and left its
            # evidence on the local disk, mirrored to the share as it landed.
            shutil.copytree(share, root / "evidence")

        if with_binary:
            binary = root / "bin" / "vllm-bench"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(
                "#!/bin/bash\n"
                f"cat <<'LOG'\n{self.FROZEN_LOG}\n"
                "Successful requests:                       4\n"
                "Total token throughput (tok/s):            1840.55\n"
                "LOG\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            # The build stages the shared library beside the binary and points
            # LD_LIBRARY_PATH at it, so a resume that finds one without the
            # other has not got a runnable binary either.
            (binary.parent / "libvllm.so.0.9.0").write_text("not an ELF\n", encoding="utf-8")
        return root, share

    def run_driver(self, *args: str, timeout: int = 90) -> subprocess.CompletedProcess:
        """Drive the script with its output going to FILES, never to a pipe.

        The heartbeat is `( while true; do sleep 60; ...; done ) &`, and the EXIT
        trap kills the subshell without reaping the `sleep` it is blocked in.
        That orphan inherits the script's stdout, so `capture_output=True` waits
        on a pipe nobody will close for up to a minute after the driver has
        exited -- which reads exactly like a hung driver. On a lease it is
        harmless: `rc` reads the job's output and the job is already over.
        """

        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        with tempfile.TemporaryDirectory() as sink:
            out = pathlib.Path(sink) / "stdout"
            err = pathlib.Path(sink) / "stderr"
            with out.open("w") as out_fh, err.open("w") as err_fh:
                done = subprocess.run(
                    ["bash", str(self.SCRIPT), *args],
                    stdout=out_fh, stderr=err_fh, timeout=timeout, env=env,
                )
            return subprocess.CompletedProcess(
                done.args, done.returncode,
                out.read_text(encoding="utf-8", errors="replace"),
                err.read_text(encoding="utf-8", errors="replace"),
            )

    def test_the_script_parses(self) -> None:
        done = subprocess.run(["bash", "-n", str(self.SCRIPT)], capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr.decode())

    def test_a_scoring_leg_resolves_the_source_the_build_recorded(self) -> None:
        # The re-entry ran before `phase/src.path` was read, so a leg always
        # used the DEFAULT source path. Invoked without `--src` -- which the
        # driver's own `--score-leg` command line does -- `check-frozen` then
        # failed on a path that does not exist, and the failure was reported as
        # `78`, the code that means A LEG RE-TUNED.
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            done = self.run_driver(
                "--score-leg", "draw00", "--evidence", str(share), "--src", "",
                "--model", tmp, "--local-root", str(root), "--num-prompts", "4",
                "--input-len", "8", "--output-len", "4", "--concurrency", "1",
                "--seed", "0", "--max-num-batched-tokens", "64",
                "--tactic-set", "full",
            )
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("Total token throughput", done.stdout)

    def test_a_scoring_leg_writes_its_own_frozen_control_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            self.run_driver(
                "--score-leg", "draw00", "--evidence", str(share), "--src", "",
                "--model", tmp, "--local-root", str(root), "--num-prompts", "4",
                "--input-len", "8", "--output-len", "4", "--concurrency", "1",
                "--seed", "0", "--max-num-batched-tokens", "64",
                "--tactic-set", "full",
            )
            record = root / "evidence" / "score" / "draw00-1" / "frozen.json"
            self.assertTrue(record.is_file(), sorted((root / "evidence" / "score").glob("*")))
            self.assertTrue(json.loads(record.read_text(encoding="utf-8"))["frozen"])

    def test_a_surviving_build_marker_over_a_wiped_tmp_refuses_by_name(self) -> None:
        # `phase/build.ok` is mirrored to the share and `$BIN` is not, so a
        # wiped `/tmp` with a surviving share makes the driver skip the build and
        # then die on a missing binary somewhere further down. Worse, the draws
        # already in the share carry the previous binary's sha256, so rebuilding
        # into the same root would produce a SECOND binary and the judge refuses
        # that at the very end with `79`.
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(
                tmp, with_binary=False, local_evidence=False
            )
            done = self.run_driver(
                "--phase", "build", "--evidence", str(share),
                "--model", tmp, "--local-root", str(root),
            )
            self.assertEqual(done.returncode, 35, done.stdout + done.stderr)
            self.assertIn("build.ok", done.stderr)

    def test_a_build_marker_with_its_artefacts_still_skips_the_build(self) -> None:
        # The control. Without it the case above passes on a driver that can
        # never resume at all.
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            done = self.run_driver(
                "--phase", "build", "--evidence", str(share),
                "--model", tmp, "--local-root", str(root),
            )
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("skipping", done.stdout)

    # --- the driver's own constants and flags, which no Python test can see ---
    #
    # THE PYTHON SIDE IS WELL GATED AND THE SHELL DELIVERY WAS NOT. `--mirror`
    # and `--score-reps` are arguments the driver PASSES; deleting either from
    # this script left all 100 tests of the previous revision green, and the
    # consequence is only visible on a lease. That is the shape
    # AGENTS.md "Nothing lands dead" names: a gate that stays green without the
    # call site measures a class, not a capability.

    def recording_bench(self, root: pathlib.Path, share: pathlib.Path,
                        witness: pathlib.Path, src: pathlib.Path) -> None:
        """Replace the fake bench with one that says WHEN it was called.

        It emits the module's own `--dry-run` instrument fixture, so the draw
        phase gets output its real preconditions accept, and it appends one
        witness line per invocation recording what the SHARE held at that
        moment. That timestamp is the whole point: mirroring at the end of the
        phase and mirroring after each draw are indistinguishable once the
        phase has returned.
        """

        binary = root / "bin" / "vllm-bench"
        binary.write_text(
            "#!/usr/bin/env python3\n"
            "import importlib.util, json, os, pathlib, sys\n"
            f"spec = importlib.util.spec_from_file_location('s', {str(src / 'tools' / 'bench' / 'gemm_tactic_draw_survey.py')!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"witness = pathlib.Path({str(witness)!r})\n"
            f"share = pathlib.Path({str(share)!r})\n"
            "n = len(witness.read_text().splitlines()) if witness.exists() else 0\n"
            "with witness.open('a') as fh:\n"
            "    fh.write('call=%d draw00=%s draw01=%s\\n' % (\n"
            "        n + 1,\n"
            "        (share / 'draws' / 'draw00' / 'record.json').is_file(),\n"
            "        (share / 'draws' / 'draw01' / 'record.json').is_file()))\n"
            "cache = pathlib.Path(os.environ['VT_FP4_AUTOTUNE_CACHE_PATH'])\n"
            "cache.parent.mkdir(parents=True, exist_ok=True)\n"
            "cache.write_text(json.dumps({'_metadata': {'call': n + 1}, 'plans': []}) + '\\n')\n"
            "cfg = {'num_prompts': 4, 'input_len': 8, 'output_len': 4}\n"
            "out, err, rc = mod.synthetic_draw_output(n, cfg)\n"
            "sys.stdout.write(out)\n"
            "sys.stderr.write(err)\n"
            "raise SystemExit(rc)\n",
            encoding="utf-8",
        )
        binary.chmod(0o755)

    def test_the_draw_phase_mirrors_each_draw_as_it_lands(self) -> None:
        # `--mirror` makes the JUDGE copy the evidence root to the share after
        # every draw. Without it the share gets the draws only when the phase
        # returns, and a crash inside a phase that is hours of model loads --
        # on a box that has gone down four times in one session (#545) --
        # loses every completed draw. The driver's `mirror_out` after the phase
        # hides that completely once the phase has ended, so the assertion has
        # to be made from INSIDE the phase: at the third draw's invocation, the
        # second draw must already be on the share.
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            # The pre-seeded draw is a RESUME fixture. This case needs the draw
            # phase to run, so both copies of it go.
            shutil.rmtree(share / "draws")
            shutil.rmtree(root / "evidence" / "draws")
            witness = pathlib.Path(tmp) / "witness"
            self.recording_bench(root, share, witness, pathlib.Path(tmp) / "src")
            done = self.run_driver(
                "--phase", "draw", "--evidence", str(share), "--draws", "3",
                "--model", tmp, "--local-root", str(root), "--num-prompts", "4",
                "--input-len", "8", "--output-len", "4", "--concurrency", "1",
                "--seed", "0", "--max-num-batched-tokens", "64",
            )
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            lines = witness.read_text(encoding="utf-8").splitlines()
            # One preflight draw plus two the loop still owed: the loop replays
            # draw00 from its `DONE` file rather than running it again.
            self.assertEqual(len(lines), 3, lines)
            self.assertIn("draw01=True", lines[2], lines)
            # THE CONTROL. Without it this would also pass on a driver that
            # mirrored the whole share before the phase started, which is not
            # per-draw mirroring and does not survive a crash any better.
            self.assertIn("draw01=False", lines[1], lines)

    def test_every_draw_invocation_hands_the_judge_the_share_to_mirror_to(self) -> None:
        # The preflight's `--mirror` cannot be caught behaviourally: it draws
        # once and `mirror_out` runs on the very next line, so deleting it
        # changes nothing observable. It is still the same obligation -- a
        # preflight that dies mid-draw should leave its evidence on the share --
        # and this states the rule over EVERY `draw` invocation rather than
        # transcribing either line.
        text = self.SCRIPT.read_text(encoding="utf-8")
        commands, current = [], None
        for line in text.splitlines():
            if current is not None:
                current.append(line)
            elif '"$SURVEY" draw' in line:
                current = [line]
            if current is not None and not line.rstrip().endswith("\\"):
                commands.append(" ".join(current))
                current = None
        self.assertEqual(len(commands), 2, commands)
        for command in commands:
            self.assertIn('--mirror "$EV_SHARE"', command, command)

    def test_the_driver_scores_enough_legs_for_a_verdict_to_exist(self) -> None:
        # `--score-reps` decides how many legs each draw gets, and
        # `speed_spread` returns INCOMPARABLE below its `min_legs`. A driver
        # that ships fewer buys a whole GPU lease and returns no verdict at all.
        # The bound is READ FROM THE PREDICATE rather than written twice, so
        # raising `min_legs` reddens here instead of being discovered on a lease.
        min_legs = inspect.signature(survey.speed_spread).parameters["min_legs"].default
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            done = self.run_driver(
                "--phase", "build", "--evidence", str(share),
                "--model", tmp, "--local-root", str(root),
            )
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            # The PROVENANCE the driver writes carries the value it will run at,
            # which is the driver's default when no `--score-reps` is given.
            provenance = (root / "evidence" / "PROVENANCE").read_text(encoding="utf-8")
            match = re.search(r"score_reps=(\d+)", provenance)
            self.assertIsNotNone(match, provenance)
            self.assertGreaterEqual(int(match.group(1)), min_legs, provenance)

    def test_a_refused_reduce_does_not_mark_the_phase_complete(self) -> None:
        # Same polarity as the resume cases: a marker that a FAILED phase writes
        # makes the next run skip the work that did not happen.
        with tempfile.TemporaryDirectory() as tmp:
            root, share = self.fake_tree(tmp)
            shutil.rmtree(share / "draws")
            shutil.rmtree(root / "evidence" / "draws")
            done = self.run_driver(
                "--phase", "reduce", "--evidence", str(share),
                "--model", tmp, "--local-root", str(root),
            )
            self.assertNotEqual(done.returncode, 0)
            self.assertFalse((root / "evidence" / "phase" / "reduce.ok").is_file())
            self.assertFalse((share / "phase" / "reduce.ok").is_file())



class ArtefactResolutionTest(unittest.TestCase):
    """#2912: the build phase must accept a STATICALLY linked vllm-bench.

    The configuration this harness invokes emits `libvllm.a` and links
    `vllm-bench` against it, so requiring a shared `libvllm.so.*` refuses a
    tree that built green. The first real GB10 run died exactly there, after
    855/855 with BUILD_RC=0 and zero compile errors.

    These drive `--check-artefacts`, which calls the SAME `resolve_artefacts`
    the build phase calls, so this gates the production predicate rather than a
    transcription of it. The other shell tests write `phase/build.ok` to skip
    the build phase, which is why this branch shipped unreached.
    """

    SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts/dgx-gemm-tactic-draw-survey.sh"

    def _run(self, build_dir):
        return subprocess.run(
            ["bash", str(self.SCRIPT), "--evidence", "/unused", "--model",
             "/unused", "--check-artefacts", str(build_dir)],
            capture_output=True, text=True, timeout=120,
        )

    def _tree(self, name, *files):
        root = pathlib.Path(tempfile.mkdtemp()) / name
        (root / "examples").mkdir(parents=True)
        for f in files:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text("", encoding="utf-8")
        return root

    def test_a_static_build_is_accepted_and_reports_no_shared_library(self):
        # The shape the GB10 run actually produced.
        tree = self._tree("static", "examples/vllm-bench", "libvllm.a")
        done = self._run(tree)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("vllm-bench", done.stdout)
        self.assertIn("statically linked", done.stdout)

    def test_a_shared_build_is_accepted_and_names_the_library(self):
        tree = self._tree("shared", "examples/vllm-bench", "libvllm.so.0.1")
        done = self._run(tree)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("libvllm.so.0.1", done.stdout)

    def test_a_tree_without_the_binary_still_refuses_with_E_ARTEFACT(self):
        # The refusal that IS real: no binary means nothing can be measured.
        tree = self._tree("empty", "libvllm.a")
        done = self._run(tree)
        self.assertEqual(done.returncode, 35, done.stdout)
        self.assertIn("no vllm-bench", done.stderr)


if __name__ == "__main__":
    unittest.main()
