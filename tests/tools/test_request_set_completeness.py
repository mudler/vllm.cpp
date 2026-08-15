"""A throughput number may never be derived from a partial request set (#931).

The Qwen3.8-27B online-serving grid published a c1 leg reading 0.675x against
vLLM while median TPOT in the SAME result file read 1.017x in our favour. The
leg had `failed: 1` of 6: `output_throughput` divides tokens by a wall duration
that still contains the dead request, so the ratio was wrong in an
unpredictable direction. Nothing in the harness asserted `failed == 0` before
turning that record into a number.

These tests pin the guard at every site that turns a benchmark result record
into a rate, not only at the validators that happen to run first today. A
guard reachable only through a caller is a guard the next caller skips.
"""

from __future__ import annotations

import copy
import unittest

from tools.bench import gdn_packed_component, online_gate, online_gate_summary
from tools.bench import run_serve_low, summarize_serve_low, vllm_closed_loop_metrics
from tools.bench.serve_low_common import HarnessError, require_complete_request_set

INPUT_LEN = 1024
OUTPUT_LEN = 128


def _serve_record(*, requests: int = 2, duration: float = 10.0) -> dict:
    """A complete `vllm bench serve` record, in the pinned client's schema."""

    record = {
        "completed": requests,
        "duration": duration,
        "errors": [""] * requests,
        "failed": 0,
        "generated_texts": [f"text-{index}" for index in range(requests)],
        "input_lens": [INPUT_LEN] * requests,
        "itls": [[0.01] * (OUTPUT_LEN - 1) for _ in range(requests)],
        "max_concurrency": 1,
        "max_concurrent_requests": 1,
        "num_prompts": requests,
        "output_lens": [OUTPUT_LEN] * requests,
        "output_throughput": requests * OUTPUT_LEN / duration,
        "request_throughput": requests / duration,
        "start_times": [float(index) for index in range(requests)],
        "total_input_tokens": requests * INPUT_LEN,
        "total_output_tokens": requests * OUTPUT_LEN,
        "total_token_throughput": requests * (INPUT_LEN + OUTPUT_LEN) / duration,
        "ttfts": [0.1] * requests,
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = 8.0
    return record


def _one_dead_request(record: dict) -> dict:
    """Exactly the #915 shape: one request died, the rest completed.

    Only the completion counters move. Every rate field keeps the value the
    client wrote, because that is precisely the defect -- the record still
    carries a number, and the number is wrong.
    """

    partial = copy.deepcopy(record)
    partial["completed"] = record["completed"] - 1
    partial["failed"] = 1
    return partial


class RequireCompleteRequestSetTest(unittest.TestCase):
    def test_complete_record_returns_the_completed_count(self):
        self.assertEqual(
            require_complete_request_set(_serve_record(), source="unit"), 2
        )

    def test_one_failed_request_is_refused(self):
        with self.assertRaisesRegex(HarnessError, "partial"):
            require_complete_request_set(
                _one_dead_request(_serve_record()), source="unit"
            )

    def test_failed_is_refused_even_when_completed_matches(self):
        # BELT AND BRACES, and recorded as such. At the pin this record shape
        # cannot occur: `completed` counts only successes (serve.py:618) and
        # `failed = len(failed_outputs)` (:728), so `completed + failed ==
        # num_prompts` identically and a non-zero `failed` always drags
        # `completed` below `num_prompts`. The guard reads `failed` anyway,
        # because it is cheap and because the client's schema is not ours to
        # pin: a future counter that reports both independently must not be
        # able to walk past a non-zero `failed` on the strength of a matching
        # `completed`.
        record = _serve_record()
        record["failed"] = 1
        with self.assertRaisesRegex(HarnessError, "partial"):
            require_complete_request_set(record, source="unit")

    def test_short_completed_is_refused(self):
        record = _serve_record()
        record["completed"] = 1
        with self.assertRaisesRegex(HarnessError, "partial"):
            require_complete_request_set(record, source="unit")

    def test_expected_requests_overrides_a_missing_num_prompts(self):
        record = _serve_record()
        del record["num_prompts"]
        self.assertEqual(
            require_complete_request_set(record, expected_requests=2, source="unit"), 2
        )
        with self.assertRaisesRegex(HarnessError, "partial"):
            require_complete_request_set(record, expected_requests=3, source="unit")

    def test_a_record_that_cannot_prove_completeness_is_refused(self):
        # No `num_prompts`, no caller-supplied expectation: the record cannot
        # establish that anything is missing, which is not the same as
        # establishing that nothing is.
        record = _serve_record()
        del record["num_prompts"]
        with self.assertRaisesRegex(HarnessError, "cannot establish"):
            require_complete_request_set(record, source="unit")

    def test_a_non_integer_completed_is_refused(self):
        record = _serve_record()
        record["completed"] = "2"
        with self.assertRaisesRegex(HarnessError, "completed"):
            require_complete_request_set(record, source="unit")

    def test_a_recorded_request_error_is_refused(self):
        record = _serve_record()
        record["errors"] = ["", "Never received a valid chunk to calculate TTFT."]
        with self.assertRaisesRegex(HarnessError, "error"):
            require_complete_request_set(record, source="unit")

    def test_the_source_is_named_in_the_message(self):
        with self.assertRaisesRegex(HarnessError, "ours-c1-r1"):
            require_complete_request_set(
                _one_dead_request(_serve_record()), source="ours-c1-r1"
            )


class OnlineGateDerivationTest(unittest.TestCase):
    def test_validate_raw_result_still_refuses_a_partial_set(self):
        with self.assertRaisesRegex(HarnessError, "partial"):
            online_gate.validate_raw_result(
                _one_dead_request(_serve_record()),
                concurrency=1,
                expected_requests=2,
            )

    def test_run_metrics_refuses_a_partial_set(self):
        # The number is derived HERE. A guard that lives only in the caller is
        # skipped by the next caller.
        with self.assertRaisesRegex(HarnessError, "partial"):
            online_gate_summary._run_metrics(_one_dead_request(_serve_record()))

    def test_run_metrics_still_computes_a_complete_set(self):
        metrics = online_gate_summary._run_metrics(_serve_record())
        self.assertAlmostEqual(metrics["input_throughput"], 2 * INPUT_LEN / 10.0)


class GdnPackedDerivationTest(unittest.TestCase):
    def test_recompute_timing_metrics_refuses_a_partial_set(self):
        with self.assertRaisesRegex(HarnessError, "partial"):
            gdn_packed_component._recompute_timing_metrics(
                _one_dead_request(_serve_record())
            )


class ServeLowDerivationTest(unittest.TestCase):
    """The SGLang schema carries no `failed`, so `completed` is the counter."""

    @staticmethod
    def _sglang_record(*, requests: int = 2, duration: float = 10.0) -> dict:
        return {
            "completed": requests,
            "duration": duration,
            "errors": [""] * requests,
            "input_lens": [INPUT_LEN] * requests,
            "output_lens": [OUTPUT_LEN] * requests,
            "ttfts": [0.1] * requests,
            "itls": [[0.01] * (OUTPUT_LEN - 1) for _ in range(requests)],
            "generated_texts": [f"text-{index}" for index in range(requests)],
            "latencies": [1.0] * requests,
            "mean_e2e_latency_ms": 1000.0,
            "median_e2e_latency_ms": 1000.0,
            "p90_e2e_latency_ms": 1000.0,
            "p99_e2e_latency_ms": 1000.0,
            "mean_tpot_ms": 8.0,
            "median_tpot_ms": 8.0,
            "p99_tpot_ms": 8.0,
        }

    def test_summarize_run_refuses_a_short_request_set(self):
        record = self._sglang_record()
        record["completed"] = 1
        with self.assertRaisesRegex(HarnessError, "partial"):
            summarize_serve_low.summarize_run(record, expected_requests=2)

    def test_summarize_run_rate_uses_the_completed_count(self):
        summary = summarize_serve_low.summarize_run(
            self._sglang_record(), expected_requests=2
        )
        self.assertAlmostEqual(summary["metrics"]["request_throughput"], 0.2)

    def test_validate_raw_result_refuses_a_failed_request(self):
        # SGLang omits `failed`; when a record does carry it, an unexamined
        # non-zero value is exactly how #931 stayed invisible.
        record = self._sglang_record()
        record["failed"] = 1
        with self.assertRaisesRegex(HarnessError, "partial"):
            run_serve_low.validate_raw_result(
                record, expected_requests=2, prompt_len=INPUT_LEN, output_len=OUTPUT_LEN
            )


class ClosedLoopDerivationTest(unittest.TestCase):
    def test_missing_records_are_refused(self):
        with self.assertRaisesRegex(RuntimeError, "partial"):
            vllm_closed_loop_metrics.require_every_request_returned(
                records=[{}, {}], expected_requests=3
            )

    def test_a_complete_set_is_accepted(self):
        vllm_closed_loop_metrics.require_every_request_returned(
            records=[{}, {}, {}], expected_requests=3
        )

    @staticmethod
    def _closed_loop_record(*, output_len: int = OUTPUT_LEN) -> dict:
        """One returned request, in `run_closed_loop`'s record schema."""

        return {
            "arrival_s": 0.0,
            "first_token_s": 0.1,
            "last_token_s": 1.0,
            "completion_s": 1.0,
            "itls_s": [0.01] * (output_len - 1),
            "output_token_ids": list(range(output_len)),
            "core_ttft_s": 0.05,
        }

    def _derive(self, records, prompt_count: int):
        return vllm_closed_loop_metrics.derive_metrics(
            records,
            [[7] * INPUT_LEN for _ in range(prompt_count)],
            10.0,
            output_len=OUTPUT_LEN,
            max_concurrency=1,
            async_scheduling="default",
        )

    def test_derive_metrics_refuses_a_partial_record_set(self):
        # The two cases above call the guard by hand, which proves the guard
        # works and nothing about whether the tool reaches it. Deleting the
        # call from `derive_metrics` left all 328 tool tests green (#931
        # review). This is the case that goes red instead: two records
        # returned where three prompts were submitted, and without the guard
        # the tool happily divides three prompts' input tokens by a duration
        # that spans a request it never got back.
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self._derive([self._closed_loop_record()] * 2, prompt_count=3)

    def test_derive_metrics_computes_a_complete_record_set(self):
        # The complementary half: the guard must not refuse a whole set, or
        # the case above would pass against a tool that refuses everything.
        result, token_ids = self._derive(
            [self._closed_loop_record()] * 3, prompt_count=3
        )
        self.assertEqual(result["successful_requests"], 3)
        self.assertEqual(len(token_ids), 3)
        self.assertAlmostEqual(result["request_throughput"], 0.3)
        self.assertAlmostEqual(
            result["input_token_throughput"], 3 * INPUT_LEN / 10.0
        )


if __name__ == "__main__":
    unittest.main()
