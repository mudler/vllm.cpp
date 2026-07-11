"""Metric and void-propagation tests for the low-concurrency summarizer.

Source formulas: SGLang ``bench_serving.py:968-1140,1588-1675`` @ 28b095c.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from tools.bench.serve_low_common import write_json_atomic, write_jsonl_atomic
from tools.bench.summarize_serve_low import summarize_evidence, summarize_run


def _record(duration: float, *, raw_latencies: bool = True) -> dict:
    requests = 80
    output_len = 128
    value = {
        "completed": requests,
        "duration": duration,
        "errors": [""] * requests,
        "generated_texts": ["x"] * requests,
        "input_lens": [1024] * requests,
        "itls": [[0.01] * (output_len - 1) for _ in range(requests)],
        "mean_e2e_latency_ms": 1370.0,
        "median_e2e_latency_ms": 1370.0,
        "p90_e2e_latency_ms": 1370.0,
        "p99_e2e_latency_ms": 1370.0,
        "mean_tpot_ms": 10.0,
        "median_tpot_ms": 10.0,
        "p99_tpot_ms": 10.0,
        "output_lens": [output_len] * requests,
        "ttfts": [0.1] * requests,
    }
    if raw_latencies:
        value["latencies"] = [1.37] * requests
    return value


def _write_preflight(evidence: pathlib.Path, engines: tuple[str, ...]) -> None:
    passed = {
        "checkpoint_manifest": True,
        "incremental_streaming": True,
        "native_output_ids": True,
        "openai_usage": True,
        "quantization_path": True,
        "tokenizer_ids": True,
    }
    write_json_atomic(
        evidence / "preflight" / "status.json",
        {"models": {"27": {engine: passed for engine in engines}}},
    )


class SummaryTests(unittest.TestCase):
    def test_hand_computed_raw_metrics(self) -> None:
        record = {
            "completed": 2,
            "duration": 4.0,
            "errors": ["", ""],
            "generated_texts": ["abc", "abc"],
            "input_lens": [4, 4],
            "itls": [[0.2, 0.4], [0.4, 0.6]],
            "latencies": [1.0, 2.0],
            "output_lens": [3, 3],
            "ttfts": [0.2, 0.4],
        }
        result = summarize_run(record, expected_requests=2, prompt_len=4, output_len=3)
        self.assertTrue(result["binding_eligible"])
        metrics = result["metrics"]
        self.assertEqual(metrics["request_throughput"], 0.5)
        self.assertEqual(metrics["input_throughput"], 2.0)
        self.assertEqual(metrics["output_throughput"], 1.5)
        self.assertEqual(metrics["median_ttft_ms"], 300.0)
        self.assertEqual(metrics["mean_itl_ms"], 400.0)
        self.assertAlmostEqual(metrics["median_tpot_ms"], 600.0)

    def test_three_arm_ratios_and_repetition_spread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = pathlib.Path(temporary)
            durations = {"ours": 10.0, "vllm": 20.0, "sglang": 16.0}
            _write_preflight(evidence, tuple(durations))
            for engine, duration in durations.items():
                for repetition in (1, 2, 3):
                    path = evidence / "raw" / "27" / engine / f"c1-r{repetition}.jsonl"
                    write_jsonl_atomic(path, [_record(duration + repetition - 2)])
            runs, ratios = summarize_evidence(evidence, require_full_grid=False)
            self.assertEqual(len(runs["aggregates"]), 3)
            self.assertTrue(all(group["binding_eligible"] for group in runs["aggregates"]))
            output_ratio = next(
                ratio for ratio in ratios["ratios"]
                if ratio["axis"] == "output_throughput"
            )
            self.assertEqual(output_ratio["direction"], "higher")
            self.assertGreater(output_ratio["pass_normalized_ratio"], 1.0)
            e2e_ratio = next(
                ratio for ratio in ratios["ratios"]
                if ratio["axis"] == "median_e2e_latency_ms"
            )
            self.assertEqual(e2e_ratio["direction"], "lower")
            full_grid_runs, _ = summarize_evidence(evidence)
            self.assertTrue(
                all(not group["binding_eligible"] for group in full_grid_runs["aggregates"])
            )
            self.assertTrue(
                any(
                    "campaign grid is incomplete" in reason
                    for reason in full_grid_runs["aggregates"][0]["reasons"]
                )
            )

    def test_missing_pinned_raw_latency_is_loud_and_voids_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = pathlib.Path(temporary)
            _write_preflight(evidence, ("ours",))
            for repetition in (1, 2, 3):
                path = evidence / "raw" / "27" / "ours" / f"c1-r{repetition}.jsonl"
                write_jsonl_atomic(path, [_record(10.0, raw_latencies=False)])
            runs, _ = summarize_evidence(evidence, require_full_grid=False)
            group = runs["aggregates"][0]
            self.assertFalse(group["binding_eligible"])
            self.assertIsNone(group["metrics"]["p90_tpot_ms"])
            self.assertTrue(
                any(
                    "lacks per-request latencies" in reason
                    for reason in group["reasons"]
                )
            )


if __name__ == "__main__":
    unittest.main()
