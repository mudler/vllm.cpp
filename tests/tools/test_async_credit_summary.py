"""Fail-closed contracts for the vLLM async-credit evidence finalizer."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.bench.finalize_async_credit import build_summary, finalize
from tools.bench.serve_low_common import HarnessError


SOURCE_COMMIT = "1" * 40


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class AsyncCreditSummaryTests(unittest.TestCase):
    def _fixture(self, root: pathlib.Path) -> pathlib.Path:
        evidence = root / "evidence"
        (evidence / "raw").mkdir(parents=True)
        (evidence / "traces" / "on").mkdir(parents=True)
        (evidence / "traces" / "off").mkdir(parents=True)
        corpus_digest = _digest("corpus")
        for mode, base in (("on", 102.0), ("off", 100.0)):
            for repetition in (1, 2, 3):
                total = base + repetition - 2
                raw = {
                    "completed": 6,
                    "failed": 0,
                    "num_prompts": 6,
                    "max_concurrency": 2,
                    "total_input_tokens": 6144,
                    "total_output_tokens": 768,
                    "errors": [""] * 6,
                    "input_lens": [1024] * 6,
                    "output_lens": [128] * 6,
                    "generated_texts": [f"{mode}-{repetition}-{i}" for i in range(6)],
                    "request_throughput": total / 1000.0,
                    "output_throughput": total / 9.0,
                    "total_token_throughput": total,
                    "mean_ttft_ms": 800.0 if mode == "on" else 700.0,
                    "mean_tpot_ms": 105.0 if mode == "on" else 107.0,
                    "mean_itl_ms": 105.0 if mode == "on" else 107.0,
                    "mean_e2el_ms": 14_000.0 if mode == "on" else 14_100.0,
                }
                (evidence / "raw" / f"{mode}-r{repetition}.json").write_text(
                    json.dumps(raw), encoding="utf-8"
                )
            digests = [_digest(f"{mode}-{index}") for index in range(4)]
            if mode == "on":
                digests = [digests[0]] * 4
            metadata = {
                "async_scheduling_requested": mode,
                "async_scheduling_resolved": mode == "on",
                "admission_mode": "closed-loop",
                "enable_prefix_caching": False,
                "input_len": 1024,
                "output_len": 128,
                "num_prompts": 6,
                "max_concurrency": 2,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 2048,
                "profiled_warmup_prompts": 6,
                "repetitions": 3,
                "corpus_sha256": corpus_digest,
                "output_digest": digests[0],
                "output_digests": digests,
                "output_digests_equal": len(set(digests)) == 1,
            }
            (evidence / "traces" / f"{mode}-metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            trace = evidence / "traces" / mode / "trace.pt.trace.json.gz"
            trace.write_bytes(mode.encode())
            kernel_summary = {
                "kernel_count": 3 if mode == "on" else 4,
                "kernel_time_us": 30.0 if mode == "on" else 32.0,
                "selected_trace_sha256": _digest(mode),
                "kernels": [
                    {
                        "name": "shared",
                        "count": 2,
                        "total_duration_us": 20.0,
                        "percent": 66.0,
                    },
                    {
                        "name": f"{mode}-only",
                        "count": 1 if mode == "on" else 2,
                        "total_duration_us": 10.0 if mode == "on" else 12.0,
                        "percent": 34.0,
                    },
                ],
            }
            (evidence / "traces" / f"{mode}-kernel-summary.json").write_text(
                json.dumps(kernel_summary), encoding="utf-8"
            )
        (evidence / "corpus").mkdir()
        (evidence / "corpus" / "c2-r1.jsonl").write_text("{}\n", encoding="utf-8")
        return evidence

    def test_complete_series_is_summarized_without_requiring_digest_stability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = self._fixture(pathlib.Path(temporary))
            summary = build_summary(evidence, source_commit=SOURCE_COMMIT)
        total = summary["timing"]["total_token_throughput"]
        self.assertEqual(total["on_median"], 102.0)
        self.assertEqual(total["off_median"], 100.0)
        self.assertAlmostEqual(total["direction_normalized_ratio"], 1.02)
        self.assertFalse(summary["speed_credit"]["meets_minimum"])
        self.assertTrue(summary["outputs"]["trace_repetitions_equal"]["on"])
        self.assertFalse(summary["outputs"]["trace_repetitions_equal"]["off"])
        self.assertEqual(summary["traces"]["kernel_delta"]["shared_names"], 1)

    def test_help_runs_from_a_shallow_absolute_path_with_pythonpath(self) -> None:
        source = (
            pathlib.Path(__file__).resolve().parents[2]
            / "tools"
            / "bench"
            / "finalize_async_credit.py"
        )
        descriptor, copied_name = tempfile.mkstemp(
            prefix="vllm-cpp-finalizer-", suffix=".py", dir="/tmp"
        )
        os.close(descriptor)
        copied = pathlib.Path(copied_name)
        try:
            shutil.copyfile(source, copied)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[2])
            result = subprocess.run(
                [sys.executable, str(copied), "--help"],
                cwd="/tmp",
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
        finally:
            copied.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-commit", result.stdout)

    def test_finalization_writes_hashed_marker_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = self._fixture(root)
            finalizer = root / "finalizer.py"
            finalizer.write_text("exact finalizer\n", encoding="utf-8")
            summary, manifest, marker = finalize(
                evidence, source_commit=SOURCE_COMMIT, finalizer=finalizer
            )
            self.assertEqual(marker["status"], "complete-diagnostic")
            self.assertEqual(marker["finalizer_sha256"], _digest("exact finalizer\n"))
            self.assertEqual(manifest["artifact_set_sha256"], marker["artifact_set_sha256"])
            self.assertEqual(summary["schema_version"], 1)
            with self.assertRaisesRegex(HarnessError, "refusing to overwrite"):
                finalize(evidence, source_commit=SOURCE_COMMIT, finalizer=finalizer)

    def test_missing_leg_and_inconsistent_digest_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = self._fixture(pathlib.Path(temporary))
            (evidence / "raw" / "off-r3.json").unlink()
            with self.assertRaisesRegex(HarnessError, "missing artifact"):
                build_summary(evidence, source_commit=SOURCE_COMMIT)
        with tempfile.TemporaryDirectory() as temporary:
            evidence = self._fixture(pathlib.Path(temporary))
            path = evidence / "traces" / "off-metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["output_digests_equal"] = True
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "digests must be False"):
                build_summary(evidence, source_commit=SOURCE_COMMIT)


if __name__ == "__main__":
    unittest.main()
