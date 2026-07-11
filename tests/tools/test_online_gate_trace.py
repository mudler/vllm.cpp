"""Trace-tool contracts grounded in pinned vLLM profiler output.

Sources: ``examples/features/profiling/simple_profiling_offline.py:20-40`` and
``vllm/entrypoints/llm.py:787-798`` at vLLM e24d1b24.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import tempfile
import unittest

from tools.bench.online_gate import INPUT_LEN
from tools.bench.profile_vllm_online_gate import load_prompts
from tools.bench.serve_low_common import HarnessError
from tools.bench.summarize_torch_kernels import summarize


class OnlineGateTraceTests(unittest.TestCase):
    def test_exact_token_prompts_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "corpus.jsonl"
            path.write_text(
                json.dumps({"prompt_token_ids": [7] * INPUT_LEN}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_prompts(path, 1), [[7] * INPUT_LEN])
            with self.assertRaisesRegex(HarnessError, "need 2"):
                load_prompts(path, 2)
            path.write_text(
                json.dumps({"prompt_token_ids": [7] * (INPUT_LEN - 1)}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HarnessError, "not exact"):
                load_prompts(path, 1)

    def test_kernel_events_are_grouped_and_largest_worker_trace_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            small = root / "frontend.pt.trace.json"
            small.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"cat": "kernel", "dur": 1.0, "name": "frontend_kernel"}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            worker = root / "worker.pt.trace.json.gz"
            with gzip.open(worker, "wt", encoding="utf-8") as output:
                json.dump(
                    {
                        "traceEvents": [
                            {"cat": "kernel", "dur": 4.0, "name": "gemm"},
                            {"cat": "kernel", "dur": 6.0, "name": "gemm"},
                            {"cat": "kernel", "dur": 5.0, "name": "attention"},
                            {"cat": "cpu_op", "dur": 99.0, "name": "ignored"},
                        ]
                    },
                    output,
                )
            result = summarize(root)
            self.assertEqual(result["selected_trace"], str(worker))
            self.assertEqual(result["kernel_count"], 3)
            self.assertEqual(result["kernel_time_us"], 15.0)
            self.assertEqual(result["kernels"][0]["name"], "gemm")
            self.assertEqual(result["kernels"][0]["count"], 2)
            self.assertAlmostEqual(
                sum(item["percent"] for item in result["kernels"]), 100.0
            )

    def test_missing_kernel_events_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "empty.pt.trace.json"
            path.write_text(json.dumps({"traceEvents": []}), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "no positive-duration"):
                summarize(path.parent)


if __name__ == "__main__":
    unittest.main()
