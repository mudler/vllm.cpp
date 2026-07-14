"""Trace-tool contracts grounded in pinned vLLM profiler output.

Sources: ``examples/features/profiling/simple_profiling_offline.py:20-40`` and
``vllm/entrypoints/llm.py:787-798`` at vLLM e24d1b24.
"""

from __future__ import annotations

import gzip
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from tools.bench.online_gate import INPUT_LEN
from tools.bench.profile_vllm_online_gate import (
    async_scheduling_override,
    load_prompts,
    run_closed_loop,
)
from tools.bench.serve_low_common import HarnessError
from tools.bench.summarize_torch_kernels import summarize


class OnlineGateTraceTests(unittest.TestCase):
    def test_profiler_help_runs_by_absolute_path_outside_repository(self) -> None:
        script = (
            pathlib.Path(__file__).resolve().parents[2]
            / "tools"
            / "bench"
            / "profile_vllm_online_gate.py"
        )
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=temporary,
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--async-scheduling", result.stdout)

    def test_async_scheduling_override_preserves_default_resolution(self) -> None:
        self.assertEqual(async_scheduling_override("default"), {})
        self.assertEqual(async_scheduling_override("on"), {"async_scheduling": True})
        self.assertEqual(async_scheduling_override("off"), {"async_scheduling": False})
        with self.assertRaisesRegex(HarnessError, "unknown async scheduling mode"):
            async_scheduling_override("invalid")

    def test_closed_loop_never_preloads_beyond_concurrency(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.active = []
                self.max_active = 0

            def add_request(self, request_id, prompt, params) -> None:
                self.active.append((request_id, prompt, params))
                self.max_active = max(self.max_active, len(self.active))

            def has_unfinished_requests(self) -> bool:
                return bool(self.active)

            def step(self):
                request_id, prompt, _ = self.active.pop(0)
                return [
                    SimpleNamespace(
                        request_id=request_id,
                        finished=True,
                        outputs=[SimpleNamespace(token_ids=[prompt])],
                    )
                ]

        engine = FakeEngine()
        llm = SimpleNamespace(llm_engine=engine)
        outputs = run_closed_loop(
            llm,
            list(range(9)),
            SimpleNamespace(output_kind=None),
            max_concurrency=3,
            request_id_base=100,
            final_output_kind="final",
        )
        self.assertEqual(engine.max_active, 3)
        self.assertEqual(
            [item.outputs[0].token_ids[0] for item in outputs], list(range(9))
        )

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
