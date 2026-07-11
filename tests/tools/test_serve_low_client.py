"""Pinned-client wrapper tests derived from SGLang bench functionality tests.

Sources: ``test_bench_serving_functionality.py`` and
``bench_serving.py:232-344,1588-1675`` @ SGLang 28b095c.
"""

from __future__ import annotations

import http.server
import json
import pathlib
import subprocess
import tempfile
import threading
import time
import unittest

from tools.bench.run_serve_low import (
    BenchRun,
    build_bench_command,
    build_dry_run_manifest,
    openai_stream_probe,
    openai_usage_preflight,
    run_usage_batch,
    validate_raw_result,
)
from tools.bench.serve_low_common import HarnessError, SGLANG_IMAGE


class _CompletionHandler(http.server.BaseHTTPRequestHandler):
    lock = threading.Lock()
    active = 0
    peak = 0
    payloads: list[dict] = []

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            type(self).payloads.append(payload)
        try:
            if self.path == "/error":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"fixture"}')
                return
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for token in "ABCD":
                    body = json.dumps({"choices": [{"text": token}]})
                    self.wfile.write(f"data: {body}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.015)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            time.sleep(0.02)
            body = json.dumps(
                {
                    "choices": [{"finish_reason": "length", "text": "ABCD"}],
                    "usage": {
                        "completion_tokens": 4,
                        "prompt_tokens": 8,
                        "total_tokens": 12,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self.lock:
                type(self).active -= 1


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _CompletionHandler.active = 0
        _CompletionHandler.peak = 0
        _CompletionHandler.payloads = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_request_shape_concurrency_cap_and_error_propagation(self) -> None:
        results = run_usage_batch(
            self.base + "/v1/completions",
            [f"prompt {index}" for index in range(6)],
            max_concurrency=2,
            prompt_tokens=8,
            completion_tokens=4,
        )
        self.assertEqual(len(results), 6)
        self.assertGreaterEqual(_CompletionHandler.peak, 2)
        self.assertLessEqual(_CompletionHandler.peak, 2)
        payload = _CompletionHandler.payloads[0]
        self.assertEqual(payload["max_tokens"], 4)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertTrue(payload["ignore_eos"])
        self.assertFalse(payload["stream"])
        with self.assertRaises(HarnessError):
            openai_usage_preflight(
                self.base + "/error",
                "prompt",
                prompt_tokens=8,
                completion_tokens=4,
            )

    def test_stream_probe_requires_incremental_exact_chunk_count(self) -> None:
        result = openai_stream_probe(
            self.base + "/v1/completions",
            "prompt",
            completion_tokens=4,
            minimum_spread_s=0.02,
        )
        self.assertEqual(result.emitted_chunks, 4)
        self.assertEqual(result.generated_text, "ABCD")
        self.assertGreater(result.spread_s, 0.02)

    def test_raw_detail_validation_is_fail_closed(self) -> None:
        record = {
            "completed": 2,
            "errors": ["", ""],
            "generated_texts": ["ABCD", "ABCD"],
            "input_lens": [8, 8],
            "itls": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
            "output_lens": [4, 4],
            "ttfts": [0.2, 0.2],
        }
        validate_raw_result(record, expected_requests=2, prompt_len=8, output_len=4)
        record["errors"][1] = "boom"
        with self.assertRaises(HarnessError):
            validate_raw_result(record, expected_requests=2, prompt_len=8, output_len=4)

    def test_pinned_client_command_and_dry_run_refuse_floating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            (root / "models").mkdir()
            corpus = evidence / "corpus" / "27"
            corpus.mkdir(parents=True)
            (corpus / "c1-r1.jsonl").write_text("{}\n")
            run = BenchRun(
                image=SGLANG_IMAGE,
                model_repo=root / "models",
                model_revision="revision",
                evidence_root=evidence,
                model_key="27",
                engine="ours",
                base_url="http://127.0.0.1:30000",
                concurrency=1,
                repetition=1,
            )
            command = build_bench_command(run)
            self.assertIn("--pull=never", command)
            self.assertNotIn("--gpus", command)
            self.assertEqual(command.count("sglang.bench_serving"), 1)
            manifest = build_dry_run_manifest(
                claim_root=root,
                vllm_cpp_sha="a" * 40,
                image=SGLANG_IMAGE,
            )
            self.assertTrue(manifest["dry_run"])
            self.assertIn("native_output_id_parity", manifest["pending_preconditions"])
            with self.assertRaises(HarnessError):
                build_dry_run_manifest(
                    claim_root=root,
                    vllm_cpp_sha="a" * 40,
                    image="sglang:latest",
                )

    def test_campaign_shell_dry_run_creates_manifest_without_gpu_work(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            claim = pathlib.Path(temporary) / "claim"
            subprocess.run(
                [
                    str(repo / "scripts" / "dgx-sglang-low-concurrency.sh"),
                    "--dry-run",
                    "--claim-root",
                    str(claim),
                    "--vllm-cpp-sha",
                    "b" * 40,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (claim / "evidence" / ("b" * 40) / "manifest.json").read_text()
            )
            self.assertTrue(manifest["dry_run"])
            self.assertIn("host_idle_proof", manifest["pending_preconditions"])
            self.assertEqual(manifest["gpu_lock_acquisitions_planned"], 1)
            self.assertFalse(manifest["pull_under_gpu_lock"])
            self.assertIn("--pull=never", manifest["planned_commands"]["client"])


if __name__ == "__main__":
    unittest.main()
