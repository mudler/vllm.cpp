"""Every-axis aggregation tests ported from vLLM bench-serve metrics.

Sources: ``vllm/benchmarks/serve.py:563-748,1188-1284`` and
``tests/benchmarks/test_serve_cli.py:58-132`` at vLLM e24d1b24.  Project-only
extensions exercise the stricter paired-output, memory-return, and every-axis
acceptance rules from ``.agents/benchmark-protocol.md``.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    OUTPUT_LEN,
    PANDAS_VERSION,
    TRACE_CONCURRENCY,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
    VLLM_ORACLE_VERSION,
)
from tools.bench.online_gate_summary import summarize_evidence
from tools.bench.serve_low_common import sha256_file


def _record(*, faster: bool, repetition: int) -> dict:
    requests = 2
    duration = 10.0 if faster else 12.0
    latency = 8.0 if faster else 10.0
    record = {
        "completed": requests,
        "duration": duration,
        "errors": [""] * requests,
        "failed": 0,
        "generated_texts": [f"same-{repetition}-0", f"same-{repetition}-1"],
        "input_lens": [INPUT_LEN] * requests,
        "itls": [[0.01] * (OUTPUT_LEN - 1) for _ in range(requests)],
        "max_concurrency": 1,
        "max_concurrent_requests": 1,
        "num_prompts": requests,
        "output_lens": [OUTPUT_LEN] * requests,
        "output_throughput": requests * OUTPUT_LEN / duration,
        "request_throughput": requests / duration,
        "start_times": [0.0, 2.0],
        "total_input_tokens": requests * INPUT_LEN,
        "total_output_tokens": requests * OUTPUT_LEN,
        "total_token_throughput": requests * (INPUT_LEN + OUTPUT_LEN) / duration,
        "ttfts": [0.1, 0.1],
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = latency
    return record


def _fixture_cache_roots(root: pathlib.Path) -> list[str]:
    return [
        str(root / "fixture-artifacts"),
        str(root / "corpus" / "27"),
        str(root / "fixture-artifacts" / "server"),
        str(root / "fixture-artifacts" / "client"),
    ]


def _write_cache_drop(path: pathlib.Path, roots: list[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "file_count": 1,
        "file_inventory_sha256": "a" * 64,
        "logical_bytes": 4096,
        "method": CACHE_DROP_METHOD,
        "resident_after_bytes": 0,
        "roots": roots,
        "succeeded": True,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "file_count": report["file_count"],
        "file_inventory_sha256": report["file_inventory_sha256"],
        "logical_bytes": report["logical_bytes"],
        "method": CACHE_DROP_METHOD,
        "path": str(path),
        "roots": report["roots"],
        "sha256": sha256_file(path),
    }


def _write_fixture(root: pathlib.Path) -> None:
    sha = "c" * 40
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "client_contract_source_commit": "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c",
                "gpu_lock_acquisitions_planned": 2,
                "vllm_oracle_bench_dependencies": {"pandas": PANDAS_VERSION},
                "vllm_oracle_version": VLLM_ORACLE_VERSION,
                "vllm_cpp_sha": sha,
            }
        ),
        encoding="utf-8",
    )
    execution_root = root / "execution"
    execution_root.mkdir()
    execution_artifacts = {}
    for name in (
        "build_command",
        "build_log",
        "client",
        "cmake_cache",
        "model_config",
        "oracle_manifest",
        "oracle:bench_datasets",
        "oracle:bench_serve",
        "oracle:cli_bench_serve",
        "oracle:client",
        "oracle:distribution_metadata",
        "oracle:distribution_record",
        "oracle:ninja",
        "oracle:package_init",
        "oracle:python",
        "oracle:pandas_distribution_metadata",
        "oracle:pandas_distribution_record",
        "oracle:pandas_package_init",
        "server",
        "tokenizer",
    ):
        path = root / "fixture-artifacts" / name.replace(":", "-")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        execution_artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    weight_name = "model-00001-of-00001.safetensors"
    weight = root / "fixture-artifacts" / weight_name
    weight.write_text("weights\n", encoding="utf-8")
    execution_artifacts[f"weight:{weight_name}"] = {
        "path": str(weight),
        "sha256": sha256_file(weight),
    }
    (execution_root / "27.json").write_text(
        json.dumps(
            {
                "artifacts": execution_artifacts,
                "bench_dependencies": {"pandas": PANDAS_VERSION},
                "cache_drop_roots": _fixture_cache_roots(root),
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS["27"],
                "max_num_seqs": MAX_NUM_SEQS,
                "model_key": "27",
                "vllm_oracle_version": VLLM_ORACLE_VERSION,
                "vllm_cpp_sha": sha,
                "vllm_source_sha": "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c",
                "snapshot_files": [],
                "weight_files": [weight_name],
            }
        ),
        encoding="utf-8",
    )
    model_gate = root / "preflight" / "model-gate"
    model_gate.mkdir(parents=True)
    gate_log = model_gate / "27.log"
    gate_log.write_text("100% tests passed\n", encoding="utf-8")
    (model_gate / "27.json").write_text(
        json.dumps(
            {
                "log": str(gate_log),
                "log_sha256": sha256_file(gate_log),
                "model_key": "27",
                "passed": True,
                "test_name": "test_qwen27_paged_engine",
                "vllm_cpp_sha": sha,
            }
        ),
        encoding="utf-8",
    )
    trace_root = root / "trace" / "27"
    trace_root.mkdir(parents=True)
    trace_files = {}
    for name in (
        "ours_command",
        "ours_profile_log",
        "ours_nsys_report",
        "ours_kernel_summary",
        "ours_client_result_1",
        "ours_client_result_2",
        "ours_client_result_3",
        "ours_client_log_1",
        "ours_client_log_2",
        "ours_client_log_3",
        "vllm_command",
        "vllm_profile_log",
        "vllm_metadata",
        "vllm_corpus",
        "vllm_torch_trace",
        "vllm_kernel_summary",
    ):
        path = trace_root / f"{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        trace_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    for index in range(1, 4):
        trace_files[f"cache_drop_{index}"] = _write_cache_drop(
            trace_root / f"cache-drop-{index}.json",
            _fixture_cache_roots(root),
        )
    (trace_root / "status.json").write_text(
        json.dumps(
            {
                "artifacts": trace_files,
                "model_key": "27",
                "ours_profiler": "nsys",
                "passed": True,
                "trace_contract": {
                    "concurrency": TRACE_CONCURRENCY,
                    "input_len": INPUT_LEN,
                    "num_prompts": TRACE_PROMPTS,
                    "output_len": OUTPUT_LEN,
                    "repetitions": TRACE_REPETITIONS,
                },
                "vllm_cpp_sha": sha,
                "vllm_profiler": "torch-profiler",
            }
        ),
        encoding="utf-8",
    )

    corpus_root = root / "corpus" / "27"
    corpus_view = corpus_root / "vllm"
    corpus_view.mkdir(parents=True)
    source_manifest = corpus_root / "manifest.json"
    source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
    corpus_files = []
    for repetition in (1, 2, 3):
        filename = f"c1-r{repetition}.jsonl"
        source = corpus_root / filename
        target = corpus_view / filename
        source.write_text(f'{{"source":{repetition}}}\n', encoding="utf-8")
        target.write_text(f'{{"prompt":"fixture-{repetition}"}}\n', encoding="utf-8")
        corpus_files.append(
            {
                "concurrency": 1,
                "file": filename,
                "repetition": repetition,
                "requests": 2,
                "sha256": sha256_file(target),
                "source_sha256": sha256_file(source),
            }
        )
    (corpus_view / "manifest.json").write_text(
        json.dumps(
            {
                "files": corpus_files,
                "model_key": "27",
                "source_manifest_sha256": sha256_file(source_manifest),
                "tokenizer_revision": "revision",
                "vllm_commit": "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c",
            }
        ),
        encoding="utf-8",
    )

    for engine, faster in (("ours", True), ("vllm", False)):
        for repetition in (1, 2, 3):
            raw = root / "raw" / "27" / engine / f"c1-r{repetition}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(
                json.dumps(_record(faster=faster, repetition=repetition)),
                encoding="utf-8",
            )
            client_log = root / "logs" / "27" / engine / f"c1-r{repetition}.log"
            client_log.parent.mkdir(parents=True, exist_ok=True)
            client_log.write_text("timed client passed\n", encoding="utf-8")
            (client_log.parent / f"r{repetition}-server-command.txt").write_text(
                "server --model /fixture --max-num-seqs 32 "
                "--max-num-batched-tokens 2048 --served-model-name gate\n",
                encoding="utf-8",
            )
            (client_log.parent / f"r{repetition}-server.log").write_text(
                "server ready\n", encoding="utf-8"
            )

            preflight = root / "preflight" / "27" / engine
            preflight.mkdir(parents=True, exist_ok=True)
            (preflight / f"r{repetition}-stream.json").write_text(
                json.dumps(
                    {
                        "emitted_chunks": OUTPUT_LEN,
                        "first_chunk_s": 0.1,
                        "generated_text": "answer",
                        "total_s": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            memory = root / "memory" / "27" / engine
            memory.mkdir(parents=True, exist_ok=True)
            base = 100.0 if faster else 120.0
            (memory / f"r{repetition}.summary.json").write_text(
                json.dumps(
                    {
                        "peak_mem_available_drop_kib": base,
                        "peak_pss_kib": base,
                        "peak_rss_kib": base,
                        "samples": 2,
                    }
                ),
                encoding="utf-8",
            )
            samples = [
                {
                    "alive": True,
                    "gpu_memory_mib": base,
                    "peak_mem_available_drop_kib": base,
                    "pids": [123],
                    "pss_kib": base,
                    "rss_kib": base,
                },
                {
                    "alive": False,
                    "gpu_memory_mib": 0,
                    "peak_mem_available_drop_kib": base,
                    "pids": [],
                    "pss_kib": 0,
                    "rss_kib": 0,
                },
            ]
            (memory / f"r{repetition}.samples.jsonl").write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            thermal = root / "thermal" / "27" / engine
            thermal.mkdir(parents=True, exist_ok=True)
            for suffix in ("before", "after"):
                (thermal / f"r{repetition}-{suffix}.txt").write_text(
                    "Temperature : 50 C\nPower : 100 W\n", encoding="utf-8"
                )

            returned = root / "memory-return" / "27" / engine
            returned.mkdir(parents=True, exist_ok=True)
            cache_root = root / "cache-drop" / "27" / engine
            cache_drops = {
                phase: _write_cache_drop(
                    cache_root / f"r{repetition}-{phase}.json",
                    _fixture_cache_roots(root),
                )
                for phase in ("before", "after")
            }
            (returned / f"r{repetition}.json").write_text(
                json.dumps(
                    {
                        "cache_drops": cache_drops,
                        "drop_caches_succeeded": True,
                        "gpu_idle": True,
                        "mem_available_within_tolerance": True,
                        "returned": True,
                    }
                ),
                encoding="utf-8",
            )


class OnlineGateSummaryTests(unittest.TestCase):
    def _summarize(self, root: pathlib.Path):
        patches = (
            mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
            mock.patch("tools.bench.online_gate_summary.POINTS", ((1, 2),)),
            mock.patch(
                "tools.bench.online_gate_summary.MODEL_REVISIONS", {"27": "revision"}
            ),
        )
        with patches[0], patches[1], patches[2]:
            return summarize_evidence(root)

    def test_complete_exact_outputs_and_better_axes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(ratios["gate_pass"])
            self.assertTrue(all(item["binding_eligible"] for item in ratios["ratios"]))
            self.assertTrue(all(item["pass"] for item in ratios["ratios"]))

    def test_text_difference_is_diagnostic_after_model_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "raw" / "27" / "vllm" / "c1-r2.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["generated_texts"][0] = "different"
            path.write_text(json.dumps(record), encoding="utf-8")
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(all(item["binding_eligible"] for item in ratios["ratios"]))
            diagnostic = next(
                item
                for item in runs["output_text_diagnostics"]
                if item["repetition"] == 2
            )
            self.assertFalse(diagnostic["all_equal"])
            self.assertEqual(diagnostic["exact_matches"], 1)

    def test_missing_memory_or_partial_request_set_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "memory" / "27" / "ours" / "r1.samples.jsonl").unlink()
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            returned_path = root / "memory-return" / "27" / "ours" / "r1.json"
            returned = json.loads(returned_path.read_text(encoding="utf-8"))
            returned["cache_drops"]["before"]["roots"][0] = "/wrong-snapshot"
            returned_path.write_text(json.dumps(returned), encoding="utf-8")
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            command = root / "logs" / "27" / "ours" / "r1-server-command.txt"
            command.write_text(
                "server --max-num-seqs 8 --max-num-batched-tokens 2048 "
                "--served-model-name gate\n",
                encoding="utf-8",
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "raw" / "27" / "ours" / "c1-r1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["completed"] = 1
            record["failed"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_missing_or_hash_drifted_execution_trace_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "trace" / "27" / "status.json").unlink()
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "fixture-artifacts" / "oracle-bench_serve").write_text(
                "tampered\n", encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "trace" / "27" / "ours_kernel_summary.txt").write_text(
                "tampered\n", encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_hash_drifted_corpus_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "corpus" / "27" / "vllm" / "c1-r2.jsonl").write_text(
                '{"prompt":"tampered"}\n', encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])


if __name__ == "__main__":
    unittest.main()
