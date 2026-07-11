"""Ported command/result contracts for the CUDA online serving gate.

Upstream sources at vLLM e24d1b24:
``tests/benchmarks/test_serve_cli.py:58-132``,
``tests/benchmarks/test_custom_dataset_seed.py:24-75``, and
``vllm/benchmarks/serve.py:1188-1284,2082-2284``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.bench.online_gate import (
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    OUTPUT_LEN,
    TRACE_CONCURRENCY,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
    VLLM_ORACLE_VERSION,
    OnlineRun,
    build_client_command,
    build_plan,
    prepare_corpus_views,
    record_execution_manifest,
    record_memory_return,
    record_model_gate,
    record_oracle_manifest,
    record_trace_status,
    validate_plan,
    validate_raw_result,
)
from tools.bench.serve_low_common import HarnessError, sha256_file


def valid_record(*, requests: int = 6, concurrency: int = 1) -> dict:
    record = {
        "completed": requests,
        "duration": 12.0,
        "errors": [""] * requests,
        "failed": 0,
        "generated_texts": [f"answer-{index}" for index in range(requests)],
        "input_lens": [INPUT_LEN] * requests,
        "itls": [[0.01] * (OUTPUT_LEN - 1) for _ in range(requests)],
        "max_concurrency": concurrency,
        "max_concurrent_requests": min(concurrency, requests),
        "num_prompts": requests,
        "output_lens": [OUTPUT_LEN] * requests,
        "output_throughput": requests * OUTPUT_LEN / 12.0,
        "request_throughput": requests / 12.0,
        "start_times": [float(index) for index in range(requests)],
        "total_input_tokens": requests * INPUT_LEN,
        "total_output_tokens": requests * OUTPUT_LEN,
        "total_token_throughput": requests * (INPUT_LEN + OUTPUT_LEN) / 12.0,
        "ttfts": [0.1] * requests,
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = 10.0
    return record


class OnlineClientContractTests(unittest.TestCase):
    def test_bench_serve_command_uses_frozen_custom_dataset_and_detailed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            client = root / "vllm"
            client.touch()
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            run = OnlineRun(
                client=client,
                tokenizer=tokenizer,
                evidence_root=root,
                model_key="27",
                engine="ours",
                base_url="http://127.0.0.1:8001",
                concurrency=16,
                repetition=2,
            )
            run.corpus_path.parent.mkdir(parents=True)
            run.corpus_path.write_text('{"prompt":"x"}\n', encoding="utf-8")
            command = build_client_command(run)
            self.assertEqual(command[:3], [str(client), "bench", "serve"])
            self.assertEqual(command[command.index("--dataset-name") + 1], "custom")
            self.assertEqual(command[command.index("--dataset-path") + 1], str(run.corpus_path))
            self.assertEqual(command[command.index("--num-prompts") + 1], "96")
            self.assertEqual(command[command.index("--max-concurrency") + 1], "16")
            self.assertEqual(command[command.index("--num-warmups") + 1], "16")
            self.assertIn("--disable-shuffle", command)
            self.assertIn("--skip-chat-template", command)
            self.assertIn("--save-detailed", command)
            self.assertIn("--ignore-eos", command)
            self.assertEqual(command[command.index("--temperature") + 1], "0")

    def test_result_validation_rejects_partial_failed_or_bundled_streams(self) -> None:
        validate_raw_result(valid_record(), concurrency=1)

        partial = valid_record()
        partial["completed"] = 1
        partial["failed"] = 5
        with self.assertRaisesRegex(HarnessError, "partial"):
            validate_raw_result(partial, concurrency=1)

        bundled = valid_record()
        bundled["itls"][0].pop()
        with self.assertRaisesRegex(HarnessError, "output_len-1"):
            validate_raw_result(bundled, concurrency=1)

        unsaturated = valid_record(requests=6, concurrency=2)
        unsaturated["max_concurrent_requests"] = 1
        with self.assertRaisesRegex(HarnessError, "peak concurrency"):
            validate_raw_result(
                unsaturated,
                concurrency=2,
                expected_requests=6,
            )

    def test_prepare_corpus_preserves_exact_prompts_and_refuses_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            rows = []
            for index in range(2):
                rows.append(
                    {
                        "conversations": [{"from": "human", "value": f"prompt-{index}"}],
                        "index": index,
                        "output_len": OUTPUT_LEN,
                        "prompt_sha256": f"{index + 1:064x}",
                        "prompt_token_ids": [index] * INPUT_LEN,
                    }
                )
            (source / "c1-r1.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "view"
            with (
                mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
                mock.patch("tools.bench.online_gate.REPETITIONS", (1,)),
            ):
                manifest = prepare_corpus_views(source, output, model_key="27")
            self.assertEqual(manifest["total_prompts"], 2)
            converted = [json.loads(line) for line in (output / "c1-r1.jsonl").read_text().splitlines()]
            self.assertEqual([row["prompt"] for row in converted], ["prompt-0", "prompt-1"])
            self.assertTrue(all(row["output_tokens"] == OUTPUT_LEN for row in converted))
            with self.assertRaisesRegex(HarnessError, "non-empty"):
                with (
                    mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
                    mock.patch("tools.bench.online_gate.REPETITIONS", (1,)),
                ):
                    prepare_corpus_views(source, output, model_key="27")

    def test_plan_has_one_lock_per_model_and_interleaves_arms(self) -> None:
        plan = build_plan(
            claim_root=pathlib.Path("/claim"),
            vllm_cpp_sha="a" * 40,
            client=pathlib.Path("/oracle/bin/vllm"),
        )
        self.assertEqual(plan["gpu_lock_acquisitions_planned"], 2)
        self.assertEqual(
            plan["interleaving"][:4],
            [
                {"engine": "ours", "repetition": 1},
                {"engine": "vllm", "repetition": 1},
                {"engine": "ours", "repetition": 2},
                {"engine": "vllm", "repetition": 2},
            ],
        )
        self.assertEqual(plan["points"][-1], {"concurrency": 32, "num_prompts": 192})
        self.assertEqual(plan["run_contract"]["max_num_seqs"], MAX_NUM_SEQS)
        self.assertEqual(
            plan["run_contract"]["max_num_batched_tokens"],
            MAX_NUM_BATCHED_TOKENS,
        )
        self.assertEqual(
            plan["run_contract"]["warmups_per_client_invocation"],
            "equals configured concurrency",
        )
        corpus_command = plan["planned_commands"]["corpus"]
        self.assertEqual(
            corpus_command[:3],
            [
                "/oracle/bin/python",
                "-m",
                "tools.bench.make_serve_low_corpus",
            ],
        )
        repo = pathlib.Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, *corpus_command[1:3], "--help"],
            cwd=repo,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_memory_return_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "return.json"
            result = record_memory_return(
                path,
                baseline_mem_available_kib=10_000,
                final_mem_available_kib=9_500,
                tolerance_kib=1_000,
                drop_caches_succeeded=True,
                gpu_idle=True,
            )
            self.assertTrue(result["returned"])
            with self.assertRaisesRegex(HarnessError, "overwrite"):
                record_memory_return(
                    path,
                    baseline_mem_available_kib=10_000,
                    final_mem_available_kib=10_000,
                    tolerance_kib=0,
                    drop_caches_succeeded=True,
                    gpu_idle=True,
                )

    def test_campaign_shell_dry_run_plans_two_locks_without_gpu_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            repo = pathlib.Path(__file__).resolve().parents[2]
            result = subprocess.run(
                [
                    "bash",
                    str(repo / "scripts" / "dgx-online-serving.sh"),
                    "--dry-run",
                    "--claim-root",
                    str(root),
                    "--client",
                    "/oracle/bin/vllm",
                    "--vllm-cpp-sha",
                    "b" * 40,
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (root / "evidence" / ("b" * 40) / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["gpu_lock_acquisitions_planned"], 2)
            self.assertNotIn("nvidia-smi", result.stdout + result.stderr)
            validated = validate_plan(
                root / "evidence" / ("b" * 40) / "manifest.json",
                vllm_cpp_sha="b" * 40,
            )
            self.assertEqual(validated["vllm_cpp_sha"], "b" * 40)

    def test_oracle_manifest_pins_version_environment_and_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            bin_dir = root / "venv" / "bin"
            package = root / "venv" / "site-packages" / "vllm"
            dist_info = root / "venv" / "site-packages" / "vllm-0.24.0.dist-info"
            for directory in (
                bin_dir,
                package / "benchmarks" / "datasets",
                package / "entrypoints" / "cli" / "benchmark",
                dist_info,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            client = bin_dir / "vllm"
            python = bin_dir / "python"
            package_init = package / "__init__.py"
            for path in (
                client,
                python,
                package_init,
                package / "benchmarks" / "serve.py",
                package / "benchmarks" / "datasets" / "datasets.py",
                package / "entrypoints" / "cli" / "benchmark" / "serve.py",
                dist_info / "METADATA",
                dist_info / "RECORD",
            ):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            distribution = types.SimpleNamespace(
                version=VLLM_ORACLE_VERSION,
                _path=dist_info,
            )
            module = types.SimpleNamespace(
                __version__=VLLM_ORACLE_VERSION,
                __file__=str(package_init),
            )
            output = root / "oracle.json"
            with (
                mock.patch.dict("sys.modules", {"vllm": module}),
                mock.patch(
                    "tools.bench.online_gate.importlib.metadata.distribution",
                    return_value=distribution,
                ),
                mock.patch("tools.bench.online_gate.sys.executable", str(python)),
            ):
                result = record_oracle_manifest(output, client=client)
            self.assertEqual(result["oracle_version"], VLLM_ORACLE_VERSION)
            self.assertEqual(result["runtime_version"], VLLM_ORACLE_VERSION)
            self.assertEqual(len(result["artifacts"]), 8)
            self.assertEqual(
                result["artifacts"]["bench_serve"]["sha256"],
                sha256_file(package / "benchmarks" / "serve.py"),
            )

    def test_execution_model_and_trace_records_hash_their_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            build = root / "build"
            (build / "examples").mkdir(parents=True)
            snapshot = root / "890bdef7a42feba6d83b6e17a03315c694112f2a"
            snapshot.mkdir()
            bin_dir = root / "oracle-bin"
            bin_dir.mkdir()
            client = bin_dir / "vllm"
            build_command = root / "build-command.txt"
            build_log = root / "build.log"
            for path in (
                build / "CMakeCache.txt",
                build / "examples" / "server",
                build_command,
                build_log,
                snapshot / "config.json",
                snapshot / "tokenizer.json",
                snapshot / "model-00001-of-00001.safetensors",
                client,
            ):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            oracle_files = {}
            for name in (
                "bench_datasets",
                "bench_serve",
                "cli_bench_serve",
                "client",
                "distribution_metadata",
                "distribution_record",
                "package_init",
                "python",
            ):
                path = client if name == "client" else root / "oracle-files" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text(f"{name}\n", encoding="utf-8")
                oracle_files[name] = {"path": str(path), "sha256": sha256_file(path)}
            oracle_manifest = root / "oracle.json"
            oracle_manifest.write_text(
                json.dumps(
                    {
                        "artifacts": oracle_files,
                        "client_contract_source_commit": "e24d1b24fe96a56ba8b0d653efa076d03eb95d6c",
                        "oracle_version": VLLM_ORACLE_VERSION,
                        "runtime_version": VLLM_ORACLE_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            execution = record_execution_manifest(
                root / "execution.json",
                model_key="27",
                vllm_cpp_sha="d" * 40,
                build_dir=build,
                client=client,
                snapshot=snapshot,
                build_command=build_command,
                build_log=build_log,
                oracle_manifest=oracle_manifest,
                port=8001,
                num_blocks=4736,
                max_num_seqs=MAX_NUM_SEQS,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS["27"],
            )
            self.assertEqual(execution["model_key"], "27")
            self.assertEqual(execution["vllm_oracle_version"], VLLM_ORACLE_VERSION)
            self.assertIn("oracle:bench_serve", execution["artifacts"])
            self.assertIn("build_command", execution["artifacts"])

            gate_log = root / "gate.log"
            gate_log.write_text("passed\n", encoding="utf-8")
            gate = record_model_gate(
                root / "gate.json",
                log=gate_log,
                model_key="27",
                test_name="test_qwen27_paged_engine",
                vllm_cpp_sha="d" * 40,
            )
            self.assertTrue(gate["passed"])

            trace_paths = []
            for index in range(4):
                path = root / f"trace-{index}"
                path.write_text("trace\n", encoding="utf-8")
                trace_paths.append(path)
            ours_client_results = []
            ours_client_logs = []
            for index in range(TRACE_REPETITIONS):
                result_path = root / f"ours-client-{index}.json"
                result_path.write_text(
                    json.dumps(
                        valid_record(
                            requests=TRACE_PROMPTS,
                            concurrency=TRACE_CONCURRENCY,
                        )
                    ),
                    encoding="utf-8",
                )
                log_path = root / f"ours-client-{index}.log"
                log_path.write_text("client passed\n", encoding="utf-8")
                ours_client_results.append(result_path)
                ours_client_logs.append(log_path)
            ours_command = root / "ours-command.txt"
            ours_profile_log = root / "ours-profile.log"
            vllm_command = root / "vllm-command.txt"
            vllm_profile_log = root / "vllm-profile.log"
            vllm_corpus = root / "vllm-corpus.jsonl"
            for path in (
                ours_command,
                ours_profile_log,
                vllm_command,
                vllm_profile_log,
                vllm_corpus,
            ):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            vllm_metadata = root / "vllm-metadata.json"
            vllm_metadata.write_text(
                json.dumps(
                    {
                        "corpus": str(vllm_corpus),
                        "corpus_sha256": sha256_file(vllm_corpus),
                        "input_len": INPUT_LEN,
                        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS["27"],
                        "max_num_seqs": TRACE_CONCURRENCY,
                        "num_prompts": TRACE_PROMPTS,
                        "output_digest": "a" * 64,
                        "output_len": OUTPUT_LEN,
                        "profiled_warmup_prompts": TRACE_PROMPTS,
                        "repetitions": TRACE_REPETITIONS,
                    }
                ),
                encoding="utf-8",
            )
            trace = record_trace_status(
                root / "trace.json",
                model_key="27",
                ours_nsys_report=trace_paths[0],
                ours_kernel_summary=trace_paths[1],
                ours_command=ours_command,
                ours_profile_log=ours_profile_log,
                ours_client_results=ours_client_results,
                ours_client_logs=ours_client_logs,
                vllm_torch_trace=trace_paths[2],
                vllm_kernel_summary=trace_paths[3],
                vllm_command=vllm_command,
                vllm_profile_log=vllm_profile_log,
                vllm_metadata=vllm_metadata,
                vllm_corpus=vllm_corpus,
                vllm_cpp_sha="d" * 40,
            )
            self.assertEqual(trace["ours_profiler"], "nsys")
            self.assertEqual(trace["vllm_profiler"], "torch-profiler")


if __name__ == "__main__":
    unittest.main()
