"""Ported command/result contracts for the CUDA online serving gate.

Upstream sources at vLLM e24d1b24:
``tests/benchmarks/test_serve_cli.py:58-132``,
``tests/benchmarks/test_custom_dataset_seed.py:24-75``, and
``vllm/benchmarks/serve.py:1188-1284,2082-2284``.
"""

from __future__ import annotations

import json
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    NSYS_CUDA_FLUSH_INTERVAL_MS,
    NSYS_CUDA_GRAPH_TRACE,
    OUTPUT_LEN,
    PANDAS_VERSION,
    TRACE_CONCURRENCY,
    TRACE_PRIMARY_GRAPH_CONTRACTS,
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
    validate_nsys_trace,
    validate_plan,
    validate_raw_result,
)
from tools.bench.serve_low_common import HarnessError, VLLM_COMMIT, sha256_file


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
        "start_times": [
            float(index // max(1, concurrency)) * 2.0
            for index in range(requests)
        ],
        "total_input_tokens": requests * INPUT_LEN,
        "total_output_tokens": requests * OUTPUT_LEN,
        "total_token_throughput": requests * (INPUT_LEN + OUTPUT_LEN) / 12.0,
        "ttfts": [0.1] * requests,
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = 10.0
    return record


def write_nsys_sqlite(
    path: pathlib.Path,
    *,
    family_drift: bool = False,
    lost_events: bool = False,
    model_contract: bool = False,
    uneven_replays: bool = False,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE DIAGNOSTIC_EVENT "
            "(timestamp INTEGER, severity INTEGER, text TEXT)"
        )
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(graphNodeId INTEGER, demangledName INTEGER)"
        )
        connection.execute(
            "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, ?, ?)",
            (
                1,
                3,
                "CUDA device 0: Unified Memory trace is not supported by the "
                "current driver version or configuration.",
            ),
        )
        if lost_events:
            connection.execute(
                "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, ?, ?)",
                (2, 2, "Not all CUDA events might have been collected."),
            )
        if model_contract:
            contract = TRACE_PRIMARY_GRAPH_CONTRACTS["27"]
            node_names = []
            for family, (pattern, count) in contract["families"].items():
                if family_drift and family == "normal_fp4_producer":
                    count -= 1
                node_names.extend([pattern] * count)
            node_names.extend(
                ["unclassified-kernel"]
                * (contract["node_count"] - len(node_names))
            )
            replay_counts = [3] * len(node_names)
        else:
            node_names = ["kernel-a", "kernel-b"]
            replay_counts = [3, 2 if uneven_replays else 3]
        string_ids = {
            name: index for index, name in enumerate(sorted(set(node_names)), start=1)
        }
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(identifier, name) for name, identifier in string_ids.items()],
        )
        for graph_node_id, (name, replay_count) in enumerate(
            zip(node_names, replay_counts, strict=True), start=1
        ):
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)",
                [(graph_node_id, string_ids[name])] * replay_count,
            )
        connection.commit()
    finally:
        connection.close()


def write_cache_drop_report(path: pathlib.Path) -> None:
    path.write_text(
        json.dumps(
            {
                "file_count": 1,
                "file_inventory_sha256": "a" * 64,
                "logical_bytes": 4096,
                "method": CACHE_DROP_METHOD,
                "resident_after_bytes": 0,
                "roots": ["/snapshot", "/corpus", "/server", "/client"],
                "succeeded": True,
            }
        ),
        encoding="utf-8",
    )


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

    def test_result_validation_accepts_merged_deltas_and_rejects_invalid_cadence(
        self,
    ) -> None:
        validate_raw_result(valid_record(), concurrency=1)
        bucketed_boundary = valid_record()
        bucketed_boundary["max_concurrent_requests"] = 2
        validate_raw_result(bucketed_boundary, concurrency=1)

        partial = valid_record()
        partial["completed"] = 1
        partial["failed"] = 5
        with self.assertRaisesRegex(HarnessError, "partial"):
            validate_raw_result(partial, concurrency=1)

        # RequestOutputCollector mirrors pinned vLLM's legal producer-ahead
        # merge. The native usage count remains exact while one SSE choice can
        # carry two token deltas, leaving 126 rather than 127 ITLs.
        merged = valid_record()
        merged["itls"][0].pop()
        validate_raw_result(merged, concurrency=1)

        over_fragmented = valid_record()
        over_fragmented["itls"][0].append(0.01)
        with self.assertRaisesRegex(HarnessError, "exceed output_len-1"):
            validate_raw_result(over_fragmented, concurrency=1)

        malformed = valid_record()
        malformed["itls"][0] = "not-a-list"
        with self.assertRaisesRegex(HarnessError, "exceed output_len-1"):
            validate_raw_result(malformed, concurrency=1)

        unsaturated = valid_record(requests=6, concurrency=2)
        unsaturated["start_times"] = [float(index) * 2.0 for index in range(6)]
        unsaturated["max_concurrent_requests"] = 1
        with self.assertRaisesRegex(HarnessError, "precise peak concurrency"):
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
        self.assertEqual(
            plan["vllm_oracle_bench_dependencies"],
            {"pandas": PANDAS_VERSION},
        )
        self.assertIn(
            "summary-<model>/{all-runs,ratios}.json",
            plan["required_artifacts"],
        )
        self.assertEqual(
            plan["planned_commands"]["trace_model"][1],
            "--trace-only",
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

    def test_vllm_profiler_exports_the_oracle_venv_path(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        block = script.split("local -a vllm_profile_cmd=(", 1)[1].split(
            "\n  )", 1
        )[0]
        self.assertIn('env "PATH=$(dirname "${client}"):${PATH}"', block)

    def test_campaign_writes_a_model_summary_before_cross_model_summary(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--model "${model}"', script)
        self.assertIn("model_summary_status", script)
        self.assertIn("summary-27/ratios.json", script)
        self.assertIn("summary-35/ratios.json", script)
        self.assertIn("--cuda-graph-trace=node", script)
        self.assertIn(
            f"--cuda-flush-interval={NSYS_CUDA_FLUSH_INTERVAL_MS}", script
        )
        self.assertIn("--cpuctxsw=none", script)
        self.assertIn("validate-nsys-trace", script)
        self.assertIn('--model-key "${model}"', script)
        self.assertIn("--trace-only", script)

    def test_nsys_trace_validation_rejects_loss_and_uneven_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            clean = root / "clean.sqlite"
            write_nsys_sqlite(clean)
            result = validate_nsys_trace(clean)
            self.assertTrue(result["lossless"])
            self.assertEqual(result["primary_graph_node_count"], 2)
            self.assertEqual(result["primary_graph_replay_count"], 3)

            lost = root / "lost.sqlite"
            write_nsys_sqlite(lost, lost_events=True)
            with self.assertRaisesRegex(HarnessError, "not lossless"):
                validate_nsys_trace(lost)

            uneven = root / "uneven.sqlite"
            write_nsys_sqlite(uneven, uneven_replays=True)
            with self.assertRaisesRegex(HarnessError, "uneven replay counts"):
                validate_nsys_trace(uneven)

            model_contract = root / "model-contract.sqlite"
            write_nsys_sqlite(model_contract, model_contract=True)
            result = validate_nsys_trace(model_contract, model_key="27")
            self.assertEqual(result["primary_graph_node_count"], 1_107)
            self.assertEqual(
                result["primary_graph_family_node_counts"]["fp4_gemm"], 208
            )

            family_drift = root / "family-drift.sqlite"
            write_nsys_sqlite(
                family_drift, family_drift=True, model_contract=True
            )
            with self.assertRaisesRegex(HarnessError, "normal_fp4_producer"):
                validate_nsys_trace(family_drift, model_key="27")

    def test_memory_return_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "return.json"
            before = root / "before.json"
            after = root / "after.json"
            write_cache_drop_report(before)
            write_cache_drop_report(after)
            result = record_memory_return(
                path,
                baseline_mem_available_kib=10_000,
                final_mem_available_kib=9_500,
                tolerance_kib=1_000,
                before_cache_drop_report=before,
                after_cache_drop_report=after,
                gpu_idle=True,
            )
            self.assertTrue(result["returned"])
            failed_report = json.loads(after.read_text(encoding="utf-8"))
            failed_report["resident_after_bytes"] = 4096
            after.write_text(json.dumps(failed_report), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "resident"):
                record_memory_return(
                    root / "failed-return.json",
                    baseline_mem_available_kib=10_000,
                    final_mem_available_kib=10_000,
                    tolerance_kib=0,
                    before_cache_drop_report=before,
                    after_cache_drop_report=after,
                    gpu_idle=True,
                )
            with self.assertRaisesRegex(HarnessError, "overwrite"):
                record_memory_return(
                    path,
                    baseline_mem_available_kib=10_000,
                    final_mem_available_kib=10_000,
                    tolerance_kib=0,
                    before_cache_drop_report=before,
                    after_cache_drop_report=after,
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
            dist_info = (
                root
                / "venv"
                / "site-packages"
                / f"vllm-{VLLM_ORACLE_VERSION}.dist-info"
            )
            pandas_package = root / "venv" / "site-packages" / "pandas"
            pandas_dist_info = (
                root / "venv" / "site-packages" / f"pandas-{PANDAS_VERSION}.dist-info"
            )
            for directory in (
                bin_dir,
                package / "benchmarks" / "datasets",
                package / "entrypoints" / "cli" / "benchmark",
                dist_info,
                pandas_package,
                pandas_dist_info,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            client = bin_dir / "vllm"
            python = bin_dir / "python"
            ninja = bin_dir / "ninja"
            package_init = package / "__init__.py"
            for path in (
                client,
                python,
                ninja,
                package_init,
                package / "benchmarks" / "serve.py",
                package / "benchmarks" / "datasets" / "datasets.py",
                package / "entrypoints" / "cli" / "benchmark" / "serve.py",
                dist_info / "METADATA",
                dist_info / "RECORD",
                pandas_package / "__init__.py",
                pandas_dist_info / "METADATA",
                pandas_dist_info / "RECORD",
            ):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            ninja.chmod(0o755)
            distribution = types.SimpleNamespace(
                version=VLLM_ORACLE_VERSION,
                _path=dist_info,
            )
            pandas_distribution = types.SimpleNamespace(
                version=PANDAS_VERSION,
                _path=pandas_dist_info,
            )
            module = types.SimpleNamespace(
                __version__=VLLM_ORACLE_VERSION,
                __file__=str(package_init),
            )
            pandas_module = types.SimpleNamespace(
                __version__=PANDAS_VERSION,
                __file__=str(pandas_package / "__init__.py"),
            )
            distributions = {
                "pandas": pandas_distribution,
                "vllm": distribution,
            }
            output = root / "oracle.json"
            with (
                mock.patch.dict(
                    "sys.modules",
                    {"pandas": pandas_module, "vllm": module},
                ),
                mock.patch(
                    "tools.bench.online_gate.importlib.metadata.distribution",
                    side_effect=distributions.__getitem__,
                ),
                mock.patch("tools.bench.online_gate.sys.executable", str(python)),
            ):
                result = record_oracle_manifest(output, client=client)
            self.assertEqual(result["oracle_version"], VLLM_ORACLE_VERSION)
            self.assertEqual(result["runtime_version"], VLLM_ORACLE_VERSION)
            self.assertEqual(result["bench_dependencies"], {"pandas": PANDAS_VERSION})
            self.assertEqual(len(result["artifacts"]), 12)
            self.assertEqual(
                result["artifacts"]["bench_serve"]["sha256"],
                sha256_file(package / "benchmarks" / "serve.py"),
            )
            self.assertEqual(
                result["artifacts"]["ninja"]["sha256"],
                sha256_file(ninja),
            )
            ninja.unlink()
            with (
                mock.patch.dict(
                    "sys.modules",
                    {"pandas": pandas_module, "vllm": module},
                ),
                mock.patch(
                    "tools.bench.online_gate.importlib.metadata.distribution",
                    side_effect=distributions.__getitem__,
                ),
                mock.patch("tools.bench.online_gate.sys.executable", str(python)),
                self.assertRaisesRegex(HarnessError, "ninja executable"),
            ):
                record_oracle_manifest(root / "missing-ninja.json", client=client)
            with (
                mock.patch.dict(
                    "sys.modules",
                    {"pandas": None, "vllm": module},
                ),
                mock.patch(
                    "tools.bench.online_gate.importlib.metadata.distribution",
                    return_value=distribution,
                ),
                mock.patch("tools.bench.online_gate.sys.executable", str(python)),
                self.assertRaisesRegex(HarnessError, "pinned pandas"),
            ):
                record_oracle_manifest(root / "missing-pandas.json", client=client)

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
                "ninja",
                "package_init",
                "python",
                "pandas_distribution_metadata",
                "pandas_distribution_record",
                "pandas_package_init",
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
                        "bench_dependencies": {"pandas": PANDAS_VERSION},
                        "client_contract_source_commit": VLLM_COMMIT,
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
            self.assertEqual(execution["bench_dependencies"], {"pandas": PANDAS_VERSION})
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

            ours_nsys_reports = []
            ours_nsys_sqlites = []
            ours_nsys_validations = []
            ours_kernel_summaries = []
            ours_commands = []
            ours_profile_logs = []
            for index in range(TRACE_REPETITIONS):
                report = root / f"ours-{index}.nsys-rep"
                report.write_text("trace\n", encoding="utf-8")
                sqlite_path = root / f"ours-{index}.sqlite"
                validation = root / f"ours-{index}-validation.json"
                write_nsys_sqlite(sqlite_path, model_contract=True)
                validation.write_text(
                    json.dumps(validate_nsys_trace(sqlite_path, model_key="27")),
                    encoding="utf-8",
                )
                summary = root / f"ours-{index}-summary.txt"
                summary.write_text("summary\n", encoding="utf-8")
                command = root / f"ours-{index}-command.txt"
                profile_log = root / f"ours-{index}-profile.log"
                profile_log.write_text("profile\n", encoding="utf-8")
                ours_nsys_reports.append(report)
                ours_nsys_sqlites.append(sqlite_path)
                ours_nsys_validations.append(validation)
                ours_kernel_summaries.append(summary)
                ours_commands.append(command)
                ours_profile_logs.append(profile_log)
            vllm_torch_trace = root / "vllm-trace.json.gz"
            vllm_kernel_summary = root / "vllm-kernels.json"
            vllm_torch_trace.write_text("trace\n", encoding="utf-8")
            vllm_kernel_summary.write_text("summary\n", encoding="utf-8")
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
            vllm_command = root / "vllm-command.txt"
            vllm_profile_log = root / "vllm-profile.log"
            vllm_corpus = root / "vllm-corpus.jsonl"
            for path in (vllm_command, vllm_profile_log, vllm_corpus):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            vllm_metadata = root / "vllm-metadata.json"
            vllm_metadata.write_text(
                json.dumps(
                    {
                        "admission_mode": "closed-loop",
                        "corpus": str(vllm_corpus),
                        "corpus_sha256": sha256_file(vllm_corpus),
                        "enable_prefix_caching": False,
                        "input_len": INPUT_LEN,
                        "max_concurrency": TRACE_CONCURRENCY,
                        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS["27"],
                        "max_model_len": MAX_MODEL_LEN["27"],
                        "max_num_seqs": MAX_NUM_SEQS,
                        "num_prompts": TRACE_PROMPTS,
                        "output_digest": "a" * 64,
                        "output_digests": [
                            "a" * 64,
                            "b" * 64,
                            "c" * 64,
                            "d" * 64,
                        ],
                        "output_digests_equal": False,
                        "output_len": OUTPUT_LEN,
                        "profiled_warmup_prompts": TRACE_PROMPTS,
                        "repetitions": TRACE_REPETITIONS,
                    }
                ),
                encoding="utf-8",
            )
            cache_drop_reports = []
            for index in range(3):
                path = root / f"cache-drop-{index}.json"
                write_cache_drop_report(path)
                cache_drop_reports.append(path)

            def write_ours_commands(value: str) -> None:
                for path in ours_commands:
                    path.write_text(value, encoding="utf-8")

            def record_trace():
                return record_trace_status(
                    root / "trace.json",
                    model_key="27",
                    ours_nsys_reports=ours_nsys_reports,
                    ours_nsys_sqlites=ours_nsys_sqlites,
                    ours_nsys_validations=ours_nsys_validations,
                    ours_kernel_summaries=ours_kernel_summaries,
                    ours_commands=ours_commands,
                    ours_profile_logs=ours_profile_logs,
                    ours_client_results=ours_client_results,
                    ours_client_logs=ours_client_logs,
                    vllm_torch_trace=vllm_torch_trace,
                    vllm_kernel_summary=vllm_kernel_summary,
                    vllm_command=vllm_command,
                    vllm_profile_log=vllm_profile_log,
                    vllm_metadata=vllm_metadata,
                    vllm_corpus=vllm_corpus,
                    cache_drop_reports=cache_drop_reports,
                    vllm_cpp_sha="d" * 40,
                )

            write_ours_commands("nsys server\n")
            with self.assertRaisesRegex(HarnessError, "disable prefix caching"):
                record_trace()
            write_ours_commands(
                "nsys server --max-num-seqs 32 --max-num-batched-tokens 2048 "
                "--no-enable-prefix-caching\n"
            )
            with self.assertRaisesRegex(HarnessError, "node-level CUDA graph"):
                record_trace()
            write_ours_commands(
                "nsys --cuda-graph-trace=node server --max-num-seqs 32 "
                "--max-num-batched-tokens 2048 --no-enable-prefix-caching\n"
            )
            with self.assertRaisesRegex(HarnessError, "periodically flush"):
                record_trace()
            write_ours_commands(
                "nsys --cuda-graph-trace=node --cuda-flush-interval=10000 "
                "server --max-num-seqs 32 --max-num-batched-tokens 2048 "
                "--no-enable-prefix-caching\n"
            )
            with self.assertRaisesRegex(HarnessError, "context-switch"):
                record_trace()
            write_ours_commands(
                "nsys --cuda-graph-trace=node --cuda-flush-interval=10000 "
                "--cpuctxsw=none server --max-num-seqs 32 "
                "--max-num-batched-tokens 2048 --no-enable-prefix-caching\n"
            )
            slow = valid_record(
                requests=TRACE_PROMPTS,
                concurrency=TRACE_CONCURRENCY,
            )
            slow["duration"] = 15.0
            ours_client_results[-1].write_text(json.dumps(slow), encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "duration by more than 20%"):
                record_trace()
            ours_client_results[-1].write_text(
                json.dumps(
                    valid_record(
                        requests=TRACE_PROMPTS,
                        concurrency=TRACE_CONCURRENCY,
                    )
                ),
                encoding="utf-8",
            )

            trace = record_trace()
            self.assertEqual(trace["ours_profiler"], "nsys")
            self.assertEqual(trace["vllm_profiler"], "torch-profiler")
            self.assertEqual(
                trace["trace_contract"]["cuda_graph_trace"],
                NSYS_CUDA_GRAPH_TRACE,
            )
            self.assertEqual(trace["trace_contract"]["nsys_captures"], 3)
            self.assertEqual(len(trace["nsys_validations"]), 3)
            self.assertEqual(
                trace["nsys_validations"][0]["primary_graph_family_node_counts"][
                    "normal_fp4_producer"
                ],
                144,
            )
            self.assertFalse(trace["output_repeatability"]["vllm"]["all_equal"])


if __name__ == "__main__":
    unittest.main()
