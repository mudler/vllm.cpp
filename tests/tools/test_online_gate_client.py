"""Ported command/result contracts for the CUDA online serving gate.

Upstream sources at vLLM e24d1b24:
``tests/benchmarks/test_serve_cli.py:58-132``,
``tests/benchmarks/test_custom_dataset_seed.py:24-75``, and
``vllm/benchmarks/serve.py:1188-1284,2082-2284``.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import pathlib
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    FLASHINFER_VERSION,
    NSYS_CAPTURE_RANGE,
    NSYS_CAPTURE_RANGE_END,
    NSYS_CUDA_FLUSH_INTERVAL_MS,
    NSYS_CUDA_GRAPH_TRACE,
    NSYS_PRODUCT_VERSION,
    OUTPUT_LEN,
    PANDAS_VERSION,
    TRACE_CONCURRENCY,
    TRACE_CAPTURE_GRAPH_REPLAYS,
    TRACE_PRIMARY_GRAPH_CONTRACTS,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
    TRACE_REQUIRED_ENV,
    VLLM_DECODE_FAMILY_CONTRACTS,
    VLLM_GENERATION_WINDOW_CONTRACTS,
    VLLM_ORACLE_VERSION,
    OnlineRun,
    build_client_command,
    build_plan,
    prepare_corpus_views,
    record_execution_manifest,
    record_memory_return,
    record_model_gate,
    record_oracle_manifest,
    record_profile_control,
    record_trace_status,
    summarize_nsys_kernels,
    validate_nsys_trace,
    validate_plan,
    validate_raw_result,
    _fingerprint_tree,
    _summarize_torch_trace,
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
    report_path: pathlib.Path | None = None,
    family_drift: bool = False,
    lost_events: bool = False,
    model_contract: bool = False,
    orphan_child: bool = False,
    profiler_start_count: int = TRACE_CAPTURE_GRAPH_REPLAYS,
    signature_drift: bool = False,
    uneven_replays: bool = False,
) -> None:
    report_path = report_path or path.with_suffix(".nsys-rep")
    if not report_path.exists():
        report_path.write_text("synthetic Nsight report\n", encoding="utf-8")
    capture_uuid = str(uuid.uuid4())
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE DIAGNOSTIC_EVENT "
            "(timestamp INTEGER, severity INTEGER, text TEXT)"
        )
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "greenContextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "globalPid INTEGER, demangledName INTEGER, shortName INTEGER, "
            "mangledName INTEGER, launchType INTEGER, cacheConfig INTEGER, "
            "registersPerThread INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
            "blockX INTEGER, blockY INTEGER, blockZ INTEGER, staticSharedMemory INTEGER, "
            "dynamicSharedMemory INTEGER, localMemoryPerThread INTEGER, "
            "localMemoryTotal INTEGER, gridId INTEGER, sharedMemoryExecuted INTEGER, "
            "graphNodeId INTEGER, sharedMemoryLimitConfig INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, eventClass INTEGER, globalTid INTEGER, "
            "correlationId INTEGER, nameId INTEGER, returnValue INTEGER, callchainId INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
            "(start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "greenContextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "globalPid INTEGER, bytes INTEGER, copyKind INTEGER, deprecatedSrcId INTEGER, "
            "srcKind INTEGER, dstKind INTEGER, srcDeviceId INTEGER, srcContextId INTEGER, "
            "dstDeviceId INTEGER, dstContextId INTEGER, migrationCause INTEGER, "
            "graphNodeId INTEGER, virtualAddress INTEGER, copyCount INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET "
            "(start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "greenContextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "globalPid INTEGER, value INTEGER, bytes INTEGER, graphNodeId INTEGER, "
            "memKind INTEGER)"
        )
        connection.execute("CREATE TABLE META_DATA_CAPTURE (name TEXT, value TEXT)")
        connection.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
        connection.execute(
            "CREATE TABLE TARGET_INFO_SESSION_START_TIME "
            "(utcEpochNs INTEGER, utcTime TEXT, localTime TEXT)"
        )
        connection.execute(
            "INSERT INTO META_DATA_CAPTURE VALUES (?, ?)",
            ("PROFILING_SESSION_UUID", capture_uuid),
        )
        connection.executemany(
            "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
            [
                ("EXPORT_PRODUCT_NAME", "NVIDIA Nsight Systems"),
                ("EXPORT_PRODUCT_VERSION", NSYS_PRODUCT_VERSION),
                ("EXPORT_PARAM_LAZY", "false"),
                ("EXPORT_PARAM_INPUT_PATH_ABS", str(report_path.resolve())),
                ("EXPORT_PARAM_OUTPUT_PATH_ABS", str(path.resolve())),
            ],
        )
        connection.execute(
            "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, ?, ?)",
            (1_783_950_501_877_615_336, "2026-07-13T13:48:21", "2026-07-13T15:48:21"),
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
            memcpy_nodes = 7
        else:
            node_names = ["kernel-a", "kernel-b"]
            memcpy_nodes = 1
        launch_name = "cudaGraphLaunch_v10000"
        profiler_start_name = "cuProfilerStart"
        string_ids = {
            name: index
            for index, name in enumerate(
                sorted(set(node_names + [launch_name, profiler_start_name])), start=1
            )
        }
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(identifier, name) for name, identifier in string_ids.items()],
        )
        launch_ids = [1001, 1002, 1003, 1004]
        for launch_index, correlation_id in enumerate(launch_ids):
            if launch_index < profiler_start_count:
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        launch_index * 100_000,
                        launch_index * 100_000 + 10,
                        0,
                        1,
                        2001 + launch_index,
                        string_ids[profiler_start_name],
                        0,
                        None,
                    ),
                )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    launch_index * 100_000 + 20,
                    launch_index * 100_000 + 100,
                    0,
                    1,
                    correlation_id,
                    string_ids[launch_name],
                    0,
                    None,
                ),
            )
            for graph_node_id, name in enumerate(node_names, start=1):
                if uneven_replays and graph_node_id == len(node_names) and launch_index == 3:
                    continue
                row_correlation = 9999 if orphan_child and graph_node_id == 1 else correlation_id
                grid_x = 2 if signature_drift and graph_node_id == 1 and launch_index == 3 else 1
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                    + ",".join("?" for _ in range(28))
                    + ")",
                    (
                        launch_index * 100_000 + graph_node_id * 10,
                        launch_index * 100_000 + graph_node_id * 10 + 5,
                        0,
                        1,
                        None,
                        7,
                        row_correlation,
                        1,
                        string_ids[name],
                        string_ids[name],
                        None,
                        0,
                        0,
                        32,
                        grid_x,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        graph_node_id,
                        None,
                        graph_node_id,
                        None,
                    ),
                )
            for node_index in range(memcpy_nodes):
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES ("
                    + ",".join("?" for _ in range(21))
                    + ")",
                    (
                        1,
                        2,
                        0,
                        1,
                        None,
                        7,
                        correlation_id,
                        1,
                        4096 + node_index,
                        1,
                        None,
                        1,
                        2,
                        0,
                        1,
                        0,
                        1,
                        None,
                        2001 + node_index,
                        None,
                        1,
                    ),
                )
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES ("
                + ",".join("?" for _ in range(12))
                + ")",
                (1, 2, 0, 1, None, 7, correlation_id, 1, 0, 4096, 3001, 1),
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


def write_profile_log(
    path: pathlib.Path,
    *,
    fixture: pathlib.Path,
    native_target: pathlib.Path,
    server_pid: int,
) -> None:
    document = json.loads(fixture.read_text(encoding="utf-8"))
    plans = []
    for key, value in document.items():
        if key == "_metadata":
            continue
        shapes = ast.literal_eval(key)[2]
        m, packed_k = shapes[0]
        _, n = shapes[1]
        plans.append((m, n, packed_k * 2, value[1]))
    plans.sort()
    lines = [
        "[VT_FP4_CACHE] prepared mode=read-only "
        f"native={native_target} flashinfer={fixture} loaded=64 "
        "(flashinfer=64 native=0) rejected=0 delay_us=5000 "
        "metadata=2e429d4cd3977f0f selected=64",
        "[VT_FP4_CACHE] complete mode=read-only loaded=64 tuned=0 "
        "rejected=0 saved=0 selected=64 metadata=2e429d4cd3977f0f",
        *[
            f"[VT_FP4_CACHE] selected M={m} N={n} K={k} tactic={tactic}"
            for m, n, k, tactic in plans
        ],
        "[VT_FP4_AUTOTUNE] pre-serve warmup complete max_tokens=2048 "
        "profiles_requested=0 profiles_tuned=0 cached_plans=64",
        f"[VT_CUDA_PROFILE] ready pid={server_pid} signal=SIGUSR2 target_replays=4",
        f"[VT_BENCH_SHUTDOWN] ready pid={server_pid} control=fifo",
        "[VT_CUDA_PROFILE] started target_replays=4 graph=0x1234 "
        "real_batch=16 padded_batch=16 prior_replays=128",
        "[VT_CUDA_PROFILE] stopped captured_replays=4 graph=0x1234",
        "[VT_BENCH_SHUTDOWN] requested control=fifo",
        "[VT_BENCH_SHUTDOWN] completed control=fifo",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_client_command_log(
    path: pathlib.Path,
    *,
    client: pathlib.Path,
    corpus: pathlib.Path,
    result: pathlib.Path,
    prompts: int,
    warmups: int,
) -> None:
    command = [
        str(client),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        "http://127.0.0.1:8001",
        "--endpoint",
        "/v1/completions",
        "--model",
        "gate",
        "--tokenizer",
        str(corpus.parent),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(corpus),
        "--custom-output-len",
        str(OUTPUT_LEN),
        "--skip-chat-template",
        "--disable-shuffle",
        "--num-prompts",
        str(prompts),
        "--max-concurrency",
        str(TRACE_CONCURRENCY),
        "--request-rate",
        "inf",
        "--num-warmups",
        str(warmups),
        "--ready-check-timeout-sec",
        "0",
        "--seed",
        "0",
        "--ignore-eos",
        "--temperature",
        "0",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result.parent),
        "--result-filename",
        result.name,
        "--disable-tqdm",
    ]
    path.write_text(f"command: {shlex.join(command)}\npassed\n", encoding="utf-8")


def write_vllm_decode_trace(path: pathlib.Path, *, model_key: str = "27") -> None:
    events = [
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": "execute_context_0(0)_generation_16(16)",
            "pid": 0,
            "tid": 7,
            "ts": 1_000.0,
            "dur": 100_000.0,
        }
    ]
    timestamp = 2_000.0
    for _, (name, count) in VLLM_DECODE_FAMILY_CONTRACTS[model_key].items():
        for _ in range(count):
            events.append(
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": name,
                    "pid": 0,
                    "tid": 7,
                    "ts": timestamp,
                    "dur": 1.0,
                    "args": {
                        "block": [256, 1, 1],
                        "grid": [128, 1, 1],
                        "registers per thread": 32,
                        "shared memory": 0,
                    },
                }
            )
            timestamp += 2.0
    document = {"traceEvents": events}
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as output:
            json.dump(document, output)
    else:
        path.write_text(json.dumps(document), encoding="utf-8")


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
            self.assertEqual(
                command[command.index("--ready-check-timeout-sec") + 1], "0"
            )
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
            {"flashinfer": FLASHINFER_VERSION, "pandas": PANDAS_VERSION},
        )
        self.assertIn(
            "summary-<model>/{all-runs,ratios}.json",
            plan["required_artifacts"],
        )
        self.assertEqual(
            plan["planned_commands"]["trace_model"][1],
            "--trace-only",
        )
        self.assertEqual(
            plan["planned_commands"]["trace_model"][2:6],
            ["--model", "27", "--snapshot", "<27_MODEL_SNAPSHOT>"],
        )
        self.assertNotIn("<27|35>", plan["planned_commands"]["trace_model"])
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

    def test_trace_driver_builds_before_requiring_the_server_binary(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        configured_preflight = script.index(
            '[[ -n ${build_dir} && -f ${build_dir}/CMakeCache.txt ]]'
        )
        build = script.index('if ! "${build_cmd[@]}" >"${build_log}" 2>&1; then')
        executable_postflight = script.index(
            '[[ -x ${build_dir}/examples/server ]]'
        )
        self.assertLess(configured_preflight, build)
        self.assertLess(build, executable_postflight)

    def test_trace_driver_tracks_the_actual_nsys_target_ancestry(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("${launcher_ppid} == \"${nsys_pid}\"", script)
        self.assertIn("${launcher_comm} == nsys-launcher", script)
        self.assertIn("${server_ppid} == \"${launcher_pid}\"", script)
        self.assertIn("${server_pgid} == \"${server_pid}\"", script)
        self.assertIn("${server_sid} == \"${server_pid}\"", script)
        self.assertIn("mkfifo --mode=600", script)
        self.assertIn("printf 'Q' >\"${shutdown_fifo}\"", script)
        self.assertNotIn('kill -USR1 "${server_pid}"', script)
        self.assertIn('kill -TERM -- "-${profiled_pgid}"', script)
        self.assertNotIn("server_pgid} == \"${nsys_pid}", script)

    def test_profile_control_requires_graceful_shutdown_lifecycle(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        fixture = (
            repo
            / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            profile_log = root / "profile.log"
            native = root / "native-must-not-exist.json"
            write_profile_log(
                profile_log,
                fixture=fixture,
                native_target=native,
                server_pid=6000,
            )
            profile_log.write_text(
                profile_log.read_text(encoding="utf-8").replace(
                    "[VT_BENCH_SHUTDOWN] completed control=fifo\n", ""
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                HarnessError, "complete graceful-shutdown lifecycle"
            ):
                record_profile_control(
                    root / "control.json",
                    profile_log=profile_log,
                    nsys_pid=5000,
                    nsys_pgid=5000,
                    nsys_sid=5000,
                    nsys_exit_status=0,
                    launcher_pid=5500,
                    launcher_ppid=5000,
                    launcher_pgid=5000,
                    launcher_sid=5000,
                    launcher_comm="nsys-launcher",
                    server_pid=6000,
                    server_ppid=5500,
                    server_pgid=6000,
                    server_sid=6000,
                    shutdown_fifo=root / "shutdown.fifo",
                )

    def test_profile_control_rejects_a_flattened_nsys_process_group(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        fixture = (
            repo
            / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            profile_log = root / "profile.log"
            native = root / "native-must-not-exist.json"
            write_profile_log(
                profile_log,
                fixture=fixture,
                native_target=native,
                server_pid=6000,
            )
            with self.assertRaisesRegex(
                HarnessError, "profiled server is not the Nsight target-session leader"
            ):
                record_profile_control(
                    root / "control.json",
                    profile_log=profile_log,
                    nsys_pid=5000,
                    nsys_pgid=5000,
                    nsys_sid=5000,
                    nsys_exit_status=0,
                    launcher_pid=5500,
                    launcher_ppid=5000,
                    launcher_pgid=5000,
                    launcher_sid=5000,
                    launcher_comm="nsys-launcher",
                    server_pid=6000,
                    server_ppid=5500,
                    server_pgid=5000,
                    server_sid=5000,
                    shutdown_fifo=root / "shutdown.fifo",
                )

    def test_profile_control_requires_removed_shutdown_fifo(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        fixture = (
            repo
            / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            profile_log = root / "profile.log"
            native = root / "native-must-not-exist.json"
            shutdown_fifo = root / "shutdown.fifo"
            os.mkfifo(shutdown_fifo)
            write_profile_log(
                profile_log,
                fixture=fixture,
                native_target=native,
                server_pid=6000,
            )
            with self.assertRaisesRegex(
                HarnessError, "shutdown FIFO is not an absent absolute path"
            ):
                record_profile_control(
                    root / "control.json",
                    profile_log=profile_log,
                    nsys_pid=5000,
                    nsys_pgid=5000,
                    nsys_sid=5000,
                    nsys_exit_status=0,
                    launcher_pid=5500,
                    launcher_ppid=5000,
                    launcher_pgid=5000,
                    launcher_sid=5000,
                    launcher_comm="nsys-launcher",
                    server_pid=6000,
                    server_ppid=5500,
                    server_pgid=6000,
                    server_sid=6000,
                    shutdown_fifo=shutdown_fifo,
                )

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
        self.assertIn(f"--capture-range-end={NSYS_CAPTURE_RANGE_END}", script)
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
            self.assertEqual(
                result["primary_graph_replay_count"], TRACE_CAPTURE_GRAPH_REPLAYS
            )
            self.assertEqual(
                result["profiler_range_start_count"], TRACE_CAPTURE_GRAPH_REPLAYS
            )

            lost = root / "lost.sqlite"
            write_nsys_sqlite(lost, lost_events=True)
            with self.assertRaisesRegex(HarnessError, "not lossless"):
                validate_nsys_trace(lost)

            uneven = root / "uneven.sqlite"
            write_nsys_sqlite(uneven, uneven_replays=True)
            with self.assertRaisesRegex(HarnessError, "uneven replay counts"):
                validate_nsys_trace(uneven)

            missing_range = root / "missing-range.sqlite"
            write_nsys_sqlite(missing_range, profiler_start_count=3)
            with self.assertRaisesRegex(HarnessError, "cuProfilerStart rows"):
                validate_nsys_trace(missing_range)

            orphan = root / "orphan.sqlite"
            write_nsys_sqlite(orphan, orphan_child=True)
            with self.assertRaisesRegex(HarnessError, "direct cudaGraphLaunch"):
                validate_nsys_trace(orphan)

            signature_drift = root / "signature-drift.sqlite"
            write_nsys_sqlite(signature_drift, signature_drift=True)
            with self.assertRaisesRegex(HarnessError, "signature differs"):
                validate_nsys_trace(signature_drift)

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
            flashinfer_package = root / "venv" / "site-packages" / "flashinfer"
            flashinfer_dist_info = (
                root
                / "venv"
                / "site-packages"
                / f"flashinfer_python-{FLASHINFER_VERSION}.dist-info"
            )
            cutlass = flashinfer_package / "data" / "cutlass"
            for directory in (
                bin_dir,
                package / "benchmarks" / "datasets",
                package / "entrypoints" / "cli" / "benchmark",
                dist_info,
                pandas_package,
                pandas_dist_info,
                flashinfer_package,
                flashinfer_dist_info,
                cutlass / "include" / "cutlass",
                cutlass / "tools" / "util" / "include",
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
                flashinfer_package / "__init__.py",
                flashinfer_dist_info / "METADATA",
                flashinfer_dist_info / "RECORD",
                cutlass / "include" / "cutlass" / "cutlass.h",
                cutlass / "tools" / "util" / "include" / "helper.h",
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
            flashinfer_distribution = types.SimpleNamespace(
                version=FLASHINFER_VERSION,
                _path=flashinfer_dist_info,
            )
            module = types.SimpleNamespace(
                __version__=VLLM_ORACLE_VERSION,
                __file__=str(package_init),
            )
            pandas_module = types.SimpleNamespace(
                __version__=PANDAS_VERSION,
                __file__=str(pandas_package / "__init__.py"),
            )
            flashinfer_module = types.SimpleNamespace(
                __version__=FLASHINFER_VERSION,
                __file__=str(flashinfer_package / "__init__.py"),
            )
            distributions = {
                "flashinfer-python": flashinfer_distribution,
                "pandas": pandas_distribution,
                "vllm": distribution,
            }
            output = root / "oracle.json"
            with (
                mock.patch.dict(
                    "sys.modules",
                    {
                        "flashinfer": flashinfer_module,
                        "pandas": pandas_module,
                        "vllm": module,
                    },
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
            self.assertEqual(
                result["bench_dependencies"],
                {"flashinfer": FLASHINFER_VERSION, "pandas": PANDAS_VERSION},
            )
            self.assertEqual(len(result["artifacts"]), 15)
            self.assertEqual(result["cutlass_source_tree"]["file_count"], 2)
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
                    {
                        "flashinfer": flashinfer_module,
                        "pandas": pandas_module,
                        "vllm": module,
                    },
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
                    {
                        "flashinfer": flashinfer_module,
                        "pandas": None,
                        "vllm": module,
                    },
                ),
                mock.patch(
                    "tools.bench.online_gate.importlib.metadata.distribution",
                    side_effect=distributions.__getitem__,
                ),
                mock.patch("tools.bench.online_gate.sys.executable", str(python)),
                self.assertRaisesRegex(HarnessError, "pinned pandas"),
            ):
                record_oracle_manifest(root / "missing-pandas.json", client=client)

    @mock.patch.dict(
        VLLM_GENERATION_WINDOW_CONTRACTS,
        {"27": {"all": 1, "clean": 1}},
    )
    def test_execution_model_and_trace_records_hash_their_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            build = root / "build"
            (build / "examples").mkdir(parents=True)
            source = root / "source"
            (source / "src/vt/cuda").mkdir(parents=True)
            (source / "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu").write_text(
                "// fixture\n", encoding="utf-8"
            )
            fixture = source / (
                "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
            )
            fixture.parent.mkdir(parents=True)
            fixture.write_bytes(
                (
                    pathlib.Path(__file__).resolve().parents[2]
                    / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
                ).read_bytes()
            )
            cutlass = root / "oracle-cutlass"
            (cutlass / "include/cutlass").mkdir(parents=True)
            (cutlass / "tools/util/include").mkdir(parents=True)
            (cutlass / "include/cutlass/cutlass.h").write_text(
                "cutlass\n", encoding="utf-8"
            )
            (cutlass / "tools/util/include/helper.h").write_text(
                "helper\n", encoding="utf-8"
            )
            snapshot = root / "890bdef7a42feba6d83b6e17a03315c694112f2a"
            snapshot.mkdir()
            bin_dir = root / "oracle-bin"
            bin_dir.mkdir()
            client = bin_dir / "vllm"
            build_command = root / "build-command.txt"
            build_log = root / "build.log"
            configure_log = root / "configure.log"
            cuda_compiler = root / "cuda" / "bin" / "nvcc"
            cuda_compiler.parent.mkdir(parents=True)
            for path in (
                build_command,
                build_log,
                configure_log,
                snapshot / "config.json",
                snapshot / "tokenizer.json",
                snapshot / "model-00001-of-00001.safetensors",
                client,
                cuda_compiler,
            ):
                path.write_text(f"{path.name}\n", encoding="utf-8")
            configure_log.write_text(
                "-- The CUDA compiler identification is NVIDIA 13.0.88\n"
                f"-- CUTLASS found at {cutlass}; enabling sm120a NVFP4 cutlass GEMM\n",
                encoding="utf-8",
            )
            cmake_cache = build / "CMakeCache.txt"
            cmake_cache.write_text(
                "\n".join(
                    (
                        "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo",
                        f"CMAKE_CUDA_COMPILER:FILEPATH={cuda_compiler}",
                        "CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
                        f"CMAKE_MAKE_PROGRAM:FILEPATH={root / 'oracle-files/ninja'}",
                        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}",
                        "VLLM_CPP_BENCH_PROFILE_CONTROL:BOOL=ON",
                        "VLLM_CPP_BUILD_TESTS:BOOL=ON",
                        "VLLM_CPP_CUDA:STRING=ON",
                        "VLLM_CPP_CUDA_ARCHITECTURES:STRING=121a",
                        f"VLLM_CPP_CUTLASS_DIR:PATH={cutlass}",
                        "VLLM_CPP_FLASH_ATTN:BOOL=ON",
                        "VLLM_CPP_SERVER:BOOL=ON",
                        "VLLM_CPP_TRITON:BOOL=ON",
                        "VLLM_CPP_TRITON_REGEN:BOOL=OFF",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            compile_command = (
                "nvcc -DVLLM_CPP_FLASH_ATTN -DVLLM_CPP_TRITON=1 "
                "-DVLLM_CPP_TRITON_CHUNKO_BF16=1 "
                "-DVT_CUTLASS_NVFP4=1 -DVT_BENCH_PROFILE_CONTROL=1 "
                '"--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]" '
                f"-isystem {cutlass / 'include'} "
                f"-isystem {cutlass / 'tools/util/include'} -c "
                f"{source / 'src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu'}"
            )
            (build / "compile_commands.json").write_text(
                json.dumps(
                    [
                        {
                            "command": compile_command,
                            "directory": str(build),
                            "file": str(
                                source / "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu"
                            ),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (build / "examples/server").write_bytes(
                b"MatmulNvfp4Cutlass\0[VT_FP4_CACHE] prepared\0"
                b"[VT_CUDA_PROFILE] started\0[VT_BENCH_SHUTDOWN] ready\0"
            )
            oracle_files = {}
            for name in (
                "bench_datasets",
                "bench_serve",
                "cli_bench_serve",
                "client",
                "distribution_metadata",
                "distribution_record",
                "flashinfer_distribution_metadata",
                "flashinfer_distribution_record",
                "flashinfer_package_init",
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
                        "bench_dependencies": {
                            "flashinfer": FLASHINFER_VERSION,
                            "pandas": PANDAS_VERSION,
                        },
                        "client_contract_source_commit": VLLM_COMMIT,
                        "cutlass_source_tree": _fingerprint_tree(cutlass),
                        "oracle_version": VLLM_ORACLE_VERSION,
                        "runtime_version": VLLM_ORACLE_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            native_target = root / "native-must-not-exist.json"
            trace_environment = {
                **TRACE_REQUIRED_ENV,
                "VT_FP4_AUTOTUNE_CACHE_PATH": str(native_target),
                "VT_FP4_FLASHINFER_CACHE_PATH": str(fixture),
            }
            with (
                mock.patch.dict(os.environ, trace_environment, clear=False),
                mock.patch(
                    "tools.bench.online_gate.DGX_CUDA_COMPILER", cuda_compiler
                ),
            ):
                def record(path: pathlib.Path) -> dict:
                    return record_execution_manifest(
                        path,
                        model_key="27",
                        vllm_cpp_sha="d" * 40,
                        build_dir=build,
                        client=client,
                        snapshot=snapshot,
                        configure_log=configure_log,
                        build_command=build_command,
                        build_log=build_log,
                        oracle_manifest=oracle_manifest,
                        port=8001,
                        num_blocks=4736,
                        max_num_seqs=MAX_NUM_SEQS,
                        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS["27"],
                        profile_control=True,
                    )

                exact_cache = cmake_cache.read_text(encoding="utf-8")
                cmake_cache.write_text(
                    exact_cache.replace(
                        "VLLM_CPP_TRITON:BOOL=ON", "VLLM_CPP_TRITON:BOOL=OFF"
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(HarnessError, "VLLM_CPP_TRITON"):
                    record(root / "wrong-triton-execution.json")
                cmake_cache.write_text(
                    exact_cache.replace(
                        "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo",
                        "CMAKE_BUILD_TYPE:STRING=Release",
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(HarnessError, "CMAKE_BUILD_TYPE"):
                    record(root / "wrong-build-type-execution.json")
                cmake_cache.write_text(exact_cache, encoding="utf-8")
                execution = record(root / "execution.json")
            self.assertEqual(execution["model_key"], "27")
            self.assertEqual(execution["vllm_oracle_version"], VLLM_ORACLE_VERSION)
            self.assertEqual(execution["bench_dependencies"], {
                "flashinfer": FLASHINFER_VERSION,
                "pandas": PANDAS_VERSION,
            })
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
            ours_profile_controls = []
            vllm_corpus = root / "vllm-corpus.jsonl"
            vllm_corpus.write_text('{"prompt":"fixture"}\n', encoding="utf-8")
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
                summary.write_text(
                    json.dumps(summarize_nsys_kernels(sqlite_path)), encoding="utf-8"
                )
                command = root / f"ours-{index}-command.txt"
                profile_log = root / f"ours-{index}-profile.log"
                native = root / f"native-{index}-must-not-exist.json"
                server_pid = 6000 + index
                nsys_pid = 5000 + index
                launcher_pid = 5500 + index
                write_profile_log(
                    profile_log,
                    fixture=fixture,
                    native_target=native,
                    server_pid=server_pid,
                )
                command_environment = {
                    **TRACE_REQUIRED_ENV,
                    "VT_FP4_AUTOTUNE_CACHE_PATH": str(native),
                    "VT_FP4_FLASHINFER_CACHE_PATH": str(fixture),
                }
                prefix = root / f"ours-{index}"
                shutdown_fifo = pathlib.Path(f"{prefix}-shutdown.fifo")
                profile_command = [
                    "env",
                    *[f"{name}={value}" for name, value in command_environment.items()],
                    "nsys",
                    "profile",
                    "--trace=cuda",
                    f"--capture-range={NSYS_CAPTURE_RANGE}",
                    f"--capture-range-end={NSYS_CAPTURE_RANGE_END}",
                    "--flush-on-cudaprofilerstop=true",
                    f"--cuda-flush-interval={NSYS_CUDA_FLUSH_INTERVAL_MS}",
                    f"--cuda-graph-trace={NSYS_CUDA_GRAPH_TRACE}",
                    "--cuda-event-trace=false",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--stats=false",
                    "--kill=none",
                    "--force-overwrite=true",
                    "--output",
                    str(prefix),
                    str(build / "examples/server"),
                    "--model",
                    str(snapshot),
                    "--port",
                    "8001",
                    "--num-blocks",
                    "4736",
                    "--max-num-seqs",
                    str(MAX_NUM_SEQS),
                    "--max-num-batched-tokens",
                    str(MAX_NUM_BATCHED_TOKENS["27"]),
                    "--max-model-len",
                    str(MAX_MODEL_LEN["27"]),
                    "--no-enable-prefix-caching",
                    "--cuda-profile-graph-replays",
                    str(TRACE_CAPTURE_GRAPH_REPLAYS),
                    "--benchmark-shutdown-fifo",
                    str(shutdown_fifo),
                    "--served-model-name",
                    "gate",
                ]
                command.write_text(shlex.join(profile_command) + "\n", encoding="utf-8")
                control = root / f"ours-{index}-control.json"
                record_profile_control(
                    control,
                    profile_log=profile_log,
                    nsys_pid=nsys_pid,
                    nsys_pgid=nsys_pid,
                    nsys_sid=nsys_pid,
                    nsys_exit_status=0,
                    launcher_pid=launcher_pid,
                    launcher_ppid=nsys_pid,
                    launcher_pgid=nsys_pid,
                    launcher_sid=nsys_pid,
                    launcher_comm="nsys-launcher",
                    server_pid=server_pid,
                    server_ppid=launcher_pid,
                    server_pgid=server_pid,
                    server_sid=server_pid,
                    shutdown_fifo=shutdown_fifo,
                )
                ours_nsys_reports.append(report)
                ours_nsys_sqlites.append(sqlite_path)
                ours_nsys_validations.append(validation)
                ours_kernel_summaries.append(summary)
                ours_commands.append(command)
                ours_profile_logs.append(profile_log)
                ours_profile_controls.append(control)
            vllm_profile_dir = root / "vllm-profile"
            vllm_profile_dir.mkdir()
            vllm_torch_trace = vllm_profile_dir / "vllm-trace.json.gz"
            vllm_kernel_summary = root / "vllm-kernels.json"
            write_vllm_decode_trace(vllm_torch_trace)
            vllm_kernel_summary.write_text(
                json.dumps(_summarize_torch_trace(vllm_torch_trace, model_key="27")),
                encoding="utf-8",
            )
            ours_client_results = []
            ours_client_logs = []
            ours_probe_results = []
            ours_probe_logs = []
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
                write_client_command_log(
                    log_path,
                    client=client,
                    corpus=vllm_corpus,
                    result=result_path,
                    prompts=TRACE_PROMPTS,
                    warmups=TRACE_CONCURRENCY,
                )
                ours_client_results.append(result_path)
                ours_client_logs.append(log_path)
                probe_result = root / f"ours-probe-{index}.json"
                probe_result.write_text(
                    json.dumps(
                        valid_record(
                            requests=TRACE_CONCURRENCY,
                            concurrency=TRACE_CONCURRENCY,
                        )
                    ),
                    encoding="utf-8",
                )
                probe_log = root / f"ours-probe-{index}.log"
                write_client_command_log(
                    probe_log,
                    client=client,
                    corpus=vllm_corpus,
                    result=probe_result,
                    prompts=TRACE_CONCURRENCY,
                    warmups=0,
                )
                ours_probe_results.append(probe_result)
                ours_probe_logs.append(probe_log)
            vllm_command = root / "vllm-command.txt"
            vllm_profile_log = root / "vllm-profile.log"
            vllm_profile_log.write_text("profile passed\n", encoding="utf-8")
            vllm_metadata = root / "vllm-metadata.json"
            oracle_python = pathlib.Path(
                execution["artifacts"]["oracle:python"]["path"]
            )
            vllm_profile_command = [
                "env",
                f"PATH={oracle_python.parent}",
                str(oracle_python),
                str(source / "tools/bench/profile_vllm_online_gate.py"),
                "--model",
                str(snapshot),
                "--corpus",
                str(vllm_corpus),
                "--profile-dir",
                str(vllm_profile_dir),
                "--metadata",
                str(vllm_metadata),
                "--num-prompts",
                str(TRACE_PROMPTS),
                "--max-concurrency",
                str(TRACE_CONCURRENCY),
                "--max-num-seqs",
                str(MAX_NUM_SEQS),
                "--max-num-batched-tokens",
                str(MAX_NUM_BATCHED_TOKENS["27"]),
                "--repetitions",
                str(TRACE_REPETITIONS),
            ]
            vllm_command.write_text(
                shlex.join(vllm_profile_command) + "\n", encoding="utf-8"
            )
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
                        "model": str(snapshot),
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
                        "profile_dir": str(vllm_profile_dir),
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
                    ours_profile_controls=ours_profile_controls,
                    ours_client_results=ours_client_results,
                    ours_client_logs=ours_client_logs,
                    ours_probe_results=ours_probe_results,
                    ours_probe_logs=ours_probe_logs,
                    vllm_torch_trace=vllm_torch_trace,
                    vllm_kernel_summary=vllm_kernel_summary,
                    vllm_command=vllm_command,
                    vllm_profile_log=vllm_profile_log,
                    vllm_metadata=vllm_metadata,
                    vllm_corpus=vllm_corpus,
                    cache_drop_reports=cache_drop_reports,
                    execution_manifest=root / "execution.json",
                    vllm_cpp_sha="d" * 40,
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
