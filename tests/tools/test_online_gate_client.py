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
    TRACE_GDN_BA_ENV,
    TRACE_GDN_BA_MODES,
    TRACE_GDN_PACKED_ENV,
    TRACE_GDN_PACKED_MODES,
    TRACE_PRIMARY_GRAPH_CONTRACTS,
    TRACE_PRIMARY_GRAPH_CONTRACTS_BY_BATCH,
    TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_BA_MODE,
    TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_PACKED_MODE,
    TRACE_PROMPTS,
    TRACE_RANGE_REPORTS,
    TRACE_REPETITIONS,
    TRACE_CLEAN_FIXED_ENV,
    TRACE_REQUIRED_ENV,
    TRACE_SYSTEM_PATH,
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
    trace_primary_graph_contract,
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
    capture_uuid: str | None = None,
    report_path: pathlib.Path | None = None,
    family_drift: bool = False,
    gdn_ba_mode: str | None = None,
    gdn_packed_mode: str | None = None,
    lost_events: bool = False,
    model_contract: bool = False,
    model_batch: int = TRACE_CONCURRENCY,
    orphan_child: bool = False,
    profiler_start_count: int | None = None,
    range_index: int = 1,
    signature_drift: bool = False,
    uneven_replays: bool = False,
    collected_event_delta: int = 0,
    gdn_packed_ba_drift: bool = False,
    unrelated_drift: bool = False,
) -> None:
    if not 1 <= range_index <= TRACE_CAPTURE_GRAPH_REPLAYS:
        raise ValueError("synthetic Nsight range index is invalid")
    if profiler_start_count is None:
        profiler_start_count = 1 if range_index == 1 else 0
    if profiler_start_count not in (0, 1):
        raise ValueError("synthetic Nsight profiler start count is invalid")
    report_path = report_path or path.with_suffix(".nsys-rep")
    if not report_path.exists():
        report_path.write_text("synthetic Nsight report\n", encoding="utf-8")
    capture_uuid = capture_uuid or str(uuid.uuid4())
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE DIAGNOSTIC_EVENT "
            "(timestamp INTEGER, timestampType INTEGER, source INTEGER, "
            "severity INTEGER, text TEXT, globalPid INTEGER)"
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
            "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
            "(start INTEGER, end INTEGER, deviceId INTEGER, contextId INTEGER, "
            "greenContextId INTEGER, streamId INTEGER, correlationId INTEGER, "
            "globalPid INTEGER, deprecatedSyncType INTEGER, syncType INTEGER, "
            "eventId INTEGER, eventSyncId INTEGER)"
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
            "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, ?, ?, ?, ?, ?)",
            (
                1,
                1,
                1,
                3,
                "CUDA device 0: Unified Memory trace is not supported by the "
                "current driver version or configuration.",
                1,
            ),
        )
        if model_contract:
            contract = trace_primary_graph_contract(
                "27",
                model_batch,
                gdn_ba_mode=gdn_ba_mode,
                gdn_packed_mode=gdn_packed_mode,
            )
            node_names = []
            for family, (pattern, count) in contract["families"].items():
                if family_drift and family == "normal_fp4_producer":
                    count -= 1
                if family == "bf16_gemm" and gdn_packed_mode is not None:
                    node_names.extend(
                        [f"{pattern}_gdn_ba_{gdn_packed_mode}"] * 48
                    )
                    count -= 48
                node_names.extend([pattern] * count)
            node_names.extend(
                ["unclassified-kernel"]
                * (contract["node_count"] - len(node_names))
            )
            if unrelated_drift:
                node_names[node_names.index("unclassified-kernel")] = (
                    "unclassified-kernel-drift"
                )
            memcpy_nodes = 7
        else:
            node_names = ["kernel-a", "kernel-b"]
            memcpy_nodes = 1
        launch_name = "cudaGraphLaunch_v10000"
        device_sync_name = "cudaDeviceSynchronize_v3020"
        profiler_start_name = "cuProfilerStart"
        string_ids = {
            name: index
            for index, name in enumerate(
                sorted(
                    set(
                        node_names
                        + [launch_name, device_sync_name, profiler_start_name]
                    )
                ),
                start=1,
            )
        }
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(identifier, name) for name, identifier in string_ids.items()],
        )
        launch_ids = [1000 + range_index]
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
                row_correlation = 9999 if orphan_child and graph_node_id == 1 else correlation_id
                if "_gdn_ba_" in name:
                    drift_this_ba = (
                        gdn_packed_ba_drift
                        and graph_node_id == node_names.index(name) + 1
                    )
                    grid_x = 7 if drift_this_ba else 8
                else:
                    grid_x = 2 if signature_drift and graph_node_id == 1 else 1
                kernel_row = (
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
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                    + ",".join("?" for _ in range(28))
                    + ")",
                    kernel_row,
                )
                if uneven_replays and graph_node_id == len(node_names):
                    connection.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ("
                        + ",".join("?" for _ in range(28))
                        + ")",
                        kernel_row,
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
            sync_correlation_id = 3000 + range_index
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    50_000,
                    60_000,
                    0,
                    1,
                    sync_correlation_id,
                    string_ids[device_sync_name],
                    0,
                    None,
                ),
            )
            connection.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES ("
                + ",".join("?" for _ in range(12))
                + ")",
                [
                    (
                        50_010,
                        59_990,
                        0,
                        1,
                        None,
                        0xFFFFFFFF,
                        sync_correlation_id,
                        1,
                        None,
                        4,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                    ),
                    (
                        60_010,
                        60_020,
                        0,
                        1,
                        None,
                        0xFFFFFFFF,
                        sync_correlation_id + 1,
                        1,
                        None,
                        4,
                        0xFFFFFFFF,
                        0xFFFFFFFF,
                    ),
                ],
            )
        if lost_events:
            runtime_count = profiler_start_count + 2
            activity_count = runtime_count + len(node_names) + memcpy_nodes + 1
            collected_count = activity_count + collected_event_delta
            connection.executemany(
                "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        2,
                        2,
                        3,
                        2,
                        "Not all CUDA events might have been collected.",
                        1,
                    ),
                    (
                        3,
                        2,
                        3,
                        1,
                        f"Number of CUDA events collected: \t{collected_count}.",
                        1,
                    ),
                    (
                        4,
                        1,
                        1,
                        1,
                        "Number of CUPTI events produced: "
                        f"\t{collected_count + 8}, CUPTI buffers: 50.",
                        1,
                    ),
                ],
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
    batch: int = TRACE_CONCURRENCY,
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
        f"real_batch={batch} padded_batch={batch} prior_replays=128",
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
        self.assertEqual(
            plan["planned_commands"]["trace_gdn_ba"][-4:],
            ["--trace-concurrency", "2", "--gdn-ba-mode", "both"],
        )
        self.assertEqual(
            plan["planned_commands"]["trace_gdn_packed"][-4:],
            ["--trace-concurrency", "2", "--gdn-packed-mode", "both"],
        )
        self.assertIn(
            "trace/27/{gdn-ba-summary,gdn-ba-manifest,status-gdn-ba}.json",
            plan["required_artifacts"],
        )
        self.assertIn(
            "trace/27/{gdn-packed-summary,gdn-packed-manifest,status-gdn-packed}.json",
            plan["required_artifacts"],
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


    def test_execute_grid_is_unheld_after_w1d3_closure_authorization(self) -> None:
        """The H1d-era unconditional --execute hold must be lifted.

        The hold's stated precondition (complete H1d/G4; separate production
        and trace builds) was met on 2026-07-13, and the W1D3 closure
        (b80663a) explicitly AUTHORIZED the fresh binding/exact-grid rerun.
        The driver must not carry the unconditional hold, and --execute must
        remain a recognized mode.
        """
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("is held while H1d", script)
        self.assertIn("--dry-run|--prepare-corpus|--trace-only|--execute", script)

    def test_vllm_profiler_uses_clean_oracle_venv_path(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        block = script.split("local -a vllm_profile_cmd=(", 1)[1].split(
            "\n  )", 1
        )[0]
        self.assertIn('"${benchmark_clean_prefix[@]}"', block)
        self.assertIn(
            '"PATH=$(dirname "${client}"):${benchmark_system_path}"',
            block,
        )
        self.assertNotIn("${PATH}", block)

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

    def test_profile_control_accepts_nsys_progress_before_stopped_marker(self) -> None:
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
            marker = "[VT_CUDA_PROFILE] stopped captured_replays=4 graph=0x1234"
            profile_log.write_text(
                profile_log.read_text(encoding="utf-8").replace(
                    marker,
                    "\r[4/4] [========================100%] ours-r3.4.nsys-rep"
                    + marker,
                ),
                encoding="utf-8",
            )

            result = record_profile_control(
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

            self.assertEqual(result["captured_replays"], 4)
            self.assertEqual(result["graph"], "0x1234")

    def test_profile_control_rejects_duplicate_stopped_marker_on_one_line(self) -> None:
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
            marker = "[VT_CUDA_PROFILE] stopped captured_replays=4 graph=0x1234"
            profile_log.write_text(
                profile_log.read_text(encoding="utf-8").replace(
                    marker, marker + marker
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                HarnessError, "complete CUDA profile sequence"
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

    def test_profile_control_rejects_suffix_after_stopped_marker(self) -> None:
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
            marker = "[VT_CUDA_PROFILE] stopped captured_replays=4 graph=0x1234"
            profile_log.write_text(
                profile_log.read_text(encoding="utf-8").replace(
                    marker, marker + " trailing-garbage"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                HarnessError, "complete CUDA profile sequence"
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

    def test_profile_control_accepts_only_the_requested_low_batch(self) -> None:
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
                batch=2,
            )
            result = record_profile_control(
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
                expected_batch=2,
            )
            self.assertEqual(result["real_batch"], 2)
            self.assertEqual(result["padded_batch"], 2)

            with self.assertRaisesRegex(HarnessError, "four-replay contract"):
                record_profile_control(
                    root / "wrong-control.json",
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
                    expected_batch=16,
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
        self.assertIn(
            'summarize-nsys-kernels \\\n'
            '        --sqlite "${ours_sqlite}" \\\n'
            '        --model-key "${model}"',
            script,
        )
        self.assertIn("local capture_closed=0", script)
        self.assertIn("for _ in $(seq 1 60); do", script)
        self.assertIn("((capture_closed == 1))", script)
        self.assertIn('kill -0 "${server_pid}" 2>/dev/null || break', script)
        self.assertIn(
            "grep -q '\\[VT_CUDA_PROFILE\\] stopped captured_replays=4 "
            "graph=0x[0-9a-f][0-9a-f]*$'",
            script,
        )
        self.assertNotIn(
            "grep -q '^\\[VT_CUDA_PROFILE\\] stopped captured_replays=4",
            script,
        )
        probe_index = script.index(
            '--artifact-tag "${artifact_prefix}trace${trace_rep}-probe"'
        )
        wait_index = script.index("local capture_closed=0", probe_index)
        shutdown_index = script.index("printf 'Q'", wait_index)
        self.assertLess(probe_index, wait_index)
        self.assertLess(wait_index, shutdown_index)
        self.assertIn("--trace-only", script)

    def test_trace_driver_records_gdn_ba_arms_under_one_lock(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("[--gdn-ba-mode both]", script)
        self.assertIn('trace_env+=("VT_GDN_MERGED_BA=${gdn_ba_env_value}")', script)
        self.assertIn('gdn_ba_validation_args=(--gdn-ba-mode "${trace_arm}")', script)
        lock_index = script.index("flock 9")
        merged_index = script.index("run_paired_traces merged", lock_index)
        split_index = script.index("run_paired_traces split", merged_index)
        self.assertLess(lock_index, merged_index)
        self.assertLess(merged_index, split_index)
        self.assertEqual(script.count("flock 9"), 1)
        self.assertIn(
            "model ${model} GDN BA merged/split node-level paired traces complete",
            script,
        )

    def test_trace_driver_records_gdn_packed_arms_under_one_lock(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("[--gdn-packed-mode both]", script)
        self.assertIn(
            'trace_env+=("VT_GDN_PACKED_DECODE=${gdn_packed_env_value}")',
            script,
        )
        self.assertIn(
            'gdn_validation_args=(--gdn-packed-mode "${trace_arm}")',
            script,
        )
        lock_index = script.index("flock 9")
        packed_index = script.index("run_paired_traces packed", lock_index)
        rollback_index = script.index("run_paired_traces rollback", packed_index)
        self.assertLess(lock_index, packed_index)
        self.assertLess(packed_index, rollback_index)
        self.assertEqual(script.count("flock 9"), 1)
        self.assertIn(
            "model ${model} GDN packed/rollback node-level paired traces complete",
            script,
        )

    def test_vllm_arm_server_pins_mamba_ssm_cache_dtype_float32(self) -> None:
        # Equivalence-audit recommendation 3
        # (.agents/specs/benchmark-equivalence-audit-2026-07-15.md): pin the
        # resolved GDN SSM cache dtype in the recorded vLLM server evidence.
        # ``--mamba-ssm-cache-dtype float32`` is a valid ``vllm serve`` flag
        # (vllm/engine/arg_utils.py:687,1182,1882; type MambaDType accepts
        # 'float32' at vllm/config/cache.py:37,130) and, differing from the
        # 'auto' default, appears in the ``non-default args:`` startup log line
        # (vllm/entrypoints/openai/api_server.py:553 ->
        # vllm/entrypoints/serve/utils/api_utils.py:209,271-273). It is a
        # record-visibility no-op matching the value the Qwen3.5 config hook
        # resolves anyway, so it lands only on the vLLM arm.
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        start_server = script.split("start_server() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        ours_block = start_server.split(
            "if [[ ${engine} == ours ]]; then", 1
        )[1].split("\n  else", 1)[0]
        vllm_block = start_server.split(
            '"${client}" serve "${snapshot}"', 1
        )[1].split("\n    )", 1)[0]
        self.assertIn("--mamba-ssm-cache-dtype float32", vllm_block)
        self.assertNotIn("--mamba-ssm-cache-dtype", ours_block)

    def test_trace_driver_sets_h1d_plan_environment_before_manifest(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        clean_start = script.index("benchmark_system_path=")
        plan_start = script.index("h1d_plan_env=(", clean_start)
        clean_end = plan_start
        plan_end = script.index("\n)", plan_start) + 2
        required = TRACE_REQUIRED_ENV | {
            "VT_FP4_AUTOTUNE_CACHE_PATH": "/evidence/native-plan-must-not-exist.json",
            "VT_FP4_FLASHINFER_CACHE_PATH": (
                f"{repo}/tests/fixtures/nvfp4_flashinfer_v025_gb10/"
                "autotune_configs.json"
            ),
        }
        polluted = {
            **os.environ,
            "PYTHONPATH": "/polluted",
            "VT_FP4_EXACT_BUCKETS": "0",
            "VT_FP4_FORCE_TACTIC": "31",
            "VT_GDN_PACKED_DECODE": "0",
            "VLLM_CPP_CUDAGRAPH": "0",
        }
        definitions = (
            "set -euo pipefail\n"
            "evidence=/evidence\n"
            f"repo_root={shlex.quote(str(repo))}\n"
            f"{script[clean_start:clean_end]}\n"
            f"{script[plan_start:plan_end]}\n"
        )
        probe = subprocess.run(
            [
                "bash",
                "-c",
                definitions
                + '"${benchmark_clean_env[@]}" "${h1d_plan_env[@]}" env',
            ],
            env=polluted,
            check=True,
            capture_output=True,
            text=True,
        )
        actual = dict(
            line.split("=", 1)
            for line in probe.stdout.splitlines()
            if "=" in line
        )
        for name, value in required.items():
            self.assertEqual(actual.get(name), value)
        self.assertEqual(actual.get("PYTHONPATH"), str(repo))
        self.assertNotEqual(actual.get("PYTHONPATH"), polluted["PYTHONPATH"])
        self.assertEqual(actual.get("PATH"), TRACE_SYSTEM_PATH)
        for name, value in TRACE_CLEAN_FIXED_ENV.items():
            self.assertEqual(actual.get(name), value)
        for forbidden in (
            "VT_FP4_EXACT_BUCKETS",
            "VT_FP4_FORCE_TACTIC",
            "VT_GDN_PACKED_DECODE",
            "VLLM_CPP_CUDAGRAPH",
        ):
            self.assertNotIn(forbidden, actual)
        subprocess.run(
            [
                "bash",
                "-c",
                definitions
                + '"${benchmark_clean_env[@]}" "${h1d_plan_env[@]}" '
                "python3 -c 'import tools.bench.online_gate'",
            ],
            env=polluted,
            check=True,
            capture_output=True,
            text=True,
        )
        environment_index = min(clean_start, plan_start)
        manifest_index = script.index('record-execution \\', environment_index)
        self.assertLess(environment_index, manifest_index)
        self.assertIn(
            '[[ ! -e ${native_plan_target} ]]',
            script[environment_index:manifest_index],
        )
        self.assertIn(
            '"${benchmark_clean_env[@]}"',
            script[manifest_index - 160 : manifest_index],
        )
        self.assertIn(
            'local -a trace_env=("${h1d_plan_env[@]}")',
            script,
        )
        self.assertIn(
            '"${benchmark_clean_env[@]}"\n      "${trace_env[@]}"\n      nsys profile',
            script,
        )
        self.assertIn(
            'local -a vllm_profile_cmd=(\n    "${benchmark_clean_prefix[@]}"',
            script,
        )
        self.assertIn(
            'if ! "${benchmark_clean_env[@]}" "${h1d_plan_env[@]}" \\\n'
            '  ctest --test-dir',
            script,
        )

    def test_trace_driver_keeps_c2_staging_non_binding(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        script = (repo / "scripts" / "dgx-online-serving.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--trace-concurrency 2|16", script)
        self.assertIn("server_cmd+=(--cuda-profile-graph-batch 2)", script)
        self.assertEqual(
            script.count('--expected-batch "${trace_concurrency}"'),
            3,
        )
        self.assertIn(
            "c2 raw paired trace capture complete; status remains PENDING",
            script,
        )
        pending = script.index(
            "c2 raw paired trace capture complete; status remains PENDING"
        )
        accepted_status = script.index("record-trace-status", pending)
        self.assertLess(pending, accepted_status)

    def test_nsys_range_validation_rejects_loss_replays_and_start_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            clean = root / "clean.sqlite"
            write_nsys_sqlite(clean)
            result = validate_nsys_trace(clean)
            self.assertTrue(result["lossless"])
            self.assertEqual(result["primary_graph_node_count"], 2)
            self.assertEqual(result["primary_graph_replay_count"], 1)
            self.assertEqual(result["profiler_range_start_count"], 1)
            self.assertEqual(result["range_index"], 1)

            lost = root / "lost.sqlite"
            write_nsys_sqlite(lost, lost_events=True)
            with self.assertRaisesRegex(HarnessError, "not lossless"):
                validate_nsys_trace(lost)

            reconciled = root / "reconciled.sqlite"
            write_nsys_sqlite(reconciled, lost_events=True, model_contract=True)
            result = validate_nsys_trace(reconciled, model_key="27")
            self.assertTrue(result["lossless"])
            self.assertEqual(result["lossless_basis"], "exact-event-reconciliation")
            self.assertTrue(result["capture_boundary_diagnostic"]["acknowledged"])
            self.assertEqual(
                result["capture_boundary_diagnostic"]["collected_cuda_events"],
                1_118,
            )
            summary = summarize_nsys_kernels(
                reconciled,
                model_key="27",
                range_index=1,
            )
            self.assertEqual(summary["kernel_count"], 1_107)

            low_batch = root / "low-batch.sqlite"
            write_nsys_sqlite(
                low_batch,
                lost_events=True,
                model_contract=True,
                model_batch=2,
            )
            result = validate_nsys_trace(
                low_batch,
                model_key="27",
                expected_batch=2,
            )
            self.assertNotIn("gdn_ba_mode", result)
            self.assertEqual(result["graph_child_rows"]["kernel"], 1_011)
            self.assertEqual(result["primary_graph_node_count"], 1_011)
            self.assertEqual(
                result["primary_graph_family_node_counts"]["fp4_gemm"], 208
            )
            summary = summarize_nsys_kernels(
                low_batch,
                model_key="27",
                expected_batch=2,
            )
            self.assertEqual(summary["kernel_count"], 1_011)
            with self.assertRaisesRegex(HarnessError, "model contract"):
                validate_nsys_trace(low_batch, model_key="27")
            with self.assertRaisesRegex(HarnessError, "no exact Nsight"):
                validate_nsys_trace(
                    low_batch,
                    model_key="27",
                    expected_batch=4,
                )

            self.assertEqual(TRACE_GDN_BA_MODES, ("merged", "split"))
            self.assertEqual(TRACE_GDN_BA_ENV, {"merged": "1", "split": "0"})
            self.assertEqual(
                TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_BA_MODE[
                    ("27", 2, "merged")
                ]["node_count"],
                963,
            )
            for mode, expected_nodes, expected_bf16 in (
                ("merged", 963, 145),
                ("split", 1_011, 193),
            ):
                mode_trace = root / f"gdn-ba-{mode}.sqlite"
                write_nsys_sqlite(
                    mode_trace,
                    lost_events=True,
                    model_contract=True,
                    model_batch=2,
                    gdn_ba_mode=mode,
                )
                mode_result = validate_nsys_trace(
                    mode_trace,
                    model_key="27",
                    expected_batch=2,
                    gdn_ba_mode=mode,
                )
                self.assertEqual(mode_result["gdn_ba_mode"], mode)
                self.assertEqual(mode_result["primary_graph_node_count"], expected_nodes)
                self.assertEqual(
                    mode_result["primary_graph_family_node_counts"]["bf16_gemm"],
                    expected_bf16,
                )
                mode_summary = summarize_nsys_kernels(
                    mode_trace,
                    model_key="27",
                    expected_batch=2,
                    gdn_ba_mode=mode,
                )
                self.assertEqual(mode_summary["gdn_ba_mode"], mode)
                self.assertEqual(mode_summary["kernel_count"], expected_nodes)

            with self.assertRaisesRegex(HarnessError, "model contract"):
                validate_nsys_trace(
                    root / "gdn-ba-split.sqlite",
                    model_key="27",
                    expected_batch=2,
                    gdn_ba_mode="merged",
                )
            with self.assertRaisesRegex(HarnessError, "requires an exact model"):
                validate_nsys_trace(clean, gdn_ba_mode="merged")
            with self.assertRaisesRegex(HarnessError, "unknown GDN BA"):
                trace_primary_graph_contract("27", 2, gdn_ba_mode="unknown")

            self.assertEqual(TRACE_GDN_PACKED_MODES, ("packed", "rollback"))
            self.assertEqual(TRACE_GDN_PACKED_ENV, {"packed": "1", "rollback": "0"})
            self.assertEqual(
                TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_PACKED_MODE[
                    ("27", 2, "packed")
                ]["node_count"],
                915,
            )
            invariant_hashes = {}
            coupled_ba_hashes = {}
            for mode, expected_nodes, packed_count, decomposed_count in (
                ("packed", 915, 48, 0),
                ("rollback", 963, 0, 48),
            ):
                mode_trace = root / f"gdn-{mode}.sqlite"
                write_nsys_sqlite(
                    mode_trace,
                    lost_events=True,
                    model_contract=True,
                    model_batch=2,
                    gdn_packed_mode=mode,
                )
                mode_result = validate_nsys_trace(
                    mode_trace,
                    model_key="27",
                    expected_batch=2,
                    gdn_packed_mode=mode,
                )
                self.assertEqual(mode_result["gdn_packed_mode"], mode)
                invariant_hashes[mode] = mode_result[
                    "gdn_packed_invariant_node_multiset_sha256"
                ]
                coupled_ba_hashes[mode] = mode_result[
                    "gdn_packed_coupled_ba_node_multiset_sha256"
                ]
                self.assertEqual(mode_result["primary_graph_node_count"], expected_nodes)
                self.assertEqual(
                    mode_result["primary_graph_family_node_counts"][
                        "gdn_packed_recurrence"
                    ],
                    packed_count,
                )
                self.assertEqual(
                    mode_result["primary_graph_family_node_counts"][
                        "gdn_decomposed_recurrence"
                    ],
                    decomposed_count,
                )
                mode_summary = summarize_nsys_kernels(
                    mode_trace,
                    model_key="27",
                    expected_batch=2,
                    gdn_packed_mode=mode,
                )
                self.assertEqual(mode_summary["gdn_packed_mode"], mode)
                self.assertEqual(mode_summary["kernel_count"], expected_nodes)
            self.assertEqual(invariant_hashes["packed"], invariant_hashes["rollback"])
            self.assertNotEqual(coupled_ba_hashes["packed"], coupled_ba_hashes["rollback"])

            unrelated_drift = root / "gdn-packed-unrelated-drift.sqlite"
            write_nsys_sqlite(
                unrelated_drift,
                lost_events=True,
                model_contract=True,
                model_batch=2,
                gdn_packed_mode="packed",
                unrelated_drift=True,
            )
            drift_result = validate_nsys_trace(
                unrelated_drift,
                model_key="27",
                expected_batch=2,
                gdn_packed_mode="packed",
            )
            self.assertNotEqual(
                drift_result["gdn_packed_invariant_node_multiset_sha256"],
                invariant_hashes["packed"],
            )

            unrelated_geometry_drift = root / "gdn-packed-geometry-drift.sqlite"
            write_nsys_sqlite(
                unrelated_geometry_drift,
                lost_events=True,
                model_contract=True,
                model_batch=2,
                gdn_packed_mode="packed",
                signature_drift=True,
            )
            geometry_result = validate_nsys_trace(
                unrelated_geometry_drift,
                model_key="27",
                expected_batch=2,
                gdn_packed_mode="packed",
            )
            self.assertNotEqual(
                geometry_result["gdn_packed_invariant_node_multiset_sha256"],
                invariant_hashes["packed"],
            )

            coupled_ba_drift = root / "gdn-packed-ba-drift.sqlite"
            write_nsys_sqlite(
                coupled_ba_drift,
                lost_events=True,
                model_contract=True,
                model_batch=2,
                gdn_packed_mode="packed",
                gdn_packed_ba_drift=True,
            )
            with self.assertRaisesRegex(HarnessError, "coupled BA node count"):
                validate_nsys_trace(
                    coupled_ba_drift,
                    model_key="27",
                    expected_batch=2,
                    gdn_packed_mode="packed",
                )

            with self.assertRaisesRegex(HarnessError, "model contract"):
                validate_nsys_trace(
                    root / "gdn-rollback.sqlite",
                    model_key="27",
                    expected_batch=2,
                    gdn_packed_mode="packed",
                )
            with self.assertRaisesRegex(HarnessError, "requires an exact model"):
                validate_nsys_trace(clean, gdn_packed_mode="packed")
            with self.assertRaisesRegex(HarnessError, "unknown GDN packed"):
                trace_primary_graph_contract("27", 2, gdn_packed_mode="unknown")
            with self.assertRaisesRegex(HarnessError, "mutually exclusive"):
                trace_primary_graph_contract(
                    "27", 2, gdn_ba_mode="merged", gdn_packed_mode="packed"
                )

            unreconciled = root / "unreconciled.sqlite"
            write_nsys_sqlite(
                unreconciled,
                lost_events=True,
                model_contract=True,
                collected_event_delta=-1,
            )
            with self.assertRaisesRegex(HarnessError, "does not reconcile"):
                validate_nsys_trace(unreconciled, model_key="27")

            uneven = root / "uneven.sqlite"
            write_nsys_sqlite(uneven, uneven_replays=True)
            with self.assertRaisesRegex(HarnessError, "uneven replay counts"):
                validate_nsys_trace(uneven)

            missing_range = root / "missing-range.sqlite"
            write_nsys_sqlite(missing_range, profiler_start_count=0)
            with self.assertRaisesRegex(HarnessError, "cuProfilerStart rows"):
                validate_nsys_trace(missing_range)

            later_range = root / "later-range.sqlite"
            write_nsys_sqlite(later_range, range_index=2)
            result = validate_nsys_trace(later_range, range_index=2)
            self.assertEqual(result["profiler_range_start_count"], 0)
            self.assertEqual(result["range_index"], 2)

            unexpected_start = root / "unexpected-start.sqlite"
            write_nsys_sqlite(
                unexpected_start, range_index=2, profiler_start_count=1
            )
            with self.assertRaisesRegex(HarnessError, "cuProfilerStart rows"):
                validate_nsys_trace(unexpected_start, range_index=2)

            orphan = root / "orphan.sqlite"
            write_nsys_sqlite(orphan, orphan_child=True)
            with self.assertRaisesRegex(HarnessError, "direct cudaGraphLaunch"):
                validate_nsys_trace(orphan)

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
            home = root / "home"
            home.mkdir()
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
            session_uuids = []
            system_path = TRACE_SYSTEM_PATH
            clean_command_environment = {
                "HOME": str(home),
                "PATH": system_path,
                "PYTHONPATH": str(source),
                **TRACE_CLEAN_FIXED_ENV,
            }
            vllm_corpus = root / "vllm-corpus.jsonl"
            vllm_corpus.write_text('{"prompt":"fixture"}\n', encoding="utf-8")
            for session_index in range(TRACE_REPETITIONS):
                command = root / f"ours-{session_index}-command.txt"
                profile_log = root / f"ours-{session_index}-profile.log"
                native = root / f"native-{session_index}-must-not-exist.json"
                server_pid = 6000 + session_index
                nsys_pid = 5000 + session_index
                launcher_pid = 5500 + session_index
                write_profile_log(
                    profile_log,
                    fixture=fixture,
                    native_target=native,
                    server_pid=server_pid,
                )
                command_environment = {
                    **clean_command_environment,
                    **TRACE_REQUIRED_ENV,
                    "VT_FP4_AUTOTUNE_CACHE_PATH": str(native),
                    "VT_FP4_FLASHINFER_CACHE_PATH": str(fixture),
                }
                prefix = root / f"ours-{session_index}"
                shutdown_fifo = pathlib.Path(f"{prefix}-shutdown.fifo")
                profile_command = [
                    "/usr/bin/env",
                    "-i",
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
                control = root / f"ours-{session_index}-control.json"
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
                ours_commands.append(command)
                ours_profile_logs.append(profile_log)
                ours_profile_controls.append(control)
                session_uuid = str(uuid.uuid4())
                session_uuids.append(session_uuid)
                for range_index in range(1, TRACE_CAPTURE_GRAPH_REPLAYS + 1):
                    range_prefix = pathlib.Path(f"{prefix}.{range_index}")
                    report = pathlib.Path(f"{range_prefix}.nsys-rep")
                    report.write_text("trace\n", encoding="utf-8")
                    sqlite_path = pathlib.Path(f"{range_prefix}.sqlite")
                    validation = pathlib.Path(
                        f"{range_prefix}-validation.json"
                    )
                    write_nsys_sqlite(
                        sqlite_path,
                        capture_uuid=session_uuid,
                        model_contract=True,
                        range_index=range_index,
                        report_path=report,
                    )
                    validation.write_text(
                        json.dumps(
                            validate_nsys_trace(
                                sqlite_path,
                                model_key="27",
                                range_index=range_index,
                            )
                        ),
                        encoding="utf-8",
                    )
                    summary = pathlib.Path(f"{range_prefix}-summary.txt")
                    summary.write_text(
                        json.dumps(
                            summarize_nsys_kernels(
                                sqlite_path,
                                model_key="27",
                                range_index=range_index,
                            )
                        ),
                        encoding="utf-8",
                    )
                    ours_nsys_reports.append(report)
                    ours_nsys_sqlites.append(sqlite_path)
                    ours_nsys_validations.append(validation)
                    ours_kernel_summaries.append(summary)
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
                "/usr/bin/env",
                "-i",
                *[
                    f"{name}={value}"
                    for name, value in {
                        **clean_command_environment,
                        "PATH": f"{oracle_python.parent}:{system_path}",
                    }.items()
                ],
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

            ours_nsys_reports[1], ours_nsys_reports[4] = (
                ours_nsys_reports[4],
                ours_nsys_reports[1],
            )
            with self.assertRaisesRegex(HarnessError, "indexed reports"):
                record_trace()
            ours_nsys_reports[1], ours_nsys_reports[4] = (
                ours_nsys_reports[4],
                ours_nsys_reports[1],
            )

            connection = sqlite3.connect(ours_nsys_sqlites[1])
            try:
                connection.execute(
                    "UPDATE META_DATA_CAPTURE SET value = ? "
                    "WHERE name = 'PROFILING_SESSION_UUID'",
                    (session_uuids[1],),
                )
                connection.commit()
            finally:
                connection.close()
            ours_nsys_validations[1].write_text(
                json.dumps(
                    validate_nsys_trace(
                        ours_nsys_sqlites[1], model_key="27", range_index=2
                    )
                ),
                encoding="utf-8",
            )
            ours_kernel_summaries[1].write_text(
                json.dumps(
                    summarize_nsys_kernels(
                        ours_nsys_sqlites[1],
                        model_key="27",
                        range_index=2,
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HarnessError, "one UUID within each session"):
                record_trace()
            connection = sqlite3.connect(ours_nsys_sqlites[1])
            try:
                connection.execute(
                    "UPDATE META_DATA_CAPTURE SET value = ? "
                    "WHERE name = 'PROFILING_SESSION_UUID'",
                    (session_uuids[0],),
                )
                connection.commit()
            finally:
                connection.close()
            ours_nsys_validations[1].write_text(
                json.dumps(
                    validate_nsys_trace(
                        ours_nsys_sqlites[1], model_key="27", range_index=2
                    )
                ),
                encoding="utf-8",
            )
            ours_kernel_summaries[1].write_text(
                json.dumps(
                    summarize_nsys_kernels(
                        ours_nsys_sqlites[1],
                        model_key="27",
                        range_index=2,
                    )
                ),
                encoding="utf-8",
            )

            original_ours_command = ours_commands[0].read_text(encoding="utf-8")
            ours_tokens = shlex.split(original_ours_command)
            del ours_tokens[1]
            ours_commands[0].write_text(
                shlex.join(ours_tokens) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(HarnessError, "/usr/bin/env -i"):
                record_trace()
            ours_tokens = shlex.split(original_ours_command)
            ours_tokens.insert(ours_tokens.index("nsys"), "VT_FP4_FORCE_TACTIC=31")
            ours_commands[0].write_text(
                shlex.join(ours_tokens) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(HarnessError, "env inventory"):
                record_trace()
            ours_commands[0].write_text(original_ours_command, encoding="utf-8")

            original_vllm_command = vllm_command.read_text(encoding="utf-8")
            vllm_tokens = shlex.split(original_vllm_command)
            del vllm_tokens[1]
            vllm_command.write_text(
                shlex.join(vllm_tokens) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(HarnessError, "/usr/bin/env -i"):
                record_trace()
            vllm_command.write_text(original_vllm_command, encoding="utf-8")

            nonrepeatable = json.loads(
                ours_client_results[1].read_text(encoding="utf-8")
            )
            nonrepeatable["generated_texts"][0] = "different-valid-branch"
            ours_client_results[1].write_text(
                json.dumps(nonrepeatable), encoding="utf-8"
            )
            trace = record_trace()
            self.assertEqual(trace["ours_profiler"], "nsys")
            self.assertEqual(trace["vllm_profiler"], "torch-profiler")
            self.assertEqual(
                trace["trace_contract"]["cuda_graph_trace"],
                NSYS_CUDA_GRAPH_TRACE,
            )
            self.assertEqual(trace["trace_contract"]["nsys_captures"], 3)
            self.assertEqual(
                trace["trace_contract"]["total_range_reports"],
                TRACE_RANGE_REPORTS,
            )
            self.assertEqual(len(trace["nsys_validations"]), TRACE_RANGE_REPORTS)
            self.assertEqual(trace["nsys_capture_session_uuids"], session_uuids)
            self.assertEqual(
                trace["nsys_range_report_session_uuids"],
                [
                    session_uuid
                    for session_uuid in session_uuids
                    for _ in range(TRACE_CAPTURE_GRAPH_REPLAYS)
                ],
            )
            self.assertEqual(
                trace["nsys_validations"][0]["primary_graph_family_node_counts"][
                    "normal_fp4_producer"
                ],
                144,
            )
            self.assertFalse(trace["output_repeatability"]["ours"]["all_equal"])
            self.assertFalse(trace["output_repeatability"]["vllm"]["all_equal"])


if __name__ == "__main__":
    unittest.main()
