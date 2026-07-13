"""Every-axis aggregation tests ported from vLLM bench-serve metrics.

Sources: ``vllm/benchmarks/serve.py:563-748,1188-1284`` and
``tests/benchmarks/test_serve_cli.py:58-132`` at vLLM e24d1b24.  Project-only
extensions exercise the stricter paired-output, memory-return, and every-axis
acceptance rules from ``.agents/benchmark-protocol.md``.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    FLASHINFER_VERSION,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MODEL_REVISIONS,
    NSYS_CUDA_FLUSH_INTERVAL_MS,
    NSYS_CAPTURE_RANGE,
    NSYS_CUDA_GRAPH_TRACE,
    NSYS_PRODUCT_VERSION,
    OUTPUT_LEN,
    PANDAS_VERSION,
    TRACE_CONCURRENCY,
    TRACE_CAPTURE_GRAPH_REPLAYS,
    TRACE_PRIMARY_GRAPH_CONTRACTS,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
    TRACE_STATUS_SCHEMA_VERSION,
    VLLM_ORACLE_VERSION,
    VLLM_GENERATION_WINDOW_CONTRACTS,
    validate_nsys_trace,
    record_profile_control,
    summarize_nsys_kernels,
    _fingerprint_tree,
    _parse_fp4_plan_log,
    _summarize_torch_trace,
)
from tools.bench.online_gate_summary import summarize_evidence
from tools.bench.serve_low_common import HarnessError, VLLM_COMMIT, sha256_file
from tests.tools.test_online_gate_client import (
    write_nsys_sqlite,
    write_profile_log,
    write_vllm_decode_trace,
)


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


def _write_model_nsys_sqlite(
    path: pathlib.Path, *, report_path: pathlib.Path
) -> None:
    write_nsys_sqlite(path, model_contract=True, report_path=report_path)


def _write_fixture(root: pathlib.Path) -> None:
    sha = "c" * 40
    cutlass = root / "fixture-artifacts" / "cutlass"
    (cutlass / "include/cutlass").mkdir(parents=True)
    (cutlass / "include/cutlass/cutlass.h").write_text(
        "cutlass\n", encoding="utf-8"
    )
    native_target = root / "fixture-artifacts" / "native-must-not-exist.json"
    source_root = root / "fixture-source"
    source_root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "client_contract_source_commit": VLLM_COMMIT,
                "gpu_lock_acquisitions_planned": 2,
                "vllm_oracle_bench_dependencies": {
                    "flashinfer": FLASHINFER_VERSION,
                    "pandas": PANDAS_VERSION,
                },
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
        "compile_commands",
        "configure_log",
        "model_config",
        "oracle_manifest",
        "oracle:bench_datasets",
        "oracle:bench_serve",
        "oracle:cli_bench_serve",
        "oracle:client",
        "oracle:distribution_metadata",
        "oracle:distribution_record",
        "oracle:flashinfer_distribution_metadata",
        "oracle:flashinfer_distribution_record",
        "oracle:flashinfer_package_init",
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
    pathlib.Path(execution_artifacts["cmake_cache"]["path"]).write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}\n", encoding="utf-8"
    )
    execution_artifacts["cmake_cache"]["sha256"] = sha256_file(
        pathlib.Path(execution_artifacts["cmake_cache"]["path"])
    )
    production_execution = {
        "artifacts": execution_artifacts,
        "bench_dependencies": {
            "flashinfer": FLASHINFER_VERSION,
            "pandas": PANDAS_VERSION,
        },
        "build_contract": {
            "compile_command_sha256": "a" * 64,
            "cutlass_source_tree": _fingerprint_tree(cutlass),
            "native_plan_target": str(native_target),
            "native_plan_target_absent": True,
            "profile_control": False,
            "sm_architecture": "121a",
            "target_compile_definitions": ["VT_CUTLASS_NVFP4=1"],
        },
        "cache_drop_roots": _fixture_cache_roots(root),
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS["27"],
        "max_num_seqs": MAX_NUM_SEQS,
        "model_key": "27",
        "num_blocks": 4736,
        "port": 8001,
        "vllm_oracle_version": VLLM_ORACLE_VERSION,
        "vllm_cpp_sha": sha,
        "vllm_source_sha": VLLM_COMMIT,
        "snapshot_files": [],
        "weight_files": [weight_name],
    }
    (execution_root / "27.json").write_text(
        json.dumps(production_execution),
        encoding="utf-8",
    )
    trace_execution = json.loads(json.dumps(production_execution))
    trace_execution["build_contract"]["profile_control"] = True
    trace_execution["build_contract"]["target_compile_definitions"] = [
        "VT_BENCH_PROFILE_CONTROL=1",
        "VT_CUTLASS_NVFP4=1",
    ]
    trace_execution_path = execution_root / "27-trace.json"
    trace_execution_path.write_text(
        json.dumps(
            trace_execution
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
    fixture = (
        pathlib.Path(__file__).resolve().parents[2]
        / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
    )
    validations = []
    summaries = []
    plans = []
    controls = []
    for index in range(1, TRACE_REPETITIONS + 1):
        suffix = "" if index == 1 else f"_{index}"
        report_name = f"ours_nsys_report{suffix}"
        report_path = trace_root / f"{report_name}.txt"
        report_path.write_text(f"{report_name}\n", encoding="utf-8")
        sqlite_path = trace_root / f"ours_nsys_sqlite{suffix}.sqlite"
        _write_model_nsys_sqlite(sqlite_path, report_path=report_path)
        validation = validate_nsys_trace(sqlite_path, model_key="27")
        validation_path = trace_root / f"ours_nsys_validation{suffix}.json"
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        summary = summarize_nsys_kernels(sqlite_path)
        summary_path = trace_root / f"ours_kernel_summary{suffix}.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        native = trace_root / f"native-{index}-must-not-exist.json"
        profile_log = trace_root / f"ours_profile_log{suffix}.log"
        write_profile_log(
            profile_log,
            fixture=fixture,
            native_target=native,
            server_pid=6000 + index,
        )
        control_path = trace_root / f"ours_profile_control{suffix}.json"
        control = record_profile_control(
            control_path,
            profile_log=profile_log,
            nsys_pid=5000 + index,
            nsys_exit_status=0,
            server_pid=6000 + index,
            server_pgid=5000 + index,
        )
        for name, path in (
            (f"ours_nsys_sqlite{suffix}", sqlite_path),
            (f"ours_nsys_validation{suffix}", validation_path),
            (f"ours_kernel_summary{suffix}", summary_path),
            (f"ours_profile_log{suffix}", profile_log),
            (f"ours_profile_control{suffix}", control_path),
            (report_name, report_path),
        ):
            trace_files[name] = {"path": str(path), "sha256": sha256_file(path)}
        for stem in ("ours_command",):
            name = f"{stem}{suffix}"
            path = trace_root / f"{name}.txt"
            path.write_text(f"{name}\n", encoding="utf-8")
            trace_files[name] = {"path": str(path), "sha256": sha256_file(path)}
        for stem in (
            "ours_client_result",
            "ours_client_log",
            "ours_probe_result",
            "ours_probe_log",
        ):
            name = f"{stem}_{index}"
            path = trace_root / f"{name}.txt"
            path.write_text(f"{name}\n", encoding="utf-8")
            trace_files[name] = {"path": str(path), "sha256": sha256_file(path)}
        validations.append(validation)
        summaries.append(summary)
        plans.append(_parse_fp4_plan_log(profile_log))
        controls.append(control)
    for name in ("vllm_command", "vllm_profile_log", "vllm_metadata", "vllm_corpus"):
        path = trace_root / f"{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        trace_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    vllm_trace = trace_root / "vllm-trace.json.gz"
    write_vllm_decode_trace(vllm_trace)
    vllm_summary = trace_root / "vllm-kernel-summary.json"
    vllm_summary.write_text(
        json.dumps(_summarize_torch_trace(vllm_trace, model_key="27")),
        encoding="utf-8",
    )
    trace_files["vllm_torch_trace"] = {
        "path": str(vllm_trace),
        "sha256": sha256_file(vllm_trace),
    }
    trace_files["vllm_kernel_summary"] = {
        "path": str(vllm_summary),
        "sha256": sha256_file(vllm_summary),
    }
    trace_files["execution_manifest"] = {
        "path": str(trace_execution_path),
        "sha256": sha256_file(trace_execution_path),
    }
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
                "node_multiset_sha256": validations[0][
                    "canonical_node_multiset_sha256"
                ],
                "nsys_capture_session_uuids": [
                    validation["capture_session_uuid"]
                    for validation in validations
                ],
                "nsys_kernel_summaries": summaries,
                "nsys_product_version": NSYS_PRODUCT_VERSION,
                "nsys_validations": validations,
                "ours_profiler": "nsys",
                "passed": True,
                "plan_validations": plans,
                "profile_controls": controls,
                "schema_version": TRACE_STATUS_SCHEMA_VERSION,
                "trace_contract": {
                    "admission_mode": "closed-loop",
                    "capture_graph_replays": TRACE_CAPTURE_GRAPH_REPLAYS,
                    "capture_range": NSYS_CAPTURE_RANGE,
                    "capture_range_end": "stop",
                    "concurrency": TRACE_CONCURRENCY,
                    "cuda_flush_interval_ms": NSYS_CUDA_FLUSH_INTERVAL_MS,
                    "cuda_graph_trace": NSYS_CUDA_GRAPH_TRACE,
                    "cuda_event_trace": False,
                    "enable_prefix_caching": False,
                    "force_overwrite": True,
                    "flush_on_cudaprofilerstop": True,
                    "input_len": INPUT_LEN,
                    "max_model_len": MAX_MODEL_LEN["27"],
                    "max_num_seqs": MAX_NUM_SEQS,
                    "nsys_captures": TRACE_REPETITIONS,
                    "nsys_kill": "none",
                    "nsys_stats": False,
                    "num_prompts": TRACE_PROMPTS,
                    "output_len": OUTPUT_LEN,
                    "probe_num_prompts": TRACE_CONCURRENCY,
                    "probe_num_warmups": 0,
                    "probe_timing_binding": False,
                    "semantic_num_warmups": TRACE_CONCURRENCY,
                    "repetitions": TRACE_REPETITIONS,
                    "sample": "none",
                    "cpu_context_switch_trace": "none",
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
                "tokenizer_revision": MODEL_REVISIONS["27"],
                "vllm_commit": VLLM_COMMIT,
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
                "--max-num-batched-tokens 2048 --no-enable-prefix-caching "
                "--served-model-name gate\n",
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
    def setUp(self) -> None:
        window_patch = mock.patch.dict(
            VLLM_GENERATION_WINDOW_CONTRACTS,
            {"27": {"all": 1, "clean": 1}},
        )
        window_patch.start()
        self.addCleanup(window_patch.stop)

    def _summarize(self, root: pathlib.Path):
        patches = (
            mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
            mock.patch("tools.bench.online_gate_summary.POINTS", ((1, 2),)),
            mock.patch(
                "tools.bench.online_gate_summary.MODEL_REVISIONS",
                {"27": MODEL_REVISIONS["27"]},
            ),
        )
        with patches[0], patches[1], patches[2]:
            return summarize_evidence(root)

    def _summarize_model(self, root: pathlib.Path):
        with (
            mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
            mock.patch("tools.bench.online_gate_summary.POINTS", ((1, 2),)),
        ):
            return summarize_evidence(root, models=("27",))

    def test_complete_exact_outputs_and_better_axes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(ratios["gate_pass"])
            self.assertTrue(all(item["binding_eligible"] for item in ratios["ratios"]))
            self.assertTrue(all(item["pass"] for item in ratios["ratios"]))

    def test_model_summary_does_not_require_the_other_gate_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            runs, ratios = self._summarize_model(root)
            self.assertEqual(runs["models"], ["27"])
            self.assertEqual(ratios["models"], ["27"])
            self.assertTrue(runs["gate_pass"])
            self.assertFalse(
                any("35" in reason for reason in runs["campaign_reasons"])
            )

    def test_legacy_graph_level_trace_cannot_bind_schema_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "trace" / "27" / "status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            status["trace_contract"].pop("cuda_graph_trace")
            path.write_text(json.dumps(status), encoding="utf-8")
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])

    def test_explicit_whole_graph_trace_cannot_pass_new_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "trace" / "27" / "status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            status["trace_contract"]["cuda_graph_trace"] = "graph"
            path.write_text(json.dumps(status), encoding="utf-8")
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])

    def test_model_selection_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(HarnessError, "empty"):
                summarize_evidence(root, models=())
            with self.assertRaisesRegex(HarnessError, "duplicates"):
                summarize_evidence(root, models=("27", "27"))
            with self.assertRaisesRegex(HarnessError, "unknown"):
                summarize_evidence(root, models=("not-a-model",))

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
            (root / "trace" / "27" / "ours_kernel_summary.json").write_text(
                "tampered\n", encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_three_capture_trace_requires_every_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            status_path = root / "trace" / "27" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            runs, _ = self._summarize(root)
            self.assertTrue(runs["gate_pass"])

            status["artifacts"].pop("ours_nsys_sqlite_2")
            status_path.write_text(json.dumps(status), encoding="utf-8")
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
