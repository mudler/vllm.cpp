#!/usr/bin/env python3
"""Fail-closed orchestration helpers for ``SERVE-GATE-ONLINE``.

The timed client remains the unmodified pinned vLLM ``bench serve`` command.
This wrapper mirrors its CLI and result schema from:

* ``vllm/benchmarks/serve.py:321-353,563-748,1188-1284,2082-2284``
* ``vllm/benchmarks/datasets/datasets.py:2482-2610``
* ``tests/benchmarks/test_serve_cli.py:58-132``

at vLLM commit ``e24d1b24fe96a56ba8b0d653efa076d03eb95d6c``.  It
does not reimplement request timing.  It constructs the exact upstream client
command, validates every detailed artifact, and makes partial request sets
fatal instead of allowing their aggregate metrics to look successful.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.serve_low_common import (
    HarnessError,
    VLLM_COMMIT,
    canonical_json,
    read_jsonl,
    require_number,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)

INPUT_LEN = 1024
OUTPUT_LEN = 128
VLLM_ORACLE_VERSION = "0.24.0"
PANDAS_VERSION = "2.2.3"
TRACE_CONCURRENCY = 16
TRACE_PROMPTS = 48
TRACE_REPETITIONS = 3
REPETITIONS = (1, 2, 3)
POINTS = ((1, 6), (2, 6), (4, 12), (8, 24), (16, 96), (32, 192))
MODEL_REVISIONS = {
    "27": "890bdef7a42feba6d83b6e17a03315c694112f2a",
    "35": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
}
MODEL_REPOSITORIES = {
    "27": "unsloth/Qwen3.6-27B-NVFP4",
    "35": "nvidia/Qwen3.6-35B-A3B-NVFP4",
}
MAX_NUM_SEQS = 32
MAX_NUM_BATCHED_TOKENS = {"27": 2048, "35": 8192}
ENGINES = ("ours", "vllm")
PERCENTILE_METRICS = ("ttft", "tpot", "itl", "e2el")
PERCENTILES = (50, 90, 99)
CACHE_DROP_METHOD = "posix_fadvise-dontneed+mincore"

_SHA_RE = re.compile(r"[0-9a-f]{40}")


def prompts_for(concurrency: int) -> int:
    try:
        return dict(POINTS)[concurrency]
    except KeyError as error:
        raise HarnessError(f"unsupported online-gate concurrency: {concurrency}") from error


@dataclasses.dataclass(frozen=True)
class OnlineRun:
    client: pathlib.Path
    tokenizer: pathlib.Path
    evidence_root: pathlib.Path
    model_key: str
    engine: str
    base_url: str
    concurrency: int
    repetition: int
    artifact_tag: str = ""
    num_prompts_override: int | None = None

    def __post_init__(self) -> None:
        if self.model_key not in MODEL_REVISIONS:
            raise HarnessError(f"unknown model key: {self.model_key}")
        if self.engine not in ENGINES:
            raise HarnessError(f"unknown engine arm: {self.engine}")
        if self.repetition not in REPETITIONS:
            raise HarnessError(f"unsupported repetition: {self.repetition}")
        if self.artifact_tag and re.fullmatch(r"[a-z0-9-]+", self.artifact_tag) is None:
            raise HarnessError(f"invalid artifact tag: {self.artifact_tag}")
        if self.num_prompts_override is not None and self.num_prompts_override <= 0:
            raise HarnessError("num-prompts override must be positive")
        prompts_for(self.concurrency)

    @property
    def filename_stem(self) -> str:
        suffix = f"-{self.artifact_tag}" if self.artifact_tag else ""
        return f"c{self.concurrency}-r{self.repetition}{suffix}"

    @property
    def num_prompts(self) -> int:
        return (
            self.num_prompts_override
            if self.num_prompts_override is not None
            else prompts_for(self.concurrency)
        )

    @property
    def corpus_path(self) -> pathlib.Path:
        return self.evidence_root / "corpus" / self.model_key / "vllm" / (
            f"c{self.concurrency}-r{self.repetition}.jsonl"
        )

    @property
    def result_path(self) -> pathlib.Path:
        return self.evidence_root / "raw" / self.model_key / self.engine / (
            f"{self.filename_stem}.json"
        )

    @property
    def log_path(self) -> pathlib.Path:
        return self.evidence_root / "logs" / self.model_key / self.engine / (
            f"{self.filename_stem}.log"
        )


def _require_full_sha(value: str, field: str) -> str:
    if _SHA_RE.fullmatch(value) is None:
        raise HarnessError(f"{field} must be a full lowercase commit ID")
    return value


def build_client_command(run: OnlineRun) -> list[str]:
    """Build the unmodified pinned ``vllm bench serve`` invocation."""

    if not run.client.is_file():
        raise HarnessError(f"missing pinned vLLM client: {run.client}")
    if not run.tokenizer.is_dir():
        raise HarnessError(f"missing tokenizer snapshot: {run.tokenizer}")
    if not run.corpus_path.is_file():
        raise HarnessError(f"missing frozen corpus partition: {run.corpus_path}")
    return [
        str(run.client),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        run.base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        "gate",
        "--tokenizer",
        str(run.tokenizer),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(run.corpus_path),
        "--custom-output-len",
        str(OUTPUT_LEN),
        "--skip-chat-template",
        "--disable-shuffle",
        "--num-prompts",
        str(run.num_prompts),
        "--max-concurrency",
        str(run.concurrency),
        "--request-rate",
        "inf",
        "--num-warmups",
        str(run.concurrency),
        "--seed",
        "0",
        "--ignore-eos",
        "--temperature",
        "0",
        "--percentile-metrics",
        ",".join(PERCENTILE_METRICS),
        "--metric-percentiles",
        ",".join(str(value) for value in PERCENTILES),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(run.result_path.parent),
        "--result-filename",
        run.result_path.name,
        "--disable-tqdm",
    ]


def _require_list(record: Mapping[str, Any], field: str, length: int) -> list[Any]:
    value = record.get(field)
    if not isinstance(value, list) or len(value) != length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise HarnessError(f"{field} has cardinality {actual}; expected {length}")
    return value


def validate_raw_result(
    record: Mapping[str, Any],
    *,
    concurrency: int,
    expected_requests: int | None = None,
) -> None:
    """Validate the detailed vLLM result schema and exact request contract."""

    expected = prompts_for(concurrency) if expected_requests is None else expected_requests
    if record.get("num_prompts") != expected:
        raise HarnessError(
            f"num_prompts={record.get('num_prompts')!r}; expected {expected}"
        )
    if record.get("completed") != expected or record.get("failed") != 0:
        raise HarnessError(
            "request set is partial: "
            f"completed={record.get('completed')!r}, failed={record.get('failed')!r}, "
            f"expected={expected}"
        )
    if record.get("max_concurrency") != concurrency:
        raise HarnessError(
            f"configured max_concurrency={record.get('max_concurrency')!r}; "
            f"expected {concurrency}"
        )
    if record.get("total_input_tokens") != expected * INPUT_LEN:
        raise HarnessError("total input tokens do not match the frozen corpus")
    if record.get("total_output_tokens") != expected * OUTPUT_LEN:
        raise HarnessError("total output tokens are not exact")

    input_lens = _require_list(record, "input_lens", expected)
    output_lens = _require_list(record, "output_lens", expected)
    ttfts = _require_list(record, "ttfts", expected)
    itls = _require_list(record, "itls", expected)
    start_times = _require_list(record, "start_times", expected)
    generated = _require_list(record, "generated_texts", expected)
    errors = _require_list(record, "errors", expected)
    if any(value != INPUT_LEN for value in input_lens):
        raise HarnessError("raw input lengths do not all equal 1024")
    if any(value != OUTPUT_LEN for value in output_lens):
        raise HarnessError("raw output lengths do not all equal 128")
    if any(value for value in errors):
        raise HarnessError("raw result contains request errors")
    if any(not isinstance(value, str) for value in generated):
        raise HarnessError("raw result did not retain every generated text")
    if any(not isinstance(row, list) or len(row) != OUTPUT_LEN - 1 for row in itls):
        raise HarnessError("each request must retain output_len-1 ITL samples")
    for name, values in (("ttfts", ttfts), ("start_times", start_times)):
        for index, value in enumerate(values):
            if require_number(value, f"{name}[{index}]") < 0.0:
                raise HarnessError(f"{name} contains a negative value")
    for row_index, row in enumerate(itls):
        for value_index, value in enumerate(row):
            if require_number(value, f"itls[{row_index}][{value_index}]") < 0.0:
                raise HarnessError("itls contains a negative value")

    configured_peak = record.get("max_concurrent_requests")
    if isinstance(configured_peak, bool) or not isinstance(configured_peak, int):
        raise HarnessError("max_concurrent_requests is not an integer")
    if configured_peak != min(concurrency, expected):
        raise HarnessError(
            f"achieved peak concurrency {configured_peak}; expected "
            f"{min(concurrency, expected)}"
        )

    numeric_fields = (
        "duration",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p90_itl_ms",
        "p99_itl_ms",
        "mean_e2el_ms",
        "median_e2el_ms",
        "p90_e2el_ms",
        "p99_e2el_ms",
    )
    for field in numeric_fields:
        if require_number(record.get(field), field) < 0.0:
            raise HarnessError(f"{field} is negative")
    if require_number(record.get("duration"), "duration") <= 0.0:
        raise HarnessError("duration must be positive")


def run_benchmark(run: OnlineRun) -> dict[str, Any]:
    command = build_client_command(run)
    for path in (run.result_path, run.log_path):
        if path.exists():
            raise HarnessError(f"refusing to overwrite online-gate evidence: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with run.log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write(f"command: {shlex.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise HarnessError(f"pinned vLLM client exited {completed.returncode}")
    try:
        record = json.loads(run.result_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HarnessError("pinned client did not write its result") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"{run.result_path}: invalid JSON: {error}") from error
    if not isinstance(record, dict):
        raise HarnessError(f"{run.result_path}: result is not an object")
    validate_raw_result(
        record,
        concurrency=run.concurrency,
        expected_requests=run.num_prompts,
    )
    return record


def prepare_corpus_views(
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    model_key: str,
) -> dict[str, Any]:
    """Convert the exact-token shared corpus into vLLM CustomDataset rows."""

    if model_key not in MODEL_REVISIONS:
        raise HarnessError(f"unknown model key: {model_key}")
    source_manifest = source_root / "manifest.json"
    if not source_manifest.is_file():
        raise HarnessError(f"missing source corpus manifest: {source_manifest}")
    if output_root.exists() and any(output_root.iterdir()):
        raise HarnessError(f"refusing to mix corpus views in non-empty {output_root}")

    files: list[dict[str, Any]] = []
    prompt_hashes: set[str] = set()
    for repetition in REPETITIONS:
        for concurrency, expected in POINTS:
            source = source_root / f"c{concurrency}-r{repetition}.jsonl"
            rows = list(read_jsonl(source))
            if len(rows) < expected:
                raise HarnessError(
                    f"{source}: found {len(rows)} rows; expected at least {expected}"
                )
            converted: list[dict[str, Any]] = []
            for index, row in enumerate(rows[:expected]):
                conversations = row.get("conversations")
                if not isinstance(conversations, list) or not conversations:
                    raise HarnessError(f"{source}:{index + 1}: no conversation prompt")
                first = conversations[0]
                prompt = first.get("value") if isinstance(first, dict) else None
                token_ids = row.get("prompt_token_ids")
                prompt_hash = row.get("prompt_sha256")
                if not isinstance(prompt, str):
                    raise HarnessError(f"{source}:{index + 1}: prompt is not text")
                if not isinstance(token_ids, list) or len(token_ids) != INPUT_LEN:
                    raise HarnessError(f"{source}:{index + 1}: token IDs are not exact")
                if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
                    raise HarnessError(f"{source}:{index + 1}: prompt hash is invalid")
                if prompt_hash in prompt_hashes:
                    raise HarnessError("online-gate prompt partitions are not disjoint")
                prompt_hashes.add(prompt_hash)
                if row.get("output_len") != OUTPUT_LEN:
                    raise HarnessError(f"{source}:{index + 1}: output length drift")
                converted.append(
                    {
                        "output_tokens": OUTPUT_LEN,
                        "prompt": prompt,
                        "prompt_sha256": prompt_hash,
                        "prompt_token_ids": token_ids,
                        "source_index": row.get("index"),
                    }
                )
            target = output_root / source.name
            write_jsonl_atomic(target, converted)
            files.append(
                {
                    "concurrency": concurrency,
                    "file": target.name,
                    "repetition": repetition,
                    "requests": expected,
                    "sha256": sha256_file(target),
                    "source_sha256": sha256_file(source),
                }
            )
    manifest = {
        "files": files,
        "format": "vllm-custom-jsonl-v1",
        "input_len": INPUT_LEN,
        "model_key": model_key,
        "output_len": OUTPUT_LEN,
        "source_manifest_sha256": sha256_file(source_manifest),
        "tokenizer_revision": MODEL_REVISIONS[model_key],
        "total_prompts": sum(item["requests"] for item in files),
        "vllm_commit": VLLM_COMMIT,
    }
    write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def build_plan(
    *,
    claim_root: pathlib.Path,
    vllm_cpp_sha: str,
    client: pathlib.Path,
) -> dict[str, Any]:
    _require_full_sha(vllm_cpp_sha, "vllm.cpp SHA")
    return {
        "artifact_root": str(claim_root / "evidence" / vllm_cpp_sha),
        "client": str(client),
        "client_contract_source_commit": VLLM_COMMIT,
        "dry_run": True,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_lock_acquisitions_planned": 2,
        "host": {
            "kernel": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version(),
        },
        "interleaving": [
            {"engine": engine, "repetition": repetition}
            for repetition in REPETITIONS
            for engine in ENGINES
        ],
        "models": {
            key: {
                "repository": MODEL_REPOSITORIES[key],
                "revision": MODEL_REVISIONS[key],
            }
            for key in MODEL_REVISIONS
        },
        "points": [
            {"concurrency": concurrency, "num_prompts": prompts}
            for concurrency, prompts in POINTS
        ],
        "planned_commands": {
            "corpus": [
                str(client.parent / "python"),
                "-m",
                "tools.bench.make_serve_low_corpus",
                "--tokenizer-json",
                "<MODEL_SNAPSHOT>/tokenizer.json",
                "--tokenizer-revision",
                "<MODEL_REVISION>",
                "--model-key",
                "<27|35>",
                "--out",
                "<EVIDENCE>/corpus/<27|35>",
                "--target-input-len",
                str(INPUT_LEN),
                "--output-len",
                str(OUTPUT_LEN),
                "--requests-per-partition",
                "192",
                "--warmup-requests",
                "1",
                "--concurrencies",
                ",".join(str(concurrency) for concurrency, _ in POINTS),
                "--repetitions",
                str(len(REPETITIONS)),
            ],
            "execute_model": [
                "scripts/dgx-online-serving.sh",
                "--execute",
                "--model",
                "<27|35>",
                "--snapshot",
                "<MODEL_SNAPSHOT>",
                "--source-corpus",
                "<EVIDENCE>/corpus/<27|35>",
                "--evidence",
                "<EVIDENCE>",
                "--build-dir",
                "<CURRENT_MAIN_BUILD>",
                "--client",
                str(client),
                "--vllm-cpp-sha",
                vllm_cpp_sha,
            ],
        },
        "required_artifacts": [
            "manifest.json",
            "corpus/<model>/vllm/manifest.json",
            "execution/<model>.json",
            "execution/<model>-oracle.json",
            "execution/<model>-build-command.txt",
            "execution/<model>-build.log",
            "raw/<model>/<engine>/c<conc>-r<rep>.json",
            "logs/<model>/<engine>/c<conc>-r<rep>.log",
            "memory/<model>/<engine>/r<rep>.samples.jsonl",
            "memory/<model>/<engine>/r<rep>.summary.json",
            "thermal/<model>/<engine>/r<rep>-{before,after}.txt",
            "cache-drop/<model>/<engine>/r<rep>-{before,after}.json",
            "memory-return/<model>/<engine>/r<rep>.json",
            "trace/<model>/ours.nsys-rep",
            "trace/<model>/ours-cuda_gpu_kern_sum.txt",
            "trace/<model>/vllm-profile/*.pt.trace.json.gz",
            "trace/<model>/vllm-kernels.json",
            "trace/<model>/cache-{before-ours,between-engines,after-vllm}.json",
            "trace/<model>/status.json",
            "summary/{all-runs,ratios}.json",
        ],
        "run_contract": {
            "input_len": INPUT_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": MAX_NUM_SEQS,
            "output_len": OUTPUT_LEN,
            "request_rate": "inf",
            "sampling": {"ignore_eos": True, "seed": 0, "temperature": 0.0},
            "warmups_per_client_invocation": "equals configured concurrency",
        },
        "series_order": "one whole-model flock; rep1 ours/vllm, rep2 ours/vllm, rep3 ours/vllm",
        "vllm_oracle_bench_dependencies": {"pandas": PANDAS_VERSION},
        "vllm_oracle_version": VLLM_ORACLE_VERSION,
        "vllm_cpp_sha": vllm_cpp_sha,
    }


def validate_plan(path: pathlib.Path, *, vllm_cpp_sha: str) -> dict[str, Any]:
    _require_full_sha(vllm_cpp_sha, "vllm.cpp SHA")
    value = _load_json_object(path)
    if value.get("vllm_cpp_sha") != vllm_cpp_sha:
        raise HarnessError("campaign plan vllm.cpp SHA differs from the execution")
    if value.get("client_contract_source_commit") != VLLM_COMMIT:
        raise HarnessError("campaign plan client source differs from the parity pin")
    if value.get("vllm_oracle_version") != VLLM_ORACLE_VERSION:
        raise HarnessError("campaign plan oracle version differs from the pinned pip oracle")
    if value.get("vllm_oracle_bench_dependencies") != {"pandas": PANDAS_VERSION}:
        raise HarnessError("campaign plan benchmark dependencies differ from the pin")
    if value.get("gpu_lock_acquisitions_planned") != 2:
        raise HarnessError("campaign plan must contain one GPU lock per model")
    if value.get("dry_run") is not True:
        raise HarnessError("campaign plan was not produced by the dry-run path")
    artifact_root = value.get("artifact_root")
    if not isinstance(artifact_root, str) or pathlib.Path(artifact_root).resolve() != path.parent.resolve():
        raise HarnessError("campaign plan artifact root differs from its location")
    return value


def record_memory_return(
    output: pathlib.Path,
    *,
    baseline_mem_available_kib: int,
    final_mem_available_kib: int,
    tolerance_kib: int,
    before_cache_drop_report: pathlib.Path,
    after_cache_drop_report: pathlib.Path,
    gpu_idle: bool,
) -> dict[str, Any]:
    if output.exists():
        raise HarnessError(f"refusing to overwrite memory-return evidence: {output}")
    if min(baseline_mem_available_kib, final_mem_available_kib, tolerance_kib) < 0:
        raise HarnessError("memory-return values must be non-negative")
    cache_drops = {
        "before": _validated_cache_drop_artifact(before_cache_drop_report),
        "after": _validated_cache_drop_artifact(after_cache_drop_report),
    }
    within = final_mem_available_kib + tolerance_kib >= baseline_mem_available_kib
    result = {
        "baseline_mem_available_kib": baseline_mem_available_kib,
        "cache_drops": cache_drops,
        "drop_caches_succeeded": True,
        "final_mem_available_kib": final_mem_available_kib,
        "gpu_idle": gpu_idle,
        "mem_available_within_tolerance": within,
        "returned": gpu_idle and within,
        "tolerance_kib": tolerance_kib,
    }
    write_json_atomic(output, result)
    return result


def _validated_cache_drop_artifact(path: pathlib.Path) -> dict[str, Any]:
    report = _load_json_object(path)
    if report.get("method") != CACHE_DROP_METHOD:
        raise HarnessError(f"cache-drop method differs in {path}")
    if report.get("succeeded") is not True:
        raise HarnessError(f"cache-drop report did not succeed: {path}")
    if report.get("resident_after_bytes") != 0:
        raise HarnessError(f"cache-drop report retains resident pages: {path}")
    file_count = report.get("file_count")
    logical_bytes = report.get("logical_bytes")
    inventory_sha = report.get("file_inventory_sha256")
    roots = report.get("roots")
    if not isinstance(file_count, int) or file_count <= 0:
        raise HarnessError(f"cache-drop report has no file inventory: {path}")
    if not isinstance(logical_bytes, int) or logical_bytes <= 0:
        raise HarnessError(f"cache-drop report has no logical byte count: {path}")
    if not isinstance(inventory_sha, str) or re.fullmatch(r"[0-9a-f]{64}", inventory_sha) is None:
        raise HarnessError(f"cache-drop inventory hash is absent: {path}")
    if (
        not isinstance(roots, list)
        or len(roots) != 4
        or any(not isinstance(root, str) or not pathlib.Path(root).is_absolute() for root in roots)
    ):
        raise HarnessError(f"cache-drop root inventory differs: {path}")
    return {
        "file_count": file_count,
        "file_inventory_sha256": inventory_sha,
        "logical_bytes": logical_bytes,
        "method": CACHE_DROP_METHOD,
        "path": str(path),
        "roots": roots,
        "sha256": sha256_file(path),
    }


def record_model_gate(
    output: pathlib.Path,
    *,
    log: pathlib.Path,
    model_key: str,
    test_name: str,
    vllm_cpp_sha: str,
) -> dict[str, Any]:
    if output.exists():
        raise HarnessError(f"refusing to overwrite model-gate evidence: {output}")
    if model_key not in MODEL_REVISIONS:
        raise HarnessError(f"unknown model key: {model_key}")
    _require_full_sha(vllm_cpp_sha, "vllm.cpp SHA")
    if not log.is_file() or log.stat().st_size == 0:
        raise HarnessError(f"model-gate log is absent or empty: {log}")
    result = {
        "log": str(log),
        "log_sha256": sha256_file(log),
        "model_key": model_key,
        "passed": True,
        "test_name": test_name,
        "vllm_cpp_sha": vllm_cpp_sha,
    }
    write_json_atomic(output, result)
    return result


def record_trace_status(
    output: pathlib.Path,
    *,
    model_key: str,
    ours_nsys_report: pathlib.Path,
    ours_kernel_summary: pathlib.Path,
    ours_command: pathlib.Path,
    ours_profile_log: pathlib.Path,
    ours_client_results: Sequence[pathlib.Path],
    ours_client_logs: Sequence[pathlib.Path],
    vllm_torch_trace: pathlib.Path,
    vllm_kernel_summary: pathlib.Path,
    vllm_command: pathlib.Path,
    vllm_profile_log: pathlib.Path,
    vllm_metadata: pathlib.Path,
    vllm_corpus: pathlib.Path,
    cache_drop_reports: Sequence[pathlib.Path],
    vllm_cpp_sha: str,
) -> dict[str, Any]:
    """Hash the mandatory paired execution-trace artifacts.

    vLLM's V1 EngineCore is profiled with the repository-mandated torch-profiler
    fallback because nsys breaks its GB10 startup.  Ours remains an nsys trace.
    """

    if output.exists():
        raise HarnessError(f"refusing to overwrite trace status: {output}")
    if model_key not in MODEL_REVISIONS:
        raise HarnessError(f"unknown model key: {model_key}")
    _require_full_sha(vllm_cpp_sha, "vllm.cpp SHA")
    if (
        len(ours_client_results) != TRACE_REPETITIONS
        or len(ours_client_logs) != TRACE_REPETITIONS
    ):
        raise HarnessError("ours trace must retain exactly three client results and logs")
    if len(cache_drop_reports) != 3:
        raise HarnessError("paired trace must retain before/between/after cache-drop reports")
    cache_drop_artifacts = {
        f"cache_drop_{index}": _validated_cache_drop_artifact(path)
        for index, path in enumerate(cache_drop_reports, start=1)
    }
    generated_texts = []
    for path in ours_client_results:
        record = _load_json_object(path)
        validate_raw_result(
            record,
            concurrency=TRACE_CONCURRENCY,
            expected_requests=TRACE_PROMPTS,
        )
        generated_texts.append(record.get("generated_texts"))
    if any(value != generated_texts[0] for value in generated_texts[1:]):
        raise HarnessError("ours trace client repetitions are not deterministic")
    metadata = _load_json_object(vllm_metadata)
    expected_metadata = {
        "input_len": INPUT_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS[model_key],
        "max_num_seqs": TRACE_CONCURRENCY,
        "num_prompts": TRACE_PROMPTS,
        "output_len": OUTPUT_LEN,
        "profiled_warmup_prompts": TRACE_PROMPTS,
        "repetitions": TRACE_REPETITIONS,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise HarnessError(
                f"vLLM trace metadata {field}={metadata.get(field)!r}; expected {expected}"
            )
    if metadata.get("corpus_sha256") != sha256_file(vllm_corpus):
        raise HarnessError("vLLM trace metadata corpus hash differs")
    if pathlib.Path(str(metadata.get("corpus"))).resolve() != vllm_corpus.resolve():
        raise HarnessError("vLLM trace metadata corpus path differs")
    output_digest = metadata.get("output_digest")
    if not isinstance(output_digest, str) or re.fullmatch(r"[0-9a-f]{64}", output_digest) is None:
        raise HarnessError("vLLM trace metadata output digest is absent")
    artifacts = {
        "ours_command": ours_command,
        "ours_profile_log": ours_profile_log,
        "ours_nsys_report": ours_nsys_report,
        "ours_kernel_summary": ours_kernel_summary,
        **{
            f"ours_client_result_{index}": path
            for index, path in enumerate(ours_client_results, start=1)
        },
        **{
            f"ours_client_log_{index}": path
            for index, path in enumerate(ours_client_logs, start=1)
        },
        "vllm_command": vllm_command,
        "vllm_profile_log": vllm_profile_log,
        "vllm_metadata": vllm_metadata,
        "vllm_corpus": vllm_corpus,
        "vllm_torch_trace": vllm_torch_trace,
        "vllm_kernel_summary": vllm_kernel_summary,
    }
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"trace artifact {name} is absent or empty: {path}")
    result = {
        "artifacts": {
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in artifacts.items()
            },
            **cache_drop_artifacts,
        },
        "model_key": model_key,
        "ours_profiler": "nsys",
        "passed": True,
        "trace_contract": {
            "concurrency": TRACE_CONCURRENCY,
            "input_len": INPUT_LEN,
            "num_prompts": TRACE_PROMPTS,
            "output_len": OUTPUT_LEN,
            "repetitions": TRACE_REPETITIONS,
        },
        "vllm_cpp_sha": vllm_cpp_sha,
        "vllm_profiler": "torch-profiler",
    }
    write_json_atomic(output, result)
    return result


def record_oracle_manifest(
    output: pathlib.Path,
    *,
    client: pathlib.Path,
) -> dict[str, Any]:
    """Record the exact immutable files behind the pip-vLLM oracle command.

    This command is intentionally run with the oracle venv's Python.  The
    repository parity pin defines the audited client contract, while pip vLLM
    0.24.0 is the executable correctness/performance oracle mandated by the
    project environment record.
    """

    if output.exists():
        raise HarnessError(f"refusing to overwrite oracle manifest: {output}")
    try:
        import vllm  # Delayed so CPU contract tests do not require vLLM.
    except ImportError as error:
        raise HarnessError("record-oracle must run with the vLLM oracle Python") from error
    try:
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise HarnessError("vLLM distribution metadata is unavailable") from error
    metadata_version = distribution.version
    runtime_version = getattr(vllm, "__version__", None)
    if (
        metadata_version != VLLM_ORACLE_VERSION
        or runtime_version != VLLM_ORACLE_VERSION
    ):
        raise HarnessError(
            "vLLM oracle version drift: "
            f"metadata={metadata_version!r}, runtime={runtime_version!r}, "
            f"expected={VLLM_ORACLE_VERSION!r}"
        )
    try:
        import pandas  # Delayed so CPU contract tests can provide a stub.
        pandas_distribution = importlib.metadata.distribution("pandas")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise HarnessError(
            f"pinned pandas {PANDAS_VERSION} is required for vllm bench serve"
        ) from error
    pandas_runtime_version = getattr(pandas, "__version__", None)
    if (
        pandas_distribution.version != PANDAS_VERSION
        or pandas_runtime_version != PANDAS_VERSION
    ):
        raise HarnessError(
            "pandas benchmark dependency drift: "
            f"metadata={pandas_distribution.version!r}, "
            f"runtime={pandas_runtime_version!r}, expected={PANDAS_VERSION!r}"
        )
    package_init_value = getattr(vllm, "__file__", None)
    if not isinstance(package_init_value, str):
        raise HarnessError("vLLM oracle package has no filesystem source")
    package_init = pathlib.Path(package_init_value).absolute()
    package_root = package_init.parent
    python = pathlib.Path(sys.executable).absolute()
    client = client.absolute()
    if python.parent != client.parent:
        raise HarnessError("oracle client and Python are not from the same environment")
    dist_info_value = getattr(distribution, "_path", None)
    if dist_info_value is None:
        raise HarnessError("vLLM distribution metadata path is unavailable")
    dist_info = pathlib.Path(dist_info_value).absolute()
    pandas_init_value = getattr(pandas, "__file__", None)
    pandas_dist_info_value = getattr(pandas_distribution, "_path", None)
    if not isinstance(pandas_init_value, str) or pandas_dist_info_value is None:
        raise HarnessError("pandas benchmark dependency has no filesystem metadata")
    pandas_init = pathlib.Path(pandas_init_value).absolute()
    pandas_dist_info = pathlib.Path(pandas_dist_info_value).absolute()
    artifacts = {
        "bench_datasets": package_root / "benchmarks" / "datasets" / "datasets.py",
        "bench_serve": package_root / "benchmarks" / "serve.py",
        "cli_bench_serve": package_root / "entrypoints" / "cli" / "benchmark" / "serve.py",
        "client": client,
        "distribution_metadata": dist_info / "METADATA",
        "distribution_record": dist_info / "RECORD",
        "package_init": package_init,
        "python": python,
        "pandas_distribution_metadata": pandas_dist_info / "METADATA",
        "pandas_distribution_record": pandas_dist_info / "RECORD",
        "pandas_package_init": pandas_init,
    }
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"oracle artifact {name} is absent or empty: {path}")
    result = {
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "bench_dependencies": {"pandas": PANDAS_VERSION},
        "client_contract_source_commit": VLLM_COMMIT,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "oracle_version": metadata_version,
        "runtime_version": runtime_version,
    }
    write_json_atomic(output, result)
    return result


def record_execution_manifest(
    output: pathlib.Path,
    *,
    model_key: str,
    vllm_cpp_sha: str,
    build_dir: pathlib.Path,
    client: pathlib.Path,
    snapshot: pathlib.Path,
    build_command: pathlib.Path,
    build_log: pathlib.Path,
    oracle_manifest: pathlib.Path,
    port: int,
    num_blocks: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> dict[str, Any]:
    if output.exists():
        raise HarnessError(f"refusing to overwrite execution manifest: {output}")
    if model_key not in MODEL_REVISIONS:
        raise HarnessError(f"unknown model key: {model_key}")
    _require_full_sha(vllm_cpp_sha, "vllm.cpp SHA")
    if port <= 0 or port > 65535 or num_blocks <= 0:
        raise HarnessError("port and num-blocks must be positive and valid")
    if max_num_seqs != MAX_NUM_SEQS:
        raise HarnessError(f"max-num-seqs must equal the gate value {MAX_NUM_SEQS}")
    if max_num_batched_tokens != MAX_NUM_BATCHED_TOKENS[model_key]:
        raise HarnessError("max-num-batched-tokens differs from the model gate value")
    if snapshot.name != MODEL_REVISIONS[model_key]:
        raise HarnessError(
            f"snapshot directory {snapshot.name!r} differs from the pinned revision"
        )
    weight_files = sorted(snapshot.glob("*.safetensors"))
    if not weight_files:
        raise HarnessError(f"checkpoint has no safetensors weights: {snapshot}")
    snapshot_files = sorted(
        path
        for path in snapshot.iterdir()
        if path.is_file()
        and path not in weight_files
        and path.name not in {"config.json", "tokenizer.json"}
    )
    artifacts = {
        "client": client,
        "build_command": build_command,
        "build_log": build_log,
        "cmake_cache": build_dir / "CMakeCache.txt",
        "model_config": snapshot / "config.json",
        "oracle_manifest": oracle_manifest,
        "server": build_dir / "examples" / "server",
        "tokenizer": snapshot / "tokenizer.json",
    }
    oracle = _load_json_object(oracle_manifest)
    if oracle.get("oracle_version") != VLLM_ORACLE_VERSION:
        raise HarnessError("oracle manifest version differs from the pinned pip oracle")
    if oracle.get("runtime_version") != VLLM_ORACLE_VERSION:
        raise HarnessError("oracle runtime version differs from the pinned pip oracle")
    if oracle.get("client_contract_source_commit") != VLLM_COMMIT:
        raise HarnessError("oracle client contract source differs from the parity pin")
    if oracle.get("bench_dependencies") != {"pandas": PANDAS_VERSION}:
        raise HarnessError("oracle benchmark dependency inventory differs from the pin")
    oracle_artifacts = oracle.get("artifacts")
    required_oracle_artifacts = {
        "bench_datasets",
        "bench_serve",
        "cli_bench_serve",
        "client",
        "distribution_metadata",
        "distribution_record",
        "package_init",
        "python",
        "pandas_distribution_metadata",
        "pandas_distribution_record",
        "pandas_package_init",
    }
    if not isinstance(oracle_artifacts, dict) or set(oracle_artifacts) != required_oracle_artifacts:
        raise HarnessError("oracle artifact inventory is incomplete or unexpected")
    for name, artifact in oracle_artifacts.items():
        if not isinstance(artifact, dict):
            raise HarnessError(f"oracle artifact {name} has no hash record")
        path_value = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(path_value, str) or not pathlib.Path(path_value).is_absolute():
            raise HarnessError(f"oracle artifact {name} path is not absolute")
        path = pathlib.Path(path_value)
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"oracle artifact {name} is absent or empty: {path}")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            raise HarnessError(f"oracle artifact {name} hash drifted before execution")
        artifacts[f"oracle:{name}"] = path
    oracle_client = pathlib.Path(oracle_artifacts["client"]["path"])
    if oracle_client.resolve() != client.resolve():
        raise HarnessError("execution client differs from the oracle manifest")
    artifacts.update({f"weight:{path.name}": path for path in weight_files})
    artifacts.update({f"snapshot:{path.name}": path for path in snapshot_files})
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"execution artifact {name} is absent or empty: {path}")
    result = {
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "kernel": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "cache_drop_roots": [
            str(snapshot.absolute()),
            str((output.parent.parent / "corpus" / model_key).absolute()),
            str((build_dir / "examples" / "server").absolute()),
            str(client.absolute()),
        ],
        "bench_dependencies": {"pandas": PANDAS_VERSION},
        "model_key": model_key,
        "model_revision": MODEL_REVISIONS[model_key],
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "vllm_oracle_version": VLLM_ORACLE_VERSION,
        "snapshot_files": [path.name for path in snapshot_files],
        "weight_files": [path.name for path in weight_files],
        "num_blocks": num_blocks,
        "port": port,
        "vllm_cpp_sha": vllm_cpp_sha,
        "vllm_source_sha": VLLM_COMMIT,
    }
    write_json_atomic(output, result)
    return result


def _load_json_object(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HarnessError(f"missing JSON artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path}: expected a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--claim-root", type=pathlib.Path, required=True)
    plan.add_argument("--vllm-cpp-sha", required=True)
    plan.add_argument("--client", type=pathlib.Path, required=True)
    plan.add_argument("--output", type=pathlib.Path, required=True)

    validate_plan_cmd = commands.add_parser("validate-plan")
    validate_plan_cmd.add_argument("path", type=pathlib.Path)
    validate_plan_cmd.add_argument("--vllm-cpp-sha", required=True)

    prepare = commands.add_parser("prepare-corpus")
    prepare.add_argument("--source", type=pathlib.Path, required=True)
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    prepare.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)

    validate = commands.add_parser("validate-raw")
    validate.add_argument("path", type=pathlib.Path)
    validate.add_argument("--concurrency", type=int, required=True)
    validate.add_argument("--requests", type=int)

    memory_return = commands.add_parser("record-memory-return")
    memory_return.add_argument("--output", type=pathlib.Path, required=True)
    memory_return.add_argument("--baseline-kib", type=int, required=True)
    memory_return.add_argument("--final-kib", type=int, required=True)
    memory_return.add_argument("--tolerance-kib", type=int, default=1048576)
    memory_return.add_argument(
        "--before-cache-drop-report", type=pathlib.Path, required=True
    )
    memory_return.add_argument(
        "--after-cache-drop-report", type=pathlib.Path, required=True
    )
    memory_return.add_argument("--gpu-idle", action="store_true")

    model_gate = commands.add_parser("record-model-gate")
    model_gate.add_argument("--output", type=pathlib.Path, required=True)
    model_gate.add_argument("--log", type=pathlib.Path, required=True)
    model_gate.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)
    model_gate.add_argument("--test-name", required=True)
    model_gate.add_argument("--vllm-cpp-sha", required=True)

    trace = commands.add_parser("record-trace-status")
    trace.add_argument("--output", type=pathlib.Path, required=True)
    trace.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)
    trace.add_argument("--ours-nsys-report", type=pathlib.Path, required=True)
    trace.add_argument("--ours-kernel-summary", type=pathlib.Path, required=True)
    trace.add_argument("--ours-command", type=pathlib.Path, required=True)
    trace.add_argument("--ours-profile-log", type=pathlib.Path, required=True)
    trace.add_argument("--ours-client-result", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--ours-client-log", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-torch-trace", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-kernel-summary", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-command", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-profile-log", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-metadata", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-corpus", type=pathlib.Path, required=True)
    trace.add_argument(
        "--cache-drop-report", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument("--vllm-cpp-sha", required=True)

    oracle = commands.add_parser("record-oracle")
    oracle.add_argument("--output", type=pathlib.Path, required=True)
    oracle.add_argument("--client", type=pathlib.Path, required=True)

    execution = commands.add_parser("record-execution")
    execution.add_argument("--output", type=pathlib.Path, required=True)
    execution.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)
    execution.add_argument("--vllm-cpp-sha", required=True)
    execution.add_argument("--build-dir", type=pathlib.Path, required=True)
    execution.add_argument("--client", type=pathlib.Path, required=True)
    execution.add_argument("--snapshot", type=pathlib.Path, required=True)
    execution.add_argument("--build-command", type=pathlib.Path, required=True)
    execution.add_argument("--build-log", type=pathlib.Path, required=True)
    execution.add_argument("--oracle-manifest", type=pathlib.Path, required=True)
    execution.add_argument("--port", type=int, required=True)
    execution.add_argument("--num-blocks", type=int, required=True)
    execution.add_argument("--max-num-seqs", type=int, required=True)
    execution.add_argument("--max-num-batched-tokens", type=int, required=True)

    bench = commands.add_parser("bench")
    bench.add_argument("--client", type=pathlib.Path, required=True)
    bench.add_argument("--tokenizer", type=pathlib.Path, required=True)
    bench.add_argument("--evidence", type=pathlib.Path, required=True)
    bench.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)
    bench.add_argument("--engine", choices=ENGINES, required=True)
    bench.add_argument("--base-url", required=True)
    bench.add_argument("--concurrency", type=int, required=True)
    bench.add_argument("--repetition", type=int, required=True)
    bench.add_argument("--artifact-tag", default="")
    bench.add_argument("--num-prompts", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite plan: {args.output}")
        if args.output.parent.exists() and any(args.output.parent.iterdir()):
            raise HarnessError(f"refusing to mix plan artifacts in {args.output.parent}")
        result = build_plan(
            claim_root=args.claim_root,
            vllm_cpp_sha=args.vllm_cpp_sha,
            client=args.client,
        )
        write_json_atomic(args.output, result)
    elif args.command == "validate-plan":
        result = validate_plan(args.path, vllm_cpp_sha=args.vllm_cpp_sha)
    elif args.command == "prepare-corpus":
        result = prepare_corpus_views(
            args.source,
            args.output,
            model_key=args.model_key,
        )
    elif args.command == "validate-raw":
        result = _load_json_object(args.path)
        validate_raw_result(
            result,
            concurrency=args.concurrency,
            expected_requests=args.requests,
        )
    elif args.command == "record-memory-return":
        result = record_memory_return(
            args.output,
            baseline_mem_available_kib=args.baseline_kib,
            final_mem_available_kib=args.final_kib,
            tolerance_kib=args.tolerance_kib,
            before_cache_drop_report=args.before_cache_drop_report,
            after_cache_drop_report=args.after_cache_drop_report,
            gpu_idle=args.gpu_idle,
        )
    elif args.command == "record-model-gate":
        result = record_model_gate(
            args.output,
            log=args.log,
            model_key=args.model_key,
            test_name=args.test_name,
            vllm_cpp_sha=args.vllm_cpp_sha,
        )
    elif args.command == "record-trace-status":
        result = record_trace_status(
            args.output,
            model_key=args.model_key,
            ours_nsys_report=args.ours_nsys_report,
            ours_kernel_summary=args.ours_kernel_summary,
            ours_command=args.ours_command,
            ours_profile_log=args.ours_profile_log,
            ours_client_results=args.ours_client_result,
            ours_client_logs=args.ours_client_log,
            vllm_torch_trace=args.vllm_torch_trace,
            vllm_kernel_summary=args.vllm_kernel_summary,
            vllm_command=args.vllm_command,
            vllm_profile_log=args.vllm_profile_log,
            vllm_metadata=args.vllm_metadata,
            vllm_corpus=args.vllm_corpus,
            cache_drop_reports=args.cache_drop_report,
            vllm_cpp_sha=args.vllm_cpp_sha,
        )
    elif args.command == "record-oracle":
        result = record_oracle_manifest(args.output, client=args.client)
    elif args.command == "record-execution":
        result = record_execution_manifest(
            args.output,
            model_key=args.model_key,
            vllm_cpp_sha=args.vllm_cpp_sha,
            build_dir=args.build_dir,
            client=args.client,
            snapshot=args.snapshot,
            build_command=args.build_command,
            build_log=args.build_log,
            oracle_manifest=args.oracle_manifest,
            port=args.port,
            num_blocks=args.num_blocks,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
    else:
        result = run_benchmark(
            OnlineRun(
                client=args.client,
                tokenizer=args.tokenizer,
                evidence_root=args.evidence,
                model_key=args.model_key,
                engine=args.engine,
                base_url=args.base_url,
                concurrency=args.concurrency,
                repetition=args.repetition,
                artifact_tag=args.artifact_tag,
                num_prompts_override=args.num_prompts,
            )
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"online-gate: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error
