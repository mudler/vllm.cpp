#!/usr/bin/env python3
"""Fail-closed orchestration helpers for ``SERVE-GATE-ONLINE``.

The timed client remains the unmodified pinned vLLM ``bench serve`` command.
This wrapper mirrors its CLI and result schema from:

* ``vllm/benchmarks/serve.py:321-353,563-748,1188-1284,2082-2284``
* ``vllm/benchmarks/datasets/datasets.py:2482-2610``
* ``tests/benchmarks/test_serve_cli.py:58-132``

at executable-oracle commit ``702f4814fe54fabff350d43cb753ae3e47c0c276``. It
does not reimplement request timing.  It constructs the exact upstream client
command, validates every detailed artifact, and makes partial request sets
fatal instead of allowing their aggregate metrics to look successful.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import importlib.metadata
import json
import mmap
import os
import pathlib
import platform
import re
import shlex
import sqlite3
import subprocess
import sys
from collections import Counter
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
VLLM_ORACLE_VERSION = "0.25.0"
FLASHINFER_VERSION = "0.6.13"
PANDAS_VERSION = "2.2.3"
TRACE_CONCURRENCY = 16
TRACE_PROMPTS = 48
TRACE_REPETITIONS = 3
TRACE_CAPTURE_GRAPH_REPLAYS = 4
TRACE_STATUS_SCHEMA_VERSION = 2
BUILD_CONTRACT_SCHEMA_VERSION = 2
NSYS_CAPTURE_RANGE = "cudaProfilerApi"
NSYS_CUDA_GRAPH_TRACE = "node:host-only"
NSYS_CUDA_FLUSH_INTERVAL_MS = 0
NSYS_PRODUCT_VERSION = "2025.3.2.474"
DGX_CUDA_COMPILER = pathlib.Path("/usr/local/cuda-13.0/bin/nvcc")
DGX_CUDA_COMPILER_VERSION = "13.0.88"
NVFP4_PLAN_FIXTURE_SHA256 = (
    "e81e9181db20d0537a43a101fe4f93aa57df9e42900e8a21c91cafa61e107edd"
)
NVFP4_SELECTED_PLAN_SHA256 = (
    "f2d9be7fc4a89de1cfa994ab9be08a423e0c4f6981fe46cb808cef485f4c1fa4"
)
NVFP4_PLAN_METADATA = "2e429d4cd3977f0f"
TRACE_REQUIRED_ENV = {
    "VT_FP4_AUTOTUNE": "1",
    "VT_FP4_AUTOTUNE_CACHE_READONLY": "1",
    "VT_FP4_AUTOTUNE_DELAY_US": "5000",
    "VT_FP4_FULL_TACTICS": "1",
    "VT_FP4_PERSISTENT_CACHE": "1",
    "VT_FP4_PLAN_CACHE": "1",
    "VT_FP4_PRE_SERVE_WARMUP": "1",
}
TRACE_PRIMARY_GRAPH_CONTRACTS = {
    "27": {
        "graph_child_nodes": {"kernel": 1_107, "memcpy": 7, "memset": 1},
        "node_count": 1_107,
        "families": {
            "fa2_combine": ("flash_fwd_splitkv_combine_kernel", 16),
            "fa2_main": ("flash_fwd_splitkv_kernel", 16),
            "fp4_gemm": (
                "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal",
                208,
            ),
            "fused_fp4_producer": ("SiluAndMulFp4QuantKernel", 64),
            "gdn_recurrence": ("GdnDecodeFusedKernel", 48),
            "normal_fp4_producer": ("ScaledFp4QuantKernel", 144),
        },
    }
}
VLLM_DECODE_FAMILY_CONTRACTS = {
    "27": {
        "fa2_combine": ("flash_fwd_splitkv_combine_kernel", 16),
        "fa2_main": ("flash_fwd_splitkv_kernel", 16),
        "fp4_gemm": ("MainloopSm120TmaWarpSpecializedBlockScaled", 208),
        "fused_fp4_producer": ("silu_mul_cvt_fp16_to_fp4", 64),
        "gdn_recurrence": (
            "fused_recurrent_gated_delta_rule_packed_decode_kernel",
            48,
        ),
        "normal_fp4_producer": ("cvt_fp16_to_fp4<__nv_bfloat16, false>", 144),
    }
}
VLLM_GENERATION_WINDOW_CONTRACTS = {
    "27": {"all": 1_588, "clean": 1_476},
}
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
MAX_MODEL_LEN = {"27": 262144, "35": 262144}
ENGINES = ("ours", "vllm")
PERCENTILE_METRICS = ("ttft", "tpot", "itl", "e2el")
PERCENTILES = (50, 90, 99)
CACHE_DROP_METHOD = "posix_fadvise-dontneed+mincore"

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_NSYS_ALLOWED_DIAGNOSTIC = re.compile(
    r"CUDA device \d+: Unified Memory trace is not supported by the current "
    r"driver version or configuration\."
)

_PLAN_PREPARED_RE = re.compile(
    r"^\[VT_FP4_CACHE\] prepared mode=(\S+) native=(\S+) flashinfer=(\S+) "
    r"loaded=(\d+) \(flashinfer=(\d+) native=(\d+)\) rejected=(\d+) "
    r"delay_us=(\d+) metadata=([0-9a-f]+) selected=(\d+)$"
)
_PLAN_COMPLETE_RE = re.compile(
    r"^\[VT_FP4_CACHE\] complete mode=(\S+) loaded=(\d+) tuned=(\d+) "
    r"rejected=(\d+) saved=(\d+) selected=(\d+) metadata=([0-9a-f]+)$"
)
_PLAN_SELECTED_RE = re.compile(
    r"^\[VT_FP4_CACHE\] selected M=(\d+) N=(\d+) K=(\d+) tactic=(\d+)$"
)
_PLAN_WARMUP_RE = re.compile(
    r"^\[VT_FP4_AUTOTUNE\] pre-serve warmup complete max_tokens=(\d+) "
    r"profiles_requested=(\d+) profiles_tuned=(\d+) cached_plans=(\d+)$"
)
_CUDA_PROFILE_READY_RE = re.compile(
    r"^\[VT_CUDA_PROFILE\] ready pid=(\d+) signal=SIGUSR2 target_replays=(\d+)$"
)
_CUDA_PROFILE_STARTED_RE = re.compile(
    r"^\[VT_CUDA_PROFILE\] started target_replays=(\d+) "
    r"graph=(0x[0-9a-f]+) real_batch=(\d+) padded_batch=(\d+) "
    r"prior_replays=(\d+)$"
)
_CUDA_PROFILE_STOPPED_RE = re.compile(
    r"^\[VT_CUDA_PROFILE\] stopped captured_replays=(\d+) "
    r"graph=(0x[0-9a-f]+)$"
)
_BENCH_SHUTDOWN_READY_RE = re.compile(
    r"^\[VT_BENCH_SHUTDOWN\] ready pid=(\d+) control=fifo$"
)
_BENCH_SHUTDOWN_REQUESTED_RE = re.compile(
    r"^\[VT_BENCH_SHUTDOWN\] requested control=fifo$"
)
_BENCH_SHUTDOWN_COMPLETED_RE = re.compile(
    r"^\[VT_BENCH_SHUTDOWN\] completed control=fifo$"
)


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
    num_warmups_override: int | None = None

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
        if self.num_warmups_override is not None and self.num_warmups_override < 0:
            raise HarnessError("num-warmups override must be non-negative")
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
    def num_warmups(self) -> int:
        return (
            self.num_warmups_override
            if self.num_warmups_override is not None
            else self.concurrency
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


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _fingerprint_tree(root: pathlib.Path) -> dict[str, Any]:
    """Hash a dependency tree by relative path, byte size, and file content."""

    if not root.is_dir():
        raise HarnessError(f"dependency source tree is absent: {root}")
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise HarnessError(f"dependency source tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    if not inventory:
        raise HarnessError(f"dependency source tree contains no files: {root}")
    return {
        "file_count": len(inventory),
        "logical_bytes": sum(item["size"] for item in inventory),
        "path": str(root.absolute()),
        "sha256": _sha256_canonical(inventory),
    }


def _file_contains(path: pathlib.Path, needle: bytes) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as source, mmap.mmap(
        source.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        return mapped.find(needle) >= 0


def _require_sqlite_columns(
    connection: sqlite3.Connection,
    table: str,
    expected: set[str],
) -> None:
    actual = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    }
    missing = sorted(expected - actual)
    if missing:
        raise HarnessError(
            f"Nsight SQLite table {table} omits required columns: "
            + ", ".join(missing)
        )


def _option_value(tokens: Sequence[str], flag: str) -> str | None:
    matches = []
    prefix = f"{flag}="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            matches.append(token[len(prefix) :])
        elif token == flag:
            if index + 1 >= len(tokens):
                raise HarnessError(f"command option {flag} has no value")
            matches.append(tokens[index + 1])
    if len(matches) > 1:
        raise HarnessError(f"command option {flag} is repeated")
    return matches[0] if matches else None


def _require_option(tokens: Sequence[str], flag: str, expected: str) -> None:
    actual = _option_value(tokens, flag)
    if actual != expected:
        raise HarnessError(f"command {flag}={actual!r}; expected {expected!r}")


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
        str(run.num_warmups),
        "--ready-check-timeout-sec",
        "0",
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


def precise_max_concurrent_requests(record: Mapping[str, Any]) -> int:
    """Compute exact request overlap from detailed half-open time intervals.

    Pinned vLLM's reported ``max_concurrent_requests`` uses inclusive one-second
    buckets, so sequential requests that straddle a bucket boundary can be
    reported as overlapping.  Detailed start/TTFT/ITL arrays retain the exact
    intervals needed for the binding concurrency check.
    """

    start_times = record.get("start_times")
    ttfts = record.get("ttfts")
    itls = record.get("itls")
    if (
        not isinstance(start_times, list)
        or not isinstance(ttfts, list)
        or not isinstance(itls, list)
    ):
        raise HarnessError("precise concurrency requires start_times, ttfts and itls")
    if not start_times or len(start_times) != len(ttfts) or len(start_times) != len(itls):
        raise HarnessError("precise concurrency arrays have inconsistent cardinality")
    events: list[tuple[float, int]] = []
    for index, (start_value, ttft_value, itl_values) in enumerate(
        zip(start_times, ttfts, itls)
    ):
        start = require_number(start_value, f"start_times[{index}]")
        ttft = require_number(ttft_value, f"ttfts[{index}]")
        if not isinstance(itl_values, list):
            raise HarnessError(f"itls[{index}] is not a list")
        latency = ttft + sum(
            require_number(value, f"itls[{index}][{value_index}]")
            for value_index, value in enumerate(itl_values)
        )
        if start < 0.0 or latency < 0.0:
            raise HarnessError("precise concurrency interval contains a negative value")
        events.append((start, 1))
        events.append((start + latency, -1))
    # End events sort before starts at identical timestamps: [start, end).
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        if active < 0:
            raise HarnessError("precise concurrency event order is invalid")
        peak = max(peak, active)
    if active != 0:
        raise HarnessError("precise concurrency intervals did not close")
    return peak


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
    # RequestOutputCollector deliberately merges DELTA outputs when the
    # producer gets ahead of the consumer (matching pinned vLLM). The pinned
    # benchmark client consequently treats ITLs as inter-*chunk* timings and
    # uses native usage for the exact token count; one chunk may carry more
    # than one token. Reject impossible extra timing events, but do not reject
    # an upstream-legal merged delta merely because it retained fewer than
    # OUTPUT_LEN - 1 intervals.
    if any(
        not isinstance(row, list) or len(row) > OUTPUT_LEN - 1
        for row in itls
    ):
        raise HarnessError("request ITL samples exceed output_len-1 chunk intervals")
    for name, values in (("ttfts", ttfts), ("start_times", start_times)):
        for index, value in enumerate(values):
            if require_number(value, f"{name}[{index}]") < 0.0:
                raise HarnessError(f"{name} contains a negative value")
    for row_index, row in enumerate(itls):
        for value_index, value in enumerate(row):
            if require_number(value, f"itls[{row_index}][{value_index}]") < 0.0:
                raise HarnessError("itls contains a negative value")

    bucketed_peak = record.get("max_concurrent_requests")
    if isinstance(bucketed_peak, bool) or not isinstance(bucketed_peak, int):
        raise HarnessError("max_concurrent_requests is not an integer")
    precise_peak = precise_max_concurrent_requests(record)
    expected_peak = min(concurrency, expected)
    if precise_peak != expected_peak:
        raise HarnessError(
            f"precise peak concurrency {precise_peak}; expected {expected_peak}"
        )
    if bucketed_peak < precise_peak:
        raise HarnessError("upstream bucketed peak is below precise concurrency")

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
                "BLOCKED_UNTIL_H1D_G4_AND_SEPARATE_PRODUCTION_TRACE_BUILDS"
            ],
            "trace_model": [
                "scripts/dgx-online-serving.sh",
                "--trace-only",
                "--model",
                "27",
                "--snapshot",
                "<27_MODEL_SNAPSHOT>",
                "--source-corpus",
                "<EVIDENCE>/corpus/27",
                "--evidence",
                "<EVIDENCE>",
                "--build-dir",
                "<H1D_TRACE_BUILD>",
                "--configure-log",
                "<H1D_TRACE_CONFIGURE_LOG>",
                "--client",
                str(client),
                "--vllm-cpp-sha",
                vllm_cpp_sha,
            ],
        },
        "required_artifacts": [
            "manifest.json",
            "corpus/<model>/vllm/manifest.json",
            "execution/<model>.json (production timing only)",
            "execution/<model>-trace.json (H1d diagnostic only)",
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
            "trace/<model>/ours-r{1,2,3}.nsys-rep",
            "trace/<model>/ours-r{1,2,3}.sqlite",
            "trace/<model>/ours-r{1,2,3}-cuda_gpu_kern_sum.txt",
            "trace/<model>/ours-r{1,2,3}-nsys-validation.json",
            "trace/<model>/ours-r{1,2,3}-profile-control.json",
            "raw/<model>/ours/c16-r1-trace{1,2,3}-probe.json",
            "trace/<model>/vllm-profile/*.pt.trace.json.gz",
            "trace/<model>/vllm-kernels.json",
            "trace/<model>/cache-{before-ours,between-engines,after-vllm}.json",
            "trace/<model>/status.json",
            "summary-<model>/{all-runs,ratios}.json",
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
        "vllm_oracle_bench_dependencies": {
            "flashinfer": FLASHINFER_VERSION,
            "pandas": PANDAS_VERSION,
        },
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
    if value.get("vllm_oracle_bench_dependencies") != {
        "flashinfer": FLASHINFER_VERSION,
        "pandas": PANDAS_VERSION,
    }:
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


def validate_nsys_trace(
    sqlite_path: pathlib.Path, *, model_key: str | None = None
) -> dict[str, Any]:
    """Validate one bounded H1d capture by direct runtime correlation."""

    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise HarnessError(f"Nsight SQLite artifact is absent or empty: {sqlite_path}")
    connection = sqlite3.connect(str(sqlite_path))
    try:
        connection.execute("PRAGMA query_only = ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required_tables = {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "DIAGNOSTIC_EVENT",
            "META_DATA_CAPTURE",
            "META_DATA_EXPORT",
            "StringIds",
            "TARGET_INFO_SESSION_START_TIME",
        }
        missing_tables = sorted(required_tables - tables)
        if missing_tables:
            raise HarnessError(
                "Nsight SQLite omits required tables: " + ", ".join(missing_tables)
            )
        _require_sqlite_columns(
            connection,
            "DIAGNOSTIC_EVENT",
            {"severity", "text", "timestamp"},
        )
        _require_sqlite_columns(connection, "StringIds", {"id", "value"})
        _require_sqlite_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            {"correlationId", "nameId", "returnValue", "start"},
        )
        _require_sqlite_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            {
                "blockX",
                "blockY",
                "blockZ",
                "cacheConfig",
                "correlationId",
                "demangledName",
                "dynamicSharedMemory",
                "graphNodeId",
                "gridX",
                "gridY",
                "gridZ",
                "launchType",
                "localMemoryPerThread",
                "localMemoryTotal",
                "mangledName",
                "registersPerThread",
                "sharedMemoryExecuted",
                "sharedMemoryLimitConfig",
                "shortName",
                "staticSharedMemory",
            },
        )
        _require_sqlite_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            {
                "bytes",
                "copyCount",
                "copyKind",
                "correlationId",
                "dstContextId",
                "dstDeviceId",
                "dstKind",
                "graphNodeId",
                "migrationCause",
                "srcContextId",
                "srcDeviceId",
                "srcKind",
            },
        )
        _require_sqlite_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            {"bytes", "correlationId", "graphNodeId", "memKind", "value"},
        )
        _require_sqlite_columns(connection, "META_DATA_CAPTURE", {"name", "value"})
        _require_sqlite_columns(connection, "META_DATA_EXPORT", {"name", "value"})
        _require_sqlite_columns(
            connection,
            "TARGET_INFO_SESSION_START_TIME",
            {"localTime", "utcEpochNs", "utcTime"},
        )

        def unique_metadata_value(table: str, name: str) -> str:
            values = [
                str(row[0])
                for row in connection.execute(
                    f"SELECT value FROM {table} WHERE name = ?", (name,)
                )
            ]
            if len(values) != 1 or not values[0]:
                raise HarnessError(
                    f"Nsight SQLite {table}.{name} is absent or non-unique"
                )
            return values[0]

        capture_uuid = unique_metadata_value(
            "META_DATA_CAPTURE", "PROFILING_SESSION_UUID"
        )
        if re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            capture_uuid,
        ) is None:
            raise HarnessError("Nsight profiling-session UUID is invalid")
        product_name = unique_metadata_value(
            "META_DATA_EXPORT", "EXPORT_PRODUCT_NAME"
        )
        product_version = unique_metadata_value(
            "META_DATA_EXPORT", "EXPORT_PRODUCT_VERSION"
        )
        if (
            product_name != "NVIDIA Nsight Systems"
            or product_version != NSYS_PRODUCT_VERSION
        ):
            raise HarnessError(
                "Nsight export product/version differs from the H1d pin"
            )
        if unique_metadata_value("META_DATA_EXPORT", "EXPORT_PARAM_LAZY") != "false":
            raise HarnessError("Nsight SQLite export was not materialized with --lazy=false")
        report_path = pathlib.Path(
            unique_metadata_value("META_DATA_EXPORT", "EXPORT_PARAM_INPUT_PATH_ABS")
        )
        exported_sqlite_path = pathlib.Path(
            unique_metadata_value("META_DATA_EXPORT", "EXPORT_PARAM_OUTPUT_PATH_ABS")
        )
        if (
            not report_path.is_absolute()
            or not report_path.is_file()
            or report_path.stat().st_size == 0
        ):
            raise HarnessError("Nsight source report linked by SQLite is absent")
        if (
            not exported_sqlite_path.is_absolute()
            or exported_sqlite_path.resolve() != sqlite_path.resolve()
        ):
            raise HarnessError("Nsight SQLite export output path differs from its artifact")
        session_rows = list(
            connection.execute(
                "SELECT utcEpochNs, utcTime, localTime "
                "FROM TARGET_INFO_SESSION_START_TIME"
            )
        )
        if (
            len(session_rows) != 1
            or not isinstance(session_rows[0][0], int)
            or session_rows[0][0] <= 0
            or any(not str(value) for value in session_rows[0][1:])
        ):
            raise HarnessError("Nsight session start metadata is absent or invalid")

        diagnostics = [
            {"severity": int(severity), "text": str(text)}
            for severity, text in connection.execute(
                "SELECT severity, text FROM DIAGNOSTIC_EVENT "
                "WHERE severity >= 2 ORDER BY timestamp"
            )
        ]
        rejected_diagnostics = [
            diagnostic
            for diagnostic in diagnostics
            if _NSYS_ALLOWED_DIAGNOSTIC.fullmatch(diagnostic["text"]) is None
        ]
        if rejected_diagnostics:
            first = rejected_diagnostics[0]
            raise HarnessError(
                "Nsight CUDA collection is not lossless: "
                f"severity={first['severity']} text={first['text']}"
            )

        launch_rows = [
            (int(correlation_id), str(name), int(return_value))
            for correlation_id, name, return_value in connection.execute(
                "SELECT runtime.correlationId, strings.value, runtime.returnValue "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime "
                "JOIN StringIds AS strings ON strings.id = runtime.nameId "
                "WHERE strings.value GLOB 'cudaGraphLaunch*' "
                "ORDER BY runtime.start"
            )
        ]
        if len(launch_rows) != TRACE_CAPTURE_GRAPH_REPLAYS:
            raise HarnessError(
                "Nsight capture has "
                f"{len(launch_rows)} cudaGraphLaunch rows; expected "
                f"{TRACE_CAPTURE_GRAPH_REPLAYS}"
            )
        launch_ids = [row[0] for row in launch_rows]
        if len(set(launch_ids)) != TRACE_CAPTURE_GRAPH_REPLAYS:
            raise HarnessError("Nsight cudaGraphLaunch correlation IDs are not unique")
        if any(return_value != 0 for _, _, return_value in launch_rows):
            raise HarnessError("Nsight cudaGraphLaunch returned a CUDA error")
        launch_id_set = set(launch_ids)

        kernel_fields = (
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
            "registersPerThread",
            "staticSharedMemory",
            "dynamicSharedMemory",
            "localMemoryPerThread",
            "localMemoryTotal",
            "cacheConfig",
            "launchType",
            "sharedMemoryExecuted",
            "sharedMemoryLimitConfig",
        )
        kernel_rows = []
        for row in connection.execute(
            "SELECT kernel.correlationId, kernel.graphNodeId, demangled.value, "
            "short.value, mangled.value, "
            + ", ".join(f"kernel.{field}" for field in kernel_fields)
            + " FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel "
            "JOIN StringIds AS demangled ON demangled.id = kernel.demangledName "
            "JOIN StringIds AS short ON short.id = kernel.shortName "
            "LEFT JOIN StringIds AS mangled ON mangled.id = kernel.mangledName "
            "WHERE kernel.graphNodeId IS NOT NULL"
        ):
            correlation_id, graph_node_id, name, short_name, mangled_name, *values = row
            kernel_rows.append(
                (
                    int(correlation_id),
                    int(graph_node_id),
                    {
                        "kind": "kernel",
                        "mangled_name": (
                            str(mangled_name) if mangled_name is not None else None
                        ),
                        "name": str(name),
                        "short_name": str(short_name),
                        **dict(zip(kernel_fields, values, strict=True)),
                    },
                )
            )
        if not kernel_rows:
            raise HarnessError("Nsight SQLite has no CUDA graph-node kernel rows")

        memcpy_fields = (
            "bytes",
            "copyKind",
            "srcKind",
            "dstKind",
            "srcDeviceId",
            "srcContextId",
            "dstDeviceId",
            "dstContextId",
            "migrationCause",
            "copyCount",
        )
        memcpy_rows = [
            (
                int(row[0]),
                int(row[1]),
                {
                    "kind": "memcpy",
                    **dict(zip(memcpy_fields, row[2:], strict=True)),
                },
            )
            for row in connection.execute(
                "SELECT correlationId, graphNodeId, "
                + ", ".join(memcpy_fields)
                + " FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE graphNodeId IS NOT NULL"
            )
        ]
        memset_fields = ("bytes", "value", "memKind")
        memset_rows = [
            (
                int(row[0]),
                int(row[1]),
                {
                    "kind": "memset",
                    **dict(zip(memset_fields, row[2:], strict=True)),
                },
            )
            for row in connection.execute(
                "SELECT correlationId, graphNodeId, "
                + ", ".join(memset_fields)
                + " FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE graphNodeId IS NOT NULL"
            )
        ]
        child_rows = {
            "kernel": kernel_rows,
            "memcpy": memcpy_rows,
            "memset": memset_rows,
        }
        for kind, rows in child_rows.items():
            outside = [row for row in rows if row[0] not in launch_id_set]
            if outside:
                raise HarnessError(
                    f"Nsight {kind} graph children lack direct cudaGraphLaunch correlation"
                )
            ungraphed = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_{kind.upper()} "
                    "WHERE graphNodeId IS NULL"
                ).fetchone()[0]
            )
            if ungraphed != 0:
                raise HarnessError(
                    f"Nsight bounded range contains {ungraphed} ungraphed {kind} rows"
                )
            replay_counts: dict[int, tuple[int, set[int]]] = {}
            for correlation_id, graph_node_id, _ in rows:
                count, correlations = replay_counts.get(graph_node_id, (0, set()))
                correlations.add(correlation_id)
                replay_counts[graph_node_id] = (count + 1, correlations)
            uneven = [
                graph_node_id
                for graph_node_id, (count, correlations) in replay_counts.items()
                if count != TRACE_CAPTURE_GRAPH_REPLAYS
                or len(correlations) != TRACE_CAPTURE_GRAPH_REPLAYS
            ]
            if uneven:
                raise HarnessError(
                    f"Nsight {kind} graph nodes have uneven replay counts"
                )
            per_node_signatures: dict[int, list[str]] = {}
            for _, graph_node_id, signature in rows:
                per_node_signatures.setdefault(graph_node_id, []).append(
                    canonical_json(signature)
                )
            if any(
                len(set(signatures)) != 1
                for signatures in per_node_signatures.values()
            ):
                raise HarnessError(
                    f"Nsight {kind} graph-node signature differs across launches"
                )

        per_launch_signatures: list[list[dict[str, Any]]] = []
        per_launch_counts = []
        for launch_id in launch_ids:
            signature = sorted(
                [
                    value
                    for rows in child_rows.values()
                    for correlation_id, _, value in rows
                    if correlation_id == launch_id
                ],
                key=canonical_json,
            )
            per_launch_signatures.append(signature)
            per_launch_counts.append(
                {
                    kind: sum(row[0] == launch_id for row in rows)
                    for kind, rows in child_rows.items()
                }
            )
        first_signature = canonical_json(per_launch_signatures[0])
        if any(canonical_json(value) != first_signature for value in per_launch_signatures[1:]):
            raise HarnessError("Nsight graph child node multiset differs across launches")
        if any(value != per_launch_counts[0] for value in per_launch_counts[1:]):
            raise HarnessError("Nsight graph child counts differ across launches")

        primary_family_counts: dict[str, int] = {}
        contract = TRACE_PRIMARY_GRAPH_CONTRACTS.get(model_key) if model_key else None
        if contract is not None:
            expected_counts = contract["graph_child_nodes"]
            if per_launch_counts[0] != expected_counts:
                raise HarnessError(
                    "Nsight graph child counts differ from the model contract: "
                    f"got {per_launch_counts[0]}, expected {expected_counts}"
                )
            primary_node_names = [
                value["name"]
                for value in per_launch_signatures[0]
                if value["kind"] == "kernel"
            ]
            for family, (pattern, expected_count) in contract["families"].items():
                actual_count = sum(pattern in name for name in primary_node_names)
                primary_family_counts[family] = actual_count
                if actual_count != expected_count:
                    raise HarnessError(
                        f"Nsight primary graph {family} node count differs: "
                        f"got {actual_count}, expected {expected_count}"
                    )
            for forbidden in ("MatmulNvfp4Fp4Naive", "MatmulNvfp4Fp4Wmma"):
                if any(forbidden in name for name in primary_node_names):
                    raise HarnessError(
                        f"Nsight trace dispatched forbidden fallback kernel {forbidden}"
                    )

        kernel_summary_rows = [
            {
                "count": int(count),
                "name": str(name),
                "total_duration_ns": int(duration),
            }
            for name, count, duration in connection.execute(
                "SELECT strings.value, COUNT(*), SUM(kernel.end-kernel.start) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel "
                "JOIN StringIds AS strings ON strings.id=kernel.demangledName "
                "WHERE kernel.graphNodeId IS NOT NULL GROUP BY strings.value"
            )
        ]
        kernel_summary_rows.sort(
            key=lambda item: (-item["total_duration_ns"], item["name"])
        )
    finally:
        connection.close()

    return {
        "allowed_diagnostics": diagnostics,
        "capture_session_uuid": capture_uuid,
        "canonical_node_multiset_sha256": hashlib.sha256(
            first_signature.encode("utf-8")
        ).hexdigest(),
        "graph_child_rows": {
            kind: len(rows) for kind, rows in child_rows.items()
        },
        "graph_launch_count": len(launch_rows),
        "graph_launch_names": [row[1] for row in launch_rows],
        "graph_node_count": len({row[1] for row in kernel_rows}),
        "kernel_summary": {
            "kernel_count": len(kernel_rows),
            "kernel_time_ns": sum(
                item["total_duration_ns"] for item in kernel_summary_rows
            ),
            "kernels": kernel_summary_rows,
        },
        "lossless": True,
        "nsys_product_version": product_version,
        "nsys_report_path": str(report_path),
        "nsys_report_sha256": sha256_file(report_path),
        "primary_graph_node_count": len({row[1] for row in kernel_rows}),
        "primary_graph_family_node_counts": primary_family_counts,
        "primary_graph_replay_count": TRACE_CAPTURE_GRAPH_REPLAYS,
        "sqlite_path": str(sqlite_path),
        "sqlite_sha256": sha256_file(sqlite_path),
    }


def summarize_nsys_kernels(sqlite_path: pathlib.Path) -> dict[str, Any]:
    validation = validate_nsys_trace(sqlite_path)
    return {
        **validation["kernel_summary"],
        "sqlite_path": str(sqlite_path),
        "sqlite_sha256": validation["sqlite_sha256"],
    }


def _parse_client_command_log(
    log_path: pathlib.Path,
    *,
    result_path: pathlib.Path,
    corpus_path: pathlib.Path,
    num_prompts: int,
    num_warmups: int,
) -> list[str]:
    try:
        first_line = log_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError) as error:
        raise HarnessError(f"client command log is absent or empty: {log_path}") from error
    prefix = "command: "
    if not first_line.startswith(prefix):
        raise HarnessError(f"client log does not begin with its exact command: {log_path}")
    tokens = shlex.split(first_line[len(prefix) :])
    if len(tokens) < 3 or tokens[1:3] != ["bench", "serve"]:
        raise HarnessError("trace client is not the pinned vllm bench serve command")
    for flag, expected in (
        ("--backend", "openai"),
        ("--endpoint", "/v1/completions"),
        ("--model", "gate"),
        ("--dataset-name", "custom"),
        ("--custom-output-len", str(OUTPUT_LEN)),
        ("--num-prompts", str(num_prompts)),
        ("--max-concurrency", str(TRACE_CONCURRENCY)),
        ("--request-rate", "inf"),
        ("--num-warmups", str(num_warmups)),
        ("--ready-check-timeout-sec", "0"),
        ("--seed", "0"),
        ("--temperature", "0"),
    ):
        _require_option(tokens, flag, expected)
    for flag in (
        "--skip-chat-template",
        "--disable-shuffle",
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--disable-tqdm",
    ):
        if tokens.count(flag) != 1:
            raise HarnessError(f"trace client command must contain {flag} exactly once")
    dataset_value = _option_value(tokens, "--dataset-path")
    if dataset_value is None or pathlib.Path(dataset_value).resolve() != corpus_path.resolve():
        raise HarnessError("trace client command corpus path differs")
    result_dir = _option_value(tokens, "--result-dir")
    result_filename = _option_value(tokens, "--result-filename")
    if result_dir is None or result_filename is None:
        raise HarnessError("trace client command omits its result path")
    if (pathlib.Path(result_dir) / result_filename).resolve() != result_path.resolve():
        raise HarnessError("trace client command result path differs")
    return tokens


def _parse_fp4_plan_log(path: pathlib.Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise HarnessError(f"profile log is absent: {path}") from error
    prepared = [match for line in lines if (match := _PLAN_PREPARED_RE.fullmatch(line))]
    complete = [match for line in lines if (match := _PLAN_COMPLETE_RE.fullmatch(line))]
    warmup = [match for line in lines if (match := _PLAN_WARMUP_RE.fullmatch(line))]
    selected = [
        (line, match)
        for line in lines
        if (match := _PLAN_SELECTED_RE.fullmatch(line))
    ]
    if len(prepared) != 1 or len(complete) != 1 or len(warmup) != 1:
        raise HarnessError("profile log does not contain one exact FP4 plan lifecycle")
    if len(selected) != 64:
        raise HarnessError(f"profile log contains {len(selected)} selected plans; expected 64")
    recognized = {
        line
        for line in lines
        if _PLAN_PREPARED_RE.fullmatch(line)
        or _PLAN_COMPLETE_RE.fullmatch(line)
        or _PLAN_SELECTED_RE.fullmatch(line)
        or _PLAN_WARMUP_RE.fullmatch(line)
    }
    unexpected = [
        line
        for line in lines
        if line.startswith("[VT_FP4_CACHE]")
        or line.startswith("[VT_FP4_AUTOTUNE]")
        if line not in recognized
    ]
    if unexpected:
        raise HarnessError(f"profile log has an unexpected FP4 lifecycle record: {unexpected[0]}")

    prepared_match = prepared[0]
    complete_match = complete[0]
    (
        mode,
        native_value,
        flashinfer_value,
        loaded,
        flashinfer_loaded,
        native_loaded,
        prepared_rejected,
        delay_us,
        prepared_metadata,
        prepared_selected,
    ) = prepared_match.groups()
    if (
        mode != "read-only"
        or int(loaded) != 64
        or int(flashinfer_loaded) != 64
        or int(native_loaded) != 0
        or int(prepared_rejected) != 0
        or int(delay_us) != 5000
        or int(prepared_selected) != 64
        or prepared_metadata != NVFP4_PLAN_METADATA
    ):
        raise HarnessError("profile log FP4 prepared semantics differ from H1d")
    (
        complete_mode,
        complete_loaded,
        tuned,
        complete_rejected,
        saved,
        complete_selected,
        complete_metadata,
    ) = complete_match.groups()
    if (
        complete_mode != "read-only"
        or int(complete_loaded) != 64
        or int(tuned) != 0
        or int(complete_rejected) != 0
        or int(saved) != 0
        or int(complete_selected) != 64
        or complete_metadata != prepared_metadata
    ):
        raise HarnessError("profile log FP4 completion semantics differ from H1d")
    max_tokens, profiles_requested, profiles_tuned, cached_plans = warmup[0].groups()
    if (
        int(max_tokens) != MAX_NUM_BATCHED_TOKENS["27"]
        or int(profiles_requested) != 0
        or int(profiles_tuned) != 0
        or int(cached_plans) != 64
    ):
        raise HarnessError("profile log FP4 pre-serve warmup semantics differ from H1d")

    flashinfer_path = pathlib.Path(flashinfer_value)
    if (
        not flashinfer_path.is_absolute()
        or not flashinfer_path.is_file()
        or sha256_file(flashinfer_path) != NVFP4_PLAN_FIXTURE_SHA256
    ):
        raise HarnessError("profile log FlashInfer plan fixture differs from H1d")
    native_path = pathlib.Path(native_value)
    if not native_path.is_absolute() or native_path.exists():
        raise HarnessError("profile log native plan target is not absent")

    selected_lines = [line for line, _ in selected]
    selected_sha = hashlib.sha256(
        "".join(f"{line}\n" for line in selected_lines).encode("utf-8")
    ).hexdigest()
    if selected_sha != NVFP4_SELECTED_PLAN_SHA256:
        raise HarnessError("profile log selected FP4 plan map differs from H1d")
    selected_keys = [tuple(int(value) for value in match.groups()[:3]) for _, match in selected]
    if len(set(selected_keys)) != 64:
        raise HarnessError("profile log selected FP4 plan keys are not unique")
    return {
        "flashinfer_path": str(flashinfer_path),
        "flashinfer_sha256": sha256_file(flashinfer_path),
        "loaded_flashinfer": 64,
        "loaded_native": 0,
        "metadata": NVFP4_PLAN_METADATA,
        "mode": "read-only",
        "native_path": str(native_path),
        "native_target_absent": True,
        "selected_count": 64,
        "selected_sha256": selected_sha,
        "tuned": 0,
    }


def _parse_profile_markers(path: pathlib.Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise HarnessError(f"profile log is absent: {path}") from error
    ready = [match for line in lines if (match := _CUDA_PROFILE_READY_RE.fullmatch(line))]
    started = [match for line in lines if (match := _CUDA_PROFILE_STARTED_RE.fullmatch(line))]
    stopped = [match for line in lines if (match := _CUDA_PROFILE_STOPPED_RE.fullmatch(line))]
    shutdown_ready = [
        match for line in lines if (match := _BENCH_SHUTDOWN_READY_RE.fullmatch(line))
    ]
    shutdown_requested = [
        match
        for line in lines
        if (match := _BENCH_SHUTDOWN_REQUESTED_RE.fullmatch(line))
    ]
    shutdown_completed = [
        match
        for line in lines
        if (match := _BENCH_SHUTDOWN_COMPLETED_RE.fullmatch(line))
    ]
    if len(ready) != 1 or len(started) != 1 or len(stopped) != 1:
        raise HarnessError("profile log does not contain one complete CUDA profile window")
    if (
        len(shutdown_ready) != 1
        or len(shutdown_requested) != 1
        or len(shutdown_completed) != 1
    ):
        raise HarnessError(
            "profile log does not contain one complete graceful-shutdown lifecycle"
        )
    ready_pid, ready_target = ready[0].groups()
    (shutdown_ready_pid,) = shutdown_ready[0].groups()
    start_target, start_graph, real_batch, padded_batch, prior_replays = started[0].groups()
    stopped_replays, stopped_graph = stopped[0].groups()
    if (
        int(ready_target) != TRACE_CAPTURE_GRAPH_REPLAYS
        or int(start_target) != TRACE_CAPTURE_GRAPH_REPLAYS
        or int(stopped_replays) != TRACE_CAPTURE_GRAPH_REPLAYS
        or int(real_batch) != TRACE_CONCURRENCY
        or int(padded_batch) != TRACE_CONCURRENCY
        or int(prior_replays) <= 0
        or stopped_graph != start_graph
        or shutdown_ready_pid != ready_pid
    ):
        raise HarnessError("CUDA profile markers differ from the eligible four-replay contract")
    return {
        "captured_replays": TRACE_CAPTURE_GRAPH_REPLAYS,
        "graph": start_graph,
        "padded_batch": TRACE_CONCURRENCY,
        "prior_replays": int(prior_replays),
        "ready_pid": int(ready_pid),
        "real_batch": TRACE_CONCURRENCY,
        "shutdown_completed": True,
        "shutdown_control": "fifo",
        "shutdown_ready_pid": int(shutdown_ready_pid),
        "shutdown_requested": True,
        "target_replays": TRACE_CAPTURE_GRAPH_REPLAYS,
    }


def record_profile_control(
    output: pathlib.Path,
    *,
    profile_log: pathlib.Path,
    nsys_pid: int,
    nsys_pgid: int,
    nsys_sid: int,
    nsys_exit_status: int,
    launcher_pid: int,
    launcher_ppid: int,
    launcher_pgid: int,
    launcher_sid: int,
    launcher_comm: str,
    server_pid: int,
    server_ppid: int,
    server_pgid: int,
    server_sid: int,
    shutdown_fifo: pathlib.Path,
) -> dict[str, Any]:
    if output.exists():
        raise HarnessError(f"refusing to overwrite profile control evidence: {output}")
    process_ids = (
        nsys_pid,
        nsys_pgid,
        nsys_sid,
        launcher_pid,
        launcher_ppid,
        launcher_pgid,
        launcher_sid,
        server_pid,
        server_ppid,
        server_pgid,
        server_sid,
    )
    if min(process_ids) <= 0:
        raise HarnessError("profile control PIDs must be positive")
    if nsys_exit_status != 0:
        raise HarnessError("Nsight profiler exit status is not zero")
    if len({nsys_pid, launcher_pid, server_pid}) != 3:
        raise HarnessError("profile control process identities are not distinct")
    if nsys_pgid != nsys_pid or nsys_sid != nsys_pid:
        raise HarnessError("Nsight profiler is not its setsid process-group leader")
    if launcher_comm != "nsys-launcher":
        raise HarnessError("profile target parent is not nsys-launcher")
    if (
        launcher_ppid != nsys_pid
        or launcher_pgid != nsys_pid
        or launcher_sid != nsys_pid
    ):
        raise HarnessError("nsys-launcher is not owned by the Nsight session")
    if server_ppid != launcher_pid:
        raise HarnessError("profiled server is not a direct nsys-launcher child")
    if server_pgid != server_pid or server_sid != server_pid:
        raise HarnessError("profiled server is not the Nsight target-session leader")
    if not shutdown_fifo.is_absolute() or shutdown_fifo.exists():
        raise HarnessError("profile shutdown FIFO is not an absent absolute path")
    markers = _parse_profile_markers(profile_log)
    if markers["ready_pid"] != server_pid:
        raise HarnessError("profile ready marker PID differs from the controlled server")
    result = {
        **markers,
        "launcher_comm": launcher_comm,
        "launcher_pgid": launcher_pgid,
        "launcher_pid": launcher_pid,
        "launcher_ppid": launcher_ppid,
        "launcher_sid": launcher_sid,
        "nsys_pid": nsys_pid,
        "nsys_pgid": nsys_pgid,
        "nsys_sid": nsys_sid,
        "nsys_exit_status": nsys_exit_status,
        "profile_log": str(profile_log),
        "profile_log_sha256": sha256_file(profile_log),
        "server_pgid": server_pgid,
        "server_pid": server_pid,
        "server_ppid": server_ppid,
        "server_sid": server_sid,
        "shutdown_fifo": str(shutdown_fifo),
        "signal": "SIGUSR2",
        "plan_validation": _parse_fp4_plan_log(profile_log),
    }
    write_json_atomic(output, result)
    return result


def _summarize_torch_trace(
    path: pathlib.Path, *, model_key: str | None = None
) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else pathlib.Path.open
    try:
        if path.suffix == ".gz":
            with opener(path, "rt", encoding="utf-8") as source:
                value = json.load(source)
        else:
            with opener(path, "r", encoding="utf-8") as source:
                value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"{path}: invalid torch-profiler trace: {error}") from error
    events = value.get("traceEvents") if isinstance(value, dict) else None
    if not isinstance(events, list):
        raise HarnessError(f"{path}: torch-profiler traceEvents array is absent")
    grouped: dict[str, list[float]] = {}
    kernel_events: list[dict[str, Any]] = []
    decode_windows: list[tuple[float, float, str]] = []
    generation_window_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", "")).lower()
        if category == "gpu_user_annotation" and model_key is not None:
            annotation_name = event.get("name")
            if isinstance(annotation_name, str) and re.fullmatch(
                r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)",
                annotation_name,
            ):
                generation_window_count += 1
            match = (
                re.fullmatch(
                    r"execute_context_0\(0\)_generation_(\d+)\((\d+)\)",
                    annotation_name,
                )
                if isinstance(annotation_name, str)
                else None
            )
            if match is not None:
                generation, repeated_generation = (int(value) for value in match.groups())
                if generation != repeated_generation or not 1 <= generation <= TRACE_CONCURRENCY:
                    raise HarnessError(
                        "vLLM pure-decode annotation has an invalid generation batch"
                    )
                start = require_number(event.get("ts"), f"{path}:decode annotation ts")
                duration = require_number(
                    event.get("dur"), f"{path}:decode annotation duration"
                )
                if event.get("ph") != "X" or start < 0.0 or duration <= 0.0:
                    raise HarnessError("vLLM pure-decode annotation is not a complete range")
                decode_windows.append((start, start + duration, annotation_name))
            continue
        if "kernel" not in category:
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            continue
        duration = require_number(event.get("dur"), f"{path}:kernel duration")
        if duration >= 0.0:
            grouped.setdefault(name, []).append(duration)
            if model_key is None:
                continue
            start = require_number(event.get("ts"), f"{path}:kernel timestamp")
            if start < 0.0:
                raise HarnessError("torch-profiler kernel timestamp is negative")
            kernel_events.append(
                {
                    "args": event.get("args"),
                    "duration": duration,
                    "end": start + duration,
                    "name": name,
                    "start": start,
                }
            )
    total = sum(sum(values) for values in grouped.values())
    if total <= 0.0 or not grouped:
        raise HarnessError("torch-profiler output contains no positive-duration kernels")
    kernels = sorted(
        (
            {
                "count": len(values),
                "name": name,
                "percent": sum(values) / total * 100.0,
                "total_duration_us": sum(values),
            }
            for name, values in grouped.items()
        ),
        key=lambda item: (-item["total_duration_us"], item["name"]),
    )
    result = {
        "kernel_count": sum(item["count"] for item in kernels),
        "kernel_time_us": total,
        "kernels": kernels,
        "selected_trace": str(path),
        "selected_trace_sha256": sha256_file(path),
    }
    if model_key is None:
        return result
    family_contract = VLLM_DECODE_FAMILY_CONTRACTS.get(model_key)
    if family_contract is None:
        raise HarnessError(f"no clean vLLM decode contract exists for model {model_key}")
    window_contract = VLLM_GENERATION_WINDOW_CONTRACTS.get(model_key)
    if window_contract is None:
        raise HarnessError(f"no vLLM generation-window contract exists for model {model_key}")
    if generation_window_count != window_contract["all"]:
        raise HarnessError(
            "vLLM generation annotation count differs: "
            f"got {generation_window_count}, expected {window_contract['all']}"
        )
    if len(decode_windows) != window_contract["clean"]:
        raise HarnessError(
            "vLLM clean decode annotation count differs: "
            f"got {len(decode_windows)}, expected {window_contract['clean']}"
        )
    if not decode_windows:
        raise HarnessError("vLLM trace contains no pure-decode GPU annotation windows")
    decode_windows.sort()
    for previous, current in zip(decode_windows, decode_windows[1:]):
        if current[0] < previous[1]:
            raise HarnessError("vLLM pure-decode GPU annotation windows overlap")

    def family_for(name: str) -> str | None:
        matches = [
            family
            for family, (needle, _) in family_contract.items()
            if needle in name
            and not (
                family == "normal_fp4_producer"
                and "silu_mul_cvt_fp16_to_fp4" in name
            )
        ]
        if len(matches) > 1:
            raise HarnessError(f"vLLM kernel matches multiple decode families: {name}")
        return matches[0] if matches else None

    kernel_events.sort(key=lambda event: (event["start"], event["end"], event["name"]))
    cursor = 0
    decode_grouped: dict[str, list[float]] = {}
    family_totals = Counter()
    family_signatures = Counter()
    for window_index, (window_start, window_end, _) in enumerate(
        decode_windows, start=1
    ):
        while cursor < len(kernel_events) and kernel_events[cursor]["end"] <= window_start:
            cursor += 1
        scan = cursor
        window_families = Counter()
        while scan < len(kernel_events) and kernel_events[scan]["start"] < window_end:
            event = kernel_events[scan]
            family = family_for(event["name"])
            if event["start"] >= window_start and event["end"] <= window_end:
                decode_grouped.setdefault(event["name"], []).append(event["duration"])
                if family is not None:
                    args = event["args"]
                    if not isinstance(args, dict):
                        raise HarnessError("vLLM decode kernel has no launch metadata")
                    signature = {
                        "block": args.get("block"),
                        "grid": args.get("grid"),
                        "name": event["name"],
                        "registers_per_thread": args.get("registers per thread"),
                        "shared_memory": args.get("shared memory"),
                    }
                    if (
                        not isinstance(signature["block"], list)
                        or len(signature["block"]) != 3
                        or any(not isinstance(value, int) for value in signature["block"])
                        or not isinstance(signature["grid"], list)
                        or len(signature["grid"]) != 3
                        or any(not isinstance(value, int) for value in signature["grid"])
                        or not isinstance(signature["registers_per_thread"], int)
                        or not isinstance(signature["shared_memory"], int)
                    ):
                        raise HarnessError(
                            "vLLM decode kernel launch geometry/resource metadata is incomplete"
                        )
                    window_families[family] += 1
                    family_totals[family] += 1
                    family_signatures[canonical_json(signature)] += 1
            elif family is not None:
                raise HarnessError(
                    "vLLM decode-family kernel crosses a pure-decode annotation boundary"
                )
            scan += 1
        expected_window_counts = {
            family: expected for family, (_, expected) in family_contract.items()
        }
        if dict(window_families) != expected_window_counts:
            raise HarnessError(
                "vLLM pure-decode family counts differ in window "
                f"{window_index}: got {dict(window_families)}, "
                f"expected {expected_window_counts}"
            )
        cursor = scan

    decode_total = sum(sum(values) for values in decode_grouped.values())
    decode_kernels = sorted(
        (
            {
                "count": len(values),
                "name": name,
                "percent": sum(values) / decode_total * 100.0,
                "total_duration_us": sum(values),
            }
            for name, values in decode_grouped.items()
        ),
        key=lambda item: (-item["total_duration_us"], item["name"]),
    )
    signature_multiset = [
        {"count": count, "signature": json.loads(signature)}
        for signature, count in sorted(family_signatures.items())
    ]
    result["clean_decode"] = {
        "annotation": "execute_context_0(0)_generation_N(N)",
        "family_counts": dict(sorted(family_totals.items())),
        "family_counts_per_window": {
            family: expected for family, (_, expected) in family_contract.items()
        },
        "family_signature_multiset_sha256": _sha256_canonical(signature_multiset),
        "kernel_count": sum(item["count"] for item in decode_kernels),
        "kernel_time_us": decode_total,
        "kernels": decode_kernels,
        "generation_window_count": generation_window_count,
        "window_count": len(decode_windows),
    }
    return result


def _validated_trace_execution(
    path: pathlib.Path, *, model_key: str, vllm_cpp_sha: str
) -> dict[str, Any]:
    execution = _load_json_object(path)
    if execution.get("model_key") != model_key:
        raise HarnessError("trace execution manifest model differs from H1d")
    if execution.get("vllm_cpp_sha") != vllm_cpp_sha:
        raise HarnessError("trace execution manifest source SHA differs from H1d")
    build_contract = execution.get("build_contract")
    if (
        not isinstance(build_contract, dict)
        or build_contract.get("schema_version") != BUILD_CONTRACT_SCHEMA_VERSION
        or build_contract.get("build_type") != "RelWithDebInfo"
        or build_contract.get("profile_control") is not True
        or build_contract.get("sm_architecture") != "121a"
        or build_contract.get("triton_aot") is not True
        or build_contract.get("target_compile_definitions")
        != [
            "VLLM_CPP_FLASH_ATTN",
            "VLLM_CPP_TRITON=1",
            "VLLM_CPP_TRITON_CHUNKO_BF16=1",
            "VT_BENCH_PROFILE_CONTROL=1",
            "VT_CUTLASS_NVFP4=1",
        ]
    ):
        raise HarnessError("trace execution manifest is not the exact H1d build")
    artifacts = execution.get("artifacts")
    if not isinstance(artifacts, dict):
        raise HarnessError("trace execution manifest artifact map is absent")
    required = {
        "cmake_cache",
        "model_config",
        "oracle:python",
        "server",
    }
    if not required.issubset(artifacts):
        raise HarnessError("trace execution manifest omits command-link artifacts")
    resolved = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise HarnessError(f"trace execution artifact {name} has no hash record")
        path_value = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(path_value, str) or not pathlib.Path(path_value).is_absolute():
            raise HarnessError(f"trace execution artifact {name} path is not absolute")
        artifact_path = pathlib.Path(path_value)
        if (
            not artifact_path.is_file()
            or not isinstance(expected_sha, str)
            or sha256_file(artifact_path) != expected_sha
        ):
            raise HarnessError(f"trace execution artifact {name} drifted")
        resolved[name] = artifact_path
    if not required.issubset(resolved):
        raise HarnessError("trace execution manifest omits command-link artifacts")
    if execution.get("vllm_source_sha") != VLLM_COMMIT:
        raise HarnessError("trace execution manifest vLLM source differs from H1d")
    if execution.get("vllm_oracle_version") != VLLM_ORACLE_VERSION:
        raise HarnessError("trace execution manifest oracle version differs from H1d")
    if execution.get("bench_dependencies") != {
        "flashinfer": FLASHINFER_VERSION,
        "pandas": PANDAS_VERSION,
    }:
        raise HarnessError("trace execution benchmark dependencies differ from H1d")
    if (
        execution.get("max_num_seqs") != MAX_NUM_SEQS
        or execution.get("max_num_batched_tokens")
        != MAX_NUM_BATCHED_TOKENS[model_key]
    ):
        raise HarnessError("trace execution scheduler contract differs from H1d")
    cutlass_tree = build_contract.get("cutlass_source_tree")
    if (
        not isinstance(cutlass_tree, dict)
        or not isinstance(cutlass_tree.get("path"), str)
        or _fingerprint_tree(pathlib.Path(cutlass_tree["path"])) != cutlass_tree
    ):
        raise HarnessError("trace execution CUTLASS source-tree fingerprint drifted")
    native_target = build_contract.get("native_plan_target")
    if (
        build_contract.get("native_plan_target_absent") is not True
        or not isinstance(native_target, str)
        or not pathlib.Path(native_target).is_absolute()
        or pathlib.Path(native_target).exists()
    ):
        raise HarnessError("trace execution native plan target is not absent")
    cache_values = {}
    for line in resolved["cmake_cache"].read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([^#/:]+):[^=]*=(.*)", line)
        if match:
            cache_values[match.group(1)] = match.group(2)
    source_value = cache_values.get("CMAKE_HOME_DIRECTORY")
    if not isinstance(source_value, str) or not pathlib.Path(source_value).is_dir():
        raise HarnessError("trace execution source root is absent")
    return {
        "execution": execution,
        "model_snapshot": resolved["model_config"].parent,
        "oracle_python": resolved["oracle:python"],
        "server": resolved["server"],
        "source_root": pathlib.Path(source_value),
    }


def record_trace_status(
    output: pathlib.Path,
    *,
    model_key: str,
    ours_nsys_reports: Sequence[pathlib.Path],
    ours_nsys_sqlites: Sequence[pathlib.Path],
    ours_nsys_validations: Sequence[pathlib.Path],
    ours_kernel_summaries: Sequence[pathlib.Path],
    ours_commands: Sequence[pathlib.Path],
    ours_profile_logs: Sequence[pathlib.Path],
    ours_profile_controls: Sequence[pathlib.Path],
    ours_client_results: Sequence[pathlib.Path],
    ours_client_logs: Sequence[pathlib.Path],
    ours_probe_results: Sequence[pathlib.Path],
    ours_probe_logs: Sequence[pathlib.Path],
    vllm_torch_trace: pathlib.Path,
    vllm_kernel_summary: pathlib.Path,
    vllm_command: pathlib.Path,
    vllm_profile_log: pathlib.Path,
    vllm_metadata: pathlib.Path,
    vllm_corpus: pathlib.Path,
    cache_drop_reports: Sequence[pathlib.Path],
    execution_manifest: pathlib.Path,
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
    trace_execution = _validated_trace_execution(
        execution_manifest, model_key=model_key, vllm_cpp_sha=vllm_cpp_sha
    )
    ours_trace_sequences = {
        "Nsight reports": ours_nsys_reports,
        "Nsight SQLite exports": ours_nsys_sqlites,
        "Nsight validations": ours_nsys_validations,
        "kernel summaries": ours_kernel_summaries,
        "profile commands": ours_commands,
        "profile logs": ours_profile_logs,
        "profile controls": ours_profile_controls,
        "semantic client results": ours_client_results,
        "semantic client logs": ours_client_logs,
        "diagnostic probe results": ours_probe_results,
        "diagnostic probe logs": ours_probe_logs,
    }
    for label, paths in ours_trace_sequences.items():
        if len(paths) != TRACE_REPETITIONS:
            raise HarnessError(
                f"ours trace must retain exactly three independent {label}"
            )
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
    for path in ours_probe_results:
        validate_raw_result(
            _load_json_object(path),
            concurrency=TRACE_CONCURRENCY,
            expected_requests=TRACE_CONCURRENCY,
        )

    command_environments = []
    command_shutdown_fifos = []
    for command_path, report_path in zip(
        ours_commands, ours_nsys_reports, strict=True
    ):
        command_tokens = shlex.split(command_path.read_text(encoding="utf-8"))
        try:
            nsys_index = command_tokens.index("nsys")
        except ValueError as error:
            raise HarnessError("ours trace command does not invoke nsys") from error
        if command_tokens[nsys_index : nsys_index + 2] != ["nsys", "profile"]:
            raise HarnessError("ours trace command does not invoke nsys profile")
        if command_tokens[:1] != ["env"]:
            raise HarnessError("ours trace command must use an explicit env prefix")
        environment_tokens = command_tokens[1:nsys_index]
        if any("=" not in token for token in environment_tokens):
            raise HarnessError("ours trace command env prefix is malformed")
        environment_names = [token.split("=", 1)[0] for token in environment_tokens]
        expected_environment_names = set(TRACE_REQUIRED_ENV) | {
            "VT_FP4_AUTOTUNE_CACHE_PATH",
            "VT_FP4_FLASHINFER_CACHE_PATH",
        }
        if (
            len(set(environment_names)) != len(environment_names)
            or set(environment_names) != expected_environment_names
        ):
            raise HarnessError("ours trace command env inventory differs from H1d")
        environment: dict[str, str] = {}
        for token in environment_tokens:
            name, value = token.split("=", 1)
            environment[name] = value
        for name, expected in TRACE_REQUIRED_ENV.items():
            if environment.get(name) != expected:
                raise HarnessError(f"ours trace environment {name} differs from H1d")
        for name in ("VT_FP4_FLASHINFER_CACHE_PATH", "VT_FP4_AUTOTUNE_CACHE_PATH"):
            if name not in environment:
                raise HarnessError(f"ours trace environment omits {name}")
        fixture = pathlib.Path(environment["VT_FP4_FLASHINFER_CACHE_PATH"])
        if (
            not fixture.is_absolute()
            or not fixture.is_file()
            or sha256_file(fixture) != NVFP4_PLAN_FIXTURE_SHA256
        ):
            raise HarnessError("ours trace environment plan fixture differs from H1d")
        native_target = pathlib.Path(environment["VT_FP4_AUTOTUNE_CACHE_PATH"])
        if not native_target.is_absolute() or native_target.exists():
            raise HarnessError("ours trace native plan target is not absent")
        command_environments.append(environment)
        output_prefix = _option_value(command_tokens, "--output")
        if output_prefix is None or pathlib.Path(
            f"{output_prefix}.nsys-rep"
        ).resolve() != report_path.resolve():
            raise HarnessError("ours trace command output prefix differs from its report")
        shutdown_fifo = pathlib.Path(f"{output_prefix}-shutdown.fifo").resolve()
        command_shutdown_fifos.append(shutdown_fifo)
        expected_tail = [
            "nsys",
            "profile",
            "--trace=cuda",
            f"--capture-range={NSYS_CAPTURE_RANGE}",
            "--capture-range-end=stop",
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
            output_prefix,
            str(trace_execution["server"]),
            "--model",
            str(trace_execution["model_snapshot"]),
            "--port",
            str(trace_execution["execution"]["port"]),
            "--num-blocks",
            str(trace_execution["execution"]["num_blocks"]),
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--max-num-batched-tokens",
            str(MAX_NUM_BATCHED_TOKENS[model_key]),
            "--max-model-len",
            str(MAX_MODEL_LEN[model_key]),
            "--no-enable-prefix-caching",
            "--cuda-profile-graph-replays",
            str(TRACE_CAPTURE_GRAPH_REPLAYS),
            "--benchmark-shutdown-fifo",
            str(shutdown_fifo),
            "--served-model-name",
            "gate",
        ]
        if command_tokens[nsys_index:] != expected_tail:
            raise HarnessError("ours trace command tail differs from the exact H1d recipe")

    for result_path, log_path in zip(
        ours_client_results, ours_client_logs, strict=True
    ):
        _parse_client_command_log(
            log_path,
            result_path=result_path,
            corpus_path=vllm_corpus,
            num_prompts=TRACE_PROMPTS,
            num_warmups=TRACE_CONCURRENCY,
        )
    for result_path, log_path in zip(ours_probe_results, ours_probe_logs, strict=True):
        _parse_client_command_log(
            log_path,
            result_path=result_path,
            corpus_path=vllm_corpus,
            num_prompts=TRACE_CONCURRENCY,
            num_warmups=0,
        )
    ours_output_digests = [
        hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
        for value in generated_texts
    ]
    if len(set(ours_output_digests)) != 1:
        raise HarnessError("ours semantic trace workloads are not output-repeatable")

    plan_validations = []
    profile_controls = []
    for index, (log_path, control_path, environment, shutdown_fifo) in enumerate(
        zip(
            ours_profile_logs,
            ours_profile_controls,
            command_environments,
            command_shutdown_fifos,
            strict=True,
        ),
        start=1,
    ):
        plan = _parse_fp4_plan_log(log_path)
        if pathlib.Path(plan["flashinfer_path"]).resolve() != pathlib.Path(
            environment["VT_FP4_FLASHINFER_CACHE_PATH"]
        ).resolve():
            raise HarnessError(f"capture {index} plan fixture differs from its command")
        if pathlib.Path(plan["native_path"]).resolve() != pathlib.Path(
            environment["VT_FP4_AUTOTUNE_CACHE_PATH"]
        ).resolve():
            raise HarnessError(f"capture {index} native plan target differs from its command")
        plan_validations.append(plan)
        markers = _parse_profile_markers(log_path)
        control = _load_json_object(control_path)
        expected_marker_fields = {
            **markers,
            "profile_log": str(log_path),
            "profile_log_sha256": sha256_file(log_path),
            "signal": "SIGUSR2",
        }
        if any(control.get(field) != value for field, value in expected_marker_fields.items()):
            raise HarnessError(f"capture {index} profile control differs from its log")
        for field in (
            "nsys_pid",
            "nsys_pgid",
            "nsys_sid",
            "launcher_pid",
            "launcher_ppid",
            "launcher_pgid",
            "launcher_sid",
            "server_pid",
            "server_ppid",
            "server_pgid",
            "server_sid",
        ):
            if not isinstance(control.get(field), int) or control[field] <= 0:
                raise HarnessError(f"capture {index} profile control {field} is invalid")
        if control.get("nsys_exit_status") != 0:
            raise HarnessError(f"capture {index} Nsight exit status is not zero")
        if control.get("launcher_comm") != "nsys-launcher":
            raise HarnessError(f"capture {index} parent is not nsys-launcher")
        if (
            control.get("shutdown_fifo") != str(shutdown_fifo)
            or shutdown_fifo.exists()
        ):
            raise HarnessError(f"capture {index} shutdown FIFO lifecycle differs")
        if (
            control["nsys_pgid"] != control["nsys_pid"]
            or control["nsys_sid"] != control["nsys_pid"]
            or control["launcher_ppid"] != control["nsys_pid"]
            or control["launcher_pgid"] != control["nsys_pid"]
            or control["launcher_sid"] != control["nsys_pid"]
            or control["server_ppid"] != control["launcher_pid"]
            or control["server_pgid"] != control["server_pid"]
            or control["server_sid"] != control["server_pid"]
        ):
            raise HarnessError(f"capture {index} does not have the recorded Nsight ancestry")
        if control["server_pid"] != markers["ready_pid"]:
            raise HarnessError(f"capture {index} controlled server PID differs")
        profile_controls.append(control)

    metadata = _load_json_object(vllm_metadata)
    expected_metadata = {
        "admission_mode": "closed-loop",
        "enable_prefix_caching": False,
        "input_len": INPUT_LEN,
        "max_concurrency": TRACE_CONCURRENCY,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS[model_key],
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN[model_key],
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
    if pathlib.Path(str(metadata.get("model"))).resolve() != trace_execution[
        "model_snapshot"
    ].resolve():
        raise HarnessError("vLLM trace metadata model snapshot differs")
    profile_dir_value = metadata.get("profile_dir")
    if not isinstance(profile_dir_value, str) or not pathlib.Path(
        profile_dir_value
    ).is_absolute():
        raise HarnessError("vLLM trace metadata profile directory is absent")
    vllm_profile_dir = pathlib.Path(profile_dir_value)
    try:
        vllm_torch_trace.resolve().relative_to(vllm_profile_dir.resolve())
    except ValueError as error:
        raise HarnessError("vLLM raw trace is outside its profile directory") from error
    output_digests = metadata.get("output_digests")
    if (
        not isinstance(output_digests, list)
        or len(output_digests) != TRACE_REPETITIONS + 1
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in output_digests
        )
    ):
        raise HarnessError("vLLM trace metadata output digests are absent")
    if metadata.get("output_digest") != output_digests[0]:
        raise HarnessError("vLLM trace warmup output digest differs")
    output_digests_equal = len(set(output_digests)) == 1
    if metadata.get("output_digests_equal") is not output_digests_equal:
        raise HarnessError("vLLM trace output-repeatability flag differs")

    vllm_tokens = shlex.split(vllm_command.read_text(encoding="utf-8"))
    if len(vllm_tokens) < 4 or vllm_tokens[0] != "env":
        raise HarnessError("vLLM trace command omits its explicit environment")
    if vllm_tokens[1].split("=", 1)[0] != "PATH" or "=" not in vllm_tokens[1]:
        raise HarnessError("vLLM trace command PATH prefix differs")
    path_value = vllm_tokens[1].split("=", 1)[1]
    if not path_value or pathlib.Path(path_value.split(os.pathsep)[0]).resolve() != (
        trace_execution["oracle_python"].parent.resolve()
    ):
        raise HarnessError("vLLM trace command PATH does not select the oracle environment")
    if pathlib.Path(vllm_tokens[2]).resolve() != trace_execution[
        "oracle_python"
    ].resolve():
        raise HarnessError("vLLM trace command Python differs from the oracle manifest")
    expected_profile_script = (
        trace_execution["source_root"]
        / "tools"
        / "bench"
        / "profile_vllm_online_gate.py"
    )
    if pathlib.Path(vllm_tokens[3]).resolve() != expected_profile_script.resolve():
        raise HarnessError("vLLM trace command script differs from the source manifest")
    expected_vllm_tokens = [
        "env",
        vllm_tokens[1],
        str(trace_execution["oracle_python"]),
        str(expected_profile_script),
        "--model",
        str(trace_execution["model_snapshot"]),
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
        str(MAX_NUM_BATCHED_TOKENS[model_key]),
        "--repetitions",
        str(TRACE_REPETITIONS),
    ]
    if vllm_tokens != expected_vllm_tokens:
        raise HarnessError("vLLM trace command differs from the exact H1d recipe")

    vllm_summary = _load_json_object(vllm_kernel_summary)
    recomputed_vllm_summary = _summarize_torch_trace(
        vllm_torch_trace, model_key=model_key
    )
    for field, expected in recomputed_vllm_summary.items():
        actual = vllm_summary.get(field)
        if field == "selected_trace":
            if not isinstance(actual, str) or pathlib.Path(actual).resolve() != pathlib.Path(
                expected
            ).resolve():
                raise HarnessError("vLLM selected trace path differs from its summary")
        elif actual != expected:
            raise HarnessError(f"vLLM kernel summary field {field} was not recomputed")

    nsys_validations = []
    nsys_kernel_summaries = []
    for report_path, sqlite_path, validation_path, summary_path in zip(
        ours_nsys_reports,
        ours_nsys_sqlites,
        ours_nsys_validations,
        ours_kernel_summaries,
        strict=True,
    ):
        validation = validate_nsys_trace(sqlite_path, model_key=model_key)
        if pathlib.Path(validation["nsys_report_path"]).resolve() != report_path.resolve():
            raise HarnessError("Nsight SQLite source report differs from its indexed artifact")
        if _load_json_object(validation_path) != validation:
            raise HarnessError(
                "recorded Nsight validation differs from the SQLite report"
            )
        nsys_validations.append(validation)
        summary = summarize_nsys_kernels(sqlite_path)
        if _load_json_object(summary_path) != summary:
            raise HarnessError("recorded Nsight kernel summary differs from its SQLite")
        nsys_kernel_summaries.append(summary)
    signature_hashes = [
        validation["canonical_node_multiset_sha256"] for validation in nsys_validations
    ]
    if len(set(signature_hashes)) != 1:
        raise HarnessError("ours graph node multiset differs across independent captures")
    capture_session_uuids = [
        validation["capture_session_uuid"] for validation in nsys_validations
    ]
    if len(set(capture_session_uuids)) != TRACE_REPETITIONS:
        raise HarnessError("ours independent traces reuse an Nsight capture session UUID")

    indexed_trace_artifacts: dict[str, pathlib.Path] = {}
    for name, paths in (
        ("ours_command", ours_commands),
        ("ours_profile_log", ours_profile_logs),
        ("ours_profile_control", ours_profile_controls),
        ("ours_nsys_report", ours_nsys_reports),
        ("ours_nsys_sqlite", ours_nsys_sqlites),
        ("ours_nsys_validation", ours_nsys_validations),
        ("ours_kernel_summary", ours_kernel_summaries),
    ):
        indexed_trace_artifacts[name] = paths[0]
        indexed_trace_artifacts.update(
            {f"{name}_{index}": path for index, path in enumerate(paths[1:], start=2)}
        )
    artifacts = {
        **indexed_trace_artifacts,
        **{
            f"ours_client_result_{index}": path
            for index, path in enumerate(ours_client_results, start=1)
        },
        **{
            f"ours_client_log_{index}": path
            for index, path in enumerate(ours_client_logs, start=1)
        },
        **{
            f"ours_probe_result_{index}": path
            for index, path in enumerate(ours_probe_results, start=1)
        },
        **{
            f"ours_probe_log_{index}": path
            for index, path in enumerate(ours_probe_logs, start=1)
        },
        "vllm_command": vllm_command,
        "vllm_profile_log": vllm_profile_log,
        "vllm_metadata": vllm_metadata,
        "vllm_corpus": vllm_corpus,
        "vllm_torch_trace": vllm_torch_trace,
        "vllm_kernel_summary": vllm_kernel_summary,
        "execution_manifest": execution_manifest,
    }
    for name, path in artifacts.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise HarnessError(f"trace artifact {name} is absent or empty: {path}")
    trace_paths = [
        path
        for paths in ours_trace_sequences.values()
        for path in paths
    ]
    resolved_paths = [path.resolve() for path in trace_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise HarnessError("independent trace artifacts reuse a path")
    inode_keys = [(path.stat().st_dev, path.stat().st_ino) for path in trace_paths]
    if len(set(inode_keys)) != len(inode_keys):
        raise HarnessError("independent trace artifacts reuse an inode")
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
        "output_repeatability": {
            "ours": {
                "all_equal": len(set(ours_output_digests)) == 1,
                "digests": ours_output_digests,
            },
            "vllm": {
                "all_equal": output_digests_equal,
                "digests": output_digests,
            },
        },
        "passed": True,
        "plan_validations": plan_validations,
        "profile_controls": profile_controls,
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
            "max_model_len": MAX_MODEL_LEN[model_key],
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
        "nsys_validations": nsys_validations,
        "nsys_kernel_summaries": nsys_kernel_summaries,
        "node_multiset_sha256": signature_hashes[0],
        "nsys_capture_session_uuids": capture_session_uuids,
        "nsys_product_version": NSYS_PRODUCT_VERSION,
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
    audited v0.25.0 target defines the client contract while the older porting
    parity pin remains explicit in the upstream-sync record until target
    goldens and behavior gates close.
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
        import flashinfer  # Delayed so CPU contract tests can provide a stub.
        flashinfer_distribution = importlib.metadata.distribution("flashinfer-python")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise HarnessError(
            f"pinned FlashInfer {FLASHINFER_VERSION} is required by the oracle"
        ) from error
    flashinfer_runtime_version = getattr(flashinfer, "__version__", None)
    if (
        flashinfer_distribution.version != FLASHINFER_VERSION
        or flashinfer_runtime_version != FLASHINFER_VERSION
    ):
        raise HarnessError(
            "FlashInfer oracle dependency drift: "
            f"metadata={flashinfer_distribution.version!r}, "
            f"runtime={flashinfer_runtime_version!r}, "
            f"expected={FLASHINFER_VERSION!r}"
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
    ninja = python.parent / "ninja"
    if not ninja.is_file() or not os.access(ninja, os.X_OK):
        raise HarnessError(
            "oracle ninja executable is absent from the pinned environment: "
            f"{ninja}"
        )
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
    flashinfer_init_value = getattr(flashinfer, "__file__", None)
    flashinfer_dist_info_value = getattr(flashinfer_distribution, "_path", None)
    if not isinstance(flashinfer_init_value, str) or flashinfer_dist_info_value is None:
        raise HarnessError("FlashInfer oracle dependency has no filesystem metadata")
    flashinfer_init = pathlib.Path(flashinfer_init_value).absolute()
    flashinfer_root = flashinfer_init.parent
    flashinfer_dist_info = pathlib.Path(flashinfer_dist_info_value).absolute()
    cutlass_tree = _fingerprint_tree(flashinfer_root / "data" / "cutlass")
    artifacts = {
        "bench_datasets": package_root / "benchmarks" / "datasets" / "datasets.py",
        "bench_serve": package_root / "benchmarks" / "serve.py",
        "cli_bench_serve": package_root / "entrypoints" / "cli" / "benchmark" / "serve.py",
        "client": client,
        "distribution_metadata": dist_info / "METADATA",
        "distribution_record": dist_info / "RECORD",
        "flashinfer_distribution_metadata": flashinfer_dist_info / "METADATA",
        "flashinfer_distribution_record": flashinfer_dist_info / "RECORD",
        "flashinfer_package_init": flashinfer_init,
        "ninja": ninja,
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
        "bench_dependencies": {
            "flashinfer": FLASHINFER_VERSION,
            "pandas": PANDAS_VERSION,
        },
        "client_contract_source_commit": VLLM_COMMIT,
        "cutlass_source_tree": cutlass_tree,
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
    configure_log: pathlib.Path,
    build_command: pathlib.Path,
    build_log: pathlib.Path,
    oracle_manifest: pathlib.Path,
    port: int,
    num_blocks: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    profile_control: bool,
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
        "configure_log": configure_log,
        "build_command": build_command,
        "build_log": build_log,
        "cmake_cache": build_dir / "CMakeCache.txt",
        "compile_commands": build_dir / "compile_commands.json",
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
    expected_bench_dependencies = {
        "flashinfer": FLASHINFER_VERSION,
        "pandas": PANDAS_VERSION,
    }
    if oracle.get("bench_dependencies") != expected_bench_dependencies:
        raise HarnessError("oracle benchmark dependency inventory differs from the pin")
    oracle_artifacts = oracle.get("artifacts")
    required_oracle_artifacts = {
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
    oracle_ninja = pathlib.Path(oracle_artifacts["ninja"]["path"])

    cutlass_source_tree = oracle.get("cutlass_source_tree")
    if not isinstance(cutlass_source_tree, dict):
        raise HarnessError("oracle CUTLASS source-tree fingerprint is absent")
    cutlass_path_value = cutlass_source_tree.get("path")
    if not isinstance(cutlass_path_value, str):
        raise HarnessError("oracle CUTLASS source-tree path is absent")
    cutlass_path = pathlib.Path(cutlass_path_value)
    if _fingerprint_tree(cutlass_path) != cutlass_source_tree:
        raise HarnessError("oracle CUTLASS source tree drifted before execution")

    cmake_cache_path = build_dir / "CMakeCache.txt"
    cache_values: dict[str, str] = {}
    for line in cmake_cache_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([^#/:]+):[^=]*=(.*)", line)
        if match:
            cache_values[match.group(1)] = match.group(2)
    expected_cache_values = {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_CUDA_COMPILER": str(DGX_CUDA_COMPILER),
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_MAKE_PROGRAM": str(oracle_ninja),
        "VLLM_CPP_BENCH_PROFILE_CONTROL": "ON" if profile_control else "OFF",
        "VLLM_CPP_BUILD_TESTS": "ON",
        "VLLM_CPP_CUDA": "ON",
        "VLLM_CPP_CUDA_ARCHITECTURES": "121a",
        "VLLM_CPP_FLASH_ATTN": "ON",
        "VLLM_CPP_SERVER": "ON",
        "VLLM_CPP_TRITON": "ON",
        "VLLM_CPP_TRITON_REGEN": "OFF",
    }
    for name, expected in expected_cache_values.items():
        if cache_values.get(name) != expected:
            raise HarnessError(f"CMake cache {name} differs from the H1d build contract")
    source_root_value = cache_values.get("CMAKE_HOME_DIRECTORY")
    configured_cutlass_value = cache_values.get("VLLM_CPP_CUTLASS_DIR")
    if not isinstance(source_root_value, str) or not pathlib.Path(source_root_value).is_dir():
        raise HarnessError("CMake cache source root is absent")
    source_root = pathlib.Path(source_root_value)
    if (
        not isinstance(configured_cutlass_value, str)
        or pathlib.Path(configured_cutlass_value).resolve() != cutlass_path.resolve()
    ):
        raise HarnessError("CMake CUTLASS path differs from the oracle dependency tree")

    configure_text = configure_log.read_text(encoding="utf-8")
    if (
        f"CUDA compiler identification is NVIDIA {DGX_CUDA_COMPILER_VERSION}"
        not in configure_text
        or "CUTLASS found at " not in configure_text
        or "enabling sm120a NVFP4 cutlass GEMM" not in configure_text
        or "cutlass NVFP4 GEMM disabled" in configure_text
    ):
        raise HarnessError(
            "configure log does not prove the pinned CUDA/CUTLASS NVFP4 toolchain"
        )

    artifacts["cuda_compiler"] = DGX_CUDA_COMPILER

    try:
        compile_entries = json.loads(
            (build_dir / "compile_commands.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise HarnessError("compile_commands.json is invalid") from error
    if not isinstance(compile_entries, list):
        raise HarnessError("compile_commands.json is not an array")
    target_source = (source_root / "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu").resolve()
    target_entries = [
        entry
        for entry in compile_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("file"), str)
        and pathlib.Path(entry["file"]).resolve() == target_source
    ]
    if len(target_entries) != 1:
        raise HarnessError("compile commands do not contain exactly one CUTLASS NVFP4 TU")
    target_entry = target_entries[0]
    command_value = target_entry.get("command")
    arguments_value = target_entry.get("arguments")
    if isinstance(command_value, str):
        compile_tokens = shlex.split(command_value)
    elif isinstance(arguments_value, list) and all(
        isinstance(value, str) for value in arguments_value
    ):
        compile_tokens = list(arguments_value)
    else:
        raise HarnessError("CUTLASS NVFP4 compile command has no token stream")
    compile_text = " ".join(compile_tokens)
    for required in (
        "-DVLLM_CPP_FLASH_ATTN",
        "-DVLLM_CPP_TRITON=1",
        "-DVLLM_CPP_TRITON_CHUNKO_BF16=1",
        "-DVT_CUTLASS_NVFP4=1",
        "arch=compute_121a",
        "sm_121a",
        str(cutlass_path / "include"),
        str(cutlass_path / "tools" / "util" / "include"),
    ):
        if required not in compile_text:
            raise HarnessError(f"CUTLASS NVFP4 compile command omits {required}")
    has_profile_definition = "-DVT_BENCH_PROFILE_CONTROL=1" in compile_tokens
    if has_profile_definition is not profile_control:
        raise HarnessError(
            "CUTLASS NVFP4 compile command profile-control definition differs"
        )

    server_path = build_dir / "examples" / "server"
    for marker in (b"MatmulNvfp4Cutlass", b"[VT_FP4_CACHE] prepared"):
        if not _file_contains(server_path, marker):
            raise HarnessError(f"server binary omits target marker {marker!r}")
    for marker in (
        b"[VT_CUDA_PROFILE] started",
        b"[VT_BENCH_SHUTDOWN] ready",
    ):
        has_profile_marker = _file_contains(server_path, marker)
        if has_profile_marker is not profile_control:
            raise HarnessError(
                f"server binary profile-control marker {marker!r} differs from its build"
            )

    fixture_path = source_root / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
    if (
        os.environ.get("VT_FP4_FLASHINFER_CACHE_PATH") is None
        or pathlib.Path(os.environ["VT_FP4_FLASHINFER_CACHE_PATH"]).resolve()
        != fixture_path.resolve()
        or sha256_file(fixture_path) != NVFP4_PLAN_FIXTURE_SHA256
    ):
        raise HarnessError("execution FlashInfer plan fixture differs from H1d")
    for name, expected in TRACE_REQUIRED_ENV.items():
        if os.environ.get(name) != expected:
            raise HarnessError(f"execution environment {name} differs from H1d")
    native_target_value = os.environ.get("VT_FP4_AUTOTUNE_CACHE_PATH")
    if native_target_value is None:
        raise HarnessError("execution environment omits native plan target")
    native_target = pathlib.Path(native_target_value)
    if not native_target.is_absolute() or native_target.exists():
        raise HarnessError("execution native plan target must be absent")

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
        "bench_dependencies": expected_bench_dependencies,
        "build_contract": {
            "schema_version": BUILD_CONTRACT_SCHEMA_VERSION,
            "build_type": "RelWithDebInfo",
            "compile_command_sha256": _sha256_canonical(compile_tokens),
            "cuda_compiler": str(DGX_CUDA_COMPILER),
            "cuda_compiler_sha256": sha256_file(DGX_CUDA_COMPILER),
            "cuda_compiler_version": DGX_CUDA_COMPILER_VERSION,
            "cutlass_source_tree": cutlass_source_tree,
            "native_plan_target": str(native_target),
            "native_plan_target_absent": True,
            "profile_control": profile_control,
            "sm_architecture": "121a",
            "triton_aot": True,
            "target_compile_definitions": (
                [
                    "VLLM_CPP_FLASH_ATTN",
                    "VLLM_CPP_TRITON=1",
                    "VLLM_CPP_TRITON_CHUNKO_BF16=1",
                    "VT_BENCH_PROFILE_CONTROL=1",
                    "VT_CUTLASS_NVFP4=1",
                ]
                if profile_control
                else [
                    "VLLM_CPP_FLASH_ATTN",
                    "VLLM_CPP_TRITON=1",
                    "VLLM_CPP_TRITON_CHUNKO_BF16=1",
                    "VT_CUTLASS_NVFP4=1",
                ]
            ),
        },
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

    validate_nsys = commands.add_parser("validate-nsys-trace")
    validate_nsys.add_argument("--sqlite", type=pathlib.Path, required=True)
    validate_nsys.add_argument(
        "--model-key", choices=tuple(MODEL_REVISIONS), required=True
    )
    validate_nsys.add_argument("--output", type=pathlib.Path, required=True)

    summarize_nsys = commands.add_parser("summarize-nsys-kernels")
    summarize_nsys.add_argument("--sqlite", type=pathlib.Path, required=True)
    summarize_nsys.add_argument("--output", type=pathlib.Path, required=True)

    profile_control = commands.add_parser("record-profile-control")
    profile_control.add_argument("--output", type=pathlib.Path, required=True)
    profile_control.add_argument("--profile-log", type=pathlib.Path, required=True)
    profile_control.add_argument("--nsys-pid", type=int, required=True)
    profile_control.add_argument("--nsys-pgid", type=int, required=True)
    profile_control.add_argument("--nsys-sid", type=int, required=True)
    profile_control.add_argument("--nsys-exit-status", type=int, required=True)
    profile_control.add_argument("--launcher-pid", type=int, required=True)
    profile_control.add_argument("--launcher-ppid", type=int, required=True)
    profile_control.add_argument("--launcher-pgid", type=int, required=True)
    profile_control.add_argument("--launcher-sid", type=int, required=True)
    profile_control.add_argument("--launcher-comm", required=True)
    profile_control.add_argument("--server-pid", type=int, required=True)
    profile_control.add_argument("--server-ppid", type=int, required=True)
    profile_control.add_argument("--server-pgid", type=int, required=True)
    profile_control.add_argument("--server-sid", type=int, required=True)
    profile_control.add_argument("--shutdown-fifo", type=pathlib.Path, required=True)

    trace = commands.add_parser("record-trace-status")
    trace.add_argument("--output", type=pathlib.Path, required=True)
    trace.add_argument("--model-key", choices=tuple(MODEL_REVISIONS), required=True)
    trace.add_argument(
        "--ours-nsys-report", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-nsys-sqlite", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-nsys-validation", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-kernel-summary", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-command", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-profile-log", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument(
        "--ours-profile-control", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument("--ours-client-result", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--ours-client-log", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--ours-probe-result", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--ours-probe-log", action="append", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-torch-trace", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-kernel-summary", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-command", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-profile-log", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-metadata", type=pathlib.Path, required=True)
    trace.add_argument("--vllm-corpus", type=pathlib.Path, required=True)
    trace.add_argument(
        "--cache-drop-report", action="append", type=pathlib.Path, required=True
    )
    trace.add_argument("--execution-manifest", type=pathlib.Path, required=True)
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
    execution.add_argument("--configure-log", type=pathlib.Path, required=True)
    execution.add_argument("--build-command", type=pathlib.Path, required=True)
    execution.add_argument("--build-log", type=pathlib.Path, required=True)
    execution.add_argument("--oracle-manifest", type=pathlib.Path, required=True)
    execution.add_argument("--port", type=int, required=True)
    execution.add_argument("--num-blocks", type=int, required=True)
    execution.add_argument("--max-num-seqs", type=int, required=True)
    execution.add_argument("--max-num-batched-tokens", type=int, required=True)
    execution.add_argument("--profile-control", choices=("on", "off"), required=True)

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
    bench.add_argument("--num-warmups", type=int)
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
    elif args.command == "validate-nsys-trace":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite Nsight validation: {args.output}")
        result = validate_nsys_trace(args.sqlite, model_key=args.model_key)
        write_json_atomic(args.output, result)
    elif args.command == "summarize-nsys-kernels":
        if args.output.exists():
            raise HarnessError(f"refusing to overwrite Nsight summary: {args.output}")
        result = summarize_nsys_kernels(args.sqlite)
        write_json_atomic(args.output, result)
    elif args.command == "record-profile-control":
        result = record_profile_control(
            args.output,
            profile_log=args.profile_log,
            nsys_pid=args.nsys_pid,
            nsys_pgid=args.nsys_pgid,
            nsys_sid=args.nsys_sid,
            nsys_exit_status=args.nsys_exit_status,
            launcher_pid=args.launcher_pid,
            launcher_ppid=args.launcher_ppid,
            launcher_pgid=args.launcher_pgid,
            launcher_sid=args.launcher_sid,
            launcher_comm=args.launcher_comm,
            server_pid=args.server_pid,
            server_ppid=args.server_ppid,
            server_pgid=args.server_pgid,
            server_sid=args.server_sid,
            shutdown_fifo=args.shutdown_fifo,
        )
    elif args.command == "record-trace-status":
        result = record_trace_status(
            args.output,
            model_key=args.model_key,
            ours_nsys_reports=args.ours_nsys_report,
            ours_nsys_sqlites=args.ours_nsys_sqlite,
            ours_nsys_validations=args.ours_nsys_validation,
            ours_kernel_summaries=args.ours_kernel_summary,
            ours_commands=args.ours_command,
            ours_profile_logs=args.ours_profile_log,
            ours_profile_controls=args.ours_profile_control,
            ours_client_results=args.ours_client_result,
            ours_client_logs=args.ours_client_log,
            ours_probe_results=args.ours_probe_result,
            ours_probe_logs=args.ours_probe_log,
            vllm_torch_trace=args.vllm_torch_trace,
            vllm_kernel_summary=args.vllm_kernel_summary,
            vllm_command=args.vllm_command,
            vllm_profile_log=args.vllm_profile_log,
            vllm_metadata=args.vllm_metadata,
            vllm_corpus=args.vllm_corpus,
            cache_drop_reports=args.cache_drop_report,
            execution_manifest=args.execution_manifest,
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
            configure_log=args.configure_log,
            build_command=args.build_command,
            build_log=args.build_log,
            oracle_manifest=args.oracle_manifest,
            port=args.port,
            num_blocks=args.num_blocks,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            profile_control=args.profile_control == "on",
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
                num_warmups_override=args.num_warmups,
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
