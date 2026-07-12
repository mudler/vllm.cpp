#!/usr/bin/env python3
"""Summarize ``SERVE-GATE-ONLINE`` without promoting void evidence.

Metric values are read from the detailed output produced by pinned vLLM
``bench serve`` (``vllm/benchmarks/serve.py:563-748,1188-1284`` at
``702f481``). The upstream client owns timing; this module validates the
complete grid, retains cross-engine output differences as diagnostics,
aggregates repeated runs, and applies the repository's direction-aware
every-axis rule. Correctness is a separate commit-bound model gate because
production FP4 accumulation variants may select different greedy near-ties.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    ENGINES,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MODEL_REVISIONS,
    OUTPUT_LEN,
    PANDAS_VERSION,
    POINTS,
    REPETITIONS,
    TRACE_CONCURRENCY,
    TRACE_PROMPTS,
    TRACE_REPETITIONS,
    VLLM_ORACLE_VERSION,
    precise_max_concurrent_requests,
    validate_raw_result,
)
from tools.bench.serve_low_common import (
    HarnessError,
    VLLM_COMMIT,
    coefficient_of_variation,
    read_jsonl,
    require_number,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)

HIGHER_AXES = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_token_throughput",
)
LOWER_AXES = tuple(
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "p90", "p99")
)
MEMORY_AXES = (
    "peak_pss_kib",
    "peak_rss_kib",
    "peak_gpu_memory_mib",
    "peak_mem_available_drop_kib",
)
RUN_RE = re.compile(r"c(?P<concurrency>1|2|4|8|16|32)-r(?P<repetition>[123])\.json$")


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HarnessError(f"missing evidence artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path}: expected a JSON object")
    return value


def _aggregate(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise HarnessError("cannot aggregate an empty axis")
    return {
        "coefficient_of_variation": coefficient_of_variation(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "range": max(values) - min(values),
    }


def _run_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    duration = require_number(record.get("duration"), "duration")
    metrics = {
        axis: require_number(record.get(axis), axis)
        for axis in (*HIGHER_AXES[0:1], *HIGHER_AXES[2:], *LOWER_AXES)
    }
    metrics["input_throughput"] = (
        require_number(record.get("total_input_tokens"), "total_input_tokens")
        / duration
    )
    return metrics


def _memory_for_leg(
    evidence_root: pathlib.Path,
    model: str,
    engine: str,
    repetition: int,
) -> tuple[dict[str, float | None], list[str]]:
    base = evidence_root / "memory" / model / engine
    samples_path = base / f"r{repetition}.samples.jsonl"
    summary_path = base / f"r{repetition}.summary.json"
    reasons: list[str] = []
    try:
        execution = _load_json(evidence_root / "execution" / f"{model}.json")
        expected_cache_roots = execution.get("cache_drop_roots")
        if not isinstance(expected_cache_roots, list):
            expected_cache_roots = []
            reasons.append("execution cache-drop root inventory is absent")
    except HarnessError as error:
        expected_cache_roots = []
        reasons.append(str(error))
    try:
        summary = _load_json(summary_path)
    except HarnessError as error:
        summary = {}
        reasons.append(str(error))
    try:
        samples = list(read_jsonl(samples_path))
    except (HarnessError, FileNotFoundError) as error:
        samples = []
        reasons.append(f"missing/invalid memory samples: {error}")

    memory: dict[str, float | None] = {}
    if len(samples) < 2:
        reasons.append("memory sample stream has fewer than two samples")
    if samples and samples[-1].get("alive") is not False:
        reasons.append("memory sample stream did not observe owned-process exit")
    if not any(
        sample.get("alive") is True
        and isinstance(sample.get("pids"), list)
        and bool(sample["pids"])
        for sample in samples
    ):
        reasons.append("memory sample stream never observed a live owned process")
    if summary.get("samples") != len(samples):
        reasons.append("memory summary sample count differs from the stream")
    for field in ("peak_pss_kib", "peak_rss_kib", "peak_mem_available_drop_kib"):
        try:
            memory[field] = require_number(summary.get(field), field)
        except HarnessError as error:
            memory[field] = None
            reasons.append(str(error))
    sample_fields = {
        "peak_pss_kib": "pss_kib",
        "peak_rss_kib": "rss_kib",
        "peak_mem_available_drop_kib": "peak_mem_available_drop_kib",
    }
    for summary_field, sample_field in sample_fields.items():
        values: list[float] = []
        for index, sample in enumerate(samples):
            try:
                values.append(
                    require_number(sample.get(sample_field), f"{sample_field}[{index}]")
                )
            except HarnessError as error:
                reasons.append(str(error))
        if values and memory[summary_field] != max(values):
            reasons.append(f"memory summary {summary_field} differs from sample peak")
    gpu_values: list[float] = []
    for index, sample in enumerate(samples):
        value = sample.get("gpu_memory_mib")
        if value is None:
            continue
        try:
            gpu_values.append(require_number(value, f"gpu_memory_mib[{index}]"))
        except HarnessError as error:
            reasons.append(str(error))
    memory["peak_gpu_memory_mib"] = max(gpu_values) if gpu_values else None
    if not gpu_values:
        reasons.append("numeric GPU-memory samples are absent")
    elif max(gpu_values) <= 0.0:
        reasons.append("GPU-memory samples never observed model residency")
    if not samples:
        reasons.append("memory sample stream is empty")

    command_path = (
        evidence_root / "logs" / model / engine / f"r{repetition}-server-command.txt"
    )
    server_log = evidence_root / "logs" / model / engine / f"r{repetition}-server.log"
    if not server_log.is_file() or server_log.stat().st_size == 0:
        reasons.append(f"server log is absent or empty: {server_log}")
    try:
        command_text = command_path.read_text(encoding="utf-8")
        if not command_text.strip():
            raise HarnessError(f"server command is empty: {command_path}")
        command = shlex.split(command_text)
        for flag, expected in (
            ("--max-num-seqs", str(MAX_NUM_SEQS)),
            ("--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS[model])),
            ("--served-model-name", "gate"),
        ):
            if flag not in command or command.index(flag) + 1 >= len(command):
                reasons.append(f"server command omits {flag}")
            elif command[command.index(flag) + 1] != expected:
                reasons.append(f"server command {flag} differs from the gate")
        if "--no-enable-prefix-caching" not in command:
            reasons.append("server command does not disable prefix caching")
    except (OSError, ValueError, HarnessError) as error:
        reasons.append(str(error))

    for suffix in ("before", "after"):
        path = (
            evidence_root
            / "thermal"
            / model
            / engine
            / f"r{repetition}-{suffix}.txt"
        )
        if not path.is_file() or path.stat().st_size == 0:
            reasons.append(f"thermal/power snapshot is absent: {path}")

    return_path = (
        evidence_root
        / "memory-return"
        / model
        / engine
        / f"r{repetition}.json"
    )
    try:
        returned = _load_json(return_path)
        for field in (
            "drop_caches_succeeded",
            "gpu_idle",
            "mem_available_within_tolerance",
            "returned",
        ):
            if returned.get(field) is not True:
                reasons.append(f"memory-return field {field} is not true")
        cache_drops = returned.get("cache_drops")
        if not isinstance(cache_drops, dict):
            reasons.append("memory-return cache-drop artifact map is absent")
        else:
            for phase in ("before", "after"):
                artifact = cache_drops.get(phase)
                if not isinstance(artifact, dict):
                    reasons.append(f"memory-return cache-drop {phase} is absent")
                    continue
                try:
                    _verify_cache_drop_artifact(
                        artifact,
                        evidence_root,
                        f"memory-return cache-drop {phase}",
                        expected_roots=expected_cache_roots,
                    )
                except HarnessError as error:
                    reasons.append(str(error))
    except HarnessError as error:
        reasons.append(str(error))
    return memory, reasons


def _artifact_path(path_value: Any, evidence_root: pathlib.Path, field: str) -> pathlib.Path:
    if not isinstance(path_value, str) or not path_value:
        raise HarnessError(f"{field} path is absent")
    path = pathlib.Path(path_value)
    return path if path.is_absolute() else evidence_root / path


def _verify_hashed_artifact(
    artifact: Mapping[str, Any], evidence_root: pathlib.Path, field: str
) -> None:
    path = _artifact_path(artifact.get("path"), evidence_root, field)
    expected = artifact.get("sha256")
    if not path.is_file() or path.stat().st_size == 0:
        raise HarnessError(f"{field} is absent or empty: {path}")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise HarnessError(f"{field} hash does not match its status record")


def _verify_cache_drop_artifact(
    artifact: Mapping[str, Any],
    evidence_root: pathlib.Path,
    field: str,
    *,
    expected_roots: Sequence[str],
) -> None:
    _verify_hashed_artifact(artifact, evidence_root, field)
    path = _artifact_path(artifact.get("path"), evidence_root, field)
    report = _load_json(path)
    if artifact.get("method") != CACHE_DROP_METHOD:
        raise HarnessError(f"{field} status method differs")
    if report.get("method") != CACHE_DROP_METHOD:
        raise HarnessError(f"{field} report method differs")
    if report.get("succeeded") is not True or report.get("resident_after_bytes") != 0:
        raise HarnessError(f"{field} did not prove zero resident pages")
    if artifact.get("roots") != list(expected_roots):
        raise HarnessError(f"{field} root inventory differs from execution")
    if report.get("roots") != list(expected_roots):
        raise HarnessError(f"{field} raw root inventory differs from execution")
    for metadata_field in (
        "file_count",
        "file_inventory_sha256",
        "logical_bytes",
    ):
        if artifact.get(metadata_field) != report.get(metadata_field):
            raise HarnessError(f"{field} {metadata_field} differs from its report")


def _model_precondition_reasons(
    evidence_root: pathlib.Path,
    model: str,
    vllm_cpp_sha: str | None,
) -> list[str]:
    reasons: list[str] = []
    expected_cache_roots: list[str] = []
    execution_path = evidence_root / "execution" / f"{model}.json"
    try:
        execution = _load_json(execution_path)
        if execution.get("model_key") != model:
            reasons.append("execution manifest model key differs")
        if execution.get("vllm_cpp_sha") != vllm_cpp_sha:
            reasons.append("execution manifest vllm.cpp SHA differs from the campaign")
        if execution.get("vllm_source_sha") != VLLM_COMMIT:
            reasons.append("execution manifest vLLM source SHA differs from the pin")
        if execution.get("vllm_oracle_version") != VLLM_ORACLE_VERSION:
            reasons.append("execution manifest pip-vLLM oracle version differs")
        if execution.get("bench_dependencies") != {"pandas": PANDAS_VERSION}:
            reasons.append("execution benchmark dependency inventory differs")
        if execution.get("max_num_seqs") != MAX_NUM_SEQS:
            reasons.append("execution manifest max-num-seqs differs from the gate")
        if execution.get("max_num_batched_tokens") != MAX_NUM_BATCHED_TOKENS[model]:
            reasons.append("execution manifest token budget differs from the gate")
        artifacts = execution.get("artifacts")
        if not isinstance(artifacts, dict):
            reasons.append("execution artifact map is absent")
        else:
            required = {
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
            }
            for field in sorted(required - set(artifacts)):
                reasons.append(f"execution artifact {field} is absent")
            weight_fields = {field for field in artifacts if field.startswith("weight:")}
            listed_weights = execution.get("weight_files")
            if not weight_fields or not isinstance(listed_weights, list):
                reasons.append("execution weight-file manifest is absent")
            elif weight_fields != {f"weight:{name}" for name in listed_weights}:
                reasons.append("execution weight-file inventory differs")
            snapshot_fields = {
                field for field in artifacts if field.startswith("snapshot:")
            }
            listed_snapshot_files = execution.get("snapshot_files")
            if not isinstance(listed_snapshot_files, list):
                reasons.append("execution support-file manifest is absent")
            elif snapshot_fields != {
                f"snapshot:{name}" for name in listed_snapshot_files
            }:
                reasons.append("execution support-file inventory differs")
            for field, artifact in sorted(artifacts.items()):
                if not isinstance(artifact, dict):
                    reasons.append(f"execution artifact {field} is absent")
                    continue
                try:
                    _verify_hashed_artifact(artifact, evidence_root, field)
                except HarnessError as error:
                    reasons.append(str(error))
            try:
                expected_cache_roots = [
                    str(pathlib.Path(artifacts["model_config"]["path"]).parent),
                    str((evidence_root / "corpus" / model).absolute()),
                    str(pathlib.Path(artifacts["server"]["path"])),
                    str(pathlib.Path(artifacts["client"]["path"])),
                ]
            except (KeyError, TypeError) as error:
                reasons.append(f"execution cache-drop roots cannot be derived: {error}")
            if execution.get("cache_drop_roots") != expected_cache_roots:
                reasons.append("execution cache-drop root inventory differs")
    except HarnessError as error:
        reasons.append(str(error))

    model_gate_path = evidence_root / "preflight" / "model-gate" / f"{model}.json"
    try:
        status = _load_json(model_gate_path)
        if status.get("passed") is not True:
            reasons.append("model-gate status is not passed")
        if status.get("model_key") != model:
            reasons.append("model-gate model key differs")
        if status.get("vllm_cpp_sha") != vllm_cpp_sha:
            reasons.append("model-gate vllm.cpp SHA differs from the campaign")
        log = _artifact_path(status.get("log"), evidence_root, "model-gate log")
        if not log.is_file() or log.stat().st_size == 0:
            reasons.append(f"model-gate log is absent or empty: {log}")
        elif sha256_file(log) != status.get("log_sha256"):
            reasons.append("model-gate log hash differs")
    except HarnessError as error:
        reasons.append(str(error))

    trace_path = evidence_root / "trace" / model / "status.json"
    try:
        status = _load_json(trace_path)
        if status.get("passed") is not True:
            reasons.append("paired trace status is not passed")
        if status.get("model_key") != model:
            reasons.append("paired trace model key differs")
        if status.get("vllm_cpp_sha") != vllm_cpp_sha:
            reasons.append("paired trace vllm.cpp SHA differs from the campaign")
        if status.get("ours_profiler") != "nsys":
            reasons.append("ours trace is not an nsys capture")
        if status.get("vllm_profiler") != "torch-profiler":
            reasons.append("vLLM trace is not the required torch-profiler fallback")
        if status.get("trace_contract") != {
            "admission_mode": "closed-loop",
            "concurrency": TRACE_CONCURRENCY,
            "enable_prefix_caching": False,
            "input_len": INPUT_LEN,
            "max_model_len": MAX_MODEL_LEN[model],
            "max_num_seqs": MAX_NUM_SEQS,
            "num_prompts": TRACE_PROMPTS,
            "output_len": OUTPUT_LEN,
            "repetitions": TRACE_REPETITIONS,
        }:
            reasons.append("paired trace token/concurrency contract differs")
        artifacts = status.get("artifacts")
        if not isinstance(artifacts, dict):
            reasons.append("paired trace artifact map is absent")
        else:
            for field in (
                "cache_drop_1",
                "cache_drop_2",
                "cache_drop_3",
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
                artifact = artifacts.get(field)
                if not isinstance(artifact, dict):
                    reasons.append(f"paired trace artifact {field} is absent")
                    continue
                try:
                    if field.startswith("cache_drop_"):
                        _verify_cache_drop_artifact(
                            artifact,
                            evidence_root,
                            field,
                            expected_roots=expected_cache_roots,
                        )
                    else:
                        _verify_hashed_artifact(artifact, evidence_root, field)
                except HarnessError as error:
                    reasons.append(str(error))
    except HarnessError as error:
        reasons.append(str(error))
    return reasons


def _stream_preflight_reasons(
    evidence_root: pathlib.Path,
    model: str,
    engine: str,
    repetition: int,
) -> list[str]:
    path = (
        evidence_root
        / "preflight"
        / model
        / engine
        / f"r{repetition}-stream.json"
    )
    try:
        value = _load_json(path)
        chunks = value.get("emitted_chunks")
        first = require_number(value.get("first_chunk_s"), "first_chunk_s")
        total = require_number(value.get("total_s"), "total_s")
        spread = require_number(value.get("spread_s", total - first), "spread_s")
        reasons = []
        if chunks != 128:
            reasons.append(f"stream preflight emitted {chunks!r} chunks instead of 128")
        if not 0.0 <= first < total:
            reasons.append("stream preflight first chunk did not precede completion")
        if spread < 0.05 or total - first < 0.05:
            reasons.append("stream preflight did not preserve measurable live spread")
        return reasons
    except HarnessError as error:
        return [str(error)]


def _corpus_reasons(evidence_root: pathlib.Path, model: str) -> list[str]:
    root = evidence_root / "corpus" / model
    manifest_path = root / "vllm" / "manifest.json"
    reasons: list[str] = []
    try:
        manifest = _load_json(manifest_path)
        if manifest.get("model_key") != model:
            reasons.append("corpus-view model key differs")
        if manifest.get("tokenizer_revision") != MODEL_REVISIONS[model]:
            reasons.append("corpus-view tokenizer revision differs")
        if manifest.get("vllm_commit") != VLLM_COMMIT:
            reasons.append("corpus-view vLLM commit differs from the pin")
        source_manifest = root / "manifest.json"
        if not source_manifest.is_file():
            reasons.append(f"source corpus manifest is absent: {source_manifest}")
        elif sha256_file(source_manifest) != manifest.get("source_manifest_sha256"):
            reasons.append("source corpus manifest hash differs")
        files = manifest.get("files")
        expected = {
            (concurrency, repetition): requests
            for concurrency, requests in POINTS
            for repetition in REPETITIONS
        }
        if not isinstance(files, list):
            reasons.append("corpus-view file inventory is absent")
            return reasons
        seen: set[tuple[int, int]] = set()
        for item in files:
            if not isinstance(item, dict):
                reasons.append("corpus-view file inventory contains a non-object")
                continue
            key = (item.get("concurrency"), item.get("repetition"))
            if key not in expected or key in seen:
                reasons.append(f"corpus-view has unexpected/duplicate partition {key}")
                continue
            seen.add(key)
            if item.get("requests") != expected[key]:
                reasons.append(f"corpus-view partition {key} request count differs")
            filename = item.get("file")
            if not isinstance(filename, str) or pathlib.Path(filename).name != filename:
                reasons.append(f"corpus-view partition {key} filename is invalid")
                continue
            target = root / "vllm" / filename
            source = root / filename
            if not target.is_file() or sha256_file(target) != item.get("sha256"):
                reasons.append(f"corpus-view partition {key} hash differs")
            if not source.is_file() or sha256_file(source) != item.get("source_sha256"):
                reasons.append(f"source corpus partition {key} hash differs")
        missing = sorted(set(expected) - seen)
        if missing:
            reasons.append(f"corpus-view partitions are missing: {missing}")
    except HarnessError as error:
        reasons.append(str(error))
    return reasons


def _select_models(models: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(MODEL_REVISIONS) if models is None else tuple(models)
    if not selected:
        raise HarnessError("model selection is empty")
    if len(set(selected)) != len(selected):
        raise HarnessError("model selection contains duplicates")
    unknown = sorted(set(selected) - set(MODEL_REVISIONS))
    if unknown:
        raise HarnessError(f"unknown model selection: {unknown}")
    return selected


def summarize_evidence(
    evidence_root: pathlib.Path,
    *,
    models: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_models = _select_models(models)
    grouped: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    raw_root = evidence_root / "raw"
    if not raw_root.is_dir():
        raise HarnessError(f"missing raw evidence directory: {raw_root}")

    for path in sorted(raw_root.glob("*/*/c*-r*.json")):
        match = RUN_RE.fullmatch(path.name)
        if match is None:
            continue
        relative = path.relative_to(raw_root)
        model, engine = relative.parts[:2]
        if model not in selected_models or engine not in ENGINES:
            continue
        concurrency = int(match.group("concurrency"))
        repetition = int(match.group("repetition"))
        record = _load_json(path)
        reasons: list[str] = []
        client_log = (
            evidence_root
            / "logs"
            / model
            / engine
            / f"c{concurrency}-r{repetition}.log"
        )
        if not client_log.is_file() or client_log.stat().st_size == 0:
            reasons.append(f"timed client log is absent or empty: {client_log}")
        try:
            validate_raw_result(record, concurrency=concurrency)
            metrics = _run_metrics(record)
        except HarnessError as error:
            reasons.append(str(error))
            metrics = {}
        grouped[(model, engine, concurrency)].append(
            (
                repetition,
                {
                    "binding_eligible": not reasons,
                    "concurrency": concurrency,
                    "engine": engine,
                    "file": str(path.relative_to(evidence_root)),
                    "generated_texts": record.get("generated_texts"),
                    "metrics": metrics,
                    "model": model,
                    "precise_max_concurrent_requests": (
                        precise_max_concurrent_requests(record)
                        if not reasons
                        else None
                    ),
                    "reasons": reasons,
                    "repetition": repetition,
                    "upstream_bucketed_max_concurrent_requests": record.get(
                        "max_concurrent_requests"
                    ),
                },
            )
        )

    campaign_reasons: list[str] = []
    vllm_cpp_sha: str | None = None
    try:
        manifest = _load_json(evidence_root / "manifest.json")
        value = manifest.get("vllm_cpp_sha")
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            campaign_reasons.append("campaign manifest has no full vllm.cpp SHA")
        else:
            vllm_cpp_sha = value
        if manifest.get("client_contract_source_commit") != VLLM_COMMIT:
            campaign_reasons.append("campaign manifest client source commit differs from the pin")
        if manifest.get("vllm_oracle_version") != VLLM_ORACLE_VERSION:
            campaign_reasons.append("campaign manifest pip-vLLM oracle version differs")
        if manifest.get("vllm_oracle_bench_dependencies") != {"pandas": PANDAS_VERSION}:
            campaign_reasons.append("campaign manifest benchmark dependencies differ")
        if manifest.get("gpu_lock_acquisitions_planned") != 2:
            campaign_reasons.append("campaign manifest does not plan one lock per model")
    except HarnessError as error:
        campaign_reasons.append(str(error))

    expected_groups = {
        (model, engine, concurrency)
        for model in selected_models
        for engine in ENGINES
        for concurrency, _ in POINTS
    }
    missing_groups = sorted(expected_groups - set(grouped))
    campaign_reasons.extend(
        "missing result group " + "/".join((model, engine, f"c{concurrency}"))
        for model, engine, concurrency in missing_groups
    )

    model_reasons = {
        model: (
            _corpus_reasons(evidence_root, model)
            + _model_precondition_reasons(evidence_root, model, vllm_cpp_sha)
        )
        for model in selected_models
    }
    stream_reasons = {
        (model, engine, repetition): _stream_preflight_reasons(
            evidence_root, model, engine, repetition
        )
        for model in selected_models
        for engine in ENGINES
        for repetition in REPETITIONS
    }

    # A paired result must exist, but generated text is diagnostic rather than
    # a performance precondition. Production FP4 accumulation variants can
    # select different greedy near-ties (including across vLLM repetitions),
    # while the commit-bound model gate plus exact native counts own correctness.
    pair_reasons: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    output_text_diagnostics: list[dict[str, Any]] = []
    for model in selected_models:
        for concurrency, _ in POINTS:
            per_engine = {
                engine: {rep: run for rep, run in grouped.get((model, engine, concurrency), [])}
                for engine in ENGINES
            }
            for repetition in REPETITIONS:
                ours = per_engine["ours"].get(repetition)
                vllm = per_engine["vllm"].get(repetition)
                if ours is None or vllm is None:
                    pair_reasons[(model, concurrency, repetition)].append(
                        "paired engine result is absent"
                    )
                    continue
                ours_texts = ours["generated_texts"]
                vllm_texts = vllm["generated_texts"]
                exact_matches = sum(
                    ours_text == vllm_text
                    for ours_text, vllm_text in zip(ours_texts, vllm_texts)
                )
                output_text_diagnostics.append(
                    {
                        "all_equal": ours_texts == vllm_texts,
                        "concurrency": concurrency,
                        "exact_matches": exact_matches,
                        "model": model,
                        "repetition": repetition,
                        "requests": len(ours_texts),
                    }
                )

    leg_memory: dict[tuple[str, str, int], dict[str, float | None]] = {}
    leg_reasons: dict[tuple[str, str, int], list[str]] = {}
    for model in selected_models:
        for engine in ENGINES:
            for repetition in REPETITIONS:
                key = (model, engine, repetition)
                leg_memory[key], leg_reasons[key] = _memory_for_leg(
                    evidence_root, model, engine, repetition
                )

    raw_runs: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    aggregate_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key in sorted(expected_groups):
        entries = sorted(grouped.get(key, []))
        repetitions = [repetition for repetition, _ in entries]
        runs = [run for _, run in entries]
        reasons: list[str] = []
        reasons.extend(model_reasons[key[0]])
        if repetitions != list(REPETITIONS):
            reasons.append(f"repetitions are {repetitions}; expected {list(REPETITIONS)}")
        for run in runs:
            run_pair_reasons = pair_reasons[(key[0], key[2], run["repetition"])]
            run["reasons"].extend(run_pair_reasons)
            run["reasons"].extend(
                stream_reasons[(key[0], key[1], run["repetition"])]
            )
            run["binding_eligible"] = not run["reasons"]
            reasons.extend(run["reasons"])
            reasons.extend(leg_reasons[(key[0], key[1], run["repetition"])] )
            raw_runs.append(run)
        axes: dict[str, dict[str, float] | None] = {}
        for axis in (*HIGHER_AXES, *LOWER_AXES):
            values = [run["metrics"].get(axis) for run in runs]
            axes[axis] = (
                _aggregate([float(value) for value in values])
                if len(values) == len(REPETITIONS)
                and all(value is not None for value in values)
                else None
            )
            if axes[axis] is None:
                reasons.append(f"axis {axis} lacks all three valid values")
        aggregate = {
            "binding_eligible": not reasons,
            "concurrency": key[2],
            "engine": key[1],
            "metrics": axes,
            "model": key[0],
            "reasons": sorted(set(reasons)),
            "repetitions": len(runs),
        }
        aggregates.append(aggregate)
        aggregate_index[key] = aggregate

    memory_aggregates: list[dict[str, Any]] = []
    memory_index: dict[tuple[str, str], dict[str, Any]] = {}
    for model in selected_models:
        for engine in ENGINES:
            reasons = [
                reason
                for repetition in REPETITIONS
                for reason in leg_reasons[(model, engine, repetition)]
            ]
            axes: dict[str, dict[str, float] | None] = {}
            for axis in MEMORY_AXES:
                values = [leg_memory[(model, engine, rep)].get(axis) for rep in REPETITIONS]
                axes[axis] = (
                    _aggregate([float(value) for value in values])
                    if all(value is not None for value in values)
                    else None
                )
                if axes[axis] is None:
                    reasons.append(f"memory axis {axis} lacks all three valid values")
            aggregate = {
                "binding_eligible": not reasons,
                "engine": engine,
                "metrics": axes,
                "model": model,
                "reasons": sorted(set(reasons)),
                "repetitions": len(REPETITIONS),
            }
            memory_aggregates.append(aggregate)
            memory_index[(model, engine)] = aggregate

    ratios: list[dict[str, Any]] = []
    for model in selected_models:
        for concurrency, _ in POINTS:
            ours = aggregate_index[(model, "ours", concurrency)]
            floor = aggregate_index[(model, "vllm", concurrency)]
            for axis in (*HIGHER_AXES, *LOWER_AXES):
                ours_axis = ours["metrics"][axis]
                floor_axis = floor["metrics"][axis]
                higher = axis in HIGHER_AXES
                ours_value = ours_axis["median"] if ours_axis else None
                floor_value = floor_axis["median"] if floor_axis else None
                normalized = None
                if ours_value is not None and floor_value is not None and ours_value and floor_value:
                    normalized = (
                        ours_value / floor_value if higher else floor_value / ours_value
                    )
                ratios.append(
                    {
                        "axis": axis,
                        "binding_eligible": (
                            ours["binding_eligible"]
                            and floor["binding_eligible"]
                            and normalized is not None
                        ),
                        "concurrency": concurrency,
                        "direction": "higher" if higher else "lower",
                        "model": model,
                        "ours": ours_value,
                        "pass": normalized is not None and normalized >= 1.0,
                        "pass_normalized_ratio": normalized,
                        "vllm": floor_value,
                    }
                )
        ours_memory = memory_index[(model, "ours")]
        floor_memory = memory_index[(model, "vllm")]
        for axis in MEMORY_AXES:
            ours_axis = ours_memory["metrics"][axis]
            floor_axis = floor_memory["metrics"][axis]
            ours_value = ours_axis["median"] if ours_axis else None
            floor_value = floor_axis["median"] if floor_axis else None
            normalized = (
                floor_value / ours_value
                if ours_value is not None and floor_value is not None and ours_value and floor_value
                else None
            )
            ratios.append(
                {
                    "axis": axis,
                    "binding_eligible": (
                        ours_memory["binding_eligible"]
                        and floor_memory["binding_eligible"]
                        and normalized is not None
                    ),
                    "concurrency": None,
                    "direction": "lower",
                    "model": model,
                    "ours": ours_value,
                    "pass": normalized is not None and normalized >= 1.0,
                    "pass_normalized_ratio": normalized,
                    "vllm": floor_value,
                }
            )

    eligible_ratios = [ratio for ratio in ratios if ratio["binding_eligible"]]
    gate_pass = (
        not campaign_reasons
        and len(eligible_ratios) == len(ratios)
        and all(ratio["pass"] for ratio in eligible_ratios)
    )
    runs_document = {
        "aggregates": aggregates,
        "campaign_reasons": campaign_reasons,
        "gate_pass": gate_pass,
        "memory_aggregates": memory_aggregates,
        "models": list(selected_models),
        "output_text_diagnostics": output_text_diagnostics,
        "raw_runs": raw_runs,
    }
    ratios_document = {
        "gate_pass": gate_pass,
        "models": list(selected_models),
        "ratios": ratios,
    }
    return runs_document, ratios_document


def _report(runs: Mapping[str, Any], ratios: Mapping[str, Any]) -> str:
    lines = ["# CUDA online serving gate summary", ""]
    lines.append(f"Models: {', '.join(runs['models'])}.")
    lines.append("")
    lines.append(f"Gate pass: **{'YES' if runs['gate_pass'] else 'NO'}**.")
    lines.append("")
    eligible = sum(item["binding_eligible"] for item in runs["aggregates"])
    lines.append(
        f"Binding-eligible performance groups: {eligible}/{len(runs['aggregates'])}."
    )
    lines.append("")
    for reason in runs["campaign_reasons"]:
        lines.append(f"- {reason}")
    failed = [
        ratio
        for ratio in ratios["ratios"]
        if not ratio["binding_eligible"] or not ratio["pass"]
    ]
    lines.append(f"Every-axis ratios failing or void: {len(failed)}/{len(ratios['ratios'])}.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_REVISIONS))
    args = parser.parse_args()
    models = (args.model,) if args.model is not None else None
    runs, ratios = summarize_evidence(args.evidence, models=models)
    output = args.evidence / (f"summary-{args.model}" if args.model else "summary")
    if output.exists() and any(output.iterdir()):
        raise HarnessError(f"refusing to overwrite summary evidence in {output}")
    write_json_atomic(output / "all-runs.json", runs)
    write_json_atomic(output / "ratios.json", ratios)
    write_text_atomic(output / "report.md", _report(runs, ratios))
    print(json.dumps({"gate_pass": runs["gate_pass"]}, sort_keys=True))
    return 0 if runs["gate_pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"online-gate-summary: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
