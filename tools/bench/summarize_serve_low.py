#!/usr/bin/env python3
"""Summarize SGLang low-concurrency raw JSONL without hiding void runs.

Metric formulas mirror pinned SGLang ``calculate_metrics``
(``python/sglang/bench_serving.py:968-1140``).  TTFT and ITL percentiles are
recomputed from ``--output-details`` arrays.  The pinned client does not emit
per-request E2E latency arrays, so E2E/TPOT raw-percentile completeness is
reported explicitly and binding promotion remains false until that evidence is
available; aggregate fields are never misrepresented as raw samples.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.run_serve_low import validate_raw_result
from tools.bench.serve_low_common import (
    HarnessError,
    coefficient_of_variation,
    percentile,
    read_jsonl,
    require_number,
    write_json_atomic,
    write_text_atomic,
)

HIGHER_AXES = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
)
LOWER_AXES = (
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
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
    "p95_itl_ms",
    "p99_itl_ms",
    "max_itl_ms",
)
RUN_RE = re.compile(r"c(?P<concurrency>1|2|4|8|16)-r(?P<repetition>[123])\.jsonl$")
PREFLIGHT_FIELDS = (
    "checkpoint_manifest",
    "incremental_streaming",
    "native_output_ids",
    "openai_usage",
    "quantization_path",
    "tokenizer_ids",
)


def _milliseconds(values: Sequence[Any], field: str) -> list[float]:
    return [require_number(value, field) * 1000.0 for value in values]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise HarnessError("cannot calculate a mean of an empty sequence")
    return sum(values) / len(values)


def summarize_run(
    record: Mapping[str, Any],
    *,
    expected_requests: int = 80,
    prompt_len: int = 1024,
    output_len: int = 128,
) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        validate_raw_result(
            record,
            expected_requests=expected_requests,
            prompt_len=prompt_len,
            output_len=output_len,
        )
    except HarnessError as error:
        reasons.append(str(error))

    duration = require_number(record.get("duration"), "duration")
    if duration <= 0.0:
        raise HarnessError("duration must be positive")
    input_lens = record.get("input_lens") if isinstance(record.get("input_lens"), list) else []
    output_lens = record.get("output_lens") if isinstance(record.get("output_lens"), list) else []
    ttfts = _milliseconds(record.get("ttfts", []), "ttfts")
    nested_itls = record.get("itls", [])
    if not isinstance(nested_itls, list) or any(not isinstance(row, list) for row in nested_itls):
        nested_itls = []
        reasons.append("ITL details are not a list of lists")
    itls = _milliseconds([value for row in nested_itls for value in row], "itls")
    total_input = sum(value for value in input_lens if isinstance(value, int))
    total_output = sum(value for value in output_lens if isinstance(value, int))

    metrics: dict[str, float | None] = {
        "request_throughput": expected_requests / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "total_throughput": (total_input + total_output) / duration,
    }
    if ttfts:
        metrics.update(
            mean_ttft_ms=_mean(ttfts),
            median_ttft_ms=statistics.median(ttfts),
            p90_ttft_ms=percentile(ttfts, 90),
            p99_ttft_ms=percentile(ttfts, 99),
        )
    else:
        for name in ("mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms"):
            metrics[name] = None
        reasons.append("raw TTFT samples are absent")
    if itls:
        metrics.update(
            mean_itl_ms=_mean(itls),
            median_itl_ms=statistics.median(itls),
            p90_itl_ms=percentile(itls, 90),
            p95_itl_ms=percentile(itls, 95),
            p99_itl_ms=percentile(itls, 99),
            max_itl_ms=max(itls),
        )
    else:
        for name in (
            "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p95_itl_ms",
            "p99_itl_ms", "max_itl_ms",
        ):
            metrics[name] = None
        reasons.append("raw ITL samples are absent")

    latencies = record.get("latencies")
    raw_e2e = isinstance(latencies, list) and len(latencies) == expected_requests
    if raw_e2e:
        e2e = _milliseconds(latencies, "latencies")
        tpots = [
            (latency - ttft) / (length - 1)
            for latency, ttft, length in zip(e2e, ttfts, output_lens)
            if length > 1
        ]
        metrics.update(
            mean_e2e_latency_ms=_mean(e2e),
            median_e2e_latency_ms=statistics.median(e2e),
            p90_e2e_latency_ms=percentile(e2e, 90),
            p99_e2e_latency_ms=percentile(e2e, 99),
            mean_tpot_ms=_mean(tpots),
            median_tpot_ms=statistics.median(tpots),
            p90_tpot_ms=percentile(tpots, 90),
            p99_tpot_ms=percentile(tpots, 99),
        )
    else:
        aggregate_names = (
            "mean_e2e_latency_ms", "median_e2e_latency_ms",
            "p90_e2e_latency_ms", "p99_e2e_latency_ms", "mean_tpot_ms",
            "median_tpot_ms", "p99_tpot_ms",
        )
        for name in aggregate_names:
            metrics[name] = require_number(record.get(name), name)
        metrics["p90_tpot_ms"] = None
        reasons.append(
            "pinned client output lacks per-request latencies; E2E/TPOT "
            "aggregates are retained but raw p90 TPOT cannot be reconstructed"
        )

    return {
        "binding_eligible": not reasons,
        "duration_s": duration,
        "metrics": metrics,
        "raw_e2e_latency_samples": raw_e2e,
        "reasons": reasons,
    }


def _aggregate_axis(values: Sequence[float]) -> dict[str, float]:
    return {
        "coefficient_of_variation": coefficient_of_variation(values),
        "max": max(values),
        "mean": _mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "range": max(values) - min(values),
    }


def _load_preflight_status(evidence_root: pathlib.Path) -> dict[str, Any] | None:
    path = evidence_root / "preflight" / "status.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HarnessError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise HarnessError(f"{path}: expected a JSON object")
    return value


def _preflight_reasons(
    status: Mapping[str, Any] | None, model: str, engine: str
) -> list[str]:
    if status is None:
        return ["preflight/status.json is absent"]
    models = status.get("models")
    if not isinstance(models, dict):
        return ["preflight status has no models object"]
    model_status = models.get(model)
    if not isinstance(model_status, dict):
        return [f"preflight status has no model {model}"]
    engine_status = model_status.get(engine)
    if not isinstance(engine_status, dict):
        return [f"preflight status has no {model}/{engine} arm"]
    reasons = [
        f"preflight {field} is not satisfied"
        for field in PREFLIGHT_FIELDS
        if engine_status.get(field) is not True
    ]
    explicit = engine_status.get("reasons", [])
    if isinstance(explicit, list):
        reasons.extend(str(reason) for reason in explicit if reason)
    return reasons


def summarize_evidence(
    evidence_root: pathlib.Path, *, require_full_grid: bool = True
) -> tuple[dict, dict]:
    groups: dict[tuple[str, str, int], list[tuple[int, dict]]] = defaultdict(list)
    raw_root = evidence_root / "raw"
    if not raw_root.is_dir():
        raise HarnessError(f"missing raw evidence directory: {raw_root}")
    preflight_status = _load_preflight_status(evidence_root)
    for path in sorted(raw_root.glob("*/*/c*-r*.jsonl")):
        match = RUN_RE.fullmatch(path.name)
        if match is None:
            continue
        relative = path.relative_to(raw_root)
        model, engine = relative.parts[:2]
        records = list(read_jsonl(path))
        if len(records) != 1:
            raise HarnessError(f"{path}: expected exactly one JSONL record")
        run = summarize_run(records[0])
        run.update(
            engine=engine,
            file=str(path.relative_to(evidence_root)),
            model=model,
            concurrency=int(match.group("concurrency")),
            repetition=int(match.group("repetition")),
        )
        groups[(model, engine, run["concurrency"])].append((run["repetition"], run))

    all_runs: list[dict] = []
    aggregates: dict[tuple[str, str, int], dict] = {}
    for key, entries in sorted(groups.items()):
        entries.sort()
        runs = [run for _, run in entries]
        all_runs.extend(runs)
        reasons = [reason for run in runs for reason in run["reasons"]]
        reasons.extend(_preflight_reasons(preflight_status, key[0], key[1]))
        if [repetition for repetition, _ in entries] != [1, 2, 3]:
            reasons.append("canonical repetitions 1,2,3 are not all present")
        axes: dict[str, dict[str, float] | None] = {}
        for axis in (*HIGHER_AXES, *LOWER_AXES):
            values = [run["metrics"].get(axis) for run in runs]
            axes[axis] = (
                _aggregate_axis([float(value) for value in values])
                if len(values) == 3 and all(value is not None for value in values)
                else None
            )
        aggregates[key] = {
            "binding_eligible": not reasons,
            "engine": key[1],
            "concurrency": key[2],
            "metrics": axes,
            "model": key[0],
            "reasons": sorted(set(reasons)),
            "repetitions": len(runs),
        }

    if require_full_grid:
        required_arms = {"ours", "sglang", "vllm"}
        required_concurrencies = {1, 2, 4, 8, 16}
        for model in {key[0] for key in aggregates}:
            present = {
                (engine, concurrency)
                for candidate_model, engine, concurrency in aggregates
                if candidate_model == model
            }
            missing = sorted(
                (engine, concurrency)
                for engine in required_arms
                for concurrency in required_concurrencies
                if (engine, concurrency) not in present
            )
            if missing:
                reason = "campaign grid is incomplete: " + ", ".join(
                    f"{engine}/c{concurrency}" for engine, concurrency in missing
                )
                for key, aggregate in aggregates.items():
                    if key[0] == model:
                        aggregate["binding_eligible"] = False
                        aggregate["reasons"].append(reason)

    ratios: list[dict] = []
    models_and_concurrency = sorted({(model, concurrency) for model, _, concurrency in groups})
    for model, concurrency in models_and_concurrency:
        arms = {
            engine: aggregates[(model, engine, concurrency)]
            for engine in ("ours", "vllm", "sglang")
            if (model, engine, concurrency) in aggregates
        }
        if "ours" not in arms:
            continue
        for axis in (*HIGHER_AXES, *LOWER_AXES):
            ours_metric = arms["ours"]["metrics"].get(axis)
            floors = {
                engine: arm["metrics"].get(axis)
                for engine, arm in arms.items()
                if engine != "ours" and arm["metrics"].get(axis) is not None
            }
            if ours_metric is None or not floors:
                continue
            ours = ours_metric["median"]
            floor_values = {engine: metric["median"] for engine, metric in floors.items()}
            higher = axis in HIGHER_AXES
            best_floor = max(floor_values.values()) if higher else min(floor_values.values())
            ratios.append(
                {
                    "axis": axis,
                    "binding_eligible": arms["ours"]["binding_eligible"] and all(
                        arms[engine]["binding_eligible"] for engine in floor_values
                    ),
                    "best_floor": best_floor,
                    "concurrency": concurrency,
                    "direction": "higher" if higher else "lower",
                    "model": model,
                    "ours": ours,
                    "ours_over_best_floor": ours / best_floor if best_floor else None,
                    "pass_normalized_ratio": (
                        ours / best_floor if higher else best_floor / ours
                    ) if ours and best_floor else None,
                    "per_floor_ours_ratio": {
                        engine: (ours / value if value else None)
                        for engine, value in floor_values.items()
                    },
                }
            )

    run_document = {
        "aggregates": list(aggregates.values()),
        "raw_runs": all_runs,
    }
    ratio_document = {"ratios": ratios}
    return run_document, ratio_document


def _report(runs: Mapping[str, Any], ratios: Mapping[str, Any]) -> str:
    lines = ["# SGLang low-concurrency preflight summary", ""]
    eligible = sum(item["binding_eligible"] for item in runs["aggregates"])
    lines.append(
        f"Binding-eligible repetition groups: {eligible}/{len(runs['aggregates'])}."
    )
    lines.append("")
    for aggregate in runs["aggregates"]:
        status = "eligible" if aggregate["binding_eligible"] else "VOID/PENDING"
        lines.append(
            f"- {aggregate['model']} {aggregate['engine']} c{aggregate['concurrency']}: "
            f"{status}"
        )
        for reason in aggregate["reasons"]:
            lines.append(f"  - {reason}")
    lines.extend(("", f"Axis ratios emitted: {len(ratios['ratios'])}.", ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    args = parser.parse_args()
    runs, ratios = summarize_evidence(args.evidence, require_full_grid=True)
    summary = args.evidence / "summary"
    if summary.exists() and any(summary.iterdir()):
        raise HarnessError(f"refusing to overwrite summary evidence in {summary}")
    write_json_atomic(summary / "all-runs.json", runs)
    write_json_atomic(summary / "ratios.json", ratios)
    write_text_atomic(summary / "report.md", _report(runs, ratios))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
