#!/usr/bin/env python3
"""Finalize the exact batch-2 packed/rollback GDN decode trace series.

The GPU driver captures two complete paired profiles under one lock. Each
local arm carries an explicit ``VT_GDN_PACKED_DECODE`` value and is
independently revalidated against the exact vLLM 0.25.0 packed-recurrence
contract at ``vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-478``.
This CPU-only finalizer proves that one packed recurrence replaces the old
post-conv plus decomposed-recurrence pair on every GDN layer. Kernel durations
remain diagnostic and earn no benchmark speed credit.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

if __package__ in (None, ""):
    script = pathlib.Path(__file__).resolve()
    if len(script.parents) > 2:
        sys.path.insert(0, str(script.parents[2]))

from tools.bench.finalize_low_batch_trace import (
    MODEL_KEY,
    SOURCE_COMMIT_RE,
    TRACE_BATCH,
    TRACE_PROMPTS,
    _validate_model_gate,
    summarize_local_c2,
    summarize_vllm_c2_trace,
)
from tools.bench.online_gate import (
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    TRACE_GDN_PACKED_COUPLED_BA_NODE_COUNT,
    TRACE_GDN_PACKED_MODES,
    TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_PACKED_MODE,
    TRACE_REPETITIONS,
    _load_json_object,
    record_trace_status,
)
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    sha256_file,
    write_json_atomic,
)


STATUS = "complete-structural"
DERIVED_PATHS = frozenset(
    {
        "trace/27/gdn-packed-manifest.json",
        "trace/27/gdn-packed-summary.json",
        "trace/27/status-gdn-packed.json",
    }
)
_GDN_FAMILIES = frozenset(
    {
        "gdn_packed_recurrence",
        "gdn_decomposed_recurrence",
        "gdn_post_conv",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _trace_paths(evidence: pathlib.Path, mode: str) -> dict[str, Any]:
    if mode not in TRACE_GDN_PACKED_MODES:
        raise HarnessError(f"unknown GDN packed trace mode: {mode}")
    trace = evidence / "trace" / MODEL_KEY / f"gdn-{mode}"
    return {
        "trace": trace,
        "reports": sorted(trace.glob("ours-r?.?.nsys-rep")),
        "sqlites": sorted(trace.glob("ours-r?.?.sqlite")),
        "validations": sorted(trace.glob("ours-r?.?-nsys-validation.json")),
        "summaries": sorted(trace.glob("ours-r?.?-cuda_gpu_kern_sum.txt")),
        "commands": sorted(trace.glob("ours-r?-profile-command.txt")),
        "logs": sorted(trace.glob("ours-r?-profile.log")),
        "controls": sorted(trace.glob("ours-r?-profile-control.json")),
    }


def _validate_run_log(path: pathlib.Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise HarnessError(f"run log is absent: {path}") from error
    required = {
        "c2 GDN packed packed raw paired trace capture complete; status remains "
        "PENDING until GDN packed finalization",
        "c2 GDN packed rollback raw paired trace capture complete; status remains "
        "PENDING until GDN packed finalization",
        "model 27 GDN packed/rollback node-level paired traces complete",
    }
    if not required.issubset(set(lines)):
        raise HarnessError("run log lacks the complete packed/rollback trace terminus")
    return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}


def _build_arm(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    mode: str,
) -> dict[str, Any]:
    paths = _trace_paths(evidence, mode)
    trace = paths["trace"]
    vllm_kernel_summary = trace / "vllm-kernels.json"
    vllm_aggregate = _load_json_object(vllm_kernel_summary)
    selected_trace = vllm_aggregate.get("selected_trace")
    if (
        not isinstance(selected_trace, str)
        or not pathlib.Path(selected_trace).is_absolute()
    ):
        raise HarnessError(f"{mode} vLLM summary lacks an absolute selected trace")
    vllm_trace = pathlib.Path(selected_trace)
    tag = f"gdn-{mode}-"
    core = record_trace_status(
        trace / "c2-core.json",
        model_key=MODEL_KEY,
        ours_nsys_reports=paths["reports"],
        ours_nsys_sqlites=paths["sqlites"],
        ours_nsys_validations=paths["validations"],
        ours_kernel_summaries=paths["summaries"],
        ours_commands=paths["commands"],
        ours_profile_logs=paths["logs"],
        ours_profile_controls=paths["controls"],
        ours_client_results=[
            evidence
            / "raw"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-{tag}trace{index}.json"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_client_logs=[
            evidence
            / "logs"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-{tag}trace{index}.log"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_probe_results=[
            evidence
            / "raw"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-{tag}trace{index}-probe.json"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        ours_probe_logs=[
            evidence
            / "logs"
            / MODEL_KEY
            / "ours"
            / f"c2-r1-{tag}trace{index}-probe.log"
            for index in range(1, TRACE_REPETITIONS + 1)
        ],
        vllm_torch_trace=vllm_trace,
        vllm_kernel_summary=vllm_kernel_summary,
        vllm_command=trace / "vllm-profile-command.txt",
        vllm_profile_log=trace / "vllm-profile.log",
        vllm_metadata=trace / "vllm-profile-metadata.json",
        vllm_corpus=evidence / "corpus" / MODEL_KEY / "vllm" / "c2-r1.jsonl",
        cache_drop_reports=[
            trace / "cache-before-ours.json",
            trace / "cache-between-engines.json",
            trace / "cache-after-vllm.json",
        ],
        execution_manifest=evidence / "execution" / "27-trace.json",
        vllm_cpp_sha=source_commit,
        expected_batch=TRACE_BATCH,
        trace_prompts=TRACE_PROMPTS,
        gdn_packed_mode=mode,
        write_output=False,
    )
    metadata = _load_json_object(trace / "vllm-profile-metadata.json")
    if metadata.get("async_scheduling_requested") != "default":
        raise HarnessError(f"{mode} vLLM trace changed default async scheduling")
    if metadata.get("async_scheduling_resolved") is not True:
        raise HarnessError(f"{mode} vLLM default async scheduling is not enabled")
    contract = TRACE_PRIMARY_GRAPH_CONTRACTS_BY_GDN_PACKED_MODE[
        (MODEL_KEY, TRACE_BATCH, mode)
    ]
    local = summarize_local_c2(core, expected_graph_contract=contract)
    oracle = summarize_vllm_c2_trace(vllm_trace)
    if oracle["selected_trace_sha256"] != vllm_aggregate.get(
        "selected_trace_sha256"
    ):
        raise HarnessError(f"{mode} vLLM selected trace hash differs")
    return {"core_trace_status": core, "local": local, "oracle": oracle}


def _family_counts(local: Mapping[str, Any], mode: str) -> dict[str, int]:
    families = local.get("family_summary")
    if not isinstance(families, Mapping):
        raise HarnessError(f"GDN packed {mode} family summary is absent")
    counts = {}
    for family, record in families.items():
        if not isinstance(family, str) or not isinstance(record, Mapping):
            raise HarnessError(f"GDN packed {mode} family summary is malformed")
        count = record.get("count_per_window")
        if isinstance(count, bool) or not isinstance(count, int):
            raise HarnessError(f"GDN packed {mode} family count is malformed")
        counts[family] = count
    return counts


def summarize_gdn_packed_topology(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one packed recurrence to replace two legacy GDN nodes."""

    if set(arms) != set(TRACE_GDN_PACKED_MODES):
        raise HarnessError("GDN topology requires exactly packed and rollback arms")
    expected = {
        "packed": {
            "kernels": 915,
            "bf16": 145,
            "packed": 48,
            "decomposed": 0,
            "post_conv": 0,
        },
        "rollback": {
            "kernels": 963,
            "bf16": 145,
            "packed": 0,
            "decomposed": 48,
            "post_conv": 48,
        },
    }
    family_counts: dict[str, dict[str, int]] = {}
    topology_hashes = {}
    invariant_topology_hashes = {}
    coupled_ba_topology_hashes = {}
    for mode in TRACE_GDN_PACKED_MODES:
        local = arms[mode].get("local")
        if not isinstance(local, Mapping):
            raise HarnessError(f"GDN packed {mode} local summary is absent")
        if local.get("kernel_count_per_window") != expected[mode]["kernels"]:
            raise HarnessError(f"GDN packed {mode} total kernel count drifted")
        counts = _family_counts(local, mode)
        if counts.get("bf16_cutlass_gemm", 0) != expected[mode]["bf16"]:
            raise HarnessError(f"GDN packed {mode} BF16 GEMM count drifted")
        if counts.get("gdn_packed_recurrence", 0) != expected[mode]["packed"]:
            raise HarnessError(f"GDN packed {mode} packed recurrence count drifted")
        if (
            counts.get("gdn_decomposed_recurrence", 0)
            != expected[mode]["decomposed"]
        ):
            raise HarnessError(
                f"GDN packed {mode} decomposed recurrence count drifted"
            )
        if counts.get("gdn_post_conv", 0) != expected[mode]["post_conv"]:
            raise HarnessError(f"GDN packed {mode} post-conv count drifted")
        topology_hash = local.get("node_multiset_sha256")
        if not isinstance(topology_hash, str):
            raise HarnessError(f"GDN packed {mode} topology hash is absent")
        family_counts[mode] = counts
        topology_hashes[mode] = topology_hash
        invariant_topology_hash = local.get(
            "gdn_packed_invariant_node_multiset_sha256"
        )
        if not isinstance(invariant_topology_hash, str):
            raise HarnessError(
                f"GDN packed {mode} invariant node topology hash is absent"
            )
        invariant_topology_hashes[mode] = invariant_topology_hash
        coupled_ba_topology_hash = local.get(
            "gdn_packed_coupled_ba_node_multiset_sha256"
        )
        if not isinstance(coupled_ba_topology_hash, str):
            raise HarnessError(
                f"GDN packed {mode} coupled BA topology hash is absent"
            )
        coupled_ba_topology_hashes[mode] = coupled_ba_topology_hash
        if (
            local.get("gdn_packed_coupled_ba_node_count")
            != TRACE_GDN_PACKED_COUPLED_BA_NODE_COUNT
        ):
            raise HarnessError(f"GDN packed {mode} coupled BA node count drifted")
    if topology_hashes["packed"] == topology_hashes["rollback"]:
        raise HarnessError("GDN packed/rollback topology hashes unexpectedly match")
    if invariant_topology_hashes["packed"] != invariant_topology_hashes["rollback"]:
        raise HarnessError("GDN packed/rollback invariant node topology differs")
    packed_other = {
        family: count
        for family, count in family_counts["packed"].items()
        if family not in _GDN_FAMILIES
    }
    rollback_other = {
        family: count
        for family, count in family_counts["rollback"].items()
        if family not in _GDN_FAMILIES
    }
    if packed_other != rollback_other:
        raise HarnessError("GDN packed/rollback unrelated family counts differ")
    kernel_delta = expected["rollback"]["kernels"] - expected["packed"]["kernels"]
    replaced_legacy = (
        expected["rollback"]["decomposed"] + expected["rollback"]["post_conv"]
    )
    replacement = expected["packed"]["packed"]
    if kernel_delta != 48 or replaced_legacy - replacement != 48:
        raise HarnessError("GDN packed structural delta is not exactly 48 kernels")
    return {
        "kernel_delta_rollback_minus_packed": kernel_delta,
        "packed_bf16_gemms": expected["packed"]["bf16"],
        "packed_kernel_count": expected["packed"]["kernels"],
        "packed_recurrence_nodes": replacement,
        "rollback_bf16_gemms": expected["rollback"]["bf16"],
        "rollback_decomposed_nodes": expected["rollback"]["decomposed"],
        "rollback_kernel_count": expected["rollback"]["kernels"],
        "rollback_post_conv_nodes": expected["rollback"]["post_conv"],
        "invariant_family_counts": packed_other,
        "invariant_node_multiset_sha256": invariant_topology_hashes["packed"],
        "packed_coupled_ba_node_multiset_sha256": (
            coupled_ba_topology_hashes["packed"]
        ),
        "rollback_coupled_ba_node_multiset_sha256": (
            coupled_ba_topology_hashes["rollback"]
        ),
        "speed_credit": False,
    }


def build_summary(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    run_log: pathlib.Path,
) -> dict[str, Any]:
    if SOURCE_COMMIT_RE.fullmatch(source_commit) is None:
        raise HarnessError("source commit must be a lowercase 40-character Git SHA")
    arms = {
        mode: _build_arm(evidence, source_commit=source_commit, mode=mode)
        for mode in TRACE_GDN_PACKED_MODES
    }
    for field in ("ordered_names_sha256", "steady_b2_kernel_count_per_window"):
        values = {arms[mode]["oracle"].get(field) for mode in TRACE_GDN_PACKED_MODES}
        if len(values) != 1:
            raise HarnessError(f"packed/rollback vLLM oracle {field} differs")
    return {
        "schema_version": 1,
        "status": STATUS,
        "benchmark_binding": False,
        "benchmark_binding_reason": (
            "exact local graph topology only; separate-profiler durations do not "
            "replace the c2/c16 every-axis component gate"
        ),
        "evidence_source_commit": source_commit,
        "workload": {
            "admission_mode": "closed-loop",
            "input_len": 1024,
            "max_concurrency": TRACE_BATCH,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS[MODEL_KEY],
            "max_num_seqs": MAX_NUM_SEQS,
            "model": "Qwen3.6-27B-NVFP4",
            "num_prompts": TRACE_PROMPTS,
            "output_len": 128,
            "prefix_caching": False,
        },
        "model_gate": _validate_model_gate(
            evidence / "preflight" / "model-gate" / "27.json", source_commit
        ),
        "run_log": _validate_run_log(run_log),
        "arms": arms,
        "topology": summarize_gdn_packed_topology(arms),
    }


def build_manifest(
    evidence: pathlib.Path, *, run_log: pathlib.Path
) -> dict[str, Any]:
    files = []
    for path in sorted(evidence.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(evidence).as_posix()
        if relative in DERIVED_PATHS:
            continue
        files.append(
            {"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size}
        )
    if not files:
        raise HarnessError(f"no immutable GDN packed trace artifacts under {evidence}")
    external = [
        {"path": str(run_log), "sha256": sha256_file(run_log), "size": run_log.stat().st_size}
    ]
    return {
        "schema_version": 1,
        "artifact_set_sha256": _digest(files + external),
        "external_files": external,
        "files": files,
    }


def finalize(
    evidence: pathlib.Path,
    *,
    source_commit: str,
    run_log: pathlib.Path,
    finalizer: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trace = evidence / "trace" / MODEL_KEY
    summary_path = trace / "gdn-packed-summary.json"
    manifest_path = trace / "gdn-packed-manifest.json"
    marker_path = trace / "status-gdn-packed.json"
    existing = [
        str(path) for path in (summary_path, manifest_path, marker_path) if path.exists()
    ]
    if existing:
        raise HarnessError(f"refusing to overwrite derived artifact(s): {existing}")
    summary = build_summary(evidence, source_commit=source_commit, run_log=run_log)
    manifest = build_manifest(evidence, run_log=run_log)
    write_json_atomic(summary_path, summary)
    write_json_atomic(manifest_path, manifest)
    marker = {
        "schema_version": 1,
        "status": STATUS,
        "evidence_source_commit": source_commit,
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "finalizer_sha256": sha256_file(finalizer),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
    }
    write_json_atomic(marker_path, marker)
    return summary, manifest, marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-log", type=pathlib.Path)
    args = parser.parse_args()
    run_log = args.run_log
    if run_log is None:
        run_log = args.evidence.parent.parent / "gdn-packed-run.log"
    summary, _, marker = finalize(
        args.evidence,
        source_commit=args.source_commit,
        run_log=run_log,
        finalizer=pathlib.Path(__file__),
    )
    print(
        canonical_json(
            {
                "packed_kernel_count": summary["topology"]["packed_kernel_count"],
                "rollback_kernel_count": summary["topology"][
                    "rollback_kernel_count"
                ],
                "status": marker["status"],
                "summary_sha256": marker["summary_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"finalize-gdn-packed-trace: {error}", file=sys.stderr)
        raise SystemExit(2) from error
