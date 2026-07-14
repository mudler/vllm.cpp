#!/usr/bin/env python3
"""Contracts for the exact batch-2 trace finalizer."""

from __future__ import annotations

import gzip
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from tools.bench import finalize_low_batch_trace as finalizer
from tools.bench.serve_low_common import HarnessError, sha256_file


FAMILY_CONTRACT = {
    "fa2_combine": ("fa2-combine", 1),
    "fa2_main": ("fa2-main", 1),
    "fp4_gemm": ("MainloopSm120TmaWarpSpecializedBlockScaled", 2),
    "fused_fp4_producer": ("fused-producer", 1),
    "gdn_recurrence": ("gdn-recurrence", 1),
    "normal_fp4_producer": ("normal-producer", 1),
}
B2_NAMES = [
    "MainloopSm120TmaWarpSpecializedBlockScaled StreamKScheduler",
    "MainloopSm120TmaWarpSpecializedBlockScaled StaticPersistentScheduler",
    "normal-producer",
    "fused-producer",
    "gdn-recurrence",
    "fa2-main",
    "fa2-combine",
    "triton_red_fused_rms_norm",
    "other-kernel",
]


def signature(name: str, registers: int = 32) -> dict[str, object]:
    return {
        "block": [32, 1, 1],
        "grid": [1, 1, 1],
        "name": name,
        "registers_per_thread": registers,
        "shared_memory": 0,
    }


def write_trace(
    path: pathlib.Path,
    batches: list[int],
    *,
    mutate_second_signature: bool = False,
    crossing_kernel: bool = False,
) -> str:
    events = []
    for window_index, batch in enumerate(batches):
        start = float(window_index * 1_000)
        events.append(
            {
                "cat": "gpu_user_annotation",
                "dur": 500.0,
                "name": f"execute_context_0(0)_generation_{batch}({batch})",
                "ph": "X",
                "ts": start,
            }
        )
        names = B2_NAMES if batch == 2 else ["drain-kernel"]
        for kernel_index, name in enumerate(names):
            registers = (
                33
                if mutate_second_signature
                and window_index == 1
                and kernel_index == 0
                else 32
            )
            duration = (
                600.0
                if crossing_kernel and window_index == 0 and not kernel_index
                else 1.0
            )
            launch = signature(name, registers)
            events.append(
                {
                    "args": {
                        "block": launch["block"],
                        "grid": launch["grid"],
                        "registers per thread": launch["registers_per_thread"],
                        "shared memory": launch["shared_memory"],
                    },
                    "cat": "kernel",
                    "dur": duration,
                    "name": name,
                    "ph": "X",
                    "ts": start + 10.0 + kernel_index * 10.0,
                }
            )
    value = {"traceEvents": events}
    with gzip.open(path, "wt", encoding="utf-8") as output:
        json.dump(value, output)
    return finalizer._digest([signature(name) for name in B2_NAMES])


def summarize(path: pathlib.Path, batches: list[int], signature_sha256: str):
    drain_count = batches.count(1)
    return finalizer.summarize_vllm_c2_trace(
        path,
        expected_pure_windows=len(batches),
        expected_base_generation_windows=len(batches) - drain_count,
        max_drain_windows=1,
        expected_kernel_count=len(B2_NAMES),
        expected_names_sha256=finalizer._digest(B2_NAMES),
        allowed_signature_sha256={signature_sha256: "test-signature"},
        family_contract=FAMILY_CONTRACT,
        expected_fp4_tactics={"static_persistent": 1, "stream_k": 1},
        expected_generated_rmsnorm_quant=1,
    )


class LowBatchTraceSummaryTests(unittest.TestCase):
    def test_accepts_invariant_b2_windows_and_one_bounded_drain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "trace.json.gz"
            signature_sha256 = write_trace(path, [2, 2, 1])
            result = summarize(path, [2, 2, 1], signature_sha256)
            self.assertEqual(result["steady_b2_window_count"], 2)
            self.assertEqual(result["drain_window_count"], 1)
            self.assertEqual(result["steady_b2_kernel_count_per_window"], 9)
            self.assertEqual(
                result["family_summary"]["fp4_gemm"]["count_per_window"], 2
            )
            self.assertEqual(
                result["family_summary"]["rmsnorm_generated_partition"][
                    "count_per_window"
                ],
                1,
            )

    def test_rejects_name_signature_drain_and_boundary_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            clean = root / "clean.json.gz"
            signature_sha256 = write_trace(clean, [2, 2])
            with self.assertRaisesRegex(HarnessError, "ordered kernel names drifted"):
                finalizer.summarize_vllm_c2_trace(
                    clean,
                    expected_pure_windows=2,
                    expected_base_generation_windows=2,
                    expected_kernel_count=9,
                    expected_names_sha256="0" * 64,
                    allowed_signature_sha256={signature_sha256: "test"},
                    family_contract=FAMILY_CONTRACT,
                    expected_fp4_tactics={"static_persistent": 1, "stream_k": 1},
                    expected_generated_rmsnorm_quant=1,
                )

            signature_drift = root / "signature.json.gz"
            write_trace(signature_drift, [2, 2], mutate_second_signature=True)
            with self.assertRaisesRegex(HarnessError, "launch signature drifted"):
                summarize(signature_drift, [2, 2], signature_sha256)

            drain = root / "drain.json.gz"
            write_trace(drain, [2, 1])
            with self.assertRaisesRegex(HarnessError, "maximum is 0"):
                finalizer.summarize_vllm_c2_trace(
                    drain,
                    expected_pure_windows=2,
                    expected_base_generation_windows=1,
                    max_drain_windows=0,
                    expected_kernel_count=9,
                    expected_names_sha256=finalizer._digest(B2_NAMES),
                    allowed_signature_sha256={signature_sha256: "test"},
                    family_contract=FAMILY_CONTRACT,
                    expected_fp4_tactics={"static_persistent": 1, "stream_k": 1},
                    expected_generated_rmsnorm_quant=1,
                )

            crossing = root / "crossing.json.gz"
            write_trace(crossing, [2, 2], crossing_kernel=True)
            with self.assertRaisesRegex(HarnessError, "crosses steady B=2"):
                summarize(crossing, [2, 2], signature_sha256)

    def test_local_summary_requires_twelve_identical_exact_ranges(self) -> None:
        kernels = []
        local_names = [
            *B2_NAMES[:7],
            "void vt::cuda::RmsNormRowKernel()",
            "other-local-kernel",
        ]
        for name in local_names:
            kernels.append({"count": 1, "name": name, "total_duration_ns": 1_000})
        core = {
            "nsys_kernel_summaries": [
                {
                    "kernel_count": 9,
                    "kernel_time_ns": 9_000,
                    "kernels": kernels,
                }
                for _ in range(12)
            ],
            "nsys_validations": [
                {"canonical_node_multiset_sha256": "a" * 64} for _ in range(12)
            ],
        }
        graph_contract = {
            "families": FAMILY_CONTRACT,
            "node_count": 9,
        }
        result = finalizer.summarize_local_c2(
            core,
            expected_graph_contract=graph_contract,
            expected_fp4_tactics={"static_persistent": 1, "stream_k": 1},
            expected_rmsnorm_count=1,
        )
        self.assertEqual(result["range_count"], 12)
        self.assertEqual(result["kernel_time_us"]["median"], 9.0)
        core["nsys_validations"][-1] = {
            "canonical_node_multiset_sha256": "b" * 64
        }
        with self.assertRaisesRegex(HarnessError, "topology is not invariant"):
            finalizer.summarize_local_c2(
                core,
                expected_graph_contract=graph_contract,
                expected_fp4_tactics={"static_persistent": 1, "stream_k": 1},
                expected_rmsnorm_count=1,
            )

    def test_finalize_writes_marker_last_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            run_log = root / "run.log"
            run_log.write_text("raw\n", encoding="utf-8")
            finalizer_path = root / "finalizer.py"
            finalizer_path.write_text("# finalizer\n", encoding="utf-8")
            summary = {"status": "complete-diagnostic"}
            manifest = {"artifact_set_sha256": "c" * 64}
            with (
                mock.patch.object(finalizer, "build_summary", return_value=summary),
                mock.patch.object(finalizer, "build_manifest", return_value=manifest),
            ):
                _, _, marker = finalizer.finalize(
                    evidence,
                    source_commit="d" * 40,
                    run_log=run_log,
                    finalizer=finalizer_path,
                )
                trace = evidence / "trace" / "27"
                self.assertTrue((trace / "c2-summary.json").is_file())
                self.assertTrue((trace / "c2-manifest.json").is_file())
                self.assertTrue((trace / "status-c2.json").is_file())
                self.assertEqual(
                    marker["summary_sha256"], sha256_file(trace / "c2-summary.json")
                )
                with self.assertRaisesRegex(HarnessError, "refusing to overwrite"):
                    finalizer.finalize(
                        evidence,
                        source_commit="d" * 40,
                        run_log=run_log,
                        finalizer=finalizer_path,
                    )


if __name__ == "__main__":
    unittest.main()
