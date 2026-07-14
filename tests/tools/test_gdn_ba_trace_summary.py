"""Fail-closed structural contracts for merged Qwen GDN BA traces.

The execution target mirrors vLLM 0.25.0
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:908-943``:
one physical BA projection is required in the merged arm.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest
from unittest import mock

from tools.bench.finalize_gdn_ba_trace import (
    STATUS,
    build_summary,
    finalize,
    summarize_gdn_ba_topology,
)
from tools.bench.serve_low_common import HarnessError


def arm(*, kernels: int, bf16: int, topology: str) -> dict:
    return {
        "local": {
            "family_summary": {
                "bf16_cutlass_gemm": {"count_per_window": bf16},
                "fa2_main": {"count_per_window": 16},
                "fp4_gemm": {"count_per_window": 208},
                "other": {"count_per_window": 241},
            },
            "kernel_count_per_window": kernels,
            "node_multiset_sha256": topology,
        }
    }


class GdnBaTraceSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arms = {
            "merged": arm(kernels=963, bf16=145, topology="a" * 64),
            "split": arm(kernels=1_011, bf16=193, topology="b" * 64),
        }

    def test_exact_delta_removes_only_48_bf16_gemms(self) -> None:
        result = summarize_gdn_ba_topology(self.arms)
        self.assertEqual(result["merged_kernel_count"], 963)
        self.assertEqual(result["split_kernel_count"], 1_011)
        self.assertEqual(result["bf16_gemm_delta_split_minus_merged"], 48)
        self.assertEqual(result["kernel_delta_split_minus_merged"], 48)
        self.assertFalse(result["speed_credit"])

    def test_rejects_missing_arm_and_total_drift(self) -> None:
        with self.assertRaisesRegex(HarnessError, "exactly merged and split"):
            summarize_gdn_ba_topology({"merged": self.arms["merged"]})
        drifted = copy.deepcopy(self.arms)
        drifted["merged"]["local"]["kernel_count_per_window"] = 964
        with self.assertRaisesRegex(HarnessError, "total kernel count"):
            summarize_gdn_ba_topology(drifted)

    def test_rejects_bf16_or_unrelated_family_drift(self) -> None:
        bf16_drift = copy.deepcopy(self.arms)
        bf16_drift["merged"]["local"]["family_summary"][
            "bf16_cutlass_gemm"
        ]["count_per_window"] = 146
        with self.assertRaisesRegex(HarnessError, "BF16 GEMM count"):
            summarize_gdn_ba_topology(bf16_drift)

        unrelated = copy.deepcopy(self.arms)
        unrelated["merged"]["local"]["family_summary"]["fa2_main"][
            "count_per_window"
        ] = 15
        with self.assertRaisesRegex(HarnessError, "non-BF16 family"):
            summarize_gdn_ba_topology(unrelated)

    def test_rejects_identical_topology_hashes(self) -> None:
        drifted = copy.deepcopy(self.arms)
        drifted["split"]["local"]["node_multiset_sha256"] = "a" * 64
        with self.assertRaisesRegex(HarnessError, "unexpectedly match"):
            summarize_gdn_ba_topology(drifted)

    def test_build_summary_requires_matching_fresh_oracles(self) -> None:
        built_arms = {
            mode: {
                **record,
                "oracle": {
                    "ordered_names_sha256": "c" * 64,
                    "steady_b2_kernel_count_per_window": 1_160,
                },
            }
            for mode, record in self.arms.items()
        }
        with (
            mock.patch(
                "tools.bench.finalize_gdn_ba_trace._build_arm",
                side_effect=lambda _evidence, *, source_commit, mode: built_arms[mode],
            ),
            mock.patch(
                "tools.bench.finalize_gdn_ba_trace._validate_model_gate",
                return_value={"passed": True},
            ),
            mock.patch(
                "tools.bench.finalize_gdn_ba_trace._validate_run_log",
                return_value={"sha256": "d" * 64},
            ),
        ):
            summary = build_summary(
                pathlib.Path("/evidence"),
                source_commit="e" * 40,
                run_log=pathlib.Path("/run.log"),
            )
            self.assertEqual(summary["status"], STATUS)
            self.assertFalse(summary["benchmark_binding"])

            built_arms["split"]["oracle"][
                "steady_b2_kernel_count_per_window"
            ] = 1_159
            with self.assertRaisesRegex(HarnessError, "vLLM oracle"):
                build_summary(
                    pathlib.Path("/evidence"),
                    source_commit="e" * 40,
                    run_log=pathlib.Path("/run.log"),
                )

    def test_finalize_writes_completion_marker_last_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = pathlib.Path(temporary)
            (evidence / "trace/27").mkdir(parents=True)
            run_log = evidence / "run.log"
            run_log.write_text("complete\n", encoding="utf-8")
            finalizer = evidence / "finalizer.py"
            finalizer.write_text("# fixture\n", encoding="utf-8")
            with (
                mock.patch(
                    "tools.bench.finalize_gdn_ba_trace.build_summary",
                    return_value={"status": STATUS},
                ),
                mock.patch(
                    "tools.bench.finalize_gdn_ba_trace.build_manifest",
                    return_value={"artifact_set_sha256": "f" * 64},
                ),
            ):
                _, _, marker = finalize(
                    evidence,
                    source_commit="e" * 40,
                    run_log=run_log,
                    finalizer=finalizer,
                )
                self.assertEqual(marker["status"], STATUS)
                self.assertTrue((evidence / "trace/27/gdn-ba-summary.json").is_file())
                self.assertTrue((evidence / "trace/27/gdn-ba-manifest.json").is_file())
                self.assertTrue((evidence / "trace/27/status-gdn-ba.json").is_file())
                with self.assertRaisesRegex(HarnessError, "refusing to overwrite"):
                    finalize(
                        evidence,
                        source_commit="e" * 40,
                        run_log=run_log,
                        finalizer=finalizer,
                    )


if __name__ == "__main__":
    unittest.main()
