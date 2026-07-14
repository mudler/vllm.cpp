"""Fail-closed structural contracts for packed Qwen GDN decode traces.

The execution target mirrors vLLM 0.25.0
``vllm/model_executor/layers/fla/ops/fused_recurrent.py:255-478``:
one packed recurrence must replace the decomposed recurrence and post-conv
pair on every pure-decode GDN layer.
"""

from __future__ import annotations

import copy
import pathlib
import tempfile
import unittest
from unittest import mock

from tools.bench.finalize_gdn_packed_trace import (
    STATUS,
    build_summary,
    finalize,
    summarize_gdn_packed_topology,
)
from tools.bench.serve_low_common import HarnessError, sha256_file


def arm(
    *,
    kernels: int,
    packed: int,
    decomposed: int,
    post_conv: int,
    topology: str,
) -> dict:
    return {
        "local": {
            "family_summary": {
                "bf16_cutlass_gemm": {"count_per_window": 145},
                "fa2_main": {"count_per_window": 16},
                "fp4_gemm": {"count_per_window": 208},
                "gdn_packed_recurrence": {"count_per_window": packed},
                "gdn_decomposed_recurrence": {"count_per_window": decomposed},
                "gdn_post_conv": {"count_per_window": post_conv},
                "other": {"count_per_window": 241},
            },
            "kernel_count_per_window": kernels,
            "node_multiset_sha256": topology,
            "gdn_packed_invariant_node_multiset_sha256": "c" * 64,
            "gdn_packed_coupled_ba_node_multiset_sha256": topology,
            "gdn_packed_coupled_ba_node_count": 48,
        }
    }


class GdnPackedTraceSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arms = {
            "packed": arm(
                kernels=915,
                packed=48,
                decomposed=0,
                post_conv=0,
                topology="a" * 64,
            ),
            "rollback": arm(
                kernels=963,
                packed=0,
                decomposed=48,
                post_conv=48,
                topology="b" * 64,
            ),
        }

    def test_exact_delta_replaces_two_legacy_nodes_with_one_packed_node(self) -> None:
        result = summarize_gdn_packed_topology(self.arms)
        self.assertEqual(result["packed_kernel_count"], 915)
        self.assertEqual(result["rollback_kernel_count"], 963)
        self.assertEqual(result["kernel_delta_rollback_minus_packed"], 48)
        self.assertEqual(result["packed_recurrence_nodes"], 48)
        self.assertEqual(result["rollback_decomposed_nodes"], 48)
        self.assertEqual(result["rollback_post_conv_nodes"], 48)
        self.assertFalse(result["speed_credit"])

    def test_rejects_missing_arm_and_total_drift(self) -> None:
        with self.assertRaisesRegex(HarnessError, "exactly packed and rollback"):
            summarize_gdn_packed_topology({"packed": self.arms["packed"]})
        drifted = copy.deepcopy(self.arms)
        drifted["packed"]["local"]["kernel_count_per_window"] = 916
        with self.assertRaisesRegex(HarnessError, "total kernel count"):
            summarize_gdn_packed_topology(drifted)

    def test_rejects_gdn_or_unrelated_family_drift(self) -> None:
        recurrence_drift = copy.deepcopy(self.arms)
        recurrence_drift["packed"]["local"]["family_summary"][
            "gdn_packed_recurrence"
        ]["count_per_window"] = 47
        with self.assertRaisesRegex(HarnessError, "packed recurrence count"):
            summarize_gdn_packed_topology(recurrence_drift)

        unrelated = copy.deepcopy(self.arms)
        unrelated["packed"]["local"]["family_summary"]["fa2_main"][
            "count_per_window"
        ] = 15
        with self.assertRaisesRegex(HarnessError, "unrelated family"):
            summarize_gdn_packed_topology(unrelated)

    def test_rejects_identical_topology_hashes(self) -> None:
        drifted = copy.deepcopy(self.arms)
        drifted["rollback"]["local"]["node_multiset_sha256"] = "a" * 64
        with self.assertRaisesRegex(HarnessError, "unexpectedly match"):
            summarize_gdn_packed_topology(drifted)

    def test_rejects_same_count_unrelated_topology_drift(self) -> None:
        drifted = copy.deepcopy(self.arms)
        drifted["rollback"]["local"][
            "gdn_packed_invariant_node_multiset_sha256"
        ] = "d" * 64
        with self.assertRaisesRegex(HarnessError, "invariant node topology"):
            summarize_gdn_packed_topology(drifted)

    def test_allows_mode_specific_coupled_ba_signatures_but_requires_48(self) -> None:
        result = summarize_gdn_packed_topology(self.arms)
        self.assertNotEqual(
            result["packed_coupled_ba_node_multiset_sha256"],
            result["rollback_coupled_ba_node_multiset_sha256"],
        )
        drifted = copy.deepcopy(self.arms)
        drifted["rollback"]["local"]["gdn_packed_coupled_ba_node_count"] = 47
        with self.assertRaisesRegex(HarnessError, "coupled BA node count"):
            summarize_gdn_packed_topology(drifted)

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
                "tools.bench.finalize_gdn_packed_trace._build_arm",
                side_effect=lambda _evidence, *, source_commit, mode: built_arms[mode],
            ),
            mock.patch(
                "tools.bench.finalize_gdn_packed_trace._validate_model_gate",
                return_value={"passed": True},
            ),
            mock.patch(
                "tools.bench.finalize_gdn_packed_trace._validate_run_log",
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

            built_arms["rollback"]["oracle"][
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
            validator = evidence / "validator.py"
            validator.write_text("# validator fixture\n", encoding="utf-8")
            with (
                mock.patch(
                    "tools.bench.finalize_gdn_packed_trace.build_summary",
                    return_value={"status": STATUS},
                ),
                mock.patch(
                    "tools.bench.finalize_gdn_packed_trace.build_manifest",
                    return_value={"artifact_set_sha256": "f" * 64},
                ),
            ):
                _, _, marker = finalize(
                    evidence,
                    source_commit="e" * 40,
                    run_log=run_log,
                    finalizer=finalizer,
                    validator=validator,
                )
                self.assertEqual(marker["status"], STATUS)
                self.assertEqual(marker["validator_sha256"], sha256_file(validator))
                self.assertTrue(
                    (evidence / "trace/27/gdn-packed-summary.json").is_file()
                )
                self.assertTrue(
                    (evidence / "trace/27/gdn-packed-manifest.json").is_file()
                )
                self.assertTrue(
                    (evidence / "trace/27/status-gdn-packed.json").is_file()
                )
                with self.assertRaisesRegex(HarnessError, "refusing to overwrite"):
                    finalize(
                        evidence,
                        source_commit="e" * 40,
                        run_log=run_log,
                        finalizer=finalizer,
                        validator=validator,
                    )


if __name__ == "__main__":
    unittest.main()
