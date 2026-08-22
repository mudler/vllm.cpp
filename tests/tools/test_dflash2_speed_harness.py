#!/usr/bin/env python3
"""Gate the SPEC-DFLASH2 harness on the paths that REFUSE (#1562).

A benchmark harness is not proven by a run that produced a number. It is proven
by the runs it DECLINED to turn into a number, because those are the paths that
protect every future measurement, and they are the paths a green run never
exercises.

So every case here drives a refusal, through the same functions and the same
`main()` entry points the leased run calls. Nothing constructs a private helper
by hand: `AGENTS.md` §Nothing lands dead says a unit test that builds the type
itself proves the class works and never that anything reaches it, so the
precheck cases run `dflash2_oracle_capture.main(["--precheck-only", ...])` and
the marker case runs `scripts/dflash2-speed-gate.sh` as a process.

Standard library only, no GPU, no `nvidia-smi`, no vLLM wheel.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bench import dflash2_oracle_capture as capture
from tools.bench import dflash2_our_arm as our_arm
from tools.bench import dflash2_speed_harness as harness
from tools.bench.serve_low_common import HarnessError

ORACLE_COMMIT = "3406ec1d0e5b4a2f9c1d8e7a6b5c4d3e2f1a0b9c"
GOOD_VERSION = "0.1.dev1+g3406ec1d0"

# A REAL file with a REAL digest. `checkpoint_reasons` MEASURES the hash against
# the bytes, so a fixture that names no path is refused -- correctly -- and a
# suite built on one would be testing the refusal it meant to satisfy.
_FIXTURES = pathlib.Path(tempfile.mkdtemp(prefix="dflash2-harness-fixture-"))
_ARTIFACT = _FIXTURES / "model.safetensors"
_ARTIFACT.write_bytes(b"a 27B checkpoint stands in for itself")
_ARTIFACT_SHA256 = hashlib.sha256(_ARTIFACT.read_bytes()).hexdigest()
atexit.register(shutil.rmtree, _FIXTURES, True)


def clock_record(*, boot: str = "boot-a", median: float = 2470.0) -> dict:
    """A HEALTHY window: 40 busy samples, no idle, nothing throttling."""

    return {
        "boot_id": boot,
        "clocks_applications_graphics_mhz": 2418,
        "clocks_max_sm_mhz": 3003,
        "driver_version": "580.159.03",
        "gpu_name": "NVIDIA GB10",
        "idle_samples_excluded": 0,
        "persistence_mode": "Enabled",
        "sm_clock_mhz": {
            "max": median + 10.0,
            "mean": median,
            "median": median,
            "min": median - 10.0,
            "n": 40,
            "spread_pct": 20.0 / median * 100.0,
        },
        "throttle_reasons_active": ["0x0"],
    }


class OracleIdentityTest(unittest.TestCase):
    def test_a_plain_release_wheel_is_refused(self) -> None:
        reasons = harness.oracle_identity_reasons("0.25.0", ORACLE_COMMIT)
        self.assertEqual(len(reasons), 1)
        self.assertIn("PLAIN RELEASE", reasons[0])

    def test_a_wheel_built_from_another_head_is_refused(self) -> None:
        reasons = harness.oracle_identity_reasons("0.1.dev1+g66e5414c6", ORACLE_COMMIT)
        self.assertEqual(len(reasons), 1)
        self.assertIn("not the declared oracle head", reasons[0])

    def test_an_absent_expected_commit_never_falls_back_to_the_pin(self) -> None:
        reasons = harness.oracle_identity_reasons(GOOD_VERSION, "")
        self.assertTrue(any("BEYOND-PIN" in reason or "no expected commit" in reason for reason in reasons))

    def test_the_declared_head_passes(self) -> None:
        self.assertEqual(harness.oracle_identity_reasons(GOOD_VERSION, ORACLE_COMMIT), [])


class AttentionBackendTest(unittest.TestCase):
    #: The exact string the committed FLASH_ATTN golden carries. #1562 exists
    #: because this label was never read back off the engine, and the gate has
    #: to refuse it by NAME or the defect can be reintroduced verbatim.
    RELABEL = (
        "corrected from the run log by w6-relabel.py; the capture's original value "
        "came from VLLM_ATTENTION_BACKEND, which does not exist in this wheel and "
        "selected nothing"
    )

    def test_the_1562_post_hoc_relabel_is_refused_by_its_source(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="FLASH_ATTN", declared="FLASH_ATTN", source=self.RELABEL
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("post-hoc relabel", reasons[0])

    def test_an_unread_backend_is_refused(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="", declared="TRITON_ATTN", source=harness.BACKEND_SOURCE_READ_BACK
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("no resolved attention backend", reasons[0])

    def test_a_substituted_denominator_is_refused(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="FLASH_ATTN",
            declared="TRITON_ATTN",
            source=harness.BACKEND_SOURCE_READ_BACK,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("substituted denominator", reasons[0])

    def test_a_read_back_label_that_agrees_passes(self) -> None:
        self.assertEqual(
            harness.attention_backend_reasons(
                resolved="TRITON_ATTN",
                declared="TRITON_ATTN",
                source=harness.BACKEND_SOURCE_READ_BACK,
            ),
            [],
        )


class KeepaliveTest(unittest.TestCase):
    def test_an_absent_variable_is_refused_even_though_zero_is_the_default(self) -> None:
        reasons = harness.sse_keepalive_reasons({})
        self.assertEqual(len(reasons), 1)
        self.assertIn("a default is not a record", reasons[0])

    def test_a_non_zero_ping_is_refused(self) -> None:
        reasons = harness.sse_keepalive_reasons({harness.SSE_PING_ENV: "5"})
        self.assertEqual(len(reasons), 1)
        self.assertIn("drops the latency tail", reasons[0])

    def test_an_explicit_zero_passes(self) -> None:
        self.assertEqual(harness.sse_keepalive_reasons({harness.SSE_PING_ENV: "0"}), [])


class DenominatorTest(unittest.TestCase):
    def test_enforce_eager_is_refused(self) -> None:
        reasons = harness.denominator_reasons({"enforce_eager": True})
        self.assertEqual(len(reasons), 1)
        self.assertIn("AGENTS.md", reasons[0])

    def test_an_unrecorded_graph_mode_is_refused(self) -> None:
        reasons = harness.denominator_reasons({})
        self.assertEqual(len(reasons), 1)
        self.assertIn("did not record", reasons[0])

    def test_the_production_configuration_passes(self) -> None:
        self.assertEqual(harness.denominator_reasons({"enforce_eager": False}), [])


class WorkloadTest(unittest.TestCase):
    def workload(self) -> dict:
        return {
            "prompts_sha256": harness.workload_fingerprint(["a", "b"]),
            "num_prompts": 2,
            "max_tokens": 64,
            "num_speculative_tokens": 7,
            "max_num_seqs": 1,
            "concurrency": 1,
            "temperature": 0.0,
            "seed": None,
        }

    def test_a_different_token_budget_voids_the_ratio(self) -> None:
        theirs = self.workload()
        theirs["max_tokens"] = 32
        reasons = harness.workload_reasons(self.workload(), theirs)
        self.assertEqual(len(reasons), 1)
        self.assertIn("max_tokens differs", reasons[0])

    def test_reordered_prompts_are_a_different_workload(self) -> None:
        self.assertNotEqual(
            harness.workload_fingerprint(["a", "b"]),
            harness.workload_fingerprint(["b", "a"]),
        )

    def test_identical_workloads_pass(self) -> None:
        self.assertEqual(harness.workload_reasons(self.workload(), self.workload()), [])


class CheckpointTest(unittest.TestCase):
    def test_a_hash_that_was_compared_with_nothing_is_refused(self) -> None:
        reasons = harness.checkpoint_reasons(
            [{"role": "target", "path": "/nonexistent/model.safetensors", "sha256": "0" * 64}]
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("compared with nothing", reasons[0])

    def test_a_repo_id_without_a_digest_is_refused(self) -> None:
        reasons = harness.checkpoint_reasons([{"role": "target", "sha256": "z-lab/Qwen3.8"}])
        self.assertEqual(len(reasons), 1)
        self.assertIn("not a\n            64-character".replace("\n            ", " "), reasons[0])

    def test_no_artifacts_at_all_is_refused(self) -> None:
        reasons = harness.checkpoint_reasons([])
        self.assertEqual(len(reasons), 1)
        self.assertIn("named no artifacts", reasons[0])

    def test_a_replaced_artifact_is_caught_by_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.safetensors"
            path.write_bytes(b"re-quantized in place")
            reasons = harness.checkpoint_reasons(
                [{"role": "target", "path": str(path), "sha256": "a" * 64}]
            )
            self.assertEqual(len(reasons), 1)
            self.assertIn("re-quantized or replaced in place", reasons[0])

    def test_a_matching_artifact_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "model.safetensors"
            payload = b"the weights the number was taken on"
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            self.assertEqual(
                harness.checkpoint_reasons(
                    [{"role": "target", "path": str(path), "sha256": digest}]
                ),
                [],
            )


class ContentionTest(unittest.TestCase):
    def test_a_run_with_no_lease_is_refused(self) -> None:
        reasons = harness.contention_reasons({"compute_processes": [], "own_pids": []})
        self.assertEqual(len(reasons), 1)
        self.assertIn("no lease id", reasons[0])

    def test_an_unsampled_process_list_is_refused(self) -> None:
        reasons = harness.contention_reasons({"lease_id": "rc-1"})
        self.assertEqual(len(reasons), 1)
        self.assertIn("compute-process list was not sampled", reasons[0])

    def test_a_foreign_job_on_the_device_is_refused(self) -> None:
        reasons = harness.contention_reasons(
            {
                "lease_id": "rc-1",
                "own_pids": [111],
                "compute_processes": [{"pid": 222, "name": "python3"}],
            }
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("foreign compute process", reasons[0])

    def test_a_leased_and_idle_device_passes(self) -> None:
        self.assertEqual(
            harness.contention_reasons(
                {"lease_id": "rc-1", "own_pids": [111], "compute_processes": []}
            ),
            [],
        )


class BuildRecipeTest(unittest.TestCase):
    def test_a_build_nobody_can_reconstruct_is_refused(self) -> None:
        reasons = harness.build_recipe_reasons({}, label="ours")
        self.assertEqual(len(reasons), 2)
        self.assertTrue(all("not reproducible" in reason for reason in reasons))

    def test_a_named_revision_and_recipe_pass(self) -> None:
        self.assertEqual(
            harness.build_recipe_reasons(
                {"revision": "e100e64e1", "build_recipe": "cmake --preset cuda-release"},
                label="ours",
            ),
            [],
        )


class HookTest(unittest.TestCase):
    """O23's abort-on-zero, which was the ONLY check that caught all three."""

    def test_a_blind_instrument_is_refused_even_with_hundreds_of_calls(self) -> None:
        reasons = harness.hook_reasons(
            {"propose_calls": 59, "skipped_dummy": 1, "skipped_capture": 0}, 0
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("ZERO draft blocks", reasons[0])

    def test_a_hook_on_the_wrong_process_is_named(self) -> None:
        reasons = harness.hook_reasons(
            {"propose_calls": 0, "skipped_dummy": 0, "skipped_capture": 0}, 0
        )
        self.assertEqual(len(reasons), 2)
        self.assertTrue(any("SEPARATE PROCESS" in reason for reason in reasons))

    def test_more_blocks_than_calls_is_refused_because_an_invented_record_is_impossible(self) -> None:
        reasons = harness.hook_reasons(
            {"propose_calls": 59, "skipped_dummy": 1, "skipped_capture": 0}, 59
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("invented one is not", reasons[0])

    def test_the_committed_triton_golden_residual_is_inside_the_one_sided_bound(self) -> None:
        # 59 propose calls - 1 skipped = 58, against 55 recorded blocks. The
        # residual of 3 is UNEXPLAINED and the bound is one-sided, so it passes.
        self.assertEqual(
            harness.hook_reasons(
                {"propose_calls": 59, "skipped_dummy": 1, "skipped_capture": 0}, 55
            ),
            [],
        )

    def test_missing_bookkeeping_is_refused(self) -> None:
        reasons = harness.hook_reasons({"propose_calls": 5}, 3)
        self.assertEqual(len(reasons), 2)


class ClockTest(unittest.TestCase):
    def test_an_unsampled_window_is_refused(self) -> None:
        reasons = harness.clock_state_reasons(None, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("quotable only", reasons[0])

    def test_a_window_nobody_watched_is_refused(self) -> None:
        record = clock_record()
        record["sm_clock_mhz"]["n"] = 1
        record["sm_clock_mhz"]["spread_pct"] = 0.0
        reasons = harness.clock_state_reasons(record, label="ours")
        self.assertTrue(any("below the 30 floor" in reason for reason in reasons))

    def test_two_arms_on_different_boots_are_not_one_comparison(self) -> None:
        reasons, block = harness.clock_pairing(
            clock_record(boot="boot-a"), clock_record(boot="boot-b")
        )
        self.assertTrue(any("DIFFERENT boots" in reason for reason in reasons))
        self.assertFalse(block["same_boot"])

    def test_the_cross_boot_override_records_a_caveat_rather_than_silence(self) -> None:
        reasons, block = harness.clock_pairing(
            clock_record(boot="boot-a"), clock_record(boot="boot-b"), allow_cross_boot=True
        )
        self.assertEqual(reasons, [])
        self.assertEqual(len(block["caveats"]), 1)

    def test_a_healthy_same_boot_pairing_passes(self) -> None:
        reasons, block = harness.clock_pairing(clock_record(), clock_record())
        self.assertEqual(reasons, [])
        self.assertEqual(block["median_offset_pct"], 0.0)


class AxisTest(unittest.TestCase):
    def test_a_lower_is_better_axis_is_oriented_so_above_one_is_better(self) -> None:
        self.assertAlmostEqual(harness.ratio(50.0, 100.0, polarity="lower_is_better"), 2.0)
        self.assertAlmostEqual(harness.ratio(100.0, 50.0, polarity="higher_is_better"), 2.0)

    def test_an_axis_below_its_floor_is_an_open_gap_and_never_a_ceiling(self) -> None:
        rows = harness.axis_rows(
            {"output_throughput_tok_s": 80.0},
            {"output_throughput_tok_s": 100.0},
            {"output_throughput_tok_s": 1.0},
        )
        row = next(row for row in rows if row["axis"] == "output_throughput_tok_s")
        self.assertEqual(row["verdict"], "OPEN GAP")
        for other in rows:
            self.assertNotIn("ceiling", other["verdict"].lower())

    def test_an_axis_neither_arm_measured_reads_NOT_MEASURED(self) -> None:
        rows = harness.axis_rows({}, {}, {})
        self.assertEqual({row["verdict"] for row in rows}, {"NOT MEASURED"})
        self.assertEqual(len(rows), len(harness.AXIS_POLARITY))


def arm_record(*, ours: bool) -> dict:
    workload = {
        "prompts_sha256": harness.workload_fingerprint(list(capture.PROMPTS)),
        "num_prompts": len(capture.PROMPTS),
        "max_tokens": 64,
        "num_speculative_tokens": 7,
        "max_num_seqs": 1,
        "concurrency": 1,
        "temperature": 0.0,
        "seed": None,
    }
    record = {
        "env": {harness.SSE_PING_ENV: "0"},
        "artifacts": [
            {"role": "target", "path": str(_ARTIFACT), "sha256": _ARTIFACT_SHA256}
        ],
        "contention": {"lease_id": "rc-42", "own_pids": [1], "compute_processes": []},
        "build": {"revision": "e100e64e1", "build_recipe": "cmake --preset cuda-release"},
        "clock": clock_record(),
        "workload": workload,
        "metrics": {"output_throughput_tok_s": 90.0 if ours else 100.0},
    }
    if not ours:
        record.update(
            {
                "oracle_runtime_version": GOOD_VERSION,
                "oracle_expected_commit": ORACLE_COMMIT,
                "attention_backend": "TRITON_ATTN",
                "attention_backend_declared": "TRITON_ATTN",
                "attention_backend_source": harness.BACKEND_SOURCE_READ_BACK,
                "config": {"enforce_eager": False},
            }
        )
    return record


class SpeedResultTest(unittest.TestCase):
    def good(self) -> tuple[dict, dict]:
        return arm_record(ours=True), arm_record(ours=False)

    def test_a_result_carries_every_axis_and_every_precondition(self) -> None:
        ours, theirs = self.good()
        result = harness.build_speed_result(ours=ours, theirs=theirs)
        self.assertEqual(len(result["axes"]), len(harness.AXIS_POLARITY))
        row = next(r for r in result["axes"] if r["axis"] == "output_throughput_tok_s")
        self.assertAlmostEqual(row["ratio"], 0.9)
        for field in ("oracle_runtime_version", "attention_backend_source", "sse_ping_s",
                      "enforce_eager", "workload", "artifacts", "build", "clock", "contention"):
            self.assertIn(field, result["preconditions"])

    def test_it_reports_EVERY_unmet_precondition_in_one_refusal(self) -> None:
        ours, theirs = self.good()
        theirs["oracle_runtime_version"] = "0.25.0"
        theirs["attention_backend_source"] = "corrected from the run log"
        theirs["config"] = {"enforce_eager": True}
        ours["env"] = {}
        ours["contention"] = {}
        with self.assertRaises(HarnessError) as raised:
            harness.build_speed_result(ours=ours, theirs=theirs)
        message = str(raised.exception)
        # One raise, many reasons: a harness that dies on the first defect
        # charges the operator a 51.75 GiB load per defect.
        self.assertGreaterEqual(message.count("\n  - "), 6)
        for fragment in ("PLAIN RELEASE", "post-hoc relabel", "AGENTS.md", "a default is not a record", "no lease id"):
            self.assertIn(fragment, message)

    def test_a_number_is_never_produced_without_a_clock(self) -> None:
        ours, theirs = self.good()
        ours["clock"] = None
        with self.assertRaises(HarnessError) as raised:
            harness.build_speed_result(ours=ours, theirs=theirs)
        self.assertIn("sampled no clock window", str(raised.exception))


class SummarizeEntryPointTest(unittest.TestCase):
    """The MODULE's own entry point, not a helper reached around it."""

    def test_summarize_writes_the_citable_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "ours.json").write_text(json.dumps(arm_record(ours=True)))
            (root / "vllm.json").write_text(json.dumps(arm_record(ours=False)))
            out = root / "result.json"
            rc = harness.main(
                [
                    "summarize",
                    "--ours", str(root / "ours.json"),
                    "--vllm", str(root / "vllm.json"),
                    "--output", str(out),
                    "--floor", "output_throughput_tok_s=1.0",
                ]
            )
            self.assertEqual(rc, 0)
            result = json.loads(out.read_text())
            row = next(r for r in result["axes"] if r["axis"] == "output_throughput_tok_s")
            self.assertEqual(row["verdict"], "OPEN GAP")
            self.assertEqual(result["issue"], "https://github.com/mudler/vllm.cpp/issues/1562")

    def test_summarize_refuses_a_relabelled_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            theirs = arm_record(ours=False)
            theirs["attention_backend_source"] = "corrected from the run log by w6-relabel.py"
            (root / "ours.json").write_text(json.dumps(arm_record(ours=True)))
            (root / "vllm.json").write_text(json.dumps(theirs))
            with self.assertRaises(HarnessError):
                harness.main(
                    ["summarize", "--ours", str(root / "ours.json"), "--vllm", str(root / "vllm.json")]
                )


def precheck_argv(root: pathlib.Path, digest: str, **overrides: object) -> list[str]:
    argv = [
        "--target", "/workspace/ckpt/qwen3.8-27b-hf",
        "--draft", "/workspace/dflash2/draft-st",
        "--oracle-commit", ORACLE_COMMIT,
        "--oracle-build-recipe", "pip install -e . at the beyond-pin head",
        "--attention-backend", "TRITON_ATTN",
        "--artifact", f"target={root / 'model.safetensors'}={digest}",
        "--lease-id", "rc-42",
        "--our-revision", "e100e64e1",
        "--our-build-recipe", "cmake --preset cuda-release",
        "--assume-compute-processes", str(overrides.get("processes", "[]")),
        "--precheck-only",
    ]
    return argv


class OracleCapturePrecheckTest(unittest.TestCase):
    """Driven through `main()`, which is the entry point the lease runs."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        payload = b"a 27B checkpoint stands in for itself"
        (self.root / "model.safetensors").write_bytes(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        self.env = {harness.SSE_PING_ENV: "0"}
        self._real_environ = capture.os.environ
        capture.os.environ = self.env  # type: ignore[assignment]

    def tearDown(self) -> None:
        capture.os.environ = self._real_environ  # type: ignore[assignment]
        self.tmp.cleanup()

    def test_a_complete_precheck_passes_and_prints_what_it_checked(self) -> None:
        rc = capture.main(precheck_argv(self.root, self.digest))
        self.assertEqual(rc, 0)

    def test_the_keepalive_must_be_set_explicitly(self) -> None:
        self.env.clear()
        with self.assertRaises(HarnessError) as raised:
            capture.main(precheck_argv(self.root, self.digest))
        self.assertIn("a default is not a record", str(raised.exception))

    def test_enforce_eager_stops_the_run_before_the_model_loads(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            capture.main(precheck_argv(self.root, self.digest) + ["--enforce-eager"])
        self.assertIn("AGENTS.md", str(raised.exception))

    def test_a_missing_lease_stops_the_run(self) -> None:
        argv = [item for item in precheck_argv(self.root, self.digest)]
        argv[argv.index("--lease-id") + 1] = ""
        with self.assertRaises(HarnessError) as raised:
            capture.main(argv)
        self.assertIn("no lease id", str(raised.exception))

    def test_a_foreign_job_on_the_device_stops_the_run(self) -> None:
        argv = precheck_argv(
            self.root, self.digest, processes='[{"pid": 999, "name": "another-lease"}]'
        )
        with self.assertRaises(HarnessError) as raised:
            capture.main(argv)
        self.assertIn("foreign compute process", str(raised.exception))

    def test_a_re_quantized_checkpoint_stops_the_run(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            capture.main(precheck_argv(self.root, "c" * 64))
        self.assertIn("re-quantized or replaced in place", str(raised.exception))

    def test_a_two_part_artifact_is_not_a_pin(self) -> None:
        argv = precheck_argv(self.root, self.digest)
        argv[argv.index("--artifact") + 1] = "target=/workspace/model.safetensors"
        with self.assertRaises(HarnessError) as raised:
            capture.main(argv)
        self.assertIn("A repo id is not a", str(raised.exception))

    def test_the_beyond_pin_head_is_required_and_never_defaulted(self) -> None:
        argv = precheck_argv(self.root, self.digest)
        argv[argv.index("--oracle-commit") + 1] = ""
        with self.assertRaises(HarnessError) as raised:
            capture.main(argv)
        self.assertIn("BEYOND-PIN", str(raised.exception))


class OurArmTest(unittest.TestCase):
    LINE = (
        "vllm-cli: run={run}/5 finish_reason=length prompt_tokens=5 "
        "completion_tokens=64 secs=1.234 tok_s={tps}\n"
    )

    def stderr(self, count: int = 5) -> str:
        return "".join(
            self.LINE.format(run=run, tps=40.0 + run) for run in range(1, count + 1)
        )

    def test_the_cli_format_string_is_parsed_positionally(self) -> None:
        legs = our_arm.parse_legs(self.stderr())
        self.assertEqual(len(legs), 5)
        self.assertEqual(legs[0]["completion_tokens"], 64)
        self.assertAlmostEqual(legs[4]["tok_s"], 45.0)

    def test_a_null_parse_is_a_refusal_and_never_an_empty_average(self) -> None:
        reasons = our_arm.leg_reasons(our_arm.parse_legs("vllm-cli: loading model\n"), max_tokens=64)
        self.assertEqual(len(reasons), 1)
        self.assertIn("A null\nparse".replace("\n", " "), reasons[0])

    def test_an_arm_that_produced_fewer_tokens_did_not_do_the_same_work(self) -> None:
        text = self.stderr().replace("completion_tokens=64", "completion_tokens=32", 1)
        reasons = our_arm.leg_reasons(our_arm.parse_legs(text), max_tokens=64)
        self.assertEqual(len(reasons), 1)
        self.assertIn("did not do the same amount of work", reasons[0])

    def test_one_leg_is_an_anecdote(self) -> None:
        reasons = our_arm.leg_reasons(our_arm.parse_legs(self.stderr(1)), max_tokens=64)
        self.assertTrue(any("anecdote" in reason for reason in reasons))

    def test_the_cold_leg_is_discarded_for_a_NAMED_cause_and_retained(self) -> None:
        folded = our_arm.fold_legs(our_arm.parse_legs(self.stderr()))
        self.assertEqual(folded["cold_legs_discarded"], 1)
        self.assertEqual(len(folded["legs"]), 5)
        self.assertIn("named cause", folded["cold_discard_cause"])
        # median of 42, 43, 44, 45 -- the cold 41 is not in it.
        self.assertAlmostEqual(folded["metrics"]["output_throughput_tok_s"], 43.5)


class ShellDriverTest(unittest.TestCase):
    SCRIPT = ROOT / "scripts/dflash2-speed-gate.sh"

    def test_the_script_parses(self) -> None:
        completed = subprocess.run(["bash", "-n", str(self.SCRIPT)], capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())

    def test_self_check_touches_no_gpu_and_passes(self) -> None:
        completed = subprocess.run(
            ["bash", str(self.SCRIPT), "--self-check"], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("self-check PASS", completed.stdout)

    def test_the_marker_is_written_from_the_TRAP_on_a_FAILING_path(self) -> None:
        """Seven waiters once blocked forever on a marker only success wrote."""

        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "evidence"
            payload = b"stand-in weights"
            artifact = pathlib.Path(tmp) / "model.safetensors"
            artifact.write_bytes(payload)
            completed = subprocess.run(
                [
                    "bash", str(self.SCRIPT),
                    "--evidence", str(evidence),
                    "--target", "/nonexistent/target",
                    "--draft", "/nonexistent/draft",
                    "--oracle-commit", ORACLE_COMMIT,
                    "--attention-backend", "TRITON_ATTN",
                    "--artifact", f"target={artifact}={hashlib.sha256(payload).hexdigest()}",
                    "--our-binary", "/nonexistent/vllm-cli",
                    "--our-model", "/nonexistent/model",
                    "--our-build-recipe", "cmake --preset cuda-release",
                    "--lease-id", "rc-42",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            # It must FAIL here: there is no GPU on a CI runner, so the
            # contention sample cannot be taken and the harness refuses.
            self.assertNotEqual(completed.returncode, 0)
            marker = evidence / "COMPLETED"
            self.assertTrue(marker.is_file(), "the trap did not write the marker")
            text = marker.read_text()
            self.assertIn("status=", text)
            self.assertNotIn("status=0", text)

    def test_a_run_with_no_lease_never_reaches_the_gpu(self) -> None:
        completed = subprocess.run(
            [
                "bash", str(self.SCRIPT),
                "--evidence", "/tmp/unused-dflash2-evidence",
                "--target", "t", "--draft", "d",
                "--oracle-commit", ORACLE_COMMIT,
                "--attention-backend", "TRITON_ATTN",
                "--artifact", "target=/x=" + "0" * 64,
                "--our-binary", "b", "--our-model", "m",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("never ssh to a fleet box", completed.stderr)


if __name__ == "__main__":
    unittest.main()
