#!/usr/bin/env python3
"""Gate the SPEC-DFLASH2 harness on the paths that REFUSE (#1562).

A benchmark harness is not proven by a run that produced a number. It is proven
by the runs it DECLINED to turn into a number, because those are the paths that
protect every future measurement, and they are the paths a green run never
exercises.

So most cases here drive a refusal, through the same functions and the same
`main()` entry points the leased run calls. The rest -- the ones below the
banner near the end of this file -- drive a COMPLETE run of an arm instead,
because the mutations that make its number WRONG rather than absent are
invisible to a case that only ever refuses. Nothing constructs a private helper
by hand: `AGENTS.md` §Nothing lands dead says a unit test that builds the type
itself proves the class works and never that anything reaches it, so the
precheck cases run `dflash2_oracle_capture.main(["--precheck-only", ...])` and
the marker case runs `scripts/dflash2-speed-gate.sh` as a process.

Standard library only, no GPU, no `nvidia-smi`, no vLLM wheel.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
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

    PROBE = harness.BACKEND_PROBES[0]

    def test_the_1562_post_hoc_relabel_is_refused_by_its_source(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="FLASH_ATTN",
            declared="FLASH_ATTN",
            source=self.RELABEL,
            probe=self.PROBE,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("post-hoc relabel", reasons[0])

    def test_an_unread_backend_is_refused(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="",
            declared="TRITON_ATTN",
            source=harness.BACKEND_SOURCE_READ_BACK,
            probe=self.PROBE,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("no resolved attention backend", reasons[0])

    def test_a_substituted_denominator_is_refused(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="FLASH_ATTN",
            declared="TRITON_ATTN",
            source=harness.BACKEND_SOURCE_READ_BACK,
            probe=self.PROBE,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("substituted denominator", reasons[0])

    def test_a_read_back_label_that_agrees_passes(self) -> None:
        self.assertEqual(
            harness.attention_backend_reasons(
                resolved="TRITON_ATTN",
                declared="TRITON_ATTN",
                source=harness.BACKEND_SOURCE_READ_BACK,
                probe=self.PROBE,
            ),
            [],
        )

    def test_a_label_with_no_probe_is_refused_because_nothing_says_WHERE(self) -> None:
        """`attention_backend_probe` was captured and never checked."""

        reasons = harness.attention_backend_reasons(
            resolved="TRITON_ATTN",
            declared="TRITON_ATTN",
            source=harness.BACKEND_SOURCE_READ_BACK,
            probe=None,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("no read-back probe was recorded", reasons[0])

    def test_a_probe_this_module_does_not_admit_is_refused(self) -> None:
        reasons = harness.attention_backend_reasons(
            resolved="TRITON_ATTN",
            declared="TRITON_ATTN",
            source=harness.BACKEND_SOURCE_READ_BACK,
            probe="llm.please_just_say_TRITON_ATTN",
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("is not one of the", reasons[0])

    def test_the_ONLY_emitter_of_the_admitted_source_is_the_read_back(self) -> None:
        """The bind is on the SHAPE of the claim, so pin what shapes it."""

        source = pathlib.Path(capture.__file__).read_text(encoding="utf-8")
        harness_source = pathlib.Path(harness.__file__).read_text(encoding="utf-8")
        emitters = source.count("BACKEND_SOURCE_READ_BACK")
        self.assertGreaterEqual(emitters, 1)
        # The constant is DEFINED once and CHECKED once in the harness; every
        # other mention is this module's read-back path.
        self.assertEqual(harness_source.count('BACKEND_SOURCE_READ_BACK = '), 1)


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
            "repeat": 5,
            "concurrency": 1,
            "temperature": 0.0,
            "seed": None,
        }

    def test_two_arms_that_repeated_differently_folded_different_populations(self) -> None:
        theirs = self.workload()
        theirs["repeat"] = 2
        reasons = harness.workload_reasons(self.workload(), theirs)
        self.assertEqual(len(reasons), 1)
        self.assertIn("repeat differs", reasons[0])

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
        "repeat": 5,
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
        # The artifact IS the model this arm loaded. Binding them is the check
        # that stops an arm hashing a file it never opened.
        "models": {"model" if ours else "target": str(_ARTIFACT)},
        "clock": clock_record(),
        "workload": workload,
        "metrics": {"output_throughput_tok_s": 90.0 if ours else 100.0},
    }
    if ours:
        record["build"] = {
            "revision": "e100e64e1",
            "build_recipe": "cmake --preset cuda-release",
        }
        record["speculative_config"] = {
            "method": "dflash",
            "model": "/workspace/dflash2/draft.gguf",
            "num_speculative_tokens": 7,
        }
    else:
        record.update(
            {
                "oracle_runtime_version": GOOD_VERSION,
                "oracle_expected_commit": ORACLE_COMMIT,
                "attention_backend": "TRITON_ATTN",
                "attention_backend_declared": "TRITON_ATTN",
                "attention_backend_source": harness.BACKEND_SOURCE_READ_BACK,
                "attention_backend_probe": harness.BACKEND_PROBES[0],
                "config": {"enforce_eager": False},
                # THE ORACLE'S OWN BUILD, not the harness's checkout. This
                # fixture carried `cmake --preset cuda-release` at revision
                # `e100e64e1` -- our tree -- against a pip-installed wheel.
                "build": {
                    "revision": ORACLE_COMMIT,
                    "build_recipe": "pip install -e . at the beyond-pin head",
                },
                "harness_build": {
                    "revision": "e100e64e1",
                    "build_recipe": "cmake --preset cuda-release",
                },
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
    # `--target` is the DIRECTORY the artifact lives in, because the oracle arm
    # now binds every model path it loads to an artifact entry and refuses when
    # nothing names it. `--draft` is bound the same way.
    argv = [
        "--target", str(root),
        "--draft", str(root / "draft"),
        "--oracle-commit", ORACLE_COMMIT,
        "--oracle-build-recipe", "pip install -e . at the beyond-pin head",
        "--attention-backend", "TRITON_ATTN",
        "--artifact", f"target={root / 'model.safetensors'}={digest}",
        "--artifact", f"draft={root / 'draft' / 'draft.safetensors'}={overrides.get('draft_digest', digest)}",
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
        (self.root / "draft").mkdir()
        (self.root / "draft" / "draft.safetensors").write_bytes(payload)
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

    def test_the_recorded_vllm_BUILD_is_the_oracles_and_not_this_checkouts(self) -> None:
        """The capture runs from OUR tree, so our revision is nearest to hand."""

        args = capture.build_parser().parse_args(precheck_argv(self.root, self.digest))
        checked = capture.precheck(args, self.env)
        self.assertEqual(checked["build"]["revision"], ORACLE_COMMIT)
        self.assertEqual(checked["build"]["build_recipe"], "pip install -e . at the beyond-pin head")
        # Our checkout is provenance for the INSTRUMENT and is recorded as such.
        self.assertEqual(checked["harness_build"]["revision"], "e100e64e1")
        self.assertEqual(
            harness.oracle_build_reasons(
                {"oracle_expected_commit": ORACLE_COMMIT, "build": checked["build"]}
            ),
            [],
        )

    def test_the_vllm_build_revision_TRACKS_the_oracle_commit_flag(self) -> None:
        """It cannot disagree by construction, and that IS the repair.

        `oracle_build_reasons` runs inside this precheck as well, so a future
        edit that refills the block from `--our-revision` refuses here rather
        than surfacing as a mislabelled recipe in a result nobody rechecks.
        """

        other = "b" * 40
        argv = precheck_argv(self.root, self.digest)
        argv[argv.index("--oracle-commit") + 1] = other
        args = capture.build_parser().parse_args(argv)
        checked = capture.precheck(args, self.env)
        self.assertEqual(checked["build"]["revision"], other)
        self.assertNotEqual(checked["build"]["revision"], checked["harness_build"]["revision"])

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

    def test_the_named_cause_is_the_REAL_one_on_OUR_arm(self) -> None:
        """The OUR-ARM half of the cause, pinned from our arm's own legs.

        It once said run 1 "loads once" with no arm named, which is false for
        three of the four oracle legs. Our arm starts one process per prompt, so
        for it the claim holds, and this case can see that arm.

        It cannot see the other one. The oracle half of the same sentence is
        bound to a RUN rather than to wording, by
        `test_the_recorded_DISCARD_CAUSE_matches_what_this_arm_actually_DOES` in
        `CaptureRunTest`. A grep here could not have detected the cause drifting
        from what `capture()` does, which is exactly how the previous cause
        shipped a false clause into every `vllm-arm.json`.
        """

        cause = our_arm.fold_legs(our_arm.parse_legs(self.stderr()))["cold_discard_cause"]
        self.assertIn("one process per prompt", cause)
        self.assertIn("loads the model", cause)
        # The one-arm claim is gone: it is now attributed to the arm it holds on.
        self.assertNotIn("run 1 of each repetition group loads once", cause)


class SpeculativeConfigTest(unittest.TestCase):
    """The one workload key this ROW is about, and nothing reconciled it."""

    CONFIG = '{"method":"dflash","model":"/d.gguf","num_speculative_tokens":3}'

    def test_no_config_at_all_is_refused_because_that_is_a_PLAIN_decode(self) -> None:
        reasons, config = harness.speculative_config_reasons("", 7, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("speculative decoding OFF", reasons[0])
        self.assertEqual(config, {})

    def test_the_config_and_the_flag_disagreeing_is_refused_naming_BOTH(self) -> None:
        reasons, config = harness.speculative_config_reasons(self.CONFIG, 7, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("k=7", reasons[0])
        self.assertIn("k=3", reasons[0])
        self.assertEqual(config["num_speculative_tokens"], 3)

    def test_a_config_with_no_k_records_a_depth_nobody_can_compare(self) -> None:
        reasons, _ = harness.speculative_config_reasons(
            '{"method":"dflash","model":"/d.gguf"}', 7, label="ours"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("names no", reasons[0])

    def test_a_config_that_is_not_json_is_refused(self) -> None:
        reasons, _ = harness.speculative_config_reasons("{not json", 7, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("not JSON", reasons[0])

    def test_the_config_and_the_flag_agreeing_passes(self) -> None:
        reasons, config = harness.speculative_config_reasons(self.CONFIG, 3, label="ours")
        self.assertEqual(reasons, [])
        self.assertEqual(config["model"], "/d.gguf")


class ModelBindingTest(unittest.TestCase):
    """A measured hash beside a path the run never opened identifies nothing."""

    ARTIFACTS = [
        {"role": "target", "path": "/ckpt/qwen/model.safetensors", "sha256": "0" * 64}
    ]

    def test_a_model_no_artifact_names_is_refused(self) -> None:
        reasons = harness.model_binding_reasons(
            {"model": "/other/target.gguf"}, self.ARTIFACTS, label="ours"
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("no --artifact entry names it", reasons[0])

    def test_an_artifact_INSIDE_the_model_directory_binds_it(self) -> None:
        self.assertEqual(
            harness.model_binding_reasons({"target": "/ckpt/qwen"}, self.ARTIFACTS, label="vllm"),
            [],
        )

    def test_an_artifact_that_IS_the_model_file_binds_it(self) -> None:
        self.assertEqual(
            harness.model_binding_reasons(
                {"model": "/ckpt/qwen/model.safetensors"}, self.ARTIFACTS, label="ours"
            ),
            [],
        )

    def test_a_model_path_that_is_empty_is_refused_rather_than_skipped(self) -> None:
        reasons = harness.model_binding_reasons({"drafter": ""}, self.ARTIFACTS, label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("named no drafter path", reasons[0])


class OracleBuildTest(unittest.TestCase):
    """The denominator's build block described the HARNESS's checkout."""

    def test_our_revision_in_the_vllm_arms_build_is_refused(self) -> None:
        reasons = harness.oracle_build_reasons(
            {
                "oracle_expected_commit": ORACLE_COMMIT,
                "build": {
                    "revision": "e100e64e1",
                    "build_recipe": "cmake --preset cuda-release",
                },
            }
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("describes the harness instead of the denominator", reasons[0])

    def test_the_oracle_head_in_its_own_build_block_passes(self) -> None:
        self.assertEqual(
            harness.oracle_build_reasons(
                {
                    "oracle_expected_commit": ORACLE_COMMIT,
                    "build": {
                        "revision": ORACLE_COMMIT,
                        "build_recipe": "pip install -e .",
                    },
                }
            ),
            [],
        )

    def test_a_short_prefix_of_the_oracle_head_still_agrees(self) -> None:
        self.assertEqual(
            harness.oracle_build_reasons(
                {
                    "oracle_expected_commit": ORACLE_COMMIT,
                    "build": {
                        "revision": ORACLE_COMMIT[:12],
                        "build_recipe": "pip install -e .",
                    },
                }
            ),
            [],
        )


class _FakeTensor:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def tolist(self) -> list:
        return list(self._rows)

    def __getitem__(self, index: int):
        return self._rows[index]


class _FakeCuda:
    def __init__(self) -> None:
        self.capturing = False
        self.calls = 0

    def is_current_stream_capturing(self) -> bool:
        self.calls += 1
        return self.capturing


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeSpeculator:
    """Just the two attributes the anchor walk reads."""

    def __init__(self, anchors: list[int] | None, drafts: list[list[int]]) -> None:
        self.drafts = drafts
        if anchors is not None:
            self._anchor_indices = _FakeTensor(list(range(len(anchors))))
            self.input_buffers = type("B", (), {"input_ids": list(anchors)})()

    def propose(self, *args, **kwargs):
        return _FakeTensor(self.drafts)


class DraftRecorderShapeTest(unittest.TestCase):
    """The capture must emit the shape the GOLDEN CONSUMER reads.

    `tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads
    `records[i]["blocks"][b]["anchor"]` and `["drafts"]`. The recorder wrote a
    FLAT top-level `blocks` list with no anchor, so its output could not become
    the golden O26 says this run exists to produce.
    """

    def recorder(self, anchors: list[int] | None = None) -> tuple:
        torch_mod = _FakeTorch()
        klass = type("Spec", (_FakeSpeculator,), {})
        rec = capture.DraftRecorder()
        rec.install(klass, torch_mod)
        return rec, klass, torch_mod

    def test_blocks_are_grouped_UNDER_the_open_record_and_carry_the_anchor(self) -> None:
        rec, klass, _ = self.recorder()
        first = klass([11], [[1, 2, 3]])
        second = klass([22], [[4, 5, 6]])
        rec.open_record(0)
        first.propose()
        rec.close_record()
        rec.open_record(1)
        second.propose()
        rec.close_record()

        zero, one = rec.blocks_for(0), rec.blocks_for(1)
        self.assertEqual(len(zero), 1)
        self.assertEqual(len(one), 1)
        self.assertEqual(zero[0]["anchor"], 11)
        self.assertEqual(one[0]["anchor"], 22)
        self.assertEqual(zero[0]["drafts"], [1, 2, 3])
        self.assertEqual(zero[0]["req_row"], 0)
        # `record` is bookkeeping and never reaches the emitted golden.
        self.assertNotIn("record", zero[0])
        self.assertEqual(rec.stats()["anchor_misses"], 0)

    def test_a_moved_anchor_attribute_is_COUNTED_and_never_invented(self) -> None:
        rec, klass, _ = self.recorder()
        blind = klass(None, [[1, 2, 3]])
        rec.open_record(0)
        blind.propose()
        rec.close_record()
        self.assertEqual(rec.stats()["anchor_misses"], 1)
        self.assertNotIn("anchor", rec.blocks_for(0)[0])

    def test_a_capture_that_missed_EVERY_anchor_is_refused(self) -> None:
        reasons = harness.hook_reasons(
            {
                "propose_calls": 10,
                "skipped_dummy": 0,
                "skipped_capture": 0,
                "anchor_misses": 4,
            },
            4,
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("pairs blocks by ORDINAL", reasons[0])

    def test_the_committed_goldens_predate_the_field_and_stay_admissible(self) -> None:
        """`anchor_misses` is optional; the W6 `hook_stats` carry no such key."""

        self.assertEqual(
            harness.hook_reasons(
                {"propose_calls": 59, "skipped_dummy": 1, "skipped_capture": 0}, 55
            ),
            [],
        )

    def test_a_graph_capture_delegates_and_records_NOTHING(self) -> None:
        rec, klass, torch_mod = self.recorder()
        torch_mod.cuda.capturing = True
        klass([11], [[1, 2, 3]]).propose()
        self.assertEqual(rec.stats()["skipped_capture"], 1)
        self.assertEqual(rec.blocks, [])


class GoldenShapeTest(unittest.TestCase):
    """Every top-level key the parity consumer reads, read OUT of that file."""

    CONSUMER = ROOT / "tests/parity/test_qwen38_dflash2_spec_decode.cpp"

    def consumed_keys(self) -> set[str]:
        import re

        text = self.CONSUMER.read_text(encoding="utf-8")
        return set(
            re.findall(r'golden\.(?:at|value|contains)\("([a-z_]+)"', text)
        )

    def test_the_declared_contract_covers_what_the_consumer_reads(self) -> None:
        consumed = self.consumed_keys()
        self.assertGreaterEqual(len(consumed), 8, "the anchor regex found nothing")
        missing = sorted(consumed - set(capture.GOLDEN_TOP_LEVEL_KEYS))
        self.assertEqual(missing, [], f"GOLDEN_TOP_LEVEL_KEYS omits {missing}")

    def test_the_envelope_EMITS_every_key_the_contract_declares(self) -> None:
        args = capture.build_parser().parse_args(
            [
                "--target", "/t", "--draft", "/d",
                "--oracle-commit", ORACLE_COMMIT,
                "--attention-backend", "TRITON_ATTN",
            ]
        )
        envelope = capture.golden_envelope(
            args, runtime_version=GOOD_VERSION, oracle_file="/w/vllm/__init__.py"
        )
        # `records`, `metrics`, `hook_stats` and `attention_backend` come from
        # the run rather than the envelope; the rest are the envelope's.
        from_run = {"records", "metrics", "hook_stats", "attention_backend"}
        for key in capture.GOLDEN_TOP_LEVEL_KEYS:
            if key in from_run:
                continue
            self.assertIn(key, envelope, f"the envelope omits {key}")
        self.assertEqual(envelope["spec"], "on")
        self.assertEqual(envelope["oracle_version"], GOOD_VERSION)


class SharedLegFoldTest(unittest.TestCase):
    """Both arms fold the SAME statistic, or the quotient is not a ratio."""

    def test_our_arm_re_exports_the_harness_rule_rather_than_owning_one(self) -> None:
        self.assertIs(our_arm.fold_legs, harness.fold_legs)
        self.assertIs(our_arm.leg_reasons, harness.leg_reasons)

    def test_the_oracle_arm_folds_through_the_same_function(self) -> None:
        self.assertIs(capture.fold_legs, harness.fold_legs)
        self.assertIs(capture.leg_reasons, harness.leg_reasons)

    def test_repeat_one_is_refused_BEFORE_the_lease_on_both_arms(self) -> None:
        for label in ("ours", "vllm"):
            reasons = harness.repeat_reasons(1, label=label)
            self.assertEqual(len(reasons), 1)
            self.assertIn("no\nwarm leg".replace("\n", " "), reasons[0])
        self.assertEqual(harness.repeat_reasons(5, label="ours"), [])

    def test_a_non_integer_repeat_is_refused_rather_than_coerced(self) -> None:
        reasons = harness.repeat_reasons("many", label="ours")
        self.assertEqual(len(reasons), 1)
        self.assertIn("not an integer", reasons[0])

    def test_legs_that_are_ALL_cold_leave_nothing_to_fold(self) -> None:
        with self.assertRaises(HarnessError):
            harness.fold_legs([{"run": 1, "tok_s": 40.0, "completion_tokens": 64, "secs": 1.0}])


def our_arm_argv(root: pathlib.Path, digest: str, **overrides: object) -> list[str]:
    config = overrides.get(
        "config",
        '{"method":"dflash","model":"' + str(root / "draft.gguf") + '","num_speculative_tokens":7}',
    )
    argv = [
        "--binary", "/nonexistent/vllm-cli",
        "--model", str(root / "target.gguf"),
        "--artifact", f"our_target={root / 'target.gguf'}={digest}",
        "--artifact", f"our_draft={root / 'draft.gguf'}={digest}",
        "--lease-id", "rc-42",
        "--our-revision", "e100e64e1",
        "--our-build-recipe", "cmake --preset cuda-release",
        "--assume-compute-processes", str(overrides.get("processes", "[]")),
        "--precheck-only",
    ]
    if config:
        argv += ["--speculative-config", str(config)]
    return argv


class OurArmPrecheckEntryPointTest(unittest.TestCase):
    """Driven through `our_arm.main()`, which nothing drove before.

    The suite reached `parse_legs`, `leg_reasons` and `fold_legs` and never the
    module's own entry point, so deleting `require_no_reasons` from
    `our_arm.precheck` left all 63 cases green -- the refusal call site this arm
    depends on was unreached by the gate that claims to prove it.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        payload = b"a GGUF target stands in for itself"
        (self.root / "target.gguf").write_bytes(payload)
        (self.root / "draft.gguf").write_bytes(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        self.env = {harness.SSE_PING_ENV: "0"}
        self._real_environ = our_arm.os.environ
        our_arm.os.environ = self.env  # type: ignore[assignment]

    def tearDown(self) -> None:
        our_arm.os.environ = self._real_environ  # type: ignore[assignment]
        self.tmp.cleanup()

    def test_a_complete_precheck_passes(self) -> None:
        self.assertEqual(our_arm.main(our_arm_argv(self.root, self.digest)), 0)

    def test_the_keepalive_must_be_set_explicitly(self) -> None:
        self.env.clear()
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(our_arm_argv(self.root, self.digest))
        self.assertIn("a default is not a record", str(raised.exception))

    def test_a_missing_lease_stops_the_run(self) -> None:
        argv = our_arm_argv(self.root, self.digest)
        argv[argv.index("--lease-id") + 1] = ""
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(argv)
        self.assertIn("no lease id", str(raised.exception))

    def test_a_foreign_job_on_the_device_stops_the_run(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(
                our_arm_argv(
                    self.root,
                    self.digest,
                    processes='[{"pid": 999, "name": "another-lease"}]',
                )
            )
        self.assertIn("foreign compute process", str(raised.exception))

    def test_a_re_quantized_checkpoint_stops_the_run(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(our_arm_argv(self.root, "c" * 64))
        self.assertIn("re-quantized or replaced in place", str(raised.exception))

    def test_running_with_NO_drafter_stops_the_run(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(our_arm_argv(self.root, self.digest, config=""))
        self.assertIn("speculative decoding OFF", str(raised.exception))

    def test_the_RECORDED_k_comes_from_the_CONFIG_and_not_from_the_flag(self) -> None:
        """Executed: the config said 3 and the record said 7."""

        argv = our_arm_argv(
            self.root,
            self.digest,
            config='{"method":"dflash","model":"'
            + str(self.root / "draft.gguf")
            + '","num_speculative_tokens":3}',
        )
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(argv)
        message = str(raised.exception)
        self.assertIn("k=7", message)
        self.assertIn("k=3", message)

        argv += ["--num-speculative-tokens", "3"]
        self.assertEqual(our_arm.main(argv), 0)
        args = our_arm.build_parser().parse_args(argv)
        checked = our_arm.precheck(args, self.env)
        self.assertEqual(checked["workload"]["num_speculative_tokens"], 3)

    def test_a_model_no_artifact_names_stops_the_run(self) -> None:
        argv = our_arm_argv(self.root, self.digest)
        argv[argv.index("--model") + 1] = str(self.root / "some-other-target.gguf")
        with self.assertRaises(HarnessError) as raised:
            our_arm.main(argv)
        self.assertIn("no --artifact entry names it", str(raised.exception))


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
        """Seven waiters once blocked forever on a marker only success wrote.

        `--assume-compute-processes` is passed so this case keeps the promise
        the module docstring makes. Without it the driver reaches
        `sample_compute_processes`, which shells out to `nvidia-smi`, and on a
        host that HAS one the harness then opens a background clock sampler --
        and preflight runs exactly on such hosts. The failure was therefore
        environment-dependent in the one direction nobody checks: green on the
        CI runner that has no driver, and a GPU touch on the developer box.

        The refusal it fails on is the ABSENT `--oracle-build-recipe`, which is
        a precheck reason, deterministic on every host, and unrelated to the
        checks this case is not about.
        """

        with tempfile.TemporaryDirectory() as tmp:
            evidence = pathlib.Path(tmp) / "evidence"
            payload = b"stand-in weights"
            artifact = pathlib.Path(tmp) / "model.safetensors"
            artifact.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            completed = subprocess.run(
                [
                    "bash", str(self.SCRIPT),
                    "--evidence", str(evidence),
                    "--target", tmp,
                    "--draft", tmp,
                    "--oracle-commit", ORACLE_COMMIT,
                    "--attention-backend", "TRITON_ATTN",
                    "--artifact", f"target={artifact}={digest}",
                    "--our-binary", "/nonexistent/vllm-cli",
                    "--our-model", tmp,
                    "--our-artifact", f"our_target={artifact}={digest}",
                    "--our-speculative-config",
                    '{"method":"dflash","model":"' + tmp + '","num_speculative_tokens":7}',
                    "--our-build-recipe", "cmake --preset cuda-release",
                    "--lease-id", "rc-42",
                    "--assume-compute-processes", "[]",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("build_recipe", completed.stderr)
            marker = evidence / "COMPLETED"
            self.assertTrue(marker.is_file(), "the trap did not write the marker")
            text = marker.read_text()
            self.assertIn("status=", text)
            self.assertNotIn("status=0", text)

    def test_the_self_check_parses_EVERY_dflash2_tool_module(self) -> None:
        """Two of three were listed, so garbage in the third still printed PASS."""

        text = self.SCRIPT.read_text(encoding="utf-8")
        marker = 'if [ "${SELF_CHECK}" -eq 1 ]; then'
        self.assertIn(marker, text)
        block = text.split(marker, 1)[1].split("exit 0", 1)[0]
        modules = sorted(
            path.name
            for path in (ROOT / "tools/bench").glob("dflash2_*.py")
            if not path.name.startswith("_")
        )
        self.assertGreaterEqual(len(modules), 3)
        missing = [name for name in modules if f"tools/bench/{name}" not in block]
        self.assertEqual(missing, [], f"the self-check never parses {missing}")

    def test_a_spec_decode_comparison_needs_a_DRAFTER_on_our_side(self) -> None:
        """`--our-speculative-config` was optional, so our arm could run plain."""

        completed = subprocess.run(
            [
                "bash", str(self.SCRIPT),
                "--evidence", "/tmp/unused-dflash2-evidence-2",
                "--target", "t", "--draft", "d",
                "--oracle-commit", ORACLE_COMMIT,
                "--attention-backend", "TRITON_ATTN",
                "--artifact", "target=/x=" + "0" * 64,
                "--our-artifact", "our_target=/y=" + "0" * 64,
                "--our-binary", "b", "--our-model", "m",
                "--lease-id", "rc-42",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--our-speculative-config is required", completed.stderr)

    def test_one_artifact_list_cannot_identify_two_different_checkpoints(self) -> None:
        completed = subprocess.run(
            [
                "bash", str(self.SCRIPT),
                "--evidence", "/tmp/unused-dflash2-evidence-3",
                "--target", "t", "--draft", "d",
                "--oracle-commit", ORACLE_COMMIT,
                "--attention-backend", "TRITON_ATTN",
                "--artifact", "target=/x=" + "0" * 64,
                "--our-binary", "b", "--our-model", "m",
                "--lease-id", "rc-42",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--our-artifact", completed.stderr)

    def test_OUR_ARMS_preconditions_are_checked_BEFORE_the_oracle_LOADS(self) -> None:
        """The banner said "before the lease"; only one arm was prechecked.

        This repair split the artifact lists, so our arm carries its own
        `checkpoint_reasons`, `model_binding_reasons` and
        `speculative_config_reasons`. With only the oracle prechecked those were
        first evaluated after the oracle's full load and its whole timed run --
        the exact cost the precheck phase exists to avoid.

        The defect here is OURS ALONE and the oracle side is complete, so the
        oracle precheck must PASS and the run must still stop with no oracle
        evidence written.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            evidence = root / "evidence"
            payload = b"stand-in weights"
            artifact = root / "model.safetensors"
            artifact.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            (root / "elsewhere").mkdir()
            completed = subprocess.run(
                [
                    "bash", str(self.SCRIPT),
                    "--evidence", str(evidence),
                    "--target", tmp,
                    "--draft", tmp,
                    "--oracle-commit", ORACLE_COMMIT,
                    "--oracle-build-recipe", "pip install -e . at the beyond-pin head",
                    "--attention-backend", "TRITON_ATTN",
                    "--artifact", f"target={artifact}={digest}",
                    "--our-binary", "/nonexistent/vllm-cli",
                    # NAMED BY NO --our-artifact ENTRY, and by nothing the
                    # oracle's own list can bind either.
                    "--our-model", str(root / "elsewhere"),
                    "--our-artifact", f"our_target={artifact}={digest}",
                    "--our-speculative-config",
                    '{"method":"dflash","model":"' + str(artifact) + '","num_speculative_tokens":7}',
                    "--our-build-recipe", "cmake --preset cuda-release",
                    "--lease-id", "rc-42",
                    "--assume-compute-processes", "[]",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("DFlash2 our-arm capture REFUSED", completed.stderr)
            self.assertIn("no --artifact entry names it", completed.stderr)
            # The ORACLE side was complete, so its precheck ran and passed...
            self.assertTrue((evidence / "precheck.json").is_file())
            # ...and nothing beyond the precheck phase ever started.
            self.assertFalse((evidence / "vllm-arm.json").exists())
            self.assertFalse((evidence / "clock-vllm.json").exists())

    def test_the_gate_can_run_at_a_k_OTHER_than_the_capture_default(self) -> None:
        """It exposed no k, so it could only ever run at the default 7.

        Any other k in `--our-speculative-config` made the two workload
        fingerprints disagree and `summarize` refused. That is the right failure
        and it is still a gate that cannot sweep the one knob this row's
        acceptance length turns on. The precheck object records what was
        threaded, so this reads it back out of both arms' evidence.
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            evidence = root / "evidence"
            payload = b"stand-in weights"
            artifact = root / "model.safetensors"
            artifact.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            subprocess.run(
                [
                    "bash", str(self.SCRIPT),
                    "--evidence", str(evidence),
                    "--target", tmp,
                    "--draft", tmp,
                    "--oracle-commit", ORACLE_COMMIT,
                    "--oracle-build-recipe", "pip install -e . at the beyond-pin head",
                    "--attention-backend", "TRITON_ATTN",
                    "--artifact", f"target={artifact}={digest}",
                    "--our-binary", "/nonexistent/vllm-cli",
                    "--our-model", str(artifact),
                    "--our-artifact", f"our_target={artifact}={digest}",
                    "--our-speculative-config",
                    '{"method":"dflash","model":"' + str(artifact) + '","num_speculative_tokens":3}',
                    "--our-build-recipe", "cmake --preset cuda-release",
                    "--num-speculative-tokens", "3",
                    "--lease-id", "rc-42",
                    "--assume-compute-processes", "[]",
                ],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
            for name in ("precheck.json", "precheck-ours.json"):
                checked = json.loads((evidence / name).read_text(encoding="utf-8"))
                self.assertEqual(
                    checked["workload"]["num_speculative_tokens"], 3, name
                )

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


# --------------------------------------------------------------------------
# The CAPTURE phase, driven end to end against a stand-in `vllm` and `torch`.
#
# Everything above this line drives a REFUSAL or a pure function. `capture()`
# itself -- the loop that times the run, counts the tokens and threads the
# read-back probe -- was executed by nothing, and the two mutations that make
# its number WRONG rather than absent both stayed green. A wrong number is
# worse than an absent one, so the arm is run here.
# --------------------------------------------------------------------------

#: The stand-in advances a clock the test owns rather than sleeping, so
#: "the model load is outside the timed span" is an EXACT assertion.
LOAD_SECS = 100.0
GENERATE_SECS = 2.0
PROMPT_TOKENS = 5
COMPLETION_TOKENS = 64


class _StandInClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


class _StandInCompletion:
    def __init__(self, completion_tokens: int) -> None:
        self.token_ids = list(range(9000, 9000 + completion_tokens))
        self.text = " Paris, and it has been since 987."
        self.finish_reason = "length"


class _StandInOutput:
    def __init__(self, prompt: str, completion_tokens: int) -> None:
        self.prompt = prompt
        self.prompt_token_ids = list(range(100, 100 + PROMPT_TOKENS))
        self.outputs = [_StandInCompletion(completion_tokens)]


class _StandInBackend:
    """`resolve_attention_backend` reads `__name__` off whatever it walks to."""


class _StandInLLM:
    """Enough of `vllm.LLM` for the capture loop, and nothing else.

    The constructor advances the clock by `LOAD_SECS` because that is what a
    51.75 GiB load costs, and a leg that contains it is not a decode
    measurement. `generate` advances it by `GENERATE_SECS`.
    """

    def __init__(self, clock, speculator, backend_name: str, completion_tokens: int, **kwargs):
        self.kwargs = dict(kwargs)
        self._clock = clock
        self._speculator = speculator
        self._completion_tokens = completion_tokens
        clock.now += LOAD_SECS
        backend = type(backend_name, (_StandInBackend,), {})
        runner = type("ModelRunner", (), {"attn_backend": backend})()
        driver = type("Worker", (), {"model_runner": runner})()
        executor = type("Executor", (), {"driver_worker": driver})()
        core = type("InprocClient", (), {})()
        self.llm_engine = type(
            "LLMEngine", (), {"engine_core": core, "model_executor": executor}
        )()
        self.generate_calls = 0

    def generate(self, prompts, sampling):
        self.generate_calls += 1
        self._speculator.propose()
        self._clock.now += GENERATE_SECS
        return [_StandInOutput(prompts[0], self._completion_tokens)]

    def get_metrics(self):
        return []


class CaptureRunTest(unittest.TestCase):
    """`capture.main()` on a stand-in wheel: the arm must COMPLETE and be right.

    The stand-in is legitimate here for the reason `AGENTS.md` §Nothing lands
    dead gives: the entry point is `main()`, the one the lease runs, and what is
    replaced is the 51.75 GiB wheel rather than any part of this tree. What it
    buys is the two guarantees a green leased run would never announce it had
    lost -- the timed span excludes the load, and the tokens counted are the
    ones generated.
    """

    SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        payload = b"a 27B checkpoint stands in for itself"
        (self.root / "model.safetensors").write_bytes(payload)
        (self.root / "draft").mkdir()
        (self.root / "draft" / "draft.safetensors").write_bytes(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        (self.root / "clock.json").write_text(json.dumps(clock_record()), encoding="utf-8")
        self.env = {harness.SSE_PING_ENV: "0"}
        self._real_environ = capture.os.environ
        capture.os.environ = self.env  # type: ignore[assignment]
        self.clock = _StandInClock()
        self._real_time = capture.time
        capture.time = self.clock  # type: ignore[assignment]
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ("vllm", "torch", self.SPECULATOR_MODULE)
        }
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        capture.os.environ = self._real_environ  # type: ignore[assignment]
        capture.time = self._real_time  # type: ignore[assignment]
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self.tmp.cleanup()

    def install_wheel(
        self, *, backend_name: str = "TRITON_ATTN", completion_tokens: int = COMPLETION_TOKENS
    ) -> types.SimpleNamespace:
        """Register a stand-in `vllm`, `torch` and speculator module."""

        _types = types
        torch_mod = _types.ModuleType("torch")
        torch_mod.cuda = _FakeCuda()  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_mod

        spec_mod = _types.ModuleType(self.SPECULATOR_MODULE)

        class DFlashSpeculator(_FakeSpeculator):
            def __init__(self) -> None:
                super().__init__([4242], [[1, 2, 3, 4, 5, 6, 7]])

        spec_mod.DFlashSpeculator = DFlashSpeculator  # type: ignore[attr-defined]
        sys.modules[self.SPECULATOR_MODULE] = spec_mod

        speculator = DFlashSpeculator()
        built: list[_StandInLLM] = []

        def _llm(**kwargs):
            llm = _StandInLLM(
                self.clock, speculator, backend_name, completion_tokens, **kwargs
            )
            built.append(llm)
            return llm

        vllm_mod = _types.ModuleType("vllm")
        vllm_mod.LLM = _llm  # type: ignore[attr-defined]
        vllm_mod.SamplingParams = lambda **kwargs: _types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
        vllm_mod.__version__ = GOOD_VERSION  # type: ignore[attr-defined]
        vllm_mod.__file__ = "/workspace/wheel/vllm/__init__.py"
        sys.modules["vllm"] = vllm_mod
        return _types.SimpleNamespace(
            built=built, speculator=speculator, cuda=torch_mod.cuda
        )

    def argv(self, **overrides: object) -> list[str]:
        argv = [
            item
            for item in precheck_argv(self.root, self.digest, **overrides)
            if item != "--precheck-only"
        ]
        return argv + [
            "--repeat", "2",
            "--max-tokens", str(COMPLETION_TOKENS),
            "--clock-summary", str(self.root / "clock.json"),
            "--output", str(self.root / "vllm-arm.json"),
        ]

    def run_capture(self, **kwargs) -> dict:
        wheel = self.install_wheel(**kwargs)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            rc = capture.main(self.argv())
        self.assertEqual(rc, 0)
        self.wheel = wheel
        return json.loads((self.root / "vllm-arm.json").read_text(encoding="utf-8"))

    def test_the_arm_COMPLETES_and_produces_a_throughput_number(self) -> None:
        """HIGH-A: the read-back probe reached the checker, so nothing refused.

        `attention_backend_reasons` appends "no read-back probe was recorded"
        whenever `probe` is None, so a call site that resolves the probe and
        does not pass it refuses on EVERY run -- after the model has loaded, on
        a lease. No test reached this call, and O23 exists to stop exactly this.
        """

        record = self.run_capture()
        self.assertEqual(record["attention_backend"], "TRITON_ATTN")
        self.assertEqual(record["attention_backend_source"], harness.BACKEND_SOURCE_READ_BACK)
        self.assertIn(record["attention_backend_probe"], harness.BACKEND_PROBES)
        self.assertEqual(
            harness.attention_backend_reasons(
                resolved=record["attention_backend"],
                declared=record["attention_backend_declared"],
                source=record["attention_backend_source"],
                probe=record["attention_backend_probe"],
            ),
            [],
        )
        self.assertGreater(record["metrics"]["output_throughput_tok_s"], 0.0)

    def test_the_TIMED_SPAN_excludes_the_model_load(self) -> None:
        """A leg that contains a 51.75 GiB load is not a decode measurement."""

        record = self.run_capture()
        self.assertEqual(len(record["legs"]), 8)  # 4 prompts x --repeat 2
        for leg in record["legs"]:
            self.assertAlmostEqual(leg["secs"], GENERATE_SECS, places=9)
        # The load happened, and it happened OUTSIDE every leg: the clock moved
        # by one load plus eight generates.
        self.assertAlmostEqual(
            self.clock.now - 1000.0, LOAD_SECS + 8 * GENERATE_SECS, places=9
        )

    def test_the_TOKENS_COUNTED_are_the_ones_GENERATED(self) -> None:
        """The denominator is completion tokens; a prompt in it inflates it."""

        record = self.run_capture()
        for leg in record["legs"]:
            self.assertEqual(leg["completion_tokens"], COMPLETION_TOKENS)
            self.assertEqual(leg["prompt_tokens"], PROMPT_TOKENS)
            self.assertAlmostEqual(
                leg["tok_s"], COMPLETION_TOKENS / GENERATE_SECS, places=9
            )
        self.assertAlmostEqual(
            record["metrics"]["output_throughput_tok_s"],
            COMPLETION_TOKENS / GENERATE_SECS,
            places=9,
        )

    def test_the_emitted_golden_carries_every_contracted_key(self) -> None:
        record = self.run_capture()
        for key in capture.GOLDEN_TOP_LEVEL_KEYS:
            self.assertIn(key, record, f"the capture omits {key}")
        self.assertEqual(len(record["records"]), 4)
        for entry in record["records"]:
            self.assertEqual(entry["num_blocks"], 1)
            self.assertEqual(entry["blocks"][0]["anchor"], 4242)
            self.assertEqual(entry["blocks"][0]["drafts"], [1, 2, 3, 4, 5, 6, 7])
            self.assertEqual(len(entry["output_token_ids"]), COMPLETION_TOKENS)

    def test_a_SUBSTITUTED_denominator_is_refused_AFTER_the_read_back(self) -> None:
        """The refusal is reachable through the arm, not only through the rule."""

        with self.assertRaises(HarnessError) as raised:
            self.run_capture(backend_name="FLASH_ATTN")
        self.assertIn("FLASH_ATTN", str(raised.exception))
        self.assertIn("TRITON_ATTN", str(raised.exception))

    def test_the_recorded_DISCARD_CAUSE_matches_what_this_arm_actually_DOES(self) -> None:
        """The cause ships in every `vllm-arm.json`, and it carried a false clause.

        It said this arm "builds one LLM for every prompt, so only prompt 1's
        run 1" carries the load, the first graph capture and the first KV
        allocation. `capture()` builds the `LLM` BEFORE the prompt loop, so none
        of the three is inside ANY leg here -- prompt 1's run 1 included -- and
        the surviving clause, that run 1 is the repetition the recorder has
        OPEN, is what the discard actually rests on.

        So the clauses are checked against what the run DID. A cause that drifts
        from `capture()` fails here rather than shipping.
        """

        record = self.run_capture()
        legs = record["legs"]
        # ONE `LLM`, for all four of this run's prompts, and built before any
        # leg was timed: every leg is exactly one `generate`, legs[0] --
        # prompt 1's run 1 -- no less than the rest.
        self.assertEqual(len(self.wheel.built), 1)
        self.assertEqual((legs[0]["record"], legs[0]["run"]), (0, 1))
        for leg in legs:
            self.assertAlmostEqual(leg["secs"], GENERATE_SECS, places=9)
        # Run 1 IS the repetition the recorder has OPEN: eight proposes, and
        # only the four run-1 proposes recorded a block.
        self.assertEqual(self.wheel.built[0].generate_calls, 8)
        self.assertEqual(sum(entry["num_blocks"] for entry in record["records"]), 4)
        # The capturing probe is NOT the part run 1 pays extra: two calls per
        # propose on every leg, the warm ones included.
        self.assertEqual(self.wheel.cuda.calls, 2 * 8)

        cause = record["cold_discard_cause"]
        self.assertIn("one LLM for all the prompts", cause)
        self.assertIn("before the prompt loop", cause)
        self.assertIn("OPEN", cause)
        self.assertNotIn("only prompt 1", cause)

    def test_an_arm_that_produced_FEWER_TOKENS_is_refused_by_the_shared_rule(self) -> None:
        with self.assertRaises(HarnessError) as raised:
            self.run_capture(completion_tokens=COMPLETION_TOKENS - 1)
        self.assertIn("did not do the same amount of work", str(raised.exception))


class OurArmRunEntryPointTest(unittest.TestCase):
    """Past the precheck, into the loop -- where the SECOND refusal lives.

    `our_arm.precheck` has its own `require_no_reasons` and the suite covered
    it. The one in `main()` after the legs are parsed did not: deleting it left
    every case green, because nothing had ever driven `main()` far enough to
    parse a leg. It is the rule that stops two arms folding different amounts of
    work into one ratio, so it is the one that makes the number wrong rather
    than absent.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        payload = b"a GGUF target stands in for itself"
        (self.root / "target.gguf").write_bytes(payload)
        (self.root / "draft.gguf").write_bytes(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        (self.root / "clock.json").write_text(json.dumps(clock_record()), encoding="utf-8")
        self.env = {harness.SSE_PING_ENV: "0"}
        self._real_environ = our_arm.os.environ
        our_arm.os.environ = self.env  # type: ignore[assignment]
        self._real_run = our_arm.run_binary
        self.completion = 64
        our_arm.run_binary = self.fake_run  # type: ignore[assignment]
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        our_arm.os.environ = self._real_environ  # type: ignore[assignment]
        our_arm.run_binary = self._real_run  # type: ignore[assignment]
        self.tmp.cleanup()

    def fake_run(self, args, prompt: str) -> str:
        line = (
            "vllm-cli: run={run}/2 finish_reason=length prompt_tokens=5 "
            "completion_tokens={completion} secs=1.250 tok_s={tps}\n"
        )
        return "".join(
            line.format(run=run, completion=self.completion, tps=40.0 + run)
            for run in (1, 2)
        )

    def argv(self) -> list[str]:
        argv = [
            item
            for item in our_arm_argv(self.root, self.digest)
            if item != "--precheck-only"
        ]
        return argv + [
            "--repeat", "2",
            "--max-tokens", "64",
            "--clock-summary", str(self.root / "clock.json"),
            "--output", str(self.root / "our-arm.json"),
        ]

    def run_arm(self) -> dict:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            rc = our_arm.main(self.argv())
        self.assertEqual(rc, 0)
        return json.loads((self.root / "our-arm.json").read_text(encoding="utf-8"))

    def test_a_complete_run_folds_a_warm_median(self) -> None:
        record = self.run_arm()
        self.assertEqual(record["warm_legs"], 4)  # 4 prompts x run 2
        self.assertEqual(record["cold_legs_discarded"], 4)
        self.assertAlmostEqual(record["metrics"]["output_throughput_tok_s"], 42.0, places=9)

    def test_legs_that_did_a_DIFFERENT_amount_of_work_stop_the_run(self) -> None:
        self.completion = 63
        with self.assertRaises(HarnessError) as raised:
            self.run_arm()
        self.assertIn("did not do the same amount of work", str(raised.exception))
        self.assertIn("DFlash2 our-arm capture", str(raised.exception))

if __name__ == "__main__":
    unittest.main()
