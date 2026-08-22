"""Startup-latency (launch -> first ready) contracts for the online gate.

The online serving gate protocol lists ``startup`` among the axes every
interleaved repetition must record (`.agents/specs/cuda-online-serving-gate.md`,
`.agents/specs/qwen35-plain-bf16-direct-load.md:128`), but the harness never
captured it: `scripts/dgx-online-serving.sh` waited for readiness and threw the
duration away, and no manifest carried a startup field.  These cases pin the
measurement contract so a recorded number always means the same thing on both
arms.

The metric is deliberately the SERVER-LIFECYCLE one, not a kernel one: elapsed
monotonic seconds from immediately before the server process is spawned to the
first ``/health`` success, with the identical probe on both engines.  Because
the number is dominated by weight paging, a record is only meaningful when the
page cache was dropped first, so the artifact embeds the leg's cache-drop report
and refuses to exist without it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    HarnessError,
    record_startup,
    summarize_startup,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "dgx-online-serving.sh"


def _cache_drop_report(path: pathlib.Path, *, succeeded: bool = True) -> pathlib.Path:
    path.write_text(
        json.dumps(
            {
                "method": CACHE_DROP_METHOD,
                "succeeded": succeeded,
                "resident_after_bytes": 0,
                "file_count": 12,
                "logical_bytes": 4096,
                "file_inventory_sha256": "a" * 64,
                "roots": ["/snapshot", "/corpus", "/build", "/client"],
            }
        ),
        encoding="utf-8",
    )
    return path


class RecordStartupTest(unittest.TestCase):
    """``record-startup`` is the single writer of a startup artifact."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.cache_drop = _cache_drop_report(self.root / "before.json")
        self.addCleanup(self._tmp.cleanup)

    def _record(self, output: pathlib.Path, **overrides: object) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "engine": "ours",
            "model_key": "35",
            "repetition": 1,
            "elapsed_seconds": 41.5,
            "poll_interval_seconds": 0.2,
            "probe_url": "http://127.0.0.1:8001/health",
            "cache_drop_report": self.cache_drop,
        }
        kwargs.update(overrides)
        return record_startup(output, **kwargs)  # type: ignore[arg-type]

    def test_records_the_measured_elapsed_and_its_provenance(self) -> None:
        output = self.root / "r1.json"
        result = self._record(output)
        self.assertEqual(result["engine"], "ours")
        self.assertEqual(result["model_key"], "35")
        self.assertEqual(result["repetition"], 1)
        self.assertAlmostEqual(float(result["startup_seconds"]), 41.5)
        self.assertAlmostEqual(float(result["poll_interval_seconds"]), 0.2)
        self.assertEqual(result["probe_url"], "http://127.0.0.1:8001/health")
        self.assertTrue(result["cold"])
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

    def test_refuses_to_overwrite_existing_startup_evidence(self) -> None:
        output = self.root / "r1.json"
        self._record(output)
        with self.assertRaises(HarnessError):
            self._record(output)

    def test_rejects_a_non_positive_elapsed(self) -> None:
        with self.assertRaises(HarnessError):
            self._record(self.root / "zero.json", elapsed_seconds=0.0)
        with self.assertRaises(HarnessError):
            self._record(self.root / "negative.json", elapsed_seconds=-1.0)

    def test_rejects_a_poll_interval_too_coarse_to_resolve_the_number(self) -> None:
        """A 5 s probe cadence cannot resolve a ~40 s startup honestly."""
        with self.assertRaises(HarnessError):
            self._record(self.root / "coarse.json", poll_interval_seconds=5.0)

    def test_refuses_a_leg_whose_page_cache_was_not_dropped(self) -> None:
        warm = _cache_drop_report(self.root / "warm.json", succeeded=False)
        with self.assertRaises(HarnessError):
            self._record(self.root / "warm-leg.json", cache_drop_report=warm)

    def test_rejects_an_unknown_engine_or_model(self) -> None:
        with self.assertRaises(HarnessError):
            self._record(self.root / "engine.json", engine="sglang")
        with self.assertRaises(HarnessError):
            self._record(self.root / "model.json", model_key="7")


class SummarizeStartupTest(unittest.TestCase):
    """The aggregate is a paired ours-vs-vLLM ratio over complete legs."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.cache_drop = _cache_drop_report(self.root / "before.json")
        self.addCleanup(self._tmp.cleanup)

    def _leg(self, engine: str, repetition: int, elapsed: float) -> None:
        directory = self.root / "startup" / "35" / engine
        directory.mkdir(parents=True, exist_ok=True)
        record_startup(
            directory / f"r{repetition}.json",
            engine=engine,
            model_key="35",
            repetition=repetition,
            elapsed_seconds=elapsed,
            poll_interval_seconds=0.2,
            probe_url="http://127.0.0.1:8001/health",
            cache_drop_report=self.cache_drop,
        )

    def _all_legs(self) -> None:
        for repetition, elapsed in ((1, 40.0), (2, 42.0), (3, 44.0)):
            self._leg("ours", repetition, elapsed)
        for repetition, elapsed in ((1, 80.0), (2, 84.0), (3, 88.0)):
            self._leg("vllm", repetition, elapsed)

    def test_reports_medians_the_ratio_and_the_observed_spread(self) -> None:
        self._all_legs()
        summary = summarize_startup(self.root, model_key="35", repetitions=(1, 2, 3))
        self.assertAlmostEqual(summary["ours"]["median_seconds"], 42.0)
        self.assertAlmostEqual(summary["vllm"]["median_seconds"], 84.0)
        self.assertAlmostEqual(summary["ours"]["min_seconds"], 40.0)
        self.assertAlmostEqual(summary["ours"]["max_seconds"], 44.0)
        # Ours is FASTER, so the reported speedup is vLLM / ours (> 1 = win).
        self.assertAlmostEqual(summary["startup_speedup_vs_vllm"], 2.0)
        self.assertEqual(summary["model_key"], "35")
        self.assertEqual(summary["repetitions"], [1, 2, 3])

    def test_refuses_an_incomplete_interleaved_series(self) -> None:
        self._all_legs()
        (self.root / "startup" / "35" / "vllm" / "r3.json").unlink()
        with self.assertRaises(HarnessError):
            summarize_startup(self.root, model_key="35", repetitions=(1, 2, 3))

    def test_refuses_legs_recorded_for_a_different_model(self) -> None:
        self._all_legs()
        stray = self.root / "startup" / "35" / "ours" / "r2.json"
        payload = json.loads(stray.read_text(encoding="utf-8"))
        payload["model_key"] = "27"
        stray.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(HarnessError):
            summarize_startup(self.root, model_key="35", repetitions=(1, 2, 3))


class DriverStartupContractTest(unittest.TestCase):
    """`dgx-online-serving.sh` must time readiness and expose a cheap mode."""

    def setUp(self) -> None:
        self.script = SCRIPT.read_text(encoding="utf-8")

    def _function(self, name: str) -> str:
        return self.script.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]

    def test_readiness_polls_finely_enough_to_resolve_startup(self) -> None:
        """The 5 s readiness cadence was the resolution floor; it must go."""
        wait_ready = self._function("wait_ready")
        self.assertNotIn("sleep 5", wait_ready)
        self.assertIn("${ready_poll_interval}", wait_ready)
        self.assertIn("ready_poll_interval=0.2", self.script)

    def test_readiness_keeps_the_previous_thirty_minute_budget(self) -> None:
        """A finer cadence must not shorten the timeout it replaced."""
        self.assertIn("ready_timeout_seconds=1800", self.script)
        wait_ready = self._function("wait_ready")
        self.assertIn("${ready_timeout_seconds}", wait_ready)
        self.assertNotIn("seq 1 360", wait_ready)

    def test_start_server_stamps_the_launch_and_ready_instants(self) -> None:
        start_server = self._function("start_server")
        # The stamp must be taken immediately before the spawn, so nothing
        # between the two reads (memory sampler, logging) is attributed to it.
        launch = start_server.index("startup_launch_epoch=$(")
        spawn = start_server.index('setsid "${server_cmd[@]}"')
        ready = start_server.index("startup_ready_epoch=$(")
        self.assertLess(launch, spawn)
        self.assertLess(spawn, ready)
        self.assertIn("wait_ready", start_server)

    def test_startup_only_is_accepted_by_the_argument_parser(self) -> None:
        """Grepping for the mode name is not proof the flag parses.

        The dispatch branch and the ``case`` arm are separate edits, and a tree
        carrying only the former silently rejects the flag as unknown.
        """
        result = subprocess.run(
            ["bash", str(SCRIPT), "--startup-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotIn("unknown argument", result.stderr)
        self.assertIn("--startup-only", self.script.split("usage() {", 1)[1])

    def test_startup_only_mode_is_dispatchable_and_interleaved(self) -> None:
        self.assertIn("--startup-only", self.script)
        block = self.script.split("if [[ ${mode} == startup-only ]]; then", 1)[1]
        block = block.split("\n  exit 0\nfi", 1)[0]
        self.assertIn("run_startup_leg ours", block)
        self.assertIn("run_startup_leg vllm", block)
        self.assertIn("summarize-startup", block)

    def test_startup_only_drops_caches_and_runs_no_timed_client(self) -> None:
        """Cold on both arms, and no bench client to pay for."""
        leg = self._function("run_startup_leg")
        self.assertIn("drop_caches", leg)
        self.assertIn("gpu_idle", leg)
        self.assertIn("record-startup", leg)
        self.assertIn("cleanup_server", leg)
        self.assertNotIn("online_gate.py bench", leg)
        self.assertNotIn("run_serve_low.py", leg)

    def test_startup_only_reuses_the_grid_server_commands_verbatim(self) -> None:
        """Both arms must launch exactly what the timed grid launches."""
        leg = self._function("run_startup_leg")
        self.assertIn("start_server", leg)
        self.assertNotIn("server_cmd=(", leg)

    def test_startup_only_builds_only_the_server_target(self) -> None:
        """No token is compared, so the model-gate binary is not owed."""
        self.assertIn("build_targets=(server)", self.script)
        guard = self.script.split("build_targets=(server)", 1)[1].split(
            "build_cmd=(", 1
        )[0]
        self.assertIn("${mode} != startup-only", guard)
        self.assertIn('--target "${build_targets[@]}"', self.script)

    def test_script_stays_shellcheck_clean(self) -> None:
        try:
            probe = subprocess.run(
                ["shellcheck", "--version"], capture_output=True, check=False
            )
        except FileNotFoundError:
            # #1661: "unavailable" on a host without the binary is a raised
            # FileNotFoundError, not a nonzero returncode — skip in both arms.
            self.skipTest("shellcheck is unavailable")
        if probe.returncode != 0:
            self.skipTest("shellcheck is unavailable")
        result = subprocess.run(
            ["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_script_parses(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main())
