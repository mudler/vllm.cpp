"""Every-axis aggregation tests ported from vLLM bench-serve metrics.

Sources: ``vllm/benchmarks/serve.py:563-748,1188-1284`` and
``tests/benchmarks/test_serve_cli.py:58-132`` at vLLM e24d1b24.  Project-only
extensions exercise the stricter paired-output, memory-return, and every-axis
acceptance rules from ``.agents/benchmark-protocol.md``.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from tools.bench.online_gate import (
    CACHE_DROP_METHOD,
    FLASHINFER_VERSION,
    INPUT_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_GATE_CONTRACTS,
    MODEL_REVISIONS,
    OUTPUT_LEN,
    PANDAS_VERSION,
    VLLM_ORACLE_VERSION,
    VLLM_GENERATION_WINDOW_CONTRACTS,
    _fingerprint_tree,
)
from tools.bench.gpu_clock_state import MIN_BUSY_SAMPLES, build_clock_record
from tools.bench.online_gate_summary import _report, summarize_evidence
from tools.bench.serve_low_common import HarnessError, VLLM_COMMIT, sha256_file

# One boot id for the whole fixture grid: a clean four-leg capture is same-boot
# by construction, and every clock case below perturbs exactly one field of it.
FIXTURE_BOOT_ID = "f6bbbfc6-0000-4000-8000-000000000000"
OTHER_BOOT_ID = "2fca2b02-0000-4000-8000-000000000000"


def _clock_samples(values, *, utilization=97, throttle="0x0000000000000000"):
    return [
        {
            "clocks_applications_graphics_mhz": 2418,
            "clocks_max_sm_mhz": 3003,
            "driver_version": "580.159.03",
            "gpu_name": "NVIDIA GB10",
            "persistence_mode": "Enabled",
            "sm_clock_mhz": value,
            "throttle_reasons_active": throttle,
            "utilization_gpu_pct": utilization,
        }
        for value in values
    ]


def _clock_window(values):
    """Repeat a clock pattern until it clears the sampler's coverage floor.

    Whole-list repetition leaves min, median, max and `spread_pct` untouched, so
    every clock case below still asserts exactly what it asserted; only the
    sample COUNT changes. The driver samples at 1 Hz across a bench loop of
    minutes, so a three-sample leg was never a leg anyone could have captured.
    """

    values = list(values)
    return values * -(-MIN_BUSY_SAMPLES // len(values))


def _write_clock_leg(root, engine, repetition, values, *, boot_id=FIXTURE_BOOT_ID, **kwargs):
    """Write one leg's clock evidence the way the sampler does."""

    samples = _clock_samples(_clock_window(values), **kwargs)
    base = root / "clocks" / "27" / engine
    base.mkdir(parents=True, exist_ok=True)
    (base / f"r{repetition}.samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    (base / f"r{repetition}.summary.json").write_text(
        json.dumps(build_clock_record(samples, boot_id=boot_id)), encoding="utf-8"
    )


def _record(*, faster: bool, repetition: int) -> dict:
    requests = 2
    duration = 10.0 if faster else 12.0
    latency = 8.0 if faster else 10.0
    record = {
        "completed": requests,
        "duration": duration,
        "errors": [""] * requests,
        "failed": 0,
        "generated_texts": [f"same-{repetition}-0", f"same-{repetition}-1"],
        "input_lens": [INPUT_LEN] * requests,
        "itls": [[0.01] * (OUTPUT_LEN - 1) for _ in range(requests)],
        "max_concurrency": 1,
        "max_concurrent_requests": 1,
        "num_prompts": requests,
        "output_lens": [OUTPUT_LEN] * requests,
        "output_throughput": requests * OUTPUT_LEN / duration,
        "request_throughput": requests / duration,
        "start_times": [0.0, 2.0],
        "total_input_tokens": requests * INPUT_LEN,
        "total_output_tokens": requests * OUTPUT_LEN,
        "total_token_throughput": requests * (INPUT_LEN + OUTPUT_LEN) / duration,
        "ttfts": [0.1, 0.1],
    }
    for metric in ("ttft", "tpot", "itl", "e2el"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = latency
    return record


def _fixture_cache_roots(root: pathlib.Path) -> list[str]:
    return [
        str(root / "fixture-artifacts"),
        str(root / "corpus" / "27"),
        str(root / "fixture-artifacts" / "server"),
        str(root / "fixture-artifacts" / "client"),
    ]


def _write_cache_drop(path: pathlib.Path, roots: list[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "file_count": 1,
        "file_inventory_sha256": "a" * 64,
        "logical_bytes": 4096,
        "method": CACHE_DROP_METHOD,
        "resident_after_bytes": 0,
        "roots": roots,
        "succeeded": True,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "file_count": report["file_count"],
        "file_inventory_sha256": report["file_inventory_sha256"],
        "logical_bytes": report["logical_bytes"],
        "method": CACHE_DROP_METHOD,
        "path": str(path),
        "roots": report["roots"],
        "sha256": sha256_file(path),
    }


def _write_forbidden_trace_artifact(root: pathlib.Path, model: str = "27") -> pathlib.Path:
    """Drop a single H1d profile-control capture into a timed evidence root.

    A pure timed production grid never runs the paired-trace machinery, so any
    profile-control artifact under ``trace/<model>/`` proves an instrumented
    (profile-control-ON) leg contaminated binding timing evidence. The summary
    must fail closed on its mere presence (no mixing)."""
    trace_root = root / "trace" / model
    trace_root.mkdir(parents=True, exist_ok=True)
    artifact = trace_root / "ours_profile_control.json"
    artifact.write_text(json.dumps({"stopped": True}), encoding="utf-8")
    return artifact


def _write_fixture(root: pathlib.Path) -> None:
    sha = "c" * 40
    cutlass = root / "fixture-artifacts" / "cutlass"
    (cutlass / "include/cutlass").mkdir(parents=True)
    (cutlass / "include/cutlass/cutlass.h").write_text(
        "cutlass\n", encoding="utf-8"
    )
    native_target = root / "fixture-artifacts" / "native-must-not-exist.json"
    source_root = root / "fixture-source"
    source_root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "client_contract_source_commit": VLLM_COMMIT,
                "gpu_lock_acquisitions_planned": 2,
                "vllm_oracle_bench_dependencies": {
                    "flashinfer": FLASHINFER_VERSION,
                    "pandas": PANDAS_VERSION,
                },
                "vllm_oracle_version": VLLM_ORACLE_VERSION,
                "vllm_cpp_sha": sha,
            }
        ),
        encoding="utf-8",
    )
    execution_root = root / "execution"
    execution_root.mkdir()
    execution_artifacts = {}
    for name in (
        "build_command",
        "build_log",
        "client",
        "cmake_cache",
        "compile_commands",
        "configure_log",
        "model_config",
        "oracle_manifest",
        "oracle:bench_datasets",
        "oracle:bench_serve",
        "oracle:cli_bench_serve",
        "oracle:client",
        "oracle:distribution_metadata",
        "oracle:distribution_record",
        "oracle:flashinfer_distribution_metadata",
        "oracle:flashinfer_distribution_record",
        "oracle:flashinfer_package_init",
        "oracle:ninja",
        "oracle:package_init",
        "oracle:python",
        "oracle:pandas_distribution_metadata",
        "oracle:pandas_distribution_record",
        "oracle:pandas_package_init",
        "server",
        "tokenizer",
    ):
        path = root / "fixture-artifacts" / name.replace(":", "-")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        execution_artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    weight_name = "model-00001-of-00001.safetensors"
    weight = root / "fixture-artifacts" / weight_name
    weight.write_text("weights\n", encoding="utf-8")
    execution_artifacts[f"weight:{weight_name}"] = {
        "path": str(weight),
        "sha256": sha256_file(weight),
    }
    pathlib.Path(execution_artifacts["cmake_cache"]["path"]).write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root}\n", encoding="utf-8"
    )
    execution_artifacts["cmake_cache"]["sha256"] = sha256_file(
        pathlib.Path(execution_artifacts["cmake_cache"]["path"])
    )
    production_execution = {
        "artifacts": execution_artifacts,
        "bench_dependencies": {
            "flashinfer": FLASHINFER_VERSION,
            "pandas": PANDAS_VERSION,
        },
        "build_contract": {
            "schema_version": 2,
            "build_type": "RelWithDebInfo",
            "compile_command_sha256": "a" * 64,
            "cutlass_source_tree": _fingerprint_tree(cutlass),
            "native_plan_target": str(native_target),
            "native_plan_target_absent": True,
            "profile_control": False,
            "sm_architecture": "121a",
            "triton_aot": True,
            "target_compile_definitions": [
                "VLLM_CPP_FLASH_ATTN",
                "VLLM_CPP_TRITON=1",
                "VLLM_CPP_TRITON_CHUNKO_BF16=1",
                "VT_CUTLASS_NVFP4=1",
            ],
        },
        "cache_drop_roots": _fixture_cache_roots(root),
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS["27"],
        "max_num_seqs": MAX_NUM_SEQS,
        "model_key": "27",
        "num_blocks": 4736,
        "port": 8001,
        "vllm_oracle_version": VLLM_ORACLE_VERSION,
        "vllm_cpp_sha": sha,
        "vllm_source_sha": VLLM_COMMIT,
        "snapshot_files": [],
        "weight_files": [weight_name],
    }
    # A pure timed production grid records ONLY the production manifest
    # (execution/27.json, profile_control False). The diagnostic
    # execution/27-trace.json manifest and the paired trace/ capture belong to
    # --trace-only and must be absent here; their presence is a mixing error.
    (execution_root / "27.json").write_text(
        json.dumps(production_execution),
        encoding="utf-8",
    )
    model_gate = root / "preflight" / "model-gate"
    model_gate.mkdir(parents=True)
    gate_log = model_gate / "27.log"
    # A real ctest -V log: the proof line the gate prints only after it has
    # compared tokens. A checkpoint-absent run exits 0 without it, which is how
    # a skipped precondition used to be recorded as passed.
    gate_log.write_text(
        f"{MODEL_GATE_CONTRACTS['test_qwen27_paged_engine']['proof']}\n"
        "100% tests passed\n",
        encoding="utf-8",
    )
    (model_gate / "27.json").write_text(
        json.dumps(
            {
                "golden_covers_benched_checkpoint": True,
                "golden_revision": MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"][
                    "golden_revision"
                ],
                "log": str(gate_log),
                "log_sha256": sha256_file(gate_log),
                "model_key": "27",
                "model_revision": MODEL_REVISIONS["27"],
                "passed": True,
                "test_name": "test_qwen27_paged_engine",
                "vllm_cpp_sha": sha,
            }
        ),
        encoding="utf-8",
    )
    corpus_root = root / "corpus" / "27"
    corpus_view = corpus_root / "vllm"
    corpus_view.mkdir(parents=True)
    source_manifest = corpus_root / "manifest.json"
    source_manifest.write_text('{"source":"fixture"}\n', encoding="utf-8")
    corpus_files = []
    for repetition in (1, 2, 3):
        filename = f"c1-r{repetition}.jsonl"
        source = corpus_root / filename
        target = corpus_view / filename
        source.write_text(f'{{"source":{repetition}}}\n', encoding="utf-8")
        target.write_text(f'{{"prompt":"fixture-{repetition}"}}\n', encoding="utf-8")
        corpus_files.append(
            {
                "concurrency": 1,
                "file": filename,
                "repetition": repetition,
                "requests": 2,
                "sha256": sha256_file(target),
                "source_sha256": sha256_file(source),
            }
        )
    (corpus_view / "manifest.json").write_text(
        json.dumps(
            {
                "files": corpus_files,
                "model_key": "27",
                "source_manifest_sha256": sha256_file(source_manifest),
                "tokenizer_revision": MODEL_REVISIONS["27"],
                "vllm_commit": VLLM_COMMIT,
            }
        ),
        encoding="utf-8",
    )

    for engine, faster in (("ours", True), ("vllm", False)):
        for repetition in (1, 2, 3):
            raw = root / "raw" / "27" / engine / f"c1-r{repetition}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(
                json.dumps(_record(faster=faster, repetition=repetition)),
                encoding="utf-8",
            )
            client_log = root / "logs" / "27" / engine / f"c1-r{repetition}.log"
            client_log.parent.mkdir(parents=True, exist_ok=True)
            client_log.write_text("timed client passed\n", encoding="utf-8")
            (client_log.parent / f"r{repetition}-server-command.txt").write_text(
                "server --model /fixture --max-num-seqs 32 "
                "--max-num-batched-tokens 2048 --no-enable-prefix-caching "
                "--served-model-name gate\n",
                encoding="utf-8",
            )
            (client_log.parent / f"r{repetition}-server.log").write_text(
                "server ready\n", encoding="utf-8"
            )

            preflight = root / "preflight" / "27" / engine
            preflight.mkdir(parents=True, exist_ok=True)
            (preflight / f"r{repetition}-stream.json").write_text(
                json.dumps(
                    {
                        "emitted_chunks": OUTPUT_LEN,
                        "first_chunk_s": 0.1,
                        "generated_text": "answer",
                        "total_s": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            memory = root / "memory" / "27" / engine
            memory.mkdir(parents=True, exist_ok=True)
            base = 100.0 if faster else 120.0
            (memory / f"r{repetition}.summary.json").write_text(
                json.dumps(
                    {
                        "peak_mem_available_drop_kib": base,
                        "peak_pss_kib": base,
                        "peak_rss_kib": base,
                        "samples": 2,
                    }
                ),
                encoding="utf-8",
            )
            samples = [
                {
                    "alive": True,
                    "gpu_memory_mib": base,
                    "peak_mem_available_drop_kib": base,
                    "pids": [123],
                    "pss_kib": base,
                    "rss_kib": base,
                },
                {
                    "alive": False,
                    "gpu_memory_mib": 0,
                    "peak_mem_available_drop_kib": base,
                    "pids": [],
                    "pss_kib": 0,
                    "rss_kib": 0,
                },
            ]
            (memory / f"r{repetition}.samples.jsonl").write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            # The clock the leg was measured at. Its absence is the #543
            # defect: a number nobody can attribute to a clock state.
            _write_clock_leg(root, engine, repetition, [2470, 2470, 2470])

            thermal = root / "thermal" / "27" / engine
            thermal.mkdir(parents=True, exist_ok=True)
            for suffix in ("before", "after"):
                (thermal / f"r{repetition}-{suffix}.txt").write_text(
                    "Temperature : 50 C\nPower : 100 W\n", encoding="utf-8"
                )

            returned = root / "memory-return" / "27" / engine
            returned.mkdir(parents=True, exist_ok=True)
            cache_root = root / "cache-drop" / "27" / engine
            cache_drops = {
                phase: _write_cache_drop(
                    cache_root / f"r{repetition}-{phase}.json",
                    _fixture_cache_roots(root),
                )
                for phase in ("before", "after")
            }
            (returned / f"r{repetition}.json").write_text(
                json.dumps(
                    {
                        "cache_drops": cache_drops,
                        "drop_caches_succeeded": True,
                        "gpu_idle": True,
                        "mem_available_within_tolerance": True,
                        "returned": True,
                    }
                ),
                encoding="utf-8",
            )


class OnlineGateSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        window_patch = mock.patch.dict(
            VLLM_GENERATION_WINDOW_CONTRACTS,
            {"27": {"all": 1, "clean": 1}},
        )
        window_patch.start()
        self.addCleanup(window_patch.stop)

    def _summarize(self, root: pathlib.Path, *, allow_cross_boot: bool = False):
        patches = (
            mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
            mock.patch("tools.bench.online_gate_summary.POINTS", ((1, 2),)),
            mock.patch(
                "tools.bench.online_gate_summary.MODEL_REVISIONS",
                {"27": MODEL_REVISIONS["27"]},
            ),
        )
        with patches[0], patches[1], patches[2]:
            return summarize_evidence(root, allow_cross_boot=allow_cross_boot)

    def _summarize_model(self, root: pathlib.Path):
        with (
            mock.patch("tools.bench.online_gate.POINTS", ((1, 2),)),
            mock.patch("tools.bench.online_gate_summary.POINTS", ((1, 2),)),
        ):
            return summarize_evidence(root, models=("27",))

    def test_complete_exact_outputs_and_better_axes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(ratios["gate_pass"])
            self.assertTrue(all(item["binding_eligible"] for item in ratios["ratios"]))
            self.assertTrue(all(item["pass"] for item in ratios["ratios"]))

    def test_pre_h1d_production_build_contract_stays_reaggregatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "execution" / "27.json"
            execution = json.loads(path.read_text(encoding="utf-8"))
            build = execution["build_contract"]
            build.pop("schema_version")
            build.pop("build_type")
            build.pop("triton_aot")
            build["target_compile_definitions"] = ["VT_CUTLASS_NVFP4=1"]
            path.write_text(json.dumps(execution), encoding="utf-8")
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(ratios["gate_pass"])

    def test_model_summary_does_not_require_the_other_gate_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            runs, ratios = self._summarize_model(root)
            self.assertEqual(runs["models"], ["27"])
            self.assertEqual(ratios["models"], ["27"])
            self.assertTrue(runs["gate_pass"])
            self.assertFalse(
                any("35" in reason for reason in runs["campaign_reasons"])
            )

    def test_profile_control_artifacts_in_a_timed_root_are_rejected(self) -> None:
        """The timed grid is a pure production build; no H1d trace mixing.

        A production (profile-control-OFF) evidence root that also carries the
        paired-trace machinery's output — a profile-control capture under
        ``trace/<model>/``, the diagnostic profile-control-ON execution
        manifest ``execution/<model>-trace.json``, or a ``build_contract`` whose
        ``profile_control`` is not ``False`` — proves an instrumented leg
        contaminated binding timing evidence. The summary fails closed on each.
        """
        # Baseline: the production fixture carries no trace artifacts and binds.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            self.assertFalse((root / "trace").exists())
            self.assertFalse((root / "execution" / "27-trace.json").exists())
            runs, _ = self._summarize_model(root)
            self.assertTrue(runs["gate_pass"])

        # A stray profile-control capture under trace/<model>/ is a mixing error.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _write_forbidden_trace_artifact(root)
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])

        # The diagnostic profile-control-ON execution manifest must not coexist.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "execution" / "27-trace.json").write_text(
                json.dumps({"build_contract": {"profile_control": True}}),
                encoding="utf-8",
            )
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])

        # profile_control=true in the production manifest is refused directly.
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "execution" / "27.json"
            execution = json.loads(path.read_text(encoding="utf-8"))
            execution["build_contract"]["profile_control"] = True
            path.write_text(json.dumps(execution), encoding="utf-8")
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])

    def test_model_gate_scope_must_be_recorded_and_consistent(self) -> None:
        """A gate that skipped, or whose scope is silent, cannot pass.

        `test_qwen27_paged_engine` returns 0 when its checkpoint is absent, so
        without the proof line the summary would accept zero tokens compared as
        a correctness precondition -- the expected shape for key "27n", whose
        goldens are a DIFFERENT checkpoint's on purpose.
        """
        gate_json = pathlib.Path("preflight") / "model-gate" / "27.json"
        gate_log = pathlib.Path("preflight") / "model-gate" / "27.log"
        mutations = {
            "golden revision": lambda s: s.__setitem__("golden_revision", "0" * 40),
            "benched revision": lambda s: s.__setitem__("model_revision", "0" * 40),
            "scope flag": lambda s: s.__setitem__(
                "golden_covers_benched_checkpoint", False
            ),
            "scope field absent": lambda s: s.pop("golden_covers_benched_checkpoint"),
            "revision field absent": lambda s: s.pop("golden_revision"),
            "unknown gate": lambda s: s.__setitem__("test_name", "invented_gate"),
        }
        for label, mutate in mutations.items():
            with self.subTest(mutation=label), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                _write_fixture(root)
                path = root / gate_json
                status = json.loads(path.read_text(encoding="utf-8"))
                mutate(status)
                path.write_text(json.dumps(status), encoding="utf-8")
                runs, _ = self._summarize_model(root)
                self.assertFalse(runs["gate_pass"])
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            log = root / gate_log
            log.write_text("100% tests passed\n", encoding="utf-8")
            status = json.loads((root / gate_json).read_text(encoding="utf-8"))
            status["log_sha256"] = sha256_file(log)
            (root / gate_json).write_text(json.dumps(status), encoding="utf-8")
            runs, _ = self._summarize_model(root)
            self.assertFalse(runs["gate_pass"])
            self.assertIn(
                "model-gate log carries no proof that a token was compared",
                json.dumps(runs),
            )

    def test_model_selection_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(HarnessError, "empty"):
                summarize_evidence(root, models=())
            with self.assertRaisesRegex(HarnessError, "duplicates"):
                summarize_evidence(root, models=("27", "27"))
            with self.assertRaisesRegex(HarnessError, "unknown"):
                summarize_evidence(root, models=("not-a-model",))

    def test_text_difference_is_diagnostic_after_model_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "raw" / "27" / "vllm" / "c1-r2.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["generated_texts"][0] = "different"
            path.write_text(json.dumps(record), encoding="utf-8")
            runs, ratios = self._summarize(root)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(all(item["binding_eligible"] for item in ratios["ratios"]))
            diagnostic = next(
                item
                for item in runs["output_text_diagnostics"]
                if item["repetition"] == 2
            )
            self.assertFalse(diagnostic["all_equal"])
            self.assertEqual(diagnostic["exact_matches"], 1)

    def test_missing_memory_or_partial_request_set_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "memory" / "27" / "ours" / "r1.samples.jsonl").unlink()
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            returned_path = root / "memory-return" / "27" / "ours" / "r1.json"
            returned = json.loads(returned_path.read_text(encoding="utf-8"))
            returned["cache_drops"]["before"]["roots"][0] = "/wrong-snapshot"
            returned_path.write_text(json.dumps(returned), encoding="utf-8")
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            command = root / "logs" / "27" / "ours" / "r1-server-command.txt"
            command.write_text(
                "server --max-num-seqs 8 --max-num-batched-tokens 2048 "
                "--served-model-name gate\n",
                encoding="utf-8",
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "raw" / "27" / "ours" / "c1-r1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["completed"] = 1
            record["failed"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_hash_drifted_execution_artifact_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "fixture-artifacts" / "oracle-bench_serve").write_text(
                "tampered\n", encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_every_ratio_carries_the_clock_it_was_measured_at(self) -> None:
        """A ratio without its clock is not quotable (#543)."""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _, ratios = self._summarize(root)
            self.assertTrue(ratios["ratios"])
            for ratio in ratios["ratios"]:
                clock = ratio["clock"]
                self.assertTrue(clock["same_boot"])
                self.assertEqual(clock["reasons"], [])
                self.assertEqual(clock["ours_median_sm_mhz"], 2470.0)
                self.assertEqual(clock["vllm_median_sm_mhz"], 2470.0)
                self.assertAlmostEqual(clock["median_offset_pct"], 0.0)
                self.assertAlmostEqual(clock["estimated_effect_pct"], 0.0)

    def test_a_missing_clock_record_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "clocks" / "27" / "ours" / "r2.summary.json").unlink()
            runs, ratios = self._summarize(root)
            self.assertFalse(runs["gate_pass"])
            self.assertFalse(ratios["gate_pass"])
            self.assertTrue(
                any(
                    "clock" in reason
                    for aggregate in runs["aggregates"]
                    for reason in aggregate["reasons"]
                )
            )

    def test_a_missing_clock_record_names_the_offending_arm_in_every_ratio(self) -> None:
        """The leg reason already voids the gate; the READER still needs to know.

        Nothing else in the summary says WHICH arm lost its clock, so this is
        the only site that can be gutted without any gate going green-to-red.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "clocks" / "27" / "vllm" / "r1.summary.json").unlink()
            _, ratios = self._summarize(root)
            self.assertTrue(ratios["ratios"])
            for ratio in ratios["ratios"]:
                named = [
                    reason
                    for reason in ratio["clock"]["reasons"]
                    if "no usable SM-clock record" in reason
                ]
                self.assertTrue(named, ratio["clock"]["reasons"])
                self.assertIn("vllm", named[0])
                self.assertNotIn("ours arm", named[0])

    def test_a_cross_boot_pair_cannot_pass(self) -> None:
        """The exact shape that produced the retracted #543 findings."""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for repetition in (1, 2, 3):
                _write_clock_leg(
                    root,
                    "vllm",
                    repetition,
                    [2470, 2470, 2470],
                    boot_id=OTHER_BOOT_ID,
                )
            runs, ratios = self._summarize(root)
            self.assertFalse(ratios["gate_pass"])
            self.assertFalse(runs["gate_pass"])
            offenders = [
                reason
                for ratio in ratios["ratios"]
                for reason in ratio["clock"]["reasons"]
                if "DIFFERENT boots" in reason
            ]
            self.assertTrue(offenders)
            self.assertIn(OTHER_BOOT_ID, offenders[0])
            # Each arm is internally clean here, so NOTHING at leg or arm level
            # voids these ratios -- only the ratio's own clock term can. The two
            # ratio families are asserted SEPARATELY because they carry that
            # term at two different call sites, and a single gate_pass assertion
            # lets either site mask the other's removal (the #520 lesson).
            throughput = [
                ratio for ratio in ratios["ratios"] if ratio["concurrency"] is not None
            ]
            memory = [
                ratio for ratio in ratios["ratios"] if ratio["concurrency"] is None
            ]
            self.assertTrue(throughput)
            self.assertTrue(memory)
            self.assertFalse(any(ratio["binding_eligible"] for ratio in throughput))
            self.assertFalse(any(ratio["binding_eligible"] for ratio in memory))

    def test_the_cross_boot_override_records_a_caveat_rather_than_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for repetition in (1, 2, 3):
                _write_clock_leg(
                    root,
                    "vllm",
                    repetition,
                    [2470, 2470, 2470],
                    boot_id=OTHER_BOOT_ID,
                )
            runs, ratios = self._summarize(root, allow_cross_boot=True)
            self.assertTrue(runs["gate_pass"])
            self.assertTrue(ratios["gate_pass"])
            for ratio in ratios["ratios"]:
                self.assertTrue(ratio["clock"]["cross_boot_override"])
                self.assertFalse(ratio["clock"]["same_boot"])
                self.assertTrue(ratio["clock"]["caveats"])

    def test_the_override_does_not_waive_a_staircase_arm(self) -> None:
        """Three individually-flat legs at three DIFFERENT clocks, one boot.

        This is the only check that lives ONLY at the compare site.
        `_clock_for_leg` sees three steady legs and says nothing;
        `merge_clock_records` folds them without complaint because they share a
        boot and every static field; `_clock_for_arm` raises only on those two.
        The merged spread -- (2470 - 2190) / 2300 = 12.17% -- is first evaluated
        inside `compare_clock_records`, so rewriting that call's reason list to
        `[] if allow_cross_boot else [...]` is invisible to every other case.

        The vLLM arm is pinned flat at the merged median so the CROSS-ARM OFFSET
        stays 0.00%: the offset reason is appended outside that expression and
        would otherwise mask the removal, which is how the two tests that name
        this guarantee came to be dominated.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for repetition, clock in zip((1, 2, 3), (2470, 2300, 2190)):
                _write_clock_leg(root, "ours", repetition, [clock])
            for repetition in (1, 2, 3):
                _write_clock_leg(root, "vllm", repetition, [2300], boot_id=OTHER_BOOT_ID)
            runs, ratios = self._summarize(root, allow_cross_boot=True)

            # Nothing below the compare site objects: every leg is steady, the
            # fold succeeds, and no aggregate carries a clock reason. Asserted so
            # a future edit that moves the check earlier makes this case say so
            # rather than passing for a new reason.
            self.assertEqual(runs["campaign_reasons"], [])
            self.assertEqual(
                [
                    reason
                    for aggregate in runs["aggregates"]
                    for reason in aggregate["reasons"]
                    if "clock" in reason
                ],
                [],
            )
            clock = ratios["clocks"]["27"]
            self.assertTrue(clock["cross_boot_override"])
            self.assertAlmostEqual(clock["median_offset_pct"], 0.0)
            self.assertAlmostEqual(clock["ours_spread_pct"], 280.0 / 2300.0 * 100.0)
            self.assertFalse(
                any("offset" in reason for reason in clock["reasons"]), clock["reasons"]
            )

            self.assertFalse(ratios["gate_pass"])
            spread = [reason for reason in clock["reasons"] if "spread" in reason]
            self.assertTrue(spread, clock["reasons"])
            self.assertIn("ours", spread[0])
            # The two ratio families carry the clock term at two different call
            # sites, so they are asserted SEPARATELY (the #520 lesson).
            throughput = [
                ratio for ratio in ratios["ratios"] if ratio["concurrency"] is not None
            ]
            memory = [ratio for ratio in ratios["ratios"] if ratio["concurrency"] is None]
            self.assertTrue(throughput)
            self.assertTrue(memory)
            self.assertFalse(any(ratio["binding_eligible"] for ratio in throughput))
            self.assertFalse(any(ratio["binding_eligible"] for ratio in memory))

    def test_a_diluted_window_cannot_pass_and_says_so_in_the_report(self) -> None:
        """One busy sample among three hundred scored a perfect +0.00%.

        The record every leg wrote already counted the exclusions; nothing
        asserted them, folded them into the ratio's clock block, or printed
        them, so the window the sampler barely observed outscored the one it
        watched.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for engine in ("ours", "vllm"):
                for repetition in (1, 2, 3):
                    base = root / "clocks" / "27" / engine
                    samples = _clock_samples([2470]) + _clock_samples(
                        [300] * 300, utilization=0
                    )
                    (base / f"r{repetition}.samples.jsonl").write_text(
                        "".join(json.dumps(sample) + "\n" for sample in samples),
                        encoding="utf-8",
                    )
                    (base / f"r{repetition}.summary.json").write_text(
                        json.dumps(build_clock_record(samples, boot_id=FIXTURE_BOOT_ID)),
                        encoding="utf-8",
                    )
            runs, ratios = self._summarize(root)
            clock = ratios["clocks"]["27"]
            self.assertEqual(clock["ours_busy_samples"], 3)
            self.assertEqual(clock["ours_idle_samples_excluded"], 900)
            self.assertAlmostEqual(clock["ours_spread_pct"], 0.0)
            self.assertFalse(runs["gate_pass"])
            self.assertFalse(ratios["gate_pass"])
            report = _report(runs, ratios)
            self.assertIn("3 busy / 900 idle", report)

    def test_the_override_does_not_waive_an_over_spread_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _write_clock_leg(
                root, "ours", 1, [1781, 2100, 2398], boot_id=OTHER_BOOT_ID
            )
            runs, _ = self._summarize(root, allow_cross_boot=True)
            self.assertFalse(runs["gate_pass"])

    def test_an_over_spread_window_cannot_pass(self) -> None:
        """Two probes eight minutes apart inside ONE boot: 2398 against 1781."""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _write_clock_leg(root, "ours", 3, [1781, 2100, 2398])
            runs, ratios = self._summarize(root)
            self.assertFalse(runs["gate_pass"])
            self.assertFalse(ratios["gate_pass"])
            self.assertTrue(
                any(
                    "spread" in reason
                    for aggregate in runs["aggregates"]
                    for reason in aggregate["reasons"]
                )
            )

    def test_a_cross_arm_clock_offset_cannot_pass_even_same_boot(self) -> None:
        """The measured pair: 2470 against 2190, one boot apart in reality."""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for repetition in (1, 2, 3):
                _write_clock_leg(root, "vllm", repetition, [2190, 2190, 2190])
            runs, ratios = self._summarize(root)
            self.assertFalse(ratios["gate_pass"])
            self.assertFalse(runs["gate_pass"])
            clock = ratios["ratios"][0]["clock"]
            self.assertTrue(clock["same_boot"])
            self.assertAlmostEqual(clock["median_offset_pct"], 12.7854, places=3)
            self.assertAlmostEqual(clock["estimated_effect_pct"], 9.65, places=1)
            self.assertTrue(any("offset" in reason for reason in clock["reasons"]))

    def test_a_throttled_window_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _write_clock_leg(
                root, "ours", 2, [2470, 2470], throttle="0x0000000000000040"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])
            self.assertTrue(
                any(
                    "throttl" in reason
                    for aggregate in runs["aggregates"]
                    for reason in aggregate["reasons"]
                )
            )

    def test_a_clock_summary_that_disagrees_with_its_stream_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            path = root / "clocks" / "27" / "ours" / "r1.summary.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["sm_clock_mhz"]["n"] = 99
            path.write_text(json.dumps(record), encoding="utf-8")
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])

    def test_one_arm_may_not_straddle_two_boots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            _write_clock_leg(
                root, "ours", 2, [2470, 2470, 2470], boot_id=OTHER_BOOT_ID
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])
            self.assertTrue(
                any(
                    "boot" in reason
                    for aggregate in runs["aggregates"]
                    for reason in aggregate["reasons"]
                )
            )

    def test_the_report_prints_the_clock_next_to_the_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            for repetition in (1, 2, 3):
                _write_clock_leg(root, "vllm", repetition, [2190, 2190, 2190])
            runs, ratios = self._summarize(root)
            report = _report(runs, ratios)
            self.assertIn("SM clock", report)
            self.assertIn("2470", report)
            self.assertIn("2190", report)

    def test_hash_drifted_corpus_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _write_fixture(root)
            (root / "corpus" / "27" / "vllm" / "c1-r2.jsonl").write_text(
                '{"prompt":"tampered"}\n', encoding="utf-8"
            )
            runs, _ = self._summarize(root)
            self.assertFalse(runs["gate_pass"])


if __name__ == "__main__":
    unittest.main()
