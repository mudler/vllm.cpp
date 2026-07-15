"""Fail-closed tests for the packed-vs-rollback serving component.

The metric schema comes from ``vllm/benchmarks/serve.py:563-748,1188-1284``
at vLLM ``702f481``.  The AB/BA/AB and every-axis acceptance contract is the
local G3 gate in ``.agents/specs/gdn-packed-decode.md``.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import shlex
import statistics
import subprocess
import tempfile
import unittest
from unittest import mock

import tools.bench.gdn_packed_component as gdn_component
from tools.bench.gdn_packed_component import (
    ARMS,
    CONCURRENCIES,
    LEG_ORDER,
    build_component_plan,
    finalize_evidence,
    summarize_component_records,
    summarize_evidence,
    verify_finalized_evidence,
)
from tools.bench.online_gate import (
    FLASHINFER_VERSION,
    MODEL_REVISIONS,
    OnlineRun,
    PANDAS_VERSION,
    POINTS,
    TRACE_CLEAN_FIXED_ENV,
    TRACE_REQUIRED_ENV,
    TRACE_SYSTEM_PATH,
    VLLM_ORACLE_VERSION,
    build_client_command,
)
from tools.bench.serve_low_common import (
    HarnessError,
    SGLANG_COMMIT,
    VLLM_COMMIT,
    sha256_file,
)
from tests.tools.test_online_gate_client import (
    write_cache_drop_report,
    write_profile_log,
)


SOURCE_SHA = "c" * 40
SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Text anchors bounding the diagnostic-c16 flow inside the driver script.
DIAG_BEGIN = "# >>> diagnostic-c16 flow"
DIAG_END = "# <<< diagnostic-c16 flow"


def _isolated_environment(
    *, arm: str, fixture: pathlib.Path, native: pathlib.Path, home: pathlib.Path
) -> list[str]:
    values = {
        "HOME": str(home),
        "PYTHONPATH": str(SOURCE_ROOT),
        "PATH": TRACE_SYSTEM_PATH,
        **TRACE_CLEAN_FIXED_ENV,
        **TRACE_REQUIRED_ENV,
        "VT_FP4_AUTOTUNE_CACHE_PATH": str(native),
        "VT_FP4_AUTOTUNE_VERBOSE": "1",
        "VT_FP4_FLASHINFER_CACHE_PATH": str(fixture),
        "VT_GDN_PACKED_DECODE": "1" if arm == "packed" else "0",
    }
    return [f"{name}={value}" for name, value in values.items()]


def _clean_environment(*, home: pathlib.Path) -> list[str]:
    values = {
        "HOME": str(home),
        "PYTHONPATH": str(SOURCE_ROOT),
        "PATH": TRACE_SYSTEM_PATH,
        **TRACE_CLEAN_FIXED_ENV,
    }
    return [f"{name}={value}" for name, value in values.items()]


def _write_model_gate_log(
    path: pathlib.Path,
    *,
    fixture: pathlib.Path,
    native: pathlib.Path,
    snapshot: pathlib.Path,
) -> None:
    write_profile_log(
        path,
        fixture=fixture,
        native_target=native,
        server_pid=99,
    )
    with path.open("a", encoding="utf-8") as output:
        output.write(
            "/source/tests/parity/test_qwen27_paged_engine.cpp:150: MESSAGE: "
            "qwen27_paged_engine: loading full 27B via FromModelDir("
            f"{snapshot}) — dense W4A4 fp4-resident loader + engine stack...\n"
            "qwen27_paged_engine M0-EXIT: produced 16/16 tokens; "
            'continuation="fixture"\n'
            "qwen27_paged_engine: full production stream 16/16 token-exact "
            "vs vLLM\n"
            "[doctest] test cases:   1 |   1 passed | 0 failed | 0 skipped\n"
            "[doctest] assertions: 235 | 235 passed | 0 failed |\n"
            "[doctest] Status: SUCCESS!\n"
        )


def _thermal_snapshot(
    *,
    sw_thermal_us: int = 0,
    hw_thermal_us: int = 0,
    hw_power_braking_us: int = 0,
    hw_thermal_active: bool = False,
) -> str:
    return (
        "==============NVSMI LOG==============\n"
        "GPU 00000000:01:00.0\n"
        "    Clocks Event Reasons\n"
        "        SW Power Cap                      : Not Active\n"
        "        HW Slowdown                       : Not Active\n"
        "            HW Thermal Slowdown           : "
        + ("Active" if hw_thermal_active else "Not Active")
        + "\n"
        "            HW Power Brake Slowdown       : Not Active\n"
        "        SW Thermal Slowdown                : Not Active\n"
        "    Temperature\n"
        "        GPU Current Temp                  : 45 C\n"
        "    GPU Power Readings\n"
        "        Average Power Draw                : 100.00 W\n"
        "        Instantaneous Power Draw          : 105.00 W\n"
        "    Clocks Event Reasons Counters\n"
        "        SW Power Capping                   : 0 us\n"
        "        Sync Boost                         : 0 us\n"
        f"        SW Thermal Slowdown               : {sw_thermal_us} us\n"
        f"        HW Thermal Slowdown               : {hw_thermal_us} us\n"
        f"        HW Power Braking                  : {hw_power_braking_us} us\n"
    )


def _record(*, concurrency: int, packed_better: bool, repetition: int) -> dict:
    requests = {2: 6, 16: 96}[concurrency]
    duration = 10.0 if packed_better else 12.0
    latency = 8.0 if packed_better else 10.0
    record = {
        "completed": requests,
        "duration": duration,
        "errors": [""] * requests,
        "failed": 0,
        "generated_texts": [f"text-{repetition}-{index}" for index in range(requests)],
        "input_lens": [1024] * requests,
        "itls": [[latency / 1000.0] * 127 for _ in range(requests)],
        "max_concurrency": concurrency,
        "max_concurrent_requests": concurrency,
        "num_prompts": requests,
        "output_lens": [128] * requests,
        "output_throughput": requests * 128 / duration,
        "request_throughput": requests / duration,
        "start_times": [float(index // concurrency) * 1.5 for index in range(requests)],
        "total_input_tokens": requests * 1024,
        "total_output_tokens": requests * 128,
        "total_token_throughput": requests * (1024 + 128) / duration,
        "ttfts": [latency / 1000.0] * requests,
    }
    for metric in ("ttft", "tpot", "itl"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = latency
    for stat in ("mean", "median", "p90", "p99"):
        record[f"{stat}_e2el_ms"] = latency * 128
    return record


def _set_latency(record: dict, latency_ms: float) -> None:
    requests = record["completed"]
    record["ttfts"] = [latency_ms / 1000.0] * requests
    record["itls"] = [[latency_ms / 1000.0] * 127 for _ in range(requests)]
    for metric in ("ttft", "tpot", "itl"):
        for stat in ("mean", "median", "p90", "p99"):
            record[f"{stat}_{metric}_ms"] = latency_ms
    for stat in ("mean", "median", "p90", "p99"):
        record[f"{stat}_e2el_ms"] = latency_ms * 128


def _apply_ttft_ms(record: dict, ttfts_ms: list[float]) -> None:
    """Replace the TTFT samples and re-derive every reported timing field.

    Only the raw ``ttfts`` array changes; the ITL/output arrays and duration
    stay fixed, so throughput, TPOT and ITL are untouched and — because decode
    dominates E2EL — essentially only the TTFT distribution (including its
    p90/p99 tail) moves.  Reassigning every ``LOWER_AXES`` field from the
    detailed recomputation keeps the record self-consistent with the harness'
    exact ``_run_metrics`` recomputation check.
    """

    if len(ttfts_ms) != record["completed"]:
        raise AssertionError("ttft sample count must equal the request count")
    record["ttfts"] = [value / 1000.0 for value in ttfts_ms]
    metrics = gdn_component._recompute_timing_metrics(record)
    for axis in gdn_component.LOWER_AXES:
        record[axis] = metrics[axis]


# --- c2 TTFT phase-lottery fixtures (ms) -------------------------------------
# Bimodal per-request TTFTs from the upstream-mirrored prefill co-schedule
# arrival lottery: a 1024-token prefill runs alone (~fast) or co-schedules
# (~slow, ~2x).  Magnitudes are small vs the ~1 s fixture decode so E2EL stays
# stable (<4%) while the c2 TTFT-family per-rep aggregates swing far past 4%.
# Reps flip between a 3/3 mix and an all-slow 6/0 rep.  Packed sits uniformly
# below rollback per request so the arm comparison and per-rep pairing stay clean.
_C2_TTFT_BIMODAL_PACKED = {
    1: [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],     # 3/3 mix,   mean 7.5
    2: [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],     # 3/3 mix,   mean 7.5
    3: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # 6/0 slow,  mean 10.0
}
_C2_TTFT_BIMODAL_ROLLBACK = {
    1: [11.0, 11.0, 11.0, 22.0, 22.0, 22.0],  # 3/3 mix,   mean 16.5
    2: [22.0, 22.0, 22.0, 22.0, 22.0, 22.0],  # 6/0 slow,  mean 22.0
    3: [11.0, 11.0, 11.0, 22.0, 22.0, 22.0],  # 3/3 mix,   mean 16.5
}
# A per-rep flip: rollback r1 dips below packed r1 (arrival phasing).  Both arms
# pool to the SAME 18-sample TTFT distribution (9x10 + 9x20 ms — the honest
# at-parity expectation, since packed decode does not move TTFT), so the pooled
# arm comparison ties (packed <= rollback holds) while the per-rep mixture flips.
# All rep aggregates stay within 50% of the shared pool (no all-fast reps, so no
# axis rides the 50% boundary).
_C2_TTFT_FLIP_PACKED = {
    1: [10.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # 1 fast/5 slow, mean 18.33
    2: [10.0, 10.0, 10.0, 10.0, 20.0, 20.0],  # 4 fast/2 slow, mean 13.33
    3: [10.0, 10.0, 10.0, 10.0, 20.0, 20.0],  # 4 fast/2 slow, mean 13.33
}
_C2_TTFT_FLIP_ROLLBACK = {
    1: [10.0, 10.0, 10.0, 10.0, 10.0, 20.0],  # 5 fast/1 slow, mean 11.67 (< packed r1)
    2: [10.0, 10.0, 20.0, 20.0, 20.0, 20.0],  # 2 fast/4 slow, mean 16.67
    3: [10.0, 10.0, 20.0, 20.0, 20.0, 20.0],  # 2 fast/4 slow, mean 16.67
}
# c16 TTFT tail fixtures (96 requests): 94 at 20 ms fix mean/median/p90; the two
# tail requests set p99 = ~0.95*low + 0.05*high.  These prove the 15% tail rule
# still governs c16 TTFT tails (c16 is NEVER pooled).
_C16_TTFT_TAIL_WITHIN_15PCT = {1: (58.0, 88.0), 2: (52.0, 82.0), 3: (50.0, 80.0)}
_C16_TTFT_TAIL_BEYOND_15PCT = {1: (62.0, 92.0), 2: (52.0, 82.0), 3: (50.0, 80.0)}


def _c16_tail_ttft(low: float, high: float) -> list[float]:
    return [20.0] * 94 + [low, high]


def _records(*, packed_better: bool) -> dict[tuple[int, str, int], dict]:
    result = {}
    for concurrency in CONCURRENCIES:
        for arm in ARMS:
            for repetition in (1, 2, 3):
                result[(concurrency, arm, repetition)] = _record(
                    concurrency=concurrency,
                    packed_better=(arm == "packed") == packed_better,
                    repetition=repetition,
                )
    return result


def _memory(*, packed_better: bool) -> dict[tuple[int, str, int], dict]:
    result = {}
    for concurrency in CONCURRENCIES:
        for arm in ARMS:
            for repetition in (1, 2, 3):
                preferred = (arm == "packed") == packed_better
                base = 100.0 if preferred else 120.0
                result[(concurrency, arm, repetition)] = {
                    "peak_gpu_memory_mib": base,
                    "peak_pss_kib": base * 1024,
                    "peak_rss_kib": base * 2048,
                    "peak_mem_available_drop_kib": base * 4096,
                    "returned": True,
                }
    return result


def _write_fixture_corpus(root: pathlib.Path, *, tokenizer: pathlib.Path) -> None:
    corpus_root = root / "corpus" / "27"
    corpus_root.mkdir(parents=True)
    source_files = []
    source_partitions = [("warmup", 1)] + [
        (f"c{concurrency}-r{repetition}", 192)
        for repetition in (1, 2, 3)
        for concurrency, _ in POINTS
    ]
    for partition, requests in source_partitions:
        filename = "warmup.jsonl" if partition == "warmup" else f"{partition}.jsonl"
        path = corpus_root / filename
        path.write_text(
            "".join(
                json.dumps({"index": index, "partition": partition}) + "\n"
                for index in range(requests)
            ),
            encoding="utf-8",
        )
        source_files.append(
            {
                "file": filename,
                "partition": partition,
                "requests": requests,
                "sha256": sha256_file(path),
            }
        )
    source_manifest = corpus_root / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "common_prefix_limit": 32,
                "files": source_files,
                "format": "sglang-custom-conversations-jsonl-v1",
                "model_key": "27",
                "output_len": 128,
                "requests_per_partition": 192,
                "seed": 0,
                "sglang_commit": SGLANG_COMMIT,
                "target_input_len": 1024,
                "tokenizer_revision": MODEL_REVISIONS["27"],
                "tokenizer_sha256": sha256_file(tokenizer),
                "total_prompts": 3457,
                "warmup_requests": 1,
            }
        ),
        encoding="utf-8",
    )

    vllm_root = corpus_root / "vllm"
    vllm_root.mkdir()
    vllm_files = []
    source_by_name = {item["file"]: item for item in source_files}
    for repetition in (1, 2, 3):
        for concurrency, requests in POINTS:
            filename = f"c{concurrency}-r{repetition}.jsonl"
            path = vllm_root / filename
            path.write_text(
                "".join(
                    json.dumps({"prompt": f"{filename}-{index}"}) + "\n"
                    for index in range(requests)
                ),
                encoding="utf-8",
            )
            vllm_files.append(
                {
                    "concurrency": concurrency,
                    "file": filename,
                    "repetition": repetition,
                    "requests": requests,
                    "sha256": sha256_file(path),
                    "source_sha256": source_by_name[filename]["sha256"],
                }
            )
    vllm_manifest = vllm_root / "manifest.json"
    vllm_manifest.write_text(
        json.dumps(
            {
                "files": vllm_files,
                "format": "vllm-custom-jsonl-v1",
                "input_len": 1024,
                "model_key": "27",
                "output_len": 128,
                "source_manifest_sha256": sha256_file(source_manifest),
                "tokenizer_revision": MODEL_REVISIONS["27"],
                "total_prompts": sum(requests for _, requests in POINTS) * 3,
                "vllm_commit": VLLM_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    gdn_component.COMPONENT_SOURCE_CORPUS_MANIFEST_SHA256 = sha256_file(
        source_manifest
    )
    gdn_component.COMPONENT_VLLM_CORPUS_MANIFEST_SHA256 = sha256_file(
        vllm_manifest
    )


def _write_complete_evidence(root: pathlib.Path, *, packed_better: bool = True) -> None:
    (root / "component-plan.json").write_text(
        json.dumps(build_component_plan(SOURCE_SHA)), encoding="utf-8"
    )
    build = root / "build"
    artifact = build / "examples" / "server"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "immutable MatmulNvfp4Cutlass [VT_FP4_CACHE] prepared\n",
        encoding="utf-8",
    )
    model_gate_binary = build / "tests" / "test_qwen27_paged_engine"
    model_gate_binary.parent.mkdir(parents=True)
    model_gate_binary.write_text("immutable model gate\n", encoding="utf-8")
    model_gate_binary.chmod(0o755)
    snapshot = root / MODEL_REVISIONS["27"]
    snapshot.mkdir()
    model_config = snapshot / "config.json"
    model_config.write_text("{}\n", encoding="utf-8")
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_text("{}\n", encoding="utf-8")
    _write_fixture_corpus(root, tokenizer=tokenizer)
    weight = snapshot / "model-00001-of-00001.safetensors"
    weight.write_text("weight\n", encoding="utf-8")
    generation_config = snapshot / "generation_config.json"
    generation_config.write_text("{}\n", encoding="utf-8")
    client = root / "oracle" / "bin" / "vllm"
    client.parent.mkdir(parents=True)
    client.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    client.chmod(0o755)
    ninja = root / "oracle" / "bin" / "ninja"
    ninja.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ninja.chmod(0o755)
    cutlass = root / "oracle" / "cutlass"
    (cutlass / "include").mkdir(parents=True)
    (cutlass / "tools" / "util" / "include").mkdir(parents=True)
    (cutlass / "include" / "cutlass.h").write_text("fixture\n", encoding="utf-8")
    (cutlass / "tools" / "util" / "include" / "util.h").write_text(
        "fixture\n", encoding="utf-8"
    )
    cutlass_record = gdn_component._fingerprint_tree(cutlass)
    oracle_artifacts = {}
    for name in sorted(gdn_component._ORACLE_ARTIFACT_NAMES):
        if name == "client":
            path = client
        elif name == "ninja":
            path = ninja
        else:
            path = root / "oracle" / "artifacts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n", encoding="utf-8")
        oracle_artifacts[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    execution_dir = root / "execution"
    execution_dir.mkdir(parents=True)
    oracle_manifest = execution_dir / "27-oracle.json"
    oracle_manifest.write_text(
        json.dumps(
            {
                "artifacts": oracle_artifacts,
                "bench_dependencies": {
                    "flashinfer": FLASHINFER_VERSION,
                    "pandas": PANDAS_VERSION,
                },
                "client_contract_source_commit": VLLM_COMMIT,
                "cutlass_source_tree": cutlass_record,
                "generated_utc": "2026-07-14T00:00:00+00:00",
                "oracle_version": VLLM_ORACLE_VERSION,
                "runtime_version": VLLM_ORACLE_VERSION,
            }
        ),
        encoding="utf-8",
    )
    configure_log = root / "configure.log"
    configure_log.write_text(
        "CUDA compiler identification is NVIDIA 13.0.88\n"
        f"CUTLASS found at {cutlass}\n"
        "enabling sm120a NVFP4 cutlass GEMM\n",
        encoding="utf-8",
    )
    build_command = execution_dir / "27-build-command.txt"
    build_command.write_text(
        shlex.join(
            [
                "cmake",
                "--build",
                str(build),
                "--target",
                "server",
                "test_qwen27_paged_engine",
                "--parallel",
                str(len(gdn_component.os.sched_getaffinity(0))),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    build_log = execution_dir / "27-build.log"
    build_log.write_text("build passed\n", encoding="utf-8")
    cmake_cache = build / "CMakeCache.txt"
    compile_commands = build / "compile_commands.json"
    compile_tokens = [
        str(gdn_component.DGX_CUDA_COMPILER),
        "-DVLLM_CPP_FLASH_ATTN",
        "-DVLLM_CPP_TRITON=1",
        "-DVLLM_CPP_TRITON_CHUNKO_BF16=1",
        "-DVT_CUTLASS_NVFP4=1",
        "--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]",
        f"-I{cutlass / 'include'}",
        f"-I{cutlass / 'tools' / 'util' / 'include'}",
        "-c",
        str(SOURCE_ROOT / "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu"),
    ]
    compile_commands.write_text(
        json.dumps(
            [
                {
                    "arguments": compile_tokens,
                    "directory": str(build),
                    "file": str(
                        SOURCE_ROOT / "src/vt/cuda/cuda_matmul_nvfp4_cutlass.cu"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    cmake_cache.write_text(
        "\n".join(
            [
                "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo",
                f"CMAKE_CUDA_COMPILER:FILEPATH={gdn_component.DGX_CUDA_COMPILER}",
                "CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
                f"CMAKE_MAKE_PROGRAM:FILEPATH={ninja}",
                "VLLM_CPP_BENCH_PROFILE_CONTROL:BOOL=OFF",
                "VLLM_CPP_BUILD_TESTS:BOOL=ON",
                "VLLM_CPP_CUDA:BOOL=ON",
                "VLLM_CPP_CUDA_ARCHITECTURES:STRING=121a",
                "VLLM_CPP_FLASH_ATTN:BOOL=ON",
                "VLLM_CPP_SERVER:BOOL=ON",
                "VLLM_CPP_TRITON:BOOL=ON",
                "VLLM_CPP_TRITON_REGEN:BOOL=OFF",
                f"VLLM_CPP_CUTLASS_DIR:PATH={cutlass}",
                f"CMAKE_HOME_DIRECTORY:INTERNAL={SOURCE_ROOT}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "build_command": build_command,
        "build_log": build_log,
        "client": client,
        "cmake_cache": cmake_cache,
        "compile_commands": compile_commands,
        "configure_log": configure_log,
        "cuda_compiler": gdn_component.DGX_CUDA_COMPILER,
        "model_config": model_config,
        "oracle_manifest": oracle_manifest,
        "server": artifact,
        "snapshot:generation_config.json": generation_config,
        "tokenizer": tokenizer,
        "weight:model-00001-of-00001.safetensors": weight,
    }
    artifacts.update(
        {
            f"oracle:{name}": pathlib.Path(record["path"])
            for name, record in oracle_artifacts.items()
        }
    )
    artifact_records = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in artifacts.items()
    }
    native = root / "native-plan-must-not-exist.json"
    execution = {
        "artifacts": artifact_records,
        "bench_dependencies": {
            "flashinfer": FLASHINFER_VERSION,
            "pandas": PANDAS_VERSION,
        },
        "build_contract": {
            "build_type": "RelWithDebInfo",
            "compile_command_sha256": gdn_component._sha256_canonical(
                compile_tokens
            ),
            "cuda_compiler": str(gdn_component.DGX_CUDA_COMPILER),
            "cuda_compiler_sha256": sha256_file(gdn_component.DGX_CUDA_COMPILER),
            "cuda_compiler_version": "13.0.88",
            "cutlass_source_tree": cutlass_record,
            "native_plan_target": str(native),
            "native_plan_target_absent": True,
            "profile_control": False,
            "schema_version": gdn_component.BUILD_CONTRACT_SCHEMA_VERSION,
            "sm_architecture": "121a",
            "target_compile_definitions": [
                "VLLM_CPP_FLASH_ATTN",
                "VLLM_CPP_TRITON=1",
                "VLLM_CPP_TRITON_CHUNKO_BF16=1",
                "VT_CUTLASS_NVFP4=1",
            ],
            "triton_aot": True,
        },
        "cache_drop_roots": [
            str(snapshot.absolute()),
            str((root / "corpus" / "27").absolute()),
            str(artifact.absolute()),
            str(client.absolute()),
        ],
        "generated_utc": "2026-07-14T00:00:00+00:00",
        "host": {"kernel": "fixture", "machine": "fixture", "node": "fixture"},
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 32,
        "model_key": "27",
        "model_revision": MODEL_REVISIONS["27"],
        "num_blocks": 4736,
        "port": 8001,
        "snapshot_files": ["generation_config.json"],
        "vllm_cpp_sha": SOURCE_SHA,
        "vllm_oracle_version": VLLM_ORACLE_VERSION,
        "vllm_source_sha": VLLM_COMMIT,
        "weight_files": ["model-00001-of-00001.safetensors"],
    }
    execution_path = root / "execution" / "27-component.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    fixture = (
        SOURCE_ROOT
        / "tests/fixtures/nvfp4_flashinfer_v025_gb10/autotune_configs.json"
    )
    for arm in ARMS:
        log = root / "model-gates" / f"{arm}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        _write_model_gate_log(
            log,
            fixture=fixture,
            native=native,
            snapshot=snapshot,
        )
        command = [
            "/usr/bin/env",
            "-i",
            *_isolated_environment(
                arm=arm,
                fixture=fixture,
                native=native,
                home=root,
            ),
            str(model_gate_binary),
        ]
        (root / "model-gates" / f"{arm}-command.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
        gate = {
            "log": str(log),
            "log_sha256": sha256_file(log),
            "model_key": "27",
            "passed": True,
            "test_name": "test_qwen27_paged_engine",
            "vllm_cpp_sha": SOURCE_SHA,
        }
        (root / "model-gates" / f"{arm}.json").write_text(
            json.dumps(gate), encoding="utf-8"
        )

    raw_records = _records(packed_better=packed_better)
    memory_records = _memory(packed_better=packed_better)
    run_lines = ["corpus_validated", "gpu_lock_acquired path=/tmp/gpu"]
    for arm in ARMS:
        run_lines.append(f"model_gate_complete arm={arm}")
    run_lines.append("model_gates_validated")
    for concurrency in CONCURRENCIES:
        for order in LEG_ORDER:
            arm, repetition_text = order.rsplit("-r", 1)
            repetition = int(repetition_text)
            key = (concurrency, arm, repetition)
            tag = f"gdn-{arm}"
            raw = (
                root
                / "raw"
                / "27"
                / "ours"
                / f"c{concurrency}-r{repetition}-{tag}.json"
            )
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps(raw_records[key]), encoding="utf-8")

            base = root / "memory" / "27" / f"c{concurrency}" / arm
            base.mkdir(parents=True, exist_ok=True)
            memory = memory_records[key]
            samples = [
                {
                    "alive": True,
                    "gpu_memory_mib": memory["peak_gpu_memory_mib"],
                    "mem_available_kib": 1000000,
                    "peak_mem_available_drop_kib": memory[
                        "peak_mem_available_drop_kib"
                    ],
                    "peak_pss_kib": memory["peak_pss_kib"],
                    "peak_rss_kib": memory["peak_rss_kib"],
                    "pids": [10],
                    "pss_kib": memory["peak_pss_kib"],
                    "rss_kib": memory["peak_rss_kib"],
                },
                {
                    "alive": False,
                    "gpu_memory_mib": 0,
                    "mem_available_kib": 1000000,
                    "peak_mem_available_drop_kib": memory[
                        "peak_mem_available_drop_kib"
                    ],
                    "peak_pss_kib": memory["peak_pss_kib"],
                    "peak_rss_kib": memory["peak_rss_kib"],
                    "pids": [],
                    "pss_kib": 0,
                    "rss_kib": 0,
                },
            ]
            (base / f"r{repetition}.samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples),
                encoding="utf-8",
            )
            (base / f"r{repetition}.summary.json").write_text(
                json.dumps(
                    {
                        "peak_mem_available_drop_kib": memory[
                            "peak_mem_available_drop_kib"
                        ],
                        "peak_pss_kib": memory["peak_pss_kib"],
                        "peak_rss_kib": memory["peak_rss_kib"],
                        "samples": len(samples),
                    }
                ),
                encoding="utf-8",
            )

            cache = root / "cache-drop" / "27" / f"c{concurrency}" / arm
            cache.mkdir(parents=True, exist_ok=True)
            before_cache = cache / f"r{repetition}-before.json"
            after_cache = cache / f"r{repetition}-after.json"
            write_cache_drop_report(before_cache)
            write_cache_drop_report(after_cache)
            for cache_report in (before_cache, after_cache):
                cache_record = json.loads(cache_report.read_text())
                cache_record["roots"] = execution["cache_drop_roots"]
                cache_report.write_text(json.dumps(cache_record), encoding="utf-8")
            returned = root / "memory-return" / "27" / f"c{concurrency}" / arm
            returned.mkdir(parents=True, exist_ok=True)
            (returned / f"r{repetition}.json").write_text(
                json.dumps(
                    {
                        "cache_drops": {
                            "after": {
                                "path": str(after_cache),
                                "sha256": sha256_file(after_cache),
                            },
                            "before": {
                                "path": str(before_cache),
                                "sha256": sha256_file(before_cache),
                            },
                        },
                        "gpu_idle": True,
                        "baseline_mem_available_kib": 1000000,
                        "drop_caches_succeeded": True,
                        "final_mem_available_kib": 1000000,
                        "mem_available_within_tolerance": True,
                        "returned": True,
                        "tolerance_kib": 1048576,
                    }
                ),
                encoding="utf-8",
            )

            logs = root / "logs" / "27" / f"c{concurrency}" / arm
            logs.mkdir(parents=True, exist_ok=True)
            write_profile_log(
                logs / f"r{repetition}-server.log",
                fixture=fixture,
                native_target=native,
                server_pid=100 + repetition,
            )
            command = [
                "/usr/bin/env",
                "-i",
                *_isolated_environment(
                    arm=arm,
                    fixture=fixture,
                    native=native,
                    home=root,
                ),
                str(artifact),
                "--model",
                str(snapshot),
                "--port",
                "8001",
                "--num-blocks",
                "4736",
                "--max-num-seqs",
                "32",
                "--max-num-batched-tokens",
                "2048",
                "--no-enable-prefix-caching",
                "--served-model-name",
                "gate",
            ]
            (logs / f"r{repetition}-server-command.txt").write_text(
                shlex.join(command) + "\n", encoding="utf-8"
            )

            client_log = (
                root
                / "logs"
                / "27"
                / "ours"
                / f"c{concurrency}-r{repetition}-{tag}.log"
            )
            client_log.parent.mkdir(parents=True, exist_ok=True)
            corpus = root / "corpus" / "27" / "vllm" / (
                f"c{concurrency}-r{repetition}.jsonl"
            )
            run = OnlineRun(
                client=client,
                tokenizer=snapshot,
                evidence_root=root,
                model_key="27",
                engine="ours",
                base_url="http://127.0.0.1:8001",
                concurrency=concurrency,
                repetition=repetition,
                artifact_tag=tag,
            )
            client_log.write_text(
                "command: " + shlex.join(build_client_command(run)) + "\npassed\n",
                encoding="utf-8",
            )

            preflight = root / "preflight" / "27" / f"c{concurrency}" / arm
            preflight.mkdir(parents=True, exist_ok=True)
            (preflight / f"r{repetition}-stream.json").write_text(
                json.dumps(
                    {
                        "emitted_chunks": 128,
                        "first_chunk_s": 0.1,
                        "generated_text": "fixture",
                        "total_s": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            preflight_command = [
                "/usr/bin/env",
                "-i",
                *_clean_environment(home=root),
                "python3",
                str(SOURCE_ROOT / "tools/bench/run_serve_low.py"),
                "stream",
                "--url",
                "http://127.0.0.1:8001/v1/completions",
                "--corpus",
                str(root / "corpus" / "27" / f"c1-r{repetition}.jsonl"),
                "--output-len",
                "128",
                "--minimum-spread",
                "0.05",
                "--output",
                str(preflight / f"r{repetition}-stream.json"),
            ]
            (preflight / f"r{repetition}-stream-command.txt").write_text(
                shlex.join(preflight_command) + "\n", encoding="utf-8"
            )

            thermal = root / "thermal" / "27" / f"c{concurrency}" / arm
            thermal.mkdir(parents=True, exist_ok=True)
            for suffix in ("before", "after"):
                (thermal / f"r{repetition}-{suffix}.txt").write_text(
                    _thermal_snapshot(), encoding="utf-8"
                )
            run_lines.append(
                f"leg_begin concurrency={concurrency} arm={arm} repetition={repetition}"
            )
            run_lines.append(
                f"leg_end concurrency={concurrency} arm={arm} repetition={repetition}"
            )
    run_lines.extend(["gpu_series_complete", "gpu_lock_released path=/tmp/gpu"])
    (root / "component-order.log").write_text(
        "\n".join(run_lines) + "\n", encoding="utf-8"
    )
    (root / "component-run.log").write_text(
        "component execution complete\n", encoding="utf-8"
    )


class GdnPackedComponentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda_compiler_patch = mock.patch.object(
            gdn_component, "DGX_CUDA_COMPILER", pathlib.Path("/usr/bin/true")
        )
        cls.cuda_compiler_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.cuda_compiler_patch.stop()

    def test_plan_is_exact_ab_ba_ab_at_c2_and_c16(self) -> None:
        plan = build_component_plan(SOURCE_SHA)

        self.assertEqual(plan["concurrencies"], [2, 16])
        self.assertEqual(plan["requests_per_run"], {"2": 6, "16": 96})
        self.assertEqual(plan["order_per_concurrency"], list(LEG_ORDER))
        self.assertEqual(len(plan["legs"]), 12)
        self.assertEqual(
            [leg["arm"] for leg in plan["legs"][:6]],
            ["packed", "rollback", "rollback", "packed", "packed", "rollback"],
        )
        self.assertEqual(
            [leg["repetition"] for leg in plan["legs"][:6]],
            [1, 1, 2, 2, 3, 3],
        )
        self.assertTrue(plan["one_gpu_lock"])
        self.assertFalse(plan["production_build"]["profile_control"])

    def test_every_axis_win_passes_40_timing_and_8_memory_axes(self) -> None:
        summary = summarize_component_records(
            _records(packed_better=True), _memory(packed_better=True)
        )

        self.assertTrue(summary["gate_pass"])
        self.assertEqual(summary["axis_pass_count"], 40)
        self.assertEqual(summary["axis_total"], 40)
        self.assertEqual(summary["memory_axis_pass_count"], 8)
        self.assertEqual(summary["memory_axis_total"], 8)

    def test_valid_regression_is_a_completed_failed_gate(self) -> None:
        summary = summarize_component_records(
            _records(packed_better=False), _memory(packed_better=False)
        )

        self.assertFalse(summary["gate_pass"])
        self.assertEqual(summary["disposition"], "FAILED")
        self.assertEqual(summary["axis_pass_count"], 0)
        self.assertEqual(summary["memory_axis_pass_count"], 0)

    def test_stable_paired_reversal_cannot_pass_on_medians(self) -> None:
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        concurrency = 2
        requests = 6
        for arm, durations in (
            ("packed", (10.0, 10.39, 10.0)),
            ("rollback", (10.1, 10.0, 10.1)),
        ):
            for repetition, duration in enumerate(durations, start=1):
                record = records[(concurrency, arm, repetition)]
                record["duration"] = duration
                record["request_throughput"] = requests / duration
                record["output_throughput"] = requests * 128 / duration
                record["total_token_throughput"] = requests * (1024 + 128) / duration
                _set_latency(record, duration)
                value = 100.0 if arm == "packed" else 101.0
                if repetition == 2:
                    value = 103.9 if arm == "packed" else 100.0
                for axis in (
                    "peak_gpu_memory_mib",
                    "peak_pss_kib",
                    "peak_rss_kib",
                    "peak_mem_available_drop_kib",
                ):
                    memory[(concurrency, arm, repetition)][axis] = value

        # The c2 TTFT-family is now compared arm-to-arm on the pooled 18-sample
        # distribution, so decouple it from the engineered throughput/memory
        # per-rep reversal this test exercises: give it a clean, stable,
        # packed-better profile (uniform, so it neither voids on stability nor
        # flips the pooled comparison).  The throughput/memory/latency reversal at
        # repetition 2 remains what drives the paired-axis failure below.
        for arm in ARMS:
            for repetition in (1, 2, 3):
                ttft = 5.0 if arm == "packed" else 9.0
                _apply_ttft_ms(
                    records[(concurrency, arm, repetition)], [ttft] * requests
                )

        summary = summarize_component_records(records, memory)

        self.assertEqual(summary["axis_pass_count"], summary["axis_total"])
        self.assertEqual(
            summary["memory_axis_pass_count"], summary["memory_axis_total"]
        )
        self.assertLess(
            summary["paired_axis_pass_count"], summary["paired_axis_total"]
        )
        self.assertFalse(summary["gate_pass"])

    def test_outlier_that_hides_inside_passing_means_is_rejected(self) -> None:
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for concurrency in CONCURRENCIES:
            record = records[(concurrency, "packed", 3)]
            requests = {2: 6, 16: 96}[concurrency]
            record["duration"] = 14.0
            record["request_throughput"] = requests / 14.0
            record["output_throughput"] = requests * 128 / 14.0
            record["total_token_throughput"] = requests * (1024 + 128) / 14.0
            _set_latency(record, 14.0)
            record["start_times"] = [
                float(index // concurrency) * 2.0 for index in range(requests)
            ]
            for axis in (
                "peak_gpu_memory_mib",
                "peak_pss_kib",
                "peak_rss_kib",
                "peak_mem_available_drop_kib",
            ):
                memory[(concurrency, "packed", 3)][axis] *= 1.6

        with self.assertRaisesRegex(HarnessError, "unstable"):
            summarize_component_records(records, memory)

    def test_c2_ttft_bimodal_phase_lottery_is_pooled_and_accepted(self) -> None:
        # The c2 reps flip 3/3 vs 6/0 bimodal TTFT mixes (the upstream-mirrored
        # prefill co-schedule arrival lottery).  The per-rep mean_ttft swings
        # ~33%, so the retained non-tail 4% rule VOIDS this pre-change; after
        # pooling the c2 TTFT-family across the three reps it is ACCEPTED, and
        # every c2 TTFT axis equals the pooled 18-sample statistic.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition in (1, 2, 3):
            _apply_ttft_ms(
                records[(2, "packed", repetition)],
                _C2_TTFT_BIMODAL_PACKED[repetition],
            )
            _apply_ttft_ms(
                records[(2, "rollback", repetition)],
                _C2_TTFT_BIMODAL_ROLLBACK[repetition],
            )

        # The per-rep mean_ttft deviation is far past the non-tail 4% rule that
        # (still) voids every non-TTFT axis — this is exactly the run-3 signature.
        deviation = gdn_component._distribution(
            [records[(2, "packed", repetition)]["mean_ttft_ms"] for repetition in (1, 2, 3)],
            label="c2/packed/mean_ttft_ms",
        )["maximum_relative_deviation_from_median"]
        self.assertGreater(deviation, gdn_component.MAX_RUN_RELATIVE_DEVIATION)

        summary = summarize_component_records(records, memory)

        self.assertTrue(summary["gate_pass"])
        self.assertTrue(summary["all_repetitions_stable"])
        c2 = summary["by_concurrency"]["2"]
        for arm, fixture in (
            ("packed", _C2_TTFT_BIMODAL_PACKED),
            ("rollback", _C2_TTFT_BIMODAL_ROLLBACK),
        ):
            pool = [value for repetition in (1, 2, 3) for value in fixture[repetition]]
            expected = {
                "mean_ttft_ms": statistics.fmean(pool),
                "median_ttft_ms": statistics.median(pool),
                "p90_ttft_ms": gdn_component.percentile(pool, 90),
                "p99_ttft_ms": gdn_component.percentile(pool, 99),
            }
            for axis, value in expected.items():
                self.assertAlmostEqual(c2["ttft_pooled"][arm][axis], value)
                # The pooled value is what drives the arm comparison/ratios.
                self.assertAlmostEqual(c2["comparison_values"][arm][axis], value)
        stability = summary["contract"]["stability"]
        self.assertTrue(stability["c2_ttft_pooled"])
        self.assertEqual(stability["c2_ttft_pooled_concurrency"], 2)
        self.assertEqual(stability["c2_ttft_pooled_sanity_bound"], 0.5)
        self.assertEqual(
            stability["c2_ttft_pooled_axes"],
            sorted(f"{stat}_ttft_ms" for stat in ("mean", "median", "p90", "p99")),
        )

    def test_c2_ttft_per_rep_flip_is_excluded_from_paired_gate(self) -> None:
        # A rep where packed TTFT > rollback (rollback r1 dips low by arrival
        # phasing) would, under per-rep pairing, manufacture a spurious
        # packed-vs-rollback TTFT regression; but the pooled arm comparison has
        # packed <= rollback on every TTFT axis, so the component is ACCEPTED and
        # the c2 TTFT-family is excluded from the gated paired axes (it remains a
        # reported diagnostic).
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition in (1, 2, 3):
            _apply_ttft_ms(
                records[(2, "packed", repetition)], _C2_TTFT_FLIP_PACKED[repetition]
            )
            _apply_ttft_ms(
                records[(2, "rollback", repetition)],
                _C2_TTFT_FLIP_ROLLBACK[repetition],
            )

        summary = summarize_component_records(records, memory)

        self.assertTrue(summary["gate_pass"])
        c2 = summary["by_concurrency"]["2"]
        # The diagnostic per-rep ratio DOES show the r1 flip (packed worse < 1.0)...
        self.assertLess(
            c2["paired_normalized_ratios_ge_1_is_packed_better"]["r1"]["mean_ttft_ms"],
            1.0,
        )
        # ...but the c2 TTFT-family axes are not in the gated paired set.
        for repetition in ("r1", "r2", "r3"):
            for axis in ("mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms"):
                self.assertNotIn(axis, c2["paired_axis_pass"][repetition])
        # The pooled arm comparison keeps packed <= rollback on every TTFT axis.
        for axis in ("mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms"):
            self.assertLessEqual(
                c2["comparison_values"]["packed"][axis],
                c2["comparison_values"]["rollback"][axis],
            )

    def test_broken_c2_ttft_leg_beyond_pooled_bound_still_voids(self) -> None:
        # Same bimodal fixture, but one packed leg is hung at ~5x: its per-rep
        # aggregate lands far past the 50% pooled sanity bound, so it still VOIDS.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition in (1, 2, 3):
            _apply_ttft_ms(
                records[(2, "packed", repetition)],
                _C2_TTFT_BIMODAL_PACKED[repetition],
            )
            _apply_ttft_ms(
                records[(2, "rollback", repetition)],
                _C2_TTFT_BIMODAL_ROLLBACK[repetition],
            )
        _apply_ttft_ms(records[(2, "packed", 3)], [50.0] * 6)  # 5x hung leg

        with self.assertRaisesRegex(HarnessError, "unstable"):
            summarize_component_records(records, memory)

    def test_c16_ttft_mean_is_not_pooled_and_voids_beyond_4pct(self) -> None:
        # c16 is NEVER pooled: a c16 mean/median TTFT swing past the retained 4%
        # rule still VOIDS (a pooled 50% bound would wrongly accept it), proving
        # pooling is confined to c2.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition, ttft in ((1, 20.0), (2, 20.0), (3, 22.0)):
            _apply_ttft_ms(records[(16, "rollback", repetition)], [ttft] * 96)

        with self.assertRaises(HarnessError) as context:
            summarize_component_records(records, memory)
        message = str(context.exception)
        self.assertIn("c16 repetitions are unstable", message)
        self.assertIn("mean_ttft_ms", message)

    def test_c16_ttft_tail_only_instability_within_15pct_is_accepted(self) -> None:
        # The 15% tail tolerance still governs c16 TTFT tails (c16 is not pooled):
        # a c16 p99 tail at ~11% (mean/median/p90/throughput/memory all stable)
        # voids under the uniform 4% rule but must be ACCEPTED under the tail rule.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition, (low, high) in _C16_TTFT_TAIL_WITHIN_15PCT.items():
            _apply_ttft_ms(records[(16, "rollback", repetition)], _c16_tail_ttft(low, high))

        deviation = gdn_component._distribution(
            [records[(16, "rollback", repetition)]["p99_ttft_ms"] for repetition in (1, 2, 3)],
            label="c16/rollback/p99_ttft_ms",
        )["maximum_relative_deviation_from_median"]
        self.assertGreater(deviation, gdn_component.MAX_RUN_RELATIVE_DEVIATION)
        self.assertLess(deviation, 0.15)

        summary = summarize_component_records(records, memory)

        self.assertTrue(summary["all_repetitions_stable"])
        self.assertTrue(summary["gate_pass"])
        stability = summary["contract"]["stability"]
        self.assertEqual(stability["maximum_relative_deviation_from_median"], 0.04)
        self.assertEqual(
            stability["maximum_tail_relative_deviation_from_median"], 0.15
        )
        self.assertEqual(
            stability["tail_axes"],
            sorted(
                f"{stat}_{metric}_ms"
                for metric in ("ttft", "tpot", "itl", "e2el")
                for stat in ("p90", "p99")
            ),
        )

    def test_c16_ttft_tail_instability_beyond_15pct_still_voids(self) -> None:
        # A c16 TTFT p99 tail past 15% still voids, so genuine tail blowups remain
        # caught even where the 15% rule (not pooling) governs.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        for repetition, (low, high) in _C16_TTFT_TAIL_BEYOND_15PCT.items():
            _apply_ttft_ms(records[(16, "rollback", repetition)], _c16_tail_ttft(low, high))

        deviation = gdn_component._distribution(
            [records[(16, "rollback", repetition)]["p99_ttft_ms"] for repetition in (1, 2, 3)],
            label="c16/rollback/p99_ttft_ms",
        )["maximum_relative_deviation_from_median"]
        self.assertGreater(deviation, 0.15)

        with self.assertRaisesRegex(HarnessError, "unstable"):
            summarize_component_records(records, memory)

    def test_non_tail_instability_at_5pct_still_voids(self) -> None:
        # A mean/throughput axis stays gated at 4%: a ~5% throughput swing on a
        # single repetition still voids, unaffected by the tail tolerance.
        records = _records(packed_better=True)
        memory = _memory(packed_better=True)
        requests = 6
        record = records[(2, "packed", 1)]
        record["duration"] = 10.53
        record["request_throughput"] = requests / 10.53
        record["output_throughput"] = requests * 128 / 10.53
        record["total_token_throughput"] = requests * (1024 + 128) / 10.53

        with self.assertRaisesRegex(HarnessError, "unstable"):
            summarize_component_records(records, memory)

    def test_missing_evidence_leg_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(HarnessError, "missing evidence artifact"):
                summarize_evidence(pathlib.Path(directory), SOURCE_SHA)

    def test_missing_driver_run_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            (root / "component-run.log").unlink()

            with self.assertRaisesRegex(HarnessError, "driver run log"):
                summarize_evidence(root, SOURCE_SHA)

    def test_complete_evidence_revalidates_provenance_order_plans_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)

            summary = summarize_evidence(root, SOURCE_SHA)

            self.assertTrue(summary["gate_pass"])
            self.assertTrue(summary["correctness_pass"])
            self.assertTrue(summary["one_lock_order_pass"])
            self.assertTrue(summary["all_process_plan_maps_equal"])

    def test_profile_instrumented_build_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "execution" / "27-component.json"
            execution = json.loads(path.read_text())
            execution["build_contract"]["profile_control"] = True
            path.write_text(json.dumps(execution))

            with self.assertRaisesRegex(HarnessError, "production build"):
                summarize_evidence(root, SOURCE_SHA)

    def test_execution_provenance_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "execution" / "27-component.json"
            execution = json.loads(path.read_text())
            execution["bench_dependencies"]["flashinfer"] = "attacker-controlled"
            path.write_text(json.dumps(execution), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "benchmark dependencies"):
                summarize_evidence(root, SOURCE_SHA)

    def test_missing_or_drifted_corpus_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            manifest = root / "corpus/27/vllm/manifest.json"
            manifest.unlink()

            with self.assertRaisesRegex(HarnessError, "corpus manifest"):
                summarize_evidence(root, SOURCE_SHA)

    def test_corpus_partition_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            partition = root / "corpus/27/vllm/c2-r1.jsonl"
            partition.write_text(
                partition.read_text(encoding="utf-8") + '{"prompt":"drift"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(HarnessError, "corpus partition"):
                summarize_evidence(root, SOURCE_SHA)

    def test_raw_latency_summaries_must_match_detailed_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "raw/27/ours/c2-r1-gdn-packed.json"
            record = json.loads(path.read_text())
            for metric in ("ttft", "tpot", "itl", "e2el"):
                for stat in ("mean", "median", "p90", "p99"):
                    record[f"{stat}_{metric}_ms"] = 0.001
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "recomputed timing"):
                summarize_evidence(root, SOURCE_SHA)

    def test_impossible_duration_is_rejected_against_detailed_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "raw/27/ours/c2-r1-gdn-packed.json"
            record = json.loads(path.read_text())
            record["duration"] = 0.001
            record["request_throughput"] = record["completed"] / record["duration"]
            record["output_throughput"] = (
                record["total_output_tokens"] / record["duration"]
            )
            record["total_token_throughput"] = (
                record["total_input_tokens"] + record["total_output_tokens"]
            ) / record["duration"]
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "detailed request span"):
                summarize_evidence(root, SOURCE_SHA)

    def test_pinned_perf_counter_skew_is_accepted_within_fixed_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "raw/27/ours/c2-r1-gdn-packed.json"
            record = json.loads(path.read_text())
            skew_ms = 0.00004
            for stat in ("mean", "median", "p90", "p99"):
                record[f"{stat}_e2el_ms"] -= skew_ms
                record[f"{stat}_tpot_ms"] -= skew_ms / 127
            path.write_text(json.dumps(record), encoding="utf-8")

            self.assertTrue(summarize_evidence(root, SOURCE_SHA)["gate_pass"])

    def test_empty_merged_delta_itl_row_remains_upstream_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            for concurrency in CONCURRENCIES:
                for arm in ARMS:
                    for repetition in (1, 2, 3):
                        path = (
                            root
                            / "raw/27/ours"
                            / f"c{concurrency}-r{repetition}-gdn-{arm}.json"
                        )
                        record = json.loads(path.read_text())
                        record["itls"][0] = []
                        metrics = gdn_component._recompute_timing_metrics(record)
                        for axis in gdn_component.LOWER_AXES:
                            record[axis] = metrics[axis]
                        path.write_text(json.dumps(record), encoding="utf-8")

            self.assertTrue(summarize_evidence(root, SOURCE_SHA)["gate_pass"])

    def test_leg_order_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "component-order.log"
            lines = path.read_text().splitlines()
            first = lines.index("leg_begin concurrency=2 arm=packed repetition=1")
            second = lines.index("leg_begin concurrency=2 arm=rollback repetition=1")
            lines[first], lines[second] = lines[second], lines[first]
            path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(HarnessError, "AB/BA/AB order"):
                summarize_evidence(root, SOURCE_SHA)

    def test_model_gates_outside_gpu_lock_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "component-order.log"
            lines = path.read_text().splitlines()
            lines.remove("model_gate_complete arm=packed")
            lines.insert(0, "model_gate_complete arm=packed")
            path.write_text("\n".join(lines) + "\n")

            with self.assertRaisesRegex(HarnessError, "correctness gates"):
                summarize_evidence(root, SOURCE_SHA)

    def test_skipped_model_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            log = root / "model-gates" / "packed.log"
            log.write_text(
                "27B checkpoint absent; skipping (dgx-only)\n"
                "[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped\n"
                "[doctest] Status: SUCCESS!\n",
                encoding="utf-8",
            )
            gate_path = root / "model-gates" / "packed.json"
            gate = json.loads(gate_path.read_text())
            gate["log_sha256"] = sha256_file(log)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "skipped"):
                summarize_evidence(root, SOURCE_SHA)

    def test_model_gate_wrong_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            log = root / "model-gates" / "packed.log"
            log.write_text(
                log.read_text().replace(
                    str(root / MODEL_REVISIONS["27"]),
                    str(root / "wrong-snapshot"),
                ),
                encoding="utf-8",
            )
            gate_path = root / "model-gates" / "packed.json"
            gate = json.loads(gate_path.read_text())
            gate["log_sha256"] = sha256_file(log)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "snapshot"):
                summarize_evidence(root, SOURCE_SHA)

    def test_model_gate_external_log_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            canonical = root / "model-gates" / "packed.log"
            external = root / "external-packed.log"
            external.write_bytes(canonical.read_bytes())
            gate_path = root / "model-gates" / "packed.json"
            gate = json.loads(gate_path.read_text())
            gate["log"] = str(external)
            gate["log_sha256"] = sha256_file(external)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "log path"):
                summarize_evidence(root, SOURCE_SHA)

    def test_plan_map_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server.log"
            text = path.read_text()
            path.write_text(text.replace(" tactic=0\n", " tactic=1\n", 1))

            with self.assertRaisesRegex(HarnessError, "selected FP4 plan map"):
                summarize_evidence(root, SOURCE_SHA)

    def test_server_command_model_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server-command.txt"
            tokens = shlex.split(path.read_text())
            tokens[tokens.index("--model") + 1] = "/wrong-snapshot"
            path.write_text(shlex.join(tokens) + "\n")

            with self.assertRaisesRegex(HarnessError, "server command"):
                summarize_evidence(root, SOURCE_SHA)

    def test_server_command_port_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server-command.txt"
            tokens = shlex.split(path.read_text())
            tokens[tokens.index("--port") + 1] = "8002"
            path.write_text(shlex.join(tokens) + "\n")

            with self.assertRaisesRegex(HarnessError, "server command"):
                summarize_evidence(root, SOURCE_SHA)

    def test_hidden_server_environment_control_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server-command.txt"
            tokens = shlex.split(path.read_text())
            tokens.insert(2, "LD_PRELOAD=/tmp/hidden.so")
            path.write_text(shlex.join(tokens) + "\n")

            with self.assertRaisesRegex(HarnessError, "environment is not exact"):
                summarize_evidence(root, SOURCE_SHA)

    def test_server_home_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server-command.txt"
            tokens = shlex.split(path.read_text())
            index = next(
                index for index, token in enumerate(tokens) if token.startswith("HOME=")
            )
            tokens[index] = "HOME=/tmp"
            path.write_text(shlex.join(tokens) + "\n")

            with self.assertRaisesRegex(HarnessError, "environment HOME"):
                summarize_evidence(root, SOURCE_SHA)

    def test_extra_server_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/c2/packed/r1-server-command.txt"
            tokens = shlex.split(path.read_text())
            tokens.extend(["--block-size", "8"])
            path.write_text(shlex.join(tokens) + "\n")

            with self.assertRaisesRegex(HarnessError, "server command differs"):
                summarize_evidence(root, SOURCE_SHA)

    def test_client_command_log_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "logs/27/ours/c2-r1-gdn-packed.log"
            path.write_text("arbitrary successful text\n", encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "client command log"):
                summarize_evidence(root, SOURCE_SHA)

    def test_stream_preflight_command_and_result_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "preflight/27/c2/packed/r1-stream.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "stream preflight"):
                summarize_evidence(root, SOURCE_SHA)

    def test_contradictory_memory_return_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "memory-return/27/c2/packed/r1.json"
            record = json.loads(path.read_text())
            record["baseline_mem_available_kib"] = 10000000
            record["final_mem_available_kib"] = 0
            record["tolerance_kib"] = gdn_component.MEMORY_RETURN_TOLERANCE_KIB
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "memory-return predicates"):
                summarize_evidence(root, SOURCE_SHA)

    def test_memory_return_tolerance_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "memory-return/27/c2/packed/r1.json"
            record = json.loads(path.read_text())
            record["baseline_mem_available_kib"] = 10000000
            record["final_mem_available_kib"] = 0
            record["tolerance_kib"] = 10000000
            path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "tolerance"):
                summarize_evidence(root, SOURCE_SHA)

    def test_thermal_throttling_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "thermal/27/c2/packed/r1-after.txt"
            path.write_text(_thermal_snapshot(hw_thermal_active=True), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "thermal throttle"):
                summarize_evidence(root, SOURCE_SHA)

    def test_thermal_counter_increase_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "thermal/27/c2/packed/r1-after.txt"
            path.write_text(_thermal_snapshot(hw_thermal_us=1), encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "thermal counter increased"):
                summarize_evidence(root, SOURCE_SHA)

    def test_malformed_thermal_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "thermal/27/c2/packed/r1-before.txt"
            path.write_text("not nvidia-smi output\n", encoding="utf-8")

            with self.assertRaisesRegex(HarnessError, "thermal snapshot"):
                summarize_evidence(root, SOURCE_SHA)

    def test_finalizer_writes_manifest_then_marker_for_a_failed_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root, packed_better=False)

            status = finalize_evidence(root, SOURCE_SHA)

            summary = root / "component-summary.json"
            manifest = root / "component-manifest.json"
            marker = root / "component-status.json"
            self.assertTrue(summary.is_file())
            self.assertTrue(manifest.is_file())
            self.assertTrue(marker.is_file())
            self.assertEqual(status["status"], "complete-failed")
            self.assertFalse(status["benchmark_binding"])
            self.assertFalse(status["speed_credit"])
            self.assertEqual(status["summary_sha256"], sha256_file(summary))
            self.assertEqual(status["manifest_sha256"], sha256_file(manifest))

    def test_finalizer_seals_unstable_evidence_as_void(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            path = root / "raw/27/ours/c2-r3-gdn-packed.json"
            record = json.loads(path.read_text())
            record["duration"] *= 2.0
            record["request_throughput"] /= 2.0
            record["output_throughput"] /= 2.0
            record["total_token_throughput"] /= 2.0
            path.write_text(json.dumps(record), encoding="utf-8")

            status = finalize_evidence(root, SOURCE_SHA)

            summary = json.loads((root / "component-summary.json").read_text())
            self.assertEqual(summary["disposition"], "VOID")
            self.assertIn("unstable", summary["validation_error"])
            self.assertEqual(status["status"], "complete-void")
            self.assertFalse(status["speed_credit"])

    def test_post_seal_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            run_log = root / "component-run.log"
            finalize_evidence(root, SOURCE_SHA)
            run_log.write_text("component execution complete\nlate output\n")

            with self.assertRaisesRegex(HarnessError, "artifact manifest drifted"):
                verify_finalized_evidence(root, SOURCE_SHA)

    def test_symlinked_evidence_directory_is_rejected_before_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            raw = root / "raw"
            external_raw = pathlib.Path(external) / "raw"
            shutil.copytree(raw, external_raw)
            shutil.rmtree(raw)
            raw.symlink_to(external_raw, target_is_directory=True)

            with self.assertRaisesRegex(HarnessError, "contains a symlink"):
                finalize_evidence(root, SOURCE_SHA)

    def test_driver_dry_run_emits_the_same_exact_plan(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                str(root / "scripts/dgx-gdn-packed-component.sh"),
                "--dry-run",
                "--vllm-cpp-sha",
                SOURCE_SHA,
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), build_component_plan(SOURCE_SHA))

    def test_driver_closes_run_log_before_marker_last_finalization(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        text = (root / "scripts/dgx-gdn-packed-component.sh").read_text()

        self.assertNotIn("exec > >(tee", text)
        self.assertLess(
            text.index("exec 1>&3 2>&4"),
            text.index('gdn_packed_component.py" finalize'),
        )

    def test_driver_validates_model_gates_before_first_timing_leg(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()

        self.assertLess(
            text.index('gdn_packed_component.py" validate-model-gates'),
            text.index("for concurrency in 2 16"),
        )

    def test_driver_records_profile_control_off_in_execution_manifest(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()
        command_start = text.index(
            'python3 "${repo_root}/tools/bench/online_gate.py" record-execution'
        )
        command_end = text.index("\nvllm_corpus=", command_start)
        command = text[command_start:command_end].replace("\\\n", " ")
        tokens = shlex.split(command)

        self.assertIn("--profile-control", tokens)
        profile_control = tokens.index("--profile-control")
        self.assertEqual(
            tokens[profile_control : profile_control + 2],
            ["--profile-control", "off"],
        )
        self.assertEqual(tokens.count("--profile-control"), 1)

    def test_gpu_idle_probe_failure_is_not_idle(self) -> None:
        failed = subprocess.CompletedProcess(
            args=["/usr/bin/nvidia-smi"], returncode=1, stdout="", stderr="failed"
        )
        with mock.patch.object(gdn_component.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(HarnessError, "GPU-idle probe failed"):
                gdn_component.probe_gpu_idle()

    def test_driver_signal_traps_exit_after_cleanup(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()

        self.assertIn("trap 'handle_signal 130' INT", text)
        self.assertIn("trap 'handle_signal 143' TERM", text)

    def test_driver_execute_requires_complete_pre_gpu_provenance(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                str(root / "scripts/dgx-gdn-packed-component.sh"),
                "--execute",
                "--vllm-cpp-sha",
                SOURCE_SHA,
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("--evidence is required", completed.stderr)

    def test_diagnostic_marker_refuses_finalize(self) -> None:
        # A component-diagnostic.json marker makes both the summary and the
        # finalizer refuse to seal a component from diagnostic evidence.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            (root / "component-diagnostic.json").write_text(
                json.dumps({"diagnostic": True, "mode": "diagnostic-c16"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HarnessError, "diagnostic"):
                summarize_evidence(root, SOURCE_SHA)
            with self.assertRaisesRegex(HarnessError, "diagnostic"):
                finalize_evidence(root, SOURCE_SHA)

        # A diagnostic/ subtree is equally disqualifying.
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_complete_evidence(root)
            subtree = root / "diagnostic" / "c16" / "packed"
            subtree.mkdir(parents=True)
            (subtree / "r1-error-body.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(HarnessError, "diagnostic"):
                summarize_evidence(root, SOURCE_SHA)
            with self.assertRaisesRegex(HarnessError, "diagnostic"):
                finalize_evidence(root, SOURCE_SHA)

    def test_driver_defines_diagnostic_c16_mode(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()
        # --diagnostic-c16 shares the single-mode case arm with the others.
        self.assertRegex(text, r"--dry-run\|--execute\|--diagnostic-c16")
        start = text.index(DIAG_BEGIN)
        end = text.index(DIAG_END)
        self.assertLess(start, end)
        branch = text[start:end]
        self.assertIn("component-diagnostic.json", branch)
        self.assertNotIn("component-summary.json", branch)
        self.assertNotIn("component-status.json", branch)
        self.assertNotIn('gdn_packed_component.py" finalize', branch)

    def test_driver_diagnostic_runs_only_packed_c16_boundary(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()
        branch = text[text.index(DIAG_BEGIN) : text.index(DIAG_END)]
        self.assertIn("for rep in 1 2 3", branch)
        self.assertIn("--concurrency 16", branch)
        self.assertIn("VT_GDN_DIAG_STEP_LOG=1", branch)
        self.assertIn("packed", branch)
        self.assertNotIn("for concurrency in 2 16", branch)
        self.assertNotIn("rollback", branch)
        self.assertNotIn("run_model_gate", branch)
        # Exactly one GPU lock across the whole diagnostic series.
        self.assertEqual(branch.count("flock 9"), 1)

    def test_driver_diagnostic_wraps_bench_to_survive_set_e(self) -> None:
        text = (SOURCE_ROOT / "scripts/dgx-gdn-packed-component.sh").read_text()
        branch = text[text.index(DIAG_BEGIN) : text.index(DIAG_END)]
        self.assertIn("|| bench_failed=1", branch)
        self.assertIn("diagnostic-error-body", branch)
        # The error-body capture and cleanup follow the failure-tolerant bench.
        self.assertLess(
            branch.index("|| bench_failed=1"), branch.index("diagnostic-error-body")
        )
        self.assertLess(
            branch.index("diagnostic-error-body"), branch.index("cleanup_server")
        )
        # The evidence directory basename must carry the diagnostic-c16 marker.
        self.assertRegex(text, r"basename[^\n]*diagnostic-c16")


if __name__ == "__main__":
    unittest.main()
