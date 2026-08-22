#!/usr/bin/env python3
"""Capture the SPEC-DFLASH2 oracle arm: the instrument O23 described and #1562 owed.

`.agents/specs/dflash2-spec-decode.md` `## Owed` O23 exists so that the next
agent to instrument this oracle does not pay three 51.75 GiB model loads again.
It described the repaired instrument in prose and committed none of it
([#1562](https://github.com/mudler/vllm.cpp/issues/1562)). This is that
instrument, committed, with the three failures it was repaired for encoded as
refusals rather than as paragraphs.

The three failures, and where each one lives in this file
---------------------------------------------------------
1. **The engine core is a SEPARATE PROCESS by default.** vLLM V1 spawns
   `EngineCore`, which re-imports vllm clean, so an in-process monkeypatch never
   lands on the object that drafts. The first capture generated 4 x 64 tokens,
   vLLM counted 50 drafts / 350 draft tokens / 209 accepted, and the hook
   recorded ZERO blocks. `VLLM_ENABLE_V1_MULTIPROCESSING=0` is the switch, and
   `_assert_inproc_client` asserts the resolved CLIENT CLASS rather than
   trusting the variable.
2. **`capture_model()` calls into the draft path**, so a `.tolist()` in the hook
   is a device-to-host copy inside a CUDA graph capture, which torch refuses
   (`RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
   capture`). The hook delegates whenever
   `torch.cuda.is_current_stream_capturing()` and counts the skip.
3. **`_generate_draft` IS THE WRONG SEAM** under the production configuration.
   `DFlashSpeculator.propose` branches `if batch_desc.cg_mode ==
   CUDAGraphMode.FULL: run_fullgraph(...) else: self._generate_draft(...)` and
   returns `self.draft_tokens[:num_reqs]` on BOTH arms, so with FULL decode
   graphs -- which is what a denominator must use -- the replay never enters
   `_generate_draft` in Python at all. That run asserted
   `ENGINE_CORE_CLIENT=InprocClient` AND `HOOK_ON_CLASS=traced`, resolved
   `TRITON_ATTN`, generated 4 x 64 tokens, vLLM counted 47 drafts / 329 draft
   tokens / 216 accepted, and the hook STILL recorded zero blocks. Both
   preconditions were true and the instrument was blind. **The hook therefore
   wraps `propose`, below the branch, and reads its RETURN VALUE**, which is the
   one seam both arms pass through.

   Only the ABORT ON ZERO caught all three, so it is unconditional and it is not
   optional. `dflash2_speed_harness.hook_reasons` holds it.

What this does NOT do
---------------------
It does not fall back. A backend that cannot be read off the built engine is a
REFUSAL, never a label copied from the request kwarg and never one corrected
from a log afterwards -- that is precisely the #1562 defect, and
`attention_backend_reasons` refuses the relabel by its SOURCE and by the PROBE
it names, so no future harness reintroduces it by transcribing one.

It measures ONE axis. `time.perf_counter()` around `generate` gives output
throughput, folded by the shared `dflash2_speed_harness.fold_legs` so both arms
fold the same statistic. It does NOT produce `ttft_ms` or `peak_device_bytes`,
because our arm cannot, and it does not derive `tpot_ms` from wall time over
completion tokens -- that quotient is `output_throughput_tok_s` restated, and a
second axis that is algebraically the first one is a number, not a measurement.
Those three stay `NOT MEASURED` and O26 says so.

What it emits is a GOLDEN. `records[i].blocks`, each block carrying `call`,
`req_row`, `anchor` and `drafts`, beside `records[i].num_blocks` and the
top-level keys `GOLDEN_TOP_LEVEL_KEYS` names -- the exact shape
`tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads. A capture whose output
the consumer cannot load could not give the FLASH_ATTN golden's label the
provenance O26 owes it.

Two phases, and why they are separate
-------------------------------------
`--precheck-only` runs every precondition that needs no GPU and no wheel: the
keepalive, the checkpoint hashes, the contention state, both build recipes and
the denominator's configuration. It exists so the failure that costs a lease is
found before the lease, and so the refusal paths that protect every future
measurement are reachable from CPU CI, where the tests drive this exact
`main()`.

Usage on a leased `dgx:gpu0` (see `scripts/dflash2-speed-gate.sh`, which is the
supported way to run it):

    VT_SERVER_SSE_PING_S=0 python3 -m tools.bench.dflash2_oracle_capture \\
        --target /workspace/ckpt/qwen3.8-27b-hf \\
        --draft /workspace/dflash2/draft-st \\
        --oracle-commit 3406ec1d... \\
        --attention-backend TRITON_ATTN \\
        --artifact target=/workspace/ckpt/qwen3.8-27b-hf/model.safetensors=<sha256> \\
        --lease-id "$RC_LEASE_ID" \\
        --our-revision "$(git rev-parse HEAD)" \\
        --our-build-recipe "cmake --preset cuda-release -DVLLM_CPP_CUDA_ARCH=121a" \\
        --oracle-build-recipe "pip install -e . at <the oracle head>" \\
        --repeat 5 \\
        --clock-summary evidence/clock-vllm.json \\
        --output evidence/vllm-arm.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.dflash2_speed_harness import (
    BACKEND_PROBES,
    BACKEND_SOURCE_READ_BACK,
    SSE_PING_ENV,
    attention_backend_reasons,
    build_recipe_reasons,
    checkpoint_reasons,
    clock_state_reasons,
    contention_reasons,
    denominator_reasons,
    fold_legs,
    hook_reasons,
    leg_reasons,
    model_binding_reasons,
    oracle_build_reasons,
    oracle_identity_reasons,
    repeat_reasons,
    require_no_reasons,
    sse_keepalive_reasons,
    workload_fingerprint,
)
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    write_json_atomic,
)

#: The class whose `propose` both graph modes return through. Overridable,
#: because the oracle head is beyond-pin and a rename must be a flag rather than
#: a patch to this file.
DEFAULT_SPECULATOR_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
DEFAULT_SPECULATOR_CLASS = "DFlashSpeculator"

PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "def fibonacci(n):",
    "The three laws of robotics are",
    "In a shocking finding, scientists discovered",
)


def _parse_artifact(item: str) -> dict[str, str]:
    """`role=path=sha256` -- all three, because two of them prove nothing."""

    parts = str(item).split("=")
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise HarnessError(
            f"--artifact expects role=path=sha256, got {item!r}. A repo id is not a "
            "pin: checkpoints get re-quantized in place under an unchanged name"
        )
    return {"role": parts[0], "path": parts[1], "sha256": parts[2].strip().lower()}


def sample_compute_processes(smi: str = "nvidia-smi") -> list[dict[str, Any]]:
    """Who else is on the device right now, straight from the driver.

    A failed query is a refusal to SAMPLE and is raised, never folded into an
    empty list: an empty list reads as "the box was ours", which is the one
    conclusion an unusable instrument may not deliver.
    """

    try:
        completed = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise HarnessError(
            f"cannot sample GPU contention with {smi!r}: {error}. Absence of "
            "information is not absence of a competing job"
        ) from error
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2 or not fields[0]:
            continue
        processes.append(
            {
                "pid": int(fields[0]),
                "name": fields[1],
                "used_memory": fields[2] if len(fields) > 2 else "",
            }
        )
    return processes


def _load_clock(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HarnessError(f"cannot read the clock summary {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"clock summary {path} is not JSON: {error}") from error


def precheck(args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    """Every precondition that needs no GPU and no wheel, checked TOGETHER.

    Returns the block it verified so the capture phase records exactly what was
    checked rather than re-deriving it, and so `--precheck-only` prints the same
    object the full run embeds.
    """

    artifacts = [_parse_artifact(item) for item in args.artifact]
    config = {
        "enforce_eager": bool(args.enforce_eager),
        "max_num_seqs": args.max_num_seqs,
        "attention_backend_declared": args.attention_backend,
    }
    contention = {
        "lease_id": args.lease_id,
        "own_pids": [os.getpid()],
        "compute_processes": (
            list(args.assume_compute_processes)
            if args.assume_compute_processes is not None
            else sample_compute_processes()
        ),
    }
    prompts = list(args.prompt) or list(PROMPTS)
    workload = {
        "prompts_sha256": workload_fingerprint(prompts),
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "num_speculative_tokens": args.num_speculative_tokens,
        "max_num_seqs": args.max_num_seqs,
        "repeat": args.repeat,
        "concurrency": 1,
        "temperature": 0.0,
        "seed": None,
    }
    # `build` IS THE ARM'S OWN BUILD. This capture runs from our checkout, so
    # our revision is the one nearest to hand -- and putting it here made the
    # summarizer attribute a `cmake --preset` line to a pip-installed wheel and
    # print it as the denominator's recipe. Our tree is provenance for the
    # INSTRUMENT and is recorded as `harness_build`, which is what it is.
    build = {"revision": args.oracle_commit, "build_recipe": args.oracle_build_recipe}
    harness_build = {"revision": args.our_revision, "build_recipe": args.our_build_recipe}
    models = {"target": args.target, "draft": args.draft}

    reasons: list[str] = []
    reasons += sse_keepalive_reasons(env)
    reasons += denominator_reasons(config)
    reasons += checkpoint_reasons(artifacts)
    reasons += contention_reasons(contention)
    reasons += model_binding_reasons(models, artifacts, label="vllm")
    reasons += repeat_reasons(args.repeat, label="vllm")
    reasons += build_recipe_reasons(harness_build, label="ours")
    # NOT `build_recipe_reasons`: that only asks whether the block is populated,
    # and the block was populated with OUR revision. `oracle_build_reasons` holds
    # it to the oracle head this same run declares.
    reasons += oracle_build_reasons(
        {"oracle_expected_commit": args.oracle_commit, "build": build}
    )
    if not str(args.oracle_commit or "").strip():
        reasons.append(
            "oracle: --oracle-commit is required. This row's oracle is BEYOND-PIN, so "
            "there is no parity pin to fall back to and a silent fallback would "
            "measure a wheel that cannot load DFlash2DraftModel at all"
        )
    require_no_reasons(reasons, what="DFlash2 oracle capture")
    return {
        "artifacts": artifacts,
        "config": config,
        "workload": workload,
        "prompts": prompts,
        "contention": contention,
        "models": models,
        "build": build,
        "harness_build": harness_build,
        "env": {SSE_PING_ENV: env[SSE_PING_ENV]},
    }


def _walk(root: Any, dotted: str) -> Any:
    node = root
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def resolve_attention_backend(llm: Any) -> tuple[str, str]:
    """Read the RESOLVED backend off the BUILT engine, or refuse.

    O22 lays the rule down and the #1562 golden does not meet it. This function
    is the only thing in the tree that EMITS `read_back_from_engine`, and the
    probe it returns beside the label is checked against `BACKEND_PROBES` by
    `attention_backend_reasons`.

    That is a bind on the SHAPE OF THE CLAIM and not on its origin, and the
    difference matters: an arm record is JSON, so a person can type both fields.
    What the pair rules out is the failure that actually happened -- a label
    corrected afterwards from a run log and recorded honestly as such, which is
    now refused by name. Passing the checks without running this function
    requires fabricating a read-back rather than transcribing a relabel, and the
    protocol answers fabrication with the fresh review, not with a checker.
    """

    tried: list[str] = []
    for probe in BACKEND_PROBES:
        try:
            value = _walk(llm, probe)
        except AttributeError:
            tried.append(probe)
            continue
        name = getattr(value, "__name__", None) or str(value)
        if name:
            return str(name), probe
        tried.append(probe)
    raise HarnessError(
        "backend: no probe resolved an attention backend off the built engine "
        f"(tried {len(tried)}: {', '.join(tried)}). REFUSING rather than labelling the "
        "run from the request kwarg: `VLLM_ATTENTION_BACKEND` does not exist in this "
        "wheel and selects nothing, and a label corrected afterwards from a log is "
        "unauditable (#1562). Add the probe this wheel needs to BACKEND_PROBES"
    )


def _assert_inproc_client(llm: Any) -> str:
    """The hook must be on the object that drafts, in THIS process."""

    try:
        client = _walk(llm, "llm_engine.engine_core")
    except AttributeError as error:
        raise HarnessError(
            f"cannot resolve the engine core client off the built LLM: {error}"
        ) from error
    name = type(client).__name__
    if "Inproc" not in name:
        raise HarnessError(
            f"engine core client is {name!r}, not an in-process client. vLLM V1 spawns "
            "`EngineCore` as a separate process by default and re-imports vllm clean, so "
            "the hook is on a class nothing uses and will record ZERO blocks over a run "
            "whose own counters report hundreds of drafted tokens. Set "
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 -- and note this assertion checks the "
            "RESOLVED CLASS, because the variable being set is not the same claim"
        )
    return name


def read_anchors(speculator: Any) -> list[int] | None:
    """The verified tokens this propose call drafts FROM, or None.

    The parity consumer pairs our blocks with the oracle's on the ANCHOR --
    `tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads
    `their_blocks[b]["anchor"]` and stops the walk on a mismatch -- because
    pairing by ordinal compares two blocks that started at different positions
    and reports the difference as a draft defect. The oracle's own
    `_generate_draft` reads it out of
    `input_buffers.input_ids[self._anchor_indices[:num_reqs]]`, so that is the
    walk taken here.

    Read BEFORE the delegate runs, because `propose` writes the draft into the
    same buffers. A miss returns None and is COUNTED rather than raised: a
    stopped 51.75 GiB run is the wrong answer to a moved attribute, and
    `hook_reasons` refuses a capture that missed EVERY anchor, which is the case
    that would ship an unpairable golden.

    UNVERIFIED against the beyond-pin wheel, like `BACKEND_PROBES` and for the
    same reason; the miss counter is what makes the failure visible.
    """

    try:
        indices = speculator._anchor_indices
        ids = speculator.input_buffers.input_ids
        return [int(ids[int(index)]) for index in indices.tolist()]
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return None


class DraftRecorder:
    """Wrap `DFlashSpeculator.propose` and record what it returned.

    Wrapping `propose` rather than `_generate_draft` is O23's third finding: with
    FULL decode graphs the replay never enters `_generate_draft` in Python, and
    both branches return `self.draft_tokens[:num_reqs]`.

    Blocks are recorded AGAINST THE OPEN RECORD, so the emitted shape is the one
    `tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads: `records[i].blocks`
    with `records[i].num_blocks`, each block carrying `call`, `req_row`, `anchor`
    and `drafts`. A flat top-level list was the previous shape, and it could not
    become the golden this capture exists to make re-derivable.
    """

    def __init__(self) -> None:
        self.blocks: list[dict[str, Any]] = []
        self.propose_calls = 0
        self.skipped_dummy = 0
        self.skipped_capture = 0
        self.anchor_misses = 0
        self.active = False
        self.record_index: int | None = None

    def open_record(self, index: int) -> None:
        """One generate call is in flight from here until `close_record`."""

        self.record_index = index
        self.active = True

    def close_record(self) -> None:
        self.active = False
        self.record_index = None

    def blocks_for(self, index: int) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in block.items() if key != "record"}
            for block in self.blocks
            if block["record"] == index
        ]

    def install(self, klass: type, torch_mod: Any) -> None:
        original = klass.propose

        def traced(speculator: Any, *args: Any, **kwargs: Any) -> Any:
            self.propose_calls += 1
            # `capture_model()` reaches this seam, and a `.tolist()` there is a
            # device-to-host copy inside a CUDA graph capture, which torch
            # refuses outright. Delegate, and COUNT the delegation. Checked on
            # ENTRY as well as on return, because the anchor read below happens
            # before the delegate and is a device-to-host copy too.
            capturing = bool(torch_mod.cuda.is_current_stream_capturing())
            anchors: list[int] | None = None
            if not capturing and self.active:
                anchors = read_anchors(speculator)
            result = original(speculator, *args, **kwargs)
            capturing = capturing or bool(torch_mod.cuda.is_current_stream_capturing())
            if capturing:
                self.skipped_capture += 1
                return result
            if not self.active:
                self.skipped_dummy += 1
                return result
            call = self.propose_calls
            drafts = result.tolist()
            for row, block in enumerate(drafts):
                entry: dict[str, Any] = {
                    "record": self.record_index,
                    "call": call,
                    "req_row": row,
                }
                if anchors is not None and row < len(anchors):
                    entry["anchor"] = int(anchors[row])
                else:
                    self.anchor_misses += 1
                entry["drafts"] = list(block)
                self.blocks.append(entry)
            return result

        traced.__name__ = "traced"
        klass.propose = traced

    def stats(self) -> dict[str, int]:
        return {
            "propose_calls": self.propose_calls,
            "skipped_dummy": self.skipped_dummy,
            "skipped_capture": self.skipped_capture,
            "anchor_misses": self.anchor_misses,
        }


#: The top-level keys `tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads
#: off a DFlash2 golden. Kept here as a CONTRACT: a capture that cannot produce
#: them cannot become the golden it exists to make re-derivable, and the
#: previous shape could not -- it wrote a FLAT `blocks` list beside
#: `oracle_runtime_version`, while the consumer reads `records[i].blocks` beside
#: `oracle_version`. `tests/tools/test_dflash2_speed_harness.py` parses this set
#: back out of the C++ file, so a new `golden.at(...)` upstream of this list
#: reddens rather than surfacing on a leased box.
GOLDEN_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "attention_backend",
    "draft_hook_installed",
    "hook_stats",
    "max_tokens",
    "metrics",
    "num_speculative_tokens",
    "oracle_version",
    "records",
    "spec",
)


def golden_envelope(
    args: argparse.Namespace, *, runtime_version: str, oracle_file: object
) -> dict[str, Any]:
    """The golden's own scalar top level, derived once so nothing diverges.

    `workload` carries `max_tokens` and `num_speculative_tokens` too, for the
    cross-arm comparison. They are repeated here because the parity consumer
    reads them off the top level, and both spellings are computed from the same
    argument in the same expression rather than transcribed.
    """

    return {
        "spec": "on",
        "pin": args.oracle_commit,
        "target": args.target,
        "draft": args.draft,
        "max_tokens": args.max_tokens,
        "max_num_seqs": args.max_num_seqs,
        "num_speculative_tokens": args.num_speculative_tokens,
        "enforce_eager": bool(args.enforce_eager),
        "oracle_version": runtime_version,
        "oracle_runtime_version": runtime_version,
        "oracle_expected_commit": args.oracle_commit,
        "oracle_file": oracle_file,
        "draft_hook_installed": True,
    }


def capture(args: argparse.Namespace, checked: Mapping[str, Any]) -> dict[str, Any]:
    """Build the oracle in its PRODUCTION configuration and record one run."""

    import importlib

    import torch
    from vllm import LLM, SamplingParams
    import vllm

    reasons = oracle_identity_reasons(vllm.__version__, args.oracle_commit)
    require_no_reasons(reasons, what="DFlash2 oracle capture")

    module = importlib.import_module(args.speculator_module)
    klass = getattr(module, args.speculator_class, None)
    if klass is None:
        raise HarnessError(
            f"{args.speculator_module}.{args.speculator_class} does not exist in this "
            "wheel; the beyond-pin oracle renamed the seam and the flags exist to "
            "follow it"
        )
    recorder = DraftRecorder()
    recorder.install(klass, torch)

    llm = LLM(
        model=args.target,
        speculative_config={
            "model": args.draft,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
        # NEVER --enforce-eager: AGENTS.md forbids it as a denominator, and with
        # FULL decode graphs off the speculator takes its non-graph branch too.
        enforce_eager=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        # TWO PERTURBATIONS OF THE DENOMINATOR, RECORDED RATHER THAN HIDDEN.
        # Both cost the ORACLE and neither costs us, so both flatter our ratio
        # and both are conservative -- which is a reason to record them, not a
        # reason to omit them.
        #
        # 1. `disable_log_stats=False` keeps vLLM's stat logger on so
        #    `llm.get_metrics()` can report the `spec_decode` counters the
        #    golden carries. A production serve leaves it on too, so this is
        #    the denominator's own configuration and not an instrument.
        # 2. `DraftRecorder.traced` stays installed for the WARM legs as well,
        #    because uninstalling it mid-run would change the object the timed
        #    repetitions call. It costs TWO
        #    `is_current_stream_capturing()` calls per propose on a warm leg:
        #    one on entry, and one after the delegate, because the `or` there
        #    does not short-circuit when the entry call returned False -- and
        #    False on entry is every timed leg. Plus the counter increments.
        #    Only the anchor read and the `.tolist()` are behind `self.active`
        #    and do not run on a warm leg.
        #
        # Neither is measured. `--repeat` with the hook uninstalled is the A/B
        # that would bound (2), and it needs the lease.
        disable_log_stats=False,
    )
    client_class = _assert_inproc_client(llm)
    resolved_backend, probe = resolve_attention_backend(llm)
    require_no_reasons(
        attention_backend_reasons(
            resolved=resolved_backend,
            declared=args.attention_backend,
            source=BACKEND_SOURCE_READ_BACK,
            # THREADED, not defaulted. `attention_backend_reasons` appends "no
            # read-back probe was recorded" whenever this is None, so omitting
            # it refused every run of this arm -- after `LLM(...)` had loaded
            # 51.75 GiB, on a lease. O23 exists to stop exactly that.
            probe=probe,
        ),
        what="DFlash2 oracle capture",
    )

    # ONE PROMPT PER `generate`, REPEATED. Three things follow from it and each
    # one is a defect this arm previously had:
    #
    # 1. THE ARM PRODUCES A NUMBER. Nothing here timed anything, and `metrics`
    #    held only vLLM's own `spec_decode` counters, so the summarizer had no
    #    key in common with our arm and rendered EVERY axis NOT MEASURED --
    #    including the denominator's throughput. `time.perf_counter()` around
    #    `generate` is the measurement, folded by the SHARED `fold_legs` so the
    #    two arms cannot fold two different statistics and call the quotient a
    #    ratio.
    # 2. THE BLOCKS BELONG TO A RECORD. `max_num_seqs` is 1 and one request is
    #    in flight per call, so every block recorded between `open_record` and
    #    `close_record` is that record's by construction rather than by an
    #    inference about batching.
    # 3. IT MATCHES OUR ARM. `dflash2_our_arm` already loops one prompt per
    #    `vllm-cli` invocation with `--repeat`, so this is the shape the ratio
    #    was always between; the oracle arm was the one doing something else.
    #
    # Only repetition 1 records blocks: greedy decode repeats the same tokens,
    # and recording them N times would inflate the golden without adding a fact.
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=None)
    records: list[dict[str, Any]] = []
    legs: list[dict[str, Any]] = []
    for index, prompt in enumerate(checked["prompts"]):
        first: Any = None
        for repetition in range(1, int(args.repeat) + 1):
            if repetition == 1:
                recorder.open_record(index)
            started = time.perf_counter()
            outputs = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - started
            recorder.close_record()
            out = outputs[0]
            completion = len(list(out.outputs[0].token_ids))
            legs.append(
                {
                    "run": repetition,
                    "record": index,
                    "finish_reason": getattr(out.outputs[0], "finish_reason", "") or "",
                    "prompt_tokens": len(list(out.prompt_token_ids)),
                    "completion_tokens": completion,
                    "secs": float(elapsed),
                    "tok_s": (float(completion) / elapsed) if elapsed > 0.0 else 0.0,
                }
            )
            if repetition == 1:
                first = out
        blocks = recorder.blocks_for(index)
        records.append(
            {
                "prompt": first.prompt,
                "prompt_token_ids": list(first.prompt_token_ids),
                "output_token_ids": list(first.outputs[0].token_ids),
                "text": first.outputs[0].text,
                "num_blocks": len(blocks),
                "blocks": blocks,
            }
        )

    require_no_reasons(
        hook_reasons(recorder.stats(), len(recorder.blocks))
        + leg_reasons(legs, max_tokens=args.max_tokens),
        what="DFlash2 oracle capture",
    )
    folded = fold_legs(legs)

    metrics: dict[str, Any] = dict(folded["metrics"])
    for metric in llm.get_metrics():
        name = getattr(metric, "name", "")
        if "spec_decode" in name:
            metrics[name] = getattr(metric, "value", None)

    return {
        **golden_envelope(
            args,
            runtime_version=vllm.__version__,
            oracle_file=getattr(vllm, "__file__", None),
        ),
        "engine_core_client": client_class,
        "attention_backend": resolved_backend,
        "attention_backend_declared": args.attention_backend,
        "attention_backend_source": BACKEND_SOURCE_READ_BACK,
        "attention_backend_probe": probe,
        "speculator_seam": f"{args.speculator_module}.{args.speculator_class}.propose",
        "hook_stats": recorder.stats(),
        "draft_hook_installed": True,
        "records": records,
        "legs": folded["legs"],
        "warm_legs": folded["warm_legs"],
        "cold_legs_discarded": folded["cold_legs_discarded"],
        "cold_discard_cause": folded["cold_discard_cause"],
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="capture the SPEC-DFLASH2 oracle arm")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--oracle-commit", default="")
    parser.add_argument("--oracle-build-recipe", default="")
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed repetitions per prompt. Must match our arm's --repeat: both "
        "arms fold a median over their warm legs, and two medians over different "
        "populations do not divide into a ratio. Blocks are recorded on "
        "repetition 1 only",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="present only so the refusal is REACHABLE and testable; AGENTS.md "
        "forbids it as a denominator and the run stops if it is passed",
    )
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH=SHA256")
    parser.add_argument("--lease-id", default="")
    parser.add_argument("--our-revision", default="")
    parser.add_argument("--our-build-recipe", default="")
    parser.add_argument("--speculator-module", default=DEFAULT_SPECULATOR_MODULE)
    parser.add_argument("--speculator-class", default=DEFAULT_SPECULATOR_CLASS)
    parser.add_argument("--clock-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="repeatable; defaults to the four PROMPTS the W6 goldens were captured "
        "over. Both arms must be given the SAME set, and workload_fingerprint is what "
        "proves they were",
    )
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument(
        "--assume-compute-processes",
        type=json.loads,
        default=None,
        help="a JSON list standing in for the driver's compute-app query, for tests "
        "and for a host with no nvidia-smi; a REAL run leaves it unset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ)
    checked = precheck(args, env)
    if args.precheck_only:
        print(canonical_json({"precheck": "PASS", **checked}))
        return 0

    clock = _load_clock(args.clock_summary)
    require_no_reasons(clock_state_reasons(clock, label="vllm"), what="DFlash2 oracle capture")

    captured = capture(args, checked)
    record = {**checked, **captured, "clock": clock}
    if args.output is not None:
        write_json_atomic(args.output, record)
    print(canonical_json(record))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
