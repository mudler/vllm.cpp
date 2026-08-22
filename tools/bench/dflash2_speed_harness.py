#!/usr/bin/env python3
"""The SPEC-DFLASH2 capture and speed harness: preconditions, refusals, result.

Why this file exists
--------------------
`.agents/specs/dflash2-spec-decode.md` `## Owed` O22 and O23 described a
repaired oracle instrument in PROSE and committed none of it
([#1562](https://github.com/mudler/vllm.cpp/issues/1562)). The two committed
DFlash2 goldens therefore carry labels nobody can re-derive: the `FLASH_ATTN`
golden's `attention_backend` is a POST-HOC RELABEL by a `w6-relabel.py` that is
not in the tree, corrected from a run log that is not in the tree either. The
next agent to instrument this oracle rewrites the harness from a paragraph,
which is the exact cost O23 was written to prevent, and #1562's own title says
so.

This module is the committed replacement. It holds every precondition a DFlash2
speed number must satisfy and the refusal that fires when one is absent. It is
standard library plus two in-tree helpers, so **every assertion here runs in CPU
CI with no GPU and no `nvidia-smi`** -- the same polarity
`tools/bench/gpu_clock_state.py` chose, for the same reason: sampling needs the
hardware, the logic does not, and a refusal path that only runs on a leased box
is a refusal path nobody has ever seen fire.

What it refuses, and why each one
---------------------------------
Every rule below is a measurement this repository has already lost.

- **Oracle identity.** A wheel's `__version__` carries the `+g<sha>` local
  version segment, and that is the only thing that distinguishes the built
  oracle from a plain release that runs, is deterministic, and answers a
  different question (#520). This row's oracle is BEYOND-PIN, so the expected
  commit is an ARGUMENT rather than the parity pin, and the harness refuses an
  absent one instead of falling back to the pin -- a fallback would silently
  measure against a wheel that cannot even load the architecture.
- **The resolved attention backend, READ BACK off the built engine.** #1456
  measured vLLM's vendored flash-attention emitting `sm_80`/`sm_75` at
  `CUDA_ARCHS=12.0`, and W6 then measured the two backends DISAGREEING with each
  other on this model: 0.597 against 0.657 acceptance, and a divergent
  continuation on 1 of 4 prompts. The denominator's backend therefore changes
  the answer, so it is read back and never assumed. `VLLM_ATTENTION_BACKEND`
  does not exist in this wheel and selects nothing, so a label sourced from it
  is meaningless, and a label CORRECTED from a log afterwards is unauditable.
  Only `read_back_from_engine` is admitted.
- **The SSE keepalive, asserted rather than inherited.** `VT_SERVER_SSE_PING_S`
  defaults to 0 since #931, but #577 measured a non-zero ping injecting comment
  frames into exactly the slowest requests, which drops the latency tail and
  flatters us. A default is not an assertion: the harness requires the variable
  to be PRESENT and `0`, so a value inherited from a shell nobody recorded
  cannot pass.
- **The denominator's configuration.** `--enforce-eager` is refused outright,
  and the two arms' workloads must be equal key by key. A ratio between two
  different workloads means nothing.
- **Checkpoint identity by sha256.** A repo id is not a pin, because
  checkpoints get re-quantized in place under an unchanged name.
- **Clock state.** Delegated whole to `tools/bench/gpu_clock_state.py`, which is
  the one helper that samples, folds and asserts it, and which
  `.agents/benchmarking.md` requires any new harness to import rather than
  reimplement. Inside an `rc` lease the clock can be SAMPLED and not pinned
  (`LGC_RC=4`, #1354), so `clock_pinned` is RECORDED and never required. Each
  ARM owns its window (`dflash2_oracle_capture.ClockWindow`), because the
  summary is written when the sampler stops and a summary handed to an arm
  before it ran could not describe it (#1657); this module still judges the
  pair, and a record whose window is unusable never becomes a ratio.
- **Contention.** The lease id, and the GPU's own compute-process list. Two
  mutexes that do not exclude each other already cost `minimax-music3` a whole
  speed axis (#777 again, 2026-08-17).
- **Both arms' build recipe and revision, EACH ARM'S OWN.** An axis measured
  against a build nobody can reconstruct is not reproducible -- and the capture
  runs from our checkout, so the revision nearest to hand is the wrong one for
  the denominator. `oracle_build_reasons` holds the vLLM arm's revision to the
  oracle head the same record declares.
- **The weights an arm LOADED, not merely the ones it hashed.**
  `checkpoint_reasons` measures a digest against bytes and never asks whether
  the file was opened; `model_binding_reasons` binds every model path an arm
  names to an artifact entry that identifies it.
- **Our arm's draft depth, taken from the CONFIG the binary read.**
  `--speculative-config` is what `vllm-cli` parses, and an absent one runs a
  plain decode that would still fingerprint-match a drafting oracle.
- **Both arms repeating the same number of times.** Each folds a MEDIAN over its
  warm legs through the one shared `fold_legs`, so two arms that repeated
  differently folded different populations.
- **The instrument SAW something.** O23's three failures each presented as a
  verdict about the CODE while the instrument was simply blind, and only the
  ABORT ON ZERO caught all three. `hook_reasons` is that abort, plus the
  one-sided bound the W6 repair wave added (`sum(blocks) <= propose_calls -
  skipped`, because a lost record is possible and an invented one is not).

Reasons, not exceptions
-----------------------
Every checker returns a LIST of reasons and `require_no_reasons` raises once
with all of them. A harness that dies on the first defect makes the operator pay
a 51.75 GiB model load per defect -- that is the TARGET CHECKPOINT read off
CIFS, `weight_utils.py:858`, and the finished load holds 54.87 GiB resident per
`model_runner.py:385`; the two numbers are the same load measured differently
-- which is the cost O23 exists to stop. The
functions are pure, so the test suite drives them directly and the drivers call
exactly the same code the gate proves.

    python3 -m tools.bench.dflash2_speed_harness summarize \
        --ours   evidence/ours.json \
        --vllm   evidence/vllm.json \
        --output evidence/dflash2-speed.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from tools.bench.gpu_clock_state import clock_reasons, compare_clock_records
from tools.bench.serve_low_common import (
    HarnessError,
    canonical_json,
    sha256_file,
    write_json_atomic,
)

# The `+g<sha>` local version segment, the only part of a wheel's version that
# names the source it was built from.
_VERSION_COMMIT_RE = re.compile(r"\+g([0-9a-f]{7,40})")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: The keepalive knob, and the ONE value a measurement may be taken at (#577).
SSE_PING_ENV = "VT_SERVER_SSE_PING_S"
REQUIRED_SSE_PING = "0"

#: The only admissible provenance for an `attention_backend` label. A relabel
#: from a log, an echo of a request kwarg, and a read of an environment variable
#: this wheel does not consult are all REFUSED by name, so the #1562 label
#: cannot be reproduced by a harness that imports this module.
BACKEND_SOURCE_READ_BACK = "read_back_from_engine"

#: The dotted attribute walks `resolve_attention_backend` is allowed to have
#: read the label off, in order. They live HERE rather than beside the reader so
#: that the summarizer can check the recorded probe against the same list: a
#: record that names a source must also name a walk this module recognises.
#: They are CANDIDATES and not a contract -- vLLM moves this object graph
#: between releases. A miss on every probe is a LOUD REFUSAL rather than a
#: fallback, so the worst outcome of a stale list is a run that stops and names
#: what it could not read.
#:
#: **The first two are MEASURED and the last three are not.** On `dgx:gpu0` on
#: 2026-08-22, inside the beyond-pin wheel and after a load `model_runner.py:385`
#: reported as `Model loading took 54.87 GiB memory and 702.391374 seconds`, all
#: three
#: of the `attn_backend` walks raised `AttributeError: 'GPUModelRunner' object
#: has no attribute 'attn_backend'` and the two `attention_config` walks each
#: returned `TRITON_ATTN`
#: ([#1658](https://github.com/mudler/vllm.cpp/issues/1658)). The retired walks
#: are KEPT rather than deleted: this list is ordered, an older wheel still
#: answers on them, and a walk that resolves nothing costs one `AttributeError`.
BACKEND_PROBES: tuple[str, ...] = (
    "llm_engine.vllm_config.attention_config.backend",
    "llm_engine.engine_core.engine_core.vllm_config.attention_config.backend",
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.attn_backend",
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.attn_backend",
    "llm_engine.model_executor.driver_worker.model_runner.attn_backend",
)

#: Where the PER-LAYER truth lives, and why one scalar is not enough.
#:
#: The same 2026-08-22 run walked `...model_runner.attn_groups` and resolved
#: THREE backends at once on this model: `GDNAttentionBackend` over 48
#: `linear_attn` layers in 10 groups, `TritonAttentionBackend` over 16
#: `self_attn.attn` layers in 4 groups, and `FlashAttentionBackend` over the
#: DFlash2 draft's five sliding-window layers (`model.layers.64-68`) in one
#: group. The counts come from the run's own `c-probe-result.json`; the 30
#: this comment carried until 2026-08-22 was the test stand-in's shape
#: (#1666). A single `attention_backend`
#: string cannot describe that, and the denominator's backend is exactly what W6
#: measured vLLM disagreeing with ITSELF over -- 0.597 against 0.657 acceptance,
#: and a divergent continuation on 1 of 4 prompts.
#:
#: So BOTH are recorded and only the SCALAR is gated. The scalar is what the run
#: DECLARED and what `attention_backend_reasons` compares against, and it is the
#: key `tests/parity/test_qwen38_dflash2_spec_decode.cpp` reads off a golden.
#: The map is what actually RAN. It is not gated because the two spellings are
#: not comparable -- the scalar is an enum name (`TRITON_ATTN`) and the map
#: holds class names (`TritonAttentionBackend`) -- and a checker that equated
#: them would refuse every correct run. A MISS is recorded as a named `miss`
#: rather than an empty map, because an empty map reads as "one backend", which
#: is the false claim this field exists to prevent.
BACKEND_GROUP_PROBES: tuple[str, ...] = (
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.attn_groups",
    "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.attn_groups",
    "llm_engine.model_executor.driver_worker.model_runner.attn_groups",
)

#: How the engine is ASKED for a backend, in the order the arm tries them.
#:
#: `capture()` built `LLM(...)` with no backend kwarg at all while
#: `attention_backend_reasons` requires `resolved == declared`, so under a
#: declared `TRITON_ATTN` the arm logged `Using FLASH_ATTN attention backend`
#: and the declared denominator was unreachable by ANY path
#: ([#1659](https://github.com/mudler/vllm.cpp/issues/1659)). That matters here
#: rather than merely being untidy: #1456 measured vLLM's vendored
#: flash-attention emitting `sm_80`/`sm_75` at `CUDA_ARCHS=12.0`, which is why
#: `TRITON_ATTN` is the declared oracle backend on this box.
#:
#: UNVERIFIED at the beyond-pin head, on the same footing as `BACKEND_PROBES`
#: and for the same reason. A wheel REJECTS a kwarg it does not declare with a
#: `TypeError`, raised while `EngineArgs` is built and therefore before anything
#: loads, so trying both costs no lease time; a spelling that is accepted and
#: IGNORED is caught by the read-back, which is the refusal that already exists.
#: `--attention-backend-kwarg` pins the working one once it is known, without a
#: code change.
ATTENTION_BACKEND_KWARGS: tuple[str, ...] = ("attention_config", "attention_backend")


def attention_backend_kwargs(spelling: str, backend: str) -> dict[str, Any]:
    """The one kwarg pair a given spelling means."""

    if spelling not in ATTENTION_BACKEND_KWARGS:
        raise HarnessError(
            f"unknown attention-backend kwarg spelling {spelling!r}; "
            f"ATTENTION_BACKEND_KWARGS admits {list(ATTENTION_BACKEND_KWARGS)}"
        )
    if spelling == "attention_config":
        return {"attention_config": {"backend": backend}}
    return {spelling: backend}

#: Workload keys that must be IDENTICAL on both arms. `.agents/benchmarking.md`:
#: "If the two sides differ in any of those, the ratio means nothing."
#:
#: `repeat` is here because both arms fold a MEDIAN over their warm legs: two
#: arms that repeated a different number of times folded different populations,
#: and the median of four legs and the median of one are not the same statistic.
WORKLOAD_KEYS: tuple[str, ...] = (
    "prompts_sha256",
    "num_prompts",
    "max_tokens",
    "num_speculative_tokens",
    "max_num_seqs",
    "repeat",
    "concurrency",
    "temperature",
    "seed",
)

#: Every axis a DFlash2 speed result must carry, and its polarity.
#: `.agents/benchmarking.md`: record every required axis -- throughput, latency,
#: memory -- as both values and ratios.
AXIS_POLARITY: dict[str, str] = {
    "output_throughput_tok_s": "higher_is_better",
    "tpot_ms": "lower_is_better",
    "ttft_ms": "lower_is_better",
    "peak_device_bytes": "lower_is_better",
}


def workload_fingerprint(prompts: Sequence[str]) -> str:
    """The ONE way both arms hash their prompt set.

    Shared so the two arms cannot each hash a slightly different thing and then
    compare equal by accident, and so a reordering of the prompts is a
    DIFFERENCE rather than a coincidence: the digest runs over the ordered list
    with a separator no prompt can contain.
    """

    if not prompts:
        raise HarnessError("a workload with no prompts has nothing to fingerprint")
    joined = "\x00".join(str(prompt) for prompt in prompts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def require_no_reasons(reasons: Sequence[str], *, what: str) -> None:
    """Raise once, naming EVERY reason *what* cannot be measured.

    One raise per defect would charge the operator one 51.75 GiB model load per
    defect. The refusal is a list because the repair is a list.
    """

    if reasons:
        listed = "\n".join(f"  - {reason}" for reason in reasons)
        raise HarnessError(f"{what} REFUSED, {len(reasons)} precondition(s) unmet:\n{listed}")


def oracle_identity_reasons(runtime_version: object, expected_commit: object) -> list[str]:
    """Reasons the wheel in this venv is not the oracle the row declares.

    The expected commit is an argument and NOT the parity pin: this row's oracle
    is beyond-pin by developer decision, and `.agents/upstream-sync.md` is not
    advanced for it. A harness that fell back to the pin would measure against a
    wheel that cannot load `DFlash2DraftModel` at all.
    """

    reasons: list[str] = []
    expected = "" if expected_commit is None else str(expected_commit).strip()
    if not expected:
        reasons.append(
            "oracle: no expected commit was given; the beyond-pin oracle head is an "
            "ARGUMENT for this row and there is no pin to fall back to"
        )
    elif not _HEX40_RE.match(expected) and len(expected) < 7:
        reasons.append(
            f"oracle: expected commit {expected!r} is too short to identify a build; "
            "give at least 7 hex characters"
        )
    text = "" if runtime_version is None else str(runtime_version).strip()
    if not text:
        reasons.append(
            "oracle: the wheel reported no runtime version; a venv that cannot say "
            "what it is cannot be a denominator"
        )
        return reasons
    match = _VERSION_COMMIT_RE.search(text)
    if match is None:
        reasons.append(
            f"oracle: runtime version {text!r} carries no `+g<sha>` segment, so it is a "
            "PLAIN RELEASE. A release wheel runs, is deterministic and answers a "
            "different question -- that is the #520 failure exactly"
        )
        return reasons
    if expected and not (
        expected.startswith(match.group(1)) or match.group(1).startswith(expected)
    ):
        reasons.append(
            f"oracle: runtime version {text!r} names commit {match.group(1)!r}, not the "
            f"declared oracle head {expected!r}"
        )
    return reasons


def attention_backend_reasons(
    *, resolved: object, declared: object, source: object, probe: object = None
) -> list[str]:
    """Reasons an `attention_backend` label may not be recorded beside a number.

    The rule O22 lays down and the #1562 golden does not meet: pass the kwarg,
    then read the RESOLVED value back off the built engine. A label that came
    from anywhere else is refused BY ITS SOURCE, so the failure is named rather
    than inferred from a value that happens to look plausible.

    **What this does and does not bind.** An arm record is a JSON file, and a
    JSON file can be authored by hand, so no checker here can tell a read-back
    from a well-typed imitation of one. What the source and probe fields DO bind
    is the shape of the claim: a record must now assert both a provenance only
    `resolve_attention_backend` emits and a walk this module recognises, and
    #1562's actual defect -- a value CORRECTED AFTERWARDS from a run log, and
    recorded honestly as such -- can no longer be written down without being
    refused. Transcribing a relabel is caught; fabricating a read-back is a
    different act, and the protocol answers that one with the fresh review
    rather than with a checker.
    """

    reasons: list[str] = []
    label = "" if resolved is None else str(resolved).strip()
    want = "" if declared is None else str(declared).strip()
    origin = "" if source is None else str(source).strip()
    walk = "" if probe is None else str(probe).strip()
    if not walk:
        reasons.append(
            "backend: no read-back probe was recorded, so nothing says WHERE on the "
            "built engine the label was read. `attention_backend_probe` was captured "
            "and never checked until now, which made the provenance field a bare "
            "string beside an unchecked one"
        )
    elif walk not in BACKEND_PROBES:
        reasons.append(
            f"backend: attention_backend_probe {walk!r} is not one of the "
            f"{len(BACKEND_PROBES)} walks BACKEND_PROBES admits. Either the wheel "
            "moved the object graph and the list needs the entry this run used, or "
            "the label did not come off the engine at all"
        )
    if not label:
        reasons.append(
            "backend: no resolved attention backend was read off the built engine; "
            "W6 measured the two backends disagreeing on this model (acceptance "
            "0.597 against 0.657, and a divergent continuation on 1 of 4 prompts), "
            "so an unrecorded backend voids the ratio"
        )
    if origin != BACKEND_SOURCE_READ_BACK:
        reasons.append(
            f"backend: attention_backend_source is {origin!r}, not "
            f"{BACKEND_SOURCE_READ_BACK!r}. A post-hoc relabel from a run log is the "
            "#1562 defect: the log and the relabel script are not in the tree, so the "
            "label cannot be re-derived by anyone"
        )
    if not want:
        reasons.append(
            "backend: the run declared no backend, so nothing says which denominator "
            "was intended and the read-back has nothing to agree with"
        )
    elif label and label != want:
        reasons.append(
            f"backend: the engine resolved {label!r} while the run declared {want!r}; "
            "a substituted denominator is the failure this protocol exists to stop"
        )
    return reasons


def sse_keepalive_reasons(env: Mapping[str, str]) -> list[str]:
    """Reasons the SSE keepalive state is not admissible.

    Asserted rather than inherited. #931 made 0 the default, and #577 measured
    what a non-zero value does: comment frames land in exactly the slowest
    requests, the latency tail drops, and the number flatters us. An ABSENT
    variable is refused too, because "it defaults to 0" is a claim about a
    version of the binary rather than an observation of this run.
    """

    if SSE_PING_ENV not in env:
        return [
            f"sse: {SSE_PING_ENV} is not set in the measured environment. The default "
            "is 0 (#931), but a default is not a record: set it explicitly so the "
            "value travels with the measurement"
        ]
    value = str(env[SSE_PING_ENV]).strip()
    if value != REQUIRED_SSE_PING:
        return [
            f"sse: {SSE_PING_ENV}={value!r}, not {REQUIRED_SSE_PING!r}. A non-zero ping "
            "injects comment frames into the slowest requests and drops the latency "
            "tail in OUR favour (#577)"
        ]
    return []


def denominator_reasons(config: Mapping[str, Any]) -> list[str]:
    """Reasons the vLLM arm is not vLLM's PRODUCTION configuration.

    `AGENTS.md` §Gates: never use `--enforce-eager` as the denominator. The
    absent key is refused as well as the true one -- a run that never recorded
    whether it was eager cannot be shown not to have been.
    """

    reasons: list[str] = []
    if "enforce_eager" not in config:
        reasons.append(
            "denominator: the vLLM arm did not record `enforce_eager`, so nothing "
            "shows it ran the production graph path"
        )
    elif bool(config["enforce_eager"]):
        reasons.append(
            "denominator: the vLLM arm ran with `enforce_eager=True`. AGENTS.md "
            "forbids it as a denominator, and with FULL decode graphs off, the "
            "DFlash2 speculator's `propose` takes its non-graph branch as well, so "
            "the two arms are not even running the same code"
        )
    return reasons


def workload_reasons(ours: Mapping[str, Any], theirs: Mapping[str, Any]) -> list[str]:
    """Reasons the two arms did not run the same workload."""

    reasons: list[str] = []
    for key in WORKLOAD_KEYS:
        if key not in ours:
            reasons.append(f"workload: our arm did not record {key}")
        if key not in theirs:
            reasons.append(f"workload: the vLLM arm did not record {key}")
        if key in ours and key in theirs and ours[key] != theirs[key]:
            reasons.append(
                f"workload: {key} differs across the arms ({ours[key]!r} against "
                f"{theirs[key]!r}); the ratio is between two different measurements"
            )
    return reasons


def checkpoint_reasons(artifacts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reasons the weights behind a number are not identified.

    A repo id is not a pin, because a checkpoint gets re-quantized in place
    under an unchanged name. The sha256 is the identity, and it is MEASURED
    against the file when the file is present -- a recorded hash that was never
    compared with anything is a string.
    """

    reasons: list[str] = []
    if not artifacts:
        return [
            "checkpoint: the run named no artifacts, so nothing says which weights "
            "produced the number"
        ]
    for index, entry in enumerate(artifacts):
        role = str(entry.get("role") or f"artifact[{index}]")
        recorded = str(entry.get("sha256") or "").strip().lower()
        if not _SHA256_RE.match(recorded):
            reasons.append(
                f"checkpoint: {role} records sha256 {recorded!r}, which is not a "
                "64-character hex digest"
            )
            continue
        raw_path = entry.get("path")
        if not raw_path:
            reasons.append(f"checkpoint: {role} records a hash but no path to check it against")
            continue
        path = pathlib.Path(str(raw_path))
        if not path.is_file():
            reasons.append(
                f"checkpoint: {role} names {path}, which is not a readable file on this "
                "host, so its recorded hash was compared with nothing"
            )
            continue
        measured = sha256_file(path)
        if measured != recorded:
            reasons.append(
                f"checkpoint: {role} at {path} hashes {measured}, not the recorded "
                f"{recorded}; the artifact was re-quantized or replaced in place"
            )
    return reasons


def contention_reasons(state: Mapping[str, Any]) -> list[str]:
    """Reasons the box was not ours alone when the arm's preconditions were taken.

    Fleet devices are reachable by lease only, so the lease id is the record of
    who held the box. On 2026-08-17 one session took the file mutex over `ssh`
    while another held the same device through `rc`; neither mutex excluded the
    other and `minimax-music3` §13.10 still carries a whole speed axis as VOID.

    **The compute-process list is a POINT SAMPLE taken during precheck, before
    the model loads -- it is not a watch over the timed window.** A job that
    arrived after the sample is invisible to it. The lease is what excludes that
    job; this list is the check that the lease was honoured at the moment the
    arm started, and it is worded that way here because "no foreign process was
    seen at t0" and "the box was ours throughout" are different claims.
    """

    reasons: list[str] = []
    lease = str(state.get("lease_id") or "").strip()
    if not lease:
        reasons.append(
            "contention: no lease id was recorded. A fleet device is reachable by "
            "`rc run`/`rc hold` only, and a run that cannot name its lease cannot "
            "show the box was not shared"
        )
    if "compute_processes" not in state:
        reasons.append(
            "contention: the GPU's compute-process list was not sampled, so nothing "
            "shows the arm started on a device it had to itself"
        )
    else:
        foreign = [
            proc
            for proc in state["compute_processes"]
            if int(proc.get("pid", -1)) not in set(state.get("own_pids") or ())
        ]
        if foreign:
            named = ", ".join(
                f"pid {proc.get('pid')} ({proc.get('name', '?')})" for proc in foreign
            )
            reasons.append(
                f"contention: {len(foreign)} foreign compute process(es) held the GPU "
                f"when this arm sampled the driver, before its model loaded: {named}"
            )
    return reasons


def build_recipe_reasons(arm: Mapping[str, Any], *, label: str) -> list[str]:
    """Reasons an arm's binary cannot be rebuilt from what it recorded."""

    reasons: list[str] = []
    for field in ("revision", "build_recipe"):
        value = str(arm.get(field) or "").strip()
        if not value:
            reasons.append(
                f"build: the {label} arm recorded no {field}; an axis measured against "
                "a build nobody can reconstruct is not reproducible"
            )
    revision = str(arm.get("revision") or "").strip()
    if revision and not _HEX40_RE.match(revision) and len(revision) < 7:
        reasons.append(
            f"build: the {label} arm's revision {revision!r} is too short to identify a "
            "tree"
        )
    return reasons


def oracle_build_reasons(arm: Mapping[str, Any]) -> list[str]:
    """Reasons the vLLM arm's recorded build is not the ORACLE's build.

    `build_recipe_reasons` asks whether an arm named a revision and a recipe. It
    cannot ask whether the arm named ITS OWN. The capture runs from this
    repository's checkout, so the two revisions in scope -- our tree's HEAD and
    the beyond-pin oracle head -- are both to hand, and the denominator's block
    was filled from the wrong one: a `cmake --preset` line was attributed to a
    pip-installed wheel, and every reader of the result would have taken the
    ratio as measured against a build that does not exist.

    A recipe string cannot be classified, and this checker does not try. The
    REVISION can: the vLLM arm's revision must agree with the oracle commit the
    same record declares, and a disagreement names both.
    """

    reasons = build_recipe_reasons(arm.get("build") or {}, label="vllm")
    declared = str(arm.get("oracle_expected_commit") or "").strip().lower()
    recorded = str((arm.get("build") or {}).get("revision") or "").strip().lower()
    if declared and recorded and not (
        declared.startswith(recorded) or recorded.startswith(declared)
    ):
        reasons.append(
            f"build: the vLLM arm recorded revision {recorded!r} while the same record "
            f"declares the oracle head {declared!r}. The capture runs from OUR checkout, "
            "so the revision that is easiest to record is the wrong one, and a build "
            "block that describes the harness instead of the denominator makes the "
            "ratio unreproducible in the one direction nobody rechecks"
        )
    return reasons


def speculative_config_reasons(
    raw: object, declared_k: object, *, label: str
) -> tuple[list[str], dict[str, Any]]:
    """Reasons our arm's recorded k is not the k the binary was CONFIGURED with.

    `vllm-cli` takes its drafter from `--speculative-config`, a JSON document,
    and this harness took a separate `--num-speculative-tokens` integer into the
    workload block. Nothing reconciled them, so the workload fingerprint could
    read k=7 over a run the binary executed at k=3 -- and, because the config
    was OPTIONAL, over a run with no speculative decoding at all, which would
    have compared a plain decode against a drafted one and called the difference
    a parity result.

    So the config is the SOURCE OF TRUTH and the flag is checked against it.
    Returns the reasons and the parsed config, because the caller records what
    was actually passed rather than what was asked for.
    """

    text = "" if raw is None else str(raw).strip()
    if not text:
        return (
            [
                f"speculative: the {label} arm was given no --speculative-config, so the "
                "binary ran with speculative decoding OFF. The oracle arm drafts; a "
                "ratio between a drafted decode and a plain one measures the feature, "
                "not the implementation, and this row exists to measure the "
                "implementation"
            ],
            {},
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return (
            [f"speculative: the {label} arm's --speculative-config is not JSON: {error}"],
            {},
        )
    if not isinstance(parsed, Mapping):
        return (
            [
                f"speculative: the {label} arm's --speculative-config is a "
                f"{type(parsed).__name__}, not a JSON object"
            ],
            {},
        )
    reasons: list[str] = []
    config = dict(parsed)
    if "num_speculative_tokens" not in config:
        reasons.append(
            f"speculative: the {label} arm's --speculative-config names no "
            "`num_speculative_tokens`, so the draft depth the binary ran at is not "
            "recorded anywhere and cannot be compared with the oracle's k"
        )
        return reasons, config
    try:
        configured = int(config["num_speculative_tokens"])
    except (TypeError, ValueError):
        reasons.append(
            f"speculative: the {label} arm's `num_speculative_tokens` is "
            f"{config['num_speculative_tokens']!r}, which is not an integer"
        )
        return reasons, config
    if declared_k is not None and int(declared_k) != configured:
        reasons.append(
            f"speculative: the {label} arm recorded k={int(declared_k)} in its workload "
            f"while --speculative-config configured the binary at k={configured}. The "
            "config is what the binary read; the flag is what the record claimed"
        )
    return reasons, config


def model_binding_reasons(
    paths: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]], *, label: str
) -> list[str]:
    """Reasons the weights an arm LOADED are not the weights it identified.

    `checkpoint_reasons` measures every named artifact against its bytes. It
    never asks whether any of them is the model the arm opened, so an arm could
    hash a file it never read and pass. The driver made that reachable by
    handing the SAME artifact list to both arms, while one loads an HF directory
    and the other a GGUF -- so at most one of them was ever describing itself.

    An artifact binds a model path when it IS that path, or when it lies inside
    it and the path is a directory. That covers the two shapes in use: a
    single-file GGUF, and a checkpoint directory whose `model.safetensors` is
    the hashed artifact.
    """

    reasons: list[str] = []
    recorded = [str(entry.get("path") or "") for entry in artifacts]
    for role, raw in paths.items():
        model = str(raw or "").strip()
        if not model:
            reasons.append(f"artifact: the {label} arm named no {role} path to bind")
            continue
        want = pathlib.PurePosixPath(model)
        bound = [
            candidate
            for candidate in recorded
            if candidate
            and (
                candidate == model
                or want in pathlib.PurePosixPath(candidate).parents
            )
        ]
        if not bound:
            reasons.append(
                f"artifact: the {label} arm loads {role} {model!r} and no --artifact "
                f"entry names it or a file inside it (recorded: {recorded or 'none'}). "
                "A measured hash beside a path the run never opened identifies "
                "nothing"
            )
    return reasons


def hook_reasons(stats: Mapping[str, Any], recorded_blocks: int) -> list[str]:
    """Reasons the capture instrument did not demonstrably SEE the drafter.

    O23's three failures each presented as a verdict about the CODE -- "the
    draft is empty", "the oracle cannot capture graphs with a DFlash2
    speculator", "the drafter proposed nothing" -- on runs where vLLM's own
    counters said it had drafted hundreds of tokens. Both preconditions the
    third run checked were TRUE and the instrument was still blind. Only the
    ABORT ON ZERO caught all three, so it is the first rule here.

    The bound is ONE-SIDED on purpose (`sum(blocks) <= propose_calls -
    skipped`): a record lost between the hook and the sink is possible, and an
    invented one is not.
    """

    reasons: list[str] = []
    for field in ("propose_calls", "skipped_dummy", "skipped_capture"):
        if field not in stats:
            reasons.append(f"hook: hook_stats omits {field}")
    if reasons:
        return reasons
    calls = int(stats["propose_calls"])
    skipped = int(stats["skipped_dummy"]) + int(stats["skipped_capture"])
    if recorded_blocks <= 0:
        reasons.append(
            f"hook: the instrument recorded ZERO draft blocks over {calls} propose "
            "call(s). Every O23 failure looked like a verdict about the code and was "
            "a blind instrument; a passing precondition is not evidence the hook SAW "
            "anything"
        )
    if calls <= 0:
        reasons.append(
            "hook: `propose` was never called, so the hook was not on the object that "
            "drafts. vLLM V1 spawns `EngineCore` as a SEPARATE PROCESS by default and "
            "re-imports vllm clean, which puts an in-process monkeypatch on a class "
            "nothing uses; assert the client class rather than trusting "
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        )
    if recorded_blocks > calls - skipped:
        reasons.append(
            f"hook: {recorded_blocks} recorded block(s) exceed the {calls - skipped} "
            f"call(s) that could have produced one ({calls} propose calls, {skipped} "
            "skipped). A lost record is possible; an invented one is not"
        )
    # OPTIONAL, because the two committed W6 goldens predate the field and their
    # `hook_stats` are quoted verbatim by the parity suite. A capture that emits
    # it and misses EVERY anchor is refused: the golden consumer falls back to
    # ordinal pairing when `anchor` is absent, and its own comment says ordinal
    # pairing compares blocks that started at different positions and reports
    # the difference as a draft defect.
    if "anchor_misses" in stats and recorded_blocks > 0:
        misses = int(stats["anchor_misses"])
        if misses >= recorded_blocks:
            reasons.append(
                f"hook: all {recorded_blocks} recorded block(s) missed the anchor read "
                f"({misses} miss(es)). Without an anchor the parity consumer pairs "
                "blocks by ORDINAL, which compares two blocks that started at "
                "different positions and reports the difference as a draft defect. "
                "Fix the attribute walk rather than shipping the golden"
            )
    return reasons


def leg_reasons(legs: Sequence[Mapping[str, Any]], *, max_tokens: int) -> list[str]:
    """Reasons a set of timed legs does not support a throughput number.

    SHARED BY BOTH ARMS on purpose. Our arm reads its legs out of `vllm-cli`'s
    stderr and the oracle arm times `llm.generate` itself, but a ratio between
    two medians folded by two different rules is not a ratio, so the rule lives
    once and each arm only has to produce legs in the same shape.
    """

    reasons: list[str] = []
    if not legs:
        return [
            "legs: no timing line this harness could parse. A null parse proves the "
            "terms wrong, never that the run was fast; check `examples/cli/main.cpp`'s "
            "format string against LEG_RE"
        ]
    if len(legs) < 2:
        reasons.append(
            f"legs: {len(legs)} leg(s) ran, so there is no warm leg after the cold one "
            "is discarded. A single leg is an anecdote"
        )
    for leg in legs:
        if int(leg["completion_tokens"]) != max_tokens:
            reasons.append(
                f"legs: run {leg['run']} produced {leg['completion_tokens']} completion "
                f"tokens against the {max_tokens} the workload asked for, so the arms "
                "did not do the same amount of work"
            )
        if float(leg["secs"]) <= 0.0:
            reasons.append(f"legs: run {leg['run']} recorded a non-positive wall time")
    return reasons


def repeat_reasons(repeat: object, *, label: str) -> list[str]:
    """Reasons a repetition count cannot produce a warm median.

    Checked in BOTH prechecks, so the failure that costs a lease is found before
    the lease. Run 1 is discarded on every arm, so `--repeat 1` leaves nothing to
    fold -- and on the oracle arm it also leaves the record loop with no
    repetition that captured blocks at all.
    """

    try:
        value = int(repeat)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return [f"repeat: the {label} arm's --repeat is {repeat!r}, which is not an integer"]
    if value < 2:
        return [
            f"repeat: the {label} arm would run --repeat {value}. Run 1 carries the "
            "first graph capture and is discarded on both arms, so this leaves no "
            "warm leg; a single leg is an anecdote"
        ]
    return []


def is_warm_leg(leg: Mapping[str, Any]) -> bool:
    """Is this leg one the median is folded from?

    ONE PREDICATE, TWO CONSUMERS. `fold_legs` discards run 1 for a named cause,
    and `dflash2_our_arm.warm_leg_spans` restricts the clock window to the legs
    the median is folded from (#1671). Inlining `run > 1` in both would let the
    number and the clock that qualifies it come to describe different legs,
    which is the defect #1671 filed at three orders of magnitude.
    """

    return int(leg["run"]) > 1


def fold_legs(legs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median of the WARM legs, with the cold discard recorded, not applied silently.

    **The named cause is not the same on both arms, so it is named for what run
    1 actually IS on each.** Our arm launches one `vllm-cli` process per prompt,
    so its run 1 loads the model, captures the first graphs and takes the first
    KV allocation. The oracle arm builds ONE `LLM` for ALL the prompts, and it
    builds it BEFORE the prompt loop, so none of those three is inside any leg
    of that arm -- prompt 1's run 1 included. What sets its run 1 apart is that
    run 1 is the repetition the recorder has OPEN: there, and only there, the
    hook reads the anchors and pays two `.tolist()` device-to-host copies per
    propose, one for the anchor indices and one for the drafts. Those are
    behind `self.active`, and they are the whole of what a warm repetition
    skips. The two `is_current_stream_capturing()` calls per propose are NOT
    skipped -- every leg of this arm pays them, warm ones included.

    Run 1 is therefore instrumented, or cold, or both, and never merely
    inconvenient. `.agents/benchmarking.md` permits a named cause and nothing
    else, and it requires the named cause to be the REAL one, so both are named.
    Every leg stays in `legs` so a reader can see what was dropped.
    """

    warm = [leg for leg in legs if is_warm_leg(leg)]
    if not warm:
        raise HarnessError(
            "legs: every leg is a cold leg, so the discard leaves nothing to fold"
        )
    return {
        "metrics": {"output_throughput_tok_s": statistics.median(leg["tok_s"] for leg in warm)},
        "legs": list(legs),
        "warm_legs": len(warm),
        "cold_legs_discarded": len(legs) - len(warm),
        "cold_discard_cause": (
            "run 1 of each repetition group is discarded for a named cause and is "
            "retained in `legs`. The cause differs by arm and both are named: our "
            "arm starts one process per prompt, so its run 1 loads the model and "
            "carries the first graph capture and the first KV allocation; the "
            "oracle arm builds one LLM for all the prompts and builds it before "
            "the prompt loop, so none of those three is inside any of its legs, "
            "and its run 1 is instead the repetition the draft recorder has OPEN, "
            "on which alone the hook reads the anchors and pays two .tolist() "
            "device-to-host copies per propose"
        ),
    }


def clock_state_reasons(
    record: Mapping[str, Any] | None, *, label: str, detail: str = ""
) -> list[str]:
    """Reasons the clock does not attribute this arm's number.

    Delegated whole to `tools/bench/gpu_clock_state.py`, which
    `.agents/benchmarking.md` names as the one helper for this and requires any
    new harness to import rather than reimplement. An ABSENT record is the
    refusal this wrapper adds: the helper judges a window, and a window nobody
    sampled is not a healthy one.

    `detail` is the SAMPLER'S own message when it produced no summary to judge
    -- an entirely idle window, a mid-window field change, a failed
    `nvidia-smi`, a sampler that had to be killed. It is appended rather than
    substituted, because the rule is the same one either way and only the cause
    differs. Omit it and the refusal cannot tell those four apart, which costs
    the next run a lease to learn what this one already knew.
    """

    if record is None:
        reason = (
            f"clock: the {label} arm sampled no clock window. A number is quotable only "
            "with the clock it was taken at -- a byte-identical kernel moved 9.65% "
            "between two boots with nothing throttling (#543)"
        )
        if detail.strip():
            reason += f". Its sampler: {detail.strip()}"
        return [reason]
    return clock_reasons(record, label=label)


def clock_pairing(
    ours: Mapping[str, Any] | None,
    theirs: Mapping[str, Any] | None,
    *,
    allow_cross_boot: bool = False,
) -> tuple[list[str], dict[str, Any] | None]:
    """The clock block that must sit BESIDE the ratio, and why it may not.

    One window per arm, compared. A single window spanning both arms cannot see
    a cross-arm offset at all, and the offset is the term that transfers into
    the ratio: a 12.79% median-clock delta repriced a byte-identical kernel by
    9.65%, larger than either deficit it was being used to rank (#543). So the
    pairing is delegated whole to `gpu_clock_state.compare_clock_records`, which
    owns the four thresholds and the argument for each of them.
    """

    reasons = clock_state_reasons(ours, label="ours") + clock_state_reasons(
        theirs, label="vllm"
    )
    if ours is None or theirs is None:
        return reasons, None
    block = compare_clock_records(ours, theirs, allow_cross_boot=allow_cross_boot)
    for reason in block["reasons"]:
        if reason not in reasons:
            reasons.append(reason)
    return reasons, block


def ratio(ours: float, theirs: float, *, polarity: str) -> float:
    """Our standing against the denominator, oriented so above 1.0 is better."""

    if polarity not in ("higher_is_better", "lower_is_better"):
        raise HarnessError(f"unknown axis polarity: {polarity!r}")
    if theirs == 0 or ours == 0:
        raise HarnessError("a ratio against a zero measurement is not defined")
    if polarity == "higher_is_better":
        return float(ours) / float(theirs)
    return float(theirs) / float(ours)


def axis_rows(
    ours: Mapping[str, Any], theirs: Mapping[str, Any], floors: Mapping[str, float]
) -> list[dict[str, Any]]:
    """One row per required axis: both absolutes, the ratio, and the verdict.

    An axis below its floor is an OPEN GAP and never a rounding error, and the
    verdict never reads as a ceiling: `AGENTS.md` forbids declaring one, so the
    failing verdict names the axis as open and leaves the next hypothesis to the
    row's spec.
    """

    rows: list[dict[str, Any]] = []
    for axis, polarity in sorted(AXIS_POLARITY.items()):
        if axis not in ours or axis not in theirs:
            rows.append(
                {
                    "axis": axis,
                    "polarity": polarity,
                    "ours": ours.get(axis),
                    "vllm": theirs.get(axis),
                    "ratio": None,
                    "floor": floors.get(axis),
                    "verdict": "NOT MEASURED",
                }
            )
            continue
        value = ratio(float(ours[axis]), float(theirs[axis]), polarity=polarity)
        floor = floors.get(axis)
        if floor is None:
            verdict = "RECORDED, no floor declared"
        elif value >= float(floor):
            verdict = "SATISFIED"
        else:
            verdict = "OPEN GAP"
        rows.append(
            {
                "axis": axis,
                "polarity": polarity,
                "ours": float(ours[axis]),
                "vllm": float(theirs[axis]),
                "ratio": value,
                "floor": floor,
                "verdict": verdict,
            }
        )
    return rows


def build_speed_result(
    *,
    ours: Mapping[str, Any],
    theirs: Mapping[str, Any],
    floors: Mapping[str, float] | None = None,
    allow_cross_boot: bool = False,
) -> dict[str, Any]:
    """Fold two committed arm records into the citable result, or refuse.

    Refuses BEFORE it computes anything. `.agents/benchmark-record.md` cites the
    object this returns, so a result that exists at all is a result whose whole
    precondition set was checked -- there is no partial shape to quote out of.
    """

    reasons: list[str] = []
    reasons += oracle_identity_reasons(
        theirs.get("oracle_runtime_version"), theirs.get("oracle_expected_commit")
    )
    reasons += attention_backend_reasons(
        resolved=theirs.get("attention_backend"),
        declared=theirs.get("attention_backend_declared"),
        source=theirs.get("attention_backend_source"),
        probe=theirs.get("attention_backend_probe"),
    )
    reasons += denominator_reasons(theirs.get("config") or {})
    for arm, label in ((ours, "ours"), (theirs, "vllm")):
        reasons += sse_keepalive_reasons(arm.get("env") or {})
        reasons += checkpoint_reasons(arm.get("artifacts") or [])
        reasons += contention_reasons(arm.get("contention") or {})
        reasons += model_binding_reasons(
            arm.get("models") or {}, arm.get("artifacts") or [], label=label
        )
    reasons += build_recipe_reasons(ours.get("build") or {}, label="ours")
    # The vLLM arm's build is checked against the ORACLE HEAD it declares, not
    # merely for being non-empty: the capture runs from our checkout, so the
    # revision nearest to hand is the wrong one.
    reasons += oracle_build_reasons(theirs)
    clock_block_reasons, clock_block = clock_pairing(
        ours.get("clock"), theirs.get("clock"), allow_cross_boot=allow_cross_boot
    )
    reasons += clock_block_reasons
    reasons += workload_reasons(ours.get("workload") or {}, theirs.get("workload") or {})
    require_no_reasons(reasons, what="DFlash2 speed result")

    return {
        "subject": "SPEC-DFLASH2",
        "issue": "https://github.com/mudler/vllm.cpp/issues/1562",
        "axes": axis_rows(
            ours.get("metrics") or {}, theirs.get("metrics") or {}, floors or {}
        ),
        "preconditions": {
            "oracle_runtime_version": theirs.get("oracle_runtime_version"),
            "oracle_expected_commit": theirs.get("oracle_expected_commit"),
            "attention_backend": theirs.get("attention_backend"),
            "attention_backend_source": theirs.get("attention_backend_source"),
            # WHAT ACTUALLY RAN, beside the one label the ratio is compared
            # against. Three backends resolve at once on this model (#1658), so
            # a citable result that carried only the scalar would under-describe
            # the denominator in the field a reader looks at first.
            "attention_backend_groups": theirs.get("attention_backend_groups"),
            "attention_backend_kwarg": theirs.get("attention_backend_kwarg"),
            "sse_ping_s": (ours.get("env") or {}).get(SSE_PING_ENV),
            "enforce_eager": (theirs.get("config") or {}).get("enforce_eager"),
            "workload": dict(ours.get("workload") or {}),
            "models": {"ours": ours.get("models"), "vllm": theirs.get("models")},
            "speculative_config": ours.get("speculative_config"),
            "artifacts": {
                "ours": list(ours.get("artifacts") or []),
                "vllm": list(theirs.get("artifacts") or []),
            },
            "build": {
                "ours": ours.get("build"),
                "vllm": theirs.get("build"),
                # The tree that RAN the instrument, which is neither arm's build
                # and was previously recorded as the denominator's.
                "harness": theirs.get("harness_build"),
            },
            "clock": {
                "ours": ours.get("clock"),
                "vllm": theirs.get("clock"),
                "pairing": clock_block,
            },
            "contention": {
                "ours": ours.get("contention"),
                "vllm": theirs.get("contention"),
            },
        },
    }


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HarnessError(f"cannot read arm record {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise HarnessError(f"arm record {path} is not JSON: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fold two DFlash2 arm records into a result")
    parser.add_argument("command", choices=["summarize"])
    parser.add_argument("--ours", type=pathlib.Path, required=True)
    parser.add_argument("--vllm", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--allow-cross-boot",
        action="store_true",
        help="waive the BOOT ID and nothing else; records a caveat rather than "
        "passing silently (gpu_clock_state owns the rule)",
    )
    parser.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="AXIS=VALUE",
        help="declare an axis floor; an axis below it is recorded as an OPEN GAP",
    )
    args = parser.parse_args(argv)

    floors: dict[str, float] = {}
    for item in args.floor:
        axis, _, value = str(item).partition("=")
        if axis not in AXIS_POLARITY or not value:
            raise HarnessError(f"--floor expects AXIS=VALUE over {sorted(AXIS_POLARITY)}")
        floors[axis] = float(value)

    result = build_speed_result(
        ours=_load(args.ours),
        theirs=_load(args.vllm),
        floors=floors,
        allow_cross_boot=args.allow_cross_boot,
    )
    if args.output is not None:
        write_json_atomic(args.output, result)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HarnessError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        sys.exit(2)
