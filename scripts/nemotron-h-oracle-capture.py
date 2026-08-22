#!/usr/bin/env python3
# Nemotron-3.5-Lightning-30B-A3B-NVFP4 (`NemotronHForCausalLM`) — the generator
# for tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json, and the
# validator for its provenance contract (#926).
#
# ── WHY THIS FILE EXISTS ────────────────────────────────────────────────────
# The golden was committed by `af8170154` (#517) with NO generator. That capture
# ran from `$HOME/venvs/vllm-oracle-next` on `dgx.casa`, and the host was
# reimaged on 2026-08-14, so the driver, its log and its engine configuration are
# gone. `oracle.json` recorded the model, the revision, the sampling parameters
# and the vllm/transformers/flashinfer versions — and NOT ONE engine knob. A
# reference nobody can regenerate and nobody can attribute is not a pin, and the
# A2-Q1 device mamba arm's 95/96 (#1289, #1388) is currently being scored against
# exactly that. `.agents/specs/nemotron-oracle-golden-provenance.md` carries the
# recovery attempt and its negative result.
#
# Two runs have since failed to reproduce the golden's third prompt, each under a
# DIFFERENT and FULLY RECORDED configuration, which is what makes the missing
# record the defect rather than a curiosity:
#
#   * 2026-08-18, `/workspace/nhspeed/oracle_only.sh` attempt `a` on `dgx:gpu0`
#     (`/mnt/nas_share/rc/nhspeed/oracle.a.out`): 32/32, 32/32, **26/32**, twice
#     in one process, `ORACLE TOKEN MATCH: 180/192`.
#   * the #926 rebuild (`enforce_eager=True`, `gpu_memory_utilization=0.25`,
#     `max_model_len=4096`): 32/32, 32/32, **29/32**, diverging at index 29.
#
# Same box, same checkpoint, same greedy sampling, two different configurations,
# two different answers, each internally repeatable. That is CONFIGURATION
# SENSITIVITY, not non-determinism, and AGENTS.md admits a ratified
# distributional gate only for a non-deterministic greedy decode. So the licensed
# repair is to re-derive the golden under a NAMED configuration — which is what
# `--capture` below does, and why it refuses to write a golden it cannot
# attribute.
#
# ── THE ONE LEAD THAT IS ALREADY CLOSED ─────────────────────────────────────
# `kv_cache_dtype=fp8_e4m3` is auto-selected on this checkpoint and vLLM imputes
# the missing q scale ("Checkpoint does not provide a q scaling factor. Setting
# it to k_scale"). It reads like an unrecorded implicit choice. It is not a
# candidate difference: the CHECKPOINT carries it. `config.json`'s
# `quantization_config.kv_cache_scheme` is `{dynamic: false, num_bits: 8, type:
# "float"}` and `hf_quant_config.json`'s `quantization.kv_cache_quant_algo` is
# `"FP8"`, and at the pin `vllm/engine/arg_utils.py:1916` resolves the DEFAULT
# `kv_cache_dtype="auto"` through `vllm/utils/torch_utils.py:374-392` ->
# `:310-362`, which maps that dict to `"fp8"`. Every run of this checkpoint that
# does not explicitly override the knob gets fp8_e4m3, including the capture. It
# is a COMMON TERM. Recorded anyway, because "the same on both sides" is a
# measurement and not an assumption.
#
# ── MODES ───────────────────────────────────────────────────────────────────
#   --check <golden>     validate the provenance contract. Needs NO vLLM, no GPU
#                        and no checkpoint; this is what CI runs.
#   --verify <golden>    run the oracle and compare against a committed golden.
#                        Reports a CONFIGURATION difference before a token
#                        difference, because an unattributable token count is
#                        not evidence.
#   --capture --out <p>  run the oracle and WRITE a golden that records the
#                        configuration it ran under.
#
# `--verify` and `--capture` need the pinned oracle. Run them under an `rc` lease
# on `dgx:gpu0` (never `ssh` to a fleet device), from the pinned venv:
#
#   $VENV/bin/python scripts/nemotron-h-oracle-capture.py --capture \
#       --model $CHECKPOINT_ROOT/nemotron-3.5-lightning-30b-nvfp4 \
#       --profile nhspeed-a --legs 2 \
#       --out tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json
#
# NOTE the body is guarded by `if __name__ == "__main__":`. That is not style:
# vLLM v1 spawns EngineCore, the module re-imports, and an unguarded driver fails
# as a `multiprocessing` traceback naming neither vLLM nor the caller. The tell
# is the banner printing twice (`.agents/specs/nemotron-h-model.md` §5a).
"""Capture, verify and validate the Nemotron-3.5-Lightning greedy oracle golden."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# The three prompts the committed golden carries, in order. Kept here so a
# capture that drifts from the committed battery is caught by the writer rather
# than by a reader months later.
PROMPTS = [
    "The capital of France is",
    "Write the first five Fibonacci numbers:",
    "Explain what a state space model is, in one sentence:",
]

# The parity pin. `vllm.__version__` at this commit spells it
# `0.23.1rc1.dev1511+g555967922` from a source tree and `0.1.dev1+g555967922`
# from the FlashInfer-only wheel, so the ASSERTION is on the commit substring,
# never on the whole version string.
ORACLE_COMMIT = "555967922"

# The checkpoint revision the golden belongs to, mirrored from
# `parity::kNemotron35LightningNvfP4Revision` (tests/parity/hf_snapshot.h).
CHECKPOINT_REVISION = "29f2d1746d8f41e316523194b19018707749b1b1"

SCHEMA = 2

# ── Named engine configurations ─────────────────────────────────────────────
# A profile is a NAME for a configuration that this repository has a recorded
# run of. It is not a guess at the lost one. `nhspeed-a` is the 2026-08-18 run
# whose full resolved config is readable at
# /mnt/nas_share/rc/nhspeed/oracle.a.out (the worker's /workspace/nhspeed), and
# it is the only oracle configuration on this checkpoint for which this
# repository has determinism evidence: two legs in one process, byte-identical.
# CUDA graphs stay ON in it, because `--enforce-eager` is never the denominator.
PROFILES = {
    "nhspeed-a": {
        "max_model_len": 512,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.30,
        "max_num_batched_tokens": 512,
        "enforce_eager": False,
        "num_gpu_blocks_override": None,
    },
}

# Every engine key a re-derived golden must carry. The list is the set of knobs
# that can move a greedy argmax on this engine — batch shape, prefill chunking,
# paging, graph capture, the resolved kernel backends, and the dtypes — plus the
# two that identify the run at all. A key that is absent is not "default"; it is
# unrecorded, which is the whole defect this file exists to close.
REQUIRED_ENGINE_KEYS = (
    "attention_backend",
    "block_size",
    "cudagraph_capture_sizes",
    "cudagraph_mode",
    "compilation_mode",
    "dtype",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enforce_eager",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "moe_backend",
    "num_gpu_blocks",
    "num_gpu_blocks_override",
    "quantization",
    "seed",
    "tensor_parallel_size",
)

# `num_gpu_blocks_override` and `cudagraph_capture_sizes` are legitimately null
# or empty (no override; eager). Everything else must carry a value: a null
# there means the reader failed, and a failed read must not be written down as
# a fact.
NULLABLE_ENGINE_KEYS = frozenset({"num_gpu_blocks_override"})

ISSUE_PREFIX = "https://github.com/mudler/vllm.cpp/issues/"

# ── What an UNATTRIBUTED golden has to carry (#926) ─────────────────────────
# The attributed arm is gated by STRUCTURE: `engine.resolved` either carries all
# twenty keys or it does not, and a checker can answer that exactly. The
# unattributed arm has no structure to gate, because its whole record IS prose —
# and a contract that asks only whether the prose is TRUTHY gates the shape and
# not the substance.
#
# That gap was demonstrated, not imagined. A reviewer gutted
# `forced_by_checkpoint_or_device`, `evidence` and `captured_utc_is` and replaced
# an 814-character `unrecoverable_reason` with the single word "dunno", and all
# four gates stayed green while `--check` kept printing
# `engine_config_recorded=False`. Every argument this artifact rests on — that
# nothing has ever reproduced prompt 2, that `kv_cache_dtype` is a common term
# rather than a candidate difference, that a distributional gate is inadmissible
# — lived in ungated prose inside a data file. Silence wearing the shape of a
# record is exactly what #926 filed, so the gap is closed here rather than noted.
#
# It is closed in the two ways a checker can actually answer, and in no way it
# cannot.

# 1. STRUCTURE. Present, an object, and every sub-key a non-empty string.
#
# `forced_by_checkpoint_or_device` names the terms that are COMMON to every
# unoverridden run of this checkpoint on this class of device. It is what
# NARROWS "unrecoverable" to the knobs a driver passes, so deleting an entry
# widens the unrecoverable set without saying so.
REQUIRED_FORCED_TERM_KEYS = ("kv_cache_dtype", "moe_backend", "dtype", "quantization")

# `evidence` carries the two findings that decide how this golden may be USED:
# whether anything has ever reproduced it, and which gate form its behaviour
# licenses. `gate_form` is the one that refuses a distributional gate; a golden
# that loses it loses the reason the refusal was on evidence rather than taste.
REQUIRED_EVIDENCE_KEYS = ("never_reproduced", "gate_form")

# 2. SUBSTANCE. A length floor, on the four fields whose content is an ARGUMENT
# rather than a value.
#
# A floor cannot prove the prose is TRUE. Nothing in a checker can. A keyword
# grep would be worse than useless here: it proves only the checker's own
# vocabulary, and it reds an honest rewording. What a floor detects is REMOVAL,
# which is the threat that was actually demonstrated — one word in place of a
# paragraph.
#
# 80 is set from measurement rather than taste. In the shipped golden the four
# floored fields are 814, 448, 702 and 293 characters, so the tightest margin is
# 3.7x and no honest rewrite of an argument comes near it, while "dunno",
# "unknown", "TBD" and "see the spec" are all under it. AGENTS.md's rule that a
# gate firing on ordinary work is the defect is why the floor is set from the
# SHORTEST real field and not from the longest.
MIN_ARGUMENT_CHARS = 80

# The floored fields, as paths under `capture`. `captured_utc_is` is required
# and must be a non-empty string, but it is deliberately NOT floored: its job is
# to say what the timestamp is, and that can honestly be said in a clause —
# "af8170154's author date, not a run time" is 44 characters and is not a
# hand-wave. Flooring it would be a gate that fires on ordinary work.
ARGUMENT_FIELDS = (
    ("unrecoverable_reason",),
    ("forced_by_checkpoint_or_device", "kv_cache_dtype"),
    ("evidence", "never_reproduced"),
    ("evidence", "gate_form"),
)


def _is_prose(value):
    """True when `value` is a string carrying something other than whitespace."""
    return isinstance(value, str) and bool(value.strip())


# ── The provenance contract ─────────────────────────────────────────────────


class ContractError(Exception):
    """A golden that cannot be attributed, with the reason it cannot."""


def check_golden(doc):
    """Return the list of contract violations in `doc`. Empty means it holds.

    Pure: no vLLM, no GPU, no checkpoint. This is the gate.
    """
    problems = []

    def need(container, key, where):
        if not isinstance(container, dict) or key not in container:
            problems.append(f"{where}: missing '{key}'")
            return False
        return True

    for key in ("vllm", "transformers", "flashinfer", "model", "revision",
                "sampling", "golden", "capture"):
        need(doc, key, "oracle.json")

    sampling = doc.get("sampling")
    if isinstance(sampling, dict):
        for key in ("temperature", "max_tokens"):
            need(sampling, key, "sampling")
    elif "sampling" in doc:
        problems.append("sampling: not an object")

    # A comparison over zero elements reports a perfect score, so the golden's
    # own width is asserted rather than trusted.
    golden = doc.get("golden")
    if not isinstance(golden, list) or not golden:
        problems.append("golden: must be a non-empty array")
    else:
        want = sampling.get("max_tokens") if isinstance(sampling, dict) else None
        for i, entry in enumerate(golden):
            where = f"golden[{i}]"
            for key in ("prompt", "prompt_token_ids", "token_ids", "text"):
                need(entry, key, where)
            if not isinstance(entry, dict):
                continue
            if not entry.get("prompt_token_ids"):
                problems.append(f"{where}: empty prompt_token_ids")
            ids = entry.get("token_ids")
            if not ids:
                problems.append(f"{where}: empty token_ids")
            elif isinstance(want, int) and len(ids) != want:
                problems.append(
                    f"{where}: {len(ids)} token_ids against sampling.max_tokens={want}")

    capture = doc.get("capture")
    if not isinstance(capture, dict):
        if "capture" in doc:
            problems.append("capture: not an object")
        return problems

    if capture.get("schema") != SCHEMA:
        problems.append(f"capture.schema: expected {SCHEMA}, got {capture.get('schema')!r}")
    for key in ("captured_utc", "host", "generator"):
        if not capture.get(key):
            problems.append(f"capture: '{key}' is empty")

    recorded = capture.get("engine_config_recorded")
    if not isinstance(recorded, bool):
        problems.append("capture.engine_config_recorded: must be true or false")
        return problems

    if not recorded:
        # An UNATTRIBUTED golden is allowed to exist — deleting it would delete
        # evidence — but it has to say so out loud and name the issue that owes
        # the re-derivation. Silence is what #926 is.
        # `_is_prose`, not truthiness. The C++ copy of this contract already
        # spells it `is_string()`, and a bare truthiness test let a NUMBER
        # through here -- `"unrecoverable_reason": 123` returned zero problems
        # from this checker while the C++ gate refused it. Two copies of one
        # contract disagreeing about what satisfies it is the drift this design
        # promises cannot happen, so the weaker copy is the one that moves.
        if not _is_prose(capture.get("unrecoverable_reason")):
            problems.append(
                "capture.engine_config_recorded is false and "
                "'unrecoverable_reason' is not prose: an unattributed golden "
                "must state IN WORDS why it cannot be attributed")
        issue = capture.get("issue") or ""
        if not str(issue).startswith(ISSUE_PREFIX):
            problems.append(
                "capture.engine_config_recorded is false and 'issue' does not "
                f"name a vllm.cpp issue (got {issue!r})")
        if capture.get("engine") is not None:
            problems.append(
                "capture.engine_config_recorded is false but 'engine' is not "
                "null: a configuration that is recorded is not unrecorded")

        # ── The substance, not only the shape ───────────────────────────────
        # Everything above this point is satisfied by a file that says
        # "unrecorded", names an issue and argues NOTHING. That file passes as
        # a record while being one, and it is the state #926 filed. What
        # follows requires the record to still be there.
        if not _is_prose(capture.get("captured_utc_is")):
            problems.append(
                "capture.captured_utc_is is empty: an unattributed golden's "
                "timestamp is not a run time unless the file says what it is, "
                "and a commit's author date read as a capture time is a "
                "fabricated provenance")
        for parent, keys in (("forced_by_checkpoint_or_device",
                              REQUIRED_FORCED_TERM_KEYS),
                             ("evidence", REQUIRED_EVIDENCE_KEYS)):
            block = capture.get(parent)
            if not isinstance(block, dict):
                problems.append(
                    f"capture.{parent}: must be an object on an unattributed "
                    f"golden, got {type(block).__name__} -- an unrecoverable "
                    "configuration is a claim, and a claim without its "
                    "supporting record is the silence this contract refuses")
                continue
            for key in keys:
                if not _is_prose(block.get(key)):
                    problems.append(
                        f"capture.{parent}['{key}'] is empty: it is named by "
                        "this contract because deleting it removes an argument "
                        "the golden's admissibility rests on")
        for path in ARGUMENT_FIELDS:
            value = capture
            for step in path:
                value = value.get(step) if isinstance(value, dict) else None
            # Absence and emptiness are already reported above, so this arm
            # only judges LENGTH and cannot report the same defect twice.
            if _is_prose(value) and len(value.strip()) < MIN_ARGUMENT_CHARS:
                problems.append(
                    "capture.%s is %d characters, under the %d this contract "
                    "requires of a field whose content is an ARGUMENT: a "
                    "one-word answer here is the record going missing while "
                    "the file keeps its shape" % (
                        ".".join(path), len(value.strip()), MIN_ARGUMENT_CHARS))
        return problems

    engine = capture.get("engine")
    if not isinstance(engine, dict):
        problems.append("capture.engine: must be an object when the config is recorded")
        return problems
    resolved = engine.get("resolved")
    if not isinstance(resolved, dict):
        problems.append("capture.engine.resolved: must be an object")
        return problems
    for key in REQUIRED_ENGINE_KEYS:
        if key not in resolved:
            problems.append(f"capture.engine.resolved: missing '{key}'")
        elif resolved[key] is None and key not in NULLABLE_ENGINE_KEYS:
            problems.append(
                f"capture.engine.resolved['{key}'] is null: a value that could "
                "not be read is not a value that was default")
    for key in ("torch", "device"):
        if not engine.get(key):
            problems.append(f"capture.engine: '{key}' is empty")

    batch = capture.get("batch")
    if not isinstance(batch, dict) or not batch.get("shape"):
        problems.append(
            "capture.batch.shape is empty: batching re-orders reductions, so a "
            "golden that does not say how the prompts were submitted is not "
            "reproducible")
    legs = capture.get("legs")
    if not isinstance(legs, int) or legs < 2:
        problems.append(
            f"capture.legs: expected at least 2 legs, got {legs!r} -- one leg "
            "cannot show the configuration is deterministic")
    if capture.get("legs_agree") is not True:
        problems.append(
            "capture.legs_agree is not true: a golden whose own legs disagree "
            "records a coin flip")
    return problems


# There is deliberately NO builder for the unattributed provenance block here.
#
# One lived at this line and had exactly one caller: a fixture in
# tests/scripts/test_nemotron_h_oracle_capture.py. `--capture` cannot reach it —
# a capture that runs records its configuration, which is the whole point of the
# mode — so no production path produced the shape it described, and the one
# unattributed golden this repository has was written by hand and carried three
# keys the builder could not emit. Under AGENTS.md's "Nothing lands dead" that
# is a helper documenting a production shape nothing in production produces.
#
# Deleting it also repairs the fixture. This suite's own rule is that a fixture
# must never be derived from the module under test, because setup and
# expectation then move together and a key deleted from the checker deletes its
# own test. The unattributed fixture was the one place that broke that rule. It
# is a test-owned literal now, and the contract's requirements are proven key by
# key by dropping each one from it.
#
# `--capture` builds the ATTRIBUTED block inline in main(), where the values it
# needs are in scope; that block is reached, and check_golden() refuses it if it
# is wrong.


# ── The oracle side ─────────────────────────────────────────────────────────


def assert_oracle_identity():
    """Abort unless the importable vLLM IS the pin. Identity is asserted, never
    assumed: `$HOME/venvs/vllm-oracle` on dgx has resolved to a 0.25.0 rollback
    that predates `NemotronHMoEDecoderLayer` entirely, and a run through it fails
    in a way that reads as 'the model is unsupported'."""
    import vllm

    version = vllm.__version__
    where = vllm.__file__
    print(f"vllm {version} {where}", flush=True)
    if ORACLE_COMMIT not in version:
        raise SystemExit(
            f"WRONG ORACLE: vllm {version} does not carry the pin {ORACLE_COMMIT}")
    identity = {"vllm": version, "vllm_file": where}
    for name in ("transformers", "flashinfer"):
        try:
            module = __import__(name)
            identity[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            identity[name] = f"IMPORT_FAILED: {exc}"
    print("ORACLE_IDENTITY_OK", json.dumps(identity), flush=True)
    return identity


def read_resolved_config(llm):
    """Read the engine configuration BACK OUT of the built engine.

    Reading back is the point. `kv_cache_dtype`, the attention backend, the MoE
    backend, the block size and the block count are all chosen by vLLM from the
    checkpoint and the device, so the kwargs a driver passed are not the
    configuration it ran. A key this cannot read stays ABSENT and the writer
    refuses; it is never filled in with a plausible default.
    """
    config = llm.llm_engine.vllm_config
    cache = config.cache_config
    scheduler = config.scheduler_config
    model = config.model_config
    parallel = config.parallel_config
    compilation = config.compilation_config
    resolved = {}

    def put(key, value):
        if value is not None:
            resolved[key] = value

    put("block_size", getattr(cache, "block_size", None))
    put("num_gpu_blocks", getattr(cache, "num_gpu_blocks", None))
    resolved["num_gpu_blocks_override"] = getattr(cache, "num_gpu_blocks_override", None)
    put("gpu_memory_utilization", getattr(cache, "gpu_memory_utilization", None))
    put("kv_cache_dtype", str(getattr(cache, "cache_dtype", "") or "") or None)
    put("enable_prefix_caching", getattr(cache, "enable_prefix_caching", None))
    put("max_model_len", getattr(model, "max_model_len", None))
    put("dtype", str(getattr(model, "dtype", "") or "") or None)
    put("quantization", getattr(model, "quantization", None))
    put("seed", getattr(model, "seed", None))
    put("enforce_eager", getattr(model, "enforce_eager", None))
    put("max_num_seqs", getattr(scheduler, "max_num_seqs", None))
    put("max_num_batched_tokens", getattr(scheduler, "max_num_batched_tokens", None))
    put("enable_chunked_prefill", getattr(scheduler, "enable_chunked_prefill", None))
    put("tensor_parallel_size", getattr(parallel, "tensor_parallel_size", None))
    put("compilation_mode", str(getattr(compilation, "mode", "") or "") or None)
    put("cudagraph_mode", str(getattr(compilation, "cudagraph_mode", "") or "") or None)
    sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    if sizes is not None:
        resolved["cudagraph_capture_sizes"] = list(sizes)
    for key, env in (("attention_backend", "VLLM_ATTENTION_BACKEND"),
                     ("moe_backend", "VLLM_FUSED_MOE_BACKEND")):
        # vLLM PRINTS the backend it selected and does not always expose it on
        # the config, so the environment override is recorded when it is set and
        # the log line is the authority otherwise. `--attention-backend` and
        # `--moe-backend` below let the caller state what the log said, and the
        # writer refuses if neither is available.
        value = os.environ.get(env)
        if value:
            resolved[key] = value
    return resolved


def generate(llm, sampling_params, prompt_inputs):
    """One prompt per `generate()` call. Batching re-orders reductions, and this
    tree has already misread that as non-determinism once (see
    scripts/deepseek-v2-oracle-capture.py). The shape is RECORDED either way."""
    results = []
    for prompt in prompt_inputs:
        out = llm.generate([prompt], sampling_params)
        results.append(list(out[0].outputs[0].token_ids))
    return results


def run_oracle(args):
    """Build the engine, run `args.legs` legs, and return (identity, resolved,
    per-leg token ids, per-prompt token ids, texts)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    identity = assert_oracle_identity()
    profile = dict(PROFILES[args.profile])
    for key in ("max_model_len", "max_num_seqs", "gpu_memory_utilization",
                "max_num_batched_tokens", "enforce_eager", "num_gpu_blocks_override"):
        override = getattr(args, key, None)
        if override is not None:
            profile[key] = override
    kwargs = {k: v for k, v in profile.items() if v is not None}
    kwargs["enforce_eager"] = profile["enforce_eager"]
    kwargs["model"] = args.model
    print("ORACLE_KW", json.dumps({k: v for k, v in kwargs.items() if k != "model"}),
          flush=True)

    started = time.time()
    llm = LLM(**kwargs)
    print("ORACLE_LOAD_S=%.1f" % (time.time() - started), flush=True)

    resolved = read_resolved_config(llm)
    for key, value in ((("attention_backend"), args.attention_backend),
                       (("moe_backend"), args.moe_backend)):
        if value:
            resolved[key] = value
    print("ORACLE_RESOLVED", json.dumps(resolved, default=str), flush=True)

    golden_prompts = None
    if args.golden_prompt_ids:
        golden_prompts = args.golden_prompt_ids
    if args.tokens_prompt:
        if golden_prompts is None:
            raise SystemExit(
                "--tokens-prompt needs prompt_token_ids, which only a golden "
                "carries; pass --verify <golden> or drop the flag")
        inputs = [TokensPrompt(prompt_token_ids=ids) for ids in golden_prompts]
        shape = "one TokensPrompt per generate() call (pre-tokenized)"
    else:
        inputs = list(PROMPTS)
        shape = "one text prompt per generate() call"

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                     ignore_eos=args.ignore_eos)
    legs = []
    texts = None
    for leg in range(args.legs):
        print("ORACLE_LEG %d" % (leg + 1), flush=True)
        out = []
        leg_texts = []
        for prompt in inputs:
            result = llm.generate([prompt], sampling_params)
            out.append(list(result[0].outputs[0].token_ids))
            leg_texts.append(result[0].outputs[0].text)
            if texts is None:
                pass
        legs.append(out)
        if leg == 0:
            texts = leg_texts
            prompt_ids = [list(r) for r in _prompt_ids(llm, inputs, sampling_params)]
    return identity, resolved, legs, prompt_ids, texts, shape


def _prompt_ids(llm, inputs, sampling_params):
    """vLLM's OWN tokenization of each prompt, taken from the RequestOutput.

    A silently unapplied BOS has scored 0/6 in this tree while emitting fluent
    English, and the committed tokenization is what caught it.
    """
    ids = []
    for prompt in inputs:
        result = llm.generate([prompt], sampling_params)
        ids.append(list(result[0].prompt_token_ids))
    return ids


def compare(expected, got):
    n = min(len(expected), len(got))
    return sum(1 for i in range(n) if expected[i] == got[i]), n


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", metavar="GOLDEN",
                     help="validate a committed golden's provenance contract "
                          "(no vLLM, no GPU, no checkpoint)")
    mode.add_argument("--verify", metavar="GOLDEN",
                     help="run the oracle and compare against a committed golden")
    mode.add_argument("--capture", action="store_true",
                      help="run the oracle and write a golden")
    parser.add_argument("--out", help="output path for --capture")
    parser.add_argument("--model", help="checkpoint directory")
    parser.add_argument("--profile", default="nhspeed-a", choices=sorted(PROFILES),
                        help="named engine configuration (default: nhspeed-a)")
    parser.add_argument("--legs", type=int, default=2,
                        help="greedy legs in one process; >=2 or the golden "
                             "cannot claim determinism at its own configuration")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    parser.add_argument("--tokens-prompt", action="store_true",
                        help="feed the golden's prompt_token_ids instead of the "
                             "prompt text, skipping the tokenizer")
    parser.add_argument("--attention-backend",
                        help="the backend vLLM's startup log named, e.g. FLASHINFER")
    parser.add_argument("--moe-backend",
                        help="the NvFp4 MoE backend vLLM's startup log named, e.g. MARLIN")
    parser.add_argument("--host", help="the box this ran on, recorded into the golden")
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens",
                "num_gpu_blocks_override"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true",
                        default=None,
                        help="NOT the denominator; only for a diagnostic leg")
    args = parser.parse_args(argv)

    if args.check:
        with open(args.check, encoding="utf-8") as handle:
            doc = json.load(handle)
        problems = check_golden(doc)
        capture = doc.get("capture") or {}
        for problem in problems:
            print("CONTRACT: " + problem, file=sys.stderr)
        print("checked %s: %d golden entr%s, engine_config_recorded=%r, %d problem(s)"
              % (args.check, len(doc.get("golden") or []),
                 "y" if len(doc.get("golden") or []) == 1 else "ies",
                 capture.get("engine_config_recorded"), len(problems)))
        return 1 if problems else 0

    if args.capture and not args.out:
        parser.error("--capture needs --out")
    if not args.model:
        parser.error("--verify and --capture need --model")

    existing = None
    if args.verify:
        with open(args.verify, encoding="utf-8") as handle:
            existing = json.load(handle)
        args.golden_prompt_ids = [e["prompt_token_ids"] for e in existing["golden"]]
        args.max_tokens = int(existing["sampling"]["max_tokens"])
    else:
        args.golden_prompt_ids = None

    if args.legs < 2:
        raise SystemExit("--legs must be at least 2")

    identity, resolved, legs, prompt_ids, texts, shape = run_oracle(args)

    agree = all(leg == legs[0] for leg in legs[1:])
    print("ORACLE_LEGS_AGREE=%s" % agree, flush=True)

    if existing is not None:
        recorded = (existing.get("capture") or {}).get("engine_config_recorded")
        if not recorded:
            print("CONFIGURATION: the committed golden records NO engine "
                  "configuration, so this run cannot be held to it -- a token "
                  "difference below is UNATTRIBUTABLE, not a defect (#926)",
                  flush=True)
        total_matched = total_compared = 0
        for i, entry in enumerate(existing["golden"]):
            matched, compared = compare(entry["token_ids"], legs[0][i])
            total_matched += matched
            total_compared += compared
            print("ORACLE prompt %d: compared=%d matched=%d" % (i, compared, matched),
                  flush=True)
        print("ORACLE TOKEN MATCH: %d/%d" % (total_matched, total_compared), flush=True)
        return 0 if total_matched == total_compared else 2

    if not agree:
        raise SystemExit(
            "the legs of this capture DISAGREE with each other; a golden written "
            "from them would record a coin flip. Nothing was written.")

    document = {
        "vllm": identity["vllm"],
        "transformers": identity.get("transformers"),
        "flashinfer": identity.get("flashinfer"),
        "model": args.model,
        "revision": CHECKPOINT_REVISION,
        "sampling": {"temperature": 0.0, "max_tokens": args.max_tokens,
                     "ignore_eos": args.ignore_eos},
        "capture": {
            "schema": SCHEMA,
            "generator": "scripts/nemotron-h-oracle-capture.py",
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": args.host or os.uname().nodename,
            "engine_config_recorded": True,
            "unrecoverable_reason": None,
            "issue": ISSUE_PREFIX + "926",
            "engine": {
                "profile": args.profile,
                "resolved": resolved,
                "torch": _torch_version(),
                "device": _device_name(),
            },
            "batch": {"shape": shape, "prompts": len(PROMPTS)},
            "legs": args.legs,
            "legs_agree": agree,
        },
        "golden": [
            {"prompt": PROMPTS[i], "prompt_token_ids": prompt_ids[i],
             "token_ids": legs[0][i], "text": texts[i]}
            for i in range(len(PROMPTS))
        ],
    }
    problems = check_golden(document)
    if problems:
        for problem in problems:
            print("CONTRACT: " + problem, file=sys.stderr)
        raise SystemExit(
            "this capture does not satisfy the provenance contract, so it was "
            "NOT written: a golden that cannot be attributed is what #926 is")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print("WROTE %s" % args.out)
    return 0


def _torch_version():
    try:
        import torch

        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


def _device_name():
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return "%s (sm_%d%d)" % (torch.cuda.get_device_name(0), major, minor)
    except Exception:  # noqa: BLE001
        pass
    return None


if __name__ == "__main__":
    sys.exit(main())
