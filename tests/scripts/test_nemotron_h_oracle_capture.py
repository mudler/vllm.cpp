#!/usr/bin/env python3
"""Unit and mutation checks for scripts/nemotron-h-oracle-capture.py (#926).

The rule under test is the golden's PROVENANCE CONTRACT: a Nemotron oracle
golden either records the engine configuration it was captured under, or says in
the file itself that it does not and names the issue that owes the
re-derivation. Silence is the third state, and silence is what #926 is.

Every case is red before the rule exists. The file the contract was written
against -- `oracle.json` exactly as `af8170154` committed it -- is reproduced
here as `AF8170154_SHAPE` and is asserted to FAIL, so the suite fails if the
contract is ever widened enough to admit the artifact that motivated it.

The suite runs with no vLLM, no GPU and no checkpoint, which is the point: the
provenance defect is a records defect and its gate must not need the hardware
that the defect blocks.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/nemotron-h-oracle-capture.py"
SHIPPED_GOLDEN = ROOT / "tests/parity/goldens/nemotron_35_lightning_greedy/oracle.json"
CPP_CONSUMER = ROOT / "tests/vllm/models/test_nemotron_h_loader.cpp"
SPEC = importlib.util.spec_from_file_location("nemotron_h_oracle_capture", GENERATOR)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


# ── Test-owned literals ─────────────────────────────────────────────────────
# Never derived from the module under test. A fixture built from the production
# constant makes setup and expectation move together, and then a key deleted
# from REQUIRED_ENGINE_KEYS deletes its own test.
EXPECTED_ENGINE_KEYS = {
    "attention_backend",
    "block_size",
    "compilation_mode",
    "cudagraph_capture_sizes",
    "cudagraph_mode",
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
}

# The knobs #926 named by name when it said the goldens can be approximated but
# not re-derived. Asserted separately from the set above, because these three
# are the ones the rebuild differed on and are the reason the contract exists.
KNOBS_926_NAMED = ("enforce_eager", "gpu_memory_utilization", "max_model_len")

# The unattributed arm's own expectation, owned here for the same reason. The
# attributed arm is gated by structure; this arm's whole record is prose, so
# "present and truthy" gates the shape and not the substance -- a reviewer
# replaced an 814-character `unrecoverable_reason` with the word "dunno",
# emptied `evidence`, `forced_by_checkpoint_or_device` and `captured_utc_is`,
# and every gate stayed green.
EXPECTED_FORCED_TERM_KEYS = {"kv_cache_dtype", "moe_backend", "dtype", "quantization"}
EXPECTED_EVIDENCE_KEYS = {"never_reproduced", "gate_form"}

# The fields whose content is an ARGUMENT rather than a value, and the floor
# they carry. The floor detects REMOVAL; it does not and cannot claim the prose
# is true. 80 is owned here as a literal so that raising it in the checker to
# hide a shrinking record is red, and lowering it to admit one is red too.
EXPECTED_MIN_ARGUMENT_CHARS = 80
EXPECTED_ARGUMENT_FIELDS = {
    ("unrecoverable_reason",),
    ("forced_by_checkpoint_or_device", "kv_cache_dtype"),
    ("evidence", "never_reproduced"),
    ("evidence", "gate_form"),
}

# Long enough to clear EXPECTED_MIN_ARGUMENT_CHARS without being a paraphrase of
# the shipped golden's text. The fixture must not borrow the artifact's prose,
# or a case that deletes the artifact's prose still passes.
_ARGUMENT = (
    "A sentence long enough to be an argument rather than a hand-wave, which is "
    "the only property a length floor can honestly test for."
)

RESOLVED = {
    "attention_backend": "FLASHINFER",
    "block_size": 512,
    "compilation_mode": "CompilationMode.VLLM_COMPILE",
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
    "cudagraph_mode": "CUDAGraphMode.FULL_AND_PIECEWISE",
    "dtype": "torch.bfloat16",
    "enable_chunked_prefill": True,
    "enable_prefix_caching": False,
    "enforce_eager": False,
    "gpu_memory_utilization": 0.30,
    "kv_cache_dtype": "fp8_e4m3",
    "max_model_len": 512,
    "max_num_batched_tokens": 512,
    "max_num_seqs": 8,
    "moe_backend": "MARLIN",
    "num_gpu_blocks": 1258,
    "num_gpu_blocks_override": None,
    "quantization": "modelopt_mixed",
    "seed": 0,
    "tensor_parallel_size": 1,
}


def _cpp_string_list(source, name):
    """The set of string literals in the C++ initializer list called `name`.

    Parsed rather than grepped: two of the keys under test also appear in the
    C++ engine-key list, so `assertIn('"dtype"', source)` would pass without the
    unattributed arm naming it at all.
    """
    match = re.search(re.escape(name) + r"\s*=\s*\{(.*?)\};", source, re.S)
    if match is None:
        raise AssertionError(f"{CPP_CONSUMER.name} has no list called {name}")
    return set(re.findall(r'"([^"]*)"', match.group(1)))


def _entry(prompt, prompt_ids, tokens):
    return {"prompt": prompt, "prompt_token_ids": list(prompt_ids),
            "token_ids": list(tokens), "text": " x"}


def attributed():
    """A golden that DOES record its configuration."""
    return {
        "vllm": "0.23.1rc1.dev1511+g555967922",
        "transformers": "5.14.1",
        "flashinfer": "0.6.15.post1",
        "model": "/checkpoints/nemotron-3.5-lightning-30b-nvfp4",
        "revision": "29f2d1746d8f41e316523194b19018707749b1b1",
        "sampling": {"temperature": 0.0, "max_tokens": 4, "ignore_eos": True},
        "capture": {
            "schema": 2,
            "generator": "scripts/nemotron-h-oracle-capture.py",
            "captured_utc": "2026-08-21T00:00:00Z",
            "host": "dgx:gpu0 (GB10)",
            "engine_config_recorded": True,
            "unrecoverable_reason": None,
            "issue": "https://github.com/mudler/vllm.cpp/issues/926",
            "engine": {
                "profile": "nhspeed-a",
                "resolved": dict(RESOLVED),
                "torch": "2.13.0+cu130",
                "device": "NVIDIA GB10 (sm_121)",
            },
            "batch": {"shape": "one text prompt per generate() call", "prompts": 1},
            "legs": 2,
            "legs_agree": True,
        },
        "golden": [_entry("The capital of France is", [1, 2, 3], [9, 8, 7, 6])],
    }


def unattributed():
    """A golden that does NOT record its configuration and SAYS so.

    A TEST-OWNED literal, deliberately. This fixture used to be built by
    `capture.unattributed_capture()`, which made setup and expectation move
    together: a key dropped from the checker's requirements was also dropped
    from the fixture, so the case could not go red. That helper had no other
    caller and is gone (#926); the shape lives here, where the suite owns it.
    """
    doc = attributed()
    doc["capture"] = {
        "schema": 2,
        "generator": "none -- af8170154 committed no generator",
        "captured_utc": "2026-08-12T22:37:57Z",
        "captured_utc_is": "the committing author date, not a recorded run time",
        "host": "dgx.casa (GB10)",
        "engine_config_recorded": False,
        "unrecoverable_reason": _ARGUMENT,
        "issue": "https://github.com/mudler/vllm.cpp/issues/926",
        "engine": None,
        "batch": None,
        "legs": None,
        "legs_agree": None,
        "forced_by_checkpoint_or_device": {
            "kv_cache_dtype": _ARGUMENT,
            "moe_backend": "a backend the device forced",
            "dtype": "a dtype the checkpoint forced",
            "quantization": "a quantization the checkpoint forced",
        },
        "evidence": {
            "never_reproduced": _ARGUMENT,
            "gate_form": _ARGUMENT,
        },
    }
    return doc


# `oracle.json` exactly as af8170154 shaped it: seven top-level keys and no
# provenance at all. Reproduced here so the contract is pinned against the
# artifact that motivated it rather than against a paraphrase of it.
AF8170154_SHAPE = {
    "vllm": "0.23.1rc1.dev1511+g555967922",
    "transformers": "5.14.1",
    "flashinfer": "0.6.15.post1",
    "model": "/mnt/nas_share/checkpoints/nemotron-3.5-lightning-30b-nvfp4",
    "revision": "29f2d1746d8f41e316523194b19018707749b1b1",
    "sampling": {"temperature": 0.0, "max_tokens": 4},
    "golden": [_entry("The capital of France is", [1, 2, 3], [9, 8, 7, 6])],
}


class RequiredKeysTests(unittest.TestCase):
    def test_the_required_engine_keys_are_the_ones_this_suite_owns(self) -> None:
        self.assertEqual(set(capture.REQUIRED_ENGINE_KEYS), EXPECTED_ENGINE_KEYS)

    def test_every_knob_926_named_is_required(self) -> None:
        for knob in KNOBS_926_NAMED:
            self.assertIn(knob, capture.REQUIRED_ENGINE_KEYS, knob)

    def test_the_cpp_consumer_names_every_required_key(self) -> None:
        # The C++ gate that reads this golden carries its own copy of the key
        # list, because a test cannot import a Python constant. Two copies can
        # drift; this is the case that refuses to let them. The expectation is
        # owned by this suite, so a key deleted from BOTH copies is still red.
        source = CPP_CONSUMER.read_text(encoding="utf-8")
        for key in sorted(EXPECTED_ENGINE_KEYS):
            self.assertIn(f'"{key}"', source,
                          f"{CPP_CONSUMER.name} does not name '{key}'")

    def test_the_cpp_consumer_names_every_unattributed_key(self) -> None:
        # The same cross-gate as above, for the unattributed arm. A SUBSTRING
        # grep would pass vacuously here -- "dtype" and "quantization" already
        # appear in the C++ engine-key list -- so the two initializer lists are
        # PARSED out of the source and compared as sets. The expectation is this
        # suite's, so a key deleted from both production copies is still red.
        source = CPP_CONSUMER.read_text(encoding="utf-8")
        self.assertEqual(_cpp_string_list(source, "kForcedTermKeys"),
                         EXPECTED_FORCED_TERM_KEYS)
        self.assertEqual(_cpp_string_list(source, "kEvidenceKeys"),
                         EXPECTED_EVIDENCE_KEYS)

    def test_the_cpp_consumer_carries_the_same_argument_floor(self) -> None:
        source = CPP_CONSUMER.read_text(encoding="utf-8")
        match = re.search(r"kMinArgumentChars\s*=\s*(\d+)", source)
        self.assertIsNotNone(match, "the C++ consumer names no argument floor")
        self.assertEqual(int(match.group(1)), EXPECTED_MIN_ARGUMENT_CHARS)

    def test_the_prompt_battery_is_the_committed_one(self) -> None:
        shipped = json.loads(SHIPPED_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual([e["prompt"] for e in shipped["golden"]], list(capture.PROMPTS))


class ContractTests(unittest.TestCase):
    def test_the_shipped_golden_satisfies_the_contract(self) -> None:
        shipped = json.loads(SHIPPED_GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(capture.check_golden(shipped), [])

    def test_the_shipped_golden_is_not_silently_attributed(self) -> None:
        # It records no engine configuration, and the file has to say so. If a
        # later capture attributes it, this case is the one that must be
        # updated deliberately rather than a green that drifted.
        shipped = json.loads(SHIPPED_GOLDEN.read_text(encoding="utf-8"))
        recorded = shipped["capture"]["engine_config_recorded"]
        self.assertIsInstance(recorded, bool)
        if not recorded:
            self.assertTrue(shipped["capture"]["unrecoverable_reason"])
            self.assertTrue(shipped["capture"]["issue"].endswith("/926"))

    def test_the_af8170154_shape_is_refused(self) -> None:
        problems = capture.check_golden(AF8170154_SHAPE)
        self.assertTrue(any("capture" in p for p in problems), problems)

    def test_an_attributed_golden_holds(self) -> None:
        self.assertEqual(capture.check_golden(attributed()), [])

    def test_an_unattributed_golden_that_says_so_holds(self) -> None:
        self.assertEqual(capture.check_golden(unattributed()), [])

    def test_an_unattributed_golden_with_no_reason_is_refused(self) -> None:
        doc = unattributed()
        doc["capture"]["unrecoverable_reason"] = ""
        self.assertTrue(capture.check_golden(doc))

    def test_an_unattributed_golden_with_no_issue_is_refused(self) -> None:
        doc = unattributed()
        doc["capture"]["issue"] = "ask mudler"
        self.assertTrue(capture.check_golden(doc))

    def test_unrecorded_and_an_engine_block_cannot_both_be_true(self) -> None:
        doc = unattributed()
        doc["capture"]["engine"] = {"resolved": dict(RESOLVED)}
        self.assertTrue(capture.check_golden(doc))

    def test_a_missing_recorded_flag_is_refused(self) -> None:
        doc = attributed()
        del doc["capture"]["engine_config_recorded"]
        self.assertTrue(capture.check_golden(doc))

    def test_each_required_engine_key_is_load_bearing(self) -> None:
        for key in sorted(EXPECTED_ENGINE_KEYS):
            doc = attributed()
            del doc["capture"]["engine"]["resolved"][key]
            problems = capture.check_golden(doc)
            self.assertTrue(any(key in p for p in problems),
                            f"dropping '{key}' left the contract green: {problems}")

    def test_a_null_value_is_not_a_recorded_value(self) -> None:
        doc = attributed()
        doc["capture"]["engine"]["resolved"]["kv_cache_dtype"] = None
        self.assertTrue(capture.check_golden(doc))

    def test_a_null_block_override_is_a_legitimate_value(self) -> None:
        doc = attributed()
        doc["capture"]["engine"]["resolved"]["num_gpu_blocks_override"] = None
        self.assertEqual(capture.check_golden(doc), [])

    def test_one_leg_cannot_claim_determinism(self) -> None:
        doc = attributed()
        doc["capture"]["legs"] = 1
        self.assertTrue(capture.check_golden(doc))

    def test_disagreeing_legs_are_refused(self) -> None:
        doc = attributed()
        doc["capture"]["legs_agree"] = False
        self.assertTrue(capture.check_golden(doc))

    def test_the_batch_shape_is_required(self) -> None:
        doc = attributed()
        doc["capture"]["batch"] = {"prompts": 3}
        self.assertTrue(capture.check_golden(doc))

    def test_an_empty_golden_is_refused(self) -> None:
        doc = attributed()
        doc["golden"] = []
        self.assertTrue(capture.check_golden(doc))

    def test_a_short_token_row_is_refused(self) -> None:
        doc = attributed()
        doc["golden"][0]["token_ids"] = [9, 8]
        problems = capture.check_golden(doc)
        self.assertTrue(any("max_tokens" in p for p in problems), problems)

    def test_an_empty_prompt_tokenization_is_refused(self) -> None:
        doc = attributed()
        doc["golden"][0]["prompt_token_ids"] = []
        self.assertTrue(capture.check_golden(doc))

    def test_a_wrong_schema_is_refused(self) -> None:
        doc = attributed()
        doc["capture"]["schema"] = 1
        self.assertTrue(capture.check_golden(doc))


class UnattributedSubstanceTests(unittest.TestCase):
    """The unattributed arm's record has to still BE there, not merely fit.

    The reviewer's mutation that motivated every case below: gut `evidence`,
    `forced_by_checkpoint_or_device` and `captured_utc_is`, and set
    `unrecoverable_reason` to "dunno". The file keeps its shape, `--check` keeps
    printing `engine_config_recorded=False`, and every argument the artifact
    rests on is gone. That is silence wearing the shape of a record, which is
    the state #926 filed.
    """

    def test_the_forced_term_keys_are_the_ones_this_suite_owns(self) -> None:
        self.assertEqual(set(capture.REQUIRED_FORCED_TERM_KEYS),
                         EXPECTED_FORCED_TERM_KEYS)

    def test_the_evidence_keys_are_the_ones_this_suite_owns(self) -> None:
        self.assertEqual(set(capture.REQUIRED_EVIDENCE_KEYS), EXPECTED_EVIDENCE_KEYS)

    def test_the_argument_fields_and_their_floor_are_the_ones_this_suite_owns(self) -> None:
        self.assertEqual({tuple(f) for f in capture.ARGUMENT_FIELDS},
                         EXPECTED_ARGUMENT_FIELDS)
        self.assertEqual(capture.MIN_ARGUMENT_CHARS, EXPECTED_MIN_ARGUMENT_CHARS)

    def test_the_reviewers_gutting_mutation_is_refused(self) -> None:
        # Mutation E, reproduced verbatim as a case so it can never go green
        # again silently.
        doc = unattributed()
        doc["capture"]["unrecoverable_reason"] = "dunno"
        doc["capture"]["captured_utc_is"] = ""
        doc["capture"]["forced_by_checkpoint_or_device"] = {}
        doc["capture"]["evidence"] = {}
        problems = capture.check_golden(doc)
        for named in ("unrecoverable_reason", "captured_utc_is",
                      "forced_by_checkpoint_or_device", "evidence"):
            self.assertTrue(any(named in p for p in problems),
                            f"gutting '{named}' left the contract silent: {problems}")

    def test_each_forced_term_is_load_bearing(self) -> None:
        for key in sorted(EXPECTED_FORCED_TERM_KEYS):
            doc = unattributed()
            del doc["capture"]["forced_by_checkpoint_or_device"][key]
            problems = capture.check_golden(doc)
            self.assertTrue(any(key in p for p in problems),
                            f"dropping '{key}' left the contract green: {problems}")

    def test_each_evidence_key_is_load_bearing(self) -> None:
        for key in sorted(EXPECTED_EVIDENCE_KEYS):
            doc = unattributed()
            del doc["capture"]["evidence"][key]
            problems = capture.check_golden(doc)
            self.assertTrue(any(key in p for p in problems),
                            f"dropping '{key}' left the contract green: {problems}")

    def test_a_missing_forced_block_is_refused(self) -> None:
        doc = unattributed()
        del doc["capture"]["forced_by_checkpoint_or_device"]
        self.assertTrue(capture.check_golden(doc))

    def test_a_missing_evidence_block_is_refused(self) -> None:
        doc = unattributed()
        del doc["capture"]["evidence"]
        self.assertTrue(capture.check_golden(doc))

    def test_a_missing_timestamp_gloss_is_refused(self) -> None:
        # A commit's author date read as a capture time is a fabricated
        # provenance, so the file has to say which one it is.
        doc = unattributed()
        del doc["capture"]["captured_utc_is"]
        self.assertTrue(capture.check_golden(doc))

    def test_each_argument_field_refuses_a_one_word_answer(self) -> None:
        for path in sorted(EXPECTED_ARGUMENT_FIELDS):
            doc = unattributed()
            block = doc["capture"]
            for step in path[:-1]:
                block = block[step]
            block[path[-1]] = "dunno"
            problems = capture.check_golden(doc)
            self.assertTrue(any(".".join(path) in p for p in problems),
                            f"'{'.'.join(path)}' accepted one word: {problems}")

    def test_a_non_string_reason_is_refused(self) -> None:
        # The C++ copy has always spelled this `is_string()`. A bare truthiness
        # test in the checker let a number through, so the two copies of one
        # contract disagreed about what satisfied it.
        doc = unattributed()
        doc["capture"]["unrecoverable_reason"] = 123
        problems = capture.check_golden(doc)
        self.assertTrue(any("unrecoverable_reason" in p for p in problems), problems)

    def test_the_floor_is_not_met_by_whitespace(self) -> None:
        doc = unattributed()
        doc["capture"]["evidence"]["gate_form"] = " " * 200
        self.assertTrue(capture.check_golden(doc))

    def test_an_argument_at_the_floor_is_accepted(self) -> None:
        # The other side of the floor, so the case cannot pass by refusing
        # everything. A gate that fires on ordinary work is the defect.
        doc = unattributed()
        doc["capture"]["evidence"]["gate_form"] = "x" * EXPECTED_MIN_ARGUMENT_CHARS
        self.assertEqual(capture.check_golden(doc), [])

    def test_the_attributed_arm_is_not_burdened_by_these_keys(self) -> None:
        # `engine.resolved` already records kv_cache_dtype, dtype, quantization
        # and moe_backend as VALUES, so an attributed golden owes no prose about
        # them. Scoping matters: a requirement that fired on both arms would red
        # every future capture this generator writes.
        doc = attributed()
        for key in ("captured_utc_is", "forced_by_checkpoint_or_device", "evidence"):
            self.assertNotIn(key, doc["capture"])
        self.assertEqual(capture.check_golden(doc), [])


class CliTests(unittest.TestCase):
    def _check(self, doc):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oracle.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = capture.main(["--check", str(path)])
            return code, out.getvalue() + err.getvalue()

    def test_check_exits_zero_on_an_attributed_golden(self) -> None:
        code, text = self._check(attributed())
        self.assertEqual(code, 0, text)
        self.assertIn("0 problem(s)", text)

    def test_check_exits_one_on_the_af8170154_shape(self) -> None:
        code, text = self._check(AF8170154_SHAPE)
        self.assertEqual(code, 1, text)
        self.assertIn("CONTRACT:", text)

    def test_check_reads_the_shipped_golden(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = capture.main(["--check", str(SHIPPED_GOLDEN)])
        self.assertEqual(code, 0, out.getvalue() + err.getvalue())
        # The count is stated, because a validator that examined nothing and a
        # validator that examined everything print the same exit code.
        self.assertIn("3 golden entries", out.getvalue())


if __name__ == "__main__":
    unittest.main()
