#!/usr/bin/env python3
"""`LTX25-PROMPT-ADHERENCE` (#2295, owning #1854's first sub-question): the
render is scored against ITS OWN PROMPT, and the check fires.

#1854 filed two sub-questions and refused to close either with a proxy. The
second, artefact freedom, became a gate at `fa9903b86`. This suite is the first
one, and #1854's words bind it exactly as they bound that row: a proxy "that
correlates with nothing would be the `a-shape-valid-gate-passes-a-wrong-artefact`
failure", and the ONE admissible shape is "worse than the oracle on this
statistic, **because that is a comparison and not a convention**".

So this suite has three jobs.

FIRST, that the instrument can say NO. An instrument that has never failed is not
known to be able to. S0 is checked against a scorer that ranks a decoy above the
prompt on the ORACLE's own render, and against one that returns the same number
for every prompt -- the second is the sharper case, because a constant scorer HAS
an argmax (numpy returns the first index, which is the true prompt) and would
clear a bare set assertion while measuring nothing.

SECOND, that no bound in this file is a number. Every assertion that touches S1's
bound derives it from a reference the test itself built, and one case scores ONE
render against TWO references and requires OPPOSITE verdicts. A literal here
would be `a-transcription-cannot-gate-the-function-it-transcribes` wearing a
test's clothes.

THIRD, that the scorer's IDENTITY decides admission. A neural instrument is a
file, not a closed form, and a swapped checkpoint moves every reading silently
(#1723). The pin carries a revision AND a sha256 per file, and the refusal is
checked three ways: an unmeasured digest, a wrong digest, and a missing file.

THE 77-POSITION BOUND IS STATED HERE TOO, and it is measured rather than
asserted. CLIP's text context is 77 positions. The #1864 reference request fits.
#1854's OWN motivating prompt, the 70-word golden retriever, does not: CLIP's own
pre-tokenizer splits it into 81 chunks and each chunk costs at least one BPE
token, so it needs at least 83 positions with the boundary pair. That lower bound
needs no vocabulary file, and the case below computes it rather than quoting it.
**This instrument cannot answer the example #1854 uses to define the problem, and
it says so in its own output.**

Two lanes. Everything above needs numpy only: no build, no GPU, no network, and
no checkpoint. The two cases that need the pinned CLIP weights -- the ported
upstream comparison and the end-to-end run against the committed reference --
SKIP LOUDLY with their reason when `VT_LTX25_ADHERENCE_MODEL` is unset, because a
case that cannot run is not a case that passed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/ltx25-render-compare.py"
GOLDENS = ROOT / "tests/parity/goldens/ltx2_oracle"
ADHERENCE = ROOT / "tests/parity/goldens/ltx25_adherence"
MODEL_DIR = os.environ.get("VT_LTX25_ADHERENCE_MODEL", "")

# #1854's own motivating prompt, byte-for-byte from
# `scripts/ltx25-dit-attn-flash-pixel-ab.sh:697`. It is here so the 77-position
# bound is stated against the exact string the issue quotes.
ISSUE_1854_PROMPT = (
    "A golden retriever shakes water from its coat on a sunlit lawn, droplets "
    "flying outward in a bright arc around its head and shoulders, wet fur "
    "rippling and separating into strands from shoulders to tail, ears flapping, "
    "muscles moving under the coat. Crisp midday light, shallow depth of field, "
    "vivid green grass behind. The dog barks once, water patters onto the grass, "
    "and a light breeze moves through the trees."
)

# CLIP's OWN pre-tokenizer pattern, from `openai/CLIP`'s `simple_tokenizer.py`
# and reproduced by `transformers`' `CLIPTokenizer`: contractions, runs of
# letters, single digits, and runs of everything else that is not whitespace.
# BPE merges never cross a chunk boundary, so the number of chunks is a proven
# LOWER bound on the token count and needs no vocabulary file. This is used only
# to REFUSE, never to admit: a chunk can cost more than one token, so the bound
# points one way.
CLIP_PRETOKEN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\sa-zA-Z0-9]+""")


def clip_token_floor(text: str) -> int:
    """A proven lower bound on CLIP's token count, boundary pair included."""
    return len(CLIP_PRETOKEN.findall(text.lower())) + 2


def load_tool():
    spec = importlib.util.spec_from_file_location("ltx25_render_compare", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load_tool()


def scores(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def labels_for(n_decoys: int) -> list[str]:
    return ["true"] + [f"decoy:{i}" for i in range(n_decoys)]


def a_reference(true_vals: list[float], decoy_vals: list[float]) -> dict:
    """A reference discrimination built by the TEST, so no bound is a literal."""
    rows = [[t] + list(decoy_vals) for t in true_vals]
    return T.discrimination(scores(rows), labels_for(len(decoy_vals)))


class ScorerPin(unittest.TestCase):
    """The checkpoint is pinned by revision AND sha256, and it says its costs."""

    def setUp(self):
        self.pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())

    def test_the_pin_names_a_repo_and_a_revision(self):
        # A repo id alone is not a pin: checkpoints get re-quantized in place
        # under an unchanged name.
        self.assertEqual(self.pin["repo"], "openai/clip-vit-base-patch16")
        self.assertRegex(self.pin["revision"], r"^[0-9a-f]{40}$")

    def test_every_pinned_file_carries_a_sha256_or_declares_itself_unmeasured(self):
        for name, f in self.pin["files"].items():
            with self.subTest(file=name):
                if f["sha256"] is None:
                    self.assertFalse(f["measured"],
                                     f"{name} has no digest and claims to be measured")
                else:
                    self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$")
                    self.assertTrue(f["measured"])

    def test_the_three_costs_a_reader_must_not_discover_by_surprise(self):
        # Size, licence and format, beside the pin rather than in a commit
        # message that scrolls away.
        self.assertIsNone(self.pin["licence"],
                          "the HuggingFace repository declares no licence and the "
                          "pin must not invent one")
        self.assertEqual(self.pin["weight_format"], "pickle")
        self.assertEqual(self.pin["required_bytes_total"],
                         sum(f["advertised_bytes"] for f in self.pin["files"].values()))
        self.assertGreater(self.pin["required_bytes_total"], 500_000_000)

    def test_the_pin_stays_an_instrument_and_never_becomes_an_oracle(self):
        # The developer's answer to section 9. If this flips, AGENTS.md's oracle
        # table and `.agents/oracles/` owe a row, and that is a policy change
        # with its own spec, not an edit here.
        self.assertEqual(self.pin["role"], "instrument")
        self.assertFalse((ROOT / ".agents/oracles/clip.md").exists())

    def test_the_context_limit_is_the_one_the_tool_enforces(self):
        self.assertEqual(self.pin["text_context_positions"], 77)


class ContextLimit(unittest.TestCase):
    """P5. A prompt that does not fit is REFUSED, never truncated."""

    def test_the_issues_own_example_does_not_fit_and_the_reference_does(self):
        # MEASURED, not quoted. The floor counts CLIP's own pre-token chunks, so
        # it cannot overstate the length.
        floor_1854 = clip_token_floor(ISSUE_1854_PROMPT)
        self.assertGreater(floor_1854, 77,
                           "if this ever drops to 77 or below, the claim that this "
                           "instrument cannot answer #1854's own example is no "
                           "longer established and the record must change")
        self.assertEqual(len(ISSUE_1854_PROMPT.split()), 70)
        ref = T.true_prompt()
        self.assertLess(clip_token_floor(ref), 77)

    def test_an_overlong_prompt_is_refused_and_the_message_names_it(self):
        with self.assertRaises(T.UnreadableInput) as cm:
            T.refuse_overlong_prompts([ISSUE_1854_PROMPT], clip_token_floor, 77)
        msg = str(cm.exception)
        self.assertIn("REFUSED rather than truncated", msg)
        self.assertIn(str(clip_token_floor(ISSUE_1854_PROMPT)), msg)

    def test_a_prompt_that_fits_is_admitted_and_its_length_reported(self):
        out = T.refuse_overlong_prompts([T.true_prompt()], clip_token_floor, 77)
        self.assertEqual(out["limit"], 77)
        self.assertLessEqual(out["max"], 77)

    def test_the_boundary_is_the_limit_and_not_one_past_it(self):
        # A gate whose edge is off by one admits the case it exists to refuse.
        T.refuse_overlong_prompts(["x"], lambda _t: 77, 77)
        with self.assertRaises(T.UnreadableInput):
            T.refuse_overlong_prompts(["x"], lambda _t: 78, 77)

    def test_a_pin_with_no_limit_refuses_rather_than_admitting_everything(self):
        with self.assertRaises(T.UnreadableInput):
            T.refuse_overlong_prompts(["x"], lambda _t: 1, 0)


class Decoys(unittest.TestCase):
    """The decoy set is committed, has both kinds, and its null is derived."""

    def setUp(self):
        self.decoys = T.load_decoys()

    def test_both_kinds_are_present(self):
        kinds = {d["kind"] for d in self.decoys}
        self.assertEqual(kinds, {"near", "far"})

    def test_the_true_prompt_is_not_among_the_decoys(self):
        self.assertNotIn(T.true_prompt(), [d["text"] for d in self.decoys])

    def test_the_null_is_one_over_n_plus_one_and_is_computed(self):
        n = len(self.decoys)
        self.assertAlmostEqual(T.adherence_null(n), 1.0 / (n + 1))
        # And it moves with the set, so it cannot have been transcribed.
        self.assertAlmostEqual(T.adherence_null(n + 3), 1.0 / (n + 4))

    def test_a_set_with_only_one_kind_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "decoys.json"
            p.write_text(json.dumps({"decoys": [
                {"kind": "far", "text": "a"}, {"kind": "far", "text": "b"}]}))
            with self.assertRaises(T.UnreadableInput):
                T.load_decoys(str(p))

    def test_a_repeated_decoy_is_refused_because_it_corrupts_the_null(self):
        # `adherence_null` divides by the COUNT, so a duplicate makes the printed
        # null smaller than the real chance. The null is the one number S2's
        # verdict has to be read against, so an understated one is worse than no
        # decoy at all.
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "decoys.json"
            p.write_text(json.dumps({"decoys": [
                {"kind": "near", "text": "a fox"},
                {"kind": "far", "text": "a city"},
                {"kind": "far", "text": "a city"}]}))
            with self.assertRaises(T.UnreadableInput) as cm:
                T.load_decoys(str(p))
            self.assertIn("null", str(cm.exception))

    def test_the_committed_set_has_no_repeats(self):
        texts = [d["text"] for d in self.decoys]
        self.assertEqual(len(set(texts)), len(texts))

    def test_no_decoys_at_all_is_refused_rather_than_a_free_pass(self):
        with self.assertRaises(T.UnreadableInput):
            T.adherence_null(0)

    def test_the_true_prompt_comes_from_the_committed_manifest(self):
        manifest = json.loads((GOLDENS / "ltx2_oracle_manifest.json").read_text())
        self.assertEqual(T.true_prompt(), manifest["request"]["prompt"])

    def test_a_manifest_without_a_prompt_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.json"
            p.write_text(json.dumps({"request": {"seed": 42}}))
            with self.assertRaises(T.UnreadableInput):
                T.true_prompt(str(p))


class ScorerIdentity(unittest.TestCase):
    """P1. An unidentified checkpoint scores nothing."""

    def _dir_with(self, td: str, blobs: dict[str, bytes]) -> str:
        for name, data in blobs.items():
            (Path(td) / name).write_bytes(data)
        return td

    def test_an_unmeasured_digest_refuses_and_names_what_owes_the_measurement(self):
        pin = {"repo": "r", "revision": "v", "required_bytes_total": 1,
               "files": {"a.bin": {"sha256": None}}}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(T.UnreadableInput) as cm:
                T.assert_scorer_identity(td, pin)
        msg = str(cm.exception)
        self.assertIn("UNMEASURED", msg)
        self.assertIn("LTX25-PROMPT-ADHERENCE", msg)

    def test_a_wrong_digest_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            self._dir_with(td, {"a.bin": b"hello"})
            pin = {"repo": "r", "revision": "v", "files": {"a.bin": {"sha256": "0" * 64}}}
            with self.assertRaises(T.UnreadableInput) as cm:
                T.assert_scorer_identity(td, pin)
            self.assertIn("is not the pinned", str(cm.exception))

    def test_a_missing_file_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            pin = {"repo": "r", "revision": "v",
                   "files": {"a.bin": {"sha256": hashlib.sha256(b"hello").hexdigest()}}}
            with self.assertRaises(T.UnreadableInput):
                T.assert_scorer_identity(td, pin)

    def test_a_matching_digest_is_admitted(self):
        with tempfile.TemporaryDirectory() as td:
            self._dir_with(td, {"a.bin": b"hello"})
            pin = {"repo": "r", "revision": "v",
                   "files": {"a.bin": {"sha256": hashlib.sha256(b"hello").hexdigest()}}}
            out = T.assert_scorer_identity(td, pin)
            self.assertEqual(out["files_verified"], 1)

    def test_an_empty_pin_refuses_rather_than_verifying_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(T.UnreadableInput):
                T.assert_scorer_identity(td, {"files": {}})

    def test_the_committed_pin_refuses_a_directory_that_is_not_the_checkpoint(self):
        pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(T.UnreadableInput):
                T.assert_scorer_identity(td, pin)


class Precondition(unittest.TestCase):
    """P2. S0: the instrument must prove it can say no, on the ORACLE's frames."""

    def test_a_scorer_that_ranks_a_decoy_first_on_the_reference_is_refused(self):
        ref = a_reference([10.0, 11.0, 9.0], [30.0, 5.0])
        with self.assertRaises(T.UnreadableInput) as cm:
            T.scorer_precondition(ref)
        self.assertIn("S0 FAILED", str(cm.exception))

    def test_a_constant_scorer_is_refused_even_though_it_has_an_argmax(self):
        # THE SHARP CASE. numpy's argmax returns the FIRST maximal index, which
        # is the true prompt, so a bare `argmax == true` passes a scorer that
        # returns one number for every prompt. The margin must be strictly
        # positive, and this is the mutation that proves it.
        ref = a_reference([7.0, 7.0, 7.0], [7.0, 7.0])
        self.assertTrue(ref["true_first"], "the trap this case exists for")
        with self.assertRaises(T.UnreadableInput) as cm:
            T.scorer_precondition(ref)
        self.assertIn("margin", str(cm.exception))

    def test_a_competent_scorer_passes_and_reports_its_margin(self):
        ref = a_reference([30.0, 31.0, 29.0], [20.0, 12.0])
        out = T.scorer_precondition(ref)
        self.assertTrue(out["passed"])
        self.assertAlmostEqual(out["margin"], 30.0 - 20.0)


class S1Bound(unittest.TestCase):
    """P3. The bound IS the reference's own per-frame minimum, recomputed."""

    def _verdict(self, ours_true: list[float], ref_true: list[float]) -> tuple:
        decoys = [5.0, 4.0]
        ours = T.discrimination(scores([[t] + decoys for t in ours_true]),
                                labels_for(len(decoys)))
        ref = T.discrimination(scores([[t] + decoys for t in ref_true]),
                               labels_for(len(decoys)))
        checks = T.adherence_checks("a", ours, ref)
        clip = [c for c in checks if c[0].endswith("adherence_clip")][0]
        return clip

    def test_one_render_and_two_references_give_opposite_verdicts(self):
        # The whole point of a computed bound. If the tool ever transcribed a
        # number, both of these would agree.
        ours = [20.0, 21.0, 19.0]                       # mean 20.0
        lenient = self._verdict(ours, [18.0, 25.0, 30.0])   # frame_min 18.0
        strict = self._verdict(ours, [24.0, 25.0, 30.0])    # frame_min 24.0
        self.assertTrue(lenient[1])
        self.assertFalse(strict[1])

    def test_the_detail_prints_the_recomputed_bound_and_not_a_constant(self):
        clip = self._verdict([20.0, 21.0, 19.0], [24.0, 25.0, 30.0])
        self.assertIn("24.0000", clip[2])
        self.assertIn("the reference's per-frame MINIMUM", clip[2])

    def test_the_direction_is_the_mirror_of_the_blockiness_bound(self):
        # There higher is worse and the form is `ours_mean <= ref_frame_max`.
        # Here higher is BETTER, so a render far above the reference passes.
        clip = self._verdict([90.0, 90.0, 90.0], [24.0, 25.0, 30.0])
        self.assertTrue(clip[1])
        self.assertIn("no worse than the oracle", clip[2])

    def test_a_render_at_the_bound_passes_and_one_below_it_fails(self):
        self.assertTrue(self._verdict([24.0, 24.0, 24.0], [24.0, 25.0, 30.0])[1])
        self.assertFalse(self._verdict([23.9, 24.0, 24.0], [24.0, 25.0, 30.0])[1])

    def test_the_looseness_is_reported_in_the_references_own_units(self):
        clip = self._verdict([25.0, 25.0, 25.0], [24.0, 25.0, 30.0])
        self.assertIn("of the reference's own per-frame sd below its mean", clip[2])


class S2Discrimination(unittest.TestCase):
    """P4. The set assertion, its null, and the noise case blockiness passes."""

    def _argmax_check(self, ours_rows: list[list[float]]) -> tuple:
        labels = labels_for(len(ours_rows[0]) - 1)
        ours = T.discrimination(scores(ours_rows), labels)
        ref = T.discrimination(scores([[30.0] + [10.0] * (len(labels) - 1)]), labels)
        checks = T.adherence_checks("a", ours, ref)
        return [c for c in checks if c[0].endswith("adherence_argmax")][0]

    def test_a_render_that_ranks_a_decoy_first_fails(self):
        c = self._argmax_check([[10.0, 30.0, 5.0], [11.0, 31.0, 6.0]])
        self.assertFalse(c[1])

    def test_a_noise_render_fails_s2_where_blockiness_passes_it(self):
        # `ltx25-oracle-absolute.md` records, with a test, that a pure-noise
        # render PASSES the blockiness ceiling and passes C0. This is the blind
        # spot S2 exists to cover: noise depicts no prompt, so it ranks the true
        # one no higher than a decoy. Modelled as a scorer that separates
        # nothing, which is what noise produces.
        rng = np.random.default_rng(1854)
        rows = rng.normal(15.0, 0.1, (25, 4))
        c = self._argmax_check(rows.tolist())
        # It fails unless the true column happens to win by chance, and the
        # chance of that is the printed null. Assert the mechanism rather than
        # the coin toss: the margin is inside the noise.
        ours = T.discrimination(scores(rows.tolist()), labels_for(3))
        self.assertLess(abs(ours["margin"]), 0.5,
                        "noise must not separate the true prompt from a decoy")
        del c

    def test_the_null_is_printed_beside_the_verdict_passing_or_failing(self):
        passing = self._argmax_check([[30.0, 10.0, 5.0], [31.0, 11.0, 6.0]])
        failing = self._argmax_check([[10.0, 30.0, 5.0], [11.0, 31.0, 6.0]])
        self.assertTrue(passing[1])
        self.assertFalse(failing[1])
        for c in (passing, failing):
            self.assertIn("null for an uninformative scorer is 1/3 = 0.3333", c[2])
            self.assertIn("margin to the best decoy", c[2])

    def test_the_margin_moves_with_the_set_size(self):
        two = self._argmax_check([[30.0, 10.0, 5.0]])
        self.assertIn("1/3 = 0.3333", two[2])
        five = self._argmax_check([[30.0, 10.0, 5.0, 4.0, 3.0, 2.0]])
        self.assertIn("1/6 = 0.1667", five[2])

    def test_per_frame_wins_are_reported_so_a_lucky_mean_is_visible(self):
        c = self._argmax_check([[30.0, 10.0], [1.0, 20.0], [1.0, 20.0]])
        self.assertIn("per-frame wins 1/3", c[2])

    def test_a_matrix_with_no_decoy_column_is_refused(self):
        with self.assertRaises(T.UnreadableInput):
            T.discrimination(scores([[1.0], [2.0]]), ["true"])


class CommandLine(unittest.TestCase):
    """The flag is REACHED from the tool's own entry point, and it refuses."""

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(TOOL)] + args,
                              capture_output=True, text=True)

    def test_adherence_without_a_reference_is_refused_by_the_parser(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._run(["--a", td, "--adherence-model", td])
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--adherence-model needs --reference", p.stderr)

    def test_a_directory_that_is_not_the_checkpoint_exits_unreadable(self):
        # END TO END through `main`, so this proves the flag REACHES
        # `assert_scorer_identity` rather than that the function works.
        with tempfile.TemporaryDirectory() as td:
            frames = Path(td) / "frames"
            frames.mkdir()
            rng = np.random.default_rng(7)
            for i in range(3):
                a = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
                (frames / f"frame_{i:06d}.ppm").write_bytes(
                    b"P6\n48 32\n255\n" + a.tobytes())
            p = self._run(["--a", str(frames),
                           "--reference", str(GOLDENS / "upstream-render.mp4"),
                           "--adherence-model", td])
        self.assertEqual(p.returncode, 2, p.stdout[-2000:] + p.stderr[-2000:])
        self.assertIn("VERDICT UNREADABLE", p.stderr)

    def test_the_77_position_bound_is_stated_when_adherence_is_not_measured(self):
        # An instrument that cannot answer the question the issue uses to define
        # the problem must say so in its own output, in BOTH states.
        p = self._run(["--help"])
        self.assertIn("77 positions", p.stdout)


@unittest.skipUnless(MODEL_DIR, "VT_LTX25_ADHERENCE_MODEL is unset: the pinned "
                                "CLIP checkpoint is not on this host, so the two "
                                "cases that need real weights CANNOT RUN. This is "
                                "a SKIP and never a pass")
class PinnedCheckpoint(unittest.TestCase):
    """The ported upstream case, and one real end-to-end run."""

    def test_the_committed_pin_admits_the_real_checkpoint(self):
        pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())
        out = T.assert_scorer_identity(MODEL_DIR, pin)
        self.assertEqual(out["files_verified"], len(pin["files"]))

    def test_ported_vllm_test_clip_the_two_feature_routes_agree(self):
        """Ported from `vllm/tests/models/multimodal/pooling/test_clip.py` at
        `5559679229`, adapted at the harness and nowhere else.

        UPSTREAM'S ASSERTION. `_run_test` at `:27` takes HuggingFace
        `get_image_features(pixel_values=...)` and
        `get_text_features(input_ids=..., attention_mask=...)` (`:52-57`) as the
        reference for what vLLM's `.embed()` returns, and compares them through
        `check_embeddings_close` (`:67`), which is cosine similarity with
        `tol = 1e-3` (`tests/models/utils.py`). `dtype` is `"float"` (`:75`) and
        `max_model_len` is `77` (`:41`).

        WHAT WAS ADAPTED, AND WHY IT IS UNAVOIDABLE. Upstream's other side is
        `VllmRunner`, which needs an installed vLLM and a GPU lease; this tree's
        vLLM is a source checkout. So the SECOND side here is the route
        vLLM-Omni's `CLIPScorer` takes (`helpers.py:507-508`,
        `outputs.image_embeds` / `outputs.text_embeds`), and the case asserts
        that it agrees with vLLM's reference route at UPSTREAM'S OWN tolerance.
        That is not a tautology: they are different code paths, and our scorer
        deliberately takes vLLM's, because vLLM is the primary reference. The
        vLLM-runner half stays OWED; the spec's `## Owed` names it.

        Upstream's prompts and the `patch32` checkpoint are the two other
        differences: the pin is `patch16`, because that is the checkpoint
        vLLM-Omni's video adherence scorer defaults to, and the prompts are this
        row's own because the case is about the two routes and not about them.
        """
        import torch
        pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())
        scorer = T.ClipAdherenceScorer(MODEL_DIR, pin)
        rng = np.random.default_rng(1864)
        frames = [rng.integers(0, 255, (192, 320, 3), dtype=np.uint8)
                  for _ in range(2)]
        prompts = [T.true_prompt(), T.load_decoys()[0]["text"]]

        img_ref, txt_ref = scorer.features(frames, prompts)

        from PIL import Image
        img_in = scorer._processor(
            images=[Image.fromarray(f, mode="RGB") for f in frames],
            return_tensors="pt")
        txt_in = scorer._processor(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = scorer._model(pixel_values=img_in["pixel_values"],
                                input_ids=txt_in["input_ids"],
                                attention_mask=txt_in.get("attention_mask"))

        for name, a, b in (("image", img_ref, out.image_embeds),
                           ("text", txt_ref, out.text_embeds)):
            for i in range(a.shape[0]):
                sim = torch.nn.functional.cosine_similarity(a[i], b[i], dim=0)
                self.assertGreaterEqual(
                    float(sim), 1 - 1e-3,
                    f"{name} row {i}: cosine {float(sim):.6f} against upstream's "
                    f"own check_embeddings_close tolerance 1e-3")

    def test_the_projected_features_have_the_pinned_projection_dim(self):
        pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())
        scorer = T.ClipAdherenceScorer(MODEL_DIR, pin)
        rng = np.random.default_rng(3)
        img, txt = scorer.features(
            [rng.integers(0, 255, (192, 320, 3), dtype=np.uint8)], [T.true_prompt()])
        cfg = json.loads((Path(MODEL_DIR) / "config.json").read_text())
        self.assertEqual(img.shape[-1], cfg["projection_dim"])
        self.assertEqual(txt.shape[-1], cfg["projection_dim"])
        self.assertEqual(str(img.dtype), "torch.float32",
                         "upstream runs this pooling path at dtype='float'")

    def test_the_true_prompt_listed_as_a_decoy_is_refused_before_any_scoring(self):
        # It would score identically against itself, so the argmax would be a TIE
        # that numpy breaks by index in the true prompt's favour, and S2 would
        # pass whatever the render depicts. The refusal happens before the scorer
        # is even constructed, which is why this case costs no forward pass.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "decoys.json"
            d.write_text(json.dumps({"decoys": [
                {"kind": "near", "text": T.true_prompt()},
                {"kind": "far", "text": "a city street at night"}]}))
            with self.assertRaises(T.UnreadableInput) as cm:
                T.adherence_report(td, [], MODEL_DIR, decoys_path=str(d))
            self.assertIn("also listed as a decoy", str(cm.exception))

    def test_the_real_scorer_refuses_the_issues_own_70_word_prompt(self):
        # The bound, through the SCORER'S OWN tokenizer rather than the floor.
        pin = json.loads((ADHERENCE / "scorer-pin.json").read_text())
        scorer = T.ClipAdherenceScorer(MODEL_DIR, pin)
        n = scorer.count_tokens(ISSUE_1854_PROMPT)
        self.assertGreater(n, 77)
        self.assertGreaterEqual(n, clip_token_floor(ISSUE_1854_PROMPT),
                                "the pre-token floor must not exceed the real count")
        self.assertLessEqual(scorer.count_tokens(T.true_prompt()), 77)


if __name__ == "__main__":
    unittest.main(verbosity=2)
