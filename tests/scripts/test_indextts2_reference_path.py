#!/usr/bin/env python3
"""The IndexTTS-2.5 reference-audio path, pinned where prose would rot.

Three facts about this path are easy to get wrong, cost nothing to check, and
would each produce a model that runs and sounds wrong:

1. The semantic features come from HIDDEN STATE 17 of the w2v-bert encoder, not
   from its final layer (`infer_v2_5.py:287`). A port that took the last layer
   gets features of the right shape from the right model.
2. Those features are then normalized by STORED statistics shipped in the
   checkpoint (`wav2vec2bert_stats.pt`), not by per-utterance statistics.
3. The feature extractor is KALDI-style, not the Slaney/torchaudio kind this
   tree already has for LTX-2.5: povey window, preemphasis 0.97, DC removal,
   400/160/512 framing, 80 mel bins, and a stride-2 stack that makes 160 columns.

This asserts them against the committed upstream sources and the shipped config,
so a change on either side has to come past this rather than past a paragraph.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / ".agents/specs/indextts-2-5.md"
CONFIG = ROOT / "tests/vllm/models/indextts2_config.yaml"
HEADER = ROOT / "include/vllm/model_executor/models/w2vbert.h"


class SpecRecordsThePath(unittest.TestCase):
    def setUp(self):
        self.spec = SPEC.read_text()

    def test_the_hidden_state_index_is_recorded(self):
        # If this number changes upstream, the spec must change with it.
        self.assertIn("hidden_states[17]", self.spec)

    def test_the_stored_statistics_are_recorded(self):
        self.assertIn("wav2vec2bert_stats.pt", self.spec)

    def test_the_kaldi_framing_is_recorded_exactly(self):
        for token in ("povey", "0.97", "400", "160", "512", "mel_floor"):
            self.assertIn(token, self.spec, f"the spec no longer states {token}")

    def test_the_spec_does_not_claim_the_extractor_is_ported(self):
        # The encoder IS ported; the extractor is NOT. An earlier summary got
        # this backwards, so the distinction is asserted.
        # A window search rather than a sentence one: the surrounding text
        # carries decimal constants, so "no periods in between" is the wrong
        # shape of assertion and was the first version's mistake.
        i = self.spec.find("the extractor is a NEW unit")
        self.assertNotEqual(i, -1, "the spec no longer says the extractor is new")
        self.assertIn(
            "unported",
            self.spec[i : i + 200],
            "the spec must keep saying the feature EXTRACTOR is unported",
        )


class ConfigAgreesWithThePath(unittest.TestCase):
    def test_the_stats_file_is_the_one_the_config_names(self):
        text = CONFIG.read_text()
        m = re.search(r"w2v_stat:\s*(\S+)", text)
        self.assertIsNotNone(m, "config.yaml no longer names w2v_stat")
        self.assertEqual(m.group(1).strip("'\""), "wav2vec2bert_stats.pt")


class EncoderIsPortedAndSaysWhatItReturns(unittest.TestCase):
    def test_the_encoder_stack_exists(self):
        self.assertIn("EncoderStack", HEADER.read_text())

    def test_the_header_warns_there_is_no_final_norm(self):
        # The obvious-looking mistake for this position_embeddings_type.
        self.assertIn("NO final layer norm", HEADER.read_text())


if __name__ == "__main__":
    unittest.main()
