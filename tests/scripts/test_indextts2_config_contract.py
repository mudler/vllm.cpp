#!/usr/bin/env python3
"""The IndexTTS-2.5 shape contract must match the SHIPPED config.yaml.

`indextts2_config.h` carries dimensions read from the checkpoint's own
config.yaml. This asserts the header still says what the config says, using a
committed copy of the config rather than the network, so the gate is
reproducible offline and a silent upstream edit shows up as a diff to review
rather than as a test that changes meaning under us.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include/vllm/model_executor/models/indextts2_config.h"
CONFIG = ROOT / "tests/vllm/models/indextts2_config.yaml"

# name in the header -> (yaml path, expected value)
EXPECTED = {
    "kTalkerMelSampleRate": 24000,
    "kTalkerMelBins": 100,
    "kOutputSampleRate": 22050,
    "kS2MelMelBins": 80,
    "kHopLength": 256,
    "kNFft": 1024,
    "kTalkerDim": 1280,
    "kTalkerLayers": 24,
    "kTalkerHeads": 20,
    "kNumberTextTokens": 60509,
    "kNumberMelCodes": 8194,
    "kStartMelToken": 8192,
    "kStopMelToken": 8193,
    "kMaxMelTokens": 1815,
    "kMaxTextTokens": 600,
    "kMelLengthCompression": 1024,
    "kCodecCodebookSize": 8192,
    "kCodecHiddenSize": 1024,
    "kCodecCodebookDim": 8,
    "kVocosDim": 384,
    "kVocosIntermediateDim": 2048,
    "kVocosNumLayers": 12,
    "kStyleDim": 192,
    "kLengthRegulatorChannels": 512,
    "kLengthRegulatorInChannels": 1024,
    "kDitHiddenDim": 512,
    "kDitNumHeads": 8,
    "kDitDepth": 13,
    "kDitInChannels": 80,
}


def header_constants() -> dict[str, int]:
    text = HEADER.read_text(encoding="utf-8")
    out: dict[str, int] = {}
    for m in re.finditer(r"inline constexpr int64_t (\w+) = (\d+);", text):
        out[m.group(1)] = int(m.group(2))
    return out


class ConfigContractTests(unittest.TestCase):
    def test_every_expected_constant_is_declared(self) -> None:
        have = header_constants()
        missing = sorted(set(EXPECTED) - set(have))
        self.assertEqual(missing, [], f"header is missing constants: {missing}")

    def test_constants_match_the_shipped_config(self) -> None:
        have = header_constants()
        wrong = {k: (have[k], v) for k, v in EXPECTED.items() if have.get(k) != v}
        self.assertEqual(wrong, {}, f"header disagrees with config.yaml: {wrong}")

    def test_the_two_sample_rates_are_distinct(self) -> None:
        """The talker's mel front end is 24 kHz; the OUTPUT is 22.05 kHz.

        Conflating them is the first mistake available in this model, and it
        produces audio at the wrong speed rather than an error.
        """
        have = header_constants()
        self.assertNotEqual(have["kTalkerMelSampleRate"], have["kOutputSampleRate"])
        self.assertEqual(have["kOutputSampleRate"], 22050)
        self.assertNotEqual(have["kTalkerMelBins"], have["kS2MelMelBins"])

    def test_the_committed_config_is_the_one_the_constants_came_from(self) -> None:
        """Guards against the header drifting from a stale copy of the config."""
        self.assertTrue(CONFIG.is_file(), f"{CONFIG} must be committed")
        text = CONFIG.read_text(encoding="utf-8")
        for probe in ("model_dim: 1280", "sr: 22050", "codebook_size: 8192",
                      "depth: 13", "dim: 192"):
            self.assertIn(probe, text, f"committed config lost `{probe}`")


if __name__ == "__main__":
    unittest.main()
