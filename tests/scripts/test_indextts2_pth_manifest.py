#!/usr/bin/env python3
"""The IndexTTS-2.5 shape contract, cross-checked against REAL tensor shapes.

`indextts2_config.h` was read from the checkpoint's `config.yaml`, and
`test_indextts2_config_contract.py` pins the header against that file. Both
sides of that check come from the same source, so it proves the header
transcribes the config, not that the config describes the weights. A config key
that lies, or that we read as the wrong thing, passes it.

This is the independent side. `tests/vllm/models/indextts2_pth_manifest.json`
carries the tensor names, shapes and dtypes of `gpt.pth`, `codec.pth` and
`s2mel.pth` -- 1712 tensors read from their pickle headers by HTTP range
request, no weights fetched, reproducible with `scripts/read-torch-manifest.py`.
Several header constants must equal a dimension that actually appears in those
shapes, and this asserts exactly that.

It also pins which tensor groups our port does NOT yet model, so the scope in
`.agents/specs/indextts-2-5.md` cannot quietly drift from the checkpoint.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/vllm/models/indextts2_pth_manifest.json"
HEADER = ROOT / "include/vllm/model_executor/models/indextts2_config.h"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _constants() -> dict:
    text = HEADER.read_text()
    return {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"k(\w+)\s*=\s*(\d+)", text)
    }


class ShapesConfirmTheHeader(unittest.TestCase):
    """Each constant must appear as a real dimension of a real tensor."""

    def test_talker_dim_is_the_talker_projection_width(self):
        p = _manifest()["gpt.pth"]["patterns"]
        dim = _constants()["TalkerDim"]
        self.assertEqual(p["emo_layer.weight"][0], [dim, dim])
        self.assertEqual(p["spk_emb_proj.bias"][0], [dim])

    def test_style_dim_is_the_speaker_embedding_width(self):
        p = _manifest()["gpt.pth"]["patterns"]
        # CAMPPlus emits [style_dim]; spk_emb_proj consumes it.
        self.assertEqual(p["spk_emb_proj.weight"][0][1], _constants()["StyleDim"])

    def test_vocos_dims_are_the_codec_decoder_dims(self):
        p = _manifest()["codec.pth"]["patterns"]
        c = _constants()
        self.assertEqual(p["model.decoder.N.convnext.N.gamma"][0], [c["VocosDim"]])
        self.assertEqual(
            p["model.decoder.N.convnext.N.pwconv1.bias"][0],
            [c["VocosIntermediateDim"]],
        )

    def test_codec_hidden_size_is_the_decoder_input_width(self):
        p = _manifest()["codec.pth"]["patterns"]
        hidden = _constants()["CodecHiddenSize"]
        self.assertEqual(p["model.decoder.N.embed.weight"][0][1], hidden)
        self.assertEqual(p["model.down.bias"][0], [hidden])


class ManifestIsIntact(unittest.TestCase):
    def test_all_three_checkpoints_are_recorded(self):
        m = _manifest()
        self.assertEqual(
            sorted(m), ["codec.pth", "gpt.pth", "s2mel.pth"]
        )
        self.assertEqual(sum(f["tensors"] for f in m.values()), 1712)

    def test_the_talker_carries_a_language_embedding_of_107_rows(self):
        # The tokenizer is zh/ja/yue, but the talker embeds 107 language ids.
        # Recorded because it bears directly on what this lane may CLAIM.
        p = _manifest()["gpt.pth"]["patterns"]
        self.assertEqual(p["lang_embedding.weight"][0], [107, 1280])


class ScopeMatchesTheCheckpoint(unittest.TestCase):
    """Tensor groups the port does not model yet, named so they stay named."""

    UNPORTED_PREFIXES = (
        "emo_conditioning_encoder.",  # Conformer: rel-pos MHA, macaron FF, conv
        "emo_perceiver_encoder.",     # Perceiver resampler with learned latents
        "emo_layer.",
        "emovec_layer.",
    )

    def test_the_emotion_path_is_present_and_substantial(self):
        p = _manifest()["gpt.pth"]["patterns"]
        hit = [k for k in p if k.startswith(self.UNPORTED_PREFIXES)]
        self.assertGreaterEqual(
            len(hit),
            50,
            "the emotion path shrank; the spec's scope note describes a "
            "Conformer encoder plus a Perceiver resampler and must be re-read",
        )

    def test_the_cfm_estimator_carries_pieces_the_port_does_not_model(self):
        p = _manifest()["s2mel.pth"]["patterns"]
        for missing in (
            "net.cfm.estimator.wavenet.cond_layer.conv.conv.bias",
            "net.cfm.estimator.skip_linear.weight",
            "net.cfm.estimator.t_embedder2.mlp.N.weight",
        ):
            self.assertIn(
                missing,
                p,
                "the spec records this as unported; if it is gone from the "
                "checkpoint the scope note is wrong",
            )


if __name__ == "__main__":
    unittest.main()
