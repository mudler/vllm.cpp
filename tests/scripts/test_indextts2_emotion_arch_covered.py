#!/usr/bin/env python3
"""The IndexTTS-2.5 emotion model is a STOCK Qwen3 we already load.

The shipped checkpoint carries a second language model under
`qwen0.6bemo4-merge/`. When it was first found (#634) it was recorded as
unscoped work, on the assumption that a second LM inside a TTS checkpoint
implied a second port. Its safetensors header says otherwise: 310 BF16 tensors
whose names and shapes are stock `Qwen3ForCausalLM`, an architecture this tree
already registers and loads.

This gate pins that reduction so it cannot rot silently. It asserts three
things against the COMMITTED manifest (read from the checkpoint's header by
range request, no weights downloaded) rather than against prose:

1. the manifest's architecture is registered in this tree,
2. every tensor-name pattern in it is a name our Qwen3 loader consumes,
3. the checkpoint ties its embeddings and therefore ships NO `lm_head.weight`,
   and our loader has the branch that handles that.

If any of those stops holding -- the emotion model is re-exported with a new
name, our loader is renamed, tying is dropped -- the emotion path stops being
free, and this fails rather than letting the spec keep claiming it is covered.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/vllm/models/indextts2_qwen_emo_manifest.json"
LOADER = ROOT / "src/vllm/model_executor/models/qwen3_weights.cpp"
REGISTRY = ROOT / "src/vllm/model_executor/models/qwen3_dense.cpp"

# Names the loader resolves without a per-layer prefix.
_TOP_LEVEL = {"model.embed_tokens.weight", "model.norm.weight"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _loader_literals(text: str) -> set:
    """Every string literal in the loader that names a checkpoint tensor."""
    return set(re.findall(r'"([A-Za-z0-9_.]+\.weight)"', text))


class EmotionModelIsStockQwen3(unittest.TestCase):
    def test_architecture_is_registered_in_this_tree(self):
        arch = _manifest()["architecture"]
        self.assertEqual(arch, "Qwen3ForCausalLM")
        self.assertIn(
            f'"{arch}"',
            REGISTRY.read_text(),
            f"{arch} is no longer registered by qwen3_dense.cpp, so the emotion "
            "model is not covered for free any more",
        )

    def test_every_tensor_name_is_one_the_loader_consumes(self):
        literals = _loader_literals(LOADER.read_text())
        uncovered = []
        for name in _manifest()["patterns"]:
            if name in _TOP_LEVEL:
                # Resolved whole, not through a layer prefix.
                leaf = name
                covered = any(lit.endswith(leaf.split(".")[-2:][0]) for lit in literals)
                covered = covered or leaf in literals
                # embed_tokens / norm are read via their own accessors; require
                # the distinguishing token to appear somewhere in the loader.
                covered = covered or name.split(".")[1] in LOADER.read_text()
            else:
                leaf = name.split(".N.", 1)[-1]
                covered = leaf in literals or any(
                    lit == leaf.split(".", 1)[-1] for lit in literals
                )
            if not covered:
                uncovered.append(name)
        self.assertEqual(
            uncovered,
            [],
            "the emotion checkpoint carries tensors our Qwen3 loader does not "
            "read; the emotion path is NOT free",
        )

    def test_checkpoint_ties_embeddings_and_ships_no_lm_head(self):
        m = _manifest()
        self.assertTrue(
            m["tie_word_embeddings"],
            "config.json no longer ties embeddings",
        )
        self.assertNotIn(
            "lm_head.weight",
            m["patterns"],
            "a tied checkpoint must not ship lm_head.weight",
        )
        self.assertIn(
            "tie_word_embeddings",
            LOADER.read_text(),
            "our loader has no tying branch, so a tied checkpoint would look "
            "like a missing lm_head",
        )

    def test_manifest_shape_matches_the_declared_config(self):
        m = _manifest()
        self.assertEqual(m["tensor_count"], 310)
        self.assertEqual(m["layers"], 28)
        self.assertEqual(m["dtype"], ["BF16"])
        # GQA: q is 16 heads x 128, k/v are 8 heads x 128, o takes q's width.
        p = m["patterns"]
        hidden = p["model.norm.weight"][0]
        self.assertEqual(p["model.layers.N.self_attn.q_proj.weight"], [2048, hidden])
        self.assertEqual(p["model.layers.N.self_attn.k_proj.weight"], [hidden, hidden])
        self.assertEqual(p["model.layers.N.self_attn.o_proj.weight"], [hidden, 2048])
        self.assertEqual(p["model.layers.N.self_attn.q_norm.weight"], [128])


if __name__ == "__main__":
    unittest.main()
