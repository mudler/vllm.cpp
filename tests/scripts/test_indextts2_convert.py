#!/usr/bin/env python3
"""The IndexTTS-2.5 checkpoint converter's contract, without the checkpoint.

`scripts/convert-indextts2-checkpoint.py` turns upstream's `.pth` files into
safetensors offline, because this tree has no torch-pickle reader and will not
grow one: pickle executes arbitrary code by design, and every other lane here
loads safetensors or GGUF.

The conversion itself needs 4 GiB of weights and torch, so CI cannot run it.
What CI CAN hold is the part that decides which tensors survive and under what
names, which is where a silent mistake would be unrecoverable: a dropped weight
looks exactly like a weight that was never there. These cases exercise that
logic directly, with fakes, and need neither torch nor the checkpoint.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/convert-indextts2-checkpoint.py"


def _module():
    spec = importlib.util.spec_from_file_location("convert_indextts2", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeTensor:
    """Duck-types the two attributes `flatten` tests for."""

    def __init__(self, shape=(1,), dtype="float32"):
        self.shape = shape
        self.dtype = dtype


class FlattenContract(unittest.TestCase):
    def test_nested_state_dicts_join_with_dots(self):
        m = _module()
        state = {"net": {"cfm": {"estimator": {"conv1": {"weight": FakeTensor()}}}}}
        names = [n for n, _ in m.flatten(state)]
        self.assertEqual(names, ["net.cfm.estimator.conv1.weight"])

    def test_the_names_match_the_committed_manifest_spelling(self):
        # The manifest was built by the same dotted join, so a converted file's
        # names ARE the manifest's names. If this drifts, the manifest stops
        # being able to check the conversion.
        import json

        m = _module()
        manifest = json.loads(
            (ROOT / "tests/vllm/models/indextts2_pth_manifest.json").read_text()
        )
        sample = next(iter(manifest["s2mel.pth"]["patterns"]))
        self.assertTrue(sample.startswith("net."), sample)
        state = {"net": {"cfm": {"estimator": {"input_pos": FakeTensor()}}}}
        self.assertIn("net.cfm.estimator.input_pos", dict(m.flatten(state)))

    def test_a_bare_tensor_yields_itself(self):
        m = _module()
        t = FakeTensor()
        self.assertEqual([(("", t))], [(n, v) for n, v in m.flatten(t)])

    def test_non_tensor_leaves_are_skipped_not_crashed_on(self):
        # `.pth` files carry ints, strings and None beside the weights.
        m = _module()
        state = {"iters": 12000, "name": "s2mel", "opt": None,
                 "net": {"w": FakeTensor()}}
        self.assertEqual([n for n, _ in m.flatten(state)], ["net.w"])


class DropContract(unittest.TestCase):
    def test_optimizer_state_is_the_only_thing_dropped(self):
        m = _module()
        self.assertEqual(m.DROP_PREFIXES, ("optimizer.",))

    def test_the_drop_prefix_matches_real_optimizer_keys(self):
        # codec.pth is 75% optimizer state: 729 of its 972 tensors. The prefix
        # has to match those keys or the conversion carries training residue
        # into a shipped artifact.
        import json

        manifest = json.loads(
            (ROOT / "tests/vllm/models/indextts2_pth_manifest.json").read_text()
        )
        codec = manifest["codec.pth"]["patterns"]
        opt = [k for k in codec if k.startswith("optimizer.")]
        self.assertTrue(opt, "codec.pth no longer carries optimizer state")
        m = _module()
        for key in opt:
            self.assertTrue(key.startswith(m.DROP_PREFIXES))

    def test_model_weights_are_never_matched_by_the_drop_prefix(self):
        import json

        manifest = json.loads(
            (ROOT / "tests/vllm/models/indextts2_pth_manifest.json").read_text()
        )
        m = _module()
        for source in ("gpt.pth", "s2mel.pth"):
            for key in manifest[source]["patterns"]:
                self.assertFalse(
                    key.startswith(m.DROP_PREFIXES),
                    f"{source}:{key} would be dropped as optimizer state",
                )


class SourceContract(unittest.TestCase):
    def test_all_three_checkpoints_are_converted(self):
        m = _module()
        self.assertEqual(sorted(m.SOURCES), ["codec.pth", "gpt.pth", "s2mel.pth"])


if __name__ == "__main__":
    unittest.main()
