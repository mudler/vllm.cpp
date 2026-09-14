#!/usr/bin/env python3
"""Mutation tests for scripts/check-deepseek-v4-vision-manifests.py.

Row `MODEL-MM-deepseek-v4-deepseek-v4-for-causal-lm`, issue #2411 and
ISSUE-LOCAL-01M2BXFMFDNZQD41HY629KAGCC.

WHAT THIS SUITE IS FOR. The checker ties the tensor map this tree derives to
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
`86f746b36186f0e567729a5c06a8c918caba82a9` without reading one weight byte:
every count, classification, vision shape, payload total and content digest in
the two committed manifests is RECOMPUTED from the committed `config.json`
rather than read back. That is a claim about what the gate can DETECT, and this
file performs each detection rather than asserting it. Every case below breaks
one guarantee in a scratch copy of the tree and requires the checker to go red
with a named diagnostic and a counted number of disagreements.

THE COUNTS HERE WERE MEASURED, NOT PREDICTED. Each `disagreements=` number is
what the checker actually printed against the mutated fixture, which is why a
case can assert an exact count instead of "at least one". Two of them are worth
naming, because they are the reason the manifests cannot drift apart quietly:

  one vision SHAPE changed        4 -- the shape, the payload total and BOTH
                                  content digests, because a shape the
                                  derivation rejects is also excluded from the
                                  records the digests are taken over
  config.json `vision_n_layers`   10 -- eight in the index manifest including
  32 -> 31                        `config_sha256`, and two more in the header
                                  manifest. The config and the manifests are
                                  held to each other, so editing the config
                                  alone cannot be made to look consistent

THE SUITE IS OFFLINE, and one case proves it: `verify_offline()` runs with
`urllib.request.urlopen` replaced by a function that fails the test. The
network path is `--refresh`, which nothing here takes -- `index_sha256` and
`header_sha256` are digests of remote bytes this repository does not mirror,
so no case may assert anything about them.

WHY A SCRATCH TREE. The checker resolves its fixtures from its own location
(`Path(__file__).resolve().parents[1]`), so a mutation is applied by copying
the script and the three fixtures into a temporary directory and running the
copy. Nothing here writes to the repository.

NO CHECKER ATTRIBUTE IS TOUCHED AT IMPORT TIME. The module is loaded, and that
is all. Under the disabled creation-contract stub that `scripts/check-pr-size.py`
substitutes for the BASE version of a checker created in its own pull request,
importing must still succeed so that the cases run and FAIL individually; a
module-level `checker.SOMETHING` would raise at import and report "Ran 0 tests",
which the evidence contract reads as no mutation evidence at all.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-deepseek-v4-vision-manifests.py"
FIXTURE_DIR = ROOT / "tests/parity/goldens/deepseek_v4_vision"
CONFIG_NAME = "config.json"
INDEX_NAME = "index_manifest.json"
HEADER_NAME = "shard1_header_manifest.json"
FIXTURE_NAMES = (CONFIG_NAME, INDEX_NAME, HEADER_NAME)

# The recorded position of the pinned artifact. These are the numbers the
# checker prints on a clean tree; a case asserts each of them appears, so a
# fixture refresh that moved one cannot pass unnoticed.
TENSOR_COUNT = 72633
SHARD_COUNT = 48
VISION_COUNT = 267
LANGUAGE_COUNT = 67658
MTP_COUNT = 4708
VISION_PAYLOAD_BYTES = 932786176

_SPEC = importlib.util.spec_from_file_location(
    "check_deepseek_v4_vision_manifests", CHECKER
)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)

_DISAGREEMENTS = re.compile(r"^([0-9]+) manifest disagreement\(s\)", re.MULTILINE)


class Tree:
    """A scratch checkout holding only the checker and its three fixtures.

    The checker derives its repository root from its own path, so a copy at
    `<tmp>/scripts/` reads `<tmp>/tests/parity/goldens/deepseek_v4_vision/`.
    """

    def __enter__(self) -> "Tree":
        self._dir = tempfile.TemporaryDirectory(prefix="dsv4v-manifest-gate-")
        root = Path(self._dir.name)
        (root / "scripts").mkdir(parents=True)
        self.script = root / "scripts" / CHECKER.name
        shutil.copy2(CHECKER, self.script)
        self.fixtures = root / "tests/parity/goldens/deepseek_v4_vision"
        self.fixtures.mkdir(parents=True)
        for name in FIXTURE_NAMES:
            shutil.copy2(FIXTURE_DIR / name, self.fixtures / name)
        return self

    def __exit__(self, *exc: object) -> None:
        self._dir.cleanup()

    # -- mutation helpers ---------------------------------------------------

    def drop(self, name: str) -> None:
        (self.fixtures / name).unlink()

    def edit(self, name: str, mutate) -> None:
        """Apply `mutate` to one fixture, rewritten in the checker's own form."""
        path = self.fixtures / name
        document = json.loads(path.read_bytes())
        mutate(document)
        path.write_bytes(
            (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

    def run(self) -> tuple[int, str]:
        done = subprocess.run(
            [sys.executable, str(self.script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return done.returncode, done.stdout + done.stderr


def _config() -> dict:
    return json.loads((FIXTURE_DIR / CONFIG_NAME).read_bytes())


def _header() -> dict:
    return json.loads((FIXTURE_DIR / HEADER_NAME).read_bytes())


class OfflineContract(unittest.TestCase):
    """What the checker must say when nothing is wrong."""

    def test_clean_fixtures_pass_and_report_the_pinned_position(self) -> None:
        with Tree() as tree:
            code, out = tree.run()
        self.assertEqual(code, 0, out)
        self.assertIn("ok deepseek-ai/DeepSeek-V4-Flash-Vision-Exp", out)
        for value in (TENSOR_COUNT, SHARD_COUNT, VISION_COUNT, VISION_PAYLOAD_BYTES):
            self.assertIn(str(value), out)
        self.assertIn("no network, no weight bytes read", out)
        self.assertIsNone(_DISAGREEMENTS.search(out), out)

    def test_the_live_repository_tree_is_green(self) -> None:
        """The committed fixtures, in place, through the real entry point."""
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = checker.verify_offline()
        self.assertEqual(code, 0, captured.getvalue())

    def test_the_default_mode_makes_no_network_call(self) -> None:
        """`--refresh` is the network path. The gate must never take it.

        A checker whose verdict depends on huggingface.co's uptime is one that
        cannot run in CI or on a disconnected machine, so this replaces the
        opener with a function that fails the test if anything reaches it.
        """

        def forbidden(*args: object, **kwargs: object):
            self.fail("the offline mode opened a network connection")

        with mock.patch("urllib.request.urlopen", forbidden):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                code = checker.verify_offline()
        self.assertEqual(code, 0, captured.getvalue())


class MissingFixtures(unittest.TestCase):
    """An absent fixture is a refusal, never a silent pass."""

    def test_absent_header_manifest_is_refused_by_name(self) -> None:
        with Tree() as tree:
            tree.drop(HEADER_NAME)
            code, out = tree.run()
        self.assertEqual(code, 1, out)
        self.assertIn(f"missing fixture tests/parity/goldens/deepseek_v4_vision/{HEADER_NAME}", out)
        self.assertIn("--refresh", out)
        # It stopped at the missing file rather than reporting derived
        # disagreements over a manifest it never read.
        self.assertIsNone(_DISAGREEMENTS.search(out), out)

    def test_absent_index_manifest_is_refused_by_name(self) -> None:
        with Tree() as tree:
            tree.drop(INDEX_NAME)
            code, out = tree.run()
        self.assertEqual(code, 1, out)
        self.assertIn(f"missing fixture tests/parity/goldens/deepseek_v4_vision/{INDEX_NAME}", out)

    def test_absent_config_is_refused_by_name(self) -> None:
        """The config is the DERIVATION's input, so its absence cannot pass."""
        with Tree() as tree:
            tree.drop(CONFIG_NAME)
            code, out = tree.run()
        self.assertEqual(code, 1, out)
        self.assertIn(f"missing fixture tests/parity/goldens/deepseek_v4_vision/{CONFIG_NAME}", out)

    def test_every_absent_fixture_is_named_not_just_the_first(self) -> None:
        with Tree() as tree:
            for name in FIXTURE_NAMES:
                tree.drop(name)
            code, out = tree.run()
        self.assertEqual(code, 1, out)
        for name in FIXTURE_NAMES:
            self.assertIn(f"missing fixture tests/parity/goldens/deepseek_v4_vision/{name}", out)


class IndexManifestMutations(unittest.TestCase):
    """Each recorded field of the index manifest is recomputed, not read."""

    def assert_red(self, out: str, code: int, count: int) -> None:
        self.assertEqual(code, 1, out)
        found = _DISAGREEMENTS.search(out)
        self.assertIsNotNone(found, out)
        self.assertEqual(int(found.group(1)), count, out)

    def test_a_bent_vision_count_reds_in_both_manifests(self) -> None:
        """267 -> 266 is caught twice: the classification and the cross-check.

        The two manifests record the vision count independently, and the
        checker holds them to each other. One edit therefore cannot be made
        self-consistent, which is the point of recording it twice.
        """
        with Tree() as tree:
            tree.edit(
                INDEX_NAME,
                lambda m: m["classifications"]["vision"].__setitem__("count", 266),
            )
            code, out = tree.run()
        self.assert_red(out, code, 2)
        self.assertIn("index manifest: classifications.vision.count is 266, derived 267", out)
        self.assertIn("header manifest: vision count against the index manifest", out)

    def test_a_tampered_config_digest_reds(self) -> None:
        with Tree() as tree:
            tree.edit(INDEX_NAME, lambda m: m.__setitem__("config_sha256", "0" * 64))
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("index manifest: config_sha256", out)

    def test_a_bent_revision_reds(self) -> None:
        """The manifest may only describe the PINNED revision."""
        with Tree() as tree:
            tree.edit(INDEX_NAME, lambda m: m.__setitem__("revision", "f" * 40))
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("index manifest: revision is", out)

    def test_a_bent_total_tensor_count_reds(self) -> None:
        with Tree() as tree:
            tree.edit(INDEX_NAME, lambda m: m.__setitem__("tensor_count", TENSOR_COUNT - 1))
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("index manifest: tensor_count", out)

    def test_a_tampered_name_digest_reds(self) -> None:
        """The digest is over the NAMES, so it cannot be edited to agree."""
        with Tree() as tree:
            tree.edit(
                INDEX_NAME, lambda m: m["all_names"].__setitem__("sha256", "0" * 64)
            )
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("index manifest: all_names.sha256", out)


class HeaderManifestMutations(unittest.TestCase):
    """Shard 1's header is the artifact's own statement about the vision group."""

    assert_red = IndexManifestMutations.assert_red

    def test_a_dropped_tensor_entry_reds_and_names_it(self) -> None:
        with Tree() as tree:
            tree.edit(HEADER_NAME, lambda m: m["tensors"].pop("image_end"))
            code, out = tree.run()
        self.assert_red(out, code, 2)
        self.assertIn("header manifest: header_tensor_count is 268, derived 267", out)
        self.assertIn("image_end", out)

    def test_an_extra_tensor_entry_reds_and_names_it(self) -> None:
        """The set is exact in both directions, not a floor."""
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m["tensors"].__setitem__(
                    "vision.blocks.0.mlp.w3.weight", {"dtype": "BF16", "shape": [1024]}
                ),
            )
            code, out = tree.run()
        self.assert_red(out, code, 2)
        self.assertIn("1 unexplained ['vision.blocks.0.mlp.w3.weight']", out)

    def test_one_changed_vision_shape_reds_the_shape_the_total_and_both_digests(self) -> None:
        """A shape is not recorded once. It is load-bearing four times over."""
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m["tensors"]["vision.norm.weight"].__setitem__("shape", [1023]),
            )
            code, out = tree.run()
        self.assert_red(out, code, 4)
        self.assertIn("header manifest: vision.norm.weight is [1023], derived [1024]", out)
        self.assertIn("header manifest: vision_payload_bytes", out)
        self.assertIn("header manifest: vision_records_fnv1a64", out)
        self.assertIn("header manifest: vision_records_sha256", out)

    def test_a_widened_dtype_reds(self) -> None:
        """BF16 is the released dtype. A token gate cannot see a wider one."""
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m["tensors"]["vision.norm.weight"].__setitem__("dtype", "F32"),
            )
            code, out = tree.run()
        self.assert_red(out, code, 4)
        self.assertIn("has dtype 'F32', expected BF16", out)

    def test_a_tampered_payload_total_reds(self) -> None:
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m.__setitem__("vision_payload_bytes", VISION_PAYLOAD_BYTES - 1),
            )
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("header manifest: vision_payload_bytes", out)

    def test_a_bent_shard_name_reds(self) -> None:
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m.__setitem__("shard", "model-00002-of-00048.safetensors"),
            )
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("header manifest: shard is", out)

    def test_the_embedding_shape_is_held_to_the_config(self) -> None:
        """`embed.weight` shares shard 1 with the vision group and is checked."""
        with Tree() as tree:
            tree.edit(
                HEADER_NAME,
                lambda m: m["tensors"]["embed.weight"].__setitem__("shape", [129279, 4096]),
            )
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("header manifest: embed.weight is [129279, 4096], config derives", out)


class ConfigMutations(unittest.TestCase):
    """The config and the manifests cannot drift independently."""

    assert_red = IndexManifestMutations.assert_red

    def test_a_bent_vision_layer_count_reds_ten_times_including_the_config_digest(self) -> None:
        """`vision_n_layers` 32 -> 31 removes eight tensors from the derivation.

        Eight disagreements land in the index manifest -- the total count, the
        three whole-map summary fields, the three vision summary fields, and
        `config_sha256`, which is what ties the edited bytes to the manifest
        that described them -- and two more in the header manifest, whose
        recorded vision group is now larger than the derivation admits.
        """
        with Tree() as tree:
            path = tree.fixtures / CONFIG_NAME
            text = path.read_text(encoding="utf-8")
            self.assertIn('"vision_n_layers": 32', text)
            path.write_text(
                text.replace('"vision_n_layers": 32', '"vision_n_layers": 31'),
                encoding="utf-8",
            )
            code, out = tree.run()
        self.assert_red(out, code, 10)
        self.assertIn("index manifest: config_sha256", out)
        self.assertIn("index manifest: classifications.vision.count is 267, derived 259", out)
        self.assertIn("header manifest: vision_tensor_count is 267, derived 259", out)

    def test_a_reformatted_config_reds_on_the_digest_alone(self) -> None:
        """The digest is over BYTES, so even a whitespace-only edit is caught.

        This is the one that stops the two files from being updated apart: the
        derived content is identical here, and only `config_sha256` moves.
        """
        with Tree() as tree:
            path = tree.fixtures / CONFIG_NAME
            document = json.loads(path.read_bytes())
            path.write_bytes(json.dumps(document, indent=4).encode("utf-8"))
            code, out = tree.run()
        self.assert_red(out, code, 1)
        self.assertIn("index manifest: config_sha256", out)


class Derivation(unittest.TestCase):
    """The derivation itself, exercised directly rather than through the file."""

    def test_the_classifier_accounts_for_every_tensor_exactly_once(self) -> None:
        config = _config()
        names = checker.official_names(config)
        self.assertEqual(len(names), TENSOR_COUNT)
        counts = {"language": 0, "mtp": 0, "vision": 0}
        for name in names:
            counts[checker.classify(name)] += 1
        self.assertEqual(counts["vision"], VISION_COUNT)
        self.assertEqual(counts["language"], LANGUAGE_COUNT)
        self.assertEqual(counts["mtp"], MTP_COUNT)
        self.assertEqual(sum(counts.values()), TENSOR_COUNT)

    def test_the_vision_set_is_exactly_what_the_loader_reads(self) -> None:
        """`vision.*`, `aligner.*` and the four learned sentinels, nothing else."""
        for name in (
            "vision.norm.weight",
            "vision.blocks.0.attn.wqkv.weight",
            "aligner.w1.weight",
            "image_start",
            "image_end",
            "image_newline",
            "image_pad",
        ):
            with self.subTest(name=name):
                self.assertEqual(checker.classify(name), "vision")
        for name in ("embed.weight", "layers.0.attn.wq_a.weight", "head.weight"):
            with self.subTest(name=name):
                self.assertEqual(checker.classify(name), "language")
        self.assertEqual(checker.classify("mtp.0.attn_norm.weight"), "mtp")

    def test_the_name_digest_is_order_sensitive(self) -> None:
        """A permutation must move the digest, or it cannot detect a reorder."""
        names = ["a", "b", "c"]
        self.assertNotEqual(
            checker.summary(names)["sha256"], checker.summary(["b", "a", "c"])["sha256"]
        )
        self.assertNotEqual(
            checker.summary(names)["fnv1a64"], checker.summary(["b", "a", "c"])["fnv1a64"]
        )

    def test_the_recorded_payload_total_is_reproduced_from_the_shape_rules(self) -> None:
        """Recomputed from `vision_shape()`, not read back from the manifest."""
        config = _config()
        total = 0
        for name, entry in _header()["tensors"].items():
            if name == "embed.weight":
                continue
            size = 2
            for dimension in checker.vision_shape(name, config):
                size *= dimension
            self.assertEqual(list(entry["shape"]), checker.vision_shape(name, config))
            total += size
        self.assertEqual(total, VISION_PAYLOAD_BYTES)

    def test_an_unknown_vision_name_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(SystemExit):
            checker.vision_shape("vision.blocks.0.attn.unknown_projection", _config())

    def test_a_config_too_short_to_derive_from_is_refused(self) -> None:
        config = _config()
        config["compress_ratios"] = config["compress_ratios"][:2]
        with self.assertRaises(SystemExit):
            checker.official_names(config)

    def test_the_derived_map_carries_no_duplicate(self) -> None:
        names = checker.official_names(_config())
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
