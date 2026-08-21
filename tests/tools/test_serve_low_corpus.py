"""Corpus tests port pinned SGLang custom-loader and vLLM seed behavior.

Sources:
- SGLang ``benchmark/datasets/custom.py:54-147`` @ 28b095c.
- vLLM ``tests/benchmarks/test_custom_dataset_seed.py`` @ e24d1b24.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
import unittest.mock

from tools.bench import make_serve_low_corpus
from tools.bench.make_serve_low_corpus import CorpusConfig, generate_corpus
from tools.bench.serve_low_common import HarnessError, read_jsonl, sha256_file

# A corpus is consumed as `evidence/corpus/<key>/`, so the key is the claim that
# these prompts were tokenized by THAT subject's tokenizer.  The expectation is
# written out here rather than read back from the tool under test.
_SUBJECT_KEY = "27"
_SUBJECT_REVISION = "890bdef7a42feba6d83b6e17a03315c694112f2a"
_FOREIGN_REVISION = "36f717a22990e82c54c1d48ee77c491b87825680"


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids) -> str:
        return bytes(token_ids).decode("utf-8")


def _files(root: pathlib.Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class CorpusTests(unittest.TestCase):
    def _config(self, seed: int = 7) -> CorpusConfig:
        return CorpusConfig(
            # A declared subject and ITS revision: a corpus is filed under the
            # key, so the generator refuses a key that names no checkpoint.
            model_key=_SUBJECT_KEY,
            tokenizer_revision=_SUBJECT_REVISION,
            seed=seed,
            target_input_len=64,
            output_len=8,
            requests_per_partition=3,
            warmup_requests=2,
            concurrencies=(1, 2),
            repetitions=2,
            common_prefix_limit=16,
        )

    def test_same_seed_is_byte_identical_and_manifest_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            left, right = root / "left", root / "right"
            generate_corpus(CharacterTokenizer(), left, self._config(), tokenizer_sha256="a" * 64)
            generate_corpus(CharacterTokenizer(), right, self._config(), tokenizer_sha256="a" * 64)
            self.assertEqual(_files(left), _files(right))

            manifest = json.loads((left / "manifest.json").read_text())
            for entry in manifest["files"]:
                self.assertEqual(entry["sha256"], sha256_file(left / entry["file"]))
            self.assertEqual(manifest["files"][0]["requests"], 2)
            self.assertTrue(
                all(entry["requests"] == 3 for entry in manifest["files"][1:])
            )

    def test_partitions_are_disjoint_exact_length_and_prefix_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = self._config()
            manifest = generate_corpus(
                CharacterTokenizer(), root, config, tokenizer_sha256="b" * 64
            )
            rows = [row for entry in manifest["files"] for row in read_jsonl(root / entry["file"])]
            self.assertEqual(len(rows), 14)
            self.assertEqual(len({row["prompt_sha256"] for row in rows}), len(rows))
            self.assertTrue(all(row["prompt_len"] == 64 for row in rows))
            self.assertTrue(all(len(row["prompt_token_ids"]) == 64 for row in rows))
            self.assertEqual({row["partition"] for row in rows}, {
                "warmup", "c1-r1", "c2-r1", "c1-r2", "c2-r2"
            })
            token_rows = [row["prompt_token_ids"] for row in rows]
            for index, left in enumerate(token_rows):
                for right in token_rows[index + 1 :]:
                    prefix = 0
                    for lhs, rhs in zip(left, right):
                        if lhs != rhs:
                            break
                        prefix += 1
                    self.assertLessEqual(prefix, config.common_prefix_limit)

    def test_different_seed_changes_the_selected_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            left, right = root / "left", root / "right"
            generate_corpus(CharacterTokenizer(), left, self._config(1), tokenizer_sha256="c" * 64)
            generate_corpus(CharacterTokenizer(), right, self._config(2), tokenizer_sha256="c" * 64)
            self.assertNotEqual(
                (left / "c1-r1.jsonl").read_bytes(),
                (right / "c1-r1.jsonl").read_bytes(),
            )


class CorpusSubjectKeyTests(unittest.TestCase):
    """The corpus is filed under a key, so the key must name a real subject.

    `make_serve_low_corpus.py` writes `manifest.json["model_key"]` verbatim and
    takes its output directory from an independent `--out`, so an undeclared
    key produced a corpus that claims a subject nothing declares -- the same
    defect as the harness's `--model-key` in a second file (#1594).
    """

    def _config(self, **overrides) -> CorpusConfig:
        fields = {
            "model_key": _SUBJECT_KEY,
            "tokenizer_revision": _SUBJECT_REVISION,
            "seed": 7,
            "target_input_len": 64,
            "output_len": 8,
            "requests_per_partition": 3,
            "warmup_requests": 2,
            "concurrencies": (1, 2),
            "repetitions": 2,
            "common_prefix_limit": 16,
        }
        fields.update(overrides)
        return CorpusConfig(**fields)

    def test_the_generator_refuses_a_key_no_subject_declares(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(HarnessError):
                generate_corpus(
                    CharacterTokenizer(),
                    pathlib.Path(temporary) / "out",
                    self._config(model_key="fixture"),
                    tokenizer_sha256="a" * 64,
                )

    def test_the_generator_refuses_another_subjects_tokenizer(self) -> None:
        """A corpus is this subject's only if its tokenizer is."""

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(HarnessError):
                generate_corpus(
                    CharacterTokenizer(),
                    pathlib.Path(temporary) / "out",
                    self._config(tokenizer_revision=_FOREIGN_REVISION),
                    tokenizer_sha256="a" * 64,
                )

    def test_the_command_line_refuses_a_key_no_subject_declares(self) -> None:
        """The parser refuses before the tokenizer is even opened.

        This is the entry point `online_gate.py`'s planned corpus command and
        `scripts/mxfp4-online-serving-grid.sh` both invoke.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            argv = [
                "make_serve_low_corpus.py",
                "--tokenizer-json", str(root / "tokenizer.json"),
                "--tokenizer-revision", _SUBJECT_REVISION,
                "--model-key", "38",
                "--out", str(root / "out"),
            ]
            with unittest.mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as refusal:
                        make_serve_low_corpus.main()
            self.assertEqual(refusal.exception.code, 2)
            self.assertIn("--model-key", stderr.getvalue())

    def test_an_admitted_key_still_generates_its_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = pathlib.Path(temporary) / "out"
            manifest = generate_corpus(
                CharacterTokenizer(), out, self._config(), tokenizer_sha256="a" * 64
            )
            self.assertEqual(manifest["model_key"], _SUBJECT_KEY)
            self.assertEqual(manifest["tokenizer_revision"], _SUBJECT_REVISION)


if __name__ == "__main__":
    unittest.main()
