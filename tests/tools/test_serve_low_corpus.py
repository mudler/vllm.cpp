"""Corpus tests port pinned SGLang custom-loader and vLLM seed behavior.

Sources:
- SGLang ``benchmark/datasets/custom.py:54-147`` @ 28b095c.
- vLLM ``tests/benchmarks/test_custom_dataset_seed.py`` @ e24d1b24.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from tools.bench.make_serve_low_corpus import CorpusConfig, generate_corpus
from tools.bench.serve_low_common import read_jsonl, sha256_file


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
            model_key="fixture",
            tokenizer_revision="fixture-rev",
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


if __name__ == "__main__":
    unittest.main()
