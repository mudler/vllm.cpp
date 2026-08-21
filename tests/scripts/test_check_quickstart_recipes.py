#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-quickstart-recipes.py.

Every case here is red before the rule it names exists: the suite imports the
checker by path, so it fails to collect at all while the checker is absent, and
each rule has a case that turns red when that rule alone is deleted.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-quickstart-recipes.py"
SPEC = importlib.util.spec_from_file_location("quickstart_recipes", CHECKER)
assert SPEC is not None and SPEC.loader is not None
quickstart_recipes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quickstart_recipes
SPEC.loader.exec_module(quickstart_recipes)


# Test-owned literals. Never derived from the checker or from the shipped
# matrix: a fixture that reads the production constant makes setup and
# expectation move together, and the assertion stops meaning anything.
PACKAGE = "ghcr.io/mudler/vllm.cpp"

MATRIX = {
    "package": PACKAGE,
    "lanes": [
        {
            "id": "cpu",
            "main_tag": "main-cpu",
            "moving_tags": ["latest-cpu", "latest"],
            "version_tag": "{version}-cpu",
        },
        {
            "id": "cuda",
            "main_tag": "main-cuda",
            "moving_tags": ["latest-cuda"],
            "version_tag": "{version}-cuda",
        },
    ],
}

# A minimal page that satisfies every rule, used as the mutation baseline.
VALID = "\n".join(
    [
        "# Quickstart",
        "",
        "```sh",
        f"docker run --rm -p 8000:8000 {PACKAGE}:latest \\",
        "  --model Qwen/Qwen3-0.6B",
        "```",
        "",
        "```sh",
        "# PENDING(#1281): nobody has run this row yet.",
        "vllm-server --model PENDING",
        "```",
        "",
        "| Model | Date run |",
        "|---|---|",
        "| `--model PENDING` | PENDING(#1281) |",
        "",
    ]
)


def errors(text: str) -> list[str]:
    return quickstart_recipes.page_errors(text, MATRIX)


class RepoIdGrammar(unittest.TestCase):
    """The mirror of IsValidHfRepoId, hf_hub.cpp:254."""

    def test_accepts_one_slash_between_base_characters(self) -> None:
        self.assertTrue(quickstart_recipes.is_valid_hf_repo_id("Qwen/Qwen3-0.6B"))
        self.assertTrue(quickstart_recipes.is_valid_hf_repo_id("a/b"))
        self.assertTrue(quickstart_recipes.is_valid_hf_repo_id("org_1/repo.v2"))

    def test_requires_exactly_one_slash(self) -> None:
        self.assertFalse(quickstart_recipes.is_valid_hf_repo_id("Qwen3-0.6B"))
        self.assertFalse(quickstart_recipes.is_valid_hf_repo_id("a/b/c"))

    def test_special_characters_need_a_base_character_on_both_sides(self) -> None:
        for bad in ("/repo", "org/", "org//repo", "org/-repo", "org/repo-", ".org/repo"):
            with self.subTest(bad=bad):
                self.assertFalse(quickstart_recipes.is_valid_hf_repo_id(bad))

    def test_rejects_empty_oversized_and_non_ascii(self) -> None:
        self.assertFalse(quickstart_recipes.is_valid_hf_repo_id(""))
        self.assertFalse(quickstart_recipes.is_valid_hf_repo_id("o/" + "r" * 300))
        self.assertFalse(quickstart_recipes.is_valid_hf_repo_id("org/repö"))


class ModelValueGrammar(unittest.TestCase):
    """The mirror of ParseModelReference, model_resolver.cpp:365-402."""

    def test_classifies_the_four_accepted_shapes(self) -> None:
        cases = {
            "/models/Qwen3-0.6B": "local path",
            "./weights/model.gguf": "local gguf file",
            "Qwen/Qwen3-0.6B": "hub snapshot",
            "unsloth/Qwen3-0.6B-GGUF:Q4_K_M": "hub gguf file",
        }
        for value, kind in cases.items():
            with self.subTest(value=value):
                self.assertEqual(quickstart_recipes.classify_model_value(value), kind)

    def test_windows_path_is_a_path_and_not_a_tagged_repository(self) -> None:
        # model_resolver.cpp:382-394 splits on the LAST colon precisely so this
        # value keeps today's behavior instead of looking like `C` plus a tag.
        self.assertEqual(
            quickstart_recipes.classify_model_value(r"C:\models\qwen"), "local path"
        )

    def test_refuses_what_the_loader_refuses(self) -> None:
        for bad in ("Qwen3-0.6B", "org//repo", "https://huggingface.co/a/b", ""):
            with self.subTest(bad=bad):
                self.assertIsNone(quickstart_recipes.classify_model_value(bad))


class PublishedTags(unittest.TestCase):
    """The tag set is derived from the matrix, never restated."""

    def test_moving_and_main_tags_are_exact(self) -> None:
        exact, _ = quickstart_recipes.published_tags(MATRIX)
        self.assertEqual(
            exact, {"latest", "latest-cpu", "latest-cuda", "main-cpu", "main-cuda"}
        )

    def test_version_tags_are_matched_by_lane_suffix(self) -> None:
        _, patterns = quickstart_recipes.published_tags(MATRIX)
        matched = [t for _, p in patterns if p.match("0.0.3-cpu") for t in ("hit",)]
        self.assertEqual(matched, ["hit"])
        self.assertFalse(any(p.match("0.0.3-gpu") for _, p in patterns))

    def test_the_version_half_of_a_version_tag_is_pinned_too(self) -> None:
        # Both halves, because a pattern that pins only the lane suffix accepts
        # `nightly-cpu` and `latest-cpu-v2` as version tags. Mutation testing
        # found this: widening `{version}` to `.+` left the suite green.
        _, patterns = quickstart_recipes.published_tags(MATRIX)
        for bad in ("nightly-cpu", "v0.0.3-cpu", "-cpu", "0.0-cpu", "main-cpu-cpu"):
            with self.subTest(bad=bad):
                self.assertFalse(any(p.match(bad) for _, p in patterns))

    def test_a_lane_added_to_the_matrix_is_accepted_without_a_code_edit(self) -> None:
        widened = json.loads(json.dumps(MATRIX))
        widened["lanes"].append(
            {
                "id": "vulkan",
                "main_tag": "main-vulkan",
                "moving_tags": ["latest-vulkan"],
                "version_tag": "{version}-vulkan",
            }
        )
        page = f"# Q\n\n```sh\ndocker run {PACKAGE}:latest-vulkan --model a/b\n```\n"
        self.assertTrue(quickstart_recipes.page_errors(page, MATRIX))
        self.assertEqual(quickstart_recipes.page_errors(page, widened), [])


class ImageRule(unittest.TestCase):
    """Rule 1: the tag the container matrix actually publishes."""

    def test_the_valid_page_is_clean(self) -> None:
        self.assertEqual(errors(VALID), [])

    def test_an_unpublished_tag_fails(self) -> None:
        broken = VALID.replace(f"{PACKAGE}:latest", f"{PACKAGE}:latest-gpu")
        found = errors(broken)
        self.assertTrue(found)
        self.assertIn("latest-gpu", found[0])

    def test_a_misspelled_repository_fails(self) -> None:
        broken = VALID.replace(PACKAGE, "ghcr.io/mudler/vllm-cpp")
        found = errors(broken)
        self.assertTrue(found)
        self.assertIn("vllm-cpp", found[0])

    def test_a_pending_marker_does_not_excuse_a_wrong_tag(self) -> None:
        # The whole point of the marker is that it excuses a MISSING value and
        # never a wrong one. An unpublished tag is spelled exactly like the
        # published one, so nothing here may depend on the marker.
        broken = "\n".join(
            [
                "# Quickstart",
                "",
                "```sh",
                "# PENDING(#1281): the first publish has not happened.",
                f"docker run {PACKAGE}:latest-gpu --model Qwen/Qwen3-0.6B",
                "```",
                "",
            ]
        )
        found = errors(broken)
        self.assertTrue(found)
        self.assertIn("latest-gpu", found[0])

    def test_an_unpublished_tag_under_a_marker_is_still_reported(self) -> None:
        # The same assertion at page level, so deleting the pending-scope
        # bypass from image_errors cannot be made to look harmless.
        marked = VALID.replace(
            f"docker run --rm -p 8000:8000 {PACKAGE}:latest \\",
            f"# PENDING(#1281)\ndocker run {PACKAGE}:9.9.9-gpu \\",
        )
        self.assertTrue(errors(marked))


class ModelRule(unittest.TestCase):
    """Rule 2: the grammar ParseModelReference implements."""

    def test_a_repository_without_an_organisation_fails(self) -> None:
        broken = VALID.replace("--model Qwen/Qwen3-0.6B", "--model Qwen3-0.6B")
        found = errors(broken)
        self.assertTrue(found)
        self.assertIn("Qwen3-0.6B", found[0])

    def test_the_equals_spelling_is_read_too(self) -> None:
        broken = VALID.replace("--model Qwen/Qwen3-0.6B", "--model=Qwen3-0.6B")
        self.assertTrue(errors(broken))

    def test_backticks_and_pipes_are_not_part_of_the_value(self) -> None:
        page = "# Q\n\n| Model |\n|---|\n| `--model Qwen/Qwen3-0.6B` |\n"
        self.assertEqual(errors(page), [])

    def test_a_tagged_gguf_repository_is_accepted(self) -> None:
        page = "# Q\n\n```sh\nvllm-server --model unsloth/Qwen3-0.6B-GGUF:Q4_K_M\n```\n"
        self.assertEqual(errors(page), [])


class PendingMarker(unittest.TestCase):
    """The tolerance, and its exact limits."""

    def test_an_unmarked_placeholder_fails(self) -> None:
        broken = VALID.replace("# PENDING(#1281): nobody has run this row yet.\n", "")
        found = errors(broken)
        self.assertTrue(found)
        self.assertIn("PENDING(#N)", found[0])

    def test_a_marker_without_an_issue_number_does_not_authorise(self) -> None:
        broken = VALID.replace("PENDING(#1281)", "PENDING")
        self.assertTrue(errors(broken))

    def test_a_marker_covers_its_whole_fenced_block(self) -> None:
        page = "\n".join(
            [
                "# Q",
                "",
                "```sh",
                "# PENDING(#1281): not run yet.",
                "vllm-server \\",
                "  --model PENDING",
                "```",
                "",
            ]
        )
        self.assertEqual(errors(page), [])

    def test_a_marker_does_not_leak_out_of_its_block(self) -> None:
        page = "\n".join(
            [
                "# Q",
                "",
                "```sh",
                "# PENDING(#1281): not run yet.",
                "vllm-server --model PENDING",
                "```",
                "",
                "```sh",
                "vllm-server --model PENDING",
                "```",
                "",
            ]
        )
        found = errors(page)
        self.assertEqual(len(found), 1)
        self.assertIn("line 9", found[0])

    def test_a_marker_outside_a_fence_covers_only_its_own_line(self) -> None:
        page = "\n".join(
            [
                "# Q",
                "",
                "| Model | Date run |",
                "|---|---|",
                "| `--model PENDING` | PENDING(#1281) |",
                "| `--model PENDING` | 2026-08-20 |",
                "",
            ]
        )
        found = errors(page)
        self.assertEqual(len(found), 1)
        self.assertIn("line 6", found[0])

    def test_a_placeholder_cannot_authorise_itself(self) -> None:
        page = "# Q\n\n```sh\nvllm-server --model PENDING(#1281)\n```\n"
        self.assertTrue(errors(page))


class ShippedPage(unittest.TestCase):
    """The shipped page passes its own gate, and the gate is wired to run."""

    def test_docs_quickstart_exists_and_passes(self) -> None:
        page = ROOT / "docs/QUICKSTART.md"
        self.assertTrue(page.exists(), "docs/QUICKSTART.md is missing")
        matrix = json.loads(
            (ROOT / "release/container-matrix.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            quickstart_recipes.page_errors(page.read_text(encoding="utf-8"), matrix), []
        )

    def test_the_readme_points_at_the_page(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/QUICKSTART.md", readme)

    def test_the_checker_and_this_suite_are_wired_into_preflight_and_ci(self) -> None:
        # A suite nothing invokes pins nothing (#1509). These two assertions are
        # what keep this one out of that population.
        preflight = (ROOT / "scripts/agent-preflight.sh").read_text(encoding="utf-8")
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        for haystack, name in ((preflight, "agent-preflight.sh"), (ci, "ci.yml")):
            with self.subTest(name=name):
                self.assertIn("check-quickstart-recipes", haystack)
                self.assertIn("test_check_quickstart_recipes", haystack)


class Entrypoint(unittest.TestCase):
    """The command-line surface fails closed."""

    def _run(self, page_text: str | None) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_file = Path(tmp) / "matrix.json"
            matrix_file.write_text(json.dumps(MATRIX), encoding="utf-8")
            page = Path(tmp) / "QUICKSTART.md"
            if page_text is not None:
                page.write_text(page_text, encoding="utf-8")
            argv = sys.argv
            sys.argv = [
                "check-quickstart-recipes.py",
                "--page",
                str(page),
                "--matrix",
                str(matrix_file),
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    return quickstart_recipes.main()
            finally:
                sys.argv = argv

    def test_a_missing_page_is_a_failure_and_not_a_pass(self) -> None:
        self.assertEqual(self._run(None), 1)

    def test_a_valid_page_exits_zero(self) -> None:
        self.assertEqual(self._run(VALID), 0)

    def test_a_broken_page_exits_one(self) -> None:
        self.assertEqual(self._run(VALID.replace("Qwen/Qwen3-0.6B", "Qwen3-0.6B")), 1)


if __name__ == "__main__":
    unittest.main()
