#!/usr/bin/env python3
"""Behavior tests for strict Git trailer enforcement."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-commit-trailers.py"


def load_checker():
    # repository root on sys.path, so provide it here.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("check_commit_trailers", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STRICT_MESSAGE = """policy: enforce trailers

The body may explain the change.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: Codex:GPT-5 [Codex]
"""


class CommitMessageContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def assertInvalid(self, message: str, needle: str) -> None:
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_valid_ai_assisted_and_human_only_messages(self) -> None:
        self.assertEqual(
            self.checker.validate_commit_message(STRICT_MESSAGE, strict=True), []
        )
        human = STRICT_MESSAGE.replace(
            "AI-Assisted: true\nAssisted-by: Codex:GPT-5 [Codex]\n",
            "AI-Assisted: false\n",
        )
        self.assertEqual(
            self.checker.validate_commit_message(human, strict=True), []
        )

    def test_substring_or_embedded_legacy_tag_is_not_a_raw_paragraph(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "FOLLOWING_AGENTS_PROTOCOL\n\n",
                "We are FOLLOWING_AGENTS_PROTOCOL today.\n\n",
            ),
            "separate paragraph",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "FOLLOWING_AGENTS_PROTOCOL\n\n",
                "FOLLOWING_AGENTS_PROTOCOL extra\n\n",
            ),
            "separate paragraph",
        )

    def test_raw_tag_must_be_separate_from_the_trailer_paragraph(self) -> None:
        message = STRICT_MESSAGE.replace(
            "FOLLOWING_AGENTS_PROTOCOL\n\nFollowing-Agents",
            "FOLLOWING_AGENTS_PROTOCOL\nFollowing-Agents",
        )
        self.assertInvalid(message, "separate paragraph")

    def test_protocol_and_ai_declarations_are_unique_and_exact(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "Following-Agents-Protocol: true",
                "Following-Agents-Protocol: false",
            ),
            "must be exactly true",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "Following-Agents-Protocol: true",
                "Following-Agents-Protocol: true\nFollowing-Agents-Protocol: true",
            ),
            "exactly once",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace("AI-Assisted: true", "AI-Assisted: maybe"),
            "true or false",
        )
        self.assertInvalid(
            STRICT_MESSAGE.replace(
                "AI-Assisted: true", "AI-Assisted: true\nAI-Assisted: false"
            ),
            "exactly once",
        )

    def test_assistance_attribution_is_required_and_has_closed_syntax(self) -> None:
        self.assertInvalid(
            STRICT_MESSAGE.replace("Assisted-by: Codex:GPT-5 [Codex]\n", ""),
            "Assisted-by",
        )
        for malformed in (
            "Codex GPT-5 [Codex]",
            "Codex:GPT-5",
            "Codex:GPT-5 []",
            "Codex: GPT-5 [Codex]",
            "Codex:GPT-5 [Codex] trailing",
        ):
            with self.subTest(malformed=malformed):
                self.assertInvalid(
                    STRICT_MESSAGE.replace("Codex:GPT-5 [Codex]", malformed),
                    "malformed Assisted-by",
                )

    def test_human_only_declaration_rejects_assistance_attribution(self) -> None:
        message = STRICT_MESSAGE.replace("AI-Assisted: true", "AI-Assisted: false")
        self.assertInvalid(message, "must omit Assisted-by")

    def test_ai_authorship_trailers_are_forbidden(self) -> None:
        for trailer in (
            "Signed-off-by: Codex:GPT-5 [Codex]",
            "Co-Authored-By: Claude Opus <noreply@anthropic.com>",
            "Signed-off-by: OpenAI Bot <bot@openai.com>",
        ):
            with self.subTest(trailer=trailer):
                self.assertInvalid(
                    STRICT_MESSAGE.rstrip() + "\n" + trailer + "\n",
                    "AI authorship",
                )

    def test_human_authorship_trailers_remain_allowed(self) -> None:
        for trailer in (
            "Signed-off-by: Alice Example <alice@example.com>",
            "Co-Authored-By: Bob Human <bob@example.org>",
        ):
            with self.subTest(trailer=trailer):
                message = STRICT_MESSAGE.rstrip() + "\n" + trailer + "\n"
                self.assertEqual(
                    self.checker.validate_commit_message(message, strict=True), []
                )

    def test_every_declared_agent_model_and_tool_identity_is_not_an_author(self) -> None:
        message = STRICT_MESSAGE.replace(
            "Codex:GPT-5 [Codex]",
            "Unfamiliar-Agent:Model-Seven [NeutralTool] [SecondTool]",
        )
        for identity in ("Model-Seven", "NeutralTool", "SecondTool"):
            with self.subTest(identity=identity):
                authored = (
                    message.rstrip()
                    + f"\nCo-Authored-By: {identity} <{identity.casefold()}@example.com>\n"
                )
                self.assertInvalid(authored, "AI authorship")

    def test_assisted_agent_identity_is_forbidden_as_an_authorship_trailer(self) -> None:
        message = STRICT_MESSAGE.replace(
            "Codex:GPT-5 [Codex]", "Unfamiliar-Agent:Model-7 [NeutralTool]"
        ).rstrip()
        message += "\nSigned-off-by: Unfamiliar-Agent <agent@example.com>\n"
        self.assertInvalid(message, "AI authorship")

    def test_legacy_is_only_accepted_in_non_strict_mode(self) -> None:
        legacy = "legacy change\n\nFOLLOWING_AGENTS_PROTOCOL\n"
        self.assertEqual(
            self.checker.validate_commit_message(legacy, strict=False), []
        )
        self.assertTrue(
            self.checker.validate_commit_message(legacy, strict=True)
        )

    def test_parser_matches_git_interpret_trailers(self) -> None:
        parsed = subprocess.check_output(
            ["git", "interpret-trailers", "--parse"],
            input=STRICT_MESSAGE,
            text=True,
        )
        self.assertEqual(self.checker.parsed_trailers(STRICT_MESSAGE), parsed)


class RangeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"],
            check=True,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def commit(self, message: str) -> str:
        marker = self.repo / "history"
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        subprocess.run(["git", "-C", str(self.repo), "add", "history"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-q", "-F", "-"],
            input=message,
            text=True,
            check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()

    def test_cutover_commit_itself_is_strict_and_parent_is_legacy(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        before = self.commit("before\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        cutover = self.commit(STRICT_MESSAGE)
        after = self.commit(STRICT_MESSAGE.replace("policy:", "policy after:"))
        errors = self.checker.validate_range(
            self.repo, base, after, cutover=cutover
        )
        self.assertEqual(errors, [])
        self.assertNotEqual(before, cutover)

    def test_post_cutover_legacy_commit_fails(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        cutover = self.commit(STRICT_MESSAGE)
        self.commit("after\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        errors = self.checker.validate_range(
            self.repo, base, "HEAD", cutover=cutover
        )
        self.assertTrue(any("Following-Agents-Protocol" in error for error in errors))

    def test_range_is_taken_from_the_merge_base_when_the_base_branch_moved(self) -> None:
        """A PR branch diverges the moment main moves; that must still validate.

        RED before GATE-FORK-ANCESTRY (#773): CI passes
        `pull_request.base.sha`, the TIP of the base branch, which stops being an
        ancestor of head as soon as main advances. `validate_range` raised
        "range base must be an ancestor of range head" and returned WITHOUT
        READING A SINGLE COMMIT -- so the trailer contract was never enforced on
        any external contribution.
        """
        root = self.commit("root\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        # The PR branch, cut from root.
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-b", "pr", root],
            check=True,
        )
        head = self.commit(STRICT_MESSAGE)
        # Main moves on after the branch was cut. This is the ordinary case.
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-B", "main", root],
            check=True,
        )
        moved_main = self.commit(STRICT_MESSAGE.replace("policy:", "mainline:"))
        self.assertNotEqual(moved_main, root)

        errors = self.checker.validate_range(
            self.repo, moved_main, head, cutover=None
        )
        self.assertEqual(errors, [], "the PR's own commits must validate")

    def test_a_bad_trailer_in_the_merge_base_range_is_still_reported(self) -> None:
        """Changing WHICH commits are read must not change what is demanded.

        Green on both sides of #773 in the ancestor case; the point is that it
        stays green in the DIVERGED case too, so the range fix cannot be
        mistaken for a way to smuggle a non-conforming commit past the gate.
        """
        root = self.commit("root\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-b", "pr2", root],
            check=True,
        )
        self.commit("no trailers here at all\n")
        head = self.commit(STRICT_MESSAGE)
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-B", "main2", root],
            check=True,
        )
        moved_main = self.commit(STRICT_MESSAGE.replace("policy:", "mainline2:"))

        errors = self.checker.validate_range(
            self.repo, moved_main, head, cutover=None
        )
        self.assertTrue(errors, "the offending commit must still be reported")

    def test_unrelated_histories_still_fail_closed(self) -> None:
        """No merge base at all is absence of INFORMATION, and must still raise.

        This is the half of the old non-ancestor assertion that must NOT move:
        an orphan branch shares no commit with the base, so there is no range to
        compute and reporting one would be an invention.
        """
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "--orphan", "orphan"],
            check=True,
        )
        subprocess.run(["git", "-C", str(self.repo), "rm", "-qrf", "."], check=True)
        orphan = self.commit(STRICT_MESSAGE)
        with self.assertRaises(ValueError):
            self.checker.validate_range(self.repo, base, orphan, cutover=None)

    def test_missing_unreachable_and_non_ancestor_revisions_fail_closed(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        head = self.commit(STRICT_MESSAGE)
        with self.assertRaises(ValueError):
            self.checker.validate_range(
                self.repo, "missing", head, cutover=head
            )
        # The divergent-but-RELATED half of this case moved to
        # test_range_is_taken_from_the_merge_base_when_the_base_branch_moved
        # (#773): two branches off a common root share a merge base, which is
        # the ordinary shape of every pull request and must validate rather than
        # raise. The genuinely unrelated case -- no shared history at all -- is
        # asserted in test_unrelated_histories_still_fail_closed, so the
        # fail-closed behaviour this case was written for is still pinned.

    def test_ambiguous_revision_name_fails_closed(self) -> None:
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        head = self.commit(STRICT_MESSAGE)
        subprocess.run(
            ["git", "-C", str(self.repo), "branch", "collision", base], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "tag", "collision", head], check=True
        )
        with self.assertRaises(ValueError):
            self.checker.validate_range(
                self.repo, base, "collision", cutover=head
            )


class MergeArtifacts(unittest.TestCase):
    """The trailer gate must judge the CLAIM, not the paragraph layout (#406).

    Every shape below was taken from a real commit on main. The gate was failing
    on how commits LAND rather than on how they are written, and 13 of the last
    30 commits on main failed it -- unnoticed only because those runs were
    cancelled (#274).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def test_github_co_authored_by_does_not_hide_the_block(self) -> None:
        """RED-BEFORE: this is dbd0d51c, 87308dea and f64f2b71 on main.

        Squash-merging through GitHub appends `Co-authored-by:` as a SEPARATE
        trailing paragraph. `git interpret-trailers --parse` reads only the last
        paragraph, so the protocol trailers become invisible and the gate counts
        zero. The commit is correct; the parse is what breaks.

        AGENTS.md forbids AI tools from adding Co-Authored-By. It does not
        forbid GitHub from recording a real human co-author, so this must pass.
        """
        message = STRICT_MESSAGE + "\nCo-authored-by: Ettore Di Giacinto <mudler@localai.io>\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_a_squash_that_doubles_the_trailer_block_still_fails(self) -> None:
        """b8293c88, the commit that turned main red, must STAY red.

        Squashing a multi-commit PR concatenates each commit's trailer block, so
        every trailer appears twice. That was tempting to collapse -- two
        identical declarations do say the same thing -- but it is NOT what this
        change fixes, and relaxing it would delete a rule that catches genuinely
        malformed messages.

        The distinction that matters: the Co-authored-by case above is a correct
        commit REJECTED BY THE PARSE, while this one is a message that really is
        malformed and is fixable at the source by writing the squash body (or by
        landing a single-commit PR). AGENTS.md's "fix the cause, not the gate"
        puts this on the process side of the line.
        """
        _, _, trailers = STRICT_MESSAGE.rpartition("FOLLOWING_AGENTS_PROTOCOL\n")
        doubled = STRICT_MESSAGE + "\nFOLLOWING_AGENTS_PROTOCOL\n" + trailers
        errors = self.checker.validate_commit_message(doubled, strict=True)
        self.assertTrue(errors, "a doubled trailer block must still be rejected")

    def test_contradictory_declarations_fail(self) -> None:
        """A commit cannot both declare and deny AI assistance."""
        conflicting = STRICT_MESSAGE + "AI-Assisted: false\n"
        errors = self.checker.validate_commit_message(conflicting, strict=True)
        self.assertTrue(errors, "contradictory declarations must fail")

    def test_a_github_merge_commit_with_no_trailers_still_fails(self) -> None:
        """This is b580452d. It must STAY red -- a real violation, not a layout
        artifact. The relaxation is about placement and duplication only."""
        message = (
            "Merge pull request #386 from mudler/row/ENG-NOW-DERIVED-DONE\n"
            "\n"
            "close the row\n"
        )
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "a message with no trailers at all must fail")

    def test_prose_after_the_trailers_still_fails(self) -> None:
        """A prose paragraph must still terminate the block.

        Without this the relaxation would accept trailers buried anywhere near
        the end, which is exactly the looseness the gate exists to prevent.
        """
        message = STRICT_MESSAGE + "\nAnd then some closing prose about the change.\n"
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "prose after the trailer block must still fail")


class ForgeAttribution(unittest.TestCase):
    """The forge's Co-authored-by is attribution, not an authorship claim (#418).

    AGENTS.md forbids AI tools from adding Co-Authored-By so a model cannot claim
    it wrote the code. GitHub adds the line itself on squash merge, naming the
    ACCOUNT that opened the PR -- and most PRs here are opened by localai-bot. The
    AI-involvement claim is already made, separately and explicitly, by
    AI-Assisted and Assisted-by. Conflating the two reds main for correct work.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def test_a_forge_bot_co_author_is_accepted(self) -> None:
        """RED-BEFORE: this is f64f2b71 on main.

        GitHub generated this exact line for the account that opened the PR. It
        was invisible until #406 fixed the parse, which is why it reads as a new
        failure and is not one.
        """
        message = STRICT_MESSAGE + (
            "\nCo-authored-by: localai-org-maint-bot "
            "<306269227+localai-org-maint-bot@users.noreply.github.com>\n"
        )
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_a_hand_written_ai_co_author_is_still_forbidden(self) -> None:
        """The guard that matters more than the relaxation.

        The exemption is keyed on the forge's own noreply domain, not on the
        name. A model crediting itself with an ordinary address still fails, so
        the rule still does the job it exists for.
        """
        message = STRICT_MESSAGE + "\nCo-authored-by: Claude <claude@anthropic.com>\n"
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "a hand-written AI co-author must still fail")

    def test_signed_off_by_gets_no_exemption(self) -> None:
        """Sign-off is a legal assertion, not attribution.

        A forge address must NOT buy an AI a Signed-off-by, so this stays red
        even with the same noreply domain that exempts a co-author.
        """
        message = STRICT_MESSAGE + (
            "\nSigned-off-by: localai-bot "
            "<1+localai-bot@users.noreply.github.com>\n"
        )
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "Signed-off-by must never be exempted")

    def test_a_human_co_author_is_unaffected(self) -> None:
        """The case #406 already fixed must keep passing."""
        message = STRICT_MESSAGE + "\nCo-authored-by: Ettore Di Giacinto <mudler@localai.io>\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )


class SquashShapes(unittest.TestCase):
    """The two squash bodies GitHub can compose, pinned as executable evidence.

    `squash_merge_commit_message` decides which body lands. GitHub writes a
    `---------` separator above the `Co-authored-by:` block it appends, and that
    happens under BOTH settings -- #829 and #850 believed it came from
    concatenation, and `617d6f452` disproved it (#861). The separator is now
    stepped over when trailers sit on both sides.
    `join_trailing_trailer_paragraphs` fuses only trailer-SHAPED paragraphs, so
    the separator orphans the block and the gate reports trailers the commit
    plainly carries as missing (#829).

    The repetition in that body is NOT the defect. `git interpret-trailers
    --parse` reads only the trailing block, so N copies parse as one. Changing
    the count rule would not have fixed anything. The separator is the defect,
    and the setting removed it.
    """

    COAUTHOR = "Co-authored-by: Ettore Di Giacinto <mudler@localai.io>"

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def body(self) -> str:
        return (
            "policy(ROW): a subject (#1)\n\n"
            "A body paragraph that explains the reason.\n\n"
            "FOLLOWING_AGENTS_PROTOCOL\n\n"
            "Following-Agents-Protocol: true\n"
            "AI-Assisted: true\n"
            "Assisted-by: AGENT:claude-opus-5 [Claude Code]"
        )

    def test_the_real_forge_shape_is_green(self) -> None:
        """Block, separator, co-author. This is what `617d6f452` actually is,
        and it is the shape every squash lands in."""
        message = f"{self.body()}\n\n---------\n\n{self.COAUTHOR}\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_prose_after_the_block_still_terminates_it(self) -> None:
        """The property the separator step-over must not cost."""
        message = f"{self.body()}\n\nSome prose afterwards.\n"
        self.assertTrue(self.checker.validate_commit_message(message, strict=True))

    def test_a_separator_followed_by_prose_is_not_stepped_over(self) -> None:
        message = f"{self.body()}\n\n---------\n\nSome prose.\n"
        self.assertTrue(self.checker.validate_commit_message(message, strict=True))

    def test_a_separator_with_prose_above_it_is_not_stepped_over(self) -> None:
        message = f"subject (#1)\n\nprose.\n\n---------\n\n{self.COAUTHOR}\n"
        self.assertTrue(self.checker.validate_commit_message(message, strict=True))

    def test_pr_body_shape_is_green(self) -> None:
        """What lands now: one body, then the forge's Co-authored-by."""
        message = f"{self.body()}\n\n{self.COAUTHOR}\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_commit_messages_shape_with_the_separator_is_red(self) -> None:
        """What used to land, and what would land again if the repository
        setting were reverted to COMMIT_MESSAGES."""
        message = (
            f"{self.body()}\n\n---------\n\n{self.body()}\n\n{self.COAUTHOR}\n"
        )
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(errors, "the orphaning separator must still be caught")

    def test_repetition_alone_is_not_what_breaks_it(self) -> None:
        """Same two blocks, NO separator between them. This isolates the cause:
        if this were red, the count rule would be the problem."""
        message = f"{self.body()}\n\n{self.body()}\n\n{self.COAUTHOR}\n"
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(
            any("FOLLOWING_AGENTS_PROTOCOL" in error for error in errors),
            f"expected the marker rule to be what fires, got {errors}",
        )


class PullRequestBodyMode(unittest.TestCase):
    """The body is the commit message now, so it is gated like one (#848)."""

    TEMPLATE = ROOT / ".github/pull_request_template.md"

    def run_checker(self, *args: str, stdin: str | None = None):
        return subprocess.run(
            [sys.executable, str(CHECKER), *args],
            input=stdin, capture_output=True, text=True, check=False, cwd=ROOT,
        )

    def test_the_shipped_template_satisfies_the_contract(self) -> None:
        """If this fails, every pull request opened from the template lands a
        commit that fails the trailer gate AFTER the merge."""
        result = self.run_checker("--message-file", str(self.TEMPLATE))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_body_without_the_block_is_rejected(self) -> None:
        result = self.run_checker("--message-file", "-", stdin="Just a description.\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FOLLOWING_AGENTS_PROTOCOL", result.stderr)

    def test_an_empty_body_is_rejected(self) -> None:
        result = self.run_checker("--message-file", "-", stdin="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("empty", result.stderr)

    def test_filled_rejects_the_unreplaced_placeholder(self) -> None:
        """The template must pass, and a body that is STILL the template must
        not. Otherwise the landed commit attributes the work to nobody."""
        result = self.run_checker(
            "--message-file", str(self.TEMPLATE), "--filled"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("placeholder", result.stderr)

    def test_filled_accepts_a_real_identity(self) -> None:
        body = self.TEMPLATE.read_text(encoding="utf-8").replace(
            "AGENT:MODEL [TOOL]", "AGENT:claude-opus-5 [Claude Code]"
        )
        result = self.run_checker("--message-file", "-", "--filled", stdin=body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_filled_ignores_the_placeholder_MENTIONED_in_prose(self) -> None:
        """A body that DOCUMENTS the flag must not trip it.

        The first implementation searched the raw message, so any body naming
        `AGENT:MODEL [TOOL]` while explaining the check was rejected -- including
        the spec for this row and the body of the pull request that introduced
        it. The comparison is against the PARSED Assisted-by value instead.
        """
        body = (
            "fix(ROW): a subject (#1)\n\n"
            "This flag rejects the placeholder AGENT:MODEL [TOOL], which "
            "satisfies the grammar while crediting nobody.\n\n"
            "FOLLOWING_AGENTS_PROTOCOL\n\n"
            "Following-Agents-Protocol: true\n"
            "AI-Assisted: true\n"
            "Assisted-by: AGENT:claude-opus-5 [Claude Code]\n"
        )
        result = self.run_checker("--message-file", "-", "--filled", stdin=body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_filled_still_rejects_the_placeholder_as_a_real_trailer(self) -> None:
        """The other half: prose is exempt, a real trailer value is not."""
        body = (
            "fix(ROW): a subject (#1)\n\n"
            "A body.\n\n"
            "FOLLOWING_AGENTS_PROTOCOL\n\n"
            "Following-Agents-Protocol: true\n"
            "AI-Assisted: true\n"
            "Assisted-by: AGENT:MODEL [TOOL]\n"
        )
        result = self.run_checker("--message-file", "-", "--filled", stdin=body)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("placeholder", result.stderr)

    def test_exactly_one_of_range_or_message_file(self) -> None:
        both = self.run_checker("--range", "HEAD~1..HEAD", "--message-file", "-")
        self.assertNotEqual(both.returncode, 0)
        neither = self.run_checker()
        self.assertNotEqual(neither.returncode, 0)


class LandedMessageExceptions(unittest.TestCase):
    """One landed commit, one exact error, excused by name (#1262).

    `281b4bc76c0e` is on `main` carrying `Assisted-by: AGENT:claude-opus-5 CLI`,
    with no bracketed tool. It cannot be repaired, because that rewrites `main`,
    and it does not clear itself, because the main lane walks `LAST_GREEN..head`
    and `LAST_GREEN` advances only on a green run.

    `--cutover` is the wrong shape for it and was measured before it was
    rejected: it downgrades every ancestor of one sha to the marker-only check
    (2986 commits here), so it waives defects nobody has read, and it does not
    even excuse the sha you name -- `merge-base --is-ancestor X X` succeeds, so
    the cutover commit itself stays strict.

    These cases are the scope proof for the narrow instrument that replaced it.
    The registry excuses ONE commit and ONE exact error string; a different
    error in the same commit, the same error in a different commit, and any
    error in a message that has not landed yet are all still red.
    """

    OFFENDER = "281b4bc76c0e635adbc7ed38317035b07c99864d"
    OFFENDING_VALUE = "AGENT:claude-opus-5 CLI"

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "config", "user.name", "Test"], check=True
        )

    def commit(self, message: str) -> str:
        marker = self.repo / "history"
        marker.write_text(marker.read_text() + "x" if marker.exists() else "x")
        subprocess.run(["git", "-C", str(self.repo), "add", "history"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-q", "-F", "-"],
            input=message, text=True, check=True,
        )
        return subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True
        ).strip()

    def register(self, oid: str, error: str) -> None:
        """Point the registry at a fixture commit for the length of one test.

        The module is loaded per class by `load_checker`, so this touches this
        class's own copy; it is restored anyway, because a test that leaves a
        registry entry behind would excuse a commit in a later test.
        """
        registry = self.checker.LANDED_MESSAGE_EXCEPTIONS
        original = dict(registry)
        self.addCleanup(lambda: (registry.clear(), registry.update(original)))
        registry.clear()
        registry[oid] = self.checker.LandedException(error, "fixture reason (#1262)")

    def malformed(self, value: str) -> str:
        return STRICT_MESSAGE.replace("Codex:GPT-5 [Codex]", value)

    def malformed_error(self, value: str) -> str:
        return f"[attribution] malformed Assisted-by value {value!r}"

    def run_checker(self, *args: str, stdin: str | None = None):
        return subprocess.run(
            [sys.executable, str(CHECKER), *args],
            input=stdin, capture_output=True, text=True, check=False, cwd=ROOT,
        )

    def offender_present(self) -> bool:
        return subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{self.OFFENDER}^{{commit}}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0

    # -- the real landed commit -------------------------------------------

    def test_the_registered_exception_clears_the_real_landed_red(self) -> None:
        """GREEN-AFTER on the commit this row exists for, run as CI runs it.

        RED-BEFORE, measured on the same command at `27d5432f9`:
        `281b4bc76c0e: [attribution] malformed Assisted-by value
        'AGENT:claude-opus-5 CLI'`, exit 1.
        """
        if not self.offender_present():
            self.skipTest(
                f"{self.OFFENDER[:12]} is not in this object store; a shallow "
                "clone cannot measure a landed message"
            )
        result = self.run_checker("--range", f"{self.OFFENDER}~1..{self.OFFENDER}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Visible debt: a reader of the GREEN lane still sees the offender.
        self.assertIn(self.OFFENDER[:12], result.stdout)
        self.assertIn(self.OFFENDING_VALUE, result.stdout)
        self.assertIn("debt", result.stdout.casefold())

    def test_every_registered_exception_is_live(self) -> None:
        """No dead and no overbroad entry.

        Each entry is re-derived against the commit it names: the error it
        excuses must be an error that commit actually produces. An entry whose
        commit was fixed, or whose text was guessed, is red here rather than
        sitting in the registry excusing nothing.
        """
        for oid, entry in self.checker.LANDED_MESSAGE_EXCEPTIONS.items():
            with self.subTest(commit=oid[:12]):
                if subprocess.run(
                    ["git", "-C", str(ROOT), "cat-file", "-e", f"{oid}^{{commit}}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ).returncode != 0:
                    self.skipTest("shallow clone: the commit is not present")
                message = subprocess.check_output(
                    ["git", "-C", str(ROOT), "show", "-s", "--format=%B", oid],
                    text=True,
                ) + "\n"
                errors = self.checker.validate_commit_message(message, strict=True)
                self.assertIn(entry.error, errors)

    def test_the_registry_holds_exactly_the_one_landed_commit(self) -> None:
        """Nothing else is excused.

        The count is the whole guard. Adding an entry has to edit this number
        and say why, in a diff a reviewer reads, which is the property a
        `--cutover` sha does not have: moving a cutover changes no count and
        names no defect.
        """
        registry = self.checker.LANDED_MESSAGE_EXCEPTIONS
        self.assertEqual(list(registry), [self.OFFENDER])
        for oid, entry in registry.items():
            self.assertEqual(len(oid), 40, "key must be a FULL commit oid")
            self.assertEqual(oid, oid.lower())
            int(oid, 16)
            self.assertIn("#", entry.reason, "an entry must name its issue")

    # -- scope: what stays red --------------------------------------------

    def test_a_different_violation_in_the_excused_commit_still_fails(self) -> None:
        """The excused commit is not an excused commit; ONE error is excused.

        This is the property `--cutover` cannot hold: a cutover downgrades the
        whole message to the marker check, so a second defect in the same
        commit disappears with the first.
        """
        base = self.commit(STRICT_MESSAGE)
        head = self.commit(
            self.malformed("Codex:GPT-5 Codex")
            + "Signed-off-by: Claude <claude@anthropic.com>\n"
        )
        self.register(head, self.malformed_error("Codex:GPT-5 Codex"))
        failures = self.checker.validate_range(self.repo, base, head, cutover=None)
        self.assertTrue(
            any("Signed-off-by is forbidden" in f for f in failures), failures
        )
        self.assertFalse(
            any("malformed Assisted-by" in f for f in failures), failures
        )

    def test_a_later_violation_is_still_caught(self) -> None:
        """A violation introduced AFTER the excused commit is still red."""
        base = self.commit(STRICT_MESSAGE)
        excused = self.commit(self.malformed("Codex:GPT-5 Codex"))
        later = self.commit(self.malformed("Codex:GPT-5 Codex"))
        self.register(excused, self.malformed_error("Codex:GPT-5 Codex"))
        failures = self.checker.validate_range(self.repo, base, later, cutover=None)
        self.assertEqual(len(failures), 1, failures)
        self.assertTrue(failures[0].startswith(later[:12]), failures)

    def test_the_exception_is_keyed_on_the_exact_error_text(self) -> None:
        """A near-miss registration excuses nothing.

        The registry cannot be written loosely enough to cover a defect it does
        not literally name.
        """
        base = self.commit(STRICT_MESSAGE)
        head = self.commit(self.malformed("Codex:GPT-5 Codex"))
        self.register(head, self.malformed_error("Codex:GPT-5 CLI"))
        failures = self.checker.validate_range(self.repo, base, head, cutover=None)
        self.assertTrue(
            any("malformed Assisted-by" in f for f in failures), failures
        )

    def test_an_unlisted_commit_is_never_excused(self) -> None:
        """The same defect in a commit the registry does not name stays red."""
        base = self.commit(STRICT_MESSAGE)
        listed = self.commit(self.malformed("Codex:GPT-5 Codex"))
        unlisted = self.commit(self.malformed("Codex:GPT-5 Codex"))
        self.register(listed, self.malformed_error("Codex:GPT-5 Codex"))
        failures = self.checker.validate_range(
            self.repo, listed, unlisted, cutover=None
        )
        self.assertTrue(
            any(f.startswith(unlisted[:12]) for f in failures), failures
        )
        self.assertNotEqual(base, listed)

    def test_a_pull_request_body_is_never_excused(self) -> None:
        """The registry cannot become a way to land a NEW malformed body.

        `--message-file` gates the pull request body BEFORE it becomes a commit,
        under `squash_merge_commit_message = PR_BODY`. It runs
        `validate_commit_message`, which does not consult the registry at all,
        so the exact string that landed in `281b4bc76c0e` is still rejected in
        anything written today.
        """
        result = self.run_checker(
            "--message-file", "-", "--filled",
            stdin=self.malformed(self.OFFENDING_VALUE),
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("malformed Assisted-by", result.stderr)


if __name__ == "__main__":
    unittest.main()
