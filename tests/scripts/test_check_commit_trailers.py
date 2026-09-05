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

    def test_a_merge_commit_is_not_authored_content_and_is_skipped(self) -> None:
        """#2157: the same CI job walks twice and only one walk skips merges.

        `ci.yml:872` skips a commit with more than one parent -- "they are not
        authored content" -- and then hands the SAME range to this checker, which
        did not. So `git merge origin/main` on a row branch, the routine way to
        take main, reddened commit-protocol-tag on a message git wrote and no
        contributor can edit without a force-push.

        RED before this change: the merge commit's default message carries no
        trailer block and the walk demanded one.
        """
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-b", "side", base],
            check=True,
        )
        self.commit(STRICT_MESSAGE.replace("policy:", "policy side:"))
        subprocess.run(
            ["git", "-C", str(self.repo), "checkout", "-q", "-"], check=True
        )
        self.commit(STRICT_MESSAGE.replace("policy:", "policy main:"))
        # `git merge --no-edit` writes "Merge branch 'side'" and nothing else.
        subprocess.run(
            ["git", "-C", str(self.repo), "merge", "--no-edit", "side"],
            check=True, capture_output=True,
        )
        errors = self.checker.validate_range(self.repo, base, "HEAD", cutover=None)
        self.assertEqual(
            errors, [], "a merge commit is git's message, not authored content"
        )

    def test_a_non_merge_commit_in_the_same_range_is_still_demanded(self) -> None:
        """The skip must be keyed on PARENT COUNT, not on looking merge-ish.

        Without this case the fix above could be widened into "stop checking",
        which is the shape AGENTS.md forbids: never make a red gate green by
        widening an assertion's scope.
        """
        base = self.commit("base\n\nFOLLOWING_AGENTS_PROTOCOL\n")
        self.commit("Merge branch 'nothing' -- but a single parent\n")
        errors = self.checker.validate_range(self.repo, base, "HEAD", cutover=None)
        self.assertTrue(errors, "a one-parent commit is authored content")

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


class PatchSectionFraming(unittest.TestCase):
    """A `---` line is git's PATCH DIVIDER, and the checker must say so (#1563).

    `squash_merge_commit_message = PR_BODY` makes a body the landed commit
    message, and git's `find_patch_start` ends a message at the first line that
    begins `---` and continues with whitespace. Everything below one leaves the
    message. Three shapes were measured at `db648fb88` with the `---` as the
    only difference, and none of them reported the cause: the rule ABOVE the
    trailers reported the trailers as missing while the body carried each
    exactly once, and the rule BELOW the trailers, plus the three-hyphen
    separator shape, both passed GREEN.

    This is the AUTHOR-side half of the class `SquashShapes` covers from the
    FORGE side. The forge writes `---------`, which is NOT a divider, and its
    treatment is unchanged; the two cases sit in one file so the difference
    between nine hyphens and three stays executable.
    """

    BODY = (
        "policy(ROW): a subject (#1)\n\n"
        "A body paragraph that explains the reason.\n\n"
        "FOLLOWING_AGENTS_PROTOCOL\n\n"
        "Following-Agents-Protocol: true\n"
        "AI-Assisted: true\n"
        "Assisted-by: AGENT:claude-opus-5 [Claude Code]"
    )
    COAUTHOR = "Co-authored-by: Ettore Di Giacinto <mudler@localai.io>"

    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def framing(self, message: str) -> list[str]:
        return [
            error
            for error in self.checker.validate_commit_message(message, strict=True)
            if error.startswith("[framing]")
        ]

    def test_the_detector_agrees_with_git_line_for_line(self) -> None:
        """The rule is git's, so it is asserted against git and not restated.

        Each form is handed to `git interpret-trailers --parse`, which is the
        program the checker already shells out to. A divider truncates the
        message, so the trailer below it disappears from the parse; a form that
        is not a divider leaves it there. The detector must predict exactly
        that, for every form, with no case left to a description.
        """
        forms = {
            "---": True,
            "--- ": True,
            "--- a/file.c": True,
            "\t---": False,
            "----": False,
            "---x": False,
            "---------": False,
            # NOT a divider, and the case that makes the ASCII guard
            # load-bearing: `str.isspace()` calls U+00A0 whitespace and
            # git's ASCII `isspace` does not, so a detector written with
            # the obvious Python idiom reports a line git walks past.
            "---\u00a0": False,
        }
        for line, is_divider in forms.items():
            with self.subTest(line=line):
                message = f"subject\n\n{line}\n\nTrailer-Key: value\n"
                parsed = subprocess.run(
                    ["git", "interpret-trailers", "--parse"],
                    input=message, text=True, capture_output=True, check=True,
                )
                git_lost_the_trailer = "Trailer-Key: value" not in parsed.stdout
                self.assertEqual(
                    git_lost_the_trailer, is_divider,
                    f"git 2.x disagrees about {line!r}: {parsed.stdout!r}",
                )
                detected = self.checker.patch_section_line(message)
                self.assertEqual(
                    detected == 3, is_divider,
                    f"detector said {detected!r} for {line!r}",
                )

    def test_a_divider_as_the_last_line_without_a_newline_is_not_one(self) -> None:
        """git tests the character AFTER `---`, and at end of string there is
        none. Mirrored exactly rather than rounded off, because a detector that
        is stricter than git reports a line git does not act on."""
        self.assertIsNone(self.checker.patch_section_line("subject\n\n---"))
        self.assertEqual(self.checker.patch_section_line("subject\n\n---\n"), 3)

    def test_a_rule_above_the_trailers_names_the_line_not_the_trailers(self) -> None:
        """The reported defect at `db648fb88`. The body carries
        `Following-Agents-Protocol` exactly once and the checker said it must
        appear exactly once, which sends the reader to count occurrences."""
        message = (
            "subject\n\nprose\n\n---\n\nmore prose\n\n"
            "FOLLOWING_AGENTS_PROTOCOL\n\n"
            "Following-Agents-Protocol: true\nAI-Assisted: true\n"
            "Assisted-by: AGENT:claude-opus-5 [Claude Code]\n"
        )
        errors = self.checker.validate_commit_message(message, strict=True)
        self.assertTrue(self.framing(message), errors)
        self.assertTrue(any("line 5" in error for error in errors), errors)
        self.assertFalse(
            any("must appear exactly once" in error for error in errors),
            f"the trailer map is not evidence once git stopped reading: {errors}",
        )

    def test_a_rule_below_the_trailers_is_red(self) -> None:
        """Was GREEN. `test_prose_after_the_trailers_still_fails` pins that
        trailing prose is red; writing `---` above that prose made the same body
        pass, because git truncates there and the block becomes the last
        paragraph of what it reads."""
        message = f"{self.BODY}\n\n---\n\nnotes nobody parses\n"
        self.assertTrue(self.framing(message), "a divider below the block is red")

    def test_a_three_hyphen_separator_before_the_co_author_is_red(self) -> None:
        """Was GREEN. `join_trailing_trailer_paragraphs` steps over a paragraph
        of three or more hyphens, so the fuse deleted the divider before git saw
        it while the LANDED message still carries it."""
        message = f"{self.BODY}\n\n---\n\n{self.COAUTHOR}\n"
        self.assertTrue(self.framing(message), "the fuse must not hide a divider")

    def test_a_diff_header_is_red(self) -> None:
        """The convention this exists for. A message that quotes a diff was
        already broken; now it says so."""
        message = f"{self.BODY}\n\n--- a/src/vt/gemm.cpp\n+++ b/src/vt/gemm.cpp\n"
        self.assertTrue(self.framing(message), "a diff header is a divider")

    def test_the_forge_separator_is_untouched(self) -> None:
        """The regression this row must not cause. `---------` is not a divider
        and `617d6f452` is the landed proof; Part 1's treatment of it stands."""
        message = f"{self.BODY}\n\n---------\n\n{self.COAUTHOR}\n"
        self.assertEqual(
            self.checker.validate_commit_message(message, strict=True), []
        )

    def test_the_other_markdown_rules_are_not_dividers(self) -> None:
        """What the error tells the author to use instead has to actually pass,
        or the message sends them to a second red."""
        for rule in ("***", "___", "----"):
            with self.subTest(rule=rule):
                block = self.BODY[self.BODY.index("FOLLOWING"):]
                message = (
                    f"subject\n\nprose\n\n{rule}\n\nmore prose\n\n{block}\n"
                )
                self.assertEqual(self.framing(message), [])

    def test_the_shipped_template_carries_no_divider(self) -> None:
        """The template is what an author starts from, so a divider in it would
        be shipped rather than authored."""
        template = (ROOT / ".github/pull_request_template.md").read_text(
            encoding="utf-8"
        )
        self.assertIsNone(self.checker.patch_section_line(template))

    def test_the_template_warns_the_author(self) -> None:
        """The gate refuses the body; the template is where the author is told
        before they write it."""
        template = (ROOT / ".github/pull_request_template.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("---", template)
        self.assertRegex(template, r"(?s)horizontal rule.*\*\*\*")

    def test_no_reachable_commit_carries_a_divider(self) -> None:
        """The measurement that licenses applying this to the RANGE walk too.

        A landed message cannot be repaired, so a rule that any landed commit
        already violates would turn the main lane permanently red. Zero of the
        3118 commits reachable from `main` at `db648fb88` carry a divider line.
        This re-derives it rather than quoting it, and it fails loudly and by
        name if one ever lands, instead of the range walk going red for a
        reason nobody can see.
        """
        log = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--format=%H%x00%B%x00", "HEAD"],
            text=True, capture_output=True, check=True,
        ).stdout
        records = [r for r in log.split("\x00") if r]
        offenders = []
        checked = 0
        for sha, message in zip(records[0::2], records[1::2]):
            checked += 1
            line = self.checker.patch_section_line(message)
            if line is not None:
                offenders.append((sha.strip()[:12], line))
        self.assertGreater(checked, 3000, f"only {checked} commits were read")
        self.assertEqual(offenders, [], f"{checked} commits read")

    def test_the_cli_reports_the_framing_rule_and_exits_one(self) -> None:
        """The door that CI and `scripts/agent-pr-body.py` both drive."""
        with tempfile.TemporaryDirectory() as directory:
            body = Path(directory) / "body"
            body.write_text(
                f"{self.BODY}\n\n---\n\nnotes\n", encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, str(CHECKER), "--message-file", str(body),
                 "--filled"],
                text=True, capture_output=True,
            )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("[framing]", result.stderr)
        self.assertIn("patch divider", result.stderr.casefold())


class AttributionIsEnforcedOnce(unittest.TestCase):
    """The trailer walk judges commits an author can still fix, and only those.

    `--range` had THREE deployments: preflight on your own branch, the CI
    pull-request lane, and the CI PUSH lane on `main`. The first two are
    diff-scoped to commits the author wrote and can still amend or force-push.
    The third re-read LANDED history, where the only repair is rewriting `main`,
    which AGENTS.md forbids outright.

    It was also REDUNDANT. `squash_merge_commit_message = PR_BODY` means the
    pull request body becomes the landed commit message, and the same checker
    already validates that body -- the exact bytes, before they freeze. The push
    lane re-checked what had just been checked, and its only unique coverage was
    of a direct push to `main`, which the landing rules already prohibit.

    What it produced instead was forgiveness work. Two mechanisms existed solely
    to excuse landed commits: `scripts/ci-enforcement-floor.txt`, whose current
    value forgives 43 commits dated 2026-08-13 to 2026-08-25, and
    `LANDED_MESSAGE_EXCEPTIONS`, which carried one. 44 commits on `main` violate
    a contract and cannot be repaired, each forgiven by a deliberate reviewed
    act. A gate whose three-week output is 44 forgiveness decisions rather than
    44 prevented defects is measuring the wrong thing.
    """

    WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    def test_the_trailer_walk_runs_on_pull_requests_only(self) -> None:
        job = self.WORKFLOW.split("  commit-protocol-tag:", 1)[1].split("\n  pr-size:", 1)[0]
        walks = [
            line for line in job.splitlines()
            if "check-commit-trailers.py" in line and "--range" in line
        ]
        self.assertTrue(walks, "precondition: no range walk found to scope")
        # Every step carrying a range walk must be pull_request-gated.
        self.assertIn(
            "if: github.event_name == 'pull_request'", job,
            "the trailer walk is not scoped to the pull-request lane; it still "
            "reads landed history no contributor can repair",
        )

    def test_the_pull_request_body_is_still_validated(self) -> None:
        """The half that must NOT move. Removing the push walk is only safe
        because this one acts BEFORE the bytes freeze."""
        self.assertIn('--message-file "$body_file" --filled', self.WORKFLOW)

    def test_no_landed_message_exception_registry_survives(self) -> None:
        """The registry excused LANDED commits from a walk that no longer reads
        any. It cannot apply, and a dead exception list invites a live one."""
        source = (ROOT / "scripts/check-commit-trailers.py").read_text(encoding="utf-8")
        self.assertNotIn("LANDED_MESSAGE_EXCEPTIONS", source)

    def test_the_enforcement_floor_no_longer_claims_the_trailer_steps(self) -> None:
        """The floor still serves documentation-checkpoint and agent-record, so
        the FILE stays. Its own comment enumerated the gates it governs, and
        leaving the trailer steps named there would be a record that lies."""
        floor = (ROOT / "scripts/ci-enforcement-floor.txt").read_text(encoding="utf-8")
        # The GOVERNING sentence, not a bare substring: the file still explains
        # that the trailer steps used to be governed here, and a crude
        # `assertNotIn` cannot tell that history from a live claim.
        governed = floor.split("never start behind it.", 1)[0]
        self.assertIn("documentation-checkpoint", governed)
        self.assertNotIn("commit-protocol-tag", governed)
        self.assertTrue(
            floor.rstrip().endswith("e1b5df1a6b5b30555639e0a0459f79a467544579"),
            "the floor VALUE moves only by a reviewed advance that re-pins this "
            "assertion; #2322 narrowed its scope and did not move it, and #2743 "
            "advanced it to e1b5df1a6 without widening the scope back",
        )


class WhichBodyIsRead(unittest.TestCase):
    """The two doors read DIFFERENT bytes, and this row does not blur that.

    The framing rule lands in the shared `validate_commit_message`, so both
    doors gain it. That is not the same as the two doors agreeing about WHICH
    body they hold to it, and #1263 is what happens when the difference is
    assumed away: CI reads the FROZEN `pull_request` event payload from the last
    push, `scripts/agent-pr-body.py --pr` reads the LIVE body, and the squash
    lands the live body. A body edited after the final push is therefore green
    in CI and lands unread.
    """

    def test_ci_reads_the_frozen_event_payload(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("PR_BODY: ${{ github.event.pull_request.body }}", workflow)
        self.assertIn('--message-file "$body_file" --filled', workflow)
        self.assertNotIn("gh pr view", workflow)

    def test_the_local_command_reads_the_live_body(self) -> None:
        tool = (ROOT / "scripts/agent-pr-body.py").read_text(encoding="utf-8")
        self.assertIn('"gh", "pr", "view", str(number)', tool)
        self.assertIn('"--json", "body"', tool)
        self.assertNotIn("github.event", tool)

    def test_the_spec_names_which_is_which(self) -> None:
        spec = (ROOT / ".agents/specs/squash-separator.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("FROZEN event payload", spec)
        self.assertIn("the LIVE body", spec)


if __name__ == "__main__":
    unittest.main()
