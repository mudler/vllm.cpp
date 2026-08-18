#!/usr/bin/env python3
"""Validate commit messages with Git's own trailer parser."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
RAW_PROTOCOL_MARKER = "FOLLOWING_AGENTS_PROTOCOL"
CHECKER = "scripts/check-commit-trailers.py"
PROTOCOL_RULE = "trailers"
ATTRIBUTION_RULE = "attribution"
# The exact string the pull request template ships. Only the literal is
# rejected, and only under --filled, so a real value containing it cannot exist.
PLACEHOLDER_ASSISTED_BY = "AGENT:MODEL [TOOL]"
ASSISTED_BY = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*:[A-Za-z0-9][A-Za-z0-9_.+-]*"
    r"(?: \[[A-Za-z0-9][A-Za-z0-9_. +:/-]*\])+\Z"
)
# Closed, reviewable vocabulary for obvious agent/vendor/model/tool authorship.
# Boundaries prevent human names such as "Alice" from matching the token "ai".
AI_AUTHORSHIP_TOKENS = (
    "ai",
    "agent",
    "anthropic",
    "bot",
    "chatgpt",
    "claude",
    "claudecode",
    "codex",
    "copilot",
    "gemini",
    "gpt",
    "llm",
    "openai",
)
# GitHub writes this address for the ACCOUNT that opened a pull request when it
# composes a squash-merge message. It is attribution of a submitter, not a claim
# that a model authored the change -- that claim lives in AI-Assisted and
# Assisted-by, which are checked above and unaffected here (#418).
#
# The exemption is keyed on the FORGE'S OWN DOMAIN rather than on the name, so it
# cannot be borrowed: a hand-written `Co-authored-by: Claude <claude@anthropic.com>`
# still fails, and Signed-off-by is never exempted at all, because a sign-off is a
# legal assertion rather than attribution.
FORGE_ACCOUNT_EMAIL = re.compile(
    r"<[^>]*@users\.noreply\.github\.com>\s*$", re.IGNORECASE
)

AI_IDENTITY = re.compile(
    r"(?<![a-z0-9])(?:"
    + "|".join(re.escape(token) for token in AI_AUTHORSHIP_TOKENS)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Git command failed"
        raise ValueError(detail)
    return result.stdout.strip()


TRAILER_LINE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:[ \t].+$")
# The bare rule GitHub writes above the `Co-authored-by:` block it appends to a
# squash message. Matched as a whole paragraph only, never inside prose (#861).
FORGE_SEPARATOR = re.compile(r"-{3,}")
CONTINUATION_LINE = re.compile(r"^[ \t]+\S")


def _is_trailer_paragraph(paragraph: str) -> bool:
    """Whether every line of a paragraph is trailer-shaped."""
    lines = [line for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False
    if not TRAILER_LINE.match(lines[0]):
        return False
    return all(
        TRAILER_LINE.match(line) or CONTINUATION_LINE.match(line) for line in lines[1:]
    )


def join_trailing_trailer_paragraphs(message: str) -> str:
    """Fuse consecutive trailer-shaped paragraphs at the END into one block.

    `git interpret-trailers --parse` reads ONLY the final paragraph, so anything
    appended after the trailer block hides it completely. GitHub does exactly
    that on a squash merge: it adds `Co-authored-by:` as a new paragraph, and the
    protocol trailers above it stop being visible. Measured on main: dbd0d51c,
    87308dea and f64f2b71 all parse to nothing but that one line, and the gate
    reported them as missing trailers they plainly carry (#406).

    Only TRAILER-SHAPED paragraphs are fused. A prose paragraph still terminates
    the block, so trailers buried mid-message remain invalid -- the looseness
    this gate exists to prevent is untouched.
    """
    paragraphs = _paragraphs(message)
    if not paragraphs:
        return message
    fused: list[str] = []
    while paragraphs:
        if _is_trailer_paragraph(paragraphs[-1]):
            fused.insert(0, paragraphs.pop())
            continue
        # GitHub writes a bare rule before the `Co-authored-by:` block it appends
        # to a squash message. Measured on `617d6f452`, the FIRST squash landed
        # under `squash_merge_commit_message = PR_BODY`: the body appears once,
        # there is one trailer block, and the separator is still there. So it is
        # not a separator between concatenated commit messages, which is what
        # #829 and #850 assumed on the strength of a simulation that omitted it.
        # It belongs to the co-author block (#861).
        #
        # Stepped over ONLY when trailer-shaped paragraphs sit on both sides, so
        # a prose paragraph still terminates the block and trailers buried
        # mid-message stay invalid. That is the property this helper exists to
        # protect, and it is untouched.
        if (
            FORGE_SEPARATOR.fullmatch(paragraphs[-1].strip())
            and fused
            and len(paragraphs) >= 2
            and _is_trailer_paragraph(paragraphs[-2])
        ):
            paragraphs.pop()
            continue
        break
    if len(fused) < 2:
        return message
    return "\n\n".join(paragraphs + ["\n".join(fused)])


def parsed_trailers(message: str) -> str:
    """Return what ``git interpret-trailers --parse`` returns, after fusing any
    trailer-shaped paragraphs appended below the block (see the helper above)."""

    result = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=join_trailing_trailer_paragraphs(message),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def _paragraphs(message: str) -> list[str]:
    normalized = message.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return re.split(r"\n[ \t]*\n+", normalized) if normalized else []


def _trailer_map(message: str) -> dict[str, list[tuple[str, str]]]:
    trailers: dict[str, list[tuple[str, str]]] = {}
    for line in parsed_trailers(message).splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        trailers.setdefault(key.casefold(), []).append((key, value))
    return trailers


def _strict_errors(message: str) -> list[str]:
    errors: list[str] = []
    paragraphs = _paragraphs(message)
    marker_indexes = [
        index for index, paragraph in enumerate(paragraphs)
        if paragraph == RAW_PROTOCOL_MARKER
    ]
    if len(marker_indexes) != 1 or marker_indexes[0] == len(paragraphs) - 1:
        errors.append(
            f"[{PROTOCOL_RULE}] {RAW_PROTOCOL_MARKER} must appear exactly once "
            "as a separate paragraph before the trailer paragraph"
        )

    trailers = _trailer_map(message)
    protocol = trailers.get("following-agents-protocol", [])
    if len(protocol) != 1:
        errors.append(
            f"[{PROTOCOL_RULE}] Following-Agents-Protocol must appear exactly once"
        )
    elif protocol[0][1] != "true":
        errors.append(
            f"[{PROTOCOL_RULE}] Following-Agents-Protocol must be exactly true"
        )

    declarations = trailers.get("ai-assisted", [])
    assisted = trailers.get("assisted-by", [])
    if len(declarations) != 1:
        errors.append(f"[{ATTRIBUTION_RULE}] AI-Assisted must appear exactly once")
    else:
        declaration = declarations[0][1]
        if declaration not in {"true", "false"}:
            errors.append(f"[{ATTRIBUTION_RULE}] AI-Assisted must be true or false")
        elif declaration == "true" and not assisted:
            errors.append(
                f"[{ATTRIBUTION_RULE}] AI-Assisted true requires Assisted-by"
            )
        elif declaration == "false" and assisted:
            errors.append(
                f"[{ATTRIBUTION_RULE}] AI-Assisted false must omit Assisted-by"
            )

    for _, value in assisted:
        if ASSISTED_BY.fullmatch(value) is None:
            errors.append(f"[{ATTRIBUTION_RULE}] malformed Assisted-by value {value!r}")

    assisted_identities: set[str] = set()
    for _, value in assisted:
        if ASSISTED_BY.fullmatch(value) is None:
            continue
        identity, remainder = value.split(":", 1)
        model = remainder.split(" ", 1)[0]
        assisted_identities.update((identity.casefold(), model.casefold()))
        assisted_identities.update(
            tool.casefold() for tool in re.findall(r"\[([^]]+)\]", value)
        )
    for key in ("signed-off-by", "co-authored-by"):
        for original, value in trailers.get(key, []):
            if key == "co-authored-by" and FORGE_ACCOUNT_EMAIL.search(value):
                # Forge-generated attribution of the submitting account.
                continue
            folded_value = value.casefold()
            if AI_IDENTITY.search(value) or any(
                identity in folded_value for identity in assisted_identities
            ):
                errors.append(
                    f"[{ATTRIBUTION_RULE}] AI authorship trailer {original} is forbidden"
                )
    return errors


def validate_commit_message(message: str, *, strict: bool) -> list[str]:
    """Return trailer-contract errors for one complete commit message."""

    if not strict:
        marker_count = sum(
            paragraph == RAW_PROTOCOL_MARKER for paragraph in _paragraphs(message)
        )
        if marker_count == 1:
            return []
    return _strict_errors(message)


def _resolve_commit(repo: Path, revision: str) -> str:
    if not revision or "\x00" in revision or "\n" in revision:
        raise ValueError(f"invalid revision {revision!r}")
    if not revision.startswith("refs/"):
        candidates = (
            f"refs/heads/{revision}",
            f"refs/tags/{revision}",
            f"refs/remotes/{revision}",
        )
        matches = 0
        for candidate in candidates:
            result = subprocess.run(
                ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", candidate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                matches += 1
            elif result.returncode not in {1}:
                raise ValueError(f"could not resolve revision {revision!r}")
        if matches > 1:
            raise ValueError(f"ambiguous revision {revision!r}")
    resolved = _git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).splitlines()
    if len(resolved) != 1 or re.fullmatch(r"[0-9a-f]{40}", resolved[0]) is None:
        raise ValueError(f"revision {revision!r} did not resolve to one commit")
    return resolved[0]


def _merge_base(repo: Path, a: str, b: str) -> str:
    """The merge base of two revisions; raises when they share no history."""
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", a, b],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("range base and head have no merge base (unrelated histories)")
    oid = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", oid):
        raise ValueError("merge base did not resolve to one commit")
    return oid


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode not in {0, 1}:
        raise ValueError("could not establish commit ancestry")
    return result.returncode == 0




class LandedException(NamedTuple):
    """One landed message, one exact error, and why it is excused."""

    error: str
    reason: str


# EXCEPTIONS FOR MESSAGES THAT HAVE ALREADY LANDED. Visible debt, not success.
#
# A malformed message on `main` cannot be repaired: correcting it rewrites
# `main`, which nobody may do. It also does not clear itself. The main lane
# walks `LAST_GREEN..head` and `LAST_GREEN` advances only on a GREEN run, so a
# range containing an unrepairable red is re-walked by every later push, by
# every session, forever. `ci.yml:74` relies on exactly that property to make a
# cancelled run lossless -- the same property makes this red permanent.
#
# Keyed on the FULL commit oid AND the EXACT rendered error string. A commit oid
# covers its message, so an entry names one immutable byte string and cannot
# grow to cover a message somebody writes later. A DIFFERENT error in the same
# commit is a different string and is still reported; the SAME error in another
# commit is another oid and is never consulted; and a message that has not
# landed is validated by `validate_commit_message`, which does not consult this
# at all -- so the `--message-file` gate on a pull request BODY, which is what
# lands under `squash_merge_commit_message = PR_BODY`, can never be excused.
#
# `--cutover` was the obvious instrument and was measured before it was
# rejected (#1262). It does not excuse the sha you name, because
# `git merge-base --is-ancestor X X` succeeds and the cutover commit is checked
# STRICTLY; excusing `281b4bc76c0e` needs its CHILD named instead, which then
# drops all 2986 of that child's ancestors to the marker-only check, waiving
# defects nobody has read. And a cutover is a value that can be MOVED to make
# the next red disappear, which is the failure mode AGENTS.md
# section "Changing the rules or a checker" exists to prevent.
#
# Every run that applies an entry PRINTS it, on a passing run as well as a
# failing one. Growing this table is a reviewable edit that names a commit, an
# error and a reason; `tests/scripts/test_check_commit_trailers.py` pins the
# count, pins the key shape, and re-derives every entry against the real commit
# so a dead or guessed entry is red.
LANDED_MESSAGE_EXCEPTIONS: dict[str, LandedException] = {
    "281b4bc76c0e635adbc7ed38317035b07c99864d": LandedException(
        error=(
            "[attribution] malformed Assisted-by value 'AGENT:claude-opus-5 CLI'"
        ),
        reason=(
            "landed by the squash of #1257 (#1189 M4). All four branch commits "
            "carried the corrected value; the PULL REQUEST BODY did not, and "
            "`squash_merge_commit_message = PR_BODY` makes the body the landed "
            "message. The guard that reads the body (ci.yml:626-635, #848) was "
            "still `pending` at merge time because the runner pool was "
            "saturated -- it did not fail, it never ran. #1262"
        ),
    ),
}


def validate_range(
    repo: Path,
    base: str,
    head: str,
    *,
    cutover: str | None,
    excused: list[str] | None = None,
) -> list[str]:
    """Validate an exact first-parent-independent ``BASE..HEAD`` commit set.

    ``excused`` is an optional sink. Each applied ``LANDED_MESSAGE_EXCEPTIONS``
    entry appends one line to it, so a caller can report the debt a green run
    is carrying. Passing nothing keeps the verdict identical.
    """

    base_oid = _resolve_commit(repo, base)
    head_oid = _resolve_commit(repo, head)
    # From the MERGE BASE, not the base tip (#773). CI passes
    # `pull_request.base.sha`, which stops being an ancestor of head as soon as
    # main advances past the branch -- so this used to raise and return WITHOUT
    # READING A SINGLE COMMIT, meaning the trailer contract was never enforced
    # on any external contribution. Unrelated histories still fail closed: no
    # merge base means no range, and inventing one would be worse than refusing.
    base_oid = _merge_base(repo, base_oid, head_oid)
    cutover_oid = _resolve_commit(repo, cutover) if cutover is not None else None
    if cutover_oid is not None and not _is_ancestor(repo, cutover_oid, head_oid):
        raise ValueError("cutover must be reachable from range head")

    commits_text = _git(repo, "rev-list", "--reverse", f"{base_oid}..{head_oid}")
    failures: list[str] = []
    for commit in (line for line in commits_text.splitlines() if line):
        if cutover_oid is None:
            strict = True
        elif _is_ancestor(repo, cutover_oid, commit):
            strict = True
        elif _is_ancestor(repo, commit, cutover_oid):
            strict = False
        else:
            raise ValueError(f"commit {commit} is incomparable with cutover")

        message = _git(repo, "show", "-s", "--format=%B", commit) + "\n"
        exception = LANDED_MESSAGE_EXCEPTIONS.get(commit)
        for error in validate_commit_message(message, strict=strict):
            # Only the ONE error this commit is registered for. Everything else
            # it carries is reported exactly as it would be without the entry.
            if exception is not None and error == exception.error:
                if excused is not None:
                    excused.append(
                        f"{commit[:12]}: {error}\n      reason: {exception.reason}"
                    )
                continue
            failures.append(f"{commit[:12]}: {error}")
    return failures


def _range(value: str) -> tuple[str, str]:
    if value.count("..") != 1 or "..." in value:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    base, head = value.split("..", 1)
    if not base or not head:
        raise argparse.ArgumentTypeError("range must be exactly BASE..HEAD")
    return base, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", dest="revision_range", type=_range)
    parser.add_argument("--cutover")
    # The repository squashes with `squash_merge_commit_message = PR_BODY`, so a
    # pull request body BECOMES the landed commit message. Validating it with
    # the same `validate_commit_message` the range walk uses is the only way the
    # two cannot drift: there is one rule and one implementation of it, applied
    # to a message before it is committed and to the same message after (#848).
    parser.add_argument(
        "--message-file",
        help="validate one message read from PATH, or from stdin when PATH is -",
    )
    # The template ships the placeholder and must pass without this flag. A
    # FILLED body must not still be the form: `AGENT:MODEL [TOOL]` satisfies the
    # Assisted-by grammar while attributing the work to nobody, which is the
    # exact defect the attribution rule exists to prevent.
    parser.add_argument(
        "--filled",
        action="store_true",
        help="additionally reject the template's unreplaced Assisted-by placeholder",
    )
    args = parser.parse_args()

    if bool(args.revision_range) == bool(args.message_file):
        parser.error("pass exactly one of --range or --message-file")

    if args.message_file:
        try:
            message = (
                sys.stdin.read()
                if args.message_file == "-"
                else Path(args.message_file).read_text(encoding="utf-8")
            )
        except OSError as exc:
            print(f"commit trailer check FAILED: {exc}", file=sys.stderr)
            return 1
        if not message.strip():
            print(
                "commit trailer check FAILED: the message is empty. Under "
                "PR_BODY an empty body lands a commit with no trailers at all",
                file=sys.stderr,
            )
            return 1
        errors = validate_commit_message(message, strict=True)
        # Compare the PARSED trailer value, never the raw text. A substring
        # search over the whole message flags any body that merely MENTIONS the
        # placeholder, which this repository's own specs and pull request bodies
        # do whenever they document the flag. Found by running this check on the
        # body of the pull request that introduces it.
        if args.filled:
            placeholders = [
                value
                for _, value in _trailer_map(message).get("assisted-by", [])
                if value == PLACEHOLDER_ASSISTED_BY
            ]
            if placeholders:
                errors.append(
                    f"[{ATTRIBUTION_RULE}] Assisted-by still reads "
                    f"{PLACEHOLDER_ASSISTED_BY!r}, the template's placeholder. "
                    "Name the agent and model that did the work"
                )
        if errors:
            print("commit trailer check FAILED:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("OK: commit trailer contract")
        return 0

    excused: list[str] = []
    try:
        failures = validate_range(
            ROOT,
            *args.revision_range,
            cutover=args.cutover,
            excused=excused,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"commit trailer check FAILED: {exc}", file=sys.stderr)
        return 1
    # Printed whatever the verdict is. An exception is visible DEBT, and a
    # reader of a green lane has to be able to see what it is carrying.
    if excused:
        print(
            f"{len(excused)} landed-message exception(s) applied. This is "
            "DEBT, not success: the message is on `main` and cannot be "
            "repaired, because that would rewrite `main`."
        )
        for note in excused:
            print(f"    ~ {note}")
    if failures:
        print("commit trailer check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("OK: commit trailer contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
