#!/usr/bin/env python3
"""GATE-SYMBOL-ANCHORS (#1143, #1139) — a citation that names a symbol must
still find it.

WHY THIS EXISTS. Cross-file citations in this tree are written `file.cpp:412`.
A line number is a coordinate into a moving file, so an edit ANYWHERE above the
cited line silently retargets every citation below it, in files the editing
change never opens. #1143 measured the blast radius on one file:
`src/vllm/entrypoints/model_loader.cpp` is cited by line from 109 sites across
45 files, and a single 45-line insertion near its top moved 203 of those
references at once. #1139 is the same defect one scale up, on an upstream
anchor: three `vllm/v1/worker/**` line anchors on one roadmap row all pointed at
unrelated code once the parity pin advanced, and two of them had already been
COPIED into a header and a spec, so the staleness was propagating.

The convention this gate enforces is the fix, not the symptom: cite the SYMBOL.
Write `` `src/vllm/entrypoints/model_loader.cpp::ResolveAutoDevice` `` instead of
`` `model_loader.cpp:100-115` ``. A symbol survives every edit that does not
rename it, `git grep` finds it, and a rename is exactly the moment a reader
WANTS the citation to break.

WHY IT IS NOT A TAUTOLOGY. #911 burned this repo with an anchor checker that
read its expectation out of the file it was checking — it reported 27/27 FRESH
while five anchors pointed at unrelated code, because "line 504 exists" is true
of any file with 504 lines. Here the two sides come from DIFFERENT places:

  * the EXPECTATION is the symbol name, written by the citing author, and it
    lives in the CITING file;
  * the EVIDENCE is the cited file's text.

No expectation is derived from the cited file, and nothing is stored in a shared
table. That also keeps the surface off the lock list AGENTS.md `## Records`
names: the expectation rides in the file that owns the claim, so N concurrent
pull requests never write one file to add N citations.

WHAT IT DOES NOT DO, so nobody cites it for more than it delivers:

  * It does not check that the symbol is DEFINED in the cited file, only that
    the file contains that identifier as a whole word. That is deliberate.
    "`ModelRegistry::Load`, which `model_loader.cpp` calls" is a legitimate and
    common citation, and a definition-only rule would reject it. A rename still
    reds, because a rename removes the token from the file.
  * It does not verify LINE citations. There is no honest way to: a bare line
    number carries no claim to check against. Converting one is a per-citation
    judgement about intent that no script can make, which is why #1143 refused
    to have a script rewrite all 109 sites.
  * By default it checks only IN-REPO citations. Upstream anchors need the
    pinned oracle checkout, which CI does not have. `--upstream-root` turns that
    half on for a box that does have one; see below.
  * It never reads the network.

UPSTREAM MODE (`--upstream-root <path-to-vllm-checkout>`). Resolves citations
whose path starts with `vllm/` against the PINNED revision, read with
`git show <pin>:<path>` so the checkout's working tree cannot substitute a
different revision for the pin. The pin comes from the ```parity-pin block in
`.agents/upstream-sync.md`; the checkout is asserted to CONTAIN that commit.
This mode is not a CI gate — it is the instrument that would have found #1139
before a reader did.

Run with `--self-test` to sweep the FIXTURES corpus below in both directions:
every `stale=True` fixture must be reported and every `stale=False` one must not.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SYNC = ROOT / ".agents/upstream-sync.md"

# A citation is a single backtick span holding a path with a source extension,
# `::`, and a (possibly qualified) identifier. The extension requirement is what
# keeps ordinary C++ prose out: `Qwen3_5MTPKind::kMoe` has no path in front of
# it and never matches.
#
# The leading class admits a DOT. It did not, and every path beginning with one
# therefore matched zero times and said nothing about it -- so a citation of
# a spec or a workflow by symbol -- a dot-leading path -- was silently unchecked.
# That was latent while nobody wrote one, and `.agents/porting.md` now invites
# exactly that.
CITATION_RE = re.compile(
    r"`([A-Za-z0-9_.][A-Za-z0-9_./+-]*\.(?:cpp|cc|h|hpp|cu|cuh|py|sh|md))"
    r"::([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_~][A-Za-z0-9_]*)*)`"
)

# Files whose citations are scanned. Everything else in the tree is binary,
# generated, or carries no prose.
SCAN_SUFFIXES = {
    ".md", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh", ".py", ".sh", ".yml", ".txt",
}

# Skipped by PREFIX, and counted so the skip is visible rather than silent.
#
# `.agents/completed/` is the frozen archive: AGENTS.md `## Records` says moved
# detail keeps its provenance, and rewriting an archived citation would forge
# what a past session wrote. `.agents/issue-index.md` is append-only by rule --
# a row there may never be edited -- so a stale anchor inside one is unrepairable
# by construction and gating it would be a gate nobody may satisfy.
FROZEN_PREFIXES = (".agents/completed/",)
FROZEN_FILES = (".agents/issue-index.md",)

# A path claims to be OURS when its DIRECTORY exists in this tree -- not when it
# starts with a familiar root. `tests/` was the first cut and it was wrong: vLLM
# has a `tests/` too, so `tests/kernels/attention/test_cache.py::...` is an
# upstream anchor that a root-prefix rule reports as a broken local path. Twenty
# citations went red that way on the first run, every one of them upstream and
# correct. Directory existence separates the two cases without a list to
# maintain: we have `tests/vllm/` and `tests/parity/`, we do not have
# `tests/kernels/`, and a typo inside a directory we DO have still reds.
#
# The directory must be at least TWO components deep. `tests/test_config.py` and
# `tests/test_flash_attn.py` are real upstream files sitting directly under a
# top-level name we share, and nothing about the path distinguishes them from a
# local file that lost its directory. Depth is the cheap discriminator; the cost
# is that a typo in a ONE-component directory falls through to the upstream
# bucket instead of reporting, which is a miss, never a false accusation.

# Bare-basename resolution is allowed only for our own source extensions. Every
# upstream oracle in the table is Python, so admitting a bare `.py` basename
# would let an upstream citation resolve onto an unrelated local script and go
# red for the wrong reason.
BASENAME_SUFFIXES = {".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh"}

# The recorded floor on the in-repo population. `checked == 0` alone is a mute
# switch: one added FROZEN_PREFIXES entry, or a narrowed CITATION_RE, takes the
# count from 91 to 1 and a zero-guard stays green the whole way down. This is a
# ratchet with headroom, not a per-PR record lock -- no change has to edit it,
# and it is raised only when the population has grown durably.
#
# It applies to THIS tree. A `--root` fixture holds one citation by design, so
# the floor there is `--min-checked` if given and 1 otherwise; the zero-guard
# still covers every root.
MIN_IN_REPO_CHECKED = 85


@dataclass(frozen=True)
class Citation:
    source: str
    line: int
    path: str
    symbol: str

    def render(self) -> str:
        return f"{self.source}:{self.line}: `{self.path}::{self.symbol}`"


@dataclass
class Counts:
    scanned_files: int = 0
    frozen_files: int = 0
    untracked_with_citations: int = 0
    citations: int = 0
    checked: int = 0
    fresh: int = 0
    stale: int = 0
    missing_local: int = 0
    ambiguous: int = 0
    upstream: int = 0
    upstream_checked: int = 0
    upstream_fresh: int = 0
    upstream_stale: int = 0
    upstream_missing: int = 0


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and ".git/" not in p.as_posix()
        )
    return sorted(result.stdout.split())


def untracked_files(root: Path) -> list[str]:
    """Files git can see but has not been told about."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return sorted(result.stdout.split())


def count_untracked_citations(root: Path) -> int:
    """How many UNTRACKED files carry a citation this run will not see.

    This change reported the blind spot about itself and then closed it only for
    its own fixtures: the first version of the test file wrote them as real
    local paths, was untracked, and ran green -- `git ls-files` does not list an
    untracked file, so neither its citations nor its existence as a cited PATH
    reach the walk. CI is sound because everything is tracked there. The local
    pre-commit run is not, and a green run before `git add` is not a green run.
    The skip is not closed here, because scanning the working tree would change
    what the gate's subject IS. It is COUNTED and PRINTED, the same discipline
    `frozen_files` already gets: a gate that cannot say how many things it left
    out has not reported.
    """

    total = 0
    for rel in untracked_files(root):
        if rel.startswith(FROZEN_PREFIXES) or rel in FROZEN_FILES:
            continue
        path = root / rel
        if path.suffix not in SCAN_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "::" in text and CITATION_RE.search(text):
            total += 1
    return total


def evidence(body: str) -> str:
    """The cited file's text with its OWN citation spans removed.

    A citation span is a claim about some file, never a definition or a call, so
    it may not stand as evidence for one. Leaving it in reopened #911 for a
    self-citation: when the citing and cited files are the SAME file, the
    expectation is read from the file under test and the citation text supplies
    its own evidence, so `` `a.cpp::GhostSymbol` `` written inside `a.cpp` read
    as fresh over a symbol that exists nowhere else. Stripping every citation
    span, not only the self-citing one, also closes the cross-file form, where
    `a.cpp` names `Ghost` only inside its citation of `b.cpp`.
    """

    return CITATION_RE.sub(" ", body)


def contains_symbol(body: str, symbol: str) -> bool:
    """Whole-word search for the FULL cited symbol, qualifiers included."""

    pattern = r"(?<![A-Za-z0-9_])" + re.escape(symbol) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, body) is not None


def collect(root: Path, counts: Counts) -> list[Citation]:
    found: list[Citation] = []
    for rel in tracked_files(root):
        if rel.startswith(FROZEN_PREFIXES) or rel in FROZEN_FILES:
            counts.frozen_files += 1
            continue
        path = root / rel
        if path.suffix not in SCAN_SUFFIXES or not path.is_file():
            continue
        counts.scanned_files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if "::" not in text:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for match in CITATION_RE.finditer(line):
                found.append(Citation(rel, number, match.group(1), match.group(2)))
    counts.citations = len(found)
    return found


def resolve_local(
    cite: Citation,
    tracked: set[str],
    by_base: dict[str, list[str]],
    root: Path,
) -> tuple[str, list[str]]:
    """Return (verdict, candidates) for one citation's PATH half."""

    if cite.path in tracked:
        return "resolved", [cite.path]
    base = cite.path.rsplit("/", 1)[-1]
    parent = cite.path.rsplit("/", 1)[0] if "/" in cite.path else ""
    if "/" in parent and (root / parent).is_dir():
        return "missing_local", []
    if Path(base).suffix not in BASENAME_SUFFIXES:
        return "upstream", []
    candidates = by_base.get(base, [])
    if len(candidates) == 1:
        return "resolved", candidates
    if len(candidates) > 1:
        return "ambiguous", candidates
    return "upstream", []


def check_tree(root: Path) -> tuple[Counts, list[str]]:
    counts = Counts()
    errors: list[str] = []
    tracked = set(tracked_files(root))
    # SORTED, and over a sorted walk. `tracked` is a set, so an unsorted walk
    # gives each basename's candidate list an arbitrary order -- which decides
    # which path a stale ambiguous citation NAMES, and made one mutation catch
    # depend on the iteration order of a hash set.
    by_base: dict[str, list[str]] = {}
    for rel in sorted(tracked):
        by_base.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)

    for cite in collect(root, counts):
        verdict, candidates = resolve_local(cite, tracked, by_base, root)
        if verdict == "missing_local":
            counts.missing_local += 1
            errors.append(
                f"{cite.render()}: names a path under this repository that does not exist"
            )
            continue
        if verdict == "upstream":
            counts.upstream += 1
            continue
        # An ambiguous basename used to be counted and dropped, so five live
        # citations were skipped for naming a file that exists twice. A
        # basename is checked against EVERY candidate instead: one of them
        # containing the symbol is what the citation claims, and reporting only
        # when NONE does keeps the ambiguity from becoming a false accusation.
        if verdict == "ambiguous":
            counts.ambiguous += 1
        counts.checked += 1
        hit = next(
            (
                rel
                for rel in candidates
                if contains_symbol(
                    evidence((root / rel).read_text(encoding="utf-8", errors="replace")),
                    cite.symbol,
                )
            ),
            None,
        )
        if hit is not None:
            counts.fresh += 1
        else:
            counts.stale += 1
            where = (
                candidates[0]
                if len(candidates) == 1
                else f"no file named `{cite.path}` ({len(candidates)} candidates)"
            )
            errors.append(
                f"{cite.render()}: {where} does not contain `{cite.symbol}`"
            )
    counts.untracked_with_citations = count_untracked_citations(root)
    return counts, errors


def parity_pin(sync_path: Path) -> str:
    text = sync_path.read_text(encoding="utf-8", errors="replace")
    block = re.search(r"```parity-pin\n(.*?)```", text, re.S)
    if block is None:
        raise ValueError(f"{sync_path}: no ```parity-pin block")
    pin = re.search(r"^vllm_commit\s*=\s*([0-9a-f]{7,40})\s*$", block.group(1), re.M)
    if pin is None:
        raise ValueError(f"{sync_path}: parity-pin block has no vllm_commit")
    return pin.group(1)


def check_upstream(root: Path, upstream_root: Path, counts: Counts) -> list[str]:
    """Verify `vllm/...::Symbol` citations against the PINNED revision."""

    errors: list[str] = []
    pin = parity_pin(root / ".agents/upstream-sync.md")
    identity = subprocess.run(
        ["git", "-C", str(upstream_root), "cat-file", "-t", pin],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if identity.returncode != 0 or identity.stdout.strip() != "commit":
        raise ValueError(
            f"{upstream_root}: does not contain the parity pin {pin}; "
            "an unpinned checkout is not the oracle"
        )
    print(f"upstream pin asserted present: {pin}")

    cache: dict[str, str | None] = {}
    seen: set[tuple[str, str]] = set()
    for cite in collect(root, Counts()):
        if not cite.path.startswith("vllm/"):
            continue
        counts.upstream_checked += 1
        key = (cite.path, cite.symbol)
        if cite.path not in cache:
            blob = subprocess.run(
                ["git", "-C", str(upstream_root), "show", f"{pin}:{cite.path}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            cache[cite.path] = blob.stdout if blob.returncode == 0 else None
        body = cache[cite.path]
        if body is None:
            counts.upstream_missing += 1
            if key not in seen:
                errors.append(f"{cite.render()}: absent from vLLM at {pin}")
        elif contains_symbol(evidence(body), cite.symbol):
            counts.upstream_fresh += 1
        else:
            counts.upstream_stale += 1
            if key not in seen:
                errors.append(
                    f"{cite.render()}: {cite.path} at {pin} does not contain `{cite.symbol}`"
                )
        seen.add(key)
    return errors


# (name, citing text, cited path, cited body, stale[, the citing file IS the
# cited file]). The last field defaults to False, which puts the citation in a
# separate `note.md`.
FIXTURES = (
    ("fresh", "`alpha/beta/a.cpp::Widget`", "alpha/beta/a.cpp", "struct Widget {};\n", False),
    (
        "self-citation is not its own evidence",
        "`alpha/beta/a.cpp::GhostSymbol`",
        "alpha/beta/a.cpp",
        "struct Widget {};\n",
        True,
        True,
    ),
    (
        "self-citation of a symbol that IS there",
        "`alpha/beta/a.cpp::Widget`",
        "alpha/beta/a.cpp",
        "struct Widget {};\n",
        False,
        True,
    ),
    (
        "a citation in the cited file is not evidence",
        "`alpha/beta/a.cpp::Ghost`",
        "alpha/beta/a.cpp",
        "// see `alpha/beta/b.cpp::Ghost`\n",
        True,
    ),
    ("renamed", "`alpha/beta/a.cpp::Widget`", "alpha/beta/a.cpp", "struct Gadget {};\n", True),
    ("call site counts", "`alpha/beta/a.cpp::Load`", "alpha/beta/a.cpp", "  Registry::Load(x);\n", False),
    ("qualified", "`alpha/beta/a.cpp::Registry::Load`", "alpha/beta/a.cpp", "  Registry::Load(x);\n", False),
    (
        "qualified miss",
        "`alpha/beta/a.cpp::Registry::Load`",
        "alpha/beta/a.cpp",
        "  Other::Load(x);\n",
        True,
    ),
    ("prefix is not a word", "`alpha/beta/a.cpp::Load`", "alpha/beta/a.cpp", "  LoadShards(x);\n", True),
    ("suffix is not a word", "`alpha/beta/a.cpp::Shards`", "alpha/beta/a.cpp", "  LoadShards(x);\n", True),
)


def self_test() -> int:
    import tempfile

    failures: list[str] = []
    for fixture in FIXTURES:
        name, citing, cited_path, cited_body, stale = fixture[:5]
        self_cite = fixture[5] if len(fixture) > 5 else False
        with tempfile.TemporaryDirectory() as raw:
            box = Path(raw)
            (box / "alpha/beta").mkdir(parents=True, exist_ok=True)
            if self_cite:
                (box / cited_path).write_text(
                    f"// see {citing} for the shape\n{cited_body}", encoding="utf-8"
                )
            else:
                (box / cited_path).write_text(cited_body, encoding="utf-8")
                (box / "note.md").write_text(
                    f"see {citing} for the shape\n", encoding="utf-8"
                )
            counts, errors = check_tree(box)
            if counts.checked != 1:
                failures.append(f"{name}: checked={counts.checked}, expected 1")
                continue
            reported = counts.stale == 1
            if reported != stale:
                failures.append(
                    f"{name}: stale={reported}, expected {stale} ({errors})"
                )
    print(f"self-test fixtures: {len(FIXTURES)}, failures: {len(failures)}")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--upstream-root",
        default=None,
        help="path to a vLLM checkout containing the parity pin; enables upstream mode",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--min-checked",
        type=int,
        default=None,
        help="floor on the in-repo checked count; defaults to the recorded "
             "MIN_IN_REPO_CHECKED for this tree and to 1 for a --root fixture",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    root = Path(args.root).resolve()
    counts, errors = check_tree(root)

    if args.min_checked is not None:
        floor = args.min_checked
    else:
        floor = MIN_IN_REPO_CHECKED if root == ROOT else 1

    buckets = counts.checked + counts.upstream + counts.missing_local
    print(
        f"symbol anchors: {counts.citations} citations in {counts.scanned_files} scanned files "
        f"({counts.frozen_files} frozen files skipped, "
        f"{counts.untracked_with_citations} untracked files carrying citations NOT scanned)"
    )
    print(
        f"  in-repo checked {counts.checked} (fresh {counts.fresh}, stale {counts.stale}); "
        f"upstream/unknown {counts.upstream}; "
        f"of which ambiguous basename {counts.ambiguous}; "
        f"missing local path {counts.missing_local}"
    )
    print(f"  buckets sum {buckets} vs {counts.citations} citations; floor {floor}")

    if counts.checked == 0:
        errors.append(
            "0 in-repo citations were checked. The tree carries symbol anchors, so a "
            "zero here means the citation grammar or the file walk stopped matching, "
            "not that everything is fresh."
        )
    elif counts.checked < floor:
        errors.append(
            f"in-repo checked {counts.checked} is below the recorded floor {floor}. "
            "A population that collapses is the grammar or the file walk failing, "
            "not the tree losing citations -- a floor of zero over a population of "
            "ninety is a mute switch."
        )

    if buckets != counts.citations:
        errors.append(
            f"buckets sum to {buckets} but {counts.citations} citations were found. "
            "Every citation must land in exactly one of checked / upstream / "
            "missing, or a citation has stopped being counted anywhere."
        )

    if args.upstream_root is not None:
        errors.extend(check_upstream(root, Path(args.upstream_root).resolve(), counts))
        print(
            f"  upstream checked {counts.upstream_checked} "
            f"(fresh {counts.upstream_fresh}, stale {counts.upstream_stale}, "
            f"file absent {counts.upstream_missing})"
        )

    if errors:
        print()
        for error in errors:
            print(f"FAIL {error}")
        print(f"\n{len(errors)} stale or unresolvable symbol anchor(s)")
        return 1
    print("OK every symbol anchor still finds what it names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
